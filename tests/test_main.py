import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.main import run_pipeline
from daily_arxiv_feed.fetch import Paper


def test_run_pipeline_no_papers(tmp_path):
    with patch("daily_arxiv_feed.main.get_client") as mock_gc:
        with patch("daily_arxiv_feed.main.fetch_papers", return_value=[]):
            result = run_pipeline(output_dir=tmp_path, date="2026-04-24")
            assert result is None


def test_run_pipeline_no_selected(tmp_path):
    papers = [Paper("2504.1", "T", ["A"], "abs", ["cs.CL"], "url")]
    with patch("daily_arxiv_feed.main.get_client") as mock_gc:
        with patch("daily_arxiv_feed.main.fetch_papers", return_value=papers):
            with patch("daily_arxiv_feed.main.filter_papers", return_value=([], [])):
                with patch("daily_arxiv_feed.main.generate_overview", return_value="overview"):
                    result = run_pipeline(output_dir=tmp_path, date="2026-04-24")
                    assert result is None


def test_run_pipeline_writes_staging(tmp_path):
    papers = [Paper("2504.1", "T", ["A"], "abs", ["cs.CL"], "url")]
    selected = [papers[0]]
    decisions = [{"arxiv_id": "2504.1", "selected": True, "categories": [1], "reasoning": "r"}]
    summaries = [{"arxiv_id": "2504.1", "title": "T", "authors": ["A"], "url": "url", "categories": [1], "summary": {"one_line_takeaway": "t", "key_contribution": "c", "method": "m", "most_important_result": "r"}}]
    verified = summaries

    with patch("daily_arxiv_feed.main.get_client"):
        with patch("daily_arxiv_feed.main.fetch_papers", return_value=papers):
            with patch("daily_arxiv_feed.main.filter_papers", return_value=(selected, decisions)):
                with patch("daily_arxiv_feed.main.summarize_papers", return_value=summaries):
                    with patch("daily_arxiv_feed.main.verify_summaries", return_value=verified):
                        with patch("daily_arxiv_feed.main.generate_overview", return_value="overview"):
                            with patch("daily_arxiv_feed.main.render_latex", return_value="\\documentclass{}"):
                                with patch("daily_arxiv_feed.main.compile_pdf") as mock_compile:
                                    with patch("daily_arxiv_feed.main.get_page_count", return_value=1):
                                        mock_compile.return_value = tmp_path / "2026-04-24.pdf"
                                        result = run_pipeline(output_dir=tmp_path, date="2026-04-24")

    staging = tmp_path / "staging" / "2026-04-24"
    assert staging.exists()
    assert (staging / "01_fetched.json").exists()
    assert (staging / "02_filtered.json").exists()
    assert (staging / "03_summaries.json").exists()
    assert (staging / "04_verified.json").exists()
