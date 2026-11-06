import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.main import run_pipeline, rebuild_digest
from daily_arxiv_feed.fetch import Paper


def test_run_pipeline_no_papers(tmp_path):
    with patch("daily_arxiv_feed.main.get_client") as mock_gc:
        with patch("daily_arxiv_feed.main.fetch_papers", return_value=[]):
            result = run_pipeline(output_dir=tmp_path, date="2026-04-24")
            assert result is None


def test_rebuild_preserves_selection_bounds_and_originals_without_pushing(tmp_path):
    stage=tmp_path/'staging'/'2026-11-05'
    stage.mkdir(parents=True)
    p={"arxiv_id":"id","title":"T","authors":["A"],"abstract":"The study compares languages.","categories":["cs.CL"],"url":"url"}
    previous=[{"arxiv_id":"id","title":"T","authors":["A"],"categories":[1],"url":"url",
               "summary":{"one_line_takeaway":"The study compares languages.","key_contribution":"It uses a benchmark.","method":"It evaluates translations.","most_important_result":"The method improves accuracy."}}]
    current=[{**previous[0],"summary":{"excerpt":"The study compares languages."},
              "summary_limits":{"max_words":16,"max_chars":144},"summary_source":"compressed_abstract"}]
    (stage/'01_fetched.json').write_text(json.dumps([p]))
    (stage/'04_verified.json').write_text(json.dumps(previous))
    (stage/'05_overview.json').write_text(json.dumps({"overview":"Existing overview."}))
    with patch('daily_arxiv_feed.main.get_client'), \
         patch('daily_arxiv_feed.main.summarize_papers',return_value=current) as summarize, \
         patch('daily_arxiv_feed.main.verify_summaries',return_value=current), \
         patch('daily_arxiv_feed.main.add_takeaways',side_effect=lambda client,papers,entries: entries), \
         patch('daily_arxiv_feed.main._compile_digest',return_value=(tmp_path/'digest.pdf',current)), \
         patch('daily_arxiv_feed.main._push_pdf') as push, \
         patch('daily_arxiv_feed.main.fetch_papers') as fetch:
        rebuild_digest('2026-11-05',output_dir=tmp_path)
    push.assert_not_called()
    fetch.assert_not_called()
    assert summarize.call_args.kwargs['previous']==previous
    assert json.loads((stage/'readability-before'/'04_verified.json').read_text())==previous
    comparison=json.loads((stage/'06_readability_comparison.json').read_text())
    assert comparison[0]['arxiv_id']=='id'
    assert comparison[0]['after_words']==4


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

    with patch("daily_arxiv_feed.main.get_client"), \
         patch("daily_arxiv_feed.main.add_takeaways",side_effect=lambda client,papers,entries: entries):
        with patch("daily_arxiv_feed.main.fetch_papers", return_value=papers):
            with patch("daily_arxiv_feed.main.filter_papers", return_value=(selected, decisions)):
                with patch("daily_arxiv_feed.main.summarize_papers", return_value=summaries):
                    with patch("daily_arxiv_feed.main.verify_summaries", return_value=verified):
                        with patch("daily_arxiv_feed.main.generate_overview", return_value="overview"):
                            with patch("daily_arxiv_feed.main.render_latex", return_value="\\documentclass{}"):
                                with patch("daily_arxiv_feed.main.compile_pdf") as mock_compile:
                                    with patch("daily_arxiv_feed.main.get_page_count", return_value=1):
                                        mock_compile.return_value = tmp_path / "2026-04-24.pdf"
                                        with patch("daily_arxiv_feed.main._push_pdf"):
                                            result = run_pipeline(output_dir=tmp_path, date="2026-04-24")

    staging = tmp_path / "staging" / "2026-04-24"
    assert staging.exists()
    assert (staging / "01_fetched.json").exists()
    assert (staging / "02_filtered.json").exists()
    assert (staging / "03_summaries.json").exists()
    assert (staging / "04_verified.json").exists()
