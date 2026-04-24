import json
from unittest.mock import MagicMock, patch, call

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.summarize import summarize_papers, build_summarize_prompt


def _make_paper(arxiv_id: str, abstract: str) -> Paper:
    return Paper(arxiv_id=arxiv_id, title="T", authors=["A"], abstract=abstract, categories=["cs.CL"], url="url")


def test_build_summarize_prompt_contains_abstract():
    p = _make_paper("2504.1", "We study cross-lingual transfer in NER.")
    decisions = [{"arxiv_id": "2504.1", "categories": [1], "reasoning": "multilingual NLP"}]
    prompt = build_summarize_prompt(p, decisions)
    assert "cross-lingual transfer" in prompt
    assert "2504.1" in prompt


def test_summarize_papers_calls_llm_per_paper():
    papers = [_make_paper("2504.1", "Abs 1"), _make_paper("2504.2", "Abs 2")]
    decisions = [
        {"arxiv_id": "2504.1", "categories": [1], "reasoning": "r1"},
        {"arxiv_id": "2504.2", "categories": [2], "reasoning": "r2"},
    ]
    summary_json = json.dumps({
        "one_line_takeaway": "Key insight",
        "key_contribution": "They did X.",
        "method": "Using Y.",
        "most_important_result": "Z improved.",
    })
    mock_client = MagicMock()
    with patch("daily_arxiv_feed.summarize.chat", return_value=summary_json):
        results = summarize_papers(mock_client, papers, decisions)
        assert len(results) == 2
        assert results[0]["arxiv_id"] == "2504.1"
        assert results[0]["summary"]["one_line_takeaway"] == "Key insight"
