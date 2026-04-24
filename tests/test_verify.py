import json
from unittest.mock import MagicMock, patch

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.verify import verify_summaries


def _make_paper(arxiv_id: str, abstract: str) -> Paper:
    return Paper(arxiv_id=arxiv_id, title="T", authors=["A"], abstract=abstract, categories=["cs.CL"], url="url")


def test_verify_passes_good_summary():
    papers = [_make_paper("2504.1", "We improve NER with cross-lingual transfer.")]
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "T",
        "authors": ["A"],
        "url": "url",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "Cross-lingual transfer improves NER.",
            "key_contribution": "They show transfer helps.",
            "method": "Fine-tuning multilingual models.",
            "most_important_result": "5 point F1 gain.",
        },
    }]
    verify_response = json.dumps({"passed": True, "issues": []})
    mock_client = MagicMock()
    with patch("daily_arxiv_feed.verify.chat", return_value=verify_response):
        result = verify_summaries(mock_client, papers, summaries)
        assert len(result) == 1
        assert result[0].get("verification_warning") is None


def test_verify_flags_and_regenerates_bad_summary():
    papers = [_make_paper("2504.1", "We study English-only sentiment analysis.")]
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "T",
        "authors": ["A"],
        "url": "url",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "Multilingual sentiment breakthrough.",
            "key_contribution": "They did multilingual work.",
            "method": "Cross-lingual transfer.",
            "most_important_result": "Works in 100 languages.",
        },
    }]
    fail_response = json.dumps({"passed": False, "issues": ["Paper is English-only, summary claims multilingual"]})
    regen_summary = json.dumps({
        "one_line_takeaway": "English sentiment analysis improved.",
        "key_contribution": "They study English sentiment.",
        "method": "Fine-tuning on English data.",
        "most_important_result": "State-of-the-art on SST.",
    })
    pass_response = json.dumps({"passed": True, "issues": []})
    mock_client = MagicMock()
    with patch("daily_arxiv_feed.verify.chat", side_effect=[fail_response, regen_summary, pass_response]):
        result = verify_summaries(mock_client, papers, summaries)
        assert result[0]["summary"]["one_line_takeaway"] == "English sentiment analysis improved."
