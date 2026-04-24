import json
from unittest.mock import MagicMock, patch

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.filter import filter_papers, build_filter_prompt, parse_filter_response


def _make_paper(arxiv_id: str, title: str, abstract: str) -> Paper:
    return Paper(arxiv_id=arxiv_id, title=title, authors=["A"], abstract=abstract, categories=["cs.CL"], url=f"https://arxiv.org/abs/{arxiv_id}")


def test_build_filter_prompt_includes_abstracts():
    papers = [_make_paper("2504.1", "Paper One", "Abstract about multilingual NLP")]
    prompt = build_filter_prompt(papers)
    assert "2504.1" in prompt
    assert "multilingual NLP" in prompt


def test_parse_filter_response_valid():
    response = json.dumps({"papers": [
        {"arxiv_id": "2504.1", "selected": True, "categories": [1], "reasoning": "multilingual"},
        {"arxiv_id": "2504.2", "selected": False, "categories": [], "reasoning": "not relevant"},
    ]})
    result = parse_filter_response(response)
    assert len(result) == 2
    assert result[0]["selected"] is True
    assert result[1]["selected"] is False


def test_filter_papers_returns_selected_only():
    papers = [
        _make_paper("2504.1", "Multilingual NER", "Cross-lingual named entity recognition..."),
        _make_paper("2504.2", "Image Segmentation", "We segment images using CNNs..."),
    ]
    llm_response = json.dumps({"papers": [
        {"arxiv_id": "2504.1", "selected": True, "categories": [1], "reasoning": "multilingual NLP"},
        {"arxiv_id": "2504.2", "selected": False, "categories": [], "reasoning": "computer vision, not relevant"},
    ]})
    mock_client = MagicMock()
    with patch("daily_arxiv_feed.filter.chat", return_value=llm_response):
        selected, all_decisions = filter_papers(mock_client, papers)
        assert len(selected) == 1
        assert selected[0].arxiv_id == "2504.1"
        assert len(all_decisions) == 2
