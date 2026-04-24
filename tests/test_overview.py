import json
from unittest.mock import MagicMock, patch

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.overview import generate_overview


def test_generate_overview_returns_string():
    papers = [
        Paper("2504.1", "T1", ["A"], "About multilingual NER", ["cs.CL"], "u1"),
        Paper("2504.2", "T2", ["B"], "About speech synthesis", ["cs.SD"], "u2"),
        Paper("2504.3", "T3", ["C"], "About model scaling", ["cs.LG"], "u3"),
    ]
    mock_client = MagicMock()
    overview_text = "Today's papers focus heavily on scaling and multilingual methods."
    with patch("daily_arxiv_feed.overview.chat", return_value=json.dumps({"overview": overview_text})):
        result = generate_overview(mock_client, papers)
        assert "scaling" in result
        assert "multilingual" in result
