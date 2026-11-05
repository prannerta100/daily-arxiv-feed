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


def test_overview_retries_oversized_output_and_uses_real_abstract_snippet():
    paper=Paper("id","Title",["A"],"arXiv:id Announce Type: new\nAbstract: The paper studies multilingual speech.",["cs.CL"],"url")
    with patch("daily_arxiv_feed.overview.chat",side_effect=[
        json.dumps({"overview":"long topic "*100}),
        json.dumps({"overview":"Researchers study multilingual speech."})]) as call:
        result=generate_overview(MagicMock(),[paper])
    assert result=="Researchers study multilingual speech."
    assert call.call_count==2
    assert "Announce Type" not in call.call_args.kwargs['user']
