import json
from unittest.mock import patch, MagicMock, AsyncMock

from daily_arxiv_feed.fetch import (
    fetch_papers,
    fetch_via_rss,
    fetch_via_api,
    fetch_via_scrape,
    deduplicate,
    Paper,
    CATEGORIES,
)


def test_paper_dataclass():
    p = Paper(
        arxiv_id="2504.12345",
        title="Test Paper",
        authors=["Author A"],
        abstract="An abstract.",
        categories=["cs.CL"],
        url="https://arxiv.org/abs/2504.12345",
    )
    assert p.arxiv_id == "2504.12345"


def test_deduplicate_removes_dupes():
    p1 = Paper("2504.1", "T1", ["A"], "abs1", ["cs.CL"], "url1")
    p2 = Paper("2504.1", "T1", ["A"], "abs1", ["cs.LG"], "url1")
    p3 = Paper("2504.2", "T2", ["B"], "abs2", ["cs.AI"], "url2")
    result = deduplicate([p1, p2, p3])
    assert len(result) == 2
    ids = {p.arxiv_id for p in result}
    assert ids == {"2504.1", "2504.2"}


def test_fetch_papers_falls_back_on_failure():
    with patch("daily_arxiv_feed.fetch.fetch_via_rss", side_effect=Exception("RSS down")):
        with patch("daily_arxiv_feed.fetch.fetch_via_api") as mock_api:
            mock_api.return_value = [
                Paper("2504.9", "Fallback", ["C"], "abs", ["cs.CL"], "url")
            ]
            result = fetch_papers()
            assert len(result) == 1
            assert result[0].title == "Fallback"


def test_categories_list():
    assert "cs.CL" in CATEGORIES
    assert "eess.AS" in CATEGORIES
    assert len(CATEGORIES) == 5
