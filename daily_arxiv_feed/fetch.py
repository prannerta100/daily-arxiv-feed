import logging
import re
import time
import random
from dataclasses import dataclass, asdict
from html import unescape

import feedparser
import httpx
from defusedxml import ElementTree as ET

logger = logging.getLogger(__name__)

CATEGORIES = ["cs.CL", "cs.SD", "eess.AS", "cs.LG", "cs.AI"]
MAX_RETRIES = 15
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 60.0


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    url: str


def _retry_request(url: str, **kwargs) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF) + random.random()
            logger.warning("Request to %s failed (%s), retry %d/%d in %.1fs", url, e, attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)


def fetch_via_rss() -> list[Paper]:
    papers = []
    for cat in CATEGORIES:
        url = f"https://rss.arxiv.org/rss/{cat}"
        resp = _retry_request(url)
        feed = feedparser.parse(resp.text)
        for entry in feed.entries:
            arxiv_id = _extract_arxiv_id(entry.get("link", ""))
            if not arxiv_id:
                continue
            abstract = entry.get("summary", "")
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
            abstract = unescape(abstract)
            authors = [a.get("name", "") for a in entry.get("authors", [])]
            if not authors:
                authors = [entry.get("author", "Unknown")]
            cats = [t["term"] for t in entry.get("tags", []) if "term" in t]
            papers.append(Paper(
                arxiv_id=arxiv_id,
                title=unescape(entry.get("title", "").strip()),
                authors=authors,
                abstract=abstract,
                categories=cats if cats else [cat],
                url=f"https://arxiv.org/abs/{arxiv_id}",
            ))
    return papers


def fetch_via_api() -> list[Paper]:
    papers = []
    for cat in CATEGORIES:
        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"cat:{cat}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": "200",
        }
        resp = _retry_request(url, params=params)
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        for entry in root.findall("atom:entry", ns):
            id_text = entry.findtext("atom:id", "", ns)
            arxiv_id = _extract_arxiv_id(id_text)
            if not arxiv_id:
                continue
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            cats = [c.get("term", "") for c in entry.findall("atom:category", ns)]
            papers.append(Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=cats,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            ))
    return papers


def fetch_via_scrape() -> list[Paper]:
    papers = []
    for cat in CATEGORIES:
        url = f"https://arxiv.org/list/{cat}/new"
        resp = _retry_request(url)
        html = resp.text
        ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})', html)
        for arxiv_id in set(ids):
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            abs_resp = _retry_request(abs_url)
            abs_html = abs_resp.text
            title_match = re.search(r'<h1 class="title mathjax">\s*<span class="descriptor">Title:</span>\s*(.*?)\s*</h1>', abs_html, re.DOTALL)
            title = unescape(title_match.group(1).strip()) if title_match else "Unknown"
            abstract_match = re.search(r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>', abs_html, re.DOTALL)
            abstract = unescape(abstract_match.group(1).strip()) if abstract_match else ""
            author_match = re.findall(r'<a href="/search/\?searchtype=author[^"]*">([^<]+)</a>', abs_html)
            authors = [unescape(a.strip()) for a in author_match] if author_match else ["Unknown"]
            papers.append(Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=[cat],
                url=abs_url,
            ))
    return papers


def _extract_arxiv_id(url_or_id: str) -> str | None:
    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', url_or_id)
    return match.group(1) if match else None


def deduplicate(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for p in papers:
        if p.arxiv_id in seen:
            existing = seen[p.arxiv_id]
            merged_cats = list(set(existing.categories + p.categories))
            seen[p.arxiv_id] = Paper(
                arxiv_id=p.arxiv_id,
                title=existing.title,
                authors=existing.authors,
                abstract=existing.abstract,
                categories=merged_cats,
                url=existing.url,
            )
        else:
            seen[p.arxiv_id] = p
    return list(seen.values())


def fetch_papers() -> list[Paper]:
    sources = [
        ("RSS", fetch_via_rss),
        ("API", fetch_via_api),
        ("Scrape", fetch_via_scrape),
    ]
    for name, fetcher in sources:
        try:
            logger.info("Fetching papers via %s...", name)
            papers = fetcher()
            if papers:
                logger.info("Fetched %d papers via %s", len(papers), name)
                return deduplicate(papers)
            logger.warning("%s returned 0 papers, trying next source", name)
        except Exception:
            logger.exception("Failed to fetch via %s", name)
    logger.error("All fetch sources failed")
    return []
