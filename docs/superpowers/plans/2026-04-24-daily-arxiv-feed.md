# Daily Arxiv Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily service that fetches arxiv papers, filters/summarizes them via GPT-5.2, and compiles a ≤2-page NeurIPS-style PDF digest.

**Architecture:** Linear pipeline: fetch → filter → summarize → verify → compile. Each stage reads the previous stage's JSON output and writes its own. A shell script handles token generation and invokes the Python pipeline. launchd runs it daily at 7 AM.

**Tech Stack:** Python 3.12, Poetry, OpenAI SDK (pointed at Cisco LLM proxy), httpx, feedparser, defusedxml, Jinja2, tectonic (LaTeX→PDF).

**Spec:** `docs/superpowers/specs/2026-04-24-daily-arxiv-feed-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Poetry project config, dependencies |
| `daily_arxiv_feed/__init__.py` | Package marker |
| `daily_arxiv_feed/llm.py` | OpenAI client wrapper for LLM proxy |
| `daily_arxiv_feed/fetch.py` | Stage 1: fetch papers from arxiv (RSS → API → scrape) |
| `daily_arxiv_feed/filter.py` | Stage 2: LLM filters papers by research interest |
| `daily_arxiv_feed/summarize.py` | Stage 3: LLM summarizes selected papers |
| `daily_arxiv_feed/verify.py` | Stage 4: LLM verifies summaries against abstracts |
| `daily_arxiv_feed/compile.py` | Stage 5: render LaTeX via Jinja2, compile PDF via tectonic |
| `daily_arxiv_feed/main.py` | Pipeline orchestrator |
| `templates/neurips_digest.tex` | Jinja2 LaTeX template |
| `run.sh` | Entry point: token generation + pipeline invocation |
| `install.sh` | launchd plist installation |
| `com.pranavpg.daily-arxiv.plist` | launchd job definition |
| `tests/test_fetch.py` | Tests for fetch stage |
| `tests/test_filter.py` | Tests for filter stage |
| `tests/test_summarize.py` | Tests for summarize stage |
| `tests/test_verify.py` | Tests for verify stage |
| `tests/test_compile.py` | Tests for compile stage |
| `tests/test_main.py` | Tests for pipeline orchestrator |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `daily_arxiv_feed/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd ~/daily-arxiv-feed
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[tool.poetry]
name = "daily-arxiv-feed"
version = "0.1.0"
description = "Daily arxiv paper digest — filtered and summarized by LLM"
authors = ["Pranav Gupta"]
packages = [{include = "daily_arxiv_feed"}]

[tool.poetry.dependencies]
python = "^3.12"
openai = "^2.30"
httpx = "^0.27"
feedparser = "^6.0"
defusedxml = "^0.7"
jinja2 = "^3.1"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 3: Create package init**

```python
# daily_arxiv_feed/__init__.py
```

(Empty file.)

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
output/
logs/
dist/
*.egg-info/
```

- [ ] **Step 5: Install dependencies**

```bash
cd ~/daily-arxiv-feed
poetry install
```

Expected: dependencies resolve and install successfully.

- [ ] **Step 6: Create directory structure**

```bash
mkdir -p templates output/staging logs tests
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml poetry.lock daily_arxiv_feed/__init__.py .gitignore
git commit -m "chore: scaffold project with poetry dependencies"
```

---

### Task 2: LLM Client (`llm.py`)

**Files:**
- Create: `daily_arxiv_feed/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_llm.py`:

```python
import os
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.llm import get_client, chat


def test_get_client_reads_env():
    with patch.dict(os.environ, {"WEBEX_TOKEN": "test-token-123"}):
        client = get_client()
        assert client.api_key == "test-token-123"
        assert "llm-proxy" in client.base_url.host


def test_get_client_missing_token_raises():
    env = os.environ.copy()
    env.pop("WEBEX_TOKEN", None)
    with patch.dict(os.environ, env, clear=True):
        try:
            get_client()
            assert False, "Should have raised"
        except ValueError as e:
            assert "WEBEX_TOKEN" in str(e)


def test_chat_calls_openai_create():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"result": "ok"}'
    mock_client.chat.completions.create.return_value = mock_response

    result = chat(
        client=mock_client,
        system="You are helpful.",
        user="Hello",
        json_mode=True,
    )
    assert result == '{"result": "ok"}'
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-5.2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_llm.py -v
```

Expected: `ModuleNotFoundError: No module named 'daily_arxiv_feed.llm'`

- [ ] **Step 3: Implement llm.py**

Create `daily_arxiv_feed/llm.py`:

```python
import os
import logging
import time
import random

from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-5.2"
BASE_URL = "https://llm-proxy.us-east-2.int.infra.intelligence.webex.com/openai/v1"
RETRYABLE_STATUSES = {429, 500, 502, 503}
MAX_RETRIES = 5


def get_client() -> OpenAI:
    token = os.environ.get("WEBEX_TOKEN")
    if not token:
        raise ValueError("WEBEX_TOKEN environment variable is not set")
    return OpenAI(
        base_url=BASE_URL,
        api_key=token,
        default_headers={"x-cisco-app": "daily-arxiv-feed"},
    )


def chat(
    client: OpenAI,
    system: str,
    user: str,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    kwargs: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                wait = min(2 ** attempt + random.random(), 60)
                logger.warning("LLM call failed (status %s), retry %d/%d in %.1fs", status, attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                raise
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_llm.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/llm.py tests/test_llm.py
git commit -m "feat: add LLM client wrapper with retry logic"
```

---

### Task 3: Fetch Stage (`fetch.py`)

**Files:**
- Create: `daily_arxiv_feed/fetch.py`
- Create: `tests/test_fetch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fetch.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_fetch.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement fetch.py**

Create `daily_arxiv_feed/fetch.py`:

```python
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
MAX_RETRIES = 100
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
        url = f"http://rss.arxiv.org/rss/{cat}"
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
        url = "http://export.arxiv.org/api/query"
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
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_fetch.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/fetch.py tests/test_fetch.py
git commit -m "feat: add paper fetch stage with RSS/API/scrape fallback"
```

---

### Task 4: Filter Stage (`filter.py`)

**Files:**
- Create: `daily_arxiv_feed/filter.py`
- Create: `tests/test_filter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_filter.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_filter.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement filter.py**

Create `daily_arxiv_feed/filter.py`:

```python
import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat

logger = logging.getLogger(__name__)

FILTER_SYSTEM_PROMPT = """You are a research paper filter for a senior NLP researcher. You will receive a list of arxiv papers with their abstracts. Select ONLY papers that match one or more of these categories:

1. **Multilingual NLP** — papers explicitly addressing multilingual or cross-lingual natural language processing. The paper must have a multilingual or cross-lingual component as a core contribution, not just multilingual evaluation tacked on.

2. **Multilingual speech processing** — papers addressing speech or audio processing in multilingual or cross-lingual contexts. Must involve multiple languages in the speech/audio domain.

3. **Foundation model theory insights** — papers presenting genuine mechanistic insights into how large foundation models work, or a demonstrably clever trick with strong evidence. Be VERY selective — most papers do NOT qualify. Only select if the paper reveals something genuinely new about why or how these models behave. Expect at most 1 paper per day in this category.

4. **LLM evaluation** — papers with novel AUTOMATIC evaluation methods or surprising insights into LLM-as-judge approaches. NOT yet-another-benchmark papers. Must present genuinely new evaluation methodology or counter-intuitive findings about evaluation. Expect at most 1 paper per day in this category.

Be strict. Most papers should NOT be selected. When in doubt, reject.

Respond with JSON: {"papers": [{"arxiv_id": "...", "selected": true/false, "categories": [1,2,...], "reasoning": "brief explanation"}]}"""


def build_filter_prompt(papers: list[Paper]) -> str:
    lines = []
    for p in papers:
        lines.append(f"--- arxiv_id: {p.arxiv_id} ---")
        lines.append(f"Title: {p.title}")
        lines.append(f"Abstract: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


def parse_filter_response(response: str) -> list[dict]:
    data = json.loads(response)
    return data["papers"]


def filter_papers(client: OpenAI, papers: list[Paper]) -> tuple[list[Paper], list[dict]]:
    if not papers:
        return [], []

    batch_size = 20
    all_decisions = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_prompt = build_filter_prompt(batch)
        response = chat(client=client, system=FILTER_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        decisions = parse_filter_response(response)
        all_decisions.extend(decisions)

    selected_ids = {d["arxiv_id"] for d in all_decisions if d.get("selected")}
    paper_map = {p.arxiv_id: p for p in papers}
    selected = [paper_map[aid] for aid in selected_ids if aid in paper_map]

    logger.info("Filtered %d/%d papers", len(selected), len(papers))
    return selected, all_decisions
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_filter.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/filter.py tests/test_filter.py
git commit -m "feat: add LLM-based paper filter stage"
```

---

### Task 5: Summarize Stage (`summarize.py`)

**Files:**
- Create: `daily_arxiv_feed/summarize.py`
- Create: `tests/test_summarize.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_summarize.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_summarize.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement summarize.py**

Create `daily_arxiv_feed/summarize.py`:

```python
import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """You are summarizing an arxiv paper for a senior NLP researcher's daily digest. The digest has a strict 2-page limit, so be concise but accurate.

You will be given the paper's title, abstract, and which interest category it matched. More relevant/insightful papers deserve more detail; less relevant ones can be brief.

Explain difficult concepts in plain language — do not just restate jargon from the abstract. The reader should understand the key idea without having read the paper.

Respond with JSON:
{
  "one_line_takeaway": "The 'so what' — why this paper matters, in one sentence",
  "key_contribution": "What they did (2-3 sentences max)",
  "method": "How they did it (2-3 sentences max)",
  "most_important_result": "The headline finding (1-2 sentences)"
}"""


def build_summarize_prompt(paper: Paper, decisions: list[dict]) -> str:
    decision = next((d for d in decisions if d["arxiv_id"] == paper.arxiv_id), None)
    cats = decision["categories"] if decision else []
    reasoning = decision["reasoning"] if decision else ""
    cat_names = {1: "Multilingual NLP", 2: "Multilingual Speech", 3: "Foundation Model Theory", 4: "LLM Evaluation"}
    matched = ", ".join(cat_names.get(c, str(c)) for c in cats)

    return f"""arxiv_id: {paper.arxiv_id}
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Matched categories: {matched}
Selection reasoning: {reasoning}

Abstract:
{paper.abstract}"""


def summarize_papers(client: OpenAI, papers: list[Paper], decisions: list[dict]) -> list[dict]:
    results = []
    for paper in papers:
        user_prompt = build_summarize_prompt(paper, decisions)
        response = chat(client=client, system=SUMMARIZE_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        summary = json.loads(response)
        results.append({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "url": paper.url,
            "categories": next((d["categories"] for d in decisions if d["arxiv_id"] == paper.arxiv_id), []),
            "summary": summary,
        })
        logger.info("Summarized: %s", paper.arxiv_id)
    return results
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_summarize.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/summarize.py tests/test_summarize.py
git commit -m "feat: add LLM summarize stage with variable-length summaries"
```

---

### Task 6: Verify Stage (`verify.py`)

**Files:**
- Create: `daily_arxiv_feed/verify.py`
- Create: `tests/test_verify.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_verify.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_verify.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement verify.py**

Create `daily_arxiv_feed/verify.py`:

```python
import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat
from daily_arxiv_feed.summarize import SUMMARIZE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """You are verifying a summary of an arxiv paper against its original abstract.

Check for:
- Factual inconsistencies between the summary and abstract
- Claims in the summary not supported by the abstract
- Misrepresented results or conclusions
- Exaggerated or understated findings

Respond with JSON:
{"passed": true/false, "issues": ["list of specific issues found, empty if passed"]}"""


def _verify_one(client: OpenAI, abstract: str, summary: dict) -> dict:
    user_prompt = f"""Original abstract:
{abstract}

Summary to verify:
- Takeaway: {summary['one_line_takeaway']}
- Key contribution: {summary['key_contribution']}
- Method: {summary['method']}
- Most important result: {summary['most_important_result']}"""

    response = chat(client=client, system=VERIFY_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    return json.loads(response)


def _regenerate_summary(client: OpenAI, paper: Paper, feedback: list[str]) -> dict:
    regen_prompt = f"""Regenerate a summary for this paper. Your previous summary had these issues:
{chr(10).join('- ' + issue for issue in feedback)}

Fix these issues. Be accurate to the abstract.

Title: {paper.title}
Abstract: {paper.abstract}

Respond with JSON:
{{"one_line_takeaway": "...", "key_contribution": "...", "method": "...", "most_important_result": "..."}}"""

    response = chat(client=client, system=SUMMARIZE_SYSTEM_PROMPT, user=regen_prompt, json_mode=True)
    return json.loads(response)


def verify_summaries(client: OpenAI, papers: list[Paper], summaries: list[dict]) -> list[dict]:
    paper_map = {p.arxiv_id: p for p in papers}
    results = []

    for entry in summaries:
        paper = paper_map.get(entry["arxiv_id"])
        if not paper:
            results.append(entry)
            continue

        verification = _verify_one(client, paper.abstract, entry["summary"])

        if verification["passed"]:
            results.append(entry)
            logger.info("Verified OK: %s", entry["arxiv_id"])
            continue

        logger.warning("Verification failed for %s: %s", entry["arxiv_id"], verification["issues"])
        new_summary = _regenerate_summary(client, paper, verification["issues"])
        entry_copy = {**entry, "summary": new_summary}

        re_verification = _verify_one(client, paper.abstract, new_summary)
        if not re_verification["passed"]:
            entry_copy["verification_warning"] = "Summary may contain inaccuracies: " + "; ".join(re_verification["issues"])
            logger.warning("Re-verification still failed for %s, adding warning", entry["arxiv_id"])
        else:
            logger.info("Re-verification passed for %s", entry["arxiv_id"])

        results.append(entry_copy)

    return results
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_verify.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/verify.py tests/test_verify.py
git commit -m "feat: add verification stage with regeneration on failure"
```

---

### Task 7: LaTeX Template & Compile Stage (`compile.py`)

**Files:**
- Create: `templates/neurips_digest.tex`
- Create: `daily_arxiv_feed/compile.py`
- Create: `tests/test_compile.py`

- [ ] **Step 1: Create NeurIPS digest LaTeX template**

Create `templates/neurips_digest.tex`:

```latex
\documentclass[10pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage[margin=0.6in,top=0.5in,bottom=0.5in]{geometry}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}

\hypersetup{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black,citecolor=blue!60!black}

\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlist{nosep,leftmargin=*}

\titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
\titlespacing{\section}{0pt}{6pt}{2pt}

\pagestyle{empty}

\begin{document}

{\small\textbf{Arxiv Digest --- \VAR{date}} \hfill {\footnotesize\VAR{paper_count} papers selected}}

\medskip
\BLOCK{for section in sections}
\section*{\VAR{section.name}}
\BLOCK{for paper in section.papers}
\textbf{\VAR{paper.title|e}} {\footnotesize[\href{\VAR{paper.url}}{arxiv}]}\\
{\footnotesize \VAR{paper.authors|e}}\\[1pt]
\BLOCK{if paper.verification_warning}
{\footnotesize\textcolor{red}{$\triangle$ \VAR{paper.verification_warning|e}}}\\[1pt]
\BLOCK{endif}
\textit{\VAR{paper.summary.one_line_takeaway|e}}\\[2pt]
\VAR{paper.summary.key_contribution|e}
\VAR{paper.summary.method|e}
\VAR{paper.summary.most_important_result|e}

\smallskip
\BLOCK{endfor}
\BLOCK{endfor}
\BLOCK{if overview}
\section*{Field Overview}
\VAR{overview|e}
\BLOCK{endif}

\end{document}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_compile.py`:

```python
import os
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.compile import render_latex, compile_pdf, get_page_count


def test_render_latex_produces_valid_tex():
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "Test Paper",
        "authors": ["Author A", "Author B"],
        "url": "https://arxiv.org/abs/2504.1",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "Key insight here.",
            "key_contribution": "They did X.",
            "method": "Using Y approach.",
            "most_important_result": "Z improved by 10%.",
        },
    }]
    tex = render_latex(summaries, "2026-04-24", overview="Multilingual NLP is hot today.")
    assert "Test Paper" in tex
    assert "2026-04-24" in tex
    assert "Key insight here" in tex
    assert "arxiv" in tex
    assert "Field Overview" in tex


def test_render_latex_groups_by_category():
    summaries = [
        {
            "arxiv_id": "2504.1", "title": "P1", "authors": ["A"], "url": "u",
            "categories": [1],
            "summary": {"one_line_takeaway": "T1", "key_contribution": "C1", "method": "M1", "most_important_result": "R1"},
        },
        {
            "arxiv_id": "2504.2", "title": "P2", "authors": ["B"], "url": "u",
            "categories": [3],
            "summary": {"one_line_takeaway": "T2", "key_contribution": "C2", "method": "M2", "most_important_result": "R2"},
        },
    ]
    tex = render_latex(summaries, "2026-04-24")
    assert "Multilingual NLP" in tex
    assert "Foundation Model Theory" in tex
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_compile.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement compile.py**

Create `daily_arxiv_feed/compile.py`:

```python
import logging
import subprocess
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"

CATEGORY_NAMES = {
    1: "Multilingual NLP",
    2: "Multilingual Speech Processing",
    3: "Foundation Model Theory",
    4: "LLM Evaluation",
}


def render_latex(summaries: list[dict], date: str, overview: str = "") -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        block_start_string="\\BLOCK{",
        block_end_string="}",
        variable_start_string="\\VAR{",
        variable_end_string="}",
        comment_start_string="\\#{",
        comment_end_string="}",
    )
    template = env.get_template("neurips_digest.tex")

    grouped: dict[int, list[dict]] = {}
    for s in summaries:
        primary_cat = s["categories"][0] if s["categories"] else 0
        grouped.setdefault(primary_cat, []).append(s)

    sections = []
    for cat_id in sorted(grouped.keys()):
        sections.append({
            "name": CATEGORY_NAMES.get(cat_id, "Other"),
            "papers": grouped[cat_id],
        })

    return template.render(
        date=date,
        paper_count=len(summaries),
        sections=sections,
        overview=overview,
    )


def compile_pdf(tex_content: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{filename}.tex"
    pdf_path = output_dir / f"{filename}.pdf"

    tex_path.write_text(tex_content, encoding="utf-8")

    result = subprocess.run(
        ["tectonic", str(tex_path)],
        capture_output=True,
        text=True,
        cwd=str(output_dir),
    )
    if result.returncode != 0:
        logger.error("tectonic failed:\n%s", result.stderr)
        raise RuntimeError(f"tectonic compilation failed: {result.stderr}")

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found at {pdf_path} after compilation")

    return pdf_path


def get_page_count(pdf_path: Path) -> int:
    result = subprocess.run(
        ["strings", str(pdf_path)],
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"/Count\s+(\d+)", result.stdout)
    if matches:
        return max(int(m) for m in matches)
    count = result.stdout.count("/Page")
    return max(count - 1, 1)
```

- [ ] **Step 5: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_compile.py::test_render_latex_produces_valid_tex tests/test_compile.py::test_render_latex_groups_by_category -v
```

Expected: 2 tests pass (template rendering tests; PDF compilation tested in integration).

- [ ] **Step 6: Commit**

```bash
git add templates/neurips_digest.tex daily_arxiv_feed/compile.py tests/test_compile.py
git commit -m "feat: add LaTeX template and compile stage"
```

---

### Task 8: General Field Overview

**Files:**
- Create: `daily_arxiv_feed/overview.py`
- Create: `tests/test_overview.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_overview.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_overview.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement overview.py**

Create `daily_arxiv_feed/overview.py`:

```python
import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat

logger = logging.getLogger(__name__)

OVERVIEW_SYSTEM_PROMPT = """You are analyzing today's arxiv submissions to identify what's hot in the field. Given a list of ALL paper titles and abstracts from today, write a brief (2-4 sentences) overview of the dominant themes and trends you see. What topics are getting the most attention? Any surprising clusters or emerging directions?

Be specific — name the actual topics and rough counts (e.g., "At least 8 papers on..." or "A noticeable cluster around..."). Don't be generic.

Respond with JSON: {"overview": "your overview text here"}"""


def generate_overview(client: OpenAI, all_papers: list[Paper]) -> str:
    lines = []
    for p in all_papers:
        lines.append(f"- {p.title}: {p.abstract[:200]}")
    user_prompt = f"Today's {len(all_papers)} arxiv papers:\n\n" + "\n".join(lines)

    response = chat(client=client, system=OVERVIEW_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    data = json.loads(response)
    return data["overview"]
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_overview.py -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/overview.py tests/test_overview.py
git commit -m "feat: add field overview generation from all papers"
```

---

### Task 9: Pipeline Orchestrator (`main.py`)

**Files:**
- Create: `daily_arxiv_feed/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement main.py**

Create `daily_arxiv_feed/main.py`:

```python
import json
import logging
import sys
from dataclasses import asdict
from datetime import date as date_type, datetime
from pathlib import Path

from daily_arxiv_feed.llm import get_client, chat
from daily_arxiv_feed.fetch import fetch_papers
from daily_arxiv_feed.filter import filter_papers
from daily_arxiv_feed.summarize import summarize_papers
from daily_arxiv_feed.verify import verify_summaries
from daily_arxiv_feed.overview import generate_overview
from daily_arxiv_feed.compile import render_latex, compile_pdf, get_page_count

logger = logging.getLogger(__name__)

MAX_PAGES = 2
MAX_COMPRESS_ITERATIONS = 3

COMPRESS_SYSTEM_PROMPT = """You are compressing paper summaries to fit a strict 2-page PDF limit. The current digest is too long. Rewrite ALL summaries to be shorter while preserving the key insight and most important result for each paper. Use tighter prose — fewer words, same information density. Do not drop any paper.

Respond with JSON: {"summaries": [{"arxiv_id": "...", "one_line_takeaway": "...", "key_contribution": "...", "method": "...", "most_important_result": "..."}]}"""


def _write_staging(staging_dir: Path, filename: str, data) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / filename).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _compress_summaries(client, summaries: list[dict]) -> list[dict]:
    lines = []
    for s in summaries:
        lines.append(f"arxiv_id: {s['arxiv_id']}")
        lines.append(f"Title: {s['title']}")
        for k, v in s["summary"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    user_prompt = "\n".join(lines)

    response = chat(client=client, system=COMPRESS_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    compressed = json.loads(response)

    summary_map = {c["arxiv_id"]: c for c in compressed["summaries"]}
    for s in summaries:
        if s["arxiv_id"] in summary_map:
            c = summary_map[s["arxiv_id"]]
            s["summary"] = {
                "one_line_takeaway": c["one_line_takeaway"],
                "key_contribution": c["key_contribution"],
                "method": c["method"],
                "most_important_result": c["most_important_result"],
            }
    return summaries


def run_pipeline(output_dir: Path | None = None, date: str | None = None) -> Path | None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output"
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    staging_dir = output_dir / "staging" / date

    client = get_client()

    logger.info("=== Stage 1: Fetch ===")
    papers = fetch_papers()
    if not papers:
        logger.warning("No papers fetched. Exiting.")
        return None
    _write_staging(staging_dir, "01_fetched.json", [asdict(p) for p in papers])
    logger.info("Fetched %d papers", len(papers))

    logger.info("=== Stage 2: Filter ===")
    selected, all_decisions = filter_papers(client, papers)
    _write_staging(staging_dir, "02_filtered.json", all_decisions)
    if not selected:
        logger.warning("No papers selected. Exiting.")
        return None
    logger.info("Selected %d papers", len(selected))

    logger.info("=== Stage 2.5: Field Overview ===")
    overview = generate_overview(client, papers)

    logger.info("=== Stage 3: Summarize ===")
    summaries = summarize_papers(client, selected, all_decisions)
    _write_staging(staging_dir, "03_summaries.json", summaries)

    logger.info("=== Stage 4: Verify ===")
    verified = verify_summaries(client, selected, summaries)
    _write_staging(staging_dir, "04_verified.json", verified)

    logger.info("=== Stage 5: Compile ===")
    tex = render_latex(verified, date, overview=overview)
    pdf_path = compile_pdf(tex, output_dir, date)
    pages = get_page_count(pdf_path)
    logger.info("Initial PDF: %d pages", pages)

    for i in range(MAX_COMPRESS_ITERATIONS):
        if pages <= MAX_PAGES:
            break
        logger.info("PDF is %d pages (max %d), compressing (iteration %d)...", pages, MAX_PAGES, i + 1)
        verified = _compress_summaries(client, verified)
        _write_staging(staging_dir, "04_verified.json", verified)
        tex = render_latex(verified, date, overview=overview)
        pdf_path = compile_pdf(tex, output_dir, date)
        pages = get_page_count(pdf_path)
        logger.info("After compression: %d pages", pages)

    if pages > MAX_PAGES:
        logger.warning("PDF still %d pages after %d compression attempts", pages, MAX_COMPRESS_ITERATIONS)

    logger.info("Done! PDF at %s", pdf_path)
    return pdf_path


def main():
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        result = run_pipeline()
        if result:
            logging.info("Pipeline complete: %s", result)
        else:
            logging.info("Pipeline complete: no PDF generated today")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
cd ~/daily-arxiv-feed
poetry run pytest tests/test_main.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add daily_arxiv_feed/main.py tests/test_main.py
git commit -m "feat: add pipeline orchestrator with compression loop"
```

---

### Task 10: Shell Scripts & launchd

**Files:**
- Create: `run.sh`
- Create: `install.sh`
- Create: `com.pranavpg.daily-arxiv.plist`

- [ ] **Step 1: Create run.sh**

Create `run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export WEBEX_TOKEN
WEBEX_TOKEN=$(token-generator personal bts)

poetry run python -m daily_arxiv_feed.main
```

```bash
chmod +x run.sh
```

- [ ] **Step 2: Create launchd plist**

Create `com.pranavpg.daily-arxiv.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pranavpg.daily-arxiv</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/pranavpg/daily-arxiv-feed/run.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/pranavpg/daily-arxiv-feed</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/pranavpg/daily-arxiv-feed/logs/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/pranavpg/daily-arxiv-feed/logs/launchd-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/Users/pranavpg/.local/bin</string>
    </dict>
</dict>
</plist>
```

- [ ] **Step 3: Create install.sh**

Create `install.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p output/staging logs

PLIST_SRC="com.pranavpg.daily-arxiv.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.pranavpg.daily-arxiv.plist"

if launchctl list | grep -q "com.pranavpg.daily-arxiv"; then
    echo "Unloading existing job..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

cp "$PLIST_SRC" "$PLIST_DST"
launchctl load "$PLIST_DST"

echo "Installed and loaded. Job will run daily at 7:00 AM."
echo "To run manually: ./run.sh"
echo "To check status: launchctl list | grep daily-arxiv"
```

```bash
chmod +x install.sh
```

- [ ] **Step 4: Verify plist syntax**

```bash
cd ~/daily-arxiv-feed
plutil -lint com.pranavpg.daily-arxiv.plist
```

Expected: `com.pranavpg.daily-arxiv.plist: OK`

- [ ] **Step 5: Commit**

```bash
git add run.sh install.sh com.pranavpg.daily-arxiv.plist
git commit -m "feat: add run.sh, install.sh, and launchd plist"
```

---

### Task 11: Integration Test — End to End

**Files:**
- No new files; uses the existing pipeline.

- [ ] **Step 1: Run the full pipeline manually**

```bash
cd ~/daily-arxiv-feed
export WEBEX_TOKEN=$(token-generator personal bts)
poetry run python -m daily_arxiv_feed.main
```

Expected: Pipeline runs through all 5 stages, produces a PDF in `output/`.

- [ ] **Step 2: Verify the output**

```bash
ls -la output/*.pdf
ls -la output/staging/$(date +%Y-%m-%d)/
```

Expected: A dated PDF exists, and all 4 staging JSON files are present.

- [ ] **Step 3: Open the PDF and verify**

```bash
open output/$(date +%Y-%m-%d).pdf
```

Expected: A ≤2-page NeurIPS-style PDF with:
- Date header
- Papers grouped by category
- Clickable arxiv links
- Field overview section
- Compact layout, no wasted space

- [ ] **Step 4: Fix any issues found during manual testing**

If any issues arise (bad formatting, missing sections, LaTeX errors), fix them in the relevant files and re-run.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test adjustments"
```

---

### Task 12: Install launchd Job

- [ ] **Step 1: Run install.sh**

```bash
cd ~/daily-arxiv-feed
./install.sh
```

Expected: "Installed and loaded" message.

- [ ] **Step 2: Verify the job is loaded**

```bash
launchctl list | grep daily-arxiv
```

Expected: A line showing `com.pranavpg.daily-arxiv` with PID `-` (not yet running).

- [ ] **Step 3: Commit**

```bash
cd ~/daily-arxiv-feed
git add -A
git commit -m "chore: installation verified"
```
