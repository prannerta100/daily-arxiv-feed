# Daily Arxiv Feed — Design Spec

**Date:** 2026-04-24
**Author:** Pranav Gupta
**Status:** Draft

## Overview

A personal daily service that fetches new arxiv papers, uses GPT-5.2 (via Cisco LLM proxy) to filter and summarize papers matching specific research interests, and compiles a concise ≤2-page PDF in NeurIPS style. Runs every morning at 7 AM via macOS launchd.

## Research Interest Filters

The LLM filter stage selects papers matching these four categories, applied to paper abstracts:

1. **Multilingual NLP** — papers explicitly addressing multilingual or cross-lingual natural language processing
2. **Multilingual speech processing** — papers addressing speech/audio in multilingual or cross-lingual contexts
3. **Foundation model theory insights** — papers presenting genuine mechanistic insights into how large foundation models work, or a demonstrably clever trick with evidence. Very selective — expect ≤1 paper/day.
4. **LLM evaluation** — papers with novel automatic evaluation methods or surprising insights into LLM judges. Not yet-another-benchmark papers. Must present genuinely new methodology or counter-intuitive findings. — expect ≤1 paper/day.
5. **General overview of the field** - what seems to be hot today based on all the papers obtained.

## Arxiv Categories Monitored

- `cs.CL` — Computation and Language
- `cs.SD` — Sound
- `eess.AS` — Audio and Speech Processing
- `cs.LG` — Machine Learning
- `cs.AI` — Artificial Intelligence

## Project Structure

```
~/daily-arxiv-feed/
├── pyproject.toml                        # Poetry project
├── run.sh                                # Entry point: generates token, runs pipeline
├── install.sh                            # Installs launchd plist, creates directories
├── com.pranavpg.daily-arxiv.plist        # launchd job definition
├── templates/
│   └── neurips_digest.tex                # NeurIPS-based LaTeX template (compact digest)
├── output/
│   ├── 2026-04-24.pdf                    # Daily PDFs named by date
│   └── staging/
│       └── 2026-04-24/                   # Intermediate JSON per stage per day
│           ├── 01_fetched.json
│           ├── 02_filtered.json
│           ├── 03_summaries.json
│           └── 04_verified.json
├── logs/
│   └── 2026-04-24.log                    # Per-run log file
└── daily_arxiv_feed/
    ├── __init__.py
    ├── main.py                           # Pipeline orchestrator
    ├── fetch.py                          # Stage 1: fetch papers from arxiv
    ├── filter.py                         # Stage 2: LLM filters by abstract relevance
    ├── summarize.py                      # Stage 3: LLM summarizes selected papers
    ├── verify.py                         # Stage 4: verify summaries against abstracts
    ├── compile.py                        # Stage 5: render LaTeX and compile PDF
    └── llm.py                            # OpenAI client wrapper for LLM proxy
```

## Dependencies

Managed via Poetry (`pyproject.toml`):

- `openai` — LLM proxy client (OpenAI-compatible API)
- `httpx` — HTTP client for arxiv API/scraping
- `feedparser` — RSS feed parsing
- `defusedxml` — safe XML parsing for arxiv API responses
- `jinja2` — LaTeX template rendering

System dependencies (already installed):

- `tectonic` — LaTeX-to-PDF compilation
- `token-generator` — OAuth token generation for LLM proxy

## Pipeline Stages

### Stage 1: Fetch (`fetch.py`)

Fetches new papers from arxiv using three sources with automatic fallback:

1. **Primary — RSS feeds:** `http://rss.arxiv.org/rss/{category}` for each monitored category. Fastest, returns today's new submissions.
2. **Backup — arxiv API:** `http://export.arxiv.org/api/query` with category and date range filters.
3. **Last resort — HTML scraping:** Parse the arxiv "new submissions" listing page for each category.

Each source returns papers normalized to:

```json
{
  "arxiv_id": "2504.12345",
  "title": "Paper Title",
  "authors": ["Author One", "Author Two"],
  "abstract": "Full abstract text...",
  "categories": ["cs.CL", "cs.LG"],
  "url": "https://arxiv.org/abs/2504.12345"
}
```

Papers are deduplicated by `arxiv_id` (papers often appear in multiple categories).

**Retry logic:** Exponential backoff starting at 2 seconds, doubling each retry, capped at 60 seconds between retries. Maximum 100 retries per source. Random jitter (0-1s) added to each wait to avoid thundering herd. If a source fails all retries, fall through to the next source.

**Output:** `staging/<date>/01_fetched.json`

### Stage 2: Filter (`filter.py`)

Sends paper abstracts to GPT-5.2 in batches (~20 abstracts per LLM call to stay within context limits). The system prompt encodes the four research interest categories with explicit selectivity guidance (especially strict for categories 3 and 4).

The LLM returns structured JSON:

```json
[
  {
    "arxiv_id": "2504.12345",
    "selected": true,
    "categories": [1, 3],
    "reasoning": "Brief explanation of why this paper was selected"
  }
]
```

Papers can match multiple interest categories. If zero papers are selected, the pipeline exits cleanly with a log message (no PDF generated that day).

**Output:** `staging/<date>/02_filtered.json`

### Stage 3: Summarize (`summarize.py`)

For each selected paper, makes an individual LLM call with the full abstract. The prompt requests:

- **One-line takeaway** — the "so what" of the paper
- **Key contribution** — what they did (2-3 sentences)
- **Method** — how they did it (2-3 sentences)
- **Most important result** — the headline finding (1-2 sentences)

Each summary targets ~100-150 words to fit the 2-page budget with ~5-20 papers. More relevant and insightful papers get more space, less relevant papers get barely a sentence or two. Make sure you explain difficult things in a few words and not just throw jargon.

**Output:** `staging/<date>/03_summaries.json`

### Stage 4: Verify (`verify.py`)

A separate LLM call per paper. Given the original abstract and the generated summary, the model checks for:

- Factual inconsistencies between summary and abstract
- Hallucinated claims not supported by the abstract
- Misrepresented results or conclusions

If a summary fails verification:

1. Regenerate the summary once, including the verification feedback as guidance
2. Re-verify the regenerated summary
3. If it still fails, include the summary with a warning annotation

**Output:** `staging/<date>/04_verified.json`

### Stage 5: Compile (`compile.py`)

1. **Render LaTeX:** Jinja2 populates the NeurIPS digest template with verified summaries, grouped by interest category. Each summary includes a clickable arxiv link. A references section at the end lists all papers with full links.
2. **Compile PDF:** `tectonic` compiles the `.tex` file to PDF.
3. **Page check:** If the PDF exceeds 2 pages, re-prompt the LLM to compress all summaries proportionally — shorter descriptions that still preserve the key insight and result, just tighter prose. Recompile. Repeat until ≤2 pages (max 3 compression iterations, then accept as-is with a log warning).

**Layout principles:**

- Minimal headers — no big title blocks, no author bylines, no abstract section header
- Small date header, then straight into summaries grouped by category with minimal section dividers
- Maximize content density — every line should carry information
- Clickable arxiv links inline with each paper
- Compact references section at the end

**Output:** `output/<date>.pdf`

## LLM Client Configuration (`llm.py`)

- **Base URL:** `https://llm-proxy.us-east-2.int.infra.intelligence.webex.com/openai/v1`
- **API key:** from `WEBEX_TOKEN` environment variable
- **Headers:** `{"x-cisco-app": "daily-arxiv-feed"}`
- **Model:** `gpt-5.2`
- **Structured output:** JSON mode used for filter, summarize, and verify stages
- **Retry on transient errors:** 429, 500, 502, 503 — exponential backoff, max 5 retries per call

## Token Management

`run.sh` handles token generation:

```bash
export WEBEX_TOKEN=$(token-generator personal bts)
python -m daily_arxiv_feed.main
```

The browser-based OAuth flow completes automatically via cached credentials. `token-generator` outputs a raw JWT token string to stdout, so shell command substitution captures it directly.

## Scheduling

### launchd plist (`com.pranavpg.daily-arxiv.plist`)

Installed at `~/Library/LaunchAgents/com.pranavpg.daily-arxiv.plist`:

- **Schedule:** Daily at 07:00 via `StartCalendarInterval`
- **Missed jobs:** launchd's built-in catch-up runs the job on wake if it was missed
- **Working directory:** `~/daily-arxiv-feed`
- **Program:** `run.sh`
- **Logs:** `StandardOutPath` and `StandardErrorPath` → `logs/`

### install.sh

- Creates `output/`, `output/staging/`, `logs/` directories
- Copies plist to `~/Library/LaunchAgents/`
- Runs `launchctl load` to activate the schedule

## Error Handling & Logging

- **Per-run log file** at `logs/<date>.log` capturing stdout and stderr
- **Arxiv fetch failure:** if all three sources fail after retries, log error and exit cleanly (no PDF)
- **LLM proxy unreachable:** retry with backoff, then exit with error logged
- **tectonic failure:** log the LaTeX compilation error, preserve the `.tex` file at `output/staging/<date>/digest.tex` for manual inspection
- **Zero papers selected:** log message, clean exit, no PDF
- **No silent failures:** every error path produces a log entry explaining what happened and why
