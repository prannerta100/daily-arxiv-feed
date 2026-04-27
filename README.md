# daily-arxiv-feed

An automated daily digest that fetches new papers from arxiv, uses an LLM to filter and summarize the most relevant ones, and compiles a concise 2-page PDF you can read with your morning coffee.

## What It Does

Every morning, the pipeline:

1. **Fetches** new papers from 5 arxiv categories (cs.CL, cs.SD, eess.AS, cs.LG, cs.AI) via RSS, with API and HTML scrape as automatic fallbacks
2. **Filters** using a two-pass LLM approach — a broad first pass identifies candidates, then a global ranking pass enforces strict per-category quotas to select 5-8 papers
3. **Summarizes** each paper with structured fields: one-line takeaway, key contribution, method, and headline result
4. **Verifies** every summary against the original abstract for factual accuracy, regenerating on failure
5. **Generates** a field overview highlighting the day's dominant themes across all submissions
6. **Compiles** everything into a compact PDF using a NeurIPS-style LaTeX template
7. **Compresses** summaries automatically if the PDF exceeds the 2-page limit

## Research Categories

| Category | Quota | Description |
|----------|-------|-------------|
| Multilingual NLP | 3-4 papers | Core multilingual or cross-lingual NLP contributions |
| Multilingual Speech | 1-2 papers | Multilingual or cross-lingual speech/audio processing |
| Foundation Model Theory | 0-1 papers | Genuine mechanistic insights into how large models work |
| LLM Evaluation | 0-1 papers | Novel automatic evaluation methodology or counter-intuitive findings |

## Sample Output

The digest is a dense, readable 2-page PDF with clickable arxiv links, grouped by category, with a field overview at the bottom summarizing the day's trends.

## Setup

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/)
- [tectonic](https://tectonic-typesetting.github.io/) (LaTeX compiler)
- An OpenAI-compatible LLM API endpoint

### Installation

```bash
git clone https://github.com/prannerta100/daily-arxiv-feed.git
cd daily-arxiv-feed
poetry install
```

### Configuration

Set the `WEBEX_TOKEN` environment variable with your API token, or modify `run.sh` to generate it however you prefer.

The LLM endpoint is configured in `daily_arxiv_feed/llm.py` — update `BASE_URL` and headers to match your provider.

### Run Manually

```bash
./run.sh
```

The PDF will be saved to `output/<date>.pdf` and you'll get a macOS notification when it's done.

### Schedule Daily Runs (macOS)

```bash
./install.sh
```

This installs a launchd job that runs every morning at 7 AM. If your Mac is asleep at 7 AM, it runs when you wake it.

```bash
# Check status
launchctl list | grep daily-arxiv

# Stop the daily job
launchctl unload ~/Library/LaunchAgents/com.pranavpg.daily-arxiv.plist

# Re-enable
launchctl load ~/Library/LaunchAgents/com.pranavpg.daily-arxiv.plist
```

## Architecture

```
arxiv (RSS/API/scrape)
    |
    v
[Fetch] --> 01_fetched.json
    |
    v
[Filter: 2-pass LLM] --> 02_filtered.json
    |
    v
[Summarize] --> 03_summaries.json
    |
    v
[Verify against abstracts] --> 04_verified.json
    |
    v
[Compile LaTeX --> PDF]
    |
    v
[Compress if > 2 pages] --> output/<date>.pdf
```

Intermediate JSON files are saved in `output/staging/<date>/` for debugging and inspection.

## Project Structure

```
daily_arxiv_feed/
    fetch.py        # Three-source arxiv fetcher with fallback
    filter.py       # Two-pass LLM filtering with per-category quotas
    summarize.py    # Structured paper summarization
    verify.py       # Summary verification against abstracts
    overview.py     # Field trend analysis
    compile.py      # LaTeX rendering and PDF compilation
    llm.py          # LLM client with retry logic
    main.py         # Pipeline orchestrator
templates/
    neurips_digest.tex  # Jinja2 LaTeX template
tests/              # 26 unit tests
```

## Testing

```bash
poetry run pytest tests/ -v
```

## License

MIT
