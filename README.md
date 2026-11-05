# daily-arxiv-feed

Two daily arxiv digests run **serially**: the NLP PDF finishes first, then the quantum computing pipeline starts. NLP uses readable compressed abstracts (up to 3 pages); quantum includes the selected papers' **full author abstracts**, with no summary or page-limit compression.

## What It Does

Every morning, the NLP pipeline:

1. **Fetches** new papers from 5 arxiv categories (cs.CL, cs.SD, eess.AS, cs.LG, cs.AI) via RSS, with API and HTML scrape as automatic fallbacks
2. **Filters** using a two-pass LLM approach — a broad first pass identifies candidates, then a global ranking pass enforces strict per-category quotas to select 5-8 papers
3. **Summarizes** each paper as one readable compressed abstract: the question, approach, and main finding in complete sentences
4. **Verifies** factual support and readability against the source abstract, regenerating on failure and falling back to whole source sentences if the rewrite still fails
5. **Generates** a field overview highlighting the day's dominant themes across all submissions
6. **Compiles** everything into a compact PDF using a NeurIPS-style LaTeX template
7. **Compresses** by selecting fewer claims if the PDF exceeds the 3-page limit, then verifies the rewritten text again

Excerpts have hard limits of **130 words and 1,000 characters**, including spaces.
Rebuilding an existing digest also caps each excerpt at **15% above its previous
word and character counts**; the smaller limits apply. Those limits persist across
rewrites so repeated regeneration cannot grow the allowance. Compression retains
full sentences rather than removing articles, subjects, or connecting words.

## Research Categories

| Category | Quota | Description |
|----------|-------|-------------|
| Multilingual NLP | 3-4 papers | Core multilingual or cross-lingual NLP contributions |
| Multilingual Speech | 1-2 papers | Multilingual or cross-lingual speech/audio processing |
| Foundation Model Theory | 0-1 papers | Genuine mechanistic insights into how large models work |
| LLM Evaluation | 0-1 papers | Novel automatic evaluation methodology or counter-intuitive findings |

## Quantum Computing Digest

Once NLP finishes, a separate fetch/filter/render pipeline scans `quant-ph`,
`cs.LG`, and `physics.app-ph`, deduplicates papers, and selects at most 8:

| Section | Focus |
|---------|-------|
| ML for quantum control and pulse optimization | Learned control, reinforcement learning, Bayesian optimization, calibration, noise robustness, and hardware-aware pulse/gate design |
| ML for error correction, mitigation and fault tolerance | Learned decoders, syndrome/noise inference, adaptive correction, mitigation, and reliable logical operations |
| Notable quantum computing research | Strong general results in algorithms, complexity, hardware, codes, logical qubits, and architectures, even without ML |

The two ML-focused sections have priority. Category preferences are not minimum
quotas: weak or irrelevant papers do not get added just to fill a section. Routine
quantum classifiers on classical datasets and quantum-inspired classical
optimization are excluded unless the work establishes a substantial advance.

The LLM selects papers but **never rewrites their abstracts**. Only the RSS
announcement header is removed. Full text, source hashes, selection reasons and
run populations are saved under `output/quantum/staging/<date>/`. Safe math
expressions are typeset; unknown source TeX commands are escaped as literal text.
Quantum has no word or page cap. Empty abstracts are recorded as exclusions;
selection failures are recorded separately from days with no qualifying papers.
Completed first-pass batches are checkpointed so interrupted selection can resume
when the source inputs and filter prompt are unchanged.

Output paths are separate:

- NLP: `output/<date>.pdf`
- Quantum: `output/quantum/<date>.pdf`

The existing 7 AM launchd job still calls `run.sh`. It now opens the NLP PDF as
soon as it completes, then runs quantum and opens that PDF when ready. No second
scheduled job or overlapping pipeline is created. An NLP failure stops the
sequence; an NLP day with no qualifying papers still allows quantum to run.

## Sample Output

The digest has clickable arxiv links, papers grouped by category, and a field overview at the bottom summarizing the day's trends. Each paper has one short paragraph rather than overlapping takeaway, contribution, method, and result fields.

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

Both pipelines run sequentially and their PDFs open as they complete. A macOS
notification reports completion or failure. To run either one separately, or to
preview both without publishing:

```bash
poetry run python -m daily_arxiv_feed.main --digest nlp
poetry run python -m daily_arxiv_feed.main --digest quantum
poetry run python -m daily_arxiv_feed.main --digest all --no-push --open
```

Fresh runs publish PDFs to the current Git branch unless `--no-push` is set.

To rewrite an existing digest using cached papers and the same paper selection:

```bash
poetry run python -m daily_arxiv_feed.main --rebuild 2026-11-05
```

This requires `WEBEX_TOKEN` as usual. Rebuilds are local: they do not fetch new
papers, commit, or push. The original PDF, LaTeX, and summaries are retained under
`output/staging/<date>/readability-before/`. Per-paper before/after text and length
counts are saved in `06_readability_comparison.json` in the same staging directory.

To retypeset cached quantum abstracts without any LLM calls or publishing:

```bash
poetry run python -m daily_arxiv_feed.main --digest quantum --rebuild 2026-11-05
```

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
[Compress and re-verify if > 3 pages] --> output/<date>.pdf
```

Intermediate JSON files are saved in `output/staging/<date>/` for debugging and inspection.

## Project Structure

```
daily_arxiv_feed/
    fetch.py        # Three-source arxiv fetcher with fallback
    filter.py       # Two-pass LLM filtering with per-category quotas
    quantum.py      # Quantum selection, source integrity and full-abstract pipeline
    abstract_rendering.py  # Safe mathematical typesetting for full abstracts
    summarize.py    # Compressed abstract generation
    excerpts.py     # Shared prose rules, length limits, source-sentence fallback
    verify.py       # Summary verification against abstracts
    overview.py     # Field trend analysis
    compile.py      # LaTeX rendering and PDF compilation
    llm.py          # LLM client with retry logic
    main.py         # Pipeline orchestrator
templates/
    neurips_digest.tex  # Jinja2 LaTeX template
tests/              # Unit and regression tests
```

## Testing

```bash
poetry run pytest tests/ -v
```

## License

MIT
