#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

poetry install --quiet --no-interaction

export WEBEX_TOKEN
WEBEX_TOKEN=$(token-generator personal bts)

if poetry run python -m daily_arxiv_feed.main --digest all --open; then
    osascript -e 'display notification "Daily run complete: NLP first, then quantum. Available PDFs have been opened." with title "Arxiv Digests" sound name "Glass"'
else
    osascript -e 'display notification "A digest failed — check logs/. Any earlier completed PDF remains available." with title "Arxiv Digests" sound name "Basso"'
    exit 1
fi
