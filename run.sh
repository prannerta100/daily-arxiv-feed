#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export WEBEX_TOKEN
WEBEX_TOKEN=$(token-generator personal bts)

TODAY=$(date +%Y-%m-%d)
PDF="$(pwd)/output/${TODAY}.pdf"

if poetry run python -m daily_arxiv_feed.main; then
    osascript -e 'display notification "Your daily arxiv digest is ready — opening now." with title "Arxiv Digest" sound name "Glass"'
    open "$PDF"
else
    osascript -e 'display notification "Pipeline failed — check logs/" with title "Arxiv Digest" sound name "Basso"'
fi
