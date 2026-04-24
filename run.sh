#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export WEBEX_TOKEN
WEBEX_TOKEN=$(token-generator personal bts)

poetry run python -m daily_arxiv_feed.main
