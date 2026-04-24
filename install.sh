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
