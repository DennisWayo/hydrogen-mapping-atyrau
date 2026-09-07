#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-monitor}"
REMOTE_HOST="${REMOTE_HOST:-macstudio}"
REMOTE_ROOT="${REMOTE_ROOT:-~/XaiGis}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

sync_monitor() {
  rsync -az --delete --prune-empty-dirs --max-size=50m \
    --exclude='__MACOSX/**' \
    --exclude='**/._*' \
    --include='*/' \
    --include='*.json' \
    --include='*.geojson' \
    --include='*.csv' \
    --include='*.txt' \
    --include='*.log' \
    --include='*.md' \
    --include='*.png' \
    --include='*.jpg' \
    --include='*.jpeg' \
    --include='*.webp' \
    --exclude='*' \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/" "$LOCAL_ROOT/outputs/"

  rsync -az --delete --prune-empty-dirs --max-size=50m \
    --exclude='__MACOSX/**' \
    --exclude='**/._*' \
    --include='*/' \
    --include='*.json' \
    --include='*.geojson' \
    --include='*.csv' \
    --include='*.txt' \
    --include='*.log' \
    --include='*.md' \
    --include='*.png' \
    --include='*.jpg' \
    --include='*.jpeg' \
    --include='*.webp' \
    --exclude='*' \
    "$REMOTE_HOST:$REMOTE_ROOT/artifacts/" "$LOCAL_ROOT/artifacts/"
}

sync_full() {
  rsync -az --delete "$REMOTE_HOST:$REMOTE_ROOT/outputs/" "$LOCAL_ROOT/outputs/"
  rsync -az --delete "$REMOTE_HOST:$REMOTE_ROOT/artifacts/" "$LOCAL_ROOT/artifacts/"
}

case "$MODE" in
  monitor)
    echo "Running lightweight monitor sync from $REMOTE_HOST to $LOCAL_ROOT"
    sync_monitor
    ;;
  full)
    echo "Running full outputs+artifacts sync from $REMOTE_HOST to $LOCAL_ROOT"
    sync_full
    ;;
  *)
    echo "Usage: $0 [monitor|full]" >&2
    exit 1
    ;;
esac

echo "Sync complete."
