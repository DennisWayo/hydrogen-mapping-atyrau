#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${1:-60}"
MODE="${2:-monitor}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SECONDS" -lt 5 ]]; then
  echo "Interval must be an integer >= 5 seconds" >&2
  exit 1
fi

while true; do
  printf "[%s] syncing mode=%s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$MODE"
  "$ROOT_DIR/scripts/sync_from_macstudio.sh" "$MODE"
  sleep "$INTERVAL_SECONDS"
done
