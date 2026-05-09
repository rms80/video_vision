#!/usr/bin/env bash
# Force-restart the segviewer Vite dev server on port 4444.
# Kills any existing listener on 4444, then starts `npm run dev`.
set -euo pipefail

PORT=4444
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGVIEWER_DIR="$(dirname "$SCRIPT_DIR")"

echo "[restart-dev] checking port $PORT..."
PIDS=$(netstat -ano | awk -v p=":$PORT" '$2 ~ p"$" && $4 == "LISTENING" {print $5}' | sort -u)

if [[ -n "${PIDS:-}" ]]; then
  for pid in $PIDS; do
    echo "[restart-dev] killing PID $pid on port $PORT"
    taskkill //F //PID "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
else
  echo "[restart-dev] port $PORT is free"
fi

cd "$SEGVIEWER_DIR"
echo "[restart-dev] starting vite dev server in $SEGVIEWER_DIR"
exec npm run dev
