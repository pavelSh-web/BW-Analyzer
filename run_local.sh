#!/usr/bin/env bash
set -euo pipefail

# Local runner for the FastAPI service
# Usage:
#   ./run_local.sh                              # runs on 0.0.0.0:8000
#   PORT=8001 ./run_local.sh                    # override port
#   HOST=127.0.0.1 ./run_local.sh               # override host
#   PYTHON=python3.11 ./run_local.sh            # override python executable

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY="${PYTHON:-python3}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python not found. Set PYTHON env var if needed, e.g. PYTHON=python3.11" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found. Some formats may fail to load. On macOS: brew install ffmpeg" >&2
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
pip install -r requirements.txt

# Ensure PYTHONPATH includes project root
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Running API on $HOST:$PORT (venv: $PROJECT_DIR/.venv)"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload


