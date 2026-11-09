#!/usr/bin/env bash
# AIO Local — Linux source-build launcher.
# Only Windows binaries are published; this script builds/runs from a checkout.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.11+ is required (python3 was not found)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[dev]"

exec python -m aiolocal "$@"
