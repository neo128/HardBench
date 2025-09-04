#!/usr/bin/env bash
set -euo pipefail

# One-command installer for Unix (Linux/macOS)
# Usage: bash scripts/install_unix.sh [--venv .venv]

VENVDIR=""
if [[ "${1:-}" == "--venv" && -n "${2:-}" ]]; then
  VENVDIR="$2"
fi

if [[ -n "$VENVDIR" ]]; then
  python3 -m venv "$VENVDIR"
  # shellcheck disable=SC1090
  source "$VENVDIR/bin/activate"
fi

python3 -m pip install --upgrade pip
pip install .

echo "Installed robot-diagnostic-suite. Try: robot-diag --help"
