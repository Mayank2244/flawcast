#!/usr/bin/env bash
# FlowCast AI — Zero-Config Hackathon Setup (SQLite + FastAPI)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════╗"
echo "║   FlowCast AI — Flipkart Gridlock Setup          ║"
echo "╚══════════════════════════════════════════════════╝"

# Python 3.10 venv
if ! command -v python3.10 &>/dev/null; then
  echo "ERROR: Python 3.10 required. Install: brew install python@3.10"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "→ Creating Python virtual environment (.venv)..."
  python3.10 -m venv .venv
fi
source .venv/bin/activate
echo "→ Installing dependencies..."
pip install -q -r requirements.txt

# Environment
if [ ! -f ".env" ]; then
  echo "→ Creating default .env file..."
  cat <<EOF > .env
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
EOF
fi

echo "→ Setting up SQLite database and importing dataset..."
# The main app will automatically create the database and import data on startup.

echo ""
echo "✓ Setup complete! Starting FlowCast AI server..."
echo ""
echo "  Open browser:     http://localhost:8000"
echo "  API docs:         http://localhost:8000/docs"
echo ""

cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
