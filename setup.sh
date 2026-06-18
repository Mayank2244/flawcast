#!/usr/bin/env bash
# FlowCast AI — One-command hackathon setup (Python 3.11 + MySQL)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════╗"
echo "║   FlowCast AI — Flipkart Gridlock Setup          ║"
echo "╚══════════════════════════════════════════════════╝"

# Python 3.11 venv
if ! command -v python3.11 &>/dev/null; then
  echo "ERROR: Python 3.11 required. Install: brew install python@3.11"
  exit 1
fi

if [ ! -d "venv311" ]; then
  echo "→ Creating Python 3.11 virtual environment..."
  python3.11 -m venv venv311
fi
source venv311/bin/activate
pip install -q -r requirements.txt

# Environment
if [ ! -f ".env" ]; then
  cp .env.example .env
fi

# MySQL via Docker (preferred) or existing local MySQL
if command -v docker &>/dev/null; then
  echo "→ Starting MySQL 8.0 via Docker..."
  docker compose up -d mysql
  echo "→ Waiting for MySQL to be ready..."
  for i in $(seq 1 40); do
    if docker compose exec -T mysql mysqladmin ping -h localhost -pflowcast123 --silent 2>/dev/null; then
      break
    fi
    sleep 2
  done
  # Update .env for Docker MySQL
  if grep -q "MYSQL_PASSWORD=your_password" .env 2>/dev/null; then
    sed -i.bak 's/MYSQL_PASSWORD=your_password/MYSQL_PASSWORD=flowcast123/' .env
    rm -f .env.bak
  fi
else
  echo "⚠ Docker not found. Ensure MySQL Workbench / MySQL Server is running."
  echo "  Import database/schema.sql manually, then set MYSQL_PASSWORD in .env"
fi

export MYSQL_HOST="${MYSQL_HOST:-localhost}"
export MYSQL_PASSWORD="${MYSQL_PASSWORD:-flowcast123}"

echo "→ Importing Astram dataset (8,200+ events)..."
cd backend
python scripts/import_data.py

echo "→ Training ML models & generating alerts..."
python scripts/train_models.py

echo ""
echo "✓ Setup complete!"
echo ""
echo "  Start dashboard:  cd backend && ../venv311/bin/uvicorn app.main:app --reload --port 8000"
echo "  Open browser:     http://localhost:8000"
echo "  API docs:         http://localhost:8000/docs"
echo ""
