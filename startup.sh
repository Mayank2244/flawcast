#!/bin/bash
# FlowCast AI — Hugging Face Spaces startup script
set -euo pipefail

export HF_DEMO_MODE="${HF_DEMO_MODE:-true}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "[FlowCast] Starting in demo mode=$HF_DEMO_MODE"

# Generate demo data if missing
if [ ! -f "data/traffic_data.csv" ] || [ ! -f "fusion_output.json" ]; then
  echo "[FlowCast] Generating demo data..."
  python generate_demo_data.py
fi

# Build features if missing
if [ ! -f "data/feature_matrix.parquet" ]; then
  echo "[FlowCast] Building feature matrix..."
  python 02_features.py || true
fi

# Download model weights from HF Hub on first run (optional)
if [ -n "${HF_MODEL_REPO:-}" ] && [ ! -f "models/lstm_ae.pt" ]; then
  echo "[FlowCast] Downloading models from $HF_MODEL_REPO..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_MODEL_REPO', local_dir='models')" || true
fi

# Start background fusion engine (weather fetched inside fusion cycle)
echo "[FlowCast] Starting fusion loop..."
python -c "
from importlib import import_module
import time
mod = import_module('07_fusion_engine')
mod.run_fusion_loop(interval_minutes=15)
while True: time.sleep(3600)
" &

# Launch Streamlit on HF default port
PORT="${PORT:-7860}"
echo "[FlowCast] Launching Streamlit on port $PORT"
exec streamlit run dashboard/app.py --server.port="$PORT" --server.address=0.0.0.0
