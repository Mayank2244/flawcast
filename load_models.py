"""FlowCast AI — model loading utilities with CPU-optimised demo fallbacks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flowcast_config import (
    DEMO_MODE_ENV,
    GNN_MODEL_PATH,
    GRAPH_PT,
    LSTM_AE_PATH,
    MODELS_DIR,
    NLP_MODEL_DIR,
    PROJECT_ROOT,
    THRESHOLD_JSON,
    TFT_CHECKPOINT,
)


def _demo_mode() -> bool:
    return os.getenv(DEMO_MODE_ENV, "false").lower() in ("1", "true", "yes")


def load_tft_model() -> Any:
    """Load TFT checkpoint or sklearn fallback bundle."""
    pkl = TFT_CHECKPOINT.with_suffix(".pkl")
    if pkl.exists():
        import joblib
        return joblib.load(pkl)
    if TFT_CHECKPOINT.exists():
        return {"path": str(TFT_CHECKPOINT), "backend": "lightning"}
    return {"backend": "demo", "predict": lambda *a, **k: 5.0}


def load_lstm_ae() -> Any:
    """Load LSTM autoencoder weights and thresholds."""
    if LSTM_AE_PATH.exists() and THRESHOLD_JSON.exists():
        import torch
        from importlib import import_module
        ae_mod = import_module("04_lstm_autoencoder")
        model = ae_mod.TrafficAutoencoder()
        model.load_state_dict(torch.load(LSTM_AE_PATH, map_location="cpu", weights_only=True))
        model.eval()
        with open(THRESHOLD_JSON, encoding="utf-8") as f:
            thresholds = json.load(f)
        return {"model": model, "thresholds": thresholds}
    return {"backend": "demo", "thresholds": {}}


def load_nlp_model() -> Any:
    """Load DistilBERT classifier if available."""
    if NLP_MODEL_DIR.exists() and (NLP_MODEL_DIR / "config.json").exists():
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        return {
            "tokenizer": DistilBertTokenizerFast.from_pretrained(str(NLP_MODEL_DIR)),
            "model": DistilBertForSequenceClassification.from_pretrained(str(NLP_MODEL_DIR)),
        }
    return {"backend": "demo"}


def load_gnn() -> Any:
    """Load GNN weights or demo propagation matrices."""
    if GNN_MODEL_PATH.exists():
        import torch
        from importlib import import_module
        gnn_mod = import_module("06_gnn_model")
        ckpt = torch.load(GNN_MODEL_PATH, map_location="cpu", weights_only=False)
        model = gnn_mod.CongestionGAT()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return {"model": model, "node_map": ckpt.get("node_map", {})}
    return {"backend": "demo"}


def load_osmnx_graph() -> Any:
    """Load cached Bengaluru road graph."""
    if GRAPH_PT.exists():
        import pickle
        with open(GRAPH_PT, "rb") as f:
            return pickle.load(f)
    return {"graph": None, "node_map": {}}


def load_all_models() -> dict[str, Any]:
    """Load all FlowCast models with graceful demo fallbacks."""
    return {
        "tft": load_tft_model(),
        "lstm_ae": load_lstm_ae(),
        "nlp_clf": load_nlp_model(),
        "gnn": load_gnn(),
        "road_graph": load_osmnx_graph(),
        "demo_mode": _demo_mode(),
    }


try:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading FlowCast AI models...")
    def cached_load_all_models() -> dict[str, Any]:
        return load_all_models()

except ImportError:
    cached_load_all_models = load_all_models  # type: ignore
