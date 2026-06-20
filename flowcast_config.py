"""Shared configuration for FlowCast AI pipeline scripts."""
from __future__ import annotations

from pathlib import Path

# Project root (this file lives at repo root)
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Dataset paths ---
# traffic_data.csv: 15-min telemetry (generated from Astram or demo generator)
# ASTRAM_DATASET_PATH: raw BTP incident records (primary source)
DATASET_PATH = PROJECT_ROOT / "data" / "traffic_data.csv"
ASTRAM_DATASET_PATH = PROJECT_ROOT / "data" / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "feature_matrix.parquet"
FUSION_OUTPUT_PATH = PROJECT_ROOT / "fusion_output.json"
EDA_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "eda" / "eda_summary.csv"

# --- Column names (traffic time-series CSV) ---
TIMESTAMP_COL = "timestamp"
SEGMENT_ID_COL = "segment_id"
SPEED_COL = "speed_kmh"
VOLUME_COL = "volume"
OCCUPANCY_COL = "occupancy"
EVENT_TYPE_COL = "event_type"
VENUE_ID_COL = "venue_id"

# --- Model / artifact paths ---
MODELS_DIR = PROJECT_ROOT / "models"
TFT_CHECKPOINT = MODELS_DIR / "tft_flowcast.ckpt"
LSTM_AE_PATH = MODELS_DIR / "lstm_ae.pt"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"
NLP_MODEL_DIR = MODELS_DIR / "nlp_classifier"
NER_MODEL_DIR = MODELS_DIR / "ner_model"
GNN_MODEL_PATH = MODELS_DIR / "gnn_model.pt"
GRAPH_GPKG = PROJECT_ROOT / "data" / "bengaluru_graph.gpkg"
GRAPH_PT = PROJECT_ROOT / "data" / "graph.pt"

# --- Reference JSON inputs ---
EVENT_CALENDAR_PATH = PROJECT_ROOT / "data" / "event_calendar.json"
BTP_STATIONS_PATH = PROJECT_ROOT / "data" / "btp_stations.json"

# --- Output dirs ---
OUTPUTS_EDA_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUTS_BRIEFS_DIR = PROJECT_ROOT / "outputs" / "briefs"
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Training hyperparameters ---
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- CRS thresholds ---
CRS_CONGESTED_THRESHOLD = 6.0
CRS_RED_THRESHOLD = 7.5
CRS_AMBER_THRESHOLD = 5.0

# --- Demo mode ---
DEMO_MODE_ENV = "HF_DEMO_MODE"

# Bengaluru anchor for synthetic geocoding
BENGALURU_CENTER = (12.9716, 77.5946)

# Event types used across pipeline
EVENT_TYPES = [
    "ipl",
    "festival",
    "rally",
    "concert",
    "marathon",
    "government",
    "none",
]

NLP_CLASSES = [
    "accident_major",
    "accident_minor",
    "vip_convoy",
    "protest_bandh",
    "waterlogging",
    "road_closure",
    "clear",
]
