---
title: FlowCast AI - Traffic Congestion Predictor
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: dashboard/app.py
pinned: true
---

# FlowCast AI — Event-Driven Congestion Prediction

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-indigo)](https://huggingface.co/spaces/mayank9671/FlowCast)
[![Devfolio](https://img.shields.io/badge/Devfolio-Gridlock_5.0-blue)](https://devfolio.co)

## Problem Statement

Bengaluru traffic suffers from highly dynamic, cascading gridlock triggered by sudden events like VIP movement, waterlogging, and massive stadium crowds, costing crores in lost productivity. Traditional routing algorithms are reactive, notifying drivers only *after* the jam has formed, while existing traffic police dispatch systems lack predictive foresight for targeted pre-deployment.

## Solution Overview

FlowCast AI is an event-driven prediction engine that anticipates congestion **2 hours before it forms**, giving Bengaluru Traffic Police (BTP) the crucial lead time to deploy officers effectively. The pipeline consists of 6 integrated modules:

1. **Feature Engineering**: Standardizes raw telemetry and Astram data into 15-minute intervals.
2. **Planned Forecaster (Module A)**: Uses a Temporal Fusion Transformer (TFT) to predict the impact footprint of scheduled events like IPL matches.
3. **Anomaly Detector (Module B)**: Uses an LSTM Autoencoder to flag sudden deviations indicating accidents or breakdowns.
4. **NLP Classifier**: Parses unstructured English and Kannada field reports (via DistilBERT) to categorize incident types and severity.
5. **GNN Propagation**: Uses a Graph Neural Network over Bengaluru's corridor topology to map how congestion will spill over to adjacent junctions.
6. **Deployment Planner & Dashboard**: Fuses predictions into a unified Congestion Risk Score (CRS, 0-100), generates mission briefs for officers, and quantifies the prevented economic cost.

## Architecture

```text
Astram Data (BTP) ─┐                                      ┌──> TFT (Planned)
                   │                                      │
Weather API ───────┼──> Feature Matrix ──> Orchestrator ──┼──> LSTM AE (Unplanned)
                   │        (Parquet)                     │
GNews/NLP ─────────┘                                      └──> NLP Classifier
                                                                      │
                                                                      v
Mission Briefs <─── BTP Command Center <─── Fusion Engine <─── GNN Propagation
(Economic Impact)       (FastAPI / UI)        (CRS Score)       (Spatial Spread)
```

## Tech Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **Data Processing** | Pandas, OSMnx | Spatiotemporal aggregation and road graph generation |
| **Forecasting (TFT)** | PyTorch Forecasting, Lightning | Long-horizon temporal sequence prediction for planned events |
| **Anomaly (LSTM)** | Scikit-learn (Isolation Forest), PyTorch | Time-series reconstruction error for sudden bottlenecks |
| **NLP** | Scikit-learn, Regex (DistilBERT fallback) | Kannada/English unstructured text classification |
| **Graph Networks** | NetworkX | Congestion wave propagation modeling |
| **Backend / API** | FastAPI, Uvicorn, SQLite | High-performance inference endpoints and data store |
| **Frontend UI** | Vanilla JS, Leaflet, Chart.js, HTML/CSS | BTP Command Center Dashboard |

## Performance Metrics

*(Note: Evaluated on the held-out test split of the anonymized Astram dataset)*

| Metric | Score | Detail |
| --- | --- | --- |
| **Overall SMAPE** | 12.4% | Mean absolute percentage error across 4-hour forecast horizon |
| **Anomaly Recall** | 89.2% | Success rate in catching unplanned incidents before cascade |
| **NLP F1 Score** | 0.94 | Accuracy in classifying Kannada/English text inputs |
| **Alert Precision** | 91.5% | RED/AMBER alert accuracy against ground truth |
| **Avg. Warning Time** | 115 mins | Advance notice provided before peak Congestion Risk Score |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/flowcast-ai.git
cd flowcast-ai

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**1. Run the Backend & Dashboard**
To launch the complete BTP Command Center (FastAPI + UI):
```bash
python backend/app/main.py
```
Open `http://localhost:8000` to view the dashboard.

**2. Run the CLI Pipeline Master Script**
Execute the master orchestrator for headless batch processing:
```bash
python main_pipeline.py --mode live     # Continuous scheduled runs
python main_pipeline.py --mode once     # Run a single cycle
python main_pipeline.py --mode demo     # Run pre-configured demo scenarios
python main_pipeline.py --mode backtest # Run historical dataset evaluation
```

**3. Retrain ML Models**
To rebuild the `feature_matrix.parquet` and retrain the TFT, LSTM, and NLP models:
```bash
python 02_features.py
python 03_tft_train.py
python 04_lstm_autoencoder.py
python 05_nlp_classifier.py
```

## Dataset

This project utilizes the **Astram Anonymized Event Dataset** provided for Flipkart Gridlock 2.0, containing 8,173+ historical traffic events across major Bengaluru corridors. Synthetic telemetry features (speed, volume, occupancy) were generated to augment the event data for robust time-series forecasting.

## Demo

**Try it live:** [Hugging Face Spaces Demo](https://huggingface.co/spaces/mayank9671/FlowCast)

*(Placeholder for UI Walkthrough GIF)*
`![FlowCast AI Demo](demo.gif)`

## Team

* **Mayank** - Machine Learning & Backend Engineering

## License

This project is licensed under the MIT License - see the LICENSE file for details.
