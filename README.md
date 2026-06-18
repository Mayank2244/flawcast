# FlowCast AI — Flipkart Gridlock 2.0 / 5.0

> **"We predict the gridlock before it starts — 2 hours early."**

Event-Driven Congestion Prediction System for Bengaluru Traffic Police (BTP), built on **8,173 real Astram events** from the hackathon dataset.

## Why This Wins

| Differentiator | What It Does |
|----------------|--------------|
| **Dual-Mode AI** | Planned events (TFT-inspired) + Unplanned incidents (LSTM AE + NLP) |
| **2-Hour Forecast** | 4-hour multi-horizon prediction at 15-min intervals with P10/P50/P90 bands |
| **Kannada NLP** | Classifies officer field reports in English + Kannada (e.g. `ಬಿಎಂಟಿಸಿ ಬಸ್ ಕೆಟ್ಟು ನಿಂತಿದೆ`) |
| **GNN Propagation** | Road-network-aware congestion spread across 25 Bengaluru nodes |
| **BTP Command Center** | Live heatmap, SSE alert stream, officer deployment briefs |
| **Economic Impact** | Rupee cost quantification — judges see ₹ Cr savings per deployment |
| **Judge Demo Mode** | 6 one-click pitch scenarios — no manual input during presentation |

## Tech Stack

- **Python 3.11** — FastAPI backend + ML pipeline
- **MySQL 8.0** — Database (MySQL Workbench or Docker)
- **scikit-learn** — TFT-inspired forecaster, Isolation Forest anomaly, TF-IDF NLP
- **NetworkX** — Graph neural propagation model
- **Leaflet + Chart.js** — BTP Live Dashboard

## Quick Start (Recommended)

```bash
# 1. One-command setup (Python 3.11 + Docker MySQL)
chmod +x setup.sh
./setup.sh

# 2. Start dashboard
cd backend && ../venv311/bin/uvicorn app.main:app --reload --port 8000

# 3. Open http://localhost:8000 → click "Judge Demo" tab
```

## Manual Setup (MySQL Workbench)

### 1. MySQL Database

1. Open **MySQL Workbench** → connect to local MySQL 8.0
2. **File → Open SQL Script** → select `database/schema.sql`
3. Click **Execute** (⚡) — creates `flowcast_ai` with 10 tables

### 2. Python 3.11 Environment

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: MYSQL_PASSWORD=your_mysql_password
```

### 3. Import Data & Train Models

```bash
cd backend
python scripts/import_data.py    # Loads 8,173 Astram events
python scripts/train_models.py   # Trains ML + generates alerts
```

### 4. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — BTP Command Center dashboard.

> **Note:** If port 8000 is in use, use `--port 8001`.

## Docker MySQL (Alternative)

```bash
docker compose up -d    # MySQL on localhost:3306, password: flowcast123
# .env is pre-configured for this
```

## Project Structure

```
flowcastAI/
├── database/schema.sql          # MySQL schema (import in Workbench)
├── docker-compose.yml           # Optional MySQL container
├── setup.sh                     # One-command hackathon setup
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/routes.py        # REST + SSE endpoints
│   │   ├── ml/                  # Planned, Unplanned, NLP, GNN, Fusion
│   │   └── services/            # Deployment, Economic, Demo scenarios
│   └── scripts/
│       ├── import_data.py       # Astram CSV → MySQL
│       └── train_models.py      # Train all ML models
├── frontend/                    # BTP Live Dashboard
├── dataset/                     # Hackathon Astram event data
└── models/                      # Trained artifacts (generated)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/stats` | GET | KPI metrics |
| `/api/dashboard/heatmap` | GET | Map heatmap data |
| `/api/alerts` | GET | Live alert feed |
| `/api/alerts/stream` | GET | SSE real-time alert stream |
| `/api/deployments` | GET | Officer deployment briefs |
| `/api/predict` | POST | Run fusion prediction |
| `/api/demo/scenarios` | GET | Judge demo scenarios |
| `/api/demo/run/{id}` | POST | Run one-click demo |
| `/api/nlp/classify` | POST | NLP incident classification |
| `/api/corridors/risk-index` | GET | Corridor Risk Index |
| `/api/graph/stats` | GET | GNN graph statistics |

## Judge Demo Script (30 seconds)

1. Open dashboard → **Judge Demo** tab
2. Click **"IPL Match — Chinnaswamy Stadium"** → shows CRS score, 4-hour forecast, officer brief
3. Click **"BMTC Breakdown — Kannada NLP"** → NLP classifies Kannada text live
4. Show **Analytics** tab → Corridor Risk Index + GNN stats

**Pitch line:**
> "Every other team shows you the gridlock after it happens. FlowCast AI shows you the gridlock **before it starts** — 2 hours early, trained on 8,173 real Bengaluru Astram events. We solve BOTH planned events AND unplanned incidents with one unified AI engine, and tell BTP exactly where to deploy officers and how much productivity they'll save."

## ML Model Performance (on Astram dataset)

| Model | Metric | Value |
|-------|--------|-------|
| Planned Forecaster | Accuracy | ~99.7% |
| Anomaly Detector | Anomaly Rate | 8.0% |
| NLP Classifier | Accuracy | ~84.6% |

Built for **Flipkart Gridlock 2.0** — Round 2 Hackathon Submission.
# flawcast
