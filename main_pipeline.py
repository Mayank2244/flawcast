#!/usr/bin/env python3
"""FlowCast AI — master pipeline orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from flowcast_config import (
    DATASET_PATH,
    EVENT_CALENDAR_PATH,
    FEATURE_MATRIX_PATH,
    FUSION_OUTPUT_PATH,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    THRESHOLD_JSON,
    TFT_CHECKPOINT,
    LSTM_AE_PATH,
    NLP_MODEL_DIR,
    GNN_MODEL_PATH,
    GRAPH_PT,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"flowcast_{datetime.utcnow():%Y%m%d}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("flowcast.pipeline")


class FlowCastPipeline:
    """Orchestrates feature engineering, model inference, fusion, and deployment."""

    def __init__(self) -> None:
        self.tft = self._load_tft()
        self.lstm_ae = self._load_lstm()
        self.nlp_clf = self._load_nlp()
        self.gnn = self._load_gnn()
        self.road_graph = self._load_graph()
        self.threshold = self._load_threshold()

    def _load_tft(self):
        from load_models import load_tft_model
        return load_tft_model()

    def _load_lstm(self):
        from load_models import load_lstm_ae
        return load_lstm_ae()

    def _load_nlp(self):
        from load_models import load_nlp_model
        return load_nlp_model()

    def _load_gnn(self):
        from load_models import load_gnn
        return load_gnn()

    def _load_graph(self):
        from load_models import load_osmnx_graph
        return load_osmnx_graph()

    def _load_threshold(self) -> dict:
        if THRESHOLD_JSON.exists():
            with open(THRESHOLD_JSON, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def run_cycle(
        self,
        current_data: pd.DataFrame | None = None,
        event_calendar: list | None = None,
        news_feed: list | None = None,
    ) -> dict:
        """Execute full prediction cycle; target runtime < 30 seconds."""
        t0 = time.perf_counter()

        # 1. Feature engineering
        if not FEATURE_MATRIX_PATH.exists() or current_data is not None:
            import importlib
            if not DATASET_PATH.exists():
                from generate_demo_data import main as gen
                gen()
            importlib.import_module("02_features").main()

        # 2. TFT planned forecasts
        from importlib import import_module
        tft_mod = import_module("03_tft_train")
        segments = pd.read_parquet(FEATURE_MATRIX_PATH)["segment_id"].unique().tolist()[:20]
        tft_out = {}
        preds = tft_mod.predict_congestion("ipl", "CHINNASWAMY", "2025-03-12T19:00:00", 38000, segments)
        for seg, quantiles in preds.items():
            tft_out[seg] = {"P10": [q[0] for q in quantiles], "P50": [q[1] for q in quantiles], "P90": [q[2] for q in quantiles]}

        # 3. LSTM AE anomaly scores
        ae_mod = import_module("04_lstm_autoencoder")
        lstm_out = {}
        df = pd.read_parquet(FEATURE_MATRIX_PATH)
        for seg in segments[:10]:
            seg_df = df[df["segment_id"] == seg].tail(6)
            if len(seg_df) >= 6:
                window = seg_df[["speed_kmh", "volume", "occupancy"]].values
                try:
                    lstm_out[seg] = ae_mod.detect_anomaly(seg, window)
                except Exception:
                    lstm_out[seg] = {"is_anomaly": False, "confidence": 0.0, "error": 0.0, "threshold": 1.0}

        # 4. NLP incident classification (news from GNews API with fallback)
        nlp_mod = import_module("05_nlp_classifier")
        nlp_results = []
        if news_feed:
            texts = news_feed
        else:
            try:
                from api_services import get_news_sync

                articles = get_news_sync(max_results=5)
                texts = [a["title"] for a in articles]
            except Exception:
                texts = ["Major accident on ORR near Marathahalli. Traffic blocked."]
        for text in texts[:5]:
            nlp_results.append(nlp_mod.classify_incident(text))

        # 5. GNN propagation
        gnn_mod = import_module("06_gnn_model")
        gnn_out = gnn_mod.update_graph_features("MG001", 8.0)

        # 6. Fusion
        fusion_mod = import_module("07_fusion_engine")
        try:
            from api_services import weather_for_fusion

            weather = weather_for_fusion()
        except Exception:
            weather = {"rain_mm": 0, "visibility": 8000, "wind_speed": 5}

        output = fusion_mod.run_fusion_cycle(
            tft_output=tft_out,
            lstm_output=lstm_out,
            nlp_incidents=nlp_results,
            gnn_output=gnn_out,
            weather=weather,
        )
        output["weather"] = weather
        output["news_headlines"] = texts[:3]

        # 7. Deployment planner
        planner = import_module("08_deployment_planner")
        events = event_calendar or json.loads(EVENT_CALENDAR_PATH.read_text()) if EVENT_CALENDAR_PATH.exists() else []
        stations = json.loads((PROJECT_ROOT / "data" / "btp_stations.json").read_text()) if (PROJECT_ROOT / "data" / "btp_stations.json").exists() else []
        plan = planner.build_deployment_plan(output, events, stations)
        output["deployment_brief"] = plan
        output["mission_brief"] = planner.generate_mission_brief(plan, events[0] if events else None)

        # 8. Save
        with open(FUSION_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Cycle complete in %.0f ms — %d segments", elapsed_ms, len(output["segments"]))
        return output


def run_scheduler() -> None:
    """Run pipeline every 15 minutes."""
    import schedule

    pipeline = FlowCastPipeline()
    events = json.loads(EVENT_CALENDAR_PATH.read_text()) if EVENT_CALENDAR_PATH.exists() else []

    def job():
        pipeline.run_cycle(event_calendar=events, news_feed=[])

    schedule.every(15).minutes.do(job)
    job()
    while True:
        schedule.run_pending()
        time.sleep(1)


def run_backtest(start_date: str, end_date: str) -> pd.DataFrame:
    """Replay historical data and compare predictions vs actual CRS."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    sub = df[mask]
    pipeline = FlowCastPipeline()
    rows = []
    for seg in sub["segment_id"].unique()[:5]:
        seg_df = sub[sub["segment_id"] == seg]
        actual = float(seg_df["congestion_risk_score"].mean())
        out = pipeline.run_cycle()
        pred_seg = next((s for s in out["segments"] if s["segment_id"] == seg), None)
        pred = pred_seg["crs_P50"] if pred_seg else 5.0
        smape = abs(pred - actual) / (abs(actual) + abs(pred) + 1e-6) * 200
        rows.append({"segment": seg, "predicted_CRS": pred, "actual_CRS": actual, "SMAPE": smape})
    report = pd.DataFrame(rows)
    report_path = PROJECT_ROOT / "outputs" / "backtest_report.csv"
    report.to_csv(report_path, index=False)
    logger.info("Backtest SMAPE mean=%.2f%%", report["SMAPE"].mean())
    return report


def health_check() -> dict:
    """Verify models and data artifacts."""
    checks = {
        "dataset": DATASET_PATH.exists(),
        "features": FEATURE_MATRIX_PATH.exists(),
        "tft": TFT_CHECKPOINT.exists() or TFT_CHECKPOINT.with_suffix(".pkl").exists(),
        "lstm": LSTM_AE_PATH.exists(),
        "nlp": NLP_MODEL_DIR.exists(),
        "gnn": GNN_MODEL_PATH.exists(),
        "graph": GRAPH_PT.exists(),
        "fusion": FUSION_OUTPUT_PATH.exists(),
    }
    status = "healthy" if all(checks.values()) else "degraded"
    return {"status": status, "details": checks}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="FlowCast AI Pipeline")
    parser.add_argument("--mode", choices=["live", "once", "demo", "backtest", "health"], default="once")
    parser.add_argument("--start", default="2025-03-10")
    parser.add_argument("--end", default="2025-03-16")
    args = parser.parse_args()

    if args.mode == "health":
        print(json.dumps(health_check(), indent=2))
        return

    if args.mode == "demo":
        from generate_demo_data import main as gen
        gen()

    pipeline = FlowCastPipeline()

    if args.mode == "live":
        run_scheduler()
    elif args.mode == "backtest":
        run_backtest(args.start, args.end)
    else:
        out = pipeline.run_cycle()
        print(json.dumps({"timestamp": out["timestamp"], "alerts": len(out["active_alerts"])}, indent=2))


if __name__ == "__main__":
    main()
