#!/usr/bin/env python3
"""FlowCast AI — CRS fusion engine combining TFT, LSTM AE, NLP, GNN, and weather."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from flowcast_config import (
    CRS_AMBER_THRESHOLD,
    CRS_RED_THRESHOLD,
    DATASET_PATH,
    FEATURE_MATRIX_PATH,
    FUSION_OUTPUT_PATH,
    PROJECT_ROOT,
    SEGMENT_ID_COL,
    SPEED_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_MAP = {
    "accident_major": 4.0,
    "accident_minor": 2.0,
    "vip_convoy": 1.5,
    "protest_bandh": 3.0,
    "waterlogging": 3.5,
    "road_closure": 3.0,
    "clear": 0.0,
}


@dataclass
class DeploymentRecommendation:
    """Officer deployment recommendation for a congested segment."""

    segment_id: str
    junction: str
    crs: float
    officers: int
    deploy_by: str
    priority: int
    station_id: str


def calc_economic_impact(segment_id: str, crs: float, duration_hours: float, avg_volume: float = 2500) -> float:
    """Estimate congestion economic impact in INR."""
    delay_per_vehicle_min = (crs / 10.0) * 30.0
    bengaluru_avg_salary_per_min = 4.50
    return avg_volume * duration_hours * delay_per_vehicle_min * bengaluru_avg_salary_per_min


def _latest_from_csv() -> pd.DataFrame:
    """Build latest-per-segment snapshot from traffic CSV."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df = df.rename(columns={"speed_kmh": "speed_kmh"})
    if "congestion_risk_score" not in df.columns:
        df["congestion_risk_score"] = (1 - df["speed_kmh"] / 45.0).clip(0, 1) * 10
    return df.sort_values("timestamp").groupby(SEGMENT_ID_COL).tail(1)


def get_historical_volume(segment_id: str) -> float:
    """Return average hourly volume for a segment from feature matrix."""
    if FEATURE_MATRIX_PATH.exists():
        try:
            df = pd.read_parquet(FEATURE_MATRIX_PATH)
            sub = df[df[SEGMENT_ID_COL] == segment_id]
            return float(sub["volume"].mean()) if not sub.empty else 2500.0
        except Exception:
            pass
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        sub = df[df[SEGMENT_ID_COL] == segment_id]
        return float(sub["volume"].mean()) if not sub.empty else 2500.0
    return 2500.0


def alert_level(crs: float) -> str:
    """Map CRS to RED/AMBER/GREEN."""
    if crs >= CRS_RED_THRESHOLD:
        return "RED"
    if crs >= CRS_AMBER_THRESHOLD:
        return "AMBER"
    return "GREEN"


def fuse_segment_crs(
    segment_id: str,
    tft_p10: float,
    tft_p50: float,
    tft_p90: float,
    lstm_out: dict[str, Any],
    nlp_incident: dict[str, Any] | None,
    gnn_crs: float,
    weather: dict[str, float],
    dist_to_incident_km: float = 99.0,
) -> dict[str, Any]:
    """Fuse all model outputs into final CRS for one segment."""
    crs_base = tft_p50
    anomaly_boost = 0.0
    if lstm_out.get("is_anomaly"):
        anomaly_boost = float(lstm_out.get("confidence", 0.5)) * 3.0

    nlp_boost = 0.0
    if nlp_incident and dist_to_incident_km < 3.0:
        inc_type = nlp_incident.get("incident_type", "clear")
        sev = SEVERITY_MAP.get(inc_type, 0.0)
        nlp_boost = sev * max(0.0, 1.0 - dist_to_incident_km / 3.0)

    weather_mult = 1.0
    rain_mm = weather.get("rain_mm", 0.0)
    visibility = weather.get("visibility", 10000.0)
    if rain_mm > 10:
        weather_mult = 1.3
    if visibility < 500:
        weather_mult *= 1.2

    crs_final = (0.5 * crs_base + 0.3 * gnn_crs) * weather_mult
    crs_final += anomaly_boost + nlp_boost
    crs_final = min(10.0, max(0.0, crs_final))

    return {
        "segment_id": segment_id,
        "crs_P50": round(crs_final, 2),
        "crs_P10": round(max(0, crs_final - (tft_p50 - tft_p10)), 2),
        "crs_P90": round(min(10, crs_final + (tft_p90 - tft_p50)), 2),
        "alert_level": alert_level(crs_final),
        "anomaly_boost": round(anomaly_boost, 2),
        "nlp_boost": round(nlp_boost, 2),
        "weather_mult": round(weather_mult, 2),
    }


def recommend_deployment(top_segments: list[dict], event_info: dict | None = None) -> list[DeploymentRecommendation]:
    """Recommend officer deployment for RED/AMBER segments."""
    recs: list[DeploymentRecommendation] = []
    for rank, seg in enumerate(top_segments, start=1):
        if seg.get("alert_level") not in ("RED", "AMBER"):
            continue
        crs = seg["crs_P50"]
        capacity = seg.get("road_capacity", 3000)
        officers = int(np.clip(np.ceil(crs * capacity / 500), 1, 8))
        if event_info and event_info.get("type") in ("ipl", "festival", "marathon"):
            deploy_by = event_info.get("datetime", datetime.utcnow().isoformat())
        else:
            deploy_by = datetime.utcnow().isoformat()
        recs.append(
            DeploymentRecommendation(
                segment_id=seg["segment_id"],
                junction=seg.get("segment_name", seg["segment_id"]),
                crs=crs,
                officers=officers,
                deploy_by=deploy_by,
                priority=rank,
                station_id=seg.get("station_id", "STN_CUB"),
            )
        )
    return recs


def run_fusion_cycle(
    tft_output: dict[str, dict[str, list[float]]] | None = None,
    lstm_output: dict[str, dict] | None = None,
    nlp_incidents: list[dict] | None = None,
    gnn_output: dict[str, float] | None = None,
    weather: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute one fusion cycle and return JSON-serializable output."""
    if weather is None:
        try:
            from api_services import weather_for_fusion

            weather = weather_for_fusion()
        except Exception as exc:
            logger.warning("Weather API unavailable (%s) — using defaults.", exc)
            weather = {"rain_mm": 0.0, "visibility": 10000.0, "wind_speed": 5.0}
    nlp_incidents = nlp_incidents or []
    nlp_primary = nlp_incidents[0] if nlp_incidents else None

    if FEATURE_MATRIX_PATH.exists():
        try:
            df = pd.read_parquet(FEATURE_MATRIX_PATH)
            latest = df.sort_values("timestamp").groupby(SEGMENT_ID_COL).tail(1)
        except Exception as exc:
            logger.warning("Parquet load failed (%s) — using CSV fallback.", exc)
            latest = _latest_from_csv()
    elif DATASET_PATH.exists():
        latest = _latest_from_csv()
    else:
        latest = pd.DataFrame()

    segments_out: list[dict] = []
    for _, row in latest.iterrows():
        seg = row[SEGMENT_ID_COL]
        tft = (tft_output or {}).get(seg, {})
        p10 = (tft.get("P10") or [4.0])[0] if isinstance(tft.get("P10"), list) else tft.get("P10", 4.0)
        p50 = (tft.get("P50") or [5.0])[0] if isinstance(tft.get("P50"), list) else tft.get("P50", 5.0)
        p90 = (tft.get("P90") or [6.0])[0] if isinstance(tft.get("P90"), list) else tft.get("P90", 6.0)
        lstm = (lstm_output or {}).get(seg, {"is_anomaly": False, "confidence": 0.0})
        gnn_crs = (gnn_output or {}).get(seg, p50)
        fused = fuse_segment_crs(seg, p10, p50, p90, lstm, nlp_primary, gnn_crs, weather)
        forecast = [round(min(10, p50 + i * 0.1), 1) for i in range(16)]
        fused["segment_name"] = row.get("segment_name", seg)
        fused["latitude"] = float(row.get("latitude", 12.97))
        fused["longitude"] = float(row.get("longitude", 77.59))
        fused["road_capacity"] = float(row.get("road_capacity", 3000))
        fused["forecast_16_steps"] = forecast
        fused["economic_impact_inr"] = int(
            calc_economic_impact(seg, fused["crs_P50"], 1.0, get_historical_volume(seg))
        )
        segments_out.append(fused)

    segments_out.sort(key=lambda s: s["crs_P50"], reverse=True)
    active_alerts = [
        {
            "segment_id": s["segment_id"],
            "alert_level": s["alert_level"],
            "crs": s["crs_P50"],
            "road_name": s.get("segment_name", s["segment_id"]),
        }
        for s in segments_out
        if s["alert_level"] in ("RED", "AMBER")
    ][:10]

    deployment = [asdict(r) for r in recommend_deployment(segments_out[:15])]
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "segments": segments_out,
        "active_alerts": active_alerts,
        "deployment_brief": deployment,
        "total_economic_impact_inr": sum(s["economic_impact_inr"] for s in segments_out),
    }
    return output


def save_fusion_output(output: dict[str, Any]) -> None:
    """Persist fusion JSON for dashboard consumption."""
    with open(FUSION_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved fusion output → %s", FUSION_OUTPUT_PATH)


def run_fusion_loop(interval_minutes: int = 15, callback: Callable[[dict], None] | None = None) -> None:
    """Background thread: run fusion every N minutes."""
    def _loop() -> None:
        while True:
            try:
                out = run_fusion_cycle()
                save_fusion_output(out)
                if callback:
                    callback(out)
            except Exception as exc:
                logger.error("Fusion cycle failed: %s", exc)
            time.sleep(interval_minutes * 60)

    thread = threading.Thread(target=_loop, daemon=True, name="fusion-loop")
    thread.start()
    logger.info("Fusion loop started (every %d min)", interval_minutes)


def main() -> None:
    """Run one fusion cycle with demo fallbacks."""
    if FUSION_OUTPUT_PATH.exists():
        with open(FUSION_OUTPUT_PATH, encoding="utf-8") as f:
            demo = json.load(f)
        logger.info("Loaded existing fusion_output.json for merge.")
    else:
        demo = None

    output = run_fusion_cycle()
    if demo and not FEATURE_MATRIX_PATH.exists():
        output = demo
    save_fusion_output(output)
    print(json.dumps({"timestamp": output["timestamp"], "segments": len(output["segments"]),
                      "total_impact_inr": output["total_economic_impact_inr"]}, indent=2))


if __name__ == "__main__":
    main()
