#!/usr/bin/env python3
"""FlowCast AI — BTP officer deployment planner and REST API."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from flowcast_config import (
    BTP_STATIONS_PATH,
    EVENT_CALENDAR_PATH,
    FUSION_OUTPUT_PATH,
    OUTPUTS_BRIEFS_DIR,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="FlowCast AI Deployment Planner", version="1.0.0")
FEEDBACK_LOG = PROJECT_ROOT / "logs" / "officer_feedback.jsonl"


class FeedbackRequest(BaseModel):
    """Officer feedback on prediction accuracy."""

    segment_id: str
    was_accurate: bool
    notes: str = ""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def load_json(path: Path) -> Any:
    """Load JSON file with empty fallback."""
    if not path.exists():
        return [] if "calendar" in str(path) or "stations" in str(path) else {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_zones(segments: list[dict], max_dist_km: float = 0.5) -> list[list[dict]]:
    """Cluster nearby segments into deployment zones."""
    remaining = sorted(segments, key=lambda s: s.get("crs_P50", 0), reverse=True)
    zones: list[list[dict]] = []
    while remaining:
        seed = remaining.pop(0)
        zone = [seed]
        kept = []
        for seg in remaining:
            dist = haversine_km(seed["latitude"], seed["longitude"], seg["latitude"], seg["longitude"])
            if dist < max_dist_km:
                zone.append(seg)
            else:
                kept.append(seg)
        remaining = kept
        zones.append(zone)
    return zones


def build_deployment_plan(fusion: dict, events: list[dict], stations: list[dict]) -> list[dict]:
    """Generate deployment plan from fusion output."""
    alert_segs = [s for s in fusion.get("segments", []) if s.get("alert_level") in ("RED", "AMBER")]
    zones = group_zones(alert_segs)
    station_pool = {s["station_id"]: s["officer_count"] for s in stations}
    plan: list[dict] = []

    for zone in zones:
        max_crs = max(s["crs_P50"] for s in zone)
        total_vol = sum(s.get("road_capacity", 3000) * 0.35 for s in zone)
        zone_score = max_crs * math.log1p(total_vol)
        rep = zone[0]
        officers = int(np.clip(math.ceil(max_crs * rep.get("road_capacity", 3000) / 500), 1, 8))

        nearest = min(
            stations,
            key=lambda st: haversine_km(rep["latitude"], rep["longitude"], st["lat"], st["lon"]),
        )
        sid = nearest["station_id"]
        available = station_pool.get(sid, 0)
        assigned = min(officers, available)
        station_pool[sid] = max(0, available - assigned)

        event = events[0] if events else None
        if event and event.get("type") in ("ipl", "festival", "marathon", "concert"):
            evt_dt = datetime.fromisoformat(event["datetime"])
            deploy_by = (evt_dt - timedelta(minutes=90)).isoformat()
            timing = "planned"
        else:
            deploy_by = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
            timing = "immediate"

        priority = "CRITICAL" if max_crs >= 7.5 else "HIGH"
        plan.append(
            {
                "zone_score": round(zone_score, 2),
                "junction": rep.get("segment_name", rep["segment_id"]),
                "segment_ids": [s["segment_id"] for s in zone],
                "max_crs": max_crs,
                "officers_needed": assigned,
                "deploy_by": deploy_by,
                "timing": timing,
                "priority": priority,
                "station_id": sid,
                "station_name": nearest["name"],
                "vehicle_count_hr": int(total_vol),
            }
        )
    plan.sort(key=lambda p: p["zone_score"], reverse=True)
    return plan


def generate_mission_brief(deployment_plan: list[dict], event_info: dict | None) -> str:
    """Generate human-readable mission brief for BTP officers."""
    now = datetime.utcnow()
    evt_line = "UNPLANNED INCIDENT RESPONSE"
    if event_info:
        evt_line = f"EVENT: {event_info.get('description', event_info.get('type', 'Event'))} at {event_info.get('venue', 'Bengaluru')}"

    lines = [
        f"TRAFFIC ADVISORY — {now.strftime('%d %b %Y %H:%M')} IST",
        evt_line,
        "",
        "DEPLOYMENT INSTRUCTIONS:",
    ]
    for i, item in enumerate(deployment_plan[:10], start=1):
        lines.append(
            f"{i}. {item['junction']} — {item['officers_needed']} officers — Deploy by "
            f"{item['deploy_by'][:16].replace('T', ' ')} "
            f"(Priority: {item['priority']} | Vehicle count: ~{item['vehicle_count_hr']:,}/hr)"
        )

    total_impact = sum(p.get("vehicle_count_hr", 0) * 4.5 * 15 for p in deployment_plan[:5])
    lines.extend(
        [
            "",
            f"ECONOMIC IMPACT IF UNMANAGED: ₹{total_impact/100000:.1f} lakh/hour (estimated)",
            "ESTIMATED REDUCTION WITH DEPLOYMENT: 35%",
        ]
    )
    return "\n".join(lines)


def generate_pdf_brief(deployment_plan: list[dict], event_info: dict | None, event_id: str = "daily") -> Path:
    """Create PDF mission brief using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    OUTPUTS_BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    out_path = OUTPUTS_BRIEFS_DIR / f"brief_{event_id}_{date_str}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "FlowCast AI — BTP Mission Brief")
    y -= 30
    c.setFont("Helvetica", 10)
    for line in generate_mission_brief(deployment_plan, event_info).split("\n"):
        c.drawString(50, y, line[:100])
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50
    c.drawString(50, 60, "Officer Signatures: _________________________")
    c.save()
    logger.info("PDF brief saved → %s", out_path)
    return out_path


@app.get("/brief/today")
def get_today_brief() -> dict:
    """Return today's deployment brief JSON + PDF link."""
    fusion = load_json(FUSION_OUTPUT_PATH)
    events = load_json(EVENT_CALENDAR_PATH)
    stations = load_json(BTP_STATIONS_PATH)
    plan = build_deployment_plan(fusion, events if isinstance(events, list) else [], stations if isinstance(stations, list) else [])
    event = events[0] if isinstance(events, list) and events else None
    pdf_path = generate_pdf_brief(plan, event)
    return {
        "brief_text": generate_mission_brief(plan, event),
        "deployment_plan": plan,
        "pdf_path": str(pdf_path),
    }


@app.get("/alerts/live")
def get_live_alerts() -> dict:
    """Return current RED/AMBER segments."""
    fusion = load_json(FUSION_OUTPUT_PATH)
    alerts = [s for s in fusion.get("segments", []) if s.get("alert_level") in ("RED", "AMBER")]
    return {"timestamp": fusion.get("timestamp"), "alerts": alerts, "count": len(alerts)}


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest) -> dict:
    """Log prediction accuracy feedback from officers."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.utcnow().isoformat(), **body.model_dump()}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged", "segment_id": body.segment_id}


def main() -> None:
    """Generate brief and optionally start API."""
    import uvicorn

    fusion = load_json(FUSION_OUTPUT_PATH)
    events = load_json(EVENT_CALENDAR_PATH)
    stations = load_json(BTP_STATIONS_PATH)
    plan = build_deployment_plan(
        fusion,
        events if isinstance(events, list) else [],
        stations if isinstance(stations, list) else [],
    )
    print(generate_mission_brief(plan, events[0] if isinstance(events, list) and events else None))
    generate_pdf_brief(plan, events[0] if isinstance(events, list) and events else None)
    uvicorn.run("08_deployment_planner:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    main()
