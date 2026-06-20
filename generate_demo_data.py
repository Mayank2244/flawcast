#!/usr/bin/env python3
"""Generate realistic demo traffic data and supporting JSON for FlowCast AI."""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flowcast_config import (
    BTP_STATIONS_PATH,
    DATASET_PATH,
    EVENT_CALENDAR_PATH,
    FUSION_OUTPUT_PATH,
    PROJECT_ROOT,
    SEGMENT_ID_COL,
    SPEED_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
    OCCUPANCY_COL,
    EVENT_TYPE_COL,
    VENUE_ID_COL,
)

# 50 Bengaluru road segments with capacity metadata
SEGMENTS = [
    ("MG001", "MG Road", 12.9716, 77.5946, 3200, 4, "arterial"),
    ("MG002", "Brigade Road", 12.9698, 77.6070, 2800, 3, "arterial"),
    ("ORR001", "ORR Marathahalli", 12.9591, 77.6974, 4500, 4, "highway"),
    ("ORR002", "ORR Silk Board", 12.9173, 77.6229, 4800, 4, "highway"),
    ("ORR003", "ORR Hebbal", 13.0354, 77.5947, 4200, 4, "highway"),
    ("HSR001", "HSR 27th Main", 12.9116, 77.6388, 2600, 2, "local"),
    ("WHF001", "Whitefield ITPL", 12.9698, 77.7500, 3500, 3, "arterial"),
    ("KRP001", "KR Puram Bridge", 13.0008, 77.6814, 3800, 3, "arterial"),
    ("HBL001", "Hebbal Flyover", 13.0354, 77.5947, 4000, 4, "highway"),
    ("JNC001", "Silk Board Junction", 12.9173, 77.6229, 5000, 5, "junction"),
    ("JNC002", "Marathahalli Junction", 12.9591, 77.6974, 4600, 4, "junction"),
    ("CBD001", "Cubbon Park", 12.9788, 77.5995, 3000, 3, "arterial"),
    ("CBD002", "Residency Road", 12.9680, 77.6010, 2400, 2, "local"),
    ("BLR001", "Bellary Road", 13.0001, 77.5840, 3600, 3, "highway"),
    ("HOS001", "Hosur Road", 12.9169, 77.6100, 3400, 3, "arterial"),
    ("MYS001", "Mysore Road", 12.9446, 77.5274, 3300, 3, "arterial"),
    ("TUM001", "Tumkur Road", 13.0374, 77.5181, 3100, 3, "arterial"),
    ("BAN001", "Bannerghata Road", 12.9077, 77.6006, 2900, 2, "arterial"),
    ("EC001", "Electronic City", 12.8456, 77.6603, 3700, 3, "highway"),
    ("IND001", "Indiranagar 100ft", 12.9784, 77.6408, 2700, 2, "local"),
    ("KOR001", "Koramangala 80ft", 12.9279, 77.6271, 2500, 2, "local"),
    ("JAY001", "Jayanagar 4th Block", 12.9274, 77.5807, 2200, 2, "local"),
    ("BTM001", "BTM Layout", 12.9165, 77.6101, 2300, 2, "local"),
    ("YLK001", "Yelahanka", 13.1007, 77.5963, 2800, 2, "local"),
    ("RJN001", "Rajajinagar", 12.9915, 77.5545, 2600, 2, "local"),
    ("MLR001", "Malleshwaram", 13.0035, 77.5647, 2400, 2, "local"),
    ("VTH001", "Vijayanagar", 12.9710, 77.5370, 2500, 2, "local"),
    ("BAS001", "Basavanagudi", 12.9423, 77.5677, 2100, 2, "local"),
    ("FRA001", "Frazer Town", 12.9987, 77.6185, 2200, 2, "local"),
    ("DOM001", "Domlur", 12.9609, 77.6389, 2300, 2, "local"),
    ("AIR001", "Airport Road", 12.9498, 77.6682, 3400, 3, "highway"),
    ("SAR001", "Sarjapur Road", 12.9028, 77.6848, 3000, 2, "arterial"),
    ("NAG001", "Nagawara", 13.0450, 77.6190, 2600, 2, "local"),
    ("PEE001", "Peenya Industrial", 13.0374, 77.5181, 3200, 3, "arterial"),
    ("HAL001", "HAL Old Airport", 12.9635, 77.6642, 2800, 2, "local"),
    ("MAD001", "Madiwala", 12.9071, 77.6286, 2400, 2, "local"),
    ("WIL001", "Wilson Garden", 12.9539, 77.5852, 2200, 2, "local"),
    ("CHN001", "Chinnaswamy Stadium", 12.9788, 77.5995, 3500, 3, "venue"),
    ("PAL001", "Palace Grounds", 13.0100, 77.5900, 3000, 3, "venue"),
    ("KEN001", "Kengeri", 12.9060, 77.4870, 2600, 2, "local"),
    ("YPR001", "Yeshwanthpur", 13.0289, 77.5442, 2900, 2, "arterial"),
    ("RRN001", "RR Nagar", 12.9250, 77.5170, 2400, 2, "local"),
    ("GOP001", "Gopalan Mall Road", 12.9560, 77.7010, 2700, 2, "local"),
    ("LAL001", "Lalbagh Road", 12.9507, 77.5848, 2300, 2, "local"),
    ("ULS001", "Ulsoor Lake Road", 12.9830, 77.6220, 2100, 2, "local"),
    ("CVV001", "CV Raman Nagar", 12.9850, 77.6600, 2400, 2, "local"),
    ("BEL001", "Bellandur Lake Road", 12.9250, 77.6700, 3200, 2, "arterial"),
    ("KAD001", "Kadugodi", 12.9900, 77.7600, 2500, 2, "local"),
    ("DEV001", "Devanahalli", 13.2470, 77.7080, 2800, 2, "highway"),
    ("MAG001", "Magadi Road", 12.9789, 77.5644, 2700, 2, "arterial"),
]

VENUES = {
    "CHINNASWAMY": (12.9788, 77.5995, 45000),
    "PALACE_GROUNDS": (13.0100, 77.5900, 80000),
    "BIAL": (13.1986, 77.7066, 20000),
    "KORAMANGALA_STADIUM": (12.9279, 77.6271, 15000),
    "NONE": (12.9716, 77.5946, 0),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometres."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def generate_traffic_csv(days: int = 7, freq_minutes: int = 15) -> pd.DataFrame:
    """Build synthetic 15-minute traffic time series with planned events and incidents."""
    rng = np.random.default_rng(42)
    start = datetime(2025, 3, 10, 0, 0, 0)
    timestamps = pd.date_range(start, periods=days * 24 * (60 // freq_minutes), freq=f"{freq_minutes}min")

    planned_events = [
        {"type": "ipl", "venue": "CHINNASWAMY", "start": datetime(2025, 3, 12, 19, 0), "hours": 4, "attendance": 38000},
        {"type": "festival", "venue": "PALACE_GROUNDS", "start": datetime(2025, 3, 14, 10, 0), "hours": 8, "attendance": 65000},
        {"type": "marathon", "venue": "KORAMANGALA_STADIUM", "start": datetime(2025, 3, 15, 6, 0), "hours": 3, "attendance": 12000},
    ]
    incidents = [
        {"segment": "ORR002", "start": datetime(2025, 3, 11, 9, 15), "hours": 1.5, "severity": 0.7},
        {"segment": "MG001", "start": datetime(2025, 3, 11, 17, 30), "hours": 1.0, "severity": 0.5},
        {"segment": "HBL001", "start": datetime(2025, 3, 13, 8, 0), "hours": 2.0, "severity": 0.8},
        {"segment": "JNC001", "start": datetime(2025, 3, 13, 18, 45), "hours": 0.75, "severity": 0.6},
        {"segment": "AIR001", "start": datetime(2025, 3, 14, 14, 0), "hours": 1.25, "severity": 0.55},
    ]

    rows: list[dict] = []
    for ts in timestamps:
        hour = ts.hour
        dow = ts.weekday()
        peak_factor = 1.0
        if hour in (8, 9, 17, 18, 19, 20):
            peak_factor = 1.6
        if dow >= 5:
            peak_factor *= 0.85

        for seg_id, name, lat, lon, capacity, lanes, road_type in SEGMENTS:
            base_speed = 45.0 if road_type in ("highway", "arterial") else 35.0
            base_volume = capacity * 0.35 * peak_factor
            speed = base_speed * rng.uniform(0.85, 1.05)
            volume = base_volume * rng.uniform(0.8, 1.2)
            occupancy = min(0.95, volume / max(capacity, 1))

            event_type = "none"
            venue_id = "NONE"
            for ev in planned_events:
                ev_end = ev["start"] + timedelta(hours=ev["hours"])
                dist = haversine_km(lat, lon, *VENUES[ev["venue"]][:2])
                if dist < 8 and ev["start"] - timedelta(hours=2) <= ts <= ev_end + timedelta(hours=1):
                    event_type = ev["type"]
                    venue_id = ev["venue"]
                    impact = max(0.2, 1 - dist / 8)
                    if ev["start"] <= ts <= ev_end:
                        speed *= max(0.15, 1 - 0.7 * impact)
                        volume *= 1 + 1.2 * impact
                    elif ts < ev["start"]:
                        speed *= max(0.4, 1 - 0.3 * impact)
                        volume *= 1 + 0.4 * impact

            for inc in incidents:
                if seg_id == inc["segment"]:
                    inc_end = inc["start"] + timedelta(hours=inc["hours"])
                    if inc["start"] <= ts <= inc_end:
                        speed *= max(0.1, 1 - inc["severity"])
                        volume *= 1 + inc["severity"]
                        occupancy = min(0.98, occupancy + inc["severity"] * 0.3)

            rows.append(
                {
                    TIMESTAMP_COL: ts,
                    SEGMENT_ID_COL: seg_id,
                    "segment_name": name,
                    "latitude": lat,
                    "longitude": lon,
                    "road_capacity": capacity,
                    "lane_count": lanes,
                    "road_type": road_type,
                    SPEED_COL: round(max(5.0, speed), 2),
                    VOLUME_COL: round(max(50.0, volume), 0),
                    OCCUPANCY_COL: round(min(0.99, occupancy), 3),
                    EVENT_TYPE_COL: event_type,
                    VENUE_ID_COL: venue_id,
                }
            )

    return pd.DataFrame(rows)


def generate_event_calendar() -> list[dict]:
    """Create planned event calendar JSON."""
    return [
        {
            "event_id": "EVT_IPL_001",
            "type": "ipl",
            "venue": "Chinnaswamy Stadium",
            "venue_id": "CHINNASWAMY",
            "datetime": "2025-03-12T19:00:00",
            "attendance": 38000,
            "description": "RCB vs CSK — IPL Match",
        },
        {
            "event_id": "EVT_FEST_001",
            "type": "festival",
            "venue": "Palace Grounds",
            "venue_id": "PALACE_GROUNDS",
            "datetime": "2025-03-14T10:00:00",
            "attendance": 65000,
            "description": "Ugadi Festival Gathering",
        },
        {
            "event_id": "EVT_MAR_001",
            "type": "marathon",
            "venue": "Koramangala",
            "venue_id": "KORAMANGALA_STADIUM",
            "datetime": "2025-03-15T06:00:00",
            "attendance": 12000,
            "description": "Bengaluru City Marathon",
        },
    ]


def generate_btp_stations() -> list[dict]:
    """Create BTP station reference JSON."""
    stations = [
        ("STN_CUB", "Cubbon Park", 12.9788, 77.5995, 15, 3),
        ("STN_HSR", "HSR Layout", 12.9219, 77.6452, 12, 2),
        ("STN_HBL", "Hebbal", 13.0354, 77.5947, 14, 3),
        ("STN_JAY", "Jayanagara", 12.9274, 77.5807, 10, 2),
        ("STN_HAL", "HAL Old Airport", 12.9635, 77.6642, 11, 2),
        ("STN_WHT", "Whitefield", 12.9698, 77.7500, 13, 3),
        ("STN_MAD", "Madiwala", 12.9071, 77.6286, 10, 2),
        ("STN_WIL", "Wilson Garden", 12.9539, 77.5852, 11, 2),
        ("STN_KRP", "KR Puram", 13.0008, 77.6814, 12, 2),
        ("STN_ELC", "Electronic City", 12.8456, 77.6603, 8, 2),
    ]
    return [
        {
            "station_id": sid,
            "name": name,
            "lat": lat,
            "lon": lon,
            "officer_count": officers,
            "vehicles": vehicles,
        }
        for sid, name, lat, lon, officers, vehicles in stations
    ]


def generate_fusion_output() -> dict:
    """Pre-computed fusion output for dashboard demo mode."""
    now = datetime(2025, 3, 12, 18, 30, 0)
    segments = []
    for seg_id, name, lat, lon, capacity, _, _ in SEGMENTS[:20]:
        crs = round(random.uniform(3.0, 9.5), 1)
        if seg_id in ("MG001", "CBD001", "CHN001", "MG002"):
            crs = round(random.uniform(7.0, 9.2), 1)
        alert = "GREEN"
        if crs >= 7.5:
            alert = "RED"
        elif crs >= 5.0:
            alert = "AMBER"
        segments.append(
            {
                "segment_id": seg_id,
                "segment_name": name,
                "latitude": lat,
                "longitude": lon,
                "crs_P50": crs,
                "crs_P10": max(0, crs - 1.1),
                "crs_P90": min(10, crs + 1.0),
                "alert_level": alert,
                "forecast_16_steps": [round(min(10, crs + random.uniform(-0.5, 0.8)), 1) for _ in range(16)],
                "economic_impact_inr": int(capacity * (crs / 10) * 30 * 4.5),
            }
        )
    return {
        "timestamp": now.isoformat(),
        "segments": segments,
        "active_alerts": [
            {
                "segment_id": "MG001",
                "incident_type": "ipl",
                "alert_level": "RED",
                "road_name": "MG Road",
                "severity": "major",
                "detected_at": (now - timedelta(minutes=15)).isoformat(),
            }
        ],
        "deployment_brief": [],
        "total_economic_impact_inr": sum(s["economic_impact_inr"] for s in segments),
    }


def main() -> None:
    """Generate all demo artifacts (optionally seeded from Astram incidents)."""
    PROJECT_ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)

    # Prefer Astram-derived traffic when raw telemetry is absent
    if not DATASET_PATH.exists():
        try:
            from astram_bridge import ensure_traffic_from_astram

            ensure_traffic_from_astram(DATASET_PATH)
            print(f"Built traffic from Astram incidents: {DATASET_PATH}")
        except Exception as exc:
            print(f"Astram bridge unavailable ({exc}); using synthetic generator.")
            df = generate_traffic_csv()
            df.to_csv(DATASET_PATH, index=False)
            print(f"Saved traffic dataset: {DATASET_PATH} ({len(df):,} rows)")
    else:
        print(f"Traffic dataset already exists: {DATASET_PATH}")

    with open(EVENT_CALENDAR_PATH, "w", encoding="utf-8") as f:
        json.dump(generate_event_calendar(), f, indent=2)
    print(f"Saved event calendar: {EVENT_CALENDAR_PATH}")

    with open(BTP_STATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(generate_btp_stations(), f, indent=2)
    print(f"Saved BTP stations: {BTP_STATIONS_PATH}")

    with open(FUSION_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(generate_fusion_output(), f, indent=2)
    print(f"Saved fusion output: {FUSION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
