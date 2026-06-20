#!/usr/bin/env python3
"""Bridge Astram incident dataset → traffic time-series format for ML pipeline."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flowcast_config import (
    ASTRAM_DATASET_PATH,
    EVENT_TYPE_COL,
    OCCUPANCY_COL,
    SEGMENT_ID_COL,
    SPEED_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
    VENUE_ID_COL,
)

# Corridor → segment metadata (lat, lon, capacity, lanes, road_type)
CORRIDOR_SEGMENTS: dict[str, dict] = {
    "Mysore Road": {"segment_id": "MYS001", "lat": 12.9446, "lon": 77.5274, "capacity": 3300, "lanes": 3, "road_type": "arterial"},
    "Bellary Road 1": {"segment_id": "BLR001", "lat": 13.0001, "lon": 77.5840, "capacity": 3600, "lanes": 3, "road_type": "highway"},
    "Bellary Road 2": {"segment_id": "BLR002", "lat": 13.0634, "lon": 77.5933, "capacity": 3400, "lanes": 3, "road_type": "highway"},
    "Tumkur Road": {"segment_id": "TUM001", "lat": 13.0374, "lon": 77.5181, "capacity": 3100, "lanes": 3, "road_type": "arterial"},
    "Hosur Road": {"segment_id": "HOS001", "lat": 12.9169, "lon": 77.6100, "capacity": 3400, "lanes": 3, "road_type": "arterial"},
    "ORR North 1": {"segment_id": "ORR003", "lat": 13.0354, "lon": 77.5947, "capacity": 4200, "lanes": 4, "road_type": "highway"},
    "ORR North 2": {"segment_id": "ORR004", "lat": 13.0447, "lon": 77.5829, "capacity": 4100, "lanes": 4, "road_type": "highway"},
    "ORR East 1": {"segment_id": "ORR_E1", "lat": 12.9465, "lon": 77.6987, "capacity": 4500, "lanes": 4, "road_type": "highway"},
    "ORR East 2": {"segment_id": "ORR002", "lat": 13.0008, "lon": 77.6814, "capacity": 4800, "lanes": 4, "road_type": "highway"},
    "ORR West 1": {"segment_id": "ORR_W1", "lat": 12.9446, "lon": 77.5274, "capacity": 4300, "lanes": 4, "road_type": "highway"},
    "Old Madras Road": {"segment_id": "OMR001", "lat": 12.9753, "lon": 77.6257, "capacity": 3200, "lanes": 3, "road_type": "arterial"},
    "Magadi Road": {"segment_id": "MAG001", "lat": 12.9789, "lon": 77.5644, "capacity": 2700, "lanes": 2, "road_type": "arterial"},
    "Bannerghata Road": {"segment_id": "BAN001", "lat": 12.9077, "lon": 77.6006, "capacity": 2900, "lanes": 2, "road_type": "arterial"},
    "West of Chord Road": {"segment_id": "WCR001", "lat": 12.9900, "lon": 77.5500, "capacity": 2800, "lanes": 2, "road_type": "arterial"},
    "CBD 1": {"segment_id": "CBD001", "lat": 12.9716, "lon": 77.5946, "capacity": 3000, "lanes": 3, "road_type": "arterial"},
    "CBD 2": {"segment_id": "CBD002", "lat": 12.9788, "lon": 77.5995, "capacity": 2800, "lanes": 3, "road_type": "arterial"},
    "Non-corridor": {"segment_id": "GEN001", "lat": 12.9716, "lon": 77.5946, "capacity": 2500, "lanes": 2, "road_type": "local"},
}

PRIORITY_SEVERITY = {"High": 0.75, "Medium": 0.5, "Low": 0.3}
CAUSE_SEVERITY = {
    "vehicle_breakdown": 0.45,
    "accident": 0.85,
    "water_logging": 0.7,
    "road_block": 0.8,
    "vip_movement": 0.6,
    "construction": 0.55,
}


def load_astram_events(path: Path | None = None) -> pd.DataFrame:
    """Load and clean Astram incident CSV."""
    path = path or ASTRAM_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Astram dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["event_cause"] = df["event_cause"].fillna("unknown")
    df["priority"] = df["priority"].fillna("Low")
    return df.dropna(subset=["start_datetime"])


def _event_duration_hours(row: pd.Series) -> float:
    end = row["end_datetime"] or row["resolved_datetime"]
    if pd.isna(end):
        return 1.5
    hours = (end - row["start_datetime"]).total_seconds() / 3600
    return float(np.clip(hours, 0.25, 8.0))


def _severity(row: pd.Series) -> float:
    base = PRIORITY_SEVERITY.get(str(row.get("priority", "Low")), 0.3)
    cause = str(row.get("event_cause", "unknown")).lower()
    for key, val in CAUSE_SEVERITY.items():
        if key in cause:
            return max(base, val)
    return base


def _build_corridor_intervals(events: pd.DataFrame) -> dict[str, list[tuple[datetime, datetime, float, str]]]:
    """Pre-compute incident intervals per corridor for fast lookup."""
    intervals: dict[str, list[tuple[datetime, datetime, float, str]]] = {}
    for _, ev in events.iterrows():
        corridor = str(ev.get("corridor", "Non-corridor"))
        start = ev["start_datetime"]
        end = ev["end_datetime"]
        if pd.isna(end):
            end = ev["resolved_datetime"]
        if pd.isna(end):
            end = start + timedelta(hours=_event_duration_hours(ev))
        sev = _severity(ev)
        etype = str(ev.get("event_type", "unplanned"))
        intervals.setdefault(corridor, []).append((start, end, sev, etype))
    return intervals


def _max_active_severity(
    intervals: list[tuple[datetime, datetime, float, str]], ts: datetime | pd.Timestamp
) -> tuple[float, str]:
    """Return max severity and event type active at timestamp."""
    ts = pd.Timestamp(ts)
    best, etype = 0.0, "none"
    for start, end, sev, et in intervals:
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end) and sev > best:
            best, etype = sev, et
    return best, etype


def astram_to_traffic_timeseries(
    astram_df: pd.DataFrame | None = None,
    freq_minutes: int = 15,
    max_days: int = 60,
) -> pd.DataFrame:
    """Convert Astram incidents into 15-min traffic telemetry per corridor segment.

    Generates traffic only on days with recorded incidents (plus padding) to keep
    the matrix tractable for EDA and TFT training.
    """
    events = astram_df if astram_df is not None else load_astram_events()
    if events.empty:
        raise ValueError("No Astram events to convert.")

    rng = np.random.default_rng(42)
    corridor_intervals = _build_corridor_intervals(events)
    event_days = events["start_datetime"].dt.floor("D").drop_duplicates().sort_values()
    if len(event_days) > max_days:
        event_days = event_days[-max_days:]

    timestamps = pd.date_range(
        event_days.iloc[0] - timedelta(hours=6),
        event_days.iloc[-1] + timedelta(days=1, hours=6),
        freq=f"{freq_minutes}min",
    )

    meta_items = list(CORRIDOR_SEGMENTS.items())
    n_ts, n_seg = len(timestamps), len(meta_items)
    speed_arr = np.zeros((n_ts, n_seg))
    volume_arr = np.zeros((n_ts, n_seg))
    occ_arr = np.zeros((n_ts, n_seg))
    etype_arr = np.full((n_ts, n_seg), "none", dtype=object)

    for j, (corridor, meta) in enumerate(meta_items):
        capacity = meta["capacity"]
        base_speed = 42.0 if meta["road_type"] in ("highway", "arterial") else 32.0
        ivals = corridor_intervals.get(corridor, [])
        for i, ts in enumerate(timestamps):
            hour, dow = ts.hour, ts.dayofweek
            peak = 1.6 if hour in (8, 9, 17, 18, 19, 20) else 1.0
            if dow >= 5:
                peak *= 0.88
            speed = base_speed * rng.uniform(0.88, 1.05) * (peak ** -0.1)
            volume = capacity * 0.32 * peak * rng.uniform(0.85, 1.15)
            occ = min(0.92, volume / capacity)
            sev, et = _max_active_severity(ivals, ts)
            if sev > 0:
                speed *= max(0.12, 1 - sev * 0.85)
                volume *= 1 + sev * 1.1
                occ = min(0.98, occ + sev * 0.25)
                etype_arr[i, j] = et if et != "none" else "unplanned"
            speed_arr[i, j] = max(5.0, speed)
            volume_arr[i, j] = max(50.0, volume)
            occ_arr[i, j] = occ

    rows: list[dict] = []
    for i, ts in enumerate(timestamps):
        for j, (corridor, meta) in enumerate(meta_items):
            rows.append(
                {
                    TIMESTAMP_COL: ts,
                    SEGMENT_ID_COL: meta["segment_id"],
                    "segment_name": corridor,
                    "latitude": meta["lat"],
                    "longitude": meta["lon"],
                    "road_capacity": meta["capacity"],
                    "lane_count": meta["lanes"],
                    "road_type": meta["road_type"],
                    SPEED_COL: round(float(speed_arr[i, j]), 2),
                    VOLUME_COL: round(float(volume_arr[i, j]), 0),
                    OCCUPANCY_COL: round(float(occ_arr[i, j]), 3),
                    EVENT_TYPE_COL: str(etype_arr[i, j]),
                    VENUE_ID_COL: "NONE",
                }
            )

    return pd.DataFrame(rows)


def ensure_traffic_from_astram(output_path: Path) -> Path:
    """Build traffic CSV from Astram if missing; return path used."""
    if output_path.exists():
        return output_path
    print(f"Building traffic telemetry from Astram → {output_path}")
    df = astram_to_traffic_timeseries()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
