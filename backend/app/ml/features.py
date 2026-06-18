"""Feature engineering for FlowCast AI ML pipeline."""
import math
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Bengaluru corridor impact radii (km) from PRD
CORRIDOR_IMPACT = {
    "ORR East 1": 5.0, "ORR East 2": 5.0, "ORR West 1": 5.0,
    "ORR North 1": 5.0, "ORR North 2": 5.0,
    "Mysore Road": 4.0, "Bellary Road 1": 4.0, "Bellary Road 2": 4.0,
    "Hosur Road": 4.0, "Bannerghata Road": 4.0, "Tumkur Road": 4.0,
    "Old Madras Road": 4.0, "Magadi Road": 3.5, "West of Chord Road": 3.5,
    "CBD 1": 3.0, "CBD 2": 3.0, "Non-corridor": 1.5,
}

CAUSE_SEVERITY = {
    "accident": 0.95, "vip_movement": 0.90, "protest": 0.85,
    "water_logging": 0.80, "construction": 0.75, "public_event": 0.85,
    "procession": 0.90, "vehicle_breakdown": 0.55, "tree_fall": 0.70,
    "pot_holes": 0.45, "congestion": 0.65, "road_conditions": 0.50,
    "others": 0.40,
}

PLANNED_CAUSE_MULTIPLIER = {
    "public_event": 1.8, "construction": 1.4, "procession": 1.6,
    "vip_movement": 1.5,
}


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def extract_temporal_features(dt: datetime) -> dict:
    hour_sin, hour_cos = cyclic_encode(dt.hour + dt.minute / 60, 24)
    dow_sin, dow_cos = cyclic_encode(dt.weekday(), 7)
    month_sin, month_cos = cyclic_encode(dt.month, 12)
    is_weekend = 1 if dt.weekday() >= 5 else 0
    is_peak = 1 if dt.hour in (8, 9, 17, 18, 19, 20) else 0
    is_monsoon = 1 if dt.month in (6, 7, 8, 9) else 0
    return {
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "is_weekend": is_weekend, "is_peak_hour": is_peak, "is_monsoon": is_monsoon,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_crs(
    event_type: str,
    event_cause: str,
    priority: str,
    corridor: str,
    requires_closure: bool,
    hour: int,
    distance_km: float = 0.0,
) -> float:
    """Compute Congestion Risk Score (0-100) using PRD-weighted formula."""
    base = CAUSE_SEVERITY.get(event_cause or "others", 0.4)
    if event_type == "planned":
        base *= PLANNED_CAUSE_MULTIPLIER.get(event_cause or "", 1.2)
    if priority == "High":
        base *= 1.3
    elif priority == "Low":
        base *= 0.7
    if requires_closure:
        base *= 1.4
    if hour in (8, 9, 17, 18, 19, 20):
        base *= 1.25
    impact_radius = CORRIDOR_IMPACT.get(corridor or "Non-corridor", 1.5)
    if distance_km > 0:
        decay = max(0.1, 1 - (distance_km / impact_radius))
        base *= decay
    return min(100.0, base * 100)


def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw event dataframe."""
    df = df.copy()
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], utc=True, errors="coerce")
    df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], utc=True, errors="coerce")

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hour"] = df["start_datetime"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["start_datetime"].dt.dayofweek.fillna(0).astype(int)
    df["month"] = df["start_datetime"].dt.month.fillna(6).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak"] = df["hour"].isin([8, 9, 17, 18, 19, 20]).astype(int)
    df["requires_road_closure"] = df["requires_road_closure"].map(
        {"TRUE": True, "FALSE": False, True: True, False: False}
    ).fillna(False)

    df["severity_score"] = df.apply(
        lambda r: CAUSE_SEVERITY.get(r.get("event_cause", "others"), 0.4), axis=1
    )
    df["impact_radius"] = df["corridor"].map(CORRIDOR_IMPACT).fillna(1.5)

    df["duration_min"] = df.apply(
        lambda r: (
            (r["end_datetime"] - r["start_datetime"]).total_seconds() / 60
            if pd.notna(r["end_datetime"]) and pd.notna(r["start_datetime"])
            else (
                (r["resolved_datetime"] - r["start_datetime"]).total_seconds() / 60
                if pd.notna(r["resolved_datetime"]) and pd.notna(r["start_datetime"])
                else 45.0
            )
        ),
        axis=1,
    )
    return df
