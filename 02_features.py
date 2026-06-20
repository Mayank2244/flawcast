#!/usr/bin/env python3
"""FlowCast AI — feature engineering pipeline for TFT training.

Reads cleaned traffic CSV, engineers temporal/lag/event/target features,
validates the matrix, performs chronological train/val/test split, and
writes parquet artifacts under ``data/``.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowcast_config import (
    DATASET_PATH,
    EVENT_CALENDAR_PATH,
    EVENT_TYPE_COL,
    EVENT_TYPES,
    FEATURE_MATRIX_PATH,
    OCCUPANCY_COL,
    RANDOM_SEED,
    SEGMENT_ID_COL,
    SPEED_COL,
    TEST_RATIO,
    TIMESTAMP_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    VENUE_ID_COL,
    VOLUME_COL,
)

# 15-minute sampling: 4 rows = 1 hour
ROWS_PER_HOUR: int = 4
ROWS_6H: int = ROWS_PER_HOUR * 6
ROWS_24H: int = ROWS_PER_HOUR * 24
ROWS_168H: int = ROWS_PER_HOUR * 168

CRS_CONGESTED_THRESHOLD: float = 6.0
EVENT_PROXIMITY_KM: float = 5.0
EVENT_TIME_WINDOW_HOURS: float = 3.0

EVENT_PHASES: tuple[str, ...] = ("pre_event", "during_event", "post_event", "no_event")

# Bengaluru venue coordinates and max attendance for normalization
VENUE_META: dict[str, dict[str, float]] = {
    "CHINNASWAMY": {"lat": 12.9788, "lon": 77.5995, "max_attendance": 45_000.0},
    "PALACE_GROUNDS": {"lat": 13.0100, "lon": 77.5900, "max_attendance": 80_000.0},
    "BIAL": {"lat": 13.1986, "lon": 77.7066, "max_attendance": 20_000.0},
    "KORAMANGALA_STADIUM": {"lat": 12.9279, "lon": 77.6271, "max_attendance": 15_000.0},
    "NONE": {"lat": 12.9716, "lon": 77.5946, "max_attendance": 1.0},
}

# Hardcoded Indian public holidays (extendable list; covers demo window 2025)
INDIAN_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2025, 1, 26),  # Republic Day
        date(2025, 3, 14),  # Holi
        date(2025, 3, 31),  # Eid-ul-Fitr (approx.)
        date(2025, 4, 14),  # Ambedkar Jayanti / Tamil New Year
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 1),   # Labour Day
        date(2025, 8, 15),  # Independence Day
        date(2025, 8, 27),  # Ganesh Chaturthi
        date(2025, 10, 2),  # Gandhi Jayanti
        date(2025, 10, 20), # Diwali
        date(2025, 12, 25), # Christmas
    }
)

# Default segment static metadata when CSV lacks TFT static columns
DEFAULT_SEGMENT_STATIC: dict[str, dict[str, Any]] = {
    "road_type": "arterial",
    "road_capacity": 3000.0,
    "lane_count": 3,
    "latitude": 12.9716,
    "longitude": 77.5946,
}


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Return sin/cos cyclic encoding for a periodic value.

    Args:
        value: Raw periodic value (e.g. hour 0–23).
        period: Cycle length (e.g. 24 for hours).

    Returns:
        Tuple of (sin_component, cos_component).
    """
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two WGS84 points in kilometres.

    Args:
        lat1: Latitude of point A (degrees).
        lon1: Longitude of point A (degrees).
        lat2: Latitude of point B (degrees).
        lon2: Longitude of point B (degrees).

    Returns:
        Distance in kilometres.
    """
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return earth_radius_km * 2.0 * math.asin(math.sqrt(a))


def load_traffic_dataset(path: Path) -> pd.DataFrame:
    """Load and sort the raw traffic CSV.

    Args:
        path: Path to ``traffic_data.csv``.

    Returns:
        Sorted dataframe with parsed timestamps.
    """
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values([SEGMENT_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df


def ensure_static_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Ensure TFT static/real columns exist on every row.

    Uses CSV values when present; otherwise fills segment-level defaults and
    assigns a deterministic degree-proxy centrality score per segment.

    Args:
        df: Input traffic dataframe.
        rng: NumPy random generator (used only if lat/lon missing).

    Returns:
        Dataframe with static TFT columns populated.
    """
    out = df.copy()
    static_cols = [
        "road_type",
        "road_capacity",
        "lane_count",
        "latitude",
        "longitude",
        "segment_centrality",
    ]

    for col, default in DEFAULT_SEGMENT_STATIC.items():
        if col not in out.columns:
            out[col] = default

    # Per-segment fill for missing static fields
    segment_defaults = (
        out.groupby(SEGMENT_ID_COL)
        .agg(
            road_type=("road_type", "first"),
            road_capacity=("road_capacity", "first"),
            lane_count=("lane_count", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .reset_index()
    )

    out = out.drop(columns=[c for c in static_cols if c in out.columns], errors="ignore")
    out = out.merge(segment_defaults, on=SEGMENT_ID_COL, how="left")

    for col, default in DEFAULT_SEGMENT_STATIC.items():
        if col in ("road_type",):
            out[col] = out[col].fillna(default).astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    # Degree-proxy centrality: higher for highway/junction/venue segments
    type_weight = {
        "highway": 0.85,
        "junction": 0.90,
        "venue": 0.95,
        "arterial": 0.70,
        "local": 0.45,
    }
    out["segment_centrality"] = out["road_type"].map(type_weight).fillna(0.55)
    noise = rng.normal(0.0, 0.05, size=len(out))
    out["segment_centrality"] = np.clip(out["segment_centrality"] + noise, 0.1, 1.0)

    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time encodings and calendar indicator features.

    Args:
        df: Traffic dataframe with ``TIMESTAMP_COL``.

    Returns:
        Dataframe with temporal feature columns appended.
    """
    out = df.copy()
    ts = out[TIMESTAMP_COL]
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek.astype(float)
    month = ts.dt.month.astype(float)

    out["hour_sin"], out["hour_cos"] = zip(*[cyclic_encode(h, 24.0) for h in hour_frac])
    out["day_sin"], out["day_cos"] = zip(*[cyclic_encode(d, 7.0) for d in dow])
    out["month_sin"], out["month_cos"] = zip(*[cyclic_encode(m, 12.0) for m in month])

    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = ts.dt.date.map(lambda d: int(d in INDIAN_HOLIDAYS))
    out["is_monsoon_season"] = ts.dt.month.isin([6, 7, 8, 9]).astype(int)
    # School day: weekday, not holiday, outside summer break (Apr–May)
    out["is_school_day"] = (
        (ts.dt.dayofweek < 5)
        & (out["is_holiday"] == 0)
        & (~ts.dt.month.isin([4, 5]))
    ).astype(int)

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling statistics for speed, volume, and occupancy.

    Assumes 15-minute frequency (4 rows per hour). Computed per segment.

    Args:
        df: Dataframe sorted by segment and timestamp.

    Returns:
        Dataframe with lag/rolling feature columns.
    """
    out = df.copy()
    metrics = [SPEED_COL, VOLUME_COL, OCCUPANCY_COL]
    prefix_map = {SPEED_COL: "speed", VOLUME_COL: "volume", OCCUPANCY_COL: "occupancy"}

    grouped = out.groupby(SEGMENT_ID_COL, sort=False)

    for col in metrics:
        prefix = prefix_map[col]
        series = grouped[col]

        out[f"{prefix}_lag_1h"] = series.shift(ROWS_PER_HOUR)
        out[f"{prefix}_lag_24h"] = series.shift(ROWS_24H)
        out[f"{prefix}_lag_168h"] = series.shift(ROWS_168H)

        out[f"{prefix}_rolling_mean_1h"] = series.transform(
            lambda s: s.rolling(ROWS_PER_HOUR, min_periods=1).mean()
        )
        out[f"{prefix}_rolling_mean_6h"] = series.transform(
            lambda s: s.rolling(ROWS_6H, min_periods=1).mean()
        )
        out[f"{prefix}_rolling_mean_24h"] = series.transform(
            lambda s: s.rolling(ROWS_24H, min_periods=1).mean()
        )
        out[f"{prefix}_rolling_std_1h"] = series.transform(
            lambda s: s.rolling(ROWS_PER_HOUR, min_periods=1).std().fillna(0.0)
        )

    return out


def load_event_intervals(calendar_path: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build event start/end intervals from calendar JSON or traffic rows.

    Args:
        calendar_path: Path to ``event_calendar.json``.
        df: Traffic dataframe (fallback inference source).

    Returns:
        List of event dicts with keys: type, venue_id, start, end, attendance.
    """
    events: list[dict[str, Any]] = []

    if calendar_path.exists():
        with open(calendar_path, encoding="utf-8") as handle:
            calendar = json.load(handle)
        default_durations = {
            "ipl": 4.0,
            "festival": 8.0,
            "rally": 3.0,
            "concert": 4.0,
            "marathon": 3.0,
            "government": 2.0,
        }
        for item in calendar:
            start = pd.Timestamp(item["datetime"])
            duration_h = default_durations.get(item.get("type", "ipl"), 4.0)
            events.append(
                {
                    "type": item.get("type", "none"),
                    "venue_id": item.get("venue_id", "NONE"),
                    "start": start,
                    "end": start + timedelta(hours=duration_h),
                    "attendance": float(item.get("attendance", 0)),
                }
            )
        return events

    # Fallback: infer contiguous active event blocks from CSV labels
    active = df[df[EVENT_TYPE_COL].ne("none")].copy()
    if active.empty:
        return events

    for (event_type, venue_id), grp in active.groupby([EVENT_TYPE_COL, VENUE_ID_COL]):
        events.append(
            {
                "type": event_type,
                "venue_id": venue_id,
                "start": grp[TIMESTAMP_COL].min(),
                "end": grp[TIMESTAMP_COL].max() + timedelta(minutes=15),
                "attendance": float(len(grp) * 100),
            }
        )
    return events


def _nearest_event_context(
    ts: pd.Timestamp,
    lat: float,
    lon: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute event-relative features for one row timestamp and location.

    Args:
        ts: Row timestamp.
        lat: Segment latitude.
        lon: Segment longitude.
        events: Parsed event intervals.

    Returns:
        Dict with hours_to_event_start, hours_since_event_end, proximity,
        attendance norm, phase, and nearest event metadata.
    """
    best: dict[str, Any] = {
        "hours_to_event_start": 999.0,
        "hours_since_event_end": 0.0,
        "expected_attendance_normalized": 0.0,
        "venue_proximity_km": 999.0,
        "event_phase": "no_event",
        "nearest_event_type": "none",
        "nearest_venue_id": "NONE",
    }

    if not events:
        return best

    for event in events:
        venue_id = str(event.get("venue_id", "NONE"))
        venue = VENUE_META.get(venue_id, VENUE_META["NONE"])
        dist_km = haversine_km(lat, lon, venue["lat"], venue["lon"])

        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        hours_to_start = (start - ts).total_seconds() / 3600.0
        hours_since_end = (ts - end).total_seconds() / 3600.0

        if start <= ts <= end:
            phase = "during_event"
            h_start = -((ts - start).total_seconds() / 3600.0)
            h_end = 0.0
        elif ts < start:
            phase = "pre_event"
            h_start = hours_to_start
            h_end = 0.0
        else:
            phase = "post_event"
            h_start = 999.0  # no upcoming start for a finished event window
            h_end = max(0.0, hours_since_end)

        attendance_norm = min(
            1.0,
            float(event.get("attendance", 0)) / max(venue["max_attendance"], 1.0),
        )

        # Prefer geographically closer events; tie-break by temporal proximity
        temporal_dist = min(abs(hours_to_start), abs(hours_since_end))
        score = dist_km + temporal_dist * 0.05

        if score < best.get("_score", float("inf")):
            best = {
                "hours_to_event_start": h_start,
                "hours_since_event_end": h_end,
                "expected_attendance_normalized": attendance_norm,
                "venue_proximity_km": dist_km,
                "event_phase": phase,
                "nearest_event_type": event.get("type", "none"),
                "nearest_venue_id": venue_id,
                "_score": score,
            }

    best.pop("_score", None)
    return best


def add_event_features(df: pd.DataFrame, events: list[dict[str, Any]]) -> pd.DataFrame:
    """Add one-hot event types and event-relative temporal/spatial features.

    Args:
        df: Traffic dataframe with lat/lon columns.
        events: Parsed event intervals from :func:`load_event_intervals`.

    Returns:
        Dataframe with event feature columns.
    """
    out = df.copy()

    # One-hot encode observed event_type column
    for event_type in EVENT_TYPES:
        out[f"event_{event_type}"] = (out[EVENT_TYPE_COL] == event_type).astype(int)

    contexts = [
        _nearest_event_context(
            row[TIMESTAMP_COL],
            float(row["latitude"]),
            float(row["longitude"]),
            events,
        )
        for _, row in out.iterrows()
    ]
    ctx_df = pd.DataFrame(contexts)
    out = pd.concat([out.reset_index(drop=True), ctx_df], axis=1)

    # Categorical phase for TFT
    out["event_phase"] = pd.Categorical(out["event_phase"], categories=list(EVENT_PHASES))

    return out


def compute_historical_speed_by_hour(df: pd.DataFrame) -> pd.Series:
    """Compute segment×hour historical average speed for CRS speed ratio.

    Args:
        df: Traffic dataframe with speed and timestamp.

    Returns:
        Series indexed by (segment_id, hour) with mean speed.
    """
    tmp = df.copy()
    tmp["_hour"] = tmp[TIMESTAMP_COL].dt.hour
    return tmp.groupby([SEGMENT_ID_COL, "_hour"])[SPEED_COL].transform("mean")


def add_target_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CRS target and supervised label columns.

    CRS formula (FlowCast prompt):
        speed_ratio = current_speed / historical_avg_speed_for_hour
        volume_ratio = current_volume / road_capacity
        CRS = clip((1 - speed_ratio) * 0.6 + volume_ratio * 0.4, 0, 1) * 10

    Args:
        df: Feature-enriched dataframe.

    Returns:
        Dataframe with ``congestion_risk_score`` and label columns.
    """
    out = df.copy()
    hist_speed = compute_historical_speed_by_hour(out)
    speed_ratio = out[SPEED_COL] / hist_speed.replace(0, np.nan)
    speed_ratio = speed_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 3.0)

    volume_ratio = out[VOLUME_COL] / out["road_capacity"].replace(0, np.nan)
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 3.0)

    raw_crs = (1.0 - speed_ratio) * 0.6 + volume_ratio * 0.4
    out["congestion_risk_score"] = raw_crs.clip(0.0, 1.0) * 10.0

    # Lagged CRS for downstream TFT unknown reals
    grouped = out.groupby(SEGMENT_ID_COL, sort=False)["congestion_risk_score"]
    out["CRS_lag_1h"] = grouped.shift(ROWS_PER_HOUR)
    out["CRS_lag_24h"] = grouped.shift(ROWS_24H)

    out["is_congested"] = (out["congestion_risk_score"] > CRS_CONGESTED_THRESHOLD).astype(int)

    def severity_label(crs: float) -> str:
        if crs < 4.0:
            return "low"
        if crs < 7.0:
            return "medium"
        return "high"

    out["congestion_severity"] = out["congestion_risk_score"].map(severity_label)

    temporal_near_event = (
        (out["hours_to_event_start"].abs() <= EVENT_TIME_WINDOW_HOURS)
        | (out["hours_since_event_end"] <= EVENT_TIME_WINDOW_HOURS)
        | (out["event_phase"] == "during_event")
    )
    spatial_near_event = out["venue_proximity_km"] <= EVENT_PROXIMITY_KM
    out["is_event_related"] = (temporal_near_event & spatial_near_event).astype(int)

    return out


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaNs after lag warm-up using segment-wise medians.

    Args:
        df: Feature matrix with possible NaN from lags.

    Returns:
        Dataframe with no NaN values.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if out[col].isna().any():
            segment_median = out.groupby(SEGMENT_ID_COL)[col].transform("median")
            global_median = out[col].median()
            out[col] = out[col].fillna(segment_median).fillna(global_median).fillna(0.0)

    # Categorical / object columns
    for col in out.select_dtypes(include=["object", "category"]).columns:
        if out[col].isna().any():
            mode_vals = out[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "no_event"
            out[col] = out[col].fillna(fill_val)

    return out


def validate_feature_matrix(df: pd.DataFrame) -> None:
    """Print validation diagnostics and assert a NaN-free matrix.

    Args:
        df: Final feature matrix.

    Raises:
        AssertionError: If any NaN remains after imputation.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    missing_pct = (numeric_df.isna().sum() / len(df) * 100.0).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 5.0]

    print("\n=== Feature Validation ===")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    if high_missing.empty:
        print("Features with >5% missing: none")
    else:
        print("Features with >5% missing (>5%):")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.2f}%")

    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr(numeric_only=True).abs()
        corr_values = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        print("\nTop 20 feature correlations (|r|):")
        for (feat_a, feat_b), value in corr_values.head(20).items():
            print(f"  {feat_a} <-> {feat_b}: {value:.4f}")

    assert not df.isna().any().any(), "Feature matrix contains NaN after imputation"
    print("\nAssertion passed: no NaN in final feature matrix.")


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test (70/15/15).

    Test set is the latest 15% of timestamps (global holdout).

    Args:
        df: Full feature matrix sorted by time.

    Returns:
        Tuple of (train, val, test) dataframes.
    """
    sorted_df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = sorted_df.iloc[:train_end].copy()
    val = sorted_df.iloc[train_end:val_end].copy()
    test = sorted_df.iloc[val_end:].copy()

    print(
        f"\nTime split — train: {len(train):,} ({TRAIN_RATIO:.0%}), "
        f"val: {len(val):,} ({VAL_RATIO:.0%}), test: {len(test):,} ({TEST_RATIO:.0%})"
    )
    print(f"Train range: {train[TIMESTAMP_COL].min()} → {train[TIMESTAMP_COL].max()}")
    print(f"Val range:   {val[TIMESTAMP_COL].min()} → {val[TIMESTAMP_COL].max()}")
    print(f"Test range:  {test[TIMESTAMP_COL].min()} → {test[TIMESTAMP_COL].max()}")

    return train, val, test


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write dataframe to parquet, creating parent directories as needed.

    Args:
        df: Dataframe to persist.
        path: Output parquet path.

    Returns:
        Resolved output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.resolve()


def build_feature_matrix(df: pd.DataFrame, calendar_path: Path, rng: np.random.Generator) -> pd.DataFrame:
    """Run the full feature engineering pipeline on raw traffic data.

    Args:
        df: Raw traffic CSV dataframe.
        calendar_path: Path to event calendar JSON.
        rng: Random generator for static column noise.

    Returns:
        Complete feature matrix ready for modelling.
    """
    events = load_event_intervals(calendar_path, df)

    result = ensure_static_columns(df, rng)
    result = add_temporal_features(result)
    result = add_lag_features(result)
    result = add_event_features(result, events)
    result = add_target_and_labels(result)
    result = impute_missing_values(result)
    return result


def ensure_dataset_exists(dataset_path: Path) -> None:
    """Ensure traffic CSV exists — prefer Astram bridge, then demo generator.

    Args:
        dataset_path: Expected CSV location from ``flowcast_config``.
    """
    if dataset_path.exists():
        return

    from flowcast_config import ASTRAM_DATASET_PATH

    if ASTRAM_DATASET_PATH.exists():
        from astram_bridge import ensure_traffic_from_astram

        ensure_traffic_from_astram(dataset_path)
        return

    print(f"Dataset not found at {dataset_path}; generating demo data...")
    from generate_demo_data import main as generate_demo_main

    generate_demo_main()


def main() -> Path:
    """Load traffic data, engineer features, validate, split, and save parquet files.

    Returns:
        Absolute path to the saved ``feature_matrix.parquet``.
    """
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    ensure_dataset_exists(DATASET_PATH)

    print(f"Loading dataset: {DATASET_PATH}")
    raw = load_traffic_dataset(DATASET_PATH)
    print(f"Loaded {len(raw):,} rows across {raw[SEGMENT_ID_COL].nunique()} segments")

    features = build_feature_matrix(raw, EVENT_CALENDAR_PATH, rng)
    validate_feature_matrix(features)

    data_dir = FEATURE_MATRIX_PATH.parent
    matrix_path = save_parquet(features, FEATURE_MATRIX_PATH)
    print(f"\nSaved feature matrix: {matrix_path}")

    train, val, test = time_based_split(features)
    train_path = save_parquet(train, data_dir / "train_features.parquet")
    val_path = save_parquet(val, data_dir / "val_features.parquet")
    test_path = save_parquet(test, data_dir / "test_features.parquet")

    print(f"Saved train split: {train_path}")
    print(f"Saved val split:   {val_path}")
    print(f"Saved test split:  {test_path}")

    return matrix_path


if __name__ == "__main__":
    output_path = main()
    print(f"\nFeature engineering complete: {output_path}")
