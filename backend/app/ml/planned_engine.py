"""Module A — Planned Event Forecasting Engine (TFT-inspired temporal model)."""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from app.ml.features import (
    build_event_features, compute_crs, extract_temporal_features,
    haversine_km, CAUSE_SEVERITY, PLANNED_CAUSE_MULTIPLIER,
)

HORIZON_STEPS = 16  # 4 hours at 15-min intervals
STEP_MINUTES = 15


class PlannedEventForecaster:
    """
    Temporal Fusion Transformer-inspired forecaster.
    Uses Gradient Boosting with temporal + event features for hackathon deployment.
    Full TFT architecture documented in PRD; this is the production inference layer.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[GradientBoostingRegressor] = None
        self.encoders: dict = {}
        self.feature_cols: list[str] = []
        self.is_trained = False

    def _build_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = build_event_features(df)
        planned = df[df["event_type"] == "planned"].copy()
        if planned.empty:
            planned = df.sample(min(500, len(df)), random_state=42)

        rows = []
        for _, ev in planned.iterrows():
            base_crs = compute_crs(
                ev["event_type"], ev.get("event_cause", "others"),
                ev.get("priority", "Low"), ev.get("corridor", "Non-corridor"),
                bool(ev.get("requires_road_closure", False)),
                int(ev.get("hour", 12) if pd.notna(ev.get("hour", 12)) else 12),
            )
            for step in range(HORIZON_STEPS):
                phase = step / HORIZON_STEPS
                if phase < 0.25:
                    phase_mult = 0.6 + phase * 1.6
                elif phase < 0.75:
                    phase_mult = 1.0
                else:
                    phase_mult = max(0.3, 1.0 - (phase - 0.75) * 2.8)
                crs = min(100, base_crs * phase_mult * (1 + 0.1 * math.sin(step * 0.5)))
                rows.append({
                    "event_cause": ev.get("event_cause", "others"),
                    "corridor": ev.get("corridor", "Non-corridor"),
                    "priority": ev.get("priority", "Low"),
                    "hour": ev.get("hour", 12),
                    "day_of_week": int(ev.get("day_of_week", 0) if pd.notna(ev.get("day_of_week", 0)) else 0),
                    "month": int(ev.get("month", 6) if pd.notna(ev.get("month", 6)) else 6),
                    "is_weekend": ev.get("is_weekend", 0),
                    "is_peak": ev.get("is_peak", 0),
                    "impact_radius": ev.get("impact_radius", 2.0),
                    "severity_score": ev.get("severity_score", 0.5),
                    "requires_closure": int(bool(ev.get("requires_road_closure", False))),
                    "forecast_step": step,
                    "phase_pre": int(step < 4),
                    "phase_during": int(4 <= step < 12),
                    "phase_post": int(step >= 12),
                    "target_crs": crs,
                })

        train_df = pd.DataFrame(rows)
        feature_cols = [c for c in train_df.columns if c != "target_crs"]
        self.feature_cols = feature_cols

        for col in ["event_cause", "corridor", "priority"]:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            self.encoders[col] = le

        return train_df[feature_cols], train_df["target_crs"]

    def train(self, df: pd.DataFrame) -> dict:
        X, y = self._build_training_data(df)
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.85, random_state=42,
        )
        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        smape = float(np.mean(2 * np.abs(preds - y) / (np.abs(preds) + np.abs(y) + 1e-8)) * 100)
        accuracy = max(0, 100 - smape)

        self.save()
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "smape": round(smape, 2), "accuracy_pct": round(accuracy, 1)}

    def save(self):
        if self.model:
            joblib.dump(self.model, self.models_dir / "planned_forecaster.joblib")
            joblib.dump({"encoders": self.encoders, "feature_cols": self.feature_cols},
                        self.models_dir / "planned_meta.joblib")

    def load(self) -> bool:
        model_path = self.models_dir / "planned_forecaster.joblib"
        meta_path = self.models_dir / "planned_meta.joblib"
        if model_path.exists() and meta_path.exists():
            self.model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            self.encoders = meta["encoders"]
            self.feature_cols = meta["feature_cols"]
            self.is_trained = True
            return True
        return False

    def predict_event(
        self,
        event_type: str = "planned",
        event_cause: str = "public_event",
        corridor: str = "CBD 2",
        priority: str = "High",
        requires_closure: bool = False,
        start_time: Optional[datetime] = None,
        latitude: float = 12.9788,
        longitude: float = 77.5995,
    ) -> list[dict]:
        start_time = start_time or datetime.utcnow()
        temporal = extract_temporal_features(start_time)

        if not self.is_trained:
            base = compute_crs(event_type, event_cause, priority, corridor, requires_closure, start_time.hour)
            return self._rule_based_forecast(base, start_time, latitude, longitude, corridor)

        forecasts = []
        for step in range(HORIZON_STEPS):
            row = {
                "event_cause": event_cause, "corridor": corridor, "priority": priority,
                "hour": start_time.hour, "day_of_week": start_time.weekday(),
                "month": start_time.month, "is_weekend": temporal["is_weekend"],
                "is_peak": temporal["is_peak_hour"],
                "impact_radius": 3.0, "severity_score": CAUSE_SEVERITY.get(event_cause, 0.5),
                "requires_closure": int(requires_closure),
                "forecast_step": step,
                "phase_pre": int(step < 4), "phase_during": int(4 <= step < 12),
                "phase_post": int(step >= 12),
            }
            for col in ["event_cause", "corridor", "priority"]:
                if col in self.encoders:
                    try:
                        row[col] = self.encoders[col].transform([str(row[col])])[0]
                    except ValueError:
                        row[col] = 0

            X = pd.DataFrame([{c: row[c] for c in self.feature_cols}])
            crs = float(np.clip(self.model.predict(X)[0], 0, 100))
            p10 = crs * 0.82
            p90 = min(100, crs * 1.18)
            target = start_time + timedelta(minutes=step * STEP_MINUTES)
            forecasts.append({
                "step": step,
                "target_time": target.isoformat(),
                "minutes_ahead": step * STEP_MINUTES,
                "crs_score": round(crs, 1),
                "crs_p10": round(p10, 1),
                "crs_p50": round(crs, 1),
                "crs_p90": round(p90, 1),
                "latitude": latitude,
                "longitude": longitude,
                "corridor": corridor,
            })
        return forecasts

    def _rule_based_forecast(self, base_crs, start_time, lat, lng, corridor) -> list[dict]:
        forecasts = []
        for step in range(HORIZON_STEPS):
            phase = step / HORIZON_STEPS
            mult = 0.7 + 0.6 * math.sin(phase * math.pi) if phase < 0.8 else max(0.2, 1 - phase)
            crs = min(100, base_crs * mult)
            target = start_time + timedelta(minutes=step * STEP_MINUTES)
            forecasts.append({
                "step": step,
                "target_time": target.isoformat(),
                "minutes_ahead": step * STEP_MINUTES,
                "crs_score": round(crs, 1),
                "crs_p10": round(crs * 0.82, 1),
                "crs_p50": round(crs, 1),
                "crs_p90": round(min(100, crs * 1.18), 1),
                "latitude": lat, "longitude": lng, "corridor": corridor,
            })
        return forecasts

    def get_feature_importance(self) -> list[dict]:
        if not self.model or not self.feature_cols:
            return []
        importances = self.model.feature_importances_
        return sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(self.feature_cols, importances)],
            key=lambda x: x["importance"], reverse=True,
        )
