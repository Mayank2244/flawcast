"""Module B — Unplanned Incident Detection (LSTM Autoencoder-inspired)."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.ml.features import build_event_features, compute_crs, CAUSE_SEVERITY

INCIDENT_ETA = {
    "accident": 90, "vehicle_breakdown": 45, "water_logging": 120,
    "tree_fall": 60, "pot_holes": 30, "construction": 180,
    "congestion": 75, "others": 40, "vip_movement": 30,
    "protest": 150, "road_conditions": 60,
}


class UnplannedAnomalyDetector:
    """
    LSTM Autoencoder-inspired anomaly detector.
    Uses Isolation Forest on temporal traffic pattern features derived from events.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.threshold = -0.15
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = build_event_features(df)
        unplanned = df[df["event_type"] == "unplanned"].copy()
        if unplanned.empty:
            unplanned = df.copy()

        features = pd.DataFrame({
            "hour": unplanned["hour"],
            "day_of_week": unplanned["day_of_week"],
            "severity_score": unplanned["severity_score"],
            "impact_radius": unplanned["impact_radius"],
            "duration_min": unplanned["duration_min"].clip(0, 300),
            "is_peak": unplanned["is_peak"],
            "is_weekend": unplanned["is_weekend"],
            "requires_closure": unplanned["requires_road_closure"].astype(int),
            "lat": unplanned["latitude"],
            "lng": unplanned["longitude"],
        }).fillna(0)
        return features, unplanned

    def train(self, df: pd.DataFrame) -> dict:
        X, events = self._build_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=200, contamination=0.08,
            random_state=42, n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled)
        anomaly_rate = float((preds == -1).mean() * 100)
        self.threshold = float(np.percentile(scores, 8))

        joblib.dump(self.model, self.models_dir / "anomaly_detector.joblib")
        joblib.dump(self.scaler, self.models_dir / "anomaly_scaler.joblib")
        joblib.dump(self.threshold, self.models_dir / "anomaly_threshold.joblib")

        return {
            "anomaly_rate_pct": round(anomaly_rate, 1),
            "threshold": round(self.threshold, 4),
            "samples_trained": len(X),
        }

    def load(self) -> bool:
        model_path = self.models_dir / "anomaly_detector.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(self.models_dir / "anomaly_scaler.joblib")
            self.threshold = joblib.load(self.models_dir / "anomaly_threshold.joblib")
            self.is_trained = True
            return True
        return False

    def detect(
        self,
        hour: int,
        day_of_week: int,
        event_cause: str,
        corridor: str,
        priority: str,
        latitude: float,
        longitude: float,
        requires_closure: bool = False,
        description: str = "",
    ) -> dict:
        severity = CAUSE_SEVERITY.get(event_cause, 0.4)
        impact = 2.0 if corridor == "Non-corridor" else 4.0

        features = np.array([[
            hour, day_of_week, severity, impact, 45.0,
            int(hour in (8, 9, 17, 18, 19, 20)),
            int(day_of_week >= 5), int(requires_closure),
            latitude, longitude,
        ]])

        is_anomaly = False
        anomaly_score = 0.0
        if self.is_trained and self.model:
            scaled = self.scaler.transform(features)
            anomaly_score = float(self.model.decision_function(scaled)[0])
            is_anomaly = anomaly_score < self.threshold
        else:
            is_anomaly = severity > 0.6 or priority == "High"
            anomaly_score = -severity

        crs = compute_crs("unplanned", event_cause, priority, corridor, requires_closure, hour)
        eta = INCIDENT_ETA.get(event_cause, 45)
        radius = 1.5 + severity * 3.0

        alert_type = "RED" if crs >= 70 else ("AMBER" if crs >= 45 else "GREEN")
        severity_level = min(5, max(1, int(crs / 20)))

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "reconstruction_error": round(max(0, -anomaly_score) * 100, 2),
            "crs_score": round(crs, 1),
            "alert_type": alert_type,
            "severity": severity_level,
            "incident_type": event_cause,
            "affected_radius_km": round(radius, 1),
            "eta_clear_min": eta,
            "title": f"{event_cause.replace('_', ' ').title()} detected on {corridor or 'road'}",
            "description": description or f"Anomaly detected with CRS {crs:.0f}/100",
        }
