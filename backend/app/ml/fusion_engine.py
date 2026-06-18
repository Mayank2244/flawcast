"""Fusion & Scoring Layer — merges Module A + Module B outputs."""
from datetime import datetime
from typing import Optional

from app.ml.planned_engine import PlannedEventForecaster
from app.ml.unplanned_engine import UnplannedAnomalyDetector
from app.ml.nlp_engine import NLPIncidentClassifier
from app.ml.graph_engine import CongestionGraphEngine
from app.ml.features import compute_crs

WEIGHTS = {"planned": 0.45, "unplanned": 0.35, "nlp": 0.20}


class FusionEngine:
    """Unified congestion risk scoring across all prediction modules."""

    def __init__(self, models_dir: str = "models"):
        self.planned = PlannedEventForecaster(models_dir)
        self.unplanned = UnplannedAnomalyDetector(models_dir)
        self.nlp = NLPIncidentClassifier(models_dir)
        self.graph = CongestionGraphEngine()
        self._load_models()

    def _load_models(self):
        self.planned.load()
        self.unplanned.load()
        self.nlp.load()

    def analyze_event(
        self,
        event_type: str,
        event_cause: str,
        corridor: str,
        priority: str,
        latitude: float,
        longitude: float,
        description: str = "",
        address: str = "",
        requires_closure: bool = False,
        start_time: Optional[datetime] = None,
    ) -> dict:
        start_time = start_time or datetime.utcnow()
        hour = start_time.hour
        dow = start_time.weekday()

        planned_forecast = self.planned.predict_event(
            event_type=event_type, event_cause=event_cause,
            corridor=corridor, priority=priority,
            requires_closure=requires_closure,
            start_time=start_time, latitude=latitude, longitude=longitude,
        )

        unplanned = self.unplanned.detect(
            hour=hour, day_of_week=dow, event_cause=event_cause,
            corridor=corridor, priority=priority,
            latitude=latitude, longitude=longitude,
            requires_closure=requires_closure, description=description,
        )

        nlp_result = self.nlp.classify(description, address)

        peak_crs = max(f["crs_score"] for f in planned_forecast) if planned_forecast else 0
        planned_score = peak_crs
        unplanned_score = unplanned["crs_score"]
        nlp_boost = 15 if nlp_result["confidence"] > 0.7 else 5
        nlp_score = min(100, unplanned_score + nlp_boost)

        if event_type == "planned":
            fusion_crs = (
                WEIGHTS["planned"] * planned_score +
                WEIGHTS["unplanned"] * unplanned_score +
                WEIGHTS["nlp"] * nlp_score
            )
        else:
            fusion_crs = (
                WEIGHTS["unplanned"] * unplanned_score +
                WEIGHTS["nlp"] * nlp_score +
                WEIGHTS["planned"] * planned_score * 0.3
            )

        fusion_crs = min(100, fusion_crs)
        alert_type = "RED" if fusion_crs >= 70 else ("AMBER" if fusion_crs >= 45 else "GREEN")

        propagation = self.graph.propagate(
            latitude, longitude, fusion_crs, corridor, horizon_minutes=120
        )

        return {
            "fusion_crs": round(fusion_crs, 1),
            "alert_type": alert_type,
            "confidence": {
                "p10": round(fusion_crs * 0.82, 1),
                "p50": round(fusion_crs, 1),
                "p90": round(min(100, fusion_crs * 1.18), 1),
            },
            "modules": {
                "planned": {"peak_crs": round(planned_score, 1), "forecast": planned_forecast},
                "unplanned": unplanned,
                "nlp": nlp_result,
            },
            "propagation": propagation,
            "feature_importance": self.planned.get_feature_importance()[:8],
            "graph_stats": self.graph.get_graph_stats(),
        }
