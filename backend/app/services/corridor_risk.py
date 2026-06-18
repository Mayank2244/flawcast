"""Corridor Risk Index — aggregate live risk across Bengaluru corridors."""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.models.orm import Event, Alert


CORRIDOR_META = {
    "ORR East 1": {"capacity": 4500, "criticality": 0.95},
    "ORR East 2": {"capacity": 4000, "criticality": 0.92},
    "ORR North 1": {"capacity": 3800, "criticality": 0.90},
    "ORR North 2": {"capacity": 3600, "criticality": 0.88},
    "ORR West 1": {"capacity": 3500, "criticality": 0.85},
    "Mysore Road": {"capacity": 3500, "criticality": 0.82},
    "Bellary Road 1": {"capacity": 3200, "criticality": 0.80},
    "Bellary Road 2": {"capacity": 3000, "criticality": 0.78},
    "Hosur Road": {"capacity": 3000, "criticality": 0.85},
    "Bannerghata Road": {"capacity": 2800, "criticality": 0.75},
    "Tumkur Road": {"capacity": 3200, "criticality": 0.80},
    "Old Madras Road": {"capacity": 2900, "criticality": 0.78},
    "Magadi Road": {"capacity": 2600, "criticality": 0.72},
    "CBD 1": {"capacity": 2500, "criticality": 0.88},
    "CBD 2": {"capacity": 2800, "criticality": 0.92},
}


class CorridorRiskService:
    def compute_index(self, db: Session) -> list[dict]:
        event_counts = dict(
            db.query(Event.corridor, func.count(Event.id))
            .group_by(Event.corridor)
            .all()
        )

        alert_scores = dict(
            db.query(Event.corridor, func.avg(Alert.crs_score))
            .join(Alert, Alert.event_id == Event.id)
            .filter(Alert.status == "active")
            .group_by(Event.corridor)
            .all()
        )

        corridors = set(list(event_counts.keys()) + list(CORRIDOR_META.keys()))
        results = []

        for corridor in corridors:
            if not corridor or corridor == "NULL":
                continue
            meta = CORRIDOR_META.get(corridor, {"capacity": 1500, "criticality": 0.5})
            count = event_counts.get(corridor, 0)
            avg_crs = float(alert_scores.get(corridor) or 0)

            frequency_factor = min(1.0, count / 500)
            risk_index = min(100, (
                avg_crs * 0.5 +
                frequency_factor * 30 +
                meta["criticality"] * 20
            ))

            status = "CRITICAL" if risk_index >= 70 else ("ELEVATED" if risk_index >= 45 else "NORMAL")

            results.append({
                "corridor": corridor,
                "risk_index": round(risk_index, 1),
                "status": status,
                "event_count": count,
                "active_avg_crs": round(avg_crs, 1),
                "capacity_vph": meta["capacity"],
                "criticality": meta["criticality"],
            })

        return sorted(results, key=lambda x: -x["risk_index"])
