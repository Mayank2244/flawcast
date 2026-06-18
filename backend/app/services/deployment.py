"""BTP Officer Deployment Planner + Economic Impact Quantifier."""
import math
from datetime import datetime, timedelta
from typing import Optional

AVG_BENGALURU_SALARY_PER_HOUR = 350  # INR
AVG_VEHICLES_PER_CORRIDOR_HOUR = 2500
OFFICER_COVERAGE_MULTIPLIER = 12


class DeploymentPlanner:
    """Generates actionable BTP mission briefs from predictions."""

    STATIONS = [
        {"name": "Cubbon Park", "lat": 12.9788, "lng": 77.5995, "zone": "Central", "officers": 15},
        {"name": "HSR Layout", "lat": 12.9219, "lng": 77.6452, "zone": "South East", "officers": 12},
        {"name": "Hebbala", "lat": 13.0354, "lng": 77.5947, "zone": "North", "officers": 14},
        {"name": "Jayanagara", "lat": 12.9274, "lng": 77.5807, "zone": "South", "officers": 10},
        {"name": "HAL Old Airport", "lat": 12.9635, "lng": 77.6642, "zone": "East", "officers": 11},
        {"name": "Peenya", "lat": 13.0374, "lng": 77.5181, "zone": "West", "officers": 9},
        {"name": "Whitefield", "lat": 12.9698, "lng": 77.7500, "zone": "East", "officers": 13},
        {"name": "Madiwala", "lat": 12.9071, "lng": 77.6286, "zone": "South", "officers": 10},
    ]

    def _haversine(self, lat1, lng1, lat2, lng2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def find_nearest_station(self, lat: float, lng: float) -> dict:
        best = self.STATIONS[0]
        best_dist = self._haversine(lat, lng, best["lat"], best["lng"])
        for s in self.STATIONS[1:]:
            d = self._haversine(lat, lng, s["lat"], s["lng"])
            if d < best_dist:
                best, best_dist = s, d
        return {**best, "distance_km": round(best_dist, 2)}

    def generate_brief(
        self,
        fusion_crs: float,
        alert_type: str,
        event_cause: str,
        corridor: str,
        latitude: float,
        longitude: float,
        junction: str = "",
        event_type: str = "unplanned",
        start_time: Optional[datetime] = None,
        propagation: list = None,
    ) -> dict:
        start_time = start_time or datetime.utcnow()

        if fusion_crs >= 80:
            officers = 10
        elif fusion_crs >= 60:
            officers = 6
        elif fusion_crs >= 40:
            officers = 4
        else:
            officers = 2

        if event_type == "planned" and event_cause in ("public_event", "procession"):
            officers = min(16, officers + 4)

        deploy_lead = 90 if event_type == "planned" else 15
        deploy_by = start_time + timedelta(minutes=deploy_lead)

        station = self.find_nearest_station(latitude, longitude)
        available = min(officers, station["officers"])
        reduction_pct = min(65, 20 + available * 4 + fusion_crs * 0.2)

        primary = junction or corridor or "Nearest major junction"
        secondary_nodes = [p["node"] for p in (propagation or [])[:3]]
        secondary = secondary_nodes[1] if len(secondary_nodes) > 1 else f"{corridor} alternate route"

        economic = EconomicImpactCalculator()
        impact = economic.calculate(fusion_crs, corridor, event_cause)
        savings = impact["cost_inr"] * (reduction_pct / 100)

        brief_text = (
            f"MISSION BRIEF — {alert_type} ALERT\n"
            f"Incident: {event_cause.replace('_', ' ').title()} on {corridor}\n"
            f"Deploy {available} officers by {deploy_by.strftime('%H:%M')} "
            f"({deploy_lead} min lead time)\n"
            f"Primary junction: {primary}\n"
            f"Secondary: {secondary}\n"
            f"Nearest station: {station['name']} ({station['distance_km']} km)\n"
            f"Expected congestion reduction: {reduction_pct:.0f}%\n"
            f"Estimated savings: ₹{savings:,.0f}"
        )

        return {
            "title": f"{alert_type} — {event_cause.replace('_', ' ').title()} on {corridor}",
            "officers_needed": available,
            "deploy_by": deploy_by.isoformat(),
            "deploy_lead_minutes": deploy_lead,
            "primary_junction": primary,
            "secondary_junction": secondary,
            "station": station,
            "estimated_reduction_pct": round(reduction_pct, 1),
            "economic_savings_inr": round(savings, 0),
            "brief_text": brief_text,
        }


class EconomicImpactCalculator:
    """Quantifies rupee cost of congestion events per PRD formula."""

    CORRIDOR_VOLUME = {
        "ORR East 1": 4500, "ORR East 2": 4000, "ORR North 1": 3800,
        "Mysore Road": 3500, "Bellary Road 1": 3200, "Hosur Road": 3000,
        "Non-corridor": 1200,
    }

    def calculate(
        self,
        crs_score: float,
        corridor: str = "Non-corridor",
        event_cause: str = "others",
        duration_min: int = 60,
    ) -> dict:
        vehicles = self.CORRIDOR_VOLUME.get(corridor, 1500)
        affected_pct = crs_score / 100
        affected_vehicles = int(vehicles * affected_pct)
        delay_min = int(duration_min * affected_pct * 0.8 + 10)

        cost = affected_vehicles * delay_min * (AVG_BENGALURU_SALARY_PER_HOUR / 60)
        cost_crore = cost / 1e7

        return {
            "affected_vehicles": affected_vehicles,
            "delay_minutes": delay_min,
            "cost_inr": round(cost, 0),
            "cost_crore": round(cost_crore, 4),
            "cost_display": f"₹{cost_crore:.2f} Cr" if cost_crore >= 0.01 else f"₹{cost:,.0f}",
            "monthly_productivity_saved": round(cost * OFFICER_COVERAGE_MULTIPLIER * 30 / 1e7, 2),
        }

    def monthly_aggregate(self, events_cost: list[float]) -> dict:
        total = sum(events_cost)
        prevented = total * 0.35
        return {
            "total_cost_inr": round(total, 0),
            "prevented_cost_inr": round(prevented, 0),
            "total_crore": round(total / 1e7, 2),
            "prevented_crore": round(prevented / 1e7, 2),
            "productivity_saved_display": f"₹{prevented / 1e7:.2f} Cr/month",
        }
