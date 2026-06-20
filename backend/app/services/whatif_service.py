"""What-If Simulator — interactive CRS sensitivity analysis."""
import math
from datetime import datetime
from typing import Optional

from app.ml.features import compute_crs, CAUSE_SEVERITY, CORRIDOR_IMPACT


class WhatIfSimulator:
    """Lets users modify event parameters and see CRS change in real-time."""

    WEATHER_MULTIPLIERS = {
        "clear": 1.0,
        "light_rain": 1.15,
        "heavy_rain": 1.35,
        "monsoon": 1.50,
        "fog": 1.20,
    }

    TIME_PROFILES = {
        # hour: traffic intensity multiplier
        0: 0.3, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.35,
        6: 0.55, 7: 0.75, 8: 1.0, 9: 0.95, 10: 0.8, 11: 0.7,
        12: 0.75, 13: 0.7, 14: 0.65, 15: 0.7, 16: 0.8, 17: 0.95,
        18: 1.0, 19: 0.9, 20: 0.75, 21: 0.6, 22: 0.45, 23: 0.35,
    }

    def simulate(
        self,
        event_type: str = "planned",
        event_cause: str = "public_event",
        corridor: str = "CBD 2",
        priority: str = "High",
        requires_closure: bool = False,
        base_hour: int = 18,
        # What-If parameters
        attendance_multiplier: float = 1.0,
        weather: str = "clear",
        override_hour: Optional[int] = None,
        override_closure: Optional[bool] = None,
    ) -> dict:
        """Run simulation with modified parameters and return comparison."""

        hour = override_hour if override_hour is not None else base_hour
        closure = override_closure if override_closure is not None else requires_closure

        # Baseline CRS (original parameters)
        baseline_crs = compute_crs(
            event_type=event_type,
            event_cause=event_cause,
            priority=priority,
            corridor=corridor,
            requires_closure=requires_closure,
            hour=base_hour,
        )

        # Modified CRS
        modified_crs = compute_crs(
            event_type=event_type,
            event_cause=event_cause,
            priority=priority,
            corridor=corridor,
            requires_closure=closure,
            hour=hour,
        )

        # Apply attendance multiplier
        attendance_effect = 1.0 + (attendance_multiplier - 1.0) * 0.6
        modified_crs *= attendance_effect

        # Apply weather multiplier
        weather_mult = self.WEATHER_MULTIPLIERS.get(weather, 1.0)
        modified_crs *= weather_mult

        # Apply time profile
        time_mult = self.TIME_PROFILES.get(hour, 0.7)
        baseline_time_mult = self.TIME_PROFILES.get(base_hour, 0.7)
        if time_mult != baseline_time_mult:
            modified_crs *= (time_mult / baseline_time_mult)

        modified_crs = min(100, max(0, modified_crs))
        baseline_crs = min(100, max(0, baseline_crs))

        delta = modified_crs - baseline_crs
        delta_pct = (delta / max(baseline_crs, 1)) * 100

        # Alert level classification
        def alert_level(crs):
            if crs >= 70:
                return "RED"
            elif crs >= 45:
                return "AMBER"
            return "GREEN"

        # Economic impact
        from app.services.deployment import EconomicImpactCalculator
        econ = EconomicImpactCalculator()
        baseline_impact = econ.calculate(baseline_crs, corridor, event_cause)
        modified_impact = econ.calculate(modified_crs, corridor, event_cause)

        # Sensitivity breakdown
        factors = []
        if attendance_multiplier != 1.0:
            factors.append({
                "factor": "Attendance",
                "change": f"{(attendance_multiplier - 1) * 100:+.0f}%",
                "crs_effect": round((attendance_effect - 1) * baseline_crs, 1),
            })
        if weather != "clear":
            factors.append({
                "factor": "Weather",
                "change": weather.replace("_", " ").title(),
                "crs_effect": round((weather_mult - 1) * baseline_crs, 1),
            })
        if override_hour is not None and override_hour != base_hour:
            factors.append({
                "factor": "Time of Day",
                "change": f"{base_hour}:00 → {override_hour}:00",
                "crs_effect": round((time_mult / baseline_time_mult - 1) * baseline_crs, 1),
            })
        if override_closure is not None and override_closure != requires_closure:
            factors.append({
                "factor": "Road Closure",
                "change": "Yes" if override_closure else "No",
                "crs_effect": round(modified_crs * 0.4 if override_closure else -modified_crs * 0.2, 1),
            })

        return {
            "baseline": {
                "crs": round(baseline_crs, 1),
                "alert_level": alert_level(baseline_crs),
                "economic_cost": baseline_impact["cost_display"],
                "cost_inr": baseline_impact["cost_inr"],
            },
            "modified": {
                "crs": round(modified_crs, 1),
                "alert_level": alert_level(modified_crs),
                "economic_cost": modified_impact["cost_display"],
                "cost_inr": modified_impact["cost_inr"],
            },
            "delta": {
                "crs": round(delta, 1),
                "percentage": round(delta_pct, 1),
                "direction": "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged",
                "cost_delta_inr": round(modified_impact["cost_inr"] - baseline_impact["cost_inr"], 0),
            },
            "sensitivity_factors": factors,
            "parameters": {
                "attendance_multiplier": attendance_multiplier,
                "weather": weather,
                "hour": hour,
                "requires_closure": closure,
            },
        }
