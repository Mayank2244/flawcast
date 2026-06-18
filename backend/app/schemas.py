"""Pydantic schemas for API."""
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class EventOut(BaseModel):
    id: str
    event_type: str
    latitude: float
    longitude: float
    event_cause: Optional[str]
    corridor: Optional[str]
    priority: Optional[str]
    status: Optional[str]
    start_datetime: Optional[datetime]
    description: Optional[str]
    junction: Optional[str]
    zone: Optional[str]

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    id: int
    alert_type: str
    severity: int
    incident_type: Optional[str]
    title: str
    latitude: float
    longitude: float
    crs_score: Optional[float]
    eta_clear_min: Optional[int]
    status: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class DeploymentBriefOut(BaseModel):
    id: int
    title: str
    officers_needed: int
    deploy_by: datetime
    primary_junction: Optional[str]
    secondary_junction: Optional[str]
    estimated_reduction_pct: Optional[float]
    economic_savings_inr: Optional[float]
    brief_text: Optional[str]
    status: str

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    event_type: str = "planned"
    event_cause: str = "public_event"
    corridor: str = "CBD 2"
    priority: str = "High"
    latitude: float = 12.9788
    longitude: float = 77.5995
    description: str = ""
    address: str = ""
    requires_closure: bool = False
    junction: str = ""


class NLPRequest(BaseModel):
    text: str
    address: str = ""


class DashboardStats(BaseModel):
    total_events: int
    active_alerts: int
    red_alerts: int
    planned_events: int
    unplanned_events: int
    prediction_accuracy_pct: float
    advance_warning_hours: int
    alert_response_seconds: int
    total_economic_cost_inr: float
    prevented_cost_inr: float
    monthly_savings_crore: float
    officer_coverage_multiplier: int
