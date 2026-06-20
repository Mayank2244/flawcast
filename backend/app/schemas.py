"""Pydantic schemas for API."""
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class EventOut(BaseModel):
    id: str
    event_type: str
    latitude: float
    longitude: float
    event_cause: Optional[str] = None
    corridor: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    start_datetime: Optional[datetime] = None
    description: Optional[str] = None
    junction: Optional[str] = None
    zone: Optional[str] = None
    title: Optional[str] = None
    peak_crs_score: Optional[float] = None

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    id: int
    alert_type: str
    severity: int
    incident_type: Optional[str] = None
    title: str
    latitude: float
    longitude: float
    crs_score: Optional[float] = None
    eta_clear_min: Optional[int] = None
    status: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DeploymentBriefOut(BaseModel):
    id: int
    title: str
    officers_needed: int
    deploy_by: datetime
    primary_junction: Optional[str] = None
    secondary_junction: Optional[str] = None
    estimated_reduction_pct: Optional[float] = None
    economic_savings_inr: Optional[float] = None
    brief_text: Optional[str] = None
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


class WhatIfRequest(BaseModel):
    event_type: str = "planned"
    event_cause: str = "public_event"
    corridor: str = "CBD 2"
    priority: str = "High"
    requires_closure: bool = False
    base_hour: int = 18
    attendance_multiplier: float = 1.0
    weather: str = "clear"
    override_hour: Optional[int] = None
    override_closure: Optional[bool] = None


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
