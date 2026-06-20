"""SQLAlchemy ORM models — SQLite compatible."""
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, Integer, Float,
    ForeignKey, JSON, Date, func,
)
from sqlalchemy.orm import relationship
from app.database import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(String(20), primary_key=True)
    event_type = Column(String(20), nullable=False)  # planned / unplanned
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    end_latitude = Column(Float)
    end_longitude = Column(Float)
    address = Column(Text)
    end_address = Column(Text)
    event_cause = Column(String(50))
    requires_road_closure = Column(Boolean, default=False)
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime)
    status = Column(String(20))
    authenticated = Column(String(10))
    description = Column(Text)
    veh_type = Column(String(50))
    corridor = Column(String(100))
    priority = Column(String(20), default="Low")
    police_station = Column(String(100))
    zone = Column(String(100))
    junction = Column(String(150))
    created_date = Column(DateTime)
    resolved_datetime = Column(DateTime)


class RoadSegment(Base):
    __tablename__ = "road_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_name = Column(String(150), unique=True, nullable=False)
    corridor = Column(String(100))
    center_lat = Column(Float)
    center_lng = Column(Float)
    capacity = Column(Integer, default=1000)
    lane_count = Column(Integer, default=2)
    speed_limit = Column(Integer, default=40)
    betweenness = Column(Float, default=0)
    zone = Column(String(100))


class PoliceStation(Base):
    __tablename__ = "police_stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    zone = Column(String(100))
    officer_count = Column(Integer, default=12)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(10), nullable=False)  # RED / AMBER / GREEN
    severity = Column(Integer, nullable=False)
    incident_type = Column(String(50))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    affected_radius_km = Column(Float, default=2.0)
    crs_score = Column(Float)
    eta_clear_min = Column(Integer)
    source = Column(String(20), default="fusion")  # sensor_anomaly / nlp / fusion / manual
    event_id = Column(String(20), ForeignKey("events.id"))
    status = Column(String(20), default="active")  # active / acknowledged / resolved
    created_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime)


class DeploymentBrief(Base):
    __tablename__ = "deployment_briefs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    event_id = Column(String(20), ForeignKey("events.id"))
    title = Column(String(255), nullable=False)
    officers_needed = Column(Integer, nullable=False)
    deploy_by = Column(DateTime, nullable=False)
    primary_junction = Column(String(150))
    secondary_junction = Column(String(150))
    station_id = Column(Integer, ForeignKey("police_stations.id"))
    estimated_reduction_pct = Column(Float)
    economic_savings_inr = Column(Float)
    brief_text = Column(Text)
    status = Column(String(20), default="pending")  # pending / deployed / completed
    created_at = Column(DateTime, default=func.now())


class CongestionForecast(Base):
    __tablename__ = "congestion_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=False)
    forecast_time = Column(DateTime, nullable=False)
    target_time = Column(DateTime, nullable=False)
    crs_score = Column(Float, nullable=False)
    crs_p10 = Column(Float)
    crs_p50 = Column(Float)
    crs_p90 = Column(Float)
    event_id = Column(String(20), ForeignKey("events.id"))
    source = Column(String(30), default="planned_tft")
    created_at = Column(DateTime, default=func.now())


class EconomicImpact(Base):
    __tablename__ = "economic_impact"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey("events.id"))
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    date = Column(Date, nullable=False)
    affected_vehicles = Column(Integer)
    delay_minutes = Column(Integer)
    cost_inr = Column(Float)
    prevented_cost_inr = Column(Float, default=0)
    notes = Column(Text)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float)
    evaluated_at = Column(DateTime, default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(50))
    details = Column(JSON)
    created_at = Column(DateTime, default=func.now())


class AccuracyRecord(Base):
    """Stores predicted vs actual CRS for accuracy tracking."""
    __tablename__ = "accuracy_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey("events.id"))
    segment = Column(String(100))
    event_type = Column(String(30))
    event_cause = Column(String(50))
    predicted_crs = Column(Float, nullable=False)
    actual_crs = Column(Float)
    smape = Column(Float)
    alert_correct = Column(Boolean)
    predicted_at = Column(DateTime, default=func.now())
    verified_at = Column(DateTime)
