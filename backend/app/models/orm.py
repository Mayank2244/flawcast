"""SQLAlchemy ORM models."""
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, Integer, BigInteger,
    Enum, DECIMAL, ForeignKey, JSON, Date, TIMESTAMP, func,
)
from sqlalchemy.orm import relationship
from app.database import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(String(20), primary_key=True)
    event_type = Column(Enum("planned", "unplanned"), nullable=False)
    latitude = Column(DECIMAL(10, 7), nullable=False)
    longitude = Column(DECIMAL(10, 7), nullable=False)
    end_latitude = Column(DECIMAL(10, 7))
    end_longitude = Column(DECIMAL(10, 7))
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
    priority = Column(Enum("High", "Low", "Medium"), default="Low")
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
    center_lat = Column(DECIMAL(10, 7))
    center_lng = Column(DECIMAL(10, 7))
    capacity = Column(Integer, default=1000)
    lane_count = Column(Integer, default=2)
    speed_limit = Column(Integer, default=40)
    betweenness = Column(DECIMAL(8, 6), default=0)
    zone = Column(String(100))


class PoliceStation(Base):
    __tablename__ = "police_stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    latitude = Column(DECIMAL(10, 7), nullable=False)
    longitude = Column(DECIMAL(10, 7), nullable=False)
    zone = Column(String(100))
    officer_count = Column(Integer, default=12)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_type = Column(Enum("RED", "AMBER", "GREEN"), nullable=False)
    severity = Column(Integer, nullable=False)
    incident_type = Column(String(50))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    latitude = Column(DECIMAL(10, 7), nullable=False)
    longitude = Column(DECIMAL(10, 7), nullable=False)
    affected_radius_km = Column(DECIMAL(4, 2), default=2.0)
    crs_score = Column(DECIMAL(5, 2))
    eta_clear_min = Column(Integer)
    source = Column(Enum("sensor_anomaly", "nlp", "fusion", "manual"), default="fusion")
    event_id = Column(String(20), ForeignKey("events.id"))
    status = Column(Enum("active", "acknowledged", "resolved"), default="active")
    created_at = Column(TIMESTAMP, server_default=func.now())
    resolved_at = Column(TIMESTAMP)


class DeploymentBrief(Base):
    __tablename__ = "deployment_briefs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_id = Column(BigInteger, ForeignKey("alerts.id"))
    event_id = Column(String(20), ForeignKey("events.id"))
    title = Column(String(255), nullable=False)
    officers_needed = Column(Integer, nullable=False)
    deploy_by = Column(DateTime, nullable=False)
    primary_junction = Column(String(150))
    secondary_junction = Column(String(150))
    station_id = Column(Integer, ForeignKey("police_stations.id"))
    estimated_reduction_pct = Column(DECIMAL(5, 2))
    economic_savings_inr = Column(DECIMAL(15, 2))
    brief_text = Column(Text)
    status = Column(Enum("pending", "deployed", "completed"), default="pending")
    created_at = Column(TIMESTAMP, server_default=func.now())


class CongestionForecast(Base):
    __tablename__ = "congestion_forecasts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=False)
    forecast_time = Column(DateTime, nullable=False)
    target_time = Column(DateTime, nullable=False)
    crs_score = Column(DECIMAL(5, 2), nullable=False)
    crs_p10 = Column(DECIMAL(5, 2))
    crs_p50 = Column(DECIMAL(5, 2))
    crs_p90 = Column(DECIMAL(5, 2))
    event_id = Column(String(20), ForeignKey("events.id"))
    source = Column(Enum("planned_tft", "unplanned_anomaly", "nlp", "fusion"), default="planned_tft")
    created_at = Column(TIMESTAMP, server_default=func.now())


class EconomicImpact(Base):
    __tablename__ = "economic_impact"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_id = Column(String(20), ForeignKey("events.id"))
    alert_id = Column(BigInteger, ForeignKey("alerts.id"))
    date = Column(Date, nullable=False)
    affected_vehicles = Column(Integer)
    delay_minutes = Column(Integer)
    cost_inr = Column(DECIMAL(15, 2))
    prevented_cost_inr = Column(DECIMAL(15, 2), default=0)
    notes = Column(Text)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(DECIMAL(8, 4))
    evaluated_at = Column(TIMESTAMP, server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(50))
    details = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
