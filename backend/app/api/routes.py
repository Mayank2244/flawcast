"""FastAPI route handlers."""
import asyncio
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.database import get_db
from app.models.orm import Event, Alert, DeploymentBrief, ModelMetric, EconomicImpact
from app.schemas import EventOut, AlertOut, DeploymentBriefOut, PredictRequest, NLPRequest
from app.services.flowcast import FlowCastService
from app.services.demo_scenarios import get_scenarios, get_scenario
from app.services.corridor_risk import CorridorRiskService
from app.config import get_settings

router = APIRouter()
settings = get_settings()
service = FlowCastService(settings.resolved_models_dir)
corridor_service = CorridorRiskService()


@router.get("/health")
def health():
    return {"status": "ok", "service": "FlowCast AI", "version": "1.0.0"}


@router.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    return service.get_dashboard_stats(db)


@router.get("/dashboard/heatmap")
def heatmap(db: Session = Depends(get_db), limit: int = Query(500, le=2000)):
    return service.get_heatmap_data(db, limit)


@router.get("/events", response_model=list[EventOut])
def list_events(
    db: Session = Depends(get_db),
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    corridor: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    q = db.query(Event)
    if event_type:
        q = q.filter(Event.event_type == event_type)
    if status:
        q = q.filter(Event.status == status)
    if corridor:
        q = q.filter(Event.corridor == corridor)
    return q.order_by(desc(Event.start_datetime)).offset(offset).limit(limit).all()


@router.get("/events/{event_id}", response_model=EventOut)
def get_event(event_id: str, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(404, "Event not found")
    return event


@router.get("/events/{event_id}/forecast")
def event_forecast(event_id: str, db: Session = Depends(get_db)):
    return service.get_timeline_forecast(db, event_id)


@router.post("/events/{event_id}/analyze")
def analyze_event(event_id: str, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(404, "Event not found")
    return service.analyze_and_store(db, event)


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    db: Session = Depends(get_db),
    alert_type: Optional[str] = None,
    status: str = "active",
    limit: int = Query(50, le=200),
):
    q = db.query(Alert)
    if alert_type:
        q = q.filter(Alert.alert_type == alert_type)
    if status:
        q = q.filter(Alert.status == status)
    return q.order_by(desc(Alert.created_at)).limit(limit).all()


@router.patch("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = "acknowledged"
    db.commit()
    return {"status": "acknowledged", "id": alert_id}


@router.get("/deployments", response_model=list[DeploymentBriefOut])
def list_deployments(db: Session = Depends(get_db), status: str = "pending", limit: int = 20):
    return (
        db.query(DeploymentBrief)
        .filter(DeploymentBrief.status == status)
        .order_by(desc(DeploymentBrief.created_at))
        .limit(limit)
        .all()
    )


@router.post("/predict")
def predict(req: PredictRequest):
    result = service.fusion.analyze_event(
        event_type=req.event_type,
        event_cause=req.event_cause,
        corridor=req.corridor,
        priority=req.priority,
        latitude=req.latitude,
        longitude=req.longitude,
        description=req.description,
        address=req.address,
        requires_closure=req.requires_closure,
    )
    brief = service.planner.generate_brief(
        fusion_crs=result["fusion_crs"],
        alert_type=result["alert_type"],
        event_cause=req.event_cause,
        corridor=req.corridor,
        latitude=req.latitude,
        longitude=req.longitude,
        junction=req.junction,
        event_type=req.event_type,
        propagation=result["propagation"],
    )
    impact = service.economic.calculate(result["fusion_crs"], req.corridor, req.event_cause)
    return {**result, "deployment_brief": brief, "economic_impact": impact}


@router.post("/nlp/classify")
def nlp_classify(req: NLPRequest):
    return service.fusion.nlp.classify(req.text, req.address)


@router.get("/analytics/corridors")
def corridor_analytics(db: Session = Depends(get_db)):
    rows = (
        db.query(Event.corridor, func.count(Event.id).label("count"))
        .group_by(Event.corridor)
        .order_by(desc("count"))
        .limit(15)
        .all()
    )
    return [{"corridor": r[0] or "Unknown", "count": r[1]} for r in rows]


@router.get("/analytics/causes")
def cause_analytics(db: Session = Depends(get_db)):
    rows = (
        db.query(Event.event_cause, func.count(Event.id).label("count"))
        .group_by(Event.event_cause)
        .order_by(desc("count"))
        .limit(12)
        .all()
    )
    return [{"cause": r[0] or "Unknown", "count": r[1]} for r in rows]


@router.get("/analytics/hourly")
def hourly_analytics(db: Session = Depends(get_db)):
    events = db.query(Event.start_datetime).all()
    hours = [0] * 24
    for (dt,) in events:
        if dt:
            hours[dt.hour] += 1
    return [{"hour": h, "count": c} for h, c in enumerate(hours)]


@router.get("/graph/stats")
def graph_stats():
    return service.fusion.graph.get_graph_stats()


@router.get("/models/metrics")
def model_metrics(db: Session = Depends(get_db)):
    metrics = db.query(ModelMetric).order_by(desc(ModelMetric.evaluated_at)).limit(20).all()
    return [{"model": m.model_name, "metric": m.metric_name, "value": float(m.metric_value), "at": str(m.evaluated_at)} for m in metrics]


@router.get("/economic/summary")
def economic_summary(db: Session = Depends(get_db)):
    total = db.query(func.sum(EconomicImpact.cost_inr)).scalar() or 0
    prevented = db.query(func.sum(EconomicImpact.prevented_cost_inr)).scalar() or 0
    return service.economic.monthly_aggregate([float(total)])


@router.get("/demo/scenarios")
def demo_scenarios():
    return get_scenarios()


@router.post("/demo/run/{scenario_id}")
def run_demo_scenario(scenario_id: str):
    scenario = get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found")
    result = service.fusion.analyze_event(
        event_type=scenario["event_type"],
        event_cause=scenario["event_cause"],
        corridor=scenario["corridor"],
        priority=scenario["priority"],
        latitude=scenario["latitude"],
        longitude=scenario["longitude"],
        description=scenario["description"],
        requires_closure=scenario.get("requires_closure", False),
    )
    brief = service.planner.generate_brief(
        fusion_crs=result["fusion_crs"],
        alert_type=result["alert_type"],
        event_cause=scenario["event_cause"],
        corridor=scenario["corridor"],
        latitude=scenario["latitude"],
        longitude=scenario["longitude"],
        junction=scenario.get("junction", ""),
        event_type=scenario["event_type"],
        propagation=result["propagation"],
    )
    impact = service.economic.calculate(result["fusion_crs"], scenario["corridor"], scenario["event_cause"])
    return {
        "scenario": scenario,
        **result,
        "deployment_brief": brief,
        "economic_impact": impact,
    }


@router.get("/corridors/risk-index")
def corridor_risk_index(db: Session = Depends(get_db)):
    return corridor_service.compute_index(db)


@router.get("/alerts/stream")
async def alert_stream():
    """Server-Sent Events stream for live dashboard updates."""
    from app.database import SessionLocal

    async def generate():
        last_id = 0
        while True:
            db = SessionLocal()
            try:
                alerts = (
                    db.query(Alert)
                    .filter(Alert.id > last_id, Alert.status == "active")
                    .order_by(Alert.id)
                    .limit(10)
                    .all()
                )
                for alert in alerts:
                    last_id = alert.id
                    payload = {
                        "id": alert.id,
                        "type": alert.alert_type,
                        "title": alert.title,
                        "crs": float(alert.crs_score or 0),
                        "lat": float(alert.latitude),
                        "lng": float(alert.longitude),
                        "incident": alert.incident_type,
                        "severity": alert.severity,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                stats = service.get_dashboard_stats(db)
                yield f"data: {json.dumps({'type': 'stats', **stats})}\n\n"
            finally:
                db.close()
            await asyncio.sleep(5)

    return StreamingResponse(generate(), media_type="text/event-stream")
