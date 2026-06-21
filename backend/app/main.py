"""FlowCast AI — FastAPI Application Entry Point."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.database import init_db

logger = logging.getLogger("flowcast")
settings = get_settings()


def auto_import_if_empty():
    """Import CSV data into SQLite if the database is empty."""
    from app.database import SessionLocal
    from app.models.orm import Event, Alert

    db = SessionLocal()
    try:
        count = db.query(Event).count()
        if count > 0:
            logger.info(f"Database has {count} events.")
            # Check if alerts exist, if not generate them
            alert_count = db.query(Alert).count()
            if alert_count == 0:
                logger.info("No alerts found. Generating initial alerts...")
                _generate_initial_alerts(db, count)
            return

        csv_path = settings.resolved_dataset_path
        if not csv_path.exists():
            # Try the root-level CSV
            alt = settings.project_root / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
            if alt.exists():
                csv_path = alt
            else:
                logger.warning(f"Dataset not found at {csv_path}")
                return

        import pandas as pd
        from datetime import datetime

        logger.info(f"Importing dataset from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)

        imported = 0
        for _, row in df.iterrows():
            try:
                start_dt = pd.to_datetime(row.get("start_datetime"), utc=True, errors="coerce")
                if pd.isna(start_dt):
                    continue

                lat = float(row.get("latitude", 0))
                lng = float(row.get("longitude", 0))
                if lat == 0 or lng == 0:
                    continue

                end_dt = pd.to_datetime(row.get("end_datetime"), utc=True, errors="coerce")
                resolved_dt = pd.to_datetime(row.get("resolved_datetime"), utc=True, errors="coerce")
                created_dt = pd.to_datetime(row.get("created_date"), utc=True, errors="coerce")

                event = Event(
                    id=str(row.get("id", f"AUTO_{imported}")),
                    event_type=str(row.get("event_type", "unplanned")),
                    latitude=lat,
                    longitude=lng,
                    end_latitude=float(row.get("endlatitude", 0)) if pd.notna(row.get("endlatitude")) else None,
                    end_longitude=float(row.get("endlongitude", 0)) if pd.notna(row.get("endlongitude")) else None,
                    address=str(row.get("address", ""))[:500] if pd.notna(row.get("address")) else None,
                    end_address=str(row.get("end_address", ""))[:500] if pd.notna(row.get("end_address")) else None,
                    event_cause=str(row.get("event_cause", "others")) if pd.notna(row.get("event_cause")) else "others",
                    requires_road_closure=str(row.get("requires_road_closure", "")).upper() == "TRUE",
                    start_datetime=start_dt.to_pydatetime().replace(tzinfo=None),
                    end_datetime=end_dt.to_pydatetime().replace(tzinfo=None) if pd.notna(end_dt) else None,
                    status=str(row.get("status", "closed")) if pd.notna(row.get("status")) else "closed",
                    authenticated=str(row.get("authenticated", "")) if pd.notna(row.get("authenticated")) else None,
                    description=str(row.get("description", ""))[:1000] if pd.notna(row.get("description")) else None,
                    veh_type=str(row.get("veh_type", "")) if pd.notna(row.get("veh_type")) else None,
                    corridor=str(row.get("corridor", "")) if pd.notna(row.get("corridor")) else None,
                    priority=str(row.get("priority", "Low")) if pd.notna(row.get("priority")) else "Low",
                    police_station=str(row.get("police_station", "")) if pd.notna(row.get("police_station")) else None,
                    zone=str(row.get("zone", "")) if pd.notna(row.get("zone")) else None,
                    junction=str(row.get("junction", "")) if pd.notna(row.get("junction")) else None,
                    created_date=created_dt.to_pydatetime().replace(tzinfo=None) if pd.notna(created_dt) else None,
                    resolved_datetime=resolved_dt.to_pydatetime().replace(tzinfo=None) if pd.notna(resolved_dt) else None,
                )
                db.add(event)
                imported += 1

                if imported % 500 == 0:
                    db.commit()
                    logger.info(f"  Imported {imported} events...")
            except Exception as e:
                continue

        db.commit()
        logger.info(f"✅ Imported {imported} events into SQLite database.")

        # Auto-analyze top events to generate alerts & deployments
        _generate_initial_alerts(db, imported)

    except Exception as e:
        logger.error(f"Import error: {e}")
    finally:
        db.close()


def _generate_initial_alerts(db, total_events: int):
    """Generate alerts and deployment briefs for recent high-priority events."""
    from app.models.orm import Event, Alert
    from app.services.flowcast import FlowCastService

    service = FlowCastService(settings.resolved_models_dir)
    events = (
        db.query(Event)
        .filter((Event.priority == "High") | (Event.event_cause == "construction") | (Event.event_cause == "water_logging"))
        .order_by(Event.start_datetime.desc())
        .limit(min(300, total_events // 4))
        .all()
    )

    generated = 0
    for event in events:
        try:
            service.analyze_and_store(db, event)
            generated += 1
        except Exception:
            continue

    logger.info(f"✅ Generated {generated} alerts & deployment briefs.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, import data if needed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logger.info("🚀 FlowCast AI starting up...")
    init_db()
    auto_import_if_empty()
    logger.info("✅ FlowCast AI ready.")
    yield
    logger.info("FlowCast AI shutting down.")


app = FastAPI(
    title="FlowCast AI",
    description="Event-Driven Congestion Prediction System — Flipkart Gridlock 5.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "FlowCast AI API running. Open /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
