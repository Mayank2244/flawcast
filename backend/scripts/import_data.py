#!/usr/bin/env python3
"""
FlowCast AI — Data Import Script
Imports Astram event dataset into MySQL and seeds reference data.

Usage (Python 3.11):
  cd backend && python scripts/import_data.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import engine, SessionLocal
from app.models.orm import Event, RoadSegment, PoliceStation, Base

settings = get_settings()

CORRIDOR_COORDS = {
    "Mysore Road": (12.9446, 77.5274),
    "Bellary Road 1": (13.0001, 77.5840),
    "Bellary Road 2": (13.0634, 77.5933),
    "Tumkur Road": (13.0374, 77.5181),
    "Hosur Road": (12.9169, 77.6100),
    "ORR North 1": (13.0354, 77.5947),
    "ORR North 2": (13.0447, 77.5829),
    "ORR East 1": (12.9465, 77.6987),
    "ORR East 2": (13.0008, 77.6814),
    "ORR West 1": (12.9446, 77.5274),
    "Old Madras Road": (12.9753, 77.6257),
    "Magadi Road": (12.9789, 77.5644),
    "Bannerghata Road": (12.9077, 77.6006),
    "West of Chord Road": (12.9900, 77.5500),
    "CBD 1": (12.9716, 77.5946),
    "CBD 2": (12.9788, 77.5995),
    "Non-corridor": (12.9716, 77.5946),
}

POLICE_STATIONS = [
    ("Cubbon Park", 12.9788, 77.5995, "Central", 15),
    ("HSR Layout", 12.9219, 77.6452, "South East", 12),
    ("Hebbala", 13.0354, 77.5947, "North", 14),
    ("Jayanagara", 12.9274, 77.5807, "South", 10),
    ("HAL Old Airport", 12.9635, 77.6642, "East", 11),
    ("Peenya", 13.0374, 77.5181, "West", 9),
    ("Whitefield", 12.9698, 77.7500, "East", 13),
    ("Madiwala", 12.9071, 77.6286, "South", 10),
    ("Wilson Garden", 12.9539, 77.5852, "Central", 11),
    ("Yeshwanthpura", 13.0289, 77.5442, "North West", 10),
    ("K.R. Pura", 13.0008, 77.6814, "East", 12),
    ("Electronic City", 12.8456, 77.6603, "South", 8),
]


def parse_bool(val):
    if val in (True, "TRUE", "True", "true", "yes", "1", 1):
        return True
    return False


def parse_datetime(val):
    if pd.isna(val) or val in ("NULL", "", None):
        return None
    try:
        return pd.to_datetime(val, utc=True).replace(tzinfo=None)
    except Exception:
        return None


def import_events(db: Session, csv_path: str) -> int:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} rows from dataset")

    existing = {e.id for e in db.query(Event.id).all()}
    imported = 0

    for _, row in df.iterrows():
        eid = str(row["id"])
        if eid in existing:
            continue

        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lng = pd.to_numeric(row.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lng):
            continue

        priority = row.get("priority", "Low")
        if priority not in ("High", "Low", "Medium"):
            priority = "Low"

        event = Event(
            id=eid,
            event_type=row.get("event_type", "unplanned"),
            latitude=float(lat),
            longitude=float(lng),
            end_latitude=pd.to_numeric(row.get("endlatitude"), errors="coerce"),
            end_longitude=pd.to_numeric(row.get("endlongitude"), errors="coerce"),
            address=str(row.get("address", "")) if pd.notna(row.get("address")) else None,
            end_address=str(row.get("end_address", "")) if pd.notna(row.get("end_address")) else None,
            event_cause=str(row.get("event_cause", "")) if pd.notna(row.get("event_cause")) else None,
            requires_road_closure=parse_bool(row.get("requires_road_closure", False)),
            start_datetime=parse_datetime(row.get("start_datetime")),
            end_datetime=parse_datetime(row.get("end_datetime")),
            status=str(row.get("status", "")) if pd.notna(row.get("status")) else None,
            authenticated=str(row.get("authenticated", "")) if pd.notna(row.get("authenticated")) else None,
            description=str(row.get("description", "")) if pd.notna(row.get("description")) else None,
            veh_type=str(row.get("veh_type", "")) if pd.notna(row.get("veh_type")) else None,
            corridor=str(row.get("corridor", "")) if pd.notna(row.get("corridor")) else None,
            priority=priority,
            police_station=str(row.get("police_station", "")) if pd.notna(row.get("police_station")) else None,
            zone=str(row.get("zone", "")) if pd.notna(row.get("zone")) else None,
            junction=str(row.get("junction", "")) if pd.notna(row.get("junction")) else None,
            created_date=parse_datetime(row.get("created_date")),
            resolved_datetime=parse_datetime(row.get("resolved_datetime")),
        )
        if event.start_datetime is None:
            event.start_datetime = datetime.utcnow()
        db.add(event)
        imported += 1

        if imported % 500 == 0:
            db.commit()
            print(f"  Imported {imported} events...")

    db.commit()
    return imported


def seed_road_segments(db: Session):
    if db.query(RoadSegment).count() > 0:
        return
    for name, (lat, lng) in CORRIDOR_COORDS.items():
        db.add(RoadSegment(
            segment_name=name, corridor=name,
            center_lat=lat, center_lng=lng,
            capacity=3000 if "ORR" in name else 2000,
            lane_count=4 if "ORR" in name else 2,
            speed_limit=60 if "ORR" in name else 40,
        ))
    db.commit()
    print(f"Seeded {len(CORRIDOR_COORDS)} road segments")


def seed_police_stations(db: Session):
    if db.query(PoliceStation).count() > 0:
        return
    for name, lat, lng, zone, officers in POLICE_STATIONS:
        db.add(PoliceStation(name=name, latitude=lat, longitude=lng, zone=zone, officer_count=officers))
    db.commit()
    print(f"Seeded {len(POLICE_STATIONS)} police stations")


def main():
    csv_path = settings.resolved_dataset_path

    if not csv_path.exists():
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)

    print("FlowCast AI — Data Import")
    print(f"Database: {settings.mysql_database}@{settings.mysql_host}")
    print(f"Dataset: {csv_path}")

    db = SessionLocal()
    try:
        seed_road_segments(db)
        seed_police_stations(db)
        count = import_events(db, str(csv_path))
        print(f"\n✓ Successfully imported {count} events")
        total = db.query(Event).count()
        print(f"✓ Total events in database: {total}")
    except Exception as e:
        print(f"ERROR: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
