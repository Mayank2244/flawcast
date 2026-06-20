#!/usr/bin/env python3
"""FlowCast AI — shared API clients for pipeline, fusion, and dashboard.

Reads credentials from environment / .env only (never writes or modifies keys).
Gracefully falls back to demo data when APIs are unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from flowcast_config import BENGALURU_CENTER, PROJECT_ROOT

logger = logging.getLogger("flowcast.api")

BENGALURU_LAT, BENGALURU_LON = BENGALURU_CENTER

# Placeholder values that should trigger fallback (matches live_feeds convention)
_PLACEHOLDER_KEYS = frozenset(
    {
        "7bc824583c3a80f768ecb81e10a01088",
        "9a1965fb75d9f4c678bb8b2cffa0dd5b",
    }
)


def _load_dotenv() -> None:
    """Load .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def _env(key: str, default: str = "") -> str:
    _load_dotenv()
    return os.getenv(key, default).strip()


def _valid_key(key: str) -> bool:
    return bool(key) and key not in _PLACEHOLDER_KEYS


def get_weather_sync() -> dict[str, Any]:
    """Fetch Bengaluru weather for fusion engine (sync).

    Returns:
        Dict with rain_mm, visibility, wind_speed, temp, condition, is_live.
    """
    fallback = {
        "rain_mm": 0.0,
        "visibility": 10000.0,
        "wind_speed": 5.0,
        "temp": 28.5,
        "condition": "Cloudy",
        "icon": "☁️",
        "is_live": False,
    }
    key = _env("OPENWEATHER_API_KEY")
    if not _valid_key(key):
        return fallback

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={BENGALURU_LAT}&lon={BENGALURU_LON}&appid={key}&units=metric"
    )
    try:
        resp = httpx.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        rain_mm = sum(r.get("rain", {}).get("3h", 0) for r in [data]) or data.get("rain", {}).get("1h", 0)
        visibility = float(data.get("visibility", 10000))
        wind = float(data.get("wind", {}).get("speed", 5.0))
        temp = float(data["main"]["temp"])
        condition = data["weather"][0]["main"]
        icon_map = {"Clear": "☀️", "Rain": "🌧️", "Clouds": "☁️", "Drizzle": "🌦️", "Thunderstorm": "⛈️"}
        return {
            "rain_mm": float(rain_mm),
            "visibility": visibility,
            "wind_speed": wind,
            "temp": round(temp, 1),
            "condition": condition,
            "icon": icon_map.get(condition, "🌤️"),
            "is_live": True,
        }
    except Exception as exc:
        logger.warning("OpenWeather fetch failed: %s", exc)
        return fallback


def get_news_sync(max_results: int = 5) -> list[dict[str, Any]]:
    """Fetch Bengaluru traffic news headlines (sync)."""
    fallback = [
        {"title": "Heavy traffic reported on Outer Ring Road due to waterlogging.", "url": "#", "is_live": False},
        {"title": "BTP issues advisory for upcoming IPL match at Chinnaswamy.", "url": "#", "is_live": False},
        {"title": "Minor accident near Silk Board Junction cleared.", "url": "#", "is_live": False},
    ]
    key = _env("GNEWS_API_KEY")
    if not _valid_key(key):
        try:
            from gnews import GNews

            gnews = GNews(language="en", country="IN", max_results=max_results)
            articles = gnews.get_news("Bengaluru traffic")
            if articles:
                return [{"title": a["title"], "url": a["url"], "is_live": True} for a in articles[:max_results]]
        except Exception as exc:
            logger.warning("GNews scraper failed: %s", exc)
        return fallback

    url = (
        f"https://gnews.io/api/v4/search?q=bengaluru%20traffic"
        f"&lang=en&country=in&max={max_results}&apikey={key}"
    )
    try:
        resp = httpx.get(url, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [{"title": a["title"], "url": a["url"], "is_live": True} for a in articles]
    except Exception as exc:
        logger.warning("GNews API failed: %s", exc)
        return fallback


def get_mapmyindia_service() -> Any:
    """Return MapMyIndia service instance (credentials from env only)."""
    try:
        from backend.app.services.mapmyindia import MapMyIndiaService
    except ImportError:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        from app.services.mapmyindia import MapMyIndiaService

    client_id = _env("MAPMYINDIA_CLIENT_ID")
    client_secret = _env("MAPMYINDIA_CLIENT_SECRET")
    return MapMyIndiaService(client_id=client_id, client_secret=client_secret)


def reverse_geocode_sync(lat: float, lon: float) -> str:
    """Reverse-geocode coordinates; returns formatted address or fallback."""
    mmi = get_mapmyindia_service()
    if mmi.enabled:
        result = mmi.reverse_geocode(lat, lon)
        if result:
            return result.get("formatted_address", f"{lat:.4f}, {lon:.4f}")
    return f"Bengaluru ({lat:.4f}, {lon:.4f})"


def weather_for_fusion() -> dict[str, float]:
    """Subset of weather fields consumed by fusion engine."""
    w = get_weather_sync()
    return {
        "rain_mm": w["rain_mm"],
        "visibility": w["visibility"],
        "wind_speed": w["wind_speed"],
    }
