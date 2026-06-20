"""Live OpenWeather and GNews integrations."""
import os
import httpx
import logging
from typing import Dict, List, Any
from app.config import get_settings

logger = logging.getLogger("flowcast.live_feeds")
settings = get_settings()

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

class LiveFeedsService:
    def __init__(self):
        self.weather_key = settings.openweather_api_key
        self.news_key = settings.gnews_api_key

    async def get_weather(self) -> Dict[str, Any]:
        """Fetch current weather for Bengaluru from OpenWeather."""
        if not self.weather_key or self.weather_key == "7bc824583c3a80f768ecb81e10a01088":
            # Fallback if key is missing or is the MapMyIndia placeholder
            return {
                "temp": 28.5,
                "condition": "Cloudy",
                "icon": "☁️",
                "is_live": False
            }

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={BENGALURU_LAT}&lon={BENGALURU_LON}&appid={self.weather_key}&units=metric"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                temp = data["main"]["temp"]
                weather_desc = data["weather"][0]["main"]
                icon_map = {"Clear": "☀️", "Rain": "🌧️", "Clouds": "☁️", "Drizzle": "🌦️", "Thunderstorm": "⛈️"}
                return {
                    "temp": round(temp, 1),
                    "condition": weather_desc,
                    "icon": icon_map.get(weather_desc, "🌤️"),
                    "is_live": True
                }
        except Exception as e:
            logger.error(f"OpenWeather API Error: {e}")
            return {"temp": 28.5, "condition": "Cloudy", "icon": "☁️", "is_live": False}

    async def get_news(self) -> List[Dict[str, str]]:
        """Fetch Bengaluru traffic news using GNews API or fallback scraper."""
        if not self.news_key or self.news_key == "9a1965fb75d9f4c678bb8b2cffa0dd5b":
            # Try to use the unofficial gnews python package as a fallback scraper
            try:
                from gnews import GNews
                gnews = GNews(language="en", country="IN", max_results=3)
                articles = gnews.get_news("Bengaluru traffic")
                if articles:
                    return [{"title": a["title"], "url": a["url"], "is_live": True} for a in articles[:3]]
            except Exception as e:
                logger.error(f"GNews Python Package Error: {e}")
            
            # Absolute fallback
            return [
                {"title": "Heavy traffic reported on Outer Ring Road due to waterlogging.", "url": "#", "is_live": False},
                {"title": "BTP issues advisory for upcoming IPL match at Chinnaswamy.", "url": "#", "is_live": False},
                {"title": "Minor accident near Silk Board Junction cleared.", "url": "#", "is_live": False}
            ]

        # Use official GNews API (gnews.io)
        url = f"https://gnews.io/api/v4/search?q=bengaluru%20traffic&lang=en&country=in&max=3&apikey={self.news_key}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("articles", [])
                return [{"title": a["title"], "url": a["url"], "is_live": True} for a in articles]
        except Exception as e:
            logger.error(f"GNews.io API Error: {e}")
            return [
                {"title": "Heavy traffic reported on Outer Ring Road due to waterlogging.", "url": "#", "is_live": False},
            ]

live_feeds = LiveFeedsService()
