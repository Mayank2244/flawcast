"""MapMyIndia (Mappls) API integration service."""
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger("flowcast.mapmyindia")

MAPPLS_TOKEN_URL = "https://outpost.mappls.com/api/security/oauth/token"
MAPPLS_GEOCODE_URL = "https://atlas.mappls.com/api/places/geocode"
MAPPLS_REVERSE_URL = "https://apis.mappls.com/advancedmaps/v1/{token}/rev_geocode"
MAPPLS_ROUTE_URL = "https://apis.mappls.com/advancedmaps/v1/{token}/route_adv/driving"


class MapMyIndiaService:
    """Handles MapMyIndia/Mappls API authentication and calls."""

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        # If they only provide one string, assume it is a direct permanent token/REST key
        self._token: Optional[str] = client_id if client_id and not client_secret else None
        self._token_expiry: float = float('inf') if self._token else 0
        self.enabled = bool(client_id)

    def get_token(self) -> Optional[str]:
        """Get OAuth2 access token, refreshing if expired."""
        if not self.enabled:
            return None

        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        try:
            resp = httpx.post(
                MAPPLS_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expiry = time.time() + data.get("expires_in", 86400)
            logger.info("MapMyIndia token refreshed successfully.")
            return self._token
        except Exception as e:
            logger.warning(f"MapMyIndia token error: {e}")
            return None

    def geocode(self, address: str) -> Optional[dict]:
        """Geocode an address to lat/lng using MapMyIndia."""
        token = self.get_token()
        if not token:
            return None

        try:
            resp = httpx.get(
                MAPPLS_GEOCODE_URL,
                params={"address": address, "itemCount": 1},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("copResults"):
                r = data["copResults"]
                return {
                    "latitude": float(r.get("latitude", 0)),
                    "longitude": float(r.get("longitude", 0)),
                    "formatted_address": r.get("formattedAddress", address),
                    "place_id": r.get("eLoc", ""),
                }
        except Exception as e:
            logger.warning(f"Geocode error: {e}")
        return None

    def reverse_geocode(self, lat: float, lng: float) -> Optional[dict]:
        """Reverse geocode lat/lng to address."""
        token = self.get_token()
        if not token:
            return None

        try:
            url = MAPPLS_REVERSE_URL.format(token=token)
            resp = httpx.get(url, params={"lat": lat, "lng": lng}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("results"):
                r = data["results"][0]
                return {
                    "formatted_address": r.get("formatted_address", ""),
                    "locality": r.get("locality", ""),
                    "district": r.get("district", ""),
                    "city": r.get("city", "Bengaluru"),
                }
        except Exception as e:
            logger.warning(f"Reverse geocode error: {e}")
        return None

    def get_route(self, start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> Optional[dict]:
        """Get driving route between two points."""
        token = self.get_token()
        if not token:
            return None

        try:
            url = MAPPLS_ROUTE_URL.format(token=token)
            coords = f"{start_lng},{start_lat};{end_lng},{end_lat}"
            resp = httpx.get(
                url,
                params={
                    "geometries": "polyline",
                    "overview": "full",
                    "alternatives": "true",
                    "steps": "true",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Route error: {e}")
        return None

    def get_map_config(self) -> dict:
        """Return config for frontend map initialization."""
        token = self.get_token()
        if token:
            return {
                "provider": "mapmyindia",
                "token": token,
                "enabled": True,
            }
        return {
            "provider": "osm",
            "token": None,
            "enabled": False,
        }
