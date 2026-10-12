"""
AQI (Air Quality Index) Service for PathFinder.
Integrates with the World Air Quality Index (WAQI / aqicn.org) API to provide
real-time AQI levels, pollutant metrics, health advisories, and color scale classifications.
"""

from __future__ import annotations
import json
import os
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional, Tuple


# WAQI Map Tiles URL Template
WAQI_TILE_URL_TEMPLATE = "https://tiles.aqicn.org/tiles/usepa-aqi/{z}/{x}/{y}.png?token={token}"

# Standard EPA / National Air Quality Index (AQI) Scales & Color Definitions
AQI_CATEGORIES = [
    {
        "min": 0,
        "max": 50,
        "label": "Good",
        "color": "#16A34A",       # Green
        "bg_color": "#DCFCE7",    # Light Green
        "border_color": "#86EFAC",
        "text_color": "#14532D",
        "icon": "🟢",
        "description": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "advisory": "Ideal air quality for outdoor activities and exercise for everyone.",
    },
    {
        "min": 51,
        "max": 100,
        "label": "Moderate",
        "color": "#EAB308",       # Yellow
        "bg_color": "#FEF9C3",    # Light Yellow
        "border_color": "#FDE047",
        "text_color": "#713F12",
        "icon": "🟡",
        "description": "Air quality is acceptable; however, some pollutants may cause moderate health concern for sensitive individuals.",
        "advisory": "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    },
    {
        "min": 101,
        "max": 150,
        "label": "Unhealthy for Sensitive Groups",
        "color": "#F97316",       # Orange
        "bg_color": "#FFEDD5",    # Light Orange
        "border_color": "#FDBA74",
        "text_color": "#7C2D12",
        "icon": "🟠",
        "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "advisory": "Children, older adults, and people with respiratory or heart conditions should limit prolonged outdoor exertion.",
    },
    {
        "min": 151,
        "max": 200,
        "label": "Unhealthy",
        "color": "#EF4444",       # Red
        "bg_color": "#FEE2E2",    # Light Red
        "border_color": "#FCA5A5",
        "text_color": "#7F1D1D",
        "icon": "🔴",
        "description": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.",
        "advisory": "Active children, adults, and people with respiratory disease should avoid prolonged outdoor exertion; everyone else should limit exertion.",
    },
    {
        "min": 201,
        "max": 300,
        "label": "Very Unhealthy",
        "color": "#9333EA",       # Purple
        "bg_color": "#F3E8FF",    # Light Purple
        "border_color": "#D8B4FE",
        "text_color": "#581C87",
        "icon": "🟣",
        "description": "Health alert: The risk of health effects is increased for everyone in the area.",
        "advisory": "Avoid prolonged outdoor exertion. Sensitive groups should remain indoors and keep activity levels low.",
    },
    {
        "min": 301,
        "max": 9999,
        "label": "Hazardous",
        "color": "#881337",       # Maroon / Dark Red
        "bg_color": "#FFE4E6",    # Light Maroon
        "border_color": "#FDA4AF",
        "text_color": "#4C0519",
        "icon": "🟤",
        "description": "Health warnings of emergency conditions: The entire population is likely to be affected.",
        "advisory": "Everyone should avoid all outdoor physical activity. Remain indoors with air purifiers active and doors/windows closed.",
    },
]


def get_aqi_category(aqi: float | int) -> Dict[str, Any]:
    """Return category metadata for a numeric AQI value."""
    try:
        val = float(aqi)
    except (ValueError, TypeError):
        val = 0.0

    for cat in AQI_CATEGORIES:
        if cat["min"] <= val <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]


def get_waqi_api_token(custom_token: Optional[str] = None) -> str:
    """Retrieve the configured WAQI API token (defaults to 'demo')."""
    if custom_token and custom_token.strip():
        return custom_token.strip()
    env_token = os.environ.get("WAQI_TOKEN") or os.environ.get("WAQI_API_KEY")
    if env_token and env_token.strip():
        return env_token.strip()
    return "demo"


def _fetch_json(url: str, timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """Helper to perform HTTP GET and parse JSON."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PathFinder-Maps/2.0 (AirQuality; +https://github.com/pathfinder)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
    except Exception as e:
        print(f"[AQI Service] Failed fetching {url}: {e}")
    return None


def fetch_open_meteo_aqi(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fallback real-time Air Quality Provider using Open-Meteo Air Quality API.
    Used when querying coordinates where WAQI demo token is restricted to Shanghai.
    Zero-key, free worldwide real-time pollutant observations.
    """
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat:.4f}&longitude={lon:.4f}&"
        f"current=us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    )
    data = _fetch_json(url)
    if not data or "current" not in data:
        return None

    cur = data["current"]
    us_aqi = cur.get("us_aqi")
    if us_aqi is None:
        return None

    cat = get_aqi_category(us_aqi)
    return {
        "aqi": int(us_aqi),
        "category": cat["label"],
        "color": cat["color"],
        "bg_color": cat["bg_color"],
        "border_color": cat["border_color"],
        "text_color": cat["text_color"],
        "icon": cat["icon"],
        "description": cat["description"],
        "advisory": cat["advisory"],
        "dominant_pollutant": "PM2.5" if (cur.get("pm2_5") or 0) > 25 else "PM10",
        "pollutants": {
            "pm25": cur.get("pm2_5"),
            "pm10": cur.get("pm10"),
            "o3": cur.get("ozone"),
            "no2": cur.get("nitrogen_dioxide"),
            "so2": cur.get("sulphur_dioxide"),
            "co": cur.get("carbon_monoxide"),
        },
        "station_name": f"Area ({lat:.4f}, {lon:.4f})",
        "station_geo": [lat, lon],
        "reporting_time": cur.get("time", "Current Observation"),
        "source": "Open Air Quality / WAQI Compatible Standards",
        "lat": lat,
        "lon": lon,
        "is_fallback": True,
    }


def fetch_waqi_reading(
    lat: float,
    lon: float,
    token: Optional[str] = None,
    area_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch comprehensive AQI data for coordinates from WAQI API.
    Falls back gracefully to Open-Meteo if token is 'demo' and outside Shanghai.
    """
    active_token = get_waqi_api_token(token)
    waqi_url = f"https://api.waqi.info/feed/geo:{lat:.5f};{lon:.5f}/?token={active_token}"

    data = _fetch_json(waqi_url)

    # Check if WAQI returned valid station data for this area
    if data and data.get("status") == "ok" and isinstance(data.get("data"), dict):
        d = data["data"]
        raw_aqi = d.get("aqi")
        city_info = d.get("city", {})
        city_geo = city_info.get("geo")

        # If using 'demo' token outside Shanghai, WAQI server redirects geo queries to Shanghai
        is_demo_redirect = (
            active_token == "demo"
            and city_geo
            and len(city_geo) >= 2
            and abs(city_geo[0] - lat) > 5.0
        )

        if not is_demo_redirect and raw_aqi is not None and isinstance(raw_aqi, (int, float)):
            cat = get_aqi_category(raw_aqi)
            iaqi = d.get("iaqi", {})
            time_info = d.get("time", {})

            return {
                "aqi": int(raw_aqi),
                "category": cat["label"],
                "color": cat["color"],
                "bg_color": cat["bg_color"],
                "border_color": cat["border_color"],
                "text_color": cat["text_color"],
                "icon": cat["icon"],
                "description": cat["description"],
                "advisory": cat["advisory"],
                "dominant_pollutant": str(d.get("dominentpol", "PM2.5")).upper(),
                "pollutants": {
                    "pm25": iaqi.get("pm25", {}).get("v"),
                    "pm10": iaqi.get("pm10", {}).get("v"),
                    "o3": iaqi.get("o3", {}).get("v"),
                    "no2": iaqi.get("no2", {}).get("v"),
                    "so2": iaqi.get("so2", {}).get("v"),
                    "co": iaqi.get("co", {}).get("v"),
                },
                "temperature": iaqi.get("t", {}).get("v"),
                "humidity": iaqi.get("h", {}).get("v"),
                "station_name": city_info.get("name", area_name or f"Area ({lat:.4f}, {lon:.4f})"),
                "station_geo": city_geo or [lat, lon],
                "station_url": city_info.get("url", "https://waqi.info/"),
                "reporting_time": time_info.get("s") or time_info.get("iso", "Real-time"),
                "source": "World Air Quality Index (WAQI)",
                "lat": lat,
                "lon": lon,
                "is_fallback": False,
            }

    # If WAQI returned demo redirect or was unavailable, use global fallback with accurate readings
    fallback = fetch_open_meteo_aqi(lat, lon)
    if fallback:
        if area_name:
            fallback["station_name"] = area_name
        return fallback

    # Never substitute a made-up value for a live AQI observation.
    return {
        "available": False,
        "aqi": None,
        "category": "Unavailable",
        "color": "#64748B",
        "bg_color": "#F1F5F9",
        "border_color": "#CBD5E1",
        "text_color": "#334155",
        "icon": "⚪",
        "description": "No live air quality observation is available for this area right now.",
        "advisory": "Try again later or select another nearby area on the map.",
        "dominant_pollutant": "--",
        "pollutants": {},
        "station_name": area_name or f"Local Area ({lat:.4f}, {lon:.4f})",
        "station_geo": [lat, lon],
        "reporting_time": "Unavailable",
        "source": "WAQI / Open-Meteo",
        "lat": lat,
        "lon": lon,
        "is_fallback": False,
    }
