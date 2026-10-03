"""
Geocoding module for PathFinder (Stage 4).

Provides location searching, address resolution, and reverse-geocoding using
OpenStreetMap Nominatim with intelligent local caching and offline landmark fallback.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from osm_loader import RANCHI_LANDMARKS, parse_coordinates

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "PathFinder-Navigation-App/4.0 (contact: support@pathfinder.local)"

# In-memory session caches to avoid redundant external network requests
_SEARCH_CACHE: Dict[str, List[GeocodedLocation]] = {}
_REVERSE_CACHE: Dict[Tuple[float, float], str] = {}
_LAST_REQUEST_TIME: float = 0.0


@dataclass(frozen=True)
class GeocodedLocation:
    """Represents a geocoded address, place, or landmark."""
    name: str
    display_name: str
    lat: float
    lon: float
    source: str = "nominatim"  # 'landmark', 'nominatim', or 'coordinate'

    def format_short(self) -> str:
        """Return a short readable label."""
        return self.name or self.display_name.split(",")[0]

    def format_display_label(self) -> str:
        """Format an informative, disambiguated label for search dropdowns."""
        if self.source == "coordinate":
            return f"📌 {self.name}"

        source_tag = "OSM Nominatim" if self.source == "nominatim" else self.source.title()
        parts = [p.strip() for p in self.display_name.split(",") if p.strip()]
        sub = ""
        if len(parts) > 1:
            if parts[0].lower() == self.name.lower():
                sub = ", ".join(parts[1:3])
            else:
                sub = ", ".join(parts[:2])
        if sub:
            return f"{self.name} — {sub} ({source_tag})"
        return f"{self.name} ({source_tag})"


def _rate_limit_pause(min_interval: float = 0.5) -> None:
    """Ensure at least min_interval seconds between Nominatim calls per usage policy."""
    global _LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_REQUEST_TIME = time.time()


def search_local_landmarks(query: str) -> List[GeocodedLocation]:
    """Search known local landmarks using case-insensitive substring matching."""
    q = query.strip().lower()
    if not q:
        return []

    results: List[GeocodedLocation] = []
    for name, (lat, lon) in RANCHI_LANDMARKS.items():
        if q in name.lower() or name.lower() in q:
            results.append(
                GeocodedLocation(
                    name=name,
                    display_name=f"{name}, Ranchi, Jharkhand, India",
                    lat=lat,
                    lon=lon,
                    source="landmark",
                )
            )
    return results


def search_locations(
    query: str,
    limit: int = 8,
    focus_ranchi: bool = True,
) -> List[GeocodedLocation]:
    """
    Search for locations matching `query` using OpenStreetMap Nominatim by default,
    with comprehensive offline landmark fallback.

    Nominatim results are prioritized first to provide accurate campus/building
    coordinates and wide geographic coverage. Local landmarks are appended for
    any items not already represented or as an offline fallback.

    Args:
        query: Address, landmark name, or GPS coordinate string.
        limit: Maximum number of suggestions to return.
        focus_ranchi: Prioritize or append Ranchi search context if true.

    Returns:
        List of GeocodedLocation suggestions with Nominatim results at top.
    """
    cleaned = query.strip()
    if not cleaned:
        return []

    # 1. Check if user typed direct GPS coordinates: '23.3699, 85.3253'
    direct_coords = parse_coordinates(cleaned)
    if direct_coords:
        lat, lon = direct_coords
        return [
            GeocodedLocation(
                name=f"GPS ({lat:.4f}, {lon:.4f})",
                display_name=f"Coordinates: Latitude {lat:.5f}, Longitude {lon:.5f}",
                lat=lat,
                lon=lon,
                source="coordinate",
            )
        ]

    # 2. Check local in-memory cache
    cache_key = f"{cleaned.lower()}_{limit}_{focus_ranchi}"
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    nominatim_results: List[GeocodedLocation] = []

    # 3. Query OpenStreetMap Nominatim FIRST for real, accurate geographic data
    if len(cleaned) >= 2:
        search_query = cleaned
        if focus_ranchi and "ranchi" not in cleaned.lower():
            search_query = f"{cleaned}, Ranchi, Jharkhand"

        headers = {"User-Agent": USER_AGENT}
        params = {
            "q": search_query,
            "format": "json",
            "limit": limit,
            "addressdetails": 1,
        }
        if focus_ranchi:
            params["viewbox"] = "85.15,23.50,85.50,23.20"
            params["bounded"] = 0

        try:
            _rate_limit_pause(0.3)
            response = requests.get(
                NOMINATIM_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=3.5,
            )
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    display_name = item.get("display_name", "")
                    name = item.get("name") or display_name.split(",")[0].strip()

                    # Deduplicate internal Nominatim results
                    if not any(abs(r.lat - lat) < 0.0008 and abs(r.lon - lon) < 0.0008 for r in nominatim_results):
                        nominatim_results.append(
                            GeocodedLocation(
                                name=name,
                                display_name=display_name,
                                lat=lat,
                                lon=lon,
                                source="nominatim",
                            )
                        )

            # If 0 results with appended Ranchi, try plain query with viewbox
            if not nominatim_results and focus_ranchi and search_query != cleaned:
                params_retry = {
                    "q": cleaned,
                    "format": "json",
                    "limit": limit,
                    "addressdetails": 1,
                    "viewbox": "85.15,23.50,85.50,23.20",
                    "bounded": 0,
                }
                res2 = requests.get(
                    NOMINATIM_SEARCH_URL,
                    params=params_retry,
                    headers=headers,
                    timeout=3.0,
                )
                if res2.status_code == 200:
                    for item in res2.json():
                        lat = float(item["lat"])
                        lon = float(item["lon"])
                        display_name = item.get("display_name", "")
                        name = item.get("name") or display_name.split(",")[0].strip()
                        if not any(abs(r.lat - lat) < 0.0008 and abs(r.lon - lon) < 0.0008 for r in nominatim_results):
                            nominatim_results.append(
                                GeocodedLocation(
                                    name=name,
                                    display_name=display_name,
                                    lat=lat,
                                    lon=lon,
                                    source="nominatim",
                                )
                            )
        except (requests.RequestException, ValueError, KeyError):
            # Graceful network degradation: rely on local matches without crashing
            pass

    # 4. Nominatim results take top priority
    results: List[GeocodedLocation] = list(nominatim_results)

    # 5. Append local landmark matches for anything not already covered
    local_matches = search_local_landmarks(cleaned)
    for lm in local_matches:
        # Check if already covered by a Nominatim result (proximity < 500m or name match)
        is_covered = any(
            (abs(r.lat - lm.lat) < 0.005 and abs(r.lon - lm.lon) < 0.005) or
            (r.name.lower() == lm.name.lower())
            for r in nominatim_results
        )
        if not is_covered:
            results.append(lm)

    # Cache and return up to limit
    final_results = results[:limit]
    _SEARCH_CACHE[cache_key] = final_results
    return final_results


def reverse_geocode(lat: float, lon: float) -> str:
    """
    Resolve (lat, lon) coordinates to a human-readable street or area name.
    
    Falls back to local landmark matching or formatted coordinate string.
    """
    rounded_key = (round(lat, 5), round(lon, 5))
    if rounded_key in _REVERSE_CACHE:
        return _REVERSE_CACHE[rounded_key]

    # Check local landmark proximity (< 80 meters)
    for name, (lm_lat, lm_lon) in RANCHI_LANDMARKS.items():
        if abs(lat - lm_lat) < 0.0008 and abs(lon - lm_lon) < 0.0008:
            _REVERSE_CACHE[rounded_key] = name
            return name

    # Try Nominatim reverse lookup
    headers = {"User-Agent": USER_AGENT}
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
    }

    try:
        _rate_limit_pause(0.5)
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params=params,
            headers=headers,
            timeout=3.0,
        )
        if response.status_code == 200:
            data = response.json()
            name = data.get("name") or data.get("display_name", "")
            if name:
                # Format to a compact label
                parts = [p.strip() for p in name.split(",")[:3] if p.strip()]
                compact = ", ".join(parts)
                _REVERSE_CACHE[rounded_key] = compact
                return compact
    except (requests.RequestException, ValueError, KeyError):
        pass

    fallback = f"Location ({lat:.4f}, {lon:.4f})"
    _REVERSE_CACHE[rounded_key] = fallback
    return fallback
