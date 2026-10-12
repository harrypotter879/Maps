"""
Wikipedia Information Provider for PathFinder Navigation App.

Fetches location-based information from the Wikipedia API strictly grounded
in geographic coordinates (lat, lon) and locality:
- Name of the place / locality
- Picture (if available)
- Description / Summary
- History (from historical sections)
- Direct Wikipedia link
"""

from __future__ import annotations
import json
import math
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

WIKI_USER_AGENT = "PathFinder-Navigation-App/4.0 (contact: support@pathfinder.local)"
REQUEST_TIMEOUT = 4.5  # seconds

PERSON_KEYWORDS = {
    "soldier", "awardee", "recipient", "player", "actor", "actress", "politician",
    "cricketer", "singer", "author", "writer", "physician", "person", "human",
    "footballer", "general", "officer", "monarch", "emperor", "king", "queen",
    "president", "prime minister", "minister", "governor", "activist", "leader",
    "saint", "poet", "artist", "scientist", "engineer", "philosopher", "biography",
    "martyr", "warrior", "colonel", "commander", "major", "captain", "lieutenant",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in kilometers between two GPS points."""
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _is_person(desc: str, extract: str) -> bool:
    """Check if a Wikipedia page description or extract refers to a person/biography."""
    d_low = (desc or "").lower()
    for w in PERSON_KEYWORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", d_low):
            return True
    e_low = (extract or "")[:150].lower()
    if re.search(r"\b(was an?|is an?|born \d|was born|died \d|killed in action)\b", e_low) and any(
        w in e_low for w in ["soldier", "awardee", "actor", "politician", "cricketer", "officer", "son of", "daughter of", "pvc"]
    ):
        return True
    return False


def _title_matches_place_name(title: str, primary_name: str) -> bool:
    """Verify that a candidate article title actually matches the searched place name."""
    t_low = title.lower()
    p_low = primary_name.lower().strip()
    if not p_low:
        return False
    if p_low in t_low:
        return True
    words = [w for w in re.split(r"[\s,\-_]+", p_low) if len(w) > 2]
    matched = [w for w in words if w in t_low]
    if len(words) >= 2 and len(matched) >= 2:
        return True
    if len(words) == 1 and len(matched) == 1:
        return True
    return False


def _clean_html_to_text(raw_html: str) -> str:
    """Strip HTML tags, styles, scripts, tables, edit links, and reference citations."""
    if not raw_html:
        return ""
    text = re.sub(r"<(script|style|table|sup)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 20]
    return "\n\n".join(lines[:5])


def _extract_history_text(title: str) -> str:
    """Fetch and parse the History, Origin, or Construction section of a Wikipedia article."""
    quoted_title = urllib.parse.quote(title.replace(" ", "_"))
    sec_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={quoted_title}&prop=sections&format=json"
    headers = {"User-Agent": WIKI_USER_AGENT}
    req = urllib.request.Request(sec_url, headers=headers)

    target_idx = None
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            sections = data.get("parse", {}).get("sections", [])
            for s in sections:
                line_lower = s.get("line", "").lower()
                if any(k in line_lower for k in [
                    "history", "origin", "origins", "construction",
                    "background", "historical", "etymology", "establishment",
                    "founding", "early years", "development", "heritage"
                ]):
                    target_idx = s.get("index")
                    break
    except Exception:
        return ""

    if not target_idx:
        return ""

    hist_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={quoted_title}&section={target_idx}&prop=text&format=json"
    req_hist = urllib.request.Request(hist_url, headers=headers)
    try:
        with urllib.request.urlopen(req_hist, timeout=REQUEST_TIMEOUT) as resp:
            hdata = json.loads(resp.read().decode("utf-8"))
            raw_html = hdata.get("parse", {}).get("text", {}).get("*", "")
            return _clean_html_to_text(raw_html)
    except Exception:
        return ""


def get_locality_from_coords(lat: float, lon: float) -> Tuple[str, str, str, str]:
    """
    Reverse-geocode coordinates to identify city, state/county, country, and display address.
    """
    headers = {"User-Agent": WIKI_USER_AGENT}
    try:
        rev_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat:.6f}&lon={lon:.6f}&format=json&addressdetails=1"
        req = urllib.request.Request(rev_url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or addr.get("suburb")
                or addr.get("county")
                or ""
            )
            state = addr.get("state") or addr.get("state_district") or ""
            country = addr.get("country") or ""
            display = data.get("display_name", "")
            return city, state, country, display
    except Exception:
        return "", "", "", ""


def _resolve_location_article(
    name: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    address: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the most accurate Wikipedia article strictly grounded in the geographic location.
    Prioritizes specific place/neighborhood/landmark articles at the location,
    while filtering out non-geographic namesakes, biographical articles of people,
    and distant landmarks with identical names.
    """
    headers = {"User-Agent": WIKI_USER_AGENT}
    clean_name = (name or "").strip()
    clean_name = re.sub(r"^[^\w\s]+", "", clean_name).strip()
    for prefix in ["my current location", "bookmark:", "bookmark", "destination:", "origin:", "searched:"]:
        if clean_name.lower().startswith(prefix):
            clean_name = clean_name[len(prefix):].strip(" -:()")
            break
    primary_name = clean_name.split("—")[0].split(",")[0].strip()
    if re.match(r"^[-+]?\d+(\.\d+)?[,\s]+[-+]?\d+(\.\d+)?$", primary_name):
        primary_name = ""

    # If coordinates are missing, fallback to search by name
    if lat is None or lon is None:
        if not primary_name:
            return None
        try:
            s_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(primary_name)}&srlimit=3&format=json"
            req = urllib.request.Request(s_url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                sdata = json.loads(resp.read().decode("utf-8"))
                res = sdata.get("query", {}).get("search", [])
                if res:
                    return res[0]["title"]
        except Exception:
            pass
        return None

    # Step 1: Query Wikipedia Geosearch near coordinates (max radius 10,000 meters)
    geo_results: List[Dict[str, Any]] = []
    try:
        geo_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=geosearch"
            f"&gscoord={lat:.6f}|{lon:.6f}&gsradius=10000&gslimit=25&format=json"
        )
        req = urllib.request.Request(geo_url, headers=headers)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            gdata = json.loads(resp.read().decode("utf-8"))
            geo_results = gdata.get("query", {}).get("geosearch", [])
    except Exception:
        pass

    clean_p_lower = primary_name.lower()

    # Priority 1: Exact or landmark match within geosearch results (dist < 10km)
    if clean_p_lower:
        for r in geo_results:
            t_low = r["title"].lower()
            if t_low == clean_p_lower or f"{clean_p_lower}," in t_low or f"({clean_p_lower})" in t_low:
                return r["title"]

        # Also check partial match for landmark if within 1.5 km
        for r in geo_results:
            t_low = r["title"].lower()
            if len(clean_p_lower) > 3 and clean_p_lower in t_low and r.get("dist", 99999) < 1500:
                return r["title"]

    # Step 2: Extract locality information (city, state, country, address parts)
    addr_parts = [p.strip() for p in address.split(",") if p.strip()] if address else []
    city, state, country, _ = get_locality_from_coords(lat, lon)
    if not city and addr_parts:
        cleaned_parts = [p for p in addr_parts if not re.match(r"^\d+$", p)]
        if len(cleaned_parts) >= 3:
            city = cleaned_parts[-3]
            state = cleaned_parts[-2]
            country = cleaned_parts[-1]
        elif cleaned_parts:
            city = cleaned_parts[0]

    # Step 3: Specific Place Search on Wikipedia (for neighborhoods, colonies, POIs without geosearch coords)
    if clean_p_lower and len(clean_p_lower) > 2 and clean_p_lower not in [city.lower(), state.lower(), country.lower()]:
        search_queries = []
        if city:
            search_queries.append(f'"{primary_name}" {city}')
        search_queries.append(f'"{primary_name}"')
        if city:
            search_queries.append(f"{primary_name} {city}")

        seen_titles = set()
        candidates: List[Dict[str, Any]] = []
        for sq in search_queries:
            s_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(sq)}&srlimit=5&format=json"
            req = urllib.request.Request(s_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    s_data = json.loads(resp.read().decode("utf-8")).get("query", {}).get("search", [])
                    for item in s_data:
                        if item["title"] not in seen_titles:
                            seen_titles.add(item["title"])
                            candidates.append(item)
            except Exception:
                pass
            if len(candidates) >= 5:
                break

        if candidates:
            # Batch fetch coordinates and short description for candidates
            t_str = "|".join([urllib.parse.quote(c["title"]) for c in candidates[:6]])
            p_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=coordinates|pageprops&titles={t_str}&format=json"
            req_p = urllib.request.Request(p_url, headers=headers)
            pages: Dict[str, Any] = {}
            try:
                with urllib.request.urlopen(req_p, timeout=REQUEST_TIMEOUT) as resp:
                    pages = json.loads(resp.read().decode("utf-8")).get("query", {}).get("pages", {})
            except Exception:
                pass

            p_map = {p.get("title"): p for p in pages.values()}
            for cand in candidates:
                cand_title = cand["title"]
                if not _title_matches_place_name(cand_title, primary_name):
                    continue

                p_info = p_map.get(cand_title, {})
                desc = p_info.get("pageprops", {}).get("wikibase-shortdesc", "")
                snippet = cand.get("snippet", "")
                if _is_person(desc, snippet):
                    # Filter out biographical articles (e.g. soldier Albert Ekka)
                    continue

                coords = p_info.get("coordinates", [{}])[0]
                c_lat, c_lon = coords.get("lat"), coords.get("lon")
                if c_lat is not None and c_lon is not None:
                    # Verified coordinate match: must be near the searched place
                    dist = _haversine_km(lat, lon, c_lat, c_lon)
                    if dist <= 35.0:
                        return cand_title
                    else:
                        # Remote namesake (e.g. London clock tower when searching in Philippines)
                        continue
                else:
                    # Article without coordinates on Wikipedia (e.g. Harmu Housing Colony, Ranchi):
                    # Must verify that locality terms (city, state, country, or address) appear in title/desc/snippet
                    cand_lower = cand_title.lower()
                    loc_terms = [city.lower(), state.lower(), country.lower()] + [p.lower() for p in addr_parts if len(p) > 2]
                    has_loc = any(
                        loc in cand_lower or loc in desc.lower() or loc in snippet.lower()
                        for loc in loc_terms
                        if loc and loc not in ["india", "philippines", "australia", "united states", "united kingdom"]
                    )
                    if has_loc:
                        return cand_title

    # Step 4: Fallback to Locality / City (when searched place has no dedicated article, or searching city directly)
    # Check address tokens against geosearch results
    for part in addr_parts:
        part_lower = part.lower()
        if len(part_lower) <= 2 or re.match(r"^\d+$", part_lower):
            continue
        for r in geo_results:
            t_low = r["title"].lower()
            if t_low == part_lower or f"{part_lower}," in t_low or f"({part_lower})" in t_low:
                return r["title"]

    if city:
        city_lower = city.lower()
        for r in geo_results:
            t_low = r["title"].lower()
            if t_low == city_lower or f"{city_lower}," in t_low or f"({city_lower})" in t_low:
                return r["title"]

    if state:
        state_lower = state.lower()
        for r in geo_results:
            t_low = r["title"].lower()
            if t_low == state_lower or f"{state_lower}," in t_low:
                return r["title"]

    # Fallback search Wikipedia for City + Country
    if city:
        query_loc = f"{city}, {country}" if country else city
        try:
            s_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query_loc)}&srlimit=3&format=json"
            req = urllib.request.Request(s_url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                sdata = json.loads(resp.read().decode("utf-8"))
                res = sdata.get("query", {}).get("search", [])
                if res:
                    return res[0]["title"]
        except Exception:
            pass

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wiki_info_for_location(
    name: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch Wikipedia information based on place name and/or GPS coordinates.
    Strictly verifies that results correspond to the actual geographic location.
    """
    headers = {"User-Agent": WIKI_USER_AGENT}
    selected_title = _resolve_location_article(name, lat, lon, address)

    disp_name = (name or "").strip() or (f"Location ({lat:.4f}, {lon:.4f})" if lat and lon else "Unknown Place")

    if not selected_title:
        search_query_encoded = urllib.parse.quote(disp_name)
        return {
            "name": disp_name,
            "searched_name": disp_name,
            "picture": None,
            "description": f"No direct Wikipedia article was found matching '{disp_name}' at this location. You can search Wikipedia directly using the link below.",
            "history": "No dedicated history section recorded for this location on Wikipedia.",
            "wiki_link": f"https://en.wikipedia.org/w/index.php?search={search_query_encoded}",
            "found": False,
        }

    # Fetch REST Summary for selected article
    quoted_title = urllib.parse.quote(selected_title.replace(" ", "_"))
    summary_data: Dict[str, Any] = {}
    try:
        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quoted_title}"
        req = urllib.request.Request(sum_url, headers=headers)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            summary_data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass

    place_name = summary_data.get("title") or selected_title
    picture_url = (
        summary_data.get("originalimage", {}).get("source")
        or summary_data.get("thumbnail", {}).get("source")
    )
    description = summary_data.get("extract") or summary_data.get("description") or "No description available."
    wiki_link = (
        summary_data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{quoted_title}"
    )

    # Fetch History section
    history_text = _extract_history_text(selected_title)
    if not history_text:
        history_text = "No dedicated history section is recorded for this location on Wikipedia. Historical context is included in the overview above."

    return {
        "name": place_name,
        "searched_name": disp_name,
        "picture": picture_url,
        "description": description,
        "history": history_text,
        "wiki_link": wiki_link,
        "found": True,
    }
