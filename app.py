"""
PathFinder — Interactive Real-World Navigation UI.
Built with Streamlit, Folium, and OpenStreetMap (OSMnx & Nominatim).

Features:
- Clean startup map with zero pre-loaded directions or cluttered routes
- Dedicated Top Search Bar with live autocomplete suggestions and instant pin drops
- Current Location support: glowing blue dot marker (🔵 My Location) and browser GPS locate control
- Single-click routing from/to My Current Location
- Sliding 'Go' Directions Drawer triggered by a single button at bottom-left
- Origin / Destination selection supporting custom search, landmarks, and map clicks
- Interactive Folium map with road-geometry routes and click-to-select endpoints
- User-friendly navigation metrics: Total Distance, Estimated Drive Time, Estimated Walk Time
- Turn-by-turn guidance corridor itinerary
- 🚦 Traffic-Aware Route Estimation: departure-time and road-class estimates provide a
  travel-time routing objective without changing physical road distances
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import random
from datetime import time
from typing import Any, Dict, List, Optional, Tuple

import folium
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from graph import Graph, create_sample_road_network
from astar import find_shortest_path_astar, find_alternative_routes
from dijkstra import PathResult
from osm_loader import get_ranchi_road_network, RANCHI_LANDMARKS
from geocoder import search_locations, reverse_geocode, GeocodedLocation
from map_view import build_folium_map, build_empty_map
from display import format_distance
from worldwide_router import fetch_worldwide_route, WORLDWIDE_LANDMARKS, is_in_ranchi
from traffic import (
    TRAFFIC_ICONS, TRAFFIC_MODE_LABELS, HIGH, MEDIUM, LOW,
    SHORTEST_DISTANCE, TRAFFIC_AWARE, PERIOD_LABELS,
    analyse_route, apply_traffic_mode, normalize_departure_time, get_traffic_period,
)
from wiki_info import fetch_wiki_info_for_location
from aqi_service import fetch_waqi_reading, get_aqi_category, AQI_CATEGORIES, get_waqi_api_token

MY_LOCATION_LABEL = "🔵 My Current Location"
MAX_BOOKMARKS = 20

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BOOKMARKS_FILE = os.path.join(DATA_DIR, "bookmarks.json")
_browser_geolocation = components.declare_component(
    "pathfinder_browser_geolocation",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_geolocation"),
)


def _enable_aqi_mode() -> None:
    """Start an AQI lookup using browser GPS when the AQI control is activated."""
    st.session_state.show_aqi_mode = True
    st.session_state.aqi_inspect_coords = None
    st.session_state.aqi_location_pending = True
    st.session_state.aqi_location_error = None
    st.session_state.aqi_location_request_id = st.session_state.get("aqi_location_request_id", 0) + 1


def _on_sidebar_aqi_toggle() -> None:
    """Keep the sidebar switch and AQI location flow in sync."""
    if st.session_state.get("sidebar_aqi_toggle"):
        if not st.session_state.get("show_aqi_mode"):
            _enable_aqi_mode()
    else:
        st.session_state.show_aqi_mode = False
        st.session_state.aqi_location_pending = False


def _sync_aqi_browser_location() -> None:
    """Receive the browser's GPS result and make it the active AQI location."""
    if not st.session_state.get("show_aqi_mode") or not st.session_state.get("aqi_location_pending"):
        return

    request_id = st.session_state.get("aqi_location_request_id", 0)
    result = _browser_geolocation(key=f"aqi-browser-geolocation-{request_id}", default=None)
    if not isinstance(result, dict):
        return

    lat, lon = result.get("latitude"), result.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            location = (float(lat), float(lon), "My Current Location")
            st.session_state.my_location = location
            st.session_state.aqi_inspect_coords = location
            st.session_state.aqi_location_pending = False
            st.session_state.aqi_location_error = None
            return

    if result.get("error"):
        st.session_state.aqi_location_pending = False
        st.session_state.aqi_location_error = result["error"]


def load_bookmarks(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load saved bookmarks from JSON file on the user's device."""
    target_path = filepath or BOOKMARKS_FILE
    if not os.path.exists(target_path):
        return []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            valid_bms: List[Dict[str, Any]] = []
            for item in data[:MAX_BOOKMARKS]:
                if isinstance(item, dict) and "name" in item and "lat" in item and "lon" in item:
                    valid_bms.append({
                        "name": str(item["name"]).strip(),
                        "address": str(item.get("address", item["name"])).strip(),
                        "lat": float(item["lat"]),
                        "lon": float(item["lon"]),
                    })
            return valid_bms
    except Exception as e:
        print(f"Error loading bookmarks from {target_path}: {e}")
    return []


def save_bookmarks(bookmarks: List[Dict[str, Any]], filepath: Optional[str] = None) -> bool:
    """Save bookmarks in JSON format on the user's device in real time."""
    target_path = filepath or BOOKMARKS_FILE
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(bookmarks[:MAX_BOOKMARKS], f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving bookmarks to {target_path}: {e}")
        return False


def add_bookmark(name: str, address: str, lat: float, lon: float, filepath: Optional[str] = None) -> Tuple[bool, str]:
    """Add a new bookmark, enforcing the maximum limit of 20 bookmarks and saving to JSON on device in real time."""
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = load_bookmarks(filepath)

    if len(st.session_state.bookmarks) >= MAX_BOOKMARKS:
        return False, f"Bookmark limit reached (maximum {MAX_BOOKMARKS} allowed). Delete an existing bookmark to add a new one."

    cleaned_name = name.strip()
    if not cleaned_name:
        return False, "Bookmark name cannot be empty."

    if any(b.get("name", "").lower() == cleaned_name.lower() for b in st.session_state.bookmarks):
        return False, f"A bookmark named '{cleaned_name}' already exists. Please choose a different name."

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return False, "Invalid GPS coordinates for bookmark."

    new_bm = {
        "name": cleaned_name,
        "address": address.strip() if address else cleaned_name,
        "lat": lat_f,
        "lon": lon_f,
    }
    st.session_state.bookmarks.append(new_bm)
    # Persist in JSON format on the user's device in real time
    save_bookmarks(st.session_state.bookmarks, filepath)
    return True, f"Bookmark '{cleaned_name}' saved to device."


def delete_bookmark(index: int, filepath: Optional[str] = None) -> Tuple[bool, str]:
    """Delete a bookmark at the specified index and update JSON file on device in real time."""
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = load_bookmarks(filepath)

    if not (0 <= index < len(st.session_state.bookmarks)):
        return False, "Invalid bookmark index."

    deleted = st.session_state.bookmarks.pop(index)
    # Update JSON file on the user's device in real time
    save_bookmarks(st.session_state.bookmarks, filepath)
    return True, f"Bookmark '{deleted.get('name', 'selected')}' deleted."


def get_bookmarks() -> List[Dict[str, Any]]:
    """Return the list of saved bookmarks in session state."""
    return st.session_state.get("bookmarks", [])


# Page Configuration - Collapsed sidebar gives full width to the map!
st.set_page_config(
    page_title="PathFinder — Map Navigator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Cached Road Network Loaders
@st.cache_resource(show_spinner="Loading OpenStreetMap Road Network...")
def load_cached_osm_network() -> Graph:
    """Load the pre-cached Ranchi OpenStreetMap road network."""
    return get_ranchi_road_network()


@st.cache_resource
def load_fictional_network() -> Graph:
    """Load the Stage 1 fictional regional road network."""
    return create_sample_road_network()


def estimate_travel_time(distance_m: float, speed_kmh: float = 30.0) -> str:
    """
    Calculate and format realistic travel time based on distance and average speed.
    - Driving: ~30 km/h in city traffic
    - Walking: ~4.5 km/h
    """
    if distance_m <= 0:
        return "0 mins"
    speed_mps = (speed_kmh * 1000.0) / 3600.0
    minutes = round((distance_m / speed_mps) / 60.0)
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{minutes} min{'s' if minutes != 1 else ''}"
    else:
        h = minutes // 60
        m = minutes % 60
        return f"{h} hr{'s' if h != 1 else ''}" if m == 0 else f"{h} hr {m} min"


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_aqi_reading(lat: float, lon: float, token: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch and cache real-time AQI observation metrics from WAQI API."""
    return fetch_waqi_reading(lat, lon, token=token, area_name=name)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "my_location" not in st.session_state:
        # Default My Current Location set to Kairali School, Ranchi
        st.session_state.my_location = (23.3191843, 85.2987681, "Kairali School, Ranchi")

    # Start and destination locations are completely empty by default
    if "start_name" not in st.session_state:
        st.session_state.start_name = None
        st.session_state.start_coords = None
        st.session_state.start_address = None

    if "dest_name" not in st.session_state:
        st.session_state.dest_name = None
        st.session_state.dest_coords = None
        st.session_state.dest_address = None

    if "searched_address" not in st.session_state:
        st.session_state.searched_address = None

    if "prev_search_suggestion" not in st.session_state:
        st.session_state.prev_search_suggestion = None

    # Clear active route if endpoints are incomplete
    if not st.session_state.get("start_coords") or not st.session_state.get("dest_coords"):
        if st.session_state.get("network_mode", "").startswith("Real-World"):
            st.session_state.route_result = None
            st.session_state.alternative_results = None

    if "origin_alternatives" not in st.session_state:
        st.session_state.origin_alternatives = []
    if "dest_alternatives" not in st.session_state:
        st.session_state.dest_alternatives = []
    if "origin_status_msg" not in st.session_state:
        st.session_state.origin_status_msg = None
    if "dest_status_msg" not in st.session_state:
        st.session_state.dest_status_msg = None

    # NO route by default at startup! Clean map until user requests directions
    if "route_result" not in st.session_state:
        st.session_state.route_result = None
    if "alternative_results" not in st.session_state:
        st.session_state.alternative_results = None
    if "active_route_index" not in st.session_state:
        st.session_state.active_route_index = 0

    if "searched_location" not in st.session_state:
        st.session_state.searched_location = None  # (lat, lon, name)

    if "worldwide_graph" not in st.session_state:
        st.session_state.worldwide_graph = None
    if "is_worldwide_active" not in st.session_state:
        st.session_state.is_worldwide_active = False

    # Route menu drawer is closed by default at startup!
    if "show_route_menu" not in st.session_state:
        st.session_state.show_route_menu = False

    # Bookmarks panel persists on user device in JSON format across refreshes (max 20)
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = load_bookmarks()
    if "show_bookmarks" not in st.session_state:
        st.session_state.show_bookmarks = False

    # Wikipedia location information right panel is closed by default at startup!
    if "show_info_panel" not in st.session_state:
        st.session_state.show_info_panel = False
    if "info_panel_target" not in st.session_state:
        st.session_state.info_panel_target = "destination"

    if "last_clicked_coords" not in st.session_state:
        st.session_state.last_clicked_coords = None

    # Interactive Click-to-Route state (Zero-typing direct map interaction)
    if "click_to_route_mode" not in st.session_state:
        st.session_state.click_to_route_mode = True
    if "last_handled_click" not in st.session_state:
        st.session_state.last_handled_click = None
    if "click_route_step" not in st.session_state:
        st.session_state.click_route_step = "start"
    if "map_height_setting" not in st.session_state:
        st.session_state.map_height_setting = 700
    if "click_behavior" not in st.session_state:
        st.session_state.click_behavior = "update_dest"

    valid_modes = [
        "Real-World Map (Worldwide Search, Routing & Ranchi Traffic)",
        "Fictional Regional Map (Stage 1)",
    ]
    if "network_mode" not in st.session_state or st.session_state.network_mode not in valid_modes:
        st.session_state.network_mode = valid_modes[0]

    # Shortest-distance is the default; departure time is a model input.
    if "traffic_choice" not in st.session_state:
        st.session_state.traffic_choice = SHORTEST_DISTANCE
    elif st.session_state.traffic_choice not in (SHORTEST_DISTANCE, TRAFFIC_AWARE):
        # Migrate the previous UI's stored radio values for existing sessions.
        old_choice = st.session_state.traffic_choice
        st.session_state.traffic_choice = SHORTEST_DISTANCE if old_choice == "off" else TRAFFIC_AWARE
    if "departure_time" not in st.session_state:
        st.session_state.departure_time = time(8, 30)
    if "time_preset" not in st.session_state:
        st.session_state.time_preset = "Morning Peak · 08:30"
    # Keep the visible 12-hour controls in step with the model's 24-hour value.
    departure = normalize_departure_time(st.session_state.departure_time)
    if "departure_hour_12" not in st.session_state:
        st.session_state.departure_hour_12 = departure.hour % 12 or 12
    if "departure_minute" not in st.session_state:
        st.session_state.departure_minute = departure.minute
    if "departure_ampm" not in st.session_state:
        st.session_state.departure_ampm = "PM" if departure.hour >= 12 else "AM"
    preset_labels = {
        time(8, 30): "Morning Peak · 08:30",
        time(13, 0): "Midday · 01:00 PM",
        time(14, 0): "School Dismissal · 02:00 PM",
        time(18, 0): "Evening Peak · 06:00 PM",
        time(22, 0): "Night · 10:00 PM",
    }
    # Repair old sessions where a preset label and the actual departure time drifted apart.
    st.session_state.time_preset = preset_labels.get(departure, "Custom time")
    if "baseline_result" not in st.session_state:
        st.session_state.baseline_result = None  # shortest-by-distance route, for comparison
    if "traffic_notice" not in st.session_state:
        st.session_state.traffic_notice = None

    # Air Quality (WAQI) Mode: off by default; directly activated via button
    if "show_aqi_mode" not in st.session_state:
        st.session_state.show_aqi_mode = False
    if "sidebar_aqi_toggle" in st.session_state:
        st.session_state.sidebar_aqi_toggle = st.session_state.show_aqi_mode
    if "aqi_inspect_coords" not in st.session_state:
        st.session_state.aqi_inspect_coords = None
    if "aqi_location_pending" not in st.session_state:
        st.session_state.aqi_location_pending = False
    if "aqi_location_error" not in st.session_state:
        st.session_state.aqi_location_error = None
    if "waqi_custom_token" not in st.session_state:
        st.session_state.waqi_custom_token = ""


def get_traffic_mode() -> Optional[str]:
    """Return the active route objective for the Ranchi network."""
    choice = st.session_state.get("traffic_choice", SHORTEST_DISTANCE)
    return TRAFFIC_AWARE if choice == TRAFFIC_AWARE else None


def compute_routes(is_realworld: bool) -> None:
    """
    Run route search and store results in session state.
    Supports:
    1. Global Worldwide road navigation (OSRM) across any city/country.
    2. Ranchi Local road network (A* / Dijkstra + Traffic estimation).
    3. Fictional regional network (Stage 1).
    """
    if not is_realworld:
        graph = load_fictional_network()
        start_node, dest_node = st.session_state.start_name, st.session_state.dest_name
        if not start_node or not dest_node:
            st.warning("Please choose both a start town and a destination town.")
            return
        alts = find_alternative_routes(graph, start_node, dest_node, max_routes=3)
        if alts:
            st.session_state.alternative_results = alts
            st.session_state.route_result = alts[0]
            st.session_state.active_route_index = 0
        else:
            st.session_state.alternative_results = []
            st.session_state.route_result = find_shortest_path_astar(graph, start_node, dest_node)
        st.session_state.baseline_result = None
        st.session_state.is_worldwide_active = False
        st.session_state.worldwide_graph = None
        st.session_state.info_panel_target = "destination"
        return

    # Real-world routing
    if not st.session_state.start_coords or not st.session_state.dest_coords:
        st.warning("Please select both an Origin and a Destination.")
        return

    start_lat, start_lon = st.session_state.start_coords
    dest_lat, dest_lon = st.session_state.dest_coords
    start_in_ranchi = is_in_ranchi(start_lat, start_lon)
    dest_in_ranchi = is_in_ranchi(dest_lat, dest_lon)

    # Use Ranchi local graph & traffic simulation whenever both points are in Ranchi
    if start_in_ranchi and dest_in_ranchi:
        mode = get_traffic_mode()
        departure = normalize_departure_time(st.session_state.get("departure_time"))
        graph = load_cached_osm_network()
        start_node, _ = graph.find_nearest_node(start_lat, start_lon)
        dest_node, _ = graph.find_nearest_node(dest_lat, dest_lon)
        cost_fn = apply_traffic_mode(mode, departure, graph=graph)

        alts = find_alternative_routes(graph, start_node, dest_node, max_routes=3, cost_function=cost_fn)
        if alts:
            st.session_state.alternative_results = alts
            st.session_state.route_result = alts[0]
            st.session_state.active_route_index = 0
        else:
            st.session_state.alternative_results = []
            st.session_state.route_result = find_shortest_path_astar(
                graph, start_node, dest_node, cost_function=cost_fn
            )
        st.session_state.baseline_result = (
            find_shortest_path_astar(graph, start_node, dest_node) if mode else None
        )
        st.session_state.is_worldwide_active = False
        st.session_state.worldwide_graph = None
    else:
        # Route via Worldwide Navigation Engine (OSRM)
        try:
            w_graph, w_results = fetch_worldwide_route(
                (start_lat, start_lon),
                (dest_lat, dest_lon),
                start_label=st.session_state.start_name,
                dest_label=st.session_state.dest_name,
            )
            st.session_state.worldwide_graph = w_graph
            st.session_state.is_worldwide_active = True
            st.session_state.baseline_result = None

            if w_results and w_results[0].found:
                st.session_state.alternative_results = w_results
                st.session_state.route_result = w_results[0]
                st.session_state.active_route_index = 0
            else:
                st.session_state.alternative_results = []
                st.session_state.route_result = w_results[0] if w_results else None
        except Exception as e:
            st.session_state.worldwide_graph = None
            st.session_state.is_worldwide_active = False
            st.session_state.alternative_results = []
            st.session_state.route_result = None
            st.warning(f"Worldwide navigation notice: {e}")

    # Set destination as info panel target without forcing panel to squeeze the giant map
    st.session_state.info_panel_target = "destination"


def on_routing_settings_change() -> None:
    """Recalculate an active route after its mode or departure time changes."""
    is_realworld = "Real-World" in st.session_state.get("network_mode", "Real-World")
    if st.session_state.get("is_worldwide_active"):
        return
    mode = get_traffic_mode()
    previous = st.session_state.route_result
    st.session_state.traffic_notice = None

    if not is_realworld or previous is None or not previous.found:
        # No active route to update; the map overlay alone changes
        st.session_state.traffic_notice = {"kind": "mode_only", "mode": mode}
        return

    try:
        graph = load_cached_osm_network()
        prev_info = analyse_route(graph, previous.path, mode, st.session_state.departure_time)
        compute_routes(is_realworld)
        new = st.session_state.route_result
        if new is None or not new.found:
            st.session_state.traffic_notice = {"kind": "no_route", "mode": mode}
            return
        st.session_state.traffic_notice = {
            "kind": "route",
            "mode": mode,
            "changed": new.path != previous.path,
            "prev_distance": previous.total_distance,
            "new_distance": new.total_distance,
            "prev_high": prev_info.high_roads if prev_info else 0,
            "prev_cost": previous.total_cost,
            "new_cost": new.total_cost,
        }
    except Exception as e:  # never crash the UI because of traffic
        st.session_state.traffic_notice = {"kind": "error", "mode": mode, "message": str(e)}


def _fmt_dist(meters: float) -> str:
    return f"{meters/1000:.2f} km" if meters >= 1000 else f"{meters:.0f} m"


def _apply_time_preset() -> None:
    presets = {
        "Morning Peak · 08:30": time(8, 30),
        "Midday · 01:00 PM": time(13, 0),
        "School Dismissal · 02:00 PM": time(14, 0),
        "Evening Peak · 06:00 PM": time(18, 0),
        "Night · 10:00 PM": time(22, 0),
    }
    selected = presets.get(st.session_state.time_preset)
    if selected is None:  # Custom time is selected by editing the clock controls.
        return
    st.session_state.departure_time = selected
    st.session_state.departure_hour_12 = selected.hour % 12 or 12
    st.session_state.departure_minute = selected.minute
    st.session_state.departure_ampm = "PM" if selected.hour >= 12 else "AM"
    on_routing_settings_change()


def _apply_manual_departure_time() -> None:
    """Convert the visible 12-hour clock to the model's 24-hour time."""
    hour_12 = int(st.session_state.departure_hour_12)
    minute = int(st.session_state.departure_minute)
    hour_24 = (hour_12 % 12) + (12 if st.session_state.departure_ampm == "PM" else 0)
    selected = time(hour_24, minute)
    st.session_state.departure_time = selected

    labels_by_time = {
        time(8, 30): "Morning Peak · 08:30",
        time(13, 0): "Midday · 01:00 PM",
        time(14, 0): "School Dismissal · 02:00 PM",
        time(18, 0): "Evening Peak · 06:00 PM",
        time(22, 0): "Night · 10:00 PM",
    }
    st.session_state.time_preset = labels_by_time.get(selected, "Custom time")
    on_routing_settings_change()


def render_aqi_dashboard_card() -> Optional[Dict[str, Any]]:
    """
    Render a Google Maps style real-time Air Quality dashboard banner
    featuring WAQI reading, severity color badge, health recommendation,
    pollutant readings breakdown, and an instant close button.
    """
    inspect_tuple = st.session_state.get("aqi_inspect_coords")
    if not inspect_tuple:
        if st.session_state.get("searched_location"):
            s_lat, s_lon, s_name = st.session_state.searched_location
            inspect_tuple = (s_lat, s_lon, s_name)
        elif st.session_state.get("start_coords"):
            inspect_tuple = (st.session_state.start_coords[0], st.session_state.start_coords[1], st.session_state.start_name or "Origin")
        else:
            inspect_tuple = st.session_state.my_location
        st.session_state.aqi_inspect_coords = inspect_tuple

    lat, lon, label = inspect_tuple
    token = st.session_state.get("waqi_custom_token", "")
    aqi_data = get_cached_aqi_reading(lat, lon, token=token, name=label)

    if not aqi_data.get("available", True):
        st.warning(
            f"Live air quality data isn't available for {label} right now. "
            "Please try again in a moment."
        )
        return aqi_data

    aqi_val = aqi_data.get("aqi", 0)
    category = aqi_data.get("category", "Moderate")
    color = aqi_data.get("color", "#EAB308")
    bg_color = aqi_data.get("bg_color", "#FEF9C3")
    border_color = aqi_data.get("border_color", "#FDE047")
    text_color = aqi_data.get("text_color", "#713F12")
    icon = aqi_data.get("icon", "🟡")
    advisory = aqi_data.get("advisory", "")
    station_name = aqi_data.get("station_name", label)
    dom_pol = aqi_data.get("dominant_pollutant", "PM2.5")
    pollutants = aqi_data.get("pollutants") or {}
    source_name = aqi_data.get("source", "World Air Quality Index (WAQI)")
    rep_time = aqi_data.get("reporting_time", "Live")

    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1.5px solid {border_color};
            border-left: 6px solid {color};
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="
                        background: {color};
                        color: #FFFFFF;
                        font-weight: 800;
                        font-size: 24px;
                        min-width: 62px;
                        height: 62px;
                        border-radius: 14px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        box-shadow: 0 4px 10px {color}55;
                        line-height: 1.1;
                    ">
                        <span>{aqi_val}</span>
                        <span style="font-size: 10px; font-weight: 600; opacity: 0.9;">AQI</span>
                    </div>
                    <div>
                        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                            <span style="font-size:16px; font-weight:700; color:#0F172A;">🍃 Air Quality · {station_name}</span>
                            <span style="background:{bg_color}; color:{text_color}; border:1px solid {border_color}; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:700;">
                                {icon} {category}
                            </span>
                        </div>
                        <div style="font-size:12.5px; color:#475569; margin-top:3px;">
                            <b>Dominant:</b> {dom_pol} &nbsp;·&nbsp; <span style="color:#64748B;">GPS: `{lat:.4f}, {lon:.4f}`</span>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:12.5px; color:#334155; line-height:1.45; background:{bg_color}66; padding:8px 12px; border-radius:8px;">
                <b>Health Advisory:</b> {advisory}
            </div>
            <!-- Pollutants Grid -->
            <div style="margin-top:10px; display:grid; grid-template-columns: repeat(auto-fit, minmax(105px, 1fr)); gap:6px;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">PM2.5</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('pm25'):.1f}" if isinstance(pollutants.get('pm25'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">µg/m³</span></div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">PM10</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('pm10'):.1f}" if isinstance(pollutants.get('pm10'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">µg/m³</span></div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">Ozone (O₃)</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('o3'):.1f}" if isinstance(pollutants.get('o3'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">ppb</span></div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">Nitrogen (NO₂)</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('no2'):.1f}" if isinstance(pollutants.get('no2'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">ppb</span></div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">Sulfur (SO₂)</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('so2'):.1f}" if isinstance(pollutants.get('so2'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">ppb</span></div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; text-align:center;">
                    <div style="font-size:10.5px; color:#64748B; font-weight:600;">Carbon (CO)</div>
                    <div style="font-size:13.5px; color:#0F172A; font-weight:700;">{f"{pollutants.get('co'):.1f}" if isinstance(pollutants.get('co'), (int, float)) else '--'} <span style="font-size:10px; font-weight:normal; color:#64748B;">ppm</span></div>
                </div>
            </div>
            <div style="margin-top:9px; display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#94A3B8; flex-wrap:wrap; gap:4px;">
                <span>💡 <b>Tip:</b> Click anywhere on the map to inspect real-time AQI for that spot.</span>
                <span>Source: <b>{source_name}</b> ({rep_time})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_close_aqi, col_inspect_info = st.columns([1.6, 8.4])
    with col_close_aqi:
        if st.button("✕ Close Air Quality", key="btn_close_aqi_banner", use_container_width=True):
            st.session_state.show_aqi_mode = False
            st.rerun()

    return aqi_data


def render_traffic_controls(is_realworld: bool) -> None:
    """Routing mode and modeled departure-time controls."""
    if not is_realworld:
        return
    with st.container(border=True):
        st.markdown("#### 🚦 Traffic-Aware Routing")
        st.radio(
            "Routing mode",
            options=[SHORTEST_DISTANCE, TRAFFIC_AWARE],
            format_func=lambda k: TRAFFIC_MODE_LABELS[k],
            horizontal=True,
            key="traffic_choice",
            on_change=on_routing_settings_change,
            help="Traffic-aware routing minimizes estimated travel time. Road distances remain unchanged.",
        )
        if st.session_state.traffic_choice == TRAFFIC_AWARE:
            clock_col, preset_col = st.columns([2.1, 1.5])
            with clock_col:
                st.markdown("**Departure time**")
                hour_col, minute_col, ampm_col = st.columns([1, 1, 0.9], gap="small")
                with hour_col:
                    st.selectbox(
                        "Hour", list(range(1, 13)), key="departure_hour_12",
                        format_func=lambda value: f"{value:02d}",
                        on_change=_apply_manual_departure_time,
                        label_visibility="collapsed",
                    )
                with minute_col:
                    st.selectbox(
                        "Minute", list(range(60)), key="departure_minute",
                        format_func=lambda value: f"{value:02d}",
                        on_change=_apply_manual_departure_time,
                        label_visibility="collapsed",
                    )
                with ampm_col:
                    st.selectbox(
                        "AM or PM", ["AM", "PM"], key="departure_ampm",
                        on_change=_apply_manual_departure_time,
                        label_visibility="collapsed",
                    )
            with preset_col:
                st.selectbox(
                    "Quick preset",
                    ["Custom time", "Morning Peak · 08:30", "Midday · 01:00 PM", "School Dismissal · 02:00 PM", "Evening Peak · 06:00 PM", "Night · 10:00 PM"],
                    key="time_preset", on_change=_apply_time_preset,
                    help="Choose a typical time to quickly compare modeled traffic.",
                )
            period = PERIOD_LABELS.get(get_traffic_period(st.session_state.departure_time), "")
            st.caption(f"{st.session_state.departure_time.strftime('%I:%M %p').lstrip('0')} · {period} · {TRAFFIC_ICONS[LOW]} Low  ·  {TRAFFIC_ICONS[MEDIUM]} Moderate  ·  {TRAFFIC_ICONS[HIGH]} High")
            st.caption("Peak hotspots: Main Road · Lalpur · Ratu Road · Kantatoli · Booty More · Upper Bazar. Other roads keep the standard time model; estimates are not live.")
        notice = st.session_state.get("traffic_notice")
        mode = get_traffic_mode()
        if mode is None and notice is None:
            return
        if notice is None:
            return
        kind = notice["kind"]
        if kind == "route" and mode is None:
            st.info("Shortest-distance route selected.")
        elif kind == "route" and notice["changed"]:
            st.success("Alternative route selected to reduce modeled travel time.")
        elif kind == "route":
            st.caption("The current route remains the best estimated choice for these settings.")
        elif kind == "mode_only" and mode:
            st.caption("Choose Find Route to calculate a traffic-aware recommendation.")
        elif kind == "no_route":
            st.error("❌ No route found under these traffic conditions.")
        elif kind == "error":
            st.error(f"Could not update the route: {notice.get('message', 'unknown error')}")


def render_route_information(result: PathResult, is_realworld: bool) -> None:
    """Show estimated time, traffic profile, and shortest-route comparison."""
    mode = get_traffic_mode()
    if not (is_realworld and mode and result and result.found):
        return
    graph = load_cached_osm_network()
    departure = normalize_departure_time(st.session_state.departure_time)
    info = analyse_route(graph, result.path, mode, departure)
    if info is None:
        return
    baseline = st.session_state.get("baseline_result")
    base_info = analyse_route(graph, baseline.path, mode, departure) if baseline and baseline.found else None
    is_alternative = bool(baseline and baseline.found and baseline.path != result.path)
    est_min = info.travel_time_seconds / 60
    impact_icon = TRAFFIC_ICONS[info.impact]

    with st.expander("📋 Traffic-Aware Route Estimate", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Distance", _fmt_dist(result.total_distance))
        c2.metric("Estimated travel time", f"{est_min:.1f} min", help="Modeled estimate based on road class and departure time.")
        c3.metric("Traffic impact", f"{impact_icon} {info.impact.capitalize()}")
        st.markdown(
            f"- Road segments: {TRAFFIC_ICONS[LOW]} {info.level_counts[LOW]} low · "
            f"{TRAFFIC_ICONS[MEDIUM]} {info.level_counts[MEDIUM]} moderate · "
            f"{TRAFFIC_ICONS[HIGH]} {info.level_counts[HIGH]} high\n"
            f"- Average estimated speed: {info.estimated_speed_kmh:.0f} km/h\n"
            f"- {'✓ Alternative route selected to avoid modeled congestion' if is_alternative else '✓ Shortest route is also the fastest estimated route'}"
        )
        if baseline and baseline.found and base_info:
            base_min = base_info.travel_time_seconds / 60
            st.caption(
                f"Route comparison · Shortest distance: {_fmt_dist(baseline.total_distance)} / {base_min:.1f} min; "
                f"recommended: {_fmt_dist(result.total_distance)} / {est_min:.1f} min."
            )
            if not is_alternative:
                st.caption("The shortest-distance route is also the fastest estimated route.")

    with st.expander("🧠 Routing Analysis", expanded=False):
        st.markdown(
            f"- Algorithm: {result.algorithm} (time cost with an admissible speed-based heuristic)\n"
            f"- Nodes explored: {result.visited_nodes_count}\n"
            f"- Estimated routing cost: {result.total_cost:.0f} seconds"
        )


def swap_locations() -> None:
    """Swap start and destination locations in session state."""
    old_start_name = st.session_state.start_name
    old_start_coords = st.session_state.start_coords
    old_start_address = st.session_state.get("start_address")

    st.session_state.start_name = st.session_state.dest_name
    st.session_state.start_coords = st.session_state.dest_coords
    st.session_state.start_address = st.session_state.get("dest_address")

    st.session_state.dest_name = old_start_name
    st.session_state.dest_coords = old_start_coords
    st.session_state.dest_address = old_start_address

    # Clear previous route so user re-computes in the opposite direction
    st.session_state.route_result = None


def clear_global_search_pin() -> None:
    """Callback function to clear search pin and search input before widget instantiation."""
    st.session_state.searched_location = None
    st.session_state.searched_address = None
    st.session_state.prev_search_suggestion = None
    st.session_state.global_search_input = ""
    if not (st.session_state.get("start_coords") and st.session_state.get("dest_coords")):
        st.session_state.show_info_panel = False
    else:
        st.session_state.info_panel_target = "destination"


def reset_click_to_route() -> None:
    """Clear endpoints and reset interactive click-to-route state."""
    st.session_state.start_coords = None
    st.session_state.start_name = None
    st.session_state.start_address = None
    st.session_state.dest_coords = None
    st.session_state.dest_name = None
    st.session_state.dest_address = None
    st.session_state.route_result = None
    st.session_state.alternative_results = None
    st.session_state.click_route_step = "start"
    st.session_state.last_handled_click = None
    st.toast("Ready! Touch or click any spot on the map to set Start (Point A).", icon="📍")


def swap_endpoints(is_realworld: bool) -> None:
    """Swap Start and Destination and instantly recalculate route in reverse."""
    if st.session_state.get("start_coords") and st.session_state.get("dest_coords"):
        st.session_state.start_coords, st.session_state.dest_coords = (
            st.session_state.dest_coords,
            st.session_state.start_coords,
        )
        st.session_state.start_name, st.session_state.dest_name = (
            st.session_state.dest_name,
            st.session_state.start_name,
        )
        st.session_state.start_address, st.session_state.dest_address = (
            st.session_state.dest_address,
            st.session_state.start_address,
        )
        try:
            compute_routes(is_realworld)
            st.toast(f"↔️ Swapped: {st.session_state.start_name} ➔ {st.session_state.dest_name}!", icon="🔄")
        except Exception:
            st.session_state.route_result = None


def generate_random_demo_route(is_realworld: bool) -> None:
    """Generate a random demonstration route to showcase the instant click-to-route algorithm."""
    if not is_realworld:
        graph = load_fictional_network()
        nodes = list(graph.get_nodes())
        if len(nodes) >= 2:
            s, d = random.sample(nodes, 2)
            st.session_state.start_name = s
            st.session_state.dest_name = d
            c_s = graph.get_node_coords(s)
            c_d = graph.get_node_coords(d)
            st.session_state.start_coords = c_s
            st.session_state.dest_coords = c_d
            st.session_state.click_route_step = "active"
            compute_routes(is_realworld)
            st.toast(f"🎲 Random Route: {s} ➔ {d}!", icon="✨")
            return

    # In real-world mode: pick 2 distinct landmarks from Ranchi
    keys = list(RANCHI_LANDMARKS.keys())
    if len(keys) >= 2:
        k1, k2 = random.sample(keys, 2)
        st.session_state.start_name = k1
        st.session_state.start_coords = RANCHI_LANDMARKS[k1]
        st.session_state.start_address = f"{k1}, Ranchi"
        st.session_state.dest_name = k2
        st.session_state.dest_coords = RANCHI_LANDMARKS[k2]
        st.session_state.dest_address = f"{k2}, Ranchi"
        st.session_state.searched_location = None
        st.session_state.click_route_step = "active"
        compute_routes(is_realworld)
        st.toast(f"🎲 Demo Route: {k1} ➔ {k2}!", icon="✨")


def render_click_to_route_dashboard_bar(is_realworld: bool) -> None:
    """Render the Interactive Click-to-Route Dashboard header, live status, and quick zero-typing actions."""
    has_start = bool(st.session_state.get("start_coords"))
    has_dest = bool(st.session_state.get("dest_coords"))
    has_route = bool(st.session_state.get("route_result") and st.session_state.route_result.found)

    start_label = st.session_state.get("start_name") or "Waiting for click..."
    dest_label = st.session_state.get("dest_name") or "Waiting for click..."

    if not has_start:
        status_msg = "👉 <b>Step 1:</b> Touch or click <b>any spot</b> on the giant live map below to set <b>Point A (Start)</b>. Zero typing required!"
        status_icon = "📍"
        status_bg = "#EFF6FF"
        status_border = "#93C5FD"
        status_text = "#1E40AF"
    elif has_start and not has_dest:
        status_msg = f"🎯 <b>Point A Set:</b> <b>{start_label}</b>! Now touch or click <b>any spot</b> for <b>Point B (Destination)</b> — path draws instantly!"
        status_icon = "🏁"
        status_bg = "#FEF3C7"
        status_border = "#FCD34D"
        status_text = "#92400E"
    else:
        res = st.session_state.route_result
        if res and res.found:
            dist_s = f"{res.total_distance/1000:.2f} km" if res.total_distance >= 1000 else f"{res.total_distance:.0f} m"
            drive_eta = estimate_travel_time(res.total_distance, speed_kmh=30.0)
            status_msg = f"⚡ <b>Path Drawn:</b> <b>{start_label}</b> ➔ <b>{dest_label}</b> &nbsp;|&nbsp; 📏 <b>{dist_s}</b> &nbsp;|&nbsp; 🚗 ~{drive_eta} &nbsp;|&nbsp; 🛣️ {len(res.legs)} legs"
        else:
            status_msg = f"🧭 Endpoints set: <b>{start_label}</b> ➔ <b>{dest_label}</b>"
        status_icon = "🚀"
        status_bg = "#ECFDF5"
        status_border = "#6EE7B7"
        status_text = "#065F46"

    badge_a = "done" if has_start else "active"
    badge_b = "done" if has_dest else ("active" if has_start else "pending")

    st.markdown(
        f"""
        <div style="background:{status_bg}; border:1.5px solid {status_border}; border-radius:12px; padding:12px 18px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:22px;">{status_icon}</span>
                <span style="color:{status_text}; font-size:14.5px;">{status_msg}</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="click-step-badge {badge_a}">
                    {'✓' if has_start else '1'} 📍 Point A: {start_label[:20] + '…' if len(start_label) > 20 else start_label}
                </span>
                <span style="color:#94A3B8; font-weight:700;">➔</span>
                <span class="click-step-badge {badge_b}">
                    {'✓' if has_dest else '2'} 🏁 Point B: {dest_label[:20] + '…' if len(dest_label) > 20 else dest_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action bar directly above the giant map
    col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns([1.6, 1.4, 1.7, 1.5, 1.8])
    with col_a1:
        if st.button("🔄 New 2-Click Route", key="btn_reset_click_nav", use_container_width=True, help="Clear endpoints to touch 2 new spots"):
            reset_click_to_route()
            st.rerun()
    with col_a2:
        if st.button("↔️ Reverse Path", key="btn_swap_click_nav", use_container_width=True, disabled=not (has_start and has_dest), help="Swap Point A and Point B"):
            swap_endpoints(is_realworld)
            st.rerun()
    with col_a3:
        if st.button("🎲 Surprise Me (Random)", key="btn_random_demo_nav", use_container_width=True, help="Show off instant routing with 2 random spots"):
            generate_random_demo_route(is_realworld)
            st.rerun()
    with col_a4:
        curr_h = st.session_state.get("map_height_setting", 700)
        h_text = "📐 Giant (700px)" if curr_h == 700 else ("🖥️ Ultra (820px)" if curr_h == 820 else "📏 Standard (580px)")
        next_h = 820 if curr_h == 700 else (580 if curr_h == 820 else 700)
        if st.button(f"{h_text}", key="btn_map_height_nav", use_container_width=True, help="Toggle giant live map height"):
            st.session_state.map_height_setting = next_h
            st.rerun()
    with col_a5:
        beh = st.session_state.get("click_behavior", "update_dest")
        beh_text = "🎯 Move Dest on Click" if beh == "update_dest" else "📍 New Pair on Click"
        next_beh = "new_route" if beh == "update_dest" else "update_dest"
        if st.button(f"{beh_text}", key="btn_beh_toggle_nav", use_container_width=True, help="Control behavior when clicking map after route is drawn"):
            st.session_state.click_behavior = next_beh
            st.rerun()


def _render_map(primary_result: Optional[PathResult], is_realworld: bool, map_height: int = 580) -> None:
    """Helper to render interactive Folium map with current points, blue dot marker, or clean state."""
    active_aqi_data = None
    if (
        st.session_state.get("show_aqi_mode", False)
        and not st.session_state.get("aqi_location_pending")
        and not st.session_state.get("aqi_location_error")
    ):
        inspect_tuple = st.session_state.get("aqi_inspect_coords")
        if inspect_tuple:
            lat, lon, label = inspect_tuple
            reading = get_cached_aqi_reading(lat, lon, token=st.session_state.get("waqi_custom_token", ""), name=label)
            if reading.get("available", True):
                active_aqi_data = reading

    has_endpoints = bool(
        (st.session_state.get("start_coords") and st.session_state.get("dest_coords"))
        if is_realworld
        else (st.session_state.get("start_name") and st.session_state.get("dest_name"))
    )
    if primary_result and primary_result.found and has_endpoints:
        if st.session_state.get("is_worldwide_active") and st.session_state.get("worldwide_graph"):
            graph = st.session_state.worldwide_graph
            traffic_mode = None
        elif is_realworld:
            graph = load_cached_osm_network()
            traffic_mode = get_traffic_mode()
        else:
            graph = load_fictional_network()
            traffic_mode = None

        alt_res = st.session_state.get("alternative_results")
        active_idx = st.session_state.get("active_route_index", 0)

        if is_realworld and st.session_state.start_coords and st.session_state.dest_coords:
            folium_map = build_folium_map(
                graph=graph,
                result=primary_result,
                start_coord=st.session_state.start_coords,
                dest_coord=st.session_state.dest_coords,
                start_label=st.session_state.start_name or "Origin",
                dest_label=st.session_state.dest_name or "Destination",
                searched_point=st.session_state.searched_location,
                current_location=st.session_state.my_location,
                alternative_results=alt_res,
                active_route_index=active_idx,
                show_hud=False,  # Keep map canvas clean!
                traffic_mode=traffic_mode,
                departure_time=st.session_state.departure_time,
                show_aqi=st.session_state.get("show_aqi_mode", False),
                aqi_data=active_aqi_data,
                waqi_token=st.session_state.get("waqi_custom_token", "") or "demo",
            )
        else:
            folium_map = build_empty_map(
                center=(23.3191843, 85.2987681),
                zoom=15,
                show_aqi=st.session_state.get("show_aqi_mode", False),
                aqi_data=active_aqi_data,
                waqi_token=st.session_state.get("waqi_custom_token", "") or "demo",
            )
    else:
        start_pt = (
            (st.session_state.start_coords[0], st.session_state.start_coords[1], st.session_state.start_name)
            if st.session_state.start_coords and st.session_state.start_name
            else None
        )
        dest_pt = (
            (st.session_state.dest_coords[0], st.session_state.dest_coords[1], st.session_state.dest_name)
            if st.session_state.dest_coords and st.session_state.dest_name
            else None
        )
        clean_center = (
            (st.session_state.searched_location[0], st.session_state.searched_location[1])
            if st.session_state.searched_location
            else (
                (st.session_state.start_coords[0], st.session_state.start_coords[1])
                if st.session_state.start_coords
                else (
                    (st.session_state.dest_coords[0], st.session_state.dest_coords[1])
                    if st.session_state.dest_coords
                    else (st.session_state.my_location[0], st.session_state.my_location[1])
                )
            )
        )
        folium_map = build_empty_map(
            center=clean_center,
            zoom=15 if (st.session_state.searched_location or start_pt or dest_pt) else 14,
            start_point=start_pt,
            dest_point=dest_pt,
            searched_point=st.session_state.searched_location,
            current_location=st.session_state.my_location,
            graph=load_cached_osm_network() if (is_realworld and get_traffic_mode()) else None,
            traffic_mode=get_traffic_mode() if is_realworld else None,
            departure_time=st.session_state.departure_time,
            show_aqi=st.session_state.get("show_aqi_mode", False),
            aqi_data=active_aqi_data,
            waqi_token=st.session_state.get("waqi_custom_token", "") or "demo",
        )

    # Add visual bookmark markers to map
    for bm in st.session_state.get("bookmarks", []):
        folium.Marker(
            location=[bm["lat"], bm["lon"]],
            popup=folium.Popup(
                f"<div style='font-family: Arial, sans-serif; min-width: 150px;'>"
                f"<h4 style='margin: 0 0 4px 0; color: #D97706;'>⭐ {bm['name']}</h4>"
                f"<b>{bm.get('address', '')}</b><br>"
                f"<span style='font-size: 11px; color: #666;'>Lat: {bm['lat']:.5f}, Lon: {bm['lon']:.5f}</span>"
                f"</div>",
                max_width=250,
            ),
            tooltip=f"⭐ Bookmark: {bm['name']}",
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        ).add_to(folium_map)

    folium_map.get_root().html.add_child(
        folium.Element("<style>.leaflet-container { cursor: crosshair !important; }</style>")
    )

    start_sig = f"{st.session_state.start_coords}" if st.session_state.get("start_coords") else "no_start"
    dest_sig = f"{st.session_state.dest_coords}" if st.session_state.get("dest_coords") else "no_dest"
    click_step_sig = st.session_state.get("click_route_step", "start")
    route_signature = "-".join(str(node) for node in primary_result.path) if primary_result and primary_result.found else "preview"
    bm_signature = "-".join(f"{b['name']}:{b['lat']:.4f},{b['lon']:.4f}" for b in st.session_state.get("bookmarks", []))
    aqi_sig = f"aqi_on:{st.session_state.get('aqi_inspect_coords')}" if st.session_state.get("show_aqi_mode") else "aqi_off"
    map_signature = "|".join((
        "drawer" if st.session_state.show_route_menu else "full",
        "bm_panel" if st.session_state.get("show_bookmarks") else "no_bm",
        "info_panel" if st.session_state.get("show_info_panel") else "no_info",
        aqi_sig,
        route_signature,
        get_traffic_mode() or "distance",
        st.session_state.departure_time.strftime("%H:%M"),
        bm_signature,
        start_sig,
        dest_sig,
        click_step_sig,
        str(map_height),
    ))
    map_data = st_folium(
        folium_map,
        width=None,
        height=map_height,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key=f"pathfinder-map-{hashlib.sha1(map_signature.encode()).hexdigest()[:12]}",
    )

    if map_data and map_data.get("last_clicked"):
        click_lat = float(map_data["last_clicked"]["lat"])
        click_lon = float(map_data["last_clicked"]["lng"])
        click_point = (round(click_lat, 6), round(click_lon, 6))

        if click_point != st.session_state.get("last_handled_click"):
            st.session_state.last_handled_click = click_point
            st.session_state.last_clicked_coords = (click_lat, click_lon)

            # If AQI mode is active, tapping the map inspects that specific location's AQI
            if st.session_state.get("show_aqi_mode", False):
                place_name = reverse_geocode(click_lat, click_lon) or f"Area ({click_lat:.4f}, {click_lon:.4f})"
                st.session_state.aqi_inspect_coords = (click_lat, click_lon, place_name)
                st.session_state.aqi_location_pending = False
                st.session_state.aqi_location_error = None
                st.toast(f"🍃 Checking AQI for {place_name}...", icon="🍃")
                st.rerun()

            if st.session_state.get("click_to_route_mode", True):
                start_is_set = bool(st.session_state.get("start_coords"))
                dest_is_set = bool(st.session_state.get("dest_coords"))
                step = st.session_state.get("click_route_step", "start")

                if not start_is_set or step == "start":
                    # STEP 1: Point A (Start)
                    place_name = reverse_geocode(click_lat, click_lon)
                    st.session_state.start_name = place_name
                    st.session_state.start_coords = (click_lat, click_lon)
                    st.session_state.start_address = place_name
                    st.session_state.dest_name = None
                    st.session_state.dest_coords = None
                    st.session_state.dest_address = None
                    st.session_state.route_result = None
                    st.session_state.alternative_results = None
                    st.session_state.searched_location = None
                    st.session_state.click_route_step = "dest"
                    st.toast(f"📍 Point A set to '{place_name}'! Now tap or click Point B on the map.", icon="🚩")
                    st.rerun()

                elif not dest_is_set or step == "dest":
                    # STEP 2: Point B (Destination) + Instant Route Calculation
                    from astar import haversine_distance
                    if haversine_distance(st.session_state.start_coords, (click_lat, click_lon)) < 15.0:
                        st.toast("⚠️ Point B is too close to Point A. Please click another spot on the map.", icon="📍")
                    else:
                        place_name = reverse_geocode(click_lat, click_lon)
                        st.session_state.dest_name = place_name
                        st.session_state.dest_coords = (click_lat, click_lon)
                        st.session_state.dest_address = place_name
                        st.session_state.searched_location = None
                        st.session_state.click_route_step = "active"
                        try:
                            compute_routes(is_realworld)
                            if st.session_state.route_result and st.session_state.route_result.found:
                                d_m = st.session_state.route_result.total_distance
                                d_str = f"{d_m/1000:.2f} km" if d_m >= 1000 else f"{d_m:.0f} m"
                                st.toast(f"⚡ Path drawn instantly! Distance: {d_str}", icon="🚀")
                        except Exception as e:
                            st.session_state.route_result = None
                            st.toast(f"Route calculation error: {e}", icon="❌")
                        st.rerun()

                else:
                    # STEP 3: Route is already active
                    beh = st.session_state.get("click_behavior", "update_dest")
                    if beh == "update_dest":
                        # Instant retargeting: Move destination to clicked spot and redraw path!
                        place_name = reverse_geocode(click_lat, click_lon)
                        st.session_state.dest_name = place_name
                        st.session_state.dest_coords = (click_lat, click_lon)
                        st.session_state.dest_address = place_name
                        st.session_state.searched_location = None
                        try:
                            compute_routes(is_realworld)
                            if st.session_state.route_result and st.session_state.route_result.found:
                                d_m = st.session_state.route_result.total_distance
                                d_str = f"{d_m/1000:.2f} km" if d_m >= 1000 else f"{d_m:.0f} m"
                                st.toast(f"🎯 Re-routed destination to '{place_name}' ({d_str})!", icon="🔄")
                        except Exception as e:
                            st.session_state.route_result = None
                        st.rerun()
                    else:
                        # Start new 2-click route from this spot
                        place_name = reverse_geocode(click_lat, click_lon)
                        st.session_state.start_name = place_name
                        st.session_state.start_coords = (click_lat, click_lon)
                        st.session_state.start_address = place_name
                        st.session_state.dest_name = None
                        st.session_state.dest_coords = None
                        st.session_state.dest_address = None
                        st.session_state.route_result = None
                        st.session_state.alternative_results = None
                        st.session_state.searched_location = None
                        st.session_state.click_route_step = "dest"
                        st.toast(f"📍 New Start set to '{place_name}'! Click Point B to route.", icon="🚩")
                        st.rerun()

    if st.session_state.get("last_clicked_coords"):
        lat_c, lon_c = st.session_state.last_clicked_coords
        with st.expander(f"📍 Click Details (`{lat_c:.5f}, {lon_c:.5f}`) & Manual Pin Options", expanded=False):
            col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1])
            with col_c1:
                if st.button("📍 Set as Start (Point A)", key="btn_click_start_manual"):
                    st.session_state.start_name = reverse_geocode(lat_c, lon_c)
                    st.session_state.start_coords = (lat_c, lon_c)
                    st.session_state.click_route_step = "dest"
                    st.rerun()
            with col_c2:
                if st.button("🎯 Set as Dest (Point B)", key="btn_click_dest_manual"):
                    st.session_state.dest_name = reverse_geocode(lat_c, lon_c)
                    st.session_state.dest_coords = (lat_c, lon_c)
                    st.session_state.click_route_step = "active"
                    compute_routes(is_realworld)
                    st.rerun()
            with col_c3:
                if st.button("🔵 Set My Location", key="btn_click_my_loc_manual"):
                    loc_lbl = reverse_geocode(lat_c, lon_c)
                    st.session_state.my_location = (lat_c, lon_c, loc_lbl)
                    st.rerun()
            with col_c4:
                if st.button("🔖 Bookmark Spot", key="btn_click_bm_manual"):
                    st.session_state.show_bookmarks = True
                    st.rerun()


def on_origin_search_enter() -> None:
    """Callback when user presses Enter in the Origin input box."""
    query = st.session_state.get("drawer_origin_search_input", "").strip()
    if not query:
        return
    results = search_locations(query, limit=5)
    if results:
        top = results[0]
        st.session_state.start_name = top.name
        st.session_state.start_coords = (top.lat, top.lon)
        st.session_state.start_address = top.display_name
        st.session_state.searched_location = None  # Ensure proper start flag marker appears
        st.session_state.origin_status_msg = f"✓ Set: {top.format_display_label()}"
        st.session_state.origin_alternatives = results[1:] if len(results) > 1 else []
        st.session_state.drawer_origin_search_input = ""
        is_rw = "Real-World" in st.session_state.get("network_mode", "Real-World")
        st.session_state.show_info_panel = True
        if st.session_state.dest_coords and st.session_state.dest_name:
            try:
                compute_routes(is_rw)
            except Exception:
                st.session_state.route_result = None
        else:
            st.session_state.route_result = None
            st.session_state.info_panel_target = "origin"
    else:
        st.session_state.origin_status_msg = f"❌ No location found for '{query}'"
        st.session_state.origin_alternatives = []


def on_dest_search_enter() -> None:
    """Callback when user presses Enter in the Destination input box."""
    query = st.session_state.get("drawer_dest_search_input", "").strip()
    if not query:
        return
    results = search_locations(query, limit=5)
    if results:
        top = results[0]
        st.session_state.dest_name = top.name
        st.session_state.dest_coords = (top.lat, top.lon)
        st.session_state.dest_address = top.display_name
        st.session_state.searched_location = None  # Ensure proper destination flag marker appears
        st.session_state.dest_status_msg = f"✓ Set: {top.format_display_label()}"
        st.session_state.dest_alternatives = results[1:] if len(results) > 1 else []
        st.session_state.drawer_dest_search_input = ""
        st.session_state.show_info_panel = True
        st.session_state.info_panel_target = "destination"
        is_rw = "Real-World" in st.session_state.get("network_mode", "Real-World")
        if st.session_state.start_coords and st.session_state.start_name:
            try:
                compute_routes(is_rw)
            except Exception:
                st.session_state.route_result = None
        else:
            st.session_state.route_result = None
    else:
        st.session_state.dest_status_msg = f"❌ No location found for '{query}'"
        st.session_state.dest_alternatives = []


def on_preset_origin_selected() -> None:
    """Callback when user picks a preset origin landmark. Never loops or crashes."""
    chosen = st.session_state.get("select_preset_origin_quick")
    if not chosen or chosen == "-- Preset Landmarks --":
        return
    bookmarks = st.session_state.get("bookmarks", [])
    available = {**WORLDWIDE_LANDMARKS, **RANCHI_LANDMARKS}
    if chosen.startswith("⭐ "):
        bname = chosen[2:]
        bmatch = next((b for b in bookmarks if b["name"] == bname), None)
        if bmatch:
            st.session_state.start_name = bmatch["name"]
            st.session_state.start_coords = (bmatch["lat"], bmatch["lon"])
    elif chosen in available:
        st.session_state.start_name = chosen
        st.session_state.start_coords = available[chosen]
    st.session_state.searched_location = None
    st.session_state.origin_alternatives = []
    st.session_state.origin_status_msg = f"✓ Set: {st.session_state.start_name}"
    # Reset widget state so it never re-triggers or loops
    st.session_state.select_preset_origin_quick = "-- Preset Landmarks --"
    is_rw = "Real-World" in st.session_state.get("network_mode", "Real-World")
    st.session_state.show_info_panel = True
    if st.session_state.dest_coords and st.session_state.dest_name:
        try:
            compute_routes(is_rw)
        except Exception:
            st.session_state.route_result = None
    else:
        st.session_state.route_result = None
        st.session_state.info_panel_target = "origin"


def on_preset_dest_selected() -> None:
    """Callback when user picks a preset destination landmark. Never loops or crashes."""
    chosen = st.session_state.get("select_preset_dest_quick")
    if not chosen or chosen == "-- Preset Landmarks --":
        return
    bookmarks = st.session_state.get("bookmarks", [])
    available = {**WORLDWIDE_LANDMARKS, **RANCHI_LANDMARKS}
    if chosen.startswith("⭐ "):
        bname = chosen[2:]
        bmatch = next((b for b in bookmarks if b["name"] == bname), None)
        if bmatch:
            st.session_state.dest_name = bmatch["name"]
            st.session_state.dest_coords = (bmatch["lat"], bmatch["lon"])
    elif chosen in available:
        st.session_state.dest_name = chosen
        st.session_state.dest_coords = available[chosen]
    st.session_state.searched_location = None
    st.session_state.dest_alternatives = []
    st.session_state.dest_status_msg = f"✓ Set: {st.session_state.dest_name}"
    # Reset widget state so it never re-triggers or loops
    st.session_state.select_preset_dest_quick = "-- Preset Landmarks --"
    is_rw = "Real-World" in st.session_state.get("network_mode", "Real-World")
    st.session_state.show_info_panel = True
    st.session_state.info_panel_target = "destination"
    if st.session_state.start_coords and st.session_state.start_name:
        try:
            compute_routes(is_rw)
        except Exception:
            st.session_state.route_result = None
    else:
        st.session_state.route_result = None


def render_directions_drawer(is_realworld: bool) -> None:
    """Render the sliding directions drawer on the left side."""
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown("### 🧭 Directions")
    with head_col2:
        if st.button("✕", key="btn_close_drawer", help="Close directions menu"):
            st.session_state.show_route_menu = False
            st.rerun()

    if is_realworld:
        bookmarks = st.session_state.get("bookmarks", [])
        available_landmarks = {**WORLDWIDE_LANDMARKS, **RANCHI_LANDMARKS}

        # ── 1. Origin (From) ──
        st.markdown("**🚩 From (Origin):**")
        if st.session_state.start_coords and st.session_state.start_name:
            start_lat, start_lon = st.session_state.start_coords
            col_orig_card, col_orig_clear = st.columns([4.2, 0.8])
            with col_orig_card:
                st.markdown(
                    f"""
                    <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:8px 12px; margin-bottom:8px;">
                        <b style="color:#166534; font-size:14px;">🚩 {st.session_state.start_name}</b><br>
                        <span style="color:#15803D; font-size:11px;">GPS: {start_lat:.5f}, {start_lon:.5f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_orig_clear:
                if st.button("✕", key="btn_clear_origin", help="Clear origin"):
                    st.session_state.start_name = None
                    st.session_state.start_coords = None
                    st.session_state.route_result = None
                    st.session_state.origin_status_msg = None
                    st.session_state.origin_alternatives = []
                    if st.session_state.get("dest_coords"):
                        st.session_state.info_panel_target = "destination"
                    elif st.session_state.get("searched_location"):
                        st.session_state.info_panel_target = "searched"
                    else:
                        st.session_state.show_info_panel = False
                    st.rerun()
        else:
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:8px; padding:8px 12px; margin-bottom:8px; color:#64748B; font-size:13px;">
                    🚩 <i>No origin selected. Type below & press Enter.</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.text_input(
            "Search Origin:",
            placeholder="Type any place & press Enter (e.g. Purulia, Eiffel Tower)...",
            key="drawer_origin_search_input",
            on_change=on_origin_search_enter,
            label_visibility="collapsed",
        )

        if st.session_state.get("origin_status_msg"):
            st.caption(st.session_state.origin_status_msg)

        if st.session_state.get("origin_alternatives"):
            with st.expander("ℹ️ Other matches for Origin", expanded=False):
                for i, alt in enumerate(st.session_state.origin_alternatives):
                    if st.button(f"📍 {alt.format_display_label()}", key=f"btn_alt_orig_{i}", use_container_width=True):
                        st.session_state.start_name = alt.name
                        st.session_state.start_coords = (alt.lat, alt.lon)
                        st.session_state.start_address = alt.display_name
                        st.session_state.searched_location = None
                        st.session_state.origin_status_msg = f"✓ Set: {alt.format_display_label()}"
                        st.session_state.origin_alternatives = []
                        if st.session_state.dest_coords and st.session_state.dest_name:
                            try:
                                compute_routes(is_realworld)
                            except Exception:
                                st.session_state.route_result = None
                        else:
                            st.session_state.route_result = None
                        st.rerun()

        col_orig_my, col_orig_pre = st.columns([1.1, 1.9])
        with col_orig_my:
            if st.button("🔵 My Location", key="btn_orig_use_my_loc", use_container_width=True, help="Set origin to your current location"):
                st.session_state.start_name = MY_LOCATION_LABEL
                st.session_state.start_coords = (st.session_state.my_location[0], st.session_state.my_location[1])
                st.session_state.searched_location = None
                st.session_state.origin_status_msg = "✓ Set: My Current Location"
                st.session_state.origin_alternatives = []
                if st.session_state.dest_coords and st.session_state.dest_name:
                    try:
                        compute_routes(is_realworld)
                    except Exception:
                        st.session_state.route_result = None
                else:
                    st.session_state.route_result = None
                st.rerun()
        with col_orig_pre:
            orig_pre_opts = ["-- Preset Landmarks --"]
            for bm in bookmarks:
                orig_pre_opts.append(f"⭐ {bm['name']}")
            orig_pre_opts.extend(sorted(list(available_landmarks.keys())))
            st.selectbox(
                "Preset Origin",
                orig_pre_opts,
                key="select_preset_origin_quick",
                on_change=on_preset_origin_selected,
                label_visibility="collapsed",
            )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        # Swap button
        if st.button("⇄ Swap Start & Destination", on_click=swap_locations, use_container_width=True, key="btn_swap_drawer"):
            if st.session_state.start_coords and st.session_state.dest_coords:
                try:
                    compute_routes(is_realworld)
                except Exception:
                    st.session_state.route_result = None
            else:
                st.session_state.route_result = None
            st.rerun()

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # ── 2. Destination (To) ──
        st.markdown("**🏁 To (Destination):**")
        if st.session_state.dest_coords and st.session_state.dest_name:
            dest_lat, dest_lon = st.session_state.dest_coords
            col_dest_card, col_dest_clear = st.columns([4.2, 0.8])
            with col_dest_card:
                st.markdown(
                    f"""
                    <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:8px; padding:8px 12px; margin-bottom:8px;">
                        <b style="color:#991B1B; font-size:14px;">🏁 {st.session_state.dest_name}</b><br>
                        <span style="color:#DC2626; font-size:11px;">GPS: {dest_lat:.5f}, {dest_lon:.5f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_dest_clear:
                if st.button("✕", key="btn_clear_dest", help="Clear destination"):
                    st.session_state.dest_name = None
                    st.session_state.dest_coords = None
                    st.session_state.route_result = None
                    st.session_state.dest_status_msg = None
                    st.session_state.dest_alternatives = []
                    if st.session_state.get("start_coords"):
                        st.session_state.info_panel_target = "origin"
                    elif st.session_state.get("searched_location"):
                        st.session_state.info_panel_target = "searched"
                    else:
                        st.session_state.show_info_panel = False
                    st.rerun()
        else:
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:8px; padding:8px 12px; margin-bottom:8px; color:#64748B; font-size:13px;">
                    🏁 <i>No destination set. Type below & press Enter.</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.text_input(
            "Search Destination:",
            placeholder="Type any place & press Enter (e.g. Adra, Louvre Museum)...",
            key="drawer_dest_search_input",
            on_change=on_dest_search_enter,
            label_visibility="collapsed",
        )

        if st.session_state.get("dest_status_msg"):
            st.caption(st.session_state.dest_status_msg)

        if st.session_state.get("dest_alternatives"):
            with st.expander("ℹ️ Other matches for Destination", expanded=False):
                for i, alt in enumerate(st.session_state.dest_alternatives):
                    if st.button(f"📍 {alt.format_display_label()}", key=f"btn_alt_dest_{i}", use_container_width=True):
                        st.session_state.dest_name = alt.name
                        st.session_state.dest_coords = (alt.lat, alt.lon)
                        st.session_state.dest_address = alt.display_name
                        st.session_state.searched_location = None
                        st.session_state.dest_status_msg = f"✓ Set: {alt.format_display_label()}"
                        st.session_state.dest_alternatives = []
                        if st.session_state.start_coords and st.session_state.start_name:
                            try:
                                compute_routes(is_realworld)
                            except Exception:
                                st.session_state.route_result = None
                        else:
                            st.session_state.route_result = None
                        st.rerun()

        col_dest_my, col_dest_pre = st.columns([1.1, 1.9])
        with col_dest_my:
            if st.button("🔵 My Location", key="btn_dest_use_my_loc", use_container_width=True, help="Set destination to your current location"):
                st.session_state.dest_name = MY_LOCATION_LABEL
                st.session_state.dest_coords = (st.session_state.my_location[0], st.session_state.my_location[1])
                st.session_state.searched_location = None
                st.session_state.dest_status_msg = "✓ Set: My Current Location"
                st.session_state.dest_alternatives = []
                if st.session_state.start_coords and st.session_state.start_name:
                    try:
                        compute_routes(is_realworld)
                    except Exception:
                        st.session_state.route_result = None
                else:
                    st.session_state.route_result = None
                st.rerun()
        with col_dest_pre:
            dest_pre_opts = ["-- Preset Landmarks --"]
            for bm in bookmarks:
                dest_pre_opts.append(f"⭐ {bm['name']}")
            dest_pre_opts.extend(sorted(list(available_landmarks.keys())))
            st.selectbox(
                "Preset Destination",
                dest_pre_opts,
                key="select_preset_dest_quick",
                on_change=on_preset_dest_selected,
                label_visibility="collapsed",
            )

    else:
        # Fictional towns
        fict_graph = load_fictional_network()
        fict_nodes = fict_graph.get_nodes()
        st.session_state.start_name = st.selectbox("From (Town)", fict_nodes, index=0)
        if st.button("⇄ Swap", on_click=swap_locations, use_container_width=True):
            pass
        st.session_state.dest_name = st.selectbox("To (Town)", fict_nodes, index=min(3, len(fict_nodes)-1))

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    find_route_clicked = st.button("🚀 Find Route", type="primary", use_container_width=True, key="btn_find_route_action")
    if find_route_clicked:
        st.session_state.traffic_notice = None
        if is_realworld:
            # Auto-resolve drafted inputs if user clicked Find Route without pressing enter
            if not st.session_state.start_coords or not st.session_state.start_name:
                draft_orig = st.session_state.get("drawer_origin_search_input", "").strip()
                if draft_orig:
                    res = search_locations(draft_orig, limit=1)
                    if res:
                        st.session_state.start_name = res[0].name
                        st.session_state.start_coords = (res[0].lat, res[0].lon)
                        st.session_state.origin_status_msg = f"✓ Set: {res[0].format_display_label()}"
                        st.session_state.drawer_origin_search_input = ""

            if not st.session_state.dest_coords or not st.session_state.dest_name:
                draft_dest = st.session_state.get("drawer_dest_search_input", "").strip()
                if draft_dest:
                    res = search_locations(draft_dest, limit=1)
                    if res:
                        st.session_state.dest_name = res[0].name
                        st.session_state.dest_coords = (res[0].lat, res[0].lon)
                        st.session_state.dest_status_msg = f"✓ Set: {res[0].format_display_label()}"
                        st.session_state.drawer_dest_search_input = ""

            if not st.session_state.start_coords or not st.session_state.start_name:
                st.warning("⚠️ Please select an Origin (From) first.")
            elif not st.session_state.dest_coords or not st.session_state.dest_name:
                st.warning("⚠️ Please select a Destination (To) first.")
            else:
                try:
                    with st.spinner("Finding optimal route..."):
                        compute_routes(is_realworld)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error mapping coordinates: {e}")
        else:
            if not st.session_state.start_name or not st.session_state.dest_name:
                st.warning("⚠️ Please select both a start and destination town.")
            else:
                compute_routes(is_realworld)
                st.rerun()


def render_bookmarks_panel(is_realworld: bool) -> None:
    """Render the collapsible Bookmarks panel on the right side of the layout."""
    bookmarks = st.session_state.get("bookmarks", [])
    count = len(bookmarks)

    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown(
            f"### 🔖 Bookmarks <span style='font-size:14px; font-weight:normal; color:#4B5563;'>({count}/{MAX_BOOKMARKS})</span>",
            unsafe_allow_html=True,
        )
    with head_col2:
        if st.button("✕", key="btn_close_bookmarks_panel", help="Close bookmarks panel"):
            st.session_state.show_bookmarks = False
            st.rerun()

    st.caption(f"Save up to {MAX_BOOKMARKS} favorite places. Bookmarks automatically appear in the dropdowns.")

    # ── 1. Dropdown showing saved bookmarks ──
    st.markdown("#### 📑 Saved Bookmarks Dropdown")
    if bookmarks:
        bm_display_options = [f"⭐ {bm['name']} ({bm.get('address', '')})" for bm in bookmarks]
        selected_bm_idx = st.selectbox(
            "Bookmarks Dropdown:",
            options=list(range(len(bookmarks))),
            format_func=lambda i: bm_display_options[i],
            key="bookmarks_panel_dropdown",
            label_visibility="collapsed",
        )
        selected_bm = bookmarks[selected_bm_idx]
        st.markdown(
            f"""
            <div style="background:#FEF9C3; border:1px solid #FDE047; border-radius:8px; padding:10px 12px; margin-top:4px; margin-bottom:8px;">
                <b style="color:#854D0E;">⭐ {selected_bm['name']}</b><br>
                <span style="font-size:12px; color:#713F12;">{selected_bm.get('address', '')}</span><br>
                <code style="font-size:11px; color:#A16207;">Lat: {selected_bm['lat']:.4f}, Lon: {selected_bm['lon']:.4f}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("🚩 Set as Start", key=f"btn_bm_start_{selected_bm_idx}", use_container_width=True):
                st.session_state.start_name = selected_bm["name"]
                st.session_state.start_coords = (selected_bm["lat"], selected_bm["lon"])
                st.session_state.searched_location = None
                st.session_state.show_route_menu = True
                if st.session_state.dest_coords and st.session_state.dest_name:
                    try:
                        compute_routes(is_realworld)
                    except Exception:
                        st.session_state.route_result = None
                else:
                    st.session_state.route_result = None
                st.rerun()
        with col_q2:
            if st.button("🏁 Set as Dest", key=f"btn_bm_dest_{selected_bm_idx}", use_container_width=True):
                st.session_state.dest_name = selected_bm["name"]
                st.session_state.dest_coords = (selected_bm["lat"], selected_bm["lon"])
                st.session_state.searched_location = None
                st.session_state.show_route_menu = True
                if st.session_state.start_coords and st.session_state.start_name:
                    try:
                        compute_routes(is_realworld)
                    except Exception:
                        st.session_state.route_result = None
                else:
                    st.session_state.route_result = None
                st.rerun()
    else:
        st.info(f"No bookmarks yet. Add up to {MAX_BOOKMARKS} bookmarks below!")

    # ── 2. Manage Bookmarks list with Delete buttons ──
    if bookmarks:
        st.markdown("##### 📍 Manage Saved Places")
        for idx, bm in enumerate(bookmarks):
            col_info, col_del = st.columns([3.0, 1.2])
            with col_info:
                st.markdown(f"**{idx+1}. ⭐ {bm['name']}**")
                st.caption(f"{bm.get('address', '')}")
            with col_del:
                if st.button("🗑️ Delete", key=f"btn_del_bm_item_{idx}", use_container_width=True):
                    ok, msg = delete_bookmark(idx)
                    if ok:
                        st.toast(f"🗑️ {msg}", icon="🗑️")
                    st.rerun()

    # ── 3. Add Bookmark Form with Limit Enforcement ──
    st.markdown("---")
    st.markdown("#### ➕ Add New Bookmark")
    if count >= MAX_BOOKMARKS:
        st.warning(f"⚠️ Bookmark limit reached ({MAX_BOOKMARKS}/{MAX_BOOKMARKS}). Delete an existing bookmark to add a new one.")
    else:
        new_name = st.text_input(
            "Bookmark Name:",
            placeholder="e.g. Home, Office, Gym, Favorite Cafe...",
            key="input_new_bookmark_name",
        )

        all_landmarks = {**WORLDWIDE_LANDMARKS, **RANCHI_LANDMARKS}
        source_choices = ["📍 Preset Landmark", "🔍 Search Address", "🔵 Current Location"]
        if st.session_state.get("last_clicked_coords"):
            source_choices.append("🖱️ Last Clicked Map Point")

        source_type = st.radio(
            "Location Source:",
            source_choices,
            key="radio_bm_source_type",
            horizontal=True,
        )

        cand_lat: Optional[float] = None
        cand_lon: Optional[float] = None
        cand_addr: str = ""

        if source_type == "📍 Preset Landmark":
            chosen_landmark = st.selectbox(
                "Choose Landmark:",
                options=sorted(list(all_landmarks.keys())),
                key="select_bm_landmark",
            )
            if chosen_landmark:
                cand_lat, cand_lon = all_landmarks[chosen_landmark]
                cand_addr = chosen_landmark
        elif source_type == "🔍 Search Address":
            search_query = st.text_input(
                "Search place/address:",
                placeholder="Type location name or address...",
                key="input_bm_search_addr",
            )
            if search_query.strip():
                sugs = search_locations(search_query.strip(), limit=5)
                if sugs:
                    chosen_sug = st.selectbox(
                        "Suggestions:",
                        options=sugs,
                        format_func=lambda s: f"📍 {s.format_display_label()}",
                        key="select_bm_search_sug",
                    )
                    if chosen_sug:
                        cand_lat, cand_lon = chosen_sug.lat, chosen_sug.lon
                        cand_addr = chosen_sug.name
                else:
                    st.caption("No matching places found.")
        elif source_type == "🔵 Current Location":
            cand_lat, cand_lon, cand_addr = st.session_state.my_location
            st.caption(f"Using My Location: **{cand_addr}** (`{cand_lat:.4f}, {cand_lon:.4f}`)")
        elif source_type == "🖱️ Last Clicked Map Point":
            if st.session_state.get("last_clicked_coords"):
                cand_lat, cand_lon = st.session_state.last_clicked_coords
                cand_addr = reverse_geocode(cand_lat, cand_lon)
                st.caption(f"Using Clicked Point: **{cand_addr}** (`{cand_lat:.4f}, {cand_lon:.4f}`)")

        if cand_lat is not None and cand_lon is not None:
            st.caption(f"Coordinates: `{cand_lat:.5f}, {cand_lon:.5f}`")

        if st.button("➕ Add Bookmark", type="primary", use_container_width=True, key="btn_submit_add_bm"):
            if not new_name.strip():
                st.error("Please enter a bookmark name.")
            elif cand_lat is None or cand_lon is None:
                st.error("Please select a valid location for the bookmark.")
            else:
                ok, msg = add_bookmark(new_name.strip(), cand_addr or new_name.strip(), cand_lat, cand_lon)
                if ok:
                    st.toast(f"✅ {msg}", icon="🔖")
                    st.rerun()
                else:
                    st.error(msg)

    # ── 4. JSON Device Storage (Persisted across refreshes) ──
    st.markdown("---")
    st.markdown("#### 💾 Device JSON Storage")
    st.caption("📁 Auto-saved in real time to: `data/bookmarks.json` on your device.")

    json_payload = json.dumps(bookmarks, indent=2, ensure_ascii=False)
    col_export, col_import_btn = st.columns(2)
    with col_export:
        st.download_button(
            label="📥 Export JSON",
            data=json_payload,
            file_name="bookmarks.json",
            mime="application/json",
            use_container_width=True,
            help="Download bookmarks.json to your device",
            key="btn_download_bm_json",
        )
    with col_import_btn:
        if st.button("📤 Import JSON", use_container_width=True, key="btn_toggle_import_bm"):
            st.session_state.show_bm_importer = not st.session_state.get("show_bm_importer", False)
            st.rerun()

    if st.session_state.get("show_bm_importer", False):
        uploaded_file = st.file_uploader(
            "Select bookmarks.json from your device:",
            type=["json"],
            key="bm_file_uploader",
        )
        if uploaded_file is not None:
            try:
                imported_data = json.load(uploaded_file)
                if isinstance(imported_data, list):
                    valid_imported = []
                    for item in imported_data[:MAX_BOOKMARKS]:
                        if isinstance(item, dict) and "name" in item and "lat" in item and "lon" in item:
                            valid_imported.append({
                                "name": str(item["name"]).strip(),
                                "address": str(item.get("address", item["name"])).strip(),
                                "lat": float(item["lat"]),
                                "lon": float(item["lon"]),
                            })
                    if valid_imported:
                        st.session_state.bookmarks = valid_imported
                        save_bookmarks(st.session_state.bookmarks)
                        st.session_state.show_bm_importer = False
                        st.toast(f"✅ Imported {len(valid_imported)} bookmark(s) from JSON!", icon="📥")
                        st.rerun()
                    else:
                        st.error("No valid bookmarks found in JSON file.")
                else:
                    st.error("JSON file must contain a list of bookmark objects.")
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")


def _close_place_info_panel() -> None:
    """Callback to hide the place info panel."""
    st.session_state.show_info_panel = False


def _open_place_info_panel() -> None:
    """Callback to show the place info panel."""
    st.session_state.show_info_panel = True


def render_place_info_panel(is_realworld: bool) -> None:
    """Render the right-side Wikipedia Location Information Panel."""
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown("### 🏛️ Place Info")
    with head_col2:
        st.button("✕", key="btn_close_info_panel", on_click=_close_place_info_panel, help="Hide location information panel")

    # Collect available candidates for the dropdown
    candidates: List[Dict[str, Any]] = []

    # 1. Destination (first priority by default for routes)
    if st.session_state.get("dest_coords") and st.session_state.get("dest_name"):
        candidates.append({
            "key": "destination",
            "type": "Destination",
            "label": f"🏁 Destination: {st.session_state.dest_name}",
            "name": st.session_state.dest_name,
            "lat": st.session_state.dest_coords[0],
            "lon": st.session_state.dest_coords[1],
            "address": st.session_state.get("dest_address", ""),
        })

    # 2. Origin (can switch to origin in dropdown)
    if st.session_state.get("start_coords") and st.session_state.get("start_name"):
        candidates.append({
            "key": "origin",
            "type": "Origin",
            "label": f"🚩 Origin: {st.session_state.start_name}",
            "name": st.session_state.start_name,
            "lat": st.session_state.start_coords[0],
            "lon": st.session_state.start_coords[1],
            "address": st.session_state.get("start_address", ""),
        })

    # 3. Searched location (if active and not identical to destination/origin)
    if st.session_state.get("searched_location"):
        s_lat, s_lon, s_name = st.session_state.searched_location
        is_dup = any(abs(c["lat"] - s_lat) < 1e-5 and abs(c["lon"] - s_lon) < 1e-5 for c in candidates)
        if not is_dup:
            candidates.append({
                "key": "searched",
                "type": "Searched",
                "label": f"📍 Searched: {s_name}",
                "name": s_name,
                "lat": s_lat,
                "lon": s_lon,
                "address": st.session_state.get("searched_address", ""),
            })

    if not candidates:
        st.info("ℹ️ Search a location or select a route to view Wikipedia information.")
        return

    # Determine default selection
    current_target = st.session_state.get("info_panel_target", "destination")
    candidate_keys = [c["key"] for c in candidates]
    if current_target in candidate_keys:
        default_idx = candidate_keys.index(current_target)
    else:
        default_idx = 0

    options = [c["label"] for c in candidates]
    selected_label = st.selectbox(
        "Location in view:",
        options=options,
        index=default_idx,
        key="info_panel_location_select",
        help="Select Destination or Origin to view its Wikipedia details",
    )

    chosen_cand = next(c for c in candidates if c["label"] == selected_label)
    st.session_state.info_panel_target = chosen_cand["key"]

    # Fetch Wikipedia info
    with st.spinner("Fetching Wikipedia details..."):
        wiki_data = fetch_wiki_info_for_location(
            name=chosen_cand["name"],
            lat=chosen_cand["lat"],
            lon=chosen_cand["lon"],
            address=chosen_cand.get("address", ""),
        )

    # 1. Name of the place
    wiki_title = wiki_data.get("name") or chosen_cand["name"]
    searched_name = chosen_cand["name"]
    is_same = wiki_title.strip().lower() == searched_name.strip().lower()
    if is_same:
        sub_info = f"GPS: {chosen_cand['lat']:.5f}, {chosen_cand['lon']:.5f} &bull; {chosen_cand['type']}"
    else:
        sub_info = f"<b>{searched_name}</b> &bull; GPS: {chosen_cand['lat']:.5f}, {chosen_cand['lon']:.5f} &bull; {chosen_cand['type']}"

    st.markdown(
        f"""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 12px; margin-top:6px; margin-bottom:10px;">
            <b style="color:#166534; font-size:15px;">🏛️ {wiki_title}</b><br>
            <span style="color:#15803D; font-size:11.5px;">{sub_info}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Picture (if available)
    picture_url = wiki_data.get("picture")
    if picture_url:
        st.image(picture_url, caption=f"📸 {wiki_title}", use_container_width=True)
    else:
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:8px; padding:8px; text-align:center; color:#64748B; font-size:12px; margin-bottom:10px;">
                🖼️ <i>No photo available on Wikipedia</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Description
    st.markdown("##### 📝 Description")
    desc_text = wiki_data.get("description") or "No description available."
    st.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 12px; font-size:13px; line-height:1.55; color:#334155; margin-bottom:10px; max-height:180px; overflow-y:auto;">
            {desc_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. History
    st.markdown("##### 📜 History")
    hist_text = wiki_data.get("history") or "No historical records available."
    st.markdown(
        f"""
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:10px 12px; font-size:12.5px; line-height:1.55; color:#78350F; margin-bottom:10px; max-height:200px; overflow-y:auto;">
            {hist_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Wiki Link
    wiki_url = wiki_data.get("wiki_link")
    if wiki_url:
        st.markdown(
            f"""
            <div style="margin-top:6px; margin-bottom:8px;">
                <a href="{wiki_url}" target="_blank" style="display:block; text-align:center; background:#2563EB; color:white; padding:8px 12px; border-radius:8px; font-size:13px; font-weight:600; text-decoration:none;">
                    🌐 Read more on Wikipedia ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_right_panel(is_realworld: bool) -> None:
    """Render the right column containing Place Info and/or Bookmarks."""
    show_info = st.session_state.get("show_info_panel", False)
    show_bm = st.session_state.get("show_bookmarks", False)
    if show_info and show_bm:
        tab_info, tab_bm = st.tabs(["🏛️ Place Info", "🔖 Bookmarks"])
        with tab_info:
            render_place_info_panel(is_realworld)
        with tab_bm:
            render_bookmarks_panel(is_realworld)
    elif show_info:
        render_place_info_panel(is_realworld)
    elif show_bm:
        render_bookmarks_panel(is_realworld)


def main() -> None:
    init_session_state()
    _sync_aqi_browser_location()

    # ── Custom CSS for Sliding Drawer & Modern Map Styling ───────────────
    st.markdown(
        """
        <style>
        @keyframes slideInFromLeft {
            0% {
                opacity: 0;
                transform: translateX(-35px);
            }
            100% {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .active-loc-chip {
            background: #F0FDF4;
            border: 1px solid #86EFAC;
            border-radius: 8px;
            padding: 8px 14px;
            margin-top: 6px;
            margin-bottom: 10px;
            font-size: 13.5px;
            color: #166534;
        }
        .my-loc-bar {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #1E293B !important;
        }
        .my-loc-bar span, .my-loc-bar b {
            color: #1E293B !important;
        }
        .route-summary-bar {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            color: #1E293B !important;
        }
        .click-step-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .click-step-badge.pending {
            background: rgba(148, 163, 184, 0.15);
            color: #64748B;
            border: 1px solid #CBD5E1;
        }
        .click-step-badge.active {
            background: rgba(37, 99, 235, 0.15);
            color: #1D4ED8;
            border: 1px solid #93C5FD;
            animation: pulseGlow 1.8s infinite;
        }
        .click-step-badge.done {
            background: rgba(22, 163, 74, 0.15);
            color: #15803D;
            border: 1px solid #86EFAC;
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.35); }
            50% { box-shadow: 0 0 0 6px rgba(37, 99, 235, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Optional Secondary Settings in Sidebar (Collapsed by default) ─────
    with st.sidebar:
        st.markdown("## 🧭 PathFinder Navigation")
        st.caption("Settings & Network Configuration")
        network_mode = st.radio(
            "Road Network",
            [
                "Real-World Map (Worldwide Search, Routing & Ranchi Traffic)",
                "Fictional Regional Map (Stage 1)",
            ],
            index=0,
            key="network_mode",
            help="Real-World Map supports both Ranchi local traffic simulation and worldwide routing. Stage 1 is the fictional regional map.",
        )
        is_realworld = "Real-World" in network_mode

        if st.session_state.route_result is not None:
            if st.button("✕ Reset Active Route", use_container_width=True):
                st.session_state.route_result = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 🍃 Air Quality (WAQI)")
        aqi_sidebar_toggle = st.toggle(
            "Show Air Quality Overlay",
            value=st.session_state.get("show_aqi_mode", False),
            key="sidebar_aqi_toggle",
            on_change=_on_sidebar_aqi_toggle,
            help="Directly activates the WAQI (World Air Quality Index) map layer and scale reading",
        )

        with st.expander("🔑 WAQI API Token Settings", expanded=False):
            st.caption("PathFinder connects to the World Air Quality Index platform. You can get a free token anytime at [aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/).")
            custom_tok = st.text_input(
                "WAQI Token:",
                value=st.session_state.get("waqi_custom_token", ""),
                type="password",
                key="waqi_sidebar_token_input",
                help="Enter your personal WAQI API token (defaults to public demo)",
            )
            if custom_tok != st.session_state.get("waqi_custom_token", ""):
                st.session_state.waqi_custom_token = custom_tok.strip()
                st.rerun()

    # ── Top Section: Search Bar & Current Location Controls ──────────────
    if is_realworld:
        col_search_input, col_search_clear = st.columns([5, 1])
        with col_search_input:
            search_val = st.text_input(
                "Search location, landmark, or address:",
                placeholder="🔍 Search any place worldwide (e.g. Eiffel Tower, Times Square, Tokyo Tower, London, Marine Drive)...",
                key="global_search_input",
                label_visibility="collapsed",
            )
        with col_search_clear:
            st.button(
                "✕ Clear Pin",
                use_container_width=True,
                key="btn_clear_search_pin",
                on_click=clear_global_search_pin,
            )

        if search_val.strip():
            suggestions = search_locations(search_val.strip(), limit=8)
            if suggestions:
                col_sug_select, col_sug_set_start, col_sug_set_dest = st.columns([3, 1, 1])
                with col_sug_select:
                    chosen_suggestion = st.selectbox(
                        "Suggestions (select to show on map):",
                        options=suggestions,
                        format_func=lambda s: f"📍 {s.format_display_label()}",
                        key="search_suggestion_select",
                    )
                    s_key = (
                        chosen_suggestion.lat,
                        chosen_suggestion.lon,
                        chosen_suggestion.name,
                    ) if chosen_suggestion else None

                    if s_key and s_key != st.session_state.get("prev_search_suggestion"):
                        st.session_state.prev_search_suggestion = s_key
                        st.session_state.searched_location = (
                            chosen_suggestion.lat,
                            chosen_suggestion.lon,
                            chosen_suggestion.name,
                        )
                        st.session_state.searched_address = chosen_suggestion.display_name
                        st.session_state.show_info_panel = True
                        st.session_state.info_panel_target = "searched"
                with col_sug_set_start:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🚩 Set as Start", key="btn_set_searched_start", use_container_width=True):
                        if chosen_suggestion:
                            st.session_state.start_name = chosen_suggestion.name
                            st.session_state.start_coords = (chosen_suggestion.lat, chosen_suggestion.lon)
                            st.session_state.start_address = chosen_suggestion.display_name
                            st.session_state.searched_location = None
                            st.session_state.show_route_menu = True
                            st.session_state.show_info_panel = True
                            if st.session_state.dest_coords and st.session_state.dest_name:
                                try:
                                    compute_routes(is_realworld)
                                except Exception:
                                    st.session_state.route_result = None
                            else:
                                st.session_state.route_result = None
                                st.session_state.info_panel_target = "origin"
                            st.rerun()
                with col_sug_set_dest:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🏁 Set as Dest", key="btn_set_searched_dest", use_container_width=True):
                        if chosen_suggestion:
                            st.session_state.dest_name = chosen_suggestion.name
                            st.session_state.dest_coords = (chosen_suggestion.lat, chosen_suggestion.lon)
                            st.session_state.dest_address = chosen_suggestion.display_name
                            st.session_state.searched_location = None
                            st.session_state.show_route_menu = True
                            st.session_state.show_info_panel = True
                            st.session_state.info_panel_target = "destination"
                            if st.session_state.start_coords and st.session_state.start_name:
                                try:
                                    compute_routes(is_realworld)
                                except Exception:
                                    st.session_state.route_result = None
                            else:
                                st.session_state.route_result = None
                            st.rerun()
            else:
                st.warning(f"No locations found matching '{search_val}'. Try another query or GPS coordinates.")

        # ── My Current Location Quick Action Bar ──────────────────────────
        my_lat, my_lon, my_label = st.session_state.my_location
        col_my_info, col_set_btn, col_aqi_btn, col_route_from_my, col_route_to_my = st.columns([2.3, 1.3, 1.3, 1.4, 1.4])
        with col_my_info:
            st.markdown(
                f"""
                <div class="my-loc-bar">
                    <span style="color:#1E293B !important;">
                        <b style="color:#1E293B !important;">🔵 My Location:</b> <b style="color:#1E3A8A !important;">{my_label}</b>
                        <span style="color:#475569 !important; font-size:12px;">(`{my_lat:.4f}, {my_lon:.4f}`)</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_set_btn:
            if st.button("✏️ Change Location", key="btn_toggle_location_picker", use_container_width=True):
                st.session_state.show_location_picker = not st.session_state.get("show_location_picker", False)
                st.rerun()

        with col_aqi_btn:
            aqi_active = st.session_state.get("show_aqi_mode", False)
            btn_txt = "✕ Close AQI" if aqi_active else "🍃 Air Quality"
            btn_tp = "primary" if aqi_active else "secondary"
            if st.button(btn_txt, key="btn_toggle_aqi_top", type=btn_tp, use_container_width=True, help="Toggle WAQI Air Quality Map & Scale"):
                if aqi_active:
                    st.session_state.show_aqi_mode = False
                    st.session_state.aqi_location_pending = False
                else:
                    _enable_aqi_mode()
                st.rerun()

        with col_route_from_my:
            if st.button("🚩 Route From My Location", key="btn_route_from_my_loc", use_container_width=True):
                st.session_state.start_name = MY_LOCATION_LABEL
                st.session_state.start_coords = (my_lat, my_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.session_state.show_info_panel = True
                st.session_state.info_panel_target = "origin"
                st.rerun()

        with col_route_to_my:
            if st.button("🏁 Route To My Location", key="btn_route_to_my_loc", use_container_width=True):
                st.session_state.dest_name = MY_LOCATION_LABEL
                st.session_state.dest_coords = (my_lat, my_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.session_state.show_info_panel = True
                st.session_state.info_panel_target = "destination"
                st.rerun()

        # Location Picker Panel (Quick Area Select or Address Search)
        if st.session_state.get("show_location_picker", False):
            st.info("💡 **Set your location:** Choose your area from the dropdown, search any place, or click anywhere on the map below!")
            col_preset, col_custom_search = st.columns([1.5, 2.5])
            with col_preset:
                is_worldwide = "Worldwide" in st.session_state.get("network_mode", "Worldwide")
                all_lm = {**WORLDWIDE_LANDMARKS, **RANCHI_LANDMARKS} if is_worldwide else RANCHI_LANDMARKS
                preset_choices = ["-- Select your neighborhood / area --"]
                for bm in st.session_state.get("bookmarks", []):
                    preset_choices.append(f"⭐ Bookmark: {bm['name']}")
                preset_choices.extend(sorted(list(all_lm.keys())))
                chosen_preset = st.selectbox(
                    "Quick Area Select:",
                    preset_choices,
                    key="preset_my_location_select",
                )
                if chosen_preset and chosen_preset != "-- Select your neighborhood / area --":
                    if st.button(f"🔵 Set to {chosen_preset}", key="btn_set_preset_loc", use_container_width=True):
                        if chosen_preset.startswith("⭐ Bookmark: "):
                            bm_name = chosen_preset.replace("⭐ Bookmark: ", "")
                            bm_match = next((b for b in st.session_state.get("bookmarks", []) if b["name"] == bm_name), None)
                            if bm_match:
                                st.session_state.my_location = (bm_match["lat"], bm_match["lon"], bm_match["name"])
                                st.session_state.searched_location = st.session_state.my_location
                        elif chosen_preset in all_lm:
                            coords = all_lm[chosen_preset]
                            st.session_state.my_location = (coords[0], coords[1], chosen_preset)
                            st.session_state.searched_location = st.session_state.my_location
                        st.session_state.show_location_picker = False
                        st.session_state.show_info_panel = True
                        st.session_state.info_panel_target = "searched"
                        st.toast(f"✅ Set My Location to '{chosen_preset}'!", icon="🔵")
                        st.rerun()

            with col_custom_search:
                my_loc_query = st.text_input(
                    "Or search any custom address or place:",
                    placeholder="e.g. Eiffel Tower, Times Square, Tokyo Tower, Tagore Hill, Plaza Road...",
                    key="my_loc_search_query",
                )
                if my_loc_query.strip():
                    my_sugs = search_locations(my_loc_query.strip(), limit=5)
                    if my_sugs:
                        col_sug_sel, col_sug_btn = st.columns([3, 1])
                        with col_sug_sel:
                            chosen_my_sug = st.selectbox(
                                "Matches found:",
                                options=my_sugs,
                                format_func=lambda s: f"📍 {s.format_display_label()}",
                                key="my_loc_suggestion_select",
                                label_visibility="collapsed",
                            )
                        with col_sug_btn:
                            if st.button("💾 Save Location", key="btn_save_custom_my_loc", use_container_width=True):
                                if chosen_my_sug:
                                    st.session_state.my_location = (
                                        chosen_my_sug.lat,
                                        chosen_my_sug.lon,
                                        chosen_my_sug.name,
                                    )
                                    st.session_state.searched_location = st.session_state.my_location
                                    st.session_state.show_location_picker = False
                                    st.session_state.show_info_panel = True
                                    st.session_state.info_panel_target = "searched"
                                    st.toast(f"✅ Set My Location to '{chosen_my_sug.name}'!", icon="🔵")
                                    st.rerun()

        # Active Searched Location Pin Banner
        if st.session_state.searched_location:
            s_lat, s_lon, s_name = st.session_state.searched_location
            st.markdown(
                f"""
                <div class="active-loc-chip">
                    <b>📍 Showing on Map:</b> <b>{s_name}</b> <span style="color:#666; font-size:12px;">(Lat: {s_lat:.5f}, Lon: {s_lon:.5f})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 🍃 Air Quality (AQI) Active Banner Card ───────────────────────────
    if st.session_state.get("show_aqi_mode", False):
        if st.session_state.get("aqi_location_pending"):
            st.info("Requesting your location so I can check nearby air quality. Allow location access in your browser.")
        elif st.session_state.get("aqi_location_error"):
            st.warning("I couldn't access your GPS location. Allow location access, then close and reopen Air Quality, or click the map to inspect an area.")
        if st.session_state.get("aqi_inspect_coords"):
            render_aqi_dashboard_card()

    # ── 🚦 Traffic Simulation controls ────────────────────────────────────
    render_traffic_controls(is_realworld)

    # ── Active Route Summary Banner (ONLY when route is calculated and endpoints exist) ───
    has_endpoints = bool(
        (st.session_state.get("start_coords") and st.session_state.get("dest_coords"))
        if is_realworld
        else (st.session_state.get("start_name") and st.session_state.get("dest_name"))
    )
    if not has_endpoints:
        st.session_state.route_result = None
        st.session_state.alternative_results = None
        primary_result = None
    else:
        primary_result = st.session_state.route_result

    if primary_result and primary_result.found:
        dist_label = f"{primary_result.total_distance/1000:.2f} km" if primary_result.total_distance >= 1000 else f"{primary_result.total_distance:.0f} m"
        drive_time = estimate_travel_time(primary_result.total_distance, speed_kmh=30.0)
        if is_realworld and get_traffic_mode():
            _info = analyse_route(load_cached_osm_network(), primary_result.path, get_traffic_mode(), st.session_state.departure_time)
            if _info:
                drive_time = f"{_info.travel_time_seconds / 60:.1f} min estimated"
        walk_time = estimate_travel_time(primary_result.total_distance, speed_kmh=4.5)

        st.markdown(
            f"""
            <div class="route-summary-bar">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:16px; font-weight:700; color:#1E293B;">🚩 {st.session_state.start_name} &nbsp;➔&nbsp; 🏁 {st.session_state.dest_name}</span>
                        <div style="margin-top:6px; display:flex; gap:18px; font-size:14px; color:#475569;">
                            <span>📏 <b>Distance:</b> {dist_label}</span>
                            <span>🚗 <b>Drive:</b> ~{drive_time}</span>
                            <span>🚶 <b>Walk:</b> ~{walk_time}</span>
                            <span>🛣️ <b>Steps:</b> {len(primary_result.legs)} legs</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, _ = st.columns([1.2, 1.2, 7.6])
        with col_c1:
            if st.button("✕ Clear Route", key="btn_clear_active_route", use_container_width=True):
                st.session_state.route_result = None
                st.session_state.alternative_results = None
                if not st.session_state.get("searched_location"):
                    st.session_state.show_info_panel = False
                else:
                    st.session_state.info_panel_target = "searched"
                st.rerun()
        with col_c2:
            if not st.session_state.show_route_menu:
                if st.button("✏️ Edit Route", key="btn_edit_active_route", use_container_width=True):
                    st.session_state.show_route_menu = True
                    st.rerun()

        alt_results = st.session_state.get("alternative_results")
        if alt_results and len(alt_results) > 1:
            st.markdown("<div style='margin-top:10px; margin-bottom:5px; font-weight:600; color:#334155;'>🔄 Alternative Routes:</div>", unsafe_allow_html=True)
            options = []
            for i, res in enumerate(alt_results):
                r_dist = f"{res.total_distance/1000:.2f} km" if res.total_distance >= 1000 else f"{res.total_distance:.0f} m"
                r_time = estimate_travel_time(res.total_distance, speed_kmh=30.0)
                if get_traffic_mode() and res.total_cost is not None:
                    options.append(f"Route {i+1} ({r_dist}, ~{res.total_cost/60:.1f} min)")
                else:
                    options.append(f"Route {i+1} ({r_dist}, ~{r_time})")
                
            selected_route = st.radio(
                "Select Route", 
                options=options, 
                index=st.session_state.get("active_route_index", 0), 
                horizontal=True, 
                label_visibility="collapsed"
            )
            
            selected_idx = options.index(selected_route)
            if selected_idx != st.session_state.get("active_route_index", 0):
                st.session_state.active_route_index = selected_idx
                st.session_state.route_result = alt_results[selected_idx]
                st.rerun()

        render_route_information(primary_result, is_realworld)

    elif primary_result and not primary_result.found:
        st.error(f"❌ No route found between '{st.session_state.start_name}' and '{st.session_state.dest_name}'.")

    # ── Interactive Click-to-Route Dashboard Bar (Zero-typing direct map interaction) ──
    render_click_to_route_dashboard_bar(is_realworld)

    # ── Main Area: Directions Drawer (Left), Map (Center), Bookmarks / Info Panel (Right) ──
    show_left = st.session_state.show_route_menu
    show_right = st.session_state.get("show_info_panel", False) or st.session_state.get("show_bookmarks", False)
    active_map_height = st.session_state.get("map_height_setting", 700)

    if show_left and show_right:
        col_menu, col_map, col_r = st.columns([1.1, 1.8, 1.1], gap="medium")
        with col_menu:
            render_directions_drawer(is_realworld)
        with col_map:
            _render_map(primary_result, is_realworld, map_height=active_map_height)
        with col_r:
            render_right_panel(is_realworld)
    elif show_left and not show_right:
        col_menu, col_map = st.columns([1.1, 2.9], gap="medium")
        with col_menu:
            render_directions_drawer(is_realworld)
        with col_map:
            has_loc = bool(st.session_state.get("dest_coords") or st.session_state.get("start_coords") or st.session_state.get("searched_location"))
            if has_loc and not st.session_state.get("show_info_panel", False):
                c_sp, c_unhide = st.columns([7.8, 2.2])
                with c_unhide:
                    st.button("🏛️ Show Place Info", key="btn_unhide_info_side_l", on_click=_open_place_info_panel, use_container_width=True, help="Unhide location info panel")
            _render_map(primary_result, is_realworld, map_height=active_map_height)
    elif not show_left and show_right:
        col_map, col_r = st.columns([2.9, 1.1], gap="medium")
        with col_map:
            _render_map(primary_result, is_realworld, map_height=active_map_height)
        with col_r:
            render_right_panel(is_realworld)
    else:
        has_loc = bool(st.session_state.get("dest_coords") or st.session_state.get("start_coords") or st.session_state.get("searched_location"))
        if has_loc and not st.session_state.get("show_info_panel", False):
            c_sp, c_unhide = st.columns([8.2, 1.8])
            with c_unhide:
                st.button("🏛️ Show Place Info", key="btn_unhide_info_side_full", on_click=_open_place_info_panel, use_container_width=True, help="Unhide location info panel")
        _render_map(primary_result, is_realworld, map_height=active_map_height)

    # ── Bottom Control Bar: Buttons to Open/Close Directions, Info Panel, and Bookmarks ──
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col_bottom_left, col_bottom_info, col_bottom_aqi, col_bottom_space, col_bottom_right = st.columns([2.0, 2.0, 2.0, 2.0, 2.0])
    with col_bottom_left:
        if not st.session_state.show_route_menu:
            if st.button("🧭 Go (Directions)", type="primary", use_container_width=True, key="btn_go_bottom_left"):
                st.session_state.show_route_menu = True
                st.rerun()
        else:
            if st.button("✕ Close Directions", use_container_width=True, key="btn_close_directions_bottom"):
                st.session_state.show_route_menu = False
                st.rerun()

    with col_bottom_info:
        has_any_location = bool(
            st.session_state.get("dest_coords")
            or st.session_state.get("start_coords")
            or st.session_state.get("searched_location")
        )
        if not st.session_state.get("show_info_panel", False):
            st.button("🏛️ Place Info", type="secondary", use_container_width=True, key="btn_open_info_bottom", disabled=not has_any_location, on_click=_open_place_info_panel, help="View Wikipedia information for location or route")
        else:
            st.button("✕ Hide Place Info", use_container_width=True, key="btn_close_info_bottom", on_click=_close_place_info_panel)

    with col_bottom_aqi:
        aqi_active = st.session_state.get("show_aqi_mode", False)
        btn_txt = "✕ Close AQI" if aqi_active else "🍃 Air Quality"
        btn_tp = "primary" if aqi_active else "secondary"
        if st.button(btn_txt, key="btn_toggle_aqi_bottom", type=btn_tp, use_container_width=True, help="Toggle WAQI Air Quality Map & Scale"):
            if aqi_active:
                st.session_state.show_aqi_mode = False
                st.session_state.aqi_location_pending = False
            else:
                _enable_aqi_mode()
            st.rerun()

    with col_bottom_right:
        if not st.session_state.show_bookmarks:
            if st.button("🔖 Bookmarks", type="secondary" if not st.session_state.show_route_menu else "primary", use_container_width=True, key="btn_open_bookmarks_bottom"):
                st.session_state.show_bookmarks = True
                st.rerun()
        else:
            if st.button("✕ Close Bookmarks", use_container_width=True, key="btn_close_bookmarks_bottom"):
                st.session_state.show_bookmarks = False
                st.rerun()

    # ── Turn-by-Turn Guidance (Collapsible, only when route active) ────────
    if primary_result and primary_result.found:
        with st.expander("🗺️ Turn-by-Turn Navigation Itinerary", expanded=False):
            corridors: List[Tuple[str, float, int]] = []
            if primary_result.legs:
                curr_road = primary_result.legs[0].road_name or "Local Road"
                curr_dist = 0.0
                curr_count = 0
                for leg in primary_result.legs:
                    r_name = leg.road_name or "Local Road / Connector"
                    if r_name == curr_road:
                        curr_dist += leg.distance
                        curr_count += 1
                    else:
                        corridors.append((curr_road, curr_dist, curr_count))
                        curr_road = r_name
                        curr_dist = leg.distance
                        curr_count = 1
                corridors.append((curr_road, curr_dist, curr_count))

                for idx, (road, dist, segs) in enumerate(corridors, start=1):
                    d_str = f"{dist/1000:.2f} km" if dist >= 1000 else f"{dist:.0f} m"
                    seg_info = f"({segs} intersections)" if segs > 1 else "(single segment)"
                    st.markdown(f"**{idx}.** Follow **[{road}]** for `{d_str}` &nbsp;<span style='color:gray;'>{seg_info}</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
