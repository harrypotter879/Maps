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
import math
from datetime import time
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit_folium import st_folium

from graph import Graph, create_sample_road_network
from astar import find_shortest_path_astar, find_alternative_routes
from dijkstra import PathResult
from osm_loader import get_ranchi_road_network, RANCHI_LANDMARKS
from geocoder import search_locations, reverse_geocode, GeocodedLocation
from map_view import build_folium_map, build_empty_map
from display import format_distance
from traffic import (
    TRAFFIC_ICONS, TRAFFIC_MODE_LABELS, HIGH, MEDIUM, LOW,
    SHORTEST_DISTANCE, TRAFFIC_AWARE, PERIOD_LABELS,
    analyse_route, apply_traffic_mode, normalize_departure_time, get_traffic_period,
)

MY_LOCATION_LABEL = "🔵 My Current Location"

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


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "my_location" not in st.session_state:
        # Default My Current Location set to Kairali School, Ranchi
        st.session_state.my_location = (23.3191843, 85.2987681, "Kairali School, Ranchi")

    if "start_name" not in st.session_state:
        st.session_state.start_name = "Albert Ekka Chowk"
        st.session_state.start_coords = RANCHI_LANDMARKS["Albert Ekka Chowk"]

    if "dest_name" not in st.session_state:
        st.session_state.dest_name = "Ranchi Railway Station"
        st.session_state.dest_coords = RANCHI_LANDMARKS["Ranchi Railway Station"]

    # NO route by default at startup! Clean map until user requests directions
    if "route_result" not in st.session_state:
        st.session_state.route_result = None
    if "alternative_results" not in st.session_state:
        st.session_state.alternative_results = None
    if "active_route_index" not in st.session_state:
        st.session_state.active_route_index = 0

    if "searched_location" not in st.session_state:
        st.session_state.searched_location = None  # (lat, lon, name)

    # Route menu drawer is closed by default at startup!
    if "show_route_menu" not in st.session_state:
        st.session_state.show_route_menu = False

    if "last_clicked_coords" not in st.session_state:
        st.session_state.last_clicked_coords = None

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
    if "baseline_result" not in st.session_state:
        st.session_state.baseline_result = None  # shortest-by-distance route, for comparison
    if "traffic_notice" not in st.session_state:
        st.session_state.traffic_notice = None


def get_traffic_mode() -> Optional[str]:
    """Return the active route objective for the Ranchi network."""
    choice = st.session_state.get("traffic_choice", SHORTEST_DISTANCE)
    return TRAFFIC_AWARE if choice == TRAFFIC_AWARE else None


def compute_routes(is_realworld: bool) -> None:
    """
    Run the existing route search (A* + alternatives) and store results in session state.
    Uses physical distance or modeled travel time according to the selected mode.
    Raises on invalid locations (callers decide how to show the error).
    """
    mode = get_traffic_mode() if is_realworld else None
    departure = normalize_departure_time(st.session_state.get("departure_time"))
    cost_fn = apply_traffic_mode(mode, departure)

    if is_realworld:
        graph = load_cached_osm_network()
        start_lat, start_lon = st.session_state.start_coords
        dest_lat, dest_lon = st.session_state.dest_coords
        start_node, _ = graph.find_nearest_node(start_lat, start_lon)
        dest_node, _ = graph.find_nearest_node(dest_lat, dest_lon)
    else:
        graph = load_fictional_network()
        start_node, dest_node = st.session_state.start_name, st.session_state.dest_name

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

    # The physical shortest path is retained for an honest comparison.
    st.session_state.baseline_result = (
        find_shortest_path_astar(graph, start_node, dest_node) if mode else None
    )


def on_routing_settings_change() -> None:
    """Recalculate an active route after its mode or departure time changes."""
    is_realworld = "Real-World" in st.session_state.get("network_mode", "Real-World")
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
        "Evening Peak · 06:00 PM": time(18, 0),
        "Night · 10:00 PM": time(22, 0),
    }
    st.session_state.departure_time = presets[st.session_state.time_preset]
    on_routing_settings_change()


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
            c1, c2 = st.columns([1, 1.6])
            with c1:
                st.time_input("Departure time", key="departure_time", on_change=on_routing_settings_change)
            with c2:
                st.selectbox("Quick preset", ["Morning Peak · 08:30", "Midday · 01:00 PM", "Evening Peak · 06:00 PM", "Night · 10:00 PM"], key="time_preset", on_change=_apply_time_preset)
            period = PERIOD_LABELS.get(get_traffic_period(st.session_state.departure_time), "")
            st.caption(f"Modeled period: {period}. {TRAFFIC_ICONS[LOW]} Low  ·  {TRAFFIC_ICONS[MEDIUM]} Moderate  ·  {TRAFFIC_ICONS[HIGH]} High")
            st.caption("Traffic is estimated from road characteristics and modeled time patterns; it is not live traffic.")
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
            st.info("The current route remains the best estimated choice for these settings.")
        elif kind == "mode_only" and mode:
            st.info("Traffic-aware estimates are ready. Choose Find Route to calculate a recommendation.")
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
        c2.metric("Estimated travel time", f"{est_min:.0f} min", help="Modeled estimate based on road class and departure time.")
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
                f"Route comparison · Shortest distance: {_fmt_dist(baseline.total_distance)} / {base_min:.0f} min; "
                f"recommended: {_fmt_dist(result.total_distance)} / {est_min:.0f} min."
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

    st.session_state.start_name = st.session_state.dest_name
    st.session_state.start_coords = st.session_state.dest_coords

    st.session_state.dest_name = old_start_name
    st.session_state.dest_coords = old_start_coords

    # Clear previous route so user re-computes in the opposite direction
    st.session_state.route_result = None


def clear_global_search_pin() -> None:
    """Callback function to clear search pin and search input before widget instantiation."""
    st.session_state.searched_location = None
    st.session_state.global_search_input = ""


def _render_map(primary_result: Optional[PathResult], is_realworld: bool, map_height: int = 580) -> None:
    """Helper to render interactive Folium map with current points, blue dot marker, or clean state."""
    if primary_result and primary_result.found:
        graph = load_cached_osm_network() if is_realworld else load_fictional_network()
        alt_res = st.session_state.get("alternative_results")
        active_idx = st.session_state.get("active_route_index", 0)

        if is_realworld:
            folium_map = build_folium_map(
                graph=graph,
                result=primary_result,
                start_coord=st.session_state.start_coords,
                dest_coord=st.session_state.dest_coords,
                start_label=st.session_state.start_name,
                dest_label=st.session_state.dest_name,
                searched_point=st.session_state.searched_location,
                current_location=st.session_state.my_location,
                alternative_results=alt_res,
                active_route_index=active_idx,
                show_hud=False,  # Keep map canvas clean!
                traffic_mode=get_traffic_mode(),
                departure_time=st.session_state.departure_time,
            )
        else:
            folium_map = build_empty_map(center=(23.3191843, 85.2987681), zoom=15)
    else:
        # Clean start map: Centered on My Location (Kairali School, Ranchi)!
        folium_map = build_empty_map(
            center=(st.session_state.my_location[0], st.session_state.my_location[1]),
            zoom=15,
            start_point=None,
            dest_point=None,
            searched_point=st.session_state.searched_location,
            current_location=st.session_state.my_location,
            graph=load_cached_osm_network() if (is_realworld and get_traffic_mode()) else None,
            traffic_mode=get_traffic_mode() if is_realworld else None,
            departure_time=st.session_state.departure_time,
        )

    map_data = st_folium(
        folium_map,
        width=None,
        height=map_height,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        st.info(f"🖱️ **Map Click Detected:** `Latitude: {click_lat:.5f}, Longitude: {click_lon:.5f}`")
        col_c1, col_c2, col_c3 = st.columns([1.3, 1.3, 1.4])
        with col_c1:
            if st.button("📍 Set as Start & Open Go", key="btn_click_start"):
                st.session_state.start_name = reverse_geocode(click_lat, click_lon)
                st.session_state.start_coords = (click_lat, click_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.rerun()
        with col_c2:
            if st.button("🎯 Set as Dest & Open Go", key="btn_click_dest"):
                st.session_state.dest_name = reverse_geocode(click_lat, click_lon)
                st.session_state.dest_coords = (click_lat, click_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.rerun()
        with col_c3:
            if st.button("🔵 Set as My Location", key="btn_click_my_loc"):
                loc_label = reverse_geocode(click_lat, click_lon)
                st.session_state.my_location = (click_lat, click_lon, loc_label)
                st.session_state.searched_location = st.session_state.my_location
                st.rerun()


def main() -> None:
    init_session_state()

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
        .route-drawer-card {
            animation: slideInFromLeft 0.28s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            background: #ffffff;
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            border: 1px solid #E2E8F0;
            margin-bottom: 12px;
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
            ["Real-World Map (Ranchi, India)", "Fictional Regional Map (Stage 1)"],
            index=0,
            key="network_mode",
            help="Choose between real-world OpenStreetMap data and the Stage 1 fictional regional network.",
        )
        is_realworld = "Real-World" in network_mode

        if st.session_state.route_result is not None:
            if st.button("✕ Reset Active Route", use_container_width=True):
                st.session_state.route_result = None
                st.rerun()

    # ── Top Section: Search Bar & Current Location Controls ──────────────
    if is_realworld:
        col_search_input, col_search_clear = st.columns([5, 1])
        with col_search_input:
            search_val = st.text_input(
                "Search location, landmark, or address:",
                placeholder="🔍 Search location or address (e.g. Radisson Blu, Albert Ekka, Nucleus Mall, Station Road)...",
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
                    if chosen_suggestion:
                        st.session_state.searched_location = (
                            chosen_suggestion.lat,
                            chosen_suggestion.lon,
                            chosen_suggestion.name,
                        )
                with col_sug_set_start:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🚩 Set as Start", key="btn_set_searched_start", use_container_width=True):
                        if chosen_suggestion:
                            st.session_state.start_name = chosen_suggestion.name
                            st.session_state.start_coords = (chosen_suggestion.lat, chosen_suggestion.lon)
                            st.session_state.show_route_menu = True
                            st.session_state.route_result = None
                            st.rerun()
                with col_sug_set_dest:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🏁 Set as Dest", key="btn_set_searched_dest", use_container_width=True):
                        if chosen_suggestion:
                            st.session_state.dest_name = chosen_suggestion.name
                            st.session_state.dest_coords = (chosen_suggestion.lat, chosen_suggestion.lon)
                            st.session_state.show_route_menu = True
                            st.session_state.route_result = None
                            st.rerun()
            else:
                st.warning(f"No locations found matching '{search_val}'. Try another query or GPS coordinates.")

        # ── My Current Location Quick Action Bar ──────────────────────────
        my_lat, my_lon, my_label = st.session_state.my_location
        col_my_info, col_set_btn, col_route_from_my, col_route_to_my = st.columns([2.6, 1.4, 1.4, 1.4])
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

        with col_route_from_my:
            if st.button("🚩 Route From My Location", key="btn_route_from_my_loc", use_container_width=True):
                st.session_state.start_name = MY_LOCATION_LABEL
                st.session_state.start_coords = (my_lat, my_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.rerun()

        with col_route_to_my:
            if st.button("🏁 Route To My Location", key="btn_route_to_my_loc", use_container_width=True):
                st.session_state.dest_name = MY_LOCATION_LABEL
                st.session_state.dest_coords = (my_lat, my_lon)
                st.session_state.show_route_menu = True
                st.session_state.route_result = None
                st.rerun()

        # Location Picker Panel (Quick Area Select or Address Search)
        if st.session_state.get("show_location_picker", False):
            st.info("💡 **Set your location:** Choose your area from the dropdown, search any place, or click anywhere on the map below!")
            col_preset, col_custom_search = st.columns([1.5, 2.5])
            with col_preset:
                preset_choices = ["-- Select your neighborhood / area --"] + sorted(list(RANCHI_LANDMARKS.keys()))
                chosen_preset = st.selectbox(
                    "Quick Area Select:",
                    preset_choices,
                    key="preset_my_location_select",
                )
                if chosen_preset and chosen_preset != "-- Select your neighborhood / area --":
                    if st.button(f"🔵 Set to {chosen_preset}", key="btn_set_preset_loc", use_container_width=True):
                        coords = RANCHI_LANDMARKS[chosen_preset]
                        st.session_state.my_location = (coords[0], coords[1], chosen_preset)
                        st.session_state.searched_location = st.session_state.my_location
                        st.session_state.show_location_picker = False
                        st.toast(f"✅ Set My Location to '{chosen_preset}'!", icon="🔵")
                        st.rerun()

            with col_custom_search:
                my_loc_query = st.text_input(
                    "Or search any custom address or place:",
                    placeholder="e.g. Tagore Hill, Harmu Housing Colony, Plaza Road, Station Road...",
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

    # ── 🚦 Traffic Simulation controls ────────────────────────────────────
    render_traffic_controls(is_realworld)

    # ── Active Route Summary Banner (ONLY when route is calculated) ───────
    primary_result: Optional[PathResult] = st.session_state.route_result
    if primary_result and primary_result.found:
        dist_label = f"{primary_result.total_distance/1000:.2f} km" if primary_result.total_distance >= 1000 else f"{primary_result.total_distance:.0f} m"
        drive_time = estimate_travel_time(primary_result.total_distance, speed_kmh=30.0)
        if is_realworld and get_traffic_mode():
            _info = analyse_route(load_cached_osm_network(), primary_result.path, get_traffic_mode(), st.session_state.departure_time)
            if _info:
                drive_time = f"{_info.travel_time_seconds / 60:.0f} min estimated"
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
                    options.append(f"Route {i+1} ({r_dist}, ~{res.total_cost/60:.0f} min)")
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

    # ── Main Area: Sliding Route Drawer vs Full-Width Map ─────────────────
    if st.session_state.show_route_menu:
        col_menu, col_map = st.columns([1.1, 2.9], gap="medium")

        # Sliding Directions Drawer on Left
        with col_menu:
            st.markdown('<div class="route-drawer-card">', unsafe_allow_html=True)
            head_col1, head_col2 = st.columns([4, 1])
            with head_col1:
                st.markdown("### 🧭 Directions")
            with head_col2:
                if st.button("✕", key="btn_close_drawer", help="Close directions menu"):
                    st.session_state.show_route_menu = False
                    st.rerun()

            if is_realworld:
                # Build Origin options list, preserving custom/searched location if active
                start_custom_label = f"📍 {st.session_state.start_name}" if (
                    st.session_state.start_name
                    and st.session_state.start_name != MY_LOCATION_LABEL
                    and st.session_state.start_name not in RANCHI_LANDMARKS
                ) else None

                start_options = [MY_LOCATION_LABEL]
                if start_custom_label:
                    start_options.append(start_custom_label)
                start_options.append("🔍 Search Address / Custom Place...")
                start_options.extend(RANCHI_LANDMARKS.keys())

                # From (Origin)
                st.markdown("**From (Origin):**")
                if start_custom_label:
                    curr_start_idx = start_options.index(start_custom_label)
                elif st.session_state.start_name in start_options:
                    curr_start_idx = start_options.index(st.session_state.start_name)
                else:
                    curr_start_idx = 0

                start_mode = st.selectbox(
                    "Origin",
                    start_options,
                    index=curr_start_idx,
                    key="drawer_start_select",
                    label_visibility="collapsed",
                )
                if start_mode == MY_LOCATION_LABEL:
                    st.session_state.start_name = MY_LOCATION_LABEL
                    st.session_state.start_coords = (st.session_state.my_location[0], st.session_state.my_location[1])
                elif start_mode == start_custom_label:
                    # Keep existing custom coordinates and name
                    pass
                elif start_mode == "🔍 Search Address / Custom Place...":
                    start_query = st.text_input(
                        "Search Origin",
                        value=st.session_state.start_name if st.session_state.start_name not in RANCHI_LANDMARKS and st.session_state.start_name != MY_LOCATION_LABEL else "",
                        placeholder="Type address or GPS coordinates...",
                        key="drawer_start_query",
                        label_visibility="collapsed",
                    )
                    if start_query.strip():
                        sugs = search_locations(start_query.strip(), limit=8)
                        if sugs:
                            chosen_s = st.selectbox(
                                "Matches",
                                options=sugs,
                                format_func=lambda s: f"📍 {s.format_display_label()}",
                                key="drawer_chosen_start_sug",
                                label_visibility="collapsed",
                            )
                            st.session_state.start_name = chosen_s.name
                            st.session_state.start_coords = (chosen_s.lat, chosen_s.lon)
                else:
                    st.session_state.start_name = start_mode
                    st.session_state.start_coords = RANCHI_LANDMARKS[start_mode]

                st.caption(f"📍 `{st.session_state.start_coords[0]:.4f}, {st.session_state.start_coords[1]:.4f}`")

                # Swap button
                col_swap, _ = st.columns([1.5, 1])
                with col_swap:
                    st.button("⇄ Swap Start & Dest", on_click=swap_locations, use_container_width=True)

                # Build Destination options list, preserving custom/searched location if active
                dest_custom_label = f"🎯 {st.session_state.dest_name}" if (
                    st.session_state.dest_name
                    and st.session_state.dest_name != MY_LOCATION_LABEL
                    and st.session_state.dest_name not in RANCHI_LANDMARKS
                ) else None

                dest_options = [MY_LOCATION_LABEL]
                if dest_custom_label:
                    dest_options.append(dest_custom_label)
                dest_options.append("🔍 Search Address / Custom Place...")
                dest_options.extend(RANCHI_LANDMARKS.keys())

                # To (Destination)
                st.markdown("**To (Destination):**")
                if dest_custom_label:
                    curr_dest_idx = dest_options.index(dest_custom_label)
                elif st.session_state.dest_name in dest_options:
                    curr_dest_idx = dest_options.index(st.session_state.dest_name)
                else:
                    curr_dest_idx = min(3, len(dest_options) - 1)

                dest_mode = st.selectbox(
                    "Destination",
                    dest_options,
                    index=curr_dest_idx,
                    key="drawer_dest_select",
                    label_visibility="collapsed",
                )
                if dest_mode == MY_LOCATION_LABEL:
                    st.session_state.dest_name = MY_LOCATION_LABEL
                    st.session_state.dest_coords = (st.session_state.my_location[0], st.session_state.my_location[1])
                elif dest_mode == dest_custom_label:
                    # Keep existing custom coordinates and name
                    pass
                elif dest_mode == "🔍 Search Address / Custom Place...":
                    dest_query = st.text_input(
                        "Search Destination",
                        value=st.session_state.dest_name if st.session_state.dest_name not in RANCHI_LANDMARKS and st.session_state.dest_name != MY_LOCATION_LABEL else "",
                        placeholder="Type address or GPS coordinates...",
                        key="drawer_dest_query",
                        label_visibility="collapsed",
                    )
                    if dest_query.strip():
                        sugs = search_locations(dest_query.strip(), limit=8)
                        if sugs:
                            chosen_d = st.selectbox(
                                "Matches",
                                options=sugs,
                                format_func=lambda s: f"🎯 {s.format_display_label()}",
                                key="drawer_chosen_dest_sug",
                                label_visibility="collapsed",
                            )
                            st.session_state.dest_name = chosen_d.name
                            st.session_state.dest_coords = (chosen_d.lat, chosen_d.lon)
                else:
                    st.session_state.dest_name = dest_mode
                    st.session_state.dest_coords = RANCHI_LANDMARKS[dest_mode]

                st.caption(f"🎯 `{st.session_state.dest_coords[0]:.4f}, {st.session_state.dest_coords[1]:.4f}`")

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
                    try:
                        with st.spinner("Finding optimal route..."):
                            compute_routes(is_realworld)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error mapping coordinates: {e}")
                else:
                    compute_routes(is_realworld)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # Right Column: Map
        with col_map:
            _render_map(primary_result, is_realworld, map_height=560)

    else:
        # Full-width Map View when menu is closed
        _render_map(primary_result, is_realworld, map_height=600)

        # Single "Go" Button at Bottom-Left of the Map
        col_go, _ = st.columns([1.8, 8.2])
        with col_go:
            if st.button("🧭 Go (Directions)", type="primary", use_container_width=True, key="btn_go_bottom_left"):
                st.session_state.show_route_menu = True
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
