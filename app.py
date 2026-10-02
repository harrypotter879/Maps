"""
PathFinder — Interactive Real-World Navigation UI.
Built with Streamlit, Folium, and OpenStreetMap (OSMnx & Nominatim).

Features:
- Clean startup map with zero pre-loaded directions or cluttered routes
- Dedicated Top Search Bar with live autocomplete suggestions and instant pin drops
- Sliding 'Go' Directions Drawer triggered by a single button at bottom-left
- Origin / Destination selection with instant Swap button
- Interactive Folium map with road-geometry routes and click-to-select endpoints
- User-friendly navigation metrics: Total Distance, Estimated Drive Time, Estimated Walk Time
- Turn-by-turn guidance corridor itinerary
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit_folium import st_folium

from graph import Graph, create_sample_road_network
from astar import find_shortest_path_astar
from dijkstra import PathResult
from osm_loader import get_ranchi_road_network, RANCHI_LANDMARKS
from geocoder import search_locations, reverse_geocode, GeocodedLocation
from map_view import build_folium_map, build_empty_map
from display import format_distance


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
    if "start_name" not in st.session_state:
        st.session_state.start_name = "Albert Ekka Chowk"
        st.session_state.start_coords = RANCHI_LANDMARKS["Albert Ekka Chowk"]

    if "dest_name" not in st.session_state:
        st.session_state.dest_name = "Ranchi Railway Station"
        st.session_state.dest_coords = RANCHI_LANDMARKS["Ranchi Railway Station"]

    # NO route by default at startup! Clean map until user requests directions
    if "route_result" not in st.session_state:
        st.session_state.route_result = None

    if "searched_location" not in st.session_state:
        st.session_state.searched_location = None  # (lat, lon, name)

    # Route menu drawer is closed by default at startup!
    if "show_route_menu" not in st.session_state:
        st.session_state.show_route_menu = False

    if "last_clicked_coords" not in st.session_state:
        st.session_state.last_clicked_coords = None


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


def _render_map(primary_result: Optional[PathResult], is_realworld: bool, map_height: int = 580) -> None:
    """Helper to render interactive Folium map with current points or clean state."""
    if primary_result and primary_result.found:
        graph = load_cached_osm_network() if is_realworld else load_fictional_network()
        if is_realworld:
            folium_map = build_folium_map(
                graph=graph,
                result=primary_result,
                start_coord=st.session_state.start_coords,
                dest_coord=st.session_state.dest_coords,
                start_label=st.session_state.start_name,
                dest_label=st.session_state.dest_name,
                searched_point=st.session_state.searched_location,
                show_hud=False,  # Keep map canvas clean!
            )
        else:
            folium_map = build_empty_map(center=(23.3699, 85.3253), zoom=14)
    else:
        # Clean start map: No route path, no default start/dest flag markers!
        folium_map = build_empty_map(
            center=st.session_state.start_coords,
            zoom=14,
            start_point=None,
            dest_point=None,
            searched_point=st.session_state.searched_location,
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
        col_c1, col_c2, _ = st.columns([1.3, 1.3, 2.4])
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
        .route-summary-bar {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
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
            help="Choose between real-world OpenStreetMap data and the Stage 1 fictional regional network.",
        )
        is_realworld = "Real-World" in network_mode

        if st.session_state.route_result is not None:
            if st.button("✕ Reset Active Route", use_container_width=True):
                st.session_state.route_result = None
                st.rerun()

    # ── Top Section: Clean Search Bar ─────────────────────────────────────
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
            if st.button("✕ Clear Pin", use_container_width=True, key="btn_clear_search_pin"):
                st.session_state.searched_location = None
                st.rerun()

        if search_val.strip():
            suggestions = search_locations(search_val.strip(), limit=5)
            if suggestions:
                col_sug_select, col_sug_set_start, col_sug_set_dest = st.columns([3, 1, 1])
                with col_sug_select:
                    chosen_suggestion = st.selectbox(
                        "Suggestions (select to show on map):",
                        options=suggestions,
                        format_func=lambda s: f"📍 {s.name} — {s.display_name.split(',')[0]} ({s.source.title()})",
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

        # Active Location Pin Banner
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

    # ── Active Route Summary Banner (ONLY when route is calculated) ───────
    primary_result: Optional[PathResult] = st.session_state.route_result
    if primary_result and primary_result.found:
        dist_label = f"{primary_result.total_distance/1000:.2f} km" if primary_result.total_distance >= 1000 else f"{primary_result.total_distance:.0f} m"
        drive_time = estimate_travel_time(primary_result.total_distance, speed_kmh=30.0)
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
                st.rerun()
        with col_c2:
            if not st.session_state.show_route_menu:
                if st.button("✏️ Edit Route", key="btn_edit_active_route", use_container_width=True):
                    st.session_state.show_route_menu = True
                    st.rerun()

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
                landmark_options = ["🔍 Search Address / Custom Place..."] + list(RANCHI_LANDMARKS.keys())

                # From (Origin)
                st.markdown("**From (Origin):**")
                start_mode = st.selectbox(
                    "Origin",
                    landmark_options,
                    index=landmark_options.index(st.session_state.start_name) if st.session_state.start_name in landmark_options else 0,
                    key="drawer_start_select",
                    label_visibility="collapsed",
                )
                if start_mode == "🔍 Search Address / Custom Place...":
                    start_query = st.text_input(
                        "Search Origin",
                        value=st.session_state.start_name if st.session_state.start_name not in RANCHI_LANDMARKS else "",
                        placeholder="Type address or GPS coordinates...",
                        key="drawer_start_query",
                        label_visibility="collapsed",
                    )
                    if start_query.strip():
                        sugs = search_locations(start_query.strip(), limit=4)
                        if sugs:
                            chosen_s = st.selectbox(
                                "Matches",
                                options=sugs,
                                format_func=lambda s: f"📍 {s.name} ({s.source.title()})",
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

                # To (Destination)
                st.markdown("**To (Destination):**")
                dest_mode = st.selectbox(
                    "Destination",
                    landmark_options,
                    index=landmark_options.index(st.session_state.dest_name) if st.session_state.dest_name in landmark_options else 0,
                    key="drawer_dest_select",
                    label_visibility="collapsed",
                )
                if dest_mode == "🔍 Search Address / Custom Place...":
                    dest_query = st.text_input(
                        "Search Destination",
                        value=st.session_state.dest_name if st.session_state.dest_name not in RANCHI_LANDMARKS else "",
                        placeholder="Type address or GPS coordinates...",
                        key="drawer_dest_query",
                        label_visibility="collapsed",
                    )
                    if dest_query.strip():
                        sugs = search_locations(dest_query.strip(), limit=4)
                        if sugs:
                            chosen_d = st.selectbox(
                                "Matches",
                                options=sugs,
                                format_func=lambda s: f"🎯 {s.name} ({s.source.title()})",
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
                if is_realworld:
                    graph = load_cached_osm_network()
                    start_lat, start_lon = st.session_state.start_coords
                    dest_lat, dest_lon = st.session_state.dest_coords
                    try:
                        start_node, _ = graph.find_nearest_node(start_lat, start_lon)
                        dest_node, _ = graph.find_nearest_node(dest_lat, dest_lon)
                        with st.spinner("Finding optimal route..."):
                            st.session_state.route_result = find_shortest_path_astar(graph, start_node, dest_node)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error mapping coordinates: {e}")
                else:
                    fict_graph = load_fictional_network()
                    st.session_state.route_result = find_shortest_path_astar(fict_graph, st.session_state.start_name, st.session_state.dest_name)
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
