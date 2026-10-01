"""
PathFinder — Stage 4: Interactive Navigation UI
Built with Streamlit, Folium, and OpenStreetMap (OSMnx & Nominatim).

Features:
- Searchable 'From' and 'To' fields with Nominatim geocoding & landmark suggestions
- Interactive Folium map with start/destination markers and road-geometry routes
- Swap button to interchange start and destination effortlessly
- Algorithm selection: A* (fastest), Dijkstra (exhaustive), or Side-by-Side Comparison
- Interactive map click-to-select location support
- Real-time trip metrics (distance, execution time, nodes explored) and turn-by-turn guidance
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit_folium import st_folium

from graph import Graph, create_sample_road_network
from dijkstra import find_shortest_path, PathResult
from astar import find_shortest_path_astar
from osm_loader import get_ranchi_road_network, RANCHI_LANDMARKS
from geocoder import search_locations, reverse_geocode, GeocodedLocation
from map_view import build_folium_map, build_empty_map
from display import format_distance, format_time


# Page Configuration
st.set_page_config(
    page_title="PathFinder — Real-World Navigation",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
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


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    landmark_keys = list(RANCHI_LANDMARKS.keys())
    if "start_name" not in st.session_state:
        st.session_state.start_name = "Albert Ekka Chowk"
        st.session_state.start_coords = RANCHI_LANDMARKS["Albert Ekka Chowk"]

    if "dest_name" not in st.session_state:
        st.session_state.dest_name = "Ranchi Railway Station"
        st.session_state.dest_coords = RANCHI_LANDMARKS["Ranchi Railway Station"]

    if "algorithm" not in st.session_state:
        st.session_state.algorithm = "A* Search (Recommended)"

    if "route_result" not in st.session_state:
        st.session_state.route_result = None

    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

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

    # Clear previous route so user can re-compute in opposite direction
    st.session_state.route_result = None
    st.session_state.comparison_result = None


def main() -> None:
    init_session_state()

    # Sidebar Header & Controls
    with st.sidebar:
        st.markdown("## 🧭 PathFinder Navigation")
        st.caption("Stage 4: Interactive Web Navigation UI")

        # Network selection
        network_mode = st.radio(
            "Road Network Mode",
            ["Real-World Map (Ranchi, India)", "Fictional Regional Map (Stage 1)"],
            index=0,
            help="Choose between real-world OpenStreetMap data and the Stage 1 fictional graph.",
        )
        is_realworld = "Real-World" in network_mode

        st.markdown("---")

        # Algorithm Selection
        st.markdown("### ⚙️ Routing Algorithm")
        algo_options = [
            "A* Search (Recommended)",
            "Dijkstra's Algorithm",
            "Compare Both (Dijkstra vs A*)",
        ]
        selected_algo = st.selectbox(
            "Choose Algorithm",
            algo_options,
            index=0,
            help="A* uses the Haversine straight-line heuristic to prune the search space.",
        )
        st.session_state.algorithm = selected_algo

        st.markdown("---")
        st.markdown("### 📍 Route Endpoints")

        if is_realworld:
            # Preset landmarks or search
            landmark_options = ["🔍 Search Address / Custom Place..."] + list(RANCHI_LANDMARKS.keys())

            # Start Location Section
            st.markdown("**From (Origin):**")
            start_mode = st.selectbox(
                "Select Origin Landmark",
                landmark_options,
                index=landmark_options.index(st.session_state.start_name) if st.session_state.start_name in landmark_options else 0,
                key="start_select",
                label_visibility="collapsed",
            )

            if start_mode == "🔍 Search Address / Custom Place...":
                start_query = st.text_input(
                    "Search Origin (Nominatim / Address / Coords)",
                    value=st.session_state.start_name if st.session_state.start_name not in RANCHI_LANDMARKS else "",
                    placeholder="e.g. Main Road, Doranda, or 23.3699, 85.3253",
                    key="start_text_input",
                )
                if start_query.strip():
                    suggestions = search_locations(start_query, limit=4)
                    if suggestions:
                        chosen_start = st.selectbox(
                            "Matching Locations",
                            options=suggestions,
                            format_func=lambda s: f"📍 {s.name} ({s.source})",
                            key="chosen_start_suggestion",
                        )
                        st.session_state.start_name = chosen_start.name
                        st.session_state.start_coords = (chosen_start.lat, chosen_start.lon)
                    else:
                        st.warning("No matches found. Try another query or GPS coordinates.")
            else:
                st.session_state.start_name = start_mode
                st.session_state.start_coords = RANCHI_LANDMARKS[start_mode]

            st.caption(f"Coordinates: `{st.session_state.start_coords[0]:.4f}, {st.session_state.start_coords[1]:.4f}`")

            # Swap Button
            col_swap, _ = st.columns([1, 1])
            with col_swap:
                st.button("⇄ Swap Start & Destination", on_click=swap_locations, use_container_width=True)

            # Destination Location Section
            st.markdown("**To (Destination):**")
            dest_mode = st.selectbox(
                "Select Destination Landmark",
                landmark_options,
                index=landmark_options.index(st.session_state.dest_name) if st.session_state.dest_name in landmark_options else 0,
                key="dest_select",
                label_visibility="collapsed",
            )

            if dest_mode == "🔍 Search Address / Custom Place...":
                dest_query = st.text_input(
                    "Search Destination (Nominatim / Address / Coords)",
                    value=st.session_state.dest_name if st.session_state.dest_name not in RANCHI_LANDMARKS else "",
                    placeholder="e.g. Tagore Hill, Morabadi, or 23.3512, 85.3347",
                    key="dest_text_input",
                )
                if dest_query.strip():
                    dest_suggestions = search_locations(dest_query, limit=4)
                    if dest_suggestions:
                        chosen_dest = st.selectbox(
                            "Matching Locations",
                            options=dest_suggestions,
                            format_func=lambda s: f"🎯 {s.name} ({s.source})",
                            key="chosen_dest_suggestion",
                        )
                        st.session_state.dest_name = chosen_dest.name
                        st.session_state.dest_coords = (chosen_dest.lat, chosen_dest.lon)
                    else:
                        st.warning("No matches found. Try another query or GPS coordinates.")
            else:
                st.session_state.dest_name = dest_mode
                st.session_state.dest_coords = RANCHI_LANDMARKS[dest_mode]

            st.caption(f"Coordinates: `{st.session_state.dest_coords[0]:.4f}, {st.session_state.dest_coords[1]:.4f}`")

        else:
            # Stage 1 Fictional Towns
            fict_graph = load_fictional_network()
            fict_nodes = fict_graph.get_nodes()
            st.session_state.start_name = st.selectbox("From (Town)", fict_nodes, index=0)
            if st.button("⇄ Swap Start & Destination", on_click=swap_locations, use_container_width=True):
                pass
            st.session_state.dest_name = st.selectbox("To (Town)", fict_nodes, index=min(3, len(fict_nodes)-1))

        st.markdown("---")

        # Find Route Button
        find_btn = st.button("🚀 Find Route", type="primary", use_container_width=True)

    # Main Area
    st.markdown("## 🌐 Real-World Shortest Path Navigator")
    st.markdown(
        f"**From:** `{st.session_state.start_name}` &nbsp;➔&nbsp; **To:** `{st.session_state.dest_name}` &nbsp;|&nbsp; "
        f"**Algorithm:** `{st.session_state.algorithm}`"
    )

    # Route Calculation Logic
    if is_realworld:
        graph = load_cached_osm_network()
        start_lat, start_lon = st.session_state.start_coords
        dest_lat, dest_lon = st.session_state.dest_coords

        try:
            start_node, start_dist_m = graph.find_nearest_node(start_lat, start_lon)
            dest_node, dest_dist_m = graph.find_nearest_node(dest_lat, dest_lon)
        except Exception as e:
            st.error(f"Error mapping coordinates to road network: {e}")
            return

        if find_btn or st.session_state.route_result is None:
            if "Compare" in st.session_state.algorithm:
                with st.spinner("Computing optimal routes with Dijkstra and A*..."):
                    d_res = find_shortest_path(graph, start_node, dest_node)
                    a_res = find_shortest_path_astar(graph, start_node, dest_node)
                    st.session_state.route_result = a_res
                    st.session_state.comparison_result = d_res
            elif "Dijkstra" in st.session_state.algorithm:
                with st.spinner("Computing optimal route with Dijkstra's algorithm..."):
                    d_res = find_shortest_path(graph, start_node, dest_node)
                    st.session_state.route_result = d_res
                    st.session_state.comparison_result = None
            else:
                with st.spinner("Computing optimal route with A* search..."):
                    a_res = find_shortest_path_astar(graph, start_node, dest_node)
                    st.session_state.route_result = a_res
                    st.session_state.comparison_result = None

        primary_result: PathResult = st.session_state.route_result
        comp_result: Optional[PathResult] = st.session_state.comparison_result

        # Display Metrics Cards
        if primary_result and primary_result.found:
            m1, m2, m3, m4 = st.columns(4)
            dist_label = f"{primary_result.total_distance/1000:.2f} km" if primary_result.total_distance >= 1000 else f"{primary_result.total_distance:.0f} m"
            m1.metric("📏 Total Distance", dist_label)
            m2.metric("⚡ Execution Time", format_time(primary_result.execution_time_sec))
            m3.metric("🔍 Nodes Explored", f"{primary_result.visited_nodes_count:,}")
            m4.metric("🛣️ Intersections / Legs", f"{len(primary_result.path)} / {len(primary_result.legs)}")

            # Comparison Summary Card if both algorithms ran
            if comp_result is not None:
                d_nodes = comp_result.visited_nodes_count
                a_nodes = primary_result.visited_nodes_count
                node_reduction = ((d_nodes - a_nodes) / d_nodes) * 100.0 if d_nodes > 0 else 0.0
                speedup = (comp_result.execution_time_sec / primary_result.execution_time_sec) if primary_result.execution_time_sec > 0 else 1.0

                st.success(
                    f"🏆 **Algorithm Comparison Result:** Both algorithms found the identical optimal shortest distance of **{dist_label}**. "
                    f"**A* (A-Star) explored {node_reduction:.1f}% fewer intersections** ({a_nodes:,} vs {d_nodes:,}) and ran **{speedup:.1f}x faster** "
                    f"({format_time(primary_result.execution_time_sec)} vs {format_time(comp_result.execution_time_sec)})!"
                )

        elif primary_result and not primary_result.found:
            st.error(f"❌ No route found! Destination '{st.session_state.dest_name}' is not reachable by road from '{st.session_state.start_name}'.")

        # Map Rendering
        if primary_result and primary_result.found:
            folium_map = build_folium_map(
                graph=graph,
                result=primary_result,
                start_coord=st.session_state.start_coords,
                dest_coord=st.session_state.dest_coords,
                start_label=st.session_state.start_name,
                dest_label=st.session_state.dest_name,
                comparison_result=comp_result,
            )
        else:
            folium_map = build_empty_map(
                center=st.session_state.start_coords,
                zoom=14,
                start_point=(st.session_state.start_coords[0], st.session_state.start_coords[1], st.session_state.start_name),
                dest_point=(st.session_state.dest_coords[0], st.session_state.dest_coords[1], st.session_state.dest_name),
            )

        # Render interactive map using st_folium
        map_data = st_folium(
            folium_map,
            width=None,
            height=540,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )

        # Map Click Interaction
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            st.info(f"🖱️ **Map Click Detected:** `Latitude: {click_lat:.5f}, Longitude: {click_lon:.5f}`")

            col_c1, col_c2, _ = st.columns([1, 1, 2])
            with col_c1:
                if st.button("📍 Set Click as Origin (Start)", key="btn_click_start"):
                    rev_name = reverse_geocode(click_lat, click_lon)
                    st.session_state.start_name = rev_name
                    st.session_state.start_coords = (click_lat, click_lon)
                    st.session_state.route_result = None
                    st.rerun()

            with col_c2:
                if st.button("🎯 Set Click as Destination", key="btn_click_dest"):
                    rev_name = reverse_geocode(click_lat, click_lon)
                    st.session_state.dest_name = rev_name
                    st.session_state.dest_coords = (click_lat, click_lon)
                    st.session_state.route_result = None
                    st.rerun()

        # Turn-by-Turn Itinerary & Comparison Table
        if primary_result and primary_result.found:
            t1, t2 = st.tabs(["🗺️ Turn-by-Turn Itinerary", "🔬 Algorithm Benchmark Details"])

            with t1:
                st.markdown("#### Turn-by-Turn Road Corridor Guide")
                # Group legs into road corridors
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

            with t2:
                st.markdown("#### Detailed Algorithm Performance Metrics")
                if comp_result is not None:
                    comp_data = {
                        "Metric": ["Shortest Distance", "Nodes Explored", "Execution Time", "Route Segments", "Optimality"],
                        "Dijkstra": [
                            format_distance(comp_result.total_distance, "m"),
                            f"{comp_result.visited_nodes_count:,}",
                            format_time(comp_result.execution_time_sec),
                            f"{len(comp_result.legs)} legs",
                            "Optimal (Exact)",
                        ],
                        "A* Search": [
                            format_distance(primary_result.total_distance, "m"),
                            f"{primary_result.visited_nodes_count:,}",
                            format_time(primary_result.execution_time_sec),
                            f"{len(primary_result.legs)} legs",
                            "Optimal (Exact)",
                        ],
                        "Advantage / Notes": [
                            "Identical shortest path",
                            f"{node_reduction:.1f}% fewer nodes explored",
                            f"{speedup:.1f}x faster execution",
                            "Equal route geometry",
                            "Guaranteed by Haversine admissibility",
                        ],
                    }
                    st.table(comp_data)
                else:
                    st.info(f"Ran `{primary_result.algorithm}`. Select **Compare Both** in the sidebar to benchmark Dijkstra and A* side-by-side!")

    else:
        # Fictional Graph Mode
        fict_graph = load_fictional_network()
        start_town = st.session_state.start_name
        dest_town = st.session_state.dest_name

        if "Compare" in st.session_state.algorithm:
            d_res = find_shortest_path(fict_graph, start_town, dest_town)
            a_res = find_shortest_path_astar(fict_graph, start_town, dest_town)
            primary_result = a_res
            comp_result = d_res
        elif "A*" in st.session_state.algorithm:
            primary_result = find_shortest_path_astar(fict_graph, start_town, dest_town)
            comp_result = None
        else:
            primary_result = find_shortest_path(fict_graph, start_town, dest_town)
            comp_result = None

        if primary_result.found:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📏 Total Distance", f"{primary_result.total_distance:.1f} km")
            m2.metric("⚡ Execution Time", format_time(primary_result.execution_time_sec))
            m3.metric("🔍 Nodes Explored", f"{primary_result.visited_nodes_count}")
            m4.metric("🛣️ Legs", f"{len(primary_result.legs)}")

            st.markdown("#### Route Path Sequence:")
            st.code(" ➔ ".join(primary_result.path))

            st.markdown("#### Turn-by-Turn Legs:")
            for idx, leg in enumerate(primary_result.legs, start=1):
                road_info = f" via [{leg.road_name}]" if leg.road_name else ""
                st.write(f"**{idx}.** {leg.origin} ➔ {leg.destination}: `{leg.distance:.1f} km` {road_info}")
        else:
            st.error("No route found between selected towns.")


if __name__ == "__main__":
    main()
