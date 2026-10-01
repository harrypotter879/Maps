"""
Interactive Map Visualization module for PathFinder (Stage 2, Stage 3 & Stage 4).

Uses Folium to render interactive HTML maps with:
- Start and destination markers (custom icons and tooltips)
- High-visibility polyline tracing exact road geometries
- Dual-route rendering when comparing Dijkstra and A*
- Floating trip metrics and algorithm benchmark HUD
- Fit-to-bounds auto-framing
- Map builders for Streamlit integration
"""

from __future__ import annotations
import os
from typing import Any, List, Optional, Tuple

from graph import Graph
from dijkstra import PathResult


def _extract_polyline_points(graph: Graph, path: List[str]) -> List[List[float]]:
    """Helper to extract exact coordinates along a path sequence using road geometries."""
    points: List[List[float]] = []
    if len(path) < 2:
        if path:
            c = graph.get_node_coords(path[0])
            if c:
                points.append([c[0], c[1]])
        return points

    for u, v in zip(path[:-1], path[1:]):
        edge = graph.get_edge(u, v)
        if edge and edge.geometry:
            for pt in edge.geometry:
                points.append([pt[0], pt[1]])
        else:
            u_coord = graph.get_node_coords(u)
            v_coord = graph.get_node_coords(v)
            if u_coord:
                points.append([u_coord[0], u_coord[1]])
            if v_coord:
                points.append([v_coord[0], v_coord[1]])
    return points


def build_empty_map(
    center: Tuple[float, float] = (23.3699, 85.3253),
    zoom: int = 14,
    start_point: Optional[Tuple[float, float, str]] = None,
    dest_point: Optional[Tuple[float, float, str]] = None,
) -> Any:
    """
    Construct an initial interactive Folium map centered on Ranchi with optional start/dest markers.
    """
    import folium

    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if start_point:
        lat, lon, label = start_point
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Start:</b> {label}",
            tooltip=f"Start: {label}",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

    if dest_point:
        lat, lon, label = dest_point
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Destination:</b> {label}",
            tooltip=f"Destination: {label}",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    return m


def build_folium_map(
    graph: Graph,
    result: PathResult,
    start_coord: Tuple[float, float],
    dest_coord: Tuple[float, float],
    start_label: str = "Origin",
    dest_label: str = "Destination",
    comparison_result: Optional[PathResult] = None,
) -> Any:
    """
    Build and return a Folium Map instance for route visualization.
    """
    import folium

    mid_lat = (start_coord[0] + dest_coord[0]) / 2.0
    mid_lon = (start_coord[1] + dest_coord[1]) / 2.0

    m = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    all_bounds_points: List[List[float]] = [
        [start_coord[0], start_coord[1]],
        [dest_coord[0], dest_coord[1]],
    ]

    # Format distance
    if result.total_distance >= 1000:
        dist_display = f"{result.total_distance / 1000:.2f} km"
    else:
        dist_display = f"{result.total_distance:.0f} meters"

    time_display = f"{result.execution_time_sec * 1000:.2f} ms"

    primary_points = _extract_polyline_points(graph, result.path)
    all_bounds_points.extend(primary_points)

    if comparison_result is not None and comparison_result.found:
        # Comparison mode with both algorithms
        comp_points = _extract_polyline_points(graph, comparison_result.path)
        all_bounds_points.extend(comp_points)

        paths_match = result.path == comparison_result.path

        if paths_match:
            folium.PolyLine(
                locations=primary_points,
                color="#2563EB",  # Royal Blue
                weight=6,
                opacity=0.85,
                popup=f"Optimal Route: {start_label} ➔ {dest_label}<br>Distance: {dist_display}<br>Identical for Dijkstra and A*",
                tooltip=f"Optimal Route ({dist_display}) — Dijkstra & A* Agree",
            ).add_to(m)
        else:
            folium.PolyLine(
                locations=comp_points,
                color="#4F46E5",  # Indigo for Dijkstra
                weight=7,
                opacity=0.65,
                popup=f"Dijkstra Route: {dist_display}",
                tooltip="Dijkstra Route",
            ).add_to(m)
            folium.PolyLine(
                locations=primary_points,
                color="#059669",  # Emerald for A*
                weight=4,
                opacity=0.9,
                dash_array="6",
                popup=f"A* Route: {dist_display}",
                tooltip="A* Route",
            ).add_to(m)

        d_res = comparison_result if comparison_result.algorithm.lower() == "dijkstra" else result
        a_res = result if result.algorithm.lower() == "a*" else comparison_result

        d_time_str = f"{d_res.execution_time_sec * 1000:.2f} ms"
        a_time_str = f"{a_res.execution_time_sec * 1000:.2f} ms"
        d_nodes = d_res.visited_nodes_count
        a_nodes = a_res.visited_nodes_count

        if d_nodes > 0:
            reduction_pct = ((d_nodes - a_nodes) / d_nodes) * 100.0
            reduct_str = f"⚡ A* explored {reduction_pct:.1f}% fewer nodes!"
        else:
            reduct_str = "Both algorithms completed successfully"

        hud_html = f"""
        <div style="
            position: fixed;
            top: 15px;
            right: 15px;
            z-index: 1000;
            background-color: white;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            max-width: 340px;
            border-left: 5px solid #059669;
        ">
            <div style="font-size: 15px; font-weight: bold; color: #1E293B; margin-bottom: 6px;">
                🔬 PathFinder Algorithm Comparison
            </div>
            <div style="color: #475569; margin-bottom: 8px;">
                <b>From:</b> {start_label}<br>
                <b>To:</b> {dest_label}
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 8px 0;">
            <div style="margin-bottom: 6px;">
                📏 <b>Distance:</b> {dist_display} (Exact Match)
            </div>
            <table style="width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 4px;">
                <tr style="border-bottom: 1px solid #E2E8F0; color: #64748B;">
                    <th style="text-align: left; padding: 3px 0;">Algorithm</th>
                    <th style="text-align: right; padding: 3px 0;">Time</th>
                    <th style="text-align: right; padding: 3px 0;">Nodes</th>
                </tr>
                <tr>
                    <td style="color: #4F46E5; font-weight: 600; padding: 3px 0;">Dijkstra</td>
                    <td style="text-align: right;">{d_time_str}</td>
                    <td style="text-align: right;">{d_nodes:,}</td>
                </tr>
                <tr>
                    <td style="color: #059669; font-weight: 600; padding: 3px 0;">A* (A-Star)</td>
                    <td style="text-align: right;">{a_time_str}</td>
                    <td style="text-align: right;">{a_nodes:,}</td>
                </tr>
            </table>
            <div style="margin-top: 8px; font-size: 11px; color: #059669; font-weight: 600;">
                {reduct_str}
            </div>
        </div>
        """
    else:
        if primary_points:
            folium.PolyLine(
                locations=primary_points,
                color="#2563EB",  # Royal blue
                weight=6,
                opacity=0.85,
                popup=f"Route: {start_label} ➔ {dest_label}<br>Distance: {dist_display}<br>Algorithm: {result.algorithm}",
                tooltip=f"Shortest Route ({dist_display}) [{result.algorithm}]",
            ).add_to(m)

        hud_html = f"""
        <div style="
            position: fixed;
            top: 15px;
            right: 15px;
            z-index: 1000;
            background-color: white;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            max-width: 320px;
            border-left: 5px solid #2563EB;
        ">
            <div style="font-size: 15px; font-weight: bold; color: #1E293B; margin-bottom: 6px;">
                🧭 PathFinder Route Navigator
            </div>
            <div style="color: #475569; margin-bottom: 8px;">
                <b>From:</b> {start_label}<br>
                <b>To:</b> {dest_label}
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 8px 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px;">
                <div>📏 <b>Distance:</b> {dist_display}</div>
                <div>⚡ <b>Time:</b> {time_display}</div>
                <div>🛣️ <b>Segments:</b> {len(result.legs)}</div>
                <div>🔍 <b>Nodes:</b> {result.visited_nodes_count}</div>
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #64748B; font-style: italic;">
                Algorithm: {result.algorithm} (from scratch)
            </div>
        </div>
        """

    m.get_root().html.add_child(folium.Element(hud_html))

    # Connect off-road start/destination points
    if result.found and result.path:
        first_node_coord = graph.get_node_coords(result.path[0])
        if first_node_coord and (first_node_coord[0] != start_coord[0] or first_node_coord[1] != start_coord[1]):
            folium.PolyLine(
                locations=[[start_coord[0], start_coord[1]], [first_node_coord[0], first_node_coord[1]]],
                color="#10B981",
                weight=3,
                dash_array="6",
                tooltip="Access to road network",
            ).add_to(m)

        last_node_coord = graph.get_node_coords(result.path[-1])
        if last_node_coord and (last_node_coord[0] != dest_coord[0] or last_node_coord[1] != dest_coord[1]):
            folium.PolyLine(
                locations=[[last_node_coord[0], last_node_coord[1]], [dest_coord[0], dest_coord[1]]],
                color="#EF4444",
                weight=3,
                dash_array="6",
                tooltip="Final arrival step",
            ).add_to(m)

    # Origin Marker (Green)
    start_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 140px;">
        <h4 style="margin: 0 0 5px 0; color: #059669;">🚩 Start</h4>
        <b>{start_label}</b><br>
        <span style="font-size: 11px; color: #666;">Lat: {start_coord[0]:.5f}, Lon: {start_coord[1]:.5f}</span>
    </div>
    """
    folium.Marker(
        location=[start_coord[0], start_coord[1]],
        popup=folium.Popup(start_html, max_width=250),
        tooltip=f"Start: {start_label}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Destination Marker (Red)
    dest_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 140px;">
        <h4 style="margin: 0 0 5px 0; color: #DC2626;">🏁 Destination</h4>
        <b>{dest_label}</b><br>
        <span style="font-size: 11px; color: #666;">Lat: {dest_coord[0]:.5f}, Lon: {dest_coord[1]:.5f}</span>
    </div>
    """
    folium.Marker(
        location=[dest_coord[0], dest_coord[1]],
        popup=folium.Popup(dest_html, max_width=250),
        tooltip=f"Destination: {dest_label}",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    m.fit_bounds(all_bounds_points, padding=[30, 30])
    return m


def generate_interactive_map(
    graph: Graph,
    result: PathResult,
    start_coord: Tuple[float, float],
    dest_coord: Tuple[float, float],
    start_label: str = "Origin",
    dest_label: str = "Destination",
    output_path: str = "route_map.html",
    comparison_result: Optional[PathResult] = None,
) -> str:
    """
    Generate and save an interactive Folium HTML map visualizing calculated route(s).
    """
    m = build_folium_map(
        graph=graph,
        result=result,
        start_coord=start_coord,
        dest_coord=dest_coord,
        start_label=start_label,
        dest_label=dest_label,
        comparison_result=comparison_result,
    )
    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    m.save(abs_output)
    return abs_output
