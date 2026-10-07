"""
Interactive Map Visualization module for PathFinder.

Uses Folium to render interactive maps with:
- Start and destination markers (custom icons and tooltips)
- High-visibility polyline tracing exact road geometries
- Interactive location search bar with instant suggestions and map pins
- Travel time estimation (driving and walking ETA)
- Clean, user-friendly trip navigation HUD
- Fit-to-bounds auto-framing
- Map builders for Streamlit integration
"""

from __future__ import annotations
import json
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


def _add_traffic_layer(
    m: Any,
    graph: Graph,
    traffic_mode: Optional[str],
    departure_time: Any,
    route_path: Optional[List[str]] = None,
    local_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Draw route traffic, or a small nearby preview before a route is selected.
    """
    import folium
    from traffic import (
        TRAFFIC_ICONS, HIGH, MEDIUM, LOW, iter_colored_traffic_segments,
    )

    segments = iter_colored_traffic_segments(
        graph, traffic_mode, departure_time, path=route_path, bounds=local_bounds,
    )
    if route_path:
        # Route mode uses a white casing and traffic-colored segments, with no
        # blue stroke covering the congestion colors.
        route_points = _extract_polyline_points(graph, route_path)
        if route_points:
            folium.PolyLine(
                locations=route_points, color="#FFFFFF", weight=12,
                opacity=1.0, interactive=False,
            ).add_to(m)
    # Group by the continuous color so hourly changes remain visible without
    # creating one Leaflet layer for every road segment.
    stroke_width = 9 if route_path else 4
    stroke_opacity = 0.95 if route_path else 0.62
    color_groups: Dict[Tuple[str, str], List[List[List[float]]]] = {}
    for color, level, points in segments:
        color_groups.setdefault((color, level), []).append(points)
    for level in (LOW, MEDIUM, HIGH):
        for (color, group_level), lines in color_groups.items():
            if group_level != level:
                continue
            scope = "on the selected route" if route_path else "nearby"
            folium.PolyLine(
                locations=lines,
                color=color,
                weight=stroke_width,
                opacity=stroke_opacity,
                tooltip=f"{TRAFFIC_ICONS[level]} {level.capitalize()} estimated traffic {scope}",
            ).add_to(m)
    legend_title = "Traffic along selected route" if route_path else "Nearby traffic estimates"
    legend = folium.Element(
        f"""<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;
        padding:10px 13px;border-radius:8px;box-shadow:0 2px 10px #0003;font:13px sans-serif;">
        <b>{legend_title}</b><br><span style="color:#16A34A">●</span> Low &nbsp;
        <span style="color:#EAB308">●</span> Moderate &nbsp;<span style="color:#DC2626">●</span> High
        <div style="margin-top:5px;color:#64748B;font-size:11px;max-width:230px">Modeled from road type and departure time; not live traffic.</div></div>"""
    )
    m.get_root().html.add_child(legend)


def _format_travel_time(distance_m: float, speed_kmh: float) -> str:
    """Calculate and format travel time based on distance and average speed."""
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


def _add_search_widget(m: Any) -> None:
    """Inject a Google Maps style floating location search bar into the Folium map."""
    import folium
    from osm_loader import RANCHI_LANDMARKS

    landmarks_list = [
        {"name": name, "lat": lat, "lon": lon, "source": "landmark"}
        for name, (lat, lon) in RANCHI_LANDMARKS.items()
    ]
    landmarks_json = json.dumps(landmarks_list)

    search_html = f"""
    <div id="pf-search-widget" style="
        position: fixed;
        top: 15px;
        left: 60px;
        z-index: 1000;
        width: 360px;
        max-width: calc(100% - 80px);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    ">
        <div id="pf-search-box" style="
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.28), 0 0 0 1px rgba(0,0,0,0.08);
            display: flex;
            align-items: center;
            padding: 4px 12px;
            height: 46px;
            box-sizing: border-box;
        ">
            <span style="font-size: 16px; color: #5f6368; margin-right: 8px; display: flex; align-items: center;">
                🔍
            </span>
            <input id="pf-search-input" type="text" placeholder="Search location or landmark..." autocomplete="off" style="
                flex: 1;
                border: none;
                outline: none;
                font-size: 14px;
                color: #202124;
                background: transparent;
            "/>
            <button id="pf-search-clear" title="Clear search" style="
                display: none;
                background: none;
                border: none;
                color: #70757a;
                cursor: pointer;
                font-size: 14px;
                padding: 4px 6px;
            ">✕</button>
        </div>
        <div id="pf-search-dropdown" style="
            display: none;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.06);
            margin-top: 6px;
            max-height: 260px;
            overflow-y: auto;
            border: 1px solid #e8eaed;
        "></div>
    </div>

    <script>
    (function() {{
        var landmarks = {landmarks_json};
        var searchWidget = document.getElementById('pf-search-widget');
        var searchInput = document.getElementById('pf-search-input');
        var searchClear = document.getElementById('pf-search-clear');
        var searchDropdown = document.getElementById('pf-search-dropdown');

        if (!searchWidget || !searchInput) return;

        ['mousedown', 'click', 'dblclick', 'touchstart', 'wheel'].forEach(function(evt) {{
            searchWidget.addEventListener(evt, function(e) {{ e.stopPropagation(); }});
        }});

        function getLeafletMap() {{
            for (var key in window) {{
                if (key.indexOf('map_') === 0 && window[key] && typeof window[key].flyTo === 'function') {{
                    return window[key];
                }}
            }}
            return null;
        }}

        var searchMarker = null;

        function selectLocation(name, lat, lon, display) {{
            var map = getLeafletMap();
            if (!map) return;

            searchInput.value = name;
            searchClear.style.display = 'block';
            searchDropdown.style.display = 'none';

            map.flyTo([lat, lon], 16, {{ animate: true, duration: 1.0 }});

            if (searchMarker) {{
                map.removeLayer(searchMarker);
            }}

            var pinHtml = '<div style="background:#7C3AED; width:32px; height:32px; border-radius:50% 50% 50% 0; transform:rotate(-45deg); display:flex; align-items:center; justify-content:center; box-shadow:0 3px 8px rgba(0,0,0,0.35); border:2px solid #ffffff;"><span style="transform:rotate(45deg); font-size:14px; color:white;">📍</span></div>';
            var customPin = L.divIcon({{
                className: 'custom-search-pin',
                html: pinHtml,
                iconSize: [32, 32],
                iconAnchor: [16, 32],
                popupAnchor: [0, -32]
            }});

            searchMarker = L.marker([lat, lon], {{ icon: customPin }}).addTo(map);
            var popupContent = '<div style="min-width:180px; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:4px;">' +
                '<h4 style="margin:0 0 4px 0; color:#7C3AED; font-size:14px;">📍 ' + name + '</h4>' +
                '<div style="font-size:12px; color:#555; margin-bottom:6px;">' + (display || (name + ', Ranchi')) + '</div>' +
                '<div style="font-size:11px; color:#888;">GPS: ' + lat.toFixed(5) + ', ' + lon.toFixed(5) + '</div>' +
                '</div>';
            searchMarker.bindPopup(popupContent).openPopup();
        }}

        function renderSuggestions(items) {{
            searchDropdown.innerHTML = '';
            if (!items || items.length === 0) {{
                searchDropdown.style.display = 'none';
                return;
            }}

            items.forEach(function(item) {{
                var row = document.createElement('div');
                row.style.padding = '8px 12px';
                row.style.cursor = 'pointer';
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.borderBottom = '1px solid #f1f3f4';
                row.style.transition = 'background 0.15s';

                row.onmouseenter = function() {{ row.style.background = '#f8f9fa'; }};
                row.onmouseleave = function() {{ row.style.background = '#ffffff'; }};

                var iconSpan = document.createElement('span');
                iconSpan.innerHTML = '📍';
                iconSpan.style.marginRight = '10px';
                iconSpan.style.fontSize = '14px';

                var textCol = document.createElement('div');
                var nameDiv = document.createElement('div');
                nameDiv.style.fontWeight = '600';
                nameDiv.style.fontSize = '13px';
                nameDiv.style.color = '#202124';
                nameDiv.innerText = item.name;

                var descDiv = document.createElement('div');
                descDiv.style.fontSize = '11px';
                descDiv.style.color = '#5f6368';
                descDiv.innerText = item.display || (item.name + ' (Landmark)');

                textCol.appendChild(nameDiv);
                textCol.appendChild(descDiv);
                row.appendChild(iconSpan);
                row.appendChild(textCol);

                row.onclick = function() {{
                    selectLocation(item.name, item.lat, item.lon, item.display);
                }};

                searchDropdown.appendChild(row);
            }});

            searchDropdown.style.display = 'block';
        }}

        var debounceTimer = null;
        searchInput.addEventListener('input', function() {{
            var q = searchInput.value.trim().toLowerCase();
            if (!q) {{
                searchClear.style.display = 'none';
                searchDropdown.style.display = 'none';
                return;
            }}
            searchClear.style.display = 'block';

            var matches = [];
            landmarks.forEach(function(lm) {{
                if (lm.name.toLowerCase().indexOf(q) !== -1) {{
                    matches.push({{
                        name: lm.name,
                        lat: lm.lat,
                        lon: lm.lon,
                        display: lm.name + ', Ranchi'
                    }});
                }}
            }});

            renderSuggestions(matches);

            clearTimeout(debounceTimer);
            if (q.length >= 3) {{
                debounceTimer = setTimeout(function() {{
                    var url = 'https://nominatim.openstreetmap.org/search?format=json&limit=5&q=' + encodeURIComponent(q + ', Ranchi');
                    fetch(url)
                        .then(function(res) {{ return res.json(); }})
                        .then(function(data) {{
                            var extra = [];
                            data.forEach(function(item) {{
                                var lat = parseFloat(item.lat);
                                var lon = parseFloat(item.lon);
                                var name = item.name || item.display_name.split(',')[0].trim();
                                if (!extra.some(function(ex) {{ return Math.abs(ex.lat - lat) < 0.0008 && Math.abs(ex.lon - lon) < 0.0008; }})) {{
                                    extra.push({{
                                        name: name,
                                        lat: lat,
                                        lon: lon,
                                        display: item.display_name + ' (OSM Nominatim)'
                                    }});
                                }}
                            }});
                            if (extra.length > 0) {{
                                var remaining = matches.filter(function(m) {{
                                    return !extra.some(function(ex) {{
                                        return (Math.abs(ex.lat - m.lat) < 0.005 && Math.abs(ex.lon - m.lon) < 0.005) ||
                                               (ex.name.toLowerCase() === m.name.toLowerCase());
                                    }});
                                }});
                                renderSuggestions(extra.concat(remaining).slice(0, 8));
                            }}
                        }})
                        .catch(function(err) {{}});
                }}, 300);
            }}
        }});

        searchClear.addEventListener('click', function() {{
            searchInput.value = '';
            searchClear.style.display = 'none';
            searchDropdown.style.display = 'none';
            if (searchMarker) {{
                var map = getLeafletMap();
                if (map) map.removeLayer(searchMarker);
                searchMarker = null;
            }}
        }});

        searchInput.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') {{
                var first = searchDropdown.querySelector('div');
                if (first) {{
                    first.click();
                }}
            }}
        }});

        document.addEventListener('click', function(e) {{
            if (!searchWidget.contains(e.target)) {{
                searchDropdown.style.display = 'none';
            }}
        }});
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(search_html))


def _add_current_location_marker(m: Any, current_location: Tuple[float, float, str]) -> None:
    """Render a glowing blue dot marker representing the user's current location."""
    import folium

    c_lat, c_lon, c_label = current_location
    blue_dot_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-width: 150px;">
        <h4 style="margin: 0 0 4px 0; color: #2563EB; font-size: 14px;">🔵 My Current Location</h4>
        <b>{c_label}</b><br>
        <span style="font-size: 11px; color: #666;">Lat: {c_lat:.5f}, Lon: {c_lon:.5f}</span>
    </div>
    """

    dot_div_icon = folium.DivIcon(
        html=f'''
        <div style="
            width: 22px;
            height: 22px;
            background-color: #2563EB;
            border-radius: 50%;
            border: 3px solid #FFFFFF;
            box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.38), 0 3px 8px rgba(0,0,0,0.35);
            transform: translate(-11px, -11px);
        "></div>
        ''',
        icon_size=(22, 22),
        icon_anchor=(11, 11),
    )

    folium.Marker(
        location=[c_lat, c_lon],
        popup=folium.Popup(blue_dot_html, max_width=250),
        tooltip=f"🔵 My Location: {c_label}",
        icon=dot_div_icon,
    ).add_to(m)


def _add_locate_control(m: Any) -> None:
    """Add Leaflet LocateControl button to map for auto-locating position."""
    try:
        from folium.plugins import LocateControl
        LocateControl(
            auto_start=False,
            flyTo=True,
            keepCurrentZoomLevel=False,
            strings={"title": "Show my location", "popup": "You are within {distance} {unit} from this point"},
        ).add_to(m)
    except Exception:
        pass


def build_empty_map(
    center: Tuple[float, float] = (23.3699, 85.3253),
    zoom: int = 14,
    start_point: Optional[Tuple[float, float, str]] = None,
    dest_point: Optional[Tuple[float, float, str]] = None,
    searched_point: Optional[Tuple[float, float, str]] = None,
    current_location: Optional[Tuple[float, float, str]] = None,
    include_search_bar: bool = False,
    graph: Optional[Graph] = None,
    traffic_mode: Optional[str] = None,
    departure_time: Any = None,
) -> Any:
    """
    Construct an initial interactive Folium map centered on Ranchi with optional markers.
    Traffic coloring is deferred until a route is selected; an empty map has no route segments to color.
    """
    import folium

    map_center = center
    map_zoom = zoom

    if current_location and not start_point and not dest_point and not searched_point:
        map_center = (current_location[0], current_location[1])
        map_zoom = 15
    elif searched_point and not start_point and not dest_point:
        map_center = (searched_point[0], searched_point[1])
        map_zoom = 15

    m = folium.Map(
        location=[map_center[0], map_center[1]],
        zoom_start=map_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if graph is not None and traffic_mode:
        # Before a route exists, show traffic only around the current map center.
        lat_margin, lon_margin = 0.009, 0.012
        _add_traffic_layer(
            m, graph, traffic_mode, departure_time,
            local_bounds=(map_center[0] - lat_margin, map_center[1] - lon_margin,
                          map_center[0] + lat_margin, map_center[1] + lon_margin),
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

    if searched_point:
        s_lat, s_lon, s_label = searched_point
        folium.Marker(
            location=[s_lat, s_lon],
            popup=folium.Popup(
                f"<div style='font-family: Arial, sans-serif; min-width: 150px;'>"
                f"<h4 style='margin: 0 0 4px 0; color: #7C3AED;'>🔍 Searched Place</h4>"
                f"<b>{s_label}</b><br>"
                f"<span style='font-size: 11px; color: #666;'>Lat: {s_lat:.5f}, Lon: {s_lon:.5f}</span>"
                f"</div>",
                max_width=250,
            ),
            tooltip=f"🔍 Searched: {s_label}",
            icon=folium.Icon(color="purple", icon="search", prefix="fa"),
        ).add_to(m)

    if current_location:
        _add_current_location_marker(m, current_location)

    _add_locate_control(m)

    if include_search_bar:
        _add_search_widget(m)

    return m


def build_folium_map(
    graph: Graph,
    result: PathResult,
    start_coord: Tuple[float, float],
    dest_coord: Tuple[float, float],
    start_label: str = "Origin",
    dest_label: str = "Destination",
    comparison_result: Optional[PathResult] = None,
    alternative_results: Optional[List[PathResult]] = None,
    active_route_index: int = 0,
    searched_point: Optional[Tuple[float, float, str]] = None,
    current_location: Optional[Tuple[float, float, str]] = None,
    show_algorithm_stats: bool = False,
    include_search_bar: bool = False,
    show_hud: bool = True,
    traffic_mode: Optional[str] = None,
    departure_time: Any = None,
) -> Any:
    """
    Build and return a Folium Map instance for route visualization.
    If `traffic_mode` is set, roads around the route are coloured by modeled traffic.
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

    # Format distance and estimated travel times
    if result.total_distance >= 1000:
        dist_display = f"{result.total_distance / 1000:.2f} km"
    else:
        dist_display = f"{result.total_distance:.0f} meters"

    est_drive = _format_travel_time(result.total_distance, speed_kmh=30.0)
    est_walk = _format_travel_time(result.total_distance, speed_kmh=4.5)

    primary_points = _extract_polyline_points(graph, result.path)
    all_bounds_points.extend(primary_points)

    if traffic_mode:
        # Route mode replaces the nearby overlay with only the roads traversed
        # by the selected route; its traffic colors are the route trace.
        _add_traffic_layer(
            m, graph, traffic_mode, departure_time, result.path,
        )

    if comparison_result is not None and comparison_result.found:
        # Comparison mode with both algorithms (retained for CLI benchmarking)
        comp_points = _extract_polyline_points(graph, comparison_result.path)
        all_bounds_points.extend(comp_points)

        paths_match = result.path == comparison_result.path

        if paths_match:
            folium.PolyLine(
                locations=primary_points,
                color="#2563EB",  # Royal Blue
                weight=6,
                opacity=0.85,
                popup=f"Optimal Route: {start_label} ➔ {dest_label}<br>Distance: {dist_display}",
                tooltip=f"Optimal Route ({dist_display})",
            ).add_to(m)
        else:
            folium.PolyLine(
                locations=comp_points,
                color="#4F46E5",
                weight=7,
                opacity=0.65,
                popup=f"Route A: {dist_display}",
                tooltip="Route A",
            ).add_to(m)
            folium.PolyLine(
                locations=primary_points,
                color="#059669",
                weight=4,
                opacity=0.9,
                dash_array="6",
                popup=f"Route B: {dist_display}",
                tooltip="Route B",
            ).add_to(m)

        d_res = comparison_result if comparison_result.algorithm.lower() == "dijkstra" else result
        a_res = result if result.algorithm.lower() == "a*" else comparison_result

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
                    <th style="text-align: right; padding: 3px 0;">Route</th>
                </tr>
                <tr>
                    <td style="color: #4F46E5; font-weight: 600; padding: 3px 0;">Dijkstra</td>
                    <td style="text-align: right;">{len(d_res.legs)} legs</td>
                </tr>
                <tr>
                    <td style="color: #059669; font-weight: 600; padding: 3px 0;">A* (A-Star)</td>
                    <td style="text-align: right;">{len(a_res.legs)} legs</td>
                </tr>
            </table>
        </div>
        """
    else:
        # Standard clean user-friendly navigation HUD
        if alternative_results and len(alternative_results) > 1:
            for i, alt_res in enumerate(alternative_results):
                if i == active_route_index or traffic_mode:
                    continue
                alt_points = _extract_polyline_points(graph, alt_res.path)
                all_bounds_points.extend(alt_points)
                alt_dist = f"{alt_res.total_distance / 1000:.2f} km" if alt_res.total_distance >= 1000 else f"{alt_res.total_distance:.0f} meters"
                folium.PolyLine(
                    locations=alt_points,
                    color="#93C5FD",  # Light blue
                    weight=5,
                    opacity=0.7,
                    popup=f"Alternative Route {i+1}<br>Distance: {alt_dist}",
                    tooltip=f"Alternative Route {i+1} ({alt_dist})",
                ).add_to(m)

            if primary_points and not traffic_mode:
                folium.PolyLine(
                    locations=primary_points,
                    color="#2563EB",  # Royal blue
                    weight=6,
                    opacity=0.9,
                    popup=f"Selected Route: {start_label} ➔ {dest_label}<br>Distance: {dist_display}<br>Est. Drive: ~{est_drive}",
                    tooltip=f"Selected Route ({dist_display})",
                ).add_to(m)
        else:
            if primary_points and not traffic_mode:
                folium.PolyLine(
                    locations=primary_points,
                    color="#2563EB",  # Royal blue
                    weight=6,
                    opacity=0.85,
                    popup=f"Route: {start_label} ➔ {dest_label}<br>Distance: {dist_display}<br>Est. Drive: ~{est_drive}",
                    tooltip=f"Shortest Route ({dist_display})",
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
                <div>🚗 <b>Drive Time:</b> ~{est_drive}</div>
                <div>🚶 <b>Walk Time:</b> ~{est_walk}</div>
                <div>🛣️ <b>Segments:</b> {len(result.legs)}</div>
            </div>
        </div>
        """

    if show_hud:
        m.get_root().html.add_child(folium.Element(hud_html))

    # Connect off-road start/destination points (only if within reasonable 500m access walk)
    if result.found and result.path:
        from astar import haversine_distance

        first_node_coord = graph.get_node_coords(result.path[0])
        if first_node_coord:
            start_offroad_m = haversine_distance(start_coord, first_node_coord)
            if 5.0 <= start_offroad_m <= 1000.0:
                folium.PolyLine(
                    locations=[[start_coord[0], start_coord[1]], [first_node_coord[0], first_node_coord[1]]],
                    color="#10B981",
                    weight=3,
                    dash_array="6",
                    tooltip=f"Access walk ({start_offroad_m:.0f}m)",
                ).add_to(m)

        last_node_coord = graph.get_node_coords(result.path[-1])
        if last_node_coord:
            dest_offroad_m = haversine_distance(dest_coord, last_node_coord)
            if 5.0 <= dest_offroad_m <= 1000.0:
                folium.PolyLine(
                    locations=[[last_node_coord[0], last_node_coord[1]], [dest_coord[0], dest_coord[1]]],
                    color="#EF4444",
                    weight=3,
                    dash_array="6",
                    tooltip=f"Final arrival step ({dest_offroad_m:.0f}m)",
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

    # Searched Location Marker (Purple) if present
    if searched_point:
        s_lat, s_lon, s_label = searched_point
        s_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 150px;">
            <h4 style="margin: 0 0 4px 0; color: #7C3AED;">🔍 Searched Place</h4>
            <b>{s_label}</b><br>
            <span style="font-size: 11px; color: #666;">Lat: {s_lat:.5f}, Lon: {s_lon:.5f}</span>
        </div>
        """
        folium.Marker(
            location=[s_lat, s_lon],
            popup=folium.Popup(s_html, max_width=250),
            tooltip=f"🔍 Searched: {s_label}",
            icon=folium.Icon(color="purple", icon="search", prefix="fa"),
        ).add_to(m)
        all_bounds_points.append([s_lat, s_lon])

    if current_location:
        _add_current_location_marker(m, current_location)
        all_bounds_points.append([current_location[0], current_location[1]])

    _add_locate_control(m)

    if include_search_bar:
        _add_search_widget(m)

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
    alternative_results: Optional[List[PathResult]] = None,
    active_route_index: int = 0,
    searched_point: Optional[Tuple[float, float, str]] = None,
    current_location: Optional[Tuple[float, float, str]] = None,
    show_algorithm_stats: bool = False,
    include_search_bar: bool = True,
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
        alternative_results=alternative_results,
        active_route_index=active_route_index,
        searched_point=searched_point,
        current_location=current_location,
        show_algorithm_stats=show_algorithm_stats,
        include_search_bar=include_search_bar,
    )
    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    m.save(abs_output)
    return abs_output
