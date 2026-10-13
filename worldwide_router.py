"""
Worldwide routing module for PathFinder.

Provides real-time road navigation, step-by-step turn guidance, and route geometries
across the entire world using OpenStreetMap's Open Source Routing Machine (OSRM).
"""

from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from graph import Graph
from dijkstra import PathResult, RouteLeg


# Famous worldwide landmarks across multiple continents for quick testing & presets
WORLDWIDE_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Europe
    "Eiffel Tower, Paris": (48.8584, 2.2945),
    "Louvre Museum, Paris": (48.8606, 2.3376),
    "Big Ben, London": (51.5007, -0.1246),
    "Tower Bridge, London": (51.5055, -0.0754),
    "Colosseum, Rome": (41.8902, 12.4922),
    "Vatican City, Rome": (41.9029, 12.4534),
    # North America
    "Times Square, New York": (40.7580, -73.9855),
    "Central Park, New York": (40.7829, -73.9654),
    "Golden Gate Bridge, San Francisco": (37.8199, -122.4783),
    # Asia
    "India Gate, New Delhi": (28.6129, 77.2295),
    "Red Fort, New Delhi": (28.6562, 77.2410),
    "Marine Drive, Mumbai": (18.9432, 72.8231),
    "Gateway of India, Mumbai": (18.9220, 72.8347),
    "Tokyo Tower, Tokyo": (35.6586, 139.7454),
    "Shibuya Crossing, Tokyo": (35.6595, 139.7005),
    "Burj Khalifa, Dubai": (25.1972, 55.2744),
    "Dubai Marina, Dubai": (25.0805, 55.1403),
    "Marina Bay Sands, Singapore": (1.2834, 103.8607),
    # Australia
    "Sydney Opera House, Sydney": (-33.8568, 151.2153),
    "Bondi Beach, Sydney": (-33.8915, 151.2767),
}

# Bounding box for the local offline Ranchi road network
RANCHI_BOUNDS = {
    "min_lat": 23.24,
    "max_lat": 23.50,
    "min_lon": 85.18,
    "max_lon": 85.47,
}

# Routing traffic estimates are intentionally limited to India. This broad
# geographic gate covers the country and its commonly routed coastal/island
# areas without treating overseas OSRM routes as India traffic routes.
INDIA_BOUNDS = {
    "min_lat": 6.0,
    "max_lat": 37.5,
    "min_lon": 68.0,
    "max_lon": 98.0,
}


def is_in_ranchi(lat: float, lon: float) -> bool:
    """Return True if (lat, lon) coordinates fall within the Ranchi local network boundary."""
    return (
        RANCHI_BOUNDS["min_lat"] <= lat <= RANCHI_BOUNDS["max_lat"]
        and RANCHI_BOUNDS["min_lon"] <= lon <= RANCHI_BOUNDS["max_lon"]
    )


def is_in_india(lat: float, lon: float) -> bool:
    """Return True when coordinates fall within India's routing area."""
    return (
        INDIA_BOUNDS["min_lat"] <= lat <= INDIA_BOUNDS["max_lat"]
        and INDIA_BOUNDS["min_lon"] <= lon <= INDIA_BOUNDS["max_lon"]
    )


def fetch_worldwide_route(
    start_coord: Tuple[float, float],
    dest_coord: Tuple[float, float],
    start_label: str = "Origin",
    dest_label: str = "Destination",
    timeout_sec: float = 8.0,
) -> Tuple[Optional[Graph], List[PathResult]]:
    """
    Fetch road navigation routes worldwide using the OpenStreetMap OSRM routing engine.

    Returns:
        Tuple of (Graph representation with geometries, List[PathResult] including alternatives)
    """
    start_lat, start_lon = start_coord
    dest_lat, dest_lon = dest_coord

    # OSRM expects coordinates in {lon},{lat} format
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start_lon:.6f},{start_lat:.6f};{dest_lon:.6f},{dest_lat:.6f}"
        f"?overview=full&geometries=geojson&steps=true&alternatives=true"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PathFinder-Worldwide-Navigation/5.0 (https://pathfinder.local)"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error querying worldwide routing engine: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error querying worldwide routing engine: {e}")

    exec_time = time.time() - t0

    code = data.get("code")
    if code != "Ok" or not data.get("routes"):
        # No route found (e.g., locations separated by ocean with no ferry/bridge)
        msg = data.get("message", "No drivable route found between the specified locations.")
        failed_res = PathResult(
            source=start_label,
            destination=dest_label,
            path=[],
            legs=[],
            total_distance=0.0,
            execution_time_sec=exec_time,
            visited_nodes_count=0,
            found=False,
            algorithm="Worldwide Navigation (OSRM)",
        )
        return None, [failed_res]

    graph = Graph()
    all_results: List[PathResult] = []

    for r_idx, route_data in enumerate(data["routes"]):
        legs_data = route_data.get("legs", [{}])[0]
        steps = legs_data.get("steps", [])
        overall_geom = route_data.get("geometry", {}).get("coordinates", [])

        route_path: List[str] = []
        route_legs: List[RouteLeg] = []

        prefix = f"r{r_idx}"

        for i, step in enumerate(steps):
            node_id = f"{prefix}_step_{i}"
            mlon, mlat = step["maneuver"]["location"]
            graph.add_node(node_id, lat=mlat, lon=mlon)
            route_path.append(node_id)

            if i > 0:
                prev_id = f"{prefix}_step_{i-1}"
                prev_step = steps[i - 1]
                dist = float(prev_step.get("distance", 0.0))
                raw_name = prev_step.get("name", "").strip()
                ref_name = prev_step.get("ref", "").strip()
                road_name = raw_name or ref_name or "Local Road"

                # Step geometry is in [lon, lat], convert to (lat, lon)
                step_coords = [
                    (pt[1], pt[0])
                    for pt in prev_step.get("geometry", {}).get("coordinates", [])
                ]
                graph.add_edge(
                    prev_id,
                    node_id,
                    weight=dist,
                    road_name=road_name,
                    geometry=step_coords,
                    bidirectional=False,
                )
                route_legs.append(
                    RouteLeg(
                        origin=prev_id,
                        destination=node_id,
                        distance=dist,
                        road_name=road_name,
                    )
                )

        # Append final destination node
        dest_node_id = f"{prefix}_step_{len(steps)}"
        if overall_geom:
            dlon, dlat = overall_geom[-1]
        else:
            dlat, dlon = dest_lat, dest_lon

        graph.add_node(dest_node_id, lat=dlat, lon=dlon)

        if steps:
            last_step = steps[-1]
            last_dist = float(last_step.get("distance", 0.0))
            last_name = last_step.get("name", "").strip() or "Destination Arrival"
            last_coords = [
                (pt[1], pt[0])
                for pt in last_step.get("geometry", {}).get("coordinates", [])
            ]
            graph.add_edge(
                route_path[-1],
                dest_node_id,
                weight=last_dist,
                road_name=last_name,
                geometry=last_coords,
                bidirectional=False,
            )
            route_legs.append(
                RouteLeg(
                    origin=route_path[-1],
                    destination=dest_node_id,
                    distance=last_dist,
                    road_name=last_name,
                )
            )
        route_path.append(dest_node_id)

        tot_dist = float(route_data.get("distance", 0.0))
        tot_dur = float(route_data.get("duration", 0.0))

        res = PathResult(
            source=start_label,
            destination=dest_label,
            path=route_path,
            legs=route_legs,
            total_distance=tot_dist,
            execution_time_sec=tot_dur,
            visited_nodes_count=len(route_path),
            found=True,
            algorithm="Worldwide Navigation (OSRM)",
        )
        all_results.append(res)

    return graph, all_results
