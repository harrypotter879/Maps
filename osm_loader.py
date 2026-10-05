"""
OpenStreetMap road network loader and coordinate resolver for PathFinder (Stage 2).

Provides utilities to:
- Load or download real-world OpenStreetMap road data using OSMnx.
- Convert OSM MultiDiGraph to PathFinder's weighted Graph representation.
- Cache road networks in JSON format for instant, offline startup.
- Resolve landmark names and GPS coordinates to nearest road network nodes.
"""

from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from graph import Graph


# Predefined key landmarks, transit hubs, hospitals, and localities in Ranchi, Jharkhand, India
RANCHI_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Central Chowks & Hubs
    "Albert Ekka Chowk": (23.36990, 85.32530),
    "Firayalal Chowk": (23.36990, 85.32530),
    "Sujata Chowk": (23.35365, 85.32448),
    "Main Road Overbridge": (23.35650, 85.32850),
    "Siramtoli Chowk": (23.35400, 85.33300),
    "Lalpur Chowk": (23.37215, 85.33829),
    "Kantatoli Chowk": (23.36700, 85.34750),
    "Kutchery Chowk": (23.37891, 85.32519),
    "Shaheed Chowk": (23.37120, 85.32480),
    "Hinoo Chowk": (23.32894, 85.31701),
    "Argora Chowk": (23.35017, 85.29807),
    "Birsa Chowk": (23.32355, 85.30759),
    "Booty More": (23.39540, 85.38440),
    "Kokar Chowk": (23.37721, 85.36003),

    # Transit Stations & Terminals
    "Ranchi Railway Station": (23.35120, 85.33470),
    "Hatia Railway Station": (23.31310, 85.30590),
    "Birsa Munda Airport": (23.31460, 85.32550),
    "Birsa Munda Bus Terminal (Khadgarha)": (23.36550, 85.35200),
    "ITI Bus Stand": (23.37615, 85.28322),

    # Higher Education & Schools
    "Kairali School": (23.31918, 85.29877),
    "Ranchi University": (23.37180, 85.32425),  # Main Campus (Kutchery / Line Tank)
    "Ranchi University (Morabadi)": (23.38880, 85.32283),
    "St. Xavier's College": (23.36747, 85.32596),
    "BIT Mesra": (23.41757, 85.43941),
    "IIM Ranchi": (23.38780, 85.39250),
    "Ranchi Women's College": (23.37850, 85.32850),

    # Hospitals & Healthcare
    "RIMS Hospital": (23.39060, 85.34800),
    "Sadar Hospital": (23.36909, 85.32679),
    "Paras HEC Hospital": (23.31362, 85.27736),
    "Orchid Medical Centre": (23.37125, 85.33349),

    # Shopping Malls & Commercial
    "Nucleus Mall": (23.37747, 85.33162),
    "Mall of Ranchi": (23.37840, 85.31470),
    "JD High Street Mall": (23.35370, 85.32830),
    "Spring City Mall": (23.32850, 85.32400),
    "Doranda Market": (23.33597, 85.32500),
    "Upper Bazar": (23.37828, 85.31777),

    # Prominent Landmarks, Parks & Culture
    "Morabadi Ground": (23.38800, 85.33000),
    "Tagore Hill": (23.40140, 85.33800),
    "Pahari Mandir": (23.37810, 85.31200),
    "Jagannath Temple": (23.31690, 85.28180),
    "JSCA International Stadium": (23.31050, 85.27490),
    "Rock Garden": (23.40350, 85.31270),
    "Kanke Dam": (23.39740, 85.30290),
    "Dhurwa Dam": (23.28492, 85.25849),
    "Raj Bhavan": (23.38054, 85.31821),
    "Jharkhand High Court": (23.31520, 85.26960),
    "Jharkhand State Assembly": (23.29650, 85.28350),

    # Key Neighborhoods & Colonies
    "Harmu Housing Colony": (23.36100, 85.31020),
    "Ashok Nagar": (23.34890, 85.30520),
    "Kadru": (23.34991, 85.31366),
    "Bariatu": (23.38900, 85.35300),
    "Kokar": (23.37721, 85.36003),
    "Namkum": (23.34300, 85.37850),
    "Tupudana": (23.28400, 85.31500),
    "Dhurwa": (23.31200, 85.29500),
    "Kanke": (23.41500, 85.32200),
    "Chutia": (23.34932, 85.33557),
    "Doranda": (23.33597, 85.32500),
    "Hinoo": (23.32894, 85.31701),
    "Radisson Blu Hotel": (23.34850, 85.32880),
}

DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "ranchi_network.json")


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """
    Attempt to parse latitude and longitude from a user string.
    
    Accepts formats such as:
    - '23.3699, 85.3253'
    - '23.3699,85.3253'
    - '(23.3699, 85.3253)'
    - '23.3699 85.3253'
    """
    cleaned = text.strip().strip("()[]{}")
    match = re.search(r"^([-+]?\d{1,2}(?:\.\d+)?)[,\s]+([-+]?\d{1,3}(?:\.\d+)?)$", cleaned)
    if not match:
        return None

    try:
        lat = float(match.group(1))
        lon = float(match.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return (lat, lon)
    except ValueError:
        pass

    return None


def graph_from_json(file_path: str) -> Graph:
    """
    Construct a Graph instance from a serialized JSON road network file.
    
    Raises FileNotFoundError or ValueError if file is missing or invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Road network cache not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = Graph()

    # Load nodes with coordinates
    nodes_map = data.get("nodes", {})
    for node_id, coords in nodes_map.items():
        graph.add_node(str(node_id), lat=coords.get("lat"), lon=coords.get("lon"))

    # Load edges with lengths, road names, and geometries
    edges_list = data.get("edges", [])
    for edge in edges_list:
        u = str(edge["u"])
        v = str(edge["v"])
        weight = float(edge["weight"])
        road_name = edge.get("road_name", "")
        geometry = edge.get("geometry")
        geom_points = [tuple(pt) for pt in geometry] if geometry else None
        graph.add_edge(
            u=u,
            v=v,
            weight=weight,
            bidirectional=False,  # OSM edges are directed; reverse edges are explicit
            road_name=road_name,
            geometry=geom_points,
        )

    return graph


def save_graph_to_json(
    graph: Graph,
    file_path: str,
    area_name: str = "Custom Area",
    center: Optional[Tuple[float, float]] = None,
    radius_m: int = 0,
) -> None:
    """Serialize a Graph instance and its geographic data to a JSON cache file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    nodes_data = {}
    for node in graph.get_nodes():
        coords = graph.get_node_coords(node)
        if coords:
            nodes_data[node] = {"lat": coords[0], "lon": coords[1]}
        else:
            nodes_data[node] = {}

    edges_data = []
    for u in graph.get_nodes():
        for edge in graph.get_neighbors(u):
            entry = {
                "u": u,
                "v": edge.destination,
                "weight": round(edge.weight, 2),
                "road_name": edge.road_name,
            }
            if edge.geometry:
                entry["geometry"] = [[round(lat, 6), round(lon, 6)] for lat, lon in edge.geometry]
            edges_data.append(entry)

    payload = {
        "area": area_name,
        "center": list(center) if center else None,
        "radius_m": radius_m,
        "nodes": nodes_data,
        "edges": edges_data,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def download_osm_road_network(
    center_lat: float = 23.3699,
    center_lon: float = 85.3253,
    dist_m: int = 7500,
    network_type: str = "drive",
    cache_path: Optional[str] = None,
) -> Graph:
    """
    Download real road network data from OpenStreetMap via OSMnx and convert to Graph.
    
    Args:
        center_lat: Latitude of central point (default: Albert Ekka Chowk, Ranchi).
        center_lon: Longitude of central point.
        dist_m: Radius around center in meters (default: 7500m for full Greater Ranchi).
        network_type: Type of street network ('drive', 'walk', 'bike').
        cache_path: Optional path to save downloaded data for offline use.
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError(
            "OSMnx is required to download live OpenStreetMap data. "
            "Install it via: pip install osmnx"
        ) from e

    g_ox = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type=network_type)

    graph = Graph()

    # Ingest nodes
    for n, data in g_ox.nodes(data=True):
        graph.add_node(str(n), lat=float(data["y"]), lon=float(data["x"]))

    # Ingest edges
    for u, v, data in g_ox.edges(data=True):
        length = float(data.get("length", 1.0))
        name = data.get("name", "")
        if isinstance(name, list):
            name = ", ".join(str(x) for x in name if x)
        elif not name:
            name = ""

        geometry = None
        geom = data.get("geometry", None)
        if geom is not None:
            geometry = [(round(lat, 6), round(lon, 6)) for lon, lat in geom.coords]

        graph.add_edge(
            u=str(u),
            v=str(v),
            weight=length,
            bidirectional=False,
            road_name=str(name),
            geometry=geometry,
        )

    if cache_path:
        save_graph_to_json(
            graph=graph,
            file_path=cache_path,
            area_name="Ranchi, Jharkhand, India",
            center=(center_lat, center_lon),
            radius_m=dist_m,
        )

    return graph


def get_ranchi_road_network(
    force_download: bool = False,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> Graph:
    """
    Load the Ranchi road network.
    
    Tries loading from local cache first for sub-50ms offline startup.
    Downloads from OSMnx if cache is missing or if force_download is True.
    """
    if not force_download and os.path.exists(cache_path):
        return graph_from_json(cache_path)

    # Download fresh using OSMnx covering Greater Ranchi (14km radius)
    return download_osm_road_network(
        center_lat=23.3699,
        center_lon=85.3253,
        dist_m=14000,
        network_type="drive",
        cache_path=cache_path,
    )


def resolve_location_or_coords(
    graph: Graph,
    user_input: str,
) -> Tuple[str, Tuple[float, float], str]:
    """
    Resolve a user input to:
    1. A nearest road network node ID (str)
    2. The resolved GPS coordinates (lat, lon)
    3. A human-readable display label (str)
    
    Accepts:
    - Landmark name (e.g., 'Albert Ekka Chowk', 'Railway Station', 'nucleus')
    - Explicit GPS coordinates (e.g., '23.3699, 85.3253')
    
    Raises:
        ValueError: If input cannot be recognized as a landmark or coordinates,
                    or if the point is too far from any road network node.
    """
    raw = user_input.strip()
    if not raw:
        raise ValueError("Location query cannot be empty.")

    # 1. Check if input is GPS coordinates
    coords = parse_coordinates(raw)
    if coords is not None:
        nearest_node, dist_m = graph.find_nearest_node(coords[0], coords[1])
        label = f"Coord ({coords[0]:.4f}, {coords[1]:.4f})"
        return (nearest_node, coords, label)

    # 2. Check if input matches known landmarks
    target = raw.lower()
    matched_landmark = None

    # Exact case-insensitive match
    for name, lm_coords in RANCHI_LANDMARKS.items():
        if name.lower() == target:
            matched_landmark = (name, lm_coords)
            break

    # Substring match if exact match not found
    if matched_landmark is None:
        for name, lm_coords in RANCHI_LANDMARKS.items():
            if target in name.lower() or name.lower() in target:
                matched_landmark = (name, lm_coords)
                break

    if matched_landmark is not None:
        name, lm_coords = matched_landmark
        nearest_node, dist_m = graph.find_nearest_node(lm_coords[0], lm_coords[1])
        return (nearest_node, lm_coords, name)

    # If neither landmark nor coords, raise helpful error
    known = ", ".join(f"'{k}'" for k in RANCHI_LANDMARKS.keys())
    raise ValueError(
        f"Unknown location '{raw}'. Please specify a recognized landmark (e.g. {known}) "
        "or GPS coordinates in 'lat, lon' format."
    )
