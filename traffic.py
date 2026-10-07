"""Deterministic, modelled traffic and travel-time estimates for Ranchi roads.

OSM highway tags are used when present. The bundled Ranchi cache predates
highway-tag storage, so its unclassified roads use a conservative documented
fallback based on recognizable road names and an urban default. This module
never changes physical edge lengths and does not represent live traffic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Callable, Dict, List, Optional, Tuple

from graph import Edge, Graph

LOW, MEDIUM, HIGH = "low", "medium", "high"
TRAFFIC_COLORS = {LOW: "#16A34A", MEDIUM: "#EAB308", HIGH: "#DC2626"}
TRAFFIC_ICONS = {LOW: "🟢", MEDIUM: "🟡", HIGH: "🔴"}

SHORTEST_DISTANCE = "shortest_distance"
TRAFFIC_AWARE = "traffic_aware"
TRAFFIC_MODE_LABELS = {
    SHORTEST_DISTANCE: "Shortest Distance",
    TRAFFIC_AWARE: "Traffic-Aware",
}

# Model assumptions in km/h, not posted speed limits or measured Ranchi speeds.
BASE_SPEEDS_KMH: Dict[str, float] = {
    "motorway": 80.0, "trunk": 60.0, "primary": 45.0,
    "secondary": 35.0, "tertiary": 30.0, "unclassified": 25.0,
    "residential": 20.0, "living_street": 12.0, "service": 15.0,
    "road": 20.0, "unknown": 25.0,
}
MAX_MODELED_SPEED_KMH = max(BASE_SPEEDS_KMH.values())

# Modeled periods are adjustable assumptions, not observations.
TIME_PERIODS = (
    (time(6, 0), time(7, 30), "early_morning"),
    (time(7, 30), time(10, 0), "morning_peak"),
    (time(10, 0), time(16, 0), "midday"),
    (time(16, 0), time(20, 0), "evening_peak"),
)
PERIOD_LABELS = {
    "early_morning": "Early morning", "morning_peak": "Morning peak",
    "midday": "Midday", "evening_peak": "Evening peak", "night": "Night",
}

# Modeled fraction of base speed remaining at each hour (00:00 through 23:00).
# Values are interpolated continuously between hours. They are gentle, generic
# urban patterns, not measured Ranchi traffic. Evening activity extends onto
# secondary roads; midday congestion is concentrated on major roads.
HOURLY_SPEED_FACTORS: Dict[str, Tuple[float, ...]] = {
    "major": (
        .96, .955, .95, .945, .94, .93, .89, .84, .78, .75, .78, .80,
        .82, .83, .82, .80, .77, .73, .70, .72, .76, .82, .88, .93,
    ),
    "secondary": (
        .98, .975, .97, .965, .96, .95, .94, .91, .88, .86, .87, .88,
        .89, .90, .89, .87, .84, .79, .74, .76, .80, .85, .91, .96,
    ),
    "local": (
        .99, .989, .988, .987, .985, .98, .975, .97, .965, .96, .962, .964,
        .966, .968, .966, .962, .958, .954, .95, .952, .96, .97, .98, .987,
    ),
    "unknown": (
        .997, .996, .995, .994, .993, .992, .99, .988, .986, .984, .985, .986,
        .987, .988, .987, .986, .985, .984, .983, .984, .986, .989, .993, .996,
    ),
}
TRAFFIC_LEVEL_THRESHOLDS = {HIGH: 0.76, MEDIUM: 0.88}

_MAJOR_NAME = re.compile(r"\b(?:nh\s*\d+|national highway|highway|expressway|bypass|ring road|main road|airport road)\b", re.I)
_LOCAL_NAME = re.compile(r"\b(?:lane|service road|living street|colony road|gali)\b", re.I)
_HIGHWAY_SPEED_CLASS = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "unclassified": "unclassified", "residential": "residential",
    "living_street": "living_street", "service": "service",
    "road": "road",
}

CostFunction = Callable[[str, Edge], float]


def normalize_departure_time(value: time | str | None) -> time:
    """Accept a ``datetime.time`` or HH:MM string; default to 08:30."""
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    if isinstance(value, str):
        try:
            hour, minute = value.strip().split(":", 1)
            return time(int(hour), int(minute))
        except (ValueError, TypeError):
            pass
    return time(8, 30)


def get_traffic_period(departure_time: time | str | None) -> str:
    departure = normalize_departure_time(departure_time)
    for start, end, period in TIME_PERIODS:
        if start <= departure < end:
            return period
    return "night"


def classify_road(edge: Edge) -> str:
    """Return the OSM highway class, with a name-only fallback for old cache."""
    tag = (edge.highway or "").strip().lower()
    # OSM can provide a list-like value in custom caches.
    tag = tag.split(",", 1)[0].strip()
    if tag in _HIGHWAY_SPEED_CLASS:
        return _HIGHWAY_SPEED_CLASS[tag]
    if tag:
        return "unknown"
    name = edge.road_name or ""
    if _MAJOR_NAME.search(name):
        return "primary"
    if _LOCAL_NAME.search(name):
        return "service"
    # Legacy cache has no highway tag. A named connector gets a modest
    # secondary-road assumption; unnamed segments use the urban default.
    if name.strip():
        return "tertiary"
    return "unknown"


def road_traffic_group(edge: Edge) -> str:
    road_class = classify_road(edge)
    if road_class in {"motorway", "trunk", "primary"}:
        return "major"
    if road_class in {"secondary", "tertiary"}:
        return "secondary"
    if road_class in {"residential", "living_street", "service", "unclassified"}:
        return "local"
    return "unknown"


def modeled_speed_factor(edge: Edge, departure_time: time | str | None) -> float:
    """Interpolate the road-group speed factor continuously across the day."""
    group = road_traffic_group(edge)
    profile = HOURLY_SPEED_FACTORS[group]
    departure = normalize_departure_time(departure_time)
    hour = departure.hour + departure.minute / 60.0
    lower_hour = int(hour)
    upper_hour = (lower_hour + 1) % 24
    fraction = hour - lower_hour
    return profile[lower_hour] + (profile[upper_hour] - profile[lower_hour]) * fraction


def get_traffic_level(edge: Edge, departure_time: time | str | None) -> str:
    factor = modeled_speed_factor(edge, departure_time)
    if factor <= TRAFFIC_LEVEL_THRESHOLDS[HIGH]:
        return HIGH
    if factor <= TRAFFIC_LEVEL_THRESHOLDS[MEDIUM]:
        return MEDIUM
    return LOW


def get_traffic_color(edge: Edge, departure_time: time | str | None) -> str:
    """Return a smooth green-to-red color from the modeled speed factor."""
    factor = modeled_speed_factor(edge, departure_time)
    green, yellow, red = (22, 163, 74), (234, 179, 8), (220, 38, 38)
    if factor >= 0.92:
        return "#16A34A"
    if factor >= 0.80:
        start, end = green, yellow
        progress = (0.92 - factor) / 0.12
    elif factor >= 0.64:
        start, end = yellow, red
        progress = (0.80 - factor) / 0.16
    else:
        return "#DC2626"
    rgb = tuple(round(a + (b - a) * progress) for a, b in zip(start, end))
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def estimated_speed_kmh(edge: Edge, departure_time: time | str | None) -> float:
    road_class = classify_road(edge)
    base_speed = BASE_SPEEDS_KMH.get(road_class, BASE_SPEEDS_KMH["unknown"])
    return base_speed * modeled_speed_factor(edge, departure_time)


def estimated_travel_time_seconds(edge: Edge, departure_time: time | str | None) -> float:
    """Estimated seconds for this edge; ``edge.weight`` remains physical meters."""
    if edge.weight <= 0:
        return 0.0
    return edge.weight * 3.6 / estimated_speed_kmh(edge, departure_time)


def apply_traffic_mode(mode: Optional[str], departure_time: time | str | None = None) -> Optional[CostFunction]:
    """Return edge travel time in seconds for traffic-aware routing."""
    if mode != TRAFFIC_AWARE:
        return None
    departure = normalize_departure_time(departure_time)

    def cost(_u: str, edge: Edge) -> float:
        return estimated_travel_time_seconds(edge, departure)

    # A* can safely use straight-line distance / maximum possible speed.
    cost.heuristic_multiplier = 3.6 / MAX_MODELED_SPEED_KMH  # type: ignore[attr-defined]
    return cost


@dataclass(frozen=True)
class RouteTrafficInfo:
    mode: str
    departure_time: time
    distance: float
    travel_time_seconds: float
    level_counts: Dict[str, int]
    level_distance: Dict[str, float]

    @property
    def high_roads(self) -> int:
        return self.level_counts.get(HIGH, 0)

    @property
    def estimated_speed_kmh(self) -> float:
        if self.travel_time_seconds <= 0:
            return 0.0
        return self.distance * 3.6 / self.travel_time_seconds

    @property
    def impact(self) -> str:
        total = sum(self.level_counts.values())
        if not total:
            return LOW
        if self.level_counts[HIGH] / total >= 0.35:
            return HIGH
        if self.level_counts[MEDIUM] + self.level_counts[HIGH] > 0:
            return MEDIUM
        return LOW


def analyse_route(graph: Graph, path: List[str], mode: Optional[str], departure_time: time | str | None = None) -> Optional[RouteTrafficInfo]:
    if mode != TRAFFIC_AWARE:
        return None
    departure = normalize_departure_time(departure_time)
    counts = {LOW: 0, MEDIUM: 0, HIGH: 0}
    dists = {LOW: 0.0, MEDIUM: 0.0, HIGH: 0.0}
    total_dist = total_time = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph.get_edge(u, v)
        if edge is None:
            continue
        level = get_traffic_level(edge, departure)
        counts[level] += 1
        dists[level] += edge.weight
        total_dist += edge.weight
        total_time += estimated_travel_time_seconds(edge, departure)
    return RouteTrafficInfo(mode, departure, total_dist, total_time, counts, dists)


def iter_colored_traffic_segments(
    graph: Graph,
    mode: Optional[str],
    departure_time: time | str | None,
    path: Optional[List[str]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_segments: int = 6000,
) -> List[Tuple[str, str, List[List[float]]]]:
    """Return colored route segments, or a bounded nearby sample before routing."""
    groups: Dict[str, List[Tuple[str, str, List[List[float]]]]] = {HIGH: [], MEDIUM: [], LOW: []}
    if mode != TRAFFIC_AWARE:
        return []
    if path:
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.get_edge(u, v)
            if edge is None:
                continue
            if edge.geometry:
                points = [[p[0], p[1]] for p in edge.geometry]
            else:
                origin = graph.get_node_coords(u)
                destination = graph.get_node_coords(v)
                if origin is None or destination is None:
                    continue
                points = [[origin[0], origin[1]], [destination[0], destination[1]]]
            level = get_traffic_level(edge, departure_time)
            groups[level].append((get_traffic_color(edge, departure_time), level, points))
        return [segment for level in (HIGH, MEDIUM, LOW) for segment in groups[level]]

    if bounds is None:
        return []
    min_lat, min_lon, max_lat, max_lon = bounds
    seen = set()
    for u in graph.get_nodes():
        origin = graph.get_node_coords(u)
        if origin is None or not (min_lat <= origin[0] <= max_lat and min_lon <= origin[1] <= max_lon):
            continue
        for edge in graph.get_neighbors(u):
            key = frozenset((u, edge.destination))
            if key in seen:
                continue
            seen.add(key)
            destination = graph.get_node_coords(edge.destination)
            if edge.geometry:
                points = [[p[0], p[1]] for p in edge.geometry]
            elif destination is not None:
                points = [[origin[0], origin[1]], [destination[0], destination[1]]]
            else:
                continue
            level = get_traffic_level(edge, departure_time)
            groups[level].append((get_traffic_color(edge, departure_time), level, points))
    remaining = max_segments
    selected = []
    for level in (HIGH, MEDIUM, LOW):
        groups[level] = groups[level][:remaining]
        remaining -= len(groups[level])
        selected.extend(groups[level])
    return selected


def iter_traffic_segments(
    graph: Graph,
    mode: Optional[str],
    departure_time: time | str | None,
    path: Optional[List[str]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_segments: int = 6000,
) -> Dict[str, List[List[List[float]]]]:
    """Group route or nearby traffic segments by their Low/Medium/High label."""
    groups: Dict[str, List[List[List[float]]]] = {HIGH: [], MEDIUM: [], LOW: []}
    for _color, level, points in iter_colored_traffic_segments(
        graph, mode, departure_time, path=path, bounds=bounds, max_segments=max_segments,
    ):
        groups[level].append(points)
    return groups
