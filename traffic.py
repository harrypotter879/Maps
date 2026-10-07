"""
Traffic Simulation module for PathFinder.

Idea (kept deliberately simple):

    Road -> distance -> traffic level -> traffic-adjusted cost -> Dijkstra/A* -> route

* The PHYSICAL distance of a road (Edge.weight) is never changed.
* A traffic level (low / medium / high) is worked out for every road from
  (a) how important the road is and (b) the selected simulation mode.
* routing cost = distance x multiplier. Only the routing algorithm uses it.

This is a SIMULATION for demonstration. It is NOT real-time traffic data.

Nothing is stored on the graph: the traffic level is a pure function of the
road and the mode, so results are repeatable (no randomness) and the shared
road network is never modified.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from graph import Edge, Graph


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (the only place where traffic numbers live)
# ─────────────────────────────────────────────────────────────────────────────

LOW, MEDIUM, HIGH = "low", "medium", "high"

TRAFFIC_MULTIPLIERS: Dict[str, float] = {LOW: 1.0, MEDIUM: 1.5, HIGH: 2.5}
DEFAULT_TRAFFIC_LEVEL = LOW  # used whenever a level is missing / invalid

TRAFFIC_COLORS: Dict[str, str] = {LOW: "#16A34A", MEDIUM: "#F59E0B", HIGH: "#DC2626"}
TRAFFIC_ICONS: Dict[str, str] = {LOW: "🟢", MEDIUM: "🟡", HIGH: "🔴"}

# Simulation modes shown in the UI
NORMAL = "normal"
RUSH_HOUR = "rush_hour"
HEAVY_TRAFFIC = "heavy_traffic"

TRAFFIC_MODE_LABELS: Dict[str, str] = {
    NORMAL: "Normal",
    RUSH_HOUR: "Rush Hour",
    HEAVY_TRAFFIC: "Heavy Traffic",
}

# Road classes (guessed, because the map data has no road-type tag)
MAIN, SECONDARY, LOCAL = "main", "secondary", "local"

# Named roads containing one of these words are treated as main roads.
MAIN_ROAD_KEYWORDS = (
    "highway", "expressway", "bypass", "ring road", "national", "main",
    "airport", "parkway", "interstate", "route", "coast",
)
# Whole-word match, plus highway numbers such as "NH33" or "I-10".
_MAIN_ROAD_PATTERN = re.compile(
    r"\b(?:" + "|".join(MAIN_ROAD_KEYWORDS) + r")\b|\bnh\s*-?\d+|\bi-\d+"
)

# Unnamed road segments at least this long (same unit as Edge.weight,
# metres for the Ranchi map) count as secondary roads; shorter ones are local.
SECONDARY_MIN_LENGTH = 100.0

# What traffic level each (mode, road class) gets.
# A value that is a tuple means "split deterministically between the two".
TRAFFIC_MODE_RULES: Dict[str, Dict[str, object]] = {
    NORMAL:        {MAIN: MEDIUM,          SECONDARY: LOW,              LOCAL: LOW},
    RUSH_HOUR:     {MAIN: HIGH,            SECONDARY: MEDIUM,           LOCAL: LOW},
    HEAVY_TRAFFIC: {MAIN: HIGH,            SECONDARY: (HIGH, MEDIUM),   LOCAL: MEDIUM},
}

# Assumed speed ONLY for the clearly-labelled simulated time estimate.
SIMULATED_DRIVE_SPEED_KMH = 30.0

# A cost function maps (from_node, edge) -> routing cost
CostFunction = Callable[[str, Edge], float]


# ─────────────────────────────────────────────────────────────────────────────
# Core calculations
# ─────────────────────────────────────────────────────────────────────────────

def normalize_traffic_level(level: Optional[str]) -> str:
    """Return a valid level; missing or unknown values become LOW."""
    key = str(level).strip().lower() if level is not None else ""
    return key if key in TRAFFIC_MULTIPLIERS else DEFAULT_TRAFFIC_LEVEL


def get_traffic_multiplier(level: Optional[str]) -> float:
    """Multiplier for a traffic level (unknown / missing -> LOW = 1.0)."""
    return TRAFFIC_MULTIPLIERS[normalize_traffic_level(level)]


def calculate_traffic_cost(distance: float, level: Optional[str]) -> float:
    """routing cost = physical distance x traffic multiplier."""
    return distance * get_traffic_multiplier(level)


def is_valid_traffic_mode(mode: Optional[str]) -> bool:
    return mode in TRAFFIC_MODE_RULES


def classify_road(road_name: str, distance: float) -> str:
    """Guess MAIN / SECONDARY / LOCAL from the road name and segment length."""
    name = (road_name or "").strip().lower()
    if name:
        return MAIN if _MAIN_ROAD_PATTERN.search(name) else SECONDARY
    return SECONDARY if distance >= SECONDARY_MIN_LENGTH else LOCAL


def _stable_split(u: str, v: str) -> int:
    """Repeatable 0/1 value for a road (same for both directions)."""
    a, b = sorted((str(u), str(v)))
    return zlib.crc32(f"{a}|{b}".encode("utf-8")) & 1


def get_traffic_level(u: str, edge: Edge, mode: Optional[str]) -> str:
    """Traffic level of one road in a given mode. Never raises."""
    rules = TRAFFIC_MODE_RULES.get(mode) if mode else None
    if rules is None:
        return DEFAULT_TRAFFIC_LEVEL
    level = rules.get(classify_road(edge.road_name, edge.weight), DEFAULT_TRAFFIC_LEVEL)
    if isinstance(level, tuple):
        level = level[_stable_split(u, edge.destination)]
    return normalize_traffic_level(level)  # type: ignore[arg-type]


def apply_traffic_mode(mode: Optional[str]) -> Optional[CostFunction]:
    """
    Turn a simulation mode into a cost function for Dijkstra / A*.

    Returns None for "no traffic" (None / unknown mode), which means the
    algorithms use plain distance - i.e. Normal Routing.
    """
    if not is_valid_traffic_mode(mode):
        return None

    def cost(u: str, edge: Edge) -> float:
        return calculate_traffic_cost(edge.weight, get_traffic_level(u, edge, mode))

    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Route analysis (used by the UI)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RouteTrafficInfo:
    """Traffic summary of one route under one mode."""
    mode: str
    distance: float          # physical distance
    traffic_cost: float      # traffic-adjusted routing cost
    level_counts: Dict[str, int]
    level_distance: Dict[str, float]

    @property
    def high_roads(self) -> int:
        return self.level_counts.get(HIGH, 0)

    @property
    def impact(self) -> str:
        """Overall traffic impact label (not a time estimate)."""
        if self.distance <= 0:
            return LOW
        extra = self.traffic_cost / self.distance
        if extra >= 1.9:
            return HIGH
        if extra >= 1.25:
            return MEDIUM
        return LOW


def analyse_route(graph: Graph, path: List[str], mode: Optional[str]) -> Optional[RouteTrafficInfo]:
    """Summarise traffic along a node path. Returns None if mode is off."""
    if not is_valid_traffic_mode(mode):
        return None
    counts = {LOW: 0, MEDIUM: 0, HIGH: 0}
    dists = {LOW: 0.0, MEDIUM: 0.0, HIGH: 0.0}
    total_dist = 0.0
    total_cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph.get_edge(u, v)
        if edge is None:
            continue
        level = get_traffic_level(u, edge, mode)
        counts[level] += 1
        dists[level] += edge.weight
        total_dist += edge.weight
        total_cost += calculate_traffic_cost(edge.weight, level)
    return RouteTrafficInfo(mode, total_dist, total_cost, counts, dists)


def simulated_drive_minutes(distance_m: float, info: Optional[RouteTrafficInfo]) -> float:
    """
    Simulated drive time in minutes using the documented assumed speed
    (SIMULATED_DRIVE_SPEED_KMH), slowed by the traffic cost ratio.
    Always label this as 'Estimated / Simulated'.
    """
    if distance_m <= 0:
        return 0.0
    base = (distance_m / 1000.0) / SIMULATED_DRIVE_SPEED_KMH * 60.0
    ratio = (info.traffic_cost / info.distance) if info and info.distance > 0 else 1.0
    return base * ratio


# ─────────────────────────────────────────────────────────────────────────────
# Map overlay data
# ─────────────────────────────────────────────────────────────────────────────

def iter_traffic_segments(
    graph: Graph,
    mode: Optional[str],
    bounds: Tuple[float, float, float, float],
    max_segments: int = 6000,
) -> Dict[str, List[List[List[float]]]]:
    """
    Collect road polylines inside bounds=(min_lat, min_lon, max_lat, max_lon),
    grouped by traffic level. At most `max_segments` are returned, with the
    most congested roads kept first so the map stays fast.
    """
    groups: Dict[str, List[List[List[float]]]] = {HIGH: [], MEDIUM: [], LOW: []}
    if not is_valid_traffic_mode(mode):
        return groups
    min_lat, min_lon, max_lat, max_lon = bounds
    seen = set()
    for u in graph.get_nodes():
        c = graph.get_node_coords(u)
        if c is None or not (min_lat <= c[0] <= max_lat and min_lon <= c[1] <= max_lon):
            continue
        for edge in graph.get_neighbors(u):
            key = frozenset((u, edge.destination))
            if key in seen:
                continue
            seen.add(key)
            if edge.geometry:
                points = [[p[0], p[1]] for p in edge.geometry]
            else:
                d = graph.get_node_coords(edge.destination)
                if d is None:
                    continue
                points = [[c[0], c[1]], [d[0], d[1]]]
            groups[get_traffic_level(u, edge, mode)].append(points)

    budget = max_segments
    for level in (HIGH, MEDIUM, LOW):
        groups[level] = groups[level][:budget]
        budget -= len(groups[level])
    return groups
