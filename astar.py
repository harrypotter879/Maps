"""
A* Pathfinding Algorithm module for PathFinder (Stage 3).

Provides an efficient implementation of the A* (A-Star) search algorithm from scratch
using a binary min-heap (`heapq`) and an admissible geographic heuristic (Haversine distance)
to compute the optimal shortest path on road networks.
"""

from __future__ import annotations
import heapq
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

from graph import Graph, Edge
from dijkstra import RouteLeg, PathResult


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two (lat, lon) coordinates in meters.
    
    Uses the Haversine formula on a spherical Earth model (mean radius R = 6,371,000 m).
    Because great-circle distance is the absolute shortest possible distance between two
    geographical points, it is mathematically guaranteed to be an admissible and consistent
    heuristic for surface road networks (i.e. h(n) <= true road distance).
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    r_earth = 6_371_000.0  # Earth's mean radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    # Guard against float precision drift exceeding 1.0
    c = 2.0 * math.atan2(math.sqrt(min(1.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return r_earth * c


def default_geographic_heuristic(graph: Graph, node: str, target: str) -> float:
    """
    Admissible heuristic function calculating straight-line distance between two nodes.
    
    If either node lacks geographical coordinates (e.g. in the Stage 1 fictional graph),
    this returns 0.0, allowing A* to degrade gracefully and correctly to Dijkstra.
    """
    c1 = graph.get_node_coords(node)
    c2 = graph.get_node_coords(target)
    if c1 is not None and c2 is not None:
        return haversine_distance(c1, c2)
    return 0.0


def find_shortest_path_astar(
    graph: Graph,
    source: str,
    destination: str,
    heuristic: Optional[Callable[[Graph, str, str], float]] = None,
    edge_penalties: Optional[Dict[Tuple[str, str], float]] = None,
    cost_function: Optional[Callable[[str, Edge], float]] = None,
) -> PathResult:
    """
    Find the shortest path between `source` and `destination` using the A* algorithm from scratch.

    Args:
        graph: The Graph containing road nodes, edges, and coordinates.
        source: The starting location name / node ID.
        destination: The target location name / node ID.
        heuristic: Optional custom heuristic function h(graph, current_node, target_node).
                   Defaults to `default_geographic_heuristic` (Haversine straight-line distance).
        cost_function: Optional f(from_node, edge) -> routing cost (traffic simulation).
                   If omitted the cost is the edge distance (normal routing). Traffic
                   multipliers are >= 1, so the distance heuristic stays admissible.

    Returns:
        PathResult containing the reconstructed path, individual legs, total distance,
        performance metrics, and `algorithm="A*"`.

    Raises:
        KeyError: If either source or destination is not in the graph.
        ValueError: If a negative edge weight is detected.
    """
    source = source.strip()
    destination = destination.strip()

    if not graph.has_node(source):
        raise KeyError(f"Source location '{source}' not found in the network.")
    if not graph.has_node(destination):
        raise KeyError(f"Destination location '{destination}' not found in the network.")

    if heuristic is None:
        heuristic = default_geographic_heuristic

    start_time = time.perf_counter()

    # Trivial case: source is the destination
    if source == destination:
        elapsed = time.perf_counter() - start_time
        return PathResult(
            source=source,
            destination=destination,
            path=[source],
            legs=[],
            total_distance=0.0,
            execution_time_sec=elapsed,
            visited_nodes_count=1,
            found=True,
            algorithm="A*",
        )

    # g_score: exact shortest cost known from source to each node
    g_score: Dict[str, float] = {node: float("inf") for node in graph.get_nodes()}
    g_score[source] = 0.0

    # predecessor map: node -> (previous_node, connecting_edge)
    previous: Dict[str, Tuple[Optional[str], Optional[Edge]]] = {
        node: (None, None) for node in graph.get_nodes()
    }

    # Initial heuristic estimate from source to destination
    h_start = heuristic(graph, source, destination)

    # Priority queue stores tuples of: (f_score, insertion_counter, node_id)
    # The insertion counter ensures strict FIFO tie-breaking when f_scores match
    counter = 0
    pq: List[Tuple[float, int, str]] = []
    heapq.heappush(pq, (h_start, counter, source))

    visited_count = 0
    settled: set[str] = set()
    found = False

    while pq:
        current_f, _, current_node = heapq.heappop(pq)

        # Skip if this node's optimal path has already been settled
        if current_node in settled:
            continue

        settled.add(current_node)
        visited_count += 1

        # Goal check: target reached with optimal cost
        if current_node == destination:
            found = True
            break

        current_g = g_score[current_node]

        for edge in graph.get_neighbors(current_node):
            weight = cost_function(current_node, edge) if cost_function else edge.weight
            if edge_penalties:
                weight *= edge_penalties.get((current_node, edge.destination), 1.0)
                
            if weight < 0:
                raise ValueError(
                    f"Negative edge weight ({weight}) detected on connection "
                    f"'{current_node}' -> '{edge.destination}'. "
                    "A* search requires non-negative edge weights."
                )

            tentative_g = current_g + weight

            # Relaxation step: update neighbor if a shorter route to it was discovered
            if tentative_g < g_score[edge.destination]:
                g_score[edge.destination] = tentative_g
                h_cost = heuristic(graph, edge.destination, destination)
                f_cost = tentative_g + h_cost
                previous[edge.destination] = (current_node, edge)
                counter += 1
                heapq.heappush(pq, (f_cost, counter, edge.destination))

    elapsed = time.perf_counter() - start_time

    if not found or g_score[destination] == float("inf"):
        return PathResult(
            source=source,
            destination=destination,
            path=[],
            legs=[],
            total_distance=float("inf"),
            execution_time_sec=elapsed,
            visited_nodes_count=visited_count,
            found=False,
            algorithm="A*",
        )

    # Reconstruct optimal path and individual route legs
    path: List[str] = []
    legs: List[RouteLeg] = []
    curr: Optional[str] = destination

    while curr is not None:
        path.append(curr)
        prev_node, edge = previous[curr]
        if prev_node is not None and edge is not None:
            legs.append(
                RouteLeg(
                    origin=prev_node,
                    destination=curr,
                    distance=edge.weight,
                    road_name=edge.road_name,
                )
            )
        curr = prev_node

    path.reverse()
    legs.reverse()

    if cost_function is not None:
        # Physical distance = sum of legs; traffic cost = same legs re-priced
        # (kept separate from g_score so alternative-route penalties don't leak in).
        traffic_cost = 0.0
        for leg in legs:
            e = graph.get_edge(leg.origin, leg.destination)
            traffic_cost += cost_function(leg.origin, e) if e else leg.distance
        return PathResult(
            source=source,
            destination=destination,
            path=path,
            legs=legs,
            total_distance=round(sum(leg.distance for leg in legs), 3),
            execution_time_sec=elapsed,
            visited_nodes_count=visited_count,
            found=True,
            algorithm="A*",
            total_cost=round(traffic_cost, 3),
        )

    return PathResult(
        source=source,
        destination=destination,
        path=path,
        legs=legs,
        total_distance=round(g_score[destination], 3),
        execution_time_sec=elapsed,
        visited_nodes_count=visited_count,
        found=True,
        algorithm="A*",
    )

def find_alternative_routes(
    graph: Graph,
    source: str,
    destination: str,
    max_routes: int = 3,
    penalty_factor: float = 1.5,
    cost_function: Optional[Callable[[str, Edge], float]] = None,
) -> List[PathResult]:
    """
    Finds up to `max_routes` alternative paths between source and destination.
    Uses an edge-penalization approach on the A* algorithm.
    """
    routes = []
    penalties: Dict[Tuple[str, str], float] = {}

    for _ in range(max_routes):
        res = find_shortest_path_astar(
            graph, source, destination, edge_penalties=penalties,
            cost_function=cost_function,
        )
        if not res.found:
            break

        # Check if this route is substantially different or identical
        # by checking if path is already in routes
        if not any(r.path == res.path for r in routes):
            routes.append(res)
        else:
            # If we keep finding the exact same route even with penalties,
            # we might want to increase the penalty factor further for its edges
            # But normally, multiplying by penalty factor breaks the tie.
            pass

        # Penalize edges of the newly found path to encourage diversity
        for i in range(len(res.path) - 1):
            u = res.path[i]
            v = res.path[i+1]
            penalties[(u, v)] = penalties.get((u, v), 1.0) * penalty_factor
            # Also penalize the reverse edge if we want to discourage backtracking
            penalties[(v, u)] = penalties.get((v, u), 1.0) * penalty_factor

    return routes
