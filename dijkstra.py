"""
Dijkstra's Algorithm module for PathFinder.

Provides an efficient implementation of Dijkstra's shortest path algorithm
using Python's built-in heapq module to find the minimum distance route between
nodes in a weighted graph.
"""

from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from graph import Graph, Edge


@dataclass(frozen=True)
class RouteLeg:
    """Represents a single segment/leg along a route."""
    origin: str
    destination: str
    distance: float
    road_name: str = ""


@dataclass(frozen=True)
class PathResult:
    """Encapsulates the result of a shortest-path query."""
    source: str
    destination: str
    path: List[str]
    legs: List[RouteLeg]
    total_distance: float
    execution_time_sec: float
    visited_nodes_count: int
    found: bool
    algorithm: str = "Dijkstra"
    # Traffic-adjusted routing cost of the route. None when traffic simulation
    # is off (total_distance is then the only cost, exactly as before).
    total_cost: Optional[float] = None


def find_shortest_path(
    graph: Graph,
    source: str,
    destination: str,
    cost_function: Optional[Callable[[str, Edge], float]] = None,
) -> PathResult:
    """
    Find the shortest path between `source` and `destination` using Dijkstra's algorithm.

    Uses a binary min-heap (via `heapq`) for priority queue operations.

    Args:
        graph: The Graph instance containing the road network.
        source: The starting location name.
        destination: The target location name.
        cost_function: Optional f(from_node, edge) -> routing cost, used for
            traffic simulation. If omitted, the cost is the edge distance
            (normal routing). The reported total_distance is always the
            physical distance when a cost function is used.

    Returns:
        PathResult containing the path, route legs, total distance,
        and execution metrics.

    Raises:
        KeyError: If either source or destination does not exist in the graph.
        ValueError: If a negative edge weight is encountered.
    """
    source = source.strip()
    destination = destination.strip()

    if not graph.has_node(source):
        raise KeyError(f"Source location '{source}' not found in the network.")
    if not graph.has_node(destination):
        raise KeyError(f"Destination location '{destination}' not found in the network.")

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
        )

    # Distances map: tracks best-known shortest distance from source to each node
    distances: Dict[str, float] = {node: float("inf") for node in graph.get_nodes()}
    distances[source] = 0.0

    # Predecessor map: node -> (previous_node, connecting_edge)
    previous: Dict[str, Tuple[Optional[str], Optional[Edge]]] = {
        node: (None, None) for node in graph.get_nodes()
    }

    # Min-heap priority queue: stores tuples of (cumulative_distance, node)
    pq: List[Tuple[float, str]] = []
    heapq.heappush(pq, (0.0, source))

    visited_count = 0
    settled: set[str] = set()

    found = False

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        # Skip if we already finalized the shortest path to this node
        if current_node in settled:
            continue
        
        settled.add(current_node)
        visited_count += 1

        # Early exit: the target has been settled with the optimal distance
        if current_node == destination:
            found = True
            break

        # If current popped distance is greater than the recorded distance, skip
        if current_dist > distances[current_node]:
            continue

        for edge in graph.get_neighbors(current_node):
            if edge.weight < 0:
                raise ValueError(
                    f"Negative edge weight ({edge.weight}) detected on connection "
                    f"'{current_node}' -> '{edge.destination}'. "
                    "Dijkstra's algorithm requires non-negative weights."
                )

            step_cost = cost_function(current_node, edge) if cost_function else edge.weight
            new_dist = current_dist + step_cost

            # Relaxation step: update neighbor distance if a shorter path is discovered
            if new_dist < distances[edge.destination]:
                distances[edge.destination] = new_dist
                previous[edge.destination] = (current_node, edge)
                heapq.heappush(pq, (new_dist, edge.destination))

    elapsed = time.perf_counter() - start_time

    if not found or distances[destination] == float("inf"):
        return PathResult(
            source=source,
            destination=destination,
            path=[],
            legs=[],
            total_distance=float("inf"),
            execution_time_sec=elapsed,
            visited_nodes_count=visited_count,
            found=False,
        )

    # Reconstruct shortest path and individual legs
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
        # Physical distance = sum of legs; routing cost = what Dijkstra minimised
        return PathResult(
            source=source,
            destination=destination,
            path=path,
            legs=legs,
            total_distance=round(sum(leg.distance for leg in legs), 3),
            execution_time_sec=elapsed,
            visited_nodes_count=visited_count,
            found=True,
            total_cost=round(distances[destination], 3),
        )

    return PathResult(
        source=source,
        destination=destination,
        path=path,
        legs=legs,
        total_distance=round(distances[destination], 3),
        execution_time_sec=elapsed,
        visited_nodes_count=visited_count,
        found=True,
    )
