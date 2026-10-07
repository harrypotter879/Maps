"""
Graph module for PathFinder.

Provides data structures to represent road networks using a weighted adjacency list.
Supports both directed and undirected (bidirectional) edges and includes a default
fictional road network.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from difflib import get_close_matches
import math
from typing import Dict, List, Optional, Tuple, Set


@dataclass(frozen=True)
class Edge:
    """Represents a weighted connection between two locations."""
    destination: str
    weight: float
    road_name: str = ""
    geometry: Optional[Tuple[Tuple[float, float], ...]] = None
    highway: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {self.weight}")


class Graph:
    """
    Weighted graph represented as an adjacency list.
    
    Nodes are unique strings (e.g., city names or OSM intersection IDs).
    Adjacency list maps each node to a list of outgoing Edge objects.
    Optionally stores latitude/longitude coordinates and edge geometries.
    """

    def __init__(self) -> None:
        self._adjacency_list: Dict[str, List[Edge]] = {}
        self._node_coords: Dict[str, Tuple[float, float]] = {}

    def add_node(
        self,
        node: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> bool:
        """
        Add a node to the graph if it doesn't already exist.
        Optionally records geographical coordinates (lat, lon).
        
        Returns True if the node was added, False if it already existed.
        """
        node = node.strip()
        if not node:
            raise ValueError("Node name cannot be empty.")
        if lat is not None and lon is not None:
            self._node_coords[node] = (float(lat), float(lon))
        if node not in self._adjacency_list:
            self._adjacency_list[node] = []
            return True
        return False

    def add_edge(
        self,
        u: str,
        v: str,
        weight: float,
        bidirectional: bool = True,
        road_name: str = "",
        geometry: Optional[List[Tuple[float, float]]] = None,
        highway: Optional[str] = None,
    ) -> None:
        """
        Add a weighted edge between nodes u and v.
        
        If nodes u or v do not exist, they are automatically added.
        If bidirectional is True, adds edges in both directions.
        Optionally stores geometry coordinates along the road curvature.
        """
        u = u.strip()
        v = v.strip()
        if not u or not v:
            raise ValueError("Node names cannot be empty.")
        if weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {weight}")

        self.add_node(u)
        self.add_node(v)

        geom_tuple = tuple(geometry) if geometry else None

        # Check if an edge already exists from u to v; update if new weight is smaller
        self._add_or_update_edge(u, v, weight, road_name, geom_tuple, highway)

        if bidirectional and u != v:
            rev_geom = tuple(reversed(geometry)) if geometry else None
            self._add_or_update_edge(v, u, weight, road_name, rev_geom, highway)

    def _add_or_update_edge(
        self,
        u: str,
        v: str,
        weight: float,
        road_name: str,
        geometry: Optional[Tuple[Tuple[float, float], ...]] = None,
        highway: Optional[str] = None,
    ) -> None:
        """Helper to add an edge or update it if a cheaper edge is added."""
        edges = self._adjacency_list[u]
        for idx, edge in enumerate(edges):
            if edge.destination == v:
                # Update if the new edge is shorter or identical
                if weight <= edge.weight:
                    edges[idx] = Edge(
                        destination=v,
                        weight=weight,
                        road_name=road_name,
                        geometry=geometry,
                        highway=highway,
                    )
                return
        edges.append(
            Edge(
                destination=v,
                weight=weight,
                road_name=road_name,
                geometry=geometry,
                highway=highway,
            )
        )

    def get_node_coords(self, node: str) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) coordinates of a node if available, else None."""
        return self._node_coords.get(node.strip())

    def has_coords(self) -> bool:
        """Return True if node coordinates are present in the graph."""
        return bool(self._node_coords)

    def find_nearest_node(self, lat: float, lon: float) -> Tuple[str, float]:
        """
        Find the closest node in the graph to the specified (lat, lon).
        Returns a tuple of (nearest_node_id, distance_in_meters).
        Raises ValueError if the graph has no recorded coordinates.
        """
        if not self._node_coords:
            raise ValueError("Cannot search nearest node: graph has no geographical coordinates.")

        try:
            import numpy as np
            node_ids = list(self._node_coords.keys())
            coords_arr = np.array([self._node_coords[n] for n in node_ids], dtype=np.float64)
            lats = coords_arr[:, 0]
            lons = coords_arr[:, 1]
            cos_lat = math.cos(math.radians(lat))
            dlat = (lats - lat) * 111320.0
            dlon = (lons - lon) * (111320.0 * cos_lat)
            dists_sq = dlat * dlat + dlon * dlon
            best_idx = int(np.argmin(dists_sq))
            return node_ids[best_idx], math.sqrt(float(dists_sq[best_idx]))
        except ImportError:
            best_node = ""
            best_dist = float("inf")
            cos_lat = math.cos(math.radians(lat))
            for n, (nlat, nlon) in self._node_coords.items():
                dlat = (nlat - lat) * 111320.0
                dlon = (nlon - lon) * (111320.0 * cos_lat)
                d_sq = dlat * dlat + dlon * dlon
                if d_sq < best_dist:
                    best_dist = d_sq
                    best_node = n
            return best_node, math.sqrt(best_dist)

    def has_node(self, node: str) -> bool:
        """Return True if the node exists in the graph."""
        return node in self._adjacency_list

    def get_nodes(self) -> List[str]:
        """Return a sorted list of all node names in the graph."""
        return sorted(self._adjacency_list.keys())

    def get_neighbors(self, node: str) -> List[Edge]:
        """
        Return the list of outgoing edges for a given node.
        
        Raises KeyError if the node does not exist in the graph.
        """
        if node not in self._adjacency_list:
            raise KeyError(f"Location '{node}' is not in the road network.")
        return list(self._adjacency_list[node])

    def get_edge(self, u: str, v: str) -> Optional[Edge]:
        """Return the Edge from u to v, or None if no direct connection exists."""
        if u not in self._adjacency_list or v not in self._adjacency_list:
            return None
        for edge in self._adjacency_list[u]:
            if edge.destination == v:
                return edge
        return None

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._adjacency_list)

    @property
    def edge_count(self) -> int:
        """Return total directed edges in the graph."""
        return sum(len(edges) for edges in self._adjacency_list.values())

    def find_close_matches(self, node: str) -> List[str]:
        """
        Find potential matching node names for fuzzy error suggestion.
        Performs case-insensitive matching, substring containment, and edit-distance matching.
        """
        raw = node.strip()
        target = raw.lower()
        all_nodes = list(self._adjacency_list.keys())

        # Exact case-insensitive match
        exact_ignore_case = [n for n in all_nodes if n.lower() == target]
        if exact_ignore_case:
            return exact_ignore_case

        # Substring containment
        substr_matches = [n for n in all_nodes if target in n.lower() or n.lower() in target]

        # Fuzzy string similarity using difflib
        fuzzy_matches = get_close_matches(raw, all_nodes, n=3, cutoff=0.5)

        # Merge results preserving order and removing duplicates
        seen: Set[str] = set()
        results: List[str] = []
        for candidate in substr_matches + fuzzy_matches:
            if candidate not in seen:
                seen.add(candidate)
                results.append(candidate)

        return results

    def __contains__(self, node: str) -> bool:
        return self.has_node(node)

    def __repr__(self) -> str:
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"


def create_sample_road_network() -> Graph:
    """
    Construct and return a rich fictional road network.
    
    Topography:
    - Central Hub: Grand Haven (Metropolis / capital port)
    - West / Coastal: Silverfall, Port Marina, Bayview
    - North / Mountainous: Pinecrest, High Peak, Frostford
    - East / Forest & Valleys: Oakridge, Riverdale, Mistwood, Greenfield
    - South / Agricultural: Sunset Valley, Ironhold, Amber Plains
    - Disconnected / Offshore: Storm Island, Isle Outpost (unreachable by road)
    
    This provides varied pathfinding options:
    - Multiple alternative routes between major hubs
    - Highways vs scenic mountain passes
    - Detours around mountain ranges
    - Disconnected island components for testing unreachable routes
    """
    g = Graph()

    # Major Highway: Route 1 (North-South Coastal Corridor)
    g.add_edge("Port Marina", "Bayview", 18.5, road_name="Route 1 (Coast Highway)")
    g.add_edge("Bayview", "Silverfall", 22.0, road_name="Route 1 (Coast Highway)")
    g.add_edge("Silverfall", "Grand Haven", 35.0, road_name="Route 1 (Bay Highway)")

    # Major Highway: Interstate 10 (East-West Central Expressway)
    g.add_edge("Silverfall", "Riverdale", 28.0, road_name="I-10 Expressway")
    g.add_edge("Grand Haven", "Riverdale", 15.0, road_name="I-10 Expressway")
    g.add_edge("Riverdale", "Oakridge", 24.5, road_name="I-10 Expressway")
    g.add_edge("Oakridge", "Mistwood", 31.0, road_name="I-10 East Extension")

    # Northern Mountain Region (High altitude, winding roads)
    g.add_edge("Grand Haven", "Pinecrest", 42.0, road_name="Pine Valley Road")
    g.add_edge("Silverfall", "Pinecrest", 50.0, road_name="Old Forest Pass")
    g.add_edge("Pinecrest", "High Peak", 19.5, road_name="High Peak Trail")
    g.add_edge("High Peak", "Frostford", 27.0, road_name="North Mountain Pass")
    g.add_edge("Frostford", "Riverdale", 55.0, road_name="Frost Gorge Parkway")

    # Eastern Woodlands & Plains
    g.add_edge("Riverdale", "Greenfield", 18.0, road_name="Green Valley Way")
    g.add_edge("Greenfield", "Oakridge", 12.0, road_name="East Meadow Road")
    g.add_edge("Greenfield", "Amber Plains", 26.0, road_name="Plains Route 5")
    g.add_edge("Oakridge", "Amber Plains", 20.0, road_name="Sunridge Pass")

    # Southern Foothills & Mining Belt
    g.add_edge("Grand Haven", "Sunset Valley", 25.0, road_name="South Coast Highway")
    g.add_edge("Sunset Valley", "Ironhold", 30.0, road_name="Iron Foothills Road")
    g.add_edge("Ironhold", "Amber Plains", 38.0, road_name="Mineral Way")
    g.add_edge("Riverdale", "Sunset Valley", 32.0, road_name="Riverbend Cutoff")

    # Disconnected offshore region (accessible only by ferry/air, no road connection)
    g.add_edge("Storm Island", "Isle Outpost", 14.0, road_name="Island Perimeter Road")

    return g
