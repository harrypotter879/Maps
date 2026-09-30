"""
Graph module for PathFinder.

Provides data structures to represent road networks using a weighted adjacency list.
Supports both directed and undirected (bidirectional) edges and includes a default
fictional road network.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple, Set


@dataclass(frozen=True)
class Edge:
    """Represents a weighted connection between two locations."""
    destination: str
    weight: float
    road_name: str = ""

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {self.weight}")


class Graph:
    """
    Weighted graph represented as an adjacency list.
    
    Nodes are unique strings (e.g., city or junction names).
    Adjacency list maps each node to a list of outgoing Edge objects.
    """

    def __init__(self) -> None:
        self._adjacency_list: Dict[str, List[Edge]] = {}

    def add_node(self, node: str) -> bool:
        """
        Add a node to the graph if it doesn't already exist.
        
        Returns True if the node was added, False if it already existed.
        """
        node = node.strip()
        if not node:
            raise ValueError("Node name cannot be empty.")
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
        road_name: str = ""
    ) -> None:
        """
        Add a weighted edge between nodes u and v.
        
        If nodes u or v do not exist, they are automatically added.
        If bidirectional is True, adds edges in both directions.
        """
        u = u.strip()
        v = v.strip()
        if not u or not v:
            raise ValueError("Node names cannot be empty.")
        if weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {weight}")

        self.add_node(u)
        self.add_node(v)

        # Check if an edge already exists from u to v; update if new weight is smaller
        self._add_or_update_edge(u, v, weight, road_name)

        if bidirectional and u != v:
            self._add_or_update_edge(v, u, weight, road_name)

    def _add_or_update_edge(self, u: str, v: str, weight: float, road_name: str) -> None:
        """Helper to add an edge or update it if a cheaper edge is added."""
        edges = self._adjacency_list[u]
        for idx, edge in enumerate(edges):
            if edge.destination == v:
                # Update if the new edge is shorter or identical
                if weight <= edge.weight:
                    edges[idx] = Edge(destination=v, weight=weight, road_name=road_name)
                return
        edges.append(Edge(destination=v, weight=weight, road_name=road_name))

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
