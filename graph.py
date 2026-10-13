"""
Graph module for PathFinder.

Provides data structures to represent road networks using a weighted adjacency list.
Supports both directed and undirected (bidirectional) edges and includes a default
fictional road network.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# We only use Python's built-in modules here — no heavy dependencies needed
# for the core data structures.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations          # Allows forward references in type hints
from dataclasses import dataclass, field    # Provides clean, auto-generated class boilerplate
from difflib import get_close_matches       # Used for fuzzy / typo-tolerant node name matching
import math                                 # For Euclidean distance and trigonometry in geo-search
from typing import Dict, List, Optional, Tuple, Set  # Type hints for clarity


# ─────────────────────────────────────────────────────────────────────────────
# EDGE — The basic building block of our road network
#
# Think of an Edge as a single road segment connecting two locations.
# It carries:
#   • destination  — where this road leads
#   • weight       — how "costly" this road is (distance in km, time, etc.)
#   • road_name    — human-readable label (e.g. "Route 1 Coast Highway")
#   • geometry     — optional GPS waypoints tracing the road's curve on a map
#   • highway      — road type tag borrowed from OpenStreetMap (e.g. "motorway")
#
# frozen=True makes Edge immutable — once created, it cannot be accidentally
# changed, which prevents subtle bugs during pathfinding.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Edge:
    """Represents a weighted connection between two locations."""
    destination: str                                          # The node this edge points to
    weight: float                                             # Cost of traversal (e.g. km or minutes)
    road_name: str = ""                                       # Optional human-readable road label
    geometry: Optional[Tuple[Tuple[float, float], ...]] = None  # GPS waypoints along the road curve
    highway: Optional[str] = None                             # OSM road type (e.g. 'motorway', 'residential')

    def __post_init__(self) -> None:
        # Validation: a road with negative distance makes no physical sense.
        # This guard catches data-entry errors early, before they corrupt
        # pathfinding results.
        if self.weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {self.weight}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH — The complete road network
#
# We model the road network as a Weighted Directed Graph, stored internally
# as an Adjacency List.
#
# Why an Adjacency List?
#   ✔ Memory-efficient for sparse networks (cities don't connect to every other city)
#   ✔ Fast neighbor lookup — O(degree) per node, which is small in road networks
#   ✔ Easy to add/remove edges dynamically
#
# Core Data:
#   _adjacency_list  — { "CityA": [Edge→CityB, Edge→CityC, ...], ... }
#   _node_coords     — { "CityA": (latitude, longitude), ... }
#                      Used for map rendering and nearest-node search.
# ─────────────────────────────────────────────────────────────────────────────
class Graph:
    """
    Weighted graph represented as an adjacency list.
    
    Nodes are unique strings (e.g., city names or OSM intersection IDs).
    Adjacency list maps each node to a list of outgoing Edge objects.
    Optionally stores latitude/longitude coordinates and edge geometries.
    """

    def __init__(self) -> None:
        # The main adjacency list: maps every node name → list of outgoing edges.
        # This is the heart of the graph data structure.
        self._adjacency_list: Dict[str, List[Edge]] = {}

        # Optional geographical coordinates for each node (latitude, longitude).
        # Populated when loading real map data from OpenStreetMap (OSMnx).
        # Used for: map rendering, nearest-node queries, and heuristic (A* search).
        self._node_coords: Dict[str, Tuple[float, float]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # add_node — Register a city/intersection in the graph
    #
    # Before we can connect two places with a road, both places must exist
    # as nodes. This method safely registers a node (city / OSM intersection)
    # and optionally pins its real-world GPS coordinates.
    #
    # Idempotent: calling it twice for the same node does nothing harmful.
    # Returns True if it was genuinely new, False if it already existed.
    # ─────────────────────────────────────────────────────────────────────────
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
        node = node.strip()         # Remove accidental leading/trailing whitespace
        if not node:
            raise ValueError("Node name cannot be empty.")

        # Store GPS coordinates if provided — these power the map view and A* heuristic
        if lat is not None and lon is not None:
            self._node_coords[node] = (float(lat), float(lon))

        # Only insert into the adjacency list if the node is brand new
        if node not in self._adjacency_list:
            self._adjacency_list[node] = []   # Start with an empty neighbor list
            return True                        # Newly added
        return False                           # Already existed — no change

    # ─────────────────────────────────────────────────────────────────────────
    # add_edge — Connect two nodes (cities) with a road
    #
    # This is the primary way to build the road network.
    # Key design decisions:
    #   • Auto-creates nodes that don't exist yet (convenience)
    #   • bidirectional=True by default — most real roads go both ways
    #   • Geometry stores the GPS curve of the road for accurate map drawing
    #   • Delegates to _add_or_update_edge to avoid duplicate / expensive edges
    # ─────────────────────────────────────────────────────────────────────────
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

        # Auto-register both endpoints — no need to call add_node separately
        self.add_node(u)
        self.add_node(v)

        # Convert geometry list → immutable tuple so it can live inside a frozen Edge
        geom_tuple = tuple(geometry) if geometry else None

        # Add the directed edge from u → v
        # Check if an edge already exists from u to v; update if new weight is smaller
        self._add_or_update_edge(u, v, weight, road_name, geom_tuple, highway)

        # For a two-way road, also add the reverse edge v → u
        # The geometry is reversed so it traces correctly in the opposite direction
        if bidirectional and u != v:
            rev_geom = tuple(reversed(geometry)) if geometry else None
            self._add_or_update_edge(v, u, weight, road_name, rev_geom, highway)

    # ─────────────────────────────────────────────────────────────────────────
    # _add_or_update_edge — Internal helper (not part of the public API)
    #
    # Ensures we never store duplicate edges between the same two nodes.
    # If a cheaper road between u and v is found later, we replace the
    # old edge — this keeps the graph accurate when parsing real map data
    # where the same road segment can appear multiple times.
    # ─────────────────────────────────────────────────────────────────────────
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

        # Scan existing edges from u to see if one already reaches v
        for idx, edge in enumerate(edges):
            if edge.destination == v:
                # An edge to v already exists — only replace it if this new one is cheaper
                # This implements a "keep the best road" policy
                if weight <= edge.weight:
                    edges[idx] = Edge(
                        destination=v,
                        weight=weight,
                        road_name=road_name,
                        geometry=geometry,
                        highway=highway,
                    )
                return  # Either updated or kept old one — done either way

        # No existing edge to v found — append this as a brand-new connection
        edges.append(
            Edge(
                destination=v,
                weight=weight,
                road_name=road_name,
                geometry=geometry,
                highway=highway,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # COORDINATE & SPATIAL HELPERS
    # These methods support map rendering and geographical search.
    # ─────────────────────────────────────────────────────────────────────────

    def get_node_coords(self, node: str) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) coordinates of a node if available, else None."""
        return self._node_coords.get(node.strip())

    def has_coords(self) -> bool:
        """Return True if node coordinates are present in the graph."""
        # Used to decide whether to show a map or just a text path
        return bool(self._node_coords)

    def find_nearest_node(self, lat: float, lon: float) -> Tuple[str, float]:
        """
        Find the closest node in the graph to the specified (lat, lon).
        Returns a tuple of (nearest_node_id, distance_in_meters).
        Raises ValueError if the graph has no recorded coordinates.
        """
        # Can only search geographically if coordinates have been loaded
        if not self._node_coords:
            raise ValueError("Cannot search nearest node: graph has no geographical coordinates.")

        try:
            # ── Fast path: vectorised NumPy computation ──────────────────────
            # If NumPy is available, we compute all distances simultaneously
            # using array arithmetic — much faster for large real-world maps
            # (OSM city graphs can have hundreds of thousands of nodes).
            import numpy as np
            node_ids = list(self._node_coords.keys())
            coords_arr = np.array([self._node_coords[n] for n in node_ids], dtype=np.float64)

            lats = coords_arr[:, 0]   # Column 0 = all latitudes
            lons = coords_arr[:, 1]   # Column 1 = all longitudes

            # Approximate planar distance in meters using equirectangular projection.
            # cos_lat corrects for the fact that longitude degrees shrink near the poles.
            cos_lat = math.cos(math.radians(lat))
            dlat = (lats - lat) * 111320.0           # 1° latitude ≈ 111,320 m
            dlon = (lons - lon) * (111320.0 * cos_lat)  # Longitude scaling varies by latitude

            # Squared distances avoid a costly sqrt on every element
            dists_sq = dlat * dlat + dlon * dlon
            best_idx = int(np.argmin(dists_sq))       # Index of the closest node
            return node_ids[best_idx], math.sqrt(float(dists_sq[best_idx]))

        except ImportError:
            # ── Slow path: pure-Python fallback ─────────────────────────────
            # NumPy is not installed — iterate manually.
            # Same math, just applied one node at a time.
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

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH QUERY METHODS
    # Standard operations used by pathfinding algorithms and the UI.
    # ─────────────────────────────────────────────────────────────────────────

    def has_node(self, node: str) -> bool:
        """Return True if the node exists in the graph."""
        return node in self._adjacency_list

    def get_nodes(self) -> List[str]:
        """Return a sorted list of all node names in the graph."""
        # Sorted for consistent ordering in dropdowns and printed output
        return sorted(self._adjacency_list.keys())

    def get_neighbors(self, node: str) -> List[Edge]:
        """
        Return the list of outgoing edges for a given node.
        
        Raises KeyError if the node does not exist in the graph.
        """
        # Pathfinding algorithms call this in their inner loop — it must be fast
        if node not in self._adjacency_list:
            raise KeyError(f"Location '{node}' is not in the road network.")
        # Return a copy so callers cannot accidentally mutate the graph's internal state
        return list(self._adjacency_list[node])

    def get_edge(self, u: str, v: str) -> Optional[Edge]:
        """Return the Edge from u to v, or None if no direct connection exists."""
        if u not in self._adjacency_list or v not in self._adjacency_list:
            return None
        # Linear scan through u's neighbor list — acceptable because node degree is small
        for edge in self._adjacency_list[u]:
            if edge.destination == v:
                return edge
        return None

    # ── Properties — quick statistics about the graph ────────────────────────

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._adjacency_list)

    @property
    def edge_count(self) -> int:
        """Return total directed edges in the graph."""
        # Each bidirectional road counts as 2 directed edges (u→v and v→u)
        return sum(len(edges) for edges in self._adjacency_list.values())

    # ─────────────────────────────────────────────────────────────────────────
    # find_close_matches — Typo-tolerant node name suggestion
    #
    # When a user types "Riverdal" instead of "Riverdale", we don't just crash
    # with "node not found". Instead we suggest the closest matching names.
    #
    # Three-stage matching strategy (from most to least strict):
    #   1. Exact match ignoring case         → highest confidence
    #   2. Substring containment             → partial name typed
    #   3. Fuzzy edit-distance (difflib)     → typos / misspellings
    # ─────────────────────────────────────────────────────────────────────────
    def find_close_matches(self, node: str) -> List[str]:
        """
        Find potential matching node names for fuzzy error suggestion.
        Performs case-insensitive matching, substring containment, and edit-distance matching.
        """
        raw = node.strip()
        target = raw.lower()
        all_nodes = list(self._adjacency_list.keys())

        # Stage 1: Exact case-insensitive match (e.g. "grand haven" → "Grand Haven")
        exact_ignore_case = [n for n in all_nodes if n.lower() == target]
        if exact_ignore_case:
            return exact_ignore_case   # Perfect match found — return immediately

        # Stage 2: Substring match (e.g. "river" matches "Riverdale", "Frost River")
        substr_matches = [n for n in all_nodes if target in n.lower() or n.lower() in target]

        # Stage 3: Fuzzy similarity via difflib — handles swapped letters, typos
        # cutoff=0.5 means at least 50% character similarity required
        fuzzy_matches = get_close_matches(raw, all_nodes, n=3, cutoff=0.5)

        # Merge both match lists, preserving order and de-duplicating
        seen: Set[str] = set()
        results: List[str] = []
        for candidate in substr_matches + fuzzy_matches:
            if candidate not in seen:
                seen.add(candidate)
                results.append(candidate)

        return results

    # ── Python dunder / magic methods ────────────────────────────────────────

    def __contains__(self, node: str) -> bool:
        # Enables natural syntax: `if "Grand Haven" in graph:`
        return self.has_node(node)

    def __repr__(self) -> str:
        # Clean debug representation, e.g. Graph(nodes=14, edges=38)
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE ROAD NETWORK — Built-in fictional map for demos and testing
#
# This gives the app something to work with right out of the box, without
# needing an internet connection or an API key.
#
# The network is carefully designed to test all pathfinding scenarios:
#
#   ┌──────────────────────────────────────────────────────┐
#   │  REGION         │  NODES                             │
#   ├──────────────────────────────────────────────────────┤
#   │  Central Hub    │  Grand Haven (capital / port)      │
#   │  West / Coast   │  Silverfall, Port Marina, Bayview  │
#   │  North / Mtn    │  Pinecrest, High Peak, Frostford   │
#   │  East / Forest  │  Oakridge, Riverdale, Mistwood,    │
#   │                 │  Greenfield                        │
#   │  South / Agri   │  Sunset Valley, Ironhold,          │
#   │                 │  Amber Plains                      │
#   │  Disconnected   │  Storm Island, Isle Outpost        │
#   └──────────────────────────────────────────────────────┘
#
# The Storm Island / Isle Outpost subgraph is intentionally unreachable from
# the mainland — useful for testing "no path found" error handling.
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── Coastal Highway Corridor (Route 1) ───────────────────────────────────
    # This is the main north-south arterial road along the western coastline.
    # Fast, flat, and direct — typically the preferred route for coastal cities.
    g.add_edge("Port Marina", "Bayview", 18.5, road_name="Route 1 (Coast Highway)")
    g.add_edge("Bayview", "Silverfall", 22.0, road_name="Route 1 (Coast Highway)")
    g.add_edge("Silverfall", "Grand Haven", 35.0, road_name="Route 1 (Bay Highway)")

    # ── Interstate 10 (East-West Central Expressway) ─────────────────────────
    # The major east-west motorway cutting across the centre of the map.
    # Connects the coastal cities to the inland forest/valley region.
    g.add_edge("Silverfall", "Riverdale", 28.0, road_name="I-10 Expressway")
    g.add_edge("Grand Haven", "Riverdale", 15.0, road_name="I-10 Expressway")
    g.add_edge("Riverdale", "Oakridge", 24.5, road_name="I-10 Expressway")
    g.add_edge("Oakridge", "Mistwood", 31.0, road_name="I-10 East Extension")

    # ── Northern Mountain Region ──────────────────────────────────────────────
    # High altitude, winding roads with steep gradients.
    # Longer distances and slower travel — a scenic but costly detour.
    # Pathfinding algorithms must weigh the mountain shortcut vs coastal bypass.
    g.add_edge("Grand Haven", "Pinecrest", 42.0, road_name="Pine Valley Road")
    g.add_edge("Silverfall", "Pinecrest", 50.0, road_name="Old Forest Pass")
    g.add_edge("Pinecrest", "High Peak", 19.5, road_name="High Peak Trail")
    g.add_edge("High Peak", "Frostford", 27.0, road_name="North Mountain Pass")
    g.add_edge("Frostford", "Riverdale", 55.0, road_name="Frost Gorge Parkway")

    # ── Eastern Woodlands & Plains ────────────────────────────────────────────
    # Quieter regional roads connecting farming communities and forest towns.
    # Short distances between nodes create useful shortcut opportunities.
    g.add_edge("Riverdale", "Greenfield", 18.0, road_name="Green Valley Way")
    g.add_edge("Greenfield", "Oakridge", 12.0, road_name="East Meadow Road")
    g.add_edge("Greenfield", "Amber Plains", 26.0, road_name="Plains Route 5")
    g.add_edge("Oakridge", "Amber Plains", 20.0, road_name="Sunridge Pass")

    # ── Southern Foothills & Mining Belt ─────────────────────────────────────
    # Roads serving the agricultural south and industrial mining towns.
    # Creates alternative south-loop routes between Grand Haven and Amber Plains.
    g.add_edge("Grand Haven", "Sunset Valley", 25.0, road_name="South Coast Highway")
    g.add_edge("Sunset Valley", "Ironhold", 30.0, road_name="Iron Foothills Road")
    g.add_edge("Ironhold", "Amber Plains", 38.0, road_name="Mineral Way")
    g.add_edge("Riverdale", "Sunset Valley", 32.0, road_name="Riverbend Cutoff")

    # ── Disconnected Offshore Region ──────────────────────────────────────────
    # Storm Island and Isle Outpost form their own isolated subgraph.
    # There is NO road connecting them to the mainland — reachable only by ferry.
    # This tests the "no path exists" error path in all search algorithms.
    g.add_edge("Storm Island", "Isle Outpost", 14.0, road_name="Island Perimeter Road")

    return g
