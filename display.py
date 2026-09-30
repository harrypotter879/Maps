"""
Display module for PathFinder.

Provides formatting and terminal rendering utilities for route summaries,
turn-by-turn legs, execution metrics, location listings, and error messages.
"""

from __future__ import annotations
from typing import List, Optional

from graph import Graph
from dijkstra import PathResult


def format_time(seconds: float) -> str:
    """Format execution time into readable units (µs, ms, s)."""
    if seconds < 0.001:
        microseconds = seconds * 1_000_000
        return f"{microseconds:.2f} µs"
    if seconds < 1.0:
        milliseconds = seconds * 1000
        return f"{milliseconds:.2f} ms"
    return f"{seconds:.3f} s"


def format_distance(distance: float, unit: str = "km") -> str:
    """Format distance to one or two decimal places with units."""
    if distance == float("inf"):
        return "Infinity (Unreachable)"
    if distance == int(distance):
        return f"{int(distance)} {unit}"
    return f"{distance:.1f} {unit}"


def print_banner() -> None:
    """Print the application welcome banner."""
    divider = "=" * 62
    print(f"\n{divider}")
    print("   🌐  PATHFINDER: SHORTEST PATH NAVIGATOR (STAGE 1)  🌐")
    print(f"{divider}\n")


def print_available_locations(graph: Graph) -> None:
    """Display all available locations in the network in a clean columnar format."""
    nodes = graph.get_nodes()
    print("📍 Available Locations in Road Network:")
    print("-" * 50)
    
    # 2-column layout for clean terminal rendering
    col_width = 24
    for i in range(0, len(nodes), 2):
        col1 = f"  • {nodes[i]}".ljust(col_width)
        col2 = f"  • {nodes[i+1]}" if i + 1 < len(nodes) else ""
        print(f"{col1}{col2}")
    print("-" * 50)
    print(f"Total locations: {len(nodes)}\n")


def print_path_result(result: PathResult, unit: str = "km") -> None:
    """
    Print the complete details of a shortest-path query.
    Includes route sequence, step-by-step turn guidance, total distance,
    and computational performance metrics.
    """
    divider = "─" * 62
    print(f"\n{divider}")
    print(f"🏁 Route Summary: {result.source}  ➔  {result.destination}")
    print(divider)

    if not result.found:
        print("\n❌ NO ROUTE FOUND!")
        print(f"   Destination '{result.destination}' is not reachable from '{result.source}'.")
        print("   (These locations reside in separate, disconnected network regions.)\n")
        print(f"⏱️  Search Execution Time : {format_time(result.execution_time_sec)}")
        print(f"🔍 Nodes Explored         : {result.visited_nodes_count}")
        print(divider)
        return

    if result.source == result.destination:
        print("\n📌 Start and destination locations are identical.")
        print(f"   Total Travel Distance : 0 {unit}")
        print(f"⏱️  Search Execution Time : {format_time(result.execution_time_sec)}")
        print(divider)
        return

    # Turn-by-turn breakdown
    print("\n🗺️  Turn-by-Turn Itinerary:")
    total_legs = len(result.legs)
    for idx, leg in enumerate(result.legs, start=1):
        road_info = f" via [{leg.road_name}]" if leg.road_name else ""
        dist_str = format_distance(leg.distance, unit)
        print(f"   {idx}. {leg.origin} ➔ {leg.destination}")
        print(f"      Leg Distance: {dist_str}{road_info}")

    # Visual path breadcrumbs
    print(f"\n🛣️  Full Path Sequence:")
    print("   " + " ➔ ".join(result.path))

    # Metrics section
    print(f"\n📊 Trip & Performance Metrics:")
    print(f"   • Total Distance       : {format_distance(result.total_distance, unit)}")
    print(f"   • Number of Segments   : {total_legs}")
    print(f"   • Nodes Explored       : {result.visited_nodes_count}")
    print(f"   • Search Execution Time: {format_time(result.execution_time_sec)}")
    print(divider + "\n")


def print_error(message: str, suggestions: Optional[List[str]] = None) -> None:
    """Display an error message and any fuzzy suggestions for corrections."""
    print(f"\n❌ Error: {message}")
    if suggestions:
        if len(suggestions) == 1:
            print(f"   💡 Did you mean: '{suggestions[0]}'?")
        else:
            options = ", ".join(f"'{s}'" for s in suggestions)
            print(f"   💡 Did you mean one of: {options}?")
    print()
