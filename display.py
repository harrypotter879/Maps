"""
Display module for PathFinder.

Provides formatting and terminal rendering utilities for route summaries,
turn-by-turn legs, execution metrics, location listings, and error messages.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

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
    """Format distance to one or two decimal places with appropriate units."""
    if distance == float("inf"):
        return "Infinity (Unreachable)"
    if unit == "m":
        if distance >= 1000:
            return f"{distance / 1000:.2f} km"
        return f"{distance:.0f} m"
    if distance == int(distance):
        return f"{int(distance)} {unit}"
    return f"{distance:.1f} {unit}"


def print_banner(stage: int = 2) -> None:
    """Print the application welcome banner."""
    divider = "=" * 65
    print(f"\n{divider}")
    if stage == 2:
        print("   🌐  PATHFINDER: REAL-WORLD MAP NAVIGATOR (STAGE 2)  🌐")
        print("         OpenStreetMap + Custom Dijkstra Routing Engine")
    else:
        print("   🌐  PATHFINDER: SHORTEST PATH NAVIGATOR (STAGE 1)  🌐")
    print(f"{divider}\n")


def print_landmarks(landmarks: Dict[str, Tuple[float, float]]) -> None:
    """Display predefined key real-world landmarks in a clean table."""
    print("📍 Popular Landmarks in Ranchi, Jharkhand:")
    print("-" * 65)
    for idx, (name, (lat, lon)) in enumerate(landmarks.items(), start=1):
        num_str = f"[{idx}]".rjust(4)
        name_str = f"{name}".ljust(26)
        coord_str = f"({lat:.4f}, {lon:.4f})"
        print(f"  {num_str} {name_str} {coord_str}")
    print("-" * 65)
    print("  💡 You can enter a landmark name, index number [1-10],")
    print("     or direct GPS coordinates as 'lat, lon'.\n")


def print_map_saved(file_path: str) -> None:
    """Display confirmation and access details for the generated interactive map."""
    print(f"🗺️  Interactive Map Saved Successfully:")
    print(f"   📁 File: {file_path}")
    print(f"   🌐 Open this file in your browser to view the interactive map!")
    print()


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


def print_path_result(
    result: PathResult,
    unit: str = "km",
    start_label: Optional[str] = None,
    dest_label: Optional[str] = None,
) -> None:
    """
    Print the complete details of a shortest-path query.
    Includes route sequence, step-by-step turn guidance, total distance,
    and computational performance metrics.
    """
    src_disp = f"{start_label} (Node {result.source})" if start_label and start_label != result.source else result.source
    dst_disp = f"{dest_label} (Node {result.destination})" if dest_label and dest_label != result.destination else result.destination

    divider = "─" * 65
    print(f"\n{divider}")
    print(f"🏁 Route Summary: {src_disp}  ➔  {dst_disp}")
    print(divider)

    if not result.found:
        print("\n❌ NO ROUTE FOUND!")
        print(f"   Destination '{dst_disp}' is not reachable from '{src_disp}'.")
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
    if total_legs <= 10:
        for idx, leg in enumerate(result.legs, start=1):
            road_info = f" via [{leg.road_name}]" if leg.road_name else ""
            dist_str = format_distance(leg.distance, unit)
            print(f"   {idx}. {leg.origin} ➔ {leg.destination}")
            print(f"      Leg Distance: {dist_str}{road_info}")
    else:
        # Group consecutive legs with the same road name for clear readability
        corridors: List[Tuple[str, float, int]] = []
        curr_road = result.legs[0].road_name or "Local Road"
        curr_dist = 0.0
        curr_count = 0
        for leg in result.legs:
            r_name = leg.road_name or "Local Road / Connector"
            if r_name == curr_road:
                curr_dist += leg.distance
                curr_count += 1
            else:
                corridors.append((curr_road, curr_dist, curr_count))
                curr_road = r_name
                curr_dist = leg.distance
                curr_count = 1
        corridors.append((curr_road, curr_dist, curr_count))

        for idx, (road, dist, segs) in enumerate(corridors, start=1):
            dist_str = format_distance(dist, unit)
            seg_info = f"({segs} intersections)" if segs > 1 else ""
            print(f"   {idx}. Follow [{road}] for {dist_str} {seg_info}")

    # Visual path breadcrumbs
    print(f"\n🛣️  Path Sequence ({len(result.path)} intersections):")
    if len(result.path) <= 10:
        print("   " + " ➔ ".join(result.path))
    else:
        first_few = " ➔ ".join(result.path[:3])
        last_few = " ➔ ".join(result.path[-3:])
        print(f"   {first_few} ➔ ... [{len(result.path)-6} intermediate intersections] ➔ {last_few}")

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
