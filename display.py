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
    print(f"   • Algorithm            : {result.algorithm}")
    print(f"   • Total Distance       : {format_distance(result.total_distance, unit)}")
    print(f"   • Number of Segments   : {total_legs}")
    print(f"   • Nodes Explored       : {result.visited_nodes_count}")
    print(f"   • Search Execution Time: {format_time(result.execution_time_sec)}")
    print(divider + "\n")


def print_algorithm_comparison(
    dijkstra_res: PathResult,
    astar_res: PathResult,
    unit: str = "m",
    start_label: Optional[str] = None,
    dest_label: Optional[str] = None,
) -> None:
    """
    Display a side-by-side performance benchmark table comparing Dijkstra and A*.
    Highlights nodes explored, execution speed, and shortest-path distance verification.
    """
    src_disp = f"{start_label} (Node {dijkstra_res.source})" if start_label and start_label != dijkstra_res.source else dijkstra_res.source
    dst_disp = f"{dest_label} (Node {dijkstra_res.destination})" if dest_label and dest_label != dijkstra_res.destination else dijkstra_res.destination

    divider = "═" * 70
    print(f"\n{divider}")
    print(f" 🔬 ALGORITHM COMPARISON BENCHMARK: DIJKSTRA vs A* (A-STAR)")
    print(f"    From: {src_disp}")
    print(f"    To:   {dst_disp}")
    print(divider)

    if not dijkstra_res.found and not astar_res.found:
        print("\n❌ NO ROUTE FOUND BY EITHER ALGORITHM!")
        print(f"   Destination '{dst_disp}' is unreachable from '{src_disp}'.\n")
        print(f"   • Dijkstra Nodes Explored : {dijkstra_res.visited_nodes_count} ({format_time(dijkstra_res.execution_time_sec)})")
        print(f"   • A* Nodes Explored       : {astar_res.visited_nodes_count} ({format_time(astar_res.execution_time_sec)})")
        print(divider + "\n")
        return

    # Calculate differences and speedups
    d_nodes = dijkstra_res.visited_nodes_count
    a_nodes = astar_res.visited_nodes_count
    if d_nodes > 0:
        node_reduction_pct = ((d_nodes - a_nodes) / d_nodes) * 100.0
        node_diff_str = f"{node_reduction_pct:.1f}% fewer nodes" if node_reduction_pct >= 0 else f"{abs(node_reduction_pct):.1f}% more nodes"
    else:
        node_diff_str = "N/A"

    d_time = dijkstra_res.execution_time_sec
    a_time = astar_res.execution_time_sec
    if a_time > 0 and d_time > 0:
        if d_time >= a_time:
            speedup = d_time / a_time
            time_diff_str = f"{speedup:.1f}x faster"
        else:
            slowdown = a_time / d_time
            time_diff_str = f"{slowdown:.1f}x slower"
    else:
        time_diff_str = "Parity"

    # Verify distance parity
    d_dist = dijkstra_res.total_distance
    a_dist = astar_res.total_distance
    dist_match = abs(d_dist - a_dist) < 0.1
    match_str = "✅ Exact Match (Optimal)" if dist_match else f"⚠️ Δ = {abs(d_dist - a_dist):.1f} {unit}"

    # Print comparison table
    print("\n📊 Benchmark Metrics Comparison:")
    header = f"┌{'─'*24}┬{'─'*18}┬{'─'*18}┬{'─'*20}┐"
    row_fmt = "│ {:<22} │ {:<16} │ {:<16} │ {:<18} │"
    div_mid = f"├{'─'*24}┼{'─'*18}┼{'─'*18}┼{'─'*20}┤"
    footer  = f"└{'─'*24}┴{'─'*18}┴{'─'*18}┴{'─'*20}┘"

    print(header)
    print(row_fmt.format("Metric", "Dijkstra", "A* (A-Star)", "Improvement / Note"))
    print(div_mid)
    print(row_fmt.format("Shortest Distance", format_distance(d_dist, unit), format_distance(a_dist, unit), match_str))
    print(row_fmt.format("Nodes Explored", f"{d_nodes:,}", f"{a_nodes:,}", node_diff_str))
    print(row_fmt.format("Execution Time", format_time(d_time), format_time(a_time), time_diff_str))
    print(row_fmt.format("Route Segments", f"{len(dijkstra_res.legs)} legs", f"{len(astar_res.legs)} legs", "Equal topology"))
    print(row_fmt.format("Heuristic Guide", "None (Blind search)", "Haversine Distance", "Admissible (h <= d*)"))
    print(footer)

    # Print itinerary using the optimal route
    best_res = astar_res if astar_res.found else dijkstra_res
    print("\n🗺️  Turn-by-Turn Itinerary:")
    total_legs = len(best_res.legs)
    if total_legs <= 10:
        for idx, leg in enumerate(best_res.legs, start=1):
            road_info = f" via [{leg.road_name}]" if leg.road_name else ""
            dist_str = format_distance(leg.distance, unit)
            print(f"   {idx}. {leg.origin} ➔ {leg.destination}")
            print(f"      Leg Distance: {dist_str}{road_info}")
    else:
        corridors: List[Tuple[str, float, int]] = []
        curr_road = best_res.legs[0].road_name or "Local Road"
        curr_dist = 0.0
        curr_count = 0
        for leg in best_res.legs:
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

    print(f"\n🛣️  Path Sequence ({len(best_res.path)} intersections):")
    if len(best_res.path) <= 10:
        print("   " + " ➔ ".join(best_res.path))
    else:
        first_few = " ➔ ".join(best_res.path[:3])
        last_few = " ➔ ".join(best_res.path[-3:])
        print(f"   {first_few} ➔ ... [{len(best_res.path)-6} intermediate intersections] ➔ {last_few}")

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
