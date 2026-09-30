"""
Main CLI entry point for PathFinder (Stage 1 & Stage 2).

Provides:
- Real-World OpenStreetMap routing (Ranchi, India) with Folium interactive maps.
- Fictional road network routing (Stage 1 backward-compatible).
- Direct command-line arguments and guided interactive prompt sessions.
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional, Tuple

from graph import Graph, create_sample_road_network
from dijkstra import find_shortest_path, PathResult
from display import (
    print_banner,
    print_available_locations,
    print_landmarks,
    print_path_result,
    print_map_saved,
    print_error,
)
from osm_loader import (
    get_ranchi_road_network,
    resolve_location_or_coords,
    RANCHI_LANDMARKS,
    parse_coordinates,
)
from map_view import generate_interactive_map


def resolve_fictional_location(graph: Graph, user_input: str) -> Optional[str]:
    """Resolve user input to a canonical node name in the fictional network."""
    cleaned = user_input.strip()
    if graph.has_node(cleaned):
        return cleaned

    matches = graph.find_close_matches(cleaned)
    if len(matches) == 1:
        return matches[0]

    return None


# Backward-compatibility alias for Stage 1 tests
resolve_location_name = resolve_fictional_location


def parse_landmark_selection(user_input: str) -> str:
    """If user enters an integer 1-10, map it to the corresponding landmark name."""
    stripped = user_input.strip()
    landmark_names = list(RANCHI_LANDMARKS.keys())
    if stripped.isdigit():
        idx = int(stripped)
        if 1 <= idx <= len(landmark_names):
            return landmark_names[idx - 1]
    return user_input


def run_realworld_routing(
    graph: Graph,
    start_query: str,
    dest_query: str,
    output_map: str = "route_map.html",
) -> int:
    """Execute real-world shortest-path finding on OpenStreetMap data."""
    try:
        start_query = parse_landmark_selection(start_query)
        dest_query = parse_landmark_selection(dest_query)

        start_node, start_coords, start_label = resolve_location_or_coords(graph, start_query)
    except ValueError as e:
        print_error(str(e))
        return 1

    try:
        dest_node, dest_coords, dest_label = resolve_location_or_coords(graph, dest_query)
    except ValueError as e:
        print_error(str(e))
        return 1

    # Run Dijkstra's algorithm from scratch
    result = find_shortest_path(graph, start_node, dest_node)

    # Print terminal output
    print_path_result(
        result=result,
        unit="m",
        start_label=start_label,
        dest_label=dest_label,
    )

    if result.found:
        try:
            map_path = generate_interactive_map(
                graph=graph,
                result=result,
                start_coord=start_coords,
                dest_coord=dest_coords,
                start_label=start_label,
                dest_label=dest_label,
                output_path=output_map,
            )
            print_map_saved(map_path)
        except Exception as err:
            print_error(f"Failed to generate interactive map: {err}")

    return 0 if result.found or start_node == dest_node else 2


def run_fictional_routing(
    graph: Graph,
    start_query: str,
    dest_query: str,
) -> int:
    """Execute shortest-path finding on the fictional road network (Stage 1)."""
    resolved_start = resolve_fictional_location(graph, start_query)
    if not resolved_start:
        suggestions = graph.find_close_matches(start_query)
        print_error(f"Invalid starting location: '{start_query}'.", suggestions)
        return 1

    resolved_dest = resolve_fictional_location(graph, dest_query)
    if not resolved_dest:
        suggestions = graph.find_close_matches(dest_query)
        print_error(f"Invalid destination location: '{dest_query}'.", suggestions)
        return 1

    result = find_shortest_path(graph, resolved_start, resolved_dest)
    print_path_result(result, unit="km")

    return 0 if result.found or resolved_start == resolved_dest else 2


def run_interactive_session() -> None:
    """Run interactive mode allowing user to pick network and route."""
    print_banner(stage=2)
    print("Select Road Network:")
    print("  [1] Real-World Map (Ranchi, Jharkhand, India) — OpenStreetMap [Default]")
    print("  [2] Fictional Regional Network (Stage 1)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    is_fictional = choice == "2"

    if is_fictional:
        graph = create_sample_road_network()
        print_banner(stage=1)
        print_available_locations(graph)
        while True:
            try:
                print("Enter locations to calculate the shortest path (or 'q' to quit).\n")
                start_input = input("📍 Starting location    : ").strip()
                if start_input.lower() in ("q", "quit", "exit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                if not start_input:
                    continue

                dest_input = input("🎯 Destination location : ").strip()
                if dest_input.lower() in ("q", "quit", "exit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                if not dest_input:
                    continue

                run_fictional_routing(graph, start_input, dest_input)

                another = input("Find another route? (Y/n): ").strip().lower()
                if another in ("n", "no", "q", "quit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                print()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated by user. Goodbye! 👋\n")
                break
    else:
        print("\nLoading Ranchi OpenStreetMap road network...")
        graph = get_ranchi_road_network()
        print(f"Loaded network with {graph.node_count} intersections and {graph.edge_count} road segments.\n")
        print_landmarks(RANCHI_LANDMARKS)

        while True:
            try:
                print("Enter start and destination (landmark name, index [1-10], or 'lat, lon'):")
                start_input = input("📍 Starting location    : ").strip()
                if start_input.lower() in ("q", "quit", "exit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                if not start_input:
                    continue

                dest_input = input("🎯 Destination location : ").strip()
                if dest_input.lower() in ("q", "quit", "exit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                if not dest_input:
                    continue

                run_realworld_routing(graph, start_input, dest_input, output_map="route_map.html")

                another = input("Find another route? (Y/n): ").strip().lower()
                if another in ("n", "no", "q", "quit"):
                    print("\nThank you for using PathFinder! Safe travels. 👋\n")
                    break
                print()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated by user. Goodbye! 👋\n")
                break


def main() -> int:
    """Parse command-line arguments and run the application."""
    parser = argparse.ArgumentParser(
        prog="pathfinder",
        description="PathFinder: Dijkstra-powered shortest path calculator for road networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-World OpenStreetMap routing (Ranchi):
  python main.py -s "Albert Ekka Chowk" -d "Ranchi Railway Station"
  python main.py --start-coords 23.3699,85.3253 --dest-coords 23.3512,85.3347
  python main.py --list-landmarks
  
  # Stage 1 Fictional Network routing:
  python main.py --fictional -s "Bayview" -d "Frostford"
  python main.py --list
  
  # Interactive mode:
  python main.py
        """,
    )

    parser.add_argument(
        "-s", "--start",
        type=str,
        help="Starting location (landmark name, town name, or 'lat, lon')",
    )
    parser.add_argument(
        "-d", "--dest", "--destination",
        dest="destination",
        type=str,
        help="Destination location (landmark name, town name, or 'lat, lon')",
    )
    parser.add_argument(
        "--start-coords",
        type=str,
        help="Starting GPS coordinates as 'latitude,longitude'",
    )
    parser.add_argument(
        "--dest-coords",
        type=str,
        help="Destination GPS coordinates as 'latitude,longitude'",
    )
    parser.add_argument(
        "-o", "--output-map",
        type=str,
        default="route_map.html",
        help="Output file path for the interactive Folium HTML map (default: route_map.html)",
    )
    parser.add_argument(
        "-f", "--fictional",
        action="store_true",
        help="Force use of the Stage 1 fictional regional network",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available locations in the fictional road network and exit",
    )
    parser.add_argument(
        "-ll", "--list-landmarks",
        action="store_true",
        help="List popular landmarks in Ranchi for real-world routing and exit",
    )
    parser.add_argument(
        "--refresh-osm",
        action="store_true",
        help="Re-download the latest OpenStreetMap data for Ranchi using OSMnx",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive prompt mode",
    )

    args = parser.parse_args()

    # List fictional towns
    if args.list:
        print_banner(stage=1)
        print_available_locations(create_sample_road_network())
        return 0

    # List real-world landmarks
    if args.list_landmarks:
        print_banner(stage=2)
        print_landmarks(RANCHI_LANDMARKS)
        return 0

    # Refresh OSM data if requested
    if args.refresh_osm:
        print("Re-downloading OpenStreetMap road data for Ranchi...")
        get_ranchi_road_network(force_download=True)
        print("Fresh OpenStreetMap network downloaded and cached successfully.")
        return 0

    # Determine start and destination from args
    start_arg = args.start or args.start_coords
    dest_arg = args.destination or args.dest_coords

    # Launch interactive mode if no routing arguments provided
    if args.interactive or (start_arg is None and dest_arg is None):
        run_interactive_session()
        return 0

    # Require both start and destination
    if start_arg is None or dest_arg is None:
        print_error("Both --start (-s) and --dest (-d) must be specified for non-interactive queries.")
        parser.print_usage()
        return 1

    # Check whether to route on fictional or real-world network:
    fictional_graph = create_sample_road_network()
    is_fictional_query = args.fictional or (
        fictional_graph.has_node(start_arg) or fictional_graph.has_node(dest_arg)
    )

    if is_fictional_query:
        return run_fictional_routing(fictional_graph, start_arg, dest_arg)
    else:
        osm_graph = get_ranchi_road_network()
        return run_realworld_routing(
            graph=osm_graph,
            start_query=start_arg,
            dest_query=dest_arg,
            output_map=args.output_map,
        )


if __name__ == "__main__":
    sys.exit(main())
