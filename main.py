"""
Main CLI entry point for PathFinder.

Provides both an interactive prompt mode and a command-line argument interface
to find shortest routes across the road network.
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional

from graph import Graph, create_sample_road_network
from dijkstra import find_shortest_path
from display import (
    print_banner,
    print_available_locations,
    print_path_result,
    print_error,
)


def resolve_location_name(graph: Graph, user_input: str) -> Optional[str]:
    """
    Resolve user input to a canonical node name in the graph.
    
    1. Direct exact match.
    2. Case-insensitive match if exactly one match exists.
    3. Returns None if unresolvable.
    """
    cleaned = user_input.strip()
    if graph.has_node(cleaned):
        return cleaned

    matches = graph.find_close_matches(cleaned)
    if len(matches) == 1:
        return matches[0]

    return None


def run_interactive_session(graph: Graph) -> None:
    """Run an interactive prompt session for the user."""
    print_banner()
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

            resolved_start = resolve_location_name(graph, start_input)
            if not resolved_start:
                suggestions = graph.find_close_matches(start_input)
                print_error(f"Unknown location '{start_input}'.", suggestions)
                continue

            dest_input = input("🎯 Destination location : ").strip()
            if dest_input.lower() in ("q", "quit", "exit"):
                print("\nThank you for using PathFinder! Safe travels. 👋\n")
                break
            if not dest_input:
                continue

            resolved_dest = resolve_location_name(graph, dest_input)
            if not resolved_dest:
                suggestions = graph.find_close_matches(dest_input)
                print_error(f"Unknown location '{dest_input}'.", suggestions)
                continue

            # Calculate and display path
            result = find_shortest_path(graph, resolved_start, resolved_dest)
            print_path_result(result)

            # Option to continue
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
  python main.py
  python main.py --list
  python main.py --start "Pinecrest" --dest "Grand Haven"
  python main.py -s "Bayview" -d "Frostford"
        """,
    )

    parser.add_argument(
        "-s", "--start",
        type=str,
        help="Starting location name",
    )
    parser.add_argument(
        "-d", "--dest", "--destination",
        dest="destination",
        type=str,
        help="Destination location name",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available locations in the network and exit",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive prompt mode",
    )

    args = parser.parse_args()
    graph = create_sample_road_network()

    if args.list:
        print_banner()
        print_available_locations(graph)
        return 0

    # If neither start nor destination is supplied, or interactive requested:
    if args.interactive or (args.start is None and args.destination is None):
        run_interactive_session(graph)
        return 0

    # If one argument is provided without the other:
    if args.start is None or args.destination is None:
        print_error("Both --start (-s) and --dest (-d) must be specified for non-interactive queries.")
        parser.print_usage()
        return 1

    # Validate start location
    resolved_start = resolve_location_name(graph, args.start)
    if not resolved_start:
        suggestions = graph.find_close_matches(args.start)
        print_error(f"Invalid starting location: '{args.start}'.", suggestions)
        return 1

    # Validate destination location
    resolved_dest = resolve_location_name(graph, args.destination)
    if not resolved_dest:
        suggestions = graph.find_close_matches(args.destination)
        print_error(f"Invalid destination location: '{args.destination}'.", suggestions)
        return 1

    # Calculate shortest path
    result = find_shortest_path(graph, resolved_start, resolved_dest)
    print_path_result(result)

    return 0 if result.found or resolved_start == resolved_dest else 2


if __name__ == "__main__":
    sys.exit(main())
