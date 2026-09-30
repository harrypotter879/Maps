"""
Automated test suite for PathFinder.

Uses Python's standard unittest library to thoroughly verify the graph data structure,
Dijkstra's shortest path algorithm, edge cases, display utilities, and CLI resolution.
"""

from __future__ import annotations
import unittest
import io
from contextlib import redirect_stdout

from graph import Graph, Edge, create_sample_road_network
from dijkstra import find_shortest_path, PathResult, RouteLeg
from display import format_time, format_distance, print_path_result, print_available_locations
from main import resolve_location_name


class TestGraph(unittest.TestCase):
    """Tests for the Graph data structure."""

    def setUp(self) -> None:
        self.graph = Graph()

    def test_add_and_has_node(self) -> None:
        self.assertTrue(self.graph.add_node("Alpha"))
        self.assertFalse(self.graph.add_node("Alpha"))  # Duplicate
        self.assertTrue(self.graph.has_node("Alpha"))
        self.assertFalse(self.graph.has_node("Beta"))
        self.assertEqual(self.graph.node_count, 1)

    def test_empty_node_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.graph.add_node("")
        with self.assertRaises(ValueError):
            self.graph.add_node("   ")

    def test_add_bidirectional_edge(self) -> None:
        self.graph.add_edge("Alpha", "Beta", 10.5, bidirectional=True, road_name="Route A")
        self.assertEqual(self.graph.node_count, 2)
        self.assertEqual(self.graph.edge_count, 2)

        alpha_neighbors = self.graph.get_neighbors("Alpha")
        self.assertEqual(len(alpha_neighbors), 1)
        self.assertEqual(alpha_neighbors[0].destination, "Beta")
        self.assertEqual(alpha_neighbors[0].weight, 10.5)
        self.assertEqual(alpha_neighbors[0].road_name, "Route A")

        beta_neighbors = self.graph.get_neighbors("Beta")
        self.assertEqual(len(beta_neighbors), 1)
        self.assertEqual(beta_neighbors[0].destination, "Alpha")

    def test_add_directed_edge(self) -> None:
        self.graph.add_edge("Alpha", "Beta", 7.0, bidirectional=False)
        self.assertEqual(len(self.graph.get_neighbors("Alpha")), 1)
        self.assertEqual(len(self.graph.get_neighbors("Beta")), 0)

    def test_negative_weight_edge_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.graph.add_edge("Alpha", "Beta", -5.0)

    def test_get_edge(self) -> None:
        self.graph.add_edge("Alpha", "Beta", 12.0, bidirectional=True, road_name="Highway 1")
        edge = self.graph.get_edge("Alpha", "Beta")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.weight, 12.0)
        self.assertEqual(edge.road_name, "Highway 1")
        self.assertIsNone(self.graph.get_edge("Alpha", "Gamma"))

    def test_get_neighbors_nonexistent_node_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.graph.get_neighbors("Unknown")

    def test_find_close_matches(self) -> None:
        self.graph.add_node("Grand Haven")
        self.graph.add_node("Silverfall")
        # Case insensitive exact
        self.assertEqual(self.graph.find_close_matches("grand haven"), ["Grand Haven"])
        # Substring
        self.assertEqual(self.graph.find_close_matches("silver"), ["Silverfall"])
        # No match
        self.assertEqual(self.graph.find_close_matches("Atlantis"), [])

    def test_sample_network_integrity(self) -> None:
        sample = create_sample_road_network()
        self.assertGreaterEqual(sample.node_count, 10)
        self.assertTrue(sample.has_node("Grand Haven"))
        self.assertTrue(sample.has_node("Storm Island"))


class TestDijkstra(unittest.TestCase):
    """Tests for Dijkstra's shortest path algorithm."""

    def setUp(self) -> None:
        self.graph = Graph()

    def test_direct_vs_indirect_shorter_path(self) -> None:
        """
        Verify that Dijkstra selects the multi-hop path with smaller cumulative weight
        over a direct but longer edge.
        
        A ----(100)----> C
        |                ^
       (10)             (15)
        v                |
        B ---------------+
        """
        self.graph.add_edge("A", "C", 100.0, bidirectional=False)
        self.graph.add_edge("A", "B", 10.0, bidirectional=False)
        self.graph.add_edge("B", "C", 15.0, bidirectional=False)

        result = find_shortest_path(self.graph, "A", "C")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertEqual(result.total_distance, 25.0)
        self.assertEqual(len(result.legs), 2)
        self.assertEqual(result.legs[0].distance, 10.0)
        self.assertEqual(result.legs[1].distance, 15.0)

    def test_same_start_and_destination(self) -> None:
        self.graph.add_node("Home")
        result = find_shortest_path(self.graph, "Home", "Home")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Home"])
        self.assertEqual(result.total_distance, 0.0)
        self.assertEqual(result.legs, [])
        self.assertGreaterEqual(result.execution_time_sec, 0.0)

    def test_unreachable_destination(self) -> None:
        """Disconnected components."""
        self.graph.add_edge("Mainland A", "Mainland B", 10.0)
        self.graph.add_edge("Island A", "Island B", 5.0)

        result = find_shortest_path(self.graph, "Mainland A", "Island B")
        self.assertFalse(result.found)
        self.assertEqual(result.path, [])
        self.assertEqual(result.total_distance, float("inf"))
        self.assertEqual(result.legs, [])

    def test_nonexistent_locations_raise_key_error(self) -> None:
        self.graph.add_node("ValidNode")
        with self.assertRaises(KeyError):
            find_shortest_path(self.graph, "FakeNode", "ValidNode")
        with self.assertRaises(KeyError):
            find_shortest_path(self.graph, "ValidNode", "FakeNode")

    def test_cyclic_graph(self) -> None:
        """Graph with cycles should terminate correctly without infinite looping."""
        self.graph.add_edge("A", "B", 1.0)
        self.graph.add_edge("B", "C", 2.0)
        self.graph.add_edge("C", "A", 4.0)  # Cycle
        self.graph.add_edge("C", "D", 1.0)

        result = find_shortest_path(self.graph, "A", "D")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["A", "B", "C", "D"])
        self.assertEqual(result.total_distance, 4.0)

    def test_zero_weight_edges(self) -> None:
        """Edges with zero weight are valid non-negative weights."""
        self.graph.add_edge("Gate1", "Gate2", 0.0)
        self.graph.add_edge("Gate2", "Terminal", 5.0)

        result = find_shortest_path(self.graph, "Gate1", "Terminal")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Gate1", "Gate2", "Terminal"])
        self.assertEqual(result.total_distance, 5.0)

    def test_route_on_fictional_network(self) -> None:
        """Verify deterministic shortest routes on the fictional road network."""
        network = create_sample_road_network()

        # Route: Port Marina to Grand Haven
        # Port Marina -> Bayview (18.5) -> Silverfall (22.0) -> Grand Haven (35.0) = 75.5
        result = find_shortest_path(network, "Port Marina", "Grand Haven")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["Port Marina", "Bayview", "Silverfall", "Grand Haven"])
        self.assertEqual(result.total_distance, 75.5)

        # Route between Grand Haven and Frostford
        # Path: Grand Haven -> Pinecrest (42.0) -> High Peak (19.5) -> Frostford (27.0) = 88.5
        # Alternative: Grand Haven -> Riverdale (15.0) -> Frostford (55.0) = 70.0!
        result2 = find_shortest_path(network, "Grand Haven", "Frostford")
        self.assertTrue(result2.found)
        self.assertEqual(result2.total_distance, 70.0)
        self.assertEqual(result2.path, ["Grand Haven", "Riverdale", "Frostford"])

        # Unreachable route: Grand Haven to Storm Island
        result_unreachable = find_shortest_path(network, "Grand Haven", "Storm Island")
        self.assertFalse(result_unreachable.found)
        self.assertEqual(result_unreachable.path, [])

    def test_legs_sum_to_total_distance(self) -> None:
        network = create_sample_road_network()
        result = find_shortest_path(network, "Silverfall", "Amber Plains")
        self.assertTrue(result.found)
        leg_sum = sum(leg.distance for leg in result.legs)
        self.assertAlmostEqual(leg_sum, result.total_distance, places=3)

    def test_negative_weight_in_search_raises(self) -> None:
        # Create graph and bypass Edge.__post_init__ using object.__new__ to simulate corrupt data
        bad_graph = Graph()
        bad_graph.add_node("A")
        bad_graph.add_node("B")
        corrupt_edge = object.__new__(Edge)
        object.__setattr__(corrupt_edge, "destination", "B")
        object.__setattr__(corrupt_edge, "weight", -5.0)
        object.__setattr__(corrupt_edge, "road_name", "Illegal Path")
        bad_graph._adjacency_list["A"].append(corrupt_edge)
        with self.assertRaises(ValueError):
            find_shortest_path(bad_graph, "A", "B")


class TestDisplay(unittest.TestCase):
    """Tests for terminal display formatting functions."""

    def test_format_time(self) -> None:
        self.assertEqual(format_time(0.000045), "45.00 µs")
        self.assertEqual(format_time(0.0034), "3.40 ms")
        self.assertEqual(format_time(1.5), "1.500 s")

    def test_format_distance(self) -> None:
        self.assertEqual(format_distance(15.0), "15 km")
        self.assertEqual(format_distance(15.75), "15.8 km")
        self.assertEqual(format_distance(float("inf")), "Infinity (Unreachable)")

    def test_print_available_locations(self) -> None:
        network = create_sample_road_network()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_available_locations(network)
        output = buffer.getvalue()
        self.assertIn("Available Locations in Road Network:", output)
        self.assertIn("Grand Haven", output)

    def test_print_path_result_output(self) -> None:
        result = PathResult(
            source="A",
            destination="B",
            path=["A", "B"],
            legs=[RouteLeg(origin="A", destination="B", distance=10.0, road_name="Main St")],
            total_distance=10.0,
            execution_time_sec=0.001,
            visited_nodes_count=2,
            found=True,
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_path_result(result)
        output = buffer.getvalue()
        self.assertIn("Route Summary: A  ➔  B", output)
        self.assertIn("Main St", output)
        self.assertIn("10 km", output)

    def test_print_unreachable_path(self) -> None:
        result = PathResult(
            source="A",
            destination="Z",
            path=[],
            legs=[],
            total_distance=float("inf"),
            execution_time_sec=0.0005,
            visited_nodes_count=3,
            found=False,
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_path_result(result)
        output = buffer.getvalue()
        self.assertIn("NO ROUTE FOUND!", output)


class TestCLIHelpers(unittest.TestCase):
    """Tests for CLI location resolution and main execution."""

    def setUp(self) -> None:
        self.graph = create_sample_road_network()

    def test_resolve_exact(self) -> None:
        self.assertEqual(resolve_location_name(self.graph, "Grand Haven"), "Grand Haven")

    def test_resolve_case_insensitive(self) -> None:
        self.assertEqual(resolve_location_name(self.graph, "grand haven"), "Grand Haven")
        self.assertEqual(resolve_location_name(self.graph, "PINECREST"), "Pinecrest")

    def test_resolve_fuzzy_single_match(self) -> None:
        # Typo with single close match resolves cleanly
        self.assertEqual(resolve_location_name(self.graph, "Pinecreest"), "Pinecrest")

    def test_resolve_invalid(self) -> None:
        self.assertIsNone(resolve_location_name(self.graph, "Atlantis City"))


class TestMainCLIIntegration(unittest.TestCase):
    """Integration tests running main() with mocked command-line arguments."""

    def test_cli_list_flag(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "--list"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Grand Haven", buffer.getvalue())
        finally:
            sys.argv = orig_argv

    def test_cli_route_found(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "-s", "Bayview", "-d", "Frostford"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Route Summary: Bayview  ➔  Frostford", buffer.getvalue())
            self.assertIn("Total Distance       : 105 km", buffer.getvalue())
        finally:
            sys.argv = orig_argv

    def test_cli_route_unreachable(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "-s", "Grand Haven", "-d", "Storm Island"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 2)
            self.assertIn("NO ROUTE FOUND!", buffer.getvalue())
        finally:
            sys.argv = orig_argv

    def test_cli_invalid_location(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "-s", "Grand Haven", "-d", "NowhereLand123"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 1)
            self.assertIn("Invalid destination location", buffer.getvalue())
        finally:
            sys.argv = orig_argv

    def test_cli_missing_destination(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "-s", "Grand Haven"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 1)
            self.assertIn("Both --start (-s) and --dest (-d) must be specified", buffer.getvalue())
        finally:
            sys.argv = orig_argv


if __name__ == "__main__":
    unittest.main()
