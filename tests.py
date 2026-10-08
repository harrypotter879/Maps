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


class TestOSMLoader(unittest.TestCase):
    """Tests for OpenStreetMap road network loader and location resolution (Stage 2)."""

    def test_parse_coordinates(self) -> None:
        from osm_loader import parse_coordinates
        # Standard format
        coords = parse_coordinates("23.3699, 85.3253")
        self.assertIsNotNone(coords)
        self.assertAlmostEqual(coords[0], 23.3699)
        self.assertAlmostEqual(coords[1], 85.3253)

        # Parenthesized format
        coords_paren = parse_coordinates("(23.3699, 85.3253)")
        self.assertEqual(coords, coords_paren)

        # Negative coordinates
        coords_neg = parse_coordinates("-33.8688, 151.2093")
        self.assertIsNotNone(coords_neg)
        self.assertAlmostEqual(coords_neg[0], -33.8688)

        # Invalid formats
        self.assertIsNone(parse_coordinates("Not a coordinate"))
        self.assertIsNone(parse_coordinates("100.0, 85.0"))  # Latitude > 90

    def test_load_cached_network_and_find_nearest(self) -> None:
        from osm_loader import get_ranchi_road_network
        graph = get_ranchi_road_network()
        self.assertGreater(graph.node_count, 1000)
        self.assertGreater(graph.edge_count, 2000)
        self.assertTrue(graph.has_coords())

        # Test finding nearest node to Albert Ekka Chowk (23.3699, 85.3253)
        node_id, dist_m = graph.find_nearest_node(23.3699, 85.3253)
        self.assertTrue(graph.has_node(node_id))
        self.assertLess(dist_m, 100.0)  # Intersection is within 100 meters

    def test_resolve_location_or_coords(self) -> None:
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        graph = get_ranchi_road_network()

        # Exact landmark name
        node_id, coords, label = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        self.assertEqual(label, "Albert Ekka Chowk")
        self.assertTrue(graph.has_node(node_id))

        # Case-insensitive / partial landmark name
        node_id2, coords2, label2 = resolve_location_or_coords(graph, "railway station")
        self.assertEqual(label2, "Ranchi Railway Station")
        self.assertTrue(graph.has_node(node_id2))

        # Direct GPS coordinates
        node_id3, coords3, label3 = resolve_location_or_coords(graph, "23.3699, 85.3253")
        self.assertTrue(graph.has_node(node_id3))
        self.assertIn("23.3699", label3)

        # Invalid location raises ValueError
        with self.assertRaises(ValueError):
            resolve_location_or_coords(graph, "Unmapped Atlantis Outpost")


class TestMapView(unittest.TestCase):
    """Tests for interactive Folium HTML map generation (Stage 2)."""

    def test_generate_interactive_map(self) -> None:
        import os
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from dijkstra import find_shortest_path
        from map_view import generate_interactive_map

        graph = get_ranchi_road_network()
        start_node, start_coords, start_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        dest_node, dest_coords, dest_name = resolve_location_or_coords(graph, "Ranchi Railway Station")

        result = find_shortest_path(graph, start_node, dest_node)
        self.assertTrue(result.found)

        test_map_path = "test_map_artifact.html"
        try:
            output = generate_interactive_map(
                graph=graph,
                result=result,
                start_coord=start_coords,
                dest_coord=dest_coords,
                start_label=start_name,
                dest_label=dest_name,
                output_path=test_map_path,
            )
            self.assertTrue(os.path.exists(output))
            self.assertGreater(os.path.getsize(output), 1000)

            with open(output, "r", encoding="utf-8") as f:
                html_content = f.read()
            self.assertIn("leaflet", html_content.lower())
            self.assertIn("Albert Ekka Chowk", html_content)
            self.assertIn("Ranchi Railway Station", html_content)
        finally:
            if os.path.exists(test_map_path):
                os.remove(test_map_path)


class TestStage2CLI(unittest.TestCase):
    """Integration tests for Stage 2 CLI commands."""

    def test_cli_list_landmarks(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "--list-landmarks"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Albert Ekka Chowk", buffer.getvalue())
            self.assertIn("Ranchi Railway Station", buffer.getvalue())
        finally:
            sys.argv = orig_argv

    def test_cli_realworld_landmark_routing(self) -> None:
        import sys, os
        from main import main
        orig_argv = sys.argv
        test_out = "test_cli_route_map.html"
        sys.argv = ["main.py", "-s", "Albert Ekka Chowk", "-d", "Ranchi Railway Station", "-o", test_out]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Route Summary: Albert Ekka Chowk", buffer.getvalue())
            self.assertIn("Ranchi Railway Station", buffer.getvalue())
            self.assertIn("Interactive Map Saved Successfully", buffer.getvalue())
            self.assertTrue(os.path.exists(test_out))
        finally:
            sys.argv = orig_argv
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_cli_coordinate_routing(self) -> None:
        import sys, os
        from main import main
        orig_argv = sys.argv
        test_out = "test_coord_map.html"
        sys.argv = [
            "main.py",
            "--start-coords", "23.3699,85.3253",
            "--dest-coords", "23.3512,85.3347",
            "-o", test_out,
        ]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Interactive Map Saved Successfully", buffer.getvalue())
            self.assertTrue(os.path.exists(test_out))
        finally:
            sys.argv = orig_argv
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_cli_invalid_landmark_error(self) -> None:
        import sys
        from main import main
        orig_argv = sys.argv
        sys.argv = ["main.py", "-s", "CompletelyBogusPlaceXYZ", "-d", "Ranchi Railway Station"]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main()
            self.assertEqual(code, 1)
            self.assertIn("Unknown location 'CompletelyBogusPlaceXYZ'", buffer.getvalue())
        finally:
            sys.argv = orig_argv


class TestAStar(unittest.TestCase):
    """Tests for the A* pathfinding algorithm implementation from scratch."""

    def test_haversine_distance(self) -> None:
        from astar import haversine_distance
        # Same point
        c = (23.3699, 85.3253)
        self.assertAlmostEqual(haversine_distance(c, c), 0.0, places=3)

        # Symmetry and accuracy
        c2 = (23.3512, 85.3347)
        d1 = haversine_distance(c, c2)
        d2 = haversine_distance(c2, c)
        self.assertAlmostEqual(d1, d2, places=3)
        self.assertGreater(d1, 2000.0)
        self.assertLess(d1, 2500.0)

    def test_admissibility_and_consistency(self) -> None:
        """Verify that straight-line Haversine heuristic never overestimates road lengths."""
        from osm_loader import get_ranchi_road_network
        from astar import haversine_distance

        graph = get_ranchi_road_network()
        sampled = 0
        for u in graph.get_nodes()[:50]:
            u_coord = graph.get_node_coords(u)
            if not u_coord:
                continue
            for edge in graph.get_neighbors(u):
                v_coord = graph.get_node_coords(edge.destination)
                if not v_coord:
                    continue
                h_dist = haversine_distance(u_coord, v_coord)
                # Straight-line distance must be <= road weight (with float precision tolerance)
                self.assertLessEqual(h_dist, edge.weight + 0.5)
                sampled += 1
        self.assertGreater(sampled, 20)

    def test_astar_on_fictional_network(self) -> None:
        """A* without coordinates degrades gracefully to optimal Dijkstra."""
        from astar import find_shortest_path_astar
        from dijkstra import find_shortest_path

        graph = create_sample_road_network()
        d_res = find_shortest_path(graph, "Bayview", "Frostford")
        a_res = find_shortest_path_astar(graph, "Bayview", "Frostford")

        self.assertTrue(a_res.found)
        self.assertEqual(a_res.algorithm, "A*")
        self.assertAlmostEqual(a_res.total_distance, d_res.total_distance, places=3)
        self.assertEqual(a_res.path, d_res.path)

    def test_astar_direct_vs_indirect_shorter_path(self) -> None:
        from astar import find_shortest_path_astar
        g = Graph()
        g.add_node("A", lat=0.0, lon=0.0)
        g.add_node("B", lat=0.0, lon=0.0001)
        g.add_node("C", lat=0.0, lon=0.0002)

        g.add_edge("A", "C", 100.0, bidirectional=False)
        g.add_edge("A", "B", 10.0, bidirectional=False)
        g.add_edge("B", "C", 15.0, bidirectional=False)

        res = find_shortest_path_astar(g, "A", "C")
        self.assertTrue(res.found)
        self.assertEqual(res.path, ["A", "B", "C"])
        self.assertEqual(res.total_distance, 25.0)

    def test_astar_same_start_and_dest(self) -> None:
        from astar import find_shortest_path_astar
        g = Graph()
        g.add_node("Home", lat=23.0, lon=85.0)
        res = find_shortest_path_astar(g, "Home", "Home")
        self.assertTrue(res.found)
        self.assertEqual(res.total_distance, 0.0)
        self.assertEqual(res.path, ["Home"])

    def test_astar_unreachable_destination(self) -> None:
        from astar import find_shortest_path_astar
        g = Graph()
        g.add_edge("Island1", "Island2", 10.0)
        g.add_edge("Main1", "Main2", 20.0)
        res = find_shortest_path_astar(g, "Island1", "Main2")
        self.assertFalse(res.found)
        self.assertEqual(res.total_distance, float("inf"))

    def test_astar_key_error(self) -> None:
        from astar import find_shortest_path_astar
        g = Graph()
        g.add_node("Real")
        with self.assertRaises(KeyError):
            find_shortest_path_astar(g, "Fake", "Real")

    def test_astar_negative_weight_raises(self) -> None:
        from astar import find_shortest_path_astar
        g = Graph()
        g.add_node("A")
        g.add_node("B")
        corrupt_edge = object.__new__(Edge)
        object.__setattr__(corrupt_edge, "destination", "B")
        object.__setattr__(corrupt_edge, "weight", -10.0)
        object.__setattr__(corrupt_edge, "road_name", "Invalid")
        object.__setattr__(corrupt_edge, "geometry", None)
        g._adjacency_list["A"].append(corrupt_edge)
        with self.assertRaises(ValueError):
            find_shortest_path_astar(g, "A", "B")

    def test_astar_parity_with_dijkstra_on_real_world(self) -> None:
        """Verify identical distance and reduced node exploration on real-world networks."""
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from dijkstra import find_shortest_path
        from astar import find_shortest_path_astar

        graph = get_ranchi_road_network()
        pairs = [
            ("Albert Ekka Chowk", "Ranchi Railway Station"),
            ("Albert Ekka Chowk", "Morabadi Ground"),
            ("Sujata Chowk", "Tagore Hill"),
        ]

        for start_name, dest_name in pairs:
            u, _, _ = resolve_location_or_coords(graph, start_name)
            v, _, _ = resolve_location_or_coords(graph, dest_name)

            d_res = find_shortest_path(graph, u, v)
            a_res = find_shortest_path_astar(graph, u, v)

            self.assertTrue(d_res.found)
            self.assertTrue(a_res.found)
            # Shortest distances must match within 0.1 meter
            self.assertAlmostEqual(d_res.total_distance, a_res.total_distance, places=1)
            # A* must explore fewer or equal nodes
            self.assertLessEqual(a_res.visited_nodes_count, d_res.visited_nodes_count)


class TestAlgorithmComparison(unittest.TestCase):
    """Tests for algorithm comparison table rendering and comparison map generation."""

    def test_print_algorithm_comparison_output(self) -> None:
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from dijkstra import find_shortest_path
        from astar import find_shortest_path_astar
        from display import print_algorithm_comparison

        graph = get_ranchi_road_network()
        u, _, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, _, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")

        d_res = find_shortest_path(graph, u, v)
        a_res = find_shortest_path_astar(graph, u, v)

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_algorithm_comparison(d_res, a_res, unit="m", start_label=u_name, dest_label=v_name)
        output = buf.getvalue()

        self.assertIn("ALGORITHM COMPARISON BENCHMARK", output)
        self.assertIn("Dijkstra", output)
        self.assertIn("A* (A-Star)", output)
        self.assertIn("Nodes Explored", output)
        self.assertIn("Execution Time", output)
        self.assertIn("Exact Match", output)

    def test_comparison_map_generation(self) -> None:
        import os
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from dijkstra import find_shortest_path
        from astar import find_shortest_path_astar
        from map_view import generate_interactive_map

        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")

        d_res = find_shortest_path(graph, u, v)
        a_res = find_shortest_path_astar(graph, u, v)

        test_out = "test_comparison_map.html"
        try:
            out_path = generate_interactive_map(
                graph=graph,
                result=a_res,
                start_coord=u_coords,
                dest_coord=v_coords,
                start_label=u_name,
                dest_label=v_name,
                output_path=test_out,
                comparison_result=d_res,
            )
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Algorithm Comparison", content)
            self.assertIn("Dijkstra", content)
            self.assertIn("A*", content)
        finally:
            if os.path.exists(test_out):
                os.remove(test_out)


class TestStage3CLI(unittest.TestCase):
    """Integration tests for Stage 3 CLI algorithm selection and comparison commands."""

    def test_cli_algorithm_astar(self) -> None:
        import sys, os
        from main import main
        orig_argv = sys.argv
        test_out = "test_astar_map.html"
        sys.argv = [
            "main.py",
            "-s", "Albert Ekka Chowk",
            "-d", "Ranchi Railway Station",
            "-a", "astar",
            "-o", test_out,
        ]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Algorithm            : A*", buf.getvalue())
            self.assertTrue(os.path.exists(test_out))
        finally:
            sys.argv = orig_argv
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_cli_algorithm_dijkstra(self) -> None:
        import sys, os
        from main import main
        orig_argv = sys.argv
        test_out = "test_dijkstra_map.html"
        sys.argv = [
            "main.py",
            "-s", "Albert Ekka Chowk",
            "-d", "Ranchi Railway Station",
            "-a", "dijkstra",
            "-o", test_out,
        ]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("Algorithm            : Dijkstra", buf.getvalue())
            self.assertTrue(os.path.exists(test_out))
        finally:
            sys.argv = orig_argv
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_cli_compare_flag(self) -> None:
        import sys, os
        from main import main
        orig_argv = sys.argv
        test_out = "test_compare_map.html"
        sys.argv = [
            "main.py",
            "-s", "Albert Ekka Chowk",
            "-d", "Ranchi Railway Station",
            "--compare",
            "-o", test_out,
        ]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main()
            self.assertEqual(code, 0)
            self.assertIn("ALGORITHM COMPARISON BENCHMARK", buf.getvalue())
            self.assertIn("Dijkstra", buf.getvalue())
            self.assertIn("A* (A-Star)", buf.getvalue())
            self.assertTrue(os.path.exists(test_out))
        finally:
            sys.argv = orig_argv
            if os.path.exists(test_out):
                os.remove(test_out)


class TestGeocoder(unittest.TestCase):
    """Tests for Geocoding and reverse geocoding with local landmark fallback (Stage 4)."""

    def test_search_local_landmarks(self) -> None:
        from geocoder import search_local_landmarks
        res = search_local_landmarks("Albert Ekka")
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0].name, "Albert Ekka Chowk")
        self.assertAlmostEqual(res[0].lat, 23.3699, places=3)
        self.assertAlmostEqual(res[0].lon, 85.3253, places=3)

    def test_search_direct_coordinates(self) -> None:
        from geocoder import search_locations
        res = search_locations("23.3699, 85.3253")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].source, "coordinate")
        self.assertAlmostEqual(res[0].lat, 23.3699, places=4)
        self.assertAlmostEqual(res[0].lon, 85.3253, places=4)

    def test_search_empty_query(self) -> None:
        from geocoder import search_locations
        self.assertEqual(search_locations(""), [])
        self.assertEqual(search_locations("   "), [])

    def test_reverse_geocode_landmark_proximity(self) -> None:
        from geocoder import reverse_geocode
        # Exact Albert Ekka Chowk coordinate
        label = reverse_geocode(23.3699, 85.3253)
        self.assertEqual(label, "Albert Ekka Chowk")

        # Reverse geocoding any valid coordinate returns a non-empty string label
        label_out = reverse_geocode(10.1234, 20.5678)
        self.assertIsInstance(label_out, str)
        self.assertGreater(len(label_out), 0)

    def test_search_caching(self) -> None:
        from geocoder import search_locations
        # First query
        r1 = search_locations("Nucleus Mall")
        # Second query should retrieve from in-memory cache
        r2 = search_locations("Nucleus Mall")
        self.assertEqual(r1, r2)

    def test_ranchi_university_coordinates_accurate(self) -> None:
        """Verify Ranchi University landmark coordinates are pinpointed to the Kutchery campus."""
        from osm_loader import RANCHI_LANDMARKS
        self.assertIn("Ranchi University", RANCHI_LANDMARKS)
        lat, lon = RANCHI_LANDMARKS["Ranchi University"]
        self.assertAlmostEqual(lat, 23.3718, places=3)
        self.assertAlmostEqual(lon, 85.3243, places=3)

    def test_expanded_ranchi_landmarks_coverage(self) -> None:
        """Verify wide geographic coverage across Ranchi (over 40 landmarks)."""
        from osm_loader import RANCHI_LANDMARKS
        self.assertGreaterEqual(len(RANCHI_LANDMARKS), 40)
        expected_places = [
            "Birsa Munda Airport",
            "Hatia Railway Station",
            "BIT Mesra",
            "RIMS Hospital",
            "Sadar Hospital",
            "Mall of Ranchi",
            "JSCA International Stadium",
            "Jharkhand High Court",
            "Lalpur Chowk",
            "Argora Chowk",
            "Morabadi Ground",
        ]
        for place in expected_places:
            self.assertIn(place, RANCHI_LANDMARKS)

    def test_format_display_label(self) -> None:
        """Verify informative label formatting for search suggestions."""
        from geocoder import GeocodedLocation
        nom = GeocodedLocation(
            name="Ranchi University",
            display_name="Ranchi University, MDR011, Chadri, Ranchi, Kanke, Ranchi, Jharkhand, 834001, India",
            lat=23.3718,
            lon=85.3243,
            source="nominatim",
        )
        self.assertIn("Ranchi University", nom.format_display_label())
        self.assertIn("OSM Nominatim", nom.format_display_label())
        self.assertIn("Chadri", nom.format_display_label())

        lm = GeocodedLocation(
            name="Albert Ekka Chowk",
            display_name="Albert Ekka Chowk, Ranchi, Jharkhand, India",
            lat=23.3699,
            lon=85.3253,
            source="landmark",
        )
        self.assertIn("Albert Ekka Chowk", lm.format_display_label())
        self.assertIn("Landmark", lm.format_display_label())

        coord = GeocodedLocation(
            name="GPS (23.3699, 85.3253)",
            display_name="Coordinates: Latitude 23.36990, Longitude 85.32530",
            lat=23.3699,
            lon=85.3253,
            source="coordinate",
        )
        self.assertIn("GPS", coord.format_display_label())

    def test_nominatim_prioritized_in_search(self) -> None:
        """Verify that OpenStreetMap Nominatim results appear first in search results."""
        from geocoder import search_locations
        results = search_locations("Ranchi University", limit=5)
        self.assertGreaterEqual(len(results), 1)
        top = results[0]
        # Top result must be Nominatim
        self.assertEqual(top.source, "nominatim")
        # And coordinates must match the true Kutchery / Chadri campus
        self.assertAlmostEqual(top.lat, 23.3718, delta=0.01)
        self.assertAlmostEqual(top.lon, 85.3243, delta=0.01)

    def test_verified_places_coordinates_accuracy(self) -> None:
        """Verify pinpoint coordinates for Sadar Hospital, ITI Bus Stand, and BIT Mesra."""
        from osm_loader import RANCHI_LANDMARKS
        # Sadar Hospital is on Purulia Road, Konka, between Albert Ekka and St. Xavier's
        self.assertIn("Sadar Hospital", RANCHI_LANDMARKS)
        s_lat, s_lon = RANCHI_LANDMARKS["Sadar Hospital"]
        self.assertAlmostEqual(s_lat, 23.36909, places=3)
        self.assertAlmostEqual(s_lon, 85.32679, places=3)

        # ITI Bus Stand is at Kaju Bagan Road / Ratu Road, not in Banhaura
        self.assertIn("ITI Bus Stand", RANCHI_LANDMARKS)
        i_lat, i_lon = RANCHI_LANDMARKS["ITI Bus Stand"]
        self.assertAlmostEqual(i_lat, 23.37615, places=3)
        self.assertAlmostEqual(i_lon, 85.28322, places=3)

        # BIT Mesra campus
        self.assertIn("BIT Mesra", RANCHI_LANDMARKS)
        b_lat, b_lon = RANCHI_LANDMARKS["BIT Mesra"]
        self.assertAlmostEqual(b_lat, 23.41757, delta=0.01)
        self.assertAlmostEqual(b_lon, 85.43941, delta=0.01)

    def test_full_ranchi_network_covers_bit_mesra(self) -> None:
        """Verify cached Greater Ranchi network extends to BIT Mesra with accurate snapping and routing."""
        from osm_loader import get_ranchi_road_network, RANCHI_LANDMARKS
        from astar import find_shortest_path_astar

        graph = get_ranchi_road_network()
        # Verify BIT Mesra snaps to road network within 100 meters
        bit_lat, bit_lon = RANCHI_LANDMARKS["BIT Mesra"]
        bit_node, bit_snap_dist = graph.find_nearest_node(bit_lat, bit_lon)
        self.assertLess(bit_snap_dist, 100.0)

        # Verify Kairali School snaps within 150 meters
        k_lat, k_lon = RANCHI_LANDMARKS["Kairali School"]
        k_node, k_snap_dist = graph.find_nearest_node(k_lat, k_lon)
        self.assertLess(k_snap_dist, 150.0)

        # Verify end-to-end routing completes successfully
        res = find_shortest_path_astar(graph, k_node, bit_node)
        self.assertTrue(res.found)
        self.assertGreater(res.total_distance, 18000.0)  # > 18 km route
        self.assertGreater(len(res.legs), 50)


class TestStage4UI(unittest.TestCase):
    """Tests for Stage 4 Streamlit UI helpers and map builders."""

    def test_build_empty_map(self) -> None:
        from map_view import build_empty_map
        m = build_empty_map(
            center=(23.3699, 85.3253),
            zoom=14,
            start_point=(23.3699, 85.3253, "Albert Ekka"),
            dest_point=(23.3512, 85.3347, "Railway Station"),
        )
        self.assertIsNotNone(m)
        html_str = m._repr_html_()
        self.assertIn("leaflet", html_str.lower())
        self.assertIn("Albert Ekka", html_str)
        self.assertIn("Railway Station", html_str)

    def test_build_folium_map_returns_instance(self) -> None:
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from astar import find_shortest_path_astar
        from map_view import build_folium_map

        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")

        res = find_shortest_path_astar(graph, u, v)
        self.assertTrue(res.found)

        m = build_folium_map(
            graph=graph,
            result=res,
            start_coord=u_coords,
            dest_coord=v_coords,
            start_label=u_name,
            dest_label=v_name,
        )
        self.assertIsNotNone(m)
        html_str = m._repr_html_()
        self.assertIn("Albert Ekka Chowk", html_str)
        self.assertIn("Ranchi Railway Station", html_str)
        self.assertIn("PathFinder Route Navigator", html_str)

    def test_map_with_searched_location(self) -> None:
        """Verify that searched location appears with dedicated search pin and popup."""
        from map_view import build_empty_map, build_folium_map
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from astar import find_shortest_path_astar

        searched = (23.3725, 85.3315, "Nucleus Mall")
        empty_map = build_empty_map(searched_point=searched)
        empty_html = empty_map._repr_html_()
        self.assertIn("Nucleus Mall", empty_html)
        self.assertIn("Searched Place", empty_html)

        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")
        res = find_shortest_path_astar(graph, u, v)

        route_map = build_folium_map(
            graph=graph,
            result=res,
            start_coord=u_coords,
            dest_coord=v_coords,
            start_label=u_name,
            dest_label=v_name,
            searched_point=searched,
        )
        route_html = route_map._repr_html_()
        self.assertIn("Nucleus Mall", route_html)
        self.assertIn("Searched Place", route_html)

    def test_estimate_travel_time(self) -> None:
        """Verify travel time estimation calculation."""
        from app import estimate_travel_time
        # 3000 meters at 30 km/h is 6 minutes
        t_drive = estimate_travel_time(3000.0, speed_kmh=30.0)
        self.assertIn("6 min", t_drive)

        # 3000 meters at 4.5 km/h is 40 minutes
        t_walk = estimate_travel_time(3000.0, speed_kmh=4.5)
        self.assertIn("40 min", t_walk)

        # 0 meters
        self.assertEqual(estimate_travel_time(0.0), "0 mins")

    def test_map_search_widget_embedded(self) -> None:
        """Verify that interactive client-side search widget is injected into Folium maps when requested."""
        from map_view import build_empty_map
        m = build_empty_map(include_search_bar=True)
        html_str = m._repr_html_()
        self.assertIn("pf-search-widget", html_str)
        self.assertIn("pf-search-input", html_str)
        self.assertIn("Search location or landmark", html_str)

    def test_standard_nav_hud_no_algorithm_stats(self) -> None:
        """Verify that standard navigation map HUD focuses on trip ETA and omits CPU stats."""
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from astar import find_shortest_path_astar
        from map_view import build_folium_map

        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")
        res = find_shortest_path_astar(graph, u, v)

        m = build_folium_map(
            graph=graph,
            result=res,
            start_coord=u_coords,
            dest_coord=v_coords,
            start_label=u_name,
            dest_label=v_name,
        )
        html_str = m._repr_html_()
        self.assertIn("Drive Time", html_str)
        self.assertIn("Walk Time", html_str)
        # Verify no CPU millisecond benchmark or visited node counts in the standard HUD
        self.assertNotIn("ms</td>", html_str)
        self.assertNotIn("Nodes Explored", html_str)

    def test_clean_startup_empty_map_no_default_flags(self) -> None:
        """Verify that starting map with no points has zero flag markers or directions."""
        from map_view import build_empty_map
        m = build_empty_map(center=(23.3699, 85.3253), zoom=14, start_point=None, dest_point=None)
        html_str = m._repr_html_()
        self.assertNotIn("Start:</b>", html_str)
        self.assertNotIn("Destination:</b>", html_str)

    def test_show_hud_false_suppresses_hud(self) -> None:
        """Verify that show_hud=False hides the intrusive top-right HUD card."""
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from astar import find_shortest_path_astar
        from map_view import build_folium_map

        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")
        res = find_shortest_path_astar(graph, u, v)

        m = build_folium_map(
            graph=graph,
            result=res,
            start_coord=u_coords,
            dest_coord=v_coords,
            start_label=u_name,
            dest_label=v_name,
            show_hud=False,
        )
        html_str = m._repr_html_()
        self.assertNotIn("PathFinder Route Navigator", html_str)

    def test_init_session_state_clean_route(self) -> None:
        """Verify session state initializes with no route and collapsed menu."""
        import inspect
        from app import init_session_state
        src = inspect.getsource(init_session_state)
        self.assertIn("st.session_state.route_result = None", src)
        self.assertIn("st.session_state.show_route_menu = False", src)

    def test_current_location_marker_rendering(self) -> None:
        """Verify that current location renders glowing blue dot marker and LocateControl."""
        from map_view import build_empty_map, build_folium_map
        from osm_loader import get_ranchi_road_network, resolve_location_or_coords
        from astar import find_shortest_path_astar

        my_loc = (23.3699, 85.3253, "My Location")

        # In empty map
        m_empty = build_empty_map(current_location=my_loc)
        html_empty = m_empty._repr_html_()
        self.assertIn("My Location", html_empty)
        self.assertIn("#2563EB", html_empty)

        # In route map
        graph = get_ranchi_road_network()
        u, u_coords, u_name = resolve_location_or_coords(graph, "Albert Ekka Chowk")
        v, v_coords, v_name = resolve_location_or_coords(graph, "Ranchi Railway Station")
        res = find_shortest_path_astar(graph, u, v)

        m_route = build_folium_map(
            graph=graph,
            result=res,
            start_coord=u_coords,
            dest_coord=v_coords,
            start_label=u_name,
            dest_label=v_name,
            current_location=my_loc,
        )
        html_route = m_route._repr_html_()
        self.assertIn("My Location", html_route)
        self.assertIn("#2563EB", html_route)


class TestTrafficAwareRouting(unittest.TestCase):
    """Traffic estimates preserve distances and provide a time-based cost."""

    @staticmethod
    def _two_route_graph() -> Graph:
        g = Graph()
        # Direct primary is shortest; the longer trunk option has a higher
        # modeled base speed and can be faster under traffic-aware routing.
        g.add_edge("A", "B", 4.0, road_name="Main Road", highway="primary")
        g.add_edge("A", "C", 2.1, highway="trunk")
        g.add_edge("C", "B", 2.1, highway="trunk")
        g.add_edge("X", "Y", 1.0)
        return g

    def test_distance_is_immutable_and_shortest_mode_is_unchanged(self) -> None:
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, apply_traffic_mode
        g = self._two_route_graph()
        before = g.get_edge("A", "B").weight
        normal = find_shortest_path(g, "A", "B")
        traffic = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        self.assertEqual(normal.path, ["A", "B"])
        self.assertEqual(normal.total_distance, 4.0)
        self.assertEqual(traffic.total_distance, 4.2)
        self.assertEqual(g.get_edge("A", "B").weight, before)

    def test_speed_and_time_order_by_traffic_level(self) -> None:
        from traffic import estimated_speed_kmh, estimated_travel_time_seconds
        low = Edge("B", 1000, highway="primary")
        medium = Edge("B", 1000, highway="primary")
        high = Edge("B", 1000, highway="primary")
        self.assertGreater(estimated_speed_kmh(low, "06:30"), estimated_speed_kmh(medium, "12:00"))
        self.assertGreater(estimated_speed_kmh(medium, "12:00"), estimated_speed_kmh(high, "08:30"))
        self.assertLess(estimated_travel_time_seconds(low, "06:30"), estimated_travel_time_seconds(medium, "12:00"))
        self.assertLess(estimated_travel_time_seconds(medium, "12:00"), estimated_travel_time_seconds(high, "08:30"))

    def test_time_cost_can_choose_faster_alternative(self) -> None:
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, apply_traffic_mode
        g = self._two_route_graph()
        dijkstra = find_shortest_path(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        astar = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        self.assertEqual(astar.path, ["A", "C", "B"])
        self.assertAlmostEqual(astar.total_cost, dijkstra.total_cost)
        self.assertEqual(astar.total_distance, 4.2)

    def test_shortest_route_can_also_be_fastest(self) -> None:
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, apply_traffic_mode
        g = Graph()
        g.add_edge("A", "B", 4.0, highway="primary")
        g.add_edge("A", "C", 3.0, highway="primary")
        g.add_edge("C", "B", 3.0, highway="primary")
        res = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "13:00"))
        self.assertEqual(res.path, ["A", "B"])

    def test_departure_time_changes_estimates_and_results_are_deterministic(self) -> None:
        from traffic import estimated_travel_time_seconds, get_traffic_color
        edge = Edge("B", 1000, highway="primary")
        morning = estimated_travel_time_seconds(edge, "08:00")
        one_hour_later = estimated_travel_time_seconds(edge, "09:00")
        self.assertNotEqual(morning, one_hour_later)
        self.assertLess(abs(one_hour_later - morning) / morning, 0.1)
        self.assertNotEqual(get_traffic_color(edge, "08:00"), get_traffic_color(edge, "09:00"))
        self.assertNotEqual(estimated_travel_time_seconds(edge, "13:00"), estimated_travel_time_seconds(edge, "14:00"))
        before_hour = estimated_travel_time_seconds(edge, "09:59")
        at_hour = estimated_travel_time_seconds(edge, "10:00")
        self.assertLess(abs(at_hour - before_hour) / before_hour, 0.01)
        self.assertEqual(estimated_travel_time_seconds(edge, "08:30"), estimated_travel_time_seconds(edge, "08:30"))

    def test_morning_and_evening_have_distinct_profiles_and_route_costs(self) -> None:
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, apply_traffic_mode, get_traffic_level
        g = Graph()
        g.add_edge("A", "B", 1000, highway="secondary")
        secondary = Edge("B", 100, highway="secondary")
        self.assertNotEqual(get_traffic_level(secondary, "08:30"), get_traffic_level(secondary, "18:00"))
        morning = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        evening = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "18:00"))
        self.assertEqual(morning.path, evening.path)
        self.assertNotEqual(morning.total_cost, evening.total_cost)

    def test_hotspots_are_localized_and_follow_ranchi_peak_windows(self) -> None:
        from traffic import HOTSPOT_ZONES, estimated_speed_kmh, estimated_travel_time_seconds, get_traffic_period, hotspot_speed_multiplier
        g = Graph()
        g.add_node("main_a", 23.36990, 85.32530)
        g.add_node("main_b", 23.36950, 85.32525)
        g.add_node("outer_a", 23.34000, 85.30000)
        g.add_node("outer_b", 23.34050, 85.30000)
        g.add_edge("main_a", "main_b", 100, highway="primary")
        g.add_edge("outer_a", "outer_b", 100, highway="primary")
        main = g.get_edge("main_a", "main_b")
        outer = g.get_edge("outer_a", "outer_b")

        # Hotspots intensify during the supplied windows and remain localized.
        speeds = {
            at: estimated_speed_kmh(main, at, graph=g, origin_node="main_a")
            for at in ("07:00", "09:00", "14:00", "18:00")
        }
        self.assertLess(speeds["09:00"], speeds["07:00"])
        self.assertLess(speeds["14:00"], speeds["07:00"])
        self.assertLess(speeds["18:00"], speeds["07:00"])
        self.assertEqual(
            estimated_travel_time_seconds(outer, "09:00", graph=g, origin_node="outer_a"),
            estimated_travel_time_seconds(outer, "09:00"),
        )
        self.assertEqual(get_traffic_period("10:15"), "morning_peak")
        self.assertEqual(get_traffic_period("14:00"), "school_dismissal")
        self.assertEqual(get_traffic_period("20:15"), "evening_peak")

        # Every configured named hotspot has a spatial slowdown; distant roads do not.
        for index, zone in enumerate(HOTSPOT_ZONES):
            lat, lon = zone["path"][0]
            near_a, near_b = f"zone_{index}_a", f"zone_{index}_b"
            g.add_node(near_a, lat, lon)
            g.add_node(near_b, lat + 0.0002, lon + 0.0002)
            g.add_edge(near_a, near_b, 80, highway="primary")
            edge = g.get_edge(near_a, near_b)
            self.assertLess(hotspot_speed_multiplier(edge, "18:00", g, near_a), 1.0, zone["name"])

    def test_missing_metadata_is_safe(self) -> None:
        from traffic import LOW, classify_road, estimated_speed_kmh, get_traffic_level
        edge = Edge("B", 120.0)
        self.assertEqual(classify_road(edge), "unknown")
        self.assertGreater(estimated_speed_kmh(edge, "08:30"), 0)
        self.assertEqual(get_traffic_level(edge, "13:00"), LOW)
        self.assertEqual(get_traffic_level(edge, "18:00"), LOW)

    def test_highway_metadata_round_trips_without_changing_length(self) -> None:
        import os
        import tempfile
        from osm_loader import save_graph_to_json, graph_from_json
        g = Graph()
        g.add_edge("A", "B", 735.0, bidirectional=False, road_name="Ring Road", highway="primary")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.json")
            save_graph_to_json(g, path)
            loaded = graph_from_json(path)
        edge = loaded.get_edge("A", "B")
        self.assertEqual(edge.weight, 735.0)
        self.assertEqual(edge.highway, "primary")

    def test_traffic_analysis_reports_physical_distance_and_time(self) -> None:
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, analyse_route, apply_traffic_mode
        g = self._two_route_graph()
        res = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        info = analyse_route(g, res.path, TRAFFIC_AWARE, "08:30")
        self.assertEqual(info.distance, res.total_distance)
        self.assertGreater(info.travel_time_seconds, 0)

    def test_map_traffic_segments_follow_only_selected_path(self) -> None:
        from traffic import TRAFFIC_AWARE, iter_traffic_segments, LOW, MEDIUM
        g = Graph()
        g.add_node("A", 23.37, 85.32)
        g.add_node("B", 23.36, 85.33)
        g.add_node("C", 23.35, 85.34)
        g.add_node("X", 23.38, 85.31)
        g.add_node("Y", 23.39, 85.30)
        g.add_edge("A", "B", 1000, bidirectional=False, highway="primary")
        g.add_edge("B", "C", 1000, bidirectional=False, highway="residential")
        g.add_edge("X", "Y", 1000, bidirectional=False, highway="primary")
        groups = iter_traffic_segments(g, TRAFFIC_AWARE, "08:30", ["A", "B", "C"])
        self.assertEqual(sum(map(len, groups.values())), 2)
        self.assertEqual(len(groups[MEDIUM]), 1)
        self.assertEqual(len(groups[LOW]), 1)
        nearby = iter_traffic_segments(
            g, TRAFFIC_AWARE, "08:30", bounds=(23.34, 85.30, 23.375, 85.34),
        )
        self.assertEqual(sum(map(len, nearby.values())), 2)

    def test_map_traffic_overlay(self) -> None:
        try:
            import folium  # noqa: F401
        except ImportError:
            self.skipTest("folium is not installed in this environment")
        from astar import find_shortest_path_astar
        from traffic import TRAFFIC_AWARE, apply_traffic_mode
        from map_view import build_folium_map, build_empty_map
        g = self._two_route_graph()
        res = find_shortest_path_astar(g, "A", "B", cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        # Minimal map check with graph geometry and traffic model exercised.
        g.add_node("A", 23.37, 85.32)
        g.add_node("B", 23.36, 85.33)
        g.add_node("C", 23.365, 85.325)
        m = build_folium_map(g, res, (23.37, 85.32), (23.36, 85.33), traffic_mode=TRAFFIC_AWARE, departure_time="08:30")
        self.assertIn("Traffic along selected route", m.get_root().render())
        empty_html = build_empty_map(center=(23.37, 85.32), graph=g, traffic_mode=TRAFFIC_AWARE, departure_time="08:30").get_root().render()
        self.assertIn("Nearby traffic estimates", empty_html)
        self.assertNotIn("Traffic along selected route", empty_html)

    def test_legacy_cached_network_has_no_mutated_lengths(self) -> None:
        from osm_loader import get_ranchi_road_network
        g = get_ranchi_road_network()
        u = g.get_nodes()[0]
        edge = g.get_neighbors(u)[0]
        original = edge.weight
        from traffic import TRAFFIC_AWARE, apply_traffic_mode
        find_shortest_path(g, u, edge.destination, cost_function=apply_traffic_mode(TRAFFIC_AWARE, "08:30"))
        self.assertEqual(g.get_edge(u, edge.destination).weight, original)


class BookmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        import os
        import streamlit as st
        from app import BOOKMARKS_FILE
        if os.path.exists(BOOKMARKS_FILE):
            os.remove(BOOKMARKS_FILE)
        st.session_state.clear()
        from app import init_session_state
        init_session_state()

    def tearDown(self) -> None:
        import os
        from app import BOOKMARKS_FILE
        if os.path.exists(BOOKMARKS_FILE):
            os.remove(BOOKMARKS_FILE)

    def test_bookmark_initial_state(self) -> None:
        import streamlit as st
        self.assertEqual(st.session_state.bookmarks, [])
        self.assertFalse(st.session_state.show_bookmarks)

    def test_add_bookmark_success(self) -> None:
        import streamlit as st
        from app import add_bookmark
        ok, msg = add_bookmark("Home", "Albert Ekka Chowk", 23.3699, 85.3253)
        self.assertTrue(ok)
        self.assertEqual(len(st.session_state.bookmarks), 1)
        self.assertEqual(st.session_state.bookmarks[0]["name"], "Home")
        self.assertAlmostEqual(st.session_state.bookmarks[0]["lat"], 23.3699)
        self.assertAlmostEqual(st.session_state.bookmarks[0]["lon"], 85.3253)

    def test_bookmark_limit_of_twenty(self) -> None:
        import streamlit as st
        from app import add_bookmark, MAX_BOOKMARKS
        self.assertEqual(MAX_BOOKMARKS, 20)
        for i in range(20):
            ok, _ = add_bookmark(f"Place {i+1}", f"Addr {i+1}", 23.30 + i * 0.001, 85.30 + i * 0.001)
            self.assertTrue(ok)
        self.assertEqual(len(st.session_state.bookmarks), 20)

        # Attempt to add 21st bookmark must be rejected
        ok21, err = add_bookmark("Place 21", "Addr 21", 23.40, 85.40)
        self.assertFalse(ok21)
        self.assertIn("limit reached", err.lower())
        self.assertEqual(len(st.session_state.bookmarks), 20)

    # Keep alias for test runners expecting previous name
    test_bookmark_limit_of_two = test_bookmark_limit_of_twenty

    def test_bookmark_validation_empty_and_duplicate_name(self) -> None:
        import streamlit as st
        from app import add_bookmark
        # Empty name
        ok, err = add_bookmark("   ", "Addr", 23.35, 85.30)
        self.assertFalse(ok)
        self.assertIn("empty", err.lower())

        # Add initial
        ok1, _ = add_bookmark("Office", "Addr 1", 23.35, 85.30)
        self.assertTrue(ok1)

        # Duplicate name
        ok2, err2 = add_bookmark("office", "Addr 2", 23.36, 85.31)
        self.assertFalse(ok2)
        self.assertIn("already exists", err2.lower())

    def test_delete_bookmark(self) -> None:
        import streamlit as st
        from app import add_bookmark, delete_bookmark
        add_bookmark("Place 1", "Addr 1", 23.35, 85.30)
        add_bookmark("Place 2", "Addr 2", 23.36, 85.31)
        self.assertEqual(len(st.session_state.bookmarks), 2)

        # Delete first bookmark
        ok, msg = delete_bookmark(0)
        self.assertTrue(ok)
        self.assertEqual(len(st.session_state.bookmarks), 1)
        self.assertEqual(st.session_state.bookmarks[0]["name"], "Place 2")

        # Now can add another bookmark again
        ok_new, _ = add_bookmark("Place 3", "Addr 3", 23.37, 85.32)
        self.assertTrue(ok_new)
        self.assertEqual(len(st.session_state.bookmarks), 2)

    def test_delete_bookmark_invalid_index(self) -> None:
        from app import delete_bookmark
        ok, err = delete_bookmark(5)
        self.assertFalse(ok)

    def test_bookmarks_in_map_rendering(self) -> None:
        import streamlit as st
        from app import add_bookmark
        add_bookmark("Favorite Spot", "Ranchi Lake", 23.36, 85.32)
        from map_view import build_empty_map
        import folium
        m = build_empty_map()
        for bm in st.session_state.bookmarks:
            folium.Marker(
                location=[bm["lat"], bm["lon"]],
                tooltip=f"⭐ Bookmark: {bm['name']}",
            ).add_to(m)
        html = m._repr_html_()
        self.assertIn("Bookmark: Favorite Spot", html)

    def test_bookmarks_appear_in_direction_dropdown_options(self) -> None:
        import streamlit as st
        from app import add_bookmark, MY_LOCATION_LABEL
        from osm_loader import RANCHI_LANDMARKS

        add_bookmark("My Gym", "Near Albert Ekka", 23.371, 85.326)
        add_bookmark("My Office", "Main Road", 23.355, 85.318)

        # Build options as done in directions drawer
        start_options = [MY_LOCATION_LABEL]
        for bm in st.session_state.bookmarks:
            start_options.append(f"🔖 {bm['name']}")
        start_options.extend(RANCHI_LANDMARKS.keys())

        self.assertIn("🔖 My Gym", start_options)
        self.assertIn("🔖 My Office", start_options)
        self.assertEqual(start_options[1], "🔖 My Gym")
        self.assertEqual(start_options[2], "🔖 My Office")

    def test_bookmark_selection_sets_coordinates(self) -> None:
        import streamlit as st
        from app import add_bookmark
        add_bookmark("Home Base", "Ranchi Station", 23.352, 85.334)

        # Simulate selecting "🔖 Home Base"
        chosen_mode = "🔖 Home Base"
        bm_name = chosen_mode.replace("🔖 ", "", 1).strip()
        bm_match = next((b for b in st.session_state.bookmarks if b["name"] == bm_name), None)
        self.assertIsNotNone(bm_match)
        st.session_state.start_name = bm_match["name"]
        st.session_state.start_coords = (bm_match["lat"], bm_match["lon"])

        self.assertEqual(st.session_state.start_name, "Home Base")
        self.assertEqual(st.session_state.start_coords, (23.352, 85.334))

    def test_bookmarks_persist_across_page_refresh(self) -> None:
        """Verify bookmarks are written to JSON on device and persist when page is refreshed."""
        import os
        import json
        import streamlit as st
        from app import add_bookmark, init_session_state, BOOKMARKS_FILE

        # 1. Add a bookmark in session
        ok, _ = add_bookmark("Persistent Place", "Main Road, Ranchi", 23.35, 85.32)
        self.assertTrue(ok)

        # 2. Check JSON file on disk exists and contains bookmark
        self.assertTrue(os.path.exists(BOOKMARKS_FILE))
        with open(BOOKMARKS_FILE, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertEqual(len(disk_data), 1)
        self.assertEqual(disk_data[0]["name"], "Persistent Place")
        self.assertAlmostEqual(disk_data[0]["lat"], 23.35)

        # 3. Simulate browser page refresh: clear session_state completely
        st.session_state.clear()
        self.assertNotIn("bookmarks", st.session_state)

        # 4. Initialize session state as happens on page reload
        init_session_state()

        # 5. Bookmarks must still be present and loaded from the device JSON file!
        self.assertEqual(len(st.session_state.bookmarks), 1)
        self.assertEqual(st.session_state.bookmarks[0]["name"], "Persistent Place")
        self.assertAlmostEqual(st.session_state.bookmarks[0]["lat"], 23.35)

    def test_bookmarks_delete_updates_json_in_real_time(self) -> None:
        """Verify delete_bookmark updates the JSON file on the device in real time."""
        import os
        import json
        import streamlit as st
        from app import add_bookmark, delete_bookmark, init_session_state, BOOKMARKS_FILE

        add_bookmark("Place A", "Addr A", 23.30, 85.30)
        add_bookmark("Place B", "Addr B", 23.40, 85.40)
        self.assertEqual(len(st.session_state.bookmarks), 2)

        # Delete Place A
        ok, _ = delete_bookmark(0)
        self.assertTrue(ok)
        self.assertEqual(len(st.session_state.bookmarks), 1)

        # Verify disk JSON file updated in real time
        with open(BOOKMARKS_FILE, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertEqual(len(disk_data), 1)
        self.assertEqual(disk_data[0]["name"], "Place B")

        # Simulate page refresh
        st.session_state.clear()
        init_session_state()
        self.assertEqual(len(st.session_state.bookmarks), 1)
        self.assertEqual(st.session_state.bookmarks[0]["name"], "Place B")

    def test_save_and_load_bookmarks_custom_path(self) -> None:
        """Verify save_bookmarks and load_bookmarks with custom JSON file paths."""
        import os
        import tempfile
        from app import save_bookmarks, load_bookmarks

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "custom_bookmarks.json")
            sample = [
                {"name": "Custom Spot", "address": "Some Street", "lat": 23.33, "lon": 85.33}
            ]
            ok = save_bookmarks(sample, filepath=test_path)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(test_path))

            loaded = load_bookmarks(filepath=test_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["name"], "Custom Spot")
            self.assertAlmostEqual(loaded[0]["lat"], 23.33)


if __name__ == "__main__":
    unittest.main()
