# 🌐 PathFinder (Stage 2: Real-World Map Integration)

A modular, high-performance shortest-pathfinding engine in Python implementing **Dijkstra's Algorithm from scratch** on real-world OpenStreetMap (OSM) road networks, featuring interactive web map visualization powered by **Folium**.

Originally developed in Stage 1 with a fictional regional road graph, Stage 2 expands the platform to ingest real-world geospatial networks (initial deployment: **Ranchi, Jharkhand, India**), mapping actual streets, intersections, and road curvature while strictly retaining our custom Dijkstra routing engine.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Real-World Map Coverage (Ranchi)](#-real-world-map-coverage-ranchi)
- [Installation & Virtual Environment](#-installation--virtual-environment)
- [Usage Guide](#-usage-guide)
  - [1. Real-World Landmark Routing](#1-real-world-landmark-routing)
  - [2. GPS Coordinate Routing](#2-gps-coordinate-routing)
  - [3. Interactive Map Generation](#3-interactive-map-generation)
  - [4. Interactive Prompt Mode](#4-interactive-prompt-mode)
  - [5. Listing Landmarks & Fictional Towns](#5-listing-landmarks--fictional-towns)
  - [6. Stage 1 Fictional Network Mode](#6-stage-1-fictional-network-mode-backward-compatible)
  - [7. Refreshing Live OpenStreetMap Data](#7-refreshing-live-openstreetmap-data)
- [Algorithm Complexity & Analysis](#-algorithm-complexity--analysis)
- [Automated Testing](#-automated-testing)
- [Roadmap](#-future-roadmap)

---

## ✨ Features

- **Custom Dijkstra Routing Engine**: Zero reliance on third-party routing libraries (such as NetworkX's or OSMnx's built-in shortest path functions). Every shortest path is calculated using our pure-Python binary min-heap Dijkstra algorithm (`heapq`).
- **Real-World OpenStreetMap Ingestion**: Converts OSM road segments and intersections into a directed, weighted adjacency-list graph via **OSMnx**.
- **Interactive Folium Web Maps**: Generates interactive HTML maps (`route_map.html`) displaying:
  - Custom start/destination markers (green/red with metadata popups).
  - High-visibility polyline tracing exact road geometries.
  - Floating HUD card showing route summary, distance, execution time, and segment count.
- **Fast Sub-50ms Offline Startup**: Pre-cached and serialized road network (`data/ranchi_network.json`) enables instant offline execution, with on-demand Overpass API re-fetching.
- **Flexible Location Resolution**:
  - Predefined landmarks (e.g., `Albert Ekka Chowk`, `Ranchi Railway Station`, `Nucleus Mall`, `Morabadi Ground`).
  - Arbitrary GPS coordinates (`--start-coords 23.3699,85.3253 --dest-coords 23.3512,85.3347`).
  - Nearest road node matching using vectorized distance approximation.
- **Full Backward Compatibility**: 100% of Stage 1 functionality, fictional road network tests, and CLI flags are preserved.
- **Automated Test Suite**: 40 automated unit and integration tests (`unittest`) verifying graph conversion, Dijkstra correctness, coordinate parsing, map generation, and CLI commands.

---

## 🏗️ Project Architecture

```text
Maps/
├── main.py              # Unified CLI entry point & interactive session manager
├── graph.py             # Weighted Graph & Edge dataclasses with geographic coordinate support
├── dijkstra.py          # Custom Dijkstra algorithm with min-heap priority queue (from scratch)
├── osm_loader.py        # OSMnx downloader, JSON cache serializer, coordinate resolver
├── map_view.py          # Folium interactive HTML map generator with markers and HUD
├── display.py           # Terminal rendering, turn-by-turn road itinerary, performance metrics
├── tests.py             # 40 automated unit & integration tests (unittest)
├── requirements.txt     # Stage 2 dependencies (osmnx, folium)
├── data/
│   └── ranchi_network.json  # Serialized real-world road network cache (~2,500 nodes, ~6,200 edges)
└── README.md            # Comprehensive documentation
```

### Module Responsibilities

| Module | Core Responsibility |
|---|---|
| [`graph.py`](graph.py) | Weighted adjacency-list `Graph` storing node coordinates `(lat, lon)`, edge geometries, and fast vectorized `find_nearest_node()` lookup. |
| [`dijkstra.py`](dijkstra.py) | Pure-Python Dijkstra algorithm. Operates directly on the converted real-world road network using a binary min-heap (`heapq`) with early destination settling. |
| [`osm_loader.py`](osm_loader.py) | Ingests real road networks from OpenStreetMap via OSMnx, manages local JSON caching, and resolves landmark names or GPS coordinates to road network nodes. |
| [`map_view.py`](map_view.py) | Renders Leaflet-powered interactive HTML maps via Folium with color-coded start/end pins, curved route polylines, and floating trip metrics dashboard. |
| [`display.py`](display.py) | Terminal formatting: turn-by-turn corridor breakdown, distance in meters/kilometers, execution duration in $\mu\text{s}/\text{ms}$, and error notifications. |
| [`main.py`](main.py) | CLI argument parser (`argparse`), dual-mode selector (Real-World OSM vs Fictional Stage 1), and interactive user interface. |
| [`tests.py`](tests.py) | Comprehensive test suite covering graph structures, Dijkstra algorithm, coordinate parsing, map generation, and CLI end-to-end integration. |

---

## 📍 Real-World Map Coverage (Ranchi)

The initial real-world deployment covers central **Ranchi, Jharkhand, India** ($2,493$ intersections, $6,265$ directed road segments) centered around Albert Ekka Chowk, including key civic and transit corridors:

| Landmark | Coordinates (Lat, Lon) | Description |
|---|---|---|
| **Albert Ekka Chowk** | `23.3699, 85.3253` | City center / Main Road hub (Firayalal) |
| **Ranchi Railway Station** | `23.3512, 85.3347` | South Eastern Railway central terminus |
| **Nucleus Mall** | `23.3725, 85.3315` | Major retail and transit node (Circular Road) |
| **Morabadi Ground** | `23.3880, 85.3300` | Northern recreational & civic grounds |
| **Main Road Overbridge** | `23.3565, 85.3285` | Central arterial flyover connecting North/South |
| **Sujata Chowk** | `23.3590, 85.3270` | Southern commercial junction |
| **St. Xavier's College** | `23.3640, 85.3260` | Historic educational campus on Purulia Road |
| **Tagore Hill** | `23.3980, 85.3420` | Historic cultural peak in northern Ranchi |
| **Ranchi University** | `23.3800, 85.3275` | Northern academic zone |
| **Doranda Market** | `23.3380, 85.3250` | Southern commercial center |

---

## ⚙️ Installation & Virtual Environment

Python 3.10+ is required.

### 1. Create and Activate Virtual Environment

```bash
# In the project root directory
python3 -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate

# Or activate on Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Installed dependencies: `osmnx` (v2.x) and `folium` (v0.20+).*

---

## 🚀 Usage Guide

### 1. Real-World Landmark Routing

Find the shortest route between two landmarks in Ranchi:

```bash
python main.py -s "Albert Ekka Chowk" -d "Ranchi Railway Station"
```

**Terminal Output:**
```text
─────────────────────────────────────────────────────────────────
🏁 Route Summary: Albert Ekka Chowk (Node 766708626)  ➔  Ranchi Railway Station (Node 3682074540)
─────────────────────────────────────────────────────────────────

🗺️  Turn-by-Turn Itinerary:
   1. Follow [Firayalal Chowk Junction] for 16 m 
   2. Follow [Mahatma Gandhi Road] for 377 m (3 intersections)
   3. Follow [Chutia road] for 1.99 km (28 intersections)
   4. Follow [Station Road] for 94 m 
   5. Follow [Ranchi Club Road] for 220 m 

🛣️  Path Sequence (35 intersections):
   766708626 ➔ 2910367381 ➔ 9105673211 ➔ ... [29 intermediate intersections] ➔ 965183707 ➔ 1805442879 ➔ 3682074540

📊 Trip & Performance Metrics:
   • Total Distance       : 2.70 km
   • Number of Segments   : 34
   • Nodes Explored       : 2181
   • Search Execution Time: 8.55 ms
─────────────────────────────────────────────────────────────────

🗺️  Interactive Map Saved Successfully:
   📁 File: /home/tejas/Documents/Maps/Maps/route_map.html
   🌐 Open this file in your browser to view the interactive map!
```

### 2. GPS Coordinate Routing

Pass arbitrary GPS coordinates anywhere in the mapped area:

```bash
python main.py --start-coords 23.3699,85.3253 --dest-coords 23.3880,85.3300
```

### 3. Interactive Map Generation

Every successful real-world routing automatically generates an interactive HTML map:
- **Default path**: `route_map.html`
- **Custom path**: Use the `-o` / `--output-map` flag:
  ```bash
  python main.py -s "Nucleus Mall" -d "Morabadi Ground" -o my_trip.html
  ```

Open the generated file in any browser (e.g. Chrome, Firefox, Safari) or run:
```bash
xdg-open route_map.html   # Linux
open route_map.html       # macOS
start route_map.html      # Windows
```

### 4. Interactive Prompt Mode

Launch the guided interactive navigator:

```bash
python main.py
```

Prompts allow selecting between the **Real-World OpenStreetMap Network** and the **Fictional Regional Network**, selecting landmarks by index `[1-10]`, typing coordinates, and running multiple queries consecutively.

### 5. Listing Landmarks & Fictional Towns

- **List Ranchi Landmarks**:
  ```bash
  python main.py --list-landmarks
  ```
- **List Fictional Stage 1 Towns**:
  ```bash
  python main.py --list
  ```

### 6. Stage 1 Fictional Network Mode (Backward-Compatible)

The fictional network from Stage 1 remains accessible:

```bash
python main.py --fictional -s "Bayview" -d "Frostford"
```
*(Fictional queries also execute automatically if the location names match Stage 1 towns).*

### 7. Refreshing Live OpenStreetMap Data

To download fresh data directly from the Overpass API:

```bash
python main.py --refresh-osm
```

---

## 🧮 Algorithm Complexity & Analysis

### 1. Data Structure Design
- **Adjacency List Graph**: `Graph` stores nodes and outgoing `Edge` structures in memory:
  - Average outgoing edge access: $\mathcal{O}(1)$
  - Storage space: $\mathcal{O}(V + E)$
- **Priority Queue**: Python's binary min-heap (`heapq`):
  - Insertion (`heappush`): $\mathcal{O}(\log |Q|)$
  - Extraction (`heappop`): $\mathcal{O}(\log |Q|)$

### 2. Time Complexity: $\mathcal{O}((V + E) \log V)$

1. **Initialization**: Distance dictionary and priority queue setup take $\mathcal{O}(V)$ time.
2. **Min-Heap Popping**: Settling each intersection at most once takes at most $\sum_{i=1}^{V} \mathcal{O}(\log V) = \mathcal{O}(V \log V)$.
3. **Edge Relaxation**: Exploring each directed street segment relaxes distances via `heapq.heappush`: $\sum_{j=1}^{E} \mathcal{O}(\log V) = \mathcal{O}(E \log V)$.
4. **Total Worst-Case Time**:
   $$\mathcal{O}((V + E) \log V)$$

> **Real-World Performance**: On the $2,493$-node Ranchi graph, Dijkstra's algorithm consistently finds the optimal route in **$5 \text{ to } 15 \text{ ms}$** on commodity hardware, utilizing early target termination.

### 3. Space Complexity: $\mathcal{O}(V + E)$
- Adjacency list and coordinates: $\mathcal{O}(V + E)$
- Predecessor map and distance map: $\mathcal{O}(V)$
- Priority queue: $\mathcal{O}(V)$
- **Total Memory Footprint**: $< 15 \text{ MB}$ in RAM.

---

## 🧪 Automated Testing

The automated test suite in [`tests.py`](tests.py) comprises **40 tests** covering both Stage 1 and Stage 2:

```bash
python -m unittest -v tests.py
```

### Coverage Summary

- **Graph Primitives**: Adding nodes/edges, coordinate tracking, negative weight rejection, nearest-node calculation.
- **Custom Dijkstra Algorithm**:
  - Triangle inequality (optimal multi-hop vs direct).
  - Source equals destination ($0 \text{ m}$).
  - Disconnected component handling (`found=False`, distance $=\infty$).
  - Negative edge weight detection defense.
  - Cycle tolerance and zero-weight edge support.
- **Stage 2 OpenStreetMap & Coordinates**:
  - Coordinate parsing (`lat, lon`, parenthesized, negative, invalid input rejection).
  - Serialized JSON network cache integrity ($> 2,000$ edges).
  - Real-world nearest road node resolution.
  - Landmark resolution and fuzzy fallback.
- **Map Visualization**:
  - HTML file generation with Folium.
  - Marker insertion (Start/Destination).
  - PolyLine geometry embedding and HUD card elements.
- **CLI & Integration**:
  - Exit code verification (`0` on success, `1` on invalid input, `2` on unreachable destination).
  - Real-world landmark CLI execution.
  - Coordinate flag CLI execution.
  - Backward compatibility with Stage 1 commands.

```text
Ran 40 tests in 1.225s
OK
```

---

## 🔮 Future Roadmap

- **Stage 3 — Heuristic Pathfinding (A\*)**: Integrate the Haversine distance heuristic $h(n)$ to further accelerate long-distance routing.
- **Stage 4 — Live Traffic & Elevation**: Factor slope/elevation gradients from DEM data and dynamic travel time penalties.