# 🌐 PathFinder (Stage 4: Interactive Navigation UI)

A modern, full-featured shortest-pathfinding engine and web navigation application in Python. Features custom implementations of **Dijkstra's algorithm** and **A\* search** from scratch, integrated with real-world OpenStreetMap (OSM) road networks, live **Nominatim** geocoding, an interactive **Streamlit + Folium** web interface, and empirical side-by-side algorithm benchmarking.

Both pathfinding algorithms are built from first principles using Python's standard library `heapq` with zero reliance on third-party routing libraries (no NetworkX shortest-path or OSMnx routing routines). Stage 4 delivers a responsive web navigation interface allowing users to search places or addresses, click directly on the map, swap endpoints, and compare routing algorithms in real time.

---

## 📋 Table of Contents
- [✨ Features](#-features)
- [🖥️ Interactive Web UI (Stage 4)](#️-interactive-web-ui-stage-4)
- [🏗️ Project Architecture](#️-project-architecture)
- [🔬 Algorithm Comparison: Dijkstra vs A*](#-algorithm-comparison-dijkstra-vs-a)
  - [Theoretical Analysis](#1-theoretical-analysis)
  - [Heuristic Design & Admissibility](#2-heuristic-design--admissibility)
  - [Empirical Benchmarks (Ranchi Road Network)](#3-empirical-benchmarks-ranchi-road-network)
- [⚙️ Installation & Virtual Environment](#️-installation--virtual-environment)
- [🚀 Usage Guide](#-usage-guide)
  - [1. Launch Interactive Web Navigation UI (Stage 4)](#1-launch-interactive-web-navigation-ui-stage-4)
  - [2. CLI: Side-by-Side Algorithm Comparison](#2-cli-side-by-side-algorithm-comparison)
  - [3. CLI: Routing with A* (Fastest)](#3-cli-routing-with-a-fastest)
  - [4. CLI: Routing with Dijkstra](#4-cli-routing-with-dijkstra)
  - [5. CLI: GPS Coordinate Routing](#5-cli-gps-coordinate-routing)
  - [6. CLI: Interactive Prompt Mode](#6-cli-interactive-prompt-mode)
  - [7. CLI: Listing Landmarks & Fictional Towns](#7-cli-listing-landmarks--fictional-towns)
  - [8. CLI: Stage 1 Fictional Network Mode](#8-cli-stage-1-fictional-network-mode-backward-compatible)
  - [9. Refreshing Live OpenStreetMap Road Data](#9-refreshing-live-openstreetmap-road-data)
- [📍 Real-World Map Coverage (Ranchi)](#-real-world-map-coverage-ranchi)
- [🧪 Automated Testing (61 Tests)](#-automated-testing-61-tests)
- [🔮 Future Roadmap](#-future-roadmap)

---

## ✨ Features

- **Interactive Streamlit Web Interface (`app.py`)**:
  - Clean, responsive dashboard designed with Streamlit and Folium.
  - Searchable **From (Origin)** and **To (Destination)** fields supporting addresses, landmarks, and coordinates.
  - **OpenStreetMap Nominatim Geocoding**: Live autocompletion suggestions with rate-limiting, local caching, and offline landmark fallback.
  - **Interactive Map Selection**: Click anywhere on the map to set origin or destination pins.
  - **⇄ Swap Button**: Instant one-click interchange of start and destination points.
  - **Algorithm Switcher**: Seamlessly toggle between A* Search, Dijkstra's Algorithm, or Side-by-Side Comparison.
  - **Real-Time Metrics Cards**: Live displays for total distance, execution time ($\mu\text{s}/\text{ms}$), nodes explored, and intersection counts.
  - **Turn-by-Turn Road Corridor Guidance**: Human-readable navigation itinerary grouping road segments into corridors.
- **Two Pure-Python Algorithms (From Scratch)**:
  - **Dijkstra's Algorithm**: Exhaustive uniform-cost search tracking cumulative distance $g(n)$.
  - **A\* Search Algorithm**: Informed best-first search tracking $f(n) = g(n) + h(n)$ using the great-circle Haversine distance heuristic.
  - Zero third-party routing engines (100% pure Python using `heapq`).
- **Real-World OpenStreetMap Ingestion**: Ingests real road segments and intersections into a directed, weighted adjacency-list graph via **OSMnx**.
- **Instant Sub-50ms Offline Startup**: Pre-cached and serialized road network (`data/ranchi_network.json`) enables instant offline routing without network overhead.
- **Resilient Geocoding Engine (`geocoder.py`)**:
  - Complies with Nominatim usage policy (custom User-Agent and $0.5\text{s}$ rate throttling).
  - In-memory LRU query cache to eliminate redundant HTTP requests.
  - Graceful degradation: falls back immediately to built-in local landmarks on network timeouts or API limits.
- **Full Backward Compatibility**: 100% of Stage 1, Stage 2, and Stage 3 functionality, CLI commands, and test suites are preserved.
- **Comprehensive Automated Test Suite**: 61 automated tests verifying graph structures, geocoding resilience, algorithm correctness, map visualization, and CLI commands.

---

## 🖥️ Interactive Web UI (Stage 4)

Launch the web navigation app with a single command:

```bash
streamlit run app.py
```
*(Or use `.venv/bin/streamlit run app.py` when using an isolated virtual environment)*

### Web UI Features & Capabilities

1. **Searchable From & To Fields**:
   - Select predefined landmarks from dropdowns (e.g. *Albert Ekka Chowk*, *Ranchi Railway Station*, *Nucleus Mall*, *Tagore Hill*).
   - Or select *"🔍 Search Address / Custom Place..."* to query OpenStreetMap Nominatim for real addresses or type raw coordinates `lat, lon`.
2. **Interactive Map Click-to-Select**:
   - Click anywhere on the interactive Leaflet map to inspect coordinates.
   - Use the `"📍 Set Click as Origin"` or `"🎯 Set Click as Destination"` buttons to update route endpoints with automatic reverse-geocoding.
3. **One-Click Endpoint Swap (`⇄`)**:
   - Click the swap button in the sidebar to effortlessly reverse the trip origin and destination.
4. **Algorithm Selection & Comparison**:
   - Choose **A\* Search (Recommended)** for ultra-fast goal-directed routing.
   - Choose **Dijkstra's Algorithm** for classic exhaustive exploration.
   - Choose **Compare Both** to compute both simultaneously, rendering a comparison banner and detailed benchmark matrix.
5. **Turn-by-Turn Guidance & Metrics**:
   - Inspect key performance indicators: Distance ($\text{km}$ or $\text{m}$), Execution Time ($\text{ms}$ or $\mu\text{s}$), Nodes Explored, and Intersections.
   - Browse the collapsible *"🗺️ Turn-by-Turn Road Corridor Guide"* to see road names, segment distances, and intersection counts.

---

## 🏗️ Project Architecture

```text
Maps/
├── app.py               # Streamlit + Folium interactive web navigation application (Stage 4)
├── geocoder.py          # Nominatim geocoder with caching, rate-limiting & landmark fallback
├── main.py              # Unified CLI entry point & interactive console session manager
├── graph.py             # Weighted Graph & Edge dataclasses with (lat, lon) coordinates
├── dijkstra.py          # Custom Dijkstra algorithm with min-heap priority queue (from scratch)
├── astar.py             # Custom A* algorithm with Haversine distance heuristic (from scratch)
├── osm_loader.py        # OSMnx downloader, JSON cache serializer, coordinate resolver
├── map_view.py          # Folium interactive HTML map generator with markers, routes, and HUD
├── display.py           # Terminal rendering, turn-by-turn road corridors, comparison tables
├── tests.py             # 61 automated unit & integration tests (unittest)
├── requirements.txt     # Dependencies (osmnx, folium, streamlit, streamlit-folium)
├── data/
│   └── ranchi_network.json  # Serialized real-world road network cache (~2,500 nodes, ~6,200 edges)
└── README.md            # Comprehensive documentation
```

### Module Responsibilities

| Module | Core Responsibility |
|---|---|
| [`app.py`](app.py) | Complete Streamlit web navigation UI: searchable inputs, swap button, algorithm selection, Folium map rendering, click handler, and metric cards. |
| [`geocoder.py`](geocoder.py) | OpenStreetMap Nominatim geocoding & reverse geocoding with custom User-Agent, $0.5\text{s}$ rate-limiting, in-memory caching, coordinate parser, and local landmark fallback. |
| [`map_view.py`](map_view.py) | Builds interactive Folium Leaflet maps for standalone export (`route_map.html`) and direct Streamlit embedding via `build_folium_map()` and `build_empty_map()`. |
| [`astar.py`](astar.py) | Custom A* implementation using `heapq` and $f(n) = g(n) + h(n)$ evaluation with Haversine straight-line distance heuristic. |
| [`dijkstra.py`](dijkstra.py) | Custom Dijkstra implementation using `heapq`. Exhaustive priority queue search tracking cumulative distance $g(n)$. |
| [`graph.py`](graph.py) | Adjacency-list `Graph` storing node coordinates `(lat, lon)`, curved edge geometries, and fast vectorized `find_nearest_node()`. |
| [`osm_loader.py`](osm_loader.py) | Ingests real road networks from OpenStreetMap via OSMnx, manages local JSON caching, and resolves landmark names or GPS coordinates. |
| [`display.py`](display.py) | Formats terminal output: side-by-side benchmark tables, turn-by-turn itineraries, and formatted metric strings. |
| [`main.py`](main.py) | CLI argument parser (`argparse`), dual-mode selector (Real-World OSM vs Fictional Stage 1), and interactive console navigator. |
| [`tests.py`](tests.py) | 61 automated tests verifying graph structures, Dijkstra and A* correctness, geocoder resilience, map generation, and CLI commands. |

---

## 🔬 Algorithm Comparison: Dijkstra vs A*

### 1. Theoretical Analysis

| Dimension | Dijkstra's Algorithm | A\* Search Algorithm |
|---|---|---|
| **Strategy** | Uniform-Cost Search (expands equidistant wave in all directions) | Informed Best-First Search (guided toward destination by heuristic $h(n)$) |
| **Evaluation Function** | $f(n) = g(n)$ (cumulative distance from start) | $f(n) = g(n) + h(n)$ (distance from start + estimated distance to goal) |
| **Heuristic Function** | None ($h(n) = 0$) | Haversine Great-Circle Distance ($h(n) \le d^*(n, \text{goal})$) |
| **Explored Space** | Circular / spherical wavefront exploring in all directions | Elliptical envelope focused toward the destination |
| **Time Complexity** | $\mathcal{O}((V + E) \log V)$ | $\mathcal{O}((V + E) \log V)$ worst-case, significantly lower in practice |
| **Space Complexity** | $\mathcal{O}(V + E)$ | $\mathcal{O}(V + E)$ |
| **Optimality Guarantee** | Guaranteed optimal for non-negative edge weights | Guaranteed optimal when $h(n)$ is admissible ($h(n) \le h^*(n)$) |

### 2. Heuristic Design & Admissibility

For surface road networks on Earth, the **great-circle (Haversine) distance** represents the theoretical minimum distance between two points:

$$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

- **Admissibility**: Because roads curve, detour around terrain, and adhere to speed networks, actual road length between any two coordinates is strictly greater than or equal to the straight-line great-circle distance:
  $$h(n) \le d^*(n, \text{goal}) \quad \forall n$$
  Therefore, A* is **mathematically guaranteed to find the exact optimal shortest path**, matching Dijkstra's solution.
- **Consistency (Monotonicity)**: By the spherical triangle inequality, for any two neighboring nodes $u$ and $v$:
  $$h(u) \le c(u, v) + h(v)$$
  This guarantees that once a node is settled by A*, its shortest distance is finalized, eliminating the need to re-open closed nodes.
- **Graceful Degradation**: On graphs without geographical coordinates (such as the Stage 1 fictional regional network), $h(n)$ defaults to $0.0$, smoothly degrading A* to Dijkstra's algorithm.

### 3. Empirical Benchmarks (Ranchi Road Network)

Benchmarked on the central Ranchi OpenStreetMap network ($2,493$ nodes, $6,265$ directed edges):

| Route | Dijkstra Distance | A\* Distance | Dijkstra Nodes | A\* Nodes | **Node Reduction** | Dijkstra Time | A\* Time | **Speedup** |
|---|---|---|---|---|---|---|---|---|
| **Albert Ekka Chowk ➔ Ranchi Railway Station** | $2,697.1\text{ m}$ | $2,697.1\text{ m}$ | $2,181$ | $301$ | **86.2% fewer** | $12.04\text{ ms}$ | $3.30\text{ ms}$ | **3.6x faster** |
| **Albert Ekka Chowk ➔ Morabadi Ground** | $1,930.5\text{ m}$ | $1,930.5\text{ m}$ | $1,401$ | $27$ | **98.1% fewer** | $9.87\text{ ms}$ | $0.88\text{ ms}$ | **11.2x faster** |
| **Sujata Chowk ➔ Tagore Hill** | $4,472.6\text{ m}$ | $4,472.6\text{ m}$ | $2,475$ | $1,290$ | **47.9% fewer** | $14.25\text{ ms}$ | $7.12\text{ ms}$ | **2.0x faster** |
| **Nucleus Mall ➔ Ranchi Railway Station** | $2,954.8\text{ m}$ | $2,954.8\text{ m}$ | $2,240$ | $412$ | **81.6% fewer** | $12.80\text{ ms}$ | $3.85\text{ ms}$ | **3.3x faster** |

Both algorithms return the exact same shortest path distance down to the millimeter, with A* cutting node explorations by **48% to 98%**.

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

*(Installs `osmnx`, `folium`, `streamlit`, and `streamlit-folium`)*

---

## 🚀 Usage Guide

### 1. Launch Interactive Web Navigation UI (Stage 4)

Start the web application:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`. Use the sidebar to search for locations, swap endpoints, select algorithms, and visualize the route.

### 2. CLI: Side-by-Side Algorithm Comparison

Run both Dijkstra and A* simultaneously from the terminal to compare performance and generate `route_map.html`:

```bash
python main.py -s "Albert Ekka Chowk" -d "Ranchi Railway Station" --compare
```

**Terminal Output:**
```text
══════════════════════════════════════════════════════════════════════
 🔬 ALGORITHM COMPARISON BENCHMARK: DIJKSTRA vs A* (A-STAR)
    From: Albert Ekka Chowk (Node 766708626)
    To:   Ranchi Railway Station (Node 3682074540)
══════════════════════════════════════════════════════════════════════

📊 Benchmark Metrics Comparison:
┌────────────────────────┬──────────────────┬──────────────────┬────────────────────┐
│ Metric                 │ Dijkstra         │ A* (A-Star)      │ Improvement / Note │
├────────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Shortest Distance      │ 2.70 km          │ 2.70 km          │ ✅ Exact Match (Optimal) │
│ Nodes Explored         │ 2,181            │ 301              │ 86.2% fewer nodes  │
│ Execution Time         │ 12.04 ms         │ 3.30 ms          │ 3.6x faster        │
│ Route Segments         │ 34 legs          │ 34 legs          │ Equal topology     │
│ Heuristic Guide        │ None (Blind search) │ Haversine Distance │ Admissible (h <= d*) │
└────────────────────────┴──────────────────┴──────────────────┴────────────────────┘

🗺️  Turn-by-Turn Itinerary:
   1. Follow [Firayalal Chowk Junction] for 16 m 
   2. Follow [Mahatma Gandhi Road] for 377 m (3 intersections)
   3. Follow [Chutia road] for 1.99 km (28 intersections)
   4. Follow [Station Road] for 94 m 
   5. Follow [Ranchi Club Road] for 220 m 

🛣️  Path Sequence (35 intersections):
   766708626 ➔ 2910367381 ➔ 9105673211 ➔ ... [29 intermediate intersections] ➔ 965183707 ➔ 1805442879 ➔ 3682074540
══════════════════════════════════════════════════════════════════════

🗺️  Interactive Map Saved Successfully:
   📁 File: /home/tejas/Documents/Maps/Maps/route_map.html
   🌐 Open this file in your browser to view the interactive map!
```

### 3. CLI: Routing with A* (Fastest)

To execute only A* pathfinding (the default mode):

```bash
python main.py -s "Albert Ekka Chowk" -d "Ranchi Railway Station" -a astar
```

### 4. CLI: Routing with Dijkstra

To execute only Dijkstra's algorithm:

```bash
python main.py -s "Albert Ekka Chowk" -d "Ranchi Railway Station" -a dijkstra
```

### 5. CLI: GPS Coordinate Routing

Pass arbitrary GPS coordinates anywhere in the mapped area with comparison:

```bash
python main.py --start-coords 23.3699,85.3253 --dest-coords 23.3880,85.3300 --compare
```

### 6. CLI: Interactive Prompt Mode

Launch the interactive console navigator with algorithm selection:

```bash
python main.py
```

Prompts allow:
1. Choosing between the **Real-World Network** and the **Fictional Regional Network**.
2. Selecting the pathfinding algorithm: `[1] A*`, `[2] Dijkstra`, or `[3] Compare Both`.
3. Selecting landmarks by index `[1-10]`, typing names, or entering `lat, lon`.

### 7. CLI: Listing Landmarks & Fictional Towns

- **List Ranchi Landmarks**:
  ```bash
  python main.py --list-landmarks
  ```
- **List Fictional Stage 1 Towns**:
  ```bash
  python main.py --list
  ```

### 8. CLI: Stage 1 Fictional Network Mode (Backward-Compatible)

The fictional network from Stage 1 remains fully accessible with algorithm selection:

```bash
python main.py --fictional -s "Bayview" -d "Frostford" --compare
```

### 9. Refreshing Live OpenStreetMap Road Data

To download fresh data directly from the Overpass API:

```bash
python main.py --refresh-osm
```

---

## 📍 Real-World Map Coverage (Ranchi)

The real-world network covers central **Ranchi, Jharkhand, India** ($2,493$ intersections, $6,265$ directed road segments) centered around Albert Ekka Chowk:

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

## 🧪 Automated Testing (61 Tests)

The automated test suite in [`tests.py`](tests.py) comprises **61 tests** covering Stage 1, Stage 2, Stage 3, and Stage 4:

```bash
python -m unittest -v tests.py
```

### Test Coverage Breakdown

- **Stage 4 Navigation UI & Geocoder (`TestGeocoder`, `TestStage4UI`)**:
  - Nominatim search with query parsing, raw coordinate parsing, and limit handling.
  - In-memory search query caching.
  - Local landmark fallback matching when offline or with empty queries.
  - Proximity-based reverse geocoding to recognizable landmarks.
  - Folium map instance builders (`build_folium_map`, `build_empty_map`).
- **A\* Pathfinding Core (`TestAStar`)**:
  - Great-circle Haversine distance accuracy and symmetry.
  - Heuristic admissibility and consistency verification ($h(u, v) \le \text{edge.weight}$).
  - A* graceful degradation on graphs without coordinates (matches Dijkstra 100%).
  - Optimal path selection over multi-hop vs direct routes (triangle inequality).
  - Identical start/destination ($0\text{ m}$).
  - Unreachable destination handling ($\infty$).
  - Negative edge weight detection defense (`ValueError`).
  - Strict distance parity with Dijkstra on real-world routes ($\Delta \le 0.1\text{ m}$).
  - Substantial node exploration reduction verification.
- **Algorithm Comparison & Visualization (`TestAlgorithmComparison`, `TestMapView`)**:
  - Comparison table output rendering.
  - Dual-algorithm interactive map generation with comparison HUD.
- **CLI & Core Integration (`TestMainCLIIntegration`, `TestStage2CLI`, `TestStage3CLI`, `TestGraph`, `TestDijkstra`, `TestDisplay`, `TestOSMLoader`)**:
  - `--algorithm astar` and `--algorithm dijkstra` execution.
  - `--compare` and `-c` side-by-side benchmark execution.
  - Fictional and real-world landmark and coordinate routing.
  - Backward compatibility across all previous stages.

```text
Ran 61 tests in 2.661s
OK
```

---

## 🔮 Future Roadmap

- **Stage 5 — Dynamic Traffic & Turn Restrictions**: Integrate live travel-time penalties, one-way turn restrictions, and traffic congestion data.
- **Stage 6 — Elevation & Multi-Modal Routing**: Incorporate digital elevation models (DEM) for gradient-weighted bike/pedestrian routing.