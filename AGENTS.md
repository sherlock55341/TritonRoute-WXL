# Repository Guidelines

## Project Structure & Module Organization
Core router code lives in `src/`, organized by subsystem: `gr/` (global routing), `ta/` (track assignment), `dr/` (detailed routing), `gc/` and `drc/` (geometry and rule checking), `pa/` and `rp/` (pin access and repair), `io/` (LEF/DEF and guide parsing), and `db/` (shared data models). Third-party LEF/DEF parser sources are vendored under `module/def/5.8-p029` and `module/lef/5.8-p029`. Build artifacts should stay in `build/` and not be committed.

## Build, Test, and Development Commands
Configure an out-of-tree build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`. Compile with `cmake --build build -j`, which produces the `TritonRoute` executable. Run the router from the build directory or repo root, for example: `./build/TritonRoute -lef tech.lef -def design.def -guide route.guide -output routed.def`. Use `cmake --install build --prefix <dir>` only when you need staged `bin/` and `include/` outputs.

## Coding Style & Naming Conventions
This repository uses C++17 and `.clang-format` based on Google style with 4-space indentation and unsorted includes. Match surrounding conventions: class names such as `FlexDRWorker` use PascalCase, methods use lowerCamelCase, and filenames typically mirror the main class or subsystem (`FlexGR.cpp`, `io_parser_helper.cpp`). Prefer keeping new code within the existing subsystem directory instead of adding cross-cutting utilities at the top level.

## Testing Guidelines
There is no top-level `ctest` or dedicated unit-test target in the current CMake setup. Validate changes by rebuilding and running representative routing flows on known LEF/DEF inputs. For parser-related work, the vendored LEF/DEF modules include their own `TEST/` assets; use them for focused regression checks when touching `module/` or `src/io/`. Document the exact command and dataset used in your PR notes.

## Commit & Pull Request Guidelines
The visible git history is minimal, so use short, imperative commit subjects such as `Fix via enclosure check in FlexDR`. Keep commits focused on one subsystem or behavior change. Pull requests should describe the routing scenario affected, summarize user-visible impact, list validation commands, and attach before/after logs or output diffs when behavior changes.

## Configuration Notes
CMake requires GCC 7+ compatible C++17 support plus Boost, OpenMP, Bison, and zlib. For global-detailed routing runs, ensure `POST9.dat` and `POWV9.dat` from `src/gr/flute/` are available in the working directory.

## Custom Route Subsystem (`src/cr/`)

### Overview
The `cr/` subsystem is a custom routing module that operates on a single frNet at a time, without the routeBox/extBox region-partitioning used by FlexDR. Its entry point is `CustomRouteWorker`.

### Key Types
- `crNet` (`src/cr/type/crNet.hpp`) — mirrors `drNet`. Holds `crPin`s, `crConnFig`s, a `terms` set, and a back-pointer to the source `frNet`.
- `crPin` (`src/cr/type/crPin.hpp`) — mirrors `drPin`. Holds a `frBlockObject*` term reference and a vector of `crAccessPoint`s.
- `crAccessPoint` (`src/cr/type/crAccessPoint.hpp`) — mirrors `drAccessPattern`. Fields: `pt` (frPoint), `layerIdx`, `mazeIdx` (crMazeType), `validAccess` (6-bool directional flags E/S/W/N/U/D), `upViaDefs[2]`, `downViaDefs[2]`.
- `crConnFig` (`src/cr/type/crFig.hpp`) — base class for routing geometry results.

### Net Initialization (`crWorker.cpp`)
`initNet(frNet*)` builds a `crNet` from an `frNet` by iterating all instTerms and terms. For each terminal, `initNetTerm` creates a `crPin` and populates its `crAccessPoint`s from the frPin → frPinAccess → frAccessPoint chain, applying the instance transform (`shiftXform`). Unlike FlexDR, there is no routeBox filtering — all access points are included.

### Data Flow: frNet → crNet
```
frNet
├── frInstTerm[] / frTerm[]
│   └── frTerm::getPins() → frPin[]
│       └── frPin::getPinAccess(pinAccessIdx) → frPinAccess
│           └── frPinAccess::getAccessPoints() → frAccessPoint[]
│               ├── point (transformed by instance shiftXform)
│               ├── layerNum
│               ├── accesses[6] (directional flags)
│               └── viaDefs[][] (by cut number)
↓
crNet
├── crPin[] (one per frInstTerm/frTerm)
│   └── crAccessPoint[]
│       ├── pt (transformed)
│       ├── layerIdx
│       ├── validAccess[6]
│       ├── upViaDefs[2], downViaDefs[2]
│       └── mazeIdx (set later, after gridGraph construction)
```

### GridGraph Construction (planned)
The `mazeIdx` field on `crAccessPoint` is not set during `initNet`. It requires a gridGraph to be built first. The DR flow for reference:

1. **Collect coordinates** — from pin access points and existing routing endpoints into `xMap`/`yMap` (keyed by physical coord, valued by `{layerNum → trackPattern*}`). Each access point adds its coord to the pref-dir map of its layer and the adjacent layer (±2).
2. **Add track coordinates** — iterate design trackPatterns within the bbox, adding all track locations to `xMap`/`yMap`.
3. **Build grid** (`initGrids`) — flatten `xMap`/`yMap`/`zMap` keys into `xCoords[]`/`yCoords[]`/`zCoords[]` arrays; allocate bit arrays sized xDim × yDim × zDim for costs, A* state, src/dst markers.
4. **Build edges** (`initEdges`) — for each grid node, determine E/N/U edge existence and cost based on trackPattern and DRC rules.
5. **Backfill mazeIdx** (`initMazeIdx_ap`) — for each access point, map `(pt, layerNum)` → grid index via `gridGraph.getMazeIdx()`.

For CustomRouteWorker, the bbox can be derived from the bounding box of all pin access points (with margin), rather than a pre-assigned routeBox.

### Region Query and Existing Routing
frNet shapes (pathSeg, via) are loaded into the global `frRegionQuery` R-tree during `frRegionQuery::initDRObj()`. FlexDRWorker queries this R-tree to discover existing routing within its work area. CustomRouteWorker can query the same R-tree if it needs to be aware of existing routing (e.g., for rip-up reroute), but does not need to for fresh routing.
