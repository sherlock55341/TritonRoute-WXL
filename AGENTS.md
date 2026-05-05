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
The `cr/` subsystem is a custom routing module that keeps a lightweight CR
object model while reusing PA access points, the shared `frDesign` database,
`frRegionQuery`, and `FlexGCWorker` checks. The current flow is per-net:
`CustomRoute::run()` starts one `CustomRouteWorker` for each pending source
`frNet`, and each worker builds its own routeBox/extBox/pattern graph.

### Key Types
- `crNet` (`src/cr/type/crNet.hpp`) mirrors the source `frNet` for one CR
  worker and owns local pins plus route result `crConnFig`s before writeback.
- `crPin` (`src/cr/type/crPin.hpp`) mirrors an instTerm/term and owns its local
  `crAccessPoint`s.
- `crAccessPoint` (`src/cr/type/crAccessPoint.hpp`) stores the transformed AP
  point, layer, derived maze index, directional access flags, access viaDefs,
  and non-owning owner context (`ownerNet`, `ownerTerm`) for same-net filtering.
- `crConnFig` (`src/cr/type/crFig.hpp`) is the base for local CR route geometry
  such as planar path segments and vias.

### Current Flow
1. Build a CR-local net model from the source `frNet` and PA APs.
2. Build a full `xCoords * yCoords * zCoords` pattern graph over selected Hanan
   and track coordinates in the worker region.
3. Backfill each AP's `mazeIdx` after graph construction.
4. Initialize DR-like graph cost channels, including shape/DRC/grid/AP
   influence costs from region-query and CR AP-query data.
5. Enumerate restricted pattern-route candidates. The current L router uses
   preferred horizontal and vertical layers separately and connects src, bend,
   and dst layer changes with via stacks.
6. Write local `crPathSeg`/`crVia` results back into the source `frNet` and keep
   `frRegionQuery` synchronized.
7. After all CR workers finish, run box-scoped `FlexGCWorker` checks over the
   worker extBoxes. If there are no CR tasks, skip post-CR DRC.

### Cost and Legality Notes
- Quick-cost storage uses one DR-like `bits` vector per grid node with
  block/grid/DRC/marker/shape fields aligned to `FlexGridGraph`.
- Quick-cost updates follow DR's `type 0/1/2/3` convention for sub/add
  `DRCCost` and sub/add `ShapeCost`.
- Use `DRCCost` for known short/spacing-style illegality pressure. Use
  `ShapeCost` for softer occupancy or influence pressure.
- Planar non-preferred direction is allowed but penalized. Non-zero L routes
  should not use one routing layer for both horizontal and vertical legs.

### Known Gaps
- No patch-wire generation, post-search min-area repair, history-cost flow, or
  full rip-up/reroute lifecycle yet.
- Graph cost still lacks several DR rule classes, including cut-spacing,
  min-area, via2via forbidden length, and via-turn forbidden length.
- AP avoidance currently handles macro/IO planar AP access; DR-like stdcell
  U/off-track AP grid cost remains a TODO.
- Post-CR DRC reports box-scoped violations but does not yet automatically
  reject, roll back, or repair illegal inter-net results.

For the latest working log and TODO list, read `CUSTOMDR_NOTES.md`.
