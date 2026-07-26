# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (default Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build (produces build/TritonRoute and build/libflexroute.a)
cmake --build build -j$(nproc)

# Debug build
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(nproc)
```

## Running

```bash
# Detailed routing with a guide file
./build/TritonRoute -lef <LEF_FILE> -def <DEF_FILE> -guide <GUIDE_FILE> -output <OUTPUT_DEF>

# Global-detailed routing (requires POST9.dat and POWV9.dat from src/gr/flute/ in run directory)
./build/TritonRoute -lef <LEF_FILE> -def <DEF_FILE> -output <OUTPUT_DEF>
```

Smoke-test inputs: LEF `~/benchmark/primarius/outdata/ispd18_test1.input.lef`, DEF `~/benchmark/primarius/outdata/pattern_route_lay.def`. Place all outputs (DEFs, guides, logs) under `build/`.

Default smoke-test command (run from `build/`, self-symmetry case):

```bash
./TritonRoute -lef ~/benchmark/primarius/case_0625/pattern_route0625.lef -def ~/benchmark/primarius/case_0625/pattern_route0625_v2.def -output route.def -drouteEndIterNum 6
```

## Architecture

`FlexRoute` (`src/FlexRoute.h/.cpp`) is the top-level orchestrator. Its `main()` calls the pipeline in order:

1. **IO** (`src/io/`) — reads LEF/DEF/guide via vendored parsers in `module/lef/` and `module/def/`; populates `frDesign`
2. **PA** (`src/pa/`) — pin access analysis; assigns access patterns to each pin
3. **RP** (`src/rp/`) — routing prep; builds net topology from access patterns
4. **GR** (`src/gr/`) — global routing on a coarse grid (`FlexGRCMap`); uses FLUTE (`src/gr/flute/`) for Steiner tree topology, then maze-routes on `FlexGRGridGraph`
5. **TA** (`src/ta/`) — track assignment; maps GR wire segments to specific routing tracks
6. **DR** (`src/dr/`) — detailed routing; maze-routes on `FlexGridGraph` within GCells, writes final wire geometry
7. **GC** (`src/gc/`) — geometry/DRC checking; validates spacing, width, and via rules

The design database lives in `src/db/` with parallel object hierarchies: `frObj` (final), `grObj` (global routing), `drObj` (detailed routing), `taObj` (track assignment), `gcObj` (geometry check). Technology data (`frTechObject`, `frLayer`, `frConstraint`) is in `src/db/tech/`. `frRegionQuery` wraps Boost R-tree spatial indices for all layers.

### Self-Symmetry Extension (current active work)

This branch adds support for self-symmetric net routing. The GR pipeline runs a scheduled 5-stage flow:

```
2D auto → 2D mirror → layerassign → 3D auto (OrdinaryOnly) → 3D mirror
```

`FlexGRSelfSymmetryMode` (`Auto` / `OrdinaryOnly` / `Mirror`) controls which nets participate in each search-repair pass. Cost logic lives in `FlexGRGridGraph::getNextPathCost()`. Full design rationale is in `docs/gr_self_symmetry_cost_design.md`.

Key invariants for self-symmetry work:
- `3d_auto` must be `OrdinaryOnly`; self-symmetric nets must not be touched until `3d_mirror`
- Mirror-mode invalid-mirror penalty must not apply to axis edges
- Both `grNet` and its owning `frNet` must be marked modified when a self-symmetric net enters the reroute queue
- Helper header: `src/gr/FlexGR_self_sym_utils.h`

## Coding Style

- Two-space indentation, braces on same line
- Namespace `fr` for all router code
- Class names: PascalCase (`FlexGRGridGraph`); functions/variables: lowerCamelCase
- When adding source files, update `CMakeLists.txt` explicitly
- Preserve BSD license headers in existing files

## Testing

No CTest suite. Validate routing behavior changes by rebuilding and running the default smoke-test command above (see Running), then comparing output DEF, guide files, DRC markers, and logs against a known baseline. For changes limited to declarations, helpers, or comments, a successful build is sufficient.
