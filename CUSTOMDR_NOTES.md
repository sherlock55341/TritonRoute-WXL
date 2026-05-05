# Custom Route Working Notes

Last updated: 2026-05-05

This file is the repo-local continuation log for the `src/cr` custom routing
work. Keep it concise and operational so another agent can resume after context
compaction without rereading the full conversation.

## Purpose

`src/cr` is a custom detailed-routing path that:

- reuses PA access points and the existing `frDesign` database
- routes selected nets with restricted pattern-routing logic
- writes CR-owned route objects back to `frNet`
- keeps the global `frRegionQuery` synchronized
- reuses `FlexGCWorker` for DRC checking

The intent is still to keep CR independent from the full `src/dr` maze-router
lifecycle while aligning selected data structures, cost semantics, and legality
checks with DR where that gives clear behavior.

## Current Architecture

- Entry flow: `CustomRoute::run()` owns the task list, AP query, worker launch,
  and post-CR box-scoped DRC.
- Worker unit: each `CustomRouteWorker` is started for exactly one source
  `frNet`, builds that net's routeBox/extBox/pattern graph, routes, and writes
  the result before the next worker starts.
- Local model: `crNet`, `crPin`, `crAccessPoint`, and `crConnFig` mirror the
  DR object-family pattern without making `drNet` or `drAccessPattern` the CR
  core model.
- Graph model: `crPatternGraph` builds a full `xCoords * yCoords * zCoords`
  grid over selected Hanan/track coordinates and routing layers. Router code
  checks `crMazeIdx` validity instead of maintaining a separate sparse node map.
- Routing model: `crPatternRouter` currently implements restricted pattern
  routing. The `L` policy scores enumerated candidates rather than running full
  Dijkstra/A* expansion.
- Writeback model: route search creates local `crConnFig`s first. End-stage
  cleanup removes old target-net `frPathSeg`/`frVia`/`frPatchWire` objects
  inside the routeBox from `frNet` and `frRegionQuery`, then writes new
  `frPathSeg`/`frVia` objects and inserts them into `frRegionQuery`.

## Implemented Behavior

### DR-Like Cost Storage

- Quick costs are stored in one DR-like `bits` vector per grid node.
- Block, grid, DRC, marker, and shape fields are mapped to the same bit ranges
  used by `FlexGridGraph`.
- Edge keys are normalized, so opposite directions share the same physical edge
  cost.
- Quick-cost producers use DR's `type 0/1/2/3` convention:
  sub/add `DRCCost`, then sub/add `ShapeCost`.
- Router scoring reads grid, shape, DRC, marker, and block channels for planar
  and via segments.
- DRC edge weighting follows DR's length-scaled formula:
  affected planar/via edges charge `edgeLength * DRCCOST`.

### Shape, DRC, and Quick Legality Cost

- Planar metal quick-cost marking uses a DR-like candidate-width
  corner-to-box distance test.
- Planar `DRCCost` is additive/removable rather than bool/set marked.
- Via `DRCCost` marks default-via placement points whose adjacent metal
  footprint would short or violate min-spacing.
- EOL spacing cost marks planar and default-via candidates from routed metal
  and via enclosure rectangles.
- SVia data maps pin-access maze points to the first AP-provided one-cut
  access viaDef, and quick DRC/writeback prefer that viaDef before falling back
  to default viaDefs.

### Preferred-Direction Pattern Routing

- Planar non-preferred direction is still allowed but penalized through graph
  cost.
- Non-zero L candidates now split horizontal and vertical legs across separate
  preferred-direction routing layers.
- Source, bend, and destination layer changes are connected with via stacks, so
  the src/dst APs connect to the line segments through actual vertical
  transitions.
- A non-zero L candidate should not use one routing layer for both horizontal
  and vertical segments.

### Access Point Context and Avoidance

- `crAccessPoint` stores CR-local owner context with non-owning `ownerNet` and
  `ownerTerm` pointers.
- AP coordinates follow the DR/PA shift-only convention: PA access points for
  unique instances are already orientation-transformed, so CR adds the
  instance/top-level shift instead of applying a full transform again.
- `CustomRoute` owns a lazy `crAPRegionQuery` that copies PA APs into CR-local
  `crAccessPoint`s and indexes transformed AP points by routing layer.
- `crPatternGraph` consumes the AP query during quick-cost initialization.
  External macro/IO APs with planar E/W/N/S access add DR-like `GridCost` along
  a `10 * layer width` ray on existing graph coordinates, including local U/D
  grid cost at affected nodes.
- Same-net APs are skipped when adding AP avoidance cost for a worker's graph.

### DRC Flow

- After all pending per-net workers finish, `CustomRoute` runs `FlexGCWorker`
  only over each worker extBox.
- Stale markers inside checked boxes are replaced, and CR prints one
  box-scoped violation count per checked box.
- If the CR task list is empty, post-CR DRC is skipped so a no-op CR invocation
  does not clear or rewrite existing top-level markers.

### Documentation

- `src/cr` comments were expanded to describe core class responsibilities,
  member ownership/back-pointers, function side effects, and algorithm phases
  for graph construction, pattern routing, quick-cost updates, and DB
  writeback.

## Design Decisions and Assumptions

- CR should borrow DR semantics where useful, but not inherit the whole DR
  worker lifecycle unless a concrete need appears.
- Per-net graph construction is the default. Other nets are visible through
  region-query-derived cost and the CR AP spatial query, not by adding every
  other net's AP coordinate to the graph.
- `ShapeCost` and `DRCCost` are separate channels. Use `DRCCost` for known
  short/spacing-style illegality pressure. Use `ShapeCost` for softer
  occupancy/influence pressure when the geometry is not being classified as a
  direct rule violation.
- AP context is stored in CR-local `crAccessPoint` objects, not in PA's shared
  `frAccessPoint` objects.
- Final DRC currently reports box-scoped violations after CR. It does not yet
  drive victim-net rip-up, rollback, or repair routing.

## Known Simplifications and Risks

- No `frPatchWire` generation, post-search min-area repair, or patch-metal
  cost/writeback flow yet.
- No cut-spacing, min-area, via2via forbidden length, or via-turn forbidden
  length modeling in graph cost yet.
- Marker/block/guide-style producers are not equivalent to DR.
- Quick DRC is bbox/influence based and is not a full DRC-faithful geometry
  proof.
- There is no DR-like history-cost/marker-decay route queue.
- There is no full rip-up/reroute lifecycle. Inter-net conflicts may be
  discouraged by cost and reported by post-CR DRC, but CR does not yet repair
  them automatically.
- AP avoidance currently covers macro/IO planar AP access. DR-like stdcell
  U/off-track AP grid cost is still missing because CR APs do not yet record
  `onTrackX`/`onTrackY`.
- Via cost/writeback uses the first AP-provided one-cut viaDef or a default
  viaDef fallback. Richer viaDef-aware graph cost is still pending.

## Next TODO

1. Validate the preferred-layer L enumeration on a small routed testcase and
   confirm Metal3 does not receive both horizontal and vertical non-zero L legs
   from new CR output.
2. Add DR-like stdcell U/off-track AP grid cost by copying `onTrackX` and
   `onTrackY`-equivalent context into `crAccessPoint` and extending
   `initAPCost()`.
3. Improve viaDef-aware quick cost beyond the first AP SVia/default-via
   fallback.
4. Decide whether CR should add DRC-clean candidate rejection, rollback, or
   rip-up/repair when post-CR box DRC finds inter-net violations.
5. Add patch-wire/min-area support if generated route geometry needs it.
6. Create or identify a minimal regression dataset for CR behavior checks.

## Validation Log

- 2026-05-05: Recent C++ validation from the CR preferred-layer L-routing work:
  `clang-format` on `src/cr/route/crPatternRouter.cpp` and `.hpp`;
  `clangd --compile-commands-dir=build --check=src/cr/route/crPatternRouter.cpp`
  with only known check-mode tweak noise; `cmake --build build -j --target
  CustomRoute`; `git diff --check -- src/cr/route/crPatternRouter.cpp
  src/cr/route/crPatternRouter.hpp CUSTOMDR_NOTES.md`.
- 2026-05-05: This document was reorganized from the earlier mixed architecture
  proposal plus append-only working notes. No C++ behavior was changed by this
  documentation cleanup.

## Historical Context

Early notes proposed a separate `src/customdr` prototype with `CustomDRDB`,
`CustomDRWorker`, `CustomDRWriteback`, and `CustomDRCheck`. The implementation
has instead evolved inside `src/cr`, using the existing CR type names and
subsystem layout. Treat the old `src/customdr` directory proposal as historical
background, not as the current target structure.
