# Custom Detailed Routing Notes

## Goal

Build a new `customdr` flow on top of this repository that:

- reuses pin access analysis from `PA`
- reads design and technology data from the existing database
- generates restricted-shape routes for selected nets
- writes routing results back to the database
- reuses the existing DRC checking flow

The intent is **not** to deeply integrate with the current `src/dr/` maze router.

## Recommended Architecture

Create a separate directory such as:

- `src/customdr/CustomDR.h/.cpp`
- `src/customdr/CustomDRWorker.h/.cpp`
- `src/customdr/CustomDRDB.h/.cpp`
- `src/customdr/CustomDRWriteback.h/.cpp`
- `src/customdr/CustomDRCheck.h/.cpp`

Suggested responsibilities:

- `CustomDRDB`: read `frDesign`, `frNet`, pins, APs, layers, vias, and region data
- `CustomDRWorker`: implement the custom routing algorithm
- `CustomDRWriteback`: convert custom route results into database objects
- `CustomDRCheck`: run legality and DRC checks using existing infrastructure

## Object Model Guidance

This codebase uses a shared base type system (`frBlockObject`, `frBlockObjectEnum`), but each stage has its own object family (`fr*`, `gr*`, `ta*`, `dr*`, `gc*`).

For `customdr`, prefer:

- input: `frDesign`, `frNet`, `frInstTerm`, `frTerm`, `frAccessPoint`
- output: `frPathSeg`, `frVia`, `frPatchWire`

Do **not** make `drNet` / `drAccessPattern` the main model unless you plan to reuse the existing detailed router internals. They are useful as references, but they carry DR-specific assumptions and lifecycle state.

A lightweight internal model is preferred, for example:

- `CustomRouteSegment`
- `CustomRouteVia`
- `CustomRoute`

## Pin Access

Reuse `PA` results instead of regenerating access candidates.

Useful sources:

- preferred APs: `frInstTerm->getAccessPoints()`
- full AP candidates: `pin->getPinAccess(inst->getPinAccessIdx())->getAccessPoints()`

Use preferred APs if your router wants a single anchor per pin. Use full AP candidates if your router wants multiple entry options.

## Writeback Strategy

Write final routing results directly into `frNet`:

- `frNet::addShape(...)`
- `frNet::addVia(...)`
- `frNet::addPatchWire(...)`

After writeback, update `frRegionQuery` so later geometry queries and DRC see the new objects:

- `design->getRegionQuery()->addDRObj(frShape*)`
- `design->getRegionQuery()->addDRObj(frVia*)`

If replacing existing routing, remove old routed objects from `frRegionQuery` before deleting or overwriting them.

## DRC Recommendation

Do not implement a separate rule checker.

Preferred choice:

- reuse `FlexGCWorker` for DRC/geometry checking

This is the closest path to the current detailed-routing legality flow and returns markers that are already understood by the repository.

Practical first version:

1. run `PA`
2. route selected nets in `customdr`
3. write results into `frNet`
4. update `frRegionQuery`
5. run `FlexGCWorker`
6. inspect markers and export DEF through `Writer::writeFromDR()`

## Development Order

Recommended implementation sequence:

1. Define a minimal internal route representation
2. Implement database readers for target nets and pin access
3. Implement one restricted-shape router, such as single-`L` or constrained `Z`
4. Implement writeback to `frNet`
5. Add `frRegionQuery` synchronization
6. Add `FlexGCWorker`-based DRC checking
7. Add fallback or rollback behavior if custom routing is illegal

## Practical Notes

- Keep `customdr` independent from `src/dr/` unless reuse is clearly beneficial.
- Reuse `drAccessPattern` only as a reference or adapter, not as the core model.
- Prefer a small end-to-end prototype over early optimization or incremental DRC.
- Start with a minimal reproducible testcase before running full designs.

## Current `src/cr` Status

The current `src/cr/` implementation has intentionally adopted only a subset of
the `dr` flow. The following are implemented:

- global Hanan axes in `crPatternGraph`, with only selected pattern points
  collected from the target nets and route/ext box bounds
- full grid instantiation in `crPatternGraph`; every valid
  `(xCoord, yCoord, routingLayer)` inside the graph dimensions is now treated
  as an available node, without a separate sparse node map
- policy-driven routing in `crPatternRouter`; current `L` policy enumerates
  candidate bend locations instead of running full maze expansion, and no
  longer keeps unused Dijkstra-style search state for `L`
- graph-owned planar quick-cost marking follows DR's typed cost update model:
  the same influence-region helper can update `DRCCost` or `ShapeCost`
  depending on caller context
- planar non-preferred-direction cost is derived on demand from layer preferred
  direction, following DR's "wrong-way is allowed but penalized" model instead
  of hard forbidding it
- graph-owned edge cost channels use DR-like normalized edge keys, so opposite
  directions share the same physical edge cost
- graph-owned quick-cost storage now uses one DR-like `bits` vector per grid
  node instead of separate cost vectors, with bit ranges aligned to DR's
  block/grid/DRC/marker/shape cost layout
- graph-owned planar `DRCCost` uses increment/decrement counting semantics,
  following the `dr` style of additive removable cost instead of bool/set
  marking
- graph-owned via `DRCCost` marks default-via placement points whose adjacent
  metal shape would short or violate the layer min-spacing table, following the
  same add/sub cost model as planar `DRCCost`
- EOL spacing graph cost marks planar and default-via candidate points in the
  EOL windows generated from routed metal and via enclosure rectangles
- graph-owned SVia table maps pin access maze points to their first one-cut
  access viaDef so quick DRC and writeback use the same special via footprint
- incremental `routeNet` flow:
  1. remove old route conn figs from pattern-graph cost
  2. search for a new path
  3. write the path back as `crPathSeg`
  4. add the new route conn figs back into pattern-graph cost
- planar `crPathSeg` writeback into the source `frNet`, with matching
  insertion/removal in the global `frRegionQuery`
- vertical `crVia` writeback into the source `frNet`, with matching
  insertion/removal in the global `frRegionQuery`
- centralized writeback flow in `CustomRouteWorker`: route search populates only
  local `crConnFig`s first, then a final end-stage removes old DB objects from
  `frRegionQuery`/`frNet` inside `routeBox` and writes back the new shapes/vias

The following parts were intentionally simplified and are not implemented yet:

- no incremental cost removal/addition for `crPatchWire`
  `crVia` is written back and participates in `addPathCost` / `subPathCost`,
  but `frPatchWire` generation is not added yet.
- limited producers for DR-like cost channels
  Current `cr` has DR-like quick-cost storage and router cost reads for
  grid/shape/DRC/marker/block channels, and the graph helpers support DR-like
  typed `DRCCost`/`ShapeCost` updates. Marker/block/guide-style producers are
  still not equivalent to DR.
- no cut-spacing, min-area, via2via forbidden length, or via-turn forbidden
  length in graph cost
  The current graph cost checks planar metal short/spacing, via adjacent-metal
  short/spacing, and EOL spacing windows with default-via plus pin-AP SVia
  footprints, but does not model the richer DR rule set.
- no history-cost / marker-cost flow like `dr`
  There is no equivalent of route-queue marker decay/addition.
- no full DRC-faithful geometry reasoning
  Current short/spacing checks use bbox-based approximations on existing
  objects queried from the region query.
- no patch-metal generation or post-search min-area repair
- no full rip-up/reroute lifecycle
  Current flow supports replacing customdr-owned planar `frPathSeg` writeback
  for the same local `crNet`; broader victim-net rip-up/requeue is not
  implemented.

## Working Notes

- Implemented: `CustomRouteWorker` now writes generated `frPathSeg`s into the
  source `frNet` and inserts/removes them from the global `frRegionQuery`.
- Implemented: `CustomRouteWorker` now writes vertical path transitions as
  `crVia`/`frVia`, using pin-AP one-cut viaDefs when available and otherwise
  falling back to the cut layer's default viaDef.
- Implemented: planar non-preferred-direction routing is now penalized in
  `crPatternRouter`, using a graph-owned edge-cost channel instead of hard
  blocking wrong-way edges.
- Implemented: `L` policy now enumerates candidate bend locations plus
  endpoint via transitions onto a shared routing layer, and scores those
  candidates directly instead of using maze search with turn penalties.
- Implemented: `crPatternGraph` no longer sparsely inserts nodes. The graph is
  now a full `xCoords x yCoords x zCoords` grid, and router traversal only
  checks `mazeIdx` validity.
- Implemented: `crPatternGraph` updates via `DRCCost` when a metal shape would
  conflict with a default-via metal footprint on the layer above or below,
  aligning the quick cost flow with DR's planar plus via min-spacing updates.
- Implemented: `crPatternGraph` updates EOL `DRCCost` for planar and default-via
  candidates from path segments and via enclosure rectangles, following DR's
  `modEolSpacingCost` structure.
- Implemented: `crPatternGraph` stores SVia access viaDefs keyed by lower-layer
  maze index. Via min-spacing, EOL via candidate filtering, and final via
  writeback prefer the SVia viaDef before falling back to default viaDefs.
- Implemented: `crPatternGraph` now stores quick costs in one DR-like `bits`
  vector, with block/grid/DRC/marker/shape fields mapped to the same bit ranges
  used by `FlexGridGraph`.
- Implemented: `crPatternRouter` now reads grid/shape/DRC/marker/block cost
  channels for planar and via segments while keeping the existing pattern-route
  search structure.
- Implemented: `crPatternGraph` quick-cost producers now follow DR's
  `type 0/1/2/3` update convention: sub/add `DRCCost` and sub/add `ShapeCost`
  share the same spacing/EOL/via influence helpers, with caller context
  selecting the target cost channel.
- Implemented: planar metal quick-cost marking now uses a DR-like
  candidate-width corner-to-box distance test instead of the earlier
  footprint-overlap versus spacing-window split.
- Implemented: CR DRC edge weighting now follows DR's length-scaled formula:
  affected planar/via edges charge `edgeLength * DRCCOST` instead of a fixed
  `CR_SPACING_DRC_PENALTY` per edge.
- Implemented: `src/cr` documentation comments now describe core class
  responsibilities, member ownership/back-pointers, function side effects, and
  algorithm flows for graph construction, pattern routing, quick-cost updates,
  and DB writeback.
- Implemented: writeback is centralized. `crNet` keeps only local
  `routeConnFigs`; end-stage cleanup now queries global `frRegionQuery` inside
  `routeBox` and removes old `frPathSeg`/`frVia`/`frPatchWire` for the routed
  target nets before adding new `frPathSeg`/`frVia`.
- Implemented: `crAccessPoint` now records CR-local owner context through
  non-owning `ownerNet` and `ownerTerm` pointers. `initNetTerm` fills these
  from the source `frNet` and source `frInstTerm`/`frTerm`, so later AP spatial
  queries can identify same-net APs and pin class without mutating PA's
  shared `frAccessPoint` objects.
- Implemented: `CustomRoute` now owns a lazy `crAPRegionQuery` that copies
  design PA access points into CR-local `crAccessPoint`s and indexes their
  transformed point by routing layer. The query returns `crAccessPoint*`
  only; AP direction, pin class, and cost interpretation remain consumer-side
  logic.
- Implemented: `crPatternGraph` now consumes the CR AP spatial query during
  quick-cost initialization. External macro/IO APs with planar E/W/N/S access
  add DR-like `GridCost` along a `10 * layer width` ray on existing graph
  coordinates, including local U/D grid cost at affected nodes, so per-net
  graphs can avoid other nets' APs without adding their AP coordinates.
- Simplification: writeback now covers `frPathSeg` and `frVia`; no
  `frPatchWire` generation is added yet.
- TODO: add DR-like stdcell U/off-track AP grid cost. Current AP cost covers
  macro/IO planar AP access only because `crAccessPoint` does not yet record
  `onTrackX/onTrackY`.
- TODO: add richer viaDef-aware graph cost beyond the first AP-provided one-cut
  candidate / default viaDef fallback.
