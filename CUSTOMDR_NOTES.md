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
- graph-owned planar `DRCCost` marking for basic short/spacing influence
- planar non-preferred-direction cost is derived on demand from layer preferred
  direction, following DR's "wrong-way is allowed but penalized" model instead
  of hard forbidding it
- graph-owned edge cost channels use DR-like normalized edge keys, so opposite
  directions share the same physical edge cost
- graph-owned planar `DRCCost` uses increment/decrement counting semantics,
  following the `dr` style of additive removable cost instead of bool/set
  marking
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

- no incremental cost removal/addition for `crVia` or `crPatchWire`
  `crVia` is now written back, but only `crPathSeg` participates in
  `addPathCost` / `subPathCost`.
- no split cost channels like `dr` (`shapeCost`, `markerCost`, `blockCost`,
  `guideCost`)
  Current `cr` stores a planar `DRCCost` channel plus a planar non-pref
  penalty channel, but still does not model the other DR cost classes.
- no cut-spacing, EOL spacing, min-area, via2via forbidden length, or
  via-turn forbidden length in graph cost
  The current graph cost only checks basic planar short/spacing.
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
- Implemented: writeback is centralized. `crNet` keeps only local
  `routeConnFigs`; end-stage cleanup now queries global `frRegionQuery` inside
  `routeBox` and removes old `frPathSeg`/`frVia`/`frPatchWire` for the routed
  target nets before adding new `frPathSeg`/`frVia`.
- Simplification: writeback now covers `frPathSeg` and `frVia`; no
  `frPatchWire` generation is added yet.
- TODO: add via-aware graph cost and richer viaDef selection beyond the first
  AP-provided one-cut candidate / default viaDef fallback.
