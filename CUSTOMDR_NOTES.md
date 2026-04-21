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
