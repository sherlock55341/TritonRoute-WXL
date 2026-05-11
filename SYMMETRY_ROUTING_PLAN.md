# Symmetry Routing Plan

This note records the current demo plan for adding symmetry-aware routing to
TritonRoute-WXL. It is intentionally scoped to the first self-symmetric net
case and should be updated as the implementation changes. The current strategy
is to solve only one canonical side at each routing stage, then mirror the
result inside that same stage so the next stage consumes already-symmetric
data.

## Demo Scope

- Test input:
  - LEF: `/home/cyzhao/benchmark/primarius/outdata/ispd18_test1.input.lef`
  - DEF: `/home/cyzhao/benchmark/primarius/outdata/pattern_route_lay.def`
- First target net: `Symmtry5`
- Symmetry type: self-symmetric single net
- Symmetry axis: horizontal axis at `y = 71820` DBU
- Axis source: demo input / constraint interface, not hard-coded inside routing
  algorithms
- Constraint count supported for the first implementation: one input symmetry
  axis for one constrained net.
- Axis directions supported for this task: horizontal and vertical only
- Diagonal axes are out of scope
- Pin pairing tolerance: `min routing pitch / 2`
- `Symmtry5` appears to have paired pins across the axis, not a pin exactly on
  the axis.

## Confirmed Design Constraints

- Keep the implementation flow ordered from early routing stages to later ones:
  GR first, then TA, then DR.
- Do not implement a post-DR visual mirror pass as the demo proof. The demo
  should show that the algorithmic flow itself can preserve symmetry.
- For GR, TA, and DR, route or assign the canonical side first, then mirror the
  result to the opposite side before the stage writes back its output.
- Canonical side selection should be deterministic. For the first
  implementation:
  - horizontal axis: canonical side is `y >= axis`;
  - vertical axis: canonical side is `x >= axis`;
  - if the selected side has no pins or guides, use the opposite side.
- Do not enlarge TA panels or DR windows. Memory footprint is sensitive, so the
  symmetry flow must preserve existing panel/window sizes.
- For DR, the symmetry window should be started so that the axis lies in the
  middle GCell of the window, rather than expanding a normal window afterward.
- If a one-sided search naturally reaches the axis, use that connection. If it
  does not, run a supplemental search from the routed one-sided tree to the
  axis.
- TA and DR may insert temporary virtual symmetry tracks. The same axis and
  mirrored track coordinates inserted during TA should also be inserted during
  DR so DR can consume the TA result. This is acceptable for the demo even if a
  virtual track does not come from the original track pattern.

## Implementation Status

- `TritonRouteSymm` has a separate demo main and CMake target.
- `frDesign` now has a minimal in-memory symmetry constraint entry point.
- No GR/TA/DR routing behavior has been changed for symmetry yet.

## Symmetry Constraint Interface

Add a shared constraint representation that all stages can query:

```cpp
struct frSymmetryConstraint {
    frString netName;
    frSymmetryAxisEnum axisDir;  // horizontal or vertical axis
    frCoord axisCoord;           // axis coordinate in DBU
    frSymmetryCanonicalSideEnum canonicalSide;
};
```

The direction describes the axis orientation:

```text
frSymmetryAxisEnum::Horizontal -> horizontal axis, y = axisCoord
frSymmetryAxisEnum::Vertical   -> vertical axis, x = axisCoord
```

For the first demo, the source of this constraint is the `TritonRouteSymm`
initializer:

```text
netName = "Symmtry5"
axisDir = frSymmetryAxisEnum::Horizontal
axisCoord = 71820
canonicalSide = frSymmetryCanonicalSideEnum::High
```

The routing stages should query the design-level constraint interface instead
of embedding `Symmtry5` or `71820` inside core algorithms.

## GR Plan

Important correction: GR is not primarily DR-style window-based routing.
`FlexGR::main()` runs:

```text
init()
ra()
initGR()
searchRepair(...)
layerAssign()
searchRepair(...)
writeToGuide()
```

Relevant code locations:

- `src/gr/FlexGR.cpp`
  - `FlexGR::main()`
  - `FlexGR::initGR()`
  - `FlexGR::initGR_genTopology_net()`
  - `FlexGR::initGR_patternRoute()`
  - `FlexGR::patternRoute_LShape()`
  - `FlexGR::layerAssign_net()`
- `src/gr/FlexGR_maze.cpp`
  - `FlexGRWorker::routeNet()`

Planned GR behavior:

- Treat the axis at the GCell/topology level, not as a routing track. Routing
  tracks are TA/DR concepts.
- Keep the DBU-level axis coordinate in the shared constraint, but snap the GR
  interpretation to the GCell row or column containing that coordinate:
  - horizontal axis: `axisRowIdx = getGCellIdx((x, coord)).y()`;
  - vertical axis: `axisColIdx = getGCellIdx((coord, y)).x()`.
- Build the first symmetric GR topology with a GCell-level axis anchor:
  - split the net pins into canonical-side pins, mirror-side pins, and any
    on-axis pins;
  - choose one anchor GCell on the axis using the canonical-side pin GCell bbox
    center projected to the axis row or column;
  - use the axis anchor as the symmetry topology root;
  - run FLUTE on `canonical-side GCell nodes + axis anchor`;
  - mirror the resulting canonical-side FLUTE tree to the opposite side by GCell
    index;
  - connect the mirrored tree to the mirror-side pin GCell nodes;
  - keep the axis anchor as the shared connection between both sides.
- Use GCell-index mirroring in GR:
  - horizontal axis: `mirrorYIdx = 2 * axisRowIdx - yIdx`;
  - vertical axis: `mirrorXIdx = 2 * axisColIdx - xIdx`.
- Do not use the normal full-net FLUTE topology directly for the symmetry net.
  The symmetry net should call a symmetry-aware topology builder that reuses
  FLUTE only for the canonical side.
- Pairing should be found geometrically around the single input symmetry axis.
  Do not add complex GCell-pair metadata for the first implementation.
- Ensure any GR pattern route for the symmetry net creates paired geometry about
  the GCell-level axis. For L-shape pattern routing, choose the canonical-side
  corner once and mirror the route to the opposite side; do not let the mirror
  side independently choose its L-shape.
- Later GR search-repair must either skip the symmetry net or use a
  symmetry-aware reroute path; ordinary non-symmetric reroute should not break
  `Symmtry5`. For the first demo, skipping ordinary search-repair for the
  symmetry net is acceptable.

## TA Plan

TA is panel-based, not net-by-net maze routing.

Relevant code locations:

- `src/ta/FlexTA.cpp`
  - `FlexTA::initTA_helper()`
  - `FlexTA::initTA()`
  - `FlexTA::searchRepair()`
- `src/ta/FlexTA_init.cpp`
  - `FlexTAWorker::initTracks()`
  - `FlexTAWorker::initIroutes()`
  - `FlexTAWorker::initIroute()`
- `src/ta/FlexTA_assign.cpp`
  - `FlexTAWorker::assignIroute_availTracks()`
  - `FlexTAWorker::assignIroute_bestTrack()`
  - `FlexTAWorker::assignIroute_getCost()`
  - `FlexTAWorker::assignIroute_updateIroute()`
- `src/ta/FlexTA_end.cpp`
  - `FlexTAWorker::saveToGuides()`

Planned TA behavior:

- Do not enlarge panels.
- Pair the `Symmtry5` guides produced by GR.
- Assign or preserve the canonical-side guide first, then mirror the chosen
  route to its paired guide before writing back to `frGuide::routes`.
- For a horizontal axis:
  - A horizontal guide's selected track coordinate is `y`; its mirror track is
    `2 * axis - y`.
  - A vertical guide's selected track coordinate is `x`; its mirror keeps the
    same `x`.
- Insert required virtual symmetry tracks into the TA worker's local track list
  when the selected or mirrored coordinate is not present in the original track
  pattern. Keep the insertion local to the worker data structures.
- Use paired cost during track selection:

```text
cost(pair) = cost(guide, track) + cost(mirrorGuide, mirrorTrack)
```

- Write both assigned routes back to their guides together so TA does not break
  GR symmetry.
- If an axis guide exists, treat it as self-mirror and prefer/fix the axis track
  when appropriate.
- A complete paired-cost track search may require cross-panel coordination for
  horizontal guide pairs. For the first demo, it is acceptable to assign the
  canonical side and mirror the selected track if the mirrored track exists and
  is legal enough for DR to refine.

## DR Plan

DR is window-based and should enforce symmetry without increasing window size.

Relevant code locations:

- `src/dr/FlexDR.cpp`
  - `FlexDR::searchRepair()`
- `src/dr/FlexDR_init.cpp`
  - `FlexDRWorker::initNets()`
  - `FlexDRWorker::initTrackCoords()`
  - `FlexDRWorker::initTrackCoords_route()`
  - `FlexDRWorker::initTrackCoords_pin()`
  - `FlexDRWorker::initGridGraph()`
- `src/dr/FlexDR_maze.cpp`
  - `FlexDRWorker::routeNet()`
  - `FlexDRWorker::routeNet_postAstarWritePath()`
- `src/dr/FlexGridGraph_maze.cpp`
  - `FlexGridGraph::search()`
  - `FlexGridGraph::expand()`
  - `FlexGridGraph::getNextPathCost()`
- `src/dr/FlexGridGraph.cpp`
  - `FlexGridGraph::init()`
  - `FlexGridGraph::initTracks()`
  - `FlexGridGraph::initEdges()`

Planned DR behavior:

- Do not enlarge windows.
- For `Symmtry5`, use windows whose y start makes the horizontal axis lie in the
  middle GCell of the window. Existing DR sizes are odd in the current flow
  (`7`, `5`, `3`), which fits this requirement.
- Ordinary non-symmetric windows should not reroute `Symmtry5`; otherwise they
  can destroy symmetry.
- In a symmetry worker, inject track coordinates during grid setup:
  - add `y = axis` to all routing layers;
  - add mirror coordinates `mirrorY = 2 * axis - y` for relevant y coordinates;
  - add the mirror coordinates to all routing layers.
- Keep the full `drNet` representation for the constrained net. During
  symmetric routing, filter mirror-side pins from the connection target set so
  the A* search connects only the canonical side and the axis connection; write
  back the mirrored route afterward.
- Route only the canonical side first. During search, evaluate every canonical
  edge with its mirrored edge so the copied result remains legal:

```text
cost_sym(e) = cost(e) + cost(mirror(e))
```

- If `e` is self-mirror on the axis, count it once.
- If the mirror edge does not exist, treat it as unavailable or assign a very
  high cost. The axis-centered window plus mirror track injection should make
  this uncommon in the demo case.
- After the one-sided route is complete, check whether the route touches the
  axis. If not, run a supplemental search from the current tree to the axis.
- Write back the canonical path, mirrored path, and axis connection together.

## Implementation Order

1. Add the shared symmetry constraint interface and the demo constraint for
   `Symmtry5`.
2. Update GR to generate a GCell-level symmetric topology using an axis anchor,
   canonical-side FLUTE, and mirrored topology.
3. Update GR search-repair so ordinary workers skip the symmetry net until a
   symmetry-aware reroute path exists.
4. Update TA to assign canonical-side guide routes and mirror them to paired
   guides without increasing panel size.
5. Update DR to use axis-centered windows, axis/mirror track injection, and
   symmetric A* cost.

## Validation Notes

- After C++ changes, run `clangd --compile-commands-dir=build --check=<changed
  translation-unit>` where practical.
- Build `TritonRoute` after changes that affect headers, source membership,
  public interfaces, or final integration:

```bash
cmake --build build -j --target TritonRoute
```

- For routing validation, use the demo LEF/DEF listed above and inspect the
  generated guide/TA/DR results for `Symmtry5` symmetry about `y = 71820`.
