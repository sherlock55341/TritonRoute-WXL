# Self-Symmetry Lead/Axis Search Repair With Mirror Shadow Demand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GR self-symmetry nets route and repair only the lead/axis source route, while mirror-side effects are represented only as shadow demand/cost derived from that source route.

**Architecture:** Search repair stores real GR objects only for the lead side and the required symmetry-axis connection. Mirror side has no `grPathSeg`, `grVia`, `frNode`, or region-query object during repair; it is recomputed from the current lead route whenever demand or cost is needed. The lead route must connect to an axis anchor, and candidate lead edges include mirrored congestion cost so mirror-side pressure influences lead choices.

**Tech Stack:** C++17, existing TritonRoute-WXL GR data model, `FlexGR`, `FlexGRWorker`, `FlexGRGridGraph`, `FlexGRCMap`, CMake.

---

## Updated Design Decisions

- Search repair source of truth for a self-symmetry net is `lead route + axis anchor`.
- Lead/source side is the current topology root side. If the root is on the axis, keep the existing fallback `rootSide = -1`.
- Do not introduce a source-side enum, and do not add side fields to nodes or route objects.
- Constraint access from a net uses a nullable pointer; pure geometry helpers take `const frSelfSymmetryConstraint&`.
- Mirror side is not a stored route tree and is not a stored GR object set during search repair.
- Mirror side is represented by shadow demand in `cmap` / `cmap2D` and by mirrored-edge cost during lead A*.
- Lead topology has a hard constraint to connect to the symmetry axis.
- Axis geometry is real and counted once. Non-axis lead geometry is mirrored into shadow demand.
- Other nets see mirror pressure through congestion maps, not through region-query objects.
- If final guide/DEF/DR requires explicit mirror shapes, materialize them after search repair from the final lead route.

## Task 3 Revision: One-Side Source Tree

Task 3 is intentionally narrower than the full lead/mirror-shadow architecture:

- Implement only the root/lead side + axis anchor parent-child tree.
- Keep mirror-side pin nodes in the net but disconnected.
- Do not add mirror A* cost, worker repair, writeback, layer-assignment specialization, or final mirror materialization in this task.
- Treat `Symmtry5` debug output plus the local smoke run as the Task 3 acceptance evidence.
- Guide connectivity validation is limited to connected source pins for self-symmetry nets during this phase; disconnected mirror pins are expected until final materialization exists.

## Code Review Baseline

Current GR flow in `src/gr/FlexGR.cpp`:
- `FlexGR::main()` runs `init()`, `ra()`, `initGR()`, 2D region query init, macro repair windows, three ordinary 2D `searchRepair()` passes, `layerAssign()`, 3D region query init, one 3D `searchRepair()`, then guide writing.
- Ordinary 2D search repair windows are `size=200`; 3D search repair windows are `size=10`.
- Ordinary worker grid windows use `i += size`, `j += size`, with `gcellIdxLL=(i,j)` and `gcellIdxUR=(min(xgp.getCount()-1,i+size-1), min(ygp.getCount(),j+size-1))`. The Y upper bound uses `ygp.getCount()` instead of `ygp.getCount()-1`; fix this separately or during validation.

Current self-symmetry topology:
- `FlexGR::initGR_genTopology()` dispatches `frcSelfSymmetry` nets to `initGR_genTopology_selfsymmetry_net()`.
- `initGR_genTopology_selfsymmetry_net()` assigns pin AP locations, identifies root/root side, creates GCell nodes, computes an axis GCell, and calls root/opposite side topology helpers.
- `genSelfSymmetryRootSideTopology()` already forces the root-side tree to touch the axis if it does not naturally do so. This behavior must become a hard invariant.
- The reviewed code computes and dumps root/opposite vertices and edges, but the self-symmetry edge result is not yet fully used as the final source route tree. Implementation must make the lead-to-axis result become the real tree.

Current search repair:
- `FlexGRWorker::initNets_roots()` queries global region query inside `routeBox` and creates worker subnets from any touched global `grPathSeg`/`grVia`.
- `FlexGRGridGraph::getNextPathCost()` uses local raw demand/supply/history/block cost only; it has no net-specific or mirror-aware hook.
- `FlexGRWorker::end()` removes and writes back routeBox-local GR objects. Mirror shadow demand outside the worker window will need explicit cmap updates because there are no mirror objects to remove from region query.

## File Structure

Modify:
- `src/db/obj/frNet.h`: add nullable `getSelfSymmetryConstraintPtr()` access for self-symmetry nets.
- `src/db/grObj/grNet.h`: mark worker subnets as self-symmetry lead-editable subnets.
- `src/gr/FlexGR.h`: declare mirror geometry, axis classification, shadow demand, and mirror-cost helper APIs.
- `src/gr/FlexGR.cpp`: update self-symmetry topology, pattern route, layer assignment, guide/materialization behavior.
- `src/gr/FlexGR_topo.cpp`: keep root-side-to-axis topology generation; stop using opposite-side topology for repair source route.
- `src/gr/FlexGR_init.cpp`: ensure workers only initialize from lead/axis real objects.
- `src/gr/FlexGR_maze.cpp`: route only lead/axis terminals, update lead demand plus mirror shadow demand.
- `src/gr/FlexGRGridGraph.h`: add active-net context for mirror-cost evaluation.
- `src/gr/FlexGRGridGraph_maze.cpp`: add mirror-shadow cost term in `getNextPathCost()`.
- `src/gr/FlexGR_end.cpp`: update source route objects and shadow demand consistently at writeback.

Optional create:
- `src/gr/FlexGR_selfsym.cpp`: recommended if helper code grows beyond a few small functions. Add it to `FLEXROUTE_SRC` in `CMakeLists.txt`.

## Implementation Tasks

### Task 1: Root-Side Geometry Helpers

**Files:**
- Modify: `src/db/obj/frNet.h`
- Modify: `src/gr/FlexGR.h`
- Modify: `src/gr/FlexGR.cpp`

- [x] Add nullable constraint access on `frNet` as the first code change:

```cpp
    const frSelfSymmetryConstraint* getSelfSymmetryConstraintPtr() const {
      return constraint == frNetRoutingConstraint::frcSelfSymmetry ? &selfSymmetryConstraint : nullptr;
    }
```

- [x] Do not modify `src/frBaseTypes.h`, node/object side fields, `CMakeLists.txt`, or routing behavior in this task.

- [x] Add helper declarations:

```cpp
    const frSelfSymmetryConstraint* getSelfSymmetryConstraintPtr(const frNet* net) const;
    bool isSelfSymmetryNet(const frNet* net) const;
    int getSelfSymmetryPointSide(const frPoint&, const frSelfSymmetryConstraint&) const;
    int getSelfSymmetryGCellSide(const frPoint&, bool isAxisHorizontal, frCoord axisGCellIdx) const;
    int getSelfSymmetryRootSide(frNet*, const frSelfSymmetryConstraint&) const;
    bool isOnSelfSymmetryAxis(const frPoint&, const frSelfSymmetryConstraint&) const;
    bool isOnSelfSymmetryAxisGCell(const frPoint&, bool isAxisHorizontal, frCoord axisGCellIdx) const;
    frPoint mirrorPoint(const frPoint&, const frSelfSymmetryConstraint&) const;
    frPoint mirrorGCellIdx(const frPoint&, bool isAxisHorizontal, frCoord axisGCellIdx) const;
```

- [x] Implement side helpers as coordinate-derived utilities. Point/GCell side returns `-1`, `0`, or `1`; `getSelfSymmetryRootSide()` maps root-on-axis side `0` to the existing fallback `-1`.

- [x] Build check:

```bash
cmake --build build -j$(nproc)
```

Expected: build succeeds without behavior change.

### Task 2: Add Mirror Shadow Demand Helpers

**Files:**
- Modify: `src/gr/FlexGR.h`
- Modify: `src/gr/FlexGR.cpp` or `src/gr/FlexGR_selfsym.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`

- [x] Add helpers to add/sub demand for a source path segment:

```cpp
    void modSelfSymmetrySourceDemand(frNet* net,
                                     const frPoint &begin,
                                     const frPoint &end,
                                     frLayerNum layerNum,
                                     bool isAdd,
                                     bool is2D);
    void modSelfSymmetryMirrorShadowDemand(frNet* net,
                                           const frPoint &begin,
                                           const frPoint &end,
                                           frLayerNum layerNum,
                                           bool isAdd,
                                           bool is2D);
```

- [x] `modSelfSymmetrySourceDemand()` updates normal lead/axis demand.

- [x] `modSelfSymmetryMirrorShadowDemand()` mirrors only non-axis source geometry and updates cmap/cmap2D at the mirrored edge. Axis geometry is skipped to avoid double count.

- [x] Add one wrapper:

```cpp
    void modSelfSymmetrySourceAndShadowDemand(frNet* net,
                                              const frPoint &begin,
                                              const frPoint &end,
                                              frLayerNum layerNum,
                                              bool isAdd,
                                              bool is2D);
```

It calls source demand plus mirror shadow demand.

- [x] Build check:

```bash
cmake --build build -j$(nproc)
```

Expected: build succeeds.

### Task 3: One-Side Parent-Child Tree, Debug Output, And Docs

**Files:**
- Modify: `src/gr/FlexGR.cpp`
- Modify: `src/io/io_guide.cpp`
- Modify: `docs/self_symmetry_lead_mirror_search_repair.md`
- Modify: `docs/superpowers/plans/2026-05-26-self-symmetry-lead-mirror-search-repair.md`

- [x] In `initGR_genTopology_selfsymmetry_net()`, keep AP assignment, root selection, axis GCell computation, and root-side selection.

- [x] Treat the axis GCell as mandatory source topology. Keep the existing `genSelfSymmetryRootSideTopology()` behavior that connects the root-side tree to the axis if needed.

- [x] Stop using `genSelfSymmetryOppositeSideTopology()` as a repair source route.

- [x] Convert `rootSideTreeVertices/rootSideTreeEdges` into a real `frNode` source tree:
  - Create root/lead terminal GCell nodes only for source-side pins.
  - Create Steiner nodes for non-terminal root-side vertices and axis anchors.
  - Parent the tree from the root GCell node.

- [x] Do not create mirror-side route nodes or mirror-side route objects in this phase.

- [x] Mirror-side original pin nodes remain as logical net pins but do not become source-tree terminals. Their physical mirror connection is implicit and can be materialized after repair if output requires it.

- [x] Add debug/error checks:
  - The root-side source tree reaches the axis anchor.
  - Every source-side non-root pin has a parent.
  - Mirror-side pins stay disconnected.
  - Source tree nodes stay on root/axis side.

- [x] Add `Symmtry5` debug dump with `pins`, `root-side terminals`, `root-side vertices`, `root-side edges`, `source tree parent-child`, and `root-side reaches axis`.

- [x] Build and smoke run:

```bash
clangd --compile-commands-dir=build --check=src/gr/FlexGR.cpp
cmake --build build -j$(nproc) --target flexroutelib
cmake --build build -j$(nproc) --target TritonRoute
mkdir -p build/selfsym-task3-tree
cd build/selfsym-task3-tree
cp ../../src/gr/flute/POST9.dat ../../src/gr/flute/POWV9.dat .
../TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_task3_tree.def \
  > selfsym_task3_tree.log 2>&1
rg -n "self-symmetry topology|axis:|pins:|source tree parent-child|root-side reaches axis" selfsym_task3_tree.log
```

Expected: `Symmtry5` dump appears, mirror-side pins show `in_source_tree=0` and `parent=null`, root/lead/axis side has a parent-child tree, `root-side reaches axis: 1`, and the smoke run exits successfully.

## Progress Update: 2026-06-01

Code has moved beyond the original Task 3 checkpoint. The implemented flow is
now GR-side self-symmetry support plus downstream guide/TA/DR consumption:

- Root/lead/axis topology is implemented and checked by the `Symmtry5` dump.
- 2D search repair updates lead source demand and mirror shadow demand in the
  worker grid, and A* includes mirror-edge cost for self-symmetry nets.
- Ordinary 2D repair is followed by `searchRepairSelfSymmetryMirror()`, which
  materializes mirror-side `frNode` topology before `layerAssign()`.
- `layerAssign()` includes a self-symmetry mirror-cost hook.
- 3D has a lead-only staging pass, mirror restore, and guided 3D repair pass.
- A current smoke run in `build/selfsym-status-read/selfsym_status.log` reaches
  detailed routing with `number of violations = 0`.

Important scope correction:

- Pattern route is not the practical self-symmetry choice point. The
  self-symmetry topology helpers produce rectilinear Hanan edges, while
  `initGR_patternRoute_init()` only collects non-colinear Steiner-Steiner edges.
  In the normal self-symmetry flow there is no two-L-shape candidate choice in
  `patternRoute_LShape()`.
- Detailed routing runs after generated guides, but DR is not yet
  self-symmetry-aware. It consumes guides and can complete successfully, but it
  does not enforce mirror-pair reroute, symmetric via/track selection, or a final
  detailed-route symmetry check.
- TA output is not ignored: `FlexTAWorker::saveToGuides()` writes TA pathSegs
  into `frGuide::routes`. DR currently reads those routes for
  `initGCell2BoundaryPin()` and can use original guides as guide cost during
  follow-guide search repair, but that is still ordinary DR behavior rather than
  a self-symmetry contract.

## Detailed Routing Follow-up Milestones

The next implementation track is documented in
`docs/self_symmetry_detailed_routing.md`. The fixed top-level strategy is:

```text
route all shaped self-symmetry nets first
-> check and repair self-symmetry-owned problems
-> freeze self-symmetry nets
-> route ordinary nets around the frozen self-symmetry shapes
```

Ordinary nets must yield to self-symmetry nets. Ordinary DR must not rip up a
self-symmetry net after the self-symmetry phase has completed. The target is
best-effort detailed-route symmetry under DRC/legal constraints, not strict
mirroring at the cost of legality.

### M0: Documentation State Lock

**Status:** Complete as documentation-only work.

Acceptance:
- GR-side self-symmetry status is recorded up to guide/TA/DR consumption.
- Pattern route is documented as not normally having two self-symmetry choices.
- The TA-to-DR `frGuide::routes` handoff is documented.
- The current DR gap is explicit: no lead/mirror pair routing, symmetric
  via/track binding, post-DR symmetry checker, or freeze-before-ordinary-DR
  policy exists yet.

### M1: DR Observation And Checker

**Status:** Planned. Observation only; no routing-result changes.

Implement:
- Log, per `frcSelfSymmetry` net, how many TA guide route objects DR sees in
  `frGuide::routes` during DR initialization.
- Log boundary-pin source/count for self-symmetry nets.
- Log whether each self-symmetry net is initialized as an ordinary `drNet`.
- Add a post-DR symmetry checker that reports only.

Checker should report:
- Missing mirrored counterpart for detailed segments.
- Missing mirrored counterpart for vias.
- Duplicate axis shapes.
- Markers owned by self-symmetry nets.
- Whether ordinary DR changed a self-symmetry net after the self-symmetry phase
  once freezing exists.

Acceptance:
- Logs prove TA routes are visible to DR for self-symmetry nets.
- Logs identify boundary pin counts and ordinary `drNet` initialization.
- Final report exists even if it reports current DR asymmetry.

### M2: Dedicated Self-symmetry DR Phase

**Status:** Planned.

Implement:
- Before ordinary DR, collect all nets with `frcSelfSymmetry`.
- Route only those self-symmetry signal nets in the self-symmetry phase.
- In this phase, avoid fixed objects, PG, OBS, pins, blockages, and other fixed
  constraints. Ordinary signal nets are not active yet.
- At phase end, self-symmetry nets must already own detailed shapes/vias.

Acceptance:
- Self-symmetry detailed route exists before ordinary DR starts.
- Ordinary nets are not enqueued during the self-symmetry phase.
- The checker can inspect self-symmetry detailed shapes/vias immediately after
  the phase.

### M3: Lead-side Plus Mirror-guide DR

**Status:** Planned.

Implement:
- Use TA `frGuide::routes` as one input skeleton for self-symmetry DR.
- Classify guides, pins, and shapes into lead, axis, and mirror by symmetry
  axis.
- Route lead/axis first.
- Generate mirror-side guide from the resulting lead detailed route.
- Route mirror side with the mirrored guide as a strong soft preference.
- Allow mirror-side deviations for DRC/legal cost.
- Keep axis shape once only.

Acceptance:
- Logs show lead pass, mirror-guide generation, and mirror pass.
- Mirror-guide hit/miss statistics are emitted.
- Self-symmetry-owned DRC is handled within this phase to an acceptable state.
- Final checker reports detailed-route symmetry deviations.

### M4: Freeze Self-symmetry Nets

**Status:** Planned.

Implement:
- Write self-symmetry shapes, vias, and patch wires back to the design.
- Update DR region query.
- Do not enqueue self-symmetry nets in ordinary DR.
- Do not rip up self-symmetry nets in ordinary DR.
- Treat frozen self-symmetry geometry as fixed obstacles for ordinary nets.

Acceptance:
- Ordinary DR queue contains no self-symmetry nets.
- Ordinary nets see self-symmetry shapes/vias as obstacles.
- Markers involving both self-symmetry and ordinary nets reroute ordinary nets
  only.

### M5: Ordinary Nets Route Around Frozen Self-symmetry

**Status:** Planned.

Implement:
- Run existing ordinary DR search repair for ordinary nets.
- Require ordinary nets to route around frozen self-symmetry geometry.
- If ordinary nets cannot route legally, report ordinary routing failure.
- Do not unfreeze or rip up self-symmetry nets for ordinary-net failures.

Acceptance:
- Self-symmetry net geometry is unchanged before/after ordinary DR.
- Ordinary nets route around frozen self-symmetry shapes/vias.
- Final DRC and final detailed-route symmetry checker reports are both emitted.

Milestone verification command pattern:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-dr
cd build/selfsym-dr
cp ../../src/gr/flute/POST9.dat ../../src/gr/flute/POWV9.dat .
../TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output <stage>.def \
  > <stage>.log 2>&1
```

### Task 4: Pattern Route Only Lead-to-Axis Source Geometry

**Status:** Superseded / not the active implementation path.

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [x] Re-reviewed `initGR_patternRoute_init()` and `patternRoute_LShape()`: current
  self-symmetry topology normally does not enter pattern route because the
  topology helpers already emit rectilinear Hanan edges.

- [x] Do not treat `patternRoute_LShape()` as the main self-symmetry route-choice
  hook. Initial self-symmetry choice belongs in Hanan topology cost; repair
  choice belongs in 2D/3D A* cost.

- [ ] Optional hardening: add a debug/assert path if a self-symmetry net ever
  produces a non-colinear Steiner-Steiner edge that reaches `patternRoute_LShape()`.

- [x] Current smoke evidence:

```bash
cmake -S . -B build/status-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
mkdir -p build/selfsym-status-read
cd build/selfsym-status-read
../status-cmake/TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_status.def \
  > selfsym_status.log 2>&1
```

Observed: flow completes through detail routing; pattern route is not the
self-symmetry selection point.

### Task 5: Keep Worker Init Lead/Axis-Only

**Status:** Partially implemented without adding a persistent `grNet` side flag.

**Files:**
- Modify: `src/gr/FlexGR_init.cpp`
- Modify: `src/db/grObj/grNet.h`

- [ ] The originally proposed `grNet` flag was not added:

```cpp
    bool isSelfSymmetryLeadEditable() const;
    void setSelfSymmetryLeadEditable(bool in);
```

- [x] Ordinary 2D repair runs before mirror materialization, so mirror-side
  route objects are absent from region query during ordinary 2D worker repair.
  Mirror side is represented through shadow demand/cost instead.

- [x] Worker logic has self-symmetry axis handling, including route-box axis
  detection, frozen boundary-axis endpoints, and forced axis endpoint insertion
  when the axis is inside the worker box.

- [ ] General proof/debug checking that every self-symmetry worker subnet is
  lead/axis-only across all windows is still missing.

- [x] Current build check:

```bash
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
```

Observed: build succeeds.

### Task 6: Add Mirror Shadow Cost to Lead A*

**Status:** Implemented for current 2D repair and guided 3D repair, with known
worker-window limitations.

**Files:**
- Modify: `src/gr/FlexGRGridGraph.h`
- Modify: `src/gr/FlexGRGridGraph_maze.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`
- Modify: `src/gr/FlexGR.h`

- [x] Add active net state to `FlexGRGridGraph`:

```cpp
    void setActiveNet(grNet* in) {
      activeNet = in;
    }
```

- [x] In `FlexGRWorker::routeNet()`, set the active `frNet` before A*.

- [x] Add worker helpers for mirrored-edge lookup and cost:

```cpp
    bool getSelfSymmetry2DMirrorEdge(frNet* net,
                                     frMIdx x,
                                     frMIdx y,
                                     frMIdx z,
                                     frDirEnum dir,
                                     frMIdx &mirrorX,
                                     frMIdx &mirrorY,
                                     frMIdx &mirrorZ,
                                     frDirEnum &mirrorDir) const;

    frCost getSelfSymmetry3DGuidedMirrorCost(frNet* net,
                                             frMIdx x,
                                             frMIdx y,
                                             frMIdx z,
                                             frDirEnum dir);
```

- [x] `FlexGRGridGraph::getNextPathCost()` adds 2D mirror congestion cost for
  self-symmetry nets and adds guided 3D mirror cost during the guided 3D phase.

- [ ] The current 2D mirror lookup is worker-local: if the mirrored edge falls
  outside the worker route box, it is not charged there. This is acceptable for
  the current smoke but not a complete global-window solution.

- [ ] The implementation does not yet subtract the old self-shadow contribution
  from the candidate mirror cost to avoid self-cost double counting in a fully
  general way.

- [x] Build and smoke run:

```bash
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
cd build/selfsym-status-read
../status-cmake/TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_status.def \
  > selfsym_status.log 2>&1
```

Observed: search repair completes; `Symmtry5` logs include mirror-cost queries.

### Task 7: Update Writeback to Modify Source Objects and Shadow Demand

**Status:** Partially implemented in worker-local demand updates; global
outside-shadow writeback and final consistency checks remain open.

**Files:**
- Modify: `src/gr/FlexGR_end.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`
- Modify: `src/gr/FlexGR.h`

- [x] Before ripping up a self-symmetry worker route, subtract worker-local
  source demand and mirror shadow demand derived from the old source pathSegs.

- [x] After routing a new self-symmetry worker route, add worker-local source
  demand and mirror shadow demand derived from the new source pathSegs.

- [x] During ordinary 2D repair, mirror route objects are not present in region
  query because mirror materialization runs only after ordinary 2D repair.

- [ ] `endWriteBackCMap()` still copies only the worker route-box region back to
  the global cmap. The code tracks `outside_shadow_delta`, but mirrored cells
  outside the worker route box are not explicitly written back globally.

- [x] Axis path segments are skipped by the mirror-shadow demand helpers.

- [ ] Add a debug check that old shadow demand has been removed and the current
  mirror shadow can be regenerated from the current lead source route.

- [x] Build and smoke run:

```bash
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
cd build/selfsym-status-read
../status-cmake/TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_status.def \
  > selfsym_status.log 2>&1
```

Observed: current smoke passes and reports `outside_shadow_delta: 0`; this does
not prove the cross-window outside-shadow case.

### Task 8: Layer Assignment Preserves Source/Shadow Split

**Status:** Implemented differently from the original split. Mirror topology is
materialized before layer assignment, and generic layer assignment uses an
added mirror-cost hook.

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [ ] No dedicated `layerAssign_selfSymmetry_net(net)` branch exists.

- [x] Mirror materialization runs before `layerAssign()`, so layer assignment
  sees the full materialized self-symmetry tree and can generate guide/DEF
  visible mirror-side route.

- [x] `getSelfSymmetryLayerAssignMirrorCost()` is called during layer selection
  and contributes mirrored congestion/blockage pressure.

- [x] After layer assignment, `stageSelfSymmetry3DLeadOnly()` temporarily removes
  mirror-side 3D objects and adds mirrored shadow demand for lead-only 3D repair.

- [ ] The implementation does not preserve a strict source/shadow split during
  layer assignment; mirror geometry is already materialized at that point.

- [x] Build and smoke run through 3D search repair:

```bash
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
cd build/selfsym-status-read
../status-cmake/TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_status.def \
  > selfsym_status.log 2>&1
```

Observed: log reaches layer assignment, 3D repair, guide generation, and detail
routing. `layerassign_mirror_cost_queries: 774` appears for `Symmtry5`.

### Task 9: Final Mirror Materialization for Output

**Status:** Implemented as pre-layer-assignment materialization, not as a final
output-only pass.

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [x] Decision made: downstream guide/TA/DR need explicit mirror topology, so
  mirror side is materialized after ordinary 2D repair and before
  `layerAssign()`.

- [x] Controlled pass exists as:

```cpp
    void searchRepairSelfSymmetryMirror();
    void buildSelfSymmetryMirror2DTopology();
    SelfSymmetryMirror2DStats buildSelfSymmetryMirror2DTopology_net(frNet *net);
```

- [x] The pass derives mirror guide edges from the final lead/axis 2D source
  route and connects mirror-side pins on a Hanan graph.

- [x] Axis-only edges are not mirrored as duplicate guide edges.

- [ ] This is not output-only materialization. The mirror topology feeds
  `layerAssign()`, TA, and DR. Later 3D handling uses temporary lead-only staging
  plus guided 3D repair rather than keeping mirror objects permanently invisible.

- [x] Current smoke evidence: `mirror_hanan_pins_covered: 3/3`,
  `mirror_repair_guide_hits: 3`, `mirror_repair_guide_misses: 0`, and
  `mirror_repair_pins_covered: 3/3`.

### Task 10: Debug Checks and Documentation

**Status:** Documentation updated; final self-symmetry consistency checker and
DR-aware symmetry verification are still open.

**Files:**
- Modify: `src/gr/FlexGR.cpp`
- Modify: `docs/self_symmetry_lead_mirror_search_repair.md`

- [ ] Add a debug check:

```cpp
    bool checkSelfSymmetrySourceAndShadow(frNet* net, bool verbose) const;
```

It should verify:
- Lead source route reaches the axis anchor.
- No stored GR object is on the mirror side during search repair.
- Axis demand is counted once.
- Mirror shadow demand can be regenerated from stored source route.

- [x] Update documentation with the current progress and corrected pattern
  route / DR scope.

- [x] Run current build:

```bash
cmake -S . -B build/status-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/status-cmake -j$(nproc) --target TritonRoute
```

- [x] Run current smoke:

```bash
mkdir -p build/selfsym-status-read
cd build/selfsym-status-read
cp ../../src/gr/flute/POST9.dat ../../src/gr/flute/POWV9.dat .
../status-cmake/TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_status.def \
  > selfsym_status.log 2>&1
```

Observed:
- `selfsym_status.log` reaches detailed routing without crash.
- `root-side reaches axis: 1`.
- `mirror_hanan_pins_covered: 3/3` and `mirror_repair_pins_covered: 3/3`.
- `layerassign_mirror_cost_queries: 774`.
- Detail routing reports `number of violations = 0`.

- [ ] Add DR-specific self-symmetry handling or a final detailed-route symmetry
  check. Current DR consumes guide/TA output but does not enforce mirror-pair
  constraints itself.

## Main Risks

- `frNode` tree ownership and `frNet::nodes` ordering matter. RPin nodes are assumed to appear first in several places, especially `layerAssign_net()`.
- Existing worker writeback is routeBox-local. Mirror shadow demand can land outside the worker routeBox, so shadow cmap updates must be explicit.
- Parallel worker execution means mirror-cost queries should treat global cmap as read-mostly during `main_mt()`. Source and shadow global updates should remain in worker `end()`, which is already single-threaded per batch.
- Axis geometry needs explicit handling to avoid double demand.
- Mirror materialization now happens after ordinary 2D GR search repair and before
  `layerAssign()`. Later ordinary 2D workers must not reopen mirror-side route;
  3D handling relies on lead-only staging plus guided mirror repair.

## Remaining Work From Current State

1. Execute M1-M5 from `docs/self_symmetry_detailed_routing.md`.
2. Add a defensive debug/assert path if a self-symmetry net ever reaches
   `patternRoute_LShape()` with a non-colinear Steiner-Steiner edge.
3. Prove or instrument that ordinary 2D worker init only opens lead/axis source
   geometry before mirror materialization.
4. Complete global writeback for mirror shadow demand that lands outside the
   worker route box.
5. Add `checkSelfSymmetrySourceAndShadow()` or equivalent final GR consistency
   checking for lead-to-axis connectivity, axis single counting, and regenerable
   mirror shadow.
6. Add DR-specific self-symmetry handling or a final detailed-route symmetry
   check; current DR only consumes guide/TA output.
7. Broaden smoke coverage beyond the current local case, especially cross-window
   mirror shadow and post-DR symmetry validation.
