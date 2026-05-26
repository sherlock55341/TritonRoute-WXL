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

- [ ] Add nullable constraint access on `frNet` as the first code change:

```cpp
    const frSelfSymmetryConstraint* getSelfSymmetryConstraintPtr() const {
      return constraint == frNetRoutingConstraint::frcSelfSymmetry ? &selfSymmetryConstraint : nullptr;
    }
```

- [ ] Do not modify `src/frBaseTypes.h`, node/object side fields, `CMakeLists.txt`, or routing behavior in this task.

- [ ] Add helper declarations:

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

- [ ] Implement side helpers as coordinate-derived utilities. Point/GCell side returns `-1`, `0`, or `1`; `getSelfSymmetryRootSide()` maps root-on-axis side `0` to the existing fallback `-1`.

- [ ] Build check:

```bash
cmake --build build -j$(nproc)
```

Expected: build succeeds without behavior change.

### Task 2: Add Mirror Shadow Demand Helpers

**Files:**
- Modify: `src/gr/FlexGR.h`
- Modify: `src/gr/FlexGR.cpp` or `src/gr/FlexGR_selfsym.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`

- [ ] Add helpers to add/sub demand for a source path segment:

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

- [ ] `modSelfSymmetrySourceDemand()` updates normal lead/axis demand.

- [ ] `modSelfSymmetryMirrorShadowDemand()` mirrors only non-axis source geometry and updates cmap/cmap2D at the mirrored edge. Axis geometry is skipped to avoid double count.

- [ ] Add one wrapper:

```cpp
    void modSelfSymmetrySourceAndShadowDemand(frNet* net,
                                              const frPoint &begin,
                                              const frPoint &end,
                                              frLayerNum layerNum,
                                              bool isAdd,
                                              bool is2D);
```

It calls source demand plus mirror shadow demand.

- [ ] Build check:

```bash
cmake --build build -j$(nproc)
```

Expected: build succeeds.

### Task 3: Make Self-Symmetry Topology a Lead-to-Axis Source Tree

**Files:**
- Modify: `src/gr/FlexGR.cpp`
- Modify: `src/gr/FlexGR_topo.cpp` only if helper extraction is needed

- [ ] In `initGR_genTopology_selfsymmetry_net()`, keep AP assignment, root selection, GCell node creation, axis GCell computation, and root-side selection.

- [ ] Treat the axis GCell as a mandatory lead-side terminal. Keep the existing `genSelfSymmetryRootSideTopology()` behavior that connects the root-side tree to the axis if needed.

- [ ] Stop using `genSelfSymmetryOppositeSideTopology()` as a repair source route. Opposite-side topology may remain only for debug comparison.

- [ ] Convert `rootSideTreeVertices/rootSideTreeEdges` into a real `frNode` source tree:
- Use existing GCell nodes for root/lead terminal GCells.
- Create Steiner nodes for non-terminal root-side vertices.
- Create or reuse an axis anchor node at the axis GCell.
- Parent the tree from the root GCell node.

- [ ] Do not create mirror-side route nodes or mirror-side route objects in this phase.

- [ ] Mirror-side original pin nodes may remain as logical net pins, but they must not become search repair terminals. Their physical mirror connection is implicit during search repair and can be materialized after repair if output requires it.

- [ ] Add assertions:
- The root-side source tree reaches the axis anchor.
- Every lead-side non-root pin has a parent.
- No stored GR route object is on the mirror side.
- Axis source objects are counted once.

- [ ] Build and smoke run:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-shadow-smoke
cd build/selfsym-shadow-smoke
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_topology.def > selfsym_topology.log 2>&1
```

Expected: no crash in topology generation; log reaches at least the first 2D congestion report.

### Task 4: Pattern Route Only Lead-to-Axis Source Geometry

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [ ] In `initGR_patternRoute_init()`, for self-symmetry nets, collect only source-tree edges from lead/axis nodes. Do not collect mirror logical pins or mirror-derived geometry.

- [ ] In `patternRoute_LShape()`, when routing a self-symmetry source edge, compare L-shape choices using:

```text
total_cost = lead_source_cost + mirror_shadow_cost
```

- [ ] After choosing the lead L-shape, update demand through `modSelfSymmetrySourceAndShadowDemand()`.

- [ ] Axis L-shape segments are allowed only if they preserve lead-to-axis connectivity and do not double count mirror shadow demand.

- [ ] Build and smoke run:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-shadow-smoke
cd build/selfsym-shadow-smoke
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_pattern.def > selfsym_pattern.log 2>&1
```

Expected: no mirror-side pattern route object is created; 2D congestion report completes.

### Task 5: Keep Worker Init Lead/Axis-Only

**Files:**
- Modify: `src/gr/FlexGR_init.cpp`
- Modify: `src/db/grObj/grNet.h`

- [ ] Add a `grNet` flag:

```cpp
    bool isSelfSymmetryLeadEditable() const;
    void setSelfSymmetryLeadEditable(bool in);
```

- [ ] In `initNets_roots()`, self-symmetry nets should be discovered only from real lead/axis objects in region query. Since mirror shadow has no objects, no mirror-only subnet should be possible.

- [ ] In `initNet_initNodes()`, self-symmetry worker subnets must only include lead/axis source nodes and boundary pins needed to preserve the existing lead-to-axis source route.

- [ ] If a worker window cuts the lead-to-axis connection, the boundary pin represents the continuation to the axis. If the window contains the axis anchor, the axis anchor is a fixed terminal.

- [ ] Build check:

```bash
cmake --build build -j$(nproc)
```

Expected: build succeeds.

### Task 6: Add Mirror Shadow Cost to Lead A*

**Files:**
- Modify: `src/gr/FlexGRGridGraph.h`
- Modify: `src/gr/FlexGRGridGraph_maze.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`
- Modify: `src/gr/FlexGR.h`

- [ ] Add active net state to `FlexGRGridGraph`:

```cpp
    void setActiveNet(grNet* in) {
      activeNet = in;
    }
```

- [ ] In `FlexGRWorker::routeNet()`, set active net before A* and clear it before return.

- [ ] Add a worker helper:

```cpp
    frCost getSelfSymmetryMirrorShadowEdgeCost(grNet* net,
                                               frMIdx gridX,
                                               frMIdx gridY,
                                               frMIdx gridZ,
                                               frDirEnum dir) const;
```

This helper should:
- Return `0` for non-self-symmetry nets.
- Return `0` for axis edges.
- Return `0` for via direction in the first implementation unless 3D mirror via cost is explicitly added.
- Convert the local candidate edge to absolute coordinates.
- Mirror the absolute edge.
- Query the global cmap/cmap2D for mirrored raw demand, raw supply, block, and history.
- Avoid charging the net against its own old shadow demand by subtracting the old self-shadow contribution derived from the old source route when possible.

- [ ] In `FlexGRGridGraph::getNextPathCost()`, add the mirror shadow cost when `activeNet->isSelfSymmetryLeadEditable()`.

- [ ] Build and smoke run:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-shadow-smoke
cd build/selfsym-shadow-smoke
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_astar.def > selfsym_astar.log 2>&1
```

Expected: search repair completes; lead A* cost includes mirrored congestion pressure.

### Task 7: Update Writeback to Modify Source Objects and Shadow Demand

**Files:**
- Modify: `src/gr/FlexGR_end.cpp`
- Modify: `src/gr/FlexGR_maze.cpp`
- Modify: `src/gr/FlexGR.h`

- [ ] Before ripping up a self-symmetry worker route, subtract demand for old source route objects and subtract mirror shadow demand derived from those old source objects.

- [ ] After routing a new self-symmetry source route, add demand for new source route objects and add mirror shadow demand derived from those new source objects.

- [ ] In `endRemoveNets_objs()`, remove only real source objects from region query and net ownership. There are no mirror objects to remove.

- [ ] In `endWriteBackCMap()`, explicitly update mirrored cmap/cmap2D cells touched by shadow demand, even when they are outside the worker `routeBox`.

- [ ] Ensure axis segments are not mirrored into shadow demand.

- [ ] Build and smoke run:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-shadow-smoke
cd build/selfsym-shadow-smoke
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_writeback.def > selfsym_writeback.log 2>&1
```

Expected: no stale mirror shadow demand remains after reroute; no mirror route objects are present in region query.

### Task 8: Layer Assignment Preserves Source/Shadow Split

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [ ] In `layerAssign_net()`, branch self-symmetry nets to `layerAssign_selfSymmetry_net(net)`.

- [ ] Assign layers only to lead/axis source nodes.

- [ ] Include mirror shadow cost while choosing source edge layers.

- [ ] Update 3D cmap with source demand plus mirrored shadow demand.

- [ ] Do not create mirror `grPathSeg`/`grVia` during layer assignment.

- [ ] Build and smoke run through 3D search repair:

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-shadow-smoke
cd build/selfsym-shadow-smoke
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_layer.def > selfsym_layer.log 2>&1
```

Expected: log reaches `reportCong3D()` and guide writing; no mirror objects are created during repair.

### Task 9: Final Mirror Materialization for Output

**Files:**
- Modify: `src/gr/FlexGR.cpp`

- [ ] Decide where output needs explicit mirror geometry:
- If guide/DEF/DR can consume implicit self-symmetry source route plus shadow demand, no materialization is needed.
- If output requires physical mirror boxes, materialize after all search repair and layer assignment are complete.

- [ ] Add a controlled final pass:

```cpp
    void materializeSelfSymmetryMirrorForOutput(frNet* net);
```

- [ ] The final pass mirrors non-axis source objects into output-only mirror objects. It must not feed those objects back into search repair.

- [ ] Axis objects are not duplicated.

- [ ] If materialized mirror objects are inserted into region query for downstream stages, do it after the final GR repair pass so no GR worker can independently reroute them.

### Task 10: Debug Checks and Documentation

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

- [ ] Run final build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- [ ] Run final smoke:

```bash
mkdir -p build/selfsym-final
cd build/selfsym-final
../TritonRoute -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef -def ~/benchmark/primarius/outdata/pattern_route_lay.def -output selfsym_final.def > selfsym_final.log 2>&1
```

Expected:
- `selfsym_final.log` reaches guide writing without crash.
- 2D and 3D congestion summaries are present.
- No self-symmetry debug-check failure appears.
- Output files are under `build/selfsym-final/`.

## Main Risks

- `frNode` tree ownership and `frNet::nodes` ordering matter. RPin nodes are assumed to appear first in several places, especially `layerAssign_net()`.
- Existing worker writeback is routeBox-local. Mirror shadow demand can land outside the worker routeBox, so shadow cmap updates must be explicit.
- Parallel worker execution means mirror-cost queries should treat global cmap as read-mostly during `main_mt()`. Source and shadow global updates should remain in worker `end()`, which is already single-threaded per batch.
- Axis geometry needs explicit handling to avoid double demand.
- If output materializes mirror objects too early, later GR workers may see them as reroutable objects. Materialization must happen after GR search repair.

## Recommended Execution Order

1. Root-side geometry helpers and nullable constraint access.
2. Shadow demand add/sub helpers.
3. Lead-to-axis self-symmetry topology source tree.
4. Lead-only pattern routing with mirror shadow cost.
5. Worker init restricted to source objects.
6. Mirror-shadow cost in A*.
7. Source object writeback plus shadow demand updates.
8. Self-symmetry layer assignment.
9. Final output materialization, if required.
10. Debug checks and smoke validation.
