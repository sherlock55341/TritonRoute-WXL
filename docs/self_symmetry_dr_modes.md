# DR Self-Symmetry Modes

This note records the intended DR self-symmetry behavior before resetting the
DR code to a clean baseline.

## DR Modes

- `Auto`: ordinary routing pass. Route all nets. Ordinary nets use ordinary
  cost; self-symmetry nets use lead self-symmetry cost.
- `Mirror`: self-symmetry mirror pass. Route only nets with a self-symmetry
  constraint. Preserve lead-side and axis route objects, delete and reroute
  only mirror-side objects, and use mirror-aware self-symmetry cost.

## Target Semantics

Use the short worker helper name `isTarget(frNet* net)`.

- `Auto`: `isTarget(net)` is true for all nets.
- `Mirror`: `isTarget(net)` is true only for self-symmetry nets.

`isTarget(net) == false` only means the net is not routed in the current DR
phase. It does not mean the net's existing shapes can be ignored. Non-target
net shapes must still affect target nets through blockage, cost, and DRC.

## Cost Setup

- Keep the cost hook in the low-level path-cost calculation, such as
  `FlexGridGraph::getNextPathCost()`.
- `Auto`: ordinary nets have no active self-symmetry context; self-symmetry nets
  use `routeMode=Lead`. Lead cost biases ordinary routing toward the symmetry
  design but does not change routing control flow.
- `Mirror`: active self-symmetry net with `routeMode=Mirror`; mirrored lead
  edges get reward/discount, while non-mirrored paths keep normal cost.

Cost must only affect A* path preference. It must not change pin selection,
boundary pin logic, source/destination updates, or writeback semantics.

## Later Implementation

- Do not add an explicit `targetNets` set; derive the routing target from
  `isTarget(frNet* net)`.
- Do not add `routeNetCore()`.
- Keep the old `routeNet()` ordinary body as the single ordinary routing
  implementation.
- Base self-symmetry target detection on `getSelfSymmetryConstraintPtr()`.
- In `Auto`, activate the lead cost context only while routing a self-symmetry
  net, then deactivate it before returning.
- In `Mirror`, use the only special control flow: freeze lead/axis components,
  split cross-axis path segments at the effective axis, and reroute the mirror
  side from preserved lead/axis sources.
- Before implementing target filtering, verify that non-target shapes remain
  visible to worker cost initialization. If filtering would remove their
  blockage/cost impact, add the smallest fix there instead of treating
  non-target nets as absent.

## GR Modes

- `Auto`: route all nets in the normal GR search-repair flow. Ordinary nets use
  ordinary cost. Self-symmetry nets use lead self-symmetry cost only at the
  low-level path-cost hook.
- `Mirror`: route only self-symmetry nets. Preserve lead-side and axis route
  objects, split cross-axis segments at the axis, and reroute only the mirror
  side with mirror-aware cost.

## GR Phase Flow

- 2D `Auto` search repair routes all nets. Self-symmetry nets differ only by
  lead cost; pin/root/boundary/A* loop/writeback stay ordinary.
- 2D `Mirror` search repair targets only self-symmetry nets. The first mirror
  maze iteration must reroute every self-symmetry net even if it has no
  congestion/DRV; later iterations may use ordinary ripup/congestion selection.
- `layerAssign()` ignores self-symmetry. It runs directly with ordinary layer
  assignment cost and no mirror reward or symmetry-specific correction.
- 3D `Auto` search repair routes all nets. Self-symmetry nets again use lead
  cost only.
- 3D `Mirror` search repair targets only self-symmetry nets. Its first mirror
  maze iteration also reroutes every self-symmetry net even if it has no
  congestion/DRV; later iterations may use ordinary ripup/congestion selection.

## GR Cost Setup

- `Auto`: ordinary nets have cost equivalent to first-commit GR. Self-symmetry
  nets use `routeMode=Lead`.
- `Mirror`: self-symmetry nets use `routeMode=Mirror`.
- Self-symmetry nets return `0` from `getEstCost()`; real path cost is still
  accumulated through `getNextPathCost()`.
- Cost hooks belong in low-level GR path cost (`FlexGRGridGraph::getNextPathCost()`
  and `getEstCost()`), not in pin selection, root selection, boundary handling,
  A* control flow, or writeback.
