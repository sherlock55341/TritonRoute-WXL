# GR Self-Symmetry Cost Design

This note records the current GR maze cost policy for self-symmetric nets. The
implementation lives in `FlexGRGridGraph::getNextPathCost()`.

## Goal

Self-symmetric nets are routed in two conceptual passes:

- `Auto`: find the first half of the route, including useful axis routing.
- `Mirror`: reuse the auto-pass route as the reference and route the mirrored
  half with strong preference for the planned mirrored geometry.

The cost rules are intentionally local to candidate planar edges. They should
make the desired topology cheaper without replacing the maze router with a hard
constraint solver.

## Edge Sides

Each planar candidate edge is classified relative to the net symmetry axis:

- `-1`: lead side.
- `0`: on-axis or crossing the axis.
- `1`: mirror side.

The current lead side constant is `-1`, so the mirror side is `1`. Axis edges
are not mirror-side edges.

The classification only applies to cardinal planar directions: `E`, `N`, `S`,
and `W`. Via directions `U` and `D` are not directly mirrored by this edge-side
logic.

## Base Step Cost

Every candidate edge starts from the ordinary GR cost components:

```text
edgeLength
+ congestion cost
+ history cost
+ block cost
+ overflow cost
```

Self-symmetry rules then adjust this `stepCost`, and the adjusted step is added
to the path cost.

## Auto Pass

In `Auto` mode, the router is allowed to explore normally, but mirror-side
planar edges are made more expensive:

```text
if edgeSide == mirrorSide:
  stepCost *= 4
```

Design intent:

- Prefer the lead side and the symmetry axis during the first route.
- Do not forbid mirror-side edges, because congestion or blockages may still
  require them.
- Keep axis routing cheap in auto mode. If the auto pass used the axis, the
  mirror pass should later recognize it as previous route and preserve that low
  effective cost.

## Mirror Pass

Before ripping up a self-symmetric net in mirror mode, the worker records the
previous planar route edges in `selfSymmetryPrevPlanarEdges`. The mirror pass
then uses that previous route as the reference.

Mirror-mode step adjustments:

```text
if edgeSide is lead side or axis:
  if candidate edge was not in previous route:
    stepCost += BLOCKCOST * edgeLength * 100

if edgeSide is mirror side:
  mirrorEdge = mirrored candidate edge
  if mirrorEdge is invalid or mirrorEdge was not in previous route:
    stepCost += 128 * edgeLength
```

Design intent:

- Lead-side and axis geometry should mostly follow the auto-pass result.
- Axis edges are valid previous-route geometry, not invalid mirror geometry.
- Mirror-side edges should be cheap when they correspond to a previous lead-side
  edge, and mildly discouraged when they do not.
- The mirror-side missing-previous penalty is deliberately much smaller than the
  lead/axis missing-previous block penalty. It nudges matching without making
  detours impossible.

## Mirrored Congestion Cost

After the step-cost adjustment, the router may add the cost of the corresponding
mirror edge. This is only enabled for:

- mirror mode on mirror-side edges;
- auto mode on lead-side edges.

For a valid mirror edge, the router adds mirrored congestion, history, block,
and overflow terms. In mirror mode, if a mirror-side candidate has no valid
mirror edge, it receives an invalid-mirror penalty:

```text
MARKERCOST * 8 * max(edgeLength, 1)
```

This penalty must only apply to mirror-side routing. It should not apply to
lead-side or axis edges. The axis case matters because an edge that lies on the
axis can have no distinct mirrored edge; treating that as invalid mirror
geometry makes an already-routed axis segment incorrectly expensive.

## 2D and 3D Scope

The main auto/mirror planar edge rules are not gated by `is2D()`. They apply in
both 2D and 3D GR whenever the candidate direction is `E`, `N`, `S`, or `W` and
the active net has a self-symmetry constraint.

Via moves are not classified as lead/axis/mirror edges by this logic. There is a
separate 3D-only self-symmetry penalty at `VIA_ACCESS_LAYERNUM` that discourages
M1/via access for self-symmetric nets:

```text
if !is2D() and layer == VIA_ACCESS_LAYERNUM:
  stepCost += getViaStepCost(upEdgeLength) * SELF_SYMMETRY_M1_VIA_PENALTY_COUNT
```

This is independent of the mirror-side edge penalty.

## Important Invariants

- Do not add net-name-specific cost behavior. Net names are useful for temporary
  diagnostics only.
- Do not apply invalid mirror penalties to axis edges.
- Do not apply mirror-side penalties before the candidate edge has been
  classified relative to the active net axis.
- Keep the auto pass permissive: it should bias the route, not hard-fail when
  the preferred side is blocked.
- Keep the mirror pass tied to `selfSymmetryPrevPlanarEdges`; that previous
  route is the contract between auto and mirror.
