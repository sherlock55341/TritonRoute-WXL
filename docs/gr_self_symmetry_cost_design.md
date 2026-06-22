# GR Self-Symmetry Cost Design

This note records the current GR search-repair schedule and maze cost policy for
self-symmetric nets. The cost implementation lives in
`FlexGRGridGraph::getNextPathCost()`.

## Goal

Self-symmetric nets are routed in two conceptual passes:

- `Auto`: find the first half of the route, including useful axis routing.
- `Mirror`: reuse the auto-pass route as the reference and route the mirrored
  half with strong preference for the planned mirrored geometry.

The scheduled 3D global-routing flow has one extra ordinary-net cleanup stage:

```text
2D auto      -> all nets
2D mirror    -> self-symmetric nets only
layerassign  -> all nets
3D auto      -> ordinary nets only
3D mirror    -> self-symmetric nets only
```

The `3d_auto` stage is intentionally `OrdinaryOnly`: it can repair regular nets
after layer assignment, but it must not rip up, boundary-split, or otherwise
rewrite self-symmetric nets. Self-symmetric 3D repair is handled by `3d_mirror`,
using the preserved layer-assigned self-symmetric route as its previous-route
reference.

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
planar edges are made more expensive, while axis planar edges and axis vias are
discounted:

```text
if edgeSide == mirrorSide:
  stepCost *= 64

if planar edgeSide == axis:
  stepCost /= 16

if via is on the axis gcell:
  stepCost /= 16
```

Design intent:

- Prefer the lead side and the symmetry axis during the first route.
- Do not forbid mirror-side edges, because congestion or blockages may still
  require them.
- Make axis routing significantly cheap in auto mode. If the auto pass used the
  axis, the mirror pass should later recognize it as previous route and preserve
  that low effective cost.

## Search-Repair Participation

Workers use `FlexGRSelfSymmetryMode` to decide which nets are targets:

- `Auto`: every net is a target.
- `OrdinaryOnly`: only nets without a self-symmetry constraint are targets.
- `Mirror`: only nets with a self-symmetry constraint are targets.

`initNets_roots()` and `route_getRerouteNets()` both use this target filter.
`initBoundary()` also uses it before splitting path segments at worker
boundaries, because boundary splitting mutates the top-level GR topology before
the later reroute filter runs. Without that guard, an ordinary-only pass could
still change self-symmetric nets even though it never routes them.

In `ripupMode == 1`, self-symmetric nets must enter mirror search-repair passes,
even when `mazeNetHasCong()` is false. Otherwise a clean but topologically bad
route would never see the self-symmetry cost bias.

When a self-symmetric `grNet` is added to the reroute queue, both the `grNet`
and its owning `frNet` must be marked modified. `FlexGRWorker::end()` only
writes modified nets back to the top-level GR shapes; without this flag the maze
can find a better axis route while the stage guide still shows the old route.

## Mirror Pass

Before ripping up a self-symmetric net in mirror mode, the worker records the
previous planar route edges in grid graph previous-edge bits. The mirror pass
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
the active net has a self-symmetry constraint. In an auto-mode 3D search-repair,
vias on the axis gcell also get the same 1/16 discount.

The current scheduled `3d_auto` stage does not use `Auto`; it uses
`OrdinaryOnly`, so self-symmetric nets are not active there and these
self-symmetry cost rules do not run for them. The next scheduled stage,
`3d_mirror`, is the only 3D search-repair stage that rewrites self-symmetric
nets.

Via moves are not classified as lead/axis/mirror edges by the planar edge-side
logic. There is a separate 3D-only self-symmetry penalty at
`VIA_ACCESS_LAYERNUM` that discourages M1/via access for self-symmetric nets:

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
- Keep the mirror pass tied to the recorded previous planar edge bits; that
  previous route is the contract between auto and mirror.
- Keep the scheduled 3D auto pass ordinary-only. It may update ordinary nets,
  but self-symmetric nets should remain byte-for-byte preserved from
  `layerassign` until `3d_mirror` records their previous planar edges.
