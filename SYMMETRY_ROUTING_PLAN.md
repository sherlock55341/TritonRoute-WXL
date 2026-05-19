# Symmetry Routing Notes

This note tracks the current direction for symmetry-aware routing in
TritonRoute-WXL.

The active design is no longer "mirror after every routing stage." The preferred
flow is:

```text
route one side, copy the mirror side, then repair with symmetry preference
```

In this flow, GR and TA keep only the reference side as the active routing
object. The mirror side is derived from the reference-side result instead of
being maintained as an independent routed tree. TA also records explicit
one-to-one symmetric track pairs because design tracks may not be perfectly
geometric mirrors and inserting extra tracks is expensive.

DR first routes the reference side and copies the result to the mirror side.
If the copied mirror geometry creates DRV, the router still uses the original
search-and-repair flow. The difference is that routes matching the opposite
side's symmetric geometry receive a reward, such as reduced effective wirelength
cost, so repairs remain as close to symmetric as practical while preserving
legality.

The prototype does not require exact symmetry at pin access and final pin
connection stubs. These regions may remain ordinary DR output when PA does not
provide matching mirrored access points or when forcing symmetry would risk
connectivity/DRC. The practical target is symmetric trunk/body routing with
legal fallback near pins.

Symmetry ownership is defined at the real logical pin level, not from DR worker
intermediate objects. Instance pins and top-level term pins should be paired by
their geometric symmetry relationship. Boundary pins, guide split pins, worker
edge pins, and other temporary DR artifacts are implementation details; they
must not decide which side of a symmetric net is routed or copied. AP symmetry
is also not a precondition for choosing the symmetric topology because AP
generation is not symmetry-aware.

The detailed plan is in:

- [SYMMETRY_ONE_SIDE_ROUTING_PLAN.md](SYMMETRY_ONE_SIDE_ROUTING_PLAN.md)

## Current Algorithm Decisions

The active DR direction is a phase-level symmetry flow, not an ad-hoc
`routeNet()` patch and not a copy/attach operation inside every local worker.
For this prototype, it is acceptable to assume the active symmetry experiment is
centered on symmetry nets. The intended large stages are:

```text
single-side initial solution
single-side optimized solution
copied solution
two-side optimized solution
```

The first stage routes only the selected single side. The second stage runs
search-and-repair on that same single-side solution. Local repair workers in
this stage may see only a reference-side pin or only a partial local route
window; that is expected and must not trigger mirror copy/attach. The copied
solution stage happens only after the single-side optimization phase has
completed. It copies the stable single-side body to the mirror side and attaches
mirror pins. The final two-side optimization stage then runs search-and-repair
on the complete two-side solution with symmetry cost guidance.

Therefore, copy/attach is a phase-boundary operation. It must not run inside
each local search-and-repair worker merely because that worker contains a
symmetry net. A local worker that lacks paired mirror pins is still a valid
single-side optimization worker; it should repair the current side according to
the phase settings rather than treating `copyPins == 0` as a copy failure.

The copy result is not meant to be protected as immutable geometry. Existing
search-and-repair may still rip up and reroute an entire net. Symmetry should be
preserved by the cost field, not by forbidding ripup of a side. The key state
that survives between repair rounds is therefore the preferred-region /
preferred-edge guide derived from a stable opposite-side solution, not the route
objects themselves.

Every search-and-repair round must explicitly specify which side is the
reference for symmetry guidance:

- When repairing or rerouting the left/reference side, derive the preferred
  region from the right/mirror-side solution.
- When repairing or rerouting the right/mirror side, derive the preferred
  region from the left/reference-side solution.

The preferred guide is side-directed. It is not generated from the complete
current route indiscriminately, because that would reward the side currently
being repaired and can preserve bad geometry. Instead, the selected reference
side is the only source of the guide. From that source side, DR generates a
two-sided preferred region: the original source-side footprint and its mirrored
footprint should both receive cost advantage.

This also means pin activation/filtering and cost preference must be separate
controls:

- Pin/AP filtering decides which pins are active in a one-side routing phase.
- Preferred-region source side decides which existing side provides the shape
  template for search-and-repair cost bias.

For example, a later repair round may route all local pins without one-side pin
filtering, while still using a symmetry preferred-edge cache generated from the
opposite side. Disabling pin filtering must not implicitly disable symmetry cost
preference.

The symmetry axis is a cost and geometry reference only. A point or segment
touching the axis must not be treated as connected merely because it is on the
axis. If a route must connect to the axis or cross the axis, that connection
must be represented by actual route objects and verified by ordinary
connectivity.

Worker-local boundary pins and boundary APs are routing infrastructure. They
may be required targets for stitching a worker-local single-side route, but they
are not symmetry-owned topology and should not decide real pin pairing or the
algorithmic objective. They also should not be blindly copied as if they were
the symmetric body.

The visual target is not mathematically exact symmetry. The desired result is a
visually symmetric body/trunk, with local deviations allowed for pin stubs,
legal DRC repair, and connectivity. Axis edges and preferred-region edges can
receive strong cost advantages, including a reduced effective wire cost such as
`wirecost * 0.5`, but legality and connectivity remain mandatory.

## Agent Workflow Notes

For implementation work in this repository, code-writing tasks should be
delegated to subagents when available, with GPT-5.3 Codex Spark preferred for
bounded edits. The main thread should focus on design framing, documentation,
diff review, validation, and acceptance. In goal-mode work that already
authorizes implementation, do not repeatedly stop for user confirmation before
edits; ask a separate subagent to challenge or confirm the plan, then integrate
that critique before proceeding.

## Demo Context

- Test input:
  - LEF: `/home/cyzhao/benchmark/primarius/outdata/ispd18_test1.input.lef`
  - DEF: `/home/cyzhao/benchmark/primarius/outdata/pattern_route_lay.def`
- First target net: `Symmtry5`
- Symmetry type: self-symmetric single net
- Symmetry axis: horizontal axis at `y = 71820` DBU
- Constraint source: `TritonRouteSymm` demo main through `frSymmetryConstraint`
- Current scope: one constrained net and one input symmetry axis
- Supported axis directions for the prototype: horizontal and vertical
- Diagonal axes are out of scope

## Implementation State

Current implemented pieces:

- `TritonRouteSymm` has a separate demo main and CMake target.
- `frDesign` stores one in-memory symmetry constraint.
- The shared symmetry interface uses `referenceSide` terminology.
- GR has an experimental symmetry topology prototype.
- TA now builds a worker-local per-layer symmetric-track pairing table after
  track initialization and uses it to bias symmetry-net track selection.
- DR now runs a real-pin preflight in the worker route flow. The preflight only
  treats `frInstTerm` and `frTerm` pins as symmetry-owned pins, classifies them
  against the symmetry axis, pairs mirror pins to reference pins by mirrored
  representative geometry, and keeps worker-generated boundary pins out of the
  ownership decision.
- With `TR_SYMM_WORKER_FLOW=1`, DR now uses a staged worker flow keyed by the
  coarse DR iteration:
  - iteration 0 builds the single-side initial solution;
  - iteration 1 runs single-side search-and-repair;
  - iteration 2 copies the optimized single-side solution and attaches mirror
    pins;
  - iteration 3 and later run two-side search-and-repair with symmetry
    preference.
- The copy/attach helper is therefore isolated to the copy phase. Single-side
  local workers may see only reference-side pins, boundary APs, or partial route
  windows, and they no longer treat missing mirror copy pins as fatal.
- The current two-side repair path uses normal all-pin routing with the
  preferred-edge source explicitly set to the reference-side solution. This keeps
  pin filtering and symmetry cost preference as separate controls.
- Symmetry nets currently run through the ordinary GR maze flow again, instead
  of being skipped by GR maze reroute.
- Worker overlap debug output is still present in `FlexGR_init.cpp` while the
  prototype is being debugged.

Planned pieces not implemented yet:

- TA still does not materialize mirror-side output from the pairing table; it
  only biases candidate selection for the symmetry net.
- DR still needs a stronger side-directed search-and-repair cost policy. The
  current preferred-edge cache exists, but the next refinement should choose the
  source side deliberately per repair round and strengthen the reward for
  preferred/axis edges so ordinary repair does not drift away from the symmetric
  body.
- Exact symmetry around pin access and final pin-connection stubs is not a
  requirement for the current prototype. The body/trunk topology should be
  reference-copy derived, while pin stubs can be locally repaired as long as
  they do not become an AP-level topology fallback.
- DR symmetry repair is planned as a cost-bias extension of the existing
  search-and-repair flow after one-side copy/replacement. The repair should
  prefer symmetry but must preserve connectivity and DRC first. The intended
  behavior is not mathematically exact symmetry; it is visually and
  topologically symmetric routing where local stubs and DRC fixes may deviate
  when needed.
- The current GR prototype may still materialize mirror-side topology nodes as
  an intermediate projection. The long-term target is for mirror-side routing
  state to be derived output, not an independently maintained routed tree.

## Validation Notes

After C++ changes, run clangd on the changed translation units where practical:

```bash
clangd --compile-commands-dir=build --check=<changed-file.cpp>
```

For this prototype, build:

```bash
cmake --build build -j16 --target TritonRouteSymm
```

For routing validation, run the demo from `src/gr/flute` so the FLUTE LUT files
are visible:

```bash
../../../build/TritonRouteSymm
```

Latest validated demo result:

- Command: `cd src/gr/flute && ../../../build/TritonRouteSymm | tee
  /tmp/tr_symm_mirror_repair1.log`
- Build checks:
  - `clangd --compile-commands-dir=build --check=src/dr/FlexDR_maze.cpp`
    completed with only known clangd check-mode tweak noise.
  - `cmake --build build -j32 --target TritonRouteSymm`
  - `cmake --build build -j32 --target TritonRoute`
- DR preflight and copy:
  - Main `Symmtry5` invocations still pair all 5 mirror real pins to 5
    reference real pins when the worker has the full net context.
  - Main copy invocations copy roughly `6-7` path segments and `8-9` vias,
    depending on the repair iteration window.
  - Mirror attach reports `targets=5`, `routed=5`, `failed=0`, and
    `bridgeRepair=done` on the full-net invocations.
- Final DR status:
  - `number of violations = 0`
  - `connectivity = connected`
  - `pins = 10/10`
  - `symmetric metal length = 22220/48880 DBU (45.46%)`
- Current gap: the route is legal and connected, but the final symmetry ratio
  and visual symmetry are not yet good enough. The remaining work is to bias
  search-and-repair toward a symmetry-derived preferred region instead of using
  ordinary repair that can move the final solution away from the copied body.
- The previous hybrid AP/boundary-driven result reached `connected, pins =
  12/12` and `60.87%` symmetric metal length, but that result is no longer the
  accepted algorithmic target because AP/boundary availability was allowed to
  influence topology and produced visibly asymmetric left-side body routing.
