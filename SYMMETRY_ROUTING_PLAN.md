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

The detailed plan is in:

- [SYMMETRY_ONE_SIDE_ROUTING_PLAN.md](SYMMETRY_ONE_SIDE_ROUTING_PLAN.md)

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
- Symmetry nets currently run through the ordinary GR maze flow again, instead
  of being skipped by GR maze reroute.
- Worker overlap debug output is still present in `FlexGR_init.cpp` while the
  prototype is being debugged.

Planned pieces not implemented yet:

- TA needs an explicit symmetric-track pairing table so reference-side tracks can
  be copied to their matched mirror-side tracks without inserting new tracks.
- DR symmetry repair is planned as a cost-bias extension of the existing
  search-and-repair flow after reference-side copy.
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
