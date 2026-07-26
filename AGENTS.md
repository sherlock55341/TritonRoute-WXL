# Repository Guidelines

## Project Structure & Module Organization
TritonRoute-WXL is a C++17 CMake project. Core source lives in `src/`, with routing stages split by subsystem: `pa/` pin access, `rp/` routing prep, `gr/` global routing, `ta/` track assignment, `dr/` detailed routing, `gc/` geometry/DRC checking, `io/` LEF/DEF/guide I/O, and `db/` design and technology objects. The executable entry point is `src/main.cpp`; shared router logic is built as `flexroutelib`. Vendored LEF/DEF parsers live under `module/lef/5.8-p029` and `module/def/5.8-p029`. Documentation belongs in `docs/`; FLUTE lookup data is in `src/gr/flute/`.

## Build, Test, and Development Commands
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: configure the default optimized build.
- `cmake --build build -j$(nproc)`: build `build/TritonRoute` and `libflexroute`.
- `./build/TritonRoute -lef <LEF_FILE> -def <DEF_FILE> -guide <GUIDE_FILE> -output <OUTPUT_DEF>`: run detailed routing with a route guide.
- `./build/TritonRoute -lef <LEF_FILE> -def <DEF_FILE> -output <OUTPUT_DEF>`: run global-detailed routing. Keep `POST9.dat` and `POWV9.dat` from `src/gr/flute/` in the run directory.

## Coding Style & Naming Conventions
Follow the existing style: two-space indentation, braces on the same line for functions and control blocks, C++17 standard library types, and namespace `fr` for router code. Class names use PascalCase (`FlexRoute`, `FlexGRGridGraph`); functions and variables generally use lowerCamelCase or established project names; global constants and parameters are uppercase. Preserve existing BSD license headers when editing files that already include them. When adding source files, update `CMakeLists.txt` explicitly.

## Testing Guidelines
There is no committed unit-test or CTest suite. Validate changes by rebuilding and running at least one representative LEF/DEF flow. For routing changes, compare output DEFs, guide files, DRC reports, logs, and congestion summaries against a known baseline. Keep large benchmark inputs out of git unless the project explicitly adds a fixture directory.
Available local smoke-test inputs: LEF `~/benchmark/primarius/outdata/ispd18_test1.input.lef`, DEF `~/benchmark/primarius/outdata/pattern_route_lay.def`.
Do not write test outputs to `/tmp`; place generated DEFs, guides, logs, summaries, and param files under `build/` or a subdirectory of `build/`. Test results should be easy for a reviewer to verify by running the compiled `build/TritonRoute` binary and reading the generated output/log files.
Keep validation lightweight by default: for local helper, declaration, comment, or other no-call-site changes, a build is enough; for routing behavior changes, run one representative smoke test. Add extra checks only when the change is risky, user-requested, or the first check fails.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `add root side tree` and `split to two sides`. Keep commit titles concise, lowercase when natural, and scoped to one logical change. Pull requests should describe the routing stage affected, list build and smoke-test commands run, note any benchmark/design used, and call out changes to external parser modules or FLUTE data.

## Agent-Specific Instructions
Do not overwrite local edits in unrelated files. Keep documentation changes focused, and avoid reformatting large C++ files unless formatting is the explicit task.

## Detailed Routing Call-Path Notes
Use this map before instrumenting DR behavior. Do not guess whether a run uses
the ordinary worker path or the queue path.

- Entry is `src/main.cpp` -> `FlexRoute::main()`. Normal command-line mode only
  accepts `-lef`, `-def`, `-guide`, `-threads`, `-output`, `-verbose`, and
  `-drouteEndIterNum`. Parameter-file mode additionally accepts keys such as
  `outputDRC`, `outputguide`, and `outputMaze`.
- `FlexRoute::main()` always runs `init()`, then if `GUIDE_FILE` is empty it
  runs global routing (`gr()`), reads the generated guide, and enables via
  generation. Then it runs the routed stages in this order:
  `prep()` -> `ta(SelfSymmetryOnly)` -> `dr(SelfSymmetryOnly)` ->
  `ta(MirrorOnly)` -> `dr(MirrorOnly)` ->
  `ta(OrdinaryOnly)` -> `dr(OrdinaryOnly)` -> `endFR()`.
- `RouteNetMode` controls net membership. `SelfSymmetryOnly` matches nets with
  `getSelfSymmetryConstraintPtr() != nullptr`; `MirrorOnly` matches nets with
  `getMirrorConstraintPtr() != nullptr` (mirror pairs linked from the
  hardcoded `Mirror*_1`/`Mirror*_2` table in `src/FlexRoute.cpp`);
  `OrdinaryOnly` matches nets without either constraint; `All` is only the
  default constructor mode and is not used by the current top-level
  `FlexRoute::main()` sequence. During
  `OrdinaryOnly` DR iter 3+ with `ripupMode=0`, queue-mode marker repair may
  also reroute self-symmetry and mirror nets implicated by current DRC
  markers.
- `FlexDR::main()` is a fixed list of `searchRepair(...)` calls. The
  `-drouteEndIterNum N` / `drouteEndIterNum:N` value sets `END_ITERATION`; each
  `searchRepair(iter, ...)` returns immediately when `iter > END_ITERATION`.
  Current calls use `fixMode=9`, so workers launched through `main_mt()` route
  through `FlexDRWorker::route_queue()`, not `FlexDRWorker::route()`.
- `searchRepair(...)` also returns early for `iter > 0` when top-block marker
  count is zero, except for forced self-symmetry/mirror reroute iters. The
  forced reroute is only active in `RouteNetMode::SelfSymmetryOnly` /
  `RouteNetMode::MirrorOnly` for all constrained nets in DR iter 2 and later.
  Prev-edge cost uses that forced reroute window, and also applies to
  constrained nets pulled into `OrdinaryOnly` iter 3+ marker repair. Mirror
  pairs additionally swap leader/follower roles every iteration
  (`iter % 2` parity), and `FlexDR::updateMirrorPathSegCaches()` publishes
  the follower cache with the NEXT iteration's parity at the end of each
  `MirrorOnly` iteration.
- Worker path: `FlexDR::searchRepair()` builds tiled `FlexDRWorker`s, sets
  `routeBox`, `extBox`, `drcBox`, `mazeEndIter`, `drIter`, `ripupMode`,
  `followGuide`, `fixMode`, and costs, then calls `worker->main_mt()` in OpenMP
  batches. With `fixMode=9`, `main_mt()` calls `init()` -> `route_queue()` ->
  `cleanup()`. `FlexDRWorker::main()` is the non-MT path and currently calls
  `route()`, but it is not the path used by the normal batched search-repair
  flow.
- Queue path: `route_queue()` creates a `FlexGCWorker`, initializes marker
  costs, fills a reroute queue with `route_queue_init_queue()`, then loops in
  `route_queue_main()`. For route items, `route_queue_main()` rips up the
  `drNet`, calls `mazeNetInit()` -> `routeNet()` -> `mazeNetEnd()`, updates the
  GC target with `gcWorker->updateDRNet(net)`, runs `gcWorker->main()`, applies
  GC patch wires, then uses `gcWorker->getMarkers()` to update the queue and
  marker costs. In `OrdinaryOnly` iter 3+ partial-ripup queue mode, marker
  owners can make self-symmetry nets route items; this does not force reroute
  all self-symmetry nets. Insert marker diagnostics here when debugging
  queue-mode DR markers.
- Ordinary `route()` path: only used when `fixMode != 9` through `main_mt()` or
  through the single-worker `main()` path. It runs `mazeIterInit()`, routes
  `rerouteNets`, then calls `route_drc()` and uses worker member `markers`.
  Instrumenting only `route_drc()` or `getMarkers()` will miss the current
  queue-mode marker flow.
- Final DRC reporting: `outputDRC`/`DRC_RPT_FILE` is written by
  `FlexDR::reportDRC()` near the end of `FlexDR::main()`, and reports final
  top-block markers. It does not show transient per-worker markers inside
  `route_queue_main()`.
