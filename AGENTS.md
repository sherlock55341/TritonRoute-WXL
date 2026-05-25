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

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `add root side tree` and `split to two sides`. Keep commit titles concise, lowercase when natural, and scoped to one logical change. Pull requests should describe the routing stage affected, list build and smoke-test commands run, note any benchmark/design used, and call out changes to external parser modules or FLUTE data.

## Agent-Specific Instructions
Do not overwrite local edits in unrelated files. Keep documentation changes focused, and avoid reformatting large C++ files unless formatting is the explicit task.
