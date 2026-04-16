# Repository Guidelines

## Project Structure & Module Organization
Core router code lives in `src/`, organized by subsystem: `gr/` (global routing), `ta/` (track assignment), `dr/` (detailed routing), `gc/` and `drc/` (geometry and rule checking), `pa/` and `rp/` (pin access and repair), `io/` (LEF/DEF and guide parsing), and `db/` (shared data models). Third-party LEF/DEF parser sources are vendored under `module/def/5.8-p029` and `module/lef/5.8-p029`. Build artifacts should stay in `build/` and not be committed.

## Build, Test, and Development Commands
Configure an out-of-tree build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`. Compile with `cmake --build build -j`, which produces the `TritonRoute` executable. Run the router from the build directory or repo root, for example: `./build/TritonRoute -lef tech.lef -def design.def -guide route.guide -output routed.def`. Use `cmake --install build --prefix <dir>` only when you need staged `bin/` and `include/` outputs.

## Coding Style & Naming Conventions
This repository uses C++17 and `.clang-format` based on Google style with 4-space indentation and unsorted includes. Match surrounding conventions: class names such as `FlexDRWorker` use PascalCase, methods use lowerCamelCase, and filenames typically mirror the main class or subsystem (`FlexGR.cpp`, `io_parser_helper.cpp`). Prefer keeping new code within the existing subsystem directory instead of adding cross-cutting utilities at the top level.

## Testing Guidelines
There is no top-level `ctest` or dedicated unit-test target in the current CMake setup. Validate changes by rebuilding and running representative routing flows on known LEF/DEF inputs. For parser-related work, the vendored LEF/DEF modules include their own `TEST/` assets; use them for focused regression checks when touching `module/` or `src/io/`. Document the exact command and dataset used in your PR notes.

## Commit & Pull Request Guidelines
The visible git history is minimal, so use short, imperative commit subjects such as `Fix via enclosure check in FlexDR`. Keep commits focused on one subsystem or behavior change. Pull requests should describe the routing scenario affected, summarize user-visible impact, list validation commands, and attach before/after logs or output diffs when behavior changes.

## Configuration Notes
CMake requires GCC 7+ compatible C++17 support plus Boost, OpenMP, Bison, and zlib. For global-detailed routing runs, ensure `POST9.dat` and `POWV9.dat` from `src/gr/flute/` are available in the working directory.
