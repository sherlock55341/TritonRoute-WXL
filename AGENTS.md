# Repository Guidelines

## Project Structure & Module Organization
Core router code lives in `src/`, organized by subsystem: `gr/` (global routing), `ta/` (track assignment), `dr/` (detailed routing), `gc/` and `drc/` (geometry and rule checking), `pa/` and `rp/` (pin access and repair), `io/` (LEF/DEF and guide parsing), and `db/` (shared data models). Third-party LEF/DEF parser sources are vendored under `module/def/5.8-p029` and `module/lef/5.8-p029`. Build artifacts should stay in `build/` and not be committed.

## Build, Test, and Development Commands
Configure an out-of-tree build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`. Compile with `cmake --build build -j16`, which produces the `TritonRoute` executable. Use Release mode for local validation unless a task explicitly requires a different configuration. Agents may use up to 32 threads for development tools and builds when useful, for example `cmake --build build -j32`, while keeping command output and system load reasonable. Run the router from the build directory or repo root, for example: `./build/TritonRoute -lef tech.lef -def design.def -guide route.guide -output routed.def`. Use `cmake --install build --prefix <dir>` only when you need staged `bin/` and `include/` outputs.

## Coding Style & Naming Conventions
This repository uses C++17 and `.clang-format` based on Google style with 4-space indentation and unsorted includes. Match surrounding conventions: class names such as `FlexDRWorker` use PascalCase, methods use lowerCamelCase, and filenames typically mirror the main class or subsystem (`FlexGR.cpp`, `io_parser_helper.cpp`). Prefer keeping new code within the existing subsystem directory instead of adding cross-cutting utilities at the top level.

## C++ Implementation Preferences
Keep function bodies out of headers by default; prefer declarations in headers and implementations in the corresponding `.cpp` file. Exceptions are acceptable for clearly trivial accessors, required `inline` definitions, template definitions that must remain visible, or very small headers where the implementation stays obvious and the total header remains roughly under 100 lines. Avoid adding `using` or `typedef` aliases unless they significantly improve readability, compatibility, or remove meaningful repetition without hiding domain meaning. Avoid introducing new function templates by default because they can obscure control flow and diagnostics; use templates only when generic reuse is clearly justified, localized, and as readable as established STL-style generic code.

Choose function and variable names that are explicit about routing-domain meaning, consistent with nearby code, and still concise. Prefer names that preserve existing subsystem vocabulary over generic abbreviations or newly invented terminology; shorten names only when clarity is not lost. Add comments sparingly but intentionally: important algorithmic blocks, boundary conditions, invariants, and non-obvious routing assumptions should include a short note describing the intent.

Avoid excessive defensive code that assumes arbitrary invalid inputs. Algorithmic code may assume that upstream and downstream stages preserve their documented invariants and data consistency. Keep checks focused on meaningful boundary conditions, external inputs, and invariants whose violation would indicate a real routing bug; do not clutter core logic with broad guard code for states the algorithm should never receive.

Do not add broad fallback branches for internally inconsistent algorithm states just to make arbitrary bad inputs limp forward. Prefer clear assumptions about stage-to-stage data consistency, and only validate cases that meaningfully protect routing correctness or expose a real invariant violation.

## Agent Collaboration Guidelines
For non-trivial design or implementation work, use subagents when available to keep individual contexts focused. Prefer delegating bounded exploration, independent implementation slices, or review/rebuttal tasks to economical models such as GPT-5.3 Codex Spark when the task fits their scope. Before committing to a broad approach for complex routing logic, ask a subagent to challenge the plan for vague reasoning, missed edge cases, and conflicts with existing code conventions. Integrate the critique into the final implementation plan instead of treating the first idea as settled.

For every code modification step, state the concrete plan before editing, including likely files, intended behavior, and validation commands. After C++ code changes, validate syntax and integration by running `clangd --compile-commands-dir=build --check=<changed-file.cpp>` on changed translation units where practical, then run a real compile with CMake, using up to 32 build threads when appropriate. Treat real `clangd` diagnostics or compile errors as blockers before reporting completion.

## Testing Guidelines
There is no top-level `ctest` or dedicated unit-test target in the current CMake setup. Validate changes by rebuilding and running representative routing flows on known LEF/DEF inputs. For parser-related work, the vendored LEF/DEF modules include their own `TEST/` assets; use them for focused regression checks when touching `module/` or `src/io/`. Document the exact command and dataset used in your PR notes.

## Commit & Pull Request Guidelines
The visible git history is minimal, so use short, imperative commit subjects such as `Fix via enclosure check in FlexDR`. Keep commits focused on one subsystem or behavior change. Pull requests should describe the routing scenario affected, summarize user-visible impact, list validation commands, and attach before/after logs or output diffs when behavior changes.

## Configuration Notes
CMake requires GCC 7+ compatible C++17 support plus Boost, OpenMP, Bison, and zlib. For global-detailed routing runs, ensure `POST9.dat` and `POWV9.dat` from `src/gr/flute/` are available in the working directory.
