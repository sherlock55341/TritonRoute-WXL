#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${repo_dir}/build"
run_dir="${build_dir}/guide_symmetry_check"

run_case() {
  local name="$1"
  local lef="$2"
  local def="$3"
  local params="${run_dir}/${name}.params"
  local log="${run_dir}/${name}.log"

  cat > "${params}" <<PARAMS
lef:${lef}
def:${def}
output:guide_symmetry_check/${name}.def
outputguide:guide_symmetry_check/${name}.guide
outputguideStagePrefix:guide_symmetry_check/${name}
threads:1
verbose:2
drouteEndIterNum:3
PARAMS

  local route_status=0
  set +e
  (
    cd "${build_dir}"
    ./TritonRoute "guide_symmetry_check/${name}.params" > "guide_symmetry_check/${name}.log" 2>&1
  )
  route_status=$?
  set -e

  if (( route_status != 0 )); then
    echo "${name}: TritonRoute failed with status ${route_status}; see ${log}" >&2
    return "${route_status}"
  fi

  if grep -En '(^ERROR|Reader returns bad status)' "${log}" >&2; then
    echo "${name}: router log contains errors; see ${log}" >&2
    return 1
  fi

  echo "${name}: route log ok (${log})"
}

visualize_case() {
  local name="$1"
  local vis_dir="${run_dir}/vis_${name}"

  rm -rf "${vis_dir}"
  if ! python3 "${script_dir}/visualize_guide_symmetry.py" \
    "${run_dir}/${name}" \
    -o "${vis_dir}"; then
    echo "${name}: guide visualization failed" >&2
    return 1
  fi

  echo "${name} guide visualization:"
  echo "${vis_dir}/index.html"
}

cmake --build "${build_dir}" -j"$(nproc)"

mkdir -p "${run_dir}"

status=0

run_case \
  "small" \
  "/home/cyzhao/benchmark/primarius/outdata/ispd18_test1.input.lef" \
  "/home/cyzhao/benchmark/primarius/outdata/pattern_route_lay.def" || status=1

run_case \
  "pattern_route0612" \
  "/home/cyzhao/benchmark/primarius/outdata/pattern_route0612.lef" \
  "/home/cyzhao/benchmark/primarius/outdata/pattern_route0612.def" || status=1

visualize_case "small" || status=1
visualize_case "pattern_route0612" || status=1

exit "${status}"
