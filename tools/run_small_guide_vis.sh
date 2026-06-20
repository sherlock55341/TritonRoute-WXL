#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${repo_dir}/build"
run_dir="${build_dir}/guide_symmetry_check"
params="${run_dir}/small.params"
vis_dir="${run_dir}/vis_small"

cmake --build "${build_dir}" -j"$(nproc)"

mkdir -p "${run_dir}"
if [[ ! -f "${params}" ]]; then
  cat > "${params}" <<'PARAMS'
lef:/home/cyzhao/benchmark/primarius/outdata/ispd18_test1.input.lef
def:/home/cyzhao/benchmark/primarius/outdata/pattern_route_lay.def
output:guide_symmetry_check/small.def
outputguide:guide_symmetry_check/small.guide
threads:1
verbose:2
drouteEndIterNum:3
PARAMS
fi

(
  cd "${build_dir}"
  ./TritonRoute guide_symmetry_check/small.params > guide_symmetry_check/small.log 2>&1
)

rm -rf "${vis_dir}"
python3 "${script_dir}/visualize_guide_symmetry.py" \
  "${run_dir}/small.guide" \
  -o "${vis_dir}"

echo "small guide visualization:"
echo "${vis_dir}/index.html"
