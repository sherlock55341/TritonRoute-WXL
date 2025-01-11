#pragma once

#include <gta/ops/apply/Apply.cuh>
#ifndef __NVCC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#endif

namespace gta::ops::cuda::device {
__device__ void assign(data::Data &data, int iter, int i);
}