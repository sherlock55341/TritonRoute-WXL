#pragma once

#include <gta/database/Data.hpp>
#ifndef __NVCC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#endif

namespace gta::data::cuda::device{
__device__ void apply(data::Data& data, int iter, int i, int coef);
}
