#pragma once

#include "Data.hpp"
#ifndef __NVCC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#endif

namespace gta::data::cuda::device {
__device__ int findPRLSpacing(Data &data, int l, int width, int prl);
}