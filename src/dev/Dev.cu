#include "Dev.hpp"
#include <iostream>
#ifndef __NVCC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#endif

namespace dev {
void set_device(int dev_id) {
    int gpu_count = -1;
    cudaGetDeviceCount(&gpu_count);
    if (dev_id < 0 || dev_id >= gpu_count) {
        std::cout << "[CUDA ERROR] select device " << dev_id
                  << ", but there are only " << gpu_count << " devices"
                  << std::endl;
        exit(0);
    }
    cudaSetDevice(dev_id);
    cudaFree(0);
}
} // namespace dev