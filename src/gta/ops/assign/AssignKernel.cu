#include "AssignInitial.cuh"
#include "AssignRefinement.cuh"
#include "AssignKernel.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace gta::ops::cuda {
void assign(data::Data &data, int iter, int d) {
    assert(data.device == data::Device::CUDA);
    if (iter == 0) {
        ops::cuda::assign_initial(data, iter, d);
    }
    else{
        ops::cuda::assign_refinement(data, iter, d);
    }
}
} // namespace gta::ops