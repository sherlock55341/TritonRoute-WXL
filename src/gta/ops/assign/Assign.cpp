#include "Assign.hpp"
#include "AssignKernel.hpp"
#include <cstdlib>
#include <iostream>

namespace gta::ops {
void assign(data::Data &data, int iter, int d) {
    if (data.device == data::Device::CPU)
        cpu::assign(data, iter, d);
    else if (data.device == data::Device::CUDA) {
        cuda::assign(data, iter, d);
    } else {
        std::cout << "[ERROR] UNIMPLEMENTED " << __FILE__ << ":" << __LINE__
                  << std::endl;
        exit(0);
    }
}
} // namespace gta::ops