#include "Init.hpp"
#include "InitKernel.hpp"
#include <iostream>

namespace gta::ops {
void init(data::Data &data, int iter, int d) {
    if (data.device == data::Device::CPU) {
        cpu::init(data, iter, d);
    } else if (data.device == data::Device::CUDA) {
        cuda::init(data, iter, d);
    } else {
        std::cout << "[ERROR] UNIMPLEMENTED " << __FILE__ << ":" << __LINE__
                  << std::endl;
        exit(0);
    }
}
} // namespace gta::ops