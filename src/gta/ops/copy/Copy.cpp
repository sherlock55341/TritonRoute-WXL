#include "Copy.hpp"
#include "CopyKernel.hpp"
#include "FreeKernel.hpp"
#include <iostream>

namespace gta::ops {
void free_data(data::Data &data) {
    if (data.device == data::Device::CPU)
        cpu::free_data(data);
    else if (data.device == data::Device::CUDA)
        cuda::free_data(data);
    else {
        std::cout << "[ERROR] UNIMPLEMENTED " << __FILE__ << ":" << __LINE__
                  << std::endl;
        exit(0);
    }
}

void malloc_device_data(data::Data &data_h, data::Data &data_d) {
    assert(data_h.device == data::Device::CPU);
    cpu::malloc_data(data_h, data_d);
}

void h2d_data(data::Data &data_h, data::Data &data_d) {
    assert(data_h.device == data::Device::CPU);
    assert(data_d.device == data::Device::CUDA);
    cpu::h2d_data(data_h, data_d);
}

void d2h_data(data::Data &data_h, data::Data &data_d) {
    assert(data_h.device == data::Device::CPU);
    assert(data_d.device == data::Device::CUDA);
    cpu::d2h_data(data_h, data_d);
}
} // namespace gta::ops