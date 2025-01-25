#include "AssignRefinement.cuh"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace gta::ops::cuda::device {
__forceinline__ __device__ bool compare_refinement(const data::Data &data,
                                                   int lhs, int rhs) {
    assert(lhs != rhs);
    assert(lhs >= 0 && lhs < data.num_guides);
    assert(rhs >= 0 && rhs < data.num_guides);
    if (data.ir_key_cost[lhs] != data.ir_key_cost[rhs])
        return data.ir_key_cost[lhs] > data.ir_key_cost[rhs];
    int l_lhs = data.ir_end[lhs] - data.ir_begin[lhs];
    int l_rhs = data.ir_end[rhs] - data.ir_begin[rhs];
    return l_lhs != l_rhs ? l_lhs < l_rhs : lhs < rhs;
}
} // namespace gta::ops::cuda::device

namespace gta::ops::cuda::kernel {
__global__ void init_ir_key_cost(data::Data data, int iter, int d) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    if (is_v != d)
        return;
    auto idx =
        data.ir_vio_cost_start[i] + (data.ir_track[i] - data.ir_track_low[i]);
    data.ir_key_cost[i] =
        data.ir_vio_cost_list[idx] + data.ir_via_vio_list[idx];
}

__global__ void select_one_step_refinement(data::Data data, int iter, int d,
                                           int *tag, bool *inc,
                                           int *inc_exist) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    if (is_v != d)
        return;
    assert(i >= 0 && i < data.num_guides);
    if (tag[i] > 0)
        return;
    if (data.ir_key_cost[i] == 0)
        return;
    bool select = true;
    int gb = max(0, data.ir_gcell_begin[i] - 1);
    int ge = min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
    // wire - wire conflict
    {
        auto p = data.ir_panel[i];
        for (auto g = gb; select && g <= ge; g++) {
            auto idx =
                data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
            for (auto list_idx = data.gcell_end_point_ir_start[idx];
                 list_idx < data.gcell_end_point_ir_start[idx + 1];
                 list_idx++) {
                auto j = data.gcell_end_point_ir_list[list_idx];
                if (i == j)
                    continue;
                assert(j >= 0 && j < data.num_guides);
                if (tag[j] > 0)
                    continue;
                if (device::compare_refinement(data, i, j) == false)
                    select = false;
            }
        }
        for (auto list_idx = data.ir_super_set_start[i];
             select && list_idx < data.ir_super_set_start[i + 1]; list_idx++) {
            auto j = data.ir_super_set_list[list_idx];
            if (i == j)
                continue;
            if (tag[j] > 0)
                continue;
            if (data.ir_key_cost[j] == 0)
                continue;
            if (device::compare_refinement(data, i, j) == false)
                select = false;
        }
    }
    // via - wire conflict
    if (data.layer_enable_via_wire_drc[l] == true) {
        int p_low = max(0, data.ir_panel[i] - 1);
        int p_high =
            min(data.layer_panel_start[l + 1] - data.layer_panel_start[l],
                data.ir_panel[i] + 1);
        if (data.ir_gcell_begin_via_offset[i] > 0) {
            for (auto p = p_low; select && p <= p_high; p++) {
                if (p == data.ir_panel[i])
                    continue;
                auto g = data.ir_gcell_begin[i];
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto cross_idx = data.gcell_cross_ir_start[idx];
                     select && cross_idx < data.gcell_cross_ir_start[idx + 1];
                     cross_idx++) {
                    auto j = data.gcell_cross_ir_list[cross_idx];
                    if (i == j)
                        continue;
                    if (tag[j] > 0)
                        continue;
                    if (data.ir_key_cost[j] == 0)
                        continue;
                    if (device::compare_refinement(data, i, j) == false)
                        select = false;
                }
            }
        }
        if (data.ir_gcell_end_via_offset[i] > 0 &&
            data.ir_gcell_begin[i] != data.ir_gcell_end[i]) {
            for (auto p = p_low; select && p <= p_high; p++) {
                if (p == data.ir_panel[i])
                    continue;
                auto g = data.ir_gcell_end[i];
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto cross_idx = data.gcell_cross_ir_start[idx];
                     select && cross_idx < data.gcell_cross_ir_start[idx + 1];
                     cross_idx++) {
                    auto j = data.gcell_cross_ir_list[cross_idx];
                    if (i == j)
                        continue;
                    if (tag[j] > 0)
                        continue;
                    if (data.ir_key_cost[j] == 0)
                        continue;
                    if (device::compare_refinement(data, i, j) == false)
                        select = false;
                }
            }
        }
        for (auto p = p_low; select && p < p_high; p++) {
            if (p == data.ir_panel[i])
                continue;
            for (auto g = data.ir_gcell_begin[i] + 1; g < data.ir_gcell_end[i];
                 g++) {
                auto idx = data.layer_gcell_start[l] +
                           p * data.layer_panel_length[l] + g;
                for (auto list_idx = data.gcell_end_point_ir_start[idx];
                     list_idx < data.gcell_end_point_ir_start[idx + 1];
                     list_idx++) {
                    auto j = data.gcell_end_point_ir_list[list_idx];
                    if (i == j)
                        continue;
                    assert(j >= 0 && j < data.num_guides);
                    if (tag[j] > 0)
                        continue;
                    if (device::compare_refinement(data, i, j) == false)
                        select = false;
                }
            }
        }
    }
    if (select) {
        inc[i] = true;
        inc_exist[0] = 1;
    }
}

__global__ void assign_one_step_refinement(data::Data data, int iter,
                                           bool *inc) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    if (inc[i]) {
        device::assign(data, iter, i);
        auto idx = data.ir_vio_cost_start[i] +
                   (data.ir_track[i] - data.ir_track_low[i]);
        data.ir_key_cost[i] =
            data.ir_vio_cost_list[idx] + data.ir_via_vio_list[idx];
    }
}

__global__ void tag_inc_refinement(data::Data data, int *tag, bool *inc,
                                   int *inc_exist) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    if (inc[i])
        tag[i]++;
    inc[i] = false;
    inc_exist[0] = 0;
}
} // namespace gta::ops::cuda::kernel

namespace gta::ops::cuda {

#define BLOCKS(x, y) (((x) + (y) - 1) / (y))

void assign_refinement(data::Data &data, int iter, int d) {
    assert(data.device == data::Device::CUDA);
    auto tp_0 = std::chrono::high_resolution_clock::now();
    kernel::init_ir_key_cost<<<BLOCKS(data.num_guides, 256), 256>>>(data, iter,
                                                                    d);
    int *tag = nullptr;
    bool *inc = nullptr;
    int *inc_exist_device = nullptr;
    int *inc_exist_host = nullptr;
    int inner_iter = 0;
    auto e = cudaMalloc((void **)&tag, sizeof(int) * data.num_guides);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaMalloc((void **)&inc, sizeof(bool) * data.num_guides);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaMalloc((void **)&inc_exist_device, sizeof(int));
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    inc_exist_host = (int *)malloc(sizeof(int));
    e = cudaMemset(tag, 0, sizeof(int) * data.num_guides);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaMemset(inc, 0, sizeof(bool) * data.num_guides);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaMemset(inc_exist_device, 0, sizeof(int));
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    inc_exist_host[0] = 1;
    // std::cout << "data.num_guides : " << data.num_guides << std::endl;
    while (inc_exist_host[0]) {
        inner_iter++;
        kernel::
            select_one_step_refinement<<<BLOCKS(data.num_guides, 256), 256>>>(
                data, iter, d, tag, inc, inc_exist_device);
        // cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                      << cudaGetErrorString(e) << std::endl;
            exit(0);
        }
        kernel::
            assign_one_step_refinement<<<BLOCKS(data.num_guides, 256), 256>>>(
                data, iter, inc);
        // cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                      << cudaGetErrorString(e) << std::endl;
            exit(0);
        }
        if (inner_iter % 3 == 0) {
            e = cudaMemcpy(inc_exist_host, inc_exist_device, sizeof(int),
                           cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__
                          << " " << cudaGetErrorString(e) << std::endl;
                exit(0);
            }
        }
        kernel::tag_inc_refinement<<<BLOCKS(data.num_guides, 256), 256>>>(
            data, tag, inc, inc_exist_device);
        // cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                      << cudaGetErrorString(e) << std::endl;
            exit(0);
        }
        if (inner_iter % 1000 == 0)
            std::cout << inner_iter << " inner loop " << inc_exist_host[0]
                      << std::endl;
    }
    std::cout << inner_iter << " inner loop in total" << std::endl;
    e = cudaFree(tag);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaFree(inc);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    e = cudaFree(inc_exist_device);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    free(inc_exist_host);
    tag = nullptr;
    inc = nullptr;
    inc_exist_device = nullptr;
    inc_exist_host = nullptr;
    auto tp_1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inner Iteration : " << inner_iter << " (" << iter << ", " << d
              << ")" << std::endl;
    std::cout << "ASSIGN REFINEMENT on device (" << iter << ", " << d << ") "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tp_1 -
                                                                       tp_0)
                         .count() /
                     1e3
              << " s" << std::endl;
}
} // namespace gta::ops::cuda