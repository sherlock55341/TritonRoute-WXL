#include "AssignInitial.cuh"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace gta::ops::cuda::kernel {
__global__ void select_one_step_initial(data::Data data, int iter, int d,
                                        int *rank, int *tag, bool *inc,
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
                 select && list_idx < data.gcell_end_point_ir_start[idx + 1];
                 list_idx++) {
                auto j = data.gcell_end_point_ir_list[list_idx];
                if (i == j)
                    continue;
                assert(j >= 0 && j < data.num_guides);
                if (tag[j] > 0)
                    continue;
                if (rank[j] < rank[i])
                    select = false;
                // return ;
            }
        }
        for (auto list_idx = data.ir_super_set_start[i];
             select && list_idx < data.ir_super_set_start[i + 1]; list_idx++) {
            auto j = data.ir_super_set_list[list_idx];
            if (i == j)
                continue;
            if (tag[j] > 0)
                continue;
            if (rank[j] < rank[i])
                select = false;
        }
    }
    if (select) {
        inc[i] = true;
        // atomicAdd(inc_exist, 1);
        inc_exist[0] = 1;
    }
}

__global__ void assign_one_step_initial(data::Data data, int iter, bool *inc) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    if (inc[i])
        device::assign(data, iter, i);
}

__global__ void tag_inc_initial(data::Data data, int *tag, bool *inc,
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

#define BLOCKS(x, y) (((x) + ((y) - 1)) / (y))

void generate_rank_initial(data::Data &data, int d, int *rank) {
    // std::cout << "generate rank begin" << std::endl;
    bool *ir_has_proj_ap = (bool *)malloc(sizeof(bool) * data.num_guides);
    float *ir_wl_weight = (float *)malloc(sizeof(float) * data.num_guides);
    short *ir_panel = (short *)malloc(sizeof(short) * data.num_guides);
    int *ir_begin = (int *)malloc(sizeof(int) * data.num_guides);
    int *ir_end = (int *)malloc(sizeof(int) * data.num_guides);
    short *ir_layer = (short *)malloc(sizeof(short) * data.num_guides);
    int *layer_direction = (int *)malloc(sizeof(int) * data.num_layers);
    int *rank_host = (int *)malloc(sizeof(int) * data.num_guides);
    // std::cout << "begin cuda memcpy" << std::endl;
    cudaMemcpy(ir_has_proj_ap, data.ir_has_proj_ap,
               sizeof(bool) * data.num_guides, cudaMemcpyDeviceToHost);
    cudaMemcpy(ir_wl_weight, data.ir_wl_weight, sizeof(float) * data.num_guides,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ir_panel, data.ir_panel, sizeof(short) * data.num_guides,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ir_begin, data.ir_begin, sizeof(int) * data.num_guides,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ir_end, data.ir_end, sizeof(int) * data.num_guides,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ir_layer, data.ir_layer, sizeof(short) * data.num_guides,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(layer_direction, data.layer_direction,
               sizeof(int) * data.num_layers, cudaMemcpyDeviceToHost);
    // std::cout << "finish cuda memcpy" << std::endl;
    auto panel_num = (d ? data.num_gcells_x : data.num_gcells_y);
    std::vector<std::vector<int>> ir_groups(panel_num);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = ir_layer[i];
        auto is_v = layer_direction[l];
        if (is_v == d)
            ir_groups[ir_panel[i]].push_back(i);
    }
#pragma omp parallel for
    for (auto &ir_group : ir_groups) {
        std::sort(ir_group.begin(), ir_group.end(), [&](int lhs, int rhs) {
            if (ir_has_proj_ap[lhs] != ir_has_proj_ap[rhs])
                return ir_has_proj_ap[lhs] > ir_has_proj_ap[rhs];
            auto coef_lhs = 1.0 * std::abs(ir_wl_weight[lhs] /
                                           (ir_end[lhs] - ir_begin[lhs] + 1));
            auto coef_rhs = 1.0 * std::abs(ir_wl_weight[rhs] /
                                           (ir_end[rhs] - ir_begin[rhs] + 1));
            return coef_lhs > coef_rhs;
        });
        for (int i = 0; i < ir_group.size(); i++)
            rank_host[ir_group[i]] = i;
    }
#pragma omp barrier
    int max_panel_iroute = 0;
    for (auto &ir_group : ir_groups)
        max_panel_iroute = std::max(max_panel_iroute, (int)ir_group.size());
    // std::cout << "Max # iroute in a panel : " << max_panel_iroute <<
    // std::endl;
    cudaMemcpy(rank, rank_host, sizeof(int) * data.num_guides,
               cudaMemcpyHostToDevice);
    // std::cout << "begin free" << std::endl;
    free(ir_has_proj_ap);
    free(ir_wl_weight);
    free(ir_panel);
    free(ir_begin);
    free(ir_end);
    free(ir_layer);
    free(layer_direction);
    free(rank_host);
    // std::cout << "generate rank end" << std::endl;
}

void assign_initial(data::Data &data, int iter, int d) {
    assert(data.device == data::Device::CUDA);
    auto tp_0 = std::chrono::high_resolution_clock::now();
    int *rank = nullptr;
    cudaMalloc((void **)&rank, sizeof(int) * data.num_guides);
    generate_rank_initial(data, d, rank);
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
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
    // std::cout << "data.num_guides : " << data.num_guides << std::endl;
    while (inc_exist_host[0]) {
        inner_iter++;
        kernel::select_one_step_initial<<<BLOCKS(data.num_guides, 256), 256>>>(
            data, iter, d, rank, tag, inc, inc_exist_device);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                      << cudaGetErrorString(e) << std::endl;
            exit(0);
        }
        // break;
        kernel::assign_one_step_initial<<<BLOCKS(data.num_guides, 256), 256>>>(
            data, iter, inc);
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                      << cudaGetErrorString(e) << std::endl;
            exit(0);
        }
        if (inner_iter % 3 == 0) {
            e = cudaMemcpy(inc_exist_host, inc_exist_device, sizeof(int),
                           cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__
                          << " " << cudaGetErrorString(e) << std::endl;
                exit(0);
            }
        }
        kernel::tag_inc_initial<<<BLOCKS(data.num_guides, 256), 256>>>(
            data, tag, inc, inc_exist_device);
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
    e = cudaFree(rank);
    if (e != cudaSuccess) {
        std::cout << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " "
                  << cudaGetErrorString(e) << std::endl;
        exit(0);
    }
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
    rank = nullptr;
    tag = nullptr;
    inc = nullptr;
    inc_exist_device = nullptr;
    inc_exist_host = nullptr;
    auto tp_1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inner Iteration : " << inner_iter << " (" << iter << ", " << d
              << ")" << std::endl;
    std::cout << "ASSIGN INITIAL on device (" << iter << ", " << d << ") "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tp_1 -
                                                                       tp_0)
                         .count() /
                     1e3
              << " s" << std::endl;
}
} // namespace gta::ops::cuda