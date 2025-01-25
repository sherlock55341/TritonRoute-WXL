#include "InitKernel.hpp"
#include <cassert>
#include <gta/database/Data.cuh>
#include <gta/ops/apply/Apply.cuh>
#include <iostream>
#ifndef __NVCC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#endif

__managed__ int counter = 0;

namespace gta::ops::cuda::device {
__device__ void getBlkVio(data::Data &data, int i, int j) {
    if (data.ir_net[i] == data.b_net[j])
        return;
    atomicAdd(&counter, 1);
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    auto begin = data.ir_begin[i] - data.layer_width[l] / 2;
    auto end = data.ir_end[i] + data.layer_width[l] / 2;
    auto d =
        (data.b_top[j] - data.b_bottom[j] > data.b_right[j] - data.b_left[j]);
    if (data.b_top[j] - data.b_bottom[j] == data.b_right[j] - data.b_left[j])
        d = is_v;
    auto prl = max(
        0, (is_v ? (min(data.b_top[j], end) - max(data.b_bottom[j], begin))
                 : (min(data.b_right[j], end) - max(data.b_left[j], begin))));
    if (data.ir_gcell_begin[i] == data.ir_gcell_end[i])
        prl = 0;
    auto vio_begin = (is_v ? data.b_left[j] : data.b_bottom[j]) -
                     data.layer_width[l] / 2 + 1;
    auto vio_end =
        (is_v ? data.b_right[j] : data.b_top[j]) + data.layer_width[l] / 2 - 1;
    int s = 0;
    if (d == is_v ||
        data.layer_enable_corner_spacing[l]) { // consider prl spacing
                                               // constraint
        auto width = (is_v ? (data.b_right[j] - data.b_left[j])
                           : (data.b_top[j] - data.b_bottom[j]));
        if (data.b_use_min_width[j])
            width = data.layer_width[l];
        s = data::cuda::device::findPRLSpacing(
            data, l, max(width, data.layer_width[l]), prl);
        vio_begin -= s;
        vio_end += s;
    }
    int vio_track_begin = ceil(1.0 * (vio_begin - data.layer_track_start[l]) /
                               data.layer_track_step[l]);
    int vio_track_end = floor(1.0 * (vio_end - data.layer_track_start[l]) /
                              data.layer_track_step[l]);
    vio_track_begin = max(vio_track_begin, data.ir_track_low[i]);
    vio_track_end = min(vio_track_end, data.ir_track_high[i]);
    for (auto t = vio_track_begin; t <= vio_track_end; t++) {
        auto coor = data.layer_track_start[l] + data.layer_track_step[l] * t;
        int extension = 0;
        if (data.layer_width[l] <
            data.layer_eol_width[l]) { // consider eol spacing constraint
            auto upperEdge =
                coor + data.layer_width[l] / 2 + data.layer_eol_within[l];
            auto lowerEdge =
                coor - data.layer_width[l] / 2 - data.layer_eol_within[l];
            auto overlap = max(
                0,
                min(upperEdge, (is_v ? data.b_right[j] : data.b_top[j])) -
                    max(lowerEdge, (is_v ? data.b_left[j] : data.b_bottom[j])));
            if (overlap > 0)
                extension = data.layer_eol_spacing[l];
        }
        if (data.layer_enable_corner_spacing[l] == true)
            extension = max(extension, s);
        auto overlap = max(
            0, min((is_v ? data.b_top[j] : data.b_right[j]) + extension, end) -
                   max((is_v ? data.b_bottom[j] : data.b_left[j]) - extension,
                       begin));
        if (overlap > 0)
            overlap = max(overlap, data.layer_pitch[l]);
        data.ir_vio_cost_list[data.ir_vio_cost_start[i] + t -
                              data.ir_track_low[i]] += overlap;
    }
}
} // namespace gta::ops::cuda::device

namespace gta::ops::cuda::kernel {
__global__ void init_begin_end(data::Data data, int d) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    if (is_v != d)
        return;
    if (data.ir_has_ap[i] == true)
        return;
    bool hasBegin = false;
    bool hasEnd = false;
    int begin = 1e9;
    int end = -1e9;
    for (auto idx = data.ir_nbr_start[i]; idx < data.ir_nbr_start[i + 1];
         idx++) {
        auto j = data.ir_nbr_list[idx];
        if (data.ir_track[j] != -1) {
            assert(data.ir_track[j] >= data.ir_track_low[j]);
            assert(data.ir_track[j] <= data.ir_track_high[j]);
            auto nbr_layer = data.ir_layer[j];
            auto coor = data.layer_track_start[nbr_layer] +
                        data.layer_track_step[nbr_layer] * data.ir_track[j];
            auto via_layer = (nbr_layer + l) / 2;
            if (data.ir_panel[j] == data.ir_gcell_begin[i]) {
                hasBegin = true;
                begin = min(begin, coor);
            }
            if (data.ir_panel[j] == data.ir_gcell_end[i]) {
                hasEnd = true;
                end = max(end, coor);
            }
        }
    }
    if (hasBegin == false) {
        if (is_v)
            begin = data.gcell_start_y +
                    data.gcell_step_y * (2 * data.ir_gcell_begin[i] + 1) / 2;
        else
            begin = data.gcell_start_x +
                    data.gcell_step_x * (2 * data.ir_gcell_begin[i] + 1) / 2;
    }
    if (hasEnd == false) {
        if (is_v)
            end = data.gcell_start_y +
                  data.gcell_step_y * (2 * data.ir_gcell_end[i] + 1) / 2;
        else
            end = data.gcell_start_x +
                  data.gcell_step_x * (2 * data.ir_gcell_end[i] + 1) / 2;
    }
    if (begin > end) {
        auto tmp = begin;
        begin = end;
        end = tmp;
    }
    if (begin == end)
        end++;
    data.ir_begin[i] = begin;
    data.ir_end[i] = end;
}

__global__ void init_blk_vio(data::Data data, int d) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    if (is_v != d)
        return;
    for (auto vio_i = data.ir_vio_cost_start[i];
         vio_i < data.ir_vio_cost_start[i + 1]; vio_i++) {
        data.ir_vio_cost_list[vio_i] = 0;
        data.ir_align_list[vio_i] = 0;
        data.ir_via_vio_list[vio_i] = 0;
    }
    auto p = data.ir_panel[i];
    auto gb = max(0, data.ir_gcell_begin[i] - 1);
    auto ge = min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
    for (auto g = gb; g <= ge; g++) {
        auto idx =
            data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
        for (auto list_idx = data.gcell_end_point_blk_start[idx];
             list_idx < data.gcell_end_point_blk_start[idx + 1]; list_idx++) {
            auto j = data.gcell_end_point_blk_list[list_idx];
            if (data.b_gcell_begin[j] == g ||
                (data.b_gcell_end[j] == g && data.b_gcell_begin[j] < gb)) {
                device::getBlkVio(data, i, j);
            }
        }
    }
    for (auto idx = data.blk_super_set_start[i];
         idx < data.blk_super_set_start[i + 1]; idx++) {
        auto j = data.blk_super_set_list[idx];
        device::getBlkVio(data, i, j);
    }
}

__global__ void init_blk_vio_offset(data::Data data, int d, int *offset) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    if (data.layer_direction[l] != d)
        return;
    offset[i + 1] = 0;
    for (auto vio_i = data.ir_vio_cost_start[i];
         vio_i < data.ir_vio_cost_start[i + 1]; vio_i++) {
        data.ir_vio_cost_list[vio_i] = 0;
        data.ir_align_list[vio_i] = 0;
        data.ir_via_vio_list[vio_i] = 0;
    }
    auto p = data.ir_panel[i];
    auto gb = max(0, data.ir_gcell_begin[i] - 1);
    auto ge = min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
    for (auto g = gb; g <= ge; g++) {
        auto idx =
            data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
        for (auto list_idx = data.gcell_end_point_blk_start[idx];
             list_idx < data.gcell_end_point_blk_start[idx + 1]; list_idx++) {
            auto j = data.gcell_end_point_blk_list[list_idx];
            if (data.ir_net[i] == data.b_net[j])
                continue;
            if (data.b_gcell_begin[j] == g ||
                (data.b_gcell_end[j] == g && data.b_gcell_begin[j] < gb))
                offset[i + 1]++;
        }
    }
    for (auto idx = data.blk_super_set_start[i];
         idx < data.blk_super_set_start[i + 1]; idx++) {
        auto j = data.blk_super_set_list[idx];
        if (data.ir_net[i] == data.b_net[j])
            continue;
        offset[i + 1]++;
    }
}

__global__ void init_blk_vio_work(data::Data data, int d, int *offset,
                                  int *work_i, int *work_j) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    if (data.layer_direction[l] != d)
        return;
    auto p = data.ir_panel[i];
    auto gb = max(0, data.ir_gcell_begin[i] - 1);
    auto ge = min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
    int count = 0;
    for (auto g = gb; g <= ge; g++) {
        auto idx =
            data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
        for (auto list_idx = data.gcell_end_point_blk_start[idx];
             list_idx < data.gcell_end_point_blk_start[idx + 1]; list_idx++) {
            auto j = data.gcell_end_point_blk_list[list_idx];
            if (data.ir_net[i] == data.b_net[j])
                continue;
            if (data.b_gcell_begin[j] == g ||
                (data.b_gcell_end[j] == g && data.b_gcell_begin[j] < gb)) {
                work_i[offset[i] + count] = i;
                work_j[offset[i] + count] = j;
                count++;
            }
        }
    }
    for (auto idx = data.blk_super_set_start[i];
         idx < data.blk_super_set_start[i + 1]; idx++) {
        auto j = data.blk_super_set_list[idx];
        if (data.ir_net[i] == data.b_net[j])
            continue;
        work_i[offset[i] + count] = i;
        work_j[offset[i] + count] = j;
        count++;
    }
    if (offset[i] + count != offset[i + 1]) {
        printf("[ERROR] %s:%d\n", __FILE__, __LINE__);
    }
}

__global__ void init_blk_vio(data::Data data, int num_works, int *work_i,
                             int *work_j) {
    __shared__ int layer_width_start[32];
    __shared__ int layer_prl_start[32];
    __shared__ int layer_spacing_start[32];
    __shared__ int layer_width[64];
    __shared__ int layer_prl[64];
    __shared__ int layer_spacing[512];
    for (int l = threadIdx.x; l <= data.num_layers; l += blockDim.x) {
        layer_width_start[l] = data.layer_spacing_table_width_start[l];
        layer_prl_start[l] = data.layer_spacing_table_prl_start[l];
        layer_spacing_start[l] = data.layer_spacing_table_spacing_start[l];
    }
    for (int i = threadIdx.x;
         i < data.layer_spacing_table_width_start[data.num_layers];
         i += blockDim.x)
        layer_width[i] = data.layer_spacing_table_width[i];
    for (int i = threadIdx.x;
         i < data.layer_spacing_table_prl_start[data.num_layers];
         i += blockDim.x)
        layer_prl[i] = data.layer_spacing_table_prl[i];
    for (int i = threadIdx.x;
         i < data.layer_spacing_table_spacing_start[data.num_layers];
         i += blockDim.x)
        layer_spacing[i] = data.layer_spacing_table_spacing[i];
    __syncthreads();
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_works)
        return;
    int i = work_i[idx];
    int j = work_j[idx];
    if (data.ir_net[i] == data.b_net[j])
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    int begin = data.ir_begin[i] - data.layer_width[l] / 2;
    int end = data.ir_end[i] + data.layer_width[l] / 2;
    auto d =
        (data.b_top[j] - data.b_bottom[j] > data.b_right[j] - data.b_left[j]);
    if (data.b_top[j] - data.b_bottom[j] == data.b_right[j] - data.b_left[j])
        d = is_v;
    int prl = max(
        0, (is_v ? (min(data.b_top[j], end) - max(data.b_bottom[j], begin))
                 : (min(data.b_right[j], end) - max(data.b_left[j], begin))));
    if (data.ir_gcell_begin[i] == data.ir_gcell_end[i])
        prl = 0;
    auto vio_begin = (is_v ? data.b_left[j] : data.b_bottom[j]) -
                     data.layer_width[l] / 2 + 1;
    auto vio_end =
        (is_v ? data.b_right[j] : data.b_top[j]) + data.layer_width[l] / 2 - 1;
    int s = 0;
    if (d == is_v ||
        data.layer_enable_corner_spacing[l]) { // consider prl spacing
                                               // constraint
        auto width = (is_v ? (data.b_right[j] - data.b_left[j])
                           : (data.b_top[j] - data.b_bottom[j]));
        if (data.b_use_min_width[j])
            width = data.layer_width[l];
        // s = data::cuda::device::findPRLSpacing(
        //     data, l, max(width, data.layer_width[l]), prl);
        int q_width = max(width, data.layer_width[l]);
        int row = 0, col = 0;
        const int row_num = layer_width_start[l + 1] - layer_width_start[l];
        const int col_num = layer_prl_start[l + 1] - layer_prl_start[l];
        while (row + 1 < row_num &&
               q_width > layer_width[layer_width_start[l] + row + 1])
            row++;
        while (col + 1 < col_num &&
               prl > layer_prl[layer_prl_start[l] + col + 1])
            col++;
        s = layer_spacing[layer_spacing_start[l] + row * col_num + col];
        vio_begin -= s;
        vio_end += s;
    }
    int vio_track_begin = ceil(1.0 * (vio_begin - data.layer_track_start[l]) /
                               data.layer_track_step[l]);
    int vio_track_end = floor(1.0 * (vio_end - data.layer_track_start[l]) /
                              data.layer_track_step[l]);
    vio_track_begin = max(vio_track_begin, data.ir_track_low[i]);
    vio_track_end = min(vio_track_end, data.ir_track_high[i]);
    for (auto t = vio_track_begin; t <= vio_track_end; t++) {
        auto coor = data.layer_track_start[l] + data.layer_track_step[l] * t;
        int extension = 0;
        if (data.layer_width[l] <
            data.layer_eol_width[l]) { // consider eol spacing constraint
            auto upperEdge =
                coor + data.layer_width[l] / 2 + data.layer_eol_within[l];
            auto lowerEdge =
                coor - data.layer_width[l] / 2 - data.layer_eol_within[l];
            auto overlap = max(
                0,
                min(upperEdge, (is_v ? data.b_right[j] : data.b_top[j])) -
                    max(lowerEdge, (is_v ? data.b_left[j] : data.b_bottom[j])));
            if (overlap > 0)
                extension = data.layer_eol_spacing[l];
        }
        if (data.layer_enable_corner_spacing[l] == true)
            extension = max(extension, s);
        auto overlap = max(
            0, min((is_v ? data.b_top[j] : data.b_right[j]) + extension, end) -
                   max((is_v ? data.b_bottom[j] : data.b_left[j]) - extension,
                       begin));
        if (overlap > 0)
            overlap = max(overlap, data.layer_pitch[l]);
        // data.ir_vio_cost_list[data.ir_vio_cost_start[i] + t -
        //                       data.ir_track_low[i]] += overlap;
        atomicAdd(data.ir_vio_cost_list +
                      (data.ir_vio_cost_start[i] + t - data.ir_track_low[i]),
                  overlap);
    }
}

__global__ void init_apply(data::Data data, int iter, int d) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= data.num_guides)
        return;
    auto l = data.ir_layer[i];
    auto is_v = data.layer_direction[l];
    if (is_v != d)
        return;
    data::cuda::device::apply(data, iter, i, 1);
}
} // namespace gta::ops::cuda::kernel

namespace gta::ops::cuda {

#define BLOCKS(x, y) (((x) + ((y) - 1)) / (y))

void init(data::Data &data, int iter, int d) {
    std::cout << "init cuda" << std::endl;
    kernel::init_begin_end<<<BLOCKS(data.num_guides, 256), 256>>>(data, d);
    counter = 0;
    cudaDeviceSynchronize();
    auto e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e)
                  << std::endl;
        exit(0);
    }
    // std::cout << "init begin end" << std::endl;
    std::cout << "total blk : " << data.num_blks << std::endl;
    // kernel::init_blk_vio<<<BLOCKS(data.num_guides, 256), 256>>>(data, d);
    int *offset_h = nullptr, *offset_d = nullptr;
    offset_h = (int *)malloc(sizeof(int) * (data.num_guides + 1));
    cudaMalloc(&offset_d, sizeof(int) * (data.num_guides + 1));
    cudaMemset(offset_d, 0, sizeof(int) * (data.num_guides + 1));
    kernel::init_blk_vio_offset<<<BLOCKS(data.num_guides, 256), 256>>>(
        data, d, offset_d);
    cudaMemcpy(offset_h, offset_d, sizeof(int) * (data.num_guides + 1),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < data.num_guides; i++)
        offset_h[i + 1] += offset_h[i];
    int *work_i_d = nullptr, *work_j_d = nullptr;
    cudaMemcpy(offset_d, offset_h, sizeof(int) * (data.num_guides + 1),
               cudaMemcpyHostToDevice);
    cudaMalloc(&work_i_d, sizeof(int) * offset_h[data.num_guides]);
    cudaMalloc(&work_j_d, sizeof(int) * offset_h[data.num_guides]);
    kernel::init_blk_vio_work<<<BLOCKS(data.num_guides, 256), 256>>>(
        data, d, offset_d, work_i_d, work_j_d);
    kernel::init_blk_vio<<<BLOCKS(offset_h[data.num_guides], 256), 256>>>(
        data, offset_h[data.num_guides], work_i_d, work_j_d);
    cudaDeviceSynchronize();
    std::cout << "old : " << counter << std::endl;
    std::cout << "new : " << offset_h[data.num_guides] << std::endl;
    free(offset_h);
    cudaFree(offset_d);
    cudaFree(work_i_d);
    cudaFree(work_j_d);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e)
                  << std::endl;
        exit(0);
    }
    // std::cout << "init blk vio " << acc_h[0] << std::endl;
    kernel::init_apply<<<BLOCKS(data.num_guides, 256), 256>>>(data, iter, d);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e)
                  << std::endl;
        exit(0);
    }
    // std::cout << "init apply" << std::endl;
    std::cout << "finish init cuda" << std::endl;
    // cudaFree(acc_d);
    // free(acc_h);
}
} // namespace gta::ops::cuda