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

namespace gta::ops::cuda::device {
__device__ void getBlkVio(data::Data &data, int i, int j) {
    if (data.ir_net[i] == data.b_net[j])
        return;
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
    cudaDeviceSynchronize();
    auto e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e)
                  << std::endl;
        exit(0);
    }
    // std::cout << "init begin end" << std::endl;
    kernel::init_blk_vio<<<BLOCKS(data.num_guides, 256), 256>>>(data, d);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e)
                  << std::endl;
        exit(0);
    }
    // std::cout << "init blk vio" << std::endl;
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
}
} // namespace gta::ops::cuda