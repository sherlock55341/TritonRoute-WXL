#include "InitKernel.hpp"
#include <gta/ops/apply/Apply.hpp>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <cmath>

namespace gta::ops::cpu::helper {
void getBlkVio(data::Data &data, int i, int j) {
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
    auto prl = std::max(0, (is_v ? (std::min(data.b_top[j], end) -
                                    std::max(data.b_bottom[j], begin))
                                 : (std::min(data.b_right[j], end) -
                                    std::max(data.b_left[j], begin))));
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
        s = data::helper::findPRLSpacing(data, l, std::max(width, data.layer_width[l]), prl);
        vio_begin -= s;
        vio_end += s;
    }
    int vio_track_begin =
        std::ceil(1.0 * (vio_begin - data.layer_track_start[l]) /
                  data.layer_track_step[l]);
    int vio_track_end = std::floor(1.0 * (vio_end - data.layer_track_start[l]) /
                                   data.layer_track_step[l]);
    vio_track_begin = std::max(vio_track_begin, data.ir_track_low[i]);
    vio_track_end = std::min(vio_track_end, data.ir_track_high[i]);
    for (auto t = vio_track_begin; t <= vio_track_end; t++) {
        auto coor = data.layer_track_start[l] + data.layer_track_step[l] * t;
        int extension = 0;
        if (data.layer_width[l] <
            data.layer_eol_width[l]) { // consider eol spacing constraint
            auto upperEdge =
                coor + data.layer_width[l] / 2 + data.layer_eol_within[l];
            auto lowerEdge =
                coor - data.layer_width[l] / 2 - data.layer_eol_within[l];
            auto overlap = std::max(
                0,
                std::min(upperEdge, (is_v ? data.b_right[j] : data.b_top[j])) -
                    std::max(lowerEdge,
                             (is_v ? data.b_left[j] : data.b_bottom[j])));
            if (overlap > 0)
                extension = data.layer_eol_spacing[l];
        }
        if (data.layer_enable_corner_spacing[l] == true)
            extension = std::max(extension, s);
        auto overlap = std::max(
            0,
            std::min((is_v ? data.b_top[j] : data.b_right[j]) + extension,
                     end) -
                std::max((is_v ? data.b_bottom[j] : data.b_left[j]) - extension,
                         begin));
        if (overlap > 0)
            overlap = std::max(overlap, data.layer_pitch[l]);
        data.ir_vio_cost_list[data.ir_vio_cost_start[i] + t -
                              data.ir_track_low[i]] += overlap;
    }
}
} // namespace gta::ops::cpu::helper

namespace gta::ops::cpu {
void init(data::Data &data, int iter, int d) {
#pragma omp parallel for
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d)
            continue;
        if (data.ir_has_ap[i] == false) {
            bool hasBegin = false;
            bool hasEnd = false;
            int begin = std::numeric_limits<int>::max();
            int end = std::numeric_limits<int>::min();
            for (auto idx = data.ir_nbr_start[i];
                 idx < data.ir_nbr_start[i + 1]; idx++) {
                auto j = data.ir_nbr_list[idx];
                if (data.ir_track[j] != -1) {
                    assert(data.ir_track[j] >= data.ir_track_low[j]);
                    assert(data.ir_track[j] <= data.ir_track_high[j]);
                    auto nbr_layer = data.ir_layer[j];
                    auto coor =
                        data.layer_track_start[nbr_layer] +
                        data.layer_track_step[nbr_layer] * data.ir_track[j];
                    auto via_layer = (nbr_layer + l) / 2;
                    if (data.ir_panel[j] == data.ir_gcell_begin[i]) {
                        hasBegin = true;
                        begin = std::min(begin, coor);
                    }
                    if (data.ir_panel[j] == data.ir_gcell_end[i]) {
                        hasEnd = true;
                        end = std::max(end, coor);
                    }
                }
            }
            if (hasBegin == false) {
                if (is_v)
                    begin = data.gcell_start_y +
                            data.gcell_step_y *
                                (2 * data.ir_gcell_begin[i] + 1) / 2;
                else
                    begin = data.gcell_start_x +
                            data.gcell_step_x *
                                (2 * data.ir_gcell_begin[i] + 1) / 2;
            }
            if (hasEnd == false) {
                if (is_v)
                    end =
                        data.gcell_start_y +
                        data.gcell_step_y * (2 * data.ir_gcell_end[i] + 1) / 2;
                else
                    end =
                        data.gcell_start_x +
                        data.gcell_step_x * (2 * data.ir_gcell_end[i] + 1) / 2;
            }
            if (begin > end)
                std::swap(begin, end);
            if (begin == end)
                end++;
            data.ir_begin[i] = begin;
            data.ir_end[i] = end;
        }
    }
#pragma omp barrier
#pragma omp parallel for
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d)
            continue;
        for (auto vio_i = data.ir_vio_cost_start[i];
             vio_i < data.ir_vio_cost_start[i + 1]; vio_i++) {
            data.ir_vio_cost_list[vio_i] = 0;
            data.ir_align_list[vio_i] = 0;
            data.ir_via_vio_list[vio_i] = 0;
        }
        auto p = data.ir_panel[i];
        auto gb = std::max(0, data.ir_gcell_begin[i] - 1);
        auto ge =
            std::min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
        for (auto g = gb; g <= ge; g++) {
            auto idx =
                data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
            for (auto list_idx = data.gcell_end_point_blk_start[idx];
                 list_idx < data.gcell_end_point_blk_start[idx + 1];
                 list_idx++) {
                auto j = data.gcell_end_point_blk_list[list_idx];
                if (data.b_gcell_begin[j] == g ||
                    (data.b_gcell_end[j] == g && data.b_gcell_begin[j] < gb)) {
                    helper::getBlkVio(data, i, j);
                }
            }
        }
        for (auto idx = data.blk_super_set_start[i];
             idx < data.blk_super_set_start[i + 1]; idx++) {
            auto j = data.blk_super_set_list[idx];
            helper::getBlkVio(data, i, j);
        }
    }
#pragma omp barrier
    // if (iter > 0) {
    memset(data.ir_lower_via_start, 0, sizeof(int) * (data.num_guides + 1));
    memset(data.ir_upper_via_start, 0, sizeof(int) * (data.num_guides + 1));
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d)
            continue;
        for (auto nbr_idx = data.ir_nbr_start[i];
             nbr_idx < data.ir_nbr_start[i + 1]; nbr_idx++) {
            auto j = data.ir_nbr_list[nbr_idx];
            bool duplicate = false;
            for (auto nbr_idx_2 = data.ir_nbr_start[i]; nbr_idx_2 < nbr_idx;
                 nbr_idx_2++) {
                auto k = data.ir_nbr_list[nbr_idx_2];
                auto on_same_layer = (data.ir_layer[j] == data.ir_layer[k]);
                auto on_same_track = (data.ir_track[j] == data.ir_track[k]);
                if (on_same_layer && on_same_track) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate == false) {
                if (data.ir_layer[j] < l)
                    data.ir_lower_via_start[i + 1]++;
                else
                    data.ir_upper_via_start[i + 1]++;
            }
        }
    }
    for (auto i = 0; i < data.num_guides; i++) {
        data.ir_lower_via_start[i + 1] += data.ir_lower_via_start[i];
        data.ir_upper_via_start[i + 1] += data.ir_upper_via_start[i];
    }
    data.ir_lower_via_coor =
        (int *)realloc(data.ir_lower_via_coor,
                       sizeof(int) * data.ir_lower_via_start[data.num_guides]);
    data.ir_upper_via_coor =
        (int *)realloc(data.ir_upper_via_coor,
                       sizeof(int) * data.ir_upper_via_start[data.num_guides]);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d)
            continue;
        int lower_offset = 0;
        int upper_offset = 0;
        for (auto nbr_idx = data.ir_nbr_start[i];
             nbr_idx < data.ir_nbr_start[i + 1]; nbr_idx++) {
            auto j = data.ir_nbr_list[nbr_idx];
            bool duplicate = false;
            for (auto nbr_idx_2 = data.ir_nbr_start[i]; nbr_idx_2 < nbr_idx;
                 nbr_idx_2++) {
                auto k = data.ir_nbr_list[nbr_idx_2];
                auto on_same_layer = (data.ir_layer[j] == data.ir_layer[k]);
                auto on_same_track = (data.ir_track[j] == data.ir_track[k]);
                if (on_same_layer && on_same_track) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate == false) {
                auto nbr_layer = data.ir_layer[j];
                auto nbr_coor =
                    data.layer_track_start[nbr_layer] +
                    data.layer_track_step[nbr_layer] * data.ir_track[j];
                if (data.ir_track[j] == -1)
                    nbr_coor = (is_v ? data.gcell_start_y +
                                           data.gcell_step_y *
                                               (2 * data.ir_panel[j] + 1) / 2
                                     : data.gcell_start_x +
                                           data.gcell_step_x *
                                               (2 * data.ir_panel[j] + 1) / 2);
                if (data.ir_layer[j] < l) {
                    data.ir_lower_via_coor[data.ir_lower_via_start[i] +
                                           lower_offset] = nbr_coor;
                    lower_offset++;
                } else {
                    data.ir_upper_via_coor[data.ir_upper_via_start[i] +
                                           upper_offset] = nbr_coor;
                    upper_offset++;
                }
            }
        }
        if (lower_offset !=
            data.ir_lower_via_start[i + 1] - data.ir_lower_via_start[i]) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << lower_offset
                      << " "
                      << data.ir_lower_via_start[i + 1] -
                             data.ir_lower_via_start[i]
                      << std::endl;
            exit(0);
        }
        if (upper_offset !=
            data.ir_upper_via_start[i + 1] - data.ir_upper_via_start[i]) {
            std::cout << __FILE__ << ":" << __LINE__ << " " << upper_offset
                      << " "
                      << data.ir_upper_via_start[i + 1] -
                             data.ir_upper_via_start[i]
                      << std::endl;
            exit(0);
        }
    }

    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d)
            continue;
        apply(data, iter, i, 1);
    }
    // }
}
} // namespace gta::ops::cpu