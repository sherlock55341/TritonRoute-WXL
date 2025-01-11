#include "Apply.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace gta::ops {
void apply(data::Data &data, int iter, int i, int coef, std::set<int> *S) {
    assert(data.device == data::Device::CPU);
    if (data.ir_track[i] == -1)
        return;
    auto p = data.ir_panel[i];
    auto l = data.ir_layer[i];
    auto w = data.layer_width[l];
    auto begin = data.ir_begin[i] - w / 2;
    auto end = data.ir_end[i] + w / 2;
    if (w < data.layer_eol_width[l]) {
        begin -= data.layer_eol_spacing[l];
        end += data.layer_eol_spacing[l];
    }
    auto gb = std::max(0, data.ir_gcell_begin[i] - 1);
    auto ge =
        std::min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
    // wire - wire violation
    {
        for (auto g = gb; g <= ge; g++) {
            auto idx =
                data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
            for (auto list_idx = data.gcell_end_point_ir_start[idx];
                 list_idx < data.gcell_end_point_ir_start[idx + 1];
                 list_idx++) {
                auto j = data.gcell_end_point_ir_list[list_idx];
                if (i == j)
                    continue;
                // if (data.ir_track_low[i] != data.ir_track_low[j] ||
                //     data.ir_track_high[i] != data.ir_track_high[j]) {
                //     std::cout << "NOT MATCH " << __FILE__ << ":" << __LINE__
                //               << std::endl;
                //     exit(0);
                // }
                if (data.ir_gcell_begin[j] == g ||
                    (data.ir_gcell_end[j] == g &&
                     data.ir_gcell_begin[j] < gb)) {
                    if (data.ir_net[i] == data.ir_net[j]) {
                        data.ir_align_list[data.ir_vio_cost_start[j] +
                                           data.ir_track[i] -
                                           data.ir_track_low[i]] += coef;
                    } else {
                        auto overlap = std::max(
                            0, std::min(end, data.ir_end[j] + w / 2) -
                                   std::max(begin, data.ir_begin[j] - w / 2));
                        if (overlap > 0)
                            overlap = std::max(overlap, data.layer_pitch[l]);
                        data.ir_vio_cost_list[data.ir_vio_cost_start[j] +
                                              data.ir_track[i] -
                                              data.ir_track_low[i]] +=
                            overlap * coef;
                    }
                    if (S && data.ir_track[i] == data.ir_track[j] &&
                        data.ir_reassign[j] < 1)
                        S->insert(j);
                }
            }
        }
        for (auto list_idx = data.ir_super_set_start[i];
             list_idx < data.ir_super_set_start[i + 1]; list_idx++) {
            auto j = data.ir_super_set_list[list_idx];
            if (i == j)
                continue;
            // if (data.ir_track_low[i] != data.ir_track_low[j] ||
            //     data.ir_track_high[i] != data.ir_track_high[j]) {
            //     std::cout << "NOT MATCH " << __FILE__ << ":" << __LINE__
            //               << std::endl;
            //     exit(0);
            // }
            if (data.ir_net[i] == data.ir_net[j]) {
                data.ir_align_list[data.ir_vio_cost_start[j] +
                                   data.ir_track[i] - data.ir_track_low[i]] +=
                    coef;
            } else {
                auto overlap =
                    std::max(0, std::min(end, data.ir_end[j] + w / 2) -
                                    std::max(begin, data.ir_begin[j] - w / 2));
                if (overlap > 0)
                    overlap = std::max(overlap, data.layer_pitch[l]);
                data.ir_vio_cost_list[data.ir_vio_cost_start[j] +
                                      data.ir_track[i] -
                                      data.ir_track_low[i]] += overlap * coef;
            }
            if (S && data.ir_track[i] == data.ir_track[j] &&
                data.ir_reassign[j] < 1)
                S->insert(j);
        }
    }
    if (iter == 0)
        return;
    // via - wire violation
    if (data.layer_enable_via_wire_drc[l]) {
        auto pitch = data.layer_pitch[l];
        if (data.ir_gcell_begin_via_offset[i] > 0) {
            int p_delta_low = 0;
            int p_delta_high = 0;
            if (data.ir_track[i] - data.ir_gcell_begin_via_offset[i] <
                data.ir_track_low[i])
                p_delta_low = -1;
            if (data.ir_track[i] + data.ir_gcell_begin_via_offset[i] >
                data.ir_track_high[i])
                p_delta_high = 1;
            for (auto p_delta = p_delta_low; p_delta <= p_delta_high;
                 p_delta++) {
                auto p = data.ir_panel[i] + p_delta;
                if (p < 0 || p >= data.layer_panel_start[l + 1] -
                                      data.layer_panel_start[l])
                    continue;
                auto g = data.ir_gcell_begin[i];
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto cross_idx = data.gcell_cross_ir_start[idx];
                     cross_idx < data.gcell_cross_ir_start[idx + 1];
                     cross_idx++) {
                    auto j = data.gcell_cross_ir_list[cross_idx];
                    if (i == j)
                        continue;
                    if (data.ir_net[i] == data.ir_net[j])
                        continue;
                    for (auto delta = -data.ir_gcell_begin_via_offset[i];
                         delta <= data.ir_gcell_begin_via_offset[i]; delta++) {
                        if (delta == 0)
                            continue;
                        auto t = data.ir_track[i] + delta;
                        if (t < data.ir_track_low[j] ||
                            t > data.ir_track_high[j])
                            continue;
                        data.ir_vio_cost_list[data.ir_vio_cost_start[j] + t -
                                              data.ir_track_low[j]] +=
                            pitch * coef;
                        if (S && data.ir_track[j] == t &&
                            data.ir_reassign[j] < 1)
                            S->insert(j);
                    }
                }
            }
        }
        if (data.ir_gcell_begin[i] != data.ir_gcell_end[i] &&
            data.ir_gcell_end_via_offset[i] > 0) {
            int p_delta_low = 0;
            int p_delta_high = 0;
            if (data.ir_track[i] - data.ir_gcell_end_via_offset[i] <
                data.ir_track_low[i])
                p_delta_low = -1;
            if (data.ir_track[i] + data.ir_gcell_end_via_offset[i] >
                data.ir_track_high[i])
                p_delta_high = 1;
            for (auto p_delta = p_delta_low; p_delta <= p_delta_high;
                 p_delta++) {
                auto p = data.ir_panel[i] + p_delta;
                if (p < 0 || p >= data.layer_panel_start[l + 1] -
                                      data.layer_panel_start[l])
                    continue;
                auto g = data.ir_gcell_end[i];
                auto idx = data.layer_gcell_start[l] +
                           data.layer_panel_length[l] * p + g;
                for (auto cross_idx = data.gcell_cross_ir_start[idx];
                     cross_idx < data.gcell_cross_ir_start[idx + 1];
                     cross_idx++) {
                    auto j = data.gcell_cross_ir_list[cross_idx];
                    if (i == j)
                        continue;
                    if (data.ir_net[i] == data.ir_net[j])
                        continue;
                    if (data.ir_gcell_begin[j] < data.ir_gcell_begin[i])
                        continue;
                    for (auto delta = -data.ir_gcell_begin_via_offset[i];
                         delta <= data.ir_gcell_begin_via_offset[i]; delta++) {
                        if (delta == 0)
                            continue;
                        auto t = data.ir_track[i] + delta;
                        if (t < data.ir_track_low[j] ||
                            t > data.ir_track_high[j])
                            continue;
                        data.ir_vio_cost_list[data.ir_vio_cost_start[j] + t -
                                              data.ir_track_low[j]] +=
                            pitch * coef;
                        if (S && data.ir_track[j] == t &&
                            data.ir_reassign[j] < 1)
                            S->insert(j);
                    }
                }
            }
        }
    }
    // via - via violation
    if (data.layer_enable_via_via_drc[l]) {
        auto is_v = data.layer_direction[l];
        int lower_width = -1;
        int lower_length = -1;
        int upper_width = -1;
        int upper_length = -1;
        int gcell_start = (is_v ? data.gcell_start_y : data.gcell_start_x);
        int gcell_step = (is_v ? data.gcell_step_y : data.gcell_step_x);
        int gcell_num = (is_v ? data.num_gcells_y : data.num_gcells_x);
        if (l - 1 >= 0 && data.layer_type[l - 1] == 1) {
            lower_width = (is_v ? data.layer_via_span_x[l - 1]
                                : data.layer_via_span_y[l - 1]);
            lower_length = (is_v ? data.layer_via_span_y[l - 1]
                                 : data.layer_via_span_x[l - 1]);
        }
        if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1) {
            upper_width = (is_v ? data.layer_via_span_x[l + 1]
                                : data.layer_via_span_y[l + 1]);
            upper_length = (is_v ? data.layer_via_span_y[l + 1]
                                 : data.layer_via_span_x[l + 1]);
        }
        int lower_max_offset = -1;
        int upper_max_offset = -1;
        if (l - 1 >= 0 && data.layer_type[l - 1] == 1)
            lower_max_offset =
                std::ceil(1.0 * (lower_width + data.layer_cut_spacing[l - 1]) /
                          data.layer_track_step[l]);
        if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1)
            upper_max_offset =
                std::ceil(1.0 * (upper_width + data.layer_cut_spacing[l + 1]) /
                          data.layer_track_step[l]);
        int panel_delta_upper = 0;
        int panel_delta_lower = 0;
        if (data.ir_track[i] + std::max(lower_max_offset, upper_max_offset) >
            data.ir_track_high[i])
            panel_delta_upper = 1;
        if (data.ir_track[i] - std::max(lower_max_offset, upper_max_offset) <
            data.ir_track_low[i])
            panel_delta_lower = -1;
        for (int p_delta = panel_delta_lower; p_delta <= panel_delta_upper;
             p_delta++) {
            auto p = data.ir_panel[i] + p_delta;
            if (p < 0 ||
                p >= data.layer_panel_start[l + 1] - data.layer_panel_start[l])
                continue;
            // lower via
            if (l - 1 >= 0 && data.layer_type[l - 1] == 1) {
                int cut_spacing = data.layer_cut_spacing[l - 1];
                for (int via_i = data.ir_lower_via_start[i];
                     via_i < data.ir_lower_via_start[i + 1]; via_i++) {
                    auto via_i_coor = data.ir_lower_via_coor[via_i];
                    auto gb = (via_i_coor - lower_length / 2 - cut_spacing -
                               gcell_start) /
                              gcell_step;
                    auto ge = (via_i_coor + lower_length / 2 + cut_spacing -
                               gcell_start) /
                              gcell_step;
                    for (auto g = gb; g <= ge; g++) {
                        auto idx = data.layer_gcell_start[l] +
                                   p * data.layer_panel_length[l] + g;
                        for (auto list_idx = data.gcell_end_point_ir_start[idx];
                             list_idx < data.gcell_end_point_ir_start[idx + 1];
                             list_idx++) {
                            auto j = data.gcell_end_point_ir_list[list_idx];
                            if (data.ir_gcell_begin[j] >= gb &&
                                data.ir_gcell_begin[j] < g)
                                continue;
                            if (data.ir_net[i] == data.ir_net[j])
                                continue;
                            bool need_insert = false;
                            for (int offset = -lower_max_offset;
                                 offset <= lower_max_offset; offset++) {
                                auto track_j = data.ir_track[i] + offset;
                                if (track_j < data.ir_track_low[j] ||
                                    track_j > data.ir_track_high[j])
                                    continue;
                                int gap = std::max(
                                    0, std::abs(offset) *
                                               data.layer_track_step[l] -
                                           lower_width);
                                if (gap >= cut_spacing)
                                    continue;
                                int extension = std::floor(
                                    std::sqrt(1.0 * cut_spacing * cut_spacing -
                                              gap * gap));
                                int via_begin =
                                    via_i_coor - lower_length / 2 - extension;
                                int via_end =
                                    via_i_coor + lower_length / 2 + extension;
                                for (auto via_j = data.ir_lower_via_start[j];
                                     via_j < data.ir_lower_via_start[j + 1];
                                     via_j++) {
                                    auto via_j_coor =
                                        data.ir_lower_via_coor[via_j];
                                    auto overlap = std::max(
                                        0, std::min(via_end,
                                                    via_j_coor +
                                                        lower_length / 2) -
                                               std::max(via_begin,
                                                        via_j_coor -
                                                            lower_length / 2));
                                    if (overlap > 0) {
                                        // data.ir_vio_cost_list
                                        //     [data.ir_vio_cost_start[j] +
                                        //      track_j - data.ir_track_low[j]]
                                        //      +=
                                        //     std::min(overlap,
                                        //              data.layer_pitch[l] / 2)
                                        //              *
                                        //     coef;
                                        data.ir_via_vio_list
                                            [data.ir_vio_cost_start[j] +
                                             track_j - data.ir_track_low[j]] +=
                                            coef * overlap;
                                        if (data.ir_track[j] == track_j)
                                            need_insert = true;
                                    }
                                }
                            }
                            if (S && need_insert && data.ir_reassign[j] < 1)
                                S->insert(j);
                        }
                    }
                }
            }
            // upper via
            if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1) {
                int cut_spacing = data.layer_cut_spacing[l + 1];
                for (int via_i = data.ir_upper_via_start[i];
                     via_i < data.ir_upper_via_start[i + 1]; via_i++) {
                    auto via_i_coor = data.ir_upper_via_coor[via_i];
                    auto gb = (via_i_coor - upper_length / 2 - cut_spacing -
                               gcell_start) /
                              gcell_step;
                    auto ge = (via_i_coor + upper_length / 2 + cut_spacing -
                               gcell_start) /
                              gcell_step;
                    for (auto g = gb; g <= ge; g++) {
                        auto idx = data.layer_gcell_start[l] +
                                   p * data.layer_panel_length[l] + g;
                        for (auto list_idx = data.gcell_end_point_ir_start[idx];
                             list_idx < data.gcell_end_point_ir_start[idx + 1];
                             list_idx++) {
                            auto j = data.gcell_end_point_ir_list[list_idx];
                            if (data.ir_gcell_begin[j] >= gb &&
                                data.ir_gcell_begin[j] < g)
                                continue;
                            if (data.ir_net[i] == data.ir_net[j])
                                continue;
                            bool need_insert = false;
                            for (int offset = -upper_max_offset;
                                 offset <= upper_max_offset; offset++) {
                                auto track_j = data.ir_track[i] + offset;
                                if (track_j < data.ir_track_low[j] ||
                                    track_j > data.ir_track_high[j])
                                    continue;
                                int gap = std::max(
                                    0, std::abs(offset) *
                                               data.layer_track_step[l] -
                                           upper_width);
                                if (gap >= cut_spacing)
                                    continue;
                                int extension = std::floor(
                                    std::sqrt(1.0 * cut_spacing * cut_spacing -
                                              gap * gap));
                                int via_begin =
                                    via_i_coor - upper_length / 2 - extension;
                                int via_end =
                                    via_i_coor + upper_length / 2 + extension;
                                for (auto via_j = data.ir_upper_via_start[j];
                                     via_j < data.ir_upper_via_start[j + 1];
                                     via_j++) {
                                    auto via_j_coor =
                                        data.ir_upper_via_coor[via_j];
                                    auto overlap = std::max(
                                        0, std::min(via_end,
                                                    via_j_coor +
                                                        upper_length / 2) -
                                               std::max(via_begin,
                                                        via_j_coor -
                                                            upper_length / 2));
                                    if (overlap > 0) {
                                        // data.ir_vio_cost_list
                                        //     [data.ir_vio_cost_start[j] +
                                        //      track_j - data.ir_track_low[j]]
                                        //      +=
                                        //     std::min(overlap,
                                        //              data.layer_pitch[l] / 2)
                                        //              *
                                        //     coef;
                                        data.ir_via_vio_list
                                            [data.ir_vio_cost_start[j] +
                                             track_j - data.ir_track_low[j]] +=
                                            coef * overlap;
                                        if (data.ir_track[j] == track_j)
                                            need_insert = true;
                                    }
                                }
                            }
                            if (S && need_insert && data.ir_reassign[j] < 1)
                                S->insert(j);
                        }
                    }
                }
            }
        }
    }
}
} // namespace gta::ops