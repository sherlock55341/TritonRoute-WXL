#include "Apply.cuh"
#include <cassert>

namespace gta::data::cuda::device {
__device__ void apply(data::Data &data, int iter, int i, int coef) {
    assert(data.device == data::Device::CUDA);
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
    auto gb = max(0, data.ir_gcell_begin[i] - 1);
    auto ge = min(data.layer_panel_length[l] - 1, data.ir_gcell_end[i] + 1);
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
                if (data.ir_gcell_begin[j] == g ||
                    (data.ir_gcell_end[j] == g &&
                     data.ir_gcell_begin[j] < gb)) {
                    if (data.ir_net[i] == data.ir_net[j]) {
                        // data.ir_align_list[data.ir_vio_cost_start[j] +
                        //                    data.ir_track[i] -
                        //                    data.ir_track_low[i]] += coef;
                        atomicAdd(data.ir_align_list +
                                      (data.ir_vio_cost_start[j] +
                                       data.ir_track[i] - data.ir_track_low[i]),
                                  coef);
                    } else {
                        auto overlap =
                            max(0, min(end, data.ir_end[j] + w / 2) -
                                       max(begin, data.ir_begin[j] - w / 2));
                        if (overlap > 0) {
                            overlap = max(overlap, data.layer_pitch[l]);
                            atomicAdd(data.ir_vio_cost_list +
                                          (data.ir_vio_cost_start[j] +
                                           data.ir_track[i] -
                                           data.ir_track_low[i]),
                                      overlap * coef);
                        }
                        // data.ir_vio_cost_list[data.ir_vio_cost_start[j] +
                        //                       data.ir_track[i] -
                        //                       data.ir_track_low[i]] +=
                        //     overlap * coef;
                    }
                }
            }
        }
        for (auto list_idx = data.ir_super_set_start[i];
             list_idx < data.ir_super_set_start[i + 1]; list_idx++) {
            auto j = data.ir_super_set_list[list_idx];
            if (i == j)
                continue;
            if (data.ir_net[i] == data.ir_net[j]) {
                // data.ir_align_list[data.ir_vio_cost_start[j] +
                //                    data.ir_track[i] - data.ir_track_low[i]]
                //                    +=
                //     coef;
                atomicAdd(data.ir_align_list +
                              (data.ir_vio_cost_start[j] + data.ir_track[i] -
                               data.ir_track_low[i]),
                          -coef);
            } else {
                auto overlap = max(0, min(end, data.ir_end[j] + w / 2) -
                                          max(begin, data.ir_begin[j] - w / 2));
                if (overlap > 0) {
                    overlap = max(overlap, data.layer_pitch[l]);
                    atomicAdd(data.ir_vio_cost_list +
                                  (data.ir_vio_cost_start[j] +
                                   data.ir_track[i] - data.ir_track_low[i]),
                              overlap * coef);
                }
                // data.ir_vio_cost_list[data.ir_vio_cost_start[j] +
                //                       data.ir_track[i] -
                //                       data.ir_track_low[i]] += overlap *
                //                       coef;
            }
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
                        atomicAdd(data.ir_vio_cost_list +
                                      (data.ir_vio_cost_start[j] + t -
                                       data.ir_track_low[j]),
                                  pitch * coef);
                        // data.ir_vio_cost_list[data.ir_vio_cost_start[j] + t -
                        //                       data.ir_track_low[j]] +=
                        //     pitch * coef;
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
                        atomicAdd(data.ir_vio_cost_list +
                                      (data.ir_vio_cost_start[j] + t -
                                       data.ir_track_low[j]),
                                  pitch * coef);
                        // data.ir_vio_cost_list[data.ir_vio_cost_start[j] + t -
                        //                       data.ir_track_low[j]] +=
                        //     pitch * coef;
                    }
                }
            }
        }
    }
    // via - via violation
    // if (data.layer_enable_via_via_drc[l]) {
    //     auto is_v = data.layer_direction[l];
    //     int lower_width = -1;
    //     int lower_length = -1;
    //     int upper_width = -1;
    //     int upper_length = -1;
    //     int gcell_start = (is_v ? data.gcell_start_y : data.gcell_start_x);
    //     int gcell_step = (is_v ? data.gcell_step_y : data.gcell_step_x);
    //     int gcell_num = (is_v ? data.num_gcells_y : data.num_gcells_x);
    //     if (l - 1 >= 0 && data.layer_type[l - 1] == 1) {
    //         lower_width = (is_v ? data.layer_via_span_x[l - 1]
    //                             : data.layer_via_span_y[l - 1]);
    //         lower_length = (is_v ? data.layer_via_span_y[l - 1]
    //                              : data.layer_via_span_x[l - 1]);
    //     }
    //     if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1) {
    //         upper_width = (is_v ? data.layer_via_span_x[l + 1]
    //                             : data.layer_via_span_y[l + 1]);
    //         upper_length = (is_v ? data.layer_via_span_y[l + 1]
    //                              : data.layer_via_span_x[l + 1]);
    //     }
    //     int max_offset = 0;
    //     if (l - 1 >= 0 && data.layer_type[l - 1] == 1)
    //         max_offset =
    //             max(max_offset,
    //                 static_cast<int>(floor(
    //                     1.0 * (lower_width + data.layer_cut_spacing[l - 1]) /
    //                     data.layer_track_step[l])) -
    //                     1);
    //     if (l + 1 < data.num_layers && data.layer_type[l + 1] == 1)
    //         max_offset =
    //             max(max_offset,
    //                 static_cast<int>(floor(
    //                     1.0 * (upper_width + data.layer_cut_spacing[l + 1]) /
    //                     data.layer_track_step[l])) -
    //                     1);
    //     int p_low = data.ir_panel[i];
    //     int p_high = data.ir_panel[i];
    //     if (data.ir_track[i] + max_offset > data.ir_track_high[i])
    //         p_high = min(data.layer_panel_start[l + 1] -
    //                          data.layer_panel_start[l] - 1,
    //                      p_high + 1);
    //     if (data.ir_track[i] - max_offset < data.ir_track_low[i])
    //         p_low = max(0, p_low - 1);
    // }
}
} // namespace gta::data::cuda::device