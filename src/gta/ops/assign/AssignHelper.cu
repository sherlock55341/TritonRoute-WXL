#include "AssignHelper.cuh"
#include <cassert>

namespace gta::ops::cuda::device {
__device__ void assign(data::Data &data, int iter, int i) {
    float best_cost = 1e9;
    int best_choice = -1;
    auto l = data.ir_layer[i];
    const float drc = ((iter == 0) ? 0.05f : 32.0f);
    for (auto t = data.ir_vio_cost_start[i]; t < data.ir_vio_cost_start[i + 1];
         t++) {
        auto track_idx = data.ir_track_low[i] + (t - data.ir_vio_cost_start[i]);
        int wire_length_cost =
            data.ir_wl_weight[i] * (data.ir_track_high[i] - track_idx);
        int pin_connect_cost = 0;
        if (data.ir_has_proj_ap[i] == true) {
            auto coor = data.layer_track_start[l] +
                        data.layer_track_step[l] * track_idx;
            int dist = abs(coor - data.ir_ap[i]);
            pin_connect_cost =
                data.layer_wire_weight[l] * dist +
                ((dist > data.layer_track_step[l] / 2) ? 4 * data.layer_pitch[l]
                                                       : 0);
        }
        assert(data.ir_align_list[t] >= 0);
        int align_cost = data.ir_align_list[t] * 4 * data.layer_pitch[l];
        auto via_vio_cost = data.layer_pitch[data.ir_layer[i]] *
                            min(data.ir_via_vio_list[t], 4);
        auto local_cost = drc * (data.ir_vio_cost_list[t] + via_vio_cost) +
                          wire_length_cost + pin_connect_cost - align_cost;
        if (local_cost < best_cost) {
            best_cost = local_cost;
            best_choice = t - data.ir_vio_cost_start[i];
        }
    }
    assert(best_choice != -1);
    data::cuda::device::apply(data, iter, i, -1);
    data.ir_track[i] = best_choice + data.ir_track_low[i];
    data::cuda::device::apply(data, iter, i, 1);
}
} // namespace gta::ops::cuda::device