#include "AssignKernel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <gta/ops/apply/Apply.hpp>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

namespace gta::ops::cpu::helper {
void assign(data::Data &data, int iter, int i, std::set<int> *S = nullptr) {
    float bestCost = std::numeric_limits<float>::max();
    int bestChoice = -1;
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
            auto dist = std::abs(coor - data.ir_ap[i]);
            pin_connect_cost =
                data.layer_wire_weight[l] * dist +
                ((dist > data.layer_track_step[l] / 2) ? 4 * data.layer_pitch[l]
                                                       : 0);
        }
        assert(data.ir_align_list[t] >= 0);
        int align_cost = data.ir_align_list[t] * 4 * data.layer_pitch[l];
        auto via_vio_cost = data.layer_pitch[data.ir_layer[i]] *
                            std::min(data.ir_via_vio_list[t], 4);
        auto local_cost = drc * (data.ir_vio_cost_list[t] + via_vio_cost) +
                          wire_length_cost + pin_connect_cost - align_cost;
        if (local_cost < bestCost) {
            bestCost = local_cost;
            bestChoice = t - data.ir_vio_cost_start[i];
        }
    }
    assert(bestChoice != -1);
    apply(data, iter, i, -1, S);
    data.ir_track[i] = bestChoice + data.ir_track_low[i];
    apply(data, iter, i, 1, S);
}
} // namespace gta::ops::cpu::helper

namespace gta::ops::cpu {
void assign(data::Data &data, int iter, int d) {
    int counter = 0;
    int panel_num = (d ? data.num_gcells_x : data.num_gcells_y);
    // constexpr int group_panel_width = 20;
    constexpr int group_panel_width = 1;
    auto group_num = (panel_num + group_panel_width - 1) / group_panel_width;
    std::vector<std::vector<int>> ir_groups(group_num);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v == d)
            ir_groups[data.ir_panel[i] / group_panel_width].push_back(i);
    }
    for (auto dir = 0; dir < 2; dir++) {
#pragma omp parallel for
        for (auto i = 0; i < ir_groups.size(); i++) {
            if (i % 2 == dir)
                continue;
            int local_counter = 0;
            auto &ir_group = ir_groups[i];
            if (iter == 0) { // initial
                std::sort(ir_group.begin(), ir_group.end(),
                          [&](int lhs, int rhs) {
                              if (data.ir_has_proj_ap[lhs] !=
                                  data.ir_has_proj_ap[rhs])
                                  return data.ir_has_proj_ap[lhs] >
                                         data.ir_has_proj_ap[rhs];
                              auto coef_lhs =
                                  1.0 * std::abs(data.ir_wl_weight[lhs] /
                                                 (data.ir_end[lhs] -
                                                  data.ir_begin[lhs] + 1));
                              auto coef_rhs =
                                  1.0 * std::abs(data.ir_wl_weight[rhs] /
                                                 (data.ir_end[rhs] -
                                                  data.ir_begin[rhs] + 1));
                              return coef_lhs > coef_rhs;
                          });
                for (auto ir : ir_group) {
                    local_counter++;
                    helper::assign(data, iter, ir);
                }
            } else { // refinement
                std::function<bool(int, int)> cmp = [&](int lhs, int rhs) {
                    if (data.ir_key_cost[lhs] != data.ir_key_cost[rhs])
                        return data.ir_key_cost[lhs] > data.ir_key_cost[rhs];
                    auto l_lhs = data.ir_end[lhs] - data.ir_begin[lhs];
                    auto l_rhs = data.ir_end[rhs] - data.ir_begin[rhs];
                    if (l_lhs != l_rhs)
                        return l_lhs < l_rhs;
                    return lhs < rhs;
                };
                std::set<int, std::function<bool(int, int)>> S(cmp);
                auto collect_set = std::make_unique<std::set<int>>();
                for (auto ir : ir_group) {
                    auto idx = data.ir_vio_cost_start[ir] +
                               (data.ir_track[ir] - data.ir_track_low[ir]);
                    data.ir_key_cost[ir] =
                        data.ir_vio_cost_list[idx] + data.ir_via_vio_list[idx];
                    if (data.ir_key_cost[ir] > 0)
                        S.insert(ir);
                    data.ir_reassign[ir] = 0;
                }
                while (S.empty() == false) {
                    auto ir = *S.begin();
                    local_counter++;
                    data.ir_reassign[ir]++;
                    S.erase(ir);
                    collect_set->clear();
                    collect_set->insert(ir);
                    helper::assign(data, iter, ir, collect_set.get());
                    for (auto it : *collect_set) {
                        if (S.find(it) != S.end())
                            S.erase(it);
                        auto idx = data.ir_vio_cost_start[it] +
                                   (data.ir_track[it] - data.ir_track_low[it]);
                        data.ir_key_cost[it] = data.ir_vio_cost_list[idx] +
                                               data.ir_via_vio_list[idx];
                        if (data.ir_reassign[it] < 1 &&
                            data.ir_key_cost[it] > 0)
                            S.insert(it);
                    }
                    collect_set->clear();
                }
            }
#pragma omp critical
            { counter += local_counter; }
        }
#pragma omp barrier
    }
    std::cout << "assign : " << counter << " iroutes" << std::endl;
}
} // namespace gta::ops::cpu