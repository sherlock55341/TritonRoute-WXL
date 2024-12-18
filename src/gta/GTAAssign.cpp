#include "GTA.hpp"

namespace gta {
void GTA::init(int iter, int d_0) {
#pragma omp parallel for
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v != d_0)
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
        if (is_v != d_0)
            continue;
        for (auto vio_i = data.ir_vio_cost_start[i];
             vio_i < data.ir_vio_cost_start[i + 1]; vio_i++) {
            data.ir_vio_cost_list[vio_i] = 0;
            data.ir_align_list[vio_i] = 0;
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
                    getBlkVio(i, j);
                }
            }
        }
        for (auto idx = data.blk_super_set_start[i];
             idx < data.blk_super_set_start[i + 1]; idx++) {
            auto j = data.blk_super_set_list[idx];
            getBlkVio(i, j);
        }
    }
#pragma omp barrier
}

void GTA::assignInitial(int d_0) {
    int counter = 0;
    int panel_num = (d_0 ? data.num_gcells_x : data.num_gcells_y);
    std::vector<std::vector<int>> ir_groups(panel_num);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v == d_0)
            ir_groups[data.ir_panel[i]].push_back(i);
    }
#pragma omp parallel for
    for (auto &ir_group : ir_groups) {
        int local_counter = 0;
        std::sort(ir_group.begin(), ir_group.end(), [&](int lhs, int rhs) {
            if (data.ir_has_proj_ap[lhs] != data.ir_has_proj_ap[rhs])
                return data.ir_has_proj_ap[lhs] > data.ir_has_proj_ap[rhs];
            auto coef_lhs =
                1.0 * std::abs(data.ir_wl_weight[lhs] /
                               (data.ir_end[lhs] - data.ir_begin[lhs] + 1));
            auto coef_rhs =
                1.0 * std::abs(data.ir_wl_weight[rhs] /
                               (data.ir_end[rhs] - data.ir_begin[rhs] + 1));
            return coef_lhs > coef_rhs;
        });
        for (auto ir : ir_group) {
            local_counter++;
            assign(0, ir);
        }
#pragma omp critical
        { counter += local_counter; }
    }
#pragma omp barrier
    std::cout << "assign : " << counter << " iroutes " << d_0 << " "
              << ir_groups.size() << std::endl;
}

void GTA::assignRefinement(int iter, int d_0) {
    int counter = 0;
    int panel_num = (d_0 ? data.num_gcells_x : data.num_gcells_y);
    constexpr int group_panel_width = 1;
    auto group_num = (panel_num + group_panel_width - 1) / group_panel_width;
    std::vector<std::vector<int>> ir_groups(group_num);
    for (auto i = 0; i < data.num_guides; i++) {
        auto l = data.ir_layer[i];
        auto is_v = data.layer_direction[l];
        if (is_v == d_0)
            ir_groups[data.ir_panel[i] / group_panel_width].push_back(i);
    }
    std::function<bool(int, int)> cmp = [&](int lhs, int rhs) {
        if (data.ir_key_cost[lhs] != data.ir_key_cost[rhs])
            return data.ir_key_cost[lhs] > data.ir_key_cost[rhs];
        auto l_lhs = data.ir_end[lhs] - data.ir_begin[lhs];
        auto l_rhs = data.ir_end[rhs] - data.ir_begin[rhs];
        if (l_lhs != l_rhs)
            return l_lhs < l_rhs;
        return lhs < rhs;
    };
    for (auto d = 0; d < 2; d++) {
#pragma omp parallel for
        for (auto i = 0; i < ir_groups.size(); i++) {
            if (i % 2 != d)
                continue;
            auto &ir_group = ir_groups[i];
            int local_counter = 0;
            std::set<int, std::function<bool(int, int)>> S(cmp);
            auto collect_set = std::make_unique<std::set<int>>();
            for (auto ir : ir_group)
                apply(ir, 1, ENABLE_TA_VIA_DRC);
            for (auto ir : ir_group) {
                data.ir_key_cost[ir] =
                    data.ir_vio_cost_list[data.ir_vio_cost_start[ir] +
                                          (data.ir_track[ir] -
                                           data.ir_track_low[ir])];
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
                assign(iter, ir, collect_set.get());
                for (auto it : *collect_set) {
                    if (S.find(it) != S.end())
                        S.erase(it);
                    data.ir_key_cost[it] =
                        data.ir_vio_cost_list[data.ir_vio_cost_start[it] +
                                              (data.ir_track[it] -
                                               data.ir_track_low[it])];
                    if (data.ir_reassign[it] < 1 && data.ir_key_cost[it] > 0)
                        S.insert(it);
                }
                collect_set->clear();
            }
#pragma omp critical
            { counter += local_counter; }
        }
#pragma omp barrier
    }
    std::cout << "assign : " << counter << " iroutes " << d_0 << " "
              << ir_groups.size() << std::endl;
}

void GTA::assign(int iter, int i, std::set<int> *S) {
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
        auto local_cost = drc * data.ir_vio_cost_list[t] + wire_length_cost +
                          pin_connect_cost - align_cost;
        if (local_cost < bestCost) {
            bestCost = local_cost;
            bestChoice = t - data.ir_vio_cost_start[i];
        }
    }
    assert(bestChoice != -1);
    apply(i, -1, iter > 0 ? ENABLE_TA_VIA_DRC : false, S);
    data.ir_track[i] = bestChoice + data.ir_track_low[i];
    apply(i, 1, iter > 0 ? ENABLE_TA_VIA_DRC : false, S);
}

void GTA::apply(int i, int coef, bool enable_via, std::set<int> *S) {
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
    for (auto g = gb; g <= ge; g++) {
        auto idx =
            data.layer_gcell_start[l] + p * data.layer_panel_length[l] + g;
        for (auto list_idx = data.gcell_end_point_ir_start[idx];
             list_idx < data.gcell_end_point_ir_start[idx + 1]; list_idx++) {
            auto j = data.gcell_end_point_ir_list[list_idx];
            if (i == j)
                continue;
            if (data.ir_gcell_begin[j] == g ||
                (data.ir_gcell_end[j] == g && data.ir_gcell_begin[j] < gb)) {
                if (data.ir_net[i] == data.ir_net[j]) {
                    data.ir_align_list[data.ir_vio_cost_start[j] +
                                       data.ir_track[i] -
                                       data.ir_track_low[i]] += coef;
                } else {
                    auto overlap = std::max(
                        0, std::min(end, data.ir_end[j] + w / 2) -
                               std::max(begin, data.ir_begin[j] - w / 2));
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
        if (data.ir_net[i] == data.ir_net[j]) {
            data.ir_align_list[data.ir_vio_cost_start[j] + data.ir_track[i] -
                               data.ir_track_low[i]] += coef;
        } else {
            auto overlap =
                std::max(0, std::min(end, data.ir_end[j] + w / 2) -
                                std::max(begin, data.ir_begin[j] - w / 2));
            data.ir_vio_cost_list[data.ir_vio_cost_start[j] + data.ir_track[i] -
                                  data.ir_track_low[i]] += overlap * coef;
        }
        if (S && data.ir_track[i] == data.ir_track[j] &&
            data.ir_reassign[j] < 1)
            S->insert(j);
    }
}

void GTA::getBlkVio(int i, int j) {
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
    auto vio_begin = (is_v ? data.b_left[j] : data.b_bottom[j]) -
                     data.layer_width[l] / 2 + 1;
    auto vio_end =
        (is_v ? data.b_right[j] : data.b_top[j]) + data.layer_width[l] / 2 - 1;
    if (d == is_v) { // consider prl spacing constraint
        auto s =
            findPRLSpacing(l,
                           std::max((is_v ? (data.b_right[j] - data.b_left[j])
                                          : (data.b_top[j] - data.b_bottom[j])),
                                    data.layer_width[l]),
                           prl);
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
        auto overlap = std::max(
            0,
            std::min((is_v ? data.b_top[j] : data.b_right[j]) + extension,
                     end) -
                std::max((is_v ? data.b_bottom[j] : data.b_left[j]) - extension,
                         begin));
        data.ir_vio_cost_list[data.ir_vio_cost_start[i] + t -
                              data.ir_track_low[i]] += overlap;
    }
    // consider via vio
    // if (enable_via) {
    //     for (auto via_idx = data.ir_via_start[i];
    //          via_idx < data.ir_via_start[i + 1]; via_idx++) {
    //         int via_width = data.ir_via_list_width[via_idx];
    //         int via_begin = data.ir_via_list_begin[via_idx];
    //         int via_end = data.ir_via_list_end[via_idx];
    //         prl = std::max(0, (is_v ? (std::min(data.b_top[j], via_end) -
    //                                    std::max(data.b_bottom[j], via_begin))
    //                                 : (std::min(data.b_right[j], via_end) -
    //                                    std::max(data.b_left[j],
    //                                    via_begin))));
    //         auto new_vio_begin =
    //             (is_v ? data.b_left[j] : data.b_bottom[j]) - via_width / 2 +
    //             1;
    //         auto new_vio_end =
    //             (is_v ? data.b_right[j] : data.b_top[j]) + via_width / 2 - 1;
    //         if (d == is_v) { // consider prl spacing constraint
    //             auto s = findPRLSpacing(
    //                 l,
    //                 std::max((is_v ? (data.b_right[j] - data.b_left[j])
    //                                : (data.b_top[j] - data.b_bottom[j])),
    //                          via_width),
    //                 prl);
    //             new_vio_begin -= s;
    //             new_vio_end += s;
    //         }
    //         int new_vio_track_begin =
    //             std::ceil(1.0 * (vio_begin - data.layer_track_start[l]) /
    //                       data.layer_track_step[l]);
    //         int new_vio_track_end =
    //             std::floor(1.0 * (vio_end - data.layer_track_start[l]) /
    //                        data.layer_track_step[l]);
    //         new_vio_track_begin =
    //             std::max(new_vio_track_begin, data.ir_track_low[i]);
    //         new_vio_track_end =
    //             std::min(new_vio_track_end, data.ir_track_high[i]);
    //         for (auto t = new_vio_track_begin; t <= new_vio_track_end; t++) {
    //             if (t >= vio_track_begin && t <= vio_track_end)
    //                 continue;
    //             auto coor =
    //                 data.layer_track_start[l] + data.layer_track_step[l] * t;
    //             int extension = 0;
    //             if (via_width < data.layer_eol_width[l]) { // consider eol
    //                                                        // spacing
    //                                                        constraint
    //                 auto upperEdge =
    //                     coor + via_width / 2 + data.layer_eol_within[l];
    //                 auto lowerEdge =
    //                     coor - via_width / 2 - data.layer_eol_within[l];
    //                 auto overlap = std::max(
    //                     0, std::min(upperEdge,
    //                                 (is_v ? data.b_right[j] : data.b_top[j]))
    //                                 -
    //                            std::max(lowerEdge, (is_v ? data.b_left[j]
    //                                                      :
    //                                                      data.b_bottom[j])));
    //                 if (overlap > 0)
    //                     extension = data.layer_eol_spacing[l];
    //             }
    //             auto overlap = std::max(
    //                 0, std::min((is_v ? data.b_top[j] : data.b_right[j]) +
    //                                 extension,
    //                             via_end) -
    //                        std::max((is_v ? data.b_bottom[j] :
    //                        data.b_left[j]) -
    //                                     extension,
    //                                 via_begin));
    //             data.ir_vio_cost_list[data.ir_vio_cost_start[i] + t -
    //                                   data.ir_track_low[i]] += overlap;
    //         }
    //     }
    // }
}
} // namespace gta