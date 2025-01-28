#include "GTA.hpp"
#include "db/infra/frTime.h"
#include "ops/assign/Assign.hpp"
#include "ops/copy/Copy.hpp"
#include "ops/extract/Extract.hpp"
#include "ops/init/Init.hpp"
#include <chrono>

namespace gta {
GTA::GTA(fr::frDesign *in) : tech(in->getTech()), design(in) {
    fr::frTime t;
    // extractGTADataFromDatabase();
    auto tp_0 = std::chrono::high_resolution_clock::now();
    ops::extract(tech, design, data);
    auto tp_1 = std::chrono::high_resolution_clock::now();
    std::cout << "extract time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tp_1 -
                                                                       tp_0)
                         .count() /
                     1e3
              << " s" << std::endl;
    t.print();
    std::cout << std::endl;
    // std::cout << "# layer " << data.num_layers << std::endl;
    // std::cout << "# gcell-x " << data.num_gcells_x << std::endl;
    // std::cout << "# gcell-y " << data.num_gcells_y << std::endl;
    // std::cout << "# gcell " << data.layer_gcell_start[data.num_layers]
    //           << std::endl;
    // std::cout << "# net " << data.num_nets << std::endl;
    // std::cout << "# guide " << data.num_guides << std::endl;
    // std::cout << "# nbr guide " << data.ir_nbr_start[data.num_guides]
    //           << std::endl;
    // std::cout << "# blockages " << data.num_blks << std::endl;
    // std::cout << "# ir end list "
    //           << data.gcell_end_point_ir_start
    //                  [data.layer_gcell_start[data.num_layers]]
    //           << std::endl;
    // std::cout << "# blk end list "
    //           << data.gcell_end_point_blk_start
    //                  [data.layer_gcell_start[data.num_layers]]
    //           << std::endl;
    // std::cout << "# ir super list " <<
    // data.ir_super_set_start[data.num_guides]
    //           << std::endl;
    // std::cout << "# blk super list "
    //           << data.blk_super_set_start[data.num_guides] << std::endl;
    // std::cout << "# vio cost list " <<
    // data.ir_vio_cost_start[data.num_guides]
    //           << std::endl;
}

template <typename T> void my_free(T *&ptr) {
    assert(ptr);
    ::free(ptr);
    ptr = nullptr;
}

GTA::~GTA() {
    my_free(data.layer_type);
    my_free(data.layer_direction);
    my_free(data.layer_width);
    my_free(data.layer_pitch);
    my_free(data.layer_track_start);
    my_free(data.layer_track_step);
    my_free(data.layer_track_num);
    my_free(data.layer_wire_weight);
    my_free(data.layer_eol_spacing);
    my_free(data.layer_eol_width);
    my_free(data.layer_eol_within);
    my_free(data.layer_panel_start);
    my_free(data.layer_panel_length);
    my_free(data.layer_gcell_start);
    my_free(data.layer_spacing_table_spacing_start);
    my_free(data.layer_spacing_table_width_start);
    my_free(data.layer_spacing_table_prl_start);
    my_free(data.layer_spacing_table_spacing);
    my_free(data.layer_spacing_table_width);
    my_free(data.layer_spacing_table_prl);
    my_free(data.layer_via_lower_width);
    my_free(data.layer_via_lower_length);
    my_free(data.layer_via_upper_width);
    my_free(data.layer_via_upper_length);
    my_free(data.layer_enable_via_wire_drc);

    my_free(data.ir_layer);
    my_free(data.ir_net);
    my_free(data.ir_panel);
    my_free(data.ir_gcell_begin);
    my_free(data.ir_gcell_end);
    if (data.ir_gcell_begin_via_offset)
        my_free(data.ir_gcell_begin_via_offset);
    if (data.ir_gcell_end_via_offset)
        my_free(data.ir_gcell_end_via_offset);
    my_free(data.ir_begin);
    my_free(data.ir_end);
    my_free(data.ir_track);
    my_free(data.ir_track_low);
    my_free(data.ir_track_high);
    my_free(data.ir_wl_weight);
    my_free(data.ir_has_ap);
    my_free(data.ir_has_proj_ap);
    my_free(data.ir_ap);
    my_free(data.ir_nbr_start);
    my_free(data.ir_nbr_list);
    my_free(data.ir_reassign);

    my_free(data.b_left);
    my_free(data.b_bottom);
    my_free(data.b_right);
    my_free(data.b_top);
    my_free(data.b_net);
    my_free(data.b_layer);
    my_free(data.b_panel_begin);
    my_free(data.b_panel_end);
    my_free(data.b_gcell_begin);
    my_free(data.b_gcell_end);

    my_free(data.gcell_end_point_ir_start);
    my_free(data.gcell_end_point_ir_list);
    my_free(data.gcell_end_point_blk_start);
    my_free(data.gcell_end_point_blk_list);
    if (data.gcell_cross_ir_start)
        my_free(data.gcell_cross_ir_start);
    if (data.gcell_cross_ir_list)
        my_free(data.gcell_cross_ir_list);
    my_free(data.ir_super_set_start);
    my_free(data.ir_super_set_list);
    my_free(data.blk_super_set_start);
    my_free(data.blk_super_set_list);

    my_free(data.ir_vio_cost_start);
    my_free(data.ir_vio_cost_list);
    my_free(data.ir_align_list);
    my_free(data.ir_key_cost);
}

void GTA::run(int maxIter, bool cuda) {
    fr::frTime t;
    ops::malloc_device_data(data, data_device);
    ops::h2d_data(data, data_device);
    for (iter = 0; iter < maxIter; iter++) {
        ops::init(data_device, iter, data.layer_direction[0]);
        ops::assign(data_device, iter, data.layer_direction[0]);
        ops::init(data_device, iter, data.layer_direction[0] ^ 1);
        ops::assign(data_device, iter, data.layer_direction[0] ^ 1);
        t.print();
        std::cout << std::endl;
    }
    ops::d2h_data(data, data_device);
    saveToGuide();
    t.print();
    std::cout << std::endl;
}

void GTA::saveToGuide() {
    auto &nets = design->getTopBlock()->getNets();
    int *net_guide_offset = (int *)calloc(sizeof(int), nets.size() + 1);
    for (auto i = 0; i < nets.size(); i++) {
        auto net = nets[i].get();
        net_guide_offset[i + 1] = net->getGuides().size();
    }
    for (auto i = 0; i < nets.size(); i++)
        net_guide_offset[i + 1] += net_guide_offset[i];
#pragma omp parallel for
    for (auto i = 0; i < nets.size(); i++) {
        auto net = nets[i].get();
        auto &guides = net->getGuides();
        for (auto j = 0; j < guides.size(); j++) {
            auto g = guides[j].get();
            auto pathSeg = std::make_unique<fr::frPathSeg>();
            auto idx = net_guide_offset[i] + j;
            auto l = data.ir_layer[idx];
            auto is_v = data.layer_direction[l];
            assert(data.ir_track[idx] >= data.ir_track_low[idx]);
            assert(data.ir_track[idx] <= data.ir_track_high[idx]);
            auto coor = data.layer_track_start[l] +
                        data.layer_track_step[l] * data.ir_track[idx];
            if (is_v) {
                auto bp = fr::frPoint(coor, data.ir_begin[idx]);
                auto ep = fr::frPoint(coor, data.ir_end[idx]);
                pathSeg->setPoints(bp, ep);
            } else {
                auto bp = fr::frPoint(data.ir_begin[idx], coor);
                auto ep = fr::frPoint(data.ir_end[idx], coor);
                pathSeg->setPoints(bp, ep);
            }
            pathSeg->setLayerNum(l + 2);
            pathSeg->addToNet(net);
            std::vector<std::unique_ptr<fr::frConnFig>> tmp;
            tmp.push_back(std::move(pathSeg));
            g->setRoutes(tmp);
        }
    }
#pragma omp barrier
    free(net_guide_offset);
}

// int GTA::findPRLSpacing(int l, int width, int prl) const {
//     int row = 0, col = 0;
//     const auto row_num = data.layer_spacing_table_width_start[l + 1] -
//                          data.layer_spacing_table_width_start[l];
//     const auto col_num = data.layer_spacing_table_prl_start[l + 1] -
//                          data.layer_spacing_table_prl_start[l];
//     while (row + 1 < row_num &&
//            width > data.layer_spacing_table_width
//                        [data.layer_spacing_table_width_start[l] + row + 1])
//         row++;
//     while (
//         col + 1 < col_num &&
//         prl >
//             data.layer_spacing_table_prl[data.layer_spacing_table_prl_start[l]
//             +
//                                          col + 1])
//         col++;
//     return data
//         .layer_spacing_table_spacing[data.layer_spacing_table_spacing_start[l]
//         +
//                                      row * col_num + col];
// }
} // namespace gta
