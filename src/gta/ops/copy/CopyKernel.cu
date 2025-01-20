#include "CopyKernel.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

namespace gta::ops::cpu::helper {
template <class T> void malloc(T *&ptr, size_t N) {
    if (ptr) {
        std::cout << "[WARNING] IS NOT NULLPTR" << std::endl;
    } else {
        auto err = cudaMalloc((void **)&ptr, sizeof(T) * N);
        if (err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << std::endl;
            exit(0);
        }
    }
}

template <class T> void h2d(T *ptr_h, T *ptr_d, size_t N) {
    if (ptr_h == nullptr || ptr_d == nullptr) {
        std::cout << "[WARNING] NULLPTR" << std::endl;
        return;
    }
    auto err = cudaMemcpy(ptr_d, ptr_h, sizeof(T) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}

template <class T> void d2h(T *ptr_h, T *ptr_d, size_t N) {
    if (ptr_h == nullptr || ptr_d == nullptr) {
        std::cout << "[WARNING] NULLPTR" << std::endl;
        return;
    }
    auto err = cudaMemcpy(ptr_h, ptr_d, sizeof(T) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
}
} // namespace gta::ops::cpu::helper

namespace gta::ops::cpu {
void malloc_data(data::Data &data_h, data::Data &data_d) {
    // std::cout << "begin malloc device" << std::endl;
    data_d.device = data::Device::CUDA;
    data_d.num_layers = data_h.num_layers;
    data_d.num_gcells_x = data_h.num_gcells_x;
    data_d.num_gcells_y = data_h.num_gcells_y;
    data_d.gcell_start_x = data_h.gcell_start_x;
    data_d.gcell_start_y = data_h.gcell_start_y;
    data_d.gcell_step_x = data_h.gcell_step_x;
    data_d.gcell_step_y = data_h.gcell_step_y;
    data_d.num_nets = data_h.num_nets;
    data_d.num_guides = data_h.num_guides;
    data_d.num_blks = data_h.num_blks;
    data_d.num_gcells = data_h.num_gcells;
    helper::malloc(data_d.layer_type, data_h.num_layers);

    helper::malloc(data_d.layer_direction, data_h.num_layers);

    helper::malloc(data_d.layer_width, data_h.num_layers);

    helper::malloc(data_d.layer_pitch, data_h.num_layers);

    helper::malloc(data_d.layer_track_start, data_h.num_layers);

    helper::malloc(data_d.layer_track_step, data_h.num_layers);

    helper::malloc(data_d.layer_track_num, data_h.num_layers);

    helper::malloc(data_d.layer_wire_weight, data_h.num_layers);

    helper::malloc(data_d.layer_eol_spacing, data_h.num_layers);

    helper::malloc(data_d.layer_eol_width, data_h.num_layers);

    helper::malloc(data_d.layer_eol_within, data_h.num_layers);

    helper::malloc(data_d.layer_panel_start, data_h.num_layers + 1);

    helper::malloc(data_d.layer_panel_length, data_h.num_layers);

    helper::malloc(data_d.layer_gcell_start, data_h.num_layers + 1);

    helper::malloc(data_d.layer_spacing_table_spacing_start,
                   data_h.num_layers + 1);

    helper::malloc(data_d.layer_spacing_table_width_start,
                   data_h.num_layers + 1);

    helper::malloc(data_d.layer_spacing_table_prl_start, data_h.num_layers + 1);

    helper::malloc(data_d.layer_spacing_table_spacing,
                   data_h.layer_spacing_table_spacing_start[data_h.num_layers]);

    helper::malloc(data_d.layer_spacing_table_width,
                   data_h.layer_spacing_table_width_start[data_h.num_layers]);

    helper::malloc(data_d.layer_spacing_table_prl,
                   data_h.layer_spacing_table_prl_start[data_h.num_layers]);

    helper::malloc(data_d.layer_via_lower_width, data_h.num_layers);

    helper::malloc(data_d.layer_via_lower_length, data_h.num_layers);

    helper::malloc(data_d.layer_via_upper_width, data_h.num_layers);

    helper::malloc(data_d.layer_via_upper_length, data_h.num_layers);

    helper::malloc(data_d.layer_via_span_x, data_h.num_layers);

    helper::malloc(data_d.layer_via_span_y, data_h.num_layers);

    helper::malloc(data_d.layer_cut_spacing, data_h.num_layers);

    helper::malloc(data_d.layer_enable_via_wire_drc, data_h.num_layers);

    helper::malloc(data_d.layer_enable_via_via_drc, data_h.num_layers);

    helper::malloc(data_d.layer_enable_corner_spacing, data_h.num_layers);

    helper::malloc(data_d.ir_layer, data_h.num_guides);

    helper::malloc(data_d.ir_net, data_h.num_guides);

    helper::malloc(data_d.ir_panel, data_h.num_guides);

    helper::malloc(data_d.ir_gcell_begin, data_h.num_guides);

    helper::malloc(data_d.ir_gcell_end, data_h.num_guides);

    helper::malloc(data_d.ir_gcell_begin_via_offset, data_h.num_guides);

    helper::malloc(data_d.ir_gcell_end_via_offset, data_h.num_guides);

    helper::malloc(data_d.ir_begin, data_h.num_guides);

    helper::malloc(data_d.ir_end, data_h.num_guides);

    helper::malloc(data_d.ir_track, data_h.num_guides);

    helper::malloc(data_d.ir_track_low, data_h.num_guides);

    helper::malloc(data_d.ir_track_high, data_h.num_guides);

    helper::malloc(data_d.ir_wl_weight, data_h.num_guides);

    helper::malloc(data_d.ir_has_ap, data_h.num_guides);

    helper::malloc(data_d.ir_has_proj_ap, data_h.num_guides);

    helper::malloc(data_d.ir_ap, data_h.num_guides);

    helper::malloc(data_d.ir_nbr_start, data_h.num_guides + 1);

    helper::malloc(data_d.ir_nbr_list, data_h.ir_nbr_start[data_h.num_guides]);

    helper::malloc(data_d.ir_reassign, data_h.num_guides);

    helper::malloc(data_d.ir_lower_via_start, data_h.num_guides + 1);

    helper::malloc(data_d.ir_upper_via_start, data_h.num_guides + 1);

    helper::malloc(data_d.b_left, data_h.num_blks);

    helper::malloc(data_d.b_bottom, data_h.num_blks);

    helper::malloc(data_d.b_right, data_h.num_blks);

    helper::malloc(data_d.b_top, data_h.num_blks);

    helper::malloc(data_d.b_net, data_h.num_blks);

    helper::malloc(data_d.b_use_min_width, data_h.num_blks);

    helper::malloc(data_d.b_layer, data_h.num_blks);

    helper::malloc(data_d.b_panel_begin, data_h.num_blks);

    helper::malloc(data_d.b_panel_end, data_h.num_blks);

    helper::malloc(data_d.b_gcell_begin, data_h.num_blks);

    helper::malloc(data_d.b_gcell_end, data_h.num_blks);

    helper::malloc(data_d.gcell_end_point_ir_start,
                   data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::malloc(data_d.gcell_end_point_ir_list,
                   data_h.gcell_end_point_ir_start
                       [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::malloc(data_d.gcell_end_point_blk_start,
                   data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::malloc(data_d.gcell_end_point_blk_list,
                   data_h.gcell_end_point_blk_start
                       [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::malloc(data_d.gcell_cross_ir_start,
                   data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::malloc(
        data_d.gcell_cross_ir_list,
        data_h
            .gcell_cross_ir_start[data_h.layer_gcell_start[data_h.num_layers]]);

    helper::malloc(data_d.gcell_lower_via_start,
                   data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::malloc(data_d.gcell_upper_via_start,
                   data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::malloc(data_d.ir_super_set_start, data_h.num_guides + 1);

    helper::malloc(data_d.ir_super_set_list,
                   data_h.ir_super_set_start[data_h.num_guides]);

    helper::malloc(data_d.blk_super_set_start, data_h.num_guides + 1);

    helper::malloc(data_d.blk_super_set_list,
                   data_h.blk_super_set_start[data_h.num_guides]);

    helper::malloc(data_d.ir_vio_cost_start, data_h.num_guides + 1);

    helper::malloc(data_d.ir_vio_cost_list,
                   data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::malloc(data_d.ir_align_list,
                   data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::malloc(data_d.ir_via_vio_list,
                   data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::malloc(data_d.ir_key_cost, data_h.num_guides);

    // std::cout << "finish malloc device" << std::endl;
}

void h2d_data(data::Data &data_h, data::Data &data_d) {
    // std::cout << "begin h2d" << std::endl;
    assert(data_h.device == data::Device::CPU);
    assert(data_d.device == data::Device::CUDA);
    data_d.num_layers = data_h.num_layers;
    data_d.num_gcells_x = data_h.num_gcells_x;
    data_d.num_gcells_y = data_h.num_gcells_y;
    data_d.gcell_start_x = data_h.gcell_start_x;
    data_d.gcell_start_y = data_h.gcell_start_y;
    data_d.gcell_step_x = data_h.gcell_step_x;
    data_d.gcell_step_y = data_h.gcell_step_y;
    data_d.num_nets = data_h.num_nets;
    data_d.num_guides = data_h.num_guides;
    data_d.num_blks = data_h.num_blks;
    data_d.num_gcells = data_h.num_gcells;
    helper::h2d(data_h.layer_type, data_d.layer_type, data_h.num_layers);

    helper::h2d(data_h.layer_direction, data_d.layer_direction,
                data_h.num_layers);

    helper::h2d(data_h.layer_width, data_d.layer_width, data_h.num_layers);

    helper::h2d(data_h.layer_pitch, data_d.layer_pitch, data_h.num_layers);

    helper::h2d(data_h.layer_track_start, data_d.layer_track_start,
                data_h.num_layers);

    helper::h2d(data_h.layer_track_start, data_d.layer_track_start,
                data_h.num_layers);

    helper::h2d(data_h.layer_track_step, data_d.layer_track_step,
                data_h.num_layers);

    helper::h2d(data_h.layer_track_num, data_d.layer_track_num,
                data_h.num_layers);

    helper::h2d(data_h.layer_wire_weight, data_d.layer_wire_weight,
                data_h.num_layers);

    helper::h2d(data_h.layer_eol_spacing, data_d.layer_eol_spacing,
                data_h.num_layers);

    helper::h2d(data_h.layer_eol_width, data_d.layer_eol_width,
                data_h.num_layers);

    helper::h2d(data_h.layer_eol_within, data_d.layer_eol_within,
                data_h.num_layers);

    helper::h2d(data_h.layer_panel_start, data_d.layer_panel_start,
                data_h.num_layers + 1);

    helper::h2d(data_h.layer_panel_length, data_d.layer_panel_length,
                data_h.num_layers);

    helper::h2d(data_h.layer_gcell_start, data_d.layer_gcell_start,
                data_h.num_layers + 1);

    helper::h2d(data_h.layer_spacing_table_spacing_start,
                data_d.layer_spacing_table_spacing_start,
                data_h.num_layers + 1);

    helper::h2d(data_h.layer_spacing_table_width_start,
                data_d.layer_spacing_table_width_start, data_h.num_layers + 1);

    helper::h2d(data_h.layer_spacing_table_prl_start,
                data_d.layer_spacing_table_prl_start, data_h.num_layers + 1);

    helper::h2d(data_h.layer_spacing_table_spacing,
                data_d.layer_spacing_table_spacing,
                data_h.layer_spacing_table_spacing_start[data_h.num_layers]);

    helper::h2d(data_h.layer_spacing_table_width,
                data_d.layer_spacing_table_width,
                data_h.layer_spacing_table_width_start[data_h.num_layers]);

    helper::h2d(data_h.layer_spacing_table_prl, data_d.layer_spacing_table_prl,
                data_h.layer_spacing_table_prl_start[data_h.num_layers]);

    helper::h2d(data_h.layer_via_lower_width, data_d.layer_via_lower_width,
                data_h.num_layers);

    helper::h2d(data_h.layer_via_lower_length, data_d.layer_via_lower_length,
                data_h.num_layers);

    helper::h2d(data_h.layer_via_upper_width, data_d.layer_via_upper_width,
                data_h.num_layers);

    helper::h2d(data_h.layer_via_upper_length, data_d.layer_via_upper_length,
                data_h.num_layers);

    helper::h2d(data_h.layer_via_span_x, data_d.layer_via_span_x,
                data_h.num_layers);

    helper::h2d(data_h.layer_via_span_y, data_d.layer_via_span_y,
                data_h.num_layers);

    helper::h2d(data_h.layer_cut_spacing, data_d.layer_cut_spacing,
                data_h.num_layers);

    helper::h2d(data_h.layer_enable_via_wire_drc,
                data_d.layer_enable_via_wire_drc, data_h.num_layers);

    helper::h2d(data_h.layer_enable_via_via_drc,
                data_d.layer_enable_via_via_drc, data_h.num_layers);

    helper::h2d(data_h.layer_enable_corner_spacing,
                data_d.layer_enable_corner_spacing, data_h.num_layers);

    helper::h2d(data_h.ir_layer, data_d.ir_layer, data_h.num_guides);

    helper::h2d(data_h.ir_net, data_d.ir_net, data_h.num_guides);

    helper::h2d(data_h.ir_panel, data_d.ir_panel, data_h.num_guides);

    helper::h2d(data_h.ir_gcell_begin, data_d.ir_gcell_begin,
                data_h.num_guides);

    helper::h2d(data_h.ir_gcell_end, data_d.ir_gcell_end, data_h.num_guides);

    helper::h2d(data_h.ir_gcell_begin_via_offset,
                data_d.ir_gcell_begin_via_offset, data_h.num_guides);

    helper::h2d(data_h.ir_gcell_end_via_offset, data_d.ir_gcell_end_via_offset,
                data_h.num_guides);

    helper::h2d(data_h.ir_begin, data_d.ir_begin, data_h.num_guides);

    helper::h2d(data_h.ir_end, data_d.ir_end, data_h.num_guides);

    helper::h2d(data_h.ir_track, data_d.ir_track, data_h.num_guides);

    helper::h2d(data_h.ir_track_low, data_d.ir_track_low, data_h.num_guides);

    helper::h2d(data_h.ir_track_high, data_d.ir_track_high, data_h.num_guides);

    helper::h2d(data_h.ir_wl_weight, data_d.ir_wl_weight, data_h.num_guides);

    helper::h2d(data_h.ir_has_ap, data_d.ir_has_ap, data_h.num_guides);

    helper::h2d(data_h.ir_has_proj_ap, data_d.ir_has_proj_ap,
                data_h.num_guides);

    helper::h2d(data_h.ir_ap, data_d.ir_ap, data_h.num_guides);

    helper::h2d(data_h.ir_nbr_start, data_d.ir_nbr_start,
                data_h.num_guides + 1);

    helper::h2d(data_h.ir_nbr_list, data_d.ir_nbr_list,
                data_h.ir_nbr_start[data_h.num_guides]);

    helper::h2d(data_h.ir_reassign, data_d.ir_reassign, data_h.num_guides);

    // helper::h2d(data_h.ir_lower_via_start, data_d.ir_lower_via_start,
    //             data_h.num_guides + 1);
    // helper::h2d(data_h.ir_lower_via_coor, data_d.ir_lower_via_coor,
    //             data_h.ir_lower_via_start[data_h.num_guides]);
    // helper::h2d(data_h.ir_upper_via_start, data_d.ir_upper_via_start,
    //             data_h.num_guides + 1);
    // helper::h2d(data_h.ir_upper_via_coor, data_d.ir_upper_via_coor,
    //             data_h.ir_upper_via_start[data_h.num_guides]);
    helper::h2d(data_h.b_left, data_d.b_left, data_h.num_blks);

    helper::h2d(data_h.b_bottom, data_d.b_bottom, data_h.num_blks);

    helper::h2d(data_h.b_right, data_d.b_right, data_h.num_blks);

    helper::h2d(data_h.b_top, data_d.b_top, data_h.num_blks);

    helper::h2d(data_h.b_net, data_d.b_net, data_h.num_blks);

    helper::h2d(data_h.b_use_min_width, data_d.b_use_min_width,
                data_h.num_blks);

    helper::h2d(data_h.b_layer, data_d.b_layer, data_h.num_blks);

    helper::h2d(data_h.b_panel_begin, data_d.b_panel_begin, data_h.num_blks);

    helper::h2d(data_h.b_panel_end, data_d.b_panel_end, data_h.num_blks);

    helper::h2d(data_h.b_gcell_begin, data_d.b_gcell_begin, data_h.num_blks);

    helper::h2d(data_h.b_gcell_end, data_d.b_gcell_end, data_h.num_blks);

    helper::h2d(data_h.gcell_end_point_ir_start,
                data_d.gcell_end_point_ir_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::h2d(data_h.gcell_end_point_ir_list, data_d.gcell_end_point_ir_list,
                data_h.gcell_end_point_ir_start
                    [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::h2d(data_h.gcell_end_point_blk_start,
                data_d.gcell_end_point_blk_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::h2d(data_h.gcell_end_point_blk_list,
                data_d.gcell_end_point_blk_list,
                data_h.gcell_end_point_blk_start
                    [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::h2d(data_h.gcell_cross_ir_start, data_d.gcell_cross_ir_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::h2d(
        data_h.gcell_cross_ir_list, data_d.gcell_cross_ir_list,
        data_h
            .gcell_cross_ir_start[data_h.layer_gcell_start[data_h.num_layers]]);

    helper::h2d(data_h.ir_super_set_start, data_d.ir_super_set_start,
                data_h.num_guides + 1);

    helper::h2d(data_h.ir_super_set_list, data_d.ir_super_set_list,
                data_h.ir_super_set_start[data_h.num_guides]);

    helper::h2d(data_h.blk_super_set_start, data_d.blk_super_set_start,
                data_h.num_guides + 1);

    helper::h2d(data_h.blk_super_set_list, data_d.blk_super_set_list,
                data_h.blk_super_set_start[data_h.num_guides]);

    helper::h2d(data_h.ir_vio_cost_start, data_d.ir_vio_cost_start,
                data_h.num_guides + 1);
    helper::h2d(data_h.ir_vio_cost_list, data_d.ir_vio_cost_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);
    helper::h2d(data_h.ir_align_list, data_d.ir_align_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);
    helper::h2d(data_h.ir_via_vio_list, data_d.ir_via_vio_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);
    helper::h2d(data_h.ir_key_cost, data_d.ir_key_cost, data_h.num_guides);

    // std::cout << "finish h2d" << std::endl;
}

void d2h_data(data::Data &data_h, data::Data &data_d) {
    helper::d2h(data_h.layer_type, data_d.layer_type, data_h.num_layers);

    helper::d2h(data_h.layer_direction, data_d.layer_direction,
                data_h.num_layers);

    helper::d2h(data_h.layer_width, data_d.layer_width, data_h.num_layers);

    helper::d2h(data_h.layer_pitch, data_d.layer_pitch, data_h.num_layers);

    helper::d2h(data_h.layer_track_start, data_d.layer_track_start,
                data_h.num_layers);

    helper::d2h(data_h.layer_track_start, data_d.layer_track_start,
                data_h.num_layers);

    helper::d2h(data_h.layer_track_step, data_d.layer_track_step,
                data_h.num_layers);

    helper::d2h(data_h.layer_track_num, data_d.layer_track_num,
                data_h.num_layers);

    helper::d2h(data_h.layer_wire_weight, data_d.layer_wire_weight,
                data_h.num_layers);

    helper::d2h(data_h.layer_eol_spacing, data_d.layer_eol_spacing,
                data_h.num_layers);

    helper::d2h(data_h.layer_eol_width, data_d.layer_eol_width,
                data_h.num_layers);

    helper::d2h(data_h.layer_eol_within, data_d.layer_eol_within,
                data_h.num_layers);

    helper::d2h(data_h.layer_panel_start, data_d.layer_panel_start,
                data_h.num_layers + 1);

    helper::d2h(data_h.layer_panel_length, data_d.layer_panel_length,
                data_h.num_layers);

    helper::d2h(data_h.layer_gcell_start, data_d.layer_gcell_start,
                data_h.num_layers + 1);

    helper::d2h(data_h.layer_spacing_table_spacing_start,
                data_d.layer_spacing_table_spacing_start,
                data_h.num_layers + 1);

    helper::d2h(data_h.layer_spacing_table_width_start,
                data_d.layer_spacing_table_width_start, data_h.num_layers + 1);

    helper::d2h(data_h.layer_spacing_table_prl_start,
                data_d.layer_spacing_table_prl_start, data_h.num_layers + 1);

    helper::d2h(data_h.layer_spacing_table_spacing,
                data_d.layer_spacing_table_spacing,
                data_h.layer_spacing_table_spacing_start[data_h.num_layers]);

    helper::d2h(data_h.layer_spacing_table_width,
                data_d.layer_spacing_table_width,
                data_h.layer_spacing_table_width_start[data_h.num_layers]);

    helper::d2h(data_h.layer_spacing_table_prl, data_d.layer_spacing_table_prl,
                data_h.layer_spacing_table_prl_start[data_h.num_layers]);

    helper::d2h(data_h.layer_via_lower_width, data_d.layer_via_lower_width,
                data_h.num_layers);

    helper::d2h(data_h.layer_via_lower_length, data_d.layer_via_lower_length,
                data_h.num_layers);

    helper::d2h(data_h.layer_via_upper_width, data_d.layer_via_upper_width,
                data_h.num_layers);

    helper::d2h(data_h.layer_via_upper_length, data_d.layer_via_upper_length,
                data_h.num_layers);

    helper::d2h(data_h.layer_via_span_x, data_d.layer_via_span_x,
                data_h.num_layers);

    helper::d2h(data_h.layer_via_span_y, data_d.layer_via_span_y,
                data_h.num_layers);

    helper::d2h(data_h.layer_cut_spacing, data_d.layer_cut_spacing,
                data_h.num_layers);

    helper::d2h(data_h.layer_enable_via_wire_drc,
                data_d.layer_enable_via_wire_drc, data_h.num_layers);

    helper::d2h(data_h.layer_enable_via_via_drc,
                data_d.layer_enable_via_via_drc, data_h.num_layers);

    helper::d2h(data_h.layer_enable_corner_spacing,
                data_d.layer_enable_corner_spacing, data_h.num_layers);

    helper::d2h(data_h.ir_layer, data_d.ir_layer, data_h.num_guides);

    helper::d2h(data_h.ir_net, data_d.ir_net, data_h.num_guides);

    helper::d2h(data_h.ir_panel, data_d.ir_panel, data_h.num_guides);

    helper::d2h(data_h.ir_gcell_begin, data_d.ir_gcell_begin,
                data_h.num_guides);

    helper::d2h(data_h.ir_gcell_end, data_d.ir_gcell_end, data_h.num_guides);

    helper::d2h(data_h.ir_gcell_begin_via_offset,
                data_d.ir_gcell_begin_via_offset, data_h.num_guides);

    helper::d2h(data_h.ir_gcell_end_via_offset, data_d.ir_gcell_end_via_offset,
                data_h.num_guides);

    helper::d2h(data_h.ir_begin, data_d.ir_begin, data_h.num_guides);

    helper::d2h(data_h.ir_end, data_d.ir_end, data_h.num_guides);

    helper::d2h(data_h.ir_track, data_d.ir_track, data_h.num_guides);

    helper::d2h(data_h.ir_track_low, data_d.ir_track_low, data_h.num_guides);

    helper::d2h(data_h.ir_track_high, data_d.ir_track_high, data_h.num_guides);

    helper::d2h(data_h.ir_wl_weight, data_d.ir_wl_weight, data_h.num_guides);

    helper::d2h(data_h.ir_has_ap, data_d.ir_has_ap, data_h.num_guides);

    helper::d2h(data_h.ir_has_proj_ap, data_d.ir_has_proj_ap,
                data_h.num_guides);

    helper::d2h(data_h.ir_ap, data_d.ir_ap, data_h.num_guides);

    helper::d2h(data_h.ir_nbr_start, data_d.ir_nbr_start,
                data_h.num_guides + 1);

    helper::d2h(data_h.ir_nbr_list, data_d.ir_nbr_list,
                data_h.ir_nbr_start[data_h.num_guides]);

    helper::d2h(data_h.ir_reassign, data_d.ir_reassign, data_h.num_guides);

    // helper::d2h(data_h.ir_lower_via_start, data_d.ir_lower_via_start,
    //             data_h.num_guides + 1);
    // helper::d2h(data_h.ir_lower_via_coor, data_d.ir_lower_via_coor,
    //             data_h.ir_lower_via_start[data_h.num_guides]);
    // helper::d2h(data_h.ir_upper_via_start, data_d.ir_upper_via_start,
    //             data_h.num_guides + 1);
    // helper::d2h(data_h.ir_upper_via_coor, data_d.ir_upper_via_coor,
    //             data_h.ir_upper_via_start[data_h.num_guides]);
    helper::d2h(data_h.b_left, data_d.b_left, data_h.num_blks);

    helper::d2h(data_h.b_bottom, data_d.b_bottom, data_h.num_blks);

    helper::d2h(data_h.b_right, data_d.b_right, data_h.num_blks);

    helper::d2h(data_h.b_top, data_d.b_top, data_h.num_blks);

    helper::d2h(data_h.b_net, data_d.b_net, data_h.num_blks);

    helper::d2h(data_h.b_use_min_width, data_d.b_use_min_width,
                data_h.num_blks);

    helper::d2h(data_h.b_layer, data_d.b_layer, data_h.num_blks);

    helper::d2h(data_h.b_panel_begin, data_d.b_panel_begin, data_h.num_blks);

    helper::d2h(data_h.b_panel_end, data_d.b_panel_end, data_h.num_blks);

    helper::d2h(data_h.b_gcell_begin, data_d.b_gcell_begin, data_h.num_blks);

    helper::d2h(data_h.b_gcell_end, data_d.b_gcell_end, data_h.num_blks);

    helper::d2h(data_h.gcell_end_point_ir_start,
                data_d.gcell_end_point_ir_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::d2h(data_h.gcell_end_point_ir_list, data_d.gcell_end_point_ir_list,
                data_h.gcell_end_point_ir_start
                    [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::d2h(data_h.gcell_end_point_blk_start,
                data_d.gcell_end_point_blk_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::d2h(data_h.gcell_end_point_blk_list,
                data_d.gcell_end_point_blk_list,
                data_h.gcell_end_point_blk_start
                    [data_h.layer_gcell_start[data_h.num_layers]]);

    helper::d2h(data_h.gcell_cross_ir_start, data_d.gcell_cross_ir_start,
                data_h.layer_gcell_start[data_h.num_layers] + 1);

    helper::d2h(
        data_h.gcell_cross_ir_list, data_d.gcell_cross_ir_list,
        data_h
            .gcell_cross_ir_start[data_h.layer_gcell_start[data_h.num_layers]]);

    helper::d2h(data_h.ir_super_set_start, data_d.ir_super_set_start,
                data_h.num_guides + 1);

    helper::d2h(data_h.ir_super_set_list, data_d.ir_super_set_list,
                data_h.ir_super_set_start[data_h.num_guides]);

    helper::d2h(data_h.blk_super_set_start, data_d.blk_super_set_start,
                data_h.num_guides + 1);

    helper::d2h(data_h.blk_super_set_list, data_d.blk_super_set_list,
                data_h.blk_super_set_start[data_h.num_guides]);

    helper::d2h(data_h.ir_vio_cost_start, data_d.ir_vio_cost_start,
                data_h.num_guides + 1);

    helper::d2h(data_h.ir_vio_cost_list, data_d.ir_vio_cost_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::d2h(data_h.ir_align_list, data_d.ir_align_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::d2h(data_h.ir_via_vio_list, data_d.ir_via_vio_list,
                data_h.ir_vio_cost_start[data_h.num_guides]);

    helper::d2h(data_h.ir_key_cost, data_d.ir_key_cost, data_h.num_guides);
}
} // namespace gta::ops::cpu