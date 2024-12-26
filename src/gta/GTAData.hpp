#pragma once

#include <cstddef>
#include <cstdint>

namespace gta {
struct GTAData {
    int num_layers;
    int num_gcells_x;
    int num_gcells_y;
    int gcell_start_x;
    int gcell_start_y;
    int gcell_step_x;
    int gcell_step_y;
    int num_nets;
    int num_guides;
    int num_blks;
    int *layer_type = nullptr;
    int *layer_direction = nullptr;
    int *layer_width = nullptr;
    int *layer_pitch = nullptr;
    int *layer_track_start = nullptr;
    int *layer_track_step = nullptr;
    int *layer_track_num = nullptr;
    float *layer_wire_weight = nullptr;
    int *layer_eol_spacing = nullptr;
    int *layer_eol_width = nullptr;
    int *layer_eol_within = nullptr;
    int *layer_panel_start = nullptr;
    int *layer_panel_length = nullptr;
    int *layer_gcell_start = nullptr;
    int *layer_spacing_table_spacing_start = nullptr;
    int *layer_spacing_table_width_start = nullptr;
    int *layer_spacing_table_prl_start = nullptr;
    int *layer_spacing_table_spacing = nullptr;
    int *layer_spacing_table_width = nullptr;
    int *layer_spacing_table_prl = nullptr;
    int *layer_via_lower_width = nullptr;
    int *layer_via_lower_length = nullptr;
    int *layer_via_upper_width = nullptr;
    int *layer_via_upper_length = nullptr;
    int *layer_cut_spacing = nullptr;
    bool *layer_enable_via_wire_drc = nullptr;
    bool *layer_enable_corner_spacing = nullptr;

    short *ir_layer = nullptr;
    int *ir_net = nullptr;
    short *ir_panel = nullptr;
    short *ir_gcell_begin = nullptr;
    short *ir_gcell_end = nullptr;
    short *ir_gcell_begin_via_offset = nullptr;
    short *ir_gcell_end_via_offset = nullptr;
    int *ir_begin = nullptr;
    int *ir_end = nullptr;
    int *ir_track = nullptr;
    int *ir_track_low = nullptr;
    int *ir_track_high = nullptr;
    float *ir_wl_weight = nullptr;
    bool *ir_has_ap = nullptr;
    bool *ir_has_proj_ap = nullptr;
    int *ir_ap = nullptr;
    int *ir_nbr_start = nullptr;
    int *ir_nbr_list = nullptr;
    int *ir_reassign = nullptr;
    int *ir_via_start = nullptr;
    int *ir_via_list_coor = nullptr;
    short *ir_via_list_layer = nullptr;

    int *b_left = nullptr;
    int *b_bottom = nullptr;
    int *b_right = nullptr;
    int *b_top = nullptr;
    int *b_net = nullptr;
    bool *b_use_min_width = nullptr;
    short *b_layer = nullptr;
    short *b_panel_begin = nullptr;
    short *b_panel_end = nullptr;
    short *b_gcell_begin = nullptr;
    short *b_gcell_end = nullptr;

    int *gcell_end_point_ir_start = nullptr;
    int *gcell_end_point_ir_list = nullptr;
    int *gcell_end_point_blk_start = nullptr;
    int *gcell_end_point_blk_list = nullptr;
    int *gcell_cross_ir_start = nullptr;
    int *gcell_cross_ir_list = nullptr;
    int *ir_super_set_start = nullptr;
    int *ir_super_set_list = nullptr;
    int *blk_super_set_start = nullptr;
    int *blk_super_set_list = nullptr;

    int *ir_vio_cost_start = nullptr;
    int *ir_vio_cost_list = nullptr;
    int8_t *ir_align_list = nullptr;
    int *ir_key_cost = nullptr;
    int8_t *via_vio_list = nullptr;
};

} // namespace gta