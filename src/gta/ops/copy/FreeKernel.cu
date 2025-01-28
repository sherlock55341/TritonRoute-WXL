#include "FreeKernel.hpp"

namespace gta::ops::cuda::helper {
template <class T> void free(T &ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    } else {
// #ifdef DEBUG_MODE
//         std::cout << "[WARNING] NULLPTR" << std::endl;
// #endif
    }
}
} // namespace gta::ops::kernel::helper

namespace gta::ops::cuda {
void free_data(data::Data &data) {
    assert(data.device == data::Device::CUDA);
    helper::free(data.layer_type);
    helper::free(data.layer_direction);
    helper::free(data.layer_width);
    helper::free(data.layer_pitch);
    helper::free(data.layer_track_start);
    helper::free(data.layer_track_step);
    helper::free(data.layer_track_num);
    helper::free(data.layer_wire_weight);
    helper::free(data.layer_eol_spacing);
    helper::free(data.layer_eol_width);
    helper::free(data.layer_eol_within);
    helper::free(data.layer_panel_start);
    helper::free(data.layer_panel_length);
    helper::free(data.layer_gcell_start);
    helper::free(data.layer_spacing_table_spacing_start);
    helper::free(data.layer_spacing_table_width_start);
    helper::free(data.layer_spacing_table_prl_start);
    helper::free(data.layer_spacing_table_spacing);
    helper::free(data.layer_spacing_table_width);
    helper::free(data.layer_spacing_table_prl);
    helper::free(data.layer_via_lower_width);
    helper::free(data.layer_via_lower_length);
    helper::free(data.layer_via_upper_width);
    helper::free(data.layer_via_upper_length);
    helper::free(data.layer_via_span_x);
    helper::free(data.layer_via_span_y);
    helper::free(data.layer_cut_spacing);
    helper::free(data.layer_enable_via_wire_drc);
    helper::free(data.layer_enable_via_via_drc);
    helper::free(data.layer_enable_corner_spacing);

    helper::free(data.ir_layer);
    helper::free(data.ir_net);
    helper::free(data.ir_panel);
    helper::free(data.ir_gcell_begin);
    helper::free(data.ir_gcell_end);
    helper::free(data.ir_gcell_begin_via_offset);
    helper::free(data.ir_gcell_end_via_offset);
    helper::free(data.ir_begin);
    helper::free(data.ir_end);
    helper::free(data.ir_track);
    helper::free(data.ir_track_low);
    helper::free(data.ir_track_high);
    helper::free(data.ir_wl_weight);
    helper::free(data.ir_has_ap);
    helper::free(data.ir_has_proj_ap);
    helper::free(data.ir_ap);
    helper::free(data.ir_nbr_start);
    helper::free(data.ir_nbr_list);
    helper::free(data.ir_reassign);
    helper::free(data.ir_lower_via_start);
    helper::free(data.ir_lower_via_coor);
    helper::free(data.ir_upper_via_start);
    helper::free(data.ir_upper_via_coor);

    helper::free(data.b_left);
    helper::free(data.b_bottom);
    helper::free(data.b_right);
    helper::free(data.b_top);
    helper::free(data.b_net);
    helper::free(data.b_use_min_width);
    helper::free(data.b_layer);
    helper::free(data.b_panel_begin);
    helper::free(data.b_panel_end);
    helper::free(data.b_gcell_begin);
    helper::free(data.b_gcell_end);

    helper::free(data.gcell_end_point_ir_start);
    helper::free(data.gcell_end_point_ir_list);
    helper::free(data.gcell_end_point_blk_start);
    helper::free(data.gcell_end_point_blk_list);
    helper::free(data.gcell_cross_ir_start);
    helper::free(data.gcell_cross_ir_list);
    helper::free(data.ir_super_set_start);
    helper::free(data.ir_super_set_list);
    helper::free(data.blk_super_set_start);
    helper::free(data.blk_super_set_list);

    helper::free(data.ir_vio_cost_start);
    helper::free(data.ir_vio_cost_list);
    helper::free(data.ir_align_list);
    helper::free(data.ir_via_vio_list);
    helper::free(data.ir_key_cost);
}
} // namespace gta::ops::kernel