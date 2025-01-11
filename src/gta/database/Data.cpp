#include "Data.hpp"

namespace gta::data::helper {
int findPRLSpacing(Data &data, int l, int width, int prl) {
    int row = 0, col = 0;
    const auto row_num = data.layer_spacing_table_width_start[l + 1] -
                         data.layer_spacing_table_width_start[l];
    const auto col_num = data.layer_spacing_table_prl_start[l + 1] -
                         data.layer_spacing_table_prl_start[l];
    while (row + 1 < row_num &&
           width > data.layer_spacing_table_width
                       [data.layer_spacing_table_width_start[l] + row + 1])
        row++;
    while (
        col + 1 < col_num &&
        prl >
            data.layer_spacing_table_prl[data.layer_spacing_table_prl_start[l] +
                                         col + 1])
        col++;
    return data
        .layer_spacing_table_spacing[data.layer_spacing_table_spacing_start[l] +
                                     row * col_num + col];
}
} // namespace gta::data::helper