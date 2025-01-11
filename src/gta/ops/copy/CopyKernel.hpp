#pragma once

#include <gta/database/Data.hpp>

namespace gta::ops::cpu {
void malloc_data(data::Data &data_h, data::Data &data_d);
void h2d_data(data::Data &data_h, data::Data &data_d);
void d2h_data(data::Data &data_h, data::Data &data_d);
} // namespace gta::ops::cpu