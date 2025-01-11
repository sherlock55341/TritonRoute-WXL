#pragma once

#include <gta/database/Data.hpp>

namespace gta::ops::cpu {
void assign(data::Data &data, int iter, int d);
}

namespace gta::ops::cuda {
void assign(data::Data &data, int iter, int d);
}