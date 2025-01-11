#pragma once

#include <gta/database/Data.hpp>

namespace gta::ops::cpu {
void init(data::Data &data, int iter, int d);
}

namespace gta::ops::cuda {
void init(data::Data &data, int iter, int d);
}