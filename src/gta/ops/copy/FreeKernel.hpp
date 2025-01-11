#pragma once

#include <cassert>
#include <cstdlib>
#include <gta/database/Data.hpp>

namespace gta::ops::cpu {
void free_data(data::Data &data);
}

namespace gta::ops::cuda {
void free_data(data::Data &data);
}