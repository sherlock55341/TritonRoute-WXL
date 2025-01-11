#pragma once

#include "AssignHelper.cuh"

namespace gta::ops::cuda {
void assign_initial(data::Data &data, int iter, int d);
}