#pragma once

#include "AssignHelper.cuh"

namespace gta::ops::cuda {
void assign_refinement(data::Data &data, int iter, int d);
}