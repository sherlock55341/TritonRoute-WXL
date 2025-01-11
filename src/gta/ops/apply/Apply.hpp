#pragma once

#include <gta/database/Data.hpp>
#include <set>

namespace gta::ops{
void apply(data::Data &data, int iter, int i, int coef, std::set<int> *S = nullptr);
}