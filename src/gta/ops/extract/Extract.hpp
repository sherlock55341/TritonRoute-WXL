#pragma once

#include "frDesign.h"
#include <gta/database/Data.hpp>

namespace gta::ops {
void extract(fr::frTechObject *tech, fr::frDesign *design, data::Data &data);
}