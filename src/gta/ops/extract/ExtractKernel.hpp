#pragma once

#include "frDesign.h"
#include <gta/database/Data.hpp>

namespace gta::ops::cpu {
void extract(fr::frTechObject *tech, fr::frDesign *design, data::Data &data);
}