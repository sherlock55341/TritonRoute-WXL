#include "Extract.hpp"
#include "ExtractKernel.hpp"

namespace gta::ops {
void extract(fr::frTechObject *tech, fr::frDesign *design, data::Data &data) {
    cpu::extract(tech, design, data);
}
} // namespace gta::ops