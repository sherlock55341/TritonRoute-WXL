#pragma once

#include "database/Data.hpp"
#include "frDesign.h"
#include <memory>

namespace gta {
class GTA {
  public:
    GTA(fr::frDesign *in);
    ~GTA();

    void run(int maxIter, bool cuda = false);

  protected:
    fr::frTechObject *tech = nullptr;
    fr::frDesign *design = nullptr;
    data::Data data;
    data::Data data_device;

    void saveToGuide();
    int iter;
};
} // namespace gta