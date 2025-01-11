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

    int findPRLSpacing(int l, int width, int prl) const;

    void init(int d_0);
    void assignInitial(int d_0);
    void assignRefinement(int d_0);
    void assign(int i, std::set<int> *S = nullptr);
    void apply(int i, int coef, std::set<int> *S = nullptr);

    // ir i, blk j
    void getBlkVio(int i, int j);

    int iter;
};
} // namespace gta