#pragma once

#include "GTAData.hpp"
#include "frDesign.h"
#include <memory>

namespace gta {
class GTA {
  public:
    GTA(fr::frDesign *in);
    ~GTA();

    void run(int maxIter);

  protected:
    fr::frTechObject *tech = nullptr;
    fr::frDesign *design = nullptr;
    GTAData data;

    void saveToGuide();

    void extractGTADataFromDatabase();
    void extractTechDesignBasicInfo();
    void extractIrInfo();
    void extractBlkInfo();
    void extractGCellInfo();
    void extractCostInfo();
    bool findAp(fr::frGuide *g, int g_idx);
    void findProjAp(fr::frGuide *g, int g_idx);
    int findPRLSpacing(int l, int width, int prl) const;

    void init(int iter, int d_0);
    void assignInitial(int d_0);
    void assignRefinement(int iter, int d_0);
    void assign(int iter, int i, std::set<int> *S = nullptr);
    void apply(int i, int coef, std::set<int> *S = nullptr);

    // ir i, blk j
    void getBlkVio(int i, int j);
};
} // namespace gta