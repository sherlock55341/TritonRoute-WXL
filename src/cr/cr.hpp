#pragma once

#include "db/tech/frTechObject.h"
#include "frDesign.h"
#include "frRegionQuery.h"

namespace fr {
enum class crPatternEnum { L };
class CustomRoute {
   public:
    CustomRoute(frDesign* _design) : design(_design) {}
    frTechObject* getTech() const { return design->getTech(); }
    frDesign* getDesign() const { return design; }
    frRegionQuery* getRegionQuery() const { return design->getRegionQuery(); }

    void addTask(const std::pair<frNet*, crPatternEnum>& _task) {
        routeTasks.push_back(_task);
    }

    void run();

   protected:
    frDesign* design;

    std::vector<std::pair<frNet*, crPatternEnum>> routeTasks;
};
}  // namespace fr