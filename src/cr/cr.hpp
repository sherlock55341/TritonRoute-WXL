#pragma once

#include "db/tech/frTechObject.h"
#include "frDesign.h"
#include "frRegionQuery.h"

namespace fr {

// Supported custom-route pattern search policies. `L` means enumerate
// single-bend rectilinear candidates plus endpoint via transitions.
enum class crPatternEnum { L };

// Top-level custom-route driver. It owns the route task list and creates
// workers that route selected frNets back into the design database.
class CustomRoute {
   public:
    CustomRoute(frDesign* _design) : design(_design) {}
    // Return the technology object from the source design.
    frTechObject* getTech() const { return design->getTech(); }
    // Return the source design database; CustomRoute does not own it.
    frDesign* getDesign() const { return design; }
    // Return the design region query used for writeback synchronization.
    frRegionQuery* getRegionQuery() const { return design->getRegionQuery(); }

    // Add one frNet plus routing policy to the pending worker task list.
    void addTask(const std::pair<frNet*, crPatternEnum>& _task) {
        routeTasks.push_back(_task);
    }

    // Execute all pending route tasks by constructing a CustomRouteWorker.
    void run();

   protected:
    // Source design database; owned by the caller/router flow.
    frDesign* design;

    // Pending route requests. frNet pointers are design-owned; policies are
    // copied by value.
    std::vector<std::pair<frNet*, crPatternEnum>> routeTasks;
};
}  // namespace fr
