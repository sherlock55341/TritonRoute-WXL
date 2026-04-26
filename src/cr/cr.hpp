#pragma once

#include <memory>
#include <vector>
#include "cr/type/crAccessPoint.hpp"
#include "db/tech/frTechObject.h"
#include "frDesign.h"
#include "frRegionQuery.h"

namespace fr {

// Supported custom-route pattern search policies. `L` means enumerate
// single-bend rectilinear candidates plus endpoint via transitions.
enum class crPatternEnum { L };

// CR-local spatial index for access points. It owns copied crAccessPoint
// objects and indexes their transformed point on each routing layer.
class crAPRegionQuery {
   public:
    crAPRegionQuery() : initialized(false), accessPoints(), aps() {}

    // Build AP copies and point R-trees from the current design PA data.
    void init(frDesign* design);
    // Clear all AP copies and R-tree state.
    void clear();
    // Return whether the query has already been initialized.
    bool isInitialized() const { return initialized; }
    // Query AP points on layerNum whose point box intersects box.
    void query(const frBox& box, frLayerNum layerNum,
               std::vector<crAccessPoint*>& result) const;
    // Add one owned AP copy and index its point on the AP layer.
    void addAccessPoint(std::unique_ptr<crAccessPoint>& ap);

   protected:
    // True after init has completed, even if the design has no APs.
    bool initialized;
    // Owned CR-local AP copies; R-tree values are non-owning pointers here.
    std::vector<std::unique_ptr<crAccessPoint>> accessPoints;
    // Per-layer point R-trees keyed by AP point boxes.
    std::vector<
        bgi::rtree<std::pair<box_t, crAccessPoint*>, bgi::quadratic<16>>>
        aps;
};

// Top-level custom-route driver. It owns the route task list and creates
// workers that route selected frNets back into the design database.
class CustomRoute {
   public:
    CustomRoute(frDesign* _design) : design(_design), apRegionQuery() {}
    // Return the technology object from the source design.
    frTechObject* getTech() const { return design->getTech(); }
    // Return the source design database; CustomRoute does not own it.
    frDesign* getDesign() const { return design; }
    // Return the design region query used for writeback synchronization.
    frRegionQuery* getRegionQuery() const { return design->getRegionQuery(); }
    // Lazily build and return CR's AP point spatial index.
    crAPRegionQuery* getAPRegionQuery();

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

    // Lazily initialized CR-local AP point index.
    std::unique_ptr<crAPRegionQuery> apRegionQuery;
};
}  // namespace fr
