#pragma once

#include <memory>
#include <set>
#include <vector>
#include "cr/type/crAccessPoint.hpp"
#include "db/obj/frBlockObject.h"
#include "db/tech/frTechObject.h"
#include "frDesign.h"
#include "frRegionQuery.h"

namespace fr {

class frMarker;
class frNet;

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
    // Maximum number of route attempts for one CR task net during lightweight
    // marker-driven reroute.
    static constexpr int kMaxRouteAttempts = 3;

    // Run FlexGC only in CR-touched query boxes after writeback and publish
    // markers whose bbox overlaps those boxes.
    void runDRCChecks(const std::vector<frBox>& checkBoxes) const;
    // Run one box-scoped FlexGC pass. When publishMarkers is true, replace
    // top-level markers in checkBox; otherwise return marker copies only for
    // reroute queue decisions.
    std::vector<std::unique_ptr<frMarker>> runBoxDRC(const frBox& checkBox,
                                                     bool publishMarkers) const;
    // Remove existing top-level markers whose bboxes intersect checkBox.
    void removePublishedMarkersInBox(const frBox& checkBox) const;
    // Extract CR task nets appearing in marker source objects.
    std::set<frNet*, frBlockObjectComp> collectTaskNetsFromMarkers(
        const std::vector<std::unique_ptr<frMarker>>& markers,
        const std::set<frNet*, frBlockObjectComp>& taskNets) const;
    // Map a GC marker source owner object back to its owning frNet when it is a
    // net, instTerm, or top-level term.
    frNet* getMarkerSourceNet(frBlockObject* src) const;

    // Source design database; owned by the caller/router flow.
    frDesign* design;

    // Pending route requests. frNet pointers are design-owned; policies are
    // copied by value.
    std::vector<std::pair<frNet*, crPatternEnum>> routeTasks;

    // Lazily initialized CR-local AP point index.
    std::unique_ptr<crAPRegionQuery> apRegionQuery;
};
}  // namespace fr
