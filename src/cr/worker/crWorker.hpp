#pragma once

#include <cr/cr.hpp>
#include <cr/graph/crPatternGraph.hpp>
#include <cr/type/crNet.hpp>
#include "cr/type/crVia.hpp"
#include "db/obj/frNet.h"
#include "db/obj/frShape.h"
#include "db/obj/frVia.h"
#include <map>
#include <set>
#include "frBaseTypes.h"
#include "db/infra/frBox.h"
#include "db/obj/frTrackPattern.h"

namespace fr {

// Physical point used while constructing pattern graph candidate coordinates.
struct crPatternPoint {
    // Physical x coordinate.
    frCoord x;
    // Physical y coordinate.
    frCoord y;

    // Order points lexicographically for deterministic set/map use.
    bool operator<(const crPatternPoint& rhs) const {
        return (x < rhs.x) || (x == rhs.x && y < rhs.y);
    }
    // Compare exact physical coordinates.
    bool operator==(const crPatternPoint& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
};

// Worker that routes a batch of selected frNets through CR-local objects and
// writes the final result back into frNet/frRegionQuery.
class CustomRouteWorker {
   public:
    CustomRouteWorker(
        CustomRoute* _cr,
        const std::vector<std::pair<frNet*, crPatternEnum>>& _tasks);
    // Return the source design database.
    frDesign* getDesign() const { return design; }
    // Return the route window derived from all selected net access points.
    const frBox& getRouteBox() const { return routeBox; }
    // Return the expanded query window used for existing-routing quick cost.
    const frBox& getExtBox() const { return extBox; }
    // Return the first CR net; valid only for single-net legacy call sites.
    crNet& getNet() { return *nets.front(); }
    // Return the first CR net; valid only for single-net legacy call sites.
    const crNet& getNet() const { return *nets.front(); }
    // Return owned CR nets for the current task batch.
    std::vector<std::unique_ptr<crNet>>& getNets() { return nets; }
    // Return owned CR nets for read-only iteration.
    const std::vector<std::unique_ptr<crNet>>& getNets() const { return nets; }
    // Return the pattern graph owned by this worker.
    crPatternGraph* getPatternGraph() { return patternGraph.get(); }
    // Return the pattern graph owned by this worker for read-only access.
    const crPatternGraph* getPatternGraph() const { return patternGraph.get(); }
    // Return the parent CR driver, which owns shared CR-local caches.
    CustomRoute* getCustomRoute() const { return cr; }
    // Route all CR nets and write successful results back to the design DB.
    void route();
    // Route one CR net with a policy and return the selected maze path.
    bool routeNet(crNet* cNet, crPatternEnum policy,
                  std::vector<crMazeType>& path) const;

   protected:
    // Build a CR-local crNet from one source frNet.
    void initNet(frNet* _net);
    // Build one crPin and its access points from an frTerm/frInstTerm.
    void initNetTerm(crNet* cNet, frBlockObject* term);
    // Compute routeBox/extBox from all selected access-point coordinates.
    void initRouteBox();
    // Build pattern graph coordinate arrays and SVia AP annotations.
    void initPatternGraph();
    // Add track-pattern coordinates inside routeBox/extBox to x/y maps.
    void initTrackCoords(
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*>>& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*>>& yMap);
    // Add pin access coordinates and adjacent-layer non-preferred coordinates
    // to x/y track maps.
    void initTrackCoordsPin(
        crNet* cNet,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*>>& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*>>& yMap);
    // Collect physical x/y coordinates needed by L-pattern candidates.
    void collectPatternGraphCoordsL(crNet* cNet, std::vector<frCoord>& xCoords,
                                    std::vector<frCoord>& yCoords) const;
    // Populate candidate physical points for L-pattern route construction.
    void initPatternGraphL(crNet* cNet, const std::vector<frCoord>& xCoords,
                           const std::vector<frCoord>& yCoords,
                           std::vector<crPatternPoint>& points) const;
    // Remove existing local route figures and subtract their graph cost.
    void clearRouteConnFigs(crNet* cNet);
    // Convert a maze path into local crPathSeg/crVia route figures.
    void writePathToNet(crNet* cNet, const std::vector<crMazeType>& path) const;
    // Create one local crPathSeg from two same-layer maze indices.
    void writePathSegToNet(crNet* cNet, const crMazeType& beginMazeIdx,
                           const crMazeType& endMazeIdx) const;
    // Create one local crVia from two same-x/y different-layer maze indices.
    void writeViaToNet(crNet* cNet, const crMazeType& beginMazeIdx,
                       const crMazeType& endMazeIdx) const;
    // Choose the viaDef for a path transition, preferring SVia/AP-specific vias
    // over the cut layer default via.
    frViaDef* getViaDefForPath(crNet* cNet, const crMazeType& beginMazeIdx,
                               const crMazeType& endMazeIdx) const;
    // Find an access-point viaDef at a physical origin/layer/direction.
    frViaDef* getAccessPointViaDef(const crNet* cNet, const frPoint& origin,
                                   frLayerNum layerNum, frDirEnum dir) const;
    // Remove old DB route objects for modified nets inside routeBox.
    void endRemoveNets(
        const std::set<frNet*, frBlockObjectComp>& modifiedNets) const;
    // Remove one DB path segment from region query and its frNet.
    void endRemoveNetsPathSeg(frPathSeg* pathSeg) const;
    // Remove one DB via from region query and its frNet.
    void endRemoveNetsVia(frVia* via) const;
    // Remove one DB patch wire from region query and its frNet.
    void endRemoveNetsPatchWire(frPatchWire* patchWire) const;
    // Add local CR route figures for routed nets into the design DB.
    void endAddNets(const std::vector<crNet*>& routedNets) const;
    // Convert one local crPathSeg to frPathSeg and insert it into DB/RQ.
    void endAddNetsPathSeg(crPathSeg* pathSeg) const;
    // Convert one local crVia to frVia and insert it into DB/RQ.
    void endAddNetsVia(crVia* via) const;

    // Source design database; owned outside this worker.
    frDesign* design;
    // CR-local nets owned by this worker.
    std::vector<std::unique_ptr<crNet>> nets;
    // Parent CustomRoute driver; non-owning.
    CustomRoute* cr;
    // Routing policy for each entry in nets.
    std::vector<crPatternEnum> policies;
    // Current DRC cost scalar copied from global DRCCOST for DR-like behavior.
    frUInt4 DRCCost;
    // Routing search box derived from access points plus margin.
    frBox routeBox;
    // Expanded box for region-query existing routing and quick-cost setup.
    frBox extBox;
    // Pattern graph owned by this worker.
    std::unique_ptr<crPatternGraph> patternGraph;
};
}  // namespace fr
