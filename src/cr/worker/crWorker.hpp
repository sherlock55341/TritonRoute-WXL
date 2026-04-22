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
struct crPatternPoint {
    frCoord x;
    frCoord y;

    bool operator<(const crPatternPoint& rhs) const {
        return (x < rhs.x) || (x == rhs.x && y < rhs.y);
    }
    bool operator==(const crPatternPoint& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
};

class CustomRouteWorker {
   public:
    CustomRouteWorker(
        CustomRoute* _cr,
        const std::vector<std::pair<frNet*, crPatternEnum>>& _tasks);
    frDesign* getDesign() const { return design; }
    const frBox& getRouteBox() const { return routeBox; }
    const frBox& getExtBox() const { return extBox; }
    crNet& getNet() { return *nets.front(); }
    const crNet& getNet() const { return *nets.front(); }
    std::vector<std::unique_ptr<crNet>>& getNets() { return nets; }
    const std::vector<std::unique_ptr<crNet>>& getNets() const { return nets; }
    crPatternGraph* getPatternGraph() { return patternGraph.get(); }
    const crPatternGraph* getPatternGraph() const { return patternGraph.get(); }
    void route();
    bool routeNet(crNet* cNet, crPatternEnum policy,
                  std::vector<crMazeType>& path) const;

   protected:
    void initNet(frNet* _net);
    void initNetTerm(crNet* cNet, frBlockObject* term);
    void initRouteBox();
    void initPatternGraph();
    void initTrackCoords(
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap);
    void initTrackCoordsPin(
        crNet* cNet,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap);
    void collectPatternGraphCoordsL(crNet* cNet, std::vector<frCoord>& xCoords,
                                    std::vector<frCoord>& yCoords) const;
    void initPatternGraphL(crNet* cNet, const std::vector<frCoord>& xCoords,
                           const std::vector<frCoord>& yCoords,
                           std::vector<crPatternPoint>& points) const;
    void clearRouteConnFigs(crNet* cNet);
    void writePathToNet(crNet* cNet, const std::vector<crMazeType>& path) const;
    void writePathSegToNet(crNet* cNet, const crMazeType& beginMazeIdx,
                           const crMazeType& endMazeIdx) const;
    void writeViaToNet(crNet* cNet, const crMazeType& beginMazeIdx,
                       const crMazeType& endMazeIdx) const;
    frViaDef* getViaDefForPath(crNet* cNet, const crMazeType& beginMazeIdx,
                               const crMazeType& endMazeIdx) const;
    frViaDef* getAccessPointViaDef(const crNet* cNet, const frPoint& origin,
                                   frLayerNum layerNum,
                                   frDirEnum dir) const;
    void endRemoveNets(
        const std::set<frNet*, frBlockObjectComp>& modifiedNets) const;
    void endRemoveNetsPathSeg(frPathSeg* pathSeg) const;
    void endRemoveNetsVia(frVia* via) const;
    void endRemoveNetsPatchWire(frPatchWire* patchWire) const;
    void endAddNets(const std::vector<crNet*>& routedNets) const;
    void endAddNetsPathSeg(crPathSeg* pathSeg) const;
    void endAddNetsVia(crVia* via) const;

    frDesign* design;
    std::vector<std::unique_ptr<crNet>> nets;
    CustomRoute* cr;
    std::vector<crPatternEnum> policies;
    frUInt4 DRCCost;
    frBox routeBox;
    frBox extBox;
    std::unique_ptr<crPatternGraph> patternGraph;
};
}  // namespace fr
