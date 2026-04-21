#pragma once

#include <cr/cr.hpp>
#include <cr/graph/crPatternGraph.hpp>
#include <cr/type/crNet.hpp>
#include <map>
#include "frBaseTypes.h"
#include "db/infra/frBox.h"
#include "db/obj/frTrackPattern.h"

namespace fr {
class CustomRouteWorker {
   public:
    CustomRouteWorker(
        CustomRoute* _cr,
        const std::vector<std::pair<frNet*, crPatternEnum>>& _tasks);
    const frBox& getRouteBox() const { return routeBox; }
    const frBox& getExtBox() const { return extBox; }
    crNet& getNet() { return *nets.front(); }
    const crNet& getNet() const { return *nets.front(); }
    std::vector<std::unique_ptr<crNet>>& getNets() { return nets; }
    const std::vector<std::unique_ptr<crNet>>& getNets() const { return nets; }
    crPatternGraph* getPatternGraph() { return patternGraph.get(); }
    const crPatternGraph* getPatternGraph() const { return patternGraph.get(); }

   protected:
    void initNet(frNet* _net);
    void initNetTerm(crNet* cNet, frBlockObject* term);
    void initRouteBox();
    void initTrackCoords(
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap);
    void initTrackCoordsPin(
        crNet* cNet,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
        std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap);

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
