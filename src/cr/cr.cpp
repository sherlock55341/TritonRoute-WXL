#include "cr.hpp"
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include "worker/crWorker.hpp"
#include "db/obj/frAccess.h"
#include "db/obj/frBlock.h"
#include "db/obj/frInst.h"
#include "db/obj/frInstTerm.h"
#include "db/obj/frMarker.h"
#include "db/obj/frNet.h"
#include "db/obj/frPin.h"
#include "db/obj/frTerm.h"
#include "db/tech/frLayer.h"
#include "db/tech/frTechObject.h"
#include "gc/FlexGC.h"

namespace fr {

namespace {

box_t getBoostBox(const frBox& box) {
    return box_t(point_t(box.left(), box.bottom()),
                 point_t(box.right(), box.top()));
}

void setAccessPointOnTrack(crAccessPoint* cAp, const frAccessPoint* ap,
                           frTechObject* tech) {
    if (!cAp || !ap || !tech) {
        return;
    }

    // PA records AP quality separately for the AP layer and the layer above.
    // Map those PA types onto DR's onTrackX/onTrackY convention: horizontal
    // preferred layers consult the X flag, vertical layers consult the Y flag.
    auto setLayerOnTrack = [&](frLayerNum layerNum, frAccessPointEnum type) {
        auto* layer = tech->getLayer(layerNum);
        if (!layer) {
            return;
        }
        if (layer->getDir() == frcHorzPrefRoutingDir) {
            cAp->setOnTrack(type == frAccessPointEnum::frcOnGridAP, true);
        } else if (layer->getDir() == frcVertPrefRoutingDir) {
            cAp->setOnTrack(type == frAccessPointEnum::frcOnGridAP, false);
        }
    };

    auto layerNum = ap->getLayerNum();
    setLayerOnTrack(layerNum, ap->getType(true));
    if (layerNum + 2 <= tech->getTopLayerNum()) {
        setLayerOnTrack(layerNum + 2, ap->getType(false));
    }
}

void copyAccessPointToQuery(frNet* net, frBlockObject* term, frInst* inst,
                            frTerm* trueTerm, frTechObject* tech,
                            crAPRegionQuery* query) {
    if (!net || !term || !trueTerm || !tech || !query) {
        return;
    }

    // Flow:
    // 1. Use the same shift-only AP placement convention as DR/CR net init.
    // 2. Copy PA access flags and viaDefs into CR-local AP objects.
    // 3. Let crAPRegionQuery own and index each AP copy by point/layer.
    frTransform shiftXform;
    int pinAccessIdx = 0;
    if (inst) {
        inst->getTransform(shiftXform);
        shiftXform.set(frOrient(frcR0));
        pinAccessIdx = inst->getPinAccessIdx();
    }

    for (auto& pin : trueTerm->getPins()) {
        if (!pin->hasPinAccess() || pinAccessIdx < 0 ||
            pinAccessIdx >= pin->getNumPinAccess()) {
            continue;
        }

        for (auto& ap : pin->getPinAccess(pinAccessIdx)->getAccessPoints()) {
            frPoint pt;
            ap->getPoint(pt);
            pt.transform(shiftXform);

            auto cAp = std::make_unique<crAccessPoint>();
            cAp->setOwnerNet(net);
            cAp->setOwnerTerm(term);
            cAp->setPt(pt);
            cAp->setLayerIdx(ap->getLayerNum());
            setAccessPointOnTrack(cAp.get(), ap.get(), tech);

            for (auto dir : {frDirEnum::E, frDirEnum::S, frDirEnum::W,
                             frDirEnum::N, frDirEnum::U, frDirEnum::D}) {
                cAp->setValidAccess(dir, ap->hasAccess(dir));
            }

            if (ap->hasAccess(frDirEnum::U)) {
                for (int cutNum = 0; cutNum < 2; ++cutNum) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addUpViaDef(cutNum, viaDef);
                        }
                    }
                }
            }
            if (ap->hasAccess(frDirEnum::D)) {
                for (int cutNum = 0; cutNum < 2; ++cutNum) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addDownViaDef(cutNum, viaDef);
                        }
                    }
                }
            }

            query->addAccessPoint(cAp);
        }
    }
}

}  // namespace

void crAPRegionQuery::clear() {
    initialized = false;
    accessPoints.clear();
    aps.clear();
}

void crAPRegionQuery::addAccessPoint(std::unique_ptr<crAccessPoint>& ap) {
    if (!ap) {
        return;
    }

    auto layerNum = ap->getLayerIdx();
    if (layerNum < 0 || static_cast<std::size_t>(layerNum) >= aps.size()) {
        return;
    }

    auto* apPtr = ap.get();
    auto pt = apPtr->getPt();
    aps[layerNum].insert({getBoostBox(frBox(pt, pt)), apPtr});
    accessPoints.push_back(std::move(ap));
}

void crAPRegionQuery::init(frDesign* design) {
    clear();
    if (!design || !design->getTopBlock() || !design->getTech()) {
        initialized = true;
        return;
    }

    // Flow:
    // 1. Allocate one AP R-tree per technology layer number.
    // 2. Traverse design nets once and copy their PA access points.
    // 3. Store only point/layer spatial membership; AP cost interpretation is
    //    intentionally left to later consumers.
    aps.resize(
        static_cast<std::size_t>(design->getTech()->getTopLayerNum() + 1));
    for (auto& net : design->getTopBlock()->getNets()) {
        if (!net) {
            continue;
        }
        for (auto* instTerm : net->getInstTerms()) {
            if (!instTerm) {
                continue;
            }
            copyAccessPointToQuery(net.get(), instTerm, instTerm->getInst(),
                                   instTerm->getTerm(), design->getTech(),
                                   this);
        }
        for (auto* term : net->getTerms()) {
            copyAccessPointToQuery(net.get(), term, nullptr, term,
                                   design->getTech(), this);
        }
    }
    initialized = true;
}

void crAPRegionQuery::query(const frBox& box, frLayerNum layerNum,
                            std::vector<crAccessPoint*>& result) const {
    if (layerNum < 0 || static_cast<std::size_t>(layerNum) >= aps.size()) {
        return;
    }

    std::vector<std::pair<box_t, crAccessPoint*>> temp;
    aps[layerNum].query(bgi::intersects(getBoostBox(box)),
                        std::back_inserter(temp));
    for (auto& value : temp) {
        auto* ap = value.second;
        if (ap) {
            result.push_back(ap);
        }
    }
}

crAPRegionQuery* CustomRoute::getAPRegionQuery() {
    if (!apRegionQuery) {
        apRegionQuery = std::make_unique<crAPRegionQuery>();
    }
    if (!apRegionQuery->isInitialized()) {
        apRegionQuery->init(design);
    }
    return apRegionQuery.get();
}

frNet* CustomRoute::getMarkerSourceNet(frBlockObject* src) const {
    if (!src) {
        return nullptr;
    }

    if (src->typeId() == frcNet) {
        return static_cast<frNet*>(src);
    }
    if (src->typeId() == frcInstTerm) {
        return static_cast<frInstTerm*>(src)->getNet();
    }
    if (src->typeId() == frcTerm) {
        return static_cast<frTerm*>(src)->getNet();
    }
    return nullptr;
}

void CustomRoute::removePublishedMarkersInBox(const frBox& checkBox) const {
    if (!design || !design->getTopBlock() || !design->getRegionQuery()) {
        return;
    }

    auto* topBlock = design->getTopBlock();
    auto* regionQuery = design->getRegionQuery();
    std::vector<frMarker*> oldMarkers;
    regionQuery->queryMarker(checkBox, oldMarkers);
    for (auto* marker : oldMarkers) {
        regionQuery->removeMarker(marker);
        topBlock->removeMarker(marker);
    }
}

std::vector<std::unique_ptr<frMarker>> CustomRoute::runBoxDRC(
    const frBox& checkBox, bool publishMarkers) const {
    std::vector<std::unique_ptr<frMarker>> markers;
    if (!design || !design->getTopBlock() || !design->getRegionQuery()) {
        return markers;
    }

    if (publishMarkers) {
        removePublishedMarkersInBox(checkBox);
    }

    FlexGCWorker gcWorker(design);
    gcWorker.setExtBox(checkBox);
    gcWorker.setDrcBox(checkBox);
    gcWorker.init();
    gcWorker.main();

    auto* topBlock = design->getTopBlock();
    auto* regionQuery = design->getRegionQuery();
    for (const auto& marker : gcWorker.getMarkers()) {
        frBox markerBox;
        marker->getBBox(markerBox);
        if (!checkBox.overlaps(markerBox)) {
            continue;
        }

        if (publishMarkers) {
            auto publishedMarker = std::make_unique<frMarker>(*marker);
            auto* markerPtr = publishedMarker.get();
            regionQuery->addMarker(markerPtr);
            topBlock->addMarker(publishedMarker);
        }

        markers.push_back(std::make_unique<frMarker>(*marker));
    }
    return markers;
}

std::set<frNet*, frBlockObjectComp> CustomRoute::collectTaskNetsFromMarkers(
    const std::vector<std::unique_ptr<frMarker>>& markers,
    const std::set<frNet*, frBlockObjectComp>& taskNets) const {
    std::set<frNet*, frBlockObjectComp> result;
    for (const auto& marker : markers) {
        for (auto* src : marker->getSrcs()) {
            auto* net = getMarkerSourceNet(src);
            if (net && taskNets.find(net) != taskNets.end()) {
                result.insert(net);
            }
        }
    }
    return result;
}

void CustomRoute::runDRCChecks(const std::vector<frBox>& checkBoxes) const {
    if (!design || !design->getTopBlock() || !design->getRegionQuery()) {
        return;
    }

    // Flow:
    // 1. For each CR-touched box, remove stale top-level markers in that box.
    // 2. Run an independent FlexGC pass with extBox/drcBox limited to the box.
    // 3. Publish only markers whose bbox overlaps the checked box.
    for (const auto& checkBox : checkBoxes) {
        auto markers = runBoxDRC(checkBox, true);
        std::cout << "CR box DRC violations = " << markers.size() << " box=("
                  << checkBox.left() << ", " << checkBox.bottom() << ") - ("
                  << checkBox.right() << ", " << checkBox.top() << ")"
                  << std::endl;
    }
}

void CustomRoute::run() {
    // Flow:
    // 1. Skip CR entirely when no net was requested, leaving existing markers
    //    untouched.
    // 2. Put unique task nets into a lightweight reroute queue.
    // 3. Pop one net at a time, route it with a single-net worker, then run a
    //    non-publishing box DRC over that worker's extBox.
    // 4. If the DRC marker sources include other CR task nets, push those nets
    //    back into the queue until their per-net attempt limit is reached.
    // 5. Run publishing box-scoped DRC passes after queue convergence.
    if (routeTasks.empty()) {
        return;
    }

    std::map<frNet*, crPatternEnum, frBlockObjectComp> policies;
    std::set<frNet*, frBlockObjectComp> taskNets;

    struct RerouteQueue {
        RerouteQueue(const std::map<frNet*, crPatternEnum, frBlockObjectComp>&
                         policiesIn,
                     int maxAttemptsIn)
            : policies(policiesIn),
              routeAttempts(),
              queuedNets(),
              nets(),
              maxAttempts(maxAttemptsIn) {}

        bool enqueue(frNet* net, bool pushFront) {
            if (!net || policies.find(net) == policies.end()) {
                return false;
            }
            if (queuedNets.find(net) != queuedNets.end()) {
                return false;
            }
            if (routeAttempts[net] >= maxAttempts) {
                std::cout << "[customdr] skip reroute net " << net->getName()
                          << " attempts=" << routeAttempts[net] << std::endl;
                return false;
            }

            if (pushFront) {
                nets.push_front(net);
            } else {
                nets.push_back(net);
            }
            queuedNets.insert(net);
            return true;
        }

        bool empty() const { return nets.empty(); }

        frNet* pop() {
            auto* net = nets.front();
            nets.pop_front();
            queuedNets.erase(net);
            return net;
        }

        int startAttempt(frNet* net) { return ++routeAttempts[net]; }

        const std::map<frNet*, crPatternEnum, frBlockObjectComp>& policies;
        std::map<frNet*, int, frBlockObjectComp> routeAttempts;
        std::set<frNet*, frBlockObjectComp> queuedNets;
        std::deque<frNet*> nets;
        int maxAttempts;
    };
    RerouteQueue rerouteQueue(policies, kMaxRouteAttempts);

    for (const auto& [net, policy] : routeTasks) {
        if (!net) {
            continue;
        }
        policies[net] = policy;
        taskNets.insert(net);
        rerouteQueue.enqueue(net, false);
    }

    std::set<frBox> checkBoxes;
    while (!rerouteQueue.empty()) {
        auto* net = rerouteQueue.pop();
        if (!net || policies.find(net) == policies.end()) {
            continue;
        }

        auto attempt = rerouteQueue.startAttempt(net);
        std::cout << "[customdr] route net " << net->getName()
                  << " attempt=" << attempt << std::endl;

        std::vector<std::pair<frNet*, crPatternEnum>> workerTasks{
            {net, policies[net]}};
        CustomRouteWorker worker(this, workerTasks);
        checkBoxes.insert(worker.getExtBox());
        auto graphNodeCount = worker.getPatternGraph()->getXCoords().size() *
                              worker.getPatternGraph()->getYCoords().size() *
                              worker.getPatternGraph()->getZCoords().size();
        std::cout << "[customdr] graph nodes=" << graphNodeCount << std::endl;
        worker.route();

        auto markers = runBoxDRC(worker.getExtBox(), false);
        std::cout << "[customdr] incremental DRC net " << net->getName()
                  << " violations=" << markers.size() << std::endl;
        auto conflictNets = collectTaskNetsFromMarkers(markers, taskNets);
        for (auto* conflictNet : conflictNets) {
            if (conflictNet == net) {
                continue;
            }
            if (rerouteQueue.enqueue(conflictNet, true)) {
                std::cout << "[customdr] enqueue reroute net "
                          << conflictNet->getName() << " due to marker near "
                          << net->getName() << std::endl;
            }
        }
    }
    runDRCChecks(std::vector<frBox>(checkBoxes.begin(), checkBoxes.end()));
}
}  // namespace fr
