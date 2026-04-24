#include "crWorker.hpp"
#include "cr/type/crPathSeg.hpp"
#include "cr/type/crVia.hpp"
#include "cr/route/crPatternRouter.hpp"
#include "db/obj/frNet.h"
#include "db/obj/frShape.h"
#include "db/obj/frVia.h"

namespace fr {
CustomRouteWorker::CustomRouteWorker(
    CustomRoute* _cr,
    const std::vector<std::pair<frNet*, crPatternEnum>>& _tasks)
    : design(_cr->getDesign()),
      nets(),
      cr(_cr),
      policies(),
      DRCCost(DRCCOST),
      patternGraph(std::make_unique<crPatternGraph>()) {
    for (auto& [net, policy] : _tasks) {
        initNet(net);
        policies.push_back(policy);
    }
    initRouteBox();
    patternGraph->build(this);
    initPatternGraph();
    patternGraph->initDRCCost();
}

void CustomRouteWorker::route() {
    // Flow:
    // 1. Route each CR net independently and keep only successful nets.
    // 2. Collect source frNets whose DB routing must be replaced.
    // 3. Remove old DB route objects in routeBox, then write local CR results
    //    back into frNet/frRegionQuery.
    std::vector<crNet*> routedNets;
    std::set<frNet*, frBlockObjectComp> modifiedNets;
    for (std::size_t i = 0; i < nets.size() && i < policies.size(); ++i) {
        std::vector<crMazeType> path;
        if (routeNet(nets[i].get(), policies[i], path)) {
            routedNets.push_back(nets[i].get());
            if (nets[i]->getNet()) {
                modifiedNets.insert(nets[i]->getNet());
            }
        }
    }

    if (!routedNets.empty()) {
        endRemoveNets(modifiedNets);
        endAddNets(routedNets);
    }
}

bool CustomRouteWorker::routeNet(crNet* cNet, crPatternEnum policy,
                                 std::vector<crMazeType>& path) const {
    // Flow:
    // 1. Remove local route figures and subtract their graph quick cost.
    // 2. Search a policy-specific pattern route on the shared graph.
    // 3. Convert the winning maze path into local CR route figures.
    const_cast<CustomRouteWorker*>(this)->clearRouteConnFigs(cNet);

    crPatternRouter router(patternGraph.get(), cNet, policy);
    if (!router.searchPath()) {
        path.clear();
        std::cout << "CANNOT FIND" << std::endl;
        return false;
    }
    path = router.getPath();
    const_cast<CustomRouteWorker*>(this)->writePathToNet(cNet, path);
    return true;
}

void CustomRouteWorker::clearRouteConnFigs(crNet* cNet) {
    if (!cNet || !patternGraph) {
        return;
    }

    // Subtract quick cost before destroying the local route figures that
    // identify which graph edges/vias were occupied.
    for (auto& connFig : cNet->getRouteConnFigs()) {
        patternGraph->subPathCost(connFig.get());
    }
    cNet->clearRouteConnFigs();
}

void CustomRouteWorker::writePathToNet(
    crNet* cNet, const std::vector<crMazeType>& path) const {
    if (!cNet || path.size() < 2 || !patternGraph) {
        return;
    }

    // Flow:
    // 1. Walk the maze path once and merge consecutive planar steps with the
    //    same direction into one crPathSeg.
    // 2. Emit each vertical same-x/y transition as one crVia.
    // 3. Add graph quick cost for the newly created local route figures.
    auto getPlanarDir = [](const crMazeType& lhs, const crMazeType& rhs) {
        if (lhs.z != rhs.z) {
            return frDirEnum::UNKNOWN;
        }
        if (lhs.x < rhs.x && lhs.y == rhs.y) {
            return frDirEnum::E;
        }
        if (lhs.x > rhs.x && lhs.y == rhs.y) {
            return frDirEnum::W;
        }
        if (lhs.y < rhs.y && lhs.x == rhs.x) {
            return frDirEnum::N;
        }
        if (lhs.y > rhs.y && lhs.x == rhs.x) {
            return frDirEnum::S;
        }
        return frDirEnum::UNKNOWN;
    };

    std::size_t segBeginIdx = 0;
    frDirEnum segDir = getPlanarDir(path[0], path[1]);
    if (segDir == frDirEnum::UNKNOWN && path[0].x == path[1].x &&
        path[0].y == path[1].y && path[0].z != path[1].z) {
        writeViaToNet(cNet, path[0], path[1]);
    }
    for (std::size_t i = 2; i < path.size(); ++i) {
        auto currDir = getPlanarDir(path[i - 1], path[i]);
        if (currDir == segDir && currDir != frDirEnum::UNKNOWN) {
            continue;
        }
        if (segDir != frDirEnum::UNKNOWN) {
            writePathSegToNet(cNet, path[segBeginIdx], path[i - 1]);
        }
        if (path[i - 1].x == path[i].x && path[i - 1].y == path[i].y &&
            path[i - 1].z != path[i].z) {
            writeViaToNet(cNet, path[i - 1], path[i]);
        }
        segBeginIdx = i - 1;
        segDir = currDir;
    }

    if (segDir != frDirEnum::UNKNOWN) {
        writePathSegToNet(cNet, path[segBeginIdx], path.back());
    }

    for (auto& connFig : cNet->getRouteConnFigs()) {
        patternGraph->addPathCost(connFig.get());
    }
}

void CustomRouteWorker::writePathSegToNet(crNet* cNet,
                                          const crMazeType& beginMazeIdx,
                                          const crMazeType& endMazeIdx) const {
    // Convert a same-layer maze run into a local path segment. The local object
    // is not inserted into frNet until endAddNets, so failed later routing can
    // still be discarded without touching the design DB.
    if (!cNet || !patternGraph || beginMazeIdx == endMazeIdx ||
        beginMazeIdx.z != endMazeIdx.z) {
        return;
    }

    auto begin = patternGraph->getPoint(beginMazeIdx);
    auto end = patternGraph->getPoint(endMazeIdx);
    if (begin.x() != end.x() && begin.y() != end.y()) {
        return;
    }

    auto layerNum = patternGraph->getLayerNum(beginMazeIdx);
    auto layer = design->getTech()->getLayer(layerNum);
    if (!layer) {
        return;
    }

    auto pathSeg = std::make_unique<crPathSeg>();
    frSegStyle style;
    style.setWidth(layer->getWidth());
    style.setBeginStyle(frEndStyle(frcTruncateEndStyle), 0);
    style.setEndStyle(frEndStyle(frcTruncateEndStyle), 0);
    pathSeg->setBegin(begin);
    pathSeg->setEnd(end);
    pathSeg->setLayerNum(layerNum);
    pathSeg->setStyle(style);
    pathSeg->setBeginMazeIdx(beginMazeIdx);
    pathSeg->setEndMazeIdx(endMazeIdx);

    std::unique_ptr<crConnFig> connFig = std::move(pathSeg);
    cNet->addRouteConnFig(connFig);
}

void CustomRouteWorker::writeViaToNet(crNet* cNet,
                                      const crMazeType& beginMazeIdx,
                                      const crMazeType& endMazeIdx) const {
    // Convert one vertical maze transition into a local via. The selected
    // viaDef is resolved before local route ownership is transferred to crNet.
    if (!cNet || !patternGraph || beginMazeIdx == endMazeIdx ||
        beginMazeIdx.x != endMazeIdx.x || beginMazeIdx.y != endMazeIdx.y ||
        beginMazeIdx.z == endMazeIdx.z) {
        return;
    }

    auto viaDef = getViaDefForPath(cNet, beginMazeIdx, endMazeIdx);
    if (!viaDef) {
        return;
    }

    auto origin = patternGraph->getPoint(beginMazeIdx);
    auto via = std::make_unique<crVia>(viaDef);
    via->setOrigin(origin);
    via->setBeginMazeIdx(beginMazeIdx);
    via->setEndMazeIdx(endMazeIdx);
    via->setOwner(cNet);

    std::unique_ptr<crConnFig> connFig = std::move(via);
    cNet->addRouteConnFig(connFig);
}

frViaDef* CustomRouteWorker::getViaDefForPath(
    crNet* cNet, const crMazeType& beginMazeIdx,
    const crMazeType& endMazeIdx) const {
    // Flow:
    // 1. Prefer graph-level SVia annotation produced from pin access.
    // 2. Fall back to an access-point viaDef at the same origin/layer/dir.
    // 3. Fall back to the cut layer's default viaDef.
    if (!patternGraph || !design || !cNet) {
        return nullptr;
    }

    auto beginLayerNum = patternGraph->getLayerNum(beginMazeIdx);
    auto endLayerNum = patternGraph->getLayerNum(endMazeIdx);
    auto origin = patternGraph->getPoint(beginMazeIdx);
    auto dir = (endLayerNum > beginLayerNum) ? frDirEnum::U : frDirEnum::D;
    auto lowerMazeIdx = (dir == frDirEnum::U) ? beginMazeIdx : endMazeIdx;
    if (auto sViaDef = patternGraph->getSViaDef(lowerMazeIdx)) {
        return sViaDef;
    }

    auto accessViaDef = getAccessPointViaDef(cNet, origin, beginLayerNum, dir);
    if (accessViaDef) {
        return accessViaDef;
    }

    auto lowerLayerNum = std::min(beginLayerNum, endLayerNum);
    auto cutLayer = design->getTech()->getLayer(lowerLayerNum + 1);
    return cutLayer ? cutLayer->getDefaultViaDef() : nullptr;
}

frViaDef* CustomRouteWorker::getAccessPointViaDef(const crNet* cNet,
                                                  const frPoint& origin,
                                                  frLayerNum layerNum,
                                                  frDirEnum dir) const {
    // Scan copied access points for an exact physical origin/layer match and
    // return the first one-cut viaDef in the requested vertical direction.
    if (!cNet) {
        return nullptr;
    }

    for (const auto& pin : cNet->getPins()) {
        for (const auto& ap : pin->getAccessPoints()) {
            if (ap->getLayerIdx() != layerNum || ap->getPt() != origin) {
                continue;
            }

            if (dir == frDirEnum::U && ap->hasUpAccessViaDef()) {
                const auto& viaDefs = ap->getUpViaDefs()[0];
                if (!viaDefs.empty()) {
                    return viaDefs.front();
                }
            }
            if (dir == frDirEnum::D && ap->hasDownAccessViaDef()) {
                const auto& viaDefs = ap->getDownViaDefs()[0];
                if (!viaDefs.empty()) {
                    return viaDefs.front();
                }
            }
        }
    }

    return nullptr;
}

void CustomRouteWorker::endRemoveNets(
    const std::set<frNet*, frBlockObjectComp>& modifiedNets) const {
    if (!design || modifiedNets.empty()) {
        return;
    }

    // Flow:
    // 1. Query old DB route objects inside routeBox.
    // 2. Filter objects to modified source frNets only.
    // 3. Remove each object from both region query and frNet ownership.
    std::vector<frBlockObject*> result;
    design->getRegionQuery()->queryDRObj(routeBox, result);
    for (auto* obj : result) {
        if (!obj) {
            continue;
        }

        if (obj->typeId() == frcPathSeg) {
            auto* pathSeg = static_cast<frPathSeg*>(obj);
            if (pathSeg->hasNet() &&
                modifiedNets.find(pathSeg->getNet()) != modifiedNets.end()) {
                endRemoveNetsPathSeg(pathSeg);
            }
        } else if (obj->typeId() == frcVia) {
            auto* via = static_cast<frVia*>(obj);
            if (via->hasNet() &&
                modifiedNets.find(via->getNet()) != modifiedNets.end()) {
                endRemoveNetsVia(via);
            }
        } else if (obj->typeId() == frcPatchWire) {
            auto* patchWire = static_cast<frPatchWire*>(obj);
            if (patchWire->hasNet() &&
                modifiedNets.find(patchWire->getNet()) != modifiedNets.end()) {
                endRemoveNetsPatchWire(patchWire);
            }
        }
    }
}

void CustomRouteWorker::endRemoveNetsPathSeg(frPathSeg* pathSeg) const {
    if (!pathSeg || !design || !pathSeg->hasNet()) {
        return;
    }

    design->getRegionQuery()->removeDRObj(pathSeg);
    pathSeg->getNet()->removeShape(pathSeg);
}

void CustomRouteWorker::endRemoveNetsVia(frVia* via) const {
    if (!via || !design || !via->hasNet()) {
        return;
    }

    design->getRegionQuery()->removeDRObj(via);
    via->getNet()->removeVia(via);
}

void CustomRouteWorker::endRemoveNetsPatchWire(frPatchWire* patchWire) const {
    if (!patchWire || !design || !patchWire->hasNet()) {
        return;
    }

    design->getRegionQuery()->removeDRObj(patchWire);
    patchWire->getNet()->removePatchWire(patchWire);
}

void CustomRouteWorker::endAddNets(
    const std::vector<crNet*>& routedNets) const {
    if (!design) {
        return;
    }

    // Flow:
    // 1. Iterate local route figures on each successfully routed CR net.
    // 2. Convert local crPathSeg/crVia into DB frPathSeg/frVia objects.
    // 3. Insert new DB objects into frNet and frRegionQuery.
    for (auto* cNet : routedNets) {
        if (!cNet || !cNet->getNet()) {
            continue;
        }

        int numPathSegs = 0;
        int numVias = 0;
        for (auto& connFig : cNet->getRouteConnFigs()) {
            if (connFig->typeId() == crcPathSeg) {
                endAddNetsPathSeg(static_cast<crPathSeg*>(connFig.get()));
                ++numPathSegs;
            } else if (connFig->typeId() == crcVia) {
                endAddNetsVia(static_cast<crVia*>(connFig.get()));
                ++numVias;
            }
        }
        auto* net = cNet->getNet();
        std::cout << "[customdr] writeback net " << net->getName()
                  << " pathSegs=" << numPathSegs << " vias=" << numVias
                  << " dbShapes=" << net->getShapes().size()
                  << " dbVias=" << net->getVias().size() << std::endl;
    }
}

void CustomRouteWorker::endAddNetsPathSeg(crPathSeg* pathSeg) const {
    if (!pathSeg || !pathSeg->hasNet() || !pathSeg->getNet() ||
        !pathSeg->getNet()->getNet() || !design) {
        return;
    }

    auto frPath = std::make_unique<frPathSeg>();
    auto begin = pathSeg->getBegin();
    auto end = pathSeg->getEnd();
    if ((begin.x() > end.x()) ||
        (begin.x() == end.x() && begin.y() > end.y())) {
        std::swap(begin, end);
    }
    frPath->setPoints(begin, end);
    frPath->setLayerNum(pathSeg->getLayerNum());
    frPath->setStyle(pathSeg->getStyle());

    auto* frPathPtr = frPath.get();
    std::unique_ptr<frShape> shape = std::move(frPath);
    auto* net = pathSeg->getNet()->getNet();
    net->addShape(shape);
    design->getRegionQuery()->addDRObj(frPathPtr);
}

void CustomRouteWorker::endAddNetsVia(crVia* via) const {
    if (!via || !via->hasNet() || !via->getNet() || !via->getNet()->getNet() ||
        !via->getViaDef() || !design) {
        return;
    }

    auto frViaObj = std::make_unique<frVia>(via->getViaDef());
    frViaObj->setOrigin(via->getOrigin());

    auto* frViaPtr = frViaObj.get();
    auto* net = via->getNet()->getNet();
    net->addVia(frViaObj);
    design->getRegionQuery()->addDRObj(frViaPtr);
}
}  // namespace fr
