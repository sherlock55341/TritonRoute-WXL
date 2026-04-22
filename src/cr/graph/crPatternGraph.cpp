#include "crPatternGraph.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "db/obj/frShape.h"
#include "db/obj/frVia.h"
#include "db/tech/frConstraint.h"
#include "cr/type/crPathSeg.hpp"
#include "cr/worker/crWorker.hpp"

namespace fr {

frDesign* crPatternGraph::getDesign() const {
    return worker ? worker->getDesign() : nullptr;
}

frTechObject* crPatternGraph::getTech() const {
    auto design = getDesign();
    return design ? design->getTech() : nullptr;
}

void crPatternGraph::setDims(std::size_t _xDim, std::size_t _yDim,
                             std::size_t _zDim) {
    xDim = _xDim;
    yDim = _yDim;
    zDim = _zDim;
}

void crPatternGraph::setCoords(std::vector<frCoord> xCoordsIn,
                               std::vector<frCoord> yCoordsIn,
                               std::vector<frLayerNum> zCoordsIn) {
    std::sort(xCoordsIn.begin(), xCoordsIn.end());
    std::sort(yCoordsIn.begin(), yCoordsIn.end());
    std::sort(zCoordsIn.begin(), zCoordsIn.end());
    xCoordsIn.erase(std::unique(xCoordsIn.begin(), xCoordsIn.end()),
                    xCoordsIn.end());
    yCoordsIn.erase(std::unique(yCoordsIn.begin(), yCoordsIn.end()),
                    yCoordsIn.end());
    zCoordsIn.erase(std::unique(zCoordsIn.begin(), zCoordsIn.end()),
                    zCoordsIn.end());

    xCoords = std::move(xCoordsIn);
    yCoords = std::move(yCoordsIn);
    zCoords = std::move(zCoordsIn);
    setDims(xCoords.size(), yCoords.size(), zCoords.size());
}

void crPatternGraph::clear() {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    xCoords.clear();
    yCoords.clear();
    zCoords.clear();
    nodes.clear();
    nodeMap.clear();
    drcEdges.clear();
    nonPrefEdges.clear();
}

template <typename T>
bool crPatternGraph::hasCoord(const std::vector<T>& coords, T coord) const {
    return std::binary_search(coords.begin(), coords.end(), coord);
}

template <typename T>
crIndex_t crPatternGraph::getCoordIdx(const std::vector<T>& coords,
                                      T coord) const {
    auto it = std::lower_bound(coords.begin(), coords.end(), coord);
    if (it == coords.end() || *it != coord) {
        return -1;
    }
    return static_cast<crIndex_t>(it - coords.begin());
}

bool crPatternGraph::hasMazeXCoord(frCoord xCoord) const {
    return hasCoord(xCoords, xCoord);
}

bool crPatternGraph::hasMazeYCoord(frCoord yCoord) const {
    return hasCoord(yCoords, yCoord);
}

bool crPatternGraph::hasMazeZCoord(frLayerNum layerNum) const {
    return hasCoord(zCoords, layerNum);
}

bool crPatternGraph::hasMazeIdx(frCoord xCoord, frCoord yCoord,
                                frLayerNum layerNum) const {
    return hasMazeXCoord(xCoord) && hasMazeYCoord(yCoord) &&
           hasMazeZCoord(layerNum);
}

crMazeType crPatternGraph::getMazeIdx(frCoord xCoord, frCoord yCoord,
                                      frLayerNum layerNum) const {
    return crMazeType(getCoordIdx(xCoords, xCoord), getCoordIdx(yCoords, yCoord),
                      getCoordIdx(zCoords, layerNum));
}

frPoint crPatternGraph::getPoint(const crMazeType& mazeIdx) const {
    if (!isValidMazeIdx(mazeIdx)) {
        return frPoint();
    }
    return frPoint(xCoords[mazeIdx.x], yCoords[mazeIdx.y]);
}

frLayerNum crPatternGraph::getLayerNum(const crMazeType& mazeIdx) const {
    if (!isValidMazeIdx(mazeIdx)) {
        return -1;
    }
    return zCoords[mazeIdx.z];
}

bool crPatternGraph::isValidMazeIdx(const crMazeType& mazeIdx) const {
    return !mazeIdx.empty() && mazeIdx.x >= 0 && mazeIdx.y >= 0 &&
           mazeIdx.z >= 0 && static_cast<std::size_t>(mazeIdx.x) < xDim &&
           static_cast<std::size_t>(mazeIdx.y) < yDim &&
           static_cast<std::size_t>(mazeIdx.z) < zDim;
}

std::uint64_t crPatternGraph::getMapKey(const crMazeType& mazeIdx) const {
    if (!isValidMazeIdx(mazeIdx)) {
        throw std::invalid_argument("cannot encode empty crMazeType");
    }

    auto x = static_cast<std::uint64_t>(mazeIdx.x);
    auto y = static_cast<std::uint64_t>(mazeIdx.y);
    auto z = static_cast<std::uint64_t>(mazeIdx.z);
    return z * static_cast<std::uint64_t>(xDim) *
               static_cast<std::uint64_t>(yDim) +
           x * static_cast<std::uint64_t>(yDim) + y;
}

std::uint64_t crPatternGraph::getNodeKey(const crMazeType& mazeIdx) const {
    return getMapKey(mazeIdx);
}

std::uint64_t crPatternGraph::getEdgeKey(const crMazeType& node,
                                         frDirEnum dir) const {
    return getNodeKey(node) * 7 + static_cast<std::uint64_t>(dir);
}

bool crPatternGraph::hasNode(const crMazeType& mazeIdx) const {
    return nodeMap.find(getMapKey(mazeIdx)) != nodeMap.end();
}

int crPatternGraph::getNodeIdx(const crMazeType& mazeIdx) const {
    auto it = nodeMap.find(getMapKey(mazeIdx));
    if (it == nodeMap.end()) {
        return -1;
    }
    return it->second;
}

int crPatternGraph::addNode(const crMazeType& mazeIdx) {
    if (!isValidMazeIdx(mazeIdx)) {
        return -1;
    }

    auto key = getMapKey(mazeIdx);
    auto it = nodeMap.find(key);
    if (it != nodeMap.end()) {
        return it->second;
    }

    nodes.push_back(mazeIdx);
    auto nodeIdx = static_cast<int>(nodes.size()) - 1;
    nodeMap[key] = nodeIdx;
    return nodeIdx;
}

void crPatternGraph::addNodes(const std::vector<crMazeType>& mazeIdxs) {
    for (auto& mazeIdx : mazeIdxs) {
        addNode(mazeIdx);
    }
}

void crPatternGraph::addRoutingLayerNodes(frCoord xCoord, frCoord yCoord) {
    if (!hasMazeXCoord(xCoord) || !hasMazeYCoord(yCoord)) {
        return;
    }

    for (auto layerNum : zCoords) {
        addNode(getMazeIdx(xCoord, yCoord, layerNum));
    }
}

bool crPatternGraph::getNextMazeIdx(const crMazeType& curr, frDirEnum dir,
                                    crMazeType& next) const {
    if (!isValidMazeIdx(curr)) {
        return false;
    }

    next = curr;
    switch (dir) {
        case frDirEnum::E:
            ++next.x;
            break;
        case frDirEnum::W:
            --next.x;
            break;
        case frDirEnum::N:
            ++next.y;
            break;
        case frDirEnum::S:
            --next.y;
            break;
        case frDirEnum::U:
            ++next.z;
            break;
        case frDirEnum::D:
            --next.z;
            break;
        default:
            return false;
    }

    return isValidMazeIdx(next) && hasNode(next);
}

bool crPatternGraph::hasDRCCost(const crMazeType& node, frDirEnum dir) const {
    auto it = drcEdges.find(getEdgeKey(node, dir));
    return it != drcEdges.end() && it->second > 0;
}

bool crPatternGraph::hasNonPrefCost(const crMazeType& node,
                                    frDirEnum dir) const {
    auto it = nonPrefEdges.find(getEdgeKey(node, dir));
    return it != nonPrefEdges.end() && it->second > 0;
}

void crPatternGraph::addNonPrefCost(const crMazeType& node, frDirEnum dir) {
    ++nonPrefEdges[getEdgeKey(node, dir)];
}

void crPatternGraph::subNonPrefCost(const crMazeType& node, frDirEnum dir) {
    auto key = getEdgeKey(node, dir);
    auto it = nonPrefEdges.find(key);
    if (it == nonPrefEdges.end()) {
        return;
    }
    --(it->second);
    if (it->second <= 0) {
        nonPrefEdges.erase(it);
    }
}

void crPatternGraph::addDRCCost(const crMazeType& node, frDirEnum dir) {
    ++drcEdges[getEdgeKey(node, dir)];
}

void crPatternGraph::subDRCCost(const crMazeType& node, frDirEnum dir) {
    auto key = getEdgeKey(node, dir);
    auto it = drcEdges.find(key);
    if (it == drcEdges.end()) {
        return;
    }
    --(it->second);
    if (it->second <= 0) {
        drcEdges.erase(it);
    }
}

frCoord crPatternGraph::getMinSpacing(const frBox& box,
                                      frLayerNum layerNum) const {
    auto tech = getTech();
    auto layer = tech ? tech->getLayer(layerNum) : nullptr;
    if (!layer) {
        return 0;
    }

    auto con = layer->getMinSpacing();
    if (!con) {
        return layer->getWidth();
    }

    auto width1 = box.width();
    auto width2 = layer->getWidth();
    auto prl = box.length();
    switch (con->typeId()) {
        case frConstraintTypeEnum::frcSpacingConstraint:
            return static_cast<frSpacingConstraint*>(con)->getMinSpacing();
        case frConstraintTypeEnum::frcSpacingTablePrlConstraint:
            return static_cast<frSpacingTablePrlConstraint*>(con)->find(
                std::max(width1, static_cast<frCoord>(width2)), prl);
        case frConstraintTypeEnum::frcSpacingTableTwConstraint:
            return static_cast<frSpacingTableTwConstraint*>(con)->find(
                width1, width2, prl);
        default:
            return layer->getWidth();
    }
}

frBox crPatternGraph::getPlanarEdgeBox(const crMazeType& curr,
                                       const crMazeType& next) const {
    auto currPt = getPoint(curr);
    auto nextPt = getPoint(next);
    auto layerNum = getLayerNum(curr);
    auto layer = getTech() ? getTech()->getLayer(layerNum) : nullptr;
    auto width = layer ? layer->getWidth() : 0;
    auto halfWidth = width / 2;

    if (currPt.y() == nextPt.y()) {
        return frBox(std::min(currPt.x(), nextPt.x()), currPt.y() - halfWidth,
                     std::max(currPt.x(), nextPt.x()), currPt.y() + halfWidth);
    }
    return frBox(currPt.x() - halfWidth, std::min(currPt.y(), nextPt.y()),
                 currPt.x() + halfWidth, std::max(currPt.y(), nextPt.y()));
}

bool crPatternGraph::hasShortViolation(const frBox& edgeBox,
                                       frLayerNum layerNum) const {
    auto design = getDesign();
    auto rq = design ? design->getRegionQuery() : nullptr;
    if (!rq) {
        return false;
    }

    std::vector<frBlockObject*> result;
    rq->queryDRObj(edgeBox, layerNum, result);
    for (auto* obj : result) {
        if (!isExternalObject(obj)) {
            continue;
        }

        frBox objBox;
        if (obj->typeId() == frcPathSeg || obj->typeId() == frcRect ||
            obj->typeId() == frcPatchWire) {
            static_cast<frShape*>(obj)->getBBox(objBox);
        } else if (obj->typeId() == frcVia) {
            static_cast<frVia*>(obj)->getBBox(objBox);
        } else {
            continue;
        }

        if (edgeBox.overlaps(objBox)) {
            return true;
        }
    }
    return false;
}

bool crPatternGraph::hasSpacingViolation(const frBox& edgeBox,
                                         frLayerNum layerNum) const {
    auto spacing = getMinSpacing(edgeBox, layerNum);
    if (spacing <= 0) {
        return false;
    }

    frBox queryBox;
    edgeBox.bloat(spacing, queryBox);

    auto design = getDesign();
    auto rq = design ? design->getRegionQuery() : nullptr;
    if (!rq) {
        return false;
    }

    std::vector<frBlockObject*> result;
    rq->queryDRObj(queryBox, layerNum, result);
    for (auto* obj : result) {
        if (!isExternalObject(obj)) {
            continue;
        }

        frBox objBox;
        if (obj->typeId() == frcPathSeg || obj->typeId() == frcRect ||
            obj->typeId() == frcPatchWire) {
            static_cast<frShape*>(obj)->getBBox(objBox);
        } else if (obj->typeId() == frcVia) {
            static_cast<frVia*>(obj)->getBBox(objBox);
        } else {
            continue;
        }

        if (edgeBox.overlaps(objBox)) {
            continue;
        }
        if (queryBox.overlaps(objBox, false)) {
            return true;
        }
    }
    return false;
}

bool crPatternGraph::isExternalObject(frBlockObject* obj) const {
    if (!obj || !worker) {
        return false;
    }

    if (obj->typeId() == frcPathSeg || obj->typeId() == frcRect ||
        obj->typeId() == frcPatchWire) {
        auto* shape = static_cast<frShape*>(obj);
        auto* net = shape->hasNet() ? shape->getNet() : nullptr;
        for (auto& cNet : worker->getNets()) {
            if (cNet->getNet() == net) {
                return false;
            }
        }
        return true;
    }

    if (obj->typeId() == frcVia) {
        auto* via = static_cast<frVia*>(obj);
        auto* net = via->hasNet() ? via->getNet() : nullptr;
        for (auto& cNet : worker->getNets()) {
            if (cNet->getNet() == net) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool crPatternGraph::isPlanarNonPrefDir(frLayerNum layerNum,
                                        frDirEnum dir) const {
    if (dir != frDirEnum::E && dir != frDirEnum::W && dir != frDirEnum::N &&
        dir != frDirEnum::S) {
        return false;
    }

    auto tech = getTech();
    auto layer = tech ? tech->getLayer(layerNum) : nullptr;
    if (!layer) {
        return false;
    }

    auto prefDir = layer->getDir();
    if (prefDir == frcHorzPrefRoutingDir) {
        return dir == frDirEnum::N || dir == frDirEnum::S;
    }
    if (prefDir == frcVertPrefRoutingDir) {
        return dir == frDirEnum::E || dir == frDirEnum::W;
    }
    return false;
}

void crPatternGraph::initNonPrefCost() {
    for (auto& node : nodes) {
        auto layerNum = getLayerNum(node);
        for (auto dir : {frDirEnum::E, frDirEnum::W, frDirEnum::N,
                         frDirEnum::S}) {
            if (!isPlanarNonPrefDir(layerNum, dir)) {
                continue;
            }

            crMazeType next;
            if (!getNextMazeIdx(node, dir, next)) {
                continue;
            }
            addNonPrefCost(node, dir);
        }
    }
}

void crPatternGraph::initPlanarDRCCost() {
    for (auto& node : nodes) {
        auto layerNum = getLayerNum(node);
        for (auto dir : {frDirEnum::E, frDirEnum::W, frDirEnum::N,
                         frDirEnum::S}) {
            crMazeType next;
            if (!getNextMazeIdx(node, dir, next)) {
                continue;
            }

            auto edgeBox = getPlanarEdgeBox(node, next);
            if (hasShortViolation(edgeBox, layerNum) ||
                hasSpacingViolation(edgeBox, layerNum)) {
                addDRCCost(node, dir);
            }
        }
    }
}

void crPatternGraph::initDRCCost() {
    drcEdges.clear();
    nonPrefEdges.clear();
    initNonPrefCost();
    initPlanarDRCCost();
}

void crPatternGraph::addPathCost(const crConnFig* connFig) {
    modPathCost(connFig, true);
}

void crPatternGraph::subPathCost(const crConnFig* connFig) {
    modPathCost(connFig, false);
}

void crPatternGraph::modPathCost(const crConnFig* connFig, bool isAdd) {
    if (!connFig) {
        return;
    }

    if (connFig->typeId() == crcPathSeg) {
        modPathSegCost(static_cast<const crPathSeg*>(connFig), isAdd);
    } else if (connFig->typeId() == crcVia) {
        return;
    }
}

void crPatternGraph::modPathSegCost(const crPathSeg* pathSeg, bool isAdd) {
    if (!pathSeg || !pathSeg->hasMazeIdx()) {
        return;
    }

    auto beginMazeIdx = pathSeg->getBeginMazeIdx();
    auto endMazeIdx = pathSeg->getEndMazeIdx();
    auto layerNum = pathSeg->getLayerNum();
    if (beginMazeIdx.empty() || endMazeIdx.empty() || beginMazeIdx.z != endMazeIdx.z) {
        return;
    }

    auto dir = frDirEnum::UNKNOWN;
    if (beginMazeIdx.x == endMazeIdx.x) {
        dir = (beginMazeIdx.y <= endMazeIdx.y) ? frDirEnum::N : frDirEnum::S;
    } else if (beginMazeIdx.y == endMazeIdx.y) {
        dir = (beginMazeIdx.x <= endMazeIdx.x) ? frDirEnum::E : frDirEnum::W;
    } else {
        return;
    }

    auto curr = beginMazeIdx;
    while (!(curr == endMazeIdx)) {
        crMazeType next;
        if (!getNextMazeIdx(curr, dir, next)) {
            break;
        }

        auto edgeBox = getPlanarEdgeBox(curr, next);
        auto hasViolation =
            hasShortViolation(edgeBox, layerNum) || hasSpacingViolation(edgeBox, layerNum);
        if (hasViolation) {
            if (isAdd) {
                addDRCCost(curr, dir);
            } else {
                subDRCCost(curr, dir);
            }
        }
        curr = next;
    }
}

void crPatternGraph::build(CustomRouteWorker* worker) {
    clear();
    this->worker = worker;
}

}  // namespace fr
