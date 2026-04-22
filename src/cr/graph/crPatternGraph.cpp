#include "crPatternGraph.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "db/obj/frGuide.h"
#include "db/obj/frInstBlockage.h"
#include "db/obj/frMarker.h"
#include "db/obj/frShape.h"
#include "db/obj/frVia.h"
#include "db/tech/frConstraint.h"
#include "frRegionQuery.h"
#include "cr/type/crPathSeg.hpp"
#include "cr/type/crVia.hpp"
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
    planarDrcCosts.assign(getGridCapacity(), 0);
    viaDrcCosts.assign(getGridCapacity(), 0);
}

void crPatternGraph::clear() {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    xCoords.clear();
    yCoords.clear();
    zCoords.clear();
    planarDrcCosts.clear();
    viaDrcCosts.clear();
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
    return crMazeType(getCoordIdx(xCoords, xCoord),
                      getCoordIdx(yCoords, yCoord),
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

std::size_t crPatternGraph::getGridCapacity() const {
    return xDim * yDim * zDim;
}

std::size_t crPatternGraph::getGridIdx(const crMazeType& mazeIdx) const {
    if (!isValidMazeIdx(mazeIdx)) {
        throw std::invalid_argument("cannot index invalid crMazeType");
    }
    return static_cast<std::size_t>(mazeIdx.z) * xDim * yDim +
           static_cast<std::size_t>(mazeIdx.x) * yDim +
           static_cast<std::size_t>(mazeIdx.y);
}

std::size_t crPatternGraph::getPlanarCostIdx(const crMazeType& node) const {
    return getGridIdx(node);
}

std::size_t crPatternGraph::getViaCostIdx(const crMazeType& node,
                                          frDirEnum dir) const {
    auto canonical = node;
    if (dir == frDirEnum::D) {
        --canonical.z;
    }
    return getGridIdx(canonical);
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

std::uint64_t crPatternGraph::getCostEdgeKey(crMazeType node,
                                             frDirEnum dir) const {
    switch (dir) {
        case frDirEnum::W:
            --node.x;
            dir = frDirEnum::E;
            break;
        case frDirEnum::S:
            --node.y;
            dir = frDirEnum::N;
            break;
        case frDirEnum::D:
            --node.z;
            dir = frDirEnum::U;
            break;
        default:
            break;
    }
    return getEdgeKey(node, dir);
}

bool crPatternGraph::hasNode(const crMazeType& mazeIdx) const {
    return isValidMazeIdx(mazeIdx);
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

    return isValidMazeIdx(next);
}

bool crPatternGraph::hasDRCCost(const crMazeType& node, frDirEnum dir) const {
    if (dir == frDirEnum::U || dir == frDirEnum::D) {
        auto idx = getViaCostIdx(node, dir);
        return idx < viaDrcCosts.size() && viaDrcCosts[idx] > 0;
    }

    auto canonical = node;
    if (dir == frDirEnum::W) {
        --canonical.x;
    } else if (dir == frDirEnum::S) {
        --canonical.y;
    }

    auto idx = getPlanarCostIdx(canonical);
    return idx < planarDrcCosts.size() && planarDrcCosts[idx] > 0;
}

bool crPatternGraph::hasNonPrefCost(const crMazeType& node,
                                    frDirEnum dir) const {
    return isPlanarNonPrefDir(getLayerNum(node), dir);
}

void crPatternGraph::addDRCCost(const crMazeType& node, frDirEnum dir) {
    if (dir == frDirEnum::U || dir == frDirEnum::D) {
        auto idx = getViaCostIdx(node, dir);
        if (idx < viaDrcCosts.size() &&
            viaDrcCosts[idx] < std::numeric_limits<std::uint16_t>::max()) {
            ++viaDrcCosts[idx];
        }
        return;
    }

    auto idx = getPlanarCostIdx(node);
    if (idx < planarDrcCosts.size() &&
        planarDrcCosts[idx] < std::numeric_limits<std::uint16_t>::max()) {
        ++planarDrcCosts[idx];
    }
}

void crPatternGraph::subDRCCost(const crMazeType& node, frDirEnum dir) {
    if (dir == frDirEnum::U || dir == frDirEnum::D) {
        auto idx = getViaCostIdx(node, dir);
        if (idx >= viaDrcCosts.size() || viaDrcCosts[idx] == 0) {
            return;
        }
        --viaDrcCosts[idx];
        return;
    }

    auto idx = getPlanarCostIdx(node);
    if (idx >= planarDrcCosts.size() || planarDrcCosts[idx] == 0) {
        return;
    }
    --planarDrcCosts[idx];
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

namespace {

frCoord getBoxDistSquare(const frBox& lhs, const frBox& rhs) {
    auto dx = std::max(
        std::max(lhs.left(), rhs.left()) - std::min(lhs.right(), rhs.right()),
        static_cast<frCoord>(0));
    auto dy = std::max(
        std::max(lhs.bottom(), rhs.bottom()) - std::min(lhs.top(), rhs.top()),
        static_cast<frCoord>(0));
    return dx * dx + dy * dy;
}

}  // namespace

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

void crPatternGraph::modMetalShapeCost(const frBox& srcBox, frLayerNum layerNum,
                                       bool isAdd) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(layerNum)) {
        return;
    }

    auto z = getCoordIdx(zCoords, layerNum);
    if (z < 0) {
        return;
    }

    auto layer = getTech() ? getTech()->getLayer(layerNum) : nullptr;
    if (!layer) {
        return;
    }

    frCoord spacing = getMinSpacing(srcBox, layerNum);
    frCoord halfWidth = layer->getWidth() / 2;
    frBox searchBox;
    srcBox.bloat(std::max(static_cast<frCoord>(0), spacing + halfWidth),
                 searchBox);
    auto spacingSquare = spacing * spacing;

    auto xBegin =
        std::lower_bound(xCoords.begin(), xCoords.end(), searchBox.left());
    auto xEnd =
        std::upper_bound(xCoords.begin(), xCoords.end(), searchBox.right());
    auto yBegin =
        std::lower_bound(yCoords.begin(), yCoords.end(), searchBox.bottom());
    auto yEnd =
        std::upper_bound(yCoords.begin(), yCoords.end(), searchBox.top());

    for (auto xIt = xBegin; xIt != xEnd; ++xIt) {
        auto xIdx = static_cast<crIndex_t>(xIt - xCoords.begin());
        for (auto yIt = yBegin; yIt != yEnd; ++yIt) {
            auto yIdx = static_cast<crIndex_t>(yIt - yCoords.begin());
            crMazeType curr(xIdx, yIdx, z);
            auto pt = getPoint(curr);
            frBox testBox(pt.x() - halfWidth, pt.y() - halfWidth,
                          pt.x() + halfWidth, pt.y() + halfWidth);
            if (!srcBox.overlaps(testBox) &&
                getBoxDistSquare(srcBox, testBox) >= spacingSquare) {
                continue;
            }
            if (isAdd) {
                addDRCCost(curr, frDirEnum::E);
            } else {
                subDRCCost(curr, frDirEnum::E);
            }
        }
    }
}

void crPatternGraph::modViaShapeCost(const frBox& cutBox,
                                     frLayerNum lowerLayerNum, bool isAdd) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(lowerLayerNum)) {
        return;
    }

    auto z = getCoordIdx(zCoords, lowerLayerNum);
    if (z < 0 || static_cast<std::size_t>(z + 1) >= zDim) {
        return;
    }

    auto xBegin =
        std::lower_bound(xCoords.begin(), xCoords.end(), cutBox.left());
    auto xEnd =
        std::upper_bound(xCoords.begin(), xCoords.end(), cutBox.right());
    auto yBegin =
        std::lower_bound(yCoords.begin(), yCoords.end(), cutBox.bottom());
    auto yEnd = std::upper_bound(yCoords.begin(), yCoords.end(), cutBox.top());

    for (auto xIt = xBegin; xIt != xEnd; ++xIt) {
        auto xIdx = static_cast<crIndex_t>(xIt - xCoords.begin());
        for (auto yIt = yBegin; yIt != yEnd; ++yIt) {
            auto yIdx = static_cast<crIndex_t>(yIt - yCoords.begin());
            crMazeType node(xIdx, yIdx, z);
            auto pt = getPoint(node);
            if (!cutBox.contains(pt)) {
                continue;
            }
            if (isAdd) {
                addDRCCost(node, frDirEnum::U);
            } else {
                subDRCCost(node, frDirEnum::U);
            }
        }
    }
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

void crPatternGraph::modFrObjCost(frBlockObject* obj, bool isAdd) {
    if (!obj || !isExternalObject(obj)) {
        return;
    }

    if (obj->typeId() == frcPathSeg || obj->typeId() == frcRect ||
        obj->typeId() == frcPatchWire) {
        auto* shape = static_cast<frShape*>(obj);
        frBox box;
        shape->getBBox(box);
        modMetalShapeCost(box, shape->getLayerNum(), isAdd);
        return;
    }

    if (obj->typeId() == frcVia) {
        auto* via = static_cast<frVia*>(obj);
        auto* viaDef = via->getViaDef();
        if (!viaDef) {
            return;
        }

        frBox box;
        via->getLayer1BBox(box);
        modMetalShapeCost(box, viaDef->getLayer1Num(), isAdd);
        via->getLayer2BBox(box);
        modMetalShapeCost(box, viaDef->getLayer2Num(), isAdd);
        via->getCutBBox(box);
        modViaShapeCost(box, viaDef->getLayer1Num(), isAdd);
    }
}

void crPatternGraph::initExternalDRCCost() {
    auto design = getDesign();
    if (!design || !worker) {
        return;
    }

    std::vector<frBlockObject*> result;
    design->getRegionQuery()->queryDRObj(worker->getExtBox(), result);
    for (auto* obj : result) {
        modFrObjCost(obj, true);
    }
}

void crPatternGraph::initDRCCost() {
    std::fill(planarDrcCosts.begin(), planarDrcCosts.end(), 0);
    std::fill(viaDrcCosts.begin(), viaDrcCosts.end(), 0);
    initExternalDRCCost();
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
        modViaCost(static_cast<const crVia*>(connFig), isAdd);
    }
}

void crPatternGraph::modPathSegCost(const crPathSeg* pathSeg, bool isAdd) {
    if (!pathSeg || !pathSeg->hasMazeIdx()) {
        return;
    }

    modMetalShapeCost(pathSeg->getBBox(), pathSeg->getLayerNum(), isAdd);
}

void crPatternGraph::modViaCost(const crVia* via, bool isAdd) {
    if (!via || !via->getViaDef()) {
        return;
    }

    auto* viaDef = via->getViaDef();
    modMetalShapeCost(via->getLowerLayerFigBBox(), viaDef->getLayer1Num(),
                      isAdd);
    modMetalShapeCost(via->getUpperLayerFigBBox(), viaDef->getLayer2Num(),
                      isAdd);
    modViaShapeCost(via->getCutFigBBox(), viaDef->getLayer1Num(), isAdd);
}

void crPatternGraph::build(CustomRouteWorker* worker) {
    clear();
    this->worker = worker;
}

}  // namespace fr
