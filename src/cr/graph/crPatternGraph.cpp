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

namespace {

constexpr unsigned kBlockCostE = 3;
constexpr unsigned kBlockCostN = 4;
constexpr unsigned kBlockCostU = 5;
constexpr unsigned kGridCostE = 12;
constexpr unsigned kGridCostN = 13;
constexpr unsigned kGridCostU = 14;
constexpr unsigned kDrcPlanarOffset = 16;
constexpr unsigned kDrcViaOffset = 24;
constexpr unsigned kMarkerPlanarOffset = 32;
constexpr unsigned kMarkerViaOffset = 40;
constexpr unsigned kShapeViaOffset = 48;
constexpr unsigned kShapePlanarOffset = 56;
constexpr unsigned kCounterWidth = 8;

struct crCostBitRange {
    unsigned offset;
    unsigned width;
};

bool isViaDir(frDirEnum dir) {
    return dir == frDirEnum::U || dir == frDirEnum::D;
}

crCostBitRange getCostBitRange(crCostClass costClass, frDirEnum dir) {
    bool isVia = isViaDir(dir);
    switch (costClass) {
        case crCostClass::Grid:
            if (isVia) {
                return {kGridCostU, 1};
            }
            return {dir == frDirEnum::N || dir == frDirEnum::S ? kGridCostN
                                                               : kGridCostE,
                    1};
        case crCostClass::Shape:
            return {isVia ? kShapeViaOffset : kShapePlanarOffset,
                    kCounterWidth};
        case crCostClass::Drc:
            return {isVia ? kDrcViaOffset : kDrcPlanarOffset, kCounterWidth};
        case crCostClass::Marker:
            return {isVia ? kMarkerViaOffset : kMarkerPlanarOffset,
                    kCounterWidth};
        case crCostClass::Block:
            if (isVia) {
                return {kBlockCostU, 1};
            }
            return {dir == frDirEnum::N || dir == frDirEnum::S ? kBlockCostN
                                                               : kBlockCostE,
                    1};
    }
    return {kDrcPlanarOffset, kCounterWidth};
}

std::uint64_t getMask(unsigned width) {
    return (std::uint64_t{1} << width) - 1;
}

std::uint16_t getBits(std::uint64_t word, crCostBitRange range) {
    return static_cast<std::uint16_t>((word >> range.offset) &
                                      getMask(range.width));
}

void setBits(std::uint64_t& word, crCostBitRange range, std::uint16_t value) {
    auto mask = getMask(range.width) << range.offset;
    word &= ~mask;
    word |= (static_cast<std::uint64_t>(value) & getMask(range.width))
            << range.offset;
}

void addBits(std::uint64_t& word, crCostBitRange range) {
    if (range.width == 1) {
        setBits(word, range, 1);
        return;
    }

    auto value = getBits(word, range);
    auto maxValue = getMask(range.width);
    if (value < maxValue) {
        setBits(word, range, value + 1);
    }
}

void subBits(std::uint64_t& word, crCostBitRange range) {
    if (range.width == 1) {
        setBits(word, range, 0);
        return;
    }

    auto value = getBits(word, range);
    if (value > 0) {
        setBits(word, range, value - 1);
    }
}

void clearDrcBits(std::uint64_t& word) {
    setBits(word, {kDrcPlanarOffset, kCounterWidth}, 0);
    setBits(word, {kDrcViaOffset, kCounterWidth}, 0);
}

void clearShapeBits(std::uint64_t& word) {
    setBits(word, {kShapePlanarOffset, kCounterWidth}, 0);
    setBits(word, {kShapeViaOffset, kCounterWidth}, 0);
}

}  // namespace

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
    // Flow:
    // 1. Sort and deduplicate physical coordinate arrays.
    // 2. Publish dimensions derived from the coordinate arrays.
    // 3. Allocate one DR-like bitfield word per dense graph node.
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
    bits.assign(getGridCapacity(), 0);
}

void crPatternGraph::clear() {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    xCoords.clear();
    yCoords.clear();
    zCoords.clear();
    bits.clear();
    sViaDefs.clear();
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

crMazeType crPatternGraph::getCanonicalCostNode(crMazeType node,
                                                frDirEnum dir) const {
    switch (dir) {
        case frDirEnum::W:
            --node.x;
            break;
        case frDirEnum::S:
            --node.y;
            break;
        case frDirEnum::D:
            --node.z;
            break;
        default:
            break;
    }
    return node;
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
    return hasCost(node, dir, crCostClass::Drc);
}

bool crPatternGraph::hasGridCost(const crMazeType& node, frDirEnum dir) const {
    return hasCost(node, dir, crCostClass::Grid);
}

bool crPatternGraph::hasShapeCost(const crMazeType& node, frDirEnum dir) const {
    return hasCost(node, dir, crCostClass::Shape);
}

bool crPatternGraph::hasMarkerCost(const crMazeType& node,
                                   frDirEnum dir) const {
    return hasCost(node, dir, crCostClass::Marker);
}

bool crPatternGraph::hasBlockCost(const crMazeType& node, frDirEnum dir) const {
    return hasCost(node, dir, crCostClass::Block);
}

bool crPatternGraph::hasNonPrefCost(const crMazeType& node,
                                    frDirEnum dir) const {
    return isPlanarNonPrefDir(getLayerNum(node), dir);
}

bool crPatternGraph::hasCost(const crMazeType& node, frDirEnum dir,
                             crCostClass costClass) const {
    // Cost storage follows DR's E/N/U canonical direction convention. W/S/D
    // are normalized to the neighboring E/N/U storage node before reading bits.
    auto canonical = getCanonicalCostNode(node, dir);
    if (!isValidMazeIdx(canonical)) {
        return false;
    }

    auto idx = getGridIdx(canonical);
    return idx < bits.size() &&
           getBits(bits[idx], getCostBitRange(costClass, dir)) > 0;
}

void crPatternGraph::addCost(const crMazeType& node, frDirEnum dir,
                             crCostClass costClass) {
    // Additive counters allow cost removal to mirror insertion when local
    // routes are ripped up or replaced.
    auto canonical = getCanonicalCostNode(node, dir);
    if (!isValidMazeIdx(canonical)) {
        return;
    }

    auto idx = getGridIdx(canonical);
    if (idx < bits.size()) {
        addBits(bits[idx], getCostBitRange(costClass, dir));
    }
}

void crPatternGraph::subCost(const crMazeType& node, frDirEnum dir,
                             crCostClass costClass) {
    auto canonical = getCanonicalCostNode(node, dir);
    if (!isValidMazeIdx(canonical)) {
        return;
    }

    auto idx = getGridIdx(canonical);
    if (idx < bits.size()) {
        subBits(bits[idx], getCostBitRange(costClass, dir));
    }
}

void crPatternGraph::addGridCost(const crMazeType& node, frDirEnum dir) {
    addCost(node, dir, crCostClass::Grid);
}

void crPatternGraph::subGridCost(const crMazeType& node, frDirEnum dir) {
    subCost(node, dir, crCostClass::Grid);
}

void crPatternGraph::addShapeCost(const crMazeType& node, frDirEnum dir) {
    addCost(node, dir, crCostClass::Shape);
}

void crPatternGraph::subShapeCost(const crMazeType& node, frDirEnum dir) {
    subCost(node, dir, crCostClass::Shape);
}

void crPatternGraph::addDRCCost(const crMazeType& node, frDirEnum dir) {
    addCost(node, dir, crCostClass::Drc);
}

void crPatternGraph::subDRCCost(const crMazeType& node, frDirEnum dir) {
    subCost(node, dir, crCostClass::Drc);
}

void crPatternGraph::addMarkerCost(const crMazeType& node, frDirEnum dir) {
    addCost(node, dir, crCostClass::Marker);
}

void crPatternGraph::subMarkerCost(const crMazeType& node, frDirEnum dir) {
    subCost(node, dir, crCostClass::Marker);
}

void crPatternGraph::addBlockCost(const crMazeType& node, frDirEnum dir) {
    addCost(node, dir, crCostClass::Block);
}

void crPatternGraph::subBlockCost(const crMazeType& node, frDirEnum dir) {
    subCost(node, dir, crCostClass::Block);
}

void crPatternGraph::setSVia(const crMazeType& node, frViaDef* viaDef) {
    if (!viaDef || !isValidMazeIdx(node)) {
        return;
    }
    sViaDefs[node] = viaDef;
}

bool crPatternGraph::isSVia(const crMazeType& node) const {
    return isValidMazeIdx(node) && sViaDefs.find(node) != sViaDefs.end();
}

frViaDef* crPatternGraph::getSViaDef(const crMazeType& node) const {
    if (!isValidMazeIdx(node)) {
        return nullptr;
    }
    auto it = sViaDefs.find(node);
    return it == sViaDefs.end() ? nullptr : it->second;
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

frCoord getPointBoxDistSquare(const frPoint& point, const frBox& box) {
    auto dx = std::max(
        std::max(box.left(), point.x()) - std::min(box.right(), point.x()),
        static_cast<frCoord>(0));
    auto dy = std::max(
        std::max(box.bottom(), point.y()) - std::min(box.top(), point.y()),
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

void crPatternGraph::modTypedCost(const crMazeType& node, frDirEnum dir,
                                  int type) {
    // DR-compatible type convention:
    // 0=sub DRC, 1=add DRC, 2=sub shape, 3=add shape.
    switch (type) {
        case 0:
            subDRCCost(node, dir);
            break;
        case 1:
            addDRCCost(node, dir);
            break;
        case 2:
            subShapeCost(node, dir);
            break;
        case 3:
            addShapeCost(node, dir);
            break;
        default:
            break;
    }
}

void crPatternGraph::modMetalShapeCost(const frBox& srcBox, frLayerNum layerNum,
                                       int type) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(layerNum)) {
        return;
    }

    // Flow:
    // 1. Compute the layer spacing requirement and candidate wire half-width.
    // 2. Visit graph points whose candidate-width box can fall inside the
    //    spacing influence window.
    // 3. Use the DR-like corner-to-box distance test to decide affected points.
    // 4. Apply the caller-selected DRC/shape typed cost update.
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
            frPoint pt1(pt.x() + halfWidth, pt.y() - halfWidth);
            frPoint pt2(pt.x() + halfWidth, pt.y() + halfWidth);
            frPoint pt3(pt.x() - halfWidth, pt.y() - halfWidth);
            frPoint pt4(pt.x() - halfWidth, pt.y() + halfWidth);
            auto distSquare = std::min(getPointBoxDistSquare(pt1, srcBox),
                                       getPointBoxDistSquare(pt2, srcBox));
            distSquare =
                std::min(distSquare, getPointBoxDistSquare(pt3, srcBox));
            distSquare =
                std::min(distSquare, getPointBoxDistSquare(pt4, srcBox));
            if (distSquare >= spacingSquare) {
                continue;
            }
            modTypedCost(curr, frDirEnum::E, type);
        }
    }
}

void crPatternGraph::modMetalShapeViaCost(const frBox& srcBox,
                                          frLayerNum layerNum, bool isUpperVia,
                                          int type) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(layerNum)) {
        return;
    }

    // Flow:
    // 1. Build the default-via metal box on the layer adjacent to srcBox.
    // 2. Enumerate graph via candidate origins in the spacing search window.
    // 3. Override the default via footprint with AP-provided SVia when present.
    // 4. Apply typed cost to candidates that overlap or violate spacing.
    auto z = getCoordIdx(zCoords, layerNum);
    if (z < 0) {
        return;
    }

    auto tech = getTech();
    auto layer = tech ? tech->getLayer(layerNum) : nullptr;
    if (!tech || !layer) {
        return;
    }

    frLayerNum cutLayerNum = isUpperVia ? layerNum + 1 : layerNum - 1;
    if (cutLayerNum < tech->getBottomLayerNum() ||
        cutLayerNum > tech->getTopLayerNum()) {
        return;
    }

    auto cutLayer = tech->getLayer(cutLayerNum);
    auto viaDef = cutLayer ? cutLayer->getDefaultViaDef() : nullptr;
    if (!viaDef) {
        return;
    }

    frVia via(viaDef);
    frBox viaBox;
    if (isUpperVia) {
        via.getLayer1BBox(viaBox);
    } else {
        via.getLayer2BBox(viaBox);
    }

    auto con = layer->getMinSpacing();
    if (!con) {
        return;
    }

    frCoord width1 = srcBox.width();
    frCoord length1 = srcBox.length();
    frCoord width2 = viaBox.width();
    frCoord length2 = viaBox.length();
    frCoord bloatDist = 0;
    if (con->typeId() == frConstraintTypeEnum::frcSpacingConstraint) {
        bloatDist = static_cast<frSpacingConstraint*>(con)->getMinSpacing();
    } else if (con->typeId() ==
               frConstraintTypeEnum::frcSpacingTablePrlConstraint) {
        bloatDist = static_cast<frSpacingTablePrlConstraint*>(con)->find(
            std::max(width1, width2), length2);
    } else if (con->typeId() ==
               frConstraintTypeEnum::frcSpacingTableTwConstraint) {
        bloatDist = static_cast<frSpacingTableTwConstraint*>(con)->find(
            width1, width2, length2);
    } else {
        return;
    }

    frBox searchBox(srcBox.left() - bloatDist - viaBox.right() + 1,
                    srcBox.bottom() - bloatDist - viaBox.top() + 1,
                    srcBox.right() + bloatDist - viaBox.left() - 1,
                    srcBox.top() + bloatDist - viaBox.bottom() - 1);

    auto xBegin =
        std::lower_bound(xCoords.begin(), xCoords.end(), searchBox.left());
    auto xEnd =
        std::upper_bound(xCoords.begin(), xCoords.end(), searchBox.right());
    auto yBegin =
        std::lower_bound(yCoords.begin(), yCoords.end(), searchBox.bottom());
    auto yEnd =
        std::upper_bound(yCoords.begin(), yCoords.end(), searchBox.top());

    frTransform xform;
    frBox testBox;
    frVia sVia;
    frBox sViaBox;
    for (auto xIt = xBegin; xIt != xEnd; ++xIt) {
        auto xIdx = static_cast<crIndex_t>(xIt - xCoords.begin());
        for (auto yIt = yBegin; yIt != yEnd; ++yIt) {
            auto yIdx = static_cast<crIndex_t>(yIt - yCoords.begin());
            crMazeType node(xIdx, yIdx, isUpperVia ? z : z - 1);
            if (!isValidMazeIdx(node)) {
                continue;
            }

            auto pt = frPoint(*xIt, *yIt);
            xform.set(pt);
            testBox.set(viaBox);
            if (auto sViaDef = getSViaDef(node)) {
                sVia.setViaDef(sViaDef);
                if (isUpperVia) {
                    sVia.getLayer1BBox(sViaBox);
                } else {
                    sVia.getLayer2BBox(sViaBox);
                }
                testBox.set(sViaBox);
            }
            testBox.transform(xform);

            frCoord reqDist = 0;
            frCoord dx = 0;
            frCoord dy = 0;
            frCoord distSquare = getBoxDistSquare(srcBox, testBox);
            auto rawDx = std::max(srcBox.left(), testBox.left()) -
                         std::min(srcBox.right(), testBox.right());
            auto rawDy = std::max(srcBox.bottom(), testBox.bottom()) -
                         std::min(srcBox.top(), testBox.top());
            dx = std::max(rawDx, static_cast<frCoord>(0));
            dy = std::max(rawDy, static_cast<frCoord>(0));
            frCoord prl = std::max(dx, dy);
            if (dx == 0 && dy > 0) {
                prl = viaBox.right() - viaBox.left();
            } else if (dx > 0 && dy == 0) {
                prl = viaBox.top() - viaBox.bottom();
            }

            if (con->typeId() == frConstraintTypeEnum::frcSpacingConstraint) {
                reqDist =
                    static_cast<frSpacingConstraint*>(con)->getMinSpacing();
            } else if (con->typeId() ==
                       frConstraintTypeEnum::frcSpacingTablePrlConstraint) {
                reqDist = static_cast<frSpacingTablePrlConstraint*>(con)->find(
                    std::max(width1, width2), prl);
            } else if (con->typeId() ==
                       frConstraintTypeEnum::frcSpacingTableTwConstraint) {
                reqDist = static_cast<frSpacingTableTwConstraint*>(con)->find(
                    width1, width2, prl);
            }

            if (srcBox.overlaps(testBox) || distSquare < reqDist * reqDist) {
                modTypedCost(node, frDirEnum::U, type);
            }
        }
    }
}

void crPatternGraph::modMetalShapeAllCost(const frBox& srcBox,
                                          frLayerNum layerNum, int type) {
    modMetalShapeCost(srcBox, layerNum, type);
    modMetalShapeViaCost(srcBox, layerNum, true, type);
    modMetalShapeViaCost(srcBox, layerNum, false, type);
}

void crPatternGraph::modViaShapeCost(const frBox& cutBox,
                                     frLayerNum lowerLayerNum, int type) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(lowerLayerNum)) {
        return;
    }

    // Mark via candidate origins whose point lies inside an existing cut box.
    // Full cut spacing/mincut modeling remains a follow-up item.
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
            modTypedCost(node, frDirEnum::U, type);
        }
    }
}

void crPatternGraph::modEolSpacingCostHelper(const frBox& testBox,
                                             frLayerNum layerNum, int eolType,
                                             int type) {
    if (xDim == 0 || yDim == 0 || zDim == 0 || !hasMazeZCoord(layerNum)) {
        return;
    }

    // eolType 0 checks planar candidates on layerNum. eolType 1/2 checks down
    // and up via candidates whose via enclosure overlaps the EOL test window.
    auto z = getCoordIdx(zCoords, layerNum);
    if (z < 0) {
        return;
    }

    auto tech = getTech();
    auto layer = tech ? tech->getLayer(layerNum) : nullptr;
    if (!tech || !layer) {
        return;
    }

    frBox searchBox;
    frBox viaBox;
    frDirEnum costDir = frDirEnum::E;
    if (eolType == 0) {
        auto halfWidth = layer->getWidth() / 2;
        searchBox.set(
            testBox.left() - halfWidth + 1, testBox.bottom() - halfWidth + 1,
            testBox.right() + halfWidth - 1, testBox.top() + halfWidth - 1);
    } else {
        frLayerNum cutLayerNum = (eolType == 1) ? layerNum - 1 : layerNum + 1;
        if (cutLayerNum < tech->getBottomLayerNum() ||
            cutLayerNum > tech->getTopLayerNum()) {
            return;
        }

        auto cutLayer = tech->getLayer(cutLayerNum);
        auto viaDef = cutLayer ? cutLayer->getDefaultViaDef() : nullptr;
        if (!viaDef) {
            return;
        }

        frVia via(viaDef);
        if (eolType == 1) {
            via.getLayer2BBox(viaBox);
            costDir = frDirEnum::D;
        } else {
            via.getLayer1BBox(viaBox);
            costDir = frDirEnum::U;
        }
        searchBox.set(testBox.left() - viaBox.right() + 1,
                      testBox.bottom() - viaBox.top() + 1,
                      testBox.right() - viaBox.left() - 1,
                      testBox.top() - viaBox.bottom() - 1);
    }

    auto xBegin =
        std::lower_bound(xCoords.begin(), xCoords.end(), searchBox.left());
    auto xEnd =
        std::upper_bound(xCoords.begin(), xCoords.end(), searchBox.right());
    auto yBegin =
        std::lower_bound(yCoords.begin(), yCoords.end(), searchBox.bottom());
    auto yEnd =
        std::upper_bound(yCoords.begin(), yCoords.end(), searchBox.top());

    frTransform xform;
    frBox shiftedViaBox;
    frVia sVia;
    frBox sViaBox;
    for (auto xIt = xBegin; xIt != xEnd; ++xIt) {
        auto xIdx = static_cast<crIndex_t>(xIt - xCoords.begin());
        for (auto yIt = yBegin; yIt != yEnd; ++yIt) {
            auto yIdx = static_cast<crIndex_t>(yIt - yCoords.begin());
            crMazeType node(xIdx, yIdx, z);
            if (eolType == 1) {
                --node.z;
            }
            if (!isValidMazeIdx(node)) {
                continue;
            }

            if (eolType != 0) {
                auto pt = frPoint(*xIt, *yIt);
                xform.set(pt);
                shiftedViaBox.set(viaBox);
                if (auto sViaDef = getSViaDef(node)) {
                    sVia.setViaDef(sViaDef);
                    if (eolType == 1) {
                        sVia.getLayer2BBox(sViaBox);
                    } else {
                        sVia.getLayer1BBox(sViaBox);
                    }
                    shiftedViaBox.set(sViaBox);
                }
                shiftedViaBox.transform(xform);
                if (!shiftedViaBox.overlaps(testBox, false)) {
                    continue;
                }
            }

            modTypedCost(node, costDir, type);
        }
    }
}

void crPatternGraph::modEolSpacingCost(const frBox& srcBox, frLayerNum layerNum,
                                       int type, bool skipVia) {
    auto tech = getTech();
    auto layer = tech ? tech->getLayer(layerNum) : nullptr;
    if (!layer || !layer->hasEolSpacing()) {
        return;
    }

    // Flow:
    // 1. For each EOL rule, derive windows on short-width or short-length ends.
    // 2. Mark planar candidates in the EOL windows.
    // 3. Optionally mark via candidates whose enclosures overlap the windows.
    frBox testBox;
    for (auto con : layer->getEolSpacing()) {
        auto eolSpace = con->getMinSpacing();
        auto eolWidth = con->getEolWidth();
        auto eolWithin = con->getEolWithin();

        if (srcBox.width() < eolWidth) {
            testBox.set(srcBox.left() - eolWithin, srcBox.top(),
                        srcBox.right() + eolWithin, srcBox.top() + eolSpace);
            modEolSpacingCostHelper(testBox, layerNum, 0, type);
            if (!skipVia) {
                modEolSpacingCostHelper(testBox, layerNum, 1, type);
                modEolSpacingCostHelper(testBox, layerNum, 2, type);
            }

            testBox.set(srcBox.left() - eolWithin, srcBox.bottom() - eolSpace,
                        srcBox.right() + eolWithin, srcBox.bottom());
            modEolSpacingCostHelper(testBox, layerNum, 0, type);
            if (!skipVia) {
                modEolSpacingCostHelper(testBox, layerNum, 1, type);
                modEolSpacingCostHelper(testBox, layerNum, 2, type);
            }
        }

        if (srcBox.length() < eolWidth) {
            testBox.set(srcBox.right(), srcBox.bottom() - eolWithin,
                        srcBox.right() + eolSpace, srcBox.top() + eolWithin);
            modEolSpacingCostHelper(testBox, layerNum, 0, type);
            if (!skipVia) {
                modEolSpacingCostHelper(testBox, layerNum, 1, type);
                modEolSpacingCostHelper(testBox, layerNum, 2, type);
            }

            testBox.set(srcBox.left() - eolSpace, srcBox.bottom() - eolWithin,
                        srcBox.left(), srcBox.top() + eolWithin);
            modEolSpacingCostHelper(testBox, layerNum, 0, type);
            if (!skipVia) {
                modEolSpacingCostHelper(testBox, layerNum, 1, type);
                modEolSpacingCostHelper(testBox, layerNum, 2, type);
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

void crPatternGraph::modFrObjCost(frBlockObject* obj, int type) {
    if (!obj || !isExternalObject(obj)) {
        return;
    }

    // Convert existing DB routing objects into graph quick-cost influence.
    // The caller decides whether the affected region updates DRC or shape bits.
    if (obj->typeId() == frcPathSeg || obj->typeId() == frcRect ||
        obj->typeId() == frcPatchWire) {
        auto* shape = static_cast<frShape*>(obj);
        frBox box;
        shape->getBBox(box);
        modMetalShapeAllCost(box, shape->getLayerNum(), type);
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
        modMetalShapeAllCost(box, viaDef->getLayer1Num(), type);
        via->getLayer2BBox(box);
        modMetalShapeAllCost(box, viaDef->getLayer2Num(), type);
        via->getCutBBox(box);
        modViaShapeCost(box, viaDef->getLayer1Num(), type);
    }
}

void crPatternGraph::initExternalDRCCost() {
    auto design = getDesign();
    if (!design || !worker) {
        return;
    }

    // Query all existing DB route objects in extBox and initialize graph quick
    // cost for objects that do not belong to the CR nets currently being
    // routed.
    std::vector<frBlockObject*> result;
    design->getRegionQuery()->queryDRObj(worker->getExtBox(), result);
    for (auto* obj : result) {
        modFrObjCost(obj, 1);
    }
}

void crPatternGraph::initDRCCost() {
    // Flow:
    // 1. Clear DRC and shape counters while preserving other cost channels.
    // 2. Rebuild quick cost from existing DB routing in the worker extBox.
    for (auto& word : bits) {
        clearDrcBits(word);
        clearShapeBits(word);
    }
    initExternalDRCCost();
}

void crPatternGraph::addPathCost(const crConnFig* connFig) {
    modPathCost(connFig, 1);
}

void crPatternGraph::subPathCost(const crConnFig* connFig) {
    modPathCost(connFig, 0);
}

void crPatternGraph::modPathCost(const crConnFig* connFig, int type) {
    if (!connFig) {
        return;
    }

    if (connFig->typeId() == crcPathSeg) {
        modPathSegCost(static_cast<const crPathSeg*>(connFig), type);
    } else if (connFig->typeId() == crcVia) {
        modViaCost(static_cast<const crVia*>(connFig), type);
    }
}

void crPatternGraph::modPathSegCost(const crPathSeg* pathSeg, int type) {
    if (!pathSeg || !pathSeg->hasMazeIdx()) {
        return;
    }

    // Route path segments affect planar spacing, adjacent via candidates, and
    // EOL windows when the segment follows the preferred routing direction.
    modMetalShapeAllCost(pathSeg->getBBox(), pathSeg->getLayerNum(), type);
    auto layer =
        getTech() ? getTech()->getLayer(pathSeg->getLayerNum()) : nullptr;
    if (layer) {
        auto begin = pathSeg->getBegin();
        auto end = pathSeg->getEnd();
        auto isHorizontal = begin.y() == end.y();
        auto isHLayer = layer->getDir() == frcHorzPrefRoutingDir;
        if (isHLayer == isHorizontal) {
            modEolSpacingCost(pathSeg->getBBox(), pathSeg->getLayerNum(), type);
        }
    }
}

void crPatternGraph::modViaCost(const crVia* via, int type) {
    if (!via || !via->getViaDef()) {
        return;
    }

    // A via contributes quick cost through both metal enclosure boxes and the
    // cut-box occupancy point on the lower routing layer.
    auto* viaDef = via->getViaDef();
    modMetalShapeAllCost(via->getLowerLayerFigBBox(), viaDef->getLayer1Num(),
                         type);
    modEolSpacingCost(via->getLowerLayerFigBBox(), viaDef->getLayer1Num(),
                      type);
    modMetalShapeAllCost(via->getUpperLayerFigBBox(), viaDef->getLayer2Num(),
                         type);
    modEolSpacingCost(via->getUpperLayerFigBBox(), viaDef->getLayer2Num(),
                      type);
    modViaShapeCost(via->getCutFigBBox(), viaDef->getLayer1Num(), type);
}

void crPatternGraph::build(CustomRouteWorker* worker) {
    clear();
    this->worker = worker;
}

}  // namespace fr
