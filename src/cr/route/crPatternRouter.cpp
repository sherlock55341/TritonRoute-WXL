#include "crPatternRouter.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include "cr/crConfig.hpp"
#include "global.h"

namespace fr {

namespace {

const char* getDirName(frDirEnum dir) {
    switch (dir) {
        case frDirEnum::E:
            return "E";
        case frDirEnum::W:
            return "W";
        case frDirEnum::N:
            return "N";
        case frDirEnum::S:
            return "S";
        case frDirEnum::U:
            return "U";
        case frDirEnum::D:
            return "D";
        default:
            return "UNKNOWN";
    }
}

}  // namespace

bool crPatternRouter::searchPath() {
    path.clear();
    bool found = false;
    switch (policy) {
        case crPatternEnum::L:
            found = searchPathL();
            break;
        default:
            found = false;
            break;
    }
    if (found) {
        printPath();
    }
    return found;
}

bool crPatternRouter::searchPathL() {
    if (!graph || !net || net->getPins().size() != 2) {
        return false;
    }

    std::vector<crMazeType> srcs;
    std::vector<crMazeType> dsts;
    if (!initEndpoints(srcs, dsts)) {
        return false;
    }

    frCoord bestCost = std::numeric_limits<frCoord>::max();
    std::vector<crMazeType> bestPath;
    std::set<std::vector<crMazeType> > seenPaths;

    for (const auto& src : srcs) {
        for (const auto& dst : dsts) {
            auto srcPt = graph->getPoint(src);
            auto dstPt = graph->getPoint(dst);
            for (auto routeLayerNum : graph->getZCoords()) {
                if (!graph->hasMazeIdx(srcPt.x(), srcPt.y(), routeLayerNum) ||
                    !graph->hasMazeIdx(dstPt.x(), dstPt.y(), routeLayerNum)) {
                    continue;
                }

                auto srcRoute =
                    graph->getMazeIdx(srcPt.x(), srcPt.y(), routeLayerNum);
                auto dstRoute =
                    graph->getMazeIdx(dstPt.x(), dstPt.y(), routeLayerNum);

                for (const auto& mid : {crMazeType(src.x, dst.y, srcRoute.z),
                                        crMazeType(dst.x, src.y, srcRoute.z)}) {
                    std::vector<crMazeType> candidate;
                    if (!buildLPath(src, srcRoute, mid, dstRoute, dst,
                                    candidate) ||
                        candidate.size() < 2) {
                        continue;
                    }
                    if (!seenPaths.insert(candidate).second) {
                        continue;
                    }

                    auto cost = getPathCost(candidate);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestPath = std::move(candidate);
                    }
                }
            }
        }
    }

    if (bestPath.empty()) {
        return false;
    }

    path = std::move(bestPath);
    return true;
}

bool crPatternRouter::initEndpoints(std::vector<crMazeType>& srcs,
                                    std::vector<crMazeType>& dsts) const {
    srcs.clear();
    dsts.clear();
    std::set<crMazeType> srcSet;
    std::set<crMazeType> dstSet;

    const auto& srcAps = net->getPins()[0]->getAccessPoints();
    const auto& dstAps = net->getPins()[1]->getAccessPoints();
    for (auto& ap : srcAps) {
        auto pt = ap->getPt();
        auto layerNum = ap->getLayerIdx();
        if (!graph->hasMazeIdx(pt.x(), pt.y(), layerNum)) {
            continue;
        }
        auto mazeIdx = graph->getMazeIdx(pt.x(), pt.y(), layerNum);
        srcSet.insert(mazeIdx);
    }
    for (auto& ap : dstAps) {
        auto pt = ap->getPt();
        auto layerNum = ap->getLayerIdx();
        if (!graph->hasMazeIdx(pt.x(), pt.y(), layerNum)) {
            continue;
        }
        auto mazeIdx = graph->getMazeIdx(pt.x(), pt.y(), layerNum);
        dstSet.insert(mazeIdx);
    }

    srcs.assign(srcSet.begin(), srcSet.end());
    dsts.assign(dstSet.begin(), dstSet.end());
    return !srcs.empty() && !dsts.empty();
}

frCoord crPatternRouter::getSegmentCost(const crMazeType& begin,
                                        const crMazeType& end) const {
    if (!graph) {
        return 0;
    }

    if (begin == end) {
        return 0;
    }
    if (begin.x == end.x && begin.y == end.y && begin.z != end.z) {
        return getViaSegmentCost(begin, end);
    }
    return getPlanarSegmentCost(begin, end);
}

frCoord crPatternRouter::getPlanarSegmentCost(const crMazeType& begin,
                                              const crMazeType& end) const {
    if (!graph || begin.z != end.z || (begin.x != end.x && begin.y != end.y)) {
        return std::numeric_limits<frCoord>::max();
    }

    auto beginPt = graph->getPoint(begin);
    auto endPt = graph->getPoint(end);
    auto cost =
        std::abs(beginPt.x() - endPt.x()) + std::abs(beginPt.y() - endPt.y());
    if (graph->getLayerNum(begin) == BOTTOM_ROUTING_LAYER) {
        cost *= bottomLayerPenaltyCoeff;
    }

    auto dir = frDirEnum::UNKNOWN;
    if (begin.x < end.x) {
        dir = frDirEnum::E;
    } else if (begin.x > end.x) {
        dir = frDirEnum::W;
    } else if (begin.y < end.y) {
        dir = frDirEnum::N;
    } else if (begin.y > end.y) {
        dir = frDirEnum::S;
    }

    if (dir != frDirEnum::UNKNOWN && graph->hasNonPrefCost(begin, dir)) {
        cost += cost * CR_NONPREF_ROUTE_PENALTY;
    }

    for (auto curr = begin; !(curr == end);) {
        crMazeType next;
        if (!graph->getNextMazeIdx(curr, dir, next)) {
            return std::numeric_limits<frCoord>::max();
        }
        if (graph->hasDRCCost(curr, dir)) {
            cost += CR_SPACING_DRC_PENALTY;
        }
        curr = next;
    }
    return cost;
}

bool crPatternRouter::buildLPath(const crMazeType& src,
                                 const crMazeType& srcRoute,
                                 const crMazeType& mid,
                                 const crMazeType& dstRoute,
                                 const crMazeType& dst,
                                 std::vector<crMazeType>& candidate) const {
    candidate.clear();
    if (!graph || !graph->hasNode(src) || !graph->hasNode(srcRoute) ||
        !graph->hasNode(mid) || !graph->hasNode(dstRoute) ||
        !graph->hasNode(dst)) {
        return false;
    }

    if (!appendViaSegment(src, srcRoute, candidate)) {
        return false;
    }
    if (!appendStraightSegment(srcRoute, mid, candidate)) {
        return false;
    }
    if (!appendStraightSegment(mid, dstRoute, candidate)) {
        return false;
    }
    if (!appendViaSegment(dstRoute, dst, candidate)) {
        return false;
    }
    return candidate.size() >= 2;
}

bool crPatternRouter::appendStraightSegment(
    const crMazeType& begin, const crMazeType& end,
    std::vector<crMazeType>& candidate) const {
    if (!graph || !graph->hasNode(begin) || !graph->hasNode(end)) {
        return false;
    }
    if (begin.z != end.z) {
        return false;
    }
    if (begin.x != end.x && begin.y != end.y) {
        return false;
    }

    if (candidate.empty()) {
        candidate.push_back(begin);
    } else if (!(candidate.back() == begin)) {
        return false;
    }

    if (begin == end) {
        return true;
    }

    auto curr = begin;
    frDirEnum dir = frDirEnum::UNKNOWN;
    if (begin.x < end.x) {
        dir = frDirEnum::E;
    } else if (begin.x > end.x) {
        dir = frDirEnum::W;
    } else if (begin.y < end.y) {
        dir = frDirEnum::N;
    } else if (begin.y > end.y) {
        dir = frDirEnum::S;
    } else {
        return true;
    }

    while (!(curr == end)) {
        crMazeType next;
        if (!graph->getNextMazeIdx(curr, dir, next)) {
            return false;
        }
        candidate.push_back(next);
        curr = next;
    }
    return true;
}

bool crPatternRouter::appendViaSegment(
    const crMazeType& begin, const crMazeType& end,
    std::vector<crMazeType>& candidate) const {
    if (!graph || !graph->hasNode(begin) || !graph->hasNode(end)) {
        return false;
    }
    if (begin.x != end.x || begin.y != end.y) {
        return false;
    }

    if (candidate.empty()) {
        candidate.push_back(begin);
    } else if (!(candidate.back() == begin)) {
        return false;
    }

    if (begin == end) {
        return true;
    }

    auto curr = begin;
    auto dir = (begin.z < end.z) ? frDirEnum::U : frDirEnum::D;
    while (!(curr == end)) {
        crMazeType next;
        if (!graph->getNextMazeIdx(curr, dir, next)) {
            return false;
        }
        candidate.push_back(next);
        curr = next;
    }
    return true;
}

frCoord crPatternRouter::getPathCost(
    const std::vector<crMazeType>& candidate) const {
    if (!graph || candidate.size() < 2) {
        return std::numeric_limits<frCoord>::max();
    }

    frCoord cost = 0;
    std::size_t segBegin = 0;
    for (std::size_t i = 2; i <= candidate.size(); ++i) {
        auto shouldFlush = (i == candidate.size());
        if (!shouldFlush) {
            const auto& prev = candidate[i - 2];
            const auto& curr = candidate[i - 1];
            const auto& next = candidate[i];
            auto samePlanarLine = (prev.z == curr.z && curr.z == next.z &&
                                   ((prev.x == curr.x && curr.x == next.x) ||
                                    (prev.y == curr.y && curr.y == next.y)));
            auto sameViaColumn = (prev.x == curr.x && curr.x == next.x &&
                                  prev.y == curr.y && curr.y == next.y);
            shouldFlush = !(samePlanarLine || sameViaColumn);
        }

        if (shouldFlush) {
            auto segCost =
                getSegmentCost(candidate[segBegin], candidate[i - 1]);
            if (segCost == std::numeric_limits<frCoord>::max()) {
                return segCost;
            }
            cost += segCost;
            segBegin = i - 1;
        }
    }
    return cost;
}

frCoord crPatternRouter::getViaSegmentCost(const crMazeType& begin,
                                           const crMazeType& end) const {
    if (!graph) {
        return 0;
    }
    if (begin.x != end.x || begin.y != end.y || begin.z == end.z) {
        return std::numeric_limits<frCoord>::max();
    }

    auto tech = graph->getTech();
    if (!tech) {
        return 0;
    }

    frCoord cost = 0;
    auto curr = begin;
    auto dir = (begin.z < end.z) ? frDirEnum::U : frDirEnum::D;
    while (!(curr == end)) {
        crMazeType next;
        if (!graph->getNextMazeIdx(curr, dir, next)) {
            return std::numeric_limits<frCoord>::max();
        }

        auto lowerLayerNum =
            std::min(graph->getLayerNum(curr), graph->getLayerNum(next));
        auto lowerLayer = tech->getLayer(lowerLayerNum);
        if (!lowerLayer) {
            return std::numeric_limits<frCoord>::max();
        }
        cost += lowerLayer->getPitch() * VIACOST;
        curr = next;
    }
    return cost;
}

bool crPatternRouter::isPlanarDir(frDirEnum dir) const {
    return dir == frDirEnum::E || dir == frDirEnum::W || dir == frDirEnum::N ||
           dir == frDirEnum::S;
}

void crPatternRouter::printPath() const {
    std::cout << "crPatternRouter path";
    if (net && net->getNet()) {
        std::cout << " net=" << net->getNet()->getName();
    }
    std::cout << " size=" << path.size() << std::endl;

    if (net) {
        for (std::size_t pinIdx = 0; pinIdx < net->getPins().size(); ++pinIdx) {
            auto pin = net->getPins()[pinIdx].get();
            std::cout << "  pin " << pinIdx << " aps";
            if (pin->getFrTerm()) {
                std::cout << " termType=" << pin->getFrTerm()->typeId();
            }
            std::cout << " count=" << pin->getAccessPoints().size()
                      << std::endl;
            for (std::size_t apIdx = 0; apIdx < pin->getAccessPoints().size();
                 ++apIdx) {
                auto ap = pin->getAccessPoints()[apIdx].get();
                auto pt = ap->getPt();
                auto layerNum = ap->getLayerIdx();
                std::cout << "    ap " << apIdx << " (x " << pt.x() << " y "
                          << pt.y() << " l " << layerNum << ")";
                if (graph && graph->hasMazeIdx(pt.x(), pt.y(), layerNum)) {
                    auto mazeIdx = graph->getMazeIdx(pt.x(), pt.y(), layerNum);
                    std::cout << " (mi " << mazeIdx.x << " " << mazeIdx.y << " "
                              << mazeIdx.z << ")";
                } else {
                    std::cout << " noMazeIdx";
                }
                std::cout << std::endl;
            }
        }
    }

    for (auto& mazeIdx : path) {
        frDirEnum dir = frDirEnum::UNKNOWN;
        static_cast<void>(dir);
    }

    for (std::size_t i = 0; i < path.size(); ++i) {
        auto& mazeIdx = path[i];
        auto pt = graph->getPoint(mazeIdx);
        auto layerNum = graph->getLayerNum(mazeIdx);
        auto stepDir = frDirEnum::UNKNOWN;
        if (i > 0) {
            auto& prevMazeIdx = path[i - 1];
            if (mazeIdx.x > prevMazeIdx.x) {
                stepDir = frDirEnum::E;
            } else if (mazeIdx.x < prevMazeIdx.x) {
                stepDir = frDirEnum::W;
            } else if (mazeIdx.y > prevMazeIdx.y) {
                stepDir = frDirEnum::N;
            } else if (mazeIdx.y < prevMazeIdx.y) {
                stepDir = frDirEnum::S;
            } else if (mazeIdx.z > prevMazeIdx.z) {
                stepDir = frDirEnum::U;
            } else if (mazeIdx.z < prevMazeIdx.z) {
                stepDir = frDirEnum::D;
            }
        }
        std::cout << "  (mi " << mazeIdx.x << " " << mazeIdx.y << " "
                  << mazeIdx.z << ")" << " (x " << pt.x() << " y " << pt.y()
                  << " l " << layerNum << ")" << " dir=" << getDirName(stepDir);
        if (i > 0 && graph->hasDRCCost(path[i - 1], stepDir)) {
            std::cout << " drcCost=1";
        }
        std::cout << std::endl;
    }
}

}  // namespace fr
