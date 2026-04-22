#include "crMazeRouter.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>
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

bool crMazeRouter::searchPath() {
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

bool crMazeRouter::searchPathL() {
    if (!graph || !net || net->getPins().size() != 2) {
        return false;
    }

    std::vector<crMazeType> srcs;
    std::vector<crMazeType> dsts;
    if (!initEndpoints(srcs, dsts)) {
        return false;
    }

    return runDijkstra(srcs, dsts);
}

bool crMazeRouter::initEndpoints(std::vector<crMazeType>& srcs,
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
        if (graph->hasNode(mazeIdx)) {
            srcSet.insert(mazeIdx);
        }
    }
    for (auto& ap : dstAps) {
        auto pt = ap->getPt();
        auto layerNum = ap->getLayerIdx();
        if (!graph->hasMazeIdx(pt.x(), pt.y(), layerNum)) {
            continue;
        }
        auto mazeIdx = graph->getMazeIdx(pt.x(), pt.y(), layerNum);
        if (graph->hasNode(mazeIdx)) {
            dstSet.insert(mazeIdx);
        }
    }

    srcs.assign(srcSet.begin(), srcSet.end());
    dsts.assign(dstSet.begin(), dstSet.end());
    return !srcs.empty() && !dsts.empty();
}

frCoord crMazeRouter::getNextPathCost(const Wavefront& curr,
                                      const SearchState& nextState,
                                      frDirEnum dir) const {
    if (!graph) {
        return curr.cost;
    }

    auto nextCost = curr.cost;
    auto& currNode = curr.state.node;
    auto& nextNode = nextState.node;
    if (dir == frDirEnum::U || dir == frDirEnum::D) {
        nextCost += getViaEdgeCost(currNode, nextNode);
    } else {
        auto edgeCost = getPlanarEdgeCost(currNode, nextNode);
        if (graph->getLayerNum(currNode) == BOTTOM_ROUTING_LAYER) {
            edgeCost *= bottomLayerPenaltyCoeff;
        }
        nextCost += edgeCost;
        if (graph->hasNonPrefCost(currNode, dir)) {
            nextCost += edgeCost * CR_NONPREF_ROUTE_PENALTY;
        }
        if (graph->hasDRCCost(currNode, dir)) {
            nextCost += edgeCost * CR_SPACING_DRC_PENALTY;
        }
        if (policy == crPatternEnum::L && isTurn(curr.state.lastDir, dir) &&
            curr.state.turnCount >= 1) {
            nextCost += CR_L_SHAPE_EXTRA_TURN_PENALTY;
        }
    }
    return nextCost;
}

frCoord crMazeRouter::getPlanarEdgeCost(const crMazeType& curr,
                                        const crMazeType& next) const {
    auto currPt = graph->getPoint(curr);
    auto nextPt = graph->getPoint(next);
    return std::abs(currPt.x() - nextPt.x()) +
           std::abs(currPt.y() - nextPt.y());
}

frCoord crMazeRouter::getViaEdgeCost(const crMazeType& curr,
                                     const crMazeType& next) const {
    if (!graph) {
        return 0;
    }

    auto tech = graph->getTech();
    if (!tech) {
        return 0;
    }

    auto lowerLayerNum = std::min(graph->getLayerNum(curr), graph->getLayerNum(next));
    auto lowerLayer = tech->getLayer(lowerLayerNum);
    if (!lowerLayer) {
        return 0;
    }
    return lowerLayer->getPitch() * VIACOST;
}

crMazeRouter::SearchState crMazeRouter::getNextState(
    const SearchState& curr, const crMazeType& next, frDirEnum dir) const {
    auto nextTurnCount = curr.turnCount;
    auto nextLastDir = curr.lastDir;
    if (isPlanarDir(dir)) {
        if (isTurn(curr.lastDir, dir)) {
            nextTurnCount = std::min(nextTurnCount + 1, maxTrackedTurnCount);
        }
        nextLastDir = dir;
    }
    return {next, nextLastDir, nextTurnCount};
}

std::uint64_t crMazeRouter::getStateKey(const SearchState& state) const {
    auto nodeKey = graph->getNodeKey(state.node);
    auto dirKey = static_cast<std::uint64_t>(state.lastDir);
    auto turnKey = static_cast<std::uint64_t>(
        std::min(state.turnCount, maxTrackedTurnCount));
    return (nodeKey * 7 + dirKey) * (maxTrackedTurnCount + 1) + turnKey;
}

bool crMazeRouter::isPlanarDir(frDirEnum dir) const {
    return dir == frDirEnum::E || dir == frDirEnum::W ||
           dir == frDirEnum::N || dir == frDirEnum::S;
}

bool crMazeRouter::isTurn(frDirEnum currDir, frDirEnum nextDir) const {
    return isPlanarDir(currDir) && isPlanarDir(nextDir) &&
           currDir != nextDir;
}

bool crMazeRouter::runDijkstra(const std::vector<crMazeType>& srcs,
                               const std::vector<crMazeType>& dsts) {
    std::priority_queue<Wavefront, std::vector<Wavefront>, WavefrontComp>
        pq;

    std::unordered_map<std::uint64_t, frCoord> dist;
    std::unordered_map<std::uint64_t, SearchState> prev;
    std::unordered_map<std::uint64_t, bool> isDst;

    for (auto& dst : dsts) {
        isDst[graph->getNodeKey(dst)] = true;
    }
    for (auto& src : srcs) {
        SearchState state{src, frDirEnum::UNKNOWN, 0};
        auto key = getStateKey(state);
        dist[key] = 0;
        pq.push({0, state});
    }

    SearchState foundDst;
    bool found = false;

    while (!pq.empty()) {
        auto wf = pq.top();
        pq.pop();
        auto currDist = wf.cost;
        auto curr = wf.state.node;
        auto currKey = getStateKey(wf.state);
        auto distIt = dist.find(currKey);
        if (distIt == dist.end() || currDist != distIt->second) {
            continue;
        }
        if (isDst.find(graph->getNodeKey(curr)) != isDst.end()) {
            foundDst = wf.state;
            found = true;
            break;
        }

        for (auto dir : {frDirEnum::E, frDirEnum::W, frDirEnum::N,
                         frDirEnum::S, frDirEnum::U, frDirEnum::D}) {
            crMazeType neighbor;
            if (!graph->getNextMazeIdx(curr, dir, neighbor)) {
                continue;
            }
            auto nextState = getNextState(wf.state, neighbor, dir);
            auto nextDist = getNextPathCost(wf, nextState, dir);
            auto neighborKey = getStateKey(nextState);
            auto nextIt = dist.find(neighborKey);
            if (nextIt == dist.end() || nextDist < nextIt->second) {
                dist[neighborKey] = nextDist;
                prev[neighborKey] = wf.state;
                pq.push({nextDist, nextState});
            }
        }
    }

    if (!found) {
        return false;
    }

    path.clear();
    auto curr = foundDst;
    while (true) {
        path.push_back(curr.node);
        auto currKey = getStateKey(curr);
        auto prevIt = prev.find(currKey);
        if (prevIt == prev.end()) {
            break;
        }
        curr = prevIt->second;
    }
    std::reverse(path.begin(), path.end());

    return !path.empty();
}

void crMazeRouter::printPath() const {
    std::cout << "crMazeRouter path";
    if (net && net->getNet()) {
        std::cout << " net=" << net->getNet()->getName();
    }
    std::cout << " size=" << path.size() << std::endl;

    if (net) {
        for (std::size_t pinIdx = 0; pinIdx < net->getPins().size();
             ++pinIdx) {
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
                    std::cout << " (mi " << mazeIdx.x << " " << mazeIdx.y
                              << " " << mazeIdx.z << ")";
                    if (!graph->hasNode(mazeIdx)) {
                        std::cout << " notInGraph";
                    }
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

    frDirEnum prevPlanarDir = frDirEnum::UNKNOWN;
    int turnCount = 0;
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

            if (isPlanarDir(stepDir)) {
                if (isTurn(prevPlanarDir, stepDir)) {
                    ++turnCount;
                }
                prevPlanarDir = stepDir;
            }
        }
        std::cout << "  (mi " << mazeIdx.x << " " << mazeIdx.y << " "
                  << mazeIdx.z << ")"
                  << " (x " << pt.x() << " y " << pt.y() << " l "
                  << layerNum << ")"
                  << " dir=" << getDirName(stepDir)
                  << " turnCount=" << turnCount;
        if (i > 0 && graph->hasDRCCost(path[i - 1], stepDir)) {
            std::cout << " drcCost=1";
        }
        std::cout << std::endl;
    }
}

}  // namespace fr
