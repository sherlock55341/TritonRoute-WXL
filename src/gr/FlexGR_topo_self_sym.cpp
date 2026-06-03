/* Authors: Lutong Wang and Bangqi Xu */
/*
 * Copyright (c) 2019, The Regents of the University of California
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include "FlexGR.h"
#include "gr/FlexGR_self_sym_utils.h"
#include "db/infra/frOrient.h"
#include "db/infra/frTransform.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace fr;

namespace {
  bool selfSymmetryPointOnSegment(const frPoint &segmentBegin,
                                  const frPoint &segmentEnd,
                                  const frPoint &point) {
    if (segmentBegin.x() == segmentEnd.x()) {
      if (point.x() != segmentBegin.x()) {
        return false;
      }
      return point.y() >= min(segmentBegin.y(), segmentEnd.y()) &&
             point.y() <= max(segmentBegin.y(), segmentEnd.y());
    }
    if (segmentBegin.y() == segmentEnd.y()) {
      if (point.y() != segmentBegin.y()) {
        return false;
      }
      return point.x() >= min(segmentBegin.x(), segmentEnd.x()) &&
             point.x() <= max(segmentBegin.x(), segmentEnd.x());
    }
    return false;
  }

  [[noreturn]] void failSelfSymmetryTopology(const string &message,
                                             int status = 1) {
    cout << message;
    if (message.empty() || message.back() != '\n') {
      cout << "\n";
    }
    exit(status);
  }
}

void FlexGR::genSelfSymmetryRootSideTopology(const vector<frPoint> &rootSideTerminalGCellIdxs,
                                             const frPoint &rootGCellIdx,
                                             bool isAxisHorizontal,
                                             frCoord axisGCellIdx,
                                             vector<frPoint> &rootSideTreeVertices,
                                             vector<pair<frPoint, frPoint> > &rootSideTreeEdges) {
  rootSideTreeVertices.clear();
  rootSideTreeEdges.clear();

  set<frCoord> xCoords;
  set<frCoord> yCoords;
  set<frPoint> terminals;

  for (auto gcellIdx: rootSideTerminalGCellIdxs) {
    xCoords.insert(gcellIdx.x());
    yCoords.insert(gcellIdx.y());
    terminals.insert(gcellIdx);
  }
  xCoords.insert(rootGCellIdx.x());
  yCoords.insert(rootGCellIdx.y());
  terminals.insert(rootGCellIdx);

  if (isAxisHorizontal) {
    yCoords.insert(axisGCellIdx);
  } else {
    xCoords.insert(axisGCellIdx);
  }

  const int LEFT = 0;
  const int RIGHT = 1;
  const int DOWN = 2;
  const int UP = 3;

  vector<frPoint> graphGCellIdxs;
  vector<array<int, 4> > graphNeighbors;
  vector<array<int, 4> > graphNeighborCosts;
  map<frPoint, int> gcellIdx2GraphIdx;

  for (auto y: yCoords) {
    for (auto x: xCoords) {
      frPoint gcellIdx(x, y);
      gcellIdx2GraphIdx[gcellIdx] = (int)graphGCellIdxs.size();
      graphGCellIdxs.push_back(gcellIdx);
      graphNeighbors.push_back({{-1, -1, -1, -1}});
      graphNeighborCosts.push_back({{0, 0, 0, 0}});
    }
  }

  set<frPoint> treeVertexSet;
  set<pair<frPoint, frPoint> > treeEdgeSet;

  auto appendVertex = [&](const frPoint &vertex) {
    if (treeVertexSet.insert(vertex).second) {
      rootSideTreeVertices.push_back(vertex);
    }
  };

  auto appendEdge = [&](const frPoint &begin, const frPoint &end) {
    if (begin == end) {
      return;
    }
    auto edge = normalizeSelfSymmetryEdge(begin, end);
    if (treeEdgeSet.insert(edge).second) {
      rootSideTreeEdges.push_back(edge);
    }
  };

  auto axisCtx = SelfSymmetryAxisContext::fromGCellAxisOnly(isAxisHorizontal,
                                                            axisGCellIdx);

  auto edgeCost = [&](const frPoint &begin, const frPoint &end) {
    auto length = getSelfSymmetryEdgeLen(begin, end);
    return length * (axisCtx.isAxisEdge(begin, end) ? 1 : 4);
  };

  auto setNeighbor = [&](int fromIdx, int direction, int toIdx) {
    graphNeighbors[fromIdx][direction] = toIdx;
    graphNeighborCosts[fromIdx][direction] = edgeCost(graphGCellIdxs[fromIdx], graphGCellIdxs[toIdx]);
  };

  for (auto y: yCoords) {
    int prevIdx = -1;
    for (auto x: xCoords) {
      int currIdx = gcellIdx2GraphIdx[frPoint(x, y)];
      if (prevIdx != -1) {
        setNeighbor(prevIdx, RIGHT, currIdx);
        setNeighbor(currIdx, LEFT, prevIdx);
      }
      prevIdx = currIdx;
    }
  }

  for (auto x: xCoords) {
    int prevIdx = -1;
    for (auto y: yCoords) {
      int currIdx = gcellIdx2GraphIdx[frPoint(x, y)];
      if (prevIdx != -1) {
        setNeighbor(prevIdx, UP, currIdx);
        setNeighbor(currIdx, DOWN, prevIdx);
      }
      prevIdx = currIdx;
    }
  }

  using WavefrontNode = pair<int, int>;
  priority_queue<WavefrontNode, vector<WavefrontNode>, greater<WavefrontNode> > wavefront;
  const int graphSize = (int)graphGCellIdxs.size();
  const int INF = numeric_limits<int>::max();
  vector<int> dist(graphSize, INF);
  vector<int> prev(graphSize, -1);
  vector<bool> inTree(graphSize, false);

  auto addTreeSource = [&](int graphIdx) {
    inTree[graphIdx] = true;
    dist[graphIdx] = 0;
    prev[graphIdx] = -1;
    wavefront.emplace(0, graphIdx);
  };

  auto searchNextTarget = [&](const function<bool(int)> &isTarget, vector<int> &path) {
    path.clear();

    while (!wavefront.empty()) {
      auto wavefrontNode = wavefront.top();
      wavefront.pop();

      int currCost = wavefrontNode.first;
      int currIdx = wavefrontNode.second;
      if (currCost != dist[currIdx]) {
        continue;
      }

      if (!inTree[currIdx] && isTarget(currIdx)) {
        int pathIdx = currIdx;
        path.push_back(pathIdx);
        while (!inTree[pathIdx]) {
          pathIdx = prev[pathIdx];
          if (pathIdx < 0) {
            return false;
          }
          path.push_back(pathIdx);
        }
        reverse(path.begin(), path.end());
        return true;
      }

      for (int dir = 0; dir < 4; dir++) {
        int nextIdx = graphNeighbors[currIdx][dir];
        if (nextIdx < 0) {
          continue;
        }
        int nextCost = currCost + graphNeighborCosts[currIdx][dir];
        if (nextCost < dist[nextIdx]) {
          dist[nextIdx] = nextCost;
          prev[nextIdx] = currIdx;
          wavefront.emplace(nextCost, nextIdx);
        }
      }
    }

    return false;
  };

  auto appendPath = [&](const vector<int> &path) {
    for (auto graphIdx: path) {
      appendVertex(graphGCellIdxs[graphIdx]);
    }
    for (int i = 1; i < (int)path.size(); i++) {
      appendEdge(graphGCellIdxs[path[i - 1]], graphGCellIdxs[path[i]]);
    }
    for (auto graphIdx: path) {
      addTreeSource(graphIdx);
    }
  };

  appendVertex(rootGCellIdx);
  int rootGraphIdx = gcellIdx2GraphIdx[rootGCellIdx];
  addTreeSource(rootGraphIdx);

  set<int> unconnectedTerminals;
  for (auto terminal: terminals) {
    int terminalIdx = gcellIdx2GraphIdx[terminal];
    if (terminalIdx != rootGraphIdx) {
      unconnectedTerminals.insert(terminalIdx);
    }
  }
  while (!unconnectedTerminals.empty()) {
    vector<int> path;
    if (!searchNextTarget([&](int graphIdx) { return unconnectedTerminals.find(graphIdx) != unconnectedTerminals.end(); },
                          path)) {
      failSelfSymmetryTopology(
          "Error: failed to find root-side self-symmetry Hanan path");
    }
    appendPath(path);
    for (auto graphIdx: path) {
      unconnectedTerminals.erase(graphIdx);
    }
  }

  bool intersectsAxis = false;
  for (int graphIdx = 0; graphIdx < graphSize; graphIdx++) {
    if (inTree[graphIdx] &&
        axisCtx.axisCoord(graphGCellIdxs[graphIdx]) == axisGCellIdx) {
      intersectsAxis = true;
      break;
    }
  }

  if (!intersectsAxis) {
    vector<bool> isAxisTarget(graphSize, false);
    for (int graphIdx = 0; graphIdx < graphSize; graphIdx++) {
      isAxisTarget[graphIdx] =
          axisCtx.axisCoord(graphGCellIdxs[graphIdx]) == axisGCellIdx;
    }

    vector<int> path;
    if (!searchNextTarget([&](int graphIdx) { return isAxisTarget[graphIdx]; }, path)) {
      failSelfSymmetryTopology(
          "Error: failed to connect root-side self-symmetry tree to axis gcell");
    }
    appendPath(path);
  }
}

void FlexGR::genSelfSymmetryOppositeSideTopology(const vector<frPoint> &oppositeSideTerminalGCellIdxs,
                                                 bool isAxisHorizontal,
                                                 frCoord axisGCellIdx,
                                                 int rootSide,
                                                 const vector<frPoint> &rootSideTreeVertices,
                                                 const vector<pair<frPoint, frPoint> > &rootSideTreeEdges,
                                                 vector<frPoint> &oppositeSideTreeVertices,
                                                 vector<pair<frPoint, frPoint> > &oppositeSideTreeEdges) {
  oppositeSideTreeVertices.clear();
  oppositeSideTreeEdges.clear();

  if (oppositeSideTerminalGCellIdxs.empty()) {
    return;
  }

  const int LEFT = 0;
  const int RIGHT = 1;
  const int DOWN = 2;
  const int UP = 3;

  auto axisCtx = SelfSymmetryAxisContext::fromGCellAxisOnly(isAxisHorizontal,
                                                            axisGCellIdx,
                                                            rootSide);

  auto isAllowedPoint = [&](const frPoint &gcellIdx) {
    int side = axisCtx.sideOfGCell(gcellIdx);
    return side == 0 || side != rootSide;
  };

  auto mirrorPoint = [&](const frPoint &gcellIdx) {
    return axisCtx.mirrorGCell(gcellIdx);
  };

  set<frCoord> xCoords;
  set<frCoord> yCoords;
  set<frPoint> terminals;
  set<frPoint> explicitAxisSources;
  vector<pair<frPoint, frPoint> > mirroredRootSideEdges;

  auto addGraphCoord = [&](const frPoint &gcellIdx) {
    if (!isAllowedPoint(gcellIdx)) {
      return;
    }
    xCoords.insert(gcellIdx.x());
    yCoords.insert(gcellIdx.y());
  };

  for (auto terminal: oppositeSideTerminalGCellIdxs) {
    if (!isAllowedPoint(terminal)) {
      failSelfSymmetryTopology(
          "Error: opposite-side self-symmetry terminal crosses into root side");
    }
    terminals.insert(terminal);
    addGraphCoord(terminal);
  }

  if (isAxisHorizontal) {
    yCoords.insert(axisGCellIdx);
  }
  else {
    xCoords.insert(axisGCellIdx);
  }

  for (auto rootSideVertex: rootSideTreeVertices) {
    if (axisCtx.axisCoord(rootSideVertex) == axisGCellIdx) {
      explicitAxisSources.insert(rootSideVertex);
      addGraphCoord(rootSideVertex);
    }

    auto mirroredVertex = mirrorPoint(rootSideVertex);
    addGraphCoord(mirroredVertex);
  }

  for (auto rootSideEdge: rootSideTreeEdges) {
    if (axisCtx.isAxisEdge(rootSideEdge.first, rootSideEdge.second)) {
      addGraphCoord(rootSideEdge.first);
      addGraphCoord(rootSideEdge.second);
    }

    auto mirroredBegin = mirrorPoint(rootSideEdge.first);
    auto mirroredEnd = mirrorPoint(rootSideEdge.second);
    if (mirroredBegin == mirroredEnd ||
        !isAllowedPoint(mirroredBegin) ||
        !isAllowedPoint(mirroredEnd)) {
      continue;
    }

    mirroredRootSideEdges.push_back(
        normalizeSelfSymmetryEdge(mirroredBegin, mirroredEnd));
    addGraphCoord(mirroredBegin);
    addGraphCoord(mirroredEnd);
  }

  vector<frPoint> graphGCellIdxs;
  vector<array<int, 4> > graphNeighbors;
  vector<array<int, 4> > graphNeighborCosts;
  map<frPoint, int> gcellIdx2GraphIdx;

  for (auto y: yCoords) {
    for (auto x: xCoords) {
      frPoint gcellIdx(x, y);
      if (!isAllowedPoint(gcellIdx)) {
        continue;
      }
      gcellIdx2GraphIdx[gcellIdx] = (int)graphGCellIdxs.size();
      graphGCellIdxs.push_back(gcellIdx);
      graphNeighbors.push_back({{-1, -1, -1, -1}});
      graphNeighborCosts.push_back({{0, 0, 0, 0}});
    }
  }

  auto isMirrorRewardEdge = [&](const frPoint &begin, const frPoint &end) {
    for (auto mirroredRootSideEdge: mirroredRootSideEdges) {
      if (selfSymmetrySegmentCovers(mirroredRootSideEdge.first,
                                    mirroredRootSideEdge.second,
                                    begin,
                                    end)) {
        return true;
      }
    }
    return false;
  };

  auto edgeCost = [&](const frPoint &begin, const frPoint &end) {
    auto length = getSelfSymmetryEdgeLen(begin, end);
    int cost = length * (isMirrorRewardEdge(begin, end) ? 1 : 8);
    if (cmap2D) {
      if (begin.y() == end.y()) {
        int yIdx = begin.y();
        for (int xIdx = min(begin.x(), end.x()); xIdx < max(begin.x(), end.x()); xIdx++) {
          auto supply = cmap2D->getRawSupply(xIdx, yIdx, 0, frDirEnum::E);
          auto demand = cmap2D->getRawDemand(xIdx, yIdx, 0, frDirEnum::E);
          cost += cmap2D->getHistoryCost(xIdx, yIdx, 0);
          if (cmap2D->hasBlock(xIdx, yIdx, 0, frDirEnum::E)) {
            cost += BLOCKCOST;
          }
          if (demand >= supply) {
            cost += MARKERCOST;
          } else if (demand > supply / 4) {
            cost += demand * 10 / (supply + 1);
          }
        }
      } else if (begin.x() == end.x()) {
        int xIdx = begin.x();
        for (int yIdx = min(begin.y(), end.y()); yIdx < max(begin.y(), end.y()); yIdx++) {
          auto supply = cmap2D->getRawSupply(xIdx, yIdx, 0, frDirEnum::N);
          auto demand = cmap2D->getRawDemand(xIdx, yIdx, 0, frDirEnum::N);
          cost += cmap2D->getHistoryCost(xIdx, yIdx, 0);
          if (cmap2D->hasBlock(xIdx, yIdx, 0, frDirEnum::N)) {
            cost += BLOCKCOST;
          }
          if (demand >= supply) {
            cost += MARKERCOST;
          } else if (demand > supply / 4) {
            cost += demand * 10 / (supply + 1);
          }
        }
      }
    }
    return cost;
  };

  auto setNeighbor = [&](int fromIdx, int direction, int toIdx) {
    graphNeighbors[fromIdx][direction] = toIdx;
    graphNeighborCosts[fromIdx][direction] = edgeCost(graphGCellIdxs[fromIdx], graphGCellIdxs[toIdx]);
  };

  for (auto y: yCoords) {
    int prevIdx = -1;
    for (auto x: xCoords) {
      auto currItr = gcellIdx2GraphIdx.find(frPoint(x, y));
      if (currItr == gcellIdx2GraphIdx.end()) {
        continue;
      }
      int currIdx = currItr->second;
      if (prevIdx != -1) {
        setNeighbor(prevIdx, RIGHT, currIdx);
        setNeighbor(currIdx, LEFT, prevIdx);
      }
      prevIdx = currIdx;
    }
  }

  for (auto x: xCoords) {
    int prevIdx = -1;
    for (auto y: yCoords) {
      auto currItr = gcellIdx2GraphIdx.find(frPoint(x, y));
      if (currItr == gcellIdx2GraphIdx.end()) {
        continue;
      }
      int currIdx = currItr->second;
      if (prevIdx != -1) {
        setNeighbor(prevIdx, UP, currIdx);
        setNeighbor(currIdx, DOWN, prevIdx);
      }
      prevIdx = currIdx;
    }
  }

  set<frPoint> treeVertexSet;
  set<pair<frPoint, frPoint> > treeEdgeSet;

  auto appendVertex = [&](const frPoint &vertex) {
    if (treeVertexSet.insert(vertex).second) {
      oppositeSideTreeVertices.push_back(vertex);
    }
  };

  auto appendEdge = [&](const frPoint &begin, const frPoint &end) {
    if (begin == end) {
      return;
    }
    auto edge = normalizeSelfSymmetryEdge(begin, end);
    if (treeEdgeSet.insert(edge).second) {
      oppositeSideTreeEdges.push_back(edge);
    }
  };

  using WavefrontNode = pair<int, int>;
  priority_queue<WavefrontNode, vector<WavefrontNode>, greater<WavefrontNode> > wavefront;
  const int graphSize = (int)graphGCellIdxs.size();
  const int INF = numeric_limits<int>::max();
  vector<int> dist(graphSize, INF);
  vector<int> prev(graphSize, -1);
  vector<bool> inTree(graphSize, false);

  auto addTreeSource = [&](int graphIdx) {
    inTree[graphIdx] = true;
    dist[graphIdx] = 0;
    prev[graphIdx] = -1;
    wavefront.emplace(0, graphIdx);
  };

  auto isAxisSource = [&](const frPoint &gcellIdx) {
    if (axisCtx.axisCoord(gcellIdx) != axisGCellIdx) {
      return false;
    }
    if (explicitAxisSources.find(gcellIdx) != explicitAxisSources.end()) {
      return true;
    }
    for (auto rootSideEdge: rootSideTreeEdges) {
      if (axisCtx.axisCoord(rootSideEdge.first) != axisGCellIdx ||
          axisCtx.axisCoord(rootSideEdge.second) != axisGCellIdx) {
        continue;
      }
      if (selfSymmetryPointOnSegment(rootSideEdge.first, rootSideEdge.second,
                                     gcellIdx)) {
        return true;
      }
    }
    return false;
  };

  bool hasAxisSource = false;
  for (int graphIdx = 0; graphIdx < graphSize; graphIdx++) {
    if (!isAxisSource(graphGCellIdxs[graphIdx])) {
      continue;
    }
    appendVertex(graphGCellIdxs[graphIdx]);
    addTreeSource(graphIdx);
    hasAxisSource = true;
  }

  if (!hasAxisSource) {
    failSelfSymmetryTopology(
        "Error: failed to find opposite-side self-symmetry axis source");
  }

  auto searchNextTarget = [&](const function<bool(int)> &isTarget, vector<int> &path) {
    path.clear();

    while (!wavefront.empty()) {
      auto wavefrontNode = wavefront.top();
      wavefront.pop();

      int currCost = wavefrontNode.first;
      int currIdx = wavefrontNode.second;
      if (currCost != dist[currIdx]) {
        continue;
      }

      if (!inTree[currIdx] && isTarget(currIdx)) {
        int pathIdx = currIdx;
        path.push_back(pathIdx);
        while (!inTree[pathIdx]) {
          pathIdx = prev[pathIdx];
          if (pathIdx < 0) {
            return false;
          }
          path.push_back(pathIdx);
        }
        reverse(path.begin(), path.end());
        return true;
      }

      for (int dir = 0; dir < 4; dir++) {
        int nextIdx = graphNeighbors[currIdx][dir];
        if (nextIdx < 0) {
          continue;
        }
        int nextCost = currCost + graphNeighborCosts[currIdx][dir];
        if (nextCost < dist[nextIdx]) {
          dist[nextIdx] = nextCost;
          prev[nextIdx] = currIdx;
          wavefront.emplace(nextCost, nextIdx);
        }
      }
    }

    return false;
  };

  auto appendPath = [&](const vector<int> &path) {
    for (auto graphIdx: path) {
      appendVertex(graphGCellIdxs[graphIdx]);
    }
    for (int i = 1; i < (int)path.size(); i++) {
      appendEdge(graphGCellIdxs[path[i - 1]], graphGCellIdxs[path[i]]);
    }
    for (auto graphIdx: path) {
      addTreeSource(graphIdx);
    }
  };

  set<int> unconnectedTerminals;
  for (auto terminal: terminals) {
    auto terminalItr = gcellIdx2GraphIdx.find(terminal);
    if (terminalItr == gcellIdx2GraphIdx.end()) {
      failSelfSymmetryTopology(
          "Error: failed to place opposite-side self-symmetry terminal on Hanan graph");
    }
    if (!inTree[terminalItr->second]) {
      unconnectedTerminals.insert(terminalItr->second);
    }
  }

  while (!unconnectedTerminals.empty()) {
    vector<int> path;
    if (!searchNextTarget([&](int graphIdx) { return unconnectedTerminals.find(graphIdx) != unconnectedTerminals.end(); },
                          path)) {
      failSelfSymmetryTopology(
          "Error: failed to find opposite-side self-symmetry Hanan path");
    }
    appendPath(path);
    for (auto graphIdx: path) {
      unconnectedTerminals.erase(graphIdx);
    }
  }
}

void FlexGR::initGR_genTopology_selfsymmetry_net(frNet* net) {
    if (net->getNodes().size() == 0)
        return ;
    if (net->getNodes().size() == 1) {
        net->setRoot(net->getNodes().front().get());
        return ;
    }
    std::map<frBlockObject*, frRPin*> pin2RPin;
    for (auto& rpin : net->getRPins()) {
        if (rpin->getFrTerm() == nullptr)
            continue ;
        pin2RPin[rpin->getFrTerm()] = rpin.get();
    }
    auto selfSymmetryConstraint = net->getSelfSymmetryConstraint();
    for (auto& node : net->getNodes()) {
        if (node->getPin()) {
            if (pin2RPin.find(node->getPin()) == pin2RPin.end()) {
                failSelfSymmetryTopology(
                    std::string("[ERROR] ") + __FILE__ + ":" +
                    std::to_string(__LINE__),
                    0);
            }
            auto rpin = pin2RPin[node->getPin()];
            frPoint pt;
            if (rpin->getFrTerm()->typeId() == frcInstTerm) {
              auto inst =
                  static_cast<frInstTerm *>(rpin->getFrTerm())->getInst();
              frTransform shiftXform;
              inst->getTransform(shiftXform);
              shiftXform.set(frOrient(frcR0));
              rpin->getAccessPoint()->getPoint(pt);
              pt.transform(shiftXform);
            }
            else {
                rpin->getAccessPoint()->getPoint(pt);
            }
            node->setLoc(pt);
            node->setLayerNum(rpin->getAccessPoint()->getLayerNum());
        }
    }
    frNode* rootNode = nullptr;
    std::vector<frNode*> nodes;
    for (auto& node : net->getNodes()) {
        if (node->getPin()) {
            if (node->getPin()->typeId() == frcInstTerm) {
                auto term = static_cast<frInstTerm*>(node->getPin())->getTerm();
                if (term->getDirection() == frTermDirectionEnum::OUTPUT)
                    rootNode = node.get();
                nodes.push_back(node.get());
            }
            else if(node->getPin()->typeId() == frcTerm) {
                auto term = static_cast<frTerm*>(node->getPin());
                if (term->getDirection() == frTermDirectionEnum::INPUT)
                    rootNode = node.get();
                nodes.push_back(node.get());
            }
        }
    }
    if (rootNode == nullptr && !nodes.empty()) {
        rootNode = nodes.back();
    }
    net->setRoot(rootNode);
    for (auto node : nodes) {
        node->setParent(nullptr);
        node->clearChildren();
        node->setConnFig(nullptr);
    }

    auto pointKey = [](const frPoint &point) {
        return getSelfSymmetryGCellKey(point);
    };

    auto getGCellIdxFromLoc = [&](const frPoint &loc) {
        frPoint gcellIdx;
        design->getTopBlock()->getGCellIdx(loc, gcellIdx);
        return gcellIdx;
    };

    auto getGCellCenter = [&](const frPoint &gcellIdx) {
        frBox gcellBox;
        design->getTopBlock()->getGCellBox(gcellIdx, gcellBox);
        return frPoint((gcellBox.left() + gcellBox.right()) / 2,
                       (gcellBox.bottom() + gcellBox.top()) / 2);
    };

    frPoint rootLoc;
    rootNode->getLoc(rootLoc);
    frPoint rootGCellIdx = getGCellIdxFromLoc(rootLoc);
    frPoint rootGCellLoc = getGCellCenter(rootGCellIdx);

    auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(
        design, selfSymmetryConstraint, rootGCellLoc);
    if (!axisCtx.valid) {
        return;
    }
    frCoord axisGCellIdx = axisCtx.axisGCellIdx;

    int rootSide = normalizeSelfSymmetryRootSide(axisCtx.sideOfPoint(rootLoc));

    std::map<std::pair<int, int>, std::vector<frNode*>> sourceGCell2PinNodes;
    std::set<frNode*> sourcePinNodes;
    std::set<frNode*> mirrorPinNodes;
    for (auto node : nodes) {
        frPoint location;
        node->getLoc(location);
        frPoint gcellLocation = getGCellIdxFromLoc(location);
        int pinSide = axisCtx.sideOfPoint(location);
        bool isSourcePin = (node == rootNode || pinSide == 0 || pinSide == rootSide);
        if (isSourcePin) {
            sourceGCell2PinNodes[pointKey(gcellLocation)].push_back(node);
            sourcePinNodes.insert(node);
        }
        else {
            mirrorPinNodes.insert(node);
        }
    }

    auto &gcellIdx2Nodes = net2GCellIdx2Nodes[net];
    auto &gcellNode2RPinNodes = net2GCellNode2RPinNodes[net];
    auto &gcellNodes = net2GCellNodes[net];
    auto &steinerNodes = net2SteinerNodes[net];
    gcellIdx2Nodes.clear();
    gcellNode2RPinNodes.clear();
    gcellNodes.clear();
    steinerNodes.clear();

    std::map<std::pair<int, int>, frNode*> sourceTreeNodeByGCellIdx;

    auto addSourceTreeNode = [&](const frPoint &gcellIdx) {
        auto uNode = std::make_unique<frNode>();
        auto node = uNode.get();
        node->setType(frNodeTypeEnum::frcSteiner);
        node->setLayerNum(2);
        node->setLoc(getGCellCenter(gcellIdx));
        net->addNode(uNode);
        sourceTreeNodeByGCellIdx[pointKey(gcellIdx)] = node;
        return node;
    };

    frNode *rootGCellNode = addSourceTreeNode(rootGCellIdx);
    gcellNodes.push_back(rootGCellNode);
    gcellNode2RPinNodes[rootGCellNode] = sourceGCell2PinNodes[pointKey(rootGCellIdx)];
    gcellIdx2Nodes[pointKey(rootGCellIdx)] = sourceGCell2PinNodes[pointKey(rootGCellIdx)];
    net->setFirstNonRPinNode(rootGCellNode);
    net->setRootGCellNode(rootGCellNode);

    std::vector<frPoint> rootSideTerminalGCellIdxs;
    rootSideTerminalGCellIdxs.reserve(sourceGCell2PinNodes.size());
    rootSideTerminalGCellIdxs.push_back(rootGCellIdx);
    for (auto &[gcellKey, pinNodes] : sourceGCell2PinNodes) {
        frPoint gcellIdx(gcellKey.first, gcellKey.second);
        if (gcellIdx == rootGCellIdx) {
            continue;
        }
        auto gcellNode = addSourceTreeNode(gcellIdx);
        gcellNodes.push_back(gcellNode);
        gcellNode2RPinNodes[gcellNode] = pinNodes;
        gcellIdx2Nodes[gcellKey] = pinNodes;
        rootSideTerminalGCellIdxs.push_back(gcellIdx);
    }

    std::vector<frPoint> rootSideTreeVertices;
    std::vector<std::pair<frPoint, frPoint>> rootSideTreeEdges;
    genSelfSymmetryRootSideTopology(rootSideTerminalGCellIdxs,
                                    rootGCellIdx,
                                    selfSymmetryConstraint.isAxisHorizontal,
                                    axisGCellIdx,
                                    rootSideTreeVertices,
                                    rootSideTreeEdges);

    auto getOrCreateSourceTreeNode = [&](const frPoint &gcellIdx) {
        auto key = pointKey(gcellIdx);
        auto nodeIt = sourceTreeNodeByGCellIdx.find(key);
        if (nodeIt != sourceTreeNodeByGCellIdx.end()) {
            return nodeIt->second;
        }
        auto node = addSourceTreeNode(gcellIdx);
        steinerNodes.push_back(node);
        return node;
    };

    for (auto vertex : rootSideTreeVertices) {
        getOrCreateSourceTreeNode(vertex);
    }
    for (auto edge : rootSideTreeEdges) {
        getOrCreateSourceTreeNode(edge.first);
        getOrCreateSourceTreeNode(edge.second);
    }

    std::map<std::pair<int, int>, std::vector<frPoint>> rootSideAdj;
    for (auto edge : rootSideTreeEdges) {
        rootSideAdj[pointKey(edge.first)].push_back(edge.second);
        rootSideAdj[pointKey(edge.second)].push_back(edge.first);
    }

    std::set<std::pair<int, int>> visitedGCellIdxs;
    std::deque<frPoint> nodeQ;
    visitedGCellIdxs.insert(pointKey(rootGCellIdx));
    nodeQ.push_back(rootGCellIdx);
    while (!nodeQ.empty()) {
        auto currGCellIdx = nodeQ.front();
        nodeQ.pop_front();
        auto currNode = getOrCreateSourceTreeNode(currGCellIdx);
        for (auto childGCellIdx : rootSideAdj[pointKey(currGCellIdx)]) {
            auto childKey = pointKey(childGCellIdx);
            if (!visitedGCellIdxs.insert(childKey).second) {
                continue;
            }
            auto childNode = getOrCreateSourceTreeNode(childGCellIdx);
            currNode->addChild(childNode);
            childNode->setParent(currNode);
            nodeQ.push_back(childGCellIdx);
        }
    }

    for (auto &[gcellNode, localPinNodes] : gcellNode2RPinNodes) {
        for (auto localPinNode : localPinNodes) {
            if (localPinNode == rootNode) {
                gcellNode->setParent(localPinNode);
                localPinNode->addChild(gcellNode);
            }
            else {
                gcellNode->addChild(localPinNode);
                localPinNode->setParent(gcellNode);
            }
        }
    }

    auto rootSideReachesAxis = [&]() {
        for (auto vertex : rootSideTreeVertices) {
            if (axisCtx.axisCoord(vertex) == axisGCellIdx) {
                return true;
            }
        }
        for (auto edge : rootSideTreeEdges) {
            if (std::min(axisCtx.axisCoord(edge.first),
                         axisCtx.axisCoord(edge.second)) <= axisGCellIdx &&
                std::max(axisCtx.axisCoord(edge.first),
                         axisCtx.axisCoord(edge.second)) >= axisGCellIdx) {
                return true;
            }
        }
        return false;
    };

    bool reachesAxis = rootSideReachesAxis();
    if (!reachesAxis) {
        cout << "Error: self-symmetry root-side source tree does not reach axis for net "
             << net->getName() << "\n";
    }

    for (auto &[gcellKey, treeNode] : sourceTreeNodeByGCellIdx) {
        frPoint gcellIdx(gcellKey.first, gcellKey.second);
        int side = axisCtx.sideOfGCell(gcellIdx);
        if (side != 0 && side != rootSide) {
            cout << "Error: self-symmetry source tree node on mirror side for net "
                 << net->getName() << "\n";
        }
        if (visitedGCellIdxs.find(gcellKey) == visitedGCellIdxs.end()) {
            cout << "Error: self-symmetry source tree node is disconnected for net "
                 << net->getName() << "\n";
        }
    }

    for (auto pinNode : sourcePinNodes) {
        if (pinNode != rootNode && pinNode->getParent() == nullptr) {
            cout << "Error: self-symmetry source-side non-root pin does not have parent for net "
                 << net->getName() << "\n";
        }
    }
    for (auto pinNode : mirrorPinNodes) {
        if (pinNode->getParent() != nullptr || !pinNode->getChildren().empty()) {
            cout << "Error: self-symmetry mirror-side pin is connected for net "
                 << net->getName() << "\n";
        }
    }

    auto shouldDumpSelfSymmetryTopology = [&]() {
        const bool enableDump = true;
        return enableDump && SelfSymmetryDebug::isDebugNet(net);
    };

    if (shouldDumpSelfSymmetryTopology()) {
        auto printPoint = [](const frPoint &point) {
            cout << "(" << point.x() << ", " << point.y() << ")";
        };
        auto printPinName = [](frBlockObject *pin) {
            if (pin == nullptr) {
                cout << "<null>";
            }
            else if (pin->typeId() == frcInstTerm) {
                auto instTerm = static_cast<frInstTerm*>(pin);
                cout << instTerm->getInst()->getName() << "/"
                     << instTerm->getTerm()->getName();
            }
            else if (pin->typeId() == frcTerm) {
                auto term = static_cast<frTerm*>(pin);
                cout << "PIN/" << term->getName();
            }
            else {
                cout << "<unknown>";
            }
        };
        auto edgeCost = [&](const frPoint &begin, const frPoint &end) {
            auto length = getSelfSymmetryEdgeLen(begin, end);
            return length * (axisCtx.isAxisEdge(begin, end) ? 1 : 4);
        };

        cout << "@@@ self-symmetry topology @@@\n";
        cout << "net: " << net->getName() << "\n";
        cout << "axis: "
             << (selfSymmetryConstraint.isAxisHorizontal ? "horizontal y=" : "vertical x=")
             << selfSymmetryConstraint.axis << ", "
             << (selfSymmetryConstraint.isAxisHorizontal ? "gcell_y=" : "gcell_x=")
             << axisGCellIdx << "\n";
        cout << "root side: " << rootSide << "\n";

        cout << "pins:\n";
        for (int i = 0; i < (int)nodes.size(); i++) {
            auto pinNode = nodes[i];
            frPoint pinLoc;
            pinNode->getLoc(pinLoc);
            frPoint pinGCellIdx = getGCellIdxFromLoc(pinLoc);
            int pinSide = axisCtx.sideOfPoint(pinLoc);
            bool inSourceTree = (pinNode == rootNode || pinNode->getParent() != nullptr);
            cout << "  p" << i << ": ";
            printPinName(pinNode->getPin());
            cout << " loc=";
            printPoint(pinLoc);
            cout << ", gcell=";
            printPoint(pinGCellIdx);
            cout << ", layer=" << pinNode->getLayerNum()
                 << ", side=" << pinSide
                 << ", root=" << (pinNode == rootNode ? 1 : 0)
                 << ", in_source_tree=" << (inSourceTree ? 1 : 0)
                 << ", parent=";
            if (pinNode->getParent()) {
                cout << pinNode->getParent()->getId();
            }
            else {
                cout << "null";
            }
            cout << "\n";
        }

        cout << "root-side terminals:\n";
        for (int i = 0; i < (int)rootSideTerminalGCellIdxs.size(); i++) {
            cout << "  t" << i << ": ";
            printPoint(rootSideTerminalGCellIdxs[i]);
            cout << "\n";
        }

        cout << "\nroot-side vertices:\n";
        for (int i = 0; i < (int)rootSideTreeVertices.size(); i++) {
            cout << "  v" << i << ": ";
            printPoint(rootSideTreeVertices[i]);
            cout << "\n";
        }

        cout << "\nroot-side edges:\n";
        for (int i = 0; i < (int)rootSideTreeEdges.size(); i++) {
            const auto &edge = rootSideTreeEdges[i];
            cout << "  e" << i << ": ";
            printPoint(edge.first);
            cout << " -> ";
            printPoint(edge.second);
            cout << ", length=" << getSelfSymmetryEdgeLen(edge.first, edge.second)
                 << ", cost=" << edgeCost(edge.first, edge.second)
                 << ", axis=" << (axisCtx.isAxisEdge(edge.first, edge.second) ? 1 : 0)
                 << "\n";
        }

        cout << "\nsource tree parent-child:\n";
        for (auto &[gcellKey, treeNode] : sourceTreeNodeByGCellIdx) {
            frPoint treeGCellIdx(gcellKey.first, gcellKey.second);
            cout << "  node " << treeNode->getId() << " gcell=";
            printPoint(treeGCellIdx);
            cout << " parent=";
            if (treeNode->getParent() && treeNode->getParent()->getType() == frNodeTypeEnum::frcSteiner) {
                cout << treeNode->getParent()->getId();
            }
            else {
                cout << "null";
            }
            cout << " children=[";
            bool isFirstChild = true;
            for (auto child : treeNode->getChildren()) {
                if (child->getType() != frNodeTypeEnum::frcSteiner) {
                    continue;
                }
                if (!isFirstChild) {
                    cout << ",";
                }
                cout << child->getId();
                isFirstChild = false;
            }
            cout << "]\n";
        }
        cout << "root-side reaches axis: " << (reachesAxis ? 1 : 0) << "\n";
        cout << "@@@ end self-symmetry topology @@@\n";
    }
}
