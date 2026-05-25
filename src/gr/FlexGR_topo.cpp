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
#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>

using namespace std;
using namespace fr;

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

  auto normalizeEdge = [](frPoint begin, frPoint end) {
    if (end < begin) {
      swap(begin, end);
    }
    return make_pair(begin, end);
  };

  auto appendEdge = [&](const frPoint &begin, const frPoint &end) {
    if (begin == end) {
      return;
    }
    auto edge = normalizeEdge(begin, end);
    if (treeEdgeSet.insert(edge).second) {
      rootSideTreeEdges.push_back(edge);
    }
  };

  auto getAxisCoord = [&](const frPoint &gcellIdx) {
    return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
  };

  auto isAxisEdge = [&](const frPoint &begin, const frPoint &end) {
    if (isAxisHorizontal) {
      return begin.y() == axisGCellIdx && end.y() == axisGCellIdx;
    }
    return begin.x() == axisGCellIdx && end.x() == axisGCellIdx;
  };

  auto edgeCost = [&](const frPoint &begin, const frPoint &end) {
    auto length = abs(end.x() - begin.x()) + abs(end.y() - begin.y());
    return length * (isAxisEdge(begin, end) ? 1 : 4);
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
      cout << "Error: failed to find root-side self-symmetry Hanan path\n";
      exit(1);
    }
    appendPath(path);
    for (auto graphIdx: path) {
      unconnectedTerminals.erase(graphIdx);
    }
  }

  bool intersectsAxis = false;
  for (int graphIdx = 0; graphIdx < graphSize; graphIdx++) {
    if (inTree[graphIdx] && getAxisCoord(graphGCellIdxs[graphIdx]) == axisGCellIdx) {
      intersectsAxis = true;
      break;
    }
  }

  if (!intersectsAxis) {
    vector<bool> isAxisTarget(graphSize, false);
    for (int graphIdx = 0; graphIdx < graphSize; graphIdx++) {
      isAxisTarget[graphIdx] = getAxisCoord(graphGCellIdxs[graphIdx]) == axisGCellIdx;
    }

    vector<int> path;
    if (!searchNextTarget([&](int graphIdx) { return isAxisTarget[graphIdx]; }, path)) {
      cout << "Error: failed to connect root-side self-symmetry tree to axis gcell\n";
      exit(1);
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

  auto getAxisCoord = [&](const frPoint &gcellIdx) {
    return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
  };

  auto getSide = [&](frCoord coord) {
    if (coord < axisGCellIdx) {
      return -1;
    }
    if (coord > axisGCellIdx) {
      return 1;
    }
    return 0;
  };

  auto isAllowedPoint = [&](const frPoint &gcellIdx) {
    int side = getSide(getAxisCoord(gcellIdx));
    return side == 0 || side != rootSide;
  };

  auto mirrorPoint = [&](const frPoint &gcellIdx) {
    frPoint mirroredGCellIdx = gcellIdx;
    if (isAxisHorizontal) {
      mirroredGCellIdx.set(gcellIdx.x(), axisGCellIdx + (axisGCellIdx - gcellIdx.y()));
    }
    else {
      mirroredGCellIdx.set(axisGCellIdx + (axisGCellIdx - gcellIdx.x()), gcellIdx.y());
    }
    return mirroredGCellIdx;
  };

  auto normalizeEdge = [](frPoint begin, frPoint end) {
    if (end < begin) {
      swap(begin, end);
    }
    return make_pair(begin, end);
  };

  auto pointOnSegment = [](const frPoint &segmentBegin,
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
  };

  auto coversSegment = [](const frPoint &segmentBegin,
                          const frPoint &segmentEnd,
                          const frPoint &candidateBegin,
                          const frPoint &candidateEnd) {
    if (candidateBegin == candidateEnd) {
      return false;
    }
    if (candidateBegin.x() == candidateEnd.x()) {
      if (segmentBegin.x() != segmentEnd.x() ||
          segmentBegin.x() != candidateBegin.x()) {
        return false;
      }
      return min(candidateBegin.y(), candidateEnd.y()) >= min(segmentBegin.y(), segmentEnd.y()) &&
             max(candidateBegin.y(), candidateEnd.y()) <= max(segmentBegin.y(), segmentEnd.y());
    }
    if (candidateBegin.y() == candidateEnd.y()) {
      if (segmentBegin.y() != segmentEnd.y() ||
          segmentBegin.y() != candidateBegin.y()) {
        return false;
      }
      return min(candidateBegin.x(), candidateEnd.x()) >= min(segmentBegin.x(), segmentEnd.x()) &&
             max(candidateBegin.x(), candidateEnd.x()) <= max(segmentBegin.x(), segmentEnd.x());
    }
    return false;
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
      cout << "Error: opposite-side self-symmetry terminal crosses into root side\n";
      exit(1);
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
    if (getAxisCoord(rootSideVertex) == axisGCellIdx) {
      explicitAxisSources.insert(rootSideVertex);
      addGraphCoord(rootSideVertex);
    }

    auto mirroredVertex = mirrorPoint(rootSideVertex);
    addGraphCoord(mirroredVertex);
  }

  for (auto rootSideEdge: rootSideTreeEdges) {
    if (getAxisCoord(rootSideEdge.first) == axisGCellIdx &&
        getAxisCoord(rootSideEdge.second) == axisGCellIdx) {
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

    mirroredRootSideEdges.push_back(normalizeEdge(mirroredBegin, mirroredEnd));
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
      if (coversSegment(mirroredRootSideEdge.first,
                        mirroredRootSideEdge.second,
                        begin,
                        end)) {
        return true;
      }
    }
    return false;
  };

  auto edgeCost = [&](const frPoint &begin, const frPoint &end) {
    auto length = abs(end.x() - begin.x()) + abs(end.y() - begin.y());
    return length * (isMirrorRewardEdge(begin, end) ? 1 : 4);
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
    auto edge = normalizeEdge(begin, end);
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
    if (getAxisCoord(gcellIdx) != axisGCellIdx) {
      return false;
    }
    if (explicitAxisSources.find(gcellIdx) != explicitAxisSources.end()) {
      return true;
    }
    for (auto rootSideEdge: rootSideTreeEdges) {
      if (getAxisCoord(rootSideEdge.first) != axisGCellIdx ||
          getAxisCoord(rootSideEdge.second) != axisGCellIdx) {
        continue;
      }
      if (pointOnSegment(rootSideEdge.first, rootSideEdge.second, gcellIdx)) {
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
    cout << "Error: failed to find opposite-side self-symmetry axis source\n";
    exit(1);
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
      cout << "Error: failed to place opposite-side self-symmetry terminal on Hanan graph\n";
      exit(1);
    }
    if (!inTree[terminalItr->second]) {
      unconnectedTerminals.insert(terminalItr->second);
    }
  }

  while (!unconnectedTerminals.empty()) {
    vector<int> path;
    if (!searchNextTarget([&](int graphIdx) { return unconnectedTerminals.find(graphIdx) != unconnectedTerminals.end(); },
                          path)) {
      cout << "Error: failed to find opposite-side self-symmetry Hanan path\n";
      exit(1);
    }
    appendPath(path);
    for (auto graphIdx: path) {
      unconnectedTerminals.erase(graphIdx);
    }
  }
}

// pinGCellNodes size always >= 2
void FlexGR::genSTTopology_FLUTE(vector<frNode*> &pinGCellNodes, vector<frNode*> &steinerNodes) {
  auto root = pinGCellNodes[0];
  auto net = root->getNet();
  // prep for flute
  int degree = pinGCellNodes.size();
  int xs[degree];
  int ys[degree];
  frPoint loc;
  for (int i = 0; i < (int)pinGCellNodes.size(); i++) {
    auto gcellNode = pinGCellNodes[i];
    gcellNode->getLoc(loc);
    xs[i] = loc.x();
    ys[i] = loc.y();
  }
  int accuracy = 3;
  auto fluteTree = flute::flute(degree, xs, ys, accuracy);

  // if (net->getName() == "net3138") {
  //   cout << "@@@ debug\n";
  //   cout << "degree = " << degree << endl;
  //   for (int i = 0; i < (int)pinGCellNodes.size(); i++) {
  //     auto gcellNode = pinGCellNodes[i];
  //     gcellNode->getLoc(loc);
  //     cout << "  loc: (" << loc.x() << ", " << loc.y() << ")\n";
  //   }
  //   cout << "@@@ fluteTree @@@\n";
  //   flute::printtree(fluteTree);
  // }

  map<frPoint, frNode*> pinGCell2Nodes, steinerGCell2Nodes;
  map<frNode*, set<frNode*, frBlockObjectComp>, frBlockObjectComp> adjacencyList;

  frPoint pinGCellLoc;
  for (auto pinNode: pinGCellNodes) {
    pinNode->getLoc(pinGCellLoc);
    pinGCell2Nodes[pinGCellLoc] = pinNode;
  }

  // iterate over branches, create new nodes and build connectivity
  for (int i = 0; i < degree * 2 - 2; i++) {
    auto &branch1 = fluteTree.branch[i];
    auto &branch2 = fluteTree.branch[branch1.n];

    frPoint bp(branch1.x, branch1.y);
    frPoint ep(branch2.x, branch2.y);
    
    if (bp == ep) {
      continue;
    }
    
    // colinear
    // if (bp.x() == ep.x() || bp.y() == ep.y()) {
      frNode *bpNode = nullptr;
      frNode *epNode = nullptr;

      if (pinGCell2Nodes.find(bp) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(bp) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          bpNode = steinerNode.get();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(bp);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[bp] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        } else {
          bpNode = steinerGCell2Nodes[bp];
        }
      } else {
        bpNode = pinGCell2Nodes[bp];
      }
      if (pinGCell2Nodes.find(ep) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(ep) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          epNode = steinerNode.get();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(ep);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[ep] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        } else {
          epNode = steinerGCell2Nodes[ep];
        }
      } else {
        epNode = pinGCell2Nodes[ep];
      }

      if (bpNode == nullptr || epNode == nullptr) {
        cout << "Error: bpNode or epNode is void\n";
      }

      adjacencyList[bpNode].insert(epNode);
      adjacencyList[epNode].insert(bpNode);
    // } else {
    //   frPoint mp(bp.x(), ep.y());
    //   frNode *bpNode = nullptr;
    //   frNode *mpNode = nullptr;
    //   frNode *epNode = nullptr;

    //   if (pinGCell2Nodes.find(bp) == pinGCell2Nodes.end()) {
    //     if (steinerGCell2Nodes.find(bp) == steinerGCell2Nodes.end()) {
    //       // add steiner
    //       auto steinerNode = make_unique<frNode>();
    //       bpNode = steinerNode.get();
    //       steinerNode->setType(frNodeTypeEnum::frcSteiner);
    //       steinerNode->setLoc(bp);
    //       steinerGCell2Nodes[bp] = steinerNode.get();
    //       steinerNodes.push_back(steinerNode.get());
    //       net->addNode(steinerNode);
    //     } else {
    //       bpNode = steinerGCell2Nodes[bp];
    //     }
    //   } else {
    //     bpNode = pinGCell2Nodes[bp];
    //   }

    //   if (pinGCell2Nodes.find(mp) == pinGCell2Nodes.end()) {
    //     if (steinerGCell2Nodes.find(mp) == steinerGCell2Nodes.end()) {
    //       // add steiner
    //       auto steinerNode = make_unique<frNode>();
    //       mpNode = steinerNode.get();
    //       steinerNode->setType(frNodeTypeEnum::frcSteiner);
    //       steinerNode->setLoc(mp);
    //       steinerGCell2Nodes[mp] = steinerNode.get();
    //       steinerNodes.push_back(steinerNode.get());
    //       net->addNode(steinerNode);
    //     } else {
    //       mpNode = steinerGCell2Nodes[mp];
    //     }
    //   } else {
    //     mpNode = pinGCell2Nodes[mp];
    //   }

    //   if (pinGCell2Nodes.find(ep) == pinGCell2Nodes.end()) {
    //     if (steinerGCell2Nodes.find(ep) == steinerGCell2Nodes.end()) {
    //       // add steiner
    //       auto steinerNode = make_unique<frNode>();
    //       epNode = steinerNode.get();
    //       steinerNode->setType(frNodeTypeEnum::frcSteiner);
    //       steinerNode->setLoc(ep);
    //       steinerGCell2Nodes[ep] = steinerNode.get();
    //       steinerNodes.push_back(steinerNode.get());
    //       net->addNode(steinerNode);
    //     } else {
    //       epNode = steinerGCell2Nodes[ep];
    //     }
    //   } else {
    //     epNode = pinGCell2Nodes[ep];
    //   }

    //   if (bpNode == nullptr || epNode == nullptr || mpNode == nullptr) {
    //     cout << "Error: bpNode or mpNode or epNode is void\n";
    //   }

    //   adjacencyList[bpNode].insert(mpNode);
    //   adjacencyList[mpNode].insert(bpNode);
    //   adjacencyList[mpNode].insert(epNode);
    //   adjacencyList[epNode].insert(mpNode);
    // }
  }

  flute::freetree(fluteTree);

  // reset nodes
  for (auto &[loc, node]: pinGCell2Nodes) {
    node->reset();
  }

  // build tree
  set<frNode*> visitedNodes;
  deque<frNode*> nodeQueue;

  nodeQueue.push_front(root);
  visitedNodes.insert(root);

  while (!nodeQueue.empty()) {
    auto currNode = nodeQueue.back();
    nodeQueue.pop_back();

    for (auto adjNode: adjacencyList[currNode]) {
      if (visitedNodes.find(adjNode) == visitedNodes.end()) {
        currNode->addChild(adjNode);
        adjNode->setParent(currNode);
        nodeQueue.push_front(adjNode);
        visitedNodes.insert(adjNode);
      }
    }
  }
}

void FlexGR::genMSTTopology(vector<frNode*> &nodes) {
  genMSTTopology_PD(nodes);
}

void FlexGR::genMSTTopology_PD(vector<frNode*> &nodes, double alpha) {
  // manhattan distance array
  vector<vector<int> > dists(nodes.size(), vector<int>(nodes.size(), INT_MAX));
  vector<bool> isVisited(nodes.size(), false);
  vector<int> pathLens(nodes.size(), INT_MAX);
  vector<int> parentIdx(nodes.size(), -1);
  vector<int> keys(nodes.size(), INT_MAX);

  // init dist array
  for (int i = 0; i < (int)nodes.size(); i++) {
    for (int j = i; j < (int)nodes.size(); j++) {
      auto node1 = nodes[i];
      auto node2 = nodes[j];
      frPoint loc1, loc2;
      node1->getLoc(loc1);
      node2->getLoc(loc2);
      int dist = abs(loc2.x() - loc1.x()) + abs(loc2.y() - loc1.y());
      dists[i][j] = dist;
      dists[j][i] = dist;
    }
  }

  pathLens[0] = 0;
  keys[0] = 0;
  parentIdx[0] = 0;

  // start pd
  for (int i = 0; i < (int)isVisited.size(); i++) {
    int minIdx = genMSTTopology_PD_minIdx(keys, isVisited);
    isVisited[minIdx] = true;
    pathLens[minIdx] = pathLens[parentIdx[minIdx]] + dists[parentIdx[minIdx]][minIdx];
    for (int idx = 0; idx < (int)nodes.size(); idx++) {
      if (isVisited[idx] == false && (pathLens[minIdx] * alpha + dists[minIdx][idx] < keys[idx])) {
        parentIdx[idx] = minIdx;
        keys[idx] = pathLens[minIdx] * alpha + dists[minIdx][idx];
      }
    }
  }

  // write back to nodes
  for (int i = 0; i < (int)nodes.size(); i++) {
    // only do for non-root (i.e., leaf) nodes
    if (parentIdx[i] != i) {
      auto parent = nodes[parentIdx[i]];
      auto child = nodes[i];
      child->setParent(parent);
      parent->addChild(child);
    }
  }

}

int FlexGR::genMSTTopology_PD_minIdx(const vector<int> &keys, const vector<bool> &isVisited) {
  int min = INT_MAX;
  int minIdx = -1;

  for (int i = 0; i < (int)isVisited.size(); i++) {
    if (isVisited[i] == false && keys[i] < min) {
      min = keys[i];
      minIdx = i;
    }
  }
  return minIdx;
}

// root is 0
void FlexGR::genSTTopology_HVW(vector<frNode*> &nodes, vector<frNode*> &steinerNodes) {
  vector<unsigned> overlapL(nodes.size(), 0);
  vector<unsigned> overlapU(nodes.size(), 0);
  vector<unsigned> bestCombL(nodes.size(), 0);
  vector<unsigned> bestCombU(nodes.size(), 0);
  vector<bool> isU(nodes.size(), false);

  // recursively compute best overlap
  genSTTopology_HVW_compute(nodes[0], nodes, overlapL, overlapU, bestCombL, bestCombU);

  // recursively get L/U
  if (overlapL[0] >= overlapU[0]) {
    genSTTopology_HVW_commit(nodes[0], false, nodes, /*overlapL, overlapU,*/ bestCombL, bestCombU, isU);
    // cout << "  overlap = " << overlapL[0] << "\n";
  } else {
    genSTTopology_HVW_commit(nodes[0], true,  nodes, /*overlapL, overlapU,*/ bestCombL, bestCombU, isU);
    // cout << "  overlap = " << overlapU[0] << "\n";
  }
  // cout << "  isU = ";
  // for (int i = 0; i < (int)isU.size(); i++) {
  //   cout << isU[i];
  // }
  // cout << "\n";

  // additional steiner nodes needed for tree construction 
  genSTTopology_build_tree(nodes, isU, steinerNodes);
}

void FlexGR::genSTTopology_HVW_compute(frNode *currNode,
                                       vector<frNode*> &nodes,
                                       vector<unsigned> &overlapL,
                                       vector<unsigned> &overlapU,
                                       vector<unsigned> &bestCombL,
                                       vector<unsigned> &bestCombU) {
  if (currNode == nullptr) {
    return;
  }

  if (currNode->getChildren().empty()) {
    genSTTopology_HVW_compute(nullptr, nodes, overlapL, overlapU, bestCombL, bestCombU);
  } else {
    for (auto child: currNode->getChildren()) {
      genSTTopology_HVW_compute(child, nodes, overlapL, overlapU, bestCombL, bestCombU);
    }
  }

  unsigned numChild = currNode->getChildren().size();
  unsigned numComb = 1 << numChild;
  unsigned currOverlap = 0;
  unsigned currMaxOverlap = 0;
  unsigned bestComb = 0;
  int currNodeIdx = distance(nodes[0]->getIter(), currNode->getIter());
  // cout << "1. currNodeIdx = " << currNodeIdx << "\n";

  // curr overlapL = min(comb(children) + L)
  // bit 1 means using U, 0 means using L
  for (unsigned comb = 0; comb < numComb; comb++) {
    currOverlap = genSTTopology_HVW_levelOvlp(currNode, false, comb);
    unsigned combIdx = 0;
    for (auto child: currNode->getChildren()) {
      int nodeIdx = distance(nodes[0]->getIter(), child->getIter());
      if ((comb >> combIdx) & 1) {
        currOverlap += overlapU[nodeIdx];
      } else {
        currOverlap += overlapL[nodeIdx];
      }
      combIdx++;
    }
    if (currOverlap > currMaxOverlap) {
      currMaxOverlap = currOverlap;
      bestComb = comb;
    }
  }

  // commit for current level
  overlapL[currNodeIdx] = currMaxOverlap;
  bestCombL[currNodeIdx] = bestComb;
  // unsigned combIdx = 0;
  // for (auto child: currNode->getChildren()) {
  //   int nodeIdx = distance(nodes[0]->getIter(), child->getIter());
  //   if ((bestComb >> combIdx) & 1) {
  //     isU[nodeIdx] = true;
  //   } else {
  //     isU[nodeIdx] = false;
  //   }
  //   combIdx++;
  // }

  currMaxOverlap = 0;
  // curr overlapU = min(comb(children) + R)
  for (unsigned comb = 0; comb < numComb; comb++) {
    currOverlap = genSTTopology_HVW_levelOvlp(currNode, true, comb);
    unsigned combIdx = 0;
    for (auto child: currNode->getChildren()) {
      int nodeIdx = distance(nodes[0]->getIter(), child->getIter());
      if ((comb >> combIdx) & 1) {
        currOverlap += overlapU[nodeIdx];
      } else {
        currOverlap += overlapL[nodeIdx];
      }
      combIdx++;
    }
    if (currOverlap > currMaxOverlap) {
      currMaxOverlap = currOverlap;
      bestComb = comb;
    }
  }
  // commit for current level
  overlapU[currNodeIdx] = currMaxOverlap;
  bestCombU[currNodeIdx] = bestComb;
  // combIdx = 0;
  // for (auto child: currNode->getChildren()) {
  //   int nodeIdx = distance(nodes[0]->getIter(), child->getIter());
  //   if ((bestComb >> combIdx) & 1) {
  //     isU[nodeIdx] = true;
  //   } else {
  //     isU[nodeIdx] = false;
  //   }
  //   combIdx++;
  // }
}

unsigned FlexGR::genSTTopology_HVW_levelOvlp(frNode *currNode,
                                             bool isCurrU,
                                             unsigned comb) {
  unsigned overlap = 0;
  map<frCoord, boost::icl::interval_map<frCoord, set<frNode*> > > horzIntvMaps; // denoted by children node index
  map<frCoord, boost::icl::interval_map<frCoord, set<frNode*> > > vertIntvMaps;

  pair<frCoord, frCoord> horzIntv, vertIntv;
  frPoint turnLoc;

  // connection to parent if exists
  auto parent = currNode->getParent();
  if (parent) {
    genSTTopology_HVW_levelOvlp_helper(parent, currNode, isCurrU, horzIntv, vertIntv, turnLoc);
    set<frNode*> currNodeSet;
    currNodeSet.insert(currNode);
    if (horzIntv.first != horzIntv.second) {
      horzIntvMaps[turnLoc.y()] += make_pair(boost::icl::interval<int>::closed(horzIntv.first, horzIntv.second), currNodeSet);
    }
    if (vertIntv.first != vertIntv.second) {
      vertIntvMaps[turnLoc.x()] += make_pair(boost::icl::interval<int>::closed(vertIntv.first, vertIntv.second), currNodeSet);
    }
  }

  // connections from children to current
  unsigned combIdx = 0;
  for (auto child: currNode->getChildren()) {
    bool isCurrU = (comb >> combIdx) & 1;
    genSTTopology_HVW_levelOvlp_helper(currNode, child, isCurrU, horzIntv, vertIntv, turnLoc);
    set<frNode*> childSet;
    childSet.insert(child);
    if (horzIntv.first != horzIntv.second) {
      horzIntvMaps[turnLoc.y()] += make_pair(boost::icl::interval<int>::closed(horzIntv.first, horzIntv.second), childSet);
    }
    if (vertIntv.first != vertIntv.second) {
      vertIntvMaps[turnLoc.x()] += make_pair(boost::icl::interval<int>::closed(vertIntv.first, vertIntv.second), childSet);
    }
    combIdx++;
  }

  // iterate over intvMaps to get overlaps
  for (auto &[coord, intvMap]: horzIntvMaps) {
    for (auto it = intvMap.begin(); it != intvMap.end(); it++) {
      auto intv = it->first;
      auto idxs = it->second;
      overlap += (intv.upper() - intv.lower()) * ((int)idxs.size() - 1);
    }
  }

  return overlap;
}

void FlexGR::genSTTopology_HVW_levelOvlp_helper(frNode *parent, frNode *child, bool isCurrU,
                                                pair<frCoord, frCoord> &horzIntv,
                                                pair<frCoord, frCoord> &vertIntv,
                                                frPoint &turnLoc) {
  frPoint parentLoc, childLoc;
  parent->getLoc(parentLoc);
  child->getLoc(childLoc);

  if (isCurrU) {
    if ((childLoc.x() >= parentLoc.x() && childLoc.y() >= parentLoc.y()) ||
        (childLoc.x() <= parentLoc.x() && childLoc.y() <= parentLoc.y())) {
      turnLoc = frPoint(min(childLoc.x(), parentLoc.x()), max(childLoc.y(), parentLoc.y()));
    } else {
      turnLoc = frPoint(max(childLoc.x(), parentLoc.x()), max(childLoc.y(), parentLoc.y()));
    }
  } else {
    if ((childLoc.x() >= parentLoc.x() && childLoc.y() >= parentLoc.y()) ||
        (childLoc.x() <= parentLoc.x() && childLoc.y() <= parentLoc.y())) {
      turnLoc = frPoint(max(childLoc.x(), parentLoc.x()), min(childLoc.y(), parentLoc.y()));
    } else {
      turnLoc = frPoint(min(childLoc.x(), parentLoc.x()), min(childLoc.y(), parentLoc.y()));
    }
  }

  // set intvs
  // horz colinear
  if (childLoc.y() == parentLoc.y()) {
    horzIntv = make_pair(min(childLoc.x(), parentLoc.x()), max(childLoc.x(), parentLoc.x()));
    vertIntv = make_pair(min(childLoc.y(), parentLoc.y()), max(childLoc.y(), parentLoc.y()));
  } else if (childLoc.x() == parentLoc.x()) {
    horzIntv = make_pair(min(childLoc.x(), parentLoc.x()), max(childLoc.x(), parentLoc.x()));
    vertIntv = make_pair(min(childLoc.y(), parentLoc.y()), max(childLoc.y(), parentLoc.y()));
  } else {
    if (turnLoc.y() == childLoc.y()) {
      horzIntv = make_pair(min(turnLoc.x(), childLoc.x()), max(turnLoc.x(), childLoc.x()));
      vertIntv = make_pair(min(turnLoc.y(), parentLoc.y()), max(turnLoc.y(), parentLoc.y()));
    } else {
      horzIntv = make_pair(min(turnLoc.x(), parentLoc.x()), max(turnLoc.x(), parentLoc.x()));
      vertIntv = make_pair(min(turnLoc.y(), childLoc.y()), max(turnLoc.y(), childLoc.y()));
    }
  }
}


void FlexGR::genSTTopology_HVW_commit(frNode *currNode,
                                      bool isCurrU,
                                      vector<frNode*> &nodes,
                                      // vector<unsigned> &overlapL,
                                      // vector<unsigned> &overlapU,
                                      vector<unsigned> &bestCombL,
                                      vector<unsigned> &bestCombU,
                                      vector<bool> &isU) {
  int currNodeIdx = distance(nodes[0]->getIter(), currNode->getIter());
  // cout << "2. currNodeIdx = " << currNodeIdx << "\n";
  isU[currNodeIdx] = isCurrU;

  unsigned comb = 0;
  if (isCurrU) {
    comb = bestCombU[currNodeIdx];
  } else {
    comb = bestCombL[currNodeIdx];
  }

  unsigned combIdx = 0;
  for (auto child: currNode->getChildren()) {
    bool isChildU = (comb >> combIdx) & 1;
    genSTTopology_HVW_commit(child, isChildU, nodes, /*overlapL, overlapU,*/ bestCombL, bestCombU, isU);

    combIdx++;
  }
}

// build tree from L-shaped segment results
void FlexGR::genSTTopology_build_tree(vector<frNode*> &pinNodes,
                                      vector<bool> &isU,
                                      vector<frNode*> &steinerNodes) {
  map<frCoord, boost::icl::interval_set<frCoord> > horzIntvs, vertIntvs;
  map<frPoint, frNode*> pinGCell2Nodes, steinerGCell2Nodes;

  genSTTopology_build_tree_mergeSeg(pinNodes, isU, horzIntvs, vertIntvs);

  frPoint pinGCellLoc;
  for (auto pinNode: pinNodes) {
    pinNode->getLoc(pinGCellLoc);
    pinGCell2Nodes[pinGCellLoc] = pinNode;
  }

  // for (auto &[trackIdx, intvs]: horzIntvs) {
  //   for (auto it = intvs.begin(); it != intvs.end(); it++) {
  //     auto beginIdx = it->lower();
  //     auto endIdx = it->upper();
  //     cout << "horz intv: (" << beginIdx << ", " << trackIdx << ") - (" << endIdx << ", " << trackIdx << ")\n";
  //   }
  // }
  // for (auto &[trackIdx, intvs]: vertIntvs) {
  //   for (auto it = intvs.begin(); it != intvs.end(); it++) {
  //     auto beginIdx = it->lower();
  //     auto endIdx = it->upper();
  //     cout << "vert intv: (" << trackIdx << ", " << beginIdx << ") - (" << trackIdx << ", " << endIdx << ")\n";
  //   }
  // }

  // split seg and build tree
  genSTTopology_build_tree_splitSeg(pinNodes, pinGCell2Nodes, horzIntvs, vertIntvs, steinerGCell2Nodes, steinerNodes);

  // sanity check
  for (int i = 0; i < (int)pinNodes.size(); i++) {
    // frPoint loc;
    // pinNodes[i]->getLoc(loc);
    // cout << "pin node " << i << ": (" << loc.x() << ", " << loc.y() << ")\n";
    if (i != 0) {
      if (pinNodes[i]->getParent() == nullptr) {
        cout << "Error: non-root pin node does not have parent\n";
      }
    }
  }
  for (int i = 0; i < (int)steinerNodes.size(); i++) {
    // frPoint loc;
    // steinerNodes[i]->getLoc(loc);
    // cout << "steiner node " << i << ": (" << loc.x() << ", " << loc.y() << ")\n";
    if (steinerNodes[i]->getParent() == nullptr) {
      cout << "Error: non-root steiner node does not have parent\n";
    }
  }
}

void FlexGR::genSTTopology_build_tree_mergeSeg(vector<frNode*> &pinNodes,
                                               vector<bool> &isU,
                                               map<frCoord, boost::icl::interval_set<frCoord> > &horzIntvs,
                                               map<frCoord, boost::icl::interval_set<frCoord> > &vertIntvs) {
  frPoint turnLoc;
  pair<frCoord, frCoord> horzIntv, vertIntv;
  for (int i = 1; i < (int)pinNodes.size(); i++) {
    genSTTopology_HVW_levelOvlp_helper(pinNodes[i], pinNodes[i]->getParent(), isU[i], horzIntv, vertIntv, turnLoc);
    if (horzIntv.first != horzIntv.second) {
      horzIntvs[turnLoc.y()].insert(boost::icl::interval<int>::closed(horzIntv.first, horzIntv.second));
    }
    if (vertIntv.first != vertIntv.second) {
      vertIntvs[turnLoc.x()].insert(boost::icl::interval<int>::closed(vertIntv.first, vertIntv.second));
    }
  }
}

void FlexGR::genSTTopology_build_tree_splitSeg(vector<frNode*> &pinNodes,
                                               map<frPoint, frNode*> &pinGCell2Nodes,
                                               map<frCoord, boost::icl::interval_set<frCoord> > &horzIntvs,
                                               map<frCoord, boost::icl::interval_set<frCoord> > &vertIntvs,
                                               map<frPoint, frNode*> &steinerGCell2Nodes,
                                               vector<frNode*> &steinerNodes) {
  map<frCoord, set<pair<frCoord, frCoord> > > horzSegs, vertSegs;
  map<frNode*, set<frNode*, frBlockObjectComp>, frBlockObjectComp> adjacencyList;

  auto root = pinNodes[0];
  auto net = root->getNet();

  // horz first dimension is y coord, second dimension is x coord
  map<frCoord, set<frCoord> > horzPinHelper, vertPinHelper;
  // init
  for (auto [loc, node]: pinGCell2Nodes) {
    horzPinHelper[loc.y()].insert(loc.x());
    vertPinHelper[loc.x()].insert(loc.y());
  }

  // trackIdx == y coord
  for (auto &[trackIdx, currIntvs]: horzIntvs) {
    for (auto it = currIntvs.begin(); it != currIntvs.end(); it++) {
      set<frCoord> lineIdx;
      auto beginIdx = it->lower();
      auto endIdx = it->upper();
      // split by vertSeg
      for (auto it2 = vertIntvs.lower_bound(beginIdx); it2 != vertIntvs.end() && it2->first <= endIdx; it2++) {
        if (boost::icl::contains(it2->second, trackIdx)) {
          lineIdx.insert(it2->first); // it2->first is intersection frCoord
        }
      }
      // split by pin
      if (horzPinHelper.find(trackIdx) != horzPinHelper.end()) {
        auto &pinHelperSet = horzPinHelper[trackIdx];
        for (auto it2 = pinHelperSet.lower_bound(beginIdx); it2 != pinHelperSet.end() && (*it2) <= endIdx; it2++) {
          lineIdx.insert(*it2);
        }
      }
      // add horz seg
      if (lineIdx.empty()) {
        cout << "Error: genSTTopology_build_tree_splitSeg lineIdx is empty\n";
        exit(1);
      } else if (lineIdx.size() == 1) {
        // cout << "Error: genSTTopology_build_tree_splitSeg lineIdx size == 1\n";
        // exit(1);
      } else {
        auto prevIt = lineIdx.begin();
        for (auto currIt = (++(lineIdx.begin())); currIt != lineIdx.end(); currIt++) {
          horzSegs[trackIdx].insert(make_pair(*prevIt, *currIt));
          prevIt = currIt;
        }
      }
    }
  }

  // trackIdx == x coord
  for (auto &[trackIdx, currIntvs]: vertIntvs) {
    for (auto it = currIntvs.begin(); it != currIntvs.end(); it++) {
      set<frCoord> lineIdx;
      auto beginIdx = it->lower();
      auto endIdx = it->upper();
      // split by horzSeg
      for (auto it2 = horzIntvs.lower_bound(beginIdx); it2 != horzIntvs.end() && it2->first <= endIdx; it2++) {
        if (boost::icl::contains(it2->second, trackIdx)) {
          lineIdx.insert(it2->first);
        }
      }
      // split by pin
      if (vertPinHelper.find(trackIdx) != vertPinHelper.end()) {
        auto &pinHelperSet = vertPinHelper[trackIdx];
        for (auto it2 = pinHelperSet.lower_bound(beginIdx); it2 != pinHelperSet.end() && (*it2) <= endIdx; it2++) {
          lineIdx.insert(*it2);
        }
      }
      // add vert seg
      if (lineIdx.empty()) {
        cout << "Error: genSTTopology_build_tree_splitSeg lineIdx is empty\n";
        exit(1);
      } else if (lineIdx.size() == 1) {
        // cout << "Error: genSTTopology_build_tree_splitSeg lineIdx size == 1\n";
        // exit(1);
      } else {
        auto prevIt = lineIdx.begin();
        for (auto currIt = (++(lineIdx.begin())); currIt != lineIdx.end(); currIt++) {
          vertSegs[trackIdx].insert(make_pair(*prevIt, *currIt));
          prevIt = currIt;
        }
      }
    }
  }

  // add steiner points
  // from horz segs
  for (auto &[trackIdx, intvs]: horzSegs) {
    for (auto &intv: intvs) {
      frPoint bp(intv.first, trackIdx);
      frPoint ep(intv.second, trackIdx);
      if (pinGCell2Nodes.find(bp) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(bp) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(bp);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[bp] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        }
      }
      if (pinGCell2Nodes.find(ep) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(ep) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(ep);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[ep] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        }
      }
    }
  }
  // from vert segs
  for (auto &[trackIdx, intvs]: vertSegs) {
    for (auto &intv: intvs) {
      frPoint bp(trackIdx, intv.first);
      frPoint ep(trackIdx, intv.second);
      if (pinGCell2Nodes.find(bp) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(bp) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(bp);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[bp] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        }
      }
      if (pinGCell2Nodes.find(ep) == pinGCell2Nodes.end()) {
        if (steinerGCell2Nodes.find(ep) == steinerGCell2Nodes.end()) {
          // add steiner
          auto steinerNode = make_unique<frNode>();
          steinerNode->setType(frNodeTypeEnum::frcSteiner);
          steinerNode->setLoc(ep);
          steinerNode->setLayerNum(2);
          steinerGCell2Nodes[ep] = steinerNode.get();
          steinerNodes.push_back(steinerNode.get());
          net->addNode(steinerNode);
        }
      }
    }
  }

  // populate adjacency list
  // from horz segs
  for (auto &[trackIdx, intvs]: horzSegs) {
    for (auto &intv: intvs) {
      frPoint bp(intv.first, trackIdx);
      frPoint ep(intv.second, trackIdx);
      frNode *bpNode = nullptr;
      frNode *epNode = nullptr;
      if (pinGCell2Nodes.find(bp) != pinGCell2Nodes.end()) {
        bpNode = pinGCell2Nodes[bp];
      } else if (steinerGCell2Nodes.find(bp) != steinerGCell2Nodes.end()) {
        bpNode = steinerGCell2Nodes[bp];
      } else {
        cout << "Error: node not found\n";
      }
      if (pinGCell2Nodes.find(ep) != pinGCell2Nodes.end()) {
        epNode = pinGCell2Nodes[ep];
      } else if (steinerGCell2Nodes.find(ep) != steinerGCell2Nodes.end()) {
        epNode = steinerGCell2Nodes[ep];
      } else {
        cout << "Error: node not found\n";
      }

      if (bpNode == nullptr || epNode == nullptr) {
        cout << "Error: node is nullptr";
      }

      adjacencyList[bpNode].insert(epNode);
      adjacencyList[epNode].insert(bpNode);
    }
  }
  // from vert segs
  for (auto &[trackIdx, intvs]: vertSegs) {
    for (auto &intv: intvs) {
      frPoint bp(trackIdx, intv.first);
      frPoint ep(trackIdx, intv.second);
      frNode *bpNode = nullptr;
      frNode *epNode = nullptr;
      if (pinGCell2Nodes.find(bp) != pinGCell2Nodes.end()) {
        bpNode = pinGCell2Nodes[bp];
      } else if (steinerGCell2Nodes.find(bp) != steinerGCell2Nodes.end()) {
        bpNode = steinerGCell2Nodes[bp];
      } else {
        cout << "Error: node not found\n";
      }
      if (pinGCell2Nodes.find(ep) != pinGCell2Nodes.end()) {
        epNode = pinGCell2Nodes[ep];
      } else if (steinerGCell2Nodes.find(ep) != steinerGCell2Nodes.end()) {
        epNode = steinerGCell2Nodes[ep];
      } else {
        cout << "Error: node not found\n";
      }

      if (bpNode == nullptr || epNode == nullptr) {
        cout << "Error: node is nullptr";
      }

      adjacencyList[bpNode].insert(epNode);
      adjacencyList[epNode].insert(bpNode);
    }
  }

  // reset nodes
  for (auto &[loc, node]: pinGCell2Nodes) {
    node->reset();
  }

  // build node based tree
  /*
  set<frNode*> visitedNodes;
  deque<frNode*> nodeQueue;

  nodeQueue.push_front(root);

  while (!nodeQueue.empty()) {
    auto currNode = nodeQueue.back();
    visitedNodes.insert(currNode);
    nodeQueue.pop_back();

    for (auto adjNode: adjacencyList[currNode]) {
      if (visitedNodes.find(adjNode) == visitedNodes.end()) {
        currNode->addChild2(adjNode);
        adjNode->setParent(currNode);
        nodeQueue.push_front(adjNode);
      }
    }
  }
  */

  set<frNode*, frBlockObjectComp> visitedNodes;
  deque<frNode*> nodeQueue;

  nodeQueue.push_front(root);
  visitedNodes.insert(root);

  while (!nodeQueue.empty()) {
    auto currNode = nodeQueue.back();
    nodeQueue.pop_back();

    for (auto adjNode: adjacencyList[currNode]) {
      if (visitedNodes.find(adjNode) == visitedNodes.end()) {
        currNode->addChild(adjNode);
        adjNode->setParent(currNode);
        nodeQueue.push_front(adjNode);
        visitedNodes.insert(adjNode);
      }
    }
  }
}
