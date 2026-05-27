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
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace std;
using namespace fr;

namespace {
  pair<int, int> getSelfSymmetryGCellKey(const frPoint &point) {
    return make_pair((int)point.x(), (int)point.y());
  }

  pair<frPoint, frPoint> normalizeSelfSymmetryEdge(frPoint begin, frPoint end) {
    if (end < begin) {
      swap(begin, end);
    }
    return make_pair(begin, end);
  }

  int getSelfSymmetryAxisCoord(const frPoint &gcellIdx, bool isAxisHorizontal) {
    return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
  }

  bool selfSymmetrySegmentCovers(const frPoint &segmentBegin,
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
  }
}

const frSelfSymmetryConstraint* FlexGR::getSelfSymmetryConstraintPtr(const frNet* net) const {
  return net ? net->getSelfSymmetryConstraintPtr() : nullptr;
}

bool FlexGR::isSelfSymmetryNet(const frNet* net) const {
  return getSelfSymmetryConstraintPtr(net) != nullptr;
}

int FlexGR::getSelfSymmetryPointSide(const frPoint &point,
                                     const frSelfSymmetryConstraint &constraint) const {
  frCoord coord = constraint.isAxisHorizontal ? point.y() : point.x();
  if (coord < constraint.axis) {
    return -1;
  }
  if (coord > constraint.axis) {
    return 1;
  }
  return 0;
}

int FlexGR::getSelfSymmetryGCellSide(const frPoint &gcellIdx,
                                     bool isAxisHorizontal,
                                     frCoord axisGCellIdx) const {
  frCoord coord = isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
  if (coord < axisGCellIdx) {
    return -1;
  }
  if (coord > axisGCellIdx) {
    return 1;
  }
  return 0;
}

int FlexGR::getSelfSymmetryRootSide(frNet *net,
                                    const frSelfSymmetryConstraint &constraint) const {
  if (net == nullptr) {
    return -1;
  }

  frNode *rootNode = net->getRootGCellNode();
  if (rootNode == nullptr) {
    rootNode = net->getRoot();
  }
  if (rootNode == nullptr) {
    return -1;
  }

  frPoint rootLoc;
  rootNode->getLoc(rootLoc);
  int rootSide = getSelfSymmetryPointSide(rootLoc, constraint);
  return rootSide == 0 ? -1 : rootSide;
}

bool FlexGR::isOnSelfSymmetryAxis(const frPoint &point,
                                  const frSelfSymmetryConstraint &constraint) const {
  return getSelfSymmetryPointSide(point, constraint) == 0;
}

bool FlexGR::isOnSelfSymmetryAxisGCell(const frPoint &gcellIdx,
                                       bool isAxisHorizontal,
                                       frCoord axisGCellIdx) const {
  return getSelfSymmetryGCellSide(gcellIdx, isAxisHorizontal, axisGCellIdx) == 0;
}

frPoint FlexGR::mirrorPoint(const frPoint &point,
                            const frSelfSymmetryConstraint &constraint) const {
  frPoint mirroredPoint(point);
  if (constraint.isAxisHorizontal) {
    mirroredPoint.set(point.x(), constraint.axis + (constraint.axis - point.y()));
  } else {
    mirroredPoint.set(constraint.axis + (constraint.axis - point.x()), point.y());
  }
  return mirroredPoint;
}

frPoint FlexGR::mirrorGCellIdx(const frPoint &gcellIdx,
                               bool isAxisHorizontal,
                               frCoord axisGCellIdx) const {
  frPoint mirroredGCellIdx(gcellIdx);
  if (isAxisHorizontal) {
    mirroredGCellIdx.set(gcellIdx.x(), axisGCellIdx + (axisGCellIdx - gcellIdx.y()));
  } else {
    mirroredGCellIdx.set(axisGCellIdx + (axisGCellIdx - gcellIdx.x()), gcellIdx.y());
  }
  return mirroredGCellIdx;
}

void FlexGR::dumpSelfSymmetry2DAscii(const string &tag,
                                     const string &netName) const {
  auto block = design ? design->getTopBlock() : nullptr;
  if (block == nullptr) {
    return;
  }

  auto netIt = block->name2net.find(netName);
  if (netIt == block->name2net.end()) {
    return;
  }
  auto net = netIt->second;
  auto constraint = getSelfSymmetryConstraintPtr(net);
  if (constraint == nullptr) {
    return;
  }

  cout << "@@@ " << netName << " 2D ASCII " << tag << " @@@\n";
  cout << "net: " << net->getName() << "\n";

  auto &gCellPatterns = block->getGCellPatterns();
  if (gCellPatterns.size() < 2 || net->getNodes().empty()) {
    cout << "empty_net: " << (net->getNodes().empty() ? 1 : 0) << "\n";
    cout << "@@@ end " << netName << " 2D ASCII " << tag << " @@@\n";
    return;
  }

  int xCnt = (int)gCellPatterns.at(0).getCount();
  int yCnt = (int)gCellPatterns.at(1).getCount();
  if (xCnt <= 0 || yCnt <= 0) {
    cout << "empty_gcell_grid: 1\n";
    cout << "@@@ end " << netName << " 2D ASCII " << tag << " @@@\n";
    return;
  }

  map<frNode*, frPoint> node2GCellIdx;
  int minX = numeric_limits<int>::max();
  int minY = numeric_limits<int>::max();
  int maxX = numeric_limits<int>::min();
  int maxY = numeric_limits<int>::min();

  auto includeGCell = [&](const frPoint &gcellIdx) {
    minX = min(minX, (int)gcellIdx.x());
    minY = min(minY, (int)gcellIdx.y());
    maxX = max(maxX, (int)gcellIdx.x());
    maxY = max(maxY, (int)gcellIdx.y());
  };

  auto getNodeGCellIdx = [&](frNode *node) {
    auto it = node2GCellIdx.find(node);
    if (it != node2GCellIdx.end()) {
      return it->second;
    }
    frPoint loc;
    frPoint gcellIdx;
    node->getLoc(loc);
    block->getGCellIdx(loc, gcellIdx);
    node2GCellIdx[node] = gcellIdx;
    return gcellIdx;
  };

  for (auto &uNode: net->getNodes()) {
    includeGCell(getNodeGCellIdx(uNode.get()));
  }

  vector<string> warnings;
  auto warnNonColinear = [&](frNode *node, frNode *parent,
                             const frPoint &nodeGCellIdx,
                             const frPoint &parentGCellIdx) {
    stringstream ss;
    ss << "warning: skipped non-colinear parent-child edge"
       << " child=" << node->getId()
       << " parent=" << parent->getId()
       << " child_gcell=(" << nodeGCellIdx.x() << "," << nodeGCellIdx.y() << ")"
       << " parent_gcell=(" << parentGCellIdx.x() << "," << parentGCellIdx.y() << ")";
    warnings.push_back(ss.str());
  };

  for (auto &uNode: net->getNodes()) {
    auto node = uNode.get();
    auto parent = node->getParent();
    if (parent != nullptr) {
      auto nodeGCellIdx = getNodeGCellIdx(node);
      auto parentGCellIdx = getNodeGCellIdx(parent);
      includeGCell(nodeGCellIdx);
      includeGCell(parentGCellIdx);
      if (nodeGCellIdx.x() == parentGCellIdx.x() ||
          nodeGCellIdx.y() == parentGCellIdx.y()) {
        minX = min(minX, (int)min(nodeGCellIdx.x(), parentGCellIdx.x()));
        minY = min(minY, (int)min(nodeGCellIdx.y(), parentGCellIdx.y()));
        maxX = max(maxX, (int)max(nodeGCellIdx.x(), parentGCellIdx.x()));
        maxY = max(maxY, (int)max(nodeGCellIdx.y(), parentGCellIdx.y()));
      } else {
        warnNonColinear(node, parent, nodeGCellIdx, parentGCellIdx);
      }
    }
  }

  frNode *axisRefNode = net->getRootGCellNode();
  if (axisRefNode == nullptr) {
    axisRefNode = net->getRoot();
  }
  if (axisRefNode == nullptr) {
    axisRefNode = net->getNodes().front().get();
  }
  frPoint axisRefLoc;
  axisRefNode->getLoc(axisRefLoc);
  frPoint axisProbe;
  if (constraint->isAxisHorizontal) {
    axisProbe.set(axisRefLoc.x(), constraint->axis);
  } else {
    axisProbe.set(constraint->axis, axisRefLoc.y());
  }
  frPoint axisGCellLocation;
  block->getGCellIdx(axisProbe, axisGCellLocation);
  int axisGCellIdx = constraint->isAxisHorizontal ?
                     (int)axisGCellLocation.y() :
                     (int)axisGCellLocation.x();

  if (constraint->isAxisHorizontal) {
    if (axisGCellIdx >= minY - 1 && axisGCellIdx <= maxY + 1) {
      minY = min(minY, axisGCellIdx);
      maxY = max(maxY, axisGCellIdx);
    }
  } else {
    if (axisGCellIdx >= minX - 1 && axisGCellIdx <= maxX + 1) {
      minX = min(minX, axisGCellIdx);
      maxX = max(maxX, axisGCellIdx);
    }
  }

  minX = max(0, minX - 1);
  minY = max(0, minY - 1);
  maxX = min(xCnt - 1, maxX + 1);
  maxY = min(yCnt - 1, maxY + 1);

  bool axisVisible = constraint->isAxisHorizontal ?
                     axisGCellIdx >= minY && axisGCellIdx <= maxY :
                     axisGCellIdx >= minX && axisGCellIdx <= maxX;

  int width = maxX - minX + 1;
  int height = maxY - minY + 1;
  vector<string> grid(height, string(width, '.'));

  auto inView = [&](int xIdx, int yIdx) {
    return xIdx >= minX && xIdx <= maxX && yIdx >= minY && yIdx <= maxY;
  };

  auto cellRef = [&](int xIdx, int yIdx) -> char& {
    return grid[maxY - yIdx][xIdx - minX];
  };

  auto putAxis = [&](int xIdx, int yIdx, char axisMarker) {
    if (inView(xIdx, yIdx)) {
      auto &cell = cellRef(xIdx, yIdx);
      if (cell == '.') {
        cell = axisMarker;
      }
    }
  };

  auto putRoute = [&](int xIdx, int yIdx, char routeMarker) {
    if (!inView(xIdx, yIdx)) {
      return;
    }
    auto &cell = cellRef(xIdx, yIdx);
    if (cell == '.' || cell == '=' || cell == ':') {
      cell = routeMarker;
    } else if (cell == routeMarker || cell == '+') {
      return;
    } else {
      cell = '+';
    }
  };

  auto markerPriority = [](char marker) {
    switch (marker) {
      case 'R':
        return 4;
      case 'P':
        return 3;
      case 'S':
        return 2;
      case '+':
        return 1;
      default:
        return 0;
    }
  };

  auto putNode = [&](int xIdx, int yIdx, char nodeMarker) {
    if (!inView(xIdx, yIdx)) {
      return;
    }
    auto &cell = cellRef(xIdx, yIdx);
    if (markerPriority(nodeMarker) >= markerPriority(cell)) {
      cell = nodeMarker;
    }
  };

  if (axisVisible) {
    if (constraint->isAxisHorizontal) {
      for (int xIdx = minX; xIdx <= maxX; xIdx++) {
        putAxis(xIdx, axisGCellIdx, '=');
      }
    } else {
      for (int yIdx = minY; yIdx <= maxY; yIdx++) {
        putAxis(axisGCellIdx, yIdx, ':');
      }
    }
  }

  for (auto &uNode: net->getNodes()) {
    auto node = uNode.get();
    auto parent = node->getParent();
    if (parent != nullptr) {
      auto nodeGCellIdx = getNodeGCellIdx(node);
      auto parentGCellIdx = getNodeGCellIdx(parent);
      if (nodeGCellIdx.x() != parentGCellIdx.x() ||
          nodeGCellIdx.y() != parentGCellIdx.y()) {
        if (nodeGCellIdx.y() == parentGCellIdx.y()) {
          int yIdx = nodeGCellIdx.y();
          int beginX = min(nodeGCellIdx.x(), parentGCellIdx.x());
          int endX = max(nodeGCellIdx.x(), parentGCellIdx.x());
          for (int xIdx = beginX; xIdx <= endX; xIdx++) {
            putRoute(xIdx, yIdx, '-');
          }
        } else if (nodeGCellIdx.x() == parentGCellIdx.x()) {
          int xIdx = nodeGCellIdx.x();
          int beginY = min(nodeGCellIdx.y(), parentGCellIdx.y());
          int endY = max(nodeGCellIdx.y(), parentGCellIdx.y());
          for (int yIdx = beginY; yIdx <= endY; yIdx++) {
            putRoute(xIdx, yIdx, '|');
          }
        }
      }
    }
  }

  auto rootPin = net->getRoot();
  for (auto &uNode: net->getNodes()) {
    auto node = uNode.get();
    auto gcellIdx = getNodeGCellIdx(node);
    char nodeMarker = 'S';
    if (node->getType() == frNodeTypeEnum::frcPin ||
        node->getType() == frNodeTypeEnum::frcBoundaryPin) {
      nodeMarker = node == rootPin ? 'R' : 'P';
    }
    putNode(gcellIdx.x(), gcellIdx.y(), nodeMarker);
  }

  cout << "bbox_gcell: ll=(" << minX << "," << minY << ")"
       << " ur=(" << maxX << "," << maxY << ")\n";
  cout << "axis: "
       << (constraint->isAxisHorizontal ? "horizontal y=" : "vertical x=")
       << constraint->axis << ", "
       << (constraint->isAxisHorizontal ? "gcell_y=" : "gcell_x=")
       << axisGCellIdx
       << ", visible=" << (axisVisible ? 1 : 0) << "\n";
  cout << "legend: R root pin, P pin, S steiner, - horizontal, | vertical, "
       << "+ crossing/branch, = horizontal axis, : vertical axis, . empty\n";
  for (auto &warning: warnings) {
    cout << warning << "\n";
  }

  int yLabelWidth = (int)to_string(maxY).size();
  for (int yIdx = maxY; yIdx >= minY; yIdx--) {
    cout << "y=" << setw(yLabelWidth) << yIdx << " " << grid[maxY - yIdx] << "\n";
  }
  cout << "x_range: " << minX << ".." << maxX << "\n";
  cout << "@@@ end " << netName << " 2D ASCII " << tag << " @@@\n";
}

void FlexGR::modSelfSymmetrySourceDemand(frNet *net,
                                         const frPoint &begin,
                                         const frPoint &end,
                                         frLayerNum layerNum,
                                         bool isAdd,
                                         bool is2D) {
  if (begin == end) {
    return;
  }
  if (begin.x() != end.x() && begin.y() != end.y()) {
    cout << "Error: non-colinear nodes in modSelfSymmetrySourceDemand";
    if (net) {
      cout << " for net " << net->getName();
    }
    cout << "\n";
    return;
  }

  frPoint bp, ep;
  if (begin < end) {
    bp = begin;
    ep = end;
  } else {
    bp = end;
    ep = begin;
  }

  frPoint bpIdx, epIdx;
  design->getTopBlock()->getGCellIdx(bp, bpIdx);
  design->getTopBlock()->getGCellIdx(ep, epIdx);
  if (bpIdx == epIdx) {
    return;
  }
  if (bpIdx.x() != epIdx.x() && bpIdx.y() != epIdx.y()) {
    cout << "Error: non-colinear nodes in modSelfSymmetrySourceDemand";
    if (net) {
      cout << " for net " << net->getName();
    }
    cout << "\n";
    return;
  }

  auto targetCMap = is2D ? cmap2D.get() : cmap.get();
  unsigned zIdx = is2D ? 0 : layerNum / 2 - 1;
  auto modRawDemand = [&](int xIdx, int yIdx, frDirEnum dir) {
    if (isAdd) {
      targetCMap->addRawDemand(xIdx, yIdx, zIdx, dir);
    } else {
      targetCMap->subRawDemand(xIdx, yIdx, zIdx, dir);
    }
  };

  if (bpIdx.y() == epIdx.y()) {
    int yIdx = bpIdx.y();
    for (int xIdx = bpIdx.x(); xIdx < epIdx.x(); xIdx++) {
      modRawDemand(xIdx, yIdx, frDirEnum::E);
      modRawDemand(xIdx + 1, yIdx, frDirEnum::E);
    }
  } else {
    int xIdx = bpIdx.x();
    for (int yIdx = bpIdx.y(); yIdx < epIdx.y(); yIdx++) {
      modRawDemand(xIdx, yIdx, frDirEnum::N);
      modRawDemand(xIdx, yIdx + 1, frDirEnum::N);
    }
  }
}

void FlexGR::modSelfSymmetryMirrorShadowDemand(frNet *net,
                                               const frPoint &begin,
                                               const frPoint &end,
                                               frLayerNum layerNum,
                                               bool isAdd,
                                               bool is2D) {
  auto constraint = getSelfSymmetryConstraintPtr(net);
  if (constraint == nullptr) {
    return;
  }
  if (isOnSelfSymmetryAxis(begin, *constraint) &&
      isOnSelfSymmetryAxis(end, *constraint)) {
    return;
  }

  modSelfSymmetrySourceDemand(net,
                              mirrorPoint(begin, *constraint),
                              mirrorPoint(end, *constraint),
                              layerNum,
                              isAdd,
                              is2D);
}

void FlexGR::modSelfSymmetrySourceAndShadowDemand(frNet *net,
                                                  const frPoint &begin,
                                                  const frPoint &end,
                                                  frLayerNum layerNum,
                                                  bool isAdd,
                                                  bool is2D) {
  modSelfSymmetrySourceDemand(net, begin, end, layerNum, isAdd, is2D);
  modSelfSymmetryMirrorShadowDemand(net, begin, end, layerNum, isAdd, is2D);
}

void FlexGR::searchRepairSelfSymmetryMirror() {
  cout << "self-symmetry mirror repair...\n";
  buildSelfSymmetryMirror2DTopology();
  cout << "done self-symmetry mirror repair...\n";
}

void FlexGR::buildSelfSymmetryMirror2DTopology() {
  auto block = design ? design->getTopBlock() : nullptr;
  if (block == nullptr) {
    return;
  }
  for (auto &uNet: block->getNets()) {
    if (isSelfSymmetryNet(uNet.get())) {
      buildSelfSymmetryMirror2DTopology_net(uNet.get());
    }
  }
}

FlexGR::SelfSymmetryMirror2DStats
FlexGR::buildSelfSymmetryMirror2DTopology_net(frNet *net) {
  SelfSymmetryMirror2DStats stats;
  auto block = design ? design->getTopBlock() : nullptr;
  auto constraint = getSelfSymmetryConstraintPtr(net);
  if (block == nullptr || constraint == nullptr || net->getRoot() == nullptr) {
    return stats;
  }

  auto getGCellIdxFromLoc = [&](const frPoint &loc) {
    frPoint gcellIdx;
    block->getGCellIdx(loc, gcellIdx);
    return gcellIdx;
  };

  auto getGCellCenter = [&](const frPoint &gcellIdx) {
    frBox gcellBox;
    block->getGCellBox(gcellIdx, gcellBox);
    return frPoint((gcellBox.left() + gcellBox.right()) / 2,
                   (gcellBox.bottom() + gcellBox.top()) / 2);
  };

  auto containsNode = [&](frNode *candidate) {
    if (candidate == nullptr) {
      return false;
    }
    for (auto &uNode: net->getNodes()) {
      if (uNode.get() == candidate) {
        return true;
      }
    }
    return false;
  };

  auto findCurrentRootGCellNode = [&]() -> frNode* {
    auto rootNode = net->getRoot();
    if (rootNode) {
      for (auto child: rootNode->getChildren()) {
        if (child && child->getType() == frNodeTypeEnum::frcSteiner &&
            containsNode(child)) {
          return child;
        }
      }
    }
    if (containsNode(net->getRootGCellNode()) &&
        net->getRootGCellNode()->getType() == frNodeTypeEnum::frcSteiner) {
      return net->getRootGCellNode();
    }
    if (containsNode(net->getFirstNonRPinNode()) &&
        net->getFirstNonRPinNode()->getType() == frNodeTypeEnum::frcSteiner) {
      return net->getFirstNonRPinNode();
    }
    return nullptr;
  };

  auto refreshFirstNonRPinNode = [&]() {
    int skipCnt = (int)net->getRPins().size();
    int nodeCnt = 0;
    for (auto &uNode: net->getNodes()) {
      if (nodeCnt++ < skipCnt) {
        continue;
      }
      net->setFirstNonRPinNode(uNode.get());
      return;
    }
  };

  auto refreshTopologyCaches = [&]() {
    auto &gcellIdx2Nodes = net2GCellIdx2Nodes[net];
    auto &gcellNodes = net2GCellNodes[net];
    auto &steinerNodes = net2SteinerNodes[net];
    auto &gcellNode2RPinNodes = net2GCellNode2RPinNodes[net];
    gcellIdx2Nodes.clear();
    gcellNodes.clear();
    steinerNodes.clear();
    gcellNode2RPinNodes.clear();

    set<frNode*> gcellNodeSet;
    set<frNode*> steinerNodeSet;
    for (auto &uNode: net->getNodes()) {
      auto node = uNode.get();
      if (node->getPin() == nullptr) {
        continue;
      }
      frPoint pinLoc;
      node->getLoc(pinLoc);
      frPoint pinGCellIdx = getGCellIdxFromLoc(pinLoc);
      gcellIdx2Nodes[getSelfSymmetryGCellKey(pinGCellIdx)].push_back(node);

      frNode *gcellNode = nullptr;
      if (node == net->getRoot()) {
        for (auto child: node->getChildren()) {
          if (child->getType() == frNodeTypeEnum::frcSteiner) {
            gcellNode = child;
            break;
          }
        }
      } else {
        gcellNode = node->getParent();
      }
      if (gcellNode && gcellNode->getType() == frNodeTypeEnum::frcSteiner) {
        if (gcellNodeSet.insert(gcellNode).second) {
          gcellNodes.push_back(gcellNode);
        }
        gcellNode2RPinNodes[gcellNode].push_back(node);
      }
    }

    for (auto &uNode: net->getNodes()) {
      auto node = uNode.get();
      if (node->getType() != frNodeTypeEnum::frcSteiner ||
          gcellNodeSet.find(node) != gcellNodeSet.end()) {
        continue;
      }
      if (steinerNodeSet.insert(node).second) {
        steinerNodes.push_back(node);
      }
    }
    if (gcellNodes.empty() && net->getRootGCellNode()) {
      gcellNodes.push_back(net->getRootGCellNode());
    }
  };

  frNode *rootGCellNode = findCurrentRootGCellNode();
  if (rootGCellNode == nullptr) {
    if (net->getName() == "Symmtry5") {
      cout << "@@@ self-symmetry mirror repair 2d @@@\n";
      cout << "net: " << net->getName() << "\n";
      cout << "mirror_hanan_pins_covered: 0/0\n";
      cout << "mirror_guide_edges: 0\n";
      cout << "mirror_repair_guide_hits: 0\n";
      cout << "mirror_repair_guide_misses: 0\n";
      cout << "mirror_repair_pins_covered: 0/0\n";
      cout << "@@@ end self-symmetry mirror repair 2d @@@\n";
    }
    return stats;
  }
  net->setRootGCellNode(rootGCellNode);

  frPoint rootGCellLoc;
  rootGCellNode->getLoc(rootGCellLoc);
  frPoint rootGCellIdx = getGCellIdxFromLoc(rootGCellLoc);
  frPoint axisProbe;
  if (constraint->isAxisHorizontal) {
    axisProbe.set(rootGCellLoc.x(), constraint->axis);
  } else {
    axisProbe.set(constraint->axis, rootGCellLoc.y());
  }
  frPoint axisGCellLocation;
  block->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint->isAxisHorizontal ?
                         axisGCellLocation.y() :
                         axisGCellLocation.x();

  int rootSide = getSelfSymmetryGCellSide(rootGCellIdx,
                                          constraint->isAxisHorizontal,
                                          axisGCellIdx);
  if (rootSide == 0) {
    rootSide = -1;
  }

  map<pair<int, int>, vector<frNode*> > mirrorGCell2PinNodes;
  vector<frPoint> mirrorTerminalGCellIdxs;
  set<pair<int, int> > mirrorTerminalKeys;
  for (auto &uNode: net->getNodes()) {
    auto node = uNode.get();
    if (node->getPin() == nullptr || node == net->getRoot()) {
      continue;
    }
    frPoint pinLoc;
    node->getLoc(pinLoc);
    frPoint pinGCellIdx = getGCellIdxFromLoc(pinLoc);
    int pinSide = getSelfSymmetryGCellSide(pinGCellIdx,
                                           constraint->isAxisHorizontal,
                                           axisGCellIdx);
    if (pinSide == 0 || pinSide == rootSide) {
      continue;
    }
    auto key = getSelfSymmetryGCellKey(pinGCellIdx);
    mirrorGCell2PinNodes[key].push_back(node);
    if (mirrorTerminalKeys.insert(key).second) {
      mirrorTerminalGCellIdxs.push_back(pinGCellIdx);
    }
    stats.mirrorPins++;
  }

  vector<frPoint> rootSideTreeVertices;
  vector<pair<frPoint, frPoint> > rootSideTreeEdges;
  set<pair<int, int> > rootSideVertexKeys;
  set<pair<frPoint, frPoint> > rootSideEdgeKeys;
  map<pair<int, int>, frNode*> treeNodeByGCellIdx;

  auto rememberTreeNode = [&](frNode *node) {
    if (node == nullptr || node->getType() != frNodeTypeEnum::frcSteiner) {
      return;
    }
    frPoint loc;
    node->getLoc(loc);
    frPoint gcellIdx = getGCellIdxFromLoc(loc);
    auto key = getSelfSymmetryGCellKey(gcellIdx);
    treeNodeByGCellIdx.insert(make_pair(key, node));
    if (rootSideVertexKeys.insert(key).second) {
      rootSideTreeVertices.push_back(gcellIdx);
    }
  };

  deque<frNode*> nodeQ;
  set<frNode*> visitedNodes;
  nodeQ.push_back(rootGCellNode);
  visitedNodes.insert(rootGCellNode);
  while (!nodeQ.empty()) {
    auto node = nodeQ.front();
    nodeQ.pop_front();
    rememberTreeNode(node);
    frPoint nodeLoc, nodeGCellIdx;
    node->getLoc(nodeLoc);
    nodeGCellIdx = getGCellIdxFromLoc(nodeLoc);
    for (auto child: node->getChildren()) {
      if (child == nullptr || child->getType() != frNodeTypeEnum::frcSteiner) {
        continue;
      }
      frPoint childLoc, childGCellIdx;
      child->getLoc(childLoc);
      childGCellIdx = getGCellIdxFromLoc(childLoc);
      rememberTreeNode(child);
      auto edge = normalizeSelfSymmetryEdge(nodeGCellIdx, childGCellIdx);
      if (edge.first != edge.second && rootSideEdgeKeys.insert(edge).second) {
        rootSideTreeEdges.push_back(edge);
      }
      if (visitedNodes.insert(child).second) {
        nodeQ.push_back(child);
      }
    }
  }

  set<pair<frPoint, frPoint> > mirrorGuideEdges;
  for (auto edge: rootSideTreeEdges) {
    bool isAxisEdge =
        getSelfSymmetryAxisCoord(edge.first, constraint->isAxisHorizontal) == axisGCellIdx &&
        getSelfSymmetryAxisCoord(edge.second, constraint->isAxisHorizontal) == axisGCellIdx;
    if (isAxisEdge) {
      continue;
    }
    auto mirrorBegin = mirrorGCellIdx(edge.first,
                                      constraint->isAxisHorizontal,
                                      axisGCellIdx);
    auto mirrorEnd = mirrorGCellIdx(edge.second,
                                    constraint->isAxisHorizontal,
                                    axisGCellIdx);
    if (mirrorBegin != mirrorEnd) {
      mirrorGuideEdges.insert(normalizeSelfSymmetryEdge(mirrorBegin, mirrorEnd));
    }
  }
  stats.mirrorGuideEdges = (int)mirrorGuideEdges.size();

  vector<frPoint> mirrorTreeVertices;
  vector<pair<frPoint, frPoint> > mirrorTreeEdges;
  if (!mirrorTerminalGCellIdxs.empty()) {
    genSelfSymmetryOppositeSideTopology(mirrorTerminalGCellIdxs,
                                        constraint->isAxisHorizontal,
                                        axisGCellIdx,
                                        rootSide,
                                        rootSideTreeVertices,
                                        rootSideTreeEdges,
                                        mirrorTreeVertices,
                                        mirrorTreeEdges);
  }

  auto isConnectedToRoot = [&](frNode *node) {
    set<frNode*> seen;
    while (node != nullptr) {
      if (node == net->getRoot()) {
        return true;
      }
      if (!seen.insert(node).second) {
        return false;
      }
      node = node->getParent();
    }
    return false;
  };

  set<pair<int, int> > axisSourceKeys;
  for (auto vertex: mirrorTreeVertices) {
    if (getSelfSymmetryAxisCoord(vertex, constraint->isAxisHorizontal) != axisGCellIdx) {
      continue;
    }
    auto key = getSelfSymmetryGCellKey(vertex);
    auto nodeIt = treeNodeByGCellIdx.find(key);
    if (nodeIt != treeNodeByGCellIdx.end() && isConnectedToRoot(nodeIt->second)) {
      axisSourceKeys.insert(key);
    }
  }

  auto getOrCreateTreeNode = [&](const frPoint &gcellIdx) {
    auto key = getSelfSymmetryGCellKey(gcellIdx);
    auto nodeIt = treeNodeByGCellIdx.find(key);
    if (nodeIt != treeNodeByGCellIdx.end()) {
      return nodeIt->second;
    }
    auto uNode = make_unique<frNode>();
    auto node = uNode.get();
    node->setType(frNodeTypeEnum::frcSteiner);
    node->setLayerNum(2);
    node->setLoc(getGCellCenter(gcellIdx));
    net->addNode(uNode);
    treeNodeByGCellIdx[key] = node;
    stats.mirrorNodesCreated++;
    return node;
  };

  for (auto vertex: mirrorTreeVertices) {
    getOrCreateTreeNode(vertex);
  }

  map<pair<int, int>, vector<frPoint> > mirrorAdj;
  for (auto edge: mirrorTreeEdges) {
    mirrorAdj[getSelfSymmetryGCellKey(edge.first)].push_back(edge.second);
    mirrorAdj[getSelfSymmetryGCellKey(edge.second)].push_back(edge.first);
  }

  deque<frPoint> mirrorQ;
  set<pair<int, int> > visitedMirrorKeys;
  for (auto key: axisSourceKeys) {
    frPoint sourceGCellIdx(key.first, key.second);
    mirrorQ.push_back(sourceGCellIdx);
    visitedMirrorKeys.insert(key);
  }
  while (!mirrorQ.empty()) {
    auto currGCellIdx = mirrorQ.front();
    mirrorQ.pop_front();
    auto currNode = getOrCreateTreeNode(currGCellIdx);
    for (auto nextGCellIdx: mirrorAdj[getSelfSymmetryGCellKey(currGCellIdx)]) {
      auto nextKey = getSelfSymmetryGCellKey(nextGCellIdx);
      if (!visitedMirrorKeys.insert(nextKey).second) {
        continue;
      }
      auto nextNode = getOrCreateTreeNode(nextGCellIdx);
      if (nextNode->getParent() == nullptr) {
        nextNode->setParent(currNode);
        currNode->addChild(nextNode);
        stats.mirrorEdgesCreated++;
      }
      mirrorQ.push_back(nextGCellIdx);
    }
  }

  auto isGuideCovered = [&](const pair<frPoint, frPoint> &edge) {
    for (auto guideEdge: mirrorGuideEdges) {
      if (selfSymmetrySegmentCovers(guideEdge.first, guideEdge.second,
                                    edge.first, edge.second)) {
        return true;
      }
    }
    return false;
  };
  for (auto edge: mirrorTreeEdges) {
    bool isAxisEdge =
        getSelfSymmetryAxisCoord(edge.first, constraint->isAxisHorizontal) == axisGCellIdx &&
        getSelfSymmetryAxisCoord(edge.second, constraint->isAxisHorizontal) == axisGCellIdx;
    if (isAxisEdge) {
      continue;
    }
    if (isGuideCovered(edge)) {
      stats.mirrorRepairGuideHits++;
    } else {
      stats.mirrorRepairGuideMisses++;
    }
  }

  for (auto &[gcellKey, pinNodes]: mirrorGCell2PinNodes) {
    frPoint gcellIdx(gcellKey.first, gcellKey.second);
    auto gcellNode = getOrCreateTreeNode(gcellIdx);
    for (auto pinNode: pinNodes) {
      if (pinNode->getParent() == nullptr) {
        gcellNode->addChild(pinNode);
        pinNode->setParent(gcellNode);
      }
    }
  }

  for (auto &[gcellKey, pinNodes]: mirrorGCell2PinNodes) {
    for (auto pinNode: pinNodes) {
      if (isConnectedToRoot(pinNode)) {
        stats.mirrorHananPinsCovered++;
      }
    }
  }
  stats.mirrorRepairPinsCovered = stats.mirrorHananPinsCovered;

  refreshFirstNonRPinNode();
  refreshTopologyCaches();

  if (net->getName() == "Symmtry5") {
    cout << "@@@ self-symmetry mirror repair 2d @@@\n";
    cout << "net: " << net->getName() << "\n";
    cout << "mirror_hanan_pins_covered: "
         << stats.mirrorHananPinsCovered << "/" << stats.mirrorPins << "\n";
    cout << "mirror_guide_edges: " << stats.mirrorGuideEdges << "\n";
    cout << "mirror_repair_guide_hits: " << stats.mirrorRepairGuideHits << "\n";
    cout << "mirror_repair_guide_misses: " << stats.mirrorRepairGuideMisses << "\n";
    cout << "mirror_repair_pins_covered: "
         << stats.mirrorRepairPinsCovered << "/" << stats.mirrorPins << "\n";
    cout << "mirror_nodes_created: " << stats.mirrorNodesCreated << "\n";
    cout << "mirror_edges_created: " << stats.mirrorEdgesCreated << "\n";
    cout << "@@@ end self-symmetry mirror repair 2d @@@\n";
  }

  return stats;
}
