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
#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

using namespace std;
using namespace fr;

namespace {
  struct SelfSymmetryLayerAssignDebug {
    bool sawDebugNet = false;
    bool printed = false;
    unsigned long long mirrorCostQueries = 0;
    unsigned long long mirrorCostHits = 0;
    unsigned long long mirrorInvalidEdges = 0;
    unsigned long long mirrorInvalidPenalty = 0;
    unsigned long long mirrorCostTotal = 0;

    void print() {
      if (!sawDebugNet || printed) {
        return;
      }
      printed = true;
      cout << "@@@ self-symmetry layer assignment mirror cost @@@\n";
      cout << "net: " << SelfSymmetryDebug::netName() << "\n";
      cout << "layerassign_mirror_cost_queries: " << mirrorCostQueries << "\n";
      cout << "layerassign_mirror_cost_hits: " << mirrorCostHits << "\n";
      cout << "layerassign_mirror_invalid_edges: " << mirrorInvalidEdges << "\n";
      cout << "layerassign_mirror_invalid_penalty: " << mirrorInvalidPenalty << "\n";
      cout << "layerassign_mirror_cost_total: " << mirrorCostTotal << "\n";
      cout << "@@@ end self-symmetry layer assignment mirror cost @@@\n";
    }
  };

  SelfSymmetryLayerAssignDebug layerAssignDebug;

  unsigned long long computeLayerAssignMirrorCongestionCost(
      frDesign *design,
      FlexGRCMap *cmap,
      const frPoint &mirrorBegin,
      const frPoint &mirrorEnd,
      frLayerNum layerNum) {
    unsigned long long mirrorCost = 0;
    if (mirrorBegin.y() == mirrorEnd.y()) {
      bool isLayerBlocked =
          design->getTech()->getLayer((layerNum + 1) * 2)->getDir() ==
          frcVertPrefRoutingDir;
      int yIdx = mirrorBegin.y();
      int xBegin = min(mirrorBegin.x(), mirrorEnd.x());
      int xEnd = max(mirrorBegin.x(), mirrorEnd.x());
      for (int xIdx = xBegin; xIdx < xEnd; xIdx++) {
        auto supply = cmap->getRawSupply(xIdx, yIdx, layerNum, frDirEnum::E);
        auto demand = cmap->getRawDemand(xIdx, yIdx, layerNum, frDirEnum::E);
        mirrorCost += cmap->getHistoryCost(xIdx, yIdx, layerNum);
        if (isLayerBlocked ||
            cmap->hasBlock(xIdx, yIdx, layerNum, frDirEnum::E)) {
          mirrorCost += (unsigned long long)BLOCKCOST * 100;
        }
        if (demand > supply / 4) {
          mirrorCost += demand * 10 / (supply + 1);
        }
        if (demand >= supply) {
          mirrorCost += (unsigned long long)MARKERCOST * 8;
        }
      }
    } else {
      bool isLayerBlocked =
          design->getTech()->getLayer((layerNum + 1) * 2)->getDir() ==
          frcHorzPrefRoutingDir;
      int xIdx = mirrorBegin.x();
      int yBegin = min(mirrorBegin.y(), mirrorEnd.y());
      int yEnd = max(mirrorBegin.y(), mirrorEnd.y());
      for (int yIdx = yBegin; yIdx < yEnd; yIdx++) {
        auto supply = cmap->getRawSupply(xIdx, yIdx, layerNum, frDirEnum::N);
        auto demand = cmap->getRawDemand(xIdx, yIdx, layerNum, frDirEnum::N);
        mirrorCost += cmap->getHistoryCost(xIdx, yIdx, layerNum);
        if (isLayerBlocked ||
            cmap->hasBlock(xIdx, yIdx, layerNum, frDirEnum::N)) {
          mirrorCost += (unsigned long long)BLOCKCOST * 100;
        }
        if (demand > supply / 4) {
          mirrorCost += demand * 10 / (supply + 1);
        }
        if (demand >= supply) {
          mirrorCost += (unsigned long long)MARKERCOST * 8;
        }
      }
    }
    return mirrorCost;
  }

  frNode* getSelfSymmetryReferenceNode(frNet *net) {
    if (net == nullptr) {
      return nullptr;
    }
    frNode *refNode = net->getRootGCellNode();
    if (refNode == nullptr) {
      refNode = net->getRoot();
    }
    if (refNode == nullptr && !net->getNodes().empty()) {
      refNode = net->getNodes().front().get();
    }
    return refNode;
  }

  SelfSymmetryAxisContext getSelfSymmetryNetAxisContext(
      frDesign *design,
      frNet *net,
      const frSelfSymmetryConstraint &constraint) {
    auto refNode = getSelfSymmetryReferenceNode(net);
    if (refNode == nullptr) {
      return SelfSymmetryAxisContext();
    }
    frPoint refLoc;
    refNode->getLoc(refLoc);
    auto ctx = SelfSymmetryAxisContext::fromReferencePoint(design,
                                                           constraint,
                                                           refLoc);
    if (!ctx.valid) {
      return ctx;
    }
    auto block = design ? design->getTopBlock() : nullptr;
    frNode *rootNode = net ? net->getRootGCellNode() : nullptr;
    if (rootNode == nullptr && net != nullptr) {
      rootNode = net->getRoot();
    }
    if (rootNode == nullptr) {
      ctx.rootSide = -1;
    } else if (block != nullptr) {
      frPoint rootLoc;
      rootNode->getLoc(rootLoc);
      ctx.rootSide =
          normalizeSelfSymmetryRootSide(ctx.sideOfPoint(rootLoc));
    }
    return ctx;
  }

  struct SelfSymmetry3DNodeDesc {
    bool valid = false;
    frPoint loc;
    frLayerNum layerNum = 0;
    frNodeTypeEnum type = frNodeTypeEnum::frcSteiner;
  };

  struct SelfSymmetry3DConnFigDesc {
    frNode *child = nullptr;
    frNode *parent = nullptr;
    SelfSymmetry3DNodeDesc childDesc;
    SelfSymmetry3DNodeDesc parentDesc;
  };

  struct SelfSymmetry3DNetState {
    vector<unique_ptr<frNode> > nodes;
    vector<unique_ptr<grShape> > shapes;
    vector<unique_ptr<grVia> > vias;
    map<frNode*, frNode*> parentByNode;
    map<frNode*, vector<frNode*> > childrenByNode;
    map<frNode*, SelfSymmetry3DNodeDesc> nodeDescByNode;
    map<grBlockObject*, SelfSymmetry3DConnFigDesc> connFigDescByObj;
    frNode *rootGCellNode = nullptr;
    frNode *firstNonRPinNode = nullptr;
  };

  struct SelfSymmetry3DState {
    bool leadOnlyActive = false;
    bool guidedActive = false;
    map<frNet*, SelfSymmetry3DNetState, frBlockObjectComp> netStates;
  };

  map<const FlexGR*, SelfSymmetry3DState> selfSymmetry3DStates;

  SelfSymmetry3DNodeDesc makeSelfSymmetry3DNodeDesc(frNode *node) {
    SelfSymmetry3DNodeDesc desc;
    if (node == nullptr) {
      return desc;
    }
    desc.valid = true;
    node->getLoc(desc.loc);
    desc.layerNum = node->getLayerNum();
    desc.type = node->getType();
    return desc;
  }

  bool sameSelfSymmetry3DNodeDesc(frNode *node,
                                  const SelfSymmetry3DNodeDesc &desc) {
    if (node == nullptr || !desc.valid) {
      return false;
    }
    frPoint loc;
    node->getLoc(loc);
    return loc == desc.loc && node->getLayerNum() == desc.layerNum;
  }

  bool containsSelfSymmetry3DNode(frNet *net, frNode *node) {
    if (net == nullptr || node == nullptr) {
      return false;
    }
    for (auto &uNode: net->getNodes()) {
      if (uNode.get() == node) {
        return true;
      }
    }
    return false;
  }

  frNode* findSelfSymmetry3DNode(frNet *net,
                                 const SelfSymmetry3DNodeDesc &desc) {
    if (net == nullptr || !desc.valid) {
      return nullptr;
    }
    for (auto &uNode: net->getNodes()) {
      if (sameSelfSymmetry3DNodeDesc(uNode.get(), desc)) {
        return uNode.get();
      }
    }
    return nullptr;
  }

  frNode* ensureSelfSymmetry3DAnchor(frNet *net,
                                     const SelfSymmetry3DNodeDesc &desc) {
    if (net == nullptr || !desc.valid) {
      return nullptr;
    }
    if (auto node = findSelfSymmetry3DNode(net, desc)) {
      return node;
    }
    auto uNode = make_unique<frNode>();
    auto node = uNode.get();
    node->setType(frNodeTypeEnum::frcSteiner);
    node->setLoc(desc.loc);
    node->setLayerNum(desc.layerNum);
    net->addNode(uNode);
    return node;
  }

  frNode* resolveSelfSymmetry3DNode(frNet *net,
                                    SelfSymmetry3DNetState &savedNet,
                                    frNode *savedNode,
                                    const SelfSymmetry3DNodeDesc &desc,
                                    const set<frNode*> &restoredNodes,
                                    bool createIfMissing) {
    if (savedNode == nullptr) {
      return nullptr;
    }
    if (restoredNodes.find(savedNode) != restoredNodes.end()) {
      return savedNode;
    }
    if (containsSelfSymmetry3DNode(net, savedNode)) {
      return savedNode;
    }
    auto it = savedNet.nodeDescByNode.find(savedNode);
    const auto &nodeDesc = (it == savedNet.nodeDescByNode.end()) ? desc : it->second;
    auto currentNode = findSelfSymmetry3DNode(net, nodeDesc);
    if (currentNode != nullptr || !createIfMissing) {
      return currentNode;
    }
    return ensureSelfSymmetry3DAnchor(net, nodeDesc);
  }

  int getSelfSymmetry3DNodeSide(frDesign *design,
                                frNode *node,
                                const SelfSymmetryAxisContext &axisCtx) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr || node == nullptr || !axisCtx.valid) {
      return 0;
    }
    frPoint loc;
    frPoint gcellIdx;
    node->getLoc(loc);
    block->getGCellIdx(loc, gcellIdx);
    return axisCtx.sideOfGCell(gcellIdx);
  }

  int getSelfSymmetry3DPointSide(frDesign *design,
                                 const frPoint &point,
                                 const SelfSymmetryAxisContext &axisCtx) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr || !axisCtx.valid) {
      return 0;
    }
    frPoint gcellIdx;
    block->getGCellIdx(point, gcellIdx);
    return axisCtx.sideOfGCell(gcellIdx);
  }

  bool isSelfSymmetry3DAxisOnly(frDesign *design,
                                const frPoint &begin,
                                const frPoint &end,
                                const SelfSymmetryAxisContext &axisCtx) {
    return getSelfSymmetry3DPointSide(design, begin, axisCtx) == 0 &&
           getSelfSymmetry3DPointSide(design, end, axisCtx) == 0;
  }

  bool isSelfSymmetry3DMirrorPathSeg(frDesign *design,
                                     grPathSeg *pathSeg,
                                     const SelfSymmetryAxisContext &axisCtx,
                                     int mirrorSide) {
    if (pathSeg == nullptr) {
      return false;
    }
    frPoint begin;
    frPoint end;
    pathSeg->getPoints(begin, end);
    return getSelfSymmetry3DPointSide(design, begin, axisCtx) == mirrorSide ||
           getSelfSymmetry3DPointSide(design, end, axisCtx) == mirrorSide;
  }

  bool isSelfSymmetry3DMirrorVia(frDesign *design,
                                 grVia *via,
                                 const SelfSymmetryAxisContext &axisCtx,
                                 int mirrorSide) {
    if (via == nullptr) {
      return false;
    }
    frPoint origin;
    via->getOrigin(origin);
    return getSelfSymmetry3DPointSide(design, origin, axisCtx) == mirrorSide;
  }

  void refreshSelfSymmetry3DNetAnchors(frNet *net) {
    if (net == nullptr) {
      return;
    }
    frNode *firstSteiner = nullptr;
    for (auto &uNode: net->getNodes()) {
      if (uNode->getType() == frNodeTypeEnum::frcSteiner) {
        firstSteiner = uNode.get();
        break;
      }
    }
    if (firstSteiner != nullptr) {
      if (net->getFirstNonRPinNode() == nullptr ||
          !containsSelfSymmetry3DNode(net, net->getFirstNonRPinNode())) {
        net->setFirstNonRPinNode(firstSteiner);
      }
      if (net->getRootGCellNode() == nullptr ||
          !containsSelfSymmetry3DNode(net, net->getRootGCellNode())) {
        net->setRootGCellNode(firstSteiner);
      }
    }
  }

}

unsigned FlexGR::getSelfSymmetryLayerAssignMirrorCost(frNode *currNode,
                                                       frNet *net,
                                                       frLayerNum layerNum) {
  auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
  if (constraint == nullptr || currNode == nullptr ||
      design == nullptr || design->getTopBlock() == nullptr) {
    return 0;
  }

  bool debugNet = SelfSymmetryDebug::isDebugNet(net);
  if (debugNet) {
    layerAssignDebug.sawDebugNet = true;
  }
  if (currNode->getParent() == nullptr) {
    if (debugNet) {
      layerAssignDebug.print();
    }
    return 0;
  }

  auto block = design->getTopBlock();
  auto &gCellPatterns = block->getGCellPatterns();
  if (gCellPatterns.size() < 2 || gCellPatterns.at(0).getCount() == 0 ||
      gCellPatterns.at(1).getCount() == 0) {
    return 0;
  }

  frPoint currLoc, parentLoc;
  currNode->getLoc(currLoc);
  currNode->getParent()->getLoc(parentLoc);

  frPoint beginIdx, endIdx;
  block->getGCellIdx(currLoc, beginIdx);
  block->getGCellIdx(parentLoc, endIdx);
  if (beginIdx == endIdx) {
    return 0;
  }

  auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(design,
                                                             *constraint,
                                                             currLoc);
  if (!axisCtx.valid) {
    return 0;
  }

  if (axisCtx.isAxisEdge(beginIdx, endIdx)) {
    return 0;
  }

  auto recordInvalidMirror = [&]() {
    unsigned long long penalty =
        computeInvalidMirrorPenalty(getSelfSymmetryEdgeLen(beginIdx, endIdx));
    if (debugNet) {
      layerAssignDebug.mirrorInvalidEdges++;
      layerAssignDebug.mirrorInvalidPenalty += penalty;
      layerAssignDebug.mirrorCostTotal += penalty;
    }
    return saturateSelfSymmetryCost(penalty);
  };

  if (debugNet) {
    layerAssignDebug.mirrorCostQueries++;
  }

  if (cmap == nullptr || layerNum < 0 || layerNum >= cmap->getNumLayers()) {
    return recordInvalidMirror();
  }

  frPoint mirrorBegin = axisCtx.mirrorGCell(beginIdx);
  frPoint mirrorEnd = axisCtx.mirrorGCell(endIdx);
  int xCnt = (int)gCellPatterns.at(0).getCount();
  int yCnt = (int)gCellPatterns.at(1).getCount();
  auto isValidGCellIdx = [&](const frPoint &gcellIdx) {
    return gcellIdx.x() >= 0 && gcellIdx.y() >= 0 &&
           gcellIdx.x() < xCnt && gcellIdx.y() < yCnt;
  };
  if (!isValidGCellIdx(mirrorBegin) || !isValidGCellIdx(mirrorEnd) ||
      (mirrorBegin.x() != mirrorEnd.x() && mirrorBegin.y() != mirrorEnd.y())) {
    return recordInvalidMirror();
  }

  unsigned long long mirrorCost =
      computeLayerAssignMirrorCongestionCost(design, cmap.get(), mirrorBegin,
                                             mirrorEnd, layerNum);

  if (debugNet) {
    layerAssignDebug.mirrorCostHits++;
    layerAssignDebug.mirrorCostTotal += mirrorCost;
  }
  return saturateSelfSymmetryCost(mirrorCost);
}

void FlexGR::dumpSelfSymmetry2DAscii(const string &tag) const {
  auto block = design ? design->getTopBlock() : nullptr;
  if (block == nullptr) {
    return;
  }

  const auto &netName = SelfSymmetryDebug::netName();
  auto netIt = block->name2net.find(netName);
  if (netIt == block->name2net.end()) {
    return;
  }
  auto net = netIt->second;
  auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
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

  frNode *axisRefNode = getSelfSymmetryReferenceNode(net);
  frPoint axisRefLoc;
  axisRefNode->getLoc(axisRefLoc);
  auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(design,
                                                             *constraint,
                                                             axisRefLoc);
  if (!axisCtx.valid) {
    return;
  }
  int axisGCellIdx = (int)axisCtx.axisGCellIdx;

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

namespace {
  void modSelfSymmetrySourceDemand(frDesign *design,
                                   FlexGRCMap *cmap,
                                   FlexGRCMap *cmap2D,
                                   frNet *net,
                                   const frPoint &begin,
                                   const frPoint &end,
                                   frLayerNum layerNum,
                                   bool isAdd,
                                   bool is2D) {
    if (begin == end) {
      return;
    }
    if (design == nullptr || design->getTopBlock() == nullptr) {
      return;
    }
    auto targetCMap = is2D ? cmap2D : cmap;
    if (targetCMap == nullptr) {
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

  void modSelfSymmetryMirrorShadowDemand(frDesign *design,
                                         FlexGRCMap *cmap,
                                         FlexGRCMap *cmap2D,
                                         frNet *net,
                                         const frPoint &begin,
                                         const frPoint &end,
                                         frLayerNum layerNum,
                                         bool isAdd,
                                         bool is2D) {
    auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
    if (constraint == nullptr) {
      return;
    }
    auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(design,
                                                               *constraint,
                                                               begin);
    if (!axisCtx.valid) {
      return;
    }
    if (axisCtx.sideOfPoint(begin) == 0 && axisCtx.sideOfPoint(end) == 0) {
      return;
    }

    modSelfSymmetrySourceDemand(design,
                                cmap,
                                cmap2D,
                                net,
                                axisCtx.mirrorPoint(begin),
                                axisCtx.mirrorPoint(end),
                                layerNum,
                                isAdd,
                                is2D);
  }

  struct SelfSymmetry3DMirrorObjects {
    set<frNode*> nodes;
    set<grShape*> shapes;
    set<grVia*> vias;

    bool empty() const {
      return nodes.empty() && shapes.empty() && vias.empty();
    }
  };

  SelfSymmetry3DMirrorObjects collectSelfSymmetry3DMirrorObjects(
      frDesign *design,
      frNet *net,
      const SelfSymmetryAxisContext &axisCtx,
      int mirrorSide) {
    SelfSymmetry3DMirrorObjects objects;
    if (net == nullptr || !axisCtx.valid) {
      return objects;
    }

    for (auto &uNode: net->getNodes()) {
      auto node = uNode.get();
      if (node == net->getRoot()) {
        continue;
      }
      if (getSelfSymmetry3DNodeSide(design, node, axisCtx) == mirrorSide) {
        objects.nodes.insert(node);
      }
    }

    for (auto &uShape: net->getGRShapes()) {
      if (uShape->typeId() != grcPathSeg) {
        continue;
      }
      auto pathSeg = static_cast<grPathSeg*>(uShape.get());
      if (isSelfSymmetry3DMirrorPathSeg(design, pathSeg, axisCtx, mirrorSide)) {
        objects.shapes.insert(pathSeg);
      }
    }

    for (auto &uVia: net->getGRVias()) {
      auto via = uVia.get();
      if (isSelfSymmetry3DMirrorVia(design, via, axisCtx, mirrorSide)) {
        objects.vias.insert(via);
      }
    }
    return objects;
  }

  void rememberSelfSymmetry3DNode(SelfSymmetry3DNetState &savedNet,
                                  frNode *node) {
    if (node != nullptr &&
        savedNet.nodeDescByNode.find(node) == savedNet.nodeDescByNode.end()) {
      savedNet.nodeDescByNode[node] = makeSelfSymmetry3DNodeDesc(node);
    }
  }

  void rememberSelfSymmetry3DConnFig(SelfSymmetry3DNetState &savedNet,
                                     grBlockObject *obj,
                                     frNode *child,
                                     frNode *parent) {
    SelfSymmetry3DConnFigDesc desc;
    desc.child = child;
    desc.parent = parent;
    desc.childDesc = makeSelfSymmetry3DNodeDesc(child);
    desc.parentDesc = makeSelfSymmetry3DNodeDesc(parent);
    savedNet.connFigDescByObj[obj] = desc;
    rememberSelfSymmetry3DNode(savedNet, child);
    rememberSelfSymmetry3DNode(savedNet, parent);
  }

  SelfSymmetry3DNetState snapshotSelfSymmetry3DTopology(
      frNet *net,
      const SelfSymmetry3DMirrorObjects &objects) {
    SelfSymmetry3DNetState savedNet;
    savedNet.rootGCellNode = net->getRootGCellNode();
    savedNet.firstNonRPinNode = net->getFirstNonRPinNode();

    for (auto node: objects.nodes) {
      rememberSelfSymmetry3DNode(savedNet, node);
      savedNet.parentByNode[node] = node->getParent();
      rememberSelfSymmetry3DNode(savedNet, node->getParent());
      auto &children = savedNet.childrenByNode[node];
      for (auto child: node->getChildren()) {
        children.push_back(child);
        rememberSelfSymmetry3DNode(savedNet, child);
      }
    }

    for (auto shape: objects.shapes) {
      rememberSelfSymmetry3DConnFig(savedNet, shape, shape->getChild(),
                                    shape->getParent());
    }
    for (auto via: objects.vias) {
      rememberSelfSymmetry3DConnFig(savedNet, via, via->getChild(),
                                    via->getParent());
    }
    return savedNet;
  }

  void detachSelfSymmetry3DTopology(SelfSymmetry3DNetState &savedNet,
                                    const SelfSymmetry3DMirrorObjects &objects) {
    for (auto node: objects.nodes) {
      auto parent = savedNet.parentByNode[node];
      if (parent != nullptr && objects.nodes.find(parent) == objects.nodes.end()) {
        parent->removeChild(node);
        node->setParent(nullptr);
      }
      auto children = savedNet.childrenByNode[node];
      for (auto child: children) {
        if (objects.nodes.find(child) == objects.nodes.end()) {
          node->removeChild(child);
          if (child->getParent() == node) {
            child->setParent(nullptr);
          }
        }
      }
    }
  }

  template <typename ModSourceDemand>
  void removeSelfSymmetry3DMirrorObjects(frNet *net,
                                         frRegionQuery *regionQuery,
                                         SelfSymmetry3DNetState &savedNet,
                                         const SelfSymmetry3DMirrorObjects &objects,
                                         ModSourceDemand modSourceDemand) {
    for (auto shape: objects.shapes) {
      auto pathSeg = static_cast<grPathSeg*>(shape);
      frPoint bp;
      frPoint ep;
      pathSeg->getPoints(bp, ep);
      modSourceDemand(bp, ep, pathSeg->getLayerNum(), /*isAdd*/false);
      regionQuery->removeGRObj(pathSeg);
      auto it = shape->getIter();
      auto ownedShape = std::move(*it);
      net->getGRShapes().erase(it);
      savedNet.shapes.push_back(std::move(ownedShape));
    }

    for (auto via: objects.vias) {
      regionQuery->removeGRObj(via);
      auto it = via->getIter();
      auto ownedVia = std::move(*it);
      net->getGRVias().erase(it);
      savedNet.vias.push_back(std::move(ownedVia));
    }

    for (auto node: objects.nodes) {
      auto it = node->getIter();
      auto ownedNode = std::move(*it);
      net->getNodes().erase(it);
      savedNet.nodes.push_back(std::move(ownedNode));
    }
  }

  set<frNode*> restoreSelfSymmetry3DNodes(frNet *net,
                                          SelfSymmetry3DNetState &savedNet) {
    set<frNode*> restoredNodes;
    for (auto &uNode: savedNet.nodes) {
      restoredNodes.insert(uNode.get());
      net->addNode(uNode);
    }
    for (auto node: restoredNodes) {
      node->setParent(nullptr);
      node->clearChildren();
    }
    return restoredNodes;
  }

  void restoreSelfSymmetry3DTopology(frNet *net,
                                     SelfSymmetry3DNetState &savedNet,
                                     const set<frNode*> &restoredNodes) {
    for (auto node: restoredNodes) {
      auto childrenIt = savedNet.childrenByNode.find(node);
      if (childrenIt == savedNet.childrenByNode.end()) {
        continue;
      }
      for (auto savedChild: childrenIt->second) {
        auto descIt = savedNet.nodeDescByNode.find(savedChild);
        SelfSymmetry3DNodeDesc desc;
        if (descIt != savedNet.nodeDescByNode.end()) {
          desc = descIt->second;
        }
        auto child = resolveSelfSymmetry3DNode(net, savedNet, savedChild,
                                               desc, restoredNodes,
                                               /*createIfMissing*/false);
        if (child == nullptr) {
          continue;
        }
        if (restoredNodes.find(child) != restoredNodes.end()) {
          node->addChild(child);
          child->setParent(node);
        }
      }
    }

    for (auto node: restoredNodes) {
      auto parentIt = savedNet.parentByNode.find(node);
      if (parentIt == savedNet.parentByNode.end()) {
        continue;
      }
      auto savedParent = parentIt->second;
      if (savedParent == nullptr ||
          restoredNodes.find(savedParent) != restoredNodes.end()) {
        continue;
      }
      auto descIt = savedNet.nodeDescByNode.find(savedParent);
      SelfSymmetry3DNodeDesc desc;
      if (descIt != savedNet.nodeDescByNode.end()) {
        desc = descIt->second;
      }
      auto parent = resolveSelfSymmetry3DNode(net, savedNet, savedParent,
                                              desc, restoredNodes,
                                              /*createIfMissing*/true);
      if (parent != nullptr) {
        parent->addChild(node);
        node->setParent(parent);
      }
    }
  }

  template <typename ModSourceDemand>
  void restoreSelfSymmetry3DMirrorObjects(frNet *net,
                                          frRegionQuery *regionQuery,
                                          SelfSymmetry3DNetState &savedNet,
                                          const set<frNode*> &restoredNodes,
                                          ModSourceDemand modSourceDemand) {
    for (auto &uShape: savedNet.shapes) {
      auto shape = uShape.get();
      auto descIt = savedNet.connFigDescByObj.find(shape);
      if (descIt != savedNet.connFigDescByObj.end()) {
        auto &desc = descIt->second;
        auto child = resolveSelfSymmetry3DNode(net, savedNet, desc.child,
                                               desc.childDesc, restoredNodes,
                                               /*createIfMissing*/true);
        auto parent = resolveSelfSymmetry3DNode(net, savedNet, desc.parent,
                                                desc.parentDesc, restoredNodes,
                                                /*createIfMissing*/true);
        shape->setChild(child);
        shape->setParent(parent);
        if (child != nullptr) {
          child->setConnFig(shape);
        }
      }
      auto pathSeg = static_cast<grPathSeg*>(shape);
      frPoint bp;
      frPoint ep;
      pathSeg->getPoints(bp, ep);
      modSourceDemand(bp, ep, pathSeg->getLayerNum(), /*isAdd*/true);
      regionQuery->addGRObj(pathSeg);
      net->addGRShape(uShape);
    }

    for (auto &uVia: savedNet.vias) {
      auto via = uVia.get();
      auto descIt = savedNet.connFigDescByObj.find(via);
      if (descIt != savedNet.connFigDescByObj.end()) {
        auto &desc = descIt->second;
        auto child = resolveSelfSymmetry3DNode(net, savedNet, desc.child,
                                               desc.childDesc, restoredNodes,
                                               /*createIfMissing*/true);
        auto parent = resolveSelfSymmetry3DNode(net, savedNet, desc.parent,
                                                desc.parentDesc, restoredNodes,
                                                /*createIfMissing*/true);
        via->setChild(child);
        via->setParent(parent);
        if (child != nullptr) {
          child->setConnFig(via);
        }
      }
      regionQuery->addGRObj(via);
      net->addGRVia(uVia);
    }
  }

  void restoreSelfSymmetry3DAnchors(frNet *net,
                                    const SelfSymmetry3DNetState &savedNet) {
    if (savedNet.rootGCellNode != nullptr &&
        containsSelfSymmetry3DNode(net, savedNet.rootGCellNode)) {
      net->setRootGCellNode(savedNet.rootGCellNode);
    } else {
      refreshSelfSymmetry3DNetAnchors(net);
    }
    if (savedNet.firstNonRPinNode != nullptr &&
        containsSelfSymmetry3DNode(net, savedNet.firstNonRPinNode)) {
      net->setFirstNonRPinNode(savedNet.firstNonRPinNode);
    } else {
      refreshSelfSymmetry3DNetAnchors(net);
    }
  }

  template <typename ModMirrorShadowDemand>
  void modSelfSymmetry3DLeadShadowDemand(frDesign *design,
                                         frNet *net,
                                         const SelfSymmetryAxisContext &axisCtx,
                                         bool isAdd,
                                         ModMirrorShadowDemand modShadowDemand) {
    if (!axisCtx.valid) {
      return;
    }
    for (auto &uShape: net->getGRShapes()) {
      if (uShape->typeId() != grcPathSeg) {
        continue;
      }
      auto pathSeg = static_cast<grPathSeg*>(uShape.get());
      frPoint bp;
      frPoint ep;
      pathSeg->getPoints(bp, ep);
      if (isSelfSymmetry3DAxisOnly(design, bp, ep, axisCtx)) {
        continue;
      }
      modShadowDemand(bp, ep, pathSeg->getLayerNum(), isAdd);
    }
  }
}

bool FlexGR::isSelfSymmetry3DLeadOnlyActive(frNet *net) const {
  auto stateIt = selfSymmetry3DStates.find(this);
  if (stateIt == selfSymmetry3DStates.end() || !stateIt->second.leadOnlyActive ||
      net == nullptr) {
    return false;
  }
  return stateIt->second.netStates.find(net) != stateIt->second.netStates.end();
}

bool FlexGR::isSelfSymmetry3DGuidedActive(frNet *net) const {
  auto stateIt = selfSymmetry3DStates.find(this);
  if (stateIt == selfSymmetry3DStates.end() || !stateIt->second.guidedActive ||
      net == nullptr) {
    return false;
  }
  return stateIt->second.netStates.find(net) != stateIt->second.netStates.end();
}

bool FlexGR::hasSelfSymmetryNets() const {
  auto block = design ? design->getTopBlock() : nullptr;
  if (block == nullptr) {
    return false;
  }
  for (auto &uNet: block->getNets()) {
    auto net = uNet.get();
    if (net != nullptr && net->getSelfSymmetryConstraintPtr() != nullptr) {
      return true;
    }
  }
  return false;
}

void FlexGR::stageSelfSymmetry3DLeadOnly() {
  auto block = design ? design->getTopBlock() : nullptr;
  if (block == nullptr || cmap == nullptr) {
    return;
  }

  auto &state = selfSymmetry3DStates[this];
  state.netStates.clear();
  state.leadOnlyActive = true;
  state.guidedActive = false;

  for (auto &uNet: block->getNets()) {
    auto net = uNet.get();
    auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
    if (constraint == nullptr) {
      continue;
    }

    auto axisCtx = getSelfSymmetryNetAxisContext(design, net, *constraint);
    if (!axisCtx.valid) {
      continue;
    }

    int mirrorSide = -axisCtx.rootSide;
    auto mirrorObjects = collectSelfSymmetry3DMirrorObjects(design, net,
                                                            axisCtx,
                                                            mirrorSide);
    if (mirrorObjects.empty()) {
      continue;
    }

    auto savedNet = snapshotSelfSymmetry3DTopology(net, mirrorObjects);
    detachSelfSymmetry3DTopology(savedNet, mirrorObjects);
    auto modSourceDemand = [&](const frPoint &bp,
                               const frPoint &ep,
                               frLayerNum layerNum,
                               bool isAdd) {
      modSelfSymmetrySourceDemand(design, cmap.get(), cmap2D.get(), net, bp,
                                  ep, layerNum, isAdd, /*is2D*/false);
    };
    removeSelfSymmetry3DMirrorObjects(net, getRegionQuery(), savedNet,
                                      mirrorObjects, modSourceDemand);
    refreshSelfSymmetry3DNetAnchors(net);
    state.netStates.emplace(net, std::move(savedNet));
  }

  for (auto &netEntry: state.netStates) {
    auto net = netEntry.first;
    auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
    if (constraint == nullptr) {
      continue;
    }
    auto axisCtx = getSelfSymmetryNetAxisContext(design, net, *constraint);
    if (!axisCtx.valid) {
      continue;
    }
    auto modShadowDemand = [&](const frPoint &bp,
                               const frPoint &ep,
                               frLayerNum layerNum,
                               bool isAdd) {
      modSelfSymmetryMirrorShadowDemand(design, cmap.get(), cmap2D.get(), net,
                                        bp, ep, layerNum, isAdd,
                                        /*is2D*/false);
    };
    modSelfSymmetry3DLeadShadowDemand(design, net, axisCtx, /*isAdd*/true,
                                      modShadowDemand);
  }
}

void FlexGR::restoreSelfSymmetry3DLayerAssignMirror() {
  auto stateIt = selfSymmetry3DStates.find(this);
  if (stateIt == selfSymmetry3DStates.end()) {
    return;
  }
  auto &state = stateIt->second;
  if (!state.leadOnlyActive) {
    return;
  }

  for (auto &netEntry: state.netStates) {
    auto net = netEntry.first;
    auto &savedNet = netEntry.second;
    auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
    if (constraint != nullptr) {
      auto axisCtx = getSelfSymmetryNetAxisContext(design, net, *constraint);
      if (axisCtx.valid) {
        auto modShadowDemand = [&](const frPoint &bp,
                                   const frPoint &ep,
                                   frLayerNum layerNum,
                                   bool isAdd) {
          modSelfSymmetryMirrorShadowDemand(design, cmap.get(), cmap2D.get(),
                                            net, bp, ep, layerNum, isAdd,
                                            /*is2D*/false);
        };
        modSelfSymmetry3DLeadShadowDemand(design, net, axisCtx,
                                          /*isAdd*/false, modShadowDemand);
      }
    }

    auto restoredNodes = restoreSelfSymmetry3DNodes(net, savedNet);
    restoreSelfSymmetry3DTopology(net, savedNet, restoredNodes);
    auto modSourceDemand = [&](const frPoint &bp,
                               const frPoint &ep,
                               frLayerNum layerNum,
                               bool isAdd) {
      modSelfSymmetrySourceDemand(design, cmap.get(), cmap2D.get(), net, bp,
                                  ep, layerNum, isAdd, /*is2D*/false);
    };
    restoreSelfSymmetry3DMirrorObjects(net, getRegionQuery(), savedNet,
                                       restoredNodes, modSourceDemand);
    restoreSelfSymmetry3DAnchors(net, savedNet);
  }

  state.leadOnlyActive = false;
  getRegionQuery()->initGRObj(getTech()->getLayers().size());
}

void FlexGR::beginSelfSymmetry3DGuidedSearchRepair() {
  auto stateIt = selfSymmetry3DStates.find(this);
  if (stateIt == selfSymmetry3DStates.end()) {
    return;
  }
  stateIt->second.guidedActive = true;
}

void FlexGR::endSelfSymmetry3DGuidedSearchRepair() {
  auto stateIt = selfSymmetry3DStates.find(this);
  if (stateIt == selfSymmetry3DStates.end()) {
    return;
  }
  auto &state = stateIt->second;
  state.guidedActive = false;
  state.netStates.clear();
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
    if (uNet->getSelfSymmetryConstraintPtr()) {
      buildSelfSymmetryMirror2DTopology_net(uNet.get());
    }
  }
}

FlexGR::SelfSymmetryMirror2DStats
FlexGR::buildSelfSymmetryMirror2DTopology_net(frNet *net) {
  SelfSymmetryMirror2DStats stats;
  auto block = design ? design->getTopBlock() : nullptr;
  auto constraint = net ? net->getSelfSymmetryConstraintPtr() : nullptr;
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
    if (SelfSymmetryDebug::isDebugNet(net)) {
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
  auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(design,
                                                             *constraint,
                                                             rootGCellLoc);
  if (!axisCtx.valid) {
    return stats;
  }
  frCoord axisGCellIdx = axisCtx.axisGCellIdx;

  frNode *rootNode = net->getRoot();
  frPoint rootLoc;
  if (rootNode != nullptr) {
    rootNode->getLoc(rootLoc);
  } else {
    rootGCellNode->getLoc(rootLoc);
  }
  int rootSide = normalizeSelfSymmetryRootSide(axisCtx.sideOfPoint(rootLoc));

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
    int pinSide = axisCtx.sideOfPoint(pinLoc);
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
    if (axisCtx.isAxisEdge(edge.first, edge.second)) {
      continue;
    }
    auto mirrorBegin = axisCtx.mirrorGCell(edge.first);
    auto mirrorEnd = axisCtx.mirrorGCell(edge.second);
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
    if (axisCtx.axisCoord(vertex) != axisGCellIdx) {
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
    if (axisCtx.isAxisEdge(edge.first, edge.second)) {
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

  if (SelfSymmetryDebug::isDebugNet(net)) {
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

namespace fr {

void get_self_symmetry_axis(const std::vector<frPoint> &points,
                            bool &is_horizontal, int &coor) {
  double mean_x = 0;
  double mean_y = 0;
  double sigma_x = 0;
  double sigma_y = 0;
  for (auto p : points) {
    mean_x += p.x();
    mean_y += p.y();
  }
  mean_x /= points.size();
  mean_y /= points.size();

  for (auto p : points) {
    auto dx = p.x() - mean_x;
    auto dy = p.y() - mean_y;
    sigma_x += dx * dx;
    sigma_y += dy * dy;
  }
  sigma_x /= points.size();
  sigma_y /= points.size();
  sigma_x = std::sqrt(sigma_x);
  sigma_y = std::sqrt(sigma_y);
  double moment_x_1 = 0;
  double moment_x_2 = 0;
  double moment_x_3 = 0;
  double moment_y_1 = 0;
  double moment_y_2 = 0;
  double moment_y_3 = 0;
  for (auto p : points) {
    auto dx = p.x() - mean_x;
    auto dy = p.y() - mean_y;
    dx /= sigma_x;
    dy /= sigma_y;
    moment_x_1 += dx * dx * dx;
    moment_x_2 += dx * dy;
    moment_x_3 += dx * dy * dy;
    moment_y_1 += dy * dy * dy;
    moment_y_2 += dy * dx;
    moment_y_3 += dy * dx * dx;
  }
  moment_x_1 /= points.size();
  moment_x_2 /= points.size();
  moment_x_3 /= points.size();
  moment_y_1 /= points.size();
  moment_y_2 /= points.size();
  moment_y_3 /= points.size();
  auto sum_moment_x =
      std::abs(moment_x_1) + std::abs(moment_x_2) + std::abs(moment_x_3);
  auto sum_moment_y =
      std::abs(moment_y_1) + std::abs(moment_y_2) + std::abs(moment_y_3);
  if (sum_moment_x * 2 < sum_moment_y) {
    is_horizontal = false;
    coor = std::round(mean_x);
  } else if (sum_moment_y * 2 < sum_moment_x) {
    is_horizontal = true;
    coor = std::round(mean_y);
  } else {
    std::vector<point_t> rtree_points;
    rtree_points.reserve(points.size());
    for (auto p : points) {
      rtree_points.push_back(point_t(p.x(), p.y()));
    }
    bgi::rtree<point_t, bgi::quadratic<16>> tree(rtree_points);
    double mirror_x_score = 0;
    double mirror_y_score = 0;
    for (auto p : points) {
      int mx = mean_x * 2 - p.x();
      int my = p.y();
      std::vector<point_t> results;
      tree.query(bgi::nearest(point_t(mx, my), 1), std::back_inserter(results));
      mirror_x_score += (results[0].x() - mx) * (results[0].x() - mx) +
                        (results[0].y() - my) * (results[0].y() - my);
      mx = p.x();
      my = mean_y * 2 - p.y();
      results.clear();
      tree.query(bgi::nearest(point_t(mx, my), 1), std::back_inserter(results));
      mirror_y_score += (results[0].x() - mx) * (results[0].x() - mx) +
                        (results[0].y() - my) * (results[0].y() - my);
    }
    if (mirror_x_score < mirror_y_score) {
      is_horizontal = false;
      coor = std::round(mean_x);
    } else {
      is_horizontal = true;
      coor = std::round(mean_y);
    }
  }
}

}
