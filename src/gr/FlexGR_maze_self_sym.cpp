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

#include "gr/FlexGR.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>

using namespace std;
using namespace fr;

namespace {
  bool isSelfSymmetry2DNet(frNet* net) {
    return net && net->getSelfSymmetryConstraintPtr();
  }

  int getAxisCoord(const frPoint &gcellIdx, bool isAxisHorizontal) {
    return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
  }

  int getGCellSide(const frPoint &gcellIdx, bool isAxisHorizontal, frCoord axisGCellIdx) {
    auto coord = getAxisCoord(gcellIdx, isAxisHorizontal);
    if (coord < axisGCellIdx) {
      return -1;
    }
    if (coord > axisGCellIdx) {
      return 1;
    }
    return 0;
  }

  unsigned saturateSelfSymmetry3DCost(unsigned long long cost) {
    return cost > numeric_limits<unsigned>::max() ?
           numeric_limits<unsigned>::max() :
           (unsigned)cost;
  }
}

void FlexGRWorker::resetSelfSymmetry2DDebug() {
  selfSym2DOldSourceSegments = 0;
  selfSym2DNewSourceSegments = 0;
  selfSym2DOldShadowCells = 0;
  selfSym2DNewShadowCells = 0;
  selfSym2DOutsideShadowDelta = 0;
  selfSym2DFrozenAxisObjs = 0;
}

bool FlexGRWorker::isSelfSymmetry2DAxisOnRouteBoxBoundary(frNet* net) const {
  if (!is2DRouting || !isSelfSymmetry2DNet(net)) {
    return false;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  if (constraint.isAxisHorizontal) {
    return axisGCellIdx == routeGCellIdxLL.y() || axisGCellIdx == routeGCellIdxUR.y();
  }
  return axisGCellIdx == routeGCellIdxLL.x() || axisGCellIdx == routeGCellIdxUR.x();
}

bool FlexGRWorker::isSelfSymmetry2DAxisInRouteBox(frNet* net) const {
  if (!is2DRouting || !isSelfSymmetry2DNet(net)) {
    return false;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  if (constraint.isAxisHorizontal) {
    return axisGCellIdx >= routeGCellIdxLL.y() && axisGCellIdx <= routeGCellIdxUR.y();
  }
  return axisGCellIdx >= routeGCellIdxLL.x() && axisGCellIdx <= routeGCellIdxUR.x();
}

bool FlexGRWorker::isSelfSymmetry2DFrozenAxisBoundaryPathSeg(grPathSeg* pathSeg) const {
  if (!pathSeg || !pathSeg->hasGrNet()) {
    return false;
  }
  auto net = pathSeg->getGrNet()->getFrNet();
  if (!isSelfSymmetry2DAxisOnRouteBoxBoundary(net)) {
    return false;
  }

  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();

  frPoint bp, ep, bpIdx, epIdx;
  pathSeg->getPoints(bp, ep);
  design->getTopBlock()->getGCellIdx(bp, bpIdx);
  design->getTopBlock()->getGCellIdx(ep, epIdx);
  auto beginAxisCoord = getAxisCoord(bpIdx, constraint.isAxisHorizontal);
  auto endAxisCoord = getAxisCoord(epIdx, constraint.isAxisHorizontal);
  if (std::min(beginAxisCoord, endAxisCoord) > axisGCellIdx ||
      std::max(beginAxisCoord, endAxisCoord) < axisGCellIdx) {
    return false;
  }

  frNode *rootNode = net->getRootGCellNode();
  if (rootNode == nullptr) {
    rootNode = net->getRoot();
  }
  int rootSide = -1;
  if (rootNode) {
    frPoint rootLoc, rootGCellIdx;
    rootNode->getLoc(rootLoc);
    design->getTopBlock()->getGCellIdx(rootLoc, rootGCellIdx);
    rootSide = getGCellSide(rootGCellIdx, constraint.isAxisHorizontal, axisGCellIdx);
    if (rootSide == 0) {
      rootSide = -1;
    }
  }

  int beginSide = getGCellSide(bpIdx, constraint.isAxisHorizontal, axisGCellIdx);
  int endSide = getGCellSide(epIdx, constraint.isAxisHorizontal, axisGCellIdx);
  return (beginSide == 0 || beginSide == rootSide) &&
         (endSide == 0 || endSide == rootSide);
}

void FlexGRWorker::mazeNetInit_collectSelfSymmetry2DFrozenAxisNodes(grNet* net,
                                                                    set<grNode*> &frozenAxisNodes) const {
  frozenAxisNodes.clear();
  for (auto &uptr: net->getRouteConnFigs()) {
    if (uptr->typeId() != grcPathSeg) {
      continue;
    }
    auto pathSeg = static_cast<grPathSeg*>(uptr.get());
    if (!isSelfSymmetry2DFrozenAxisBoundaryPathSeg(pathSeg)) {
      continue;
    }
    if (pathSeg->getGrChild()) {
      frozenAxisNodes.insert(pathSeg->getGrChild());
    }
    if (pathSeg->getGrParent()) {
      frozenAxisNodes.insert(pathSeg->getGrParent());
    }
  }
}

bool FlexGRWorker::isSelfSymmetry2DEdgeOnAxis(frNet* net, const FlexMazeIdx &begin,
                                              const FlexMazeIdx &end) const {
  if (!isSelfSymmetry2DNet(net)) {
    return false;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  frPoint beginGCellIdx(begin.x() + routeGCellIdxLL.x(), begin.y() + routeGCellIdxLL.y());
  frPoint endGCellIdx(end.x() + routeGCellIdxLL.x(), end.y() + routeGCellIdxLL.y());
  return getAxisCoord(beginGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
         getAxisCoord(endGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx;
}

bool FlexGRWorker::getSelfSymmetry2DMirrorEdge(frNet* net, frMIdx x, frMIdx y, frMIdx z,
                                               frDirEnum dir, frMIdx &mirrorX,
                                               frMIdx &mirrorY, frMIdx &mirrorZ,
                                               frDirEnum &mirrorDir) const {
  if (!isSelfSymmetry2DNet(net) ||
      (dir != frDirEnum::E && dir != frDirEnum::N &&
       dir != frDirEnum::S && dir != frDirEnum::W)) {
    return false;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();

  frMIdx x2 = x;
  frMIdx y2 = y;
  frMIdx z2 = z;
  frDirEnum dir2 = dir;
  switch (dir) {
    case frDirEnum::E:
      ++x2;
      break;
    case frDirEnum::W:
      --x2;
      break;
    case frDirEnum::N:
      ++y2;
      break;
    case frDirEnum::S:
      --y2;
      break;
    default:
      return false;
  }
  frPoint beginGCellIdx(x + routeGCellIdxLL.x(), y + routeGCellIdxLL.y());
  frPoint endGCellIdx(x2 + routeGCellIdxLL.x(), y2 + routeGCellIdxLL.y());
  if (getAxisCoord(beginGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
      getAxisCoord(endGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx) {
    return false;
  }

  frPoint mirrorBegin = beginGCellIdx;
  frPoint mirrorEnd = endGCellIdx;
  if (constraint.isAxisHorizontal) {
    mirrorBegin.set(beginGCellIdx.x(), axisGCellIdx + (axisGCellIdx - beginGCellIdx.y()));
    mirrorEnd.set(endGCellIdx.x(), axisGCellIdx + (axisGCellIdx - endGCellIdx.y()));
  } else {
    mirrorBegin.set(axisGCellIdx + (axisGCellIdx - beginGCellIdx.x()), beginGCellIdx.y());
    mirrorEnd.set(axisGCellIdx + (axisGCellIdx - endGCellIdx.x()), endGCellIdx.y());
  }
  if (mirrorBegin.x() < routeGCellIdxLL.x() || mirrorBegin.x() > routeGCellIdxUR.x() ||
      mirrorEnd.x() < routeGCellIdxLL.x() || mirrorEnd.x() > routeGCellIdxUR.x() ||
      mirrorBegin.y() < routeGCellIdxLL.y() || mirrorBegin.y() > routeGCellIdxUR.y() ||
      mirrorEnd.y() < routeGCellIdxLL.y() || mirrorEnd.y() > routeGCellIdxUR.y()) {
    return false;
  }

  mirrorX = mirrorBegin.x() - routeGCellIdxLL.x();
  mirrorY = mirrorBegin.y() - routeGCellIdxLL.y();
  mirrorZ = z;
  if (mirrorBegin.x() != mirrorEnd.x()) {
    mirrorDir = mirrorBegin.x() < mirrorEnd.x() ? frDirEnum::E : frDirEnum::W;
  } else {
    mirrorDir = mirrorBegin.y() < mirrorEnd.y() ? frDirEnum::N : frDirEnum::S;
  }
  return true;
}

int FlexGRWorker::modSelfSymmetry2DPathSegDemand(grPathSeg* pathSeg, bool isAdd) {
  FlexMazeIdx bi, ei;
  frPoint bp, ep;
  pathSeg->getPoints(bp, ep);
  gridGraph.getMazeIdx(bp, 2, bi);
  gridGraph.getMazeIdx(ep, 2, ei);
  int modCnt = 0;
  auto modRawDemand = [&](int xIdx, int yIdx, frDirEnum dir) {
    if (isAdd) {
      gridGraph.addRawDemand(xIdx, yIdx, 0, dir);
    } else {
      gridGraph.subRawDemand(xIdx, yIdx, 0, dir);
    }
    modCnt++;
  };

  if (bi.x() == ei.x()) {
    for (auto yIdx = bi.y(); yIdx < ei.y(); yIdx++) {
      modRawDemand(bi.x(), yIdx, frDirEnum::N);
      modRawDemand(bi.x(), yIdx + 1, frDirEnum::N);
    }
  } else if (bi.y() == ei.y()) {
    for (auto xIdx = bi.x(); xIdx < ei.x(); xIdx++) {
      modRawDemand(xIdx, bi.y(), frDirEnum::E);
      modRawDemand(xIdx + 1, bi.y(), frDirEnum::E);
    }
  } else {
    cout << "Error: non-colinear pathSeg in modSelfSymmetry2DPathSegDemand\n";
  }
  return modCnt;
}

int FlexGRWorker::modSelfSymmetry2DPathSegMirrorDemand(grPathSeg* pathSeg, bool isAdd,
                                                       int &outsideDelta) {
  outsideDelta = 0;
  auto net = pathSeg->getGrNet()->getFrNet();
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint bp, ep, bpIdx, epIdx;
  pathSeg->getPoints(bp, ep);
  design->getTopBlock()->getGCellIdx(bp, bpIdx);
  design->getTopBlock()->getGCellIdx(ep, epIdx);
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  if (getAxisCoord(bpIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
      getAxisCoord(epIdx, constraint.isAxisHorizontal) == axisGCellIdx) {
    return 0;
  }

  auto mirrorGCellIdx = [&](const frPoint &gcellIdx) {
    frPoint mirrored(gcellIdx);
    if (constraint.isAxisHorizontal) {
      mirrored.set(gcellIdx.x(), axisGCellIdx + (axisGCellIdx - gcellIdx.y()));
    } else {
      mirrored.set(axisGCellIdx + (axisGCellIdx - gcellIdx.x()), gcellIdx.y());
    }
    return mirrored;
  };
  frPoint mirrorBpIdx = mirrorGCellIdx(bpIdx);
  frPoint mirrorEpIdx = mirrorGCellIdx(epIdx);
  if (mirrorBpIdx == mirrorEpIdx) {
    return 0;
  }
  if (mirrorEpIdx < mirrorBpIdx) {
    std::swap(mirrorBpIdx, mirrorEpIdx);
  }

  int modCnt = 0;
  auto modRawDemand = [&](int xIdx, int yIdx, frDirEnum dir) {
    if (xIdx < routeGCellIdxLL.x() || xIdx > routeGCellIdxUR.x() ||
        yIdx < routeGCellIdxLL.y() || yIdx > routeGCellIdxUR.y()) {
      outsideDelta++;
      return;
    }
    int localXIdx = xIdx - routeGCellIdxLL.x();
    int localYIdx = yIdx - routeGCellIdxLL.y();
    if (isAdd) {
      gridGraph.addRawDemand(localXIdx, localYIdx, 0, dir);
    } else {
      gridGraph.subRawDemand(localXIdx, localYIdx, 0, dir);
    }
    modCnt++;
  };

  if (mirrorBpIdx.y() == mirrorEpIdx.y()) {
    for (int xIdx = mirrorBpIdx.x(); xIdx < mirrorEpIdx.x(); xIdx++) {
      modRawDemand(xIdx, mirrorBpIdx.y(), frDirEnum::E);
      modRawDemand(xIdx + 1, mirrorBpIdx.y(), frDirEnum::E);
    }
  } else if (mirrorBpIdx.x() == mirrorEpIdx.x()) {
    for (int yIdx = mirrorBpIdx.y(); yIdx < mirrorEpIdx.y(); yIdx++) {
      modRawDemand(mirrorBpIdx.x(), yIdx, frDirEnum::N);
      modRawDemand(mirrorBpIdx.x(), yIdx + 1, frDirEnum::N);
    }
  } else {
    cout << "Error: non-colinear mirror pathSeg in modSelfSymmetry2DPathSegMirrorDemand\n";
  }
  return modCnt;
}

int FlexGRWorker::modSelfSymmetry3DPathSegDemand(grPathSeg* pathSeg, bool isAdd) {
  FlexMazeIdx bi, ei;
  frPoint bp, ep;
  auto lNum = pathSeg->getLayerNum();
  pathSeg->getPoints(bp, ep);
  gridGraph.getMazeIdx(bp, lNum, bi);
  gridGraph.getMazeIdx(ep, lNum, ei);
  int modCnt = 0;
  auto modRawDemand = [&](int xIdx, int yIdx, int zIdx, frDirEnum dir) {
    if (isAdd) {
      gridGraph.addRawDemand(xIdx, yIdx, zIdx, dir);
    } else {
      gridGraph.subRawDemand(xIdx, yIdx, zIdx, dir);
    }
    modCnt++;
  };

  if (bi.x() == ei.x()) {
    for (auto yIdx = bi.y(); yIdx < ei.y(); yIdx++) {
      modRawDemand(bi.x(), yIdx, bi.z(), frDirEnum::N);
      modRawDemand(bi.x(), yIdx + 1, bi.z(), frDirEnum::N);
    }
  } else if (bi.y() == ei.y()) {
    for (auto xIdx = bi.x(); xIdx < ei.x(); xIdx++) {
      modRawDemand(xIdx, bi.y(), bi.z(), frDirEnum::E);
      modRawDemand(xIdx + 1, bi.y(), bi.z(), frDirEnum::E);
    }
  } else {
    cout << "Error: non-colinear pathSeg in modSelfSymmetry3DPathSegDemand\n";
  }
  return modCnt;
}

int FlexGRWorker::modSelfSymmetry3DPathSegMirrorDemand(grPathSeg* pathSeg,
                                                       bool isAdd,
                                                       int &outsideDelta) {
  outsideDelta = 0;
  auto net = pathSeg->getGrNet()->getFrNet();
  auto constraint = net->getSelfSymmetryConstraint();
  frPoint bp, ep, bpIdx, epIdx;
  auto lNum = pathSeg->getLayerNum();
  FlexMazeIdx bi, ei;
  pathSeg->getPoints(bp, ep);
  design->getTopBlock()->getGCellIdx(bp, bpIdx);
  design->getTopBlock()->getGCellIdx(ep, epIdx);
  gridGraph.getMazeIdx(bp, lNum, bi);
  gridGraph.getMazeIdx(ep, lNum, ei);

  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  if (getAxisCoord(bpIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
      getAxisCoord(epIdx, constraint.isAxisHorizontal) == axisGCellIdx) {
    return 0;
  }

  auto mirrorGCellIdx = [&](const frPoint &gcellIdx) {
    frPoint mirrored(gcellIdx);
    if (constraint.isAxisHorizontal) {
      mirrored.set(gcellIdx.x(), axisGCellIdx + (axisGCellIdx - gcellIdx.y()));
    } else {
      mirrored.set(axisGCellIdx + (axisGCellIdx - gcellIdx.x()), gcellIdx.y());
    }
    return mirrored;
  };
  frPoint mirrorBpIdx = mirrorGCellIdx(bpIdx);
  frPoint mirrorEpIdx = mirrorGCellIdx(epIdx);
  if (mirrorBpIdx == mirrorEpIdx) {
    return 0;
  }
  if (mirrorEpIdx < mirrorBpIdx) {
    std::swap(mirrorBpIdx, mirrorEpIdx);
  }

  int modCnt = 0;
  auto modRawDemand = [&](int xIdx, int yIdx, frDirEnum dir) {
    if (xIdx < routeGCellIdxLL.x() || xIdx > routeGCellIdxUR.x() ||
        yIdx < routeGCellIdxLL.y() || yIdx > routeGCellIdxUR.y()) {
      outsideDelta++;
      return;
    }
    int localXIdx = xIdx - routeGCellIdxLL.x();
    int localYIdx = yIdx - routeGCellIdxLL.y();
    if (isAdd) {
      gridGraph.addRawDemand(localXIdx, localYIdx, bi.z(), dir);
    } else {
      gridGraph.subRawDemand(localXIdx, localYIdx, bi.z(), dir);
    }
    modCnt++;
  };

  if (mirrorBpIdx.y() == mirrorEpIdx.y()) {
    for (int xIdx = mirrorBpIdx.x(); xIdx < mirrorEpIdx.x(); xIdx++) {
      modRawDemand(xIdx, mirrorBpIdx.y(), frDirEnum::E);
      modRawDemand(xIdx + 1, mirrorBpIdx.y(), frDirEnum::E);
    }
  } else if (mirrorBpIdx.x() == mirrorEpIdx.x()) {
    for (int yIdx = mirrorBpIdx.y(); yIdx < mirrorEpIdx.y(); yIdx++) {
      modRawDemand(mirrorBpIdx.x(), yIdx, frDirEnum::N);
      modRawDemand(mirrorBpIdx.x(), yIdx + 1, frDirEnum::N);
    }
  } else {
    cout << "Error: non-colinear mirror pathSeg in modSelfSymmetry3DPathSegMirrorDemand\n";
  }
  return modCnt;
}

frCost FlexGRWorker::getSelfSymmetry3DGuidedMirrorCost(frNet* net,
                                                       frMIdx x,
                                                       frMIdx y,
                                                       frMIdx z,
                                                       frDirEnum dir) {
  if (!net || !net->getSelfSymmetryConstraintPtr() ||
      !gr->isSelfSymmetry3DGuidedActive(net) ||
      (dir != frDirEnum::E && dir != frDirEnum::N &&
       dir != frDirEnum::S && dir != frDirEnum::W)) {
    return 0;
  }

  auto constraint = net->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();

  frMIdx x2 = x;
  frMIdx y2 = y;
  switch (dir) {
    case frDirEnum::E:
      ++x2;
      break;
    case frDirEnum::W:
      --x2;
      break;
    case frDirEnum::N:
      ++y2;
      break;
    case frDirEnum::S:
      --y2;
      break;
    default:
      return 0;
  }

  frPoint beginGCellIdx(x + routeGCellIdxLL.x(), y + routeGCellIdxLL.y());
  frPoint endGCellIdx(x2 + routeGCellIdxLL.x(), y2 + routeGCellIdxLL.y());
  if (getAxisCoord(beginGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
      getAxisCoord(endGCellIdx, constraint.isAxisHorizontal) == axisGCellIdx) {
    return 0;
  }

  auto edgeLen = max<frCoord>(1, gridGraph.getEdgeLength(x, y, z, dir));
  unsigned long long invalidPenalty =
      (unsigned long long)edgeLen *
      ((unsigned long long)BLOCKCOST * 100 + (unsigned long long)MARKERCOST * 8);
  auto getInvalidPenalty = [&]() {
    return saturateSelfSymmetry3DCost(invalidPenalty);
  };

  frPoint mirrorBegin = beginGCellIdx;
  frPoint mirrorEnd = endGCellIdx;
  if (constraint.isAxisHorizontal) {
    mirrorBegin.set(beginGCellIdx.x(), axisGCellIdx + (axisGCellIdx - beginGCellIdx.y()));
    mirrorEnd.set(endGCellIdx.x(), axisGCellIdx + (axisGCellIdx - endGCellIdx.y()));
  } else {
    mirrorBegin.set(axisGCellIdx + (axisGCellIdx - beginGCellIdx.x()), beginGCellIdx.y());
    mirrorEnd.set(axisGCellIdx + (axisGCellIdx - endGCellIdx.x()), endGCellIdx.y());
  }

  frMIdx zDimX = 0;
  frMIdx zDimY = 0;
  frMIdx zDim = 0;
  gridGraph.getDim(zDimX, zDimY, zDim);
  if (z < 0 || z >= zDim ||
      mirrorBegin.x() < routeGCellIdxLL.x() || mirrorBegin.x() > routeGCellIdxUR.x() ||
      mirrorEnd.x() < routeGCellIdxLL.x() || mirrorEnd.x() > routeGCellIdxUR.x() ||
      mirrorBegin.y() < routeGCellIdxLL.y() || mirrorBegin.y() > routeGCellIdxUR.y() ||
      mirrorEnd.y() < routeGCellIdxLL.y() || mirrorEnd.y() > routeGCellIdxUR.y()) {
    return getInvalidPenalty();
  }

  frMIdx mirrorX = mirrorBegin.x() - routeGCellIdxLL.x();
  frMIdx mirrorY = mirrorBegin.y() - routeGCellIdxLL.y();
  frDirEnum mirrorDir = frDirEnum::UNKNOWN;
  if (mirrorBegin.x() != mirrorEnd.x()) {
    mirrorDir = mirrorBegin.x() < mirrorEnd.x() ? frDirEnum::E : frDirEnum::W;
  } else if (mirrorBegin.y() != mirrorEnd.y()) {
    mirrorDir = mirrorBegin.y() < mirrorEnd.y() ? frDirEnum::N : frDirEnum::S;
  } else {
    return getInvalidPenalty();
  }

  frMIdx tmpMirrorX = mirrorX;
  frMIdx tmpMirrorY = mirrorY;
  frMIdx tmpMirrorZ = z;
  frDirEnum tmpMirrorDir = mirrorDir;
  if (tmpMirrorDir == frDirEnum::W) {
    --tmpMirrorX;
    tmpMirrorDir = frDirEnum::E;
  } else if (tmpMirrorDir == frDirEnum::S) {
    --tmpMirrorY;
    tmpMirrorDir = frDirEnum::N;
  }
  if (tmpMirrorX < 0 || tmpMirrorY < 0 ||
      tmpMirrorX >= zDimX || tmpMirrorY >= zDimY) {
    return getInvalidPenalty();
  }

  unsigned mirrorRawDemand = gridGraph.getRawDemand(tmpMirrorX, tmpMirrorY,
                                                    tmpMirrorZ, tmpMirrorDir);
  unsigned mirrorRawSupply = gridGraph.getRawSupply(tmpMirrorX, tmpMirrorY,
                                                    tmpMirrorZ, tmpMirrorDir);
  bool mirrorOverflowCost = mirrorRawDemand >= mirrorRawSupply * getCongThresh();
  unsigned long long mirrorCost =
      (unsigned long long)(gridGraph.getCongCost(mirrorRawDemand,
                                                 mirrorRawSupply * getCongThresh()) *
                           edgeLen);
  auto histCost = gridGraph.getHistoryCost(tmpMirrorX, tmpMirrorY, tmpMirrorZ);
  if (histCost) {
    mirrorCost +=
        (unsigned long long)(4 * gridGraph.getCongCost(mirrorRawDemand,
                                                       mirrorRawSupply * getCongThresh()) *
                             histCost * edgeLen);
  }
  if (gridGraph.hasBlock(tmpMirrorX, tmpMirrorY, tmpMirrorZ, tmpMirrorDir)) {
    mirrorCost += (unsigned long long)BLOCKCOST * edgeLen * 100;
  }
  if (mirrorOverflowCost) {
    mirrorCost += (unsigned long long)128 * edgeLen;
  }

  return saturateSelfSymmetry3DCost(mirrorCost);
}

bool FlexGRWorker::hasSelfSymmetry2DAxisContact(grNet* net) const {
  if (!isSelfSymmetry2DNet(net->getFrNet())) {
    return false;
  }
  auto constraint = net->getFrNet()->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();
  for (auto &node: net->getNodes()) {
    frPoint gcellIdx;
    design->getTopBlock()->getGCellIdx(node->getLoc(), gcellIdx);
    if (getAxisCoord(gcellIdx, constraint.isAxisHorizontal) == axisGCellIdx &&
        (node->hasParent() || node->hasChildren())) {
      return true;
    }
  }
  return false;
}

void FlexGRWorker::routeNet_collectFrozenAxisEndpoints(grNet* net,
                                                       map<FlexMazeIdx, grNode*> &mazeIdx2FrozenAxisEndpoint) {
  mazeIdx2FrozenAxisEndpoint.clear();
  for (auto &uptr: net->getRouteConnFigs()) {
    if (uptr->typeId() != grcPathSeg) {
      continue;
    }
    auto pathSeg = static_cast<grPathSeg*>(uptr.get());
    if (!isSelfSymmetry2DFrozenAxisBoundaryPathSeg(pathSeg)) {
      continue;
    }
    for (auto endpointNode: {pathSeg->getGrChild(), pathSeg->getGrParent()}) {
      if (!endpointNode) {
        continue;
      }
      FlexMazeIdx mi;
      gridGraph.getMazeIdx(endpointNode->getLoc(), endpointNode->getLayerNum(), mi);
      mazeIdx2FrozenAxisEndpoint[mi] = endpointNode;
    }
  }
}

void FlexGRWorker::routeNet_addSelfSymmetry2DAxisEndpoint(
    grNet* net,
    set<grNode*, frBlockObjectComp> &unConnPinGCellNodes,
    map<FlexMazeIdx, grNode*> &mazeIdx2unConnPinGCellNode,
    map<FlexMazeIdx, grNode*> &mazeIdx2endPointNode) {
  if (!isSelfSymmetry2DAxisInRouteBox(net->getFrNet())) {
    return;
  }

  auto constraint = net->getFrNet()->getSelfSymmetryConstraint();
  frPoint axisProbe;
  if (constraint.isAxisHorizontal) {
    axisProbe.set(routeBox.left(), constraint.axis);
  } else {
    axisProbe.set(constraint.axis, routeBox.bottom());
  }
  frPoint axisGCellLocation;
  design->getTopBlock()->getGCellIdx(axisProbe, axisGCellLocation);
  frCoord axisGCellIdx = constraint.isAxisHorizontal ? axisGCellLocation.y() : axisGCellLocation.x();

  frPoint rootGCellIdx = routeGCellIdxLL;
  if (!net->getPinGCellNodes().empty()) {
    design->getTopBlock()->getGCellIdx(net->getPinGCellNodes()[0]->getLoc(), rootGCellIdx);
  }
  frPoint dstGCellIdx = rootGCellIdx;
  if (constraint.isAxisHorizontal) {
    dstGCellIdx.set(std::max(routeGCellIdxLL.x(), std::min(routeGCellIdxUR.x(), rootGCellIdx.x())),
                    axisGCellIdx);
  } else {
    dstGCellIdx.set(axisGCellIdx,
                    std::max(routeGCellIdxLL.y(), std::min(routeGCellIdxUR.y(), rootGCellIdx.y())));
  }
  FlexMazeIdx mi(dstGCellIdx.x() - routeGCellIdxLL.x(),
                 dstGCellIdx.y() - routeGCellIdxLL.y(),
                 0);
  if (mazeIdx2endPointNode.find(mi) != mazeIdx2endPointNode.end()) {
    return;
  }

  frPoint axisLoc;
  gridGraph.getPoint(mi.x(), mi.y(), axisLoc);
  auto uAxisNode = make_unique<grNode>();
  auto axisNode = uAxisNode.get();
  axisNode->addToNet(net);
  axisNode->setLoc(axisLoc);
  axisNode->setLayerNum(2);
  axisNode->setType(frNodeTypeEnum::frcSteiner);
  net->addNode(uAxisNode);
  gridGraph.setDst(mi);
  unConnPinGCellNodes.insert(axisNode);
  mazeIdx2unConnPinGCellNode[mi] = axisNode;
  mazeIdx2endPointNode[mi] = axisNode;
}

void FlexGRWorker::printSelfSymmetry2DDebug(grNet* net, bool mustTouchAxis,
                                            bool axisContactAfter) const {
  if (!net || !net->getFrNet() || net->getFrNet()->getName() != "Symmtry5" ||
      !isSelfSymmetry2DNet(net->getFrNet()) || !is2DRouting) {
    return;
  }
  bool axisOnBoundary = isSelfSymmetry2DAxisOnRouteBoxBoundary(net->getFrNet());
  cout << "@@@ self-symmetry search-repair 2d @@@\n";
  cout << "net: " << net->getFrNet()->getName() << "\n";
  cout << "worker_gcell: ll=(" << routeGCellIdxLL.x() << "," << routeGCellIdxLL.y()
       << ") ur=(" << routeGCellIdxUR.x() << "," << routeGCellIdxUR.y() << ")\n";
  cout << "axis_on_boundary: " << (axisOnBoundary ? 1 : 0) << "\n";
  cout << "frozen_axis_objs: " << selfSym2DFrozenAxisObjs << "\n";
  cout << "must_touch_axis: " << (mustTouchAxis ? 1 : 0) << "\n";
  cout << "axis_contact_after: " << (axisContactAfter ? 1 : 0) << "\n";
  if (axisContactAfter && axisOnBoundary) {
    cout << "axis_contact_loc: frozen_boundary\n";
  } else if (axisContactAfter) {
    cout << "axis_contact_loc: axis\n";
  } else {
    cout << "axis_contact_loc: null\n";
  }
  cout << "old_source_segments: " << selfSym2DOldSourceSegments << "\n";
  cout << "new_source_segments: " << selfSym2DNewSourceSegments << "\n";
  cout << "old_shadow_cells: " << selfSym2DOldShadowCells << "\n";
  cout << "new_shadow_cells: " << selfSym2DNewShadowCells << "\n";
  cout << "outside_shadow_delta: " << selfSym2DOutsideShadowDelta << "\n";
  cout << "mirror_cost_queries: " << gridGraph.getMirrorCostQueries() << "\n";
  cout << "mirror_cost_total: " << gridGraph.getMirrorCostTotal() << "\n";
  cout << "@@@ end self-symmetry search-repair 2d @@@\n";
}
