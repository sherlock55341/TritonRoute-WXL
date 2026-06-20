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

#include "gr/FlexGRGridGraph.h"
#include "gr/FlexGR.h"
#include "gr/FlexGR_self_sym_utils.h"
#include <cmath>

using namespace std;
using namespace fr;

namespace {

  bool isSelfSymmetryCardinalDir(frDirEnum dir) {
    return dir == frDirEnum::E || dir == frDirEnum::N ||
           dir == frDirEnum::S || dir == frDirEnum::W;
  }

  unsigned long long getInvalidMirrorPenalty(long long edgeLen) {
    return (unsigned long long)std::max(1ll, edgeLen) *
           ((unsigned long long)MARKERCOST * 8);
  }

  unsigned long long getViaStepCost(frCoord edgeLen) {
    return (unsigned long long)edgeLen * (1 + 128);
  }

  bool getSelfSymmetryEdgeGCells(FlexGRGridGraph* gridGraph,
                                 frMIdx x,
                                 frMIdx y,
                                 frDirEnum dir,
                                 frPoint &beginGCellIdx,
                                 frPoint &endGCellIdx) {
    if (!gridGraph || !isSelfSymmetryCardinalDir(dir)) {
      return false;
    }

    auto worker = gridGraph->getGRWorker();
    if (!worker) {
      return false;
    }
    auto x2 = x;
    auto y2 = y;
    switch (dir) {
      case frDirEnum::E:
        ++x2;
        break;
      case frDirEnum::S:
        --y2;
        break;
      case frDirEnum::W:
        --x2;
        break;
      case frDirEnum::N:
        ++y2;
        break;
      default:
        return false;
    }

    auto routeGCellIdxLL = worker->getRouteGCellIdxLL();
    beginGCellIdx.set(x + routeGCellIdxLL.x(), y + routeGCellIdxLL.y());
    endGCellIdx.set(x2 + routeGCellIdxLL.x(), y2 + routeGCellIdxLL.y());
    return true;
  }

  int getSelfSymmetryEdgeSide(FlexGRGridGraph* gridGraph,
                              const SelfSymmetryAxisContext &axisCtx,
                              frMIdx x,
                              frMIdx y,
                              frDirEnum dir) {
    frPoint beginGCellIdx;
    frPoint endGCellIdx;
    if (!getSelfSymmetryEdgeGCells(gridGraph, x, y, dir,
                                   beginGCellIdx, endGCellIdx)) {
      return 0;
    }

    auto beginCoord = axisCtx.axisCoord(beginGCellIdx);
    auto endCoord = axisCtx.axisCoord(endGCellIdx);
    if (beginCoord == axisCtx.axisGCellIdx &&
        endCoord == axisCtx.axisGCellIdx) {
      return 0;
    }
    if (beginCoord < axisCtx.axisGCellIdx &&
        endCoord < axisCtx.axisGCellIdx) {
      return -1;
    }
    if (beginCoord > axisCtx.axisGCellIdx &&
        endCoord > axisCtx.axisGCellIdx) {
      return 1;
    }
    return 0;
  }

  bool getSelfSymmetryAxisContext(FlexGRGridGraph* gridGraph,
                                  frNet* net,
                                  SelfSymmetryAxisContext &axisCtx) {
    if (!gridGraph || !net || !net->getSelfSymmetryConstraintPtr()) {
      return false;
    }

    auto worker = gridGraph->getGRWorker();
    if (!worker) {
      return false;
    }
    auto constraint = net->getSelfSymmetryConstraint();
    auto routeBox = worker->getRouteBox();
    frPoint axisProbe;
    if (constraint.isAxisHorizontal) {
      axisProbe.set(routeBox.left(), constraint.axis);
    } else {
      axisProbe.set(constraint.axis, routeBox.bottom());
    }
    axisCtx = SelfSymmetryAxisContext::fromAxisProbe(
        gridGraph->getDesign(), constraint, axisProbe);
    return axisCtx.valid;
  }

  int getSelfSymmetryRootSide(FlexGRGridGraph* gridGraph,
                              frNet* net,
                              const SelfSymmetryAxisContext &axisCtx) {
    auto block = gridGraph && gridGraph->getDesign() ?
                 gridGraph->getDesign()->getTopBlock() : nullptr;
    auto rootGCellNode = net ? net->getRootGCellNode() : nullptr;
    if (!block || !rootGCellNode) {
      return normalizeSelfSymmetryRootSide(0);
    }

    frPoint rootGCellIdx;
    block->getGCellIdx(rootGCellNode->getLoc(), rootGCellIdx);
    auto rootCoord = axisCtx.axisCoord(rootGCellIdx);
    if (rootCoord < axisCtx.axisGCellIdx) {
      return -1;
    }
    if (rootCoord > axisCtx.axisGCellIdx) {
      return 1;
    }
    return normalizeSelfSymmetryRootSide(0);
  }

  bool getSelfSymmetryMirrorEdge(FlexGRGridGraph* gridGraph,
                                 frNet* net,
                                 frMIdx x,
                                 frMIdx y,
                                 frMIdx z,
                                 frDirEnum dir,
                                 frMIdx &mirrorX,
                                 frMIdx &mirrorY,
                                 frMIdx &mirrorZ,
                                 frDirEnum &mirrorDir) {
    if (!gridGraph || !net || !net->getSelfSymmetryConstraintPtr() ||
        !isSelfSymmetryCardinalDir(dir)) {
      return false;
    }

    auto worker = gridGraph->getGRWorker();
    if (!worker) {
      return false;
    }
    auto constraint = net->getSelfSymmetryConstraint();
    auto routeGCellIdxLL = worker->getRouteGCellIdxLL();
    auto routeGCellIdxUR = worker->getRouteGCellIdxUR();
    auto routeBox = worker->getRouteBox();
    frPoint axisProbe;
    if (constraint.isAxisHorizontal) {
      axisProbe.set(routeBox.left(), constraint.axis);
    } else {
      axisProbe.set(constraint.axis, routeBox.bottom());
    }
    auto axisCtx = SelfSymmetryAxisContext::fromAxisProbe(
        gridGraph->getDesign(), constraint, axisProbe);
    if (!axisCtx.valid) {
      return false;
    }

    frPoint beginGCellIdx;
    frPoint endGCellIdx;
    if (!getSelfSymmetryEdgeGCells(gridGraph, x, y, dir,
                                   beginGCellIdx, endGCellIdx)) {
      return false;
    }
    if (axisCtx.axisCoord(beginGCellIdx) == axisCtx.axisGCellIdx &&
        axisCtx.axisCoord(endGCellIdx) == axisCtx.axisGCellIdx) {
      return false;
    }

    auto mirrorBegin = axisCtx.mirrorGCell(beginGCellIdx);
    auto mirrorEnd = axisCtx.mirrorGCell(endGCellIdx);
    auto getDir = [](const frPoint &begin, const frPoint &end) {
      if (begin.x() != end.x()) {
        return begin.x() < end.x() ? frDirEnum::E : frDirEnum::W;
      }
      if (begin.y() != end.y()) {
        return begin.y() < end.y() ? frDirEnum::N : frDirEnum::S;
      }
      return frDirEnum::UNKNOWN;
    };
    mirrorDir = getDir(mirrorBegin, mirrorEnd);
    if (!isSelfSymmetryCardinalDir(mirrorDir)) {
      return false;
    }
    if (mirrorBegin.x() < routeGCellIdxLL.x() ||
        mirrorBegin.x() > routeGCellIdxUR.x() ||
        mirrorBegin.y() < routeGCellIdxLL.y() ||
        mirrorBegin.y() > routeGCellIdxUR.y() ||
        mirrorEnd.x() < routeGCellIdxLL.x() ||
        mirrorEnd.x() > routeGCellIdxUR.x() ||
        mirrorEnd.y() < routeGCellIdxLL.y() ||
        mirrorEnd.y() > routeGCellIdxUR.y()) {
      return false;
    }

    mirrorX = mirrorBegin.x() - routeGCellIdxLL.x();
    mirrorY = mirrorBegin.y() - routeGCellIdxLL.y();
    mirrorZ = z;
    return true;
  }

}

bool FlexGRGridGraph::search(vector<FlexMazeIdx> &connComps, grNode* nextPinNode,
                             vector<FlexMazeIdx> &path, FlexMazeIdx &ccMazeIdx1, 
                             FlexMazeIdx &ccMazeIdx2, const frPoint &centerPt) {
  bool enableOutput = false;
  int stepCnt = 0;

  // prep nextPinBox
  frMIdx xDim, yDim, zDim;
  getDim(xDim, yDim, zDim);
  FlexMazeIdx dstMazeIdx1(xDim - 1, yDim - 1, zDim - 1);
  FlexMazeIdx dstMazeIdx2(0, 0, 0);
  FlexMazeIdx mi;

  auto loc = nextPinNode->getLoc();
  auto lNum = nextPinNode->getLayerNum();
  getMazeIdx(loc, lNum, mi);
  // update dstMazeIdx1, dstMazeIdx2
  dstMazeIdx1.set(min(dstMazeIdx1.x(), mi.x()),
                  min(dstMazeIdx1.y(), mi.y()),
                  min(dstMazeIdx1.z(), mi.z()));
  dstMazeIdx2.set(max(dstMazeIdx2.x(), mi.x()),
                  max(dstMazeIdx2.y(), mi.y()),
                  max(dstMazeIdx2.z(), mi.z()));

  wavefront = FlexGRWavefront();

  frPoint currPt;
  // push connected components to wavefront
  for (auto &idx: connComps) {
    if (isDst(idx.x(), idx.y(), idx.z())) {
      if (enableOutput) {
        cout <<"message: astarSearch dst covered (" <<idx.x() <<", " <<idx.y() <<", " <<idx.z() <<")" <<endl;
      }
      path.push_back(FlexMazeIdx(idx.x(), idx.y(), idx.z()));
      return true;
    }
    getPoint(idx.x(), idx.y(), currPt);
    frCoord currDist = abs(currPt.x() - centerPt.x()) + abs(currPt.y() - centerPt.y());
    FlexGRWavefrontGrid currGrid(idx.x(), idx.y(), idx.z(), currDist, 0, getEstCost(idx, dstMazeIdx1, dstMazeIdx2, frDirEnum::UNKNOWN));
    wavefront.push(currGrid);
    if (enableOutput) {
      cout <<"src add to wavefront (" <<idx.x() <<", " <<idx.y() <<", " <<idx.z() <<")" <<endl;
    }
  }

  while (!wavefront.empty()) {
    auto currGrid = wavefront.top();
    wavefront.pop();
    if (getPrevAstarNodeDir(currGrid.x(), currGrid.y(), currGrid.z()) != frDirEnum::UNKNOWN) {
      continue;
    }
    // test
    if (enableOutput) {
      ++stepCnt;
    }
    // if (stepCnt % 100000 == 0) {
    //   std::cout << "wavefront size = " << wavefront.size() << " at step = " << stepCnt << "\n";
    // }
    if (isDst(currGrid.x(), currGrid.y(), currGrid.z())) {
      traceBackPath(currGrid, path, connComps, ccMazeIdx1, ccMazeIdx2);
      if (enableOutput) {
        cout << "path found. stepCnt = " << stepCnt << "\n";
      }
      return true;
    } else {
      // expand and update wavefront
      expandWavefront(currGrid, dstMazeIdx1, dstMazeIdx2, centerPt);
    }
  }
  return false;
}

frCost FlexGRGridGraph::getEstCost(const FlexMazeIdx &src, const FlexMazeIdx &dstMazeIdx1,
                                   const FlexMazeIdx &dstMazeIdx2, const frDirEnum &dir) {
  if (activeNet && activeNet->getSelfSymmetryConstraintPtr()) {
    return 0;
  }
  bool enableOutput = false;
  if (enableOutput) {
    cout <<"est from (" <<src.x() <<", " <<src.y() <<", " <<src.z() <<") "
         <<"to ("       <<dstMazeIdx1.x() <<", " <<dstMazeIdx1.y() <<", " <<dstMazeIdx1.z() <<") ("
                        <<dstMazeIdx2.x() <<", " <<dstMazeIdx2.y() <<", " <<dstMazeIdx2.z() <<")";
  }
  // bend cost
  int bendCnt = 0;
  frPoint srcPoint, dstPoint1, dstPoint2;
  getPoint(src.x(), src.y(), srcPoint);
  getPoint(dstMazeIdx1.x(), dstMazeIdx1.y(), dstPoint1);
  getPoint(dstMazeIdx2.x(), dstMazeIdx2.y(), dstPoint2);
  //auto minCostX = std::abs(srcPoint.x() - dstPoint.x()) * 1;
  //auto minCostY = std::abs(srcPoint.y() - dstPoint.y()) * 1;
  //auto minCostZ = std::abs(gridGraph.getZHeight(src.z()) - gridGraph.getZHeight(dst.z())) * VIACOST;
  frCoord minCostX = max(max(dstPoint1.x() - srcPoint.x(), srcPoint.x() - dstPoint2.x()), 0);
  frCoord minCostY = max(max(dstPoint1.y() - srcPoint.y(), srcPoint.y() - dstPoint2.y()), 0);
  frCoord minCostZ = max(max(getZHeight(dstMazeIdx1.z()) - getZHeight(src.z()), 
                             getZHeight(src.z()) - getZHeight(dstMazeIdx2.z())), 0);
  if (enableOutput) {
    cout <<" x/y/z min cost = (" <<minCostX <<", " <<minCostY <<", " <<minCostZ <<") " <<endl;
  }

  bendCnt += (minCostX && dir != frDirEnum::UNKNOWN && dir != frDirEnum::E && dir != frDirEnum::W) ? 1 : 0;
  bendCnt += (minCostY && dir != frDirEnum::UNKNOWN && dir != frDirEnum::S && dir != frDirEnum::N) ? 1 : 0;
  bendCnt += (minCostZ && dir != frDirEnum::UNKNOWN && dir != frDirEnum::U && dir != frDirEnum::D) ? 1 : 0;
  if (enableOutput) {
    cout << "  est cost = " << minCostX + minCostY + minCostZ + bendCnt << endl;
  }
  return (minCostX + minCostY + minCostZ + bendCnt);
}

void FlexGRGridGraph::traceBackPath(const FlexGRWavefrontGrid &currGrid, vector<FlexMazeIdx> &path, vector<FlexMazeIdx> &root,
                                    FlexMazeIdx &ccMazeIdx1, FlexMazeIdx &ccMazeIdx2) {
  bool enableOutput = false;
  if (enableOutput) {
    cout << "    start traceBackPath...\n";
  }
  frDirEnum prevDir = frDirEnum::UNKNOWN, currDir = frDirEnum::UNKNOWN;
  int currX = currGrid.x(), currY = currGrid.y(), currZ = currGrid.z();
  // pop content in buffer
  auto backTraceBuffer = currGrid.getBackTraceBuffer();
  for (int i = 0; i < GRWAVEFRONTBUFFERSIZE; ++i) {
    // current grid is src
    if (isSrc(currX, currY, currZ)) {
      break;
    }
    // get last direction
    currDir = getLastDir(backTraceBuffer);
    backTraceBuffer >>= DIRBITSIZE;
    if (currDir == frDirEnum::UNKNOWN) {
      cout << "Warning: unexpected direction in tracBackPath\n";
      break;
    }
    root.push_back(FlexMazeIdx(currX, currY, currZ));
    // push point to path
    if (currDir != prevDir) {
      path.push_back(FlexMazeIdx(currX, currY, currZ));
      if (enableOutput) {
        cout <<" -- (" <<currX <<", " <<currY <<", " <<currZ <<")";
      }
    }
    getPrevGrid(currX, currY, currZ, currDir);
    prevDir = currDir;
  }
  // trace back according to grid prev dir
  while (isSrc(currX, currY, currZ) == false) {
    // get last direction
    currDir = getPrevAstarNodeDir(currX, currY, currZ);
    // add to root
    // if (prevDir != frDirEnum::UNKNOWN && 
    //     currDir != prevDir && 
    //     (currDir != frDirEnum::U && currDir != frDirEnum::D && prevDir != frDirEnum::U && prevDir != frDirEnum::U)) {
      root.push_back(FlexMazeIdx(currX, currY, currZ));
    // }
    if (currDir == frDirEnum::UNKNOWN) {
      cout << "Warning: unexpected direction in tracBackPath\n";
      break;
    }
    if (currDir != prevDir) {
      path.push_back(FlexMazeIdx(currX, currY, currZ));
      if (enableOutput) {
        cout <<" -- (" <<currX <<", " <<currY <<", " <<currZ <<")";
      }
    }
    getPrevGrid(currX, currY, currZ, currDir);
    prevDir = currDir;
  }
  // add final path to src, only add when path exists; no path exists (src = dst)
  if (!path.empty()) {
    path.push_back(FlexMazeIdx(currX, currY, currZ));
    if (enableOutput) {
      cout <<" -- (" <<currX <<", " <<currY <<", " <<currZ <<")";
    }
  }
  for (auto &mi: path) {
    ccMazeIdx1.set(min(ccMazeIdx1.x(), mi.x()),
                   min(ccMazeIdx1.y(), mi.y()),
                   min(ccMazeIdx1.z(), mi.z()));
    ccMazeIdx2.set(max(ccMazeIdx2.x(), mi.x()),
                   max(ccMazeIdx2.y(), mi.y()),
                   max(ccMazeIdx2.z(), mi.z()));
  }
  if (enableOutput) {
    cout <<endl;
  }
}

frDirEnum FlexGRGridGraph::getLastDir(const std::bitset<GRWAVEFRONTBITSIZE> &buffer) {
  auto currDirVal = buffer.to_ulong() & 0b111u;
  return static_cast<frDirEnum>(currDirVal);
}

void FlexGRGridGraph::getPrevGrid(frMIdx &gridX, frMIdx &gridY, frMIdx &gridZ, const frDirEnum dir) const {
  switch(dir) {
    case frDirEnum::E:
      --gridX;
      break;
    case frDirEnum::S:
      ++gridY;
      break;
    case frDirEnum::W:
      ++gridX;
      break;
    case frDirEnum::N:
      --gridY;
      break;
    case frDirEnum::U:
      --gridZ;
      break;
    case frDirEnum::D:
      ++gridZ;
      break;
    default:
      ;
  }
  return;
}


void FlexGRGridGraph::expandWavefront(FlexGRWavefrontGrid &currGrid, const FlexMazeIdx &dstMazeIdx1, 
                                      const FlexMazeIdx &dstMazeIdx2, const frPoint &centerPt) {
  bool enableOutput = false;
  //bool enableOutput = true;
  if (enableOutput) {
    cout << "start expand from (" << currGrid.x() << ", " << currGrid.y() << ", " << currGrid.z() << ")\n";
  }
    // N
  if (isExpandable(currGrid, frDirEnum::N)) {
    expand(currGrid, frDirEnum::N, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
  // else {
  //   std::cout <<"no N" <<endl;
  // }
  // E
  if (isExpandable(currGrid, frDirEnum::E)) {
    expand(currGrid, frDirEnum::E, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
  // else {
  //   std::cout <<"no E" <<endl;
  // }
  // S
  if (isExpandable(currGrid, frDirEnum::S)) {
    expand(currGrid, frDirEnum::S, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
  // else {
  //   std::cout <<"no S" <<endl;
  // }
  // W
  if (isExpandable(currGrid, frDirEnum::W)) {
    expand(currGrid, frDirEnum::W, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
  // else {
  //   std::cout <<"no W" <<endl;
  // }
  // U
  if (isExpandable(currGrid, frDirEnum::U)) {
    expand(currGrid, frDirEnum::U, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
  // else {
  //   std::cout <<"no U" <<endl;
  // }
  // D
  if (isExpandable(currGrid, frDirEnum::D)) {
    expand(currGrid, frDirEnum::D, dstMazeIdx1, dstMazeIdx2, centerPt);
  }
}

bool FlexGRGridGraph::isExpandable(const FlexGRWavefrontGrid &currGrid, frDirEnum dir) {
  //bool enableOutput = true;
  bool enableOutput = false;
  frMIdx gridX = currGrid.x();
  frMIdx gridY = currGrid.y();
  frMIdx gridZ = currGrid.z();
  //bool hg = hasEdge(gridX, gridY, gridZ, dir) && hasGuide(gridX, gridY, gridZ, dir);
  bool hg = hasEdge(gridX, gridY, gridZ, dir);
  if (enableOutput) {
    if (!hasEdge(gridX, gridY, gridZ, dir)) {
      cout <<"no edge@(" <<gridX <<", " <<gridY <<", " <<gridZ <<") " <<(int)dir <<endl;
    }
  }
  reverse(gridX, gridY, gridZ, dir);
  if (!hg || 
      isSrc(gridX, gridY, gridZ) || 
      getPrevAstarNodeDir(gridX, gridY, gridZ) != frDirEnum::UNKNOWN ||
      currGrid.getLastDir() == dir) {
    return false;
  } else {
    return true;
  }
}

void FlexGRGridGraph::expand(FlexGRWavefrontGrid &currGrid, const frDirEnum &dir, 
                             const FlexMazeIdx &dstMazeIdx1, const FlexMazeIdx &dstMazeIdx2,
                             const frPoint &centerPt) {
  bool enableOutput = false;
  //bool enableOutput = true;
  frCost nextEstCost, nextPathCost;
  int gridX = currGrid.x();
  int gridY = currGrid.y();
  int gridZ = currGrid.z();

  auto currDir = currGrid.getLastDir();

  getNextGrid(gridX, gridY, gridZ, dir);
  
  FlexMazeIdx nextIdx(gridX, gridY, gridZ);
  // get cost
  nextEstCost = getEstCost(nextIdx, dstMazeIdx1, dstMazeIdx2, dir);
  nextPathCost = getNextPathCost(currGrid, dir);
  if (enableOutput) {
    std::cout << "  expanding from (" << currGrid.x() << ", " << currGrid.y() << ", " << currGrid.z() 
              << ") [pathCost / totalCost = " << currGrid.getPathCost() << " / " << currGrid.getCost() << "] to "
              << "(" << gridX << ", " << gridY << ", " << gridZ << ") [pathCost / totalCost = " 
              << nextPathCost << " / " << nextPathCost + nextEstCost << "]\n";
  }

  frPoint currPt;
  getPoint(gridX, gridY, currPt);
  frCoord currDist = abs(currPt.x() - centerPt.x()) + abs(currPt.y() - centerPt.y());

  FlexGRWavefrontGrid nextWavefrontGrid(gridX, gridY, gridZ, currDist, nextPathCost, nextPathCost + nextEstCost, currGrid.getBackTraceBuffer());
  // update wavefront buffer
  auto tailDir = nextWavefrontGrid.shiftAddBuffer(dir);
  // commit grid prev direction if needed
  auto tailIdx = getTailIdx(nextIdx, nextWavefrontGrid);
  if (tailDir != frDirEnum::UNKNOWN) {
    if (getPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z()) == frDirEnum::UNKNOWN ||
        getPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z()) == tailDir) {
      setPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z(), tailDir);
      wavefront.push(nextWavefrontGrid);
      if (enableOutput) {
        std::cout << "    commit (" << tailIdx.x() << ", " << tailIdx.y() << ", " << tailIdx.z() << ") prev accessing dir = " << (int)tailDir << "\n";
      }
    }
  } else {  
    // add to wavefront
    wavefront.push(nextWavefrontGrid);
  }
  return;
}

void FlexGRGridGraph::getNextGrid(frMIdx &gridX, frMIdx &gridY, frMIdx &gridZ, const frDirEnum dir) {
  switch(dir) {
    case frDirEnum::E:
      ++gridX;
      break;
    case frDirEnum::S:
      --gridY;
      break;
    case frDirEnum::W:
      --gridX;
      break;
    case frDirEnum::N:
      ++gridY;
      break;
    case frDirEnum::U:
      ++gridZ;
      break;
    case frDirEnum::D:
      --gridZ;
      break;
    default:
      ;
  }
  return;
}

frCost FlexGRGridGraph::getNextPathCost(const FlexGRWavefrontGrid &currGrid, const frDirEnum &dir) {
  bool enableOutput = false;
  frMIdx gridX = currGrid.x();
  frMIdx gridY = currGrid.y();
  frMIdx gridZ = currGrid.z();
  frCost nextPathCost = currGrid.getPathCost();
  // bending cost
  auto currDir = currGrid.getLastDir();
  if (currDir != dir && currDir != frDirEnum::UNKNOWN) {
    // original
    ++nextPathCost;
  }

  // currently no congeston on via direction
  // bool congCost   = (dir == frDirEnum::U || dir == frDirEnum::D) ? false : hasCongCost(gridX, gridY, gridZ, dir);
  bool congCost   = (dir == frDirEnum::U || dir == frDirEnum::D) ? false : true;
  
  bool blockCost  = hasBlock(gridX, gridY, gridZ, dir);
  bool overflowCost = false;
  // if (!grWorker->is2D() && gridZ <= (VIA_ACCESS_LAYERNUM / 2 - 1)) {
  //   blockCost = true;
  // }

  frMIdx tmpX = gridX;
  frMIdx tmpY = gridY;
  frMIdx tmpZ = gridZ;
  frDirEnum tmpDir = dir;
  correct(tmpX, tmpY, tmpZ, tmpDir);
  unsigned rawDemand = 0;
  unsigned rawSupply = 0;

  if (tmpDir != frDirEnum::U && tmpDir != frDirEnum::D) {
    rawDemand = getRawDemand(tmpX, tmpY, tmpZ, tmpDir);
    rawSupply = getRawSupply(tmpX, tmpY, tmpZ, tmpDir);
  }

  bool histCost   = hasHistoryCost(tmpX, tmpY, tmpZ);

  overflowCost = (rawDemand >= rawSupply * grWorker->getCongThresh());

  auto edgeLength = getEdgeLength(gridX, gridY, gridZ, dir);
  auto congStepCost = congCost ?
                      getCongCost(rawDemand,
                                  rawSupply * grWorker->getCongThresh()) *
                      edgeLength : 0;
  auto histStepCost = histCost ?
                      4 * getCongCost(rawDemand,
                                      rawSupply * grWorker->getCongThresh()) *
                      getHistoryCost(gridX, gridY, gridZ) * edgeLength : 0;
  auto blockStepCost = blockCost ? BLOCKCOST * edgeLength * 100 : 0;
  auto overflowStepCost = overflowCost ? 128 * edgeLength : 0;
  auto stepCost = edgeLength + congStepCost + histStepCost +
                  blockStepCost + overflowStepCost;
  auto selfSymmetryEdgeSide = 0;
  auto selfSymmetryRootSide = 0;
  if (activeNet && activeNet->getSelfSymmetryConstraintPtr() &&
      grWorker->isSelfSymmetryAuto() && isSelfSymmetryCardinalDir(dir)) {
    SelfSymmetryAxisContext axisCtx;
    if (getSelfSymmetryAxisContext(this, activeNet, axisCtx)) {
      selfSymmetryEdgeSide =
          getSelfSymmetryEdgeSide(this, axisCtx, gridX, gridY, dir);
      selfSymmetryRootSide =
          getSelfSymmetryRootSide(this, activeNet, axisCtx);
      if (selfSymmetryEdgeSide == -selfSymmetryRootSide) {
        stepCost = saturateSelfSymmetryCost(
            (unsigned long long)stepCost * 4);
      }
    }
  }
  if (activeNet && activeNet->getSelfSymmetryConstraintPtr() &&
      !grWorker->is2D() && isSelfSymmetryCardinalDir(dir) &&
      getLayerNum(gridZ) == VIA_ACCESS_LAYERNUM) {
    frMIdx xDim = 0;
    frMIdx yDim = 0;
    frMIdx zDim = 0;
    getDim(xDim, yDim, zDim);
    if (gridZ + 1 < zDim) {
      auto m1PenaltyCost =
          getViaStepCost(getEdgeLength(gridX, gridY, gridZ, frDirEnum::U)) *
          SELF_SYMMETRY_M1_VIA_PENALTY_COUNT;
      stepCost = saturateSelfSymmetryCost(
          (unsigned long long)stepCost + m1PenaltyCost);
    }
  }
  nextPathCost += stepCost;
  if (activeNet && activeNet->getSelfSymmetryConstraintPtr() &&
      isSelfSymmetryCardinalDir(dir)) {
    frMIdx mirrorX = 0;
    frMIdx mirrorY = 0;
    frMIdx mirrorZ = 0;
    frDirEnum mirrorDir = frDirEnum::UNKNOWN;
    auto edgeLen = getEdgeLength(gridX, gridY, gridZ, dir);
    bool addMirrorCost = grWorker->isSelfSymmetryMirror() ||
                         (grWorker->isSelfSymmetryAuto() &&
                          selfSymmetryEdgeSide == selfSymmetryRootSide &&
                          selfSymmetryEdgeSide != 0);
    if (addMirrorCost &&
        getSelfSymmetryMirrorEdge(this, activeNet, gridX, gridY, gridZ, dir,
                                  mirrorX, mirrorY, mirrorZ, mirrorDir)) {
      auto tmpMirrorX = mirrorX;
      auto tmpMirrorY = mirrorY;
      auto tmpMirrorZ = mirrorZ;
      auto tmpMirrorDir = mirrorDir;
      correct(tmpMirrorX, tmpMirrorY, tmpMirrorZ, tmpMirrorDir);
      if (tmpMirrorDir != frDirEnum::U && tmpMirrorDir != frDirEnum::D) {
        auto mirrorRawDemand = getRawDemand(tmpMirrorX, tmpMirrorY,
                                            tmpMirrorZ, tmpMirrorDir);
        auto mirrorRawSupply = getRawSupply(tmpMirrorX, tmpMirrorY,
                                            tmpMirrorZ, tmpMirrorDir);
        auto mirrorCost =
            getCongCost(mirrorRawDemand,
                        mirrorRawSupply * grWorker->getCongThresh()) *
            edgeLen * 5;
        if (getHistoryCost(tmpMirrorX, tmpMirrorY, tmpMirrorZ)) {
          mirrorCost += 4 *
                        getCongCost(mirrorRawDemand,
                                    mirrorRawSupply * grWorker->getCongThresh()) *
                        getHistoryCost(tmpMirrorX, tmpMirrorY, tmpMirrorZ) *
                        edgeLen;
        }
        if (hasBlock(tmpMirrorX, tmpMirrorY, tmpMirrorZ, tmpMirrorDir)) {
          mirrorCost += BLOCKCOST * edgeLen * 100;
        }
        if (mirrorRawDemand >= mirrorRawSupply * grWorker->getCongThresh()) {
          mirrorCost += 128 * edgeLen;
        }
        nextPathCost += mirrorCost;
      }
    } else if (addMirrorCost && grWorker->isSelfSymmetryMirror()) {
      nextPathCost += saturateSelfSymmetryCost(getInvalidMirrorPenalty(edgeLen));
    }
  }
  // discourage using layer below VIA_ACCESS_LAYERNUM
  // if ((tmpZ + 1) * 2 <= VIA_ACCESS_LAYERNUM && !grWorker->is2D() && (dir != frDirEnum::U && dir != frDirEnum::D)) {
  //   nextPathCost += BLOCKCOST * getEdgeLength(gridX, gridY, gridZ, dir) * 100;
  // }
  return nextPathCost;
}

double FlexGRGridGraph::getCongCost(int demand, int supply) {
  return (demand * (4 / (1.0 + exp(supply - demand))) / (supply + 1));
  // return (demand * demand * (8) / (supply + 1) / (supply + 1));
}

FlexMazeIdx FlexGRGridGraph::getTailIdx(const FlexMazeIdx &currIdx, const FlexGRWavefrontGrid &currGrid) {
  int gridX = currIdx.x();
  int gridY = currIdx.y();
  int gridZ = currIdx.z();
  auto backTraceBuffer = currGrid.getBackTraceBuffer();
  for (int i = 0; i < GRWAVEFRONTBUFFERSIZE; ++i) {
    int currDirVal = backTraceBuffer.to_ulong() - ((backTraceBuffer.to_ulong() >> DIRBITSIZE) << DIRBITSIZE);
    frDirEnum currDir = static_cast<frDirEnum>(currDirVal);
    backTraceBuffer >>= DIRBITSIZE;
    getPrevGrid(gridX, gridY, gridZ, currDir);
  }
  return FlexMazeIdx(gridX, gridY, gridZ);
}
