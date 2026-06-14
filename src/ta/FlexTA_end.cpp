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

#include "ta/FlexTA.h"
#include "gr/FlexGR_self_sym_utils.h"

using namespace std;
using namespace fr;

namespace {

  struct SelfSymmetryTAAxisSnapStats {
    int nets = 0;
    int routes = 0;
    int snapped = 0;
    int skipped = 0;
  };

  bool isSelfSymmetryTAAxisRouteDummy(const frPoint &begin,
                                      const frPoint &end) {
    auto delta = end.x() - begin.x() + end.y() - begin.y();
    return delta == 0 || delta == 1;
  }

  bool isSelfSymmetryTAAxisAlignedSegment(
      const frPoint &begin,
      const frPoint &end,
      const SelfSymmetryAxisContext &axisCtx) {
    if (axisCtx.isAxisHorizontal) {
      return begin.y() == end.y() && begin.x() != end.x();
    }
    return begin.x() == end.x() && begin.y() != end.y();
  }

  frNode* getSelfSymmetryTAReferenceNode(frNet *net) {
    if (net == nullptr) {
      return nullptr;
    }
    auto rootNode = net->getRootGCellNode();
    if (rootNode == nullptr) {
      rootNode = net->getRoot();
    }
    if (rootNode == nullptr && !net->getNodes().empty()) {
      rootNode = net->getNodes().front().get();
    }
    return rootNode;
  }

  void collectSelfSymmetryTAPinLocations(frNet *net, set<frPoint> &pinLocs) {
    if (net == nullptr) {
      return;
    }
    for (auto &uNode: net->getNodes()) {
      auto node = uNode.get();
      if (node == nullptr || node->getPin() == nullptr) {
        continue;
      }
      frPoint loc;
      node->getLoc(loc);
      pinLocs.insert(loc);
    }
  }

  frCoord getSelfSymmetryTAAxisRouteLength(
      const frPoint &begin,
      const frPoint &end,
      const SelfSymmetryAxisContext &axisCtx) {
    return axisCtx.isAxisHorizontal ?
           getSelfSymmetryAbsDiff(begin.x(), end.x()) :
           getSelfSymmetryAbsDiff(begin.y(), end.y());
  }

  frCoord getSelfSymmetryTAAxisRouteDistance(
      const frPoint &begin,
      const SelfSymmetryAxisContext &axisCtx) {
    return axisCtx.isAxisHorizontal ?
           getSelfSymmetryAbsDiff(begin.y(), axisCtx.axis) :
           getSelfSymmetryAbsDiff(begin.x(), axisCtx.axis);
  }

  bool isSelfSymmetryTAAxisPinLocalRoute(const frPoint &begin,
                                         const frPoint &end,
                                         const set<frPoint> &pinLocs,
                                         const SelfSymmetryAxisContext &axisCtx,
                                         frCoord minAxisRouteLength) {
    if (getSelfSymmetryTAAxisRouteLength(begin, end, axisCtx) >=
        minAxisRouteLength) {
      return false;
    }
    return pinLocs.find(begin) != pinLocs.end() ||
           pinLocs.find(end) != pinLocs.end();
  }

  void printSelfSymmetryTAAxisSnapTrace(const char *status,
                                        frNet *net,
                                        frGuide *guide,
                                        frPathSeg *pathSeg,
                                        const SelfSymmetryAxisContext &axisCtx,
                                        const frPoint &oldBegin,
                                        const frPoint &oldEnd,
                                        const frPoint &newBegin,
                                        const frPoint &newEnd,
                                        const char *reason) {
    frBox guideBox;
    if (guide != nullptr) {
      guide->getBBox(guideBox);
    }
    cout << "SSTA_AXIS_SNAP_TRACE"
         << " status=" << status
         << " net=" << (net == nullptr ? string("null") : net->getName())
         << " axis=" << axisCtx.axis
         << " originalAxis=" << axisCtx.originalAxis
         << " isHorizontal=" << (axisCtx.isAxisHorizontal ? 1 : 0)
         << " rootSide=" << axisCtx.rootSide
         << " guideBox=" << guideBox.left() << "," << guideBox.bottom()
         << ":" << guideBox.right() << "," << guideBox.top()
         << ":layer=" << (guide == nullptr ? -1 : guide->getBeginLayerNum())
         << " old=" << oldBegin.x() << "," << oldBegin.y()
         << ":" << oldEnd.x() << "," << oldEnd.y()
         << ":layer=" << (pathSeg == nullptr ? -1 : pathSeg->getLayerNum())
         << " new=" << newBegin.x() << "," << newBegin.y()
         << ":" << newEnd.x() << "," << newEnd.y()
         << " reason=" << reason << "\n";
  }

  void printSelfSymmetryTAAxisSnapSummary(
      const SelfSymmetryTAAxisSnapStats &stats) {
    cout << "SSTA_AXIS_SNAP_SUMMARY"
         << " nets=" << stats.nets
         << " routes=" << stats.routes
         << " snapped=" << stats.snapped
         << " skipped=" << stats.skipped
         << "\n";
  }

  bool snapSelfSymmetryTAAxisRoute(frNet *net,
                                   frGuide *guide,
                                   frPathSeg *pathSeg,
                                   const SelfSymmetryAxisContext &axisCtx,
                                   const set<frPoint> &pinLocs,
                                   frCoord minAxisRouteLength,
                                   frCoord maxAxisSnapDistance,
                                   SelfSymmetryTAAxisSnapStats &stats) {
    if (guide == nullptr || pathSeg == nullptr) {
      return false;
    }

    frPoint begin;
    frPoint end;
    pathSeg->getPoints(begin, end);
    frPoint newBegin(begin);
    frPoint newEnd(end);
    ++stats.routes;

    if (isSelfSymmetryTAAxisRouteDummy(begin, end)) {
      ++stats.skipped;
      printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                       begin, end, newBegin, newEnd, "dummy");
      return false;
    }
    if (!isSelfSymmetryTAAxisAlignedSegment(begin, end, axisCtx)) {
      ++stats.skipped;
      printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                       begin, end, newBegin, newEnd,
                                       "wrong_direction");
      return false;
    }
    if (isSelfSymmetryTAAxisPinLocalRoute(begin, end, pinLocs, axisCtx,
                                          minAxisRouteLength)) {
      ++stats.skipped;
      printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                       begin, end, newBegin, newEnd,
                                       "pin_local_route");
      return false;
    }
    if (getSelfSymmetryTAAxisRouteLength(begin, end, axisCtx) <
        minAxisRouteLength) {
      ++stats.skipped;
      printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                       begin, end, newBegin, newEnd,
                                       "short_axis_route");
      return false;
    }
    if (getSelfSymmetryTAAxisRouteDistance(begin, axisCtx) >
        maxAxisSnapDistance) {
      ++stats.skipped;
      printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                       begin, end, newBegin, newEnd,
                                       "too_far_from_axis");
      return false;
    }

    if (axisCtx.isAxisHorizontal) {
      if (begin.y() == axisCtx.axis && end.y() == axisCtx.axis) {
        ++stats.skipped;
        printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                         begin, end, newBegin, newEnd,
                                         "already_on_axis");
        return false;
      }
      newBegin.set(begin.x(), axisCtx.axis);
      newEnd.set(end.x(), axisCtx.axis);
    } else {
      if (begin.x() == axisCtx.axis && end.x() == axisCtx.axis) {
        ++stats.skipped;
        printSelfSymmetryTAAxisSnapTrace("skip", net, guide, pathSeg, axisCtx,
                                         begin, end, newBegin, newEnd,
                                         "already_on_axis");
        return false;
      }
      newBegin.set(axisCtx.axis, begin.y());
      newEnd.set(axisCtx.axis, end.y());
    }

    pathSeg->setPoints(newBegin, newEnd);
    ++stats.snapped;
    printSelfSymmetryTAAxisSnapTrace("rewrite", net, guide, pathSeg, axisCtx,
                                     begin, end, newBegin, newEnd,
                                     "axis_snapped");
    return true;
  }

  void snapSelfSymmetryTAAxisGuides(frDesign *design,
                                    SelfSymmetryTAAxisSnapStats &stats) {
    auto block = design == nullptr ? nullptr : design->getTopBlock();
    if (block == nullptr) {
      return;
    }

    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      if (block->isRoutedNet(net->getName())) {
        continue;
      }
      auto constraint = net == nullptr ?
                        nullptr :
                        net->getSelfSymmetryConstraintPtr();
      if (constraint == nullptr) {
        continue;
      }
      ++stats.nets;

      auto refNode = getSelfSymmetryTAReferenceNode(net);
      frPoint refLoc;
      if (refNode != nullptr) {
        refNode->getLoc(refLoc);
      }
      auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(
          design, *constraint, refLoc);
      if (!axisCtx.valid || axisCtx.axisSnapFailed) {
        cout << "SSTA_AXIS_SNAP_TRACE"
             << " status=skip_net"
             << " net=" << net->getName()
             << " axis=" << constraint->axis
             << " originalAxis=" << constraint->axis
             << " isHorizontal=" << (constraint->isAxisHorizontal ? 1 : 0)
             << " rootSide=0"
             << " guideBox=0,0:0,0:layer=-1"
             << " old=0,0:0,0:layer=-1"
             << " new=0,0:0,0"
             << " reason=axis_snap_failed\n";
        continue;
      }
      axisCtx.rootSide =
          normalizeSelfSymmetryRootSide(axisCtx.sideOfPoint(refLoc));

      set<frPoint> pinLocs;
      collectSelfSymmetryTAPinLocations(net, pinLocs);
      auto &gCellPatterns = block->getGCellPatterns();
      if (gCellPatterns.size() < 2) {
        continue;
      }
      auto xGCellPitch = (frCoord)gCellPatterns[0].getSpacing();
      auto yGCellPitch = (frCoord)gCellPatterns[1].getSpacing();
      auto minAxisRouteLength = constraint->isAxisHorizontal ?
                                xGCellPitch :
                                yGCellPitch;
      auto maxAxisSnapDistance = (constraint->isAxisHorizontal ?
                                  yGCellPitch :
                                  xGCellPitch) / 2;
      for (auto &uGuide: net->getGuides()) {
        auto guide = uGuide.get();
        if (guide == nullptr) {
          continue;
        }
        for (auto &connFig: guide->getRoutes()) {
          if (connFig->typeId() != frcPathSeg) {
            ++stats.skipped;
            cout << "SSTA_AXIS_SNAP_TRACE"
                 << " status=skip"
                 << " net=" << net->getName()
                 << " axis=" << axisCtx.axis
                 << " originalAxis=" << axisCtx.originalAxis
                 << " isHorizontal=" << (axisCtx.isAxisHorizontal ? 1 : 0)
                 << " rootSide=" << axisCtx.rootSide
                 << " guideBox=0,0:0,0:layer=-1"
                 << " old=0,0:0,0:layer=-1"
                 << " new=0,0:0,0"
                 << " reason=unsupported_route\n";
            continue;
          }
          auto pathSeg = static_cast<frPathSeg*>(connFig.get());
          snapSelfSymmetryTAAxisRoute(net, guide, pathSeg, axisCtx, pinLocs,
                                      minAxisRouteLength, maxAxisSnapDistance,
                                      stats);
        }
      }
    }
  }
}

void FlexTAWorker::saveToGuides() {
  for (auto &iroute: iroutes) {
    for (auto &uPinFig: iroute->getFigs()) {
      if (uPinFig->typeId() == tacPathSeg) {
        unique_ptr<frPathSeg> pathSeg = make_unique<frPathSeg>(*static_cast<taPathSeg*>(uPinFig.get()));
        pathSeg->addToNet(iroute->getGuide()->getNet());
        auto guide = iroute->getGuide();
        vector<unique_ptr<frConnFig> > tmp;
        tmp.push_back(std::move(pathSeg));
        guide->setRoutes(tmp);
      }
      // modify upper/lower segs
      // upper/lower seg will have longest wirelength
    }
  }
}


void FlexTAWorker::end() {
  //if (getTAIter() <= 0) {
    saveToGuides();
  //}
}

void FlexTA::snapSelfSymmetryAxisGuides() {
  SelfSymmetryTAAxisSnapStats stats;
  snapSelfSymmetryTAAxisGuides(getDesign(), stats);
  printSelfSymmetryTAAxisSnapSummary(stats);
}
