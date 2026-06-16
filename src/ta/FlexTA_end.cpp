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

  bool snapSelfSymmetryTAAxisRoute(frPathSeg *pathSeg,
                                   const SelfSymmetryAxisContext &axisCtx) {
    if (pathSeg == nullptr) {
      return false;
    }

    frPoint begin;
    frPoint end;
    pathSeg->getPoints(begin, end);
    frPoint newBegin(begin);
    frPoint newEnd(end);

    if (isSelfSymmetryTAAxisRouteDummy(begin, end)) {
      return false;
    }
    if (!isSelfSymmetryTAAxisAlignedSegment(begin, end, axisCtx)) {
      return false;
    }

    if (axisCtx.isAxisHorizontal) {
      if (begin.y() == axisCtx.axis && end.y() == axisCtx.axis) {
        return false;
      }
      newBegin.set(begin.x(), axisCtx.axis);
      newEnd.set(end.x(), axisCtx.axis);
    } else {
      if (begin.x() == axisCtx.axis && end.x() == axisCtx.axis) {
        return false;
      }
      newBegin.set(axisCtx.axis, begin.y());
      newEnd.set(axisCtx.axis, end.y());
    }

    pathSeg->setPoints(newBegin, newEnd);
    return true;
  }

  void snapSelfSymmetryTAAxisGuides(frDesign *design) {
    auto block = design == nullptr ? nullptr : design->getTopBlock();
    if (block == nullptr) {
      return;
    }

    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      auto constraint = net == nullptr ?
                        nullptr :
                        net->getSelfSymmetryConstraintPtr();
      if (constraint == nullptr) {
        continue;
      }

      auto refNode = getSelfSymmetryTAReferenceNode(net);
      frPoint refLoc;
      if (refNode != nullptr) {
        refNode->getLoc(refLoc);
      }
      auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(
          design, *constraint, refLoc);
      if (!axisCtx.valid || axisCtx.axisSnapFailed) {
        continue;
      }
      axisCtx.rootSide =
          normalizeSelfSymmetryRootSide(axisCtx.sideOfPoint(refLoc));

      for (auto &uGuide: net->getGuides()) {
        auto guide = uGuide.get();
        if (guide == nullptr) {
          continue;
        }
        for (auto &connFig: guide->getRoutes()) {
          if (connFig->typeId() != frcPathSeg) {
            continue;
          }
          auto pathSeg = static_cast<frPathSeg*>(connFig.get());
          snapSelfSymmetryTAAxisRoute(pathSeg, axisCtx);
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
  snapSelfSymmetryTAAxisGuides(getDesign());
}
