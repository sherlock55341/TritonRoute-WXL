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
  void normalizeGuidePoints(frPoint &begin, frPoint &end) {
    if (end < begin) {
      swap(begin, end);
    }
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

  frPathSeg* getGuidePathSeg(frGuide *guide) {
    if (guide == nullptr) {
      return nullptr;
    }
    for (auto &connFig: guide->getRoutes()) {
      if (connFig->typeId() == frcPathSeg) {
        return static_cast<frPathSeg*>(connFig.get());
      }
    }
    return nullptr;
  }
}

frGuide* fr::findSelfSymmetryMirrorGuide(frDesign* design, frGuide* guide) {
  if (design == nullptr || guide == nullptr || guide->getNet() == nullptr) {
    return nullptr;
  }
  auto net = guide->getNet();
  auto constraint = net->getSelfSymmetryConstraintPtr();
  if (constraint == nullptr) {
    return nullptr;
  }

  frPoint begin;
  frPoint end;
  guide->getPoints(begin, end);
  auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(design, *constraint, begin);
  if (!axisCtx.valid) {
    return nullptr;
  }

  auto mirrorBegin = axisCtx.mirrorPoint(begin);
  auto mirrorEnd = axisCtx.mirrorPoint(end);
  normalizeGuidePoints(begin, end);
  normalizeGuidePoints(mirrorBegin, mirrorEnd);
  if (begin == mirrorBegin && end == mirrorEnd) {
    return guide;
  }

  // ponytail: linear scan; add a guide index only if this becomes hot.
  for (auto &uGuide: net->getGuides()) {
    auto candidate = uGuide.get();
    frPoint candidateBegin;
    frPoint candidateEnd;
    candidate->getPoints(candidateBegin, candidateEnd);
    normalizeGuidePoints(candidateBegin, candidateEnd);
    if (candidate->getBeginLayerNum() == guide->getBeginLayerNum() &&
        candidate->getEndLayerNum() == guide->getEndLayerNum() &&
        candidateBegin == mirrorBegin && candidateEnd == mirrorEnd) {
      return candidate;
    }
  }
  return nullptr;
}

void FlexTA::alignSelfSymmetryMirrorGuides() {
  auto block = getDesign() == nullptr ? nullptr : getDesign()->getTopBlock();
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
    if (refNode == nullptr) {
      continue;
    }
    frPoint refLoc;
    refNode->getLoc(refLoc);
    auto axisCtx = SelfSymmetryAxisContext::fromReferencePoint(
        getDesign(), *constraint, refLoc);
    if (!axisCtx.valid || axisCtx.axisSnapFailed) {
      continue;
    }
    axisCtx.rootSide =
        normalizeSelfSymmetryRootSide(axisCtx.sideOfPoint(refLoc));

    for (auto &uGuide: net->getGuides()) {
      auto mirrorGuide = uGuide.get();
      if (mirrorGuide == nullptr) {
        continue;
      }

      frPoint guideBegin;
      frPoint guideEnd;
      mirrorGuide->getPoints(guideBegin, guideEnd);
      if (axisCtx.sideOfPoint(guideBegin) != -axisCtx.rootSide ||
          axisCtx.sideOfPoint(guideEnd) != -axisCtx.rootSide) {
        continue;
      }

      auto rootGuide = findSelfSymmetryMirrorGuide(getDesign(), mirrorGuide);
      if (rootGuide == nullptr || rootGuide == mirrorGuide) {
        continue;
      }

      auto rootPathSeg = getGuidePathSeg(rootGuide);
      auto mirrorPathSeg = getGuidePathSeg(mirrorGuide);
      if (rootPathSeg == nullptr || mirrorPathSeg == nullptr) {
        continue;
      }

      frPoint rootBegin;
      frPoint rootEnd;
      rootPathSeg->getPoints(rootBegin, rootEnd);
      mirrorPathSeg->setPoints(axisCtx.mirrorPoint(rootBegin),
                               axisCtx.mirrorPoint(rootEnd));
    }
  }
}
