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

#include "dr/FlexDR.h"
#include "db/obj/frInstTerm.h"
#include "db/obj/frTerm.h"
#include "gr/FlexGR_self_sym_utils.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <tuple>

using namespace std;
using namespace fr;

namespace {
  struct SelfSymmetryDRSegmentRecord {
    frLayerNum layerNum = 0;
    frPoint begin;
    frPoint end;

    bool operator<(const SelfSymmetryDRSegmentRecord &rhs) const {
      return make_tuple(layerNum, begin.x(), begin.y(), end.x(), end.y()) <
             make_tuple(rhs.layerNum, rhs.begin.x(), rhs.begin.y(), rhs.end.x(), rhs.end.y());
    }
  };

  struct SelfSymmetryDRViaRecord {
    string viaDefName;
    frPoint origin;

    bool operator<(const SelfSymmetryDRViaRecord &rhs) const {
      return make_tuple(viaDefName, origin.x(), origin.y()) <
             make_tuple(rhs.viaDefName, rhs.origin.x(), rhs.origin.y());
    }
  };

  vector<frNet*> collectSelfSymmetryDRNets(frDesign *design) {
    vector<frNet*> nets;
    if (design == nullptr || design->getTopBlock() == nullptr) {
      return nets;
    }
    for (auto &net: design->getTopBlock()->getNets()) {
      if (net && net->getSelfSymmetryConstraintPtr() != nullptr) {
        nets.push_back(net.get());
      }
    }
    return nets;
  }

  bool isSelfSymmetryDRGuideDummy(frPathSeg *pathSeg) {
    frPoint begin, end;
    pathSeg->getPoints(begin, end);
    auto delta = end.x() - begin.x() + end.y() - begin.y();
    return delta == 0 || delta == 1;
  }

  bool selfSymmetryDRPointLess(const frPoint &lhs, const frPoint &rhs) {
    if (lhs.x() != rhs.x()) {
      return lhs.x() < rhs.x();
    }
    return lhs.y() < rhs.y();
  }

  void normalizeSelfSymmetryDRSegment(frPoint &begin, frPoint &end) {
    if (selfSymmetryDRPointLess(end, begin)) {
      swap(begin, end);
    }
  }

  frPoint getSelfSymmetryDRMirrorPoint(const frSelfSymmetryConstraint &constraint,
                                       const frPoint &point) {
    frPoint mirrorPoint(point);
    if (constraint.isAxisHorizontal) {
      mirrorPoint.set(point.x(), constraint.axis + (constraint.axis - point.y()));
    } else {
      mirrorPoint.set(constraint.axis + (constraint.axis - point.x()), point.y());
    }
    return mirrorPoint;
  }

  bool isSelfSymmetryDRPointOnAxis(const frSelfSymmetryConstraint &constraint,
                                   const frPoint &point) {
    return constraint.isAxisHorizontal ? point.y() == constraint.axis
                                       : point.x() == constraint.axis;
  }

  bool isSelfSymmetryDRSegmentOnAxis(const frSelfSymmetryConstraint &constraint,
                                     const SelfSymmetryDRSegmentRecord &record) {
    return isSelfSymmetryDRPointOnAxis(constraint, record.begin) &&
           isSelfSymmetryDRPointOnAxis(constraint, record.end);
  }

  bool selfSymmetryDRSegmentTouchesAxis(
      const frSelfSymmetryConstraint &constraint,
      const SelfSymmetryDRSegmentRecord &record) {
    if (constraint.isAxisHorizontal) {
      if (record.begin.y() == constraint.axis || record.end.y() == constraint.axis) {
        return true;
      }
      return record.begin.x() == record.end.x() &&
             min(record.begin.y(), record.end.y()) <= constraint.axis &&
             max(record.begin.y(), record.end.y()) >= constraint.axis;
    }
    if (record.begin.x() == constraint.axis || record.end.x() == constraint.axis) {
      return true;
    }
    return record.begin.y() == record.end.y() &&
           min(record.begin.x(), record.end.x()) <= constraint.axis &&
           max(record.begin.x(), record.end.x()) >= constraint.axis;
  }

  SelfSymmetryDRSegmentRecord getSelfSymmetryDRMirrorSegment(
      const frSelfSymmetryConstraint &constraint,
      const SelfSymmetryDRSegmentRecord &record) {
    SelfSymmetryDRSegmentRecord mirrorRecord;
    mirrorRecord.layerNum = record.layerNum;
    mirrorRecord.begin = getSelfSymmetryDRMirrorPoint(constraint, record.begin);
    mirrorRecord.end = getSelfSymmetryDRMirrorPoint(constraint, record.end);
    normalizeSelfSymmetryDRSegment(mirrorRecord.begin, mirrorRecord.end);
    return mirrorRecord;
  }

  SelfSymmetryDRViaRecord getSelfSymmetryDRMirrorVia(
      const frSelfSymmetryConstraint &constraint,
      const SelfSymmetryDRViaRecord &record) {
    SelfSymmetryDRViaRecord mirrorRecord;
    mirrorRecord.viaDefName = record.viaDefName;
    mirrorRecord.origin = getSelfSymmetryDRMirrorPoint(constraint, record.origin);
    return mirrorRecord;
  }

  string selfSymmetryDRPointString(const frPoint &point) {
    stringstream ss;
    ss << point.x() << "," << point.y();
    return ss.str();
  }

  string selfSymmetryDRBoxString(const frBox &box) {
    stringstream ss;
    ss << box.left() << "," << box.bottom() << "," << box.right() << "," << box.top();
    return ss.str();
  }

  string selfSymmetryDRSegmentString(const SelfSymmetryDRSegmentRecord &record) {
    stringstream ss;
    ss << "S " << record.layerNum << " "
       << selfSymmetryDRPointString(record.begin) << " "
       << selfSymmetryDRPointString(record.end);
    return ss.str();
  }

  string selfSymmetryDRViaString(const SelfSymmetryDRViaRecord &record) {
    stringstream ss;
    ss << "V " << record.viaDefName << " "
       << selfSymmetryDRPointString(record.origin);
    return ss.str();
  }

  string selfSymmetryDRPatchWireString(frShape *shape) {
    stringstream ss;
    frBox box;
    shape->getBBox(box);
    ss << "P " << shape->getLayerNum() << " " << selfSymmetryDRBoxString(box);
    if (shape->typeId() == frcPatchWire) {
      auto patchWire = static_cast<frPatchWire*>(shape);
      frBox offsetBox;
      frPoint origin;
      patchWire->getOffsetBox(offsetBox);
      patchWire->getOrigin(origin);
      ss << " offset=" << selfSymmetryDRBoxString(offsetBox)
         << " origin=" << selfSymmetryDRPointString(origin);
    } else {
      ss << " type=" << static_cast<int>(shape->typeId());
    }
    return ss.str();
  }

  vector<string> collectSelfSymmetryDRRouteFingerprint(frNet *net) {
    vector<string> records;
    for (auto &shape: net->getShapes()) {
      if (shape->typeId() == frcPathSeg) {
        auto pathSeg = static_cast<frPathSeg*>(shape.get());
        SelfSymmetryDRSegmentRecord record;
        record.layerNum = pathSeg->getLayerNum();
        pathSeg->getPoints(record.begin, record.end);
        normalizeSelfSymmetryDRSegment(record.begin, record.end);
        records.push_back(selfSymmetryDRSegmentString(record));
      } else {
        frBox box;
        shape->getBBox(box);
        stringstream ss;
        ss << "S? " << static_cast<int>(shape->typeId()) << " "
           << shape->getLayerNum() << " " << selfSymmetryDRBoxString(box);
        records.push_back(ss.str());
      }
    }
    for (auto &via: net->getVias()) {
      SelfSymmetryDRViaRecord record;
      record.viaDefName = via->getViaDef() ? via->getViaDef()->getName() : string("<null>");
      via->getOrigin(record.origin);
      records.push_back(selfSymmetryDRViaString(record));
    }
    for (auto &shape: net->getPatchWires()) {
      records.push_back(selfSymmetryDRPatchWireString(shape.get()));
    }
    sort(records.begin(), records.end());
    return records;
  }

  frNet* getSelfSymmetryDRMarkerNet(frBlockObject *src) {
    if (src == nullptr) {
      return nullptr;
    }
    switch (src->typeId()) {
      case frcNet:
        return static_cast<frNet*>(src);
      case frcInstTerm:
        return static_cast<frInstTerm*>(src)->getNet();
      case frcTerm:
        return static_cast<frTerm*>(src)->getNet();
      default:
        return nullptr;
    }
  }

  map<frNet*, int, frBlockObjectComp> countSelfSymmetryDRMarkers(frDesign *design) {
    map<frNet*, int, frBlockObjectComp> markerCounts;
    if (design == nullptr || design->getTopBlock() == nullptr) {
      return markerCounts;
    }
    for (auto &marker: design->getTopBlock()->getMarkers()) {
      set<frNet*, frBlockObjectComp> markerNets;
      for (auto src: marker->getSrcs()) {
        auto net = getSelfSymmetryDRMarkerNet(src);
        if (net != nullptr && net->getSelfSymmetryConstraintPtr() != nullptr) {
          markerNets.insert(net);
        }
      }
      for (auto net: markerNets) {
        ++markerCounts[net];
      }
    }
    return markerCounts;
  }

  void collectSelfSymmetryDRCheckerRecords(
      frNet *net,
      multiset<SelfSymmetryDRSegmentRecord> &segments,
      multiset<SelfSymmetryDRViaRecord> &vias) {
    for (auto &shape: net->getShapes()) {
      if (shape->typeId() != frcPathSeg) {
        continue;
      }
      auto pathSeg = static_cast<frPathSeg*>(shape.get());
      SelfSymmetryDRSegmentRecord record;
      record.layerNum = pathSeg->getLayerNum();
      pathSeg->getPoints(record.begin, record.end);
      normalizeSelfSymmetryDRSegment(record.begin, record.end);
      segments.insert(record);
    }
    for (auto &via: net->getVias()) {
      SelfSymmetryDRViaRecord record;
      record.viaDefName = via->getViaDef() ? via->getViaDef()->getName() : string("<null>");
      via->getOrigin(record.origin);
      vias.insert(record);
    }
  }

  int countSelfSymmetryDRSegmentsMissingMirror(
      const frSelfSymmetryConstraint &constraint,
      const multiset<SelfSymmetryDRSegmentRecord> &segments) {
    int missing = 0;
    for (auto &segment: segments) {
      if (isSelfSymmetryDRSegmentOnAxis(constraint, segment)) {
        continue;
      }
      auto mirrorSegment = getSelfSymmetryDRMirrorSegment(constraint, segment);
      if (segments.find(mirrorSegment) == segments.end()) {
        ++missing;
      }
    }
    return missing;
  }

  int countSelfSymmetryDRViasMissingMirror(
      const frSelfSymmetryConstraint &constraint,
      const multiset<SelfSymmetryDRViaRecord> &vias) {
    int missing = 0;
    for (auto &via: vias) {
      if (isSelfSymmetryDRPointOnAxis(constraint, via.origin)) {
        continue;
      }
      auto mirrorVia = getSelfSymmetryDRMirrorVia(constraint, via);
      if (vias.find(mirrorVia) == vias.end()) {
        ++missing;
      }
    }
    return missing;
  }

  int countSelfSymmetryDRAxisDuplicateShapes(
      const frSelfSymmetryConstraint &constraint,
      const multiset<SelfSymmetryDRViaRecord> &vias) {
    map<SelfSymmetryDRViaRecord, int> axisViaCounts;
    for (auto &via: vias) {
      if (isSelfSymmetryDRPointOnAxis(constraint, via.origin)) {
        ++axisViaCounts[via];
      }
    }
    int duplicates = 0;
    for (auto &[via, count]: axisViaCounts) {
      if (count > 1) {
        duplicates += count - 1;
      }
    }
    return duplicates;
  }

  bool hasSelfSymmetryDRAxisContact(
      const frSelfSymmetryConstraint &constraint,
      const multiset<SelfSymmetryDRSegmentRecord> &segments,
      const multiset<SelfSymmetryDRViaRecord> &vias) {
    for (auto &segment: segments) {
      if (selfSymmetryDRSegmentTouchesAxis(constraint, segment)) {
        return true;
      }
    }
    for (auto &via: vias) {
      if (isSelfSymmetryDRPointOnAxis(constraint, via.origin)) {
        return true;
      }
    }
    return false;
  }

  long long selfSymmetryDRAbsDiff(frCoord lhs, frCoord rhs) {
    return lhs >= rhs ? (long long)(lhs - rhs) : (long long)(rhs - lhs);
  }

  int selfSymmetryDRSideOfCoord(frCoord coord, frCoord axis) {
    if (coord < axis) {
      return -1;
    }
    if (coord > axis) {
      return 1;
    }
    return 0;
  }

  int normalizeSelfSymmetryDRRootSide(int side) {
    return side == 0 ? -1 : side;
  }

  bool findNearestSelfSymmetryDRTrack(frDesign *design,
                                      bool isAxisHorizontal,
                                      frCoord axis,
                                      const frBox *preferredBox,
                                      frCoord &trackCoord,
                                      frLayerNum *layerNum = nullptr,
                                      frTrackPattern **trackPattern = nullptr) {
    return findNearestSelfSymmetryRoutingTrack(design,
                                               isAxisHorizontal,
                                               axis,
                                               preferredBox,
                                               trackCoord,
                                               layerNum,
                                               trackPattern);
  }

  frSelfSymmetryConstraint makeEffectiveSelfSymmetryDRConstraint(
      const frSelfSymmetryConstraint &constraint,
      frCoord effectiveAxis) {
    auto effectiveConstraint = constraint;
    effectiveConstraint.axis = effectiveAxis;
    return effectiveConstraint;
  }

  tuple<frMIdx, frMIdx, frMIdx, int> selfSymmetryDREdgeKey(
      frMIdx x,
      frMIdx y,
      frMIdx z,
      frDirEnum dir) {
    switch (dir) {
      case frDirEnum::W:
        --x;
        dir = frDirEnum::E;
        break;
      case frDirEnum::S:
        --y;
        dir = frDirEnum::N;
        break;
      case frDirEnum::D:
        --z;
        dir = frDirEnum::U;
        break;
      default:
        ;
    }
    return make_tuple(x, y, z, (int)dir);
  }

  void selfSymmetryDRGetNextGrid(frMIdx &x, frMIdx &y, frMIdx &z,
                                 frDirEnum dir) {
    switch (dir) {
      case frDirEnum::E:
        ++x;
        break;
      case frDirEnum::S:
        --y;
        break;
      case frDirEnum::W:
        --x;
        break;
      case frDirEnum::N:
        ++y;
        break;
      case frDirEnum::U:
        ++z;
        break;
      case frDirEnum::D:
        --z;
        break;
      default:
        ;
    }
  }

  frCost selfSymmetryDRCeilCost(frCoord edgeLen,
                                unsigned numerator,
                                unsigned denominator) {
    if (denominator == 0 || edgeLen <= 0) {
      return 0;
    }
    auto scaled =
        (static_cast<unsigned long long>(edgeLen) * numerator + denominator - 1) /
        denominator;
    return static_cast<frCost>(
        min<unsigned long long>(scaled, numeric_limits<frCost>::max()));
  }

  bool selfSymmetryDRPointOnEffectiveAxis(
      const FlexDRWorker::SelfSymmetryDRAxisContext &ctx,
      const frPoint &point) {
    return ctx.isAxisHorizontal ? point.y() == ctx.effectiveAxis
                                : point.x() == ctx.effectiveAxis;
  }

  int selfSymmetryDRSideOfPoint(
      const FlexDRWorker::SelfSymmetryDRAxisContext &ctx,
      const frPoint &point) {
    return ctx.isAxisHorizontal ?
           selfSymmetryDRSideOfCoord(point.y(), ctx.effectiveAxis) :
           selfSymmetryDRSideOfCoord(point.x(), ctx.effectiveAxis);
  }

  frPoint selfSymmetryDRMirrorPoint(
      const FlexDRWorker::SelfSymmetryDRAxisContext &ctx,
      const frPoint &point) {
    frPoint mirrorPoint(point);
    if (ctx.isAxisHorizontal) {
      mirrorPoint.set(point.x(),
                      ctx.effectiveAxis + (ctx.effectiveAxis - point.y()));
    } else {
      mirrorPoint.set(ctx.effectiveAxis + (ctx.effectiveAxis - point.x()),
                      point.y());
    }
    return mirrorPoint;
  }

  bool skipSelfSymmetryDRMirrorPass(frNet *net) {
    return net == nullptr;
  }

  int getSelfSymmetryDRRootSide(frNet *net,
                                bool isAxisHorizontal,
                                frCoord effectiveAxis) {
    frNode *rootNode = net == nullptr ? nullptr : net->getRootGCellNode();
    if (rootNode == nullptr && net != nullptr) {
      rootNode = net->getRoot();
    }
    if (rootNode == nullptr) {
      return -1;
    }
    frPoint rootLoc;
    rootNode->getLoc(rootLoc);
    auto rootCoord = isAxisHorizontal ? rootLoc.y() : rootLoc.x();
    return normalizeSelfSymmetryDRRootSide(
        selfSymmetryDRSideOfCoord(rootCoord, effectiveAxis));
  }

}

FlexDRWorker::SelfSymmetryDRRouteContext&
FlexDRWorker::initSelfSymmetryDRRoutingContext(frNet* net) {
  auto &ctx = selfSymmetryDRRouteContexts[net];
  ctx = SelfSymmetryDRRouteContext();
  selfSymmetryDRActiveNet = net;
  return ctx;
}

void FlexDRWorker::deactivateSelfSymmetryDRRoutingContext() {
  selfSymmetryDRActiveNet = nullptr;
}

void FlexDRWorker::initSelfSymmetryDRAxisContext(
    frNet* net,
    SelfSymmetryDRRouteContext &routeCtx) {
  routeCtx.axis = SelfSymmetryDRAxisContext();
  if (net == nullptr || net->getSelfSymmetryConstraintPtr() == nullptr) {
    return;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  auto &ctx = routeCtx.axis;
  ctx.valid = true;
  ctx.isAxisHorizontal = constraint.isAxisHorizontal;
  ctx.originalAxis = constraint.axis;
  ctx.effectiveAxis = constraint.axis;

  auto sharedState = getSelfSymmetryDRSharedState(net);
  if (sharedState != nullptr) {
    ctx.effectiveAxis = sharedState->snappedAxis;
    ctx.rootSide = sharedState->rootSide;
  } else {
    frCoord snappedAxis = constraint.axis;
    bool foundPreferredTrack = false;
    bool axisInExtBox = constraint.isAxisHorizontal ?
                        constraint.axis >= extBox.bottom() &&
                        constraint.axis <= extBox.top() :
                        constraint.axis >= extBox.left() &&
                        constraint.axis <= extBox.right();
    if (axisInExtBox) {
      foundPreferredTrack = findNearestSelfSymmetryDRTrack(design,
                                                           constraint.isAxisHorizontal,
                                                           constraint.axis,
                                                           &extBox,
                                                           snappedAxis);
    }
    bool foundTrack = foundPreferredTrack ||
                      findNearestSelfSymmetryDRTrack(design,
                                                     constraint.isAxisHorizontal,
                                                     constraint.axis,
                                                     nullptr,
                                                     snappedAxis);
    if (foundTrack) {
      ctx.effectiveAxis = snappedAxis;
    } else {
      ctx.axisSnapFailed = true;
    }
  }
  ctx.axisSnapDelta = ctx.effectiveAxis - ctx.originalAxis;
  ctx.axisInRouteBox = constraint.isAxisHorizontal ?
                       ctx.effectiveAxis >= routeBox.bottom() &&
                       ctx.effectiveAxis <= routeBox.top() :
                       ctx.effectiveAxis >= routeBox.left() &&
                       ctx.effectiveAxis <= routeBox.right();

  if (constraint.isAxisHorizontal) {
    if (gridGraph.hasMazeYIdx(ctx.effectiveAxis)) {
      ctx.axisMazeIdx = gridGraph.getMazeYIdx(ctx.effectiveAxis);
    }
  } else {
    if (gridGraph.hasMazeXIdx(ctx.effectiveAxis)) {
      ctx.axisMazeIdx = gridGraph.getMazeXIdx(ctx.effectiveAxis);
    }
  }

  if (sharedState == nullptr) {
    frNode *rootNode = net->getRootGCellNode();
    if (rootNode == nullptr) {
      rootNode = net->getRoot();
    }
    if (rootNode != nullptr) {
      frPoint rootLoc;
      rootNode->getLoc(rootLoc);
      ctx.rootSide =
          normalizeSelfSymmetryDRRootSide(selfSymmetryDRSideOfPoint(ctx, rootLoc));
    } else {
      ctx.rootSide = -1;
    }
  }
}

bool FlexDRWorker::hasSelfSymmetryDRAxisInRouteBox(frNet *net) {
  if (net == nullptr || net->getSelfSymmetryConstraintPtr() == nullptr) {
    return false;
  }
  SelfSymmetryDRRouteContext routeCtx;
  initSelfSymmetryDRAxisContext(net, routeCtx);
  return routeCtx.axis.valid && routeCtx.axis.axisInRouteBox;
}

void FlexDRWorker::initTrackCoords_selfSymmetryAxis(
    frNet* net,
    map<frCoord, map<frLayerNum, frTrackPattern*> > &xMap,
    map<frCoord, map<frLayerNum, frTrackPattern*> > &yMap) {
  if (net == nullptr || net->getSelfSymmetryConstraintPtr() == nullptr) {
    return;
  }
  auto constraint = net->getSelfSymmetryConstraint();
  frCoord snappedAxis = constraint.axis;
  frLayerNum layerNum = 0;
  frTrackPattern *trackPattern = nullptr;
  auto sharedState = getSelfSymmetryDRSharedState(net);
  if (sharedState != nullptr) {
    snappedAxis = sharedState->snappedAxis;
    bool axisInExtBox = constraint.isAxisHorizontal ?
                        snappedAxis >= extBox.bottom() &&
                        snappedAxis <= extBox.top() :
                        snappedAxis >= extBox.left() &&
                        snappedAxis <= extBox.right();
    if (!axisInExtBox) {
      return;
    }
    if (!findNearestSelfSymmetryDRTrack(design, constraint.isAxisHorizontal,
                                        snappedAxis, &extBox, snappedAxis,
                                        &layerNum, &trackPattern)) {
      return;
    }
    if (constraint.isAxisHorizontal) {
      yMap[snappedAxis][layerNum] = trackPattern;
    } else {
      xMap[snappedAxis][layerNum] = trackPattern;
    }
    return;
  }
  if (!findNearestSelfSymmetryDRTrack(design, constraint.isAxisHorizontal,
                                      constraint.axis, &extBox, snappedAxis,
                                      &layerNum, &trackPattern)) {
    return;
  }
  if (constraint.isAxisHorizontal) {
    yMap[snappedAxis][layerNum] = trackPattern;
  } else {
    xMap[snappedAxis][layerNum] = trackPattern;
  }
}

bool FlexDRWorker::hasSelfSymmetryDRAxisContact(
    const SelfSymmetryDRRouteContext &ctx,
    const vector<FlexMazeIdx> &connComps) const {
  auto &axis = ctx.axis;
  if (!axis.valid || axis.axisMazeIdx < 0) {
    return false;
  }
  for (auto &mi: connComps) {
    if (axis.isAxisHorizontal) {
      if (mi.y() == axis.axisMazeIdx) {
        return true;
      }
    } else if (mi.x() == axis.axisMazeIdx) {
      return true;
    }
  }
  return false;
}

void FlexDRWorker::collectSelfSymmetryDRAxisCandidates(
    const SelfSymmetryDRRouteContext &ctx,
    vector<FlexMazeIdx> &candidates) const {
  candidates.clear();
  auto &axis = ctx.axis;
  if (!axis.valid || axis.axisMazeIdx < 0) {
    return;
  }
  frMIdx xDim, yDim, zDim;
  gridGraph.getDim(xDim, yDim, zDim);
  if (axis.isAxisHorizontal) {
    auto y = axis.axisMazeIdx;
    if (y < 0 || y >= yDim) {
      return;
    }
    for (frMIdx x = 0; x < xDim; ++x) {
      for (frMIdx z = 0; z < zDim; ++z) {
        candidates.push_back(FlexMazeIdx(x, y, z));
      }
    }
  } else {
    auto x = axis.axisMazeIdx;
    if (x < 0 || x >= xDim) {
      return;
    }
    for (frMIdx y = 0; y < yDim; ++y) {
      for (frMIdx z = 0; z < zDim; ++z) {
        candidates.push_back(FlexMazeIdx(x, y, z));
      }
    }
  }
}

int FlexDRWorker::collectSelfSymmetryDRAxisSources(
    const vector<FlexMazeIdx> &connComps,
    const SelfSymmetryDRRouteContext &ctx,
    vector<FlexMazeIdx> &axisSources) const {
  axisSources.clear();
  set<FlexMazeIdx> uniqueSources;
  auto &axis = ctx.axis;
  if (!axis.valid || axis.axisMazeIdx < 0) {
    return 0;
  }
  for (auto &mi: connComps) {
    bool onAxis = axis.isAxisHorizontal ?
                  mi.y() == axis.axisMazeIdx :
                  mi.x() == axis.axisMazeIdx;
    if (onAxis && uniqueSources.insert(mi).second) {
      axisSources.push_back(mi);
    }
  }
  return axisSources.size();
}

void FlexDRWorker::buildSelfSymmetryDRMirrorGuides(
    drNet* net,
    SelfSymmetryDRRouteContext &ctx) {
  ctx.mirrorRewardEdges.clear();
  if (net == nullptr || !ctx.axis.valid) {
    return;
  }

  auto addMirrorRewardEdge = [&](const FlexMazeIdx &begin,
                                 const FlexMazeIdx &end) {
    if (begin == end) {
      return;
    }
    auto lNum = gridGraph.getLayerNum(begin.z());
    frPoint beginPoint, endPoint;
    gridGraph.getPoint(beginPoint, begin.x(), begin.y());
    gridGraph.getPoint(endPoint, end.x(), end.y());
    if (selfSymmetryDRPointOnEffectiveAxis(ctx.axis, beginPoint) &&
        selfSymmetryDRPointOnEffectiveAxis(ctx.axis, endPoint)) {
      return;
    }
    auto mirrorBeginPoint = selfSymmetryDRMirrorPoint(ctx.axis,
                                                      beginPoint);
    auto mirrorEndPoint = selfSymmetryDRMirrorPoint(ctx.axis,
                                                    endPoint);
    if (!gridGraph.hasMazeIdx(mirrorBeginPoint, lNum) ||
        !gridGraph.hasMazeIdx(mirrorEndPoint, lNum)) {
      return;
    }
    FlexMazeIdx mirrorBegin, mirrorEnd;
    gridGraph.getMazeIdx(mirrorBegin, mirrorBeginPoint, lNum);
    gridGraph.getMazeIdx(mirrorEnd, mirrorEndPoint, lNum);
    frDirEnum dir = frDirEnum::UNKNOWN;
    if (mirrorBegin.x() != mirrorEnd.x() &&
        mirrorBegin.y() == mirrorEnd.y() &&
        mirrorBegin.z() == mirrorEnd.z()) {
      dir = mirrorBegin.x() < mirrorEnd.x() ? frDirEnum::E : frDirEnum::W;
    } else if (mirrorBegin.y() != mirrorEnd.y() &&
               mirrorBegin.x() == mirrorEnd.x() &&
               mirrorBegin.z() == mirrorEnd.z()) {
      dir = mirrorBegin.y() < mirrorEnd.y() ? frDirEnum::N : frDirEnum::S;
    } else {
      return;
    }
    auto key = selfSymmetryDREdgeKey(mirrorBegin.x(), mirrorBegin.y(),
                                     mirrorBegin.z(), dir);
    ctx.mirrorRewardEdges.insert(key);
  };

  for (auto &uConnFig: net->getRouteConnFigs()) {
    if (uConnFig->typeId() == drcPathSeg) {
      FlexMazeIdx begin, end;
      static_cast<drPathSeg*>(uConnFig.get())->getMazeIdx(begin, end);
      if (begin.z() != end.z()) {
        continue;
      }
      if (begin.x() != end.x()) {
        auto y = begin.y();
        auto z = begin.z();
        for (auto x = min(begin.x(), end.x()); x < max(begin.x(), end.x()); ++x) {
          addMirrorRewardEdge(FlexMazeIdx(x, y, z), FlexMazeIdx(x + 1, y, z));
        }
      } else if (begin.y() != end.y()) {
        auto x = begin.x();
        auto z = begin.z();
        for (auto y = min(begin.y(), end.y()); y < max(begin.y(), end.y()); ++y) {
          addMirrorRewardEdge(FlexMazeIdx(x, y, z), FlexMazeIdx(x, y + 1, z));
        }
      }
    } else if (uConnFig->typeId() == drcVia) {
      auto via = static_cast<drVia*>(uConnFig.get());
      frPoint origin;
      via->getOrigin(origin);
      if (selfSymmetryDRPointOnEffectiveAxis(ctx.axis, origin)) {
        continue;
      }
      auto mirrorOrigin = selfSymmetryDRMirrorPoint(ctx.axis,
                                                    origin);
      auto lNum = via->getViaDef()->getLayer1Num();
      if (!gridGraph.hasMazeIdx(mirrorOrigin, lNum)) {
        continue;
      }
      FlexMazeIdx mirrorIdx;
      gridGraph.getMazeIdx(mirrorIdx, mirrorOrigin, lNum);
      ctx.mirrorRewardEdges.insert(
          selfSymmetryDREdgeKey(mirrorIdx.x(), mirrorIdx.y(),
                                mirrorIdx.z(), frDirEnum::U));
    }
  }
}

frCost FlexDRWorker::getSelfSymmetryDRCost(frMIdx x,
                                           frMIdx y,
                                           frMIdx z,
                                           frDirEnum dir,
                                           bool hasGuide) {
  if (selfSymmetryDRActiveNet == nullptr) {
    return 0;
  }
  auto contextIt = selfSymmetryDRRouteContexts.find(selfSymmetryDRActiveNet);
  if (contextIt == selfSymmetryDRRouteContexts.end()) {
    return 0;
  }
  auto &ctx = contextIt->second;
  auto &axis = ctx.axis;
  if (!axis.valid || ctx.routeMode == SelfSymmetryDRRouteMode::None) {
    return 0;
  }

  auto edgeLen = gridGraph.getEdgeLength(x, y, z, dir);
  if (edgeLen <= 0) {
    edgeLen = 1;
  }

  if (ctx.routeMode == SelfSymmetryDRRouteMode::Mirror) {
    auto key = selfSymmetryDREdgeKey(x, y, z, dir);
    if (ctx.mirrorRewardEdges.find(key) != ctx.mirrorRewardEdges.end()) {
      return 0;
    }
    auto missMultiplier = max(1u, min((unsigned)(4 * GUIDECOST),
                                      workerDRCCost > 0 ? workerDRCCost - 1 : 1));
    return (hasGuide ? GUIDECOST : missMultiplier) * edgeLen;
  }

  if (dir == frDirEnum::U || dir == frDirEnum::D) {
    return 0;
  }

  frMIdx nextX = x;
  frMIdx nextY = y;
  frMIdx nextZ = z;
  selfSymmetryDRGetNextGrid(nextX, nextY, nextZ, dir);
  frPoint beginPoint, endPoint;
  gridGraph.getPoint(beginPoint, x, y);
  gridGraph.getPoint(endPoint, nextX, nextY);
  auto beginDist = axis.isAxisHorizontal ?
                   selfSymmetryDRAbsDiff(beginPoint.y(),
                                         axis.effectiveAxis) :
                   selfSymmetryDRAbsDiff(beginPoint.x(),
                                         axis.effectiveAxis);
  auto endDist = axis.isAxisHorizontal ?
                 selfSymmetryDRAbsDiff(endPoint.y(),
                                       axis.effectiveAxis) :
                 selfSymmetryDRAbsDiff(endPoint.x(),
                                       axis.effectiveAxis);

  frCost cost = 0;
  if (beginDist == 0 && endDist == 0) {
    cost = selfSymmetryDRCeilCost(edgeLen, 1, 10);
  } else if (endDist > beginDist) {
    cost = selfSymmetryDRCeilCost(edgeLen, 2, 1);
  } else if (endDist < beginDist) {
    cost = selfSymmetryDRCeilCost(edgeLen, 3, 4);
  } else {
    cost = static_cast<frCost>(edgeLen);
  }

  return cost;
}

bool FlexDRWorker::routeNet_selfSymmetry(drNet* net) {
  if (net == nullptr || net->getFrNet() == nullptr ||
      net->getFrNet()->getSelfSymmetryConstraintPtr() == nullptr) {
    return true;
  }
  if (net->getPins().size() <= 1) {
    return true;
  }

  auto &ctx = initSelfSymmetryDRRoutingContext(net->getFrNet());
  initSelfSymmetryDRAxisContext(net->getFrNet(), ctx);
  ctx.routeMode = SelfSymmetryDRRouteMode::Lead;
  auto sharedState = getSelfSymmetryDRSharedState(net->getFrNet());

  vector<drPin*> leadPins;
  vector<drPin*> mirrorPins;
  auto rootNode = net->getFrNet()->getRoot();
  auto rootPin = rootNode ? rootNode->getPin() : nullptr;
  for (auto &uPin: net->getPins()) {
    auto pin = uPin.get();
    frPoint point;
    bool hasPoint = false;
    for (auto &ap: pin->getAccessPatterns()) {
      ap->getPoint(point);
      hasPoint = true;
      break;
    }
    if (!hasPoint) {
      continue;
    }
    int side = selfSymmetryDRSideOfPoint(ctx.axis, point);
    bool isRootPin = rootPin != nullptr && pin->getFrTerm() == rootPin;
    if (isRootPin || side == 0 || side == ctx.axis.rootSide) {
      leadPins.push_back(pin);
    } else {
      mirrorPins.push_back(pin);
    }
  }
  drPin axisBoundaryPin;
  if (sharedState != nullptr &&
      sharedState->axisLinkDone &&
      sharedState->axisLinkPointValid) {
    frPoint axisPoint = sharedState->axisLinkPoint;
    if (ctx.axis.isAxisHorizontal) {
      if (axisPoint.x() <= routeBox.left()) {
        axisPoint.set(routeBox.left(), ctx.axis.effectiveAxis);
      } else if (axisPoint.x() >= routeBox.right()) {
        axisPoint.set(routeBox.right(), ctx.axis.effectiveAxis);
      } else {
        axisPoint.set(axisPoint.x(), ctx.axis.effectiveAxis);
      }
    } else {
      if (axisPoint.y() <= routeBox.bottom()) {
        axisPoint.set(ctx.axis.effectiveAxis, routeBox.bottom());
      } else if (axisPoint.y() >= routeBox.top()) {
        axisPoint.set(ctx.axis.effectiveAxis, routeBox.top());
      } else {
        axisPoint.set(ctx.axis.effectiveAxis, axisPoint.y());
      }
    }
    if (routeBox.contains(axisPoint) &&
        gridGraph.hasMazeIdx(axisPoint, sharedState->axisLinkLayerNum)) {
      FlexMazeIdx axisMazeIdx;
      gridGraph.getMazeIdx(axisMazeIdx, axisPoint,
                           sharedState->axisLinkLayerNum);
      auto axisAP = make_unique<drAccessPattern>();
      axisAP->setPoint(axisPoint);
      axisAP->setBeginLayerNum(sharedState->axisLinkLayerNum);
      axisAP->setMazeIdx(axisMazeIdx);
      axisBoundaryPin.addAccessPattern(axisAP);
      leadPins.push_back(&axisBoundaryPin);
    }
  }

  auto routeSelectedPins = [&](const vector<drPin*> &selectedPins,
                               const vector<FlexMazeIdx> *forcedSources,
                               vector<FlexMazeIdx> &connComps,
                               FlexMazeIdx &ccMazeIdx1,
                               FlexMazeIdx &ccMazeIdx2,
                               frPoint &centerPt) {
    if (selectedPins.empty()) {
      return true;
    }

    set<drPin*, frBlockObjectComp> unConnPins;
    map<FlexMazeIdx, set<drPin*, frBlockObjectComp> > mazeIdx2unConnPins;
    set<FlexMazeIdx> apMazeIdx;
    set<FlexMazeIdx> realPinAPMazeIdx;
    for (auto pin: selectedPins) {
      unConnPins.insert(pin);
      for (auto &ap: pin->getAccessPatterns()) {
        FlexMazeIdx mi;
        ap->getMazeIdx(mi);
        mazeIdx2unConnPins[mi].insert(pin);
        apMazeIdx.insert(mi);
        gridGraph.setDst(mi);
        if (pin->hasFrTerm()) {
          realPinAPMazeIdx.insert(mi);
        }
      }
    }

    map<FlexMazeIdx, frCoord> areaMap;
    if (ENABLE_BOUNDARY_MAR_FIX) {
      routeNet_prepAreaMap(net, areaMap);
    }

    if (forcedSources == nullptr) {
      routeNet_setSrc(unConnPins, mazeIdx2unConnPins, connComps,
                      ccMazeIdx1, ccMazeIdx2, centerPt);
    } else {
      connComps = *forcedSources;
      frMIdx xDim, yDim, zDim;
      gridGraph.getDim(xDim, yDim, zDim);
      ccMazeIdx1.set(xDim - 1, yDim - 1, zDim - 1);
      ccMazeIdx2.set(0, 0, 0);
      centerPt.set(0, 0);
      int sourceCnt = 0;
      frPoint sourcePoint;
      for (auto &mi: connComps) {
        gridGraph.setSrc(mi);
        ccMazeIdx1.set(min(ccMazeIdx1.x(), mi.x()),
                       min(ccMazeIdx1.y(), mi.y()),
                       min(ccMazeIdx1.z(), mi.z()));
        ccMazeIdx2.set(max(ccMazeIdx2.x(), mi.x()),
                       max(ccMazeIdx2.y(), mi.y()),
                       max(ccMazeIdx2.z(), mi.z()));
        gridGraph.getPoint(sourcePoint, mi.x(), mi.y());
        centerPt.set(centerPt.x() + sourcePoint.x(),
                     centerPt.y() + sourcePoint.y());
        ++sourceCnt;
      }
      if (sourceCnt > 0) {
        centerPt.set(centerPt.x() / sourceCnt, centerPt.y() / sourceCnt);
      }
    }

    vector<FlexMazeIdx> path;
    bool isFirstConn = forcedSources == nullptr;
    while (!unConnPins.empty()) {
      mazePinInit();
      auto nextPin = routeNet_getNextDst(ccMazeIdx1, ccMazeIdx2,
                                         mazeIdx2unConnPins);
      path.clear();
      if (nextPin != nullptr &&
          gridGraph.search(connComps, nextPin, path, ccMazeIdx1,
                           ccMazeIdx2, centerPt)) {
        routeNet_postAstarUpdate(path, connComps, unConnPins,
                                 mazeIdx2unConnPins, isFirstConn);
        routeNet_postAstarWritePath(net, path, realPinAPMazeIdx);
        routeNet_postAstarPatchMinAreaVio(net, path, areaMap);
        isFirstConn = false;
      } else {
        for (auto &mi: apMazeIdx) {
          gridGraph.resetDst(mi);
        }
        return false;
      }
    }
    for (auto &mi: apMazeIdx) {
      gridGraph.resetDst(mi);
    }
    return true;
  };

  vector<FlexMazeIdx> leadConnComps;
  FlexMazeIdx leadCCMazeIdx1, leadCCMazeIdx2;
  frPoint leadCenterPt;
  if (!leadPins.empty()) {
    if (!routeSelectedPins(leadPins, nullptr, leadConnComps, leadCCMazeIdx1,
                           leadCCMazeIdx2, leadCenterPt)) {
      deactivateSelfSymmetryDRRoutingContext();
      return false;
    }
  }

  ctx.leadAxisContactBeforeLink =
      hasSelfSymmetryDRAxisContact(ctx, leadConnComps);
  if (sharedState != nullptr && ctx.leadAxisContactBeforeLink) {
    sharedState->axisContactSeen = true;
  }

  bool axisLinkAttempted = false;
  bool axisLinkSucceeded = false;
  if (!leadPins.empty() && !ctx.leadAxisContactBeforeLink) {
    vector<FlexMazeIdx> axisCandidates;
    collectSelfSymmetryDRAxisCandidates(ctx, axisCandidates);
    vector<FlexMazeIdx> boundaryAxisCandidates;
    for (auto &mi: axisCandidates) {
      frPoint point;
      gridGraph.getPoint(point, mi.x(), mi.y());
      if (point.x() == routeBox.left() ||
          point.x() == routeBox.right() ||
          point.y() == routeBox.bottom() ||
          point.y() == routeBox.top()) {
        boundaryAxisCandidates.push_back(mi);
      }
    }
    if (!boundaryAxisCandidates.empty()) {
      axisCandidates = std::move(boundaryAxisCandidates);
    }
    bool maySearchAxisLink = sharedState == nullptr ||
                             (!sharedState->axisContactSeen &&
                              !sharedState->axisLinkDone);
    if (maySearchAxisLink && !axisCandidates.empty()) {
      axisLinkAttempted = true;
      ctx.routeMode = SelfSymmetryDRRouteMode::AxisLink;
      drPin axisPin;
      set<FlexMazeIdx> axisDst;
      for (auto &mi: axisCandidates) {
        auto uAP = make_unique<drAccessPattern>();
        frPoint axisPoint;
        gridGraph.getPoint(axisPoint, mi.x(), mi.y());
        uAP->setPoint(axisPoint);
        uAP->setBeginLayerNum(gridGraph.getLayerNum(mi.z()));
        uAP->setMazeIdx(mi);
        axisPin.addAccessPattern(uAP);
        gridGraph.setDst(mi);
        axisDst.insert(mi);
      }
      vector<FlexMazeIdx> path;
      mazePinInit();
      if (gridGraph.search(leadConnComps, &axisPin, path, leadCCMazeIdx1,
                           leadCCMazeIdx2, leadCenterPt)) {
        if (sharedState != nullptr) {
          for (auto it = path.rbegin(); it != path.rend(); ++it) {
            frPoint point;
            gridGraph.getPoint(point, it->x(), it->y());
            if ((point.x() == routeBox.left() ||
                 point.x() == routeBox.right() ||
                 point.y() == routeBox.bottom() ||
                 point.y() == routeBox.top()) &&
                selfSymmetryDRPointOnEffectiveAxis(ctx.axis, point)) {
              sharedState->axisLinkPoint = point;
              sharedState->axisLinkLayerNum = gridGraph.getLayerNum(it->z());
              sharedState->axisLinkPointValid = true;
              break;
            }
          }
        }
        set<drPin*, frBlockObjectComp> emptyPins;
        map<FlexMazeIdx, set<drPin*, frBlockObjectComp> > emptyPinMap;
        routeNet_postAstarUpdate(path, leadConnComps, emptyPins,
                                 emptyPinMap, false);
        set<FlexMazeIdx> emptyAPs;
        routeNet_postAstarWritePath(net, path, emptyAPs);
        map<FlexMazeIdx, frCoord> emptyAreaMap;
        routeNet_postAstarPatchMinAreaVio(net, path, emptyAreaMap);
        axisLinkSucceeded = true;
      } else {
        cout << "Error: self-symmetry DR failed to link lead tree to axis for "
             << net->getFrNet()->getName() << "\n";
      }
      for (auto &mi: axisDst) {
        gridGraph.resetDst(mi);
      }
    } else {
      cout << "Error: self-symmetry DR has no axis candidates for "
           << net->getFrNet()->getName() << "\n";
    }
    if (axisLinkAttempted && !axisLinkSucceeded) {
      if (sharedState != nullptr) {
        sharedState->failed = true;
      }
      deactivateSelfSymmetryDRRoutingContext();
      return false;
    }
    if (axisLinkSucceeded && sharedState != nullptr &&
        !sharedState->axisLinkPointValid) {
      cout << "Error: self-symmetry DR axis-link missed tile boundary for "
           << net->getFrNet()->getName() << "\n";
      sharedState->failed = true;
      deactivateSelfSymmetryDRRoutingContext();
      return false;
    }
  }
  ctx.leadAxisContactAfterLink =
      hasSelfSymmetryDRAxisContact(ctx, leadConnComps);
  if (sharedState != nullptr && ctx.leadAxisContactAfterLink) {
    sharedState->axisContactSeen = true;
  }
  if (sharedState != nullptr && axisLinkSucceeded) {
    sharedState->axisLinkDone = true;
  }
  if (axisLinkAttempted && !ctx.leadAxisContactAfterLink) {
    cout << "Error: self-symmetry DR lead tree still misses axis after link for "
         << net->getFrNet()->getName() << "\n";
    if (sharedState != nullptr) {
      sharedState->failed = true;
    }
    deactivateSelfSymmetryDRRoutingContext();
    return false;
  }
  if (sharedState == nullptr && !leadPins.empty() &&
      !ctx.leadAxisContactAfterLink) {
    cout << "Error: self-symmetry DR lead tree still misses axis after link for "
         << net->getFrNet()->getName() << "\n";
    deactivateSelfSymmetryDRRoutingContext();
    return false;
  }

  if (skipSelfSymmetryDRMirrorPass(net->getFrNet())) {
  } else {
    buildSelfSymmetryDRMirrorGuides(net, ctx);

    vector<FlexMazeIdx> axisSources;
    collectSelfSymmetryDRAxisSources(leadConnComps, ctx, axisSources);
    if (!mirrorPins.empty()) {
      auto forcedMirrorSources = axisSources.empty() ? nullptr : &axisSources;
      ctx.routeMode = SelfSymmetryDRRouteMode::Mirror;
      vector<FlexMazeIdx> mirrorConnComps;
      FlexMazeIdx mirrorCCMazeIdx1, mirrorCCMazeIdx2;
      frPoint mirrorCenterPt;
      if (!routeSelectedPins(mirrorPins, forcedMirrorSources, mirrorConnComps,
                             mirrorCCMazeIdx1, mirrorCCMazeIdx2,
                             mirrorCenterPt)) {
        deactivateSelfSymmetryDRRoutingContext();
        return false;
      }
    }
  }

  routeNet_postRouteAddPathCost(net);
  deactivateSelfSymmetryDRRoutingContext();
  return true;
}

bool FlexDR::initSelfSymmetryDRSharedState(
    frNet *net,
    SelfSymmetryDRSharedState &state) const {
  if (design == nullptr || design->getTopBlock() == nullptr ||
      net == nullptr || net->getSelfSymmetryConstraintPtr() == nullptr) {
    return false;
  }

  auto constraint = net->getSelfSymmetryConstraint();
  frCoord snappedAxis = constraint.axis;
  if (!findNearestSelfSymmetryDRTrack(design,
                                      constraint.isAxisHorizontal,
                                      constraint.axis,
                                      nullptr,
                                      snappedAxis)) {
    cout << "Error: self-symmetry DR failed to snap axis for "
         << net->getName() << "\n";
    return false;
  }

  frBox dieBox;
  design->getTopBlock()->getBoundaryBBox(dieBox);
  bool axisInDie = constraint.isAxisHorizontal ?
                   snappedAxis >= dieBox.bottom() &&
                   snappedAxis <= dieBox.top() :
                   snappedAxis >= dieBox.left() &&
                   snappedAxis <= dieBox.right();
  if (!axisInDie) {
    cout << "Error: self-symmetry DR snapped axis is outside die for "
         << net->getName() << "\n";
    return false;
  }

  state = SelfSymmetryDRSharedState();
  state.net = net;
  state.isAxisHorizontal = constraint.isAxisHorizontal;
  state.snappedAxis = snappedAxis;
  state.rootSide = getSelfSymmetryDRRootSide(net,
                                             constraint.isAxisHorizontal,
                                             snappedAxis);
  return true;
}

void FlexDR::collectSelfSymmetryDRTargetNets(
    set<frNet*, frBlockObjectComp> &targetNets) const {
  targetNets.clear();
  if (getDesign() == nullptr || getDesign()->getTopBlock() == nullptr) {
    return;
  }
  auto block = getDesign()->getTopBlock();
  for (auto &net: getDesign()->getTopBlock()->getNets()) {
    if (net && !block->isRoutedNet(net->getName()) &&
        net->getSelfSymmetryConstraintPtr() != nullptr) {
      targetNets.insert(net.get());
    }
  }
}

void FlexDR::collectOrdinaryDRTargetNets(
    set<frNet*, frBlockObjectComp> &targetNets) const {
  targetNets.clear();
  if (getDesign() == nullptr || getDesign()->getTopBlock() == nullptr) {
    return;
  }
  auto block = getDesign()->getTopBlock();
  for (auto &net: getDesign()->getTopBlock()->getNets()) {
    if (net && !block->isRoutedNet(net->getName()) &&
        net->getSelfSymmetryConstraintPtr() == nullptr) {
      targetNets.insert(net.get());
    }
  }
}

bool FlexDR::runSelfSymmetryDRPhase() {
  set<frNet*, frBlockObjectComp> selfSymmetryNets;
  collectSelfSymmetryDRTargetNets(selfSymmetryNets);
  if (selfSymmetryNets.empty()) {
    return false;
  }

  cout << endl << "@@@ self-symmetry dr phase @@@" << endl;
  cout << "self_symmetry_nets: " << selfSymmetryNets.size() << "\n";
  int ordinaryNetsInPhase = 0;
  SelfSymmetryDRSharedStateMap selfSymmetryDRSharedStates;
  for (auto net: selfSymmetryNets) {
    SelfSymmetryDRSharedState state;
    if (!initSelfSymmetryDRSharedState(net, state)) {
      cout << "Error: self-symmetry DR shared state init failed for "
           << net->getName() << "\n";
      exit(1);
    }
    selfSymmetryDRSharedStates[net] = state;
    cout << "snapped_axis: " << net->getName() << " "
         << state.snappedAxis << "\n";
    cout << "root_side: " << net->getName() << " "
         << state.rootSide << "\n";
  }
  FlexDRSearchRepairPhase selfSymmetryPhase;
  selfSymmetryPhase.targetNets = &selfSymmetryNets;
  selfSymmetryPhase.stageName = "self-symmetry dr phase";
  selfSymmetryPhase.ordinaryNetsInPhase = &ordinaryNetsInPhase;
  selfSymmetryPhase.removeBoundaryPinsOnInit = false;
  selfSymmetryPhase.selfSymmetryDRSharedStates = &selfSymmetryDRSharedStates;
  searchRepair(0, 7, 0, 3, DRCCOST, 0, 0, 0, true, 2, true, 9, false,
               &selfSymmetryPhase);
  for (auto &[net, state]: selfSymmetryDRSharedStates) {
    if (state.failed || (!state.axisContactSeen && !state.axisLinkDone)) {
      cout << "Error: self-symmetry DR failed for "
           << net->getName() << "\n";
      exit(1);
    }
  }
  cout << "ordinary_nets_in_phase: " << ordinaryNetsInPhase << "\n";
  reportSelfSymmetryDRPhaseRouteCount(selfSymmetryNets);
  reportSelfSymmetryDRChecker();
  snapshotSelfSymmetryDRRoutes();
  keepOnlySelfSymmetryDRTargetRoutes(selfSymmetryNets);
  return true;
}

void FlexDR::reportSelfSymmetryDRGuideRoutes() const {
  auto nets = collectSelfSymmetryDRNets(design);
  if (nets.empty()) {
    return;
  }

  cout << "@@@ self-symmetry dr guide routes @@@" << endl;
  for (auto net: nets) {
    int total = 0;
    int pathSegs = 0;
    int dummy = 0;
    int usable = 0;
    for (auto &guide: net->getGuides()) {
      for (auto &connFig: guide->getRoutes()) {
        ++total;
        if (connFig->typeId() != frcPathSeg) {
          continue;
        }
        ++pathSegs;
        auto pathSeg = static_cast<frPathSeg*>(connFig.get());
        if (isSelfSymmetryDRGuideDummy(pathSeg)) {
          ++dummy;
        } else {
          ++usable;
        }
      }
    }
    cout << "net: " << net->getName() << "\n";
    cout << "guide_routes_total/pathseg/dummy/usable: "
         << total << "/" << pathSegs << "/" << dummy << "/" << usable << "\n";
  }
}

void FlexDR::reportSelfSymmetryDRBoundaryPins() const {
  auto nets = collectSelfSymmetryDRNets(design);
  if (nets.empty()) {
    return;
  }

  map<frNet*, int, frBlockObjectComp> boundaryPinCounts;
  for (auto &xVec: gcell2BoundaryPin) {
    for (auto &gcellPins: xVec) {
      for (auto &[net, pins]: gcellPins) {
        if (net != nullptr && net->getSelfSymmetryConstraintPtr() != nullptr) {
          boundaryPinCounts[net] += pins.size();
        }
      }
    }
  }

  cout << "@@@ self-symmetry dr boundary pins @@@" << endl;
  for (auto net: nets) {
    cout << "net: " << net->getName() << "\n";
    cout << "boundary_pins: " << boundaryPinCounts[net] << "\n";
    cout << "source: guide_routes\n";
  }
}

void FlexDR::reportSelfSymmetryDRPhaseRouteCount(const set<frNet*, frBlockObjectComp> &targetNets) const {
  int shapes = 0;
  int vias = 0;
  int patchWires = 0;
  for (auto net: targetNets) {
    if (net == nullptr) {
      continue;
    }
    shapes += net->getShapes().size();
    vias += net->getVias().size();
    patchWires += net->getPatchWires().size();
  }
  cout << "after_phase_shapes/vias/patch_wires: "
       << shapes << "/" << vias << "/" << patchWires << "\n";
}

void FlexDR::keepOnlySelfSymmetryDRTargetRoutes(
    const set<frNet*, frBlockObjectComp> &targetNets) {
  if (getDesign() == nullptr || getDesign()->getTopBlock() == nullptr) {
    return;
  }
  auto regionQuery = getRegionQuery();
  auto block = getDesign()->getTopBlock();
  for (auto &uNet: getDesign()->getTopBlock()->getNets()) {
    auto net = uNet.get();
    if (targetNets.find(net) != targetNets.end() ||
        block->isRoutedNet(net->getName())) {
      continue;
    }
    for (auto &shape: net->getShapes()) {
      if (regionQuery != nullptr) {
        regionQuery->removeDRObj(shape.get());
      }
    }
    for (auto &via: net->getVias()) {
      if (regionQuery != nullptr) {
        regionQuery->removeDRObj(via.get());
      }
    }
    for (auto &patchWire: net->getPatchWires()) {
      if (regionQuery != nullptr) {
        regionQuery->removeDRObj(patchWire.get());
      }
    }
    net->getShapes().clear();
    net->getVias().clear();
    net->getPatchWires().clear();
    net->setModified(false);
  }
}

void FlexDR::snapshotSelfSymmetryDRRoutes() {
  selfSymmetryDRRouteSnapshots.clear();
  for (auto net: collectSelfSymmetryDRNets(design)) {
    selfSymmetryDRRouteSnapshots[net] = collectSelfSymmetryDRRouteFingerprint(net);
  }
}

void FlexDR::reportSelfSymmetryDRChecker() const {
  auto nets = collectSelfSymmetryDRNets(design);
  if (nets.empty()) {
    return;
  }

  auto markerCounts = countSelfSymmetryDRMarkers(design);

  cout << "@@@ self-symmetry dr checker @@@" << endl;
  for (auto net: nets) {
    auto originalConstraint = net->getSelfSymmetryConstraint();
    auto constraint = originalConstraint;
    frCoord effectiveAxis = originalConstraint.axis;
    bool axisSnapFailed = !findNearestSelfSymmetryDRTrack(
        design, originalConstraint.isAxisHorizontal, originalConstraint.axis,
        nullptr, effectiveAxis);
    if (!axisSnapFailed) {
      constraint = makeEffectiveSelfSymmetryDRConstraint(originalConstraint,
                                                         effectiveAxis);
    }
    multiset<SelfSymmetryDRSegmentRecord> segments;
    multiset<SelfSymmetryDRViaRecord> vias;
    collectSelfSymmetryDRCheckerRecords(net, segments, vias);

    auto currFingerprint = collectSelfSymmetryDRRouteFingerprint(net);
    auto snapshotIt = selfSymmetryDRRouteSnapshots.find(net);
    bool changedByCurrentDR = snapshotIt == selfSymmetryDRRouteSnapshots.end() ||
                              snapshotIt->second != currFingerprint;

    cout << "net: " << net->getName() << "\n";
    cout << "original_axis: " << originalConstraint.axis << "\n";
    cout << "effective_axis: " << constraint.axis << "\n";
    cout << "axis_snap_delta: " << constraint.axis - originalConstraint.axis << "\n";
    cout << "axis_snap_failed: " << (axisSnapFailed ? 1 : 0) << "\n";
    cout << "missing_axis_contact: "
         << (hasSelfSymmetryDRAxisContact(constraint, segments, vias) ? 0 : 1)
         << "\n";
    cout << "segments_missing_mirror: "
         << countSelfSymmetryDRSegmentsMissingMirror(constraint, segments) << "\n";
    cout << "vias_missing_mirror: "
         << countSelfSymmetryDRViasMissingMirror(constraint, vias) << "\n";
    cout << "axis_duplicate_shapes: "
         << countSelfSymmetryDRAxisDuplicateShapes(constraint, vias) << "\n";
    cout << "self_markers: " << markerCounts[net] << "\n";
    cout << "changed_by_current_dr: " << (changedByCurrentDR ? 1 : 0) << "\n";
  }
}
