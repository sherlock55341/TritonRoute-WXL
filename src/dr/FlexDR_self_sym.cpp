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

#include <algorithm>
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
    auto constraint = net->getSelfSymmetryConstraint();
    multiset<SelfSymmetryDRSegmentRecord> segments;
    multiset<SelfSymmetryDRViaRecord> vias;
    collectSelfSymmetryDRCheckerRecords(net, segments, vias);

    auto currFingerprint = collectSelfSymmetryDRRouteFingerprint(net);
    auto snapshotIt = selfSymmetryDRRouteSnapshots.find(net);
    bool changedByCurrentDR = snapshotIt == selfSymmetryDRRouteSnapshots.end() ||
                              snapshotIt->second != currFingerprint;

    cout << "net: " << net->getName() << "\n";
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
