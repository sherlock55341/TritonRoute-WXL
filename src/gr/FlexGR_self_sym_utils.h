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

#ifndef _FLEX_GR_SELF_SYM_UTILS_H_
#define _FLEX_GR_SELF_SYM_UTILS_H_

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "frDesign.h"
#include "global.h"

namespace fr {

  inline unsigned long long computeInvalidMirrorPenalty(long long edgeLen) {
    return (unsigned long long)std::max(1ll, edgeLen) *
           ((unsigned long long)BLOCKCOST * 100 +
            (unsigned long long)MARKERCOST * 8);
  }

  inline int normalizeSelfSymmetryRootSide(int side) {
    return side == 0 ? -1 : side;
  }

  inline std::pair<int, int> getSelfSymmetryGCellKey(const frPoint &point) {
    return std::make_pair((int)point.x(), (int)point.y());
  }

  inline std::pair<frPoint, frPoint> normalizeSelfSymmetryEdge(frPoint begin,
                                                               frPoint end) {
    if (end < begin) {
      std::swap(begin, end);
    }
    return std::make_pair(begin, end);
  }

  inline long long getSelfSymmetryEdgeLen(const frPoint &begin,
                                          const frPoint &end) {
    long long dx = begin.x() >= end.x() ? begin.x() - end.x() :
                                          end.x() - begin.x();
    long long dy = begin.y() >= end.y() ? begin.y() - end.y() :
                                          end.y() - begin.y();
    return std::max(1ll, dx + dy);
  }

  inline unsigned saturateSelfSymmetryCost(unsigned long long cost) {
    return cost > std::numeric_limits<unsigned>::max() ?
           std::numeric_limits<unsigned>::max() :
           (unsigned)cost;
  }

  inline long long getSelfSymmetryAbsDiff(frCoord lhs, frCoord rhs) {
    return lhs >= rhs ? (long long)(lhs - rhs) : (long long)(rhs - lhs);
  }

  inline bool isSelfSymmetryRoutingTrackForAxis(frTrackPattern *trackPattern,
                                                bool isAxisHorizontal) {
    if (trackPattern == nullptr) {
      return false;
    }
    return isAxisHorizontal ? !trackPattern->isHorizontal() :
                              trackPattern->isHorizontal();
  }

  inline bool findNearestSelfSymmetryRoutingTrack(
      frDesign *design,
      bool isAxisHorizontal,
      frCoord axis,
      const frBox *preferredBox,
      frCoord &trackCoord,
      frLayerNum *layerNum = nullptr,
      frTrackPattern **trackPattern = nullptr) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr) {
      return false;
    }

    bool found = false;
    long long bestDist = std::numeric_limits<long long>::max();
    frCoord bestCoord = axis;
    frLayerNum bestLayerNum = 0;
    frTrackPattern *bestTrackPattern = nullptr;
    for (auto &layer: design->getTech()->getLayers()) {
      if (layer->getType() != frLayerTypeEnum::ROUTING) {
        continue;
      }
      auto currLayerNum = layer->getLayerNum();
      for (auto &uTrackPattern: block->getTrackPatterns(currLayerNum)) {
        auto tp = uTrackPattern.get();
        if (!isSelfSymmetryRoutingTrackForAxis(tp, isAxisHorizontal) ||
            tp->getNumTracks() == 0 || tp->getTrackSpacing() == 0) {
          continue;
        }

        int minTrackNum = 0;
        int maxTrackNum = (int)tp->getNumTracks() - 1;
        if (preferredBox != nullptr) {
          auto low = isAxisHorizontal ? preferredBox->bottom() :
                                        preferredBox->left();
          auto high = isAxisHorizontal ? preferredBox->top() :
                                         preferredBox->right();
          minTrackNum = (low - tp->getStartCoord()) /
                        (int)tp->getTrackSpacing();
          if (minTrackNum < 0) {
            minTrackNum = 0;
          }
          if (minTrackNum * (int)tp->getTrackSpacing() +
                  tp->getStartCoord() < low) {
            ++minTrackNum;
          }
          maxTrackNum = (high - tp->getStartCoord()) /
                        (int)tp->getTrackSpacing();
          if (maxTrackNum >= (int)tp->getNumTracks()) {
            maxTrackNum = (int)tp->getNumTracks() - 1;
          }
          if (maxTrackNum * (int)tp->getTrackSpacing() +
                  tp->getStartCoord() > high) {
            --maxTrackNum;
          }
          if (minTrackNum > maxTrackNum) {
            continue;
          }
        }

        int nearestTrackNum = (axis - tp->getStartCoord()) /
                              (int)tp->getTrackSpacing();
        nearestTrackNum = std::max(minTrackNum,
                                   std::min(maxTrackNum, nearestTrackNum));
        for (int delta = -1; delta <= 1; ++delta) {
          int trackNum = nearestTrackNum + delta;
          if (trackNum < minTrackNum || trackNum > maxTrackNum) {
            continue;
          }
          auto currCoord = trackNum * (int)tp->getTrackSpacing() +
                           tp->getStartCoord();
          auto currDist = getSelfSymmetryAbsDiff(currCoord, axis);
          if (!found || currDist < bestDist ||
              (currDist == bestDist && currCoord < bestCoord)) {
            found = true;
            bestDist = currDist;
            bestCoord = currCoord;
            bestLayerNum = currLayerNum;
            bestTrackPattern = tp;
          }
        }
      }
    }

    if (!found) {
      return false;
    }
    trackCoord = bestCoord;
    if (layerNum != nullptr) {
      *layerNum = bestLayerNum;
    }
    if (trackPattern != nullptr) {
      *trackPattern = bestTrackPattern;
    }
    return true;
  }

  inline bool selfSymmetrySegmentCovers(const frPoint &segmentBegin,
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
      return std::min(candidateBegin.y(), candidateEnd.y()) >=
                 std::min(segmentBegin.y(), segmentEnd.y()) &&
             std::max(candidateBegin.y(), candidateEnd.y()) <=
                 std::max(segmentBegin.y(), segmentEnd.y());
    }
    if (candidateBegin.y() == candidateEnd.y()) {
      if (segmentBegin.y() != segmentEnd.y() ||
          segmentBegin.y() != candidateBegin.y()) {
        return false;
      }
      return std::min(candidateBegin.x(), candidateEnd.x()) >=
                 std::min(segmentBegin.x(), segmentEnd.x()) &&
             std::max(candidateBegin.x(), candidateEnd.x()) <=
                 std::max(segmentBegin.x(), segmentEnd.x());
    }
    return false;
  }

  struct SelfSymmetryAxisContext {
    bool valid = false;
    bool axisSnapFailed = false;
    bool isAxisHorizontal = false;
    frCoord originalAxis = 0;
    frCoord axis = 0;
    frCoord axisSnapDelta = 0;
    frCoord axisGCellIdx = 0;
    int rootSide = 0;

    static SelfSymmetryAxisContext fromGCellAxisOnly(bool isAxisHorizontalIn,
                                                     frCoord axisGCellIdxIn,
                                                     int rootSideIn = 0) {
      SelfSymmetryAxisContext ctx;
      ctx.valid = true;
      ctx.isAxisHorizontal = isAxisHorizontalIn;
      ctx.originalAxis = axisGCellIdxIn;
      ctx.axis = axisGCellIdxIn;
      ctx.axisGCellIdx = axisGCellIdxIn;
      ctx.rootSide = rootSideIn;
      return ctx;
    }

    static SelfSymmetryAxisContext fromAxisProbe(frDesign *design,
                                                 const frSelfSymmetryConstraint &constraint,
                                                 const frPoint &axisProbe,
                                                 int rootSideIn = 0) {
      SelfSymmetryAxisContext ctx;
      auto block = design ? design->getTopBlock() : nullptr;
      if (block == nullptr) {
        return ctx;
      }
      frCoord effectiveAxis = constraint.axis;
      if (!findNearestSelfSymmetryRoutingTrack(design,
                                               constraint.isAxisHorizontal,
                                               constraint.axis,
                                               nullptr,
                                               effectiveAxis)) {
        ctx.axisSnapFailed = true;
      }
      frPoint effectiveAxisProbe(axisProbe);
      if (constraint.isAxisHorizontal) {
        effectiveAxisProbe.set(axisProbe.x(), effectiveAxis);
      } else {
        effectiveAxisProbe.set(effectiveAxis, axisProbe.y());
      }
      frPoint axisGCellLocation;
      block->getGCellIdx(effectiveAxisProbe, axisGCellLocation);
      ctx.valid = true;
      ctx.isAxisHorizontal = constraint.isAxisHorizontal;
      ctx.originalAxis = constraint.axis;
      ctx.axis = effectiveAxis;
      ctx.axisSnapDelta = ctx.axis - ctx.originalAxis;
      ctx.axisGCellIdx = constraint.isAxisHorizontal ?
                         axisGCellLocation.y() :
                         axisGCellLocation.x();
      ctx.rootSide = rootSideIn;
      return ctx;
    }

    static SelfSymmetryAxisContext fromReferencePoint(
        frDesign *design,
        const frSelfSymmetryConstraint &constraint,
        const frPoint &referencePoint,
        int rootSideIn = 0) {
      frPoint axisProbe;
      if (constraint.isAxisHorizontal) {
        axisProbe.set(referencePoint.x(), constraint.axis);
      } else {
        axisProbe.set(constraint.axis, referencePoint.y());
      }
      return fromAxisProbe(design, constraint, axisProbe, rootSideIn);
    }

    frCoord axisCoord(const frPoint &gcellIdx) const {
      return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
    }

    int sideOfGCell(const frPoint &gcellIdx) const {
      auto coord = axisCoord(gcellIdx);
      if (coord < axisGCellIdx) {
        return -1;
      }
      if (coord > axisGCellIdx) {
        return 1;
      }
      return 0;
    }

    int sideOfPoint(const frPoint &point) const {
      auto coord = isAxisHorizontal ? point.y() : point.x();
      if (coord < axis) {
        return -1;
      }
      if (coord > axis) {
        return 1;
      }
      return 0;
    }

    bool isAxisEdge(const frPoint &begin, const frPoint &end) const {
      return axisCoord(begin) == axisGCellIdx &&
             axisCoord(end) == axisGCellIdx;
    }

    frPoint mirrorPoint(const frPoint &point) const {
      frPoint mirroredPoint(point);
      if (isAxisHorizontal) {
        mirroredPoint.set(point.x(), axis + (axis - point.y()));
      } else {
        mirroredPoint.set(axis + (axis - point.x()), point.y());
      }
      return mirroredPoint;
    }

    frPoint mirrorGCell(const frPoint &gcellIdx) const {
      frPoint mirroredGCellIdx(gcellIdx);
      if (isAxisHorizontal) {
        mirroredGCellIdx.set(gcellIdx.x(),
                             axisGCellIdx + (axisGCellIdx - gcellIdx.y()));
      } else {
        mirroredGCellIdx.set(axisGCellIdx + (axisGCellIdx - gcellIdx.x()),
                             gcellIdx.y());
      }
      return mirroredGCellIdx;
    }
  };

  struct SelfSymmetryMirrorEdgeResult {
    bool valid = false;
    bool axisOnly = false;
    frMIdx mirrorX = 0;
    frMIdx mirrorY = 0;
    frMIdx mirrorZ = 0;
    frDirEnum mirrorDir = frDirEnum::UNKNOWN;
    unsigned long long invalidPenalty = 0;
  };

  inline SelfSymmetryMirrorEdgeResult makeSelfSymmetryMirrorEdge(
      const SelfSymmetryAxisContext &ctx,
      const frPoint &beginGCellIdx,
      const frPoint &endGCellIdx,
      const frPoint &boxLL,
      const frPoint &boxUR,
      frMIdx z,
      long long edgeLenForPenalty = 0) {
    SelfSymmetryMirrorEdgeResult result;
    auto getDir = [](const frPoint &begin, const frPoint &end) {
      if (begin.x() != end.x()) {
        return begin.x() < end.x() ? frDirEnum::E : frDirEnum::W;
      }
      if (begin.y() != end.y()) {
        return begin.y() < end.y() ? frDirEnum::N : frDirEnum::S;
      }
      return frDirEnum::UNKNOWN;
    };
    auto sourceDir = getDir(beginGCellIdx, endGCellIdx);
    long long edgeLen = edgeLenForPenalty;
    if (edgeLen <= 0) {
      long long dx = beginGCellIdx.x() >= endGCellIdx.x() ?
                     beginGCellIdx.x() - endGCellIdx.x() :
                     endGCellIdx.x() - beginGCellIdx.x();
      long long dy = beginGCellIdx.y() >= endGCellIdx.y() ?
                     beginGCellIdx.y() - endGCellIdx.y() :
                     endGCellIdx.y() - beginGCellIdx.y();
      edgeLen = std::max(1ll, dx + dy);
    }
    result.invalidPenalty = computeInvalidMirrorPenalty(edgeLen);

    if (!ctx.valid || sourceDir == frDirEnum::UNKNOWN) {
      return result;
    }
    if (ctx.isAxisEdge(beginGCellIdx, endGCellIdx)) {
      result.axisOnly = true;
      return result;
    }

    auto mirrorBegin = ctx.mirrorGCell(beginGCellIdx);
    auto mirrorEnd = ctx.mirrorGCell(endGCellIdx);
    auto mirrorDir = getDir(mirrorBegin, mirrorEnd);
    if (mirrorDir == frDirEnum::UNKNOWN ||
        (mirrorBegin.x() != mirrorEnd.x() &&
         mirrorBegin.y() != mirrorEnd.y())) {
      return result;
    }
    auto isGCellInBox = [&](const frPoint &gcellIdx) {
      return gcellIdx.x() >= boxLL.x() && gcellIdx.x() <= boxUR.x() &&
             gcellIdx.y() >= boxLL.y() && gcellIdx.y() <= boxUR.y();
    };
    if (!isGCellInBox(mirrorBegin) || !isGCellInBox(mirrorEnd)) {
      return result;
    }

    result.mirrorX = mirrorBegin.x() - boxLL.x();
    result.mirrorY = mirrorBegin.y() - boxLL.y();
    result.mirrorZ = z;
    result.mirrorDir = mirrorDir;
    result.valid = true;
    return result;
  }

  void get_self_symmetry_axis(const std::vector<frPoint> &points,
                              bool &is_horizontal, int &coor);

  struct SelfSymmetryDebug {
    static const std::string& netName() {
      static const std::string name = "Symmtry5";
      return name;
    }

    static bool isDebugNet(const frNet *net) {
      return net != nullptr && net->getName() == netName();
    }
  };

}

#endif
