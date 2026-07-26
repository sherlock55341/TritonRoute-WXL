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

  // Scales the extra cost of consuming a planar M1 edge, where a symmetric
  // route is especially likely to require an additional access via.
  constexpr unsigned SELF_SYMMETRY_M1_VIA_PENALTY_COUNT = 4;

  inline long long getSelfSymmetryAbsDiff(frCoord lhs, frCoord rhs) {
    return lhs >= rhs ? (long long)(lhs - rhs) : (long long)(rhs - lhs);
  }

  inline unsigned saturateSelfSymmetryCost(unsigned long long cost) {
    return cost > std::numeric_limits<unsigned>::max() ?
           std::numeric_limits<unsigned>::max() :
           (unsigned)cost;
  }

  inline bool isSelfSymmetryRoutingTrackForAxis(frTrackPattern *trackPattern,
                                                bool isAxisHorizontal) {
    if (trackPattern == nullptr) {
      return false;
    }
    return isAxisHorizontal ? !trackPattern->isHorizontal() :
                              trackPattern->isHorizontal();
  }

  // Admission test for self-symmetry candidate nets; the "Symmtry" spelling
  // is the benchmark's net-name convention and must stay byte-identical.
  inline bool isSelfSymmetryNetName(const std::string &name) {
    return name.compare(0, 7, "Symmtry") == 0;
  }

  // Classifies a coordinate against an axis: -1 below, 0 on, +1 above.
  // Shared by GR (GCell-index space) and DR (DBU space) side tests.
  inline int getSelfSymmetrySide(frCoord coord, frCoord axis) {
    if (coord < axis) {
      return -1;
    }
    if (coord > axis) {
      return 1;
    }
    return 0;
  }

  // Reflects a point about the given axis; shared by the GR/DR cache
  // rebuilds and the maze mirror-edge computations.
  inline frPoint mirrorPointAboutAxis(const frPoint &point,
                                      bool axisHorizontal,
                                      frCoord axis) {
    frPoint mirroredPoint;
    if (axisHorizontal) {
      mirroredPoint.set(point.x(), axis + (axis - point.y()));
    } else {
      mirroredPoint.set(axis + (axis - point.x()), point.y());
    }
    return mirroredPoint;
  }

  inline bool findNearestSelfSymmetryRoutingTrack(
      frDesign *design,
      bool isAxisHorizontal,
      frCoord axis,
      const frBox *preferredBox,
      frCoord &trackCoord) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr) {
      return false;
    }

    // Search every routing layer because the axis is a design-space invariant,
    // not a commitment to one layer.  Ties choose the lower DBU coordinate so
    // all stages derive the same axis deterministically.
    bool found = false;
    long long bestDist = std::numeric_limits<long long>::max();
    frCoord bestCoord = axis;
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
          }
        }
      }
    }

    if (!found) {
      return false;
    }
    trackCoord = bestCoord;
    return true;
  }

  // Carries the same authoritative axis in both design DBU and GCell-index
  // spaces.  The GCell form is derived for maze decisions; it must never be
  // written back as a physical coordinate.
  struct SelfSymmetryAxisContext {
    bool valid = false;
    bool isAxisHorizontal = false;
    frCoord axis = 0;
    frCoord axisGCellIdx = 0;

    static SelfSymmetryAxisContext fromAxisProbe(
        frDesign *design,
        bool isAxisHorizontal,
        frCoord axis,
        const frPoint &axisProbe) {
      SelfSymmetryAxisContext ctx;
      auto block = design ? design->getTopBlock() : nullptr;
      if (block == nullptr) {
        return ctx;
      }
      frPoint effectiveAxisProbe(axisProbe);
      if (isAxisHorizontal) {
        effectiveAxisProbe.set(axisProbe.x(), axis);
      } else {
        effectiveAxisProbe.set(axis, axisProbe.y());
      }
      frPoint axisGCellLocation;
      block->getGCellIdx(effectiveAxisProbe, axisGCellLocation);
      ctx.valid = true;
      ctx.isAxisHorizontal = isAxisHorizontal;
      ctx.axis = axis;
      ctx.axisGCellIdx = isAxisHorizontal ?
                         axisGCellLocation.y() :
                         axisGCellLocation.x();
      return ctx;
    }

    static SelfSymmetryAxisContext fromAxisProbe(
        frDesign *design,
        const frSelfSymmetryConstraint &constraint,
        const frPoint &axisProbe) {
      return fromAxisProbe(design, constraint.isAxisHorizontal,
                            constraint.axis, axisProbe);
    }

    static SelfSymmetryAxisContext fromReferencePoint(
        frDesign *design,
        const frSelfSymmetryConstraint &constraint,
        const frPoint &referencePoint) {
      frPoint axisProbe;
      if (constraint.isAxisHorizontal) {
        axisProbe.set(referencePoint.x(), constraint.axis);
      } else {
        axisProbe.set(constraint.axis, referencePoint.y());
      }
      return fromAxisProbe(design, constraint, axisProbe);
    }

    frCoord axisCoord(const frPoint &gcellIdx) const {
      return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
    }

    frPoint mirrorPoint(const frPoint &point) const {
      return mirrorPointAboutAxis(point, isAxisHorizontal, axis);
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

}

#endif
