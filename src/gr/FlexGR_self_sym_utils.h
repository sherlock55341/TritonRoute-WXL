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
#include <cmath>
#include <iterator>
#include <limits>
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

    // Search every routing layer because the axis is a design-space invariant,
    // not a commitment to one layer.  Ties choose the lower DBU coordinate so
    // all stages derive the same axis deterministically.
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
        const frSelfSymmetryConstraint &constraint,
        const frPoint &axisProbe) {
      SelfSymmetryAxisContext ctx;
      auto block = design ? design->getTopBlock() : nullptr;
      if (block == nullptr) {
        return ctx;
      }
      frPoint effectiveAxisProbe(axisProbe);
      if (constraint.isAxisHorizontal) {
        effectiveAxisProbe.set(axisProbe.x(), constraint.axis);
      } else {
        effectiveAxisProbe.set(constraint.axis, axisProbe.y());
      }
      frPoint axisGCellLocation;
      block->getGCellIdx(effectiveAxisProbe, axisGCellLocation);
      ctx.valid = true;
      ctx.isAxisHorizontal = constraint.isAxisHorizontal;
      ctx.axis = constraint.axis;
      ctx.axisGCellIdx = constraint.isAxisHorizontal ?
                         axisGCellLocation.y() :
                         axisGCellLocation.x();
      return ctx;
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

    frCoord axisCoord(const frPoint &gcellIdx) const {
      return isAxisHorizontal ? gcellIdx.y() : gcellIdx.x();
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

  inline void get_self_symmetry_axis(const std::vector<frPoint> &points,
                                     bool &is_horizontal,
                                     int &coor) {
    // Prefer the cheap moment test when one orientation is unambiguous, then
    // fall back to nearest-neighbor mirror error for nearly balanced samples.
    double mean_x = 0;
    double mean_y = 0;
    double sigma_x = 0;
    double sigma_y = 0;
    for (auto p: points) {
      mean_x += p.x();
      mean_y += p.y();
    }
    mean_x /= points.size();
    mean_y /= points.size();

    for (auto p: points) {
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
    for (auto p: points) {
      auto dx = sigma_x == 0 ? 0 : (p.x() - mean_x) / sigma_x;
      auto dy = sigma_y == 0 ? 0 : (p.y() - mean_y) / sigma_y;
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
      return;
    }
    if (sum_moment_y * 2 < sum_moment_x) {
      is_horizontal = true;
      coor = std::round(mean_y);
      return;
    }

    std::vector<point_t> rtree_points;
    rtree_points.reserve(points.size());
    for (auto p: points) {
      rtree_points.push_back(point_t(p.x(), p.y()));
    }
    bgi::rtree<point_t, bgi::quadratic<16> > tree(rtree_points);

    double mirror_x_score = 0;
    double mirror_y_score = 0;
    for (auto p: points) {
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

#endif
