/* Authors: Lutong Wang and Bangqi Xu */
/*
 * Copyright (c) 2019, The Regents of the University of California
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain this copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce this copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _FR_SYMMETRY_CONSTRAINT_H_
#define _FR_SYMMETRY_CONSTRAINT_H_

#include "frBaseTypes.h"
#include "db/infra/frBox.h"
#include "db/infra/frPoint.h"

namespace fr {

enum class frSymmetryAxisEnum { Horizontal = 0, Vertical = 1 };
enum class frSymmetryReferenceSideEnum { Low = 0, High = 1 };
enum class frSymmetrySideEnum { Low = 0, OnAxis = 1, High = 2 };

class frSymmetryConstraint {
   public:
    frSymmetryConstraint()
        : axisDir(frSymmetryAxisEnum::Horizontal),
          axisCoord(0),
          referenceSide(frSymmetryReferenceSideEnum::High) {}
    frSymmetryConstraint(const frString &inNetName,
                         frSymmetryAxisEnum inAxisDir, frCoord inAxisCoord,
                         frSymmetryReferenceSideEnum inReferenceSide)
        : netName(inNetName),
          axisDir(inAxisDir),
          axisCoord(inAxisCoord),
          referenceSide(inReferenceSide) {}

    const frString &getNetName() const { return netName; }
    void setNetName(const frString &in) { netName = in; }

    frSymmetryAxisEnum getAxisDir() const { return axisDir; }
    void setAxisDir(frSymmetryAxisEnum in) { axisDir = in; }

    frCoord getAxisCoord() const { return axisCoord; }
    void setAxisCoord(frCoord in) { axisCoord = in; }

    frSymmetryReferenceSideEnum getReferenceSide() const {
        return referenceSide;
    }
    void setReferenceSide(frSymmetryReferenceSideEnum in) {
        referenceSide = in;
    }

    bool isAxis(const frPoint &point) const {
        return getPointSide(point) == frSymmetrySideEnum::OnAxis;
    }
    frSymmetrySideEnum getPointSide(const frPoint &point) const {
        if (axisDir == frSymmetryAxisEnum::Horizontal) {
            if (point.y() == axisCoord) {
                return frSymmetrySideEnum::OnAxis;
            }
            return (point.y() >= axisCoord) ? frSymmetrySideEnum::High
                                            : frSymmetrySideEnum::Low;
        }
        if (point.x() == axisCoord) {
            return frSymmetrySideEnum::OnAxis;
        }
        return (point.x() >= axisCoord) ? frSymmetrySideEnum::High
                                        : frSymmetrySideEnum::Low;
    }

    bool isReference(const frPoint &point) const {
        auto side = getPointSide(point);
        if (side == frSymmetrySideEnum::OnAxis) {
            return true;
        }
        return side == (referenceSide == frSymmetryReferenceSideEnum::High
                            ? frSymmetrySideEnum::High
                            : frSymmetrySideEnum::Low);
    }

    bool isMirror(const frPoint &point) const {
        auto side = getPointSide(point);
        if (side == frSymmetrySideEnum::OnAxis) {
            return false;
        }
        return side != (referenceSide == frSymmetryReferenceSideEnum::High
                            ? frSymmetrySideEnum::High
                            : frSymmetrySideEnum::Low);
    }

    frPoint getMirroredPoint(const frPoint &point) const {
        if (axisDir == frSymmetryAxisEnum::Horizontal) {
            return frPoint(point.x(), axisCoord + (axisCoord - point.y()));
        }
        return frPoint(axisCoord + (axisCoord - point.x()), point.y());
    }

    frBox getMirroredBox(const frBox &box) const {
        frPoint mirroredLl = getMirroredPoint(box.lowerLeft());
        frPoint mirroredUr = getMirroredPoint(box.upperRight());
        return frBox(mirroredLl, mirroredUr);
    }

   protected:
    frString netName;
    frSymmetryAxisEnum axisDir;
    frCoord axisCoord;
    frSymmetryReferenceSideEnum referenceSide;
};

}  // namespace fr

#endif
