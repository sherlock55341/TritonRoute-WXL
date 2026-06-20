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

#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include "global.h"
#include "FlexRoute.h"
#include "db/infra/frTransform.h"
#include "db/obj/frAccess.h"
#include "db/obj/frInst.h"
#include "db/obj/frInstTerm.h"
#include "db/obj/frRPin.h"
#include "io/io.h"
#include "pa/FlexPA.h"
#include "ta/FlexTA.h"
#include "dr/FlexDR.h"
//#include "io/frPinPrep.h"
#include "gc/FlexGC.h"
#include "gr/FlexGR.h"
#include "gr/FlexGR_self_sym_utils.h"
#include "rp/FlexRP.h"

using namespace std;
using namespace fr;

namespace {

  bool isSelfSymmetryCandidateNet(frNet *net) {
    static const string prefix = "Symmtry";
    return net != nullptr &&
           net->getName().compare(0, prefix.size(), prefix) == 0;
  }

  bool getOldHardcodedSelfSymmetryAxis(const string &name,
                                       bool &isHorizontal,
                                       int &axis) {
    struct AxisSpec {
      const char *name;
      bool isHorizontal;
      int axis;
    };
    static const AxisSpec specs[] = {
        {"Symmtry1", false, 17400},
        {"Symmtry2", false, 23400},
        {"Symmtry3", true, 45790},
        {"Symmtry4", true, 59850},
        {"Symmtry5", true, 72000},
    };
    for (auto &spec: specs) {
      if (name == spec.name) {
        isHorizontal = spec.isHorizontal;
        axis = spec.axis;
        return true;
      }
    }
    return false;
  }

  frPoint getRPinGlobalAccessPoint(frRPin *rpin) {
    frPoint pt;
    auto ap = rpin->getAccessPoint();
    ap->getPoint(pt);
    if (rpin->getFrTerm()->typeId() == frcInstTerm) {
      auto inst = static_cast<frInstTerm*>(rpin->getFrTerm())->getInst();
      frTransform shiftXform;
      inst->getTransform(shiftXform);
      shiftXform.set(frOrient(frcR0));
      pt.transform(shiftXform);
    } else if (rpin->getFrTerm()->typeId() == frcTerm) {
      ;
    } else {
      cout << "Error: unknown rpin term type in getRPinGlobalAccessPoint\n";
      exit(1);
    }
    return pt;
  }

  void reportSelfSymmetryAxis(frNet *net,
                              const vector<frPoint> &points,
                              bool isHorizontal,
                              int axis) {
    bool oldIsHorizontal = false;
    int oldAxis = 0;
    if (net == nullptr ||
        !getOldHardcodedSelfSymmetryAxis(net->getName(), oldIsHorizontal,
                                         oldAxis)) {
      return;
    }

    int minX = numeric_limits<int>::max();
    int minY = numeric_limits<int>::max();
    int maxX = numeric_limits<int>::min();
    int maxY = numeric_limits<int>::min();
    long long sumX = 0;
    long long sumY = 0;
    for (auto &pt: points) {
      minX = min(minX, pt.x());
      minY = min(minY, pt.y());
      maxX = max(maxX, pt.x());
      maxY = max(maxY, pt.y());
      sumX += pt.x();
      sumY += pt.y();
    }
    auto meanX = static_cast<double>(sumX) / points.size();
    auto meanY = static_cast<double>(sumY) / points.size();
    auto axisName = isHorizontal ? "y" : "x";
    auto oldAxisName = oldIsHorizontal ? "y" : "x";

    cout << "[selfsym-axis] net=" << net->getName()
         << " auto=" << axisName << "=" << axis
         << " old_hardcoded=" << oldAxisName << "=" << oldAxis;
    if (isHorizontal == oldIsHorizontal) {
      cout << " delta=" << axis - oldAxis;
    } else {
      cout << " delta=orientation-mismatch";
    }
    cout << " ap_count=" << points.size()
         << " ap_bbox=(" << minX << "," << minY << ")-("
         << maxX << "," << maxY << ")"
         << " ap_mean=(" << meanX << "," << meanY << ")"
         << "\n";
    for (auto &pt: points) {
      cout << "[selfsym-axis]   ap=(" << pt.x() << "," << pt.y() << ")\n";
    }
  }

  void initSelfSymmetryConstraints(frDesign *design) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr) {
      return;
    }

    frBox dieBox;
    block->getBoundaryBBox(dieBox);
    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      if (!isSelfSymmetryCandidateNet(net)) {
        continue;
      }

      vector<frPoint> points;
      points.reserve(net->getRPins().size());
      for (auto &rpin: net->getRPins()) {
        if (rpin->getAccessPoint() == nullptr || rpin->getFrTerm() == nullptr) {
          continue;
        }
        points.push_back(getRPinGlobalAccessPoint(rpin.get()));
      }
      if (points.empty()) {
        continue;
      }

      bool isHorizontal = false;
      int axis = 0;
      get_self_symmetry_axis(points, isHorizontal, axis);
      reportSelfSymmetryAxis(net, points, isHorizontal, axis);

      auto axisName = isHorizontal ? "y" : "x";
      bool axisInDie = isHorizontal ?
                       axis >= dieBox.bottom() && axis <= dieBox.top() :
                       axis >= dieBox.left() && axis <= dieBox.right();
      if (!axisInDie) {
        cout << "Error: " << net->getName() << " self-symmetry axis "
             << axisName << "=" << axis
             << " is outside die box " << dieBox << "\n";
        exit(1);
      }

      frSelfSymmetryConstraint selfSymmetryConstraint;
      selfSymmetryConstraint.isAxisHorizontal = isHorizontal;
      selfSymmetryConstraint.axis = axis;
      net->setSelfSymmetryConstraint(selfSymmetryConstraint);
      net->setConstraint(frNetRoutingConstraint::frcSelfSymmetry);
    }
  }

}

void FlexRoute::init() {
  io::Parser parser(getDesign());
  parser.readLefDef();
  if (GUIDE_FILE != string("")) {
    parser.readGuide();
  } else {
    ENABLE_VIA_GEN = false;
  }
  parser.postProcess();
  FlexPA pa(getDesign());
  pa.main();
  if (GUIDE_FILE != string("")) {
    parser.postProcessGuide();
  }
  // GR-related
  parser.initRPin();
  initSelfSymmetryConstraints(getDesign());
}

void FlexRoute::prep() {
  FlexRP rp(getDesign(), getDesign()->getTech());
  rp.main();
}

void FlexRoute::gr() {
  FlexGR gr(getDesign());
  gr.main();
}

void FlexRoute::ta() {
  FlexTA ta(getDesign());
  ta.main();
  io::Writer writer(getDesign());
  writer.writeFromTA();
}

void FlexRoute::dr() {
  FlexDR dr(getDesign());
  dr.main();
}

void FlexRoute::endFR() {
  io::Writer writer(getDesign());
  writer.writeFromDR();
  if (REF_OUT_FILE != DEF_FILE) {
    remove(REF_OUT_FILE.c_str());
  }
}

int FlexRoute::main() {
  init();
  if (GUIDE_FILE == string("")) {
    gr();
    io::Parser parser(getDesign());
    GUIDE_FILE = OUTGUIDE_FILE;
    ENABLE_VIA_GEN = true;
    parser.readGuide();
    parser.initDefaultVias();
    parser.writeRefDef();
    parser.postProcessGuide();
  }
  prep();
  ta();
  dr();
  endFR();


  /*
  // rtree test
  vector<rtree_frConnFig_value_t> result1;
  design->getTopBlock()->queryRtree4Routes(frBox(585000, 1098000, 590000, 1101000), 6, result1);
  cout <<endl <<"query1:" <<endl;
  for (auto &it: result1) {
    if (it.second->typeId() == frcPathSeg) {
      frPoint pt1, pt2;
      dynamic_pointer_cast<frPathSeg>(it.second)->getPoints(pt1, pt2);
      cout <<"found pathseg " <<pt1.x() <<" " <<pt1.y() <<" " << pt2.x() <<" " <<pt2.y() 
           <<" " <<dynamic_pointer_cast<frPathSeg>(it.second)->getNet()->getName() <<endl;
    } else if (it.second->typeId() == frcGuide) {
      frPoint pt1, pt2;
      dynamic_pointer_cast<frGuide>(it.second)->getPoints(pt1, pt2);
      cout <<"found guide   " <<pt1.x() <<" " <<pt1.y() <<" " << pt2.x() <<" " <<pt2.y()
           <<" " <<dynamic_pointer_cast<frGuide>(it.second)->getNet()->getName();
      if (dynamic_pointer_cast<frGuide>(it.second)->getBeginLayerNum() !=
          dynamic_pointer_cast<frGuide>(it.second)->getEndLayerNum()) {
        cout <<" via guide";
      }
      cout <<endl;
    }
  }
  exit(0);
  */
  return 0;
}
