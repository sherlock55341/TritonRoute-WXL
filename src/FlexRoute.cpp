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
#include <vector>
#include <unordered_map>
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

  // Candidate naming is only the admission filter; a net becomes constrained
  // after initSelfSymmetryConstraints installs a validated axis on frNet.
  bool isSelfSymmetryCandidateNet(frNet *net) {
    return net && isSelfSymmetryNetName(net->getName());
  }

  // Mirror candidates are paired by the _1/_2 naming convention below; each
  // one still needs its own hardcoded axis entry to become constrained.
  bool isMirrorCandidateNet(frNet *net) {
    if (!net) return false;
    const string &name = net->getName();
    return name.compare(0, 6, "Mirror") == 0;
  }

  struct HardcodedAxis {
    const char* name;
    bool isHorizontal; // H=true (axis is y), V=false (axis is x)
    int axis;
  };

  // Hardcoded from axis.txt — maps net name to self-symmetry axis
  const HardcodedAxis hardcodedAxes[] = {
    {"Symmtry1",  false, 17420},
    {"Symmtry2",  false, 23420},
    {"Symmtry3",  true,  45850},
    {"Symmtry4",  true,  59530},
    {"Symmtry5",  true,  71820},
    {"Symmtry6",  true,  58140},
    {"Symmtry7",  true,  58140},
    {"Symmtry8",  true,  58140},
    {"Symmtry9",  false, 43180},
    {"Symmtry10", true,  58140},
    {"Symmtry11", false, 45035},
    {"Symmtry12", false, 46600},
    {"Symmtry13", false, 46980},
    {"Symmtry14", true,  58140},
    {"Symmtry15", true,  58140},
    {"Symmtry16", false, 42430},
    {"Symmtry17", false, 43940},
    {"Symmtry18", false, 44020},
    {"Symmtry19", false, 43500},
    {"Symmtry20", true,  58140},
    {"Symmtry24", true,  13680},
    {"Symmtry25", true,  13680},
    {"Symmtry26", true,  13686},
    {"Symmtry27", true,  13680},
    {"Symmtry28", true,  13680},
    {"Symmtry29", true,  13680},
    {"Symmtry30", true,  13680},
    {"Symmtry31", false, 44420},
    {"Symmtry32", false, 45600},
    {"Symmtry33", false, 43070},
    {"Symmtry34", false, 45860},
    {"Mirror1_1", true,  76950},
    {"Mirror1_2", true,  76950},
    {"Mirror2_1", false, 28420},
    {"Mirror2_2", false, 28420},
    {"Mirror3_1", false, 37940},
    {"Mirror3_2", false, 37940},
    {"Mirror4_1", true,  88919},
    {"Mirror4_2", true,  88919},
    {"Mirror5_1", true,  99180},
    {"Mirror5_2", true,  99180},
    {"Mirror6_1", false, 41445},
    {"Mirror6_2", false, 41445},
    {"Mirror7_1", false, 42245},
    {"Mirror7_2", false, 42245},
    {"Mirror8_1", false, 41430},
    {"Mirror8_2", false, 41430},
    {"Mirror9_1", false, 43020},
    {"Mirror9_2", false, 43020},
    {"Mirror10_1", false, 42500},
    {"Mirror10_2", false, 42500},
    {"Mirror11_1", false, 42523},
    {"Mirror11_2", false, 42523},
    {"Mirror12_1", false, 42940},
    {"Mirror12_2", false, 42940},
    {"Mirror13_1", true,  10259},
    {"Mirror13_2", true,  10259},
    {"Mirror14_1", true,  10259},
    {"Mirror14_2", true,  10259},
    {"Mirror15_1", true,  13679},
    {"Mirror15_2", true,  13679},
    {"Mirror16_1", true,  13680},
    {"Mirror16_2", true,  13680},
    {"Mirror17_1", true,  13680},
    {"Mirror17_2", true,  13680},
    {"Mirror18_1", true,  13680},
    {"Mirror18_2", true,  13680},
    {"Mirror19_1", true,  13680},
    {"Mirror19_2", true,  13680},
    {"Mirror20_1", true,  13679},
    {"Mirror20_2", true,  13679},
    {"Mirror21_1", true,  13679},
    {"Mirror21_2", true,  13679},
    {"Mirror22_1", true,  13680},
    {"Mirror22_2", true,  13680},
  };

  // Validates a candidate hardcoded axis against the die box and snaps it to
  // the nearest routing track, shared by the self-symmetry and mirror init
  // paths so the two don't drift out of sync.
  bool resolveHardcodedAxis(frDesign *design,
                             const frBox &dieBox,
                             bool isHorizontal,
                             frCoord axis,
                             const string &netName,
                             frCoord &snappedAxisOut) {
    auto axisName = isHorizontal ? "y" : "x";
    bool axisInDie = isHorizontal ?
                     axis >= dieBox.bottom() && axis <= dieBox.top() :
                     axis >= dieBox.left() && axis <= dieBox.right();
    if (!axisInDie) {
      cout << "Error: " << netName << " axis "
           << axisName << "=" << axis
           << " is outside die box " << dieBox << "\n";
      exit(1);
    }

    // Snap to a routing track; later mirror operations treat this snapped
    // coordinate as authoritative.
    snappedAxisOut = axis;
    if (!findNearestSelfSymmetryRoutingTrack(design,
                                             isHorizontal,
                                             axis,
                                             &dieBox,
                                             snappedAxisOut)) {
      cout << "Error: " << netName << " axis "
           << axisName << "=" << axis
           << " has no routing track inside die box " << dieBox << "\n";
      exit(1);
    }
    return true;
  }

  // Look up each candidate net's hardcoded axis by name; shared by the
  // self-symmetry and mirror init paths.
  std::unordered_map<std::string, const HardcodedAxis*> buildHardcodedAxisMap() {
    std::unordered_map<std::string, const HardcodedAxis*> axisMap;
    for (auto &ha : hardcodedAxes) {
      axisMap[ha.name] = &ha;
    }
    return axisMap;
  }

  void initSelfSymmetryConstraints(frDesign *design) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr) {
      return;
    }

    auto axisMap = buildHardcodedAxisMap();

    frBox dieBox;
    block->getBoundaryBBox(dieBox);

    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      if (!isSelfSymmetryCandidateNet(net)) {
        continue;
      }

      auto it = axisMap.find(net->getName());
      if (it == axisMap.end()) {
        cout << "Warning: self-symmetry candidate net " << net->getName()
             << " has no hardcoded axis, skipped\n";
        continue;
      }

      bool isHorizontal = it->second->isHorizontal;
      frCoord snappedAxis;
      resolveHardcodedAxis(design, dieBox, isHorizontal, it->second->axis,
                           net->getName(), snappedAxis);

      // Set the constraint only after validation succeeds, so a non-null
      // constraint always denotes a routable axis.
      frSelfSymmetryConstraint selfSymmetryConstraint;
      selfSymmetryConstraint.isAxisHorizontal = isHorizontal;
      selfSymmetryConstraint.axis = snappedAxis;
      net->setSelfSymmetryConstraint(selfSymmetryConstraint);
      net->setConstraint(frNetRoutingConstraint::frcSelfSymmetry);
    }
  }

  // Installs a validated axis on every Mirror-prefixed net with a hardcoded
  // table entry, then links each net to its _1/_2 partner. A mirror
  // constraint is only meaningful once both halves are linked, so nets that
  // can't find a partner are unconstrained again with a warning.
  void initMirrorConstraints(frDesign *design) {
    auto block = design ? design->getTopBlock() : nullptr;
    if (block == nullptr) {
      return;
    }

    auto axisMap = buildHardcodedAxisMap();

    frBox dieBox;
    block->getBoundaryBBox(dieBox);

    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      if (!isMirrorCandidateNet(net)) {
        continue;
      }

      auto it = axisMap.find(net->getName());
      if (it == axisMap.end()) {
        cout << "Warning: mirror candidate net " << net->getName()
             << " has no hardcoded axis, skipped\n";
        continue;
      }

      bool isHorizontal = it->second->isHorizontal;
      frCoord snappedAxis;
      resolveHardcodedAxis(design, dieBox, isHorizontal, it->second->axis,
                           net->getName(), snappedAxis);

      frMirrorConstraint mirrorConstraint;
      mirrorConstraint.isAxisHorizontal = isHorizontal;
      mirrorConstraint.axis = snappedAxis;
      mirrorConstraint.partnerNet = nullptr;
      mirrorConstraint.isLeader = false; // finalized once partners are linked below
      net->setMirrorConstraint(mirrorConstraint);
      net->setConstraint(frNetRoutingConstraint::frcMirror);
    }

    // ponytail: partner derived from the _1/_2 naming convention already used
    // in the benchmark DEFs; switch to an explicit pair table if a future
    // dataset breaks that convention.
    for (auto &uNet: block->getNets()) {
      auto net = uNet.get();
      if (net->getConstraint() != frNetRoutingConstraint::frcMirror) {
        continue;
      }
      if (net->getMirrorConstraint().partnerNet) {
        continue; // already linked from the other side
      }

      const string &name = net->getName();
      char suffix = name.empty() ? '\0' : name.back();
      if (suffix != '1' && suffix != '2') {
        continue;
      }
      string partnerName = name.substr(0, name.size() - 1) + (suffix == '1' ? '2' : '1');
      frNet *partner = block->getNet(partnerName);
      if (!partner || partner->getConstraint() != frNetRoutingConstraint::frcMirror) {
        cout << "Warning: mirror candidate net " << name
             << " has no matching partner, unlinking\n";
        net->setConstraint(frNetRoutingConstraint::frcNone);
        continue;
      }

      auto c = net->getMirrorConstraint();
      c.partnerNet = partner;
      c.isLeader = (suffix == '1'); // "_1" leads first; GR flips the
                                    // effective role per-pass, see plan.
      net->setMirrorConstraint(c);
      auto pc = partner->getMirrorConstraint();
      pc.partnerNet = net;
      pc.isLeader = !c.isLeader;
      partner->setMirrorConstraint(pc);

      cout << "Mirror pair linked: " << name << " <-> " << partnerName
           << " (axis " << (c.isAxisHorizontal ? "y" : "x") << "=" << c.axis << ")\n";
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
  // Constraints must exist before GR/TA/DR classify nets and build mirrored
  // caches, but after tracks and top-block geometry have been parsed.
  initSelfSymmetryConstraints(getDesign());
  initMirrorConstraints(getDesign());
}

void FlexRoute::prep() {
  FlexRP rp(getDesign(), getDesign()->getTech());
  rp.main();
}

void FlexRoute::gr() {
  FlexGR gr(getDesign());
  gr.main();
}

void FlexRoute::ta(RouteNetMode mode) {
  FlexTA ta(getDesign(), mode);
  ta.main();
  io::Writer writer(getDesign());
  writer.writeFromTA();
}

void FlexRoute::dr(RouteNetMode mode) {
  FlexDR dr(getDesign(), mode);
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
  // Route constrained nets first and write their symmetric geometry back
  // before ordinary routing; Mirror pairs get their own pass so ordinary
  // DR no longer treats them as plain nets. Ordinary DR may still pull a
  // self-symmetry net into marker repair when it owns a violation in its
  // repair window (see isOrdinarySelfSymmetryRepairMode).
  ta(RouteNetMode::SelfSymmetryOnly);
  dr(RouteNetMode::SelfSymmetryOnly);
  ta(RouteNetMode::MirrorOnly);
  dr(RouteNetMode::MirrorOnly);
  ta(RouteNetMode::OrdinaryOnly);
  dr(RouteNetMode::OrdinaryOnly);
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
