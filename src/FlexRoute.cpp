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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "FlexRoute.h"
#include "dr/FlexDR.h"
#include "global.h"
#include "io/io.h"
#include "pa/FlexPA.h"
#include "ta/FlexTA.h"
#include <iostream>
// #include "io/frPinPrep.h"
#include "gc/FlexGC.h"
#include "gr/FlexGR.h"
#include "gta/GTA.hpp"
#include "rp/FlexRP.h"
#include <chrono>

using namespace std;
using namespace fr;

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
    auto tp_0 = std::chrono::high_resolution_clock::now();
    if (ENABLE_GTA) {
        gta::GTA gtaWorker(getDesign());
        gtaWorker.run(2);
    } else {
        FlexTA ta(getDesign());
        ta.main();
    }
    auto tp_1 = std::chrono::high_resolution_clock::now();
    std::cout << "Track Assignment Runtime : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tp_1 -
                                                                       tp_0)
                         .count() /
                     1e3
              << " s" << std::endl;
    io::Writer writer(getDesign());
    writer.writeFromTA();
    // exit(0);
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
    return 0;
    dr();
    endFR();

    /*
    // rtree test
    vector<rtree_frConnFig_value_t> result1;
    design->getTopBlock()->queryRtree4Routes(frBox(585000, 1098000, 590000,
    1101000), 6, result1); cout <<endl <<"query1:" <<endl; for (auto &it:
    result1) { if (it.second->typeId() == frcPathSeg) { frPoint pt1, pt2;
        dynamic_pointer_cast<frPathSeg>(it.second)->getPoints(pt1, pt2);
        cout <<"found pathseg " <<pt1.x() <<" " <<pt1.y() <<" " << pt2.x() <<" "
    <<pt2.y()
             <<" "
    <<dynamic_pointer_cast<frPathSeg>(it.second)->getNet()->getName() <<endl; }
    else if (it.second->typeId() == frcGuide) { frPoint pt1, pt2;
        dynamic_pointer_cast<frGuide>(it.second)->getPoints(pt1, pt2);
        cout <<"found guide   " <<pt1.x() <<" " <<pt1.y() <<" " << pt2.x() <<" "
    <<pt2.y()
             <<" "
    <<dynamic_pointer_cast<frGuide>(it.second)->getNet()->getName(); if
    (dynamic_pointer_cast<frGuide>(it.second)->getBeginLayerNum() !=
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
