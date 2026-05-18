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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "FlexRoute.h"
#include "db/infra/frSymmetryConstraint.h"
#include "global.h"

using namespace fr;
using namespace std;

namespace {

constexpr const char* kDemoNetName = "Symmtry5";
constexpr const char* kDemoLefFile =
    "/home/cyzhao/benchmark/primarius/outdata/ispd18_test1.input.lef";
constexpr const char* kDemoDefFile =
    "/home/cyzhao/benchmark/primarius/outdata/pattern_route_lay.def";
constexpr const char* kDemoOutGuideName = "symmetry.route.guide";
constexpr const char* kDemoOutName = "symmetry_routed.def";
constexpr int kDemoAxisY = 71820;

string getExecutableDir(const char* executablePath) {
    string path(executablePath);
    auto slashPos = path.find_last_of('/');
    if (slashPos == string::npos) {
        return ".";
    }
    return path.substr(0, slashPos);
}

string joinPath(const string& dir, const string& fileName) {
    if (dir.empty() || dir == ".") {
        return "./" + fileName;
    }
    return dir + "/" + fileName;
}

void initDemoDefaults(const char* executablePath) {
    string outputDir = getExecutableDir(executablePath);
    LEF_FILE = kDemoLefFile;
    DEF_FILE = kDemoDefFile;
    REF_OUT_FILE = DEF_FILE;
    GUIDE_FILE.clear();
    OUTGUIDE_FILE = joinPath(outputDir, kDemoOutGuideName);
    OUTTA_FILE.clear();
    OUT_FILE = joinPath(outputDir, kDemoOutName);
}

void printUsage() {
    cout << "Usage: ./TritonRouteSymm "
            "[-lef <LEF_FILE>] [-def <DEF_FILE>] [-guide <GUIDE_FILE>] "
            "[-output <OUTPUT_DEF>] [-outputguide <GUIDE_FILE>] "
            "[-threads <N>] [-verbose <N>]\n";
}

enum class ParseResult { kRun, kHelp, kError };

ParseResult parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-lef") == 0 && i + 1 < argc) {
            LEF_FILE = argv[++i];
        } else if (strcmp(argv[i], "-def") == 0 && i + 1 < argc) {
            DEF_FILE = argv[++i];
            REF_OUT_FILE = DEF_FILE;
        } else if (strcmp(argv[i], "-guide") == 0 && i + 1 < argc) {
            GUIDE_FILE = argv[++i];
        } else if (strcmp(argv[i], "-output") == 0 && i + 1 < argc) {
            OUT_FILE = argv[++i];
        } else if (strcmp(argv[i], "-outputguide") == 0 && i + 1 < argc) {
            OUTGUIDE_FILE = argv[++i];
        } else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            MAX_THREADS = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-verbose") == 0 && i + 1 < argc) {
            VERBOSE = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-help") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            printUsage();
            return ParseResult::kHelp;
        } else {
            cout << "ERROR: Illegal command line option: " << argv[i] << endl;
            printUsage();
            return ParseResult::kError;
        }
    }
    return ParseResult::kRun;
}

void printDemoConfig() {
    cout << "Symmetry routing demo configuration:\n"
         << "  net: " << kDemoNetName << "\n"
         << "  axis: horizontal y = " << kDemoAxisY << "\n"
         << "  lef: " << LEF_FILE << "\n"
         << "  def: " << DEF_FILE << "\n"
         << "  guide: "
         << (GUIDE_FILE.empty() ? string("<run GR>") : GUIDE_FILE) << "\n"
         << "  output guide: " << OUTGUIDE_FILE << "\n"
         << "  output def: " << OUT_FILE << endl;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace std::chrono;
    auto t1 = high_resolution_clock::now();

    initDemoDefaults(argv[0]);
    auto parseResult = parseArgs(argc, argv);
    if (parseResult == ParseResult::kHelp) {
        return 0;
    }
    if (parseResult == ParseResult::kError) {
        return 2;
    }

    printDemoConfig();

    FlexRoute router;
    router.getDesign()->setSymmetryConstraint(
        frSymmetryConstraint(kDemoNetName, frSymmetryAxisEnum::Horizontal,
                             kDemoAxisY, frSymmetryReferenceSideEnum::High));
    router.main();

    auto t2 = high_resolution_clock::now();
    duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
    if (VERBOSE > 0) {
        cout << endl << "Runtime taken (hrt): " << timeSpan.count() << endl;
    }
    return 0;
}
