#include <iostream>
#include <cstdio>
#include <cr/cr.hpp>
#include <cstring>
#include <io/io.h>
#include "global.h"
#include <pa/FlexPA.h>

std::string lefFile;
std::string defFile;

void readCommandLineParams(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-lef") == 0)
            lefFile = argv[i + 1];
        else if (strcmp(argv[i], "-def") == 0)
            defFile = argv[i + 1];
        else {
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl;
            exit(0);
        }
    }
    LEF_FILE = lefFile;
    DEF_FILE = defFile;
}

int main(int argc, char** argv) {
    readCommandLineParams(argc, argv);
    auto design = std::make_unique<fr::frDesign>();
    fr::io::Parser parser(design.get());
    parser.readLefDef();
    parser.postProcess();
    auto block = design->getTopBlock();
    fr::FlexPA pa(design.get());
    pa.main();
    fr::CustomRoute cr(design.get());
    for (auto& net : block->getNets()) {
        if (net->getName()[0] == 'L') {
            cr.addTask(std::make_pair(net.get(), fr::crPatternEnum::L));
        }
    }
    cr.run();
    std::cout << "Finish Normally" << std::endl;
}