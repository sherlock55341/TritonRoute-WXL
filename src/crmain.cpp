#include <iostream>
#include <cstdio>
#include <cr/cr.hpp>
#include <cstring>
#include <io/io.h>
#include "global.h"
#include <pa/FlexPA.h>

std::string lefFile;
std::string defFile;
std::string outputFile;

void readCommandLineParams(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-lef") == 0)
            lefFile = argv[i + 1];
        else if (strcmp(argv[i], "-def") == 0)
            defFile = argv[i + 1];
        else if (strcmp(argv[i], "-output") == 0)
            outputFile = argv[i + 1];
        else {
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl;
            exit(0);
        }
    }
    if (outputFile.empty()) {
        std::cout << "[ERROR] missing -output <file>" << std::endl;
        exit(0);
    }
    LEF_FILE = lefFile;
    DEF_FILE = defFile;
    OUT_FILE = outputFile;
    REF_OUT_FILE = OUT_FILE + ".ref";
}

int main(int argc, char** argv) {
    readCommandLineParams(argc, argv);
    auto design = std::make_unique<fr::frDesign>();
    fr::io::Parser parser(design.get());
    parser.readLefDef();
    parser.postProcess();
    parser.initDefaultVias();
    parser.writeRefDef();
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
    for (auto& net : block->getNets()) {
        if (!net->getName().empty() && net->getName()[0] == 'L') {
            std::cout << "[customdr] postRoute net " << net->getName()
                      << " shapes=" << net->getShapes().size()
                      << " vias=" << net->getVias().size() << std::endl;
        }
    }
    fr::io::Writer writer(design.get());
    writer.writeFromDR();
    if (REF_OUT_FILE != DEF_FILE) {
        remove(REF_OUT_FILE.c_str());
    }
    std::cout << "Finish Normally" << std::endl;
}
