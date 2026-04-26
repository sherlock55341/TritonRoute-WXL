#include "cr.hpp"
#include <iterator>
#include "worker/crWorker.hpp"
#include "db/obj/frAccess.h"
#include "db/obj/frBlock.h"
#include "db/obj/frInst.h"
#include "db/obj/frInstTerm.h"
#include "db/obj/frNet.h"
#include "db/obj/frPin.h"
#include "db/obj/frTerm.h"

namespace fr {

namespace {

box_t getBoostBox(const frBox& box) {
    return box_t(point_t(box.left(), box.bottom()),
                 point_t(box.right(), box.top()));
}

void copyAccessPointToQuery(frNet* net, frBlockObject* term, frInst* inst,
                            frTerm* trueTerm, crAPRegionQuery* query) {
    if (!net || !term || !trueTerm || !query) {
        return;
    }

    // Flow:
    // 1. Use the same shift-only AP placement convention as DR/CR net init.
    // 2. Copy PA access flags and viaDefs into CR-local AP objects.
    // 3. Let crAPRegionQuery own and index each AP copy by point/layer.
    frTransform shiftXform;
    int pinAccessIdx = 0;
    if (inst) {
        inst->getTransform(shiftXform);
        shiftXform.set(frOrient(frcR0));
        pinAccessIdx = inst->getPinAccessIdx();
    }

    for (auto& pin : trueTerm->getPins()) {
        if (!pin->hasPinAccess() || pinAccessIdx < 0 ||
            pinAccessIdx >= pin->getNumPinAccess()) {
            continue;
        }

        for (auto& ap : pin->getPinAccess(pinAccessIdx)->getAccessPoints()) {
            frPoint pt;
            ap->getPoint(pt);
            pt.transform(shiftXform);

            auto cAp = std::make_unique<crAccessPoint>();
            cAp->setOwnerNet(net);
            cAp->setOwnerTerm(term);
            cAp->setPt(pt);
            cAp->setLayerIdx(ap->getLayerNum());

            for (auto dir : {frDirEnum::E, frDirEnum::S, frDirEnum::W,
                             frDirEnum::N, frDirEnum::U, frDirEnum::D}) {
                cAp->setValidAccess(dir, ap->hasAccess(dir));
            }

            if (ap->hasAccess(frDirEnum::U)) {
                for (int cutNum = 0; cutNum < 2; ++cutNum) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addUpViaDef(cutNum, viaDef);
                        }
                    }
                }
            }
            if (ap->hasAccess(frDirEnum::D)) {
                for (int cutNum = 0; cutNum < 2; ++cutNum) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addDownViaDef(cutNum, viaDef);
                        }
                    }
                }
            }

            query->addAccessPoint(cAp);
        }
    }
}

}  // namespace

void crAPRegionQuery::clear() {
    initialized = false;
    accessPoints.clear();
    aps.clear();
}

void crAPRegionQuery::addAccessPoint(std::unique_ptr<crAccessPoint>& ap) {
    if (!ap) {
        return;
    }

    auto layerNum = ap->getLayerIdx();
    if (layerNum < 0 || static_cast<std::size_t>(layerNum) >= aps.size()) {
        return;
    }

    auto* apPtr = ap.get();
    auto pt = apPtr->getPt();
    aps[layerNum].insert({getBoostBox(frBox(pt, pt)), apPtr});
    accessPoints.push_back(std::move(ap));
}

void crAPRegionQuery::init(frDesign* design) {
    clear();
    if (!design || !design->getTopBlock() || !design->getTech()) {
        initialized = true;
        return;
    }

    // Flow:
    // 1. Allocate one AP R-tree per technology layer number.
    // 2. Traverse design nets once and copy their PA access points.
    // 3. Store only point/layer spatial membership; AP cost interpretation is
    //    intentionally left to later consumers.
    aps.resize(
        static_cast<std::size_t>(design->getTech()->getTopLayerNum() + 1));
    for (auto& net : design->getTopBlock()->getNets()) {
        if (!net) {
            continue;
        }
        for (auto* instTerm : net->getInstTerms()) {
            if (!instTerm) {
                continue;
            }
            copyAccessPointToQuery(net.get(), instTerm, instTerm->getInst(),
                                   instTerm->getTerm(), this);
        }
        for (auto* term : net->getTerms()) {
            copyAccessPointToQuery(net.get(), term, nullptr, term, this);
        }
    }
    initialized = true;
}

void crAPRegionQuery::query(const frBox& box, frLayerNum layerNum,
                            std::vector<crAccessPoint*>& result) const {
    if (layerNum < 0 || static_cast<std::size_t>(layerNum) >= aps.size()) {
        return;
    }

    std::vector<std::pair<box_t, crAccessPoint*>> temp;
    aps[layerNum].query(bgi::intersects(getBoostBox(box)),
                        std::back_inserter(temp));
    for (auto& value : temp) {
        auto* ap = value.second;
        if (ap) {
            result.push_back(ap);
        }
    }
}

crAPRegionQuery* CustomRoute::getAPRegionQuery() {
    if (!apRegionQuery) {
        apRegionQuery = std::make_unique<crAPRegionQuery>();
    }
    if (!apRegionQuery->isInitialized()) {
        apRegionQuery->init(design);
    }
    return apRegionQuery.get();
}

void CustomRoute::run() {
    // Flow:
    // 1. Build one worker for the current task batch.
    // 2. Print dense graph size for current debug visibility.
    // 3. Delegate routing and DB writeback to the worker.
    std::vector<std::unique_ptr<CustomRouteWorker>> workers;
    if (!routeTasks.empty()) {
        workers.push_back(
            std::make_unique<CustomRouteWorker>(this, routeTasks));
    }
    std::cout << workers.front()->getPatternGraph()->getXCoords().size() *
                     workers.front()->getPatternGraph()->getYCoords().size() *
                     workers.front()->getPatternGraph()->getZCoords().size()
              << std::endl;
    workers.front()->route();
}
}  // namespace fr
