#include "crWorker.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include "cr/crConfig.hpp"
#include "global.h"
#include "db/obj/frAccess.h"
#include "db/obj/frInst.h"
#include "db/obj/frInstTerm.h"
#include "db/obj/frNet.h"
#include "db/obj/frPin.h"
#include "db/obj/frTerm.h"

namespace fr {

void CustomRouteWorker::initPatternGraph() {
    if (!patternGraph) {
        return;
    }

    std::vector<frCoord> xCoords;
    std::vector<frCoord> yCoords;
    std::vector<crPatternPoint> points;
    std::vector<frLayerNum> zCoords;

    for (auto& layer : design->getTech()->getLayers()) {
        if (layer->getType() == frLayerTypeEnum::ROUTING) {
            zCoords.push_back(layer->getLayerNum());
        }
    }

    for (std::size_t i = 0; i < nets.size() && i < policies.size(); ++i) {
        auto cNet = nets[i].get();
        switch (policies[i]) {
            case crPatternEnum::L:
                collectPatternGraphCoordsL(cNet, xCoords, yCoords);
                break;
            default:
                break;
        }
    }

    xCoords.push_back(routeBox.left());
    xCoords.push_back(routeBox.right());
    xCoords.push_back(extBox.left());
    xCoords.push_back(extBox.right());
    yCoords.push_back(routeBox.bottom());
    yCoords.push_back(routeBox.top());
    yCoords.push_back(extBox.bottom());
    yCoords.push_back(extBox.top());

    auto uniqueCoords = [](auto& coords) {
        std::sort(coords.begin(), coords.end());
        coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    };
    uniqueCoords(xCoords);
    uniqueCoords(yCoords);
    uniqueCoords(zCoords);

    patternGraph->setCoords(std::move(xCoords), std::move(yCoords),
                            std::move(zCoords));

    for (std::size_t i = 0; i < nets.size() && i < policies.size(); ++i) {
        auto cNet = nets[i].get();
        switch (policies[i]) {
            case crPatternEnum::L:
                initPatternGraphL(cNet, patternGraph->getXCoords(),
                                  patternGraph->getYCoords(), points);
                break;
            default:
                break;
        }
    }

    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());

    for (auto& point : points) {
        patternGraph->addRoutingLayerNodes(point.x, point.y);
    }
}

void CustomRouteWorker::collectPatternGraphCoordsL(
    crNet* cNet, std::vector<frCoord>& xCoords, std::vector<frCoord>& yCoords)
    const {
    if (!cNet || cNet->getPins().size() != 2) {
        return;
    }

    const auto& pin0Aps = cNet->getPins()[0]->getAccessPoints();
    const auto& pin1Aps = cNet->getPins()[1]->getAccessPoints();

    for (auto& ap : pin0Aps) {
        auto pt = ap->getPt();
        xCoords.push_back(pt.x());
        yCoords.push_back(pt.y());
    }
    for (auto& ap : pin1Aps) {
        auto pt = ap->getPt();
        xCoords.push_back(pt.x());
        yCoords.push_back(pt.y());
    }
}

void CustomRouteWorker::initPatternGraphL(
    crNet* cNet, const std::vector<frCoord>& xCoords,
    const std::vector<frCoord>& yCoords, std::vector<crPatternPoint>& points)
    const {
    if (!cNet || cNet->getPins().size() != 2) {
        return;
    }

    const auto& pin0Aps = cNet->getPins()[0]->getAccessPoints();
    const auto& pin1Aps = cNet->getPins()[1]->getAccessPoints();

    std::vector<frCoord> pin0Xs;
    std::vector<frCoord> pin0Ys;
    std::vector<frCoord> pin1Xs;
    std::vector<frCoord> pin1Ys;

    for (auto& ap : pin0Aps) {
        auto pt = ap->getPt();
        pin0Xs.push_back(pt.x());
        pin0Ys.push_back(pt.y());
    }
    for (auto& ap : pin1Aps) {
        auto pt = ap->getPt();
        pin1Xs.push_back(pt.x());
        pin1Ys.push_back(pt.y());
    }

    auto uniqueCoords = [](auto& coords) {
        std::sort(coords.begin(), coords.end());
        coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    };
    uniqueCoords(pin0Xs);
    uniqueCoords(pin0Ys);
    uniqueCoords(pin1Xs);
    uniqueCoords(pin1Ys);

    auto addPoint = [&points](frCoord xCoord, frCoord yCoord) {
        points.push_back({xCoord, yCoord});
    };
    auto addHorizontalSegment = [&](frCoord yCoord, frCoord xCoord1,
                                   frCoord xCoord2) {
        auto lo = std::min(xCoord1, xCoord2);
        auto hi = std::max(xCoord1, xCoord2);
        auto begin = std::lower_bound(xCoords.begin(), xCoords.end(), lo);
        auto end = std::upper_bound(xCoords.begin(), xCoords.end(), hi);
        for (auto it = begin; it != end; ++it) {
            addPoint(*it, yCoord);
        }
    };
    auto addVerticalSegment = [&](frCoord xCoord, frCoord yCoord1,
                                  frCoord yCoord2) {
        auto lo = std::min(yCoord1, yCoord2);
        auto hi = std::max(yCoord1, yCoord2);
        auto begin = std::lower_bound(yCoords.begin(), yCoords.end(), lo);
        auto end = std::upper_bound(yCoords.begin(), yCoords.end(), hi);
        for (auto it = begin; it != end; ++it) {
            addPoint(xCoord, *it);
        }
    };

    for (auto x0 : pin0Xs) {
        for (auto y1 : pin1Ys) {
            for (auto y0 : pin0Ys) {
                addVerticalSegment(x0, y0, y1);
            }
            for (auto x1 : pin1Xs) {
                addHorizontalSegment(y1, x0, x1);
            }
        }
    }

    for (auto x1 : pin1Xs) {
        for (auto y0 : pin0Ys) {
            for (auto y1 : pin1Ys) {
                addVerticalSegment(x1, y1, y0);
            }
            for (auto x0 : pin0Xs) {
                addHorizontalSegment(y0, x1, x0);
            }
        }
    }
}

void CustomRouteWorker::initNet(frNet* _net) {
    auto net = std::make_unique<crNet>();
    net->setNet(_net);
    for (auto instTerm : _net->getInstTerms()) {
        initNetTerm(net.get(), instTerm);
    }
    for (auto term : _net->getTerms()) {
        initNetTerm(net.get(), term);
    }
    nets.push_back(std::move(net));
}

void CustomRouteWorker::initNetTerm(crNet* cNet, frBlockObject* term) {
    auto cPin = std::make_unique<crPin>();
    cPin->setFrTerm(term);
    cNet->getTerms().insert(term);

    frTerm* trueTerm = nullptr;
    frInst* inst = nullptr;
    frTransform shiftXform;

    if (term->typeId() == frcInstTerm) {
        auto instTerm = static_cast<frInstTerm*>(term);
        inst = instTerm->getInst();
        inst->getTransform(shiftXform);
        shiftXform.set(frOrient(frcR0));
        trueTerm = instTerm->getTerm();
    } else if (term->typeId() == frcTerm) {
        trueTerm = static_cast<frTerm*>(term);
    }

    int pinAccessIdx = inst ? inst->getPinAccessIdx() : -1;

    for (auto& pin : trueTerm->getPins()) {
        if (!pin->hasPinAccess()) {
            continue;
        }
        if (pinAccessIdx == -1) {
            continue;
        }
        for (auto& ap : pin->getPinAccess(pinAccessIdx)->getAccessPoints()) {
            frPoint bp;
            ap->getPoint(bp);
            bp.transform(shiftXform);
            auto layerNum = ap->getLayerNum();

            auto cAp = std::make_unique<crAccessPoint>();
            cAp->setPt(bp);
            cAp->setLayerIdx(layerNum);

            for (auto dir : {frDirEnum::E, frDirEnum::S, frDirEnum::W,
                             frDirEnum::N, frDirEnum::U, frDirEnum::D}) {
                cAp->setValidAccess(dir, ap->hasAccess(dir));
            }

            if (ap->hasAccess(frDirEnum::U)) {
                for (int cutNum = 0; cutNum < 2; cutNum++) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addUpViaDef(cutNum, viaDef);
                        }
                    }
                }
            }
            if (ap->hasAccess(frDirEnum::D)) {
                for (int cutNum = 0; cutNum < 2; cutNum++) {
                    if (ap->hasViaDef(cutNum + 1)) {
                        for (auto viaDef : ap->getViaDefs(cutNum + 1)) {
                            cAp->addDownViaDef(cutNum, viaDef);
                        }
                    }
                }
            }

            cPin->addAccessPoint(cAp);
        }
    }

    cNet->addPin(cPin);
}

void CustomRouteWorker::initRouteBox() {
    frCoord xl = std::numeric_limits<frCoord>::max();
    frCoord yl = std::numeric_limits<frCoord>::max();
    frCoord xh = std::numeric_limits<frCoord>::min();
    frCoord yh = std::numeric_limits<frCoord>::min();
    frCoord maxPitch = 0;

    for (auto& net : nets) {
        for (auto& pin : net->getPins()) {
            for (auto& ap : pin->getAccessPoints()) {
                auto pt = ap->getPt();
                auto lNum = ap->getLayerIdx();
                xl = std::min(xl, pt.x());
                yl = std::min(yl, pt.y());
                xh = std::max(xh, pt.x());
                yh = std::max(yh, pt.y());
                maxPitch = std::max(
                    maxPitch,
                    static_cast<frCoord>(
                        design->getTech()->getLayer(lNum)->getPitch()));
                if (lNum + 2 <= design->getTech()->getTopLayerNum()) {
                    maxPitch = std::max(
                        maxPitch,
                        static_cast<frCoord>(
                            design->getTech()->getLayer(lNum + 2)->getPitch()));
                } else if (lNum - 2 >= design->getTech()->getBottomLayerNum()) {
                    maxPitch = std::max(
                        maxPitch,
                        static_cast<frCoord>(
                            design->getTech()->getLayer(lNum - 2)->getPitch()));
                }
            }
        }
    }

    if (xl == std::numeric_limits<frCoord>::max()) {
        std::cout << "ERROR " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(0);
    }

    frBox apBBox;
    apBBox.set(xl, yl, xh, yh);
    auto routeMargin = maxPitch * CR_ROUTE_BOX_MARGIN_PITCHES;
    apBBox.bloat(routeMargin, routeBox);
    routeBox.bloat(MTSAFEDIST, extBox);
}

void CustomRouteWorker::initTrackCoordsPin(
    crNet* cNet,
    std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
    std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap) {
    for (auto& pin : cNet->getPins()) {
        for (auto& ap : pin->getAccessPoints()) {
            auto pt = ap->getPt();
            auto lNum = ap->getLayerIdx();
            frLayerNum lNum2 = 0;
            if (lNum + 2 <= design->getTech()->getTopLayerNum()) {
                lNum2 = lNum + 2;
            } else if (lNum - 2 >= design->getTech()->getBottomLayerNum()) {
                lNum2 = lNum - 2;
            } else {
                std::cout << "Error: initTrackCoords cannot add non-pref track"
                          << std::endl;
                continue;
            }

            if (design->getTech()->getLayer(lNum)->getDir() ==
                frcHorzPrefRoutingDir) {
                yMap[pt.y()][lNum] = nullptr;
            } else {
                xMap[pt.x()][lNum] = nullptr;
            }
            if (design->getTech()->getLayer(lNum2)->getDir() ==
                frcHorzPrefRoutingDir) {
                yMap[pt.y()][lNum2] = nullptr;
            } else {
                xMap[pt.x()][lNum2] = nullptr;
            }
        }
    }
}

void CustomRouteWorker::initTrackCoords(
    std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& xMap,
    std::map<frCoord, std::map<frLayerNum, frTrackPattern*> >& yMap) {
    yMap[routeBox.bottom()][-10] = nullptr;
    yMap[routeBox.top()][-10] = nullptr;
    yMap[extBox.bottom()][-10] = nullptr;
    yMap[extBox.top()][-10] = nullptr;
    xMap[routeBox.left()][-10] = nullptr;
    xMap[routeBox.right()][-10] = nullptr;
    xMap[extBox.left()][-10] = nullptr;
    xMap[extBox.right()][-10] = nullptr;

    for (auto& net : nets) {
        initTrackCoordsPin(net.get(), xMap, yMap);
    }
}

}  // namespace fr
