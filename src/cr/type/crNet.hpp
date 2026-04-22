#pragma once

#include "crBlockObject.hpp"
#include "crPin.hpp"
#include "crFig.hpp"
#include "db/obj/frBlockObject.h"
#include <memory>

namespace fr {
class frNet;
class crNet : public crBlockObject {
   public:
    crNet() : crBlockObject(), pins(), routeConnFigs(), terms(), net(nullptr) {}
    // getters
    frBlockObjectEnum typeId() const override { return crcNet; }
    const std::vector<std::unique_ptr<crPin>>& getPins() const { return pins; }
    const std::vector<std::unique_ptr<crConnFig>>& getRouteConnFigs() const { return routeConnFigs; }
    const std::set<frBlockObject*>& getTerms() const { return terms; }
    std::vector<std::unique_ptr<crPin>>& getPins() { return pins; }
    std::vector<std::unique_ptr<crConnFig>>& getRouteConnFigs() { return routeConnFigs; }
    std::set<frBlockObject*>& getTerms() { return terms; }
    frNet* getNet() { return net; }
    // setters
    void setNet(frNet* _net) { net = _net; }
    void addPin(std::unique_ptr<crPin>& _pin) {
        _pin->setNet(this);
        pins.push_back(std::move(_pin));
    }
    void addRouteConnFig(std::unique_ptr<crConnFig>& connFig) {
        connFig->addToNet(this);
        routeConnFigs.push_back(std::move(connFig));
    }
    void clearRouteConnFigs() { routeConnFigs.clear(); }
   protected:
    std::vector<std::unique_ptr<crPin>> pins;
    std::vector<std::unique_ptr<crConnFig>> routeConnFigs;
    std::set<frBlockObject*> terms;
    frNet* net;
};
}  // namespace fr
