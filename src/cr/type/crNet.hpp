#pragma once

#include "crBlockObject.hpp"
#include "crPin.hpp"
#include "crFig.hpp"
#include "db/obj/frBlockObject.h"
#include <memory>

namespace fr {
class frNet;

// CR-local representation of one source frNet. It owns copied pins and local
// route geometry until CustomRouteWorker writes the final result back to frNet.
class crNet : public crBlockObject {
   public:
    crNet() : crBlockObject(), pins(), routeConnFigs(), terms(), net(nullptr) {}
    // Return the CR object type for nets.
    frBlockObjectEnum typeId() const override { return crcNet; }
    // Return CR pins owned by this net.
    const std::vector<std::unique_ptr<crPin>>& getPins() const { return pins; }
    // Return local routed geometry owned by this net.
    const std::vector<std::unique_ptr<crConnFig>>& getRouteConnFigs() const {
        return routeConnFigs;
    }
    // Return source database terms represented by pins on this net.
    const std::set<frBlockObject*>& getTerms() const { return terms; }
    // Return mutable CR pins while initializing the net.
    std::vector<std::unique_ptr<crPin>>& getPins() { return pins; }
    // Return mutable routed geometry while routing or writeback is pending.
    std::vector<std::unique_ptr<crConnFig>>& getRouteConnFigs() {
        return routeConnFigs;
    }
    // Return mutable source-term set while initializing the net.
    std::set<frBlockObject*>& getTerms() { return terms; }
    // Return the source frNet back-pointer.
    frNet* getNet() { return net; }
    // Set the source frNet back-pointer; frNet is owned by the design DB.
    void setNet(frNet* _net) { net = _net; }
    // Transfer ownership of a CR pin into this net and set its parent.
    void addPin(std::unique_ptr<crPin>& _pin) {
        _pin->setNet(this);
        pins.push_back(std::move(_pin));
    }
    // Transfer ownership of routed geometry into this net and set its parent.
    void addRouteConnFig(std::unique_ptr<crConnFig>& connFig) {
        connFig->addToNet(this);
        routeConnFigs.push_back(std::move(connFig));
    }
    // Drop all local route geometry before rerouting this CR net.
    void clearRouteConnFigs() { routeConnFigs.clear(); }

   protected:
    // CR pins owned by this net.
    std::vector<std::unique_ptr<crPin>> pins;
    // Local routing result owned by this net before final DB writeback.
    std::vector<std::unique_ptr<crConnFig>> routeConnFigs;
    // Source frInstTerm/frTerm objects covered by this CR net.
    std::set<frBlockObject*> terms;
    // Source frNet owned by the design database.
    frNet* net;
};
}  // namespace fr
