#pragma once

#include "crBlockObject.hpp"
#include "db/obj/frBlockObject.h"
#include "crAccessPoint.hpp"

namespace fr {
class crNet;

// CR pin wrapper for one frTerm or frInstTerm. It owns all access points copied
// from PA and keeps a back-pointer to the CR net that owns the pin.
class crPin : public crBlockObject {
   public:
    crPin() : crBlockObject(), term(nullptr), accessPoints(), net(nullptr) {}

    // Return the source frTerm/frInstTerm represented by this CR pin.
    frBlockObject* getFrTerm() const { return term; }
    // Return owned access points copied from PA.
    const std::vector<std::unique_ptr<crAccessPoint> >& getAccessPoints()
        const {
        return accessPoints;
    }
    // Return mutable access points while building or annotating the pin.
    std::vector<std::unique_ptr<crAccessPoint> >& getAccessPoints() {
        return accessPoints;
    }
    // Return the owning crNet back-pointer.
    crNet* getNet() const { return net; }
    // Set the source frTerm/frInstTerm back-pointer.
    void setFrTerm(frBlockObject* in) { term = in; }
    // Transfer ownership of an access point into this pin and set its parent.
    void addAccessPoint(std::unique_ptr<crAccessPoint>& in) {
        in->setPin(this);
        accessPoints.push_back(std::move(in));
    }
    // Set the owning crNet back-pointer; crNet still owns this pin.
    void setNet(crNet* in) { net = in; }

    // Return the CR object type for pins.
    frBlockObjectEnum typeId() const override { return crcPin; }

   protected:
    // Source frInstTerm/frTerm; owned by the main design database.
    frBlockObject* term;
    // Access points owned by this CR pin.
    std::vector<std::unique_ptr<crAccessPoint> > accessPoints;
    // Non-owning parent net back-pointer.
    crNet* net;
};
}  // namespace fr
