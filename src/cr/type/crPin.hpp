#pragma once

#include "crBlockObject.hpp"
#include "db/obj/frBlockObject.h"
#include "crAccessPoint.hpp"

namespace fr {
class crNet;
class crPin : public crBlockObject {
   public:
    // constructors
    crPin() : crBlockObject(), term(nullptr), accessPoints(), net(nullptr) {}
    // getters
    frBlockObject* getFrTerm() const { return term; }
    const std::vector<std::unique_ptr<crAccessPoint> >& getAccessPoints()
        const {
        return accessPoints;
    }
    std::vector<std::unique_ptr<crAccessPoint> >& getAccessPoints() {
        return accessPoints;
    }
    crNet* getNet() const { return net; }
    // setters
    void setFrTerm(frBlockObject* in) { term = in; }
    void addAccessPoint(std::unique_ptr<crAccessPoint>& in) {
        in->setPin(this);
        accessPoints.push_back(std::move(in));
    }
    void setNet(crNet* in) { net = in; }

    frBlockObjectEnum typeId() const override { return crcPin; }

   protected:
    frBlockObject* term;
    std::vector<std::unique_ptr<crAccessPoint> > accessPoints;
    crNet* net;
};
}  // namespace fr