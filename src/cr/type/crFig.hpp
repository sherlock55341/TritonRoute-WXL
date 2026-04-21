#pragma once

#include "crBlockObject.hpp"

namespace fr {
class crFig : public crBlockObject {
   public:
    // constructors
    crFig() {}
    virtual ~crFig() {}
    // getters
    // setters
    // others
};

class crNet;
class crConnFig : public crFig {
   public:
    crConnFig() : crFig() {}
    virtual bool hasNet() const = 0;
    virtual crNet* getNet() const = 0;

    virtual void addToNet(crNet* _net) = 0;
};

class crPin;
class crPinFig : public crConnFig {
   public:
    crPinFig() : crConnFig() {}
    virtual bool hasPin() const = 0;
    virtual crPin* getPin() const = 0;
    virtual void addToPin(crPin* in) = 0;
};

}  // namespace fr