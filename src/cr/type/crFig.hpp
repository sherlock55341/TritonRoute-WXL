#pragma once

#include "crBlockObject.hpp"

namespace fr {

// Base class for all custom-route geometry-like objects.
class crFig : public crBlockObject {
   public:
    crFig() {}
    virtual ~crFig() {}
};

class crNet;

// Base class for CR geometry that can be attached to a crNet. Implementations
// own their net relationship through a back-pointer; crNet owns the object.
class crConnFig : public crFig {
   public:
    crConnFig() : crFig() {}
    // Return whether this figure currently belongs to a crNet.
    virtual bool hasNet() const = 0;
    // Return the owning crNet back-pointer, or nullptr when unowned.
    virtual crNet* getNet() const = 0;

    // Attach this figure to a crNet. Ownership is still held by the caller's
    // unique_ptr until it is inserted into crNet.
    virtual void addToNet(crNet* _net) = 0;
};

class crPin;

// Base class for CR geometry that can also be attached to a crPin.
class crPinFig : public crConnFig {
   public:
    crPinFig() : crConnFig() {}
    // Return whether this figure currently belongs to a crPin.
    virtual bool hasPin() const = 0;
    // Return the owning crPin back-pointer, or nullptr when not pin-owned.
    virtual crPin* getPin() const = 0;
    // Attach this figure to a crPin without taking ownership.
    virtual void addToPin(crPin* in) = 0;
};

}  // namespace fr
