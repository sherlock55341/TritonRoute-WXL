#pragma once

#include "crBlockObject.hpp"
#include "crShape.hpp"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"

namespace fr {
class crPatchWire : public crShape {
   public:
    crPatchWire() : crShape(), offsetBox(), origin(), layer(), owner(nullptr) {}
    crPatchWire(const crPatchWire& other)
        : crShape(other),
          offsetBox(other.offsetBox),
          origin(other.origin),
          layer(other.layer),
          owner(other.owner) {}
    frBlockObjectEnum typeId() const override { return crcPatchWire; }

    // getters
    frLayerNum getLayerNum() const override { return layer; }
    crBlockObject* getOwner() const { return owner; }
    frBox getOffsetBox() const { return offsetBox; }
    frPoint getOrigin() const { return origin; }

    // setters
    void setLayerNum(frLayerNum _layerNum) override { layer = _layerNum; }
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    void setOffsetBox(frBox _offsetBox) { offsetBox = _offsetBox; }
    void setOrigin(frPoint _origin) { origin = _origin; }

    bool hasPin() const override { return owner && owner->typeId() == crcPin; }

    crPin* getPin() const override {
        return hasPin() ? reinterpret_cast<crPin*>(owner) : nullptr;
    }

    void addToPin(crPin* _pin) override {
        owner = reinterpret_cast<crBlockObject*>(_pin);
    }

    bool hasNet() const override { return owner && owner->typeId() == crcNet; }

    crNet* getNet() const override {
        return hasNet() ? reinterpret_cast<crNet*>(owner) : nullptr;
    }

    void addToNet(crNet* _net) override {
        owner = reinterpret_cast<crBlockObject*>(_net);
    }

   protected:
    frBox offsetBox;
    frPoint origin;
    frLayerNum layer;
    crBlockObject* owner;
};
}  // namespace fr