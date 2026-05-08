#pragma once

#include "crBlockObject.hpp"
#include "crShape.hpp"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"

namespace fr {

// CR-local patch wire placeholder. Patch-wire writeback is not implemented yet,
// but the type mirrors frPatchWire ownership and layer/origin fields.
class crPatchWire : public crShape {
   public:
    crPatchWire() : crShape(), offsetBox(), origin(), layer(), owner(nullptr) {}
    crPatchWire(const crPatchWire& other)
        : crShape(other),
          offsetBox(other.offsetBox),
          origin(other.origin),
          layer(other.layer),
          owner(other.owner) {}
    // Return the CR object type for patch wires.
    frBlockObjectEnum typeId() const override { return crcPatchWire; }

    // Return the routing layer for this patch wire.
    frLayerNum getLayerNum() const override { return layer; }
    // Return the owning crPin/crNet as a CR base pointer.
    crBlockObject* getOwner() const { return owner; }
    // Return the patch box relative to origin.
    frBox getOffsetBox() const { return offsetBox; }
    // Return the physical origin used to place offsetBox.
    frPoint getOrigin() const { return origin; }

    // Set the routing layer.
    void setLayerNum(frLayerNum _layerNum) override { layer = _layerNum; }
    // Set the owning crPin/crNet pointer without transferring ownership.
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    // Set the patch box relative to origin.
    void setOffsetBox(frBox _offsetBox) { offsetBox = _offsetBox; }
    // Set the physical origin used to place offsetBox.
    void setOrigin(frPoint _origin) { origin = _origin; }

    // Return whether the owner is a crPin.
    bool hasPin() const override { return owner && owner->typeId() == crcPin; }

    // Return the owner as crPin when pin-owned.
    crPin* getPin() const override {
        return hasPin() ? reinterpret_cast<crPin*>(owner) : nullptr;
    }

    // Attach this patch wire to a pin without taking ownership.
    void addToPin(crPin* _pin) override {
        owner = reinterpret_cast<crBlockObject*>(_pin);
    }

    // Return whether the owner is a crNet.
    bool hasNet() const override { return owner && owner->typeId() == crcNet; }

    // Return the owner as crNet when net-owned.
    crNet* getNet() const override {
        return hasNet() ? reinterpret_cast<crNet*>(owner) : nullptr;
    }

    // Attach this patch wire to a net without taking ownership.
    void addToNet(crNet* _net) override {
        owner = reinterpret_cast<crBlockObject*>(_net);
    }

   protected:
    // Box relative to origin, matching frPatchWire's offset representation.
    frBox offsetBox;
    // Physical origin used to place offsetBox.
    frPoint origin;
    // Database routing layer number.
    frLayerNum layer;
    // Non-owning crPin/crNet owner pointer.
    crBlockObject* owner;
};
}  // namespace fr
