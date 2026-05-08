#pragma once

#include <limits>
#include "crMazeType.hpp"
#include "crFig.hpp"
#include "crBlockObject.hpp"
#include "db/infra/frTransform.h"
#include "db/tech/frViaDef.h"
#include "crMazeType.hpp"
#include "frBaseTypes.h"

namespace fr {
class crNet;
class crPin;

// CR-local routed via. It stores the chosen viaDef, origin, and the maze
// transition endpoints so graph cost and DB writeback use the same object.
class crVia : public crPinFig {
   public:
    crVia() : crPinFig() {}
    crVia(frViaDef* _viaDef)
        : crPinFig(),
          origin(),
          viaDef(_viaDef),
          owner(nullptr),
          beginMazeIdx(),
          endMazeIdx() {}
    crVia(const crVia& via)
        : crPinFig(),
          origin(via.origin),
          viaDef(via.viaDef),
          owner(via.owner),
          beginMazeIdx(via.beginMazeIdx),
          endMazeIdx(via.endMazeIdx) {}

    // Return the via origin in physical database coordinates.
    frPoint getOrigin() const { return origin; }
    // Return the selected technology via definition; owned by the tech DB.
    frViaDef* getViaDef() const { return viaDef; }
    // Return the owning crPin/crNet as a CR base pointer.
    crBlockObject* getOwner() const { return owner; }
    // Return the lower/upper transition begin maze index used by the path.
    crMazeType getBeginMazeIdx() const { return beginMazeIdx; }
    // Return the lower/upper transition end maze index used by the path.
    crMazeType getEndMazeIdx() const { return endMazeIdx; }

    // Return the transformed bounding box of all lower-layer via figures.
    frBox getLowerLayerFigBBox() const {
        frBox box;
        auto& figs = viaDef->getLayer1Figs();
        frCoord xl = std::numeric_limits<frCoord>::max();
        frCoord yl = std::numeric_limits<frCoord>::max();
        frCoord xh = std::numeric_limits<frCoord>::min();
        frCoord yh = std::numeric_limits<frCoord>::min();
        for (auto& fig : figs) {
            frBox figBox;
            fig->getBBox(figBox);
            xl = std::min(xl, figBox.left());
            yl = std::min(yl, figBox.bottom());
            xh = std::max(xh, figBox.right());
            yh = std::max(yh, figBox.top());
        }
        box.set(xl, yl, xh, yh);
        frTransform xform;
        xform.set(origin);
        box.transform(xform);
        return box;
    }

    // Return the transformed bounding box of all cut-layer via figures.
    frBox getCutFigBBox() const {
        frBox box;
        auto& figs = viaDef->getCutFigs();
        frCoord xl = std::numeric_limits<frCoord>::max();
        frCoord yl = std::numeric_limits<frCoord>::max();
        frCoord xh = std::numeric_limits<frCoord>::min();
        frCoord yh = std::numeric_limits<frCoord>::min();
        for (auto& fig : figs) {
            frBox figBox;
            fig->getBBox(figBox);
            xl = std::min(xl, figBox.left());
            yl = std::min(yl, figBox.bottom());
            xh = std::max(xh, figBox.right());
            yh = std::max(yh, figBox.top());
        }
        box.set(xl, yl, xh, yh);
        frTransform xform;
        xform.set(origin);
        box.transform(xform);
        return box;
    }

    // Return the transformed bounding box of all upper-layer via figures.
    frBox getUpperLayerFigBBox() const {
        frBox box;
        auto& figs = viaDef->getLayer2Figs();
        frCoord xl = std::numeric_limits<frCoord>::max();
        frCoord yl = std::numeric_limits<frCoord>::max();
        frCoord xh = std::numeric_limits<frCoord>::min();
        frCoord yh = std::numeric_limits<frCoord>::min();
        for (auto& fig : figs) {
            frBox figBox;
            fig->getBBox(figBox);
            xl = std::min(xl, figBox.left());
            yl = std::min(yl, figBox.bottom());
            xh = std::max(xh, figBox.right());
            yh = std::max(yh, figBox.top());
        }
        box.set(xl, yl, xh, yh);
        frTransform xform;
        xform.set(origin);
        box.transform(xform);
        return box;
    }
    // Set the via origin.
    void setOrigin(const frPoint& _origin) { origin = _origin; }
    // Set the selected technology via definition.
    void setViaDef(frViaDef* _viaDef) { viaDef = _viaDef; }
    // Set the owning crPin/crNet pointer without transferring ownership.
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    // Set the path begin maze index for this via transition.
    void setBeginMazeIdx(const crMazeType& _beginMazeIdx) {
        beginMazeIdx = _beginMazeIdx;
    }
    // Set the path end maze index for this via transition.
    void setEndMazeIdx(const crMazeType& _endMazeIdx) {
        endMazeIdx = _endMazeIdx;
    }

    // Return whether the owner is a crPin.
    bool hasPin() const override {
        return owner && (owner->typeId() == crcPin);
    }
    // Return the owner as crPin when pin-owned.
    crPin* getPin() const override {
        return hasPin() ? reinterpret_cast<crPin*>(owner) : nullptr;
    }
    // Attach this via to a pin without taking ownership.
    void addToPin(crPin* pin) override {
        owner = reinterpret_cast<crBlockObject*>(pin);
    }

    // Return whether the owner is a crNet.
    bool hasNet() const override {
        return owner && (owner->typeId() == crcNet);
    }
    // Return the owner as crNet when net-owned.
    crNet* getNet() const override {
        return hasNet() ? reinterpret_cast<crNet*>(owner) : nullptr;
    }
    // Attach this via to a net without taking ownership.
    void addToNet(crNet* net) override {
        owner = reinterpret_cast<crBlockObject*>(net);
    }

    // Return the CR object type for vias.
    frBlockObjectEnum typeId() const override { return crcVia; }

   protected:
    // Physical via origin.
    frPoint origin;
    // Technology via definition; not owned by crVia.
    frViaDef* viaDef;
    // Non-owning crPin/crNet owner pointer.
    crBlockObject* owner;
    // Graph coordinate for one end of the vertical transition.
    crMazeType beginMazeIdx;
    // Graph coordinate for the other end of the vertical transition.
    crMazeType endMazeIdx;
};
}  // namespace fr
