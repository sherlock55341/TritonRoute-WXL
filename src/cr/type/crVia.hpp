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
    // getters
    frPoint getOrigin() const { return origin; }
    frViaDef* getViaDef() const { return viaDef; }
    crBlockObject* getOwner() const { return owner; }
    crMazeType getBeginMazeIdx() const { return beginMazeIdx; }
    crMazeType getEndMazeIdx() const { return endMazeIdx; }

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
    // setters
    void setViaDef(frViaDef* _viaDef) { viaDef = _viaDef; }
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    void setBeginMazeIdx(const crMazeType& _beginMazeIdx) {
        beginMazeIdx = _beginMazeIdx;
    }
    void setEndMazeIdx(const crMazeType& _endMazeIdx) {
        endMazeIdx = _endMazeIdx;
    }

    bool hasPin() const override {
        return owner && (owner->typeId() == crcPin);
    }
    crPin* getPin() const override {
        return hasPin() ? reinterpret_cast<crPin*>(owner) : nullptr;
    }
    void addToPin(crPin* pin) override { owner = reinterpret_cast<crBlockObject*>(pin); }

    bool hasNet() const override {
        return owner && (owner->typeId() == crcNet);
    }
    crNet* getNet() const override {
        return hasNet() ? reinterpret_cast<crNet*>(owner) : nullptr;
    }
    void addToNet(crNet* net) override { owner = reinterpret_cast<crBlockObject*>(net); }

    frBlockObjectEnum typeId() const override { return crcVia; }

   protected:
    frPoint origin;
    frViaDef* viaDef;
    crBlockObject* owner;
    crMazeType beginMazeIdx;
    crMazeType endMazeIdx;
};
}  // namespace fr