#pragma once

#include "crBlockObject.hpp"
#include "crShape.hpp"
#include "crMazeType.hpp"
#include "db/infra/frSegStyle.h"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"

namespace fr {

// CR-local routed path segment. The object stores both physical endpoints and
// maze endpoints so graph cost removal, writeback, and debug printing can refer
// to the same segment.
class crPathSeg : public crShape {
   public:
    crPathSeg()
        : crShape(),
          begin(),
          end(),
          layerIdx(),
          style(),
          owner(nullptr),
          beginMazeIdx(),
          endMazeIdx(),
          isPatch(false) {}
    crPathSeg(const crPathSeg& other)
        : crShape(other),
          begin(other.begin),
          end(other.end),
          layerIdx(other.layerIdx),
          style(other.style),
          owner(other.owner),
          beginMazeIdx(other.beginMazeIdx),
          endMazeIdx(other.endMazeIdx),
          isPatch(other.isPatch) {}

    // Return the CR object type for path segments.
    frBlockObjectEnum typeId() const override { return crcPathSeg; }
    // Return the physical begin point in database coordinates.
    frPoint getBegin() const { return begin; }
    // Return the physical end point in database coordinates.
    frPoint getEnd() const { return end; }
    // Return the routing layer number.
    frLayerNum getLayerNum() const override { return layerIdx; }
    // Return width and end-style information used for DB writeback.
    frSegStyle getStyle() const { return style; }
    // Return the owning crPin/crNet as a CR base pointer.
    crBlockObject* getOwner() const { return owner; }
    // Return the maze index corresponding to begin.
    crMazeType getBeginMazeIdx() const { return beginMazeIdx; }
    // Return the maze index corresponding to end.
    crMazeType getEndMazeIdx() const { return endMazeIdx; }
    // Return whether this segment represents patch metal rather than routing.
    bool getIsPatch() const { return isPatch; }
    // Set the physical begin point.
    void setBegin(frPoint _begin) { begin = _begin; }
    // Set the physical end point.
    void setEnd(frPoint _end) { end = _end; }
    // Set the routing layer number.
    void setLayerNum(frLayerNum _layerNum) override { layerIdx = _layerNum; }
    // Set width and end-style information for this segment.
    void setStyle(frSegStyle _style) { style = _style; }
    // Set the owning crPin/crNet pointer without transferring ownership.
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    // Set the maze index corresponding to begin.
    void setBeginMazeIdx(crMazeType _beginMazeIdx) {
        beginMazeIdx = _beginMazeIdx;
    }
    // Set the maze index corresponding to end.
    void setEndMazeIdx(crMazeType _endMazeIdx) { endMazeIdx = _endMazeIdx; }
    // Mark whether this shape is patch metal.
    void setIsPatch(bool _isPatch) { isPatch = _isPatch; }

    // Return whether the owner is a crPin.
    bool hasPin() const override {
        return owner && (owner->typeId() == crcPin);
    }
    // Return the owner as crPin when pin-owned.
    crPin* getPin() const override {
        return hasPin() ? reinterpret_cast<crPin*>(owner) : nullptr;
    }
    // Attach this segment to a pin without taking ownership.
    void addToPin(crPin* _pin) override {
        owner = reinterpret_cast<crBlockObject*>(_pin);
    }

    // Return whether the owner is a crNet.
    bool hasNet() const override { return owner && owner->typeId() == crcNet; }
    // Return the owner as crNet when net-owned.
    crNet* getNet() const override {
        return hasNet() ? reinterpret_cast<crNet*>(owner) : nullptr;
    }
    // Attach this segment to a net without taking ownership.
    void addToNet(crNet* _net) override {
        owner = reinterpret_cast<crBlockObject*>(_net);
    }

    // Return the physical metal box implied by endpoints, width, and end style.
    frBox getBBox() const {
        bool isH = (begin.y() == end.y());
        auto width = style.getWidth();
        auto beginExt = style.getBeginExt();
        auto endExt = style.getEndExt();
        frBox box;
        if (isH) {
            box.set(begin.x() - beginExt, begin.y() - width / 2,
                    end.x() + endExt, end.y() + width / 2);
        } else {
            box.set(begin.x() - width / 2, begin.y() - beginExt,
                    end.x() + width / 2, end.y() + endExt);
        }
        return box;
    }

    // Return whether begin/end maze indices have been assigned.
    bool hasMazeIdx() const { return beginMazeIdx.empty() == false; }

   protected:
    // Physical begin point.
    frPoint begin;
    // Physical end point.
    frPoint end;
    // Database routing layer number.
    frLayerNum layerIdx;
    // Segment width and end extension style.
    frSegStyle style;
    // Non-owning crPin/crNet owner pointer.
    crBlockObject* owner;
    // Graph coordinate for begin.
    crMazeType beginMazeIdx;
    // Graph coordinate for end.
    crMazeType endMazeIdx;
    // True when this object is intended as patch metal.
    bool isPatch;
};
}  // namespace fr
