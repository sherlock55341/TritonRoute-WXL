#pragma once

#include "crBlockObject.hpp"
#include "crShape.hpp"
#include "crMazeType.hpp"
#include "db/infra/frSegStyle.h"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"

namespace fr {
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

    frBlockObjectEnum typeId() const override { return crcPathSeg; }
    // getters
    frPoint getBegin() const { return begin; }
    frPoint getEnd() const { return end; }
    frLayerNum getLayerNum() const override { return layerIdx; }
    frSegStyle getStyle() const { return style; }
    crBlockObject* getOwner() const { return owner; }
    crMazeType getBeginMazeIdx() const { return beginMazeIdx; }
    crMazeType getEndMazeIdx() const { return endMazeIdx; }
    bool getIsPatch() const { return isPatch; }
    // setters
    void setBegin(frPoint _begin) { begin = _begin; }
    void setEnd(frPoint _end) { end = _end; }
    void setLayerNum(frLayerNum _layerNum) override { layerIdx = _layerNum; }
    void setStyle(frSegStyle _style) { style = _style; }
    void setOwner(crBlockObject* _owner) { owner = _owner; }
    void setBeginMazeIdx(crMazeType _beginMazeIdx) {
        beginMazeIdx = _beginMazeIdx;
    }
    void setEndMazeIdx(crMazeType _endMazeIdx) { endMazeIdx = _endMazeIdx; }
    void setIsPatch(bool _isPatch) { isPatch = _isPatch; }

    bool hasPin() const override {
        return owner && (owner->typeId() == crcPin);
    }
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

    bool hasMazeIdx() const { return beginMazeIdx.empty() == false; }

   protected:
    frPoint begin;
    frPoint end;
    frLayerNum layerIdx;
    frSegStyle style;
    crBlockObject* owner;
    crMazeType beginMazeIdx;
    crMazeType endMazeIdx;
    bool isPatch;
};
}  // namespace fr