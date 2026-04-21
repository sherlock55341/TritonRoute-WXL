#pragma once

#include "crBlockObject.hpp"
#include "crMazeType.hpp"
#include "frBaseTypes.h"
#include <db/infra/frPoint.h>

namespace fr {
class crPin;
class frViaDef;
class crAccessPoint : public crBlockObject {
   public:
    crAccessPoint()
        : crBlockObject(),
          pt(),
          layerIdx(),
          mazeIdx(),
          pin(nullptr),
          validAccess(6, false),
          upViaDefs(),
          downViaDefs() {}
    // getters
    frBlockObjectEnum typeId() const override { return crcAccessPoint; }
    frPoint getPt() const { return pt; }
    frLayerNum getLayerIdx() const { return layerIdx; }
    crMazeType getMazeIdx() const { return mazeIdx; }
    crPin* getPin() const { return pin; }
    const std::vector<bool>& getValidAccess() const { return validAccess; }
    const std::array<std::vector<frViaDef*>, 2>& getUpViaDefs() const {
        return upViaDefs;
    }
    const std::array<std::vector<frViaDef*>, 2>& getDownViaDefs() const {
        return downViaDefs;
    }
    bool hasValidAccess(const frDirEnum& dir) const {
        switch (dir) {
            case frDirEnum::E:
                return validAccess[0];
                break;
            case frDirEnum::S:
                return validAccess[1];
                break;
            case frDirEnum::W:
                return validAccess[2];
                break;
            case frDirEnum::N:
                return validAccess[3];
                break;
            case frDirEnum::U:
                return validAccess[4];
                break;
            case frDirEnum::D:
                return validAccess[5];
                break;
            default:
                return false;
        }
    }
    bool hasUpAccessViaDef() const {
        return upViaDefs[0].size() + upViaDefs[1].size() > 0;
    }
    bool hasDownAccessViaDef() const {
        return downViaDefs[0].size() + downViaDefs[1].size() > 0;
    }
    // setters
    void setPin(crPin* in) { pin = in; }
    void setPt(frPoint in) { pt = in; }
    void setLayerIdx(frLayerNum in) { layerIdx = in; }
    void setMazeIdx(crMazeType in) { mazeIdx = in; }
    void setValidAccess(const frDirEnum& dir, bool in) {
        switch (dir) {
            case frDirEnum::E:
                validAccess[0] = in;
                break;
            case frDirEnum::S:
                validAccess[1] = in;
                break;
            case frDirEnum::W:
                validAccess[2] = in;
                break;
            case frDirEnum::N:
                validAccess[3] = in;
                break;
            case frDirEnum::U:
                validAccess[4] = in;
                break;
            case frDirEnum::D:
                validAccess[5] = in;
                break;
            default:
                break;
        }
    }
    void addUpViaDef(int cutNum, frViaDef* in) {
        upViaDefs[cutNum].push_back(in);
    }
    void addDownViaDef(int cutNum, frViaDef* in) {
        downViaDefs[cutNum].push_back(in);
    }

   protected:
    frPoint pt;
    frLayerNum layerIdx;
    crMazeType mazeIdx;
    crPin* pin;
    std::vector<bool> validAccess;
    std::array<std::vector<frViaDef*>, 2> upViaDefs;
    std::array<std::vector<frViaDef*>, 2> downViaDefs;
};
}  // namespace fr