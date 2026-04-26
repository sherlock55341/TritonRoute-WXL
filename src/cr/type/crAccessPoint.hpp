#pragma once

#include "crBlockObject.hpp"
#include "crMazeType.hpp"
#include "frBaseTypes.h"
#include <db/infra/frPoint.h>

namespace fr {
class crPin;
class frNet;
class frViaDef;

// CR copy of a PA access point. Coordinates are physical database coordinates;
// ownerNet/ownerTerm identify the source net terminal, and mazeIdx is filled
// only after crPatternGraph creates its coordinate arrays.
class crAccessPoint : public crBlockObject {
   public:
    crAccessPoint()
        : crBlockObject(),
          pt(),
          layerIdx(),
          mazeIdx(),
          pin(nullptr),
          ownerNet(nullptr),
          ownerTerm(nullptr),
          validAccess(6, false),
          upViaDefs(),
          downViaDefs() {}

    // Return the CR object type for access points.
    frBlockObjectEnum typeId() const override { return crcAccessPoint; }
    // Return the transformed physical access point location.
    frPoint getPt() const { return pt; }
    // Return the routing layer where this access point lies.
    frLayerNum getLayerIdx() const { return layerIdx; }
    // Return the graph index assigned after crPatternGraph setup.
    crMazeType getMazeIdx() const { return mazeIdx; }
    // Return the owning crPin back-pointer.
    crPin* getPin() const { return pin; }
    // Return the design-owned source frNet for this CR-local AP.
    frNet* getOwnerNet() const { return ownerNet; }
    // Return the source frInstTerm/frTerm that carries this AP.
    frBlockObject* getOwnerTerm() const { return ownerTerm; }
    // Return directional access flags in E/S/W/N/U/D order.
    const std::vector<bool>& getValidAccess() const { return validAccess; }
    // Return candidate upward viaDefs grouped by cut-count index 0/1.
    const std::array<std::vector<frViaDef*>, 2>& getUpViaDefs() const {
        return upViaDefs;
    }
    // Return candidate downward viaDefs grouped by cut-count index 0/1.
    const std::array<std::vector<frViaDef*>, 2>& getDownViaDefs() const {
        return downViaDefs;
    }
    // Return whether the access point permits entry in the requested direction.
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
    // Return whether at least one upward access viaDef was copied from PA.
    bool hasUpAccessViaDef() const {
        return upViaDefs[0].size() + upViaDefs[1].size() > 0;
    }
    // Return whether at least one downward access viaDef was copied from PA.
    bool hasDownAccessViaDef() const {
        return downViaDefs[0].size() + downViaDefs[1].size() > 0;
    }
    // Set the owning crPin back-pointer; crPin still owns this object.
    void setPin(crPin* in) { pin = in; }
    // Set the design-owned source frNet context; crAccessPoint does not own it.
    void setOwnerNet(frNet* in) { ownerNet = in; }
    // Set the source frInstTerm/frTerm context; crAccessPoint does not own it.
    void setOwnerTerm(frBlockObject* in) { ownerTerm = in; }
    // Set the transformed physical access point location.
    void setPt(frPoint in) { pt = in; }
    // Set the routing layer for this access point.
    void setLayerIdx(frLayerNum in) { layerIdx = in; }
    // Set the graph index assigned by crPatternGraph.
    void setMazeIdx(crMazeType in) { mazeIdx = in; }
    // Set one directional access flag in E/S/W/N/U/D order.
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
    // Add an upward viaDef for cutNum index 0/1; viaDef is tech-owned.
    void addUpViaDef(int cutNum, frViaDef* in) {
        upViaDefs[cutNum].push_back(in);
    }
    // Add a downward viaDef for cutNum index 0/1; viaDef is tech-owned.
    void addDownViaDef(int cutNum, frViaDef* in) {
        downViaDefs[cutNum].push_back(in);
    }

   protected:
    // Physical database coordinate after instance transform is applied.
    frPoint pt;
    // Database routing layer number of pt.
    frLayerNum layerIdx;
    // Graph index for pt/layerIdx; empty until graph coordinate setup.
    crMazeType mazeIdx;
    // Non-owning parent pin back-pointer.
    crPin* pin;
    // Design-owned source net, used by AP-cost logic to skip same-net APs.
    frNet* ownerNet;
    // Design-owned frInstTerm/frTerm source, used for pin-class context.
    frBlockObject* ownerTerm;
    // Directional access flags in E/S/W/N/U/D order.
    std::vector<bool> validAccess;
    // Upward viaDefs copied from PA, grouped by one-cut/two-cut index.
    std::array<std::vector<frViaDef*>, 2> upViaDefs;
    // Downward viaDefs copied from PA, grouped by one-cut/two-cut index.
    std::array<std::vector<frViaDef*>, 2> downViaDefs;
};
}  // namespace fr
