#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "db/tech/frTechObject.h"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"
#include "../type/crFig.hpp"
#include "../type/crMazeType.hpp"

namespace fr {

class CustomRouteWorker;
class frDesign;
class frBlockObject;
class crPathSeg;
class crVia;

class crPatternGraph {
   public:
    crPatternGraph()
        : worker(nullptr),
          xDim(0),
          yDim(0),
          zDim(0),
          planarDrcCosts(),
          viaDrcCosts() {}

    std::size_t getNumNodes() const { return getGridCapacity(); }
    const std::vector<frCoord>& getXCoords() const { return xCoords; }
    const std::vector<frCoord>& getYCoords() const { return yCoords; }
    const std::vector<frLayerNum>& getZCoords() const { return zCoords; }
    std::size_t getXDim() const { return xDim; }
    std::size_t getYDim() const { return yDim; }
    std::size_t getZDim() const { return zDim; }
    frDesign* getDesign() const;
    frTechObject* getTech() const;

    void setDims(std::size_t _xDim, std::size_t _yDim, std::size_t _zDim);
    void setCoords(std::vector<frCoord> xCoordsIn,
                   std::vector<frCoord> yCoordsIn,
                   std::vector<frLayerNum> zCoordsIn);
    void clear();
    void build(CustomRouteWorker* worker);
    bool hasMazeXCoord(frCoord xCoord) const;
    bool hasMazeYCoord(frCoord yCoord) const;
    bool hasMazeZCoord(frLayerNum layerNum) const;
    bool hasMazeIdx(frCoord xCoord, frCoord yCoord, frLayerNum layerNum) const;
    crMazeType getMazeIdx(frCoord xCoord, frCoord yCoord,
                          frLayerNum layerNum) const;
    frPoint getPoint(const crMazeType& mazeIdx) const;
    frLayerNum getLayerNum(const crMazeType& mazeIdx) const;
    bool hasNode(const crMazeType& mazeIdx) const;
    std::uint64_t getNodeKey(const crMazeType& mazeIdx) const;
    bool getNextMazeIdx(const crMazeType& curr, frDirEnum dir,
                        crMazeType& next) const;
    bool hasNonPrefCost(const crMazeType& node, frDirEnum dir) const;
    bool hasDRCCost(const crMazeType& node, frDirEnum dir) const;
    void addDRCCost(const crMazeType& node, frDirEnum dir);
    void subDRCCost(const crMazeType& node, frDirEnum dir);
    void initDRCCost();
    void addPathCost(const crConnFig* connFig);
    void subPathCost(const crConnFig* connFig);

   protected:
    template <typename T>
    bool hasCoord(const std::vector<T>& coords, T coord) const;
    template <typename T>
    crIndex_t getCoordIdx(const std::vector<T>& coords, T coord) const;
    bool isValidMazeIdx(const crMazeType& mazeIdx) const;
    std::size_t getGridCapacity() const;
    std::size_t getGridIdx(const crMazeType& mazeIdx) const;
    std::size_t getPlanarCostIdx(const crMazeType& node) const;
    std::size_t getViaCostIdx(const crMazeType& node, frDirEnum dir) const;
    std::uint64_t getMapKey(const crMazeType& mazeIdx) const;
    std::uint64_t getEdgeKey(const crMazeType& node, frDirEnum dir) const;
    std::uint64_t getCostEdgeKey(crMazeType node, frDirEnum dir) const;
    frCoord getMinSpacing(const frBox& box, frLayerNum layerNum) const;
    frBox getPlanarEdgeBox(const crMazeType& curr,
                           const crMazeType& next) const;
    bool isExternalObject(frBlockObject* obj) const;
    bool isPlanarNonPrefDir(frLayerNum layerNum, frDirEnum dir) const;
    void modMetalShapeCost(const frBox& srcBox, frLayerNum layerNum,
                           bool isAdd);
    void modViaShapeCost(const frBox& cutBox, frLayerNum lowerLayerNum,
                         bool isAdd);
    void modFrObjCost(frBlockObject* obj, bool isAdd);
    void initExternalDRCCost();
    void modPathCost(const crConnFig* connFig, bool isAdd);
    void modPathSegCost(const crPathSeg* pathSeg, bool isAdd);
    void modViaCost(const crVia* via, bool isAdd);

    CustomRouteWorker* worker;
    std::size_t xDim;
    std::size_t yDim;
    std::size_t zDim;
    std::vector<frCoord> xCoords;
    std::vector<frCoord> yCoords;
    std::vector<frLayerNum> zCoords;
    std::vector<std::uint16_t> planarDrcCosts;
    std::vector<std::uint16_t> viaDrcCosts;
};

}  // namespace fr
