#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "db/tech/frTechObject.h"
#include "db/infra/frBox.h"
#include "frBaseTypes.h"
#include "../type/crFig.hpp"
#include "../type/crMazeType.hpp"

namespace fr {

class CustomRouteWorker;
class frDesign;
class crPathSeg;

class crPatternGraph {
   public:
    crPatternGraph()
        : worker(nullptr),
          xDim(0),
          yDim(0),
          zDim(0),
          nodes(),
          nodeMap(),
          drcEdges(),
          nonPrefEdges() {}

    const std::vector<crMazeType>& getNodes() const { return nodes; }
    std::vector<crMazeType>& getNodes() { return nodes; }
    const std::unordered_map<std::uint64_t, int>& getNodeMap() const {
        return nodeMap;
    }
    std::size_t getNumNodes() const { return nodes.size(); }
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
    int getNodeIdx(const crMazeType& mazeIdx) const;
    int addNode(const crMazeType& mazeIdx);
    void addNodes(const std::vector<crMazeType>& mazeIdxs);
    void addRoutingLayerNodes(frCoord xCoord, frCoord yCoord);
    std::uint64_t getNodeKey(const crMazeType& mazeIdx) const;
    bool getNextMazeIdx(const crMazeType& curr, frDirEnum dir,
                        crMazeType& next) const;
    bool hasNonPrefCost(const crMazeType& node, frDirEnum dir) const;
    void addNonPrefCost(const crMazeType& node, frDirEnum dir);
    void subNonPrefCost(const crMazeType& node, frDirEnum dir);
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
    std::uint64_t getMapKey(const crMazeType& mazeIdx) const;
    std::uint64_t getEdgeKey(const crMazeType& node, frDirEnum dir) const;
    frCoord getMinSpacing(const frBox& box, frLayerNum layerNum) const;
    frBox getPlanarEdgeBox(const crMazeType& curr, const crMazeType& next) const;
    bool hasShortViolation(const frBox& edgeBox, frLayerNum layerNum) const;
    bool hasSpacingViolation(const frBox& edgeBox, frLayerNum layerNum) const;
    bool isExternalObject(frBlockObject* obj) const;
    bool isPlanarNonPrefDir(frLayerNum layerNum, frDirEnum dir) const;
    void initNonPrefCost();
    void initPlanarDRCCost();
    void modPathCost(const crConnFig* connFig, bool isAdd);
    void modPathSegCost(const crPathSeg* pathSeg, bool isAdd);

    CustomRouteWorker* worker;
    std::size_t xDim;
    std::size_t yDim;
    std::size_t zDim;
    std::vector<frCoord> xCoords;
    std::vector<frCoord> yCoords;
    std::vector<frLayerNum> zCoords;
    std::vector<crMazeType> nodes;
    std::unordered_map<std::uint64_t, int> nodeMap;
    std::unordered_map<std::uint64_t, int> drcEdges;
    std::unordered_map<std::uint64_t, int> nonPrefEdges;
};

}  // namespace fr
