#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "../type/crMazeType.hpp"

namespace fr {

class CustomRouteWorker;

class crPatternGraph {
   public:
    crPatternGraph()
        : worker(nullptr),
          xDim(0),
          yDim(0),
          zDim(0),
          nodes(),
          nodeMap() {}

    const std::vector<crMazeType>& getNodes() const { return nodes; }
    std::vector<crMazeType>& getNodes() { return nodes; }
    const std::unordered_map<std::uint64_t, int>& getNodeMap() const {
        return nodeMap;
    }
    std::size_t getXDim() const { return xDim; }
    std::size_t getYDim() const { return yDim; }
    std::size_t getZDim() const { return zDim; }

    void setDims(std::size_t _xDim, std::size_t _yDim, std::size_t _zDim);
    void clear();
    void build(CustomRouteWorker* worker);
    bool hasNode(const crMazeType& mazeIdx) const;
    int getNodeIdx(const crMazeType& mazeIdx) const;
    int addNode(const crMazeType& mazeIdx);
    void addNodes(const std::vector<crMazeType>& mazeIdxs);

   protected:
    std::uint64_t getMapKey(const crMazeType& mazeIdx) const;

    CustomRouteWorker* worker;
    std::size_t xDim;
    std::size_t yDim;
    std::size_t zDim;
    std::vector<crMazeType> nodes;
    std::unordered_map<std::uint64_t, int> nodeMap;
};

}  // namespace fr
