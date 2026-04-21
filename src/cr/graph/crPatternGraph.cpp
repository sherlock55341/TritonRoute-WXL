#include "crPatternGraph.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace fr {

void crPatternGraph::setDims(std::size_t _xDim, std::size_t _yDim,
                             std::size_t _zDim) {
    xDim = _xDim;
    yDim = _yDim;
    zDim = _zDim;
}

void crPatternGraph::setCoords(std::vector<frCoord> xCoordsIn,
                               std::vector<frCoord> yCoordsIn,
                               std::vector<frLayerNum> zCoordsIn) {
    std::sort(xCoordsIn.begin(), xCoordsIn.end());
    std::sort(yCoordsIn.begin(), yCoordsIn.end());
    std::sort(zCoordsIn.begin(), zCoordsIn.end());
    xCoordsIn.erase(std::unique(xCoordsIn.begin(), xCoordsIn.end()),
                    xCoordsIn.end());
    yCoordsIn.erase(std::unique(yCoordsIn.begin(), yCoordsIn.end()),
                    yCoordsIn.end());
    zCoordsIn.erase(std::unique(zCoordsIn.begin(), zCoordsIn.end()),
                    zCoordsIn.end());

    xCoords = std::move(xCoordsIn);
    yCoords = std::move(yCoordsIn);
    zCoords = std::move(zCoordsIn);
    setDims(xCoords.size(), yCoords.size(), zCoords.size());
}

void crPatternGraph::clear() {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    xCoords.clear();
    yCoords.clear();
    zCoords.clear();
    nodes.clear();
    nodeMap.clear();
}

template <typename T>
bool crPatternGraph::hasCoord(const std::vector<T>& coords, T coord) const {
    return std::binary_search(coords.begin(), coords.end(), coord);
}

template <typename T>
crIndex_t crPatternGraph::getCoordIdx(const std::vector<T>& coords,
                                      T coord) const {
    auto it = std::lower_bound(coords.begin(), coords.end(), coord);
    if (it == coords.end() || *it != coord) {
        return -1;
    }
    return static_cast<crIndex_t>(it - coords.begin());
}

bool crPatternGraph::hasMazeXCoord(frCoord xCoord) const {
    return hasCoord(xCoords, xCoord);
}

bool crPatternGraph::hasMazeYCoord(frCoord yCoord) const {
    return hasCoord(yCoords, yCoord);
}

bool crPatternGraph::hasMazeZCoord(frLayerNum layerNum) const {
    return hasCoord(zCoords, layerNum);
}

bool crPatternGraph::hasMazeIdx(frCoord xCoord, frCoord yCoord,
                                frLayerNum layerNum) const {
    return hasMazeXCoord(xCoord) && hasMazeYCoord(yCoord) &&
           hasMazeZCoord(layerNum);
}

crMazeType crPatternGraph::getMazeIdx(frCoord xCoord, frCoord yCoord,
                                      frLayerNum layerNum) const {
    return crMazeType(getCoordIdx(xCoords, xCoord), getCoordIdx(yCoords, yCoord),
                      getCoordIdx(zCoords, layerNum));
}

std::uint64_t crPatternGraph::getMapKey(const crMazeType& mazeIdx) const {
    if (mazeIdx.empty()) {
        throw std::invalid_argument("cannot encode empty crMazeType");
    }

    auto x = static_cast<std::uint64_t>(mazeIdx.x);
    auto y = static_cast<std::uint64_t>(mazeIdx.y);
    auto z = static_cast<std::uint64_t>(mazeIdx.z);
    return z * static_cast<std::uint64_t>(xDim) *
               static_cast<std::uint64_t>(yDim) +
           x * static_cast<std::uint64_t>(yDim) + y;
}

bool crPatternGraph::hasNode(const crMazeType& mazeIdx) const {
    return nodeMap.find(getMapKey(mazeIdx)) != nodeMap.end();
}

int crPatternGraph::getNodeIdx(const crMazeType& mazeIdx) const {
    auto it = nodeMap.find(getMapKey(mazeIdx));
    if (it == nodeMap.end()) {
        return -1;
    }
    return it->second;
}

int crPatternGraph::addNode(const crMazeType& mazeIdx) {
    if (mazeIdx.empty() || mazeIdx.x < 0 || mazeIdx.y < 0 || mazeIdx.z < 0 ||
        static_cast<std::size_t>(mazeIdx.x) >= xDim ||
        static_cast<std::size_t>(mazeIdx.y) >= yDim ||
        static_cast<std::size_t>(mazeIdx.z) >= zDim) {
        return -1;
    }

    auto key = getMapKey(mazeIdx);
    auto it = nodeMap.find(key);
    if (it != nodeMap.end()) {
        return it->second;
    }

    nodes.push_back(mazeIdx);
    auto nodeIdx = static_cast<int>(nodes.size()) - 1;
    nodeMap[key] = nodeIdx;
    return nodeIdx;
}

void crPatternGraph::addNodes(const std::vector<crMazeType>& mazeIdxs) {
    for (auto& mazeIdx : mazeIdxs) {
        addNode(mazeIdx);
    }
}

void crPatternGraph::addRoutingLayerNodes(frCoord xCoord, frCoord yCoord) {
    if (!hasMazeXCoord(xCoord) || !hasMazeYCoord(yCoord)) {
        return;
    }

    for (auto layerNum : zCoords) {
        addNode(getMazeIdx(xCoord, yCoord, layerNum));
    }
}

void crPatternGraph::build(CustomRouteWorker* worker) {
    clear();
    this->worker = worker;
}

}  // namespace fr
