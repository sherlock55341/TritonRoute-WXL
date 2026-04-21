#include "crPatternGraph.hpp"
#include <stdexcept>

namespace fr {

void crPatternGraph::setDims(std::size_t _xDim, std::size_t _yDim,
                             std::size_t _zDim) {
    xDim = _xDim;
    yDim = _yDim;
    zDim = _zDim;
}

void crPatternGraph::clear() {
    nodes.clear();
    nodeMap.clear();
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

void crPatternGraph::build(CustomRouteWorker* worker) {
    clear();
    this->worker = worker;
}

}  // namespace fr
