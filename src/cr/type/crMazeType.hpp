#pragma once

namespace fr {
using crIndex_t = int;

// Integer grid coordinate used by the custom routing pattern graph.
// x/y/z are indices into crPatternGraph's xCoords/yCoords/zCoords arrays, not
// physical database coordinates.
class crMazeType {
   public:
    crMazeType() : x(-1), y(-1), z(-1) {}
    crMazeType(crIndex_t xIn, crIndex_t yIn, crIndex_t zIn)
        : x(xIn), y(yIn), z(zIn) {}
    // True when the index is unset and should not be used for graph lookup.
    bool empty() const { return (x == -1 && y == -1 && z == -1); }
    // Lexicographic ordering so maze indices can be map/set keys.
    bool operator<(const crMazeType& rhs) const {
        if (x != rhs.x) {
            return x < rhs.x;
        }
        if (y != rhs.y) {
            return y < rhs.y;
        }
        return z < rhs.z;
    }
    bool operator==(const crMazeType& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
    // Index into crPatternGraph::xCoords.
    crIndex_t x;
    // Index into crPatternGraph::yCoords.
    crIndex_t y;
    // Index into crPatternGraph::zCoords.
    crIndex_t z;
};
}  // namespace fr
