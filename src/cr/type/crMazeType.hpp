#pragma once

namespace fr {
using crIndex_t = int;
class crMazeType {
   public:
    crMazeType() : x(-1), y(-1), z(-1) {}
    crMazeType(crIndex_t xIn, crIndex_t yIn, crIndex_t zIn)
        : x(xIn), y(yIn), z(zIn) {}
    bool empty() const { return (x == -1 && y == -1 && z == -1); }
    crIndex_t x;
    crIndex_t y;
    crIndex_t z;
};
}  // namespace fr