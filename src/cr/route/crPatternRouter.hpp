#pragma once

#include <cstdint>
#include <vector>
#include "cr/cr.hpp"
#include "cr/graph/crPatternGraph.hpp"
#include "cr/type/crNet.hpp"

namespace fr {

class crPatternRouter {
   public:
    crPatternRouter(crPatternGraph* graphIn, crNet* netIn,
                    crPatternEnum policyIn)
        : graph(graphIn), net(netIn), policy(policyIn), path() {}

    bool searchPath();
    const std::vector<crMazeType>& getPath() const { return path; }
    std::vector<crMazeType>& getPath() { return path; }

   protected:
    static constexpr frUInt4 bottomLayerPenaltyCoeff = 2;

    bool searchPathL();
    bool initEndpoints(std::vector<crMazeType>& srcs,
                       std::vector<crMazeType>& dsts) const;
    bool buildLPath(const crMazeType& src, const crMazeType& srcRoute,
                    const crMazeType& mid, const crMazeType& dstRoute,
                    const crMazeType& dst,
                    std::vector<crMazeType>& candidate) const;
    bool appendStraightSegment(const crMazeType& begin, const crMazeType& end,
                               std::vector<crMazeType>& candidate) const;
    bool appendViaSegment(const crMazeType& begin, const crMazeType& end,
                          std::vector<crMazeType>& candidate) const;
    frCoord getPathCost(const std::vector<crMazeType>& candidate) const;
    frCoord getSegmentCost(const crMazeType& begin,
                           const crMazeType& end) const;
    frCoord getPlanarSegmentCost(const crMazeType& begin,
                                 const crMazeType& end) const;
    frCoord getViaSegmentCost(const crMazeType& begin,
                              const crMazeType& end) const;
    bool isPlanarDir(frDirEnum dir) const;
    void printPath() const;

    crPatternGraph* graph;
    crNet* net;
    crPatternEnum policy;
    std::vector<crMazeType> path;
};

}  // namespace fr
