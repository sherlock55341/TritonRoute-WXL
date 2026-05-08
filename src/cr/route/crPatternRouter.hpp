#pragma once

#include <cstdint>
#include <vector>
#include "cr/cr.hpp"
#include "cr/graph/crPatternGraph.hpp"
#include "cr/type/crNet.hpp"

namespace fr {

// Pattern router for one crNet. It enumerates policy-specific route candidates
// on crPatternGraph and stores the best path as a sequence of maze indices.
class crPatternRouter {
   public:
    crPatternRouter(crPatternGraph* graphIn, crNet* netIn,
                    crPatternEnum policyIn)
        : graph(graphIn), net(netIn), policy(policyIn), path() {}

    // Run the configured pattern search and store the winning maze path.
    bool searchPath();
    // Return the immutable winning path from the most recent successful search.
    const std::vector<crMazeType>& getPath() const { return path; }
    // Return the mutable winning path for caller-side post-processing.
    std::vector<crMazeType>& getPath() { return path; }

   protected:
    // Multiplier for bottom-layer planar routes to discourage routing on M1.
    static constexpr frUInt4 bottomLayerPenaltyCoeff = 2;

    // Enumerate all L-pattern candidates between two-pin endpoints and keep the
    // lowest-cost valid path.
    bool searchPathL();
    // Collect source/destination access-point maze indices from the two pins.
    bool initEndpoints(std::vector<crMazeType>& srcs,
                       std::vector<crMazeType>& dsts) const;
    // Build one same-preferred-direction straight candidate with source and
    // destination via stacks.
    bool buildStraightPath(const crMazeType& src, const crMazeType& srcRoute,
                           const crMazeType& dstRoute, const crMazeType& dst,
                           std::vector<crMazeType>& candidate) const;
    // Build one layered L candidate: source via stack, first preferred leg,
    // bend via stack, second preferred leg, and destination via stack.
    bool buildLayeredLPath(const crMazeType& src, const crMazeType& firstRoute,
                           const crMazeType& bendFirst,
                           const crMazeType& bendSecond,
                           const crMazeType& secondRoute, const crMazeType& dst,
                           std::vector<crMazeType>& candidate) const;
    // Return whether layerNum has horizontal preferred routing direction.
    bool isHorizontalPreferredLayer(frLayerNum layerNum) const;
    // Return whether layerNum has vertical preferred routing direction.
    bool isVerticalPreferredLayer(frLayerNum layerNum) const;
    // Append a same-layer Manhattan segment to candidate.
    bool appendStraightSegment(const crMazeType& begin, const crMazeType& end,
                               std::vector<crMazeType>& candidate) const;
    // Append a vertical same-x/y via stack to candidate.
    bool appendViaSegment(const crMazeType& begin, const crMazeType& end,
                          std::vector<crMazeType>& candidate) const;
    // Score a full candidate path by merging collinear runs into cost segments.
    frCoord getPathCost(const std::vector<crMazeType>& candidate) const;
    // Dispatch a segment to planar or via cost scoring.
    frCoord getSegmentCost(const crMazeType& begin,
                           const crMazeType& end) const;
    // Score a same-layer straight segment using length and graph cost channels.
    frCoord getPlanarSegmentCost(const crMazeType& begin,
                                 const crMazeType& end) const;
    // Score a vertical transition using via cost and graph cost channels.
    frCoord getViaSegmentCost(const crMazeType& begin,
                              const crMazeType& end) const;
    // Return whether a direction is planar rather than vertical.
    bool isPlanarDir(frDirEnum dir) const;
    // Print the selected path and endpoint AP debug information.
    void printPath() const;

    // Non-owning graph used for coordinate lookup and quick-cost queries.
    crPatternGraph* graph;
    // Non-owning net being routed.
    crNet* net;
    // Pattern search policy selected for this net.
    crPatternEnum policy;
    // Winning route as a sequence of graph nodes after searchPath succeeds.
    std::vector<crMazeType> path;
};

}  // namespace fr
