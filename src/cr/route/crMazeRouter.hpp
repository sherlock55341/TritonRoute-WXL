#pragma once

#include <cstdint>
#include <vector>
#include "cr/cr.hpp"
#include "cr/graph/crPatternGraph.hpp"
#include "cr/type/crNet.hpp"

namespace fr {

class crMazeRouter {
   public:
    crMazeRouter(crPatternGraph* graphIn, crNet* netIn, crPatternEnum policyIn)
        : graph(graphIn),
          net(netIn),
          policy(policyIn),
          path() {}

    bool searchPath();
    const std::vector<crMazeType>& getPath() const { return path; }
    std::vector<crMazeType>& getPath() { return path; }

   protected:
    struct SearchState {
        crMazeType node;
        frDirEnum lastDir;
        int turnCount;
    };

    struct Wavefront {
        frCoord cost;
        SearchState state;
    };

    struct WavefrontComp {
        bool operator()(const Wavefront& lhs, const Wavefront& rhs) const {
            return lhs.cost > rhs.cost;
        }
    };

    static constexpr frUInt4 bottomLayerPenaltyCoeff = 2;
    static constexpr int maxTrackedTurnCount = 2;

    bool searchPathL();
    bool initEndpoints(std::vector<crMazeType>& srcs,
                       std::vector<crMazeType>& dsts) const;
    frCoord getNextPathCost(const Wavefront& curr, const SearchState& nextState,
                            frDirEnum dir) const;
    frCoord getPlanarEdgeCost(const crMazeType& curr,
                              const crMazeType& next) const;
    frCoord getViaEdgeCost(const crMazeType& curr,
                           const crMazeType& next) const;
    SearchState getNextState(const SearchState& curr, const crMazeType& next,
                             frDirEnum dir) const;
    std::uint64_t getStateKey(const SearchState& state) const;
    bool isPlanarDir(frDirEnum dir) const;
    bool isTurn(frDirEnum currDir, frDirEnum nextDir) const;
    bool runDijkstra(const std::vector<crMazeType>& srcs,
                     const std::vector<crMazeType>& dsts);
    void printPath() const;

    crPatternGraph* graph;
    crNet* net;
    crPatternEnum policy;
    std::vector<crMazeType> path;
};

}  // namespace fr
