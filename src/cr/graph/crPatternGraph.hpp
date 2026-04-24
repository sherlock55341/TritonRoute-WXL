#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
class frViaDef;
class crPathSeg;
class crVia;

// Logical quick-cost channel stored in crPatternGraph::bits. The bit layout is
// intentionally aligned with FlexGridGraph where CR currently supports it.
enum class crCostClass { Grid, Shape, Drc, Marker, Block };

// Dense pattern graph for CR routing. It owns coordinate arrays, quick-cost
// bitfields, and special-via annotations, but it does not own nets or design DB
// objects.
class crPatternGraph {
   public:
    crPatternGraph() : worker(nullptr), xDim(0), yDim(0), zDim(0), bits() {}

    // Return the number of allocated graph nodes.
    std::size_t getNumNodes() const { return getGridCapacity(); }
    // Return sorted physical x coordinates indexed by crMazeType::x.
    const std::vector<frCoord>& getXCoords() const { return xCoords; }
    // Return sorted physical y coordinates indexed by crMazeType::y.
    const std::vector<frCoord>& getYCoords() const { return yCoords; }
    // Return sorted routing-layer numbers indexed by crMazeType::z.
    const std::vector<frLayerNum>& getZCoords() const { return zCoords; }
    // Return x dimension of the dense graph.
    std::size_t getXDim() const { return xDim; }
    // Return y dimension of the dense graph.
    std::size_t getYDim() const { return yDim; }
    // Return z dimension of the dense graph.
    std::size_t getZDim() const { return zDim; }
    // Return the design database through the owning worker.
    frDesign* getDesign() const;
    // Return the technology database through the owning worker.
    frTechObject* getTech() const;

    // Set graph dimensions; normally called by setCoords.
    void setDims(std::size_t _xDim, std::size_t _yDim, std::size_t _zDim);
    // Sort/deduplicate coordinate arrays and allocate per-node bit storage.
    void setCoords(std::vector<frCoord> xCoordsIn,
                   std::vector<frCoord> yCoordsIn,
                   std::vector<frLayerNum> zCoordsIn);
    // Clear dimensions, coordinates, quick costs, and special-via annotations.
    void clear();
    // Attach the owning worker after clearing graph state.
    void build(CustomRouteWorker* worker);
    // Return whether xCoord is present in the graph coordinate array.
    bool hasMazeXCoord(frCoord xCoord) const;
    // Return whether yCoord is present in the graph coordinate array.
    bool hasMazeYCoord(frCoord yCoord) const;
    // Return whether layerNum is present in the graph coordinate array.
    bool hasMazeZCoord(frLayerNum layerNum) const;
    // Return whether a physical point/layer can be mapped to a maze index.
    bool hasMazeIdx(frCoord xCoord, frCoord yCoord, frLayerNum layerNum) const;
    // Convert physical point/layer to a maze index; invalid coordinates map to
    // -1 components.
    crMazeType getMazeIdx(frCoord xCoord, frCoord yCoord,
                          frLayerNum layerNum) const;
    // Convert a maze index to a physical point.
    frPoint getPoint(const crMazeType& mazeIdx) const;
    // Convert a maze z index to a routing layer number.
    frLayerNum getLayerNum(const crMazeType& mazeIdx) const;
    // Return whether the maze index is inside graph dimensions.
    bool hasNode(const crMazeType& mazeIdx) const;
    // Return the dense node key used by maps/debug output.
    std::uint64_t getNodeKey(const crMazeType& mazeIdx) const;
    // Step one graph node in dir and return false if the neighbor is invalid.
    bool getNextMazeIdx(const crMazeType& curr, frDirEnum dir,
                        crMazeType& next) const;
    // Return whether planar movement is on the layer non-preferred direction.
    bool hasNonPrefCost(const crMazeType& node, frDirEnum dir) const;
    // Return whether DR-like grid cost is set on the normalized edge.
    bool hasGridCost(const crMazeType& node, frDirEnum dir) const;
    // Return whether DR-like shape cost is set on the normalized edge/node.
    bool hasShapeCost(const crMazeType& node, frDirEnum dir) const;
    // Return whether DR-like quick DRC cost is set on the normalized edge/node.
    bool hasDRCCost(const crMazeType& node, frDirEnum dir) const;
    // Return whether marker/history cost is set on the normalized edge/node.
    bool hasMarkerCost(const crMazeType& node, frDirEnum dir) const;
    // Return whether blockage cost is set on the normalized edge/node.
    bool hasBlockCost(const crMazeType& node, frDirEnum dir) const;
    // Add one grid-cost count or set the grid-cost bit.
    void addGridCost(const crMazeType& node, frDirEnum dir);
    // Remove one grid-cost count or clear the grid-cost bit.
    void subGridCost(const crMazeType& node, frDirEnum dir);
    // Add one shape-cost count.
    void addShapeCost(const crMazeType& node, frDirEnum dir);
    // Remove one shape-cost count.
    void subShapeCost(const crMazeType& node, frDirEnum dir);
    // Add one quick-DRC-cost count.
    void addDRCCost(const crMazeType& node, frDirEnum dir);
    // Remove one quick-DRC-cost count.
    void subDRCCost(const crMazeType& node, frDirEnum dir);
    // Add one marker-cost count.
    void addMarkerCost(const crMazeType& node, frDirEnum dir);
    // Remove one marker-cost count.
    void subMarkerCost(const crMazeType& node, frDirEnum dir);
    // Add one blockage-cost count or set the blockage bit.
    void addBlockCost(const crMazeType& node, frDirEnum dir);
    // Remove one blockage-cost count or clear the blockage bit.
    void subBlockCost(const crMazeType& node, frDirEnum dir);
    // Initialize quick DRC/shape cost from existing design routing in extBox.
    void initDRCCost();
    // Record a special viaDef for a lower-layer maze node.
    void setSVia(const crMazeType& node, frViaDef* viaDef);
    // Return whether a special viaDef is recorded at node.
    bool isSVia(const crMazeType& node) const;
    // Return a special viaDef recorded at node, or nullptr.
    frViaDef* getSViaDef(const crMazeType& node) const;
    // Add one local routed figure's quick-cost footprint to the graph.
    void addPathCost(const crConnFig* connFig);
    // Remove one local routed figure's quick-cost footprint from the graph.
    void subPathCost(const crConnFig* connFig);

   protected:
    // Return whether coord exists in a sorted coordinate vector.
    template <typename T>
    bool hasCoord(const std::vector<T>& coords, T coord) const;
    // Return coord index in a sorted coordinate vector, or -1 when absent.
    template <typename T>
    crIndex_t getCoordIdx(const std::vector<T>& coords, T coord) const;
    // Return whether mazeIdx is non-empty and inside graph dimensions.
    bool isValidMazeIdx(const crMazeType& mazeIdx) const;
    // Return xDim * yDim * zDim.
    std::size_t getGridCapacity() const;
    // Return the dense vector index for a valid maze index.
    std::size_t getGridIdx(const crMazeType& mazeIdx) const;
    // Return the planar-cost storage index for a valid node.
    std::size_t getPlanarCostIdx(const crMazeType& node) const;
    // Return the via-cost storage index normalized to the lower z node.
    std::size_t getViaCostIdx(const crMazeType& node, frDirEnum dir) const;
    // Normalize W/S/D cost lookups to E/N/U storage nodes.
    crMazeType getCanonicalCostNode(crMazeType node, frDirEnum dir) const;
    // Read one logical cost class from the bits vector.
    bool hasCost(const crMazeType& node, frDirEnum dir,
                 crCostClass costClass) const;
    // Add one logical cost class to the bits vector.
    void addCost(const crMazeType& node, frDirEnum dir, crCostClass costClass);
    // Subtract one logical cost class from the bits vector.
    void subCost(const crMazeType& node, frDirEnum dir, crCostClass costClass);
    // Encode a maze index into a dense map key.
    std::uint64_t getMapKey(const crMazeType& mazeIdx) const;
    // Encode a directed edge key from a node plus direction.
    std::uint64_t getEdgeKey(const crMazeType& node, frDirEnum dir) const;
    // Encode a normalized physical edge key for opposite-direction sharing.
    std::uint64_t getCostEdgeKey(crMazeType node, frDirEnum dir) const;
    // Return layer min spacing for a candidate box, considering spacing tables.
    frCoord getMinSpacing(const frBox& box, frLayerNum layerNum) const;
    // Return the physical metal rectangle for one planar graph edge.
    frBox getPlanarEdgeBox(const crMazeType& curr,
                           const crMazeType& next) const;
    // Return whether a DB object belongs to a net outside this worker's CR set.
    bool isExternalObject(frBlockObject* obj) const;
    // Return whether dir is non-preferred on layerNum.
    bool isPlanarNonPrefDir(frLayerNum layerNum, frDirEnum dir) const;
    // Apply DR-like typed cost update: 0/1 = sub/add DRC, 2/3 = sub/add shape.
    void modTypedCost(const crMazeType& node, frDirEnum dir, int type);
    // Update planar metal spacing influence using a DR-like corner-distance
    // test and a typed target cost channel.
    void modMetalShapeCost(const frBox& srcBox, frLayerNum layerNum, int type);
    // Update via-candidate influence caused by metal on an adjacent layer.
    void modMetalShapeViaCost(const frBox& srcBox, frLayerNum layerNum,
                              bool isUpperVia, int type);
    // Update planar and adjacent-via influence for one metal shape.
    void modMetalShapeAllCost(const frBox& srcBox, frLayerNum layerNum,
                              int type);
    // Update via-cut candidate influence for one cut box.
    void modViaShapeCost(const frBox& cutBox, frLayerNum lowerLayerNum,
                         int type);
    // Update EOL influence windows for planar and via candidates.
    void modEolSpacingCost(const frBox& srcBox, frLayerNum layerNum, int type,
                           bool skipVia = false);
    // Apply one EOL test window to planar/via graph candidates.
    void modEolSpacingCostHelper(const frBox& testBox, frLayerNum layerNum,
                                 int eolType, int type);
    // Update graph quick cost from one external DB route object.
    void modFrObjCost(frBlockObject* obj, int type);
    // Query extBox and initialize quick cost from existing DB routing.
    void initExternalDRCCost();
    // Update graph quick cost from one local CR route figure.
    void modPathCost(const crConnFig* connFig, int type);
    // Update graph quick cost from one local CR path segment.
    void modPathSegCost(const crPathSeg* pathSeg, int type);
    // Update graph quick cost from one local CR via.
    void modViaCost(const crVia* via, int type);

    // Owning worker back-pointer; graph does not own it.
    CustomRouteWorker* worker;
    // Dense graph x dimension.
    std::size_t xDim;
    // Dense graph y dimension.
    std::size_t yDim;
    // Dense graph z dimension.
    std::size_t zDim;
    // Sorted physical x coordinates.
    std::vector<frCoord> xCoords;
    // Sorted physical y coordinates.
    std::vector<frCoord> yCoords;
    // Sorted routing-layer numbers.
    std::vector<frLayerNum> zCoords;
    // DR-like per-node bitfield storing grid/block flags and 8-bit
    // shape/DRC/marker counters.
    std::vector<std::uint64_t> bits;
    // Special viaDefs keyed by lower-layer maze node for pin access vias.
    std::map<crMazeType, frViaDef*> sViaDefs;
};

}  // namespace fr
