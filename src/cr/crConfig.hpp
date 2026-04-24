#pragma once

namespace fr {

// Number of max-track-pitch margins added around the access-point bounding box
// to form CustomRouteWorker::routeBox.
constexpr int CR_ROUTE_BOX_MARGIN_PITCHES = 4;
// Multiplier applied to planar routes on a layer's non-preferred direction.
constexpr int CR_NONPREF_ROUTE_PENALTY = 2;
// Legacy CR short penalty constant; current quick-cost routing uses DRCCOST for
// DR-aligned DRC edge weighting.
constexpr int CR_SHORT_DRC_PENALTY = 8;
// Legacy CR spacing penalty constant; current quick-cost routing uses DRCCOST
// scaled by edge length.
constexpr int CR_SPACING_DRC_PENALTY = 8;

}  // namespace fr
