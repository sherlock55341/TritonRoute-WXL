#pragma once

#include "crFig.hpp"

namespace fr {

// Abstract base for CR shapes that live on a routing layer.
class crShape : public crPinFig {
   public:
    crShape() : crPinFig() {}

    // Set the database routing-layer number for this shape.
    virtual void setLayerNum(frLayerNum _layerNum) = 0;

    // Return the database routing-layer number for this shape.
    virtual frLayerNum getLayerNum() const = 0;
};
}  // namespace fr
