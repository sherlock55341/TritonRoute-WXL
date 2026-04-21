#pragma once

#include "crFig.hpp"

namespace fr {
class crShape : public crPinFig {
public:
    crShape() : crPinFig() {}
    virtual void setLayerNum(frLayerNum _layerNum) = 0;
    virtual frLayerNum getLayerNum() const = 0;
};
}