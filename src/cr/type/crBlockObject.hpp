#pragma once

#include "frBaseTypes.h"
#include "db/obj/frBlockObject.h"
#include "db/infra/frPoint.h"

namespace fr {
class crBlockObject : public frBlockObject {
   public:
    // constructors
    crBlockObject() {}
    virtual ~crBlockObject() {}
    // getters
    // setters
    // others
    frBlockObjectEnum typeId() const { return crcBlockObject; }
};
}  // namespace fr
