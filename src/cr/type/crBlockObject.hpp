#pragma once

#include "frBaseTypes.h"
#include "db/obj/frBlockObject.h"
#include "db/infra/frPoint.h"

namespace fr {

// Base object for the custom-route object hierarchy. It keeps CR objects in
// the same frBlockObject type system as the rest of TritonRoute while allowing
// CR-specific typeIds.
class crBlockObject : public frBlockObject {
   public:
    crBlockObject() {}
    virtual ~crBlockObject() {}

    // Return the CR base object type; subclasses override this with specific
    // CR enum values.
    frBlockObjectEnum typeId() const { return crcBlockObject; }
};
}  // namespace fr
