#include "cr.hpp"
#include "worker/crWorker.hpp"

namespace fr {
void CustomRoute::run() {
    std::vector<std::unique_ptr<CustomRouteWorker>> workers;
    if (!routeTasks.empty()) {
        workers.push_back(std::make_unique<CustomRouteWorker>(this, routeTasks));
    }
}
}  // namespace fr
