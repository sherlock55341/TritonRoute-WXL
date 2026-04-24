#include "cr.hpp"
#include "worker/crWorker.hpp"

namespace fr {
void CustomRoute::run() {
    // Flow:
    // 1. Build one worker for the current task batch.
    // 2. Print dense graph size for current debug visibility.
    // 3. Delegate routing and DB writeback to the worker.
    std::vector<std::unique_ptr<CustomRouteWorker>> workers;
    if (!routeTasks.empty()) {
        workers.push_back(
            std::make_unique<CustomRouteWorker>(this, routeTasks));
    }
    std::cout << workers.front()->getPatternGraph()->getXCoords().size() *
                     workers.front()->getPatternGraph()->getYCoords().size() *
                     workers.front()->getPatternGraph()->getZCoords().size()
              << std::endl;
    workers.front()->route();
}
}  // namespace fr
