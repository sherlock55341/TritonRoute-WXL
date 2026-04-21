#include "crWorker.hpp"
#include "cr/route/crMazeRouter.hpp"

namespace fr {
CustomRouteWorker::CustomRouteWorker(
    CustomRoute* _cr,
    const std::vector<std::pair<frNet*, crPatternEnum>>& _tasks)
    : design(_cr->getDesign()),
      nets(),
      cr(_cr),
      policies(),
      DRCCost(DRCCOST),
      patternGraph(std::make_unique<crPatternGraph>()) {
    for (auto& [net, policy] : _tasks) {
        initNet(net);
        policies.push_back(policy);
    }
    initRouteBox();
    patternGraph->build(this);
    initPatternGraph();
}

void CustomRouteWorker::route() {
    for (std::size_t i = 0; i < nets.size() && i < policies.size(); ++i) {
        std::vector<crMazeType> path;
        routeNet(nets[i].get(), policies[i], path);
    }
}

bool CustomRouteWorker::routeNet(crNet* cNet, crPatternEnum policy,
                                 std::vector<crMazeType>& path) const {
    crMazeRouter router(patternGraph.get(), cNet, policy);
    if (!router.searchPath()) {
        path.clear();
        std::cout << "CANNOT FIND" << std::endl;
        return false;
    }
    path = router.getPath();
    return true;
}
}  // namespace fr
