#include "crWorker.hpp"
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
}
}  // namespace fr
