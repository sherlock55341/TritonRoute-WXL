# Self-Symmetry Routing Design

本文是当前代码中 self-symmetry 布线的唯一设计入口。原来的
`self_symmetry_lead_mirror_search_repair.md`、
`self_symmetry_detailed_routing.md` 和 superpowers plan 都已合并到这里；旧文件只保留
归档和跳转用途。

本文按当前实现写事实，不按早期 planned milestone 写状态。

## Scope

Self-symmetry 由 `frNet` 上的 `frcSelfSymmetry` 标记驱动，贯穿 IO、GR、TA 和 DR。
第三方 LEF/DEF parser、FLUTE、Boost geometry/rtree 都不理解 self-symmetry 语义。

主流程由 `FlexRoute::main()` 串起来：

```text
readLefDef
-> PA
-> GR, when no guide is provided
-> read generated guide
-> RP
-> TA
-> DR
-> write DEF
```

相关入口：

- `src/FlexRoute.cpp`: PA 后按 net 名前缀识别 self-symmetry net，并注入 constraint。
- `src/db/obj/frNet.h`: 保存 constraint 并提供 nullable accessor。
- `src/gr/FlexGR_*self_sym*`: GR topology、mirror shadow、2D/3D repair。
- `src/ta/FlexTA_end.cpp`: TA route 写回 guide，并做 axis guide snap。
- `src/dr/FlexDR_self_sym.cpp`: dedicated self-symmetry DR phase、worker modes、checker。

## Constraint Source

Self-symmetry 不是从标准 DEF/LEF constraint 语法读入的。当前代码在
`FlexRoute::init()` 完成 LEF/DEF 读取和 PA 后调用 `initSelfSymmetryConstraints()`。
所有 net 名以 `Symmtry` 为前缀的 net 都会作为 self-symmetry candidate：

```cpp
static const string prefix = "Symmtry";
return net != nullptr &&
       net->getName().compare(0, prefix.size(), prefix) == 0;
```

约束不是按 net list 或固定 axis 表注入。每个 candidate net 会收集其 RPin access
point 的全局坐标，然后调用 `get_self_symmetry_axis(points, isHorizontal, axis)`
自动推导 axis：

- 先用三阶矩判断更像 vertical-axis symmetry 还是 horizontal-axis symmetry。
- 若矩判断不明显，则用 R-tree 最近镜像点打分，选择 mirror error 更小的一侧。
- `isAxisHorizontal = false`: vertical axis，`axis` 是 physical `x` 坐标。
- `isAxisHorizontal = true`: horizontal axis，`axis` 是 physical `y` 坐标。
- `axis`: DBU 坐标，后续 GR/TA/DR 会按 routing track/gcell 语义 snap。

注入前做轻量检查：

- axis 必须在 die box 内。
- RPin/access point 缺失的 candidate 会跳过；axis 在 die 外会报错退出。

通过后设置：

```cpp
net->setSelfSymmetryConstraint(selfSymmetryConstraint);
net->setConstraint(frNetRoutingConstraint::frcSelfSymmetry);
```

注意：外部 net 名前缀拼写是 `Symmtry`，不是 `Symmetry`。只要前缀匹配就会进入
self-symmetry flow，例如 `Symmtry1`、`Symmtry31`、`Symmtry32_1`。

## DB Model

DB 层只保存轻量约束：

```cpp
struct frSelfSymmetryConstraint {
  bool isAxisHorizontal;
  int axis;
};
```

`frNet` 持有：

- `frNetRoutingConstraint constraint`
- `frSelfSymmetryConstraint selfSymmetryConstraint`

后续阶段统一用 nullable pointer 判断是否启用：

```cpp
const frSelfSymmetryConstraint* frNet::getSelfSymmetryConstraintPtr() const {
  return constraint == frNetRoutingConstraint::frcSelfSymmetry
      ? &selfSymmetryConstraint
      : nullptr;
}
```

这个接口边界很重要：从 net 进入时用 nullable pointer 表达“有没有约束”；确认有约束后，
纯几何 helper 使用 `const frSelfSymmetryConstraint&`，避免每个几何函数重复处理空状态。

## Axis Semantics

GR/TA/DR 都会把 physical axis snap 到可布线坐标。GR/TA 共享的主要 helper 在
`src/gr/FlexGR_self_sym_utils.h`：

- `findNearestSelfSymmetryRoutingTrack()`: 找最近合法 routing track。
- `SelfSymmetryAxisContext::fromReferencePoint()`: 用参考点构造 effective axis 和 axis gcell。
- `sideOfPoint()` / `sideOfGCell()`: 判定点在 axis 哪一侧，返回 `-1/0/1`。
- `mirrorPoint()` / `mirrorGCell()`: 按 effective axis 或 axis gcell 做镜像。
- `isAxisEdge()`: 判断 gcell edge 是否完全在 axis 上。

DR 有自己的 snapped-axis context 和 checker helper，语义保持一致。

几个约定：

- 日志分析时要区分 `original_axis` 和 snapped/effective axis。
- root 在 axis 上时，`normalizeSelfSymmetryRootSide(0)` fallback 成 `-1`。
- lead/source side 是当前拓扑 root side；axis side 共享；另一侧是 mirror side。
- 不在 `frNode`、`grNode`、`grPathSeg`、`grVia` 上持久化 side 字段，side 由坐标和
  constraint 临时推导。

## GR Overview

`FlexGR::main()` 中 self-symmetry 相关顺序是：

```text
init / resource analysis
-> initGR
-> 2D macro repair
-> 2D searchRepair passes
-> searchRepairSelfSymmetryMirror
-> layerAssign
-> stageSelfSymmetry3DLeadOnly
-> 3D searchRepair
-> restoreSelfSymmetry3DLayerAssignMirror
-> beginSelfSymmetry3DGuidedSearchRepair
-> guided 3D searchRepair
-> write guide
```

如果存在 self-symmetry net，GR worker 会串行执行，避免 shared state、shadow demand
和 cross-window writeback 发生竞态。

## GR Initial Topology

`FlexGR::initGR_genTopology()` 对普通 net 调 `initGR_genTopology_net()`，对
`frcSelfSymmetry` net 调 `initGR_genTopology_selfsymmetry_net()`。

Self-symmetry 初始 topology 的核心行为：

- AP 坐标写回 pin node location/layer。
- 根据 root pin/root gcell 决定 root side。
- root pin、axis pins、root-side pins 进入 source tree。
- mirror-side pin nodes 保留在 `frNet::nodes` 中，但 `parent=null`，不接入 tree。
- root-side tree 在 Hanan grid 上生成。
- root-side tree 必须接触 axis；如果自然 tree 没接触，额外搜索一段到 axis gcell。

相关函数：

- `initGR_genTopology_selfsymmetry_net()`
- `genSelfSymmetryRootSideTopology()`
- `genSelfSymmetryOppositeSideTopology()`

这个阶段 mirror-side pin 断开是预期状态，不是连接性失败。进入 layer assignment 前，
mirror side 会被 materialize。

## GR Lead/Mirror Strategy

GR search repair 期间的 source of truth 是：

```text
real route object = lead route + axis route
mirror effect     = mirror shadow demand / mirror-aware cost
```

设计取舍是：

```text
严格自对称 > mirror 侧局部自由度 > 全局共同最优
```

也就是说，repair 阶段不让 mirror side 独立 ripup/reroute。mirror side 的资源压力通过
shadow demand 和 mirror edge cost 反馈到 lead candidate 选择中。

### Mirror Shadow Demand

对一条非 axis 的 lead edge `e`：

```text
shadow(e) = mirror(e)
```

add route 时：

```text
add lead demand(e)
add mirror shadow demand(mirror(e))
```

ripup 时：

```text
sub lead demand(e)
sub mirror shadow demand(mirror(e))
```

axis edge 只加/减一次 demand，不做 mirror shadow 复制，避免重复计数。

### 2D Search Repair

2D worker 只打开 lead/axis 真实 objects。关键行为：

- worker tile 覆盖 axis 时，添加 axis endpoint，强制 lead tree 连接 axis。
- 修改 self-symmetry path segment demand 时，同时更新 lead demand 和 mirror shadow demand。
- `FlexGRGridGraph::getNextPathCost()` 查询候选 edge 的 mirror edge，把 mirror-side
  congestion/overflow/blockage 加进 cost。
- mirror side 不在 region query 里保存为 `grPathSeg` / `grVia`，因此普通 worker 不会独立打开它。

这解决了跨 worker 独立修改两侧导致自对称破坏的问题；代价是 mirror side 没有本地最优能力，
只能通过 lead path 间接调整。

### Pattern Route

早期计划曾考虑在 `patternRoute_LShape()` 里做 mirror-aware L-shape 选择。复查代码后，
这不是当前实现的主路径。

原因是 self-symmetry 初始 topology 已经是 Hanan graph 上的 rectilinear tree，普通
`patternRoute_LShape()` 只处理 `x/y` 都不同的非共线 Steiner-Steiner 边。当前
self-symmetry net 通常不会在 pattern route 阶段产生两种 L-shape 可选。

后续如果要改初始选择，应优先改：

- `genSelfSymmetryRootSideTopology()` / `genSelfSymmetryOppositeSideTopology()` 的 Hanan cost。
- 2D/3D A* mirror-aware cost。
- mirror materialization 的 guide cost 和 pin-cover 策略。

### 2D Mirror Materialization

普通 2D repair 完成后、`layerAssign()` 前，`searchRepairSelfSymmetryMirror()` materialize
mirror side。

流程：

- 从最终 lead/axis parent-child tree 收集 root-side vertices/edges。
- 非 axis lead edges 镜像成 mirror guide edges。
- 以 axis 上已有 tree node 为 source。
- 在 Hanan grid 上连接 mirror-side pin gcells。
- guide edge cost 低，非 guide edge cost 高；pin 覆盖是硬要求，精确走 guide 是软偏好。
- 把生成的 mirror-side tree 写回同一个 `frNet` 的真实 parent-child tree。
- 刷新 `rootGCellNode`、`firstNonRPinNode` 和 self-symmetry topology cache。

进入 `layerAssign()` 前，mirror pins 应该已经被真实 topology 覆盖，后续 guide/TA/DR
能看到完整 topology。

## GR Layer Assignment And 3D

`layerAssign()` 有 self-symmetry mirror cost hook：

```cpp
getSelfSymmetryLayerAssignMirrorCost(currNode, net, layerNum)
```

3D routing 分两轮：

1. `stageSelfSymmetry3DLeadOnly()`
   - 收集 mirror-side nodes/shapes/vias。
   - snapshot 并临时移除 mirror topology/objects。
   - 对 lead route 添加 mirror shadow demand。
   - 让第一轮 3D repair 只稳定 lead side。
2. `restoreSelfSymmetry3DLayerAssignMirror()`
   - 移除 lead shadow demand。
   - 恢复 mirror topology/objects。
   - 重建 region query。
3. `beginSelfSymmetry3DGuidedSearchRepair()`
   - 开启 guided mirror cost。
   - 第二轮 3D repair 中，A* 对 mirror edge 的拥塞/非法状态加 cost。
4. `endSelfSymmetry3DGuidedSearchRepair()`
   - 关闭 guided state 并清空临时状态。

这套流程的目标是让 3D repair 先稳定 lead，再恢复 mirror，并用 guided mirror cost 做第二轮校正。

## Guide And TA Handoff

GR 输出 guide 后，TA 做普通 track assignment。TA 结束阶段：

- `FlexTAWorker::saveToGuides()` 把 assigned `taPathSeg` 转成 `frPathSeg`，通过
  `guide->setRoutes(tmp)` 写回 `frGuide::routes`。
- `FlexTA::snapSelfSymmetryAxisGuides()` 遍历 self-symmetry nets，把接近 axis 的长 guide route
  吸附到 effective axis track。

axis snap 会跳过：

- dummy axis route。
- 不沿 axis 方向的 segment。
- pin-local short route。
- 太短或离 axis 太远的 route。
- unsupported route object。

TA route 进入 DR 的路径：

```text
TA pathSeg
-> frGuide::routes
-> DR initGCell2BoundaryPin boundary pin
-> DR follow-guide origGuides / maze guide cost
```

`FlexDR::initFromTA()` 当前没有在 `FlexDR::init()` 中启用，所以 TA routes 不会直接复制成
`frNet::shapes`；但它们确实作为 boundary pin 和 guide cost 进入 DR。

## DR Overview

当前 DR 已经有 dedicated self-symmetry phase。`FlexDR::main()` 先调用
`runSelfSymmetryDRPhase()`；如果该 phase routed，后续普通 DR 只收集 ordinary target nets。

```text
FlexDR::init
-> runSelfSymmetryDRPhase
-> collectOrdinaryDRTargetNets
-> ordinary searchRepair passes with ordinary targetNets
-> final checker/report
```

这已经不是早期文档里说的“DR 仍是普通 detailed routing”。当前实现已经有
self-symmetry-aware DR path，但仍是 best-effort 对称，合法性/DRC 优先。

## Self-Symmetry DR Phase

`runSelfSymmetryDRPhase()` 做：

1. `collectSelfSymmetryDRTargetNets()` 收集所有 `frcSelfSymmetry` nets。
2. `initSelfSymmetryDRSharedState()` 计算 snapped axis、root side 等共享状态。
3. 构造 `FlexDRSearchRepairPhase`：
   - `targetNets = selfSymmetryNets`
   - `stageName = "self-symmetry dr phase"`
   - `removeBoundaryPinsOnInit = false`
   - `selfSymmetryDRSharedStates = &selfSymmetryDRSharedStates`
4. 调普通 `searchRepair()`，但 worker 带 self-symmetry shared state。
5. 检查每个 shared state：不能 failed，必须看到 axis contact 或 axis link done。
6. 输出 phase route count 和 checker。
7. snapshot self-symmetry routes。
8. `keepOnlySelfSymmetryDRTargetRoutes()` 保留 self-symmetry 结果并清掉非目标 net 的临时 route。

有 self-symmetry shared state 时，DR tiled workers 串行执行，避免跨 tile axis-link 状态竞争。

## DR Worker Modes

`FlexDRWorker::routeNet()` 在满足以下条件时进入 self-symmetry routing：

```text
net has frcSelfSymmetry && fixMode == 9
```

核心函数是 `routeNet_selfSymmetry()`。route mode 定义：

```cpp
enum class SelfSymmetryDRRouteMode {
  None,
  Lead,
  AxisLink,
  Mirror
};
```

### Lead Mode

- 根据 root pin、axis side 把 pins 分成 lead pins 和 mirror pins。
- root pin、axis pins、root-side pins 进入 lead pass。
- 如果 shared state 已有跨 tile axis boundary link，把它作为 lead boundary pin 注入。
- 用普通 A* helper route selected lead pins。

### AxisLink Mode

- 如果 lead result 没有接触 effective axis，收集 axis candidates。
- 优先选择 tile boundary 上的 axis candidates，方便跨 tile 共享。
- 从 lead connected components 搜到 axis pin。
- 成功后写 shared `axisLinkPoint` 和 `axisLinkLayerNum`。
- 如果 axis link 失败或 link 后仍未接触 axis，phase 失败并退出。

### Mirror Mode

- 用 lead detailed result 构造 mirror reward guide。
- 收集 axis sources。
- 从 axis sources 强制出发 route mirror pins。
- `getSelfSymmetryDRCost()` 在 Mirror mode 下奖励 mirror guide edge、惩罚 miss；在 Lead/AxisLink
  mode 下按靠近/接触 axis 的关系加 cost。

Mirror guide 是软约束。为了 DRC/legal，mirror side 可以偏离 mirrored guide；checker 会报告偏差。

## Ordinary DR After Self-Symmetry

如果 self-symmetry phase 成功：

- `collectOrdinaryDRTargetNets()` 收集非 self-symmetry nets。
- 后续普通 `searchRepair()` 使用 `ordinaryPhase.targetNets`。
- self-symmetry nets 不再作为普通 DR target enqueue。
- self-symmetry shapes/vias 保留在 region query 中，让 ordinary nets 绕开它们。

这实现了当前设计顺序：

```text
route self-symmetry nets first
-> check/report self-symmetry result
-> keep self-symmetry routes
-> route ordinary nets around them
```

普通 net 必须给 self-symmetry net 让路；当前策略不为了 ordinary failure 回头拆已经完成的
self-symmetry net。

## Reports And Debug

当前默认 flow 不应产生 `[gr-debug]`、`[gr-cost]`、`[gr-stage]` 或
`[guide-debug]` 这类实验调试日志。需要定位对称性问题时，常看的 report / trace
包括：

- GR 2D search-repair
  - `@@@ self-symmetry search-repair 2d @@@`
  - axis contact、shadow cell counts、mirror cost queries
- GR 2D mirror repair
  - `@@@ self-symmetry mirror repair 2d @@@`
  - `mirror_hanan_pins_covered`
  - `mirror_repair_pins_covered`
  - guide hits/misses
- TA axis snap
  - `SSTA_AXIS_SNAP_TRACE`
  - `SSTA_AXIS_SNAP_SUMMARY`
- DR phase
  - `@@@ self-symmetry dr phase @@@`
  - `self_symmetry_nets`
  - `ordinary_nets_in_phase`
  - `after_phase_shapes/vias/patch_wires`
- DR checker
  - `@@@ self-symmetry dr checker @@@`
  - `missing_axis_contact`
  - `segments_missing_mirror`
  - `vias_missing_mirror`
  - `axis_duplicate_shapes`
  - `self_markers`
  - `changed_by_current_dr`

旧调试路径里曾有只针对 `Symmtry5` 的 topology dump；这不是当前默认验收路径。
当前 self-symmetry 目标由 `Symmtry` 前缀统一决定。

## Validation

没有 committed unit/CTest。推荐至少做一次本地 smoke：

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-flow
cd build/selfsym-flow
cp ../../src/gr/flute/POST9.dat ../../src/gr/flute/POWV9.dat .
../TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output selfsym_flow.def \
  > selfsym_flow.log 2>&1
```

建议检查：

```bash
rg -n "self-symmetry topology|root-side reaches axis|self-symmetry mirror repair 2d|SSTA_AXIS_SNAP_SUMMARY|self-symmetry dr phase|self-symmetry dr checker|number of violations" selfsym_flow.log
```

重点验收：

- `root-side reaches axis: 1`
- mirror pins 在 GR 初始 source tree 阶段断开，mirror materialization 后被覆盖。
- `mirror_hanan_pins_covered` 和 `mirror_repair_pins_covered` 为 `N/N`。
- TA guide route 被写入并参与 DR boundary pin / guide cost。
- self-symmetry DR phase 先于 ordinary DR。
- `ordinary_nets_in_phase` 为 0。
- 普通 DR target nets 不包含 self-symmetry nets。
- final checker 报告 axis contact、missing mirror、duplicate axis shapes 和 markers。

## Current Caveats

- Constraint 来源仍是项目约定，不是通用 parser 支持；当前约定是 net 名前缀
  `Symmtry`。
- `Symmtry` 拼写是现有输入契约，不是 `Symmetry`。
- strict symmetry 不压过 DRC/legal；mirror guide miss 是允许并需要报告的偏差。
- GR mirror shadow demand 的跨 worker/window outside shadow 路径仍是需要重点验证的风险点。
- Pattern route 不是当前 self-symmetry 主要选择点。
- Checker 是报告机制，不等于形式化证明所有 detailed objects 都严格成对。
- 如果 root 恰好在 axis 上，fallback root side 会影响 lead/mirror 分类。
- 代码没有单测覆盖，修改后需要 build + benchmark smoke + log/DEF/DRC 对比。

## Historical Notes

旧文档中有几类已经过期的说法：

- “DR 仍是普通 detailed routing”
- “M1-M5 planned”
- “没有 dedicated self-symmetry DR phase”
- “没有 post-DR checker”

这些描述对应 2026-06-01 附近的设计阶段，不再反映当前代码。当前事实是：

- GR 已有 lead/axis topology、mirror shadow demand/cost、2D mirror materialization、
  layer assignment mirror cost、3D lead-only staging 和 guided 3D repair。
- TA 会把 assigned routes 写回 `frGuide::routes`，并 snap axis guide。
- DR 已有 dedicated self-symmetry phase、Lead/AxisLink/Mirror worker modes、route snapshot、
  checker、ordinary target net 隔离。

旧文件保留为归档入口，当前阅读和维护请以本文为准。
