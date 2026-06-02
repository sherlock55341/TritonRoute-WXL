# Self-Symmetry Detailed Routing Plan

本文记录 self-symmetry net 在 detailed routing 侧的当前事实、目标阶段和验收方式。
截至 2026-06-01，GR 已经产生 self-symmetry-aware guide/TA/DR 可消费结果，但 DR
仍是普通 detailed routing。

## Current State

已经完成的部分：

- GR 侧已有 root/lead/axis source tree。
- GR 2D repair 有 lead repair + mirror shadow cost。
- GR 2D repair 后会 materialize mirror topology。
- Layer assignment 有 mirror cost hook。
- 3D 有 lead-only staging、mirror restore 和 guided 3D repair。
- TA 会把分配后的 track route 写回 `frGuide::routes`。

DR 当前事实：

- `FlexTAWorker::saveToGuides()` 把 TA `taPathSeg` 转成 `frPathSeg`，并通过
  `guide->setRoutes(tmp)` 写入 `frGuide::routes`。
- `FlexDR::initFromTA()` 当前没有在 `FlexDR::init()` 中启用，所以 DR 不直接把
  TA route 复制成 `frNet::shapes`。
- `FlexDR::initGCell2BoundaryPin()` 会读取 `guide->getRoutes()`，用 TA routes
  生成 worker boundary pin。
- `FlexDRWorker::initNetObjs()` 在 `followGuide=true` 时从 region query 读取
  original guides，保存到 `drNet::origGuides`。
- `FlexDRWorker::initMazeCost_guide_helper()` 把 `drNet::origGuides` 转成 grid guide，
  `FlexGridGraph::getNextPathCost()` 对不在 guide 上的 edge 收 `GUIDECOST`。
- DR 当前不会识别 lead/mirror detailed route pair，不会绑定对称 via/track，也没有
  post-DR symmetry checker。

因此，TA 结果通过 `frGuide::routes` 进入了 DR 的 boundary pin 和 guide-cost 数据路径；
但这还不是 self-symmetry-aware detailed routing。

## Target Order

整体顺序固定为：

```text
先把所有有形状要求的自对称 net 做好
-> 检查并修好自对称 net 自身问题
-> 冻结自对称 net
-> 再 route 没有形状要求的普通 net
```

普通 net 必须给自对称 net 让路。普通 DR 阶段不能拆已经完成的自对称 net。
没有新增 CLI 开关；检测到 `frcSelfSymmetry` 后自动启用。

设计优先级：

```text
DRC/legal first
-> best-effort detailed-route symmetry
-> ordinary-net success if it can route around frozen self-symmetry shapes
```

严格镜像不能压过合法性；mirror side 可以为了 DRC/legal 偏离 mirrored guide。
普通 net 不能要求 self-symmetry net 回头让路。

## M0: Documentation State Lock

状态：已完成，且不改变 routing 行为。

记录内容：

- GR self-symmetry 已完成到 guide/TA/DR 可消费的程度。
- Pattern route 对 self-symmetry net 通常没有两种 L-shape 方案可选。
- TA 结果不是没用，`frGuide::routes` 被 DR 用于 boundary pin 和 guide cost。
- DR 当前没有 lead/mirror pair routing、对称 via/track 绑定、post-DR symmetry check。
- 后续目标是尽可能对称，但 DRC/legal 优先。

验收方式：

- 本文件存在并记录 DR 目标阶段。
- `docs/self_symmetry_lead_mirror_search_repair.md` 记录 GR/TA/DR 当前边界。
- `docs/superpowers/plans/2026-05-26-self-symmetry-lead-mirror-search-repair.md`
  拆出 M0-M5。

## M1: DR Observation And Checker

状态：planned。只加观测，不改变 routing 结果。

实现内容：

- 在 DR init 中记录 self-symmetry net 读到的 TA guide route 数量。
- 记录 self-symmetry boundary pin 来源和数量。
- 记录 self-symmetry net 是否被普通 DR 初始化成 `drNet`。
- 增加 post-DR symmetry checker，只报告不修复。

checker 报告项：

- Segment 是否有 mirror counterpart。
- Via 是否有 mirror counterpart。
- Axis shape 是否重复。
- Self-symmetry net 自身是否有 DRC marker。
- 普通 DR 是否改动过 self-symmetry net。

验收方式：

- Smoke log 中能看到每个 self-symmetry net 的 TA route count。
- Smoke log 中能看到 boundary pin count 和 `drNet` 初始化记录。
- Final log 中有 post-DR symmetry report。
- DEF 输出不因 checker 改变。

## M2: Self-symmetry Net Dedicated DR Phase

状态：planned。

实现内容：

- 在普通 DR 前收集所有 `frcSelfSymmetry` net。
- Self-symmetry phase 中普通 signal net 暂不参与 routing。
- Self-symmetry net 只避开 fixed objects、PG、OBS、pin/blockage 等固定约束。
- 该阶段结束后，self-symmetry net 必须已经生成 detailed shapes/vias。

验收方式：

- Self-symmetry net 不再依赖后续普通 DR 才生成 detailed route。
- 普通 net 尚未参与布线。
- Checker 能看到 self-symmetry detailed route。

## M3: Lead-side Plus Mirror-guide Self-symmetry Routing

状态：planned，是核心实现阶段。

实现内容：

- 以 TA `frGuide::routes` 作为 DR 输入 skeleton 之一。
- 根据 symmetry axis 把 guide、shape、pin 分成 lead、axis、mirror。
- 先 route lead/axis side。
- 用 lead detailed result 生成 mirror-side guide。
- Mirror side 以 mirrored guide 为强偏好继续 route。
- Mirrored guide 是软约束，允许为了 DRC/legal cost 偏离。
- Axis shape 只保留一份。

验收方式：

- 日志中有 lead pass、mirror guide generation、mirror pass。
- Mirror guide hit/miss 有统计。
- Self-symmetry net 自身 DRC 在这个阶段内处理到可接受状态。
- Checker 能报告最终 detailed route 对称偏差。

## M4: Freeze Self-symmetry Net

状态：planned。

实现内容：

- 写回 self-symmetry `shapes/vias/patchWires`。
- 更新 DR region query。
- 普通 DR 不再 enqueue self-symmetry net。
- 普通 DR 不 ripup self-symmetry net。
- Self-symmetry shape/via 在普通 DR 中按类似 PG fixed obstacle 处理。

验收方式：

- 普通 DR queue 中没有 self-symmetry net。
- Self-symmetry shape/via 被普通 net 当障碍物看到。
- Marker 涉及 self-symmetry 和 ordinary net 时，只 reroute ordinary net。

## M5: Ordinary Nets Around Frozen Self-symmetry

状态：planned。

实现内容：

- 普通 net 使用现有 DR search repair。
- 普通 net 必须绕开已冻结的 self-symmetry net。
- 如果绕不开，报告 ordinary routing failure。
- 不允许因为普通 net 的问题回头拆 self-symmetry net。

验收方式：

- Self-symmetry net 在普通 DR 后保持不变。
- 普通 net 绕开 self-symmetry shape/via。
- Final DRC 和 final symmetry checker 都有报告。

## Smoke Command

每个 milestone 至少运行：

```bash
cmake --build build -j$(nproc)
mkdir -p build/selfsym-dr
cd build/selfsym-dr
cp ../../src/gr/flute/POST9.dat ../../src/gr/flute/POWV9.dat .
../TritonRoute \
  -lef ~/benchmark/primarius/outdata/ispd18_test1.input.lef \
  -def ~/benchmark/primarius/outdata/pattern_route_lay.def \
  -output <stage>.def \
  > <stage>.log 2>&1
```

检查项：

- TA guide route 被 DR 读取。
- Self-symmetry phase 先于 ordinary phase。
- Ordinary DR 不 enqueue self-symmetry net。
- Self-symmetry net 在 ordinary DR 前后 shape/via 不变。
- Final DRC 结果。
- Final symmetry checker 结果。

## Risks

- 直接把 TA routes 当 fixed detailed route 会绕过现有 DR DRC/repair 能力；M2/M3
  需要通过 DR 数据结构产生 legal detailed shapes，而不是盲目复制。
- DR worker 当前是 marker/search-repair 驱动；冻结 self-symmetry net 后，marker
  ownership 和 queue policy 必须保证只 ordinary net reroute。
- Axis geometry 必须只写一次，尤其是 patch wire 和 via around axis 的去重。
- 如果 self-symmetry shape 作为 fixed obstacle 注入 region query 的时机不对，普通
  net 可能看不到障碍物，或者 self-symmetry net 又被普通 DR 初始化成可 ripup net。
- Strict symmetry 和 DRC/legal 冲突时，日志必须清楚报告 mirror guide miss / deviation，
  不能隐藏成 ordinary routing failure。
