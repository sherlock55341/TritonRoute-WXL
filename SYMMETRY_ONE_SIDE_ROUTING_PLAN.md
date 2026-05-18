# 单侧对称布线计划

这是目前更倾向采用的 symmetry routing 方案。

核心想法是：GR 和 TA 只主动维护 reference side，mirror side 由 reference
side 的结果复制/镜像得到，不作为一棵独立的 routed tree 维护。TA 需要额外记录
track 之间的一对一对称匹配，因为真实设计里的 track 未必严格关于对称轴几何对
称，而额外插入 track 的代价比较大。DR 先 route reference side，再复制出
mirror side；如果复制后产生 DRV，再沿用原有 search-and-repair 流程修复，但对
保持两侧对称的走线给 cost reward，使修复结果尽量接近对称。

一句话概括：

```text
route one side
copy to the matched mirror tracks / mirror geometry
repair copied-side DRV with symmetry-biased search-and-repair
```

这个方案比“GR mirror 一次、TA mirror 一次、DR 再 mirror 一次”更容易推理。
如果太早把 mirror side materialize 出来，普通 router 很容易把两侧当作普通
net 的两个独立分支来修复，后面就会出现重复 anchor、worker boundary split、
merge warning 之类的问题。

当前 GR prototype 可能仍会把 mirror-side topology 作为中间投影显式建出来，用
于验证和调试连通性。这是实现中间态，不是最终 ownership 模型。长期目标仍然是
reference-side routing state 作为 source of truth，mirror-side state 由复制/
镜像和后续 repair 派生出来。

## Shared Constraint

symmetry constraint 描述一个 self-symmetric net：

```text
netName
axisDir
axisCoord
referenceSide
```

`referenceSide` 表示当两边都可以作为起点时，由哪一侧来主动驱动 routing。
如果 source pin 明确落在某一侧，可以直接使用 source 所在侧；如果 source 在
axis 上，则由 `referenceSide` 决定先 route 哪一半。

mirror side 不是被忽略。它的作用是检查 reference side 选出来的 route 被
mirror 之后是否也可行、可连、合法。

## GR Direction

GR 阶段应该生成单侧 global route，而不是一开始就生成完整的 symmetric tree。

第一步是按 symmetry axis 对 net 的 GCell terminals 分类：

```text
reference-side terminals
mirror-side terminals
on-axis terminals
```

只有 reference-side terminals 和必要的 on-axis connection 参与主动的 GR
topology generation。mirror-side terminals 不应该被 mirror 回 reference
side 当作 FLUTE 输入。它们只用于后续验证：最终 mirror 出来的结果能不能覆盖
或连接到真实的 mirror-side pins。

GR flow 建议是：

1. 构造 reference-side terminal set。
2. 只对 reference-side terminals 跑 FLUTE。
3. 检查 reference-side tree 是否连接到 symmetry axis。
4. 如果没有连接到 axis，就从当前 tree 找一个合理的最近点，补一条到 axis
   contact 的连接。
5. 对这个 one-sided topology 跑 layer assignment 和 GR maze。
6. 在 maze cost 评估里，尽量加入 mirror-side feasibility：如果某条
   reference-side edge mirror 过去会被 block 或不可用，那么这条 edge 应该变
   得很贵，或者直接不可走。
7. 写出 reference side 和 axis connection 的 one-sided guide 结果。

mirror-side GCell terminals 不进入 active topology。它们是 validation
targets：最终 mirror 出来的 mirror-side route 必须能覆盖或连接到它们。

这个方案也改变了 layer assignment 的理解方式。layer assignment 应该先只在
reference-side topology 上运行，得到 reference side 的 layer 和 via stack。
之后这些 layer / via stack 直接随最终 route mirror 到另一侧。这样 mirror side
天然使用相同的 layer 选择，而不是让普通 layer assignment 在完整树上自己“猜”
哪些 branch 应该对称。

## TA Direction

TA 阶段应该给 reference-side guides 分配 track，同时使用显式的 symmetric
track pairing table 找到对应的 mirror track。不要默认假设 track 坐标严格关于
axis 几何对称，也不要优先通过插入新 track 来补齐对称性；插入 track 代价较大，
应该作为后续扩展或极少数无法匹配时的例外。

TA 不应该让 mirror-side guide 独立选 track。每一个 reference-side track
candidate 都应该通过 pairing table 找到一个确定的 mirror track。理想情况下，
几何镜像坐标可以作为匹配目标：

```text
horizontal axis:
  horizontal wire track y -> 2 * axis - y
  vertical wire track x   -> x

vertical axis:
  vertical wire track x   -> 2 * axis - x
  horizontal wire track y -> y
```

但实际实现不应该只依赖这个公式。更稳妥的做法是先为每个相关 routing layer 建
立一组一对一 track pair：

```text
reference track -> matched mirror track
mirror track    -> matched reference track
```

每一对 track 在当前 symmetry constraint 下被认为是对称的。pairing 可以优先选
离几何镜像坐标最近、方向和 layer 一致、且可用于 mirror-side guide 的 track；
如果没有可接受的匹配，则对应 reference-side candidate 应该被拒绝或给予很大
penalty。

pairing table 需要满足几个不变量：

- pairing 按 routing layer 和 track direction 建立；同一 layer/direction 内
  一个 reference track 最多匹配一个 mirror track，一个 mirror track 也最多被
  一个 reference track 匹配。
- on-axis track 是 self-mirror；如果 track 坐标就在 axis 上，它可以和自己配
  对，但不应该再被其它 off-axis track 复用。
- 如果多个 reference candidates 竞争同一个 mirror track，先按几何镜像距离、
  guide 可用性和 blockage/DRC feasibility 排序；无法唯一且稳定决策时，不要强
  行共享，应拒绝较差 candidate 或加大 penalty。
- pairing 是 symmetry metadata，不代表真实新增 track。真实 track 插入必须作为
  单独设计决策处理。

track cost 应该包含 paired mirror-track feasibility：

```text
cost = reference track cost + paired mirror track feasibility cost
```

如果 matched mirror track 不存在、被 block、或者明显不可用，那么这个
reference-side candidate 应该被拒绝，或者被加很大的 penalty。只有当问题只是
worker-local track list 不完整、而全局 pairing table 已经确认存在匹配 track
时，prototype 才考虑引入 worker-local virtual view；不要把 virtual view 理解
为真实插入新 track。

TA 写回时，active result 仍然以 reference side 为主。mirror side 不需要在 TA
阶段 materialize 成一条独立 route。最终复制应该留到 DR 之后，因为只有 DR 之后
才能确认 detailed legality。

## DR Direction

DR 是最终几何决定阶段。这里的方向调整为：先把 reference side route 好，再把
结果复制到 mirror side；复制后的结果如果出现 DRV，不直接推翻 reference-side
route，而是进入原有 search-and-repair 流程修复。

第一步仍然只主动 route reference side。之后按 symmetry constraint 和 TA 的
track pairing 把 detailed geometry 复制到 mirror side。复制时 wire / via /
pin-connection 的坐标和 track 都应该从 reference result 推导出来，reference
route 仍然是初始 source of truth。

复制后可能出现 DRV，例如 mirror-side 局部 blockage、track 不完全对称、或者
worker 切分造成的局部冲突。此时不要只做一次硬失败，也不要手写一个完全独立的
mirror patcher；更合适的是复用原来的 DR search-and-repair，把 copied mirror
result 当作初始解的一部分继续修。

search-and-repair 的区别在 cost model。对于和另一侧已经存在的 symmetric
geometry 匹配的候选 edge，应给予 reward，使其比完全非对称的替代路线更便宜：

```text
effective_cost(edge) =
  normal_drc_and_history_cost(edge)
  + symmetry_adjusted_wirelength(edge)

symmetry_adjusted_wirelength(edge) =
  full wirelength cost, if no matched symmetric edge exists
  reduced wirelength cost, e.g. half, if the edge matches the opposite side
```

这个 reward 不能压过真实 DRC。DRC violation、不可用 track、blocking、pin
connectivity 仍然是硬约束或高优先级 cost；reward 只用于在多个合法或可修复候选
之间偏向更对称的结果。edge 正好在 axis 上 self-mirror 时只计算一次，不应该双
重奖励。

DR worker 的 window placement 仍然需要考虑 symmetry axis 和 paired mirror
geometry。prototype 阶段可以优先使用 axis-centered 或 pair-aware windows，确
保 search-and-repair 同时看得到冲突点和它的 symmetric counterpart。

DR flow 建议是：

1. route reference-side pins 和 axis connection。
2. 确认 reference route 连接到了 axis，并满足 reference-side legality。
3. 根据 symmetry constraint 和 TA track pair 在 DR 工作结果中复制出
   mirror-side detailed route。
4. 检查复制后的 mirror-side DRV 和 pin connectivity。
5. 如果有 DRV，使用原有 search-and-repair 修复，但在 cost 中奖励与另一侧
   symmetric geometry 匹配的候选走线。
6. 修复收敛后，一起写回 reference-side shapes、mirrored shapes，以及
   axis/self-mirror shapes。
7. 最终输出可能不再严格逐段完全对称，但应该在 DRC 合法前提下尽量接近
   对称。

如果最终 mirror 出来的 route 不能合法连接 mirror-side pins，优先通过
symmetry-biased search-and-repair 修复。只有当修复持续失败，或者发现
reference-side route 本身导致 mirror side 没有合理可修路径时，才回到
reference side 重新选择更适合复制的 route。

## 为什么这个 Flow 更合适

只维护一个 active side 有几个明显好处：

- reference side 和 mirror side 不会在 GR、TA、DR 中逐步漂移。
- layer 选择和 via stack 天然对称，因为它们直接从 reference side 复制过去。
- worker boundary split 和普通 repair 逻辑看到的 duplicated branch 更少。
- ownership 更清楚：reference route 是 source of truth，mirror route 是从它
  推导出来的。

代价是每个阶段都要保留足够的 symmetry metadata。GR 需要知道哪些 terminal 属
于 reference / mirror / axis；TA 需要保存 track pair；DR 需要能找到 edge 的
symmetric counterpart，并在 repair cost 中使用这个关系。只在 reference side
看起来好是不够的；复制和 repair 后的 mirror side 也必须可用、可连、合法。

目标行为可以总结为：

```text
GR decides the one-sided topology and axis connection.
TA chooses reference tracks through explicit symmetric track pairs.
DR copies the reference route, then repairs copied-side DRV with symmetry reward.
Final output is legal first and symmetry-biased second.
```
