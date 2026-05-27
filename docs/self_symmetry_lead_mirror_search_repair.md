# Self-Symmetry Lead/Mirror Search Repair 设计记录

本文记录自对称 net 在 GR search repair 中的设计取舍。当前结论是：

```text
ordinary 2D search repair 只修改 lead/axis 侧的真实 route。
repair 期间 mirror 侧用 shadow demand / cost 影响 lead 侧选择。
所有 root/lead-side 2D repair 结束后，在 layerAssign() 前 materialize mirror 侧。
mirror materialization 使用最终 lead route 的镜像 guide，在 Hanan grid 上覆盖 mirror pins。
```

这不是追求全局最优的方案，而是主动把优先级定为：

```text
严格自对称 > mirror 侧局部自由度 > 全局共同最优
```

## 背景

当前 GR search repair 会按普通 worker window 局部重布线。
普通 net 的局部重布线可以在不同 window 内独立发生，但 self-symmetry net
如果两侧被不同 worker 独立修改，很容易破坏自对称结构。

已有 self-symmetry topology 生成逻辑主要在：

- `src/gr/FlexGR.cpp`: `initGR_genTopology_selfsymmetry_net`
- `src/gr/FlexGR_topo.cpp`: `genSelfSymmetryRootSideTopology`
- `src/gr/FlexGR_topo.cpp`: `genSelfSymmetryOppositeSideTopology`

本记录讨论的是 search repair 期间应该如何表示和修改这类 net。

## 术语

```text
lead route
  真实存在、允许被 topology / pattern route / search repair 修改的半边 route。

axis anchor
  lead route 必须连接到的对称轴锚点。它是真实拓扑的一部分，不是 mirror shadow。

mirror shadow demand
  由 lead route 镜像投影到 cmap/cmap2D 的需求和代价记录。
  它影响其他 net 和 lead A* 的 cost，但不是 grPathSeg/grVia/frNode。

source of truth
  search repair 期间，self-symmetry net 的 source of truth 是 lead route + axis anchor。
  mirror shadow 永远由 source of truth 重新生成。
```

## 核心约束

对 self-symmetry net 引入 lead / axis / mirror-shadow 分工：

- ordinary worker repair 期间，lead 侧是唯一允许被 topology、pattern route、search repair 修改的一侧。
- lead 侧拓扑必须连接到 symmetry axis；不能只是布一半孤立子网。
- axis 上的拓扑和 demand 是真实共享部分，只记录一次，不能被 mirror 重复计数。
- root-side repair 完成前，mirror 侧不参与独立 topology 生成，不参与独立 pattern route，不参与独立 maze search。
- root-side repair 完成前，mirror 侧不保存 `grPathSeg`、`grVia`、`frNode` 等具体 route object。
- root-side repair 期间，mirror 侧只通过 shadow demand / shadow cost 进入 `cmap` / `cmap2D`。
- 普通 worker 不会扫到 self-symmetry net 的 mirror objects，因为 mirror objects 在 worker repair 阶段还不存在。
- `searchRepairSelfSymmetryMirror()` 运行后，mirror 侧 pin 会挂到真实 `frNode` tree，随后由 `layerAssign()` 生成 guide/DEF 可见的 route objects。

因此 search repair 期间 self-symmetry net 的表示应满足：

```text
ordinary repair real route objects = lead route + axis anchor/axis edges
ordinary repair mirror effect      = demand(mirror(non-axis lead route))
post-repair mirror route           = guide-aware Hanan tree from axis to mirror pins
```

## 当前实现阶段：Root Repair 后 Mirror Materialization

当前实现分成两个阶段。

第一阶段在 `initGR_genTopology_selfsymmetry_net()` 只 materialize root/lead side + axis
anchor 的 `frNode` parent-child tree：

- 保留所有原始 pin node。
- root/lead/axis 侧 pin node 挂到对应 GCell node。
- root pin 连接到 root GCell node。
- axis anchor 和额外 Steiner vertex 按 GCell center 创建 `frNode`。
- `genSelfSymmetryRootSideTopology()` 产出的 root-side edges 从 `rootGCellIdx`
  开始定向为真实 parent-child tree。
- mirror-side pin node 保留在 net 中，但 `parent=null`，不挂 child，不进入 source
  tree terminal list。
- 不创建 mirror-side GCell node、Steiner tree、`grPathSeg`、`grVia` 或 region-query
  object。

这是第一阶段的预期行为，不是连接性失败。此时 guide 生成/检查只验证已连接的
self-symmetry source pins；断开的 mirror pins 会在第二阶段 materialize。

第二阶段在三轮普通 2D `searchRepair()` 后、`layerAssign()` 前运行
`searchRepairSelfSymmetryMirror()`：

- 从最终 lead/axis parent-child tree 收集 root-side route vertices/edges。
- 将非 axis lead edges 镜像为 mirror guide edges。
- 以 axis 上已有 route node 为 source，在 Hanan grid 上连接所有 opposite-side pin GCell。
- guide edge cost 低，非 guide edge cost 高；pin 覆盖是硬要求，精确镜像是软偏好。
- 将 Hanan 结果写回同一个 `frNet` 的真实 parent-child tree，mirror pins 不再保持断开。
- 刷新 `rootGCellNode` / `firstNonRPinNode` 和 self-symmetry topology cache，供
  `layerAssign()` 使用。

这意味着 root-side repair 期间 mirror pins 断开是预期状态，但进入 `layerAssign()` 前，
mirror pins 应已经被 materialized 并覆盖。最终 guide/DEF/DR 可以看到 mirror 侧 route。

## Lead Side 判定

lead/source side 不需要额外枚举或持久化字段。实现上直接使用当前拓扑 root 所在侧：

```text
lead/source side = root side
```

root 在 symmetry axis 上时沿用当前 fallback：

```text
rootSide = -1
```

也就是说，search repair 期间不引入新的 source-side enum，也不给 `frNode`、`grNode`、
`grPathSeg`、`grVia` 或其他 route object 增加 side 字段。
axis / lead / mirror 的判断应由坐标和 self-symmetry constraint 临时推导。

## Constraint 访问策略

constraint 入口和几何 helper 的接口语义需要分开：

- 从 net 进入时，使用 nullable pointer 表达“这个 net 是否有 self-symmetry constraint”。
- 已经确认有 constraint 后，纯几何 helper 使用 `const frSelfSymmetryConstraint&`。

建议接口形态是：

```cpp
const frSelfSymmetryConstraint* getSelfSymmetryConstraintPtr(const frNet* net) const;
bool isSelfSymmetryNet(const frNet* net) const;
int getSelfSymmetryPointSide(const frPoint&, const frSelfSymmetryConstraint&) const;
bool isOnSelfSymmetryAxis(const frPoint&, const frSelfSymmetryConstraint&) const;
frPoint mirrorPoint(const frPoint&, const frSelfSymmetryConstraint&) const;
```

这样调用方在 net 层面处理“有没有约束”，几何函数不用反复处理空状态。

## Axis Anchor 约束

lead side 有一个硬约束：必须连到对称轴。

拓扑上应表达为：

```text
root / lead pins -> lead-side tree -> axis anchor
```

然后 mirror shadow 才能从 lead-side tree 派生出来。否则 mirror 只是 demand 投影，
但 self-symmetry net 的拓扑语义是不完整的。

代码里已有这个方向：

- `genSelfSymmetryRootSideTopology()` 会检查 root-side tree 是否碰到 axis。
- 如果没有碰到 axis，它会额外找一条到 axis gcell 的路径。

后续实现应该保留并强化这个行为：

- axis GCell / axis node 是 lead topology 的强制 terminal。
- search repair 不能破坏 lead-to-axis 的连通性。
- worker window 内如果包含 axis anchor，需要把它当固定 terminal。
- worker window 外如果 axis anchor 不在窗口内，则通过 boundary pin 保持原有 lead-to-axis 连接。

## Symmtry5 Debug Dump

`Symmtry5` 会输出 self-symmetry topology dump，用于验收当前单侧树：

```text
@@@ self-symmetry topology @@@
net: Symmtry5
axis: horizontal y=<physical_axis>, gcell_y=<axis_gcell_idx>
root side: <-1|1>
pins:
  p0: <inst>/<term> loc=(x,y), gcell=(gx,gy), layer=<layer>, side=<-1|0|1>, root=<0|1>, in_source_tree=<0|1>, parent=<node_id_or_null>
root-side terminals:
root-side vertices:
root-side edges:
source tree parent-child:
  node <id> gcell=(gx,gy) parent=<id|null> children=[...]
root-side reaches axis: <0|1>
@@@ end self-symmetry topology @@@
```

字段含义：

- `side` 是 pin AP 相对 physical symmetry axis 的位置。
- `in_source_tree=1` 表示 pin 参与当前 root/lead/axis source tree。
- mirror-side pin 应显示 `in_source_tree=0` 且 `parent=null`。
- `source tree parent-child` 只列 route tree node，不把 pin child 混入 children 列表。
- `root-side reaches axis: 1` 表示 root-side vertex 或 edge 接触 axis GCell；mirror
  侧断开不算失败。

mirror 阶段还会输出：

```text
@@@ self-symmetry mirror repair 2d @@@
net: Symmtry5
mirror_hanan_pins_covered: <covered>/<total>
mirror_guide_edges: <N>
mirror_repair_guide_hits: <N>
mirror_repair_guide_misses: <N>
mirror_repair_pins_covered: <covered>/<total>
@@@ end self-symmetry mirror repair 2d @@@
```

期望 `mirror_hanan_pins_covered` 和 `mirror_repair_pins_covered` 都是 `N/N`。

验收方式：

```bash
rg -n "self-symmetry topology|axis:|pins:|source tree parent-child|root-side reaches axis" \
  build/selfsym-task3-tree/selfsym_task3_tree.log
```

期望能看到 `pins`、`source tree parent-child` 和 `root-side reaches axis: 1`。

## Mirror Shadow Demand

mirror side 的线会影响布线 demand 和代价，但不保存为具体物体。

对一条非 axis 的 lead edge `e`：

```text
shadow(e) = mirror(e)
```

更新 cmap/cmap2D 时：

```text
add lead demand(e)
add mirror shadow demand(mirror(e))
```

ripup 时：

```text
sub lead demand(e)
sub mirror shadow demand(mirror(e))
```

axis edge 特殊处理：

```text
axis edge 只加/减一次 demand，不做 mirror shadow 复制。
```

这样 mirror side 对其他 net 是“真实占资源”的，但对 region query 和 worker subnet
来说它不是一个可被选中、可被 ripup、可被 reroute 的对象。

## Search Repair 行为

普通 search repair 只对 lead/axis 侧执行真实搜索。
lead 侧搜索时，cost 会感知 mirror shadow 的代价。

对 lead 侧候选 edge `e`，计算其镜像 edge `mirror(e)`，总代价可以表达为：

```text
cost(e) = lead_cost(e)
        + alpha * mirror_shadow_congestion_cost(mirror(e))
        + beta  * mirror_shadow_block_penalty(mirror(e))
        + gamma * mirror_shadow_overflow_penalty(mirror(e))
```

这样 mirror 侧不会自己改 route，但 mirror 侧的资源压力会反馈到 lead 侧选择中。
如果 mirror 侧走不通或代价很高，lead 侧应该被迫换一条镜像后更合理的路径。

mirror shadow 的 source of truth 永远是 lead route。不要单独维护一份“mirror 物体表”。
需要删除旧 mirror demand 时，遍历旧 lead route 并镜像扣除；需要加入新 mirror demand 时，
遍历新 lead route 并镜像加入。

普通 2D repair 全部结束后，mirror 阶段不再修改 lead route。它以最终 lead route 镜像
出的 guide 为软约束，materialize mirror-side tree 并覆盖所有 mirror pins。若 guide
不能直接覆盖所有 pins，Hanan 搜索允许偏离 guide。

## 与 Worker Window 的关系

这个方案的关键收益是 mirror side 不会进入 region query：

```text
ordinary worker repair 阶段 region query 中只有 lead/axis 真实对象
cmap/cmap2D 中有 lead demand + mirror shadow demand
```

因此：

- mirror side 不会被其他 worker 当成 self-symmetry subnet 打开。
- mirror side 不会被普通 search repair 独立 ripup/reroute。
- 其他 net 仍会通过 cmap/cmap2D 看到 mirror side 的资源压力。
- 如果 mirror side 造成拥塞，只能通过 lead candidate 的 mirror cost 反馈回来。
- mirror materialization 在 ordinary 2D worker repair 全部结束后才运行，因此新写入的
  mirror topology 不会再被普通 2D worker 独立打开。

## 一次 Reroute 的状态更新

对 self-symmetry net reroute lead 侧时，需要保持 demand 一致性：

1. 从 local/global cmap 中扣除旧 lead demand。
2. 由旧 lead route 镜像，扣除旧 mirror shadow demand。
3. 搜索新的 lead route，并保持 lead-to-axis 连通。
4. 加入新 lead demand。
5. 由新 lead route 镜像，加入新 mirror shadow demand。
6. 只更新 lead/axis 的真实 route objects 和 region query。

不能出现以下状态：

```text
lead 已更新，mirror shadow 还是旧 lead 派生出来的 demand
mirror shadow 已删除，lead 搜索却仍按旧 self demand 计费
mirror side 被普通 worker 独立 ripup/reroute
axis edge 被 lead 和 mirror 重复计数
```

## Mirror Materialization

ordinary search repair 期间不保存 mirror route object。

guide 输出、DEF 输出、DR 输入需要看到具体 mirror 形状，因此当前实现会在 ordinary
2D search repair 全部结束后做一次受控 materialization：

```text
mirror guide = mirror(final lead route objects)
final mirror topology = Hanan(axis source, mirror pins, mirror guide)
```

这个阶段以 guide-aware Hanan search 新增 mirror-side parent-child tree：

- 在所有 GR worker search repair 完成之后执行。
- 在 `layerAssign()` 之前执行，使 mirror 侧进入后续 layer assignment。
- 不 ripup 或重写 lead/axis source tree。
- axis 上共享的对象仍只保留一份，不复制 axis-only edge。
- 输出 guide / DEF 时，mirror-side route 已经是 `frNode` tree 的一部分。

## 已接受的代价

- mirror 侧没有独立局部最优能力。
- 如果 lead 侧和 mirror 侧 routing 环境差异很大，lead 侧可能需要绕远路。
- 某些 mirror 侧拥塞无法通过 mirror 本地调整解决，只能通过 lead 间接缓解。
- ordinary search repair 期间 `frNet::grShapes/grVias` 不完整表达最终物理全网，只表达 lead/axis source route。
- mirror materialization 后，进入 `layerAssign()` 的 parent-child tree 覆盖 lead/axis 和 mirror pins。
- 这个方案不保证全局最优，目标是工程上可控的强自对称。

## 后续实现关注点

- 明确区分真实 lead/axis route object 与 mirror shadow demand。
- self-symmetry topology 生成必须保留 lead-to-axis anchor。
- pattern route 只选择 lead 侧 L-shape，但 cost 包含 mirror shadow。
- lead A* cost 能查询候选 edge 镜像后的 congestion/blockage。
- writeback 阶段更新真实 lead/axis objects，同时按 lead route 增删 mirror shadow demand。
- mirror materialization 阶段需要维持 `mirror_hanan_pins_covered: N/N` 和
  `mirror_repair_pins_covered: N/N`，并避免让后续普通 2D worker 再打开 mirror-side route。
