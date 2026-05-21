# FlexGR::initGR_genTopology_net 流程说明

本文记录 `FlexGR::initGR_genTopology_net(frNet *net)` 中
`net->getNodes()` 的来源、节点的 2D/3D 含义，以及这个函数的主要流程。

相关代码：

- `src/gr/FlexGR.cpp:1345`: `FlexGR::initGR_genTopology_net`
- `src/gr/FlexGR.cpp:806`: `FlexGR::initGR`
- `src/gr/FlexGR_topo.cpp:37`: `FlexGR::genSTTopology_FLUTE`
- `src/io/io.cpp:529`: `io::Parser::getDefNets`
- `src/io/io_parser_helper.cpp:926`: `io::Parser::initRPin_rpin`
- `src/db/obj/frNet.h:176`: `frNet::addNode`
- `src/db/obj/frNet.h:267`: `frNet::nodes`
- `src/db/obj/frNode.h:53`: `frNode::setLoc`
- `src/db/obj/frNode.h:56`: `frNode::setLayerNum`

## 入口前的 nodes 是什么时候来的

`initGR_genTopology_net()` 入口处的 `net->getNodes()` 不是 GR 临时算出来的，
而是在 DEF 读入 net connection 时创建的。

调用链大致是：

```text
FlexRoute::init()
  parser.readLefDef()
    readDef()
      defrSetNetCbk(getDefNets)
        io::Parser::getDefNets()
          for each DEF net connection:
            create frNode
            node->setPin(term or instTerm)
            node->setType(frcPin)
            netIn->addNode(node)
```

也就是说，最初每个 DEF net connection 会对应一个 `frNode`。
这些初始 node 是 pin node，类型是 `frcPin`。

创建位置：

- IO pin: `src/io/io.cpp:572`
- instance pin: `src/io/io.cpp:615`

`frNet::addNode()` 做的事情很简单：

```text
node->addToNet(this)
set node id
nodes.push_back(node)
set iterator
```

所以 `frNet::nodes` 是一个 `list<unique_ptr<frNode>>`，由 `frNet` 持有。

## RPin 和 access point 的关系

DEF 读完后，`FlexPA` 会先计算 pin access point。
之后 `parser.initRPin()` 会根据这些 access point 创建 `frRPin`。

调用链：

```text
FlexRoute::init()
  parser.readLefDef()
  parser.postProcess()
  FlexPA pa(getDesign())
  pa.main()
  parser.postProcessGuide()
  parser.initRPin()
```

`initRPin_rpin()` 会遍历每个 net 的 instTerm 和 term，
为每个 pin/access point 创建 `frRPin`，并 `net->addRPin(rpin)`。

因此进入 `FlexGR::initGR()` 时，代码会检查：

```cpp
if (net->getNodes().size() != net->getRPins().size()) {
  cout << "Error: net " << net->getName() << " initial #node != #rpin\n";
}
```

这里说明一个重要约定：

```text
initGR_genTopology_net() 入口处：
  net->nodes 的前半部分就是 pin/rpin 对应的初始 pin nodes
  初始状态下 #nodes 应该等于 #rpins
```

`frNet.h` 里也有注释：

```cpp
std::list<std::unique_ptr<frNode> > nodes;
// the nodes at the beginning of the list correspond to rpins
// there is no guarantee that first element is root
```

## 这个 node 是 2D 还是 3D

`frNode` 数据结构本身是带 3D 信息能力的。
它有：

```text
loc      -> x/y
layerNum -> routing layer
```

所以从结构上说，`frNode` 是一个带 layer 的点。

但是在 `initGR_genTopology_net()` 这个阶段，拓扑主体是 2D 的：

- 初始 pin nodes 会被填上真实 access point 的 x/y/layer。
- gcell center nodes 和 FLUTE Steiner nodes 暂时都放在 `layerNum = 2`。
- 后面的 `layerAssign()` 才会把 topology 分配到多层，并插入竖向连接。

可以理解为：

```text
initGR_genTopology_net 阶段 =
  2D gcell-level topology
  + pin 端点保留真实 3D access point 信息
```

## initGR_genTopology_net 的整体目标

这个函数做的是：

```text
拿每个 pin 的 access point
  -> 找到它所在的 gcell
  -> 为每个有 pin 的 gcell 创建一个 gcell-center node
  -> 用 FLUTE 在这些 gcell center node 上生成 2D Steiner tree
  -> 把真实 pin node 挂到对应的 gcell-center node 上
```

最终形成一棵 parent/children 树。

## 详细流程

### 1. 空网和单 pin 网快速返回

位置：`src/gr/FlexGR.cpp:1348`

逻辑：

```text
if #nodes == 0:
  return

if #nodes == 1:
  setRoot(the only node)
  return
```

这里的 `#nodes` 指的是进入 GR 前已有的 pin nodes。

### 2. 把 pin nodes 排成 source + sinks

位置：`src/gr/FlexGR.cpp:1359`

函数创建局部数组：

```cpp
vector<frNode*> nodes(net->getNodes().size(), nullptr); // 0 is source
```

然后遍历 `net->getNodes()`。

driver 判断规则：

```text
instance term:
  OUTPUT 是 driver

IO term:
  INPUT 是 driver
```

driver 放到 `nodes[0]`。
其他 pin node 依次放到 `nodes[1]`, `nodes[2]`, ...

同时还建立：

```text
pin2Nodes:
  frTerm/frInstTerm -> vector<frNode*>
```

这个 map 后面用来和 `frRPin` 对齐。

注意：代码里原本有 “没有 driver 就报错退出” 的检查，但现在被注释掉了。

### 3. 设置 root

位置：`src/gr/FlexGR.cpp:1421`

```cpp
net->setRoot(nodes[0]);
```

这里 root 是 driver pin node。

### 4. 用 RPin/access point 给 pin node 填 x/y/layer

位置：`src/gr/FlexGR.cpp:1423`

先遍历 `net->getRPins()`，建立：

```text
pin2RPins:
  frTerm/frInstTerm -> vector<frRPin*>
```

然后对每个 pin：

```text
pin2Nodes[pin].size() 必须等于 pin2RPins[pin].size()
```

接着逐个对齐 node 和 rpin：

```text
pt = rpin->getAccessPoint()->getPoint()
layer = rpin->getAccessPoint()->getLayerNum()

node->setLoc(pt)
node->setLayerNum(layer)
```

如果是 instance pin，access point 原本在 master/local 坐标系里，
代码会用 instance transform 转到 design 坐标。

这一步之后，初始 pin nodes 才有真实的 AP 坐标和 layer。

### 5. 按 gcell 聚合 pin nodes

位置：`src/gr/FlexGR.cpp:1467`

对局部 `nodes` 数组里的每个 pin node：

```text
apLoc = node->getLoc()
apGCellIdx = getGCellIdx(apLoc)
gcellIdx2Nodes[apGCellIdx].push_back(node)
```

含义：

```text
同一个 gcell 里的多个 pin/access point 会聚在一起
后面的 topology 不直接在 AP 点之间连
而是在 gcell center 点之间连
```

### 6. 为每个有 pin 的 gcell 创建 gcell-center node

位置：`src/gr/FlexGR.cpp:1480`

对每个 `gcellIdx`：

```text
create frNode
type = frcSteiner
loc = gcell box center
layerNum = 2
```

这些 node 叫 gcell nodes。
它们被追加到 `net->nodes` 后面。

同时记录：

```text
gcellNode2RPinNodes[gcellNode] = local pin nodes in this gcell
```

也就是一个 gcell-center node 对应哪些真实 pin nodes。

代码还会设置：

```text
net->setFirstNonRPinNode(gcellNodes[0])
```

意思是从这里开始，`net->nodes` 里进入非 RPin/pin node 区域。

### 7. 如果只有一个 gcell，直接返回

位置：`src/gr/FlexGR.cpp:1533`

如果所有 pins 都落在同一个 gcell：

```text
gcellNodes.size() <= 1
```

那就不需要生成 gcell 间 Steiner tree。
函数直接返回。

这种情况下已经创建了 gcell-center node，
但不会调用 FLUTE。

### 8. 用 FLUTE 生成 gcell-level Steiner tree

位置：`src/gr/FlexGR.cpp:1544`

当前实际执行的是：

```cpp
genSTTopology_FLUTE(gcellNodes, steinerNodes);
```

代码里还有 MST/HVW 分支，但被包在：

```cpp
if (false) {
  ...
} else {
  genSTTopology_FLUTE(...)
}
```

所以正常不会走 MST/HVW。

`genSTTopology_FLUTE()` 做的事情：

```text
1. 收集所有 gcell node 的 x/y
2. 调 flute::flute()
3. 遍历 FLUTE tree branches
4. 对非 pin/gcell 的 Steiner 点创建新的 frNode
5. net->addNode(steinerNode)
6. 建 adjacency list
7. 从 root gcell node BFS
8. 设置 parent/children
```

FLUTE 创建出来的 Steiner nodes：

```text
type = frcSteiner
loc = FLUTE point
layerNum = 2
```

### 9. 把真实 pin node 接到对应的 gcell node 上

位置：`src/gr/FlexGR.cpp:1660`

前面 FLUTE 只连接了 gcell/Steiner nodes。
这一步把真实 AP pin nodes 接回 topology。

逻辑：

```text
for each gcellNode:
  for each local pin node in this gcell:
    if local pin node is root pin:
      gcellNode parent = root pin node
      root pin node child += gcellNode
    else:
      local pin node parent = gcellNode
      gcellNode child += local pin node
```

也就是说：

```text
root pin
  -> root gcell-center node
    -> Steiner/gcell topology
      -> other gcell-center nodes
        -> sink pin nodes
```

### 10. sanity check

位置：`src/gr/FlexGR.cpp:1673`

检查：

```text
非 root pin node 必须有 parent
root pin node 必须至少有 child
```

## 函数结束后的 net->nodes 长什么样

结束后，`net->nodes` 大致分为几段：

```text
[0 ... #rpins-1]
  初始 pin nodes
  type = frcPin
  loc/layerNum = access point 的真实 x/y/layer

[#rpins ...]
  gcell-center nodes
  type = frcSteiner
  loc = gcell center
  layerNum = 2

[后续追加]
  FLUTE Steiner nodes
  type = frcSteiner
  loc = FLUTE 2D topology point
  layerNum = 2
```

parent/children 已经形成一棵树。

## 后续谁继续改这些 nodes

`initGR_genTopology_net()` 之后还有几个重要阶段会继续处理这些 nodes：

```text
initGR_updateCongestion2D_net()
  根据 2D topology 更新 congestion

initGR_patternRoute()
  对部分 2D 连接做 pattern routing / 修正

initGR_initObj()
  根据 Steiner-to-Steiner 的连接生成初始 grPathSeg

layerAssign()
  把 2D topology 分配到真实 routing layers
  插入需要的 vertical sub-nodes/vias
```

特别是 `layerAssign()`：

```text
它才是真正把 topology 从 2D 推到多层 routing graph 的阶段
```

所以不要把 `initGR_genTopology_net()` 里的 `layerNum = 2`
理解成最终 routing layer。
它只是 2D GR topology 阶段的临时层。

## genTopology_net 之后结果如何存储

`initGR_genTopology_net()` 生成的 topology 没有单独存成一个
`Topology` 或 `Graph` 对象。

它的结果分散存在下面几类数据结构里：

```text
1. 节点本体:
   frNet::nodes

2. 拓扑边:
   frNode::parent
   frNode::children

3. root / 分界指针:
   frNet::root
   frNet::rootGCellNode
   frNet::firstNonRPinNode

4. FlexGR 临时/辅助索引:
   net2GCellIdx2Nodes
   net2GCellNodes
   net2SteinerNodes
   net2GCellNode2RPinNodes
```

### 节点存储在 frNet::nodes

`frNet::nodes` 是：

```cpp
std::list<std::unique_ptr<frNode> > nodes;
```

`initGR_genTopology_net()` 之后，`net->nodes` 大致变成：

```text
net->nodes:
  [原始 pin/rpin nodes]
  [gcell-center nodes]
  [FLUTE Steiner nodes]
```

第一段是 DEF 读入 net connection 时创建的 pin nodes。

第二段是 `initGR_genTopology_net()` 为每个有 pin 的 gcell
创建的 gcell-center nodes。

第三段是 `genSTTopology_FLUTE()` 根据 FLUTE tree 额外创建的
Steiner nodes。

gcell-center node 的加入位置：

```cpp
net->addNode(tmpGCellNodes[rootIdx]);
for (unsigned i = 0; i < tmpGCellNodes.size(); i++) {
  if (i != rootIdx) {
    net->addNode(tmpGCellNodes[i]);
  }
}
```

FLUTE Steiner node 的加入位置在 `genSTTopology_FLUTE()` 中：

```cpp
auto steinerNode = make_unique<frNode>();
steinerNode->setType(frNodeTypeEnum::frcSteiner);
steinerNode->setLoc(bp);
steinerNode->setLayerNum(2);
steinerNodes.push_back(steinerNode.get());
net->addNode(steinerNode);
```

所以所有 topology node 最终都归 `frNet::nodes` 拥有。

### 树边存储在 parent / children

FLUTE 阶段内部会临时建立一个 adjacency list：

```cpp
map<frNode*, set<frNode*, frBlockObjectComp>, frBlockObjectComp> adjacencyList;
```

这个 `adjacencyList` 只是局部变量，不是最终存储。

最终会从 root gcell node 开始 BFS，把无向 adjacency 转成一棵有向树：

```cpp
currNode->addChild(adjNode);
adjNode->setParent(currNode);
```

因此最终拓扑边存在每个 `frNode` 自己身上：

```text
child node:
  parent -> parent frNode*

parent node:
  children -> list<frNode*>
```

所以要遍历 topology，应该从 `net->getRoot()` 或
`net->getRootGCellNode()` 出发，沿着 `getChildren()` 走。

### root / gcell root / firstNonRPinNode

`initGR_genTopology_net()` 会设置几个关键指针。

#### net->root

```cpp
net->setRoot(nodes[0]);
```

`nodes[0]` 是 driver pin node。
也就是说 `net->root` 指向真实 pin/access point node，
不是 gcell-center node。

#### net->rootGCellNode

```cpp
net->setRootGCellNode(gcellNodes[0]);
```

`rootGCellNode` 是 root pin 所在 gcell 的 gcell-center node。
后续 layer assignment 主要从这个 gcell-level root 开始做。

#### net->firstNonRPinNode

```cpp
net->setFirstNonRPinNode(gcellNodes[0]);
```

它用于标记 `net->nodes` 中从哪里开始进入非 pin/rpin node 区域。

因为 `frNet::nodes` 的前面一段是 pin/rpin nodes，
后面才是 gcell-center / Steiner nodes。

### FlexGR 里的辅助索引

`FlexGR` 类里还有几张 map，用来快速找到不同类别的节点。

定义位置：`src/gr/FlexGR.h`

```cpp
std::map<frNet*, std::map<std::pair<int, int>, std::vector<frNode*> >,
         frBlockObjectComp> net2GCellIdx2Nodes;

std::map<frNet*, std::vector<frNode*>, frBlockObjectComp> net2GCellNodes;

std::map<frNet*, std::vector<frNode*>, frBlockObjectComp> net2SteinerNodes;

std::map<frNet*,
         std::map<frNode*, std::vector<frNode*>, frBlockObjectComp>,
         frBlockObjectComp> net2GCellNode2RPinNodes;
```

含义如下。

#### net2GCellIdx2Nodes

```text
net2GCellIdx2Nodes[net][gcellIdx] = pin nodes in this gcell
```

它记录某个 net 在某个 gcell 里有哪些真实 pin/access point nodes。

构建位置：

```cpp
node->getLoc(apLoc);
design->getTopBlock()->getGCellIdx(apLoc, apGCellIdx);
gcellIdx2Nodes[make_pair(apGCellIdx.x(), apGCellIdx.y())].push_back(node);
```

#### net2GCellNodes

```text
net2GCellNodes[net] = this net's gcell-center nodes
```

它保存这个 net 的所有 gcell-center nodes。

注意这些 pointer 指向的 node 本体仍然在 `net->nodes` 里。
这个 vector 只是索引，不拥有 node。

#### net2SteinerNodes

```text
net2SteinerNodes[net] = FLUTE extra Steiner nodes
```

FLUTE 生成非 terminal 点时，会把新 node 加到这里。

同样，node 本体由 `net->nodes` 拥有。

#### net2GCellNode2RPinNodes

```text
net2GCellNode2RPinNodes[net][gcellNode] = local pin/rpin nodes represented by this gcell node
```

它记录一个 gcell-center node 代表哪些真实 pin/access point nodes。

例如：

```text
gcell-center node G34
  represents:
    pin A AP node
    pin B AP node
```

最后把真实 pin node 接到 gcell-center node 时就是用这张 map：

```cpp
for (auto &[gcellNode, localNodes]: gcellNode2RPinNodes) {
  for (auto localNode: localNodes) {
    if (localNode == nodes[0]) {
      gcellNode->setParent(localNode);
      localNode->addChild(gcellNode);
    } else {
      gcellNode->addChild(localNode);
      localNode->setParent(gcellNode);
    }
  }
}
```

### grShape 不是这个函数的主要输出

在当前实际路径中：

```cpp
if (false) {
  ...
} else {
  genSTTopology_FLUTE(gcellNodes, steinerNodes);
}
```

所以 MST/HVW 分支不会执行。

在 FLUTE 路径下，`initGR_genTopology_net()` 主要输出的是：

```text
node objects
parent/children topology
FlexGR 辅助索引
```

它不会在这里完整生成最终 `grPathSeg`。

后面的 `initGR_initObj()` 会根据 parent/children，
为 `frcSteiner -> frcSteiner` 的边创建 `grPathSeg`，
然后存到：

```text
net->grShapes
```

因此：

```text
genTopology_net 后:
  topology tree 已经存在
  但 route shape 对象还不是主要结果

initGR_initObj 后:
  tree 中的 Steiner-Steiner 边会被物化成 grPathSeg
```

## 如何在代码里看这个结果

如果想 debug 某个 net 的 topology，可以从 root 开始遍历：

```cpp
std::deque<frNode*> q;
q.push_back(net->getRoot());

while (!q.empty()) {
  auto node = q.front();
  q.pop_front();

  frPoint loc = node->getLoc();
  std::cout << "node id=" << node->getId()
            << " type=" << (int)node->getType()
            << " loc=(" << loc.x() << "," << loc.y() << ")"
            << " layer=" << node->getLayerNum();

  if (node->getParent()) {
    std::cout << " parent=" << node->getParent()->getId();
  }
  std::cout << "\n";

  for (auto child: node->getChildren()) {
    q.push_back(child);
  }
}
```

如果只想看 gcell-level topology，可以从 `net->getRootGCellNode()`
开始遍历，并跳过 `frcPin` 类型的 node。

## 一句话总结

`initGR_genTopology_net()` 的输入 nodes 是 DEF net connection 阶段创建的 pin nodes。
这个函数用 RPin/access point 给 pin nodes 补真实 x/y/layer，
再按 gcell 聚合，创建 gcell-center nodes，
用 FLUTE 生成 2D Steiner topology，
最后把真实 pin nodes 挂回这棵树。

此时的 topology 主体是 2D 的；
pin 端点带真实 3D access point 信息；
后续 `layerAssign()` 才完成真正的多层分配。
