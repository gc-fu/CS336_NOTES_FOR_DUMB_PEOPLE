# CS336 Lecture 8：Parallelism II——ZeRO / FSDP、PP、TP、SP、EP、CP 与组合策略

> **目标读者：** 只会四则运算、刚知道“多张 GPU 可以一起训练”的初学者。  
> **目标：** 不看视频，也能手算 ZeRO/FSDP 显存与通信，追踪 PP/TP/SP/EP/CP 的 shape 与数据移动，并为一个模型组合并行 axes。  
> **本讲官方 PDF：**[Stanford CS336 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)，73 页，课程标题为 *Parallelism Basics*。  
> **官方视频：**[Stanford Online：Lecture 8](https://www.youtube.com/watch?v=6-cXp-aOmdg)。

## 0. 开始之前：怎么读、五分钟复习卡与资料核验

### 0.1 第一次阅读怎么走

如果这是第一次学本讲：

1. **先跳过 §0.2 的五分钟复习卡。** 那是学完后压缩记忆用的，不是第一次建立概念用的。
2. 先读 §1–§2，弄清楚我们为什么需要多 GPU，以及 collective 到底搬了什么。
3. 再读 §3–§4，把 naive data parallel 的训练和 16 bytes/param 一项一项算明白。
4. 按 §5→§7 读 ZeRO-1、ZeRO-2、ZeRO-3；再按 §9→§19 读 PP、TP、SP、EP、CP。
5. 用 §20 的统一表和 §21–§24 的组合/案例把方法串起来，最后做 §26 自测。
6. 每看到“约 $`2P`$”或“约 $`3P`$”，都先问：这里的 $`P`$ 是参数个数、参数 payload，还是 bytes？正文会逐次说明。

本讲视频开场 [00:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=5s) 承接 Lecture 7 的底层并行机制；[00:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=25s) 把目标定为理解大模型、大集群的复杂性；[00:35](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=35s) 预告多种并行策略需要同时使用；[00:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=50s) 说明最大规模训练往往需要其中大部分；[01:08](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=68s) 预告最后会看真实大训练案例。

### 0.2 五分钟复习卡（首次阅读请跳过）

**一句话主线：**

```text
naive DP 把计算分开，却复制全部模型状态
    ↓
ZeRO-1 只切 optimizer state
    ↓
ZeRO-2 再切 gradient
    ↓
ZeRO-3 / FSDP 连 parameter 也切
    ↓
用按层 all-gather / reduce-scatter 换取显存，并尽量把通信藏在计算后面
```

**十个最重要结论：**

1. **理想 compute scaling（计算扩展）：**$`M`$ 张同样快的 GPU 各做 $`1/M`$ 的工作，纯计算时间理想上降到 $`1/M`$；这只是上界，不包括通信和等待。
2. **Naive data parallel（朴素数据并行）：** 每张 GPU 拿不同样本，但持有完整 parameters、gradients、optimizer state；因此它能分计算，不能把这些静态模型状态除以 GPU 数。
3. 在本讲常用的 16-byte 训练口径中，每参数静态账为：

   $`2+2+4+4+4=16\ \text{bytes/parameter}.`$

4. 若参数和梯度各占 2 bytes，optimizer state 共占 $`K`$ bytes，$`N`$ 张 GPU 的持久静态内存近似为：

   | 方法 | 每 rank 静态 bytes |
   |---|---:|
   | Naive DP | $`(4+K)P`$ |
   | ZeRO-1 | $`(4+K/N)P`$ |
   | ZeRO-2 | $`\left(2+(2+K)/N\right)P`$ |
   | ZeRO-3 | $`((4+K)/N)P`$ |

   这里 $`P`$ 是参数个数；括号里的系数单位是 bytes/parameter。
5. 在课件忽略 ring 的 $`(N-1)/N`$、延迟、临时 buffer 和 overlap 细节的归一化模型里：DDP（Distributed Data Parallel，分布式数据并行）、ZeRO-1、ZeRO-2 都约搬 $`2P`$ 个“参数大小的元素”，ZeRO-3 / FSDP（Fully Sharded Data Parallel，全分片数据并行）约搬 $`3P`$。这不等于真实 wall-clock 一定分别是 2 倍和 3 倍。
6. **PP** 切 layers/depth；microbatch 可填 pipeline，但 bubble 比例必须说明分母。
7. **TP** 切单层 matrix，**SP** 再切 pointwise activation；TP-only 不保证全部 activation 除以 $`t`$。
8. 这里先把 attention 的三个字母说清：$`Q`$ 是 **query（查询）**，$`K`$ 是 **key（键）**，$`V`$ 是 **value（值）**；KV block 就是把一段 tokens 的 K 与 V 向量装成的块。**EP** 分 whole routed experts，**CP** 分 context tokens；前者付 token all-to-all，后者付 KV block exchange。
9. Dense 的正交 $`DP\times TP\times PP`$ 可相乘；MoE 的 TP/EP/ETP/EDP 可能复用或折叠 groups，不能数缩写盲乘。
10. 配置的正确顺序是“先 fit，再量 throughput，最后做 failure recovery”；真实模型表只是课程时点快照，`??` 仍是未知。

**第 28 页必须会背后的计算：** 课件改用另一套 12 bytes/param 口径：bfloat16（常缩写 BF16）parameter 2、bfloat16 gradient 2、FP32（32-bit floating point）master 4、bfloat16 Adam $`m`$ 2、bfloat16 Adam $`v`$ 2。这里 $`m/v`$ 是优化器跨训练步骤保存的两份历史统计。8 张 80 GB A100 上，按十进制容量且不留任何余量：

```text
Baseline: 80 / 12       = 6.667B parameters
ZeRO-1:  80 / 5        = 16B
ZeRO-2:  80 / 3.25     = 24.615B
ZeRO-3:  80 / 1.5      = 53.333B
```

这只是静态模型状态的纸面上限，不含 activation（forward 产生、常为 backward 保存的中间值）、通信 buffer、allocator（负责申请和复用显存的分配器）碎片和安全余量，不能当成部署保证。文中的 **rank** 是参加一次分布式训练的一个 worker 进程编号。

### 0.3 全讲因果链

**【课程内容｜PDF 1–3、13–14 页】【视频补充】** 老师在 [01:22](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=82s) 说明前半先复习 collective 与 networking，在 [11:07](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=667s) 转入并行算法主体。

完整课程的逻辑不是“记住一堆并行缩写”，而是连续回答四个问题：

```text
问题 1：一张 GPU 的 compute 或 memory 不够，怎么办？
问题 2：多张 GPU 之间允许怎样交换数据？
问题 3：parameter / gradient / optimizer state 哪些必须复制，哪些可以 shard？
问题 4：静态模型状态解决后，activation、batch size、网络拓扑又带来什么限制？
```

§0–§7 回答前三问；§8–§24 继续回答 activation、model parallel、组合拓扑、真实配置与可靠训练。

### 0.4 来源标签怎么读

- **【课程内容】**：官方 PDF 中明确出现的公式、图、表或结论。
- **【视频补充】**：人工英文字幕中的口头解释、课堂提问或限定条件。
- **【补充理解】**：为了让零基础读者能复算而补的中间步骤、单位和小例子。
- **【延伸】**：来自论文或官方文档的实现边界；不是课程逐字原话。

主要一手资料：

- [ZeRO 原论文](https://arxiv.org/abs/1910.02054)：三阶段 sharding 的原始内存与通信分析。
- [PyTorch 官方 FSDP 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html)：forward/backward 中 all-gather、reduce-scatter 与释放参数的工作流。
- [Google Cloud TPU v4 官方拓扑文档](https://docs.cloud.google.com/tpu/docs/v4)：3D mesh/torus、wrap-around 和拓扑映射。
- [NVIDIA 官方 NVLink/NVSwitch 拓扑说明](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)：GPU、NVLink、NVSwitch 关系及“拓扑不等于实测带宽”的边界。

### 0.5 PDF 与字幕核验记录

**PDF：**

- 本地文件 `work/pdfs/lecture_08.pdf`，pypdf 与 pypdfium2 都确认 **73 页**。
- 全部 73/73 页以 pypdfium2 渲染，页面尺寸均为 720×405.36 PDF points。
- 生成了 8 张 contact sheets，逐页检查了 1–73 页；没有空白丢页、黑页、裁切或页序异常。
- 第 7–73 页全部另以 2.5 倍比例渲染，共 67 张高分辨率图；其中公式、时间表、配置表与故障表逐张以原图细节检查。
- 关键视觉结论：第 8 页的 “Reduce” 文案与图示 all-reduce 语义存在歧义；第 18 页是 ZeRO 三级内存总表；第 23 页用单个 reduce 图标表现一个 shard 发给 owner；第 25 页画出 forward/backward 两次 all-gather；第 26 页区分 GPU compute stream 与 communication stream；第 28 页表采用 12 bytes/param，而不是第 17 页的 16 bytes/param；第 34 页的 bubble ratio 以 useful time 为分母；第 37 页说明更复杂的交错 schedule 会增加带宽需求；第 38 页把 backward 拆成 $`dX`$ 与 $`dW`$；第 43 页给出 TP 与 PP 的课程通信近似式；第 46 页公式是 $`sbh(34+5as/h)`$；第 47 页 TP 公式是 $`sbh(10+24/t+5as/(ht))`$；第 49 页 TP+SP 公式是 $`sbh(34/t+5as/(ht))`$；第 53 页把 attention 的 TP/CP/DP 与 MoE 的 ETP/EP/EDP 分开；第 54 页只给 CP/Ring Attention 高层图；第 56 页的四行FLOPs/bytes表另以6倍分辨率复读；第 59 页十行配置表和第 67 页failure表逐格复算。

**字幕：**

- 轨道：`English (United States)`，语言码 `en-US`，`kind` 为空，因此是人工字幕，不是 `asr` 自动轨。
- 共 **1870 segments**。
- 最后一段从 **80:01** 开始，到约 **80:05** 结束，文本是下一讲将谈 scaling laws。

### 0.6 73 页内容地图

这张表用于证明每页已经进入视觉巡检范围；后续正文按教学因果链重排，不机械照页翻译。

| PDF 页 | 视觉内容 | 本讲笔记位置 |
|---:|---|---|
| 1–3 | 标题、目标、三部分组织 | §0 |
| 4–6 | 单 GPU compute/memory 限制，多机图 | §1 |
| 7–8 | collective 与 all-reduce 分解 | §2 |
| 9–12 | TPU mesh、GPU switched/tree、TPU8、domain size | §2 |
| 13–14 | 第一部分回顾、并行原语地图 | §0–§2 |
| 15–16 | naive data parallel | §3 |
| 17–18 | 16 bytes/param 与 ZeRO 总览 | §4 |
| 19–21 | ZeRO-1 | §5 |
| 22–23 | ZeRO-2 | §6 |
| 24–28 | ZeRO-3/FSDP、overlap、fit 表 | §7 |
| 29–31 | DP 剩余限制、model parallel 入口 | §8 |
| 32–38 | layer-wise、pipeline schedule、zero-bubble | §9–§10 |
| 39–43 | tensor parallel 及其与 pipeline 的通信比较 | §11–§13 |
| 44–49 | dynamic activation、重计算、sequence parallel | §14–§16 |
| 50–53 | expert parallel 及 attention/MoE parallel folding | §17–§18 |
| 54–55 | context/ring attention 与并行策略回顾 | §19 |
| 56–62 | 全策略比较、3D/4D 组合与 scaling 图 | §20–§22、§24 |
| 63–72 | OLMo/Dolma、DeepSeek、Yi、Llama、Gemma、Mixtral、Nemotron、Qwen 案例 | §23–§24 |
| 73 | 全讲回顾 | §30 |

### 0.7 最少前置知识

开始正文只需要会下面几件事：

1. 加、减、乘、除；例如 $`80/5=16`$。
2. 知道 $`[1,2,3]`$ 是三个数排成的一维 tensor，逐元素相加就是对应位置相加。
3. 知道 $`1\ \text{byte}=8\ \text{bits}`$；不会 GB/GiB 换算也没关系，§4.4 会从定义重算。
4. 知道训练大致按 `forward → loss → backward → optimizer update` 进行。正文会在术语第一次进入主线时重新解释。
5. Lecture 7 的 collective 可以完全忘掉；§2 会用四个小向量从头重建。

---

## 1. 为什么要从一张 GPU 走向多 GPU、多机器

### 1.1 两堵墙：compute 与 memory

**【课程内容｜PDF 4–6 页】【视频补充｜[01:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=90s)】** 训练大模型遇到两类不同的“不够”：

1. **Compute（计算量/算力）不够。** 单张 GPU 每秒只能完成有限次浮点运算。视频 [01:36](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=96s) 说，需要的计算超过单 chip 能提供的量，所以把许多机器连起来。
2. **Memory（显存容量）不够。** 模型和训练状态的 bytes 超过单张 GPU 能放下的量。视频 [01:55](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=115s) 把“模型装不下”列为第二个原因。

这两堵墙不能混为一谈：

- 一个模型能放下，但训练一轮要一年，是 **compute 问题**。
- 一个模型只需一秒算完，但需要 120 GB、GPU 只有 80 GB，是 **memory 问题**。
- 现实常常两者同时存在。

### 1.2 “线性计算扩展”到底是什么意思

**【课程内容｜PDF 13 页】** 设一张 GPU 每秒做 $`C`$ 次有效计算，$`M`$ 张相同 GPU 的理想总算力为：

```math
C_{\text{ideal,total}}=M\times C.
```

符号逐个解释：

- $`C`$：一张 GPU 的有效计算吞吐，单位可写 FLOP/s；FLOP 是一次浮点运算，FLOP/s 是每秒多少次。
- $`M`$：GPU 数量，没有单位。
- $`C_{\text{ideal,total}}`$：理想总吞吐，单位仍是 FLOP/s。

**四则运算例：** 一张 GPU 每秒做 100 个训练工作单位，4 张的理想总量为：

```math
4\times100=400\ \text{units/s}.
```

如果固定总工作为 800 units：

- 1 张耗时 $`800/100=8`$ 秒；
- 4 张理想耗时 $`800/400=2`$ 秒；
- 理想 speedup（加速倍数）为 $`8/2=4`$。

**Kernel（计算核）** 是一次在GPU上执行某类底层计算的程序，例如一次矩阵乘或归约。**“理想”不等于承诺。** 实际还要扣掉通信、同步、负载不均、kernel 效率下降、故障和尾部等待。$`M`$ 张 GPU 得到的实际加速通常小于 $`M`$。

### 1.3 “线性显存扩展”到底是什么意思

**【课程内容｜PDF 13 页】** 如果所有必须长期保存的状态都能平均 shard（切片）到 $`M`$ 张 GPU，理想上每张只存 $`1/M`$：

```math
\text{memory per rank}=\frac{S}{M}.
```

- $`S`$：单副本总状态大小，单位 bytes。
- $`M`$：GPU/rank 数。
- $`S/M`$：每 rank 理想持有 bytes。

例：一个静态状态共 64 GB，8 张卡平均分：

```math
64/8=8\ \text{GB per rank}.
```

反过来，如果每张卡都能给模型状态 80 GB，8 张的纸面容量总和是：

```math
8\times80=640\ \text{GB}.
```

但这只在“状态可均匀切分、没有复制、没有临时峰值”时成立。activation、通信 buffer、allocator 预留和碎片会让实际可用量更小。

### 1.4 Node、intra-node、inter-node

**【课程内容｜PDF 6 页】【视频补充｜[02:16](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=136s)】**

- **Node（节点）**：一台服务器，里面可能有 8 张 GPU。
- **Intra-node（节点内）**：同一台服务器内的 GPU 通信，通常有较快 NVLink/NVSwitch 或 PCIe 路径。
- **Inter-node（节点间）**：跨服务器通信，通常经过 NIC/HCA 与数据中心网络，路径和延迟更复杂。

视频 [02:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=145s) 强调 inter-node 较慢，因此并行算法要尊重链路约束。人话理解：

```text
同一个房间里递纸条，通常比跨楼寄快递便宜。
所以“每层都聊天”的并行，优先放在快连接域；
“偶尔传一次大包”的并行，才更可能跨慢连接。
```

### 1.5 本讲追求的是两个理想，不是一个口号

```text
更多 GPU
├─ 希望 compute 总量随 GPU 数近似线性增长
└─ 希望可容纳的模型状态随 GPU 数近似线性增长
```

Naive data parallel 主要解决前者。ZeRO 的三阶段逐步修补后者。

---

## 2. Collective 与 topology 复习：数据到底怎样走

### 2.1 Collective 是“一群 ranks 共同调用的通信动作”

**【课程内容｜PDF 7 页】【视频补充｜[02:44](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=164s)】** 本讲不逐包分析网络，而在 collective communication primitive（集合通信原语）层面记账。

- **Rank**：一个分布式进程在某个通信组里的编号。
- **Collective**：组内多个 ranks 必须以匹配方式参与的通信操作。
- **Primitive（原语）**：上层算法拿来组合的基本积木。

视频 [02:53](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=173s) 举例：算法会说“做一次 all-reduce”或“做一次 all-gather”，而不是描述每个 packet。

### 2.2 五个最小 collective 的输入输出

用 4 ranks，每个 rank 起初有一个数：

| rank | 本地输入 |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |

1. **Broadcast（广播）**，root=2：把 rank 2 的 3 发给所有人。结果 `[3,3,3,3]`。
2. **Reduce（归约）**，SUM、root=2：$`1+2+3+4=10`$，只有 root 必须得到 10；其他 rank 的输出不作同样保证。
3. **All-reduce（全归约）**，SUM：先求和 10，再让每个 rank 都得到 10；结果 `[10,10,10,10]`。
4. **All-gather（全收集）**：不求和，把四块按 rank 顺序拼起来；每个 rank 都得到 `[1,2,3,4]`。
5. **Reduce-scatter（归约散发）**：每个 rank 先提供一整个、可分成 4 块的向量；对应块逐元素 reduce，再让每个 rank 只保留其中一块。

后面 ZeRO 最常用的是 all-reduce、reduce-scatter、all-gather。

### 2.3 手算 all-reduce = reduce-scatter + all-gather 的逻辑结果

**【课程内容｜PDF 8 页】【视频补充｜[03:14](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=194s)】** 设四个 rank 各有长度 4 的向量：

| rank | 输入向量 |
|---:|---|
| 0 | `[1, 10, 100, 1000]` |
| 1 | `[2, 20, 200, 2000]` |
| 2 | `[3, 30, 300, 3000]` |
| 3 | `[4, 40, 400, 4000]` |

先逐列求和：

```math
1+2+3+4=10,
```

```math
10+20+30+40=100,
```

```math
100+200+300+400=1000,
```

```math
1000+2000+3000+4000=10000.
```

Reduce-scatter 之后：

| rank | 只保留的归约 shard |
|---:|---:|
| 0 | `[10]` |
| 1 | `[100]` |
| 2 | `[1000]` |
| 3 | `[10000]` |

再 all-gather，每个 rank 都拼回：

```text
[10, 100, 1000, 10000]
```

这和直接 all-reduce 的**逻辑输出**完全相同。视频 [03:21](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=201s) 点出这个等价；[03:31](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=211s) 说，在特定带宽记账下，两边成本相当，因此后面能“换一种顺序”而不增加主导通信量。

### 2.4 PDF 第 8 页的 “Reduce” 是简写/歧义

高分辨率视觉核验显示，第 8 页正文写：

```text
Reduce can be implemented as two steps: reduce-scatter and all-gather
```

但同页图的左侧标题是 **All Reduce**，最终每张 GPU 都持有完整 $`A+B+C+D`$。所以这里应读作：

> **All-reduce 可以在逻辑上分解为 reduce-scatter 加 all-gather。**

不能把它推广为“普通 reduce 等于 reduce-scatter+all-gather”：普通 reduce 只要求 root 得到结果；再做 all-gather 会让所有 ranks 都得到结果，语义已经变成 all-reduce。

课堂在 [03:44](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=224s) 结束基础复习；学生在 [04:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=243s) 问为何强调这一种分解，老师在 [04:11](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=251s) 回答：还有别的分解，但这一种会直接服务后面的算法。

### 2.5 逻辑等价不锁死物理算法

```text
逻辑层：调用者想要什么输入/输出？
实现层：ring、tree、分层算法或硬件 collective 怎样搬？
物理层：数据实际经过哪些 links/switches？
```

“All-reduce = RS + AG”在这里首先是逻辑与带宽记账工具，不表示每个 backend 永远严格执行两个可见 kernel，也不表示所有 topology 的 latency 都相同。

### 2.6 Topology 是“谁能沿什么路径和谁通信”

**Topology（拓扑）** 描述 devices 与 switches 的连接关系。连接图会影响：

- 一条消息要经过多少 hops（跳）；
- 多组通信能否并发；
- 中间链路是否成为瓶颈；
- 哪些 ranks 应放进同一个高带宽并行 group。

视频在 [04:48](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=288s) 开始比较 TPU/GPU；这里的硬件描述和数字都应当视为 **Spring 2026 课程时点快照**，不能写成未来所有 TPU/GPU 的永久结构。

### 2.7 TPU mesh / torus：固定邻居，边界绕回

**【课程内容｜PDF 9 页】【视频补充｜[05:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=300s)】**

- **Mesh（网格）**：chip 通常直接连附近邻居。
- **Torus（环面网络）**：网格的一边还会 wrap around（绕回）连接另一边。
- **Toroidal mesh**：把这种绕回结构推广到多维。

先看 2×2 的极小示意：

```text
(0,0) ── (0,1)
  │          │
(1,0) ── (1,1)
```

在更大的 torus 中，最左和最右、最上和最下也有 wrap-around link。视频 [05:32](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=332s) 说 chip 连接邻居且边界绕回。重要性质是：网络变大时，每个 chip 的直接邻居数仍可保持近似固定，不等于每个 chip 直接连所有 chip。

Google 的 TPU v4 官方文档进一步说明其 3D mesh 可以在特定 slice shape 配成 3D torus；这是对课件简图的官方边界补充，不代表每代 TPU 完全相同。

### 2.8 GPU switched/fat-tree 与 all-to-all 的含义

**【课程内容｜PDF 9–10 页】【视频补充｜[05:48](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=348s)】** 视频用一种对比心智模型：GPU 系统更偏 switched/fat-tree 风格；[05:54](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=354s) 解释底层 GPU 先高速互连，pods 再通过 leaf/spine switches 连接。

- **Switch（交换机）**：在多个端口之间转发数据的设备。
- **Fat tree（胖树）**：越往上可能配置越多聚合带宽，避免树根过细。
- **All-to-all（A2A，全互换）**：每个 rank 可向每个其他 rank 发送不同数据的 collective traffic pattern。

“支持 all-to-all”不等于任意两张 GPU 都有一根专属直连线；数据通常仍经过 NVSwitch、leaf/spine 或别的交换层。

视频 [06:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=385s) 给出课程直觉：规则、邻居型通信可很好映射 mesh；[06:36](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=396s) 说随机、不可预测的路由更需要灵活 switched fabric。老师又在 [07:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=430s) 用 MoE token 路由举例；这是一种 workload/topology 匹配直觉，不是“TPU 不能跑 MoE”或“GPU 永远更好”的定律。

### 2.9 TPU8i/t 与架构演化：只按课程时点理解

**【课程内容｜PDF 11 页】【视频补充】** 老师在 [07:37](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=457s) 提到当天发布的 TPU8i/t；[07:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=466s) 把 TPU8i 的图解释为更接近 tree；[08:14](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=494s) 又说训练用 TPU8t 的跨 rack 网络更像 switched fabric，[08:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=505s) 点名 Virgo 网络。

这些是课程发布当天的解读，PDF 自己还写了 “maybe for MoEs?”。正确记法是：

> 网络会随 workload 演化；不要把“TPU=torus、GPU=tree”背成永恒二分法。

### 2.10 Domain size：为什么不把所有 chip 都放进一个最快域

**【课程内容｜PDF 12 页】**

**Domain（域）** 在这里指能通过某一类互连、以某种性能边界直接参与通信的一组 accelerators。**Domain size** 就是组内 accelerator 数。

更大高速域有好处：更多 ranks 可用高带宽通信。代价也会增加：

- switch、fiber、功耗、布线与可靠性成本；
- 更复杂的路由与拥塞控制；
- 为极端通信模式付出的硬件成本可能浪费在普通 workload 上。

视频 [09:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=544s) 提问“为何不把所有东西都高速全连”；[09:27](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=567s) 用大量光交换连接更多 chip；[09:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=589s) 转向功耗代价。PDF 12 页表里的厂商型号、带宽和功耗是课程引用的 2026 快照，不把它们写成经过本笔记逐项实测的永久事实。

视频 [10:34](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=634s) 总结“新的计算单位是整个 data center”；[10:43](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=643s) 把目标说成同时控制 memory 与 compute。

---

## 3. Naive data parallel SGD：先把 8 个样本完整算一遍

### 3.1 术语先翻成人话

**【课程内容｜PDF 15–16 页】【视频补充｜[12:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=721s)】**

- **Data parallelism（数据并行，DP）**：每个 rank 跑同一份模型，但吃不同数据。
- **Naive（朴素）**：最直接、还没做状态 sharding 的版本。
- **SGD（Stochastic Gradient Descent，随机梯度下降）**：沿着让 loss 下降的方向更新参数。
- **Loss（损失）**：一个“预测有多坏”的数，越小越好。
- **Gradient（梯度）**：参数稍微改变时，loss 局部怎样改变。这里只需把它当成每个样本算出的更新建议。
- **Batch（批次）**：一次更新共同使用的一组样本。

视频 [12:11](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=731s) 暂时忽略 Adam，先用 naive SGD；[12:16](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=736s) 定义 batch size $`B`$。

### 3.2 课件更新公式逐符号解释

课件写：

```math
\theta_{t+1}=\theta_t-\eta\sum_{i=1}^{B}\nabla f(x_i).
```

- $`\theta_t`$：第 $`t`$ 步更新前的参数；可以先想成一个数。
- $`\theta_{t+1}`$：更新后的参数。
- $`\eta`$：learning rate（学习率），决定走多大一步。
- $`B`$：全局 batch 里的样本数。
- $`i`$：样本编号，从 1 数到 $`B`$。
- $`x_i`$：第 $`i`$ 个样本。
- $`f(x_i)`$：该样本对应的 loss。
- $`\nabla f(x_i)`$：该样本对参数的 gradient。
- $`\sum`$：把所有样本的 gradient 加起来。

课件公式使用 **sum convention（求和口径）**，没有写 $`1/B`$。很多框架的 loss 默认取平均，对应 **mean convention（平均口径）**：

```math
\theta_{t+1}=\theta_t-\eta_{\text{mean}}\frac{1}{B}
\sum_{i=1}^{B}\nabla f(x_i).
```

二者不是互相矛盾；只要学习率口径配套即可。

### 3.3 $`B=8,M=4`$：每张卡分到哪些样本

视频 [12:27](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=747s) 开始把 batch 切开；[12:31](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=751s) 说 $`B`$ 个样本分到 $`M`$ 台机器。这里取：

```math
B=8,\qquad M=4,\qquad B/M=8/4=2.
```

每 rank 两个样本。假设它们算出的标量 gradients 是：

| rank | 两个样本的 gradients | local sum | local mean |
|---:|---|---:|---:|
| 0 | 1, 3 | $`1+3=4`$ | $`4/2=2`$ |
| 1 | 2, 4 | $`2+4=6`$ | $`6/2=3`$ |
| 2 | 5, 7 | $`5+7=12`$ | $`12/2=6`$ |
| 3 | 6, 8 | $`6+8=14`$ | $`14/2=7`$ |

### 3.4 用 local sums 得到 global sum / mean

视频 [12:39](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=759s) 说每台机器算 gradient，再同步得到总和。

四个 local sums all-reduce SUM：

```math
4+6+12+14=36.
```

这等于逐样本直接加：

```math
1+3+2+4+5+7+6+8=36.
```

全局平均 gradient 为：

```math
36/B=36/8=4.5.
```

所以可采用两条等价路径：

```text
路径 A：all-reduce local sums，得到 36，再除全局 B=8 → 4.5
路径 B：all-reduce local means，得到 2+3+6+7=18，再除 ranks M=4 → 4.5
```

这个等价依赖每 rank 的 local batch 大小相同。若 rank 0 有 1 个样本、rank 1 有 3 个样本，简单平均两个 local means 会给小 batch 过高权重，必须按样本数加权。

### 3.5 真正更新一次参数

设更新前参数：

```math
\theta_t=10,
```

平均口径学习率：

```math
\eta_{\text{mean}}=0.1.
```

那么：

```math
\theta_{t+1}=10-0.1\times4.5.
```

先乘：

```math
0.1\times4.5=0.45.
```

再减：

```math
10-0.45=9.55.
```

若坚持使用课件的 sum 36，又想得到相同更新，应把 sum 口径学习率设成：

```math
\eta_{\text{sum}}=\eta_{\text{mean}}/B=0.1/8=0.0125.
```

验证：

```math
10-0.0125\times36=10-0.45=9.55.
```

所以看到“SUM 还是 AVG”时，不能脱离 loss reduction 与 learning-rate convention 单独判断对错。

### 3.6 Compute 为什么理想扩展，communication 为什么约 $`2P`$

视频 [12:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=767s) 把 compute scaling 称为理想：每张 GPU 只算 $`B/M`$ 个样本。这里每卡算 2 个，而单卡原本算 8 个，纯样本计算量降为：

```math
2/8=1/4.
```

但每个参数都有一个 gradient，需要在 ranks 间同步。令：

- $`P`$：参数个数；也就是 gradient 元素个数。
- “$`P`$ payload”：“一个完整参数/gradient 向量那么多元素”的归一化通信量。

在大消息、带宽主导、ring 模型下，每 rank all-reduce 的发送量更精确是：

```math
2\frac{M-1}{M}P.
```

若 $`M`$ 很大，$`(M-1)/M`$ 接近 1，所以课件写约：

```math
2P.
```

当 $`M=4`$：

```math
2\times\frac{4-1}{4}P
=2\times\frac34P
=1.5P.
```

因此“$`2P`$”是忽略有限 rank 修正的课程量级口径，不是 4-rank 精确字节数。视频 [12:52](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=772s) 开始讨论每 batch 的通信；后面 ZeRO 对比会统一使用同一口径。

### 3.7 为什么 naive DP 不降低静态模型状态显存

视频 [13:07](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=787s) 说 memory scaling 为零，因为每张 GPU 都有同一模型副本；[13:21](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=801s) 转入显存问题。

每张 GPU 都要持有：

```text
完整 parameters
+ 完整 gradients
+ 完整 optimizer state
```

如果这三项单卡共 16 GB：

| GPU 数 | 集群合计复制量 | 每张 GPU 仍需 |
|---:|---:|---:|
| 1 | 16 GB | 16 GB |
| 2 | 32 GB | 16 GB |
| 4 | 64 GB | 16 GB |

GPU 越多，**集群里副本越多**，但单卡静态模型状态没有从 16 GB 变 4 GB。

一个细微边界：若固定 global batch，local batch 从 $`B`$ 降成 $`B/M`$，某些 activation 确实会随 local batch 降低；但本节说“不省 memory”主要指 parameters/gradients/optimizer state 没有 shard。后面还会单独处理 activation memory。

---

## 4. 为什么训练会到 16 bytes/parameter：五份状态逐项相加

### 4.1 Parameter、gradient、master weight、moment 分别是什么

**【课程内容｜PDF 17–18 页】【视频补充｜[13:33](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=813s)】** 训练不能只保存“模型参数”。第一次出现的五类状态如下：

1. **Parameter（参数）**：模型当前用于 forward 的权重。
2. **Gradient（梯度）**：backward 算出的更新方向；通常每个 parameter 对应一个 gradient 元素。
3. **Master weight（主权重）**：一份更高精度参数副本，optimizer 在它上面累积小更新，再转换成低精度 parameter 使用。
4. **First moment $`m`$（一阶矩）**：Adam 保存的、类似历史 gradients 指数移动平均的状态。
5. **Second moment $`v`$（二阶矩）**：Adam 保存的、类似历史 gradient squares 指数移动平均的状态。

**Optimizer（优化器）** 是根据 gradient 更新 parameter 的算法。**Optimizer state（优化器状态）** 是它跨 step 记住的辅助数据。本讲 16-byte 口径把 FP32 master weight、Adam $`m`$、Adam $`v`$ 都计入 optimizer state。

视频 [13:55](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=835s) 开始给经验内存口径；[14:48](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=888s) 把 accumulator、first moment、second moment 统称 optimizer state。

### 4.2 BF16、FP16、FP32 的“16”与“32”是什么意思

- **FP32（32-bit floating point，32 位浮点）**：每元素 32 bits。
- **FP16（16-bit floating point，16 位浮点）**：每元素 16 bits。
- **bfloat16（常缩写 BF16）**：每元素 16 bits，但 exponent/fraction 的分配不同；名称来自 *Brain Floating Point*，这里不把它生硬翻译成中文术语。

因为：

```math
8\ \text{bits}=1\ \text{byte},
```

所以：

```math
16\ \text{bits}/8=2\ \text{bytes},
```

```math
32\ \text{bits}/8=4\ \text{bytes}.
```

视频 [14:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=843s) 给出约 16 bytes/parameter；[14:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=850s) 说明还要给 gradients 留位置；[14:21](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=861s) 讲更高精度 accumulator；[14:29](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=869s) 转入 Adam 的两个 moments。

### 4.3 16 bytes 的完整加法

**【课程内容｜PDF 17 页】** 本节采用课件的高精度 Adam 状态口径：

| 每个参数对应的状态 | dtype | bytes/parameter |
|---|---|---:|
| 模型 parameter | BF16/FP16 | 2 |
| gradient | BF16/FP16 | 2 |
| master weight | FP32 | 4 |
| Adam first moment $`m`$ | FP32 | 4 |
| Adam second moment $`v`$ | FP32 | 4 |
| **合计** |  | **16** |

一步一步加：

```math
2+2=4,
```

```math
4+4=8,
```

```math
8+4=12,
```

```math
12+4=16\ \text{bytes/parameter}.
```

课件把 optimizer state 记成 $`K`$ bytes/parameter。在这套口径：

```math
K=4+4+4=12.
```

因此 naive DP 每 rank 静态状态为：

```math
(2+2+K)P=(4+K)P=(4+12)P=16P\ \text{bytes}.
```

- $`P`$：参数个数。
- 乘号前的 16：每参数 16 bytes。
- 所以 $`16P`$ 的单位是 bytes，不是“16 个参数”。

### 4.4 $`P=1`$ billion：GB 与 GiB 都算一次

**Billion（十亿）**：

```math
1\ \text{billion}=1{,}000{,}000{,}000=10^9.
```

**GB（十进制 gigabyte）**：

```math
1\ \text{GB}=1{,}000{,}000{,}000\ \text{bytes}.
```

**GiB（二进制 gibibyte）**：

```math
1\ \text{GiB}=2^{30}=1{,}073{,}741{,}824\ \text{bytes}.
```

令 $`P=1{,}000{,}000{,}000`$：

| 状态 | 计算 | 十进制大小 |
|---|---:|---:|
| BF16 parameter | $`2\times10^9`$ bytes | 2 GB |
| BF16 gradient | $`2\times10^9`$ bytes | 2 GB |
| FP32 master | $`4\times10^9`$ bytes | 4 GB |
| FP32 $`m`$ | $`4\times10^9`$ bytes | 4 GB |
| FP32 $`v`$ | $`4\times10^9`$ bytes | 4 GB |
| **总计** | $`16\times10^9`$ bytes | **16 GB** |

换成 GiB：

```math
16{,}000{,}000{,}000/1{,}073{,}741{,}824
\approx14.901\ \text{GiB}.
```

所以“1B 参数 × 16 bytes = 16 GB”没有错；如果系统工具用 GiB 显示，就会看到约 14.9 GiB。不要把单位差误判成模型少了状态。

### 4.5 这 16 bytes 没算什么

**【补充理解】** 这只是一个静态、理想的模型状态账，没有包括：

- **Activation（激活）**：forward 为 backward 保留的中间 tensor；
- 临时 **GEMM（General Matrix Multiply，通用稠密矩阵乘）**/attention **workspace（计算期间临时使用的显存工作区）**；
- collective 的 send/receive buffer；
- allocator（显存分配器）预留和 fragmentation（碎片）；
- **CUDA**（NVIDIA GPU编程与运行时平台）context、kernel library 与通信库自身占用；
- **NCCL（NVIDIA Collective Communications Library）** 的通信context；它负责GPU collective，但具体算法仍由版本和拓扑选择；
- gradient accumulation、loss、embedding 临时输出等额外 tensor。

因此：

```text
模型状态纸面占 79 GB
≠
一定能塞进 80 GB GPU
```

视频 [14:58](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=898s) 说明 optimizer state 常是更新内存的大头；[15:23](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=923s) 再用颜色图展示绿色 state、橙色 gradient、蓝色 parameter；[15:39](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=939s) 指出 naive DP 会在每张 GPU 复制这些状态。

### 4.6 p18 的完整 $`K=12,\Psi=7.5B,N_d=64`$ 显存表

**【课程内容｜PDF p18；高分辨率逐格核验】** 课件这里用：

- $`K=12`$：每参数 optimizer-state bytes；
- $`\Psi=7.5B=7.5\times10^9`$：模型参数数；
- $`N_d=64`$：data-parallel ranks 数；
- parameter 与 gradient 各2 bytes/parameter。

课件用十进制GB，所以“每参数bytes × 参数B数”可直接得到GB。

**Baseline：**

```math
(2+2+K)\Psi=(2+2+12)\times7.5=16\times7.5=120\ \text{GB}.
```

**ZeRO-1，记为 $`P_{os}`$：**

```math
2\Psi+2\Psi+\frac{K\Psi}{N_d}
=4\times7.5+\frac{12\times7.5}{64}.
```

```math
=30+\frac{90}{64}
=30+1.40625
=31.40625\ \text{GB}.
```

课件显示为31.4GB。

**ZeRO-2，记为 $`P_{os+g}`$：**

```math
2\Psi+\frac{(2+K)\Psi}{N_d}
=2\times7.5+\frac{14\times7.5}{64}.
```

```math
=15+\frac{105}{64}
=15+1.640625
=16.640625\ \text{GB}.
```

课件显示为16.6GB。

**ZeRO-3，记为 $`P_{os+g+p}`$：**

```math
\frac{(2+2+K)\Psi}{N_d}
=\frac{16\times7.5}{64}
=\frac{120}{64}
=1.875\ \text{GB}.
```

课件显示为1.9GB。四个精确教学结果是：

| 方法 | 每rank静态状态 |
|---|---:|
| baseline | 120GB |
| ZeRO-1 | 31.40625GB |
| ZeRO-2 | 16.640625GB |
| ZeRO-3 | 1.875GB |

它们仍不含activation、workspace、communication buffers和安全余量。

### 4.7 ZeRO 到底缩写什么、做什么

**ZeRO（Zero Redundancy Optimizer，零冗余优化器）** 的核心不是把数据删掉，而是：

> 同一份训练状态不必在每个 data-parallel rank 都长期保留完整副本；把它切成 shards，让不同 rank 各自负责一部分，需要完整值时再通信。

**Replicate（复制）**：每 rank 都有完整副本。  
**Shard（切分）**：每 rank 只长期持有一部分。  
**Owner（所有者）**：负责长期保存并更新某个 shard 的 rank。

视频 [15:55](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=955s) 开始提出 shard optimizer state；[16:09](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=969s) 再问能否 shard gradients；[16:13](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=973s) 把极端情况推进到 shard everything。

---

## 5. ZeRO Stage 1：只 shard optimizer state

### 5.1 哪些复制，哪些切分

**【课程内容｜PDF 19–21 页】【视频补充｜[16:48](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1008s)】**

| 状态 | ZeRO-1 是否每 rank 完整保存？ |
|---|---|
| Parameters | 是，replicated |
| Gradients | 是，replicated；但通信中可先 reduce-scatter 给 owner |
| Optimizer state | 否，sharded |

视频 [16:57](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1017s) 明确“只 shard optimizer state”；[17:06](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1026s) 说每个 worker 负责更新一个参数切片。

### 5.2 两 ranks、四 parameters：先画所有权表

设完整参数：

```math
\theta=[10,20,30,40].
```

两个 ranks 平均负责：

| 参数索引 | 初值 | Owner | 谁长期保存对应 master/$`m`$/$`v`$ |
|---:|---:|---:|---|
| 0 | 10 | rank 0 | rank 0 |
| 1 | 20 | rank 0 | rank 0 |
| 2 | 30 | rank 1 | rank 1 |
| 3 | 40 | rank 1 | rank 1 |

但 parameters 与本步 local gradients 仍暂时完整：

| rank | 完整 parameter 副本 | 本地数据算出的完整 local gradient |
|---:|---|---|
| 0 | `[10,20,30,40]` | `[1,2,3,4]` |
| 1 | `[10,20,30,40]` | `[5,6,7,8]` |

### 5.3 Step 1–2：计算 gradient，再 reduce-scatter 给 owner

**【课程内容｜PDF 20 页】【视频补充】** 视频 [17:23](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1043s) 开始四步流程；[17:33](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1053s) 进入 gradient reduce-scatter；[17:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1066s) 说明每个 worker 只需拿与自己负责参数对应的 gradient。

先逐元素 SUM：

```math
[1,2,3,4]+[5,6,7,8]=[6,8,10,12].
```

Reduce-scatter 按 owner 分：

| rank | 得到的 SUM gradient shard |
|---:|---|
| 0 | `[6,8]`，对应参数 0、1 |
| 1 | `[10,12]`，对应参数 2、3 |

若训练目标使用两个 ranks 等权平均，还要除以 2：

```math
[6,8]/2=[3,4],
```

```math
[10,12]/2=[5,6].
```

因此 owner 最终使用的平均 gradients 是：

| owner | gradient shard |
|---:|---|
| rank 0 | `[3,4]` |
| rank 1 | `[5,6]` |

### 5.4 Step 3：每个 owner 只更新自己的参数

为了只展示所有权，暂用 SGD，学习率 $`\eta=0.1`$。

Rank 0：

```math
10-0.1\times3=10-0.3=9.7,
```

```math
20-0.1\times4=20-0.4=19.6.
```

Rank 1：

```math
30-0.1\times5=30-0.5=29.5,
```

```math
40-0.1\times6=40-0.6=39.4.
```

更新后两个 owner 各自拥有：

```text
rank 0: [9.7, 19.6]
rank 1: [29.5, 39.4]
```

视频 [18:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1082s) 描述 owner 使用自己的 parameter、归约后的 gradient 与 optimizer state 更新。

### 5.5 Step 4：all-gather 更新后的 parameter shards

视频 [18:13](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1093s) 开始 all-gather；每个 rank 最终恢复完整参数：

```text
[9.7, 19.6] + [29.5, 39.4]
→ [9.7, 19.6, 29.5, 39.4]
```

这里的 `+` 是拼接，不是数值加法。All-gather 后 rank 0 和 rank 1 都拿到同一个完整新模型，下一批数据才能继续做普通 data-parallel forward。

### 5.6 ZeRO-1 显存公式从哪里来

沿用 §4 的 16-byte 口径：

- replicated parameter：$`2P`$ bytes；
- replicated gradient：$`2P`$ bytes；
- sharded optimizer state：$`KP/N`$ bytes。

所以每 rank：

```math
M_{\text{Z1}}
=2P+2P+\frac{KP}{N}.
```

合并前两项：

```math
M_{\text{Z1}}
=\left(4+\frac{K}{N}\right)P\ \text{bytes}.
```

- $`M_{\text{Z1}}`$：每 rank 持久静态模型状态 bytes。
- $`N`$：data-parallel ranks 数。
- $`K`$：每 parameter 的 optimizer-state bytes。

取 $`K=12,N=2,P=1`$ billion：

```math
4+12/2=4+6=10\ \text{bytes/parameter}.
```

所以：

```math
10\times10^9=10\ \text{GB per rank}.
```

Naive DP 是 16 GB/rank，因此纸面节省：

```math
16-10=6\ \text{GB/rank}.
```

### 5.7 为什么通信仍约 $`2P`$

视频 [18:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1105s) 把 RS+AG 和 all-reduce 联系起来；[18:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1110s) 先回顾 naive DP 的 all-reduce；[18:43](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1123s) 数 ZeRO-1 的两个 collective；[18:56](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1136s) 结论是主导通信特征相同。

按课件归一化大 $`N`$ 口径：

```math
\underbrace{P}_{\text{reduce-scatter gradients}}
+
\underbrace{P}_{\text{all-gather updated params}}
=2P.
```

更精确的 ring 每-rank发送量是：

```math
\frac{N-1}{N}P+\frac{N-1}{N}P
=2\frac{N-1}{N}P.
```

这正好与一次 ring all-reduce 相同。因此课件第 21 页把 ZeRO-1 称为 bandwidth-limited regime 中“free”：意思是**主要通信字节量不比 naive DDP 多**，不是零通信、零 latency、零 kernel overhead。

视频 [19:06](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1146s) 再回到内存：replicated parameter/gradient 保留，optimizer state 除以 $`N`$。

---

## 6. ZeRO Stage 2：再 shard gradients，但每 rank 仍计算完整模型

### 6.1 ZeRO-2 比 ZeRO-1 多做了哪件事

**【课程内容｜PDF 22–23 页】【视频补充｜[19:16](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1156s)】**

| 状态 | ZeRO-1 | ZeRO-2 |
|---|---|---|
| Parameter | replicated | replicated |
| Gradient | replicated 持久保存 | **sharded 持久保存** |
| Optimizer state | sharded | sharded |

重要区别：

> ZeRO-2 不要求每 rank 长期存完整 gradient vector；但 data parallel 的每个 rank 仍会对自己数据跑完整模型，并为每层计算该层的完整 local gradients。

视频 [19:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1165s) 说难点是不能让完整 gradient vector 长期 materialize（实体化）出来；[19:35](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1175s) 把解决办法称为系统技巧。

### 6.2 两层、两 ranks 的 incremental backward 时间线

设模型只有两层：

```text
forward:  Layer 1 → Layer 2 → loss
backward: Layer 2 → Layer 1
```

每层各 2 个参数，共 4 个参数。Owner 分配：

```text
rank 0 owns parameter/gradient shard: 每层第 0 个参数
rank 1 owns parameter/gradient shard: 每层第 1 个参数
```

Backward 按以下顺序：

| 时刻 | rank 0 / rank 1 做什么 | 哪些 gradient 暂时存在 |
|---:|---|---|
| 1 | 两 rank 计算 Layer 2 的 2 个 local gradients | 当前层的 local `[g20,g21]` |
| 2 | 立即对 Layer 2 做 reduce-scatter | rank 0 留归约后的 `g20`，rank 1 留 `g21` |
| 3 | 释放不属于自己的 Layer 2 local/full 临时 gradient | 每 rank 只持久留一个 shard |
| 4 | 两 rank 计算 Layer 1 的 2 个 local gradients | 当前层 local `[g10,g11]` 加之前 shard |
| 5 | 立即对 Layer 1 做 reduce-scatter | rank 0 留 `g10`，rank 1 留 `g11` |
| 6 | 释放当前层不再需要的完整临时 gradient | 最终每 rank 只有全模型 gradient 的一半 |

视频 [19:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1187s) 强调不是等全部 gradient vector 出现才通信；[19:56](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1196s) 说每算完一层就立刻发送给正确 worker；[20:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1201s) 说不再被 backward graph 使用时就释放；[20:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1205s) 总结为边计算、边发送。

### 6.3 PDF 第 23 页为什么画的是单个 Reduce 图标

第 23 页文字说“算完一层，立刻 reduce 到正确 worker”，中间小图画了一个 root reduce。应这样读：

- 对某一个 shard 来看，它确实被 reduce 到 owner；
- 对全层所有 shards 一起看，不同 shard 有不同 owner，整体效果是 reduce-scatter；
- 不是把全模型 gradient 永久堆到某一个全局 root。

这也解释了“每 worker 仍计算完整层 gradient”和“每 worker 不长期保存完整 gradient”为什么能同时成立。

### 6.4 ZeRO-2 持久静态显存公式

沿用 parameter/gradient 各 2 bytes、optimizer state $`K`$ bytes：

- replicated parameter：$`2P`$；
- sharded gradient：$`2P/N`$；
- sharded optimizer state：$`KP/N`$。

所以：

```math
M_{\text{Z2}}
=2P+\frac{2P}{N}+\frac{KP}{N}.
```

把后两项合并：

```math
M_{\text{Z2}}
=\left(2+\frac{2+K}{N}\right)P\ \text{bytes}.
```

取 $`K=12,N=2,P=1`$ billion：

```math
2+\frac{2+12}{2}
=2+\frac{14}{2}
=2+7
=9\ \text{bytes/parameter}.
```

因此纸面持久静态状态为：

```math
9\times10^9=9\ \text{GB/rank}.
```

### 6.5 “持久内存”不等于“运行峰值”

上述 $`9`$ GB 是稳态公式。真实 backward 某一时刻还可能有：

- 当前层尚未 reduce-scatter 的 local gradient；
- collective input/output buffer；
- 等待释放的上一 bucket；
- activations；
- allocator 预留和通信 overlap 引入的同时驻留。

**Peak memory（峰值显存）** 是整个时间线上最高的一瞬间，不是训练 step 结束后还留着多少。因此 ZeRO-2 “不长期 materialize 完整全模型 gradient”不等于任何瞬间都只有精确 $`2P/N`$ gradient bytes。

### 6.6 ZeRO-2 通信为什么仍约 $`2P`$

每层的 reduce-scatter 加起来覆盖全模型 gradients，总量仍是一个完整 $`P`$；最后 all-gather 更新后 parameters，又是一个完整 $`P`$：

```math
P+P=2P.
```

视频 [20:16](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1216s) 说明最后仍 all-gather parameters。分层让操作次数变多，但每次 payload 变小；把所有层 payload 相加仍约为全模型大小。

“通信元素总量相同”仍不保证时间完全相同，因为更多小 collective 会受到 launch latency、bucket size、overlap 和网络调度影响。

---

## 7. ZeRO Stage 3 / FSDP：parameters、gradients、state 全部 shard

### 7.1 FSDP 是什么，和 ZeRO-3 是什么关系

**【课程内容｜PDF 24–28 页】【视频补充】**

**FSDP（Fully Sharded Data Parallel，全分片数据并行）** 让每个 data-parallel rank 长期只保留 parameter、gradient、optimizer state 的 shard；轮到某个模块计算时，再临时 all-gather 该模块的完整参数。

视频 [20:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1225s) 把最后一级称为最复杂的推进；[20:36](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1236s) 开始 shard parameters；[20:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1249s) 说按计算图需要来发送/请求参数；[21:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1264s) 将其称为 ZeRO stage 3，也称 FSDP。

“FSDP ≈ ZeRO-3”是本讲的教学口径：两者核心 sharding 思想一致。真实 PyTorch FSDP 的 wrapping、resharding、prefetch、mixed precision、state dict 和设备 mesh 选项有具体实现语义，不能把所有版本、配置与 DeepSpeed ZeRO-3 说成逐 API 完全相同。

### 7.2 两层 × 两 ranks：长期存什么

设有 Layer 0、Layer 1；每层各有 4 个参数：

```text
W0 = [a0, b0, c0, d0]
W1 = [a1, b1, c1, d1]
```

两个 ranks 平均 shard：

| rank | 长期 parameter shards |
|---:|---|
| 0 | `W0[a0,b0]` 与 `W1[a1,b1]` |
| 1 | `W0[c0,d0]` 与 `W1[c1,d1]` |

每 rank 长期只存每层一半。但是普通矩阵计算需要当前层完整 $`W`$，所以计算前必须临时拼齐。

### 7.3 Forward：all-gather 当前层，算完就 reshard/free

**【课程内容｜PDF 25 页】【视频补充｜[21:17](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1277s)】**

```text
时刻 F0:
  rank 0 提供 W0[a0,b0]
  rank 1 提供 W0[c0,d0]
  all-gather → 两 rank 临时都有完整 W0[a0,b0,c0,d0]

时刻 F1:
  两 rank 各自用不同 local data 做 Layer 0 forward

时刻 F2:
  释放/reshard 临时完整 W0，只保留各自原 shard

时刻 F3:
  对 W1 重复 all-gather → forward → free
```

视频 [21:32](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1292s) 从获取一层权重开始；[21:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1300s) 说 forward 后可释放完整 weights。

**Reshard（重新分片）**：临时完整参数使用完后，回到每 rank 只保留自己 shard 的状态。实现上可释放非本地部分，或恢复 shard view；核心是避免完整参数长期驻留。

### 7.4 Backward：再次 all-gather，再 reduce-scatter gradients

Backward 顺序与 forward 相反：先 Layer 1，再 Layer 0。

```text
时刻 B0: all-gather W1，临时恢复完整 Layer 1 参数
时刻 B1: 用保存的 activation + W1 做 Layer 1 backward
时刻 B2: reduce-scatter Layer 1 gradients 给各 owner
时刻 B3: free 完整 W1

时刻 B4: all-gather W0
时刻 B5: 做 Layer 0 backward
时刻 B6: reduce-scatter Layer 0 gradients
时刻 B7: free 完整 W0
```

视频 [21:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1307s) 转入 backward；[21:57](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1317s) 说明再次按需 all-gather；[22:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1323s) 说明立即 reduce-scatter gradients；[22:12](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1332s) 总结为发送 gradient 后重复下一层。

注意：activation 仍要为 backward 保存或 **recompute（反向需要时重新执行相关 forward）**。ZeRO-3 切静态模型状态，不自动把所有 activation 除以 $`N`$。

### 7.5 为什么是 forward AG + backward AG + RS ≈ $`3P`$

**【课程内容｜PDF 25、27 页】【视频补充｜[22:18](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1338s)】** 对全模型把各层相加：

1. Forward 前 all-gather parameters：约 $`P`$；
2. Backward 前再次 all-gather parameters：约 $`P`$；
3. Backward 后 reduce-scatter gradients：约 $`P`$。

所以课程归一化为：

```math
P+P+P=3P.
```

Ring 的有限 $`N`$ 每-rank发送量更精确是：

```math
3\frac{N-1}{N}P.
```

DDP/ZeRO-1/2 则是：

```math
2\frac{N-1}{N}P.
```

二者相除：

```math
\frac{3(N-1)P/N}{2(N-1)P/N}=\frac32=1.5.
```

所以第 27 页写 ZeRO-3 约 1.5× communication cost。这个比值假设：相同 dtype payload、相同 bandwidth 模型、忽略每次 collective latency 和额外 **metadata（描述 tensor/通信的辅助信息）** 及 buffer。

### 7.6 Prefetch、free 与 overlap：怎样把通信藏在计算后面

**Prefetch（预取）**：当前层还在计算时，提前请求下一层参数。  
**Overlap（重叠）**：让 communication 与 computation 在时间上同时发生。  
**Stream（流）**：GPU 上保持命令顺序的队列；不同 stream 可在依赖允许时并发。

PDF 26 页把 GPU computation stream 与 GPU communication stream 分开。视频 [22:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1369s) 说 ZeRO-3 依靠 incremental communication 和 overlap；[23:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1390s) 引出通信/计算重叠；[23:41](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1421s) 从 all-gather Layer 0 开始；[23:53](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1433s) 说做 forward 0 时请求 Layer 1；[24:06](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1446s) 总结两者重叠。

最小时间线：

```text
时间      1        2        3        4
计算流   等W0     FWD0     FWD1     ...
通信流   AG W0    AG W1    AG W2    ...
```

第一个 all-gather 通常在 critical path（关键路径）上，因为没有 W0 就不能开始 FWD0。之后若：

```math
T_{\text{compute,current layer}}
\ge
T_{\text{all-gather,next layer}},
```

下一层通信可能被当前层计算完全遮住。若通信更慢，就会留下 bubble（空等区间）。视频 [24:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1465s) 给出“计算够多、网络够快”的条件；[24:32](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1472s) 才说 FSDP 可以接近免费，并立刻承认仍有 bubbles。

所以“overlap”不等于通信 bytes 消失，只是部分通信时间不再增加 step 的关键路径。

### 7.7 ZeRO-3 持久静态内存公式

Parameter、gradient 与 optimizer state 全 shard：

```math
M_{\text{Z3}}
=\frac{2P}{N}+\frac{2P}{N}+\frac{KP}{N}.
```

合并：

```math
M_{\text{Z3}}
=\frac{(4+K)P}{N}\ \text{bytes}.
```

取 §4 的 $`K=12,N=2,P=1`$ billion：

```math
\frac{4+12}{2}=\frac{16}{2}=8\ \text{bytes/parameter},
```

```math
8\times10^9=8\ \text{GB/rank}.
```

但 forward/backward all-gather 当前层时，峰值还会多出临时完整 layer 参数与通信 buffer。Wrap 的 module 太大，临时峰值也会大；wrap 太碎，collective 次数和 latency 又会多。

### 7.8 课堂问答：ZeRO-3 不是 pipeline parallel

视频 [25:51](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1551s) 说作业会实现 FSDP wrapper；[26:14](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1574s) 出现学生提问；老师在 [26:26](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1586s) 解释“把 gradient 从下一张 GPU 传回上一张”更像 pipeline；[26:35](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1595s) 强调 FSDP 中每个 GPU 仍走完整模型。

区别：

| 方法 | 每个 rank 计算哪些层 | 主要搬什么 |
|---|---|---|
| FSDP / ZeRO-3 | 每个 rank 都按顺序计算全部层，只是按需临时取参数 | parameter shards、gradient shards |
| Pipeline parallel | 不同 stage 长期负责不同层 | stage 之间的 activations 与 activation gradients |

视频 [27:12](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1632s) 的问答确认每个 GPU 有每层的一部分；[27:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1645s) 又问为何通信量不乘层数。答案是：collective 次数随层增多，但每层 payload 只是该层参数；所有层参数量相加仍为全模型 $`P`$。

### 7.9 第 28 页为何从 16 bytes 改成 12 bytes

**【课程内容｜PDF 28 页】【视频补充｜[27:55](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1675s)】** 这一页为了展示更省状态的训练口径，改用：

| 状态 | dtype | bytes/parameter |
|---|---|---:|
| Parameter | BF16 | 2 |
| Gradient | BF16 | 2 |
| Master weight | FP32 | 4 |
| Adam first moment $`m`$ | BF16 | 2 |
| Adam second moment $`v`$ | BF16 | 2 |
| **合计** |  | **12** |

逐项相加：

```math
2+2+4+2+2=12.
```

因此本小节的 optimizer state：

```math
K=4+2+2=8\ \text{bytes/parameter},
```

不是 §4 高精度 moments 口径的 $`K=12`$。

PDF 称它为 “Pure BF16 training (with Kahan summation)” 但又保留 FP32 master weight；严谨读法是“parameter、gradient、moments 采用 BF16，而 master weight 仍为 FP32”，并非所有 tensor 都是 BF16。

**Kahan summation（Kahan 补偿求和）**：普通低精度加法可能吞掉很小的更新；Kahan 额外维护一个 compensation（补偿误差），把上次丢掉的小量带入后续求和。它减少累计舍入误差，不会自动解决所有 BF16 optimizer 稳定性问题。

### 7.10 8×A100 80 GB：Baseline 完整复算

课件的每-rank 容量预算是 80 **十进制 GB**：

```math
80\ \text{GB}=80\times10^9\ \text{bytes}.
```

Baseline 每参数 12 bytes，因此最大参数个数：

```math
P_{\max}=\frac{80\times10^9\ \text{bytes}}
{12\ \text{bytes/parameter}}.
```

bytes 约掉，剩 parameters：

```math
P_{\max}=6.666666\ldots\times10^9
\approx6.667\ \text{B parameters}.
```

8 张卡没有帮助，因为 baseline 在每 rank 复制完整 12 bytes/parameter。

### 7.11 ZeRO-1：$`4+8/8=5`$ bytes/parameter

ZeRO-1 复制 BF16 parameter 与 gradient：

```math
2+2=4\ \text{bytes/parameter}.
```

Optimizer state 共 8 bytes，被 8 ranks 平均 shard：

```math
8/8=1\ \text{byte/parameter per rank}.
```

合计：

```math
4+1=5\ \text{bytes/parameter per rank}.
```

最大参数：

```math
80/5=16\ \text{B parameters}.
```

### 7.12 ZeRO-2：$`2+10/8=3.25`$ bytes/parameter

ZeRO-2 只复制 BF16 parameter：

```math
2\ \text{bytes/parameter}.
```

被 shard 的 gradient + optimizer state 为：

```math
2+8=10\ \text{bytes/parameter}.
```

除以 8 ranks：

```math
10/8=1.25\ \text{bytes/parameter per rank}.
```

合计：

```math
2+1.25=3.25\ \text{bytes/parameter per rank}.
```

最大参数：

```math
80/3.25
=\frac{8000}{325}
\approx24.6153846\ \text{B parameters}.
```

课件表四舍五入写 24.62 B；按要求保留三位可写 24.615 B。

### 7.13 ZeRO-3：$`12/8=1.5`$ bytes/parameter

全部 12 bytes 都被 8 ranks shard：

```math
12/8=1.5\ \text{bytes/parameter per rank}.
```

最大参数：

```math
80/1.5
=\frac{800}{15}
=53.333333\ldots\ \text{B parameters}.
```

汇总：

| 方法 | 每 rank bytes/param | $`80/\text{系数}`$ | 纸面最大参数 |
|---|---:|---:|---:|
| Baseline | 12 | $`80/12`$ | 6.667 B |
| ZeRO-1 | $`4+8/8=5`$ | $`80/5`$ | 16 B |
| ZeRO-2 | $`2+10/8=3.25`$ | $`80/3.25`$ | 24.615 B |
| ZeRO-3 | $`12/8=1.5`$ | $`80/1.5`$ | 53.333 B |

视频 [28:24](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1704s) 出现关于其他 GPU 容量的课堂提问；[28:39](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1719s) 老师说只需按容量比例换算；[28:53](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1733s) 结束 FSDP 小节。

### 7.14 为什么这张表不是“能部署”的保证

上述计算故意采用最乐观的静态账，没留：

- activation；
- all-gather/reduce-scatter buffer；
- 当前 FSDP unit 临时完整参数；
- allocator reserve 与 fragmentation；
- CUDA/NCCL context；
- loss/output、embedding、模型 checkpoint 保存/加载与 dataloader 相关 buffer；这里的模型 checkpoint 是训练状态快照，不是后文的 activation checkpointing；
- 故障恢复和稳定运行余量。

因此正确结论是：

> **在课件 12-byte 静态模型状态、十进制 80 GB、完美均匀 sharding 的纸面模型中，ZeRO-3 上限为 53.333B。**

不能写成：

> “8×A100 80 GB 一定能训练 53.333B 模型。”

最后再把本阶段主线压成一张状态表：

| 方法 | Parameter | Gradient | Optimizer state | 课程归一化通信 |
|---|---|---|---|---:|
| Naive DP | replicated | replicated | replicated | $`\approx2P`$ |
| ZeRO-1 | replicated | replicated | sharded | $`\approx2P`$ |
| ZeRO-2 | replicated | sharded | sharded | $`\approx2P`$ |
| ZeRO-3/FSDP | sharded | sharded | sharded | $`\approx3P`$ |

---

## 8. Data parallel 还没有解决什么：从“复制模型”走向 model parallel

### 8.1 先把两个名字分开

**【课程内容｜PDF p29–31｜视频 [29:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1740s)】**

- **Data parallel，数据并行（DP）**：每个 rank 处理不同数据；各 rank 逻辑上执行同一个模型。
- **Model parallel，模型并行（MP）**：把一个模型的不同部分放到不同 rank；一次样本的计算要经过多个 rank。

“模型的不同部分”还可以继续细分：

- 按层切，是 pipeline parallel；
- 按一个矩阵的宽度切，是 tensor parallel；
- 后面还会遇到按 token、序列位置或 expert 切。

本节先只抓住一个差别：

> DP 主要切数据；model parallel 主要切模型计算。

### 8.2 DP 会消耗 global batch size

**Global batch size（全局批大小）**：一次 optimizer update 总共使用多少训练样本。

设全局 batch 为 $`B`$，data-parallel ranks 数为 $`M`$。如果每个 rank 分到相同数量，那么：

```math
B_{\text{local}}=\frac{B}{M}.
```

例如 $`B=8`$：

| ranks $`M`$ | 每 rank 样本数 $`B/M`$ | 能否继续按整数样本均分 |
|---:|---:|---|
| 1 | 8 | 能 |
| 2 | 4 | 能 |
| 4 | 2 | 能 |
| 8 | 1 | 能 |
| 16 | 0.5 | 一般的逐样本训练不能这样直接分 |

因此课件 p29 与视频 [29:08](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1748s) 的第一层意思是：固定 $`B=8`$ 时，朴素 DP 最多让 8 个 rank 各拿 1 个样本。

但“把 batch 调大”也不是无限免费的。**Critical batch size（临界批大小）** 是一个经验边界：batch 增大到某个范围后，新增样本提供的梯度信息越来越重复，继续加 batch 未必等比例减少达到目标效果所需的 optimizer updates。视频 [29:19](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1759s) 强调，机器更多时会遇到 diminishing returns，也就是收益递减。

这不是说“任何模型的临界批大小都相同”。它取决于模型、数据、优化器、学习率 schedule 和训练阶段。

### 8.3 ZeRO 各阶段仍有各自没有切掉的东西

**【课程内容｜PDF p30｜视频 [30:11](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1811s)】**

| 方法 | parameter | gradient | optimizer state | activation |
|---|---|---|---|---|
| ZeRO-1 | replicated | replicated | sharded | 不因 ZeRO-1 自动变小 |
| ZeRO-2 | replicated | sharded | sharded | 不因 ZeRO-2 自动变小 |
| ZeRO-3 | sharded | sharded | sharded | 不因 ZeRO-3 自动变小 |

所以：

1. ZeRO-1/2 仍要求每 rank 放得下完整 parameters。
2. ZeRO-3 可以切 parameters，但没有自动切小每层 activation。
3. 如果序列很长或 batch 很大，activation 仍可能成为 **OOM（out of memory，显存不足而失败）** 来源。

> **材料内部口述校正：** 视频约 [30:16](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1816s) 附近有一句听起来像 “stage 2 cuts parameter memory”。这与 p18、p30 的表和本讲前面的公式冲突；按 ZeRO 的正式定义，ZeRO-2 不 shard parameters，ZeRO-3 才 shard parameters。应把那句视为口头滑误，不能据此改写算法。

### 8.4 Model parallel 与 ZeRO-3 搬运的对象不同

**【课程内容｜PDF p31｜视频 [30:41](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1841s)】**

设模型只有两层：

```math
x\xrightarrow{W_0}a\xrightarrow{W_1}y.
```

- $`x`$：输入 activation；
- $`W_0,W_1`$：两层 parameters；
- $`a`$：第一层产生、第二层要消费的中间 activation；
- $`y`$：输出。

**按层 model parallel：**

| GPU 0 | 通信 | GPU 1 |
|---|---|---|
| 永久保存 $`W_0`$，算 $`a=xW_0`$ | 把 $`a`$ 发给 GPU 1 | 永久保存 $`W_1`$，算 $`y=aW_1`$ |

backward 时通常还要把 $`da`$ 从 GPU 1 发回 GPU 0。这里 $`da`$ 表示 loss 对中间 activation $`a`$ 的梯度。

**ZeRO-3 / FSDP：**

两个 data-parallel ranks 各自处理自己的数据。某 rank 要算第 0 层时，先 all-gather $`W_0`$ 的 parameter shards；算完后释放完整 $`W_0`$。到第 1 层，再 all-gather $`W_1`$。它主要搬的是当前层 parameters，而不是把本 rank 的 activation 交给另一个 rank 接着算。

### 8.5 一个两层、二维的极小例子

**【补充理解】**

令行向量：

```math
x=[1,2],
\qquad
W_0=\begin{bmatrix}1&0\\0&2\end{bmatrix},
\qquad
W_1=\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
```

GPU 0 先算：

```math
a=xW_0
=[1\times1+2\times0,\;1\times0+2\times2]
=[1,4].
```

它向 GPU 1 发送 2 个 activation 元素 $`[1,4]`$。GPU 1 再算：

```math
y=aW_1
=[1\times1+4\times1,\;1\times1+4\times(-1)]
=[5,-3].
```

这个例子显示，按层切模型后：

- $`W_0`$ 不必在 GPU 1；
- $`W_1`$ 不必在 GPU 0；
- 代价是中间 activation $`[1,4]`$ 必须跨 GPU 边界。

若用 ZeRO-3 做 DP，两 rank 各有 $`W_0`$、$`W_1`$ 的一半 shard；每个 rank 为自己的样本先拼出当前层完整参数，再本地得到自己的 $`a`$。它不会把样本的 $`a`$ 交给另一个 rank 继续第二层。

### 8.6 参数通信与 activation 通信谁更小？不能只看名字

视频 [31:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1862s) 用一句很有用的话区分它们：model parallel 常传 activations，ZeRO-3 常传 parameters。

但不能因此断言“activation 永远更小”。要代数字：

- 一层 parameter 有多少元素？
- 边界 activation 的 shape 是多少？
- 一个 iteration 传几次？
- dtype 每元素多少 bytes？
- 是否重算、缓存、压缩或 overlap？

后面的 PP/TP 通信账会把这些问题具体化。

---

## 9. Pipeline parallel：让按层切开的 GPU 不要轮流发呆

### 9.1 Naive layer-wise model parallel 的 1/4 利用率

**【课程内容｜PDF p32–33｜视频 [32:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1921s)】**

设 4 个 pipeline stages，每个 stage 放在一张 GPU：

```math
x\to S_0\to S_1\to S_2\to S_3\to y.
```

一个 stage 是一段连续的模型层。先假设：

- 每个 stage 的 forward 都恰好用 1 个时间格；
- stage 间通信时间为 0；
- 只有一个大 batch，尚未切 microbatches。

时间表：

| 时间格 | GPU 0 / $`S_0`$ | GPU 1 / $`S_1`$ | GPU 2 / $`S_2`$ | GPU 3 / $`S_3`$ |
|---:|---|---|---|---|
| 1 | 算 batch | 空闲 | 空闲 | 空闲 |
| 2 | 空闲 | 算 batch | 空闲 | 空闲 |
| 3 | 空闲 | 空闲 | 算 batch | 空闲 |
| 4 | 空闲 | 空闲 | 空闲 | 算 batch |

共有：

```math
4\ \text{GPUs}\times4\ \text{time slots}=16\ \text{GPU-slots}.
```

真正做计算的只有 4 个 GPU-slots，所以系统平均利用率：

```math
\frac{4}{16}=\frac14=25\%.
```

“1/4”不是说每张 GPU 的峰值算力变成四分之一；它说在这个极简时间表中，平均每个时刻只有四张卡中的一张忙。

### 9.2 Microbatch 是什么

**【课程内容｜PDF p34｜视频 [33:06](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1986s)】**

把一个大 batch 切成多个更小的批次，每一小块叫一个 **microbatch（微批次）**。

例如 global batch 有 16 个样本，切成 8 个 microbatches：

```math
b_{\text{micro}}=16/8=2\ \text{samples}.
```

$`S_0`$ 把 microbatch 0 交给 $`S_1`$ 后，不必等它走完全模型；$`S_0`$ 可以立刻开始 microbatch 1。这就是 pipeline 的来源。

### 9.3 $`p=4,m=8`$ 的完整 forward 时间表

**【补充理解；对 PDF p34 逐格展开】**

令：

- $`p=4`$：4 个 stages；
- $`m=8`$：microbatch 编号 0–7；
- `F3`：对 microbatch 3 做 forward；
- 每个 stage、每个 microbatch 都正好 1 个时间格。

| 时间 | $`S_0`$ | $`S_1`$ | $`S_2`$ | $`S_3`$ | 阶段 |
|---:|---|---|---|---|---|
| 1 | F0 | — | — | — | warmup |
| 2 | F1 | F0 | — | — | warmup |
| 3 | F2 | F1 | F0 | — | warmup |
| 4 | F3 | F2 | F1 | F0 | steady |
| 5 | F4 | F3 | F2 | F1 | steady |
| 6 | F5 | F4 | F3 | F2 | steady |
| 7 | F6 | F5 | F4 | F3 | steady |
| 8 | F7 | F6 | F5 | F4 | steady |
| 9 | — | F7 | F6 | F5 | drain |
| 10 | — | — | F7 | F6 | drain |
| 11 | — | — | — | F7 | drain |

- **warmup（预热段）**：pipeline 还没填满。
- **steady state（稳态）**：四个 stages 同时工作。
- **drain（排空段）**：$`S_0`$ 已没新 microbatch，后段仍在处理旧 microbatch。
- **bubble（气泡）**：某 stage 因依赖尚未满足而空闲的时间格。

理想 forward 总时间格：

```math
m+p-1=8+4-1=11.
```

每个 stage 有 8 个有用格，所以按“有用格/总格”定义的利用率：

```math
U=\frac{m}{m+p-1}=\frac8{11}\approx0.7273=72.73\%.
```

### 9.4 三个看似相近的百分比，分母其实不同

课件 p34 写的 bubble ratio 是：

```math
\frac{\text{bubble}}{\text{useful}}
=\frac{p-1}{m}
=\frac3{8}
=37.5\%.
```

如果问 bubble 占总时间的比例，分母要换成 total：

```math
\frac{\text{bubble}}{\text{total}}
=\frac{p-1}{m+p-1}
=\frac3{11}
\approx27.27\%.
```

利用率则是：

```math
\frac{\text{useful}}{\text{total}}
=\frac8{11}
\approx72.73\%.
```

最后两个相加为：

```math
72.73\%+27.27\%=100\%.
```

而 $`37.5\%`$ 的分母是 useful，不应拿来和 $`72.73\%`$ 直接相加。视频 [33:37](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2017s) 口头把 bubble time 与 useful compute 相比，正是在提醒这个分母。

### 9.5 一次边界 activation 有多少 bytes

**【课程内容｜PDF p35｜视频 [34:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2087s)】**

定义：

- $`b_{\text{micro}}`$：每个 microbatch 的样本数；
- $`s`$：每个样本的 token 数，也就是 sequence length；
- $`h`$：每个 token 的 hidden dimension；
- $`q`$：dtype 的 bytes/element。

activation shape 是：

```math
[b_{\text{micro}},s,h].
```

元素数：

```math
b_{\text{micro}}sh.
```

字节数：

```math
\text{activation bytes}=b_{\text{micro}}shq.
```

后面会用 **KiB（kibibyte，二进制千字节）**：

```math
1\ \text{KiB}=1024\ \text{bytes}.
```

例：$`b_{\text{micro}}=2,s=4,h=8`$，BF16 每元素 $`q=2`$ bytes：

```math
2\times4\times8=64\ \text{elements},
```

```math
64\times2=128\ \text{bytes per boundary per forward send}.
```

4 stages 有 3 条相邻边界。若只数 forward 的 aggregate sends，一个 microbatch 共：

```math
3\times128=384\ \text{bytes}.
```

若 backward 还发送同 shape 的 activation gradient，再加 384 bytes。这里没有算协议、对齐和网络 packet metadata。

### 9.6 为什么 PP 常放在较慢的跨节点链路，但不是铁律

视频 [35:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2101s) 与 p35 的比较重点是：

- PP 通常只在 stage 边界做 point-to-point send/receive；
- Tensor parallel 往往每一层都有阻塞式 collective；
- 因此常见部署会把 TP 放在同节点高速互联，把 PP 跨节点。

**Point-to-point（点对点通信）**：一个指定 sender 向一个指定 receiver 发送，不要求整个 group 同时得到同一结果。

但“PP 必须跨节点”是错的。还要看：

- activation 有多大；
- stage 是否均衡；
- microbatch 是否足够填满 pipeline；
- 跨节点带宽与 latency；
- 是否有空闲高速链路；
- 与 DP/TP 如何组合。

### 9.7 Batch 越大，bubble 常越小；优化难度却可能变

p36 的横轴是 batch size，纵轴画出随机器规模变化的可扩展趋势；它不是所有模型共享的一条定律。更多 microbatches 会让：

```math
\frac{p-1}{m}
```

变小，但也可能改变有效 batch、收敛、activation memory 和每步 latency。视频 [35:54](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2154s) 将它称为 pipeline 与 batch size 的耦合。

---

## 10. Pipeline schedules：GPipe、1F1B、interleaving 与 zero-bubble

### 10.1 Schedule 决定“谁在第几个时间格做什么”

**【课程内容｜PDF p37–38｜视频 [36:35](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2195s)】**

**Schedule（调度）** 就是为每个 stage 排出 forward/backward 的执行顺序。相同的层切分，换 schedule 后会改变：

- bubble；
- 同时存活的 activations；
- 通信是否容易 overlap；
- 每个 rank 的等待和峰值 memory；
- 实现复杂度。

### 10.2 GPipe：先全部 forward，再全部 backward

**【课程内容；方法边界来自 [GPipe 原论文](https://arxiv.org/abs/1811.06965)】**

用 $`p=2,m=4`$，`F2` 表示 microbatch 2 的 forward，`B2` 表示它的完整 backward。下面是一张忽略通信、F/B 都各占一格的教学时间表：

| 时间 | $`S_0`$ | $`S_1`$ |
|---:|---|---|
| 1 | F0 | — |
| 2 | F1 | F0 |
| 3 | F2 | F1 |
| 4 | F3 | F2 |
| 5 | — | F3 |
| 6 | — | B3 |
| 7 | B3 | B2 |
| 8 | B2 | B1 |
| 9 | B1 | B0 |
| 10 | B0 | — |

两个 stages 各完成 $`4F+4B=8`$ 个有用格，总时间 10 格：

```math
U=8/10=80\%.
```

$`S_0`$ 在做 B0–B3 前，曾经保存 4 个 microbatches 的 forward activations；这说明 GPipe 的一个主要 memory 压力来自“先把所有 forward 都推过去”。实际可以配合 **activation checkpoint（只保存若干边界 activation，反向时重算段内中间值）**，但这又增加计算。

### 10.3 1F1B：稳态尽量一前一后

**1F1B（one-forward-one-backward）**：进入稳态后，一个 stage 尽量交替执行一次 forward 与一次 backward。

下面是一张依赖合法的 $`p=2,m=4`$ 教学 schedule；不同框架可能排出等价但不完全相同的格子：

| 时间 | $`S_0`$ | $`S_1`$ |
|---:|---|---|
| 1 | F0 | — |
| 2 | F1 | F0 |
| 3 | F2 | B0 |
| 4 | B0 | F1 |
| 5 | F3 | B1 |
| 6 | B1 | F2 |
| 7 | — | B2 |
| 8 | B2 | F3 |
| 9 | — | B3 |
| 10 | B3 | — |

依赖检查：

- $`S_1`$ 只能在收到 $`S_0`$ 的 F0 后做 F0；
- $`S_0`$ 只能在收到 $`S_1`$ 的 B0 后做 B0；
- 同一 stage 不能在同一时间格同时做 F 与 B。

这张玩具表仍是 10 格，因此不声称仅改成 1F1B 就自动消灭所有 bubble。它的直接收益是较早释放旧 microbatch 的 activations。$`S_0`$ 最多先积累 F0、F1、F2 三份，随后就处理 B0；相较上面的 GPipe 教学表，峰值存活 forward activations 从 4 份降到 3 份。

### 10.4 Virtual/interleaved pipeline stages

**【课程内容｜PDF p37｜视频 [37:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2221s)】【延伸边界：[Megatron interleaved pipeline 原论文](https://arxiv.org/abs/2104.04473)】**

- **Virtual stage（虚拟 stage）**：一张物理 GPU 负责多个模型 chunks。
- **Interleaved（交错）**：这些 chunks 不一定是一个连续大段，而是以交错次序排进 schedule。

这样可能缩短某些 bubble、改善 stage load balance。但 p37 上看起来更“满”的彩色格子不是免费午餐：

1. 一个 microbatch 可能更频繁跨设备边界；
2. send/receive 次数可能增加；
3. overlap 需要更复杂的依赖和 buffer 管理；
4. 因而网络需要更高 bandwidth 才不会把计算收益吃掉。

视频 [36:52](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2212s) 展示了把不同 forward/backward 元素交错排列的思路；正确理解是“调度空间很大且有代价”，不是推荐越复杂越好。

### 10.5 Backward 不是一块不可拆的黑箱

**【课程内容｜PDF p38｜视频 [37:28](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2248s)】**

对一层 $`Y=XW`$，backward 至少要得到：

- $`dX`$：loss 对输入 activation $`X`$ 的梯度；前一个 stage 要靠它继续 backward；
- $`dW`$：loss 对本层 parameter $`W`$ 的梯度；optimizer update 要靠它更新 $`W`$。

用矩阵写：

```math
dX=dY W^T,
```

```math
dW=X^T dY.
```

$`dX`$ 在跨 stage 的 backward 依赖链上，因此通常更紧急。$`dW`$ 不需要送给前一个 stage，所以在满足下列条件时可以较晚计算：

- optimizer update 尚未开始；
- 保存 $`X`$ 与 $`dY`$ 的 buffer 尚未被释放或覆盖；
- 延后不会造成新的 memory 峰值；
- 与后续通信/计算没有资源冲突。

因此只能说“$`dW`$ 有一定可延后空间”，不能说“$`dW`$ 可任意晚算”。

### 10.6 Zero-bubble 的目标和边界

**【课程内容｜PDF p38】【延伸边界：[Zero Bubble Pipeline Parallelism 原论文](https://arxiv.org/abs/2401.10241)】**

Zero-bubble schedule 利用 $`dX`$、$`dW`$ 依赖不同，把原来写成一个 `B` 的格子拆开重排，尝试用 $`dW`$ 计算填补空闲格。视频 [38:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2320s) 的核心不是“数学上保证永远 0% 空闲”，而是：更细的 backward 分解给调度器更多填空机会。

实际仍受：

- stage 时间不均；
- 网络冲突；
- memory；
- kernel 粒度；
- optimizer step 边界；
- microbatch 数量

约束。论文名字不能被读成所有硬件和所有配置下都真正零气泡。

### 10.7 五种说法放在一起

| 名称 | 最小人话解释 | 主要收益 | 主要代价/边界 |
|---|---|---|---|
| naive layer-wise | 一个 batch 顺次走 stages | 最简单 | 大量空闲 |
| GPipe | 所有 microbatch forward 后再 backward | 容易理解、吞吐高于 naive | activation 存活较多 |
| 1F1B | 稳态交替 F/B | 更早释放 activation | 仍有 warmup/drain |
| virtual/interleaved | 每设备多个 chunks，交错运行 | 可减 bubble、平衡 stage | 通信/调度更复杂 |
| zero-bubble family | 拆 $`dX,dW`$ 后重排 | 更多填 bubble 的机会 | 依赖、memory、实现条件严格 |

---

## 11. Tensor parallel：同一层的矩阵也可以切开

### 11.1 先从普通两层 MLP 开始

**【课程内容｜PDF p39–40｜视频 [39:27](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2367s)】**

MLP 是 multilayer perceptron，这里把它理解成 Transformer 中的前馈子层。先写一个不带 bias 的两层版本：

```math
H=\phi(XW_1),
```

```math
Y=HW_2.
```

符号：

- $`X`$：输入，shape $`[b,h]`$；
- $`W_1`$：扩宽矩阵，shape $`[h,f]`$；
- $`H`$：中间 activation，shape $`[b,f]`$；
- $`\phi`$：逐元素 activation function，例如 ReLU；
- $`W_2`$：缩回 hidden width 的矩阵，shape $`[f,h]`$；
- $`Y`$：输出，shape $`[b,h]`$。

这里 $`f`$ 是 FFN intermediate width，不是后文课件图中的抽象函数 $`f(\cdot)`$。同一个字母可能复用，必须看本小节定义。

### 11.2 Column parallel：按 $`W_1`$ 的输出列切

设两个 tensor-parallel ranks。把 $`W_1`$ 的列切成：

```math
W_1=[W_1^{(0)}\mid W_1^{(1)}].
```

因为矩阵的每一列产生一个输出维度，所以：

```math
XW_1=[XW_1^{(0)}\mid XW_1^{(1)}].
```

两个 rank 可以各算一半 hidden features。逐元素函数 $`\phi`$ 不会把不同 feature 混在一起，所以每 rank 可以直接本地算：

```math
H^{(r)}=\phi(XW_1^{(r)}).
```

这一段称为 **column-parallel linear layer（列并行线性层）**。

### 11.3 Row parallel：按 $`W_2`$ 的输入行切

$`H`$ 已经按 feature 切成 $`[H^{(0)}\mid H^{(1)}]`$。于是把 $`W_2`$ 对应地按行切：

```math
W_2=
\begin{bmatrix}
W_2^{(0)}\\
W_2^{(1)}
\end{bmatrix}.
```

完整输出是：

```math
Y=HW_2
=H^{(0)}W_2^{(0)}+H^{(1)}W_2^{(1)}.
```

每 rank 先得到一个相同 shape 的 partial output：

```math
Y^{(0)}_{\text{partial}}=H^{(0)}W_2^{(0)},
```

```math
Y^{(1)}_{\text{partial}}=H^{(1)}W_2^{(1)}.
```

最后用 all-reduce SUM 逐元素相加，才得到完整 $`Y`$。这叫 **row-parallel linear layer（行并行线性层）**。

### 11.4 用同一组整数把 column→row 全部算完

**【补充理解；用于复算课件 p40】**

令：

```math
X=[1,2],
```

```math
W_1=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8
\end{bmatrix}.
```

shape 检查：

```math
X:[1,2],\quad W_1:[2,4],\quad XW_1:[1,4].
```

先算完整参考结果的四个格：

```math
(XW_1)_0=1\times1+2\times5=11,
```

```math
(XW_1)_1=1\times2+2\times6=14,
```

```math
(XW_1)_2=1\times3+2\times7=17,
```

```math
(XW_1)_3=1\times4+2\times8=20.
```

所以：

```math
XW_1=[11,14,17,20].
```

四个数都大于 0。取 $`\phi=\mathrm{ReLU}`$，ReLU 对正数不改值：

```math
H=[11,14,17,20].
```

### 11.5 两个 ranks 的 column shards

rank 0 拿前两列：

```math
W_1^{(0)}=
\begin{bmatrix}
1&2\\
5&6
\end{bmatrix},
\qquad
H^{(0)}=XW_1^{(0)}=[11,14].
```

rank 1 拿后两列：

```math
W_1^{(1)}=
\begin{bmatrix}
3&4\\
7&8
\end{bmatrix},
\qquad
H^{(1)}=XW_1^{(1)}=[17,20].
```

shape：

| tensor | rank 0 | rank 1 |
|---|---|---|
| $`W_1^{(r)}`$ | $`[2,2]`$ | $`[2,2]`$ |
| $`H^{(r)}`$ | $`[1,2]`$ | $`[1,2]`$ |

把 $`H^{(0)}`$ 与 $`H^{(1)}`$ 按最后一维拼起来，正是 $`[11,14,17,20]`$；但实际不用现在 all-gather，因为下一层可以直接消费分片。

### 11.6 两个 ranks 的 row shards 和 partial outputs

取：

```math
W_2=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
2&-1
\end{bmatrix},
```

shape $`[4,2]`$。对应 $`H`$ 的切法，把前两行给 rank 0，后两行给 rank 1：

```math
W_2^{(0)}=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_2^{(1)}=
\begin{bmatrix}
1&1\\
2&-1
\end{bmatrix}.
```

rank 0：

```math
Y_{\text{partial}}^{(0)}
=[11,14]
\begin{bmatrix}1&0\\0&1\end{bmatrix}
=[11,14].
```

rank 1 的第一个输出格：

```math
17\times1+20\times2=17+40=57.
```

第二个输出格：

```math
17\times1+20\times(-1)=17-20=-3.
```

所以：

```math
Y_{\text{partial}}^{(1)}=[57,-3].
```

all-reduce SUM：

```math
Y=[11,14]+[57,-3]=[68,11].
```

### 11.7 与未切分 dense 计算逐格核对

完整第一格：

```math
11\times1+14\times0+17\times1+20\times2
=11+0+17+40
=68.
```

完整第二格：

```math
11\times0+14\times1+17\times1+20\times(-1)
=0+14+17-20
=11.
```

因此：

```math
HW_2=[68,11],
```

与两个 partial outputs 的 all-reduce 结果完全相同。

这不是近似；在相同算术精度与相同求和顺序的理想数学中，它只是把同一个矩阵乘分组计算。实际浮点数可能因加法顺序不同有很小舍入差。

### 11.8 课件的 $`f`$ 与 $`g`$ 到底在做什么

**【课程内容｜PDF p40｜视频 [40:45](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2445s)】**

课件用两个抽象算子：

- $`f`$：forward 是 identity，backward 是 all-reduce；
- $`g`$：forward 是 all-reduce，backward 是 identity。

**Identity（恒等操作）**：输入是什么，输出就是什么，不改数值。

不要把它背成神秘口诀。用数据流解释：

1. Column-parallel $`W_1`$ 前，每个 rank 都已有同一个完整输入 $`X`$。forward 不必通信，所以 $`f`$ 的 forward 是 identity。
2. backward 时，每个 column shard 只算出了 loss 对 $`X`$ 梯度的一部分；这些部分必须 all-reduce SUM，才是完整 $`dX`$。所以 $`f`$ 的 backward 通信。
3. Row-parallel $`W_2`$ 后，每个 rank 只有 $`Y`$ 的一个 partial sum；forward 必须 all-reduce，所以 $`g`$ 的 forward 通信。
4. backward 输入 $`dY`$ 在每个 rank 都相同；各 rank 用自己的 row shard 得到本地所需的 $`dH^{(r)}`$，不必先相加，所以 $`g`$ 的 backward 是 identity。

视频 [41:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2463s) 指出 $`g`$ 在 forward 做 all-reduce；column+row 这样搭配后，不需要在两个 linear 中间先把完整 $`H`$ all-gather 出来。

### 11.9 这里的“切宽度”不等于把 batch 切开

在本例中两个 ranks 都处理同一个 $`X=[1,2]`$，只是：

- rank 0 算 hidden features 0–1；
- rank 1 算 hidden features 2–3。

这与 DP 的“rank 0 处理样本 A、rank 1 处理样本 B”是不同轴。TP ranks 必须协作完成同一个样本的同一层，因此每层通信常在下一层开始前形成阻塞依赖。

---

## 12. 一个 Transformer block 具体怎样切

### 12.1 先给未切分地图

**【课程内容｜PDF p41–42｜视频 [41:33](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2493s)】**

用一个简化 Transformer block：

```text
X [b,s,h]
  → norm
  → Q,K,V projections
  → attention
  → output projection
  → residual
  → norm
  → FFN up + activation
  → FFN down
  → residual
```

这里：

- $`b`$：batch；
- $`s`$：sequence length；
- $`h`$：model hidden width；
- $`f`$：FFN intermediate width。

### 12.2 哪些矩阵按列切，哪些按行切

**QKV** 是 attention 中三组向量的合称：$`Q`$ 是 **query（查询）**，用来问“我该关注谁”；$`K`$ 是 **key（键）**，供query匹配；$`V`$ 是 **value（值）**，匹配后真正聚合的信息。Combined QKV projection 一次产生这三组向量。

一种经典 Megatron-LM 风格切法是：

| 子层 | 完整 parameter shape | 切法 | 为什么 |
|---|---|---|---|
| combined QKV projection | $`[h,3h]`$ | 按输出列切 | 各 rank 得到一部分 heads/features |
| attention output projection | $`[h,h]`$ | 按输入行切 | 各 rank 的 local attention output 形成 partial output |
| FFN up projection | $`[h,f]`$ | 按输出列切 | 各 rank 得到一部分 FFN hidden features |
| FFN down projection | $`[f,h]`$ | 按输入行切 | local FFN features 形成 partial output |
| norm scale | $`[h]`$ | 常 replicated | 每 rank 的 residual-width 输入都要归一化 |
| router | 架构相关 | 课程图中 replicated | 每 rank 要按同一 token hidden state 做路由 |

这里的 router 是 MoE 中给 token 选择 experts 的小网络；没有 MoE 的 dense block 就没有 router。

该表是一种常见设计，不是所有 attention/FFN 实现的唯一切法。特别是 GQA、SwiGLU、MoE、sequence parallel 会改变具体 shape 或通信位置。

### 12.3 $`h=8,f=16,p=2`$：QKV 与 attention output

**【补充理解；对 PDF p41 的 shape 逐项展开】**

先忽略 bias，假设 attention 的 Q/K/V 总输出宽度都为 $`h=8`$。

Combined QKV：

```math
W_{QKV}:[8,3\times8]=[8,24].
```

总 parameter 数：

```math
8\times24=192.
```

两个 ranks 按输出列均分：

```math
W_{QKV}^{(r)}:[8,12],
```

每 rank：

```math
8\times12=96\ \text{parameters}.
```

若输入 $`X:[b,s,8]`$，每 rank 输出 combined local QKV：

```math
[b,s,12].
```

把最后 12 维分成 Q/K/V 三份，每份：

```math
Q^{(r)},K^{(r)},V^{(r)}:[b,s,4].
```

这假设 head partition 与 $`h/2=4`$ 兼容。

Attention output projection 完整 shape：

```math
W_O:[8,8],
```

共 $`8\times8=64`$ parameters。按输入行切后每 rank：

```math
W_O^{(r)}:[4,8],
```

共：

```math
4\times8=32\ \text{parameters}.
```

每 rank 的 local attention output 是 $`[b,s,4]`$，乘 $`[4,8]`$ 得 local partial output：

```math
[b,s,4]\times[4,8]\to[b,s,8].
```

两个 $`[b,s,8]`$ partial outputs all-reduce SUM，得到完整 attention output $`[b,s,8]`$。

### 12.4 $`h=8,f=16,p=2`$：FFN up 与 down

普通非 gated FFN 的 up parameter：

```math
W_{up}:[8,16],
```

总数：

```math
8\times16=128.
```

按输出列切后，每 rank：

```math
W_{up}^{(r)}:[8,8],
\qquad8\times8=64\ \text{parameters}.
```

输入 $`[b,s,8]`$ 乘本地 $`[8,8]`$：

```math
[b,s,8]\times[8,8]\to H^{(r)}:[b,s,8].
```

逐元素 activation 后 shape 不变。

Down parameter：

```math
W_{down}:[16,8],
```

总数也是 $`16\times8=128`$。按输入行切后每 rank：

```math
W_{down}^{(r)}:[8,8],
\qquad8\times8=64\ \text{parameters}.
```

本地乘法：

```math
[b,s,8]\times[8,8]\to Y_{partial}^{(r)}:[b,s,8].
```

all-reduce SUM 后恢复完整 $`[b,s,8]`$。

### 12.5 每 rank 的主要 linear parameter 数

忽略 bias、norm 和 embedding：

| 子矩阵 | 完整参数数 | 每 rank 参数数 |
|---|---:|---:|
| QKV | 192 | 96 |
| attention output | 64 | 32 |
| FFN up | 128 | 64 |
| FFN down | 128 | 64 |
| 合计 | $`192+64+128+128=512`$ | $`96+32+64+64=256`$ |

核对：

```math
2\ \text{ranks}\times256=512.
```

一个 norm scale vector 有 8 个 parameters；若 replicated，则每 rank 都额外保存 8 个，不会变成 4 个。两个 norms 就是每 rank 额外 16 个 scale parameters，若有 bias 还会再加。

如果 FFN 是 SwiGLU 一类 gated FFN，会有额外 gate projection；上面的 128+128 只属于普通两矩阵 FFN，不能直接套到 gated FFN。

### 12.6 为什么 TP 通常偏爱节点内高速链路

视频 [42:22](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2542s) 与 p42 的“通常在 8 GPUs 节点内”是课程时点经验，不是数学上限。原因是：

1. attention output 的 partial sums 要 all-reduce；
2. FFN down 的 partial sums 也要 all-reduce；
3. backward 还有对应梯度通信；
4. 下一子层常等完整 residual-width activation 才能继续。

因此这些 collective 位于频繁的 layer critical path 上。**Critical path（关键路径）**：决定整个 iteration 最早何时结束、无法被其他工作完全遮住的依赖链。

如果跨节点网络足够快、模型特别大、或使用不同切法，TP 也可以跨节点；“节点内”只是常见性能选择。

---

## 13. TP 与 PP 的通信账：先固定计数口径，再比较

### 13.1 课件 p43 的两个式子

**【课程内容｜PDF p43｜视频 [44:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2642s)】**

课件用近似量级比较：

```math
\text{TP communication}
\approx 8bsh\frac{p-1}{p}
\quad\text{elements sent per rank per layer},
```

```math
\text{PP communication}
\approx bsh
\quad\text{per stage boundary per microbatch direction}.
```

符号：

- $`b`$：本次通信对应的 batch 或 microbatch size；
- $`s`$：sequence length；
- $`h`$：hidden width；
- $`p`$：参与 tensor parallel 的 ranks 数；
- $`(p-1)/p`$：每 rank 不在本地、需和其他 ranks 交换的归一化比例。

两个式子故意省略很多实现细节。它们首先写的是 activation **元素量级**，不是已经乘好 dtype 的 bytes；必须先说 factor 8 在什么口径下能重建。

### 13.2 一种能得到 factor 8 的明确教学口径

**【补充理解；用于拆开 p43】**

假设：

1. 一个 Transformer layer 的 training forward+backward 合计有 4 个 activation-sized all-reduces：attention 与 FFN 各有 forward/对应 backward 通信；
2. 每个 message 的逻辑 tensor shape 都近似 $`[b,s,h]`$，共有 $`bsh`$ 元素；
3. 用 ring all-reduce 教学模型；一次 all-reduce 由 reduce-scatter 与 all-gather 两个阶段组成；
4. 因而一次 all-reduce 的 per-rank send 量近似 $`2(p-1)/p`$ 份 message；
5. 只数每 rank sends，不把 receives 再加一遍，不计协议、padding，也不假装 NCCL 必然选择 ring。

那么一个 all-reduce 每 rank 的发送元素数：

```math
bsh\times2\ \text{ring stages}\times\frac{p-1}{p}.
```

四个 all-reduces：

```math
4\times2\times bsh\frac{p-1}{p}
=8bsh\frac{p-1}{p}\ \text{elements sent}.
```

factor 8 就来自：

```math
4\ \text{all-reduces}\times2\ \text{ring stages}=8.
```

若 dtype 是 BF16，再把元素数乘 $`2\ \text{bytes/element}`$；dtype 不是 factor 8 的来源。

### 13.3 为什么换一种流量口径会得到别的系数

上节已采用 ring per-rank sends：

```math
2\frac{p-1}{p}\times\text{message elements},
```

这里的 2 来自 reduce-scatter 与 all-gather 两阶段。若改成只数 logical payload、完全不展开 ring 两阶段，4 个 collectives 只会写成：

```math
4bsh\frac{p-1}{p}\ \text{logical elements}.
```

如果在 §13.2 的 ring send 量上再把 receive 也加进 endpoint traffic，则会从 factor 8 变成 factor 16。若 collective 采用 tree 或其他算法，steps 与实际 link traffic 也会不同。

所以 §13.2 只是用常见 ring training 口径复现 p43 的 8；课件页本身没有把物理算法和端点计数逐项写出。它不是“所有 NCCL 实现用 **profiler（记录运行时间、通信和资源行为的分析工具）** 必然测到的精确字节数”。比较两个公式时要保证：

- 都数 logical payload，或都数 physical link traffic；
- 都是 per rank，或都是 aggregate；
- 都只数 send，或都数 send+receive；
- dtype 相同；
- forward/backward 范围相同。

### 13.4 用 $`b=2,s=4,h=8,p=2`$ 算 TP

先算 activation 元素数：

```math
bsh=2\times4\times8=64\ \text{elements}.
```

归一化比例：

```math
\frac{p-1}{p}=\frac{2-1}{2}=\frac12.
```

代入课程式：

```math
8bsh\frac{p-1}{p}
=8\times64\times\frac12
=256\ \text{elements sent per rank per layer}.
```

按 §13.2 的四个 all-reduces 拆开检查：

```math
64\ \text{elements}\times2\ \text{ring stages}\times\frac12
=64\ \text{elements sent per all-reduce},
```

```math
4\times64=256\ \text{elements sent}.
```

BF16 每元素 2 bytes，因此：

```math
256\times2=512\ \text{bytes sent per rank per layer}.
```

### 13.5 同一 shape 下算 PP

PP 在一个 stage boundary 的 forward 发送完整 activation：

```math
bsh=64\ \text{elements}.
```

BF16 字节：

```math
64\times2=128\ \text{bytes per boundary per forward direction}.
```

backward 若发同 shape 的 $`dX`$：

```math
128\ \text{bytes}.
```

一个 forward+backward 对：

```math
128+128=256\ \text{bytes per boundary}.
```

注意，课件 p43 的 `PP: bsh` 是元素量级写法，未显式乘 BF16 的 2 bytes，也未说明是否同时数 backward。把它与带 factor 8 的 byte 口径并排时，必须补齐这些假设，不能只看表面数字。

### 13.6 为什么这个小例不能推出“TP 与 PP 一样贵”

在选定口径下，本例得到 TP 512 send bytes、PP forward+backward 256 payload bytes，但这仍不能直接变成运行时间比较：

- TP 的 512 是**每一层、ring per-rank sends**，未再加 receives；
- PP 的 256 是**一条 stage boundary、一个 microbatch 的双向 payload**；
- 一个 stage 内可能有很多层，却只有 stage 边界才跨 PP rank；
- PP 有 $`m`$ 个 microbatches；
- TP collective 与 PP point-to-point 的 latency、拓扑和 overlap 能力不同。

若一个 stage 有 10 层，TP 的 per-layer 通信会重复 10 次；PP 仍只在 stage 边界传一次该 stage 输出，但会对每个 microbatch 传。到底谁占用更多时间必须把 layers、microbatches 和链路都带进去。

### 13.7 TP vs PP 决策表

**【课程内容｜PDF p43｜视频 [44:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2665s)】**

| 维度 | Tensor parallel | Pipeline parallel |
|---|---|---|
| 切什么 | 同一层的矩阵宽度 | 连续层/stages |
| parameters | 每层在 TP ranks 间切 | 每 stage 只存自己的层 |
| 通信位置 | 常每层多次 collective | 常只在 stage 边界 p2p |
| 依赖 | 下一子层常等 collective | 下一 stage 等 activation；不同 microbatches 可流水 |
| 对 batch 的要求 | 不靠增大 microbatch 才能切矩阵 | 足够多 microbatches 才能降低 bubble |
| 常见链路选择 | 节点内高速互联 | 可跨较慢节点，但非必然 |
| 主要挑战 | collective latency/bandwidth、可整除 shape | bubble、stage balance、activation memory、schedule |

### 13.8 本轮主线压缩成六句话

1. ZeRO-1/2 没有 shard parameters；ZeRO-3 没有自动 shard activations。
2. 按层 model parallel 保留本地 parameters，跨边界发送 activations；ZeRO-3 常 all-gather 当前层 parameters。
3. $`p`$ stages 不切 microbatch 时，naive layer-wise 平均利用率约 $`1/p`$。
4. 只看 forward 的理想 pipeline 时间为 $`m+p-1`$；$`p=4,m=8`$ 时利用率 $`8/11=72.73\%`$。
5. Tensor parallel 用 column-parallel 产生分片 features，再用 row-parallel partial sums 加回完整输出。
6. 通信式必须带计数口径；p43 的 factor 8 不能脱离“4 次 activation-sized all-reduce、ring 两阶段、per-rank sends”等假设使用，换成 bytes 还要再乘 dtype 大小。

---

## 14. Activation memory：静态底座之上，还有随时间涨落的动态山峰

### 14.1 Static model state 与 dynamic activation 不是一回事

**【课程内容｜PDF p44–45｜视频 [44:55](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2695s)】**

前面算过的 parameters、gradients、optimizer states，大多是训练期间长期存在的 **static model state（静态模型状态）**。它们像一直放在仓库里的箱子。

**Dynamic activation memory（动态激活内存）** 会随 forward/backward 进度改变：

1. forward 逐层产生 activations；
2. backward 尚未到某层前，某些 activation 仍需保留；
3. backward 使用后可释放；
4. 此外，非 activation 的临时 kernel workspace、通信 buffer 和 autograd metadata 也会同时出现；kernel workspace 是 GPU 程序运行时临时工作区，autograd metadata 是自动微分系统记录计算依赖的辅助信息。

p44 的图横轴是时间 ms，纵轴是 memory GB。绿色 parameter 与黄色 optimizer state 像底座；红色 activation 和蓝色 gradient 随时间形成山峰。视频 [45:23](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2723s) 说明该图来自 profiler，也就是记录程序在不同时间做什么、占多少资源的性能分析工具。

峰值不一定正好在 forward 结束。视频 [45:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2746s) 指出，backward 刚开始时，许多 activations 尚未释放，而 gradients 已开始出现，二者重叠可能形成峰值。

### 14.2 p46 公式的准确版本和适用边界

**【课程内容｜PDF p46｜视频 [46:44](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2804s)】【论文来源：[Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)】**

高分辨率视觉核验确认，课件写的是：

```math
M_{\text{base}}
=sbh\left(34+5\frac{as}{h}\right)
\quad\text{bytes per Transformer layer}.
```

每个符号：

- $`M_{\text{base}}`$：不做这些并行/选择性重算时，每层需保存的 activation memory；单位 bytes；
- $`s`$：sequence length，每个样本多少 tokens；
- $`b`$：**microbatch size**，本次流水计算有多少样本，不是 global batch；
- $`h`$：hidden dimension，每个 token 的 hidden vector 有多少数；
- $`a`$：number of attention heads，注意力头数；
- $`t`$：tensor-parallel size；本式尚未使用，下一式才出现。

**最重要的 bytes 口径：** 原论文 Table 2 明确说这些公式给的是 bytes。$`34`$ 和 $`5`$ 已经把论文所假设的训练 activation、精度与保存项折算进系数；这里不能再把整个结果乘一次“BF16 2 bytes”，否则会重复计算。

这些系数不是 Transformer 永恒常数。论文针对其 GPT-like Transformer、普通多头注意力、当时的实现与 activation 保存方案推导。GQA（grouped-query attention，共享较少 K/V heads）、SwiGLU（带门控的 FFN）、不同 dropout、不同 kernel 或 dtype 都可能改系数。

### 14.3 为什么第二项是 sequence 的平方

把括号外的 $`sbh`$ 乘进去：

```math
sbh\times5\frac{as}{h}.
```

先约掉 $`h`$：

```math
h/h=1.
```

于是：

```math
sbh\times5\frac{as}{h}
=5ab s^2.
```

这里出现了 $`s\times s=s^2`$，所以它是 quadratic attention activation term。课件 p46 说明，这个 $`5as/h`$ 部分包括注意力矩阵及其 dropout 等需要保存的二次项。视频 [47:36](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2856s) 也明确把它归因于 quadratic attention terms。

### 14.4 基础数值：先算 $`sbh`$

**【补充理解】**

本节固定：

```math
b=2,\quad s=1024,\quad h=4096,\quad a=32,\quad t=8.
```

先算：

```math
sbh=1024\times2\times4096.
```

因为：

```math
1024\times4096=4{,}194{,}304,
```

所以：

```math
sbh=2\times4{,}194{,}304=8{,}388{,}608.
```

再算：

```math
\frac{as}{h}
=\frac{32\times1024}{4096}
=\frac{32{,}768}{4096}
=8.
```

### 14.5 Baseline：592 MiB/层

这里首次定义本节的二进制内存单位：

```math
1\ \text{MiB}=2^{20}=1{,}048{,}576\ \text{bytes}.
```

括号系数：

```math
34+5\frac{as}{h}
=34+5\times8
=34+40
=74.
```

bytes：

```math
M_{\text{base}}
=8{,}388{,}608\times74
=620{,}756{,}992\ \text{bytes}.
```

因此：

```math
620{,}756{,}992/1{,}048{,}576
=592\ \text{MiB per layer}.
```

### 14.6 TP-only：为何仍剩一个除不掉的 10

**【课程内容｜PDF p47｜视频 [47:52](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2872s)】**

高分辨率核验后的 TP 公式：

```math
M_{\text{TP}}
=sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right).
```

课件把 34 分成：

```math
34=10+24.
```

- $`24/t`$：attention/MLP 中随 TP 切开的矩阵乘相关 activation 项；
- $`5as/(ht)`$：attention heads 分片后的 quadratic 项；
- 剩余 $`10`$ 不随普通 TP 缩小。

p47 又把 10 写成：

```math
10=4+2+4.
```

| 来源 | 系数 |
|---|---:|
| LayerNorm 相关 | 4 |
| Dropout 相关 | 2 |
| attention 与 MLP 输入/残差保存 | 4 |
| 合计 | 10 |

视频 [48:42](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2922s) 说明这些 pointwise/residual-side 项不会被只切矩阵宽度的 TP 自动除以 $`t`$。

代 $`t=8`$：

```math
10+\frac{24}{8}+5\frac{32\times1024}{4096\times8}.
```

分别算：

```math
24/8=3,
```

```math
5\frac{32\times1024}{4096\times8}
=\frac{40}{8}=5.
```

括号：

```math
10+3+5=18.
```

所以：

```math
M_{\text{TP}}=8{,}388{,}608\times18
=150{,}994{,}944\ \text{bytes}.
```

```math
150{,}994{,}944/1{,}048{,}576
=144\ \text{MiB per layer}.
```

TP 已从 592 MiB 降到 144 MiB，但不是简单的 $`592/8=74`$ MiB；未切的 10 阻止它线性缩放。

### 14.7 TP+SP：74 MiB/层

**【课程内容｜PDF p49｜视频 [50:57](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3057s)】**

Sequence parallel 把剩余 pointwise activation 也沿 sequence 轴切开。公式变成：

```math
M_{\text{TP+SP}}
=sbh\left(\frac{34}{t}+5\frac{as}{ht}\right).
```

代入：

```math
\frac{34}{8}+\frac{40}{8}
=4.25+5
=9.25.
```

所以：

```math
M_{\text{TP+SP}}
=8{,}388{,}608\times9.25
=77{,}594{,}624\ \text{bytes}.
```

```math
77{,}594{,}624/1{,}048{,}576
=74\ \text{MiB per layer}.
```

这也正好等于：

```math
592/8=74\ \text{MiB}.
```

在论文公式假设中，TP+SP 才把该层所有列出的 activation 项整体除以 $`t=8`$。

### 14.8 三种结果放在一起

| 配置 | 括号系数 | bytes/层 | MiB/层 |
|---|---:|---:|---:|
| baseline | 74 | 620,756,992 | 592 |
| TP-only, $`t=8`$ | 18 | 150,994,944 | 144 |
| TP+SP, $`t=8`$ | 9.25 | 77,594,624 | 74 |

这只是论文口径的 **per-layer saved activations**。实际训练峰值还会受：

- 本 rank 有多少 layers；
- pipeline 同时在途多少 microbatches；
- forward/backward schedule；
- embedding、loss 与输出层；
- communication buffers；
- kernel workspaces；
- allocator reserved memory 与 fragmentation；
- activation checkpoint/recompute 策略

影响。不能把 `每层MiB × 本rank层数` 当成完整部署保证。

---

## 15. Recomputation、activation checkpointing 与 FlashAttention：用额外计算换内存

### 15.1 三个词先翻成人话

**【课程内容｜PDF p46、p49｜视频 [47:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2866s)】**

- **Recomputation（重计算）**：forward 后不一直保存某个中间结果；backward 真要用时，再重新执行相应 forward 计算。
- **Activation checkpointing（激活检查点）**：只保留若干层/segment 边界的 activation，丢掉段内中间值；backward 从边界重算。
- **Selective activation recomputation（选择性激活重计算）**：只重算“占内存大但重算相对便宜”的部分，而不是整层全部重跑。

三者共同交换：

```math
\text{less saved memory}\quad\Longleftrightarrow\quad\text{more FLOPs and runtime work}.
```

### 15.2 两层 tiny timeline

设：

```math
x_0\xrightarrow{L_1}x_1\xrightarrow{L_2}x_2\xrightarrow{}loss.
```

**不重算：**

| 顺序 | 动作 | 保存什么 |
|---:|---|---|
| 1 | forward $`L_1`$ | $`x_0`$ 与 $`L_1`$ 内部中间值 |
| 2 | forward $`L_2`$ | $`x_1`$ 与 $`L_2`$ 内部中间值 |
| 3 | backward $`L_2`$ | 读已保存的 $`x_1`$ 和中间值 |
| 4 | backward $`L_1`$ | 读已保存的 $`x_0`$ 和中间值 |

**把两层作为一个 checkpoint segment：**

| 顺序 | 动作 | 保存/丢弃 |
|---:|---|---|
| 1 | forward $`L_1,L_2`$ | 保存 segment 输入 $`x_0`$；丢掉段内大部分中间值 |
| 2 | backward 要进入 $`L_2`$ | 从 $`x_0`$ 重跑 $`L_1`$，重新得到 $`x_1`$；再重跑 $`L_2`$ 所需部分 |
| 3 | backward $`L_2,L_1`$ | 使用刚重建的值计算 gradients |

内存下降，但 $`L_1`$、$`L_2`$ 的部分 forward 做了第二遍。

### 15.3 Selective recompute 如何改 p49 公式

课件/论文的 selective 策略去掉 quadratic attention 保存项：

TP-only：

```math
M_{\text{TP+selective}}
=sbh\left(10+\frac{24}{t}\right).
```

本例：

```math
10+24/8=10+3=13,
```

```math
8{,}388{,}608\times13
=109{,}051{,}904\ \text{bytes}
=104\ \text{MiB/层}.
```

TP+SP+selective：

```math
M_{\text{TP+SP+selective}}
=sbh\frac{34}{t}.
```

```math
34/8=4.25,
```

```math
8{,}388{,}608\times4.25
=35{,}651{,}584\ \text{bytes}
=34\ \text{MiB/层}.
```

视频 [51:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3090s) 所说的“drop the second term”是“不再长期保存该二次 activation 项”，不是取消 attention 的 $`s^2`$ 数学计算。

### 15.4 FlashAttention 去掉什么、没有去掉什么

**HBM（High Bandwidth Memory，高带宽显存）** 是GPU旁用于存放大tensor的主显存，容量大于片上存储，但数据搬运较慢。**Online softmax（在线softmax）** 指不一次保存全部scores，而是分块维护running maximum、指数和与加权分子；§19.4会从三个数手算。**FlashAttention** 是一种精确 attention kernel：把 Q/K/V 分块搬入片上存储，用 online softmax 逐块更新，不把完整 $`[s,s]`$ attention score/probability 矩阵写入 HBM；backward 时重算所需块或利用小型统计量。

因此在 p46/p49 的教学账中，可以消掉长期保存的 $`5as/h`$ quadratic term。但它没有做到：

- attention FLOPs 从 $`O(s^2)`$ 变成 $`O(s)`$；
- 所有 activation memory 变成 0；
- Q、K、V、输出、running softmax statistics、边界 activation、workspace 全都消失。

更准确的说法：

> FlashAttention 通过 tiling 与 recomputation 避免把完整 quadratic attention matrix 常驻 HBM；它仍有线性 activation、元数据和计算代价。

### 15.5 为什么不把整个 MLP 都随便重算

视频 [52:44](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3164s) 回答课堂提问：MLP 也能重算，但要重新执行大矩阵乘，计算通常更贵。视频 [53:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3184s) 对比指出，选择性重算 attention 存储项通常更划算。

这不是说 attention 重算永远便宜；要以具体 sequence、kernel、模型宽度和硬件 profile 为准。

### 15.6 48 layers：每层小数字累积起来有多大

继续使用 §14 的同一组 $`b,s,h,a,t`$，假设每层公式相同，并暂时不计 pipeline schedule：

| 配置 | MiB/层 | 48 层 MiB | 48 层 GiB |
|---|---:|---:|---:|
| baseline | 592 | $`592\times48=28{,}416`$ | $`28{,}416/1024=27.75`$ |
| TP-only | 144 | $`144\times48=6{,}912`$ | $`6{,}912/1024=6.75`$ |
| TP+SP | 74 | $`74\times48=3{,}552`$ | $`3{,}552/1024=3.46875`$ |
| TP+SP+selective | 34 | $`34\times48=1{,}632`$ | $`1{,}632/1024=1.59375`$ |

这张表展示“每层几十 MiB × 48”会成为数 GiB；仍不是完整训练峰值。

### 15.7 Checkpoint interval：存几个边界、重算多长一段

**【延伸；基于 Korthikanti et al. 论文的 full-recompute 边界模型】**

原论文把完整 layer-input activation 的 BF16 保存近似写成 $`2sbh`$ bytes；若该边界也按 TP/SP group 分片，per rank 近似：

```math
A_{\text{boundary}}=\frac{2sbh}{t}.
```

本例：

```math
A_{\text{boundary}}
=\frac{2\times8{,}388{,}608}{8}
=2{,}097{,}152\ \text{bytes}
=2\ \text{MiB}.
```

对 $`L=48`$ 层，规定每隔 $`k`$ 层存一个 segment 输入；若 48 可整除 $`k`$，边界数按 $`48/k`$ 计：

| 间隔 $`k`$ | segment 数/保存输入数 | 只算持久边界内存 | backward 一次最多需重跑的段长 |
|---:|---:|---:|---:|
| 1 | $`48/1=48`$ | $`48\times2=96`$ MiB | 1 层 |
| 4 | $`48/4=12`$ | $`12\times2=24`$ MiB | 4 层 |
| 8 | $`48/8=6`$ | $`6\times2=12`$ MiB | 8 层 |

$`k`$ 越大，持久边界更少；但 backward 重建一个 segment 时，临时 activation 与额外 forward 工作更多。表中的 96/24/12 MiB **只数保存的 segment inputs**，没有把重算当时的 workspace、临时峰值和模型其他内存加进去。

---

## 16. Sequence parallel：把 pointwise activation 沿 token 轴切开

### 16.1 为什么 LayerNorm/dropout 可以按 sequence 切

**【课程内容｜PDF p48–49｜视频 [49:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2960s)】【论文来源：[Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)】**

**Sequence parallel（SP，序列并行）** 在本讲特指：与 tensor parallel 配合，把 LayerNorm、dropout、residual/input 等 pointwise 区域的 activations 沿 sequence 轴分片。

**Pointwise over sequence（对序列逐 token）**：token 0 的这个操作不需要读取 token 1 的值。

- LayerNorm 对一个 token 的 hidden 维做 mean/variance；只要该 token 的完整 hidden vector 在本 rank，就不需要其他 tokens。
- dropout 对元素独立应用 mask。
- residual add 对相同 token、相同 hidden position 相加。

Attention 不属于这种 pointwise 操作，因为一个 query token 要读取其他 tokens 的 K/V。SP 因此要在 attention/MLP 的 TP 区域前后转换布局，不能把整层始终当成完全独立的 token 分片。

### 16.2 $`b=1,s=4,h=2,t=2`$ 的起始数据

令：

```math
X=
\begin{bmatrix}
1&10\\
2&20\\
3&30\\
4&40
\end{bmatrix},
```

shape 是 $`[1,4,2]`$；为表格简洁，省略最前面的 batch 维。

按 sequence 切成两个 shards：

| rank | token positions | 本地数据 | 本地 shape |
|---:|---|---|---|
| 0 | 0,1 | $`[[1,10],[2,20]]`$ | $`[1,2,2]`$ |
| 1 | 2,3 | $`[[3,30],[4,40]]`$ | $`[1,2,2]`$ |

每 rank 可对自己的两个 tokens 做 LayerNorm/dropout；不需取另一个 rank 的 tokens。

### 16.3 Forward 的 $`g`$：all-gather sequence shards

**【课程内容｜PDF p48｜视频 [50:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3001s)】**

进入需要完整 sequence 布局的 TP 区域前，$`g`$ 做 all-gather：

```text
rank 0: tokens 0,1 --\
                       +--> each rank gets tokens 0,1,2,3
rank 1: tokens 2,3 --/
```

all-gather 后，每个 rank 都有：

```math
X_{full}=
\begin{bmatrix}
1&10\\
2&20\\
3&30\\
4&40
\end{bmatrix},
\qquad\text{shape }[1,4,2].
```

“每个 rank 都有完整 sequence”不等于两个 rank 做相同矩阵分片；TP 仍会让它们处理不同 hidden/features shards。

### 16.4 Forward 的 $`\bar g`$：reduce-scatter partial outputs

为了只展示通信，假设两个 TP ranks 得到以下 partial outputs，shape 都为 $`[1,4,2]`$：

```math
Y^{(0)}_{partial}=
\begin{bmatrix}
10&100\\20&200\\30&300\\40&400
\end{bmatrix},
```

```math
Y^{(1)}_{partial}=
\begin{bmatrix}
1&10\\2&20\\3&30\\4&40
\end{bmatrix}.
```

先逐元素 SUM：

```math
Y_{full}=Y^{(0)}_{partial}+Y^{(1)}_{partial}
=\begin{bmatrix}
11&110\\22&220\\33&330\\44&440
\end{bmatrix}.
```

reduce-scatter 不把完整 $`Y_{full}`$ 留在每个 rank，而是边相加边按 sequence 散开：

| rank | 得到的 SUM shard | shape |
|---:|---|---|
| 0 | $`[[11,110],[22,220]]`$ | $`[1,2,2]`$ |
| 1 | $`[[33,330],[44,440]]`$ | $`[1,2,2]`$ |

现在又回到 SP layout，可本地做后续 LayerNorm/dropout/residual。

### 16.5 Backward 为什么“反过来”

视频 [50:41](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3041s) 说 forward 与 backward 的 $`g/\bar g`$ 角色反转。逐 shape 看：

1. backward 开始时，每 rank 只有本地 sequence gradient，shape $`[1,2,2]`$。
2. forward reduce-scatter 的反向需要 all-gather，得到完整 $`dY:[1,4,2]`$。
3. 各 TP rank 反传自己的矩阵 shard，产生 partial $`dX:[1,4,2]`$。
4. forward all-gather 的反向需要 reduce-scatter SUM，把 partial $`dX`$ 相加并切回每 rank $`[1,2,2]`$。

这就是：

| 算子 | forward | backward |
|---|---|---|
| $`g`$ | all-gather | reduce-scatter |
| $`\bar g`$ | reduce-scatter | all-gather |

### 16.6 SP 与 TP 的通信量关系

原论文指出，在它的 ring bandwidth 模型中：

```math
\text{all-reduce}=\text{reduce-scatter}+\text{all-gather}.
```

经典 TP 的四次 all-reduces 被 TP+SP 的对应 all-gather/reduce-scatter 对替换，因此总 bandwidth volume 可保持同量级，同时把非 TP 区域的 sequence activations 分片。

这不等于任何实现都“零额外时间”：

- collective 次序与 latency 可能不同；
- overlap 能力不同；
- sequence 长度要能合理分片或 padding；
- kernel 是否支持该 layout 会影响性能；
- 它依赖 SP 与 TP group 配合，不是单独加一个任意 group 就奏效。

### 16.7 SP 解决了什么、没有解决什么

| 问题 | SP 的作用 |
|---|---|
| LayerNorm/dropout/residual activations | 沿 sequence 分片，约除以 $`t`$ |
| TP matrix activations | 由 TP 本身切 feature，SP 负责边界 layout |
| 参数/optimizer state | SP 本身不切；由 TP/DP/FSDP 等处理 |
| 完整 attention 的跨 token 依赖 | SP 不直接解决；进入 attention 前仍需适当 gather/layout 转换 |
| 超长序列的全层 sequence/KV 分片 | 属于 §19 的 context parallel，而不是本节 SP |

---

## 17. Expert parallel：不切一个 expert 的矩阵，改为分配完整 experts

### 17.1 EP 与 TP 切的对象不同

**【课程内容｜PDF p50–51｜视频 [53:21](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3201s)】**

- **Tensor parallel（TP）**：把一个 dense matrix/一个 expert matrix 的行或列切给多个 ranks。
- **Expert parallel（EP，专家并行）**：把不同的完整 experts 放到不同 ranks，再把 token activation 路由到拥有目标 expert 的 rank。

这里：

- **Expert（专家）**：MoE layer 中的一份 FFN；
- **Router（路由器）**：读 token hidden vector，给 experts 打分并选 top-$`k`$；
- **Dispatch（分发）**：把 token activation 送到目标 expert 所在 rank；
- **Combine/return（合并/返回）**：把 expert 输出送回 token 原来的 rank，恢复原 token 顺序并按 routing weight 合并。

视频 [53:43](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3223s) 说 EP 在系统行为上“roughly like TP”：两者都在 MLP 处引入频繁 activation communication。但 EP 使用 token routing/all-to-all，TP 使用矩阵分片与 collective；二者不能当成同一个算法。

### 17.2 两个 ranks、四个 experts 的放置

**【补充理解】**

设 top-1 routing：每个 token 只选一个 expert。

| rank | 拥有的完整 experts |
|---:|---|
| 0 | $`E_0,E_1`$ |
| 1 | $`E_2,E_3`$ |

输入的 8 个 tokens 起初分布为：

| origin rank 0 | origin rank 1 |
|---|---|
| $`T_0,T_1,T_2,T_3`$ | $`T_4,T_5,T_6,T_7`$ |

Router 选择：

| token | origin | chosen expert | expert owner | 是否跨 rank dispatch |
|---|---:|---:|---:|---|
| $`T_0`$ | 0 | $`E_0`$ | 0 | 否 |
| $`T_1`$ | 0 | $`E_2`$ | 1 | 是，0→1 |
| $`T_2`$ | 0 | $`E_3`$ | 1 | 是，0→1 |
| $`T_3`$ | 0 | $`E_1`$ | 0 | 否 |
| $`T_4`$ | 1 | $`E_1`$ | 0 | 是，1→0 |
| $`T_5`$ | 1 | $`E_2`$ | 1 | 否 |
| $`T_6`$ | 1 | $`E_2`$ | 1 | 否 |
| $`T_7`$ | 1 | $`E_0`$ | 0 | 是，1→0 |

### 17.3 第一次 all-to-all：dispatch

**All-to-all（全互换）**：group 中每个 rank 都可给每个其他 rank 发送不同数据块。它不要求每块相同，也不保证 split sizes 均匀。

dispatch 后：

| compute rank | 本地保留 | 从别处收到 | 最终计算的 tokens | token 数 |
|---:|---|---|---|---:|
| 0 | $`T_0\to E_0,T_3\to E_1`$ | $`T_4\to E_1,T_7\to E_0`$ | $`T_0,T_3,T_4,T_7`$ | 4 |
| 1 | $`T_5\to E_2,T_6\to E_2`$ | $`T_1\to E_2,T_2\to E_3`$ | $`T_1,T_2,T_5,T_6`$ | 4 |

每个 token 随 dispatch 携带至少：

- hidden activation；
- origin rank；
- 原 token index；
- expert id；
- routing weight 或能恢复它的信息。

这些索引/描述信息称为 routing metadata；它们不是 hidden vector 主 payload，但仍占空间和处理时间。

### 17.4 Expert compute 与第二次 all-to-all

各 owner 用本地 expert 计算：

```math
O_i=E_{route(i)}(H_i).
```

随后 return all-to-all：

- rank 0 把 $`O_4,O_7`$ 发回 origin rank 1；
- rank 1 把 $`O_1,O_2`$ 发回 origin rank 0；
- 本地 token outputs 不跨网；
- origin rank 按 $`T_0,T_1,\ldots`$ 原顺序放回。

若是 top-2，token 会复制给两个 selected experts，回来后按 gate weights 加权相加；通信和计算都比这个 top-1 例更大。

### 17.5 用 $`h=4`$、BF16 算通信 bytes

每个 token hidden vector：

```math
h\times2\ \text{bytes}=4\times2=8\ \text{bytes}.
```

dispatch 时每 rank 向对方发送 2 个远程 tokens：

```math
2\times8=16\ \text{payload bytes sent per rank}.
```

return 时也发送 2 个 outputs：

```math
2\times8=16\ \text{payload bytes sent per rank}.
```

一来一回 per rank send：

```math
16+16=32\ \text{bytes}.
```

两 ranks aggregate sends：

```math
2\times32=64\ \text{bytes}.
```

这些只算 BF16 hidden/output payload；没算 routing metadata、对齐、capacity padding、协议和 receives。

### 17.6 Load imbalance 怎样制造慢尾

**Load imbalance（负载不均）**：某些 ranks/experts 收到的 tokens 多，另一些少。同步进入下一层前，快 rank 常要等最慢 rank，形成 **straggler tail（慢尾）**。

改成以下路由：

| expert | tokens | owner |
|---|---|---:|
| $`E_0`$ | $`T_0,T_1,T_2,T_3,T_4`$ | 0 |
| $`E_1`$ | $`T_7`$ | 0 |
| $`E_2`$ | $`T_5`$ | 1 |
| $`E_3`$ | $`T_6`$ | 1 |

rank 0 算：

```math
5+1=6\ \text{tokens}.
```

rank 1 算：

```math
1+1=2\ \text{tokens}.
```

平均：

```math
(6+2)/2=4\ \text{tokens/rank}.
```

若每 token expert compute 时间相同，最慢 rank 相对理想平均的负载倍数：

```math
6/4=1.5.
```

所以即使总 token 数没变，step 也可能接近由 6-token rank 决定，而不是平均 4-token rank。

### 17.7 为什么 expert layer 常优先 EP，而不是把每个 expert 再切很小

**【课程内容｜PDF p51｜视频 [54:21](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3261s)】【一手补充：[Megatron Core MoE 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)】**

课件与 NVIDIA 指南给的系统直觉：

- EP 保留较大的 local expert GEMM；
- 过高 TP 会把一个 expert matrix 切成很小 GEMMs，GPU utilization 可能下降；
- EP 只把 token 发给它实际选择的 experts，而不是为整个 dense MLP 交换所有 TP partials；
- dispatch/compute/return 有时更易分阶段 overlap。

这不是“EP 在任何硬件上永远快于 TP”。视频 [55:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3347s) 随即强调，高效 EP dispatch 仍是复杂的系统工程。

### 17.8 EP 没有自动切掉整块模型的所有东西

EP 主要 shard routed expert weights。下面这些可能仍 replicated，或由其他 parallel axis 处理：

- attention parameters 与 KV/attention activations；
- router；
- LayerNorm；
- shared expert；
- embeddings/output head；
- residual-stream activation。

p55 的课程总表因此把 EP 的 activation/KV memory scaling 标成 `None`：EP 可能减少 local expert intermediates 或只计算收到的 tokens，但它不保证整个 Transformer activation/KV 都按 $`1/EP`$ 缩放。

视频 [57:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3421s) 点明性能难点：几乎每个 MoE MLP 都有 latency-sensitive all-to-all dispatch，expert compute 必须等目标 tokens 到齐。

---

## 18. Attention 与 MoE 需要不同 parallel groups：用 8 GPUs 看“解耦”

### 18.1 为什么不能永远把所有并行轴当 LEGO 随便乘

**【课程内容｜PDF p52｜视频 [57:59](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3479s)】**

DP、TP、PP 常可组合，但组合仍受 shape、group size、拓扑和通信影响。MoE 又多出 token routing 与 expert ownership。

p52 的两个提醒：

1. 较早/朴素的组合中，EP group 常取自一个 DP group 内，因此 EP degree 被 DP group size 限制；课件写的 `EP<DP` 应读成这种 group inclusion/degree 约束，不是所有系统永恒要求严格小于。
2. DP 缩小 local token batch，TP 又缩小 local matrix；两者同时很大时，local GEMM 可能小到 utilization 下降。

视频 [58:28](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3508s) 用 DP=8 的口头例解释了“在这些 replicas 中再切 experts”的旧式自然映射。

### 18.2 MoE 只替换 MLP，attention 仍是另一套工作

**【课程内容｜PDF p53｜视频 [59:32](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3572s)】**

典型 MoE Transformer：

```text
attention: 仍是 dense attention
MLP:       被 router + routed experts 替换
```

于是需求冲突：

- attention 可能需要较大的 TP 或 CP；
- expert MLP 更希望用 EP，且不想让 expert tensor matrix 被过度 TP 切小；
- 同一个固定 TP degree 若同时套 attention 和 experts，可能顾此失彼。

视频 [60:08](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3608s) 明确说，高 TP 与高 EP 叠加可能让 expert matrices 变得太小，降低 utilization。

### 18.3 Parallel folding 的两组轴

课件 p53 与当前 Megatron Core 文档把它写成：

| 模型部分 | 逻辑并行轴 |
|---|---|
| Attention layers | $`TP\times CP\times DP\times PP`$ |
| MoE layers | $`ETP\times EP\times EDP\times PP`$ |

新缩写：

- **CP（context parallel）**：沿长序列分片整个网络 inputs/activations，§19 详解；
- **ETP（expert tensor parallel）**：把单个 expert matrix 再做 tensor parallel；
- **EP（expert parallel）**：不同完整 experts 分给不同 ranks；
- **EDP（expert data parallel）**：复制 expert set，处理不同数据，并同步对应 expert gradients；
- **PP（pipeline parallel）**：attention 与 MoE 共用的层深度/stage 轴。

“解耦”是 attention 进入一组 process groups，MoE MLP 进入另一组 process groups；不是创建两套物理 GPUs。

### 18.4 8 GPUs 的一组教学坐标：attention 侧

**【补充理解；不是唯一 Megatron 配置】**

设物理 ranks 0–7。Attention 选择：

```math
TP=2,\quad CP=2,\quad DP=2,\quad PP=1,
```

核对：

```math
2\times2\times2\times1=8.
```

用坐标 $`(d,c,t)`$，rank 映射为：

```math
r=4d+2c+t.
```

| rank | $`(d,c,t)`$ |
|---:|---|
| 0 | (0,0,0) |
| 1 | (0,0,1) |
| 2 | (0,1,0) |
| 3 | (0,1,1) |
| 4 | (1,0,0) |
| 5 | (1,0,1) |
| 6 | (1,1,0) |
| 7 | (1,1,1) |

固定另外两维，只让目标维变化：

- Attention TP groups：`{0,1}`, `{2,3}`, `{4,5}`, `{6,7}`；
- Attention CP groups：`{0,2}`, `{1,3}`, `{4,6}`, `{5,7}`；
- Attention DP groups：`{0,4}`, `{1,5}`, `{2,6}`, `{3,7}`。

例如 rank 0：与 rank 1 一起切 attention matrix；与 rank 2 交换 sequence/KV chunks；与 rank 4 是同一 attention shard 的 data replica。

### 18.5 同样 8 GPUs：MoE MLP 侧换一套坐标

MoE 选择：

```math
ETP=1,\quad EP=4,\quad EDP=2,\quad PP=1.
```

核对：

```math
1\times4\times2\times1=8.
```

用坐标 $`(edp,ep)`$：

```math
r=4\,edp+ep.
```

| rank | $`(edp,ep)`$ | MoE 意义 |
|---:|---|---|
| 0 | (0,0) | replica 0 的 expert shard 0 |
| 1 | (0,1) | replica 0 的 expert shard 1 |
| 2 | (0,2) | replica 0 的 expert shard 2 |
| 3 | (0,3) | replica 0 的 expert shard 3 |
| 4 | (1,0) | replica 1 的 expert shard 0 |
| 5 | (1,1) | replica 1 的 expert shard 1 |
| 6 | (1,2) | replica 1 的 expert shard 2 |
| 7 | (1,3) | replica 1 的 expert shard 3 |

groups：

- ETP=1：每个 ETP group 是单个 rank；expert matrix 不再横切。
- EP groups：`{0,1,2,3}`、`{4,5,6,7}`，组内 dispatch tokens 到四份 expert ownership。
- EDP groups：`{0,4}`, `{1,5}`, `{2,6}`, `{3,7}`，对应 expert shard 在两份数据副本间同步。

### 18.6 同一个 rank 在两种 layer 中身份不同

以 rank 2 为例：

- attention 时坐标 $`(d=0,c=1,t=0)`$；它属于 TP `{2,3}`、CP `{0,2}`、DP `{2,6}`；
- MoE MLP 时坐标 $`(edp=0,ep=2)`$；它属于 EP `{0,1,2,3}`、EDP `{2,6}`，没有 ETP partner。

这就是“attention TP=2，但 expert ETP=1、EP=4”的具体含义。视频 [60:27](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3627s) 说的 decouple，就是允许这两个模型部分用不同 tensor/expert 切法。

### 18.7 为什么这不是唯一合法配置

上面的 rank 编号与连续 groups 只是为了让初学者能列集合。真实配置还要根据：

- experts 数与 top-$`k`$；
- node/NVLink/network topology；
- attention heads 与 hidden dimensions 的可整除性；
- local token count 与 load balance；
- PP stages；
- Megatron Core 版本和 parallel-folding implementation

决定。不能从 p53 推出“所有 8-GPU MoE 都必须使用这组 rank 列表”。

一手实现边界见 [Megatron Core MoE 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)。该页面随版本更新；本节只把课件 p53 的逻辑轴做成教学坐标，不声称复现唯一生产配置。

---

## 19. Context parallel 与 Ring Attention：query 留本地，让 KV blocks 轮流经过

### 19.1 CP 与 SP 名字相近，范围不同

**【课程内容｜PDF p54–55｜视频 [61:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3660s)】**

- **SP（本讲的 sequence parallel）**：主要切 LayerNorm/dropout/residual 等 sequence-pointwise activation；在 TP attention/MLP 边界用 all-gather/reduce-scatter 转换。
- **CP（context parallel）**：沿 sequence 轴 partition network inputs 与各层 activations；attention 中每个 local query 仍需访问全序列 K/V，因此专门交换 KV。
- **Ring Attention**：一种让 KV blocks 沿 ring 移动、逐块完成 attention 的算法思想。

课上约 [61:08](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3668s) 把 “context parallel or ring attention” 放在一起讲高层直觉。工程上不应把名字完全画等号：CP 是更广的 parallel dimension；ring KV exchange 是实现 attention 通信的一类方法。Megatron Core 当前 CP 也说明其实现与原始 Ring Attention 相似但不完全相同。

一手边界：

- [Ring Attention 原论文](https://arxiv.org/abs/2310.01889)
- [Megatron-LM Context Parallel 官方说明](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md)

### 19.2 四个 causal tokens 分给两个 ranks

设 positions 0–3：

| rank | local query block | 初始 local KV block |
|---:|---|---|
| 0 | $`Q_0,Q_1`$ | $`(K_0,V_0),(K_1,V_1)`$ |
| 1 | $`Q_2,Q_3`$ | $`(K_2,V_2),(K_3,V_3)`$ |

**Causal mask（因果遮罩）** 要求 position $`i`$ 只能看 $`j\le i`$ 的 keys：

| query | 最终允许 keys |
|---:|---|
| $`Q_0`$ | $`K_0`$ |
| $`Q_1`$ | $`K_0,K_1`$ |
| $`Q_2`$ | $`K_0,K_1,K_2`$ |
| $`Q_3`$ | $`K_0,K_1,K_2,K_3`$ |

### 19.3 两个 compute rounds 的 KV 环流表

**【补充理解；对 p54 高层箭头逐轮展开】**

Queries 始终留在 owner rank；只有 KV transport blocks 移动。

| round | rank 0 的 local queries 看哪个 KV block | causal 后真正贡献 | rank 1 的 local queries看哪个 KV block | causal 后真正贡献 |
|---:|---|---|---|---|
| 0 | $`Q_0,Q_1`$ × local $`KV_{0:1}`$ | $`Q_0\leftarrow K_0`$；$`Q_1\leftarrow K_0,K_1`$ | $`Q_2,Q_3`$ × local $`KV_{2:3}`$ | $`Q_2\leftarrow K_2`$；$`Q_3\leftarrow K_2,K_3`$ |
| 1 | $`Q_0,Q_1`$ × received $`KV_{2:3}`$ | 全是未来 positions，全部 masked | $`Q_2,Q_3`$ × received $`KV_{0:1}`$ | 两个 queries 都可看 $`K_0,K_1`$ |

两轮后：

- $`Q_0`$ 累积了 $`K_0`$；
- $`Q_1`$ 累积了 $`K_0,K_1`$；
- $`Q_2`$ 把 round 0 的 $`K_2`$ 与 round 1 的 $`K_0,K_1`$ 合起来；
- $`Q_3`$ 合起全部 $`K_0,K_1,K_2,K_3`$。

因此每个 local query 都覆盖了其全部合法 causal keys。

### 19.4 不能把每轮各自 softmax 后直接平均

先从零定义 **softmax**。有scores $`z_1,\ldots,z_n`$ 时，第$`i`$个权重是：

```math
p_i=\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.
```

- $`\exp(x)=e^x`$；$`e\approx2.718`$ 是自然指数的底数；
- $`e^0=1`$；
- 分母把所有合法scores的指数相加，因此全部$`p_i`$加起来等于1。

#### 三个相同score，为什么分块平均会错

设三个scores都是0，values是：

```math
z=[0,0,0],\qquad v=[0,0,3].
```

**正确的global softmax：**

```math
e^0=e^0=e^0=1,
```

```math
\text{denominator}=1+1+1=3.
```

所以weights是：

```math
[1/3,1/3,1/3].
```

正确output：

```math
\frac13\times0+\frac13\times0+\frac13\times3=1.
```

现在错误地分成两块：前两枚`[0,0]`与后一枚`[0]`。

- 第一块local softmax是$`[1/2,1/2]`$，local output $`=0`$；
- 第二块只有一项，local softmax是$`[1]`$，local output $`=3`$；
- 若把两个block outputs等权平均：$`(0+3)/2=1.5`$。

这个错误算法对三个values的实际weights是：

```math
[1/4,1/4,1/2],
```

不是正确的$`[1/3,1/3,1/3]`$。原因是“每块权重和都被单独强制成1”，一个只有1项的块竟与另一个有2项的块同权。

#### Online softmax 怎样保留全局分母

**Online softmax（在线softmax）** 不是平均local outputs；它为每个query逐块维护：

- running maximum；
- running exponential sum；
- running weighted-value numerator。

在上例，全部score都是0，所以running maximum一直是0：

| 处理后 | running max $`m`$ | exponential sum $`\ell`$ | weighted numerator $`n`$ |
|---|---:|---:|---:|
| 第一块 | 0 | $`1+1=2`$ | $`1\times0+1\times0=0`$ |
| 再并第二块 | 0 | $`2+1=3`$ | $`0+1\times3=3`$ |

最终：

```math
\text{output}=n/\ell=3/3=1.
```

若新块maximum更大，还必须把旧统计量重缩放。令旧状态为$`(m_{old},\ell_{old},n_{old})`$，新块统计为$`(m_b,\ell_b,n_b)`$：

```math
m_{new}=\max(m_{old},m_b),
```

```math
\ell_{new}
=e^{m_{old}-m_{new}}\ell_{old}
+e^{m_b-m_{new}}\ell_b,
```

```math
n_{new}
=e^{m_{old}-m_{new}}n_{old}
+e^{m_b-m_{new}}n_b.
```

最终仍是$`n_{new}/\ell_{new}`$。减去running maximum让指数不容易溢出；指数缩放又把不同blocks放回同一个全局分母。

这与 FlashAttention 的 online softmax 思路相通：不保存完整 attention matrix，但精确维护全局归一化所需统计量。

### 19.5 Memory 为什么约按 $`1/CP`$，又为什么不是所有内存都除掉

设 CP degree 为 $`c`$，sequence 长度为 $`s`$。每 rank 持久拥有约：

```math
s/c\ \text{tokens}.
```

因此 sequence-side activation 与本地 KV shard 常近似变为原来的：

```math
1/c.
```

在本例 $`s=4,c=2`$：

```math
s/c=4/2=2\ \text{tokens per rank}.
```

但峰值还可能有：

- 正在发送的 KV block；
- 正在接收的下一个 KV block；
- double buffering；
- local Q、output、online-softmax statistics；
- parameters/optimizer states；
- communication workspace。

所以只能说“sequence-side activation/KV 持久分片约 $`1/CP`$”，不能说整张 GPU memory 精确除以 $`CP`$。

### 19.6 通信与 latency 没有消失

对 $`c`$ 个 CP ranks，每个 local query block 要依次处理 $`c`$ 份 KV blocks，其中 1 份本地、$`c-1`$ 份来自其他 ranks。Ring 实现通常有 $`c`$ 个 compute rounds，并在 rounds 间传 KV block。

收益：

- 每 rank 只永久保存一段长序列；
- 可把更长 context 分到更多 devices；
- 理想情况下把 KV 通信与 blockwise attention compute overlap。

代价：

- $`c-1`$ 份远程 KV 必须到达；
- rounds 带来 latency 与同步依赖；
- causal 三角结构可能让某些 rank/round 工作少，如上表 rank 0 的 round 1 全 masked；
- 生产实现需要更聪明的 token placement/负载均衡。

视频 [61:17](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3677s) 只给出了“activation 沿 ring 传到需要的 device”的高层说明；本节的逐轮 causal 表是补充教学，不应误称为课件展示的完整 kernel schedule。

### 19.7 SP、CP、Ring Attention 最后比较

| 名称 | 主要切什么 | Attention 怎么办 | 主要内存收益 | 主要通信 |
|---|---|---|---|---|
| SP | pointwise 区域的 sequence activations | 在 TP 区域前后转换 layout | TP group 内非矩阵 activation 约 $`1/t`$ | AG/RS，替代部分 TP all-reduce |
| CP | 全层 inputs/activations 的 sequence 轴 | local Q 需收集/流过全序列 KV | sequence activation/KV 约 $`1/c`$ | KV exchange，backward 还有对应 gradient communication |
| Ring Attention | blockwise attention 的 KV transport/compute algorithm | KV blocks 按 ring 逐轮经过 local Q | 不物化全序列 KV 在单卡常驻 | 每轮邻居 p2p，可尝试与 compute overlap |

### 19.8 p55 的课程结论：没有单一并行策略全面占优

**【课程内容｜PDF p55｜视频 [61:52](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3712s)】**

| 方法 | 主要省什么 | 主要付出什么 |
|---|---|---|
| DDP/ZeRO-1 | optimizer state（ZeRO-1）与 compute scaling | global batch、gradient traffic；parameters 不切 |
| FSDP/ZeRO-3 | parameter/gradient/state | parameter gather、activation 不自动切 |
| PP | 按层 parameters | bubble、schedule、边界 activation p2p |
| TP+SP | 每层 matrix parameters 与相关 activations | 频繁阻塞式 activation collectives |
| EP | routed expert weights | token all-to-all、load imbalance；全层 activation 不自动切 |
| CP | sequence-side activation/KV | KV communication、round latency |

视频 [62:18](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3738s) 的主结论是：没有严格支配其他方法的单一策略。真实大模型通常根据 memory、batch、sequence、拓扑和模型结构组合多个 axes。

---

## 20. 把十种并行方法放进同一张表：到底切了什么

### 20.1 先规定表里的两个“每卡”口径

**【课程内容｜PDF p55｜视频 [62:22](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3742s)】**

下面的“parameter memory”只问长期驻留的模型参数是否分片；“activation memory”只问该方法本身是否分片激活。它们不是完整峰值显存。完整峰值还含 gradients、optimizer states、通信 buffers、临时工作区和 allocator 余量。

令：

- $`P`$：全模型 parameter 个数；
- $`d`$：data-parallel degree；
- $`t`$：tensor-parallel degree；
- $`p`$：pipeline stages；
- $`e`$：expert-parallel degree；
- $`c`$：context-parallel degree。

“约 $`1/x`$”都需要切分均匀、shape 可整除，而且不计临时聚合峰值。

### 20.2 统一比较表

| 方法 | 主要切什么 | 每 rank 长期 parameter | 每 rank activation | 典型通信 | 会不会自然扩大 global batch | 最大的限制 |
|---|---|---|---|---|---|---|
| DDP | batch/samples | 完整 $`P`$ | 每 rank 的 local batch | gradient all-reduce | 会，若 local batch 不变则乘 $`d`$ | 不省 model-state memory |
| ZeRO-1 | optimizer state | 完整 $`P`$ | 与普通 DP 相同 | gradient RS + updated parameter AG；逻辑量约 DDP | 会 | parameter、gradient 仍复制 |
| ZeRO-2 | optimizer state + gradient | 完整 $`P`$ | 与普通 DP 相同 | gradient RS + parameter AG | 会 | parameter 仍复制；activation 不降 |
| ZeRO-3 | parameter + gradient + optimizer state | 长期约 $`P/d`$，计算前临时 AG layer | 与普通 DP 相同 | forward/backward parameter AG + gradient RS | 会 | 通信更多；activation 不自动降 |
| FSDP | 通常采用 ZeRO-3 类 full sharding | 长期约 $`P/d`$ | 不自动下降 | module parameter AG + gradient RS | 会 | wrap/prefetch/free 策略影响峰值与速度 |
| PP | 连续 layer stages | 约 $`P/p`$ | 与 schedule、在途 microbatches 有关 | stage 边界 activation/gradient point-to-point | 不因 $`p`$ 自动扩大 | pipeline bubble、负载不均、schedule 复杂 |
| TP | 一个 layer 内的 matrix/head/hidden width | 相关 weights 约 $`1/t`$ | 只做 TP 时不一定整体 $`1/t`$ | 每层频繁 AG/RS/all-reduce | 不会 | 要高速低延迟链路；切太细 matrix utilization 下降 |
| SP | pointwise activation 的 sequence 轴 | 不切 parameter | 相关 activation 约 $`1/t`$ | 常与 TP 共享 group 做 AG/RS | 不会 | 通常依附 TP；不是额外独立 degree |
| EP | whole routed experts | routed-expert weights 约 $`1/e`$ | 普通/attention activation 不自动降 | token all-to-all + return | 不会 | load imbalance、跨节点 A2A、每 expert token 太少 |
| CP | inputs/activations 的 context 轴 | 不切 parameter | sequence-side activation/KV 约 $`1/c`$ | attention KV block exchange | 不会 | ring rounds、causal imbalance、通信 latency |

**RS** 是 reduce-scatter；**AG** 是 all-gather；**point-to-point** 是一端明确 send 给另一端，而不是整个 group 一起参加 collective。

### 20.3 p55 的表为什么只能当“第一层地图”

视频 [62:29](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3749s) 说 FSDP 很好，但它不帮助 activation memory；[62:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3769s) 又说 TP 能切 activation，却要求高 bandwidth。这正说明每一列都有条件。

课件 p55 做了几处教学简化：

1. 把 ZeRO stages/FSDP 合并成少数行，没有展开不同 stage 的 persistent state 差异。
2. 把通信写成短标签；实际是 RS、AG、all-reduce 或 P2P 的组合，不是一个固定物理算法。
3. “easy to use”等判断带主观性，也随框架版本改变。
4. 表中的 activation 项只说该 axis 的主要效果，不代表完整训练峰值。
5. SP 常复用 TP group，CP 可以成为独立 axis；不能因为二者都切 sequence 就混为一谈。

所以这张表的正确用途是：先找到“哪个资源被切”，再回到 shape、bytes、collective 和 topology 做账。

### 20.4 一道选择题的完整判断

问题：模型静态状态已能放下，但长序列 activation OOM；高速节点内有 8 GPUs，跨节点较慢。先考虑什么？

1. 先确认 OOM 主体是 activation，而不是 parameter。
2. ZeRO-1 只切 optimizer state，方向不对。
3. TP+SP 可切 matrix 与 pointwise activation，并适合节点内高速连接。
4. 若主要是超长 context，再评估 CP。
5. 如果仍需跨节点装下 layers，可叠加 PP。

这不是“TP 一定最佳”，而是按证据缩小候选。视频 [63:13](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3793s) 也只说许多大型架构会组合多种方法，没有宣布唯一答案。

### 20.5 p56 cost model：先把FLOPs与bytes变成时间，才能读三条曲线

**【课程内容｜PDF p56，6倍原分辨率复核｜视频64:04–66:19】【一手推导：[How to Scale Your Model](https://jax-ml.github.io/scaling-book/training/)】**

这页只给一个Transformer MLP layer做模型，并注明忽略gating einsum；这里的 **MP** 指model/tensor parallel，不是pipeline parallel。**Einsum（Einstein summation，爱因斯坦求和写法）** 用字母下标描述tensor乘加；“gating einsum”是门控/路由相关的额外tensor乘加，p56明确没有把它计入本表。

#### 六个符号先逐个固定

符号会随公式语境复用：前文DP小例把$`B`$说成“样本数”，而p56/Scaling Book把一个sequence里的tokens也摊平，$`B`$专指global batch的token总数。以下只按p56口径。

- $`B`$：global batch中的**token总数**；不是sequence条数；单位tokens。
- $`D`$：model/embedding hidden dimension；每token主hidden vector的宽度；单位是元素个数。
- $`F`$：feed-forward hidden dimension，也就是MLP中间宽度；单位是元素个数。
- $`X`$：用于data/FSDP sharding的chips数；每chip约处理$`B/X`$ tokens。
- $`Y`$：用于model/tensor parallel sharding的chips数；每chip约持有$`1/Y`$的相关MLP宽度。
- $`N`$：总chips数；在正交混合FSDP+MP模型里：

  $`N=XY.`$

$`X,Y,N`$都是计数，没有物理单位。

#### p56表格四行原公式

表头说明compute是“per layer，忽略gating einsum”，communication是“bytes，forward + backward pass”。每个加号左边是forward，右边是backward：

| Strategy | Compute per layer | Comms per layer |
|---|---:|---:|
| DP | $`4BDF/X+8BDF/X`$ | $`0+8DF`$ |
| FSDP | $`4BDF/X+8BDF/X`$ | $`4DF+8DF`$ |
| MP | $`4BDF/Y+8BDF/Y`$ | $`4BD+4BD`$ |
| FSDP + MP | $`4BDF/(XY)+8BDF/(XY)`$ | $`(4BD/X+4DF/Y)+(8BD/X+8DF/Y)`$ |

#### 先别背4和8：compute系数从一个乘加数起

先看没有并行的一层简化MLP。它只有两个权重矩阵：

```math
W_{in}:[D,F],\qquad W_{out}:[F,D].
```

对$`B`$个tokens，forward有两个矩阵乘：

1. $`[B,D]\times[D,F]\to[B,F]`$；每个输出格要做$`D`$次multiply-add，总共有$`BDF`$个multiply-add。
2. $`[B,F]\times[F,D]\to[B,D]`$；同样有$`BDF`$个multiply-add。

一个 **multiply-add（乘加）** 是“先乘一次，再加一次”，课件按$`1+1=2`$ FLOPs记。因此一个矩阵乘是：

```math
BDF\times2=2BDF\ \text{FLOPs}.
```

两个forward矩阵乘合计：

```math
2BDF+2BDF=4BDF.
```

Backward对每个权重层都要算两类东西：输入梯度$`dX`$与权重梯度$`dW`$。两层一共是下面4个同量级矩阵乘：

| 原forward层 | backward矩阵乘1 | backward矩阵乘2 | FLOPs |
|---|---|---|---:|
| $`W_{out}`$ | 算$`dTmp`$，也就是这层的$`dX`$ | 算$`dW_{out}`$ | $`2BDF+2BDF`$ |
| $`W_{in}`$ | 算$`dIn`$，也就是这层的$`dX`$ | 算$`dW_{in}`$ | $`2BDF+2BDF`$ |

所以backward合计：

```math
4\times2BDF=8BDF.
```

这就是compute列中的$`4BDF+8BDF`$。DP/FSDP让每chip只处理$`B/X`$个tokens，所以除$`X`$；MP把MLP宽度切成$`Y`$份，所以除$`Y`$；两者同时用时除$`XY`$。

#### Weight communication的4DF与8DF从哪里来

每个权重矩阵都有$`DF`$个元素。两个矩阵共有：

```math
DF+DF=2DF\ \text{elements}.
```

p56按bfloat16每元素2 bytes记，所以两矩阵本体共：

```math
2DF\times2=4DF\ \text{bytes}.
```

- **FSDP forward的$`4DF`$：** 依次AG $`W_{in}`$与$`W_{out}`$。两个完整bfloat16矩阵的逻辑payload合计就是$`4DF`$ bytes。
- **FSDP backward的$`8DF`$：** 以一个矩阵为例，为算输入梯度要AG它的weight，payload为$`2DF`$ bytes；算完本地weight gradient后要RS，gradient也是$`2DF`$ bytes。因此一个矩阵是$`2DF+2DF=4DF`$，两个矩阵是$`2\times4DF=8DF`$。
- **普通DP backward的$`8DF`$：** 没有forward weight AG；但两个完整gradient各做一次all-reduce。该页的一维有效带宽模型把一次all-reduce记为约“tensor bytes的2倍”，所以$`2`$个矩阵$`\times2DF`$ bytes/矩阵$`\times2=8DF`$。

这里的AG是all-gather，RS是reduce-scatter。以上$`4DF/8DF`$是课件在**有效双向带宽**下的一阶逻辑payload账，不是任意网络上每条link实际经过的wire bytes。真实值还会带$`(p-1)/p`$、ring/tree算法、mesh axes、分块、重叠与协议开销；缓存forward权重而不在backward重取，也会用更多显存换通信。

#### MP activation communication为什么forward与backward各是4BD

一个$`[B,D]`$的bfloat16 activation有：

```math
B\times D\times2=2BD\ \text{bytes}.
```

Standalone MP的forward包含：

1. 第一层矩阵乘前，AG输入$`In[B,D]`$：$`2BD`$ bytes；
2. 第二层矩阵乘后，RS输出$`Out[B,D]`$：$`2BD`$ bytes。

所以forward是：

```math
2BD+2BD=4BD.
```

Backward是它的梯度方向对应物：

1. AG $`dOut[B,D]`$：$`2BD`$ bytes；
2. RS $`dIn[B,D]`$：$`2BD`$ bytes。

所以backward也是$`4BD`$。这里假设算$`dW_{in}`$所需的$`In`$由forward保存或复用，没有额外再AG一次。

混合FSDP+MP时，每个FSDP shard只处理$`B/X`$个tokens，所以forward activation项变成$`4BD/X`$；每份weight又只有$`F/Y`$宽，所以weight项是$`4DF/Y`$。表中混合行把backward两类通信统一近似为forward的2倍，即$`8BD/X+8DF/Y`$。其中weight的$`8DF/Y`$可对应“weight AG+gradient RS”；activation的精确操作数则取决于是否保存/重取$`In`$、布局转换与实现schedule，不能永远逐项映射成同一组collectives。**因此该混合行是用于看主导缩放的一阶逻辑payload模型，不是生产实现的逐kernel wire trace。**

逐行翻成人话：

1. DP/FSDP都把tokens沿$`X`$切，因此每chip compute除以$`X`$；DP forward不通信weights，backward同步gradients。
2. FSDP compute相同，但forward还要取weights，所以多$`4DF`$；backward账为$`8DF`$。
3. MP沿$`Y`$切MLP宽度，因此compute除$`Y`$；它搬的是与batch有关的activation，所以communication含$`BD`$。
4. 混合时compute除总chips $`XY`$；activation流量又被$`X`$分小，weight流量又被$`Y`$分小。

#### 单位检查：不能直接拿FLOPs除bytes

- $`BDF`$是三个元素计数相乘；乘课件系数后是FLOPs。
- $`BD`$或$`DF`$是tensor元素量；communication列的系数已经纳入该页bfloat16与collective流量口径，所以结果是bytes。
- 计算设备吞吐记为$`C`$ FLOP/s，网络有效带宽记为$`W`$ bytes/s，则：

```math
T_{math}=\frac{\text{FLOPs}}{C},
\qquad
T_{comms}=\frac{\text{bytes}}{W}.
```

两个结果单位都是seconds，才可以形成无单位比值：

```math
R=\frac{T_{math}}{T_{comms}}.
```

- $`R>1`$：compute time更长，称compute-bound；communication有机会藏在计算下面。
- $`R<1`$：communication time更长，称communication-bound。
- $`R=1`$：两者正好相等，是图中的虚线边界。视频64:49–65:03正在解释这条边界。

#### 极小数字：四行都亲手算一次

取：

```math
B=8,\quad D=2,\quad F=4,\quad X=2,\quad Y=2,\quad N=XY=4.
```

先算$`BDF=8\times2\times4=64`$、$`BD=16`$、$`DF=8`$。

| Strategy | Compute展开 | FLOPs | Comms展开 | bytes |
|---|---|---:|---|---:|
| DP | $`4\times64/2+8\times64/2`$ | $`128+256=384`$ | $`0+8\times8`$ | 64 |
| FSDP | 同DP | 384 | $`4\times8+8\times8`$ | $`32+64=96`$ |
| MP | $`4\times64/2+8\times64/2`$ | 384 | $`4\times16+4\times16`$ | $`64+64=128`$ |
| FSDP+MP | $`4\times64/4+8\times64/4`$ | $`64+128=192`$ | $`(4\times16/2+4\times8/2)+(8\times16/2+8\times8/2)`$ | $`48+96=144`$ |

再假设一个纯教学设备$`C=24`$ FLOP/s、$`W=12`$ bytes/s。以FSDP行为例：

```math
T_{math}=384/24=16\ \text{s},
```

```math
T_{comms}=96/12=8\ \text{s},
```

```math
R=16/8=2>1.
```

因此这个玩具FSDP例是compute-bound。这组随意小数只演示单位与代入，不复现p56 TPU 4×4×4 mesh的具体曲线。

#### 图的横轴、纵轴与三个batch regions

横轴是：

```math
B/N=\text{global batch tokens divided by total chips},
```

也就是每chip分到的平均tokens。纵轴是$`R=T_{math}/T_{comms}`$，采用 **log scale（对数刻度）**：相同竖直距离代表乘相同倍数，例如$`0.1\to1\to10`$，而不是每格加同一个数。黑色水平虚线是$`R=1`$。

- **MP only，橙线：** 近似水平且低于1；因为compute与activation communication都随$`B`$一起增长，比值在该模型里近似不变。
- **FSDP only，蓝线：** 随$`B/N`$近似线性升高，约在850跨过1。视频65:20说明大batch时FSDP可compute-bound；65:40说明batch变小时会跌入communication-bound。
- **FSDP+MP，绿线：** 优化$`X/Y`$后随$`B/N`$约按平方根升高，约在400跨过1；视频65:51说加入MP把可用区推向更小batch。

#### 为什么优化后的混合曲线随$`\sqrt{B}`$增长

**“优化$`X/Y`$”只是在固定总设备数$`N`$下选择mesh形状：多少chips放在FSDP轴$`X`$、多少放在MP轴$`Y`$。它不改变batch $`B`$，也不改变模型的$`D`$或$`F`$。** 始终有：

```math
XY=N.
```

先暂时省略相同的系数4、有效带宽，以及不同mesh轴可能有的带宽常数。混合forward中有两类竞争的通信：

```math
\text{activation项}=\frac{BD}{X},
\qquad
\text{weight项}=\frac{DF}{Y}.
```

如果一项远大于另一项，总时间会被大项控制。一个一阶平衡办法是让两项约相等：

```math
\frac{BD}{X}\approx\frac{DF}{Y}.
```

把$`Y=N/X`$代进去：

```math
\frac{BD}{X}\approx\frac{DF}{N/X}=\frac{DFX}{N}.
```

两边约去共同的$`D`$，再同时乘$`NX`$：

```math
BN\approx FX^2.
```

最后两边除以$`F`$：

```math
X^2\approx\frac{BN}{F},
\qquad
X_{opt}\approx\sqrt{\frac{BN}{F}}.
```

Scaling Book的更完整式还保留mesh带宽倍率$`M_X,M_Y`$：

```math
X_{opt}=\sqrt{\frac{B}{F}\frac{M_X}{M_Y}N}.
```

$`M_X,M_Y`$表示FSDP轴与MP轴分别能并用多少个硬件mesh方向来提供有效带宽。本节小学代数桥令两边带宽常数相同，也就是把$`M_X/M_Y`$暂按1处理，才得到上面的简式；真实placement必须把它放回来。

这里$`\sqrt{a}`$是“一个数乘自己得到$`a`$的那个非负数”；例如$`\sqrt{16}=4`$，因为$`4\times4=16`$。在$`D,F,N`$固定时，$`B`$增大4倍，$`X_{opt}`$只增大2倍。

把这个$`X_{opt}`$放回activation项：

```math
\frac{BD}{X_{opt}}
=\frac{BD}{\sqrt{BN/F}}
=D\sqrt{\frac{BF}{N}}.
```

所以最优通信时间随$`\sqrt{B}`$增长；而compute $`12BDF/N`$随$`B`$线性增长。于是：

```math
\frac{T_{math}}{T_{comms}}\propto\frac{B}{\sqrt{B}}=\sqrt{B}.
```

一个只用整数的验证：固定$`N=16,F=4,D=1`$，并为看比例暂取$`C=1`$ FLOP/s、$`W=1`$ byte/s。

先区分两种计时口径。上面的形状推导把两条mesh轴看成能重叠，因而实际瓶颈近似由两项中的较大者控制，即$`T_{comms}\approx\max(T_{FSDP},T_{MP})`$；下表为了让初学者逐项核账，故意把两个方向的逻辑payload串行相加。两种口径在最优点都令两项大致相等，所以会得到同一个$`X_{opt}`$和同一个$`\sqrt B`$缩放规律；表中绝对communication与ratio只属于“串行相加”的教学算例，不能当成真实重叠后的秒数。

| $`B`$ | $`X_{opt}=\sqrt{BN/F}`$ | $`Y=N/X`$ | $`BD/X`$ | $`DF/Y`$ | 总compute $`12BDF/N`$ | 表中总comms $`12(BD/X+DF/Y)`$ |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | $`\sqrt{4\times16/4}=4`$ | 4 | 1 | 1 | 12 | 24 |
| 16 | $`\sqrt{16\times16/4}=8`$ | 2 | 2 | 2 | 48 | 48 |

$`B`$从4增到16，是4倍；最佳$`X`$从4到8，是2倍；communication从24到48，是2倍；compute从12到48，是4倍；因此ratio从$`12/24=0.5`$到$`48/48=1`$，也是2倍。真实mesh只能选整数且还受轴带宽与拓扑约束，所以$`X_{opt}`$通常要四舍五入到可实现的mesh degree。

[Scaling Book 的 FSDP、TP 与组合章节](https://jax-ml.github.io/scaling-book/training/)支持上述公式来源和“FSDP约随$`B`$、MP近似不随$`B`$、优化混合约随$`\sqrt B`$”的曲线形状；它**不支持**把本课p56图上的约400当成普遍精确阈值。该阈值仍只属于课件所画硬件、带宽、模型宽度与mesh假设。

所以p56标出三个区域：

| $`B/N`$ | 图上判断 |
|---:|---|
| 小于约400 | 三种曲线都未越过1；没有一种方案compute-bound |
| 约400到850 | 只有优化后的FSDP+MP越过1 |
| 大于约850 | FSDP+MP与FSDP-only都越过1；MP-only仍低于1 |

例如$`B/N=300,600,1000`$分别落在这三个区域。400与850是该4×4×4 TPU mesh、模型宽度和带宽假设下的约数，不是GPU/TPU永恒阈值。视频66:16也把结论限定在不同communication topologies下维持compute utilization。

---

## 21. 3D/4D parallelism：degree 何时相乘，group 到底是谁和谁通信

### 21.1 Dense model 的最小乘法

**【课程内容｜PDF p56–59｜视频 [64:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3844s)】**

若 DP、TP、PP 是互相正交的三个 axes，dense model 的 GPU 总数是：

```math
N_{\text{GPU}}=d\times t\times p.
```

意思不是“一张卡同时变成三张”，而是每个 rank 有三维坐标：

```math
(\text{dp index},\text{pp stage},\text{tp lane}).
```

若再把 CP 作为独立正交 axis，可能写成：

```math
N_{\text{GPU}}=d\times t\times p\times c.
```

但 SP 往往复用 TP group，不再额外乘一次。是否独立必须看框架的 group layout，而不是看名字数量。

### 21.2 64 GPUs：$`TP=8,PP=4,DP=2`$

先验算：

```math
8\times4\times2=64.
```

假设每节点 8 GPUs，正好 8 nodes。采用教学上的连续 rank placement：

| DP replica | PP stage | TP group | 所在 node |
|---:|---:|---|---:|
| 0 | 0 | `{0,1,2,3,4,5,6,7}` | 0 |
| 0 | 1 | `{8,9,10,11,12,13,14,15}` | 1 |
| 0 | 2 | `{16,17,18,19,20,21,22,23}` | 2 |
| 0 | 3 | `{24,25,26,27,28,29,30,31}` | 3 |
| 1 | 0 | `{32,33,34,35,36,37,38,39}` | 4 |
| 1 | 1 | `{40,41,42,43,44,45,46,47}` | 5 |
| 1 | 2 | `{48,49,50,51,52,53,54,55}` | 6 |
| 1 | 3 | `{56,57,58,59,60,61,62,63}` | 7 |

三种 group：

- **TP group**：同一 replica、同一 stage 的 8 lanes；例如 `{0..7}`。它们共同计算同一层。
- **PP chain**：同一 replica、同一 TP lane 穿过 4 stages；例如 `{0,8,16,24}`。
- **DP group**：相同 stage、相同 TP lane 的两份 replicas；例如 `{0,32}`、`{1,33}`，一直到 `{31,63}`。

总共有：8 个 TP groups、16 条 PP chains、32 个二元 DP groups。真实 rank 编排可不同，但 group 关系必须等价。

### 21.3 为什么 TP 放节点内、PP 可以跨节点

视频 [64:45](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3885s) 用 compute/communication ratio 判断能否 hide communication；[66:38](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3998s) 给出课程经验：TP/EP 优先用 fast interconnect。

原因按频率分解：

- TP 在几乎每个 Transformer layer 都有 collective，消息密且阻塞关键路径；最好留在 NVLink/NVSwitch 域。
- PP 主要在 stage 边界传 activation/gradient；边界次数比逐层 TP 少，较能容忍跨节点网络。
- DP 的 gradient communication 可 bucket、overlap，也常跨节点。

这只是 placement heuristic（经验规则），不是数学定理。若硬件拓扑、消息大小或实现改变，profiler 结果可能推翻它。

### 21.4 Global batch 为什么只乘 DP，不乘 TP/PP

令：

- 每个 DP replica 每次送入 pipeline 的 microbatch size = 2；
- gradient accumulation steps = 8；也就是连续累积 8 个 microbatches 后再更新一次；
- DP degree $`d=2`$。

则：

```math
B_{\text{global}}
=B_{\text{micro}}\times A_{\text{grad}}\times d
=2\times8\times2
=32.
```

TP=8 的 8 张卡一起处理同一批 tokens；PP=4 的 4 stages 让同一批 tokens 依次经过整模型。因此 TP、PP 不产生新 samples，不能再乘进 global batch。

### 21.5 MoE 的 axes 为什么不能盲目全乘

**【课程内容｜PDF p57、p72｜视频 [66:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3965s)】【补充边界：[MoE Parallel Folding 论文](https://arxiv.org/abs/2504.14960)】**

MoE 可出现 DP、PP、attention TP、EP、ETP、EDP、CP。可是：

- EP ranks 可能同时充当 attention 的 DP/TP ranks；
- EDP 可能是 EP 之外剩余 ranks 的数据副本轴；
- ETP 只切 expert matrix，attention TP 可用另一 degree；
- SP 常绑定 attention TP；
- parallel folding 会让 attention 与 MoE layer 使用不同 group layouts。

**World size（全局进程数）** 是一次分布式job中全部ranks的数量。看到 `TP=2, EP=32, PP=8` 时，不能在不知道 DP/ETP/group reuse 时就断言world size为 $`2\times32\times8`$；先查完整配置和group construction。

### 21.6 3D/4D 的本质不是数名字

课程约 [66:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3970s) 把组合称为 3D/4D parallelism。真正的问题有三项：

1. 每个 axis 分了哪一种数据或状态？
2. 哪些 axes 正交相乘，哪些复用同一批 ranks？
3. 每个 group 在哪条物理链路通信？

只回答“我们用了 4D”而不回答这三项，无法算 memory、communication 或 global batch。

---

## 22. 从“先装得下”到“跑得快”：用 40B 模型走完整决策

### 22.1 第一步只算 static state 下界

**【课程内容｜PDF p57–62｜视频 [66:31](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3991s)】**

设：

- $`P=40`$ billion parameters = $`40\times10^9`$；
- 沿用 p28 的 12 bytes/parameter 训练状态口径；
- 每 GPU 标称 80 GB，十进制 $`1\ \text{GB}=10^9`$ bytes。

完整 static states：

```math
40\times10^9\times12
=480\times10^9\ \text{bytes}
=480\ \text{GB}.
```

单 GPU 只有 80 GB，因此单卡装不下。即使 $`480/80=6`$，也不能说 6 张卡一定够，因为还要留 activation、buffers 和 transient gathers。

### 22.2 四个候选方案逐个算

| 候选 | static memory/rank | 剩余于80GB | 第一判断 |
|---|---:|---:|---|
| ZeRO-1, $`d=8`$ | parameters+gradients $`=4P=160`$GB；optimizer $`=8P/8=40`$GB；合计200GB | -120GB | 装不下 |
| ZeRO-3/FSDP, $`d=8`$ | $`480/8=60`$GB | 20GB | 可能，但还要容纳 layer AG transient 与 activations |
| TP=8 | 若相关全状态均匀切分，$`480/8=60`$GB | 20GB | 可能；每层高速 collective 很多 |
| TP=4, PP=4 | 总 model-shard degree $`=16`$，理想 $`480/16=30`$GB | 50GB | static 余量较大；需要16 GPUs/replica且有 PP bubble |

这里“可能”不是部署保证。比如 FSDP 长期驻留 60GB，不代表峰值也是60GB；某层完整参数 all-gather、activation、communication buffer 都会抬高峰值。

### 22.3 为什么课程规则要先 fit，再 throughput

视频 [66:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4006s) 的课程规则是：先用 TP/EP、PP 或 ZeRO-3 让模型 fit；fit 后尽量把剩余 GPUs 用于 DP。其因果链是：

1. 不 fit，训练不能开始，throughput 为零。
2. fit 之后，减少不必要的 model parallel 常能保留更大的 matrix 和更少的关键路径通信。
3. DP 可增加独立 data work，但也会扩大 global batch；必须检查优化目标是否允许。

Megatron 的实践建议约 [67:26](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4046s) 说“minimize model parallelism, maximize data parallelism”。这是特定硬件与软件下的工程 heuristic，不是说所有模型都应把 DP 开到最大。

### 22.4 p59 的十行规模表：TP先到8，随后PP增大、DP下降

**【课程内容｜PDF p59，高分辨率逐格核验｜视频69:37–70:35】【一手来源：[Narayanan et al. 2021](https://arxiv.org/abs/2104.04473)】**

课件摘录表如下；右侧DP size是课件额外标注。$`1`$ TFLOP/s是$`10^{12}`$次浮点运算/秒，$`1`$ PFLOP/s是$`10^{15}`$次浮点运算/秒。`TFLOP/s/GPU`与`% peak`是该论文实验口径，不等同于所有后来硬件的MFU。

| Params B | Heads | Hidden | Layers | TP | PP | GPUs | Batch | TFLOP/s/GPU | % theoretical peak | Aggregate PFLOP/s | DP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7 | 24 | 2304 | 24 | 1 | 1 | 32 | 512 | 137 | 44% | 4.4 | 32 |
| 3.6 | 32 | 3072 | 30 | 2 | 1 | 64 | 512 | 138 | 44% | 8.8 | 32 |
| 7.5 | 32 | 4096 | 36 | 4 | 1 | 128 | 512 | 142 | 46% | 18.2 | 32 |
| 18.4 | 48 | 6144 | 40 | 8 | 1 | 256 | 1024 | 135 | 43% | 34.6 | 32 |
| 39.1 | 64 | 8192 | 48 | 8 | 2 | 512 | 1536 | 138 | 44% | 70.8 | 32 |
| 76.1 | 80 | 10240 | 60 | 8 | 4 | 1024 | 1792 | 140 | 45% | 143.8 | 32 |
| 145.6 | 96 | 12288 | 80 | 8 | 8 | 1536 | 2304 | 148 | 47% | 227.1 | 24 |
| 310.1 | 128 | 16384 | 96 | 8 | 16 | 1920 | 2160 | 155 | 50% | 297.4 | 15 |
| 529.6 | 128 | 20480 | 105 | 8 | 35 | 2520 | 2520 | 163 | 52% | 410.2 | 9 |
| 1008.0 | 160 | 25600 | 128 | 8 | 64 | 3072 | 3072 | 163 | 52% | 502.0 | 6 |

至少三行自己验GPU乘积：

**小模型1.7B：**

```math
DP\times TP\times PP=32\times1\times1=32\ \text{GPUs}.
```

**18.4B：** TP刚到8，PP仍为1：

```math
32\times8\times1=256\ \text{GPUs}.
```

**中大模型310.1B：**

```math
15\times8\times16=1920\ \text{GPUs}.
```

**最大1008B：**

```math
6\times8\times64=3072\ \text{GPUs}.
```

沿表从上往下读：TP是$`1\to2\to4\to8`$，之后停在8；PP再从1增至64；DP先保持32，随后降至24、15、9、6。视频70:13开始解释这个prescription，70:20说TP到8停止，70:26说PP继续增大，70:31指出最大规模时DP下降。

这十行是2021论文的选定模型/硬件实验，不是“TP必须等于8”的数学证明。

### 22.5 Recomputation 为什么“多算反而更快”有可能成立

**【课程内容｜PDF p62｜视频 [71:51](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4311s)】**

构造一个教学例：

- 不重算时，activation 限制 local batch=2，step time=1.00秒；throughput $`=2/1.00=2.0`$ samples/s。
- 开重算后，额外计算令同样 batch 的时间变为1.25秒，但省出的内存让 local batch=4；假设时间仍约1.25秒，则 throughput $`=4/1.25=3.2`$ samples/s。
- 相对提高：

```math
\frac{3.2-2.0}{2.0}=0.6=60\%.
```

但如果重算后 batch 仍只能是2：

```math
2/1.25=1.6\ \text{samples/s},
```

反而比2.0慢20%。所以“重算提高 throughput”的条件是：省下的 memory 能换来更好的 batch/parallel utilization，并覆盖新增 FLOPs。

### 22.6 证据驱动的选择流程

1. 用 memory snapshot 分清 static states、activations、buffers 谁是峰值主体。
2. 用最保守方案先 fit，并给临时峰值留余量。
3. 用 profiler 找 communication wait、pipeline bubble、small GEMM、straggler。
4. 若 TP collective 卡住，尝试减少 TP、增加 PP/DP；若 PP bubble 大，增加 microbatches 或重新平衡 stages。
5. 若 activation OOM，试 SP/CP/recomputation/FlashAttention，而不是只增加 ZeRO stage。
6. 每次改动后重新量 correctness、peak memory 与 tokens/s。

课件 p61 的 162.2B、64-GPU 实验中 $`PP=8,TP=8`$ 最好，只能说明该实验设置；不能推出任何模型的万能 $`8\times8`$。

---

## 23. 真实训练配置：把“课程快照”“原始报告”“推荐 recipe”分开

### 23.1 读配置表的四条纪律

**【课程内容｜PDF p63–72｜视频 [72:35](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4355s)】**

1. `??` 就是课件没有给出，不能靠乘法补成事实。
2. degree 正常是正整数；`PP=0` 多半表示“不用/未列”，不能拿0参与乘法。
3. 官方 model report、课程整理表、NVIDIA 推荐 benchmark recipe 是三种不同来源。
4. 同一模型不同阶段、sequence length、hardware、software version 可以用不同配置。

### 23.2 OLMo-7B：不是“Dolma-7B”

**【课程内容｜PDF p63｜视频 [72:46](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4366s)】**

- **课件/口述：** 讲者立刻自我纠正：模型是 OLMo，训练数据集是 Dolma；约7B模型使用 FSDP，具体 accelerator 数忘记了。
- **一手核对：**[OLMo 技术报告](https://arxiv.org/abs/2402.00838) 与 [Dolma 数据集论文](https://arxiv.org/abs/2402.00159) 支持“OLMo 是模型、Dolma 是语料”的边界。
- **不能推出：** TP/PP/EP/CP degree；p63 的 “probably fits intra-node” 是课程推测，不是报告中的保证。课件写 `FDSP`，视觉上是字母顺序笔误，应读作 FSDP。

### 23.3 DeepSeek LLM 与 DeepSeek-V3 不是一套配置

**【课程内容｜PDF p64、p72｜视频 [73:31](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4411s)】**

- **DeepSeek LLM/早期 dense 口径：**[DeepSeek LLM 报告](https://arxiv.org/abs/2401.02954) 明确列 DP、TP、SP、1F1B PP、ZeRO-1 与通信/计算 overlap，但未在这里给完整 degrees。
- **DeepSeek-V3：**[V3 技术报告](https://arxiv.org/abs/2412.19437) 支持 $`PP=16`$、$`EP=64`$（跨8个8-GPU nodes）与 ZeRO-1；模型的 MoE 通信与 pipeline 被专门 overlap。课件约 [73:54](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4434s) 强调64-way EP。
- **p72 边界：**`DP=??`、`CP=??` 保持未知；TP/SP=1 是该表口径。p72 第一行笼统写 “Deepseek, EP=8”，模型标签不够精确，不能与 dense DeepSeek LLM 行强行合并。

### 23.4 Yi 与 Yi-Lightning

**【课程内容｜PDF p65｜视频 [74:23](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4463s)】**

- [Yi 原始报告](https://arxiv.org/abs/2403.04652) 是6B/34B dense family；课件把经典训练组合概括为 ZeRO-1 + TP + PP，具体 degrees 未列。
- [Yi-Lightning 报告](https://arxiv.org/abs/2412.01253) 确认它是 MoE；课件说其以 EP 取代部分 TP。这是架构演进案例，不代表“MoE 永远不使用 TP”。
- p72 中 Yi 的 `DP=??, TP/SP>0, EP=1, PP>0, CP=??` 只表达方向，不是可相乘的完整配置。

### 23.5 Llama 3 405B：同一模型不同阶段改变 CP/DP

**【课程内容｜PDF p66｜视频 [75:06](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4506s)】【一手来源：[Llama 3 报告](https://arxiv.org/abs/2407.21783)】**

本节表格第一次使用 **MFU（Model FLOPs Utilization，模型浮点运算利用率）**：它是“按模型公式估算的有效 FLOP/s ÷ 硬件理论峰值 FLOP/s”，不是GPU忙碌时间百分比；§24.1会用小数字重算。

| 阶段快照 | GPUs | TP | CP | PP | DP | sequence | batch/DP | tokens/global batch | MFU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm-up | 8192 | 8 | 1 | 16 | 64 | 8192 | 32 | 16M | 43% |
| main pretrain | 16384 | 8 | 1 | 16 | 128 | 8192 | 16 | 16M | 41% |
| long-context | 16384 | 8 | 16 | 16 | 8 | 131072 | 16 | 16M | 38% |

检查 main pretrain：

```math
8\times1\times16\times128=16{,}384.
```

检查 long-context：

```math
8\times16\times16\times8=16{,}384.
```

sequence 从8192变131072：

```math
131{,}072/8192=16.
```

于是 CP 从1增至16、DP从128降至8，GPU总数仍相同。Llama 3 405B 是 dense model，表中没有 EP axis；p72 的 `EP=0` 表示不使用 EP，不应作为 degree 乘进公式。

### 23.6 Gemma 2：TPU 上 model sharding 随模型大小增加

**【课程内容｜PDF p68｜视频 [76:24](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4584s)】【一手来源：[Gemma 2 报告](https://arxiv.org/abs/2408.00118)】**

| 模型 | TPU chips | data replicas | model-shard degree |
|---|---:|---:|---:|
| 2B | 512 TPUv5e | 512 | 1 |
| 9B | 4096 TPUv4 | 1024 | 4 |
| 27B | 6144 TPUv5p | 768 | 8 |

逐行验算：

```math
512\times1=512,
```

```math
1024\times4=4096,
```

```math
768\times8=6144.
```

课件把 model sharding 概括为 TP+SP，并把 optimizer state sharding 类比 ZeRO-3。这里是 TPU mesh 语境；不能直接把 degree 映射成某个 GPU NCCL 配置。视频对“无需 pipeline”的评价带讲者判断，不是无限 scale-out 定律。

### 23.7 Mixtral 8x22B：这是 NVIDIA recipe，不是原始训练披露

**【课程内容｜PDF p69｜视频 [77:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4624s)】**

课件摘录的 NVIDIA Megatron 配置：

```math
TP=4,\quad PP=4,\quad CP=1,\quad EP=8,\quad \text{GPUs}=256.
```

若这些 axes 全独立，已知部分乘积：

```math
4\times4\times1\times8=128.
```

为了到256，课件推测 DP 可能是：

```math
256/128=2.
```

但 p69 用了 “Likely has DP of 2”。因此 `DP=2` 必须标为课程推断，不能写成 Mistral 原始训练事实。

还要再缩小证据边界：$`TP=4,PP=4,CP=1,EP=8,256`$ GPUs 这组精确数字只由**课件p69截图**支持。当前 [Megatron-LM Mixtral README](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mixtral/README.md) 的明确示例是Mixtral 8x7B的$`TP=1,EP=8,PP=4`$，并只说8x22B也可适配；它不能反向验证课件的8x22B精确配置。该链接只证明框架支持这些axes，不证明这行recipe数字。

### 23.8 Nemotron 3 Super：`PP=0` 不是零张 GPU

**【课程内容｜PDF p70、p72｜视频 [77:54](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4674s)】【一手来源：[NVIDIA Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)】**

- 课件模型名：120B total、12B active，长 context 1,048,576。
- 页面列 `TP/PP/CP/EP=(2/0/64/64)`；其中 degree 不可能为0，所以 `PP=0` 只能解释为“不使用 PP”或字段记法异常，不能相乘。
- p72 更谨慎地写 `PP=??, DP=??`。本笔记采用这个边界，不把未知补成1。
- NVIDIA 官方页面确认120B-A12B和最高1M context；具体课件所示 training degrees 属于课程时点快照，不由模型卡自动保证。

### 23.9 Qwen3：课件标题有225/235冲突，recipe 也随硬件变

**【课程内容｜PDF p71–72｜视频 [78:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4684s)】**

- p71 标题写 “225B-A22B”，表内和官方模型名都是 **Qwen3-235B-A22B**；按视觉证据，标题225应视为课件笔误。
- 课件表：30B-A3B 用 $`TP=1,PP=1,EP=8`$、8 GPUs；235B-A22B 的一个 NVIDIA recipe 用 $`TP=2,PP=8,EP=32`$、512 GPUs。
- 对后一行，已知乘积：

```math
2\times8\times32=512.
```

这恰好用尽512；但这仍是课件引用的 Megatron recipe，不代表 Qwen 团队原始预训练唯一配置。[Qwen3 技术报告](https://arxiv.org/abs/2505.09388) 确认235B-A22B型号；[Megatron Bridge Qwen3 文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/qwen/qwen3-moe.md) 显示不同任务/硬件可使用不同 recipes。

### 23.10 p72 汇总表的安全读法

| 模型/配置快照 | DP | TP/SP | EP | PP | CP | 能否直接相乘 |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek（标签含糊） | `??` ZeRO-1 | 1 | 8 | 16 | `??` | 不能 |
| DeepSeek-V3 | `??` ZeRO-1 | 1 | 64 | 16 | `??` | 不能 |
| Yi | `??` ZeRO-1 | `>0` | 1 | `>0` | `??` | 不能 |
| Llama3 405B main | 128 | 8 | 未使用 | 16 | 1 | 可以，$`16384`$ |
| Gemma2 27B | 768 | 8 | 未使用 | 未使用 | 未使用 | TPU表中可算 $`6144`$ |
| Mixtral8x22B NVIDIA recipe | 课程推测2 | 4 | 8 | 4 | 1 | 若接受推测则256 |
| Nemotron3 long-context | `??` | 2 | 64 | `??` | 64 | 不能 |
| Qwen3 Megatron recipe | `??` | 2 | 32 | 8 | 1 | 表中512已由三轴相乘，但不能泛化 |

从这几个有限样本可以观察：TP 常不超过8、长 context 会增大 CP、MoE 的 EP 可很大。视频 [79:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4744s) 也这样总结。但这是2024–2026材料的少量系统/模型快照，不是永恒定律。

---

## 24. 大规模训练不只要快：MFU、失败、checkpoint 与 straggler

### 24.1 Linear scaling 与 MFU 各是什么意思

**【课程内容｜PDF p59–61｜视频 [70:47](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4247s)】**

**Linear compute scaling（线性计算扩展）** 的理想是：GPU 数翻倍，单位时间完成的总计算近似翻倍。现实会被 communication、bubble、load imbalance 和 failures 拉低。

**MFU（Model FLOPs Utilization，模型浮点运算利用率）** 常写成：

```math
MFU=
\frac{\text{按模型公式估算的有效 FLOP/s}}
{\text{硬件理论峰值 FLOP/s}}.
```

例如单卡理论峰值1000 TFLOP/s，模型有效吞吐对应400 TFLOP/s：

```math
MFU=400/1000=0.4=40\%.
```

MFU不是“GPU有40%的时间亮着”，也不包含所有非模型辅助工作；不同论文 FLOPs 口径可能不同，比较前要核定义。

### 24.2 p60–61 的图怎样读

- p60 横向增加 GPUs，图中的 PTD-P（pipeline+tensor+data parallel）保持较平的 per-GPU throughput；这是特定模型、batch、硬件的 scale 实验。
- p61 固定162.2B模型和64 GPUs，比较多组 $`(PP,TP)`$；$`PP=8,TP=8`$ 在该图最好。
- p61 文字写“64 machines”，但图标题/横轴说明是64 GPUs；本笔记保留这个课件内部不一致，以图的实验口径为准。

不能把一张图改写成“所有训练都能线性扩展”或“8×8永远最佳”。

### 24.3 Llama 3 的失败表说明了什么

**【课程内容｜PDF p67｜视频 [76:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4562s)】**

课件引用的54天窗口中，failure/event 分类原样写着：Faulty GPU 148次（30.1%）、HBM3 72次（17.2%）、software 54次（12.9%）、network 35次（8.4%），以及 host maintenance、SRAM、GPU processing、NIC、NCCL watchdog、silent corruption、thermal 等。

**这张课件表内部至少有一个字段错误，不能让读者把148与30.1%当成可同时复算：**

- 用第二行反推总数：$`72/0.172\approx418.6`$，也就是约419；
- 若用419作分母：

  $`148/419\approx0.3532=35.32\%,`$

  不是30.1%；
- 本地p67可见18行counts逐项相加为419；而且其余代表行基本与这个分母一致：$`72/419\approx17.18\%`$、$`54/419\approx12.89\%`$、$`35/419\approx8.35\%`$，分别对应课件四舍五入后的17.2%、12.9%、8.4%。

因此在这张可见表内，**最可能是Faulty GPU行的30.1%单元格有误**；但本笔记没有底层event数据，不擅自把源课件改成某个新百分比，仍原样记录148与30.1%的冲突。

它证明大规模训练必须设计恢复流程；但表是观察到的 event counts，不提供总 GPU-days 暴露量，不能直接把148除以54当“每GPU每天故障率”。

### 24.4 一个独立故障假设的手算

**【补充例子，不是 Llama 3 报告数据】**

假设每张 GPU 每天失败概率：

```math
q=0.1\%=0.001.
```

单卡一天不失败概率：

```math
1-q=0.999.
```

若1024张卡彼此独立，全部不失败概率：

```math
0.999^{1024}\approx0.359.
```

至少一张失败概率：

```math
1-0.999^{1024}
\approx1-0.359
=0.641
=64.1\%.
```

期望失败张数是：

```math
1024\times0.001=1.024\ \text{failures/day}.
```

64.1%与1.024不是矛盾：前者问“有没有至少一次”，后者问“平均共有几次”。真实 failures 可能相关，例如机架电源或网络故障同时影响许多 GPUs，独立假设会失真。

### 24.5 四个系统词必须分清

- **Checkpoint（训练检查点）**：定期把 model、optimizer、step、随机数等恢复所需状态写入持久存储。
- **Restart（重启）**：失败后重新建立 jobs/process groups，从最近可用 checkpoint 继续。
- **Fault tolerance（容错）**：检测、隔离、恢复并限制损失的整体能力，不只是“有个文件”。
- **Straggler（掉队者）**：同一同步 step 中异常慢的 rank；collective 常需等它，其他 ranks 即使健康也会空等。

若每30分钟 checkpoint 一次、失败在两个 checkpoint 之间均匀发生，平均丢失计算约：

```math
30/2=15\ \text{minutes}.
```

但 checkpoint 自身若每次停2分钟，每小时做2次，则纯写盘停顿：

```math
2\times2=4\ \text{minutes/hour},
```

占：

```math
4/60=6.67\%.
```

因此间隔太长会多丢工作，太短会多付保存开销；异步 checkpoint、增量保存和冗余存储都是对这个权衡的工程改进。

### 24.6 课程末尾真正想留下的结论

视频 [79:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4760s) 总结：规模继续增长时，必须同时考虑 multi-GPU、multi-node，甚至 multi-datacenter；[79:37](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4777s) 又提醒链路有快有慢、方法消耗的资源不同。

因此“成功的大规模训练配置”至少要同时满足：

1. static states 与 activation 峰值放得下；
2. 关键通信能被高速拓扑或计算 overlap 承受；
3. matrix/token work 足够大，设备不是切得只剩小碎片；
4. global batch 与优化目标匹配；
5. pipeline/MoE/CP 负载不过分失衡；
6. 失败后能检测、保存、重启并验证状态。

---

## 25. 决策树与 45 个常见误区

### 25.1 一棵可以实际执行的决策树

1. **训练是否 OOM？** 是：先用 memory snapshot 找 static/activation/temporary 的最大项；否：跳到第4步。
2. **static state 最大？** 是：比较 ZeRO/FSDP、TP/EP、PP；逐项算长期与临时峰值。不是：看 activation。
3. **activation 最大？** 比较 SP、CP、recomputation、FlashAttention、microbatch；再量新增 FLOPs/通信。
4. **throughput 低在哪里？** profiler 分成 compute、collective wait、P2P wait、bubble、input、checkpoint、straggler。
5. **collective 慢？** 查 bytes、消息次数、group placement、链路、能否 overlap；不要只看标称 bandwidth。
6. **compute 慢？** 查 matrix shape 是否被 TP/EP 切得过小、kernel utilization、recompute work。
7. **PP 慢？** 查 stage imbalance、microbatch 数、bubble denominator 与 schedule。
8. **MoE 慢？** 查 tokens/expert、A2A bytes、load balance、attention 与 expert groups 是否应解耦。
9. **长 context 慢/OOM？** 查 attention algorithm、CP degree、causal imbalance 与 KV transport。
10. 每次只改少量变量，重新验证 loss、峰值、tokens/s 和 failure recovery。

### 25.2 错误说法、为什么错、正确说法

| # | 错误说法 | 为什么错 | 正确说法/反例 |
|---:|---|---|---|
| 1 | 多GPU一定线性加速 | 通信、bubble、straggler会增加 | 线性扩展是理想上界，要用吞吐实测 |
| 2 | DDP把参数显存除以GPU数 | DDP每rank复制完整模型 | DDP切batch；ZeRO-3/FSDP才长期切parameters |
| 3 | DP degree越大越好 | local batch固定时global batch随之增大 | 同时检查优化所允许的global batch |
| 4 | SUM和AVG gradients永远等价 | 学习率/损失归一化未调整时尺度不同 | 先固定loss是local sum还是local mean |
| 5 | 不等local batch也直接平均rank means | 每个rank权重应按样本数 | 用local sum和global count，或加权平均 |
| 6 | 16 bytes/param包含全部训练显存 | 它不含activation/buffer/allocator | 它只是指定mixed-precision state口径 |
| 7 | p28也使用16 bytes/param | p28改用12-byte教学口径 | 必须跟随该页的BF16 moments/Kahan假设 |
| 8 | 80GB卡可安全放80GB static state | 没给activation和temporary留空间 | static必须明显低于容量并实测峰值 |
| 9 | ZeRO-1把parameter也切了 | 它只切optimizer states | parameter和gradient仍复制 |
| 10 | ZeRO-2把所有model state都切了 | parameter仍复制 | ZeRO-2切gradient+optimizer |
| 11 | ZeRO-3通信仍只有DDP的2P | backward前也要再次取参数 | 课程简化约3P；实现/重用会改变常数 |
| 12 | FSDP使activation自动除以DP | 它切model states，不切local activation | 另用SP/CP/recompute等 |
| 13 | FSDP和ZeRO-3在所有实现细节相同 | wrapping、prefetch、reshard策略不同 | “约等于”只是本讲算法层口径 |
| 14 | PP degree会乘global batch | stages处理同一批samples | global batch只乘独立data replicas |
| 15 | microbatch就是global batch | microbatch只是一次pipeline chunk | global=batch micro×accum×DP |
| 16 | bubble ratio只有一个定义 | 可除useful或除total | 写出 $`(p-1)/m`$ 还是 $`(p-1)/(m+p-1)`$ |
| 17 | microbatches越多总是更好 | 太小会降低GEMM效率并增加launch/通信 | 在bubble与kernel效率之间量测 |
| 18 | 1F1B等于零bubble | 它主要降低activation residency | warmup/drain仍可能存在 |
| 19 | $`dW`$可无限延后 | 仍依赖保存的$`X,dY`$和更新边界 | 只能在依赖与buffer寿命允许时调度 |
| 20 | TP只在层首通信一次 | row/column切分通常每层有collectives | 所以常留在节点内高速域 |
| 21 | TP使所有activation严格除以$`t`$ | pointwise/residual项可能复制 | TP+SP在论文假设下才整体线性分片 |
| 22 | TP越大越好 | local matrix太小会降利用率 | TP degree取决于fit、shape与网络 |
| 23 | SP是一个总要额外相乘的独立axis | 常与TP共享group | 看group construction，不数缩写 |
| 24 | EP把整个Transformer参数除以$`e`$ | 只切routed experts | attention、router、shared expert可能复制或另切 |
| 25 | EP不需要TP | attention和expert matrix需求不同 | parallel folding可让attention TP与ETP不同 |
| 26 | EP degree不影响计算效率 | degree大可让每expert tokens太少 | 小GEMM和load imbalance都会伤效率 |
| 27 | all-to-all流量均匀 | router可能把多数tokens送同一rank | 要量每rank split和慢尾 |
| 28 | CP就是SP的另一个名字 | CP让attention交换KV；SP主要切pointwise activation | 二者目标、通信和group不同 |
| 29 | Ring Attention每块softmax后平均即可 | 每块归一化分母不同 | 维护全局running max/sum/numerator |
| 30 | CP把整卡显存除以$`c`$ | parameters和buffers不随CP切 | 只说sequence-side activation/KV近似$`1/c`$ |
| 31 | causal CP每个round工作完全相同 | 有些KV块对local Q全是未来 | placement/schedule需处理三角不均衡 |
| 32 | FlashAttention让activation memory为零 | 它避免物化大attention matrix，不消除所有saved tensors | 仍有输入、统计量和其他layer activation |
| 33 | recomputation一定加速 | 它增加FLOPs | 只有省内存换来的batch/utilization收益足够才可能更快 |
| 34 | 3D parallelism就是把三个degree盲乘 | axes可能复用group | dense正交轴可乘，MoE folding必须查布局 |
| 35 | PP=0表示模型使用0张GPU | degree不能为0 | 课件Nemotron字段应视为“不用/未知”的异常记法 |
| 36 | EP=0也应乘进world size | 0常表示未使用该axis | 乘法只包含实际启用且degree≥1的正交轴 |
| 37 | `??`可由相邻模型配置补出 | 不同阶段/硬件会变 | 未知就保持未知 |
| 38 | OLMo和Dolma是同一个模型名 | Dolma是数据集 | 模型是OLMo，使用Dolma语料 |
| 39 | Mixtral p69 的DP=2是官方训练事实 | 课件明确用了推测口吻 | 它是从NVIDIA 256-GPU recipe反推的课程推断 |
| 40 | Qwen3是225B-A22B | p71标题与表/官方名冲突 | 官方/表中是235B-A22B，225是课件笔误 |
| 41 | Llama3所有阶段都用DP128 | 长context阶段改为CP16、DP8 | 配置随sequence与阶段改变 |
| 42 | Llama3 405B是MoE，所以该用EP | 它是dense Transformer | 课件表使用TP/PP/DP/CP，没有EP |
| 43 | MFU=40%表示GPU只工作40%的时间 | MFU是模型有效FLOPs与理论峰值之比 | 非模型工作和口径差异都可能存在 |
| 44 | 148次GPU故障可直接变成每GPU日概率 | 缺少GPU-days分母且事件可能相关 | 只能说明故障恢复很重要 |
| 45 | checkpoint越频繁越好 | 保存本身消耗时间/带宽 | 平衡重算损失、写盘开销和恢复目标 |

---

## 26. 自测题：先遮住答案，至少做完中间步骤

标有“手算/填表”的题共有60道。不要只写最后一个数。

1. **[手算]** $`B=8,M=4`$，样本梯度为 $`1,2,3,4,5,6,7,8`$；每rank连续拿2个。列每rank local mean，并算四个mean的AVG。
2. **[手算]** 两rank分别有1个和3个样本，local gradient sums为2和12。global mean是多少？直接平均两个local means为什么错？
3. **[手算]** 1B parameters按16 bytes/param需多少十进制GB？五项分别多少？
4. **[手算]** 把16GB十进制bytes换成GiB，保留两位小数。
5. **[手算]** p28的12-byte口径中，8张80GB卡baseline最多多少B parameters？
6. **[手算]** p28 ZeRO-1每param每rank为何是5 bytes？最多多少B parameters？
7. **[手算]** p28 ZeRO-2每param每rank为何是3.25 bytes？最多多少B parameters？
8. **[手算]** p28 ZeRO-3每param每rank为何是1.5 bytes？最多多少B parameters？
9. **[手算]** 4个parameters在2 ranks做ZeRO-1，写出optimizer state ownership和更新后为什么要AG parameters。
10. **[手算]** 两rank各有gradient向量 $`[1,2,3,4]`$ 与 $`[10,20,30,40]`$。SUM reduce-scatter每rank各拿连续2格，输出是什么？
11. **[手算]** 全模型参数通信量$`P=8`$GB，DDP课程bandwidth模型约$`2P`$，是多少GB？
12. **[手算]** 同一$`P=8`$GB，ZeRO-3课程简化约$`3P`$，是多少GB？比DDP多多少？
13. **[手算]** 40B parameters、12 bytes/param，完整static state多少GB？
14. **[手算]** 上题ZeRO-1、$`d=8`$，parameters+gradients与optimizer各多少GB/rank？合计？
15. **[手算]** 上题ZeRO-3、$`d=8`$，理想长期static多少GB/rank？80GB卡还剩多少GB？
16. **[手算]** $`b=2,s=1024,h=4096,a=32`$，先算$`sbh`$与$`as/h`$。
17. **[手算]** 用 $`M=sbh(34+5as/h)`$ 算baseline bytes与MiB/层。
18. **[手算]** $`t=8`$，用TP-only公式算括号系数和MiB/层。
19. **[手算]** 同样参数，用TP+SP公式算MiB/层。
20. **[手算]** $`s`$从1024增至2048，其余固定，quadratic项$`5abs^2`$变几倍？
21. **[手算]** 48 layers都按592MiB/层粗相加是多少MiB和GiB？为什么不是训练峰值保证？
22. **[手算]** 只看forward（forward-only），$`p=4`$、不切microbatch，一共有几个time slots？stage利用率多少？
23. **[手算]** forward-only，$`p=4,m=8`$，总slots、利用率、bubble/useful、bubble/total各是多少？
24. **[手算]** microbatch activation shape $`[32,1024]`$ FP32，stage边界一次传多少bytes/KiB？4个microbatches共多少KiB？
25. **[手算]** $`p=2,m=4`$ forward-only，画每时刻两stage的F表并算利用率。
26. **[手算]** microbatch=2、gradient accumulation=8、DP=2，global batch是多少？为什么不乘TP8和PP4？
27. **[手算]** $`x=[1,2]`$，$`W_1=\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}`$。两rank按列各取2列，算local hidden并拼回。
28. **[手算]** 若ReLU不改上题正数，**这里新给一个矩阵** $`W_2=\begin{bmatrix}1&0\\0&1\\1&1\\2&1\end{bmatrix}`$；不要沿用正文另一例的$`W_2`$。按行对应切分，算两rank partial outputs和all-reduce结果。
29. **[手算]** $`h=8,ffn=16,t=2`$，普通FFN两矩阵$`W_1[8,16],W_2[16,8]`$总参数多少？每rank均匀切多少？
30. **[手算]** 课件TP通信元素公式 $`8bsh(p-1)/p`$，代$`b=2,s=4,h=8,p=2`$算元素和BF16 bytes。
31. **[手算]** PP每boundary/microbatch传$`bsh`$元素；代同样$`b,s,h`$算bfloat16 bytes。先做裸除$`512/128`$，再解释为什么不能据此说“TP一定比PP贵4倍”。
32. **[手算]** $`[b,s,h]=[1,4,2],t=2`$ 做SP，列rank0/1各持哪些token和local shape；AG后shape是什么？
33. **[手算]** 8 tokens各hidden size4、BF16，EP dispatch若每token跨网发一次，单向payload共多少bytes？返回同样大小时双向多少？
34. **[手算]** 8 tokens路由到4 experts的counts为$`[5,1,1,1]`$。平均每expert多少？最忙/平均是多少倍？
35. **[手算]** 4 causal tokens分2 CP ranks，列$`Q_0,Q_1,Q_2,Q_3`$各能看哪些keys。
36. **[手算]** sequence-side activation原为800MiB，CP=4，理想persistent shard多少MiB？为什么整卡不一定只剩1/4？
37. **[手算]** 每GPU每日失败率0.1%，1024独立GPU至少一故障概率约多少？期望故障数多少？
38. **[手算]** 每30分钟checkpoint、失败时刻均匀，平均丢失多少分钟？若每次停2分钟，每小时开销比例多少？
39. **[手算]** 有效400 TFLOP/s、理论1000 TFLOP/s，MFU多少？
40. **[手算]** $`TP=8,PP=4,DP=2`$ 共多少GPU？若每节点8GPU，共多少nodes？
41. **[手算]** 上题有多少TP groups、每组多大？多少PP chains、每条多长？多少DP groups、每组多大？
42. **[手算]** 按§21连续rank布局，rank5的TP group、PP chain、DP group分别是什么？
43. **[手算]** rank44属于哪个DP replica、PP stage、TP lane？列其三个groups。
44. **[手算]** Llama3 main配置$`TP8,CP1,PP16,DP128`$，乘积是多少？
45. **[手算]** Llama3 long-context配置$`TP8,CP16,PP16,DP8`$，乘积是多少？sequence增大几倍？DP减小几倍？
46. **[手算]** Gemma2 9B：data replicas1024、model shard4，共多少chips？27B的768×8呢？
47. **[手算]** Mixtral recipe已知$`TP4,PP4,CP1,EP8`$，乘积多少？若world=256，课程推测DP多少？
48. **[手算]** Qwen3 recipe$`TP2,PP8,EP32`$乘积多少？这能否证明原始训练配置？
49. **[手算]** Recomputation例：batch2/1.0s与batch4/1.25s各多少samples/s？提升百分比？
50. **[手算]** 若重算后仍batch2、time1.25s，吞吐和下降比例是多少？
51. **[手算]** static state 480GB，TP4×PP4均匀分片后每rank多少GB？80GB卡理论余量？
52. **[手算]** 若8-GPU FSDP长期state60GB，某时刻额外AG layer 6GB、activation12GB、buffers5GB，峰值账是多少？80GB还剩多少？
53. **[手算]** 一个step有100ms compute与30ms communication；完全不能overlap是多少ms？完全隐藏communication是多少ms？
54. **[手算]** PP四stages每段时间$`[1,1,2,1]`$ms。一个microbatch穿越四段的latency是多少？稳态cadence由谁决定、是多少？若四段都为1ms，这两个数分别是多少？
55. **[手算]** 两个MoE ranks收到tokens数$`[6,2]`$，每token expert compute 3ms且串行教学模型，两个rank各多久、同步阶段多久、空等多久？
56. ZeRO-1、2、3分别长期切哪些model states？
57. 为什么FSDP不能直接解决长序列activation OOM？给两个候选方法。
58. PP、TP、SP、EP、CP分别切哪一个轴/对象？
59. 为什么TP通常优先放节点内，PP较常跨节点？
60. 为什么SP通常不额外乘进world size？
61. 为什么MoE的TP、EP、ETP、EDP不能看到名字就全乘？
62. Ring Attention为何不能先做每块softmax再平均？
63. `PP=0`和`EP=0`在课件表中应怎样读？
64. `DP=??`应怎样处理？
65. OLMo与Dolma分别是什么？p63还有哪个拼写问题？
66. DeepSeek LLM与DeepSeek-V3各有哪些课程/官方可确认的并行方法？
67. Yi与Yi-Lightning为什么不能写成同一个dense配置？
68. Llama3 long-context阶段为什么提高CP、降低DP？
69. Gemma2的TPU model sharding为何不能直接当GPU NCCL recipe？
70. Mixtral的DP=2为何要标“课程推测”？
71. Nemotron `TP/PP/CP/EP=(2/0/64/64)`为何不能直接相乘？
72. Qwen3页面的225B/235B冲突怎样处理？
73. MFU与“GPU忙碌时间比例”有什么区别？
74. 为什么failure table不能直接给出每GPU日故障率？
75. 用六句话复述本讲从fit到reliable throughput的主线。
76. **[手算]** 先说明p56中forward $`4BDF`$、backward $`8BDF`$及FSDP $`4DF+8DF`$的来源；再用$`B=8,D=2,F=4,X=2`$计算FLOPs与communication bytes。若$`C=24`$ FLOP/s、$`W=12`$ bytes/s，再算$`R=T_{math}/T_{comms}`$。
77. **[手算]** 再取$`Y=2,N=XY=4`$，用p56混合FSDP+MP公式计算FLOPs和bytes。
78. **[手算/读图]** p56中$`B/N=300,600,1000`$分别落在哪个region？哪些曲线超过$`R=1`$？
79. **[手算]** scores$`=[0,0,0]`$、values$`=[0,0,3]`$，前两项和后一项分块。算global softmax output与“两个local outputs等权平均”的错误output。
80. **[手算/查错]** p67写Faulty GPU 148次占30.1%，且18行counts合计419。算$`148/419`$；再用419检查72、54、35三行的百分比。哪一格最可能有误？为什么仍不擅自修改源课件？

---

## 27. 自测完整答案

### 27.1 第1–15题：DP、bytes 与 ZeRO

1. Rank0：$`(1+2)/2=1.5`$；rank1：$`(3+4)/2=3.5`$；rank2：$`(5+6)/2=5.5`$；rank3：$`(7+8)/2=7.5`$。四个local means的AVG：
   $`(1.5+3.5+5.5+7.5)/4=18/4=4.5.`$
   因为每rank样本数都等于2，这也等于8个样本的global mean。
2. 第一个rank local mean $`=2/1=2`$；第二个rank local mean $`=12/3=4`$。直接平均得 $`(2+4)/2=3`$，错误地给两个rank相同权重。global mean应为：
   $`(2+12)/(1+3)=14/4=3.5.`$
3. 1B即$`10^9`$。BF16 parameter $`2\times10^9=2`$GB；BF16 gradient 2GB；FP32 master 4GB；Adam $`m`$ 4GB；Adam $`v`$ 4GB。合计：
   $`2+2+4+4+4=16\ \text{GB}.`$
4. $`16`$GB十进制是$`16{,}000{,}000{,}000`$ bytes。除以$`2^{30}=1{,}073{,}741{,}824`$：
   $`16{,}000{,}000{,}000/1{,}073{,}741{,}824\approx14.90\ \text{GiB}.`$
5. Baseline在每rank复制全部12 bytes/param；8张卡不能把同一副本容量相加。每卡约：
   $`80/12=6.666\ldots\ \text{B parameters}.`$
   即约6.667B；还未给activation留余量。
6. p28 ZeRO-1：parameter+gradient复制，$`2+2=4`$ bytes；master+$`m+v`$共$`4+2+2=8`$ bytes，由8 ranks切成$`8/8=1`$。每rank $`4+1=5`$ bytes/param，因此：
   $`80/5=16\ \text{B parameters}.`$
7. ZeRO-2长期复制parameter 2 bytes；其余gradient+master+$`m+v`$为$`2+4+2+2=10`$ bytes，由8切：$`10/8=1.25`$。合计$`2+1.25=3.25`$，所以：
   $`80/3.25\approx24.615\ \text{B parameters}.`$
8. ZeRO-3把12 bytes全切8份：$`12/8=1.5`$ bytes/param/rank。容量：
   $`80/1.5=53.333\ldots\ \text{B parameters}.`$
9. 一种连续ownership：rank0拥有parameters 0、1对应的master/$`m`$/$`v`$；rank1拥有2、3的states。两rank仍需完整parameters做forward。gradient RS后，rank0更新0、1，rank1更新2、3；此时每rank只知道自己更新的两格，所以AG四格updated parameters，才能开始下一次完整forward。
10. 先逐元素SUM：
    $`[1,2,3,4]+[10,20,30,40]=[11,22,33,44].`$
    连续切2格：rank0得到$`[11,22]`$，rank1得到$`[33,44]`$。
11. $`2P=2\times8=16\ \text{GB}.`$
    这是课程bandwidth口径的逻辑量，不指定ring/tree物理步骤。
12. $`3P=3\times8=24\ \text{GB}.`$
    比DDP的16GB多$`24-16=8`$GB，也就是多$`P`$。
13. $`40\times10^9\times12=480\times10^9\ \text{bytes}=480\ \text{GB}.`$
14. ZeRO-1：parameters+gradients每param4 bytes：$`40\times4=160`$GB。optimizer部分每param8 bytes，被8切：$`40\times8/8=40`$GB。合计：
    $`160+40=200\ \text{GB/rank},`$
    所以80GB卡装不下。
15. ZeRO-3理想长期分片：
    $`480/8=60\ \text{GB/rank}.`$
    标称余量$`80-60=20`$GB；还要支付activation、AG transient与buffers。

### 27.2 第16–38题：activation、pipeline、TP、EP、CP

16. $`sbh=1024\times2\times4096=8{,}388{,}608.`$
    $`as/h=(32\times1024)/4096=32{,}768/4096=8.`$
17. 括号$`=34+5\times8=74`$。bytes：
    $`8{,}388{,}608\times74=620{,}756{,}992.`$
    换MiB：
    $`620{,}756{,}992/1{,}048{,}576=592\ \text{MiB/layer}.`$
18. TP-only括号：
    $`10+24/8+5\times8/8=10+3+5=18.`$
    bytes$`=8{,}388{,}608\times18=150{,}994{,}944`$；除$`1{,}048{,}576`$得$`144`$MiB/层。
19. TP+SP括号：
    $`34/8+(5\times8)/8=4.25+5=9.25.`$
    bytes$`=8{,}388{,}608\times9.25=77{,}594{,}624`$；换算得$`74`$MiB/层。
20. 二次项正比$`s^2`$。倍数：
    $`2048^2/1024^2=(2048/1024)^2=2^2=4.`$
21. $`592\times48=28{,}416\ \text{MiB}.`$
    $`28{,}416/1024=27.75\ \text{GiB}.`$
    它只是把每层保存项相加；实际峰值还依赖schedule、recompute、哪些layers同时驻留、buffers和allocator。
22. $`m=1,p=4`$时forward穿过4 stages，要4 slots。共有$`4\times4=16`$个stage-slots，只有4个有用：
    $`4/16=1/4=25\%.`$
23. 总slots：$`m+p-1=8+4-1=11`$。利用率：$`8/11=72.73\%`$。bubble/useful：$`(p-1)/m=3/8=37.5\%`$。bubble/total：$`3/11=27.27\%`$。后三个数的分母不同。
24. 元素数$`32\times1024=32{,}768`$。FP32每元素4 bytes：
    $`32{,}768\times4=131{,}072\ \text{bytes}=128\ \text{KiB}.`$
    4个microbatches：$`128\times4=512`$KiB。
25. 表：t1=`S0:F0,S1:-`；t2=`S0:F1,S1:F0`；t3=`S0:F2,S1:F1`；t4=`S0:F3,S1:F2`；t5=`S0:-,S1:F3`。有用格$`2\times4=8`$，总格$`2\times5=10`$：$`8/10=80\%`$。
26. $`B_{global}=2\times8\times2=32.`$
    TP8共同算同一microbatch；PP4让同一microbatch走4段，都没有创造新样本，所以不乘。
27. Rank0两列：$`[1+2\times5,\ 1\times2+2\times6]=[11,14]`$。Rank1两列：$`[1\times3+2\times7,\ 1\times4+2\times8]=[17,20]`$。拼回$`[11,14,17,20]`$。
28. Rank0使用$`W_2`$前两行：
    $`[11,14]\begin{bmatrix}1&0\\0&1\end{bmatrix}=[11,14].`$
    Rank1：
    $`[17,20]\begin{bmatrix}1&1\\2&1\end{bmatrix}=[17+40,17+20]=[57,37].`$
    all-reduce SUM：$`[11,14]+[57,37]=[68,51]`$。
29. 总参数：
    $`8\times16+16\times8=128+128=256.`$
    两rank均匀切：$`256/2=128`$ parameters/rank。
30. 元素数：
    $`8\times2\times4\times8\times(2-1)/2=512/2=256.`$
    BF16 bytes：$`256\times2=512`$ bytes。
31. PP单边界：$`bsh=2\times4\times8=64`$元素；bfloat16为$`64\times2=128`$ bytes。裸除确实是：
    $`512/128=4.`$
    但不能据此说TP一定比PP贵4倍，因为两数口径未统一：（1）TP式可能是per-rank ring-equivalent traffic，PP是单boundary payload；（2）TP式按per-layer，PP按per-boundary/per-microbatch；（3）TP式含课程forward+backward计数，PP的$`bsh`$页未明确是否含backward；（4）ring send bytes、endpoint send+receive与逻辑payload不是同一traffic metric。必须先固定总layers、boundaries、microbatches、forward/backward directions和traffic定义，才能比较总通信。
32. Rank0持token0、1，rank1持token2、3；每rank local shape $`[1,2,2]`$。AG sequence shards后每rank shape恢复$`[1,4,2]`$。
33. 单向：
    $`8\times4\times2=64\ \text{bytes}.`$
    返回同样payload，总endpoint payload $`64+64=128`$ bytes；实际协议metadata/对齐未计。
34. 平均$`8/4=2`$ tokens/expert。最忙为5，所以：
    $`5/2=2.5\times\text{ average}.`$
35. $`Q_0\to\{K_0\}`$；$`Q_1\to\{K_0,K_1\}`$；$`Q_2\to\{K_0,K_1,K_2\}`$；$`Q_3\to\{K_0,K_1,K_2,K_3\}`$。
36. $`800/4=200\ \text{MiB}.`$
    只分sequence-side persistent activation/KV；parameters、optimizer、communication buffers、正在传输的KV与workspace不一定除4。
37. 单卡不失败$`=0.999`$；1024卡全不失败$`=0.999^{1024}\approx0.359`$。至少一个失败：$`1-0.359=0.641=64.1\%`$。期望数：$`1024\times0.001=1.024`$。
38. 均匀失败时刻的平均回退是半个间隔：$`30/2=15`$分钟。每小时2次、每次2分钟：$`2\times2=4`$分钟；比例$`4/60=6.67\%`$。

### 27.3 第39–55题：组合、配置与系统账

39. $`MFU=400/1000=0.4=40\%.`$
40. $`8\times4\times2=64\ \text{GPUs}.`$
    每node 8张：$`64/8=8`$ nodes。
41. TP groups：$`64/8=8`$组，每组8 ranks。PP chains：$`64/4=16`$条，每条4 ranks。DP groups：$`64/2=32`$组，每组2 ranks。
42. Rank5：TP group `{0,1,2,3,4,5,6,7}`；PP chain保持lane5，为`{5,13,21,29}`；对应第二replica rank为$`5+32=37`$，DP group `{5,37}`。
43. 先定义两个小学算术操作：$`\lfloor x\rfloor`$（floor，向下取整）是“不超过$`x`$的最大整数”；$`a\bmod b`$（mod，余数）是$`a`$除以$`b`$后剩下多少。先算$`44-32=12`$，所以是DP replica1。stage：
    $`\left\lfloor12/8\right\rfloor=\lfloor1.5\rfloor=1.`$
    lane：因为$`12=1\times8+4`$，所以：
    $`12\bmod8=4.`$
    TP group `{40..47}`；PP chain `{36,44,52,60}`；DP group `{12,44}`。
44. $`8\times1\times16\times128=128\times128=16{,}384.`$
45. $`8\times16\times16\times8=16{,}384.`$
    sequence倍数$`131{,}072/8192=16`$；DP缩小倍数$`128/8=16`$。
46. 9B：$`1024\times4=4096`$ chips。27B：$`768\times8=6144`$ chips。
47. 已知乘积：$`4\times4\times1\times8=128`$。若world256且axes正交，推测$`DP=256/128=2`$。因为课件只说“likely”，答案必须保留推测标签。
48. $`2\times8\times32=512.`$
    这只验证课件引用的NVIDIA recipe内部卡数，不证明Qwen团队原始预训练唯一采用这组degree。
49. 不重算$`2/1.0=2.0`$ samples/s。重算后$`4/1.25=3.2`$ samples/s。提升：
    $`(3.2-2.0)/2.0=0.6=60\%.`$
50. $`2/1.25=1.6\ \text{samples/s}.`$
    下降：$`(2.0-1.6)/2.0=0.2=20\%`$。
51. 总model-shard degree$`=4\times4=16`$；
    $`480/16=30\ \text{GB/rank}.`$
    理论余量$`80-30=50`$GB。
52. 峰值账：$`60+6+12+5=83`$GB。$`80-83=-3`$GB，也就是超过3GB，发生OOM；“长期60GB”不够判断峰值。
53. 不能overlap：$`100+30=130`$ms。communication全藏在compute下：$`\max(100,30)=100`$ms。
54. 单个microbatch依次穿越四段，latency是：
    $`1+1+2+1=5\ \text{ms}.`$
    稳态每隔多久能吐出一个microbatch由最慢stage决定：$`\max(1,1,2,1)=2`$ms，所以cadence为2ms。若四段都1ms，单microbatch latency$`=1+1+1+1=4`$ms，稳态cadence$`=1`$ms。题目没有给microbatch总数$`m`$，因此不询问完整step总时间。
55. Rank0：$`6\times3=18`$ms；rank1：$`2\times3=6`$ms。同步阶段要等最慢者，因此18ms；rank1空等$`18-6=12`$ms。

### 27.4 第56–80题：概念、来源边界与p56回归

56. ZeRO-1切optimizer states；ZeRO-2再切gradients；ZeRO-3再切parameters。三者都以data-parallel ranks为sharding group。
57. FSDP主要切parameter/gradient/optimizer state，不改变每rank local tokens产生的saved activations。候选包括TP+SP、CP、activation recomputation、FlashAttention或缩小microbatch；写出任意两个并说明目标即可。
58. PP切layers/depth；TP切单层matrix/head/hidden width；SP切pointwise activation的sequence轴；EP分配whole experts；CP切全层context tokens并在attention交换KV。
59. TP几乎每层都需collective，频率高且常在关键路径，所以优先高速节点内链路。PP主要在stage边界P2P，通信频率较低，较能容忍跨节点；这仍需实测。
60. SP通常与TP共享同一group：同一批$`t`$ ranks在matrix区用TP、pointwise区用SP。它是layout切换，不代表再创建$`t`$倍新ranks。
61. Attention TP与expert ETP可不同；EP ranks还可能复用attention的DP/TP坐标，EDP又是剩余副本轴。必须读world size与process-group构造，不能按缩写全乘。
62. Softmax权重是$`e^{z_i}/\sum_j e^{z_j}`$。对scores$`=[0,0,0]`$，$`e^0=1`$，global weights是$`[1/3,1/3,1/3]`$。配values$`=[0,0,3]`$：
    $`0/3+0/3+3/3=1.`$
    若前两项一块，local output$`=(0+0)/2=0`$；后一项一块，local output$`=3`$；两个blocks等权平均得$`(0+3)/2=1.5`$。它相当于错误weights$`[1/4,1/4,1/2]`$。Online softmax应维护running maximum、exponential sum和weighted numerator；本例最后sum$`=3`$、numerator$`=3`$，output$`=3/3=1`$。
63. Degree不能为0。课件中的`PP=0`/`EP=0`应读作未启用该axis或表格记法，而不是乘法因子0；必要时写成“none/degree1”。
64. 保持未知，并说明因缺DP/CP degree无法复原完整world-size乘积；不要从别的模型或阶段补值。
65. OLMo是模型，Dolma是其使用的开放语料。p63把FSDP写成了`FDSP`，是课件拼写问题。
66. DeepSeek LLM报告确认DP、TP、SP、1F1B PP和ZeRO-1，但完整degrees未列。DeepSeek-V3报告/课件确认ZeRO-1、PP16、EP64以及通信计算overlap；p72的DP/CP仍未知。
67. Yi原系列是dense family，课程列ZeRO-1+TP+PP；Yi-Lightning是MoE并引入EP。模型结构不同，不能把两者压成同一套degrees。
68. sequence从8K增到128K，单序列activation/KV压力大。CP16沿context分片；固定总GPU数时，对应DP从128降到8，让卡转用于同一序列而不是更多data replicas。
69. TPU的mesh、collective实现和model-shard语义不等同GPU NCCL拓扑。只能学习“data replicas×model shards=chips”的结构，不能复制degree后保证同样性能。
70. 已知axes乘积128，world256时可反推2，但课件写“likely”。缺少原始训练披露，所以只能说NVIDIA recipe在正交假设下推测DP2。
71. $`PP=0`$不是合法degree；p72又把PP写成`??`。应保留未知/未使用边界，不能算$`2\times0\times64\times64=0`$。
72. p71标题225与表、官方Qwen3-235B-A22B名称冲突。记录内部冲突，并采用官方/表中的235B；不能静默把所有来源说成一致。
73. MFU是“模型有效FLOP/s÷理论峰值FLOP/s”。GPU可能忙于通信、重算、数据移动等但不计入模型有效FLOPs，因此它不是简单的忙碌时间百分比。
74. 表给event counts，却没给每类设备的总暴露GPU-days；事件还可能由同一机架/网络故障相关触发。因此不能求单GPU独立日故障率。
75. （1）先分别算static state与dynamic activation峰值，让模型fit。（2）用ZeRO/FSDP、TP/EP、PP解决不同model-state限制。（3）用SP/CP/recomputation处理activation与长context。（4）把高频TP/EP通信放快链路，PP/DP按拓扑组合。（5）再用profiler平衡batch、bubble、matrix size和communication overlap。（6）最后用checkpoint、故障检测、重启和一致性验证把瞬时吞吐变成可持续训练。
76. 两个forward矩阵乘各有$`BDF`$个multiply-add，每个乘加按2 FLOPs，所以$`2\times2BDF=4BDF`$。Backward对两层分别算$`dX`$与$`dW`$，共4个同规模矩阵乘，所以$`4\times2BDF=8BDF`$。两个bfloat16权重矩阵共有$`2DF`$元素，即$`4DF`$ bytes；forward两次weight AG合计$`4DF`$，backward对两个矩阵各做weight AG与gradient RS，合计$`8DF`$。

    再算$`BDF=8\times2\times4=64`$、$`DF=2\times4=8`$。FSDP FLOPs：
    $`4BDF/X+8BDF/X=4\times64/2+8\times64/2=128+256=384.`$
    Communication：
    $`4DF+8DF=4\times8+8\times8=32+64=96\ \text{bytes}.`$
    $`T_{math}=384/24=16`$s；$`T_{comms}=96/12=8`$s；$`R=16/8=2>1`$，所以玩具例compute-bound。
77. $`XY=2\times2=4`$。混合FLOPs：
    $`4\times64/4+8\times64/4=64+128=192.`$
    $`BD=8\times2=16`$。Forward bytes：
    $`4BD/X+4DF/Y=4\times16/2+4\times8/2=32+16=48.`$
    Backward bytes：
    $`8BD/X+8DF/Y=8\times16/2+8\times8/2=64+32=96.`$
    总bytes$`=48+96=144`$。
78. $`300<400`$：没有曲线超过1。$`400<600<850`$：只有FSDP+MP绿线超过1。$`1000>850`$：FSDP+MP绿线与FSDP-only蓝线超过1；MP-only橙线仍低于1。400/850只是该图约数。
79. Global：三个$`e^0`$都是1，分母$`=3`$，output$`=(0+0+3)/3=1`$。分块错误法：第一块weights$`[1/2,1/2]`$、output0；第二块weight$`[1]`$、output3；等权平均$`(0+3)/2=1.5`$。错误法隐含weights$`[1/4,1/4,1/2]`$。
80. $`148/419\approx0.3532=35.32\%,`$
    不是30.1%。检查另外三行：
    $`72/419\approx17.18\%\to17.2\%,`$
    $`54/419\approx12.89\%\to12.9\%,`$
    $`35/419\approx8.35\%\to8.4\%.`$
    这里箭头表示按一位小数四舍五入。其余三行都与总数419吻合，因此在这张表内，最可能是Faulty GPU行的30.1%单元格有误。但没有底层event数据便不能判断源文件应改成35.3%、还是148本身来自另一版本；所以只报告冲突，不擅自修源。

---

## 28. 全视频时间导航：按问题跳，不必从头拖进度条

以下每个链接都命中人工 `en-US` 字幕 cue；秒数在全文中不重复。

| 时间 | 视频在回答什么 | 对应正文 |
|---|---|---|
| [00:07](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=7s) | 从并行底层机制进入本讲 | §0–§1 |
| [02:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=120s) | 大模型不能塞进一张GPU | §1 |
| [06:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=361s) | 通信层次与近处快速链路 | §2 |
| [08:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=480s) | MoE token route为何需要灵活网络 | §2、§17 |
| [12:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=720s) | 从网络基础转入训练并行 | §3 |
| [14:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=840s) | 为什么会有多份weights相关状态 | §4 |
| [16:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=962s) | ZeRO分片带来的显存节省 | §5 |
| [18:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1084s) | parameters、gradients与states | §5–§6 |
| [20:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1204s) | gradient何时可立即释放 | §6 |
| [22:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1322s) | backward后gradient shard | §7 |
| [24:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1442s) | FSDP预取下一层parameter | §7 |
| [26:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1560s) | 把module包成FSDP版本 | §7 |
| [28:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1684s) | 静态纸面上限仍可能装不下 | §7.8、§22 |
| [30:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1802s) | batch与GPU idle的冲突 | §8–§9 |
| [32:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1923s) | 最朴素的layer split | §9 |
| [34:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2043s) | microbatch把bubble预算换到别处 | §9 |
| [36:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2163s) | pipeline课堂暂停与问题边界 | §9–§10 |
| [38:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2280s) | 怎样理解更复杂pipeline patterns | §10 |
| [40:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2401s) | TP核心primitive反复出现 | §11 |
| [42:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2520s) | MLP down-projection怎样切 | §12 |
| [44:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2640s) | TP和PP通信形态不同 | §13 |
| [46:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2760s) | 如何压低红色activation峰 | §14–§15 |
| [48:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2883s) | TP后还剩哪些activation | §14 |
| [50:04](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3004s) | SP在操作边界做AG/RS | §16 |
| [52:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3121s) | optimizer state仍需占内存 | §16、§20 |
| [54:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3240s) | TP切matrix与EP切experts的差异 | §17 |
| [56:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3360s) | EP/TP/PP的systems比较 | §17–§18 |
| [58:03](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3483s) | 不同parallel primitives不能孤立看 | §18 |
| [60:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3601s) | 高TP也会把MoE matrix切太小 | §18 |
| [62:01](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3721s) | 总结表的data/model parallel rows | §20 |
| [64:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3842s) | 作业会实际算compute/communication | §20–§22 |
| [66:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3962s) | 为什么不断叠加parallel strategies | §21 |
| [68:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4080s) | practitioner版本的组合规则 | §22 |
| [70:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4202s) | 大量scale实验怎样选degrees | §22、§24 |
| [72:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4320s) | recomputation换batch和utilization | §22 |
| [74:02](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4442s) | DeepSeek的大EP如何跨节点 | §23.3 |
| [76:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4565s) | Llama训练中的GPU failures | §24 |
| [78:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4680s) | Nemotron long-context与CP/EP | §23.8 |
| [79:31](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4771s) | 快慢链路与全部资源的总总结 | §30 |

---

## 29. PDF逐页覆盖、视觉核验、来源边界与验证范围

### 29.1 1–73页无遗漏覆盖索引

下面的页段互不重叠，首尾连续覆盖1–73；“映射”表示这些页的实质内容落在对应章节，不表示逐页逐字翻译。

| PDF页 | 视觉主题 | 笔记章节 |
|---:|---|---|
| 1–3 | 标题、目标、三段课程组织 | §0 |
| 4–6 | 单GPU compute/memory上限，多GPU/多机 | §1 |
| 7–8 | collective回顾，RS+AG分解 | §2 |
| 9–12 | TPU mesh、GPU switched/tree、domain size | §2 |
| 13–14 | Part 1回顾与parallelism目录 | §0、§8 |
| 15–17 | naive DP、SGD、16 bytes/param | §3–§4 |
| 18–21 | ZeRO总览与ZeRO-1 | §5 |
| 22–23 | ZeRO-2 incremental gradient reduce/free | §6 |
| 24–28 | ZeRO-3/FSDP、overlap、通信与12-byte fit表 | §7 |
| 29–31 | DP剩余限制与model parallel入口 | §8 |
| 32–36 | naive layer split、microbatch pipeline与bubble | §9 |
| 37–38 | GPipe/1F1B/interleaving/zero-bubble、dX/dW | §10 |
| 39–40 | column/row tensor parallel数学 | §11 |
| 41–42 | Transformer block TP切法与前后向通信 | §12 |
| 43 | TP与PP通信近似式 | §13 |
| 44–45 | 训练memory timeline与activation峰值 | §14 |
| 46–47 | baseline与TP activation公式 | §14–§15 |
| 48–49 | sequence parallel与TP+SP公式 | §14–§16 |
| 50–51 | expert parallel、TP/EP对比 | §17 |
| 52–53 | attention与MoE parallel groups解耦 | §18 |
| 54 | context parallel/Ring Attention | §19 |
| 55 | 全parallelism比较表 | §19–§20 |
| 56 | DP/FSDP/MP/混合cost表，4/8系数来源，$`X_{opt}\sim\sqrt B`$推导，$`B/N`$横轴与400/850边界 | §20.5 |
| 57–58 | 3D/4D组合规则与Megatron建议 | §21–§22 |
| 59–60 | Narayanan规模表、per-GPU throughput | §22、§24 |
| 61–62 | 162.2B/64-GPU sweep与recomputation图 | §22、§24 |
| 63 | OLMo/Dolma + FSDP | §23.2 |
| 64 | DeepSeek LLM/V3 | §23.3 |
| 65 | Yi/Yi-Lightning | §23.4 |
| 66–67 | Llama3 stages与failure table | §23.5、§24 |
| 68 | Gemma2 TPU配置 | §23.6 |
| 69 | Mixtral8x22B Megatron recipe | §23.7 |
| 70 | Nemotron3 Super长context配置 | §23.8 |
| 71 | Qwen3 Megatron recipes | §23.9 |
| 72 | 多模型汇总表与`??` | §23.10 |
| 73 | 全讲recap | §30 |

### 29.2 视觉检查如何完成

- `pypdf`确认73页并抽取文字；公式/图表不只依赖文字层。
- `pypdfium2`渲染73/73页，生成8张contact sheets逐页看页序、裁切、黑/空页。
- p7–73另渲染2.5倍高分辨率PNG，共67张；对p8、15–28、34、37–49、52–73逐页放大核对公式、轴、图例、表格与笔误。
- p52–55核对parallel folding/CP；p56–62核对曲线轴与“64 machines/64 GPUs”冲突；p63–72核对每一个degree、`??`、`0`、225/235和FDSP/FSDP；p73确认只有三条recap，无隐藏数字。

### 29.3 课程来源

- [官方 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)，本地版本73页。
- [Stanford Online官方视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)，人工`English (United States)`字幕，1870 segments，末段80:01–80:05。

课程中的公式、课堂经验和模型表都按该讲2026时点呈现。尤其p63–72不是“当前所有模型的永恒配置库”。

### 29.4 一手补充来源

- [ZeRO论文](https://arxiv.org/abs/1910.02054)；[PyTorch FSDP官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)。
- [Megatron-LM论文](https://arxiv.org/abs/2104.04473)；[sequence parallel activation论文](https://arxiv.org/abs/2205.05198)。
- [How to Scale Your Model：training parallelism](https://jax-ml.github.io/scaling-book/training/)：支持p56表格变量、逻辑通信/计算公式与混合曲线的$`\sqrt B`$形状；不把课件图上的约400阈值推广为该来源在任意硬件上的精确结论。
- [Ring Attention论文](https://arxiv.org/abs/2310.01889)；[Megatron Core CP文档](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md)。
- [Megatron Core MoE指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)；[MoE Parallel Folding论文](https://arxiv.org/abs/2504.14960)。
- [OLMo报告](https://arxiv.org/abs/2402.00838) 与 [Dolma论文](https://arxiv.org/abs/2402.00159)。
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954)；[DeepSeek-V3](https://arxiv.org/abs/2412.19437)。
- [Yi](https://arxiv.org/abs/2403.04652)；[Yi-Lightning](https://arxiv.org/abs/2412.01253)。
- [Llama 3](https://arxiv.org/abs/2407.21783)；[Gemma 2](https://arxiv.org/abs/2408.00118)。
- [NVIDIA Megatron Mixtral示例](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mixtral/README.md)；[Nemotron 3 Super官方页](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)。
- [Qwen3报告](https://arxiv.org/abs/2505.09388)；[NVIDIA Megatron Bridge Qwen3文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/qwen/qwen3-moe.md)。

这些来源只用于核对术语、算法边界和公开配置；没有把后来更新的recipe倒灌成课程原话。

### 29.5 测试与诚实边界

- 所有显存、shape、group、bubble、失败概率与throughput小例都可在CPU上独立四则复算。
- 本环境没有真实多节点GPU/TPU集群，未运行NCCL/FSDP/Megatron distributed benchmark；本文不声称测得真实MFU、网络带宽或峰值显存。
- 公式中的`≈`表示忽略latency、协议、temporary buffers、overlap或不均衡后的教学近似。
- 动态文档与recipes会变化；复现实验必须记录framework commit、hardware、precision、global/micro batch及全部parallel degrees。

---

## 30. 一页复习流程与学完后的能力清单

### 30.1 一分钟从OOM走到方案

```text
先量峰值是谁
  ├─ static model state → ZeRO/FSDP、TP/EP、PP
  └─ activation/context → SP、CP、recompute、FlashAttention、microbatch
              ↓
写出每rank shape、bytes、临时峰值与通信
              ↓
把高频TP/EP放快链路；PP/DP按拓扑安排
              ↓
检查global batch、pipeline bubble、tokens/expert和matrix大小
              ↓
profile → 改一项 → 再验correctness/memory/tokens/s
              ↓
checkpoint、故障恢复、straggler监控
```

### 30.2 十个必须带走的句子

1. DP切samples，不自动切model states；ZeRO按stage逐步切optimizer、gradient、parameter。
2. 16 bytes/param与p28的12 bytes/param是两套明确假设，不能混算。
3. FSDP长期分片省static memory，但layer AG transient与activation仍可OOM。
4. PP切depth、TP切layer width、SP切pointwise sequence、EP切experts、CP切context。
5. Pipeline的bubble百分比必须写分母；更多microbatches也会缩小local compute。
6. TP-only不保证所有activation除以$`t`$；TP+SP才处理剩余pointwise项。
7. CP/Ring Attention需要全局一致的online softmax统计，不能平均block softmax。
8. Dense正交DP×TP×PP可相乘；MoE group复用/parallel folding下不能盲乘所有缩写。
9. 真实配置是模型、阶段、硬件与软件版本的快照；`??`必须保持未知。
10. 峰值吞吐没有故障恢复就不是可持续训练吞吐。

### 30.3 你现在应该能做到

- 从parameter个数和dtype/state口径手算baseline、ZeRO-1/2/3静态显存。
- 画出FSDP两层forward/backward的AG、RS、prefetch和free时间线。
- 独立算PP利用率的三种分母、TP/PP通信元素与activation公式。
- 用小矩阵验证column+row TP，用token表验证SP/EP/CP数据移动。
- 为64 GPUs列出TP groups、PP chains、DP groups和global batch。
- 对课程模型表逐格判断“确认、推测、未知、笔误”，不把recipe误称原始训练事实。
- 用memory snapshot与profiler证据选择组合，而不是背一个万能degree。
- 计算checkpoint间隔、故障概率、MFU，并解释每个数不能说明什么。
