# CS336 Lecture 7：Parallelism I——从通信原语到多 GPU 互连

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Percy Liang  
> 视频：[Lecture 7](https://www.youtube.com/watch?v=SzpOcwdIL0Y)（约 80:57）  
> 官方可执行讲义：[lecture_07.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)

这里的 **GPU（Graphics Processing Unit，图形处理器）**指训练模型用的并行加速器；“多 GPU”就是让多个加速器协作完成一次训练。

> **资料版本核验：**本讲没有 PDF。2026-08-28 核验的 GitHub 当前页面与仓库提交中，`lecture_07.py` 都是 **619 个物理行**；最近一次修改该文件的提交是 `0be5c6121acb3ce2cef5ec1cad1a0b7ebc8d2012`（2026-04-20，`update lecture 7`），检查时仓库 `HEAD` 为 `8b59b50730766695c2ffedd1a79c50cd09b9eb91`。抓取工具曾返回 **556 行旧 raw 缓存**，它不是本笔记的覆盖基准；本笔记以当前 619 行版本为准。

> **字幕核验：**YouTube 同时提供自动轨 `English (auto-generated)` 与人工轨 `English (United States)`。本笔记完整抓取并使用后者：语言代码 `en-US`、`kind` 为空（即非自动生成），共 **1312 个字幕片段**；末段从 **80:54** 开始，约在 **80:57** 结束，文字为 `on more parallelism techniques.`。自动轨只用来确认轨道存在，没有作为主字幕。

本讲使用四种来源标签：

- **【课程代码】**：当前 619 行官方 `lecture_07.py` 明写的内容；
- **【视频补充】**：老师的口头解释、现场演示或课堂问答；
- **【补充理解】**：为只会四则运算的读者补上的定义与推导；
- **【补充例子】**：本笔记自建、可以手算的数字例；
- **【课程时点快照】**：硬件型号、带宽和集群形态依赖课程时点与具体产品，不能当成永久常数。

### 资料与图片怎样核验

讲义一共调用 `image(...)` 7 次，其中 `gpu-node-overview.png` 重复出现两次，因此是 **6 个不同的被引用图像资产：5 个本地图 + 1 个外链图**。所有本地图按原始分辨率打开；外链 Springer 图片直链当时拒绝新下载，但同一官方仓库提交保存了讲义运行时的原始缓存，因此从 Git 对象中恢复后按原始分辨率检查。不是只根据文件名或图注猜内容。

为读懂下表先知道：**CPU（Central Processing Unit，中央处理器）**负责主机侧通用计算与程序控制；**RAM（Random Access Memory，随机存取内存）**是 CPU 一侧的主存；**SM（Streaming Multiprocessor，流式多处理器）**是 GPU 上执行并行计算的硬件单元；**register（寄存器）**是 SM 内很小、很快的线程私有存储；**L1/L2（Level-1/Level-2 cache，一/二级缓存）**是靠近计算单元、容量较小的高速缓存；**shared memory（共享内存）**是同一 GPU thread block（线程块，即一组协作线程）可共用的片上存储；**HBM（High Bandwidth Memory，高带宽内存）**是 GPU 的大容量显存；**NVLink** 是 NVIDIA 的 GPU 高速互连；**NVSwitch** 是把多条 NVLink 接起来、为多 GPU 转发数据的交换芯片；**PCIe（Peripheral Component Interconnect Express，高速外设互连）**是 CPU、GPU、网卡等设备常用的总线；**InfiniBand** 是数据中心常用的高性能网络；**Ethernet（以太网）**是更通用的网络技术。**Rank** 暂时只理解成参与进程的编号，§2 会从头解释。

| 图像资产 | 像素 | 实际视觉检查到的内容 | 笔记去向 |
|---|---:|---|---|
| `gpu-node-overview.png` | 2469×1381 | 4 个 GPU；每个 GPU 内画出多个 SM、寄存器、L1/shared memory、L2 与 HBM；4 条 NVLink 接到一个 NVSwitch，底部再连 InfiniBand/Ethernet | §1、§6 |
| `ranks.png` | 813×114 | 4 个并列框，依次标 `Rank 0` 到 `Rank 3` | §2 |
| Springer `Fig1` 外链缓存 | 685×350 | 两台服务器；每台都有 RAM、2 个 CPU、10 个 GPU，GPU 经 PCIe 接 CPU；两台服务器再经 Ethernet 相连 | §6 |
| `data-parallelism.png` | 625×889 | 4 层模型完整保留；橙线横切底部 `Data`，表示按数据切 | §11 |
| `tensor-parallelism.png` | 699×897 | 橙线纵向穿过每一层，表示每层按宽度切；数据不切 | §12 |
| `pipeline-parallelism.png` | 697×865 | 橙线横切 layer 1 与 layer 2 之间，表示按深度分层 | §13 |

同目录还存在 `siglip-parallelism.png`；也已视觉打开核验，但当前 619 行 `lecture_07.py` **没有引用它**，所以不把它冒充本讲第 7 张课程图。

### 官方 619 行源码连续覆盖索引

这张表的区间从 1 到 619 首尾相接，没有遗漏或重叠。阶段 1 尚未展开的后半内容先标出最终去向；“被索引”不等于“已经逐行讲完”。索引中会提前出现后文术语，第一次学习可跳过；**collective operation（集体通信操作）**先理解成“多个参与者共同完成的一种通信模板”，§3 再正式拆解。

| 官方行段 | 代码内容 | 对应笔记 |
|---:|---|---|
| 1–18 | imports、课程展示工具、无 CUDA 时的同步替身 | 来源说明；后续代码运行边界 |
| 19–74 | `main()`：层级、两种扩展动机、上下半讲地图与总结 | §0–1、§11–15 |
| 75–138 | collective 定义；broadcast、scatter、gather、reduce | §2–3 |
| 139–184 | all-gather、reduce-scatter、all-reduce | §4 |
| 185–208 | all-to-all 与术语记忆法 | §5 |
| 209–238 | 传统/现代互连、RDMA、RoCE、NCCL | §6–7 |
| 239–286 | `torch.distributed`、4 进程示例、三种 workhorse collective | §2–4、§7–8 |
| 287–337 | all-reduce benchmark、同步与带宽口径 | §6、§9–10 |
| 338–374 | reduce-scatter benchmark 与带宽口径 | §6、§9–10 |
| 375–389 | data parallel 总入口和结论 | §11 |
| 390–396 | 生成 `[128,1024]` 样例数据 | §11 |
| 397–438 | 朴素 data parallel 训练循环 | §11 |
| 439–446 | tensor parallel 总入口 | §12 |
| 447–483 | column tensor parallel 前向与 all-gather | §12 |
| 484–491 | pipeline parallel 总入口 | §13 |
| 492–537 | 两级 pipeline、micro-batches、send/recv | §13 |
| 538–556 | 分隔、setup（未 `set_device`）与 cleanup | §2、§7 |
| 557–576 | tracing 时禁用分布式调用的 context manager | §7 |
| 577–593 | `spawn`：真实多进程与 trace rank0/原 world-size 分支 | §2、§7 |
| 594–619 | 参数初始化、整除/摘要/时间 helper、程序入口 | §7、§11–13；后续代码附录 |

---

## 0. 五分钟复习卡、全讲地图与第一次阅读方法

> **第一次阅读请先跳过复习卡。**先读 §1 的“小模型显存账”，再把 §3–5 每一张输入/输出表亲手抄一遍；第二遍复习再回来背关键词。

### 0.1 一句话主线

**单个 GPU 装不下或算得太慢时，我们把模型状态、数据或计算分给多个进程；但分开之后必须搬数据，所以真正的问题不是“GPU 越多越快”，而是“怎样用正确的 collective（集体通信）完成必要的数据交换，并让通信成本小于新增计算能力带来的收益”。**

**【课程代码｜行 19–39】【视频补充｜[00:15](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=15s)】**上一讲研究单个 GPU 内的 kernels；本讲把画面拉远到多个 GPU。老师在 [01:45](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=105s) 给出共同主题：计算单元离数据很远，优化就是减少或隐藏数据搬运。

### 0.2 全讲因果链

```text
模型状态或训练工作量超出一张 GPU 的能力
                    ↓
先决定：复制、切分、重算，还是跨设备通信
                    ↓
每张 GPU 通常由一个 process 控制；process 在 group 中有 rank
                    ↓
broadcast / scatter / gather / reduce
帮助理解三个 workhorse：all-gather / reduce-scatter / all-reduce
                    ↓
MoE（Mixture of Experts，混合专家模型；§5.2 正式解释）动态路由还需要 all-to-all
                    ↓
逻辑 collective 交给通信库映射成 ring、tree 或其他物理算法
                    ↓
真实时间取决于消息步数、latency（每轮固定等待）、
bytes、bandwidth（每秒可传的 bytes）与硬件拓扑
                    ↓
后半讲再用这些积木构造 data parallelism（数据并行：切 batch）、
tensor parallelism（张量并行：切 layer 内 width）、
pipeline parallelism（流水线并行：切 layers/depth）
```

**【视频补充｜[02:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=124s)】**老师把存储/通信画成层级：同一 GPU 内的 L1/shared memory 最快，然后是 HBM，再到同节点 NVLink/NVSwitch，最后才是跨节点 InfiniBand/Ethernet。这里的“快慢”是课程的概念排序，不代表所有机器都只有一种固定带宽。

### 0.3 五分钟复习卡

1. 多 GPU 有两个不同目标：**装得下（capacity）**与**算得快（speed）**；前者成功不保证后者成功。
2. `world_size=4` 表示一个 process group 中有 4 个参与者；rank 是组内编号 `0,1,2,3`。
3. `broadcast`：一个 root 复制给所有人；`scatter`：root 拆开分发；`gather`：各片段收回 root；`reduce`：各值先按 SUM/MAX 等合并，再把结果给 root。
4. `all-gather`：每人最后都有完整拼接；`reduce-scatter`：先逐位置合并，再让每人只留一片；`all-reduce`：每人最后都有完整合并结果。
5. 对四个向量 `[0,1,2,3]`、`[1,2,3,4]`、`[2,3,4,5]`、`[3,4,5,6]` 做 SUM，逐列是 `[6,10,14,18]`。
6. 逻辑上 `all-reduce = reduce-scatter + all-gather`；**API（Application Programming Interface，应用编程接口）**是程序调用软件功能的约定入口。物理实现不保证永远按这两次 API 调用，也不保证永远是 ring。
7. `all-to-all` 的平衡例子像把 4×4 矩阵转置；不平衡路由仍能做，但最忙的目标 rank 会拖慢大家。
8. 粗略通信时间：

   $`T\approx sL+\frac{Q}{B}.`$

   小消息常被延迟项 $`sL`$ 支配，大消息常被搬运项 $`Q/B`$ 支配。
9. `per-rank bytes`、全网络 `aggregate traffic` 与应用看到的 payload 大小不是同一个数。
10. NVLink/NVSwitch、PCIe、InfiniBand、Ethernet 是不同层级；NCCL 会结合 topology（拓扑，即设备怎样连接）选择实现路径。
11. DDP 每 rank 保存完整模型、处理不同 local batch；等大 local batch 且 local loss 都取 mean 时，AVG gradients 等于 global-batch mean gradient。
12. 课程 TP 把 `[1024,1024]` 权重按输出列切成四个 `[1024,256]`，每层 local output `[128,256]` 再 all-gather 成 `[128,1024]`。
13. 课程 PP 把 4 layers 分给 2 stages；4 个 microbatches（微批次，即从一个大 batch 再切出的小批；§13.3 详解）的理想 forward utilization 是 $`4/(4+2-1)=80\%`$。
14. DP、TP、PP 可组成多维 device mesh；一个 rank 在不同 process groups 中分别做梯度、层内和 stage 边界通信。

### 0.4 本讲最终地图

| 部分 | 要回答的问题 |
|---|---|
| §1 | 为什么要多 GPU？复制、切分、重算、通信分别改变什么？ |
| §2 | process、device、rank、world size、group、backend 是什么？ |
| §3–5 | 八种 collective 的输入和输出究竟怎样变化？ |
| §6 | GPU/节点怎样连？延迟与带宽怎样估算时间？ |
| §7–8 | NCCL/process group 怎样承接 API？课程 collective 代码怎样执行？ |
| §9–10 | benchmark 区间怎样定义？algbw/busbw 每个因子从哪里来？ |
| §11 | data parallel 怎样同步梯度？ |
| §12 | tensor parallel 怎样切矩阵宽度？ |
| §13 | pipeline parallel 怎样切模型深度并减少 bubble？ |
| §14–15 | 三种切法怎样比较、怎样组成 device mesh？ |
| 后续收尾 | 常见误区、自测、视频导航与来源边界 |

**【课程代码｜行 45–64】【视频补充｜[04:56](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=296s)】**视频把本讲分成两部分：先学通信积木、硬件和 PyTorch 接口，再用 MLP（Multilayer Perceptron，多层感知机）演示 data、tensor、pipeline 三种并行；MLP 用来保留核心矩阵计算，避免完整 Transformer 的额外记账遮住主线。

### 0.5 最少前置知识

读者只需要会下面四件事；不会编写分布式代码也可以开始：

1. 加法，例如 $`0+1+2+3=6`$；
2. 乘除法，例如 $`16/4=4`$；
3. 把向量看成一排数字，例如 `[2,3,4]` 的长度是 3；
4. 知道 1 byte 是存储单位，8 bits（位）等于 1 byte。

复习卡中的 **latency（延迟）**先理解成“每轮通信开始前的固定等待”，**bandwidth（带宽）**先理解成“每秒最多搬多少 bytes”。§6 会把单位和算式从头展开。

---

## 1. 为什么需要多 GPU：先分清“装得下”与“算得快”

### 1.1 训练时显存里到底装了什么

先把四个词说成人话：

- **parameter（参数）**：模型在训练中要学习的数字，例如矩阵中的权重；
- **activation（激活值）**：把一个 batch（一次共同处理的一小批样本）送进模型时，中间层临时算出的数字；反向传播常要用它们；
- **gradient（梯度）**：loss（损失，越小越好的“坏程度”）对每个参数的局部变化率；优化器根据它决定参数往哪边改；
- **optimizer state（优化器状态）**：优化方法为了记住历史而额外保存的数字。Adam 是常见优化方法，通常为每个参数保存一阶、二阶两个 moment（可理解为两份历史统计数组）。

这些数字通常存在 **tensor（张量，多维数字数组）**中。每个元素的数字存储格式叫 **dtype（data type，数据类型）**。例如 FP32（32-bit floating point，32 位浮点数）每个元素占 4 bytes（字节）。

**【课程代码｜行 37–39】【视频补充｜[03:28](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=208s)】**老师把扩展到多 GPU 的第一个原因说成：参数、activation、gradient 和 optimizer state 放不进一张 GPU 的 HBM。第二个原因是即使放得下，也想调用更多计算单元更快完成训练。

### 1.2 一个只用乘法和加法的显存账

**【补充例子】**假设小模型恰好有 $`1,048,576=2^{20}`$ 个参数，所有持久状态都用 FP32；先忽略通信 buffer（缓冲区）和框架开销。

先定义容量单位：

```math
1\ \text{MiB}=2^{20}=1,048,576\ \text{bytes}.
```

一份参数占：

```math
1,048,576\ \text{elements}\times4\ \text{bytes/element}
=4,194,304\ \text{bytes}=4\ \text{MiB}.
```

逐项列账：

| 项目 | 有几份“参数大小” | 计算 | 大小 |
|---|---:|---:|---:|
| 参数 | 1 | $`1\times4`$ MiB | 4 MiB |
| 梯度 | 1 | $`1\times4`$ MiB | 4 MiB |
| Adam 一阶 moment | 1 | $`1\times4`$ MiB | 4 MiB |
| Adam 二阶 moment | 1 | $`1\times4`$ MiB | 4 MiB |
| 持久状态小计 | 4 | $`4+4+4+4`$ | 16 MiB |
| 本例保存的 activations | 不按参数份数算 | 已知 | 6 MiB |
| 每张卡总计 |  | $`16+6`$ | **22 MiB** |

这只是教学账。真实训练还可能有低精度权重、FP32 master weights（高精度主副本）、临时 buffer、内存碎片、不同优化器状态；activation 大小也随 batch、sequence length（序列长度）、层数与是否重算而变化。

### 1.3 四个动作：replicate、shard、recompute、communicate

#### 动作一：replicate

**Replicate（复制）**：每个 rank 都保存一份相同数据。上例若 4 张卡都复制全部状态，那么**每张卡**仍是 22 MiB；4 张卡合计物理存储：

```math
4\times22=88\ \text{MiB}.
```

复制没有降低单卡这部分显存；收益是每张卡可直接使用本地副本，不必每次跨卡取。

#### 动作二：shard

**Shard（切分）**：把完整对象分片，每个 rank 只保存一部分。若把 16 MiB 持久状态均匀切到 4 张卡，但每张卡仍需要 6 MiB activations：

```math
\frac{16\ \text{MiB}}{4}+6\ \text{MiB}=4+6=10\ \text{MiB/rank}.
```

单卡从 22 MiB 降到 10 MiB，但某个 rank 需要完整参数时，必须临时收集别人的分片。

#### 动作三：recompute

**Recompute（重算）**：不保存某些中间 activation，反向传播需要时重新做一部分前向计算。若把 activation 存储从 6 MiB 降到 2 MiB：

```math
4\ \text{MiB sharded persistent state}+2\ \text{MiB activation}
=6\ \text{MiB/rank}.
```

显存再降 4 MiB，代价是多做计算。

#### 动作四：communicate

**Communicate（通信）**：把 tensor 从一个参与者传到另一个参与者。切分让单卡少存，通常会让训练过程增加 all-gather、reduce-scatter 等通信；通信又消耗时间和网络带宽。

因此没有免费的选择：

```text
复制更多 → 单卡显存高，通信可能少
切分更多 → 单卡显存低，通信通常多
重算更多 → activation 显存低，计算更多
加 GPU  → 峰值计算更多，也增加同步与传输问题
```

**【视频补充｜[03:46](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=226s)】**老师明确提醒：即使模型放得下，把它铺到更多 GPU 也要付通信带宽；是否更快必须计算，不能从 GPU 数量直接得出。

### 1.4 “更快”中的 FLOP、FLOPs 与 FLOP/s

- 一个 **FLOP（floating-point operation，浮点运算）**是一次浮点加法或乘法；
- **FLOPs** 在“需要多少 FLOPs”中指总工作量，不是秒；
- **FLOP/s** 是每秒完成多少浮点运算，才是吞吐率。

假设训练一步要 400 FLOPs：

- 单卡实际 100 FLOP/s：理想计算时间 $`400/100=4`$ 秒；
- 4 卡若完美分工，总计 400 FLOP/s：理想计算时间 $`400/400=1`$ 秒；
- 若每步另花 0.6 秒通信，总时间是 $`1+0.6=1.6`$ 秒，不是 1 秒；
- 加速比是 $`4/1.6=2.5`$ 倍，不是 4 倍。

这就是两个目标的区别：**fit memory** 问“能不能跑”；**faster** 问“端到端一步是否真的更短”。

---

## 2. 分布式程序的六个基础词：process、device、rank、world size、group、backend

### 2.1 Process 不等于 GPU

- **process（进程）**：正在运行的一份程序实例，有自己的 Python 状态和内存空间；
- **device（设备）**：真正执行计算或存放 tensor 的硬件，本讲主要是 GPU；
- 常见部署是“一个 process 控制一张 GPU”，但这是常见映射，不是词义相等。

**【课程代码｜行 249–251、577–593】【视频补充｜[37:13](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2233s)】**`spawn(..., world_size=4)` 会启动 4 个 process，让每个 process 执行同一个函数，并传入不同 rank。课堂 trace 为方便逐行展示会走单进程替身；直接运行讲义才走真实 multiprocessing（多进程）。

### 2.2 Rank 与 world size

**Rank（秩/编号）**是一个 process 在某个通信组里的整数编号。**World size（世界大小）**是该组共有多少个参与者。

当 `world_size=4`：

```math
\text{valid ranks}=0,1,2,\ldots,\text{world\_size}-1=0,1,2,3.
```

注意是从 0 开始，因此最后一个编号是 $`4-1=3`$，不是 4。

```text
process P0 ── controls GPU 0 ── rank 0
process P1 ── controls GPU 1 ── rank 1
process P2 ── controls GPU 2 ── rank 2
process P3 ── controls GPU 3 ── rank 3

world_size = 4
```

**【课程代码｜行 81–89】【视频补充｜[06:55](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=415s)】**课程图就是四个并列 rank。老师在课堂问答 [42:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2528s) 限定“在本课里 rank 对应 GPU”；工程上仍应记住，rank 首先是 process/group 的编号。

### 2.3 Process group

**Process group（进程组）**是一组会共同参加 collective 的 processes。默认全体组常称 world group；也可以创建子组。

例如 8 个全局 ranks 可分成：

```text
全体 group：{0,1,2,3,4,5,6,7}，world size = 8
子组 A：    {0,1,2,3}，group size = 4
子组 B：    {4,5,6,7}，group size = 4
```

在子组 A 里做 all-reduce，不应把数据发送给 4–7。一个 rank 号码必须结合“在哪个 group”理解。

### 2.4 Global rank 与 local rank【补充理解】

课程前半把每个 rank 直接画成 GPU。多节点工程中还常区分：

- **global rank（全局编号）**：整个训练作业中的唯一编号；
- **local rank（节点内编号）**：只在当前机器内从 0 重新数，常用来选择本机哪张 GPU。

两台节点、每台两张 GPU：

| node（服务器节点） | process | global rank | local rank | device |
|---|---|---:|---:|---|
| node 0 | P0 | 0 | 0 | GPU 0 |
| node 0 | P1 | 1 | 1 | GPU 1 |
| node 1 | P2 | 2 | 0 | GPU 0 |
| node 1 | P3 | 3 | 1 | GPU 1 |

`local rank=0` 出现两次并不冲突，因为它们在不同 node；`global rank` 才在整个作业中唯一。

### 2.5 Backend 是“同一接口背后的实现”

**Backend（后端）**是 `torch.distributed` 接口背后真正执行通信的软件实现。课程代码在有 CUDA GPU 时用 NCCL，没有时用 Gloo：

```python
# 【课程代码的关键分支；不是本节要读者直接运行的完整程序】
if torch.cuda.is_available():
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
else:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

- **CUDA（Compute Unified Device Architecture）**是 NVIDIA GPU 的编程平台；
- **NCCL（NVIDIA Collective Communications Library）**是 NVIDIA 面向 GPU collective 的通信库，通常读作 “nickel”；
- **Gloo** 是 PyTorch 支持的另一通信后端，课程在 CPU/laptop 路径使用它。

**【视频补充｜[36:13](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2173s)】**PyTorch 给出统一 collective 接口；[36:34](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2194s) 说明 GPU 常用 NCCL backend，CPU 示例可用 Gloo。相同 `all_reduce` 语义可由不同 backend 完成。

### 2.6 Setup、metadata 与 barrier

**Metadata（元数据）**是描述“谁参加、怎样协调、地址在哪里”的小量控制信息，不是模型的大 tensor 本身。课程的 `MASTER_ADDR` 和 `MASTER_PORT` 用于建立/协调 process group；视频强调实际 GPU payload（有效载荷，即真正要搬的大数组）不会因此全走这个地址。

**Barrier（屏障）**是一种同步点：先到的 process 等，直到组内所有 process 都到达，大家才继续。

```text
时间向右 →
rank 0: 做事 ──到 barrier──等待────────继续
rank 1: 做事较久──────────到 barrier──继续
rank 2: 做事 ─────到 barrier──等待────继续
rank 3: 做事更久────────────到 barrier──继续
```

**【视频补充｜[38:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2313s)】**master address/port 主要负责协调；[39:13](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2353s) 逐步解释 barrier。Barrier 过多会让快的 rank 无谓等待，因此它不是“越多越安全越快”。

---

## 3. 四个基础 collective：broadcast、scatter、gather、reduce

### 3.1 先定义共同语言

**Collective operation（集体通信操作）**是让一个 group 中许多参与者共同遵守某种数据交换模板。它描述逻辑输入/输出，不要求使用者手写每一对 rank 的 point-to-point（点对点）发送。

**【课程代码｜行 75–89】【视频补充｜[05:50](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=350s)】**这些通信原语早于大语言模型；`collective` 的意思是一次声明多设备间的整体通信模式。老师在 [06:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=378s) 强调，它让系统有机会替使用者安排更合适的实现。

本节还要区分三个词：

- **root（根 rank）**：broadcast、scatter、gather、reduce 中被指定为源或最终目的地的 rank；不一定永远是 rank 0；
- **in-place（原地）**：调用后结果覆盖输入 tensor；
- **out-of-place（非原地）**：输入和输出是不同 tensor，需预先分配输出；
- **shape（形状）**：tensor 每个轴有多少个元素，例如 `[4]` 是长度 4 的向量，`[1]` 是长度 1 的向量。

课堂问答 [20:32](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1232s) 明确：示例为了方便选 rank 0；真实调用会指定 root，调用时必须让所有参与者对这个选择一致。另一问答 [21:27](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1287s) 说明这些既是概念积木，也会很快对应真实 API。

### 3.2 Broadcast：一个完整值复制给所有 rank

**Broadcast（广播）**的逻辑是：root 有完整输入，结束后所有 rank 都有它的副本。

**【课程代码｜行 91–101】【视频补充｜[08:16](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=496s)】**令 root 为 rank 0：

| rank | 操作前逻辑有效载荷 | shape | 操作后 | shape |
|---:|---|---:|---|---:|
| 0（root） | `[0,1,2,3]` | `[4]` | `[0,1,2,3]` | `[4]` |
| 1 | 无有效源数据 | — | `[0,1,2,3]` | `[4]` |
| 2 | 无有效源数据 | — | `[0,1,2,3]` | `[4]` |
| 3 | 无有效源数据 | — | `[0,1,2,3]` | `[4]` |

这里“无有效源数据”不代表实际 API 可以不给 buffer；许多接口要求非 root 也准备匹配 shape/dtype 的接收 tensor。表只描述逻辑内容。

常见一次性用途：rank 0 从 **checkpoint（检查点，即保存在磁盘上的模型/训练状态快照）**读取初始权重，再广播给所有 rank。视频在 [08:57](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=537s) 给出这个例子。

> **不要和 NumPy broadcasting 混淆。**课堂问答 [12:06](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=726s) 说两者都有“一份扩到多份”的直觉，但 NumPy broadcasting 是单进程里的 shape 运算规则；这里是跨 ranks 的数据通信。

### 3.3 Scatter：root 的完整向量被切成片段分发

**Scatter（散发）**：root 先有完整输入，把它按约定切成 $`p`$ 片；$`p`$ 是参与 rank 数，每个 rank 收一片。

**【课程代码｜行 103–113】【视频补充｜[09:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=550s)】**输入长度 4，world size 4，因此每片长度：

```math
\frac{4\ \text{elements}}{4\ \text{ranks}}=1\ \text{element/rank}.
```

| rank | 操作前 | 输入 shape | 操作后 | 输出 shape |
|---:|---|---:|---|---:|
| 0（root） | `[0,1,2,3]` | `[4]` | `[0]` | `[1]` |
| 1 | — | — | `[1]` | `[1]` |
| 2 | — | — | `[2]` | `[1]` |
| 3 | — | — | `[3]` | `[1]` |

检查总元素数：操作前是 4 个；操作后 $`1+1+1+1=4`$ 个逻辑输出元素。Scatter 改变“放在哪里”，没有把元素相加。

### 3.4 Gather：把各 rank 的片段收回 root

**Gather（收集）**与 scatter 方向相反：每个 rank 有一片，root 按 rank 顺序把它们拼起来。

**【课程代码｜行 115–125】【视频补充｜[09:50](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=590s)】**

| rank | 操作前 | shape | 操作后逻辑有效结果 | shape |
|---:|---|---:|---|---:|
| 0（root） | `[0]` | `[1]` | `[0,1,2,3]` | `[4]` |
| 1 | `[1]` | `[1]` | 非 root 没有完整结果 | — |
| 2 | `[2]` | `[1]` | 非 root 没有完整结果 | — |
| 3 | `[3]` | `[1]` | 非 root 没有完整结果 | — |

拼接顺序是：先 rank 0 的 `[0]`，再 rank 1 的 `[1]`，再 rank 2 的 `[2]`，最后 rank 3 的 `[3]`，所以得到 `[0,1,2,3]`。

### 3.5 Reduce：先合并数值，再把结果交给 root

**Reduce（归约）**不是拼接。它用 SUM（求和）、MAX（最大值）、MIN（最小值）等运算，把各 rank 对应位置的值合并。

**Scalar（标量）**就是单个数。严格说，shape `[1]` 是“含 1 个标量元素的一维 tensor”，不是 0-D 标量；真正 0-D scalar tensor 的 shape 是 `[]`。

**【课程代码｜行 127–137】【视频补充｜[10:45](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=645s)】**每个 rank 各有一个 shape `[1]` 的一维 tensor，其中装 1 个标量：

| rank | 操作前 | SUM 的中间算式 | 操作后逻辑有效结果 |
|---:|---:|---|---:|
| 0（root） | `[0]` | $`0+1+2+3`$ | `[6]` |
| 1 | `[1]` | 参与求和 | 非 root 不持有最终结果 |
| 2 | `[2]` | 参与求和 | 非 root 不持有最终结果 |
| 3 | `[3]` | 参与求和 | 非 root 不持有最终结果 |

逐步算：

```math
0+1=1,\qquad 1+2=3,\qquad 3+3=6.
```

所以 root 得到 `[6]`，shape 仍是 `[1]`。视频在 [11:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=680s) 现场算出 6。

### 3.6 四种基础操作放在一张表里

| 操作 | 谁一开始有完整数据/片段？ | 谁最后有结果？ | 是否做数值合并？ |
|---|---|---|---|
| broadcast | root 有完整数据 | 所有人有同一完整副本 | 否 |
| scatter | root 有完整数据 | 每人一片 | 否 |
| gather | 每人一片 | 只有 root 有完整拼接 | 否，只拼接 |
| reduce | 每人有可对应合并的数据 | 只有 root 有归约结果 | 是 |

记忆法：scatter 是“向外撒”，gather 是“向内收”，reduce 是“数值合并”，前缀 `all-` 会在下一节把最终目的地扩展到所有 ranks。

---

## 4. 三个主力 collective：all-gather、reduce-scatter、all-reduce

### 4.1 All-gather：每个 rank 最后都有完整拼接

**All-gather（全收集）**可以拆词记：

- `gather`：把各 rank 的片段按顺序拼接；
- `all`：不是只给 root，而是让**所有 rank**都得到完整拼接。

**【课程代码｜行 139–152】【视频补充｜[12:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=755s)】**

| rank | 操作前输入 | 输入 shape | 操作后输出 | 输出 shape |
|---:|---|---:|---|---:|
| 0 | `[0]` | `[1]` | `[0,1,2,3]` | `[4]` |
| 1 | `[1]` | `[1]` | `[0,1,2,3]` | `[4]` |
| 2 | `[2]` | `[1]` | `[0,1,2,3]` | `[4]` |
| 3 | `[3]` | `[1]` | `[0,1,2,3]` | `[4]` |

单个 rank 的输出元素数从 1 变为 4，所以它要准备长度 4 的输出空间。4 个 rank 合计输出副本元素数为：

```math
4\ \text{ranks}\times4\ \text{elements/rank}=16\ \text{stored elements}.
```

这 16 个物理副本来自 4 个不同输入元素的复制，不代表数学上产生了 16 个不同值。

**【视频补充｜[13:16](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=796s)】**用途预告：若每个 rank 只保存参数的一片，前向计算某层前可 all-gather 得到临时完整参数。

### 4.2 Reduce-scatter：逐位置 reduce，再把结果切开

**Reduce-scatter（归约散发）**做两件逻辑工作：

1. 对所有 rank 的输入逐位置 reduce；
2. 把归约后的完整向量按位置散发，每个 rank 留一片。

**【课程代码｜行 154–167】【视频补充｜[13:43](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=823s)】**四个输入都为 shape `[4]`：

| rank | 输入向量 |
|---:|---|
| 0 | `[0,1,2,3]` |
| 1 | `[1,2,3,4]` |
| 2 | `[2,3,4,5]` |
| 3 | `[3,4,5,6]` |

把它们竖着对齐：

```text
位置 j       0   1   2   3
rank 0       0   1   2   3
rank 1       1   2   3   4
rank 2       2   3   4   5
rank 3       3   4   5   6
            ───────────────── SUM
完整归约      6  10  14  18
```

每一列都展开：

```math
\begin{aligned}
r_0 &= 0+1+2+3=6,\\
r_1 &= 1+2+3+4=10,\\
r_2 &= 2+3+4+5=14,\\
r_3 &= 3+4+5+6=18.
\end{aligned}
```

然后 scatter：

| rank | 操作后输出 | shape | 为什么 |
|---:|---|---:|---|
| 0 | `[6]` | `[1]` | 留完整归约的第 0 片 |
| 1 | `[10]` | `[1]` | 留第 1 片 |
| 2 | `[14]` | `[1]` | 留第 2 片 |
| 3 | `[18]` | `[1]` | 留第 3 片 |

检查：每个输入有 4 个元素，输出每 rank 只有 1 个元素。Reduce-scatter 既完成跨 rank 求和，也让结果存储保持切分。

**【视频补充｜[14:27](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=867s)】**训练用途预告：不同数据分片产生不同局部 gradients；reduce-scatter 可先把它们逐元素求和，再让每个 rank 只保存一片最终 gradient。

### 4.3 All-reduce：每个 rank 都得到完整归约结果

**All-reduce（全归约）**的逻辑输出是：先把所有输入逐位置 reduce，再让所有 rank 都得到完整结果。

同一组输入的 SUM 结果为：

| rank | 操作前输入 | 操作后输出 | 输出 shape |
|---:|---|---|---:|
| 0 | `[0,1,2,3]` | `[6,10,14,18]` | `[4]` |
| 1 | `[1,2,3,4]` | `[6,10,14,18]` | `[4]` |
| 2 | `[2,3,4,5]` | `[6,10,14,18]` | `[4]` |
| 3 | `[3,4,5,6]` | `[6,10,14,18]` | `[4]` |

**【课程代码｜行 169–183】【视频补充｜[15:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=918s)】**用刚才两步严格验证：

```text
阶段 A：reduce-scatter
rank 0:[6]   rank 1:[10]   rank 2:[14]   rank 3:[18]

阶段 B：all-gather
每个 rank 按 0→1→2→3 的顺序拼接
→ [6,10,14,18]
```

所以在**逻辑语义**上：

```math
\text{all-reduce}
=\text{reduce-scatter}+\text{all-gather}.
```

视频在 [16:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=982s) 说明：基础 data parallel 常直接 all-reduce 完整 gradients；FSDP/ZeRO 等方法会把这两个阶段拆开，以便在中间维持 sharded storage。这里的 **FSDP（Fully Sharded Data Parallel，全切分数据并行）**和 **ZeRO（Zero Redundancy Optimizer，零冗余优化器）**只作后续预告。

### 4.4 SUM 与 AVG 不同

Collective 还要指定 reduce operation（归约运算）。对 4 个输入：

- SUM 输出：`[6,10,14,18]`；
- AVG（average，平均）要再除以参与者数量 4：

```math
\left[
\frac{6}{4},
\frac{10}{4},
\frac{14}{4},
\frac{18}{4}
\right]
=[1.5,2.5,3.5,4.5].
```

不能把 SUM 与 AVG 当同义词。若学习率等其余设置不变，梯度和比梯度平均大 $`4`$ 倍。课程后半 data parallel 代码使用 `ReduceOp.AVG`；当前概念例使用 SUM。

### 4.5 逻辑等式不指定唯一物理算法

“All-reduce = reduce-scatter + all-gather”回答的是**最终输入/输出语义**。通信库实际可能使用：

- **ring（环）**：rank 按环形邻居分块传输；
- **tree（树）**：先沿树向上合并，再向下分发；
- 分层算法：节点内先走 NVLink，节点间再走 InfiniBand，之后节点内分发；
- 其他依消息大小、拓扑和库版本选择的算法。

因此不能说“物理上 all-reduce 永远就是先调用一次 reduce-scatter API，再调用一次 all-gather API”。NCCL 看到的是逻辑 collective，然后选择具体 schedule（调度顺序）。

**【视频补充｜[30:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1803s)】**老师把 NCCL 描述为：读取硬件 topology，选择 GPU 间路径，并启动通信 kernels。这里的 **kernel（核函数）**是 GPU 上执行的底层程序。

### 4.6 In-place 与 out-of-place 的课程代码证据

课程的 all-reduce：

```python
# 【课程代码片段】data 同时是输入和输出：in-place
dist.all_reduce(
    tensor=data,
    op=dist.ReduceOp.SUM,
    async_op=False,
)
```

`data` 调用前是本 rank 输入，调用后被覆盖为 `[6,10,14,18]`。视频在 [40:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2401s) 开始逐行演示，在 [41:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2481s) 明说结果写回 `data`。

课程的 reduce-scatter：

```python
# 【课程代码片段】input 和 output 分开：out-of-place
dist.reduce_scatter_tensor(
    output=output,
    input=input,
    op=dist.ReduceOp.SUM,
    async_op=False,
)
```

输入 shape `[4]`，输出预分配 shape `[1]`。视频 [42:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2542s) 展开这个例子，强调输入不改、结果写到另一 tensor。

**CUDA stream（CUDA 流）**可以先理解为 GPU 上的一条**有序命令队列**：放进同一 stream 的 kernels/通信按提交先后建立顺序；不同 streams 之间默认不能凭提交先后猜依赖。**Work handle（工作句柄）**是 `async_op=True` 返回的“这项异步工作仍可被查询/等待”的凭证。

`async_op=False` 表示调用者不会拿到这个 Work handle；它不是“Python 返回时 GPU 所有物理工作已经完成”的简单承诺。若设为异步，程序可以先做不依赖结果的 host 或 GPU 工作；何时可安全使用结果还要看 `work.wait()`、CUDA stream 依赖和显式同步，§8.6 会完整拆开。课堂问答 [43:51](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2631s) 用“先发通信，再加载下一步数据”解释 communication/computation overlap（通信与计算重叠）。

最后，课程把 reduce-scatter 的 `[6]`、`[10]`、`[14]`、`[18]` 接入 all-gather；[45:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2703s) 开始逐行演示，[45:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2759s) 得到四份 `[6,10,14,18]`，与直接 all-reduce 相同。

---

## 5. All-to-all：每个发送 rank 都给每个目标 rank 一片

### 5.1 平衡 all-to-all 的 4×4 路由表

**All-to-all（全互换）**让每个 source rank（发送方）都为每个 destination rank（目标方）准备一片。平衡且每片一个元素时，可把输入看成一个矩阵：行是发送者，列是接收者。

**【课程代码｜行 185–201】【视频补充｜[16:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1008s)】**输入：

| source \ destination | rank 0 | rank 1 | rank 2 | rank 3 |
|---:|---:|---:|---:|---:|
| rank 0 | 0 | 1 | 2 | 3 |
| rank 1 | 4 | 5 | 6 | 7 |
| rank 2 | 8 | 9 | 10 | 11 |
| rank 3 | 12 | 13 | 14 | 15 |

逐个元素读：

- `0`：rank 0 → rank 0；`1`：rank 0 → rank 1；`2`：rank 0 → rank 2；`3`：rank 0 → rank 3；
- `4`：rank 1 → rank 0；`5`：rank 1 → rank 1；`6`：rank 1 → rank 2；`7`：rank 1 → rank 3；
- `8`：rank 2 → rank 0；`9`：rank 2 → rank 1；`10`：rank 2 → rank 2；`11`：rank 2 → rank 3；
- `12`：rank 3 → rank 0；`13`：rank 3 → rank 1；`14`：rank 3 → rank 2；`15`：rank 3 → rank 3。

按目标列收集：

| destination | 从 rank 0 收 | 从 rank 1 收 | 从 rank 2 收 | 从 rank 3 收 | 最终输出 shape `[4]` |
|---:|---:|---:|---:|---:|---|
| rank 0 | 0 | 4 | 8 | 12 | `[0,4,8,12]` |
| rank 1 | 1 | 5 | 9 | 13 | `[1,5,9,13]` |
| rank 2 | 2 | 6 | 10 | 14 | `[2,6,10,14]` |
| rank 3 | 3 | 7 | 11 | 15 | `[3,7,11,15]` |

视频在 [17:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1023s) 逐项解释 rank 0/1 的发送规则，在 [17:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1064s) 从“列”读取每个目标输出。若把输入矩阵记为 $`X`$，输出排布看起来像 $`X^T`$，即矩阵转置。

### 5.2 为什么 MoE 需要它

**MoE（Mixture of Experts，混合专家模型）**包含多个 expert（专家子网络）。Router（路由器）会根据 token 的 activation 动态选择 expert；expert 又可能分布在不同 ranks。

假设：

- 每个 rank 当前各有一批 tokens；
- rank 0 放 expert 0，rank 1 放 expert 1，依此类推；
- 一个 token 被路由到 expert 2，就必须把它的 activation 发往 rank 2。

于是每个 source rank 都可能给每个 expert rank 发不同数量的 tokens，正好是 all-to-all 模式。**【视频补充｜[18:16](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1096s)】**老师把它概括为“每个 rank 同时拿一片数据和一部分 experts，必须按数据动态路由 activations”。

### 5.3 不平衡 split：形状不再是整齐转置

**Split size（分片大小）**是 source 发给某个 destination 的元素/token 数。平衡例里每格都是 1；实际 MoE 可能如下：

| source \ destination | rank 0 | rank 1 | rank 2 | rank 3 | source 总发送 |
|---:|---:|---:|---:|---:|---:|
| rank 0 | 2 | 0 | 1 | 1 | $`2+0+1+1=4`$ |
| rank 1 | 0 | 3 | 1 | 0 | $`0+3+1+0=4`$ |
| rank 2 | 1 | 0 | 2 | 1 | $`1+0+2+1=4`$ |
| rank 3 | 0 | 1 | 0 | 3 | $`0+1+0+3=4`$ |

每个目标收到：

```math
\begin{aligned}
\text{rank 0 receives} &= 2+0+1+0=3,\\
\text{rank 1 receives} &= 0+3+0+1=4,\\
\text{rank 2 receives} &= 1+1+2+0=4,\\
\text{rank 3 receives} &= 1+0+1+3=5.
\end{aligned}
```

总发送 $`4+4+4+4=16`$，总接收 $`3+4+4+5=16`$，数据没有丢；但 rank 3 收 5 个，rank 0 只收 3 个。若所有 rank 进入下一同步点前都要完成 expert 计算，rank 3 可能成为 straggler（拖慢全组的慢参与者）。

**【视频补充｜[18:49](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1129s)】**平衡 split 可看成转置；[19:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1148s) 明确说一般 all-to-all 能处理不同 byte 数，但仍希望尽量均衡。处理 variable splits（可变分片）时，通信双方还要知道每个 split 的大小；这组大小属于 metadata。

### 5.4 一张记忆表

| 名字里的词 | 含义 |
|---|---|
| reduce | 用 SUM/MIN/MAX 等把多个值合并 |
| scatter | 把一份完整逻辑结果拆给不同 ranks |
| gather | 把不同 ranks 的片段拼起来 |
| all- | 最终目的地是所有 ranks，而不只是 root |
| all-to-all | 每个 rank 都可为每个目标 rank 准备一片 |

**【视频补充｜[19:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1184s)】**这是老师在讲完所有 collective 后给出的口头记忆法。

---

## 6. 通信硬件层次与最小 cost model

### 6.1 先把“机器”分层

- **Node（节点）**：一台服务器；通常有 CPU、系统 RAM、若干 GPU 和网卡；
- **Rack（机架）**：把多块服务器/交换设备竖向安装在一起的物理机柜；
- **Tray（托盘/计算托盘）**：插在 rack 中的一块硬件模块，具体装几颗 CPU/GPU 取决于产品；
- **Pod**：把一组 nodes/racks 当成一个部署单元的工程称呼，没有跨厂商统一的固定节点数；
- **Cluster（集群）**：共同运行任务的一组计算节点。

课程的统一图从里到外是：

```text
GPU 内：SM ↔ L1/shared memory ↔ L2 ↔ HBM
                     │
同一 node：多个 GPU ─NVLink→ NVSwitch ─NVLink→ 其他 GPU
                     │
跨 node：GPU ─PCIe→ HCA/NIC ─InfiniBand 或 Ethernet→ 远端
```

**【课程代码｜行 19–35】【视频补充｜[21:53](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1313s)】**老师先从传统服务器拓扑讲起；[22:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1330s) 的旧式外链图就是两台服务器经 Ethernet 相连、每台 GPU 经 PCIe 接 CPU。它是“典型概念图”，不是所有服务器必须有 2 CPU 和 10 GPU。

### 6.2 九个硬件/网络词逐个说成人话

#### HBM

**HBM（High Bandwidth Memory，高带宽内存）**是 GPU 本地的大容量显存，存参数、activations 等。它比 SM 附近的小存储容量大，但从运算单元看更远。

#### NVLink

**NVLink** 是 NVIDIA 的高速 GPU 互连。它负责链路传输；“某代 NVLink 的 GB/s”必须说明是每 GPU 总带宽、每条 link，还是整个系统 aggregate（汇总）带宽，以及是否双向合计。

#### NVSwitch

**NVSwitch** 是连接多条 NVLink 的交换芯片。它像高速路口：GPU 不必只和一个固定邻居直连，switch 可为 GPU 之间转发流量。视频在 [24:28](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1468s) 说明这种“任意 GPU 到另一个 GPU”的编程直觉。

#### PCIe

**PCIe（Peripheral Component Interconnect Express，高速外设互连）**把 GPU、CPU、HCA/NIC 等设备连在服务器内部。跨节点数据常先从 GPU 经 PCIe 到网络适配器；PCIe 不是专门的“互联网线”。

#### HCA 与 NIC

**NIC（Network Interface Card，网络接口卡）**是让服务器接入网络的设备。**HCA（Host Channel Adapter，主机通道适配器）**是 InfiniBand 体系常用的网络适配器称呼；它把主机/GPU 一侧接到 InfiniBand fabric（交换网络）。

#### InfiniBand

**InfiniBand** 是面向高性能计算与数据中心的低延迟、高带宽互连体系。课程现代集群图中，它连接已经超出单个 NVLink domain（NVLink 可互通范围）的多个 nodes。

#### Ethernet

**Ethernet（以太网）**是广泛使用的通用网络技术，从家庭/办公到数据中心都有。不能把视频旧图中的 `~200 MB/s` 当成所有 Ethernet 的上限；现代数据中心 Ethernet 可高得多。

#### RDMA

**RDMA（Remote Direct Memory Access，远程直接内存访问）**是一种能力：一个设备可直接读写远端设备内存，让 CPU 不必在 payload 的数据路径中做多次拷贝。CPU 仍可能负责初始化、控制与错误处理；“绕过 CPU 数据拷贝”不等于“CPU 完全不存在”。

**【视频补充｜[26:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1578s)】**传统 socket 路径会经过 CPU kernel buffer、网络包和 NIC buffer；[27:05](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1625s) 定义 RDMA。这里的 CPU **kernel** 指操作系统核心，不是 GPU kernel。

#### RoCE

**RoCE（RDMA over Converged Ethernet，在融合以太网上运行 RDMA）**让具备相应网卡和网络配置的 Ethernet fabric 提供 RDMA。课程说“standard Ethernet 不支持 RDMA”，准确理解应是：普通 Ethernet/TCP 路径本身不是 RDMA；RoCE 是在 Ethernet 体系上加入 RDMA 能力，并非随便一块以太网卡自动拥有。

**【视频补充｜[28:46](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1726s)】**老师用 RoCE 说明 Ethernet 也在演进。课堂问答 [32:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1942s) 又区分：RDMA 是想实现的“直接访问能力”，NVLink、InfiniBand、RoCE 是提供这种能力的不同硬件/协议路径。

### 6.3 层级为什么重要：慢链路会成为共同等待点

**【视频补充｜[23:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1401s)】**课程现代图以常见 8-GPU node 为例；[25:14](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1514s) 解释跨更多 nodes 时需要离开 NVLink domain，走 PCIe 与 InfiniBand，速度通常低一个层级。

如果一个 all-reduce 同时覆盖 8 个 node，每个 node 内部链路再快，也可能被节点间最慢/最拥塞的路径限制。通信库常用分层算法：

1. node 内先聚合；
2. 各 node 代表之间跨网络聚合；
3. node 内再分发。

这只是常见策略，不是所有 collective 的唯一物理实现。

### 6.4 课程数字与当前官方资料的边界

先定义单位：小写 `b` 是 bit（位），大写 `B` 是 byte（字节），$`8\ \text{bits}=1\ \text{byte}`$。厂商网络/显存表常用十进制：$`1\ \text{GB}=10^9`$ bytes，$`1\ \text{TB}=10^{12}`$ bytes。若写 `GiB`，才是 $`2^{30}`$ bytes。

**【课程时点快照｜代码行 209–230】【视频补充｜[23:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1415s)】**课程给出 B200 的 NVLink 5 约 1.8 TB/s、HBM 约 8 TB/s；[24:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1443s) 用 $`8/1.8\approx4.44`$ 说明跨 GPU 仍比本地 HBM 慢约 4 倍。这个比值只作数量级直觉：两项的方向、聚合与实际可达口径必须一致才适合严谨比较。

下面把课程型号放进一张**课程时点 + 官方产品规格边界表**。不同 form factor（外形/封装）可能有不同规格。**SXM** 是 NVIDIA 数据中心 GPU 使用的一类高带宽模块/封装形态，不等于同型号的 PCIe 插卡版；**HBM2e、HBM3、HBM3e** 是不同代际/版本的 HBM：

| 型号/系统 | HBM 容量与带宽 | NVLink/系统互连 | 口径与一手来源 |
|---|---|---|---|
| A100 80GB SXM | 80 GB HBM2e；2,039 GB/s | 600 GB/s | NVIDIA [A100 官方规格](https://www.nvidia.com/en-us/data-center/a100/)；每 GPU 产品规格 |
| H100 SXM | 80 GB HBM3；3.35 TB/s | 900 GB/s | NVIDIA [H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)；每 GPU SXM 规格 |
| B200 SXM | 180 GB HBM3e；最高 8 TB/s | DGX B200 八 GPU 总 NVLink 14.4 TB/s，即 $`14.4/8=1.8`$ TB/s/GPU 的规格口径 | NVIDIA [HGX 组件表](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html) 与 [DGX B200 用户指南](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) |
| GB200 NVL72 | 72 个 Blackwell GPUs、36 个 Grace CPUs | 每 GPU 1.8 TB/s；72-GPU fabric 汇总 130 TB/s | NVIDIA [GB200 NVL72 调优指南](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/overview.html) |

为什么 $`72\times1.8=129.6`$，官方表写 130 TB/s？因为产品页把 129.6 四舍五入为 130。

课程代码第 229 行/视频 [27:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1668s) 把 NVL72 简化成“每 tray 8 GPU、9 trays 共 72 GPU”。当前 NVIDIA 架构文档写的是 **18 个 compute trays + 9 个 NVLink switch trays**，每个 compute tray 含 4 GPU。应把课堂表述当作“72 GPU 同一 NVLink domain”的教学简图，不能当机械结构说明。

课程代码还给出 PCIe 7.0 ×16 `242 GB/s`、传统跨节点 Ethernet `~200 MB/s`、InfiniBand `~0.05 TB/s` 等数。它们混合了不同代际、方向/编码与教学场景；不能按一张永久排名表直接比较。一个可复算转换是：

```math
400\ \text{Gb/s}\div8=50\ \text{GB/s}=0.05\ \text{TB/s}
```

这是忽略协议开销的理论换算。真实应用带宽通常更低。

### 6.5 NCCL 怎样把逻辑 collective 映射到硬件

**【课程代码｜行 232–243】【视频补充｜[29:47](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1787s)】**NCCL 做三类工作：

1. 探测 topology，例如哪些 GPU 同一 NVSwitch、哪些路径要过 PCIe/HCA；
2. 选择数据经过哪些 peers（通信伙伴）以及采用哪种 collective algorithm；
3. 启动 GPU communication kernels 完成 send/receive/reduce。

课堂问答 [33:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2013s) 询问 NCCL 是否优化多节点；老师没有声称知道所有内部细节，只说明 NVIDIA 会针对大模型训练栈持续优化。笔记也不把“自动选择”写成“任何拓扑都一定最优”。

另一个问答 [34:32](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2072s) 用“8-GPU node 外多出第 9 张卡”说明 topology 的离散边界：若第 9 张卡落到另一个 node 且没有同级高速互连，它增加的少量计算可能抵不过昂贵通信。

### 6.6 只会四则运算也能用的 cost model

通信的最小估算式：

```math
T\approx sL+\frac{Q}{B}.
```

逐符号解释：

- $`T`$：估计总时间，单位 seconds（秒）；
- $`s`$：串行依赖的通信 rounds/steps（轮数），无单位；
- $`L`$：每轮固定 latency（延迟），单位 seconds/step；
- $`Q`$：关键路径上需要传输的 bytes；
- $`B`$：可用 bandwidth（带宽），单位 bytes/second；
- $`Q/B`$ 的单位是 $`\text{bytes}/(\text{bytes/second})=\text{seconds}`$，可以与 $`sL`$ 相加。

#### 例 1：小消息被 latency 支配

已知：$`s=4`$ 轮，每轮 $`L=10\ \mu s`$，$`Q=100`$ bytes，$`B=1\ \text{GB/s}=10^9`$ bytes/s。$`\mu s`$ 是 microsecond（微秒），$`1\ \mu s=10^{-6}`$ 秒。

固定等待：

```math
sL=4\times10\ \mu s=40\ \mu s.
```

纯搬运：

```math
\frac{Q}{B}
=\frac{100}{1,000,000,000}\ \text{s}
=0.0000001\ \text{s}
=0.1\ \mu s.
```

总计：

```math
T\approx40+0.1=40.1\ \mu s.
```

$`40\ \mu s`$ 远大于 $`0.1\ \mu s`$，所以增加带宽几乎帮不到这个小消息；减少轮数/固定延迟更重要。

#### 例 2：大消息被 bandwidth 支配

保持 $`s=4,L=10\ \mu s,B=1\ \text{GB/s}`$，改为 $`Q=10\ \text{MB}=10,000,000`$ bytes。

```math
\frac{Q}{B}
=\frac{10,000,000}{1,000,000,000}\ \text{s}
=0.01\ \text{s}=10\ \text{ms}.
```

`ms` 是 millisecond（毫秒），$`1\ \text{ms}=1,000\ \mu s`$，所以 $`40\ \mu s=0.04\ \text{ms}`$：

```math
T\approx0.04\ \text{ms}+10\ \text{ms}=10.04\ \text{ms}.
```

此时搬运项是 10 ms，固定延迟只有 0.04 ms；提高有效带宽更有意义。

这个式子是教学近似。真实 collective 还会受双向链路、contention（多个流量争同一链路）、分块流水线、协议开销、GPU kernel 启动与计算/通信 overlap 影响。

### 6.7 Payload、per-rank bytes 与 aggregate traffic

三个口径必须写全：

- **payload size**：应用眼中一个完整逻辑 tensor 有多大；
- **per-rank bytes**：某个 rank 在具体算法中发/收多少；
- **aggregate network traffic**：所有 ranks 的发送量相加，或全网络链路上的总流量；必须说明是否把 receive 再算一次。

**【补充例子：ring all-reduce 的一种理想化算法账】**令 payload $`S=16`$ MiB，$`p=4`$ ranks。Ring 的 reduce-scatter 阶段，每 rank 发送：

```math
\frac{p-1}{p}S
=\frac{4-1}{4}\times16
=\frac34\times16
=12\ \text{MiB}.
```

All-gather 再发送 12 MiB，所以每 rank 总**发送**：

```math
12+12=24\ \text{MiB}.
```

所有 4 ranks 的 aggregate sends：

```math
4\times24=96\ \text{MiB}.
```

若还把每 rank 收到的 24 MiB 再算一次“send + receive traffic”，就是：

```math
4\times(24+24)=192\ \text{MiB}.
```

同一个操作可以出现 16、24、96、192 MiB 四个数，因为它们分别回答不同问题；不写口径就无法比较。

这个 $`24`$ MiB/rank 是 **ring、均匀分块、忽略协议开销**的结果。Tree 或分层算法的 steps、每轮消息大小、经过的物理链路都可能不同。课程 benchmark 在 [48:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2910s) 计算的是一种 normalized effective bandwidth（归一化有效带宽）；它不是“把机房所有物理线缆上的 bytes 全部抓包相加”。

### 6.8 Benchmark 只能验证具体机器，不能把模型变成定律

**Benchmark（基准测试）**是在明确输入与环境下测时间/吞吐。课程代码对 $`100\times1024^2`$ 个元素做 all-reduce、reduce-scatter，并在计时前 warmup（预热）。它的结构意图是用 CUDA synchronize 等本 rank GPU，再用 barrier 等齐 processes；但当前 setup 未绑定 current device，所以无参 synchronize 可能等错 device，详见 §7.3 与 §9.3。

**【视频补充｜[46:38](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2798s)】**老师进入通信 benchmark；[47:23](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2843s) 解释 warmup 和两层异步：GPU kernels 可能尚未完成，各 processes 也可能进度不同。若不等齐，CPU 计时可能只量到“发出命令”。

有效带宽的一般单位检查：

```math
\text{effective bandwidth}
=\frac{\text{algorithmic bytes}}{\text{measured seconds}}
\quad[\text{bytes/s}].
```

视频 [50:40](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3040s) 讨论课程 all-reduce 归一化口径；[51:40](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3100s) 再对 reduce-scatter 复用相同测量结构。这里观测到的“约 400 GB/s”等值属于老师当时的机器/进程/消息大小，不应复制成另一台机器的保证值。

课堂最后一个相关问答 [53:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3200s) 追问为什么要同步 CUDA：Python 执行到下一行时 GPU 操作可能尚未结束；先让 GPU 工作完成，再让所有 process 到 barrier，才能把计时区间定义清楚。

### 6.9 本阶段结束时应能复述的主线

```text
先问：是单卡放不下，还是希望更快？
  ↓
选择复制 / 切分 / 重算 / 通信的组合
  ↓
每个 process 在 group 中由 rank 标识，backend 实现 collective
  ↓
用输入/输出语义选择 broadcast/scatter/gather/reduce/
all-gather/reduce-scatter/all-reduce/all-to-all
  ↓
再看逻辑通信落在哪一层硬件：NVLink、PCIe、InfiniBand、Ethernet
  ↓
用 steps×latency + bytes/bandwidth 做第一遍估算
  ↓
最后在具体 topology、消息大小和实现上 benchmark
```

## 7. NCCL 与 `torch.distributed`：谁组织 ranks，谁搬 GPU 数据

### 7.1 先分清三层，不要把名字混成一团

**【课程内容｜源码 209–249】【视频补充｜[30:05](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1805s)】**前面已经知道“all-reduce 要把各 rank 的值加起来”。现在要回答：代码究竟交给谁执行？

从上到下看三层：

1. **PyTorch `torch.distributed`**：给 Python 程序统一的分布式 API（应用程序接口）。它提供 `broadcast`、`all_reduce`、`barrier` 等函数。
2. **process group（进程组）**：告诉 API “本次有哪些 ranks 参加、每个 rank 是谁、用哪个 backend”。**Backend（后端）**是实际承接通信工作的实现。
3. **NCCL**：NVIDIA Collective Communications Library，即 **NVIDIA 集合通信库**。当 tensor 在 NVIDIA GPU 上时，NCCL 可以选择 topology-aware（感知连接拓扑）的路径和算法，并启动 GPU 通信 kernel。

这里的 **kernel** 是“在 GPU 上执行的底层程序”，不是操作系统 kernel。NCCL 不负责定义你的模型，也不替你决定“梯度什么时候该 all-reduce”；它接到 collective 请求后，负责更底层的传输编排。

可以把职责画成：

```text
训练代码：现在同步梯度
    ↓ 调用
torch.distributed.all_reduce(gradient)
    ↓ process group 指定 ranks + backend
NCCL：根据可见的 GPU / NVLink / PCIe / 网络拓扑选通信路径和算法
    ↓ 启动一个或多个 GPU 通信 kernels
各 GPU tensor 得到 collective 的逻辑结果
```

**Topology（拓扑）**就是“设备和链路怎样连接”的地图。**Path（路径）**是数据实际经过哪些链路。NCCL 可能依据拓扑选择 ring、tree 或分层方案；所以“API 叫 all-reduce”不等于“物理上永远沿一个 ring 走”。视频 [30:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1822s) 开始说明 NCCL，[30:43](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1843s) 强调它会把 collective 变成对当前硬件更合适的实现。

### 7.2 Gloo 和 NCCL：本讲最小选择规则

**【课程内容｜源码 540–548】**课程的 `setup` 用：

```python
if torch.cuda.is_available():
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
else:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

逐行翻成人话：

- 有 CUDA GPU：选择 `nccl` backend，当前 rank 主要通信 GPU tensor；
- 没有 CUDA GPU：选择 `gloo` backend，让教学代码也能在 CPU 上演示 collective 语义；
- `rank=rank`：把这个 process 的全局编号交给 group；
- `world_size=world_size`：告诉 group 总共有多少个参与者。

**Gloo** 是 PyTorch 可用的 collective 通信后端之一，本讲把它当 CPU 情形的首选；**NCCL** 是 CUDA GPU 情形的首选。这是 PyTorch 当前文档给出的常用规则，不是“Gloo 永远不能碰 GPU”或“NCCL 可通信任意 CPU tensor”的反向定律。

**【课程基础设施｜源码 16–17】**无 CUDA 时，讲义还执行：

```python
torch.cuda.synchronize = lambda: None
```

也就是把 `torch.cuda.synchronize()` 临时替成什么都不做的 **no-op**。目的只是让同一份教学代码在 CPU/Gloo 路径走到 benchmark，不因调用 CUDA API 报错；副作用是这条 CPU 路径根本没有测 CUDA device completion，所得时间不能冒充 GPU/NCCL benchmark。这个 lambda 也只接受无参数调用，因此若采用后文“显式传 device”的安全改法，CPU 分支应使用单独 helper 或条件判断，不能直接照抄。

**版本边界：**本节于 2026-08-28 复核 [PyTorch stable distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html) 与 [NCCL 2.31.2 collective 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。库的 API、可用 backend 和同步细节可能随版本变化；这里把课程代码和该时点官方文档的语义分开说明。

### 7.3 `MASTER_ADDR` 与 `MASTER_PORT`：先让陌生 processes 找到彼此

**【课程代码｜源码 538–549】【视频补充｜[30:56](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1856s)】**课程 setup 的完整骨架是：

```python
def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

- `MASTER_ADDR`：负责 rendezvous（会合）的主地址。课程写 `localhost`，意思是“所有 processes 都在同一台电脑上”。
- `MASTER_PORT`：该地址上用于会合的端口号。端口可以先理解成同一台机器上的“门牌号”。
- `init_process_group(...)`：建立默认 process group；之后未显式传 `group=` 的 collective 就在这个默认组中执行。

**关键事实：当前 619 行源码没有调用 `torch.cuda.set_device(rank)`。**`cuda_if_available(rank)` 只为新 tensor 返回类似 `torch.device("cuda:1")` 的显式 device；它**不会**改变该 process 的 current CUDA device（当前默认 CUDA 设备）。然而 benchmark 使用无参数 `torch.cuda.synchronize()`，它同步 current device。若 rank 1–3 的 current device 仍是默认 `cuda:0`，就可能等待错设备，正式计时可能在本 rank 通信尚未完成时继续。这是课程代码的真实风险，不是推荐模式。

**【安全改良写法，不是课程原码】**单机一 process 一 GPU 时，可在初始化早期绑定当前设备：

```python
if torch.cuda.is_available():
    torch.cuda.set_device(rank)  # 多机应传 local_rank，不是 global rank
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

也可以在有 CUDA 时明确写 `torch.cuda.synchronize(device=cuda_if_available(rank))`。实际多机应使用 local rank；global rank 13 不能机械变成 `cuda:13`。`MASTER_ADDR="localhost"` 也只适合同机演示，多机必须用其他 nodes 能访问的地址。

### 7.4 `destroy_process_group`：正常拆掉通信上下文

**【课程代码｜源码 551–553】**

```python
def cleanup():
    dist.destroy_process_group()
```

`destroy_process_group()` 会拆除当前 process 使用的默认通信组，让 NCCL 等资源能按正常次序释放。它不是“清空 GPU tensor”，也不是“把所有 rank 的 Python process 一起杀掉”。

正确生命周期是：

```text
每个 rank 启动
  → setup/init_process_group
  → 所有 ranks 以匹配顺序调用 collectives
  → cleanup/destroy_process_group
  → process 返回
```

若某个 rank 提前异常退出，其他 rank 还在等 collective，就不能假设 cleanup 会神奇修复次序不匹配。

### 7.5 `mp.spawn`：同一个函数，给每个 child 一个 rank

**【课程代码｜源码 577–593】【视频补充｜[31:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1864s)】**`mp` 是 `torch.multiprocessing`。课程包装器的真实多进程分支可缩成：

```python
mp.spawn(
    fn,
    args=(world_size, *args),
    nprocs=world_size,
    join=True,
)
```

假设 `world_size=4`，且 `fn` 写成：

```python
def fn(rank, world_size, message):
    ...
```

那么 `spawn` 让四个 child processes 分别调用：

```text
fn(0, 4, message)
fn(1, 4, message)
fn(2, 4, message)
fn(3, 4, message)
```

`nprocs=4` 是 child 数；`join=True` 表示父 process 在这里等待 children 结束，并接收异常。官方文档还要求被 spawn 的函数能够被 pickle（序列化给 child）且定义在模块顶层。详见 [PyTorch multiprocessing 官方文档](https://docs.pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses)。

### 7.6 为什么讲义 trace 时故意“不真的分布式”

**【课程基础设施｜源码 557–593】**本讲义既要能真实运行，也要能生成课堂网页 trace。调试器/trace 工具通常不适合在四个 child processes 中同时抓取展示。因此课程写了 `DisableDistributed` context manager（上下文管理器）：进入时暂时把若干 `torch.distributed` 函数换成 no-op（不做事的函数），退出时再恢复。

源码逻辑是：

```text
若 sys.gettrace() 为假：
    正常 mp.spawn，真实启动 world_size 个 processes
若 sys.gettrace() 为真：
    只调用 fn(rank=0, world_size=原请求值, ...)
    并用 DisableDistributed 暂停 setup/barrier/cleanup 等动作
```

例如请求 `world_size=4` 时，trace 分支实际调用 `fn(rank=0, world_size=4, ...)`，不是把 world size 改成 1；只是所有 `torch.distributed` 函数被 no-op。于是 shape/index 代码仍按 4 ranks 的配置走，但没有另三个 processes，也没有真实通信。这个 trace 只能展示控制流，不能作为分布式数值正确性证据。

这解释了两个看似冲突的现象：

- 页面上的 trace 可以只显示一个顺序执行流；
- 讲义的已保存 stdout 又能展示 4 ranks 的真实 collective 结果。

`DisableDistributed` 是**讲义展示基础设施**，不是训练分布式模型时应该照抄的模式。若在真实训练中把 collective 换成 no-op，各 rank 会各算各的，模型不会同步。

### 7.7 NCCL 在 API 下面可能做什么

**【视频补充｜[36:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2180s)–[38:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2301s)】**课堂从硬件层回到 `torch.distributed`：程序提交一个 collective 后，NCCL 的工作包括但不限于：

- 探测或接收 GPU、NVLink、PCIe、NIC 等 topology 信息；
- 选择 ring、tree、分层或其他可用算法/协议；
- 把大 tensor 分块、建立流水；
- 选择可行的 peer-to-peer path；
- 启动通信 kernels，并与 CUDA stream 的顺序规则协作。

**Profiler（性能分析器）**是记录 kernels、通信、CPU gaps 和时间线的分析工具。这里只能从官方语义和 profiler/NCCL 日志证据判断某次运行采用了什么，不能从 Python 函数名反推“必然是一条环”。

---

## 8. `collective_operations_main`：从 setup 到三个 collective 逐行读

### 8.1 函数骨架与最初的 barrier

**【课程代码｜源码 249–284】【视频补充｜[39:52](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2392s)】**函数入口：

```python
def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)

    # All-Reduce
    dist.barrier()
    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank
    ...
```

逐行：

1. `setup(rank, world_size)`：只让这个 process 加入默认 process group；它没有绑定 CUDA current device。
2. `dist.barrier()`：**barrier（屏障）**要求 group 里的所有 ranks 都到达这里，才允许大家继续。这次 barrier 在 all-reduce 示例开始前把进度对齐。
3. 下一行 tensor 构造中的 `device=cuda_if_available(rank)` 才选择存放 device：有 CUDA 时返回该 rank 的 GPU，否则返回 CPU。它只决定这个新 tensor 放在哪里，仍不会改变 current device。
4. 基础浮点 tensor 是 `[0.,1,2,3]`；`+ rank` 对四个元素都加同一个 rank。小数点表明课程原码在这里使用浮点值。

四 rank 的创建结果：

| rank | `[0,1,2,3] + rank` |
|---:|---|
| 0 | `[0,1,2,3]` |
| 1 | `[1,2,3,4]` |
| 2 | `[2,3,4,5]` |
| 3 | `[3,4,5,6]` |

### 8.2 `all_reduce` 是 in-place：同一个 tensor 被改掉

**【课程代码｜源码 256–260】【视频补充｜[40:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2420s)】**

```python
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
```

- `data`：输入 tensor，也是输出落回的 tensor；这叫 **in-place（原地）**修改。
- `op=SUM`：逐位置相加，不是拼接。
- `async_op=False`：调用没有返回供用户稍后等待的异步 `Work` handle；同步边界还要结合 backend 和 CUDA stream 语义理解，见 §8.6。

四个位置逐列加：

```math
0+1+2+3=6,
```

```math
1+2+3+4=10,
```

```math
2+3+4+5=14,
```

```math
3+4+5+6=18.
```

所以 `all_reduce` 返回后，每个 rank 的**同一个 `data` 变量**都变为：

```text
[6, 10, 14, 18]
```

视频 [40:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2430s) 强调它没有创建另一个输出参数；[40:51](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2451s) 展示所有 rank 得到同一完整结果。

### 8.3 `reduce_scatter_tensor`：输入 4 个位置，输出 1 个位置

**【课程代码｜源码 262–270】【视频补充｜[41:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2468s)】**课程接着重新建 tensor：

```python
dist.barrier()
input = torch.arange(
    world_size,
    dtype=torch.float32,
    device=cuda_if_available(rank),
) + rank
output = torch.empty(1, device=cuda_if_available(rank))
dist.reduce_scatter_tensor(
    output=output,
    input=input,
    op=dist.ReduceOp.SUM,
    async_op=False,
)
```

上面是**当前 619 行课程原码**的关键参数及关键字参数。`input` 显式指定 `dtype=torch.float32`；在本讲文件没有修改默认 dtype 的前提下，`torch.empty(1, ...)` 也是 FP32，所以当前这段课程代码的 input/output dtype 一致。

**【通用防御性写法，不是课程原码，也不是在修当前课程 bug】**如果代码在其他文件中更改了默认 dtype，可显式继承 input dtype：

```python
output = torch.empty(1, device=cuda_if_available(rank), dtype=input.dtype)
```

当 `world_size=4`：

- `torch.arange(4)` 是 `[0,1,2,3]`；
- 每 rank 加自己的编号，四个 `input` 仍是上表四行；
- 每个 `input.shape == [4]`；
- 每个 `output.shape == [1]`；
- 输入元素数 4 正好是输出元素数 1 的 `world_size=4` 倍。

下面就是当前 FP32 课程代码的**预期逻辑语义**。先 reduce 得完整逻辑和 `[6,10,14,18]`，再按 rank 取一块：

| output 所在 rank | 属于该 rank 的位置 | `output.shape` | output 值 |
|---:|---:|---:|---:|
| 0 | 0 | `[1]` | `[6]` |
| 1 | 1 | `[1]` | `[10]` |
| 2 | 2 | `[1]` | `[14]` |
| 3 | 3 | `[1]` | `[18]` |

视频 [41:40](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2500s) 开始追踪输入形状，[42:26](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2546s) 对照每个 rank 最终只持有一个 shard。

### 8.4 `all_gather_into_tensor`：把四个 `[1]` 按 rank 顺序拼回 `[4]`

**【课程代码｜源码 272–280】【视频补充｜[42:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2564s)】**

```python
dist.barrier()
input = output
output = torch.empty(world_size, device=cuda_if_available(rank))
dist.all_gather_into_tensor(
    output_tensor=output,
    input_tensor=input,
    async_op=False,
)
```

这同样保留了课程原码的关键字参数。由于此时 `input` 是前一步的 FP32 `output`，新的 `torch.empty` 在本文件默认 dtype 未改动时也是 FP32。通用库代码仍可采用下列**防御性写法**，但它不是在修复当前课程 bug：

```python
output = torch.empty(world_size, device=cuda_if_available(rank), dtype=input.dtype)
```

注意变量名发生了两次重新绑定：

1. `input = output`：等号右边还是 reduce-scatter 产生的 `[1]` tensor；把这个对象另取名为 `input`。
2. 新的 `output = torch.empty(4, ...)`：再创建一个形状 `[4]` 的接收 tensor；这不会把上一行的 `input` 变成 `[4]`。

输入表：

| rank | 本 rank 的 `input.shape` | 本 rank 的 input |
|---:|---:|---:|
| 0 | `[1]` | `[6]` |
| 1 | `[1]` | `[10]` |
| 2 | `[1]` | `[14]` |
| 3 | `[1]` | `[18]` |

all-gather 按 rank 0、1、2、3 的顺序拼接，于是每个 rank 的新 `output` 都是：

```text
[6, 10, 14, 18]    shape = [4]
```

因此在逻辑结果上，本例验证了：

```text
all-reduce
    与
reduce-scatter → all-gather
```

都使每个 rank 最终得到 `[6,10,14,18]`。视频 [43:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2584s) 开始 all-gather，[43:36](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2616s) 展示重新拼回完整向量。

### 8.5 函数末尾为什么 cleanup

**【课程代码｜源码 282–284】**最后一次打印后调用 `cleanup()`。这表示：三个示例全完成后，当前 rank 才销毁默认 process group。不能让 rank 0 在 all-gather 前 cleanup，而 rank 1 还准备进入 all-gather。

### 8.6 `async_op=False` 究竟保证到哪一步

这里最容易说得过头，所以分三层。回忆：CUDA stream 是 GPU 有序命令队列；Work handle 是异步 collective 的可等待凭证。

1. **Python API 层：**对 NCCL/CUDA，`async_op=False` 会等到 collective 已成功排入 CUDA stream，再返回且不提供 Work handle；GPU 物理执行未必结束。
2. **CUDA stream 层：**随后在同一 stream 排入的依赖操作，会因该 stream 的顺序而排在 collective 后。换到另一个 stream 时，没有这条天然顺序，必须显式建立依赖。
3. **CPU wall-clock 测量层：**若要让 CPU 计时器确认指定设备此前提交的 CUDA 工作完成，要同步正确 device；课程无参 `torch.cuda.synchronize()` 有 §7.3 所述 current-device 风险。

三条最小时间线：

```text
A. async_op=False
CPU: 调用 collective ──等“成功排入stream”──返回──继续Python
GPU同stream:          collective执行 ───────→ 后续依赖kernel
结论：同stream有顺序；CPU返回不等于GPU已经做完。
```

```text
B. async_op=True
work = collective(..., async_op=True)  # 得到Work handle
做不依赖通信结果的独立计算
work.wait()                            # 用官方等待机制建立完成/stream依赖
使用通信结果
```

`work.wait()` 的精确 CPU/stream 阻塞行为依 backend 与当前 PyTorch 版本；对 NCCL，官方文档重点是它为后续使用建立正确同步语义。若要测 CPU 看到的 device 完成时间，仍应同步正确 device/stream，而不是把 `wait()` 名字扩写成“全机所有 GPU 都停住”。

```text
C. same stream 与 different stream
same stream:      collective → dependent kernel       （队列顺序足够）
different stream: collective ─┐
                              ├─ 必须用Work/事件/stream等待显式连依赖
                 consumer ────┘
```

所以不能把 `async_op=False` 读成“GPU 全部工作和所有 ranks 的 Python 时钟都已经在同一纳秒停住”。也不能反过来说“函数一返回结果完全不能用”；同 stream 有明确的执行顺序。官方边界见 [PyTorch distributed 同步/异步说明](https://docs.pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations) 与 [`torch.cuda.synchronize` 文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。

### 8.7 `barrier` 等谁，`cuda.synchronize` 又等谁

| 调用 | 主要等待对象 | 它不自动等于什么 |
|---|---|---|
| `dist.barrier()` | process group 中其他 ranks 到达匹配的 barrier | 不是“给所有业务 CUDA kernels 做全设备计时同步”的通用替身 |
| `torch.cuda.synchronize()` | 无参数时等 current CUDA device 上所有 streams 中此前提交的 kernels | 不会让其他 ranks 的 Python 进度自动对齐；课程未绑定 current device，见 §7.3 |

当前 ProcessGroupNCCL 文档还说明：NCCL backend 的 barrier 通过一个 1-element tensor 的 all-reduce 实现，并阻塞 CPU thread 到 barrier 完成。但这个实现细节仍不应被扩写成“barrier 等价于任意设备工作全同步”。课程 benchmark 把两者都写上，正是为了分别处理“本 rank 的 GPU 是否结束”和“所有 ranks 是否到齐”。

### 8.8 Collective 顺序不一致会怎样：最小 deadlock 例

**Deadlock（死锁）**是参与者互相等待，程序无法向前。下面是**故意错误、不要运行**的伪代码：

```python
# 错误伪代码：不同 rank 进入了不匹配的 collective 次序
if rank == 0:
    dist.all_reduce(x)   # rank 0 等 rank 1 一起 all-reduce
else:
    dist.barrier()       # rank 1 却在等 rank 0 一起 barrier
```

rank 0 说：“等 rank 1 来 all-reduce。”rank 1 说：“等 rank 0 来 barrier。”两边都不会抵达对方在等的调用。

NCCL 官方文档要求 collective 中所有 ranks 使用匹配的 count 与 datatype，并以一致次序参加；违反时可能 hang（卡住）、crash 或得到错误数据。**Count** 是本次 collective 约定的元素个数，**datatype/dtype** 是每个元素的数字存储格式，例如 FP32。视频 [44:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2641s) 到 [45:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2708s) 的三段输出，成立的前提正是四个 processes 都按同一顺序执行 barrier→collective。

---

## 9. 通信 benchmark：到底从哪里开始计时，到哪里停止

### 9.1 先定义六个常被混用的量

**【课程内容｜源码 287–374】【视频补充｜[46:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2804s)】**Benchmark 是对一个明确输入、环境和计时边界做测量。读打印值之前，先问它的分子与分母。

1. **Latency（延迟）**：完成一次操作花多少时间，单位常为 s、ms、µs。$`1\text{ ms}=0.001\text{ s}`$。
2. **Algorithm bandwidth，记作 `algbw`（算法带宽）**：逻辑数据大小除以操作时间。它回答“应用眼中的 payload 以多快速度完成”。
3. **Bus bandwidth，记作 `busbw`（总线归一化带宽）**：给 `algbw` 乘 collective 特定的校正因子，试图让不同 collective 更容易横向比较。它是**归一化指标**，不是抓包得到的每根线缆字节和。
4. **Per-rank send bytes**：一个 rank 在指定算法模型下发送多少 bytes。
5. **Aggregate sends**：所有 ranks 的发送量相加。
6. **Send + receive traffic**：aggregate sends 再加所有接收量。若发送者的 1 byte 和接收者的同一 1 byte 两端各计一次，这会是 aggregate sends 的 2 倍。

先记单位：

```math
1\ \text{byte}=8\ \text{bits},
```

```math
1\ \text{MiB}=2^{20}=1{,}048{,}576\ \text{bytes},
```

```math
1\ \text{GiB}=2^{30}=1{,}073{,}741{,}824\ \text{bytes}.
```

十进制 `GB` 则是 $`10^9=1{,}000{,}000{,}000`$ bytes；`GiB` 和 `GB` 数字不同。

### 9.2 课程消息为什么恰好是 400 MiB

**【课程代码｜源码 289–295】**主程序调用：

```python
spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)
spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)
```

上面是当前源码两行；真实函数名就是 `all_reduce` 与 `reduce_scatter`，两者都收到相同的 `num_elements` 数值，但 reduce-scatter 还会在函数内部创建 leading dimension=`world_size` 的完整 input。

`1024**2` 是 $`1024^2`$。逐步算：

```math
1024^2=1024\times1024=1{,}048{,}576,
```

```math
100\times1{,}048{,}576=104{,}857{,}600\ \text{elements}.
```

课程 `torch.randn` 默认生成 FP32；**FP32（32-bit floating point，32 位浮点）**每元素 32 bits，即：

```math
32\div8=4\ \text{bytes/element}.
```

因此一条长度 `num_elements` 的 tensor 是：

```math
104{,}857{,}600\times4
=419{,}430{,}400\ \text{bytes}.
```

换成 MiB：

```math
419{,}430{,}400\div1{,}048{,}576=400\ \text{MiB}.
```

这就是后文 all-reduce payload $`S=400`$ MiB，也是 reduce-scatter 每 rank 的 output chunk $`C=400`$ MiB。两段 benchmark 虽传入同一个 `num_elements`，reduce-scatter 的 **input** 还多了 `world_size` 这一维，不能把两者输入 tensor 大小混为一谈。

### 9.3 All-reduce 的 warmup 与测量区间逐行

**【课程代码｜源码 301–321】【视频补充｜[46:55](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2815s)】**课程骨架：

```python
setup(rank, world_size)
data = torch.randn(num_elements, device=cuda_if_available(rank))

# warmup
dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()
dist.barrier()

# measured interval
start_time = time.time()
dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()
dist.barrier()
end_time = time.time()

duration = end_time - start_time
```

逐行解释：

- `setup`：建立 group；不计入测量。
- `randn`：分配并填充输入；也不计入测量。
- 第一次 `all_reduce` 是 **warmup（预热）**。它让 lazy initialization（第一次才发生的初始化）、连接建立、kernel/module 加载等冷启动成本不全部塞进正式样本。
- 第一个 `torch.cuda.synchronize()`：课程意图是等当前 rank 的 GPU 工作完成；但原码无参数且未 `set_device`，实际只等 current device，rank 1–3 可能等错，见 §7.3。
- 第一个 `dist.barrier()`：等所有 ranks 结束 warmup、到达同一起跑线。
- `start_time = time.time()`：CPU 读取墙上时钟；**wall-clock time（墙钟时间）**就是用户从现实钟表看到经过多久。
- 第二次 `all_reduce`：正式测量的 collective。
- 第二个 `cuda.synchronize()`：课程的设计意图是让 CPU 等本 rank GPU 干完；但因 setup 未 `set_device`、此调用又没有 device 参数，它实际可能同步错 current device。
- 第二个 `barrier()`：让先完成的 rank 等最慢 rank 到齐。
- `end_time` 在这个末尾 barrier 之后，所以 `duration` 包含 collective 提交/执行、本 rank 设备等待、以及末 barrier 里的 straggler 等待。

因此这段区间的**设计意图**是“collective + 正确 device completion + 末 barrier”，但当前源码缺少 device binding，不能无条件声称实际测量已经实现这个意图。推荐在 CUDA 分支初始化时设 local device，或对 synchronize 显式传本 rank device 后再测。

**Straggler（拖后腿者）**是这一轮完成得最慢的 rank。若 rank 0 的 GPU 先完成，但 rank 3 晚 2 ms 才到 barrier，rank 0 的计时包含这 2 ms 等待。

视频 [47:17](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2837s) 说明先预热，[47:28](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2848s) 到 [48:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2881s) 逐项解释 CUDA synchronize 与 barrier。

### 9.4 这个计时不是“纯 NCCL kernel 时间”

因为课程用 CPU `time.time()` 包围 Python 调用、设备同步和末 barrier，测到的是这段代码的 wall-clock 区间。它**不是**只量 profiler 中某一个 NCCL kernel 的 device event 时间，也不是完整训练 step 的端到端时间。

它包含：

- Python 进入 collective API 的一小段 host 开销；
- backend 提交与执行 collective；
- 本 rank 的 `torch.cuda.synchronize()` 等待；
- 最慢 rank 导致的末 barrier 等待。

它不包含：

- `setup`、输入分配与随机数生成；
- warmup collective 本身；
- `end_time` 之后的打印和 cleanup。

这个 benchmark 的**结构意图是合理的**：它想用带同步的墙钟区间量“collective 与最慢 rank 到齐”。但当前代码的 **device binding 有缺陷**：无参 `synchronize()` 可能等错 current device，所以修好绑定前不能把所得时间当成已正确实现这个口径。若问题是“纯 GPU kernel 多久”，应使用 CUDA events 或 profiler；若问题是“训练程序从调用到所有 rank 都完成要多久”，在绑定正确后，这种带同步的墙钟测量更接近目标。视频 [48:19](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2899s) 显示不同 ranks 的测量值略有不同，也说明它不是一条脱离进程进度的理想常数。

### 9.5 一次 warmup、一次样本为什么还不够稳健

**【补充理解】**课程代码适合说明原理，但正式性能报告通常还要：

1. 多次 warmup，直到连接与缓存状态稳定；
2. 重复测很多次，不只取一个样本；
3. 报 median（中位数）和 p95，而不只报 mean（平均数）；
4. 记录 GPU 型号、节点数、topology、NCCL/PyTorch/CUDA 版本、dtype、消息大小；
5. 确认没有其他作业抢占链路或制造 contention；
6. 分开报告每个 rank，或明确用最大 duration 代表全局完成时间。

**Median（中位数）**是把测量从小到大排后位于中间的值。**p95** 是约 95% 样本不超过的值，可暴露偶发慢尾。平均数容易被少数极慢样本拉高；中位数又可能隐藏尾部，所以最好一起报。

### 9.6 Reduce-scatter 使用相同时间骨架，但输入 shape 不同

**【课程代码｜源码 338–360】【视频补充｜[51:05](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3065s)】**

```python
input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))
output = torch.empty(num_elements, device=cuda_if_available(rank))

# warmup → CUDA synchronize → barrier
# start_time
dist.reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()
dist.barrier()
# end_time
```

当 $`p=4`$、`num_elements=104,857,600`：

- `output.shape = [104,857,600]`，大小 $`C=400`$ MiB；
- `input.shape = [4,104,857,600]`；
- input 元素数是 $`4\times104{,}857{,}600=419{,}430{,}400`$；
- input bytes 是 $`419{,}430{,}400\times4=1{,}677{,}721{,}600`$ bytes；
- 换成 MiB：$`1{,}677{,}721{,}600\div1{,}048{,}576=1600`$ MiB。

视频 [51:19](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3079s) 指出 reduce-scatter 输入是四个 output chunks 的合体；[52:02](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3122s) 开始解释它的带宽分子。

---

## 10. 审计课程带宽公式：每一个因子从哪里来

### 10.1 All-reduce：先定义符号与 ring 假设

**【课程公式 + 补充分解｜源码 323–333】【视频补充｜[49:24](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2964s)】**令：

- $`p=4`$：rank 数；
- $`S=400`$ MiB：每 rank 的完整 all-reduce payload；
- $`t=10`$ ms $`=0.010`$ s：假设测得 duration；
- 算法模型：均匀分块的理想 ring all-reduce；
- 暂时忽略协议 headers、对齐、重传和 topology 绕路。

Ring all-reduce 可分为一次 reduce-scatter 加一次 all-gather。在每个阶段，一个 rank 发出 payload 的 $`(p-1)/p`$：

```math
\frac{p-1}{p}S
=\frac{4-1}{4}\times400
=\frac34\times400
=300\ \text{MiB}.
```

两个阶段每 rank 总发送：

```math
300+300=600\ \text{MiB}.
```

在对称理想 ring 中，每 rank 也接收 600 MiB。

所有 4 ranks 的 aggregate sends：

```math
4\times600=2400\ \text{MiB}.
```

若发送端和接收端都算：

```math
4\times(600+600)=4800\ \text{MiB}.
```

### 10.2 All-reduce 的 `algbw` 与 normalized `busbw`

NCCL-tests 的口径先定义算法带宽：

```math
\text{algbw}=\frac{S}{t}.
```

这里 $`S=400`$ MiB $`=400/1024=0.390625`$ GiB，所以：

```math
\text{algbw}
=\frac{0.390625\ \text{GiB}}{0.010\ \text{s}}
=39.0625\ \text{GiB/s}.
```

用十进制 GB/s 复算：

```math
\frac{419{,}430{,}400\ \text{bytes}}{0.010\ \text{s}}
=41{,}943{,}040{,}000\ \text{bytes/s}
=41.94304\ \text{GB/s}.
```

两数不同只是单位底数不同。

All-reduce 的归一化 bus bandwidth 校正因子是：

```math
2\frac{p-1}{p}
=2\times\frac34
=1.5.
```

因此：

```math
\text{busbw}
=\text{algbw}\times1.5
=39.0625\times1.5
=58.59375\ \text{GiB/s}.
```

十进制口径对应：

```math
41.94304\times1.5=62.91456\ \text{GB/s}.
```

这个 `busbw` 是 [NCCL-tests 官方性能口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) 的归一化量。它便于比较，不承诺等于某一根 NVLink 或 NIC 的实际抓包速率；现代分层/硬件卸载算法尤其不能只靠这个数还原物理路径。

### 10.3 逐字符审计源码 all-reduce 公式

课程写：

```python
size_bytes = data.element_size() * data.numel()
sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
total_duration = world_size * duration
bandwidth = sent_bytes / total_duration
print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
```

最后一行是当前源码的 `round(...)` 整数舍入，不是 `.2f` 保留两位小数。若教学时想看小数，可以另写格式化版本，但不能冒充原行。

把 $`S=400`$ MiB、$`p=4`$、$`t=0.010`$ s 代入：

```math
\text{sent\_bytes numerator}
=S\times2\times(p-1)
=400\times2\times3
=2400\ \text{MiB}.
```

这个 2400 MiB 正好等于前面算出的 **aggregate sends**。这里必须纠正源码行 325 的注释：公式里的 factor 2 来自 ring all-reduce 的 **reduce-scatter + all-gather 两个发送阶段**，不是把同一阶段的 send 与 receive 两端各数一次。若按 endpoint 的 send+receive 口径，本例还要在 2400 MiB aggregate sends 上再乘 2，得到 4800 MiB。不能一边把 factor 2 解释为“两阶段”，一边又把它重复解释为“发送端+接收端”。

分母：

```math
\text{total\_duration}
=p\times t
=4\times0.010
=0.040\ \text{rank-seconds}.
```

再除：

```math
\frac{2400\ \text{MiB}}{0.040\ \text{s}}
=60{,}000\ \text{MiB/s}
=58.59375\ \text{GiB/s}.
```

为什么等于 §10.2 的 busbw？把代数约掉：

```math
\frac{S\,2(p-1)}{p\,t}
=\frac{S}{t}\times\frac{2(p-1)}{p}
=\text{algbw}\times\frac{2(p-1)}p.
```

**关键纠错：**变量名 `total_duration = world_size * duration` 不表示“真实操作先后跑了 4 次，所以墙钟用了 40 ms”。四 ranks 是并发的，真实这次测量仍约 10 ms。$`p\times t`$ 是为了把 aggregate numerator 归一到 per-rank 平均口径而写出的 **rank-seconds** 分母。

视频 [49:29](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2969s) 开始数两阶段发送，[50:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3003s) 展开 world-size 校正，[50:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3030s) 对照打印结果。

### 10.4 `GB/s` 标签与 `/1024**3` 其实不一致

源码最后除的是：

```math
1024^3=1{,}073{,}741{,}824\ \text{bytes/GiB}.
```

所以打印出来的数值单位其实是 **GiB/s**，字符串却写成 `GB/s`。若要严格：

- 保留 `/1024**3`，标签改成 `GiB/s`；或
- 标签保留 `GB/s`，改除 `/1e9`。

这不是 2% 内可以随便忽略的拼写：本例 58.59375 GiB/s 对应 62.91456 GB/s，读硬件规格表时混用会让比较偏移。

### 10.5 Reduce-scatter：先从 output chunk $`C`$ 开始

**【课程公式 + 补充分解｜源码 338–370】【视频补充｜[52:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3128s)】**令：

- $`p=4`$；
- 每 rank output chunk $`C=400`$ MiB；
- 每 rank 输入总大小 $`S_{\text{in}}=pC=4\times400=1600`$ MiB；
- $`t=0.010`$ s；
- 仍用理想均匀 ring reduce-scatter 作流量模型。

Reduce-scatter 有 $`p-1=3`$ 个发送 steps，每步发送一个 chunk $`C`$。每 rank 发送：

```math
(p-1)C
=(4-1)\times400
=1200\ \text{MiB}.
```

每 rank 接收同量 1200 MiB。Aggregate sends：

```math
p(p-1)C
=4\times3\times400
=4800\ \text{MiB}.
```

Aggregate send + receive：

```math
2\times4800=9600\ \text{MiB}.
```

### 10.6 Reduce-scatter 的两种“看起来不同”带宽从哪里来

NCCL-tests 对 reduce-scatter 的算法数据量 $`S`$ 定义为**完整输入大小**，即本例 1600 MiB，而不是 output chunk 400 MiB：

```math
\text{algbw}
=\frac{S_{\text{in}}}{t}
=\frac{1600/1024\ \text{GiB}}{0.010\ \text{s}}
=156.25\ \text{GiB/s}.
```

Reduce-scatter 的校正因子是：

```math
\frac{p-1}{p}=\frac34=0.75.
```

所以：

```math
\text{busbw}
=156.25\times0.75
=117.1875\ \text{GiB/s}.
```

十进制单位：

```math
S_{\text{in}}=1{,}677{,}721{,}600\ \text{bytes},
```

```math
\text{algbw}=167.77216\ \text{GB/s},
```

```math
\text{busbw}=167.77216\times0.75=125.82912\ \text{GB/s}.
```

如果有人只用 output chunk 算 $`C/t=39.0625`$ GiB/s，他回答的是“每 rank 最终留下的输出 bytes / 时间”，**不是 NCCL-tests 在这里定义的 reduce-scatter algbw**。两者都可以成为自定义指标，但必须命名分子。

### 10.7 逐字符审计源码 reduce-scatter 公式

课程写：

```python
data_bytes = input.element_size() * input.numel()
sent_bytes = data_bytes * (world_size - 1)
total_duration = world_size * duration
bandwidth = sent_bytes / total_duration
print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
```

这里的课程原码同样使用 `round(...)`，不是 `.2f`。

此时 `data_bytes` 是完整 input，1600 MiB。代入：

```math
\text{sent\_bytes numerator}
=1600\times(4-1)
=4800\ \text{MiB}.
```

这等于四 ranks 的 aggregate sends。再除：

```math
\frac{4800\ \text{MiB}}{4\times0.010\ \text{s}}
=\frac{1200\ \text{MiB}}{0.010\ \text{s}}
=120{,}000\ \text{MiB/s}
=117.1875\ \text{GiB/s}.
```

代数化简：

```math
\frac{S_{\text{in}}(p-1)}{pt}
=\frac{S_{\text{in}}}{t}\times\frac{p-1}{p}
=\text{algbw}\times\frac{p-1}{p}.
```

视频 [52:24](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3144s) 逐项讨论 input 大小，[52:47](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3167s) 到 [53:06](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3186s) 对照发送次数与打印值。

### 10.8 “All-reduce 是 Reduce-scatter 两倍流量”需要固定同一个完整输入 $`S`$

若两个操作比较的是**同一个完整输入大小 $`S`$**，理想 ring 中：

```math
Q_{\text{RS, per-rank}}
=\frac{p-1}{p}S,
```

```math
Q_{\text{AR, per-rank}}
=2\frac{p-1}{p}S
=2Q_{\text{RS, per-rank}}.
```

这里 AR 的 factor 2 是 reduce-scatter 与 all-gather 两阶段。因此源码行 370 “all-reduce moves 2x the data ... compared to reduce-scatter”只有在比较相同完整输入 $`S`$ 时才成立。

但课程两次实际调用的完整输入**不同**：

- All-reduce input $`S_{\text{AR}}=400`$ MiB；
- Reduce-scatter input $`S_{\text{RS}}=1600`$ MiB，output chunk 才是 400 MiB。

所以实际教学调用：

```math
Q_{\text{AR}}
=2\times\frac34\times400
=600\ \text{MiB/rank},
```

```math
Q_{\text{RS}}
=\frac34\times1600
=1200\ \text{MiB/rank}.
```

这一次反而是 reduce-scatter per-rank sends 为 all-reduce 的 2 倍，因为它的完整输入大 4 倍。两句话不冲突：一个固定 $`S`$ 比算法，一个比较课程实际不同 shapes。

### 10.9 课程 trace 的数字是什么、又不是什么

**【课程运行快照】**讲义仓库保存的 `var/traces/lecture_07_stdout.txt` 显示，老师当时环境里：

- all-reduce 各 rank 约 1.38–1.60 ms，源码打印约 366–426 `GB/s`；
- reduce-scatter 各 rank 约 2.39–2.61 ms，源码打印约 450–490 `GB/s`。

但按 `/1024**3` 审计，这些数值标签应读作 **GiB/s**。它们只是在课程当时 GPU、拓扑、软件版本和 400 MiB chunk 下的观测，不是“任意 NCCL 集群保证达到 400”。

### 10.10 Ring 公式是一种模型，不是 NCCL 的永久实现承诺

本节用 ring 是因为它能让初学者把每步 chunk 数清楚；逻辑 collective 本身并未规定物理算法。NCCL 可以根据 topology、消息大小、协议、GPU 代际和跨节点层次选择 ring、tree 或其他方案。

因此，看到：

```math
2\frac{p-1}{p}S
```

应该读作“ring all-reduce 的理想 per-rank send 量，也对应 NCCL-tests all-reduce 的标准 busbw 校正因子”，不能读成“所有硬件链路恰好只搬这些 bytes”。同理，`busbw` 是归一化比较量；要知道某一物理链路实际走了多少，还需要 NCCL debug topology、profiler、硬件计数器或网络 telemetry（遥测）证据。

### 10.11 本阶段最小复算清单

不看正文，能从以下七步重建本轮核心数字，才算真的会：

1. $`100\times1024^2=104{,}857{,}600`$ 个 FP32 元素；
2. $`104{,}857{,}600\times4=419{,}430{,}400`$ bytes $`=400`$ MiB；
3. All-reduce ring 每 rank send $`2\times(3/4)\times400=600`$ MiB；
4. All-reduce aggregate sends $`4\times600=2400`$ MiB；
5. 10 ms 时 all-reduce algbw $`=39.0625`$ GiB/s，busbw $`=58.59375`$ GiB/s；
6. Reduce-scatter input $`=4\times400=1600`$ MiB，每 rank send $`3\times400=1200`$ MiB；
7. 10 ms 时 reduce-scatter algbw $`=156.25`$ GiB/s，busbw $`=117.1875`$ GiB/s。

## 11. Data parallel：每张卡看不同样本，梯度再求平均

### 11.1 一句话地图：切 batch，不切模型

**Data parallelism（数据并行）**把一个 batch 的样本行切给多个 ranks；每个 rank 仍保存一份完整模型，独立做 forward/backward，最后同步 gradients。**DDP（Distributed Data Parallel，分布式数据并行）**是这种模式的常用实现。

**【课程内容｜源码 375–389】【视频补充｜[55:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3308s)】**老师用三张图建立对照：[55:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3333s) 列出 data/tensor/pipeline parallel，[55:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3358s) 说明 data parallel 切的是数据，模型参数在各 GPU 上完整复制。

```text
global batch [128, 1024]
  ├─ rank 0: rows  0..31   [32, 1024] ─┐
  ├─ rank 1: rows 32..63   [32, 1024] ─┤ 各自通过同一形状、同一初值的完整模型
  ├─ rank 2: rows 64..95   [32, 1024] ─┤
  └─ rank 3: rows 96..127  [32, 1024] ─┘
                                         ↓
                                  AVG all-reduce gradients
                                         ↓
                                  各 rank 做相同更新
```

### 11.2 `128 ÷ 4 = 32` 不是结论，要把索引范围列出来

**【课程代码｜源码 390–410】【视频补充｜[56:24](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3384s)】**样例数据：

```python
batch_size = 128
num_dim = 1024
data = torch.randn(batch_size, num_dim)  # shape [128, 1024]
```

`int_divide(128,4)` 先检查 $`128`$ 能否整除 $`4`$，再得到：

```math
\text{local\_batch\_size}
=\frac{128}{4}
=32.
```

对 rank $`r`$：

```math
\text{start}=r\times32,
\qquad
\text{end}=\text{start}+32.
```

Python 切片 `data[start:end]` 包含 `start`，不包含 `end`：

| rank $`r`$ | start | end | 实际行号 | 本地 shape |
|---:|---:|---:|---|---:|
| 0 | $`0\times32=0`$ | $`0+32=32`$ | 0–31 | `[32,1024]` |
| 1 | $`1\times32=32`$ | $`32+32=64`$ | 32–63 | `[32,1024]` |
| 2 | $`2\times32=64`$ | $`64+32=96`$ | 64–95 | `[32,1024]` |
| 3 | $`3\times32=96`$ | $`96+32=128`$ | 96–127 | `[32,1024]` |

四份合计：

```math
32+32+32+32=128\ \text{samples}.
```

既没有重复，也没有漏行。视频 [56:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3404s) 开始按 rows 切，[57:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3440s) 得到每 rank 32 行，[57:29](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3449s) 对应 `start_index:end_index`。

本地输入 FP32 bytes：

```math
32\times1024\times4
=131{,}072\ \text{bytes}
=128\ \text{KiB}.
```

这里 $`1\text{ KiB}=1024`$ bytes。

### 11.3 教学代码先复制完整数据再切，不是实际数据管线

**【课程边界｜源码 379–380、400–410】【视频补充｜[57:52](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3472s)】**`main` 先在父 process 生成完整 `[128,1024]` 数据，再把 `data` 作为 `mp.spawn` 参数交给各 children；各 rank 进入函数后才 slice，并把自己的 slice 搬到 GPU。

这便于逐行教学，却可能产生：

- 每个 child 都能看到完整 CPU data；
- 父子进程传递/共享对象有额外开销；
- 大数据集不可能每 rank 先装完整副本再丢掉大部分。

实际训练通常让每 rank 的 dataloader（数据加载器）配合 distributed sampler（分布式采样器），直接读取属于本 rank 的 samples。课程源码注释也明确写了这一点。

### 11.4 四层模型的参数、梯度和 Adam 状态逐项显存账

**【课程代码｜源码 412–415】【视频补充｜[58:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3484s)】**每层参数矩阵 shape 是 `[1024,1024]`。一层元素数：

```math
1024\times1024=1{,}048{,}576.
```

FP32 一层参数 bytes：

```math
1{,}048{,}576\times4
=4{,}194{,}304\ \text{bytes}
=4\ \text{MiB}.
```

四层：

```math
4\times4=16\ \text{MiB}.
```

每个 rank 都有完整四层，所以**每 rank**最小训练状态账：

| 项目 | shape/元素口径 | 每 rank FP32 大小 |
|---|---|---:|
| parameters | 4 个 `[1024,1024]` | 16 MiB |
| gradients | 每个 parameter 对应一个同 shape grad | 16 MiB |
| Adam first moment $`m`$ | 每参数一个同 shape 一阶动量 | 16 MiB |
| Adam second moment $`v`$ | 每参数一个同 shape 二阶动量 | 16 MiB |
| 合计 | params + grads + $`m`$ + $`v`$ | **64 MiB/rank** |

**Moment（动量状态）**是 AdamW 为每个参数记住的历史梯度统计；$`m`$ 类似带衰减的梯度平均，$`v`$ 类似带衰减的梯度平方平均。课程 `torch.optim.AdamW` 的小 step counter 还会占少量空间，但相对 16 MiB tensor 可忽略。

PyTorch AdamW 通常在第一次 `optimizer.step()` 时才懒分配 $`m,v`$；因此“刚构造 optimizer、尚未 step”的瞬间可能还看不到这 32 MiB，而稳定训练状态会持有它们。这里算的是**第一次更新后的训练状态**。

这张 64 MiB 表**只数** FP32 parameters、gradients、Adam $`m,v`$。它明确不包括：

- forward 保存的 activations；
- allocator（内存分配器）的保留/碎片；
- matrix multiplication workspace；
- optimizer 临时 buffers；
- CUDA context、NCCL buffers；
- 混合精度时可能出现的额外 master weights。

因此它是教学下界账，不是 `nvidia-smi` 应显示的精确值。DDP 在这里没有减少每 rank 这 64 MiB 的模型状态复制。

### 11.5 Forward、loss、backward：每一行改变了什么

**【课程代码｜源码 416–425】【视频补充｜[58:36](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3516s)】**

回忆 §3.5：scalar 是单个数，真正 0-D scalar tensor 的 PyTorch shape 写作 `[]`；而 `[1]` 是含一个标量元素的一维 tensor。

```python
x = data                         # [32, 1024]
for param in params:             # 四次
    x = x @ param                # [32,1024] @ [1024,1024] -> [32,1024]
    x = F.gelu(x)                # 逐元素，shape 仍 [32,1024]
loss = x.square().mean()         # 所有元素平方后求平均 -> scalar
loss.backward()                  # 填充每个 param.grad，shape [1024,1024]
```

**Scalar（标量）**是只有一个数的量；这里 `loss.shape=[]`。Loss 是越小越好的“坏程度”。每个 rank 用不同 samples，所以 local loss 通常不同；`backward()` 沿计算图把这个 local mean loss 对每个 parameter 的 gradient 算出来。

视频 [58:57](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3537s) 进入 backward，[59:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3543s) 指出不同数据产生不同本地 gradients。

### 11.6 为什么“等大 local batch + local mean + AVG”恰好等于 global mean

先不用矩阵，只看一个标量参数 $`w`$。假设每个 sample 的 loss 是：

```math
\ell_j(w)=a_jw.
```

当 $`w`$ 增加一点点 $`\Delta w`$，loss 增加 $`a_j\Delta w`$，所以该 sample 对 $`w`$ 的 gradient 就是 $`a_j`$。

两个 ranks，每 rank 两个 samples：

| rank | 两个 sample gradients | local mean gradient |
|---:|---|---:|
| 0 | $`2,6`$ | $`(2+6)/2=4`$ |
| 1 | $`10,14`$ | $`(10+14)/2=12`$ |

对两个 local means 做 AVG all-reduce：

```math
\frac{4+12}{2}=8.
```

若把四个 samples 当一个 global batch，global mean gradient 是：

```math
\frac{2+6+10+14}{4}
=\frac{32}{4}
=8.
```

两者相同。一般地，$`p`$ 个 ranks，每 rank 恰好 $`b`$ 个 samples，本地先除 $`b`$，跨 rank 再除 $`p`$：

```math
\frac1p\sum_{r=1}^{p}
\left(\frac1b\sum_{j=1}^{b}g_{r,j}\right)
=\frac1{pb}\sum_{r=1}^{p}\sum_{j=1}^{b}g_{r,j}.
```

右边就是 $`pb`$ 个 samples 的 global mean gradient。课程在 [59:15](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3555s) 进入关键同步，[59:37](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3577s) 明确用 AVG，[59:47](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3587s) 得到各 rank 相同 gradient。

### 11.7 本地 batch 不等大时，直接平均 local means 会错

改成：

- rank 0 只有 1 个 sample，gradient 为 $`2`$，local mean $`=2`$；
- rank 1 有 3 个 samples，gradients 为 $`6,10,14`$，local mean $`=(6+10+14)/3=10`$。

错误的“两个 rank 等权 AVG”：

```math
\frac{2+10}{2}=6.
```

真正四样本 global mean：

```math
\frac{2+6+10+14}{4}=8.
```

正确做法按本地样本数加权：

```math
\frac14\times2+\frac34\times10
=0.5+7.5
=8.
```

所以“AVG gradients 等价 global mean”有条件：各 rank 的有效样本数/有效 token 权重相同，且 local loss 的 reduction 口径一致。Padding 的无效 tokens、最后一个不满 batch、动态 batch 都可能破坏这个条件。

### 11.8 同步 gradient 后为什么 parameters 继续一样

**【课程代码｜源码 427–434】【视频补充｜[60:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3603s)】**

```python
for param in params:
    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
optimizer.step()
```

成立需要三件事：

1. 各 rank 开始时 parameter 值一样；
2. AVG 后各 rank 的 gradient 一样；
3. 各 rank 的 optimizer state 与更新规则一样。

于是相同旧参数、相同 gradient、相同 Adam 状态会产生相同新参数。视频 [60:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3622s) 用“像处理完整数据一样更新”总结；[62:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3728s) 再串起 local loss 不同→gradient 同步→parameters 保持相同。

### 11.9 Manual DDP 与 PyTorch DDP 不是同一个工程层级

课程写的是 **manual DDP（手动 DDP 教学骨架）**：backward 完成后，Python 循环遍历四个 parameters，逐个调用 all-reduce。

PyTorch `DistributedDataParallel` 包装模型后，会利用 autograd hooks（反向传播时触发的回调）在 gradients 准备好时同步，通常还会：

- 把多个小 gradients 装进 **buckets（梯度桶，即合成一次通信块的一组小 gradients）**，减少小 collective 数；
- 让后面 layers 继续 backward 时，前面已完成的 buckets 开始通信，以 overlap computation/communication；
- 处理 unused parameters 等工程语义与一致性检查。

**【课程口头边界｜[60:36](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3636s)】**老师称核心数学只比单卡训练多 gradient all-reduce；这说明原理很精炼，不表示生产 DDP 的实现只有一行。课堂还说明 batch 通常至少不小于 world size，[60:52](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3652s)；整除会更简单，[61:13](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3673s)。

### 11.10 三个不看 helper 就发现不了的源码边界

#### 边界 A：每次 `get_init_params` 都重新设同一个 seed

**Seed（随机种子）**是伪随机数生成器的起点；相同算法、shape、device 条件下，相同 seed 会生成可复现序列。Helper 是：

```python
def get_init_params(num_inputs, num_outputs, rank):
    torch.random.manual_seed(0)
    return nn.Parameter(torch.randn(num_inputs, num_outputs, ...) / math.sqrt(num_outputs))
```

因为 `manual_seed(0)` 写在函数**里面**，而 comprehension 每层都重新调用函数：

- 不同 ranks 的同 shape 参数数值相同——这对 DP 的共同初值是需要的；
- 同一 rank 的 layer 0、1、2、3 也从同一 seed 重新生成，因此四层初值矩阵彼此相同——这只是教学简化，不是一般初始化意图。

生产代码常在模型构建前统一设一次 seed，或明确广播 rank 0 的初始化参数；不会为了每一层都相同而反复重置。

#### 边界 B：`num_steps=1` 掩盖了没有 `zero_grad`

PyTorch 默认让连续 `backward()` 把新 gradient **加到**旧 `.grad` 上，这叫 gradient accumulation（梯度累积）。`optimizer.step()` 更新参数，但不会自动清掉 `.grad`。

课程只传 `num_steps=1`，所以旧 gradient 不存在，缺少清零暂时不暴露。若改成两步：

```text
step 0 backward: grad = g0
optimizer.step(): grad 仍是 g0
step 1 backward: grad = g0 + g1   ← 若本意只用第二步，这是错的
```

通常每步在 backward 前写：

```python
optimizer.zero_grad(set_to_none=True)
```

如果确实想把一个大 batch 切成多个小批次（microbatches）做 gradient accumulation，也必须明确除数与何时同步/step，不能靠遗漏 `zero_grad` 偶然实现。

#### 边界 C：optimizer state 完整复制

每个 rank 都各自构建 `AdamW(params, ...)`，所以 $`m,v`$ 也各复制一份。源码 [62:32](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3752s) 预告 FSDP/ZeRO，就是为了下一讲讨论怎样不让每个 rank 永久持有全部模型状态；本讲不提前把它们的具体阶段当作已讲内容。

---

## 12. Tensor parallel：把一个矩阵的输出列切给多个 ranks

### 12.1 Column tensor parallel 切的是 $`W`$ 的 columns

**Tensor parallelism（张量并行，TP）**把一个 layer 内部的大 tensor/矩阵运算拆到多个 ranks。课程只演示 **column parallel（列并行）**：按 weight matrix 的输出列切分。

**【课程内容｜源码 439–459】【视频补充｜[62:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3779s)】**数据不切，每个 rank 都先有：

```math
x:\ [128,1024].
```

完整权重若写成：

```math
W:\ [1024,1024],
```

按 4 ranks 切输出宽度：

```math
\text{local\_num\_dim}
=1024/4
=256,
```

```math
W_r:\ [1024,256].
```

可视为：

```math
W=[W_0\mid W_1\mid W_2\mid W_3].
```

竖线表示沿 columns 横向拼起来。视频 [63:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3783s) 说明“不切 data，切每层”，[64:07](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3847s) 给出本地 `[num_dim,local_num_dim]`，[64:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3875s) 命名 column tensor parallel。

### 12.2 每层 forward 的 shape 从 `[128,1024]` 缩到 `[128,256]` 再拼回

**【课程代码｜源码 461–475】【视频补充｜[64:43](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3883s)】**每个 rank：

```math
[128,1024]\ @\ [1024,256]
\longrightarrow [128,256].
```

矩阵乘的内侧 1024 对齐；输出保留左矩阵 rows=128 和右矩阵 columns=256。

接着 `F.gelu` 是逐元素函数，shape 不变：

```math
x_r:[128,256].
```

每个 rank 预分配 4 个 `[128,256]` buffers，all-gather 后：

```text
activations[0] = rank 0 的 [128,256]
activations[1] = rank 1 的 [128,256]
activations[2] = rank 2 的 [128,256]
activations[3] = rank 3 的 [128,256]
```

再执行：

```python
x = torch.cat(activations, dim=1)
```

`dim=1` 是 column/output-feature 轴：

```math
[128,256]\times4\ \text{份}
\longrightarrow [128,1024].
```

于是下一层又能接收完整宽度 `[128,1024]`。视频 [65:07](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3907s) 追踪 local matmul，[65:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3918s) 说明 GeLU 可在本地逐元素做，[65:54](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3954s) 选择 all-gather，[66:54](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4014s) 拼回完整宽度。

### 12.3 每层 activation bytes 逐项算

每 rank 的 local partial 元素数：

```math
128\times256=32{,}768.
```

FP32 bytes：

```math
32{,}768\times4
=131{,}072\ \text{bytes}
=128\ \text{KiB}.
```

四个 all-gather receive buffers：

```math
4\times128=512\ \text{KiB}.
```

拼好的 full activation 也有：

```math
128\times1024\times4
=524{,}288\ \text{bytes}
=512\ \text{KiB}.
```

因此，在 `torch.cat` 发生的一瞬间，四个 receive buffers 共 512 KiB，新的 concatenated output 又是 512 KiB；若两者同时存活，仅这两项就约 1 MiB。实际 peak 还要加 local `x`、旧 tensor、allocator 与 kernels 的临时空间。

逻辑上，每 rank 自己贡献 128 KiB，需要取得其他三个 ranks 共：

```math
3\times128=384\ \text{KiB}
```

的远端 partials；物理发送量仍依 all-gather 算法与 topology，不能把 384 KiB 当作所有链路的唯一流量。

### 12.4 参数分片 bytes：每 rank 每层 1 MiB

本地 $`W_r`$ 元素：

```math
1024\times256=262{,}144.
```

FP32 bytes：

```math
262{,}144\times4
=1{,}048{,}576\ \text{bytes}
=1\ \text{MiB/layer/rank}.
```

四层是 4 MiB/rank。四 ranks 合计持有 16 MiB，恰好等于四个完整 `[1024,1024]` 层的 parameter bytes；TP 的目标是分摊到各 rank，而不是把数学模型参数凭空删掉。

### 12.5 用真实小矩阵验证“列切 + concat = 完整 matmul”

**【补充例子】**设两 ranks，输入：

```math
x=
\begin{bmatrix}
1&2&0&-1\\
0&1&1&2
\end{bmatrix}
\quad[2,4],
```

```math
W=
\begin{bmatrix}
1&0&2&1\\
0&1&1&0\\
1&1&0&2\\
0&2&1&1
\end{bmatrix}
\quad[4,4].
```

Rank 0 拿前两列：

```math
W_0=
\begin{bmatrix}
1&0\\0&1\\1&1\\0&2
\end{bmatrix}
\quad[4,2].
```

第一行 `[1,2,0,-1]` 乘两列：

```math
1\times1+2\times0+0\times1+(-1)\times0=1,
```

```math
1\times0+2\times1+0\times1+(-1)\times2=0.
```

第二行 `[0,1,1,2]`：

```math
0\times1+1\times0+1\times1+2\times0=1,
```

```math
0\times0+1\times1+1\times1+2\times2=6.
```

所以：

```math
xW_0=
\begin{bmatrix}1&0\\1&6\end{bmatrix}.
```

Rank 1 拿后两列：

```math
W_1=
\begin{bmatrix}
2&1\\1&0\\0&2\\1&1
\end{bmatrix}.
```

第一输入行 `[1,2,0,-1]` 的两个格：

```math
1\times2+2\times1+0\times0+(-1)\times1=3,
```

```math
1\times1+2\times0+0\times2+(-1)\times1=0.
```

第二输入行 `[0,1,1,2]` 的两个格：

```math
0\times2+1\times1+1\times0+2\times1=3,
```

```math
0\times1+1\times0+1\times2+2\times1=4.
```

所以：

```math
xW_1=
\begin{bmatrix}3&0\\3&4\end{bmatrix}.
```

沿 columns concat：

```math
\mathrm{cat}(xW_0,xW_1)
=
\begin{bmatrix}
1&0&3&0\\
1&6&3&4
\end{bmatrix}
=xW.
```

直接算完整 $`xW`$ 的第一行四个 dot products 也得到：

```math
1\times1+2\times0+0\times1+(-1)\times0=1,
```

```math
1\times0+2\times1+0\times1+(-1)\times2=0,
```

```math
1\times2+2\times1+0\times0+(-1)\times1=3,
```

```math
1\times1+2\times0+0\times2+(-1)\times1=0.
```

所以第一行确实是 `[1,0,3,0]`，不是只凭 concat 形式猜出来。

这不是近似；当 $`W`$ 真正按不重叠 columns 切分时，它由矩阵乘每个输出 column 独立计算的定义得到。逐元素 GeLU 也可在 concat 前各自作用，因为它不会混合 columns。

### 12.6 源码初始化没有真的把一个完整 $`W`$ 切成四块

**【关键源码边界｜源码 459、594–597】**课程每 rank 调用：

```python
get_init_params(1024, 256, rank)
```

Helper 的完整计算是 `torch.randn(num_inputs,num_outputs) / sqrt(num_outputs)`。它每次先 `manual_seed(0)`，再生成同 shape `[1024,256]`。因此在相同随机实现条件下：

```text
rank 0 local block == rank 1 local block == rank 2 local block == rank 3 local block
```

它们不是一个随机 `[1024,1024]` 完整矩阵的四个**不同** columns。四块 concat 后，相当于把同一 256-column block 重复四次。

还有第二个不等价：课程 local shard 的除数是：

```math
\sqrt{256}=16,
```

而先创建完整 `[1024,1024]` 再切 columns 时，helper 口径的除数会是：

```math
\sqrt{1024}=32.
```

对同一个原始随机数 $`z`$，local helper 给 $`z/16`$，完整矩阵 helper 给 $`z/32`$：

```math
\frac{z/16}{z/32}=\frac{32}{16}=2.
```

所以课程 TP block 的单元素尺度是该 full-helper-then-slice 对照的 2 倍。重复 block 与缩放差异都不影响 shape/collective 教学，却不能用来验证真实 TP 与某个未切模型的逐值等价。

概念上正确的教学初始化可以是：

```python
# 伪代码：只用于说明全局矩阵和列分片的关系
if rank == 0:
    torch.manual_seed(0)
    full_W = torch.randn(1024, 1024)
# 把 full_W 的 [rank*256:(rank+1)*256] 列送到对应 rank
local_W = full_W[:, rank * 256 : (rank + 1) * 256]
```

大模型不会要求 rank 0 永久先装完整 $`W`$；可用按 global index 可复现的 distributed initialization，直接让各 rank 生成自己负责且互不重复的 shard。核心条件是：所有 local shards 合起来要对应同一个定义明确的 global parameter。

同理，因为每层也重置 seed，课程四层 local matrices 彼此相同；这是教学可复现设置，不是 TP 必须如此。

### 12.7 Backward 在源码中明确省略

**【课程边界｜源码 479】【视频补充｜[67:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4038s)】**源码只写 `# Backward pass: homework exercise`。因此本讲运行示例只证明 forward shape 与通信路径，不证明完整训练正确。

对真实 tensor parallel：

- column-parallel linear 的 parameter gradient 在本 shard 上计算；
- input gradient 往往需要把各输出分片的贡献相加；
- row-parallel 与 column-parallel layers 常成对安排，以避免每层都完整 gather；
- 具体是 all-reduce、reduce-scatter 还是其他 collective，取决于前后 sharding layout。

**【视频补充｜[68:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4088s)】**课堂问答提到 forward all-gather 与 backward reduce-scatter 的对偶；[68:36](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4116s) 说明裸 `.backward()` 不会凭空知道你设计的跨 rank 分片语义。生产框架可以封装这些规则，但这份从零代码选择显式展示而未实现。

### 12.8 TP 为什么通常偏爱高速节点内互连

每一层都生成 activation partial，并在课程版本里 all-gather 成 full activation；四层就通信四次。相比 DP 通常每训练 step 在 backward 同步 gradients，TP 通信频率更贴近每层执行路径。

视频 [67:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4041s) 对比 DP 的“模型黑盒”与 TP 必须改 layer 内部；[67:42](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4062s) 用“小矩阵乘后再合并”解释可分性。后文 §14 会把“高频 activation 通信→通常放在 NVLink domain”纳入选择表。

---

## 13. Pipeline parallel：按深度切 layers，用 microbatches 填流水线

### 13.1 一句话地图：切 layers，不切单层 width

**Pipeline parallelism（流水线并行，PP）**把连续 layers 分给不同 stages。**Stage（流水线阶段）**是持有一段模型并执行这段计算的 rank 或 rank 组。前一个 stage 把 boundary activation（分段边界处的激活）发给下一个 stage。

**【课程内容｜源码 484–505】【视频补充｜[69:38](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4178s)】**课程配置：

- `num_layers=4`；
- `world_size=2`；
- `local_num_layers=4/2=2`；
- rank 0/stage 0 持有前 2 层；
- rank 1/stage 1 持有后 2 层；
- 每个本地层仍是完整 `[1024,1024]`，没有再按 width 切。

```text
data [128,1024]
   ↓
stage 0 / rank 0: layer 0 → layer 1
   ↓ send boundary activation [micro_batch,1024]
stage 1 / rank 1: layer 2 → layer 3
   ↓
final activation
```

视频 [69:42](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4182s) 说明沿网络深度切，[70:23](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4223s) 计算每 rank 持有多少 layers。

### 13.2 参数 bytes：每 stage 是两层，共 8 MiB

§11 已算一层 `[1024,1024]` FP32 参数是 4 MiB。每 rank 持有两层：

```math
2\times4=8\ \text{MiB parameters/rank}.
```

两 ranks 合计：

```math
8+8=16\ \text{MiB},
```

等于完整四层模型。PP 分摊 parameters，但每个 stage 要保存本地 layers 的 gradients/optimizer state；若使用 AdamW FP32，单看 params+grads+$`m`$+$`v`$，本教学模型每 stage 是：

```math
8\times4=32\ \text{MiB}.
```

仍不含 activations、allocator、通信 buffers 等。

**源码初始化边界：**PP 也调用同一个 `get_init_params(1024,1024,rank)`。因为 helper 每次内部重置 `manual_seed(0)` 且各层 shape 相同，rank 0 的两层彼此同值，rank 1 的两层也彼此同值；相同生成条件下，两 stages 的这些层还会重复同一初值矩阵。它演示的是“每 stage 持有两张完整 shape 的矩阵”，不是一个正常四层模型中四组不同参数的逐值切分。

### 13.3 Batch 128 切 4 个 microbatches，每个 32 行

**Microbatch（微批次）**是从一个训练 batch 再切出来、依次送进 pipeline 的小批。课程：

```math
\text{micro\_batch\_size}
=\frac{128}{4}
=32.
```

四个 microbatches：

| microbatch | 原 batch 行号 | shape |
|---:|---|---:|
| 0 | 0–31 | `[32,1024]` |
| 1 | 32–63 | `[32,1024]` |
| 2 | 64–95 | `[32,1024]` |
| 3 | 96–127 | `[32,1024]` |

视频 [70:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4248s) 首次引入 microbatches，[71:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4264s) 说明 batch 还要再切，[71:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4280s) 对应 `data.chunk(chunks=4, dim=0)`。

### 13.4 一个 boundary activation 是 128 KiB，四次共 512 KiB

每个 microbatch 通过 stage 0 两层后，shape 仍 `[32,1024]`。FP32 bytes：

```math
32\times1024\times4
=131{,}072\ \text{bytes}
=128\ \text{KiB}.
```

四个 microbatches 从 rank 0 发给 rank 1：

```math
4\times128=512\ \text{KiB sends}.
```

rank 1 接收同样 512 KiB。这里是本例单个 stage boundary 的应用 payload；协议 overhead 与物理链路流量另算。

### 13.5 `recv → compute → send` 逐行解释

**【课程代码｜源码 506–530】【视频补充｜[71:32](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4292s)】**

```python
for x in micro_batches:
    if rank - 1 >= 0:
        dist.recv(tensor=x, src=rank - 1)

    for param in local_params:
        x = x @ param
        x = F.gelu(x)

    if rank + 1 < world_size:
        dist.send(tensor=x, dst=rank + 1)
```

Rank 0：

1. `rank-1=-1`，没有前驱，不 `recv`；`x` 已是本地 data chunk。
2. 依次做本地 layer 0、1，shape 一直 `[32,1024]`。
3. `rank+1=1<2`，把 boundary activation send 给 rank 1。

Rank 1：

1. `rank-1=0`，先把 rank 0 的 tensor recv 到预分配 `[32,1024]` buffer `x`。
2. 做本地 layer 2、3。
3. `rank+1=2` 不小于 world size 2，它是最后 stage，不再 send。

**Point-to-point communication（点对点通信）**指定一个 sender 和一个 receiver；这里 `send(dst=1)` 与 `recv(src=0)` 必须匹配。它不同于所有 group 成员共同参加的 collective。视频 [71:51](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4311s) 展开 `recv`/`send` 的两端。

### 13.6 Blocking send/recv 与最小 deadlock

课程使用 blocking（阻塞式）`dist.send`/`dist.recv`：调用要等对应通信取得进展/完成到 API 规定的边界后才返回。当前两 stage 次序能匹配：rank 1 先 `recv`，rank 0 算完后 `send`。

下面是**故意错误、不要运行**的两-rank 伪代码：

```python
# rank 0
dist.send(x0, dst=1)  # 等 rank 1 接收
dist.recv(y0, src=1)

# rank 1
dist.send(x1, dst=0)  # 也先发送，等 rank 0 接收
dist.recv(y1, src=0)
```

若 backend 的 blocking send 需要匹配 recv 才能返回，两边都卡在第一行。安全 schedule 要让一边先 recv，或使用经过正确 `isend`/`irecv` 配对与 `wait` 管理的异步方案。

### 13.7 $`m=4,p=2`$ 的五个时刻表

先做一个**理想 forward-only 模型**：

- $`m=4`$ microbatches；
- $`p=2`$ stages；
- 每个 stage 处理一个 microbatch 都恰好用 1 个时间单位；
- 暂时忽略通信时间；
- stage 0 处理下一个 microbatch 可与 stage 1 处理上一个重叠。

| 时刻 | stage 0 / rank 0 | stage 1 / rank 1 |
|---:|---|---|
| $`t_1`$ | microbatch 0 | idle |
| $`t_2`$ | microbatch 1 | microbatch 0 |
| $`t_3`$ | microbatch 2 | microbatch 1 |
| $`t_4`$ | microbatch 3 | microbatch 2 |
| $`t_5`$ | idle | microbatch 3 |

总设备时间槽：

```math
p\times(m+p-1)
=2\times(4+2-1)
=2\times5
=10.
```

有用计算槽：

```math
m\times p=4\times2=8.
```

Utilization（利用率）：

```math
\frac{8}{10}=0.8=80\%.
```

Idle slots 是 $`10-8=2`$，所以 bubble fraction：

```math
\frac2{10}=0.2=20\%.
```

**Pipeline bubble（流水线气泡）**是 stage 因数据尚未到达或 pipeline 正在排空而 idle 的时间槽。视频 [72:28](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4348s) 说明为什么需要 microbatches，[73:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4383s) 命名 bubbles，[73:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4398s) 解释更多小批能减少气泡比例。

### 13.8 一般 forward utilization 公式从哪里来

第一 microbatch 从 stage 0 走到 stage $`p-1`$ 要填满 pipeline；最后 microbatch 离开还要排空。等时 stage 下，总时刻数：

```math
m+p-1.
```

每个 $`p`$ stages 对 $`m`$ microbatches 各做一次计算，有用槽 $`mp`$；总槽 $`p(m+p-1)`$：

```math
U_{\text{forward}}
=\frac{mp}{p(m+p-1)}
=\frac{m}{m+p-1}.
```

代 $`m=4,p=2`$：

```math
U=\frac4{4+2-1}=\frac45=80\%.
```

这条公式只适用于本节假设：forward-only、各 stage 等时、通信可忽略或完全隐藏、microbatches 连续排程。真实 stage 不平衡、链路延迟、backward 与调度策略都会改变利用率。

### 13.9 课程 pipeline 实现缺了什么

**【课程边界｜源码 532–534】【视频补充｜[73:32](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4412s)】**源码明确未处理 communication/computation overlap，并把 backward 留作作业。具体缺项：

- 没有 loss 与 backward；
- 没有 activation/gradient 的反向传输；
- 没有 **1F1B（one-forward-one-backward，一次前向接一次反向）**等训练 schedule；
- 使用 blocking send/recv，没有显式 `isend`/`irecv` overlap；
- 没有 optimizer、gradient accumulation 或 parameter update；
- 没有处理 stage 不均衡与跨节点 topology；
- 所有 ranks 进入函数时先把完整 `data` 搬到自己的 device，随后只有 rank 0 真正 chunk 原数据；rank 1 又另分配 recv buffers。这同样是教学简化。

视频 [73:45](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4425s) 说明异步通信需要更多管理，[74:23](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4463s) 总结课程还未覆盖的 overlap；不能把这段 forward demo 的 80% 理想利用率冒充源码实测。

---

## 14. 三种切法比较：batch、width、depth 到底各换来了什么

### 14.1 一张表先回答“切了什么”

| 维度 | Data parallel / DDP | Tensor parallel / TP | Pipeline parallel / PP |
|---|---|---|---|
| 切分对象 | batch rows / samples | layer 内 tensor width | 模型 layers / depth |
| 每 rank parameters | 完整复制 | 每层只存本地 shard | 只存本地 stages 的完整 layers |
| 每 rank optimizer state | 朴素 DDP 完整复制 | 对本地 parameter shard 保存 | 对本地 layers 保存 |
| 主要 activation 形态 | local batch、完整 hidden width | 课程每层 local partial 后 gather full | boundary microbatch 在 stages 间流动 |
| 本讲通信 | backward 后 gradient all-reduce | 每层 activation all-gather | 相邻 stages point-to-point send/recv |
| 通信频率直觉 | 通常每 step/bucket | 贴近每层，频繁 | 每个 microbatch 过 stage boundary |
| 主要约束 | global batch、有效样本加权 | hidden dimensions 可分、快链路 | layers 可平衡分段、bubble |
| 本讲代码完整度 | 一步 manual training | forward only | forward only |

视频 [74:41](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4481s) 补充 DP 可在 backward 中提早同步已就绪 gradient buckets；[75:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4510s) 提醒 MLP 只展示核心，Transformer 还会增加 bookkeeping（形状、布局和通信的管理工作）。

### 14.2 DDP、FSDP、ZeRO：都与 data parallel 有关，但复制程度不同

- **DDP**：本讲版本。每 rank 完整持有 parameters、gradients、optimizer state；不同 data，梯度 all-reduce。
- **FSDP（Fully Sharded Data Parallel，完全分片数据并行）**：在 data-parallel ranks 间分片 parameters、gradients、optimizer state，并按计算需要 all-gather/reshard；这里只作下一讲预告。
- **ZeRO（Zero Redundancy Optimizer，零冗余优化器）**：按阶段分片 optimizer state、gradients、parameters，以减少 data-parallel replicas 的冗余；具体阶段和生命周期下一讲再讲。

它们都可以让不同 ranks 处理不同 data，但 FSDP/ZeRO 不能被概括为“普通 DDP 换个名字”。课程在 [62:39](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3759s) 的预告重点正是：DDP 的完整模型状态可能放不下。

### 14.3 TP、expert parallel、sequence parallel 也不是同一个切法

- **Tensor parallel**：切一个 dense layer 的矩阵维度，多个 ranks 合作完成每个 token 的同一层计算。
- **Expert parallel（专家并行）**：把 Mixture-of-Experts 的不同 experts 放在不同 ranks，router 后常用 all-to-all 把 tokens 送到所选 experts。
- **Sequence parallel（序列并行）**：按 token/sequence 位置切 activation 或某些计算维度，降低每 rank 的序列相关内存/计算。

视频 [75:36](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4536s) 先列 sequence parallel，[75:47](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4547s) 把 expert parallel 与 all-to-all 联系起来，[76:02](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4562s) 强调实际会组合多种 parallelism。

### 14.4 链路为什么会影响切法

TP 往往每 layer 都要交换 activations/partials，对 bandwidth 和 latency 很敏感，所以常把一个 TP group 放在同一 NVLink/NVSwitch domain。PP 主要在 stage boundary 传 microbatch activations，若计算块足够大，通信占比可更小，对较慢跨节点链接可能更宽容。DP/FSDP 的梯度/参数通信也很大，但可通过 buckets 与 backward overlap，且不一定每一层 forward 都立刻需要 full activation gather。

**【视频补充｜[76:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4570s)】**老师明确说选择强依赖硬件；[76:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4581s) 指出 TP 每层 communication 多，[76:31](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4591s) 建议通常放在 node 内高速 NVLink 域；[76:46](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4606s) 对比 PP 可容忍更慢 interconnect。

这些是常见系统设计倾向，不是“TP 绝不跨节点”或“PP 在慢网一定快”的定律。消息大小、计算粒度、network contention 和软件实现仍要 benchmark。

### 14.5 一个具体选型例：16 GPUs、每卡 64 GiB

**【补充例子】**假设：

- 共 16 GPUs，2 nodes，每 node 8 GPUs；
- node 内 NVLink 快，node 间网络慢一些；
- 模型当前训练状态（params+grads+optimizer state）共 160 GiB；
- 每 GPU 64 GiB，还要给 activations 和通信 buffers 留空间。

逐步判断：

1. **纯 DDP 能否 fit？**每 rank 要复制 160 GiB，$`160>64`$，不能 fit。
2. **先用 TP=4？**理想均分模型状态约 $`160/4=40`$ GiB/rank，留下约 $`64-40=24`$ GiB 给 activations/buffers；从容量看可能可行。
3. **TP group 放哪里？**每组 4 GPUs，优先放同一 node 的 NVLink 域，因为 TP 每层频繁通信。
4. **16/4=4 个 TP replicas 怎么利用？**可令这 4 个 TP groups 处理不同 data，形成 DP degree=4；跨 replicas 同步相应 parameter-shard gradients。
5. **若 24 GiB activation 仍不够？**先考虑 activation checkpoint/recompute；也可再引入 sequence parallel/FSDP。若模型按 layers 易平衡且必须跨慢链接，可考虑 PP，但要为 bubbles 和更复杂 schedule 付代价。
6. **最后依据什么决定？**对候选 mesh 做真实 memory peak、collective profile 与 step-time benchmark，不能只看 $`160/4`$。

这个例子不是唯一答案；它展示的是约束顺序：先 fit，再看通信域，再看利用率与优化复杂度。

### 14.6 Shape 与 batch 约束也会拒绝某些方案

- 本讲 DP helper 要求 $`128`$ 被 world size 整除；真实系统可处理不等 batch，但 gradient 必须按有效样本/token 加权。
- 本讲 TP 要求 1024 被 4 整除；若维度不整除，需要不均匀 shard、padding 或改 TP degree。
- 本讲 PP 要求 layers 能平均切、batch 能整分 microbatches；即使 layer 数整除，不同 layer 计算量也可能不同，导致一个 stage 成 straggler。
- DP 继续扩大 global batch 还会遇到 optimization 的 critical batch size（临界批大小）：再增 batch 可能不再带来等比例学习收益。视频 [77:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4655s) 提到这一限制。

---

## 15. 把 DP×TP×PP 组合成 device mesh

### 15.1 Device mesh 是带坐标的设备网格

**Device mesh（设备网格）**给每个 GPU/rank 一个多维坐标；每一维代表一种 parallelism。例：

```math
\text{DP degree}=2,
\quad
\text{TP degree}=2,
\quad
\text{PP degree}=2.
```

总 GPUs：

```math
2\times2\times2=8.
```

本例坐标写成 $`(d,t,p)`$，分别是 data、tensor、pipeline index；定义 rank 编号：

```math
\text{rank}=4d+2p+t.
```

| rank | DP $`d`$ | TP $`t`$ | PP $`p`$ | 人话 |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | data replica 0、TP shard 0、前半 pipeline |
| 1 | 0 | 1 | 0 | data replica 0、TP shard 1、前半 pipeline |
| 2 | 0 | 0 | 1 | data replica 0、TP shard 0、后半 pipeline |
| 3 | 0 | 1 | 1 | data replica 0、TP shard 1、后半 pipeline |
| 4 | 1 | 0 | 0 | data replica 1、TP shard 0、前半 pipeline |
| 5 | 1 | 1 | 0 | data replica 1、TP shard 1、前半 pipeline |
| 6 | 1 | 0 | 1 | data replica 1、TP shard 0、后半 pipeline |
| 7 | 1 | 1 | 1 | data replica 1、TP shard 1、后半 pipeline |

### 15.2 同一个 rank 同时属于三个不同通信域

固定另外两维、只改变一维，就得到该 parallelism 的 process groups：

**TP groups：同一 data replica、同一 pipeline stage，改变 $`t`$**

```text
{0,1}, {2,3}, {4,5}, {6,7}
```

这些 ranks 在一个 layer 内交换 activation/partial results。

**PP groups：同一 data replica、同一 TP shard，改变 $`p`$**

```text
{0,2}, {1,3}, {4,6}, {5,7}
```

这些 ranks 在相邻 stages 间发送 boundary activations/gradients。

**DP groups：同一 TP shard、同一 pipeline stage，改变 $`d`$**

```text
{0,4}, {1,5}, {2,6}, {3,7}
```

这些 replicas 处理不同 data，随后同步对应 parameter shards 的 gradients。

例如 rank 1 坐标 $`(0,1,0)`$：

- 与 rank 0 做 TP layer 内通信；
- 与 rank 3 做 PP stage 边界通信；
- 与 rank 5 做 DP gradient 同步。

这就是为什么大型训练不能只有一个“全世界默认 group”：不同 collective 必须在正确子 group 内发生，否则 shape、语义或通信量都会错。

视频 [77:18](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4638s) 给出常见层次组合：node 内 TP，再配 data/FSDP，必要时跨更慢域做 PP；[79:09](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4749s) 总结可以按 data、tensor/expert、pipeline/sequence 多轴切。

### 15.3 Compute–memory–communication 三角

面对一个中间量或模型状态，系统常在三件事之间换：

1. **Store（存）**：留在本 GPU memory。优点是以后直接读；代价是占 HBM。
2. **Recompute（重算）**：不保存 activation，需要时重做前向。优点是省 HBM/通信；代价是增加 FLOPs 和时间。
3. **Communicate（通信）**：放在另一个 rank 或分片保存，需要时跨链路取得。优点是分摊本地 memory；代价是 bandwidth、latency、同步与 failure complexity。

还可以 **replicate（复制）**：在多个 ranks 都保存/计算一份。DDP 复制 parameters 和 optimizer state，换来每个 rank forward 时无需逐层取得参数，但 backward 后要同步 gradients。

视频 [79:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4784s) 把全讲提升为 store/recompute 的旧权衡；[80:04](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4804s) 加上“存到另一张 GPU、需要时通信”；[80:12](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4812s) 以 DP 的冗余参数更新换较少状态移动为例。

没有一个动作免费：

| 选择 | 少了什么 | 多了什么 |
|---|---|---|
| 保存 activation | 少重算 | 多 memory |
| checkpoint/recompute | 少 activation memory | 多 compute |
| sharding + all-gather | 少常驻本地 memory | 多 communication/同步 |
| replicate | 少按层取参数的通信 | 多每 rank memory 与冗余状态更新 |

### 15.4 选择的正确顺序不是“先背流行缩写”

```text
1. 单卡能否放下 params + grads + optimizer + activations + buffers？
   ├─ 否：先用 sharding / TP / PP / recompute 解决 fit
   └─ 是：继续
2. 哪类 tensor 最大、生命周期多长？
3. 通信发生每层、每 microbatch，还是每 gradient bucket？
4. 对应 ranks 之间是什么链路和 topology？
5. shape/batch/layer 是否能均匀切，是否出现 straggler/bubble？
6. 能否 overlap communication with computation？
7. 最后以峰值 memory 与 step time 的测量决定
```

硬件会变快，模型也会变大；课程 [80:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4833s) 的最终观点是这种层级权衡会长期存在。

### 15.5 本讲代码仍缺哪些生产关键项

**【课程边界 + 补充整理】**到 §15 为止，必须诚实保留：

- 模型只是方形 MLP，没有 attention、embedding、normalization、residual、vocabulary loss；
- manual DP 没有 buckets/overlap，且只有一步、未 `zero_grad`；
- TP 只演示 column forward，初始化 shards 不对应不同 global columns，backward 省略；
- PP 只演示两 stages 的 blocking forward，无 backward/1F1B/显式 overlap；
- 所有演示先复制完整 data，不是可扩展 input pipeline；
- 没有 mixed precision、activation checkpoint、fault tolerance、elastic membership；
- 没有自动 topology-aware device-mesh placement 或真实性能搜索；
- 课程图与公式传达核心结构，不等于 PyTorch DDP/FSDP、Megatron、DeepSpeed 等生产实现的全部行为。

老师在 [78:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4683s) 说明本讲故意用原始 PyTorch collectives 展示机械过程，而某些编译器系统可以从 sharding annotations 自动推导通信；这两种抽象层级各有用途。

### 15.6 本阶段的闭环

```text
DP：切 batch
    每 rank 完整模型 → local backward → AVG gradients

TP：切 layer width
    每 rank 算 partial activation → layer 内 collective 合并

PP：切 model depth
    microbatch 沿 stages 点对点流动 → 需要减少 bubble

组合：给 rank 一个 (DP,TP,PP) 坐标
    在不同坐标轴建立不同 process groups

最终目标：
    在正确性约束下，同时平衡 compute、memory、communication
```

## 16. 性能与并行选择决策树：先解决 fit，再用证据找 throughput 瓶颈

### 16.1 第一个问题不是“哪个缩写最强”，而是单卡能不能放下

**Fit** 指训练所需的峰值显存能否装进每张 GPU。先量而不是猜：

```math
M_{\text{peak}}
=M_{\text{params}}+M_{\text{grads}}+M_{\text{optimizer}}
+M_{\text{activations}}+M_{\text{temporary}}+M_{\text{communication}}.
```

公式中：

- $`M_{\text{peak}}`$：实测或估算的峰值显存；
- params/grads/optimizer：模型状态；
- activations：forward 为 backward 留下的中间值；
- temporary：kernels/workspace/allocator 碎片；
- communication：collective buckets、send/recv buffers 等。

若 $`M_{\text{peak}}`$ 大于 GPU 可用 HBM：

```text
模型状态占主导？
  ├─ 是：FSDP/ZeRO、TP、PP 分片；必要时组合
  └─ 否：activations 占主导
          ├─ activation checkpoint/recompute
          ├─ 减小 micro/local batch
          └─ sequence parallel / TP / PP 分摊相关 activation
```

证据包括：框架的 peak allocated/reserved memory、**OOM（Out Of Memory，显存不足而分配失败）**时最后一个 tensor shape、optimizer step 前后峰值、profiler memory timeline。只看 parameter 文件大小会漏掉 gradients、Adam 状态和 activations。

### 16.2 能 fit 之后，目标才是 throughput

**Throughput（吞吐）**可以写成 tokens/s 或 samples/s；必须说明是每 GPU、每 node 还是全 job。若已经 fit：

```text
GPU compute 长时间忙？
  ├─ 是：可能 compute-bound
  │      ├─ 若优化允许增 global batch：增加 DP
  │      └─ 若达到 critical batch：再加 DP 可能浪费 compute
  └─ 否：看 profiler 空洞
         ├─ collective 占时长：communication-bound
         ├─ stage 等待：PP bubble / imbalance
         ├─ input 等待：dataloader bottleneck
         └─ 小 kernels/CPU gap：launch/host bottleneck
```

需要一起看：step time、GPU utilization 时间线、每个 collective 的 duration/bytes、各 rank 最大与中位时间、链路 counters、pipeline stage durations。一个低 utilization 百分比不能独自证明是哪类瓶颈。

### 16.3 通信慢时先问“在哪一轴、走哪条链路”

```text
TP collective 很慢？
  → TP group 是否跨出 NVLink/NVSwitch domain？
  → hidden shard 是否太小，latency/launch 占比过高？
  → 能否用 row/column layout 减少 full all-gather？

DP gradient sync 很慢？
  → bucket 是否能与 backward overlap？
  → DP group 是否跨节点、是否有 straggler？
  → global batch 是否还有增大 DP 的优化价值？

PP 很慢？
  → 各 stage compute 是否平衡？
  → microbatch 数是否足以摊薄 bubble？
  → boundary activation bytes/链路时间是否可被 compute 隐藏？
```

证据要落到“哪个 process group、哪个 tensor shape、多少 bytes、哪段时间”。不能说“网络慢”就结束诊断。

### 16.4 Batch、width、depth 三个可切轴的拒绝条件

**GEMM（General Matrix Multiply，通用稠密矩阵乘）**是形如 `[m,k]@[k,n]` 的矩阵乘；TP shard 太细会让每个 GEMM 变小，GPU 可能更难充分利用计算单元。

| 候选轴 | 先检查 | 拒绝/警告证据 |
|---|---|---|
| DP / batch | 有效样本数、critical batch、gradient weighting | local batch 太小，通信占比高；global batch 再增不改善训练效率 |
| TP / width | hidden/head/FFN dimensions 是否可切；节点内链路 | 每层 collective 主导；TP group 跨慢网；shard 后 GEMM 太小 |
| PP / depth | layers 能否平衡；microbatch 数；边界 activation | stage 时长偏斜；bubble 大；blocking communication 无法隐藏 |

“维度能整除”只是最小 correctness 条件，不表示分片后 kernel 一定高效。

### 16.5 一个带测量证据的闭环

假设 8 GPU job：

- 单卡峰值 70 GiB，设备只有 64 GiB：先解决 fit；
- TP=2 后峰值降到 43 GiB；
- profiler 显示每层 TP all-gather 占 step time 35%，且 TP ranks 跨节点；
- 把每个 TP pair 重排到同节点 NVLink 后，all-gather 降到 12%；
- 此时 PP stage 1 比 stage 0 慢 1.6 倍，出现大片 idle；
- 重新平衡 layers 后再测 step time。

这里每一步都由可观察证据触发：memory peak→collective trace/topology→stage timeline。最终选择不是“TP 总比 PP 好”，而是当前 shape、拓扑与实现下的测量结果。

### 16.6 最终决策树

```text
先验证单卡/小规模数值正确
  ↓
算 + 测峰值 memory，能 fit 吗？
  ├─ 否：判断模型状态还是 activation 主导
  │      → shard / checkpoint / TP / PP / FSDP 的组合
  └─ 是
      ↓
定义吞吐单位与训练质量约束
      ↓
profile：compute、collective、pipeline idle、input、host gap 谁主导？
      ↓
把慢项定位到 DP/TP/PP 哪个 process group 和哪条链路
      ↓
修改 degree、placement、bucket、schedule、microbatch 或切分轴
      ↓
重新检查 correctness、peak memory、step time、straggler、训练曲线
```

---

## 17. 常见误区：错误说法、为什么错、正确说法

### 17.1 Collective 语义与算法

1. **错误：**“`all_reduce` 这个名字已经规定物理网络必须走 ring。”  
   **原因：**API 只规定逻辑输入/输出；backend 可按 topology 和消息大小选 ring、tree 或分层算法。  
   **正确：**用 profiler/NCCL 日志确认某次运行的算法，不能从 API 名猜。

2. **错误：**“All-reduce 物理上永远就是先调用 reduce-scatter API，再调用 all-gather API。”  
   **原因：**两者在逻辑结果上等价；库可融合、分块流水或采用另一 schedule。  
   **正确：**把分解用于理解和设计，不当作永恒实现轨迹。

3. **错误：**“SUM 与 AVG 只差函数名，结果相同。”  
   **原因：**AVG 还除以 rank 数；四 rank SUM `[6,10,14,18]`，AVG 是 `[1.5,2.5,3.5,4.5]`。  
   **正确：**先确认 loss reduction 与期望全局缩放，再选 op。

4. **错误：**“Reduce-scatter 后每 rank 都有完整 reduce 结果。”  
   **原因：**它只给每 rank 一块；本例分别是 `[6]`,`[10]`,`[14]`,`[18]`。  
   **正确：**要完整结果还需 all-gather，或直接使用 all-reduce。

5. **错误：**“All-gather 会把数值相加。”  
   **原因：**Gather 是拼接/收集，不做 SUM。  
   **正确：**`[6]`,`[10]`,`[14]`,`[18]` gather 为 `[6,10,14,18]`。

6. **错误：**“All-to-all 总是把一个方阵转置。”  
   **原因：**4×4 转置只是均匀一元素分块的可视化；真实 split 可为多元素或不均匀。  
   **正确：**逐 sender→receiver chunk 表描述；MoE 还会有不等 token counts。

7. **错误：**“所有 collective 都有 root。”  
   **原因：**Broadcast/scatter/gather/reduce 有 root；all-gather/all-reduce/all-to-all 的结果域不同，不需要单一 root。  
   **正确：**按每个 collective 的输出位置判断。

8. **错误：**“`all_reduce` 默认会新建输出 tensor。”  
   **原因：**课程 `dist.all_reduce(data,...)` 原地覆盖 `data`。  
   **正确：**查 API 的 in-place/out-of-place 约定，别把输入值留存假设写错。

### 17.2 Rank、同步与 deadlock

9. **错误：**“Rank 就是一张 GPU。”  
   **原因：**Rank 是 process group 内的编号；课程恰好一 process 对一 GPU。  
   **正确：**显式区分 global rank、local rank、process 与 device mapping。

10. **错误：**“`world_size=8` 就表示 8 台机器。”  
    **原因：**它表示 group 中 8 个 processes/ranks，可能全在一台 8-GPU node。  
    **正确：**node count、process count、device count 分开记录。

11. **错误：**“Global rank 13 就应该 `set_device(13)`。”  
    **原因：**多机本地可能只有 GPU 0–7。  
    **正确：**用 local rank 选本机 GPU；课程 `rank→device` 只适合单机示例。

12. **错误：**“Barrier 等价于 `torch.cuda.synchronize()`。”  
    **原因：**Barrier 主要等其他 ranks 到达；CUDA synchronize 等本 device 已提交 kernels。  
    **正确：**按要等的是 process 还是 device 选择；benchmark 可能两者都需要。课程还未 `set_device`，无参 synchronize 可能等错 current device，不能只看函数名宣布计时正确。

13. **错误：**“`async_op=False` 返回时，所有 GPU、所有 streams、所有 ranks 都物理完成。”  
    **原因：**API 没有异步 Work handle，但 CUDA 仍有 stream 语义；跨 stream/CPU 计时要按文档同步。  
    **正确：**区分 API completion、same-stream ordering、device completion、rank rendezvous。

14. **错误：**“`async_op=True` 一定让通信与计算重叠并加速。”  
    **原因：**依赖关系、stream、资源竞争或太早 `wait()` 都可消灭 overlap。  
    **正确：**用 timeline 证明确有重叠，并验证结果使用前的同步。

15. **错误：**“Collective 次序不一致最多结果不一样，不会卡住。”  
    **原因：**rank 0 等 all-reduce，rank 1 等 barrier 可互相等待。  
    **正确：**所有 ranks 以匹配顺序、count、dtype 参加；为错误路径设超时与诊断。

16. **错误：**“Blocking send/recv 随便写顺序都安全。”  
    **原因：**双方先 blocking send 可能都在等对方 recv。  
    **正确：**设计可证明匹配的 send/recv 次序，或正确管理 nonblocking requests。

### 17.3 Cost model 与带宽

17. **错误：**“带宽高，小消息一定快。”  
    **原因：**$`T\approx sL+Q/B`$ 中，小 $`Q`$ 时固定 latency/launch 可能主导。  
    **正确：**同时报告消息大小、latency、bandwidth 与 steps。

18. **错误：**“Payload、per-rank sends、aggregate sends 是一个数。”  
    **原因：**400 MiB all-reduce 在四-rank ring 中分别可对应 payload 400、per-rank send 600、aggregate send 2400 MiB。  
    **正确：**每个数字先写口径。

19. **错误：**“Aggregate sends 已经把 receive 再算一次。”  
    **原因：**本讲定义 aggregate sends 只加发送端；send+receive 是另一口径。  
    **正确：**四-rank 400 MiB ring all-reduce 是 2400 MiB aggregate sends，4800 MiB endpoint send+receive。

20. **错误：**“GB 和 GiB 可以交换标签。”  
    **原因：**$`1\text{GB}=10^9`$ bytes，$`1\text{GiB}=2^{30}`$ bytes。  
    **正确：**源码除 `1024**3` 得 GiB/s；打印 `GB/s` 是标签不严谨。

21. **错误：**“源码 `total_duration=p*duration` 表示四 ranks 串行跑了四倍墙钟。”  
    **原因：**ranks 并发；$`p\times t`$ 是 aggregate numerator 的 rank-seconds 归一化。  
    **正确：**本例 operation wall time 仍约 10 ms。

22. **错误：**“NCCL-tests busbw 就是一根网线的物理吞吐。”  
    **原因：**它是 collective-specific normalized metric；分层算法可能经过多类链路。  
    **正确：**物理链路量需 topology、profiler 和硬件 counters。

23. **错误：**“Ring 公式对所有 NCCL 算法都是实际 bytes 真值。”  
    **原因：**Tree、hierarchical/offload 路径不同。  
    **正确：**把 $`2(p-1)S/p`$ 标成理想 ring 或标准归一化因子；factor 2 是 reduce-scatter+all-gather 两阶段，不是 endpoint send+receive。固定同一完整输入 $`S`$ 时 AR 是 RS 流量 2 倍；课程实际 AR input 400 MiB、RS input 1600 MiB，所以实际 per-rank sends 却是 600 与 1200 MiB。

24. **错误：**“一次 warmup、一次测量就足够发表性能结论。”  
    **原因：**冷启动、抖动、contention、straggler 会改变单样本。  
    **正确：**多次 warmup/repeat，报 median/p95，并记录软硬件与 topology。

### 17.4 网络名词与课程快照

25. **错误：**“RDMA 意味着 CPU 完全不参与任何事情。”  
    **原因：**数据路径可绕过远端 CPU 拷贝/协议处理，但连接建立、控制、队列管理仍有 CPU/软件参与。  
    **正确：**说清是 data movement path 的优化，不是 CPU 从系统消失。

26. **错误：**“RoCE 是换了名字的 InfiniBand，行为必然完全一样。”  
    **原因：**RoCE 在 Ethernet 上承载 RDMA，需要相应网络配置；拥塞与运维边界不同。  
    **正确：**共同点是 RDMA 语义，底层网络和部署条件要分开。

27. **错误：**“A100/H100/B200/GB200 表中的数字永远适用于同名所有卡。”  
    **原因：**产品形态、SXM/PCIe、容量、规格口径与课程时点不同。  
    **正确：**把硬件表标为 2026 课程快照，并回查具体官方 SKU。

### 17.5 Data parallel

28. **错误：**“DDP 自动把 parameters、Adam state 都除以 world size。”  
    **原因：**朴素 DDP 每 rank 完整复制模型状态，只切 data。  
    **正确：**分片状态需 FSDP/ZeRO 等方法。

29. **错误：**“课程每 rank 从磁盘只加载自己的 32 行。”  
    **原因：**源码先把完整 `[128,1024]` data 交给 children，再在函数里 slice。  
    **正确：**生产 dataloader/sampler 应直接读取 local shard。

30. **错误：**“不等 local batch 时，把 local mean gradients 等权 AVG 仍是 global mean。”  
    **原因：**1 样本 rank 与 3 样本 rank 被错误赋予相同权重；示例得到 6 而不是 8。  
    **正确：**按有效样本/token counts 加权。

31. **错误：**“`optimizer.step()` 会自动清 gradient。”  
    **原因：**PyTorch `.grad` 默认累加；课程 `num_steps=1` 才未暴露。  
    **正确：**正常每步显式 `zero_grad`，或有意设计 accumulation 周期。

32. **错误：**“课程四层随机初始化彼此不同。”  
    **原因：**`get_init_params` 每次内部 `manual_seed(0)`，同 shape 各层重放同一序列。  
    **正确：**这保证 ranks 对齐却也让同 shape layers 同值；DP 四层、TP local 四层、PP 两 stages 的同 shape layers 都受影响，是教学 bug/简化。

33. **错误：**“Manual DDP 的四次 Python all-reduce 等同生产 PyTorch DDP 的全部实现。”  
    **原因：**生产 DDP 还有 autograd hooks、buckets、overlap 与一致性处理。  
    **正确：**课程只保留数学核心。

34. **错误：**“只要 gradients 一样，parameters 必然一直一样。”  
    **原因：**旧 parameters 或 optimizer state 若不同，相同 gradient 也可能更新到不同值。  
    **正确：**初值、gradient、optimizer state/规则都要一致。

### 17.6 Tensor parallel

35. **错误：**“TP 的 all-gather 不占额外 activation memory。”  
    **原因：**课程每 rank 有 4×128 KiB receive buffers，cat 还新建 512 KiB full output。  
    **正确：**通信分片、gather buffers 与峰值生命周期都要计。

36. **错误：**“课程四个 `[1024,256]` blocks 是完整随机 $`W`$ 的不同 columns。”  
    **原因：**各 rank 同 shape、同 seed，实际 blocks 相同；local helper 还除 $`\sqrt{256}`$，而 full helper 会除 $`\sqrt{1024}`$，元素尺度相差 2 倍。  
    **正确：**真实验证需从一个 global $`W`$ 切不重叠 columns，或按 global indices 做缩放一致的分布式初始化。

37. **错误：**“源码写 `.backward()` 就会自动产生正确 TP 通信。”  
    **原因：**TP 源码根本未写 backward；裸 autograd 不知道自定义分片语义。  
    **正确：**实现相应 input/parameter gradients 与 collectives，或用已验证 TP 框架。

38. **错误：**“逐元素 GeLU 必须 gather 后才能算。”  
    **原因：**GeLU 不混合 columns，可对每个 local partial 独立作用。  
    **正确：**跨维度归约/normalization 才需额外分析。

39. **错误：**“TP degree 越大越快。”  
    **原因：**shard 后 GEMM 变小、collective 更多，慢链路会主导。  
    **正确：**在可用高速域和有效 kernel shape 内 benchmark degree。

### 17.7 Pipeline 与组合 mesh

40. **错误：**“切成 microbatches 会自动改变 global batch 的数学定义。”  
    **原因：**若 loss/gradient 正确累计，microbatch 只是 schedule/内存切分。  
    **正确：**明确累加、平均、optimizer step 的边界。

41. **错误：**“Microbatch 越多，pipeline 永远越快。”  
    **原因：**bubble 比例会降，但每批变小可能降低 GEMM 效率、增加 launch/通信次数。  
    **正确：**在 bubble 与单 microbatch 效率间调参。

42. **错误：**“$`U=m/(m+p-1)`$ 是任何 pipeline 训练的精确利用率。”  
    **原因：**它假设 forward-only、stage 等时、通信隐藏；真实 backward/1F1B/不均衡会改变。  
    **正确：**把它当最小模型，再用 stage timeline 测量。

43. **错误：**“课程代码已经实现 1F1B。”  
    **原因：**源码只有 forward、blocking send/recv，backward 留作作业。  
    **正确：**不能从 microbatch loop 推断完整训练 schedule。

44. **错误：**“PP 每 rank 生产运行时都需要先复制完整 input data。”  
    **原因：**这是课程简化；通常第一 stage/input pipeline 持有输入，后续 stages 接 boundary activations。  
    **正确：**避免把完整 data 无意义搬到后续 stages。

45. **错误：**“FSDP、TP、expert parallel 都是同一种 parameter 分片。”  
    **原因：**FSDP 按生命周期分片模型状态；TP 合作算 dense layer；expert parallel 把 tokens 路由到 experts。  
    **正确：**按数学轴、状态生命周期与通信原语分别描述。

46. **错误：**“8-GPU mesh 只建一个全局 group 就够。”  
    **原因：**TP、PP、DP 需要固定不同坐标轴的子 groups；全局 collective 会混合不该合并的 shards/stages。  
    **正确：**为每一 mesh 轴构造明确 process groups。

47. **错误：**“Rank 6 的所有通信伙伴都一样。”  
    **原因：**在本例 mesh 中，rank 6 的 TP group 是 `{6,7}`、PP group `{4,6}`、DP group `{2,6}`。  
    **正确：**操作前先确定它属于哪一轴的通信域。

48. **错误：**“选择并行策略只看峰值显存。”  
    **原因：**能 fit 仍可能被 communication、bubble、critical batch 或小 GEMM 拖慢。  
    **正确：**按 §16 做 correctness→fit→profile→placement→retest 的闭环。

49. **错误：**“当前 619 行 `setup` 已经调用 `torch.cuda.set_device(rank)`。”  
    **原因：**原码只初始化 process group；显式 device 仅在创建 tensor 时由 `cuda_if_available(rank)` 返回。  
    **正确：**`set_device(local_rank)` 是推荐修正，不是课程原行；缺它会使无参 synchronize 有等错 current device 的风险。

50. **错误：**“Trace 分支把 `world_size=4` 改成 1。”  
    **原因：**源码传 `fn(0, world_size,...)`，仍传原请求值 4，只把 distributed 函数替成 no-op。  
    **正确：**Trace 仍按 world-size-4 shape/index 控制流执行，但不代表真实四-rank 数值通信。

51. **错误：**“当前课程 reduce-scatter 的 input/output dtype 不一致。”  
    **原因：**原码明确给 `torch.arange` 写了 `dtype=torch.float32`；本文件未更改默认 dtype 时，`torch.empty(1, ...)` 也是 FP32。  
    **正确：**当前课程代码两者都是 FP32。`dtype=input.dtype` 只是通用防御性写法，不是修复当前 bug。

52. **错误：**“CPU/Gloo 路径调用 `torch.cuda.synchronize()`，所以也验证了 CUDA 同步。”  
    **原因：**无 CUDA 时源码把该函数 monkey-patch 成返回 `None` 的 no-op。  
    **正确：**它只让教学控制流可运行，不能产生 GPU/NCCL timing 证据。

---

## 18. 自测题（70 题；第 4–60、63–69 题含手算或填表）

> 建议先遮住 §19。标有“手算/填表”的题必须写中间步骤和单位，不要只看心算答案。

1. `process`、`rank`、`world_size` 各是什么？为什么 rank 不能直接定义成 GPU 编号？

2. `backend` 与 `process group` 分别解决什么问题？课程在 CUDA 和 CPU 情形各用什么 backend？

3. `MASTER_ADDR`、`MASTER_PORT` 的作用是什么？为什么 payload 不必全部经过 master？

4. **【填表】**单机 4 ranks，课程用 `cuda_if_available(rank)` 把新 tensor 放到 `cuda:rank`。写出 rank 0–3 的 tensor device；再说明这为什么不等于设置 current device，以及多机 global rank 13 为什么不能机械映射到 `cuda:13`。

5. **【手算】**四 ranks 初始标量为 2、4、6、8。以 rank 1 为 root 做 broadcast，写出每 rank 输出。

6. **【填表】**Root 持有 `[10,20,30,40]`，scatter 给四 ranks，每 rank 一个数。写输出表。

7. **【填表】**四 ranks 分别持有 `[10]`,`[20]`,`[30]`,`[40]`，gather 到 root 2。写 root 2 与其他 ranks 的逻辑输出。

8. **【手算】**四 ranks 分别持有 `[0,1]`,`[2,3]`,`[4,5]`,`[6,7]`。SUM reduce 到 root 0，算结果。

9. **【手算】**沿用第 8 题，若 op=AVG，结果是多少？

10. **【填表】**四 ranks 分别持有 `[0]`,`[1]`,`[2]`,`[3]`，all-gather 后每 rank 有什么？

11. **【手算/填表】**对 `[0,1,2,3]`、`[1,2,3,4]`、`[2,3,4,5]`、`[3,4,5,6]` 做 SUM reduce-scatter。先算完整 reduce，再写每 rank shard。

12. **【手算/填表】**沿用第 11 题，SUM all-reduce 后每 rank 输出什么？

13. **【手算】**用第 11 题结果验证 `reduce-scatter → all-gather` 与 all-reduce 逻辑等价。

14. **【填表】**四 ranks 的 all-to-all 输入行分别为 `[a00,a01,a02,a03]` 到 `[a30,a31,a32,a33]`；每个 `aij` 表示 sender $`i`$ 发给 destination $`j`$。写 rank 0 与 rank 2 的输出。

15. **【手算】**不均匀 all-to-all 中，四 senders 发往 rank 0 的元素数分别为 3、0、2、5。Rank 0 共收多少？为什么这不再等于“每列恰好四个元素”的转置例？

16. **【手算】**Cost model $`T\approx sL+Q/B`$。若 $`s=4`$、$`L=2\ \mu s`$、$`Q=16`$ MiB、$`B=8`$ GiB/s，估算 $`T`$（ms）。

17. **【手算】**沿用第 16 题，把 $`Q`$ 改为 8 KiB，其余不变。算 latency 项与 bandwidth 项各多少 $`\mu s`$，谁主导？

18. **【手算】**理想 ring all-reduce，$`p=4,S=16`$ MiB。每 rank 在 reduce-scatter 与 all-gather 两阶段各发送多少？总发送多少？

19. **【手算】**沿用第 18 题，aggregate sends 与 aggregate send+receive 各多少 MiB？

20. **【手算】**`num_elements=100*1024**2`。先算元素数，再按 FP32 4 bytes/element 算 bytes 与 MiB。

21. **【手算】**400 MiB all-reduce 用时 10 ms。算 `algbw`，分别用 GiB/s 和十进制 GB/s 表示。

22. **【手算】**$`p=4`$ all-reduce 的 busbw 校正因子是多少？用第 21 题算 `busbw` 的 GiB/s 与 GB/s。

23. **【手算】**课程 all-reduce 源码分子 `S*2*(p-1)`、分母 `p*t`。代 $`S=400`$ MiB、$`p=4`$、$`t=0.01`$s，逐项复算为何得到 58.59375 GiB/s。

24. **【概念+手算】**第 23 题中的 `p*t=0.04` 为什么不是实际 wall time？实际 wall time 是多少？

25. **【手算】**58.59375 GiB/s 换成 GB/s。提示：先乘 $`2^{30}`$ 得 bytes/s，再除 $`10^9`$。

26. **【手算】**Reduce-scatter 每 rank output chunk $`C=400`$ MiB、$`p=4`$。完整 input 是多少 MiB？

27. **【手算】**理想 ring reduce-scatter 中，每 rank send、每 rank receive、aggregate sends 各多少 MiB？

28. **【手算】**第 26 题用时 10 ms。按 NCCL-tests 口径算 reduce-scatter `algbw`（GiB/s）。

29. **【手算】**第 28 题乘 $`(p-1)/p`$，算 `busbw`（GiB/s）。

30. **【手算】**课程 reduce-scatter 源码分子 `input_bytes*(p-1)` 与分母 `p*t` 各是多少？复算第 29 题；再解释为什么课程实际 RS per-rank send 1200 MiB，反而是实际 AR 600 MiB 的 2 倍，而固定同一完整输入 $`S`$ 时 AR 又是 RS 的 2 倍。

31. 课程通信 benchmark 的正式 `duration` 包含哪三类主要时间？哪些准备工作在区间外？

32. `dist.barrier()` 与 `torch.cuda.synchronize()` 分别主要等待谁？再定义 CUDA stream、Work handle，并写出 `async_op=False` 与 `async_op=True→独立工作→wait→使用结果` 的最小顺序。

33. 写一个两-rank collective 次序不匹配的 deadlock 伪代码，并用一句话解释双方各等谁。

34. **【填表】**课程 all-reduce 前四 ranks 的 `data` 分别是什么？调用后同一个 `data` 变量分别是什么？它是 in-place 还是 out-of-place？

35. **【填表】**课程 reduce-scatter 的 input/output shape 各是什么？之后 all-gather 的 input/output shape 各是什么？设 world size=4。

36. **【手算/填表】**DP 的 global batch 128、world size 4。写每 rank 的 `[start,end)`、实际行号与 local batch size。

37. **【手算】**一个 `[32,1024]` FP32 local batch 是多少 bytes 和 KiB？

38. **【手算】**一个 `[1024,1024]` FP32 parameter matrix 有多少元素、bytes、MiB？

39. **【手算】**四层第 38 题矩阵的 parameters 总共多少 MiB？Gradients 同 shape 又是多少？

40. **【手算/填表】**每 rank 使用 FP32 AdamW，列 params、grads、$`m`$、$`v`$ 的大小并求合计；明确不含哪些至少三项。

41. **【填表】**DP local input `[32,1024]` 连续过四个 `[1024,1024]` 矩阵和逐元素 GeLU。写每层 matmul 前后 shape、loss shape、每个 `param.grad` shape。

42. **【手算】**两个 ranks，每 rank 两个 sample gradients：rank0 为 2、6；rank1 为 10、14。算两个 local means、AVG 后值与四样本 global mean。

43. **【手算】**Rank0 只有 gradient 2；rank1 有 6、10、14。算 local means、错误等权 AVG、正确按样本数加权值。

44. **【推理填表】**`get_init_params` 每次内部 `manual_seed(0)`，shape 都相同。填“跨 rank 同 layer”“同 rank 不同 layer”是否数值相同，并解释。

45. **【手算】**若 step0 backward 得 $`g_0=3`$，未 zero_grad；step1 backward 新贡献 $`g_1=5`$。第二次 `.grad` 是多少？若每步独立训练，本来应是多少？

46. **【填表】**要保证 DDP ranks 更新后 parameters 仍相同，旧 parameters、同步后 gradients、optimizer state/规则三项各应满足什么？

47. **【手算/填表】**TP 中 $`x[128,1024]`$、$`W[1024,1024]`$ 按 4 ranks 列切。写 local $`W_r`$、local output、all-gather 后 full output shape。

48. **【手算】**第 47 题 local output `[128,256]` FP32 是多少 KiB？四个 receive buffers 与 full concatenated output 各多少 KiB？

49. **【手算】**每 rank 每层 `[1024,256]` FP32 parameter shard 是多少 MiB？四层是多少？四 ranks 合计是多少？

50. **【手算】**用 §12.5 的 $`x,W`$，只计算 rank 0 的前两列输出，写出 2×2 结果。

51. **【手算】**用同一 $`x,W`$，计算 rank 1 的后两列输出。

52. **【手算】**把第 50、51 题沿 column concat；再直接检查完整 $`xW`$ 的第一行，验证相同。

53. **【推理】**为什么课程 `get_init_params(1024,256,rank)` 不能证明四 ranks 拿到一个随机完整 $`W`$ 的不同列？给一种概念上正确的初始化/切片办法。

54. **【手算】**PP 中 4 layers、2 stages。每 stage 几层？每层参数 4 MiB 时，每 stage parameters 和 FP32 Adam 训练状态下界各多少 MiB？

55. **【手算/填表】**Batch128 切 4 microbatches。写每份行号、shape；一个 `[32,1024]` FP32 boundary activation 多大？四次 send payload 多大？

56. **【填表】**$`m=4,p=2`$ forward-only pipeline。写 $`t_1`$ 到 $`t_5`$ 两 stages 分别处理哪个 microbatch/idle。

57. **【手算】**第 56 题总 slots、有用 slots、利用率、bubble fraction 各是多少？

58. **【手算】**理想公式 $`U=m/(m+p-1)`$。若 $`m=8,p=4`$，算利用率和 bubble fraction（百分比保留两位）。

59. **【手算】**若 $`p=4`$ 固定，希望理想 forward utilization 至少 80%，最小整数 $`m`$ 是多少？解不等式。

60. **【填表】**课程 pipeline 在 rank0 与 rank1 上分别执行 recv/compute/send 哪些步骤？列出它未实现的四个训练功能。

61. 比较 DP、TP、PP：各切什么；本讲各自主要通信是什么；哪一种有 pipeline bubble？

62. 解释 DDP、FSDP、ZeRO、TP 为什么不能当同义词。

63. **【手算/填表】**在 §15 的 $`2\times2\times2`$ mesh 中，rank 6 的 $`(d,t,p)`$ 是多少？它的 TP、PP、DP groups 分别是什么？

64. **【填表】**列出 $`2\times2\times2`$ mesh 的全部四个 TP groups、四个 PP groups、四个 DP groups。

65. **【手算】**DP degree=3、TP degree=2、PP degree=4，共需多少 GPUs？一个固定 $`(t,p)`$ 的 DP group 有几个 ranks？

66. **【手算】**训练状态 160 GiB，单卡 64 GiB。纯 DDP 能否 fit？若理想 TP=4 均分状态，每卡状态多少 GiB、剩余多少 GiB？

67. **【手算】**Cost model 中 $`s=3,L=5\ \mu s,Q=400`$ MiB、$`B=100`$ GiB/s。算总时间（ms）。

68. **【手算】**两个 DP ranks 的有效 token 数为 100 与 300，local mean gradients 为 2 与 6。错误等权 AVG 与正确 token-weighted global mean 各是多少？

69. **【手算/判断】**一个 TP layer 每 rank local partial 128 KiB，world size 4。忽略算法 overhead，每 rank 需要获得其他 ranks 多少 KiB 才能拥有完整 activation？完整 activation 多大？

70. 综合场景：单卡 OOM；TP=2 后能 fit，但 profiler 显示 TP all-gather 占 40% 且跨节点；PP 两 stages 又有 35% bubble。按 §16 写出至少四步有证据的优化顺序。

---

## 19. 自测答案（70 题逐题对应）

1. **Process** 是一个独立运行的程序实例；**rank** 是该 process 在某个 process group 中的编号；**world size** 是 group 中 rank 总数。Rank 是逻辑身份，不是硬件编号。课程恰好让每 process 控制一张 GPU，才可在单机写 `cuda:rank`；多机要用 local rank 选本地 device。

2. **Process group** 定义哪些 ranks 一起通信及它们的编号关系；**backend** 是执行 collective 的底层实现。课程有 CUDA 时用 NCCL，无 CUDA 的 CPU 教学路径用 Gloo。

3. `MASTER_ADDR` 与 `MASTER_PORT` 给 processes 一个共同 rendezvous 地址/端口，用来发现并建立 group。它们是控制/metadata 会合点；group 建好后，大 GPU payload 可沿 ranks 间 NVLink、PCIe、NIC 等数据路径直接传，不需把全部数据先集中到 master。

4. 单机课程的 tensor placement 映射：

   | rank | device |
   |---:|---|
   | 0 | `cuda:0` |
   | 1 | `cuda:1` |
   | 2 | `cuda:2` |
   | 3 | `cuda:3` |

   但 `cuda_if_available(rank)` 只把新 tensor 放到该显式 device，不会执行 `torch.cuda.set_device(rank)`；所以无参 `torch.cuda.synchronize()` 仍同步 current device，可能不是表中本 rank device。多机 global rank 13 可能在第二台 node 上，本机只有 `cuda:0..7`；应由 local rank，例如 5，映射 `cuda:5`，不能访问不存在的 `cuda:13`。

5. Root 是 rank 1，所以广播值为 4。输出：rank0=4、rank1=4、rank2=4、rank3=4。Root 自己也保留同一值。

6. Scatter 表：

   | rank | 输出 |
   |---:|---:|
   | 0 | 10 |
   | 1 | 20 |
   | 2 | 30 |
   | 3 | 40 |

7. Gather 到 root 2 后，root 2 得 `[10,20,30,40]`，顺序按 rank 0→3。其他 ranks 只作为 sender；普通 gather 的 API 语义不保证它们获得完整拼接结果。

8. 逐位置 SUM：

   $`0+2+4+6=12,`$

   $`1+3+5+7=16.`$

   Root 0 得 `[12,16]`。

9. 四 ranks AVG 就把第 8 题 SUM 除以 4：

   $`[12/4,16/4]=[3,4].`$

10. All-gather 后每个 rank 都有按 rank 顺序拼成的 `[0,1,2,3]`。

11. 逐列 reduce：

   $`[0+1+2+3,\ 1+2+3+4,\ 2+3+4+5,\ 3+4+5+6] =[6,10,14,18].`$

   再 scatter：rank0=`[6]`、rank1=`[10]`、rank2=`[14]`、rank3=`[18]`。

12. All-reduce 把第 11 题完整 reduce 结果给所有 ranks，所以 rank0、1、2、3 都是 `[6,10,14,18]`。

13. Reduce-scatter 先留下 `[6]`,`[10]`,`[14]`,`[18]`；all-gather 按 rank 顺序拼接为 `[6,10,14,18]`，并复制给所有 ranks。这与第 12 题 all-reduce 的每-rank 逻辑输出相同。

14. Destination 0 从每个 sender 取第 0 列，所以 rank0 输出 `[a00,a10,a20,a30]`。Destination 2 取第 2 列，所以 rank2 输出 `[a02,a12,a22,a32]`。

15. Rank0 收到：

   $`3+0+2+5=10\ \text{elements}.`$

   均匀 4×4 转置例假设每 sender 给每 destination 1 个元素，所以每列 4 个；这里 splits 是 3/0/2/5，不是固定一元素列。

16. Latency 项：

   $`sL=4\times2\ \mu s=8\ \mu s=0.008\ \text{ms}.`$

   Bandwidth 项先换单位：$`16`$ MiB $`=16/1024=0.015625`$ GiB。

   $`Q/B=0.015625/8\ \text{s}=0.001953125\ \text{s}=1.953125\ \text{ms}.`$

   总计：

   $`T\approx0.008+1.953125=1.961125\ \text{ms}.`$

17. $`8`$ KiB $`=8/1024/1024=1/131072`$ GiB。Latency 仍是 $`8\ \mu s`$。Bandwidth 项：

   $`\frac{1/131072}{8}\ \text{s} =\frac1{1{,}048{,}576}\ \text{s} \approx0.953674\ \mu s.`$

   因为 $`8>0.954`$，latency 项主导；总约 $`8.954\ \mu s`$。

18. 每阶段发送：

   $`\frac{p-1}{p}S=\frac34\times16=12\ \text{MiB}.`$

   Reduce-scatter 12 MiB，all-gather 12 MiB，总发送 $`12+12=24`$ MiB/rank。

19. Aggregate sends：

   $`4\times24=96\ \text{MiB}.`$

   对称 ring 每 rank 也接收 24 MiB，所以 endpoint send+receive：

   $`4\times(24+24)=192\ \text{MiB}.`$

20. 元素数：

   $`100\times1024^2 =100\times1{,}048{,}576 =104{,}857{,}600.`$

   Bytes：

   $`104{,}857{,}600\times4=419{,}430{,}400.`$

   MiB：

   $`419{,}430{,}400/1{,}048{,}576=400\ \text{MiB}.`$

21. $`400`$ MiB $`=400/1024=0.390625`$ GiB；$`10`$ ms $`=0.01`$s：

   $`\text{algbw}=0.390625/0.01=39.0625\ \text{GiB/s}.`$

   十进制：

   $`419{,}430{,}400/0.01/10^9=41.94304\ \text{GB/s}.`$

22. 校正因子：

   $`2(p-1)/p=2\times3/4=1.5.`$

   所以：

   $`39.0625\times1.5=58.59375\ \text{GiB/s},`$

   $`41.94304\times1.5=62.91456\ \text{GB/s}.`$

23. 分子：

   $`400\times2\times(4-1)=2400\ \text{MiB}.`$

   分母：

   $`4\times0.01=0.04\ \text{rank-seconds}.`$

   相除：

   $`2400/0.04=60{,}000\ \text{MiB/s},`$

   $`60{,}000/1024=58.59375\ \text{GiB/s}.`$

24. 四 ranks 并发执行，并未串行跑四次；`p*t` 是把所有 ranks 的 aggregate-send 分子归一化所用的 rank-seconds。每个 rank 实测 operation duration 仍是 $`t=0.01`$s=10ms，不是 40ms。

25. 先换 bytes/s：

   $`58.59375\times2^{30}=62{,}914{,}560{,}000\ \text{bytes/s}.`$

   再除 $`10^9`$：

   $`62{,}914{,}560{,}000/10^9=62.91456\ \text{GB/s}.`$

26. 完整 input 有 $`p`$ 个 output chunks：

   $`pC=4\times400=1600\ \text{MiB}.`$

27. 每 rank 发送：

   $`(p-1)C=3\times400=1200\ \text{MiB}.`$

   对称模型下每 rank receive 也是 1200 MiB。Aggregate sends：

   $`4\times1200=4800\ \text{MiB}.`$

28. $`1600`$ MiB $`=1600/1024=1.5625`$ GiB：

   $`\text{algbw}=1.5625/0.01=156.25\ \text{GiB/s}.`$

29. 校正因子 $`(p-1)/p=3/4=0.75`$：

   $`\text{busbw}=156.25\times0.75=117.1875\ \text{GiB/s}.`$

30. 分子：

   $`1600\times(4-1)=4800\ \text{MiB}.`$

   分母：$`4\times0.01=0.04`$ rank-seconds。于是：

   $`4800/0.04=120{,}000\ \text{MiB/s},`$

   $`120{,}000/1024=117.1875\ \text{GiB/s}.`$

   固定相同完整输入 $`S`$ 时，RS send $`=(p-1)S/p`$，AR send $`=2(p-1)S/p`$，所以 AR 是 RS 两倍。课程却给 AR 完整输入 400 MiB、RS 完整输入 1600 MiB；后者大 4 倍，抵消算法的 $`1/2`$ 后仍是 $`4/2=2`$ 倍，因此课程实际 RS 1200 MiB 是 AR 600 MiB 的两倍。

31. 按设计意图，正式区间包含 collective 的 host 提交/执行、正确 device 的 `cuda.synchronize` 等待、末尾 barrier 的 straggler 等待。区间外有 setup、输入随机分配、warmup collective、warmup 后同步与起跑 barrier；打印和 cleanup 也在 `end_time` 后。当前原码没 `set_device` 且 synchronize 无参数，可能同步错 current device，所以“正确 device 等待”是意图，不是已无条件满足的事实。

32. `dist.barrier()` 主要等 process group 其他 ranks 到达匹配屏障；无参 `torch.cuda.synchronize()` 主要等 current CUDA device 上此前提交到各 streams 的 kernels 完成。一个不自动替代另一个。课程没有 `set_device(rank)`，所以 rank 1–3 的无参同步还可能等错 device；安全实现应绑定 local device 或显式传 device。

   CUDA stream 是 GPU 的有序命令队列；Work handle 是 `async_op=True` 返回的可查询/等待异步工作的凭证。

   ```text
   async_op=False:
   collective成功排入stream → Python返回 → 同stream依赖操作按顺序执行

   async_op=True:
   work=collective → 做不依赖结果的工作 → work.wait/建立stream依赖 → 使用结果
   ```

   Different stream 没有 same-stream 天然顺序，必须用 Work/事件/stream wait 显式建立依赖。

33. 故意错误：

   ```python
   if rank == 0:
       dist.all_reduce(x)
   else:
       dist.barrier()
   ```

   Rank0 等 rank1 进入 all-reduce；rank1 等 rank0 进入 barrier，双方互等。

34. 调用前：rank0 `[0,1,2,3]`，rank1 `[1,2,3,4]`，rank2 `[2,3,4,5]`，rank3 `[3,4,5,6]`。调用后四者同一个变量 `data` 都被覆盖为 `[6,10,14,18]`，所以是 in-place。

35. Reduce-scatter：每 rank input `[4]`、output `[1]`。随后 all-gather：每 rank input `[1]`、预分配 output `[4]`；完成后每 rank output `[4]`。

36. Local batch：$`128/4=32`$。

   | rank | `[start,end)` | 实际行号 | local batch |
   |---:|---|---|---:|
   | 0 | `[0,32)` | 0–31 | 32 |
   | 1 | `[32,64)` | 32–63 | 32 |
   | 2 | `[64,96)` | 64–95 | 32 |
   | 3 | `[96,128)` | 96–127 | 32 |

37. 元素数 $`32\times1024=32{,}768`$。FP32 bytes：

   $`32{,}768\times4=131{,}072\ \text{bytes}.`$

   $`131{,}072/1024=128\ \text{KiB}.`$

38. 元素数：$`1024^2=1{,}048{,}576`$。Bytes：

   $`1{,}048{,}576\times4=4{,}194{,}304\ \text{bytes}.`$

   $`4{,}194{,}304/1{,}048{,}576=4\ \text{MiB}.`$

39. 四层 parameters：$`4\times4=16`$ MiB。每个 parameter 有同 shape gradient，所以 gradients 也为 16 MiB。

40. 分项：parameters 16 MiB、gradients 16 MiB、Adam $`m`$ 16 MiB、Adam $`v`$ 16 MiB；合计：

   $`16+16+16+16=64\ \text{MiB/rank}.`$

   不含 activations、allocator 保留/碎片、GEMM workspace、通信 buffers、CUDA context 等。Adam $`m,v`$ 通常在第一次 step 才懒分配。

41. 每层都是：

   $`[32,1024]@[1024,1024]\to[32,1024],`$

   GeLU 后仍 `[32,1024]`。四层都相同。`square().mean()` 得 scalar，shape `[]`。每个 `param.grad` 与 parameter 同 shape `[1024,1024]`。

42. Local means：

   $`(2+6)/2=4, \qquad (10+14)/2=12.`$

   两 rank AVG：$`(4+12)/2=8`$。Global mean：

   $`(2+6+10+14)/4=32/4=8.`$

43. Local means：rank0=$`2`$；rank1=$`(6+10+14)/3=30/3=10`$。错误等权 AVG：$`(2+10)/2=6`$。正确加权：

   $`\frac14\times2+\frac34\times10=0.5+7.5=8.`$

44. 在相同生成条件与 shape 下：

   | 比较 | 是否相同 | 原因 |
   |---|---|---|
   | 跨 rank、同 layer | 是 | 每次都从 seed 0 生成同 shape 序列 |
   | 同 rank、不同 layer | 也是 | 每调用一次 helper 都重新把 seed 设回 0 |

   第一项是 DP 需要的共同初值；第二项是课程简化/bug，不是一般模型初始化。

45. 默认 accumulation：第二次 `.grad` 为：

   $`g_0+g_1=3+5=8.`$

   若每 step 应独立且 step1 前已 zero_grad，第二次应只有 $`g_1=5`$。

46. 三条件都要对齐：

   | 状态 | 要求 |
   |---|---|
   | 旧 parameters | 各 ranks 对应参数值相同 |
   | 同步后 gradients | AVG/SUM 缩放后值相同 |
   | optimizer | $`m,v`$/step 等状态相同，超参数与更新规则相同 |

   缺任何一项都可能更新到不同参数。

47. 输出宽度 $`1024/4=256`$：

   $`W_r:[1024,256],`$

   $`[128,1024]@[1024,256]\to[128,256].`$

   All-gather 四份并沿 `dim=1` concat：`[128,1024]`。

48. Local output：

   $`128\times256\times4=131{,}072\ \text{bytes}=128\ \text{KiB}.`$

   四 receive buffers：$`4\times128=512`$ KiB。Full concat：

   $`128\times1024\times4=524{,}288\ \text{bytes}=512\ \text{KiB}.`$

49. 每层：

   $`1024\times256\times4=1{,}048{,}576\ \text{bytes}=1\ \text{MiB}.`$

   四层 $`=4`$ MiB/rank；四 ranks 合计 $`4\times4=16`$ MiB，等于完整四层 parameter bytes。

50. Rank0 拿 $`W`$ 前两列。第一输入行输出：$`[1,0]`$；第二输入行输出：$`[1,6]`$，所以：

   $`xW_0=\begin{bmatrix}1&0\\1&6\end{bmatrix}.`$

   例如第二行第二列：$`0\times0+1\times1+1\times1+2\times2=6`$。

51. Rank1 拿后两列：

   四个格分别是：

   $`1\times2+2\times1+0\times0+(-1)\times1=3,`$

   $`1\times1+2\times0+0\times2+(-1)\times1=0,`$

   $`0\times2+1\times1+1\times0+2\times1=3,`$

   $`0\times1+1\times0+1\times2+2\times1=4.`$

   $`xW_1=\begin{bmatrix}3&0\\3&4\end{bmatrix}.`$

52. Concat：

   $`[xW_0\mid xW_1] =\begin{bmatrix}1&0&3&0\\1&6&3&4\end{bmatrix}.`$

   完整 $`xW`$ 第一行四个 dot products：

   $`1\times1+2\times0+0\times1+(-1)\times0=1,`$

   $`1\times0+2\times1+0\times1+(-1)\times2=0,`$

   $`1\times2+2\times1+0\times0+(-1)\times1=3,`$

   $`1\times1+2\times0+0\times2+(-1)\times1=0.`$

   因而完整第一行 `[1,0,3,0]` 与 concat 第一行相同；第二行也为 `[1,6,3,4]`。

53. Helper 在每 rank 对同 shape 先设 seed 0，因此四个 blocks 数值相同，不是不同 columns。它还除以 $`\sqrt{256}=16`$；若按同 helper 先生成 full `[1024,1024]`，会除以 $`\sqrt{1024}=32`$。同一原始数 $`z`$ 的尺度比为 $`(z/16)/(z/32)=2`$，所以 local block 还大 2 倍。概念正确方法：只生成一次 global $`W[1024,1024]`$，给 rank $`r`$ 切 `W[:,r*256:(r+1)*256]`；规模大时用按 global index 可复现的分布式初始化直接生成互不重叠且缩放一致的 shards。

54. 每 stage：$`4/2=2`$ layers。Parameters：$`2\times4=8`$ MiB。若按 params+grads+$`m`$+$`v`$ 四份 FP32 状态：

   $`8\times4=32\ \text{MiB/stage}.`$

   课程 PP 同样反复调用内部 `manual_seed(0)` 的 helper，所以同 shape 的两层/两 stages 会重复初值；这不影响本题 bytes，却不是正常四层模型的不同参数。

55. 四份各 32 行：0–31、32–63、64–95、96–127，shape 都 `[32,1024]`。每份 bytes：

   $`32\times1024\times4=131{,}072\ \text{bytes}=128\ \text{KiB}.`$

   四次 send：$`4\times128=512`$ KiB。

56. 时间表：

   | 时刻 | stage0 | stage1 |
   |---:|---|---|
   | $`t_1`$ | mb0 | idle |
   | $`t_2`$ | mb1 | mb0 |
   | $`t_3`$ | mb2 | mb1 |
   | $`t_4`$ | mb3 | mb2 |
   | $`t_5`$ | idle | mb3 |

57. 总 slots：$`2\times5=10`$。有用 slots：$`4\times2=8`$。利用率 $`8/10=80\%`$；idle/bubble slots $`10-8=2`$，bubble fraction $`2/10=20\%`$。

58. 

   $`U=\frac8{8+4-1}=\frac8{11}\approx0.72727=72.73\%.`$

   Bubble fraction：

   $`1-U=3/11\approx27.27\%.`$

59. 解：

   $`\frac{m}{m+4-1}\ge0.8,`$

   $`m\ge0.8(m+3)=0.8m+2.4,`$

   $`0.2m\ge2.4,`$

   $`m\ge12.`$

   最小整数 $`m=12`$；检查 $`12/(12+3)=12/15=80\%`$。

60. Rank0：不 recv→本地两层 compute→send rank1。Rank1：recv rank0→本地两层 compute→不再 send。未实现至少：loss、backward、反向 gradient 通信、1F1B schedule、显式 communication/computation overlap、optimizer update；写任意四项即可。

61. DP 切 batch，主要做 gradient all-reduce；TP 切 layer width，本讲每层 activation all-gather；PP 切 depth，本讲相邻 stages send/recv boundary activations。PP 有 fill/drain pipeline bubble。

62. DDP 复制完整模型状态、只切 data；FSDP 分片 parameters/gradients/optimizer 并按需 gather/reshard；ZeRO 按阶段消除不同模型状态的 data-parallel 冗余；TP 切 layer 内矩阵并让 ranks 合作完成同一层。它们的状态生命周期与通信位置不同。

63. Rank 公式 $`r=4d+2p+t`$。对 rank6：$`d=1`$；余数 $`6-4=2`$，所以 $`p=1,t=0`$，坐标 $`(1,0,1)`$。TP group 固定 $`d=1,p=1`$、变 $`t`$：`{6,7}`。PP 固定 $`d=1,t=0`$、变 $`p`$：`{4,6}`。DP 固定 $`t=0,p=1`$、变 $`d`$：`{2,6}`。

64. 

   - TP：`{0,1}`, `{2,3}`, `{4,5}`, `{6,7}`；
   - PP：`{0,2}`, `{1,3}`, `{4,6}`, `{5,7}`；
   - DP：`{0,4}`, `{1,5}`, `{2,6}`, `{3,7}`。

65. 总 GPUs：

   $`3\times2\times4=24.`$

   固定 $`(t,p)`$、只改变 data coordinate，有 DP degree 3，所以该 DP group 有 3 ranks。

66. 纯 DDP 每 rank 复制 160 GiB，$`160>64`$，不能 fit。理想 TP=4：

   $`160/4=40\ \text{GiB/rank}.`$

   账面剩余：

   $`64-40=24\ \text{GiB}.`$

   还需验证 activations、buffers、碎片的实际峰值。

67. Latency：

   $`sL=3\times5=15\ \mu s=0.015\ \text{ms}.`$

   $`400`$ MiB $`=0.390625`$ GiB：

   $`Q/B=0.390625/100\ \text{s}=0.00390625\ \text{s}=3.90625\ \text{ms}.`$

   总计：

   $`T\approx3.90625+0.015=3.92125\ \text{ms}.`$

68. 错误等权 AVG：

   $`(2+6)/2=4.`$

   正确 token-weighted：总 tokens $`100+300=400`$：

   $`\frac{100}{400}\times2+\frac{300}{400}\times6 =0.5+4.5=5.`$

69. 其他三个 ranks：

   $`3\times128=384\ \text{KiB}.`$

   加上自己的 128 KiB，完整 activation：

   $`384+128=512\ \text{KiB}.`$

70. 一种有证据的顺序：

   1. 先保留 TP=2，因为 memory peak 已证明它解决 OOM，并复验数值正确；
   2. 根据 topology 证据把每个 TP pair 重排到同一 node/NVLink 域，再量 all-gather 占比；
   3. 若仍高，检查每层 tensor shape、TP degree 和 row/column layout，避免不必要 full gather，并再次 profile；
   4. 用 stage timeline 找到 35% bubble 来自填排空还是 stage imbalance；重新分 layers/调整 microbatch 数；
   5. 若增加 microbatch，检查 GEMM 是否因 batch 太小变慢、boundary 通信是否上升；
   6. 最终同时回归 peak memory、step throughput、collective time、bubble、straggler 与训练 loss。  
   核心是每个改动都由 peak-memory、topology、collective trace 或 stage timeline 证据触发。

---

## 20. 视频时间导航（人工英文字幕轨）

> 下列全部链接使用本讲前文尚未占用的人工字幕 cue 起点；点击即可跳转。它们是导航，不替代正文中的逐步推导。

| 时间 | 对应内容 | 推荐搭配笔记 |
|---|---|---:|
| [00:05](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=5s) | 开场 | §0 |
| [03:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=180s) | 单 GPU memory hierarchy 与多 GPU 通信权衡过渡 | §1、§6 |
| [05:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=359s) | 分布式 primitives 的历史背景 | §0、§3 |
| [07:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=479s) | Collectives 会反复出现在语言模型训练 | §3–5 |
| [09:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=599s) | Scatter 后各 GPU 做本地计算 | §3 |
| [14:02](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=842s) | Reduce-scatter 的逐位置向量例 | §4 |
| [15:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=959s) | All-reduce：reduce 后所有 ranks 都得到结果 | §4 |
| [18:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1080s) | All-to-all 按 destination columns 发送 | §5 |
| [20:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1200s) | Collective 名称与 associative/commutative reduce op 总结 | §3–5 |
| [22:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1321s) | 从 GPU 内部转向 networking | §6 |
| [24:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1440s) | 链路总 bandwidth 的课程时点比较 | §6 |
| [25:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1558s) | 跨更远层级通常更慢 | §6 |
| [27:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1679s) | GB200 NVL72 课程快照 | §6 |
| [29:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1798s) | 从硬件进入 `torch.distributed` 编程 | §7 |
| [32:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1920s) | GPU 与 NVSwitch topology 例 | §6–7 |
| [38:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2280s) | 开始走读 collective 代码 | §8 |
| [44:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2643s) | PyTorch collective 启动 CUDA kernels | §7–8 |
| [46:02](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2762s) | 用例子验证 all-reduce 分解 | §4、§8 |
| [47:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2879s) | Benchmark 停表位置 | §9 |
| [49:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2999s) | All-reduce 发送 bytes 分子 | §10 |
| [51:54](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3114s) | Reduce-scatter timings | §9–10 |
| [54:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3241s) | 课堂问答：CUDA operation 默认异步 | §8.6–8.7、§9 |
| [56:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3361s) | Data parallel 切 data | §11 |
| [57:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3478s) | 实践中每 rank 应直接加载自己的 data | §11.3 |
| [59:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3599s) | DDP 的 local training + gradient average 主线 | §11.8–11.9 |
| [61:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3719s) | DDP 不关心 forward 是 MLP 还是 Transformer | §11.9 |
| [64:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3840s) | TP 每 rank 负责部分 dimensions | §12.1 |
| [65:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3958s) | All-gather activation buffers | §12.2–12.3 |
| [69:59](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4199s) | Pipeline 中 data 怎样经过 stages | §13 |
| [72:02](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4322s) | `recv` 与 `send` 的含义 | §13.5–13.6 |
| [73:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4438s) | Pipeline communication/computation overlap | §13.9 |
| [75:57](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4557s) | 组合不同 parallelism 技术 | §14–15 |
| [77:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4678s) | 选型还受更多训练条件影响 | §14、§16 |
| [79:57](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4797s) | Store/recompute/communicate 的统一视角 | §15.3 |
| [80:49](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4849s) | 下讲继续更深入的并行技术 | §22 |

---

## 21. 来源、619 行覆盖、图片与测试边界

### 21.1 课程原始来源与版本

- **官方课程代码讲义：**[`lecture_07.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)。
- **官方 Stanford Online 视频：**[Lecture 7: Parallelism I](https://www.youtube.com/watch?v=SzpOcwdIL0Y)。
- **人工字幕：**YouTube `English (United States)`，语言代码 `en-US`，`kind` 为空；1312 segments，末 cue 80:54，视频约 80:57。自动轨存在，但没有作为笔记主字幕。
- **版本：**2026-08-28 核验 GitHub 当前页面与仓库提交，`lecture_07.py` 为 **619 个物理行**。文件最近提交 `0be5c6121acb3ce2cef5ec1cad1a0b7ebc8d2012`（2026-04-20，`update lecture 7`），检查时仓库 HEAD `8b59b50730766695c2ffedd1a79c50cd09b9eb91`。
- **旧缓存差异：**抓取工具曾返回 556 行旧 raw 视图；本笔记没有混用该版本，源码映射以 619 行当前版本为准。

### 21.2 官方 619 行连续覆盖表

> 下表是 range index：从 1 到 619 首尾相接，无 gap/overlap。它证明所有行都有去向；**不表示每一个 import、展示 helper 和 print 字符都值得在正文逐字符重复**。关键数学/通信代码已逐行解释，课程展示基础设施在相应边界框说明。

| 官方行段 | 内容 | 笔记位置 |
|---:|---|---:|
| 1–18 | Imports、展示工具、无 CUDA 同步替身 | §0、§7、§21.5 |
| 19–74 | `main`、通信层级、并行策略地图与总结 | §0–1、§6、§14–16、§22 |
| 75–138 | Collective 定义；broadcast/scatter/gather/reduce | §2–3 |
| 139–184 | All-gather/reduce-scatter/all-reduce | §4 |
| 185–208 | All-to-all 与术语记忆法 | §5 |
| 209–238 | 互连、RDMA/RoCE/NCCL | §6–7 |
| 239–286 | `torch.distributed` 与三种 workhorse collectives | §7–8 |
| 287–337 | All-reduce benchmark | §9–10 |
| 338–374 | Reduce-scatter benchmark | §9–10 |
| 375–389 | Data-parallel 入口与结论 | §11、§14 |
| 390–396 | `[128,1024]` 样例数据 | §11.2–11.3 |
| 397–438 | Manual data-parallel training loop | §11 |
| 439–446 | Tensor-parallel 入口 | §12、§14 |
| 447–483 | Column TP forward/all-gather、backward 占位 | §12 |
| 484–491 | Pipeline-parallel 入口 | §13–14 |
| 492–537 | 两-stage pipeline、microbatches、send/recv | §13 |
| 538–556 | Setup（未 `set_device`）、process group、cleanup | §2、§7 |
| 557–576 | `DisableDistributed` trace context manager | §7.6 |
| 577–593 | `spawn` 的真实多进程与 trace rank0/原 world-size 分支 | §2、§7.5–7.6 |
| 594–619 | 初始化/整除/摘要/时间 helpers、程序入口 | §7、§11.10、§12.6、§13、§21.5 |

### 21.3 六个不同引用图片、七次 `image(...)` 调用

所有 5 个本地图片均以原分辨率打开视觉核验；外链 Springer 图的新下载当时被拒，但官方仓库提交保存了课程运行缓存，已从 Git object 恢复并按原始分辨率核验。

| 不同资产 | 调用次数 | 像素 | 视觉核验结论 |
|---|---:|---:|---|
| `gpu-node-overview.png` | 2 | 2469×1381 | 4 GPU，各自含 SM/register/L1/shared/L2/HBM；NVLink→NVSwitch→外部网络 |
| `ranks.png` | 1 | 813×114 | 四个并列框 Rank 0–3 |
| Springer `Fig1` 外链缓存 | 1 | 685×350 | 两服务器，各有 RAM、2 CPU、10 GPU；GPU 经 PCIe，服务器经 Ethernet |
| `data-parallelism.png` | 1 | 625×889 | 完整 layers，按 data 轴切 |
| `tensor-parallelism.png` | 1 | 699×897 | 每层按 width 纵切 |
| `pipeline-parallelism.png` | 1 | 697×865 | 在 layers 之间按 depth 横切 |
| **合计** | **7** | **6 个不同资产** | `gpu-node-overview.png` 被重复用于两处 |

目录中的 `siglip-parallelism.png` 也已打开，但当前 619 行源码没有引用，故不虚报为第七个课程引用资产。

### 21.4 一手补充来源

本终稿的来源边界按以下标签理解：

- **【课程】**官方 619 行讲义中直接给出的代码、公式、图与结论；
- **【视频补充】**人工字幕中老师的口头解释、演示与课堂问答；
- **【补充解释】**本笔记为零基础读者增加的逐步 shape、单位和代数桥梁；
- **【补充】**本笔记自建的可手算矩阵、反例、决策树与自测；
- **【延伸】**为说明生产边界而引用的当前官方 PyTorch/NCCL/硬件文档。延伸不冒充课程原话。

- [PyTorch stable distributed 官方文档](https://docs.pytorch.org/docs/stable/distributed.html)：process group、backend、collective、barrier 与同步/异步语义。
- [PyTorch multiprocessing 官方文档](https://docs.pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses)：`mp.spawn` 入口、参数、join 与异常传播。
- [`torch.cuda.synchronize` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)：设备同步边界。
- [NCCL 2.31.2 collective 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)：collective 语义、参与 ranks 的匹配条件。
- [NCCL-tests 官方性能说明](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)：`algbw` 与 `busbw` 的 collective-specific 校正口径。
- NVIDIA 官方 A100/H100/HGX/DGX/GB200 文档：用于核对 §6 硬件表；具体链接已贴在该表对应行。

这些补充只用于解释 API 边界、单位和硬件规格；没有把第三方笔记当课程内容，也没有假装课程每句话逐字来自这些文档。

### 21.5 课程快照与本环境测试边界

- A100/H100/B200/GB200、NVLink 与 network 数字是 **Spring 2026 讲义/当时官方 SKU 的课程时点快照**，不是同系列所有产品或未来产品的永久常数。
- PyTorch/NCCL API 行为按 2026-08-28 当前 stable 文档核对；未来版本可能变化。
- 本环境没有可用于四-GPU NCCL 实测的分布式 GPU runtime；因此没有声称运行 `mp.spawn(world_size=4)`、NCCL collectives 或测得带宽。
- 数学向量、DP gradient average、TP 小矩阵、PP timeline、bytes/单位与 mesh groups 已由独立 CPU/Python 脚本复算。
- 课程仓库保存的 `var/traces/lecture_07_stdout.txt` 只作为**课程运行快照**交叉核对 6/10/14/18 与当时 timings；不是本机实测。
- 教学代码缺陷仍是终稿结论的一部分：setup 未绑定 current CUDA device、无参 synchronize 可能等错卡；CPU 路径把 synchronize 变为 no-op；trace 仍传请求的 world size 但通信 no-op；DP 先复制完整 data、反复重置 seed、仅一步且缺 `zero_grad`；TP shards 重复且缩放与 full-helper 切片相差 2 倍、无 backward；PP 同 shape layers 也重复初值，且只有 blocking forward、无 backward/1F1B/显式 overlap。当前 reduce-scatter 的 input 显式为 FP32，output 在本文件默认 dtype 下也是 FP32，不把它列为缺陷。

---

## 22. 一页复习流程与学完后的能力清单

### 22.1 看到 collective，按这四问

```text
1. 每个 rank 输入是什么 shape/value？
2. 是复制、拼接、reduce、scatter，还是任意 sender→receiver 路由？
3. 每个 rank 最终拿完整结果、一个 shard，还是只有 root 拿？
4. 逻辑 API 下面采用什么算法/topology？必须靠实现证据确认。
```

必须能手算：

```text
[0,1,2,3]
[1,2,3,4]
[2,3,4,5]
[3,4,5,6]
逐列 SUM → [6,10,14,18]
```

### 22.2 看到通信性能，按这五问

```text
1. 分子是 payload、per-rank send、aggregate sends，还是 send+receive？
2. 分母是纯 device event、CPU wall time，还是含 barrier/straggler？
3. 单位是 GB/s 还是 GiB/s？
4. 小消息是否被 steps×latency 支配？
5. Ring 只是模型，还是这次 profiler/NCCL 日志确认的实际算法？
```

400 MiB、4-rank、10 ms 复习锚点：

| 操作 | Per-rank sends | `algbw` | normalized `busbw` |
|---|---:|---:|---:|
| Ring all-reduce | 600 MiB | 39.0625 GiB/s | 58.59375 GiB/s |
| Ring reduce-scatter，input 1600 MiB | 1200 MiB | 156.25 GiB/s | 117.1875 GiB/s |

表中 RS sends 比 AR 大 2 倍，是因为课程 RS 完整输入 1600 MiB、AR 只有 400 MiB；若固定同一个完整输入 $`S`$ 比算法，AR 的两阶段 sends 才是 RS 的 2 倍。

### 22.3 看到并行策略，先画切分轴

```text
DP：切 batch rows
    完整模型 × 多 replicas
    local backward → gradient all-reduce

TP：切 layer width
    partial matmul → layer 内 collective
    高频通信，通常偏爱高速域

PP：切 model depth
    microbatch → stage send/recv
    要处理 bubble、stage balance、forward/backward schedule
```

### 22.4 学完后应能独立完成

- 从输入表手算 broadcast/scatter/gather/reduce/all-gather/reduce-scatter/all-reduce/all-to-all；
- 区分 logical collective 与 ring/tree/hierarchical physical algorithm；
- 解释 process/rank/world size/group/backend/device 的边界；
- 解释 barrier、CUDA synchronize、`async_op` 和 deadlock；
- 从 tensor 元素数推 bytes、MiB/GiB、per-rank/aggregate traffic；
- 审计课程 all-reduce/reduce-scatter 带宽公式与 `GB/s` 标签错误；
- 从 batch128 写出 DP 四 rank slices，并证明等 batch AVG gradient 等于 global mean；
- 算出 DDP params/grads/Adam 状态 64 MiB/rank，并指出未计项；
- 从 `[128,1024]@[1024,256]` 追踪 TP partial、gather、cat 的 shape/bytes；
- 用 2×4 与 4×4 小矩阵验证列切 concat 等于完整 matmul；
- 画 $`m=4,p=2`$ pipeline timeline，算 80% utilization/20% bubble；
- 为 2×2×2 device mesh 列出 DP/TP/PP 三类子 groups；
- 识别课程代码的 data、seed、zero-grad、TP-shard、backward、overlap 局限；
- 按 correctness→fit→profile→topology/placement→retest 做并行选型，而不是背一条万能规则。

### 22.5 最终一句话

> 多 GPU 训练不是“把同一段代码复制八遍”；它是先决定**切哪一个数学轴**，再用正确的 communication semantics 让 shards 合成同一个训练结果，最后依据 topology、memory peak 和 timeline 证据把通信与空闲成本压下去。

> **状态：**Lecture 7 主编撰终稿已完成，但仍等待 Beginner Reviewer；这里未宣称审核通过。
