# CS336 Lecture 5：GPUs、TPUs 与数据移动

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Tatsunori Hashimoto  
> 视频：[Lecture 5: GPUs, TPUs](https://www.youtube.com/watch?v=izZba4UA7iY)（约 78 分钟）  

> 来源边界：**【课程】** 表示官方 lecture PDF 中的内容；**【视频补充】** 表示讲师在视频中的口头说明；**【补充解释】** 表示为零基础读者补出的中间推导；**【补充】** 表示额外小例子；**【延伸】** 表示来自官方文档或论文、但并非讲师逐字讲授的知识。
> 讲义：[lecture_05.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)（55 页）

这不是逐字字幕，而是一份可以替代视频学习的课程重构讲义。正文保留课程的教学顺序、关键例子、课堂问答和数值，同时补上幻灯片中省略的推导。标有“补充”的内容不是老师在本讲中的原话，而是为了让概念闭环而增加的背景知识。

---

## 0. 五分钟复习卡

> **第一次学习请先跳到第 1 节。** 这一节是“学完以后快速回忆”的卡片，所以会提前出现尚未解释的术语。看完正文再回来，它才会真正有用。

### 0.1 一句话主线

现代 GPU 的矩阵计算能力增长得比显存带宽更快，因此许多机器学习程序真正稀缺的不是 FLOPs，而是把数据送到计算单元的能力；高性能算法的核心是**少搬数据、连续搬数据、把搬来的数据重复使用**。

### 0.2 全讲知识链

```text
更大的语言模型需要更多计算
        ↓
单核频率增长放缓，转向大规模并行
        ↓
GPU：大量轻量线程 + 专用矩阵乘单元
        ↓
计算增长快于内存带宽，出现 memory wall
        ↓
控制分歧 / 低精度 / 融合 / 重计算 / 合并访存 / 分块
        ↓
用同一组原则解释矩阵乘性能波动
        ↓
用 tiling + fusion + online softmax + recomputation 得到 FlashAttention
```

### 0.3 必须记住的三个公式

先给复习时用的最短符号表：FLOP 是一次浮点数学操作；byte（字节）是数据量单位，1 byte = 8 bits；FLOP/s 是每秒能做多少次浮点操作；`min(a,b)` 表示在 $`a,b`$ 中取较小者。Attention 公式里的 $`Q/K/V`$ 分别是 query（查询）、key（键）和 value（值），$`K^\top`$ 表示把 $`K`$ 的行列交换，$`d_k`$ 是每个 key 的数字个数，softmax 把一排分数变成总和为 1 的权重。这些概念会在第 8 节从零手算。

1. **算术强度（arithmetic intensity）**

   $`I=\frac{\text{FLOPs}}{\text{从内存传输的字节数}}`$

2. **Roofline 模型**

   $`P_{\text{attainable}}=\min\left(P_{\text{peak}},\;B_{\text{memory}}\cdot I\right)`$

   - $`P_{\text{peak}}`$：硬件峰值计算吞吐，单位通常是 FLOP/s。
   - $`B_{\text{memory}}`$：内存带宽，单位 byte/s。
   - $`I`$：每搬一个 byte 能做多少 FLOPs。

3. **Attention**

   $`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`$

   FlashAttention 不改变这个数学结果；它改变的是计算顺序和数据移动方式。

### 0.4 六种性能技巧

| 技巧 | 解决的问题 | 核心动作 |
|---|---|---|
| Control divergence | 一个 warp 内不同线程走不同分支，硬件利用率下降 | 尽量让同一 warp 执行相同路径 |
| Low precision | 每个数占用的字节太多，矩阵单元吞吐也受限 | 用 BF16、FP8 等降低传输和计算成本 |
| Operator fusion | 每个小算子都往返 HBM | 把多个算子合成一个 kernel |
| Recomputation | 为反向传播保存大量 activation | 丢弃部分中间结果，反向时重算 |
| Coalescing | 一个 warp 的访存地址分散，产生很多事务 | 让相邻线程访问连续或同一内存段 |
| Tiling | 同一数据被反复从 HBM 读取 | 分块搬入 shared memory 后重复使用 |

### 0.5 最容易混淆的五点

- GPU 追求的是**总吞吐（throughput）**，不是单个任务的最低延迟。
- 显存“有 80 GB/144 GB”说的是容量；程序快不快往往受**带宽和访问模式**影响。
- shared memory 和 L1 cache 都很快，但前者由程序员显式管理，后者主要由硬件自动管理。
- 矩阵维度取 16、32 的倍数不是“二的幂有魔法”，而是经常能更好地匹配 transaction、tile 和硬件并行度。
- FlashAttention 是**精确 attention**，不是用近似换速度。

---

## 1. 前置知识：为什么语言模型课程要讲 GPU？

本讲是课程从“神经网络结构”进入“系统”部分的转折点。后续的 kernel、并行训练和推理优化，都建立在本讲的硬件心智模型之上。

### 1.1 Compute 是语言模型扩展的货币

在当前的 scaling 范式下，更多有效计算通常意味着更大的模型、更多训练 token 或更多推理时计算。获得更多有效计算有四条常见路径：

1. 更快的硬件；
2. 更高的硬件利用率；
3. 更多芯片；
4. 更好的并行和通信方案。

系统优化的意义不是让同一个模型“优雅一点”，而是让有限预算能够完成更多训练或推理。

把它换成一个具体例子。假设训练一次模型原本需要 100 天：

| 改进 | 简化计算 | 新时间 |
|---|---:|---:|
| 芯片速度变成 2 倍 | $`100/2`$ | 50 天 |
| 硬件有效利用率从 40% 提到 80% | 有效工作速度也约变成 2 倍 | 50 天 |
| 用 4 张 GPU，且暂时假设没有额外损耗 | $`100/4`$ | 25 天 |

现实中的 4 张卡通常达不到恰好 4 倍，因为它们需要互相传数据并等待对方。所谓“通信方案”，就是决定哪些数据由哪张卡计算、何时交换，以及怎样让通信与计算同时发生。整门系统课都可以理解为：尽量把“买到的理论计算量”变成“真正完成的模型计算量”。

### 1.2 为什么不能一直提高 CPU 主频？

先解释四个词：

- **晶体管**：芯片里极小的电子开关；许多开关组合起来才能做加法、判断和存储。
- **缩小晶体管**：把这些开关做得更小，于是同样面积能放更多开关。
- **频率**：芯片每秒推进多少个时钟节拍；频率越高，单个工人动作越快。
- **功耗密度**：每一小块面积产生多少热。它太高，芯片就无法安全散热。

过去一段时间，晶体管缩小后，单个开关需要的电能也会下降，所以芯片能同时“装更多开关”和“跑得更快”，却不至于热得无法使用，这常被概括为 Dennard scaling。后来漏电、功耗和散热不再按同样的好比例缩小。继续一味提高频率，就像让一个已经冒烟的工人跑得更快：电费和热量先失控。

于是扩展方向从：

```text
让一个执行单元越来越快
```

转向：

```text
让大量执行单元同时工作
```

GPU 正是这一并行扩展路线的代表。

---

## 2. GPU 的硬件心智模型

### 2.1 CPU 与 GPU：延迟优先和吞吐优先

| 维度 | CPU | GPU |
|---|---|---|
| 优化目标 | 单个线程尽快完成，低延迟 | 同一时间完成尽可能多的总工作，高吞吐 |
| 执行单元 | 数量较少、单核复杂 | 数量很多、单元相对轻量 |
| 控制逻辑 | 强，擅长分支、预测、复杂控制流 | 相对弱，更偏好规则、重复的工作 |
| 缓存 | 大量芯片面积用于复杂缓存层级 | 更多面积用于计算单元和带宽 |
| 典型任务 | 操作系统、串行逻辑、复杂服务控制 | 矩阵乘、卷积、图形渲染、大规模数据并行 |

一个 GPU kernel 的单个线程未必比 CPU 线程更快。GPU 的优势来自成百上千个线程共同产生的总吞吐。

用搬箱子来区分两个目标：

- CPU 像 4 个很能干的师傅。每个箱子交给师傅后 1 分钟就能搬完。第一个箱子的等待时间，也就是 **latency（延迟）**，很短。
- GPU 像 100 个动作较简单的工人。每人搬一个箱子也许要 2 分钟，但 2 分钟能完成 100 箱，即每分钟 50 箱。单位时间完成的总量，也就是 **throughput（吞吐）**，很大。

只有 1 个箱子时 CPU 可能更合适；有 10 万个彼此独立的箱子时 GPU 更有优势。矩阵中大量元素的乘加正是这种规则、重复、可并行的工作。

### 2.2 SM、执行单元与 Tensor Core

可以把 GPU 想象成一个由许多小工厂组成的园区：

- **SM（Streaming Multiprocessor）**：相对独立的计算与调度单元，可看作 GPU 的主要工作站。
- **ALU（算术逻辑单元）/ FP unit**：真正执行加、乘、比较等指令的小机器。FP 是 floating point（浮点数）。
- **Tensor Core**：专门加速小块矩阵乘加（matrix multiply-accumulate）的硬件。
- **Warp scheduler**：选择下一组要运行的线程。

“标量”是一个数，例如 3；“向量”是一排数，例如 $`[3,5,7]`$。普通 FP unit 擅长若干独立的数值运算；Tensor Core 一次处理规则的小矩阵块。所谓**矩阵乘加**可以写成：

```math
D=A\times B+C
```

例如：

```math
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5&6\\7&8\end{bmatrix}
+\begin{bmatrix}1&1\\1&1\end{bmatrix}
=\begin{bmatrix}20&23\\44&51\end{bmatrix}
```

左上角是 $`1\times5+2\times7+1=20`$。真实 Tensor Core 处理的 tile 更大，但思想相同。

其余三格同样按“行乘列，再加 C”：

```math
D_{01}=1\times6+2\times8+1=23
```

```math
D_{10}=3\times5+4\times7+1=44,\qquad
D_{11}=3\times6+4\times8+1=51
```

一份工作从软件到硬件的大致路线是：

```text
一个 thread 有一小份任务
        ↓
32 个 threads 组成一个 warp
        ↓
warp scheduler 选择一个已经准备好的 warp
        ↓
它的指令送到普通 ALU/FP unit 或 Tensor Core
        ↓
结果先放寄存器，必要时再写回显存
```

Tensor Core 的出现让 matmul 成为“被硬件特别优待”的操作。课程中的量级判断是：矩阵乘吞吐可以比普通浮点操作高一个数量级。因此，未来可扩展的模型结构通常需要把主要计算组织成大而规则的矩阵乘。

### 2.3 GPU 的内存层次

从快而小到慢而大，可以建立如下近似心智模型：

```text
每线程 registers
        ↓
每个 SM 内的 shared memory / L1 cache
        ↓
芯片上的 L2 cache
        ↓
GPU 板上的 HBM / global memory
        ↓
CPU host memory 或其他设备
```

重要区别：

- **Registers**：每个线程私有，最快、最少。寄存器压力过高还会降低同时驻留的线程数。
- **Shared memory**：位于 SM 上，由同一 block 的线程共享，并由 kernel 显式装载和管理。
- **L1/L2 cache**：由硬件自动缓存近期或重复访问的数据。
- **Global memory / HBM**：容量大，但一次访问的等待时间和能耗远高于片上存储。
- **Host memory**：CPU 侧内存；GPU 内存不足时可以 offload，但跨设备传输通常更慢。

一个 **cycle（时钟周期）** 是 GPU 节拍器的一拍。课程引用的示意数据中，L1/shared memory 读取可能只需几十拍，而 global memory 可能慢约一个数量级。具体数字会变，应该记住的是“层级差异很大”。

现在跟踪一次 `A[0]` 读取：

1. thread 发出“我要 `A[0]`”的请求。
2. 硬件先查离计算单元很近的 cache。若这里已有副本，叫 **cache hit（命中）**，很快返回。
3. 若没命中，就继续向 L2，最后可能去芯片外的 HBM 找。**片上/on-chip** 是 GPU 芯片内部；**片外/off-chip** 是芯片外部。箭头向下不表示数据真的掉下去，只表示通常容量更大、距离更远、等待更久。
4. 等待数据的 warp 暂时不能继续，但 SM 不必发呆。scheduler 可以切换到另一个已经准备好的 warp。这叫 **latency hiding（隐藏延迟）**：没有消灭等待，只用别的工作填住等待时间。
5. 数据回来后，该 warp 再继续。

如果 GPU 内存装不下，程序还可以把数据 **offload（转移暂存）** 到 CPU 内存；需要时再传回。但 PCIe/设备互连通常比 HBM 更慢，所以这是容量救急，不是免费加速。

### 2.4 为什么不把整块芯片都做成 SRAM？

一个 **bit（比特）** 只能表示 0 或 1；8 bits 组成 1 byte。shared memory/cache 通常由 SRAM 实现，HBM/global memory 使用 DRAM 家族的技术。

可以把 SRAM 想成每个工位旁边的小储物柜：伸手就能拿，但柜子小而贵。DRAM/HBM 像园区大仓库：能放很多箱子，每平方米便宜，但来回运送更慢。SRAM 快，却有三类成本：

- 每 bit 占用更大的芯片面积，因此昂贵；
- 为了低延迟，它必须靠近计算单元，物理布线困难；
- SRAM 保存状态需要持续供电，能耗高。

所以现实设计不是“选择快内存或慢内存”，而是建立层次，然后让软件尽量复用少量快内存。

---

## 3. GPU 的执行与编程模型

### 3.1 Thread、block、warp、grid

| 概念 | 直觉 | 与硬件/内存的关系 |
|---|---|---|
| Thread | 执行一份 kernel 的最小逻辑工作者 | 有自己的索引、寄存器和状态 |
| Warp | NVIDIA GPU 上通常由 32 个线程组成的调度组 | SM 以 warp 为关键执行/调度单位 |
| Block | 一组线程 | 一个 block 被调度到一个 SM；block 内可共享 shared memory、进行同步 |
| Grid | 一次 kernel launch 的全部 blocks | blocks 可分布到多个 SM 上 |

最重要的映射是：

```text
grid
 ├─ block 0 → 某个 SM
 │   ├─ warp 0 → 32 threads
 │   └─ warp 1 → 32 threads
 └─ block 1 → 另一个或稍后空闲的 SM
```

用“把两个长度为 8 的向量相加”走一遍。要算：

```math
[1,2,3,4,5,6,7,8]+[10,20,30,40,50,60,70,80]
```

我们启动 8 个 threads，每个 thread 只负责一个位置：thread 0 算 $`1+10`$，thread 1 算 $`2+20`$，……，thread 7 算 $`8+80`$。假设每个 block 最多放 4 个 threads，那么：

```text
一次 kernel launch = 一个 grid
grid 有 2 个 blocks
block 0: threads 0,1,2,3
block 1: threads 4,5,6,7
```

每个 thread 怎样知道自己该算哪格？一维情形最常用：

```math
\text{global index}=\text{block id}\times\text{block size}+\text{thread index inside block}
```

所以 block 0 的局部编号 0–3 得到全局编号 0–3；block 1 得到 $`1\times4+[0,1,2,3]=[4,5,6,7]`$。

> **不要把 block size 和 warp size 混在一起。** 真实 NVIDIA warp 通常仍固定为 32 threads。若一个 block 只有 4 threads，它会占用一个只有 4 个 active lanes 的 warp，其余 28 lanes 空闲。上图只画 4 格是为了省略空位，不是硬件 warp 真缩成 4。后文的“8-lane 迷你 warp”也只是教学模型。

必须分清两句话：

- **一个 block 在一次执行期间不会跨越两个 SM。** 所以它的线程能共同使用该 SM 上的 shared memory，并用同步点互相等齐。
- **一个 SM 可以容纳不止一个 block。** 如果寄存器和 shared memory 够用，它们可同时驻留；资源不够时则先后执行。一个 block 太“胖”会挤掉其他 block，降低可同时准备的工作量。

### 3.2 SIMT 不等于“所有线程永远执行完全相同的事”

SIMT 是 Single Instruction, Multiple Threads，意思是“一条指令，许多线程各自处理自己的数据”。可以把 8 个 lane 想成 8 个座位：同一拍大家收到“做加法”的指令，但每个座位里的数字不同。**lane** 就是 warp 中的一个执行位置。

如果 warp 内线程遇到不同的条件分支，硬件常要分别执行各路径，并用 **active mask（活动遮罩）** 标记这一阶段哪些 lane 的结果有效。这就是 **warp divergence（分歧）**。简单条件有时会变成 **predication**：指令仍执行，但只让 mask 为真的 lane 写结果。**向量化**则是让一条指令处理一排规则数据。现代 GPU 有更灵活的独立线程调度，但它不让两条昂贵分支凭空免费；“同一 warp 内分支越不一致，吞吐往往越差”仍是正确直觉。

例如：

```python
# 概念示意，不是完整 CUDA kernel
if x[i] > 0:
    y[i] = x[i]
else:
    y[i] = 0
```

若一个 warp 的 32 个元素中，一半大于 0、一半小于 0，两条路径都可能需要执行。相比之下，规则的逐元素表达式通常更容易被编译为高效的 predication 或向量化代码。

> 不要把它误读成“GPU 代码绝对不能有 `if`”。真正的问题是 warp 内是否发生严重且昂贵的分歧，以及分支内有多少工作。

---

## 4. TPU：另一条相似的加速器演化路线

**TPU（Tensor Processing Unit）** 是 Google 为张量/机器学习计算设计的专用加速器。这里的 tensor 可以先理解成“多维数字表”；向量是一维表，矩阵是二维表。

TPU 的代表性核心是 **systolic array（脉动阵列）**：数字像接力棒一样按固定节奏从一个小乘加单元流向相邻单元。每个单元接到 $`A`$ 的一项和 $`B`$ 的一项，做一次“乘后累加”，再把数据传下去。这样同一数据在阵列内部被反复使用，不必每次回大内存。

TPU 与 GPU 在高层结构上非常相似：

- 都有轻量控制；
- 都有很强的矩阵乘单元；
- 都有片上快内存与片外大内存的层次；
- 都必须认真管理数据移动。

主要差异在于资源配置与网络设计：

- GPU 通常有更多、较小、较灵活的并行单元；
- TPU 往往以更少、更大的矩阵乘单元为中心，更强地假设工作负载是规则的大 matmul；
- 多芯片互连和编译栈是 GPU/TPU 实际体验差异的重要来源，会在并行课程中继续展开。

可用同一个矩阵乘对比：

```text
GPU：许多较小且灵活的工作组，用不同 kernel/tile 完成任务
TPU：把规则的大矩阵乘送入少数很大的脉动阵列流水线
```

用 $`A=\begin{bmatrix}1&2\\3&4\end{bmatrix}`$、$`B=\begin{bmatrix}5&6\\7&8\end{bmatrix}`$ 想象一个 $`2\times2`$ 小阵列。四个位置各自累积一个输出：

| 阵列位置 | 依次收到的乘法 | 累积结果 |
|---|---|---:|
| 左上，负责 $`C_{00}`$ | $`1\times5`$，再 $`2\times7`$ | 19 |
| 右上，负责 $`C_{01}`$ | $`1\times6`$，再 $`2\times8`$ | 22 |
| 左下，负责 $`C_{10}`$ | $`3\times5`$，再 $`4\times7`$ | 43 |
| 右下，负责 $`C_{11}`$ | $`3\times6`$，再 $`4\times8`$ | 50 |

在真实脉动阵列里，A 的数通常横向传，B 的数纵向传，并故意错开到达时间；例如 1 可沿行继续供右边位置使用，5 可沿列继续供下面位置使用。管线填满后，许多位置每拍都在乘加。上表展示数据复用关系，不宣称是精确逐拍时序。

因此，规则、尺寸足够大的 matmul 很适合二者；大量小分支、不断变化的形状更难喂满专用大阵列。**编译栈**负责把模型变成硬件可执行的分块与指令，**互连**负责多颗芯片交换数据。即使单颗芯片算得一样快，编译和互连不同，实际训练体验仍会差很多。

### 术语陷阱：Tensor Core

- 在 NVIDIA GPU 语境中，Tensor Core 通常指 SM 内的矩阵乘单元。
- 在 TPU 的一些资料中，tensor core 可能指更高层的一整个处理核心。

看到 “tensor core” 时必须先确认硬件语境。

---

## 5. Memory wall 与 Roofline 模型

### 5.1 为什么 FLOPs 很多仍然可能很慢？

程序执行一项运算，至少要经历：

1. 从某级内存取得输入；
2. 在计算单元上运算；
3. 把结果保存到某级内存。

如果第 1、3 步耗时远大于第 2 步，再增加 ALU/Tensor Core 也没用。它们只会更长时间地等待数据。

近年的趋势是计算吞吐增长快于 HBM 带宽和设备间通信带宽，因此越来越多工作负载变成 memory-bound 或 communication-bound。这解释了为什么本讲后半段几乎一直在讨论数据移动。

#### 直觉类比：厨房为什么会“有很多厨师却出不了菜”？

把 GPU 想象成一家餐厅：

- Tensor Cores/ALUs 是厨师；
- HBM 是远处的仓库；
- 内存带宽是仓库到厨房的送货通道；
- arithmetic intensity 是每送来一箱食材，可以做出多少道菜。

如果一箱食材只能做一道菜，厨师很快就会等货，这是 memory-bound。若一箱食材能在厨房内重复使用并做出上百道菜，送货不再是主要问题，厨师会全忙起来，此时更可能 compute-bound。

这个类比也解释了为什么“增加更多厨师”不一定有用：如果送货通道没有变宽，新增厨师只会一起等待。

### 5.2 Roofline 的两段

```math
P=\min(P_{\text{peak}},B\cdot I)
```

先检查单位，这一步能防止把公式背反：

```math
\underbrace{B}_{\text{byte/s}}\times
\underbrace{I}_{\text{FLOP/byte}}
=\frac{\text{byte}}{\text{s}}\times\frac{\text{FLOP}}{\text{byte}}
=\underbrace{\text{FLOP/s}}_{\text{每秒运算量}}
```

分子、分母里的 byte 抵消了，所以 $`B\cdot I`$ 能和峰值计算速度比较。这里使用十进制单位：$`1\ \text{TB/s}=10^{12}\ \text{byte/s}`$，$`1\ \text{TFLOP/s}=10^{12}\ \text{FLOP/s}`$。

- 当 $`B\cdot I<P_{\text{peak}}`$ 时，程序处于**内存受限区**。增加数据复用、减少字节数、改善访存才有效。
- 当 $`B\cdot I\ge P_{\text{peak}}`$ 时，程序处于**计算受限区**。此时计算单元已经饱和，继续提高算术强度不再增加吞吐。

两条线的交点称为 ridge point：

```math
I_{\text{ridge}}=\frac{P_{\text{peak}}}{B}
```

Roofline 图的横轴是算术强度 $`I`$，越往右表示每搬 1 byte 能多算几次；纵轴是实际可达到的速度 $`P`$，越往上越快。屋顶为何先斜后平，可以画成：

```text
性能 P (TFLOP/s)
100 |                    ● I=100   ● I=250
    |                  ┌──────────────── 计算上限：再复用也超过不了 100
 50 |          ● I=50 /
    |                /
 10 |  ● I=10      /   内存上限：P = B×I
  0 +-------------+--------------------------> 算术强度 I
                 100
                 ridge point
```

在下面这台 $`B=1`$ TB/s、$`P_{peak}=100`$ TFLOP/s 的假想 GPU 上：

| $`I`$ (FLOP/byte) | 带宽允许的 $`B\times I`$ | 再和 100 取较小值 | 最终上限 |
|---:|---:|---:|---:|
| 10 | 10 TFLOP/s | $`\min(100,10)`$ | 10 TFLOP/s |
| 50 | 50 TFLOP/s | $`\min(100,50)`$ | 50 TFLOP/s |
| 100 | 100 TFLOP/s | $`\min(100,100)`$ | 100 TFLOP/s |
| 250 | 250 TFLOP/s | $`\min(100,250)`$ | 100 TFLOP/s |

#### 补充例子：用数字判断瓶颈

假设 GPU 峰值为 100 TFLOP/s，显存带宽为 1 TB/s：

```math
I_{\text{ridge}}=\frac{100\times10^{12}}{1\times10^{12}}=100\ \text{FLOP/byte}
```

- kernel 的 $`I=10`$ FLOP/byte 时，最多约 $`10`$ TFLOP/s，是 memory-bound。
- kernel 的 $`I=250`$ FLOP/byte 时，带宽允许 250 TFLOP/s，但硬件峰值只有 100，所以是 compute-bound。

#### 补充例子：三个算子的算术强度为什么差很多？

下面都用 FP32，并暂时忽略 cache、索引和 launch 开销。

**例 1：向量加法 $`z=x+y`$**

每个元素需要读取 $`x,y`$ 共 8 bytes，写出 $`z`$ 需要 4 bytes，做一次加法：

```math
I_{\text{add}}\approx\frac{1}{12}=0.083\ \text{FLOP/byte}
```

它几乎注定是 memory-bound。即使 ALU 再快十倍，也不能让 12 bytes 更快到达。

**例 2：ReLU**

读取、写回共约 8 bytes，做一次比较/选择：

```math
I_{\text{ReLU}}\approx\frac{1}{8}=0.125\ \text{FLOP/byte}
```

这也是典型的 memory-bound pointwise operation，因而非常适合与邻近算子融合。

> 把 ReLU 近似记成 1 FLOP 只是为了建立数量级。严格说，比较/选择未必按硬件计数规则算作一个浮点加法或乘法；这不是性能合同，只是说明“它搬 8 bytes，却只做极少工作”。

**例 3：$`N\times N`$ 方阵乘**

先用 $`2\times2`$ 看懂计数。设 $`C=AB`$：

```math
C_{00}=A_{00}B_{00}+A_{01}B_{10}
```

这里有 2 次乘法和 1 次加法。工程上通常把每一个“乘加”近似计为 2 FLOPs，所以每个输出约 $`2N`$ FLOPs。矩阵共有 $`N^2`$ 个输出：

```math
N^2\times 2N=2N^3\ \text{FLOPs}
```

当 $`N=2`$ 时是 $`2\times2^3=16`$ FLOPs；若把每格最后少一次加法严格扣掉会略少，但大矩阵里近似足够好。

理想情况下，整个 $`A`$ 读一次、整个 $`B`$ 读一次、整个 $`C`$ 写一次。每个矩阵有 $`N^2`$ 个 FP32 数，每个数 4 bytes，因此：

```math
\underbrace{4N^2}_{\text{读 A}}+
\underbrace{4N^2}_{\text{读 B}}+
\underbrace{4N^2}_{\text{写 C}}
=12N^2\ \text{bytes}
```

于是：

```math
I_{\text{matmul}}\approx\frac{2N^3}{12N^2}=\frac{N}{6}\ \text{FLOP/byte}
```

当 $`N=1024`$ 时：

```math
I\approx\frac{1024}{6}=170.67\approx171\ \text{FLOP/byte}
```

矩阵越大，搬来的元素被重复用于越多乘加，因此 matmul 更容易走到 Roofline 的水平部分。

> 这里的 $`12N^2`$ bytes 是理想最小数据流量，也就是流量下界；因此 $`N/6`$ FLOP/byte 是算术强度的乐观上界。若数据不能留在 cache/shared memory、矩阵被重复读入，分母会增大，真实算术强度会更低。算术强度既由算法决定，也由实现的数据复用决定。

#### 补充例子：同一个 kernel 换一块 GPU，瓶颈会改变

考虑算术强度为 150 FLOP/byte 的 kernel：

| 假想硬件 | 峰值计算 | 带宽 | Ridge point | 该 kernel 的判断 |
|---|---:|---:|---:|---|
| GPU A | 100 TFLOP/s | 1 TB/s | 100 FLOP/byte | compute-bound，最多 100 TFLOP/s |
| GPU B | 300 TFLOP/s | 1.5 TB/s | 200 FLOP/byte | memory-bound，最多 225 TFLOP/s |

GPU B 的 225 不是猜出来的：

```math
1.5\times10^{12}\ \text{byte/s}\times150\ \text{FLOP/byte}
=225\times10^{12}\ \text{FLOP/s}=225\ \text{TFLOP/s}
```

它低于计算屋顶 300 TFLOP/s，所以最终取 225。

同一份代码在 GPU A 上继续减少内存流量可能收益不大；到了计算能力增长更快的 GPU B 上，反而再次被内存卡住。这就是为什么性能结论必须同时说明**算法、实现和硬件**。

#### Roofline 诊断后的行动

这张表是工程索引；第一次学习只需先掌握前两行。其余术语的最短解释如下：

- **occupancy（驻留度）**：一个 SM 能同时保留多少可运行的 warps；等待内存时，有更多候补 warp 才容易隐藏延迟。
- **kernel launch**：CPU 向 GPU 下达“运行这个并行函数”的一次命令；小任务里命令本身的固定成本会显眼。
- **batch**：一次合并处理的样本数；batch 大常能摊薄固定成本。
- **profiler**：性能检查工具，记录时间花在哪、带宽和计算单元用了多少。
- **CUDA Graph**：把一串固定 GPU 工作预先记录后重复提交，减少每次 launch 的 CPU 开销。
- **communication overlap**：一部分数据在卡间传输时，同时计算另一部分，尽量不让两者串行等待。

| 判断 | 优先尝试 | 通常不应首先做什么 |
|---|---|---|
| Memory-bound | fusion、低精度、tiling、coalescing、减少中间量 | 只增加理论 FLOPs 峰值 |
| Compute-bound | 使用 Tensor Core、改善占用率、减少无效 FLOPs、选更合适的 matmul kernel | 只优化已经很小的 HBM 流量 |
| Launch/框架开销明显 | fusion、CUDA Graph、批处理更大的工作 | 只分析 FLOP/byte |
| Communication-bound | 更好的并行切分、重叠通信与计算、减少通信量 | 只优化单卡 kernel |

### 5.3 讲义中的单位纠正

讲义第 25 页把 ReLU 例子写成 “Intensity: 8 bytes/FLOP”。老师在视频中随即指出：这其实是算术强度的倒数。

对 FP32 ReLU，若近似计为一次 4-byte 读取、一次 4-byte 写入和一次操作：

```math
\text{bytes per FLOP}=8,\qquad I=\frac{1}{8}=0.125\ \text{FLOP/byte}
```

换成 FP16 后，传输约 4 bytes：

```math
\text{bytes per FLOP}=4,\qquad I=\frac{1}{4}=0.25\ \text{FLOP/byte}
```

所以低精度把算术强度提高了一倍。

---

## 6. 让 GPU 变快的六种技巧

### 6.1 技巧一：减少 control divergence

一个 warp 内的线程若分成两条控制路径，两条路径通常需要分阶段执行。假设分支 A 需要 10 个 cycle，分支 B 需要 20 个 cycle：

- 所有线程都走 A：约 10 cycle；
- 所有线程都走 B：约 20 cycle；
- 同一 warp 一部分走 A、一部分走 B：可能接近 30 cycle，同时每一阶段都有部分 lane 空闲。

实用思路包括：

- 让相近数据落到相同分支；
- 用规则的 mask/predication 表达简单条件；
- 把工作重排，使一个 warp 内任务尽量同质；
- 不要为了消灭一个很便宜的分支而引入更昂贵的计算或内存访问，最终仍要 benchmark。

#### 例子一：用 8 个 lane 看见浪费

**lane** 是 warp 中的一个执行座位；**cycle** 是硬件的一拍。为了便于画图，假设一个“迷你 warp”只有 8 个 lanes，输入符号如下：

```text
lane:   0  1  2  3  4  5  6  7
x>0?:   是 是 否 否 是 否 是 否
```

代码有两个分支：正数路径做 2 步，非正数路径做 3 步。简化执行过程是：

```text
阶段 A（2 步）: lanes 0,1,4,6 工作；2,3,5,7 空闲
阶段 B（3 步）: lanes 2,3,5,7 工作；0,1,4,6 空闲
总时间约 5 步，但每一步只有一半 lanes 有效
```

如果每一步本来可以让 8 个 lane 工作，那么 5 步总共有 $`8\times5=40`$ 个“lane-step”座位。真正做有效工作的只有：

```math
4\text{ 个 lane}\times2\text{ 步}+4\text{ 个 lane}\times3\text{ 步}=20
```

所以这个简化例子的执行效率是：

```math
\frac{20}{40}=50\%
```

如果先按条件把数据分组，让一个 warp 全是正数、另一个全是非正数，两组各自不再 divergence。不过，分组本身需要读写和调度成本，只有当分支足够昂贵、数据会重复使用时才值得。

#### 例子二：LLM 中的 MoE routing

> 这是一个进阶例子；只想掌握分歧概念可以先跳过。

Mixture-of-Experts（MoE）可以想成有多个“专家小网络”，每个 token 只送给其中少数专家。假设 8 个 token 被送到 2 个专家：

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expert id | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |

若这 8 个 token 恰好在一个 warp，先跑 expert 0 时只有 token 0、3、5、6 工作；再跑 expert 1 时只有 1、2、4、7 工作。两阶段都空一半。

若先重排成 `[0,3,5,6]` 和 `[1,2,4,7]`，一个工作组全做 expert 0，另一个全做 expert 1，路径和权重访问都更规则。高性能实现常按下面流程做：

1. 先计算每个 token 的 expert id；
2. 按 expert 对 token 分桶或排序；
3. 对每个 expert 形成较规则的矩阵乘；
4. 最后把输出还原到原 token 顺序。

这同时改善了 control coherence、内存连续性和 matmul 规模。代价是额外的 routing/sort/all-to-all，因此小 batch 下未必划算。

#### 反例：用 `where` 不一定更快

```python
y = torch.where(mask, expensive_a(x), expensive_b(x))
```

`mask` 是一排真/假选择，例如 `[真, 假, 真, 假]`。`where` 最后在每个位置从两排结果中选一个，但 Python 参数通常要先算好：如果 `expensive_a` 需 10 次操作，`expensive_b` 也需 10 次，那么程序可能先做完 $`10+10=20`$ 次，再选择；它不是只做被选中的 10 次。消除了显式 `if`，却可能把计算量翻倍。正确目标不是“代码里没有 `if`”，而是让硬件执行的总工作和空闲 lane 都尽可能少。

### 6.2 技巧二：低精度计算

低精度有两种收益：

1. 每个数占用更少字节，降低内存流量并提高缓存容量；
2. Tensor Core 对低精度格式通常具有更高的计算吞吐。

#### 先补齐：数字怎样放进计算机？

- **dtype（数据类型）** 规定每个数用多少 bits、怎样解释这些 bits。FP32 用 32 bits，BF16/FP16 用 16 bits，FP8 用 8 bits。
- 浮点格式可以粗略看成“符号 + 指数 + 有效数字”。符号管正负，指数管能表示多大/多小的范围，有效数字（mantissa/significand）管细节有多精。
- bits 固定时，给指数更多位置，范围通常更大，但留给有效数字的位置变少，精度下降。

十进制类比：若只能保留两位有效数字，$`1.234`$ 会被舍入成 $`1.2`$，$`1234`$ 可能写成 $`1.2\times10^3`$。它还能表示“大约一千二百”，却丢失了 34。低精度浮点也会把真实数舍入到附近可表示的格点，例如某个玩具格式可能把 $`1.234`$ 存成 $`1.25`$。

**tensor** 就是规则排列的一堆数。**quantize（量化）** 是把高精度数映射成少 bit 的整数/浮点编码；**dequantize（反量化）** 是用 scale 把编码解释回近似实数。最简单形式是：

```math
q=\mathrm{round}(x/s),\qquad \hat x=q\times s
```

例如 scale $`s=0.1`$，$`x=1.26`$：$`q=\mathrm{round}(12.6)=13`$，还原后 $`\hat x=13\times0.1=1.3`$。误差是 $`0.04`$。scale 太大，格子太粗；scale 太小，大数可能超出可表示范围。

#### Mixed precision 的基本模式

实际训练很少把所有东西无脑变成低精度。常见模式是：

- 权重和 activation 以 BF16/FP16/FP8 参与 matmul；
- 部分累加器使用 FP32；
- softmax、归一化、损失或数值敏感操作保留更高精度；
- 输出再按需要 cast。

难点不是“会不会把 dtype 改小”，而是判断**哪些操作可以降精度且不破坏稳定性**。

#### FP8：指数范围与有效数字的权衡

- E4M3：4 个 exponent bits、3 个 mantissa bits，精度相对更好，范围较小。
- E5M2：5 个 exponent bits、2 个 mantissa bits，范围更大，精度更低。

当 bit 数很少时，单一格式难以覆盖所有 tensor 的分布。

在同一个指数区间内，可以粗略想成：M3 用 3 bits 把区间切成约 $`2^3=8`$ 份，M2 只切成约 $`2^2=4`$ 份，所以像 1.2 这种数，E4M3 通常能找到更近的格点。反过来，E5 多一个 exponent bit，可覆盖更大的数量级；当数非常大时，E4M3 可能溢出，而 E5M2 仍能表示一个较粗的近似值。真实 FP8 的保留值、NaN/无穷和舍入规则依具体标准而定，不能只用这张粗略切格图替代格式规范。

#### MXFP8 的 block scaling

课程介绍的 MXFP8 使用多个缩放因子，而不是整个 tensor 共用一个 scale。例如每 32 个数对应一个低精度 scale。这样局部区域可以适配自己的数值范围，但带来新问题：

用只有四个数的玩具例子看原因：$`[0.1,0.2,100,120]`$。假设低精度编码只能用整数 $`-127`$ 到 $`127`$。

**整组共用一个 scale：** 为了装下 120，可取 $`s=120/127\approx0.945`$。

- $`0.1/0.945\approx0.11`$，舍入后为 0，还原后也是 0；
- $`0.2/0.945\approx0.21`$，舍入后为 0，还原后也是 0；
- $`120/0.945\approx127`$，可以还原成约 120。

大数保住了，两个小数却全消失。

**分成 $`[0.1,0.2]`$ 和 $`[100,120]`$ 两组：** 第一组可用 $`s_1=0.2/127\approx0.00157`$，0.1 会编码成约 64，不再变成 0；第二组仍用 $`s_2\approx0.945`$。这就是 block scaling：不同局部范围用不同尺子。真实 MX 格式的 scale 和编码规则更具体，本例只解释“为什么分组”。

- 转置后，原来的 32 元素分组方向改变；
- 因而训练实现可能同时维护原矩阵和转置矩阵各自量化后的副本；
- quantize/dequantize 和 scale 统计本身有成本，所以 FP8 matmul 的理论吞吐提升不会原样变成端到端加速。

课程给出的经验量级是：真实训练的**端到端总时间**可能改善约 20%-30%，不是说每个矩阵乘都只快这么多，也不是硬件理论峰值必然完整兑现。量化/反量化、统计 scale、不能降精度的层、通信和其他 kernel 都会吃掉收益，具体仍取决于矩阵大小和实现。

MXFP4 更极端：可表示值非常稀疏，因此需要更细粒度的 scale。它说明了一个普遍规律：bit 越少，系统设计、统计估计和训练稳定性的复杂度越高。

#### 例子一：十亿参数只看权重需要多少空间？

先只计算模型权重，不包括 gradient、optimizer state、activation 和 allocator 开销：

| 格式 | 每参数字节 | 10 亿参数的十进制容量 | 直觉 |
|---|---:|---:|---|
| FP32 | 4 | 4 GB | 基准 |
| BF16/FP16 | 2 | 2 GB | 容量与理想带宽需求减半 |
| FP8 | 1 | 1 GB | 再减半，但需要 scale 和更复杂的数值处理 |

若显存带宽恰为 1 TB/s，而且一次 decode step 必须把全部权重完整流过一次，那么：

```math
\frac{4\ \text{GB}}{1000\ \text{GB/s}}=0.004\ \text{s}=4\ \text{ms}
```

同理 2 GB 需 2 ms，1 GB 需 1 ms。这是只读权重的理想下界。这里 GB 是十进制 $`10^9`$ bytes；GiB 是二进制 $`2^{30}`$ bytes，$`1\ \text{GiB}\approx1.074\ \text{GB}`$。真实系统还受 cache、并行、kernel 和通信影响，但这个估算解释了低精度为什么对 memory-bound inference 很有吸引力。

> 训练显存不能用“参数量 × dtype”直接估完，因为还可能存在高精度 master weights、gradients、Adam 的一阶/二阶状态和各种 activation。

#### 例子二：KV cache 的 dtype 为什么影响长上下文？

自回归模型每生成一个新 token，都要让新 query 与过去 token 的 key/value 交互。为了不把过去全部重新计算，会保存每层的 K 和 V，这就是 **KV cache**。

- layer：Transformer 重复堆叠的一层；
- head：一组独立做 attention 的小通道；
- head dimension：每个 head 的 K/V 各含多少个数；
- context length：当前保留多少个 token；
- batch size：同时处理多少条序列。

先算极小模型：2 层、3 个旧 token、每层 2 个 KV heads、每个 head 2 个数、batch=1、FP16/BF16 每数 2 bytes。容量为：

```math
\underbrace{2}_{K\text{ 和 }V}\times
\underbrace{2}_{\text{层}}\times
\underbrace{3}_{\text{token}}\times
\underbrace{2}_{\text{heads}}\times
\underbrace{2}_{\text{每 head 数字}}\times
\underbrace{2}_{\text{bytes}}=96\ \text{bytes}
```

考虑一个用于建立量级直觉的 dense multi-head attention 模型：

- 32 层；
- 32 个 heads；
- 每个 head dimension 为 128；
- context length 为 32,768；
- batch size 为 1；
- 同时保存 K 和 V。

BF16 KV cache 的容量约为：

```math
2\times32\times32768\times32\times128\times2\ \text{bytes}=16\ \text{GiB}
```

把大数也验一遍：

```math
2\times32\times32768\times32\times128\times2
=17{,}179{,}869{,}184\ \text{bytes}
```

```math
\frac{17{,}179{,}869{,}184}{2^{30}}
=\frac{17{,}179{,}869{,}184}{1{,}073{,}741{,}824}
=16\ \text{GiB}
```

若能安全使用 1-byte 格式，理想容量约降到 8 GiB。实际模型可能使用 GQA（多个 query heads 分组共享 K/V）或 MQA（所有 query heads 共享更少的 K/V heads），从结构上减少 KV heads；这说明低精度和架构设计可以共同缓解内存瓶颈。

#### 例子三：为什么累加常保留高精度？

dot product 会把许多乘积相加。如果每个乘积有很小的舍入误差，累加 1024、4096 次后误差可能放大；当当前累加值很大时，低精度格式甚至无法表示“再加一个很小的数”的变化。

因此常见做法是低精度输入进入 Tensor Core，但 partial sums 使用 FP32 或其他更高精度格式。这是 mixed precision 的核心思想：把低精度放在收益最大的地方，把高精度留给误差最容易累积的地方。

一个夸张玩具格式只能保留 4 位有效数字：累加器已有 1000，再加 0.1，精确结果是 1000.1，但 4 位只能保存成 1000，小增量消失。连续加十次 0.1，若每次都立刻舍入，结果可能仍是 1000；若高精度累加到 1001 后再舍入，至少保住了总变化。真实格式细节不同，但“许多小误差会在长 reduction 中积累”就是高精度累加的原因。

### 6.3 技巧三：Operator fusion

**operator（算子）** 是一次数学操作，例如加法、sin 或 GELU；**pointwise（逐元素）** 表示数组每个位置独立做同一操作。**kernel** 是一次送到 GPU 并行执行的函数；**kernel launch** 是 CPU 发出这次运行命令。

老师使用了“仓库与工厂”的类比：

- HBM 是远处的仓库；
- SM/Tensor Core 是工厂；
- 内存总线是运输带。

若每做一道小工序就把半成品运回仓库，运输会压倒生产。

例子：

```math
y=\sin^2(x)+\cos^2(x)
```

数学恒等式告诉我们，对任何 $`x`$，$`\sin^2(x)+\cos^2(x)=1`$。这里故意不把它直接化简为 1，因为要用它观察一串逐元素算子怎样产生中间数组。

朴素动态图可能产生：

```text
sin → square ┐
             ├→ add
cos → square ┘
```

若每个节点是独立 kernel，就会多次读取、写回 HBM，并承担多次 kernel launch。融合后，一个 kernel 可以读取一次 $`x`$，在寄存器或 shared memory 中完成所有逐元素操作，再写回一次 $`y`$。

按“完整数组读/写一次”计，未融合版本可能是：

1. `sin`：读 $`x`$，写 $`s`$；
2. `square`：读 $`s`$，写 $`s^2`$；
3. `cos`：再读 $`x`$，写 $`c`$；
4. `square`：读 $`c`$，写 $`c^2`$；
5. `add`：读 $`s^2`$、读 $`c^2`$，写 $`y`$。

合计 6 次数组读取、5 次数组写入、5 次 launch。融合后只需读 $`x`$ 一次、写 $`y`$ 一次，并在寄存器中保留单个元素的临时值。

```python
import torch

def f(x):
    return torch.sin(x) ** 2 + torch.cos(x) ** 2

compiled_f = torch.compile(f)
```

普通 **dynamic/eager graph（动态图/立即执行）** 会边运行 Python 边逐个提交算子；编译器先看到整段计算图，才有机会把相邻算子合并。`torch.compile`/TorchInductor、JAX/XLA 等可以自动完成许多 pointwise fusion；复杂融合仍可能需要 Triton/CUDA kernel 或库实现。

> Fusion 不是越多越好。每个 thread 需要的临时数太多，会产生**寄存器压力**；寄存器不够时，数据可能 **spill（溢出）** 到更慢的内存。一个 block 占用资源太多，也会降低 **occupancy**，即同一 SM 能同时准备的 warps 数量。

#### 例子一：`bias + GELU` 能省多少内存流量？

`bias` 是给每个位置加的偏置；GELU 是 Transformer 常用的平滑门函数，负数被压小，较大正数大致保留。先看四个数：

```math
x=[-1,0,1,2],\qquad b=[1,1,1,1]
```

先加 bias 得 $`t=x+b=[0,1,2,3]`$，再做 GELU，约得：

```math
y\approx[0,\ 0.841,\ 1.955,\ 2.996]
```

未融合时，完整的中间数组 $`t`$ 会写回 HBM，随后又读回来；融合时，每个 thread 算出一个 $`t_i`$ 后立即算 GELU，不必保存整排 $`t`$。

设 $`x,b,y`$ 都有 $`N`$ 个 FP16 元素，并把 bias 简化为同形状张量。

**未融合：**

1. bias add 读取 $`x,b`$：$`4N`$ bytes；
2. 写出中间量：$`2N`$ bytes；
3. GELU 读中间量：$`2N`$ bytes；
4. 写出 $`y`$：$`2N`$ bytes。

总计约 $`10N`$ bytes。

**融合：** 读取 $`x,b`$ 后在寄存器中立即做 GELU，只写最终 $`y`$：

```math
4N+2N=6N\ \text{bytes}
```

这个简化例子把 HBM traffic 降低约 40%。若 $`N=10^8`$，传输量从约 1.0 GB 降到 0.6 GB。对 memory-bound pointwise chain，这种差异很可能直接变成可见的加速。

#### 例子二：Transformer 中常见的 fusion 机会

以下是进阶索引，不要求第一次全部记住：

- matmul **epilogue（尾处理）**：矩阵乘刚结束时顺便做 `Linear（线性矩阵乘） → Bias → GELU`；
- **residual（残差）** 块：`Dropout（随机置零） → Add（加回旁路） → LayerNorm（按特征归一化）`；
- **gated MLP（带门控前馈层）**：`SiLU（一种平滑门函数）(xW_1) ⊙ (xW_2)`，$`\odot`$ 表示逐元素乘；
- attention：scale、mask、softmax、dropout 与后续 matmul；
- **optimizer（优化器）**：根据梯度更新参数及其状态。

共同规律是：前一个算子的输出只被紧邻的下一个算子消费，因此中间量没有必要回到 HBM。

#### 反例：为什么两个能连写的算子不一定能融合？

- 中间结果被多个后续分支使用，不能立即丢弃；
- 算子之间需要所有 blocks 都等齐的**全局同步**，或需要把许多数合成一个数的 **reduction（归约，例如求和）**；
- 动态控制流使编译器看不见完整连续计算，形成 **graph break（计算图中断）**；
- 融合后寄存器需求过大，引发 register spill；
- 一个算子已有高度优化的库 kernel，硬塞进自定义融合反而失去 Tensor Core 优势。

因此 fusion 是“减少 IO 与 launch”对“资源压力和实现质量”的权衡。

### 6.4 技巧四：Recomputation

**forward（前向）** 是输入从第 1 层一路计算到损失；**backward（反向）** 从损失倒着使用导数，求每个参数怎样影响损失。导数常需要前向的中间结果。例如 $`y=\mathrm{sigmoid}(x)`$ 时，$`\frac{dy}{dx}=y(1-y)`$，所以 backward 需要知道前向算出的 $`y`$。

这些中间结果叫 **activation（激活）**。通常会保存它们，避免 backward 重复计算。但如果读取和保存 activation 比重算更贵，就可以交换资源：

```text
少量额外 FLOPs  ↔  大量 activation memory 和内存流量
```

课程用三层 sigmoid 举例：

- 传统方案：forward 保存中间量，backward 再读取，约 8 次内存读写；
- 重计算方案：丢弃中间 activation，backward 需要时从输入重做，约 5 次内存读写；
- 内存访问变为原来的 $`5/8`$，代价是多算几次 sigmoid。

把这 8 次和 5 次逐项列出来。设 $`s_2=\sigma(x)`$、$`s_1=\sigma(s_2)`$、$`out=\sigma(s_1)`$，$`dout`$ 是从后续计算传回的输出梯度。

> **先限定记账口径。** 这是课程第 35–36 页的“计算图边界/activation 保存”示意，不是把每个 sigmoid 当作独立 kernel 后统计真实 HBM transactions。它假设链内刚算出的值可以当场传给下一步；旧方案额外把 $`s_2,s_1,out`$ 写一份供以后使用。若三层真是三个独立 kernels，中间量还会被下一层读，绝对次数会更多。这个 8/5 例子只隔离“保存 activation”与“以后重算”的差别。

**保存所有中间量（8 次）：**

| 阶段 | 访问 |
|---|---|
| forward | ①读 $`x`$；②写 $`s_2`$；③写 $`s_1`$；④写 $`out`$ |
| backward | ⑤读 $`s_2`$；⑥读 $`s_1`$；⑦读 $`dout`$；⑧写最终输入梯度 $`dx`$ |

**重计算（5 次）：**

| 阶段 | 访问 |
|---|---|
| forward | ①读 $`x`$；②写最终输出 $`out`$，不写 $`s_2,s_1`$ |
| backward | ③再读 $`x`$；在当前计算中重算 $`s_2,s_1,out`$；④读 $`dout`$；⑤写 $`dx`$ |

因此旧图是 $`1+3+3+1=8`$，新图是 $`1+1+2+1=5`$。不同实现的精确计数可能不同，结论不是永远恰好省 3 次，而是：少把大 activation 写到 HBM、再读回来，代价是多执行便宜的数学函数。

工程中常见名称包括 activation checkpointing、gradient checkpointing。真正实现通常不会丢弃所有 activation，而是在网络中保存若干 checkpoint，在 checkpoint 之间重算，以控制计算和内存的折中。

#### 例子一：用 12 层链理解 checkpoint

假设网络是 12 个连续层，每层 forward 都产生一个 activation：

```text
x → L1 → a1 → L2 → a2 → ... → L12 → a12
```

**全部保存：** 保存 $`a_1`$ 到 $`a_{12}`$，backward 不必重做 forward。

**每 3 层保存一次：** 必须保留重算起点 $`x,a_3,a_6,a_9`$。最终输出 $`a_{12}`$ 还要交给损失/后续层，本来就会被持有；这里不把它重复叫作下一段的重算起点。反向传播到 $`L_{10}`$-$`L_{12}`$ 时，从 $`a_9`$ 重新执行这三层；处理 $`L_7`$-$`L_9`$ 时，从 $`a_6`$ 重做；处理第一段时从原输入 $`x`$ 重做。

```text
recompute boundaries: x | a3 | a6 | a9
recompute segments:      1-3  4-6  7-9  10-12
final output:                              a12 → loss
```

除原始输入 $`x`$ 与最终输出外，保存的内部边界 activation 从 11 份降到 3 份（$`a_3,a_6,a_9`$），而每个 segment 的 forward 需要额外执行一次。实际 autograd 还要保存一些不可重建或成本过高的量，然而这个模型足以解释基本交换关系。

这里 $`x,a_3,a_6,a_9`$ 是各段的起点；内部的 $`a_3,a_6,a_9`$ 通常称为 **checkpoints（检查点）**，好比只保留每三站的地图。区间内部的 $`a_1,a_2,a_4,a_5,\ldots`$ 是临时 activation，forward 之后丢掉；backward 到该区间时，再从最近边界重新走一遍。12 个层的原始 forward 仍执行一次，反向期间 4 个三层区间各额外执行一次，所以被 checkpoint 覆盖的层通常多做约一遍 forward，而不是无限重算。

#### 例子二：Transformer activation 的容量量级

一个形状为 $`[B,T,D]=[8,4096,4096]`$ 的 BF16 activation tensor 包含：

这里 $`B`$ 是 batch size（同时几条序列），$`T`$ 是每条序列的 token 数，$`D`$ 是每个 token 的隐藏向量长度。

```math
8\times4096\times4096=134{,}217{,}728\ \text{elements}
```

容量约为：

```math
134{,}217{,}728\times2\ \text{bytes}=256\ \text{MiB}
```

因为 $`1\ \text{MiB}=2^{20}=1{,}048{,}576`$ bytes，所以 $`268{,}435{,}456/1{,}048{,}576=256`$ MiB。32 层若各保存一份：

```math
256\ \text{MiB}\times32=8192\ \text{MiB}=8\ \text{GiB}
```

仅仅每层保存一份这样的 tensor，32 层就是约 8 GiB。真实 Transformer 每层还会产生 Q/K/V、MLP 中间量、归一化统计等，因此 activation memory 可能更大。

如果对大部分层使用 checkpointing，就有机会把若干 GiB 的 activation 换成一次额外 forward。训练长序列时，这往往决定 batch 能否放进显存。

#### 什么时候不应该重计算？

- kernel 已经 compute-bound，额外 FLOPs 会直接拉长训练时间；
- 被丢弃的操作很昂贵，但 activation 很小；
- 随机操作或有副作用的操作无法可靠重放；
- 推理阶段没有 backward，通常不存在同样的 activation 保存问题。

所以 checkpoint 策略应优先覆盖“activation 大、重算相对便宜”的区域。

### 6.5 技巧五：Coalesced memory access

global memory 不是按“单个标量的理想价格”服务每次读取。硬件以 transaction/segment 为单位搬运一段连续数据。一个 warp 的 32 个线程发出 load 时，硬件会把它们的地址合并成尽可能少的内存事务。

以连续 4-byte 元素为例：

- 32 个线程读取连续元素，总共需要 128 bytes；
- 因为 $`32\times4=128`$ bytes，而 $`128/32=4`$，在当前 CUDA 文档的简化模型中可由 4 个 32-byte transaction 满足；
- **利用率** $`=`$ 真正需要的字节数 $`/`$ 硬件实际搬运的字节数。这里两者都是 128 bytes，所以是 100%；
- 若相邻线程访问相隔至少 32 bytes 的地址，最差可能需要 32 个 transaction，实际只用其中很少的字节。

所以最常用的经验规则是：

```text
相邻线程 → 相邻地址
```

对 row-major 矩阵，元素在内存中的顺序是：

```text
A[0,0], A[0,1], A[0,2], ..., A[1,0], A[1,1], ...
```

例如一个 $`3\times4`$ FP32 矩阵从 byte address 0 开始：

```math
\begin{bmatrix}
A_{00}@0&A_{01}@4&A_{02}@8&A_{03}@12\\
A_{10}@16&A_{11}@20&A_{12}@24&A_{13}@28\\
A_{20}@32&A_{21}@36&A_{22}@40&A_{23}@44
\end{bmatrix}
```

row-major 的地址公式是：

```math
\mathrm{addr}(A[r,c])=\mathrm{base}+4(r\times4+c)
```

同一行从 $`c`$ 到 $`c+1`$ 只增加 4 bytes；同一列从 $`r`$ 到 $`r+1`$ 增加整行的 $`4\times4=16`$ bytes。因此让相邻线程沿列索引变化通常更连续；若相邻线程跨行取同一列，地址步长很大，容易产生更多事务。

#### 例子一：直接列出线程访问地址

仍用 8 个 lanes 和 4-byte FP32 元素建立直觉。

**连续访问：**

| Lane | 元素索引 | byte address |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 4 |
| 2 | 2 | 8 |
| 3 | 3 | 12 |
| 4 | 4 | 16 |
| 5 | 5 | 20 |
| 6 | 6 | 24 |
| 7 | 7 | 28 |

这 8 个地址位于同一个连续的 32-byte 区间，可以由很少的 transaction 满足。

**stride-16 访问：**

| Lane | 元素索引 | byte address |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 16 | 64 |
| 2 | 32 | 128 |
| 3 | 48 | 192 |
| 4 | 64 | 256 |
| 5 | 80 | 320 |
| 6 | 96 | 384 |
| 7 | 112 | 448 |

虽然仍然只使用 32 bytes 的有效数据，这些地址却分散在多个 segment 中，需要很多 transaction。硬件搬来了大量没有被当前 warp 使用的字节。

这里 stride-16 指元素索引每次跳 16 格。每格 4 bytes，所以相邻 lane 的地址差是 $`16\times4=64`$ bytes，而不是 16 bytes。

#### 例子二：矩阵转置为什么常用 shared-memory tile？

先看 $`2\times3`$ 数组：

```math
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
```

row-major 内存顺序是 [1, 2, 3, 4, 5, 6]，其元素地址可记成 [0, 4, 8, 12, 16, 20]。转置后的逻辑矩阵是：

```math
A^\top=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}
```

如果 transpose 只是 **view（视图）**，并没有搬动这些数：逻辑上的第一行 [1, 4] 实际地址仍是 [0, 12]，相差 12 bytes；下一行 [2, 5] 是 [4, 16]。**stride** 就是某个维度前进 1 格时，底层地址要跳多少格。permute/transpose 可以只改 shape 与 strides，所以操作本身便宜，但后续访问可能不连续。

**contiguous（连续化）** 会真的复制成转置后 row-major 的 [1, 4, 2, 5, 3, 6]。复制有一次成本，却可能让后面反复使用的 kernel 更快。

直接转置：

```python
out[col, row] = inp[row, col]
```

若读取 `inp[row, col]` 是连续的，写入 `out[col, row]` 就可能是跨行的；调换线程映射后，写入连续了，读取又可能变得分散。经典 CUDA transpose kernel 会：

1. 让一个 block 连续读取输入 tile 到 shared memory；
2. 在 shared memory 中交换索引；
3. 再以连续地址写出转置后的 tile。

这样用一次片上重排，让 global read 和 global write 都 coalesced。

#### 例子三：LLM tensor layout

若 Q/K/V 布局是 `[batch, sequence, heads, head_dim]`，最后一个 `head_dim` 通常连续。让相邻线程遍历 `head_dim` 往往很自然；若 kernel 让相邻线程跨 sequence 或 head 跳跃，而 stride 很大，就可能增加 transaction。

因此 `permute`/`transpose` 本身即使只是 view，后续 kernel 也可能因为 stride 改变而变慢；有时显式生成 contiguous layout 虽然多一次拷贝，却能让后续许多次计算更快。

> **补充澄清**：讲义使用“128-byte burst section”建立直觉；具体 transaction/segment 大小与 GPU 架构、数据宽度和访问模式有关。应记住“最大化有效字节/传输字节”，而不是把某个数字当作所有 GPU 的固定规则。

### 6.6 技巧六：Tiling

Tiling 是本讲最重要的技术。核心流程是：

1. 把大矩阵切成小块；
2. 每个 block 协作把所需 tile 从 global memory 搬到 shared memory；
3. block 内所有 threads 在同步点等齐，保证 tile 已经装完；
4. 在 shared memory 中反复使用这些元素，计算局部结果；
5. 必要时装入下一批 tile；
6. 最后把输出 tile 写回 global memory。

这里有三种容易混叫成 “tile” 的东西：

- **输出 tile**：一个 block 最终负责的 $`C`$ 的小方块；
- **A 输入 tile**与**B 输入 tile**：为了算这个输出块，在某一轮从 A、B 装进来的小方块；
- **k-phase（reduction 阶段）**：沿矩阵乘内部的 $`k`$ 方向分轮。每轮换一对 A/B 输入 tiles，并把结果加到同一个输出 tile。

下面为了讲复用，会跟踪两个相邻输出 tiles。真实 kernel 通常让不同 blocks 各算一个输出 tile；因此某个 A tile 可能会被另一个 block 再从 global memory 读取。复用发生在“同一个 block、同一个输出 tile 内”，不是全 GPU 永久只读一次。

#### 为什么能减少 global memory 读取？

下面不跳步，从最小例子开始。

##### 第一步：矩阵乘到底在算什么？

两个 $`N\times N`$ 矩阵相乘：

```math
C=AB
```

输出中的一个数 $`C_{i,j}`$，等于 $`A`$ 的第 $`i`$ 行与 $`B`$ 的第 $`j`$ 列做点积：

```math
C_{i,j}=\sum_{k=0}^{N-1}A_{i,k}B_{k,j}
```

先看 $`4\times4`$ 的情况。输出第一行的四个数分别是：

```math
\begin{aligned}
C_{0,0}&=A_{0,0}B_{0,0}+A_{0,1}B_{1,0}+A_{0,2}B_{2,0}+A_{0,3}B_{3,0}\\
C_{0,1}&=A_{0,0}B_{0,1}+A_{0,1}B_{1,1}+A_{0,2}B_{2,1}+A_{0,3}B_{3,1}\\
C_{0,2}&=A_{0,0}B_{0,2}+A_{0,1}B_{1,2}+A_{0,2}B_{2,2}+A_{0,3}B_{3,2}\\
C_{0,3}&=A_{0,0}B_{0,3}+A_{0,1}B_{1,3}+A_{0,2}B_{2,3}+A_{0,3}B_{3,3}
\end{aligned}
```

注意：同一个 $`A_{0,0}`$ 出现了四次。

```text
A[0,0] 被用于：C[0,0]、C[0,1]、C[0,2]、C[0,3]
```

原因很简单：$`A_{0,0}`$ 要和 $`B`$ 第 0 行的每一列配对，所以它参与输出第 0 行的全部 4 个结果。

同理，$`B_{0,0}`$ 会用于：

```text
B[0,0] 被用于：C[0,0]、C[1,0]、C[2,0]、C[3,0]
```

##### 第二步：朴素 GPU kernel 为什么把同一个数读很多次？

最容易写的朴素方法是：**一个线程负责一个输出元素**。

```text
线程 0 计算 C[0,0]
线程 1 计算 C[0,1]
线程 2 计算 C[0,2]
线程 3 计算 C[0,3]
```

四个线程彼此独立。如果它们不通过 shared memory 合作：

- 线程 0 去 global memory 读取一次 $`A_{0,0}`$；
- 线程 1 又读取一次 $`A_{0,0}`$；
- 线程 2 再读取一次 $`A_{0,0}`$；
- 线程 3 再读取一次 $`A_{0,0}`$。

虽然四个线程想要的是同一个数，但每个线程都自己去“远处仓库”取了一遍。因此在 $`4\times4`$ 例子中，每个 $`A`$ 元素大致会被读取 4 次，每个 $`B`$ 元素也大致会被读取 4 次。

这里的 “global reads” 是教学记账：读取一个标量元素算一次。真实硬件会把一组相邻标量请求合并成 transaction，也可能由 cache 命中，所以不要把标量次数误当成实际总线事务数。

推广到 $`N\times N`$：

- 固定一个 $`A_{i,k}`$，它要贡献给 $`C_{i,0},C_{i,1},\ldots,C_{i,N-1}`$，共 $`N`$ 个输出；
- 固定一个 $`B_{k,j}`$，它要贡献给 $`C_{0,j},C_{1,j},\ldots,C_{N-1,j}`$，也有 $`N`$ 个输出；
- 没有线程间复用时，一个输入元素因此大致从 global memory 读取 $`N`$ 次。

整个 $`A`$ 有 $`N^2`$ 个元素，所以读取次数约为：

```math
\underbrace{N^2}_{A\text{ 的元素数}}\times\underbrace{N}_{\text{每个元素重复读取次数}}=N^3
```

$`B`$ 也一样，因此两份输入合计：

```math
\text{naive input reads}\approx 2N^3
```

##### 第三步：tile size $`T=2`$ 时发生了什么？

仍然看 $`4\times4`$，现在让一个 thread block 一起计算一个 $`2\times2`$ 的输出 tile：

```text
第一个输出 tile：

C[0,0]  C[0,1]
C[1,0]  C[1,1]
```

block 中的线程先合作，把下面两个 $`2\times2`$ 输入 tiles 各从 global memory 读取一次：

```text
A tile                  B tile
A[0,0] A[0,1]           B[0,0] B[0,1]
A[1,0] A[1,1]           B[1,0] B[1,1]
```

然后把它们放进 shared memory。此时：

- $`A_{0,0}`$ 从 global memory 只读了 1 次；
- 但 shared memory 中的 $`A_{0,0}`$ 可以同时用于 $`C_{0,0}`$ 和 $`C_{0,1}`$；
- 一次昂贵读取，支持了 $`T=2`$ 个输出。

接着计算右上角输出 tile：

```text
C[0,2]  C[0,3]
C[1,2]  C[1,3]
```

这时需要再把包含 $`A_{0,0}`$ 的 tile 读取一次，让它服务 $`C_{0,2}`$ 和 $`C_{0,3}`$。

更精确地说，右上角输出 tile 一般由另一个 block 计算；那个 block 无法直接读取前一个 block 的 shared memory，所以它会再装一次包含 $`A_{0,0}`$ 的输入 tile。这正是为什么每个元素仍是 $`N/T`$ 次，而不是全局只读 1 次。

所以 $`A_{0,0}`$ 的读取过程从：

```text
朴素：为了 4 个输出，去 global memory 4 次
分块：每次服务 2 个输出，只去 global memory 2 次
```

也就是：

```math
4\quad\longrightarrow\quad\frac{4}{2}=2
```

##### 第四步：为什么一般情况是 $`N/T`$？

固定一个 $`A_{i,k}`$：

1. 它总共要服务 $`N`$ 个输出列；
2. 一个宽度为 $`T`$ 的输出 tile 一次覆盖 $`T`$ 个输出列；
3. 因而需要的输出 tiles 数量是：

```math
\frac{\text{总输出列数}}{\text{每个 tile 覆盖的列数}}=\frac{N}{T}
```

每个输出 tile 把 $`A_{i,k}`$ 从 global memory 装入一次，所以一个 $`A`$ 元素的 global reads 从 $`N`$ 次降到 $`N/T`$ 次。

对 $`B_{k,j}`$ 完全对称：它要服务 $`N`$ 个输出行，而每个 tile 一次覆盖 $`T`$ 行，所以也是 $`N/T`$ 次。

若 $`N`$ 不能被 $`T`$ 整除，严格写法是 $`\lceil N/T\rceil`$；课程先使用能整除的情况建立直觉。

##### 第五步：把总读取次数写完整

| 项目 | 朴素方法 | Tiled 方法 |
|---|---:|---:|
| $`A`$ 的 global reads | $`N^3`$ | $`N^3/T`$ |
| $`B`$ 的 global reads | $`N^3`$ | $`N^3/T`$ |
| 输入合计 | $`2N^3`$ | $`2N^3/T`$ |
| 输出写回 | $`N^2`$ | $`N^2`$ |
| 主要数学运算 | 约 $`2N^3`$ FLOPs | 仍约 $`2N^3`$ FLOPs |

global input reads 的缩减倍数为：

```math
\frac{2N^3}{2N^3/T}=T
```

所以 tile size 为 32 时，理想模型下 global input traffic 约减少 32 倍。数学运算并没有减少；只是从远处搬来的每个数被重复使用了 32 次。

##### 第六步：完整计算 $`N=1024,T=32`$

一个 $`1024\times1024`$ 矩阵有：

```math
1024^2=1{,}048{,}576\ \text{个元素}
```

**朴素方法：** 每个元素读取 1024 次。

```math
A\text{ reads}=1{,}048{,}576\times1024=1{,}073{,}741{,}824
```

```math
B\text{ reads}=1{,}073{,}741{,}824
```

```math
\text{总 input reads}=2{,}147{,}483{,}648
```

**Tiled 方法：** 每次加载服务 32 个输出，所以每个元素读取：

```math
\frac{1024}{32}=32\ \text{次}
```

于是：

```math
A\text{ reads}=1{,}048{,}576\times32=33{,}554{,}432
```

```math
B\text{ reads}=33{,}554{,}432
```

```math
\text{总 input reads}=67{,}108{,}864
```

验证缩减倍数：

```math
\frac{2{,}147{,}483{,}648}{67{,}108{,}864}=32
```

如果输入是 FP16，每个数 2 bytes，那么这个简化模型中的输入读取流量是：

```text
朴素：2,147,483,648 × 2 bytes = 4 GiB
分块：   67,108,864 × 2 bytes = 128 MiB
```

> **重要边界**：这是为了理解 tiling 而使用的理论模型，暂时忽略硬件 cache、transaction 合并、寄存器、输出写回和边界 tile。真实的“朴素 kernel”可能因 cache 获得一部分复用，因此实测不一定刚好是 32 倍；但“shared-memory tiling 主动把一次 global read 复用于 $`T`$ 个输出”的推导不变。

上面的二进制容量可逐步验证：

```math
4\ \text{GiB}=4\times2^{30}=4{,}294{,}967{,}296\ \text{bytes}
```

```math
128\ \text{MiB}=128\times2^{20}=134{,}217{,}728\ \text{bytes}
```

前者除以后者仍是 32。

#### Tile size 为什么需要调优？

先认五个词：

- **alignment（对齐）**：数据起点/尺寸能否整齐落在硬件偏好的地址边界；
- **transaction**：内存系统一次实际搬运的一整段；
- **occupancy**：一个 SM 能同时保留多少候补 warps；
- **kernel**：GPU 上运行的一份并行函数；
- **autotune**：把多个候选 tile/kernel 真正跑一遍，选择实测最快者。

tile 过小：

- 数据复用不足；
- block/kernel 管理开销相对更高。

tile 过大：

- shared memory 和寄存器可能不足；
- 一个 block 占用太多资源，降低 occupancy；
- 边界 tile 可能有大量空白；
- 形状不对齐时产生额外 transaction。

因此最优 tile 取决于矩阵形状、dtype、shared memory、寄存器、warp 数量和具体 GPU。`torch.compile(..., mode="max-autotune")` 的一个作用就是在支持的算子上测试多种实现/配置，付出更长编译时间换取更快运行时间。

只看理论复用，$`N=8`$ 时有：

| tile size $`T`$ | 每个输入元素约读 $`N/T`$ 次 | 理想流量缩减 | 直观代价 |
|---:|---:|---:|---|
| 2 | 4 | 2 倍 | 小块，复用少，资源轻 |
| 4 | 2 | 4 倍 | 中间选择 |
| 8 | 1 | 8 倍 | 复用最高，但一个 block 的 shared memory/寄存器更多 |

所以“大 tile 复用更好”只是公式的一面；资源装不下或让 occupancy 太低时，实际反而更慢。

#### 例子一：手算一个 $`4\times4`$ 矩阵乘的 $`2\times2`$ tile

令：

```math
A=\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
13&14&15&16
\end{bmatrix},\qquad
B=\begin{bmatrix}
1&0&2&0\\
0&1&0&2\\
1&0&1&0\\
0&1&0&1
\end{bmatrix}
```

只计算输出左上角 tile $`C[0:2,0:2]`$。切片 $`0:2`$ 表示“从编号 0 开始，到 2 之前停止”，也就是行 0、1 与列 0、1。**reduction 维度**就是公式中最后要被求和消掉的 $`k`$ 轴；这里 $`k=0,1,2,3`$，tile 宽 2，所以分为 $`k=0,1`$ 与 $`k=2,3`$ 两个阶段。每阶段先得到 partial result（部分和），最后相加。

**阶段 1：** 装入 $`A`$ 左上和 $`B`$ 左上两个 tiles：

```math
\begin{bmatrix}1&2\\5&6\end{bmatrix}
\begin{bmatrix}1&0\\0&1\end{bmatrix}
=\begin{bmatrix}1&2\\5&6\end{bmatrix}
```

例如这一阶段左上输出的部分结果是 $`1\times1+2\times0=1`$。

**阶段 2：** 装入 $`A`$ 右上和 $`B`$ 左下两个 tiles：

```math
\begin{bmatrix}3&4\\7&8\end{bmatrix}
\begin{bmatrix}1&0\\0&1\end{bmatrix}
=\begin{bmatrix}3&4\\7&8\end{bmatrix}
```

第二阶段同一位置的部分结果是 $`3\times1+4\times0=3`$。因此最终 $`C_{0,0}=1+3=4`$。

把两个 partial results 在寄存器/shared memory 中相加：

```math
C[0:2,0:2]=\begin{bmatrix}4&6\\12&14\end{bmatrix}
```

关键不是这个小矩阵本身，而是每次装入的 $`A/B`$ tile 会被用来计算多个输出元素。若逐个输出元素直接访问 global memory，同样的数据会被反复搬运。

#### 例子二：边界 tile 的浪费

矩阵是 $`5\times5`$，tile 是 $`4\times4`$。每个轴都需要 $`\lceil5/4\rceil=2`$ 个 tiles，因此输出覆盖区相当于 $`8\times8`$：

$`\lceil x\rceil`$ 表示向上取整：$`5/4=1.25`$，但一个 tile 不够，必须要 2 个。

```text
┌────┬────┐
│4×4 │4×1 │
├────┼────┤
│1×4 │1×1 │
└────┴────┘
```

4 个 blocks 各有 $`4\times4=16`$ 个 thread slots，总共 $`4\times16=64`$ 个位置。只有 25 个输出有效，所以 $`64-25=39`$ 个 slots 越界。kernel 用 mask 让这 39 个位置既不读取非法地址，也不写出结果，但这些边界 blocks 仍占据调度和执行资源。这就是把维度从 256 改成 257 时性能可能突然下降的微观原因。

#### 例子三：Attention 中 Q tile 的复用

FlashAttention 会把一个 $`Q`$ tile 留在片上，然后依次让它与多个 $`K`$ tiles 相乘。若 $`Q`$ 每次都从 HBM 重读，序列越长重复流量越大；把 $`Q`$ 留在 SRAM/register 中，就能让一次读取参与多个 score tiles。

这与矩阵乘 tiling 完全是同一个思想，只是还要用 online softmax 解决跨 tiles 的归一化。

### 6.7 六种技巧如何一起选择？

| 观察到的现象 | 首先怀疑 | 可能使用的技巧 |
|---|---|---|
| 大量短小 pointwise kernels | launch 和 HBM 往返 | fusion、低精度 |
| profiler 显示带宽接近峰值，Tensor Core 很空 | memory-bound | fusion、tiling、recomputation、低精度 |
| warp execution efficiency 低 | divergence 或边界浪费 | 重排任务、减少分歧、调整 tile |
| global load efficiency 低 | 地址分散/stride 大 | coalescing、layout 变换、shared-memory tile |
| activation 导致 OOM | 保存中间量过多 | recomputation/checkpointing、低精度 |
| matmul 维度只差 1，性能突然下降 | alignment、边界 tile 或 wave quantization | padding、换 tile/kernel、autotune |
| 理论 FLOPs 很少但运行仍慢 | IO 或框架开销占主导 | Roofline + profiler，而不是继续数 FLOPs |

这些技巧不是六个独立开关。一次优化经常同时改变多个因素：例如 tiling 既增加数据复用，也可能改善 coalescing；fusion 既减少 HBM 流量，也减少 launch；低精度既降低流量，也提高 Tensor Core 吞吐。

---

## 7. 解释“矩阵乘性能之谜”

课程开头展示了一张不平滑的曲线：随着方阵维度增大，matmul 吞吐总体上升，但有很多锯齿和周期性骤降。到这里可以把它拆成三种效应。

### 7.1 效应一：算术强度

小矩阵做的工作太少，数据移动和 launch 开销占比高，Tensor Core 也难以完全饱和。矩阵变大后，每个元素参与更多乘加，算术强度提高，吞吐沿 roofline 的斜坡上升。

### 7.2 效应二：对齐和可整除性

当矩阵维度能很好地被 kernel 的 tile/向量宽度整除时：

- 边界浪费少；
- 内存访问更容易 coalesce；
- Tensor Core tile 更容易填满。

课程中的曲线显示，维度只被 1 或 2 整除的形状吞吐较差，能被 16、32 整除的形状通常更好。但原因不是数字 16/32 本身，而是它们恰好匹配了该硬件和 kernel 的布局。

典型实例是 nanoGPT 的 vocabulary padding：把 vocab size 从 50,257 补到 50,304，课程引用的结果约有 25% speedup。虽然参数和 FLOPs 略有增加，但更好的矩阵形状让硬件有效吞吐显著提高。

为什么词表大小会变成矩阵维度？语言模型最后要为词表中的每个 token 产生一个分数，常见输出权重形状含有 [hidden_size, vocab_size]，所以 vocab size 就是大矩阵的一边。padding 增加的行/列只是硬件对齐用的空槽，不代表训练文本真的多出正常 token；这些假位置不会作为真实目标使用。多算少量空槽，反而可能因 tile 填得更整齐而整体更快。

### 7.3 效应三：Wave quantization

假设一个 matmul kernel 让每个 thread block 负责一个 $`256\times128`$ 的输出 tile。输出沿第一轴有 $`\lceil N/256\rceil`$ 块，沿第二轴有 $`\lceil N/128\rceil`$ 块，所以 block 总数是两者相乘：

- 维度 1792 时：$`\lceil1792/256\rceil\times\lceil1792/128\rceil=7\times14=98`$ 个 tiles；
- 维度 1793 时：$`8\times15=120`$ 个 tiles。

A100 有 108 个启用的 SM。课程前面的芯片结构图画的是完整 GA100 设计中的 128 个 SM，而实际 A100 产品常启用 108 个；本例讨论实际设备调度，所以使用 108，两处没有矛盾。为建立最简单时间模型，暂时假设每个 SM 同时只运行这个 kernel 的 1 个 block，而且每个 block 用时相近：

- 98 blocks：108 个 SM 一次都能接到，约 1 个“block 时间”完成；
- 120 blocks：第一波最多做 108 个，剩 12 个必须等第二波，约 2 个“block 时间”完成；
- 第二波只有 12/108 的 SM 工作，其余 96 个空着等尾巴。

因此工作量只从 98 增到 120（约 22.4%），简化运行时间却可能从 1 波跳到 2 波，吞吐 $`=`$ 工作量/时间 会出现明显下跌。真实硬件上一个 SM 可能同时驻留多个 blocks，block 时间也不完全相同；此例只是解释阶梯产生的机制。

一般形式是：

```math
\text{waves}=\left\lceil\frac{\text{number of thread blocks}}{\text{可同时执行的 blocks}}\right\rceil
```

只要工作块数量刚好越过一波的容量，尾部就可能产生低利用率。这就是性能曲线的周期性来源之一。

---

## 8. FlashAttention：把全部技巧组合起来

### 8.1 标准 attention 在做什么？

先不看大公式。假设序列只有三个 token：“猫 / 追 / 鱼”。每个 token 产生三种向量：

- **query（Q）**：“我现在想找什么信息？”；
- **key（K）**：“我这里有什么信息的标签？”；
- **value（V）**：“如果你关注我，真正拿走的内容是什么？”。

以某个 query $`q=[1,0]`$ 为例，三个 key 是：

```math
k_{\text{猫}}=[1,0],\quad k_{\text{追}}=[0.5,0.5],\quad k_{\text{鱼}}=[0,1]
```

**dot product（点积）** 是对应数字相乘再相加。因此未缩放分数为：

```math
q\cdot k_{\text{猫}}=1,\quad q\cdot k_{\text{追}}=0.5,\quad q\cdot k_{\text{鱼}}=0
```

标准 scaled dot-product attention 还要除以 $`\sqrt d`$。这里每条向量有 $`d=2`$ 个数，所以 $`\sqrt d=\sqrt2\approx1.414`$：

```math
[1,0.5,0]/\sqrt2\approx[0.707,0.354,0]
```

softmax 把它变成约 [0.456, 0.320, 0.225]，它们都为正且总和约为 1。若三个 value 是：

```math
v_{\text{猫}}=[10,0],\quad v_{\text{追}}=[0,10],\quad v_{\text{鱼}}=[5,5]
```

最终输出是加权和：

```math
0.456[10,0]+0.320[0,10]+0.225[5,5]\approx[5.68,4.32]
```

所以 attention 的一句人话是：**query 与每个 key 打分，用 softmax 变成关注比例，再按比例混合 values。**

对单个 attention head，设：

- $`Q,K,V\in\mathbb{R}^{N\times d}`$；
- $`N`$ 是序列长度；
- $`d`$ 是 head dimension。

计算过程：

```math
S=\frac{QK^\top}{\sqrt d}\in\mathbb{R}^{N\times N}
```

$`Q`$ 有 $`N`$ 行，每行是一个 token 的 query；$`K^\top`$ 把 $`K`$ 的行列交换，使每个 query 能与每个 key 做点积。因此 $`QK^\top`$ 有 $`N\times N`$ 个分数。除以 $`\sqrt d`$ 是为了当向量维度 $`d`$ 变大时，分数不要自然变得过大，从而让 softmax 过早饱和。

```math
P=\mathrm{softmax}_{\text{row}}(S)
```

```math
O=PV\in\mathbb{R}^{N\times d}
```

> **补充澄清**：attention core 包含两个主要 batched matmul：$`QK^\top`$ 和 $`PV`$。如果把生成 $`Q,K,V`$ 的三个线性投影也算入，则还有三次 projection matmul。课程幻灯片把它口头概括为围绕 $`Q,K,V`$ 的矩阵乘链；理解时应以公式为准。

### 8.2 朴素实现的问题不是只有 $`O(N^2)`$ FLOPs

$`O(N^2)`$ 的最直观含义是“每个 query 都要看每个 key”：$`N=4`$ 时有 $`4\times4=16`$ 对；$`N=8`$ 时有 $`8\times8=64`$ 对。序列翻倍，分数个数变成 4 倍。

朴素实现通常会把 $`N\times N`$ 的 score $`S`$ 和 probability $`P`$ 写到 HBM，再读回来做 softmax 和第二个 matmul。长序列下，这些中间量非常大。

**materialize（物化）** 是指真的为整个中间矩阵分配内存并把每个元素写进去，而不只是数学上说它存在。**batched matmul** 是把多个 batch/head 的矩阵乘一起交给 GPU；每一个 head 内仍有上面的两个核心 matmul。

FlashAttention 的关键观点是 **IO-aware**：不仅计算 FLOPs，还显式优化 HBM 与片上 SRAM 之间的读写次数。

它仍然计算精确的 softmax attention；主要收益来自：

- 不把完整 $`S`$、$`P`$ 物化到 HBM；
- 按 tile 读取 $`Q,K,V`$；
- 在 SRAM/register 中融合 score、softmax 和 $`V`$ 加权；
- backward 时重算需要的局部量，而不是保存 $`N^2`$ activation。

### 8.3 难点：softmax 是全局归一化

矩阵乘很容易切块，但一行 softmax 的分母依赖该行全部元素：

```math
\mathrm{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j
```

只看一个 tile，既不知道全局最大值，也不知道完整分母。解决方法是 **online softmax**。

$`e^x`$ 是指数函数，$`e\approx2.718`$。它把分数变成正数，而且分数大一点，权重会明显更大。先对 [1, 2, 4] 做普通 softmax：

```math
[e^1,e^2,e^4]\approx[2.718,7.389,54.598]
```

总和约 $`64.705`$，所以概率约：

```math
[2.718,7.389,54.598]/64.705\approx[0.042,0.114,0.844]
```

直接算很大的 $`e^{1000}`$ 会溢出。减去同一个最大值 4 不改变比例：

```math
[e^{-3},e^{-2},e^0]\approx[0.0498,0.1353,1]
```

除以总和 $`1.1851`$，仍约为 [0.042, 0.114, 0.844]。所以“减最大值”既稳定又不改变答案。

### 8.4 Online softmax 的可合并状态（补充推导）

对已经处理的一组 logits，维护：

- 最大值 $`m`$；
- 稳定化分母 $`\ell=\sum_i e^{x_i-m}`$；
- 若同时累积 value，维护 $`o=\sum_i e^{x_i-m}v_i`$。

这里 logit 就是 softmax 前的原始分数；$`\ell`$ 读作 ell，不是数字 1。$`v_i`$ 可以是一个数，也可以是一条 value 向量；对应地 $`o`$ 也是数或向量。$`o`$ 还没除以分母，所以最终才做 $`O=o/\ell`$。

新 tile 有自己的 $`(m_b,\ell_b,o_b)`$。合并时令：

```math
m'=\max(m,m_b)
```

```math
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b
```

```math
o'=e^{m-m'}o+e^{m_b-m'}o_b
```

最终输出：

```math
O=\frac{o'}{\ell'}
```

为什么要乘 $`e^{m-m'}`$？因为旧累计量原本以旧最大值 $`m`$ 为基准；发现更大值后，必须把它重新缩放到新基准 $`m'`$。这个状态可以按 tile 递增更新，因此无需一次看到整行 logits。

#### 一个两块小例子

取完整 logits [1, 2, 4, 3]，对应标量 values [10, 20, 30, 40]，每块两个元素。

**第一块 [1, 2]：**

```math
m_1=2
```

```math
\ell_1=e^{1-2}+e^{2-2}=e^{-1}+1\approx1.367879
```

```math
o_1=e^{-1}\times10+1\times20\approx23.678794
```

**第二块 [4, 3]：**

```math
m_2=4
```

```math
\ell_2=1+e^{-1}\approx1.367879
```

```math
o_2=1\times30+e^{-1}\times40\approx44.715178
```

**合并：** 新最大值 $`m'=4`$。旧块原本以 2 为基准，现在必须乘 $`e^{2-4}=e^{-2}\approx0.135335`$：

```math
\ell'=e^{-2}\ell_1+\ell_2
\approx0.135335\times1.367879+1.367879
\approx1.553002
```

```math
o'=e^{-2}o_1+o_2
\approx0.135335\times23.678794+44.715178
\approx47.919754
```

最终加权输出：

```math
O=o'/\ell'\approx47.919754/1.553002\approx30.856213
```

直接一次算完整行时，以 4 为最大值的未归一化权重为：

```math
[e^{-3},e^{-2},1,e^{-1}]\approx[0.049787,0.135335,1,0.367879]
```

分母同样是 1.553002；与 [10, 20, 30, 40] 的加权和同样是 47.919754，答案仍是 30.856213。分块只改变计算顺序，没有改变 softmax 结果。

### 8.5 FlashAttention forward 的数据流

可用下面的简化流程理解：

1. 把一个 $`Q`$ tile 保持在片上；
2. 依次加载 $`K,V`$ tiles；
3. 在片上计算局部 $`S=QK^\top`$；
4. 立刻更新 online max、normalizer 和加权 value 累加器；
5. 不把完整 $`S`$ 或 $`P`$ 写入 HBM；
6. 遍历完所有 $`K,V`$ tiles 后，只写回最终输出 tile。

用“1 个 query、4 个 K/V、tile size=2”完整走一遍：

1. 从 HBM 读入 query $`q`$，留在片上；
2. 从 HBM 读入 $`(k_0,k_1)`$ 和 $`(v_0,v_1)`$；
3. 在片上算两个 scores，用它们建立第一块的 $`m,\ell,o`$，不写出 scores/probabilities；
4. 丢弃已消费的局部 scores，再读 $`(k_2,k_3)`$ 和 $`(v_2,v_3)`$；
5. 算后两个 scores，用第 8.4 节公式更新同一个 $`m,\ell,o`$；
6. 算 $`O=o/\ell`$，只把最终输出 $`O`$ 写回 HBM。

为了突出中间量，暂时不数双方都必须读取的 Q/K/V。朴素方案会：

| 中间步骤 | HBM 访问的标量个数 |
|---|---:|
| 写 score $`S`$ | 4 |
| softmax 读 $`S`$ | 4 |
| 写 probability $`P`$ | 4 |
| 第二个 matmul 读 $`P`$ | 4 |
| 合计 | 16 |

FlashAttention 对这 16 次 $`S/P`$ 中间访问是 0；score 和 probability tile 算完立刻在片上消费。它仍要读 Q/K/V、写最终 O，而且真实 tile 是矩阵块，但这个四元素例子展示了关键差别。

这里同时出现了：

- **tiling**：把 $`Q,K,V`$ 分块放入片上存储；
- **fusion**：把 matmul、scale、mask、softmax、$`PV`$ 尽量放在同一个 kernel/流水中；
- **online softmax**：让全局归一化可以分块递增；
- **recomputation**：backward 重建局部 score/probability，避免保存完整 $`N^2`$ 中间矩阵。

### 8.6 为什么 FlashAttention 会更快？

它不一定显著减少理论 FLOPs，甚至会因为重计算增加一部分 FLOPs；但它大幅减少了昂贵的 HBM traffic。现代 GPU 上 matmul 很快，而 HBM 往返相对昂贵，因此“多算一点、少搬很多”可能更快。

这正是整堂课的缩影：

> 算法复杂度只告诉你做多少数学工作；硬件性能还取决于这些工作如何映射到计算单元与内存层次。

---

## 9. 课堂问答与工程补充

### Q1：L1 cache 和 shared memory 有什么区别？

二者都位于 SM 附近，很多架构上还共享/动态划分部分物理资源，但行为不同：

- cache 主要由硬件自动填充和替换；
- shared memory 是 kernel 显式声明、装载、同步和复用的程序员可控空间。

### Q2：PyTorch 的 matmul 会自动 tiling 吗？

会。cuBLAS、cuBLASLt、Triton/Inductor 等成熟实现会在底层分块。普通用户通常不直接选择 tile size，但需要关心矩阵形状、dtype 和编译配置，因为它们决定库能否选到高效 kernel。

### Q3：`max-autotune` 在做什么？

它会为支持的算子尝试和测量多种候选实现，例如不同 matmul 模板或 Triton 配置，然后选择实测更快的方案。代价是首次编译明显变慢，因此更适合会重复运行很多次的稳定工作负载。

### Q4：量化的 scale 是训练出来的吗？

不一定。训练时常用当前 block/tensor 的 max、历史统计或其他 calibration 规则计算 scale，通常不把它当作通过普通梯度学习的模型参数。推理量化可能使用更充分的校准数据或优化过的 scale。

### Q5：为什么 inference 尤其容易 memory-bound？

自回归 decode 每一步只生成少量 token，却要读取大量模型权重和 KV cache，矩阵乘的批量维度可能不够大，数据复用较低。课程在问答中提到 prefill/decode disaggregation：让擅长大矩阵计算的资源处理 prefill，让带宽更合适的资源处理 decode。这是“按瓶颈分配硬件”的进一步应用。

### Q6：为什么不能记住“矩阵维度永远补到 32 的倍数”？

因为最优形状取决于 dtype、kernel tile、硬件架构和其他维度。32 是常见经验，不是普适定理。正确流程是理解 alignment/divisibility 的原因，再用目标硬件 benchmark。

---

## 10. 常见误区

### 误区 1：FLOPs 少的算法一定快

实际时间还取决于内存流量、通信、kernel launch、并行度和硬件支持。FlashAttention 通过增加部分重计算减少 IO，就是反例。

### 误区 2：GPU 利用率高说明 Tensor Core 很忙

监控工具中的“GPU utilization”常表示一段时间内有 kernel 活跃，不等于矩阵单元达到了峰值吞吐。判断性能还要看 achieved FLOPs、memory bandwidth、occupancy、stall reason 等指标。

### 误区 3：矩阵越大，吞吐一定单调提高

总体趋势可能提高，但边界 tile、对齐、wave quantization 和 kernel 选择会产生锯齿与骤降。

### 误区 4：Fusion 总能提高性能

fusion 通常减少 HBM traffic 和 launch overhead，但也可能增加寄存器/共享内存压力，造成 spill 或 occupancy 下降。

### 误区 5：FlashAttention 是近似 attention

不是。FlashAttention 通过重排精确计算减少 IO。浮点运算顺序改变可能带来正常的舍入差异，但目标数学函数没有被近似替代。

---

## 11. 自测题与答案

1. 为什么 GPU 能容忍单个线程较高的等待时间，却仍有很高总吞吐？
2. block 为什么是 shared memory 复用的自然边界？
3. 一个 kernel 的算术强度为 20 FLOP/byte，GPU 带宽 2 TB/s、峰值 150 TFLOP/s。理论上它受什么限制？最高约多少 TFLOP/s？
4. 为什么 FP16 ReLU 相比 FP32 ReLU 算术强度更高？
5. `sin(x)**2 + cos(x)**2` 为什么适合 fusion？
6. recomputation 为什么可能同时减少显存容量和内存流量？
7. 对 row-major 矩阵，为什么相邻线程读取相邻列通常比读取同一列的不同行更容易 coalesce？
8. $`N=2048,T=64`$ 时，课程简化模型下 tiling 把每个输入元素的 global reads 从多少降到多少？
9. 1792 到 1793 的例子中，为什么多出的 22 个 tiles 会造成远大于 $`22/98`$ 的性能损失？
10. Online softmax 需要维护哪几个状态？为什么新最大值出现时必须重新缩放旧累计量？
11. FlashAttention 为什么可以不保存完整 $`N\times N`$ attention matrix？
12. 请用一句话解释本讲中低精度、fusion、recomputation、coalescing、tiling 的共同目标。
13. FP16 向量加法 $`z=x+y`$ 每个元素约有多少算术强度？在峰值 100 TFLOP/s、带宽 1 TB/s 的 GPU 上，Roofline 上限约是多少？
14. 10 亿参数仅保存一份 BF16 权重约需要多少十进制 GB？为什么训练总显存会远高于它？
15. 简化的 `bias + GELU` 例子中，fusion 把流量从 $`10N`$ 降到 $`6N`$ bytes，降低了百分之多少？
16. $`5\times5`$ 输出使用 $`4\times4`$ tiles 时，tile 网格覆盖 64 个位置，其中有效输出比例是多少？
17. 为什么一个不发生拷贝的 `permute`/`transpose` 仍可能让后续 GPU kernel 变慢？

<details>
<summary>参考答案</summary>

1. GPU 在大量 warps 之间切换；一个 warp 等待内存时，SM 可以执行其他就绪 warp，以总吞吐隐藏延迟。
2. 同一 block 保证被调度到同一 SM，线程可以访问同一 shared memory 并进行 block 内同步。
3. 带宽上限为 $`2\times20=40`$ TFLOP/s，低于 150 TFLOP/s，因此 memory-bound，最高约 40 TFLOP/s。
4. 运算量近似不变，但每次读写字节减半，所以 FLOP/byte 翻倍。
5. 中间量只被相邻逐元素操作消费，可在寄存器/片上存储中完成，无需每一步写回 HBM。
6. 它丢弃部分 forward activations，backward 时重算，用额外 FLOPs 换更少的保存和读取。
7. row-major 中同一行的相邻列在地址上连续；同一列的不同行相隔一整行。
8. 从 $`N=2048`$ 次降到 $`N/T=32`$ 次，约减少 64 倍。
9. 98 个 tiles 可在 108 个 SM 上一波完成；120 个需要第二波，而第二波只有 12 个工作块，大部分 SM 空闲。在每波都算 1 个时间单位的玩具模型里，tile throughput 从 $`98/1=98`$ 降到 $`120/2=60`$，下降 $`(98-60)/98\approx38.8\%`$；真实降幅受 block 驻留数和每块耗时影响。
10. 最大值 $`m`$、稳定化分母 $`\ell`$，若直接累积输出还要有加权和 $`o`$。最大值改变后，旧指数项的基准变了，必须乘指数比例换到新基准。
11. 它逐块计算 score，在线更新 softmax 与 value 加权结果；中间 score/probability 留在片上并及时消费，backward 再按 tile 重算。
12. 减少昂贵的数据移动，并让已搬入快内存的数据产生更多有效计算。
13. 两次 FP16 读取加一次写入共约 6 bytes，一次加法，所以 $`I\approx1/6=0.167`$ FLOP/byte；带宽上限约 $`0.167`$ TFLOP/s，远低于峰值，是 memory-bound。
14. 约 2 GB。训练还可能需要 gradients、optimizer states、高精度 master weights、activations、临时 buffer 和通信空间。
15. $`(10N-6N)/(10N)=40\%`$。
16. $`25/64\approx39.1\%`$；约 60.9% 的覆盖位置是边界填充或无效 lane。
17. view 虽未搬数据，却改变了 strides。相邻线程可能因此访问相距很远的地址，破坏 coalescing；若这个布局被重复使用，先做一次 contiguous copy 反而可能更快。

</details>

---

## 12. 复习清单：如果只能记住七件事

1. GPU 是吞吐机器，CPU 更偏向延迟机器。
2. warp 是 32 线程的关键调度组；block 是 shared memory 与同步的边界。
3. 现代 GPU 的主要矛盾是 compute 增长快于 memory/communication。
4. Roofline 用 $`\min(P_{peak},B\cdot I)`$ 判断 memory-bound 还是 compute-bound。
5. 高性能 kernel 的核心是少搬、连续搬、重复用。
6. 矩阵形状会影响对齐、tile 利用率、kernel 选择和 wave 数量，所以性能并不平滑。
7. FlashAttention 用 tiling、fusion、online softmax 和 recomputation 精确地计算 attention，同时显著减少 HBM IO。

---

## 13. 视频导航

| 时间 | 讲义页 | 内容 |
|---|---:|---|
| [00:05](https://www.youtube.com/watch?v=izZba4UA7iY&t=5s) | 2-4 | 动机与三部分课程结构 |
| [04:15](https://www.youtube.com/watch?v=izZba4UA7iY&t=255s) | 5-7 | Compute scaling 与 Dennard scaling |
| [07:31](https://www.youtube.com/watch?v=izZba4UA7iY&t=451s) | 8 | CPU 与 GPU 的设计哲学 |
| [09:01](https://www.youtube.com/watch?v=izZba4UA7iY&t=541s) | 9-10 | SM、计算单元与内存层次 |
| [13:27](https://www.youtube.com/watch?v=izZba4UA7iY&t=807s) | 11-12 | thread、block、warp 与内存模型 |
| [17:18](https://www.youtube.com/watch?v=izZba4UA7iY&t=1038s) | 13-15 | GPU 与 TPU 对比 |
| [24:35](https://www.youtube.com/watch?v=izZba4UA7iY&t=1475s) | 16-19 | Tensor Core、matmul 与 compute-memory gap |
| [30:09](https://www.youtube.com/watch?v=izZba4UA7iY&t=1809s) | 20-21 | 性能优化部分与 Roofline model |
| [32:57](https://www.youtube.com/watch?v=izZba4UA7iY&t=1977s) | 22-23 | Control divergence |
| [34:53](https://www.youtube.com/watch?v=izZba4UA7iY&t=2093s) | 24-29 | 低精度、FP8 与 MXFP4 |
| [47:43](https://www.youtube.com/watch?v=izZba4UA7iY&t=2863s) | 30-33 | Operator fusion |
| [50:13](https://www.youtube.com/watch?v=izZba4UA7iY&t=3013s) | 34-36 | Recomputation |
| [52:52](https://www.youtube.com/watch?v=izZba4UA7iY&t=3172s) | 37-39 | Coalesced memory access |
| [57:51](https://www.youtube.com/watch?v=izZba4UA7iY&t=3471s) | 40-44 | Tiling、tile size 与 alignment |
| [1:06:16](https://www.youtube.com/watch?v=izZba4UA7iY&t=3976s) | 45-49 | 矩阵乘吞吐曲线与 wave quantization |
| [1:11:52](https://www.youtube.com/watch?v=izZba4UA7iY&t=4312s) | 50-54 | FlashAttention |
| [1:17:28](https://www.youtube.com/watch?v=izZba4UA7iY&t=4648s) | 55 | 全讲总结 |

---

## 14. 术语表

| 术语 | 中文解释 |
|---|---|
| Throughput | 单位时间完成的总工作量 |
| Latency | 单个任务从开始到完成的时间 |
| SM | Streaming Multiprocessor，GPU 的主要计算/调度单元 |
| Warp | NVIDIA GPU 上通常由 32 个线程组成的调度组 |
| Block | 被调度到一个 SM、可共享 shared memory 的线程组 |
| SIMT | Single Instruction, Multiple Threads |
| HBM | High Bandwidth Memory，GPU 的大容量 global memory |
| SRAM | 常用于片上 cache/shared memory 的快速存储技术 |
| DRAM | global memory/HBM 所属的动态存储技术家族 |
| Arithmetic intensity | 每传输一个 byte 所完成的 FLOPs |
| Memory-bound | 性能主要被内存带宽或延迟限制 |
| Compute-bound | 性能主要被计算单元峰值限制 |
| Kernel | 在 GPU 上并行执行的函数 |
| Coalescing | 把 warp 中多个线程的访存合并为较少的内存事务 |
| Tiling | 把数据分块搬入片上存储并重复使用 |
| Fusion | 把多个操作合并到一个 kernel/计算区域 |
| Recomputation | 丢弃中间量，需要时重新计算 |
| Quantization | 用更少 bit 的数值格式表示权重或 activation |
| Wave quantization | 工作块数量跨过一波硬件容量时产生的阶梯式利用率变化 |
| IO-aware | 设计时显式考虑不同内存层级之间的数据传输成本 |

---

## 15. 来源、资料与延伸阅读

### 本讲原始资料

- [课程视频：Lecture 5 - GPUs, TPUs](https://www.youtube.com/watch?v=izZba4UA7iY)
- [课程讲义：lecture_05.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)
- [CS336 Spring 2026 课程主页](https://cs336.stanford.edu/)

### 官方文档与原论文

- [NVIDIA CUDA Programming Guide：SIMT kernels 与 memory performance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch `torch.compile` 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
