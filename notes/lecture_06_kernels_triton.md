# CS336 Lecture 6：Kernels、Benchmarking、Profiling 与 Triton

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Percy Liang  
> 视频：[Lecture 6](https://www.youtube.com/watch?v=xnDHaNUvHBg)（约 86:36）  
> 官方可执行讲义：[lecture_06.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)

> **资料核验说明：**本讲没有 PDF。任务上下文中的旧 raw 行数记录是 671 行，但 2026-08-28 抓取的官方 `main` 页面显示 744 行、554 LOC（Lines of Code，代码行；这里是不把部分空行/注释计入的口径），2026-04-15 首次发布提交 `15d7589` 的同一文件也为 744 个物理行。本笔记以当前官方 **744 行**版本为覆盖基准，不能为了凑旧数字删掉后 73 行。视频主字幕使用人工轨 `English (United States)`，语言代码 `en-US`、非自动生成，共 1552 个 segments（字幕片段）；末段从 86:34 开始，约在 86:36 结束。自动轨 `English (auto-generated)` 只用于确认轨道存在，没有作为主字幕。

本讲使用四种来源标签：

- **【课程代码】**：官方 `lecture_06.py` 明写的代码、数字或文字；
- **【视频补充】**：老师的口头限定、现场演示或课堂问答；
- **【补充理解】**：为了让只会四则运算的读者跟上而加入的定义与推导；
- **【补充例子】**：本笔记自建、可以手算的数字例；
- **【课程时点快照】**：A100/H100/B200 数字、profiler kernel 名称等依赖 2026 年课堂机器或软件版本，不能当成所有 GPU 的永久常数。

### 资料与图片如何核验

我没有只读图片文件名，而是下载后按原始像素逐张视觉检查：

为让核验表本身也能读懂：GPU 是 Graphics Processing Unit（图形处理器，这里指通用并行加速器）；SM 是 Streaming Multiprocessor（流式多处理器）；CTA 是 NVIDIA 正式文档中的 **Cooperative Thread Array（协作线程数组，即 thread block）**；L1/L2 是 Level-1/Level-2 cache（一/二级缓存）；HBM 是 High Bandwidth Memory（高带宽内存）。课程源码第 61 行误写成 `concurrent thread array`，本笔记保留这条勘误，不把笔误升级成正式全称。正文会结合层级逐一重讲。

| 图片 | 像素 | 视觉检查到的内容 | 笔记去向 |
|---|---:|---|---|
| `gpu-hardware.png` | 1189×933 | 左侧 GPU 内有多个 SM；每个 SM 内标有 `Reg`、`L1+shmem`；芯片共享 L2；右侧独立大块为 HBM | §1–2 |
| NVIDIA `grid-with-CTAs.png` | 1048×583 | 一个绿色 Grid 中有 8 个 CTA；每个 CTA 内有一组向下箭头表示 threads | §2 |
| NVIDIA `wave-quantization.png` | 305×158 | 纵轴是 SM、横轴是 time；wave 0 填满，wave 1 只有部分绿色条形成 tail | §4 |
| `triton-softmax.png` | 1536×426 | row 1 由 `pid=1` 的一个 program 依次 load、subtract max、exp+sum、normalize/store | §11 |
| `triton-row-sum.png` | 1077×871 | row 1 被 block 1 以 4 列一 tile 循环；图示 `t0–t3` 表示 4 个 vector accumulator 位置，是课程硬件映射简化；最终 CUDA threads 映射与 register/shared 落点由 compiler 决定；最后两列 mask，结果为 39 | §12 |
| `gemm_tiled.png` | 850×368 | A 的横向 tile 与 B 的纵向 tile 相乘，累加到 C 的一个橙色输出 tile；紫/绿区分外层 tile 和当前元素 | §13–15 |

四张本地图与两张 NVIDIA 外链图均可正常打开，没有裁切、透明空白或文字不可辨问题。讲义第 59、133、306 行两次/多次引用 NVIDIA 图；本笔记将图片的箭头和层级翻成文字，不要求读者靠猜图学习。

### 官方 744 行源码覆盖概览

下面与 §22.3 的终稿表使用同一组无缝区间；§22 还会解释课程基础设施、每个 helper 和版本边界。

| 官方行段 | 内容 | 对应笔记 |
|---:|---|---|
| 1–38 | imports、课程工具、`main()` 全讲路线、分隔行 | 来源边界、§0、§16、§22.2 |
| 39–57 | GPU 存储层次、A100/H100/B200 表、B200 TMEM | §1、§22.2 |
| 58–81 | grid、CTA、thread、SM 映射、thread-block clusters | §2、§22.2 |
| 82–91 | warp、lockstep、divergence、warp switching | §3 |
| 92–113 | register 限制与 warp occupancy 代码 | §4 |
| 114–125 | shared-memory banks、conflict、swizzling | §5 |
| 126–131 | HBM memory coalescing | §5 |
| 132–143 | block waves、硬件总结、分隔 | §4–5 |
| 144–205 | benchmarking、warmup、CUDA（Compute Unified Device Architecture）events、mean、分隔 | §6 |
| 206–261 | profiler、kernel name、追加 `var/profiles.txt`、分隔 | §6、§22.2 |
| 262–304 | naive/builtin/compiled GeLU、fusion、分隔 | §7 |
| 305–316 | CUDA 与 Triton 编程抽象、分隔 | §8 |
| 317–391 | Triton GeLU host wrapper、kernel、PTX（Parallel Thread Execution）、分隔 | §9–10 |
| 392–486 | naive/fused softmax、分隔 | §11 |
| 487–537 | 超长 row 的 tiled sum、分隔 | §12 |
| 538–601 | matmul 的 naive/ideal/tiled 主线、stride、分隔 | §13–14 |
| 602–676 | fused matmul+ReLU Triton wrapper/kernel、分隔 | §14–15 |
| 677–744 | 测试 helper、GeLU/softmax reference、mean、PTX 输出、入口 | §16.5 |

其中 **PTX（Parallel Thread Execution，并行线程执行）**是 NVIDIA 定义的 GPU 虚拟指令集/中间汇编：Triton 源码会编译成 PTX，再由驱动变成具体 GPU 的机器指令。它不是 Python，也不是最终硬件二进制；正式代码推导在 §10。

---

## 0. 五分钟复习卡与第一次阅读方法

> **第一次阅读不要从复习卡硬背。**请先读 §1，把每个存储层想成不同远近的仓库；再读 §2–5，最后读 §6。第二遍复习再回这里。

### 0.1 一句话主线

**Kernel（核函数）**是一次交给 GPU 并行执行的小程序。写对只需要理解 grid、block、thread；写快还必须让 warp 少分叉、让 SM 保持足够可运行工作、让 shared memory 少 bank conflict、让 HBM 访问合并，并用 benchmark/profile 实测而不是凭感觉。

**【课程代码｜行 13–36｜视频 [00:19](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=19s)】**老师把本讲定位为 Lecture 5 的继续：Lecture 5 讲 GPU 性能的高层模型，本讲会进入代码、benchmarking（基准计时）、profiling（性能剖析）和 Triton kernels。

### 0.2 全讲因果链

```text
PyTorch 表达“要算什么”
        ↓
底层 kernel 决定“一次 GPU launch 具体怎样算”
        ↓
grid 分成 blocks/CTAs，block 分成 threads
        ↓
硬件把 block 放到 SM，把每 32 threads 组成一个 warp
        ↓
分支、register 用量、shared-memory bank、HBM 地址模式
都会改变实际速度
        ↓
先声明要量 device elapsed 还是 application wall-clock，再 benchmark
再 profile 看时间花在哪些 kernels
        ↓
用 fusion、Triton、tiling 等方法修改
        ↓
重新 benchmark/profile，验证而不是猜
```

### 0.3 本讲前半必须会复算的五个数字

1. 128 threads/block、160 registers/thread：

   $`128\times160=20,480\ \text{registers/block}.`$

2. 一个 SM 有 65,536 registers：

   $`\left\lfloor\frac{65,536}{20,480}\right\rfloor=3\ \text{blocks}.`$

3. 每 block 有 $`128/32=4`$ warps，所以 resident warps 为：

   $`3\times4=12.`$

4. 硬件最多 64 resident warps：

   $`\text{warp occupancy}=\frac{12}{64}=0.1875=18.75\%.`$

5. 32 个 thread 各连续读一个 FP32（32-bit floating point，32 位浮点数）：

   $`32\ \text{threads}\times4\ \text{bytes/thread}=128\ \text{bytes},`$

   理想化情况下正好装进一个 128-byte HBM transaction。

### 0.4 三个不能混用的词

- **warp occupancy**：一个 SM 上实际 resident warps ÷ 允许的最大 warps；
- **block occupancy / wave utilization**：一波 block 有没有填满所有可用 SM slots；
- **arithmetic intensity（算术强度）**：做了多少 FLOPs ÷ 从某一层存储搬了多少 bytes。

它们都可能影响速度，但单位和问题不同。

### 0.5 最容易记错的边界

- 源码第 97 行的前置文字写“64 threads”，但第 100 行真实变量是 `128`；本笔记按实际变量复算。
- 低 warp occupancy 不自动等于慢；若每个 thread 做更多有用工作，较少 warps 仍可能更快。
- 同一 bank 的**同一地址读取**可以 broadcast（广播），不是 conflict；同一 bank 的**不同地址**才需要串行。
- Shared-memory bank conflict 与 HBM coalescing 是两个不同存储层的问题。
- CPU 看到 kernel launch 返回，不表示 GPU 已执行完成；错误计时常常只量到“发命令”的时间。
- 课程说“一次 128-byte transaction”是教学简化；真实 transaction 数还受对齐、cache、数据宽度和具体 GPU 架构影响。

### 0.6 最终笔记地图

| 部分 | 要回答的问题 |
|---|---|
| §1–2 | GPU 有哪些仓库？grid/block/thread 怎样落到 SM？ |
| §3–5 | warp 分支、occupancy、banks、coalescing 为什么让同一代码快慢不同？ |
| §6 | 怎样可靠计时并定位瓶颈？ |
| §7–10 | 为什么 fusion 快？怎样写第一个 Triton GeLU kernel？PTX 暴露了什么？ |
| §11–15 | softmax、跨 tile row sum、matmul tiling 与 ReLU fusion 怎样逐步实现？ |
| §16 | 怎样从正确性走到 benchmark/profile，并覆盖课程 helper？ |
| §17–18 | 怎样诊断性能？哪些常见说法是错的？ |
| §19–20 | 60 道自测与 60 份逐步答案 |
| §21–23 | 视频导航、精确来源覆盖与一页复习流程 |

---

## 1. 前置知识：从 Lecture 5 过桥到 kernel、存储层次与算术强度

### 1.1 Kernel 到底是什么

**Kernel（核函数）**不是操作系统内核。GPU 语境下，它是 CPU 一次提交给 GPU、由许多 GPU threads 并行执行的函数。一次 **kernel launch（核函数启动）**就是 CPU 把“运行哪个 kernel、用多大的 grid/block、参数地址是什么”排进 GPU 工作队列。

**【补充例子】**有 8 个数字，要把每个数加 1：

```text
输入： [10,20,30,40,50,60,70,80]
thread 0 处理第 0 项：10→11
thread 1 处理第 1 项：20→21
...
thread 7 处理第 7 项：80→81
```

“每个 thread 把自己的数字加 1”是 kernel code；“启动 8 个逻辑 threads 去执行”是 launch 配置。

**【课程代码｜行 13–19】**课程先讲 benchmark/profile，再写 Triton，不是因为 Triton 不重要，而是因为优化前应先测量瓶颈。视频 [22:09](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1329s) 直接给出循环：测量 → 修改 → 再测量。

### 1.2 GPU 的四层常用存储

**【课程代码｜行 39–57｜视频 [00:30](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=30s)】**先看从近到远的仓库：

| 存储 | 首次定义 | 谁能直接使用 | 典型特征 |
|---|---|---|---|
| register | **寄存器**，thread 正在使用的极少量数字槽 | 单个 thread 私有 | 最小、最快；过多会压低 occupancy |
| L1 cache | **Level-1 cache，一级缓存** | 每个 SM 附近，由硬件管理 | 小而快；常与 shared-memory 容量共享物理资源 |
| shared memory | **共享内存** | 同一个 thread block/CTA 内 threads 显式协作 | 程序员可控制；有 32-bank 地址问题 |
| L2 cache | **Level-2 cache，二级缓存** | 整块 GPU 共享，由硬件管理 | 比 L1 大、比 HBM 近 |
| HBM | **High Bandwidth Memory，高带宽内存** | 整块 GPU 的 global memory | 最大、最远、带宽最低的一层常用设备内存 |

**SRAM（Static Random-Access Memory，静态随机存取存储器）**是常用于片上 register file、cache、shared-memory 等快速小容量存储结构的电路类型。本笔记后文说“留在 SRAM”，人话是“尽量留在 GPU 芯片附近的 register/shared/cache 层”，不是指另一个可无限分配的大 tensor 仓库。

**SM（Streaming Multiprocessor，流式多处理器）**是 GPU 芯片上的一个并行计算单元。一个 GPU 有许多 SM；每个 SM 有自己的 registers、L1/shared memory 和 warp scheduler（warp 调度器），而 L2/HBM 被更多 SM 共享。视频 [01:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=76s) 首次展开 SM；[01:52](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=112s) 区分 L1 与 shared memory；[02:17](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=137s) 展开 HBM。

### 1.3 课程硬件表是 2026 时点快照

**【课程代码｜行 42–54｜课程时点快照】**讲义列出：

| Accelerator | A100 | H100 | B200 |
|---|---:|---:|---:|
| SM 数 | 108 | 132 | 148 |
| registers / SM | 256 KB | 256 KB | 256 KB |
| L1 + shared / SM | 192 KB | 256 KB | 256 KB |
| L2 | 40 MB | 50 MB | 96–126 MB |
| HBM 容量 | 80 GB | 80 GB | 192 GB |
| register bandwidth | 约 116 TB/s | 约 401 TB/s | 约 447 TB/s |
| L1/shared bandwidth | 约 19 TB/s | 约 33 TB/s | 约 19 TB/s |
| L2 bandwidth | 约 5–8 TB/s | 约 12 TB/s | 约 9 TB/s |
| HBM bandwidth | 约 2 TB/s | 约 3.35 TB/s | 约 8 TB/s |

单位人话：

- `B` 是 byte（字节）；
- `KB/MB/GB` 是容量量级；厂商十进制与二进制口径可能不同；
- `TB/s` 是 terabytes per second（每秒万亿字节量级），表示带宽，不是容量；
- 65,536 个 32-bit register 的字节数是：

  $`65,536\times4=262,144\ \text{bytes}=256\ \text{KiB}.`$

  这里 $`1\ \text{KiB}=1,024`$ bytes。它解释了源码 occupancy 例的 65,536 registers 与表中“256 KB”为什么能对上。

这些数依赖具体 SKU、配置和测量方法。一般规律是“越靠近计算单元，容量通常越小、访问越快”，而不是必须背某个型号的 TB/s。老师在 [02:32](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=152s) 按带宽从 registers 讲到 HBM，并在 [02:59](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=179s) 总结“大而远较慢，小而近较快”。

### 1.4 算术强度从单位开始

**Arithmetic intensity（算术强度）**回答：“每从指定存储层搬 1 byte，做多少次浮点运算？”

```math
I=\frac{F}{Q}.
```

符号与单位：

- $`I`$：arithmetic intensity，单位 FLOPs/byte；
- $`F`$：这段工作完成的 **FLOPs（floating-point operations，浮点运算次数）**；
- $`Q`$：针对某个明确存储边界的数据流量，单位 bytes。

必须说清 $`Q`$ 是 HBM↔chip、shared↔register，还是别的层。只写“搬了多少”而不说层级，公式不完整。

**【补充例子】**32 个 FP32 数，每个做一次乘法和一次加法，再写回。每元素：

- 读 4 bytes；
- 写 4 bytes；
- 1 次乘法 + 1 次加法 = 2 FLOPs。

32 个元素：

```math
F=32\times2=64\ \text{FLOPs},
```

```math
Q_{HBM}=32\times(4+4)=256\ \text{bytes}.
```

所以：

```math
I_{HBM}=\frac{64}{256}=0.25\ \text{FLOPs/byte}.
```

若把两个逐元素操作拆成两个 kernels，中间结果还要写回并重读 HBM：每元素流量可能从 8 bytes 增到 16 bytes，工作仍为 2 FLOPs：

```math
I_{HBM}=\frac{2}{16}=0.125\ \text{FLOPs/byte}.
```

Fusion（融合）把两步留在一个 kernel 内，有机会消掉中间 HBM 往返，把强度从 0.125 提回 0.25。后续 GeLU 会把这个直觉落到真实代码。

### 1.5 本讲真正要优化的是“数据怎么走”

```text
HBM：放大张量，容量大
  ↓ 一次尽量连续、多取有用数据
L2：全芯片共享 cache
  ↓
L1/shared：block 内复用和协作
  ↓
register：thread 的当前值
  ↓
算术单元执行 FLOPs
```

**【课程代码｜行 66–73｜视频 [05:48](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=348s)】**thread block 的意义是让一组 threads 在同一 SM 上，通过 shared memory 协作。典型 kernel 流程是：从 HBM 读一块 → 在 SM 上复用/计算 → 写回 HBM。若每做一步小算术都回 HBM，GPU 可能大部分时间在搬数据。

---
## 2. GPU 编程模型：grid → CTA/block → warp → thread → SM

### 2.1 先分清“程序里的盒子”和“硬件上的机器”

**【课程代码｜行 58–80｜视频 [03:18](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=198s)】**CUDA 编程模型从大到小是：

1. **grid（网格）**：一次 kernel launch 创建的全部 thread blocks；
2. **CTA（Cooperative Thread Array，协作线程数组）**：NVIDIA 正式文档中 thread block 的名称；课程源码第 61 行写成 `concurrent` 是文字笔误；
3. **thread block（线程块）**：一组能通过 shared memory 协作的 threads；本文把 CTA 与 block 当同义词；
4. **warp（线程束）**：硬件把同一 block 内相邻的 32 threads 分成一组执行；
5. **thread（线程）**：执行 kernel 代码的最小逻辑实例，各自有 thread ID 和 registers。

物理硬件这一侧最重要的是 **SM（Streaming Multiprocessor）**。Block 是程序员定义的工作组；SM 是芯片上真正接收、调度并执行 block 的计算单元。不要说“block 就是 SM”。

```text
逻辑编程模型                         物理硬件

Grid
├─ CTA / Block 0  ──调度到──► 某个 SM
│  ├─ Warp 0: threads 0–31
│  └─ Warp 1: threads 32–63
├─ CTA / Block 1  ──调度到──► 某个 SM
└─ CTA / Block 2  ──等待或调度到某个 SM
```

一个 block 在其执行期间驻留于一个 SM，因为 block 内 threads 要看到同一块 shared memory。若 registers、shared memory、warp/block 上限都允许，一个 SM 可以同时 resident（驻留）多个 blocks；一个 block 通常不能为了负载均衡拆到多个普通 SM 上执行。

### 2.2 Kernel launch 的 grid 是怎样展开的

**【补充例子】**先用远小于真实 GPU 的教学 grid。假设输入有 32 个元素：

```text
grid_dim = 4 blocks
threads_per_block = 8
总 logical threads = 4 × 8 = 32
```

完整文字图：

```text
Grid
├─ Block 0: local thread 0 1 2 3 4 5 6 7
│            global item 0 1 2 3 4 5 6 7
├─ Block 1: local thread 0 1 2 3 4 5 6 7
│            global item 8 9 10 11 12 13 14 15
├─ Block 2: local thread 0 1 2 3 4 5 6 7
│            global item 16 17 18 19 20 21 22 23
└─ Block 3: local thread 0 1 2 3 4 5 6 7
             global item 24 25 26 27 28 29 30 31
```

若每 block 有 $`B_t=8`$ threads，block ID 为 $`b`$，block 内 thread ID 为 $`r`$，常见的一维全局索引是：

```math
i=bB_t+r.
```

逐个代数：

- block 0、thread 5：$`i=0\times8+5=5`$；
- block 1、thread 5：$`i=1\times8+5=13`$；
- block 3、thread 7：$`i=3\times8+7=31`$。

这个 8-thread block 只是为了把全部 ID 写完；真实 NVIDIA GPU 常把 block thread 数设成 32 的倍数，否则最后一个 warp 会有许多 inactive lanes（不工作的通道）。

### 2.3 为什么不能只有一大堆互不相干的 threads

**【课程代码｜行 66–73｜视频 [04:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=274s)】**若操作完全逐元素，例如每个数做 GeLU，一个 thread 处理一个元素很自然。但 softmax 需要知道整行的最大值与指数和；matrix multiplication（矩阵乘）需要多个输出复用 A/B 的数据。Threads 必须协作。

如果没有 block/shared memory，协作只能不断通过 HBM：

```text
thread A 计算部分结果 → 写 HBM
thread B 从 HBM 读部分结果 → 继续计算 → 再写 HBM
```

有 block 后：

```text
同一 block 的 threads 一起读一块数据
        ↓
放进同一 SM 的 shared memory
        ↓
多次复用、同步、归约
        ↓
最终结果写回 HBM
```

老师在 [05:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=305s) 用 softmax/matmul 说明 threads 必须通信；[06:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=370s) 明确把一个 block 调度到一个 SM；[06:40](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=400s) 预告 tiling 的核心就是这种块内复用。

### 2.4 一次调度不等于永久绑定

**【补充理解】**程序员通常定义“有多少 blocks、每个 block 做什么”，硬件 scheduler（调度器）决定 blocks 何时放到哪些 SM。若 grid 有 1,000 blocks 而 GPU 有 148 SM，逻辑上仍完全合法；blocks 会分波执行。

要分清三句话：

- 正确：一个普通 block 执行时驻留在一个 SM 上；
- 正确：一个 SM 在资源允许时可同时驻留多个 blocks；
- 错误：grid 中第 $`b`$ 个 block 永远固定属于第 $`b`$ 个 SM。

视频课堂问答 [20:19](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1219s) 讨论能否让 blocks 分享 SM。**Tensor Core（张量核心）**是 NVIDIA GPU 中专门加速小矩阵乘加的计算单元。答案的关键不是“一律不行”，而是资源：若一个 block 已吃满 Tensor Cores/registers/shared memory，再放一个也不会加速；若资源允许，硬件本来就可以让多个 blocks resident。

### 2.5 SIMT 与 PTX 放在编译链的哪里

**SIMT（Single Instruction, Multiple Threads，单指令多线程）**是 NVIDIA warp 的编程模型：一个 warp 的 32 个 threads 共同推进同一条 kernel 指令，但每个 thread 有自己的 ID、register values 和 active mask（当前是否参与）。它不是说 32 个 threads 的所有数据必须相同。

```text
PyTorch/Triton Python source
        ↓ 编译
PTX（Parallel Thread Execution，GPU 虚拟指令）
        ↓ driver 针对具体架构编译
machine instructions
        ↓
SM 以 SIMT/warp 方式执行
```

**【课程代码｜行 75–80｜视频 [07:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=436s)】**编程模型足以描述正确性，但性能高度依赖硬件细节。换句话说：相同 grid/block/thread 逻辑可以算出相同答案，却因 warp、register、bank、地址对齐而速度完全不同。

---

## 3. Warp lockstep 与 control divergence：32 人为什么要分两次走

### 3.1 Warp 是 32 个连续 threads 的执行组

**【课程代码｜行 82–90｜视频 [08:47](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=527s)】****Warp（线程束）**是 NVIDIA GPU 把同一个 block 内 threads 分成的 32-thread 组。典型编号：

```text
threads 0–31   → warp 0
threads 32–63  → warp 1
threads 64–95  → warp 2
threads 96–127 → warp 3
```

因此：

```math
\text{warps/block}=\frac{\text{threads/block}}{32}
```

只在 threads/block 是 32 的倍数时为整数。64 threads 是 2 warps；128 threads 是 4 warps。课程在 [09:09](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=549s) 定义 warp，并在 [09:31](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=571s) 说明 lockstep。

**Lockstep（锁步）**的人话是：一个 warp 在某次 instruction issue（发出指令）上共同执行同一条指令。不同 lanes（warp 内编号 0–31 的通道）可以用不同数据；暂时不应执行的 lanes 会被 mask 掉。

### 3.2 32-thread A/B 分支逐周期例

**【课程代码｜行 86–89｜视频 [09:44](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=584s)】**假设一个 warp 的 lanes 0–7 满足条件 A，lanes 8–31 满足条件 B：

```python
if lane_id < 8:
    do_A()       # 假设只需 1 条指令
else:
    do_B()       # 假设只需 1 条指令
```

逻辑上每个 thread 只走自己的分支；warp 执行上却要把两条路径分开发出：

| issue slot（教学上叫一拍） | 发出的指令 | active lanes | inactive lanes |
|---:|---|---|---|
| 1 | A | 0–7，共 8 个 | 8–31，共 24 个 |
| 2 | B | 8–31，共 24 个 | 0–7，共 8 个 |

若 A/B 各只有一条等成本指令，总 lane-slots 容量是：

```math
2\ \text{slots}\times32\ \text{lanes}=64.
```

真正 active 的 lane-slots：

```math
8\times1+24\times1=32.
```

这个极简分支的平均 lane 利用率：

```math
\frac{32}{64}=0.5=50\%.
```

没有 divergence 且全部 lanes 走同一条 1-instruction 路径时，只需 1 个 issue slot，利用率为 $`32/32=100\%`$。所以 divergence 的损失不是“答案错了”，而是同一个 warp 的不同路径要分别执行。

### 3.3 分支长度不同时怎样算

**【补充例子】**仍有 8 lanes 走 A、24 lanes 走 B，但 A 有 3 条指令、B 有 5 条：

```text
slots 1–3：执行 A1,A2,A3；每个 slot 只有 8 lanes active
slots 4–8：执行 B1...B5；每个 slot 有 24 lanes active
```

总 capacity：

```math
32\times(3+5)=256\ \text{lane-slots}.
```

有效 active lane-slots：

```math
8\times3+24\times5=24+120=144.
```

平均利用率：

```math
\frac{144}{256}=0.5625=56.25\%.
```

不能只看“8 对 24”就断言永远 50%；还要看每条 branch path 有多少实际指令。这里的“slot”是帮助计数的简化，不表示每条真实 GPU 指令都恰好只需一个物理 clock cycle。

### 3.4 哪些分支不造成同一 warp 的 divergence

若整个 warp 0 都走 A、整个 warp 1 都走 B：

```text
warp 0 lanes 0–31：全走 A
warp 1 lanes 0–31：全走 B
```

每个 warp 内部仍一致，所以没有 **warp 内** divergence；两个 warps 本来就可独立调度。优化分支时，关键是让条件尽量沿 32-thread warp 边界对齐，而不是禁止程序出现任何 `if`。

### 3.5 Warp switching 为什么能隐藏等待

**【课程内容｜视频 [10:18](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=618s)】**一个 SM 会保留多个 resident warps 的状态。当 warp A 等 HBM 数据时，warp scheduler 可选择已经 ready（可执行）的 warp B：

```text
time 0：warp A 发出 HBM load → 等待
time 1：SM 发出 warp B 的算术指令
time 2：SM 发出 warp C 的算术指令
...
later：warp A 的数据到达 → 再调度 A
```

课程口头称这种切换“zero cost”，应理解为：warp 的 register/context 已 resident，硬件不需要像 CPU 操作系统线程那样保存/恢复一大套软件上下文。它不表示调度、依赖或空闲永远没有代价。

这也解释 occupancy 的价值：resident warps 太少时，一个 warp 等 HBM，SM 可能找不到别的 ready warp；但 warps 多也会争 registers/shared memory，所以不是越多越快。

---

## 4. Occupancy：先复算 18.75%，再区分三种“没填满”

### 4.1 Warp occupancy 的定义

**Occupancy（占用率）**这个词必须带限定。本节首先说 **warp occupancy**：

```math
\text{warp occupancy}
=\frac{\text{resident warps per SM}}
{\text{hardware maximum resident warps per SM}}.
```

`resident` 是“资源已经分配、可被调度”，不等于每个时刻都正在发指令。Occupancy 也不是 FLOP/s，不直接等于利用率或速度。

**【课程代码｜行 92–113｜视频 [11:17](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=677s)】**每个 thread 使用更多 registers，单个 block 需要的 register pool 越大，一个 SM 同时容纳的 blocks/warps 可能越少。视频同时提醒低 occupancy 未必坏，因为 thread coarsening（线程粗化）可让每个 thread 做更多工作。

### 4.2 先披露源码的 64/128 内部不一致

源码第 97 行说明文字写：

```text
thread block has 64 threads
```

但真正进入计算的第 100–101 行变量是：

```python
num_threads_per_block = 128
num_registers_per_thread = 160
```

视频 [12:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=769s) 口头也明确说 128 threads。因此本节以 **128** 为实际课程例；不能把说明文字的 64 偷换进公式。

### 4.3 从 registers/block 开始逐步复算

已知：

```text
threads/block          = 128 threads
registers/thread       = 160 registers
registers/SM           = 65,536 registers
maximum warps/SM       = 64 warps
warp size              = 32 threads/warp
```

第一步，一个 block 要多少 registers：

```math
R_{block}
=128\ \text{threads/block}
\times160\ \text{registers/thread}
=20,480\ \text{registers/block}.
```

单位中的 `threads` 相消，留下 registers/block。

第二步，register 容量最多放几个完整 blocks：

```math
B_{resident}
=\left\lfloor\frac{65,536}{20,480}\right\rfloor.
```

逐个试：

```math
20,480\times3=61,440\le65,536,
```

```math
20,480\times4=81,920>65,536.
```

所以：

```math
B_{resident}=3\ \text{blocks/SM}.
```

此时剩余 registers：

```math
65,536-61,440=4,096,
```

不够再放需要 20,480 registers 的第 4 个 block。

第三步，一个 128-thread block 有多少 warps：

```math
\frac{128\ \text{threads/block}}
{32\ \text{threads/warp}}
=4\ \text{warps/block}.
```

第四步，3 blocks 有多少 resident warps：

```math
3\ \text{blocks/SM}\times4\ \text{warps/block}
=12\ \text{warps/SM}.
```

第五步，除以硬件最大 64 warps/SM：

```math
\text{occupancy}
=\frac{12}{64}
=\frac{3}{16}
=0.1875
=18.75\%.
```

视频在 [13:20](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=800s) 逐步口算 registers/block、3 blocks、12 warps，并口头约成 18%；精确值是 18.75%。

### 4.4 如果误用文字中的 64，为什么结果看似没变

**【补充检查】**错误地取 64 threads/block：

```math
64\times160=10,240\ \text{registers/block},
```

```math
\left\lfloor\frac{65,536}{10,240}\right\rfloor=6\ \text{blocks},
```

```math
64/32=2\ \text{warps/block},
```

```math
6\times2=12\ \text{warps},
\qquad
12/64=18.75\%.
```

这个例子碰巧仍得到 12 warps，所以只看最终百分比发现不了前置不一致；但 resident blocks 是 6 而不是 3，后续 block 上限、shared-memory 用量和同步行为都可能不同。正确推导必须跟实际变量。

### 4.5 Occupancy 不是越高越好

高 occupancy 的潜在收益：有更多 warps 可在 HBM latency（延迟）期间切换。代价可能是：为了塞入更多 warps，减少每 thread registers，导致 register spilling（寄存器溢出：临时值被迫放到更慢的 memory）或让每 thread 做的工作过小。

因此真实优化问题是：

```text
occupancy 是否足以隐藏 latency？
        +
每个 resident warp 是否有足够有用工作？
        +
有没有 spilling / shared-memory / block 上限？
```

课程在 [11:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=710s) 明确说较低 occupancy 不一定坏；这是必须保留的条件，而不是小字免责声明。

### 4.6 Warp occupancy、block residency、wave quantization

三者比较：

| 名称 | 问题 | 典型公式/数字 |
|---|---|---|
| warp occupancy | 单个 SM resident 了多少 warps？ | 本例 $`12/64=18.75\%`$ |
| block residency | 单个 SM 同时能容纳多少 blocks？ | 本例受 registers 限制为 3 |
| wave utilization / wave quantization | 整个 grid 的最后一波是否填满所有 SM slots？ | 160 blocks 对 148 SMs 的尾波 |

**Wave（波次）**是“当前可以一起安排的一批 blocks”。**Wave quantization（波量化/尾波效应）**是 block 数不能整齐铺满硬件并行 slots 时，最后一波很稀。

### 4.7 160 blocks、148 SMs 的尾波完整计算

**【课程代码｜行 132–137｜课程时点快照｜视频 [18:15](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1095s)】**讲义用 B200 的 148 SMs，教学上假设每个 SM 此刻接一个 block：

第一波：

```math
\min(160,148)=148\ \text{blocks}.
```

剩余：

```math
160-148=12\ \text{blocks}.
```

第二波只用 12 个 SM slots，尾波瞬时 slot utilization：

```math
\frac{12}{148}
\approx0.081081
=8.11\%.
```

若所有 blocks 耗时完全相同，两波共提供的 slots 为：

```math
2\times148=296\ \text{block-slots}.
```

真正使用 160 个，跨两波的平均 slot utilization：

```math
\frac{160}{296}
\approx0.54054
=54.05\%.
```

这不是说整个 GPU 一定只有 54.05% compute utilization：真实 SM 可同时 resident 多个 blocks，blocks 时长不必相同，还有 memory/tensor-core 限制。本例只隔离“尾波”这一种损失。

老师在 [18:35](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1115s) 算 148+12 两波，在 [19:13](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1153s) 建议让 block 数与 SM 数更整齐。更一般地，block 很多、波数很多时，一小段尾波占总时间的比例也会下降；不必为了整除而破坏更重要的数据 locality（局部性）或 tile shape。

---

## 5. Shared-memory banks 与 HBM coalescing：都是地址问题，但不是同一个问题

### 5.1 先记住两个存储层

```text
shared memory bank conflict：发生在同一 SM、同一 block/warp 访问 shared memory
HBM coalescing：发生在 warp 向 global/HBM 地址发起 load/store
```

两者都喜欢规则地址，却不能混称：bank conflict 看“请求落到哪个 bank”；coalescing 看“请求落到几个 memory transaction/cache-line 范围”。

### 5.2 32 banks × 4 bytes 到底是什么意思

**【课程代码｜行 114–125｜视频 [14:24](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=864s)】**教学模型中 shared memory 有 32 个 banks，每个 bank 每个 clock 可服务一个 32-bit（4-byte）word。地址按连续 4-byte word 轮流映射到 banks：

```math
\mathrm{bank}(a)
=\left(\frac{a}{4}\right)\bmod32,
```

其中：

- $`a`$：从一个 4-byte 对齐基址开始算的 byte address；
- $`a/4`$：这是第几个 32-bit word；
- `mod 32`：除以 32 后取余数，结果为 bank 0–31。

例如：

| byte address $`a`$ | word index $`a/4`$ | bank |
|---:|---:|---:|
| 0 | 0 | 0 |
| 4 | 1 | 1 |
| 124 | 31 | 31 |
| 128 | 32 | 0 |
| 132 | 33 | 1 |

`32 banks × 4 bytes = 128 bytes` 只表示“一排连续 32-bit words 横跨全部 banks”，**不是** shared memory 总容量只有 128 bytes。每个 bank 里还有很多后续 words，地址 128 又回到 bank 0 的下一行。

### 5.3 路由表一：连续 FP32，无 bank conflict

假设 warp lane $`i`$ 访问 FP32 word $`i`$，byte address 为 $`4i`$：

```math
\mathrm{bank}(4i)
=\left(\frac{4i}{4}\right)\bmod32
=i\bmod32=i.
```

完整路由：

```text
lane:  0  1  2  3  4  ... 28 29 30 31
word:  0  1  2  3  4  ... 28 29 30 31
bank:  0  1  2  3  4  ... 28 29 30 31
```

32 lanes 去 32 个不同 banks，可并行服务。这是 bank-friendly（对 bank 友好）的访问。

### 5.4 路由表二：stride 32 FP32，32-way conflict

**Stride（步长）**是相邻 thread 地址相差多少个元素。Stride 32 表示 lane $`i`$ 访问 word $`32i`$，byte address 为 $`4\times32i=128i`$：

```math
\mathrm{bank}(128i)
=\left(\frac{128i}{4}\right)\bmod32
=(32i)\bmod32
=0.
```

完整模式：

```text
lane:  0   1   2   3   4  ... 31
word:  0  32  64  96 128  ... 992
bank:  0   0   0   0   0  ... 0
addr:  0 128 256 384 512  ... 3968 bytes
```

32 lanes 请求 bank 0 中 **32 个不同 locations**。在课程简化模型下必须序列化成 32 份服务，叫 **32-way bank conflict**。视频 [15:14](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=914s) 从“同 bank 的不同地址排队”解释 conflict；[15:25](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=925s) 给出按矩阵第一列访问的最坏例。

### 5.5 路由表三：同一地址读取是 broadcast，不是 conflict

若 32 lanes 都读取 word 0：

```text
lane:  0 1 2 3 ... 31
word:  0 0 0 0 ... 0
bank:  0 0 0 0 ... 0
```

虽然全落 bank 0，但地址完全相同。现代 CUDA shared memory 可 **broadcast（广播）**同一个读取结果给所有请求 lanes，因此不按 32 个不同请求序列化。

必须比较：

| 模式 | bank | 地址 | 结果 |
|---|---|---|---|
| stride 32 | 都是 bank 0 | 32 个不同 words | 32-way conflict |
| broadcast | 都是 bank 0 | 同一个 word | broadcast，无普通 read conflict |

课程代码第 120 行的括号 `if not the same exact location` 正是在保留这个例外。NVIDIA 官方 CUDA 文档也把同一位置读取列为 broadcast 例外。

### 5.6 Swizzling 只讲到必要边界

**Swizzling（地址重排）**通过改变 logical row/column 到 physical shared-memory address 的映射，让本来同 bank 的列访问分散到不同 banks。课程第 124 行给 `row xor col` 作为直觉示例。

这里的 `xor` 是 bitwise exclusive OR（按位异或）：相同 bit 得 0，不同 bit 得 1。例如：

```text
row = 01₂
col = 11₂
row xor col = 10₂
```

完整 tensor-core swizzle 依架构、tile layout、element size 而变化；第一次阅读只需知道“重新排 shared 地址，不改变数学矩阵内容”。视频 [16:24](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=984s) 也只把它作为解决方向，没有展开生产实现。

### 5.7 HBM coalescing 的 128-byte 教学模型

**【课程代码｜行 126–131｜视频 [16:53](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1013s)】****Memory coalescing（内存合并访问）**是把一个 warp 的邻近 global-memory 请求合并成少量 transaction（事务）。课程用 128-byte cache line/transaction 做教学模型。

在这个模型中，byte address $`a`$ 所在的 128-byte line 编号是：

```math
\mathrm{line}(a)
=\left\lfloor\frac{a}{128}\right\rfloor.
```

Line 0 覆盖 byte 0–127，line 1 覆盖 128–255，依次类推。

### 5.8 连续且对齐：一个 transaction，100% 有效载荷

32 lanes 连续读取 FP32，lane $`i`$ 读 byte address $`4i`$：

```text
lane 0  → bytes 0–3
lane 1  → bytes 4–7
...
lane 31 → bytes 124–127
```

全部落在 line 0。请求有用 bytes：

```math
32\times4=128\ \text{bytes}.
```

移动一个 128-byte transaction：

```math
\text{load efficiency}
=\frac{128\ \text{useful bytes}}
{128\ \text{transferred bytes}}
=100\%.
```

视频 [17:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1025s) 说明 warp 请求被组合进 128-byte transaction；[17:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1054s) 展示连续访问的 full coalescing。

### 5.9 连续但错开 4 bytes：可能跨两个 lines

仍读 32 个连续 FP32，但第一个从 byte 4 开始：

```text
lane 0  → bytes 4–7      （line 0）
...
lane 30 → bytes 124–127  （line 0）
lane 31 → bytes 128–131  （line 1）
```

有用数据仍为 128 bytes，但覆盖两个 128-byte lines。按课程简化模型，搬运：

```math
2\times128=256\ \text{bytes}.
```

有效比例：

```math
\frac{128}{256}=0.5=50\%.
```

真实 GPU cache/**memory sector（内存扇区，即一个较大 cache line 或 memory transaction 内可单独搬运、记账的较小数据分片）**可能让细节不同；这个例子只教“连续还不够，对齐也影响 transaction 数”。

### 5.10 Stride 32：32 个 lines，只用每条 4 bytes

复用刚才 stride-32 FP32 地址：$`0,128,256,\ldots,3968`$。每个 lane 恰落到不同 128-byte line：

```math
32\ \text{transactions}\times128\ \text{bytes}
=4,096\ \text{transferred bytes}.
```

有用数据：

```math
32\times4=128\ \text{useful bytes}.
```

教学有效比例：

```math
\frac{128}{4,096}
=\frac1{32}
=0.03125
=3.125\%.
```

这同一地址模式在 shared memory 上产生 bank conflict，在 HBM 上产生 uncoalesced transactions，但机制仍不同。视频 [17:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1069s) 用“沿列访问会抓取许多没用数据”说明后者，并在 [18:04](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1084s) 明确说两者相似却属于不同约束。

### 5.11 本节路由检查表

看到一组地址时，按四步检查：

1. 数据类型每元素多少 bytes？
2. 这是 shared memory 还是 HBM/global memory？
3. Shared：用 `(byte_address/4) mod 32` 列 bank；相同 bank 是否其实同一地址 broadcast？
4. HBM：用 `floor(byte_address/128)` 列 line；请求覆盖几个 transactions，有用 bytes 比例多少？

---

## 6. Benchmark 与 profile：先量总时间，再找时间花在哪

### 6.1 先定义“你到底想量哪段时间”

**【课程代码｜行 144–165｜视频 [21:55](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1315s)】**

- **Benchmark（基准测试）**是“在写清输入和边界后，重复测某个目标的时间或吞吐”的泛称。目标可以是 GPU device elapsed，也可以是用户可见的 application wall-clock；两者不是同一数字；
- **Profile（性能剖析）**回答“运行中调用了哪些 CPU/GPU operations/kernels，各自花多久、调用几次、使用哪些硬件资源？”

课程 `benchmark()` 的具体目标是 **同一 GPU stream 中 start/end 两个 CUDA events 之间的 device elapsed time**。Python 调用和 CPU 等待时间通常不会被直接计入，但若 GPU 执行完 start event 后一度等着 CPU 提交后续 kernel，这段设备时间线上的空档仍可能落在两个 events 之间。因此它不是完整应用从函数入口到返回的 wall-clock。若要更接近用户可见端到端，应在正确同步的前提下，用 CPU wall timer 包住 launch 与等待；还要明确输入创建、编译和数据传输是否包含在范围内。

比喻：device-event benchmark 像只量火车真正行驶 30 分钟；同步包围的 application wall-clock 还可能包含进站、等车、发令和下车。Profile 再把选定区间拆成各个活动。只 profile 不看目标总时间，可能优化了一个小步骤却让用户等待更久；只 benchmark 不 profile，又不知道该改哪里。

课程的 recipe：

```text
1. benchmark + profile
2. 修改
3. 再 benchmark + profile
```

老师在 [22:36](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1356s) 用“端到端”描述 benchmark；结合源码，应读成“所选 GPU operation 在两个 events 之间的完整 device 区间”，不能扩张成完整 application wall-clock。视频 [26:25](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1585s) 定义 profile 为时间分解。

### 6.2 为什么 `CPU stop - CPU start` 会骗你

**CUDA（Compute Unified Device Architecture，统一计算设备架构）**是 NVIDIA 的 GPU 编程平台/接口体系。CUDA kernel launch 对 CPU 通常是 asynchronous（异步）的：CPU 把命令排队后可以先返回，不必等 GPU 做完。

错误时间线：

```text
CPU:  t0 start ─ launch kernel ─ t1 stop
                    │
                    └────命令进入 GPU queue
GPU:                        [稍后真正执行很久]
```

若 CPU timer 只量 `t1-t0`，可能主要量到 dispatch（发令）开销，而非 kernel execution。

正确理解 `synchronize`：

```text
CPU: enqueue work ─ synchronize() 等待 ─────► work 已完成
GPU:       [执行先前排队工作] ──────────────►
```

**`torch.cuda.synchronize()`**让调用它的 CPU thread 阻塞，直到此前 GPU 工作完成。课程代码第 185 行先在 warmup 后同步，第 198 行在每个 timed trial 的 end event 后同步。视频 [24:57](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1497s) 明确解释 GPU 异步执行与 synchronization barrier（同步屏障）。

### 6.3 Warmup 为什么不能混进 steady-state

**Warmup（预热）**是在正式计时前先运行若干次。第一次可能额外包含：

- lazy compilation（首次才编译）；
- library/kernel 选择与 autotuning；
- memory allocation/cache 初始化；
- GPU power/clock 状态变化。

若目标是重复训练步骤的 steady-state（稳定阶段）速度，就不应把一次性启动成本混进平均。课程代码第 179–185 行默认 warmup 1 次；视频 [24:04](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1444s) 说明预热是为了排除首次编译等成本。

边界：若用户真实工作就是“只运行一次”，冷启动成本本身就重要，应另做 cold-start benchmark，不能永远丢掉。

### 6.4 CUDA events 怎样排进 GPU 时间线

**CUDA event（CUDA 事件）**是排入 GPU stream（有顺序的工作队列）的标记，可记录 GPU 到达该点的时间戳。课程流程：

```text
CPU 发令顺序：record(start) → launch kernel → record(end) → synchronize

GPU stream： [start timestamp] → [kernel execution] → [end timestamp]
                                                    ↓
elapsed_time(start,end) 读取 GPU 两事件之间的毫秒数
```

因为 start/end 与 kernel 在同一有序 stream，`elapsed_time(start,end)` 覆盖的是 GPU stream 从 start event 到 end event 的 **device elapsed**。CPU 随后的 `synchronize()` 只是等待 end 已完成；它花掉的 CPU 等待时间不被直接加进两个 event 的时间差。Python 循环、event 对象创建、host launch 与 synchronize 的 CPU 耗时通常不直接计入；但 GPU 若在 start 与 end 之间等 CPU 继续提交工作，设备时间线上的空档仍会被计入。视频 [24:41](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1481s) 介绍 start/end CUDA events。

若测用户可见的同步调用时间，可写：

```python
import time
import torch

torch.cuda.synchronize()
t0 = time.perf_counter()
run()
torch.cuda.synchronize()
t1 = time.perf_counter()
wall_seconds = t1 - t0
```

这个 wall timer 包含 host launch 与等待，更接近调用者感受到的同步端到端；它仍不自动包含计时区间外的 input allocation、JIT 或 CPU preprocessing。两种计时都合理，前提是先声明目标。

**【课程代码｜行 179–203】**可运行骨架需要显式 import：

```python
import torch


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def benchmark(run, num_warmups: int = 1, num_trials: int = 3) -> float:
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # milliseconds

    return mean(times)
```

逐行状态变化：

1. warmup 的返回值不要计入 `times`；
2. 第一个 synchronize 清掉 warmup 的未完成工作；
3. 每个 trial 新建 start/end events；
4. `record()` 是把事件排入 stream，不是 CPU 当场替 GPU完成工作；
5. `run()` 把被测 kernel 排在两事件之间；
6. synchronize 等 GPU 到达 end；
7. `elapsed_time` 返回毫秒；
8. `mean(times)` 返回算术平均。

### 6.5 为什么不只跑一次：均值、方差、中位数

**【课程代码｜行 187–203｜视频 [25:24](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1524s)】**课程跑多次以观察 variance（方差），最终只取 mean（均值）。

均值：

```math
\bar t=\frac{t_1+t_2+\cdots+t_n}{n}.
```

**【补充例子】**四次时间 `[1,1,1,5] ms`：

```math
\bar t=\frac{1+1+1+5}{4}=\frac8{4}=2\ \text{ms}.
```

Population variance（把这四次当整组数据的总体方差）：

```math
\sigma^2
=\frac{(1-2)^2+(1-2)^2+(1-2)^2+(5-2)^2}{4}.
```

逐项：

```math
=\frac{1+1+1+9}{4}
=\frac{12}{4}
=3\ \text{ms}^2.
```

Standard deviation（标准差）：

```math
\sigma=\sqrt3\approx1.732\ \text{ms}.
```

Median（中位数）是排序后的中间值。偶数个样本取中间两个平均：

```math
\mathrm{median}([1,1,1,5])
=\frac{1+1}{2}=1\ \text{ms}.
```

这个 5 ms outlier（离群值）把 mean 从常见的 1 ms 拉到 2 ms；median 仍为 1 ms。因此可靠报告常同时给 median、**p95（95th percentile，第 95 百分位：约 95% 的样本不超过这个值）**、分布或误差条。同一段视频口头也提到更严谨时应看完整 distribution/p95；这不是课程代码默认实现。

### 6.6 小矩阵为什么看不出 $`n^3`$

**【课程代码｜行 167–176｜视频 [25:45](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1545s)】**方阵 matmul 的算术量随维度约按 $`n^3`$ 增长，但视频曲线中较小维度的时间近似常数，约到 2,000 维才显出增长。

原因不是数学从 $`n^3`$ 变成 $`n^0`$，而是小工作无法填满 GPU，launch、调度、固定延迟等占比大：

```text
粗略总时间 = 固定开销 + n³ 工作 / 实际吞吐
```

若固定开销是 10 微秒，而小 matmul 主计算只需 1 微秒：总计约 11 微秒；主计算翻 4 倍变 4 微秒，总计 14 微秒，没有按 4 倍明显增长。工作足够大后，$`n^3`$ 项才主导。

### 6.7 Profile 看见的是底层实际 kernels

**【课程代码｜行 206–259｜视频 [27:24](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1644s)】**PyTorch profiler 在 `ProfilerActivity.CUDA` 下记录 CUDA activities，并按 `cuda_time_total` 排序。课程还提到 **NVIDIA Nsight**：NVIDIA 的 profiler 工具系列，可进一步看 timeline、memory、occupancy、bank conflicts 等。

课程示例：

```text
cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_...
```

只作课堂机器示例拆解：

- `cutlass`：NVIDIA CUTLASS 线性代数 kernel 库；
- `sm100`：课程解释为 Blackwell/B200 架构目标；
- `simt`：SIMT 路径标记；
- `sgemm`：single-precision general matrix multiplication，单精度通用矩阵乘；
- `f32`：32-bit floating point；
- `64x64x16`：该 kernel 名称编码的 tile shape，正式 tiling 后讲。

视频 [27:54](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1674s) 显示 tensor add 也落成一个底层 CUDA kernel；[28:24](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1704s) 显示不同 matmul shapes 选择不同 kernels；[29:27](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1767s) 解释名字中的 CUTLASS、SM100、F32 和 tile。

这个长名字依赖 PyTorch、CUDA、CUTLASS、GPU 架构和 tensor shape。读者不应期待自己机器出现完全相同字符串；要学习的是“高层 `a @ b` 下面有真实 kernel，shape 改变可能换实现”。

### 6.8 Benchmark/profile 的闭环检查单

```text
[ ] 写清被测输入 shape、dtype、device
[ ] 区分 cold start 与 steady state
[ ] warmup 后 synchronize
[ ] 用 CUDA events 或正确同步的计时方法
[ ] 多 trial，报告 mean 之外的分布/median（重要时）
[ ] profile 确认到底运行了哪些 kernels
[ ] 修改后重新测；同时检查 numerical correctness
```

**【课程内容｜视频 [30:01](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1801s)】**老师在进入 GeLU 例前再次总结“记得 benchmark/profile”。后半讲会用 profiler 发现 naive GeLU 启动多个 kernels、反复读写 HBM，再引出 fusion 与 Triton；这些内容从 §7 开始逐步展开。

### 6.9 本讲前半的一手补充来源

- [NVIDIA CUDA Programming Guide：Programming Model、Warps 与 SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [NVIDIA CUDA Programming Guide：Shared-memory access patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA CUDA C++ Best Practices Guide：timing、CUDA events、bank conflicts](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA CUDA Programming Guide：Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [PyTorch 官方 Benchmark recipe](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch 官方 Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

【来源边界】CUDA/NVIDIA 与 PyTorch 官方文档用于核对一般编程模型、异步计时、bank broadcast 等机制；课程代码和视频仍决定本讲教学顺序。所有 8-thread grid、A/B lane-slots、错位 4-byte transaction、stride-32 效率、均值/方差例均为本笔记教学构造，不冒充课程原数据。

## 7. GeLU、多个小 kernel 与 fusion

### 7.1 先看 GeLU 到底在算什么

**【课程代码｜行 262–303｜视频 [30:14](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1814s)】**GeLU 是 **Gaussian Error Linear Unit**，中文常译“高斯误差线性单元”。它是逐元素 activation function（激活函数）：输入 tensor 中每个数独立经过同一个小公式，shape 不变。

课程代码使用常见的 tanh 近似：

```math
\mathrm{GeLU}(x)
\approx
\frac12x\left[1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)\right].
```

先逐个解释符号：

- $`x`$：一个输入数字。tensor 有多少元素，就对多少个 $`x`$ 各算一次；
- $`x^3=x\times x\times x`$；
- $`\pi\approx3.14159265`$；
- $`\sqrt{2/\pi}\approx0.79788456`$；$`\sqrt z`$ 是平方后等于 $`z`$ 的非负数；
- $`\tanh(a)`$：hyperbolic tangent（双曲正切），把任意实数压到 $`(-1,1)`$；本讲不要求推导它，只需会按计算器的 `tanh` 键；
- $`\approx`$：近似相等。这是 exact GeLU 的常用近似，不是代数恒等式；
- 最外面的 $`\frac12x`$ 让正数大多保留、负数大多压小，但不像 ReLU 那样把所有负数直接变成 0。

把长式子拆成小步：

```math
\begin{aligned}
u&=x^3,\\
v&=x+0.044715u,\\
a&=0.79788456v,\\
t&=\tanh(a),\\
y&=0.5x(1+t).
\end{aligned}
```

这是数学步骤，不等于承诺 GPU 一定启动五个或九个 kernels。kernel 是否融合，要看 eager/builtin/compiler、版本、dtype、shape 和硬件，§7.4 会单独说。

### 7.2 手算 $`x=0`$

**【课程内容 + 补充手算｜视频 [30:32](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1832s)】**

```math
x^3=0^3=0.
```

```math
v=0+0.044715\times0=0.
```

```math
a=0.79788456\times0=0.
```

```math
\tanh(0)=0.
```

所以：

```math
\mathrm{GeLU}(0)
\approx0.5\times0\times(1+0)=0.
```

### 7.3 手算 $`x=1`$ 与 $`x=-1`$

**【补充】**先算 $`x=1`$：

```math
x^3=1.
```

```math
v=1+0.044715\times1=1.044715.
```

```math
a=0.79788456\times1.044715
\approx0.83356197.
```

计算器输入 `tanh(0.83356197)`：

```math
t\approx0.68238398.
```

最后：

```math
y=0.5\times1\times(1+0.68238398)
=0.84119199.
```

再算 $`x=-1`$：

```math
x^3=(-1)^3=-1,
```

```math
v=-1+0.044715\times(-1)=-1.044715,
```

```math
a=0.79788456\times(-1.044715)
\approx-0.83356197,
```

```math
t=\tanh(-0.83356197)
\approx-0.68238398.
```

于是：

```math
y=0.5\times(-1)\times(1-0.68238398)
\approx-0.15880801.
```

检查直觉：正的 1 变成约 0.841；负的 $`-1`$ 只留下约 $`-0.159`$。GeLU 不是“所有负数归零”。

### 7.4 一行 PyTorch 为什么可能变成多个 kernels

**【课程代码｜行 694–695｜视频 [30:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1849s)】**课程的 eager PyTorch 写法是：

```python
def naive_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (
        1.0
        + torch.tanh(
            0.79788456 * (x + 0.044715 * x * x * x)
        )
    )
```

**eager** 是“Python 执行到一个操作，就立刻请求执行它”的默认模式。把表达式按依赖拆开，会看见这些概念节点：

```text
x*x        -> 临时量 u1
u1*x       -> 临时量 u2 = x³
0.044715*u2-> 临时量 u3
x+u3       -> 临时量 u4
0.79788456*u4 -> 临时量 u5
tanh(u5)   -> 临时量 u6
1+u6       -> 临时量 u7
x*u7       -> 临时量 u8
0.5*u8     -> 输出 y
```

这列出九个 **pointwise operations**（逐元素操作）。它不是“固定启动九个 CUDA kernels”的保证：

- 某个 **backend（后端，即接收 PyTorch 计算图并选择/生成底层 CPU、CUDA 或 Triton 实现的编译执行层）**可能在单个 kernel 内合并相邻标量操作；
- 某些表达式可能被改写；
- 临时量可能留在 cache，不一定每次真去 HBM；
- 版本、dtype、shape 和 GPU 会改变实现；
- 判断事实要看当前机器的 profiler，而不是数 Python 运算符。

**【课程视频｜[32:33](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1953s)–[33:23](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2003s)】**课堂 profiler 的核心观察是：该机器上的 naive 版本出现多个底层 pointwise kernels，并有中间结果读写；不是要求读者背某个永久不变的 kernel 数。

### 7.5 builtin、`torch.compile` 和 fusion

**【课程代码｜行 270–303｜视频 [31:03](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1863s)】**三条路径：

```python
# 1. naive：多个高层逐元素操作
y_naive = naive_gelu(x)

# 2. builtin：PyTorch 提供的 GeLU 实现
y_builtin = torch.nn.functional.gelu(x, approximate="tanh")

# 3. compiled：编译器观察整个函数图后尝试优化
compiled_gelu = torch.compile(naive_gelu)
y_compiled = compiled_gelu(x)
```

- **builtin**：框架作者已经提供的操作实现。课程 profiler 中它落到一个融合 kernel；这仍是课堂软硬件快照。
- **`torch.compile`**：PyTorch 捕获计算图，再生成优化代码；课程机器上生成一个 Triton kernel。
- **fusion（融合）**：让多个相邻操作在同一个 kernel 中完成，使中间值尽量留在 register/SRAM，而不必每步物化到 HBM。

**【课程视频｜[33:54](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2034s)、[34:42](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2082s)】**课程 profiler 分别展示 builtin 与 compiled 的单-kernel 结果。正确结论是“在该例、该环境里成功融合”，不是“builtin 永远一 kernel”或“`torch.compile` 永远能融合任意程序”。

Fusion 常见边界包括：

- 中间 tensor 被别处使用，不能随便消失；
- 两步之间有全局同步、复杂 reduction 或不支持的操作；
- 动态 shape、别名关系或副作用让编译器不敢改；
- 一个过大的融合 kernel 可能寄存器压力过高，反而降低 occupancy；
- 数值语义必须被保留，不能为省流量任意改运算顺序。

### 7.6 $`N=16{,}384`$、FP32 的教学流量账

**【补充】**下面依据课程 fusion 直觉做流量推导。设输入是一维 tensor：

```math
N=16{,}384,
```

每个元素用 FP32，即 4 bytes。

先算“理想融合下界”：只从 HBM 读一次输入、只写一次输出。

元素传输次数：

```math
N\ \text{reads}+N\ \text{writes}=2N.
```

代入：

```math
2\times16{,}384=32{,}768\ \text{element-transfers}.
```

换成 bytes：

```math
32{,}768\times4
=131{,}072\ \text{bytes}.
```

因为 $`1\ \text{KiB}=1{,}024\ \text{bytes}`$：

```math
131{,}072/1{,}024=128\ \text{KiB}.
```

这里的 **element-transfer** 是“搬一个 tensor 元素一次”；不是“执行一个 FLOP”，也不是“一条 128-byte memory transaction”。

再做一个醒目标成“教学模型”的对照。若假设九个概念 pointwise stages 都各自：

1. 读一个 $`N`$ 元素缓冲区；
2. 写一个 $`N`$ 元素中间缓冲区；

那么：

```math
9\times2N=18N
```

次 element-transfers。换成 bytes：

```math
18\times16{,}384\times4
=1{,}179{,}648\ \text{bytes}
=1{,}152\ \text{KiB}.
```

这个 $`1{,}152`$ KiB **不是实测值，也不是严格上界**：

- 二元运算可能需要读两个数组，所以模型会少算某些读；
- 同一个 $`x`$ 在一个 kernel 内可能只 load 一次，所以按操作数数读又可能多算；
- L1/L2 cache 可能命中，逻辑读不全变成 HBM traffic；
- 编译器可能只融合其中一部分；
- profiler 才能告诉你实际 kernel 列表，硬件 counter 才能估计实际 HBM bytes。

它只帮助理解：若许多阶段真的把中间 tensor 来回写 HBM，流量会远高于 128 KiB 的理想融合下界。

### 7.7 Launch overhead 为什么也值得省

**【课程视频｜[35:03](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2103s)–[36:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2194s)】**每次 kernel launch 都有近似固定的准备成本：CPU/runtime 把工作入队、GPU 接收并调度、建立执行所需状态。它叫 **launch overhead（启动开销）**。

教学例：若每个极小 kernel 真计算只需 $`1\ \mu s`$，启动固定花 $`5\ \mu s`$：

```text
9 个小 kernel：9 × (5 + 1) = 54 μs
1 个融合 kernel：5 + 9 = 14 μs
```

这不是课程硬件实测，只说明小任务中固定开销会压过计算。大任务还必须同时检查 HBM traffic、计算吞吐、occupancy 与寄存器压力；不能只靠“kernel 少”判断速度。

## 8. CUDA → Triton → PTX：三层不要混在一起

### 8.1 三层各自回答什么问题

**【课程代码｜行 305–315｜视频 [36:51](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2211s)】**

| 层级 | 初学者心智模型 | 它主要回答 |
|---|---|---|
| CUDA C++ kernel | 程序员经常写“一个 CUDA thread 做什么” | 每个 thread 用索引处理哪些数据，threads 如何协作 |
| Triton kernel | 程序员写“一个 program instance 处理一个数据 block” | 整块向量怎样 load、算、store |
| PTX | NVIDIA 的 virtual ISA（虚拟指令集架构） | 编译后接近机器的 load、算术、分支、register 指令是什么 |

**ISA** 是 instruction set architecture（指令集架构）：规定“有哪些指令、寄存器怎样命名、指令怎样表达”。PTX 是 NVIDIA GPU 的虚拟 ISA；驱动还会把它变成特定 GPU 真正执行的 SASS，§10 细讲。

“Triton 按 block 写”是编程抽象，不代表硬件少了 threads。编译器仍会把一个 Triton program instance 映射到 CUDA threads/warps/CTAs 上执行。

### 8.2 Host wrapper 与 device kernel

**【课程代码｜行 326–364｜视频 [39:44](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2384s)】**一份 Triton 程序通常有两半：

```text
CPU 上的 Python host wrapper
    ├─检查输入
    ├─分配输出
    ├─决定 grid 和 meta-parameters
    └─launch
             ↓
GPU 上的 Triton device kernel
    ├─取得 program_id
    ├─算 offsets 和 mask
    ├─load
    ├─compute
    └─store
```

- **host**：发起工作的 CPU 一侧；
- **device**：执行并行计算的 GPU 一侧；
- **wrapper**：包住 kernel 的普通 Python 函数，准备 shape、输出和启动参数；
- **device kernel**：真正由 GPU 执行的函数。

不能在 device kernel 里随便运行任意 Python。`@triton.jit` 标出的函数会被 Triton 的 JIT 编译器处理。

### 8.3 JIT 是什么时候编译

**【课程视频｜[38:32](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2312s)】**JIT 是 **Just-In-Time compilation（即时编译）**：通常在第一次遇到某种参数/shape/meta-parameter 组合时生成并编译 kernel，之后可缓存复用。

因此第一次调用可能包含 compilation overhead（编译开销），不能直接拿来当 steady-state kernel 时间：

```text
第一次：Python wrapper + JIT compile + launch + execute
后续：Python wrapper + cache lookup + launch + execute
```

这也是 benchmark 要 warmup 的原因之一。

### 8.4 Grid 与 program instance

**【课程代码｜行 343–347｜视频 [41:07](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2467s)】**Triton 的 **grid** 指定要启动多少个 program instances。若一维输入有 $`N`$ 个元素、每个 program 最多处理 $`B`$ 个：

```math
\text{num\_programs}=\left\lceil\frac NB\right\rceil.
```

$`\lceil z\rceil`$ 是 ceiling（向上取整）：只要还有余数，就再开一个 program。

例：$`N=10,B=8`$：

```math
10/8=1.25,
```

向上取整为 2，所以 grid 是 `(2,)`：

```text
program 0 候选处理 offsets 0..7
program 1 候选处理 offsets 8..15；10..15 必须 mask 掉
```

**program instance** 是同一个 Triton kernel 的一次网格实例，类似 CUDA grid 中的一个 CTA/block，但它的实际 threads/warps 布局仍由编译配置和编译器决定。

### 8.5 Meta-parameter 与 `tl.constexpr`

**【课程代码｜行 366–390｜视频 [41:22](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2482s)】**

```python
@triton.jit
def triton_gelu_kernel(
    x_ptr,
    y_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    ...
```

- **parameter**：函数运行时接收的输入；例如 `x_ptr`、`num_elements`。
- **meta-parameter（元参数）**：控制编译特化和 launch 形状的参数；这里是 `BLOCK_SIZE`。
- `tl.constexpr`：告诉 Triton 该值在编译时已知。编译器可据此展开向量宽度、做常量传播并生成特化版本。

这不是说 `BLOCK_SIZE` 写死只能有一个值。调用时可传不同值，JIT 可能为新组合编译另一版本。

### 8.6 最重要的映射警告：`tl.arange` 元素不是 CUDA thread

**【课程讲解 + 必要澄清｜视频 [44:17](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2657s)、[46:33](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2793s)】**

```python
offsets = start + tl.arange(0, BLOCK_SIZE)
```

会构造一个长度为 `BLOCK_SIZE` 的 Triton 向量。例如 `BLOCK_SIZE=8` 时，语义上得到八个 offsets。

错误理解：

```text
tl.arange 的第 i 个元素 = 第 i 个 CUDA thread
```

正确理解：

```text
Triton 源码描述“这一个 program 要处理的一整组元素”。
编译器再决定由多少 threads/warps、每个 thread 处理多少元素、用哪些 registers 完成。
```

课程有时用“block size/threads”帮助建立直觉，那是教学简化。实际 mapping（映射）要从编译配置、PTX/SASS 或 profiler 看，不能从 `tl.arange` 长度直接断定 CUDA thread 数。

### 8.7 这一层次关系的最短复述

```text
Python host wrapper 决定 grid 和 meta-parameters
        ↓ JIT
Triton device kernel 描述一个 program 的块级向量工作
        ↓ compiler
PTX 描述虚拟GPU指令
        ↓ driver assembler
SASS 是某一GPU架构真正执行的机器指令
```

**【课程内容｜视频 [48:02](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2882s)】**视频说 Triton 源码是对实际线程执行方式的高层抽象。这里的重点不是“抽象不真实”，而是“抽象保留结果语义，把低层分工交给编译器”。

## 9. Triton GeLU：wrapper 与 kernel 一行一行读

### 9.1 一份足以跟读的完整代码

**【课程代码｜行 317–390｜视频 [39:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2390s)–[45:58](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2758s)】**下面保留课程结构，并把导入补齐。需要 NVIDIA CUDA GPU、与 CUDA 匹配的 PyTorch，以及 Triton 才能真正运行。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def triton_gelu_kernel(
    x_ptr,
    y_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp_2a = tl.exp(2.0 * a)
    tanh_a = (exp_2a - 1.0) / (exp_2a + 1.0)
    y = 0.5 * x * (1.0 + tanh_a)
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](
        x,
        y,
        num_elements,
        BLOCK_SIZE=block_size,
    )
    return y
```

课程源码为了展示等价的 tanh，会用指数恒等式：

```math
\tanh(a)=\frac{e^{2a}-1}{e^{2a}+1}.
```

$`e\approx2.71828`$ 是自然指数的底，`tl.exp(z)` 算 $`e^z`$。这条替换不改变本节的索引重点。

### 9.2 Wrapper：输入检查和输出分配

**【课程代码｜行 337–342｜视频 [40:31](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2431s)】**

```python
assert x.is_cuda
assert x.is_contiguous()
y = torch.empty_like(x)
num_elements = x.numel()
```

逐行翻译：

1. `x.is_cuda`：确认 `x` 放在 CUDA device 上；CPU pointer 不能交给这个 GPU kernel。
2. `x.is_contiguous()`：确认 tensor 的逻辑相邻元素在 storage 中按标准连续顺序排放。
3. **contiguous（连续）** 不等于“所有 tensor 天生连续”。例如 transpose 后的 view 可能有非标准 strides；这个 kernel 把输入当一维连续数组读，所以先拒绝非连续输入。
4. `torch.empty_like(x)`：分配一个与 `x` shape、dtype、device 相同的输出；`empty` 表示初始内容没有初始化，必须由 kernel 写好后才能读。
5. `x.numel()`：返回 tensor 的总元素数。若 shape 是 `[2, 3, 4]`，则 `numel=2×3×4=24`。

这里 kernel 不需要知道原始是 `[2,3,4]` 还是 `[24]`，因为 GeLU 对每个元素独立算，wrapper 把它们当连续的一维元素序列。

### 9.3 `cdiv`、grid 与启动语法

**【课程代码｜行 343–350｜视频 [41:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2476s)–[41:42](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2502s)】**

```python
block_size = 1024
num_blocks = triton.cdiv(num_elements, block_size)

triton_gelu_kernel[(num_blocks,)](
    x, y, num_elements, BLOCK_SIZE=block_size
)
```

- `cdiv(a,b)` 是 ceiling division（向上取整除法）：

```math
\mathrm{cdiv}(a,b)=\left\lceil\frac ab\right\rceil.
```

- `triton_gelu_kernel[(num_blocks,)]` 的方括号不是普通 Python 数组索引，而是 Triton launch syntax（启动语法）；`(num_blocks,)` 是一维 grid。
- 圆括号里的 `x,y,num_elements` 是运行时参数。
- `BLOCK_SIZE=block_size` 是编译时 meta-parameter。

例：`num_elements=2050, block_size=1024`：

```math
2050/1024=2.001953125,
```

向上取整得到 3 个 programs。最后一个 program 只有 offsets 2048、2049 有效，其余候选位置必须 mask。

### 9.4 Kernel 前四行：我是谁、我负责哪里

**【课程代码｜行 366–378｜视频 [42:37](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2557s)–[44:47](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2687s)】**

```python
pid = tl.program_id(axis=0)
start = pid * BLOCK_SIZE
offsets = start + tl.arange(0, BLOCK_SIZE)
mask = offsets < num_elements
```

逐行：

1. `tl.program_id(axis=0)` 取当前 program 在 grid 第 0 维的编号。两个 programs 就是 `pid=0,1`。
2. `start=pid*BLOCK_SIZE` 算本 program 的第一项全局元素编号。
3. `tl.arange(0,BLOCK_SIZE)` 产生语义上的向量 `[0,1,...,BLOCK_SIZE-1]`。
4. 加 `start`，把局部编号变成全局 offsets。
5. `offsets < num_elements` 对每一项比较，产生布尔 mask；`True` 可访问，`False` 不可访问。

### 9.5 $`N=10,B=8`$：两个 programs 全部列出

**【补充】**设输入：

```text
x = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
num_elements = 10
BLOCK_SIZE = 8
num_blocks = ceil(10/8) = 2
```

Program 0：

```math
pid=0,
\quad start=0\times8=0.
```

| lane-like 向量位置 | offset | `offset < 10` | load | store |
|---:|---:|---:|---|---|
| 0 | 0 | True | `x0` | `y0` |
| 1 | 1 | True | `x1` | `y1` |
| 2 | 2 | True | `x2` | `y2` |
| 3 | 3 | True | `x3` | `y3` |
| 4 | 4 | True | `x4` | `y4` |
| 5 | 5 | True | `x5` | `y5` |
| 6 | 6 | True | `x6` | `y6` |
| 7 | 7 | True | `x7` | `y7` |

Program 1：

```math
pid=1,
\quad start=1\times8=8.
```

| lane-like 向量位置 | offset | `offset < 10` | load | store |
|---:|---:|---:|---|---|
| 0 | 8 | True | `x8` | `y8` |
| 1 | 9 | True | `x9` | `y9` |
| 2 | 10 | False | 不访问 | 不写 |
| 3 | 11 | False | 不访问 | 不写 |
| 4 | 12 | False | 不访问 | 不写 |
| 5 | 13 | False | 不访问 | 不写 |
| 6 | 14 | False | 不访问 | 不写 |
| 7 | 15 | False | 不访问 | 不写 |

注意表头写“lane-like 向量位置”，没有写“CUDA thread”。这是 Triton 向量语义位置；实际 thread 分工由编译器决定。

如果删掉 mask，program 1 会试图读 `x10...x15` 和写 `y10...y15`。合法下标只有 0 到 9，所以这些地址越界。越界访问可能报错、读垃圾、污染别的数据，不能靠“最后几个结果不用”来合理化。

### 9.6 `tl.load`、pointer arithmetic 与 `tl.store`

**【课程代码｜行 379–390｜视频 [45:13](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2713s)–[46:02](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2762s)】**

```python
x = tl.load(x_ptr + offsets, mask=mask)
...
tl.store(y_ptr + offsets, y, mask=mask)
```

- `x_ptr` 是指向输入第 0 个元素的 typed pointer（带元素类型的指针）。
- `x_ptr + offsets` 的 offsets 按“元素”计，不是永远按 bytes 计。
- 若元素是 FP32，offset 增 1，实际 byte address 增 4；若元素是 FP16，增 1 对应 2 bytes。
- `tl.load` 让 mask=True 的位置从 memory 读入向量值。
- 课程这个 GeLU load 没写 `other=`；mask=False 项的计算值无需依赖，因为最后 store 也 mask 掉。不要读取或使用这些无效 lane 的结果。
- `tl.store` 只对 mask=True 的目标地址写回。

FP32 具体地址例：若 `x_ptr` 的 byte address 是 1000：

```text
offset 0 -> byte address 1000 + 0×4 = 1000
offset 1 -> byte address 1000 + 1×4 = 1004
offset 8 -> byte address 1000 + 8×4 = 1032
```

所以源码里的 `+8` 表示“第 8 个元素”，不是“向后 8 bytes”。

### 9.7 Kernel 里的 GeLU 与 reference 对上

**【课程代码｜行 323–329、379–390】**课程先用 PyTorch reference 检查 Triton 输出。最小核对：

| $`x`$ | $`a=0.79788456(x+0.044715x^3)`$ | $`\tanh(a)`$ | $`0.5x(1+\tanh(a))`$ |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0.83356197 | 0.68238398 | 0.84119199 |
| -1 | -0.83356197 | -0.68238398 | -0.15880801 |

Triton 用

```math
\frac{e^{2a}-1}{e^{2a}+1}
```

算 tanh；PyTorch reference 用 `torch.tanh(a)`。在普通大小输入上，有限浮点精度可能造成最后几位小差，所以检查函数一般用 tolerance（容差），而不是要求每个 bit 完全相同。

但课程这个指数改写是 **教学实现，不是完整数值稳定实现**。用 $`x=20`$ 做反例：

```math
x^3=20^3=8{,}000.
```

```math
x+0.044715x^3
=20+0.044715\times8{,}000
=20+357.72
=377.72.
```

```math
a=0.79788456\times377.72
\approx301.376956
\approx301.37.
```

```math
2a\approx602.753912\approx602.75.
```

若中间先粗略截断，可能写成约 $`602.74`$；无论取哪一个，结论相同：FP32 的有限最大值只能支持指数输入到约 88 左右，`exp(602.75)` 会 overflow（溢出）成 `inf`。课程代码随后做：

```math
\frac{\text{inf}-1}{\text{inf}+1}
=\frac{\text{inf}}{\text{inf}},
```

这个不定式可能得到 `NaN`，最终 GeLU 也可能 NaN。数值稳定的 `tanh(a)` 在 $`a`$ 很大时接近 1，所以正确近似应是：

```math
\mathrm{GeLU}(20)
\approx0.5\times20\times(1+1)
=20.
```

课程随机检查用标准随机输入，极端的 20 几乎不会被抽到，所以“随机 `allclose` 通过”不能覆盖这个反例。可靠测试还应显式加入大正数、大负数、`NaN`、`+Inf`、`-Inf`，并写清期望传播规则。生产实现应直接用数值稳定的 `tanh`/builtin，或使用避免中间指数溢出的等价实现。

### 9.8 本环境验证边界

本笔记工作环境中未发现可调用的 `nvidia-smi`、`nvcc` 或可用 Python/Triton CUDA runtime，所以本节没有声称运行 GPU kernel、生成本机 PTX 或复现课程 benchmark。已经完成的是：

- 对 744 行官方源码逐行读取；
- 对 wrapper/kernel 的索引做手算；
- 用独立基础数学复算 $`x=0,1,-1`$ 的 reference 数值，并复算 $`x=20`$ 的指数溢出反例；
- 对代码围栏、符号和结构做静态检查。

要在自己的 CUDA 环境运行，必须安装相互兼容的 NVIDIA driver、CUDA-enabled PyTorch 与 Triton，再把课程辅助的 `check_equal`/benchmark 函数一并带上。

## 10. PTX 与 thread coarsening：只从证据读结论

### 10.1 PTX 不是 GPU 最终机器码

**【课程代码｜行 351–362｜视频 [48:23](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2903s)–[51:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3060s)】**课程调用辅助函数输出编译后的 PTX。

- **PTX**：Parallel Thread Execution，NVIDIA 定义的虚拟 ISA，面向一个抽象 NVIDIA GPU。
- **SASS**：特定 GPU architecture 的真正 machine instructions（机器指令）；driver assembler 会把 PTX 进一步翻译为 SASS。

关系：

```text
Triton source --Triton compiler--> PTX
PTX --NVIDIA driver/toolchain--> 某块GPU的SASS
SASS --hardware--> 执行
```

因此从 PTX 可以理解低层方向，但不能把它当作某块 GPU 最终每周期怎样发射的完整答案。真正的 instruction scheduling、cache hit 和 latency 还需 SASS/hardware profiling。

### 10.2 课程 PTX 中能可靠认出的几类名字

**【课程视频｜[50:41](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3041s)–[54:52](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3292s)】**这里只解释课程展示中确实用来讲解的线索：

| PTX 线索 | 人话 |
|---|---|
| `%ctaid.x` | 当前 CTA/block 在 grid 的 x 维编号，和高层 `program_id(axis=0)` 有关联 |
| `%tid.x` | 当前 CUDA thread 在 CTA 内的 x 维编号 |
| `ld.global...` | 从 global address space 读数据；通常对应 GPU global memory 访问路径 |
| `st.global...` | 向 global address space 写数据 |
| `%r...`、`%f...` 等 | PTX virtual registers（虚拟寄存器）；前缀会体现类型类别 |

**【课程视频｜[51:43](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3103s)、[52:17](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3137s)】**老师用 `ld.global` 和 `st.global` 对应 GeLU 的 load/store，用 register 名称指出中间量不必物化为全局 tensor。

注意：看到 `.global` 说明 address space，不等于每条访问必然绕过 L1/L2、每次都打到 HBM。实际 memory transaction 要结合 cache policy 与 profiler。

### 10.3 Thread coarsening 是什么

**【课程内容｜视频 [52:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3169s)】****thread coarsening（线程粗化）**指一个 physical/logical CUDA thread 连续处理多个数据元素，而不是每个 thread 只处理一个元素。

教学对比：

```text
不粗化：thread 0 -> element 0
        thread 1 -> element 1
        ...

每thread粗化4项：thread 0 -> elements 0, 32, 64, 96
                 thread 1 -> elements 1, 33, 65, 97
                 ...
```

具体布局只是示意。粗化可能：

- 减少索引/调度开销；
- 增加每个 thread 的独立工作；
- 也增加 register pressure，过多会降低 occupancy。

### 10.4 “一 thread 处理 8 elements”只属于课程这次编译观察

**【课程视频｜[52:55](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3175s)–[53:52](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3232s)】**课程通过当时生成的 PTX 观察到：一个 thread 处理八个元素。这说明编译器没有把 `tl.arange` 每一项一一映射成 thread。

不能推广成：

```text
所有 Triton kernel 都是一 thread八元素
```

实际 coarsening factor 可能随这些条件变化：

- `BLOCK_SIZE`；
- `num_warps` 等编译 meta-parameters；
- dtype 与操作复杂度；
- Triton/compiler 版本；
- 目标 GPU 架构；
- register pressure 与 **vectorization（向量化，即让一条或一组生成指令批量处理多个数据元素）**决策。

如果 `BLOCK_SIZE=1024`，也不能从“课程观察为 8”倒推永远是 128 threads；必须查看本次生成代码或 profiler。

### 10.5 PTX 读码的初学者边界

按下面顺序就够：

1. 找 block/thread id；
2. 找 global loads/stores；
3. 看中间值是否主要在 registers；
4. 看每个 thread 是否重复处理多个 offsets；
5. 不从一段 PTX 猜整个 GPU 的 cache hit、warp scheduling 或实际 HBM bandwidth；
6. 需要最终硬件指令时再看 SASS，需要时间/traffic 时用 profiler/hardware counters。

**【课程视频｜[55:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3300s)–[55:18](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3318s)】**视频也提醒 PTX 没把 SM/warp 的实际调度全部写在源表面；通常 PTX 是编译器产物，不是人手写优化的首要层级。

## 11. Fused softmax：一行在片上完成 reduction

### 11.1 Softmax 是“每行变成和为 1 的非负权重”

**【课程代码｜行 392–418｜视频 [58:06](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3486s)】**对一行输入 $`x=[x_0,x_1,\ldots,x_{N-1}]`$，softmax 第 $`j`$ 项定义为：

```math
\mathrm{softmax}(x)_j
=\frac{e^{x_j}}{\sum_{k=0}^{N-1}e^{x_k}}.
```

逐个符号：

- $`j`$：当前要算的列；
- $`k`$：求和时从第 0 列走到第 $`N-1`$ 列；
- $`e^{x_j}`$：把第 $`j`$ 个 logit 指数化；
- 分母是这一行所有指数的和；
- 同一行每项除同一个正分母，所以结果非负且总和为 1。

**logit** 是 softmax 前尚未归一化的任意实数分数。它可以是负数，也不必加起来等于 1。

### 11.2 为什么先减 row maximum

**【课程视频｜[58:21](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3501s)–[59:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3550s)】**直接算很大的 $`e^x`$ 可能 overflow（溢出）。例如 FP32 中 $`e^{100}`$ 太大，不能用普通有限数表示。

设这一行最大值：

```math
m=\max_j x_j.
```

改算：

```math
\frac{e^{x_j-m}}{\sum_k e^{x_k-m}}.
```

为什么输出不变？分子：

```math
e^{x_j-m}=e^{x_j}e^{-m}.
```

分母：

```math
\sum_k e^{x_k-m}
=e^{-m}\sum_k e^{x_k}.
```

上下都有同一个 $`e^{-m}`$，相除抵消：

```math
\frac{e^{x_j}e^{-m}}{e^{-m}\sum_k e^{x_k}}
=\frac{e^{x_j}}{\sum_k e^{x_k}}.
```

减最大值后，每个 $`x_j-m\le0`$，至少一项等于 0；指数都不超过 $`e^0=1`$，就避免了正方向的指数溢出。

### 11.3 课程 $`2\times3`$ 输入逐行手算

**【课程】**输入取自官方代码行 404–408。**【补充】**以下把课程只运行给出的结果完整展开：

```math
X=
\begin{bmatrix}
5&5&5\\
0&0&100
\end{bmatrix}.
```

Shape 是 $`[M,N]=[2,3]`$：$`M=2`$ 行，$`N=3`$ 列。Softmax 独立处理每一行。

第一行 $`[5,5,5]`$：

1. row max：

```math
m_0=5.
```

2. 每项减最大值：

```math
[5-5,5-5,5-5]=[0,0,0].
```

3. 指数：

```math
[e^0,e^0,e^0]=[1,1,1].
```

4. 求和：

```math
d_0=1+1+1=3.
```

5. 归一化：

```math
y_0=\left[\frac13,\frac13,\frac13\right].
```

第二行 $`[0,0,100]`$：

1. row max：

```math
m_1=100.
```

2. 减最大值：

```math
[0-100,0-100,100-100]=[-100,-100,0].
```

3. 指数：

```math
[e^{-100},e^{-100},e^0]
\approx[3.72008\times10^{-44},3.72008\times10^{-44},1].
```

4. 求和：

```math
d_1=1+2\times3.72008\times10^{-44}
\approx1.
```

5. 归一化：

```math
y_1\approx
[3.72008\times10^{-44},3.72008\times10^{-44},1].
```

检查：两行每项都非负；第一行和为 $`1/3+1/3+1/3=1`$，第二行在显示精度下也约为 1。

### 11.4 Naive softmax 的五次大步骤

**【课程代码｜行 419–440｜视频 [58:39](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3519s)–[59:40](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3580s)】**

```python
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # x shape: [M, N]
    x_max = x.max(dim=1)[0]       # shape [M]
    shifted = x - x_max[:, None]  # shape [M, N]
    numerator = torch.exp(shifted)# shape [M, N]
    denominator = numerator.sum(dim=1)  # shape [M]
    y = numerator / denominator[:, None]# shape [M, N]
    return y
```

- `dim=1`：沿列方向归约，每一行得到一个数；
- `x_max[:,None]`：把 shape `[M]` 变成 `[M,1]`，才能按行 broadcast 到 `[M,N]`；
- **broadcast（广播）**：不手工复制数据，也能把 `[M,1]` 的每行标量用于该行全部 $`N`$ 项；
- **reduction（归约）**：把多项合成较少项，例如一行的 `max` 或 `sum` 从 $`N`$ 个数变 1 个数。

数学只写五行；eager 执行可能形成多个 kernels 和中间 tensors。实际 kernel 数仍由 profiler 确认。

### 11.5 严格复算课程的 reads：$`5MN+M`$

课程注释采用一个简化的 element-read accounting（元素读取记账）：

| 步骤 | 课程计的 reads | 为什么 |
|---|---:|---|
| 1. row max | $`MN`$ | 读完整输入矩阵一次 |
| 2. subtract max | $`MN+M`$ | 读矩阵 $`MN`$ 项，再读每行 max 共 $`M`$ 项 |
| 3. exponentiate | $`MN`$ | 读 shifted matrix |
| 4. row sum | $`MN`$ | 读 numerator matrix |
| 5. normalize | $`MN`$ | 读 numerator；课程模型把 denominator row scalar 的广播读取折叠/不另计 |

相加：

```math
MN+(MN+M)+MN+MN+MN.
```

先合五个 $`MN`$：

```math
=5MN+M\ \text{reads}.
```

这个口径必须说清：若你把最后一步从 HBM 读取每行 denominator 也另计一次，会再加 $`M`$，得到 $`5MN+2M`$。课程源码行 435–438 选择前一种简化，目的是抓住主导的 $`MN`$ 项；这里按课程口径复算，不把两套规则悄悄混用。

### 11.6 严格复算课程的 writes：$`3MN+2M`$

| 步骤 | writes | 写了什么 |
|---|---:|---|
| 1. row max | $`M`$ | 每行一个 maximum |
| 2. subtract max | $`MN`$ | shifted matrix |
| 3. exponentiate | $`MN`$ | numerator matrix |
| 4. row sum | $`M`$ | 每行一个 denominator |
| 5. normalize | $`MN`$ | 最终 output matrix |

相加：

```math
M+MN+MN+M+MN
=3MN+2M\ \text{writes}.
```

### 11.7 代 $`M=2,N=3`$：为什么是 32 reads、22 writes

**【补充】**下面严格沿用课程口径手算；$`MN=2\times3=6`$。

Reads 分项：

```text
row max       : MN     = 6
subtract max  : MN + M = 6 + 2 = 8
exponentiate  : MN     = 6
row sum       : MN     = 6
normalize     : MN     = 6
总 reads      : 6+8+6+6+6 = 32
```

公式复核：

```math
5MN+M=5\times2\times3+2=30+2=32.
```

Writes 分项：

```text
row max       : M  = 2
subtract max  : MN = 6
exponentiate  : MN = 6
row sum       : M  = 2
normalize     : MN = 6
总 writes     : 2+6+6+2+6 = 22
```

公式复核：

```math
3MN+2M=3\times2\times3+2\times2=18+4=22.
```

总 element-transfers：

```math
32+22=54.
```

若每项 FP32：

```math
54\times4=216\ \text{bytes}
```

是这个小例在教学模型里的流量；真实 memory transactions/cache traffic 不会简单等于 216 bytes。

### 11.8 Fused 理想流量与 $`4.5\times`$ 从哪里来

**【课程内容 + 补充推导｜视频 [60:23](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3623s)】**若一个 fused kernel 把一行读进片上存储，max、减法、exp、sum、division 都在片上完成，最后只写 output：

```math
MN\ \text{reads}+MN\ \text{writes}=2MN.
```

对 $`M=2,N=3`$：

```math
2MN=2\times2\times3=12
```

次 element-transfers，即 FP32 的：

```math
12\times4=48\ \text{bytes}.
```

教学流量倍数：

```math
\frac{54}{12}=4.5.
```

一般 $`M,N`$ 下，naive 总传输：

```math
(5MN+M)+(3MN+2M)=8MN+3M.
```

除以 fused 的 $`2MN`$：

```math
\frac{8MN+3M}{2MN}
=4+\frac{3}{2N}.
```

当 $`N`$ 很大，$`3/(2N)`$ 趋近 0，所以流量比趋近 $`4\times`$，不是趋近 $`4.5\times`$。

**流量减少 $`4\times`$ 不保证 wall-clock speedup 恰好 $`4\times`$**：还有 launch、exp/max/sum 的计算、同步、occupancy、cache、compiler 与行宽资源限制。这个比值是教学 traffic model 的上游线索，不是测速结果。

### 11.9 Wrapper：每行一个 program

**【课程代码｜行 443–459｜视频 [60:44](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3644s)–[61:51](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3711s)】**

```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    M, N = x.shape

    block_size = triton.next_power_of_2(N)
    triton_softmax_kernel[(M,)](
        x_ptr=x,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        num_cols=N,
        BLOCK_SIZE=block_size,
    )
    return y
```

- `next_power_of_2(N)`：返回不小于 $`N`$ 的最小 $`2^k`$。例：$`N=3`$ 得 4，$`N=8`$ 得 8，$`N=9`$ 得 16。
- 这里 grid 是 `(M,)`，所以 program 0 处理 row 0，program 1 处理 row 1，直到 row $`M-1`$。
- `x.stride(0)`：从一行第 0 项走到下一行第 0 项，要跨多少“元素”。连续 `[M,N]` tensor 通常是 $`N`$，但传 stride 比硬编码 $`N`$ 更清楚。
- 一整个 row 要在该 program 中完成 max/sum reduction，才能避免把中间矩阵写回 HBM。

这里有一个隐藏条件：wrapper **只传 row stride**，kernel 却用 `x_start_ptr + col_offsets` 和 `y_start_ptr + col_offsets`，等价于假定：

```text
x.stride(1) == 1
y.stride(1) == 1
```

它并不支持任意 strided tensor。完整反例：

```python
base = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
])
x = base.T

# x.shape  = [3, 2]
# x.stride = [1, 3]
# x逻辑row 0 = [1, 4]
```

底层 storage 顺序仍是 `[1,2,3,4,5,6]`。逻辑 row 0 应读 offsets：

```math
0\times1+[0\times3,1\times3]=[0,3],
```

得到 `[1,4]`。课程 kernel 却算：

```math
0\times1+[0,1]=[0,1],
```

读成 storage 的 `[1,2]`，答案错误。

此外，`torch.empty_like(x)` 默认采用 preserve-format 语义，可能保留这种非标准 strides；output 也不能假定 `+col_offsets` 正确。

两种修法：

1. 只支持 column-contiguous 布局：wrapper 明确 `assert x.stride(1)==1`，并分配/检查 `y.stride(1)==1`；
2. 真支持 strides：额外传 `x_col_stride`、`y_col_stride`，分别写成：

   ```python
   x_ptrs = x_start_ptr + col_offsets * x_col_stride
   y_ptrs = y_start_ptr + col_offsets * y_col_stride
   ```

### 11.10 Kernel 逐行：row pointer、padding 和 `-inf`

**【课程代码｜行 462–484｜视频 [62:28](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3748s)–[63:33](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3813s)】**

```python
@triton.jit
def triton_softmax_kernel(
    x_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    assert num_cols <= BLOCK_SIZE

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(
        x_ptrs,
        mask=col_offsets < num_cols,
        other=float("-inf"),
    )

    shifted = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(
        y_ptrs,
        y_row,
        mask=col_offsets < num_cols,
    )
```

逐块翻译：

1. `row_idx` 决定当前 program 处理哪一行。
2. `row_idx*x_row_stride` 算该行开头相对 tensor 开头的元素偏移。
3. `col_offsets` 列出本行候选列。
4. `mask=col_offsets<num_cols` 禁止 padding lane 真读越界地址。
5. `other=-inf` 给无效 lane 一个数学上的负无穷值。
6. row max 不会被 padding 扰乱，因为任何有限数都大于 $`-\infty`$。
7. 减去有限 row max 后，padding 仍为 $`-\infty`$。
8. $`e^{-\infty}=0`$，所以 padding 对 denominator 的 sum 贡献 0。
9. store 再用同一 mask，不把 padding 写到输出之外。

`other=-inf` 不只是“随便填个很小的数”；在**至少有一个有限有效 logit**的前提下，它恰好是 max/exp/sum softmax 链的中性处理：不抢 maximum，指数后变 0。

若一整行的所有有效值也都是 $`-\infty`$：

```math
\max(-\infty,-\infty,\ldots)=-\infty,
```

接着：

```math
-\infty-(-\infty)=\mathrm{NaN}.
```

后续 exp/sum/divide 会传播 NaN。数学上这行也对应 $`0/0`$，没有普通 softmax 概率分布。Attention 实现应避免产生 fully masked row，或显式规定这种行输出全 0 等特殊语义；不能用 padding 推导假装它自动安全。

### 11.11 $`N=3,B=4`$ 的 padding 表

**【补充手算】**以 row 0 的 `[5,5,5]` 为例，`next_power_of_2(3)=4`：

| `col_offset` | `offset < 3` | load 后值 | 减 row max 5 | exp |
|---:|---:|---:|---:|---:|
| 0 | True | 5 | 0 | 1 |
| 1 | True | 5 | 0 | 1 |
| 2 | True | 5 | 0 | 1 |
| 3 | False | $`-\infty`$ | $`-\infty`$ | 0 |

Reduction：

```math
\max(5,5,5,-\infty)=5,
```

```math
1+1+1+0=3.
```

归一化向量在 Triton 语义上是：

```math
\left[\frac13,\frac13,\frac13,0\right].
```

最终 store mask 只写前三项；第四项连输出地址都不写。

对 row 1，若连续 row stride 是 3：

```text
row_idx = 1
x_start_ptr = x_ptr + 1×3
候选元素 offsets = [3,4,5,6]
有效的是全局元素3,4,5；元素6被mask
```

这同时说明 row pointer 与 column offsets 是怎样合成二维访问的。

### 11.12 每行一个 program 的收益与边界

**【课程内容｜视频 [64:03](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3843s)–[65:23](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3923s)】**收益是：同一个 program 看见整行，可在片上做 max 和 sum，避免多次把整行中间结果写回 HBM。

但“每行一个 program”不是无限扩展的魔法。行很宽时：

- `next_power_of_2(N)` padding 会浪费更多 lanes；
- 整行中间值需要更多 registers/SRAM；
- register pressure 可能降低 occupancy 或发生 spill（溢出到较慢 memory）；
- reduction 的线程间通信成本增长；
- 某些 block/vector 宽度超过 compiler 或 hardware 支持范围。

因此这个教学 kernel 有适用行宽。更宽的行可能需要分块、多阶段 reduction 或不同算法。§12 用更简单的 row sum 展示“同一 program 循环多个 tiles”；它说明 tiling 的骨架，但不等于已经给出任意宽度 softmax 的完整生产实现。

### 11.13 从 GeLU 到 fused softmax 的因果链

```text
GeLU：每个元素互相独立
  -> 一个program处理一块连续元素
  -> 融合后中间量留片上

Softmax：同一行元素必须共同求max与sum
  -> 一个program看见整行
  -> 先减max保证数值稳定
  -> padding用 -inf，exp后为0
  -> 只读输入一次、写输出一次是理想流量目标
```

不要只背“fusion 快”。应能回答三件事：融合省了哪些中间 HBM transfers；reduction 所需的数据是否能同时放在片上；融合后是否产生新的 register/occupancy 边界。

## 12. Row sum：一整行放不进一个 tile 时怎么办

### 12.1 问题不是“行不属于一个 program”，而是“一次装不下整行”

**【课程代码｜行 487–505｜视频 [65:30](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3930s)】**§11 的 softmax 让一个 program 一次构造覆盖整行的向量。如果一行有 4,096 列，而当前 tile 只有 1,024 个位置，一次 load 装不下整行。

课程策略：

```text
一个 Triton program 仍负责一整行
        ↓
把这一行沿列切成多个 tiles
        ↓
同一组向量位置循环访问 tile 0、tile 1、tile 2……
        ↓
每个位置维护自己的 accumulator
        ↓
最后 tl.sum 把 accumulator 向量归约成一个标量
```

**accumulator（累加器）**是“不断加上新值的临时和”。初始为 0，每读一块就更新一次。

课程口头把向量位置叫 thread，便于建立硬件直觉；严格的 Triton 心智模型仍是：源码描述长度为 `BLOCK_SIZE` 的向量 accumulator，compiler 再决定 threads/warps/registers/shared memory 的实际映射。不能机械写成“一项 accumulator 永远对应一个 CUDA thread”。

### 12.2 Wrapper 的 shape 与 pointer 假设

**【课程代码｜行 506–518｜视频 [66:27](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3987s)】**

```python
def triton_row_sum(
    x: torch.Tensor,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](
        x,
        y,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

Shape 逐项：

- 输入 `x`：`[M,N]`，$`M`$ 行、每行 $`N`$ 个数；
- 输出 `y`：`[M]`，每行只留下一个和；
- grid：`(M,)`，每行一个 program；
- `BLOCK_SIZE=B`：一次 tile 有 $`B`$ 个语义向量位置。

课程 kernel 用：

```python
x_ptr + row * N + cols
```

它假设 `x` 按连续 row-major 顺序存放，即 row stride 正好是 $`N`$。课程 wrapper 没有显式 `assert x.is_contiguous()`，也没传 `x.stride(0)`；因此这是教学简化。要支持任意 strided tensor，应传 row stride 并写成 `row*x_row_stride`。

### 12.3 Kernel 的循环逐行翻译

**【课程代码｜行 517–535｜视频 [68:32](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4112s)–[70:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4205s)】**

```python
@triton.jit
def row_sum_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(
            x_ptr + row * N + cols,
            mask=mask,
            other=0.0,
        )
        acc += x

    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)
```

逐行状态：

1. `row=program_id(0)`：选择当前行。
2. `tl.zeros([B],float32)`：建立长度 $`B`$ 的零向量。即使输入 dtype 较低，FP32 accumulator 通常能减少累加舍入误差。
3. `range(0,N,B)`：`start` 依次为 $`0,B,2B,\ldots`$，直到最后一个小于 $`N`$ 的起点。
4. `cols=start+[0,1,...,B-1]`：生成当前 tile 的列号。
5. `mask=cols<N`：最后一个不足 $`B`$ 项的 tile 禁止越界。
6. `other=0.0`：无效位置补 0；0 是求和的 identity（单位元），因为 $`a+0=a`$。
7. `acc+=x`：按向量位置逐项累加当前 tile。
8. 循环结束后，`acc` 仍是长度 $`B`$ 的向量，不是最终 row sum。
9. `tl.sum(acc,axis=0)`：compiler 生成所需的 reduction，把 $`B`$ 项合成一个 scalar（标量）。可能使用 warp shuffle/shared memory 等低层机制，源码没有固定指定。
10. `store(out_ptr+row,result)`：每行只写一个输出。

### 12.4 $`N=12,B=4`$：把 `[1..12]` 全部走一遍

**【补充】**输入只有一行：

```math
x=[1,2,3,4,5,6,7,8,9,10,11,12].
```

Shape 是 `[1,12]`，输出 shape 是 `[1]`。初始：

```math
acc^{(0)}=[0,0,0,0].
```

Tile 0，`start=0`：

```text
cols = [0,1,2,3]
mask = [T,T,T,T]
load = [1,2,3,4]
```

逐项相加：

```math
acc^{(1)}
=[0+1,0+2,0+3,0+4]
=[1,2,3,4].
```

Tile 1，`start=4`：

```text
cols = [4,5,6,7]
load = [5,6,7,8]
```

```math
acc^{(2)}
=[1+5,2+6,3+7,4+8]
=[6,8,10,12].
```

Tile 2，`start=8`：

```text
cols = [8,9,10,11]
load = [9,10,11,12]
```

```math
acc^{(3)}
=[6+9,8+10,10+11,12+12]
=[15,18,21,24].
```

最后 reduction：

```math
result=15+18+21+24.
```

逐步：

```math
15+18=33,
```

```math
21+24=45,
```

```math
33+45=78.
```

直接检查：

```math
1+2+\cdots+12
=\frac{12\times13}{2}
=78.
```

两条路径一致。

### 12.5 $`N=10,B=4`$：最后两项为什么必须补 0

**【补充】**现在一行是 `[1,2,...,10]`。前两个 tiles 后：

```math
acc=[6,8,10,12].
```

最后 `start=8`：

| 向量位置 | `col` | `col<10` | 真内存元素 | `tl.load(...,other=0)` |
|---:|---:|---:|---:|---:|
| 0 | 8 | True | $`x[8]=9`$ | 9 |
| 1 | 9 | True | $`x[9]=10`$ | 10 |
| 2 | 10 | False | 越界 | 0 |
| 3 | 11 | False | 越界 | 0 |

更新：

```math
acc=[6+9,8+10,10+0,12+0]
=[15,18,10,12].
```

最后：

```math
15+18+10+12=55.
```

直接检查：

```math
1+2+\cdots+10
=\frac{10\times11}{2}
=55.
```

若 `other` 错写成 1，结果会多 $`1+1=2`$；若没有 mask，会读 `x[10]`、`x[11]` 的非法地址。求和 padding 必须用 0，而 softmax padding 用 $`-\infty`$，因为两个 reduction 的单位元不同。

### 12.6 Tiles 不是多个独立 blocks

**【视频补充｜[71:06](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4266s)】**老师专门区分：GeLU 可把不同块交给互不相关的 programs；这里同一行的 tiles 必须共同得到一个 row sum，所以它们由同一个 program 循环处理。

```text
错误：tile 0、1、2 各是一个独立 program，最后自然就有总和

正确：同一个 row program 依次访问 tile 0、1、2，保留acc；
      循环后再对acc做一次 program内 reduction。
```

**【视频补充｜[70:30](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4230s)】**accumulator 最终放 registers 还是 shared memory，不由这段 Python/Triton 源码逐项指定；compiler 根据大小和目标硬件决定。若资源不够，也可能发生 spill。不要把“课程图画在 register”当永久保证。

## 13. Matrix multiplication：从每个输出单算到 tiling 复用

### 13.1 Shape 先对齐：$`A[M,K]B[K,N]=C[M,N]`$

**【课程代码｜行 538–568｜视频 [71:57](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4317s)–[73:01](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4381s)】**矩阵乘：

```math
A\in\mathbb{R}^{M\times K},
\quad
B\in\mathbb{R}^{K\times N},
\quad
C=AB\in\mathbb{R}^{M\times N}.
```

- $`M`$：$`A`$ 和 $`C`$ 的行数；
- $`K`$：被消去的 inner dimension（内维），必须同时是 $`A`$ 的列数和 $`B`$ 的行数；
- $`N`$：$`B`$ 和 $`C`$ 的列数；
- $`C[m,n]`$：输出第 $`m`$ 行、第 $`n`$ 列。

单个输出：

```math
C[m,n]=\sum_{k=0}^{K-1}A[m,k]B[k,n].
```

它要做 $`K`$ 次乘法，并把 $`K`$ 个乘积相加。把一次 multiply 与一次 accumulate 近似记 2 FLOPs，则全部输出约 $`2MKN`$ FLOPs。

### 13.2 Naive reads 的两个口径必须说清

**【课程内容｜视频 [73:44](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4424s)–[74:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4450s)】**课程写“$`MKN`$ reads”，是在 big-O 讨论中把“为一个 $`k`$ 取得一对 $`A/B`$ 操作数”当一次读事件，省略常数 2。

若严格按“读一个标量元素一次”记：每个 $`(m,n,k)`$ 要读：

```text
A[m,k]：1 个元素
B[k,n]：1 个元素
```

所以 input reads 是：

```math
2MKN\ \text{scalar-element reads},
```

最后另写：

```math
MN\ \text{output elements}.
```

两种说法的大 O 都是 $`O(MKN)`$，但做精确 bytes 账时必须保留 2。下面所有精确流量都按标量元素口径。

### 13.3 为什么 naive 重复读同一个元素

**【课程内容｜视频 [74:43](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4483s)】**固定一个 $`A[m,k]`$：它参与同一输出行的所有 $`N`$ 列：

```math
C[m,0],C[m,1],\ldots,C[m,N-1].
```

若每个 $`C[m,n]`$ 独立从 HBM 取操作数，这个 $`A[m,k]`$ 会读约 $`N`$ 次。

固定一个 $`B[k,n]`$：它参与同一输出列的所有 $`M`$ 行，所以会读约 $`M`$ 次。

方阵 $`M=N=K`$ 时，每个 $`A/B`$ 输入元素都大约重复读 $`N`$ 次。这就是 tiling 要消除的复用浪费。

### 13.4 $`4\times4,T=2`$：具体矩阵

**【补充】**取：

```math
A=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
13&14&15&16
\end{bmatrix},
```

```math
B=
\begin{bmatrix}
1&2&0&1\\
0&1&1&0\\
2&0&1&1\\
1&1&0&2
\end{bmatrix}.
```

$`M=N=K=4`$，输出 tile 大小 $`T=2`$。先只算 $`C`$ 左上 $`2\times2`$：rows 0–1、columns 0–1。

它沿 $`K`$ 分两段：

```text
K-tile 0：k=0,1
K-tile 1：k=2,3
```

### 13.5 左上 tile：partial accumulator 0

第一个 $`K`$ tile 读：

```math
A_0=
\begin{bmatrix}
1&2\\
5&6
\end{bmatrix},
\quad
B_0=
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}.
```

初始 accumulator：

```math
acc_{\text{init}}=
\begin{bmatrix}
0&0\\
0&0
\end{bmatrix}.
```

算 $`A_0B_0`$ 四项：

```math
(0,0):1\times1+2\times0=1,
```

```math
(0,1):1\times2+2\times1=2+2=4,
```

```math
(1,0):5\times1+6\times0=5,
```

```math
(1,1):5\times2+6\times1=10+6=16.
```

所以：

```math
acc_0=
\begin{bmatrix}
1&4\\
5&16
\end{bmatrix}.
```

### 13.6 左上 tile：partial accumulator 1 与最终值

第二个 $`K`$ tile：

```math
A_1=
\begin{bmatrix}
3&4\\
7&8
\end{bmatrix},
\quad
B_1=
\begin{bmatrix}
2&0\\
1&1
\end{bmatrix}.
```

这次贡献：

```math
(0,0):3\times2+4\times1=6+4=10,
```

```math
(0,1):3\times0+4\times1=4,
```

```math
(1,0):7\times2+8\times1=14+8=22,
```

```math
(1,1):7\times0+8\times1=8.
```

贡献矩阵：

```math
P_1=
\begin{bmatrix}
10&4\\
22&8
\end{bmatrix}.
```

累加：

```math
acc_1=acc_0+P_1
=
\begin{bmatrix}
1+10&4+4\\
5+22&16+8
\end{bmatrix}
=
\begin{bmatrix}
11&8\\
27&24
\end{bmatrix}.
```

这就是最终 $`C`$ 左上 tile。

### 13.7 完整 $`C`$ 与一项交叉检查

对其他三个输出 tiles 做同样的两段 $`K`$ 累加，得到：

```math
C=AB=
\begin{bmatrix}
11&8&5&12\\
27&24&13&28\\
43&40&21&44\\
59&56&29&60
\end{bmatrix}.
```

不能只信表，抽查右下角：

```math
C[3,3]
=13\times1+14\times0+15\times1+16\times2.
```

```math
=13+0+15+32=60.
```

与矩阵右下角一致。

### 13.8 一个 tile 内到底复用了几次

对左上输出 tile 的第一个 $`K`$ tile：

- $`A[0,0]=1`$ 同时用于 $`C[0,0]`$ 和 $`C[0,1]`$，复用 $`B_N=2`$ 次；
- $`A[0,1]=2`$ 也用于这两个输出，复用 2 次；
- $`B[0,0]=1`$ 同时用于 $`C[0,0]`$ 和 $`C[1,0]`$，复用 $`B_M=2`$ 次；
- $`B[0,1]=2`$ 也跨两行输出复用 2 次。

一般输出 tile 是 $`B_M\times B_N`$：

- 每个载入的 $`A`$ 元素横向供 $`B_N`$ 个输出使用；
- 每个载入的 $`B`$ 元素纵向供 $`B_M`$ 个输出使用。

这就是“从 HBM 读一次，在片上用多次”。

### 13.9 一般 $`B_M,B_N,B_K`$ 的 input reads 推导

**【课程内容 + 补充精确化｜视频 [76:19](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4579s)–[78:22](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4702s)】**先假设 $`M,N,K`$ 都能分别整除 $`B_M,B_N,B_K`$，不考虑边界 padding。

输出 tiles 数：

```math
\frac{M}{B_M}\times\frac{N}{B_N}.
```

每个输出 tile 要走的 $`K`$ tiles 数：

```math
\frac{K}{B_K}.
```

每次 $`K`$ tile 读取：

```math
B_MB_K\ \text{个 A 元素}
```

和：

```math
B_KB_N\ \text{个 B 元素}.
```

总 input reads：

```math
\frac{M}{B_M}
\frac{N}{B_N}
\frac{K}{B_K}
(B_MB_K+B_KB_N).
```

分别约掉：

```math
=\frac{MNK}{B_N}+\frac{MNK}{B_M}.
```

合写：

```math
=MNK\left(\frac1{B_N}+\frac1{B_M}\right).
```

为什么 $`B_K`$ 在这个理想计数里约掉？$`B_K`$ 加倍时，每次 tile 读量加倍，但 $`K`$ 循环次数减半。$`B_K`$ 仍影响 Tensor Core shape、shared memory、registers、pipeline 和实际速度；“在此公式里约掉”不等于“它不重要”。

若不能整除，调度的候选 load slots 为：

```math
\left\lceil\frac{M}{B_M}\right\rceil
\left\lceil\frac{N}{B_N}\right\rceil
\left\lceil\frac{K}{B_K}\right\rceil
(B_MB_K+B_KB_N),
```

边界 mask 会让一部分 slots 不真正访问 global memory。精确有效元素读数依赖各边界 tile 的剩余尺寸。

### 13.10 方阵、方形输出 tile：为什么是 $`2N^3/T`$

令：

```math
M=N=K,
\quad B_M=B_N=T.
```

代入：

```math
N^3\left(\frac1T+\frac1T\right)
=\frac{2N^3}{T}
```

个 input elements。

另有输出写：

```math
N^2\ \text{elements}.
```

不能把 output writes 藏进 input reads；它们在精确总流量比里会让改善略小于 $`T`$。

### 13.11 用户关心的完整例：$`N=1024,T=32`$

**【补充】**方阵 $`A,B,C`$ 都是 $`1024\times1024`$，FP32 每元素 4 bytes。

先定义本节第一次使用的二进制容量单位：

```math
1\ \text{MiB}=2^{20}=1{,}048{,}576\ \text{bytes},
```

```math
1\ \text{GiB}=1{,}024\ \text{MiB}
=2^{30}=1{,}073{,}741{,}824\ \text{bytes}.
```

`MiB/GiB` 是二进制单位；这里不要把它们与十进制厂商容量标签 `MB/GB` 混算。

先算元素数：

```math
N^2=1024^2=1{,}048{,}576.
```

```math
N^3=1024^3=1{,}073{,}741{,}824.
```

#### Naive input reads

每个 $`A`$ 元素要服务 1,024 个输出列，约读 1,024 次；每个 $`B`$ 元素要服务 1,024 个输出行，也约读 1,024 次。

```math
2N^3
=2\times1{,}073{,}741{,}824
=2{,}147{,}483{,}648
```

个 input elements。

换 bytes：

```math
2{,}147{,}483{,}648\times4
=8{,}589{,}934{,}592\ \text{bytes}.
```

```math
8{,}589{,}934{,}592/1{,}073{,}741{,}824
=8\ \text{GiB}.
```

#### Tiled input reads

输出每边有：

```math
N/T=1024/32=32
```

个 tiles。一个 $`A`$ 元素只需针对 32 个输出-column tiles 各载入一次，所以约读 32 次；一个 $`B`$ 元素也针对 32 个输出-row tiles 各载入一次。每次进 tile 后，它分别被 32 个输出复用。

公式：

```math
\frac{2N^3}{T}
=\frac{2\times1{,}073{,}741{,}824}{32}
=67{,}108{,}864
```

个 input elements。

换 bytes：

```math
67{,}108{,}864\times4
=268{,}435{,}456\ \text{bytes}.
```

再除以每 MiB 的 bytes：

```math
268{,}435{,}456/1{,}048{,}576
=256\ \text{MiB}.
```

#### Output writes

两种方法都至少写 $`N^2`$ 个输出：

```math
1{,}048{,}576\times4
=4{,}194{,}304\ \text{bytes}.
```

```math
4{,}194{,}304/1{,}048{,}576
=4\ \text{MiB}.
```

#### 总流量与真实比值

先把 naive 的 8 GiB 换成 MiB：

```math
8\ \text{GiB}
=8\times1{,}024\ \text{MiB}
=8{,}192\ \text{MiB}.
```

Naive 教学总量：

```math
8{,}192+4=8{,}196\ \text{MiB}.
```

换回 GiB：

```math
8{,}196/1{,}024
=8.00390625\ \text{GiB}
\approx8.004\ \text{GiB}.
```

Tiled 教学总量：

```math
256\ \text{MiB}+4\ \text{MiB}
=260\ \text{MiB}.
```

两者都用 MiB 后求比值：

```math
\frac{8192+4}{256+4}
=\frac{8196}{260}
\approx31.523.
```

约 $`31.52\times`$，不是包含 output 后精确 $`32\times`$。若只比 input reads：

```math
8\ \text{GiB}/256\ \text{MiB}=32.
```

### 13.12 连接 arithmetic intensity 与 Roofline

**【补充理解】**矩阵乘 FLOPs 约：

```math
F\approx2N^3.
```

Naive FP32 input bytes 约 $`8N^3`$，忽略较小 output 后：

```math
I_{naive}\approx\frac{2N^3}{8N^3}=0.25\ \text{FLOP/byte}.
```

Tiled input bytes 约：

```math
4\times\frac{2N^3}{T}=\frac{8N^3}{T}.
```

所以大 $`N`$ 时：

```math
I_{tiled}\approx
\frac{2N^3}{8N^3/T}
=\frac{T}{4}\ \text{FLOP/byte}.
```

$`T=32`$ 时约 $`8`$ FLOP/byte。Roofline 会用 arithmetic intensity 判断 bandwidth roof 与 compute roof 哪个更低；tiling 把点向右推，但是否跨过 ridge point 仍取决于具体 GPU 的带宽、峰值计算、dtype 和实际 kernel 效率。这里不重复 Lecture 5 全章，也不把 $`T/4`$ 当实测值：cache、额外读写、边界、pipeline 和 fusion 都会改变它。

## 14. Triton matmul wrapper 与 kernel：每个索引都追踪 shape

### 14.1 Wrapper 先拒绝不合法 shape

**【课程代码｜行 607–632｜视频 [79:03](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4743s)】**

```python
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device)
    ...
```

逐行：

1. 两个输入必须在 CUDA device；CPU pointer 不能交给 GPU kernel。
2. 课程实现只处理 contiguous 输入，因为后面 strides 和 pointer 访问按这种条件测试。
3. `a.shape[1]==b.shape[0]` 检查两个 inner dimensions 相等。
4. `M,K=a.shape` 先得到 $`A`$ 的 $`M,K`$。
5. `K,N=b.shape` 又把 Python 变量 `K` 覆盖为 $`B`$ 的行数。前一行 `assert` 已保证新旧两个 $`K`$ 相同，所以结果正确；教学或生产代码写成 `K_a`、`K_b` 再 assert 会更不易误读。
6. 输出 shape 是 `[M,N]`。

课程 `torch.empty((M,N),device=a.device)` 没显式传 `dtype=a.dtype`。在默认设置下它通常创建默认浮点 dtype，未必适配所有输入 dtype；这是教学简化。通用 wrapper 应明确输出 dtype，并检查 `a.dtype/b.dtype/device` 是否兼容。

### 14.2 Stride：二维下标怎样变成一维元素偏移

**【课程代码｜行 591–597｜视频 [79:09](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4749s)】**对 tensor：

```python
x = torch.tensor([
    [0., 1, 2, 3],
    [4,  5, 6, 7],
])
```

Shape `[2,4]`，连续 row-major stride 是：

```text
stride_row = 4
stride_col = 1
```

元素 `(row=1,col=2)` 的元素偏移：

```math
1\times4+2\times1=6.
```

扁平 storage 的第 6 项是 6，检查一致。公式：

```math
\mathrm{offset}(r,c)
=r\cdot stride_{row}+c\cdot stride_{col}.
```

Stride 以“元素”为单位，不是固定以 bytes 为单位。

### 14.3 Grid：每个 program 负责一个 $`C`$ tile

**【课程代码｜行 619–630｜视频 [79:54](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4794s)】**

```python
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

grid = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N),
)
```

- grid axis 0 枚举输出 row tiles；
- grid axis 1 枚举输出 column tiles；
- $`B_K`$ 不形成独立 grid 维，而在同一 program 内通过 loop 累加。

若 $`M=130,N=70,B_M=B_N=64`$：

```math
\lceil130/64\rceil=3,
\quad
\lceil70/64\rceil=2,
```

所以 grid shape 是 `[3,2]`，一共 $`3\times2=6`$ 个 program instances。

### 14.4 `pid_m,pid_n` 与三个一维 index vectors

**【课程代码｜行 635–653｜视频 [79:57](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4797s)–[80:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4834s)】**

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)

indices_m = (
    pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
)  # shape [BLOCK_M]

indices_n = (
    pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
)  # shape [BLOCK_N]

indices_k = tl.arange(0, BLOCK_K)  # shape [BLOCK_K]
```

例：$`B_M=2,B_N=3,B_K=2`$，program `(pid_m=1,pid_n=2)`：

```math
indices_m=1\times2+[0,1]=[2,3],
```

```math
indices_n=2\times3+[0,1,2]=[6,7,8],
```

```math
indices_k=[0,1].
```

这里 `indices_k` 是当前 $`K`$ tile 内的相对索引；loop 变量 `k` 和 pointer advance 会把它移到后续 $`K`$ tiles。

### 14.5 `None` 是新增长度为 1 的 axis

**【补充解释】**若：

```text
indices_m.shape = [BM]
indices_k.shape = [BK]
```

则：

```text
indices_m[:, None].shape = [BM, 1]
indices_k[None, :].shape = [1, BK]
```

加法时 broadcasting：

```text
[BM,1] + [1,BK] -> [BM,BK]
```

小例 $`indices_m=[2,3]`$、$`indices_k=[0,1]`$：

```math
indices_m[:,None]
=\begin{bmatrix}2\\3\end{bmatrix},
```

```math
indices_k[None,:]
=\begin{bmatrix}0&1\end{bmatrix}.
```

把 row index 乘 row stride、column index 乘 column stride，再广播相加，就得到一个 pointer matrix。

### 14.6 $`A/B`$ pointer matrices 的 shape

**【课程代码｜行 655–657｜视频 [80:38](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4838s)】**

```python
a_ptrs = (
    a_ptr
    + indices_m[:, None] * stride_am
    + indices_k[None, :] * stride_ak
)  # shape [BLOCK_M, BLOCK_K]

b_ptrs = (
    b_ptr
    + indices_k[:, None] * stride_bk
    + indices_n[None, :] * stride_bn
)  # shape [BLOCK_K, BLOCK_N]
```

- `a_ptrs[i,q]` 指向 $`A[indices_m[i],indices_k[q]]`$；
- `b_ptrs[q,j]` 指向 $`B[indices_k[q],indices_n[j]]`$；
- 一个是 `[B_M,B_K]`，另一个是 `[B_K,B_N]`，正好可以矩阵乘出 `[B_M,B_N]`。

`a_ptrs` 或 `b_ptrs` 不是已经 load 的数字；它们是地址组成的矩阵。真正读 memory 发生在 `tl.load`。

### 14.7 $`K`$ loop：mask、`tl.dot` 与 pointer advance

**【课程代码｜行 659–667｜视频 [80:46](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4846s)–[81:29](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4889s)】**

```python
acc = tl.zeros(
    [BLOCK_M, BLOCK_N],
    dtype=tl.float32,
)

for k in range(0, K, BLOCK_K):
    a = tl.load(
        a_ptrs,
        mask=(indices_m[:, None] < M)
             & (indices_k[None, :] + k < K),
        other=0.0,
    )
    b = tl.load(
        b_ptrs,
        mask=(indices_k[:, None] + k < K)
             & (indices_n[None, :] < N),
        other=0.0,
    )
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

逐步：

1. `acc` shape `[B_M,B_N]`，FP32 累加降低低精度输入的累计误差；
2. `&` 是逐元素 logical AND（逻辑与），左右条件都 True 才 load；不能换成 Python scalar `and`；
3. $`A`$ mask 同时检查输出 row 是否小于 $`M`$、有效 $`K`$ 是否小于总 $`K`$；
4. $`B`$ mask 同时检查有效 $`K`$、输出 column 是否小于 $`N`$；
5. 越界项填 0，因为矩阵乘中 $`0\times z=0`$，不会改变 accumulator；
6. `tl.dot(a,b)` 语义是 `[B_M,B_K]@[B_K,B_N]→[B_M,B_N]`；compiler 依 dtype、shape 和硬件选择可用的矩阵乘实现，不能仅凭这一行保证使用某种 Tensor Core 指令；
7. `a_ptrs += B_K*stride_ak` 把 $`A`$ 指针向右移动 $`B_K`$ 列；
8. `b_ptrs += B_K*stride_bk` 把 $`B`$ 指针向下移动 $`B_K`$ 行；
9. `indices_k` 本身仍是 `[0,...,B_K-1]`，所以 mask 用 `indices_k+k` 检查当前全局 $`K`$。

### 14.8 Store pointer 与二维边界 mask

**【课程代码｜行 669–674｜视频 [81:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4894s)】**

```python
c_ptrs = (
    c_ptr
    + indices_m[:, None] * stride_cm
    + indices_n[None, :] * stride_cn
)

tl.store(
    c_ptrs,
    acc,
    mask=(indices_m[:, None] < M)
         & (indices_n[None, :] < N),
)
```

`c_ptrs` shape `[B_M,B_N]`。边界 tile 的候选 row/column 可能超过真正 $`M,N`$，所以 store 也必须 mask；load mask 正确不代表 store 自动安全。

### 14.9 非整除完整例：$`M=3,N=5,K=3`$，blocks 都是 2

**【补充】**设：

```math
B_M=B_N=B_K=2.
```

Grid：

```math
\lceil3/2\rceil\times\lceil5/2\rceil
=2\times3.
```

六个 programs 的候选输出：

| `(pid_m,pid_n)` | candidate rows | candidate cols | 越界部分 |
|---|---|---|---|
| `(0,0)` | `[0,1]` | `[0,1]` | 无 |
| `(0,1)` | `[0,1]` | `[2,3]` | 无 |
| `(0,2)` | `[0,1]` | `[4,5]` | col 5 |
| `(1,0)` | `[2,3]` | `[0,1]` | row 3 |
| `(1,1)` | `[2,3]` | `[2,3]` | row 3 |
| `(1,2)` | `[2,3]` | `[4,5]` | row 3、col 5 |

最难的 program `(1,2)`：

```math
indices_m=[2,3],
\quad indices_n=[4,5],
\quad indices_k=[0,1].
```

假设连续 row-major：

```text
A strides = [3,1]
B strides = [5,1]
C strides = [5,1]
```

#### 第一个 $`K`$ tile：`k=0`

$`A`$ 候选元素 offsets：

```math
\begin{bmatrix}
2\times3+0&2\times3+1\\
3\times3+0&3\times3+1
\end{bmatrix}
=
\begin{bmatrix}
6&7\\
9&10
\end{bmatrix}.
```

$`A`$ mask：row 2 有效、row 3 越界；$`k=0,1`$ 都有效：

```math
mask_A^{(0)}=
\begin{bmatrix}
T&T\\
F&F
\end{bmatrix}.
```

$`B`$ 候选 offsets：

```math
\begin{bmatrix}
0\times5+4&0\times5+5\\
1\times5+4&1\times5+5
\end{bmatrix}
=
\begin{bmatrix}
4&5\\
9&10
\end{bmatrix}.
```

两个 $`k`$ 都有效，但 col 5 越界：

```math
mask_B^{(0)}=
\begin{bmatrix}
T&F\\
T&F
\end{bmatrix}.
```

#### 第二个 $`K`$ tile：`k=2`

Pointers 分别前进：

```math
B_K\cdot stride_{ak}=2\times1=2,
```

```math
B_K\cdot stride_{bk}=2\times5=10.
```

有效全局 $`K`$ 候选为 `[2,3]`，而 $`K=3`$，所以只有 2 有效。

```math
mask_A^{(1)}=
\begin{bmatrix}
T&F\\
F&F
\end{bmatrix},
```

```math
mask_B^{(1)}=
\begin{bmatrix}
T&F\\
F&F
\end{bmatrix}.
```

#### Store

Candidate $`C`$ offsets：

```math
\begin{bmatrix}
2\times5+4&2\times5+5\\
3\times5+4&3\times5+5
\end{bmatrix}
=
\begin{bmatrix}
14&15\\
19&20
\end{bmatrix}.
```

Store mask：

```math
\begin{bmatrix}
T&F\\
F&F
\end{bmatrix}.
```

因此这个 program 最终只写合法的 $`C[2,4]`$。其他三个 candidate positions 只是 tile padding，不能写 memory。

### 14.10 读 kernel 的 shape 检查表

| 名称 | Shape | 角色 |
|---|---|---|
| `indices_m` | `[B_M]` | 输出 rows / A rows |
| `indices_n` | `[B_N]` | 输出 cols / B cols |
| `indices_k` | `[B_K]` | 当前 inner tile 相对索引 |
| `a_ptrs`, loaded `a` | `[B_M,B_K]` | A tile |
| `b_ptrs`, loaded `b` | `[B_K,B_N]` | B tile |
| `tl.dot(a,b)` | `[B_M,B_N]` | 当前 K tile 的 partial product |
| `acc` | `[B_M,B_N]` | 跨所有 K tiles 的累计输出 |
| `c_ptrs` | `[B_M,B_N]` | 最终 output tile 地址 |

只要内维 `[B_K]` 对齐，`tl.dot` 的输出 shape 就能检查；只要 `acc` 与 `c_ptrs` shape 相同，store 才逐项对应。

## 15. Matmul + ReLU fusion：中间矩阵不必落回 HBM

### 15.1 ReLU 是逐元素 `max(z,0)`

**【课程代码｜行 602–604、669–674｜视频 [72:19](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4339s)】**ReLU 是 Rectified Linear Unit（修正线性单元）：

```math
\mathrm{ReLU}(z)=\max(z,0).
```

- $`z>0`$：保留 $`z`$；
- $`z=0`$：仍为 0；
- $`z<0`$：变为 0；
- 逐元素操作，shape 不变。

Naive 高层代码：

```python
def naive_matmul_relu(x, y):
    return torch.nn.functional.relu(x @ y)
```

概念上先产生完整中间矩阵 $`Z=X@Y`$，再算 $`C=\mathrm{ReLU}(Z)`$。

### 15.2 含负数的 $`2\times2`$ 完整例

**【补充】**取：

```math
A=
\begin{bmatrix}
1&-2\\
-3&1
\end{bmatrix},
\quad
B=
\begin{bmatrix}
1&2\\
3&-1
\end{bmatrix}.
```

先算 matmul：

```math
Z[0,0]=1\times1+(-2)\times3=1-6=-5,
```

```math
Z[0,1]=1\times2+(-2)\times(-1)=2+2=4,
```

```math
Z[1,0]=(-3)\times1+1\times3=-3+3=0,
```

```math
Z[1,1]=(-3)\times2+1\times(-1)=-6-1=-7.
```

所以：

```math
Z=AB=
\begin{bmatrix}
-5&4\\
0&-7
\end{bmatrix}.
```

逐项 ReLU：

```math
C=
\begin{bmatrix}
\max(-5,0)&\max(4,0)\\
\max(0,0)&\max(-7,0)
\end{bmatrix}
=
\begin{bmatrix}
0&4\\
0&0
\end{bmatrix}.
```

### 15.3 Fused kernel 在哪里做 ReLU

**【课程代码｜行 659–674｜视频 [78:39](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4719s)–[81:47](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4907s)】**所有 $`K`$ tiles 累加后，`acc` 已是最终 matmul output tile，但还在 program 的片上计算状态中：

```python
for k in range(0, K, BLOCK_K):
    ...
    acc += tl.dot(a, b)

# 在写HBM之前逐项做ReLU
acc = tl.maximum(acc, 0.0)

# 只写ReLU后的最终结果
tl.store(c_ptrs, acc, mask=output_mask)
```

对上一小例，语义状态：

```text
acc before ReLU = [[-5, 4], [0, -7]]
acc after ReLU  = [[ 0, 4], [0,  0]]
store once
```

### 15.4 Output-side 流量和 launch 账

**【补充推导】**这里只比较 matmul input tiles 已经读完之后的 output-side logical transfers；$`A/B`$ 的读取两种路径相同，不重复计。

Separate kernels：

```text
matmul kernel：写中间 Z       MN writes
ReLU kernel ：读中间 Z       MN reads
ReLU kernel ：写最终 C       MN writes
合计                         3MN transfers
launch 数                    2
```

Fused kernel：

```text
片上acc做ReLU：              0 HBM transfer
写最终 C：                  MN writes
合计                         MN transfers
launch 数                    1
```

逻辑节省：

```math
3MN-MN=2MN
```

个 element-transfers，也就是省掉中间 $`Z`$ 的一次写与一次读。

对 $`2\times2`$ FP32：

```math
2MN=2\times2\times2=8
```

个元素传输：

```math
8\times4=32\ \text{bytes}.
```

**Epilogue（尾处理）**是 matmul 完成主要乘加、写回 output 前执行的 bias、activation、类型转换等末尾操作。这是教学 logical traffic；真实 HBM bytes 受 cache、已有 library epilogue、compiler fusion 和 memory transaction 粒度影响。若高层编译器本来就把 activation 融入 matmul epilogue，naive 源码也不一定真的启动两个 kernels；必须 profile。

### 15.5 Fusion 不是越大越好

Fusion 可能增加：

- **register pressure**：更多中间值同时存活；
- 单个 kernel 的代码量和 compilation time；
- program 资源占用，进而降低 occupancy；
- 数值/调试复杂度。

这些情况不能随便融合：

- 中间 $`Z`$ 还被另一个 **consumer（下游使用者，即另一个需要读取 $`Z`$ 的操作）**使用；若不 store，就得重新计算；
- 两步之间需要跨 programs 的 global synchronization；
- 融合后 tile 太大、register spill，实际反而慢；
- vendor/library matmul kernel 极强，自写 fused kernel 的 matmul 主体损失超过 epilogue 收益；
- 改变计算顺序会破坏所需数值语义。

结论不是“融合必快”，而是“它提供了少一次 launch、少一次中间写读的机会；最终重新 benchmark/profile”。

## 16. 从 correctness 到 profile 的完整优化闭环

### 16.1 不可跳步的顺序

**【课程总结｜视频 [82:12](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4932s)–[83:13](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4993s)】**

```text
1. 写清数学结果、shape、dtype、device
2. 用可信 reference 检查 correctness
3. warmup + synchronize 后 benchmark
4. profile 看 kernel、launch、memory、occupancy
5. 一次改变一个假设
6. 再做 correctness check
7. 在相同输入条件重新 benchmark/profile
8. 记录硬件、软件版本与输入分布
```

**correctness（正确性）**不只指“代码没有报错”，还包括：

- 输出 shape 正确；
- 边界 mask 没漏数据或越界；
- 数值在合理 tolerance 内与 reference 一致；
- dtype/device 正确；
- 特殊输入如负数、极大 logits、非整除尺寸也正确。

优化后若快了但答案错了，不叫优化成功。

### 16.2 看到慢时逐项问什么

| 观察维度 | 先问的问题 | 可能尝试 | 不能直接断言 |
|---|---|---|---|
| launch | 是否大量极小 kernels？ | 合理 fusion、batch 更多工作 | kernel 少就一定快 |
| HBM | 是否反复物化中间 tensor？地址是否 coalesced？ | fusion、tiling、改变 layout | logical reads 等于实际 HBM transactions |
| shared memory | 地址是否落到同 bank 不同位置？ | padding、swizzle、改 layout | 所有同 bank 访问都 conflict |
| registers | 每 thread/program 临时量是否过多？ | 调小 tile、减少 live values | register 越少永远越快 |
| occupancy | resident warps 是否被 register/shared 限制？ | 调 block/warps/tile | occupancy 低必然慢 |
| waves | blocks 数是否让最后一波只占少数 SM？ | 调 grid/tile、batch | blocks 越多越好 |
| compute | `tl.dot`/dtype/shape 是否适合硬件矩阵单元？ | 调 $`B_M,B_N,B_K`$、dtype | 某个固定 tile 适合所有 GPU |
| numerical | fusion/重排是否改变舍入或稳定性？ | FP32 acc、减 max、tolerance | bit 不同就是错误，或差很多也没关系 |

### 16.3 为什么没有万能最佳 tile size

增大 tile 常常提高数据复用：

```math
\text{square matmul teaching intensity}\approx T/4.
```

但同时会增加：

- tile 内 accumulator 元素数；
- registers/shared-memory 使用；
- padding 浪费；
- reduction 或同步负担；
- 单个 block 运行时间；
- block 数减少后产生 wave quantization tail 的风险。

因此 $`T=16,32,64,128`$ 没有脱离上下文的永久冠军。最佳选择依赖 $`M,N,K`$、dtype、GPU、compiler、并发 workload 和是否融合 epilogue。现实实现常用 autotuning（自动调参）：在合法候选配置中实测，再选该 shape/设备上更快者。

### 16.4 三种例子形成的难度阶梯

**【课程总结｜视频 [83:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4996s)–[84:08](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5048s)】**

```text
GeLU
  每元素独立
  -> 一块向量load/compute/store

Softmax / row sum
  行内要reduction
  -> 一program看整行；装不下就循环tiles

Matmul
  输出tile既要横向复用A，又要纵向复用B
  -> 二维grid + K-tile loop + matrix accumulator
  -> 写回前可融合ReLU
```

难度提高的真正原因是“哪些数据必须在片上共同存在并通信”，不只是代码行数增加。

### 16.5 课程代码最后 68 行覆盖

**【课程代码｜行 677–744】**这些 helper 没有新的 kernel 算法，但决定演示如何构造输入、验证和输出 PTX，不能在覆盖表中略过：

| 行段 | Helper | 人话解释 | 本笔记位置 |
|---:|---|---|---|
| 679–683 | `run_operation1` | 建一个随机 `[dim,dim]` tensor，返回无参数 closure 供 benchmark 调用 | §6 benchmark、§7 GeLU |
| 686–691 | `run_operation2` | 建两个随机矩阵，返回执行二输入 operation 的 closure | §6、§13 matmul |
| 694–697 | `naive_gelu` | tanh 近似 reference | §7.1–7.4 |
| 700–702 | `builtin_gelu` | PyTorch builtin、`approximate="tanh"` | §7.5 |
| 705–706 | `pytorch_softmax` | 沿最后一个 axis 的官方 reference | §11 |
| 709–728 | 三个 `check_equal_*` | 随机生成 1D/2D 输入，用 `torch.allclose(...,atol=1e-6)` 比实现 | §9.7、§16.1 |
| 731–732 | `mean` | `sum(xs)/len(xs)` 算 benchmark 均值 | §6.5 |
| 735–740 | `output_ptx` | 从已编译 kernel 的 `asm["ptx"]` 写文本文件 | §10 |
| 743–744 | `if __name__...` | 直接运行讲义脚本时调用 `main()` | 全讲入口 |

**closure（闭包）**在这里是“记住已经创建的输入 tensor、之后被 benchmark 反复调用的零参数函数”。它避免每次 trial 都把随机输入分配时间混进 kernel 时间。

`torch.allclose(a,b)` 的逐元素判断是：

```math
|a-b|\le atol+rtol\,|b|.
```

- `atol` 是 absolute tolerance（绝对容差）；
- `rtol` 是 relative tolerance（相对容差），会随参考值 $`|b|`$ 增大。

课程源码只显式传 `atol=1e-6`，没有传 `rtol`，因此仍保留 PyTorch 默认 `rtol=1e-5`；它不是“纯绝对误差 $`10^{-6}`$”。若真要只检查绝对误差，应显式写 `rtol=0, atol=1e-6`。默认情况下，两个对应位置都是 `NaN` 也不会被视为相等；若测试契约明确要求“对应 NaN 算相同”，还要有意识地设置 `equal_nan=True`，不能让默认行为替你决定。

另一个覆盖边界：`check_equal_2d_2d` 虽在行 723–728 定义，可用于两个二维输入的函数，但 `main()`/`triton_matmul_relu_example()` 并没有调用它验证 matmul+ReLU。定义了 helper 不等于该路径已被测试。生产测试还应加入 matmul reference、边界 shape、非整除尺寸、NaN/Inf 策略和按 dtype 选择的容差。

### 16.6 课程时点与本环境边界

**【课程总结｜视频 [84:22](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5062s)–[85:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5110s)】**老师回答“除 Triton 还有什么”时强调不同语言/DSL 有不同 inductive bias（更容易表达的程序结构）。PTX 更低层，但不建议初学者把手写 PTX 当第一步；其他 DSL/library 也不一定简单排成上下级。

这份课程 kernel 是为教学展示最小结构，不是生产 matmul 的完整替代品。它没有展开 autotuning、double buffering、async copies、不同 dtype/Tensor Core 路径、split-K、持久化 kernel、复杂 layout 与全部数值边界。

**【视频补充｜[86:12](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5172s)–[86:34](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5194s)】**最后一个问题问“整块读还是逐元素处理更好”；老师的回答是脱离计算性质无法抽象决定。这正是本讲方法：写出数据复用与资源假设，然后测。

本笔记环境仍没有可调用的 NVIDIA CUDA/Triton runtime，因此：

- 所有小矩阵、mask、element-count 与 byte-count 均独立复算；
- 744 行源码已完整静态读取；
- 图片已按原像素视觉检查；
- 没有声称课程 B200 profiler 数字在本机复现；
- 没有为任何 GPU 宣布万能最佳 tile；
- 真正部署前必须在目标硬件做 correctness、benchmark、profile 闭环。

### 16.7 截止 §16 的执行检查单

```text
[ ] A[M,K] 与 B[K,N] 的 inner K 相等
[ ] output C shape 是 [M,N]
[ ] 每个pointer矩阵shape写清
[ ] M/N/K三个边界mask都测试非整除输入
[ ] accumulator dtype满足精度需求
[ ] output store也有M/N mask
[ ] benchmark排除JIT和异步计时错误
[ ] profiler核对实际kernel与HBM行为
[ ] 比较tile候选时输入、dtype、device相同
[ ] fusion后重新检查correctness与occupancy
[ ] 记录硬件/软件版本，不把课堂快照写成定律
```

## 17. 全讲决策树：一个 GPU 操作慢，下一步看什么

### 17.1 第零步：先证明答案基本正确

```text
输出不正确？
  ├─是：停止性能优化
  │    检查shape/dtype/device/stride/mask/边界/数值稳定性
  │    直到和可信reference在合理容差内一致
  └─否：进入可靠benchmark
```

可观察证据：

- 多组 shape，尤其 `N%BLOCK_SIZE != 0` 的边界输入；
- `torch.allclose` 的最大绝对/相对误差；
- NaN、Inf、极大/极小输入；
- output shape、dtype、device；
- memory checker 或测试是否发现越界。

不能因为“随机大矩阵跑一次没报错”就断言 mask 正确。边界 bug 可能只在最后一个 tile 出现。

### 17.2 第一步：建立可信时间，而不是量 CPU 发令速度

```text
先warmup JIT/cache
      ↓
在正确stream边界同步
      ↓
多次测量CUDA events或可靠benchmark工具
      ↓
报告median/mean/分位数与输入条件
```

可观察证据：第一次明显慢于后续，说明包含编译/cold cache；未同步的 CPU timer 极短，说明可能只量到 launch；多次时间有长尾，说明只报 mean 会掩盖分布。

进入下一步前必须写下：shape、dtype、device、warmup 次数、trial 次数、计时单位和软件/硬件版本。

### 17.3 第二步：用 profiler 分类，不靠猜

| 可能瓶颈 | Profiler/计数器中要找的证据 | 还不能仅凭什么下结论 |
|---|---|---|
| launch-bound | 很多持续时间很短的 kernels；GPU timeline 间有 launch gaps；小输入时间近似常数 | “kernel 数多”本身 |
| memory-bound | arithmetic intensity 低；HBM bytes/带宽高；大量中间 tensor load/store；memory stalls 高 | FLOPs 少、或源码里有很多数组 |
| compute-bound | 计算单元利用率高；时间随 FLOPs 增长；memory bandwidth 未饱和 | 看到 `tl.dot` 或 matmul 名字 |
| occupancy/resource-bound | registers/shared memory 限制 resident blocks/warps；同时有 latency stalls | occupancy 百分比低一个指标 |
| bank-conflict | shared-memory replay/conflict 指标高，地址映射到同 bank 不同 word | global load 不连续 |
| uncoalesced HBM | warp global addresses 分散，memory transactions 多、有效 bytes 比低 | shared bank 编号相同 |
| tail/wave | timeline 最后一波只有少量 SM 工作；block 数与可并行 slots 不匹配 | 总 block 数小或大 |

一个 kernel 可同时受两项影响。例如 registers 太多降低 occupancy，导致没有足够 warps 隐藏 HBM latency。分类是形成可检验假设，不是贴唯一标签。

### 17.4 第三步：证据对应修改

```text
launch-bound
  -> 尝试fusion、batch更多工作、减少极小launch

HBM traffic高
  -> 尝试fusion、tiling、复用、避免中间物化

global地址分散
  -> 调layout/offset，让warp访问连续或可合并区域

shared bank conflict
  -> padding/swizzle/改变shared layout

register/shared限制occupancy
  -> 调小tile/减少live temporaries/调num_warps

wave tail明显
  -> 调grid/tile/batch，让最后一波利用率更好

compute利用不足且matmul shape不合适
  -> 调BM/BN/BK、dtype、layout，核对硬件支持路径
```

每个修改后都回到 §17.1：先重新验正确，再用相同条件 benchmark/profile。一次尽量只改变一个主要变量，才能知道差异来自哪里。

### 17.5 为什么单指标不能当判决书

- Occupancy 从 25% 升到 50%，但每 thread 有用工作减半，总时间可能不变或更慢。
- HBM logical traffic 降 4 倍，但 kernel 原本 compute-bound，速度不会跟着 4 倍。
- Kernel 从 5 个融合到 1 个，但 registers 暴涨并 spill，可能更慢。
- Tile 从 32 加到 64，理论 intensity 翻倍，但 padding、资源和 wave tail 可能恶化。
- Mean 下降，但 p95 上升；对 latency-sensitive 服务未必是进步。

正确的结论形式应是：

```text
“在给定GPU、版本、shape、dtype和测量方法下，
 profiler显示X证据；修改Y后，correctness保持，
 median从a降到b，同时关键资源指标怎样变化。”
```

而不是“某技巧永远快”。

## 18. 常见误区：错误说法、为什么错、正确说法

### 18.1 硬件层次与 occupancy

1. **错误说法：low occupancy 必然慢。**
   - 为什么错：少量 warps 可能每个做更多有用工作；kernel 也可能 compute-bound，现有 warps 已够。
   - 正确说法：低 occupancy 是诊断信号；结合 latency stalls、register/shared 限制和实测判断。

2. **错误说法：occupancy 就是“GPU 使用率”。**
   - 为什么错：warp occupancy 是 resident warps 与上限的比，不直接等于 ALU 利用率或整卡忙碌率。
   - 正确说法：明确写 warp occupancy、block/wave utilization 或设备利用率中的哪一个。

3. **错误说法：源码写 64 threads，所以 18.75% 的推导必须用 64。**
   - 为什么错：课程前置文字写 64，但实际变量是 128；用 64 会得到每 block 10,240 registers 和 6 resident blocks，而不是源码实际 3 blocks。
   - 正确说法：按实际变量 `128×160` 复算，并记录这处课程内部不一致。

4. **错误说法：block 越多，wave tail 一定越小。**
   - 为什么错：关键是 blocks 与同时可运行 slots 的整除/余数关系；161 blocks 在 148 SM 上仍有一波很小的尾巴。
   - 正确说法：画每一波并计算最后一波占用，再看每 block 运行时间。

5. **错误说法：register 使用越少越快。**
   - 为什么错：过度减少 registers 可能造成 spill，或让每 thread 做更少复用工作。
   - 正确说法：在 occupancy、spill、instruction count 和时间之间实测权衡。

### 18.2 Banks、coalescing 与同步

6. **错误说法：bank conflict 就是 uncoalesced global access。**
   - 为什么错：bank conflict 是 shared memory banks 的问题；coalescing 是 global/HBM transaction 的问题，存储层不同。
   - 正确说法：shared 地址用 bank mapping 查 conflict；global 地址用 warp transaction 查 coalescing。

7. **错误说法：同一 bank 的任何访问都会冲突。**
   - 为什么错：多个 threads 读取同一个 shared address 可 broadcast。
   - 正确说法：同一 bank 的不同 words 并发访问才需要拆分服务。

8. **错误说法：32 个连续 FP32 永远只需一次 128-byte transaction。**
   - 为什么错：还依赖 128-byte 对齐、cache line/sector 规则和架构；跨边界可能多次 transaction。
   - 正确说法：它是对齐良好时的教学理想，最终看地址与硬件计数器。

9. **错误说法：CPU timer 包住 CUDA 调用就自动量到 GPU 执行。**
   - 为什么错：CUDA launch 通常异步，CPU 可能在 GPU 完成前返回。
   - 正确说法：在正确位置 synchronize，或使用 CUDA events/可靠 benchmark 工具。

10. **错误说法：只在计时前同步就够。**
    - 为什么错：计时后不等 GPU 完成，终点仍只是 CPU 发令完成。
    - 正确说法：清空此前工作后记录 start，并在 end event 或计时结束处等待 GPU 完成。

### 18.3 Benchmark、统计与 profiler

11. **错误说法：第一次运行最快或最真实，不需要 warmup。**
    - 为什么错：第一次可能包含 JIT compilation、allocator 初始化、cache/cudnn/autotuner 等冷启动成本。
    - 正确说法：区分 cold-start 与 steady-state；测后者前 warmup。

12. **错误说法：报告 mean 就完整描述性能。**
    - 为什么错：少数 outliers 会拉高 mean；延迟分布可能有长尾。
    - 正确说法：重要场景同时报告 median、p95/分布、trial 数和异常策略。

13. **错误说法：Profiler 显示一个长 kernel 名，所有机器都会一样。**
    - 为什么错：名字随 GPU 架构、PyTorch/CUDA/CUTLASS 版本、shape/dtype 改变。
    - 正确说法：学习名字编码的线索，不背课堂机器的完整字符串。

14. **错误说法：课程 A100/H100/B200 表是永久硬件常数。**
    - 为什么错：型号、规格、软件支持会变；同系列不同 SKU 也可能不同。
    - 正确说法：标成 2026 课程时点快照，查目标设备官方规格。

### 18.4 Triton、PTX 与 mapping

15. **错误说法：一个 `tl.arange` 元素就是一个 CUDA thread。**
    - 为什么错：Triton 描述 program 的块级向量语义，compiler 决定 threads/warps 和 coarsening。
    - 正确说法：元素是语义位置；实际 mapping 看生成代码与 profiler。

16. **错误说法：课程观察一 thread 处理 8 elements，所以所有 Triton 都是 8。**
    - 为什么错：factor 随 block、num_warps、dtype、编译器、GPU 和 register pressure 变化。
    - 正确说法：只把 8 当该次编译观察。

17. **错误说法：PTX 就是 GPU 最终执行的 SASS。**
    - 为什么错：PTX 是虚拟 ISA，driver 还会翻译成特定架构的 SASS。
    - 正确说法：PTX 用于理解中间低层结构；最终指令看 SASS。

18. **错误说法：Triton compiler 会自动找到全局最优 kernel。**
    - 为什么错：它优化给定程序和配置，但 tile/layout/算法选择仍影响性能，也可能需要 autotuning。
    - 正确说法：Triton 降低低层编程负担，不取消 benchmark/profile。

### 18.5 Mask、padding、fusion 与 traffic

19. **错误说法：有 load mask，就不需要 store mask。**
    - 为什么错：边界 program 仍可能把候选输出写到合法 tensor 之外。
    - 正确说法：load 与 store 各自按地址范围建立 mask。

20. **错误说法：任何 reduction 的 padding 都填 0。**
   - 为什么错：softmax 先做 max；补 0 可能把全负行的 maximum 错改为 0。
   - 正确说法：softmax padding 在至少一个有效值有限时用 $`-\infty`$，使 max 不变且 exp 后为 0；row sum padding 才用 0。若有效行全是 $`-\infty`$，仍会因 $`-\infty-(-\infty)`$ 得 NaN，必须单独处理。

21. **错误说法：fusion 总会更快。**
    - 为什么错：融合可能增加 registers、spill、编译时间，或失去高度优化的 library kernel。
    - 正确说法：先算省掉的中间流量和 launch，再 profile 融合后资源与总时间。

22. **错误说法：traffic 降 4 倍，速度必然快 4 倍。**
    - 为什么错：kernel 可能受 compute、launch、occupancy 或 reduction 限制。
    - 正确说法：traffic ratio 是 Roofline/诊断输入，不是 wall-clock 保证。

### 18.6 Matmul tiling

23. **错误说法：课程写 $`MKN`$ reads，所以 scalar input reads 精确是 $`MKN`$。**
    - 为什么错：每个 $`(m,n,k)`$ 要读一个 $`A`$ 和一个 $`B`$ 标量，精确是 $`2MKN`$；课程省略常数 2 做 big-O。
    - 正确说法：渐近讨论可写 $`O(MKN)`$；精确 bytes 账写 $`2MKN`$。

24. **错误说法：tile 越大永远越好。**
    - 为什么错：复用增加的同时，register/shared 用量、padding、wave tail 和 spill 也可能增加。
    - 正确说法：限定合法候选，在目标 shape、dtype、GPU 上 autotune/benchmark。

25. **错误说法：$`B_K`$ 在简化 global-read 公式中约掉，所以 $`B_K`$ 不影响性能。**
    - 为什么错：它仍影响 `tl.dot` shape、Tensor Core 路径、pipeline、shared/register 资源和循环次数。
    - 正确说法：只说“在可整除的理想 global input element count 中约掉”。

26. **错误说法：$`N=1024,T=32`$ 的总流量精确减少 32 倍。**
    - 为什么错：input reads 是 32 倍，但两种方案都还要写 4 MiB 输出。
    - 正确说法：教学总量是约 8.004 GiB 对 260 MiB，比值约 31.52。

27. **错误说法：课程代码在本笔记环境已经 GPU 实测通过。**
    - 为什么错：当前环境没有可调用的 NVIDIA CUDA/Triton runtime。
    - 正确说法：已做源码、数学、mask、静态结构检查；性能和 GPU correctness 必须在目标 CUDA 环境复测。

28. **错误说法：softmax wrapper 传了 row stride，所以任意 transpose tensor 都支持。**
    - 为什么错：kernel 的 `+col_offsets` 仍隐含 input/output column stride 为 1；`empty_like` 还可能保留非标准 output strides。
    - 正确说法：要么 assert column-contiguous 并分配连续 output，要么传入并乘上 input/output column strides。

29. **错误说法：随机 `allclose` 通过就证明 GeLU 对所有输入稳定。**
    - 为什么错：随机正态样本几乎抽不到 $`x=20`$；课程 `exp(2a)` 路径会在此 FP32 overflow，可能产生 `inf/inf=NaN`。
    - 正确说法：除随机比较外，显式测试极大正负值、NaN、Inf；生产实现使用稳定 builtin/tanh 路径。

## 19. 自测题：先独立写，再看 §20

> 共 60 题。标有“手算”的题要写出中间步骤，不能只报最后数字。

1. 用一句话区分 GPU kernel 与操作系统 kernel。

2. 把 `grid → CTA/thread block → warp → thread → SM` 用人话连接起来；哪些是软件组织，哪个是硬件执行单元？CTA 的 NVIDIA 正式全称是什么，课程源码哪一个词写错？

3. 【手算】一个 warp 有 32 threads。若 20 个走 A 分支、12 个走 B 分支，两个分支各需 3 个相同周期，忽略其他开销，一共消耗多少 warp lane-slots？真正做有用工作的 lane-slots 有多少？利用率是多少？

4. 【手算】每 block 128 threads、每 thread 160 registers、SM 共 65,536 registers、每 warp 32 threads、最多 64 resident warps。只考虑 registers，求 registers/block、resident blocks、resident warps 和 warp occupancy。

5. 【手算】若误用课程文字中的 64 threads/block，而其他数字不变，会得到多少 registers/block、resident blocks、resident warps？为什么 occupancy 百分比碰巧仍可能是 18.75%，但结论仍有问题？

6. 【手算】有 160 个相同 blocks、148 个 SM，每个 SM 每波只跑 1 个该 block。需要几波？最后一波有多少 SM 工作、多少空闲？最后一波利用率多少？

7. 【手算】shared memory 有 32 banks、每 word 4 bytes，bank 公式为 `(byte_address/4) mod 32`。地址 0、4、124、128、132 分别在哪个 bank？

8. 【手算】32 threads 分别读 shared word index 0–31。列出 thread 0、1、30、31 的 bank；是否 conflict？

9. 【手算】32 threads 分别读 word index `32×thread_id`。它们落到哪些 banks？若地址不同，教学模型下要串行多少路？

10. 32 threads 全部读 shared memory 的同一个地址，为什么不按 32-way bank conflict 处理？

11. 【手算】32 threads 连续读 32 个 FP32，总有效 bytes 是多少？在对齐良好的教学模型中需要几个 128-byte transactions？

12. 【手算】同样读连续 128 bytes，但首地址相对 128-byte 边界偏移 4 bytes。覆盖哪两个 128-byte 区间？最少需要几个区间/transactions？

13. 【手算】一个操作做 4,096 FLOPs，从 HBM 读 8,192 bytes、写 8,192 bytes。按总 HBM bytes 求 arithmetic intensity。

14. 画出 `CPU record start → launch kernel → record end → CPU继续` 的异步时间线；为什么普通 CPU 终点不能代表 GPU 完成？课程 CUDA-event elapsed 与同步包围的 CPU wall-clock 分别包含什么？

15. Warmup 至少排除哪两类一次性或冷启动成本？

16. 【手算】四次时间 `[1,1,1,5] ms` 的 mean、population variance 和 median 分别是多少？为什么只报 mean 容易误导？

17. Profiler 里看到许多持续极短、之间有 gaps 的 kernels，支持什么瓶颈假设？还要用什么证据确认？

18. 【手算】按 tanh GeLU 近似，已知 $`a(1)=0.83356197`$、`tanh(a)=0.68238398`，求 GeLU(1)。

19. 【手算】先利用 $`a(-1)=-0.83356197`$、`tanh(a)=-0.68238398` 求 GeLU(-1)。再说明课程 kernel 对 $`x=20`$ 为什么可能产生 NaN，而稳定 GeLU 应约为多少；随机 `allclose` 为什么可能漏掉它？

20. 【手算】$`N=16,384`$ 个 FP32 元素，理想 fused GeLU 只读一次输入、写一次输出。共有多少 element-transfers、bytes、KiB？

21. 为什么不能从 naive GeLU 表达式里数出 9 个运算符，就断言所有机器固定启动 9 个 kernels？

22. 依次解释 CUDA、Triton、PTX、SASS 各在哪一层；PTX 与 SASS 是否相同？

23. 【手算】Triton GeLU 的 $`N=10,BLOCK\_SIZE=8`$，`cdiv` 得多少 programs？两个 programs 的 offsets 分别是什么？

24. 【手算】承接第 23 题，写出 program 1 的 8 项 mask，并列出真正 load/store 的下标。

25. 【手算】typed pointer 的 base byte address 为 1000。offset=8 时，FP32 与 FP16 分别对应哪个 byte address？

26. 在课程 PTX 中，`%ctaid.x`、`%tid.x`、`ld.global`、`st.global` 分别提供什么线索？为什么 `global` 不保证每次都穿透 cache 到 HBM？

27. 什么是 thread coarsening？课程观察“一 thread 处理 8 elements”为什么不能泛化？

28. 【手算】对 `[5,5,5]` 做数值稳定 softmax：写 row max、shifted、exp、denominator 与输出。

29. 【手算】对 `[0,0,100]` 做数值稳定 softmax，使用 $`e^{-100}\approx3.72008\times10^{-44}`$。

30. 【手算】课程 traffic 口径下，$`M=2,N=3`$ 的 naive softmax reads、writes、总 transfers 各多少？逐步骤相加。

31. 【手算】同一 $`2\times3`$ softmax 的 fused 理想 transfers 是多少？naive/fused 比是多少？一般比值为何趋近 4 而不是 4.5？

32. 【手算】`next_power_of_2(3)` 是多少？对长度 3 的 row，写出四项 offsets、mask 和 padding load values。再对 transpose tensor shape `[3,2]`、stride `[1,3]`、storage `[1,2,3,4,5,6]`，说明逻辑 row 0 应读哪些 offsets，而课程 `+col_offsets` 错读哪些 offsets。

33. 【手算】若真实 row 为 `[-5,-6,-7]`，padding 错填 0，row max 会错成什么？用 $`-\infty`$ 时 max 是什么？若所有有效 logits 本身全是 $`-\infty`$，减 max 后发生什么？

34. 【手算】row sum 的输入是 `[1,2,...,12]`、$`B=4`$。写三轮 accumulator，最后求和。

35. 【手算】row sum 的输入是 `[1,2,...,10]`、$`B=4`$。写最后一轮 mask/load、最终 accumulator 与 row sum。

36. 【手算】连续 `x` shape `[4,10]`，row-sum program 处理 `row=2`、当前 `start=4,B=4`。`x_ptr + row*N + cols` 的四个元素 offsets 是多少？

37. 【手算】$`A`$ shape `[3,4]`、$`B`$ shape `[4,2]`，输出 shape 是什么？总输出元素数多少？若 $`B`$ 是 `[5,2]`，为什么不能乘？

38. 【手算】某个 $`A`$ row 为 `[1,2,3]`，对应 $`B`$ column 为 `[4,5,6]`，求一个输出元素。

39. 【手算】$`M=3,K=4,N=2`$ 的 naive matmul，按 scalar-element 口径求 input reads 与 output writes；课程写 $`MKN`$ reads 对应什么简化？

40. 【手算】§13 的 $`4\times4,T=2`$ 例中，左上输出 tile 的两个 partial matrices 是 `[[1,4],[5,16]]` 与 `[[10,4],[22,8]]`。逐项相加得到什么？

41. 方形输出 tile $`T=32`$ 内，每个载入的 A 元素和 B 元素各被多少个输出复用？它们在整个 $`N=1024`$ 乘法中各从 HBM 读约多少次？

42. 【手算】$`M=128,N=256,K=64,B_M=32,B_N=64,B_K=16`$，假设全整除。用“tiles 数×每 tile reads”与化简公式两种方法求 input element reads。

43. 【手算】$`N=1024,T=32`$、FP32：naive 与 tiled input element reads、input bytes 分别是多少？

44. 【手算】承接第 43 题，加上共同的 4 MiB output，求 naive 总 GiB、tiled 总 MiB 和总流量比。

45. 【手算】方阵 FP32 tiled matmul 的教学 arithmetic intensity 约 $`T/4`$ FLOP/byte。$`T=32`$ 时是多少？这是否保证 compute-bound？

46. 【手算】$`M=130,N=70,B_M=B_N=64`$，grid shape 和 program 总数是多少？哪些方向有边界 tile？

47. 【手算】`indices_m=[2,3]`、`indices_k=[0,1,2]`，$`A`$ strides `[4,1]`。写出 `indices_m[:,None]`、`indices_k[None,:]` 的 shapes，以及广播得到的 $`2\times3`$ element-offset matrix。

48. 【手算】`indices_k=[0,1]`、`indices_n=[4,5]`，$`B`$ strides `[5,1]`。写出初始 $`B`$ pointer element-offset matrix。

49. 【手算】边界例 $`M=3,N=5,K=3,B_M=B_N=B_K=2`$，program `(1,2)` 在 `k=0` 时写出 A load mask 和 B load mask。

50. 【手算】承接第 49 题，在第二个 K tile `k=2` 时写 A/B masks；最终 C store mask 是什么？真正写哪个元素？

51. Wrapper 先 `M,K=a.shape`，再 `K,N=b.shape`。为什么覆盖变量仍能工作？怎样写更清楚？

52. `a` tile shape `[B_M,B_K]`、`b` tile `[B_K,B_N]`，`tl.dot` 与 accumulator shape 各是什么？为什么用 FP32 accumulator？

53. 【手算】$`A=[[1,-2],[-3,1]]`$、$`B=[[1,2],[3,-1]]`$。求 $`AB`$ 与 `ReLU(AB)`。

54. 【手算】输出 shape `[128,64]`、元素 FP16（2 bytes）。只比较 output side，separate matmul+ReLU 与 fused 各搬多少 bytes？融合逻辑节省多少 KiB？

55. 给出两个 fusion 可能变慢或不能做的原因。

56. 一个小输入 kernel 的时间几乎不随元素数变，timeline 有许多短 kernels 与 gaps。优先假设什么？可尝试什么？如何验证？

57. Profiler 显示 HBM bytes 很高、arithmetic intensity 低、中间 tensor 多。应优先检查哪三类修改？

58. Occupancy 只有 25%。还要看哪些证据，才能判断提高 occupancy 是否值得？

59. 为什么本笔记用 744 行作为官方源码覆盖基准，而不是旧 raw 工具缓存的 671 行？

60. 写出一次完整性能改动的闭环，从 reference 到最终报告至少包含哪些步骤？同时写出 `torch.allclose(a,b)` 的判定式，并说明课程只传 `atol=1e-6` 时 `rtol` 是否为 0、matmul+ReLU 是否在 `main()` 中调用了已定义的双输入 helper 验证。

## 20. 自测答案：60 题逐步核对

1. GPU kernel 是一次提交给 GPU、由许多 GPU threads 并行执行的小程序；操作系统 kernel 是管理进程、内存、设备等的操作系统核心。两者只是在中文里都常译“内核”，不是同一个概念。

2. 软件先定义一个 grid；grid 切成 CTAs/thread blocks；一个 block 内有许多 threads。硬件把 block 调度到 SM 上，并把 threads 每 32 个组成 warp 执行。Grid/block/thread 是编程组织；SM 是硬件执行与资源单元；warp 是硬件调度/执行分组。CTA 的正式全称是 **Cooperative Thread Array**；课程源码第 61 行误写 `Concurrent`。

3. A 分支执行时，一个 warp 的 32 lane positions 都占 3 周期：

   $`32\times3=96\ \text{lane-slots}.`$

   B 分支同样：

   $`32\times3=96.`$

   总消耗：

   $`96+96=192.`$

   有用 lane-slots：

   $`20\times3+12\times3=60+36=96.`$

   利用率：

   $`96/192=0.5=50\%.`$

4. Registers/block：

   $`128\times160=20{,}480.`$

   Resident blocks：

   $`\left\lfloor65{,}536/20{,}480\right\rfloor=3.`$

   每 block warps：

   $`128/32=4.`$

   Resident warps：

   $`3\times4=12.`$

   Warp occupancy：

   $`12/64=0.1875=18.75\%.`$

5. 误用 64 threads：

   $`64\times160=10{,}240\ \text{registers/block}.`$

   $`\left\lfloor65{,}536/10{,}240\right\rfloor=6\ \text{blocks}.`$

   每 block 是 $`64/32=2`$ warps，所以：

   $`6\times2=12\ \text{warps},`$

   occupancy 仍是：

   $`12/64=18.75\%.`$

   百分比碰巧相同，但 resident blocks 是 6 而不是 3；block 上限、shared memory、同步和 wave 行为都会不同，所以不能用错误输入得到的同百分比冒充正确推导。

6. 第一波可放 148 blocks，还剩：

   $`160-148=12.`$

   因此共 2 波。最后一波 12 个 SM 工作，空闲：

   $`148-12=136.`$

   最后一波利用率：

   $`12/148\approx0.081081=8.11\%.`$

7. 先除以 4 得 word index，再模 32：

   ```text
   address 0   -> 0/4=0   -> bank 0
   address 4   -> 4/4=1   -> bank 1
   address 124 -> 124/4=31-> bank 31
   address 128 -> 128/4=32-> bank 0
   address 132 -> 132/4=33-> bank 1
   ```

8. Thread 0→word0→bank0；thread1→bank1；thread30→bank30；thread31→bank31。32 个 threads 各去不同 bank，教学模型中没有 bank conflict。

9. Thread $`t`$ 的 bank：

   $`(32t)\bmod32=0.`$

   所有 32 个地址落 bank 0，且 word addresses 不同，所以是教学模型中的 32-way bank conflict，需要拆成约 32 路服务。

10. 所有 threads 读的是同一个 word，shared memory 可以把这个值 broadcast 给多个 lanes；冲突的关键是“同 bank 的不同地址”，不是只有 bank 编号相同。

11. 有效 bytes：

   $`32\times4=128\ \text{bytes}.`$

   对齐良好的教学模型中正好覆盖一个 128-byte 区间，所以 1 个 transaction。

12. 若 128-byte 边界从 byte 0 开始，访问从 byte 4 开始连续 128 bytes，会覆盖 byte 4–131。它跨过：

   ```text
   区间0：0–127
   区间1：128–255
   ```

   所以至少涉及 2 个 128-byte 区间/transactions，而不是 1 个。

13. 总 HBM bytes：

   $`8{,}192+8{,}192=16{,}384.`$

   Arithmetic intensity：

   $`4{,}096/16{,}384=0.25\ \text{FLOP/byte}.`$

14. 时间线：

   ```text
   CPU: record start -> enqueue kernel -> record end -> 继续运行
                              |              |
   GPU:                 稍后开始kernel -----完成
   ```

   Launch 是异步入队；CPU 到“record end/普通计时终点”时 GPU 可能尚未完成。需要 CUDA end event 加等待，或结束处 synchronize。

   课程 `elapsed_time(start_event,end_event)` 只量同一 GPU stream 两个 events 之间的 device elapsed；Python 调用与 CPU 等待通常不直接计入，但 GPU 在两个 events 之间等待 host 继续提交工作的设备空档仍可能计入。若用 CPU wall timer 包住 `run()`，并在前后都正确 synchronize，则会包含 host 发令与等待，更接近同步调用者可见时间；计时区间外的 allocation/JIT 仍不自动包含。

15. 至少包括 JIT compilation/cached kernel 生成，以及 allocator/context/cache/autotuner 等冷启动。要区分测 cold-start 还是 steady-state，不能把两者混成一个数字。

16. Mean：

   $`(1+1+1+5)/4=8/4=2\ \text{ms}.`$

   Population variance：

   $`\frac{(1-2)^2+(1-2)^2+(1-2)^2+(5-2)^2}{4} =\frac{1+1+1+9}{4}=3\ \text{ms}^2.`$

   排序仍是 `[1,1,1,5]`，median 是中间两项平均：

   $`(1+1)/2=1\ \text{ms}.`$

   Mean 被 5 ms outlier 拉到 2 ms，不能代表最常见的 1 ms。

17. 它支持 launch-bound 或 CPU/runtime 提交间隙主导的假设。还要检查 GPU timeline、每 kernel 实际 duration、设备空闲 gaps、输入放大后的时间变化，并尝试融合后在 correctness 不变的条件下重新测。

18. 代入：

   $`\mathrm{GeLU}(1) \approx0.5\times1\times(1+0.68238398).`$

   $`=0.5\times1.68238398 =0.84119199.`$

19. 代入：

   $`\mathrm{GeLU}(-1) \approx0.5\times(-1)\times(1-0.68238398).`$

   $`=-0.5\times0.31761602 =-0.15880801.`$

   结果仍为负，不是 0；GeLU 不把所有负数清零。

   对 $`x=20`$：

   $`x^3=8{,}000,`$

   $`a=0.79788456(20+0.044715\times8{,}000) \approx301.376956,`$

   $`2a\approx602.753912.`$

   FP32 的 `exp(602.75)` overflow 为 `inf`，课程改写会出现：

   $`(\text{inf}-1)/(\text{inf}+1) =\text{inf}/\text{inf},`$

   因而可能 NaN。稳定 `tanh(a)\approx1`，所以：

   $`\mathrm{GeLU}(20) \approx0.5\times20\times2=20.`$

   标准随机测试几乎抽不到 20，必须另加极端/NaN/Inf cases。

20. Reads+writes：

   $`2N=2\times16{,}384=32{,}768`$

   次 element-transfers。Bytes：

   $`32{,}768\times4=131{,}072\ \text{bytes}.`$

   KiB：

   $`131{,}072/1{,}024=128\ \text{KiB}.`$

21. 运算符是计算图的概念节点，不是固定 kernel 边界。Backend 可合并标量操作，compiler 可 fusion，某些 temporary 可留在 cache/register；版本、dtype、shape 与 GPU 也会改变生成代码。实际 kernel 数要看当前环境 profiler。

22. CUDA 常以每 thread 写 device kernel；Triton 以每 program/block 的向量工作写 kernel；Triton/CUDA 可编译到 PTX；PTX 是 NVIDIA 虚拟 ISA；driver 再生成特定 GPU 的 SASS。PTX 不是 SASS。

23. Programs：

   $`\lceil10/8\rceil=2.`$

   Program 0：

   $`0\times8+[0,1,\ldots,7]=[0,1,2,3,4,5,6,7].`$

   Program 1：

   $`1\times8+[0,1,\ldots,7]=[8,9,10,11,12,13,14,15].`$

24. 与 `offset<10` 比较：

   ```text
   offsets = [8,9,10,11,12,13,14,15]
   mask    = [T,T,F,F,F,F,F,F]
   ```

   真正 load/store 下标只有 8、9；10–15 全部越界，必须屏蔽。

25. FP32 每元素 4 bytes：

   $`1000+8\times4=1032.`$

   FP16 每元素 2 bytes：

   $`1000+8\times2=1016.`$

26. `%ctaid.x` 给 CTA/block 的 x 编号；`%tid.x` 给 block 内 thread x 编号；`ld.global`/`st.global` 表示 global address-space load/store。Global address space 的访问还可经过 L1/L2 cache，所以名字本身不证明每次触达 HBM。

27. Thread coarsening 是让一个 CUDA thread 处理多个元素。Factor 取决于 block/meta-parameters、dtype、compiler、GPU 和 register pressure；课程的 8 只是该次 PTX 观察，不能泛化。

28. Row max：

   $`m=5.`$

   Shifted：

   $`[5,5,5]-5=[0,0,0].`$

   Exp：`[1,1,1]`；denominator：$`1+1+1=3`$；输出：

   $`[1/3,1/3,1/3].`$

29. Max 是 100；shifted：

   $`[-100,-100,0].`$

   Exp：

   $`[3.72008\times10^{-44},3.72008\times10^{-44},1].`$

   Denominator：

   $`1+2\times3.72008\times10^{-44}\approx1.`$

   输出约等于同一向量。

30. $`MN=2\times3=6`$。Reads：

   ```text
   max 6
   subtract 6+2=8
   exp 6
   sum 6
   normalize 6
   total 6+8+6+6+6=32
   ```

   Writes：

   ```text
   max 2
   subtract 6
   exp 6
   sum 2
   normalize 6
   total 2+6+6+2+6=22
   ```

   总 transfers：

   $`32+22=54.`$

31. Fused 只读输入 $`MN`$、写输出 $`MN`$：

   $`2MN=2\times2\times3=12.`$

   比值：

   $`54/12=4.5.`$

   一般比值：

   $`\frac{8MN+3M}{2MN}=4+\frac{3}{2N}.`$

   当 $`N`$ 增大，$`3/(2N)`$ 趋近 0，所以比值趋近 4。

32. 不小于 3 的最小 2 的幂是 4，所以 `next_power_of_2(3)=4`。

   ```text
   offsets = [0,1,2,3]
   mask    = [T,T,T,F]
   loads   = [x0,x1,x2,-inf]
   ```

   最后一项不能访问真实地址，语义值用 $`-\infty`$。

   Transpose 例中 row 0 base 是 $`0\times stride(0)=0`$。正确 column stride 是 3：

   $`0+[0\times3,1\times3]=[0,3],`$

   从 storage 读 `[1,4]`。课程 kernel 直接加 `[0,1]`，错读 offsets `[0,1]`，得到 `[1,2]`。所以只传 row stride 不足以支持任意 strided tensor。

33. 错填 0：

   $`\max(-5,-6,-7,0)=0,`$

   把真实最大值改坏。填 $`-\infty`$：

   $`\max(-5,-6,-7,-\infty)=-5,`$

   保留真实 row max；且 $`e^{-\infty}=0`$，不进入 denominator。这个结论假设至少有一个有限有效 logit。

   若所有有效值都是 $`-\infty`$，row max 也是 $`-\infty`$，然后每项做：

   $`-\infty-(-\infty)=\mathrm{NaN}.`$

   所以该行不会自动得到全 0 概率；实现要避免 fully masked row 或显式定义特殊结果。

34. 初始 `[0,0,0,0]`。第一轮加 `[1,2,3,4]`：

   $`[1,2,3,4].`$

   第二轮加 `[5,6,7,8]`：

   $`[6,8,10,12].`$

   第三轮加 `[9,10,11,12]`：

   $`[15,18,21,24].`$

   最后：

   $`15+18+21+24=33+45=78.`$

35. 前两轮后 accumulator 是 `[6,8,10,12]`。最后：

   ```text
   cols = [8,9,10,11]
   mask = [T,T,F,F]
   load = [9,10,0,0]
   ```

   更新：

   $`[6+9,8+10,10+0,12+0]=[15,18,10,12].`$

   总和：

   $`15+18+10+12=55.`$

36. Row base：

   $`row\times N=2\times10=20.`$

   当前 `cols=[4,5,6,7]`，所以 offsets：

   $`20+[4,5,6,7]=[24,25,26,27].`$

37. Inner dimensions 都是 4，所以：

   $`[3,4]@[4,2]\to[3,2].`$

   输出元素数：

   $`3\times2=6.`$

   若 $`B=[5,2]`$，inner dimensions 是 4 与 5，不相等；无法把长度 4 的 row 与长度 5 的 column 逐项乘加。

38. Dot product：

   $`1\times4+2\times5+3\times6 =4+10+18=32.`$

39. Input scalar reads：

   $`2MKN=2\times3\times4\times2=48.`$

   Output writes：

   $`MN=3\times2=6.`$

   课程的 $`MKN=24`$ 把每个 $`(m,n,k)`$ 取得的一对 A/B 操作数省略常数 2，当成一次概念读事件用于 big-O。

40. 逐项：

   $`\begin{bmatrix} 1+10&4+4\\ 5+22&16+8 \end{bmatrix} = \begin{bmatrix} 11&8\\ 27&24 \end{bmatrix}.`$

41. 在 $`T=32`$ 输出 tile 内，每个 A 元素横向服务 32 个 output columns；每个 B 元素纵向服务 32 个 output rows，所以都复用 32 次。整个 $`N=1024`$ 输出每边有：

   $`N/T=1024/32=32`$

   个 tiles，所以每个 A/B 输入元素约为不同输出 tile 从 HBM 读 32 次，而不是 naive 的 1,024 次。

42. 输出 tile 数：

   $`(128/32)\times(256/64)=4\times4=16.`$

   每个输出 tile 有：

   $`K/B_K=64/16=4`$

   个 K tiles。每个 K tile 读：

   $`B_MB_K+B_KB_N =32\times16+16\times64 =512+1024=1536.`$

   总读：

   $`16\times4\times1536=98{,}304.`$

   化简公式复核：

   $`MNK=128\times256\times64=2{,}097{,}152,`$

   $`\frac1{B_N}+\frac1{B_M} =\frac1{64}+\frac1{32} =\frac3{64}.`$

   $`2{,}097{,}152\times\frac3{64} =32{,}768\times3 =98{,}304.`$

43. 先重述单位，避免依赖正文：

   $`1\ \text{MiB}=1{,}048{,}576\ \text{bytes}, \quad 1\ \text{GiB}=1{,}024\ \text{MiB} =1{,}073{,}741{,}824\ \text{bytes}.`$

   Naive input elements：

   $`2N^3=2\times1024^3 =2{,}147{,}483{,}648.`$

   FP32 bytes：

   $`2{,}147{,}483{,}648\times4 =8{,}589{,}934{,}592\ \text{bytes}.`$

   $`8{,}589{,}934{,}592/1{,}073{,}741{,}824 =8\ \text{GiB}.`$

   Tiled input elements：

   $`2N^3/T =2{,}147{,}483{,}648/32 =67{,}108{,}864.`$

   Bytes：

   $`67{,}108{,}864\times4 =268{,}435{,}456\ \text{bytes}.`$

   $`268{,}435{,}456/1{,}048{,}576 =256\ \text{MiB}.`$

44. 每个 MiB 仍是 $`1{,}048{,}576`$ bytes。Output bytes：

   $`N^2\times4 =1024^2\times4 =4{,}194{,}304\ \text{bytes}.`$

   $`4{,}194{,}304/1{,}048{,}576 =4\ \text{MiB}.`$

   先把 naive 8 GiB 变成 MiB：

   $`8\times1{,}024=8{,}192\ \text{MiB}.`$

   Naive total 是：

   $`8{,}192+4=8{,}196\ \text{MiB}.`$

   $`8{,}196/1{,}024 =8.00390625\ \text{GiB}.`$

   Tiled total：

   $`256+4=260\ \text{MiB}.`$

   两者已统一为 MiB，所以：

   $`\frac{8192+4}{260} =8196/260 \approx31.523.`$

45. 代入：

   $`I\approx T/4=32/4=8\ \text{FLOP/byte}.`$

   不保证 compute-bound；还要把 8 与该 GPU、该 dtype 的 Roofline ridge point 比较，并检查实际 bytes、计算利用率和资源限制。

46. Row tiles：

   $`\lceil130/64\rceil=3.`$

   Column tiles：

   $`\lceil70/64\rceil=2.`$

   Grid `[3,2]`，programs：

   $`3\times2=6.`$

   $`130`$ 不是 64 的倍数，所以最后 row tile 是边界；$`70`$ 也不是 64 的倍数，所以最后 column tile 也是边界。

47. Shapes：

   ```text
   indices_m[:,None] -> [2,1]
   indices_k[None,:] -> [1,3]
   broadcast result  -> [2,3]
   ```

   Row 2 base：$`2\times4=8`$；row 3 base：$`3\times4=12`$。加 column offsets `[0,1,2]`：

   $`\begin{bmatrix} 8&9&10\\ 12&13&14 \end{bmatrix}.`$

48. $`B[k,n]`$ offset 是 $`k\times5+n`$：

   $`\begin{bmatrix} 0\times5+4&0\times5+5\\ 1\times5+4&1\times5+5 \end{bmatrix} = \begin{bmatrix} 4&5\\ 9&10 \end{bmatrix}.`$

   Col 5 逻辑越界，即使扁平 offset 5/10 落到 storage 的别处，也必须 mask。

49. Program `(1,2)` 有 rows `[2,3]`、cols `[4,5]`、当前 K `[0,1]`。$`M=3`$ 只 row2 有效，$`N=5`$ 只 col4 有效，两个 K 都有效。

   $`mask_A^{(0)}= \begin{bmatrix} T&T\\ F&F \end{bmatrix}, \quad mask_B^{(0)}= \begin{bmatrix} T&F\\ T&F \end{bmatrix}.`$

50. 第二轮有效 K candidates 是 `[2,3]`，而总 $`K=3`$，所以只 K=2 有效。结合 row/column：

   $`mask_A^{(1)}= \begin{bmatrix} T&F\\ F&F \end{bmatrix},`$

   $`mask_B^{(1)}= \begin{bmatrix} T&F\\ F&F \end{bmatrix}.`$

   Store 只检查 rows/cols：

   $`mask_C= \begin{bmatrix} T&F\\ F&F \end{bmatrix}.`$

   真正写 $`C[2,4]`$。

51. 前面的 `assert a.shape[1]==b.shape[0]` 保证 $`K_a=K_b`$，所以覆盖后数值相同。更清楚写法：

   ```python
   M, K_a = a.shape
   K_b, N = b.shape
   assert K_a == K_b
   K = K_a
   ```

52. Matrix multiply：

   $`[B_M,B_K]@[B_K,B_N] \to[B_M,B_N].`$

   `acc` 也必须是 `[B_M,B_N]`。FP32 accumulator 可减少 FP16/BF16 多次乘加的舍入误差；最终输出可再按需求转换。

53. 四项 matmul：

   $`1\times1+(-2)\times3=-5,`$

   $`1\times2+(-2)\times(-1)=4,`$

   $`(-3)\times1+1\times3=0,`$

   $`(-3)\times2+1\times(-1)=-7.`$

   所以：

   $`AB= \begin{bmatrix}-5&4\\0&-7\end{bmatrix}, \quad \mathrm{ReLU}(AB)= \begin{bmatrix}0&4\\0&0\end{bmatrix}.`$

54. 元素数：

   $`MN=128\times64=8{,}192.`$

   Separate output-side 是 $`3MN`$ element-transfers：

   $`3\times8{,}192\times2 =49{,}152\ \text{bytes} =48\ \text{KiB}.`$

   Fused 是 $`MN`$ writes：

   $`8{,}192\times2 =16{,}384\ \text{bytes} =16\ \text{KiB}.`$

   节省：

   $`48-16=32\ \text{KiB}.`$

55. 任举两项：融合可能让 live temporaries/register pressure 增长并 spill；中间结果有其他 consumer 时不能删掉 store；vendor matmul kernel 可能比自写 fused 主体快；跨 program global synchronization 的边界不易融合；运算重排可能改变数值语义。

56. 优先提出 launch-bound/fixed overhead 假设。尝试合理 fusion 或把更多小工作 batch 到一次 launch；用 profiler 看 gaps 是否减少，并在相同 shape/dtype、correctness 保持时比较多次 CUDA-event 时间。若输入放大后 compute/memory 时间开始主导，还要重新分类。

57. 优先检查：能否 fusion 消掉中间 tensor；能否 tiling/reuse 让输入从 HBM 读一次后片上多用；global 地址能否 coalesce。之后核对实际 HBM counters、cache 与 fusion 后资源，而不是只看逻辑公式。

58. 还要看 registers/shared memory 谁限制 resident warps、是否 spill、latency stalls 是否因 warps 不够、计算/带宽利用率、每 thread 工作量和真实时间。若 25% 已隐藏延迟或 kernel compute-bound，强行提高 occupancy 未必收益。

59. 2026-08-28 检查的当前官方 GitHub 页面与首次发布 commit 均显示 744 个物理行；旧 raw 抓取工具曾返回 671 行缓存视图，不能代表当前权威文件。覆盖必须以可回查的当前官方版本为准，同时如实记录差异来源。

60. 一个完整闭环：

   ```text
   定义数学/shape/dtype/device
   -> 与可信reference比较，覆盖边界与极端输入
   -> warmup JIT/cache
   -> 正确同步，多trial benchmark并报告分布
   -> profile kernels、launch、HBM、compute、occupancy、tail
   -> 基于证据选择一个修改
   -> 再做correctness
   -> 同条件重新benchmark/profile
   -> 记录硬件/软件/输入/容差/结果与剩余边界
   ```

   `allclose` 判定：

   $`|a-b|\le atol+rtol|b|.`$

   课程只写 `atol=1e-6`，`rtol` 仍采用 PyTorch 默认 $`10^{-5}`$，不是 0；纯绝对容差要显式传 `rtol=0`。`check_equal_2d_2d` 虽已定义，但 `main()` 的 matmul+ReLU 路径没有调用它，因此不能声称该实现已由 helper 验证。

## 21. 视频导航：使用未在正文重复的人工字幕 cue

> 主字幕轨：`English (United States)`，人工轨、`en-US`、1552 segments，末段约 86:34–86:36。为满足全文 URL 秒点唯一性，本表专门选择正文尚未使用的 cue；每个显示时间都与 `&t=秒数` 一致。

| 时间 | 从这里看什么 | 对应笔记 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5s) | 开场与课程定位 | §0 |
| [04:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=250s) | CTA/thread blocks 与硬件组织的口头说明 | §2 |
| [08:20](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=500s) | 同一正确代码的速度为何受硬件影响 | §2–4 |
| [12:29](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=749s) | 一个 thread 也可处理多个元素 | §2、§10 |
| [16:40](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1000s) | Profiler 可观察 shared-memory bank conflicts | §5–6 |
| [20:48](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1248s) | 课堂问答：性能正确做法与硬件因素 | §4–6 |
| [25:01](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1501s) | CPU 必须等 GPU 完成才能正确计时 | §6 |
| [29:10](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1750s) | Profiler 展示底层 activities | §6.7 |
| [33:21](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2001s) | Naive GeLU 的计算图与底层操作过渡 | §7 |
| [37:31](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2251s) | Triton 与张量式写法的关系 | §8 |
| [41:45](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2505s) | Triton 方括号 launch/grid 语法 | §8–9 |
| [45:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2749s) | GeLU kernel 得到并写回 `y` | §9 |
| [50:01](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3001s) | 从高层 Triton 回看低层执行问题 | §10 |
| [54:07](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3247s) | `%tid.x` 表示 block 内 thread 编号 | §10.2 |
| [58:23](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3503s) | Softmax 按行 exponentiate 与 normalize | §11 |
| [62:31](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3751s) | 一个 softmax program 醒来负责一行 | §11.9–11.10 |
| [66:42](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4002s) | Row sum 把每行归约为一个数 | §12 |
| [70:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4250s) | Accumulator 太大时可能使用 shared memory | §12.6 |
| [75:02](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4502s) | Matmul 中 A 行被多个输出重复使用 | §13.3、§13.8 |
| [79:07](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4747s) | 二维 tensor 的 stride 复习 | §14.2 |
| [83:21](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5001s) | Triton 以 thread blocks 思考的总结 | §16.4 |
| [85:49](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5149s) | 课堂问答：整块读还是逐元素处理没有抽象唯一答案 | §16.6、§17 |

## 22. 来源、744 行覆盖与六张图核验

### 22.1 官方课程来源

- 官方可执行讲义：[Stanford CS336 `lecture_06.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- 官方视频：[Stanford Online Lecture 6](https://www.youtube.com/watch?v=xnDHaNUvHBg)
- 主字幕：YouTube 人工 `English (United States)` 轨，语言代码 `en-US`，1552 segments，末 cue 86:34、约 86:36 结束。

版本事实：旧 raw 抓取工具曾返回 671 行缓存视图；2026-08-28 当前 GitHub 页面与课程首次发布 commit `15d7589` 的文件均为 **744 个物理行**。因此下面以 744 行为唯一覆盖基准，同时保留 671 的差异记录，不用旧缓存删减内容。

### 22.2 课程基础设施与未展开硬件特性

在看行号路由表前，补齐几条容易被“imports/括号说明”藏住的语义：

- `from edtrace import text, link, image`：课程自己的可执行讲义展示层；`text` 渲染说明，`link` 放链接，`image` 放图片。它们不是 GPU kernel API。
- `get_local_url(...)`：把课程生成的本地 PTX 文本路径转成讲义界面可点击的本地 URL；它不编译 PTX。
- `cuda_if_available()`：为 PyTorch 教学输入选择 CUDA device（若可用），否则可回退 CPU。Triton wrappers 仍明确要求 CUDA，所以“PyTorch 能 CPU 跑”不代表 Triton kernel 能 CPU 跑。
- **B200 TMEM（Tensor Memory，张量内存）**：课程行 56 说明这是供 tensor cores 使用、位于 registers 与 shared memory 层级之间、普通程序员不可直接按常规 tensor 操作的片上存储。课程后续最小 kernels 没展开它。
- **Thread-block cluster（线程块簇）**：课程行 64 说明 H100/B200 可把多个 thread blocks 组成 cluster，并使用 **distributed shared memory（分布式共享内存）**跨 cluster 内 blocks 协作。本讲 kernels 仍采用普通“一个 block 驻留一个 SM、使用本 block shared memory”的入门模型。
- `profile()` 行 254–258 不只返回 table，还用 append 模式把时间戳和结果追加到 `var/profiles.txt`；重复运行会保留多次记录，而不是覆盖旧文件。

下面的“1–744 精确覆盖”准确含义是：**行号区间索引没有 gap/overlap，并且每段能路由到笔记解释**。它不宣称 744 个物理行都逐行复刻；例如 import、空行、展示调用由本框集中解释，硬件括号说明也在本框补齐。

### 22.3 1–744 无缝源码覆盖表

> 每个物理行恰好属于一个连续区间；空行/分隔线也合并进相邻教学单元，所以不会因“没有代码”而漏行。

| 行段 | 官方内容 | 笔记位置 |
|---:|---|---|
| 1–38 | imports、课程展示工具、`main()` 全讲路线、分隔行 | 来源说明、§0、§16、§22.2 |
| 39–57 | GPU memory hierarchy、课程硬件快照、B200 TMEM | §1、§22.2 |
| 58–81 | grid、CTA、threads、SM 映射、thread-block clusters | §2、§22.2 |
| 82–91 | warp、lockstep、divergence、latency hiding | §3 |
| 92–113 | registers 限制与 occupancy 计算 | §4 |
| 114–125 | shared-memory banks、conflict、swizzling | §5.1–5.4 |
| 126–131 | HBM memory coalescing | §5.5–5.8 |
| 132–143 | block waves、wave quantization、硬件小结与分隔 | §4.6–4.8、§5 |
| 144–205 | benchmark、CPU timer、warmup、CUDA events、mean 与分隔 | §6.1–6.6 |
| 206–261 | PyTorch profiler、kernel names、matmul/add、追加 `var/profiles.txt` 与分隔 | §6.7–6.9、§22.2 |
| 262–304 | naive/builtin/compiled GeLU、fusion 与分隔 | §7 |
| 305–316 | CUDA per-thread 与 Triton per-block 编程抽象、分隔 | §8 |
| 317–391 | Triton GeLU wrapper/kernel、PTX 输出、分隔 | §9–10 |
| 392–486 | naive/fused softmax wrapper/kernel、流量与分隔 | §11 |
| 487–537 | 超长 row 的 tiled sum wrapper/kernel、分隔 | §12 |
| 538–601 | matmul+ReLU 例、naive/ideal/tiled 思路、stride 演示、分隔 | §13–14 |
| 602–676 | naive matmul+ReLU、Triton wrapper/kernel、fusion、分隔 | §14–15 |
| 677–744 | benchmark closures、GeLU/softmax references、correctness helpers、mean、PTX writer、入口 | §16.5 |

区间检查：第一段从 1 开始，后一段总从前一段末行加 1 开始，最后一段到 744；没有 gap，也没有 overlap。

### 22.4 六张图的视觉核验记录

| 图 | 来源/像素 | 实际看见的结构 | 用到哪里 |
|---|---|---|---|
| `gpu-hardware.png` | 课程本地，1189×933 | GPU 含多个 SM；SM 内有 Reg、L1+shmem；芯片共享 L2；右侧 HBM | §1–2 |
| `grid-with-CTAs.png` | NVIDIA 外链，1048×583 | 绿色 grid 内 8 个 CTAs，每个 CTA 内箭头表示 threads | §2 |
| `wave-quantization.png` | NVIDIA 外链，305×158 | 纵轴 SM、横轴 time；wave 0 填满，wave 1 是部分绿色 tail | §4 |
| `triton-softmax.png` | 课程本地，1536×426 | row 1 由 `pid=1` program load、减 max、exp/sum、normalize/store | §11 |
| `triton-row-sum.png` | 课程本地，1077×871 | row 1 分 tiles 循环；`t0–t3` 是四个 vector accumulator 位置的课程简图，不承诺一位置一 CUDA thread，也不承诺最终只放 registers；mapping 与 register/shared 落点由 compiler 决定；最后两列 mask，和为 39 | §12 |
| `gemm_tiled.png` | 课程本地，850×368 | A 横向 tile 与 B 纵向 tile 乘加到 C 的一个橙色 output tile | §13–15 |

六张均按原像素打开检查；没有以文件名猜内容，也没有发现透明空白、无法读取或关键标注被裁掉。

### 22.5 一手补充来源

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：grid/block/thread、warp/SIMT、memory hierarchy、异步执行。
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：coalescing、shared banks、timing 与 profiling 原则。
- [Triton 官方 Fused Softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：课程 softmax 例直接说明“roughly follow”该教程。
- [Triton 官方 Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：pointer blocks、K-loop、autotuning 与 matmul 实现边界。
- [PyTorch GELU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)：tanh 近似公式。
- [PyTorch `torch.compile` 文档](https://docs.pytorch.org/docs/stable/torch.compiler.html)：编译与 graph capture 的官方边界。
- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：profiler 的官方用法。

这些补充来源只用于核对一般机制和扩展边界；课程的教学顺序、课堂硬件表、profiler 字符串和源码内部 64/128 threads 不一致，均按官方讲义/视频如实呈现，不假装来自补充文档。

### 22.6 测试与环境边界

已完成：

- 完整读取官方 744 行代码；
- 人工字幕 1552 segments 与时间 cue 核对；
- 六图视觉检查；
- 所有自建小矩阵、mask、bytes、traffic、occupancy、wave 数字独立脚本复算；
- Markdown 结构、公式分隔、非法控制字符、时间戳唯一性和题号机械检查。

未完成、也没有冒充完成：

- 当前环境无可调用 NVIDIA CUDA/Triton runtime；
- 未运行 Triton kernel、未生成本机 PTX/SASS；
- 未复现课程 B200 benchmark/profiler 数字；
- 未替任意 GPU 选择“最佳” tile。

因此本笔记可用于理解、手算和代码阅读；部署结论必须在目标 GPU 按 §17 闭环重新验证。

## 23. 学完后的能力清单与一页复习流程

### 23.1 你现在应该能独立完成

- 从 grid/CTA/warp/thread 追到 SM，而不把软件层与硬件层混在一起；
- 复算 registers 限制的 occupancy，并发现 64/128 threads 输入不一致；
- 用 address→bank 表判断 conflict/broadcast，用 warp 地址区分 coalescing；
- 写出不漏同步的 CUDA 异步 benchmark 时间线；
- 用 profiler 证据提出 launch/memory/compute/resource/tail 假设；
- 手算 GeLU、stable softmax、row sum 与小矩阵乘；
- 逐 offset 写 Triton mask，区分 pointer 元素偏移与 byte address；
- 解释 `tl.arange` 向量语义为何不等于 CUDA thread 一一映射；
- 从 `pid_m/pid_n`、`None` broadcasting 构造 A/B/C pointer matrices；
- 推导 naive $`2MKN`$ input reads 与 tiled $`MNK(1/B_N+1/B_M)`$；
- 完整复算 $`N=1024,T=32`$ 的 8.004 GiB、260 MiB、31.52 倍；
- 计算 fusion 省掉的中间 transfers，同时说明它何时可能变慢；
- 区分 PTX 与 SASS，并保持“无 GPU 就不声称实测”的证据边界。

### 23.2 考前一页流程

```text
看数学
  shape对吗？单个输出怎么手算？

看地址
  pointer offset是什么？mask覆盖load和store吗？

看复用
  一个HBM输入载入后在tile内用几次？

看资源
  registers/shared限制多少blocks/warps？最后一波多空？

看时间
  warmup了吗？同步了吗？多trial和分布呢？

看profile
  证据指向launch、HBM、compute、occupancy还是tail？

选修改
  fusion / tiling / coalescing / bank-layout / tile-size

回到开头
  correctness -> benchmark -> profile，保持同一条件比较
```

### 23.3 最后五句话

1. 正确的公式不等于快的 kernel。
2. 快慢来自工作量、数据移动、并行映射与有限硬件资源共同作用。
3. Triton 让你按 program/tile 思考，但 compiler 不替你保证全局最优。
4. Fusion 和 tiling 的共同目标是“从远处搬一次，在近处多用几次”，代价是片上资源。
5. 任何性能结论都必须带 shape、dtype、设备、版本和测量证据。
