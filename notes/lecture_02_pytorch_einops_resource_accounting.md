# CS336 Lecture 2：PyTorch、Einops 与资源会计

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Percy Liang  
> 视频：[Lecture 2: PyTorch, Resource Accounting](https://www.youtube.com/watch?v=kuYAsz7zspQ)（77:16）  
> 官方讲义：[lecture_02.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)（executable lecture，可执行讲义）

> **资料形式说明：**这一讲没有单独的静态 PDF。官方材料是 `lecture_02.py`：老师运行程序来显示文字、公式、代码和实测结果。本笔记以该文件为讲义源，用 Stanford Online 视频的人工字幕 `English (United States)` 补齐口头解释；该轨共有 1425 个字幕片段，最后一个片段的时间戳约为 77:16。自动生成的 `English (auto-generated)` 轨只用于交叉核对听错处，不作为主字幕。

这不是字幕翻译，也不是 API 速查表。目标是：一个从未用过 PyTorch、也不知道 FLOP 和显存是什么的人，可以只读这份笔记，完整走过课程中的代码和计算，并能亲手重算结论。

来源标签：

- **【课程】**：官方 `lecture_02.py` 与视频共同呈现的主内容；
- **【视频补充】**：幻灯片代码中没有，但老师在视频里说出的解释、问答或提醒；
- **【补充解释】**：为了让零基础读者不跳步而加的定义和推导；
- **【补充例子】**：课程之外、用于手算的新例子；
- **【延伸】**：不影响本讲主线，可在第二遍阅读。

本讲所有标为“可运行”的 Python 片段都假设先执行下面这组完整 import：

```python
import time
import torch
import torch.utils.checkpoint
from torch import nn
from einops import einsum, rearrange, reduce, repeat
```

若代码只是为了说明控制流程、仍含有未定义占位名，会在代码块前醒目标成 **“骨架伪代码”**，并逐项解释占位名。没有该标记的代码，可以在执行上述 import 后独立运行，或会明确写“承接上一段代码”。

---

## 0. 五分钟复习卡

> **第一次学习请跳到第 1 节。**本节是压缩后的复习索引，会提前使用还没有解释的词。

### 0.1 一句话主线

大模型训练不是“能运行就够了”：先把数据、参数和中间结果表示成 tensor，再用 shape 明确表达运算，随后逐项计算 **做多少运算、搬多少字节、占多少显存**，才能判断程序受计算速度还是内存带宽限制，并选择混合精度、梯度累积、activation checkpointing 等正确优化。

### 0.2 全讲因果链

```text
模型里的数字必须被存起来并参与运算
        ↓
Tensor = storage 中的数据 + shape/dtype/device/stride 等解释方式
        ↓
Einops/einsum 把每条 axis 的语义写在代码里，减少 shape 错误
        ↓
每个 tensor 都占字节；每个算子都做 FLOPs 并搬 bytes
        ↓
时间下界 = max(计算量/计算吞吐, 数据量/内存带宽)
        ↓
算术强度 I = FLOPs/bytes；与机器转折点 P_peak/B_mem 比较
        ↓
大矩阵乘常为 compute-bound；逐元素算子、matvec 常为 memory-bound
        ↓
反向传播约是前向的 2 倍，dense 训练总计算约 6 × token × 参数
        ↓
训练显存还要放梯度、优化器状态和 activation
        ↓
梯度累积减少一次保存的 activation；checkpointing 用重算换显存
```

### 0.3 必须记住的公式

1. Tensor 大小：

```math
M=\left(\prod_i s_i\right)b
```

$`s_i`$ 是第 $`i`$ 条 axis 的长度，$`b`$ 是每元素字节数，$`M`$ 的单位是 byte。

2. 矩阵乘 $`X_{B\times D}W_{D\times K}`$：

```math
F=BK(2D-1)\approx 2BDK\quad\text{FLOPs}
```

3. 运行时间的理想下界：

```math
t_{\text{ideal}}=\max\left(\frac{F}{P_{\text{peak}}},\frac{Q}{B_{\text{mem}}}\right)
```

$`F`$ 是 FLOPs，$`P_{\text{peak}}`$ 是 FLOP/s，$`Q`$ 是 bytes，$`B_{\text{mem}}`$ 是 byte/s。

4. 算术强度与 Roofline：

```math
I=\frac{F}{Q},\qquad
P_{\text{actual}}\le P_{\text{roofline}}
=\min(P_{\text{peak}},B_{\text{mem}}I)
```

5. dense 网络训练粗估：

```math
C_{\text{train}}\approx 6N D_{\text{tok}}
```

$`N`$ 是每个 token 大致激活的参数数，$`D_{\text{tok}}`$ 是训练 token 数；前向约 $`2N D_{\text{tok}}`$，反向约 $`4N D_{\text{tok}}`$。下标 `tok` 用来避免与矩阵 hidden width $`D`$ 混淆。

### 0.4 三个课程数字

- H100 SXM dense BF16 峰值按课程口径约 `1979/2 = 989.5 TFLOP/s`；显存带宽约 `3.35 TB/s`。
- 机器 Roofline 转折点约 `989.5e12 / 3.35e12 = 295.37 FLOP/byte`。
- AdamW 的课程简化账：参数 2 B + 梯度 2 B + 两个 FP32 状态 8 B = `12 bytes/parameter`，**尚未算 activation、临时 buffer、通信和可能的 FP32 master weight**。

---

## 1. 开始之前：只需要这些前置知识

### 1.1 四则运算和单位

**【补充解释】**本讲的大多数“系统知识”其实是单位换算：

- `1 byte = 8 bits`；
- 十进制：`1 GB = 10^9 bytes`，`1 TB/s = 10^12 bytes/s`；
- 二进制：`1 MiB = 2^20 bytes`，`1 GiB = 2^30 bytes`；
- `1 TFLOP/s = 10^12 FLOP/s`。

`FLOP` 是一次浮点运算，例如一次浮点加法或乘法；`FLOP/s` 是每秒能做多少次。前者是工作量，后者是速度，不能混用。

### 1.2 最小 Python 语法框

> **已经会 Python 可跳过。**

下面只是 **语法示意伪代码**，不是训练循环；其中 `model` 代表稍后定义的网络，`p` 代表循环当前拿到的一个参数：

```python
x = torch.zeros(2, 3)           # 调函数；2、3 是两个位置参数
x.shape                         # 访问 x 的 shape 属性
x.numel()                       # 调 x 的方法，返回元素数
for p in model.parameters():    # model 是占位名；依次拿一个参数
    p.data -= 0.1 * p.grad      # 语法示意；正式更新交给 optimizer
```

- `=` 是“把右边结果交给左边名字”，不是数学上的恒等关系；
- `#` 后面是注释，不会执行；
- `2e9` 表示 $`2\times10^9`$；
- `f"{x}"` 会把变量 `x` 填进字符串；
- `class Block(nn.Module)` 定义一种模块；`def forward(self, x)` 定义输入怎样变成输出；
- `@` 在 tensor 之间表示矩阵乘，不是逐元素乘。

### 1.3 本讲到底在回答什么

**【课程】**老师先提出两个餐巾纸估算问题：

1. 用 1024 张 H100，以 50% 利用率，在 15 万亿 token 上训练 700 亿参数模型要多久？
2. 8 张、每张 80 GB 的 H100，用 AdamW（一种维护两个 moment 的常用 optimizer）最多能容纳多少参数？

答案不是靠猜，而是沿着同一条链：

```text
模型规模 → 工作量/字节量 → 硬件速度/容量 → 时间/能否放下
```

**【视频补充】**老师说这种估算只求量级和结构，不假装包含所有现实开销。先看“骨架”是否合理，再做更细的 profiler（性能分析）。

### 1.4 第一次出现的训练术语

**【补充解释】**后面会反复使用这些词，先把“它是什么、活多久”讲清楚：

- **parameter（参数）**：模型要从数据中学到的数，例如线性层权重；通常从训练开始一直保存到结束；
- **activation（激活/中间结果）**：一次 forward 处理当前 batch 时产生的中间 tensor；backward 常要用，做完这个 batch 后通常可释放；
- **gradient（梯度）**：loss 对 parameter 的变化率，告诉更新方向；每次 backward 产生或累加；
- **optimizer（优化器）**：读取 gradient、维护历史并修改 parameter 的更新算法，不是一个“让代码更快”的编译器；
- **optimizer state（优化器状态）**：optimizer 跨训练 step 保存的历史 tensor；
- **moment（矩）**：Adam 类算法保存的梯度统计。一阶矩可粗看成梯度的移动平均，二阶矩可粗看成梯度平方的移动平均；
- **AdamW**：常用 optimizer；它使用一阶、二阶 moment，并把 weight decay（权重衰减）从梯度更新中解耦；本讲只计算它的显存，不推导完整更新式；
- **master weight（主权重副本）**：某些低精度训练方案额外保存的 FP32 参数副本，用它承接微小更新；不是所有 BF16 配置都有，所以必须查实际实现。

关系可以压成：

```text
parameter --forward--> activation --backward--> gradient
     ^                                         |
     |---- optimizer 读取 gradient 和 state ---|
```

---

## 2. Tensor：先把“装数字的盒子”讲清楚

### 2.1 从标量到高维 tensor

**【补充解释】**tensor（张量）可以先理解为“规则排列的一盒数字”。

| 名称 | 例子 | shape | rank |
|---|---:|---:|---:|
| scalar（标量） | `7` | `()` | 0 |
| vector（向量） | `[2, 5, 8]` | `(3,)` | 1 |
| matrix（矩阵） | `[[1,2,3],[4,5,6]]` | `(2,3)` | 2 |
| rank-3 tensor | 两张 `2×3` 表 | `(2,2,3)` | 3 |

这里第一次定义：

- **shape（形状）**：每条方向有多少个位置。例如 `(2,3)` 表示 2 行、每行 3 个数；
- **rank（阶数）**：shape 里有几个数字。`(2,3)` 有两条 axis，所以 rank 是 2。它不是矩阵的线性代数“秩”；
- **axis（轴，也叫 dimension）**：一条可独立编号的方向。对 `(2,3)`，axis 0 长 2，axis 1 长 3；
- **element（元素）**：盒子中的一个数。

```python
scalar = torch.tensor(7.0)       # shape ()，0 条 axis，1 个元素
vector = torch.tensor([2, 5, 8]) # shape (3,)，1 条 axis，3 个元素
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]]) # shape (2, 3)，2×3=6 个元素
```

每一行翻成人话：

- `torch.tensor(...)` 把 Python 数字或列表变成 tensor；
- `7.0` 没有排成任何方向，所以 shape 是空 tuple `()`；
- 一层列表提供一条 axis；两层嵌套列表提供两条 axis；
- `(3,)` 的逗号表示“只有一个成员的 tuple”，不是小数点。

### 2.2 课程里的 shape 逐级展开

**【课程】**：

```python
x1 = torch.zeros(4)             # (4,)：4 个数
x2 = torch.zeros(4, 8)          # (4, 8)：4×8=32 个数
x3 = torch.zeros(4, 8, 2)       # (4, 8, 2)：4×8×2=64 个数
x4 = torch.zeros(32, 16, 16, 64)
```

最后一行在 Transformer 中可以解释为：

- axis 0：`batch=32`，一次处理 32 条序列；
- axis 1：`sequence=16`，每条序列 16 个 token 位置；
- axis 2：`heads=16`，16 个 attention head；
- axis 3：`head_dim=64`，每个 head 用 64 个数字表示一个位置。

shape 是 `(32,16,16,64)`，rank 是 4，元素总数逐步乘：

```math
32\times16=512
```

```math
512\times16=8192
```

```math
8192\times64=524{,}288\text{ elements}
```

axis 不是固定的“行/列”。同一个 axis 0 在这里表示 batch，在一张图片里可能表示 channel。**数字只告诉长度，变量名和上下文才告诉语义。**

### 2.3 dtype：一个元素用什么格式

**【课程】**`dtype` 是 data type（数据类型）：每个元素怎样用 bit 编码。它同时影响精度、可表示范围、显存和硬件速度。

浮点数可先用这个近似结构理解：

```math
\text{value}=\text{sign}\times\text{significand}\times2^{\text{exponent}}
```

- **sign（符号）**决定正负；
- **exponent（指数）**像科学记数法的“$`10^k`$”中的 $`k`$，主要决定能表示多大、多小的数量级；
- **fraction（小数/尾数字段）**参与组成 significand（有效数字），位数越多，同一数量级中相邻可表示数通常越密，精度越高。

因此“exponent 更多”主要扩大 dynamic range（动态范围），“fraction 更多”主要提高 resolution/precision（分辨率/精度）。真实 IEEE 编码还有 bias、特殊值与隐含位；这里先抓住因果。

| dtype | 每元素 | 课程中的直觉 |
|---|---:|---|
| `float32` / FP32 | 4 bytes | 范围和精度都较好，较耗显存 |
| `float16` / FP16 | 2 bytes | 更省、更快，但范围小，容易 overflow/underflow |
| `bfloat16` / BF16 | 2 bytes | 与 FP32 有相同宽度的 exponent，范围大；fraction 少，精度较粗 |
| FP8 | 1 byte | 还要更谨慎地缩放；H100 支持 E4M3/E5M2 等形式 |
| FP4 / NVFP4 | 0.5 byte（数值本体） | 常按 block 共用 scale；scale 等元数据另占空间 |

两个词：

- **underflow（下溢）**：非零小数太靠近 0，格式装不下，被舍成 0；
- **overflow（上溢）**：数太大，超过格式最大范围，可能变成 `inf`。

课程代码的关键现象：

```python
x_fp16 = torch.tensor([1e-8], dtype=torch.float16)
x_bf16 = torch.tensor([1e-8], dtype=torch.bfloat16)
```

- `x_fp16` 通常显示为 `0`：$`10^{-8}`$ 对 FP16 太小；
- `x_bf16` 仍非零：BF16 的 exponent 范围接近 FP32。

但“范围大”不等于“精度高”。BF16 尾数位更少，相邻可表示数的间距更大。

FP8 名字直接写出字段宽度：

- **E4M3**：4 个 exponent bits、3 个 mantissa/fraction bits，另有 1 个 sign bit；精度相对较好、范围相对较小；
- **E5M2**：5 个 exponent bits、2 个 mantissa/fraction bits，另有 1 个 sign bit；范围更大、精度更粗。

**Tensor Core** 是 NVIDIA GPU 中专门执行小块矩阵 multiply-accumulate 的硬件单元。规格表的 BF16/FP16/FP8 高吞吐通常指符合 shape、布局和 dtype 条件时走 Tensor Core，不是任意一条普通加法都能达到该数字。

### 2.4 device：数字放在哪台计算设备

**【课程】**`device` 表示 tensor 的 storage 所在设备：

```python
x = torch.zeros(2, 3)           # 默认通常在 CPU，shape (2,3)
device = "cuda" if torch.cuda.is_available() else "cpu"
x_device = x.to(device)         # 有 CUDA 就复制到 GPU；否则仍在 CPU；shape (2,3)
```

- CPU 是通用处理器；
- GPU 有大量并行计算单元，擅长大规模规则运算；
- shape 没有变，存放位置变了；
- 一般不能直接让 CPU tensor 与 GPU tensor 做同一条算术运算，需先放到同一 device；
- `.to("cuda")` 可能发生设备间数据传输，这本身要时间。

### 2.5 storage、stride、view、copy、contiguous

这一小节是 **【补充解释】**，但它能解释大量 `reshape`/`transpose` bug。

一个 PyTorch tensor 不只是嵌套列表。可以把它拆成：

```text
storage：底层连续的一维数据
+ shape：每条 axis 多长
+ stride：沿每条 axis 前进一步，要在 storage 跳过几个元素
+ offset：tensor 的第一个元素从 storage 哪里开始
+ dtype/device：元素编码与所在设备
```

以 tensor 为例：

```text
x = [[0, 1, 2],
     [3, 4, 5]]
shape  = (2, 3)
storage= [0, 1, 2, 3, 4, 5]
stride = (3, 1)
```

为什么 stride 是 `(3,1)`？

- axis 0 从 `x[0,0]` 到 `x[1,0]`，storage 从位置 0 跳到位置 3，要跳 3 个元素；
- axis 1 从 `x[0,0]` 到 `x[0,1]`，从位置 0 跳到位置 1，要跳 1 个元素。

现在转置：

```python
x = torch.tensor([[0, 1, 2],
                  [3, 4, 5]])
y = x.transpose(0, 1)
```

结果逻辑上是：

```text
y = [[0, 3],
     [1, 4],
     [2, 5]]
shape  = (3, 2)
stride = (1, 3)
```

PyTorch 可以只交换 shape/stride，不搬 `[0,1,2,3,4,5]`。这叫 **view（视图）**：新 tensor 与旧 tensor 共享同一个 storage，只换了“怎么看”。修改共享数据可能同时影响两者。下面的检查代码**承接上一段中定义的 `y`**：

**copy（拷贝）**则新建 storage 并复制数据，之后两份可以独立修改。copy 更耗时间和显存，但有时不可避免。

**contiguous（连续）**在默认 C-order 下，意思是逻辑上最后一条 axis 的相邻元素也在 storage 中相邻，再往前逐层排好。原始 `x` 是 contiguous；只改 stride 得到的 `y` 通常不是。

```python
assert y.is_contiguous() is False
z = y.contiguous() # 按 y 的逻辑顺序复制到新 storage；shape 仍为 (3,2)
assert z.is_contiguous() is True
```

三个常混淆的方法：

- `view(new_shape)`：必须能只改元数据；不满足 stride 条件时会报错；
- `reshape(new_shape)`：能共享时返回 view，否则可以悄悄 copy；不要依赖它一定不复制；
- `contiguous()`：已经连续则返回自己，否则明确复制为连续布局。

官方说明见 [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)、[`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) 与 [Storage](https://docs.pytorch.org/docs/stable/storage.html)。

### 2.6 第一笔显存账：shape × dtype

**【课程】**tensor 占用字节数：

```math
M=E\times b
```

- $`M`$：memory，单位 byte；
- $`E`$：元素总数，单位 element；
- $`b`$：每个元素的字节数，单位 byte/element。

单位相消：

```math
\text{element}\times\frac{\text{byte}}{\text{element}}=\text{byte}
```

课程小例子：`torch.zeros(4,8,dtype=torch.float32)`。

1. 元素数：$`4\times8=32`$；
2. FP32：$`4`$ bytes/element；
3. 大小：$`32\times4=128`$ bytes。

代码核对：

```python
x = torch.zeros(4, 8, dtype=torch.float32)
elements = x.numel()                 # 32
bytes_per_element = x.element_size() # 4
memory = elements * bytes_per_element# 128 bytes
```

逐行翻译：先创建 4×8 的 FP32 零矩阵；`numel()` 数盒子里有几个数；`element_size()` 问一个数几字节；最后相乘。

课程大例子：GPT-3 风格 MLP 权重，shape 为 `(12288×4, 12288)=(49152,12288)`，FP32。

```math
E=49{,}152\times12{,}288=603{,}979{,}776
```

```math
M=603{,}979{,}776\times4=2{,}415{,}919{,}104\text{ bytes}
```

```math
M/2^{20}=2304\text{ MiB},\qquad M/2^{30}=2.25\text{ GiB}
```

讲义口语化写约 `2.3 GB`；精确说是 `2.416 GB` 十进制或 `2.25 GiB` 二进制。

### 2.7 混合精度不是“全部改成 16 bit”

**【课程】**mixed precision（混合精度）是不同数据或算子使用不同 dtype：

- 课程简化图：参数、activation、gradient 用 BF16；optimizer state 用 FP32；
- `autocast` 让适合低精度的运算（例如大矩阵乘）自动使用低精度，而 `exp`、归一化等敏感运算可以保留更高精度；
- FP16 小梯度可能下溢，可用 gradient scaling（先放大 loss 和梯度，更新前再缩回）减轻问题；BF16 通常范围更安全，但也不是永不出数值问题。

**【补充解释】**实际框架/优化器配置可能还保存 FP32 master weights。不能看到“BF16 训练”就武断地把每参数显存算成 2 bytes。要逐项问：参数本体、梯度、master copy、moment、activation 各是什么 dtype。

官方 AMP 建议只用 autocast 包住 forward 和 loss，随后在 autocast 区域外 backward；详见 [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)。

**【视频补充：课堂问答】**block-scaled 低精度让一组数共享 scale，能扩展整组覆盖的范围，但组内相邻数仍不能各自拥有任意远的量级；老师还区分了“1-bit 量化用于训练后压缩”和“真正用极低 bit 训练”，后者更困难。这是概念介绍，不代表所有系统采用相同格式。

---

## 3. Axis 有名字以后：einsum 与 einops

### 3.1 为什么只写 `transpose(-2,-1)` 很危险

**【课程】**旧式代码：

```python
x = torch.ones(2, 2, 3)              # (batch=2, seq=2, hidden=3)
y = torch.ones(2, 2, 3)              # (batch=2, seq=2, hidden=3)
z = x @ y.transpose(-2, -1)           # (2,2,3) @ (2,3,2) -> (2,2,2)
```

逐行翻译：

1. `x` 有 2 个 batch；每个 batch 有 2 个序列位置；每个位置 3 个 hidden 数；
2. `-1` 是最后一条 axis（hidden），`-2` 是倒数第二条（seq）；
3. `y.transpose(-2,-1)` 把最后两条 axis 对换，`(2,2,3)→(2,3,2)`；通常只是 view；
4. 每个 batch 内做 `(2×3)@(3×2)→(2×2)`；最终 `z` 是 `(batch=2, seq_x=2, seq_y=2)`。

问题是：`-2`、`-1` 没说轴的含义。一旦上游 shape 改了，代码可能报错，更糟的是可能 shape 仍合法、语义却错了。

### 3.2 `einsum`：相同名字对齐，未输出的名字求和

`einsum` 来自 Einstein summation（爱因斯坦求和记号）。核心规则只有三句：

1. 输入 pattern 给每条 axis 起名字；
2. 同名 axis 的位置对应相乘；
3. 出现在输入、没出现在 `->` 右侧的名字，被求和消掉。

**【课程例 1：普通矩阵乘】**

```python
x = torch.ones(3, 4)  # (seq1=3, hidden=4)
y = torch.ones(4, 3)  # (hidden=4, seq2=3)
z = torch.einsum("ij,jk->ik", x, y)  # (3,3)
```

课程使用的 `einops.einsum` 可以写长名字：

```python
x = torch.ones(3, 4)
y = torch.ones(4, 3)
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
assert z.shape == (3, 3)
```

两行含义相同。`hidden` 在输入出现、输出消失，所以对 hidden 的 4 个位置求和。`seq1` 和 `seq2` 保留，输出 shape 是 `(3,3)`。

手算第 `(0,0)` 个输出：所有输入都是 1，

```math
z_{0,0}=1\times1+1\times1+1\times1+1\times1=4
```

**【课程例 2：带 batch】**

```python
x = torch.ones(2, 3, 4) # (batch=2, seq1=3, hidden=4)
y = torch.ones(2, 3, 4) # (batch=2, seq2=3, hidden=4)
z = einsum(x, y,
           "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# shape: (2,3,4),(2,3,4) -> (2,3,3)
```

- `batch` 出现在两输入和输出：同一个 batch 内计算，不跨 batch 混合；
- `hidden` 没在输出：长度 4 被乘加求和；
- `seq1,seq2` 留在输出：每个 batch 得到 `3×3` 的两两分数；
- `...` 可代表暂时不想逐一命名的一串前导 axis，但初学时建议先写全。

### 3.3 `rearrange`：只重排/拆分/合并，不做求和

**【课程原例】**输入 `x` shape `(seq=3, hidden=8)`，把 hidden 拆成 `heads=2` 和 `hidden1=4`：

```python
x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
# (3,8)，24 个 FP32 元素；设为 float32 是为了与后面的 w dtype 一致
xh = rearrange(x, "seq (heads hidden1) -> seq heads hidden1",
               heads=2)
# (seq=3, hidden=8) -> (seq=3, heads=2, hidden1=4)
```

因为总元素数不变：

```math
3\times8=24=3\times2\times4
```

`heads=2` 已知，故 `hidden1=8/2=4`。括号 `(heads hidden1)` 表示这两条 axis 在输入中合成一条。下面代码**承接上一段中定义的 `xh`**：

```python
w = torch.ones(4, 4) # (hidden1=4, hidden2=4)
yh = einsum(xh, w,
            "seq heads hidden1, hidden1 hidden2 -> seq heads hidden2")
# (3,2,4),(4,4) -> (3,2,4)
y = rearrange(yh, "seq heads hidden2 -> seq (heads hidden2)")
# (3,2,4) -> (3,8)
```

第一行对 `hidden1=4` 求和；最后一行把 `heads=2` 与 `hidden2=4` 合并，$`2\times4=8`$。

这里不能只看 shape。把值全部摊开：

```text
xh[seq=0] = [[ 0, 1, 2, 3], [ 4, 5, 6, 7]]
xh[seq=1] = [[ 8, 9,10,11], [12,13,14,15]]
xh[seq=2] = [[16,17,18,19], [20,21,22,23]]
```

`w` 的每一列全是 1，所以每个 head 的 4 个数会被求和，再把同一个和复制到 4 个 `hidden2` 位置：

```text
seq 0: head 0 sum = 0+1+2+3   =  6；head 1 sum = 4+5+6+7   = 22
seq 1: head 0 sum = 8+9+10+11 = 38；head 1 sum = 12+13+14+15 = 54
seq 2: head 0 sum = 16+17+18+19 = 70；head 1 sum = 20+21+22+23 = 86

yh = [
  [[ 6, 6, 6, 6], [22,22,22,22]],
  [[38,38,38,38], [54,54,54,54]],
  [[70,70,70,70], [86,86,86,86]],
]

merge 保持 `(heads hidden2)` 的顺序：
y = [
  [ 6, 6, 6, 6,22,22,22,22],
  [38,38,38,38,54,54,54,54],
  [70,70,70,70,86,86,86,86],
]
```

这就闭合了“拆 heads → 每个 head 独立运算 → 按原 head 顺序合回 hidden”。

**【补充例子 1：转置一个小矩阵】**

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])          # (row=2, col=3)
out = rearrange(a, "row col -> col row")
# [[1,4],
#  [2,5],
#  [3,6]]                   # (3,2)
```

只是把 axis 顺序从 `(row,col)` 改成 `(col,row)`，元素总数仍是 6。

**【补充例子 2：合并顺序真的会改变排列】**

```python
a = torch.tensor([[[1,2], [3,4]],
                  [[5,6], [7,8]]]) # (h=2,w=2,c=2)
r1 = rearrange(a, "h w c -> h (w c)")
# [[1,2,3,4], [5,6,7,8]]            shape (2,4)
r2 = rearrange(a, "h w c -> h (c w)")
# [[1,3,2,4], [5,7,6,8]]            shape (2,4)
```

`(w c)` 表示 `c` 变化最快；`(c w)` 表示 `w` 变化最快。两者 shape 相同，数的顺序不同。

**【视频补充：课堂问答】**学生问这是 row-major 还是 column-major。老师回答：einops pattern 中括号内名字的顺序已经指定 flatten 顺序。官方 einops 文档称组合 axis 使用 C-order enumeration，即括号最右侧 axis 变化最快。

### 3.4 `reduce`：消掉的 axis 真的被汇总

**【课程原例】**

```python
x = torch.ones(2, 3, 4)                  # (batch=2, seq=3, hidden=4)
y = reduce(x, "... hidden -> ...", "sum")# (2,3)
```

`hidden` 长度 4 在右边消失，所以每个 `(batch,seq)` 位置把 4 个 1 相加，输出每项是 4；`...` 保留 `(2,3)` 两条 axis。

**【补充例子 1：按行求和】**

```python
a = torch.tensor([[1,2,3],
                  [4,5,6]])                    # (row=2,col=3)
out = reduce(a, "row col -> row", "sum")
# [1+2+3, 4+5+6] = [6,15]            # shape (2,)
```

**【补充例子 2：2×2 分块平均】**

```python
a = torch.tensor([[1, 3, 5, 7],
                  [2, 4, 6, 8]], dtype=torch.float32) # (h=2,w=4)
out = reduce(a, "h (group two) -> h group", "mean", two=2)
# [[(1+3)/2, (5+7)/2],
#  [(2+4)/2, (6+8)/2]]
# = [[2,6],[3,7]]                     # shape (2,2)
```

`rearrange` 不能丢元素，`reduce` 可以通过 `sum/mean/max/min` 等规则减少元素数。

### 3.5 `repeat`：创建新 axis 或沿 axis 复制

`repeat` 没有出现在本讲主代码中，是 **【补充例子】**，但与另外两个 einops 核心操作一起学最不容易混。

**例 1：给向量复制两个 batch**

```python
v = torch.tensor([10, 20, 30])         # (hidden=3,)
out = repeat(v, "hidden -> batch hidden", batch=2)
# [[10,20,30],
#  [10,20,30]]                         # shape (2,3)
```

元素数从 3 变成 $`2\times3=6`$。

**例 2：tile 整行与重复每个元素不同**

```python
v = torch.tensor([1,2,3])              # shape (3,)
tiled = repeat(v, "w -> (tile w)", tile=2)
# [1,2,3,1,2,3]                       # shape (6,)

copied = repeat(v, "w -> (w copy)", copy=2)
# [1,1,2,2,3,3]                       # shape (6,)
```

括号顺序仍决定谁变化最快。官方用法见 [einops 首页](https://einops.rocks/) 和 [`repeat` API](https://einops.rocks/api/repeat/)。

### 3.6 `reshape` 与 `transpose` 的经典陷阱

**【补充解释】**假设：

```text
x = [[1,2,3],
     [4,5,6]]               shape (2,3)
```

- 正确转置得到 `[[1,4],[2,5],[3,6]]`，shape `(3,2)`；
- 直接 `x.reshape(3,2)` 只是按 storage 顺序重新切：`[[1,2],[3,4],[5,6]]`。

两者 shape 一样，值不同。**reshape 不等于 transpose。**

还有一个性能陷阱：

```python
x = torch.tensor([[1,2,3], [4,5,6]])
y = x.transpose(0, 1) # 通常共享 storage，non-contiguous，shape (3,2)
z = y.reshape(6)      # 可能必须复制，不能假设 O(1)
```

einops 让语义更清楚，但底层能否返回 view 仍取决于布局；必要时也会 copy。

---

## 4. 资源会计第一条闭环：参数量 → 字节 → 能否放下

### 4.1 参数是什么，怎样计数

**【补充解释】**parameter（参数）是训练要修改的 tensor 元素。一个无 bias 的线性层权重 $`W`$ shape 为 `(D,K)`，参数数：

```math
N_W=D\times K
```

若另有 bias $`b`$ shape `(K,)`，总参数：

```math
N=DK+K
```

例：输入宽度 3、输出宽度 2：

```text
W shape (3,2): 3×2=6 parameters
b shape (2,):  2 parameters
total:         8 parameters
```

参数量是“有几个数”，显存是“这些数以及训练附属数据一共多少字节”。

### 4.2 单个 tensor 与整个模型

模型若有参数 tensor $`p_1,p_2,\ldots`$，其参数本体显存：

```math
M_{\text{param}}=\sum_j \mathrm{numel}(p_j)\times\mathrm{element\_size}(p_j)
```

如果所有 $`N`$ 个参数都是 BF16：

```math
M_{\text{param}}=N\times2\text{ bytes}
```

700 亿参数的本体：

```math
70\times10^9\times2=140\times10^9\text{ bytes}=140\text{ GB}
```

这已经大于一张 80 GB GPU，但训练总显存远不止参数本体。

### 4.3 8×80 GB 与 AdamW 的课程估算

**【课程】**总设备显存：

```math
8\times80\text{ GB}=640\text{ GB}=640\times10^9\text{ bytes}
```

课程的 mixed-precision AdamW 简化账，对每个参数：

| 项目 | dtype | bytes/parameter |
|---|---:|---:|
| 参数 | BF16 | 2 |
| 梯度 | BF16 | 2 |
| Adam 一阶矩 $`m`$ | FP32 | 4 |
| Adam 二阶矩 $`v`$ | FP32 | 4 |
| 合计 |  | 12 |

因此参数上界：

```math
N_{\max}=\frac{640\times10^9\text{ bytes}}{12\text{ bytes/parameter}}
```

先约分单位，再算数字：

```math
N_{\max}=53.333\ldots\times10^9\text{ parameters}\approx53.3\text{B parameters}
```

这是**不可能真正达到的容量上界**，因为没有给 activation、临时 workspace、CUDA context、allocator 碎片、通信 buffer 留空间，也假设 8 张卡能完美切分。

若实现还保存 4-byte FP32 master parameter，每参数变为 $`12+4=16`$ bytes：

```math
640\times10^9/16=40\times10^9=40\text{B parameters}
```

仅一个配置差异，上界就从 53.3B 降到 40B。

---

## 5. 资源会计第二条闭环：一项运算到底做多少 FLOPs

### 5.1 FLOP 和 FLOP/s 再分一次

**【课程】**：

- `FLOP`（floating-point operation）是一次浮点加、乘等操作；复数运算、除法、`exp` 的硬件代价不一定都等于一次简单加法，所以按 FLOP 计数是模型，不是物理定律；
- `FLOPs` 在口语中常表示复数“很多次浮点运算”；
- `FLOP/s` 是速度。例如 `989.5 TFLOP/s = 989.5×10^12 FLOP/s`。

若工作量 $`F`$ 的单位是 FLOP，实际吞吐 $`P`$ 的单位是 FLOP/s：

```math
t=\frac{F}{P}
```

单位检查：

```math
\frac{\text{FLOP}}{\text{FLOP}/\text{s}}=\text{s}
```

### 5.2 从一个输出元素推导矩阵乘

**【课程】**设：

- $`X`$ shape `(B,D)`；$`B`$ 可理解为 batch 中有多少行，$`D`$ 是输入宽度；
- $`W`$ shape `(D,K)`；$`K`$ 是输出宽度；
- $`Y=XW`$ shape `(B,K)`。

输出第 $`b`$ 行、第 $`k`$ 列：

```math
Y_{b,k}=\sum_{d=1}^{D}X_{b,d}W_{d,k}
```

先只算一个 $`Y_{b,k}`$：

- 有 $`D`$ 对数相乘：$`D`$ 次 multiplication；
- 把 $`D`$ 个乘积加起来：$`D-1`$ 次 addition；
- 合计：$`D+(D-1)=2D-1`$ FLOPs。

输出一共有 $`B\times K`$ 个元素，所以精确朴素计数：

```math
F_{\text{matmul}}=BK(2D-1)
```

当 $`D`$ 很大，$`-1`$ 相对 $`2D`$ 很小，常写：

```math
F_{\text{matmul}}\approx2BDK
```

这里约定一次 multiply 和一次 add 各算 1 FLOP；硬件可能用一条 fused multiply-add 指令完成，但行业通常仍把它算 2 FLOPs。

**【补充例子：所有数字手算】**

```math
X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
W=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
```

$`X`$ 是 `(B=2,D=3)`，$`W`$ 是 `(D=3,K=2)`，输出 `(2,2)`。

第一个输出：

```math
Y_{1,1}=1\times1+2\times0+3\times1=4
```

这用了 3 次乘法、2 次加法，共 $`2D-1=5`$ FLOPs。4 个输出共：

```math
B\times K\times(2D-1)=2\times2\times5=20\text{ FLOPs}
```

近似式给 $`2BDK=2\times2\times3\times2=24`$ FLOPs。小矩阵上相差 4，说明 `≈` 不是 `=`；当 $`D=12288`$ 时，少算的相对比例才很小。

### 5.3 训练 70B 模型的时间估算

**【课程】**dense Transformer 训练常用：

```math
C\approx6N D_{\text{tok}}
```

- $`C`$：总训练计算，单位 FLOP；
- $`N`$：每个 token 大致激活的模型参数数，单位 parameter；
- $`D_{\text{tok}}`$：训练 token 数，单位 token；
- 系数 6：每个参数、每个 token，前向大约 2 FLOPs，反向大约是前向 2 倍，即再约 4 FLOPs；第 10 节会推。

给 $`N=70\times10^9`$、$`D_{\text{tok}}=15\times10^{12}`$：

```math
C\approx6\times(70\times10^9)\times(15\times10^{12})
```

先算普通数字：

```math
6\times70\times15=6300
```

再算十的幂：

```math
10^9\times10^{12}=10^{21}
```

所以：

```math
C=6300\times10^{21}=6.3\times10^{24}\text{ FLOPs}
```

课程采用 H100 SXM 标称 BF16 Tensor Core `1979 TFLOP/s` 的 sparse 数字，再除以 2 得 dense。这里先把三个词拆开：

- **dense（稠密）矩阵**：不假定权重中有可跳过的 0，所有位置都按普通矩阵计算；
- **supported structured sparsity（硬件支持的结构化稀疏）**：0 必须满足硬件规定的排列。课程规格口径对应常见 2:4 结构，即每连续 4 个候选值中只有 2 个非零；不是“随便删掉一半权重”也自动加速；
- H100 Tensor Core 能跳过这种受支持结构里的 0。规格表按 dense-equivalent operations 报告时，同一时间相当于处理两倍 dense 工作量，因此 sparse 数字约为 dense 的 2 倍。

所以本讲训练的是一般 dense 模型时，不能把 1979 直接当峰值，要除以 2：

```math
P_{\text{one,peak}}=\frac{1979\times10^{12}}{2}
=989.5\times10^{12}\text{ FLOP/s}
```

若 model FLOPs utilization（MFU，模型浮点运算利用率）为 50%，1024 张卡的实际模型吞吐粗估：

```math
P_{\text{cluster}}=989.5\times10^{12}\times0.5\times1024
```

```math
=5.06624\times10^{17}\text{ FLOP/s}
```

一天 86400 秒，每天完成：

```math
5.06624\times10^{17}\times86400
=4.37723136\times10^{22}\text{ FLOP/day}
```

所需天数：

```math
\frac{6.3\times10^{24}}{4.37723136\times10^{22}}
=143.93\text{ days}\approx144\text{ days}
```

**【视频补充】**课堂口头报告约 143 天；按讲义中上述显示数字逐位计算是 143.93 天，即四舍五入约 144 天。这一级估算不应被理解为精确工期。

**没有算进去的现实因素：**通信、数据加载、checkpoint 写盘、故障、验证、warmup、非矩阵算子、负载不均与集群可用率。因此 50% MFU 已把一部分损失揉进一个经验系数，但不能替代详细计划。

### 5.4 GPU benchmark 为什么要 synchronize

**【课程】**GPU kernel launch（启动 GPU 任务）通常是 asynchronous（异步）：CPU 发出命令后可以立刻继续，不等 GPU 做完。

下面两段都是 **计时骨架伪代码**，用于对比错误和正确顺序，不可直接复制运行。占位名：`x,w` 是已经放在 CUDA GPU、shape 可相乘的 tensor；`flops` 是按 shape 算出的工作量；真实代码还需 warmup，并确保机器有 CUDA。

错误计时骨架：

```python
start = time.time()
y = x @ w
elapsed = time.time() - start
```

这可能只量到“CPU 把任务放进队列”所需时间。课程正确骨架：

```python
torch.cuda.synchronize()       # 等待此前所有 GPU 工作结束
start = time.time()            # 起点干净
y = x @ w                      # 发起矩阵乘
torch.cuda.synchronize()       # 等矩阵乘真的结束
elapsed = time.time() - start  # 现在才是完成时间
actual_flops_per_sec = flops / elapsed
```

逐行状态变化：第一次同步清空旧工作；记时；发起 kernel；第二次同步等结果；相减得到秒；最后必须用“工作量除以时间”。

**【视频勘误】**约 36:11 处口头一度把实际 FLOP/s 说成 FLOPs “乘”时间；官方代码和单位分析都明确是 `flops / elapsed`。

实测还应 warm up、多次重复、报告中位数或分位数。第一次运行可能包含初始化、编译和 cache 冷启动。

---

## 6. 峰值、实际值与 MFU

### 6.1 峰值不是保证

**【课程】**硬件规格的 `P_peak` 是满足特定 dtype、形状、Tensor Core、功耗和稀疏性等条件时的上限。H100 表格中的 `1979 TFLOP/s` 是受支持 structured sparsity 的 dense-equivalent 口径；普通 dense BF16 没有 2:4 零模式可跳，课程口径除以 2 得 989.5 TFLOP/s。

实际程序还会损失在：

- 数据没及时送到计算单元；
- shape 不利于硬件 tile；
- kernel 启动和 Python overhead；
- 通信、同步、数据加载；
- 非矩阵算子；
- 小问题没有足够并行度。

### 6.2 MFU 怎样算

**【课程】**对明确的模型计算量：

```math
\mathrm{MFU}=\frac{P_{\text{actual model}}}{P_{\text{promised peak}}}
```

- 分子：按模型公式统计的有用 FLOP/s；
- 分母：同 dtype、同 dense/sparse 口径下的硬件峰值 FLOP/s；
- MFU 无单位，通常写百分比。

例：某 H100 dense BF16 训练实际模型吞吐 `500 TFLOP/s`：

```math
\mathrm{MFU}=\frac{500\times10^{12}}{989.5\times10^{12}}
=0.5053=50.53\%
```

十的幂抵消，所以也可直接算 `500/989.5`。

**【视频补充】**老师说训练 MFU 超过约 50% 通常已经不错；单纯大矩阵乘有时可到约 80%。这不是所有硬件、所有模型的硬阈值。

MFU 与 hardware FLOPs utilization（HFU）口径可能不同：activation checkpointing 的重算对硬件确实做了 FLOPs，可能提高 HFU 计数，却不是模型原始前后向的“有用”FLOPs。比较论文时必须读定义。

---

## 7. 不只会算：数据搬运也要时间

### 7.1 bandwidth 与最小搬运时间

**【课程】**memory bandwidth（内存带宽）是显存每秒最多可传多少字节。课程对 H100 使用：

```math
B_{\text{mem}}=3.35\text{ TB/s}=3.35\times10^{12}\text{ byte/s}
```

若算子至少要搬 $`Q`$ bytes，理想搬运时间：

```math
t_{\text{memory}}=\frac{Q}{B_{\text{mem}}}
```

单位：

```math
\frac{\text{byte}}{\text{byte}/\text{s}}=\text{s}
```

**【补充解释：先指定在哪一层数 bytes】**GPU 不是只有一个“内存”：

```text
HBM（High Bandwidth Memory，高带宽显存）
    容量最大、离计算单元最远，本讲用 3.35 TB/s 描述这一层
        ↓
片上 SRAM（例如 L2 cache/shared memory）
    容量小得多、速度更高，可保存反复使用的 tile
        ↓
register（寄存器）
    每个线程直接使用的极小、极快存储
        ↓
Tensor Core / 普通计算单元
```

同一个矩阵元素可能只从 HBM 读一次，却从 SRAM/register 被使用很多次。因此 $`Q`$ 不是脱离层级的唯一真值。**本讲所有 Roofline 数字默认 $`Q=\text{HBM 与芯片之间传输的 bytes}`$**，因为 $`B_{\text{mem}}=3.35`$ TB/s 也是 HBM 带宽。若改用 SRAM bandwidth，就必须重新数 SRAM 边界的 $`Q`$，不能拿一种 $`Q`$ 配另一层 bandwidth。

### 7.2 arithmetic intensity：每搬一个字节做多少运算

**【课程】**arithmetic intensity（算术强度）：

```math
I=\frac{F}{Q}
```

- $`I`$：单位 FLOP/byte；
- $`F`$：这个算子的浮点工作量；
- $`Q`$：数据在目标内存层级与计算单元之间至少搬运的字节。

注意“目标内存层级”：从 HBM 读到芯片与从 SRAM/register 读不是同一个 $`Q`$。本讲 Roofline 主要针对 GPU HBM 带宽，并采用理想的“必要数据各读一次、输出写一次”估算。

### 7.3 Roofline 从时间公式推出来

若计算和搬运完美重叠，总时间仍不能小于较慢者：

```math
t_{\text{ideal}}=\max\left(\frac{F}{P_{\text{peak}}},\frac{Q}{B_{\text{mem}}}\right)
```

Roofline 给出的不是实测值，而是在上述理想 traffic 和完美重叠假设下的**吞吐上界**。先定义：

- $`P_{\text{roofline}}`$：Roofline 模型允许的最高吞吐；
- $`P_{\text{actual}}`$：真实程序测到的吞吐。

于是：

```math
P_{\text{actual}}\le P_{\text{roofline}}
=\min(P_{\text{peak}},B_{\text{mem}}I)
```

机器的转折算术强度（ridge point）：

```math
I_{\text{ridge}}=\frac{P_{\text{peak}}}{B_{\text{mem}}}
```

代入课程 H100 dense BF16 数字：

```math
I_{\text{ridge}}
=\frac{989.5\times10^{12}\text{ FLOP/s}}
{3.35\times10^{12}\text{ byte/s}}
```

秒与 $`10^{12}`$ 同时消掉：

```math
I_{\text{ridge}}=\frac{989.5}{3.35}=295.373\ldots\text{ FLOP/byte}
```

判断法：

- $`I<295.37`$：带宽先碰顶，memory-bound（内存受限）；
- $`I>295.37`$：计算先碰顶，compute-bound（计算受限）；
- 附近两者都重要。

理想 MFU 上界也可写：

```math
\mathrm{MFU}_{\text{roofline}}\le
\min\left(1,\frac{I}{I_{\text{ridge}}}\right)
```

### 7.4 Roofline 例 1：一百万个 BF16 ReLU

**【课程原例】**令 $`n=1{,}048{,}576=2^{20}`$。ReLU 对每项做 `max(x,0)`，课程粗算 1 FLOP/element。

数据量：

- 读 $`n`$ 个 BF16：$`2n`$ bytes；
- 写 $`n`$ 个 BF16：$`2n`$ bytes；
- 总 $`Q=4n`$ bytes。

逐位代入：

```math
Q=4\times1{,}048{,}576=4{,}194{,}304\text{ bytes}=4\text{ MiB}
```

```math
F=n=1{,}048{,}576\text{ FLOPs}
```

```math
I=\frac{1{,}048{,}576}{4{,}194{,}304}=0.25\text{ FLOP/byte}
```

计算时间下界：

```math
t_{\text{compute}}=\frac{1.048576\times10^6}{989.5\times10^{12}}
=1.06\times10^{-9}\text{ s}
```

搬运时间下界：

```math
t_{\text{memory}}=\frac{4.194304\times10^6}{3.35\times10^{12}}
=1.252\times10^{-6}\text{ s}
```

搬运约比计算慢：

```math
\frac{1.252\times10^{-6}}{1.06\times10^{-9}}\approx1181
```

所以 ReLU 强烈 memory-bound。把 1 FLOP 改快一点几乎没用；把 ReLU 融合进前后 kernel、避免额外 HBM 往返更重要。

**【视频补充】**GELU 可粗算约 20 FLOPs/element，同样读写 4 bytes/element：

```math
I_{\text{GELU}}=20/4=5\text{ FLOP/byte}<295.37
```

它仍可能是 memory-bound，所以孤立 GELU 不一定比 ReLU 慢 20 倍。

### 7.5 Roofline 例 2：长度 1024 的 dot product

**【课程】**两个 BF16 向量 $`x,y`$，长度 $`n=1024`$，输出一个 BF16 数。

FLOPs：$`n`$ 次乘、$`n-1`$ 次加：

```math
F=2n-1=2047\text{ FLOPs}
```

bytes：读两个向量、写一个输出：

```math
Q=2n+2n+2=4n+2=4098\text{ bytes}
```

```math
I=2047/4098=0.4995\text{ FLOP/byte}
```

明显小于 295.37，因此 memory-bound。这里忽略 reduction 的中间同步和累加精度，实际只会更复杂。

### 7.6 Roofline 例 3：BF16 方阵乘

**【课程】**$`X,W,Y`$ 都是 `(n,n)`，理想假设每个输入从 HBM 只读一次、输出只写一次。

FLOPs：

```math
F=n^2(2n-1)
```

bytes：两个输入和一个输出，各 $`n^2`$ 个 BF16：

```math
Q=3\times n^2\times2=6n^2\text{ bytes}
```

算术强度：

```math
I=\frac{n^2(2n-1)}{6n^2}=\frac{2n-1}{6}
```

当 $`n=1024`$：

```math
I=\frac{2048-1}{6}=\frac{2047}{6}=341.17\text{ FLOP/byte}
```

因为 $`341.17>295.37`$，在这套理想模型与硬件上刚刚进入 compute-bound。

把工作量也展开：

```math
F=1024^2\times2047
=1{,}048{,}576\times2047
=2{,}146{,}435{,}072\text{ FLOPs}
```

```math
Q=6\times1{,}048{,}576=6{,}291{,}456\text{ bytes}=6\text{ MiB}
```

### 7.7 Roofline 例 4：为什么单 token 推理常变成 matvec

**【课程+补充推导】**若只有一个输入向量 `(1,n)` 乘权重 `(n,n)`：

- FLOPs：$`F=n(2n-1)`$；
- bytes：读输入 $`2n`$，读权重 $`2n^2`$，写输出 $`2n`$，故 $`Q=2n^2+4n`$。

$`n=1024`$：

```math
F=1024\times2047=2{,}096{,}128\text{ FLOPs}
```

```math
Q=2\times1024^2+4\times1024
=2{,}101{,}248\text{ bytes}
```

```math
I=2{,}096{,}128/2{,}101{,}248=0.9976\text{ FLOP/byte}
```

远小于 295.37。训练时很多 token 一起复用权重，像大 matmul；自回归解码每次只有很少 token，像 matvec，同一份权重搬进来只做很少计算，因此常受带宽限制。

### 7.8 Roofline 的假设不能忘

**【补充解释】**上面的线只是上界，不是实测承诺：

- 假设输入只从 HBM 读一次；tile 不好或 cache 不够会重复读；
- 假设计算与传输完美重叠；
- 忽略 kernel launch、同步、通信、索引和控制流；
- 峰值要求适合 Tensor Core 的 dtype/shape；
- $`Q`$ 是否包含临时输出取决于是否 kernel fusion；
- 小矩阵即便理论 compute-bound，也可能因并行度不足达不到峰值。

因此课程总结“大 matmul compute-bound，其他常 memory-bound”要带条件：是这台机器、这种 dtype、足够大且实现良好的算子，不是宇宙定律。

---

## 8. 从 tensor 拼出一个深网络

### 8.1 课程网络及每个 shape

**【课程】**设 hidden width `D=8`、层数 `L=3`、batch `B=4`：

```python
class Block(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(D, D))

    def forward(self, x):
        return torch.relu(x @ self.weight)

class Model(nn.Module):
    def __init__(self, D, L):
        super().__init__()
        self.layers = nn.ModuleList([Block(D) for _ in range(L)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

逐行翻成人话：

- `Block(nn.Module)`：定义一个可被 PyTorch 管理的网络块；
- `super().__init__()`：初始化父类内部的模块/参数登记机制；
- `torch.randn(D,D)`：创建 shape `(8,8)` 的随机权重；
- `nn.Parameter(...)`：告诉模块“这个 tensor 要训练”，默认 `requires_grad=True`；
- 输入 `x` shape `(B,D)=(4,8)`；权重 `(8,8)`；矩阵乘输出 `(4,8)`；ReLU 不改 shape；
- list comprehension `[Block(D) for _ in range(L)]` 创建 3 个独立 Block；`_` 表示循环编号本身不用；
- `ModuleList` 让 PyTorch 找到其中所有子模块参数；普通 Python list 可能不会被正确注册；
- forward 循环依次把上一层 `(4,8)` 输出交给下一层，三层后仍是 `(4,8)`。

参数数：每层 $`D\times D=8\times8=64`$，3 层：

```math
N=D^2L=8^2\times3=64\times3=192
```

此代码没有 bias，所以不要额外加 $`D`$。

---

## 9. Autograd：让 PyTorch 沿计算图自动求导

### 9.1 先把六个术语说成人话

**【补充解释】**：

- **autograd（automatic differentiation，自动微分）**：程序记录你做过哪些可微运算，再用链式法则反向算梯度；
- **computation graph（计算图）**：节点是 tensor/运算，箭头表示“这个结果依赖谁”；PyTorch eager 模式每次 forward 动态重建图；
- **leaf tensor（叶子 tensor）**：不是由另一个需要求导的运算产生的起点 tensor。模型 `Parameter` 通常是 leaf；
- **`requires_grad=True`**：告诉 PyTorch，要记录涉及这个 tensor 的运算，以便最终求导；
- **`backward()`**：从标量输出（通常是 loss）按链式法则反向传播；
- **gradient（梯度）**：loss 对某个数的偏导数，表示这个数稍微增加时 loss 首先向哪边、变化多快。

默认只有需要梯度的 **leaf tensor** 在 `.grad` 中累积梯度。中间 non-leaf tensor 为了反向会参与计算，但默认不保留 `.grad`；若调试需要可对它调用 `.retain_grad()`。

### 9.2 只会四则运算也能理解的“导数桥”

**【补充解释】**符号 $`\partial`$ 读作“偏”。$`\frac{\partial f}{\partial a}`$ 叫 $`f`$ 对 $`a`$ 的偏导数：**只让 $`a`$ 增加一点点，把另一个变量 $`b`$ 固定不动，观察 $`f`$ 每增加 1 单位 $`a`$ 大约改变多少。**

如果把“小一点点”取为 $`\Delta a=0.001`$，有限差分（finite difference）就是：

```math
\frac{f(a+0.001,b)-f(a,b)}{0.001}
```

间隔越接近 0，它越接近瞬时变化率，也就是导数。本题只需三条规则，而且都能由小增量推出：

1. **自己的变化率是 1：**$`a`$ 增加 $`\Delta a`$，表达式 $`a`$ 也增加 $`\Delta a`$，所以 $`\partial a/\partial a=1`$；
2. **乘固定数：**固定 $`b`$ 时，$`ab`$ 从 $`ab`$ 变成 $`(a+\Delta a)b=ab+b\Delta a`$，所以每 1 单位 $`a`$ 带来 $`b`$ 单位变化，即 $`\partial(ab)/\partial a=b`$；同理固定 $`a`$ 时，$`\partial(ab)/\partial b=a`$；
3. **平方规则：**$`u`$ 增加 $`\Delta u`$ 时，

```math
(u+\Delta u)^2-u^2=2u\Delta u+(\Delta u)^2
```

两边除以 $`\Delta u`$ 得 $`2u+\Delta u`$；当 $`\Delta u`$ 趋近 0，剩下 $`2u`$，所以 $`\partial(u^2)/\partial u=2u`$。这也是幂规则在平方情形的来源。

为什么计算图上“沿链相乘”？如果 $`a`$ 先改变 $`u`$，$`u`$ 再改变 $`f`$，对很小变化有：

```math
\Delta u\approx\frac{\partial u}{\partial a}\Delta a
```

```math
\Delta f\approx\frac{\partial f}{\partial u}\Delta u
```

把第一行代入第二行：

```math
\Delta f\approx
\frac{\partial f}{\partial u}
\frac{\partial u}{\partial a}\Delta a
```

再除以 $`\Delta a`$，就得到链式法则：

```math
\frac{\partial f}{\partial a}
=\frac{\partial f}{\partial u}
\frac{\partial u}{\partial a}
```

直觉例子：如果 $`a`$ 每增加 1，$`u`$ 约增加 4；而 $`u`$ 每增加 1，$`f`$ 约增加 16；那么 $`a`$ 每增加 1，$`f`$ 就约增加 $`4\times16=64`$。

### 9.3 两变量函数，先手算再交给 PyTorch

**【补充例子】**定义：

```math
u=ab+a,\qquad f=u^2
```

取 $`a=2,b=3`$。forward：

```math
u=2\times3+2=8
```

```math
f=8^2=64
```

反向用链式法则。先求最靠近输出的局部导数：

```math
\frac{\partial f}{\partial u}=2u=16
```

再求 $`u`$ 对两个输入：

```math
\frac{\partial u}{\partial a}=b+1=3+1=4
```

为什么有 `+1`？因为 $`u=ab+a`$ 中，$`ab`$ 对 $`a`$ 的导数是 $`b`$，第二个 $`a`$ 对自己的导数是 1。

```math
\frac{\partial u}{\partial b}=a=2
```

沿路径相乘：

```math
\frac{\partial f}{\partial a}
=\frac{\partial f}{\partial u}\frac{\partial u}{\partial a}
=16\times4=64
```

```math
\frac{\partial f}{\partial b}
=\frac{\partial f}{\partial u}\frac{\partial u}{\partial b}
=16\times2=32
```

现在不用相信符号，用 $`0.001`$ 做两次有限差分验算。

**只改变 $`a`$，固定 $`b=3`$：**

```math
f(2,3)=(2\times3+2)^2=8^2=64
```

```math
f(2.001,3)=(2.001\times3+2.001)^2
=8.004^2=64.064016
```

```math
\frac{f(2.001,3)-f(2,3)}{0.001}
=\frac{64.064016-64}{0.001}=64.016
```

这非常接近解析结果 64；多出的 0.016 来自步长还不是 0。

**只改变 $`b`$，固定 $`a=2`$：**

```math
f(2,3.001)=(2\times3.001+2)^2
=8.002^2=64.032004
```

```math
\frac{f(2,3.001)-f(2,3)}{0.001}
=\frac{64.032004-64}{0.001}=32.004
```

它非常接近解析结果 32。现在 $`\partial`$、固定另一个变量、局部变化率和沿链相乘都落到了四则运算上。

对应 PyTorch：

```python
a = torch.tensor(2.0, requires_grad=True) # leaf，a.grad 起初是 None
b = torch.tensor(3.0, requires_grad=True) # leaf，b.grad 起初是 None
u = a * b + a                            # non-leaf，值 8，有 grad_fn
f = u ** 2                               # non-leaf scalar，值 64
f.backward()                             # 从 df/df=1 开始反向
print(a.grad)                            # tensor(64.)
print(b.grad)                            # tensor(32.)
```

每行状态：创建两个需要梯度的起点；算 `u` 并记录乘、加；算 `f` 并记录平方；`backward` 反向遍历；结果**累加**到两个 leaf 的 `.grad`。

若不清零又重新做一次 forward/backward，`.grad` 会从 `(64,32)` 变成 `(128,64)`。这是设计行为，方便梯度累积，不是 bug。

### 9.4 课程的最小回归例子

**【课程】**：

```python
x = torch.tensor([1., 2., 3.])
w = torch.tensor([1., 1., 1.], requires_grad=True)
target = torch.tensor(5.)
pred = x @ w
loss = 0.5 * (pred - target) ** 2
loss.backward()
```

逐步手算：

```math
\text{pred}=1\times1+2\times1+3\times1=6
```

```math
\text{loss}=\frac12(6-5)^2=\frac12
```

对第 $`i`$ 个权重：

```math
\frac{\partial\text{loss}}{\partial w_i}
=(\text{pred}-\text{target})x_i
```

因为误差 $`6-5=1`$：

```math
w.\text{grad}=1\times[1,2,3]=[1,2,3]
```

### 9.5 `zero_grad`、`no_grad`、`detach` 不一样

**【补充解释】**：

- `optimizer.zero_grad()`：把优化器管理参数的旧 `.grad` 清掉；现代 PyTorch 默认常设为 `None`，不是填一块全零 tensor；
- `with torch.no_grad():`：块内运算不建反向图，适合参数更新或验证；
- `x.detach()`：返回一个与 `x` 共享数据但切断 autograd 历史的 tensor；
- `model.eval()`：改变 dropout/batchnorm 行为，**不会自动关闭梯度**；验证常同时用 `model.eval()` 和 `torch.no_grad()`。

官方依据：[Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)、[`zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)。

---

## 10. 为什么反向传播大约是前向的两倍

### 10.1 单层矩阵乘的三次大矩阵乘

**【课程】**一层：

```math
H_2=H_1W_2
```

shape：

- $`H_1`$：`(B,D)`；
- $`W_2`$：`(D,D)`；
- $`H_2`$：`(B,D)`。

forward 矩阵乘：

```math
F_{\text{forward}}\approx2BDD=2BD^2
```

backward 要算两样：

1. 给上一层的梯度：

```math
\frac{\partial L}{\partial H_1}
=\frac{\partial L}{\partial H_2}W_2^T
```

这是 `(B,D)@(D,D)->(B,D)`，约 $`2BD^2`$ FLOPs。

2. 权重梯度：

```math
\frac{\partial L}{\partial W_2}
=H_1^T\frac{\partial L}{\partial H_2}
```

这是 `(D,B)@(B,D)->(D,D)`，也约 $`2BD^2`$ FLOPs。

所以：

```math
F_{\text{backward}}\approx2BD^2+2BD^2=4BD^2
```

```math
F_{\text{forward+backward}}\approx6BD^2
```

反向约为前向的 $`4/2=2`$ 倍；整个训练步约为只做 forward 的 $`6/2=3`$ 倍。

### 10.2 从一层推到 $`6N D_{\text{tok}}`$

**【课程+补充解释】**对 dense Transformer，把每个大权重矩阵的元素总数粗略合成 $`N`$ 个参数。对一个 token：

- forward 每个参数大致参与一次 multiply-add，约 2 FLOPs/parameter/token；
- backward 计算输入梯度约 2；
- backward 计算权重梯度约 2；
- 合计约 6 FLOPs/parameter/token。

训练 $`D_{\text{tok}}`$ 个 token：

```math
C_{\text{train}}\approx6N D_{\text{tok}}
```

这里刻意写 $`D_{\text{tok}}`$，避免与本讲矩阵 hidden width $`D`$ 混淆。

公式成立的主要条件：

- dense 模型，每 token 大致激活全部 $`N`$ 个参数；
- 大矩阵乘占主要计算；
- 训练包括 forward 和 backward；
- optimizer update 相对矩阵乘计算较小。

它粗略忽略或合并了 attention 的序列长度平方项、embedding/loss、归一化、激活函数、optimizer、通信、重算和数据移动。长 context 下 attention $`T^2`$ 项可能显著；小模型或特殊算子中参数矩阵乘不主导时会失真。MoE（mixture of experts）应使用**每 token 实际激活的参数量**，不是所有 expert 的总参数量。它不是推理公式；推理没有权重梯度那两次矩阵乘。

---

## 11. Activation、gradient 与 optimizer state 为什么吃显存

### 11.1 四类东西各是什么

**【课程+补充解释】**训练时至少有：

1. **parameter**：要学习的权重；生命周期通常贯穿训练；
2. **gradient**：每个参数的 loss 导数；backward 产生，optimizer step 消费；
3. **optimizer state**：优化器跨 step 保存的历史统计。AdaGrad 每参数一个累计平方；Adam 通常两个 moment；
4. **activation（激活/中间结果）**：forward 中每层产生的 tensor。backward 求局部导数常要它们，因此不能在 forward 后立刻全删。

此外还有 input、label、loss、临时 workspace、allocator 保留块、通信 buffer、CUDA context 等。因此“参数账”不等于“峰值显存账”。

### 11.2 课程微型网络：AdaGrad 与 Adam 全账

**【课程原例】**`B=2,D=4,L=3`，每层无 bias 的 `(4,4)` 权重。

参数数：

```math
N=D^2L=4^2\times3=16\times3=48
```

课程 mixed-precision 简化：BF16 参数和梯度，FP32 optimizer state。

参数：

```math
M_p=48\times2=96\text{ bytes}
```

梯度：

```math
M_g=48\times2=96\text{ bytes}
```

AdaGrad 一个 FP32 state：

```math
M_{\text{Ada}}=48\times4=192\text{ bytes}
```

课程为教学简化，按每层保存一个 `(B,D)` BF16 activation：

```math
M_a=B\times D\times L\times2
=2\times4\times3\times2=48\text{ bytes}
```

总计：

```math
M_{\text{total,Ada}}=96+96+192+48=432\text{ bytes}
```

若换 Adam 两个 FP32 state：

```math
M_{\text{Adam state}}=48\times(4+4)=384\text{ bytes}
```

```math
M_{\text{total,Adam}}=96+96+384+48=624\text{ bytes}
```

这是“端到端组成项”示范，但没有计 allocator、输入、ReLU mask、临时矩阵乘 workspace 等，不能当 PyTorch 实测峰值。

### 11.3 补充端到端例子：1B 参数为何一张 80 GB 卡也未必宽松

**【补充例子】**假设：

- $`N=1\times10^9`$ 参数；
- BF16 参数 2 B、BF16 梯度 2 B；
- Adam 两个 FP32 state 共 8 B；
- 另存 FP32 master parameter 4 B；
- `batch=8, sequence=2048, hidden=4096, layers=24`；
- 教学估算每层 backward 需保存 8 个 BF16、shape `(B,S,H)` 等价大小的 activation。真实模型保存项取决于实现。

模型状态每参数：

```math
2+2+8+4=16\text{ bytes/parameter}
```

```math
M_{\text{model state}}=10^9\times16=16\times10^9\text{ bytes}=16\text{ GB}
```

先算一份 `(B,S,H)` activation 元素数：

```math
8\times2048\times4096
=8\times8{,}388{,}608
=67{,}108{,}864\text{ elements}
```

24 层、每层 8 份：

```math
67{,}108{,}864\times24\times8
=12{,}884{,}901{,}888\text{ elements}
```

BF16 字节：

```math
M_a=12{,}884{,}901{,}888\times2
=25{,}769{,}803{,}776\text{ bytes}=24\text{ GiB}
```

粗略合计（统一用十进制 GB）：

```math
16\text{ GB}+25.77\text{ GB}=41.77\text{ GB}
```

还没有计算临时 workspace、attention score、logits、通信和框架开销。例子显示：即使参数本体只有 2 GB，训练峰值也可能几十 GB。若 microbatch 从 8 降到 2，在其他假设不变时 activation 线性降为 $`24/4=6`$ GiB，但 16 GB 模型状态不变。

### 11.4 哪些显存随什么缩放

| 项目 | 主要随谁增长 | 降 microbatch 是否降低 |
|---|---|---:|
| 参数 | 参数量 $`N`$ | 否 |
| 梯度 | 参数量 $`N`$ | 否 |
| optimizer state | 参数量 $`N`$ | 否 |
| activation | batch × sequence × hidden × layers × 保存份数 | 是 |
| attention score（朴素） | batch × heads × sequence² × layers | 是，且强烈受序列长度影响 |

这张表解释了：梯度累积主要救 activation OOM，不会让一个连参数/状态都放不下的模型突然放下。

---

## 12. Optimizer 与训练循环：每行之后机器里发生了什么

### 12.1 AdaGrad 的状态与更新

**【课程】**对每个参数 $`p`$ 和当前梯度 $`g`$，AdaGrad 保存累计平方 $`G`$：

```math
G\leftarrow G+g^2
```

```math
p\leftarrow p-\eta\frac{g}{\sqrt{G}+\epsilon}
```

- $`G`$：optimizer state，shape 与参数相同；
- $`g`$：当前累计梯度；
- $`\eta`$：learning rate（学习率）；
- $`\epsilon`$：防止分母为 0 的小数，例如 $`10^{-5}`$。

课程简化实现：

```python
class AdaGrad:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr
        self.state = [torch.zeros_like(p, dtype=torch.float32)
                      for p in self.params]

    def step(self):
        with torch.no_grad():
            for p, g2 in zip(self.params, self.state):
                g2 += p.grad.float() ** 2
                p -= self.lr * p.grad / (torch.sqrt(g2) + 1e-5)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
```

逐行翻译：

- `list(params)` 固定保存所有参数引用；若保留一次性 generator 而重复遍历，可能第二次为空；
- `zeros_like(p,dtype=float32)` 给每个参数建同 shape 的 FP32 `G`，初始全 0；
- `zip` 把第一个参数与第一个状态、第二个与第二个配对；
- `no_grad` 防止“更新参数”本身被记录成下一张计算图；
- `p.grad.float()**2` 把梯度转 FP32、逐元素平方并累加；
- 除以历史平方根让长期大梯度方向的有效步长变小；
- `p -= ...` 原地修改 parameter；
- `p.grad=None` 删除旧梯度标记，让下次 backward 从空开始。

课程代码是教学版；生产代码还要处理参数组、无梯度参数、稀疏梯度、device、state dict 等。

### 12.2 标准训练循环逐行状态表

**【课程+补充解释】**下面是可运行的最小例子；它承接第 8 节的 `Model` 类和页首 imports。这里补出课程中用于示意的 `get_batch`：它随机生成输入，并把 target 暂定为全 0；真实训练会从数据集读取。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(batch_size, hidden, device):
    x = torch.randn(batch_size, hidden, device=device)
    target = torch.zeros_like(x)
    return x, target

model = Model(D=16, L=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(3):
    optimizer.zero_grad()
    x, target = get_batch(batch_size=4, hidden=16, device=device)
    pred = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optimizer.step()
```

| 行执行后 | 关键状态变化 |
|---|---|
| 创建 `model` | 参数在 `device` 上，shape 每层 `(16,16)`，两层共 $`16²×2=512`$ 参数 |
| 创建 AdamW | optimizer 持有参数引用；moment 通常在第一次 step 时惰性创建 |
| `zero_grad()` | 上一步 `.grad` 设为 `None`/清零；参数值不变，moment 不变 |
| `get_batch` | 新建 `x,target`，此处 shape 都是 `(4,16)` |
| `pred=model(x)` | forward 两层；`(4,16)@(16,16)→(4,16)` 两次；建立动态图并保存所需 activation |
| 算 `loss` | 先逐元素差 `(4,16)`，平方 `(4,16)`，`mean` 后变 scalar shape `()` |
| `loss.backward()` | 从 scalar 1 开始，生成/累积每个 parameter 的 `.grad`；参数还没更新 |
| `optimizer.step()` | 读取 gradient 和旧 moment，更新 moment，再修改 parameter；gradient 通常仍在，等下轮清除 |

为什么 loss 通常要是 scalar？`backward()` 需要一个反向起点；scalar 默认导数是 1。对 non-scalar 也能 backward，但必须提供匹配的上游 gradient，本讲不展开。

### 12.3 最常见的循环错误

1. 忘记 `zero_grad`：每步梯度叠加，等效 batch/学习率悄悄改变，最终可能爆炸；
2. 把 `zero_grad` 放在 `backward` 与 `step` 之间：刚算出的梯度被清掉，optimizer 无从更新；
3. 在普通 grad mode 下原地更新 leaf parameter：既可能报错，也会把更新写入图；
4. 只 `model.eval()` 却不 `no_grad()`：仍建图、保存 activation，验证可能 OOM；
5. 把整个 `loss` tensor 存入 Python list 而不 `.item()`/`detach()`：可能把整张旧图一起留住，显存逐步增长。

---

## 13. OOM 之后的第一招：gradient accumulation

### 13.1 它保持“大 batch 梯度”，一次只放小 batch activation

**【课程】**假设目标 batch $`B=64`$，拆成 $`K=4`$ 个 microbatch，每个：

```math
B_{\text{micro}}=64/4=16
```

课程示例 `D=1024,L=16`，每层一份 BF16 `(B,D)` activation。完整 batch：

```math
M_a=2\times B\times D\times L
```

```math
=2\times64\times1024\times16
=2{,}097{,}152\text{ bytes}=2\text{ MiB}
```

一个 microbatch：

```math
M_{a,\text{micro}}
=2\times16\times1024\times16
=524{,}288\text{ bytes}=0.5\text{ MiB}
```

因为每个 microbatch backward 后，其中间 activation 可释放，而 parameter `.grad` 保留并累加，峰值 activation 约降 4 倍。

### 13.2 正确代码与 loss scaling

下面代码可运行，但要先执行第 12.2 节的 `model`、`optimizer` 和 `get_batch` 定义。`four_microbatches` 是四对 `(micro_x,micro_y)`；`loss_fn` 是返回每个 microbatch 平均平方误差的函数：

```python
full_x, full_y = get_batch(batch_size=64, hidden=16, device=device)
four_microbatches = list(zip(full_x.chunk(4), full_y.chunk(4)))
loss_fn = nn.MSELoss(reduction="mean")

optimizer.zero_grad()                  # 只在整组开始前清一次
for micro_x, micro_y in four_microbatches:
    pred = model(micro_x)              # 只保存这一小批 activation
    loss = loss_fn(pred, micro_y) / 4  # 四个等大 microbatch，先除 K
    loss.backward()                    # 梯度加到已有 .grad
optimizer.step()                       # 累够四份，只更新一次
```

为什么除以 4？若每个 `loss_fn` 返回该 microbatch 的 mean，直接相加得到四个 mean 的和，是目标大 batch mean 的 4 倍。除以 $`K=4`$ 后：

```math
g=\frac{g_1+g_2+g_3+g_4}{4}
```

才等于四个等大 microbatch 合成的大 batch 平均梯度。若 microbatch 大小不等，应按样本/token 数加权，不能简单都除 $`K`$。

### 13.3 梯度累积的边界与视频勘误

- 它降低 activation 峰值，不降低参数、梯度 tensor、optimizer state；
- 总矩阵运算量大致不变，小 kernel 次数更多，可能更慢；
- batchnorm、dropout 随机性和按 microbatch 计算的 loss 可能使结果不与真大 batch 完全相同；
- 分布式训练还需避免每个 microbatch 都做不必要的梯度通信。

**【视频勘误】**约 73:16 的一句口头表述像是“save on compute”；结合官方代码、显存公式和上下文，gradient accumulation 节省的是**峰值 activation memory**，并不节省必要的模型计算，通常还有额外 overhead。

---

## 14. 第二招：activation checkpointing

### 14.1 用“以后重算”换“现在不存”

**【课程】**正常 forward 保存每层 backward 所需 activation，显存随层数 $`L`$ 约为 $`O(L)`$。checkpointing 只保存少数边界；backward 需要内部 activation 时，再重跑对应 forward 段。

下面是可运行的**单层 API 示意**；它承接第 8 节定义的 `Block` 类以及页首 imports：

```python
layer = Block(D=8)
x = torch.randn(2, 8, requires_grad=True) # shape (B=2,D=8)
y = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
assert y.shape == (2, 8)
y.sum().backward()
```

逐项翻译：不要长期保存 `layer(x)` 内部所有中间值；保留恢复所需的输入/边界；反向时重新调用 `layer`。输出 shape 不变，只改变“存还是重算”。

**【视频补充】**课程 toy block 是 matrix multiply 后 ReLU；checkpoint 整块时无需同时长期保存块内所有中间值，直觉上可省掉其中一部分。本例的“约一半”不能推广成所有模型恒定比例。

但是，`checkpoint(layer, x)` 只说明“一个 layer 可以被 checkpoint”。如果把每一层各包一次，每层输入仍是一个边界，跨层保存的边界仍可能有 $`L`$ 个；它主要省**层内部**的中间量。要得到下面的 $`O(\sqrt L)`$ 结论，必须把连续多层组成一个 segment（分段），再以 segment 为单位 checkpoint。**逐层 checkpoint 与每 $`\sqrt L`$ 层一个 segment 不是一回事。**

### 14.2 为什么最佳间隔约为 $`\sqrt L`$

**【课程+补充推导】**先声明这个教学模型的条件：

- 网络是无分支的顺序链，共 $`L`$ 层；
- 各层 activation 大小近似相同，forward 重算成本也近似相同；
- 只统计 activation，不统计参数、梯度、optimizer state 和 allocator；
- 每连续 $`k`$ 层组成一个 segment，在 segment 边界保存 checkpoint；为简化先假设 $`k`$ 整除 $`L`$。

如果每个 segment 有 $`k`$ 层：

1. segment 数是 $`L/k`$，所以长期保存的边界 activation 约有 $`L/k`$ 份；
2. backward 处理当前 segment 时，要重算并暂时保留段内最多约 $`k`$ 层 activation；
3. 因而峰值 activation 的“层数因子”近似：

```math
M_{\text{factor}}(k)\approx\frac{L}{k}+k
```

第一项随 $`k`$ 增大而下降，第二项随 $`k`$ 增大而上升。最平衡的位置让两项相等：

```math
\frac{L}{k}=k
```

两边乘 $`k`$：

```math
L=k^2
```

所以：

```math
k\approx\sqrt L
```

也可以用 AM-GM（算术—几何平均不等式）确认：

```math
\frac{L}{k}+k\ge2\sqrt{\frac{L}{k}\times k}=2\sqrt L
```

等号恰好在 $`L/k=k`$ 时成立。因此最优量级不是背出来的，而是“边界数”和“当前段长度”两种显存相互平衡。

三种方案现在可以精确区分：

1. 全保存：activation memory $`O(L)`$，不额外重算；
2. 只保存最初输入：为反向第 $`i`$ 层可能从开头重跑到 $`i`$，总重算 $`1+2+\cdots+L=O(L^2)`$，保存量 $`O(1)`$；
3. 每约 $`k=\sqrt L`$ 层组成一个 checkpointed segment：边界 $`L/k\approx\sqrt L`$ 份，当前段最多 $`k\approx\sqrt L`$ 份，峰值 activation 为 $`O(\sqrt L)`$；每段在 backward 中大致重跑一次，额外重算总量为 $`O(L)`$。

例：$`L=16`$，每 4 层一个 checkpoint：

- 边界大约 4 个；
- 反向某段时，最多临时重建约 4 层；
- 峰值的层数因子从 16 级别降到约 $`4+4=8`$ 级别；
- 每段 forward 在 backward 时至多再跑一次，额外工作是线性量级，不是 $`16^2`$。

这是在上述顺序链、等 activation、等重算成本条件下的渐近直觉。真实 Transformer 各算子保存量不同，PyTorch 实际保存哪些 tensor、能否 early-stop 重算也取决于实现；此时应按每个 segment 的真实 bytes 与时间重新优化，而不是机械地取 $`\sqrt L`$。

### 14.3 随机性和副作用陷阱

**【补充解释】**重算要求函数的重新执行与原 forward 兼容。Dropout 使用随机数；PyTorch 默认会保存/恢复相关 RNG state 来保持结果一致，这也有开销。若函数修改全局状态、在重算时走不同控制流，梯度可能错误或报错。官方细节见 [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint)。

### 14.4 两招怎样组合

```text
gradient accumulation：减小一次进入模型的 microbatch
checkpointing：减小每个 microbatch 每层要长期保存的中间结果
```

二者都主要动 activation，可同时使用；若 OOM 来自参数/optimizer state，则需参数分片、optimizer state 分片、offload、低精度状态或减少模型规模，这些在后续课程展开。

---

## 15. 把整讲串起来：遇到性能或 OOM 时怎么想

**【补充解释】**按这个顺序排查，不要看到慢就只盯 FLOPs：

1. **先写 shape。**每个输入、权重、输出逐 axis 标出长度与语义；
2. **再写 dtype/device。**每元素多少 bytes，tensor 是否在同一设备；
3. **数常驻显存。**参数、梯度、optimizer state；
4. **数随 batch/sequence 变化的显存。**activation、attention 中间量、workspace；
5. **数 FLOPs。**从一个输出元素的乘法和加法开始，不背空公式；
6. **数最少 bytes。**读几个输入、写几个输出，有没有中间结果落 HBM；
7. **算 $`I=F/Q`$。**与 $`P_{\text{peak}}/B_{\text{mem}}`$ 比；
8. **再决定优化。**compute-bound 优化计算/kernel；memory-bound 减搬运、做 fusion；activation OOM 用累积/checkpoint；模型状态 OOM 要分片或改变精度/规模；
9. **最后实测。**同步、warmup、多次重复，用 profiler 查模型遗漏。

---

## 16. 常见误区

1. **“rank 2 就是矩阵的秩为 2。”**错。本讲 rank 是 axis 数；线性代数 rank 是独立行/列数。
2. **“shape `(2,3)` 有 5 个元素。”**错。元素数是 $`2\times3=6`$，不是相加。
3. **“BF16 和 FP16 都是 2 bytes，所以完全一样。”**错。exponent/fraction 分配不同，范围与精度不同。
4. **“把 tensor `.to('cuda')` 不花时间。”**错。跨设备通常要复制数据。
5. **“transpose 会把数据按新顺序复制。”**通常错。它常只改 stride、返回共享 storage 的 view。
6. **“reshape 就是 transpose。”**错。reshape 改分组，transpose 换 axis；可能得到相同 shape、不同值。
7. **“reshape 一定零拷贝。”**错。兼容时是 view，不兼容时可以 copy。
8. **“contiguous 是数值连续，没有 NaN。”**错。它描述内存布局。
9. **“einsum 中重复出现的 axis 都被求和。”**不完整。只有未出现在输出 pattern 的名字才被消掉。
10. **“rearrange 可以把 axis 丢掉。”**错。它保留元素数；要汇总用 reduce。
11. **“repeat 只改 metadata。”**通常错。逻辑元素变多，实际使用时需要物化或广播读取。
12. **“一个 FMA 指令只算 1 FLOP。”**性能规格约定通常把一次乘和一次加算 2 FLOPs。
13. **“TFLOP 是速度。”**错。TFLOP 是工作量；TFLOP/s 才是速度。
14. **“规格表 1979 TFLOP/s 就是 dense BF16 峰值。”**按课程所用 H100 表格，1979 是带 structured sparsity 的数字，dense 口径除 2。
15. **“MFU 低一定是代码写坏了。”**不一定。算子可能天生 memory-bound，模型也可能受通信限制。
16. **“GELU 约 20 FLOPs，所以一定比 ReLU 慢 20 倍。”**错。两者单独运行都可能等待内存。
17. **“所有 matmul 都 compute-bound。”**错。矩阵小、batch 小或退化成 matvec 时算术强度低。
18. **“Roofline 预测的就是实际时间。”**错。它是理想上界/下界模型，现实 overhead 会更慢。
19. **“参数本体能放进显存，模型就能训练。”**错。还要梯度、optimizer state、activation 等。
20. **“BF16 训练每参数固定只要 2 bytes。”**错。那只是参数本体；课程 AdamW 简化账是 12 bytes/param，某些实现含 master weight 会到 16。
21. **“backward 会覆盖旧 gradient。”**错。PyTorch 默认累加到 leaf `.grad`。
22. **“`model.eval()` 等于关掉 autograd。”**错。它主要改变 dropout/batchnorm；还要 `no_grad`/inference mode。
23. **“梯度累积同时省参数、状态和 activation。”**错。主要省 activation 峰值。
24. **“累积四次却不除 loss，仍是同一个大 batch mean。”**错。等大 microbatch 时梯度放大 4 倍。
25. **“checkpointing 让计算更少。”**错。它通过重算增加计算，换更低 activation memory。
26. **“$`6N D_{\text{tok}}`$ 对所有模型、训练和推理都准确。”**错。它是 dense、矩阵乘主导训练的粗估；MoE、长 context、推理等要改账。

---

## 17. 术语表

| 术语 | 一句话解释 |
|---|---|
| tensor | 带 shape、dtype、device、stride 等元数据的多维数字容器 |
| scalar / vector / matrix | rank 0 / rank 1 / rank 2 的常见 tensor |
| shape | 每条 axis 的长度组成的 tuple |
| rank | axis 的数量，不是线性代数矩阵秩 |
| axis / dimension | tensor 中一条可独立编号的方向 |
| dtype | 每个元素的编码格式，决定字节数、范围和精度 |
| device | tensor storage 所在计算设备，如 CPU 或 CUDA GPU |
| storage | 底层一维连续字节区，可能被多个 view 共用 |
| stride | 沿某 axis 前进一步时在 storage 跳几个元素 |
| view | 与原 tensor 共享 storage、使用不同 shape/stride 的解释 |
| copy | 复制到新的 storage |
| contiguous | 默认布局下，逻辑相邻关系按规则连续落在 storage 中 |
| parameter | 训练要学习和更新的 tensor 元素 |
| activation | forward 中产生、backward 可能需要的中间 tensor |
| gradient | loss 对参数/输入的偏导数 |
| optimizer state | 优化器跨 step 保留的统计，如 Adam 的一阶、二阶矩 |
| mixed precision | 不同 tensor/算子选择不同数值精度 |
| exponent / fraction | 浮点格式中主要控制范围 / 有效数字精度的字段 |
| E4M3 / E5M2 | 两种 FP8：分别使用 4/3 或 5/2 个 exponent/fraction bits |
| underflow / overflow | 数太小变成 0 / 数太大超出范围 |
| Tensor Core | GPU 中专门执行小块矩阵乘加的硬件单元 |
| dense / structured sparse | 不假定可跳零的稠密矩阵 / 满足硬件规定零模式的结构化稀疏矩阵 |
| einsum | 用有名 axis 表达乘、对齐和求和 |
| rearrange | 重排、拆分、合并 axis，不做 reduction |
| reduce | 用 sum/mean/max 等汇总并消除 axis |
| repeat | 新建或扩张 axis，重复元素 |
| FLOP | 一次浮点运算；是工作量单位 |
| FLOP/s | 每秒浮点运算数；是吞吐单位 |
| bandwidth | 每秒能搬运多少 byte |
| HBM / SRAM / register | 从大而较远到小而较近的 GPU 存储层级；算 $`Q`$ 必须指定边界 |
| arithmetic intensity | 每搬 1 byte 做多少 FLOPs |
| compute-bound | 计算吞吐先成为瓶颈 |
| memory-bound | 内存带宽先成为瓶颈 |
| Roofline | 用峰值计算与带宽共同给性能上界的模型 |
| MFU | 实际有用模型 FLOP/s ÷ 对应硬件峰值 FLOP/s |
| autograd | PyTorch 自动记录计算并反向求导的系统 |
| computation graph | 表达 tensor/运算依赖关系的有向无环图 |
| leaf tensor | 不是由需梯度运算产生的起点 tensor |
| backward | 从输出沿链式法则向输入传播梯度 |
| zero_grad | 清理上一次累积在参数 `.grad` 中的梯度 |
| optimizer / AdamW | 使用梯度更新参数的算法 / 保存两个 moment 的常用 optimizer |
| moment / master weight | optimizer 的梯度历史统计 / 某些低精度方案保存的 FP32 参数副本 |
| gradient accumulation | 多个 microbatch 的梯度累加后只更新一次 |
| activation checkpointing | 少存中间结果，反向时重算 forward 片段 |
| OOM | out of memory，设备无法再满足显存分配 |

---

## 18. 自测题

> 建议先遮住第 19 节。题目刻意覆盖 shape、字节、FLOPs、Roofline、autograd 与训练循环；只会背结论不够。

1. scalar、长度 5 向量、`2×3` 矩阵的 shape 和 rank 各是什么？
2. tensor shape `(2,3,4)` 有多少元素？axis 1 长多少？
3. `torch.zeros(4,8,dtype=torch.float32)` 占多少 bytes？
4. shape `(32,16,16,64)` 有多少元素？若 BF16 占多少 MiB？
5. `(49152,12288)` FP32 权重有多少元素、多少 MiB、多少 GiB？
6. FP16 与 BF16 都为 2 bytes，关键差异是什么？
7. 对连续 `(2,3)` tensor，stride 为什么通常是 `(3,1)`？转置后通常是多少？
8. `view` 与 copy 的核心区别是什么？`reshape` 是否保证 view？
9. `[[1,2,3],[4,5,6]]` 直接 reshape `(3,2)` 与 transpose 的结果各是什么？
10. `einsum("batch i d, batch j d -> batch i j")` 哪条 axis 被求和？输入 `(2,3,4)` 和 `(2,5,4)` 输出 shape 是什么？
11. `rearrange(x,"s (h d)->s h d",h=2)` 输入 `(3,8)` 输出 shape 是什么？`d` 多长？
12. 对 `[[1,2,3],[4,5,6]]`，`reduce("r c->r","sum")` 的值与 shape？
13. 对 `[1,2,3]`，`repeat("w->(tile w)",tile=2)` 与 `repeat("w->(w copy)",copy=2)` 各得什么？
14. 无 bias 线性层权重 `(3,2)` 有几个参数？有 shape `(2,)` bias 后呢？
15. 70B BF16 参数本体多少 GB？为什么这不是训练显存？
16. 按课程 AdamW 12 bytes/parameter，8×80 GB 理论最多多少 B 参数？若加 FP32 master weight 呢？
17. `X(B,D)@W(D,K)` 的一个输出元素为什么是 $`2D-1`$ FLOPs？总 FLOPs？
18. $`B=2,D=3,K=2`$ 时精确 matmul FLOPs 与 $`2BDK`$ 近似各多少？
19. 工作量 $`6.3×10^{24}`$ FLOPs，吞吐 $`5.06624×10^{17}`$ FLOP/s，需要多少天？
20. 实际 500 TFLOP/s、峰值 989.5 TFLOP/s，MFU 是多少？
21. 为什么 GPU 计时在起点和终点都要 synchronize？
22. 峰值 989.5 TFLOP/s、带宽 3.35 TB/s，ridge intensity 是多少？
23. BF16 ReLU 每项读 2 bytes、写 2 bytes、做 1 FLOP，算术强度多少？在上述 H100 上受什么限制？
24. BF16 GELU 粗算 20 FLOPs/element、同样读写，算术强度多少？为什么不会自动 compute-bound？
25. BF16 `1024×1024` 方阵乘的理想算术强度是多少？判断瓶颈。
26. BF16 matvec $`X_{(1,1024)}W_{(1024,1024)}`$，按正文公式计算 $`F,Q,I`$ 并判断瓶颈。
27. 对 $`f=(ab+a)^2`$，在 $`a=2,b=3`$ 时，从平方、乘固定数和链式法则推出 $`f,\partial f/\partial a,\partial f/\partial b`$；再用步长 0.001 的有限差分验算两个偏导。
28. 为什么连续两次 `backward` 而不清梯度会得到两倍 `.grad`？
29. 单层 `H1(B,D)@W(D,D)` 为什么 backward 约有两次与 forward 同量级的 matmul？
30. 课程 toy 网络 `B=2,D=4,L=3` 有多少参数？AdaGrad 总账 432 bytes 怎样组成？Adam 624 bytes 怎样组成？
31. 1B 参数采用 2-byte 参数、2-byte 梯度、8-byte Adam states、4-byte master weight，模型状态多少 GB？
32. 标准训练循环中 `zero_grad`、forward、`backward`、`step` 各改变什么？
33. 64 batch 分 4 个等大 microbatch，为什么每份 mean loss 要除以 4？activation 峰值理想降几倍？
34. 梯度累积能不能解决“参数+optimizer state 本身已超过显存”的 OOM？为什么？
35. 顺序链有 $`L`$ 个等 activation、等重算成本的层，每 $`k`$ 层组成一个 checkpointed segment。为什么峰值 activation 层数因子约为 $`L/k+k`$？怎样推出 $`k\approx\sqrt L`$？它与逐层 `checkpoint(layer,x)` 有何区别？
36. $`6N D_{\text{tok}}`$ 的 6 从哪里来？列出两个会让公式失真的条件。

---

## 19. 自测答案

1. scalar：shape `()`、rank 0；长度 5 向量：`(5,)`、rank 1；矩阵：`(2,3)`、rank 2。
2. 元素数 $`2×3×4=24`$；axis 1 是 shape 的第二项，长度 3。
3. $`4×8=32`$ elements；FP32 4 B/element；$`32×4=128`$ bytes。
4. $`32×16×16×64=524{,}288`$ elements；BF16 共 $`524{,}288×2=1{,}048{,}576`$ bytes = 1 MiB。
5. $`49{,}152×12{,}288=603{,}979{,}776`$ elements；乘 4 得 $`2{,}415{,}919{,}104`$ bytes；除 $`2^{20}`$ 得 2304 MiB；除 $`2^{30}`$ 得 2.25 GiB。
6. FP16 exponent 较少、fraction 较多，范围小；BF16 exponent 宽度与 FP32 相同、范围大，但 fraction 较少、精度粗。$`10^{-8}`$ 可在 FP16 下溢而 BF16 仍非零。
7. 同行下一列在 storage 跳 1；下一行同列跳过整行 3 个元素，故 `(3,1)`。transpose 只交换解释后通常 `(1,3)`，并成为 non-contiguous。
8. view 共享 storage，只改 shape/stride 等解释；copy 有独立 storage。`reshape` 不保证 view：兼容则 view，不兼容可复制。
9. reshape 按原 storage 切成 `[[1,2],[3,4],[5,6]]`；transpose 是 `[[1,4],[2,5],[3,6]]`。
10. `d` 未出现在输出，所以求和；保留 `batch=2,i=3,j=5`，输出 `(2,3,5)`。
11. 输入末 axis 8 拆为 $`h×d`$；$`h=2`$，故 $`d=8/2=4`$；输出 `(3,2,4)`。
12. 每行消掉 `c` 并求和：`[1+2+3,4+5+6]=[6,15]`，shape `(2,)`。
13. `(tile w)` 得 `[1,2,3,1,2,3]`；`(w copy)` 得 `[1,1,2,2,3,3]`。括号最右 axis 变化最快。
14. 无 bias：$`3×2=6`$；加 bias：$`6+2=8`$ parameters。
15. $`70×10^9×2=140×10^9`$ bytes = 140 GB。训练还要 gradient、optimizer state、activation、workspace 等。
16. 总 $`8×80=640`$ GB。$`640×10^9/12=53.333×10^9≈53.3`$B；加 4-byte master 后每参数 16 B，$`640/16=40`$B。二者都未留 activation 等空间。
17. 一个 dot product 有 $`D`$ 次乘、$`D-1`$ 次加，合计 $`2D-1`$；输出 $`BK`$ 个，故 $`BK(2D-1)≈2BDK`$。
18. 精确：$`2×2×(2×3-1)=4×5=20`$ FLOPs；近似：$`2×2×3×2=24`$ FLOPs。
19. 秒数必须把整个吞吐放在分母：$`\frac{6.3\times10^{24}}{5.06624\times10^{17}}=1.2435\times10^7`$ s；再算 $`\frac{1.2435\times10^7\text{ s}}{86400\text{ s/day}}=143.93`$ days，约 144 天。
20. $`500/989.5=0.5053=50.53\%`$。
21. GPU launch 异步；不同步可能只量到 CPU 排队时间。起点同步排除旧任务，终点同步保证被测任务已完成。
22. $`I_{ridge}=989.5×10^{12}/(3.35×10^{12})=295.37`$ FLOP/byte。
23. $`I=1/(2+2)=0.25`$ FLOP/byte；$`0.25<295.37`$，memory-bound。
24. $`I=20/4=5`$ FLOP/byte；仍远低于 295.37，所以仍可受内存带宽限制。
25. $`I=(2n-1)/6=(2047)/6=341.17`$ FLOP/byte；理想模型中 $`341.17>295.37`$，compute-bound。
26. $`F=n(2n-1)=1024×2047=2{,}096{,}128`$ FLOPs；$`Q=2n^2+4n=2{,}101{,}248`$ bytes；$`I=0.9976`$ FLOP/byte，memory-bound。
27. 令 $`u=ab+a`$。在 $`(2,3)`$，$`u=8,f=u^2=64`$。平方规则给 $`\partial f/\partial u=2u=16`$；固定 $`b`$ 时 $`\partial u/\partial a=b+1=4`$；固定 $`a`$ 时 $`\partial u/\partial b=a=2`$。沿链相乘：$`\partial f/\partial a=16×4=64`$，$`\partial f/\partial b=16×2=32`$。有限差分：$`[f(2.001,3)-64]/0.001=(64.064016-64)/0.001=64.016`$；$`[f(2,3.001)-64]/0.001=(64.032004-64)/0.001=32.004`$，分别逼近 64 和 32。
28. PyTorch 的 backward 把新梯度加到 leaf `.grad`，不是覆盖。两次相同图计算的贡献相加；应在不想累积时 `zero_grad`。
29. 既要算输入梯度 $`(dL/dH_2)W^T`$，又要算权重梯度 $`H_1^T(dL/dH_2)`$；两者各约 $`2BD^2`$，所以 backward 约 $`4BD^2`$，是 forward $`2BD^2`$ 的两倍。
30. $`N=4^2×3=48`$。BF16 参数 96 B + BF16 梯度 96 B + AdaGrad FP32 state 192 B + 简化 activation 48 B = 432 B。Adam states 是 $`48×8=384`$ B，故 $`96+96+384+48=624`$ B。
31. 每参数 $`2+2+8+4=16`$ B；$`10^9×16=16×10^9`$ bytes = 16 GB，未含 activation/临时项。
32. `zero_grad` 清旧 `.grad`；forward 算预测、loss、建图并保存中间量；`backward` 生成并累积梯度但不改参数；`step` 读取梯度、更新 optimizer state 和参数。
33. 四个 microbatch mean 的直接和是完整 mean 的 4 倍；各除 4 后梯度平均一致。activation 同一时刻只放 $`64/4=16`$ 样本，理想峰值降 4 倍。
34. 不能。累积不减少参数、parameter gradient 或 optimizer state，只减一次需要保存的 activation；要用分片、offload、低精度状态或缩小模型。
35. 每段 $`k`$ 层时共有约 $`L/k`$ 个 segment 边界，需要长期保存约 $`L/k`$ 份；backward 重算当前段时临时保留最多约 $`k`$ 份，所以峰值因子约 $`L/k+k`$。令下降项与上升项平衡：$`L/k=k\Rightarrow k^2=L\Rightarrow k\approx\sqrt L`$；此时两项都为 $`O(\sqrt L)`$，每段约重跑一次使总额外重算为 $`O(L)`$。逐层 `checkpoint(layer,x)` 是 $`k=1`$ 的 API 示意，仍可能保存 $`L`$ 个跨层边界；只有按约 $`\sqrt L`$ 层分 segment 才得到上述跨层结论。
36. forward 约 2 FLOPs/parameter/token，输入梯度约 2，权重梯度约 2，总约 6。长 context 的 attention $`T^2`$ 主导、MoE 只激活部分参数、矩阵乘不主导、小模型、checkpoint 重算或推理都可让公式失真；任举两项。

---

## 20. 视频时间导航

> 每个链接都可直接跳到对应位置。时间点按人工字幕与画面交叉对齐，前后可能有数秒过渡。

| 时间 | 内容 |
|---:|---|
| [00:05](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=5s) | Marin 训练结果与预报吻合，资源估算为何有用 |
| [01:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=60s) | 上讲回顾、本讲路线 |
| [02:01](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=121s) | 70B/15T/1024 H100 与 8×80 GB 两个动机问题 |
| [04:43](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=283s) | Tensor、shape、rank |
| [05:33](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=333s) | FP32 与 tensor memory |
| [08:46](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=526s) | FP16 与下溢 |
| [09:53](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=593s) | BF16 的范围/精度折中 |
| [11:01](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=661s) | Mixed precision 与 autocast |
| [13:06](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=786s) | FP8、FP4、block scaling |
| [15:14](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=914s) | 课堂问答：共享 scale 与 1-bit |
| [17:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1020s) | CPU/GPU device |
| [18:02](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1082s) | 为什么用 einops |
| [19:17](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1157s) | 普通 einsum |
| [20:53](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1253s) | Batched einsum 与 axis 语义 |
| [22:50](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1370s) | reduce |
| [24:03](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1443s) | rearrange：拆 head、合 head |
| [26:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1590s) | 课堂问答：flatten 顺序 |
| [27:27](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1647s) | FLOP 与 FLOP/s |
| [29:13](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1753s) | H100 规格表的 sparse 细则 |
| [30:50](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1850s) | 矩阵乘 FLOPs 推导 |
| [33:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1980s) | 课堂问答：渐近快速矩阵乘与实际系统 |
| [34:46](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2086s) | Benchmark、GPU synchronize |
| [37:12](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2232s) | MFU |
| [38:44](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2324s) | 课堂问答：promised 与 actual |
| [40:28](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2428s) | Arithmetic intensity 开始 |
| [42:27](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2547s) | ReLU 的 bytes/FLOPs/time |
| [45:53](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2753s) | 机器 ridge intensity |
| [48:23](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2903s) | GELU 为什么仍可能 memory-bound |
| [49:46](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2986s) | Dot product |
| [50:33](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3033s) | Matrix-vector multiply |
| [51:21](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3081s) | Matrix-matrix multiply |
| [53:26](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3206s) | 单 token inference 为何像 matvec |
| [54:48](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3288s) | Roofline 总结 |
| [57:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3430s) | 深网络参数与 activation 账 |
| [59:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3550s) | 最小梯度例子 |
| [60:08](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3608s) | backward FLOPs |
| [65:40](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3940s) | $`6N D_{\text{tok}}`$ 粗估 |
| [66:47](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4007s) | AdaGrad optimizer |
| [69:20](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4160s) | 训练显存组成 |
| [72:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4330s) | Gradient accumulation |
| [73:21](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4401s) | Activation checkpointing |
| [76:16](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4576s) | 全讲总结 |

---

## 21. 来源、边界与延伸阅读

### 21.1 本笔记怎样使用来源

- **官方讲义主源：**[`lecture_02.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。课程中的 H100 数字、代码、公式、toy model、AdaGrad、训练显存、梯度累积和 checkpointing 顺序均以它为准。
- **视频主源：**[Stanford Online Lecture 2](https://www.youtube.com/watch?v=kuYAsz7zspQ)，人工轨 `English (United States)`，1425 段，约 77:16。用来补课堂问答、讲义之外的口头解释与勘误语境。
- **自动字幕：**`English (auto-generated)` 只在人工轨可能断句/错词时交叉检查，未把自动轨当权威文本。
- **第三方笔记：**仅曾用于确认章节导航，没有复制其正文、推导或例子。

### 21.2 哪些是课程，哪些是本笔记补充

**【课程内容】**tensor rank/shape/dtype/device、FP16/BF16/FP8/FP4、mixed precision、einsum/reduce/rearrange、矩阵乘 FLOPs、H100 dense 峰值口径、MFU、ReLU/GELU/dot/matvec/matmul arithmetic intensity、Roofline、toy network、基础梯度、backward FLOPs、$`6N D_{\text{tok}}`$、AdaGrad、训练显存、gradient accumulation、activation checkpointing。

**【视频补充】**Marin 预测结果、低精度课堂问答、flatten 顺序、快速矩阵乘问答、benchmark 异步解释、MFU 经验、单 token inference、课程中的两处口头滑误及其上下文。

**【补充解释/例子】**storage/stride/view/copy/contiguous 完整展开，repeat 两组手算，reshape/transpose 反例，两变量 autograd，现代 `zero_grad`/AMP 注意点，1B 模型端到端显存例子，训练循环状态表、Roofline 假设、梯度累积 loss scaling、checkpoint 随机性。它们用于闭合初学者前置知识，不声称老师逐字讲过。

### 21.3 官方延伸资料

- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch Storage](https://docs.pytorch.org/docs/stable/storage.html)
- [PyTorch Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [PyTorch Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
- [Einops 官方教程](https://einops.rocks/1-einops-basics/)
- [NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)

---

## 22. 最后一页：学完应能独立做到什么

如果你已经能不看答案完成下面六件事，本讲才算真正结束：

1. 看到任意 tensor 代码，写出每一步 shape、dtype、device 与可能的 view/copy；
2. 用有名 axis 解释 einsum/einops，而不是只说“反正 shape 对”；
3. 从一个输出元素推到整个矩阵乘 FLOPs；
4. 从读写数据推到 arithmetic intensity，再用 ridge point 判断瓶颈；
5. 把训练显存拆成参数、梯度、状态、activation，而不是只报参数量；
6. 解释 autograd、训练循环、梯度累积和 checkpointing 每一步改变了什么状态、换来了什么、没解决什么。

这就是 resource accounting 的真正用途：在运行昂贵实验之前，先让数量、单位和因果关系说话。
