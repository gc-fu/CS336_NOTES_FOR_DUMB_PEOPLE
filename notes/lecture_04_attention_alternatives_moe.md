# CS336 Lecture 4：Attention Alternatives 与 Mixture of Experts

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Tatsu Hashimoto  
> 视频：[Lecture 4](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)（约 86:15）  
> 官方讲义：[lecture_04.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)（60 页）

> **资料核对说明：**本笔记逐页检查了官方 PDF 的 60 页渲染结果；全部页面先以 contact sheet 巡检，公式、模型结构图、路由图和对比表再按原分辨率复核。视频主字幕使用 YouTube 人工轨 `English (United States)`（语言代码 `en-US`），共 1938 个片段；最后一段从 86:13 开始、约在 86:15 结束。YouTube 同时提供 `English (auto-generated)` 轨，但本笔记没有把它当作主字幕。笔记不是字幕直译，而是用讲义定主线、用人工字幕补出口头限定，再加入可手算的教学例子。

本讲使用以下来源标签：

- **【课程内容】**：官方 PDF 与视频明确讲授的内容；
- **【视频补充】**：讲义上没有写全、老师在视频中口头补充或回答的问题；
- **【补充理解】**：为了让零基础读者跟上而添加的定义、推导与连接步骤；
- **【补充例子】**：不来自课程原数值，用来逐步复算结论；
- **【延伸】**：有助于建立全貌，但第一次阅读可以跳过。

本文件的完整结构如下；第一次阅读可按顺序学习，复习时可直接使用第 0、19–22 节：

```text
0–1   复习卡、前置知识和全讲地图
2–3   标准 attention 与线性 attention
4     因果线性 attention 的 recurrent form
5–6   RetNet、Mamba-2、Gated DeltaNet
7     hybrid、local/global、sparse attention 与 DSA
8–13  MoE、router、训练信号、负载均衡、容量与系统实现
14–15 Router 稳定性、fine-tuning 与 upcycling
16    DeepSeek MoE v1/v2/v3 的课程时点快照
17–18 MLA 与 MTP
19–23 全讲主线、常见误区、自测与答案、视频导航、来源边界
```

---

## 0. 五分钟复习卡

> **第一次学习请跳到第 1 节。**这一节是复习索引，会提前使用后文才逐字解释的术语。

### 0.1 一句话主线

长序列让 full attention 的“token 两两配对”成本按 $n^2$ 增长；线性/递归模型把历史压进固定大小状态来换速度，hybrid 和 sparse attention 再补回精确检索能力；MoE 则在 FFN 处让每个 token 只激活少数专家，用近似不变的每-token 计算换取更多总参数，但会带来路由、负载和通信问题。

### 0.2 Attention alternatives 的因果链

```text
full attention 保存每个历史 token 的 K/V
        ↓
任意 query 可以直接查任意历史位置，表达灵活
        ↓
但 n 个 query × n 个 key 形成 n² 个位置对
        ↓
若去掉或核化 softmax，可把 Q(KᵀV) 重新加括号
        ↓
历史被压缩成固定大小状态 S = KᵀV
        ↓
推理解码每步只更新 S，不必重读全部历史 K/V
        ↓
压缩会丢失逐 token 的精确地址，因此加入遗忘门、定向擦除，
并与少量 full attention 或 sparse attention 混合
```

### 0.3 四个必须会复算的式子

1. 课程把一般 attention 写成：

   $$
   \operatorname{Attn}(Q,K,V)=\rho(QK^\top)V.
   $$

   $QK^\top$ 有 $n\times n=n^2$ 个分数；若 $d_k,d_v$ 都与模型宽度 $d$ 同量级，计算是 $O(n^2d)$。

2. 当 $\rho$ 是恒等映射，即“不做 softmax 等非线性处理”时，矩阵乘满足结合律：

   $$
   (QK^\top)V=Q(K^\top V).
   $$

   右边先做出大小固定为 $d_k\times d_v$ 的状态，而不是 $n\times n$ 的分数表。

3. 因果线性 attention 可以逐 token 更新：

   $$
   S_t=S_{t-1}+k_tv_t^\top,
   \qquad
   y_t^\top=q_t^\top S_t.
   $$

4. Gated DeltaNet 在旧状态上增加遗忘、定向擦除和写入门：

   $$
   S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}
   +\beta_tk_tv_t^\top.
   $$

这些公式不能只背。第 2–6 节会给同一组小矩阵，把每个乘法和每次状态更新都算出来。

### 0.4 五个边界条件

- `linear attention` 的“linear”是指随序列长度 $n$ 线性增长，不是说模型里没有非线性。
- 普通 `softmax(QKᵀ)` **不能**直接改成 `Q(KᵀV)`；第 3 节会具体说明原因。
- 固定大小状态节省推理解码的历史存储与读取，但也把许多 token 压进同一个矩阵，可能发生信息覆盖。
- Mamba-2、Gated DeltaNet 的真实实现比课件的一行式子更完整；本讲公式用于解释共同机制，不等于复刻论文全部模块。
- hybrid 模型保留少量 full attention，不是因为线性层“毫无用处”，而是用昂贵但精确的全局路径弥补固定状态压缩。

### 0.5 MoE 复习时必须同时记住的三笔账

$$
y=\sum_{i\in\operatorname{TopK}(z)}g_i(x)E_i(x)
$$

只表示当前 token 混合被选 expert 的输出。它没有说全部 expert 都免费：

- **total parameters** 决定权重存储与切分；
- **active parameters/FLOPs per token** 主要由 top-$k$ 和单 expert 大小决定；
- **device workload/communication** 由一个 batch 的实际路由分布决定。

Switch 式平衡损失：

$$
L_{\text{balance}}=\alpha N\sum_i f_iP_i.
$$

$f_i$ 是硬分配比例，$P_i$ 是平均软概率质量；它抑制 expert collapse。Router z-loss 则抑制过大 logit 尺度，**不是**另一个负载均衡公式。

---

## 1. 开始之前：最少前置知识与全讲地图

### 1.1 scalar、vector、matrix、tensor

**【补充理解】**先分清数字放在什么盒子里：

- **scalar（标量）**：一个数字，例如 $0.5$；
- **vector（向量）**：一排数字，例如 $[1,2]$；
- **matrix（矩阵）**：按行和列摆放的数字，例如 $2\times2$ 的方阵；
- **tensor（张量）**：标量、向量、矩阵和更高维数组的总称。

`shape=[4,2]` 表示 4 行、每行 2 个元素，共：

$$
4\times2=8\ \text{个元素}.
$$

本讲先省略 batch 轴和 attention head 轴，只研究一条序列的一个 head。实际模型常见 shape 是：

$$
[B,H,n,d_h],
$$

其中：

- $B$ 是 **batch size（批大小）**，一次并行处理多少条序列；
- $H$ 是 attention head 数；一个 head 可以理解为一套独立的 Q/K/V 子空间；
- $n$ 是 **sequence length（序列长度）**，即 token 位置数；
- $d_h$ 是每个 head 的向量宽度。

为减少字母冲突，正文用 $d_k$ 表示 key/query 的宽度，用 $d_v$ 表示 value 和输出的宽度。

### 1.2 token、Q、K、V 到底是什么

**【补充理解】**一个 token 经过前面网络后有一条 hidden state（隐藏状态）向量。attention 用三个学习矩阵把它投影成：

- **query（查询，$q_t$）**：“当前位置想找什么？”
- **key（键，$k_j$）**：“历史位置 $j$ 提供什么检索标签？”
- **value（值，$v_j$）**：“若位置 $j$ 被选中，实际取走什么信息？”

这里的 $t$ 和 $j$ 是 token 位置下标。把全部位置叠起来：

$$
Q\in\mathbb{R}^{n\times d_k},
\qquad
K\in\mathbb{R}^{n\times d_k},
\qquad
V\in\mathbb{R}^{n\times d_v}.
$$

$\mathbb{R}^{a\times b}$ 的人话是“由实数组成、$a$ 行 $b$ 列的矩阵”。矩阵 $Q$ 的第 $t$ 行是 $q_t^\top$。上标 $\top$ 表示 **transpose（转置）**：把列变成行，或把行变成列。

本讲写单个 $q_t,k_t,v_t$ 时，把它们看成列向量：

$$
q_t,k_t\in\mathbb{R}^{d_k},
\qquad
v_t\in\mathbb{R}^{d_v}.
$$

因此 $q_t^\top$ 是 `shape=[1,d_k]` 的行，$k_tv_t^\top$ 是 `shape=[d_k,d_v]` 的矩阵。

### 1.3 点积和外积不是同一件事

给两个长度为 2 的向量：

$$
a=\begin{bmatrix}a_1\\a_2\end{bmatrix},
\qquad
b=\begin{bmatrix}b_1\\b_2\end{bmatrix}.
$$

**dot product（点积）**把对应位置相乘再相加，结果是一个标量：

$$
a^\top b=a_1b_1+a_2b_2.
$$

例如：

$$
\begin{bmatrix}2&-1\end{bmatrix}
\begin{bmatrix}3\\4\end{bmatrix}
=2\times3+(-1)\times4=2.
$$

**outer product（外积）**让左边每个元素乘右边每个元素，结果是矩阵：

$$
ab^\top
=
\begin{bmatrix}
a_1b_1&a_1b_2\\
a_2b_1&a_2b_2
\end{bmatrix}.
$$

同一组数字的外积是：

$$
\begin{bmatrix}2\\-1\end{bmatrix}
\begin{bmatrix}3&4\end{bmatrix}
=
\begin{bmatrix}6&8\\-3&-4\end{bmatrix}.
$$

后文的 $q_t^\top k_j$ 是点积，$k_tv_t^\top$ 是外积。顺序一换，shape 和含义都会变。

### 1.4 矩阵乘法只做“行乘列”

若 $A$ 的 shape 是 `[r,m]`，$B$ 的 shape 是 `[m,c]`，中间的 $m$ 相同，所以 $AB$ 能相乘，结果 shape 是 `[r,c]`。

结果第 $i$ 行、第 $j$ 列是：

$$
(AB)_{ij}=\sum_{u=1}^{m}A_{iu}B_{uj}.
$$

符号逐个解释：

- $i$：结果选哪一行；
- $j$：结果选哪一列；
- $u$：沿共同的中间维度逐项相乘；
- $\sum$：把所有这些乘积相加。

例如：

$$
\begin{bmatrix}1&2\end{bmatrix}
\begin{bmatrix}3&4\\5&6\end{bmatrix}
=
\begin{bmatrix}
1\times3+2\times5&1\times4+2\times6
\end{bmatrix}
=\begin{bmatrix}13&16\end{bmatrix}.
$$

### 1.5 $O(\cdot)$、FLOP 与内存单位

**【补充理解】**$O(\cdot)$ 读作“大 O”，只描述输入变大时，工作量按什么速度增长。它忽略固定常数，但固定常数在真实硬件上仍然重要。

- $O(n)$：$n$ 翻倍，主工作量约翻倍；
- $O(n^2)$：$n$ 翻倍，主工作量约变成 4 倍；
- $O(n^2d)$：还要再乘每个向量的宽度 $d$。

**FLOP（floating-point operation，浮点运算）**是一次浮点加法或乘法。`FLOPs` 在本笔记表示“完成一次任务需要多少次运算”；`FLOP/s` 才是设备每秒能做多少次。为了和课件的式子完全对应，第 2–3 节主要数“标量乘法项”；若把乘和加分别记作一个 FLOP，主数量级通常再乘约 2，但两种写法的 $n$ 次方不会改变。

一个 tensor 的内存为：

$$
\text{元素数}\times\text{每元素字节数}.
$$

**dtype（data type，数据类型）**是 tensor 每个元素采用的数字存储格式。FP32 每元素 4 bytes，BF16 每元素 2 bytes。$1\ \text{MiB}=2^{20}=1,048,576$ bytes。

### 1.6 本讲两条主线

**【课程内容】**老师在 [00:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=25s) 把本讲称为建立在普通 Transformer 之上的高级架构思路，并在 [00:49](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=49s) 给出两大主题：

1. **attention alternatives（注意力替代方案）**：目标主要是把上下文长度的二次成本降下来；
2. **Mixture of Experts，MoE（混合专家）**：让每个 token 只走少数 FFN 专家，以较少的激活计算承载更多总参数。

这两条线优化的部位不同：前者主要改 attention，后者主要改 FFN。不要把“线性 attention”和“MoE 只激活少数专家”混成同一个技巧。

**【课程内容】**课件第 2 页和视频 [01:32](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=92s) 指出模型支持的 context window（上下文窗口）快速增长。视频 [02:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=131s) 进一步说明：短序列时 FFN 可能占较多计算，$n$ 足够长后，attention 的 $n^2$ 项会占主导。

**【视频补充】**在 [02:49](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=169s)，老师先提醒两个较基础的办法：混合 local/global attention，以及系统优化。到 [03:18](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=198s) 提到 FlashAttention 时，重点是减少 GPU 不同存储层之间的数据搬运；它能显著改善速度和显存，却没有把 full attention 的数学配对数从 $n^2$ 变成 $n$。

---

## 2. 标准 attention 为什么是 $O(n^2d)$

### 2.1 先看公式和每一步 shape

**【课程内容｜PDF 第 4 页｜视频 [06:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=363s)】**课程把一般 attention 写成：

$$
\operatorname{Attn}(Q,K,V)=\rho(QK^\top)V.
$$

$\rho$ 读作希腊字母 rho。这里它代表对 score matrix（分数矩阵）做的处理，例如缩放、causal mask 和逐行 softmax。先只跟踪 shape：

```text
Q              [n, d_k]
Kᵀ             [d_k, n]
QKᵀ            [n, n]
rho(QKᵀ)       [n, n]
V              [n, d_v]
rho(QKᵀ)V      [n, d_v]
```

第一步 $QK^\top$ 的第 $i,j$ 个元素是：

$$
(QK^\top)_{ij}=q_i^\top k_j.
$$

它问的是：“第 $i$ 个 query 与第 $j$ 个 key 有多匹配？”有 $n$ 个 query，每个都和 $n$ 个 key 配对，所以共有：

$$
n\times n=n^2
$$

个分数。每个点积要乘 $d_k$ 对数字，因此第一步约有 $n^2d_k$ 个标量乘法项。

第二步是 `[n,n] × [n,d_v] → [n,d_v]`。每个输出元素沿 $n$ 个位置求和，共约有：

$$
n^2d_v
$$

个标量乘法项。两步合计：

$$
n^2d_k+n^2d_v
=n^2(d_k+d_v).
$$

若 $d_k$ 和 $d_v$ 都与某个共同宽度 $d$ 同量级，便简写成 $O(n^2d)$。这里不是把 $d_k+d_v$ 错写成 $d$，而是大 O 省略了常数倍。

### 2.2 为什么内存也会有 $n^2$

普通实现若把 score 或 softmax 权重完整写到内存，会产生 `[n,n]` tensor，即 $n^2$ 个元素。只算单个 batch、单个 head：

- $n=4$：$4^2=16$ 个元素；FP32 占 $16\times4=64$ bytes；
- $n=4096$：$4096^2=16,777,216$ 个元素；BF16 占

  $$
  16,777,216\times2=33,554,432\ \text{bytes}=32\ \text{MiB}.
  $$

若有 $B=8$ 个 batch 项、$H=32$ 个 head，仅这类 `[B,H,n,n]` 数组的朴素大小就是：

$$
32\ \text{MiB}\times8\times32=8192\ \text{MiB}=8\ \text{GiB}.
$$

**【补充理解】**这不是说所有现代实现都真的常驻这样一块数组。FlashAttention 会分块计算，尽量不把完整 $n\times n$ 中间结果写回 HBM（GPU 的大容量外部显存）；训练反向也可重算一些量。但每个允许的 query-key 位置对仍要参与数学计算，所以 exact full attention 的主算术量仍是二次的。

causal mask（因果遮罩）只允许位置 $i$ 看自己和前面的位置，允许的配对数为：

$$
1+2+\cdots+n=\frac{n(n+1)}{2}.
$$

它大约是 $n^2/2$，省了常数约一半，随 $n$ 增长仍是 $O(n^2)$。

### 2.3 一组贯穿三节的 $Q,K,V$

**【补充例子】**取 $n=4,d_k=d_v=2$：

$$
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
2&-1
\end{bmatrix},
\quad
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
1&-1
\end{bmatrix},
\quad
V=
\begin{bmatrix}
1&2\\
3&1\\
2&0\\
-1&1
\end{bmatrix}.
$$

三个矩阵的 shape 都是 `[4,2]`。为让算术可以手算，并与第 3 节的重新加括号完全比较，本例暂时令 $\rho$ 为恒等映射：输入什么就原样输出什么。也就是说，本例是 **未做 softmax 的 attention 核心乘法**。

先转置 $K$：

$$
K^\top=
\begin{bmatrix}
1&0&1&1\\
0&1&1&-1
\end{bmatrix}.
$$

现在逐行算 $QK^\top$。

第一行 $q_1^\top=[1,0]$：

$$
\begin{aligned}
q_1^\top k_1&=1\times1+0\times0=1,\\
q_1^\top k_2&=1\times0+0\times1=0,\\
q_1^\top k_3&=1\times1+0\times1=1,\\
q_1^\top k_4&=1\times1+0\times(-1)=1.
\end{aligned}
$$

第二行 $q_2^\top=[0,1]$：

$$
\begin{aligned}
q_2^\top k_1&=0,\\
q_2^\top k_2&=1,\\
q_2^\top k_3&=1,\\
q_2^\top k_4&=-1.
\end{aligned}
$$

第三行 $q_3^\top=[1,1]$ 得到 `[1,1,2,0]`；第四行 $q_4^\top=[2,-1]$ 得到 `[2,-1,1,3]`。完整结果为：

$$
QK^\top=
\begin{bmatrix}
1&0&1&1\\
0&1&1&-1\\
1&1&2&0\\
2&-1&1&3
\end{bmatrix}.
$$

检查 shape：`[4,2] × [2,4] → [4,4]`。

### 2.4 再把分数矩阵乘 $V$

第一行 `[1,0,1,1]` 表示 $1v_1+0v_2+1v_3+1v_4$：

$$
\begin{aligned}
y_1^\top
&=[1,2]+[2,0]+[-1,1]\\
&=[1+2-1,\ 2+0+1]\\
&=[2,3].
\end{aligned}
$$

第二行：

$$
\begin{aligned}
y_2^\top
&=0v_1+1v_2+1v_3-v_4\\
&=[3,1]+[2,0]-[-1,1]\\
&=[3+2+1,\ 1+0-1]\\
&=[6,0].
\end{aligned}
$$

第三行：

$$
\begin{aligned}
y_3^\top
&=v_1+v_2+2v_3\\
&=[1,2]+[3,1]+[4,0]\\
&=[8,3].
\end{aligned}
$$

第四行：

$$
\begin{aligned}
y_4^\top
&=2v_1-v_2+v_3+3v_4\\
&=[2,4]-[3,1]+[2,0]+[-3,3]\\
&=[-2,6].
\end{aligned}
$$

所以：

$$
(QK^\top)V=
\begin{bmatrix}
2&3\\
6&0\\
8&3\\
-2&6
\end{bmatrix}.
$$

检查 shape：`[4,4] × [4,2] → [4,2]`。

### 2.5 本讲计算器数学：平方根、$e^x$ 与自然对数

> **【补充理解｜只会四则运算也能学】**后面的 softmax、router z-loss 和 MTP loss 都会用到本小节。这里不是要求先学完整高等数学，只要会按计算器。

#### 2.5.1 $\sqrt{x}$ 是“哪个非负数自乘得到 $x$”

$\sqrt{x}$ 读作“$x$ 的平方根”。例如：

$$
\sqrt4=2,
\qquad
2\times2=4.
$$

$\sqrt2$ 不能写成有限小数，计算器给出：

$$
\sqrt2\approx1.41421356.
$$

用四则运算检查前三位近似 $1.414$：

$$
1.414\times1.414=1.999396\approx2.
$$

所以把 attention 分数除以 $\sqrt{d_k}$，当 $d_k=2$ 时，就是除以约 $1.414$。

#### 2.5.2 $e$ 与 $e^x$ 是什么

$e\approx2.7182818$ 是一个固定常数。它可以从“连续增长”理解：把一年 100% 的增长拆得越来越细并不断复利，最终倍率会趋近 $e$。本讲只需把它当成计算器上的常数。

$e^x$ 读作“$e$ 的 $x$ 次方”，也叫 exponential（指数函数）：

$$
e^0=1,
\qquad
e^1=e\approx2.718,
\qquad
e^2\approx7.389.
$$

负指数表示倒数：

$$
e^{-1}=\frac1e\approx0.3679.
$$

在普通科学计算器上，通常按 `eˣ`，输入指数，再按 `=`。例如输入 `0.707`，会得到：

$$
e^{0.707}\approx2.028.
$$

#### 2.5.3 $\ln$ 是 $e^x$ 的反操作

本文出现的 `ln` 与公式中的 `log` **都表示自然对数**，底数都是 $e$。$\ln y$ 问：“$e$ 的多少次方等于 $y$？”因此：

$$
\ln(e^x)=x,
\qquad
e^{\ln y}=y\quad(y>0).
$$

例如 $e^2\approx7.389$，所以：

$$
\ln(7.389)\approx2.
$$

乘法进入 $\ln$ 会变成加法：

$$
\ln(ab)=\ln a+\ln b.
$$

一个后文会直接用到的例子：

$$
\ln(e^{10}+e^{10})
=\ln(2e^{10})
=\ln2+\ln(e^{10})
=\ln2+10.
$$

计算器按键示例：输入 `2` 后按 `ln`，得到 $\ln2\approx0.6931$；输入 `0.8` 后按 `ln`，得到 $\ln0.8\approx-0.2231$，所以负对数 $-\ln0.8\approx0.2231$。

#### 2.5.4 三类后文计算分别按什么键

- **Softmax（第 2.6、9 节，自测 21）**：对每个 logit 按 `eˣ`，把结果相加，再让每个结果除以总和。
- **Router z-loss（第 14 节，自测 46）**：先算每个 $e^{z_j}$，求和，按 `ln`，最后平方。
- **交叉熵/MTP loss（第 18 节，自测 55）**：对正确答案概率 $p$ 按 `ln`，再取负号，即 $-\ln p$。

### 2.6 普通 softmax attention 多了什么

**【课程内容｜视频 [06:22](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=382s)】**普通 scaled dot-product attention 会先除以 $\sqrt{d_k}$，再按行做 softmax。以第一行 logits 为例：

$$
\frac{[1,0,1,1]}{\sqrt2}
\approx[0.707,0,0.707,0.707].
$$

softmax 把每个数取指数，再除以这一行的指数总和：

$$
\operatorname{softmax}(z)_j
=\frac{e^{z_j}}{\sum_{u=1}^{n}e^{z_u}}.
$$

因为 $e^{0.707}\approx2.028$、$e^0=1$，分母约为：

$$
2.028+1+2.028+2.028=7.084.
$$

权重约为：

$$
[0.2863,0.1412,0.2863,0.2863].
$$

它们都非负且加起来约为 1。加权 value 得：

$$
\begin{aligned}
y_{1,1}&=0.2863(1)+0.1412(3)+0.2863(2)+0.2863(-1)\\
&\approx0.996,\\
y_{1,2}&=0.2863(2)+0.1412(1)+0.2863(0)+0.2863(1)\\
&\approx1.000.
\end{aligned}
$$

这不是前一小节的 `[2,3]`，因为 softmax 改变了权重。这个差别正是第 3 节不能越过的边界。

---

## 3. 线性 attention：只重新加括号，为什么能更快

### 3.1 结合律做了什么

**【课程内容｜PDF 第 4 页｜视频 [05:58](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=358s)】**矩阵乘法满足 associativity（结合律）：只要 shape 合法，三个矩阵相乘时可以先算左边，也可以先算右边：

$$
(AB)C=A(BC).
$$

它不是交换律。矩阵的顺序仍是 $A,B,C$，只是括号位置改变。套到本讲：

$$
(QK^\top)V=Q(K^\top V).
$$

左边先产生 `[n,n]`；右边先产生 `[d_k,d_v]`：

```text
左括号：([n,d_k] × [d_k,n]) × [n,d_v]
        [n,n]                  × [n,d_v] → [n,d_v]

右括号：[n,d_k] × ([d_k,n] × [n,d_v])
        [n,d_k] × [d_k,d_v]             → [n,d_v]
```

当 $d_k,d_v$ 固定而 $n$ 很大时，`[d_k,d_v]` 可以远小于 `[n,n]`。

### 3.2 用同一组数字先算 $K^\top V$

**【补充例子】**$K^\top V$ 也可以看成 4 个外积相加：

$$
K^\top V=\sum_{t=1}^{4}k_tv_t^\top.
$$

第 1 个位置：

$$
k_1v_1^\top
=
\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}1&2\end{bmatrix}
=
\begin{bmatrix}1&2\\0&0\end{bmatrix}.
$$

第 2 个位置：

$$
k_2v_2^\top
=
\begin{bmatrix}0\\1\end{bmatrix}
\begin{bmatrix}3&1\end{bmatrix}
=
\begin{bmatrix}0&0\\3&1\end{bmatrix}.
$$

第 3 个位置：

$$
k_3v_3^\top
=
\begin{bmatrix}1\\1\end{bmatrix}
\begin{bmatrix}2&0\end{bmatrix}
=
\begin{bmatrix}2&0\\2&0\end{bmatrix}.
$$

第 4 个位置：

$$
k_4v_4^\top
=
\begin{bmatrix}1\\-1\end{bmatrix}
\begin{bmatrix}-1&1\end{bmatrix}
=
\begin{bmatrix}-1&1\\1&-1\end{bmatrix}.
$$

四块逐格相加：

$$
\begin{aligned}
K^\top V
&=
\begin{bmatrix}1&2\\0&0\end{bmatrix}
+\begin{bmatrix}0&0\\3&1\end{bmatrix}
+\begin{bmatrix}2&0\\2&0\end{bmatrix}
+\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\\
&=
\begin{bmatrix}
1+0+2-1&2+0+0+1\\
0+3+2+1&0+1+0-1
\end{bmatrix}\\
&=
\begin{bmatrix}2&3\\6&0\end{bmatrix}.
\end{aligned}
$$

检查 shape：`Kᵀ [2,4] × V [4,2] → [2,2]`。

### 3.3 再用 $Q$ 查询这个小状态

现在算：

$$
Q(K^\top V)
=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
2&-1
\end{bmatrix}
\begin{bmatrix}2&3\\6&0\end{bmatrix}.
$$

逐行结果：

$$
\begin{aligned}
[1,0]\begin{bmatrix}2&3\\6&0\end{bmatrix}&=[2,3],\\
[0,1]\begin{bmatrix}2&3\\6&0\end{bmatrix}&=[6,0],\\
[1,1]\begin{bmatrix}2&3\\6&0\end{bmatrix}&=[2+6,3+0]=[8,3],\\
[2,-1]\begin{bmatrix}2&3\\6&0\end{bmatrix}&=[4-6,6-0]=[-2,6].
\end{aligned}
$$

因此：

$$
Q(K^\top V)
=
\begin{bmatrix}
2&3\\
6&0\\
8&3\\
-2&6
\end{bmatrix}
=(QK^\top)V.
$$

两个答案逐元素相同，不是近似。这次相等的条件是：使用相同 $Q,K,V$，并且中间没有插入不能交换的非线性 $\rho$。

### 3.4 两种括号的算术账

**【课程内容｜视频 [06:52](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=412s)、[07:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=422s)】**左括号的标量乘法项：

$$
C_{\text{standard}}
=n^2d_k+n^2d_v
=n^2(d_k+d_v).
$$

符号与单位：

- $C_{\text{standard}}$：一次前向的标量乘法项数；
- $n$：token 数，无单位；
- $d_k,d_v$：向量元素数，无单位；
- 结果单位是“个标量乘法项”。

右括号的两次矩阵乘分别是 $K^\top V$ 和 $Q(K^\top V)$，各有 $nd_kd_v$ 个标量乘法项：

$$
C_{\text{linear}}=nd_kd_v+nd_kd_v=2nd_kd_v.
$$

代入本例 $n=4,d_k=d_v=2$：

$$
\begin{aligned}
C_{\text{standard}}
&=4^2(2+2)\\
&=16\times4\\
&=64,\\[4pt]
C_{\text{linear}}
&=2\times4\times2\times2\\
&=32.
\end{aligned}
$$

本例右括号的标量乘法项数是一半。但这不保证小矩阵在真实 GPU 上一定快两倍，因为 kernel launch（启动一次 GPU 底层计算程序）、矩阵 shape、内存搬运和硬件利用率也会影响实测时间。

### 3.5 什么时候线性形式才开始少算

令两种估算相等：

$$
n^2(d_k+d_v)=2nd_kd_v.
$$

假设 $n>0$，两边除以 $n$：

$$
n(d_k+d_v)=2d_kd_v.
$$

再除以 $d_k+d_v$：

$$
n_*=\frac{2d_kd_v}{d_k+d_v}.
$$

$n_*$ 是只按上述乘法项估计的 crossover（交叉点）。若 $d_k=d_v=d$：

$$
n_*=\frac{2d^2}{2d}=d.
$$

**例 1：$d_k=d_v=2$。**交叉点 $n_*=2$。在 $n=4$ 时，刚才算出标准 64，线性 32。

**例 2：$d_k=d_v=64$。**

- $n=32$：

  $$
  C_{\text{standard}}=32^2(64+64)=1024\times128=131,072,
  $$

  $$
  C_{\text{linear}}=2\times32\times64\times64=262,144.
  $$

  序列比宽度短时，线性重排反而多算。

- $n=64$：

  $$
  C_{\text{standard}}=64^2\times128=524,288,
  $$

  $$
  C_{\text{linear}}=2\times64\times4096=524,288.
  $$

  两者在这份理想估算里相等。

- $n=4096$：

  $$
  C_{\text{standard}}=4096^2\times128=2,147,483,648,
  $$

  $$
  C_{\text{linear}}=2\times4096\times4096=33,554,432.
  $$

  比值为 $2,147,483,648/33,554,432=64$，线性形式的主乘法项少 64 倍。

**例 3：$d_k=64,d_v=128$。**

$$
n_*=\frac{2\times64\times128}{64+128}
=\frac{16,384}{192}
\approx85.33.
$$

- $n=64$：标准 $64^2\times192=786,432$；线性 $2\times64\times64\times128=1,048,576$；
- $n=128$：标准 $128^2\times192=3,145,728$；线性 $2\times128\times64\times128=2,097,152$。

所以“线性”是渐近优势，不是对任何 $n$、任何硬件都必胜。

### 3.6 为什么普通 softmax 不能直接交换

**【课程内容｜视频 [06:38](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=398s)】**老师明确说这里暂时把 $\rho$，尤其 softmax，拿掉。原因是：

$$
\operatorname{softmax}(QK^\top)V
\ne
Q(K^\top V)
$$

一般情况下不相等。

softmax 需要先看到同一 query 对所有 key 的分数，计算指数和分母，再逐行归一化。`softmax(A)B` 不是三个裸矩阵的连续乘法，因此不能用矩阵结合律把 softmax 穿过去。

第 2.6 节已有直接反例：相同第一行在恒等 $\rho$ 下得到 `[2,3]`，做 scaled softmax 后约为 `[0.996,1.000]`。既然值已经不同，不能声称重排保留了普通 softmax attention。

### 3.7 可核化 attention 怎样恢复线性形式

**【课程内容 + 延伸】**课件第 4 页提到 kernel version。核心不是“把 softmax 硬删掉后还假装一样”，而是选择能分解的相似度：

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k).
$$

$\phi$ 是 feature map（特征映射），把原向量变成一组可用于内积的特征。因果、带归一化的核 attention 可以维护两个前缀状态：

$$
S_t=\sum_{j\le t}\phi(k_j)v_j^\top,
\qquad
z_t=\sum_{j\le t}\phi(k_j),
$$

然后：

$$
y_t^\top
=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t}.
$$

分子产生 value 向量，分母是一个标量归一化因子。

边界必须说清：

- 对“选定的 kernel attention”而言，这种重排可以是精确的；
- 若 $\phi$ 只是用有限特征近似 softmax kernel，那么结果是对 softmax attention 的近似；
- 课程接下来讲的递归式先从最简单的恒等/纯线性情形出发。

**【视频补充】**老师在 [20:41](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1241s) 回答问题时再次划清边界：损失来自拿掉或改变 $\rho$/softmax；一旦已经定义了纯线性 attention，它的 dense 形式与 recurrent 形式之间是精确等价的。

---

## 4. 从因果前缀 attention 推到 recurrent form

### 4.1 “因果”先限制能看哪些位置

**causal（因果）**表示位置 $t$ 只能读取 $1,2,\ldots,t$，不能偷看未来 $t+1,t+2,\ldots$。在自回归语言模型里，这是为了保证模型预测下一个 token 时没有提前看到答案。

先继续使用 $\rho$ 为恒等映射的纯线性 attention。第 $t$ 个输出为：

$$
y_t^\top
=\sum_{j=1}^{t}(q_t^\top k_j)v_j^\top.
$$

符号逐个解释：

- $t$：当前 query 的位置；
- $j$：被读取的历史位置；
- $q_t^\top k_j$：一个标量匹配分数；
- $v_j^\top$：位置 $j$ 的 value 行向量，shape `[1,d_v]`；
- $\sum_{j=1}^{t}$：只把当前位置及以前的贡献加起来；
- $y_t^\top$：输出行向量，shape `[1,d_v]`。

每一项可以重新加括号：

$$
(q_t^\top k_j)v_j^\top
=q_t^\top(k_jv_j^\top).
$$

左边先把 `[1,d_k] × [d_k,1]` 变成一个标量，再乘 `[1,d_v]`；右边先把 `[d_k,1] × [1,d_v]` 变成 `[d_k,d_v]`，再用 `[1,d_k]` 查询。数值相同。

把 $q_t^\top$ 提到求和外：

$$
\begin{aligned}
y_t^\top
&=\sum_{j=1}^{t}q_t^\top(k_jv_j^\top)\\
&=q_t^\top\left(\sum_{j=1}^{t}k_jv_j^\top\right).
\end{aligned}
$$

定义 state（状态）矩阵：

$$
S_t=\sum_{j=1}^{t}k_jv_j^\top.
$$

因为前 $t-1$ 项就是 $S_{t-1}$，所以：

$$
\boxed{S_t=S_{t-1}+k_tv_t^\top},
\qquad
\boxed{y_t^\top=q_t^\top S_t}.
$$

这就是课件第 5 页的 recurrent form（递归形式）：新状态只依赖旧状态和当前输入。

### 4.2 用 4 个 token 逐次更新 $2\times2$ 状态

**【补充例子】**继续使用第 2.3 节的 $Q,K,V$。初始化：

$$
S_0=
\begin{bmatrix}0&0\\0&0\end{bmatrix}.
$$

#### 第 1 个 token

当前写入量在第 3.2 节已经算过：

$$
k_1v_1^\top=
\begin{bmatrix}1&2\\0&0\end{bmatrix}.
$$

所以：

$$
S_1=S_0+k_1v_1^\top
=\begin{bmatrix}1&2\\0&0\end{bmatrix}.
$$

用 $q_1^\top=[1,0]$ 查询：

$$
y_1^\top
=[1,0]\begin{bmatrix}1&2\\0&0\end{bmatrix}
=[1,2].
$$

#### 第 2 个 token

$$
S_2
=S_1+k_2v_2^\top
=\begin{bmatrix}1&2\\0&0\end{bmatrix}
+\begin{bmatrix}0&0\\3&1\end{bmatrix}
=\begin{bmatrix}1&2\\3&1\end{bmatrix}.
$$

用 $q_2^\top=[0,1]$ 查询：

$$
y_2^\top
=[0,1]\begin{bmatrix}1&2\\3&1\end{bmatrix}
=[3,1].
$$

#### 第 3 个 token

$$
S_3
=S_2+k_3v_3^\top
=\begin{bmatrix}1&2\\3&1\end{bmatrix}
+\begin{bmatrix}2&0\\2&0\end{bmatrix}
=\begin{bmatrix}3&2\\5&1\end{bmatrix}.
$$

用 $q_3^\top=[1,1]$ 查询：

$$
y_3^\top
=[1,1]\begin{bmatrix}3&2\\5&1\end{bmatrix}
=[3+5,2+1]
=[8,3].
$$

#### 第 4 个 token

$$
S_4
=S_3+k_4v_4^\top
=\begin{bmatrix}3&2\\5&1\end{bmatrix}
+\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
=\begin{bmatrix}2&3\\6&0\end{bmatrix}.
$$

用 $q_4^\top=[2,-1]$ 查询：

$$
y_4^\top
=[2,-1]\begin{bmatrix}2&3\\6&0\end{bmatrix}
=[4-6,6-0]
=[-2,6].
$$

四个 causal 输出依次是：

$$
[1,2],\quad[3,1],\quad[8,3],\quad[-2,6].
$$

### 4.3 用直接的 causal attention 再验一遍

为证明递归计算没有偷偷换问题，直接从分数读取前缀：

- $t=1$ 只准用分数 `[1]`，所以 $y_1=v_1=[1,2]$；
- $t=2$ 只准用第二行前两个分数 `[0,1]`，所以 $y_2=0v_1+v_2=[3,1]$；
- $t=3$ 用 `[1,1,2]`，所以 $y_3=v_1+v_2+2v_3=[8,3]$；
- $t=4$ 已没有未来位置，用 `[2,-1,1,3]`，所以 $y_4=[-2,6]$。

它与第 4.2 节逐 token 的答案逐项相同。

第 2.4 节的 **non-causal（非因果）**完整矩阵给前两行得到 `[2,3]`、`[6,0]`，是因为前两行还读取了未来位置。不要拿非因果答案去检查因果递归，然后误以为公式出错。

### 4.4 为什么它像 RNN

**【课程内容｜PDF 第 5 页｜视频 [08:09](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=489s)】****RNN（recurrent neural network，循环神经网络）**的核心特征是：按顺序读输入，并维护一个会更新的 state。

这里每一步只有两件事：

```text
写入：S_t = S_(t-1) + k_t v_t^T
读取：y_t^T = q_t^T S_t
```

若 $k_t$ shape `[d_k]`、$v_t$ shape `[d_v]`，那么无论已经读了多少 token，$S_t$ 永远是：

$$
[d_k,d_v].
$$

在本例里，不论 $t=1$ 还是 $t=1,000,000$，状态 shape 都是 `[2,2]`。这就是 fixed-size state（固定大小状态）。

**【视频补充】**老师在 [08:33](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=513s) 逐步解释了把当前 $k_tv_t^\top$ 累加到 $S$，并在 [08:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=530s) 强调：对已经定义好的纯线性 attention，矩阵形式和递归形式完全相同。

### 4.5 训练为什么喜欢并行形式，解码为什么喜欢递归形式

**【课程内容｜视频 [09:06](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=546s)、[09:19](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=559s)】**同一个数学操作可以有两种执行视图：

1. **训练的并行视图**：一整段 $Q,K,V$ 已知，可一次交给大矩阵运算；GPU 擅长并行矩阵乘。本节最朴素的 dense causal 视图仍会做二次位置配对。
2. **推理解码的串行视图**：第 $t$ 个 token 出来后，才知道第 $t+1$ 个输入；保留 $S_t$，下一步只做一次外积更新和一次查询，不必保存并重读全部历史 K/V。

这里的 **prefill（提示词预填充）**是一次处理整段已知 prompt；**decode（增量解码）**是之后一次生成一个 token。递归形式尤其适合 decode。

**【补充理解】**对每个新 token：

- 更新 $k_tv_t^\top$ 约有 $d_kd_v$ 个乘法项；
- 查询 $q_t^\top S_t$ 约有 $d_kd_v$ 个乘法项；
- 合计约 $2d_kd_v$，不随历史长度 $t$ 增长。

而标准 cached attention 的新 query 仍要和已有 $t$ 个 key 做点积，再加权 $t$ 个 value，约为 $t(d_k+d_v)$；单步随上下文线性增长，连续生成很多步时累积成二次增长。

现代线性/recurrent 模型的训练 kernel 还可使用 parallel scan（并行扫描）或 chunkwise（分块）算法，不一定真的物化完整 $n\times n$。因此“训练用二次 dense、推理用线性 recurrent”是课程用来解释 duality（对偶执行方式）的起点，不是所有实现永远只能这样做。

### 4.6 KV cache 与固定状态的内存数字

标准自回归 attention 会保存过去所有 key 和 value，这块内存称为 **KV cache（键值缓存）**。忽略 batch 和多头，长度 $n=4096,d_k=d_v=64$ 时，元素数为：

$$
n(d_k+d_v)=4096(64+64)=524,288.
$$

使用 BF16，每元素 2 bytes：

$$
524,288\times2=1,048,576\ \text{bytes}=1\ \text{MiB}.
$$

线性状态只有：

$$
d_kd_v=64\times64=4096\ \text{个元素},
$$

即：

$$
4096\times2=8192\ \text{bytes}=8\ \text{KiB}.
$$

比例是：

$$
\frac{1\ \text{MiB}}{8\ \text{KiB}}
=\frac{1024\ \text{KiB}}{8\ \text{KiB}}
=128.
$$

所以这个参数下，单 head 的固定状态比逐 token K/V 少 128 倍。真实模型还要乘层数、batch 和 head，并计入其他 activation；这里故意只比较同一 attention head 的历史信息。

### 4.7 固定状态不是免费午餐

**【视频补充｜[09:41](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=581s)】**老师指出纯线性形式的能力有限。原因可以用“多人把内容写到同一块白板”理解。

**【补充例子】**若两个 token 的 key 完全相同：

$$
k_1=k_2=\begin{bmatrix}1\\0\end{bmatrix},
\quad
v_1=\begin{bmatrix}10\\0\end{bmatrix},
\quad
v_2=\begin{bmatrix}-10\\0\end{bmatrix},
$$

两次写入相加：

$$
k_1v_1^\top+k_2v_2^\top
=\begin{bmatrix}10&0\\0&0\end{bmatrix}
+\begin{bmatrix}-10&0\\0&0\end{bmatrix}
=\begin{bmatrix}0&0\\0&0\end{bmatrix}.
$$

后来 $q^\top=[1,0]$ 查询时只能读到 `[0,0]`，无法恢复“先有 `[10,0]`、后有 `[-10,0]`”这两个独立事件。

这不是说 recurrent 实现近似了纯线性 attention；它仍精确实现自己定义的线性模型。问题在于：这个线性模型本身已经选择把许多历史位置压缩进有限的 $S$，没有像 full attention 那样保留每个位置的独立地址。遗忘门、定向擦除、扩大状态，以及周期性 full attention 都是在改善这个表达力—效率交换。

---

## 5. RetNet 与 Mamba-2：给旧状态加遗忘，给当前值加直通路

### 5.1 RetNet 的最小变化：乘一个 $\gamma$

**【课程内容｜PDF 第 5 页】**课件在递归式旁注说明：若旧状态 $S_{t-1}$ 先乘 $\gamma$，就得到 RetNet 风格的 retention（保留/衰减）机制：

$$
S_t=\gamma S_{t-1}+k_tv_t^\top.
$$

$\gamma$ 读作 gamma，是 retention/decay gate（保留/衰减门）。在最简单解释中它是 0 到 1 之间的标量：

- $\gamma=1$：旧状态完全保留，退化回普通累加；
- $\gamma=0$：旧状态全部清空，只留下当前写入；
- $0<\gamma<1$：越久以前的信息被乘越多次，权重越小。

若 $\gamma$ 固定，展开前三步：

$$
\begin{aligned}
S_1&=k_1v_1^\top,\\
S_2&=\gamma k_1v_1^\top+k_2v_2^\top,\\
S_3&=\gamma^2 k_1v_1^\top+\gamma k_2v_2^\top+k_3v_3^\top.
\end{aligned}
$$

因此离当前位置两步远的第 1 项乘 $\gamma^2$，离一步的第 2 项乘 $\gamma$，当前项乘 1。

### 5.2 RetNet 极小数字例

**【补充例子】**假设：

$$
S_{t-1}=\begin{bmatrix}4&0\\0&2\end{bmatrix},
\quad
\gamma=0.5,
\quad
k_t=\begin{bmatrix}1\\0\end{bmatrix},
\quad
v_t=\begin{bmatrix}2\\1\end{bmatrix}.
$$

先衰减旧状态：

$$
\gamma S_{t-1}
=0.5\begin{bmatrix}4&0\\0&2\end{bmatrix}
=\begin{bmatrix}2&0\\0&1\end{bmatrix}.
$$

当前写入：

$$
k_tv_t^\top
=\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}2&1\end{bmatrix}
=\begin{bmatrix}2&1\\0&0\end{bmatrix}.
$$

相加：

$$
S_t
=\begin{bmatrix}2&0\\0&1\end{bmatrix}
+\begin{bmatrix}2&1\\0&0\end{bmatrix}
=\begin{bmatrix}4&1\\0&1\end{bmatrix}.
$$

本例的 $\gamma$ 是教学用标量。真实 retention 设计可以按 head、位置或通道使用更细的衰减结构。

### 5.3 Mamba-2 的课程公式

**【课程内容｜PDF 第 7 页｜视频 [11:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=671s)】**课件用下面的机制把线性 attention 连接到 Mamba-2：

$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top,
$$

$$
y_t^\top=q_t^\top S_t+v_t^\top D,
\qquad
\gamma_t=f(x_t).
$$

首次出现的符号：

- $x_t$：当前位置输入 hidden state；
- $f$：从输入算 gate 的函数；实现会约束或参数化 gate，这里不把它固定成某个唯一函数；
- $\gamma_t$：由当前位置 $x_t$ 决定的保留门；下标 $t$ 表示每个位置可不同；
- $D$：value 的 skip matrix（跳连矩阵），shape `[d_v,d_v]`；常见讲法会把它看成对角或逐通道参数；
- $v_t^\top D$：不经过历史状态、让当前 value 直接影响输出的路径。

**【视频补充】**老师在 [12:16](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=736s) 强调 $\gamma_t$ 依赖当前输入，并控制多少旧状态继续流向现在；在 [13:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=783s) 说明这类门控仍保留可用并行/递归两种视图的 duality。

### 5.4 Mamba-2 极小数字例

**【补充例子】**设：

$$
S_{t-1}=\begin{bmatrix}2&1\\0&3\end{bmatrix},
\quad
\gamma_t=0.25,
\quad
k_t=\begin{bmatrix}1\\2\end{bmatrix},
\quad
v_t=\begin{bmatrix}2\\-1\end{bmatrix}.
$$

第一步，衰减旧状态：

$$
\gamma_tS_{t-1}
=0.25\begin{bmatrix}2&1\\0&3\end{bmatrix}
=\begin{bmatrix}0.5&0.25\\0&0.75\end{bmatrix}.
$$

第二步，算当前写入：

$$
k_tv_t^\top
=\begin{bmatrix}1\\2\end{bmatrix}
\begin{bmatrix}2&-1\end{bmatrix}
=\begin{bmatrix}2&-1\\4&-2\end{bmatrix}.
$$

第三步，得到新状态：

$$
\begin{aligned}
S_t
&=\begin{bmatrix}0.5&0.25\\0&0.75\end{bmatrix}
+\begin{bmatrix}2&-1\\4&-2\end{bmatrix}\\
&=\begin{bmatrix}2.5&-0.75\\4&-1.25\end{bmatrix}.
\end{aligned}
$$

取 query：

$$
q_t=\begin{bmatrix}1\\0.5\end{bmatrix}.
$$

从状态读取：

$$
\begin{aligned}
q_t^\top S_t
&=[1,0.5]
\begin{bmatrix}2.5&-0.75\\4&-1.25\end{bmatrix}\\
&=[1\times2.5+0.5\times4,\ 1\times(-0.75)+0.5\times(-1.25)]\\
&=[4.5,-1.375].
\end{aligned}
$$

再取一个对角 skip matrix：

$$
D=\begin{bmatrix}0.5&0\\0&2\end{bmatrix}.
$$

当前 value 的直通量为：

$$
\begin{aligned}
v_t^\top D
&=[2,-1]\begin{bmatrix}0.5&0\\0&2\end{bmatrix}\\
&=[2\times0.5+(-1)\times0,\ 2\times0+(-1)\times2]\\
&=[1,-2].
\end{aligned}
$$

最终：

$$
y_t^\top=[4.5,-1.375]+[1,-2]=[5.5,-3.375].
$$

shape 全部闭合：

```text
q_t^T [1,2] × S_t [2,2] → [1,2]
v_t^T [1,2] × D   [2,2] → [1,2]
两条 [1,2] 路径逐元素相加 → y_t^T [1,2]
```

### 5.5 $D$ 路径为什么有用，边界在哪里

**【视频补充｜[15:32](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=932s)、[15:38](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=938s)】**老师口头回看公式后，把 $v_t^\top D$ 解释为当前 value 的 residual/skip（残差/直通）路径。即使历史状态压缩或门控不适合表达当前局部信息，输出仍可直接拿到当前 value 的变换。

**【边界说明】**真实 Mamba-2 是 structured state space model（结构化状态空间模型），还包含输入投影、卷积、门控、归一化、分头/分组状态和专门 kernel 等。本节严格复算的是课件用于建立“线性 attention—状态空间模型 duality”直觉的简化式，不把这两行公式冒充完整实现。

**【课程内容】**课件第 8 页给出 Nemotron 3 的 Mamba/attention hybrid 作为当时模型案例；视频约 13:25 强调它并非只使用 Mamba 层。hybrid 为什么反复出现，会在第 7 节统一解释。

---

## 6. Gated DeltaNet：不仅忘记，还要沿当前 key 方向擦除和重写

### 6.1 公式先拆成四个动作

**【课程内容｜PDF 第 9 页｜视频 [14:17](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=857s)】**Gated DeltaNet（门控 Delta 网络，简称 GDN）的课件式为：

$$
S_t
=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}
+\beta_tk_tv_t^\top,
$$

$$
y_t^\top=q_t^\top S_t,
\qquad
\gamma_t=f(x_t),
\qquad
\beta_t=f(x_t).
$$

虽然课件把两个 gate 都写成 $f(x_t)$，真实实现会用各自的参数/投影算出它们，不表示 $\gamma_t$ 必须等于 $\beta_t$。

按执行顺序拆开：

1. $k_tk_t^\top$：找出与当前 key 对齐的状态方向；
2. $(I-\beta_tk_tk_t^\top)S_{t-1}$：沿该方向擦除一部分旧内容；
3. 乘 $\gamma_t$：整体保留/衰减擦除后的旧状态；
4. 加 $\beta_tk_tv_t^\top$：沿当前 key 方向写入当前 value。

这里：

- $I$ 是 identity matrix（单位矩阵），对角线为 1，其余为 0；shape `[d_k,d_k]`；
- $\beta_t$ 读作 beta，是 write/erase gate（写入/擦除门）；
- $k_tk_t^\top$ 的 shape 是 `[d_k,d_k]`；
- $(I-\beta_tk_tk_t^\top)S_{t-1}$ 的 shape 是 `[d_k,d_v]`；
- $\beta_tk_tv_t^\top$ 的 shape 也是 `[d_k,d_v]`；
- 因此两个分支可以相加，$S_t$ 仍是 `[d_k,d_v]`。

### 6.2 为什么 $k_tk_t^\top$ 表示“key 方向”

**【补充理解】**先选最容易看的单位 key：

$$
k_t=\begin{bmatrix}1\\0\end{bmatrix}.
$$

它的外积为：

$$
k_tk_t^\top
=\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}1&0\end{bmatrix}
=\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

左乘旧状态：

$$
\begin{bmatrix}1&0\\0&0\end{bmatrix}
\begin{bmatrix}a&b\\c&d\end{bmatrix}
=\begin{bmatrix}a&b\\0&0\end{bmatrix}.
$$

它只取出旧状态的第一行，也就是与 key `[1,0]` 对齐的部分。因此：

$$
I-\beta_tk_tk_t^\top
=\begin{bmatrix}1-\beta_t&0\\0&1\end{bmatrix}
$$

会把第一行乘 $1-\beta_t$，第二行保持不变。

### 6.3 完整数字例：$\gamma=0.5,\beta=0.25$

**【补充例子】**设：

$$
S_{t-1}=\begin{bmatrix}4&2\\1&3\end{bmatrix},
\quad
\gamma_t=0.5,
\quad
\beta_t=0.25,
$$

$$
k_t=\begin{bmatrix}1\\0\end{bmatrix},
\quad
v_t=\begin{bmatrix}2\\-1\end{bmatrix},
\quad
q_t=\begin{bmatrix}1\\1\end{bmatrix}.
$$

第一步，算 key 的方向矩阵：

$$
k_tk_t^\top=\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

第二步，算部分擦除矩阵：

$$
\begin{aligned}
I-\beta_tk_tk_t^\top
&=\begin{bmatrix}1&0\\0&1\end{bmatrix}
-0.25\begin{bmatrix}1&0\\0&0\end{bmatrix}\\
&=\begin{bmatrix}0.75&0\\0&1\end{bmatrix}.
\end{aligned}
$$

第三步，擦除旧状态对应方向：

$$
\begin{aligned}
(I-\beta_tk_tk_t^\top)S_{t-1}
&=\begin{bmatrix}0.75&0\\0&1\end{bmatrix}
\begin{bmatrix}4&2\\1&3\end{bmatrix}\\
&=\begin{bmatrix}
0.75\times4+0\times1&0.75\times2+0\times3\\
0\times4+1\times1&0\times2+1\times3
\end{bmatrix}\\
&=\begin{bmatrix}3&1.5\\1&3\end{bmatrix}.
\end{aligned}
$$

第四步，整体乘保留门：

$$
\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}
=0.5\begin{bmatrix}3&1.5\\1&3\end{bmatrix}
=\begin{bmatrix}1.5&0.75\\0.5&1.5\end{bmatrix}.
$$

第五步，算带门的当前写入：

$$
\begin{aligned}
\beta_tk_tv_t^\top
&=0.25
\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}2&-1\end{bmatrix}\\
&=0.25\begin{bmatrix}2&-1\\0&0\end{bmatrix}\\
&=\begin{bmatrix}0.5&-0.25\\0&0\end{bmatrix}.
\end{aligned}
$$

第六步，相加得到新状态：

$$
\begin{aligned}
S_t
&=\begin{bmatrix}1.5&0.75\\0.5&1.5\end{bmatrix}
+\begin{bmatrix}0.5&-0.25\\0&0\end{bmatrix}\\
&=\begin{bmatrix}2&0.5\\0.5&1.5\end{bmatrix}.
\end{aligned}
$$

最后用 query 读取：

$$
\begin{aligned}
y_t^\top
&=[1,1]\begin{bmatrix}2&0.5\\0.5&1.5\end{bmatrix}\\
&=[2+0.5,\ 0.5+1.5]\\
&=[2.5,2].
\end{aligned}
$$

### 6.4 $\beta=0$ 的准确含义

**【课程内容｜视频 [16:06](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=966s)】**令 $\beta_t=0$：

$$
\begin{aligned}
S_t
&=\gamma_t(I-0\cdot k_tk_t^\top)S_{t-1}
+0\cdot k_tv_t^\top\\
&=\gamma_tIS_{t-1}\\
&=\gamma_tS_{t-1}.
\end{aligned}
$$

因此“no input operation”的准确人话是：**当前 token 不擦除、不写入**。若 $\gamma_t<1$，旧状态仍会整体衰减；只有同时 $\gamma_t=1$ 时，状态才一字不变。

### 6.5 $\beta=1$ 怎样像覆盖一行

**【补充例子】**继续用单位 key $k_t=[1,0]^\top$，改设 $\beta_t=1,\gamma_t=1$：

$$
I-k_tk_t^\top
=\begin{bmatrix}0&0\\0&1\end{bmatrix}.
$$

对旧状态：

$$
\begin{bmatrix}0&0\\0&1\end{bmatrix}
\begin{bmatrix}4&2\\1&3\end{bmatrix}
=\begin{bmatrix}0&0\\1&3\end{bmatrix}.
$$

当前写入仍是：

$$
k_tv_t^\top=\begin{bmatrix}2&-1\\0&0\end{bmatrix}.
$$

所以：

$$
S_t
=\begin{bmatrix}0&0\\1&3\end{bmatrix}
+\begin{bmatrix}2&-1\\0&0\end{bmatrix}
=\begin{bmatrix}2&-1\\1&3\end{bmatrix}.
$$

旧状态第一行 `[4,2]` 被擦掉，再写成当前 value `[2,-1]`；第二行 `[1,3]` 保留。这就是“沿当前 key 方向定向覆盖”的最小例子。

### 6.6 为什么老师说“投影”只是直觉

**【视频补充｜[16:47](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1007s)、[17:28](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1048s)】**若 $k_t$ 是单位向量，即 $k_t^\top k_t=1$，那么 $k_tk_t^\top$ 的确是投到 $k_t$ 方向的正交 projector（投影矩阵）。

若 key 没有单位归一化，这个说法不能照搬。取：

$$
k_t=\begin{bmatrix}2\\0\end{bmatrix}.
$$

则：

$$
k_tk_t^\top=\begin{bmatrix}4&0\\0&0\end{bmatrix},
$$

当 $\beta=1$ 时：

$$
I-k_tk_t^\top
=\begin{bmatrix}-3&0\\0&1\end{bmatrix}.
$$

它把第一方向乘成 $-3$，不是“刚好删掉”。对非单位向量，精确的正交投影矩阵应为：

$$
P_k=\frac{k_tk_t^\top}{k_t^\top k_t}.
$$

本例 $k_t^\top k_t=2^2+0^2=4$，所以：

$$
P_k
=\frac14\begin{bmatrix}4&0\\0&0\end{bmatrix}
=\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

因此课件中的“擦除 key 方向”是理解门控作用的好直觉，但精确强度还取决于 key 的归一化、$\beta_t$ 和实际参数化。老师在视频中也当场补了这个限定。

### 6.7 四种状态更新放在一起

| 机制 | 状态更新 | 新增控制 | 最小直觉 |
|---|---|---|---|
| 纯线性 attention | $S_t=S_{t-1}+k_tv_t^\top$ | 无 | 永远累加 |
| RetNet 风格 | $S_t=\gamma S_{t-1}+k_tv_t^\top$ | 旧状态衰减 | 旧记忆逐渐淡化 |
| 课件中的 Mamba-2 视图 | $S_t=\gamma_tS_{t-1}+k_tv_t^\top$ | 输入相关的 $\gamma_t$；输出有 $v_t^\top D$ | 按当前输入决定保留多少，并给当前值直通路 |
| Gated DeltaNet | $S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | $\beta_t$ 同时控制定向擦除和写入 | 先擦旧槽，再把新值写进同一方向 |

表中“课件中的 Mamba-2 视图”这个措辞很重要：课程用共同的状态更新语言帮助比较架构，但实际论文/代码还包含更多投影、卷积、门控和系统细节。

### 6.8 为什么现代模型仍常做 hybrid

**【课程内容】**课件第 6、8、10 页分别用 MiniMax M1、Nemotron 3、Qwen 3.5/Qwen Next 展示 linear/recurrent layer 与 full attention layer 混合的案例。视频给出的课程时点描述包括：

- [10:07](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=607s)：MiniMax M1 使用约 7 个线性层配 1 个 full-attention 层；
- [10:59](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=659s)：老师强调完全线性的方案在大规模上还没有被充分证明可以无代价替代 full attention；
- [13:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=805s)：Nemotron 3 也是 Mamba 与 attention 的混合；
- [18:21](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1101s)：Qwen 3.5/Qwen Next 的课件快照约为 3:1 的 GDN/full-attention 混合。

这些型号和比例是课程在 2026 年的案例快照，不是所有未来模型都必须遵守的定律。

**【补充理解】**recurrent 层用固定状态高效压缩大部分历史；偶尔的 full-attention 层保留逐 token 的直接寻址机会。下一节会用具体 token 路由表，比较 local、global、hybrid 与 sparse attention 的允许边数，并解释它们怎样给长程信息开“高速公路”。

---

## 7. Hybrid、local/global attention 与 DSA：少连边，但别切断重要信息

### 7.1 先把一层 attention 看成“有向连边”

**【补充理解】**把 token 位置写成 $1,2,\ldots,n$。若 query $q_i$ 能读取 key $k_j$，就画一条 $i\rightarrow j$ 的有向边。

对 causal full attention，位置 $i$ 能看 $1$ 到 $i$。长度 $n=8$ 时逐行是：

| query | 可见 key | 边数 |
|---|---|---:|
| $q_1$ | `{1}` | 1 |
| $q_2$ | `{1,2}` | 2 |
| $q_3$ | `{1,2,3}` | 3 |
| $q_4$ | `{1,2,3,4}` | 4 |
| $q_5$ | `{1,2,3,4,5}` | 5 |
| $q_6$ | `{1,2,3,4,5,6}` | 6 |
| $q_7$ | `{1,2,3,4,5,6,7}` | 7 |
| $q_8$ | `{1,2,3,4,5,6,7,8}` | 8 |

总边数为：

$$
1+2+3+4+5+6+7+8=36.
$$

一般长度 $n$ 的 causal full attention 有：

$$
\frac{n(n+1)}{2}=O(n^2)
$$

条允许边。**full（全）**的意思是“因果约束允许的所有历史位置都能直接读取”，不是让模型看未来。

### 7.2 Causal local attention：只看最近窗口

**local attention（局部注意力）**限制每个 query 只读取附近位置。这里定义窗口 $w=3$，并明确本笔记的计数口径：窗口包含当前位置和最多两个过去位置。

因此 $q_i$ 能看：

$$
\{\max(1,i-w+1),\ldots,i\}.
$$

$n=8,w=3$ 的完整表：

| query | 可见 key | 边数 |
|---|---|---:|
| $q_1$ | `{1}` | 1 |
| $q_2$ | `{1,2}` | 2 |
| $q_3$ | `{1,2,3}` | 3 |
| $q_4$ | `{2,3,4}` | 3 |
| $q_5$ | `{3,4,5}` | 3 |
| $q_6$ | `{4,5,6}` | 3 |
| $q_7$ | `{5,6,7}` | 3 |
| $q_8$ | `{6,7,8}` | 3 |

总边数：

$$
1+2+6\times3=21.
$$

当 $n\gg w$ 时，每个 query 最多连 $w$ 条边，总量约为 $nw$，即 $O(nw)$。若 $w$ 是固定常数，就随 $n$ 线性增长。

代价是长程信息不能一步到达。第 1 个 token 的信息要到第 8 个位置，在 $w=3$ 时每层最多向后传播 2 个位置，至少需要：

$$
\left\lceil\frac{8-1}{w-1}\right\rceil
=\left\lceil\frac7{2}\right\rceil
=4
$$

层接力。$\lceil x\rceil$ 表示向上取整。

### 7.3 加一个 global token

**global token（全局词元）**是所有 query 都允许读取的特殊位置。为避免违反因果性，本例把位置 1 当作已经出现的全局/摘要 token：所有后续位置可看它，但它不能看未来。

在上一小节的 local 表中，$q_1,q_2,q_3$ 已经能看位置 1；只需给 $q_4$ 到 $q_8$ 各加一条到 key 1 的边：

| query | local + global 可见 key | 边数 |
|---|---|---:|
| $q_1$ | `{1}` | 1 |
| $q_2$ | `{1,2}` | 2 |
| $q_3$ | `{1,2,3}` | 3 |
| $q_4$ | `{1,2,3,4}` | 4 |
| $q_5$ | `{1,3,4,5}` | 4 |
| $q_6$ | `{1,4,5,6}` | 4 |
| $q_7$ | `{1,5,6,7}` | 4 |
| $q_8$ | `{1,6,7,8}` | 4 |

总边数：

$$
21+5=26.
$$

它比 full causal 的 36 条少 10 条。但请注意：若 global token 自己从未汇聚别处的信息，“大家都能看它”不会自动提供所有历史细节。实践中还要设计哪些 token 全局、它们怎样更新，以及是否周期性插入 full layer。

### 7.4 Hybrid layer：让少数层提供全局高速公路

**【课程内容｜PDF 第 3、6、8、10–11 页】****hybrid（混合）**表示模型不是全用一种层，而是在不同层交替使用低成本和高表达力机制。视频 [22:27](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1347s) 总结线性 attention 后，也把这些模型与 LSTM 风格的递归对象联系起来。

先做一个 local/full 的纯连边账。4 层中使用 3 个 `w=3` local layer 和 1 个 full layer，记作 `3:1 local/full`：

$$
3\times21+1\times36=63+36=99\ \text{条边计算}.
$$

若 4 层全是 full：

$$
4\times36=144.
$$

减少：

$$
144-99=45,
$$

相对减少：

$$
\frac{45}{144}=0.3125=31.25\%.
$$

这个数字只数本例允许位置边，不包含 projection、softmax、FFN 和 kernel 常数。

前面课程案例里的 `3:1 GDN/attention` 则表示每 4 个相关层中约 3 个 GDN、1 个 full attention。GDN 不是 local attention，不能拿每层 21 条边硬套；共同点是“多数低成本层 + 少数精确全局层”。

为何保留 full layer：

- recurrent state 会把历史压缩，full layer 能重新按 token 地址精确读取；
- local layer 的长程信息要逐层接力，full layer 可一步跨越很远；
- 对复制、人名回指、代码依赖等精确 retrieval（检索）任务，少量全局通路可能很重要。

**【课程边界】**视频 [19:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1143s) 提醒 hybrid 的受控 ablation（消融实验：只改一个因素再比较）仍不算多；小比例 full layer 看起来损失较低，而纯 recurrent 往往退化。不能把几个课程时点案例升级成永恒定律。

### 7.5 Sparse attention 与 DSA 的两阶段思路

**sparse attention（稀疏注意力）**不计算所有允许边，只挑一小部分。**DSA（DeepSeek Sparse Attention，DeepSeek 稀疏注意力）**的课程版本在 [23:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1383s) 开始介绍。

**【课程内容｜PDF 第 12–13 页｜视频 [23:33](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1413s)】**DSA 先用 lightweight indexer（轻量索引器）给候选历史位置打分，再选 top-$k$，最后只在这 $k$ 个位置上做较昂贵的完整 attention：

```text
所有候选历史 key
        ↓ 低维/低精度 indexer 快速打分
每个 query 的 top-k 位置
        ↓ 较高精度 full attention
加权读取这些位置的 V
```

**indexer（索引器）**就是一个便宜的候选筛选器，不是数据库索引文件。视频 [24:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1442s) 口头描述了 query-key 内积、ReLU 和轻量权重；[24:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1460s) 再取 top-$k$ 位置。

### 7.6 一个 $n=8,k=3$ 的 DSA 路由表

**【补充例子】**设 indexer 每个位置最多选 3 个 causal key，并把当前 key 也算在 3 个以内。下面是人为构造的结果，目的是学会数边，不代表真实 DSA 必然选择这些位置：

| query | indexer 选中的 key | 边数 |
|---|---|---:|
| $q_1$ | `{1}` | 1 |
| $q_2$ | `{1,2}` | 2 |
| $q_3$ | `{1,2,3}` | 3 |
| $q_4$ | `{1,3,4}` | 3 |
| $q_5$ | `{1,2,5}` | 3 |
| $q_6$ | `{2,4,6}` | 3 |
| $q_7$ | `{1,5,7}` | 3 |
| $q_8$ | `{2,6,8}` | 3 |

总共仍是：

$$
1+2+6\times3=21\ \text{条昂贵 attention 边}.
$$

它与 `w=3` local attention 恰好都是 21 条，但可见位置不同：local 固定看最近邻，DSA 可以跳到很远的 key。例如 $q_7$ 能直接看位置 1。

### 7.7 Indexer 不是免费的线性魔法

**【视频补充｜[26:48](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1608s)】**老师明确说 DSA indexer 仍可能要看全体 query-key 候选，因此按 $n$ 的次数仍是二次。若索引维度为 $d_{\text{idx}}$，粗略工作量是：

$$
O(n^2d_{\text{idx}})+O(nkd),
$$

第一项是便宜 indexer，第二项是只在 $k$ 个位置做昂贵 attention。它能快的条件是：

- $d_{\text{idx}}\ll d$，索引向量远小于正式 Q/K；
- indexer 可使用更低精度；
- $k\ll n$，正式 attention 的候选集合很小；
- 系统 kernel 能真正利用稀疏选择，而不是先做完全部昂贵计算再丢弃。

视频 [27:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1660s) 的课堂问答确认 indexer 没有递归状态捷径，仍是 brute-force inner products；[28:18](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1698s) 强调常数因子也很重要，不能只看“二次”两个字。

### 7.8 Recall error：索引漏掉关键位置会怎样

**recall（召回率）**在这里指真正重要的位置中，有多少被 indexer 放进 top-$k$。若 $q_8$ 的正确答案必须读取位置 3，但表中 $q_8$ 只选 `{2,6,8}`，正式 attention 根本看不到 $v_3$。

后果是：

- 本层无法直接恢复位置 3 的精确信息；
- 若别的已选位置之前汇聚过位置 3，深层网络可能间接补救；
- 若任务要求精确复制且没有替代路径，漏选会直接伤害输出。

因此 DSA 不是“只算 21 条边但与 36 条边天然完全一样”，而是训练 indexer 在节省成本的同时尽量保留任务所需边。

**【课程内容】**视频 [26:28](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1588s) 介绍的课程时点实验显示，经过完整 DSA 训练后，在一些困难长上下文检索任务上相对 full attention 损失不大；这是特定模型与实验的证据，不保证每个数据集、每个 $k$ 都不掉点。

### 7.9 Post-hoc adaptation 是什么时候加 indexer

**post-hoc adaptation（事后适配）**不是“训练结束后不再学习，直接强行删边”。课程讲的是：

1. 先用较短上下文训练 dense/full-attention 模型；
2. 长上下文 extension（扩展训练）阶段加入 indexer；
3. 继续训练，让模型和 indexer 适应稀疏选择；
4. 再进入后续 post-training。

**【视频补充】**老师在 [24:43](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1483s) 说明无需从预训练第一步就带 indexer；[25:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1501s) 说明可在长上下文 extension 阶段加入；课堂问答 [28:45](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1725s) 又解释本来就常有短上下文预训练到长上下文扩展的第二阶段，因此把稀疏适配放在那里。

---

## 8. MoE 最小模型：总参数很多，每个 token 只走少数专家

### 8.1 Dense FFN 先算一遍

**【课程内容｜PDF 第 14–15 页｜视频 [34:23](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2063s)】****MoE（Mixture of Experts，混合专家）**通常替换 Transformer block 里的 FFN/MLP，不是默认替换 attention。

一个简化 dense FFN（稠密前馈网络）是：

$$
\operatorname{FFN}(x)=\sigma(xW_{\text{up}})W_{\text{down}}.
$$

$\sigma$ 不是一个额外矩阵，而是 **activation function（激活函数）**：对中间向量的每个元素分别做同一种非线性变换，例如 ReLU 把负数变成 0、正数保留。它让 FFN 不只是两个可合并成一个的线性矩阵乘。

设输入宽度为 $d$，中间宽度为 $m$：

```text
x                 [1,d]
W_up              [d,m]
x W_up            [1,m]
activation sigma  [1,m]
W_down            [m,d]
output            [1,d]
```

忽略 bias，参数量为：

$$
P_{\text{FFN}}=dm+md=2dm.
$$

若一次乘法和一次加法各算 1 FLOP，两次矩阵向量乘的前向 FLOPs 约为：

$$
F_{\text{FFN}}\approx2dm+2md=4dm.
$$

参数量的单位是“个可学习数字”，FLOPs 的单位是“次浮点运算”。它们数值相关，但不是同一个概念。

### 8.2 加 $E$ 个 expert 和一个 router

**expert（专家）**在本讲就是一套 FFN 参数，不要想象成会说“我是医学专家”的独立智能体。**router（路由器）**是一个很小的选择层：对每个 token 打 $E$ 个专家分数。

若路由矩阵 $W_r$ 的 shape 是 `[d,E]`：

$$
z=xW_r,
$$

$z$ shape `[1,E]`，每个元素是一个 router logit（路由未归一化分数）。之后选择 top-$k$ 个 expert。

简化的 MoE 输出：

$$
y=\sum_{i\in\operatorname{TopK}(z)}g_i(x)E_i(x).
$$

- $E$：routed expert 总数；
- $k$：每个 token 选中的 routed expert 数，$k\le E$；
- $E_i(x)$：第 $i$ 个 expert 的输出向量；
- $g_i(x)$：选中 expert 的合并权重；
- top-$k$：挑出分数最大的 $k$ 项。

**【课程内容｜视频 [35:26](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2126s)】**老师从“把一个普通 MLP 替换成多个 FFN，再为每个输入挑少数 FFN”建立 MoE 心智模型。视频 [43:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2600s) 补充路由粒度通常是 token：同一句中的不同 token 可以去不同 expert。

### 8.3 四种账不能混在一起

1. **total parameters（总参数）**：模型存储的全部 expert、router 和其他层参数；影响磁盘、显存、加载与跨设备切分。
2. **active parameters per token（每 token 激活参数）**：这个 token 的前向实际经过哪些参数。它不是模型总参数。
3. **per-token FLOPs（每 token 运算量）**：这些激活矩阵实际做多少浮点加乘；还包括 router、shared experts 等开销。
4. **per-batch device work（每 batch 各设备实际工作量）**：取决于多少 token 被路由到每个 expert；即使总工作量一样，不均衡也会让某台设备成为慢尾。

“增加 expert 数而不影响 FLOPs”只能在下列近似条件下理解：$k$ 和单个 expert 大小固定，只看主 expert 矩阵乘。它不表示总参数内存、router、通信、负载均衡、**checkpoint（检查点：保存到存储中的模型权重和训练状态快照，用来恢复训练、微调或部署）**或设备数量都不变。

### 8.4 一个完整资源会计例子

**【补充例子】**取 $d=4,m=8$。一个 dense FFN：

$$
P_{\text{dense}}=2\times4\times8=64\ \text{parameters}.
$$

每 token 前向：

$$
F_{\text{dense}}\approx4\times4\times8=128\ \text{FLOPs}.
$$

现在放 $E=4$ 个同样大小 expert，top-$1$ 路由。

Expert 总参数：

$$
P_{\text{experts,total}}=4\times64=256.
$$

Router 参数：

$$
P_{\text{router}}=dE=4\times4=16.
$$

这一层总参数：

$$
P_{\text{MoE,total}}=256+16=272.
$$

一个 token 只经过一个 expert，但 router 的 16 个权重也要用于算 4 个 logits，所以简化 active parameters 为：

$$
P_{\text{active/token}}=64+16=80.
$$

主 expert FLOPs 仍为 128；router `[1,4]×[4,4]` 约为：

$$
2dE=2\times4\times4=32\ \text{FLOPs}.
$$

合计约：

$$
F_{\text{MoE/token}}\approx128+32=160\ \text{FLOPs}.
$$

所以这里“expert 主计算与 dense 相同”成立，但连 router 后不是精确的 128。真实大 FFN 中 router 通常相对小，才会近似忽略。

若改成 top-$2$，单 token 主 expert FLOPs 变为：

$$
2\times128=256,
$$

不是仍然 128。

### 8.5 Batch 总量一样，设备时间也可能不同

设一个 batch 有 $T=8$ 个 token，4 个 expert 各放在一台设备，top-$1$。总 expert 工作永远是：

$$
8\times128=1024\ \text{FLOPs}.
$$

若路由计数为 `[2,2,2,2]`，每台设备：

$$
2\times128=256\ \text{FLOPs}.
$$

若计数为 `[5,1,1,1]`，四台设备分别做：

$$
[5,1,1,1]\times128=[640,128,128,128]\ \text{FLOPs}.
$$

总和仍为 1024，但同步执行时其他设备会等做 640 FLOPs 的第一台。这叫 **load imbalance（负载不均）**，是后面 balancing loss 和 capacity 的系统动机。

### 8.6 MoE 的收益和代价都要写全

**【课程内容】**视频 [36:10](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2170s) 用 4 个同尺寸 FFN 说明：FFN 参数约增 4 倍，而 top-1 主 FFN 计算只付一份。视频 [36:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2210s) 把目标概括为增加参数而不增加主 forward FLOPs。

课程第 16–23 页列出的曲线和模型是课程时点证据：在若干实验里，固定 active compute、增加 sparse parameters 能改善 loss 或质量。它不是“MoE 永远免费胜出”的数学定理。

代价包括：

- 总 expert 权重仍要存储，单卡可能放不下；
- token 需要跨设备传输；
- 不均衡会产生等待、溢出或额外容量；
- sparse kernel 不自动拥有 dense **GEMM（General Matrix Multiplication，通用稠密矩阵乘；例如一次 `[b,d]×[d,m]`，通常由高度优化的底层 kernel 执行）**的高利用率；
- router 和训练目标可能不稳定；
- checkpoint、加载、部署和 fine-tuning 更复杂。

**【视频补充】**老师在 [39:55](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2395s) 说明 expert 给模型增加了新的并行切分轴；课堂问答 [42:08](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2528s) 同时强调它用更多 aggregate compute/内存分布换来通信成本，是否净收益依赖设备互联 topology（拓扑：机器怎样连接）。[42:38](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2558s) 又明确训练也必须保持稀疏，否则会支付全部 expert 的 FLOPs。

---

## 9. Top-$k$ 路由：从 hidden vector 一直算到合并输出

### 9.1 Router logits 手算

**【课程内容｜PDF 第 31 页｜视频 [53:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3183s)】**取 token hidden vector：

$$
x=\begin{bmatrix}1&2\end{bmatrix},
$$

4 个 expert 的 router 方向按列放进：

$$
W_r=
\begin{bmatrix}
1&0&1&-1\\
0&1&1&1
\end{bmatrix}.
$$

shape：`x [1,2] × W_r [2,4] → z [1,4]`。逐列点积：

$$
\begin{aligned}
z_1&=1\times1+2\times0=1,\\
z_2&=1\times0+2\times1=2,\\
z_3&=1\times1+2\times1=3,\\
z_4&=1\times(-1)+2\times1=1.
\end{aligned}
$$

所以 router logits：

$$
z=[1,2,3,1].
$$

top-$2$ 的 expert ID 是 $E_3$ 和 $E_2$。top-$k$ 只决定“哪几个最大”，权重怎样归一化还存在不同顺序。

### 9.2 变体 A：先对全部 expert softmax，再 top-$k$，不重归一化

先算指数：

$$
e^1\approx2.7183,
\quad e^2\approx7.3891,
\quad e^3\approx20.0855.
$$

分母：

$$
2.7183+7.3891+20.0855+2.7183=32.9112.
$$

全体概率：

$$
p\approx[0.0826,0.2245,0.6103,0.0826].
$$

选中 $E_3,E_2$ 后，若保留原概率：

$$
g_3=0.6103,
\qquad
g_2=0.2245.
$$

两者和为：

$$
0.6103+0.2245=0.8348<1.
$$

未选 expert 的概率质量被删掉，因此选中权重不一定和为 1。

### 9.3 变体 B：top-$k$ 后在选中集合内 softmax

只拿 logits $[z_3,z_2]=[3,2]$：

$$
g_3=\frac{e^3}{e^3+e^2}
=\frac{20.0855}{27.4746}
\approx0.7311,
$$

$$
g_2=\frac{e^2}{e^3+e^2}
=\frac{7.3891}{27.4746}
\approx0.2689.
$$

两者相加等于 1。这也等价于把变体 A 的两个选中概率除以 $0.8348$ 重新归一化。

**【课程边界】**课件第 31 页专门区分“先在全体上产生 gate 再 top-$k$”与“top-$k$ 后才对选中项 softmax”的模型变体。阅读模型代码时必须查清顺序，不能看到 `topk` 就猜权重。

### 9.4 变体 C：sigmoid 分数、top-$k$、再归一化

sigmoid 把每个 logit 独立压到 0 和 1 之间：

$$
\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}}.
$$

对 `[1,2,3,1]`：

$$
s\approx[0.7311,0.8808,0.9526,0.7311].
$$

top-$2$ 仍是 $E_3,E_2$。若把选中 sigmoid 分数归一化：

$$
s_3+s_2=0.9526+0.8808=1.8334,
$$

$$
g_3=\frac{0.9526}{1.8334}\approx0.5196,
\qquad
g_2=\frac{0.8808}{1.8334}\approx0.4804.
$$

sigmoid 是逐 expert 独立分数；softmax 从一开始就在 expert 之间竞争同一份总概率。即使它们选出相同 top-$k$，合并权重也会不同。

### 9.5 两个 expert 输出怎样合并

**【补充例子】**假设对同一个 $x$：

$$
E_2(x)=[2,0],
\qquad
E_3(x)=[-1,4].
$$

用“top-$k$ 后 softmax”的权重：

$$
\begin{aligned}
y_{\text{routed}}
&=0.2689E_2(x)+0.7311E_3(x)\\
&=0.2689[2,0]+0.7311[-1,4]\\
&=[0.5378,0]+[-0.7311,2.9244]\\
&=[-0.1933,2.9244].
\end{aligned}
$$

若用“全体 softmax 后不重归一化”：

$$
\begin{aligned}
y_{\text{routed}}
&=0.2245[2,0]+0.6103[-1,4]\\
&=[0.4490,0]+[-0.6103,2.4412]\\
&=[-0.1613,2.4412].
\end{aligned}
$$

若用归一化 sigmoid 权重：

$$
\begin{aligned}
y_{\text{routed}}
&=0.4804[2,0]+0.5196[-1,4]\\
&=[0.9608,0]+[-0.5196,2.0784]\\
&=[0.4412,2.0784].
\end{aligned}
$$

同一 top-$2$ 专家，三种 gate 口径给出三个不同输出。这就是必须写清“是否重归一化”的原因。

### 9.6 Shared expert：永远开启的公共路径

**【课程内容｜PDF 第 32–34 页｜视频 [54:27](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3267s)】****shared expert（共享专家）**绕过 router，对每个 token 都执行；routed experts 只在被选中时执行。

假设共享 expert 输出：

$$
E_s(x)=[0.5,0.5].
$$

使用第 9.5 节 top-$k$ 后 softmax 的 routed 输出：

$$
y_{\text{experts}}
=E_s(x)+y_{\text{routed}}
=[0.5,0.5]+[-0.1933,2.9244]
=[0.3067,3.4244].
$$

若 block 还有 residual connection，再加原输入 $x=[1,2]$：

$$
y_{\text{block}}=[1,2]+[0.3067,3.4244]=[1.3067,5.4244].
$$

共享路径让所有 token 都能获得公共变换，但它也对所有 token 付计算，不能算进“条件跳过”的节省里。视频 [58:13](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3493s) 说明共享 expert 不带来稀疏并行节省；可复制它来减少通信，但会增加权重内存副本。

### 9.7 Fine-grained expert：把大专家切成更小块

**fine-grained experts（细粒度专家）**是把一个传统大 expert 的中间宽度切成多个小 expert，让 router 可组合更细的功能块。

**【补充例子】**仍取 $d=4$。一个中间宽 $m=8$ 的传统 expert 有：

$$
2dm=2\times4\times8=64\ \text{parameters}.
$$

若按 `fine-grained ratio = 1/4` 切块，每个小 expert 宽：

$$
m_{\text{small}}=8\times\frac14=2.
$$

每个小 expert 参数：

$$
2\times4\times2=16.
$$

4 个小 expert 的总参数仍是：

$$
4\times16=64.
$$

若每 token 选择其中 2 个，激活 expert 参数是：

$$
2\times16=32,
$$

即原大 expert 的一半。也可以在保持 active budget 的同时放更多小专家，例如选择 4 个小 expert 才回到 64 个激活参数。

若再加 1 个同尺寸 shared expert，并选 2 个 routed small experts，则本层 expert active parameters 为：

$$
(1+2)\times16=48.
$$

**【课程边界】**视频 [55:07](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3307s) 解释 DeepSeekMoE 把专家切细，并让一部分 shared experts 始终开启。课程展示的消融并不完全一致：DeepSeek 的实验支持 shared experts，而 [56:41](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3401s) 讲到 OlMoE 的受控实验没有发现同样明显的 shared-expert 收益；两边都较支持 fine-grained experts。不要把单个表格写成普遍定律。

---

## 10. Routing variants 与 top-$k$ 的梯度：断开的只是选择边界

### 10.1 五种 routing 先看同一张分数表

**【课程内容｜PDF 第 27–30 页｜视频 [48:15](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2895s)】**routing function（路由函数）决定 token 与 expert 怎样配对。先用 4 个 token、2 个 expert 的 affinity score（亲和分数）建立共同例子：

| token | 对 $E_1$ 的分数 | 对 $E_2$ 的分数 |
|---|---:|---:|
| A | 9 | 8 |
| B | 8 | 1 |
| C | 7 | 6 |
| D | 1 | 9 |

分数越高表示 router 越想做这条分配。下面所有方法都看这张表，但“谁做选择”和“容量是否全局协调”不同。

### 10.2 Token-choice：每个 token 选自己的 top-$k$

**token-choice routing（token 选择专家）**让每行独立取最大值。top-$1$：

| token | 比较 | 选择 |
|---|---|---|
| A | $9>8$ | $E_1$ |
| B | $8>1$ | $E_1$ |
| C | $7>6$ | $E_1$ |
| D | $1<9$ | $E_2$ |

结果：

```text
E1 ← A, B, C   （3 个 token）
E2 ← D         （1 个 token）
```

优点是每个 token 都得到自己最喜欢的 expert，算法和 batched top-k 简单。缺点是 expert 负载没有硬保证，本例就是 3:1。

**【课程内容】**视频 [48:55](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2935s) 说明绝大多数课程时点的大型 MoE 使用 token-choice top-$k$；课程展示的 OlMoE 对比中，它通常比 expert-choice 更容易取得较好结果。这是经验结论，不是数学上支配所有方案。

### 10.3 Expert-choice：每个 expert 选自己的 token

**expert-choice routing（专家选择 token）**让每一列挑最喜欢的 token。若每个 expert 容量为 2：

```text
E1 的前两名：A(9), B(8)
E2 的前两名：D(9), A(8)
```

结果：

- A 被两个 expert 同时选中；
- B 被 $E_1$ 选中；
- D 被 $E_2$ 选中；
- C 没被任何 expert 选中。

优点是每个 expert 恰有 2 个 token，硬件负载天然平衡。缺点是 token 获得几个 expert 不固定，甚至可能一个也没有，需要额外规则保证 token coverage（覆盖）。

### 10.4 Hash routing：不学习，按固定规则分配

**hash routing（哈希路由）**用 token ID、字符串或其他固定特征算 expert ID，不训练路由矩阵。

**【补充例子】**给 A/B/C/D 编号 0/1/2/3，并用：

$$
\operatorname{expert\_id}=\operatorname{token\_id}\bmod2.
$$

`mod 2` 表示除以 2 的余数：

| token ID | 除以 2 的余数 | expert |
|---:|---:|---|
| A = 0 | 0 | $E_1$ |
| B = 1 | 1 | $E_2$ |
| C = 2 | 0 | $E_1$ |
| D = 3 | 1 | $E_2$ |

它得到完美 2:2，但完全没看前面的任务分数。优点是确定、便宜、容易复现；缺点是不能学会“这个上下文下哪个 expert 更有用”，哈希碰撞也可能把不相干 token 固定塞在一起。

**【视频补充】**老师在 [50:36](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3036s) 说 hash routing 是论文中常见 baseline，能得到一些收益，但课程时点很少作为大型部署主方案。

### 10.5 Global assignment：一次解全 batch 的匹配

**global assignment（全局分配）**不是逐行或逐列贪心，而是在容量约束下最大化全部分数之和。

仍要求每个 expert 恰收 2 个 token。一组全局可行解：

```text
E1 ← B(8), C(7)
E2 ← A(8), D(9)
```

总分：

$$
8+7+8+9=32.
$$

另一组同样 2:2 的可行解：

```text
E1 ← A(9), B(8)
E2 ← C(6), D(9)
```

总分也是：

$$
9+8+6+9=32.
$$

这说明最优解可能不唯一。优点是能把质量分数和硬容量一起考虑；缺点是要收集全 batch 分数并求 matching/linear assignment（匹配/线性分配），延迟和实现复杂度高。

**【视频补充】**视频 [52:16](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3136s) 解释这类方法会先计算 token-expert 全配对分数，再解全局最优匹配；课程判断是小规模论文中出现过，但相对 top-$k$ 很昂贵，未成为大规模共识。

### 10.6 RL routing：把选择当成策略动作

**RL（reinforcement learning，强化学习）routing**把 router 当 policy（策略），expert 选择当 action（动作），任务 loss 改善当 reward（奖励）。

这里先预告两个词：**loss（损失）**是衡量结果有多坏的数字，通常越小越好；**gradient（梯度）**描述参数发生极小变化时 loss 会怎样局部变化。下一节会从四则运算逐步定义它们。

一个最小采样表：

| token | 策略采样动作 | 只观察到的结果 |
|---|---|---|
| A | 选 $E_1$ | loss 比基线少 0.4，reward $+0.4$ |
| B | 选 $E_2$ | loss 比基线多 0.2，reward $-0.2$ |
| C | 选 $E_1$ | reward $+0.1$ |
| D | 选 $E_2$ | reward $+0.5$ |

对 A，我们没有同时运行 $E_2$，所以不知道“若选 $E_2$ 会怎样”。这就是 bandit（多臂老虎机）式 partial feedback（部分反馈）。REINFORCE 等算法可以估计离散动作梯度，但估计方差高，还要采样和 baseline。

**【课程内容】**视频 [51:05](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3065s) 把路由描述成 bandit/RL 问题；[60:04](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3604s) 说明 RL 确实能工作，但梯度方差和复杂度让它没有成为常见方案。

### 10.7 Loss、变化率与梯度下降：先用四则运算理解

**loss（损失）**是一个表示“当前答案有多坏”的数字：通常越小越好。训练的目标不是把每个中间数字随便改小，而是调整模型参数，让最终 loss 下降。

先不使用微积分符号。假设某个可调数字 $u$ 从 $2.00$ 增加到 $2.01$：

$$
\Delta u=2.01-2.00=0.01.
$$

若 loss 从 $5.00$ 增加到 $5.03$：

$$
\Delta L=5.03-5.00=0.03.
$$

“$u$ 每改变 1 单位，loss 大约改变多少”的局部变化率是：

$$
\frac{\Delta L}{\Delta u}
=\frac{0.03}{0.01}=3.
$$

把 $\Delta u$ 取得非常小时，这个比值趋近 **derivative（导数）**。若函数同时依赖许多变量，只改变其中一个、把其他变量暂时固定，叫 **partial derivative（偏导数）**，写成：

$$
\frac{\partial L}{\partial u}.
$$

$\partial$ 只是在提醒“还有别的变量，但这次先固定它们”。所有参数偏导数组成 **gradient（梯度）**。若偏导为正，增大 $u$ 会让 loss 变坏；于是 gradient descent（梯度下降）按反方向更新：

$$
u_{\text{new}}
=u_{\text{old}}
-\eta\frac{\partial L}{\partial u}.
$$

$\eta>0$ 是 learning rate（学习率），控制一步走多远。上式不是“正梯度让参数增加”，而是 **减去** 正梯度。第 11.5 节会用完全相同的有限差分，逐步推出 Switch balance loss 的变化率。

### 10.8 Top-$k$ 为什么不可微

先看两个 expert 的 logits：

$$
z=[2.000,1.999],
\qquad k=1.
$$

top-1 选择 $E_1$。只把第 2 个 logit 增加 $0.002$：

$$
z'=[2.000,2.001],
$$

选择突然跳到 $E_2$。输出路径离散切换，没有一条在边界处平滑连续的普通导数。这就是“top-$k$ 选择不可微”。

但这句话不等于“MoE 里任何东西都没有梯度”。

### 10.9 固定选中区域内，哪些参数收到任务梯度

回到第 9 节，假设当前 top-2 始终是 $E_3,E_2$，小幅改变 logits 尚未改变排名。前向：

$$
y=g_2E_2(x)+g_3E_3(x).
$$

在这个固定区域内：

1. **选中 expert 参数**：$E_2,E_3$ 的输出影响 $y$，因此从任务 loss 收到正常梯度。
2. **未选 expert 参数**：$E_1,E_4$ 没有执行，对这个 token 的任务 loss 梯度为 0。
3. **选中 router logits**：$g_2,g_3$ 是 logits 的 softmax/sigmoid 函数，连续可微，因此 router 能学会提高产生好输出的 gate。
4. **离散选中 ID**：训练通常把当前 mask 当常量，不对“若换一个 expert 会怎样”求普通反向梯度。

Router 未选 logits 的梯度还取决于门控顺序：

- **先全体 softmax、再 top-$k$，不重归一化**：全体 softmax 分母含未选 logits，未选 logits 仍可能从任务 loss 得到间接梯度；
- **先 top-$k$、只在选中集合 softmax**：未选 logits 不在这条任务计算图中，对该 token 的任务 loss 梯度为 0；
- load-balancing auxiliary loss（负载均衡辅助损失）可另外推动 router，包括降低过热 expert 的分数。

在排名边界处，mask 会跳变；常规实现不对这个跳变本身求导。训练仍能进行，是因为参数空间绝大多数点都不恰好在并列边界，而且每一步选中分支内仍有大量连续梯度。

**【视频补充】**视频 [58:57](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3537s) 把训练时必须保持稀疏与 gating 不可微并列为核心困难；[69:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4151s) 总结实际做法是直接沿选中 experts 反向传播，再用 balancing 机制阻止选择塌缩。

### 10.10 Jitter：在接近并列时探索

**stochastic jitter（随机抖动）**是在 router score 上加入小噪声。若 $E_1,E_2$ 的分数是 `[2.000,1.999]`，一次噪声可能变成 `[1.997,2.003]`，让 $E_2$ 获得一次训练信号。

优点：

- 接近并列的 expert 都有机会被探索；
- expert 不会只适应非常脆弱的固定边界。

代价：

- 路由有随机性，复现和稳定性更复杂；
- 噪声不保证带来更优 expert；
- 后续消融发现并非所有设置都需要 jitter。

**【课程内容】**视频 [60:47](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3647s) 介绍 Gaussian perturbation（高斯扰动）；[62:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3745s) 介绍 Switch 的均匀乘性 jitter，并说明后续 Google 工作移除了它，某些消融中不用反而更稳定。它是历史方案，不是必选配方。

---

## 11. Load balancing：阻止“越常被选，就越垄断”

### 11.1 Expert collapse 从哪里来

**expert collapse（专家塌缩）**是大多数 token 都涌向极少数 expert；**expert starvation（专家饿死）**是其他 expert 长期收不到 token 和训练信号。

正反馈链：

```text
某 expert 偶然略好
        ↓
router 更常选它
        ↓
它获得更多任务梯度，训练得更快
        ↓
它与冷门 expert 的差距继续增大
        ↓
更多 token 涌入它，设备也开始过载
```

**【课程内容｜视频 [63:28](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3808s)】**老师把这个 rich-get-richer（富者愈富）循环称为启发式训练必须解决的核心问题。

### 11.2 Switch load-balancing loss 的完整定义

**【课程内容｜PDF 第 40 页｜视频 [64:13](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3853s)】**Switch Transformer 的课件口径是 top-1 路由。一个 batch $\mathcal{B}$ 中有 $T$ 个 token、$N$ 个 expert：

$$
L_{\text{balance}}
=\alpha N\sum_{i=1}^{N}f_iP_i.
$$

符号逐个解释：

- $L_{\text{balance}}$：加到语言模型主 loss 上的辅助损失；
- $\alpha$：人为设定的权重，控制平衡项有多强；
- $N$：expert 数；
- $i$：expert 下标，从 1 到 $N$；
- $f_i$：硬路由后，实际发给 expert $i$ 的 token 比例；
- $P_i$：router 在 expert $i$ 上分配的平均软概率质量。

硬比例：

$$
f_i
=\frac1T\sum_{x\in\mathcal{B}}
\mathbf{1}\{\arg\max p(x)=i\}.
$$

- $p(x)$ 是 token $x$ 对全部 experts 的 router 概率；
- $\arg\max p(x)$ 返回最大概率 expert 的 ID；
- indicator $\mathbf{1}\{\text{条件}\}$ 在条件为真时等于 1，否则等于 0。

软质量：

$$
P_i=\frac1T\sum_{x\in\mathcal{B}}p_i(x).
$$

$f_i$ 回答“最后真发了多少 token”，$P_i$ 回答“router 平均有多想选它”。

### 11.3 $T=8,N=4$ 的完整概率表

**【补充例子】**取 8 个 token 的 router 概率：

| token | $p_1$ | $p_2$ | $p_3$ | $p_4$ | argmax |
|---|---:|---:|---:|---:|---|
| $x_1$ | 0.70 | 0.10 | 0.10 | 0.10 | $E_1$ |
| $x_2$ | 0.60 | 0.20 | 0.10 | 0.10 | $E_1$ |
| $x_3$ | 0.55 | 0.15 | 0.20 | 0.10 | $E_1$ |
| $x_4$ | 0.40 | 0.30 | 0.20 | 0.10 | $E_1$ |
| $x_5$ | 0.10 | 0.60 | 0.20 | 0.10 | $E_2$ |
| $x_6$ | 0.10 | 0.20 | 0.60 | 0.10 | $E_3$ |
| $x_7$ | 0.10 | 0.20 | 0.10 | 0.60 | $E_4$ |
| $x_8$ | 0.20 | 0.50 | 0.20 | 0.10 | $E_2$ |

每行和为 1。硬分配计数是 `[4,2,1,1]`，所以：

$$
f_1=\frac48=0.5,
\quad
f_2=\frac28=0.25,
\quad
f_3=f_4=\frac18=0.125.
$$

逐列求软概率和：

$$
\begin{aligned}
\sum_xp_1(x)&=0.70+0.60+0.55+0.40+0.10+0.10+0.10+0.20=2.75,\\
\sum_xp_2(x)&=0.10+0.20+0.15+0.30+0.60+0.20+0.20+0.50=2.25,\\
\sum_xp_3(x)&=0.10+0.10+0.20+0.20+0.20+0.60+0.10+0.20=1.70,\\
\sum_xp_4(x)&=0.10+0.10+0.10+0.10+0.10+0.10+0.60+0.10=1.30.
\end{aligned}
$$

检查：

$$
2.75+2.25+1.70+1.30=8.00.
$$

除以 $T=8$：

$$
P=[0.34375,0.28125,0.2125,0.1625].
$$

再检查 $P_i$ 总和：

$$
0.34375+0.28125+0.2125+0.1625=1.
$$

### 11.4 把 loss 每一项乘出来

先算点积：

$$
\begin{aligned}
\sum_i f_iP_i
&=0.5(0.34375)+0.25(0.28125)\\
&\quad+0.125(0.2125)+0.125(0.1625)\\
&=0.171875+0.0703125+0.0265625+0.0203125\\
&=0.2890625.
\end{aligned}
$$

乘 $N=4$：

$$
L_{\text{balance}}
=\alpha\times4\times0.2890625
=1.15625\alpha.
$$

若硬分配和软概率都完美均匀：

$$
f_i=P_i=\frac14,
$$

则：

$$
L_{\text{balance}}
=\alpha\times4\times
4\left(\frac14\times\frac14\right)
=\alpha.
$$

这个 loss 的均匀基线不是 0，而是 $\alpha$。训练比较的是梯度方向，不要误以为辅助损失必须降到 0。

极端塌缩到 $E_1$，并假设软概率也为 `[1,0,0,0]`：

$$
f=P=[1,0,0,0],
$$

$$
L_{\text{balance}}=\alpha\times4\times1=4\alpha.
$$

它比均匀的 $\alpha$ 大很多。

### 11.5 梯度为什么会压低热门 expert

先沿用第 10.7 节的四则运算变化率，不直接跳到偏导符号。

在一个没有跨过 top-$k$ 排名边界的小区域内，hard assignment $f_i$ 暂时不变。只把某个 token $x$ 给 expert $i$ 的软概率增加：

$$
\Delta p_i(x)=0.04.
$$

因为 $P_i$ 是 $T$ 个 token 概率的平均值，所以只有这一项变化时：

$$
\Delta P_i
=\frac{\Delta p_i(x)}{T}.
$$

用本节 $T=8$：

$$
\Delta P_i=\frac{0.04}{8}=0.005.
$$

Balance loss 中与 expert $i$ 有关的项是 $\alpha Nf_iP_i$。$P_i$ 增加 $\Delta P_i$，loss 就增加：

$$
\Delta L
=\alpha Nf_i\Delta P_i.
$$

取一个具体热门 expert：$\alpha=0.1,N=4,f_i=0.5$。逐位代入：

$$
\Delta L
=0.1\times4\times0.5\times0.005
=0.001.
$$

现在用“loss 变化除以输入概率变化”求局部变化率：

$$
\frac{\Delta L}{\Delta p_i(x)}
=\frac{0.001}{0.04}
=0.025.
$$

直接把两个增量式相除，也会得到：

$$
\frac{\Delta L}{\Delta p_i(x)}
=\alpha Nf_i\frac1T
=\frac{0.1\times4\times0.5}{8}
=0.025.
$$

把 $\Delta p_i(x)$ 取得越来越小，这个变化率就写成偏导：

$$
\frac{\partial L}{\partial p_i(x)}.
$$

这不是要背的新魔法；它只是“极小的 loss 变化 ÷ 极小的概率变化”的简写。

#### 为什么局部变化率会沿计算链相乘

Router 真正更新的是参数 $\theta$，不是把概率 $p$ 当独立旋钮。假设某次极小变化中：

$$
\Delta\theta=0.01
\quad\Longrightarrow\quad
\Delta p=0.04
\quad\Longrightarrow\quad
\Delta L=0.001.
$$

两段变化率分别是：

$$
\frac{\Delta p}{\Delta\theta}=\frac{0.04}{0.01}=4,
\qquad
\frac{\Delta L}{\Delta p}=\frac{0.001}{0.04}=0.025.
$$

相乘时中间的 $\Delta p$ 正好约掉：

$$
\frac{\Delta L}{\Delta p}
\times
\frac{\Delta p}{\Delta\theta}
=0.025\times4
=0.1
=\frac{\Delta L}{\Delta\theta}.
$$

极小增量极限下，这就是 **chain rule（链式法则）**：

$$
\frac{\partial L}{\partial\theta}
=\frac{\partial L}{\partial p_i(x)}
\frac{\partial p_i(x)}{\partial\theta}.
$$

现在再写课件导数。因为：

$$
P_i=\frac1T\sum_xp_i(x),
$$

所以对某个 token 的 $p_i(x)$：

$$
\frac{\partial P_i}{\partial p_i(x)}=\frac1T.
$$

于是：

$$
\frac{\partial L_{\text{balance}}}{\partial p_i(x)}
=\alpha Nf_i\frac1T
=\frac{\alpha Nf_i}{T}.
$$

也可以把 $f_i=\text{count}_i/T$ 代入：

$$
\frac{\partial L}{\partial p_i(x)}
=\frac{\alpha N}{T^2}\text{count}_i,
$$

这就是课件第 40 页写的导数口径。

代入 $N=4,T=8$：

$$
\begin{aligned}
\frac{\partial L}{\partial p_1(x)}&=\frac{\alpha\times4\times0.5}{8}=0.25\alpha,\\
\frac{\partial L}{\partial p_2(x)}&=\frac{\alpha\times4\times0.25}{8}=0.125\alpha,\\
\frac{\partial L}{\partial p_3(x)}
=\frac{\partial L}{\partial p_4(x)}
&=\frac{\alpha\times4\times0.125}{8}=0.0625\alpha.
\end{aligned}
$$

$E_1$ 越热门，正的 penalty gradient 越大。若用第 10.7 节定义的学习率 $\eta$，参数更新为：

$$
\theta_{\text{new}}
=\theta_{\text{old}}
-\eta\frac{\partial L}{\partial\theta}.
$$

也就是“旧参数减去学习率乘梯度”，从而更强地下调通向热门 expert 的方向。概率实际由 softmax 耦合，不能把 $p_i$ 当完全独立旋钮；严格参数梯度还要乘 $\partial p_i/\partial\theta$，但上面的有限差分已经给出 balance 项希望推动的方向。

**【视频补充】**老师在 [64:44](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3884s) 区分硬比例 $f_i$ 与软质量 $P_i$；[65:14](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3914s) 建议从导数理解公式，而不是把它当作第一性原理必然推出的唯一 loss。

### 11.6 Per-device balance 不等于重复做同一件事

**per-expert balance（逐专家平衡）**关心每个 expert；**per-device balance（逐设备平衡）**先把同一设备上的 experts 聚合，再关心设备总负载。

假设：

```text
GPU 0 放 E1, E2
GPU 1 放 E3, E4
```

第 11.3 节的硬负载为：

$$
f=[0.5,0.25,0.125,0.125].
$$

设备负载：

$$
f_{\text{GPU0}}=0.5+0.25=0.75,
$$

$$
f_{\text{GPU1}}=0.125+0.125=0.25.
$$

所以第一台设备处理 75% token，第二台只处理 25%。设备 balance loss 可直接惩罚这个 3:1，而不要求同一设备内部的两个 expert 必须完全一样忙。

**【课程内容｜PDF 第 41 页｜视频 [66:31](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3991s)】**DeepSeek v1/v2 的课程示例在 per-expert loss 外还加 per-device 聚合目标，因为系统吞吐由设备慢尾决定。视频后续问答 [71:12](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4272s) 解释：若把逐 expert 平衡强推到完全均匀，理论上设备也会均匀；但太强可能伤训练，因此可另加较弱而系统上更重要的 device 约束。

### 11.7 DeepSeek V3 bias balancing 与“aux-loss-free”边界

**【课程内容｜PDF 第 42 页｜视频 [67:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4031s)】**DeepSeek V3 风格给每个 expert 一个 selection bias（选择偏置）$b_i$。简化理解：

$$
\text{selection score}_i=s_i(x)+b_i.
$$

- expert 过载：在线统计发现它收得太多，就降低 $b_i$；
- expert 欠载：提高 $b_i$，让它更容易进入 top-$k$；
- 这个 bias 用控制回路直接调负载，不必完全依赖一个会改变模型任务梯度的 per-expert aux loss。

**【补充理解】**例如原分数 `[0.8,0.7]` 会选 $E_1$。若负载控制给 bias `[-0.2,+0.2]`：

$$
[0.8,0.7]+[-0.2,0.2]=[0.6,0.9],
$$

选择改成 $E_2$。bias 调的是进入 top-$k$ 的机会；具体模型还需查清合并权重是否仍用未加 bias 的原分数。

“auxiliary-loss-free balancing”不能缩写成“整个模型没有任何辅助平衡损失”。课程明确指出 V3 仍使用 complementary sequence-wise auxiliary loss（补充的逐序列辅助损失）防止极端不均衡。

**sequence-wise（逐序列）**表示在每条序列内部统计路由，而不是只看跨许多样本的大 batch 总平均。若序列 A 的 token 全去 $E_1$、序列 B 全去 $E_2$，合在 batch 看可能 50:50，但每条序列内部都已塌缩；seq-wise aux 能看见这个问题。

**【视频补充】**老师在 [67:23](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4043s) 说 online bias 去掉了“一些”辅助损失，紧接着在 [67:31](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4051s) 强调仍需某些辅助项防极端失衡。本文因此不会把 V3 写成完全 aux-free。

### 11.8 为什么不能干脆删掉 balancing

**【课程内容｜PDF 第 43 页｜视频 [67:52](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4072s)】**课程展示 OlMoE 的消融：移除 load-balancing loss 后，loss 曲线和 expert 利用率明显变差。视频 [68:38](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4118s) 描述几乎所有 token 涌到两个 expert，其余专家多数训练时间没有工作。

这同时伤害两件事：

- **学习**：冷门 expert 无任务梯度，存着大量没学好的参数；
- **系统**：热门 expert 所在设备变慢或溢出，其他设备空等。

Balancing 不是为了让每个 expert 获得完全相同、毫无语义差异的 token；目标是在允许一定 specialization（专门化）的同时，避免灾难性垄断。

---

## 12. Capacity、token dropping 与 batch dependence

### 12.1 为什么要给每个 expert 限容量

若 router 让 1000 个 token 都去同一个 expert，这个 expert 的临时 activation、计算时间和通信队列都可能爆掉。早期 MoE 系统常给每个 expert 一个 **capacity（容量）**：一个 batch 内最多接收多少条 token-expert assignment（token—专家分配）。

注意一条 token 在 top-$k$ 中会产生 $k$ 条 assignment。因此一个有 $T$ 个 token 的 batch，总 assignment 数为：

$$
A=T\times k.
$$

平均每个 expert 的 assignment 数为：

$$
\frac{A}{N}=\frac{Tk}{N}.
$$

引入 **capacity factor（容量因子）** $c\ge1$，给平均值留出冗余。本笔记的手算约定为：

$$
\boxed{C=\left\lceil c\frac{Tk}{N}\right\rceil}.
$$

- $C$：每个 expert 本 batch 的整数容量；
- $T$：batch 内参与该 MoE layer 的 token 数；
- $k$：每个 token 选中的 routed experts 数；
- $N$：routed expert 总数；
- $c$：容量冗余倍率；
- $\lceil\cdot\rceil$：向上取整，保证槽位不小于括号内实数。

**实现边界：**不同系统可能向下取整、向最近整数取整，或向上对齐到 8/16/硬件 block 的倍数。有些论文还以 $T/N$ 而不是 $Tk/N$ 定义单路容量，因为它们给第一、第二路由设置独立 buffer，或把 $k$ 吸收到 capacity factor 中。看到论文中的 capacity factor 时，必须同时查 assignment 口径与 rounding rule（取整规则），不能只抄一个 $c$。后文所有数字使用“总 assignment 为 $Tk$、统一 buffer、最后 `ceil`”的约定。

### 12.2 两个容量例子

**【补充例子 1】**$T=8,k=1,N=2,c=1$：

$$
C=\left\lceil1\times\frac{8\times1}{2}\right\rceil
=\lceil4\rceil
=4.
$$

每个 expert 最多接收 4 个 token，总槽位 $2\times4=8$，刚好等于 8 条 assignment。只有分配恰好 4:4 时完全不溢出。

**【补充例子 2】**$T=10,k=2,N=4,c=1.25$：

总 assignment：

$$
A=10\times2=20.
$$

平均每 expert：

$$
\frac{20}{4}=5.
$$

加冗余并向上取整：

$$
C=\lceil1.25\times5\rceil
=\lceil6.25\rceil
=7.
$$

总槽位：

$$
N C=4\times7=28.
$$

28 比实际 20 条 assignment 多 8 个槽位。这些空槽能吸收不均衡，但若实现真的为每个槽位预留 dense buffer，也会增加临时内存和无效填充。

### 12.3 Overflow 后系统能做什么

**overflow（溢出）**表示某 expert 收到的 assignment 超过 $C$。常见处理策略：

1. **drop（丢弃）**：超过容量的 expert 分支返回零；若 block 有 residual，token 仍走残差主路；
2. **reroute（改道）**：尝试 token 的第二或后续候选 expert；计算和实现更复杂；
3. **增大 capacity factor**：少丢 token，但占更多 buffer，负载慢尾仍可能存在；
4. **dropless（不丢 token）**：使用可变长度分组、动态内存或 block-sparse kernel 处理实际负载；现代系统更常采用，但仍要面对慢尾和峰值内存。

**dispatch（派发）**是把 token activation 按 expert ID 分组并送过去；**combine（合并）**是把 expert 输出送回原 token 顺序，并按 gate weight 加权相加。drop 通常发生在 dispatch 容量决策时。

### 12.4 同一个 token，为什么换一批同伴就可能被 drop

**【补充例子】**继续用 $T=8,N=2,k=1,c=1$，所以 $E_1$ 容量 $C=4$。假设容量冲突时保留 router score 最高的 4 个 assignment。

目标 token X 的 hidden state、router score 和选择始终不变：

$$
\text{X routes to }E_1\text{ with score }0.80.
$$

**Batch A 中发往 $E_1$ 的 token：**

| token | score | 排名 | 结果 |
|---|---:|---:|---|
| a | 0.95 | 1 | keep |
| b | 0.90 | 2 | keep |
| c | 0.85 | 3 | keep |
| X | 0.80 | 4 | keep |

恰好 4 个，X 被执行。

**Batch B 中发往 $E_1$ 的 token：**

| token | score | 排名 | 结果 |
|---|---:|---:|---|
| a | 0.95 | 1 | keep |
| b | 0.90 | 2 | keep |
| c | 0.85 | 3 | keep |
| d | 0.82 | 4 | keep |
| X | 0.80 | 5 | **drop** |

X 自己完全没变，只因 Batch B 多了分数 0.82 的 d，就从第 4 名掉到第 5 名。

假设 X 的输入为：

$$
h_X=[1,1],
$$

被保留时 expert 输出：

$$
E_1(h_X)=[2,-1].
$$

若 MoE block 是 residual：

$$
h_X'=h_X+\operatorname{MoE}(h_X),
$$

Batch A：

$$
h_X'=[1,1]+[2,-1]=[3,0].
$$

Batch B 若 drop 分支并返回零：

$$
h_X'=[1,1]+[0,0]=[1,1].
$$

同一个请求可能因别人的 token 与它同 batch 而得到不同中间结果，这叫 **batch dependence（批依赖）**。若 batch 组成、到达顺序或并发请求变化，表现看起来会像随机性。

### 12.5 课程中的 token-dropping 口头例子

**【课程内容｜PDF 第 47 页｜视频 [75:17](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4517s)】**老师描述热门 expert 的队列不断增长，早期系统到容量上限后会丢 token。[75:42](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4542s) 开始具体说明超过队列后只能 drop；[76:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4563s) 指出其他用户的 query 也可能把你的 token 挤出队列。

**【视频补充边界】**视频 [76:21](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4581s) 随即强调现代 dropless 架构与常见 MoE 框架已经消除了很多这种静默丢弃问题。因此：

- token dropping 是某些 capacity-limited 实现的行为；
- 它不是 MoE 数学定义的必然组成；
- “不 drop”也不等于负载问题消失：热门 expert 仍可能决定延迟和峰值内存。

### 12.6 Capacity、balancing 和 dropless 各解决哪一层问题

| 方法 | 直接解决 | 没有自动解决 |
|---|---|---|
| balancing loss / bias | 训练时让 router 少塌缩 | 单个 batch 仍可能偶然偏斜 |
| capacity | 给 buffer/队列一个硬上限 | 溢出后必须 drop、reroute 或报错 |
| dropless kernel | 不因固定槽位静默丢 token | 热门 expert 的慢尾、内存和通信 |

三者不是同义词，也不能互相完全替代。

---

## 13. MoE systems：expert parallel、all-to-all 与稀疏矩阵乘

### 13.1 Expert parallel 是怎样切模型的

**【课程内容｜PDF 第 44 页｜视频 [71:38](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4298s)】****expert parallelism（专家并行）**把不同 expert 权重放到不同设备：

```text
GPU 0: E0 的权重
GPU 1: E1 的权重
GPU 2: E2 的权重
GPU 3: E3 的权重
```

每台设备不必存全部 routed experts，但 token activation 必须去持有被选 expert 的设备。

它与两种常见并行轴不同：

- **data parallelism（数据并行）**：每台设备放模型副本，各处理不同小 batch；
- **model/tensor parallelism（模型/张量并行）**：把同一个大矩阵或不同层切到多设备；
- **expert parallelism**：天然按 expert 这一组相互独立的 FFN 权重切分。

视频 [72:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4320s) 先回顾 data/model parallel，[72:36](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4356s) 再说明 expert 提供额外切分轴。

### 13.2 All-to-all 是什么

**all-to-all（全互换通信）**是一种 collective communication（集体通信）：每台设备可给每台其他设备发送不同数据块，同时也从它们接收不同数据块。

MoE 一层的简化顺序：

```text
1. 每台设备对本地 token 算 router
2. 按目的 expert 对 token activation 做 permutation（重排）
3. all-to-all dispatch：送到 expert 所在设备
4. 各设备运行本地 experts
5. all-to-all return：把输出送回 token 原设备
6. unpermute + combine：恢复原顺序并按 gate 合并
```

**permutation（置换/重排）**只改变行顺序，不改变每行数值；**unpermute** 用反向索引恢复原 token 顺序。

### 13.3 4 个 token 的逐字节通信账

**【补充例子】**只有 2 台 GPU：

```text
GPU 0 持有 E0，也拥有输入 token t1,t2
GPU 1 持有 E1，也拥有输入 token t3,t4
```

每个 token activation 的 hidden size 为 $d=8$，dtype 为 BF16，每元素 2 bytes：

$$
8\times2=16\ \text{bytes/token}.
$$

top-1 路由：

| token | 输入所在设备 | 选择 expert | expert 所在设备 | dispatch |
|---|---|---|---|---|
| $t_1$ | GPU 0 | $E_0$ | GPU 0 | 本地，不上网络 |
| $t_2$ | GPU 0 | $E_1$ | GPU 1 | 远程 16 bytes |
| $t_3$ | GPU 1 | $E_0$ | GPU 0 | 远程 16 bytes |
| $t_4$ | GPU 1 | $E_1$ | GPU 1 | 本地，不上网络 |

dispatch 跨设备流量：

$$
2\ \text{个远程 token}\times16\ \text{bytes}=32\ \text{bytes}.
$$

Expert 输出 shape 仍 `[8]`，返回原设备再走一次相同流量：

$$
32\ \text{bytes return}.
$$

忽略 expert ID、gate、对齐和通信协议 header，总跨设备 payload：

$$
32+32=64\ \text{bytes}.
$$

若 top-2 且两个 expert 都选，每个 token 有一条本地、一条远程 assignment。4 个 token 的远程 dispatch：

$$
4\times16=64\ \text{bytes},
$$

加返回共：

$$
64+64=128\ \text{bytes}.
$$

所以 top-$k$ 不只增加 expert FLOPs，也通常增加 dispatch/return 流量。

### 13.4 为什么跨节点更难

**node（节点）**是一台含若干加速器的服务器。节点内 GPU 往往有较快专用互联，跨节点要走网络，bandwidth（带宽，每秒可传多少 bytes）通常更小、latency（延迟，一次通信等待多久）通常更大。

若某 expert 在远端节点，activation 必须跨网络往返。总时间粗略受：

$$
\text{communication time}
\approx\text{latency}
+\frac{\text{bytes}}{\text{bandwidth}}
$$

约束。小消息容易被 latency 主导，大消息容易被 bytes/bandwidth 主导。

Load imbalance 又会放大问题：上一节若 GPU 0 收 5 份、其余各 1 份，all-to-all 结束后大家还得等 GPU 0 做完。这个最慢参与者叫 **straggler（慢尾设备/任务）**。

**【视频补充】**课堂问答 [42:26](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2546s) 已提前说明 expert parallel 是否划算高度依赖网络 topology；[44:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2640s) 说明设备切得越多，通信最终会限制并行扩展。

### 13.5 为什么 sparse 不自动等于快

**sparse（稀疏）**只表示许多 token-expert 配对没有计算。硬件速度还取决于剩下的工作能否排成足够大的规则矩阵乘。

若 4 个 expert 分别收到 `[5,1,1,1]` 个 token，朴素实现会启动 4 次大小不同的小 GEMM（general matrix multiplication，通用矩阵乘）：

```text
E1: [5,d] × [d,m]
E2: [1,d] × [d,m]
E3: [1,d] × [d,m]
E4: [1,d] × [d,m]
```

`[1,d]×[d,m]` 这类很小的乘法可能填不满 GPU；4 次 kernel launch、索引 gather/scatter（收集/散回）和 padding 也有开销。若只说“跳过 75% experts，所以必然快 4 倍”，会漏掉这些成本。

### 13.6 MegaBlocks 在什么层级解决问题

**【课程内容｜PDF 第 45 页｜视频 [72:56](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4376s)】****MegaBlocks** 是面向 MoE 的系统/算法库思路：把 token 按 expert 分组，把许多不同 expert 的乘法表示成 block-sparse matrix multiplication（块稀疏矩阵乘），尽量由少数大 kernel 高效完成。

**block-sparse（块稀疏）**不是一个个零元素随意散落，而是把矩阵分成规则小块，只计算有 token-expert 连接的块。课程第 45 页的三幅图从：

1. 多个独立小矩阵乘；
2. 一个块对角大矩阵乘；
3. 可表达不均匀动态路由的块稀疏矩阵乘；

逐步说明怎样把 MoE workload 映射到硬件友好的形式。

MegaBlocks 解决的是“怎样更好执行已选的稀疏路由”，不是替 router 选择，也不是数学上消除 all-to-all。若 block 很空、尺寸不适合硬件、索引开销大或通信主导，稀疏实现仍可能不如高利用率 dense GEMM。

**【视频补充】**老师在 [73:14](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4394s) 从块对角与结构化稀疏解释合并计算，[73:47](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4427s) 把它概括为硬件—架构协同设计。

### 13.7 Activation down-projection 的字节收益

**【课程内容｜PDF 第 46 页｜视频 [74:06](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4446s)】**课件用 Nemotron 3 的 LatentMoE 示例说明：在 all-to-all 前把 routed activation 从宽度 $d$ 下投影到较小宽度 $r$，通信后再恢复；shared expert 可保留较大本地宽度。

**down-projection（下投影）**是用学习矩阵把长向量映射成较短 latent vector（潜在向量），例如 `[d]→[r]`，其中 $r<d$。

先用第 13.3 节小例子，$d=8\rightarrow r=2$，BF16：

- 压缩前每 token：

  $$
  8\times2=16\ \text{bytes};
  $$

- 压缩后每 token：

  $$
  2\times2=4\ \text{bytes};
  $$

- 缩减倍数：

  $$
  \frac{16}{4}=4.
  $$

若有 4 条远程 assignment，并假设 dispatch 和 return 都传宽度 $r$ 的 latent activation：

$$
\text{压缩前 round trip}
=4\times16\times2=128\ \text{bytes},
$$

$$
\text{压缩后 round trip}
=4\times4\times2=32\ \text{bytes}.
$$

大型数字例：$d=4096\rightarrow r=512$，BF16。

每 token 压缩前：

$$
4096\times2=8192\ \text{bytes}=8\ \text{KiB}.
$$

压缩后：

$$
512\times2=1024\ \text{bytes}=1\ \text{KiB}.
$$

缩减 8 倍。1024 条远程 assignment 的单向 payload：

$$
1024\times8\ \text{KiB}=8192\ \text{KiB}=8\ \text{MiB}
$$

降为：

$$
1024\times1\ \text{KiB}=1024\ \text{KiB}=1\ \text{MiB}.
$$

若来回都传 latent，round trip 从 16 MiB 降到 2 MiB。

### 13.8 下投影不是免费压缩

它还会增加：

- down/up projection 的参数与 FLOPs；
- 低维瓶颈可能丢信息；
- 新 projection kernel 和 activation；
- shared 与 routed 路径重新合并的设计复杂度。

是否更快要比较“省下的网络时间”与“新增投影计算/信息损失”。当跨节点通信是瓶颈时，省 bytes 可能远比多做两个本地矩阵乘重要；当 expert 都在同一高速互联内，收益可能较小。

**【视频补充】**老师在 [74:34](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4474s) 描述先对 residual stream 下投影，再发起 collective communication；[74:46](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4486s) 强调它在不把整个模型 hidden size 都缩小的情况下显著减少通信。

### 13.9 本轮系统闭环

```text
router 产生 token→expert assignments
        ↓
permutation 把同 expert token 分组
        ↓
all-to-all 把 activation 送到 expert 设备
        ↓
MegaBlocks/其他 kernel 执行稀疏 FFN
        ↓
all-to-all 把 expert output 返回
        ↓
unpermute + gate-weighted combine
```

吞吐取决于整条链中最慢的一段。总 expert FLOPs 少，不代表 all-to-all、慢尾、临时内存和小矩阵利用率自动消失。

---

## 14. MoE 稳定性与 fine-tuning

### 14.1 Router 为什么是数值敏感点

**【课程内容｜PDF 第 48 页｜视频 [76:35](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4595s)】**router 同时包含：

1. 线性投影得到 logits；
2. softmax 或 sigmoid；
3. top-$k$ 排名；
4. 可能的归一化与辅助 loss。

指数会放大差异，top-$k$ 又会把很小的排名变化变成离散 expert 切换，所以 router 常比普通线性层更怕舍入误差。

**FP32（32-bit floating point，32 位浮点）**有 1 个符号位、8 个 exponent bits（指数位）和 23 个 fraction bits（小数/尾数存储位）。BF16 同样有 8 个指数位，但只有 7 个 fraction bits，所以数值范围接近 FP32，精细分辨率低得多；FP16 的指数范围也更小。

在数值 1 附近，BF16 相邻可表示数的间距约为：

$$
2^{-7}=0.0078125.
$$

两个 logits：

$$
[1.000,1.001]
$$

在 BF16 中可能被舍入成相同值，top-1 只能靠 tie-break；FP32 能保留这 0.001 的差异。Router 用 FP32 的意义是：

- 更准确地保留接近并列的 logit 差；
- 更准确地累加 softmax/log-sum-exp 的 reduction；
- 若原计算是 FP16，还获得更大的指数范围。

它仍不能保证训练永不发散，也不能代替 stable softmax（先减最大 logit）、合理初始化和 balancing。

**【视频补充】**视频 [77:10](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4630s) 指出 MoE 新增的 router softmax 是额外数值风险；[77:35](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4655s) 给出只把 expert router 保持 FP32 的常见策略。

### 14.2 Router z-loss 的公式

**【课程内容｜PDF 第 48–49 页】**设一个 batch 有 $B$ 个参与路由的 token、$N$ 个 expert，token $i$ 对 expert $j$ 的 router logit 是 $z_j^{(i)}$：

$$
L_z
=\frac1B\sum_{i=1}^{B}
\left(
\log\sum_{j=1}^{N}e^{z_j^{(i)}}
\right)^2.
$$

这里的 $\log$ 与第 2.5 节的 $\ln$ 是同一个自然对数；手算时按计算器的 `ln` 键，不是按以 10 为底的 `log₁₀`。

符号逐个解释：

- $B$：router 这次看多少个 token；
- $N$：expert 数；
- $z_j^{(i)}$：第 $i$ 个 token 给第 $j$ 个 expert 的未归一化分数；
- $\sum_j e^{z_j^{(i)}}$：softmax 分母，也叫 partition function（配分函数）；
- $\log\sum_j e^{z_j}$：log-sum-exp，记作 $\operatorname{LSE}(z)$；
- 平方：让绝对值很大的 log partition 付更大惩罚；
- 实际总目标常写 $L_{\text{task}}+\lambda_zL_z$，$\lambda_z$ 是很小的 loss weight。

### 14.3 相同 softmax、不同 z-loss 的极小例

**【补充例子】**只有两个 expert，先取：

$$
z^{(1)}=[0,0].
$$

Softmax：

$$
\left[\frac{e^0}{e^0+e^0},\frac{e^0}{e^0+e^0}\right]
=[0.5,0.5].
$$

Log-sum-exp：

$$
\operatorname{LSE}([0,0])=\log(1+1)=\log2\approx0.6931.
$$

单 token z-loss：

$$
L_z\approx0.6931^2\approx0.4805.
$$

把两个 logits 都加 10：

$$
z^{(2)}=[10,10].
$$

Softmax 完全不变，因为分子分母都有共同的 $e^{10}$：

$$
\left[\frac{e^{10}}{2e^{10}},\frac{e^{10}}{2e^{10}}\right]
=[0.5,0.5].
$$

但：

$$
\operatorname{LSE}([10,10])
=\log(2e^{10})
=10+\log2
\approx10.6931,
$$

$$
L_z\approx10.6931^2\approx114.3434.
$$

若 $\lambda_z=0.001$，加进总 loss 的量分别约为：

$$
0.001\times0.4805=0.0004805,
$$

$$
0.001\times114.3434=0.1143434.
$$

z-loss 强烈反对“softmax 概率没变、logits 却整体漂得很大”的情况，这有助于数值稳定。

### 14.4 Z-loss 不负责 expert balance

设 batch 的每个 token 都有 logits `[1,-1]`。每个 token 都选 $E_1$，所以路由完全塌缩；z-loss 虽会压低整体 logit 尺度，却没有公式要求不同 token 分给不同 expert。

$$
\operatorname{LSE}([1,-1])
=\log(e+e^{-1})
\approx1.1269,
$$

$$
L_z\approx1.1269^2\approx1.26997.
$$

它优化的是 log partition 尺度；第 11 节的 $L_{\text{balance}}$ 才显式使用 $f_i$ 统计 expert 负载。两者可同时存在，功能不能互换。

**【视频补充】**视频 [77:49](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4669s) 把 z-loss 与早期 MoE router 稳定性联系起来；[78:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4680s) 提到课程展示的 OlMoE 消融中，移除 router z-loss 会出现更多 loss spikes。该曲线是特定训练设置证据，不表示任何 MoE 的最佳 $\lambda_z$ 都相同。

### 14.5 为什么 sparse MoE 在小数据 fine-tuning 中更易过拟合

**fine-tuning（微调）**是在预训练 checkpoint 上用较小目标数据继续训练。**overfitting（过拟合）**表示训练集表现继续改善，验证/新数据表现却恶化。

MoE 的风险来自两层稀疏：

- 总 expert 参数很多，小数据却不足以约束每套参数；
- 每个 expert 只看到路由给自己的子集，单 expert 的有效样本比总 fine-tuning 集还少。

**【补充例子】**只有 1000 个 fine-tuning token、8 个 expert、top-1。即使完美均匀，每个 expert 平均只看：

$$
1000/8=125\ \text{tokens}.
$$

若负载 `[500,100,100,100,50,50,50,50]`，最后四个 expert 各只看 50 token，却可能各有数百万参数。它们很容易记住小样本噪声。

### 14.6 课程给出的 fine-tuning 策略边界

**【课程内容｜PDF 第 50 页｜视频 [78:21](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4701s)】**课程展示 sparse MoE 在小下游数据上出现较大 train-validation gap，并给出两类案例：

1. Zoph 等工作的策略：冻结/少动 MoE experts，微调 non-MoE MLP 或 attention 等 dense 部分；视频 [78:57](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4737s) 提到 attention-only 等做法。
2. DeepSeek 课程案例：使用约 1.4M SFT（supervised fine-tuning，有监督微调）样本，让数据量足以更新更多 MoE 参数；视频 [79:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4760s) 用“多用数据”解释这一路线。

这些是课程时点策略，不是互斥定律。实际还可按验证集选择学习率、冻结范围、**regularization（正则化：给训练加入约束或噪声，降低模型死记小数据的倾向）**和 **early stopping（提前停止：验证集表现不再改善时停止训练，避免继续过拟合）**；具体做法必须依据模型与数据规模验证。

---

## 15. Upcycling：把 dense checkpoint 改造成 MoE 起点

### 15.1 什么被复用，什么必须新建

**【课程内容｜PDF 第 51 页｜视频 [79:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4780s)】****upcycling（升级再利用）**从已经训练好的 dense model checkpoint 初始化 MoE，而不是从随机权重训练整个新模型。

典型步骤：

1. 复用 embedding、attention、norm、output head 等非 FFN 权重；
2. 把每层 dense FFN 权重复制成多个 expert；
3. 新建 router，通常随机或近均匀初始化；
4. 可给 expert 副本加入极小扰动，或依赖随机路由/数据让副本逐渐分化；
5. 继续预训练，让 router 学路由、experts 学 specialization。

“复制”增加总参数，但没有凭空增加已经学到的不同知识：初始 expert 若完全相同，输出也完全相同。

### 15.2 两 expert 的极简例

**【补充例子】**dense 模型中只有一个标量 FFN：

$$
E(x)=wx,
\qquad w=2.
$$

对 $x=3$：

$$
E(3)=2\times3=6.
$$

Upcycle 成 2 个 experts：

$$
w_1=2,
\qquad w_2=2.
$$

总 expert 参数从 1 个变成 2 个，但 top-1 每 token 只用 1 个。无论 router 初始选谁：

$$
E_1(3)=E_2(3)=6.
$$

因此刚转换时模型可保留 dense FFN 行为，但两个 expert 没有 specialization。

给极小对称破坏：

$$
w_1=2.01,
\qquad w_2=1.99.
$$

则：

$$
E_1(3)=6.03,
\qquad
E_2(3)=5.97.
$$

后续若某类 token 走 $E_1$、另一类走 $E_2$，各自梯度会继续把它们拉向不同功能。Router 仍必须训练；复制 FFN 并不会自动告诉 router 哪个 expert 适合哪个 token。

### 15.3 收益与局限

收益：

- 重用 dense pretraining 已经付出的算力；
- 转换瞬间可近似保持原模型功能；
- 后续用 sparse compute 扩展总参数和 specialization 空间。

局限：

- 总权重内存和 checkpoint 体积立刻增加；
- expert 初始高度相同，分化需要额外数据与训练；
- 随机 router 可能造成负载不均或短期质量扰动；
- dense checkpoint 的缺陷也会被复制；
- 若最终本来就计划大规模 MoE 训练，从一开始训练 MoE 可能更直接。

**【视频补充】**老师在 [80:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4800s) 定义从 dense model 复制 MLP 并随机初始化 router；[80:23](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4823s) 说明随机路由与继续训练让 experts 分化。

### 15.4 MiniCPM 与 Qwen 只是课程时点案例

**【课程内容｜PDF 第 52 页｜视频 [81:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4863s)】**MiniCPM 课程材料在总参数上有一个小冲突：PDF 第 52 页表格写 **13.6B**，而视频在约 81:15 口述为 **13.4B**。两者相差 0.2B；本笔记如实保留，不擅自说成完全一致。其余课程快照为：

- 从约 2.4B dense 模型 upcycle 到上述约 13.4B（视频）/13.6B（PDF）总参数；
- top-$k=2$、8 experts；
- 约 4B active parameters；
- 继续用约 520B tokens 训练并展示相对 base model 的收益。

这些数字描述该实验，不代表 upcycling 普遍只需很少继续训练；520B tokens 本身已经很大。

**【课程内容｜PDF 第 53 页】**Qwen MoE 课程快照：

- 从 Qwen 1.8B 初始化；
- top-$k=4$；
- 60 routed experts、4 shared experts；
- 课程将其作为较早的大规模 upcycling 成功案例。

视频 [81:24](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4884s) 给出口头总结。老师在 [81:46](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4906s) 同时提醒课程时点的新模型较少再采用 upcycling，因为大型主训练往往直接从 MoE 开始。这是 2026 年课堂观察，不是未来永远不会 upcycle。

---

## 16. DeepSeek MoE v1 → v2 → v3：同一设计怎样加系统约束

### 16.1 先把课件数字原样放在一起

**【课程内容｜PDF 第 35、54–56 页】**下表只整理课件口径，不把型号快照写成架构定律：

| 版本 | 总参数 / 每 token active | routed experts | routed active | shared | fine-grained 口径 | 课件强调的新机制 |
|---|---|---:|---:|---:|---|---|
| DeepSeek MoE v1 | 16B / 2.8B | 64 | 第35页写6；第54页排版写`64/4` | 2 | 第35页 ratio `1/4` | standard top-k；expert + device aux balance |
| DeepSeek V2 | 236B / 21B | 160 | 第55页写6 | 2 | 第55页写`160/10`，可读作160 routed、细粒度约`1/10` | top-M device routing；双向通信平衡 loss |
| DeepSeek V3 | 671B / 37B | 第35页明确256 | 8 | 1 | 第35页 ratio `1/14` | sigmoid/selected normalization；top-k + top-M；bias balance + seq-wise aux |

`B` 在这张表中是 billion（十亿参数），不是前文 batch size。`16B / 2.8B` 表示约 160 亿总参数、每 token 激活约 28 亿参数；它不是显存字节。

### 16.2 第 56 页的 `258` / `256` 与 `V2` 标题冲突

第 56 页视觉上明确印的是普通数字 **`258`**，并不是 $2^8$，也没有“文本抽取丢上标”的证据。可是第 35 页模型表明确写 **256**，DeepSeek V3 的课程模型口径也与 256 routed experts 对齐。因此最合理的处理是：把第 56 页的 `258` 标为课件误印，同时保留原页事实；不能悄悄改成 256，更不能虚构一个上标解释。

同页大标题是 DeepSeek MoE V3，正文参数行却印成 `V2 (671B – 37 active)`。结合 671B/37B、页面标题和后续机制，应理解为 V3 页内的 `V2` 标签笔误；本文仍明确披露，不假装课件没有不一致。

V1 的另一处口径也要保留：第 35 页 active 列写 6，而第 54 页 `64/4` 的斜杠排版可能让人误读为 4 active。这里把第35页的“6 active”和第54页原字符串同时列出，读具体模型时以相应一手报告定义为准。

### 16.3 Routed、active、shared、fine-grained 再对齐

- **routed experts**：需要 router 条件选择的 expert 总池；
- **routed active**：每 token 从 routed pool 选多少个；
- **shared experts**：每 token 必走，不在 top-$k$ 竞争中；
- **fine-grained ratio**：小 expert 相对传统大 expert 的宽度/参数比例；例如 `1/4` 表示一个小块约为原 expert 四分之一；
- **active parameters**：还包括 attention、embedding、norm、shared experts 等全模型路径，不能只用 `routed active × 单 expert 参数` 猜标题数字。

例如 V3 课程口径是 256 routed、top-8、1 shared。一个 token 不会运行全部 256；它运行 8 个 routed 小 experts，再加 shared 和其他 dense 部分。

### 16.4 V1：建立共享 + 细粒度原型

**【课程内容｜视频 [82:33](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4953s)】**V1 已包含：

- shared experts 负责公共加工；
- fine-grained routed experts 提供更灵活组合；
- standard top-k；
- expert/device balancing auxiliary objectives。

它回答的是“怎样让 routed experts 更专门化，同时不重复学习所有公共模式”。

### 16.5 V2：路由也必须尊重网络拓扑

**top-M device routing（前 M 台设备路由）**先根据候选 expert 分数挑少数目标设备，再把 top-$k$ expert 限制在这些设备内。人话：不让一个 token 为 6 个 experts 横跨 6 台远端机器。

**【补充例子】**4 台设备各放若干 expert，token 的 top-6 原本分散在设备 `{0,1,2,3}`。若 top-$M=2$ 先选设备 `{0,2}`，最终 routed experts 只能来自这两台，可能牺牲一点纯 score 最优，却减少通信扇出。

**communication balance（通信平衡）**不仅看每台设备收到多少 activation，也关注发出/返回流量。若一台机器总是净接收大量 token，它的链路和 expert 都可能成为慢尾。

**【课程内容｜视频 [82:55](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4975s)】**老师把 V2 的 top-M device routing 与 communication balancing 解释为“把系统目标写进训练设计”，并在 [83:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4991s) 强调成功训练不只有建模，还要尊重系统约束。

### 16.6 V3：改变 gate 与 balance，但不是完全无 aux

V3 课程快照继续 shared + fine-grained 设计，并加入：

- sigmoid 产生独立 affinity；
- top-k/top-M 做稀疏选择；
- 选中权重再归一化；
- per-expert online bias 调负载；
- complementary seq-wise aux 防极端失衡。

**【课程内容｜视频 [83:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5005s)】**老师描述 V3 的 balancing 变化，并在 [83:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5020s) 提到 sigmoid 加归一化的 gate 变化。这里仍沿用第 11.7 节的边界：“aux-loss-free”指主专家平衡策略，不代表训练目标里绝无辅助项。

### 16.7 不要从这三行表推出错误定律

- expert 数从 64→160→256 是三个模型选择，不代表下一代必须继续按比例增加；
- active parameters 从 2.8B→21B→37B 同时受模型宽度、层数、shared 和 dense 部分影响；
- top-$k$、fine-grained ratio、设备数共同决定计算和通信；只看总参数无法预测吞吐；
- 表中模型事实属于课件的历史快照，性能结论必须在相同训练数据、硬件和评测下比较。

---

## 17. MLA：缓存低维 latent，而不是缓存完整 K/V

### 17.1 普通 KV cache 的 shape 与字节

**【课程内容｜PDF 第 57–58 页｜视频 [83:59](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5039s)】****MLA（Multi-head Latent Attention，多头潜在注意力）**先把 hidden state 压成低维 latent，再从 latent 生成 K/V。

先做一个单 head 教学例，忽略 batch：

$$
d_{\text{model}}=8,
\quad d_c=2,
\quad d_k=4,
\quad d_v=4,
\quad n=4.
$$

普通 KV cache 每 token 存：

$$
d_k+d_v=4+4=8\ \text{elements}.
$$

BF16 每元素 2 bytes：

$$
8\times2=16\ \text{bytes/token}.
$$

4 个历史 token：

$$
4\times16=64\ \text{bytes}.
$$

### 17.2 从 $h$ 算 latent，再上投影 K/V

采用 row-vector 约定：

$$
c_t=h_tW^{DKV},
$$

$$
k_t^C=c_tW^{UK},
\qquad
v_t^C=c_tW^{UV}.
$$

- $W^{DKV}$：down-projection，shape `[8,2]`；
- $c_t$：KV latent，shape `[1,2]`；
- $W^{UK}$：key up-projection，shape `[2,4]`；
- $W^{UV}$：value up-projection，shape `[2,4]`；
- 上标 $C$ 表示 content/non-rotary 教学分量，不是 capacity。

取：

$$
h_t=[1,0,2,0,0,1,0,0],
$$

$$
W^{DKV}=
\begin{bmatrix}
1&0\\
0&0\\
1&0\\
0&0\\
0&0\\
0&1\\
0&0\\
0&0
\end{bmatrix}.
$$

第一 latent 维读取 $h_1+h_3=1+2=3$，第二维读取 $h_6=1$：

$$
c_t=h_tW^{DKV}=[3,1].
$$

再取：

$$
W^{UK}=
\begin{bmatrix}
1&0&1&0\\
0&1&0&-1
\end{bmatrix},
$$

$$
W^{UV}=
\begin{bmatrix}
1&1&0&0\\
0&1&1&0
\end{bmatrix}.
$$

Key：

$$
\begin{aligned}
k_t^C
&=[3,1]
\begin{bmatrix}
1&0&1&0\\
0&1&0&-1
\end{bmatrix}\\
&=[3,1,3,-1].
\end{aligned}
$$

Value：

$$
\begin{aligned}
v_t^C
&=[3,1]
\begin{bmatrix}
1&1&0&0\\
0&1&1&0
\end{bmatrix}\\
&=[3,4,1,0].
\end{aligned}
$$

所有 shape：

```text
h_t    [1,8] × W_DKV [8,2] → c_t [1,2]
c_t    [1,2] × W_UK  [2,4] → k_t [1,4]
c_t    [1,2] × W_UV  [2,4] → v_t [1,4]
```

### 17.3 Cache 从 64 bytes 降到 16 bytes

若推理时只缓存 $c_t$，每 token：

$$
d_c\times2\ \text{bytes}=2\times2=4\ \text{bytes}.
$$

$n=4$：

$$
4\times4=16\ \text{bytes}.
$$

相对普通 K/V 的 64 bytes：

$$
\frac{64}{16}=4
$$

倍压缩。

这是简化教学口径，只比较单 head 的 content K/V 与 latent。真实 MLA 还会缓存小的 decoupled rotary key 分量，处理多头共享/投影，并可能有量化、对齐和额外 **metadata（元数据：不是向量内容本身、而是记录位置、分页或布局等管理信息的小数据）**，所以实际字节不能只套 4 倍。

**【视频补充】**视频 [84:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5051s) 说明先生成低维 $c$，再由它生成 Q/K/V；[84:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5065s) 说明 KV cache 可只存更小的 $c$。

### 17.4 没有 RoPE 时，为什么能把 $W^{UK}$ 吸到 query 侧

取 query：

$$
q_s=[1,2,0,1].
$$

与刚才 $k_t^C=[3,1,3,-1]$ 的点积：

$$
q_sk_t^{C\top}
=1\times3+2\times1+0\times3+1\times(-1)
=4.
$$

因为：

$$
k_t^C=c_tW^{UK},
$$

所以：

$$
\begin{aligned}
q_sk_t^{C\top}
&=q_s(c_tW^{UK})^\top\\
&=q_s(W^{UK})^\top c_t^\top\\
&=\underbrace{\left(q_s(W^{UK})^\top\right)}_{q_s'}c_t^\top.
\end{aligned}
$$

先算 absorbed query（吸收后的 query）：

$$
\begin{aligned}
q_s'
&=[1,2,0,1]
\begin{bmatrix}
1&0\\
0&1\\
1&0\\
0&-1
\end{bmatrix}\\
&=[1,1].
\end{aligned}
$$

再和 latent 点积：

$$
q_s'c_t^\top=[1,1]\begin{bmatrix}3\\1\end{bmatrix}=3+1=4.
$$

答案与完整 4 维 key 点积完全相同。于是无需为每个历史 token 展开并缓存 $k_t^C$；可把固定 $(W^{UK})^\top$ 合并进 query-side projection，再直接查询缓存的 $c_t$。

### 17.5 RoPE 为什么阻碍这次吸收

**RoPE（Rotary Position Embedding，旋转位置编码）**的人话是：token 的位置决定一个旋转角度，位置不同，Q/K 向量就旋转不同角度；点积因而能感知两个位置的相对距离。令 $R_s,R_t$ 分别是 query 位置 $s$、key 位置 $t$ 的 rotation matrix（旋转矩阵）。继续使用前文 row-vector（行向量写在矩阵左边）约定：

$$
q_s^{R}=q_sR_s,
\qquad
k_t^{R}=k_t^CR_t.
$$

旋转后点积：

$$
\begin{aligned}
q_s^R(k_t^R)^\top
&=(q_sR_s)(k_t^CR_t)^\top\\
&=q_sR_sR_t^\top k_t^{C\top}\\
&=q_sR_sR_t^\top(W^{UK})^\top c_t^\top.
\end{aligned}
$$

没有 RoPE 时，query 侧固定乘 $(W^{UK})^\top$。有 RoPE 后，中间多了：

$$
R_sR_t^\top.
$$

$R_s$ 对当前 query 固定，但 $R_t$ 随每个历史 key 的位置 $t$ 变化。若 cache 中有 $t=0$ 和 $t=1$ 两个位置：

$$
q_sR_sR_0^\top(W^{UK})^\top
\ne
q_sR_sR_1^\top(W^{UK})^\top
$$

一般不相等。因此不能为 query 只算一个固定的二维 $q_s'$，再同时点积所有 $c_t$。位置相关旋转把 key-side up-projection 与位置绑在了一起。

#### 一个完整二维例子：位置 0 与位置 1 需要不同的 absorbed query

**【补充例子】**下面是前一节 $k=cW^{UK}$、$q'=q(W^{UK})^\top$ 的二维缩小版，仍使用行向量口径。取：

$$
c=[3,1],
\qquad
W^{UK}=I=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
q=[1,0].
$$

这里 $q,c$ 都是 `[1,2]` 行向量，$W^{UK},R_0,R_1$ 都是 `[2,2]` 矩阵；最终 $q'$ 仍是 `[1,2]`，乘 $c^\top$ 的 `[2,1]` 后得到一个 scalar（标量分数）。

于是未旋转的 key 是：

$$
k=cW^{UK}=[3,1].
$$

令位置 0 不旋转；位置 1 在本文行向量约定下逆时针旋转 $90^\circ$：

$$
R_0=I=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
R_1=
\begin{bmatrix}0&1\\-1&0\end{bmatrix}.
$$

检查行向量 `[x,y]`：

$$
[x,y]R_1=[-y,x],
$$

所以 `[1,0]` 变成 `[0,1]`，确实逆时针转了 $90^\circ$。现在把当前 query 放在位置 $s=1$，因此 causal attention 允许它查看历史位置 $t=0$ 和当前位置 $t=1$，两者都满足 $t\le s$。先旋转 query：

$$
qR_1
=[1,0]
\begin{bmatrix}0&1\\-1&0\end{bmatrix}
=[0,1].
$$

**历史 key 在位置 $t=0$：**

$$
kR_0=[3,1].
$$

直接算旋转后分数：

$$
(qR_1)(kR_0)^\top
=[0,1]\begin{bmatrix}3\\1\end{bmatrix}
=1.
$$

若把 key projection 吸到 query 侧，本位置需要：

$$
\begin{aligned}
q'_0
&=qR_1R_0^\top(W^{UK})^\top\\
&=[0,1]II\\
&=[0,1].
\end{aligned}
$$

它查询 latent 的分数是：

$$
q'_0c^\top=[0,1]\begin{bmatrix}3\\1\end{bmatrix}=1.
$$

**同样的 content key 在位置 $t=1$：**

先旋转 key：

$$
kR_1
=[3,1]
\begin{bmatrix}0&1\\-1&0\end{bmatrix}
=[-1,3].
$$

直接分数变成：

$$
(qR_1)(kR_1)^\top
=[0,1]\begin{bmatrix}-1\\3\end{bmatrix}
=3.
$$

吸收进 query 侧时，这个历史位置需要：

$$
\begin{aligned}
q'_1
&=qR_1R_1^\top(W^{UK})^\top\\
&=[0,1]
\begin{bmatrix}0&-1\\1&0\end{bmatrix}I\\
&=[1,0].
\end{aligned}
$$

再查同一个 latent：

$$
q'_1c^\top
=[1,0]\begin{bmatrix}3\\1\end{bmatrix}
=3.
$$

两次都与“显式展开并旋转 key”的直接分数一致，但：

$$
q'_0=[0,1]
\ne
q'_1=[1,0].
$$

这就是不能用单一固定 $q'$ 同时查询所有历史位置的具体证据：即便 $c$ 和 $W^{UK}$ 完全相同，$R_t^\top$ 也会让每个历史位置要求不同的 query-side 变换。

### 17.6 Rotary / non-rotary 拆分思路

课程给出的解决直觉是保留少量不压进 latent 的 rotary key dimensions（旋转位置维度）：

$$
q_s=[q_s^C;q_s^R],
\qquad
k_t=[k_t^C;k_t^R],
$$

$$
\operatorname{score}(s,t)
=q_s^Ck_t^{C\top}
+q_s^Rk_t^{R\top}.
$$

- content/non-rotary 部分 $k_t^C$ 由 $c_t$ 产生，可做 query-side absorption；
- 小的 rotary 部分 $k_t^R$ 显式带位置并缓存；
- 总 cache 是“低维 $c_t^{KV}$ + 小 rotary key”，仍比完整 K/V 小。

**【课程内容｜视频 [84:42](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5082s)】**老师指出 RoPE 与 latent caching 冲突，并在 [84:56](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5096s) 给出保留 non-latent positional dimensions 的解决方向。

**边界说明：**本节用单 head、小矩阵和 row-vector 推导。完整 DeepSeek MLA 还有 query compression、多头 projection、decoupled RoPE、权重吸收和实现融合；本节保证代数直觉正确，不冒充生产实现逐行复刻。

---

## 18. MTP：让训练目标多看一个未来位置

### 18.1 Main next-token 与额外未来目标

**【课程内容｜PDF 第 59 页｜视频 [85:04](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5104s)】****MTP（Multi-Token Prediction，多 token 预测）**在普通 next-token loss 外，增加轻量模块预测更远的未来 token。

序列：

```text
位置输入： A  B  C  D  E
main目标： B  C  D  E
MTP目标：  C  D  E  ...
```

在输入 A 的位置：

- main head 预测下一个 token B；
- 一个额外 MTP module 预测再后一个 token C。

### 18.2 Tiny loss 手算

假设 main 给正确 B 的概率为 0.8：

$$
L_{\text{main}}=-\log0.8\approx0.2231.
$$

MTP module 给正确 C 的概率为 0.5：

$$
L_{\text{MTP}}=-\log0.5\approx0.6931.
$$

若额外目标权重 $\lambda_{\text{MTP}}=0.3$：

$$
\begin{aligned}
L_{\text{total}}
&=L_{\text{main}}+\lambda_{\text{MTP}}L_{\text{MTP}}\\
&=0.2231+0.3\times0.6931\\
&=0.2231+0.20793\\
&\approx0.4310.
\end{aligned}
$$

上式逐位使用已经四舍五入的 `0.2231` 和 `0.6931`；若保留对数的完整精度，结果约为 $0.43109$。

MTP 在训练时给 hidden state 一个额外信号：“不仅要足以预测 B，还要包含有助于预测 C 的信息。”它是否提高主任务、提高多少，必须看消融实验。

### 18.3 推理时可能怎样用

轻量 MTP module 可以提出未来 draft tokens（草稿 token），主模型再批量验证；若验证通过，就可能一次接受多个 token。这与 speculative decoding（投机解码）的系统思路相连。

边界：

- 训练有 MTP loss 不自动保证推理一定采用它；
- draft 接受率、验证 kernel 和额外模块成本共同决定加速；
- 辅助头可在部署中丢弃，只保留训练收益，也可作为 draft 组件；必须查具体系统。

**【视频补充】**视频 [85:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5111s) 定义一次预测多个未来 token；[85:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5125s) 提到内建 speculative decoder 的系统动机。

### 18.4 “只做 one-token-ahead”怎样理解

课件第 59 页图画了多个 MTP modules，但旁注明确写 DeepSeek V3 “only do MTP with one token ahead”。这里的 one-token-ahead 指：

- main 已负责 $t+1$；
- 只增加一个 MTP module 负责额外的 $t+2$；
- 不是同时训练一长串 $t+2,t+3,t+4,\ldots$ 辅助头。

所以“Multi-Token”描述总体方法，而该课程口径中的 V3 配置只向主 next-token 之外多走一步。不要夸大成一次可靠生成许多未来 token。

---

## 19. 把全讲串起来：决策树、常见误区与术语表

### 19.1 两条主线最后汇合在哪里

```text
主线 A：长上下文
full attention 的 token 对为 n²
        ↓
local/sparse 通过少连边省成本
linear/recurrent 通过固定状态压缩历史
        ↓
两者都会牺牲某种直接访问能力
        ↓
hybrid 用少量 full attention 补全局精确路径

主线 B：参数扩展
dense FFN 每 token 激活全部 FFN 参数
        ↓
MoE 放许多 experts，但每 token 只选 top-k
        ↓
总参数增加，主 expert FLOPs 主要由 active experts 决定
        ↓
离散路由带来梯度、负载、capacity、all-to-all 与稳定性问题
        ↓
balance、dropless kernel、expert parallel、FP32 router 等共同补救

汇合点：架构不能只看公式 FLOPs；表达力、内存、通信、硬件 kernel、
训练稳定性和数据规模必须一起算。
```

### 19.2 选架构时问什么

```text
是否必须让任意 token 一步精确访问任意历史位置？
├─ 是
│  ├─ n 尚可接受 → exact full attention + FlashAttention 类系统优化
│  └─ n 很长 → sparse/DSA 候选检索，或周期性 full layer
└─ 否，允许把历史压缩
   ├─ decode 内存/延迟最重要 → linear/recurrent/GDN 类固定状态
   └─ 又怕丢精确检索 → recurrent + 少量 full 的 hybrid

是否希望在相近每-token FFN 计算下增加总参数？
├─ 否，基础设施简单优先 → dense FFN
└─ 是
   ├─ 有 expert-parallel/all-to-all/sparse kernel 能力 → MoE
   └─ 没有 → 先算通信、负载和内存，MoE 可能不比 dense 快

已经有昂贵 dense checkpoint，是否有足够继续训练数据？
├─ 是 → 可评估 upcycling
└─ 否 → 克隆 experts 容易同质化或在小数据上过拟合
```

### 19.3 资源会计总表

| 对象 | 主计算 | 主状态/内存 | 主要风险 |
|---|---|---|---|
| full attention | $n^2(d_k+d_v)$ | score/weights $O(n^2)$；decode KV $O(n(d_k+d_v))$ | 长序列二次增长 |
| linear attention | $2nd_kd_v$ | state $d_kd_v$ | softmax 边界；固定状态压缩 |
| local attention | 约 $nwd$ | 约 $nw$ 边 | 长程信息逐层接力 |
| DSA | indexer $O(n^2d_{idx})$ + 正式 $O(nkd)$ | top-$k$ 索引和所选 K/V | indexer 漏召回 |
| dense FFN | 约 $4dm$ FLOPs/token | $2dm$ 参数 | 每 token 激活全部 FFN |
| MoE FFN | 约 $4kdm_e$ + router | $E\times2dm_e$ 总 expert 参数 | 路由塌缩、通信、慢尾 |
| MLA cache | 重建/吸收 projection | 主要缓存 $nd_c$ 加小 rotary key | RoPE 与权重吸收冲突 |

这些都是主项近似。实际还要加入 batch、heads、layers、dtype bytes、反向、optimizer state、通信协议和 kernel 利用率。

### 19.4 常见误区：错误 → 为什么错 → 正确说法

| # | 错误说法 | 为什么错 | 正确说法 |
|---:|---|---|---|
| 1 | `softmax(QKᵀ)V` 可直接换成 `Q(KᵀV)` | softmax 是作用在 $n\times n$ 分数表上的非线性归一化，不能穿过括号 | 仅在 $\rho$ 为恒等或相似度可核化并另维护归一化时重排 |
| 2 | 名字叫 linear attention，所以模型没有非线性 | linear 指对序列长度 $n$ 的主成本，不是网络函数类别 | 门控、projection、activation 仍可非线性 |
| 3 | $O(n)$ 模型一定与 full attention 效果等价 | 少算来自少连边或压缩历史，表达集合已变 | 要用任务实验证明质量，并常用 hybrid 补精确路径 |
| 4 | FlashAttention 把 full attention 从 $n^2$ 变成 $n$ | 它主要减少中间结果的 HBM 搬运/物化 | exact full attention 的允许位置对仍约 $n^2$ |
| 5 | Causal mask 把复杂度变成线性 | 三角边数是 $n(n+1)/2$ | 它只省常数约一半，仍是二次 |
| 6 | Recurrent form 只是线性 attention 的近似 | 对定义好的纯线性因果 attention，状态递归是代数精确重排 | 与普通 softmax attention 的差异发生在选择 $\rho$ 时 |
| 7 | 固定状态既省内存又不会丢信息 | 多个 token 写入同一有限矩阵，可能覆盖/抵消 | 它精确实现自己的模型，但该模型相对逐 token cache 有压缩代价 |
| 8 | $I-\beta kk^\top$ 永远是正交投影 | 只有单位归一化 $k$ 时 $kk^\top$ 才是正交 projector | 非单位 $k$ 的精确 projector 要除以 $k^\top k$ |
| 9 | DSA 已是严格 $O(n)$ | 课程 indexer 仍可能全体 Q-K 打分 | 它靠低维/低精度 indexer 与小 top-$k$ 改善常数和昂贵阶段 |
| 10 | Sparse 只要边少就自动更快 | gather/scatter、小 GEMM、padding、通信和 kernel launch 有开销 | 必须有硬件友好的 sparse/block-sparse 实现并实测 |
| 11 | MoE 有 256 experts，所以每 token 做 256 个 FFN | top-$k$ 只激活少数 routed experts | 区分总 expert 数、routed active、shared 与其他 dense 部分 |
| 12 | Active parameters 就等于 FLOPs | 参数读取一次与矩阵乘加次数不是同一单位 | 参数量数权重，FLOPs 数运算；还要算 router/activation |
| 13 | 增加 expert 数不影响任何成本 | 总权重内存、router 宽度、通信和部署都会变 | 仅在 $k$ 与 expert 大小固定时，主 expert FLOPs 近似不随 $E$ 变 |
| 14 | Top-$k$ 不可微，所以整个 MoE 没梯度 | 离散 ID 在边界不可微，选中 experts 与 gate 权重仍在连续计算图 | 对当前 mask 反传；未选分支任务梯度取决于 gate 口径，常为零 |
| 15 | Shared expert 也享受条件计算节省 | shared expert 对每个 token 都执行 | 它提供公共路径，但 FLOPs 必须计入每 token active budget |
| 16 | Load-balancing loss 越强越好 | 强迫完全均匀可能妨碍有用 specialization | 目标是防灾难性塌缩，并兼顾任务 loss 与设备吞吐 |
| 17 | DeepSeek V3 完全 aux-loss-free | online bias 替代主要 per-expert aux，但课程仍写 seq-wise aux | “aux-loss-free”有明确局部口径，不等于总目标无任何 auxiliary loss |
| 18 | Z-loss 会自动把 token 均匀分给 experts | z-loss 不含硬负载 $f_i$ | 它控制 log-sum-exp/logit 尺度；balance loss/bias 才直接调负载 |
| 19 | Dropped token 从模型中彻底消失 | 常见 residual block 仍保留输入主路 | 被丢的通常是该 expert 分支；输出如何处理取决于实现 |
| 20 | Dropless 表示没有负载问题 | 不丢 token 仍可能有热门 expert 慢尾和峰值内存 | Dropless 解决正确性/固定槽位问题，balance 仍重要 |
| 21 | MLA 就是 GQA/MQA 换名字 | GQA/MQA 共享 KV heads；MLA 把 K/V 表成低维 latent 的函数 | 两者都减 cache，但压缩轴与重建方式不同 |
| 22 | MTP 表示 V3 一次无验证生成很多 token | 课件明确 V3 只加一个额外未来步，推理还需验证设计 | 区分训练辅助目标、draft proposal 与最终接受 token |

### 19.5 术语速查

| 术语 | 一句话人话 |
|---|---|
| full attention | 每个 causal query 可直接看全部历史 key |
| local attention | 每个 query 只看固定邻域 |
| sparse attention | 只计算选中的少量 query-key 边 |
| linear attention | 对序列长度的主计算可线性增长的 attention 变体 |
| recurrent form | 用前一步状态和当前输入逐步更新 |
| state | 压缩历史、传给下一步的固定 shape 数组 |
| hybrid | 在不同层混合高效机制与 full attention |
| DSA | 用轻量 indexer 选历史 top-$k$，再做正式 attention |
| recall | 真正重要候选被 indexer 选中的比例 |
| MoE | 多套 experts 中每 token 只激活少数的层 |
| expert | 本讲中一套 FFN 参数 |
| router | 给 token-expert 配对打分并选择的模块 |
| top-$k$ | 取分数最大的 $k$ 项 |
| routed expert | 需通过 router 才执行的 expert |
| shared expert | 每个 token 都执行的公共 expert |
| fine-grained expert | 从传统大 expert 切出的更小功能块 |
| expert collapse | 大量 token 垄断性涌向少数 expert |
| load balancing | 调整路由，使 expert/设备不过热或饿死 |
| capacity factor | 每 expert 槽位相对平均负载的冗余倍率 |
| dispatch/combine | 把 token 送到 expert / 把结果送回并加权合并 |
| expert parallel | 把不同 expert 放到不同设备 |
| all-to-all | 各设备互相发送不同数据块的集体通信 |
| block-sparse GEMM | 只算有连接矩阵块的大矩阵乘 |
| router z-loss | 惩罚过大的 router log-sum-exp |
| upcycling | 从 dense checkpoint 初始化 MoE |
| MLA | 缓存低维 KV latent 的多头潜在注意力 |
| weight absorption | 把 key up-projection 代数合并到 query 侧 |
| MTP | 在 next-token 外增加更远未来 token 训练目标 |

---

## 20. 自测题（55 题）

> 建议先写出单位和 shape，再按四则运算作答。答案在第 21 节。

1. $Q\in\mathbb{R}^{n\times d_k}$、$K\in\mathbb{R}^{n\times d_k}$ 时，$QK^\top$ 的 shape 是什么？每个元素表示什么？
2. 为什么 full attention 的两次主矩阵乘约有 $n^2(d_k+d_v)$ 个标量乘法项？
3. $n=8$ 的 causal full attention 有多少条允许边？写出求和。
4. 本文定义窗口含当前位置，$n=8,w=3$ 的 causal local attention 有多少条边？
5. 第 7.3 节给位置 1 加 global 可见性后，边数为什么从 21 变成 26？
6. 写出 `(QKᵀ)V=Q(KᵀV)` 成立的关键条件，并说明普通 softmax 为什么破坏直接交换。
7. $d_k=d_v=64$ 时，两种括号的理论交叉序列长度 $n_*$ 是多少？
8. $n=4096,d_k=d_v=64$ 时，标准与线性主乘法项各是多少，比例是多少？
9. 写出 causal linear attention 的 $S_t$ 更新与读取式，并给出 $S_t$ shape。
10. 若 $S_{t-1}=\begin{bmatrix}1&2\\3&4\end{bmatrix}$、$k_t=[1,0]^\top$、$v_t=[2,-1]^\top$，纯线性更新后的 $S_t$ 是什么？
11. RetNet 式 $S_t=\gamma S_{t-1}+k_tv_t^\top$ 中，$\gamma=0,1,0.5$ 分别表示什么？
12. GDN 中 $\beta_t=0$ 时，$S_t$ 化简为什么？状态是否必然完全不变？
13. 为什么非单位 $k$ 时 $kk^\top$ 不是正交投影？精确 projector 是什么？
14. 4 层里 3 个 local 层各 21 条边、1 个 full 层 36 条边，总边数和相对 4 个 full 层的减少比例是多少？
15. DSA 的 indexer 仍做全候选打分时，为什么不能只写成严格 $O(n)$？
16. DSA 若 $q_8$ 没选中答案所在 key 3，本层会发生什么？
17. Dense FFN 的两个主矩阵为 `[d,m]` 和 `[m,d]`。忽略 bias，参数量与前向 FLOPs 近似各是什么？
18. $d=4,m=8$ 的 dense FFN 有多少参数、约多少前向 FLOPs/token？
19. $E=4$ 个同尺寸 experts、router `[4,4]`、top-1 时，总 expert+router 参数和简化 active 参数各是多少？
20. 上题若改 top-2，主 expert FLOPs/token 从多少变多少？
21. Router logits `[1,2,3,1]` 的 top-2 expert 是谁？只在选中 logits `[3,2]` 上 softmax 的两个权重约是多少？
22. 若 $E_2(x)=[2,0]$、$E_3(x)=[-1,4]$，用上题权重合并的输出是什么？
23. 为什么“全体 softmax 后 top-k 不重归一化”和“top-k 后 softmax”可选中相同 experts 却产生不同输出？
24. Shared expert 与 routed expert 在执行条件上有什么区别？
25. 原 expert 中间宽 8，fine-grained ratio 为 $1/4$，小 expert 宽多少？若 $d=4$，每个小 expert 参数多少？
26. Token-choice 与 expert-choice 各由谁做选择？各自最直接的负载优缺点是什么？
27. Hash routing 的主要优点和主要局限各是什么？
28. Global assignment 为什么可能质量/平衡更好，却很少成为大规模默认？
29. Top-k 不可微具体发生在什么位置？为什么选中 experts 仍有梯度？
30. Top-k 后只在选中集合 softmax 时，未选 expert 参数和未选 router logits 从该 token 的任务 loss 收到什么梯度？
31. Jitter 的作用是什么？为什么它不是必选配方？
32. Switch loss 中 $f_i$ 与 $P_i$ 分别是什么？
33. 对第 11.3 节的 8-token 表，写出 $f$ 与 $P$。
34. 用上题数据计算 $L_{balance}/\alpha$。
35. 在固定 hard assignment 区域，求 $\partial L/\partial p_i(x)$，并解释为何热门 expert 被更强地下调。
36. 为什么 per-device balance 不能简单省略？
37. DeepSeek V3 的“aux-loss-free”为什么不能理解为完全没有辅助损失？
38. 按本文约定，$T=10,k=2,N=4,c=1.25$ 时每 expert capacity 是多少？总槽位多少？
39. 同一 token 为什么可能在 Batch A 被保留、Batch B 被 drop？
40. Dropless kernel 解决了什么，又没有解决什么？
41. 2 台 GPU、2 个远程 BF16 activation、hidden size 8，dispatch 与 return 总 payload 多少 bytes？
42. 上题若 top-2 使 4 个 token 各有一条远程 assignment，总往返 payload 多少？
43. BF16 activation 从 $d=4096$ 下投影到 $r=512$，每 token 从多少 KiB 降到多少 KiB，缩减几倍？
44. 为什么 MegaBlocks 的 block-sparse 表达比许多小 GEMM 更可能高效？为什么仍不保证必然更快？
45. 写出 router z-loss 公式，并说明它惩罚什么。
46. `[0,0]` 与 `[10,10]` 的 softmax 是否相同？两者单 token z-loss 约是多少？
47. 为什么 z-loss 不能代替 load-balancing loss？
48. FP32 router 相比 BF16 router为什么可能更稳定？BF16 在 1 附近的间距约是多少？
49. 1000 个 fine-tuning token 均匀分给 8 个 top-1 experts，每 expert 平均得到多少 token？为什么仍可能过拟合？
50. Upcycling 时哪些参数通常复用，哪些通常新建？两个完全相同的 cloned experts 是否已经 specialization？
51. 按课件表，DeepSeek V1/V2/V3 的总参数与 active 参数分别是多少？
52. 第 56 页的 `258` 与第 35 页的 `256` 应怎样解释？同页 `V2 (671B...)` 又怎样处理？
53. 教学例中 $n=4,d_k=d_v=4,d_c=2$、BF16 时，普通 KV cache 与只存 latent 分别多少 bytes，压缩几倍？
54. 用第 17 节矩阵，验证 $qk^\top=q'c^\top=4$；为什么 RoPE 后不能为所有历史位置复用同一个 $q'$？
55. 序列 A,B,C 中，main 在 A 预测 B、一个 MTP module 在 A 预测什么？若正确概率分别 0.8、0.5 且 $\lambda=0.3$，总 loss 约多少？

---

## 21. 自测题完整答案

1. **答案：**$K^\top$ shape 为 `[d_k,n]`，所以：

   $$
   [n,d_k]\times[d_k,n]\rightarrow[n,n].
   $$

   第 $i,j$ 个元素是 $q_i^\top k_j$，表示 query 位置 $i$ 与 key 位置 $j$ 的点积分数。

2. **答案：**第一乘法 `[n,d_k]×[d_k,n]` 产生 $n^2$ 个输出，每个点积含 $d_k$ 个乘法项，所以为 $n^2d_k$。第二乘法 `[n,n]×[n,d_v]` 产生 $nd_v$ 个输出，每个沿 $n$ 求和，所以为 $n^2d_v$。相加：

   $$
   n^2d_k+n^2d_v=n^2(d_k+d_v).
   $$

3. **答案：**第 $i$ 个 query 能看 $i$ 个 key：

   $$
   1+2+3+4+5+6+7+8=36.
   $$

4. **答案：**前两行分别 1、2 条，从第 3 行到第 8 行各 3 条，共 6 行：

   $$
   1+2+6\times3=21.
   $$

5. **答案：**原 local 表中 $q_1,q_2,q_3$ 已能看 key 1；$q_4$ 到 $q_8$ 共 5 个 query 新增到 key 1 的边：

   $$
   21+5=26.
   $$

6. **答案：**裸矩阵乘满足结合律，且中间不能插入不可交换的非线性。$\rho$ 为恒等时：

   $$
   (QK^\top)V=Q(K^\top V).
   $$

   普通 softmax 要先对每行全部 key 分数取指数并归一化，`softmax(A)B` 已不是三个裸矩阵连续相乘，所以不能移动括号。可核化 attention 只有在相似度可分解、归一化另行维护时才能线性化。

7. **答案：**

   $$
   n_*=\frac{2d_kd_v}{d_k+d_v}
   =\frac{2\times64\times64}{64+64}
   =\frac{8192}{128}
   =64.
   $$

8. **答案：**

   $$
   C_{standard}=4096^2(64+64)
   =16,777,216\times128
   =2,147,483,648.
   $$

   $$
   C_{linear}=2\times4096\times64\times64
   =2\times4096\times4096
   =33,554,432.
   $$

   比例：

   $$
   2,147,483,648/33,554,432=64.
   $$

9. **答案：**

   $$
   S_t=S_{t-1}+k_tv_t^\top,
   \qquad
   y_t^\top=q_t^\top S_t.
   $$

   $k_t$ 长 $d_k$、$v_t$ 长 $d_v$，所以外积和状态 shape 都是 `[d_k,d_v]`，与序列长度 $t$ 无关。

10. **答案：**当前写入：

    $$
    k_tv_t^\top
    =\begin{bmatrix}1\\0\end{bmatrix}[2,-1]
    =\begin{bmatrix}2&-1\\0&0\end{bmatrix}.
    $$

    相加：

    $$
    S_t
    =\begin{bmatrix}1&2\\3&4\end{bmatrix}
    +\begin{bmatrix}2&-1\\0&0\end{bmatrix}
    =\begin{bmatrix}3&1\\3&4\end{bmatrix}.
    $$

11. **答案：**$\gamma=0$ 时旧状态清空，只留当前写入；$\gamma=1$ 时旧状态完整保留，等于纯累加；$\gamma=0.5$ 时旧状态每经过一步乘一半，较早信息指数衰减。

12. **答案：**令 $\beta_t=0$：

    $$
    S_t=\gamma_t(I-0\cdot kk^\top)S_{t-1}+0\cdot kv^\top
    =\gamma_tS_{t-1}.
    $$

    当前 token 不擦除也不写入，但若 $\gamma_t<1$，旧状态仍衰减；只有 $\gamma_t=1$ 时完全不变。

13. **答案：**正交投影要求消除向量长度影响。非单位 $k$ 的 $kk^\top$ 带有 $\|k\|^2$ 缩放。精确投影矩阵为：

    $$
    P_k=\frac{kk^\top}{k^\top k}.
    $$

14. **答案：**hybrid：

    $$
    3\times21+36=99.
    $$

    4 个 full：

    $$
    4\times36=144.
    $$

    少 45，比例：

    $$
    45/144=0.3125=31.25\%.
    $$

15. **答案：**若 indexer 仍给所有 query-key 候选打分，其主项为 $O(n^2d_{idx})$。DSA 的正式 attention 只做 $O(nkd)$，而 indexer 可低维、低精度，所以实测更便宜；但按 $n$ 的指数不能把第一项删除。

16. **答案：**正式 attention 的候选集合中没有 key 3，所以本层无法直接读取 $v_3$。其他层/位置若已传播该信息可能间接补救，否则精确检索失败。这是 indexer recall error 的代价。

17. **答案：**参数：

    $$
    dm+md=2dm.
    $$

    每个矩阵向量乘的乘加约记 $2dm$ FLOPs，两次共：

    $$
    4dm\ \text{FLOPs/token}.
    $$

18. **答案：**

    $$
    P=2\times4\times8=64,
    $$

    $$
    F\approx4\times4\times8=128\ \text{FLOPs/token}.
    $$

19. **答案：**4 个 expert：

    $$
    4\times64=256\ \text{parameters}.
    $$

    Router：

    $$
    4\times4=16.
    $$

    总数 $256+16=272$。top-1 每 token 经过 1 个 64 参数 expert，并使用 16 参数 router，所以本教学口径 active 为：

    $$
    64+16=80.
    $$

20. **答案：**一个 expert 主 FLOPs 为 128。top-1 是 128，top-2 是：

    $$
    2\times128=256.
    $$

    这里未加 router 与 combine 开销。

21. **答案：**最大 logits 是 3 和 2，所以选 $E_3,E_2$。在 `[3,2]` 内 softmax：

    按第 2.5 节的计算器步骤，先分别按 `eˣ`：

    $$
    e^3\approx20.0855,
    \qquad
    e^2\approx7.3891.
    $$

    分母：

    $$
    20.0855+7.3891=27.4746.
    $$

    $$
    g_3=\frac{20.0855}{27.4746}\approx0.7311,
    $$

    $$
    g_2=\frac{7.3891}{27.4746}\approx0.2689.
    $$

22. **答案：**

    $$
    \begin{aligned}
    y&=0.2689[2,0]+0.7311[-1,4]\\
    &=[0.5378,0]+[-0.7311,2.9244]\\
    &=[-0.1933,2.9244].
    \end{aligned}
    $$

23. **答案：**top-$k$ 排名对 softmax 是单调的，所以可能选中相同 ID；但全体 softmax 的分母还含未选 experts，删掉后选中权重和小于 1。选中后 softmax 用新分母，让选中权重和等于 1，因此加权输出不同。

24. **答案：**shared expert 绕过 router，每个 token 都执行；routed expert 只有进入 top-$k$ 才执行。Shared 提供公共能力，但其参数/FLOPs 必须计入每 token active budget。

25. **答案：**小 expert 宽：

    $$
    8\times\frac14=2.
    $$

    两矩阵参数：

    $$
    2dm=2\times4\times2=16.
    $$

26. **答案：**token-choice 让每个 token 选高分 experts，保证 token 获得偏好的路径，但 expert 负载可能失衡。expert-choice 让每个 expert 按容量选 token，负载天然受控，但 token 可能被多个 expert 重复选或完全漏掉。

27. **答案：**优点是无需学习、便宜、确定、易复现。局限是忽略上下文任务分数，无法学习更有用的路由，哈希碰撞还会固定绑定不相干 token。

28. **答案：**它能在全 batch 容量约束下最大化总匹配分数，兼顾质量和平衡；代价是收集全配对分数并求匹配，通信、延迟和实现复杂度远高于局部 top-$k$。

29. **答案：**当 logits 排名交换或并列时，选中 expert ID 突然跳变，离散 mask 没有普通连续导数。固定 mask 的小区域内，选中 expert 输出与 gate 权重仍是连续函数，因此可正常反向传播。

30. **答案：**未选 expert 没执行，所以其参数从该 token 的任务 loss 得 0 梯度。若先 top-$k$ 再只在选中集合 softmax，未选 router logits 也不在任务路径中，任务梯度为 0；balancing aux 等额外目标仍可能给 router 梯度。

31. **答案：**jitter 给接近并列的 experts 探索机会，避免某个分支永远无信号；但它增加随机性和不稳定来源，且后续实验并未总发现收益，所以是可验证的训练技巧而非必选数学条件。

32. **答案：**$f_i$ 是 hard argmax 后实际发给 expert $i$ 的 token 比例；$P_i$ 是 batch 中 router 对 expert $i$ 的平均软概率质量。

33. **答案：**硬计数 `[4,2,1,1]` 除以 8：

    $$
    f=[0.5,0.25,0.125,0.125].
    $$

    概率列和 `[2.75,2.25,1.70,1.30]` 除以 8：

    $$
    P=[0.34375,0.28125,0.2125,0.1625].
    $$

34. **答案：**

    $$
    \sum_if_iP_i
    =0.171875+0.0703125+0.0265625+0.0203125
    =0.2890625.
    $$

    $N=4$：

    $$
    L/\alpha=4\times0.2890625=1.15625.
    $$

35. **答案：**先不用背偏导。固定 hard assignment，若一个 token 的 $p_i(x)$ 增加 $\Delta p_i(x)=0.04$，而 $T=8$：

    $$
    \Delta P_i=\frac{0.04}{8}=0.005.
    $$

    取 $\alpha=0.1,N=4$ 和热门 expert 的 $f_i=0.5$：

    $$
    \Delta L
    =\alpha Nf_i\Delta P_i
    =0.1\times4\times0.5\times0.005
    =0.001.
    $$

    因此局部变化率：

    $$
    \frac{\Delta L}{\Delta p_i(x)}
    =\frac{0.001}{0.04}
    =0.025.
    $$

    把增量取得越来越小，才把这个比值写成偏导。一般式为：

    $$
    \frac{\partial L}{\partial p_i(x)}
    =\alpha Nf_i\frac1T
    =\frac{\alpha Nf_i}{T}.
    $$

    上述数例也给出 $0.1\times4\times0.5/8=0.025$，两条路一致。$f_i$ 越大，正 penalty gradient 越大；若参数是 $\theta$，链式法则还要乘 $\partial p_i/\partial\theta$。Gradient descent 用 $\theta_{new}=\theta_{old}-\eta\,\partial L/\partial\theta$，即减去正梯度，因此更强地下调热门 expert 方向。Softmax 使各概率耦合，但总体方向仍成立。

36. **答案：**系统吞吐由设备负载决定。多个 experts 可能共处一台设备；即使逐 expert 不完全均匀，只要设备总量平衡仍可高利用。反之某设备聚合多个热门 experts 会成为慢尾，所以需直接约束设备收发/计算。

37. **答案：**V3 用 online per-expert bias 取代主要 expert-level balancing aux，但课程明确仍保留 complementary seq-wise aux 防极端不均衡。“aux-loss-free”是局部机制名称，不是整个训练目标无 auxiliary loss。

38. **答案：**总 assignment：

    $$
    Tk=10\times2=20.
    $$

    每 expert 平均 5，加容量因子：

    $$
    C=\left\lceil1.25\times5\right\rceil
    =\lceil6.25\rceil=7.
    $$

    总槽位：

    $$
    NC=4\times7=28.
    $$

39. **答案：**capacity 在 batch 内竞争。X 的 score 不变，但 Batch B 若多出分数更高、同去该 expert 的 token，X 的容量排名会下降到 $C$ 之外而被 drop。这是由同伴组成造成的 batch dependence。

40. **答案：**dropless kernel 避免因固定 capacity 静默删除 expert 分支，改善正确性与容量浪费；它仍不能消除热门 expert 的长队列、设备慢尾、峰值内存或 all-to-all 不均衡。

41. **答案：**每 activation：

    $$
    8\times2=16\ \text{bytes}.
    $$

    两个远程 activation 的 dispatch：

    $$
    2\times16=32\ \text{bytes}.
    $$

    Return 同样 32，总计：

    $$
    32+32=64\ \text{bytes}.
    $$

42. **答案：**4 条远程 assignment：

    $$
    4\times16=64\ \text{bytes one-way}.
    $$

    往返：

    $$
    64\times2=128\ \text{bytes}.
    $$

43. **答案：**压缩前：

    $$
    4096\times2=8192\ \text{bytes}=8\ \text{KiB}.
    $$

    压缩后：

    $$
    512\times2=1024\ \text{bytes}=1\ \text{KiB}.
    $$

    缩减 $8/1=8$ 倍。

44. **答案：**block-sparse 把许多 expert 工作装进少数较大的规则 kernel，减少小 GEMM 的启动并提高硬件占用。它仍有 permutation、索引、空块、通信和 shape 对齐开销；若矩阵太小或网络主导，仍可能不比 dense 快。

45. **答案：**

    $$
    L_z=\frac1B\sum_{i=1}^B
    \left(\log\sum_{j=1}^Ne^{z_j^{(i)}}\right)^2.
    $$

    它惩罚 router log partition 的大绝对值，尤其抑制不改变 softmax 概率却让 logits 整体漂大的共同偏移。

46. **答案：**两组 logits 的两个元素分别相等，所以各自 softmax 都是 `[0.5,0.5]`。按第 2.5 节，本文 `log` 是自然对数，且 $\ln(ab)=\ln a+\ln b$：

    $$
    \ln(e^0+e^0)=\ln(1+1)=\ln2\approx0.6931,
    $$

    $$
    L_z([0,0])=(\ln2)^2
    \approx0.6931^2
    \approx0.4805.
    $$

    $$
    \begin{aligned}
    \ln(e^{10}+e^{10})
    &=\ln(2e^{10})\\
    &=\ln2+\ln(e^{10})\\
    &=0.6931+10\\
    &=10.6931,
    \end{aligned}
    $$

    $$
    L_z([10,10])=10.6931^2\approx114.3434.
    $$

47. **答案：**z-loss 只看每个 token 的 log-sum-exp，不含 expert 的 batch 硬负载 $f_i$。所有 token 仍可偏向同一 expert。Load balance 显式使用 $f_i,P_i$ 或 online load statistics 调分配。

48. **答案：**FP32 有 23 个 fraction bits，能保留很接近的 logits 和更准确的 reduction；BF16 只有 7 个 fraction bits，top-$k$ 可能因舍入改变并列关系。在 1 附近的间距约：

    $$
    2^{-7}=0.0078125.
    $$

49. **答案：**均匀时：

    $$
    1000/8=125\ \text{tokens/expert}.
    $$

    每个 expert 可能有大量参数，125 个 token 仍远不足以约束它；实际路由还可能不均，让冷门 expert 数据更少，所以容易记住训练噪声。

50. **答案：**通常复用 embedding、attention、norm、output head，并把 dense FFN 克隆成 experts；router 通常新建并随机/近均匀初始化。完全相同的 cloned experts 只复制了同一函数，没有 specialization；需扰动、不同路由数据和继续训练才能分化。

51. **答案：**按课件标题：

    - V1：16B total / 2.8B active；
    - V2：236B total / 21B active；
    - V3：671B total / 37B active。

    这些 active 是全模型每 token 路径，不只 routed expert 参数。

52. **答案：**第 56 页视觉上明确印普通数字 `258`，不是 $2^8$；第 35 页表与模型口径写 256，因此最合理是把第 56 页 `258` 披露为课件误印，不能虚构“抽取丢上标”。同页标题是 V3、参数也是 V3 的 671B/37B，但正文印 `V2`，这是另一处页内标签笔误，也应明确披露而不是隐去。

53. **答案：**普通 cache：

    $$
    n(d_k+d_v)\times2
    =4(4+4)\times2
    =64\ \text{bytes}.
    $$

    Latent：

    $$
    nd_c\times2
    =4\times2\times2
    =16\ \text{bytes}.
    $$

    压缩 $64/16=4$ 倍。本题忽略小 rotary cache。

54. **答案：**完整 key 为 `[3,1,3,-1]`：

    $$
    qk^\top=[1,2,0,1]\cdot[3,1,3,-1]
    =3+2+0-1=4.
    $$

    吸收后 $q'=q(W^{UK})^\top=[1,1]$，latent $c=[3,1]$：

    $$
    q'c^\top=1\times3+1\times1=4.
    $$

    RoPE 是 Rotary Position Embedding（旋转位置编码）：位置决定 Q/K 的旋转角。第 17.5 节二维例中取当前 query 位置 $s=1$，历史 key 位置 $t=0,1$，所以都满足 causal 条件 $t\le s$。另取 $q=[1,0]$、$c=[3,1]$、$W^{UK}=I$、$R_0=I$、$R_1=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$。先算：

    $$
    qR_1=[0,1].
    $$

    对 $t=0$，直接完整 key 路径是：

    $$
    kR_0=[3,1],
    \qquad
    (qR_1)(kR_0)^\top=1.
    $$

    吸收到 query 侧也得到：

    $$
    q'_0=qR_1R_0^\top(W^{UK})^\top=[0,1],
    \qquad
    q'_0c^\top=1.
    $$

    对 $t=1$，直接完整 key 路径是：

    $$
    kR_1=[-1,3],
    \qquad
    (qR_1)(kR_1)^\top=3.
    $$

    吸收到 query 侧同样得到：

    $$
    q'_1=qR_1R_1^\top(W^{UK})^\top=[1,0],
    \qquad
    q'_1c^\top=3.
    $$

    因为 $q'_0\ne q'_1$，单一固定 $q'$ 无法同时复用到所有历史位置；问题来自随 $t$ 改变的 $R_t^\top$。

55. **答案：**额外 MTP module 在 A 预测 C，即主目标之外再远一步。按第 2.5 节，在计算器输入概率后按 `ln`，并取负号：

    $$
    -\log0.8\approx0.2231,
    \qquad
    -\log0.5\approx0.6931.
    $$

    $$
    L=0.2231+0.3\times0.6931
    =0.2231+0.20793
    \approx0.4310.
    $$

    使用未截断的自然对数计算约为 $0.43109$。

---

## 22. 视频时间导航（人工英文字幕）

> 下列链接使用本笔记其他位置尚未使用的秒点；点击即可跳转。右栏给出对应笔记章节，而不是要求按视频顺序重新学习。

| 时间 | 视频内容 | 笔记 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5s) | 开场 | §1 |
| [01:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=60s) | 本讲目标：随序列长度线性而非二次 | §1、§3 |
| [02:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=120s) | 上下文增长带来的工作负载 | §1–2 |
| [03:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=183s) | local/global/hybrid 基础工具 | §7 |
| [04:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=241s) | 系统优化与硬件友好执行 | §2、§13 |
| [05:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=301s) | 能否让 attention 对 $n$ 线性 | §3 |
| [06:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=361s) | 矩阵乘顺序与结合律 | §3 |
| [08:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=483s) | 从线性复杂度转向递归状态 | §4 |
| [09:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=542s) | dense 与 recurrent 两种执行视图 | §4 |
| [10:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=600s) | MiniMax 混合线性 attention | §6–7 |
| [11:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=663s) | 过渡到 Mamba-2 | §5 |
| [12:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=720s) | state-space 来源与状态更新 | §5 |
| [14:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=840s) | Nemotron hybrid 案例 | §5、§7 |
| [15:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=902s) | Gated DeltaNet 过渡 | §6 |
| [16:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=961s) | $\beta$ 写入门的架构意义 | §6 |
| [17:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1021s) | 定向擦除与更新 | §6 |
| [19:06](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1146s) | hybrid 受控研究仍有限 | §7 |
| [20:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1201s) | hybrid 比例继续讨论 | §7 |
| [22:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1321s) | Mamba-2 当前 value skip 的问答 | §5 |
| [23:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1380s) | DSA 作为另一条高效 attention 路线 | §7 |
| [24:05](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1445s) | Indexer 的 ReLU/打分细节 | §7 |
| [25:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1503s) | 长上下文阶段适配 indexer | §7 |
| [26:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1560s) | GLM5/DSA 课程案例 | §7 |
| [27:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1620s) | Indexer 仍二次但可便宜 | §7 |
| [28:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1681s) | 课堂问答：低维、低精度与常数因子 | §7 |
| [29:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1741s) | 课堂问答：continued pretraining 阶段 | §7 |
| [30:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1801s) | 课堂问答：移除 softmax 与稳定性 | §3、§14 |
| [31:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1861s) | 未来架构可能整合多种技巧 | §19 |
| [32:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1922s) | 课堂问答：低精度 softmax 风险 | §14 |
| [33:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1980s) | Full attention 的表达力 | §4、§7 |
| [34:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2040s) | 课堂问答：context 与 state 大小交换 | §4 |
| [35:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2100s) | 为什么需要理解 MoE | §8 |
| [36:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2160s) | 多个同尺寸 FFN、只激活一个 | §8 |
| [37:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2222s) | MoE 普及与 sparse parameters | §8 |
| [38:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2281s) | 固定训练 compute 的曲线解释 | §8 |
| [39:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2342s) | 模型案例与 active parameters | §8 |
| [40:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2402s) | MoE 增加并行切分轴 | §13 |
| [42:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2521s) | 课堂问答：expert parallel 通信瓶颈 | §13 |
| [43:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2581s) | 训练也只激活 $k$ 个 experts | §8、§10 |
| [46:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2762s) | MoE 基础设施与训练复杂度 | §8、§13–14 |
| [47:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2822s) | Attention-head MoE 为何较少见 | §8 |
| [48:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2881s) | Router、expert size、training 三个设计轴 | §9–11 |
| [49:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2940s) | Token-choice top-$k$ | §10 |
| [50:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3002s) | 线性 router 内积 | §9–10 |
| [51:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3063s) | Hash baseline 与 RL 过渡 | §10 |
| [52:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3120s) | 启发式 recipe 与全局 assignment | §10 |
| [53:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3181s) | Top-$k$ 成为共识路由 | §9 |
| [54:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3240s) | Gate 怎样由 router 权重学习 | §9 |
| [55:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3300s) | Shared expert 动机 | §9 |
| [56:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3361s) | Fine-grained + shared ablation | §9 |
| [57:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3420s) | 近期 MoE expert 配置表 | §16 |
| [58:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3481s) | 课堂问答：shared expert 的并行 | §9、§13 |
| [59:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3541s) | 训练保持稀疏的必要性 | §10 |
| [60:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3600s) | RL router 与高方差 | §10 |
| [61:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3661s) | 随机 perturbation router | §10 |
| [63:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3783s) | Jitter 后续消融 | §10 |
| [64:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3843s) | Expert collapse/starvation | §11 |
| [65:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3902s) | Switch loss 的启发式性质 | §11 |
| [66:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3960s) | DeepSeek v1/v2 balancing | §11、§16 |
| [67:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4021s) | 逐设备聚合负载 | §11 |
| [68:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4080s) | 删除 balance loss 的消融 | §11 |
| [69:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4140s) | 冷门 experts 等于浪费参数 | §11 |
| [70:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4201s) | Top-$k$ + aux 的更广泛用途 | §10–11 |
| [71:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4261s) | 课堂问答：experts 是否有可读语义 | §19 |
| [72:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4322s) | Data/model/expert parallel | §13 |
| [73:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4380s) | 小矩阵乘到块稀疏乘 | §13 |
| [74:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4440s) | Nemotron activation down-project | §13 |
| [75:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4501s) | MoE batch stochasticity 过渡 | §12 |
| [76:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4560s) | Drop 分支返回零的旧实现 | §12 |
| [77:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4621s) | Router 新增 softmax 风险 | §14 |
| [78:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4683s) | Router z-loss 曲线 | §14 |
| [79:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4741s) | Fine-tune attention/non-MoE 层 | §14 |
| [80:04](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4804s) | 从 dense model upcycle | §15 |
| [81:08](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4868s) | MiniCPM upcycling 案例 | §15 |
| [82:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4921s) | 过渡到 DeepSeek v1/v2/v3 | §16 |
| [83:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4981s) | V2 device/communication 机制 | §16 |
| [84:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5041s) | MLA：Q/K/V 来自 latent | §17 |
| [85:01](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5101s) | 从 MLA 过渡到 MTP | §18 |
| [86:02](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5162s) | 总结：简单路由可在规模上工作 | §19 |

---

## 23. PDF 全页覆盖与来源边界

### 23.1 60 页覆盖表

| PDF 页 | 视觉内容 | 笔记落点 |
|---:|---|---|
| 1 | 标题：Attention Alternatives and Mixtures of Experts | 页首、§1 |
| 2 | Context window 增长、attention 成本动机 | §1.6、§2 |
| 3 | local+global、系统优化基础工具 | §1.6、§7、§13 |
| 4 | 线性 attention 结合律和两种复杂度 | §2–3 |
| 5 | recurrent form、训练/推理 duality、RetNet 注记 | §4–5 |
| 6 | MiniMax M1 7:1 hybrid | §6.8、§7.4 |
| 7 | Mamba-2 的 $\gamma_t$ 与 $D$ skip | §5 |
| 8 | Nemotron 3 Mamba/attention hybrid | §5.5、§7.4 |
| 9 | Gated DeltaNet 的 $\beta$ 与定向擦除 | §6 |
| 10 | Qwen 3.5/Qwen Next 3:1 GDN/attention | §6.8、§7.4 |
| 11 | Hybrid performance 与受控实验不足 | §7.4 |
| 12 | Sparse adaptation、indexer、post-hoc | §7.5–7.9 |
| 13 | DSA/GLM5 性能图 | §7.7–7.9 |
| 14 | MoE 章节转场 | §8 |
| 15 | 用 experts + selector 替换 FFN | §8.1–8.3 |
| 16 | 固定 FLOPs、增加 experts 的曲线 | §8.6 |
| 17 | OlMoE 训练速度/质量图 | §8.6 |
| 18 | MoE 与 dense active-parameter 比较 | §8.3–8.6 |
| 19 | Expert parallel 的自然切分 | §13.1 |
| 20 | 西方开源 MoE 模型快照 | §8.6、§16.7 |
| 21 | Qwen MoE 历史结果 | §8.6、§15.4 |
| 22 | DeepSeek MoE 消融 | §9.6–9.7、§16 |
| 23 | DeepSeek V3 快照 | §16 |
| 24 | 基础设施、启发式目标、稳定性挑战 | §8.6、§10–14 |
| 25 | 通常替换 MLP，较少替换 attention heads | §8.1、§22 导航 |
| 26 | Router、expert size、training 三个设计轴 | §9–11 |
| 27 | token/expert/global routing 概览 | §10.1–10.5 |
| 28 | token-choice 与 expert-choice 对比 | §10.2–10.3 |
| 29 | top-k 与 hash 变体 | §10.2、§10.4 |
| 30 | RL 与 linear assignment | §10.5–10.6 |
| 31 | Top-k gate 公式与顺序变体 | §9.1–9.5 |
| 32 | Shared + fine-grained experts | §9.6–9.7 |
| 33 | DeepSeek shared/fine-grained 消融 | §9.7 |
| 34 | OlMoE shared/fine-grained 消融 | §9.7 |
| 35 | 近期 MoE routed/active/shared/ratio 表 | §16.1–16.3 |
| 36 | 离散路由训练的三类方案 | §10.6–10.10、§11 |
| 37 | REINFORCE router | §10.6 |
| 38 | Gaussian stochastic routing | §10.10 |
| 39 | Switch multiplicative jitter | §10.10 |
| 40 | Switch load-balancing loss 与导数 | §11.2–11.5 |
| 41 | DeepSeek v1/v2 expert + device balance | §11.6、§16.4–16.5 |
| 42 | V3 online bias 与 seq-wise aux 限定 | §11.7、§16.6 |
| 43 | 移除 balancing 的 OlMoE 消融 | §11.8 |
| 44 | Expert/data/model parallel 图 | §13.1–13.4 |
| 45 | 小 GEMM、块对角与 block-sparse | §13.5–13.6 |
| 46 | LatentMoE activation down-project | §13.7–13.8 |
| 47 | Batch-level token dropping 随机性 | §12.3–12.5 |
| 48 | Router FP32 与 z-loss 公式 | §14.1–14.4 |
| 49 | 移除 router z-loss 曲线 | §14.4 |
| 50 | Sparse MoE fine-tuning overfit | §14.5–14.6 |
| 51 | Dense checkpoint upcycling 概念 | §15.1–15.3 |
| 52 | MiniCPM upcycling | §15.4 |
| 53 | Qwen MoE upcycling | §15.4 |
| 54 | DeepSeek MoE V1 配置 | §16.1–16.4 |
| 55 | DeepSeek V2、top-M、communication balance | §16.1、§16.5 |
| 56 | DeepSeek V3 gate/balance；页内排版冲突 | §16.1–16.6 |
| 57 | MLA 总体架构图 | §17.1–17.3 |
| 58 | MLA weight absorption 与 RoPE 冲突 | §17.4–17.6 |
| 59 | MTP/DeepSeek/EAGLE 图与 one-step 注记 | §18 |
| 60 | MoE sparsity、离散路由、成本总结 | §19 |

### 23.2 PDF 视觉核验记录

- `lecture_04.pdf` 共 60 页，使用 `pypdf` 提取文字；
- 使用 `pypdfium2` 把 60/60 页渲染为 PNG，并制作 6 张每张 10 页的 contact sheets；
- 对全部 contact sheets 逐页检查标题、图表、公式区域是否缺失或裁切；
- 原分辨率重点检查页 2–5、7、9、31、35、39–49、52、54–59；
- 特别核对第 40 页 Switch 公式、第 42 页并非完全 aux-free、第 54–56 页模型数字、第 58 页 RoPE/MLA 等式和第 59 页 MTP 注记。

### 23.3 课程来源

- [Stanford CS336 官方 Lecture 4 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)
- [Stanford Online Lecture 4 视频](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)
- 视频人工字幕轨：YouTube `English (United States)`，语言代码 `en-US`，1938 segments；末段 86:13–约 86:15。

【课程来源边界】标为【课程内容】的结论来自 PDF、人工字幕或两者交叉；模型名称、参数表、实验曲线和“当前流行”的描述是 Spring 2026 课堂时点快照。本笔记没有假装每句课堂评论都逐字来自论文，也没有用第三方课程笔记替代原资料。

### 23.4 补充来源：只列论文或官方技术报告

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Transformers are SSMs: Mamba-2 / Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [DeepSeek-V3.2 Technical Report（DSA）](https://arxiv.org/abs/2512.02556)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
- [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

【补充来源边界】第 2–21 节中标为【补充理解】或【补充例子】的逐步算术、$n=8$ 路由表、router 小向量、capacity batches、通信字节、MLA 小矩阵和 55 道自测均为本笔记教学构造，不冒充论文原实验。论文链接用于核对机制与术语；若课程简化式与完整论文实现不同，正文已经明确标出简化范围。

### 23.5 学完后应能做到

- 从 shape 推出 full/linear attention 的计算与状态大小；
- 用同一组数字验证 dense、recurrent 两种线性 attention 形式；
- 解释 gamma、beta、directional erase 与 hybrid 的因果关系；
- 把 MoE 总参数、active 参数、FLOPs、设备负载、通信分开算；
- 从 logits 手算不同 top-$k$ gate 顺序与 weighted output；
- 手算 Switch loss、capacity、drop 与 all-to-all bytes；
- 区分 router z-loss、load balance、online bias 的不同目标；
- 解释 upcycling、DeepSeek MoE 演进、MLA weight absorption/RoPE 冲突和 MTP 辅助目标；
- 看到新架构时，不只问“大 O 是多少”，还会问表达力、内存层级、硬件 kernel、网络和训练数据。
