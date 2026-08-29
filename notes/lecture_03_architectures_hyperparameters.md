# CS336 Lecture 3：Transformer 架构、超参数与稳定性

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：John Hewitt  
> 视频：[Lecture 3: Architectures, Hyperparameters](https://www.youtube.com/watch?v=lVynu4bo1rY)（89:09）  
> 官方讲义：[lecture_03.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)（67 页）

> **资料核对说明：**本笔记逐页检查了官方 PDF 的 67 页渲染结果；公式、架构图、模型对比表和曲线页另以高分辨率复核。视频主字幕使用 YouTube 的人工轨 `English (United States)`，共 2020 个片段，最后一段从 89:08 开始、约在 89:09 结束。`English (auto-generated)` 轨存在，但不作为主字幕。笔记不是字幕翻译，而是把讲义、视频口头说明和必要前置知识重新编成一条可独立学习的路线。

本讲的来源标签：

- **【课程】**：官方 PDF 和视频共同呈现的主内容；
- **【视频补充】**：幻灯片上没有、老师在视频里说出的解释、限定或课堂问答；
- **【补充解释】**：为零基础读者补出的定义、推导和连接步骤；
- **【补充例子】**：课程之外的新手算例子；
- **【延伸】**：不影响本讲主线，第一次阅读可以跳过。

同一个词若在论文里有多种写法，本笔记会先给公式，再说明名字，避免只背术语。例如“非残差 post-norm”在不同模型报告中的具体组合并不总是完全相同，不能只凭名字猜数据流。

---

## 0. 五分钟复习卡

> **第一次学习请直接跳到第 1 节。**这一节是复习索引，会提前使用尚未解释的词。

### 0.1 一句话主线

现代语言模型仍是 Transformer，但大家不断调整 **norm 放哪里、FFN 用什么门控、位置怎样编码、头怎样共享、局部和全局注意力怎样交替**；每个选择都在表达能力、训练稳定性、显存、通信量和推理速度之间做交换。

### 0.2 全讲因果链

```text
token ID 只是整数，必须先查 embedding 变成向量
        ↓
每层用 attention 混合 token 之间的信息
再用 FFN 独立加工每个 token 的特征
        ↓
residual connection 保留一条“旧信息直通路”
norm 控制数值尺度，放置位置影响梯度和直通路
        ↓
激活函数与 gated FFN 改变非线性和参数分配
位置编码让 attention 知道顺序；RoPE 把相对位移写进 Q·K
        ↓
d_model、d_ff、层数、头数、词表大小共同决定参数、shape 和系统成本
        ↓
z-loss、QK norm、soft-capping 控制 softmax 附近的数值风险
        ↓
MHA → GQA/MQA 缩小 KV cache；滑动窗口减少长上下文成本
```

### 0.3 先记结论，复习时再展开

1. **pre-norm 的基本式**：

   $`y=x+F(\mathrm{Norm}(x)).`$

   残差主路中的 $`x`$ 可以不经过 norm，直接加到下一状态。

2. **RMSNorm** 不减均值，只按均方根缩放：

   $`\mathrm{RMSNorm}(x)_i =\gamma_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}}.`$

3. 普通两层 FFN 的主权重约为：

   $`2d_{\text{model}}d_{\text{ff}}.`$

   gated FFN 有 up、gate、down 三个矩阵，约为：

   $`3d_{\text{model}}d_{\text{ff,gated}}.`$

   同预算令两式相等，所以 $`d_{\text{ff,gated}}\approx\frac23d_{\text{ff}}`$。

4. 若有 $`H_q`$ 个 query heads、$`H_{kv}`$ 个 KV heads、每头宽 $`d_h`$，单层 BF16 KV cache 大小为：

   $`2BT H_{kv}d_h\times 2\ \text{bytes}.`$

5. 长度为 $`n`$ 的 full attention 要看约 $`n^2`$ 个位置对；窗口宽 $`w`$ 的 sliding-window attention 只看约 $`nw`$ 个位置对。

这些式子会在后面逐符号解释并手算；第一次阅读不要靠背诵通过。

### 0.4 稳定性、cache 与长上下文速查

- 课件把相对位移命名为 $`\delta=i-j`$；按本文“query 在点积左边、key 在右边”的列向量约定，RoPE 中间矩阵是 $`R(j-i)=R(-\delta)`$；
- stable softmax 先做 `logits - max(logits)`；这是数学等价的数值实现；
- z-loss 是训练 objective 的附加项，QK norm 与 soft-cap 则会改变 forward；
- MHA 有 $`H_{kv}=H_q`$，MQA 有 $`H_{kv}=1`$，GQA 介于二者；$`H_{kv}=1`$ **不等于** $`d_h=1`$；
- 单层 KV cache 元素数为 $`2BnH_{kv}d_h`$；**dtype（data type，数据类型）**是 tensor 每个元素采用的数字存储格式，例如 BF16；换 bytes 还要乘该 dtype 的每元素字节数；
- prefill 可并行处理 prompt，incremental decode 必须逐 token，并反复读取权重与历史 K/V；
- local attention 的长程信息要逐层接力；周期性 full layer 用较高成本换短的全局路径。

---

## 1. 开始之前：最少前置知识

### 1.1 标量、向量、矩阵和 tensor

**【补充解释】**先把“数字装在什么盒子里”分清：

- **scalar（标量）**：一个数字，例如温度 `20`；shape 可写成 `[]`；
- **vector（向量）**：一排数字，例如 `[1, 3, -2]`；shape 是 `[3]`；
- **matrix（矩阵）**：按行、列排好的数字，例如 2 行 3 列；shape 是 `[2, 3]`；
- **tensor（张量）**：标量、向量、矩阵以及更高维数组的总称。

`shape=[B,T,d]` 的意思不是“有三个数字”，而是有三条轴：

- $`B`$：batch size，一次并行处理多少条序列；
- $`T`$：sequence length，每条序列有多少个 token 位置；
- $`d`$：每个 token 用多少个特征数字表示。

总元素数是三条轴长度相乘：$`B\times T\times d`$。

### 1.2 token、embedding、hidden state、logit

**【补充解释】**本讲反复使用五个词：

- **token（词元）**：分词器切出的单位，例如一个字、半个单词或标点；
- **token ID**：词表给 token 分配的整数编号；编号本身没有“大小含义”；
- **embedding（嵌入）**：用一条可学习向量表示 token；
- **hidden state（隐藏状态）**：模型中间对某个 token 的当前表示；
- **logit（未归一化分数）**：模型给词表中每个候选 token 的原始分数，尚不是概率。

假设词表只有 6 个 token，ID `2` 只是“查 embedding 表的第 2 行”。它不表示该 token 是 ID `1` 的两倍。

### 1.3 只需要这一点矩阵乘

向量 $`x`$ 的 shape 是 `[d_in]`，矩阵 $`W`$ 的 shape 是 `[d_in,d_out]`，那么：

```math
y=xW
```

的 shape 是 `[d_out]`。中间的 $`d_{in}`$ 必须对上。

**【补充例子】**令：

```math
x=[2,3],\qquad
W=\begin{bmatrix}1&4\\5&2\end{bmatrix}.
```

第一个输出取 $`W`$ 的第一列：

```math
y_1=2\times1+3\times5=17.
```

第二个输出取第二列：

```math
y_2=2\times4+3\times2=14.
```

所以 $`y=[17,14]`$。shape 从 `[2] @ [2,2]` 变成 `[2]`。

### 1.4 “逐元素”和“混合位置”是两件事

**【补充解释】**若函数对向量的每个数字分别做同一件事，例如：

```math
\mathrm{ReLU}([-2,3])=[0,3],
```

这叫 **elementwise（逐元素）**。它没有把第 1 个 token 的信息搬到第 2 个 token。

attention（注意力）才会让不同 token 位置彼此取信息。FFN（feed-forward network，前馈网络）通常在每个位置分别运行，但会混合该位置内部的特征维。

### 1.5 概率和 softmax 的最小概念

**【补充解释】**softmax 把一组任意实数 logits 变成总和为 1 的正数：

```math
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
```

- $`z_i`$：第 $`i`$ 个候选的 logit；
- $`e^{z_i}`$：把分数变为正数；
- 分母：把全部正数相加；
- $`p_i`$：第 $`i`$ 个候选的概率。

若 logits 是 `[0,0]`，两个指数都是 1，分母是 2，所以概率是 `[0.5,0.5]`。

### 1.6 本讲符号表

| 符号 | 意义 | 单位/shape |
|---|---|---|
| $`B`$ | batch size | 条序列 |
| $`T`$ 或 $`n`$ | sequence length | token 个数 |
| $`V`$ | vocabulary size | token 种类数 |
| $`L`$ | Transformer block 数 | 层 |
| $`d`$ 或 $`d_{\text{model}}`$ | 每个 token 的主隐藏宽度 | features/token |
| $`d_{ff}`$ | FFN 的中间宽度 | features/token |
| $`H`$ 或 $`H_q`$ | query head 数 | heads |
| $`H_{kv}`$ | key/value head 数 | heads |
| $`d_h`$ | 每个 attention head 的宽度 | features/head |

除非明确说明，本笔记中的 $`T`$ 指 token 数，不表示矩阵转置。矩阵转置写成上标 $`\mathsf T`$，例如 $`K^{\mathsf T}`$。

---

## 2. 整讲地图：我们到底在改 Transformer 的哪里

### 2.1 一个语言模型的外壳

**【课程，PDF 第 2–4 页】**课程先把原始 Transformer 与现代常见变体并排比较。无论细节怎样改，decoder-only 语言模型的主干可以画成：

```text
token IDs [B,T]
    ↓ 查 embedding 表
初始 hidden states X0 [B,T,d]
    ↓
Transformer block 1：attention + FFN + residual + norm
    ↓
Transformer block 2
    ↓
...
    ↓
Transformer block L 的输出 XL [B,T,d]
    ↓ 最终 norm（具体模型可能不同）
    ↓ 乘输出权重
logits [B,T,V]
    ↓ 对最后一轴 V 做 softmax
下一个 token 的概率 [B,T,V]
```

模型并不是把整个序列压成一个数字。每个位置都保留一条宽度为 $`d`$ 的表示，并为每个位置给出 $`V`$ 个候选 token 分数。

### 2.2 block 里面的两种“加工厂”

一个标准 block 的两个大部件：

1. **self-attention（自注意力）**：当前序列里的 token 互相看；作用重点是“位置之间传信息”；
2. **FFN/MLP（前馈网络/多层感知机）**：每个位置独立通过相同的小网络；作用重点是“加工该位置的特征”。

两者通常都被 residual connection（残差连接）包住。残差连接就是把部件输出加回输入：

```math
y=x+F(x).
```

$`F`$ 可以代表 attention 或 FFN。这个加法要求 $`x`$ 和 $`F(x)`$ 的 shape 完全相同。

### 2.3 本讲的改动可归成六个旋钮

**【课程】**67 页材料看似列了很多模型名，实际围绕六类旋钮：

| 旋钮 | 它回答的问题 | 主要代价/收益 |
|---|---|---|
| norm 位置与种类 | 数值在哪里被缩放？ | 梯度稳定、数据搬运、表达方式 |
| FFN 激活与门控 | 特征怎样被筛选？ | 参数量、非线性、效果 |
| serial/parallel | attention 与 FFN 谁先谁后？ | 语义深度、融合机会、速度 |
| 位置编码 | 模型怎样知道先后与距离？ | 长度泛化、相对位置信号 |
| 超参数 | 宽度、深度、头、词表怎样配？ | 参数、质量、并行和通信 |
| attention 变体 | K/V 是否共享、看多远？ | KV cache、带宽、长上下文 |

**【视频补充，约 [00:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5s)】**老师强调，本讲更像一份跨模型“调查地图”。最可靠的学习方式仍是自己实现和跑小实验；模型报告里的单次结果不能自动变成永恒定律。

### 2.4 原始版本与现代版本的第一眼差别

**【课程，PDF 第 3–4 页】**课程用一个简化对比建立方向：

| 部件 | 2017 原始 Transformer 的典型选择 | 课程总结的现代常见选择 |
|---|---|---|
| norm 放置 | post-norm | pre-norm，或额外 non-residual post-norm |
| norm 类型 | LayerNorm | RMSNorm 常见 |
| FFN | ReLU，普通两矩阵 FFN | SwiGLU 等 gated FFN 常见 |
| 位置 | sinusoidal absolute position | RoPE 常见 |
| bias | 线性层常带 bias | 很多模型删除 bias |

这是“常见趋势”，不是 API 规定。BERT 等模型保留 post-norm；一些现代模型也会混合不同位置的 norm。后文会把每项拆开。

---

## 3. 用一个微型 Transformer 把所有 shape 对上

### 3.1 先规定一个可以手数的世界

**【补充例子】**设：

- batch size $`B=1`$；
- 序列长度 $`T=3`$；
- 词表大小 $`V=6`$；
- 模型宽度 $`d=4`$；
- attention 头数 $`H=2`$；
- 每头宽度 $`d_h=2`$；
- FFN 中间宽度 $`d_{ff}=8`$；
- block 数 $`L=2`$。

检查最常见的头宽关系：

```math
H\times d_h=2\times2=4=d.
```

现在输入一条 3-token 序列：

```math
\text{token IDs}=[2,5,1].
```

因为 $`B=1,T=3`$，放进 tensor 后 shape 是 `[1,3]`。其中元素 `[0,1]=5` 的意思是“第 0 条序列、第 1 个位置的 token ID 是 5”。

### 3.2 embedding：从编号查到向量

embedding 表 $`E`$ 的 shape 是 `[V,d]=[6,4]`：6 行对应 6 种 token，每行 4 个可学习数字。

假设表中相关三行是：

```math
E_2=[1,0,1,0],\quad
E_5=[0,2,0,2],\quad
E_1=[1,1,1,1].
```

查表后：

```math
X_0=
\begin{bmatrix}
1&0&1&0\\
0&2&0&2\\
1&1&1&1
\end{bmatrix}.
```

忽略 batch 轴看是 `[3,4]`；带 batch 轴是 `[1,3,4]`。元素 `X0[0,1,3]=2` 表示：第 0 条序列、第 1 个 token 位置、第 3 个隐藏特征的值为 2。

### 3.3 attention 的 Q、K、V：shape 先不丢

**Q（query，查询）**、**K（key，键）**、**V（value，值）**都是由 $`X`$ 乘不同权重得到。**符号复用警告：**这里的张量 $`V`$ 是 attention 的 value；本讲别处单独出现的标量 $`V`$ 是 vocabulary size（词表大小）。必须根据 shape 和上下文区分。

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
```

本例每个 $`W`$ 都是 `[d,d]=[4,4]`：

```text
X       [1,3,4]
W_Q     [4,4]
Q       [1,3,4]
```

然后把最后的宽度 4 拆成 `2 heads × 2 features/head`：

```text
Q before split    [B,T,d]     = [1,3,4]
Q after split     [B,H,T,d_h] = [1,2,3,2]
K after split     [1,2,3,2]
V after split     [1,2,3,2]
```

这里没有删除或复制数字，因为拆分前后每条序列都有：

```math
T\times d=3\times4=12
```

个 Q 元素；拆分后也是：

```math
H\times T\times d_h=2\times3\times2=12.
```

### 3.4 attention score 与 causal mask

每个 query 会和同一 head 内的每个 key 做点积：

```math
S=\frac{QK^{\mathsf T}}{\sqrt{d_h}}.
```

- $`S`$：attention logits；
- $`K^{\mathsf T}`$：把 key 的 token 轴转过来；
- $`\sqrt{d_h}`$：缩放因子，防止点积随维度增大得太快。

shape：

```text
Q                 [1,2,3,2]
K^T               [1,2,2,3]
S = Q @ K^T       [1,2,3,3]
```

最后两个 `3,3` 的含义是：3 个 query 位置 × 3 个 key 位置。`S[0,1,2,0]` 是第 0 条序列、第 1 个 head 中，位置 2 看位置 0 的分数。

decoder-only 模型还加 **causal mask（因果遮罩）**：位置 $`t`$ 不能看未来位置。3-token 的可见关系是：

```text
query 0 可看 key 0
query 1 可看 key 0,1
query 2 可看 key 0,1,2
```

对每个 query 的 3 个 logits 做 softmax 后，shape 仍是 `[1,2,3,3]`。再乘 $`V`$：

```text
attention probabilities    [1,2,3,3]
V                           [1,2,3,2]
weighted values             [1,2,3,2]
merge two heads             [1,3,4]
output projection W_O       [4,4]
attention output            [1,3,4]
```

最终又回到 `[B,T,d]`，才能与残差输入逐元素相加。

### 3.5 residual 加法不会改变 shape

设 attention 输出是 $`A`$，则一次残差加法是：

```math
X_{\text{after-attn}}=X+A.
```

```text
X                       [1,3,4]
A                       [1,3,4]
X_after_attn            [1,3,4]
```

若 $`A`$ 是 `[1,3,8]`，就不能直接相加；这也是 attention 最后需要 output projection 回到宽度 $`d`$ 的原因。

### 3.6 FFN：位置不混，特征先变宽再变窄

普通 FFN 可写成：

```math
\mathrm{FFN}(x)=\phi(xW_1)W_2.
```

- $`W_1`$ shape `[d,d_ff]=[4,8]`；
- $`\phi`$ 是逐元素激活函数；
- $`W_2`$ shape `[d_ff,d]=[8,4]`。

完整 shape：

```text
block input                [1,3,4]
after W1                   [1,3,8]
after activation           [1,3,8]
after W2                   [1,3,4]
after residual addition    [1,3,4]
```

`T=3` 没变，说明 FFN 没有把不同 token 位置混在一起；它对三个位置使用同一组 $`W_1,W_2`$。

### 3.7 两层以后怎样得到词表概率

本例有 $`L=2`$ 个 block。两层结束仍是：

```text
X2                    [B,T,d] = [1,3,4]
output weight         [d,V]   = [4,6]
logits                [B,T,V] = [1,3,6]
softmax over V        [1,3,6]
```

`logits[0,2,:]` 是“看完前 3 个输入位置后，对 6 个词表 token 的分数”。训练时每个位置通常都有 next-token 目标；生成时常只读取最后一个位置的分布来采样下一个 token。

### 3.8 一条完整 shape 复述

```text
[1,3] token IDs
 → [1,3,4] embeddings
 → [1,2,3,2] Q/K/V per head
 → [1,2,3,3] attention scores
 → [1,2,3,2] weighted values
 → [1,3,4] merged attention output
 → [1,3,8] FFN hidden
 → [1,3,4] block output
 → 重复第 2 个 block，仍为 [1,3,4]
 → [1,3,6] vocabulary logits/probabilities
```

如果这条链还不能从头解释，先不要进入架构变体；后面的每个名字都只是在改这条链里的某一段。

---

## 4. 原始 Transformer 与课程的现代基线

### 4.1 原始版本不是“错误版本”

**【课程，PDF 第 3 页】**2017 年 Transformer 的代表性选择包括：

- sinusoidal positional encoding（正弦位置编码）；
- ReLU 激活的普通 FFN；
- post-LayerNorm；
- 带 bias 的线性层。

这些选择成功建立了 Transformer。后来改变它们，不表示原方案数学错误，而是规模、训练配方和硬件变化后，人们找到了更稳定或更高效的默认值。

### 4.2 课程的现代简化版本

**【课程，PDF 第 4 页】**课程用下列组合作为现代 decoder-only LM 的教学起点：

- pre-norm；
- RMSNorm；
- RoPE（Rotary Position Embedding，旋转位置编码）；
- SwiGLU gated FFN；
- 多数线性层不使用 bias。

后面的模型调查页展示的是趋势，不是“所有现代模型都一样”。

### 4.3 为什么不能只背模型清单

**【视频补充，约 [05:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=300s)】**老师把大量发布模型放在同一张时间线上，是为了看共识怎样形成。正确问题不是“某模型用了什么，所以我也用什么”，而是：

**FLOPs** 在这里指完成一次给定计算所需的浮点加法、乘法等操作次数，是工作量；它不是经过了多少秒，也不是表示每秒速度的 FLOP/s。

1. 它改变了哪条数据流？
2. 参数量和 tensor shape 变了吗？
3. 训练稳定性是否改善？
4. 是 FLOPs 变少，还是数据搬运/并行更方便？
5. 证据来自多规模重复实验，还是单次报告？

例如 RMSNorm 的 FLOPs 占比很小，但它要读写整个 activation，仍可能影响墙钟时间。“FLOPs 少”与“运行一定快”不是同一句话。

### 4.4 一张因果对比，而不是流行度榜单

| 改动 | 直接改变 | 通常不直接改变 |
|---|---|---|
| post-norm → pre-norm | residual/gradient 的路径 | 主 tensor `[B,T,d]` shape |
| LayerNorm → RMSNorm | 是否减均值、是否有 beta | 主 tensor shape |
| ReLU FFN → SwiGLU | 中间运算和矩阵数 | block 输入/输出 `[B,T,d]` |
| sinusoid → RoPE | Q/K 中的位置作用方式 | token 数和词表大小 |
| MHA → GQA | KV head 数、cache | query head 数可保持不变 |
| full → sliding window | 每个 query 可看的 key 集合 | 单个 hidden state 的宽度 |

这一表的用途是定位“旋钮改了哪里”。具体效果还取决于数据、规模、优化器、实现和硬件。

---

## 5. residual 与 norm 到底放在哪里

### 5.1 norm 先做什么

**normalization（归一化）**在这里指：把某个 token 的 $`d`$ 个特征按统计量重新缩放，使数值尺度更可控。它不是把概率归一到和为 1；那是 softmax 的工作。

令 $`F`$ 代表 attention 或 FFN。下面只画一个子层，实际 block 通常有 attention 子层和 FFN 子层各一次。

### 5.2 post-norm：先做子层与残差，再 norm

**【课程，PDF 第 10 页】**原始 Transformer 常写为：

```math
y=\mathrm{Norm}(x+F(x)).
```

逐步读：

1. 输入 $`x`$ 进入 $`F`$；
2. 得到 $`F(x)`$；
3. 做残差加法 $`x+F(x)`$；
4. 整个和再经过 norm 得到 $`y`$。

数据流：

```text
x ───────────────┐
│                + → Norm → y
└→ F(x) ─────────┘
```

虽然图中有跳接，但从 $`x`$ 走到 $`y`$ 的所有路线最后都必须经过 Norm。很多层叠起来时，不再有一条完全不经过变换的主路。

### 5.3 pre-norm：先 norm 再进入子层，主路直接通过

**【课程，PDF 第 10–12 页】**pre-norm 写为：

```math
y=x+F(\mathrm{Norm}(x)).
```

逐步读：

1. 复制一份 $`x`$；
2. 支路先做 $`\mathrm{Norm}(x)`$；
3. 支路进入 $`F`$；
4. 原始 $`x`$ 沿主路原样来到加号；
5. 两路相加。

```text
x ────────────────────┐
│                     + → y
└→ Norm → F ──────────┘
```

**【视频补充，约 [09:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=580s)】**早期解释常强调 pre-norm 可以减轻深层训练难题。现代大模型仍常使用 learning-rate warmup，所以不能把 pre-norm 简化成“用了它就不需要 warmup”。更稳妥的直觉是：它保留了更干净的 residual stream（残差流），信息和梯度有直接路径。

### 5.4 什么叫“干净的 residual stream”

**【补充解释】**先看连续两层 pre-norm：

```math
x_1=x_0+F_0(N(x_0)),
```

```math
x_2=x_1+F_1(N(x_1)).
```

把第一式代入第二式：

```math
x_2=x_0+F_0(N(x_0))+F_1(N(x_1)).
```

可以看到最初的 $`x_0`$ 以系数 1 直接出现在后面。模型每层是在主状态上“写入一个增量”。这不保证梯度永远完美，也不表示支路没有复杂变换；它只说明存在一条加法直通路。

### 5.5 用二维数字手算一次 residual 路径

**【补充例子】**令：

```math
x=[1,3].
```

暂时不关心 $`F`$ 内部怎样算，假设 pre-norm 支路输出：

```math
F(N(x))=[0.2,-0.1].
```

那么：

```math
y=[1,3]+[0.2,-0.1]=[1.2,2.9].
```

第一个旧特征 1 没被替换，而是在其上加 0.2；第二个旧特征 3 上加了 -0.1。

shape 也必须一致：`[2]+[2]→[2]`。对实际 tensor 则是 `[B,T,d]+[B,T,d]→[B,T,d]`。

### 5.6 non-residual post-norm：在支路出口再 norm

**【课程，PDF 第 13、19 页】**如果担心把 norm 放在整个残差和之后会破坏主路，可以只对支路输出再做一次 norm：

```math
y=x+\mathrm{Norm}_{out}\left(F(\mathrm{Norm}_{in}(x))\right).
```

```text
x ──────────────────────────────┐
│                               + → y
└→ Norm_in → F → Norm_out ──────┘
```

`Norm_out` 在子层之后，所以可称 post-norm；但它不包住 residual 加法，因此是 **non-residual post-norm（非残差 post-norm）**。主路中的 $`x`$ 仍不经过它。

课程提到 Grok、Gemma 2 等会使用额外 norm，OLMo 2 使用非残差 post-norm。不同报告所说的 “double norm” 可能在 attention、FFN 或 block 末端组合不同，阅读模型代码时应按箭头和公式确认，不要只看名字。

### 5.7 三种放置方式并排

| 名称 | 一个子层的简式 | $`x`$ 的直通路最后是否必须过 norm |
|---|---|---|
| post-norm | $`N(x+F(x))`$ | 是 |
| pre-norm | $`x+F(N(x))`$ | 否 |
| non-residual post-norm | $`x+N_{out}(F(N_{in}(x)))`$ | 否 |

**常见误读：**“non-residual post-norm”不是“完全没有 residual”，而是 **post-norm 不作用在 residual sum 上**。

### 5.8 attention 与 FFN 都要各自看一次

完整 serial pre-norm block 常写为：

```math
u=x+\mathrm{Attention}(N_1(x)),
```

```math
y=u+\mathrm{FFN}(N_2(u)).
```

每一步 shape 都是 `[B,T,d]`：

```text
x                         [B,T,d]
N1(x)                     [B,T,d]
Attention(N1(x))          [B,T,d]
u                         [B,T,d]
N2(u)                     [B,T,d]
FFN(N2(u))                [B,T,d]
y                         [B,T,d]
```

这里的 $`N_1,N_2`$ 通常是两个拥有各自可学习 scale 的 norm 模块，不要误以为同一个对象被调用两次。

---

## 6. LayerNorm、RMSNorm 与 bias：从四则运算手算

### 6.1 LayerNorm 逐符号解释

**【课程，PDF 第 14 页】**对一个 token 的特征向量 $`x=[x_1,\ldots,x_d]`$，LayerNorm（层归一化）先算均值和方差：

```math
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,
```

```math
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2.
```

再对每一维：

```math
\mathrm{LayerNorm}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_i.
```

符号逐个读：

- $`d`$：特征数，也就是最后一条 axis 的长度；
- $`\mu`$：这一个 token 的 $`d`$ 个特征的平均数；
- $`\sigma^2`$：方差，衡量这些数字离均值有多散；
- $`\varepsilon`$：很小的正数，防止分母为 0；
- $`\gamma_i`$：第 $`i`$ 维可学习的缩放；
- $`\beta_i`$：第 $`i`$ 维可学习的平移，也常叫 bias；
- 输出 shape 与输入完全相同。

LayerNorm 对 `[B,T,d]` 的最后一轴分别做统计。不同 batch、不同 token 位置不会共用同一个均值。

### 6.2 二维 LayerNorm 完整手算

**【补充例子】**令：

```math
x=[1,3],\quad d=2,\quad \gamma=[1,1],\quad\beta=[0,0],
```

为了手算先取 $`\varepsilon=0`$。

第 1 步，均值：

```math
\mu=\frac{1+3}{2}=2.
```

第 2 步，每个数字减均值：

```math
x-\mu=[1-2,3-2]=[-1,1].
```

第 3 步，方差：

```math
\sigma^2=\frac{(-1)^2+1^2}{2}=\frac{1+1}{2}=1.
```

第 4 步，标准差：

```math
\sqrt{\sigma^2}=\sqrt1=1.
```

第 5 步，除以标准差：

```math
\frac{x-\mu}{\sqrt{\sigma^2}}=[-1,1].
```

第 6 步，乘 $`\gamma`$、加 $`\beta`$：

```math
[-1,1]\odot[1,1]+[0,0]=[-1,1].
```

$`\odot`$ 表示逐元素乘。输入 `[2]`，输出仍 `[2]`。

### 6.3 四维 LayerNorm 再算一次，防止只会背二维答案

令：

```math
x=[0,2,4,6].
```

均值：

```math
\mu=\frac{0+2+4+6}{4}=3.
```

离均差：

```math
x-\mu=[-3,-1,1,3].
```

平方后是 `[9,1,1,9]`，所以方差：

```math
\sigma^2=\frac{9+1+1+9}{4}=5.
```

若 $`\gamma=1,\beta=0,\varepsilon=0`$，输出：

```math
\frac{[-3,-1,1,3]}{\sqrt5}
\approx[-1.342,-0.447,0.447,1.342].
```

输出的均值约为 0，均方约为 1。

### 6.4 RMSNorm 逐符号解释

RMS 是 **root mean square（均方根）**。先平方，再求平均，最后开平方：

```math
\mathrm{RMS}(x)=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\varepsilon}.
```

RMSNorm 通常写为：

```math
\mathrm{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.
```

与 LayerNorm 相比：

- 不计算、也不减去均值 $`\mu`$；
- 通常没有加法参数 $`\beta`$；
- 仍有逐维可学习缩放 $`\gamma`$；
- shape 仍不变。

### 6.5 同一个 `[1,3]` 的 RMSNorm 手算

仍取：

```math
x=[1,3],\quad d=2,\quad\gamma=[1,1],\quad\varepsilon=0.
```

第 1 步，平方：

```math
[1^2,3^2]=[1,9].
```

第 2 步，平方的平均：

```math
\frac{1+9}{2}=5.
```

第 3 步，开平方：

```math
\mathrm{RMS}(x)=\sqrt5\approx2.236.
```

第 4 步，每个数字除以 RMS：

```math
\mathrm{RMSNorm}(x)
=\left[\frac1{\sqrt5},\frac3{\sqrt5}\right]
\approx[0.447,1.342].
```

请对比同一输入的 LayerNorm 输出 `[-1,1]`。RMSNorm 没有减均值，所以第一个正数仍为正；两种 norm 不是完全相同的函数。

验证 RMSNorm 输出的均方为 1：

```math
\frac{0.447^2+1.342^2}{2}
\approx\frac{0.200+1.801}{2}
\approx1.0005.
```

误差来自小数截断。

### 6.6 为什么有 $`\varepsilon`$

**【补充解释】**若 $`x=[0,0]`$，RMS 是 0；直接除会变成 `0/0`，没有定义。加入例如 $`\varepsilon=10^{-6}`$：

```math
\sqrt{0+10^{-6}}=10^{-3},
```

于是输出仍可算为 `[0,0]`。$`\varepsilon`$ 是数值保险，不是可学习参数。

不同实现可能把 $`\varepsilon`$ 放在平方根内，或有细微约定差别。读取代码时要看公式，不只看名字。

### 6.7 参数量：LayerNorm 比 RMSNorm 多什么

假设隐藏宽度 $`d=4`$：

- LayerNorm 的 $`\gamma`$ 有 4 个参数，$`\beta`$ 有 4 个，共 8 个；
- RMSNorm 通常只有 $`\gamma`$，共 4 个。

大型模型里，这几个参数相对矩阵参数极少。课程重点却不只是参数容量，而是 norm 要读取、统计、再写回整条 activation。

### 6.8 FLOPs 少不等于墙钟时间一定少

**【补充解释：首次系统术语盒】**

- **FLOP** 是一次浮点加法、乘法等操作，是“工作量的一小格”；
- **FLOPs** 在本笔记中表示总浮点操作数，例如一个算子共做多少次；
- **FLOP/s** 是每秒可做多少浮点操作，是硬件或实测吞吐速度。

总 FLOPs 少不保证时间短，因为运行时间还受 FLOP/s、内存带宽、启动一次 GPU 底层计算程序（kernel）和数据移动影响。架构选择会同时改变工作量与硬件实际利用率。

**【课程，PDF 第 15–17 页】**课程引用的一个极端小模型工作负载中：

| 算子类 | FLOPs 占比 | runtime 占比 |
|---|---:|---:|
| tensor contraction（主要是矩阵乘） | 99.80% | 61.0% |
| statistical normalization（统计归一化） | 0.17% | 25.5% |
| elementwise（逐元素） | 0.03% | 13.5% |

三个 runtime 比例检查：

```math
61.0\%+25.5\%+13.5\%=100.0\%.
```

**【视频补充，约 [15:49](https://www.youtube.com/watch?v=lVynu4bo1rY&t=949s)】**老师明确提醒，这是为了展示“FLOPs 不是 runtime”的极端例子，不应把 `25.5%` 当成所有现代大模型的固定比例。

为什么可能这样？矩阵乘能让大量计算复用已加载的数据；norm 的算术很少，却仍需扫描 activation、求统计量、再写回。若受 memory bandwidth（内存带宽）限制，少几次统计和参数搬运仍可能带来实际速度收益。

### 6.9 bias 是什么，为什么很多模型删掉

线性层通常写成：

```math
y=xW+b.
```

- $`xW`$ 是矩阵乘；
- $`b`$ 是 bias（偏置），shape 为 `[d_out]`；
- 同一个 $`b`$ 会加到每个 batch、每个 token 位置。

**【补充例子】**若：

```math
xW=[2,-1],\qquad b=[0.5,0.25],
```

则：

```math
y=[2+0.5,-1+0.25]=[2.5,-0.75].
```

bias 给每个输出维度一个可学习的平移量。删除它不会改变输出 shape，但少了参数、一次读取和逐元素加法。

**【课程，PDF 第 18–19 页】**许多现代 Transformer 在线性层和 RMSNorm 中删除 bias。课程给出的动机包括内存/数据移动和优化稳定性；这里应理解为经验设计选择，不是“bias 永远没用”的数学定理。

### 6.10 一次把三组概念分开

| 概念 | 做什么 | 会不会混合 token 位置 | 输入输出 shape |
|---|---|---|---|
| LayerNorm | 减均值、按标准差缩放、再乘 $`\gamma`$ 加 $`\beta`$ | 不会 | 相同 |
| RMSNorm | 按均方根缩放、再乘 $`\gamma`$ | 不会 | 相同 |
| bias | 每个输出特征加一个学习量 | 不会 | 相同 |

### 6.11 本节检查点

读者现在应能不用术语背诵，回答：

1. 为什么 pre-norm 有一条不经过 norm 的 residual 主路？
2. 为什么 non-residual post-norm 仍然有 residual？
3. 为什么 `[1,3]` 经 LayerNorm 与 RMSNorm 得到不同符号的第一个分量？
4. 为什么 RMSNorm 减少的 FLOPs 很少，却仍可能节省时间？
5. 删除 bias 改变参数和数据移动，但为什么不改变 tensor shape？

若第 3 问不清楚，请回到 §6.2 和 §6.5，把减均值那一步重新手算一遍。

---

## 7. 普通 FFN 与激活函数：先理解“为什么需要弯一下”

### 7.1 只有线性层叠线性层，仍然只是一个线性层

**【补充解释】**假设没有激活函数：

```math
y=(xW_1)W_2.
```

矩阵乘满足结合律，所以：

```math
y=x(W_1W_2).
```

令 $`W_{new}=W_1W_2`$，两层就等价于一层 $`y=xW_{new}`$。堆再多这样的线性层，也不能表达真正弯曲、分段或门控的关系。

因此普通 FFN 在两个矩阵之间放一个 nonlinear activation（非线性激活）：

```math
\mathrm{FFN}(x)=\phi(xW_{up})W_{down}.
```

- $`W_{up}`$：把宽度从 $`d`$ 扩到 $`d_{ff}`$；
- $`\phi`$：逐元素非线性；
- $`W_{down}`$：把宽度从 $`d_{ff}`$ 降回 $`d`$；
- 输入、输出都是 `[B,T,d]`，所以能做 residual 加法。

“up/down”只说中间宽度先增后减；它们都是学习到的矩阵。

### 7.2 一个普通 FFN 的完整小数组

**【补充例子】**令单个 token 的隐藏向量：

```math
x=[1,-2],
```

所以 $`d=2`$。取 $`d_{ff}=3`$，令：

```math
W_{up}=
\begin{bmatrix}
1&0&-1\\
0&1&1
\end{bmatrix}.
```

shape 是 `[2,3]`。先算三个中间特征：

```math
h=xW_{up}.
```

逐列计算：

```math
h_1=1\times1+(-2)\times0=1,
```

```math
h_2=1\times0+(-2)\times1=-2,
```

```math
h_3=1\times(-1)+(-2)\times1=-3.
```

所以：

```math
h=[1,-2,-3].
```

若激活是 ReLU：

```math
\mathrm{ReLU}(h)=[1,0,0].
```

再令：

```math
W_{down}=
\begin{bmatrix}
2&1\\
1&0\\
-1&3
\end{bmatrix},
```

shape 是 `[3,2]`。于是：

```math
[1,0,0]W_{down}=[2,1].
```

shape 链完整闭合：

```text
x                 [2]
W_up              [2,3]
h                 [3]
ReLU(h)           [3]
W_down            [3,2]
FFN(x)            [2]
```

在真实模型里，前面的 `[B,T]` 两轴都保留：`[B,T,d]→[B,T,d_ff]→[B,T,d]`。

### 7.3 ReLU：负数归零

**【课程，PDF 第 20–21 页】**ReLU 是 Rectified Linear Unit（修正线性单元）：

```math
\mathrm{ReLU}(z)=\max(0,z).
```

对小向量逐项算：

```math
z=[-2,-1,0,1,2],
```

```math
\mathrm{ReLU}(z)=[0,0,0,1,2].
```

它的直觉很简单：正信号原样通过，负信号被关掉。缺点之一是负半轴的导数为 0；某个单元若长期落在负侧，梯度可能难以把它救回来。

### 7.4 GELU：不是硬切断，而是平滑缩放

GELU 是 Gaussian Error Linear Unit（高斯误差线性单元）：

```math
\mathrm{GELU}(z)=z\Phi(z).
```

$`\Phi(z)`$ 是标准正态分布的累积分布函数。人话解释：$`\Phi(z)`$ 是一个从接近 0 平滑升到接近 1 的门；输入 $`z`$ 再乘这扇门。

几个常用近似值：

| $`z`$ | $`\Phi(z)`$ 约为 | $`z\Phi(z)`$ 约为 |
|---:|---:|---:|
| -2 | 0.0228 | -0.0456 |
| -1 | 0.1587 | -0.1587 |
| 0 | 0.5 | 0 |
| 1 | 0.8413 | 0.8413 |
| 2 | 0.9772 | 1.9544 |

所以：

```math
\mathrm{GELU}([-1,0,1,2])
\approx[-0.159,0,0.841,1.954].
```

与 ReLU 不同，GELU 会让一部分小负数以较小负值通过，并在 0 附近平滑变化。

### 7.5 Swish/SiLU：输入乘 sigmoid 门

Swish 常写为：

```math
\mathrm{Swish}(z)=z\,\sigma(z),
```

其中 sigmoid：

```math
\sigma(z)=\frac{1}{1+e^{-z}}.
```

在许多代码库中，`SiLU`（Sigmoid Linear Unit）就是这个函数。

逐项手算的近似表：

| $`z`$ | $`\sigma(z)`$ 约为 | $`z\sigma(z)`$ 约为 |
|---:|---:|---:|
| -1 | 0.269 | -0.269 |
| 0 | 0.5 | 0 |
| 1 | 0.731 | 0.731 |
| 2 | 0.881 | 1.762 |

所以：

```math
\mathrm{Swish}([-1,0,1,2])
\approx[-0.269,0,0.731,1.762].
```

### 7.6 三个函数并排看

| 输入 $`z`$ | ReLU | GELU（约） | Swish（约） |
|---:|---:|---:|---:|
| -1 | 0 | -0.159 | -0.269 |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0.841 | 0.731 |
| 2 | 2 | 1.954 | 1.762 |

**【课程】**原始 Transformer 使用 ReLU；GPT 系模型推动了 GELU；许多 2023 年以后的模型使用 SwiGLU 等 gated 变体。

**【视频补充，约 [20:15](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1215s)】**老师展示“激活函数动物园”，重点不是记下所有缩写，而是先区分普通逐元素激活和下一节的 gated FFN。视频的判断是：门控本身带来的差异常比 GeGLU 与 SwiGLU 两种门之间的差异更重要。

### 7.7 普通 FFN 的主权重参数量

忽略 bias：

- $`W_{up}`$ shape `[d,d_ff]`，有 $`d\times d_{ff}`$ 个参数；
- $`W_{down}`$ shape `[d_ff,d]`，有 $`d_{ff}\times d`$ 个参数。

相加：

```math
N_{FFN}
=d d_{ff}+d_{ff}d
=2d d_{ff}.
```

**【补充例子】**若 $`d=6,d_{ff}=12`$：

```math
N_{FFN}=2\times6\times12=144.
```

也可逐矩阵复算：

```math
6\times12+12\times6=72+72=144.
```

若带 bias，还要加 $`d_{ff}+d`$ 个参数；课程讨论大模型比例时通常忽略这项，因为它相对矩阵项很小，而且许多现代模型本来就不用 bias。

---

## 8. GLU、GeGLU 与 SwiGLU：一条内容支路乘一扇门

### 8.1 gated 到底多了什么

**【课程，PDF 第 22–26 页】**普通 FFN 只有一份向上投影：

```math
h=\phi(xW_{up}).
```

gated FFN 生成两份同宽向量：

```math
u=xW_{up},
```

```math
g=xW_{gate}.
```

然后用门 $`\phi(g)`$ 逐元素控制内容 $`u`$：

```math
m=u\odot\phi(g).
```

最后降回模型宽度：

```math
\mathrm{GatedFFN}(x)
=\left[(xW_{up})\odot\phi(xW_{gate})\right]W_{down}.
```

$`\odot`$ 是逐元素乘。`up` 和 `gate` 必须 shape 相同，才能一格对一格相乘。

### 8.2 一个两维 gate 的完整手算

**【补充例子】**假设两条投影已经得到：

```math
u=[2,-1],\qquad g=[-1,2].
```

若用 ReLU 作门：

```math
\mathrm{ReLU}(g)=[0,2].
```

逐元素相乘：

```math
u\odot\mathrm{ReLU}(g)
=[2\times0,(-1)\times2]=[0,-2].
```

第一个内容值 2 被门完全关掉；第二个内容值 -1 被门系数 2 放大成 -2。

### 8.3 名字由“门用什么函数”决定

使用统一记号 $`u\odot\phi(g)`$：

| 名称 | 门函数 $`\phi(g)`$ | 本例 $`u=[2,-1],g=[-1,2]`$ 的门后结果 |
|---|---|---|
| GLU | sigmoid | $`[2\times0.269,-1\times0.881]\approx[0.538,-0.881]`$ |
| ReGLU | ReLU | $`[0,-2]`$ |
| GeGLU | GELU | $`[2\times(-0.159),-1\times1.954]\approx[-0.318,-1.954]`$ |
| SwiGLU | Swish/SiLU | $`[2\times(-0.269),-1\times1.762]\approx[-0.538,-1.762]`$ |
| LiGLU | identity，即 $`\phi(g)=g`$ | $`[2\times(-1),-1\times2]=[-2,-2]`$ |

不同论文可能把两条投影谁叫 `gate`、谁叫 `value/up` 调换；只要最终是两条同宽支路逐元素相乘，数学结构相同。读代码时看乘号两边，不要只看变量名。

### 8.4 gated FFN 的 shape 链

设 gated 中间宽度为 $`d_g`$：

```text
x                         [B,T,d]
x @ W_up                  [B,T,d_g]
x @ W_gate                [B,T,d_g]
activation(gate)          [B,T,d_g]
elementwise product       [B,T,d_g]
@ W_down                  [B,T,d]
```

权重 shape：

```text
W_up                      [d,d_g]
W_gate                    [d,d_g]
W_down                    [d_g,d]
```

如果 `W_up` 输出 `[B,T,8]`，`W_gate` 输出 `[B,T,7]`，逐元素乘不能成立。门控不是矩阵乘，而是一格对一格的乘法。

### 8.5 为什么是三个矩阵

忽略 bias：

- $`W_{up}`$：$`d d_g`$ 个参数；
- $`W_{gate}`$：$`d d_g`$ 个参数；
- $`W_{down}`$：$`d_g d`$ 个参数。

总数：

```math
N_{gated}
=d d_g+d d_g+d_g d
=3d d_g.
```

若错误地让 gated FFN 与普通 FFN 使用同一个中间宽度，它的主权重会从 $`2dd_{ff}`$ 增到 $`3dd_{ff}`$，也就是 1.5 倍。

### 8.6 等参数预算为什么要乘 $`2/3`$

**【课程，PDF 第 23–24、37–38 页】**现在从头推，不背结论。

普通 FFN 的矩阵参数：

```math
N_{plain}=2d d_{ff,plain}.
```

gated FFN 的矩阵参数：

```math
N_{gated}=3d d_{ff,gated}.
```

要让两者参数相等，先令：

```math
3d d_{ff,gated}=2d d_{ff,plain}.
```

两边都除以非零的 $`d`$：

```math
3d_{ff,gated}=2d_{ff,plain}.
```

再除以 3：

```math
d_{ff,gated}=\frac{2}{3}d_{ff,plain}.
```

所以 `2/3` 来自“三个矩阵对两个矩阵”的预算平衡，不是神秘经验常数。

### 8.7 用整数把等预算复算一遍

令 $`d=6`$，普通 FFN 取 $`d_{ff,plain}=12`$：

```math
N_{plain}=2\times6\times12=144.
```

gated 宽度取：

```math
d_{ff,gated}=\frac23\times12=8.
```

于是：

```math
N_{gated}=3\times6\times8=144.
```

逐矩阵核对：

```text
普通：W_up 6×12 = 72；W_down 12×6 = 72；总计 144
门控：W_up 6×8 = 48；W_gate 6×8 = 48；W_down 8×6 = 48；总计 144
```

两种 FFN 的中间 activation 元素数不同，但这里比较的是三块主权重的参数预算。

### 8.8 `8/3` 规则从哪里来

原始 Transformer 风格常取：

```math
d_{ff,plain}=4d.
```

代入等预算式：

```math
d_{ff,gated}
=\frac23d_{ff,plain}
=\frac23\times4d
=\frac83d.
```

因此 `8/3 ≈ 2.667` 不是“所有 SwiGLU 必须用的宽度”，而是 **从普通 FFN 的 `4d` 基准出发，再做等矩阵参数换算** 的结果。

**【补充例子】**若 $`d=3072`$：

```math
d_{ff,gated}=\frac83\times3072.
```

先算 $`3072/3=1024`$，再乘 8：

```math
1024\times8=8192.
```

若 $`d`$ 不能整除 3，真实模型通常会取对硬件更友好的整数倍，例如 128 或 256 的倍数；此时只是接近 $`8d/3`$，不必严格相等。

### 8.9 门控的效果证据怎样读

**【视频补充，约 [21:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1292s)–[25:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1501s)】**课程的总结是：gating 在多项比较中带来小但较一致的收益，而 GeGLU 与 SwiGLU 之间的差通常更小。老师也提醒，论文里的小差异要看重复实验和误差条；一次跑分领先不等于可靠定律。

首次阅读应记住的不是模型榜单，而是：

```text
普通 FFN：一条中间支路，两个矩阵
gated FFN：内容支路 × 门支路，再 down，三个矩阵
等参数预算：gated 中间宽度 = 普通宽度 × 2/3
```

---

## 9. serial 与 parallel block：两家工厂串联还是并联

### 9.1 serial：FFN 能看到 attention 刚写入的内容

**【课程，PDF 第 27–29 页】**常见 serial（串联）pre-norm block：

```math
u=x+\mathrm{Attn}(N_1(x)),
```

```math
y=u+\mathrm{FFN}(N_2(u)).
```

因果顺序：

```text
x
├─ residual ───────────────────────────────┐
└→ Norm1 → Attention → + → u              │
                          └→ Norm2 → FFN → + → y
```

关键不是图画成一条线，而是 FFN 的输入是 $`u`$。因为 $`u`$ 已包含当前层 attention 的输出，FFN 可以立刻加工刚从其他 token 收到的信息。

### 9.2 parallel：attention 与 FFN 看同一个旧状态

parallel（并联）block 可写为：

```math
y=x+\mathrm{Attn}(N(x))+\mathrm{FFN}(N(x)).
```

```text
                     ┌→ Attention ─┐
x → Norm（可共享） ──┤             + → 加回 x → y
                     └→ FFN ───────┘
```

两条支路都看 $`N(x)`$；本层 FFN 看不到本层 attention 刚产生的结果，只能等下一层再处理它。

### 9.3 shape 为什么都能相加

无论 serial 还是 parallel：

```text
x                  [B,T,d]
Attention output   [B,T,d]
FFN output         [B,T,d]
y                  [B,T,d]
```

parallel 的加法是三个同 shape tensor 逐元素相加。它不是在特征轴拼接；若拼接会得到 `[B,T,3d]`，那是另一种操作。

### 9.4 二维数值例子看数据依赖

**【补充例子】**令：

```math
x=[1,2].
```

假设 attention 支路输出：

```math
A(N(x))=[0.5,-0.5].
```

serial 先得到：

```math
u=[1,2]+[0.5,-0.5]=[1.5,1.5].
```

FFN 读取这个新 $`u`$。假设：

```math
M(N(u))=[1,0.2].
```

则：

```math
y_{serial}=[1.5,1.5]+[1,0.2]=[2.5,1.7].
```

parallel 中 FFN 读取旧的 $`x`$。假设同一 FFN 对旧状态给出：

```math
M(N(x))=[-0.2,0.4].
```

则：

```math
y_{parallel}
=[1,2]+[0.5,-0.5]+[-0.2,0.4]
=[1.3,1.9].
```

数字不同不是为了证明谁更好，而是说明两种图有不同依赖：serial 的 FFN 输入是 attention 后状态，parallel 的不是。

### 9.5 parallel 为什么有机会更快

**【视频补充，约 [27:19](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1639s)–[28:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1712s)】**可能的系统收益：

1. attention 与 FFN 没有前后依赖，可同时调度一部分工作；
2. 两条支路可共享一次输入 norm；
3. Q/K/V 投影和 FFN 的 up/gate 投影都读取同一个输入，实现可能把若干投影融合成较大的矩阵乘，减少 kernel launch 与数据读取。

这些是实现机会，不表示任意框架写成 parallel 就自动快。

### 9.6 速度与表达深度的交换

**【课程/视频补充】**PaLM 报告过 parallel block 在其规模和实现下约 15% 的训练速度收益，且报告中没有明显质量下降。但老师同时指出：后来的许多模型没有继续采用，可能因为串联结构让每层拥有两步依次加工的语义深度；公开材料缺少足够干净、跨规模的受控消融。

因此应写成：

> PaLM 的特定实验报告约 15% 系统收益；parallel 提供共享 norm 和融合机会，但其质量/表达权衡不是已证明的普适结论。

不能写成“parallel 一定快 15%”或“parallel 一定损失一半能力”。

---

## 10. 四类位置编码：如果不给位置，模型不知道谁先谁后

### 10.1 为什么 token 内容本身不够

**【补充解释】**假设只有两个 token 向量 `A` 和 `B`。不加任何位置信号时，self-attention 对集合 `{A,B}` 做同样的点积规则。把输入从 `[A,B]` 换成 `[B,A]`，输出也只是对应地交换；模型没有一个固定信号说明“这是第 0 位”或“二者相距 1”。

语言顺序却会改变意思：

```text
狗 追 猫
猫 追 狗
```

token 种类相同，角色不同。因此要把位置或相对距离送进 attention。

### 10.2 sinusoidal absolute position：固定正弦表

**【课程，PDF 第 30 页】**原始 Transformer 使用固定 sinusoidal（正弦）编码。常见公式为：

```math
PE(pos,2r)=\sin\left(\frac{pos}{10000^{2r/d}}\right),
```

```math
PE(pos,2r+1)=\cos\left(\frac{pos}{10000^{2r/d}}\right).
```

- $`pos`$：整数位置，例如 0、1、2；
- $`r`$：第几对特征；
- $`2r,2r+1`$：一对 sin/cos 坐标；
- $`d`$：模型宽度；
- 不同 $`r`$ 对应不同变化速度。

位置向量通常在模型输入处与 token embedding 相加：

```math
X_0=E[token\_id]+PE[pos].
```

两者 shape 都是 `[B,T,d]`（位置表可通过广播扩到 batch）。固定公式不增加可学习参数。

### 10.3 learned absolute position：位置也查表

learned absolute embedding（可学习绝对位置嵌入）建立表：

```text
P shape = [T_max,d]
```

位置 0 查第 0 行，位置 17 查第 17 行，再与 token embedding 相加。

优点：模型能为每个训练位置学习任意向量。限制：

- 新增 $`T_{max}d`$ 个参数；
- 超出训练表长度的位置没有天然条目；
- “位置 100 与 101 相差 1”没有被公式直接编码，需从数据学。

### 10.4 relative position：直接给距离一个分数修正

一种相对位置方法是在 attention logit 上加偏置：

```math
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_h}}+b_{i-j}.
```

- $`i`$：query 位置；
- $`j`$：key 位置；
- $`i-j`$：相对距离；
- $`b_{i-j}`$：该距离对应的可学习偏置，实际模型可能把远距离分桶（bucket）。

这里是 relative-bias 方法自行选择的索引约定。后面的 RoPE 仍按课件命名 $`\delta=i-j`$，但在本文 query-left/key-right 的点积矩阵中出现 $`R(j-i)=R(-\delta)`$；不要因为都叫“相对位置”就忽略符号方向。

**【补充例子】**若内容点积分数是 1.2，位置 $`i=5`$ 看 $`j=3`$，相对距离为 $`5-3=2`$，而 $`b_2=-0.1`$：

```math
s_{5,3}=1.2+(-0.1)=1.1.
```

它直接修改 attention 的“愿意看多远”，而不是先给输入 token 加一个绝对坐标。

### 10.5 RoPE：把位置变成 Q/K 的旋转角

RoPE 是 Rotary Position Embedding（旋转位置编码）。它不是把一条位置向量简单加到输入，而是在每层 attention 中：

1. 从 hidden state 投影出 Q 和 K；
2. 按 token 位置旋转 Q、K 的每对坐标；
3. 用旋转后的 Q、K 做点积。

这样点积中的位置部分能化成相对位移。完整证明在第 11 节。

### 10.6 四类方法放在同一张表

| 方法 | 位置信号放在哪里 | 可学习参数 | 相对距离是否直接进入公式 |
|---|---|---:|---|
| sinusoidal absolute | 输入 embedding 相加 | 0 | 间接，可由 sin/cos 关系推得 |
| learned absolute | 输入 embedding 相加 | $`T_{max}d`$ | 否，需学习 |
| relative bias/embedding | attention logit 或相对项 | 取决于实现 | 是 |
| RoPE | 每层 attention 的 Q、K | 常见基础版 0 | 是，出现在旋转后的点积 |

**【课程/视频补充，约 [31:04](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1864s)–[32:34](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1954s)】**课程把 RoPE 视为 2024 年后模型中的强势默认趋势。这是课程对其模型调查时间点的经验总结，不是说其他位置方案已经数学失效。

---

## 11. RoPE 从二维旋转一步步推到相对位置

### 11.1 二维旋转矩阵先读懂

角度为 $`\theta`$ 的二维旋转矩阵：

```math
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
```

它乘二维列向量：

```math
\begin{bmatrix}x'\\y'\end{bmatrix}
=R(\theta)
\begin{bmatrix}x\\y\end{bmatrix}.
```

展开就是：

```math
x'=x\cos\theta-y\sin\theta,
```

```math
y'=x\sin\theta+y\cos\theta.
```

它改变方向，不改变长度。

### 11.2 旋转 90° 的最小手算

令向量：

```math
v=\begin{bmatrix}1\\0\end{bmatrix},
```

取 $`\theta=90^\circ`$。因为：

```math
\cos90^\circ=0,\qquad\sin90^\circ=1,
```

所以：

```math
R(90^\circ)=
\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
```

矩阵乘：

```math
R(90^\circ)v
=\begin{bmatrix}0\times1+(-1)\times0\\1\times1+0\times0\end{bmatrix}
=\begin{bmatrix}0\\1\end{bmatrix}.
```

即 `[1,0]` 逆时针转 90° 变成 `[0,1]`。

### 11.3 RoPE 怎样给位置分配角度

先只看一对坐标。选择一个频率 $`\omega`$。位于位置 $`i`$ 的向量旋转角为：

```math
\theta_i=i\omega.
```

于是原始 query $`q_i`$ 和 key $`k_j`$ 变成：

```math
q_i'=R(i\omega)q_i,
```

```math
k_j'=R(j\omega)k_j.
```

注意：下标 $`i,j`$ 表示 token 位置；向量内容 $`q_i,k_j`$ 仍来自模型投影。RoPE 没有把所有位置变成相同向量。

**相对位移的符号约定：**课件把 query 位置减 key 位置记作：

```math
\delta=i-j.
```

本文始终把 query 放在点积左边、key 放在右边，并用列向量。因此稍后推出的旋转矩阵是：

```math
R(j-i)=R(-\delta).
```

$`\delta`$ 与 $`j-i`$ 只差负号，二者都只依赖相对位置；但一旦固定了点积方向，旋转方向的正负不能混写。

### 11.4 为什么点积里出现转置

两个列向量的点积写成：

```math
(q_i')^{\mathsf T}k_j'.
```

代入旋转定义：

```math
(R(i\omega)q_i)^{\mathsf T}(R(j\omega)k_j).
```

使用 $`(AB)^{\mathsf T}=B^{\mathsf T}A^{\mathsf T}`$：

```math
(R(i\omega)q_i)^{\mathsf T}
=q_i^{\mathsf T}R(i\omega)^{\mathsf T}.
```

因此：

```math
(q_i')^{\mathsf T}k_j'
=q_i^{\mathsf T}R(i\omega)^{\mathsf T}R(j\omega)k_j.
```

接下来只需证明中间两个矩阵只依赖 $`j-i`$。

### 11.5 逐项乘出 $`R(i)^\mathsf T R(j)=R(j-i)=R(-\delta)`$

为减少符号，先令：

```math
a=i\omega,\qquad b=j\omega.
```

写出两个矩阵：

```math
R(a)^{\mathsf T}
=\begin{bmatrix}
\cos a&\sin a\\
-\sin a&\cos a
\end{bmatrix},
```

```math
R(b)
=\begin{bmatrix}
\cos b&-\sin b\\
\sin b&\cos b
\end{bmatrix}.
```

在四格矩阵乘之前，先学两条差角公式：

```math
\cos(b-a)=\cos b\cos a+\sin b\sin a,
```

```math
\sin(b-a)=\sin b\cos a-\cos b\sin a.
```

因为普通数字乘法可交换，例如 $`\cos a\cos b=\cos b\cos a`$，所以矩阵格子可以逐项与这两条公式对照。

四个格子逐个相乘。

左上：

```math
\cos a\cos b+\sin a\sin b
=\cos b\cos a+\sin b\sin a
=\cos(b-a).
```

右上：

```math
-\cos a\sin b+\sin a\cos b
=-\left(\sin b\cos a-\cos b\sin a\right)
=-\sin(b-a).
```

右上角最后必须是 $`-\sin(b-a)`$；负号来自旋转矩阵右上角的定义。

左下：

```math
-\sin a\cos b+\cos a\sin b
=\sin b\cos a-\cos b\sin a
=\sin(b-a).
```

右下：

```math
\sin a\sin b+\cos a\cos b
=\cos b\cos a+\sin b\sin a
=\cos(b-a).
```

所以：

```math
R(a)^{\mathsf T}R(b)
=\begin{bmatrix}
\cos(b-a)&-\sin(b-a)\\
\sin(b-a)&\cos(b-a)
\end{bmatrix}
=R(b-a).
```

把 $`a=i\omega,b=j\omega`$ 放回去：

```math
R(i\omega)^{\mathsf T}R(j\omega)
=R((j-i)\omega).
```

因此旋转后点积为：

```math
(q_i')^{\mathsf T}k_j'
=q_i^{\mathsf T}R((j-i)\omega)k_j.
```

位置角度只通过 $`j-i=-\delta`$ 出现。这就是“RoPE 让点积依赖相对位置”的核心。课件若用 $`\delta=i-j`$ 命名距离，本式就是 $`R(-\delta)`$，不能把旋转方向的负号省掉。

### 11.6 为什么相对位置不等于“内容消失”

最后的式子仍有 $`q_i`$ 和 $`k_j`$：

```math
q_i^{\mathsf T}R((j-i)\omega)k_j.
```

所以 attention 分数同时取决于：

1. query 的内容 $`q_i`$；
2. key 的内容 $`k_j`$；
3. 相对位移 $`j-i=-\delta`$。

RoPE 不是只按距离决定注意力，也不是把绝对位置完全从整个模型中“删除”；它保证这个二维点积的旋转关系以相对差出现。

### 11.7 一个带具体角度的完整验证

**【补充例子】**取频率 $`\omega=90^\circ`$ 每位置，原始向量：

```math
q=\begin{bmatrix}1\\0\end{bmatrix},\qquad
k=\begin{bmatrix}0\\1\end{bmatrix}.
```

令 query 在 $`i=1`$，key 在 $`j=2`$。

query 旋转角：

```math
i\omega=1\times90^\circ=90^\circ.
```

所以：

```math
q'=R(90^\circ)q
=\begin{bmatrix}0\\1\end{bmatrix}.
```

key 旋转角：

```math
j\omega=2\times90^\circ=180^\circ.
```

因为 $`R(180^\circ)=\begin{bmatrix}-1&0\\0&-1\end{bmatrix}`$：

```math
k'=R(180^\circ)k
=\begin{bmatrix}0\\-1\end{bmatrix}.
```

直接算旋转后点积：

```math
(q')^{\mathsf T}k'
=0\times0+1\times(-1)=-1.
```

再走相对位置公式。相对位移：

```math
j-i=2-1=1,
```

所以只旋转 key 一个 $`90^\circ`$：

```math
R(90^\circ)k
=\begin{bmatrix}-1\\0\end{bmatrix}.
```

与原始 query 点积：

```math
q^{\mathsf T}R(90^\circ)k
=1\times(-1)+0\times0=-1.
```

两条路线都得到 -1，数值验证了推导。

### 11.8 高维向量怎样变成很多二维旋转

真实 head 不是二维。若 $`d_h=8`$，RoPE 把坐标按实现约定组成 4 对：

```text
(feature 0, feature 1)
(feature 2, feature 3)
(feature 4, feature 5)
(feature 6, feature 7)
```

每一对独立用二维旋转，但频率不同：

```math
\theta_{i,r}=i\omega_r,
```

其中 $`r=0,1,2,3`$。通常有的坐标对转得快，表达短距离变化；有的转得慢，表达较长尺度。

**【视频补充，约 [36:20](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2180s)–[36:38](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2198s)】**视频字幕一处转写成 “3D”，但幻灯片矩阵、前后解释和实现都明确是把坐标按 **二维对** 分组。这里以官方图和数学式为准。

### 11.9 Q、K、V 的边界：通常只旋转 Q 和 K

attention 权重来自：

```math
\mathrm{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}\right).
```

为了让位置关系进入这组点积，标准 RoPE 对 Q、K 旋转：

```math
Q'=\mathrm{RoPE}(Q,pos),\qquad
K'=\mathrm{RoPE}(K,pos).
```

再算：

```math
\mathrm{softmax}\left(\frac{Q'(K')^{\mathsf T}}{\sqrt{d_h}}\right)V.
```

V 通常不旋转，因为 V 是被权重汇总的内容，不参与决定权重的 Q·K 点积。某些研究变体可能另作处理；“标准 RoPE”应明确写 Q/K，而不是含糊说“给 attention 加位置”。

### 11.10 为什么每层 attention 都应用

每层都会从当前 hidden state 重新投影出该层自己的 Q、K：

```text
layer 0 hidden → W_Q0/W_K0 → Q0/K0 → RoPE → attention 0
layer 1 hidden → W_Q1/W_K1 → Q1/K1 → RoPE → attention 1
...
```

若只在输入 embedding 做一次旋转，后续层新生成的 Q/K 并没有按该层的点积方式注入同样结构。标准实现是在每个使用 RoPE 的 attention 层、Q/K 投影之后应用。

### 11.11 安装匹配 PyTorch 后可运行的最小实现

**【补充解释】**下面代码只演示 RoPE 的 shape 和二维配对，不包含完整 attention。安装与本机环境匹配的 PyTorch 后，它可以独立运行：

```python
import torch


def apply_rope_pairs(x: torch.Tensor,
                     cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """x: [B,H,T,d_h]；cos/sin: [1,1,T,d_h/2]。"""
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE head dimension must be even")

    x_even = x[..., 0::2]  # [B,H,T,d_h/2]
    x_odd = x[..., 1::2]   # [B,H,T,d_h/2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # stack 后 [B,H,T,d_h/2,2]；flatten 后恢复 [B,H,T,d_h]
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


B, H, T, d_h = 1, 2, 3, 4
q = torch.arange(B * H * T * d_h, dtype=torch.float32).reshape(B, H, T, d_h)
k = q + 1.0
v = q + 2.0

positions = torch.arange(T, dtype=torch.float32)       # [T]
frequencies = torch.tensor([1.0, 0.1])                 # [d_h/2]
angles = positions[:, None] * frequencies[None, :]     # [T,d_h/2]
cos = angles.cos()[None, None, :, :]                   # [1,1,T,d_h/2]
sin = angles.sin()[None, None, :, :]                   # [1,1,T,d_h/2]

q_rot = apply_rope_pairs(q, cos, sin)
k_rot = apply_rope_pairs(k, cos, sin)

assert q_rot.shape == (1, 2, 3, 4)
assert k_rot.shape == (1, 2, 3, 4)
assert v.shape == (1, 2, 3, 4)  # 标准 RoPE 中 V 原样保留
```

> **代码验证状态：**该代码块已通过 Python AST 语法解析；本次编写环境没有安装 `torch`，因此没有做 PyTorch 运行时实跑。shape、broadcast 和旋转公式已分别用独立算术检查，但读者首次执行仍需先安装与环境匹配的 PyTorch。

逐行翻成人话：

- `x[..., 0::2]`：保留前面所有轴，在最后一轴取编号 0、2、4…；
- `x[..., 1::2]`：取编号 1、3、5…；
- `out_even/out_odd`：正是二维旋转的两条展开式；
- `torch.stack(..., dim=-1)`：把每对新坐标放回相邻的最后一轴；
- `.flatten(-2)`：把最后两轴 `[d_h/2,2]` 合回 `[d_h]`；
- `torch.arange(B * H * T * d_h, ...)`：依次生成从 0 开始的测试数字，总数正好等于目标 tensor 的元素数；
- `.reshape(B, H, T, d_h)`：只把这些数字重新解释为 `[B,H,T,d_h]`，不增删元素；
- `positions[:, None]` 把 `[T]` 变成 `[T,1]`；
- `frequencies[None, :]` 把 `[d_h/2]` 变成 `[1,d_h/2]`；
- 两者广播相乘得到每个位置、每个坐标对的角度；
- `cos/sin` 前面补两个长度为 1 的轴，使它们能广播到 batch 和 heads；
- `q_rot`、`k_rot` 的 shape 不变，只有数值按位置旋转；
- `v` 没有传进函数，明确表示标准实现不旋转 V。
- `assert condition` 会在条件为假时立刻报错；三行 `assert` 用来验证输出 shape 与预期一致。

### 11.12 RoPE 最容易犯的五个错

1. **错：**RoPE 给 token embedding 加一个向量。  
   **正：**标准 RoPE 在每层 attention 内旋转投影后的 Q、K。

2. **错：**`R(i)^T R(j)` 还依赖两个绝对位置，或可随意写成 $`R(i-j)`$。  
   **正：**按本文 query-left/key-right 的列向量约定，它等于 $`R(j-i)=R(-\delta)`$，其中课件 $`\delta=i-j`$；只剩相对差，但符号方向不可互换。

3. **错：**旋转后 attention 只看距离，不看内容。  
   **正：**最终仍有 $`q_i^{\mathsf T}`$ 和 $`k_j`$。

4. **错：**“二维配对”表示 head dimension 必须是 2。  
   **正：**$`d_h`$ 可很大，只需拆成很多二维对；常见实现要求 $`d_h`$ 为偶数。

5. **错：**Q/K/V 都必须旋转。  
   **正：**标准 RoPE 旋转 Q、K；V 通常不旋转。

**【课程/视频补充，约 [33:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1981s)–[38:24](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2304s)】**PDF 第 31–35 页和视频按“把相对距离命名为 $`\delta=i-j`$ → 二维旋转 → 多频率配对 → 每层 Q/K 应用”的顺序展开。按本文点积约定矩阵中出现的是 $`R(-\delta)=R(j-i)`$。本节保留课程因果链，并补上了差角公式、四格矩阵乘法和数值验证。

---

## 12. $`d_{ff}/d_{model}`$：常见范围不是自然常数

### 12.1 两个宽度分别控制什么

**【课程，PDF 第 36–41 页】**先重新定义：

- $`d_{model}`$，本笔记也简写为 $`d`$：residual stream 中每个 token 的宽度；
- $`d_{ff}`$：FFN 中间层的宽度。

普通 FFN：

```text
[B,T,d_model]
 → W_up [d_model,d_ff]
 → [B,T,d_ff]
 → activation
 → W_down [d_ff,d_model]
 → [B,T,d_model]
```

$`d_{ff}`$ 越大，每个 token 在 FFN 里可使用的中间特征越多，但权重、FLOPs 和中间 activation 也随之增加。

### 12.2 普通 FFN 的 `4×` 从哪里来

原始 Transformer 的常见配置是：

```math
d_{ff}=4d_{model}.
```

因此普通 FFN 主权重数为：

```math
2d_{model}d_{ff}
=2d_{model}(4d_{model})
=8d_{model}^2.
```

这不是从数学公理推出的最优值，而是早期配置被大量沿用后形成的强经验基线。

### 12.3 gated FFN 为什么常在 `2.5×–3.5×` 一带

上一节已经推导：若普通 FFN 的宽度是 $`4d`$，三个矩阵的 gated FFN 在等主权重预算下取：

```math
d_{ff,gated}=\frac23\times4d=\frac83d\approx2.667d.
```

**【课程，PDF 第 38 页；课程 2026 模型调查快照】**讲义列出的部分公开模型报告值包括：

| 模型 | 报告的 $`d_{ff}/d_{model}`$ | 如何理解 |
|---|---:|---|
| PaLM | 4.0 | 高于等预算 `8/3` |
| Mistral 7B | 3.5 | 比 `8/3` 更宽 |
| LLaMA 2 70B | 3.5 | 比 `8/3` 更宽 |
| LLaMA 70B | 2.68 | 接近 `8/3≈2.667` |
| Qwen 14B | 2.67 | 接近 `8/3` |
| DeepSeek 67B | 2.68 | 接近 `8/3` |
| Yi 34B | 2.85 | 稍高于 `8/3` |
| T5 v1.1 | 2.5 | 稍低于 `8/3` |

这些行来自不同年份、规模、数据和硬件，不能当成一次受控实验。它们只说明课程调查到的现代 gated 模型常落在相近数量级。

### 12.4 T5 11B 的 64 倍例外逐位复算

**【课程，PDF 第 39 页】**原始 T5 11B 配置：

```math
d_{ff}=65{,}536,\qquad d_{model}=1{,}024.
```

比例：

```math
\frac{d_{ff}}{d_{model}}
=\frac{65{,}536}{1{,}024}.
```

因为：

```math
1{,}024\times64=65{,}536,
```

所以：

```math
\frac{d_{ff}}{d_{model}}=64.
```

这说明极端宽 FFN 也可以被成功训练。它不说明 64 倍更优：课程指出后续 T5 v1.1 转为 GeGLU，比例约 2.5，更接近主流范围。

**【视频补充，约 [47:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2823s)】**老师提出一种系统直觉：极大的 FFN 会形成很大的矩阵乘，在某些硬件上可能有较好利用率；但这不是对 T5 选择的唯一因果证明。大矩阵也会增加参数、计算、模型切分和通信负担。

### 12.5 更新近年的其他例外，必须带时间边界

**【课程，PDF 第 39 页；2026 讲义快照】**课程还列出 Gemma 2 约 `8×`，以及 SmolLM、Gemma 3、Gemma 4 等 gated 配置约 `4×`。这些名字是 2026 课程对当时公开模型的观察，不能向未来外推为“新模型都在变宽”。

对于 gated FFN，`4×` 还意味着三个矩阵预算约：

```math
3d(4d)=12d^2,
```

而等预算 `8/3×` 是：

```math
3d\left(\frac83d\right)=8d^2.
```

所以 gated `4×` 比 gated `8/3×` 的 FFN 主权重多：

```math
\frac{12d^2}{8d^2}=1.5
```

倍。比较模型时必须同时看“是不是 gated”，不能只比较比例数字。

### 12.6 课程图表的轴到底说了什么

**【课程，PDF 第 40 页；Kaplan et al., 2020 的 50M 参数实验】**图中：

- 横轴是 Feed-Forward Ratio，即 $`d_{ff}/d_{model}`$；使用对数刻度；
- 纵轴是 Loss Increase（损失相对最佳点增加多少百分比）；越低越好；
- 蓝线固定 head 数 $`n_{head}=8`$；
- 橙线固定每头相关比例 $`d_{model}/n_{head}=64`$；
- 图下标明实验模型约 50M parameters。

两条曲线在 ratio 大约 1 到 10 之间形成较宽的低谷；继续增到很大比例时，损失明显上升。能支持的结论是：

> 在 Kaplan 2020 的约 5000 万参数实验设置里，`1–10` 存在宽容区，`4` 并不是一根针尖般唯一的最优点。

不能从图中推出：

- 所有万亿参数模型的最优范围永远是 1–10；
- gated 与普通 FFN 可不做参数匹配直接比较；
- 训练速度、显存和通信也在纵轴中被同时优化。

**【视频补充，约 [48:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2883s)】**老师用这张图强调“经验 basin（低谷/宽容区）”，而不是宣称精确常数。

### 12.7 初学者怎样选一个起点

**【补充解释】**如果只是实现课程作业、没有预算做大规模搜索：

1. 普通 ReLU/GELU FFN 可从 $`d_{ff}=4d`$ 起；
2. SwiGLU/GeGLU 若想匹配普通 `4d` FFN 的矩阵参数，可从 $`d_g\approx8d/3`$ 起；
3. 把结果圆整到硬件友好的整数；
4. 若改宽度，要同时重新算参数、FLOPs 和 activation；
5. 把 `2.5–4×` 当起点区间，不写成模型定律。

---

## 13. attention 内宽、head 数与深宽 aspect ratio

### 13.1 `num_heads × head_dim = model_dim` 是常见选择，不是 shape 定律

**【课程，PDF 第 42–43 页】**定义：

- $`H`$：attention query head 数；
- $`d_h`$：每个 head 的宽度；
- $`d`$：residual stream 的模型宽度；
- $`D_{attn}=H d_h`$：所有 heads 合并后的 attention 内宽。

常见配置令：

```math
Hd_h=d.
```

但只要投影矩阵 shape 合法，$`Hd_h`$ 可以大于或小于 $`d`$。用比例表示：

```math
r_{attn}=\frac{Hd_h}{d}.
```

- $`r_{attn}=1`$：attention 内宽等于 residual 宽；
- $`r_{attn}=2`$：attention 内宽是 residual 宽两倍；
- 这不表示 token 数或层数翻倍。

### 13.2 当 attention 内宽不等于模型宽，shape 怎样闭合

以标准 MHA、Q/K/V 同 head 数为例：

```text
X                          [B,T,d]
W_Q/W_K/W_V                [d,H*d_h]
Q/K/V before split         [B,T,H*d_h]
Q/K/V after split          [B,H,T,d_h]
heads merged after attn    [B,T,H*d_h]
W_O                        [H*d_h,d]
output                     [B,T,d]
```

即使 $`Hd_h\ne d`$，最后的 $`W_O`$ 仍把内宽投影回 $`d`$，所以 residual 加法合法。

### 13.3 attention 权重参数量怎样随比例变

三个输入投影各有：

```math
d(Hd_h)
```

个参数；输出投影有：

```math
(Hd_h)d
```

个参数。相加：

```math
N_{attn}
=3d(Hd_h)+(Hd_h)d
=4d(Hd_h).
```

因为 $`Hd_h=r_{attn}d`$：

```math
N_{attn}=4r_{attn}d^2.
```

所以 ratio 从 1 增到 2，在这些假设下 attention 四块主矩阵参数也翻倍。

### 13.4 课程表中可独立复算的几行

**【课程，PDF 第 43 页；模型报告跨越不同年份】**下面只列能从幻灯片打印数字直接复算的行：

| 模型 | $`H`$ | $`d_h`$ | $`d`$ | $`Hd_h/d`$ 复算 |
|---|---:|---:|---:|---:|
| GPT-3 | 96 | 128 | 12,288 | $`96\times128/12{,}288=1`$ |
| T5 | 128 | 128 | 1,024 | $`128\times128/1{,}024=16`$ |
| T5 v1.1 | 64 | 64 | 4,096 | $`64\times64/4{,}096=1`$ |
| LaMDA | 128 | 128 | 8,192 | $`128\times128/8{,}192=2`$ |
| LLaMA 2 | 64 | 128 | 8,192 | $`64\times128/8{,}192=1`$ |
| Qwen 3.5 27B | 24 | 256 | 5,120 | $`24\times256/5{,}120=1.2`$ |

以 T5 行为例：

```math
128\times128=16{,}384,
```

```math
16{,}384/1{,}024=16.
```

这是显著例外。课程的样本总结是“多数接近 1，一些 Google 模型例外”。这张表不是控制其他配置不变的质量实验，不能证明 ratio 1 总是最优。

### 13.5 aspect ratio 在本讲里是什么意思

**【课程，PDF 第 44 页】**这里的 architecture aspect ratio（架构深宽比）定义为：

```math
A=\frac{d_{model}}{L},
```

$`L`$ 是 block 层数。

- $`A`$ 大：相对更宽、更浅；
- $`A`$ 小：相对更窄、更深。

它不是矩阵长宽比，也不是 $`d_{ff}/d`$。本讲有两个“ratio”，必须看分子分母。

### 13.6 课程模型表的经验范围与边界

**【课程，PDF 第 44、51 页；2026 讲义对跨年代公开配置的汇总】**表中许多模型落在约 `100–200`，但也列有更低或更高的例子：BLOOM 约 205、PaLM 540B 约 156、一组 GPT-3/OPT/Mistral/Qwen/OLMo 3 约 128、LLaMA/LLaMA 2 约 102、Gemma 3 约 87、Gemma 4 约 61、T5 11B 约 33。

正确说法是“该课程调查样本中有一个宽的常见带”，而不是“aspect ratio 必须等于 128”。

### 13.7 课程图表与系统因果

**【补充解释：首次系统术语盒】**

- **latency（延迟）**：一个请求或一个 token 从开始到得到结果所经历的时间；层越深，必须顺序经过的阶段通常越多；
- **tensor parallel（张量并行）**：把同一层的大矩阵沿某条轴切给多个设备一起算；宽矩阵更容易提供可切分工作，但设备间要通信；
- **pipeline parallel（流水线并行）**：把不同层放在不同设备，让把一个大 batch 切成的多个小批次（microbatches）像流水线一样错开前进；
- **bubble（流水线气泡）**：流水线尚未填满、正在排空或等待依赖时，某些设备没有有效工作的一段空闲时间。

这些概念解释为什么参数量相近的深窄与宽浅模型可能有不同实际速度。

**【课程，PDF 第 45–46 页】**第 45 页的流水线图用 4 个 layer 顺序跨 GPU：forward 必须由 layer 0 走到 layer 3，backward 反向返回。它表达：

- depth 方向有数据依赖，单个样本不能让后层越过前层；
- 极深模型会增加串行 latency；
- pipeline parallelism 可以分设备放层，但存在流水线填充、排空和 bubble。

第 46 页左图：

- 横轴是 $`d_{model}/n_{layer}`$，对数尺度；
- 三条曲线对应约 50M、274M、1.5B 参数；
- 纵向曲线表现模型质量/损失，低谷覆盖较宽的 aspect ratio；
- 极端很深或很宽处变差，但没有一根所有规模共享的尖锐最佳线。

右侧 Tay et al. 2021 的四幅图：

- 横轴为 FLOPs；
- 上方纵轴为 negative log perplexity，下方为 success accuracy；
- 圆点大小表示参数量；
- 不同宽深配置随计算增加都可能改善。

图支持“较宽范围可达到相近表现，系统约束会影响最终选择”。它没有单独给出一条适用于所有数据与训练预算的解析最优式。

**【视频补充，约 [51:31](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3091s)–[54:18](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3258s)】**老师强调：宽度通常更容易做 tensor parallel，深度更适合切 pipeline 但增加串行性；最终选择常由硬件拓扑和 latency 目标推动。

### 13.8 参数/shape 例 1：标准内宽、普通 FFN、12 层

**【补充例子；合成配置，不对应某个具体模型】**设：

```text
d=768, H=12, d_h=64, L=12
d_ff=3072, V=50,000
```

先核对 head ratio：

```math
H d_h=12\times64=768,
```

所以：

```math
r_{attn}=768/768=1.
```

若 $`B=2,T=4`$：

```text
X                         [2,4,768]
Q/K/V after split         [2,12,4,64]
score                     [2,12,4,4]
merged attention          [2,4,768]
FFN hidden                [2,4,3072]
block output              [2,4,768]
```

每层 attention 主权重：

```math
4d^2=4\times768^2.
```

先算：

```math
768^2=589{,}824,
```

所以：

```math
N_{attn}=4\times589{,}824=2{,}359{,}296.
```

普通 FFN：

```math
N_{ffn}=2\times768\times3{,}072=4{,}718{,}592.
```

每 block：

```math
2{,}359{,}296+4{,}718{,}592=7{,}077{,}888.
```

12 个 block：

```math
7{,}077{,}888\times12=84{,}934{,}656.
```

token embedding：

```math
Vd=50{,}000\times768=38{,}400{,}000.
```

假设输出权重与 embedding tied（共享），忽略 norm 和 bias：

```math
N_{total}\approx84{,}934{,}656+38{,}400{,}000
=123{,}334{,}656.
```

aspect ratio：

```math
A=d/L=768/12=64.
```

### 13.9 参数/shape 例 2：更深、gated FFN

**【补充例子；合成配置】**设：

```text
d=512, H=8, d_h=64, L=24
d_g=1536, V=32,000
```

head ratio：

```math
8\times64/512=512/512=1.
```

若 $`B=1,T=8`$：

```text
Q/K/V                    [1,8,8,64]
score                    [1,8,8,8]
up branch                [1,8,1536]
gate branch              [1,8,1536]
gated product            [1,8,1536]
block output             [1,8,512]
```

每层 attention：

```math
4\times512^2=4\times262{,}144=1{,}048{,}576.
```

每层 gated FFN：

```math
3\times512\times1{,}536=2{,}359{,}296.
```

每 block：

```math
1{,}048{,}576+2{,}359{,}296=3{,}407{,}872.
```

24 层：

```math
3{,}407{,}872\times24=81{,}788{,}928.
```

共享 embedding：

```math
32{,}000\times512=16{,}384{,}000.
```

粗略总数：

```math
81{,}788{,}928+16{,}384{,}000=98{,}172{,}928.
```

aspect ratio：

```math
512/24\approx21.33.
```

它比【例 1】更深、更窄；参数相近不表示 latency 相同。

### 13.10 参数/shape 例 3：attention 内宽为模型宽 2 倍

**【补充例子；合成配置】**设：

```text
d=1024, H=16, d_h=128, L=8
d_ff=4096, V=16,000
```

attention 内宽：

```math
Hd_h=16\times128=2{,}048.
```

比例：

```math
r_{attn}=2{,}048/1{,}024=2.
```

若 $`B=2,T=3`$：

```text
X                         [2,3,1024]
Q/K/V before split        [2,3,2048]
Q/K/V after split         [2,16,3,128]
score                     [2,16,3,3]
heads merged              [2,3,2048]
after W_O                 [2,3,1024]
FFN hidden                [2,3,4096]
```

每个 Q/K/V 投影参数：

```math
1{,}024\times2{,}048=2{,}097{,}152.
```

三个共：

```math
3\times2{,}097{,}152=6{,}291{,}456.
```

输出投影：

```math
2{,}048\times1{,}024=2{,}097{,}152.
```

attention 总计：

```math
6{,}291{,}456+2{,}097{,}152=8{,}388{,}608.
```

也可由公式核对：

```math
4r_{attn}d^2=4\times2\times1{,}024^2=8{,}388{,}608.
```

普通 FFN：

```math
2\times1{,}024\times4{,}096=8{,}388{,}608.
```

每 block：

```math
16{,}777{,}216.
```

8 层：

```math
16{,}777{,}216\times8=134{,}217{,}728.
```

共享 embedding：

```math
16{,}000\times1{,}024=16{,}384{,}000.
```

粗略总数：

```math
134{,}217{,}728+16{,}384{,}000=150{,}601{,}728.
```

aspect ratio：

```math
1{,}024/8=128.
```

这组例子展示：不能看到 `d=1024` 就自动假定 Q/K/V 合并宽也是 1024。

---

## 14. vocabulary size：词表大，序列可能短，但两端矩阵也变大

### 14.1 词表大小同时影响输入和输出

**【课程，PDF 第 47 页】**词表大小记为 $`V`$。

输入 embedding 表：

```math
E\in\mathbb R^{V\times d},
```

参数数：

```math
N_{embed}=Vd.
```

输出 projection：

```math
W_{out}\in\mathbb R^{d\times V},
```

参数数也是：

```math
N_{out}=dV.
```

若 weight tying（权重绑定/共享）让 $`W_{out}=E^{\mathsf T}`$，两处共用同一组 $`Vd`$ 参数；若不共享，则两组共 $`2Vd`$。

### 14.2 `V=32,000,d=4,096` 的完整显存账

**MiB（mebibyte，二进制兆字节）**满足：

```math
1\ \text{MiB}=2^{20}=1{,}048{,}576\ \text{bytes}.
```

它与十进制 MB（$`10^6`$ bytes）不同。显存百分比和“能否装下模型”取决于使用哪种单位，必须写清。

**【补充例子】**embedding 参数：

```math
32{,}000\times4{,}096=131{,}072{,}000.
```

若用 BF16，每参数 2 bytes：

```math
131{,}072{,}000\times2
=262{,}144{,}000\ \text{bytes}.
```

换成 MiB：

```math
262{,}144{,}000/1{,}048{,}576=250\ \text{MiB}.
```

若输入输出不共享：

```math
2\times131{,}072{,}000=262{,}144{,}000
```

个参数，BF16 是：

```math
500\ \text{MiB}.
```

这还不含梯度和 optimizer state。

### 14.3 把词表扩大 4 倍会怎样

若 $`V`$ 从 32,000 增到 128,000，而 $`d=4,096`$ 不变：

```math
128{,}000\times4{,}096=524{,}288{,}000
```

个共享 embedding 参数。BF16 字节：

```math
524{,}288{,}000\times2
=1{,}048{,}576{,}000\ \text{bytes}
=1{,}000\ \text{MiB}.
```

词表变 4 倍，embedding 参数和每位置 logits 宽度也变 4 倍：

```text
小词表 logits    [B,T,32,000]
大词表 logits    [B,T,128,000]
```

### 14.4 大词表的好处与代价

大词表可能：

- 把常见词、非拉丁文字或代码片段用更少 token 表示；
- 缩短同一段文本的 sequence length，减少主干层处理的 token 数；
- 让多语言字符不必大量退化为字节碎片。

代价：

- embedding/output 参数增加；
- 每个位置要产生更多 logits；
- 输出 softmax 的计算与临时数据更大；
- 稀有 token 的训练样本可能更少；
- tokenizer 设计和数据混合变得更复杂。

所以“大词表减少序列长度”与“大词表增加两端成本”必须一起看。

### 14.5 课程表格怎样读

**【课程，PDF 第 47 页；2026 课程汇总的跨年代公开配置】**部分表中值：

| 类型/模型 | 词表大小 |
|---|---:|
| 原始 Transformer | 37,000 |
| GPT | 40,257 |
| GPT-2/3 | 50,257 |
| T5/T5 v1.1 | 32,128 |
| LLaMA | 32,000 |
| mT5 | 250,000 |
| PaLM | 256,000 |
| GPT-4 | 100,276 |
| Gemma 4 | 262,144 |
| DeepSeek | 100,000 |
| Yi | 64,000 |

课程将单语模型的传统常见带概括为约 `30k–50k`，多语种/生产系统常见约 `100k–250k`。这只是样本概括：表中的 Gemma 4 为 262,144，已经略高于 250k；不同 tokenizer 的 normalization、byte fallback 和特殊 token 也会改变可比性。

**【视频补充，约 [55:11](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3311s)】**老师把多语言覆盖视为大词表的重要推动因素；更大的模型也更有参数预算承受大词表。

### 14.6 课堂问答：怎样公平比较 tokenizer

**【视频补充，约 [57:12](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3432s)】**课堂问到 tokenizer 质量比较。老师提到 bits per byte（每字节比特）可作为较可比的语言建模度量，但前提是：

- 两个系统覆盖同样的原始字节信息；
- normalization 没有不可逆地丢掉不同内容；
- 不能只看 token 数，因为不同词表定义了不同 token 单位。

本讲不重新展开 tokenization；这里要记住：token 数短不自动等于语言建模质量高。

---

## 15. dropout 与 weight decay：一个丢 activation，一个改更新

### 15.1 regularization 不只是“防止背训练集”

**regularization（正则化）**是对训练过程加入约束或噪声的一大类方法。小数据监督学习里常说它防 overfitting（过拟合）；大模型预训练却有不同语境：

- 训练数据可能有万亿 token；
- 常只对语料做一遍或很少几遍；
- 参数未必有机会反复记住每条样本；
- 但正则手段仍会改变梯度噪声、权重尺度和可用学习率。

因此“没明显过拟合”不能推出“正则化对优化完全没作用”。

### 15.2 dropout 的公式与手算

**dropout（随机失活）**在训练时随机把一些 activation 设为 0。常用 inverted dropout：

```math
y_i=\frac{m_i}{1-p}x_i,
```

其中：

- $`p`$：丢弃概率；
- $`m_i\in\{0,1\}`$：随机 mask；保留概率是 $`1-p`$；
- 除以 $`1-p`$：让训练输出的期望保持不变。

**【补充例子】**令：

```math
x=[2,4,-6],\qquad p=0.5,
```

某一次抽到 mask：

```math
m=[1,0,1].
```

因为：

```math
\frac{1}{1-p}=\frac{1}{0.5}=2,
```

所以：

```math
y=[1\times2\times2,\;0\times2\times4,\;1\times2\times(-6)]
=[4,0,-12].
```

这一次的数字不是原输入，但取很多随机 mask 的平均：

```math
\mathbb E\left[\frac{m_i}{1-p}x_i\right]
=\frac{\mathbb E[m_i]}{1-p}x_i
=\frac{1-p}{1-p}x_i=x_i.
```

推理时通常关闭 dropout，直接使用完整 activation。它不会永久删除参数，也不等同于把模型结构剪枝。

### 15.3 weight decay 的最小更新式

**weight decay（权重衰减）**在更新时把参数向 0 缩一点。用简单 SGD 方向说明 decoupled 形式：

```math
w_{new}=(1-\eta\lambda)w-\eta g.
```

- $`w`$：更新前参数；
- $`g`$：本步优化方向；在简单 SGD 中就是梯度；
- $`\eta`$：learning rate（学习率）；
- $`\lambda`$：weight decay 系数；
- $`(1-\eta\lambda)w`$：先缩权重；
- $`-\eta g`$：再沿优化方向走。

AdamW 中的梯度方向会经过一阶、二阶 moment 归一化；这里用 $`g`$ 只是为了让四则运算可见，衰减项的“与优化方向分开”不变。

### 15.4 小向量的一次 weight-decay 更新

**【补充例子】**令：

```math
w=[2,-1],\quad g=[0.3,-0.4],\quad
\eta=0.1,\quad\lambda=0.01.
```

第 1 步，衰减乘数：

```math
1-\eta\lambda
=1-0.1\times0.01
=1-0.001
=0.999.
```

第 2 步，缩权重：

```math
0.999w=[1.998,-0.999].
```

第 3 步，学习率乘优化方向：

```math
\eta g=0.1[0.3,-0.4]=[0.03,-0.04].
```

第 4 步，相减：

```math
w_{new}
=[1.998,-0.999]-[0.03,-0.04]
=[1.968,-0.959].
```

若没有 decay：

```math
w-\eta g
=[2,-1]-[0.03,-0.04]
=[1.97,-0.96].
```

差值：

```math
[1.968,-0.959]-[1.97,-0.96]=[-0.002,0.001].
```

它正是把原参数 `[2,-1]` 额外乘了 $`-\eta\lambda=-0.001`$ 得到的 `[-0.002,0.001]`。

### 15.5 dropout 与 weight decay 不可互换

| 比较 | dropout | weight decay |
|---|---|---|
| 作用对象 | 当前 batch 的 activation | 跨 step 保存的 parameter |
| 随机吗 | 通常随机 mask | 更新式本身确定 |
| 训练时 | 丢一部分 activation 并缩放 | 每步把权重向 0 缩 |
| 推理时 | 通常关闭 | 已经影响了训练出的权重，不额外随机 |
| 主要超参 | 丢弃概率 $`p`$ | 系数 $`\lambda`$，并与 $`\eta`$ 联动 |

### 15.6 课程模型调查的时间边界

**【课程，PDF 第 48–50 页；2026 讲义对公开报告的汇总】**表中：原始 Transformer、GPT-2、T5、GPT-3、OPT 等较早模型常报告 dropout 0.1；T5 v1.1、PaLM、LLaMA 等报告或开源配置中 dropout 为 0；不少模型仍使用 weight decay 0.1。Qwen 14B 在表中同时为 dropout 0.1、weight decay 0.1。

幻灯片脚注非常重要：论文常不讨论 dropout；对开源模型可从配置确认，对 closed model，“没写”不一定等于 0。因此不能把缺失报告补成确定事实。

### 15.7 为什么 weight decay 可能主要影响优化动态

**【课程/视频补充，约 [59:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3561s)–[62:24](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3744s)】**课程引用 Andriushchenko et al. 2023 的观察：LLM 中 weight decay 的作用不能只说成控制 overfitting，它会与 learning-rate schedule（学习率日程）相互作用。

从更新式即可看到：

```math
\text{本步衰减强度}=\eta_t\lambda.
```

若 cosine schedule 让 $`\eta_t`$ 后期变小，即使 $`\lambda`$ 不变，每步的乘法收缩也变弱。若同时调整 $`\lambda`$，整体权重尺度轨迹还会改变。这可能影响可用学习率、梯度尺度和稳定性。

结论应写成：

> 现代大规模预训练往往少用 dropout，但常保留 weight decay；其收益常通过优化动态体现，而不只是传统“防过拟合”。

这仍是经验总结，不是说 weight decay 对泛化永远没有影响。

---

## 16. softmax 附近的稳定性：别让指数把训练炸掉

### 16.1 课程稳定性图先读坐标轴

**【课程，PDF 第 52 页】**图比较两个约 7B 模型训练轨迹：

- 横轴：训练 step，从 0 到约 600,000；
- 上图纵轴：loss，约 2.0–3.0；
- 下图纵轴：gradient 的 L2 norm，约 0–3；
- 蓝线 loss 下降较快，但频繁出现尖峰；其梯度 L2 norm 也不断尖峰；
- 橙线整体更平滑。

课程用“不要训练成蓝线”强调：最终 loss 低一点不代表训练过程健康。频繁尖峰可能导致发散、回滚 checkpoint 或浪费巨额计算。

这张图是具体训练日志示例，不代表所有蓝色曲线或所有 7B 模型。

### 16.2 softmax 的两个危险动作

softmax：

```math
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
```

危险来自：

1. $`z_i`$ 很大时，$`e^{z_i}`$ 可能 overflow（上溢）成无穷；
2. $`z_i`$ 都很负时，指数可能 underflow（下溢）成 0，分母接近 0；
3. 极大 logits 还会让概率接近 one-hot、梯度尺度恶化。

语言模型有两个重要 softmax：

- attention softmax：在 key 位置轴上归一；
- output softmax：在词表轴 $`V`$ 上归一。

### 16.3 普通小 logits 的完整 softmax

**【补充例子 1】**令：

```math
z=[1,2,3].
```

指数近似：

```math
e^1\approx2.7183,\quad
e^2\approx7.3891,\quad
e^3\approx20.0855.
```

分母：

```math
Z=2.7183+7.3891+20.0855=30.1929.
```

概率：

```math
p_1=2.7183/30.1929\approx0.0900,
```

```math
p_2=7.3891/30.1929\approx0.2447,
```

```math
p_3=20.0855/30.1929\approx0.6652.
```

检查：

```math
0.0900+0.2447+0.6652\approx0.9999,
```

差 0.0001 来自小数截断。

### 16.4 减最大值为什么不改概率

对所有 logits 同时减常数 $`c`$：

```math
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=\frac{e^{z_i}e^{-c}}{e^{-c}\sum_j e^{z_j}}
=\frac{e^{z_i}}{\sum_j e^{z_j}}.
```

分子分母的共同因子 $`e^{-c}`$ 约掉了。

数值稳定实现选择：

```math
c=\max_j z_j.
```

这样最大的移位 logit 是 0，其他都不大于 0，指数不会因很大的正数上溢。

### 16.5 `[1000,1001,1002]` 的稳定手算

**【补充例子 2】**直接计算 $`e^{1000}`$ 在常见浮点格式会溢出。先减最大值 1002：

```math
z-\max(z)=[-2,-1,0].
```

指数：

```math
e^{-2}\approx0.1353,\quad
e^{-1}\approx0.3679,\quad
e^0=1.
```

分母：

```math
0.1353+0.3679+1=1.5032.
```

概率：

```math
[0.1353,0.3679,1]/1.5032
\approx[0.0900,0.2447,0.6652].
```

它与 `[1,2,3]` 的概率相同，因为两组 logits 只差共同常数 999。

### 16.6 全是大负数也用同一招

**【补充例子 3】**令：

```math
z=[-1000,-1001].
```

直接指数可能都下溢为 0，形成 `0/0`。最大值是 -1000，减去它：

```math
[0,-1].
```

指数 `[1,0.3679]`，分母：

```math
1+0.3679=1.3679.
```

概率：

```math
[1/1.3679,0.3679/1.3679]
\approx[0.7311,0.2689].
```

### 16.7 log-sum-exp 与 output z-loss

定义 softmax normalizer：

```math
Z=\sum_j e^{z_j}.
```

其对数：

```math
\log Z=\log\left(\sum_j e^{z_j}\right)
```

叫 log-sum-exp。若正确类别是 $`y`$，cross-entropy 可写成：

```math
L_{CE}=-z_y+\log Z.
```

给所有 logits 加常数 $`c`$：

```math
z_j'=z_j+c.
```

softmax 概率不变，但：

```math
\log Z'
=\log\left(\sum_j e^{z_j+c}\right)
=\log\left(e^c\sum_j e^{z_j}\right)
=c+\log Z.
```

概率“不在乎”共同漂移，数值格式却在乎 logits 是否越来越大。

z-loss 加一项：

```math
L_z=\alpha(\log Z)^2,
```

- $`\alpha`$：很小的权重；
- 平方使正负偏离都受罚；
- 最低点在 $`\log Z=0`$，即 $`Z=1`$ 附近。

总训练目标：

```math
L=L_{CE}+L_z.
```

### 16.8 z-loss 的两组相同概率、不同惩罚

**【补充例子】**为了看清差异，手算取较大的 $`\alpha=0.01`$；真实模型常小得多。

第一组：

```math
z=[0,0].
```

概率是 `[0.5,0.5]`，而：

```math
Z=e^0+e^0=2,
```

```math
\log Z=\log2\approx0.6931.
```

z-loss：

```math
L_z=0.01\times0.6931^2
\approx0.01\times0.4805
=0.004805.
```

第二组给每个 logit 加 10：

```math
z'=[10,10].
```

概率仍 `[0.5,0.5]`，但：

```math
\log Z'=10+\log2\approx10.6931.
```

z-loss：

```math
L_z'=0.01\times10.6931^2
\approx0.01\times114.342
\approx1.14342.
```

所以 z-loss 给优化器一个理由阻止整组 logits 无意义地共同变大。

**【课程/视频补充，PDF 第 54 页；约 [67:06](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4026s)–[69:23](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4163s)】**课程追溯到 Devlin 2014，并列出 PaLM、Baichuan 2、DCLM、OLMo 2/3 等采用案例。PaLM 报告的系数例为 $`10^{-4}`$；这是具体配方，不是通用最佳值。

z-loss 是 **训练目标附加项**：推理时不需要再给 loss 加它；但它已经改变了训练出的参数。

### 16.9 QK norm：在 attention softmax 前控制 Q、K 尺度

**【课程，PDF 第 55 页】**标准 attention logit：

```math
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_h}}.
```

若 $`q_i,k_j`$ 的长度不断增大，点积也可能非常大。QK norm 在点积前分别归一化：

```math
\hat q_i=\mathrm{Norm}(q_i),
```

```math
\hat k_j=\mathrm{Norm}(k_j),
```

```math
s_{ij}=\frac{\hat q_i\cdot\hat k_j}{\sqrt{d_h}}.
```

它是 **attention 结构改动**，不只是 loss 正则项。

### 16.10 QK norm 的二维手算

**【补充例子】**取：

```math
q=[3,4],\qquad k=[6,8],\qquad d_h=2.
```

未归一化点积：

```math
q\cdot k=3\times6+4\times8=18+32=50.
```

缩放后 logit：

```math
50/\sqrt2\approx50/1.4142\approx35.355.
```

现在用不带可学习 scale、忽略 $`\varepsilon`$ 的 RMSNorm 手算。q 的 RMS：

```math
\mathrm{RMS}(q)
=\sqrt{\frac{3^2+4^2}{2}}
=\sqrt{\frac{9+16}{2}}
=\sqrt{12.5}
\approx3.5355.
```

所以：

```math
\hat q\approx[3/3.5355,4/3.5355]
\approx[0.8485,1.1314].
```

k 的 RMS：

```math
\mathrm{RMS}(k)
=\sqrt{\frac{6^2+8^2}{2}}
=\sqrt{50}
\approx7.0711.
```

所以：

```math
\hat k\approx[0.8485,1.1314].
```

归一化后点积：

```math
\hat q\cdot\hat k
\approx0.8485^2+1.1314^2
\approx0.72+1.28=2.
```

再除 $`\sqrt2`$：

```math
s\approx2/1.4142\approx1.4142.
```

同方向但整体放大 2 倍的 k，在归一化后变成相同方向尺度；logit 从约 35.355 降到约 1.414。

真实 RMSNorm 有 $`\varepsilon`$ 和可学习 $`\gamma`$，所以 QK norm 是“控制尺度”，不是证明所有 logits 永远被硬限制在某个常数内。

**【视频补充，约 [69:27](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4167s)–[71:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4300s)】**老师把它概括为在进入 QK 矩阵乘前再放 norm，使 attention softmax 输入尺度更一致；课程将其视为近年常见的稳定化手段。幻灯片列出的模型名是 2026 时点的样本，不构成“所有模型都必须用”的证明。

QK norm 若是模型架构的一部分，训练和推理都要执行。

### 16.11 logit soft-capping：不是截断，是平滑压到范围内

**【课程，PDF 第 56 页】**给定 cap $`c>0`$：

```math
\mathrm{softcap}(z)
=c\tanh\left(\frac{z}{c}\right).
```

因为：

```math
-1<\tanh(u)<1,
```

所以：

```math
-c<\mathrm{softcap}(z)<c.
```

它不是 `min(max(z,-c),c)` 的硬截断；tanh 平滑压缩，大数逐渐靠近边界。

### 16.12 `c=2` 的 soft-cap 与概率手算

**【补充例子】**令：

```math
z=[-10,0,10],\qquad c=2.
```

先除 cap：

```math
z/c=[-5,0,5].
```

近似：

```math
\tanh(-5)\approx-0.99991,\quad
\tanh(0)=0,\quad
\tanh(5)\approx0.99991.
```

乘 2：

```math
z_{cap}\approx[-1.9998,0,1.9998].
```

对 capped logits 做 softmax。指数约为：

```math
[e^{-1.9998},e^0,e^{1.9998}]
\approx[0.1354,1,7.3876].
```

分母：

```math
0.1354+1+7.3876=8.5230.
```

概率约为：

```math
[0.0159,0.1173,0.8668].
```

若不用 cap，`[-10,0,10]` 的最大项概率约 0.99995。soft-cap 明显限制了过分自信，也确实改变了模型能表达的分布。

### 16.13 小 logits 也会被轻微改变

再取 $`z=[-1,0,1],c=2`$：

```math
z/c=[-0.5,0,0.5].
```

因为 $`\tanh(0.5)\approx0.4621`$：

```math
z_{cap}\approx[-0.9242,0,0.9242].
```

原 logits 的 softmax 约为：

```math
[0.0900,0.2447,0.6652].
```

capped 后指数约 `[0.3968,1,2.5197]`，分母：

```math
0.3968+1+2.5197=3.9165,
```

概率约：

```math
[0.1013,0.2553,0.6434].
```

因此 soft-cap 不是只在数值已经溢出时才起作用；它会改变正常范围 logits 的梯度和概率，只是小值时变化较温和。

### 16.14 课程表格怎样读，不夸大 soft-cap

**【课程，PDF 第 56 页；讲义引用的具体实验】**幻灯片引用 Gemma 风格配置：attention logits cap 50、final logits cap 30。下方消融表的 perplexity：baseline 11.19，单独 soft-cap 11.24，而若干 QK/QKV norm 配置约 10.8–11.0。

图表想表达：soft-cap 是强干预，能提高安全边界，但单独使用可能损害质量；QK norm 在该实验里权衡更好。不能把这几个 perplexity 数字直接迁移到别的数据、规模和训练配方。

**【视频补充，约 [72:07](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4327s)–[73:50](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4430s)】**老师称它更像 Google/Gemma 系特定技巧，而不如 QK norm 普遍。其理由正是：cap 会限制 softmax 可表达的置信度。

如果模型定义在 attention 或 final logits 上使用 soft-cap，训练与推理必须一致执行；它不是只在 loss 中出现的训练正则项。

### 16.15 三种稳定手段不要混在一起

| 手段 | 作用位置 | 是否改变 forward | 推理时是否执行 |
|---|---|---|---|
| z-loss | 训练 objective，加 $`\alpha(\log Z)^2`$ | 概率公式本身不变 | 不计算该 loss 项 |
| QK norm | Q/K 投影后、点积前 | 改变 attention logits | 是 |
| logit soft-cap | softmax 前的 logits | 直接压缩 logits | 若架构定义使用，则是 |

### 16.16 本节最终因果链

```text
softmax 用指数
  ↓
共同放大的 logits 不改概率，却增加上溢和尺度风险
  ↓
减 max：每次 softmax 都应做的数值稳定实现
z-loss：给共同漂移一个训练惩罚
QK norm：控制 attention 点积输入的尺度
soft-cap：直接平滑限制 logits，但可能牺牲表达/质量
```

“减 max”不改变数学概率；后面三项会影响训练路径或模型函数，不能把它们当成完全免费的等价重写。

---

## 17. MHA、prefill 与 incremental decode：先把缓存对象认准

### 17.1 MHA 的四个整数

**【课程，PDF 第 57–60 页】**MHA 是 Multi-Head Attention（多头注意力）。这一节使用课程符号：

- $`b`$：batch size；
- $`n`$：当前序列长度；
- $`h`$：attention head 数；
- $`k`$：每个 head 的维度，也就是前文的 $`d_h`$；
- $`d`$：模型宽度；课程这几页先假设 $`d=hk`$。

在标准 MHA 中，每个 query head 都有自己的一组 K 和 V head，因此：

```math
H_q=H_{kv}=h.
```

这里：

- $`H_q`$ 是 query head 数；
- $`H_{kv}`$ 是 key/value head 数；
- 标准 MHA 中二者相等；下一节的 MQA/GQA 才会让它们不同。

### 17.2 从 `[b,n,d]` 投影到每个 head

输入：

```math
X\in\mathbb R^{b\times n\times d}.
```

三个投影：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
```

在 $`d=hk`$ 的 MHA 中：

```text
X                         [b,n,d]
Q/K/V before split        [b,n,h*k] = [b,n,d]
Q/K/V after split         [b,h,n,k]
attention logits          [b,h,n,n]
attention probabilities   [b,h,n,n]
weighted values           [b,h,n,k]
merged heads              [b,n,h*k] = [b,n,d]
```

`attention logits[b_index, head_index, query_pos, key_pos]` 是一个标量；不要把最后的 `[n,n]` 误认为 hidden features。

### 17.3 一个 head 的三位置 attention 手算

**【补充例子】**只看一个 head，令 $`k=2`$。第三个位置的 query：

```math
q_2=[1,1].
```

三个 key：

```math
k_0=[1,0],\quad k_1=[0,1],\quad k_2=[1,1].
```

因为 query 在位置 2，causal mask 允许它看位置 0、1、2。三个点积：

```math
q_2\cdot k_0=1\times1+1\times0=1,
```

```math
q_2\cdot k_1=1\times0+1\times1=1,
```

```math
q_2\cdot k_2=1\times1+1\times1=2.
```

除以 $`\sqrt{k}=\sqrt2\approx1.4142`$：

```math
s\approx[0.7071,0.7071,1.4142].
```

为稳定 softmax，减最大值 1.4142：

```math
s'\approx[-0.7071,-0.7071,0].
```

指数：

```math
e^{-0.7071}\approx0.4931,\qquad e^0=1.
```

分母：

```math
0.4931+0.4931+1=1.9862.
```

attention 权重约为：

```math
a\approx[0.2483,0.2483,0.5035].
```

取 values：

```math
v_0=[10,0],\quad v_1=[0,10],\quad v_2=[10,10].
```

加权和第一维：

```math
0.2483\times10+0.2483\times0+0.5035\times10
=2.483+0+5.035=7.518.
```

第二维：

```math
0.2483\times0+0.2483\times10+0.5035\times10
=0+2.483+5.035=7.518.
```

输出约 `[7.518,7.518]`，shape 仍为 `[k]=[2]`。多头版本对每个 head 做同类运算，再合并。

### 17.4 prefill 是什么

**prefill（提示词预填充）**是推理的第一阶段：用户一次给出整段 prompt，模型并行处理这些已知 token，并为每层建立 K/V。

若 prompt 长度 $`n=4`$：

```text
输入 token 0,1,2,3 一次送进模型
Q/K/V                     [b,h,4,k]
causal score              [b,h,4,4]
每个位置只使用自己和过去位置
缓存 K 与 V               各 [b,h,4,k]
```

“并行处理”不表示取消 causal mask。位置 3 可以看 0–3，位置 0 仍不能看 1–3；只是 GPU 能把许多 query 的矩阵运算一起做。

训练时所有 target token 也已知，数据流与 prefill 相似，都可并行计算整段 Q/K/V；训练还要保存反向传播所需内容，推理 prefill 不需要训练梯度。

### 17.5 incremental decode 为什么必须一步一步

**incremental decode（增量解码）**是生成阶段：

```text
根据 prompt 生成 token 4
把 token 4 加进上下文，再生成 token 5
把 token 5 加进上下文，再生成 token 6
...
```

上面 `4,5,6` 是承接位置 `0,1,2,3` 的 **0-based 位置编号**。下面公式为避免歧义，改用第 1、2、…、$`n`$ 个 token 的 **1-based 个数**。

token 5 的输入依赖刚生成的 token 4，所以同一条序列的未来 token 不能提前全部并行计算。这是 autoregressive（自回归）依赖。

为避免 off-by-one（差一位）歧义，本节统一使用 **1-based 计数**：本步正在处理“第 $`n`$ 个已知 token”。进入本层、尚未加入它时，cache 中已有前 $`n-1`$ 个 token；算出本 token 的 K/V 并 append 后，cache 总长度变成 $`n`$。随后这个新 query 看 append 后的全部 $`n`$ 个 keys，并产生用来预测第 $`n+1`$ 个 token 的输出。

```text
K/V cache before append  [b,h,n-1,k]
new hidden state          [b,1,d]
new Q                     [b,h,1,k]
new K                     [b,h,1,k]
new V                     [b,h,1,k]
K cache after append      [b,h,n,k]
V cache after append      [b,h,n,k]
new attention logits      [b,h,1,n]
new attention output      [b,h,1,k]
```

只需要为新 token 计算一行 query-to-past 分数，不必重算过去 token 的 K/V。

**从这里到 §19，cache 大小和 decode 强度公式中的 $`n`$ 都指 append 后的总序列长度**，不是 append 前的 $`n-1`$。

### 17.6 KV cache 真正保存什么

**KV cache（键值缓存）**在每个 attention layer 保存过去 token 投影后的：

- K tensor；
- V tensor。

它通常 **不保存**：

- 所有过去的 Q；旧 query 不会被未来 query 复用；
- 完整 $`QK^{\mathsf T}`$ 分数矩阵；新 query 的分数要与 cached K 重新点积；
- softmax 概率矩阵。

**【视频口头表述精确化，约 [77:23](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4643s)】**视频有一段把 cache 口语化描述成保留过去的“keys and queries”或已算子矩阵。严格实现应以上述 K/V 为准；缓存名称本身也是 KV，而不是 QKV 或 score cache。

### 17.7 MHA 的 KV cache 元素数

每层：

```text
K cache    [b,h,n,k]  → b*h*n*k elements
V cache    [b,h,n,k]  → b*h*n*k elements
```

相加：

```math
N_{cache,layer}=2bnhk.
```

这里 $`n`$ 是 append 后 cache 中已有的 token 总数，与 §17.5 的统一约定一致。

若每元素占 $`s`$ bytes：

```math
M_{cache,layer}=2bnhk\,s\ \text{bytes}.
```

若有 $`L`$ 层且每层同配置：

```math
M_{cache,total}=2Lbnhk\,s\ \text{bytes}.
```

### 17.8 一个微型 cache 手算

**【补充例子】**令：

```text
b=1, n=4, h=4, k=2, dtype=BF16
BF16 每元素 2 bytes
```

K 元素：

```math
1\times4\times4\times2=32.
```

V 也是 32，所以总元素：

```math
2\times32=64.
```

字节：

```math
64\times2=128\ \text{bytes/layer}.
```

若 3 层：

```math
128\times3=384\ \text{bytes}.
```

小数字只为看清乘法；真实长上下文模型会达到 MiB 或 GiB。

### 17.9 cache 公式没有算什么

上式是理想 K/V payload。真实推理服务还可能有：

- 分页/块分配产生的未用空间；
- batch slot padding；
- 对齐、索引和元数据；
- cache quantization 的 scale/zero-point；
- beam search 或多候选复制；
- tensor parallel 分片后的通信 buffer。

所以它是理解缩减倍数的核心账，不是整个服务器显存账单。

---

## 18. MQA 与 GQA：减少的是 KV heads，不是 query 的维度

### 18.1 三种结构先用等式定义

**【课程，PDF 第 61–63 页】**令：

- $`H_q`$：query head 数；
- $`H_{kv}`$：key/value head 数；
- $`k`$：每个 head 的 feature 数，head dimension；
- 通常保持 $`H_qk=d`$。

三种结构：

1. **MHA（Multi-Head Attention）**：

   $`H_{kv}=H_q.`$

   每个 query head 有自己的 K/V head。

2. **MQA（Multi-Query Attention，多查询注意力）**：

   $`H_{kv}=1.`$

   所有 query heads 共用一组 K/V head。

3. **GQA（Grouped-Query Attention，分组查询注意力）**：

   $`1<H_{kv}<H_q.`$

   多个 query heads 分成组，每组共用一个 K/V head。

通常要求 $`H_q`$ 可被 $`H_{kv}`$ 整除。每个 KV head 服务的 query head 数：

```math
g=\frac{H_q}{H_{kv}}.
```

### 18.2 “one dimension”绝不表示 `head_dim=1`

**【课程图示精确化，PDF 第 61 页】**幻灯片标题/口语用了 “one dimension for keys and values”。图中真正含义是“一组共享的 K/V head”，不是 key/value 向量只剩 1 个数字。

在 MQA 中可以同时有：

```math
H_{kv}=1,\qquad k=128.
```

K cache shape 是 `[b,1,n,128]`，最后仍有 128 个 features。若误写成 `k=1`，会把表达能力和 cache 大小都算错 128 倍。

### 18.3 三种结构的 Q/K/V shape

统一写法：

```text
Q                  [b,H_q,n,k]
K                  [b,H_kv,n,k]
V                  [b,H_kv,n,k]
```

若 $`H_q=8,H_{kv}=2`$：

```math
g=8/2=4.
```

可以分组：

```text
query heads 0,1,2,3  → 共用 KV head 0
query heads 4,5,6,7  → 共用 KV head 1
```

实现可通过逻辑 broadcast 或专门 kernel 让同一个 K/V head 服务多条 query；不一定真的在内存复制 4 份 K/V。

### 18.4 KV cache 缩减倍数的总公式

沿用 §17.5：$`n`$ 表示本步 K/V append 后 cache 内的 token 总数。

每层元素：

```math
N_{cache}=2bnH_{kv}k.
```

以同一个 $`b,n,H_q,k`$ 的 MHA 为基准：

```math
N_{MHA}=2bnH_qk.
```

缩减倍数：

```math
\frac{N_{MHA}}{N_{GQA/MQA}}
=\frac{2bnH_qk}{2bnH_{kv}k}
=\frac{H_q}{H_{kv}}
=g.
```

batch、长度、K/V 两份和 head width 都约掉，缩减倍数就是每组 query heads 数。

### 18.5 cache 例 1：32 query heads、2048 token、BF16

**【补充例子】**统一配置：

```text
b=1, n=2048, H_q=32, k=128, dtype=BF16=2 bytes
```

#### MHA：$`H_{kv}=32`$

元素：

```math
2\times1\times2{,}048\times32\times128.
```

逐步：

```math
2{,}048\times32=65{,}536,
```

```math
65{,}536\times128=8{,}388{,}608,
```

```math
2\times8{,}388{,}608=16{,}777{,}216\ \text{elements}.
```

字节：

```math
16{,}777{,}216\times2
=33{,}554{,}432\ \text{bytes}
=32\ \text{MiB/layer}.
```

#### GQA：$`H_{kv}=8`$

元素：

```math
2\times1\times2{,}048\times8\times128
=4{,}194{,}304.
```

字节：

```math
4{,}194{,}304\times2
=8{,}388{,}608\ \text{bytes}
=8\ \text{MiB/layer}.
```

缩减：

```math
32/8=4\times.
```

#### MQA：$`H_{kv}=1`$

元素：

```math
2\times1\times2{,}048\times1\times128
=524{,}288.
```

字节：

```math
524{,}288\times2
=1{,}048{,}576\ \text{bytes}
=1\ \text{MiB/layer}.
```

缩减：

```math
32/1=32\times.
```

若有 32 层，只看理想 KV payload：

```text
MHA    32 MiB/layer × 32 = 1024 MiB = 1 GiB
GQA     8 MiB/layer × 32 =  256 MiB
MQA     1 MiB/layer × 32 =   32 MiB
```

### 18.6 cache 例 2：更大 batch 和长度、64 query heads、FP16

**【补充例子】**统一配置：

```text
b=4, n=4096, H_q=64, k=128, dtype=FP16=2 bytes
```

#### MHA：$`H_{kv}=64`$

元素：

```math
2\times4\times4{,}096\times64\times128
=268{,}435{,}456.
```

字节：

```math
268{,}435{,}456\times2
=536{,}870{,}912\ \text{bytes}
=512\ \text{MiB/layer}.
```

#### GQA：$`H_{kv}=8`$

元素：

```math
2\times4\times4{,}096\times8\times128
=33{,}554{,}432.
```

字节：

```math
33{,}554{,}432\times2
=67{,}108{,}864\ \text{bytes}
=64\ \text{MiB/layer}.
```

缩减：

```math
64/8=8\times.
```

#### MQA：$`H_{kv}=1`$

元素：

```math
2\times4\times4{,}096\times1\times128
=4{,}194{,}304.
```

字节：

```math
4{,}194{,}304\times2
=8{,}388{,}608\ \text{bytes}
=8\ \text{MiB/layer}.
```

缩减：

```math
64/1=64\times.
```

两个例子都说明：dtype 改变绝对 bytes；在 dtype 相同的比较里，MHA→GQA/MQA 的倍数只由 $`H_q/H_{kv}`$ 决定。

### 18.7 Q/K/V 投影参数也会减少

统一保留 $`H_q`$ 个 query heads：

- Q projection：$`d\times(H_qk)`$；
- K projection：$`d\times(H_{kv}k)`$；
- V projection：$`d\times(H_{kv}k)`$；
- output projection：$`(H_qk)\times d`$。

主权重总数：

```math
N_{attn}
=dH_qk+2dH_{kv}k+H_qkd
=2dH_qk+2dH_{kv}k.
```

MHA 令 $`H_{kv}=H_q`$：

```math
N_{MHA}=4dH_qk.
```

MQA 令 $`H_{kv}=1`$：

```math
N_{MQA}=2dH_qk+2dk.
```

但部署收益的核心常是 K/V cache 和每步 cache 数据移动，而不仅是少了一些参数。

### 18.8 算术强度先定义单位

**arithmetic intensity（算术强度）**：

```math
I=\frac{F_{ops}}{Q_{traffic}}.
```

- $`F_{ops}`$：算术操作数，课程在这些页只看数量级。**符号复用警告：**它是一个工作量标量，不是 §5 中代表子层的函数 $`F(x)`$；
- $`Q_{traffic}`$：从目标内存层级搬的数据量。**符号复用警告：**它不是 attention 的 query tensor $`Q`$；若以 bytes 计，强度单位是 FLOP/byte；
- $`I`$ 高：每搬一个 byte 做很多计算，较容易用满计算单元；
- $`I`$ 低：搬很多、算很少，较容易被带宽限制。

课程公式把 tensor 元素访问、常数 2、dtype bytes、cache 层级和底层算子合并细节都省略了。因此它们是 **大 O 数量级模型**，不是实测性能数据。

### 18.9 prefill/训练式逐项解释

**【补充解释：首次系统术语盒】**

- **kernel（计算核）**：一次交给 GPU 执行的底层程序单元；相同数学式用不同 kernel，数据读取和同步次数可能完全不同；
- **HBM（High Bandwidth Memory，高带宽显存）**：GPU 上容量较大、但离计算核心比片上存储更远的主显存；把中间 tensor 写入 HBM 再读回常是性能成本；
- **FlashAttention**：一种精确 attention kernel 家族，通过分块在较小片上存储中复用 Q/K/V，避免把完整 $`n\times n`$ score 矩阵写入 HBM。它减少 HBM traffic，并没有把 full attention 的数学连接改成 sparse window。

因此，课件的访存式必须注明它假设的实现；架构公式相同不保证 HBM traffic 相同。

**【课程，PDF 第 58 页】**课程先假设：

```math
n<d,\qquad d=hk.
```

并用 projection 主导的近似：

```math
F_{prefill}=O(bnd^2).
```

人话解释：$`bn`$ 个 token 都乘宽度约 $`d\times d`$ 的投影矩阵。完整 attention 还有 $`O(bn^2d)`$ 的 score/value 计算；因为课件此处假设 $`n<d`$，先把它作为较小项合并掉。

课程列的经典实现内存访问量：

```math
Q_{prefill}
=O(bnd+bhn^2+d^2).
```

三项：

1. $`bnd`$：读写输入/输出 activation；
2. $`bhn^2`$：每个 batch、每个 head 的 $`n\times n`$ attention/softmax 数据；
3. $`d^2`$：投影权重，整批 token 可复用一次加载的权重。

严格把课件两式相除。先算 $`Q/F`$：

```math
\frac{bnd}{bnd^2}=\frac1d,
```

```math
\frac{bhn^2}{bnd^2}
=\frac{hn}{d^2}.
```

因为 $`d=hk`$，所以 $`h=d/k`$：

```math
\frac{hn}{d^2}
=\frac{(d/k)n}{d^2}
=\frac{n}{kd}.
```

最后：

```math
\frac{d^2}{bnd^2}=\frac1{bn}.
```

所以从打印的三项直接得到：

```math
I_{prefill}
=O\left(
\left[
\frac1d+\frac{n}{kd}+\frac1{bn}
\right]^{-1}
\right).
```

幻灯片进一步写成：

```math
O\left(\left[\frac1k+\frac1{bn}\right]^{-1}\right).
```

这里不是严格恒等变形：它把 $`1/d`$ 忽略，并在数量级/保守直觉上用 $`n/d\lesssim1`$ 把 $`n/(kd)`$ 概括为至多约 $`1/k`$。因此要读成：head width 要足够大，且 $`bn`$ 要足够大，prefill 才容易有高复用。

现代 FlashAttention 类 kernel 不把完整 $`bhn^2`$ score 矩阵写入 HBM，实际 HBM 流量模型会不同；课程式用于建立经典 attention 的直觉。

### 18.10 incremental decode 的强度为什么下降

**【课程，PDF 第 59–60 页】**把 append 后总长度从 1 累积到 $`n`$ 的 decode 过程加总；这里的 $`n`$ 仍表示最终 append 后的总长度。课程近似算术工作：

```math
F_{decode}=O(bnd^2).
```

内存访问：

```math
Q_{decode}=O(bn^2d+nd^2).
```

- $`bn^2d`$：随着历史增长，每步读越来越长的 MHA K/V cache；把 1 到 $`n`$ 的历史长度相加是 $`O(n^2)`$；
- $`nd^2`$：每个生成 step 都要读模型投影权重；同一权重可在 batch 内共享，所以这一项不乘 $`b`$。

相除：

```math
I_{decode}
=O\left(\frac{bnd^2}{bn^2d+nd^2}\right).
```

分子分母都除以 $`bnd^2`$：

```math
I_{decode}
=O\left(
\left[
\frac{n}{d}+\frac1b
\right]^{-1}
\right).
```

因此：

- batch $`b`$ 大，可把同一权重加载分摊给更多请求；
- 序列 $`n`$ 相对 $`d`$ 越长，读 cache 的负担越重；
- 小 batch 的逐 token decode 常比 prefill 更 memory-bound。

**【视频补充，约 [75:53](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4553s)–[79:28](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4768s)】**老师用“生成不能并行、参数要一遍遍读”解释 $`1/b`$ 项。这里的“算术操作相同”指对整段生成的数量级仍类似，调度和复用却完全不同。

### 18.11 MQA 的强度式怎样出现 head 数收益

**【课程，PDF 第 61 页】**MQA 只有一个 K/V head，cache 宽从所有 heads 合计的 $`d=hk`$ 降到 $`k`$。课程内存式：

```math
Q_{MQA}
=O(bnd+bn^2k+nd^2).
```

仍取：

```math
F=O(bnd^2).
```

逐项用 $`F`$ 除：

```math
\frac{bnd}{bnd^2}=\frac1d,
```

```math
\frac{bn^2k}{bnd^2}=\frac{nk}{d^2}.
```

因为 $`d=hk`$，所以 $`k=d/h`$：

```math
\frac{nk}{d^2}
=\frac{n(d/h)}{d^2}
=\frac{n}{dh}.
```

权重项：

```math
\frac{nd^2}{bnd^2}=\frac1b.
```

所以：

```math
I_{MQA}
=O\left(
\left[
\frac1d+\frac{n}{dh}+\frac1b
\right]^{-1}
\right).
```

相比 MHA 的 $`n/d`$，cache 项成为 $`n/(dh)`$，理想上缩小 $`h`$ 倍。这表示 memory traffic 变少、算术强度 **升高**，不是算术强度降低。

### 18.12 把 MQA 推广到 GQA

GQA 有 $`H_{kv}`$ 个 KV heads，cache 读取项约为：

```math
bn^2H_{kv}k.
```

令 query head 数 $`H_q=h`$、$`d=hk`$、每组大小：

```math
g=\frac{h}{H_{kv}}.
```

cache 项除以 $`F=bnd^2`$：

```math
\frac{bn^2H_{kv}k}{bnd^2}
=\frac{nH_{kv}k}{d^2}.
```

又因为 $`H_{kv}=h/g`$ 且 $`hk=d`$：

```math
\frac{n(h/g)k}{d^2}
=\frac{nd/g}{d^2}
=\frac{n}{dg}.
```

因此数量级可写为：

```math
I_{GQA}
=O\left(
\left[
\frac1d+\frac{n}{dg}+\frac1b
\right]^{-1}
\right).
```

- MHA：$`g=1`$；
- GQA：$`1<g<h`$；
- MQA：$`g=h`$。

这正是可调折中：组越大，cache 越小，但更多 queries 共用同一份 K/V 表示。

### 18.13 这些数量级式的适用条件

**【补充解释：内存层级】**

- **HBM**：容量最大、离计算核心较远，适合放模型参数和大 tensor；
- **SRAM（Static Random-Access Memory，静态随机存储）**：GPU 芯片上的小容量高速存储，例如 shared memory/cache，用于临时 tile；
- **register（寄存器）**：每个执行线程最靠近算术单元、容量最小的存储，用于眼前几个数。

数据从 HBM 搬一次与在 register 中复用多次不是同一种 traffic。写算术强度 $`F_{ops}/Q_{traffic}`$ 时必须说明 $`Q_{traffic}`$ 针对哪一层，否则同一个 kernel 会得到不同数字。

- **PagedAttention**：把动态增长的 KV cache 按固定大小页块管理，减少连续大块分配和碎片问题；
- **kernel fusion**：把多个相邻算子合进一个 kernel，减少启动次数以及中间 tensor 往返 HBM；
- **profiler（性能分析器）**：实测每个 kernel 的时间、利用率和内存流量，用来检验大 O 模型是否符合真实实现。

必须同时记住：

1. 课程假设 $`n<d`$，因此将 $`bn^2d`$ attention 算术视为次要项；超长上下文 $`n\gg d`$ 时会失真；
2. 常数、读/写次数、K 和 V 的因子 2、dtype bytes 被省略；
3. 公式没有指定 HBM、SRAM 还是 register；谈 FLOP/byte 时必须指定目标内存层级；
4. FlashAttention、PagedAttention、量化 cache、kernel fusion 会改变真实流量；
5. batch 中请求长度不同、padding 和调度会降低理想复用；
6. 这些式子解释趋势，最终 latency 要以实现和 profiler 为准。

### 18.14 表达能力与系统成本的折中

**【课程/视频补充，约 [79:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4772s)–[82:54](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4974s)】**课程引用早期对比图：

- 横向关注 time per sample，越低越好；
- 纵向关注下游模型表现，越高越好；
- MHA 表现最好但推理成本高；
- MQA 成本低但有较明显性能损失；
- GQA 的少量 KV groups 接近 MHA 表现，同时获得大部分推理收益。

这是特定实验年代和设置的经验结果。课程据此解释 2026 调查中 GQA 已很常见；不能把图读成任意模型、任意 KV head 数都无质量损失。

模型必须按选定的 $`H_{kv}`$ 训练。不能训练成 MHA 后，在推理时直接把 K/V heads 随意平均成 MQA 而期望等价。

### 18.15 【延伸】MLA 不是“另一个 GQA 数字”

MLA 是 Multi-head Latent Attention（多头潜变量注意力），由 DeepSeek-V2 等工作推动。核心思想是把 K/V 信息压到一个较低维 latent 表示，cache latent，再通过投影恢复 attention 所需成分。

它与 GQA 都想减少 cache，但机制不同：

- GQA：让多个 query heads 共享离散的 KV heads；
- MLA：通过低秩/潜变量投影压缩 K/V 表示，并要处理位置编码相关部分。

**【课程】**本讲只在第 62 页列出 MLA，视频说下一讲再详细讨论。本笔记也只保留定位，不在没有完整铺垫时展开实现公式。

---

## 19. sparse 与 sliding-window attention：少看一些位置，怎样仍传远信息

### 19.1 full causal attention 实际有多少可见对

长度为 $`n`$ 的 causal self-attention：

```text
位置 0 看 1 个位置
位置 1 看 2 个位置
...
位置 n-1 看 n 个位置
```

总可见 pair：

```math
1+2+\cdots+n=\frac{n(n+1)}2.
```

数量级：

```math
O(n^2).
```

人们常说“full attention 有 $`n^2`$”，是在忽略 causal 三角形的约 1/2 常数；不是说未来位置也真的可见。

### 19.2 sliding window 的定义必须说清是否含自己

本节定义窗口宽 $`w`$ **包含当前 token 自己**。位置 $`i`$ 可看：

```math
j\in[\max(0,i-w+1),i].
```

即当前位和最多 $`w-1`$ 个过去位置。

不同库可能把 `window_size` 定义成“左侧 token 数”，那样总可见数会是 `window_size+1`。使用 API 时必须查约定。

### 19.3 `n=8,w=3` 的每个位置完整列表

**【补充例子】**位置编号 0–7：

| query 位置 | 可见 key 位置 | 数量 |
|---:|---|---:|
| 0 | `[0]` | 1 |
| 1 | `[0,1]` | 2 |
| 2 | `[0,1,2]` | 3 |
| 3 | `[1,2,3]` | 3 |
| 4 | `[2,3,4]` | 3 |
| 5 | `[3,4,5]` | 3 |
| 6 | `[4,5,6]` | 3 |
| 7 | `[5,6,7]` | 3 |

总 pair：

```math
1+2+3+3+3+3+3+3=21.
```

full causal 是：

```math
\frac{8\times9}{2}=36.
```

本例 pair 数从 36 降到 21。

用 $`nw`$ 作上界：

```math
8\times3=24.
```

实际 21 小于 24，因为序列开头还没有足够的过去 token 填满窗口。

### 19.4 为什么复杂度从 $`n^2`$ 变成 $`nw`$

full attention 中，每个约 $`n`$ 个 query 看约 $`n`$ 个 keys：

```math
n\times n=n^2.
```

sliding window 中，每个约 $`n`$ 个 query 最多看 $`w`$ 个 keys：

```math
n\times w=nw.
```

当 $`w`$ 是固定常数或远小于 $`n`$：

```math
O(nw)\ll O(n^2).
```

但 attention 之外的 Q/K/V、FFN projection 仍有 $`O(nd^2)`$ 工作。只有使用真正跳过窗口外位置的 sparse kernel 才能获得相应算术节省；只生成 full score 再把窗口外设为负无穷，仍可能已经付了 $`n^2`$ 计算。

### 19.5 局部层怎样逐层扩大 receptive field

**receptive field（感受野）**：某层某位置的状态最终可能依赖哪些原始输入位置。

在 $`w=3`$ 时，每个 local layer 最多向过去再跨 $`w-1=2`$ 个位置。忽略序列边界，经过 $`\ell`$ 层，最大过去距离约：

```math
\ell(w-1).
```

对位置 7：

```text
1 个 local layer 后：可直接依赖原始位置 5–7
2 个 local layer 后：位置 5 的状态又含 3–5，所以可间接依赖 3–7
3 个 local layer 后：可间接依赖 1–7
4 个 local layer 后：可间接到达 0–7
```

这不是“第 1 层位置 7 直接看见位置 0”。信息要逐层接力，路径更长，可能形成长程信息瓶颈。

### 19.6 sparse attention 不只一种图案

**【课程，PDF 第 64 页】**早期 sparse attention 图案包括：

- local/banded：只看附近；
- strided：每隔固定距离看一个位置；
- local + global token：大多数位置局部看，少数枢纽全局看；
- block sparse：按块规定哪些 query/key block 相连。

共同目标是让可见边的数量少于 $`n^2`$，同时给长距离信息保留较短路径。不同 pattern 的 kernel 支持和实际速度可能差很多。

### 19.7 interleaved full attention：三层近看，一层远看

**【课程，PDF 第 65 页】**课程以 Cohere Command A 风格举例：每 4 层中 3 层 sliding-window，1 层 full attention。

```text
layer 1   local window
layer 2   local window
layer 3   local window
layer 4   full causal attention
然后重复
```

full 层让每个位置直接汇总所有过去位置；之后的 local 层虽然只看邻居，但邻居状态已经含有全局混合信息。

每 4 层 attention pair 数量级：

```math
3O(nw)+O(n^2).
```

四层全 attention 则为：

```math
4O(n^2).
```

hybrid 仍有 $`n^2`$ 项，所以大 O 不是纯线性；但 full 层比例从 100% 降到 25%，常数成本显著下降。

### 19.8 小序列的 hybrid 精确计数

仍取 $`n=8,w=3`$：

- 一层 local causal：21 pairs；
- 一层 full causal：36 pairs。

三 local 加一 full：

```math
3\times21+36=63+36=99.
```

四层 full：

```math
4\times36=144.
```

pair 数缩减倍数：

```math
144/99\approx1.455.
```

在更大的 $`n/w`$ 比例下，local 相对 full 会便宜得更多。

### 19.9 一个长序列数量级例子

**【补充例子；用非 causal 方阵数量级便于比较】**令：

```math
n=8{,}192,\qquad w=1{,}024.
```

一层 full 位置对上界：

```math
n^2=8{,}192^2=67{,}108{,}864.
```

一层 window 上界：

```math
nw=8{,}192\times1{,}024=8{,}388{,}608.
```

window/full 比：

```math
8{,}388{,}608/67{,}108{,}864=1/8.
```

三 local 加一 full：

```math
3\times8{,}388{,}608+67{,}108{,}864
=92{,}274{,}688.
```

四 full：

```math
4\times67{,}108{,}864=268{,}435{,}456.
```

缩减约：

```math
268{,}435{,}456/92{,}274{,}688\approx2.91.
```

真实 causal 边界、Flash/sparse kernel、projection 和访存会改变墙钟倍数，所以 `2.91×` 只是位置对数量的示范，不是端到端加速承诺。

### 19.10 NoPE 与 RoPE 在 hybrid 中怎样分工

**NoPE** 表示某些 attention 层不显式使用 positional embedding。课程介绍的一类设计：

- local sliding-window 层使用 RoPE，保留精细短程顺序；
- global full-attention 层使用 NoPE，避免把非常远距离强行套进同一 RoPE 相位范围。

NoPE full layer 不等于整个模型完全不知道顺序：

- causal mask 仍区分可见过去与不可见未来；
- 输入状态已经经过带 RoPE 的 local 层；
- token 内容和累积上下文不同。

但单独看一个无位置信号、无此前位置状态的 self-attention，确实会缺乏明确绝对/相对距离信息。必须按整个堆叠判断。

其他模型会在 local 与 full 层都使用 RoPE；这仍是活跃设计空间。

### 19.11 课程列出的模型趋势要带日期

**【课程/视频补充，PDF 第 64–66 页；约 [85:08](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5108s)–[88:34](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5314s)】**课程回顾：

- GPT-3 时代已有 full 与 sparse/banded pattern 交替的设计；
- 2026 讲义将 Command A 视为近期 open-model 复兴中的突出例子；
- Llama 4、Gemma 4、OLMo 3 等课程样本也混合 sliding-window 与 full attention，但位置编码选择不同；
- Qwen 3.5 的课程例子用 Gated DeltaNet 等便宜层与周期性 full attention 交替，不是普通 sliding-window。

这些是截至课程制作时对公开模型的观察。长上下文架构仍快速变化，不应外推成“未来模型都会每四层放一层 full attention”。

### 19.12 sparse/sliding 的边界总结

1. $`n^2\to nw`$ 只描述 attention 位置交互项，不会删除 FFN 和 projection；
2. local 多层能间接传远信息，但路径长度随距离增长；
3. 周期性 full 层缩短长程路径，但重新引入 $`n^2`$ 成本；
4. sparse pattern 必须有高效 kernel，否则理论少算不等于实际快；
5. KV cache 同样会随“每个 query 实际访问多少历史 K/V”改变，但实现可能仍为方便而保存完整 local cache 或环形 buffer；
6. NoPE/RoPE 是位置表示选择，full/local 是连接图选择，两条轴不能混成一个概念。

---

## 20. 把整讲串成一棵决策树

### 20.1 如果你只想先搭一个可工作的现代小模型

**【补充总结；不是唯一最佳配方】**按本讲证据，一个保守起点是：

```text
先固定 residual stream 宽度 d 与层数 L
        ↓
pre-RMSNorm，让 residual 主路保持直接
        ↓
attention：H_q * d_h ≈ d
        ↓
FFN：SwiGLU/GeGLU，d_g 从约 8d/3 起
        ↓
位置：每层 attention 对 Q/K 应用 RoPE
        ↓
线性层通常不加 bias，预训练 dropout 可从 0 起做受控实验
        ↓
训练先保证 stable softmax；规模增大时评估 z-loss/QK norm
        ↓
若部署 KV cache 是瓶颈：MHA → GQA；不急着直接压到 MQA
        ↓
若上下文很长：比较 full 与 local/full hybrid，并核对 sparse kernel
```

每个箭头都是“合理起点”，不是跳过实验的许可。

### 20.2 norm 决策树

```text
是否要保留最直接 residual path？
  ├─ 是 → pre-norm：x + F(N(x))
  │        └─ 支路输出仍有尺度尖峰？
  │             └─ 评估 non-residual post-norm：x + N(F(N(x)))
  └─ 使用既有 post-norm 架构/兼容 checkpoint？
           └─ 保持 N(x + F(x))，但更关注深层稳定和 warmup

LayerNorm 还是 RMSNorm？
  ├─ 需要完全复现已有模型 → 按原公式
  └─ 新建 decoder-only baseline → RMSNorm 常是更轻量起点
```

选择 norm 类型与 norm 位置是两件事：可以有 pre-LayerNorm，也可以有 pre-RMSNorm。

### 20.3 FFN 决策树

```text
普通 FFN 还是 gated？
  ├─ 教学/最简实现 → ReLU 或 GELU，d_ff≈4d
  └─ 现代质量 baseline → SwiGLU/GeGLU
          ├─ 要等普通 4d FFN 的矩阵参数 → d_g≈8d/3
          └─ 有额外参数预算 → 在受控实验里加宽，不把 8/3 当上限
```

若改 `d_ff`，至少重算：每层 FFN 参数、activation shape、FLOPs、模型并行切分。

### 20.4 attention 部署决策树

```text
当前瓶颈在哪个阶段？
  ├─ prefill/训练 compute-bound
  │      └─ 优先看大矩阵效率、FlashAttention、batch/token 并行
  └─ incremental decode memory-bound
         ├─ batch 太小 → 连续批处理/调度是否可改善？
         ├─ KV cache 太大 → MHA 改 GQA，计算 H_q/H_kv 缩减倍数
         └─ 长历史读取太贵 → local/full hybrid 或 cache 量化
```

不能只说“GQA 更快”；要说明是减少 K/V heads、cache bytes 和 decode 数据移动。

### 20.5 长上下文决策树

```text
必须让每层每个 token 直接看全部过去吗？
  ├─ 是 → full causal attention，接受 O(n²) 位置交互
  └─ 否 → sliding window，选 w
           ├─ 局部层数足够让信息逐层传递？
           ├─ 是否周期性插入 full layer 缩短长程路径？
           ├─ local/full 各用 RoPE 还是部分 NoPE？
           └─ 框架是否真的有 sparse kernel，而非算完再 mask？
```

### 20.6 稳定性排查顺序

```text
先确认基础实现
  1. softmax 是否减 max / 使用 logsumexp？
  2. dtype、loss scaling、mask 是否正确？
  3. 学习率和 warmup 是否合理？
        ↓
定位尖峰来自 output logits 还是 attention logits
  ├─ output normalizer 漂移 → 评估 z-loss
  ├─ Q/K 尺度膨胀 → 评估 QK norm
  └─ 必须强限制 logits → 评估 soft-cap，并测质量损失
```

不要用强干预掩盖 shape、mask 或 optimizer 的基础 bug。

### 20.7 三条复习路线

**15 分钟复习：**

1. §0 因果链；
2. §5.7 三种 norm 公式；
3. §8.6–8.8 `2/3` 与 `8/3`；
4. §11.5 RoPE 矩阵恒等式；
5. §16.15 稳定手段对比；
6. §18.4 cache 缩减公式；
7. §19.12 sliding-window 边界。

**45 分钟手算复习：**

1. §3 全部 shape；
2. §6 LN/RMSNorm；
3. §7–8 FFN/gate；
4. §11 RoPE 数值验证；
5. §13 三组参数量；
6. §16 softmax/z-loss/QK norm；
7. §18 两组 cache 与算术强度；
8. §19 window pair 数。

**实现前复习：**§3、§5、§8、§11.9–11.11、§13.2、§16.15、§17.5–17.9、§18.1–18.4、§19.2。

---

## 21. 常见误区：看到这些句子就停下来检查

### 21.1 norm 与 residual

1. **误区：LayerNorm 在整个 batch 上求一个均值。**  
   纠正：Transformer 的 LayerNorm 通常对每个 token 的最后一轴 $`d`$ 单独求统计量。

2. **误区：RMSNorm 与 LayerNorm 完全等价。**  
   纠正：RMSNorm 不减均值；`[1,3]` 分别得到约 `[0.447,1.342]` 与 `[-1,1]`。

3. **误区：RMSNorm 没有任何可学习参数。**  
   纠正：常见 RMSNorm 仍有逐维 scale $`\gamma`$，只是通常没有 $`\beta`$。

4. **误区：pre-norm 表示 block 输出没有 norm。**  
   纠正：norm 在子层输入支路；模型末尾也常有 final norm。

5. **误区：non-residual post-norm 没有 residual connection。**  
   纠正：它是 $`x+N(F(N(x)))`$；没有被 norm 包住的是 residual sum。

6. **误区：用了 pre-norm 就不需要 warmup。**  
   纠正：pre-norm 改善 residual/gradient 路径，但现代训练仍常使用 warmup。

7. **误区：norm FLOPs 少，所以时间一定可忽略。**  
   纠正：扫描、统计、写回 activation 可能受内存带宽限制；FLOPs 不是 runtime。

8. **误区：删除 bias 会改变 hidden shape。**  
   纠正：它减少参数和加法，输出 shape 不变。

### 21.2 激活、GLU 与 block 结构

9. **误区：ReLU、GELU、SwiGLU 都只换一个逐元素函数。**  
   纠正：SwiGLU 是两条投影支路相乘的 gated FFN，结构和矩阵数都变了。

10. **误区：GLU 中只有 gate，没有内容支路。**  
    纠正：输出是内容 $`u`$ 逐元素乘门 $`\phi(g)`$，再经 down projection。

11. **误区：gated FFN 与普通 FFN 同宽时参数相同。**  
    纠正：同宽时主矩阵从两个变三个，是 1.5 倍。

12. **误区：`8/3` 是 SwiGLU 的数学定义。**  
    纠正：它来自把普通 `4d` 两矩阵 FFN 与三矩阵 gated FFN做等参数匹配。

13. **误区：所有 gated 模型都必须精确使用 `8/3`。**  
    纠正：真实模型会因质量、预算和硬件圆整取 2.5、2.67、3.5、4 等不同值。

14. **误区：parallel block 把 attention 和 FFN 输出拼成 `[B,T,2d]`。**  
    纠正：常见 parallel block 把两个 `[B,T,d]` 支路与 residual 逐元素相加。

15. **误区：PaLM 的约 15% 意味着 parallel 在所有系统都快 15%。**  
    纠正：这是特定规模和实现的报告，依赖融合、调度和硬件。

### 21.3 位置与 RoPE

16. **误区：token ID 的大小本身编码顺序。**  
    纠正：ID 只是查词表的编号；序列位置是另一条信息。

17. **误区：RoPE 给输入 embedding 加一条位置向量。**  
    纠正：标准 RoPE 在每层 attention 内旋转投影后的 Q/K。

18. **误区：RoPE 旋转后只剩距离，内容消失。**  
    纠正：点积仍是 $`q_i^TR(j-i)k_j=q_i^TR(-\delta)k_j`$，同时依赖 q、k 内容；课件定义 $`\delta=i-j`$。

19. **误区：`R(i)^TR(j)=R(i-j)` 与符号方向无关。**  
    纠正：课件 $`\delta=i-j`$；按本讲列向量、query-left/key-right 约定得到 $`R(j-i)=R(-\delta)`$。交换 query/key 会改变符号。

20. **误区：二维 RoPE 要求整个 head 只有 2 维。**  
    纠正：高维 head 被拆成许多二维坐标对，每对频率不同。

21. **误区：标准 RoPE 必须旋转 V。**  
    纠正：标准做法旋转决定 score 的 Q/K，V 通常不旋转。

22. **误区：NoPE 层等于整个模型完全不知道顺序。**  
    纠正：causal mask 和此前带位置信号层的 hidden states 仍携带结构；要看整个堆叠。

### 21.4 超参数与正则化

23. **误区：`d_ff=4d` 是经过证明的最优定理。**  
    纠正：它是强经验基线；课程图只在特定 50M 参数实验中显示宽低谷。

24. **误区：`H*d_h` 必须等于 `d_model`，否则无法 residual。**  
    纠正：output projection 可以把 $`Hd_h`$ 投影回 $`d`$。

25. **误区：aspect ratio 指 $`d_{ff}/d_{model}`$。**  
    纠正：本讲 aspect ratio 指 $`d_{model}/L`$；两个比例不要混。

26. **误区：参数量相近的深窄与宽浅模型，latency 必然相同。**  
    纠正：深度增加串行依赖，宽度改变 tensor parallel 和通信方式。

27. **误区：大词表一定让模型更好。**  
    纠正：它可能缩短多语种文本，却增加 embedding、logits 和 softmax 成本。

28. **误区：dropout 永久删掉神经元。**  
    纠正：训练时随机丢 activation；推理时通常关闭。

29. **误区：weight decay 只是防过拟合。**  
    纠正：大模型预训练中它还通过 $`\eta_t\lambda`$ 改变优化与权重尺度动态。

30. **误区：论文没写 dropout 就能断定为 0。**  
    纠正：开源配置可查，closed model 的未报告不能补成事实。

### 21.5 稳定性、cache 与稀疏注意力

31. **误区：减去 max 改变 softmax 概率。**  
    纠正：共同指数因子在分子分母约掉；这是数学等价的稳定实现。

32. **误区：z-loss、QK norm、soft-cap 是同一种正则。**  
    纠正：z-loss 改训练目标；QK norm 和 soft-cap 改 forward。

33. **误区：soft-cap 只在发生 overflow 时改变 logits。**  
    纠正：tanh 对有限非零输入也会压缩，只是小值变化较小。

34. **误区：QK norm 保证所有 attention logits 绝对小于 1。**  
    纠正：可学习 scale、$`\varepsilon`$ 和向量夹角都影响结果；它控制尺度，不是简单硬界。

35. **误区：KV cache 保存历史 Q、K、V 和 score 矩阵。**  
    纠正：标准 cache 保存历史 K/V；新 Q 每步计算，score 每步产生。

36. **误区：MQA 的 one 表示 `head_dim=1`。**  
    纠正：表示 $`H_{kv}=1`$；每个 K/V head 仍可有例如 128 个 features。

37. **误区：GQA 减少 query head 数。**  
    纠正：它可保持 $`H_q`$，只减少 $`H_{kv}`$ 并让 queries 分组共享 K/V。

38. **误区：cache 缩减倍数还要乘 sequence length。**  
    纠正：同一 $`b,n,k,dtype`$ 比较时都约掉，倍数为 $`H_q/H_{kv}`$。

39. **误区：prefill 与 decode 只是 batch shape 不同，算术强度一样。**  
    纠正：prefill 能跨 token 复用权重；decode 逐步读权重和越来越长的 cache。

40. **误区：FlashAttention 把 attention 的 $`n^2`$ 算术完全删除。**  
    纠正：它主要减少 HBM 中间读写；full attention 的位置点积仍是二次数量级。

41. **误区：给 full attention 加一个 window mask 就自动得到 $`O(nw)`$。**  
    纠正：若先算完整 score 再 mask，仍付出 $`n^2`$；需要真正 sparse kernel。

42. **误区：一层 $`w=3`$ 的 local attention 能直接看 3 层以外的信息。**  
    纠正：一层只看当前与两个过去位置；更远信息要逐层接力。

43. **误区：每四层一层 full 的 hybrid 是 $`O(nw)`$。**  
    纠正：周期仍有一个 $`O(n^2)`$ 项，只是 full 层比例下降。

44. **误区：课程模型清单是未来模型的架构规则。**  
    纠正：它是 2026 讲义对跨年代公开报告的快照，需保留年份、数据和系统边界。

---

## 22. 术语表

| 术语 | 全称/中文 | 一句话解释 |
|---|---|---|
| activation | 激活/中间结果 | forward 中产生、常供 backward 使用的 tensor；也可指非线性函数的输出 |
| arithmetic intensity | 算术强度 | 计算量除以从指定内存层级搬的数据量 |
| aspect ratio | 架构深宽比 | 本讲特指 $`d_{model}/L`$ |
| attention head | 注意力头 | 使用自己 query 子空间的一组 attention 计算 |
| attention logit | 注意力未归一化分数 | Q·K 缩放后、softmax 前的数 |
| autoregressive | 自回归 | 下一个 token 依赖此前已知/已生成 token |
| bias | 偏置 | 线性变换后逐输出维度相加的可学习量 |
| causal mask | 因果遮罩 | 阻止位置看到未来 token |
| checkpoint | 检查点 | 保存的模型/优化器状态；不要与 activation checkpointing 混淆 |
| compute-bound | 计算受限 | 运行主要受算力而非带宽限制 |
| decoder-only | 仅解码器式 | 用 causal self-attention 逐 token 建模的 Transformer |
| decode | 解码/生成阶段 | 自回归地一个接一个产生新 token |
| dropout | 随机失活 | 训练时随机丢 activation 并作期望缩放 |
| embedding | 嵌入 | token ID 查得的可学习向量 |
| FFN | Feed-Forward Network，前馈网络 | 每个 token 位置独立使用的特征加工网络 |
| FLOP | 浮点操作 | 一次浮点加、乘等工作的计数单位 |
| full attention | 全注意力 | causal 情况下每个位置可看所有过去位置 |
| GELU | Gaussian Error Linear Unit | $`x\Phi(x)`$ 的平滑非线性 |
| GeGLU | GELU-gated Linear Unit | 用 GELU 作门的 gated FFN |
| GLU | Gated Linear Unit，门控线性单元 | 内容支路逐元素乘一条门支路 |
| GQA | Grouped-Query Attention | 多个 query heads 分组共享较少的 KV heads |
| head dimension | 头维度 | 每个 head 内的 feature 数，记为 $`d_h`$ 或 $`k`$ |
| hidden state | 隐藏状态 | 某层对某 token 的当前向量表示 |
| HBM | High Bandwidth Memory | GPU 上容量较大、相对片上存储更远的高带宽内存 |
| KV cache | 键值缓存 | decode 时按层保存历史 K/V，避免重复投影 |
| LayerNorm | 层归一化 | 对 token 特征减均值、除标准差，再缩放平移 |
| logit | 未归一化分数 | softmax 之前的实数分数 |
| log-sum-exp | 对数指数和 | $`\log\sum_j e^{z_j}`$，稳定 softmax/交叉熵中的核心量 |
| MHA | Multi-Head Attention | 每个 query head 有对应 K/V head 的多头注意力 |
| MLP | Multi-Layer Perceptron | 在本讲常与 FFN 近义 |
| MLA | Multi-head Latent Attention | 通过潜变量/低秩表示压缩 KV 信息的注意力变体 |
| MQA | Multi-Query Attention | 所有 query heads 共用一个 KV head |
| NoPE | No Positional Embedding | 某些层不加入显式位置嵌入 |
| non-residual post-norm | 非残差后归一化 | 在子层出口 norm，但 residual sum 本身不被包住 |
| parameter | 参数 | 训练中学习并跨 step 保存的权重 |
| post-norm | 后归一化 | 常指 $`N(x+F(x))`$ |
| prefill | 预填充 | 并行处理整段 prompt 并建立 KV cache |
| pre-norm | 前归一化 | 常指 $`x+F(N(x))`$ |
| query/key/value | 查询/键/值 | Q/K 决定注意力权重，权重再汇总 V |
| receptive field | 感受野 | 某层某位置最终可能依赖的原输入位置集合 |
| regularization | 正则化 | 改变训练约束、噪声或权重轨迹的一类方法 |
| residual connection | 残差连接 | 把子层输出作为增量加回输入 |
| residual stream | 残差流 | 各层通过加法持续更新的主 hidden state |
| RMSNorm | Root Mean Square Norm | 不减均值，按均方根缩放的 norm |
| RoPE | Rotary Position Embedding | 按位置旋转每层 attention 的 Q/K 坐标对 |
| serial block | 串联 block | FFN 读取本层 attention 更新后的状态 |
| parallel block | 并联 block | attention 与 FFN 读取同一个旧状态并相加 |
| sigmoid | S 形函数 | $`1/(1+e^{-x})`$，输出在 0 与 1 之间 |
| sinusoidal position | 正弦位置编码 | 用固定多频率 sin/cos 向量表示绝对位置 |
| sliding-window attention | 滑动窗口注意力 | 每个 query 只看附近固定数量的 keys |
| soft-capping | 软上限 | 用 $`c\tanh(z/c)`$ 平滑限制 logits 范围 |
| softmax | 软最大归一化 | 把 logits 转为总和为 1 的概率 |
| sparse attention | 稀疏注意力 | 只计算预定位置连接子集的 attention |
| Swish/SiLU | Sigmoid Linear Unit | $`x\sigma(x)`$ 的平滑激活 |
| SwiGLU | Swish-gated Linear Unit | 用 Swish/SiLU 作门的 gated FFN |
| tensor parallel | 张量并行 | 把同一层的大矩阵/张量切到多个设备并行 |
| token ID | 词元编号 | 用来查 embedding 的离散整数，不带大小语义 |
| weight decay | 权重衰减 | 更新时把 parameter 向 0 缩的机制 |
| weight tying | 权重绑定 | 输入 embedding 与输出投影共享参数 |
| z-loss | Z 正则损失 | 用 $`\alpha(\log Z)^2`$ 抑制 output normalizer 漂移 |

---

## 23. 自测题：先合上答案再做

### 23.1 基础 shape 与数据流（1–10）

1. 用自己的话区分 token ID、embedding、hidden state 和 logit。
2. 若 $`B=2,T=5,d=8,V=100`$，token IDs、每层 hidden states、最终 logits 的 shape 各是什么？每个 logits 元素代表什么？
3. 承接第 2 题，若 $`H=4,d_h=2`$，split 后 Q 的 shape 和 attention score 的 shape 是什么？元素数是否守恒？
4. 长度 5 的 causal attention 中，query 位置 3 可看哪些 key 位置？共有多少个？
5. attention 与 FFN 分别主要混合哪一种信息？为什么 FFN 的 $`T`$ 轴不变？
6. 为什么 residual 加法两边必须同 shape？若 attention 合并后宽度为 16、residual 宽为 8，要先做什么？
7. 分别写出 post-norm 与 pre-norm 的一个子层公式。
8. 写出 non-residual post-norm 的一个常见公式，并解释名字中的 “non-residual”。
9. 两层 pre-norm 中，为什么可以说 $`x_0`$ 有一条直通路径到 $`x_2`$？
10. 若 $`x=[1,3]`$，某 pre-norm 子层支路给出 `[0.2,-0.1]`，残差输出是多少？

### 23.2 norm、激活与 gated FFN（11–20）

11. 取 $`x=[1,3]`$、$`\gamma=[1,1]`$、$`\beta=[0,0]`$、$`\varepsilon=0`$，完整算 LayerNorm。
12. 对同一 $`x=[1,3]`$ 完整算 RMSNorm，并解释为何第一个输出仍为正。
13. 对 $`x=[0,2,4,6]`$ 算均值、方差和无 affine 的 LayerNorm 输出。
14. norm 公式中的 $`\varepsilon`$ 解决什么问题？它是可学习参数吗？
15. 若 $`xW=[2,-1]`$、bias `[0.5,0.25]`，输出是什么？删除 bias 会不会改变 shape？
16. 分别给出 ReLU、GELU、Swish 在输入 -1 上的近似输出。
17. 普通无 bias FFN 取 $`d=6,d_{ff}=12`$，主权重参数是多少？逐矩阵算。
18. gated 中 $`u=[2,-1]`$、$`g=[-1,2]`$。用 ReLU 门时，逐元素乘的结果是什么？
19. gated FFN 取 $`d=6,d_g=8`$，三个主矩阵参数是多少？它与第 17 题是否等预算？
20. 从普通 FFN 与 gated FFN 参数式推出 `2/3`；若普通基准 $`d_{ff}=4d`$ 且 $`d=3072`$，gated 宽度是多少？

### 23.3 block、位置与超参数（21–30）

21. serial 与 parallel block 的 FFN 分别读什么输入？为什么这会改变语义依赖？
22. 比较 sinusoidal absolute 与 learned absolute position：位置信号放哪里、是否增加参数、长度外推有什么差别？
23. relative bias 中，$`i=5,j=3`$，内容 score 为 1.2，$`b_{i-j}=-0.1`$，最终 score 是多少？
24. 课件定义 $`\delta=i-j`$。从 $`(R(i)q)^T(R(j)k)`$ 推出为什么中间是 $`R(j-i)=R(-\delta)`$；写出两条差角公式和转置后的矩阵链，并解释符号为何不能写反。
25. 取 $`q=[1,0],k=[0,1],i=1,j=2,\omega=90^\circ`$，用“分别旋转”与“相对旋转”两条路线算点积。
26. 标准 RoPE 旋转 Q/K/V 中哪些张量？在什么位置、多少层应用？为什么？
27. T5 11B 的 $`d_{ff}=65,536,d=1,024`$，比例是多少？这能否证明 64 倍普遍最优？
28. 某 attention 有 $`H=128,d_h=128,d=1,024`$，计算 $`Hd_h/d`$。这表示 `head_dim=16d` 吗？
29. 两个模型分别为 A：$`d=768,L=12`$，B：$`d=512,L=24`$。各自 aspect ratio 是多少？哪个相对更深？
30. §13.8 的合成配置每层 attention 为 2,359,296 参数、FFN 为 4,718,592 参数、12 层、共享 embedding 38,400,000。忽略 norm/bias，总参数是多少？

### 23.4 词表、正则化与稳定性（31–40）

31. $`V=32,000,d=4,096`$，共享 embedding 有多少参数？BF16 是多少 MiB？若输入输出不共享呢？
32. inverted dropout 中 $`x=[2,4,-6],p=0.5,m=[1,0,1]`$，本次输出是什么？为什么长期期望不变？
33. 用 $`w=[2,-1],g=[0.3,-0.4],\eta=0.1,\lambda=0.01`$ 算一次 decoupled weight-decay 更新。
34. 完整算 logits `[1,2,3]` 的 softmax，结果保留四位小数。
35. 为什么 `[1000,1001,1002]` 与 `[1,2,3]` softmax 相同？写出稳定计算的移位 logits。
36. 取 $`\alpha=0.01`$，分别算 `[0,0]` 与 `[10,10]` 的 z-loss。二者概率有什么关系？
37. 对 $`q=[3,4],k=[6,8],d_h=2`$，算 raw scaled logit，再用无 affine RMSNorm 做 QK norm 后重算。
38. 对 logits `[-10,0,10]` 使用 cap $`c=2`$，求 capped logits 与近似 softmax 概率。
39. z-loss、QK norm、soft-cap 中，哪些只在训练 objective 中出现？哪些会改变 forward 并须在推理执行？
40. 为什么 stable softmax 的减 max 与 soft-cap 不能混为一谈？

### 23.5 cache、算术强度与长上下文（41–50）

41. append 后总 cache 长度 $`n=4`$；取 $`b=1,H_{kv}=4,k=2`$，BF16 单层 KV cache 多少元素、多少 bytes？append 前长度是多少？
42. append 后总 cache 长度 $`n=2048`$；取 $`b=1,H_q=32,k=128`$、BF16：MHA、$`H_{kv}=8`$ GQA、MQA 的单层 cache 分别多少 MiB？缩减倍数呢？
43. 为什么 MQA 的 $`H_{kv}=1`$ 不表示 $`k=1`$？若 $`H_q=8,H_{kv}=2`$，每个 KV head 服务几个 query heads？
44. 令 $`n`$ 为 decode 结束时 append 后的总长度。从 $`F=O(bnd^2)`$ 和 $`Q=O(bn^2d+nd^2)`$ 推导 decode 算术强度。
45. MQA 把 cache 访问项从 MHA 的数量级 $`bn^2d`$ 改为 $`bn^2k`$。用 $`d=hk`$ 推出强度分母中的 cache 项为何从 $`n/d`$ 变成 $`n/(dh)`$。
46. 定义窗口宽 $`w`$ 包含当前 token。对 $`n=8,w=3`$，列出 query 位置 5、7 的可见 keys，并算全序列 local pairs。
47. 长度 8 的 full causal attention 有多少 pairs？为什么复杂度仍写 $`O(n^2)`$？
48. 三层 local（每层 21 pairs）加一层 full（36 pairs），总 pairs 是多少？与四层 full 相比缩减多少倍？
49. $`w=3`$ 时，位置 7 的信息要通过多少层纯 local attention 才可能间接到达原始位置 0？写出每层最大向后距离。
50. 综合题：你要部署一个长上下文模型，incremental decode 的 KV cache/带宽是主要瓶颈，但不能接受 MQA 的明显质量损失。基于本讲给出一组架构方向，并逐项说明收益、代价和必须验证的条件。

---

## 24. 自测完整答案

### 24.1 基础 shape 与数据流（1–10）

1. **token ID** 是查表编号；**embedding** 是该编号查得的初始向量；**hidden state** 是经过若干层后某 token 的中间表示；**logit** 是模型对某个候选输出 token 的 softmax 前分数。ID 没有连续数值语义，另外三者是浮点 tensor。

2. token IDs：`[B,T]=[2,5]`；hidden states：`[B,T,d]=[2,5,8]`；logits：`[B,T,V]=[2,5,100]`。`logits[b,t,v]` 是第 b 条序列位置 t 对词表候选 v 的未归一化分数。

3. Q split 后：`[B,H,T,d_h]=[2,4,5,2]`；score：`[B,H,T,T]=[2,4,5,5]`。split 前 Q 元素为 $`2\times5\times8=80`$；split 后 $`2\times4\times5\times2=80`$，守恒。

4. 位置 3 可看 `[0,1,2,3]`，共 4 个；位置 4 是未来，必须被 causal mask 遮住。

5. attention 主要混合 **不同 token 位置** 的信息；FFN 在每个 token 内混合 **特征维**。同一 FFN 权重独立用于各位置，因此 `[B,T,d]→[B,T,d_ff]→[B,T,d]`，$`T`$ 不变。

6. residual 是逐元素加法，对应元素必须一一存在。宽 16 的 attention 输出不能直接加宽 8 的 residual；需用 $`W_O`$ 将 `[B,T,16]` 投影为 `[B,T,8]`。

7. post-norm：$`y=N(x+F(x))`$。pre-norm：$`y=x+F(N(x))`$。

8. 常见式：$`y=x+N_{out}(F(N_{in}(x)))`$。“non-residual”修饰 post-norm 的放置：$`N_{out}`$ 不包住 residual sum，残差连接仍存在。

9. 两层为 $`x_1=x_0+F_0(N(x_0))`$、$`x_2=x_1+F_1(N(x_1))`$。代入：$`x_2=x_0+F_0(N(x_0))+F_1(N(x_1))`$，所以 $`x_0`$ 以系数 1 通过加法主路出现。

10. 逐元素加：

    $`[1,3]+[0.2,-0.1]=[1.2,2.9].`$

### 24.2 norm、激活与 gated FFN（11–20）

11. 均值 $`\mu=(1+3)/2=2`$；离均差 `[-1,1]`；方差 $`[(-1)^2+1^2]/2=1`$；标准差 1。乘 $`\gamma`$ 加 $`\beta`$ 后仍为 `[-1,1]`。

12. 平方平均 $`(1^2+3^2)/2=5`$；RMS $`=\sqrt5\approx2.236`$；输出：

    $`[1/\sqrt5,3/\sqrt5]\approx[0.447,1.342].`$

    RMSNorm 不减均值，所以正输入 1 不会因低于均值而变负。

13. $`\mu=(0+2+4+6)/4=3`$；离均差 `[-3,-1,1,3]`；方差 $`(9+1+1+9)/4=5`$；输出：

    $`[-3,-1,1,3]/\sqrt5 \approx[-1.342,-0.447,0.447,1.342].`$

14. 若所有输入相同，方差/RMS 分母可能为 0；$`\varepsilon`$ 保证分母为正并改善数值稳定。它是固定超参数，不是学习的 parameter。

15. $`[2,-1]+[0.5,0.25]=[2.5,-0.75]`$。删除 bias 只少参数和加法，输出仍是 `[2]`。

16. 输入 -1：ReLU 为 0；GELU $`\approx-0.1587`$；Swish $`=-1\times\sigma(-1)\approx-0.2689`$。

17. $`W_{up}`$：$`6\times12=72`$；$`W_{down}`$：$`12\times6=72`$；总计 $`72+72=144`$。

18. $`\mathrm{ReLU}(g)=[0,2]`$；逐元素乘：

    $`[2,-1]\odot[0,2]=[0,-2].`$

19. 三个矩阵分别 $`6\times8=48`$，总数 $`3\times48=144`$，与第 17 题等预算。

20. 等预算：

    $`3d d_g=2d d_{ff,plain} \Rightarrow d_g=\frac23d_{ff,plain}.`$

    若 $`d_{ff,plain}=4d`$：

    $`d_g=\frac23\times4d=\frac83d.`$

    $`d=3072`$ 时，$`3072/3=1024`$，再乘 8 得 $`8192`$。

### 24.3 block、位置与超参数（21–30）

21. serial：$`u=x+A(N_1(x))`$，FFN 读 $`N_2(u)`$，能处理本层 attention 新写入的信息。parallel：attention 与 FFN 都读 $`N(x)`$，本层 FFN 看不到本层 attention 输出；它要等下一层。两者因此不是只改调度。

22. sinusoidal absolute 用固定 sin/cos 向量在输入处相加，不增参数，公式能为新位置生成值；learned absolute 用 `[T_max,d]` 参数表相加，表达自由，但超过表长没有天然条目。二者都是绝对位置注入。

23. $`i-j=2`$，所以 $`s_{5,3}=1.2+b_2=1.2-0.1=1.1`$。

24. 先写课件与本文约定：$`\delta=i-j`$，所以 $`j-i=-\delta`$。两条差角公式是：

    $`\cos(j-i)=\cos j\cos i+\sin j\sin i,`$

    $`\sin(j-i)=\sin j\cos i-\cos j\sin i.`$

    转置后的点积链：

    $`(R(i)q)^T(R(j)k) =q^TR(i)^TR(j)k.`$

    旋转矩阵正交，$`R(i)^T=R(-i)`$，旋转相乘角度相加：

    $`R(-i)R(j)=R(j-i).`$

    四格中右上角为：

    $`-\cos i\sin j+\sin i\cos j =-\left(\sin j\cos i-\cos j\sin i\right) =-\sin(j-i),`$

    与 $`R(j-i)`$ 右上角一致。因此结果为：

    $`q^TR(j-i)k=q^TR(-\delta)k.`$

    $`R(j-i)`$ 与 $`R(i-j)`$ 旋转方向相反；两者都只依赖相对差，但在固定的 query/key 点积方向下不能互换。

25. 分别旋转：$`q'=R(90^\circ)[1,0]=[0,1]`$；$`k'=R(180^\circ)[0,1]=[0,-1]`$；点积 $`-1`$。课件 $`\delta=i-j=-1`$，所以 $`j-i=-\delta=1`$。相对路线用 $`R(90^\circ)k=[-1,0]`$；与原 q 点积也是 $`-1`$。

26. 标准 RoPE 旋转 Q 和 K，不旋转 V；位置在每层 attention 的 Q/K projection 之后、Q·K 之前。因为位置要进入决定 attention 权重的 Q·K；每层会新投影 Q/K，所以每个使用 RoPE 的 attention 层都要做。

27. $`65{,}536/1{,}024=64.`$

    它只证明该配置可成功训练；后续 T5 v1.1 回到约 2.5，课程图也显示宽容区，不证明 64 普遍最优。

28. $`Hd_h/d=128\times128/1{,}024=16{,}384/1{,}024=16.`$

    它表示所有 heads 合并的 attention 内宽为 model width 的 16 倍；单个 `head_dim` 仍是 128，不是 $`16d`$。

29. A：$`768/12=64`$。B：$`512/24\approx21.33`$。B 的 $`d/L`$ 更小，所以相对更深窄。

30. 每 block：

    $`2{,}359{,}296+4{,}718{,}592=7{,}077{,}888.`$

    12 层：

    $`7{,}077{,}888\times12=84{,}934{,}656.`$

    加共享 embedding：

    $`84{,}934{,}656+38{,}400{,}000=123{,}334{,}656.`$

### 24.4 词表、正则化与稳定性（31–40）

31. 参数：

    $`32{,}000\times4{,}096=131{,}072{,}000.`$

    BF16 bytes：$`131{,}072{,}000\times2=262{,}144{,}000`$，除 $`2^{20}`$ 得 250 MiB。若不共享输入/输出，参数和 bytes 都翻倍：262,144,000 参数、500 MiB。

32. 保留缩放 $`1/(1-p)=2`$：

    $`[2,4,-6]\odot[1,0,1]\times2=[4,0,-12].`$

    因 $`E[m_i]=1-p`$，$`E[m_i x_i/(1-p)]=x_i`$。

33. 衰减因子：$`1-0.1\times0.01=0.999`$；缩后 `[1.998,-0.999]`；$`\eta g=[0.03,-0.04]`$；相减得：

    $`[1.998,-0.999]-[0.03,-0.04]=[1.968,-0.959].`$

34. 指数 `[2.7183,7.3891,20.0855]`，和 30.1929；概率约：

    $`[0.0900,0.2447,0.6652].`$

35. 两组只差共同常数 999；softmax 对共同平移不变。稳定计算对 `[1000,1001,1002]` 减 1002，得到 `[-2,-1,0]`，概率仍 `[0.0900,0.2447,0.6652]`。

36. `[0,0]`：$`\log Z=\log2\approx0.6931`$，

    $`L_z=0.01\times0.6931^2\approx0.004805.`$

    `[10,10]`：$`\log Z'=10+\log2\approx10.6931`$，

    $`L_z'=0.01\times10.6931^2\approx1.14342.`$

    两组 softmax 都是 `[0.5,0.5]`，z-loss 惩罚共同漂移。

37. raw 点积 $`3\times6+4\times8=50`$；scaled 为 $`50/\sqrt2\approx35.355`$。q 的 RMS $`\sqrt{12.5}\approx3.5355`$，k 的 RMS $`\sqrt{50}\approx7.0711`$，归一化后都约 `[0.8485,1.1314]`；点积约 2，再除 $`\sqrt2`$ 得约 1.4142。

38. $`z/c=[-5,0,5]`$；tanh 约 `[-0.99991,0,0.99991]`；乘 2 得 `[-1.9998,0,1.9998]`。指数约 `[0.1354,1,7.3876]`，和 8.5230，概率约 `[0.0159,0.1173,0.8668]`。

39. z-loss 只加到训练 objective，推理不计算该项。QK norm 和 soft-cap 改 forward；若模型采用，训练、推理都必须执行。

40. 减 max 是：

    $`\mathrm{softmax}(z-c)=\mathrm{softmax}(z),`$

    完全不改数学概率。soft-cap 用非线性 $`c\tanh(z/c)`$，不同 logits 被不同程度压缩，会改变概率和梯度。

### 24.5 cache、算术强度与长上下文（41–50）

41. append 后有 $`n=4`$ 个 token，append 前有 $`n-1=3`$ 个。append 后元素：

    $`2\times1\times4\times4\times2=64.`$

    BF16 每元素 2 bytes，所以 $`64\times2=128`$ bytes/layer。

42. $`n=2048`$ 是 append 后长度。统一公式 $`2bnH_{kv}k\times2`$ bytes。

    - MHA $`H_{kv}=32`$：32 MiB/layer；
    - GQA $`H_{kv}=8`$：8 MiB/layer，为 4 倍缩减；
    - MQA $`H_{kv}=1`$：1 MiB/layer，为 32 倍缩减。

    具体元素分别为 16,777,216、4,194,304、524,288。

43. $`H_{kv}`$ 数“有几组 K/V heads”，$`k`$ 数“一组 head 内有几个特征”，是不同轴。MQA 可有 `[b,1,n,128]`，所以 $`H_{kv}=1,k=128`$。若 $`H_q=8,H_{kv}=2`$：

    $`g=H_q/H_{kv}=8/2=4,`$

    每个 KV head 服务 4 个 query heads。

44. 将从长度 1 到最终 append 后长度 $`n`$ 的 decode 工作/访存合并，按课程近似：

    $`I=\frac{F}{Q} =O\left(\frac{bnd^2}{bn^2d+nd^2}\right).`$

    分子分母同除 $`bnd^2`$：第一项变 $`n/d`$，第二项变 $`1/b`$，所以：

    $`I=O\left(\left[\frac nd+\frac1b\right]^{-1}\right).`$

45. MQA cache 项除工作量：

    $`\frac{bn^2k}{bnd^2}=\frac{nk}{d^2}.`$

    因 $`d=hk`$，即 $`k=d/h`$：

    $`\frac{n(d/h)}{d^2}=\frac{n}{dh}.`$

    因此相较 MHA 的 $`n/d`$ 理想缩小 h 倍。

46. $`w=3`$ 含自己。位置 5 看 `[3,4,5]`；位置 7 看 `[5,6,7]`。全序列计数：

    $`1+2+3+3+3+3+3+3=21.`$

47. full causal pairs：

    $`1+2+\cdots+8=8\times9/2=36.`$

    一般为 $`n(n+1)/2`$，最高次项是 $`n^2/2`$，忽略常数仍为 $`O(n^2)`$。

48. hybrid：

    $`3\times21+36=99.`$

    四层 full：$`4\times36=144`$。缩减倍数：

    $`144/99\approx1.455.`$

49. 每层最多向过去扩 $`w-1=2`$。位置 7 到位置 0 距离 7，需要：

    $`\lceil7/2\rceil=4`$

    层。最大覆盖依次约 2、4、6、8 个位置距离；具体集合从 `[5,7]`、`[3,7]`、`[1,7]` 到 `[0,7]`。

50. 一组合理方向：

    - 用 **GQA** 保持 $`H_q`$，选中间 $`H_{kv}`$，cache 理想缩减 $`H_q/H_{kv}`$；代价是 queries 共享 K/V，必须重新训练并验证质量；
    - 用 **sliding-window + 周期性 full attention**，local 层把每步历史读取从约 n 降到 w，full 层维持短的长程路径；代价是周期仍含 $`n^2`$，需要真实 sparse kernel；
    - 评估 **KV cache 低精度/分页**，进一步减少 bytes 和碎片；代价是量化误差、scale 元数据和 kernel 支持；
    - 保留或评估 **QK norm** 以稳定长上下文 attention logits；它不是 cache 优化，需单独测质量/吞吐；
    - 选择 RoPE/NoPE 组合时验证训练长度外推与检索任务，不能只看短 context perplexity；
    - 最后用目标 batch、序列长度、硬件 profiler 验证，因为课程大 O 忽略 dtype、Flash/PagedAttention、padding 和通信。

---

## 25. 视频时间导航

> 主字幕：YouTube 人工 `English (United States)`；2020 个片段；最后片段从 89:08 开始，约 89:09 结束。以下链接都使用 `&t=秒数s`，点击可直接跳转。

### 25.1 开场、norm 与 bias

| 时间 | 内容 |
|---|---|
| [00:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5s) | 本讲是跨模型架构调查；实现、小实验与模型比较的关系 |
| [02:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=123s) | 原始 Transformer 的 sinusoid/ReLU/post-norm 与作业现代 variant |
| [05:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=300s) | 本讲路线：架构变化、超参数与稳定技巧 |
| [06:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=381s) | 历史趋势：LLaMA-like 共识、稳定性改动与长上下文改动 |
| [07:31](https://www.youtube.com/watch?v=lVynu4bo1rY&t=451s) | normalization 主题开始 |
| [08:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=480s) | residual stream、pre/post-norm 基本图 |
| [09:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=580s) | pre-norm、深层稳定与 warmup 的边界 |
| [11:08](https://www.youtube.com/watch?v=lVynu4bo1rY&t=668s) | “保持 residual stream 干净”的直觉 |
| [12:46](https://www.youtube.com/watch?v=lVynu4bo1rY&t=766s) | double norm、non-residual post-norm |
| [14:19](https://www.youtube.com/watch?v=lVynu4bo1rY&t=859s) | LayerNorm 与 RMSNorm |
| [15:49](https://www.youtube.com/watch?v=lVynu4bo1rY&t=949s) | FLOPs 不等于 runtime，数据移动的重要性 |
| [16:46](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1006s) | 课堂问答：norm FLOPs/runtime 示例的理解 |
| [18:02](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1082s) | 删除 bias 的现代趋势与经验动机 |

### 25.2 激活、门控与 parallel block

| 时间 | 内容 |
|---|---|
| [20:15](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1215s) | activation zoo：ReLU/GELU/GLU 系 |
| [21:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1292s) | gated activation 成为主流 |
| [22:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1352s) | 一条内容支路乘一条门支路 |
| [24:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1443s) | GeGLU 与 SwiGLU；gating 比门函数差别更重要 |
| [24:16](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1456s) | 三矩阵 vs 两矩阵，宽度乘 `2/3` |
| [25:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1501s) | 小收益、重复实验与误差条 |
| [27:19](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1639s) | serial vs parallel block |
| [28:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1712s) | parallel 共享 norm、融合投影的速度机会 |

### 25.3 位置编码与 RoPE

| 时间 | 内容 |
|---|---|
| [31:04](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1864s) | absolute/relative/RoPE 位置方案 |
| [32:34](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1954s) | 课程模型调查中的 RoPE 趋势 |
| [33:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1981s) | 希望点积依赖相对位移的出发点 |
| [34:25](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2065s) | 旋转保持几何关系的直觉 |
| [35:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2103s) | token/位置小例子 |
| [36:20](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2180s) | 坐标按二维对旋转；字幕 “3D” 的转写误差 |
| [36:38](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2198s) | 不同坐标对使用不同频率 |
| [37:15](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2235s) | partial/p-RoPE 例子 |
| [38:24](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2304s) | 每层 attention 对 Q/K 应用 RoPE |
| [39:16](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2356s) | 课堂问答：是否有高维非二维配对旋转 |
| [39:37](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2377s) | 课堂问答：怎样从模型报告提炼可靠知识 |
| [40:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2421s) | 课堂问答：parallel block 的质量证据与缺少干净消融 |
| [41:27](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2487s) | p-RoPE 与位置设计继续讨论 |
| [42:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2523s) | 课堂问答：相对位置信号作用对象；以 PDF Q/K 公式为准 |

### 25.4 超参数、词表与正则化

| 时间 | 内容 |
|---|---|
| [43:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2620s) | hyperparameters 主题开始 |
| [45:04](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2704s) | 普通 FFN `4×` 经验基线 |
| [45:45](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2745s) | GLU `2.67×` 与更宽配置 |
| [47:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2823s) | T5 `64×` 极端例外及系统直觉 |
| [48:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2883s) | Kaplan 50M 参数实验的宽 basin |
| [50:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3005s) | `head_dim × heads / model_dim` |
| [51:31](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3091s) | depth-width aspect ratio |
| [52:50](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3170s) | 深度串行、宽度 tensor parallel、系统权衡 |
| [53:51](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3231s) | Kaplan 的 aspect-ratio 证据 |
| [54:18](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3258s) | Tay 的深宽/系统证据与宽容区 |
| [55:11](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3311s) | 单语与多语种 vocabulary size |
| [56:38](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3398s) | 课堂问答：多模态是否使用独立 image vocabulary |
| [57:12](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3432s) | 课堂问答：tokenizer 比较与 bits per byte |
| [59:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3561s) | dropout/regularization 主题 |
| [59:49](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3589s) | 大数据、单遍预训练与 overfitting 直觉 |
| [60:47](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3647s) | 公开报告中的 dropout/weight decay |
| [61:36](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3696s) | weight decay 常通过优化动态起作用 |
| [62:24](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3744s) | weight decay 与 cosine learning rate 互动 |
| [63:38](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3818s) | 超参数经验规则总结 |
| [64:07](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3847s) | 课堂问答：其他生成架构能否移植这些技巧 |
| [64:34](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3874s) | 课堂问答：为什么正则化会影响优化 |

### 25.5 稳定性与 attention 部署

| 时间 | 内容 |
|---|---|
| [65:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3901s) | stable training 与尖峰曲线 |
| [66:24](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3984s) | output/attention 两处 softmax 风险 |
| [67:06](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4026s) | output softmax 与 z-loss |
| [68:31](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4111s) | 共同 logit 平移不改概率，z-loss 锚定 normalizer |
| [69:27](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4167s) | attention 稳定性主题 |
| [70:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4203s) | QK norm 放置与机制 |
| [71:26](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4286s) | QK norm 的现代采用趋势与经验边界 |
| [72:07](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4327s) | logit soft-capping |
| [73:08](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4388s) | 稳定干预消融：soft-cap 可能损害质量 |
| [74:09](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4449s) | attention efficiency 变体开始 |
| [75:53](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4553s) | 训练/prefill 的工作量与访存 |
| [77:02](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4622s) | autoregressive incremental decode |
| [77:23](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4643s) | KV cache 口头说明；笔记中精确化为缓存 K/V |
| [78:02](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4682s) | decode 算术强度下降 |
| [79:32](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4772s) | MQA：共享 K/V heads |
| [81:07](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4867s) | GQA：在 MHA/MQA 之间分组 |
| [81:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4900s) | MLA 只作下一讲预告 |
| [82:54](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4974s) | GQA 性能/成本图与后续 inference 课程 |
| [83:17](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4997s) | 课堂问答：规则默认值 vs hyperparameter search |
| [84:16](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5056s) | 课堂问答：训练中动态改变超参数 |
| [84:55](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5095s) | 课堂问答：MQA/GQA 必须按共享结构训练 |
| [85:08](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5108s) | sliding-window attention 开始 |
| [86:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5161s) | Command A：每四层一层 full 的例子 |
| [86:44](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5204s) | NoPE/RoPE 与 local/global 组合 |
| [87:29](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5249s) | 2026 课程样本中的 hybrid 趋势 |
| [88:04](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5284s) | Qwen 3.5 以便宜层和 full attention 交替 |
| [88:37](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5317s) | 全讲总结：共识与仍在变化的部分 |
| [89:08](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5348s) | 最后一句与字幕末段 |

---

## 26. 来源与内容边界

### 26.1 本讲直接来源

1. **官方讲义**：[Stanford CS336 `lecture_03.pdf`](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)，67 页。架构图、模型汇总表、曲线和课程公式均以此为准。
2. **官方公开视频**：[Stanford Online Lecture 3](https://www.youtube.com/watch?v=lVynu4bo1rY)。用于补充老师口头限定、推理过程和课堂问答。
3. **字幕来源**：该视频 YouTube 人工 `English (United States)` timed-text 轨；2020 段；最后片段约 89:08–89:09。自动生成 `English (auto-generated)` 轨只用于识别轨道存在，不作为正文源。

### 26.2 【课程】覆盖了什么

- 原始/现代 Transformer 对比；
- pre/post/non-residual post-norm，LayerNorm/RMSNorm/bias；
- activation、GLU variants、serial/parallel；
- absolute/relative/RoPE；
- $`d_{ff}/d`$、heads、aspect、vocab、dropout/weight decay；
- z-loss、QK norm、soft-cap；
- prefill/decode 算术强度，MHA/MQA/GQA/MLA 定位；
- sparse/sliding/interleaved attention 与模型调查。

### 26.3 【视频补充】覆盖了什么

- “clean residual stream”、warmup 边界；
- FLOPs 与 runtime 的区别；
- gated 收益的证据强弱、parallel 缺少干净消融；
- 如何从模型报告提炼经验而不写成定律；
- tokenizer、multimodal vocab、动态超参数等课堂问答；
- decode 逐步执行、GQA 成本/质量直觉；
- 2026 时点的 hybrid attention 趋势。

### 26.4 【补充解释/例子】由本笔记新增

- 所有二维/四维 norm、激活、门控、RoPE 数值手算；
- 三组合成模型参数/shape 账；
- vocab bytes、dropout 和 weight-decay 小向量；
- softmax、z-loss、QK norm、soft-cap 数字例子；
- 两组 KV cache MiB、GQA 一般强度式；
- `n=8,w=3` 可见位置、hybrid pair 数；
- 决策树、误区、术语表和 50 道自测。

这些例子用于解释课程结论，不是 Stanford 课堂原例，也不是新实验证据。

### 26.5 课件表格数字的核验原则

所有在正文中继续参与推导的数字都独立复算。PDF 第 43 页的 PaLM 行打印 `48 heads / head dim 258 / model dim 18432 / ratio 1.48`：

```math
48\times258/18{,}432=0.671875,
```

其倒数约：

```math
1/0.671875\approx1.488.
```

这与同表其他行采用的 $`Hd_h/d`$ 方向不一致，且 `258` 也可能是课件转录/排版问题。正文因此只采用统一定义：

```math
r_{attn}=Hd_h/d,
```

并没有把 PaLM 的 `1.48` 混入同方向表格。遇到模型 meta-table，应回到原报告和实现确认。

### 26.6 进一步阅读：原始论文/一手材料

- Transformer：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- RMSNorm：[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- GLU variants：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- RoPE：[RoFormer](https://arxiv.org/abs/2104.09864)
- scaling/shape sweeps：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- T5：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- PaLM 与 z-loss 配方：[PaLM](https://arxiv.org/abs/2204.02311)
- MQA：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- GQA：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Sparse Transformer：[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- Longformer：[The Longformer](https://arxiv.org/abs/2004.05150)

模型采用表以课程 PDF 为历史调查源；上列论文帮助理解机制，不表示其所有实验设置能直接迁移到今天的模型。

---

## 27. PDF 67 页覆盖表与视觉核对

### 27.1 检查方式

本地 `lecture_03.pdf` 经 pypdf 确认 67 页，并用 pypdfium2 将 **67/67 页全部渲染**为页面图；7 张 contact sheets 覆盖全部页，逐页检查标题、图、表和裁切。关键公式/表格/曲线页又以原始分辨率复核，包括第 3、4、10、13、14、21–23、28、30–35、38–50、52–67 页。

视觉结果：没有缺页或被裁掉的主体内容；密集模型时间线/元数据表的小字不适合逐行转录，本笔记只使用它们支持课程明确讲出的趋势，并对进入公式的表格数字单独复算。

### 27.2 逐页段落到正文的映射

| PDF 页 | 页面主题 | 本笔记落点 |
|---:|---|---|
| 1 | 标题、课程信息 | 页首资料说明 |
| 2 | goals：现代架构与模型趋势 | §0、§2 |
| 3 | 2017 原始 Transformer 简图 | §4.1 |
| 4 | 现代教学 variant | §4.2 |
| 5–7 | 模型发布数据与架构趋势样本 | §2.3、§4.3、§26.5；作为趋势背景，不逐行抄表 |
| 8 | 本讲纲要：架构、超参数、稳定性 | §2.3、§20 |
| 9 | LLaMA-like、QK norm、hybrid attention 趋势过渡 | §2.3、§4.3 |
| 10 | post-norm vs pre-norm 图 | §5.2–5.3 |
| 11–12 | residual/gradient 稳定图与证据 | §5.3–5.4 |
| 13 | double/non-residual post-norm | §5.6–5.7 |
| 14 | LayerNorm vs RMSNorm 公式 | §6.1–6.5 |
| 15–16 | RMSNorm 的 FLOPs/runtime/数据移动 | §6.7–6.8 |
| 17 | RMSNorm validation | §6.8、§26.6 |
| 18 | 删除 bias | §6.9 |
| 19 | norm recap | §6.10–6.11 |
| 20 | activation zoo | §7.3–7.6 |
| 21 | ReLU/GELU 公式与模型例 | §7.3–7.6 |
| 22 | GLU/ReGLU 门控结构 | §8.1–8.3 |
| 23 | GeGLU/SwiGLU 与 `2/3` | §8.3–8.6 |
| 24 | gated 效果实验：一致的小幅收益 | §8.9 |
| 25–26 | activation/gating 实验表 | §8.9；只总结证据方向与误差边界 |
| 27 | serial/parallel 主题 | §9.1–9.3 |
| 28 | 两种 block 公式与 PaLM 速度 | §9.1–9.6 |
| 29 | parallel 的模型采用/趋势 | §9.5–9.6 |
| 30 | sinusoidal/absolute/relative/RoPE 总览 | §10 |
| 31 | 课件命名 $`\delta=i-j`$；点积约定下得到 $`R(j-i)=R(-\delta)`$ | §11.3–11.5 |
| 32 | 旋转直觉图 | §11.1–11.2 |
| 33 | 二维坐标配对 | §11.3、§11.8 |
| 34 | block-diagonal RoPE 矩阵 | §11.5、§11.8 |
| 35 | RoPE 代码/每层 QK 应用 | §11.9–11.11 |
| 36 | hyperparameter 问题列表 | §12–15 导言 |
| 37 | 普通 FFN `4×` | §12.1–12.2 |
| 38 | GLU ratios 模型表 | §12.3 |
| 39 | T5 64× 与新例外 | §12.4–12.5 |
| 40 | 50M 参数 FF ratio 曲线 | §12.6 |
| 41 | FF ratio 结论边界 | §12.7 |
| 42 | head 内宽关系 | §13.1–13.3 |
| 43 | heads/head_dim/model_dim 表 | §13.4、§26.5 |
| 44 | depth-width aspect 表 | §13.5–13.6 |
| 45 | 深度串行与 pipeline 图 | §13.7 |
| 46 | Kaplan/Tay aspect 图 | §13.7 |
| 47 | vocabulary size 表 | §14.1–14.5 |
| 48 | 预训练是否需要 regularization | §15.1 |
| 49 | dropout/weight-decay 模型表 | §15.5–15.6 |
| 50 | weight decay 与学习率/优化动态 | §15.7 |
| 51 | hyperparameter recap | §0、§20 |
| 52 | stable/unstable 训练曲线 | §16.1 |
| 53 | 两个 softmax 风险 | §16.2–16.6 |
| 54 | z-loss | §16.7–16.8 |
| 55 | QK norm | §16.9–16.10 |
| 56 | logit soft-capping 与消融 | §16.11–16.14 |
| 57 | attention interventions 过渡 | §17.1 |
| 58 | prefill 工作量/访存/强度 | §18.8–18.9 |
| 59 | incremental KV cache 动画 | §17.4–17.6 |
| 60 | decode 算术强度 | §18.10 |
| 61 | MQA 图与强度式 | §18.1–18.3、§18.11 |
| 62 | GQA 与 MLA | §18.12、§18.15 |
| 63 | MHA/MQA/GQA 性能成本图 | §18.14 |
| 64 | sparse attention patterns | §19.1–19.6 |
| 65 | interleaved full/local、NoPE/RoPE | §19.7–19.10 |
| 66 | 近期 hybrid 模型例 | §19.11 |
| 67 | 全讲 recap | §20、§28 |

---

## 28. 最终能力清单

完成本讲后，你应能在不看视频的情况下：

- [ ] 从 token IDs 开始写出 embedding、Q/K/V、score、FFN、logits 的所有 shape；
- [ ] 用公式和箭头区分 post/pre/non-residual post-norm；
- [ ] 手算 2–4 维 LayerNorm 与 RMSNorm；
- [ ] 解释 RMSNorm/bias 删除为何可能通过数据移动影响 runtime；
- [ ] 手算 ReLU/GELU/Swish 和一条 gated FFN；
- [ ] 从两个矩阵 vs 三个矩阵推出 `2/3` 和 `8/3`；
- [ ] 区分 serial/parallel 的真实数据依赖；
- [ ] 比较四类位置编码；
- [ ] 从二维旋转逐项推出 $`R(i)^TR(j)=R(j-i)=R(-\delta)`$，并说明课件 $`\delta=i-j`$；
- [ ] 说明标准 RoPE 在每层旋转 Q/K、通常不旋转 V；
- [ ] 计算 $`d_{ff}/d`$、$`Hd_h/d`$、$`d/L`$，并区分经验值与定律；
- [ ] 对给定 $`V,d,L,H,d_h,d_{ff}`$ 做 block/embedding 粗略参数账；
- [ ] 区分 dropout 与 weight decay，并手算一次更新；
- [ ] 稳定计算 softmax，并解释 z-loss、QK norm、soft-cap 的不同作用层级；
- [ ] 准确说明 KV cache 保存 K/V，而不是 Q 或 score；
- [ ] 用 $`2bnH_{kv}k\times bytes`$ 算单层/全模型 cache；
- [ ] 区分 MHA、MQA、GQA 的 $`H_q,H_{kv},k`$；
- [ ] 从工作量/访存式推出 prefill、decode、MQA/GQA 算术强度趋势；
- [ ] 列出 sliding window 的可见位置，计算 exact pairs 与 $`O(nw)`$；
- [ ] 解释 local 信息接力、周期性 full 层和 NoPE/RoPE 的独立权衡；
- [ ] 面对新模型报告时，先查公式、shape、参数、硬件与实验条件，不把课程 2026 快照写成永恒规则。

如果上面任何一项只能“背一句话”而不能复算，请使用 §20.7 的 45 分钟路线，并做对应的 §23 题目。真正掌握的标准不是认得缩写，而是能从 shape、四则运算和数据流把结论重新推出。
