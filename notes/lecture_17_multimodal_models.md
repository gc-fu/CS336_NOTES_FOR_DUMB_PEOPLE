# Lecture 17：多模态模型——怎样让 Transformer 看图，也能生成图

> CS336 Spring 2026 · Multimodal Models
>
> 官方课程页：[Stanford CS336](https://cs336.stanford.edu/) · [课程视频](https://www.youtube.com/watch?v=26FtD08ZpOU) · 官方可执行讲义 `lecture_17.py`

<a id="sec-0"></a>
## 0. 第一次阅读怎么用这份笔记

这份笔记默认你只会四则运算。第一次读时：

1. 先读 §2–§4，弄清“图片为什么也要变成 token”。
2. 再读 §5–§10，亲手算 CLIP 与 SigLIP；这是一切后续模型的地基。
3. 按 §11–§21 看 LLaVA、Qwen-VL、Chameleon 怎样把视觉接进语言模型。
4. 最后做 §28 自测；不会再回正文。§1 的五分钟卡首次阅读可以跳过。

来源标签：

- 【课程】官方源码、课程图片或字幕明确展示的内容。
- 【视频补充】讲者口头解释、问答或源码图没有写出的限定。
- 【补充解释】为了让初学者算得出来而增加的教学桥梁。
- 【补充】由一手论文或官方仓库核对的信息。
- 【延伸】课程主线外、但有助于建立全局地图的内容。

本讲没有 PDF，所以没有“PDF 页码覆盖”。§31 给出官方 Python 源码 1–302 行的无缝覆盖，§32 记录 32 张本地图的实际视觉检查。

<a id="toc"></a>
### 导航

- [五分钟复习卡](#sec-1)
- [最低前置知识与词典](#sec-2)
- [omni 与两类目标](#sec-3)
- [非文本如何变 token](#sec-4)
- [CLIP 形状与目标](#sec-5)
- [CLIP 三配对手算](#sec-6)
- [CLIP 数据、ViT 与 zero-shot](#sec-7)
- [CLIP 边界](#sec-8)
- [SigLIP](#sec-9)
- [LLaVA](#sec-11)
- [LLaVA-OneVision / AnyRes](#sec-12)
- [Qwen-VL → Qwen3-VL](#sec-15)
- [Chameleon 与 VQ-VAE](#sec-21)
- [决策树与误区](#sec-25)
- [自测题](#sec-28)
- [视频导航](#sec-30)
- [源码与图片覆盖](#sec-31)
- [来源与边界](#sec-33)
- [学完能力清单](#sec-34)

<a id="sec-1"></a>
## 1. 五分钟复习卡（首次阅读跳过）

1. **共同接口**：Transformer 接收 token；图片可变成连续向量 token，或变成离散整数 code token。（§4）
2. **CLIP**：同一 batch 内，正确图文配对的相似度要高；图找文、文找图各做一次交叉熵（正确答案概率越小，惩罚越大）。（§5–§6）
3. **SigLIP**：不做整行 softmax（把一组分数变成和为 1 的概率）；每个图文 pair 用 sigmoid（把一个分数压到0–1）单独判断“匹配/不匹配”。（§9）
4. **标准理解型 VLM**：视觉编码器 → projector/adaptor → 语言模型。projector 把视觉宽度变成 LM 宽度。（§11）
5. **AnyRes**：高分辨率图切 tile；单图、多图、视频必须在清晰度与 token 预算间取舍。（§12）
6. **MRoPE**：为时间、高度、宽度分别给位置；Qwen3 再把三轴交错放进低/高频维度。（§17–§19）
7. **64 vs 66**：224/14 后是 16×16 patch；2×2 merge 得 8×8=64 个视觉内容 token；再加 `vision_start`、`vision_end` 两个边界 token，进入 LM 共 66。（§16）
8. **Chameleon**：VQ-VAE 把图变成 codebook 整数，统一交给自回归 Transformer，因此能读图也能吐出图 token。（§21）
9. **离散化代价**：512×512 图变 1024 个、每个从 8192 个 code 选一项；接口统一，但文字/OCR 等细节可能丢失。（§21）
10. **总原则**：理解偏语义；生成还要细节。一个 representation 不一定同时最适合两者。（§3、§24）

<a id="sec-2"></a>
## 2. 最低前置知识与词典：先把每个词变成人话

【补充解释】

- **模态（modality）**：信息的形式，例如文字、图像、音频、视频。
- **多模态模型**：一次能处理两种或更多模态的模型。
- **Transformer**：把一串 token 反复混合信息的神经网络。
- **token**：模型一次处理的一个信息单元。文字 token 可以是字节片段；视觉 token 常代表一块图像。
- **离散（discrete）**：只能从一张有限清单选整数，例如 0、1、2；**连续（continuous）**：用实数向量，例如 `[0.2,-1.1,3.0]`。
- **encoder（编码器）**：把原始输入压成向量；**decoder（解码器）**：从表示还原或生成输出。
- **embedding（嵌入）**：用一列数字代表语义。`[N,d]` 表示 N 个对象、每个 d 个数。
- **tensor（张量）**：多维数字表；标量是 0 维、向量是 1 维、矩阵是 2 维，`[3,4]` tensor 就是 3 行 4 列。
- **batch**：一次一起计算的一组样本；batch size `N=3` 就是同时放 3 对图文。
- **logit**：softmax/sigmoid 之前的未归一化分数；越大通常表示模型越偏向该选项。
- **loss（损失）**：越小越好的“答错程度”。训练会改变参数来降低它。
- **softmax**：把同一组 logits 变成总和为 1 的概率，$`p_i=e^{z_i}/\sum_j e^{z_j}`$。例如 logits `[0,0]` 变成 `[1/2,1/2]`。
- **sigmoid**：把一个 logit 单独压到 0 与 1 之间，$`\sigma(z)=1/(1+e^{-z})`$；它不要求和别的格加起来等于 1。
- **cross-entropy（CE，交叉熵）**：只有一个正确类别时，若模型给正确项概率 `p`，损失是 `-ln p`；若 `p=1`，损失 0；若 `p` 很小，惩罚很大。
- **temperature（温度）** `tau`：控制分数尖锐度，常见 `logit=s/tau`。小 `tau` 会放大差距。
- **shape（形状）**：tensor 每一轴有多少格。例如 `[3,4]` 是 3 行、每行 4 个数。
- **L2 length（L2 欧氏长度）**：把每个分量平方、相加、再开平方；`[3,4]` 的长度是 $`\sqrt{3^2+4^2}=5`$。
- **forward（前向计算）**：数据从输入依次经过模块得到输出；冻结参数仍然可以做 forward。
- **bias（偏置）**：线性层乘法后给每个输出维度再加的一个数；`d_out=6` 通常有 6 个 bias 参数。
- **gradient（梯度）**：参数轻微变化时 loss 会怎样变化的局部斜率；训练按它决定参数更新方向。
- **Q/K（query/key，查询/键）**：attention 中，Q 像“我要找什么”，K 像“我有哪些索引”；Q 与 K 的点积形成 attention logit。
- **V（value，值）**：attention 找到相关 K 后真正取回并加权汇总的内容；不要与词表大小 `V` 混淆，具体小节会重申。
- **TPU（Tensor Processing Unit）**：Google 的张量计算加速器。**FLOP** 是一次浮点算术操作，**FLOPs** 是完成任务所需操作数，**FLOP/s** 是每秒执行多少操作；三者不能混。
- **device-day（设备天）**：设备数乘运行天数，例如 32 台跑 5 天是 160 device-days；不同硬件的一个 device-day 计算量可能完全不同。
- **autoregressive（自回归）**：每次根据已经出现的 token 预测下一个 token，再把新 token 放回前缀继续生成。
- **latent（潜在表示）**：模型内部较紧凑的连续表示；它不是人类直接看到的像素，也不一定具有离散 token ID。
- **context（上下文）**：当前一次预测能够看到的 token/信息；context length 是最多能同时看到多少 token。
- **processor（处理器/预处理程序）**：把图片或视频 resize、切 patch、合并 token、插入边界标记，并整理成模型输入的程序；这里不是泛指 CPU。
- **optimizer（优化器）**：读取 gradient 后决定每个参数如何更新的训练规则，例如 AdamW。
- **BOS/EOS**：begin/end of sequence，序列开始/结束 token。
- **ViT**：Vision Transformer，把图片切 patch，再像处理 token 一样处理 patch。
- **MLP**：multi-layer perceptron，多层全连接网络；projector 常用它变换向量宽度。
- **OCR**：optical character recognition，从图片中识别文字。
- **VLM**：vision-language model，视觉语言模型。
- **MoE**：mixture of experts，每个 token 只激活部分“专家”子网络；`235B-A22B` 表示总参数约 235B、每 token 激活约 22B，`B=10^9`。
- **CoT（Chain of Thought，思维链）**：回答中写出的中间推理步骤；**SFT（Supervised Fine-Tuning，监督微调）**：用示范答案做监督训练；**distillation（蒸馏）**：让较小模型学习 teacher 产生的答案或分布；**RL（Reinforcement Learning，强化学习）**：按奖励信号调整生成策略。

数学只需这几件：

```math
e^0=1,\qquad e^1\approx2.718,\qquad e^2\approx7.389.
```

`exp(x)=e^x`；`ln` 是 `exp` 的反函数。计算器输入 `ln(0.5)` 得约 `-0.6931`，所以正确项概率 0.5 的交叉熵是 `0.6931`。

<a id="sec-3"></a>
## 3. 一条主线：从 text→text 到 omni

【课程｜[00:32](https://www.youtube.com/watch?v=26FtD08ZpOU&t=32s)】过去课程主要研究文字输入、文字输出。现实世界还有图、声、视频。

**omni model（全模态模型）**的理想是：

- 输入任意模态组合，理解它们；
- 输出任意模态组合，生成它们。

这里有两项不能混：

| 任务 | 问题 | 例子 | representation 最需要什么 |
|---|---|---|---|
| understanding | 输入非文字怎样读懂？ | 图 → “一只黑猫” | 保住语义 |
| generation | 输出非文字怎样产生？ | “黑猫” → 图 | 保住像素、纹理、空间细节 |

一个只保留“这是猫”的向量，分类很够用；但无法还原猫每根胡须。故“擅长理解”不自动推出“能高质量生成”。

【视频补充｜[03:57](https://www.youtube.com/watch?v=26FtD08ZpOU&t=237s)】讲者用两个问题组织整讲：怎样输入非文本，以及怎样输出非文本。前半以 CLIP/SigLIP 和注入 LM 为主；Chameleon 才把输出图像纳入同一 token 流。

<a id="sec-4"></a>
## 4. 图片为什么必须先变成 token

【课程】Transformer 的接口是一串 token。文字也不是天生 token：tokenizer 先把字符串变整数，再查 embedding。图片同样需要翻译器。

### 4.1 连续视觉 token

假设图片被切成 4 块，每块编码成 3 维向量：

```math
X_{vision}\in\mathbb{R}^{4\times3}
=
\begin{bmatrix}
0.2&0.1&-0.3\\
0.7&0.4&0.0\\
-0.1&0.8&0.2\\
0.5&0.5&0.5
\end{bmatrix}.
```

这里 `4` 是视觉 token 数，`3` 是每个 token 的宽度。它们是连续实数，适合“把图交给语言模型理解”。

### 4.2 离散视觉 token

另一条路是准备 codebook（码本），例如 8 个视觉小字典项。每块只选最近的编号：

```text
连续向量 -> 最近 code -> 整数
[0.9,0.1] -> code 3 -> token 3
[0.0,0.8] -> code 6 -> token 6
```

得到 `[3,6,...]` 后，普通自回归模型可以像预测文字整数一样预测图像整数，因此也能生成图。

### 4.3 同一个 Transformer，不等于同一种信息密度

文字 token “cat”已经高度语义化；一个图像 token 可能只对应局部颜色纹理。视频每秒又有许多帧。若不限制，视觉 token 会淹没文字 token，导致计算量、训练权重和稳定性都变差。

<a id="sec-5"></a>
## 5. CLIP：让同一张图和它的文字靠近

【课程｜[04:17](https://www.youtube.com/watch?v=26FtD08ZpOU&t=257s)】【补充：[CLIP 原论文](https://arxiv.org/abs/2103.00020)】

CLIP 全称 **Contrastive Language–Image Pre-training**，即“对比式图文预训练”。contrastive 的生活类比：三张证件照配三个人名；模型不必逐字写传记，只要把正确名字排在错误名字前。

### 5.1 形状字典

一个 batch 有 `N` 对图文：

- 图像编码器输出 `I_raw [N,d]`；
- 文本编码器输出 `T_raw [N,d]`；
- 每行除以自身长度，得到单位向量 `I,T [N,d]`；
- 余弦相似度矩阵 `C=I T^T [N,N]`；
- logits 矩阵 `Z=C/tau [N,N]`。

`C[i,j]` 表示第 `i` 张图与第 `j` 段文字的余弦相似度。对角线 `C[i,i]` 是数据标注的正确 pair。softmax 只作用在 logits `Z` 上，不会再把 `Z` 除一次温度。

**embedding 到矩阵乘法的完整桥。** 取两张图、两段文字，每个 embedding 宽度为 2：

```math
I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},\qquad
T=
\begin{bmatrix}
1&0\\
0.6&0.8
\end{bmatrix}.
```

四行长度都为 1：例如第二个文字向量长度 $`\sqrt{0.6^2+0.8^2}=\sqrt1=1`$，所以它们已经 L2-normalized（L2 归一化）。`transpose`（转置）把行列互换：

```math
T^T=
\begin{bmatrix}
1&0.6\\
0&0.8
\end{bmatrix}.
```

矩阵乘法的每一格都是“`I` 的一行点乘 `T^T` 的一列”，也就是一个图 embedding 点乘一个文字 embedding：

```math
\begin{aligned}
C[0,0]&=1\times1+0\times0=1,\\
C[0,1]&=1\times0.6+0\times0.8=0.6,\\
C[1,0]&=0\times1+1\times0=0,\\
C[1,1]&=0\times0.6+1\times0.8=0.8.
\end{aligned}
```

所以：

```math
C=IT^T=
\begin{bmatrix}
1&0.6\\
0&0.8
\end{bmatrix}.
```

若 $`\tau=1`$，则 $`Z=C`$。图0那一行 softmax 的正确概率：

```math
\frac{e^1}{e^1+e^{0.6}}
=\frac{2.718}{2.718+1.822}
=0.5987.
```

其余行/列也按同一个公式算，不能只看矩阵最大值：

| 方向 | 候选 logits | 正确概率 | CE=`-ln p` |
|---|---:|---:|---:|
| 图0→文字 | `[1,0.6]` | `2.718/(2.718+1.822)=0.5987` | `0.5130` |
| 图1→文字 | `[0,0.8]` | `2.226/(1+2.226)=0.6900` | `0.3711` |
| 文0→图片 | `[1,0]` | `2.718/(2.718+1)=0.7311` | `0.3133` |
| 文1→图片 | `[0.6,0.8]` | `2.226/(1.822+2.226)=0.5498` | `0.5981` |

所以图找文均值 `(0.5130+0.3711)/2=0.4421`，文找图均值 `(0.3133+0.5981)/2=0.4557`，双向平均 `(0.4421+0.4557)/2=0.4489`。

这一步说明 `[N,d]@[d,N]` 不只是 shape 口号：`N×N` 的每一格确实是一对图文的点积。

### 5.2 为什么要归一化：余弦相似度手算

两向量 `u=[3,4]`,`v=[6,8]`：

```math
u\cdot v=3\times6+4\times8=50,
```

```math
\lVert u\rVert=\sqrt{3^2+4^2}=5,\qquad
\lVert v\rVert=\sqrt{6^2+8^2}=10.
```

余弦相似度：

```math
\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\lVert v\rVert}
=\frac{50}{5\times10}=1.
```

它只比较方向，不奖励“把所有数字放大 100 倍”。对两个**非零向量**，余弦相似度必在 `[-1,1]`；零向量长度为0，原公式会除0，必须另作数值处理。

### 5.3 温度与 logits

若相似度是 `[0.8,0.4]`：

- `tau=1` 时 logits `[0.8,0.4]`，差 0.4；
- `tau=0.1` 时 logits `[8,4]`，差 4，softmax 更确信第一项。

正式记号统一为：

```math
C=IT^T,\qquad Z=C/\tau.
```

CLIP 实现也可写成 `Z=exp(t)*C`，此时 `exp(t)=1/tau`；不要把可学习的 log-scale `t` 和温度 `tau` 混成同一个数，也不要对 `Z` 再除一次 $`\tau`$。

### 5.4 双向损失

图找文：对 `Z` 每一行做 softmax，正确列是 `i`。

```math
L_{i\to t}=-\frac1N\sum_{i=1}^{N}
\ln\frac{e^{Z_{ii}}}{\sum_{j=1}^{N}e^{Z_{ij}}}.
```

文找图：对每一列做 softmax，正确行是 `i`。

```math
L_{t\to i}=-\frac1N\sum_{j=1}^{N}
\ln\frac{e^{Z_{jj}}}{\sum_{i=1}^{N}e^{Z_{ij}}}.
```

最终常取：

```math
L_{CLIP}=\frac{L_{i\to t}+L_{t\to i}}2.
```

“双向”不是把矩阵复制两份；同一 `N×N` 矩阵分别按行、按列解释。

<a id="sec-6"></a>
## 6. CLIP 的 N=3 完整手算：每行、每列都不跳

【补充解释】这一例先给一个合法的余弦相似度矩阵，所有元素都在 `[-1,1]`：

```math
C=
\begin{bmatrix}
0.2&0.1&0\\
0&0.2&0\\
0.1&0&0.2
\end{bmatrix}.
```

令 $`\tau=0.1`$，所以每格除以0.1，也就是乘10：

```math
Z=C/\tau=
\begin{bmatrix}
2&1&0\\
0&2&0\\
1&0&2
\end{bmatrix}.
```

对角线三格 `(0,0),(1,1),(2,2)` 是正确配对。

### 6.1 图找文：逐行

已知 `e²=7.389,e¹=2.718,e⁰=1`。

| 图 | 该行 `exp` | 分母 | 正确概率 | 损失 `-ln p` |
|---|---:|---:|---:|---:|
| 图0 `[2,1,0]` | `[7.389,2.718,1]` | `11.107` | `7.389/11.107=0.6652` | `0.4076` |
| 图1 `[0,2,0]` | `[1,7.389,1]` | `9.389` | `7.389/9.389=0.7870` | `0.2395` |
| 图2 `[1,0,2]` | `[2.718,1,7.389]` | `11.107` | `7.389/11.107=0.6652` | `0.4076` |

```math
L_{i\to t}=\frac{0.4076+0.2395+0.4076}{3}=0.3516.
```

### 6.2 文找图：逐列

| 文 | 该列 logits | 分母 | 正确概率 | 损失 |
|---|---:|---:|---:|---:|
| 文0 | `[2,0,1]` | `11.107` | `7.389/11.107=0.6652` | `0.4076` |
| 文1 | `[1,2,0]` | `11.107` | `7.389/11.107=0.6652` | `0.4076` |
| 文2 | `[0,0,2]` | `9.389` | `7.389/9.389=0.7870` | `0.2395` |

表中“分母”是该列三个指数的和；例如文1：`e¹+e²+e⁰=2.718+7.389+1=11.107`。

```math
L_{t\to i}=\frac{0.4076+0.4076+0.2395}{3}=0.3516,
```

```math
L_{CLIP}=\frac{0.3516+0.3516}{2}=0.3516.
```

这次两方向平均恰好相等，是数字构造的结果，不是一般定律。

### 6.3 batch 为什么提供负例

`N=3` 有 3 个正 pair、`3²-3=6` 个 batch 内负 pair。`N=32768` 时，每张图面对 32767 个错误文字候选。更多负例能教更细的区分，但也增加全局矩阵、通信和 false negative 风险。

<a id="sec-7"></a>
## 7. CLIP 的数据、预处理、ViT 与 zero-shot

### 7.1 400M 不等于 500K×20K

【课程｜[08:47](https://www.youtube.com/watch?v=26FtD08ZpOU&t=527s)】课程/CLIP 论文说用约 50 万查询词，**每个 query 最多取约 2 万条**，并最终训练 4 亿图文 pair。

裸乘法是：

```math
500{,}000\times20{,}000=10{,}000{,}000{,}000=10\text{ billion个理论上限槽位}.
```

只有每个 query 都恰好装满 20K 时才达到 10B；它不是论文报告的实际候选量。query 未装满和重复命中是口径本身就允许的；抓取失败、过滤、去重也可能进一步减少数据，但课程/论文没有在这里给出每种机制的精确减量。因此 10B 不能当最终独立样本数，也不能用它否定 400M。原数据没有公开，OpenCLIP/LAION 是复现路线，不是原 CLIP 数据的副本。

### 7.2 336 resize + center crop 会丢什么

【课程】先把短边缩放到 336，再中心裁出 `336×336`。例如原图 `672×336`，短边已经 336，中心裁切只留下横向 336；左右合计 336 像素被剪掉。中心主体方便，边缘文字/物体可能消失。

### 7.3 ViT-L/14@336 的 patch 数

patch 边长 14：

```math
336/14=24,
```

所以每行 24 块、共：

```math
24\times24=576\text{ 个patch}.
```

OpenAI CLIP 的 ViT-L/14@336 在这些 patch 前加 1 个 class token，因此 vision Transformer 输入长度是：

```math
576+1=577.
```

模型取 `token 0`（class token）的最后表示作为整图表示。这里的 577 是视觉侧；文本侧的 BOS/EOS 是另一条 encoder 序列，不能拿来解释视觉 577。

文本侧以 BOS 开头、EOS 结束，课程模型取最高层 EOS activation 当整段文字 embedding。activation 是网络中间算出的向量，不是新的文字 token。

### 7.4 源码还给了哪些 encoder 事实

【课程内容，源码 74–82】CLIP 同时实验了 ResNet 与 ViT。ResNet 版本把普通全局平均池化替换成 **attention pooling**：先把 feature map 的全局平均 activation 当 query，再让它对各空间位置的 key/value 做 attention 汇聚。文本 encoder 是 12 层、约 63M 参数的 GPT-2 风格 Transformer；输入 BOS…EOS，并取最高层 EOS activation。

### 7.5 zero-shot classifier 怎样做

假设类别为猫、狗：

1. 写 prompt：`a photo of a cat`、`a photo of a dog`。
2. 文本 encoder 得两个类别向量。
3. 新图 encoder 得图向量。
4. 算两个余弦相似度，取较高者。

若相似度猫 0.72、狗 0.31，就预测猫。zero-shot 是“不用该下游任务标注再训练”；不等于模型从未在预训练数据看过猫。

【课程快照】原论文报告 zero-shot CLIP 在特定 ImageNet 协议下达到/超过其所比较的监督 ResNet-50。它是特定模型、prompt ensemble、数据和评测设置的证据，不是“CLIP 永远胜过监督学习”。

### 7.6 为什么 ranking 比逐字生成 caption 更省

CLIP 只需一次图/文编码后比较矩阵；直接预测 caption 要按 token 一个接一个做语言建模。源码的效率图显示在该论文实验中，对比目标以更少处理样本达到更高 zero-shot ImageNet accuracy；它不证明所有生成目标、所有任务都更低成本。

<a id="sec-8"></a>
## 8. CLIP 学到了什么，以及没有保证什么

1. caption 常写“谁/在做什么”，所以 embedding 偏语义。
2. caption 不一定写左下角细字，所以 OCR/精细定位未必被迫学好。
3. batch 内两张不同狗图可能都配“a dog”；算法把它们当负例，这叫 **false negative（假负例）**。
4. 大 batch 给更多负例，也要求跨设备看到全局 softmax 分母，通信和内存变重。
5. 相似度高是统计关联，不保证事实正确、公平或适合部署。

【视频补充｜[19:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1140s)】课堂问答强调：噪声和 false negatives 并没有被神奇消除；规模可能让模型容忍一部分噪声，但不能把它解释成无偏监督。

<a id="sec-9"></a>
## 9. SigLIP：把 N 类排名改成 N² 个二分类

【课程｜[22:37](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1357s)】【补充：[SigLIP 原论文](https://arxiv.org/abs/2303.15343)】

SigLIP 全称 **Sigmoid Loss for Language–Image Pre-training**。它仍算 `N×N` 图文 logits，但每格单独问：“这对匹配吗？”

### 9.1 label 符号与公式

这是 **BCE（Binary Cross-Entropy，二元交叉熵）**：每一格只有“匹配/不匹配”两类。

令：

- 正确对角 pair 标签 `y_ij=+1`；
- 错误非对角 pair 标签 `y_ij=-1`；
- logit 为 `z_ij`；SigLIP 还学习一个全局 scale/temperature 与一个 bias。常见写法是 $`z_{ij}=e^{t'}C_{ij}+b`$：$`e^{t'}>0`$ 控制分数差距，$`b`$ 平移所有 pair 的匹配先验；
- `sigma(a)=1/(1+e^{-a})` 是 sigmoid，把实数压进 `(0,1)`。

每格损失：

```math
\ell_{ij}=-\ln\sigma(y_{ij}z_{ij}).
```

方向检查：

- 正 pair 希望 `z` 大正数，使 `y z` 大正数，loss 小；
- 负 pair 希望 `z` 大负数，此时 `(-1)×(负数)` 大正数，loss 也小。

### 9.2 2×2 完整手算

令：

```math
Z=
\begin{bmatrix}
2&-1\\
0&1
\end{bmatrix},\qquad
Y=
\begin{bmatrix}
+1&-1\\
-1&+1
\end{bmatrix}.
```

逐格先算 `a=yz`，再算 `-ln sigma(a)`：

| 格 | `y` | `z` | `a=yz` | `sigma(a)` | loss |
|---|---:|---:|---:|---:|---:|
| (0,0) 正 | +1 | 2 | 2 | 0.8808 | 0.1269 |
| (0,1) 负 | -1 | -1 | 1 | 0.7311 | 0.3133 |
| (1,0) 负 | -1 | 0 | 0 | 0.5000 | 0.6931 |
| (1,1) 正 | +1 | 1 | 1 | 0.7311 | 0.3133 |

若报告 **每 pair 平均**：

```math
\bar L_{pair}=\frac{0.1269+0.3133+0.6931+0.3133}{4}=0.3617.
```

【课程公式口径】源码截图对应原论文伪代码，外层除 `N` 而不是 `N²`：

```math
L_{paper}=\frac{1.4466}{2}=0.7233.
```

**只有在 batch size `N` 固定时**，两者的梯度才只差固定倍数 `N`；若比较不同 `N`，这个倍数也改变。报告数字时必须说明分母，不能把 `0.3617` 和 `0.7233` 当模型矛盾。

### 9.3 和 CLIP 的关键差别

CLIP 某格概率依赖整行/整列所有候选；SigLIP 某格损失只看该 pair 的 logit。因此跨设备不必为了一个全局 softmax 分母先聚齐所有 embedding。

这不等于“没有通信”：参数梯度仍需同步；计算所有负 pair 也仍有工作量。源码的三设备图展示的是分块轮转 embedding、累加 pair loss 的一种并行方式。

### 9.4 batch 结论的边界

【课程内容，源码 106–111】SigLIP 使用的 WebLI 是十亿量级互联网图文数据；pipeline 还会用 OCR 从图里抽文字、保留质量最高的约 10%，覆盖约 100 种语言。这些是 WebLI/SigLIP 配方快照：OCR 文字可能识错，top-10% 由质量模型定义，100 种语言也不表示各语言数量或质量相同。

【课程时点快照】原论文实验观察：SigLIP 在小于约 16K 的若干设置优于其 CLIP 对照；把 batch 推到 1M 后收益很快饱和，32K 已足够好。它是该论文数据、模型和优化设置下的经验，不是“所有任务最佳 batch=32000”的定律。

课程把 CLIP 的 `256 TPUv3×10天` 与 SigLIP 的 `32 TPUv4×5天` 并列。裸设备天分别是 `256×10=2560` 与 `32×5=160` device-days；但设备代际、每台 FLOP/s、数值精度、实际利用率、训练 pair 和软件栈都不同，所以 `2560/160=16` 不能解释成“精确少了16倍FLOPs”，也不能推出某一代 TPU 更慢。源码第115行括号写“TPUv4 的 FLOP/s 比 TPUv3 低”；没有指定具体芯片变体、精度和指标口径，本文只把它记作课程原话，不用它作硬件事实比较。

<a id="sec-10"></a>
## 10. 从图文 embedding 到会回答问题的 VLM

CLIP/SigLIP 输出的是“整图或 patch 的视觉向量”。语言模型的 hidden width 往往不同，不能直接拼。

最常见模板：

```text
image -> vision encoder -> visual features -> projector/adaptor
                                              |
text tokens -> text embeddings ---------------+-> language model -> text answer
```

设视觉 features `[B,P,d_v]`，其中：

- `B`：batch；
- `P`：视觉 token 数；
- `d_v`：vision encoder 宽度。

若语言模型宽度是 `d_l`，线性 projector：

```math
W\in\mathbb{R}^{d_v\times d_l},\qquad
H_v=X_vW\in\mathbb{R}^{B\times P\times d_l}.
```

例：`X_v [1,4,3]`，`W [3,5]`，输出 `H_v [1,4,5]`。文本有 6 token，embedding `[1,6,5]`；沿序列轴拼接后 `[1,10,5]`。只能在最后一轴同为 5 时拼。

<a id="sec-11"></a>
## 11. LLaVA：一个线性层把 CLIP 接到 Vicuna

【课程｜[29:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1780s)】【补充：[LLaVA 原论文](https://arxiv.org/abs/2304.08485)】

LLaVA 是 **Large Language and Vision Assistant**：

- vision encoder：CLIP ViT-L/14；
- projector：线性矩阵 `W`；
- language model：Vicuna（在 ShareGPT 对话上微调的 LLaMA 系模型）。

课程源码还用 Flamingo 与 Q-Former 作复杂度对照：LLaVA 的 `W` 是一个直接的线性映射；Flamingo/Q-Former 使用更复杂的跨模态交互或 learned-query 模块。这里的结论只是“接口结构更复杂”，不是说复杂模型一定更准或线性 `W` 永远足够。

### 11.1 158K 合成 instruction 数据怎样来

MS COCO 有图片、caption 和 bounding box（框的位置/类别）。课程图展示：把这些文字化的 caption/box 喂给 GPT-4，请它写三类数据：对话、详细描述、复杂推理；再把生成文本和原图配回去，共约 158K。

信息上限很重要：教师当时看的是 caption/box 的文本，不是像素。若 caption 没写招牌上的小字，GPT-4 不能可靠地产生那个视觉事实。生成数据可以扩写任务形式，不会凭空增加源表示缺失的像素证据。

### 11.2 两阶段训练：到底谁动

| 阶段 | vision encoder | `W` | LM | 目的 |
|---|---|---|---|---|
| 1 alignment | 冻结 | 训练 | 冻结 | 学会把视觉坐标翻成 LM 坐标 |
| 2 fine-tuning | 冻结 | 训练 | 训练 | 学会按指令用视觉信息回答 |

“冻结”是参数不更新，不是该模块不做 forward。图仍须经过 vision encoder 才能产生 features。

### 11.3 hallucination 边界

hallucination（幻觉）是输出了图中没有、也无足够证据支持的内容。线性 projector 简单并不自动导致或消除幻觉；训练数据覆盖、语言先验、视觉分辨率和解码都有关。

课程示例里的车辆/人物回答说明模型可组织视觉描述；单个好例子不能给总体准确率，更不能证明对所有细粒度位置都可靠。

<a id="sec-12"></a>
## 12. LLaVA-OneVision：AnyRes 如何保清晰度又守 token 预算

【课程｜[35:29](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2129s)】【补充：[LLaVA-OneVision 论文](https://arxiv.org/abs/2408.03326)】

模型模板：SigLIP vision encoder → 2-layer MLP projector → Qwen2-72B language model。课程源码还明确说 projector 使用 vision Transformer **最后一层之前与之后的 grid features**，不是只取一个整图 class embedding。论文研究单图、多图、视频三种场景。

### 12.1 AnyRes 的生活类比

一张大报纸缩成 336×336，字会糊。AnyRes 像用放大镜分块读：

1. 选 `a×b` 个 tile；
2. 每 tile 缩放到 vision encoder 接受的分辨率；
3. 分别编码；
4. 按空间顺序拼回 token；
5. 通常再保留一个全局/base 视图，告诉模型整张图的布局。

课程图的 tile 分辨率是 `384×384`、patch 边长 14。每边只能放 27 个完整 patch，因为整数除法 `384 // 14=27`；`floor` 表示向下取整数，等价地，kernel/stride 都为14且不额外padding时，输出格数是 `floor((384-14)/14)+1=floor(370/14)+1=26+1=27`。因此每 tile：

```math
27^2=27\times27=729\text{ tokens}.
```

不能把裸除法 `384/14=27.43...` 当成 27.43 个真实 token。若每 tile 729 token，`a=2,b=3`：局部共 `2×3×729=4374`；加 1 个全局图 `729`，总 `5103` token。论文/课程的预算逻辑是在 tile 太多时用双线性插值压缩网格；若预算只有4000，就必须减少 tile 或插值压缩，不能既无限清晰又零成本。具体边缘padding、裁切与插值shape仍以对应processor实现为准。

### 12.2 源图中的三种预算

实际视觉核验的课程图给出：

| 场景 | 图中口径 | 算术 |
|---|---:|---:|
| 单图 | 1 个 base + 最多 9 tile，每块729 | `(1+9)×729=7290` |
| 多图 | 最多12图，每图729 | `12×729=8748` |
| 视频 | 32帧，每帧196 | `32×196=6272` |

单图给高分辨率；多图每张用 base；视频帧更多，所以每帧更低分辨率。目标是数量级接近，不是三者 token 数完全相等。

### 12.3 bilinear interpolation 是什么

双线性插值用周围四格的加权平均估计新格。若左右值 0 和 10，中点是一维简化的 `(0+10)/2=5`。它能平滑改 token 网格尺寸，但不会创造原图已经丢失的文字细节。

<a id="sec-13"></a>
## 13. OneVision 的数据、训练与“跨模态迁移”怎样读

【课程图片视觉核验】

### 13.1 数据图不是只看总数

第一张数据图约 3.2M 单图样本；类别占比为 general 36.1%、document/chart/screen 20.6%、math 20.1%、OCR 8.9%、language 14.3%。3.2M 的 20.1%：

```math
3.2\text{M}\times0.201=0.6432\text{M}.
```

这是约 643,200 个“数据配额”级别估计；图中类别/数据源口径未必严格互斥，不能把乘法结果当逐样本审计。

第二张图约 1.6M OneVision 数据，展示 single 31.2%、multi 43.0%、video 25.9%；因四舍五入和为 100.1%。这不是算错，而是百分比显示精度。

### 13.2 easier → harder 的三阶段

课程训练图：先 alignment projector，再较大规模 full-model 训练，最后更难/更高质量 instruction 数据。具体分辨率、token 和数据表是该版本 recipe 快照；“先易后难”是一种 curriculum（课程式训练顺序），不是所有模型必须照搬。

### 13.3 transfer 图能推出什么

- 单图 chart/table 训练后，模型可在多图保险表格例中联合计算；
- 单图 OCR + 多图关系数据后，在 GUI 截图序列上出现可用行为；
- 单图圈选提示后，在视频帧中追踪被圈球员。

这些是该论文设置里的实证样例/评测趋势。它们支持“任务技能可以跨场景迁移”的假设，不证明没有专门视频数据也能在任意视频任务可靠工作，更不证明迁移因果只来自某一类样本。

课程还把该版本概括为开放模型权重与数据。开放部分材料有利于审计，但仍需核对具体 license、数据版本、训练代码和未公开处理步骤，不能把“open-source”自动等同“逐位可复现”。

<a id="sec-14"></a>
## 14. 一个标准理解型 VLM 的 shape 总账

设：

- batch `B=2`；
- 每图 `P=4` 视觉 token；
- vision width `d_v=3`；
- LM width `d_l=6`；
- 每个 prompt `T=5` 文字 token。

| 步骤 | 输入 shape | 参数 | 输出 shape |
|---|---|---|---|
| vision encoder | 2 images | — | `[2,4,3]` |
| projector | `[2,4,3]` | `W[3,6]` | `[2,4,6]` |
| text embedding | token IDs `[2,5]` | table `[V,6]` | `[2,5,6]` |
| sequence concat | `[2,4,6]` + `[2,5,6]` | — | `[2,9,6]` |
| LM | `[2,9,6]` | Transformer | next-token logits `[2,9,V]` |

`V` 在最后一行表示 vocabulary size（词表大小），不是视觉 value。projector 参数量 `3×6=18`；若带 bias，再加 6，共 24。

<a id="sec-15"></a>
## 15. Qwen-VL：固定 256 个视觉输出 token

【课程｜[46:01](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2761s)】【补充：[Qwen-VL 原论文](https://arxiv.org/abs/2308.12966)】

Qwen-VL 的课程结构：

- vision encoder：OpenCLIP ViT-bigG，patch `14×14`；
- adaptor：一层 cross-attention（交叉注意力），加入二维位置，把可变视觉输入汇成固定 256 个输出；
- LM：Qwen；
- 特殊 token：`<img>` 标图、`<box>` 标框、`<ref>` 标引用对象。

【课程勘误】源码第199行写 `ViT-bigC`；[Qwen-VL 官方仓库](https://github.com/QwenLM/Qwen-VL)与论文使用 `ViT-bigG`。本文采用一手模型报告的 bigG，并保留这条差异，避免把源码拼写当新架构。

cross-attention 的直觉：准备 256 个“提问槽”，每个槽向所有图像 patch 取信息。若输入 patch features `[B,P,d]`，256 个 query `[256,d]`，输出固定 `[B,256,d]`。这和把 256 个 patch 随便平均不同：每个 query 可学会关注不同内容。

### 15.1 三阶段谁训练

| 阶段 | 数据 | vision | adaptor | LM |
|---|---|---|---|---|
| 1 | 大规模、较噪图文 | 训练 | 训练 | 冻结 |
| 2 | 更高质量、任务型、更高分辨率 | 训练 | 训练 | 训练 |
| 3 | instruction 对话 | 冻结 | 训练 | 训练 |

源码数据图显示 stage1 由约 5B 原始候选清理成约 1.4B；stage2 图分列 caption、VQA、grounding、OCR、纯文本等。数字是该报告的构成快照，不等于所有样本互斥、也不证明量大那栏导致某项能力。

<a id="sec-16"></a>
## 16. Qwen2-VL：动态分辨率，不再把所有图压成同样长度

【课程｜[49:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2956s)】【补充：[Qwen2-VL 原论文](https://arxiv.org/abs/2409.12191)】

**dynamic resolution（动态分辨率）**：图大、细节多就产生更多视觉 token；图小则少一些。课程架构图实际标出不同输入可得到 11,427、8、1,125 个图像 token，以及一个视频例的 2,208 token。这说明它不是固定 256。

### 16.1 224×224、ViT/14、2×2 merge：自己算

每边 patch 数：

```math
224/14=16.
```

未合并网格：

```math
16\times16=256\text{ patches}.
```

每 `2×2` patch merge 成一个视觉内容 token：

```math
(16/2)\times(16/2)=8\times8=64\text{ 个视觉内容token}.
```

课程说“66 token”时把送入 LM 的两个视觉边界 token 也算进去：

```math
64\text{ content}+1\text{ vision\_start}+1\text{ vision\_end}=66.
```

因此两句话不冲突：**64 是压缩后的视觉内容 token，66 是加边界后进入语言模型的序列长度。**不同实现的边界命名/模板仍应以实际 processor 输出为准。

### 16.2 视频 2 fps 与 16,384 上限

fps 是 frames per second，每秒取几帧。10 秒视频按 2 fps 抽：

```math
10\text{s}\times2\frac{frame}{s}=20\text{ frames}.
```

Qwen2-VL 的视频 temporal patch size（时间维 patch 大小）为 2：相邻 2 帧组成一个 temporal tube。于是：

```math
20\text{ frames}/2=10\text{ temporal tubes}.
```

空间上 `224/14=16`，再做 `2×2` spatial merge 得 `8×8=64` 个内容 token/tube。因此：

```math
10\times64=640\text{ 个视频内容token},
```

加视觉起止边界后进入 LM 共 `640+2=642` token。`20×64=1280` 只适用于“每帧独立、不做 temporal patch”的纯假设，不是本节 Qwen2-VL 实际口径。

课程列出的 16,384 上限采用 Qwen2-VL 报告/processor 的总视觉 token 预算口径；它不是视频的物理属性，也不是说任意版本都按同一边界计数。更长、更高分辨率视频必须截断、降采样或压缩。

### 16.3 视觉初始化与三阶段训练

【课程内容，源码 217、226–230】Qwen2-VL 使用约 675M 参数的 ViT；语言模型从 Qwen2 初始化，vision encoder 从 DFN 路线初始化。课程给出的三阶段是：stage1 只训练 visual encoder；stage2 训练全部参数；stage3 在 instruction-following 数据上训练 language model。这里“只训练某模块”仍可能要求其他模块做 forward，具体冻结/优化器参数组以实现为准。

<a id="sec-17"></a>
## 17. MRoPE：一个位置拆成时间、高度、宽度

RoPE 全称 **Rotary Position Embedding（旋转位置编码）**：位置决定向量旋转角，使注意力可感知相对位置。MRoPE 的 M 是 multimodal，把位置拆成三轴：

- `t`：temporal，时间/帧；
- `h`：height，行；
- `w`：width，列。

一个 2 帧、每帧 `2×2` 网格：

| token | `(t,h,w)` |
|---|---|
| frame0 左上 | `(0,0,0)` |
| frame0 右上 | `(0,0,1)` |
| frame0 左下 | `(0,1,0)` |
| frame0 右下 | `(0,1,1)` |
| frame1 左上 | `(1,0,0)` |
| frame1 右下 | `(1,1,1)` |

普通一维位置若只写 0、1、2……，模型还得从排列猜“下一格是换行还是换帧”；三轴直接给结构。

Qwen2 的课程图还显示：输出图像后的文字 token 把三轴同步成同一个递增位置，避免普通文字凭空带二维网格含义。

<a id="sec-18"></a>
## 18. Qwen3-VL：interleaved MRoPE、显式时间、DeepStack

【课程｜[52:50](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3170s)】【补充：[Qwen3-VL 技术报告](https://arxiv.org/abs/2511.21631)】

这是 2025/课程 2026 时点的报告快照：dense 2B/4B/8B/32B 与 MoE 30B-A3B、235B-A22B；原生支持最长 256K 的交错多模态 context。它不是永久“最好”排名。

【课程内容，源码 244–245】视觉 encoder 采用 SigLIP-2；课程把它概括为与 SigLIP 同一类架构。这里是视觉 backbone 的来源，不表示整个 Qwen3-VL 只用 SigLIP-2 的原始训练目标。

### 18.1 interleaved MRoPE

旧式可把若干旋转维度连续分给：

```text
t t t t | w w w w | h h h h
```

Qwen3 报告的交错形式：

```text
t w h | t w h | t w h | t w h
```

RoPE 不同维度对应不同旋转频率；交错让 t/w/h 都能拿到低频与高频维度，而不是某一轴只拿某一频段。低频适合慢变化/长距离，高频适合细位置差；这是设计直觉，效果仍须实验验证。

### 18.2 显式 timestamp

Qwen2 主要把时间放位置编码。Qwen3 的课程图把 `<0.0 seconds>`、`<2.0 seconds>` 作为可读文本 token 放在视频片段间。好处是模型能直接把回答中的“第几秒”对齐到输入；代价是增加 token，时间标注也可能有采样误差。

### 18.3 DeepStack

普通 VLM 只把 vision encoder 最后层 features 注入 LM。DeepStack 把不同深度的视觉 features 注入多个 LM 层。生活类比：不只把最终摘要给读者，还在不同章节补充草图、中层结构和最终语义。

这不是“复制同一图 token 多次”。不同视觉层可能偏局部边缘、结构或高层语义；注入也增加接口与训练复杂度。

<a id="sec-19"></a>
## 19. 平方根长度归一化：长视频不能吞掉整个 batch

【课程｜[56:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3365s)】普通 token mean 与平方根归一口径先分清。

两条样本：短样本长度 `L1=4`，长样本 `L2=16`；假设每 token loss 都是 1。

### 19.1 全 batch token mean

```math
L_{tokenmean}=\frac{4\times1+16\times1}{4+16}=1.
```

虽然结果是 1，长样本贡献了分子 `16/20=80%`，短样本仅 20%。

### 19.2 每样本平方根归一，再平均

设第 `i` 条的 token loss 和为 `S_i`。为把“平方根重权”变成可手算公式，本文采用每条先除 $`\sqrt{L_i}`$、再对 batch 取平均的教学口径：

```math
L_{sqrt}=\frac1B\sum_{i=1}^{B}\frac{S_i}{\sqrt{L_i}}.
```

代入：

```math
S_1=4,\quad S_2=16,
```

```math
\frac{S_1}{\sqrt4}=4/2=2,\qquad
\frac{S_2}{\sqrt{16}}=16/4=4,
```

```math
L_{sqrt}=(2+4)/2=3.
```

长/短贡献从 `16:4=4:1` 降到 `4:2=2:1`。数值 3 不是概率，也不需要落在 `[0,1]`；它是优化目标的尺度。改变尺度会影响等效学习率，所以不能只替换公式、不重新验证训练 recipe。

【来源边界】课程源码与 Qwen3-VL 报告公开文本明确说的是 *square-root-normalized per-token loss / square-root reweighting*，却没有在这一处公开完整训练代码及所有跨设备最终分母。上式是忠实表达“每条样本权重从 $`L_i`$ 降到 $`\sqrt{L_i}`$”的最小教学实例，不冒充未公开实现逐字符公式。若实际代码最后再除 $`\sum_i\sqrt{L_i}`$ 而不是 $`B`$，所有样本的**相对权重**仍是 $`\sqrt{L_i}`$，但 loss 的整体尺度不同；必须重新对齐学习率与日志数值。

### 19.3 四阶段预训练快照

课程视觉图经核验：stage0 冻结语言模型，只训练 merger/adaptor，按约 67B token、8K 长度；stage1 全参数、约 1T、8K；stage2 全参数、约 1T、32K；stage3 全参数、约 100B、262,144 长度。这里 `B` 是 billion tokens，不是 batch size；`K` 是 thousand context tokens。

post-training 包括长 chain-of-thought SFT、蒸馏与 RL。图中 benchmark 表只是报告时快照；不同模型可能使用不同数据、工具、采样和评测协议，不能把每一列差值全归因于某个架构变化。

<a id="sec-20"></a>
## 20. Qwen 系列演进：别把变化背成型号广告

| 模型 | 视觉长度 | 位置 | 接入 LM | 主要训练变化 |
|---|---|---|---|---|
| Qwen-VL | adaptor 固定256 | 2D视觉位置 | cross-attention adaptor | 低质大规模→高质任务→instruction |
| Qwen2-VL | dynamic resolution | MRoPE `(t,h,w)` | merge后可变 token | 原生图/视频统一处理 |
| Qwen3-VL | 最长256K交错上下文 | interleaved MRoPE + 文本时间戳 | DeepStack多层注入 | 8K→32K→256K阶段扩长 |

这是“改变了哪些轴”的地图，不是胜负表。更长 context 会增加计算/内存；dynamic resolution 让小图省 token，却使 batch 内长度不齐；DeepStack 增强接口，也增加实现复杂度。

<a id="sec-21"></a>
## 21. Chameleon：把图片也翻成离散整数，再自回归生成

【课程｜[67:18](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4038s)】【补充：[Chameleon 原论文](https://arxiv.org/abs/2405.09818)】

之前的理解型 VLM：图 → 连续 features → LM，通常只输出文字。Chameleon 选择 **early fusion（早融合）**：文字 token 与图片 code token 放在一条序列里，由同一自回归 Transformer 预测下一个 token。

```text
文字：<text> 一只鸟 ...
图片：<start_image> 391 27 8001 ... <end_image>
继续文字：它站在树枝上 ...
```

因此“读图”和“生成图”共享 next-token 接口；真正图像由 image decoder 把 code 还原成像素。

### 21.1 VQ-VAE 从零拆开

VQ-VAE 是 **Vector-Quantized Variational Autoencoder（向量量化变分自编码器）**。本讲只需四步：

1. encoder：图片 `x` → 连续 latent 网格 `z_e(x)`；
2. quantize：每格找 codebook 中最近向量，得到整数索引；
3. decoder：code 向量网格 → 重建图 `x_hat`；
4. 用 reconstruction loss（重建误差）训练，使 `x_hat` 接近 `x`，并让 encoder/codebook 配合。

极小例：codebook：

```math
e_0=[0,0],\quad e_1=[1,0],\quad e_2=[0,1].
```

encoder 输出 `z=[0.8,0.1]`。平方距离：

```math
d(z,e_0)=0.8^2+0.1^2=0.65,
```

```math
d(z,e_1)=(-0.2)^2+0.1^2=0.05,
```

```math
d(z,e_2)=0.8^2+(-0.9)^2=1.45.
```

最近是 `e1`，所以离散 token=1。nearest code 是硬选择；微小输入跨过边界时，token 可能突然改变。

### 21.2 512×512 → 1024 token、8192 code 的位数账

【课程】1024 token 可以看成 `32×32` 网格，因为 `32²=1024`。codebook 8192：

```math
8192=2^{13},
```

所以理想固定长度索引至少 13 bit/token：

```math
1024\times13=13{,}312\text{ bits}=1{,}664\text{ bytes}.
```

这只是 code 索引；不包括 codebook、模型、文件头，也不代表无损压缩。原始 RGB 图若每通道8bit：

```math
512\times512\times3\times8=6{,}291{,}456\text{ bits}.
```

压得很狠，就可能丢 OCR 小字和精细纹理。

### 21.3 为什么还训练新 BPE tokenizer

BPE 是文字离散 tokenizer。图像 code 占用一组 token ID，文字 BPE 占另一组，还要加 `<start_image>` 等边界 token；重新设计词表可避免 ID 冲突并调节文字效率。它不意味着 BPE 本身在压缩像素。

<a id="sec-22"></a>
## 22. Chameleon 的数据、80/20 阶段与稳定性

【课程，源码287–289】stage1 占训练约 80%，课程口径是 **large-scale, unsupervised（大规模、无监督）**：列出 2.9T text tokens、1.5T text/image tokens、400B interleaved text/image tokens。这里“无监督”指课程对这一预训练阶段的标签，不是说互联网图文没有天然共现信号。【补充：Chameleon 原论文称 **completely unsupervised datasets**】这是论文的更强措辞，不应倒灌成课程源码的逐字表述。stage2 占约 20%，用 50% stage1 数据与 50% 高质量数据。

不要把 `2.9T+1.5T+0.4T=4.8T` 当“互斥唯一 token 总量”，因为课程行只给数据类别/规模描述，没有证明三集合完全不重叠、采样位置互斥或计数口径一致。`T=10^12`,`B=10^9`；400B=0.4T 只是单位转换。

### 22.1 text 低 entropy、image 高 entropy

entropy（熵）可理解为“下一个 token 有多难猜”。若文字处只有两个候选概率 `[0.9,0.1]`，很集中；图像 code 若 8 个候选都约 `1/8`，更分散。

先只比较“真实 token 的 NLL（negative log-likelihood，负对数似然）”。低熵例若真实 token 是概率 0.9 的第一项：

```math
\mathrm{NLL}_{text}=-\ln(0.9)\approx0.1053.
```

高熵例有 8 个等概率候选，每个 $`p=1/8=0.125`$，无论真实项是哪一个：

```math
\mathrm{NLL}_{image}=-\ln(1/8)=\ln8\approx2.0794.
```

这一小例的图 token loss 是文字 token 的约 $`2.0794/0.1053\approx19.74`$ 倍，但这**不证明**“任何高熵模态必然发散”：真实模型的概率不一定是这两个分布，token 数、loss 权重、优化器和参数共享方式也都会改变动态。

Chameleon 论文/课程观察的是：文字和图像共享参数及同一个 softmax 输出头时，两种 token 的熵、频率和梯度统计竞争，在其实验中伴随 norm growth（向量长度持续长大）与 logit drift（分数尺度漂移）。这是特定系统的实证机制，不是由“高熵”三个字单独推出的数学定律。课程列的修复：

- **QK norm**：Q 是 query、K 是 key；在 attention 的 Q/K 上归一化，限制点积分数尺度；
- **z-loss**：先定义 softmax 的标量归一化常数 $`Z_{part}=\sum_v e^{z_v}`$，再惩罚 $`(\log Z_{part})^2`$ 过大，抑制整组 logits 一起漂大。这里 $`Z_{part}`$ 不是 §5 的 CLIP logits 矩阵 $`Z`$。

本文不展开 z-loss 对每个 logit 的微积分梯度，只使用可从公式读出的方向：若 $`|\log Z_{part}|`$ 变大，平方惩罚变大。z-loss 不负责保证答案正确，也不等于把 logits 全设为零；它是稳定正则项。

<a id="sec-23"></a>
## 23. 连续 encoder + LM + diffusion：另一条生成路径

【课程总结】离散统一接口很优雅，但离散 code 会损失细节。另一常见路线：

```text
image -> continuous vision encoder -> LM做理解/规划
                                     |
                                     +-> conditioning -> diffusion生成像素/latent
```

diffusion（扩散模型）从噪声多步去噪生成连续图像。它可以保留更丰富生成细节，但系统不再是“一个 next-token loss 解决全部”，训练/推理链更复杂。两条路线没有永恒赢家：取决于是否需要精细生成、速度、token 预算和训练数据。

<a id="sec-24"></a>
## 24. 全讲因果链：每一步解决什么，又留下什么

| 设计 | 解决 | 新代价/边界 |
|---|---|---|
| CLIP 双向对比 | 学图文共同语义空间 | 大 batch/global softmax、false negatives |
| SigLIP pair BCE | 去掉全局 softmax 耦合 | 仍有 N² pair 工作与负例定义 |
| LLaVA projector | 把连续视觉 token 接入 LM | 通常只能文字输出；视觉细节受 encoder 限制 |
| AnyRes/dynamic resolution | 少丢小字/高分辨率细节 | token 变多、长度不齐 |
| MRoPE | 显式表达时空位置 | 坐标/采样协议更复杂 |
| DeepStack | 多层视觉特征注入 | 接口与训练成本增加 |
| VQ-VAE+Chameleon | 图也变离散 token，可统一生成 | 量化/重建损失，训练稳定性困难 |
| continuous+diffusion | 保留连续生成细节 | 不再是单一 token decoder |

理解题先问：模型在哪一步压缩信息？生成题再问：输出端怎样把 representation 还原为目标模态？

<a id="sec-25"></a>
## 25. 选架构时的 IQ60 决策树

1. **只需图文检索/分类表示？**先考虑 CLIP/SigLIP 式双 encoder；不必背一个生成 LM。
2. **要看图后输出文字？**视觉 encoder + projector/adaptor + LM 是最小模板。
3. **小字、图表或大图重要？**检查 crop 是否丢边缘，再预算 AnyRes/dynamic-resolution token。
4. **输入是视频？**同时决定抽帧率、每帧分辨率、总 token 上限和时间位置表示。
5. **要输出图片？**理解型 encoder 不够；选离散 image-token decoder 或连续 diffusion decoder。
6. **希望单一自回归接口？**VQ 式离散 token 简洁，但必须测 OCR/重建损失和稳定性。
7. **训练不稳定？**先分别记录文字/图像 token 的 loss、logit、norm、长度贡献；再判断归一化、z-loss 或采样平衡，不要盲加正则。
8. **读 benchmark 表？**先对齐模型大小、数据、分辨率、prompt、工具、采样与评测版本；只看加粗数字不能作因果结论。

<a id="sec-26"></a>
## 26. 常见误区：错误说法 → 为什么错 → 正确说法

1. **“图片本来就是 token。”**像素数组不是 Transformer 约定的语义 token；先经 patch/encoder 或量化器。
2. **“连续 token 也是整数 ID。”**连续 token 是实数向量；离散 token 才从有限 codebook 选 ID。
3. **“会看图就会生成图。”**理解 encoder 可丢像素细节；生成还需 decoder/扩散或离散 code。
4. **“CLIP 只做图找文。”**它还做文找图，两个交叉熵取平均。
5. **“CLIP 对整张 `N×N` 做一次 softmax。”**实际分别按每行、每列归一化。
6. **“余弦越大只是向量越长。”**归一化后比较方向，长度被除掉。
7. **“温度越大越尖。”**在 `logit=s/tau` 口径，温度越小差距越大。
8. **“N 个 pair 只有 N 个比较。”**相似度矩阵有 `N²` 格；其中 N 正、`N²-N` 负。
9. **“batch 里所有非对角格都是真负例。”**不同图可能语义相同，形成 false negative。
10. **“500K×20K 就是 10B 独立训练对。”**10B 只是每个query都装满20K的理论上限；实际query可未满、结果可重复，之后还可能抓取失败、过滤和去重，最终训练集约400M。
11. **“zero-shot 是从未见过类别。”**它指下游不用该任务标注再训练，不排除预训练看过相关概念。
12. **“ImageNet 一项领先就证明通用视觉理解。”**分类协议不能覆盖 OCR、定位、视频等能力。
13. **“SigLIP 只算 N 格。”**仍有 `N²` pair loss，只是不做跨 pair softmax。
14. **“论文除 N，故每 pair 平均也除 N。”**每 pair mean 除 `N²`；论文除 N 是不同尺度约定。
15. **“SigLIP 不需要通信。”**它降低全局 softmax 耦合；分布式梯度和数据仍通信。
16. **“32K 永远是最佳 batch。”**只是原论文设置的经验饱和区域。
17. **“设备天数可直接比较 FLOPs。”**芯片、利用率、数据量和软件不同。
18. **“冻结 vision encoder 就不执行它。”**冻结只是不更新参数；forward 仍产生视觉 features。
19. **“projector 可把任意 shape 直接拼起来。”**最后一轴必须先变成 LM width。
20. **“GPT-4 合成 LLaVA 数据时看到了像素。”**课程 recipe 中教师主要拿 caption/box 文本，信息有上限。
21. **“AnyRes 无损保留所有信息。”**切块/插值仍有上限，token 预算也限制分辨率。
22. **“三种 OneVision 输入 token 必须完全相同。”**目标是大致平衡；课程图实际是7290、8748、6272。
23. **“跨模态样例证明普遍因果。”**样例/评测支持特定设置的迁移，不隔离所有可能原因。
24. **“Qwen-VL 的256就是256个原 patch。”**是 adaptor 汇聚后的固定输出槽，不必一一对应原 patch。
25. **“动态分辨率让每图 token 相同。”**恰恰相反，分辨率不同会产生不同长度。
26. **“224/14再2×2 merge本身就得到66个内容token。”**几何只得64个视觉内容token；加入 `vision_start`、`vision_end` 两个边界后，送入LM才是66。
27. **“2 fps 是每两秒一帧。”**2 frames per second 是每秒两帧。
28. **“MRoPE 让一个 token 同时处于三个物理位置。”**它给同一 token 的时间、行、列三个坐标。
29. **“interleaved MRoPE 是把 token 顺序打乱。”**打乱的是旋转特征维度的轴分配，不是输入 token 顺序。
30. **“显式时间戳没有误差。”**它仍受抽帧和标注对齐影响。
31. **“DeepStack 就是复制最后层 feature。”**它使用多个视觉深度的特征并注入多层。
32. **“平方根归一后 loss 应在0到1。”**loss 不是概率；尺度可大于1。
33. **“平方根归一让长短样本同权。”**它只把长度权重从 L 降为约 `sqrt(L)`，并非完全相等。
34. **“235B-A22B 表示只存22B参数。”**总参数约235B，每 token 激活约22B；存储仍看总模型和分片。
35. **“VQ-VAE 的最近 code 没有损失。”**量化是有损映射，尤其可能损 OCR 小字。
36. **“8192 code 需要8192 bit/token。”**只需 `log2(8192)=13` 个理想索引 bit。
37. **“1024 token 表示1024像素。”**这里通常是32×32 latent 网格，每 token 覆盖原图一片区域。
38. **“Chameleon 只有图片 token。”**它混合文字 BPE、图像 code 与边界 token。
39. **“三类数据规模可无条件相加。”**集合/采样/计数口径未证明互斥。
40. **“QK norm 与 z-loss 保证正确。”**它们主要控制数值尺度/稳定性，不替代事实监督。
41. **“离散统一一定胜过 diffusion。”**统一接口与细节质量、速度、训练难度是不同轴。
42. **“benchmark 第一就是任何产品场景第一。”**实际输入分布、延迟、安全和评测协议都可能不同。

<a id="sec-27"></a>
## 27. 公式与 shape 一页卡

| 内容 | 公式/shape | 读法 |
|---|---|---|
| 余弦 | `u·v/(||u||||v||)` | 方向相似度 |
| CLIP cosine | `C=IT^T [N,N]` | N图对N文的余弦相似度 |
| CLIP logits | `Z=C/tau [N,N]` | 只在这里除一次温度 |
| 行 softmax | `exp(Zij)/sum_j exp(Zij)` | 固定图，文字中选 |
| 列 softmax | `exp(Zij)/sum_i exp(Zij)` | 固定文，图片中选 |
| CLIP loss | `(L_i2t+L_t2i)/2` | 双向平均 |
| SigLIP | `-ln sigma(yij zij)` | `y=+1/-1` |
| projector | `[B,P,dv]@[dv,dl]->[B,P,dl]` | 视觉宽度变LM宽度 |
| ViT patch数 | `(H/p)×(W/p)` | 高宽分别除patch边 |
| 2×2 merge | patch数除4 | 两轴各减半 |
| 视频帧数 | `seconds×fps` | 单位消掉秒 |
| sqrt loss | `(1/B)sum_i S_i/sqrt(L_i)` | 长样本权重约sqrt长度 |
| VQ 最近码 | `argmin_k ||z-e_k||²` | 选距离最小code |
| code位数 | `log2(K)` | K个code需几bit |

每个 shape 的顺序必须在本节自己声明。本文统一视觉 features `[B,P,d]`；论文/代码可能用 `[P,B,d]`，转置后语义等价，但不能混着相乘。

<a id="sec-28"></a>
## 28. 自测题（80题）

### A. 地基与 CLIP（1–25）

1. 【判断解释】omni model 的 understanding 与 generation 有什么不同？
2. 【分类】图片 encoder 输出 `[0.1,0.7]` 是连续还是离散表示？code ID `17` 呢？
3. 【shape】4个 patch、每个宽3，写视觉 tensor shape。
4. 【手算】`u=[3,4]` 的 L2 长度是多少？
5. 【手算】`u=[3,4]`,`v=[6,8]` 的余弦相似度是多少？
6. 【判断解释】把 `u,v` 都乘100，余弦会变吗？
7. 【shape+手算】给定归一化 `I=[[1,0],[0,1]]`、`T=[[1,0],[0.6,0.8]]` 与 `tau=1`，先写 `T^T`，逐格算 `C=IT^T`，再算两行、两列正确pair的softmax概率与双向平均CE。
8. 【填表】`C[i,j]` 的 `i` 和 `j` 分别指什么？若温度是 $`\tau`$，logits `Z` 与 `C` 有什么关系？softmax 应作用在哪个矩阵？
9. 【手算】N=3 时正 pair 与非对角负 pair 各多少？
10. 【手算】N=4 时矩阵总格、正格、负格各多少？
11. 【手算】相似度 `[0.8,0.4]`，`tau=.1` 时 logits 是什么？
12. 【判断解释】`logit=s/tau` 口径下，tau 从1降到.1会更尖还是更平？
13. 【手算】softmax logits `[0,0]`，两个概率各多少？正确项 CE 多少？
14. 【手算】§6 图0行的分母为何是11.107？
15. 【手算】§6 图1正确概率与 loss 是多少？
16. 【手算】§6 三行 loss 平均是多少？
17. 【手算】§6 文2列的正确概率是多少？
18. 【判断解释】为什么图找文正确不保证文找图也同样容易？
19. 【手算】500K×20K 的全满理论上限是多少？为什么它既不是已知实际候选量，也不是最终样本数？
20. 【手算】原图672×336中心裁成336×336，横向共剪多少像素？
21. 【手算】ViT-L/14@336 每边/总 patch 数是多少？
22. 【手算+解释】OpenAI CLIP ViT-L/14@336 的 vision Transformer 输入为什么是577，而 patch 数为什么仍是576？模型取哪一个 token 作整图表示？
23. 【设计】用“猫/狗”写出 CLIP zero-shot 分类四步。
24. 【错误诊断】“zero-shot 表示预训练从未见过猫。”错在哪里？
25. 【判断解释】同 batch 两张狗图为何可能形成 false negative？

### B. SigLIP 与 projector（26–40）

26. 【填表】SigLIP 对角/非对角标签分别是什么？
27. 【手算】正 pair `y=+1,z=2` 时 `yz` 与 loss 各多少？
28. 【手算】负 pair `y=-1,z=-1` 时 `yz` 与 loss 各多少？
29. 【手算】负 pair `y=-1,z=0` 的预测置信和 loss 是多少？
30. 【手算】§9 四格 loss 总和、每pair mean、论文除N口径分别多少？
31. 【判断解释】两个分母口径为何不代表结果矛盾？“只差固定N倍”需要什么条件？SigLIP的learnable scale与bias分别做什么？
32. 【判断解释】SigLIP 为什么降低跨设备全局 softmax 耦合？
33. 【错误诊断】“SigLIP 只有N次pair计算。”请用N=4反驳。
34. 【判断解释】为什么不能用 TPU device-days 直接当 FLOPs？
35. 【shape】`Xv[2,4,3]` 乘 `W[3,6]` 后 shape？
36. 【手算】上一题 W 无bias/有bias参数量各多少？
37. 【shape】视觉 `[2,4,6]` 与文字 `[2,5,6]` 沿序列拼后 shape？
38. 【错误诊断】视觉 `[2,4,3]` 能否直接和文字 `[2,5,6]` 拼？为什么？
39. 【判断解释】冻结 vision encoder 时 forward 还运行吗？
40. 【设计】LLaVA stage1 与 stage2 各训练哪些模块？

### C. AnyRes 与 Qwen（41–62）

41. 【手算】AnyRes tile 是384×384、patch边14且无额外padding：每边完整patch数、每tile token数各多少？6个局部tile再加1个global共多少token？
42. 【手算】课程单图最大 `(1+9)×729` 等于多少？
43. 【手算】12张图每张729是多少？
44. 【手算】32帧每帧196是多少？
45. 【判断解释】三者为什么不必完全相等？
46. 【手算】3.2M数据的20.1%约多少条？
47. 【手算】31.2%+43.0%+25.9%=多少？为什么可超过100%一点？
48. 【shape】Qwen-VL 256 query 对 `[B,P,d]` 视觉 features 做cross-attention，输出视觉序列长度多少？
49. 【判断解释】这256个输出是否一定一一对应256个原patch？
50. 【填表】Qwen-VL三个阶段分别冻结/训练谁？再写Qwen2-VL的Qwen2/DFN初始化与三阶段课程口径。
51. 【手算】224/14 每边patch数？总patch数？
52. 【手算】再做2×2 merge，网格token数？
53. 【手算+解释】上一题的64个内容token加哪些边界后成为进入LM的66个token？
54. 【手算】10秒视频、2fps，共抽几帧？
55. 【手算】Qwen2-VL 对10秒视频按2fps取20帧，temporal patch size=2；每个tube有64个空间内容token。算tube数、内容token数、加视觉边界后的LM序列长度，并与16384比较。
56. 【填表】给2帧2×2网格的frame1右下写 `(t,h,w)`。
57. 【判断解释】MRoPE 比一维位置多告诉模型什么？
58. 【判断解释】interleaved MRoPE 是否打乱输入token顺序？
59. 【设计】显式时间戳怎样放进一个两帧序列？
60. 【判断解释】DeepStack 与只注入vision最后层的区别？
61. 【手算】两条样本长度4和16、每token loss=1，全token mean中各贡献分子百分比？
62. 【手算】同例平方根归一后两条贡献值与比值？

### D. Chameleon、综合与设计（63–80）

63. 【判断解释】`235B-A22B` 中两个数各表示什么？
64. 【shape】codebook有8192个、每个宽D，写shape。
65. 【手算】§21中z到e0/e1/e2的平方距离各多少，选哪个code？
66. 【手算】`32×32` 等于多少个图像token？
67. 【手算】8192为何对应13bit/token？
68. 【手算】1024个13bit索引共多少bit、多少byte？
69. 【手算】512×512 RGB、每通道8bit共多少bit？
70. 【判断解释】题68与69之比为何不能叫无损压缩比？
71. 【流程填表】按顺序写 VQ-VAE 的 encoder、nearest code、decoder、loss。
72. 【错误诊断】“新BPE负责把像素量化成code。”错在哪里？
73. 【手算】2.9T+1.5T+400B 裸换单位后相加是多少T？为何正文不报告为确定总量？
74. 【判断解释】text低熵、image高熵可能怎样影响共享训练？
75. 【判断解释】QK norm 与 z-loss 分别控制什么？
76. 【设计】要做发票OCR问答，选择center-crop还是AnyRes，并写理由和代价。
77. 【设计】要由文字生成高细节海报，为什么“CLIP projector+LM”不够？给两条输出路线。
78. 【错误诊断】benchmark表中Qwen3一列最高，能否推出它在你的医疗视频产品最好？列三项缺失条件。
79. 【综合shape】`B=1,P=64,dv=8,dl=16,T=10`，projector后、拼文字后、LM logits的shape分别是什么（词表V=50000）？
80. 【综合设计】给一个“图片+文字→文字+图片”的omni系统画模块链，并指出至少三处有损/失败边界。

<a id="sec-29"></a>
## 29. 自测答案（1–80）

### A. 1–25

1. understanding 是把输入模态变成可用于推理的表示；generation 是把模型输出还原为目标模态。前者可只留语义，后者还需细节。
2. `[0.1,0.7]` 是连续实数向量；ID 17 是从有限表选出的离散表示。
3. `[4,3]`：4个token，每个3个数。
4. `sqrt(3²+4²)=sqrt(9+16)=sqrt25=5`。
5. 点积 `3×6+4×8=50`；长度5和10；`50/(5×10)=1`。
6. 不变。点积放大 `100²`，两个长度之积也放大 `100²`，相除抵消。
7. $`T^T=[[1,0.6],[0,0.8]]`$。四格：$`C_{00}=1\times1+0\times0=1`$，$`C_{01}=1\times0.6+0\times0.8=0.6`$，$`C_{10}=0\times1+1\times0=0`$，$`C_{11}=0\times0.6+1\times0.8=0.8`$，所以 $`C=[[1,0.6],[0,0.8]]`$，shape `[2,2]@[2,2]=[2,2]`。$`\tau=1`$ 时 $`Z=C`$。两行正确概率为 `e¹/(e¹+e^.6)=.5987`、`e^.8/(e⁰+e^.8)=.6900`，CE均值 `(.5130+.3711)/2=.4421`；两列正确概率为 `e¹/(e¹+e⁰)=.7311`、`e^.8/(e^.6+e^.8)=.5498`，CE均值 `(.3133+.5981)/2=.4557`；双向平均 `(.4421+.4557)/2=.4489`。
8. `i` 是图像编号；`j` 是文字编号；`C[i,j]` 是该图与该文的余弦相似度。`Z=C/tau`；按行/列 softmax 都只作用于 `Z`，不能再除一次温度。
9. 总格9；对角正格3；负格 `9-3=6`。
10. 总格 `4²=16`；正格4；负格 `16-4=12`。
11. `[0.8/.1,0.4/.1]=[8,4]`。
12. 更尖，因为分数差从0.4放大到4。
13. `e⁰=e⁰=1`，分母2，概率各1/2；CE `-ln(0.5)=0.6931`。
14. `e²+e¹+e⁰=7.389+2.718+1=11.107`。
15. 概率 `7.389/(1+7.389+1)=7.389/9.389=0.7870`；loss `-ln(.7870)=0.2395`。
16. `(0.4076+0.2395+0.4076)/3=1.0547/3=0.3516`。
17. 文2列 `[0,0,2]`，概率 `7.389/(1+1+7.389)=7.389/9.389=0.7870`。
18. 一张图可能对应多段近义描述，一段文字也可能对应多张相似图；行/列候选分布不同。
19. `500,000×20,000=10,000,000,000=10B`，但这是每个 query 都装满20K时的理论上限；实际 query 可未满、结果可重复，所以连实际候选量也不能由裸乘法推出。抓取失败、过滤、去重还可能继续减少，最终论文报告训练集约400M。
20. `672-336=336` 像素；若居中且对称，左右各约168。
21. 每边 `336/14=24`；总 `24×24=576`。
22. patch 数是 `(336/14)^2=24^2=576`；OpenAI CLIP ViT 再在前面加一个 class token，所以 Transformer 输入 `576+1=577`。模型取 token0/class token 的最后表示作整图 embedding；文本 BOS/EOS 是另一条 encoder 序列。
23. 写“a photo of a cat/dog”→文本编码；新图编码；分别算余弦；取较高类别。
24. zero-shot 只说下游不再用该任务标签训练；预训练互联网数据可能已有猫图和文字。
25. 两图都是真狗、caption都近义；数据索引只认对角为正，因此一个狗pair会被另一个狗pair当负。

### B. 26–40

26. 对角 `+1`，非对角 `-1`。
27. `yz=1×2=2`；`sigma(2)=0.8808`；loss `-ln(.8808)=0.1269`。
28. `yz=(-1)×(-1)=1`；loss `-ln sigma(1)=0.3133`。
29. `yz=0`，`sigma(0)=.5`，表示不确定；loss `-ln(0.5)=0.6931`。
30. 总和 `0.1269+0.3133+0.6931+0.3133=1.4466`；pair mean `1.4466/4=.3617`；论文口径 `1.4466/N=1.4466/2=.7233`。
31. 它们对相同逐格项求和；当batch size `N` 固定时，除N与除N²只差N倍，学习率/报告口径必须配套。若N改变，倍率也变。learnable正scale控制相似度差距的放大程度；bias把所有pair logits平移，调节匹配先验。
32. 每格 `-ln sigma(yz)` 不需要整行/列的全局分母；可以分块算 pair 后求和。
33. N=4 有 `4²=16` 格，不是4格；4正、12负。
34. TPUv3/v4 单芯片能力、实际利用率、训练数据/步数、软件不同；device-days 缺少这些乘数。
35. `[2,4,3]@[3,6] -> [2,4,6]`，batch和token轴保留，最后宽度变6。
36. 无bias `3×6=18`；有bias再加每个输出宽一个，共 `18+6=24`。
37. token轴 `4+5=9`，得到 `[2,9,6]`。
38. 不能直接沿序列拼，因为最后宽度3与6不相同；先project 3→6。
39. 运行。冻结只阻止参数更新；不forward就没有视觉features。
40. stage1 冻结视觉和LM，只训W；stage2 仍冻视觉，训练W与LM。

### C. 41–62

41. 每边 `384//14=27` 个完整patch；每tile `27²=729` token；总计 `(6+1)×729=7×729=5103`。
42. `10×729=7290`。
43. `12×729=8748`。
44. `32×196=6272`。
45. 单图重清晰度，多图重图片数，视频重帧数；只是把总长度控制在相近量级。
46. `3,200,000×.201=643,200`，约0.6432M。
47. `31.2+43.0+25.9=100.1%`；各栏四舍五入造成0.1百分点差。
48. 256。输出是 `[B,256,d]`（忽略后续宽度映射记号）。
49. 不一定。256个learned query各自从全部patch聚合信息，不是一格绑定一patch。
50. Qwen-VL：stage1 LM冻、vision+adaptor训；stage2全训；stage3 vision冻、adaptor+LM训。Qwen2-VL：LM从Qwen2初始化、vision约675M并从DFN路线初始化；stage1只训visual encoder，stage2全训，stage3在instruction数据上训练LM（其他模块是否forward与具体冻结以实现为准）。
51. 每边 `224/14=16`；总 `16×16=256`。
52. 两轴各除2：`8×8=64`，也可 `256/4=64`。
53. `64+vision_start+vision_end=66`。64 是 merge 后视觉内容 token；66 是加两个视觉边界后进入 LM 的序列长度。
54. `10s×2 frame/s=20 frames`。
55. temporal tubes `=20/2=10`；内容 token `=10×64=640`；加 `vision_start/vision_end` 后 `640+2=642`；`642<16,384`。若写1280，就是误把每帧独立编码、漏了 temporal patch size 2。
56. frame1右下 `(1,1,1)`。
57. 额外告诉时间帧、网格行、网格列，不必从一维序号猜换行/换帧。
58. 否。它交错的是RoPE特征维度分给t/w/h的顺序，不改输入token排列。
59. 例如 `<0.0 seconds> frame0_tokens <0.5 seconds> frame1_tokens`；时间是文字token。
60. 单层只送最终视觉features一次；DeepStack把不同视觉层features送进多个LM层。
61. 分子总20；短 `4/20=20%`，长 `16/20=80%`。
62. 短 `4/sqrt4=2`；长 `16/sqrt16=4`；长短比 `4:2=2:1`。

### D. 63–80

63. 235B是总参数数量级；A22B是每token激活参数数量级。存储不能只按22B。
64. `[8192,D]`：8192行code，每个D维。
65. 到e0：`.8²+.1²=.65`；e1：`(-.2)²+.1²=.05`；e2：`.8²+(-.9)²=1.45`；最小`.05`，选code1。
66. `32×32=1024`。
67. `2¹³=8192`，13个二进制格可编码0到8191。
68. `1024×13=13,312 bit`；`13,312/8=1,664 byte`。
69. `512×512=262,144`像素；乘3通道=`786,432`；乘8=`6,291,456 bit`。
70. VQ 是有损重建；1664字节还不含codebook/decoder/容器，不能与原RGB当无损文件一一恢复比较。
71. `x -> encoder -> continuous z -> nearest code index -> code vector -> decoder -> x_hat`；用重建误差等训练。
72. 图像encoder/VQ量化器把像素变code；BPE负责文字片段与词表，虽可统一ID空间却不做视觉最近邻。
73. `400B=.4T`，裸和 `2.9+1.5+.4=4.8T`；但集合是否重叠、采样/计数是否同口径未知，故不能报确定唯一总量。
74. 例如文字真实项概率0.9时NLL `=-ln(.9)=.1053`；8个图code等分时NLL `=-ln(1/8)=ln8=2.0794`。共享参数/softmax时不同loss、频率和梯度统计可能竞争；Chameleon观察到norm growth/logit drift，但“高熵”本身不必然导致发散。
75. QK norm归一化attention的query/key，控制点积分数尺度。z-loss先取 $`Z_{part}=\sum_v e^{z_v}`$，惩罚 $`(\log Z_{part})^2`$，抑制整组logits尺度漂大；二者都不保证语义正确。
76. 选AnyRes：发票小字/表格会被center crop和缩小损坏；代价是更多token、显存、计算和长度不齐。
77. CLIP projector只提供理解features且LM吐文字token。可选：(a) LM条件+diffusion；(b) VQ离散图token+自回归decoder/Chameleon式。
78. 不能。至少缺医疗视频分布与标注、同一解码/工具/分辨率协议、延迟与安全/隐私要求；也可能缺统计不确定性。
79. projector后 `[1,64,16]`；拼10文字后 `[1,74,16]`；LM logits `[1,74,50000]`。
80. 一例：`image -> vision encoder/projector -> shared LM <- text tokenizer`；LM生成文字token，并输出图条件给diffusion，或输出VQ code给image decoder。失败边界至少：视觉crop/量化丢信息；图文训练数据噪声/幻觉；token预算截断视频；decoder重建失真；多模态loss失衡。任写三项且方向正确即可。

<a id="sec-30"></a>
## 30. 视频导航：中文主题 + 人工字幕证据

以下均来自人工英文字幕轨 `en-US`。英文证据只保留很短的定位片段；中文列说明这一段真正解决的问题，而不是机械翻译残句。全部秒点唯一。

### 30.1 开场到 CLIP 总结（00:05–22:20）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=5s) | 课程临时由RL改讲多模态的背景 | “plan was to talk more” |
| [00:35](https://www.youtube.com/watch?v=26FtD08ZpOU&t=35s) | 前16讲只处理语言模型 | “exclusively … language models” |
| [01:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=65s) | omni作为长期目标 | “North Star” |
| [01:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=91s) | 模态之间可相互转换 | “audio into an image” |
| [02:02](https://www.youtube.com/watch?v=26FtD08ZpOU&t=122s) | 为什么仍以Transformer为核心 | “transformers work really well” |
| [02:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=150s) | token概念扩展到非文字 | “expanding the notion” |
| [02:58](https://www.youtube.com/watch?v=26FtD08ZpOU&t=178s) | 像素不天然是语义token | “a pixel is certainly not” |
| [03:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=210s) | 文字BPE tokenizer虽不完美，但已有可用离散接口；非文本还需另找对应接口 | “We had a BPE tokenizer … one could wish for a better tokenizer … for non-text modalities … figure out the equivalent” |
| [04:01](https://www.youtube.com/watch?v=26FtD08ZpOU&t=241s) | 非文本输入与输出的两个问题 | “non-text data” |
| [04:25](https://www.youtube.com/watch?v=26FtD08ZpOU&t=265s) | CLIP正式登场 | “CLIP model” |
| [04:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=295s) | 从标注视觉模型走向foundation model | “foundation model era” |
| [05:25](https://www.youtube.com/watch?v=26FtD08ZpOU&t=325s) | 用海量图文网页替代昂贵人工类标 | “large amount of image” |
| [05:53](https://www.youtube.com/watch?v=26FtD08ZpOU&t=353s) | CLIP核心思想很简单 | “idea of CLIP” |
| [06:20](https://www.youtube.com/watch?v=26FtD08ZpOU&t=380s) | 大batch产生许多图像embedding | “32,000 … image encodings” |
| [06:52](https://www.youtube.com/watch?v=26FtD08ZpOU&t=412s) | 正确图文点积应高 | “want this alignment” |
| [07:22](https://www.youtube.com/watch?v=26FtD08ZpOU&t=442s) | N行+N列共2N个分类问题 | “2 times N … softmax” |
| [07:48](https://www.youtube.com/watch?v=26FtD08ZpOU&t=468s) | 温度缩放相似度 | “some temperature” |
| [08:15](https://www.youtube.com/watch?v=26FtD08ZpOU&t=495s) | 转入CLIP构建细节 | “more details” |
| [08:43](https://www.youtube.com/watch?v=26FtD08ZpOU&t=523s) | 查询词抓取图文候选 | “mine … image and text pairs” |
| [09:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=556s) | 原始网页数据还需处理 | “bunch of processing” |
| [09:45](https://www.youtube.com/watch?v=26FtD08ZpOU&t=585s) | 互联网图片尺寸任意 | “internet image” |
| [10:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=616s) | 短边缩到目标336 | “target size” |
| [10:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=642s) | center crop依赖主体居中的假设 | “object is in the middle” |
| [11:24](https://www.youtube.com/watch?v=26FtD08ZpOU&t=684s) | 问答：为何用图文对而非纯图片 | “why … image-text pairs” |
| [11:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=701s) | SimCLR式数据增强自监督对照 | “data augmentation” |
| [12:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=729s) | 增强不应改变狗的细粒度类别 | “one type of dog” |
| [12:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=761s) | CLIP比较过多种视觉encoder | “different ones” |
| [13:10](https://www.youtube.com/watch?v=26FtD08ZpOU&t=790s) | ViT patch尺寸背景 | “16 by 16” |
| [13:36](https://www.youtube.com/watch?v=26FtD08ZpOU&t=816s) | patch序列进入标准Transformer | “standard transformer” |
| [14:07](https://www.youtube.com/watch?v=26FtD08ZpOU&t=847s) | 全局汇聚patch向量 | “average all these vectors” |
| [14:37](https://www.youtube.com/watch?v=26FtD08ZpOU&t=877s) | ViT 用 patch token 编码整张图的结构小结 | “that’s the VIT” |
| [15:03](https://www.youtube.com/watch?v=26FtD08ZpOU&t=903s) | patch仍含RGB三通道 | “three channels” |
| [15:33](https://www.youtube.com/watch?v=26FtD08ZpOU&t=933s) | 课堂问答延伸任务边界 | “task only has” |
| [16:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=965s) | 分类导向会影响位置设计取舍 | “classification in mind” |
| [16:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=990s) | 文本侧使用GPT-2式encoder | “GPT-2 style transformer” |
| [17:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1020s) | 两侧encoder联合训练 | “process to train” |
| [17:28](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1048s) | ImageNet zero-shot headline | “outperform a ResNet” |
| [17:57](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1077s) | 复用已有数据的价值 | “leverage … existing data” |
| [18:28](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1108s) | zero-shot预测流程 | “zero shot prediction” |
| [19:02](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1142s) | 问答：不同狗caption会互相干扰 | “other captions … dogs” |
| [19:23](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1163s) | 网页图像数据自身有噪声 | “images are basically” |
| [19:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1195s) | 大规模噪声训练的经验现象 | “surprising but … interesting” |
| [20:24](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1224s) | 对照目标：从图逐字预测文本 | “predict a text” |
| [20:49](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1249s) | 生成/BoW目标比CLIP排名低效 | “less efficient” |
| [21:20](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1280s) | 图像表示捕捉caption语义 | “of the image” |
| [21:50](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1310s) | 分类导向表示不够细粒度 | “not very fine-grained” |
| [22:20](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1340s) | CLIP全局softmax不易拆分 | “not … decomposable” |

### 30.2 SigLIP、LLaVA 与 OneVision（22:49–45:33）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [22:49](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1369s) | 从CLIP切换到SigLIP | “of CLIP” |
| [23:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1396s) | 每个pair单独二分类 | “any given image text pair” |
| [23:44](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1424s) | 非配对标签取负一 | “labels … minus 1” |
| [24:17](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1457s) | 问答：负pair如何采样 | “question is” |
| [24:45](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1485s) | 采样要避免系统偏置 | “not biased” |
| [25:14](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1514s) | SigLIP数据处理还有额外步骤 | “extra work” |
| [25:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1542s) | 课程给出TPU训练时间对照 | “five days on 32v4s” |
| [26:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1569s) | 为什么pair loss更易并行 | “way that this is faster” |
| [26:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1601s) | 每设备先存图文子集 | “each device stores a subset” |
| [27:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1629s) | 本地先算pair，再轮转数据块 | “pairs it has locally” |
| [27:38](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1658s) | SigLIP让batch和loss耦合变弱 | “nice about SigLIP” |
| [28:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1685s) | batch过小仍会退化 | “too small a batch size” |
| [28:37](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1717s) | 图像encoder阶段小结 | “CLIP and SigLIP” |
| [29:03](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1743s) | 接下来比较LLaVA与Qwen家族 | “two families” |
| [29:33](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1773s) | 复用已有视觉与语言模型再拼接 | “stitch it together” |
| [30:03](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1803s) | Vicuna能力背景并非GPT-4等价 | “wasn’t as good as GPT-4” |
| [30:33](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1833s) | ShareGPT对话数据来源 | “conversations … with ChatGPT” |
| [31:01](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1861s) | LLaVA合成instruction数据集 | “synthesized a data set” |
| [31:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1891s) | 用caption/box提示GPT-4写对话 | “generate me a conversation” |
| [32:03](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1923s) | 合成文本与原图配回后训练 | “trained a model” |
| [32:28](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1948s) | 原图先过CLIP vision encoder | “vision encoder CLIP” |
| [32:58](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1978s) | 图与文字都变为LM宽度向量 | “into these vectors” |
| [33:24](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2004s) | LLaVA两阶段训练 | “two stages” |
| [33:54](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2034s) | alignment阶段训练W，把视觉输出映射到语言embedding空间；随机W还没有这种对齐 | “mapped into the same space … if you just use a random W … they're not embeddings” |
| [34:26](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2066s) | 图像与对话/描述共同作为样本 | “conversation or a description” |
| [34:54](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2094s) | 车辆上熨衣示例的视觉描述 | “iron … back of a minivan” |
| [35:20](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2120s) | LLaVA问答过渡 | “questions about LLaVA” |
| [35:51](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2151s) | OneVision主要改进方向 | “main thing … did” |
| [36:19](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2179s) | 视频由抽样帧表示 | “sampling of the frames” |
| [36:49](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2209s) | projector由线性层升级两层MLP | “two-layer MLP” |
| [37:20](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2240s) | 低分辨率会让J/I等小字混淆 | “J looks like an I” |
| [37:46](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2266s) | AnyRes基本思路开始 | “idea is basically” |
| [38:15](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2295s) | 固定encoder不能直接吃超大分辨率 | “can’t handle high resolution” |
| [38:46](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2326s) | Transformer可处理拼接后的token | “transformer … handles that” |
| [39:15](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2355s) | 大图切成多个局部块 | “break it up” |
| [39:43](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2383s) | 统一考虑单图/多图/视频长度 | “all of these” |
| [40:11](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2411s) | 额外保留全局下采样图 | “full image downsampled” |
| [40:43](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2443s) | 视频每帧用更少token | “fewer tokens … each frame” |
| [41:11](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2471s) | OneVision数据策展的性质 | “data here” |
| [41:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2500s) | 表格问答等专项数据 | “questions about tables” |
| [42:07](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2527s) | 没有人工作标预算时依赖合成 | “don’t have an annotation budget” |
| [42:36](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2556s) | OneVision训练承接两阶段思路 | “roughly the same idea” |
| [43:06](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2586s) | 实际扩为三个训练阶段 | “three stages” |
| [43:34](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2614s) | 观察到跨模态场景迁移 | “transfer between modalities” |
| [44:04](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2644s) | 多图chart/table联合推理示例 | “one for the chart” |
| [44:32](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2672s) | GUI多截图关系推理示例 | “screenshots” |
| [45:02](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2702s) | 视频多帧跟踪圈选球员 | “across multiple frames” |
| [45:33](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2733s) | 标准VLM模板再次归纳 | “vision encoder plus … projector” |

### 30.3 Qwen-VL、Qwen2-VL、Qwen3-VL（46:06–67:13）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [46:06](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2766s) | Qwen多模态系列从2023起步 | “Qwen started … in 2023” |
| [46:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2791s) | adaptor使用一层cross-attention | “one layer of cross-attention” |
| [46:58](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2818s) | 幻灯截图有被裁掉的信息 | “got cut off” |
| [47:26](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2846s) | stage1涉及从头视觉预训练 | “pre-training from scratch” |
| [47:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2875s) | 清理后约1.4B图文例 | “1.4 billion examples” |
| [48:25](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2905s) | stage2连语言模型一起训练 | “language model” |
| [48:56](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2936s) | 视觉定位/人物识别能力示例 | “Spider-Man and the Hulk” |
| [49:22](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2962s) | Qwen2带来若干新设计 | “some new ideas” |
| [49:54](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2994s) | 动态分辨率可产生约11K token | “11,000 tokens” |
| [50:18](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3018s) | 224块、ViT与merge口径开始 | “224 by 224 patch” |
| [50:50](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3050s) | 视频2fps与最大token预算 | “2 frames a second” |
| [51:17](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3077s) | RoPE位置编码直觉 | “idea behind RoPE” |
| [51:47](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3107s) | 每个视觉位置用三元组 | “each patch … a triple” |
| [52:17](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3137s) | Qwen2初始化语言/视觉模块 | “initialize the language model” |
| [52:46](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3166s) | 课程列举Qwen2-VL报告中的视频理解、数学、代码与function-calling能力 | “some video understanding … math and code, function calling, and so on” |
| [53:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3196s) | 转入Qwen3关键变化 | “point out” |
| [53:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3222s) | 256K上下文对长视频重要 | “long video” |
| [54:13](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3253s) | MRoPE轴如何分给旋转维度 | “allocate” |
| [54:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3282s) | 高频旋转维度的含义 | “high frequency” |
| [55:19](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3319s) | 对比Qwen2的隐式时间位置与Qwen3随后引入的显式时间token | “before, the time stamp was implicit in the positional encodings” |
| [55:39](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3339s) | 显式“0 seconds”文字token | “tokens zero seconds” |
| [56:10](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3370s) | 长短多模态样本权重问题 | “some examples” |
| [56:35](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3395s) | 用长度平方根降低长样本支配 | “square root of the length” |
| [57:08](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3428s) | DeepStack比普通注入更复杂 | “more sophisticated” |
| [57:34](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3454s) | 多层视觉语言深融合 | “deep fusion” |
| [58:04](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3484s) | 四阶段长上下文预训练pipeline | “pipelines” |
| [58:34](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3514s) | SFT、蒸馏、RL的post-training | “SFT … knowledge distillation” |
| [59:01](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3541s) | 报告与闭源模型做benchmark对比 | “closed models” |
| [59:11](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3551s) | 表格粗体表示该行最好 | “bold means … best” |
| [59:54](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3594s) | 问答：这些VLM能否生成视频 | “what about video generation” |
| [60:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3631s) | 描述质量主要由数据监督 | “description is working” |
| [60:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3641s) | 问答：多模态系统训练是否更难 | “harder … system perspective” |
| [61:27](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3687s) | 讨论多模态数据规模变大 | “data sets are larger” |
| [61:56](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3716s) | 系统问题的第二部分 | “second part” |
| [62:24](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3744s) | 多模态数据mixture与先前课程连接 | “data mixtures” |
| [62:48](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3768s) | 讲者不认为训练中多模态token数量必然远超文本token | “I wouldn't say that the number of multimodal tokens vastly outnumbers text tokens” |
| [63:23](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3803s) | 对齐时是否从预训练LM初始化 | “do this alignment” |
| [63:54](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3834s) | alignment时冻结预训练LM，只训练adaptor，并按67B token预算把视觉encoder接到LM | “the language model is frozen … training the adapter … pick a token budget, 67 billion tokens … just train” |
| [64:32](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3872s) | 问答：vision encoder参数规模 | “number … vision encoder” |
| [64:49](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3889s) | encoder不必和LM同规模 | “reason … vision encoder” |
| [65:19](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3919s) | 回看架构图定位参数 | “up here” |
| [65:49](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3949s) | projector相比两大模块很小 | “projector is much smaller” |
| [66:18](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3978s) | SOTA表述的玩笑与时效性 | “SOTA” |
| [66:45](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4005s) | 技术报告包含更多细节 | “more details” |
| [67:13](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4033s) | 理解型VLM结束，准备转生成 | “image processing” |

### 30.4 Chameleon 与课程总结（67:44–77:25）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [67:44](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4064s) | 先前理解型VLM不能直接生成图 | “can’t generate images” |
| [68:14](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4094s) | 用语言模型视角引出统一token | “language person” |
| [68:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4122s) | 图文可任意交错排列 | “flip … language and text” |
| [69:14](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4154s) | 图片也要变得像文字token | “like text” |
| [69:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4181s) | VQ将图映射到离散码 | “map an image” |
| [70:11](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4211s) | 图像先形成latent patch网格 | “patches” |
| [70:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4240s) | 最近码硬选择不可直接求导 | “not differentiable” |
| [71:06](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4266s) | 图片最终表示为code序列 | “images … these codes” |
| [71:36](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4296s) | 图文统一后主体就是语言模型 | “just a language model” |
| [72:04](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4324s) | 离散统一方案很简洁 | “great and elegant” |
| [72:36](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4356s) | 文字next-token比图像token低熵 | “predicting the next word” |
| [73:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4385s) | 训练不稳定有缓解方法 | “mitigate or fix” |
| [73:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4411s) | 离散图像生成的实际性能代价 | “downside” |
| [74:02](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4442s) | OCR等细节损失在图像更严重 | “more exacerbated here” |
| [74:32](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4472s) | 扩散模型成为主流生成路线 | “popular … for generation” |
| [75:01](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4501s) | frontier模型宣传全模态能力 | “GPT comes out” |
| [75:31](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4531s) | 最终仍回到非文本编码挑战 | “fundamental challenge” |
| [75:57](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4557s) | 连续视觉向量可做得较紧凑 | “vectors … fairly small” |
| [76:26](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4586s) | 图像/视频/文字训练权重需平衡 | “weighing them properly” |
| [76:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4615s) | 连续encoder加生成器仍是常用途径 | “go-to way” |
| [77:25](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4645s) | 课程结束并鼓励亲自试模型 | “if you’re curious” |

<a id="sec-31"></a>
## 31. 官方源码 1–302 行连续覆盖

覆盖表是“行段映射无 gap/overlap”，不是声称每个空行都逐字讲解。版本文件 SHA256：`EDF08D34E25EB03D519B8D0C2B17DF261076BC03A2907B33C673D2B3F7B7660F`。

| 行范围 | 源码内容 | 笔记映射 |
|---:|---|---|
| 1–3 | `edtrace` / lecture helper imports | §31.1 |
| 4–45 | `main()`：omni、两问题、函数调用、总结 | §3–§4、§23–§25 |
| 46–47 | 函数间空行 | 本覆盖说明 |
| 48–95 | `clip()`：目标、数据、处理、ViT、结果、ablation | §5–§8 |
| 96–97 | 函数间空行 | 本覆盖说明 |
| 98–121 | `siglip()`：二分类、WebLI、并行、batch | §9 |
| 122–123 | 函数间空行 | 本覆盖说明 |
| 124–145 | `llava()`：数据、结构、两阶段 | §10–§11 |
| 146–147 | 函数间空行 | 本覆盖说明 |
| 148–192 | `llava_onevision()`：AnyRes、三模态、数据、迁移 | §12–§13 |
| 193–194 | 函数间空行 | 本覆盖说明 |
| 195–211 | `qwen_vl()`：结构、三阶段、示例 | §15 |
| 212–213 | 函数间空行 | 本覆盖说明 |
| 214–233 | `qwen2_vl()`：动态分辨率、MRoPE、训练 | §16–§17 |
| 234–235 | 函数间空行 | 本覆盖说明 |
| 236–265 | `qwen3_vl()`：模型、interleaved MRoPE、DeepStack、训练 | §18–§20 |
| 266–267 | 函数间空行 | 本覆盖说明 |
| 268–298 | `chameleon()`：离散图token、VQ-VAE、数据、稳定性 | §21–§23 |
| 299–300 | 尾部空行 | 本覆盖说明 |
| 301–302 | 直接运行时调用 `main()` | §31.1 |

### 31.1 讲义运行基础设施

`text/image/link` 是课程展示工具：把文字、图片和链接画到 executable lecture 页面；`article_link/post_link` 是链接样式 helper。`if __name__ == "__main__": main()` 表示直接执行该文件时从 `main` 开始。它们不属于多模态模型算法。

源码共 20 次外部链接调用，其中 OpenCLIP 论文被复用两次；目标依次覆盖 CLIP、OpenCLIP、CLIP preprocess 源码、ViT、SigLIP、WebLI、LLaVA、Vicuna、OneVision、AnyRes、Qwen-VL、Qwen2-VL、DFN、Qwen3-VL、SigLIP2、DeepStack、Chameleon、图tokenizer、VQ-VAE。§33 给出主线直链；未把博客案例升级成通用事实。

<a id="sec-32"></a>
## 32. 32 张本地图：原分辨率视觉核验记录

路径均为课程仓库 `images/`。32 次 `image(...)` 调用恰好对应 32 个唯一资产；无重复调用。以下不是凭文件名猜图义，而是实际打开原图后记录。

| # | 文件（像素） | 实际看到的内容 | 对应正文 |
|---:|---|---|---|
| 1 | `multimodality.png` 1536×1024 | 文/图/音频/视频四象限 | §3 |
| 2 | `clip.png` 2073×760 | 图文对比矩阵、prompt类别、zero-shot预测 | §5–§7 |
| 3 | `clip-code.png` 1020×930 | 两encoder、归一化、温度、双向CE伪代码 | §5–§6 |
| 4 | `vit.png` 1558×895 | patch+class token进入Transformer encoder | §7 |
| 5 | `clip-efficiency.png` 1225×809 | accuracy对processed images，CLIP优于生成/BoW对照 | §7.6 |
| 6 | `siglip-code.png` 1519×764 | 对角+1/非对角-1、log-sigmoid loss | §9.1–§9.2 |
| 7 | `siglip-parallelism.png` 2124×693 | 三设备本地块轮转、累加loss | §9.3 |
| 8 | `llava-gen.png` 1710×1070 | caption/box提示生成对话、描述、推理 | §11.1 |
| 9 | `llava-architecture.png` 1722×585 | vision encoder→W，与instruction拼入LM | §10–§11 |
| 10 | `llava-example.png` 1383×1378 | 车辆上熨衣的多模型回答对比 | §11.3 |
| 11 | `llava-onevision.png` 1383×628 | SigLIP→2层MLP→Qwen2，单/多图/视频 | §12 |
| 12 | `llava-onevision-anyres.png` 2424×748 | 分块、双线性插值、flatten与global path | §12.1–§12.3 |
| 13 | `llava-onevision-modalities.png` 2011×754 | 单图7290、多图8748、视频6272预算 | §12.2 |
| 14 | `llava-onevision-data-1.png` 2452×1030 | 3.2M单图及五类百分比 | §13.1 |
| 15 | `llava-onevision-data-2.png` 2396×921 | 1.6M数据的single/multi/video比例 | §13.1 |
| 16 | `llava-onevision-training.png` 2153×1088 | 三stage的模块、分辨率、token、数据 | §13.2 |
| 17 | `llava-onevision-transfer-s1.png` 1823×916 | 两张表/图联合计算保险数值 | §13.3 |
| 18 | `llava-onevision-transfer-s2.png` 1485×1357 | 四张GUI截图的操作序列 | §13.3 |
| 19 | `llava-onevision-transfer-s8.png` 1038×1496 | 视频帧中持续追踪被圈球员 | §13.3 |
| 20 | `qwen-vl-stages.png` 2367×936 | 三训练阶段冻结/解冻关系 | §15.1 |
| 21 | `qwen-vl-stage1.png` 1126×666 | raw 5B→cleaned 1.4B及来源表 | §15.1 |
| 22 | `qwen-vl-stage2.png` 1902×606 | caption/VQA/grounding/OCR/text构成 | §15.1 |
| 23 | `qwen-vl-examples.png` 2029×1276 | grounding、OCR、比较、代码等示例 | §15 |
| 24 | `qwen2-vl-architecture.png` 2179×1375 | 不同图/视频分辨率映射不同token数；64内容/66含边界口径在§16拆开 | §16 |
| 25 | `qwen2-vl-mrope.png` 2061×504 | t/h/w位置ID及图后文字位置 | §17 |
| 26 | `qwen2-vl-capabilities.png` 2169×1210 | 视频、grounding、OCR、文档、数学、代码案例 | §16–§17 |
| 27 | `qwen3-vl.png` 1958×1133 | 显式时间token与DeepStack多层注入 | §18 |
| 28 | `qwen3-vl-pretraining.png` 1814×325 | 67B/1T/1T/100B四stage与长度 | §19.3 |
| 29 | `qwen3-vl-results.png` 1003×1479 | 多benchmark与闭源/开源模型快照表 | §18、§20 |
| 30 | `chameleon.png` 963×541 | 图文交错自回归、image tokenizer/detokenizer | §21 |
| 31 | `chameleon-example.png` 1110×995 | 鸟类文字与图片交错长输出 | §21 |
| 32 | `vq-vae.png` 1693×929 | encoder latent→最近code→decoder与loss | §21.1 |

视觉检查边界：图片是课程引用的论文图/截图；表中只转述可见对象、轴、数字与箭头。单个成功样例不被当作总体准确率；Qwen3结果表不被当作当前永久榜单。

<a id="sec-33"></a>
## 33. 来源、SHA、外链调用与验证边界

### 33.1 课程主来源与本地证据

- [Stanford CS336 Spring 2026 官方课程页](https://cs336.stanford.edu/)。
- [官方 `lecture_17.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)：本笔记主题以它的 **Multimodal Models** 为准。
- [Stanford Online Lecture 17 视频](https://www.youtube.com/watch?v=26FtD08ZpOU)。

| 本地证据 | 物理口径 | SHA256 |
|---|---:|---|
| `lecture_17.py` | 302 physical lines；14,192 bytes；commit `8b59b50730766695c2ffedd1a79c50cd09b9eb91` | `EDF08D34E25EB03D519B8D0C2B17DF261076BC03A2907B33C673D2B3F7B7660F` |
| `transcript_en_us.txt` | 人工 `en-US`；1,264 cues；63,818 bytes；00:05–77:29 | `96BA19B88661B635204556713697C0305AF60E27D9351FE23F994FE5492F3A94` |

源码的九个函数入口 `main / clip / siglip / llava / llava_onevision / qwen_vl / qwen2_vl / qwen3_vl / chameleon` 已全部映射到 §3–§23；其中 `main` 是总入口，后八个函数分别承担模型模块。这里说“覆盖”是行段和语义入口不漏，不是假装每个空行都需要算法解释。

### 33.2 源码 20 次外链调用审计

源码有 **20 次** `link/post_link` 调用、**19 个唯一 URL**；OpenCLIP 在 CLIP 复现和 Qwen-VL 视觉 encoder 两处复用。源码没有远程 `image(URL)` 调用：32 次图片调用全部是 §32 已打开的本地资产。

| 源码行 | 外链目标 | 正文用途与边界 |
|---:|---|---|
| 49 | [CLIP paper](https://arxiv.org/abs/2103.00020) | §5–§8；400M、对比目标、zero-shot 的一手论文 |
| 67 | [OpenCLIP](https://arxiv.org/abs/2212.07143) | §7.1；开放复现，不是原 CLIP 私有数据副本 |
| 69 | [OpenAI CLIP preprocess code](https://github.com/openai/CLIP/blob/main/clip/clip.py#L79) | §7.2；resize/crop 的官方代码入口 |
| 75 | [Vision Transformer](https://arxiv.org/pdf/2010.11929) | §7.3；patch-token 背景 |
| 99 | [SigLIP](https://arxiv.org/abs/2303.15343) | §9；pair sigmoid、batch 实验边界 |
| 107 | [WebLI](https://arxiv.org/pdf/2209.06794) | §9.4；WebLI 数据来源的一手论文 |
| 125 | [LLaVA](https://arxiv.org/abs/2304.08485) | §10–§11；projector、两阶段训练 |
| 128 | [Vicuna 官方项目博客](https://www.lmsys.org/blog/2023-03-30-vicuna/) | §11；课程所用 decoder 背景；它是官方项目说明，不是同行评审实验替代品 |
| 149 | [LLaVA-OneVision](https://arxiv.org/pdf/2408.03326) | §12–§13；AnyRes、多图/视频与 transfer |
| 161 | [LLaVA-1.5 / AnyRes paper](https://static.hliu.cc/files/llava/improved_llava.pdf) | §12；AnyRes 来源 |
| 196 | [Qwen-VL](https://arxiv.org/abs/2308.12966) | §15；固定 adaptor 与三阶段训练 |
| 199 | [OpenCLIP（第二次调用）](https://arxiv.org/abs/2212.07143) | §15；ViT-bigG/14 背景；与第 67 行是同一 URL |
| 215 | [Qwen2-VL](https://arxiv.org/abs/2409.12191) | §16–§17；dynamic resolution 与 MRoPE |
| 226 | [DFN](https://arxiv.org/abs/2309.17425) | §16；vision encoder 初始化背景 |
| 237 | [Qwen3-VL](https://arxiv.org/abs/2511.21631) | §18–§20；2025 报告、课程 2026 快照 |
| 245 | [SigLIP 2](https://arxiv.org/pdf/2502.14786) | §18；视觉 encoder 架构来源 |
| 252 | [DeepStack](https://arxiv.org/abs/2406.04334) | §18.3；跨层视觉融合来源 |
| 269 | [Chameleon](https://arxiv.org/pdf/2405.09818) | §21–§23；early fusion、训练与稳定性 |
| 279 | [Chameleon image tokenizer](https://arxiv.org/pdf/2203.13131) | §21；离散图像 tokenizer 背景 |
| 281 | [VQ-VAE](https://arxiv.org/pdf/1711.00937) | §21.1；codebook 与重建目标来源 |

### 33.3 事实层级与没有声称做过的验证

- 【课程】表示 2026 讲义/字幕的陈述；硬件、模型版本、token 数与 benchmark 表都是当时快照。
- 【补充】只用上表论文、官方代码或官方项目材料。论文自己报告的提升仍是其数据、prompt、训练预算和评测协议下的结果。
- CLIP 的 500K×20K 上限、Qwen2 的64内容/66含边界、Qwen3 的平方根权重、Chameleon 的三类数据都明确标了计数边界。
- 本地没有训练 CLIP/VLM、TPU 集群、Qwen 私有 processor 或 Chameleon 权重；数值验证仅覆盖手算、shape、源码/字幕/图片与公开资料，不声称复现论文 accuracy、吞吐或生成质量。
- 32 张本地图均以原分辨率打开；远程外链只核其目标与语义，没有把外部网页上的动态榜单复制成永久事实。

<a id="sec-34"></a>
## 34. 一页复习流程与学完能力清单

### 34.1 看到一个新多模态模型，按六问拆开

```text
1. 输入是什么模态？目标是理解、生成，还是两者？
2. 原始图/视频怎样变 token：连续 feature 还是离散 code？
3. 视觉 token 的 shape、数量和位置轴怎样算？
4. 怎样接 LM：linear projector、MLP、cross-attention，还是同一词表？
5. loss 的正负方向与分母是什么：row/column、pair、token、sample、batch？
6. 省下了什么，又付出什么：细节、token 预算、通信、稳定性或生成质量？
```

### 34.2 学完后你应该能独立完成

- 从 $`[N,d]`$ 图文 embedding 算出 $`[N,N]`$ 相似度矩阵，并分别按行、按列复算 CLIP 双向 loss；
- 用 $`y=+1/-1`$ 检查 SigLIP 正负 pair 的梯度方向，并明确论文分母与 per-pair mean 的固定倍率差；
- 由图片边长、patch、merge 算出64个视觉内容token，再说明加视觉起止边界后为何进入LM是66；
- 为单图、多图、视频列 token budget，解释 AnyRes/dynamic resolution 的清晰度—成本交换；
- 给视觉 token 写 $`(t,h,w)`$，解释 interleaved MRoPE、显式 timestamp 与 DeepStack 分别改哪一层；
- 复算长短样本在 token mean 与 square-root weighting 下的相对贡献，并说明权重不是概率；
- 从小 codebook 找 nearest code，计算 8192 个 code 需要 13 bit、1024 索引需要 1,664 bytes；
- 区分“能理解图片”“能输出图 token”“能还原高质量像素”，不把其中一个 benchmark 当成另外两个的证明；
- 审计任何模型卡中的数据总量、设备天数和排行榜，先问集合是否互斥、协议是否一致、数字是否仍属当前时点。

最后一句：**多模态模型的共同问题不是“把图片塞进去”这么简单，而是选择什么信息表示、给它多少 token、用什么目标训练，并诚实支付每个选择的代价。**
