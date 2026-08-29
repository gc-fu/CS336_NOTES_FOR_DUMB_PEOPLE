# Lecture 11：Scaling Laws II 与 μP——怎样把小实验可靠地搬到大模型

> CS336 Spring 2026，Lecture 11。  
> 官方视频：[Stanford Online](https://www.youtube.com/watch?v=vTfEyOyzV9E)。  
> 课程讲义：[Stanford CS336 官方 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)，共 58 页。  
> 本笔记把 PDF、完整人工英文字幕和一手论文交叉整理成一份可以独立学习的中文教程。

> **第一次阅读怎么用：**
>
> 1. 先跳过 §1 的“五分钟复习卡”，从 §0、§2 开始。
> 2. 每遇到公式，先看“字母卡”，再跟着小数字例算一遍。
> 3. 【课程内容】是课件或讲者明确讲到的内容；【视频补充】是只在口头出现的解释；【补充解释】是为了把课程拆成更小步；【补充】是可靠的一手资料；【延伸】可以先跳过。
> 4. 图表上的最佳点只说明那一组实验；它不是宇宙定律，也不能自动证明因果。
> 5. §23–§24 有 80 道题和完整答案。首次学习不必一次做完。

---

## 0. 导航、目标与最少前置知识

### 0.1 这讲到底解决什么问题

Lecture 9 讲了“模型、数据和训练计算增加时，loss 大致怎样变”。本讲追问更实际的问题：

> 如果我只能在小模型上试几十次，怎样选择大模型的宽度、深度、batch size、learning rate、训练日程和 optimizer，才不至于在昂贵的大训练上才发现设置错了？

视频 [00:24](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=24s) 把目标说成“怎样在实践中扩大语言模型”；[01:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=62s) 提醒 optimizer 行为会随规模改变。

全讲只有一条主线：

1. 先用小而便宜的实验找到候选 recipe。
2. 再问哪些量会随规模漂移，哪些量可能迁移。
3. 固定 aspect ratio 是一种简单路线；μP 是一套更系统的宽度缩放参数化。
4. 不管用哪条路线，都要留真正没参与拟合的规模做验证。

### 0.2 可点击目录

- [§1 五分钟复习卡](#1-五分钟复习卡首次阅读跳过)
- [§2 从真实训练问题开始](#2-从真实训练问题开始scaling-recipe-不是一条公式)
- [§3 符号和四则运算桥](#3-本讲符号和四则运算桥)
- [§4 MiniCPM 的固定形状路线](#4-minicpm固定aspect-ratio再配合宽度缩放)
- [§5 batch size 与 learning rate](#5-batch-size与learning-rate为什么也要scale)
- [§6 WSD 训练日程](#6-wsd怎样复用一条训练主干)
- [§7 Chinchilla 的三种实验方法](#7-chinchilla式compute-optimal-scaling的三种方法)
- [§8 DeepSeek 的拟合路线](#8-deepseek路线不使用μp也能做系统拟合)
- [§9 公司案例怎样读](#9-qwenkimi-k2hunyuanllama-3与minimax案例)
- [§10 StepFun 的二维超参数曲面](#10-stepfun把learning-rate与batch一起拟合)
- [§11–§12 optimizer 与 Muon](#11-optimizer是一个会污染scaling结论的变量)
- [§13–§18 μP 完整推导](#13-μp要解决什么standard-parameterization哪里漂)
- [§19 实践决策树](#19-一套可执行的scaling-recipe)
- [§20 公式卡](#20-公式卡)
- [§21 常见误区](#21-常见误区错误为什么错正确说法)
- [§22 术语表](#22-术语表)
- [§23 自测](#23-自测题80题)
- [§24 答案](#24-自测答案)
- [§25 视频导航](#25-视频时间导航)
- [§26 PDF 覆盖](#26-pdf-1–58页覆盖表)
- [§27 来源与边界](#27-来源边界与学完后的能力)

### 0.3 先认识十三个词

- **Scaling law（缩放规律）**：用实验拟合“规模变大时某个指标怎样变化”的经验关系。
- **Recipe（训练配方）**：模型结构、初始化、batch、learning rate、optimizer、训练日程等设置的组合。
- **Hyperparameter（超参数）**：训练前由人选择、不是模型从数据中直接学出的设置。
- **Parameter（参数）**：模型训练中会被更新的数字，例如矩阵权重。
- **Width（宽度）**：一层里向量或隐藏维度的大小；本讲常用 $`n`$ 或 $`d_m`$ 表示。
- **Depth（深度）**：层数。
- **Aspect ratio（形状比例）**：宽度、**FFN（feed-forward network，前馈网络）中间宽度**、attention **head（注意力头，即并行处理不同关系的一组 query/key/value，查询/键/值子空间）**数、深度之间如何按固定比例一起变。
- **Gradient（梯度）**：参数增加一点时，loss 会朝哪个方向、变化多快的局部变化率。
- **Learning rate，LR（学习率）**：每次更新沿负梯度方向走多大一步。
- **Optimizer（优化器）**：把梯度变成参数更新的规则，例如 AdamW、Muon、Lion。
- **Loss（损失）**：衡量模型预测有多差的数字；同一评测口径下通常越小越好。
- **Token（词元）**：模型实际处理的离散编号单位，不一定等于一个汉字或英文单词。
- **Tokenizer（分词器）**：把文字按固定规则转换成 token IDs 的程序。

一个最小梯度下降例：

```math
\theta_{\text{new}}=\theta_{\text{old}}-\eta g.
```

这里 $`\theta`$ 是一个参数，$`g`$ 是它的梯度，$`\eta`$ 是 learning rate。若 $`\theta=5,g=2,\eta=0.1`$，则

```math
\theta_{\text{new}}=5-0.1\times2=4.8.
```

减号的意思是朝让 loss 下降的方向走；步子太小会慢，太大可能越过低点甚至发散。

---

<a id="1-五分钟复习卡首次阅读跳过"></a>
## 1. 五分钟复习卡（首次阅读跳过）

1. Scaling recipe 不只选模型大小；还要选数据、batch、LR、optimizer、初始化和 schedule。（§2）
2. 固定 aspect ratio 让不同规模保持几何形状接近，减少变量，但不会自动保证最优。（§4）
3. MiniCPM 课程案例同时缩放 embedding（token ID 到连续向量的查表层）、residual、初始化和矩阵 LR；不能只抄一个 0.01。（§4）
4. Optimal batch 通常随训练规模变；“一个 batch 走天下”可能浪费吞吐或增加噪声。（§5）
5. Cosine 若为每个目标训练长度从头重跑，成本会累加；WSD 复用 stable 主干，再从 checkpoint 接短 decay。（§6）
6. Chinchilla method 1 看 lower envelope，method 2 做 IsoFLOP 扫描，method 3 联合拟合；三者受不同 confound 影响。（§7）
7. DeepSeek 课程案例拟合 $`\eta_{\text{opt}}\propto C^{-0.125}`$、$`B_{\text{opt}}\propto C^{0.3271}`$；这是其设置中的经验式。（§8）
8. 不同公司报告的坐标、tokenizer、数据和 compute 口径不同，不能把点放在一张表就当公平竞赛。（§9）
9. StepFun 把 LR 与 batch 同时扫描成二维 loss 曲面；单独只调一个可能错过真正低点。（§10）
10. 比 optimizer 时必须分别调 LR、weight decay 等；默认值不同会制造“算法优势”。（§11）
11. Muon 对矩阵梯度的 momentum 做近似正交化；它不是“AdamW 永远更快”的证明。（§12）
12. μP 的两个目标：初始化后每个 activation 坐标保持 $`\Theta(1)`$，一次更新引起的 activation 变化也保持 $`\Theta(1)`$。（§13–§16）
13. 课程 p47 的初始化尺度同时看 fan-in 和 fan-out；当两者同阶时约为 $`1/\sqrt{n_{\text{in}}}`$。（§14）
14. μP 迁移的是 base 超参数在一套明确参数化下的最优区域；不是所有实际 tensor LR 都数值不变。（§16）
15. RMSNorm gain、Lion、强 decoupled weight decay 在课程实验中可能破坏 transfer；先做 stress test。（§18）

---

<a id="2-从真实训练问题开始scaling-recipe-不是一条公式"></a>
## 2. 从真实训练问题开始：scaling recipe 不是一条公式

### 2.1 “预测最佳模型大小”还不够

【课程内容，PDF p.2–5】讲者先列出实际训练会问的问题：

- 我有固定计算预算，模型和数据各用多少？
- 选多大的 **global batch（全局批量：所有设备和 gradient accumulation 合计、一次参数更新用掉的样本或 token 数）**？
- 最佳 LR 会不会随模型或数据量改变？
- cosine、WSD 等 schedule 怎么选？
- AdamW、Muon 或别的 optimizer 在小模型上的排序能否搬到大模型？
- 小模型调好的初始化，在宽模型上会不会让 activation 或 update 爆掉？

视频 [00:40](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=40s) 说还要优化许多 learning-rate 技巧；[01:10](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=70s) 把初始化、LR、batch 一起列为敏感变量。

### 2.2 一次大训练为什么容错很低

【补充解释】假设一次目标训练要 100 万 GPU-hours。两个 recipe：

- A：最终 loss 2.00；
- B：最终 loss 2.02，但每步吞吐快 5%。

只看 loss 会选 A，只看吞吐会选 B。真实选择还要问：

1. B 的 2.02 是否在误差范围内？
2. A 会不会因为更大 batch 导致显存不足？
3. **Downstream（下游）**任务——预训练之后的具体任务或评测——是否同样排序？
4. 训练走到 80% 才发现 LR 太大，能否恢复？

所以 recipe 是多目标决策，不是把某个幂律代入一次。

### 2.3 训练数据流的最小地图

一次训练 step：

1. **Forward pass（前向）**：token 经 **embedding（把 token ID 查成连续向量的表）**、Transformer 层得到 **logits（softmax 前、尚未归一化的每个候选 token 分数）**和 loss。
2. **Backward pass（反向）**：从 loss 往回计算每个参数的 gradient。
3. Optimizer 用 gradient 更新参数。

**Activation（激活）**是 forward 中间算出的向量；**activation gradient** 是 loss 对中间向量的梯度；**weight gradient** 是 loss 对权重的梯度。

形状例：batch $`B=2`$，序列长 $`S=3`$，hidden width $`M=4`$，则一层输入 $`X`$ shape 为 $`[2,3,4]`$。线性权重 $`W`$ shape 为 $`[4,8]`$，输出为 $`[2,3,8]`$。宽度从 4 改成 8，权重和 activation 的 shape 都会变，这正是超参数可能漂移的来源。

### 2.4 “相关”不能偷换成“原因”

若图上大模型常用更小 LR，只能先说：

> 在这套 architecture、data、optimizer、schedule 和搜索范围里，拟合到的最佳 LR 随规模下降。

不能直接说“模型大必然导致 LR 小”。可能同时改变的变量叫 **confound（混杂因素）**。例如大模型也用了更长 **warmup（预热：训练开始时把 LR 从很小逐渐升到目标值）**、更高 batch 或不同 optimizer；它们都可能改变最佳 LR。

视频 [01:36](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=96s) 开始介绍两条实践路线；[02:18](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=138s) 说明接下来会逐篇看实践案例。

---

<a id="3-本讲符号和四则运算桥"></a>
## 3. 本讲符号和四则运算桥

### 3.1 符号卡

| 符号 | 人话 | 常见单位 |
|---|---|---|
| $`N`$ | 模型参数量，某些节也用作规模变量 | parameters |
| $`D`$ | 训练 token 数 | tokens |
| $`C`$ | 训练计算量 | FLOPs |
| $`B`$ | global batch size | tokens/step 或 sequences/step |
| $`\eta`$ | learning rate | 无单位缩放因子 |
| $`M`$ | Transformer 主宽度 | elements |
| $`F`$ | FFN 中间宽度 | elements |
| $`L`$ | 层数 | layers |
| $`n_{l-1},n_l`$ | 第 $`l`$ 层输入、输出宽度 | coordinates |
| $`\sigma`$ | 初始化标准差；注意课件某表也把同字母用于“variance”标签 | 与权重同单位 |
| $`\Theta(1)`$ | 宽度增大时仍在常数量级，不随宽度趋零或爆大 | 量级记号 |

本讲的字母会复用。每节开头会重置口径；看到 $`D`$ 时先看它是 data tokens 还是某篇论文的 head dimension。

### 3.2 FLOP 与 compute

**FLOP（floating-point operation，浮点运算）**是一次浮点加、乘等操作。FLOPs 在“工作量”语境中指很多次操作，不是每秒速度。稠密 Transformer 训练常用粗略式

```math
C\approx6ND.
```

含义：$`N`$ 个参数、$`D`$ 个训练 token，每个 token 的 forward 约 $`2N`$ FLOPs，backward 约 forward 的两倍，总约 $`6N`$。它忽略 attention 的额外项、embedding/loss、通信和 data movement；**MoE（Mixture of Experts，混合专家）**让每个 token 只经过一部分专家，因此应用每 token 激活参数而非专家总参数。它是粗账，不是硬件实测。

例：$`N=10^9`$，$`D=2\times10^{10}`$：

```math
C\approx6\times10^9\times2\times10^{10}
=12\times10^{19}=1.2\times10^{20}\text{ FLOPs}.
```

### 3.3 指数与对数只要懂这几步

```math
y=x^a
```

表示把 $`x`$ 按指数 $`a`$ 缩放。若 $`a=1/2`$，就是平方根；$`16^{1/2}=4`$，因为 $`4\times4=16`$。若 $`a=-1/2`$，则 $`16^{-1/2}=1/4`$。

**Log（对数）**回答“底数要乘自己多少次得到这个数”。若用自然对数 $`\ln`$，底数是 $`e\approx2.718`$。本讲图上 log 轴常只利用：

```math
\log(x^a)=a\log x,\qquad
\log(xy)=\log x+\log y.
```

因此幂律在 log-log 图上变成直线。它只是画图和拟合工具，不会把经验关系变成理论定律。

### 3.4 $`O(1)`$、$`\Theta(1)`$ 和“常数量级”

假设宽度依次为 100、1000、10000：

- 数列 2、2.1、1.9 保持常数量级，可写 $`\Theta(1)`$。
- 数列 10、31.6、100 约按 $`\sqrt n`$ 增长，不是 $`\Theta(1)`$。
- 数列 0.1、0.0316、0.01 约按 $`1/\sqrt n`$ 变小，也不是 $`\Theta(1)`$。

μP 说 activation 是 $`\Theta(1)`$，通常指**每个坐标**的典型大小保持常数，不是整条向量的欧氏长度不变。若 $`n`$ 个坐标都约 1，向量长度约 $`\sqrt n`$。

---

<a id="4-minicpm固定aspect-ratio再配合宽度缩放"></a>
## 4. MiniCPM：固定 aspect ratio，再配合宽度缩放

### 4.1 什么叫固定 aspect ratio

【课程内容，PDF p.6–9】MiniCPM 路线让模型保持相似形状：主宽度 $`d_m`$ 变大时，FFN 宽度、head 数和层数按预定规则一起变。视频 [03:35](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=215s) 开始 MiniCPM 案例。

小例：

| 模型 | $`d_m`$ | $`d_{ff}`$ | heads | layers | $`d_{ff}/d_m`$ |
|---|---:|---:|---:|---:|---:|
| A | 4 | 10 | 2 | 2 | 2.5 |
| B | 8 | 20 | 4 | 4 | 2.5 |

两者 $`d_{ff}/d_m`$ 一样，但 B 的 depth/width 比仍可能不同。所谓“固定形状”必须具体说明哪些比例固定。

### 4.2 课程表的模型梯子

PDF p.9 给出约 9M、30M、70M、0.1B、0.17B、0.2B、0.5B 的小模型，主宽度从 320 增到 1344，层数从 8 增到 24。课程把目标模型设为小梯子最大点的约 5 倍规模。视频 [06:32](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=392s) 开始逐列解释；[07:11](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=431s) 说明目标外推。

关键不是背表，而是看实验设计：

1. 有多个规模，不是只比最小和最大。
2. 模型形状沿一条受控路径变化。
3. 目标规模只比已测范围大几倍，不是跨一百倍盲猜。

### 4.3 MiniCPM 的四个缩放动作

【课程内容，PDF p.8】课件列：

1. embedding 输出乘 $`12`$；
2. 每层 residual increment 乘 $`1.4/\sqrt L`$；
3. 二维权重初始化标准差按宽度比缩放；
4. 二维权重的实际 LR 也按宽度比缩放，输出 logits 同理。

视频 [05:49](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=349s) 指向 embedding scaling；[06:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=365s) 解释 tensor learning-rate scaling；[06:25](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=385s) 转入模型梯子。

设 base width $`d_{\text{base}}=4`$，当前 width $`d_m=16`$，则宽度比

```math
r=\frac{d_m}{d_{\text{base}}}=\frac{16}{4}=4.
```

若基础二维权重标准差为 0.1，课程规则可写成

```math
\sigma_{\text{current}}
=\frac{0.1}{\sqrt r}
=\frac{0.1}{2}
=0.05.
```

若 base LR 为 $`0.01`$，二维矩阵有效 LR 乘 $`1/r`$：

```math
\eta_{\text{matrix}}=\frac{0.01}{4}=0.0025.
```

注意：用户界面中搜索到的 **base LR** 仍可能是 0.01；参数组内部乘了 width multiplier 后，矩阵实际更新步长是 0.0025。这就是“最佳 LR 不漂”看起来反直觉的原因之一。

### 4.4 residual 的 $`1/\sqrt L`$ 为什么合理

【补充解释】若每层 residual 增量是相互近似不相关、每个方差为 $`v`$，相加 $`L`$ 层后的方差约 $`Lv`$。每层先乘 $`1/\sqrt L`$，方差会乘 $`1/L`$，总方差约：

```math
L\times\frac1L v=v.
```

小例：4 层，每层原增量标准差 2。乘 $`1/\sqrt4=1/2`$ 后，每层标准差 1、方差 1；四层独立相加方差 $`1+1+1+1=4`$，总标准差 2。没有缩放时每层方差 4，总方差 16，总标准差 4。

这是独立近似的直觉，不是说真实 Transformer 各层完全独立。

### 4.5 课程图怎样读

PDF p.10 的不同宽度曲线在 base LR 约 $`10^{-2}`$ 附近取得低 loss。视频 [07:26](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=446s) 开始读 LR 曲线；[07:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=479s) 强调这是 μP 的成功案例。

正确结论：

> 在这套 MiniCPM 参数化、模型梯子和训练设置里，经过宽度 multiplier 后，base LR 的最佳区域较稳定。

错误结论：

> 所有模型都应该直接把每个 tensor 的实际 LR 设成 0.01。

### 4.6 固定 aspect ratio 的边界

- 它减少实验维度，但可能错过更深、更窄或更浅、更宽的好模型。
- 若 tokenizer、数据、optimizer 或 schedule 改了，旧最优点可能移动。
- 参数量相近不等于 FLOPs、activation memory、通信都相近。
- 图上曲线的最低采样点不一定是真实连续最优点；需要在附近加密。

---

<a id="5-batch-size与learning-rate为什么也要scale"></a>
## 5. Batch size 与 learning rate：为什么也要 scale

### 5.1 batch 先定义清楚

**Global batch size** 是一次 optimizer update 合计使用的训练样本或 token 数。若 8 张 GPU，每卡 4 条序列，每条 512 tokens：

```math
B_{\text{tokens}}=8\times4\times512=16{,}384\text{ tokens/update}.
```

若用 2 次 gradient accumulation（先累积两小批梯度再更新），global batch 变成

```math
8\times4\times512\times2=32{,}768.
```

不写单位只说“batch=32”很危险：可能指 32 sequences，也可能指 32K tokens。

### 5.2 batch 太小与太大

【补充解释】

- 太小：gradient 噪声大，硬件矩阵不够大，吞吐可能低。
- 太大：显存压力大；每处理同样 tokens 的 optimizer steps 变少；超过 **critical batch（临界批量：继续增大 batch 后边际收益开始明显变小的转折量级）** 后，继续加大收益可能很小。

“Gradient noise”是不同小批样本给出的 gradient 不完全相同。若每个样本 gradient $`g_i`$ 独立、同分布，且单样本方差是 $`\mathrm{Var}(g)`$，batch 平均

```math
\bar g=\frac1B\sum_{i=1}^B g_i
```

的方差为

```math
\mathrm{Var}(\bar g)
=\frac1{B^2}\sum_{i=1}^B\mathrm{Var}(g_i)
=\frac{B\mathrm{Var}(g)}{B^2}
=\frac{\mathrm{Var}(g)}B.
```

中间没有 covariance（协方差）交叉项，是因为这里明确假设不同样本独立；真实训练样本相关时不一定严格成立。小例：单样本 gradient 均匀取 $`1,3,5,7`$，均值为 4，方差为

```math
[(1-4)^2+(3-4)^2+(5-4)^2+(7-4)^2]/4
=(9+1+1+9)/4=5.
```

因此独立抽样时，batch 1/2/4 的平均 gradient 方差分别是 $`5,2.5,1.25`$。这才是在说明“batch 大，均值噪声通常小”。若只是拿固定前几个数计算：

- batch 1 可能抽到 1 或 7，波动大；
- batch 2 取前两个，均值 $`(1+3)/2=2`$；
- batch 4，均值 $`(1+3+5+7)/4=4`$。这组观测均值用于练算术；上面的 $`5/B`$ 才是跨重复抽样的方差结论。

稳定不等于一定更好：batch 4 的每一步处理了 4 倍样本。

### 5.3 MiniCPM 的 batch 经验式

【课程内容，PDF p.11–12】课件拟合：

```math
\log(BS)=-6.24\log(L)+20.91.
```

这里 $`BS`$ 是 batch size，$`L`$ 是 loss；不要把这里的 $`L`$ 和“层数”混用。视频 [08:16](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=496s) 开始 batch 图；[09:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=545s) 解释跨模型规模的 token 曲线；[09:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=584s) 总结拟合。

利用 $`\log x^a=a\log x`$，可改写为：

```math
BS=e^{20.91}L^{-6.24}.
```

若 loss 从 2 降到 1，比例为

```math
\frac{BS(1)}{BS(2)}
=\frac{1^{-6.24}}{2^{-6.24}}
=2^{6.24}\approx75.5.
```

这个巨大倍数不是普遍建议；它只说明该拟合非常陡，外推前必须验证。

### 5.4 LR 与 batch 常常互相影响

平均 batch gradient 变稳定后，某些 recipe 可以用更大 LR；但关系取决于 optimizer 和 loss surface。只扫描 LR、固定一个糟糕 batch，会得到错误的“最佳 LR”。StepFun 在 §10 会把二者同时扫描。

小型网格例：

| batch | LR 0.01 | LR 0.03 |
|---:|---:|---:|
| 8 | loss 2.2 | loss 2.0 |
| 32 | loss 1.9 | loss 2.4 |

若只看 batch 8，会选 LR 0.03；若目标 batch 32，LR 0.01 才更好。两个超参数有 interaction（交互）。

---

<a id="6-wsd怎样复用一条训练主干"></a>
## 6. WSD：怎样复用一条训练主干

### 6.1 cosine 的重复训练问题

**Schedule（学习率日程）**规定训练过程中 LR 怎样变化。Cosine schedule 通常从大 LR 平滑降到接近 0；它需要预先知道训练终点。

假设想测试目标长度 10、20、30、40 tokens 单位。若每个 cosine run 都从头训练，工作量是：

```math
10+20+30+40=100.
```

一般测试 $`1,2,\ldots,n`$ 个单位：

```math
1+2+\cdots+n=\frac{n(n+1)}2,
```

随 $`n^2`$ 增长。视频 [10:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=605s) 指出反复 cosine 的问题；[10:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=644s) 说明换训练长度需要重跑。

### 6.2 WSD 三段

**WSD** 是 warmup–stable–decay：

1. **Warmup**：LR 从很小升到目标值，避免一开始更新过猛。
2. **Stable**：长时间保持大致稳定 LR。
3. **Decay**：从一个 checkpoint 分叉，短时间降低 LR，收敛到较低 loss。

**Checkpoint（检查点）**是训练时保存的参数、optimizer state 和进度快照，可以从那里继续。

视频 [11:13](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=673s) 定义 WSD；[11:37](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=697s) 转入快速 decay；[12:15](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=735s) 说明回到最后一个 stable checkpoint。

### 6.3 小时间线

想得到 10、20、30、40 四个目标点：

- stable 主干训练到 40，成本 40；
- 在 10、20、30、40 各保存 checkpoint；
- 每个 checkpoint 接长度 4 的 decay，成本 $`4\times4=16`$；
- 总成本 $`40+16=56`$，小于从头 cosine 的 100。

课件 PDF p.14 的示意比较 cosine 40N、WSD stable 40N+decay4N，以及 stable80N+decay8N。PDF p.15 图中 stable 阶段 loss 可能暂时高于正在 decay 的 run，但 decay 后迅速下降。视频 [12:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=764s) 解释分叉；[13:08](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=788s) 比较曲线；[14:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=842s) 说明通常约最后 10% 做 decay。

### 6.4 “线性成本”要有条件

若最大主干长度为 $`C`$，分叉数 $`m`$，每段 decay 长度 $`d`$，总成本：

```math
C+md.
```

若 $`m`$ 是固定小数，且 $`d`$ 不随 $`C`$ 线性增长，则关于 $`C`$ 近似线性。若你为每个整数 checkpoint 都接一个占其长度 10% 的 decay：

```math
0.1(1+2+\cdots+C),
```

仍是 $`O(C^2)`$。因此“WSD 把一切变线性”是过度表述；它依靠复用主干和有限分叉。

### 6.5 WSD 不是免费午餐

- 每个 decay 仍要训练和存 checkpoint。
- stable 与 cosine 的训练轨迹不同，最终质量可能略有差别。
- 换数据、batch 或 optimizer 后，最佳 decay 长度可能变。
- 若目标训练必须在某个特定 token 数结束，仍要做目标规模验证。

---

<a id="7-chinchilla式compute-optimal-scaling的三种方法"></a>
## 7. Chinchilla 式 compute-optimal scaling 的三种方法

### 7.1 问题重新写清楚

给定训练计算预算 $`C`$，选模型大小 $`N`$ 和数据 token 数 $`D`$，使 validation loss 最低。粗略约束：

```math
C\approx6ND.
```

因此固定 $`C`$ 时，$`N`$ 变大，$`D`$ 必须变小。例：忽略系数 6，只令 $`ND=100`$，候选有 $`(1,100),(2,50),(4,25),(10,10)`$。

**Validation loss（验证损失）**是在没有用于参数更新的验证数据上算的 loss，用来估计泛化。它不是训练 loss，也不等于 downstream accuracy。

视频 [14:09](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=849s) 回顾 Chinchilla；[14:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=884s) 开始读 MiniCPM 的 method 1。

### 7.2 Method 1：lower envelope

训练许多模型，在每个 compute 区间选 loss 最低的点，低点连起来叫 **lower envelope（下包络）**。PDF p.17 展示这种读法。

小例：

| compute | 模型 A loss | 模型 B loss | 下包络 |
|---:|---:|---:|---:|
| 10 | 2.5 | 2.8 | 2.5（A） |
| 20 | 2.2 | 2.1 | 2.1（B） |
| 40 | 2.0 | 1.9 | 1.9（B） |

优点：不强迫所有点服从一个联合式。  
缺点：只有靠近已跑配置的离散低点；采样网格稀疏会把“谁最低”看错。

### 7.3 Method 2：IsoFLOP

**IsoFLOP** 意为固定相同 FLOPs。在每个 $`C`$ 下扫多组 $`N,D`$，拟合一条 U 形或碗形曲线，找最低点，再拟合最优 $`N(C),D(C)`$。

例：固定 $`ND=100`$，假设目标函数

```math
L(N,D)=1+\frac4{\sqrt N}+\frac4{\sqrt D}.
```

| $`N`$ | $`D=100/N`$ | $`4/\sqrt N`$ | $`4/\sqrt D`$ | $`L`$ |
|---:|---:|---:|---:|---:|
| 1 | 100 | 4 | 0.4 | 5.4 |
| 4 | 25 | 2 | 0.8 | 3.8 |
| 10 | 10 | 1.265 | 1.265 | 3.530 |
| 25 | 4 | 0.8 | 2 | 3.8 |
| 100 | 1 | 0.4 | 4 | 5.4 |

中间最低，因为太小的模型和太少的数据都会罚 loss。视频 [15:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=901s) 讨论 data 与 non-embedding parameter trade-off。

### 7.4 Method 3：联合拟合

假设：

```math
L(N,D)=L_0+\frac{A}{N^\alpha}+\frac{B_D}{D^\beta}.
```

这里：

- $`L_0`$：再大也消不掉的拟合渐近项；
- $`A,B_D>0`$：两项的尺度；
- $`\alpha,\beta>0`$：模型项、数据项下降速度。

把所有 $`(N,D,L)`$ 点一起拟合这些参数，再在 $`C\approx6ND`$ 下求最优。优点是能用所有点；缺点是如果函数形状写错、参数口径错、训练没收敛，所有点会一起把结论拉偏。

视频 [15:28](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=928s) 说联合拟合容易受形式影响；后面的公司报告展示了不同选择。

【课程内容，PDF p.18 高清核对】紧接 Method 3 的具体图属于 **MiniCPM 的 UltraText 数据设置**，不是 DeepSeek。图中联合式是：

```math
\boxed{
L(N,D)=0.0754N^{-0.30}+0.292D^{-0.30}+0.25
}
```

图的横轴是 non-embedding parameters（以 $`10^9`$ 为单位），纵轴是 compute（以 $`10^{18}`$ FLOPs 为单位），颜色/等高线表示拟合 loss。它在自己的参数与单位口径下报告

```math
\left.\frac{D_{\mathrm{opt}}}{N_{\mathrm{opt}}}\right|_{C=10^{21}}
\approx95.60.
```

这句话只表示：在该 UltraText 联合拟合、该 compute 点和该 $`N,D`$ 定义下，最优数据 token 数约为 non-embedding 参数量的 95.60 倍。它不是普遍 tokens-per-parameter 常数。

数字例：若用图中归一化单位取 $`N=D=1`$，

```math
L=0.0754+0.292+0.25=0.6174.
```

若 $`N=D=1000=10^3`$，则 $`1000^{0.30}=10^{0.9}\approx7.943`$，

```math
L\approx0.0754/7.943+0.292/7.943+0.25
\approx0.00949+0.03676+0.25
=0.29625.
```

前两项随规模下降，拟合渐近 0.25；这里的 0.25 也只属于该拟合。

### 7.5 三种方法的共同 confound

1. **Embedding/output 参数是否计入 $`N`$**；
2. warmup 是否对小模型占比过大；
3. 小模型是否处在不同优化 **regime（工作区间：主导机制和趋势大致相同的一段范围）**；
4. batch 是否对每个规模公平调过；
5. FLOPs 用 $`6ND`$ 还是真实 non-embedding FLOPs/token；
6. 数据质量和 tokenizer 是否一致；
7. 低 loss 是否也预测 downstream。

因此三种方法互相校验，比只信一种更稳。

---

<a id="8-deepseek路线不使用μp也能做系统拟合"></a>
## 8. DeepSeek 路线：不使用 μP，也能做系统拟合

### 8.1 两条路线不要混成一条

【课程内容，PDF p.16、p.20】课程先用 MiniCPM 展示“固定 aspect ratio + μP 式缩放”，再用 DeepSeek 展示另一条路线：直接在小计算预算做 LR×batch 网格，拟合它们随 compute 的关系。视频 [15:57](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=957s) 转入 DeepSeek；[16:31](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=991s) 明确说这里没有采用 μP。

因此：

- μP 不是做 scaling law 的必要条件；
- 网格拟合也不是 μP 的替代证明；
- 两条路线都必须在更大规模留验证点。

### 8.2 DeepSeek 的小规模二维网格

【课程内容，PDF p.20–21】课程案例在固定的小 compute（约 $`10^{17}`$ FLOPs）附近，扫描 model、LR 和 batch。视频 [17:19](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1039s) 解释网格；[18:06](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1086s) 说明从大网格找 minimizer。

一个 3×3 教学网格：

| batch \ LR | 0.001 | 0.003 | 0.009 |
|---:|---:|---:|---:|
| 256 | 2.20 | 2.05 | 2.40 |
| 512 | 2.12 | **2.00** | 2.31 |
| 1024 | 2.15 | 2.03 | 2.25 |

最低是 batch 512、LR 0.003。但周围的 2.03、2.05 只差很小；若训练噪声约 0.02，不能把第三位小数当确定排序。

### 8.3 课程拟合的 LR 与 batch 经验式

PDF p.21 给出：

```math
\eta_{\mathrm{opt}}=0.3118C^{-0.125},
\qquad
B_{\mathrm{opt}}=0.2920C^{0.3271}.
```

这里 $`C`$ 是训练 compute；$`\eta_{\mathrm{opt}}`$ 是拟合的最佳 learning rate；$`B_{\mathrm{opt}}`$ 是拟合的最佳 batch。常数的数值依赖论文使用的单位，不能把 $`C`$ 换单位后仍原样使用。

只看比例最安全。若 compute 增 $`16`$ 倍：

```math
\frac{\eta_2}{\eta_1}=16^{-0.125}=16^{-1/8}=2^{-1/2}\approx0.707.
```

因为 $`16=2^4`$，$`16^{1/8}=2^{4/8}=\sqrt2\approx1.414`$，倒数约 0.707。

batch 比例：

```math
\frac{B_2}{B_1}=16^{0.3271}
=e^{0.3271\ln16}
\approx e^{0.907}
\approx2.48.
```

这表示该拟合里 compute 大 16 倍，最佳 LR 约乘 0.707，batch 约乘 2.48；不是所有模型的固定规则。

### 8.4 “0.25% 内都差不多”怎样理解

【视频补充】视频 [18:30](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1110s) 读宽低谷；[18:52](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1132s) 批评只报一个尖锐最优值。

若最低 loss 是 2.000，0.25% 是：

```math
2.000\times0.0025=0.005.
```

所以 loss 不超过 2.005 的点都在这个相对范围。实践中可在这些点里再选吞吐更高、显存更稳或对失败更鲁棒的设置。

### 8.5 DeepSeek 的 WSD 与 Method 2

【课程内容，PDF p.22–24】课程列出多阶段 LR：

- 先 warmup；
- 大部分训练保持最大 LR；
- 约 80% 后降到最大值的 31.6%；
- 约 90% 后降到 10%。

这些是该报告的 schedule 快照，不是 WSD 唯一实现。视频 [20:24](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1224s) 明确说采用 WSD 风格；[20:50](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1250s) 指向 IsoFLOP sweeps。

PDF p.23 的 Method 2 使用

```math
C=M D,
```

这里 $`M`$ 是 **non-embedding FLOPs per token（每 token 的非 embedding 运算量）**，不是参数量 $`N`$；这与粗式 $`6ND`$ 不是同一个横轴。课程展示外推到约 67B 模型、$`4.3\times10^{11}`$ FLOPs/token、$`1.04\times10^{12}`$ tokens，计算：

```math
4.3\times10^{11}\times1.04\times10^{12}
=4.472\times10^{23}\text{ FLOPs},
```

约 $`4.5\times10^{23}`$。视频 [21:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1265s) 说明口径；[22:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1322s) 给目标点。

### 8.6 不要把 p.18 的 UltraText 拟合归到 DeepSeek

PDF p.18 仍在 Chinchilla Method 3 / MiniCPM UltraText 案例中，具体式与 $`95.60`$ ratio 已在 §7.4 完整讲解。DeepSeek 章节从 PDF p.19 才开始；本节只保留这条来源分界，防止把两个团队、两套数据和两套 compute 口径拼成一项证据。

### 8.7 课程与一手来源边界

【补充】DeepSeek LLM 报告给出了它的 scaling、batch/LR 与模型配置实验；这里的数字只支持该报告语境。[DeepSeek LLM 一手论文](https://arxiv.org/abs/2401.02954)。课程把多个图拼成一条教学主线，笔记不声称每句话都是论文逐字原文。

---

<a id="9-qwenkimi-k2hunyuanllama-3与minimax案例"></a>
## 9. Qwen、Kimi K2、Hunyuan、Llama 3 与 MiniMax：案例怎样读

### 9.1 案例表先看“它测了什么”

| 课程案例 | PDF | 主要变化轴 | 可以说 | 不能说 |
|---|---:|---|---|---|
| Qwen2.5/Qwen3 | 25 | dense/MoE、model、data | 团队做过多个规模的经验拟合 | 所有 Qwen 私有 recipe 已公开 |
| Kimi K2 | 26 | MoE sparsity ratio | 在固定 activated experts 下比较总 experts | 48 一定是任何系统最佳 |
| Hunyuan | 27 | activated params 与 tokens | 报告给出约 96 tokens/active-param 的点 | 总参数可完全忽略 |
| Llama 3 | 28 | 多个 IsoFLOP budget | 特定实验最低点约 39 tokens/param | 39 是永恒常数 |
| MiniMax | 29 | attention architecture 与规模 | 做了 local architecture scaling | 图单独证明某机制是原因 |

视频 [22:54](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1374s) 进入 Qwen；[23:34](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1414s) 进入 Kimi K2；[25:04](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1504s) 进入 Hunyuan。

### 9.2 Qwen：公开范围就是边界

【课程内容，PDF p.25】课件称 Qwen2.5 以 dense 44M–14B、MoE activated 44M–1B、data 0.8B–600B tokens 做 scaling；Qwen3 报告了类似路线但细节更少。视频 [23:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1385s) 说由 scaling law 预测 optimum。

【补充】[Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115) 可支持公开模型家族和训练方法；没有公开的内部点不能自行补齐。

### 9.3 Kimi K2：总参数与激活参数分开

这里继续使用 §3.2 已定义的 MoE。每 token 只路由到部分 FFN experts，因此：

- total parameters 决定模型存储；
- activated parameters 更接近每 token 计算；
- expert 数与路由还改变通信和负载。

PDF p.26 高清图把 **sparsity ratio** 定义为“routed experts 总数 / 每 token 激活 routed experts 数”。Kimi K2 的课程快照是 384 个 routed experts 中激活 8 个，另有 1 个 shared expert；因此课件选的 ratio 48 来自：

```math
\frac{384}{8}=48.
```

反过来，routed expert 激活比例是：

```math
\frac8{384}=\frac1{48}\approx0.02083=2.083\%.
```

在目标 loss 1.5 附近，课件报告 ratio 48 相对 8、16、32 的 FLOPs 减少约 1.69×、1.39×、1.15×。这不是把整个模型 FLOPs 直接除 48：shared expert、attention、router、embedding 仍运行，专家通信和负载也没有按 $`1/48`$ 自动缩小。视频 [24:08](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1448s) 提问 activated parameters 是否合适；[25:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1501s) 强调系统复杂度。

【补充】[Kimi K2 技术报告](https://arxiv.org/abs/2507.20534) 是模型的公开一手材料；课程图只是一项 scaling 选择，不足以推出 MoE 普遍定律。

### 9.4 Hunyuan 与 Llama 3

PDF p.27 用 activated parameters 做 MoE IsoFLOP，课件例约 58.1B activated、96 tokens/activated-param。若真按 58.1B 算：

```math
58.1\times10^9\times96
=5.5776\times10^{12}\text{ tokens}.
```

这只是比例换算，不代表报告真的训练了恰好该整数，先回查原表单位。来源：[Hunyuan-Large 一手报告](https://arxiv.org/abs/2411.02265)。

PDF p.28 的 Llama 3 图在多个 compute 预算拟合二次曲线最低点，课程总结约 39 tokens/parameter。图里的 **sigmoid（S 形函数：把输入压到 0–1 区间）**用于拟合 loss 到 accuracy。视频 [25:37](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1537s) 开始 Llama 3；[26:26](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1586s) 提醒不同模型会系统偏离 loss→accuracy 拟合。来源：[Llama 3 Herd 一手报告](https://arxiv.org/abs/2407.21783)。

若 $`N=8`$B、比例 39：

```math
D=39\times8\text{B}=312\text{B tokens}.
```

它不是说 312B 后训练一定“过拟合”；deployment-optimal 可能继续多喂数据。

### 9.5 MiniMax：architecture scaling 不是因果证明

PDF p.29 比较 softmax attention、lightning attention 和 hybrid；这里 **softmax** 是把一组 attention scores 变成总和为 1 的权重，在约 70M–7B 范围读趋势。视频 [26:53](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1613s) 开始 MiniMax；[27:38](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1658s) 解释 hybrid；[28:04](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1684s) 收束。

图能支持“在他们实验里某条曲线较好”；不能单独证明：

- 差异只由 attention 类型引起；
- 换数据、参数预算、kernel 后排序不变；
- loss 的小差异等于用户体验差异。

一手来源：[MiniMax-01 技术报告](https://arxiv.org/abs/2501.08313)。

### 9.6 从案例提炼什么

视频 [28:09](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1689s) 做案例总结；[29:10](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1750s) 提醒这些 recipe 很不统一。

能提炼的不是一个神奇常数，而是实验习惯：

1. 覆盖多个规模；
2. 固定或记录形状变化；
3. 明确参数、激活参数、FLOPs/token 的口径；
4. 留 held-out scale；
5. 把 pretraining loss 与 downstream 分开；
6. 对公开空白保持“未知”。

---

<a id="10-stepfun把learning-rate与batch一起拟合"></a>
## 10. StepFun：把 learning rate 与 batch 一起拟合

### 10.1 二维曲面是什么

【课程内容，PDF p.31–37】横轴 LR，纵轴 batch，颜色或高度是最终 loss。每个点是一条真实训练；低处形成一个“碗”。视频 [30:26](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1826s) 介绍 StepFun；[31:11](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1871s) 转入 Step Law。

**Convex surface（凸形曲面）**在这里是“实验范围内看起来像一个碗”的描述：朝最低点走，loss 降；越过后升。它不保证整个神经网络目标在数学上全局凸。

### 10.2 一个可手算的小碗

令

```math
L(\eta,B)=2+100(\eta-0.02)^2+0.01(B-16)^2.
```

逐点算：

- $`\eta=0.02,B=16`$：两平方项为 0，$`L=2`$；
- $`\eta=0.01,B=16`$：$`100(-0.01)^2=0.01`$，$`L=2.01`$；
- $`\eta=0.02,B=20`$：$`0.01(4)^2=0.16`$，$`L=2.16`$；
- $`\eta=0.01,B=20`$：$`L=2+0.01+0.16=2.17`$。

二维最低点是 $`(0.02,16)`$。如果只在 $`B=20`$ 那一行搜 LR，会找到 0.02，但仍误以为最优 loss 是 2.16。

### 10.3 课件表不是一条统一定律

PDF p.33 并列多个经验式：

- OpenAI、Microsoft、DeepSeek、Porian、MiniCPM、Meituan、Step Law；
- 有的自变量是 $`N,D`$，有的是 $`C,L`$；
- 有的拟合 LR，有的拟合 batch；
- 误差指标和实验数据不同。

例如 Step Law 表中：

```math
\eta_{\text{opt}}=1.79N^{-0.713}D^{0.307},
\qquad
B_{\text{opt}}=0.58D^{0.571}.
```

只算比例：固定 $`N`$，数据 $`D`$ 增 4 倍：

```math
\frac{\eta_2}{\eta_1}=4^{0.307}
=e^{0.307\ln4}
\approx e^{0.4256}\approx1.53,
```

```math
\frac{B_2}{B_1}=4^{0.571}
\approx e^{0.7916}\approx2.21.
```

这和 DeepSeek 的“LR 随 compute 下降”表面相反，但自变量和同时变化的量不同；不能直接判谁错。视频 [31:51](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1911s) 讨论公式分歧；[33:28](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2008s) 强调实验口径。

### 10.4 p34–36 图怎样读

PDF p.34：

- 每个小面板是一组 $`N,D,d_m,d_{ff},heads,layers`$；
- 红色叉是扫描到的 global minimum；
- 星号或其它标记是不同经验式预测；
- 标记靠近但不重合，表示经验式近似而非精确。

PDF p.35 把二维面切成一维：固定 LR 看 batch，或固定 batch 看 LR，都呈实验范围内的 U 形。PDF p.36 观察到固定 $`N`$ 时最佳 LR 还随 $`D`$ 变化，batch 主要随 $`D`$ 变化；讲者也说 WSD 下找 LR 比想象中脆弱。

视频 [33:31](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2011s) 逐面板；[35:15](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2115s) 看一维切片；[36:07](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2167s) 讨论 LR 对 model/data 的依赖。

### 10.5 “batch 近似平方根增长”不是精确定理

【视频补充】课程在 [38:04](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2284s)–[39:16](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2356s) 把一些 batch 指数描述成接近平方根。Step Law 的 0.571 和 $`1/2=0.5`$ 接近但不相等。

若数据增 4 倍：

- 平方根规律给 $`4^{0.5}=2`$；
- 0.571 规律给约 2.21。

相差 $`2.21/2-1=10.5\%`$。大规模外推时不能把 0.571 偷换成 0.5。

### 10.6 Robustness 只在测过的轴上成立

PDF p.37 把规则拿到 MoE 或另一数据设置检验。若点仍接近低谷，可说“在这些测试上较稳健”；不能说跨 tokenizer、optimizer、architecture 都一定迁移。视频 [36:45](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2205s) 开始 robustness；[38:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2281s) 收束。

---

<a id="11-optimizer是一个会污染scaling结论的变量"></a>
## 11. Optimizer 是一个会污染 scaling 结论的变量

### 11.1 AdamW 从零解释

**AdamW** 为每个参数保存一阶动量 $`m`$ 和二阶平方平均 $`v`$，再把 weight decay 与梯度更新解耦。简化式：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
```

```math
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
```

$`\lambda`$ 是 decoupled weight decay；$`\epsilon`$ 防止除零；帽子表示对初始偏差做修正。

只演示一维、不做帽子修正：$`g=2,m_0=v_0=0,\beta_1=0.9,\beta_2=0.99`$：

```math
m_1=0.1\times2=0.2,\qquad
v_1=0.01\times4=0.04.
```

若 $`\epsilon`$ 很小，

```math
\frac{m_1}{\sqrt{v_1}}\approx\frac{0.2}{0.2}=1.
```

这显示 Adam 会按历史平方梯度调整坐标尺度；真实实现还会 bias correction。

视频 [39:32](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2372s) 转入 optimizer；[41:39](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2499s) 开始比较研究。

### 11.2 不公平调参能制造两倍差距

PDF p.39 的课程例里，同一个 AdamW 仅 LR 从 $`6\times10^{-4}`$ 换到 $`8\times10^{-3}`$，达到目标 loss 的 steps 可相差约 2 倍。它说明：

> 若 A optimizer 用默认 LR，B optimizer 用精调 LR，所谓“B 快两倍”可能主要是调参差异。

视频 [42:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2525s) 看速度曲线；[43:24](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2604s) 提醒 wall-clock 与 steps 都要看；[43:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2639s) 开始 confound。

### 11.3 step efficiency 与 wall-clock 分开

假设：

| optimizer | 到目标 loss steps | 每 step 时间 | 总时间 |
|---|---:|---:|---:|
| A | 1000 | 1.0 s | 1000 s |
| B | 800 | 1.4 s | 1120 s |

B 用的 steps 少 20%，却慢 120 秒。原因可能是 optimizer 额外矩阵操作。报告“更快”必须说是少 steps、少 tokens、少 FLOPs，还是少 wall-clock。

### 11.4 scale 与 Chinchilla ratio 也会混进来

PDF p.38–40 的图显示某些 optimizer 相对 AdamW 的优势随模型规模从约 130M 到 1.2B 下降；课程还问 Chinchilla ratio 改变是否影响排序。视频 [45:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2700s) 讨论模型规模；[46:03](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2763s) 讨论 data/model ratio；[47:20](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2840s) 总结。

不能由此得出“优势最终一定归零”。现有点可能：

- 覆盖规模太窄；
- 每个规模 LR 没完全重调；
- data/model ratio 同时变；
- per-step kernel overhead 在小模型占比更大。

### 11.5 外推失败的形状

PDF p.41 展示小规模拟合继续外推后，大规模 run 可能偏离甚至发散。视频 [47:23](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2843s) 开始错误外推；[48:33](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2913s) 说必须留验证；[49:14](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2954s) 转入 Muon。

实践规则中的 **held-out scale（留出规模）**是完全不参与拟合和调参、只用于检验外推的模型规模：

1. 用一部分规模拟合；
2. 用更大但仍负担得起的规模验证；
3. 验证通过才外推目标；
4. 大 run 前做短稳定性试跑；
5. 记录 loss spike、gradient norm、activation norm，而不只看最终 loss。

---

## 12. Muon：矩阵更新的方向整形

### 12.1 课程公式逐符号

【课程内容，PDF p.42】

```math
B_0=0,
\qquad G_t=\nabla_\theta\mathcal L_t,
```

```math
B_t=\mu B_{t-1}+G_t,
```

```math
O_t=\mathrm{NewtonSchulz5}(B_t),
```

```math
\theta_t=\theta_{t-1}-\eta O_t.
```

- $`G_t`$：当前矩阵参数的 gradient；
- $`B_t`$：带 momentum 的梯度矩阵；
- $`\mu`$：momentum 系数；
- `NewtonSchulz5`：课程代码/伪代码中的函数或近似名称，用 Newton–Schulz 类多项式迭代把矩阵奇异值推向 1；函数名中的 `5` 不能单独证明“恰好迭代 5 步”，具体多项式与步数必须查对应实现；
- $`O_t`$：用于更新的近似正交化矩阵。

视频 [49:21](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2961s) 定义 Muon；[50:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3001s) 说明先忽略后面的正交化行；[50:52](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3052s) 进入 Newton–Schulz。

### 12.2 SVD 只用二维直觉

**SVD（singular value decomposition，奇异值分解）**把矩阵写成：

```math
B=U\Sigma V^\top.
```

$`\Sigma`$ 的非负对角数是 singular values，表示不同方向被放大多少。Muon 的直觉目标是从 $`U\Sigma V^\top`$ 得到接近 $`UV^\top`$，保留方向，弱化奇异值大小差异。

最简单矩阵：

```math
B=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}.
```

它的 $`U=V=I`$，$`\Sigma=\mathrm{diag}(3,1)`$。理想正交化后：

```math
UV^\top=I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}.
```

原更新在第一轴是第二轴 3 倍；整形后两轴都是 1。真实 Muon 不会每步做昂贵精确 SVD，而用 Newton–Schulz 近似。

视频 [51:22](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3082s) 讲 $`U\Sigma V^\top`$；[52:11](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3131s) 转到 spectral norm；[52:48](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3168s) 提醒实现成本。

### 12.3 哪些参数适合

Muon 主要针对二维矩阵参数。bias、norm gain、embedding 等其它形状常用 AdamW 或不同规则；具体 optimizer parameter groups 必须记录。视频 [55:34](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3334s) 讨论 per-layer/per-tensor 处理；[56:13](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3373s) 强调实现选择。

### 12.4 “works at scale”能说到哪里

【视频补充】PDF p.43 列出 nanoGPT 小实验、scaling study、Kimi K2 等证据。课程结论是 Muon 已被带到较大规模；不是说：

- 每个任务都优于 AdamW；
- 不用重调 LR；
- wall-clock 必然更短；
- 旧 Muon 细节和后来大规模实现完全相同。

视频 [52:51](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3171s) 看规模证据；[54:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3242s) 提醒实现演化；[55:03](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3303s) 回答 SVD 问题。

【延伸】Kimi K2 报告使用的是公开描述的 MuonClip 变体，不应把课件的简化 Muon 四行式当完整生产 recipe。来源仍以 [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534) 为准。

### 12.5 Lion 和 weight decay 预告

**Lion** 使用 momentum 的符号决定更新方向，状态比 Adam 少；但 LR、weight decay 的合适范围不同。PDF p.39 的课程例提到 Lion 在其设置中 weight decay 约 0.6；这不是通用默认值。来源：[Lion 一手论文](https://arxiv.org/abs/2302.06675)。

**Weight decay** 在 AdamW 中每步把参数乘 $`1-\eta\lambda`$。例：$`\theta=10,\eta=0.1,\lambda=0.2`$，只看 decay：

```math
\theta_{\text{new}}=(1-0.1\times0.2)10
=0.98\times10=9.8.
```

宽度变化若令有效 $`\eta`$ 改变，固定 $`\lambda`$ 的每步缩小率也会改；§18 会解释为什么这可能破坏 μP transfer。

---

<a id="13-μp要解决什么standard-parameterization哪里漂"></a>
## 13. μP 要解决什么：standard parameterization 哪里漂

### 13.1 名字与目标

**μP（Maximal Update Parameterization，最大更新参数化）**是一套规定“不同宽度下怎样初始化、怎样缩放每类参数的 LR 和输出”的规则。希腊字母 μ 读 “mu”。视频 [57:22](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3442s) 正式开始 μP；[57:54](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3474s) 说明目标是让小模型的最佳超参数迁到大模型。

它想同时满足：

- **A1：activation stability**。初始化后，每个 activation 坐标保持 $`\Theta(1)`$；
- **A2：update stability**。一步训练引起的每个 activation 坐标变化保持 $`\Theta(1)`$。

“Maximal”不是让数值越大越好，而是在不让网络随宽度发散的前提下，让更新保持尽可能不消失的量级。

### 13.2 一层线性层地图

```math
h_l=W_l h_{l-1}.
```

形状：

- $`h_{l-1}\in\mathbb R^{n_{l-1}}`$：输入列向量；
- $`W_l\in\mathbb R^{n_l\times n_{l-1}}`$：权重；
- $`h_l\in\mathbb R^{n_l}`$：输出。

Tiny 例：

```math
W=
\begin{bmatrix}
1&2&0\\
-1&0&1
\end{bmatrix},
\quad
h=
\begin{bmatrix}1\\2\\3\end{bmatrix}.
```

第一行点积 $`1(1)+2(2)+0(3)=5`$，第二行 $`-1(1)+0(2)+1(3)=2`$，故

```math
Wh=\begin{bmatrix}5\\2\end{bmatrix}.
```

输入宽度 3、输出宽度 2；不是所有层都方阵。

### 13.3 Standard parameterization 的直觉问题

**Standard parameterization（标准参数化）**通常让矩阵权重标准差约 $`1/\sqrt{\text{fan-in}}`$，并对很多参数使用宽度无关的 base LR。**Fan-in** 是每个输出接收多少输入，即 $`n_{l-1}`$；**fan-out** 是输出坐标数，即 $`n_l`$。

初始化可能保持 forward activation，但训练更新还会聚合许多坐标。尤其 Adam 把每个坐标的 gradient 归一化后，若矩阵实际 LR 不随输入宽度下降，整张更新矩阵的作用会随宽度变大，于是最佳 base LR 会漂。

PDF p.44 的图表现为：

- standard parameterization 不同 width 的最低点横向移动；
- μP 曲线的低点更接近同一个 base LR。

视频 [58:09](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3489s) 说允许调整初始化；[58:31](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3511s) 进入 Cerebras-GPT 案例。

### 13.4 base LR 与 effective tensor LR

设 base width $`P=4`$，当前 width $`M=16`$，width multiplier：

```math
r=\frac MP=\frac{16}4=4.
```

PDF p.44 的 AdamW 教学表对“matrix-like hidden weights”做：

- matrix LR：base LR 再除 $`r`$；
- matrix initialization variance：再除 $`r`$；
- output multiplier：再除 $`r`$；
- 其它参数可能保持 base rule。

若 base LR $`=0.008`$，矩阵实际 LR：

```math
\eta_W=\frac{0.008}{4}=0.002.
```

宽度 4 时实际 LR 0.008，宽度 16 时 0.002；迁移的是“0.008 这个 base knob 的最佳位置”，不是每个矩阵实际 LR 都不变。

### 13.5 课件 $`\sigma`$ 记号有一次过载

PDF p.44 表头把初始化的 $`\sigma`$ 说成 variance；PDF p.47 公式里的 $`\sigma`$ 明确出现在 $`W_{ij}\sim\mathcal N(0,\sigma^2)`$，所以那里 $`\sigma`$ 是 standard deviation（标准差）。

若 variance 除 $`r=4`$，standard deviation 只除 $`\sqrt4=2`$。例：原 variance $`0.04`$，std $`0.2`$；新 variance $`0.01`$，std $`0.1`$。不能把 variance 和 std 都除 4。

### 13.6 Cerebras-GPT 支持什么

PDF p.45 展示 Cerebras-GPT 约 0.1B–13B 的 Chinchilla 风格模型家族，以及用约 40M proxy 调 μP 后迁移到约 2.7B 的实验。视频 [59:07](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3547s) 说他们用 μP 变体调超参数；[59:55](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3595s) 说它是证据之一。

它支持“在该架构和实现里做到了较稳定迁移”，不支持“任意新架构不用再调”。一手来源：[Cerebras-GPT](https://arxiv.org/abs/2304.03208) 和 [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466)。

---

## 14. A1：从 p47 推出初始化后 activation 不爆不消失

### 14.1 先定义向量长度与矩阵谱范数

向量欧氏长度：

```math
\|x\|_2=\sqrt{x_1^2+\cdots+x_n^2}.
```

例如 $`x=[3,4]`$：

```math
\|x\|_2=\sqrt{3^2+4^2}=\sqrt{25}=5.
```

矩阵的 **spectral norm（谱范数）** $`\|W\|_*`$ 是它对某个方向最多能把向量长度放大多少：

```math
\|Wx\|_2\le\|W\|_*\|x\|_2.
```

对对角矩阵 $`\mathrm{diag}(3,1)`$，最大放大倍数是 3，所以谱范数是 3。

### 14.2 随机矩阵的尺度

【课程内容，PDF p.47 高清核对】设

```math
W_l\sim\mathcal N(0,\sigma^2 I_{n_l\times n_{l-1}}).
```

这句话表示 $`W_l`$ 有 $`n_l`$ 行、$`n_{l-1}`$ 列，每个元素均值 0、标准差 $`\sigma`$。对大的独立高斯矩阵，课程使用近似：

```math
\|W_l\|_*\longrightarrow
\sigma\left(\sqrt{n_{l-1}}+\sqrt{n_l}\right).
```

它描述典型大矩阵极限，不是 2×2 小矩阵的精确等号。

视频 [60:39](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3639s) 给 A1/A2；[62:51](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3771s) 开始 p47 推导。

### 14.3 目标：输入、输出每个坐标都约为 1

若 $`h_{l-1}`$ 有 $`n_{l-1}`$ 个坐标，每个坐标 $`\Theta(1)`$，则总长度：

```math
\|h_{l-1}\|_2=\Theta(\sqrt{n_{l-1}}).
```

输出有 $`n_l`$ 个坐标，也希望每个 $`\Theta(1)`$，所以希望：

```math
\|h_l\|_2=\Theta(\sqrt{n_l}).
```

因为 $`h_l=W_lh_{l-1}`$，足够的 operator-scale 目标是：

```math
\|W_l\|_*
=\Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right).
```

它说矩阵最多把输入长度 $`\sqrt{n_{l-1}}`$ 放到输出长度 $`\sqrt{n_l}`$ 的量级。

### 14.4 解出 $`\sigma`$，不跳步

令随机矩阵谱范数近似等于目标：

```math
\sigma(\sqrt{n_{l-1}}+\sqrt{n_l})
=\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}.
```

两边除以括号：

```math
\boxed{
\sigma
=\frac{\sqrt{n_l}}
{\sqrt{n_{l-1}}(\sqrt{n_l}+\sqrt{n_{l-1}})}
}.
```

等价写法：

```math
\boxed{
\sigma
=\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\frac1{\sqrt{n_l}+\sqrt{n_{l-1}}}
}.
```

量级写法：

```math
\sigma=
\Theta\left(
\frac1{\sqrt{n_{l-1}}}
\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right).
```

这里 $`\min(a,b)`$ 取较小者。

### 14.5 方阵数字例

$`n_{l-1}=n_l=100`$：

```math
\sigma
=\frac{10}{10(10+10)}
=\frac{10}{200}
=0.05
=\frac1{2\sqrt{100}}.
```

谱范数近似：

```math
0.05(10+10)=1.
```

输入长度约 10，输出长度上界量级约 $`1\times10=10`$，对应 100 个输出坐标每个约常数。

常见初始化写 $`1/\sqrt{100}=0.1`$，差一个常数 2。Θ 记号忽略固定常数，所以两者都是 $`\Theta(1/\sqrt n)`$；具体常数仍会影响训练。

### 14.6 宽变窄数字例

$`n_{l-1}=100,n_l=25`$：

```math
\sqrt{n_{l-1}}=10,\qquad\sqrt{n_l}=5,
```

```math
\sigma=\frac5{10(5+10)}
=\frac5{150}
=\frac1{30}\approx0.0333.
```

谱范数近似：

```math
\frac1{30}(10+5)=0.5.
```

输入长度量级约 10 时，谱范数给出的**最坏方向上界/设计目标量级**是 $`0.5\times10=5=\sqrt{25}`$。随机 $`h`$ 未必对齐最大奇异向量，所以这不是宣称实际输出长度一定等于 5。若盲用 $`1/\sqrt{\text{fan-in}}=0.1`$，谱范数近似 $`0.1(15)=1.5`$，最坏方向上界量级变成 15，明显大于设计目标 5。

### 14.7 窄变宽数字例

$`n_{l-1}=25,n_l=100`$：

```math
\sigma=\frac{10}{5(10+5)}
=\frac{10}{75}
=0.1333.
```

谱范数近似 $`0.1333(5+10)=2`$。输入长度量级约 5 时，最坏方向上界/设计目标量级是 $`2\times5=10=\sqrt{100}`$；实际随机输入未必取到 10。

### 14.8 这不是说每次都达到上界

$`\|Wh\|\le\|W\|_*\|h\|`$ 是最坏方向上界。随机 $`h`$ 通常不完全对齐最大奇异向量。课程 p47 用 spectral scaling 建立可组合的量级条件；不是声称每个 batch 的 activation norm 精确等于 $`\sqrt{n_l}`$。

视频 [63:35](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3815s) 说谱范数式在高维 regime 近似成立；[64:15](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3855s) 开始代入 $`\sigma`$ 假设；[65:15](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3915s) 收束 A1。

---

## 15. A2：从 p48–49 推出一次更新也保持常数量级

### 15.1 单样本深线性层下，gradient 才是 rank-one outer product

仍看

```math
h_l=W_lh_{l-1}.
```

先把本节结论成立的条件一次写全：

1. 先看一个样本、一个深线性网络层；
2. optimizer 是最朴素 SGD，更新是同一个标量 LR 乘 raw gradient；
3. 本层 $`G_l=\nabla_{W_l}\ell=\delta_lh_{l-1}^{\top}`$ 是 rank one（秩一）；
4. 用一阶 Taylor 近似，忽略二阶及更高阶项；
5. 不同层和不同 leading terms（主导量级项）之间没有严重 cancellation（互相抵消）；
6. 设计目标是一步 loss change 保持 $`\Theta(1)`$，而不是随宽度趋零或爆大。

离开这些条件，下面的等式桥不能原封不动使用。

令

```math
\delta_l=\nabla_{h_l}\ell
```

表示 loss 对输出 activation 的 gradient。则权重 gradient：

```math
\nabla_{W_l}\ell
=\delta_l h_{l-1}^{\top}.
```

为缩短式子，以下记

```math
G_l:=\nabla_{W_l}\ell.
```

这是 **outer product（外积）**：长度 $`n_l`$ 的列向量乘长度 $`n_{l-1}`$ 的行向量，得到 shape $`[n_l,n_{l-1}]`$ 的矩阵。

Tiny 例：

```math
\delta=\begin{bmatrix}2\\-1\end{bmatrix},
\quad
h=\begin{bmatrix}3\\4\end{bmatrix}.
```

```math
\delta h^\top
=
\begin{bmatrix}2\\-1\end{bmatrix}
\begin{bmatrix}3&4\end{bmatrix}
=
\begin{bmatrix}
6&8\\
-3&-4
\end{bmatrix}.
```

视频 [65:18](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3918s) 开始 A2；[65:49](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3949s) 写 outer product。

**Batch 边界：**batch gradient 是多个外积的平均，

```math
G_l^{\text{batch}}
=\frac1B\sum_{b=1}^{B}\delta_l^{(b)}(h_{l-1}^{(b)})^\top.
```

多个 rank-one 矩阵相加通常不再 rank one，因此后文
$`\|G_l\|_F=\|G_l\|_*`$ 的精确等式一般不再成立。课程 p.48–49 是为了讲宽度量级的单样本简化，不是完整 minibatch 定理。

### 15.2 一步更新怎样改变输出

在上述 SGD 条件下：

```math
\boxed{\Delta W_l=-\eta_lG_l}.
```

参数更新后，输出变化包含：

```math
\Delta h_l
=W_l\Delta h_{l-1}
+\Delta W_l(h_{l-1}+\Delta h_{l-1}).
```

第一项是前一层变化经旧权重传来；第二项是本层新权重造成的直接变化。为让直接变化的每坐标 $`\Theta(1)`$，仿照 A1，希望：

```math
\boxed{
\|\Delta W_l\|_*
=\Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right)
}.
```

因为再乘 $`\|h_{l-1}\|=\Theta(\sqrt{n_{l-1}})`$，最坏方向上界/设计目标量级得到 $`\Theta(\sqrt{n_l})`$，即每坐标 $`\Theta(1)`$。在单样本 rank-one 情况，$`\Delta W_l\propto\delta_lh_{l-1}^{\top}`$ 的右奇异方向正是 $`h_{l-1}`$ 的方向，所以对这里的同一个 $`h_{l-1}`$ 可达到对应 rank-one 等式；一般 batch 矩阵或别的输入不能这样说。

### 15.3 为什么还要约束 loss 一步变化

小更新的一阶 Taylor 近似：

```math
\Delta\ell
\approx
\langle G_l,\Delta W_l\rangle_F.
```

尖括号是矩阵对应元素相乘再求和。例：

```math
A=\begin{bmatrix}1&2\\0&-1\end{bmatrix},
\quad
B=\begin{bmatrix}3&4\\5&6\end{bmatrix},
```

```math
\langle A,B\rangle
=1(3)+2(4)+0(5)+(-1)(6)
=3+8-6=5.
```

**Frobenius norm（F 范数）**把矩阵所有元素平方后求和再开根：

```math
\|A\|_F=\sqrt{\sum_{i,j}A_{ij}^2}.
```

例如 $`A=[[1,2],[0,-1]]`$，$`\|A\|_F=\sqrt{1+4+0+1}=\sqrt6`$。

现在把 SGD 更新逐字代入，不跳过中间等式：

```math
\begin{aligned}
\langle \Delta W_l,G_l\rangle_F
&=\langle-\eta_lG_l,G_l\rangle_F\\
&=-\eta_l\sum_{i,j}(G_l)_{ij}^2\\
&=\boxed{-\eta_l\|G_l\|_F^2}.
\end{aligned}
```

负号表示按一阶近似朝降低 loss 的方向走；讨论量级时看绝对值。

对任意 rank-one 矩阵 $`ab^\top`$，它只有一个非零奇异值，因此

```math
\|ab^\top\|_F=\|ab^\top\|_*=\|a\|_2\|b\|_2.
```

单样本下 $`G_l=\delta_lh_{l-1}^\top`$ 是 rank one，$`\Delta W_l=-\eta_lG_l`$ 也 rank one，所以此处才有

```math
\|G_l\|_F=\|G_l\|_*,
\qquad
\|\Delta W_l\|_F=\|\Delta W_l\|_*.
```

于是，在“一阶项主导且没有严重 cancellation”的条件下，希望一步 loss change 的绝对量级为 $`\Theta(1)`$，就可把

```math
|\Delta\ell|
\approx\eta_l\|G_l\|_F^2
=\|\Delta W_l\|_F\|G_l\|_F
```

连接到 rank-one spectral 量级。若：

```math
\|\Delta W_l\|_*
=\Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right),
```

则相配的 gradient norm 量级应为：

```math
\boxed{
\|\nabla_{W_l}\ell\|_*
=\Theta\left(\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}\right)
}.
```

两者相乘：

```math
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}
=1.
```

这条“spectral 乘积为 1”的桥依赖 rank-one。对 batch gradient，它通常非 rank-one；对 Adam，更新还经过 momentum 和逐坐标归一化，并不是 $`-\eta`$ 乘同一张 raw gradient，所以也不能使用同一条精确等式。

### 15.4 SGD LR 逐步推出

在 §15.1 的全部条件下，SGD 满足：

```math
\|\Delta W_l\|_*=\eta_l\|\nabla_{W_l}\ell\|_*.
```

代入目标：

```math
\eta_l
\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}
=\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}.
```

两边乘 $`\sqrt{n_l}/\sqrt{n_{l-1}}`$：

```math
\boxed{\eta_l=\Theta\left(\frac{n_l}{n_{l-1}}\right)}.
```

数字例：输入宽 100、输出宽 25：

```math
\eta\propto\frac{25}{100}=0.25.
```

若输入宽 25、输出宽 100：

```math
\eta\propto\frac{100}{25}=4.
```

这只给宽度 multiplier；共同常数 base LR 仍靠调参。若换成 batch gradient、非线性网络、Adam 或存在主导项抵消，这个 baby derivation 只能作量级直觉，必须回到完整理论或实验验证。

视频 [66:23](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3983s) 解释直接变化；[67:28](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4048s) 给 desired norm；[68:22](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4102s) 解 loss-change 条件。

### 15.5 Adam LR 为什么是 $`1/n_{\text{in}}`$

Adam 对每个坐标用历史平方 gradient 归一化后，教学直觉是矩阵 raw update 的许多元素都在常数量级。极端易算例：$`n_l\times n_{l-1}`$ 的 raw update 每个元素都是 1。

全 1 矩阵的谱范数是：

```math
\sqrt{n_l n_{l-1}}.
```

乘 LR $`\eta`$ 后：

```math
\|\Delta W\|_*=\eta\sqrt{n_ln_{l-1}}.
```

令它等于 A2 目标：

```math
\eta\sqrt{n_ln_{l-1}}
=\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}.
```

两边除以 $`\sqrt{n_l}`$，得到：

```math
\eta\sqrt{n_{l-1}}
=\frac1{\sqrt{n_{l-1}}},
```

```math
\boxed{\eta=\Theta(1/n_{l-1})}.
```

若 fan-in 从 4 增到 16，矩阵有效 Adam LR 应乘：

```math
\frac{1/16}{1/4}=\frac14.
```

这正是 base width multiplier $`r=4`$ 时矩阵 LR 除 4 的来源。

边界：真实 Adam update 不是全 1，也可能有相关结构；这个小例用于解释宽度阶数，正式结论来自 tensor-program 极限分析。

### 15.6 一维参数不能盲用矩阵规则

Norm gain shape $`[M]`$、bias shape $`[M]`$、embedding shape $`[V,M]`$、output layer shape $`[M,V]`$ 在网络中的角色不同。直接对所有 tensor 统一除 $`M`$ 会破坏其它路径。μP 实现必须按参数类型分类。

视频 [69:04](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4144s) 对比 SGD/Adam；[70:20](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4220s) 完成 p49；[70:24](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4224s) 开始复盘。

---

## 16. Maximal update、tensor programs 与“为什么能迁移”

### 16.1 p50 公式卡

【课程内容，PDF p.50 高清核对】一般矩阵层的量级：

```math
\boxed{
\sigma_l
=\Theta\left(
\frac1{\sqrt{n_{l-1}}}
\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right)
}
```

```math
\boxed{
\eta_l^{\mathrm{SGD}}
=\Theta(n_l/n_{l-1})
}
```

```math
\boxed{
\eta_l^{\mathrm{Adam}}
=\Theta(1/n_{l-1})
}.
```

标准参数化常见粗规则是 $`\sigma=\Theta(1/\sqrt{n_{l-1}})`$、LR $`\Theta(1)`$。当 fan-out 远小于 fan-in，μP 初始化多出的 $`\sqrt{n_l/n_{l-1}}`$ 因子重要；对于 Adam hidden matrix，LR 的 $`1/n_{\text{in}}`$ 更是 transfer 核心。

### 16.2 为什么旧 base LR 还能迁

把用户调的 base LR 记作 $`\eta_{\text{base}}`$，系统按 width 产生 multiplier $`m_W(M)`$：

```math
\eta_W(M)=\eta_{\text{base}}m_W(M).
```

例：base width $`P=256`$：

| 当前 width $`M`$ | $`r=M/P`$ | Adam matrix multiplier $`1/r`$ | base LR 0.004 对应实际 LR |
|---:|---:|---:|---:|
| 256 | 1 | 1 | 0.004 |
| 512 | 2 | 1/2 | 0.002 |
| 1024 | 4 | 1/4 | 0.001 |

网络看到的有效更新量级被 multiplier 抵消了宽度增长，因此 loss 对 **base LR** 的最佳区域可能对齐。视频 [70:55](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4255s) 总结公式；[71:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4304s) 转入直觉。

### 16.3 Maximal 的意思

若更新缩得更快，例如 Adam matrix LR 用 $`1/n^2`$，宽度增大后 activation update 可能趋 0，模型在有限训练时间学不动。若缩得更慢，例如常数 LR，update 可能爆大。$`1/n`$ 是保持非零且不爆的最大量级之一，所以叫 maximal update。

数字例，令 raw update 谱范数约 $`n`$：

- LR $`1`$：更新 norm $`n`$，爆大；
- LR $`1/n`$：更新 norm 1，常数量级；
- LR $`1/n^2`$：更新 norm $`1/n`$，趋零。

### 16.4 Tensor programs 的直觉

**Tensor Programs（张量程序）**是一套分析宽度趋于无穷时，网络中矩阵乘、非线性、残差和梯度的随机极限如何传播的数学框架。它不是一段自动替你训练的程序。

直觉流程：

1. 为每类 tensor 写 shape；
2. 假设每坐标的量级；
3. 看求和跨多少个宽度坐标；
4. 选择初始化和 LR 的宽度幂，抵消这些求和；
5. 检查 forward 和 update 在宽度极限都不爆、不消失。

视频 [60:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3601s) 提到 Tensor Programs；[60:23](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3623s) 提到谱范数视角的综述；[71:49](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4309s) 用“features 能真正学习”解释 maximal。

### 16.5 μP 不迁移什么

μP 主要针对 width scaling 下的超参数稳定。它不保证迁移：

- batch size；
- token 数或 data mixture；
- depth；
- tokenizer/vocabulary；
- optimizer 类型；
- weight decay、dropout 等 regularization；
- architecture 从 dense 变 MoE；
- attention head dimension 改变；
- kernel、并行布局或硬件吞吐。

**Regularization（正则化）**是改变训练以约束参数或表示的机制，例如 weight decay、dropout。它可能与宽度 multiplier 交互。

---

## 17. Transformer 里 μP 不是“所有矩阵一条式”

### 17.1 p51 的符号先解释

PDF p.51 引用 “A Large-Scale Exploration of μ-Transfer” 的 Transformer 表。该表使用：

- $`M`$：当前模型主宽度；
- $`P`$：proxy/base 模型主宽度；
- $`H`$：attention heads；
- $`D_h`$：每个 head 的 dimension；为避免和 data tokens 的 $`D`$ 混淆，本笔记加下标。课件 p.51 原表写 $`D`$；
- $`F`$：FFN width；
- $`W^E`$：embedding；
- $`W^{AQ},W^{AK},W^{AV},W^{AO}`$：attention Q/K/V/output 矩阵；
- $`W^{FI},W^{FO}`$：FFN input/output 矩阵；
- $`W^U`$：unembedding/output head。

原表的具体常数依赖该论文/课件实现。最重要的口径警告：

- p.51 两列都明确写 **Init Variance**，所以表中 $`1/M`$ 是方差，不是标准差；
- 若 variance $`=1/M`$，对应 standard deviation 是 $`1/\sqrt M`$；
- $`\Theta`$ 列只说随宽度的阶；Exact 列还保留该实现选择的 $`0.25`$、base width $`P`$ 等常数。

### 17.2 从 §13 的列向量约定桥到 Transformer 代码的行向量约定

§13 用列向量写线性层：

```math
h_{\text{out}}[M,1]=W[M,N]h_{\text{in}}[N,1].
```

Transformer/PyTorch 叙述常把一个 token 的向量放成行。**One-hot（独热向量）**只有当前 token ID 位置为 1，其余为 0：

```math
\mathrm{onehot}[1,V]\ @\ W^E[V,M]
\longrightarrow h[1,M].
```

输出时：

```math
h[1,M]\ @\ W^U[M,V]
\longrightarrow \mathrm{logits}[1,V].
```

例：$`V=3,M=2`$，token ID=1：

```math
\mathrm{onehot}=[0,1,0],\qquad
W^E=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}.
```

```math
[0,1,0]W^E=[3,4].
```

若

```math
W^U=
\begin{bmatrix}
1&0&-1\\
2&1&0
\end{bmatrix},
```

```math
[3,4]W^U
=[3(1)+4(2),\ 3(0)+4(1),\ 3(-1)+4(0)]
=[11,4,-3].
```

若坚持使用列向量，同样映射要转置矩阵：

```math
h_{\text{col}}[M,1]=(W^E)^\top[M,V]\mathrm{onehot}_{\text{col}}[V,1],
```

```math
\mathrm{logits}_{\text{col}}[V,1]=(W^U)^\top[V,M]h_{\text{col}}[M,1].
```

数值没变，只是 shape 与乘法方向转置。后面 p.51 参数 shape 统一采用行向量代码口径。

### 17.3 p51 Transformer 简化完整表：阶数和具体常数分开

【课程内容，PDF p.51 高清逐格核对】

先把这张表的两个**具体实现假设**写在最前面，否则 exact 常数会像凭空出现：

```math
H D_h=M,
\qquad
F=4M.
```

也就是说，所有 attention heads 拼起来的总宽度等于模型主宽度，而 FFN 中间宽度取主宽度的 4 倍。因此：

```math
\frac{1}{H D_h}=\frac{1}{M},
```

```math
\frac{1}{F}=\frac{1}{4M}=\frac{0.25}{M}.
```

表里的 exact Adam LR $`\alpha P/M`$ 也要读清条件：当 base LR $`\alpha`$ 与 proxy width $`P`$ 对当前变化的 $`M`$ 都固定时，$`\alpha P`$ 是常数，所以

```math
\frac{\alpha P}{M}=\Theta(1/M).
```

这里“exact”是这套实现的具体常数，“$`\Theta(1/M)`$”只说宽度变大时的量级。若另一模型用 $`F=3M`$，那么 $`1/F=1/(3M)`$，不再是 $`0.25/M`$；若 $`H D_h\ne M`$，也不能把 $`1/(H D_h)`$ 直接抄成 $`1/M`$。

**两步数字自检。**取 $`H=8,D_h=64,M=512,F=2048`$：

1. $`H D_h=8\times64=512=M`$，所以 $`1/(H D_h)=1/512=1/M`$。
2. $`F=2048=4\times512=4M`$，所以 $`1/F=1/2048=1/(4M)=0.25/M`$。

| 参数 | 行向量 shape/角色 | Init variance $`\Theta`$ | Init variance exact | Adam LR $`\Theta`$ | Adam LR exact |
|---|---|---:|---:|---:|---:|
| $`W^E`$ | $`[V,M]`$，token embedding | $`1`$ | $`1`$ | $`1`$ | $`\alpha`$ |
| $`W^{AQ}`$ | $`[M,HD_h]`$，Q projection | $`1/M`$ | $`1/M`$ | $`1/M`$ | $`\alpha P/M`$ |
| $`W^{AK}`$ | $`[M,HD_h]`$，K projection | $`1/M`$ | $`1/M`$ | $`1/M`$ | $`\alpha P/M`$ |
| $`W^{AV}`$ | $`[M,HD_h]`$，V projection | $`1/M`$ | $`1/M`$ | $`1/M`$ | $`\alpha P/M`$ |
| $`W^{AO}`$ | $`[HD_h,M]`$，attention output | $`1/(HD_h)`$ | $`1/M`$ | $`1/(HD_h)`$ | $`\alpha P/M`$ |
| $`W^{FI}`$ | $`[M,F]`$，FFN input | $`1/M`$ | $`1/M`$ | $`1/M`$ | $`\alpha P/M`$ |
| $`W^{FO}`$ | $`[F,M]`$，FFN output | $`1/F`$ | $`0.25/M`$ | $`1/F`$ | $`\alpha P/M`$ |
| $`W^U`$ | $`[M,V]`$，softmax linear/unembedding | $`1/M^2`$ | $`1/M^2`$ | $`1/M`$ | $`\alpha P/M`$ |

这里 $`H`$ 是 head 数，$`D_h`$ 是每 head 维度，$`F`$ 是 FFN 中间宽度，$`\alpha`$ 是用户调的 base Adam LR。Exact 列采用 base width $`P`$，current width $`M`$。

**不要把 variance 当 std。**例如 $`M=512`$，$`W^{AQ}`$ exact variance $`=1/512=0.001953125`$，std 是

```math
\sqrt{1/512}\approx0.04419,
```

不是 0.001953125。

### 17.4 把 $`P\to M`$ 数字逐行代入 actual LR

取 $`P=128,M=512,\alpha=0.004`$，width ratio $`r=M/P=4`$。embedding：

```math
\eta_{W^E}=\alpha=0.004.
```

其余七类参数都用：

```math
\alpha P/M=0.004\times128/512
=0.004/4=0.001.
```

| 参数 | actual Adam LR |
|---|---:|
| $`W^E`$ | 0.004 |
| $`W^{AQ}`$ | 0.001 |
| $`W^{AK}`$ | 0.001 |
| $`W^{AV}`$ | 0.001 |
| $`W^{AO}`$ | 0.001 |
| $`W^{FI}`$ | 0.001 |
| $`W^{FO}`$ | 0.001 |
| $`W^U`$ | 0.001 |

同一组数字下，exact init variances 是：

| 参数组 | exact variance |
|---|---:|
| $`W^E`$ | $`1`$ |
| $`W^{AQ},W^{AK},W^{AV},W^{AO},W^{FI}`$ | $`1/512=0.001953125`$ |
| $`W^{FO}`$ | $`0.25/512=0.00048828125`$ |
| $`W^U`$ | $`1/512^2=1/262144\approx0.0000038147`$ |

这张表是在“该论文/课件实现具体常数”下算 actual LR；$`\Theta`$ 阶只保证随 $`M`$ 的方向，不决定 0.004 这个 base 数。

### 17.5 Output multiplier 与 attention multiplier 不混算

PDF p.44 的通用 μP 表另列 **output multiplier**：若模型宽 $`M'=rM`$，有限维 output layer 的 multiplier 从 $`\tau`$ 变成 $`\tau/r`$。取 $`r=4`$，就是原 multiplier 的 $`1/4`$。p.51 的 exact 表没有另开 output-multiplier 一列，而是明确给 $`W^U`$ 的 variance $`1/M^2`$ 和 LR $`\alpha P/M`$。具体实现若采用显式 output multiplier，要按其参数化检查，不能看到 p.44 与 p.51 就未经核对重复除两次。

p.51 表下方另写 attention score scale：

```math
\tau_{\text{attn}}^{-1}=\Theta(1/D_h),
\qquad
\text{该实验 exact 取 }1/D_h,
```

而标准 scaled dot-product attention 常取 $`1/\sqrt{D_h}`$。若 $`D_h=64`$：

```math
1/D_h=1/64=0.015625,\qquad
1/\sqrt{D_h}=1/8=0.125.
```

前者是后者的 $`0.015625/0.125=1/8`$。这是 forward attention score 的 multiplier，不是 Adam LR。

### 17.6 为什么输入、hidden、输出角色不同

词表大小 $`V`$ 常不随主宽度 $`M`$ 一起变：

- embedding $`W^E`$ 的输入是 fixed-vocabulary one-hot；
- hidden 矩阵两侧可能都随宽度变；
- unembedding $`W^U`$ 从可扩宽 hidden 映射到固定 $`V`$；
- norm gain 是一维参数，不是矩阵。

因此它们需要不同 init/output multiplier/LR 规则。把 embedding 当普通 $`[M,M]`$ 会错。

### 17.7 Attention scale 的边界

标准 scaled dot-product attention 常用：

```math
\frac{q^\top k}{\sqrt{D_h}}.
```

p51 表在某些 μP 宽度极限下给 attention multiplier 量级 $`1/D_h`$。如果实验只增 $`M`$，却固定 $`D_h=64`$，那么 $`1/\sqrt{64}=1/8`$ 和 $`1/64`$ 都是与 $`M`$ 无关的常数，从“随 $`M`$ 的 Θ 阶数”看都不变，但性能和 transfer 仍会受具体常数影响。

Tiny 例：$`q=[1,1],k=[1,1]`$，点积 2。

- 除 $`\sqrt2`$：score $`=1.414`$；
- 除 2：score $`=1`$。

两者都不随其它宽度变化，但 softmax 温度不同。

### 17.8 可迁移实现的检查清单

1. 明确 base shapes 和 width multiplier。
2. 标注每个参数是 input、hidden、output、norm 还是 bias。
3. 记录参数初始化是 variance 还是 standard deviation。
4. 记录用户 base LR 与 optimizer parameter-group 实际 LR。
5. 确认 tied embedding/output（输入 embedding 与输出矩阵共享同一组权重）是否改变角色。
6. 新增 SwiGLU、MoE router、QK norm 时重新推规则。
7. 用小、中、较大三个宽度画 loss-vs-base-LR 曲线。

【补充】μP 的正式规则和可迁移条件见 [Tensor Programs V](https://arxiv.org/abs/2203.03466)；课程的谱范数视角可与 [Spectral Conditions for Feature Learning](https://arxiv.org/abs/2310.17813) 对照。

---

## 18. μP 会在哪里失效或变脆

### 18.1 先读 p52 的成功证据

PDF p.52 的离散搜索列是 $`2^{-10},2^{-8},2^{-6},2^{-4},2^{-2}`$。逐行取加粗的最低 loss：

| Ablation | width 128 | width 512 | width 2048 | 离散网格判断 |
|---|---:|---:|---:|---|
| baseline μP | $`2^{-6}`$ | $`2^{-6}`$ | $`2^{-6}`$ | 三者最低采样列一致 |
| projection biases | $`2^{-6}`$ | $`2^{-6}`$ | $`2^{-6}`$ | 三者最低采样列一致 |

**Projection bias（投影偏置）**是在线性投影结果上再逐坐标加的可训练向量；p.52 这项实验说明加入这类 bias 后，三个 width 的最低采样列仍对齐。

计算：

```math
2^{-6}=\frac1{64}=0.015625.
```

这支持该设置及 projection-bias 变体的 transfer。视频 [72:38](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4358s) 开始 stress tests；[73:35](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4415s) 说正确 μP 下得到 LR 最优性不变。

但“最低采样列相同”不等于连续最优点完全一样；真实最优可能分别是 0.014、0.016、0.017。

### 18.2 新 activation 与 batch

PDF p.53 列 **SwiGLU（两条线性分支中，一条经过 Swish 门函数后与另一条逐元素相乘）**、squared ReLU、batch、初始化等可能偏离。原因不是这些结构一定不兼容，而是：

- gate 让两条分支相乘，量级传播改变；
- squared ReLU 把正值平方，尾部更敏感；
- batch 改 gradient noise；
- 初始化常数改变 early dynamics。

必须重新验证，而不是套旧规则。

### 18.3 RMSNorm gain

**RMSNorm** 用向量均方根归一化，再乘可训练 gain：

```math
\mathrm{RMSNorm}(x)_i
=g_i\frac{x_i}{\sqrt{\frac1M\sum_{j=1}^M x_j^2+\epsilon}}.
```

若 $`x=[3,4]`$，忽略 $`\epsilon`$：

```math
\mathrm{RMS}(x)
=\sqrt{(9+16)/2}
=\sqrt{12.5}\approx3.535.
```

若 $`g=[1,1]`$，输出约 $`[0.849,1.131]`$。

PDF p.54 的课程实验中，可训练 vector 或 scalar gain 让不同 width 的最佳 base LR 漂移，并伤到最大模型；去掉 gain 在该设置损失不大。正确说法是“这是一个实证 failure mode”，不是“RMSNorm 永远不应有 gain”。

逐行最低采样列：

| gain | width 128 | width 512 | width 2048 |
|---|---:|---:|---:|
| vector | $`2^{-4}`$ | $`2^{-4}`$ | $`2^{-8}`$ |
| scalar | $`2^{-4}`$ | $`2^{-4}`$ | $`2^{-6}`$ |

这些是 $`2^{-10},2^{-8},2^{-6},2^{-4},2^{-2}`$ 离散网格中的最低列，不是连续 LR 真最优值。

视频 [74:24](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4464s) 开始列失败项；[74:30](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4470s) 明确 trainable RMSNorm gain 会破坏 transfer。

### 18.4 Lion

PDF p.55 的最低采样列是：

| width | 128 | 512 | 2048 |
|---|---:|---:|---:|
| Lion | $`2^{-10}`$ | $`2^{-8}`$ | $`2^{-8}`$ |

这显示该 Lion 设置的最佳 base LR 随 width 移动。可能原因包括更新是 sign/momentum 型，与 Adam 的 coordinate normalization 和 μP multiplier 假设不同。不能把 Adam 的 $`1/n_{\text{in}}`$ 直接复制给 Lion 后宣布完成。这里仍只是离散网格结论。

视频 [74:41](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4481s) 转入更特殊的 optimizer；[74:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4484s) 点名 Lion。

### 18.5 强 decoupled weight decay

AdamW 的 decay factor：

```math
1-\eta_W\lambda.
```

若宽度增 4 倍，μP 让矩阵实际 $`\eta_W`$ 除 4，但 $`\lambda`$ 固定：

- base：$`\eta=0.004,\lambda=0.1`$，factor $`=1-0.0004=0.9996`$；
- 宽模型：$`\eta=0.001,\lambda=0.1`$，factor $`=1-0.0001=0.9999`$。

同样 step 数下，宽模型受到的 decay 更弱。这会改变最佳 base LR/regularization 配合。PDF p.56 的 weight decay 0.1 是课程实验中显著 failure；不是所有较小 decay 都必然失败。

p.56 strong decoupled weight decay 的最低采样列：

| width | 128 | 512 | 2048 |
|---|---:|---:|---:|
| strong WD 0.1 | $`2^{-8}`$ | $`2^{-6}`$ | $`2^{-6}`$ |

它说明三条离散最低列没有完全对齐；不能从这三点拟合普遍的 weight-decay scaling law。

视频 [74:58](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4498s) 进入 decoupled weight decay；[75:06](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4506s) 总结这是显著 stress-test failure。

### 18.6 p57 的正反对照

PDF p.57 两张表的最低采样列：

| parameterization | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| standard | $`2^{-6}`$ | $`2^{-8}`$ | $`2^{-10}`$ | 未列 |
| μP large-scale | $`2^{-6}`$（2M） | $`2^{-6}`$（40M） | $`2^{-6}`$（600M） | $`2^{-6}`$（10B） |

数值：

```math
2^{-6}=0.015625,\quad
2^{-8}=0.00390625,\quad
2^{-10}=0.0009765625.
```

standard 的离散例子每宽 4 倍，最低采样 LR 约除 4；μP 把这个变化放进内部 multiplier，所以四行最低采样列对齐。因为只扫了有限列，仍不能说连续最优值精确相等。

视频 [75:23](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4523s) 说明 standard 下最佳 LR 随 width 可预测地移动；[76:04](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4564s) 批评把 scaling 描述成确定科学；[76:55](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4615s) 结束。

### 18.7 迁移前的最小 stress-test 网格

至少测试：

| 轴 | 三个点示例 |
|---|---|
| width | $`P,2P,4P`$ |
| base LR | $`\eta/2,\eta,2\eta`$ |
| batch | $`B/2,B,2B`$ |
| norm gain | fixed / vector trainable |
| decay | 0 / 小 / 目标值 |
| optimizer | AdamW / 候选 |

画每个 width 的 loss-vs-base-LR。若最低区域不重叠，不要直接外推；先查参数分类、实际 LR、初始化、output multiplier 和新增模块。

---

<a id="19-一套可执行的scaling-recipe"></a>
## 19. 一套可执行的 scaling recipe

### 19.1 先写清目标和口径

在跑实验前填这张表：

| 问题 | 必须记录 |
|---|---|
| 优化什么 | validation loss、downstream、wall-clock、成本或混合目标 |
| compute | 理论 $`6ND`$、真实 FLOPs/token，还是 GPU-hours |
| model size | total、non-embedding、activated parameters |
| data | tokenizer 后 tokens、去重/重复、mixture |
| batch | sequences/update 或 tokens/update，DP 与 accumulation |
| recipe | optimizer、LR、warmup、schedule、weight decay、init |
| systems | dtype、hardware、parallelism、achieved throughput |

没统一口径，拟合小数点再多也没意义。

### 19.2 决策树

1. **架构还没定？**  
   先用便宜模型比较 architecture；同时匹配 FLOPs、参数、数据和调参预算。
2. **只沿 width 扩？**  
   考虑 μP；逐参数分类，先验证 $`P,2P,4P`$ 的 base-LR 曲线。
3. **width、depth、MoE 都改？**  
   μP 只能处理其中一部分；另做 shape/active-param 网格。
4. **不知道 batch/LR？**  
   做二维网格，不要只做两条一维线。
5. **要比较许多训练长度？**  
   用 WSD 复用稳定主干，并验证 decay 长度。
6. **要定 $`N,D`$？**  
   至少两个 IsoFLOP budgets；拟合后用未参与拟合的更大 budget 验证。
7. **要换 optimizer？**  
   各自调 LR、decay、momentum；同时报 steps 与 wall-clock。
8. **准备最大 run？**  
   做短稳定性 run、容错测试、checkpoint 恢复和预先定义的停机指标。

### 19.3 一个完整 toy 计划

目标 compute 约 $`C=6ND=6\times10^{18}`$，先忽略常数单位。候选 $`ND=10^{18}`$：

| $`N`$ params | $`D`$ tokens | $`ND`$ |
|---:|---:|---:|
| $`0.25`$B | 4B | $`1`$B² |
| $`0.5`$B | 2B | $`1`$B² |
| $`1`$B | 1B | $`1`$B² |

对每个候选：

1. batch $`\{0.5,1,2\}`$M tokens/update；
2. base LR $`\{0.001,0.002,0.004\}`$；
3. 每格至少 2 个 random seeds（控制随机初始化和数据抽样的编号）；
4. 先 stable trunk 到 90% tokens；
5. 接 10% decay；
6. 以 validation loss 选低谷，再以吞吐/显存选低谷内的点；
7. 在 2B 模型做 held-out 规模验证。

总实验并不便宜，所以可先用 sequential design：粗 3×3 网格，定位后只加密低谷附近。

### 19.4 证据等级

由弱到强：

1. 单次训练曲线；
2. 同规模多个 LR；
3. 多规模、同 recipe；
4. 多 seed；
5. held-out scale 验证；
6. 换 data/architecture stress test；
7. 目标规模完整 run；
8. downstream 与系统指标也一致。

“论文里有一张漂亮直线”最多只覆盖其中几层。

---

<a id="20-公式卡"></a>
## 20. 公式卡

### 20.1 训练 compute

```math
C\approx6ND.
```

适用：dense Transformer 粗略训练账。  
不含：通信、data movement、硬件利用率和很多小项。

### 20.2 联合 loss law

```math
L(N,D)=L_0+A N^{-\alpha}+B_D D^{-\beta}.
```

每项越大，模型或数据不足的 penalty 越大；$`L_0`$ 是拟合渐近项。

### 20.3 DeepSeek 课程拟合

```math
\eta_{\mathrm{opt}}=0.3118C^{-0.125},
\qquad
B_{\mathrm{opt}}=0.2920C^{0.3271}.
```

只在原论文单位和实验范围解释；跨单位优先算比例。

### 20.4 Step Law 课程表

```math
\eta_{\mathrm{opt}}=1.79N^{-0.713}D^{0.307},
\qquad
B_{\mathrm{opt}}=0.58D^{0.571}.
```

它和 DeepSeek 式自变量不同，不可直接比较指数正负。

### 20.5 WSD 复用成本

```math
C_{\mathrm{total}}=C_{\mathrm{stable}}+mC_{\mathrm{decay}}.
```

线性优势要求分叉数和 decay 设计受控。

### 20.6 A1 初始化

```math
\sigma
=\frac{\sqrt{n_l}}
{\sqrt{n_{l-1}}(\sqrt{n_l}+\sqrt{n_{l-1}})}
=\Theta\left[
\frac1{\sqrt{n_{l-1}}}
\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right].
```

### 20.7 A2 更新

```math
\|\Delta W_l\|_*
=\Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right),
\qquad
\|\nabla_{W_l}\ell\|_*
=\Theta\left(\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}\right).
```

```math
\eta_l^{\mathrm{SGD}}=\Theta(n_l/n_{l-1}),
\qquad
\eta_l^{\mathrm{Adam}}=\Theta(1/n_{l-1}).
```

### 20.8 AdamW decay

```math
\theta\leftarrow(1-\eta\lambda)\theta
-\eta\,\text{adaptive-gradient}.
```

若 effective LR 随 width 变，固定 $`\lambda`$ 的实际每步 decay 也随 width 变。

---

<a id="21-常见误区错误为什么错正确说法"></a>
## 21. 常见误区：错误、为什么错、正确说法

| # | 错误说法 | 为什么错 | 正确说法 |
|---:|---|---|---|
| 1 | Scaling law 是物理定律 | 它来自有限实验拟合 | 它是带范围和误差的经验规律 |
| 2 | 一条 loss 曲线就能选 recipe | 没扫 LR、batch、seed | 至少比较邻域和重复试验 |
| 3 | 参数多就一定更好 | 固定 compute 时数据会减少 | 同时看 $`N,D,C`$ |
| 4 | $`C=6ND`$ 是精确 GPU 时间 | 它只算粗 FLOPs | 另测吞吐、通信和 wall-clock |
| 5 | 固定 aspect ratio 保证最优 | 它只减少搜索轴 | 还要验证 depth/width 选择 |
| 6 | μP 就是把 init 除 $`\sqrt n`$ | 还含 LR、输出和参数分类 | 使用完整参数化 |
| 7 | μP 后所有 tensor LR 一样 | hidden、input、output 角色不同 | base LR 可同，effective LR 可不同 |
| 8 | “LR transfer”指实际矩阵 LR 不变 | multiplier 会随 width 变 | 稳定的是 base knob 的最佳区域 |
| 9 | variance 与 std 是一回事 | variance $`=`$ std² | 先看课件符号口径 |
| 10 | activation $`\Theta(1)`$ 指整向量 norm 为 1 | $`n`$ 个常数坐标的 norm 约 $`\sqrt n`$ | 指每坐标量级 |
| 11 | spectral norm 上界每次取等号 | 输入未必对齐最大方向 | 它是量级/最坏方向工具 |
| 12 | 方阵公式可直接套所有矩阵 | fan-in/out 可能不同 | 用 p47 rectangular 式 |
| 13 | Standard init 一定让 forward 爆 | 方阵 forward 常可稳定 | 常见问题还在 update/参数角色 |
| 14 | Adam LR $`1/n`$ 是凭经验猜 | 可由 normalized matrix update 的谱尺度解释 | 正式结论来自 μP 理论 |
| 15 | SGD 和 Adam 的 multiplier 相同 | 更新结构不同 | p50 给不同宽度阶数 |
| 16 | Maximal 是让 LR 最大 | 是保持不爆时让 feature update 不消失 | 仍需调 base 常数 |
| 17 | μP 能迁 batch | batch 改 gradient noise | batch 需单独拟合 |
| 18 | μP 能迁 depth | 本讲主要分析 width | depth 要另验证 |
| 19 | μP 能自动适配 MoE | router/expert 角色新增 | 需推导并 stress test |
| 20 | WSD 不需要 decay | stable 主干通常还需收尾 | 从 checkpoint 接 decay |
| 21 | WSD 总成本永远线性 | 分叉过多、decay随长度增会回到二次 | 写 $`C+md`$ 再判断 |
| 22 | Stable 阶段 loss 高说明 WSD 差 | 它尚未 decay | 比较相同终点 recipe |
| 23 | Cosine 只能训练一次长度 | 可以重跑但贵 | WSD 的价值是复用主干 |
| 24 | Lower envelope 是理论下界 | 只是已跑点里的最低 | 新配置可能更低 |
| 25 | IsoFLOP 只需一个 compute budget | 一个碗不能拟合随 $`C`$ 的指数 | 需要多个 budgets |
| 26 | Method 3 用所有点所以不会错 | 函数形式错会系统偏 | 用 held-out 验证 |
| 27 | Validation loss 就是 accuracy | 指标不同 | 下游需另拟合/评测 |
| 28 | 低 pretraining loss 必然带来所有能力 | 映射可能 sigmoid、噪声或任务特异 | 谨慎报告相关性 |
| 29 | DeepSeek LR 随 $`C`$ 降证明所有模型如此 | 单一研究范围 | 标注 architecture/data/optimizer |
| 30 | Step Law 与 DeepSeek 指数相反，所以一方错 | 自变量与共同变化量不同 | 统一口径后再比较 |
| 31 | Batch 指数 0.571 就是 0.5 | 外推会累积差异 | 保留原指数和误差 |
| 32 | 二维图最低采样点是真实精确最优 | 网格离散且有噪声 | 在邻域加密并多 seed |
| 33 | 曲面看起来像碗说明神经网络全局凸 | 只观察局部超参数面 | 不推广到参数空间 |
| 34 | 公司没公开的超参数可以从图猜 | 会伪造事实 | 标为未知 |
| 35 | MoE 总参数可直接代 $`6ND`$ | 每 token 只激活部分专家 | 计算账用 activated path |
| 36 | activated params 决定所有成本 | 总权重存储和通信仍受总参影响 | 分开四本账 |
| 37 | Optimizer 少 steps 就更快 | 每 step 可能更贵 | 同时报 wall-clock |
| 38 | 每个 optimizer 用同一 LR 才公平 | 不同算法最佳 LR 不同 | 分别给同等调参预算 |
| 39 | Muon 是精确 SVD | 实现用 Newton–Schulz 近似 | 区分直觉与 kernel |
| 40 | Muon 适合所有参数 | 主要针对矩阵 | 其它参数常用另一 optimizer |
| 41 | Kimi K2 证明原始 Muon 四行式可直接复制 | 实际是演化实现 | 回查一手报告 |
| 42 | Weight decay 只防过拟合，不影响 transfer | 它改变每步参数尺度 | 与 effective LR 联动 |
| 43 | RMSNorm gain 永远破坏 μP | 只是课程实验的 failure mode | 在目标架构复测 |
| 44 | Lion 永远不兼容 μP | 课程只显示某规则未迁 | 需要适配该 optimizer |
| 45 | p52 同一最低列证明最优 LR 完全相等 | 网格分辨率有限 | 只能说最佳区域对齐 |
| 46 | 小模型验证一次就能外推 1000× | 可能跨 regime | 逐级 held-out scale |
| 47 | $`R^2`$（决定系数：拟合对训练点变动的解释比例）高就能信外推 | 插值好不等于外推好 | 检查 residual 与 held-out |
| 48 | 一次大 run 失败只看最终 loss | 早期已有 norm/spike 信号 | 监控稳定性指标 |

---

<a id="22-术语表"></a>
## 22. 术语表

| 术语 | 零基础解释 |
|---|---|
| scaling law | 规模与指标之间的经验拟合关系 |
| scaling recipe | 结构、数据、batch、LR、optimizer、schedule 等整套选择 |
| proxy/base model | 用来便宜调参的小模型 |
| target model | 真正要训练的大模型 |
| hyperparameter | 人先选的训练设置 |
| aspect ratio | width、FFN、heads、depth 的比例 |
| width/depth | 向量宽度/层数 |
| activation | forward 的中间数值 |
| gradient | 参数或 activation 改一点时 loss 的局部变化率 |
| learning rate | 把 gradient 变成多大更新的步长 |
| optimizer | 产生参数更新的算法 |
| AdamW | 带一阶/二阶动量和解耦 weight decay 的 optimizer |
| Muon | 对矩阵 momentum 更新做近似正交化的 optimizer 家族 |
| Lion | 主要按 momentum 符号更新的 optimizer |
| batch | 一次参数更新合计处理的数据 |
| global batch | 所有设备和 accumulation 合计 batch |
| schedule | LR 随训练进度变化的规则 |
| WSD | warmup–stable–decay |
| checkpoint | 可恢复训练的参数/optimizer/进度快照 |
| warmup/stable/decay | LR 上升、稳定、下降阶段 |
| IsoFLOP | 固定 compute 比较不同模型/数据组合 |
| lower envelope | 已跑点中每个预算下的最低边界 |
| confound | 同时变化、可能干扰因果解释的变量 |
| validation loss | 未用于更新的数据上的 loss |
| held-out scale | 不参与拟合、专门检验外推的规模 |
| FLOP | 一次浮点运算 |
| $`\Theta(1)`$ | 随 width 增长仍保持常数量级 |
| fan-in/fan-out | 一层输入/输出宽度 |
| standard deviation | 数值波动尺度，variance 的平方根 |
| variance | 标准差平方 |
| spectral norm | 矩阵最大长度放大倍数 |
| Frobenius norm | 矩阵所有元素平方和再开根 |
| outer product | 列向量乘行向量得到矩阵 |
| μP | maximal update parameterization |
| μTransfer | 在 μP 下从小模型迁移 base 超参数 |
| maximal update | 不发散且不趋零的最大更新量级 |
| tensor programs | 分析宽度极限中张量计算量级的框架 |
| base LR | 用户调的共同 LR knob |
| effective LR | 乘参数组 width multiplier 后的实际 LR |
| residual | 拟合值与观测值的差 |
| convex surface | 在观察区间像碗的超参数曲面 |
| SVD | 把矩阵分成左右方向和奇异值 |
| Newton–Schulz | 迭代近似矩阵逆/正交化相关操作的方法 |
| weight decay | 每步按比例缩小参数的正则机制 |
| RMSNorm gain | RMS 归一化后乘的可训练尺度 |
| downstream | 预训练之后的具体任务或评测 |

---

<a id="23-自测题80题"></a>
## 23. 自测题（80 题）

> 第 1–60 题都是手算、填表、画 shape 或“判断并解释”，满足不了只背名词。先独立写过程，再看 §24。

### 23.1 四则运算与实践 scaling（1–20）

1. **【手算】**$`\theta=5,g=2,\eta=0.1`$。算一次 gradient descent 后的 $`\theta`$。
2. **【手算】**8 GPU，每卡 4 sequences，每条 512 tokens，gradient accumulation 2 次。global batch 是多少 sequences/update 和 tokens/update？
3. **【手算】**dense 粗账 $`N=2`$B、$`D=30`$B，按 $`C=6ND`$ 算训练 FLOPs。
4. **【手算】**算 $`16^{-1/2}`$、$`1000^{0.30}`$。
5. **【填式】**把 $`y=Ax^{-0.4}`$ 取 log，写成斜率与截距形式。
6. **【判断+解释】**一个 width 为 $`n`$ 的向量，每坐标约 2。它的每坐标与整向量 norm 哪个是 $`\Theta(1)`$？norm 约多少？
7. **【填表】**模型 A：$`d_m=4,d_{ff}=10,heads=2,L=2`$；B：8,20,4,4。算两者 $`d_{ff}/d_m`$、head width $`d_m/heads`$、$`L/d_m`$。
8. **【手算】**base width 4、current width 16、base std 0.1、base LR 0.01。按 §4 的 $`1/\sqrt r`$ 与 $`1/r`$ 算 current std、matrix LR。
9. **【手算】**4 层 residual，每层原增量 std 2。乘 $`1/\sqrt4`$ 后，每层 variance、总 variance、总 std 各多少？写独立近似条件。
10. **【手算】**样本 gradients 为 $`1,3,5,7`$。算 batch 1 取第一个、batch 2 取前两个、batch 4 的平均 gradient。
11. **【手算】**按 $`BS\propto L^{-6.24}`$，loss 从 2 降到 1，batch 比例约多少？若只用 $`6.24\approx6`$，粗估是多少？
12. **【手算】**目标长度 10、20、30、40。四次从头 cosine 总成本；WSD stable40 加四段 decay4 的总成本与节省百分比。
13. **【判断+手算】**若在每个整数 $`i=1,\ldots,100`$ 都接长度 $`0.1i`$ 的 decay，这部分成本是多少？它关于 100 是线性还是二次累积？
14. **【填表】**compute 10/20/40 下，A loss 2.5/2.2/2.0，B loss 2.8/2.1/1.9。写 lower envelope 和获胜模型。
15. **【手算】**固定 $`ND=100`$，用 $`L=1+4/\sqrt N+4/\sqrt D`$ 计算 $`(N,D)=(1,100),(4,25),(10,10),(25,4),(100,1)`$，找最低。
16. **【手算+来源】**MiniCPM/UltraText 的 $`L=0.0754N^{-0.3}+0.292D^{-0.3}+0.25`$。算归一化 $`N=D=1`$ 的 loss，并写出课件在 $`C=10^{21}`$ 报告的 $`D_{\mathrm{opt}}/N_{\mathrm{opt}}`$。
17. **【手算】**DeepSeek 式 $`\eta\propto C^{-0.125}`$。compute 增 16 倍，LR 比例是多少？
18. **【手算】**DeepSeek 式 $`B\propto C^{0.3271}`$。compute 增 16 倍，batch 比例约多少？
19. **【手算】**最低 loss 2.000，“0.25% 内”允许到多少？loss 2.006 是否在内？
20. **【手算】**$`M=4.3\times10^{11}`$ FLOPs/token，$`D=1.04\times10^{12}`$ tokens，按 $`C=MD`$ 算 compute。

### 23.2 案例、超参数曲面与 optimizer（21–40）

21. **【手算+解释】**Kimi K2 课程快照中 384 个 routed experts 每 token 选 8 个。sparsity ratio 与 routed 激活比例各是多少？为什么不能把全模型 FLOPs 直接除以 sparsity ratio？
22. **【手算】**58.1B activated parameters，96 tokens/activated-param。对应 tokens 约多少？
23. **【手算】**Llama 3 课程比例 39 tokens/param，模型 8B。算训练 tokens。
24. **【手算】**对 $`L(\eta,B)=2+100(\eta-0.02)^2+0.01(B-16)^2`$，计算 $`(.02,16),(.01,16),(.02,20),(.01,20)`$。
25. **【手算】**Step Law 固定 $`N`$，$`D`$ 增 4 倍。算 LR 比例 $`4^{0.307}`$ 和 batch 比例 $`4^{0.571}`$。
26. **【判断+手算】**把 0.571 偷换成 0.5，$`D`$ 增 4 倍时预测分别多少？相对差约多少？
27. **【手算】**不做 bias correction：$`g=2,m_0=v_0=0,\beta_1=.9,\beta_2=.99`$。算 $`m_1,v_1,m_1/\sqrt{v_1}`$。
28. **【手算】**只看 AdamW decay，$`\theta=10,\eta=.1,\lambda=.2`$。更新后多少？
29. **【填表】**Optimizer A 1000 steps×1.0s；B 800×1.4s。谁 step-efficient，谁 wall-clock 快？
30. **【手算+解释】**$`B=\mathrm{diag}(3,1)`$。写 SVD 的 $`U,\Sigma,V`$ 和理想 $`UV^\top`$。两个方向的更新比例怎样变？
31. **【画 shape】**$`h_{l-1}`$ 宽 3，$`W_l`$ 输出宽 2。写 $`h_{l-1},W_l,h_l`$ shapes；用 §13 的矩阵算输出。
32. **【手算】**算向量 $`[3,4]`$ 的 norm；$`\mathrm{diag}(3,1)`$ 的 spectral norm。
33. **【手算】**p47 式取 $`n_{in}=n_{out}=100`$，算 $`\sigma`$、近似谱范数和输入 norm10 对应的最坏方向上界/设计目标量级；它是否保证随机输入实际取等号？
34. **【手算】**p47 式取 $`n_{in}=100,n_{out}=25`$，算 $`\sigma`$、近似谱范数、最坏方向上界/设计目标量级。
35. **【手算】**p47 式取 $`n_{in}=25,n_{out}=100`$，重复第34题，并说明不是实际输出长度等式。
36. **【判断+手算】**对第34题，若只用 $`1/\sqrt{100}=0.1`$，近似谱范数和输入长度10的上界量级是多少？和目标5比较。
37. **【手算】**$`\delta=[2,-1]^\top,h=[3,4]^\top`$。算 outer product $`\delta h^\top`$ 并写 shape。
38. **【手算】**矩阵 $`A=[[1,2],[0,-1]]`$、$`B=[[3,4],[5,6]]`$。算内积 $`\langle A,B\rangle`$。
39. **【手算】**$`n_{in}=100,n_{out}=25`$。A2 目标 update spectral norm 是多少？
40. **【手算+条件】**同一宽度，目标 gradient norm 是多少？与第39题相乘验证 loss-change 量级；完整复述把 Frobenius norm 连到 spectral norm 所需的条件。

### 23.3 μP 更新、失败边界与实验设计（41–60）

41. **【手算】**SGD μP multiplier $`n_{out}/n_{in}`$：分别算 100→25、25→100、64→64。
42. **【手算】**Adam matrix LR $`\propto1/n_{in}`$。fan-in 从 256 变 1024，实际 LR 比例是多少？
43. **【手算】**全 1 矩阵 shape $`[25,100]`$ 的 spectral norm 是多少？乘 LR $`1/100`$ 后是多少？是否等于 $`\sqrt{25}/\sqrt{100}`$？
44. **【填表】**base width $`P=256`$，current $`M=256,512,1024`$，base LR .004。填 $`r`$、matrix multiplier $`1/r`$、effective LR。
45. **【手算】**初始化 variance 0.04，width ratio $`r=4`$，variance 除 4 后是多少？std 从多少变多少？
46. **【手算】**$`x=[3,4]`$，RMSNorm gain $`[1,1]`$，忽略 $`\epsilon`$。算 RMS 与输出。
47. **【手算+解释】**$`\lambda=.1`$。effective LR .004 与 .001 时 decay factor 各多少？哪个 decay 更弱？
48. **【手算】**算 $`2^{-6},2^{-8},2^{-10}`$，相邻两者比例多少？
49. **【填表】**固定 $`ND=1`$B²，给 $`N=.25,.5,1`$B，算 $`D`$。
50. **【手算】**3 个模型×3 batches×3 LRs×2 seeds，共多少 runs？每 run 100 GPU-hours，总多少？
51. **【判断+解释】**$`\|Wh\|\le\|W\|_*\|h\|`$ 是否表示每个随机 $`h`$ 都取等号？
52. **【手算+解释】**100 个坐标都为 2，向量 norm 是多少？每坐标与 norm 哪个保持常数？
53. **【手算】**若 raw Adam update 谱范数随方阵宽度 $`n`$ 为 $`n`$，分别用 LR $`1,1/n,1/n^2`$ 时 update norm 是什么？
54. **【判断+推导】**WSD 最大主干 $`C`$，$`m`$ 个固定长度 $`d`$ 的 decay，总成本什么式？什么条件下关于 $`C`$ 线性？
55. **【判断+解释】**StepFun 局部曲面像碗，能否推出神经网络参数空间全局凸？
56. **【填四本账】**MoE 总参数、activated parameters、每 token FLOPs、通信分别受什么影响？
57. **【判断+解释】**两个 recipe pretraining loss 2.00 与 2.01，可否直接说前者所有 downstream 都更好？
58. **【判断+解释】**比较 AdamW 与 Muon 时共用同一个 LR，是否公平？最低限度怎样做？
59. **【设计题】**用 width $`P,2P,4P`$ 拟合后，怎样设置一个 held-out scale？
60. **【画分类表】**用行向量约定写 one-hot→embedding→hidden→unembedding/logits 的 shapes；再写成列向量约定。为什么 embedding、hidden matrix、unembedding、norm gain 不能同用一条 multiplier？

### 23.4 概念与综合（61–80）

61. **【概念】**Scaling recipe 比 scaling law 多包含哪些东西？
62. **【概念】**Fixed aspect ratio 的收益和主要风险各是什么？
63. **【概念】**WSD 的 warmup、stable、decay 各做什么？
64. **【概念】**Method 1、2、3 各用什么数据找 compute-optimal 点？
65. **【判断+解释】**为什么 DeepSeek 式与 Step Law 的 LR 指数不能直接比正负？
66. **【判断+解释】**为什么模型总参数、activated parameters 和 non-embedding FLOPs/token 不能混用？
67. **【判断+解释】**“0.25% 内的低谷很平”对工程选择有什么用？
68. **【判断+解释】**为什么 optimizer 比较要同时报 steps 和 wall-clock？
69. **【概念】**Muon 的 Newton–Schulz 在直觉上改变矩阵的什么？
70. **【概念】**A1 与 A2 分别约束什么？
71. **【推导】**为什么 activation 每坐标 $`\Theta(1)`$ 时整向量 norm 是 $`\Theta(\sqrt n)`$？
72. **【推导】**p47 初始化式为什么同时看 fan-in 和 fan-out？
73. **【判断+解释】**μP 的 base LR transfer 与实际 tensor LR 有何区别？
74. **【判断+解释】**“Maximal update”为什么既不能太大也不能太小？
75. **【概念】**Tensor Programs 在本讲扮演什么角色？
76. **【判断+解释】**Attention scale $`1/\sqrt{D_h}`$ 与 $`1/D_h`$ 在固定 $`D_h`$ 时都对 width 是常数，为什么仍不能随便互换？
77. **【判断+解释】**RMSNorm gain、Lion、weight decay 的课程 stress test 各显示什么？
78. **【概念+例子】**什么叫 confound？给本讲一个例子。
79. **【设计题】**大 run 前至少做哪五项检查？
80. **【综合】**用一句完整因果链总结本讲。

---

<a id="24-自测答案"></a>
## 24. 自测答案

### 24.1 答案 1–20

1. 公式 $`\theta_{\text{new}}=\theta-\eta g`$。代入：
   $`5-0.1\times2=5-0.2=4.8.`$

2. Sequences/update：
   $`8\text{ GPUs}\times4\times2\text{ accum}=64.`$
   Tokens/update：
   $`64\times512=32{,}768.`$
   所以是 64 sequences/update、32,768 tokens/update。

3. $`2`$B $`=2\times10^9`$，$`30`$B $`=30\times10^9`$：
   $`C=6(2\times10^9)(30\times10^9) =360\times10^{18} =3.6\times10^{20}\text{ FLOPs}.`$

4. 
   $`16^{-1/2}=1/\sqrt{16}=1/4=0.25.`$
   $`1000^{0.30}=(10^3)^{0.30}=10^{0.9}\approx7.943.`$

5. 两边取同底数 log：
   $`\log y=\log A+\log(x^{-0.4}) =\log A-0.4\log x.`$
   在横轴 $`\log x`$、纵轴 $`\log y`$ 上，斜率 $`-0.4`$，截距 $`\log A`$。

6. 每坐标始终约 2，所以每坐标是 $`\Theta(1)`$。整向量：
   $`\|x\|=\sqrt{n\times2^2}=2\sqrt n,`$
   是 $`\Theta(\sqrt n)`$，不是 $`\Theta(1)`$。

7. A：
   $`d_{ff}/d_m=10/4=2.5,\quad d_m/heads=4/2=2,\quad L/d_m=2/4=0.5.`$
   B：
   $`20/8=2.5,\quad8/4=2,\quad4/8=0.5.`$
   这三个比例都相同。

8. 宽度比：
   $`r=16/4=4,\quad\sqrt r=2.`$
   Std：
   $`0.1/2=0.05.`$
   Matrix LR：
   $`0.01/4=0.0025.`$

9. 每层缩放后 std：
   $`2/\sqrt4=2/2=1.`$
   每层 variance $`=1^2=1`$。独立或近似不相关时，总 variance：
   $`1+1+1+1=4,`$
   总 std $`=\sqrt4=2`$。若层间高度相关，variance 不能只相加。

10. Batch 1 取第一个：$`1`$。Batch 2 前两个：
    $`(1+3)/2=2.`$
    Batch 4：
    $`(1+3+5+7)/4=16/4=4.`$

11. 精确比例：
    $`\frac{BS(1)}{BS(2)}=2^{6.24}\approx75.5.`$
    粗估 $`6.24\approx6`$：
    $`2^6=64.`$
    粗估低了约 $`75.5-64=11.5`$。

12. Cosine 从头：
    $`10+20+30+40=100.`$
    WSD：
    $`40+4\times4=56.`$
    节省：
    $`(100-56)/100=44/100=44\%.`$

13. Decay 成本：
    $`0.1(1+2+\cdots+100) =0.1\frac{100\times101}{2} =0.1\times5050=505.`$
    一般是 $`0.1n(n+1)/2`$，主项与 $`n^2`$ 成比例，所以是二次累积。

14. Compute 10：$`\min(2.5,2.8)=2.5`$，A。  
    Compute 20：$`\min(2.2,2.1)=2.1`$，B。  
    Compute 40：$`\min(2.0,1.9)=1.9`$，B。  
    Lower envelope 是 2.5、2.1、1.9。

15. 
    - $`(1,100)`$：$`1+4+0.4=5.4`$。
    - $`(4,25)`$：$`1+4/2+4/5=1+2+0.8=3.8`$。
    - $`(10,10)`$：$`1+8/\sqrt{10}\approx1+2.5298=3.5298`$。
    - $`(25,4)`$：$`1+0.8+2=3.8`$。
    - $`(100,1)`$：$`1+0.4+4=5.4`$。
    
    最低是 $`(10,10)`$，约 3.530。

16. $`1^{-0.3}=1`$，所以：
    $`L=0.0754+0.292+0.25=0.6174.`$
    这是 PDF p.18 的 MiniCPM/UltraText Method 3 拟合；同图在 $`C=10^{21}`$ 的口径下报告：
    $`D_{\mathrm{opt}}/N_{\mathrm{opt}}\approx95.60.`$

17. 
    $`16^{-0.125}=16^{-1/8}=(2^4)^{-1/8} =2^{-1/2}=1/\sqrt2\approx0.7071.`$

18. 
    $`16^{0.3271}=e^{0.3271\ln16}.`$
    $`\ln16\approx2.7726`$，乘积约 0.9069：
    $`e^{0.9069}\approx2.477.`$
    约 2.48 倍。

19. 允许增加：
    $`2.000\times0.25\% =2\times0.0025=0.005.`$
    上界 $`2.005`$。$`2.006>2.005`$，不在内。

20. 
    $`C=(4.3\times10^{11})(1.04\times10^{12}) =(4.3\times1.04)\times10^{23} =4.472\times10^{23}\text{ FLOPs}.`$

### 24.2 答案 21–40

21. Sparsity ratio：
    $`384/8=48.`$
    Routed 激活比例：
    $`8/384=1/48\approx0.02083=2.083\%.`$
    另有 shared expert；attention、embedding、router 也仍运行，总参数存储、all-to-all 通信和负载都不按 $`1/48`$ 缩小，所以全模型 FLOPs 不能直接除 48。

22. 
    $`58.1\times10^9\times96 =(58.1\times96)\times10^9.`$
    $`58.1\times(100-4)=5810-232.4=5577.6`$，所以：
    $`5.5776\times10^{12}\text{ tokens}.`$

23. 
    $`39\times8\text{B}=312\text{B tokens}.`$

24. 
    - $`(.02,16)`$：两差都 0，$`L=2`$。
    - $`(.01,16)`$：$`100(-.01)^2=100(.0001)=.01`$，$`L=2.01`$。
    - $`(.02,20)`$：$`0.01(4)^2=.16`$，$`L=2.16`$。
    - $`(.01,20)`$：$`2+.01+.16=2.17`$。

25. 
    $`4^{0.307}=e^{0.307\ln4} \approx e^{0.4256}\approx1.530.`$
    $`4^{0.571}=e^{0.571\ln4} \approx e^{0.7916}\approx2.207.`$

26. 原指数预测约 2.207；平方根预测：
    $`4^{0.5}=2.`$
    相对平方根预测的差：
    $`2.207/2-1=0.1035\approx10.35\%.`$
    所以不能在大范围外推时悄悄换指数。

27. 
    $`m_1=.9(0)+.1(2)=.2,`$
    $`v_1=.99(0)+.01(2^2)=.04.`$
    $`m_1/\sqrt{v_1}=.2/\sqrt{.04}=.2/.2=1.`$
    正式 Adam 还会做 bias correction。

28. 
    $`(1-\eta\lambda)\theta =(1-.1\times.2)10 =(1-.02)10=.98(10)=9.8.`$

29. A 总时间 $`1000\times1=1000`$ s。B 总时间 $`800\times1.4=1120`$ s。B 用 steps 更少，step-efficient；A wall-clock 更快。

30. 对正对角矩阵：
    $`U=I,\quad \Sigma=\begin{bmatrix}3&0\\0&1\end{bmatrix}, \quad V=I.`$
    理想 $`UV^\top=I`$。原两方向比例 3:1；整形后 1:1。

31. $`h_{l-1}`$ shape $`[3]`$ 或列向量 $`[3,1]`$；$`W_l`$ shape $`[2,3]`$；$`h_l`$ shape $`[2]`$。正文矩阵：
    $`\begin{bmatrix}1&2&0\\-1&0&1\end{bmatrix} \begin{bmatrix}1\\2\\3\end{bmatrix} = \begin{bmatrix} 1+4+0\\-1+0+3 \end{bmatrix} = \begin{bmatrix}5\\2\end{bmatrix}.`$

32. 
    $`\sqrt{3^2+4^2}=\sqrt{25}=5.`$
    $`\mathrm{diag}(3,1)`$ 最大放大倍数是 3，所以 spectral norm 3。

33. $`\sqrt{100}=10`$：
    $`\sigma=\frac{10}{10(10+10)}=\frac1{20}=0.05.`$
    谱范数近似：
    $`.05(10+10)=1.`$
    输入 norm 10 时，谱范数给出的最坏方向上界/设计目标量级是 $`1\times10=10=\sqrt{100}`$。随机输入未必对齐最大奇异向量，不保证实际取等号。

34. $`\sqrt{100}=10,\sqrt{25}=5`$：
    $`\sigma=\frac5{10(5+10)}=\frac1{30}\approx.03333.`$
    谱范数近似：
    $`\frac1{30}(10+5)=.5.`$
    输入 norm 10 时，最坏方向上界/设计目标量级 $`0.5(10)=5=\sqrt{25}`$；不是每个随机输入的实际长度等式。

35. $`\sqrt{25}=5,\sqrt{100}=10`$：
    $`\sigma=\frac{10}{5(10+5)}=\frac{10}{75}=\frac2{15}\approx.13333.`$
    谱范数近似：
    $`\frac2{15}(5+10)=2.`$
    输入 norm 5 时，最坏方向上界/设计目标量级 $`2(5)=10=\sqrt{100}`$；随机输入未必取到上界。

36. 
    $`\sigma=1/\sqrt{100}=0.1.`$
    谱范数近似：
    $`0.1(10+5)=1.5.`$
    输入长度 10 的上界量级 $`1.5(10)=15`$，而目标是 $`\sqrt{25}=5`$。大 3 倍。

37. 
    $`\delta h^\top = \begin{bmatrix}2\\-1\end{bmatrix} \begin{bmatrix}3&4\end{bmatrix} = \begin{bmatrix}6&8\\-3&-4\end{bmatrix}.`$
    Shape 是 $`[2,2]`$。

38. 对应元素乘加：
    $`1(3)+2(4)+0(5)+(-1)(6) =3+8+0-6=5.`$

39. 
    $`\sqrt{n_{out}}/\sqrt{n_{in}} =\sqrt{25}/\sqrt{100}=5/10=0.5.`$

40. Gradient norm 目标：
    $`\sqrt{n_{in}}/\sqrt{n_{out}}=10/5=2.`$
    与 update norm 相乘：
    $`0.5\times2=1,`$
    是 $`\Theta(1)`$ loss-change 量级。
    
    这条桥成立的完整条件是：单样本、深线性层、朴素 SGD；$`G=\delta h^\top`$ 为 rank one；$`\Delta W=-\eta G`$，所以 $`\Delta W`$ 也 rank one；使用一阶 Taylor $`\Delta\ell\approx\langle G,\Delta W\rangle_F`$；leading terms 没有严重 cancellation；目标是一步 loss change 为 $`\Theta(1)`$。只有 rank-one 才有 $`\|G\|_F=\|G\|_*`$ 和 $`\|\Delta W\|_F=\|\Delta W\|_*`$，并且
    $`\langle\Delta W,G\rangle_F=-\eta\|G\|_F^2.`$
    Batch gradient 是多个 outer products 的和，通常非 rank-one；Adam 也不是 raw gradient 乘同一标量，不能直接套这条等式。

### 24.3 答案 41–60

41. 
    $`100\to25:\quad25/100=.25.`$
    $`25\to100:\quad100/25=4.`$
    $`64\to64:\quad64/64=1.`$
    它们是宽度 multiplier，仍需乘共同 base LR。

42. 
    $`\frac{\eta_{1024}}{\eta_{256}} =\frac{1/1024}{1/256} =\frac{256}{1024} =\frac14.`$
    实际矩阵 LR 变为原来的四分之一。

43. 全 1 矩阵 spectral norm：
    $`\sqrt{25\times100}=\sqrt{2500}=50.`$
    乘 $`1/100`$：
    $`50/100=0.5.`$
    A2 目标：
    $`\sqrt{25}/\sqrt{100}=5/10=0.5.`$
    相等。

44. 
    | $`M`$ | $`r=M/P`$ | $`1/r`$ | effective LR |
    |---:|---:|---:|---:|
    | 256 | 1 | 1 | .004 |
    | 512 | 2 | .5 | .002 |
    | 1024 | 4 | .25 | .001 |

45. 原 variance 0.04，原 std：
    $`\sqrt{.04}=.2.`$
    新 variance：
    $`.04/4=.01.`$
    新 std：
    $`\sqrt{.01}=.1.`$
    variance 除 4，std 只除 2。

46. 
    $`\mathrm{RMS}(x)=\sqrt{(3^2+4^2)/2} =\sqrt{25/2}=\sqrt{12.5}\approx3.5355.`$
    输出：
    $`[3/3.5355,4/3.5355] \approx[.8485,1.1314].`$

47. 
    $`1-.004(.1)=1-.0004=.9996.`$
    $`1-.001(.1)=1-.0001=.9999.`$
    .9999 每步乘得更接近 1，所以 decay 更弱。

48. 
    $`2^{-6}=1/64=.015625,`$
    $`2^{-8}=1/256=.00390625,`$
    $`2^{-10}=1/1024=.0009765625.`$
    每相邻一项都除 4。

49. $`D=(1\text{B}^2)/N`$：
    | $`N`$ | $`D`$ |
    |---:|---:|
    | .25B | 4B |
    | .5B | 2B |
    | 1B | 1B |

50. Runs：
    $`3\times3\times3\times2=54.`$
    GPU-hours：
    $`54\times100=5400.`$

51. 不是。它是上界；只有 $`h`$ 对齐最大 singular direction 时才可能接近等号。随机输入通常不完全对齐。

52. 
    $`\|x\|=\sqrt{100\times2^2} =\sqrt{400}=20.`$
    每坐标是常数 2，即 $`\Theta(1)`$；norm 随 $`\sqrt{100}`$ 变化，一般是 $`\Theta(\sqrt n)`$。

53. Raw norm $`n`$：
    - LR 1：$`n`$；
    - LR $`1/n`$：$`(1/n)n=1`$；
    - LR $`1/n^2`$：$`(1/n^2)n=1/n`$。
    
    第一个爆大，第二个常数，第三个趋零。

54. 
    $`C_{\text{total}}=C+md.`$
    若 $`m,d`$ 固定或至少 $`md=O(C)`$，总成本关于 $`C`$ 是 $`O(C)`$。若 $`m`$ 随 $`C`$ 增长且每段 $`d`$ 也随 checkpoint 长度增长，可能回到 $`O(C^2)`$。

55. 不能。图只显示 LR×batch 这两个超参数在测过区域的 validation-loss 面像碗；网络参数空间有数十亿维，且未被这张图证明全局凸。

56. 
    | 账 | 主要受什么影响 |
    |---|---|
    | total params | 所有 experts、attention、embedding 等 |
    | activated params | 每 token 选中 experts + shared path + dense modules |
    | FLOPs/token | 激活路径、各矩阵 shape、attention |
    | communication | token 路由、expert placement、负载和网络拓扑 |

57. 不可以。2.00 只说明该 pretraining validation metric 较低；差 0.01 可能在噪声内，下游映射也可能非线性或换序。应给 confidence/seed，并实际评测 downstream。

58. 不一定公平。不同 optimizer 的最佳 LR 和 decay 区间不同。最低限度应给相同的调参预算，分别做 LR×decay（必要时 momentum/batch）搜索，再同时报 best validation loss、steps、tokens 和 wall-clock。

59. 例如用 $`P,2P,4P`$ 拟合后，把 $`8P`$ 完全留出；先锁定规则，不用 $`8P`$ 调参数，再预测其最佳 base LR/loss，最后运行 $`8P`$ 检验。通过后才考虑更大目标。

60. 行向量代码约定：
    $`\mathrm{onehot}[1,V]@W^E[V,M]\to h[1,M],`$
    hidden matrix 例如 $`h[1,M]@W[M,F]\to[1,F]`$，输出：
    $`h[1,M]@W^U[M,V]\to\mathrm{logits}[1,V].`$
    列向量约定把矩阵转置：
    $`(W^E)^\top[M,V]\mathrm{onehot}_{col}[V,1]\to h_{col}[M,1],`$
    $`(W^U)^\top[V,M]h_{col}[M,1]\to logits_{col}[V,1].`$
    
    | 参数 | 行向量典型 shape | 角色 |
    |---|---|---|
    | embedding | $`[V,M]`$ | 固定词表输入到可扩宽 hidden |
    | hidden matrix | $`[M,M]`$、$`[M,F]`$ 或 $`[F,M]`$ | 两侧可能随 width 变 |
    | unembedding | $`[M,V]`$ | hidden 到固定词表 logits |
    | norm gain | $`[M]`$ | 一维逐坐标尺度 |
    
    它们的 fan-in/out、网络位置和 gradient 结构不同，所以不能统一套 hidden square-matrix multiplier。

### 24.4 答案 61–80

61. Scaling law 只描述规模与指标关系；recipe 还包括 architecture、初始化、batch、LR、optimizer、schedule、data mixture、dtype 和 systems 设置。

62. 收益：减少搜索维度、让跨规模更可比。风险：被固定路径排除的 depth/width/heads 组合可能更好，系统约束也可能随规模改变。

63. Warmup 把 LR 从小值升上来；stable 用较高近常数 LR 学习大部分 tokens；decay 在终点前降低 LR，细化收敛。

64. Method 1 从不同训练轨迹按 compute 取 lower envelope；Method 2 在每个固定 compute 下扫描 $`N,D`$ 找碗底；Method 3 把全部 $`N,D,L`$ 点联合拟合参数化 loss 式。

65. DeepSeek 用 $`C`$ 作自变量，Step Law 把 $`N,D`$ 分开；当 $`C`$ 增长时 $`N,D`$ 怎样共同变会改变总指数。单位、optimizer、schedule 也不同。

66. 总参决定权重存储；activated params 更接近 MoE 每 token 路径；non-embedding FLOPs/token 是实际运算口径。三者数值和用途不同，混用会错算 compute optimum。

67. 可在几乎相同 loss 的点中选吞吐更高、显存更低、稳定性更好或实现更简单的点；不要为了噪声内的第三位小数牺牲系统性能。

68. 某 optimizer 可能少 steps 但每 step 做更多矩阵操作。只报 steps 会漏硬件成本，只报 wall-clock 又可能受实现质量影响；两者一起才能分开算法和系统。

69. 它近似把 momentum 梯度矩阵的奇异值压向 1，较多保留左右方向，减弱不同方向尺度差异。

70. A1 约束初始化后的 activation 每坐标保持 $`\Theta(1)`$；A2 约束一次训练更新引起的 activation 变化每坐标也保持 $`\Theta(1)`$。

71. $`n`$ 个坐标各约常数 $`c`$，平方和约 $`nc^2`$，开根得 $`|c|\sqrt n`$。

72. 随机矩阵谱范数近似 $`\sigma(\sqrt{n_{in}}+\sqrt{n_{out}})`$，目标映射还需把输入 norm $`\sqrt{n_{in}}`$ 变到输出 norm $`\sqrt{n_{out}}`$；所以两侧都出现。

73. Base LR 是用户搜索的共同 knob；框架按 width 和参数角色乘 multiplier 得到实际 tensor LR。μP 追求前者的最佳区域迁移，后者通常随 width 变化。

74. 太大使 activation update 随 width 爆；太小使 update 趋零、feature 学不动。Maximal 是不爆条件下仍不消失的最大宽度量级。

75. 它提供宽度趋于无穷时各 tensor、activation 和 gradient 的量级传播规则，支撑哪些 init/LR multiplier 能让 feature learning 稳定。

76. 固定 $`D_h`$ 时二者对主 width 的阶数都为常数，但数值不同，改变 attention logits 温度和 softmax；有限模型性能仍会不同。

77. 课程实验中：trainable RMSNorm gain 让 LR transfer 漂；某 Lion 规则未迁；强 decoupled weight decay 0.1 是显著 failure。它们是 stress-test 结果，不是普遍禁令。

78. Confound 是与研究变量同时变化、也可能造成结果的因素。例如大模型用了更小 LR，同时 batch、warmup 和 data/model ratio 都变；不能把差异只归因于模型大小。

79. 至少：held-out scale；短稳定性 run；loss/gradient/activation norm 监控；checkpoint 保存与恢复；data/tokenizer/compute 口径核查。还应做多 seed、系统吞吐和下游验证。

80. 小模型实验先控制 shape、data 和系统口径，二维调 batch/LR 并用 WSD/IsoFLOP 降低搜索成本；再用 μP 让 width 增长时 forward 与 update 保持常数量级，用 held-out scale 和 stress test 检查迁移，最后才锁 recipe 跑大模型。

---

<a id="25-视频时间导航"></a>
## 25. 视频时间导航

> 下表使用正文尚未使用的人工字幕 cue；因此全文每个 YouTube 秒点都只出现一次。正文中的 120 个链接提供更细导航。

| 时间 | 内容 | 笔记 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=5s) | 继续 scaling journey | §0 |
| [02:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=120s) | 实践 frontier 问题 | §2 |
| [03:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=239s) | MiniCPM 工作的启发 | §4 |
| [05:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=359s) | matrix initialization scaling | §4.3 |
| [08:03](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=483s) | 从 μP 成功例转向其它模型 | §4–§5 |
| [09:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=599s) | MiniCPM 最后一个成本问题 | §6 |
| [12:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=722s) | decay learning rate | §6.2 |
| [13:57](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=837s) | final annealing 很重要 | §6.3–§6.5 |
| [16:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=961s) | 原始 DeepSeek 案例 | §8 |
| [17:58](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1078s) | optimal batch 会随设置变 | §8.2 |
| [20:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1202s) | 2024 年公开模型的 scaling stack | §8.5 |
| [21:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1319s) | 预测目标 loss | §8.5 |
| [23:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1439s) | MoE scaling 新问题 | §9.3 |
| [26:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1560s) | pretraining loss 到 downstream | §9.4 |
| [28:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1680s) | architecture 与部署系统 | §9.5 |
| [30:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1802s) | 更细的 scaling 研究 | §10 |
| [32:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1921s) | StepFun study | §10.1 |
| [33:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2039s) | contour plot | §10.4 |
| [36:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2161s) | log-log scaling | §10.3–§10.5 |
| [38:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2280s) | 对数据的依赖 | §10.5–§10.6 |
| [40:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2401s) | LR/batch 课堂问答 | §10 |
| [42:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2521s) | optimizer 对比 | §11 |
| [44:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2640s) | 不同模型的超参数不同 | §11.2 |
| [46:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2760s) | algorithm development 要查 scaling | §11.4 |
| [47:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2879s) | 漂亮曲线也会外推失败 | §11.5 |
| [49:58](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2998s) | Muon 的基本更新 | §12.1 |
| [51:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3119s) | coordinate 与 spectral 直觉 | §12.2 |
| [54:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3240s) | Muon at scale 的证据边界 | §12.4 |
| [55:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3359s) | 是否每层用不同 optimizer | §12.3 |
| [57:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3479s) | width 改变时 standard 最佳 LR 漂 | §13 |
| [60:03](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3603s) | μP 的可读论文入口 | §16.4 |
| [61:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3719s) | 大宽度极限 | §13–§14 |
| [64:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3840s) | 输出 norm 目标 | §14.3 |
| [65:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3959s) | 权重变化 $`\Delta W`$ | §15.2 |
| [67:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4079s) | 前层 activation change | §15.2 |
| [70:02](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4202s) | 解 update-scale 条件 | §15.4–§15.5 |
| [72:01](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4321s) | 从理论得到实际算法 | §16–§17 |
| [73:59](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4439s) | 真实模型偏离 μP 理论 | §18 |
| [76:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4560s) | 对 scaling 确定性的最后提醒 | §18–§19 |

---

<a id="26-pdf-1–58页覆盖表"></a>
## 26. PDF 1–58 页覆盖表

> 下面每页恰好出现一次。“覆盖”表示该页的语义落在对应正文；它不表示逐字抄写课件。

| PDF 页 | 视觉内容/公式 | 正文 |
|---:|---|---|
| 1 | 标题：Scaling – case study and details | §0 |
| 2 | 今日动机与三个实践问题 | §2.1 |
| 3 | 近期公开模型时间线 | §2、§9 |
| 4 | 初始化、optimizer、LR/batch 的 scale sensitivity | §2.1、§11 |
| 5 | 两个详细案例 MiniCPM/DeepSeek | §2、§4、§8 |
| 6 | MiniCPM 背景与 μP | §4.1 |
| 7 | MiniCPM 1–2.5B 性能背景 | §4.2、§4.6 |
| 8 | scale_emb/depth/init/LR 公式与伪代码 | §4.3–§4.4 |
| 9 | 9M–0.5B 模型梯子与约 5×外推 | §4.2 |
| 10 | 不同 width 的最佳 base LR 曲线 | §4.5 |
| 11 | batch×tokens×loss 曲面、红色 minima | §5.3 |
| 12 | $`\log(BS)=-6.24\log L+20.91`$ | §5.3 |
| 13 | 重跑完整训练导致 $`n^2`$ 成本 | §6.1 |
| 14 | WSD 三段与 checkpoint 分叉 | §6.2–§6.3 |
| 15 | stable 落后、decay 快降、约 10% decay | §6.3–§6.5 |
| 16 | Chinchilla 型数据/模型 trade-off | §7.1、§7.5 |
| 17 | Method 1 lower envelope | §7.2 |
| 18 | MiniCPM/UltraText Method 3 联合式与 $`95.60`$ data/model ratio | §7.4 |
| 19 | DeepSeek 章节页、7B/67B 背景 | §8.1 |
| 20 | 不用 μP，直接估 batch/LR | §8.1–§8.2 |
| 21 | near-optimal 0.25% 点与 compute 拟合 | §8.3–§8.4 |
| 22 | 两段下降的 WSD 风格 schedule | §8.5 |
| 23 | Method 2、$`C=MD`$ IsoFLOP | §8.5 |
| 24 | scaling 对目标 loss 的预测图 | §8.5、§19.4 |
| 25 | Qwen2.5/Qwen3 scaling 快照 | §9.2 |
| 26 | Kimi K2 sparsity scaling | §9.3 |
| 27 | Hunyuan activated-param IsoFLOP | §9.4 |
| 28 | Llama 3 39:1 与 loss→downstream sigmoid | §9.4 |
| 29 | MiniMax architecture scaling | §9.5 |
| 30 | 近期 recipe 总结 | §9.6、§19 |
| 31 | optimizer scaling 引子与对比图 | §11 |
| 32 | StepFun 研究问题 | §10.1 |
| 33 | 多家 LR/batch 经验式表 | §10.3 |
| 34 | 18 组配置的二维 contour/grid | §10.4 |
| 35 | LR 与 batch 的一维切片和 3D 碗 | §10.1–§10.2 |
| 36 | 随 model/data 的 scaling trends | §10.3–§10.5 |
| 37 | MoE/其它 data robustness | §10.6 |
| 38 | AdamW/NAdamW/Muon/Soap 随规模 speedup | §11.3–§11.4 |
| 39 | 错误超参数可制造巨大差异 | §11.2 |
| 40 | 模型规模与 Chinchilla ratio confound | §11.4 |
| 41 | 好看的外推可能 blow up | §11.5 |
| 42 | Muon 四行公式与 $`USV^\top\to UV^\top`$ | §12.1–§12.2 |
| 43 | nanoGPT/scaling/Kimi K2 证据层级 | §12.4 |
| 44 | standard 与 μP 曲线、width multiplier 表 | §13.3–§13.5 |
| 45 | Cerebras-GPT μP 迁移 | §13.6 |
| 46 | A1、A2 与坐标/norm 关系 | §13.1、§14.3 |
| 47 | A1：谱范数与初始化 $`\sigma`$ | §14 全节 |
| 48 | A2：outer product 与 $`\Delta h_l`$ | §15.1–§15.2 |
| 49 | loss change、gradient norm、SGD/Adam LR | §15.3–§15.5 |
| 50 | baby μP 初始化/LR 总结 | §16.1 |
| 51 | Transformer 参数类别完整 $`\Theta`$/exact variance、Adam LR 表，output/attention multiplier | §17.1–§17.8 |
| 52 | baseline 与 projection biases 在 widths 128/512/2048 的最低采样列 | §18.1 |
| 53 | SwiGLU/batch/init/norm/optimizer/regularizer stress tests | §18.2、§18.7 |
| 54 | RMSNorm vector/scalar gain 三宽度最低采样列 | §18.3 |
| 55 | Lion 三宽度最低采样列 | §18.4 |
| 56 | strong weight decay 三宽度最低采样列 | §18.5 |
| 57 | standard 漂移与 μP 2M–10B 最低采样列对照 | §18.6 |
| 58 | 全讲 recap | §1、§19、§27 |

### 26.1 视觉核验记录

- 用 pypdf 提取了 58/58 页文字，但公式以图片为准。
- 用 pypdfium2 渲染 58/58 张普通页，并生成 6 张 contact sheets，逐页检查是否有漏页、图、表和公式。
- p.8–18、p.20–58 的公式/表格页又查看了原分辨率高清 PNG；p.19 是 DeepSeek 章节过渡页，普通页与 contact sheet 足够。
- p.47–50 单独逐式核对：p47 的 $`\sigma`$ 是 $`W\sim\mathcal N(0,\sigma^2I)`$ 中的标准差；p48 的 update 是单样本 SGD rank-one outer product；p49 用一阶 loss change 和 rank-one norm 得 SGD LR，并单列 Adam；p50 明确 Adam 为 $`1/n_{l-1}`$。
- p.51 高清表逐格核对了 8 类参数的 $`\Theta`$/exact **variance** 与 Adam LR，并核对表下 attention $`1/D_h`$ 文字；§17 没有把 variance 偷换成 std。
- p.52–57 高清表逐行读取加粗最低列；§18 明确它们只是离散 LR 网格最低采样列，不是连续最优值。
- p.44 表里的 $`\sigma`$ 标注为 variance，与 p.47 的标准差符号过载，正文 §13.5 已显式拆开。

---

<a id="27-来源边界与学完后的能力"></a>
## 27. 来源边界与学完后的能力

### 27.1 课程来源

- 官方 Lecture 11 PDF：58 页；本地版本由课程仓库 source map 对应。
- 官方 Stanford Online 视频：[Scaling Laws II / μP](https://www.youtube.com/watch?v=vTfEyOyzV9E)。
- 字幕轨：人工 **English (United States)**，共 1808 个 cue；首 cue 00:05，末 cue 76:55。正文和导航均只使用该 cue 集中的秒点。

【课程内容】中的具体模型表、曲线读法和 2024–2026 叙述，代表 Spring 2026 课堂时点。课程把多篇材料压成教学故事；笔记不把讲者的口头概括冒充论文原句。

### 27.2 一手补充来源

- [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395)：MiniCPM scaling、WSD、μP recipe。
- [Training Compute-Optimal Large Language Models / Chinchilla](https://arxiv.org/abs/2203.15556)：三种 compute-optimal 方法的原始背景。
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954)：DeepSeek scaling 与模型报告。
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)、[Kimi K2](https://arxiv.org/abs/2507.20534)、[Hunyuan-Large](https://arxiv.org/abs/2411.02265)、[Llama 3 Herd](https://arxiv.org/abs/2407.21783)、[MiniMax-01](https://arxiv.org/abs/2501.08313)：§9 的公开案例边界。
- [Step Law](https://arxiv.org/abs/2503.04715)：LR×batch 经验曲面与拟合。
- [Decoupled Weight Decay Regularization / AdamW](https://arxiv.org/abs/1711.05101)、[Symbolic Discovery of Optimization Algorithms / Lion](https://arxiv.org/abs/2302.06675)：optimizer 定义。
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)：μP/μTransfer 正式来源。
- [A Large-Scale Exploration of μ-Transfer](https://arxiv.org/abs/2404.05728)：p.51–57 的 Transformer 迁移与 stress tests。
- [Spectral Conditions for Feature Learning](https://arxiv.org/abs/2310.17813)：课程采用的谱范数量级视角。
- [Cerebras-GPT](https://arxiv.org/abs/2304.03208)：p.45 模型家族。

这些一手来源用于核对边界和补解释；没有公开的企业内部 recipe 一律不补猜。

### 27.3 哪些数字是“课程快照”

- MiniCPM 模型梯子、0.01 base LR、batch fit；
- DeepSeek 的 0.3118/0.2920 经验式；
- Kimi/Hunyuan/Llama/MiniMax 图中数字；
- StepFun 表中的多家拟合式；
- p.52–57 stress-test minima。

它们都依赖数据、tokenizer、architecture、optimizer、compute 定义和搜索范围。新项目必须重新验证。

### 27.4 学完后应该能做什么

你应该能：

1. 把“训练大模型”拆成 model/data/batch/LR/optimizer/schedule 的实验设计；
2. 手算 WSD 成本、IsoFLOP 小表和经验幂律比例；
3. 分辨 total params、activated params、FLOPs/token；
4. 从二维 LR×batch 曲面读低谷而非迷信单点；
5. 识别 optimizer 比较中的调参和规模 confound；
6. 从一层线性层完整推出 p47 的初始化尺度；
7. 从 outer product、谱范数目标推出 SGD 与 Adam 的 width LR 量级；
8. 解释 base LR 为什么可迁而实际 tensor LR 会变；
9. 识别 RMSNorm gain、Lion、weight decay 等 μP failure modes；
10. 用 held-out scale 和 stress test 决定何时可以安全扩大训练。

最后一句：

> Scaling 的真正价值不是让一条线替你做决定，而是用小实验排除明显错误、量化不确定性，再用严格的 held-out 证据决定是否值得下注一次昂贵的大训练。
