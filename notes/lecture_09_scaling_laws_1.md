# CS336 Lecture 9：Scaling Laws I——从幂律、临界 Batch 到 Kaplan 与 Chinchilla

> **目标读者：**只会加、减、乘、除，第一次接触 scaling law（缩放定律）的初学者。  
> **目标：**不看 77 分钟视频，也能从零推导本讲公式、看懂图、复算数字，并知道哪些结论只能在实验范围内使用。  
> **官方课件：**[Stanford CS336 Lecture 9 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)，57 页。  
> **官方视频：**[Stanford Online：Lecture 9](https://www.youtube.com/watch?v=Q15rhEWZPQ4)。

## 0. 阅读说明、来源边界与资料核验

### 0.1 第一次阅读路线

第一次读时，请**跳过 §1 的五分钟复习卡**，按下面顺序走：

1. §2 先学最少的数学：次方、对数、斜率、均值和方差。
2. §3–§7 从“缩放定律是什么”走到两个可手算来源：均值估计和分箱回归。
3. §8–§12 看数据、架构、batch size 和学习率为什么会改变曲线。
4. §13–§18 是本讲核心：联合模型—数据规律、固定 compute 最优解、Kaplan 与 Chinchilla、部署最优。
5. 最后用 §19–§21 复习，再做 §22 的题；答案在 §23。

视频 [00:08](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=8s) 暂时离开 systems；[00:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=20s) 说明本讲只讲 basics；更高级的 scaling 与 parameterization 留到后续课。

### 0.1.1 可点击目录

- [§1 五分钟复习卡](#1-五分钟复习卡首次阅读请跳过)
- [§2 前置数学](#2-前置知识与本讲计算器数学从只会四则到幂律)
- [§3–§7 scaling law、幂律、均值与分箱](#3-scaling-law-到底是什么)
- [§8–§10 数据与模型工程](#8-数据不是只有多少compositionshift-与-mixture)
- [§11 Critical batch](#11-critical-batchsteps-与-examples-的交换)
- [§12–§13 muP 与联合 scaling](#12-学习率与-mup本讲只预告)
- [§14–§18 Kaplan、Chinchilla 与部署最优](#14-kaplan-与-chinchilla两个不同资源处方)
- [§19–§21 决策树、误区与术语](#19-一页决策树遇到-scaling-问题怎么走)
- [§22 自测](#22-自测题80-题第-1670-题为手算填表) · [§23 答案](#23-自测答案180)
- [§24 视频导航](#24-视频时间导航全部命中人工字幕-cue) · [§25 PDF 覆盖与来源](#25-pdf-157-覆盖索引来源与核验边界)

### 0.2 四类来源标签

- **【课程内容】**：PDF 明确出现的图、公式、表或结论。
- **【视频补充】**：人工英文字幕中的口头限定、课堂问答或纠错。
- **【补充理解/例子】**：为零基础读者补出的中间算术、反例和单位。
- **【延伸】**：来自论文或官方资料；不是老师逐字说法。

当课程口语与 PDF 冲突时，本笔记不暗自“选一个”，而会把两种口径同时写出。

### 0.3 PDF 与字幕核验记录

**PDF：**

- 本地 `work/pdfs/lecture_09.pdf`，共 **57 页**。
- 用 pypdf 提取文字用于搜索，但公式以渲染图为准。
- 57/57 页都用 pypdfium2 渲染普通页与高分辨率页；6 张 contact sheets 用于逐页总览。
- 全部含图、公式或表的页面都查看了高分辨率图，重点复读 p6–20、p22–26、p30–41、p43–55。
- 特别核验：p19 是 $`n^{-1/4}`$；p24 是饱和的重复数据公式；p38 是 critical-batch 精确曲线；p43 是联合 law；p45–53 是 Kaplan/Chinchilla 指数与争议；p54 是 tokens-per-parameter 列表。

**字幕：**

- 轨道为人工 `English (United States)`，不是 ASR 自动字幕。
- 共 **1812 segments**，第一段 **00:06**，末段从 **77:50** 开始，视频正文到约 **77:53**。
- 正文时间戳只选实际字幕 cue；链接显示时间与 `t=秒数` 一致。

### 0.4 本讲的因果链

**【课程】【PDF 1–4】【视频补充】[01:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=69s)**

```text
大训练太贵，不能把所有大方案都试一遍
    ↓
训练许多较小模型，记录资源 x 与 loss y
    ↓
拟合简单曲线，检查小规模内是否稳定
    ↓
外推到更大资源，预留大 run 验证预算
    ↓
据此选择 data / model / batch / architecture
    ↓
部署请求很多时，再把 inference 成本加入目标
```

Scaling law 的价值不是“算命”，而是让昂贵决策有实验依据。视频 [02:48](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=168s) 称它是强力工具，但 [03:04](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=184s) 随即提醒它可能很 tricky（棘手）。

本讲说的 **pretraining（预训练）**是先让语言模型在大规模通用文本上学习 next-token prediction（根据前文预测下一个 token），之后才进入具体任务或服务。

---

## 1. 五分钟复习卡（首次阅读请跳过）

1. **Scaling law** 是资源与损失之间在特定实验范围内观察到的经验规律，不是宇宙定律。（见 §3）
2. 常见形式：

   $`L(n)=L_\infty+A n^{-\alpha}.`$

   $`L_\infty`$ 是不可约损失，$`A`$ 是尺度，$`\alpha>0`$ 控制改善速度。（见 §5.1）
3. 减掉 $`L_\infty`$ 后取对数（见 §5.3）：

   $`\log(L-L_\infty)=\log A-\alpha\log n.`$

   所以 log-log 图上斜率是 $`-\alpha`$；原坐标仍是弯曲的。
4. 若 $`\alpha=0.5`$，资源增 4 倍，额外损失乘 $`4^{-0.5}=1/2`$。（见 §5.2）
5. 样本均值满足 $`\mathrm{MSE}=\sigma^2/n`$；这是能从方差相加推出来的真正 $`1/n`$ 例子。（见 §6）
6. 二维分箱启发：边长 $`n^{-1/4}`$，箱数约 $`\sqrt n`$，每箱约 $`\sqrt n`$ 个样本，方差项约 $`1/\sqrt n`$；还必须加 bias/smoothness 项。（见 §7）
7. 重复数据有递减价值；有效数据不是“看了 $`R`$ 遍就乘 $`R`$”。（见 §9）
8. Critical batch（临界批量）是 batch 继续增大时开始明显回报递减的尺度；课程精确曲线为（见 §11）：

   $`\frac{S}{S_{\min}}-1=\left(\frac{E}{E_{\min}}-1\right)^{-1}, \qquad B_{\text{crit}}=\frac{E_{\min}}{S_{\min}}.`$

9. 联合 law 用统一符号写作（见 §13）：

   $`L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.`$

10. 对 dense Transformer 训练，粗略 $`C\approx6ND`$；固定 $`C`$ 时（见 §13.2–§13.3）：

    $`N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)},\qquad D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}.`$

11. Kaplan 课件口径约 $`N\propto C^{0.73},D\propto C^{0.27}`$；Chinchilla 前两种方法约各 $`C^{0.5}`$。差异受到 parameter count、warmup、batch、compute range 等影响。（见 §14–§16）
12. Chinchilla 的“约 20 tokens/parameter”是训练 compute-optimal 的历史经验，不是 inference（推理，即训练完后用模型回答/生成）场景的黄金比例。（见 §14.3、§17.4）
13. “Overtrained”要加引号：更多 tokens 可能超过训练算力最优，但并不等于传统 overfitting（训练 loss 继续改善、validation/test loss 却变差）。Validation loss 是在不用于参数更新的验证集上计算的损失。（见 §17.3）
14. IsoFLOP：固定 FLOPs，扫不同 $`N,D`$，找碗底；容易执行，但仍会被数据质量、超参不公平和扫描范围影响。（见 §15.2、§18）
15. 预测 pretraining loss 不等于预测 accuracy、推理能力或每个 downstream task。（见 §10.5）

---

## 2. 前置知识与本讲计算器数学：从只会四则到幂律

### 2.1 次方和负指数

**Exponent（指数）**告诉我们一个数要乘几次：

```math
2^3=2\times2\times2=8.
```

负指数表示“取倒数”：

```math
n^{-1}=\frac1n,\qquad n^{-1/2}=\frac1{\sqrt n}.
```

$`\sqrt n`$（平方根）是“哪个正数乘自己得到 $`n`$”。例如 $`\sqrt{16}=4`$，因为 $`4\times4=16`$。

### 2.2 log 是什么

**Logarithm（对数）**是指数的反问题。本文未写底数时，$`\log`$ 可理解为任一固定底；斜率推导对底数选择不敏感。手算表常用 $`\log_{10}`$：

```math
\log_{10}(1000)=3,
```

因为 $`10^3=1000`$。自然对数写 $`\ln`$，底数是 $`e\approx2.718`$。计算器中：

```text
输入 log10(1000) → 3
输入 ln(e^2)     → 2
输入 4^(-0.5)    → 0.5
```

两条会用到的规则：

```math
\log(ab)=\log a+\log b,
\qquad
\log(a^r)=r\log a.
```

### 2.3 斜率、截距、单调与渐近线

直线 $`y=b+mx`$ 中：

- $`m`$ 是 **slope（斜率）**：$`x`$ 增 1，$`y`$ 变多少。
- $`b`$ 是 **intercept（截距）**：$`x=0`$ 时的 $`y`$。
- **Monotonic（单调）下降**：$`x`$ 越大，$`y`$ 从不升高。
- **Asymptote（渐近线）**：曲线越来越接近但有限资源下未必到达的线。

例如点 $`(1,5)`$ 到 $`(3,1)`$：

```math
m=\frac{1-5}{3-1}=\frac{-4}{2}=-2.
```

### 2.4 loss、accuracy 与 residual

**Loss（损失）**是训练目标的“坏程度”，通常越小越好。**Accuracy（准确率）**是答对比例，通常越大越好。一个 scaling law 预测 loss，不代表 accuracy 一定按同样形状变化。

**Residual（残差）**不是 Transformer 的 residual connection；这里指：

```math
\text{residual}=\text{实测值}-\text{曲线预测值}.
```

预测 2.4、实测 2.5，则 residual 为 $`2.5-2.4=0.1`$。

### 2.5 单位：parameter、token、FLOP

- **Parameter（参数）**：训练会更新的一个数。
- **Tokenizer（分词器）**：把原始文本按固定规则切成 token IDs 的程序；token ID 是词表中某个 token 的整数编号。**Token** 是切分后给模型处理的一个单位，不必等于一个单词。
- **FLOP**（floating-point operation）：一次浮点加、减、乘或除；**FLOPs**常被口语混用为总次数；**FLOP/s**才是每秒速度。
- $`1\text{K}=10^3`$、$`1\text{M}=10^6`$、$`1\text{B}=10^9`$、$`1\text{T}=10^{12}`$。这里 B 是 billion，不是 byte。

---

## 3. Scaling law 到底是什么

### 3.1 一句人话定义

**【课程】【PDF 3–6】【视频补充】[03:08](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=188s)**

Scaling law（缩放定律）是一个简单预测规则：当训练资源有规律地增加时，某个指标怎样变化。资源可以是数据量、参数量或训练 compute；指标常是 test/pretraining loss。

它通常通过小实验拟合，不是从 Transformer 公理严格证明出来的。

### 3.2 interpolation 与 extrapolation

- **Interpolation（插值）**：在已经测过的范围中间预测。
- **Extrapolation（外推）**：预测测量范围之外。

例：已测 $`n=1,2,4,8`$ 亿 tokens。

- 预测 $`n=3`$ 亿是插值。
- 预测 $`n=80`$ 亿是外推。

外推更有价值也更危险：公式在小范围拟合得好，不保证跨 10 倍、100 倍资源仍成立。视频 [04:32](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=272s) 把历史工作的共同目标描述为用曲线预测尚未训练的大模型。

### 3.3 经验律不是物理定律，也不是理论上界

**Gradient（梯度）**是 loss 对 parameters 的局部变化率：某个 parameter 改一点，loss 大约怎样变。**Learning rate（学习率）**是一次参数更新要走多大一步。**Optimizer（优化器）**读取 gradient，并按 learning rate 等规则更新 parameters。现在才能定义：**Training recipe（训练配方）**是 architecture、optimizer、learning rate、batch size、warmup、数据处理等训练选择的整套组合。**Empirical law（经验律）**由观察拟合出来，意思只是：“在我们试过的模型、数据和 training recipe 中像这样。”

**Generalization bound（泛化界）**：理论给出的最坏情况保证，形式常为“真实误差不超过某个上界”。它可能很松，但目标是保证。

两者不能互换：

| 对比 | 经验 scaling law | 理论 generalization bound |
|---|---|---|
| 来源 | 实验拟合 | 数学假设与证明 |
| 常见用途 | 预测实际趋势 | 给最坏情况保证 |
| 失效原因 | recipe、数据分布、范围改变 | 假设不满足或界太松 |

课件 p6 的理论曲线高于实测，表示“保证没有被违反”，不表示界能精确预测。视频 [06:44](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=404s) 也指出传统 learning theory 往往不把它叫 scaling law。

### 3.4 sample complexity

**Sample complexity（样本复杂度）**：达到目标误差需要多少训练样本。

若误差 $`e(n)=1/\sqrt n`$，要求 $`e\le0.1`$：

```math
\frac1{\sqrt n}\le0.1
\Rightarrow \sqrt n\ge10
\Rightarrow n\ge100.
```

这就是从 scaling law 反推资源。

---

## 4. 一条够用的历史线

**【课程】【PDF 7–10】**历史不是为了背年份，而是看问题如何从“画学习曲线”变成“大模型资源决策”。

1. **1993，Cortes 等：** **Learning curve（学习曲线）**是横轴放数据量、steps 或 compute，纵轴放 loss/error 等表现的曲线。这项工作比较 training set size 与泛化表现，并讨论从小样本预测更大样本。视频 [07:42](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=462s) 展示多种候选函数。
2. **2001，Banko 与 Brill：** **NLP（Natural Language Processing，自然语言处理）**研究计算机怎样处理人类语言。论文在 NLP 歧义消解上把训练语料扩到很大规模，展示更多数据可持续改善不同方法；重点不是“算法无所谓”，而是数据规模本身是重要轴。[原论文入口](https://aclanthology.org/P01-1005/)
3. **2012，Kolachina 等：**研究机器翻译 learning curve 的提前预测，说明“预测尚未训练的数据规模”早于现代 LLM。[原论文入口](https://aclanthology.org/P12-1003/)
4. **2017，Hestness 等：**跨语言、视觉、语音任务观察 **power-law generalization error**，即未见数据上的 error 随数据量按固定幂次下降；论文强调这些指数当时缺少充分理论解释。[原论文](https://arxiv.org/abs/1712.00409)
5. **2020 以后：** **LLM（Large Language Model，大语言模型）**是用大量参数和文本训练的语言模型。Kaplan、Rosenfeld、Chinchilla 把 data、model、compute 联合起来，用于决定昂贵 LLM 怎么训。视频 [09:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=549s) 将此称为 2020s 的现代阶段。

---

## 5. 幂律：从公式到 log-log 直线

### 5.1 每个符号先解释

**【课程】【PDF 11–15】【视频补充】[13:37](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=817s)**

常见数据缩放式：

```math
L(n)=L_\infty+A n^{-\alpha}.
```

- $`n`$：训练样本或 tokens 数量。
- $`L(n)`$：用 $`n`$ 数据训练后的 loss。
- $`L_\infty`$：假想无限数据时仍剩下的 **irreducible loss（不可约损失）**。
- $`A>0`$：纵向尺度；同一 $`\alpha`$ 下，$`A`$ 大则曲线整体更高。
- $`\alpha>0`$：power-law exponent（幂律指数）；越大表示资源翻倍时下降得越快。

它是 **power law（幂律）**，因为资源 $`n`$ 被提升到一个固定次方。

### 5.2 资源翻倍时究竟乘多少

只看可改善部分 $`\Delta L=L-L_\infty=A n^{-\alpha}`$。资源从 $`n`$ 变 $`rn`$：

```math
\frac{\Delta L(rn)}{\Delta L(n)}
=\frac{A(rn)^{-\alpha}}{An^{-\alpha}}
=r^{-\alpha}.
```

$`A`$ 和 $`n^{-\alpha}`$ 都约掉了，所以只剩资源倍数 $`r`$ 与指数 $`\alpha`$。

取 $`\alpha=0.5`$、$`A=8`$、$`L_\infty=1`$，从 $`n=1`$ 开始：

| $`n`$ 增长倍数 | $`n^{-0.5}=1/\sqrt n`$ | 额外 loss $`8/\sqrt n`$ | 总 loss |
|---:|---:|---:|---:|
| 1 | 1 | 8 | 9 |
| 2 | $`1/\sqrt2\approx0.707`$ | $`5.656`$ | $`6.656`$ |
| 4 | $`1/2`$ | 4 | 5 |
| 10 | $`1/\sqrt{10}\approx0.316`$ | $`2.530`$ | $`3.530`$ |

注意：资源 4 倍只让“超过 $`L_\infty`$ 的部分”减半，不是让总 loss 减半。

**课程中的真实小指数例。**PDF p15 从 Kaplan 图中写出：

```math
L(D)=\left(\frac{D}{5.4\times10^{13}}\right)^{-0.095}.
```

$`5.4\times10^{13}`$ 是让横轴无量纲的参考 token 数；$`-0.095`$ 很浅，所以数据要增很多才明显下降。只比较额外项倍率：

| $`D`$ 倍数 $`r`$ | $`r^{-0.095}`$ | 约剩多少 |
|---:|---:|---:|
| 2 | $`2^{-0.095}`$ | 0.936 |
| 4 | $`4^{-0.095}`$ | 0.877 |
| 10 | $`10^{-0.095}`$ | 0.803 |

也就是数据翻 10 倍，该拟合项只下降约 $`1-0.803=19.7\%`$。PDF p18 同时展示约 0.13、0.30、0.095 等不同经验斜率，说明 exponent 会随任务与实验口径变化。

### 5.3 为什么 log-log 图是一条直线

从：

```math
L-L_\infty=A n^{-\alpha}
```

两边取 log：

```math
\log(L-L_\infty)
=\log(A n^{-\alpha})
=\log A+\log(n^{-\alpha})
=\log A-\alpha\log n.
```

令 $`y=\log(L-L_\infty)`$，$`x=\log n`$：

```math
y=\log A+(-\alpha)x.
```

这正是“截距 $`\log A`$、斜率 $`-\alpha`$”的直线。视频 [15:18](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=918s) 称其为 scale-free/power-law relation。

**防坑：**log-log 图直，不代表原坐标图直。原图是向 $`L_\infty`$ 弯曲的曲线。

### 5.4 least squares 与残差

**Least squares（最小二乘）**：选择参数，使 residual 的平方和最小：

```math
\text{SSE}=\sum_i (L_i-\widehat L_i)^2.
```

若三点 residual 为 $`0.1,-0.2,0.1`$：

```math
\text{SSE}=0.1^2+(-0.2)^2+0.1^2=0.01+0.04+0.01=0.06.
```

只报告高 $`R^2`$ 不够：残差若随规模系统性变正或变负，说明函数形状可能错。

$`R^2`$（决定系数）粗略衡量曲线解释了多少观测波动，越接近 1 通常表示区间内拟合越紧；它不检验区间外外推是否正确。

---

## 6. 均值估计：一个真的能推出来的 $`1/n`$ law

### 6.1 parametric 是什么意思

**Parametric（参数化统计模型）**：未知对象能用固定有限个数描述。估计总体均值 $`\mu`$ 只需估一个数，所以是最简单的 parametric 问题。

假设有样本 $`X_1,\ldots,X_n`$。**独立（independent）**的意思是：知道 $`X_i`$ 取了什么值，不会改变另一个 $`X_j`$ 的概率分布。这里还假设每个样本都来自同一总体，因此：

```math
\mathbb E[X_i]=\mu,
\qquad
\mathrm{Var}(X_i)=\sigma^2.
```

$`\mathbb E[X]`$ 是把同一随机过程重复很多次后 $`X`$ 的长期平均。**Variance（方差）**的定义是：

```math
\mathrm{Var}(X)
=\mathbb E\left[(X-\mathbb E[X])^2\right].
```

先算“结果离长期平均多远”，再平方、再取长期平均。$`\sigma^2`$ 是方差；$`\sigma=\sqrt{\sigma^2}`$ 是 standard deviation（标准差）。

样本均值：

```math
\bar X=\frac{X_1+\cdots+X_n}{n}.
```

视频 [16:07](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=967s) 从这个问题开始连接 Machine Learning 101。

### 6.2 为什么它无偏：把期望逐项写开

**Unbiased（无偏）**表示平均预测正好等于真值：

```math
\begin{aligned}
\mathbb E[\bar X]
&=\mathbb E\left[\frac{X_1+\cdots+X_n}{n}\right]\\
&=\frac1n\left(\mathbb E[X_1]+\cdots+\mathbb E[X_n]\right)\\
&=\frac1n(\mu+\cdots+\mu)\\
&=\frac1n(n\mu)\\
&=\mu.
\end{aligned}
```

第二行用了期望的线性：总和的长期平均等于各项长期平均之和；这一步本身不要求独立。无偏不是说每次 $`\bar X`$ 都刚好等于 $`\mu`$，而是重复整次实验后的平均不偏高也不偏低。

### 6.3 为什么独立时 covariance 交叉项为 0

两个随机量的 **covariance（协方差）**定义为：

```math
\mathrm{Cov}(X,Y)
=\mathbb E\left[(X-\mathbb E[X])(Y-\mathbb E[Y])\right].
```

它衡量两个量是否一起偏高或一起偏低。若 $`X_i,X_j`$ 独立，则两个中心化后的乘积期望可以拆开：

```math
\begin{aligned}
\mathrm{Cov}(X_i,X_j)
&=\mathbb E[(X_i-\mu)(X_j-\mu)]\\
&=\mathbb E[X_i-\mu]\;\mathbb E[X_j-\mu]\\
&=(\mu-\mu)(\mu-\mu)\\
&=0.
\end{aligned}
```

因此总和的方差公式：

```math
\mathrm{Var}\left(\sum_{i=1}^nX_i\right)
=\sum_{i=1}^n\mathrm{Var}(X_i)
+2\sum_{i<j}\mathrm{Cov}(X_i,X_j)
```

在独立假设下，所有 $`i\ne j`$ 的 covariance 都为 0，只剩：

```math
\mathrm{Var}\left(\sum_{i=1}^nX_i\right)
=\sum_{i=1}^n\mathrm{Var}(X_i)
=n\sigma^2.
```

若样本相关，交叉项不一定为 0，下面的 $`\sigma^2/n`$ 就不能直接套。

### 6.4 方差为什么变成 $`\sigma^2/n`$

常数 $`c`$ 乘随机变量时，方差乘 $`c^2`$：

```math
\mathrm{Var}(cX)=c^2\mathrm{Var}(X).
```

所以：

```math
\begin{aligned}
\mathrm{Var}(\bar X)
&=\mathrm{Var}\left(\frac1n\sum_iX_i\right)\\
&=\frac1{n^2}\mathrm{Var}\left(\sum_iX_i\right)\\
&=\frac1{n^2}\sum_i\mathrm{Var}(X_i)\\
&=\frac1{n^2}(n\sigma^2)\\
&=\frac{\sigma^2}{n}.
\end{aligned}
```

**MSE（mean squared error，均方误差）**是估计值与真值之差的平方的期望。因为刚刚证明 $`\mathbb E[\bar X]=\mu`$，所以：

```math
\begin{aligned}
\mathrm{MSE}
&=\mathbb E[(\bar X-\mu)^2]\\
&=\mathbb E[(\bar X-\mathbb E[\bar X])^2]\\
&=\mathrm{Var}(\bar X)\\
&=\frac{\sigma^2}{n}.
\end{aligned}
```

**SE（standard error，标准误差）**是估计量 $`\bar X`$ 自身的标准差，不是 MSE：

```math
\mathrm{SE}(\bar X)
=\sqrt{\mathrm{Var}(\bar X)}
=\sqrt{\frac{\sigma^2}{n}}
=\frac\sigma{\sqrt n}.
```

### 6.5 $`n=4`$ 与 $`n=16`$ 手算

令 $`\sigma=2`$，则 $`\sigma^2=4`$：

| $`n`$ | variance/MSE $`=4/n`$ | 标准误差 $`=\sqrt{4/n}`$ |
|---:|---:|---:|
| 4 | $`4/4=1`$ | $`\sqrt1=1`$ |
| 16 | $`4/16=0.25`$ | $`\sqrt{0.25}=0.5`$ |

数据增 4 倍，MSE 除以 4；标准误差只除以 2。不要混淆二者。

取 log：

```math
\log\mathrm{MSE}=\log\sigma^2-\log n,
```

斜率是 $`-1`$。视频 [17:21](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1041s) 用它解释 scaling law 的一个统计来源；神经网络数据曲线的指数通常小得多，不能直接套 $`-1`$。

---

## 7. Nonparametric 分箱：为什么可能更慢

### 7.1 nonparametric 不是“没有参数”

**Nonparametric（非参数）**不是零参数，而是模型复杂度可随数据增加。这里的 regression（回归）是从输入位置预测一个数值函数。要估计任意平滑二维函数 $`f(x_1,x_2)`$，我们可以把平面切成小箱，每箱估一个均值；箱越多，要估的局部值越多。

视频 [18:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1105s) 从均值转到 arbitrary smooth function。

### 7.2 PDF p19 的二维分箱推导

**【课程】【PDF 19；已查看高清原页】**设总样本数为 $`n`$。下面的 $`h`$ **只表示这个二维例子里每个正方形箱的边长**，不是最后的误差率。课件给定：

```math
h=n^{-1/4}.
```

一步一步数：

1. 一条边能放：

   $`\frac1h=\frac1{n^{-1/4}}=n^{1/4}`$

   个箱。
2. 二维总箱数：

   $`n^{1/4}\times n^{1/4}=n^{1/2}=\sqrt n.`$

3. $`n`$ 个样本平均分到 $`\sqrt n`$ 个箱，每箱约：

   $`\frac n{\sqrt n}=\sqrt n`$

   个样本。
4. 假设观测噪声 variance 是不随 $`n`$ 增长的常数量级。每箱有约 $`\sqrt n`$ 个独立样本，所以沿用 §6 的“均值 variance = 单样本 variance / 样本数”，每箱均值的 **variance 型误差**量级约为：

   $`\frac1{\sqrt n}.`$

请把两个幂分开记：

```text
二维箱边长 h         = n^(-1/4)
这个例子的方差型误差 = n^(-1/2) = 1/sqrt(n)
```

仅从“有多少箱、每箱多少样本”能推出方差项；**不能仅靠数箱子推出完整误差**，因为同箱内函数并不完全相等，还存在下一节的 smoothness/bias 项。

### 7.3 用 $`n=10,000`$ 复算

```math
n^{1/4}=10,
\qquad h=1/10.
```

- 每边 10 箱；总箱数 $`10\times10=100=\sqrt{10,000}`$。
- 每箱约 $`10,000/100=100=\sqrt{10,000}`$ 个样本。
- 每箱均值方差量级约 $`1/100=0.01=1/\sqrt{10,000}`$。这里的 0.01 是 variance 型误差，不是边长；边长是 0.1。

### 7.4 bias/smoothness 不能丢

箱很大时，同箱内不同位置的真实函数值不同，拿一个均值代表全箱会产生 **bias（系统偏差）**。箱很小时，箱内样本少，variance 变大。因此完整误差是：

```text
误差 ≈ variance 项 + smoothness/bias 项
```

课件 p19 的 $`1/\sqrt n+`$ smoothness 是教学启发，不是所有二维回归的万能精确公式。

**【补充解释】一个可能的平衡直觉。**如果额外假设函数足够平滑，使 squared-bias 项约为 $`h^2`$，二维每箱样本数约为 $`nh^2`$，variance 项约为 $`1/(nh^2)`$。让两种错误同量级：

```math
h^2=\frac1{nh^2}
\Rightarrow nh^4=1
\Rightarrow h=n^{-1/4}.
```

这时两项都是 $`h^2=n^{-1/2}`$。这个推导依赖“squared bias 约 $`h^2`$”等 smoothness 假设；换平滑度、估计器或 loss，幂次可能改变。

### 7.5 d 维与 intrinsic dimension 的边界

PDF p19 进一步写了一个简化的 $`d`$ 维 **误差率** $`n^{-1/d}`$；它不是在说 $`d`$ 维箱边长等于 $`n^{-1/d}`$。代入 $`d=2`$：

```math
n^{-1/d}=n^{-1/2}=\frac1{\sqrt n},
```

恰好回到上面二维例子的 variance 型误差。这个 $`d`$ 维写法是课件的直觉化简式，依赖分箱方式、smoothness、loss 与噪声等条件；没有这些条件，不能从箱数推出一个普遍定理。严谨 nonparametric rate 往往还显式包含 smoothness 阶数。

**Intrinsic dimension（内在维度）**：数据真正变化所需的自由方向数，可能小于原始向量坐标数。视频 [20:13](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1213s) 明确说相关解释依赖难以可靠估计的 intrinsic dimension；所以“从指数反推出语言维度”只能当不严密直觉，不是定理。

**Regime（区间/状态范围）**是同一组近似规律仍适用的一段规模和训练状态。课堂问题 [20:42](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1242s) 问模型大于数据是什么意思。老师的边界是：当模型相对数据过小，模型容量会成为瓶颈并接近另一条 asymptote；本讲主要研究尚处于 power-law regime 的区域。

---

## 8. 数据不是只有“多少”：composition、shift 与 mixture

### 8.1 distribution shift

**Distribution（分布）**描述哪些样本常见、哪些少见。**Distribution shift（分布偏移）**表示训练数据与目标测试/部署数据的分布不同。

例：训练 90% 是新闻、10% 是代码；部署问题 80% 是代码。即便 token 数相同，模型面对的学习问题已经不同。

**【课程】【PDF 21–23】【视频补充】[21:39](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1299s)** Scaling law 能把数据选择变成可测问题：分别用不同来源或比例训练小模型，再观察曲线。

### 8.2 offset 与 slope 不能泛化

在：

```math
\log(L-L_\infty)=\log A-\alpha\log D
```

中：

- 改变 **offset（上下位置）**主要对应 $`A`$ 变。
- 改变 **slope（斜率）**对应 $`\alpha`$ 变。

PDF p22 引用的特定研究观察到数据分布变化主要移动 offset、斜率较稳定。视频 [22:45](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1365s) 也用这个口径讲解。但这只是特定任务、模型与数据范围的观察，不是“所有数据清洗都只改截距”的定律。若过滤改变样本难度、覆盖面或进入另一 regime，指数也可能变。

### 8.3 data mixture

**Data mixture（数据混合）**：训练时从多个来源按比例抽样。例如 70% 网页、20% 代码、10% 书籍。

小例：总共训练 1000 tokens：

| mixture | 网页 | 代码 | 书籍 |
|---|---:|---:|---:|
| A：70/20/10 | 700 | 200 | 100 |
| B：40/50/10 | 400 | 500 | 100 |

在多个总 token 规模分别训练 A/B，才能比较“曲线随规模怎么变”。视频 [23:27](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1407s) 说可以对 mixture level 拟合函数。

### 8.4 为什么小模型排序可能失效

假设小规模 loss：A=2.0，B=2.1，于是 A 暂时更好。若：

```math
L_A=1+4D^{-0.2},\qquad L_B=1+6D^{-0.3},
```

B 虽然初始 offset 更高，却下降更快，规模足够大时可能反超。实际曲线还会受 tokenizer、目标域、数据质量和模型容量影响。因此视频 [25:15](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1515s) 明确警告：小模型的最佳 mixture 不保证也是大模型最佳 mixture。

---

## 9. 数据重复、effective data 与 compute-unbounded 失效

### 9.1 repetition 与 effective data

**Data repetition（数据重复）**：同一批 unique tokens 被看多次。**Effective data（有效数据量）**：把重复样本递减的信息价值折算成“相当于多少全新 tokens”。

**Epoch（训练遍数）**表示把当前可用的 unique training data 完整看一遍；总计 1 epoch 是首遍，2 epochs 是首遍后再重复一次。

**【课程】【PDF 24；高清核验】**课件给出：

```math
D'=U_D+U_D R_D^*\left(1-e^{-R_D/R_D^*}\right).
```

- $`D'`$：effective data，单位 tokens。
- $`U_D`$：unique data，单位 tokens。
- $`R_D`$：**首个 epoch 之后的额外重复次数**，无量纲；论文定义单 epoch 时 $`R_D=0`$，因此总 epochs 数是 $`R_D+1`$。
- $`R_D^*`$：控制何时饱和的无量纲常数。
- $`e\approx2.718`$，$`e^{-x}=1/e^x`$。

括号没有单位，所以 $`U_D\times`$括号仍是 tokens，维度正确。

### 9.2 三个极限

1. **不额外重复 $`R_D=0`$：**

   $`D'=U_D+U_DR_D^*(1-e^0)=U_D.`$

2. **有限重复：**会增加 $`D'`$，但小于把每遍都当全新数据。
3. **重复很多 $`R_D\to\infty`$：**$`e^{-R_D/R_D^*}\to0`$，所以：

   $`D'\to U_D(1+R_D^*).`$

   有效数据饱和，不会无限增长。

### 9.3 数字例

设 $`U_D=100`$ tokens、$`R_D^*=2`$：

| $`R_D`$ | 代入 | $`D'`$ |
|---:|---|---:|
| 0（总计 1 epoch） | $`100+200(1-e^0)`$ | 100 |
| 1（总计 2 epochs） | $`100+200(1-e^{-0.5})`$ | $`178.69`$ |
| 2（总计 3 epochs） | $`100+200(1-e^{-1})`$ | $`226.42`$ |
| 很大 | $`100+200(1-0)`$ | 300 |

计算器输入 `exp(-0.5)` 得约 0.6065。注意 $`R_D=1`$ 表示“首遍之后再重复一次”，也就是总共 2 epochs；这次重复带来的有效新增只有 78.69，而不是把 100 个 tokens 全算成全新。论文附录也用“5 epochs 对应 $`R_D=4`$”明确了这个口径。

Muennighoff 等的原始实验在数据受限 regime 中观察到少量重复的损失变化很小，但更多重复最终收益趋近零；这是该公式的实证背景，不应跨数据集直接复用常数。[原论文](https://arxiv.org/abs/2305.16264)

### 9.4 compute-unbounded scaling 为什么会 break

**Compute-unbounded law** 假装 compute 可以无限加，并继续按同一 power law 改善。若 unique data 固定、重复价值饱和，继续增加 steps 会离开原拟合 regime；曲线必须弯折或接近新平台。视频 [26:50](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1610s) 展示红色重复曲线偏离无限数据的 law。

### 9.5 lower envelope 的第一层含义

**Lower envelope（下包络）**：在同一资源横坐标上，所有实验曲线中最低的可达 loss 边界。若三个 recipe 在 compute=100 时 loss 为 2.4、2.1、2.3，下包络点是 2.1。

它不是自然界不可突破的数学下界，只表示“当前候选 recipes 中最好”。更好数据或优化器可以把 envelope 向下推。视频 [27:15](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1635s) 强调数据处理能改变 intercept。

---

## 10. Model engineering 的曲线怎么读

### 10.1 architecture 与 optimizer

回忆 §3.3：gradient 是 loss 对所有 parameters 的局部变化率向量，它告诉参数往哪个方向动会使 loss 增减；learning rate 是每次更新沿 gradient 方向走多大一步。**Architecture（架构）**是数据怎样流过模型的结构。**Hyperparameter（超参数）**是训练前由人选择、而不是通过 gradient 学出的设置，例如 learning rate、batch size、层数。

**【课程】【PDF 29–36】【视频补充】[30:16](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1816s)** 课程把 scaling law 当工程比较工具：不是只在一个小模型上比较 A/B，而是画多种规模的曲线。

- PDF p30 的 Transformer 与 LSTM 图：Transformer 是用 attention 让 tokens 相互读取的架构；LSTM（Long Short-Term Memory，长短期记忆网络）是按序列步递归更新状态的较早架构。横轴是 compute/规模，纵轴是 loss；比较的是整条趋势，不是一个点。
- PDF p31 的多个 architecture 图：许多改动在小规模的微小差异可能随规模扩大、缩小或交叉。
- PDF p32 的 Adam 与 SGD 图：mini-batch 是一次参数更新中使用的一小批样本；SGD（Stochastic Gradient Descent，随机梯度下降）直接用 mini-batch gradient 更新；Adam 还保存梯度均值/平方均值的移动统计来调节各坐标步幅。Optimizer 会改变达到某个 loss 的资源效率，比较时必须让各自超参数足够公平。

视频 [31:30](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1890s) 半开玩笑地说本讲得“相信”scaling laws；正确理解是把它当待验证的经验模型。

### 10.2 depth、width 与 aspect ratio

- **Depth（深度）**：层数。
- **Width（宽度）**：hidden dimension 等层内维度。
- **Aspect ratio（形状比例）**：例如 width/depth 或 FFN width/model width。

PDF p33 显示从 1 层到 2 层差异很大，更多层在该实验的某些参数范围回报递减；p34 显示固定 non-embedding parameters 时，较宽范围的形状得到相近 performance。正确结论是“该研究范围内形状较宽容”，不是“深宽永远不重要”。视频 [35:38](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2138s) 开始讨论深宽。

### 10.3 parameter count 口径会改变图

**Embedding** 把 token ID 映射为向量；**output/softmax matrix** 把 hidden vector 映射回 vocabulary logits。Logit 是 softmax 前未归一化的分数；softmax 把一组 logits 变成总和为 1 的 probabilities。

PDF p35 左图横轴包含 embedding parameters，右图不含；右图不同层数更接近共同趋势。原因不是 embedding “没用”，而是不同类型参数的边际价值可能不同。

例：词表 $`V=50,000`$、宽度 $`h=1,000`$，embedding 有：

```math
Vh=50,000\times1,000=50,000,000
```

个参数。小模型主体若也只有 50M，是否计入 embedding 会让横坐标翻倍；大模型主体 5B 时影响只约 1%。这会让低 compute 区的拟合指数明显变化。

视频 [37:58](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2278s) 展示包含 embedding 时曲线很怪；[38:28](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2308s) 的核心提醒是 scaling laws 不是 magic。

### 10.4 MoE：total 与 active parameters 不同

**MoE（Mixture of Experts，专家混合）**包含多个 expert（专家，即不同的 FFN；FFN 是 feed-forward network，逐 token 运行的前馈子网络），并用 router（路由器，即给 experts 打分并选择去向的小网络）为每个 token 选择少数 experts。**Sparsity（稀疏度）**表示大量 experts 对当前 token 没被激活。Total parameters 决定存储容量；active parameters 决定该 token 实际经过多少权重，进而影响 compute。

例如 8 个各 100M 的 experts，top-2：

- experts total parameters：$`8\times100\text{M}=800\text{M}`$。
- 每 token active expert parameters：$`2\times100\text{M}=200\text{M}`$。

PDF p36 的图分别以 total 和 active parameters 为横轴，并加入 sparsity；说明“一个参数”在 dense 与 MoE 中并非同一种 compute 价值。视频 [39:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2349s) 开始解释这一点。

### 10.5 pretraining loss 与 downstream

**Downstream task（下游任务）**是预训练后用于问答、分类、推理等具体评测。**Cross-entropy loss（交叉熵损失）**衡量模型给真实下一个 token 的 probability 有多低；给真 token 的 probability 越高，loss 通常越小。Perplexity（困惑度）是由平均 cross-entropy 指数化得到的语言模型指标，越低通常越好；PDF p41 左轴画其对数的负数，右边画 SuperGLUE benchmark（由多项语言理解任务组成的评测集合）accuracy。左图随参数较平滑，右图排序更乱。

因此：

```text
能预测 pretraining loss
≠ 能精确预测每个 benchmark accuracy
≠ 已证明能力会/不会“涌现”
```

视频 [50:55](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3055s) 转入 downstream caution；课堂问答 [54:49](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3289s) 说明也有人直接拟合 downstream metrics，但噪声与不稳定性可能更强。

---

## 11. Critical batch：steps 与 examples 的交换

### 11.1 四个量不要混

**【课程】【PDF 37–39】【视频补充】[42:22](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2542s)**

- $`B`$：global batch size，一步中全设备合计处理的 examples 或 tokens。
- $`S`$：training steps，参数更新次数。
- $`E`$：examples processed，总共处理的样本数。
- 三者关系：

  $`E=S\times B.`$

若每步 32 examples、走 100 steps，则 $`E=32\times100=3200`$ examples。语言模型也常把 $`B`$ 用 tokens/step 表示；必须写清单位。

> **§11 当前符号卡：**$`B`$=batch、$`S`$=steps、$`E`$=processed examples；$`B_{crit}`$=临界 batch、$`S_{min}`$=最少 steps 极限、$`E_{min}`$=最少 examples 极限。这里的 $`E`$ **不是** §13 的 irreducible loss。

### 11.2 noise-limited 与 bias-limited 直觉

回忆 §10.1：gradient 是 loss 对所有 parameters 的局部变化率向量，优化器通常沿其反方向减小 loss。单个小 batch 的 gradient 是有限样本估计，带随机噪声。**Gradient noise（梯度噪声）**是不同 mini-batches 给出的 gradient 波动。Batch 较小时，多放样本能显著减少这种噪声，steps 近似按比例下降，称 noise-limited。

Batch 很大后，gradient 已较准确，继续放更多相似样本不能让更新方向无限变好；优化器沿损失曲面的系统方向/曲率限制占主导，可粗称 bias-limited。此处 bias 是优化更新的系统性限制，不是 §7 分箱 bias 的同一具体量。

论文用 gradient covariance（梯度协方差矩阵）描述各坐标共同波动；**trace（迹）**是矩阵对角线之和，即各 gradient 坐标 variance 的总和。无需用它复算本讲曲线，但它帮助估计 gradient noise scale。[原论文](https://arxiv.org/abs/1812.06162)

### 11.3 精确 trade-off 曲线

PDF p38 给：

```math
\frac{S}{S_{\min}}-1
=\left(\frac{E}{E_{\min}}-1\right)^{-1}.
```

- $`S_{\min}`$：batch 极大时仍不能再减少的最少 steps。
- $`E_{\min}`$：batch 很小时最省样本的最少 examples。
- 右边 $`x^{-1}=1/x`$。

等价地，两边相乘：

```math
\left(\frac{S}{S_{\min}}-1\right)
\left(\frac{E}{E_{\min}}-1\right)=1.
```

临界 batch：

```math
B_{\text{crit}}=\frac{E_{\min}}{S_{\min}}.
```

单位检查：examples 除以 steps = examples/step，正是 batch 的单位。

现在从这条曲线**推出**任意指定 batch 对应的 $`S,E`$，而不是直接背答案。定义三个没有单位的比值：

```math
s=\frac{S}{S_{min}},
\qquad
e=\frac{E}{E_{min}},
\qquad
b=\frac{B}{B_{crit}}.
```

先用 $`B=E/S`$ 与 $`B_{crit}=E_{min}/S_{min}`$：

```math
\begin{aligned}
b
&=\frac{B}{B_{crit}}\\
&=\frac{E/S}{E_{min}/S_{min}}\\
&=\frac{E}{E_{min}}\frac{S_{min}}S\\
&=\frac es.
\end{aligned}
```

因此：

```math
e=bs.
```

原双曲线变为 $`(s-1)(e-1)=1`$。把 $`e=bs`$ 代进去：

```math
(s-1)(bs-1)=1.
```

逐项展开左边：

```math
bs^2-s-bs+1=1.
```

两边减 1，再把中间两项合并：

```math
bs^2-(b+1)s=0.
```

提出共同因子 $`s`$：

```math
s\bigl(bs-(b+1)\bigr)=0.
```

$`S,S_{min}`$ 都是正数，所以 $`s=S/S_{min}>0`$，不能取 $`s=0`$。因此第二个因子必须为 0：

```math
bs=b+1
\Rightarrow
s=\frac{b+1}{b}=1+\frac1b.
```

最后代回 $`e=bs`$：

```math
e=b\left(1+\frac1b\right)=1+b.
```

所以完整结果是：

```math
\boxed{\frac{S}{S_{min}}=1+\frac1b},
\qquad
\boxed{\frac{E}{E_{min}}=1+b}.
```

### 11.4 从曲线推出三个 batch

把 §11.3 刚推出来的结果代入三个 batch：

取 $`S_{\min}=100`$ steps、$`E_{\min}=1000`$ examples，所以 $`B_{\text{crit}}=1000/100=10`$ examples/step：

| $`B`$ | $`b=B/10`$ | $`S=100(1+1/b)`$ | $`E=1000(1+b)`$ | 检查 $`E/S`$ |
|---:|---:|---:|---:|---:|
| 5 | 0.5 | $`100(1+2)=300`$ | $`1000(1.5)=1500`$ | $`1500/300=5`$ |
| 10 | 1 | $`100(2)=200`$ | $`1000(2)=2000`$ | $`2000/200=10`$ |
| 20 | 2 | $`100(1.5)=150`$ | $`1000(3)=3000`$ | $`3000/150=20`$ |

再验证精确曲线：

- $`B=5`$：$`(300/100-1)(1500/1000-1)=2\times0.5=1`$。
- $`B=10`$：$`(2-1)(2-1)=1`$。
- $`B=20`$：$`(1.5-1)(3-1)=0.5\times2=1`$。

### 11.5 临界点为什么是“两边约 2 倍”

当 $`B=B_{\text{crit}}`$，$`b=1`$：

```math
S=2S_{\min},\qquad E=2E_{\min}.
```

它在 time efficiency（少 steps）与 sample/compute efficiency（少 examples）之间平衡。不是说此 batch “最快且最省样本”同时达到各自绝对最优，而是两种代价都在各自极限的 2 倍。

视频 [43:44](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2624s) 称 critical batch 是 rule of thumb；[46:41](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2801s) 建议读原论文理解精确对象；[47:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2860s) 说明随着 loss 降低，critical batch 往往增大。

---

## 12. 学习率与 muP：本讲只预告

回忆 §10.1：**learning rate（学习率，LR）**决定每次参数沿 negative gradient 方向走多大步。普通 parameterization（参数化方式，即权重初始化和缩放规则）下，宽度改变会改变 activation、gradient 的尺度，所以在小模型上最佳的 LR 到大模型可能漂移。

PDF p40 左图显示 standard practice 中不同 width 的最佳 LR 横向移动；右图的 scale-aware 方法让 optimum 更稳定。

**muP（Maximal Update Parameterization，最大更新参数化）**是一套随 width 调整初始化与学习率尺度的规则，目标是让小模型调好的许多 hyperparameters 能转移到大模型。视频 [49:57](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2997s) 首次命名 muP。

本讲只需记住：

```text
普通参数化：最佳 LR 可能随 width 漂移
muP：尝试把不同 width 的更新尺度对齐
具体哪些 tensor 怎样缩放：留到 Lecture 11
```

不要从这一页编造“所有超参数都 width-invariant”。原论文说的是在特定 muP 规则下，许多最佳 hyperparameters 更稳定，不是每个设置都自动可转移。[原论文](https://arxiv.org/abs/2203.03466)

---

## 13. 联合 data-model scaling 与固定 compute 最优解

### 13.1 统一符号

**【课程】【PDF 42–44】【视频补充】[56:44](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3404s)** 为避免课件不同论文把 $`n,m`$ 互换，本文统一写：

```math
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
```

- $`N`$：model parameters。
- $`D`$：training tokens。
- $`E`$：irreducible loss；这里不再用 $`L_\infty`$，含义相同。
- $`A,B`$：两项的尺度常数。
- $`\alpha,\beta>0`$：model 与 data exponent。

> **§13 当前符号卡：**$`N`$=参数量，$`D`$=训练 tokens，$`C`$=训练 FLOPs，$`L`$=loss，$`E`$=irreducible loss，$`A/B`$=两项系数，$`\alpha/\beta`$=model/data exponent。它们与 §11 的 batch 符号是两套局部记号。

Rosenfeld 的课件简写是 `Error = n^{-α}+m^{-β}+C`；Kaplan p43 的组合形式更复杂。本文用上式教学，因为符号和 Chinchilla 后续一致；不是说两篇论文原式逐字相同。[Rosenfeld 原论文](https://arxiv.org/abs/1909.12673)

**符号复用警告：**§11 的 $`B`$ 是 batch size、$`E`$ 是 examples；从本节起，$`B`$ 是 data-loss 项的系数、$`E`$ 是 irreducible loss。判断它们要看单位与所在公式。

PDF p43 对 Kaplan 形式按其自己的 $`m,n`$ 记号原样写作：

```math
\text{Error}=\left[m^{-\alpha}+n^{-1}\right]^\beta.
```

这里不能未经论文定义就把课件的 $`m,n`$ 直接换成本笔记的 $`N,D`$；课程想强调的是 Kaplan 也把 model/data 饱和行为放进一个联合函数，而不是它与 Rosenfeld 的加法式逐项相同。

### 13.2 为什么训练 compute 粗略是 $`C\approx6ND`$

适用对象是 **dense Transformer training**：每 token 大致激活全部 $`N`$ 个非 embedding 主体参数。先把训练的数据流说清：

- **Forward（前向）**：输入 tokens 依次经过各层，产生预测并算出 loss。
- **Activation（激活/中间值）**：forward 中某层输出、会交给下一层的数字。
- **Backward（反向传播）**：从 loss 往回走，用链式法则计算“每个中间值或参数变化一点，loss 会变多少”。链式法则的人话是：若 A 改一点会让 B 改，B 改一点又让 loss 改，总影响就是两段局部变化率相乘。
- **Activation gradient**：loss 对某个 activation 的梯度；它把“错误信号”传给更前面的层。
- **Weight gradient**：loss 对某个 weight/parameter 的梯度；optimizer 用它更新该参数。

现在才开始算 FLOPs。**Multiply-add（乘加）**是一次乘法后把结果加到累加器，按“一乘 + 一加 = 2 FLOPs”计：

1. forward 中参数矩阵乘每个参数约参与一次 multiply-add。
2. 一个 multiply-add 约按 2 FLOPs（一次乘、一次加）计，因此 forward 约 $`2ND`$。
3. backward 要算 activation gradient 与 weight gradient，粗略约 forward 的 2 倍，即 $`4ND`$。
4. 合计：

   $`C\approx2ND+4ND=6ND.`$

这不是 inference 公式；它忽略/合并 attention 的序列平方项、embedding/output、loss、optimizer、重计算、稀疏 MoE、通信与数据移动。参数矩阵乘不主导时会失真。

### 13.3 用代入法推 $`N_{\text{opt}}`$ 与 $`D_{\text{opt}}`$

固定训练 compute $`C`$：

```math
D=\frac{C}{6N}.
```

代回可改善 loss：

```math
f(N)=A N^{-\alpha}+B\left(\frac{C}{6N}\right)^{-\beta}
=A N^{-\alpha}+B\left(\frac{6N}{C}\right)^\beta.
```

第一项随 $`N`$ 增大而降，第二项因为 $`D`$ 被挤小而升，所以有平衡点。

用导数找最低点。这里唯一需要的求导规则是：若 $`k,r`$ 都是不随 $`N`$ 变化的常数，

```math
\frac{d}{dN}\left(kN^r\right)=krN^{r-1}.
```

人话理解：导数是数轴上曲线在当前位置的坡度。内部最低点左侧还在下坡，坡度为负；右侧已经上坡，坡度为正；两者交界处坡度为 0。这里 $`N>0`$ 且两项一降一升，所以在我们讨论的内部平衡点令导数为 0。若实验最优点落在扫描边界，就不能用“内部坡度为 0”代替扩宽扫描。

把第二项先写成 $`B(6/C)^\beta N^\beta`$，再逐项套规则：

```math
\frac{df}{dN}
=-\alpha A N^{-\alpha-1}
+\beta B\left(\frac6C\right)^\beta N^{\beta-1}.
```

最低点斜率为 0：

```math
\alpha A N^{-\alpha-1}
=\beta B\left(\frac6C\right)^\beta N^{\beta-1}.
```

两边乘 $`N^{\alpha+1}`$：

```math
\alpha A
=\beta B\left(\frac6C\right)^\beta N^{\alpha+\beta}.
```

整理：

```math
N^{\alpha+\beta}
=\frac{\alpha A}{\beta B}\left(\frac C6\right)^\beta.
```

为了把左边的 $`\alpha+\beta`$ 次方去掉，两边同时开 $`\alpha+\beta`$ 次方，也就是整体取 $`1/(\alpha+\beta)`$ 次方：

```math
N
=\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\left(\frac C6\right)^{\beta/(\alpha+\beta)}.
```

前面的 $`A,B,\alpha,\beta,6`$ 都不随 $`C`$ 变，所以谈随 compute 的比例时：

```math
\boxed{N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}}.
```

再由 $`D=C/(6N)`$，把不随 $`C`$ 变化的 $`1/6`$ 收进比例常数：

```math
D\propto\frac CN
\propto\frac{C^1}{C^{\beta/(\alpha+\beta)}}
=C^{1-\beta/(\alpha+\beta)}.
```

把指数中的 1 写成 $`(\alpha+\beta)/(\alpha+\beta)`$：

```math
1-\frac\beta{\alpha+\beta}
=\frac{\alpha+\beta-\beta}{\alpha+\beta}
=\frac\alpha{\alpha+\beta}.
```

因此：

```math
\boxed{D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}}.
```

两个 exponent 相加为 1，保证 $`ND\propto C`$。

### 13.4 $`\alpha=\beta=0.5`$ 的整数例

为方便手算，用归一化单位令 $`C/6=ND`$，且 $`A=B=1`$。

初始 budget：$`ND=100`$。先不靠“对称所以最小”这句话，实际扫五个候选。不可约常数 $`E`$ 对每行都相同，不影响谁最低，所以只算额外 loss：

```math
f(N,D)=\frac1{\sqrt N}+\frac1{\sqrt D}.
```

| $`N`$ | $`D=100/N`$ | model 项 | data 项 | $`f`$ |
|---:|---:|---:|---:|---:|
| 1 | 100 | $`1/\sqrt1=1`$ | $`1/\sqrt{100}=0.1`$ | 1.1 |
| 4 | 25 | $`1/2=0.5`$ | $`1/5=0.2`$ | 0.7 |
| 10 | 10 | $`1/\sqrt{10}\approx0.316`$ | 同为 0.316 | **0.632** |
| 25 | 4 | $`1/5=0.2`$ | $`1/2=0.5`$ | 0.7 |
| 100 | 1 | $`1/10=0.1`$ | $`1/1=1`$ | 1.1 |

表中从 1.1 降到 0.632，再升回 1.1；所以 $`(10,10)`$ 至少是这五个候选中最低的。连续范围内的严格最优已经由 §13.3 的导数证明；这张表只负责建立数值直觉，不能单靠五个离散点排除两点之间还有更低位置。

compute 增 4 倍：$`ND=400`$。公式指数：

```math
\frac\beta{\alpha+\beta}=\frac{0.5}{1}=0.5.
```

所以 $`N,D`$ 都乘：

```math
4^{0.5}=2.
```

新最优 $`N=D=20`$，并且 $`20\times20=400`$。这只是比例教学例；现实 parameters 与 tokens 的数值单位不同，不要求 $`N=D`$。

---

## 14. Kaplan 与 Chinchilla：两个不同资源处方

### 14.1 Kaplan 的课件口径

**【课程】【PDF 45】【视频补充】[60:21](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3621s)** 课件列：

```math
N_{\text{opt}}\propto C^{0.73},
\qquad
D_{\text{opt}}\propto C^{0.27}.
```

compute 增 100 倍：

```math
N\text{ 倍数}=100^{0.73}=10^{1.46}\approx28.84,
```

```math
D\text{ 倍数}=100^{0.27}=10^{0.54}\approx3.47.
```

因此更偏向增大 model，而 data 增得较慢。回忆 §10.5 的 cross-entropy loss：Kaplan 原论文研究的是这种 loss 的经验规律，不是所有任务的定理。[Kaplan 原论文](https://arxiv.org/abs/2001.08361)

### 14.2 Chinchilla 的核心结论

Hoffmann 等训练 400 多个模型，在其研究范围内得到 compute-optimal model size 与 tokens 大致同比例扩张，常概括为：

```math
N_{\text{opt}}\propto C^{0.5},
\qquad
D_{\text{opt}}\propto C^{0.5}.
```

compute 增 100 倍时，两者各增 $`\sqrt{100}=10`$ 倍。这与 Kaplan 的资源分配明显不同。[Chinchilla 原论文](https://arxiv.org/abs/2203.15556)

### 14.3 tokens/parameter 会怎么变

比率：

```math
\frac DN\propto C^{d-n},
```

其中 $`d,n`$ 分别是 data/model exponent。

- Kaplan：$`D/N\propto C^{0.27-0.73}=C^{-0.46}`$，compute 越大，比率越低。
- Chinchilla 0.5/0.5：$`D/N\propto C^0`$，比率近似保持常数。

“约 20 tokens/parameter”来自特定 Chinchilla 训练设置与前两种估计，不是由 $`0.5/0.5`$ 单独决定；常数还要由实验拟合。

---

## 15. Chinchilla 的三种方法，全部做一个小例

### 15.1 Method 1：lower envelope

**【课程】【PDF 47】【视频补充】[63:19](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3799s)** 收集不同模型的 training curves，在每个 compute 水平选最低 loss。

| compute | 模型 A loss | B loss | C loss | lower envelope |
|---:|---:|---:|---:|---:|
| 10 | 3.0 | 3.2 | 3.5 | 3.0（A） |
| 20 | 2.7 | 2.5 | 2.9 | 2.5（B） |
| 40 | 2.6 | 2.3 | 2.1 | 2.1（C） |

连接 envelope 点，再拟合最优 $`N`$ 随 $`C`$ 的关系。风险：没有在某段 compute 试到真正好模型，envelope 会偏高。

### 15.2 Method 2：IsoFLOP

**IsoFLOP** 表示相同 FLOPs。固定 $`C\approx6ND`$，等价于固定 $`ND`$，扫描不同 $`N,D`$。

用 $`ND=36`$、额外 loss $`f=1/\sqrt N+1/\sqrt D`$：

| $`N`$ | $`D=36/N`$ | $`f`$ |
|---:|---:|---:|
| 1 | 36 | $`1+1/6=1.167`$ |
| 2 | 18 | $`0.707+0.236=0.943`$ |
| 3 | 12 | $`0.577+0.289=0.866`$ |
| 4 | 9 | $`0.5+0.333=0.833`$ |
| 6 | 6 | $`0.408+0.408=0.816`$（最低） |
| 9 | 4 | $`0.333+0.5=0.833`$ |
| 18 | 2 | $`0.236+0.707=0.943`$ |
| 36 | 1 | $`0.167+1=1.167`$ |

在 log-$`N`$ 横轴上像一个碗，碗底给该 compute 的 $`N_{\text{opt}}`$。再换多个 compute budgets，拟合碗底怎样移动。视频 [64:28](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3868s) 开始 Method 2。

### 15.3 Method 3：联合参数曲面

固定形式：

```math
L=E+A N^{-\alpha}+B D^{-\beta},
```

让最小二乘同时拟合 $`E,A,B,\alpha,\beta`$。若暂时已知 $`E=1,\alpha=\beta=0.5`$，三个实验：

| $`(N,D)`$ | 预测 $`L`$（取 $`A=B=1`$） |
|---|---:|
| (4,4) | $`1+1/2+1/2=2`$ |
| (4,16) | $`1+1/2+1/4=1.75`$ |
| (16,4) | $`1+1/4+1/2=1.75`$ |

真实拟合还要从噪声数据中估五个量，容易出现参数互相补偿。数值优化器还可能停在 **local optimum（局部最优）**：在附近移动都更差，但更远处仍有更低的解。PDF p49 展示的是二维 $`N,D`$ 上的 loss surface，不是只拟合一条线。

### 15.4 三种方法为什么可能不一致

- Method 1 受候选 training curves 覆盖影响。
- Method 2 每个预算独立找碗底，直观但实验数多。
- Method 3 借所有数据联合拟合，样本利用率高，但更依赖函数形式与优化器是否找到正确解。

**Confidence interval（置信区间）**是按某套重复抽样程序表达估计不确定性的区间；它不是“真值保证有某个百分比一定在这里”。PDF p46 的 2022 课件表（括号为页上 confidence interval）是：

| 方法 | $`N_{opt}`$ exponent | $`D_{opt}`$ exponent |
|---|---:|---:|
| Approach 1 | 0.50（0.488, 0.502） | 0.50（0.501, 0.512） |
| Approach 2 | 0.49（0.462, 0.534） | 0.51（0.483, 0.529） |
| Approach 3 | 0.46（0.454, 0.455） | 0.54（0.542, 0.543） |
| Kaplan | 0.73 | 0.27 |

表中四舍五入后的中心值不一定落在打印区间正中央；本笔记照录，不擅自“修齐”。Method 3 区间异常窄正是后续 replication 质疑的一点。不要把三种方法写成完全一致。

---

## 16. 为什么 Kaplan 与 Chinchilla 会不同

### 16.1 口径与训练 recipe 会改变 exponent

**【课程】【PDF 50–53】【视频补充】[67:42](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4062s)** 重要因素：

1. **Parameter count：**是否计 embedding/output layer；小模型中这些占比很大。
2. **Warmup：**训练初期逐步升高 LR。若小模型训练很短，warmup 结束时它还没正常收敛，会让小模型看起来异常差。
3. **Batch size：**一个固定大 batch 对小模型可能远超合适范围。
4. **Compute range：**低 compute 区对固定开销和非线性特别敏感。
5. **Optimization fairness：**不同规模是否都调好 LR、decay、batch 和训练时长。这里 decay 是学习率到训练后期逐步降低的 schedule（时间安排）。

视频 [68:53](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4133s) 从 parameter count 开始；[69:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4180s) 讲 warmup；[70:03](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4203s) 讲固定 batch。

PDF p51 对 Porian 等结果给了一条定量“修正阶梯”；纵轴拟合的是 $`N^*(C)\propto C^a`$ 中的 model exponent $`a`$：

| 设置 | $`a`$（页上区间） |
|---|---:|
| 复现 Kaplan 设置 | 0.835（0.82, 0.85） |
| 计入 last-layer FLOPs | 0.706（0.69, 0.72） |
| 再纠正 warmup | 0.602（0.59, 0.62） |
| optimizer tuning，no decay | 0.497（0.49, 0.50） |
| cosine decay，no tuning | 0.571（0.56, 0.59） |

这张表显示小口径改动可逐步移动 exponent；它不是说五个数字可直接用于任意新模型。

### 16.2 2024 的复核要准确分层

Porian 等 2024 重现 Kaplan-style 分析，报告三项主要因素：last-layer compute、warmup duration、scale-dependent optimizer tuning；纠正后与 Chinchilla 更一致。原论文并不证明“所有未来数据集都必须是 0.5/0.5”。[原论文](https://arxiv.org/abs/2406.19146)

Besiroglu 等 2024 针对 **Chinchilla Method 3**，从论文图中重建数据并重新拟合。他们报告原 Method 3 参数与前两种方法不一致、拟合重建数据较差、置信区间不可信；重拟合更接近 Method 1/2。因为没有原始训练数据，这仍是对图中重建数据的 replication attempt，而非拿到了全部原数据后的最终裁决。[原论文](https://arxiv.org/abs/2404.10102)

视频 [72:17](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4337s) 把它讲成一段“Chinchilla saga”；[73:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4400s) 明确说研究者未取得 raw data/code，只能从 plots 提取。老师 [73:39](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4419s) 的“must have underfitted”是课堂口语判断；更严谨表述是“重建数据上的拟合证据支持原 Method 3 存在问题”。

### 16.3 lower bound 的严谨翻译

视频 [70:32](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4232s) 说 scaling laws “in some sense” 像 lower bounds。人话是：它描述某套 recipe 经过良好调优时可达到的 frontier；训练坏掉、warmup 不合适会落在曲线上方。

它不是数学证明的全领域最低 loss。新架构、新数据或新 optimizer 可以让 frontier 下降。

---

## 17. Train-optimal 与 deployment-optimal

### 17.1 两种目标不同

- **Train-optimal（训练最优）**：给定一次预训练 FLOPs，找最低预训练 loss。
- **Deployment/inference-optimal（部署/推理最优）**：把训练后所有请求的推理成本也算进去。

粗略总成本：

```math
C_{\text{total}}
=C_{\text{train}}
+Q\left(C_{\text{prefill per request}}+T_{\text{gen}}C_{\text{decode per token}}\right).
```

- $`Q`$：请求数。
- $`T_{\text{gen}}`$：每请求生成 tokens。
- prefill：一次读入 prompt；decode：逐 token 生成。

若只做极粗 dense matmul 估算，每生成 token forward 约 $`2N`$ FLOPs；实际还受 attention、KV cache、batching、hardware utilization 影响。

### 17.2 break-even 手算

两个达到相近质量的候选：

| 候选 | 训练成本 | 每请求推理成本 |
|---|---:|---:|
| A：大模型、少训练 | 1000 | 10 |
| B：小模型、多训练 | 1600 | 4 |

总成本：

```math
C_A=1000+10Q,
\qquad C_B=1600+4Q.
```

break-even（两者相等的分界点）：

```math
1000+10Q=1600+4Q
```

```math
6Q=600
\Rightarrow Q=100.
```

- 少于 100 请求：A 的额外训练节省占优。
- 多于 100 请求：B 每请求省 6，最终摊回多花的 600 训练成本。

所以服务量大时，宁可训练更久获得更小模型。视频 [74:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4465s) 转入这个结论。

### 17.3 “overtrained”不是过拟合

**Validation loss** 是在不用于 gradient 更新的验证集上计算的 loss，用来观察未训练数据上的表现。**传统 overfitting（过拟合）**是 training loss 继续改善，但 validation/test loss 反而变差。视频 [75:08](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4508s) 刻意给 “overtrained” 加引号：它只指训练 tokens 超过 **training-compute-optimal** 配比，不表示 validation loss 已经反弹，也不等于传统 overfitting。

### 17.4 课件 p54 的 tokens/parameter 快照与冲突

**【课程】【PDF 54；高清核验】**

| 模型 | PDF p54 tokens/parameter |
|---|---:|
| GPT-3 | 2 |
| Chinchilla | 20 |
| LLaMA 65B | 22 |
| Llama 2 70B | 29 |
| Mistral 7B | 110 |
| Llama 3 70B | 215 |

**材料内部冲突：**视频 [75:18](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4518s) 口述 GPT-3 为 **3** tokens/parameter，PDF 明确印 **2**。本笔记保留两者，不暗自修正。其余数字也只是课程时点的粗比例快照，会受 parameter/token count 口径影响。

---

## 18. IsoFLOP 实战流程：好用，但不是免检通行证

### 18.1 为什么好用

视频 [76:02](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4562s) 称 IsoFLOP 是经久耐用的研究工具；[76:12](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4572s) 给出操作：固定 FLOPs，再 sweep（扫描）自由度。

优点：

- 每条曲线直接回答“同样训练预算下哪个配置 loss 最低”。
- 碗形曲线容易看见扫描范围是否覆盖 optimum。
- 可扩展到 diffusion、MoE sparsity 等，不限 $`N,D`$。

### 18.2 从小网格到大 run

1. **先定义 x 轴。**$`N`$ 是 total 还是 non-embedding？$`D`$ 是 unique 还是 seen tokens？
2. **固定 recipe。**tokenizer、数据质量、optimizer family、目标 loss 尽量一致。
3. **选多个 compute budgets。**每个 budget 用 $`C\approx6ND`$ 生成候选网格。
4. **每个 budget 扫 $`N/D`$。**确保碗底左右都有点，不是最低点落在边界。
5. **拟合每个碗底。**不要只挑肉眼最好点；记录不确定度与 residual。
6. **跨 budgets 拟合 $`N_{\text{opt}}(C),D_{\text{opt}}(C)`$。**
7. **做 backtest（回测）。**故意只用较小预算拟合，把一个已实际跑过的较大预算留作 held-out scale（拟合时不让模型看到的规模），再比较预测与实测。
8. **保留验证预算。**不要把全部钱都花在拟合网格；留一个大 run 检验外推。
9. **大 run 仍监控。**若 early loss、gradient、throughput 偏离小模型趋势，及时停查。

### 18.3 何时失真

- 外推跨出太远；
- 数据 distribution/mixture/quality 随规模变化；
- 大小模型用不公平 LR、warmup、batch；
- tokenizer 或 sequence length 改了；
- train/test contamination（泄漏）让 loss/accuracy 虚好；
- 日志噪声、短 run 未收敛；
- 只看 $`R^2`$，不看 residual pattern；
- 候选网格没覆盖碗底；
- $`C\approx6ND`$ 在 attention、MoE 或重计算主导时不准；
- 目标其实是 deployment cost，却只最小化 train FLOPs。

PDF p55 展示 autoregressive、diffusion、MoE 的多种 IsoFLOP 曲面，说明方法通用；不说明每一种都能被同一个 $`6ND`$ 模型精确描述。视频 [76:44](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4604s) 给出的正确态度是“good default”，不是“永不失败”。

---

## 19. 一页决策树：遇到 scaling 问题怎么走

```text
先问：我要预测什么？
├─ pretraining loss
│  ├─ 只变 data → 先试 L∞ + A D^-β
│  ├─ 只变 model → 先试 E + A N^-α
│  └─ 固定训练 compute 分 N/D → 做多预算 IsoFLOP
├─ downstream accuracy
│  └─ 单独建模并报告噪声；不要把 pretraining-loss law 直接改标签
└─ 总产品成本
   └─ train cost + 请求数 × prefill/decode cost

然后问：各实验可比吗？
├─ tokenizer / data mixture / quality 相同？
├─ parameter count 口径相同？
├─ LR / warmup / batch 对各规模公平？
└─ 每个 run 都进入可比较的收敛区？

再问：拟合可靠吗？
├─ 覆盖至少多个规模与多个 compute budgets
├─ 看 residual，不只看 R²
├─ 让验证点落在拟合范围外做 backtest
└─ 明写外推倍数与不确定度

最后才做大 run：
保留验证预算 → 检查早期偏差 → 必要时停下重拟合
```

视频 [77:07](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4627s) 把最终目的概括为 evidence-driven engineering；[77:14](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4634s) 强调无需先做最大训练就能做规模决策。

---

## 20. 常见误区：错误说法 → 为什么错 → 正确说法

| # | 错误说法 | 为什么错 | 正确说法 |
|---:|---|---|---|
| 1 | Scaling law 是物理定律。 | 它来自有限模型、数据、recipe 的拟合。 | 把它当需 backtest 的经验模型。 |
| 2 | 插值准，所以外推 1000 倍也准。 | 外推可能跨 regime、数据饱和或系统瓶颈。 | 明写外推倍数并留外部验证点。 |
| 3 | 理论 generalization bound 就是经验曲线。 | 一个是证明的上界，一个是实际趋势拟合。 | 分开报告 bound 与 fit。 |
| 4 | log-log 直线说明原坐标也是直线。 | 取 log 改变了坐标。 | 原坐标是 power curve。 |
| 5 | $`\alpha`$ 越大，loss 越大。 | $`n^{-\alpha}`$ 在 $`n>1`$ 时随 $`\alpha`$ 增大而更小。 | $`\alpha`$ 大表示随规模改善更快。 |
| 6 | 数据翻 4 倍，总 loss 一定减半。 | 只有当 $`\alpha=0.5`$ 时，额外 loss 减半；还要加 $`L_\infty`$。 | 计算 $`r^{-\alpha}`$ 并只作用于可改善项。 |
| 7 | $`L_\infty=0`$。 | 数据本身有熵，模型/loss 口径也可能有不可约项。 | 把 $`L_\infty`$ 拟合或说明固定依据。 |
| 8 | MSE 与标准误差都是 $`1/n`$。 | 标准误差是 MSE/variance 的平方根。 | MSE $`\propto1/n`$，标准误差 $`\propto1/\sqrt n`$。 |
| 9 | 方差总能相加。 | 需要独立，或加入 covariance 项。 | 说明样本独立假设。 |
| 10 | Nonparametric 就是没有参数。 | 所需局部自由度可随数据增长。 | 它是不固定有限维参数族。 |
| 11 | p19 的 $`n^{-1/d}`$ 是万能定理。 | rate 还依赖 smoothness、loss、估计器。 | 把它当课程简化启发。 |
| 12 | scaling exponent 等于语言的真实 intrinsic dimension。 | 该连接依赖模型与估计假设，dimension 也难测。 | 只能作谨慎解释。 |
| 13 | 好数据只改 offset，永远不改 slope。 | 这是特定研究观察。 | mixture、质量、regime 都可能改两者。 |
| 14 | 小模型最佳 mixture 必是大模型最佳。 | 两条曲线可能交叉。 | 多规模比较 mixture。 |
| 15 | 重复两遍等于两倍新数据。 | 重复信息价值递减并饱和。 | 用 effective-data 模型或直接实验。 |
| 16 | 无限重复仍服从原 compute-unbounded law。 | unique 信息耗尽后会离开原 regime。 | 允许曲线弯折/饱和。 |
| 17 | Lower envelope 是不可突破的理论下界。 | 它只覆盖已试 recipes。 | 称“当前候选的经验 frontier”。 |
| 18 | 架构只要一个小模型点赢了就更好。 | 曲线可能交叉。 | 比较多个规模与相同 compute。 |
| 19 | 所有 parameters 价值相同。 | embedding、dense、active MoE 参数的 compute 角色不同。 | 明写 parameter count 口径。 |
| 20 | MoE total parameters 决定每 token FLOPs。 | 每 token 只激活部分 experts。 | 同时报 total 与 active parameters。 |
| 21 | pretraining loss 可直接换算 downstream accuracy。 | 下游评测可噪声更大、排序不同。 | 分别拟合并验证。 |
| 22 | Batch 越大，steps 永远按比例减少。 | 超过 critical batch 后回报递减。 | 同时看 steps 与 examples。 |
| 23 | $`E=SB`$ 中 $`E`$ 是 irreducible loss。 | §11 的 $`E`$ 是 examples，§13 的 $`E`$ 才是 irreducible loss。 | 看所在章节和单位。 |
| 24 | Critical batch 同时达到最少 steps 与最少 examples。 | 临界点是两者各约极限 2 倍的折中。 | 用精确曲线复算。 |
| 25 | $`B_{crit}`$ 是训练全过程固定常数。 | gradient noise scale 常随 loss/训练进度变。 | 可随训练阶段调整 batch。 |
| 26 | 普通参数化下 LR 一定跨 width 转移。 | update scale 会随 width 变。 | 调参或使用经过验证的 scale-aware parameterization。 |
| 27 | muP 保证所有超参数都不漂移。 | 论文结论有规则和适用范围。 | 本讲只作预告，细节留 L11。 |
| 28 | $`C=6ND`$ 是精确硬件耗时。 | 它是 dense 训练 FLOPs 粗估，不含通信/利用率。 | 区分 FLOPs 与 wall-clock。 |
| 29 | $`C=6ND`$ 也直接用于推理。 | backward 不存在，prefill/decode 结构不同。 | 推理单独建成本模型。 |
| 30 | 固定 $`ND`$ 就保证所有 IsoFLOP run 真同成本。 | embedding、attention、sequence length、重计算可不同。 | 记录实测 FLOPs/时间并说明近似。 |
| 31 | Kaplan 错、Chinchilla 永远对。 | 两者实验范围和 recipe 不同，后续复核只解释特定差异。 | 报告方法、范围与不确定度。 |
| 32 | Chinchilla Method 1/2/3 原本完全一致。 | p46 的 Method 3 exponent 不同。 | 分开列三种结果。 |
| 33 | 2024 replication 拿到了 Chinchilla 全部原数据。 | Besiroglu 等从论文图重建。 | 明确数据来源边界。 |
| 34 | 20 tokens/parameter 是黄金比例。 | 这是特定 train-optimal 快照，服务目标不同。 | 加 inference 请求量做 deployment optimization。 |
| 35 | “Overtrained”就是过拟合。 | 这里指超过训练 compute-optimal token 比。 | 检查 validation loss 是否恶化才谈传统过拟合。 |
| 36 | GPT-3 的课件值没有歧义。 | PDF 写 2，视频口述 3。 | 保留材料冲突。 |
| 37 | 高 $`R^2`$ 就证明外推可靠。 | 错误函数也可在窄区间高 $`R^2`$。 | 看 residual 与 out-of-range backtest。 |
| 38 | Scaling law 预测 emergence 已被严格证明。 | loss 的平滑变化不决定阈值评测的视觉形状。 | 对能力/accuracy 单独、谨慎建模。 |
| 39 | 数据量只要写“1T”就够。 | unique/seen tokens、tokenizer、重复和 mixture 不同。 | 首先定义 $`D`$。 |
| 40 | 一次最佳大 run 能替代网格。 | 没有对照就不知道是否在 frontier。 | 小网格、拟合、验证、大 run。 |

---

## 21. 术语表

| 术语 | 零基础解释 |
|---|---|
| scaling law | 资源与表现间、在特定实验范围观察到的规律 |
| learning curve | 数据/steps/compute 增加时，表现怎样变化的曲线 |
| interpolation | 在已测区间内部预测 |
| extrapolation | 向已测区间外预测 |
| empirical | 来自实验观察，不等于数学证明 |
| generalization bound | 对未见数据误差的理论上界 |
| sample complexity | 达到目标误差所需样本量 |
| power law | 形如 $`y=Ax^k`$ 的幂函数关系 |
| exponent | 幂次 $`k`$；控制缩放速度 |
| log-log plot | 横纵轴都取对数的图 |
| monotonic | 输入增加时，输出只朝一个方向变化 |
| asymptote | 曲线越来越接近的极限线 |
| irreducible loss | 无限资源假想下仍剩的损失 |
| parametric | 用固定有限个参数描述未知对象 |
| nonparametric | 有效自由度可随数据量增长 |
| intrinsic dimension | 数据变化真正需要的自由方向数 |
| distribution shift | 训练分布与目标分布不同 |
| data mixture | 多个数据来源的抽样比例 |
| repetition | 同一 unique data 被再次训练 |
| effective data | 折算重复递减价值后的等效新数据量 |
| lower envelope | 每个资源水平上候选实验的最低 loss 边界 |
| critical batch | steps 效率与 examples 效率折中的 batch 尺度 |
| gradient noise | 不同 mini-batches 导致 gradient 波动 |
| covariance | 两个随机量怎样共同波动 |
| trace | 方阵对角线之和；协方差矩阵中是各坐标方差总和 |
| muP | Maximal Update Parameterization，让宽度变化时更新尺度更可转移的规则 |
| IsoFLOP | 固定训练 FLOPs 扫其它选择 |
| least squares | 让残差平方和最小的拟合方法 |
| residual | 实测值减预测值 |
| train-optimal | 只按训练预算定义的最优 |
| deployment-optimal | 把部署/推理请求成本也加入的最优 |
| overtrained（本讲引号义） | tokens 超过训练 compute-optimal 比例，不等于过拟合 |

---

## 22. 自测题（80 题；第 16–70 题为手算/填表）

### 22.1 概念地基（1–15）

1. 用一句话定义 scaling law，并说出一个它不保证的事情。
2. 已测 1、2、4、8B tokens；预测 3B 与 80B 分别叫什么？哪一个更危险？
3. Empirical scaling law 与 theoretical generalization bound 的来源分别是什么？
4. 分别用人话解释 monotonic 与 asymptote。
5. 为什么预测 pretraining loss 不能直接等同于预测 downstream accuracy？
6. 为什么 nonparametric 不等于“没有参数”？
7. 什么是 intrinsic dimension？为什么不能从一个 exponent 严格反推出它？
8. 什么是 distribution shift？给一个训练/部署例子。
9. 什么是 lower envelope？它为什么不是宇宙级数学下界？
10. Critical batch 在优化中平衡哪两种效率？
11. muP 想解决哪种随 width 变化的问题？本讲为什么不讲其具体缩放规则？
12. 什么是 IsoFLOP？
13. Train-optimal 与 deployment-optimal 的目标函数差在哪里？
14. 本讲带引号的 “overtrained” 为什么不等于传统过拟合？
15. FLOP、FLOPs 总量、FLOP/s 各是什么？

### 22.2 四则运算与公式复算（16–70）

16. **【手算】**算 $`2^4`$、$`2^{-2}`$、$`16^{1/2}`$。
17. **【手算】**为什么 $`\log_{10}(1000)=3`$？计算器中 `4^(-0.5)` 是多少？
18. **【手算】**点 $`(2,7)`$ 与 $`(6,3)`$ 之间斜率是多少？
19. **【手算】**预测为 $`[2.0,2.5,3.0]`$，实测为 $`[2.1,2.3,3.2]`$。列 residual，并算 SSE。
20. **【手算】**$`L_\infty=1,A=8,\alpha=0.5,n=1`$ 时，总 loss 是多少？
21. **【手算】**沿用 Q20，$`n=4`$ 时总 loss 是多少？资源 4 倍改变的是哪一部分？
22. **【手算】**沿用 Q20，$`n=10`$ 时额外 loss 与总 loss 各约多少？
23. **【手算】**资源每翻倍，额外 loss 乘 0.8。由 $`2^{-\alpha}=0.8`$ 算 $`\alpha=-\ln0.8/\ln2`$，保留三位小数。
24. **【手算】**从 $`L-L_\infty=An^{-\alpha}`$ 逐步取 log，写出直线斜率与截距。
25. **【手算】**某 log-log 直线为 $`y=\ln4-0.25x`$。对应 $`A`$、$`\alpha`$ 各是多少？
26. **【手算】**若误差 $`e(n)=1/\sqrt n`$，要 $`e\le0.05`$，最少需要多少样本？
27. **【手算】**样本 $`[1,3,5,7]`$ 的样本均值是多少？
28. **【手算】**若每个 $`X_i`$ 的期望都是 10，写出 $`n=4`$ 时 $`\mathbb E[\bar X]`$ 的逐项计算。
29. **【手算】**独立样本 variance 都是 9，$`n=9`$ 时：先写含 covariance 交叉项的均值 variance 公式，再说明交叉项为何为 0，最后算结果。
30. **【手算】**$`\sigma=2`$ 时，先从无偏性写出“均值估计量的 MSE = variance”，再算 $`n=4,16`$ 的 MSE。
31. **【手算】**沿用 Q30，先用一句人话定义 standard error，再算两种情况下的 standard error。说明为何不是都按 $`1/n`$。
32. **【手算】**二维分箱 $`n=10,000`$：边长、每边箱数、总箱数、每箱样本数各是多少？
33. **【手算】**二维分箱 $`n=65,536=16^4`$：重复 Q32 四个量，并算方差型量级。
34. **【手算】**仅按课件 $`d`$ 维简化的**误差率** $`n^{-1/d}`$，$`n=10^6,d=3`$ 是多少？这里的 $`n^{-1/d}`$ 是边长还是误差？为什么不能把它当完整定理？
35. **【手算】**2000 tokens 按 60% 网页、25% 代码、15% 书籍混合，各是多少 tokens？
36. **【手算】**两曲线 $`L_A=1+4D^{-0.2}`$、$`L_B=1+6D^{-0.3}`$ 在何处相交？提示化为 $`D^{0.1}=1.5`$，算 $`1.5^{10}`$。
37. **【手算】**重复公式中 $`U_D=100,R_D^*=2,R_D=0`$，算 $`D'`$；总 epochs 是多少？
38. **【手算】**沿用 Q37，$`R_D=1`$ 时算 $`D'`$；总 epochs 是多少？用 $`e^{-0.5}\approx0.6065`$。
39. **【手算】**沿用 Q37，当 $`R_D\to\infty`$，$`D'`$ 上限是多少？
40. **【手算】**沿用 Q37，$`R_D=2`$ 时实际 processed tokens 与 effective tokens 各是多少？用 $`e^{-1}\approx0.3679`$。
41. **【手算】**词表 50,000、hidden width 1000，embedding 参数量是多少？若主体也是 50M，计入后总参数翻几倍？
42. **【手算】**8 个 100M experts、top-2：total expert parameters 与每 token active expert parameters 各多少？
43. **【手算】**$`B=32`$ examples/step、$`S=250`$，$`E`$ 是多少？
44. **【手算】**$`E_{\min}=1200`$ examples、$`S_{\min}=100`$ steps，$`B_{crit}`$ 是多少，单位是什么？
45. **【手算】**$`S_{min}=100,E_{min}=1000,B=5`$，算 $`S,E`$，并验证 $`E=SB`$。
46. **【手算】**同 Q45，但 $`B=10`$；验证双曲线两括号乘积为 1。
47. **【手算】**同 Q45，但 $`B=20`$；算出相对最少 steps/examples 的倍数。
48. **【手算】**定义 $`s=S/S_{min},e=E/E_{min},b=B/B_{crit}`$。从 $`b=e/s`$ 与 $`(s-1)(e-1)=1`$ 完整推出 $`s=1+1/b,e=1+b`$，再验证 $`E/S=B`$。
49. **【手算】**$`S_{min}=100,E_{min}=1000`$，观察到 $`S=250`$。由精确曲线求 $`E`$ 与 $`B`$。
50. **【手算】**gradient covariance 对角线是 $`[4,9,16]`$，trace 是多少？它用人话表示什么？
51. **【手算】**dense 训练 $`N=2`$M parameters、$`D=3`$M tokens，按 $`C=6ND`$ 算 FLOPs。
52. **【手算】**Q51 中 forward 与 backward 粗略各多少 FLOPs？
53. **【手算】**固定 $`C/6=ND=120`$，若 $`N=10`$，$`D`$ 是多少；若 $`N=20`$，$`D`$ 又是多少？
54. **【手算】**联合 law 中 $`\alpha=0.4,\beta=0.6`$。$`N_{opt}`$ 与 $`D_{opt}`$ 对 $`C`$ 的 exponent 各多少？
55. **【手算】**沿用 Q54，compute 增 $`16`$ 倍，$`N,D`$ 分别约增多少倍？计算 $`16^{0.6}`$ 与 $`16^{0.4}`$。
56. **【手算】**$`\alpha=\beta=0.5`$，compute 增 9 倍时，最优 $`N,D`$ 各增几倍？验证乘积增 9 倍。
57. **【手算】**取 $`E=1,A=B=1,\alpha=\beta=0.5`$，比较 $`(N,D)=(4,16)`$ 与 $`(16,4)`$ 的 loss。
58. **【手算】**固定 $`ND=36`$，计算 $`(N,D)=(4,9),(6,6),(9,4)`$ 的 $`1/\sqrt N+1/\sqrt D`$，哪个最低？
59. **【手算】**compute=20 时三个 recipe loss 为 2.7、2.5、2.9；lower envelope 是哪一个？若新 recipe 得 2.3，envelope 如何变？
60. **【手算】**模型主体 100M、embedding 50M、output 50M。分别算 total、non-embedding、排除 embedding 但保留 output 三种参数口径。
61. **【手算】**Kaplan exponent 下 compute 增 100 倍，$`N,D`$ 各约多少倍？
62. **【手算】**Chinchilla 0.5/0.5 下 compute 增 100 倍，$`N,D`$ 各多少倍？
63. **【手算】**若 $`N\propto C^{0.7},D\propto C^{0.3}`$，$`D/N`$ 对 $`C`$ 的 exponent 是多少？compute 增 100 倍时比率乘多少？
64. **【手算】**Method 3 小例中 $`E=1,A=B=1,\alpha=\beta=0.5`$，算 $`(N,D)=(9,16)`$ 的 loss。
65. **【手算】**四个 residual 为 $`0.1,-0.1,0.2,-0.2`$，SSE 是多少？残差平均虽为 0，为何仍不能说拟合完美？
66. **【手算】**候选 A：$`1000+10Q`$；B：$`1600+4Q`$。求 break-even 请求数，并判断 $`Q=200`$ 谁便宜。
67. **【手算】**粗略每生成 token 是 $`2N`$ FLOPs。$`N=7`$B、生成 1000 tokens，算 FLOPs；这里忽略了什么？
68. **【手算】**按 PDF 快照，Llama 3 70B、215 tokens/parameter，对应约多少训练 tokens？用 T 表示。
69. **【手算/核对】**PDF 与视频给 GPT-3 的 tokens/parameter 分别是多少？应该怎样记录？
70. **【填表】**把下列现象分别归入“x 轴定义、recipe 公平、数据问题、拟合诊断”：不计 output layer；小模型 warmup 占满全程；validation 泄漏；残差随规模持续为正。

### 22.3 综合判断（71–80）

71. 什么是 backtest？为什么应故意让验证点位于拟合区之外？
72. 为什么高 $`R^2`$ 不能单独证明 scaling law 可外推？
73. 给出两个 $`C\approx6ND`$ 会明显失真的场景。
74. 一个模型训练成本更低但每请求贵，另一个相反。最少还需要知道什么量才能做 deployment-optimal 决策？
75. 列出至少四个可能造成 Kaplan/Chinchilla exponent 不同的口径或 recipe 细节。
76. 想比较 5 个固定 compute budgets 下的最佳 $`N/D`$，三种 Chinchilla 方法中哪种最直接？为什么？
77. 语言模型 batch 用 tokens/step，而图表用 examples/step。公式 $`E=SB`$ 还能用吗？需要怎样改单位？
78. 本讲重复数据公式能否表示“重复太多后 validation loss 反而变坏”？为什么？
79. 写出一次 scaling 实验从 x 轴定义到大 run 的至少六步流程。
80. 用五句话串起本讲：幂律来源、数据/模型联合、critical batch、Kaplan/Chinchilla、deployment。

---

## 23. 自测答案（1–80）

### 23.1 答案 1–15

1. **答：**Scaling law 是在特定模型、数据、训练 recipe 与规模范围内，资源和表现之间观察并拟合出的简单规律。它不保证跨出该范围仍准，也不保证 pretraining loss 能精确预测 downstream accuracy。
2. **答：**3B 位于 2B 与 4B 之间，是 interpolation；80B 超出最大 8B，是 extrapolation。80B 更危险，因为可能跨入没有测过的新 regime。
3. **答：**Empirical law 来自实验点拟合；generalization bound 来自明确假设下的数学证明。前者追求实际预测，后者追求最坏情况保证。
4. **答：**Monotonic 表示输入增加时输出只朝同一方向走，例如 loss 只下降；asymptote 是曲线越来越接近的极限线，例如 $`L_\infty`$。
5. **答：**Loss 是连续训练目标；accuracy 常由离散答对/答错形成，还受 prompt、任务分布与阈值影响。因此两者排序与曲线形状可能不同。
6. **答：**Nonparametric 方法仍可有大量局部参数，只是有效自由度不是预先固定的有限数，会随数据增加。
7. **答：**Intrinsic dimension 是描述数据真正变化所需的自由方向数。把 exponent 连接到它需要 smoothness、估计器与数据流形等假设，而且 dimension 本身难可靠测量，所以不能严格反推。
8. **答：**Distribution shift 是训练分布与目标分布不一致。例如训练 90% 新闻、10% 代码，部署却 80% 问题是代码。
9. **答：**Lower envelope 是每个资源水平上已试候选的最低 loss。新 recipe、新数据或新架构仍可更低，所以不是数学上的不可突破下界。
10. **答：**它平衡少 training steps 的时间效率与少 processed examples 的样本/计算效率。
11. **答：**muP 想减少普通参数化下最佳 LR 等超参数随 width 漂移。本讲只预告，具体 tensor 缩放需完整 parameterization 背景，留到 Lecture 11。
12. **答：**IsoFLOP 是固定训练 FLOPs，在同一预算下扫描 $`N,D`$ 或其它自由度，寻找最低 loss 的实验设计。
13. **答：**Train-optimal 只最小化一次训练成本下的 loss；deployment-optimal 还加上所有未来请求的 prefill/decode 推理成本。
14. **答：**这里的 “overtrained” 只表示 tokens 超过 training-compute-optimal 配比；validation loss 仍可继续下降。传统过拟合则指训练表现改善、泛化表现恶化。
15. **答：**FLOP 是一次浮点操作；FLOPs 在本文语境常指完成任务所需操作总数；FLOP/s 才是每秒吞吐。

### 23.2 答案 16–40

16. **答：**

    $`2^4=2\times2\times2\times2=16,`$

    $`2^{-2}=\frac1{2^2}=\frac14=0.25,`$

    $`16^{1/2}=\sqrt{16}=4.`$

17. **答：**因为 $`10^3=1000`$，所以 $`\log_{10}(1000)=3`$。$`4^{-0.5}=1/\sqrt4=1/2=0.5`$。
18. **答：**

    $`m=\frac{3-7}{6-2}=\frac{-4}{4}=-1.`$

19. **答：**Residual = 实测 − 预测：

    $`[2.1-2.0,\ 2.3-2.5,\ 3.2-3.0]=[0.1,-0.2,0.2].`$

    $`\text{SSE}=0.1^2+(-0.2)^2+0.2^2=0.01+0.04+0.04=0.09.`$

20. **答：**

    $`L(1)=1+8\times1^{-0.5}=1+8=9.`$

21. **答：**

    $`L(4)=1+\frac8{\sqrt4}=1+\frac82=5.`$

    资源从 1 增到 4，只把额外 loss 从 8 减到 4；不可约的 1 没变，所以总 loss 从 9 到 5，不是到 4.5。
22. **答：**

    $`\sqrt{10}\approx3.162, \qquad \Delta L=8/3.162\approx2.530.`$

    总 loss $`\approx1+2.530=3.530`$。
23. **答：**

    $`\alpha=-\frac{\ln0.8}{\ln2} =-\frac{-0.22314}{0.69315} \approx0.32193\approx0.322.`$

24. **答：**

    $`L-L_\infty=An^{-\alpha}`$

    $`\log(L-L_\infty)=\log A+\log(n^{-\alpha}) =\log A-\alpha\log n.`$

    横轴 $`x=\log n`$、纵轴 $`y=\log(L-L_\infty)`$，斜率 $`-\alpha`$，截距 $`\log A`$。
25. **答：**与 $`y=\log A-\alpha x`$ 对照：截距是 $`\ln4`$，所以 $`A=4`$；斜率是 $`-0.25`$，所以 $`\alpha=0.25`$。
26. **答：**

    $`\frac1{\sqrt n}\le0.05=\frac1{20} \Rightarrow \sqrt n\ge20 \Rightarrow n\ge400.`$

27. **答：**

    $`\bar X=\frac{1+3+5+7}{4}=\frac{16}{4}=4.`$

28. **答：**

    $`\mathbb E[\bar X] =\frac{\mathbb E[X_1]+\cdots+\mathbb E[X_4]}4 =\frac{10+10+10+10}{4}=10.`$

    这里用了“期望可以逐项相加”。这一步本身不要求 $`X_i`$ 独立；独立性会在下一题消掉 covariance 交叉项时用到。

29. **答：**

    先写完整公式。对 $`\bar X=(X_1+\cdots+X_9)/9`$：

    $`\mathrm{Var}(\bar X) =\frac1{9^2}\left[ \sum_{i=1}^{9}\mathrm{Var}(X_i) +2\sum_{i<j}\mathrm{Cov}(X_i,X_j) \right].`$

    独立表示知道一个样本的取值，不会改变另一个样本的概率分布。因此

    $`\mathbb E[X_iX_j]=\mathbb E[X_i]\mathbb E[X_j] \Rightarrow \mathrm{Cov}(X_i,X_j)=0.`$

    于是交叉项全为 0：

    $`\mathrm{Var}(\bar X) =\frac1{81}(9\times9) =\frac{81}{81}=1.`$

30. **答：**因为样本均值无偏，即 $`\mathbb E[\bar X]=\mu`$：

    $`\mathrm{MSE}(\bar X) =\mathbb E[(\bar X-\mu)^2] =\mathbb E[(\bar X-\mathbb E[\bar X])^2] =\mathrm{Var}(\bar X) =\frac{\sigma^2}{n}.`$

    $`\sigma=2\Rightarrow\sigma^2=4`$，所以：

    $`n=4:\ 4/4=1; \qquad n=16:\ 4/16=0.25.`$

31. **答：**Standard error（标准误）是“如果反复抽样，估计量本身会晃动多少”的标准差。对样本均值：

    $`\mathrm{SE}(\bar X) =\sqrt{\mathrm{Var}(\bar X)} =\sqrt{\frac{\sigma^2}{n}} =\frac{\sigma}{\sqrt n}.`$

    代入 $`\sigma=2`$：

    $`n=4:\ 2/\sqrt4=2/2=1; \qquad n=16:\ 2/\sqrt{16}=2/4=0.5.`$

    Variance/MSE 按 $`1/n`$；standard error 按 $`1/\sqrt n`$，所以数据 4 倍时后者只减半。
32. **答：**

    $`n^{1/4}=10, \qquad h=n^{-1/4}=0.1.`$

    $`h`$ 是二维平面上每个正方形箱子的**边长**。每边 $`1/h=10`$ 箱，总箱 $`10^2=100`$，每箱平均 $`10,000/100=100`$ 样本。若只看每箱平均带来的 variance 型误差，其量级是 $`1/100=0.01=n^{-1/2}`$；不能把这个数误叫边长。
33. **答：**因为 $`65,536=16^4`$：

    $`h=1/16,`$

    这里 $`h`$ 是二维箱边长。每边 16 箱，总箱 $`16^2=256`$，每箱 $`65,536/256=256`$ 样本，方差型误差量级 $`1/256=0.00390625=n^{-1/2}`$。
34. **答：**

    $`n^{-1/d}=(10^6)^{-1/3}=10^{-2}=0.01.`$

    课件这里的 $`n^{-1/d}`$ 指简化的**误差率**，不是箱子的边长。代 $`d=2`$ 会得到 $`n^{-1/2}`$，正好与本讲二维例的方差型误差一致。不能当完整定理，因为从“有多少箱、每箱多少样本”本身推不出总误差；真正 rate 还依赖函数 smoothness、bias、loss、噪声与估计器。
35. **答：**

    $`\text{网页}=2000\times0.60=1200,`$

    $`\text{代码}=2000\times0.25=500,`$

    $`\text{书籍}=2000\times0.15=300.`$

    检查：$`1200+500+300=2000`$。
36. **答：**相交时：

    $`4D^{-0.2}=6D^{-0.3} \Rightarrow D^{0.1}=6/4=1.5.`$

    两边取 10 次方：

    $`D=1.5^{10}\approx57.665.`$

    所以约在 $`D=58`$ 的归一化单位附近交叉。
37. **答：**

    $`D'=100+200(1-e^0)=100+200(0)=100.`$

    $`R_D=0`$ 表示没有额外重复，总计 1 epoch。
38. **答：**

    $`D'=100+200(1-0.6065) =100+200(0.3935) =178.7.`$

    $`R_D=1`$ 是额外重复 1 次，总计 2 epochs。
39. **答：**$`e^{-R_D/2}\to0`$：

    $`D'\to100+200(1-0)=300.`$

40. **答：**$`R_D=2`$ 表示总计 3 epochs，所以实际 processed tokens：

    $`100(2+1)=300.`$

    Effective tokens：

    $`D'=100+200(1-e^{-1}) =100+200(1-0.3679) =226.42.`$

### 23.3 答案 41–70

41. **答：**

    $`50,000\times1000=50,000,000=50\text{M}.`$

    主体也是 50M 时，总参数 $`50+50=100`$M，是只计主体的 $`100/50=2`$ 倍。
42. **答：**Total：

    $`8\times100\text{M}=800\text{M}.`$

    每 token active：

    $`2\times100\text{M}=200\text{M}.`$

    还未计 router/shared layers。
43. **答：**

    $`E=SB=250\times32=8000\text{ examples}.`$

44. **答：**

    $`B_{crit}=\frac{E_{min}}{S_{min}}=\frac{1200}{100}=12.`$

    单位是 examples/step。
45. **答：**$`B_{crit}=1000/100=10`$，所以 $`b=5/10=0.5`$：

    $`S=100(1+1/0.5)=100(3)=300,`$

    $`E=1000(1+0.5)=1500.`$

    检查：$`SB=300\times5=1500=E`$。
46. **答：**$`b=10/10=1`$：

    $`S=100(1+1)=200, \qquad E=1000(1+1)=2000.`$

    双曲线：

    $`(200/100-1)(2000/1000-1)=1\times1=1.`$

47. **答：**$`b=20/10=2`$：

    $`S=100(1+1/2)=150=1.5S_{min},`$

    $`E=1000(1+2)=3000=3E_{min}.`$

48. **答：**

    先把三个无单位比值写出来：

    $`s=\frac S{S_{min}}, \qquad e=\frac E{E_{min}}, \qquad b=\frac B{B_{crit}}.`$

    因为 $`B=E/S`$ 且 $`B_{crit}=E_{min}/S_{min}`$：

    $`b =\frac{E/S}{E_{min}/S_{min}} =\frac{E/E_{min}}{S/S_{min}} =\frac es,`$

    所以 $`e=bs`$。把它放进双曲线：

    $`(s-1)(e-1)=1 \Rightarrow(s-1)(bs-1)=1.`$

    展开左边：

    $`bs^2-bs-s+1=1 \Rightarrow bs^2-(b+1)s=0.`$

    提出 $`s`$：

    $`s[bs-(b+1)]=0.`$

    $`s=S/S_{min}>0`$，不能取 0，所以：

    $`bs=b+1 \Rightarrow s=\frac{b+1}{b}=1+\frac1b.`$

    再用 $`e=bs`$：

    $`e=b\left(1+\frac1b\right)=b+1.`$

    最后检查实际 batch：

    $`\frac ES =\frac{E_{min}(1+b)}{S_{min}(1+1/b)}.`$

    因为 $`1+1/b=(b+1)/b`$：

    $`\frac ES =\frac{E_{min}}{S_{min}}\frac{1+b}{(1+b)/b} =B_{crit}b =B_{crit}\frac B{B_{crit}} =B.`$

49. **答：**

    $`S/S_{min}-1=250/100-1=1.5.`$

    因双曲线乘积为 1：

    $`E/E_{min}-1=1/1.5=2/3.`$

    $`E=1000(1+2/3)=1666.67.`$

    $`B=E/S=1666.67/250\approx6.667\text{ examples/step}.`$

50. **答：**

    $`\mathrm{trace}=4+9+16=29.`$

    它是三个 gradient 坐标 variance 的总和，粗略表示总波动规模；不包含 off-diagonal covariance 的符号信息。
51. **答：**

    $`C=6ND=6(2\times10^6)(3\times10^6) =36\times10^{12} =3.6\times10^{13}\text{ FLOPs}.`$

52. **答：**

    Forward 是 tokens 从输入经过模型各层、产生预测并算 loss；这一路会产生 activations。Backward 从 loss 反向传播：一部分工作算 activation gradients，把错误信号传回前层；另一部分算 weight gradients，供 optimizer 更新参数。

    $`C_{forward}\approx2ND =2(2\times10^6)(3\times10^6) =1.2\times10^{13}.`$

    $`C_{backward}\approx4ND=2.4\times10^{13}.`$

    合计 $`3.6\times10^{13}`$。
53. **答：**固定 $`ND=120`$：

    $`N=10\Rightarrow D=120/10=12;`$

    $`N=20\Rightarrow D=120/20=6.`$

    模型翻倍时，同 compute 下 data 必须减半。
54. **答：**

    $`N\text{ exponent}=\frac\beta{\alpha+\beta} =\frac{0.6}{1.0}=0.6.`$

    $`D\text{ exponent}=\frac\alpha{\alpha+\beta} =\frac{0.4}{1.0}=0.4.`$

    注意方向交叉：model exponent 用 data-loss exponent $`\beta`$。
55. **答：**

    $`N\text{ 倍数}=16^{0.6}\approx5.278,`$

    $`D\text{ 倍数}=16^{0.4}\approx3.031.`$

    检查：$`5.278\times3.031\approx16.0`$，与 compute 倍数一致。
56. **答：**两者 exponent 都是 $`0.5`$：

    $`9^{0.5}=\sqrt9=3.`$

    所以 $`N,D`$ 各增 3 倍，乘积增 $`3\times3=9`$ 倍。
57. **答：**

    $`L(4,16)=1+1/\sqrt4+1/\sqrt{16} =1+1/2+1/4=1.75.`$

    $`L(16,4)=1+1/4+1/2=1.75.`$

    因为这里 $`A=B,\alpha=\beta`$，交换 $`N,D`$ 不变。
58. **答：**

    $`(4,9):\ 1/2+1/3=0.8333;`$

    $`(6,6):\ 2/\sqrt6\approx2/2.449=0.8165;`$

    $`(9,4):\ 1/3+1/2=0.8333.`$

    最低是 $`(6,6)`$。
59. **答：**原 envelope 是三者最小值 2.5，对应 B。加入 2.3 后，新 envelope 变为 2.3；这证明 envelope 只是候选集合相关。
60. **答：**

    - total：$`100+50+50=200`$M。
    - non-embedding 若按“主体、不计 embedding 与 output”口径：100M。
    - 只排 embedding、保留 output：$`100+50=150`$M。

    名称可能被论文用得不一致，所以必须把包含项写出来。
61. **答：**

    $`N\text{ 倍数}=100^{0.73}=10^{1.46}\approx28.84,`$

    $`D\text{ 倍数}=100^{0.27}=10^{0.54}\approx3.47.`$

62. **答：**

    $`100^{0.5}=\sqrt{100}=10.`$

    所以 $`N,D`$ 都增 10 倍。
63. **答：**

    $`D/N\propto C^{0.3-0.7}=C^{-0.4}.`$

    compute 增 100 倍：

    $`100^{-0.4}=10^{-0.8}\approx0.1585.`$

    即 tokens/parameter 变为原来的约 15.85%。
64. **答：**

    $`L=1+1/\sqrt9+1/\sqrt{16} =1+1/3+1/4 =1+7/12 \approx1.5833.`$

65. **答：**

    $`\text{SSE}=0.1^2+(-0.1)^2+0.2^2+(-0.2)^2 =0.01+0.01+0.04+0.04=0.10.`$

    平均 residual 为 0 只说明正负抵消；每个点仍有误差，且 residual 还可能有随规模变化的结构。
66. **答：**令成本相等：

    $`1000+10Q=1600+4Q \Rightarrow6Q=600 \Rightarrow Q=100.`$

    $`Q=200`$：

    $`C_A=1000+2000=3000, \qquad C_B=1600+800=2400.`$

    B 便宜 600。
67. **答：**

    $`2N\times T=2(7\times10^9)(1000) =14\times10^{12}=1.4\times10^{13}\text{ FLOPs}.`$

    它忽略 prompt prefill、attention/context length、KV cache、batching、MoE、hardware utilization 与数据移动。
68. **答：**

    $`D=70\times10^9\times215 =15,050\times10^9 =15.05\times10^{12} =15.05\text{T tokens}.`$

69. **答：**PDF p54 写 2；视频 75:18 口述 3。正确做法是同时记录并标“课程材料内部冲突”，不能无说明地挑一个。
70. **答：**

    | 现象 | 类别 |
    |---|---|
    | 不计 output layer | x 轴定义 |
    | 小模型 warmup 占满全程 | recipe 公平 |
    | validation 泄漏 | 数据问题 |
    | residual 随规模持续为正 | 拟合诊断 |

### 23.4 答案 71–80

71. **答：**Backtest 是只用小规模点拟合，再预测一个当时不让拟合器看到、但我们已有实测的大规模点。验证点在拟合区外，才能真正检查 extrapolation，而非只检查 interpolation。
72. **答：**窄范围内许多不同曲线都能给高 $`R^2`$；若 residual 有系统形状，或外部点偏离，外推仍会失败。还要检查 residual、参数稳定性、不同拟合窗口与 held-out scale。
73. **答：**例如：（1）长序列 attention 的 $`O(s^2)`$ 项主导；（2）MoE 每 token 只激活部分 total parameters。其它例子包括 activation checkpoint 重算、embedding/output 占比很高和通信主导 wall-clock。
74. **答：**至少要知道预计请求数，以及每请求 prompt/generated tokens 或实际推理成本。否则无法判断更高训练成本何时被更低 serving 成本摊回。
75. **答：**任选四个：embedding/output parameter count；last-layer FLOPs；warmup 占训练比例；batch size 是否随规模调；LR/optimizer tuning；compute range；小模型是否收敛；数据/tokenizer 口径。
76. **答：**Method 2 IsoFLOP 最直接，因为它在每个固定 compute budget 内直接扫 $`N/D`$ 并看碗底，不必先从不同训练曲线估 lower envelope，也不必立即联合拟合五个参数。
77. **答：**能用，但单位必须统一。若 $`B`$ 是 tokens/step，$`E`$ 就必须是 processed tokens，不能仍写 examples；若一个 example 长度不同，还要先把它们换成 token 数。
78. **答：**不能。这个 exponential effective-data 公式随重复单调增加并最终 plateau，不会让有效数据减少。原论文拟合时也排除了“过多 epoch/参数导致表现变坏”的样本；真实训练仍可能因过拟合或超参不合适而变坏。
79. **答：**一种合格流程：（1）定义 $`N,D,C,loss`$ 口径；（2）固定可比 recipe；（3）选多个 compute budgets；（4）每个预算扫足够宽的候选网格；（5）拟合并看 residual/不确定度；（6）做 held-out-scale backtest；（7）保留验证预算；（8）执行大 run 并监控早期偏差。
80. **答：**（1）许多学习问题在一定范围出现 power law，均值与分箱例子说明不同 exponent 可来自不同统计难度。（2）模型与数据的误差项可联合写成 $`E+A/N^\alpha+B/D^\beta`$，固定 $`6ND`$ 后能推最优扩张指数。（3）Critical batch 用精确双曲线交换 steps 与 examples，在 $`B_{crit}`$ 处两者各约极限 2 倍。（4）Kaplan 与 Chinchilla 给出不同 compute 分配，差异受 parameter count、warmup、batch、范围与拟合方法影响。（5）训练 compute-optimal 不等于部署最优，请求多时，更多 tokens 训练的小模型可能总成本更低。

---

## 24. 视频时间导航（全部命中人工字幕 cue）

下面是**导航**，不是第二份正文；第一次学习仍按 §2–§18。所有秒点在本文只出现一次。

### 24.1 开场、动机与历史

- [00:06](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=6s)：正式开场。
- [01:01](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=61s)：课程前后讲次安排。
- [01:56](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=116s)：直接跑大实验为什么“很可怕”。
- [02:52](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=172s)：scaling law 也可被看作一种研究范式。
- [03:50](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=230s)：依赖可重复 regularity 的基本想法。
- [04:47](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=287s)：历史回顾的边界。
- [05:42](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=342s)：训练数据增大时会怎样。
- [06:37](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=397s)：经典学习理论脉络。
- [07:34](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=454s)：早期 NLP learning-curve 工作。
- [08:31](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=511s)：早期函数形式并非总能拟合。
- [09:27](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=567s)：从小数据预测大数据。
- [10:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=625s)：学生问规律只是经验还是有来源。
- [11:22](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=682s)：进入现代 neural scaling。
- [12:18](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=738s)：常见横轴是 compute。
- [13:16](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=796s)：曲线两端未必都处于同一 regime。

### 24.2 数据幂律、均值与 nonparametric

- [14:12](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=852s)：next-token prediction 数据曲线。
- [15:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=909s)：Kaplan 的多项 ablation。
- [16:06](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=966s)：均值估计例开始。
- [17:02](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1022s)：parametric estimation 的 polynomial rate。
- [17:58](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1078s)：神经 scaling 的收敛更慢。
- [18:53](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1133s)：每个二维箱约 $`\sqrt n`$ 样本。
- [19:49](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1189s)：理论解释工作与其局限。
- [20:45](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1245s)：课堂问题继续澄清 model/data 相对大小。

### 24.3 数据 composition、mixture 与 repetition

- [21:42](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1302s)：单纯 data-count law 的局限。
- [22:38](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1358s)：把数据曲线看成经验 generalization 规律。
- [23:34](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1414s)：pretraining data optimization。
- [24:30](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1470s)：mixture 经验并不总简单。
- [25:29](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1529s)：compute 增长与数据有限性。
- [26:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1585s)：无限 compute 假设会怎样。
- [27:22](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1642s)：数据处理研究的 scaling 语境。
- [28:17](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1697s)：大规模还从哪里获得数据。
- [29:16](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1756s)：线性轴与 log-log 轴区别。

### 24.4 Architecture、optimizer、参数口径与 MoE

- [30:19](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1819s)：model engineering 进入主线。
- [31:15](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1875s)：架构与 optimizer 都可做 scaling 分析。
- [32:11](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1931s)：不同大小 LSTM 的实验设计。
- [33:08](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1988s)：架构 scaling 证据仍有空白。
- [34:03](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2043s)：某些架构改动的总体趋势。
- [35:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2100s)：用 scaling law 比单点更精确。
- [35:55](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2155s)：深宽的量化判断。
- [36:53](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2213s)：层数与参数预算。
- [37:49](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2269s)：参数如何计数会让 law 看起来好或坏。
- [38:46](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2326s)：只有条件设对才有 predictability。
- [39:43](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2383s)：MoE total parameters 增大。
- [40:39](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2439s)：active/total 参数含义总结。

### 24.5 Critical batch 与 muP

- [41:34](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2494s)：batch size 与 learning rate 入口。
- [42:29](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2549s)：batch 开始回报递减。
- [43:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2605s)：不同 batch 目标之间的分歧。
- [44:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2660s)：local quadratic approximation 背景。
- [45:15](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2715s)：需要的 steps 怎样变化。
- [46:15](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2775s)：临界点平衡曲线两侧。
- [47:11](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2831s)：平衡 convergence rate 与 compute。
- [48:06](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2886s)：critical batch 随训练变化的直觉。
- [49:01](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2941s)：模型规模与最佳 LR。
- [50:03](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3003s)：scale-aware 方法的实践反馈。

### 24.6 Predictability 边界与联合 law

- [51:01](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3061s)：进入 scaling 不完全可靠的部分。
- [51:56](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3116s)：scaling law 通常预测何种对象。
- [52:51](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3171s)：理想情况下应能精确预测大点。
- [53:47](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3227s)：课堂问要做多少 runs。
- [54:43](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3283s)：课堂问能否直接拟合 downstream metrics。
- [55:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3340s)：跨设置转移证据的边界。
- [56:35](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3395s)：课堂问答结束。
- [57:30](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3450s)：更多 data 还是更大 model。
- [58:30](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3510s)：Kaplan 与 Rosenfeld 的函数形式。
- [59:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3565s)：联合 scaling laws 的广泛使用。
- [60:25](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3625s)：Kaplan 资源处方。
- [61:24](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3684s)：2022 DeepMind/Chinchilla 转折。

### 24.7 三种方法、争议与部署

- [62:21](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3741s)：开始走读 Chinchilla。
- [63:17](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3797s)：Method 1 lower envelope。
- [64:13](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3853s)：从 envelope 得到一个 scaling 答案。
- [65:08](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3908s)：不同 run 的 terminal loss 形成曲线。
- [66:03](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3963s)：Method 3 把 model/data 连接到 loss。
- [67:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4020s)：Kaplan 与 Chinchilla 方法并非一疯一智。
- [67:55](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4075s)：作者与后续复现澄清。
- [68:55](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4135s)：第一项差异是 parameter counting。
- [69:52](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4192s)：小模型在 warmup 后仍未正常收敛。
- [70:47](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4247s)：错误 recipe 会给坏 scaling law。
- [71:44](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4304s)：Pearce/Song 提供另一种解释。
- [72:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4360s)：Method 3 差异的含义不小。
- [73:37](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4417s)：重拟合结果指向原拟合问题。
- [74:35](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4475s)：生产模型不只优化 training compute。
- [75:31](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4531s)：大规模 serving 改变资源选择。
- [76:27](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4587s)：IsoFLOP 扫描自由度。
- [77:22](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4642s)：全讲最后回顾。

---

## 25. PDF 1–57 覆盖索引、来源与核验边界

### 25.1 每页恰好进入一个区间

| PDF 页 | 视觉内容 | 对应正文 |
|---:|---|---|
| 1–4 | 标题、讲纲、scaling 的使用场景 | §0、§3 |
| 5–6 | 学习理论背景、upper bounds | §3.3 |
| 7–10 | 1993、Banko/Brill、Kolachina、Hestness 历史 | §4 |
| 11–12 | Hestness 扩展、现代 neural scaling 讲纲 | §3–§5 |
| 13–15 | 多资源 power laws、Kaplan data law | §5 |
| 16–18 | mean estimation、log rate、经验指数 | §6 |
| 19–20 | 二维分箱与 intrinsic-dimension 解释 | §7 |
| 21–23 | data composition、distribution、mixture | §8 |
| 24–28 | repetition、compute-unbounded、filtering/data limitation | §9 |
| 29–32 | 线性/log 轴、architecture、Transformer/LSTM、optimizer | §10.1 |
| 33–36 | depth/width/aspect、参数价值、MoE | §10.2–§10.4 |
| 37–39 | critical batch、精确曲线、loss-dependent batch | §11 |
| 40–41 | muP 与 downstream caution | §12、§10.5 |
| 42–44 | model-data joint laws 与外推 | §13 |
| 45–46 | Kaplan exponent、Chinchilla 三方法表 | §14–§15 |
| 47–49 | lower envelope、IsoFLOP、joint fit | §15 |
| 50–53 | Kaplan/Chinchilla discrepancy、Method 3 replication | §16 |
| 54–55 | tokens/parameter、deployment 与 IsoFLOP 案例 | §17–§18 |
| 56–57 | takeaways 与引用 | §19、§25 |

以上区间连续覆盖 1–57，无 gap、无 overlap；“覆盖索引”表示每页已视觉检查并映射到正文，不表示逐字翻译。

### 25.2 课程主来源

- [CS336 Lecture 9 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)：课堂图、公式、表格的主依据。
- [Stanford Online Lecture 9 视频](https://www.youtube.com/watch?v=Q15rhEWZPQ4)：口头解释与课堂问答；本文使用完整人工 `en-US` 字幕轨。

### 25.3 补充一手来源

- [Banko & Brill 2001](https://aclanthology.org/P01-1005/)；[Kolachina et al. 2012](https://aclanthology.org/P12-1003/)：早期 NLP data scaling / learning-curve prediction。
- [Hestness et al. 2017](https://arxiv.org/abs/1712.00409)：跨任务 empirical power laws。
- [Rosenfeld et al. 2019](https://arxiv.org/abs/1909.12673)：model/data joint generalization form。
- [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)：语言模型 model/data/compute scaling。
- [McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)：gradient noise scale 与 critical batch。
- [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)：Chinchilla 三种 compute-optimal 方法。
- [Yang et al. 2022](https://arxiv.org/abs/2203.03466)：muP / muTransfer。
- [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)：重复数据的 effective-data law；原文定义 $`R_D`$ 为首遍后的重复次数。
- [Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102)：从图中重建数据复核 Chinchilla Method 3。
- [Porian et al. 2024](https://arxiv.org/abs/2406.19146)：复现并解释 Kaplan/Chinchilla 差异。

### 25.4 边界声明

- 历史和论文结论按其原实验时间与范围陈述；不是 2026 后所有模型的永恒事实。
- 本地没有执行任何真实 scaling training grid；数值验证只覆盖本文教学公式和小例。
- 外部论文只用于核对公式定义、实验范围与后续复核，不把补充材料冒充课堂逐字原话。
- PDF p54 与视频 75:18 的 GPT-3 比率冲突已显式保留。

---

## 26. 学完后的能力清单

读完并做完题后，你应该能：

- 从 $`L_\infty+An^{-\alpha}`$ 推出资源翻倍倍率与 log-log 斜率；
- 从独立样本方差相加推到 $`\mathrm{MSE}=\sigma^2/n`$；
- 手算二维分箱的边长、箱数、每箱样本与 variance/bias trade-off；
- 区分 unique、processed、effective data，并正确把 $`R_D`$ 解释为额外重复次数；
- 用 $`E=SB`$ 和 critical-batch 双曲线复算 steps/examples 表；
- 区分 total/active/non-embedding parameters；
- 从 $`C\approx6ND`$ 与联合 loss law 推出两个 compute-optimal exponents；
- 手算 lower-envelope、IsoFLOP 碗底和 Method 3 小曲面；
- 准确说明 Kaplan、Chinchilla 与 2024 两类复核各自证明了什么、没有证明什么；
- 把 training cost 与 serving requests 合并做 break-even；
- 设计一个带公平 recipe、residual 检查、backtest 和保留预算的 scaling 实验。

视频 [77:41](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4661s) 结束本讲，并预告 inference 后会回到更高级 scaling 主题。
