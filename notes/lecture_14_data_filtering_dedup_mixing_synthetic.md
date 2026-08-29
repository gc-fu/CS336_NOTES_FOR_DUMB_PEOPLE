# Lecture 14：数据处理 II——抽取、过滤、去重、混合与合成数据

> CS336 Spring 2026 · Data II  
> 官方视频：[YouTube](https://www.youtube.com/watch?v=5sxHosTLPF8)  
> 官方 executable lecture：[`lecture_14.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)  
> 本地核验版本：464 个物理行、24,633 bytes、SHA256 `53E2B997DCE030AF6DAA519BE7F297AFD2CCF23E2152CE8DD776D85CD26E36B7`  
> 字幕：YouTube 人工英文 `en-US` 轨，1,448 个 cue，首段 00:05，末段 84:36

## 0. 怎么读、来源标签与目录

### 0.1 第一次读请按这个顺序

如果你只会四则运算，也可以从头读。建议顺序是：

1. 先读 §2 的总流水线；
2. 按 §3–§10 学“抽取和过滤”；
3. 按 §11–§15 学“去重”；
4. 按 §16–§18 学“数据混合”；
5. 按 §19–§21 学“后训练与合成数据”；
6. 最后做 §26 自测，再用 §1 复习。

**首次阅读不要先背 §1。** 那是已经理解之后的五分钟复习卡。

### 0.2 本文标签

- 【课程】来自本讲官方代码讲义或视频。
- 【视频补充】讲者口头说了，但代码讲义没有完整写出。
- 【补充解释】为初学者补出的推导、小例子或实现边界。
- 【补充】来自论文、官方文档等一手资料。
- 【延伸】帮助建立全局地图，但不是本讲要求背诵的内容。

### 0.3 可点击目录

- [§1 五分钟复习卡](#1-五分钟复习卡首次阅读先跳过)
- [§2 数据流水线总地图](#2-全讲总地图原料不是训练数据)
- [§3 HTML、PDF、代码仓库怎样变成文本](#3-transformation原料怎样变成文本)
- [§4 抽取为什么有损](#4-抽取是有损压缩不是复制粘贴)
- [§5 怎样评价抽取器](#5-怎样知道抽取得好不好)
- [§6 过滤问题与分数](#6-filtering从目标样本到整个原料池)
- [§7 生成式与判别式过滤](#7-两类过滤器生成式与判别式)
- [§8 GPT-3 的 Pareto 随机保留](#8-gpt-3-式-pareto-随机保留完整推导)
- [§9 过滤案例](#9-语言数学代码毒性四类过滤案例)
- [§10 过滤阈值随训练规模变化](#10-为什么没有一个永远最好的过滤阈值)
- [§11–§15 去重](#11-deduplication先把问题拆成三根轴)
- [§16–§18 混合](#16-mixing多个来源到底各抽多少)
- [§19–§21 后训练与合成数据](#19-post-training数据从哪里来)
- [§22 端到端决策树](#22-端到端数据流水线与决策树)
- [§24 常见误区](#24-常见误区错误原因正确说法)
- [§26 自测](#26-自测题80题)
- [§27 答案](#27-自测答案)
- [§28 视频导航](#lecture14-video-nav)
- [§29–§31 覆盖与来源](#lecture14-source-coverage)

## 1. 五分钟复习卡（首次阅读先跳过）

1. **原始网页不等于文本。** HTML、PDF 和仓库都有结构；抽取是在保留有用内容与删除噪声之间做取舍（§3–§5）。
2. **过滤分数不是真理。** 它只是模型或规则对“像目标数据”的估计；必须查阈值、误报、漏报和群体偏差（§6–§10）。
3. `np.random.pareto(9) > 1-score` 的保留概率是 `(2-score)^(-9)`，不是 `score`（§8）。
4. **去重有三根轴：** 比较什么 item、怎样算 match、命中后做什么 action（§11）。
5. Jaccard 是交集大小除以并集大小。MinHash 的单次碰撞概率等于 Jaccard；多次碰撞比例只是估计（§13）。
6. LSH 候选概率是 `1-(1-s^r)^b`。增大 `r` 更严格；增大 `b` 更宽松（§14）。
7. 本文用 $`p_i`$ 表示来源的 **token share（token 份额）**，满足 $`\sum_i p_i=1`$；训练位置数是 $`p_iT`$，epoch 数是 $`p_iT/N_i`$。若实现按整条 sequence 抽来源，另用 $`q_i`$ 表示 sequence 抽样概率，二者只在等长期望或按 token 预算校正时相同（§16）。
8. UniMax 的 epoch cap 应写成 `p_i T <= C N_i`；源码漏了来源大小 `N_i`（§17）。
9. 小模型上拟合的最佳 mixture 不保证迁移到大模型；要查 evaluation overfit 与重复次数（§18）。
10. 合成数据不是“免费真相”。要审计环境、任务、teacher response、验证器、许可、隐私与污染（§19–§21）。

## 2. 全讲总地图：原料不是训练数据

【课程】讲者在 [00:50](https://www.youtube.com/watch?v=5sxHosTLPF8&t=50s) 给出本讲四步：transformation（转换/抽取）→ filtering（过滤）→ deduplication（去重）→ mixing（混合），最后再谈 post-training synthetic data（后训练合成数据）。

把它想成做饭：

```text
网页/PDF/仓库
    │  transformation：变成线性文本
    ▼
候选文本池 R
    │  filtering：留下更像目标 T 的 T'
    ▼
过滤后的多个来源
    │  dedup：删重复；decontamination：隔离测试题
    ▼
每个来源各自的干净集合
    │  mixing：决定每个来源被抽到多少次
    ▼
预训练/中训练序列

环境 + 任务/提示 + teacher response
    └──────────────────────► 后训练数据
```

### 2.1 前置知识与本讲最少词汇

- **raw data（原始数据）**：刚抓下来的 HTML、PDF 字节或仓库文件，还不能直接当训练句子。
- **document（文档）**：一次处理的基本内容单位，例如一个网页、一篇 PDF 或一个文件。
- **token（词元）**：tokenizer（分词器）把文本切成、并映射为整数 ID 的小单位；token 不一定等于汉字或英文单词。
- **pre-training（预训练）**：在大规模、通常较通用的文本上学习“下一个 token”。
- **mid-training（中训练）**：预训练与任务化后训练之间的继续训练阶段，常加入长上下文或特定领域数据；名字不是统一标准。
- **post-training（后训练）**：在基础模型之后，用任务/指令/偏好等数据改变行为。
- **SFT（Supervised Fine-Tuning，监督微调）**：给定输入和期望输出，让模型模仿这些输出。
- **epoch（轮）**：平均把数据集里每个样本用一次。`0.5 epoch` 是平均只用一半；`50 epochs` 是平均重复 50 次。
- **FLOP（floating-point operation，浮点运算一次）**：一次浮点加法或乘法。这里说“浪费 FLOPs”，就是把计算花在重复或不想要的内容上。
- **ablation（消融实验）**：只改一个组件，其他尽量不变，看性能差异是否来自它。

### 2.2 四步为什么不能随便换顺序

【补充解释】顺序不是绝对法律，但每一步的输入假设不同：

- 过滤器通常要读文本，所以 HTML/PDF 要先抽取；
- 去重需要稳定的 item 表示，通常在规范化与抽取之后做；
- mixture 权重依赖过滤/去重后的真实 token 数；若先定权重，后删掉一半数据，epoch 数会改变；
- decontamination（去污染）最好在所有来源合并视野下再检查，否则同一测试题可能从另一个来源漏进来。

## 3. Transformation：原料怎样变成文本

【课程】原始数据可能是 HTML、PDF 或代码目录，而不是纯文本（[01:23](https://www.youtube.com/watch?v=5sxHosTLPF8&t=83s)）。

### 3.1 HTML 是“结构说明”，不是页面正文

HTML（HyperText Markup Language，超文本标记语言）会把标题、链接、导航、广告、表格都编码进去。

```html
<nav>首页 | 广告</nav>
<h1>泡茶</h1>
<p>水温 90°C。</p>
<table><tr><td>绿茶</td><td>90°C</td></tr></table>
```

一个抽取器可能输出：

```text
泡茶
水温 90°C。
绿茶 | 90°C
```

它做了三件事：

1. 删除 boilerplate（样板噪声），例如导航和广告；
2. 保留主内容；
3. linearize（线性化）：把二维/树状布局排成从左到右的一串 token。

【视频补充】“什么算正文”没有唯一答案：菜单可能是噪声，也可能教模型理解网页结构（[02:20](https://www.youtube.com/watch?v=5sxHosTLPF8&t=140s)）。

### 3.2 WARC 到文本

WARC（Web ARChive）是保存网页抓取响应的归档格式。它可能含 HTTP（网页传输协议）头、HTML、状态码和抓取 metadata（描述数据的数据，如 URL、时间）。大致是：

```text
WARC 记录 → 取出 HTTP body → 解码字符 → 解析 HTML
          → 识别主内容 → 线性化 → 文本 + provenance metadata
```

provenance（来源链）回答：“这段文本从哪个 URL、哪次抓取、用哪个抽取器版本产生？”没有它，之后很难删除、修错或证明处理过程。

【课程】常见规则抽取器包括 Trafilatura、Resiliparse、jusText、lynx。规则很快，适合扫海量网页（[03:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=190s)）。

### 3.3 PDF 更麻烦

PDF（Portable Document Format）主要描述“字符画在哪里”，未必保存“这是标题、这是第二列表格”的语义。扫描 PDF 甚至只有图片。

- **OCR（Optical Character Recognition，光学字符识别）**：从图片识别文字。
- **VLM（Vision-Language Model，视觉语言模型）**：同时处理图像与文字的模型，可辅助恢复阅读顺序和结构。
- **Docling**：文档转换工具链；这里是课程列举的方案之一。
- **RolmOCR**：课程在 FinePDFs 流程中列举的视觉 OCR 模型。

【课程】FinePDFs 的课程快照：从 Common Crawl 找 PDF；被截断的大文件要重新抓；再用 RolmOCR/VLM 或 Docling 抽取并清理（[05:08](https://www.youtube.com/watch?v=5sxHosTLPF8&t=308s)）。这是具体项目流程，不代表所有 PDF 都这样处理。

### 3.4 代码仓库也不是“把所有文件拼起来”

仓库含源码、测试、生成文件、依赖锁文件、二进制、重复 fork 和版本历史。至少要决定：

- 哪些扩展名是文本；
- 是否保留 README、测试和 issue；
- 文件按什么顺序排列；
- fork 与 vendored dependency（复制进仓库的第三方依赖）怎样去重；
- commit 时间和 license metadata 是否保留。

只把单文件拼接会破坏跨文件语义。例：`main.py` 写 `from util import old_name`，而同一 commit 的 `util.py` 已把函数 rename（重命名）为 `new_name`；若抽取时混了不同 commit，import（导入）会断。另一个例子是类定义在 `a.py`、调用在 `b.py`，只保留调用文件会让类型与行为无从追踪。因此 repo extraction 要保存相对路径、commit、import/rename 关系，必要时把相关文件打包成同一训练单元。

## 4. 抽取是有损压缩，不是复制粘贴

【课程】HTML 的视觉/树状结构必须被压成 token 序列，所以 inherently lossy（必然有损）（[02:39](https://www.youtube.com/watch?v=5sxHosTLPF8&t=159s)）。

### 4.1 一个小表格怎样丢掉含义

原表：

| 商品 | 2025 价格 | 2026 价格 |
|---|---:|---:|
| A | 10 | 12 |
| B | 20 | 18 |

坏抽取：

```text
商品 2025 价格 2026 价格 A 10 12 B 20 18
```

现在模型还要猜 `12` 属于哪一列。更坏时列顺序可能变成 `A B 10 20 12 18`。

【补充解释】因此“抽到了所有字符”不代表“保留了所有关系”。字符召回率可以很高，表格语义却已经错。

### 4.2 图片与阅读顺序

如果段落写“见右图红线”，只保留段落会留下无法解释的指代。双栏论文若先读左栏第一行、再读右栏第一行，会把两篇句子交错。需要明确：

- 是否保留图片 caption（图注）；
- 是否运行 OCR/VLM；
- 多栏、脚注、页眉、公式按什么 reading order（阅读顺序）展开；
- 无法可靠抽取时是丢文档，还是保留低置信度标记。

### 4.3 规则与模型的取舍

| 方法 | 优点 | 风险 |
|---|---|---|
| 规则抽取 | 快、便宜、可解释 | 新模板会失败；复杂表格弱 |
| 模型抽取 | 能理解更复杂布局 | 慢、贵、会幻觉；版本漂移 |
| 混合 | 规则处理简单页，模型处理疑难页 | 路由器本身也会错 |

【课程】代码讲义中的 DCLM 表报告 CORE/EXTENDED 两组下游聚合分数；在这张表中都是**越高越好**。CORE 最高是 Trafilatura 的 24.5，EXTENDED 最高是 Resiliparse 的 13.4；没有一行同时赢两列。完整值是 Resiliparse `24.1/13.4`、Trafilatura `24.5/12.5`、WET `20.7/12.2`。它量的是 DCLM 该训练/评估设置下不同 extraction pipeline 对下游结果的影响，不是字符级 precision/recall，也不证明某工具永远最好（[03:57](https://www.youtube.com/watch?v=5sxHosTLPF8&t=237s)）。

## 5. 怎样知道抽取得好不好

### 5.1 先造一个人工 gold

gold（人工参考答案）是人审过的期望输出。假设页面应保留 10 个正文句子：

- 抽取器输出 8 个句子；
- 其中 6 个真是正文；
- 另 2 个是广告；
- 它漏了 4 个正文句子。

先固定标签契约。**actual positive（真实正类）**是人工 gold 判定应保留的正文块；**predicted positive（预测正类）**是抽取器实际输出的块。

| | predicted positive：抽出 | predicted negative：没抽出 |
|---|---:|---:|
| actual positive：gold 正文 | TP：正文且抽出 | FN：正文却漏掉 |
| actual negative：非正文 | FP：广告等却抽出 | TN：非正文且没抽 |

所以：

- TP（true positive）：真正文且抽出；
- FP（false positive）：非正文却抽出；
- FN（false negative）：真正文却漏掉；
- TN（true negative）：非正文且没抽。

定义：

```math
\text{precision}=\frac{\text{正确抽出的正文}}{\text{所有抽出的内容}},
\qquad
\text{recall}=\frac{\text{正确抽出的正文}}{\text{所有应抽出的正文}}.
```

逐步代数：

```math
\text{precision}=6/8=0.75=75\%,
```

```math
\text{recall}=6/10=0.60=60\%.
```

precision（精确率）低说明混入很多噪声；recall（召回率）低说明漏掉很多正文。两者不能互相替代。

### 5.2 下游 ablation

【补充解释】还要做 end-to-end（端到端）消融：同一抓取、同一过滤、同一模型和训练 token 数，只换抽取器。**validation loss（验证损失）**是在没有用来更新参数的验证数据上算的 loss；同一 tokenizer、数据、模型和口径下通常越低越好。若验证损失或任务指标稳定改善，才说明抽取差异真的影响训练。

反例：抽取器 A 的字符 recall 更高，只因为保留了大量页脚；下游模型反而更差。此时单个抽取指标误导了我们。

### 5.3 一张最小抽取审计卡

```text
source URL / crawl time:
extractor + version:
input type: HTML / born-digital PDF / scanned PDF / repo
gold sample construction:
text precision / recall:
table and reading-order error:
language/group slices:
downstream ablation:
known failures and deletion route:
```

## 6. Filtering：从目标样本到整个原料池

【课程】给一小份 target data `T`（你想要的样子）和一大池 raw data `R`，目标是在 `R` 中找到类似 `T` 的子集 `T'`（[07:04](https://www.youtube.com/watch?v=5sxHosTLPF8&t=424s)）。

### 6.1 三个集合不要混

- `T`：目标示例，例如人审的优质数学文本。
- `R`：原料池，例如 100 亿网页文档。
- `T'`：过滤器从 `R` 留下的文档；它应该“像 `T`”，但不是把 `T` 复制进去。

为什么要 generalize（泛化）？因为 `T` 本来就有，过滤的价值是找到新的、未在 `T` 里的好文档。

【课程】过滤器还必须快；它可能要跑过约 100T token 的候选池（[08:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=510s)）。`T` 在这里是 trillion（万亿），不要和 target 集合 `T` 混；本文数量用 `10^{12}` 明写。

### 6.2 从文档到决定的流水线

```text
文档 x → 规则/模型 → score(x) → threshold/随机规则 → keep 或 drop
```

- **score（分数）**：过滤器的数值输出，不等于客观真相。
- **threshold（阈值）**：超过/低于哪条线才保留。
- **calibration（校准）**：若分数为 0.8 的文档长期约 80% 真符合人工定义，才叫校准较好。
- **false positive（FP，误报）**：坏文档被留了。
- **false negative（FN，漏报）**：好文档被删了。

这里是第二份标签契约，不能偷用上一节而不声明：

| filtering evaluation | predicted positive：系统保留 | predicted negative：系统删除 |
|---|---|---|
| actual positive：按明确人工规则判为好文档 | TP | FN |
| actual negative：按同一规则判为坏文档 | FP | TN |

所以 filtering 的 precision 是“留下的文档中，按该人工规则真好的比例”；recall 是“全部人工判好文档中，被留下的比例”。“好”的规则必须先写出来，例如“正确、可读、符合数学领域”，不能把模型 score 当 actual label。

**判别器训练标签又是第三个口径。** 训练时常把 target 来源 `T` 标 1、raw 来源 `R` 标 0；这只是带噪的来源代理标签。`R` 里仍可能有好文档，`T` 也可能有错。分类器学到的是“像 T 还是像 R”，不等于学到客观质量真值。

### 6.3 一个阈值表

假设人审 100 篇，其中 40 篇好、60 篇坏：

| 阈值 | 留下的真好 | 留下的坏 | 漏掉的真好 | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 36 | 24 | 4 | `36/(36+24)=60%` | `36/40=90%` |
| 0.8 | 28 | 4 | 12 | `28/(28+4)=87.5%` | `28/40=70%` |

提高阈值后，precision 上升，但 recall 下降。选择哪一个取决于：训练预算、可用数据量、误删某类语言的代价、训练多久。

### 6.4 score 不等于“质量”

若正例全来自学术百科，分类器可能把“长句、引用多、标准英语”学成捷径。它可能给优秀口语、方言或低资源语言低分。这是 **selection bias（选择偏差）**：target 的收集方式改变了模型认为什么是好。

【补充解释】至少按语言、地区、文体、长度做 slice（分组切片）检查。例如总体 recall 90%，少数语言 recall 只有 35%，总体数字会遮住伤害。

## 7. 两类过滤器：生成式与判别式

### 7.1 生成式分数 `p_T(x)`

【课程】在目标数据 `T` 上训练一个便宜语言模型，问“文档 `x` 在这个模型下有多像目标”。KenLM 是常见 n-gram 工具；n-gram 只看最近若干 token，速度快（[09:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=570s)）。

`p_T(x)` 是目标模型赋给 `x` 的概率。长文概率是许多小数连乘，常改用每 token negative log-likelihood 或 perplexity（困惑度）；**不同 tokenizer、长度归一化和数据处理下的 perplexity 不可直接比较。**

小例：目标语料常见 `机器 学习`，很少见 `机器 香蕉`。模型可能给：

```math
p_T(\text{学习}\mid\text{机器})=0.4,
\qquad
p_T(\text{香蕉}\mid\text{机器})=0.001.
```

前者更像目标，但这不证明前者事实正确。

### 7.2 判别式分数 `p(T|x)`

【课程】把 target 当正例，把 raw 抽样当负例，训练分类器预测 `x` 属于 target 风格的概率（[09:48](https://www.youtube.com/watch?v=5sxHosTLPF8&t=588s)）。fastText 常用词袋 + 线性分类器，便宜、快。

注意：`p_T(x)` 与 `p(T|x)` 不是同一个东西。

- `p_T(x)`：目标语言模型觉得文本本身多常见；
- `p(T|x)`：分类器觉得标签是“目标”的概率。

### 7.3 rule-based 与 model-based

- **rule-based（基于规则）**：如“含 LaTeX 命令”“行太短则删”。易解释、快，但规则边界僵硬。
- **model-based（基于模型）**：可组合许多弱信号，但会继承训练标签偏差。

课程代码把 C4、Gopher、RefinedWeb、FineWeb、Dolma 概括为“刻意不做模型过滤”，把 GPT-3、LLaMA、DCLM 列为模型过滤。应把它理解为课程用于比较主要 pipeline 的简化标签；大型数据集常还含语言 ID、规则或别的模型组件，不能据此断言“完全没有模型”。

## 8. GPT-3 式 Pareto 随机保留：完整推导

【课程】源码是：

```python
def keep_document(score: float) -> bool:
    return np.random.pareto(9) > 1 - score
```

### 8.1 先认识随机变量

`np.random.pareto(9)` 按 NumPy 的定义抽的是 Pareto II/Lomax 随机变量 `X>=0`，shape 参数 `a=9`。其概率密度是：

```math
f(x)=\frac{a}{(1+x)^{a+1}},\qquad x\ge 0.
```

不用微积分也可直接使用 NumPy 文档给出的 survival function（超过阈值的概率）：

```math
P(X>u)=(1+u)^{-a}.
```

这里：

- `score=s`，假设 `0<=s<=1`；
- 比较阈值 `u=1-s`；
- `a=9`。

逐项代入：

```math
P(\text{keep}\mid s)
=P(X>1-s)
=[1+(1-s)]^{-9}
=(2-s)^{-9}.
```

**所以保留概率不是 `s`。** 这是把高分样本强烈上权、但仍让低分样本有极小机会进入的非线性随机规则。

### 8.2 四个分数手算

| `s` | `2-s` | 保留概率 `(2-s)^-9` | 约每多少篇留 1 篇 |
|---:|---:|---:|---:|
| 0 | 2 | `1/2^9=1/512=0.001953` | 512 |
| 0.5 | 1.5 | `1/1.5^9=1/38.443=0.02601` | 38.4 |
| 0.9 | 1.1 | `1/1.1^9=1/2.35795=0.42410` | 2.36 |
| 1 | 1 | `1` | 1 |

例如 10,000 篇分数都为 0.5，期望保留约：

```math
10{,}000\times0.02601\approx260\text{ 篇}.
```

“期望 260”不是每次恰好 260；随机抽样会波动。

### 8.3 为什么不用硬阈值

硬阈值 0.8 会让 0.799 与 0.801 得到完全相反决定。随机保留能保留一些分数较低的多样内容，也让高分更常出现。但它仍继承 classifier 的偏差，且随机种子、重复运行与概率校准都要记录。

【补充】NumPy 官方文档称该 API 为 Pareto II/Lomax，并建议新代码使用 `Generator.pareto`。这支持上面的分布口径；它不替课程决定这个过滤方案是否合理。

## 9. 语言、数学、代码、毒性：四类过滤案例

这些案例的共同模板都是：**先定义目标，再寻找便宜信号，最后用训练实验验证。** 数字是论文/课程当时设置，不是永恒最佳阈值。

### 9.1 Language ID：语言识别

【课程】fastText 的公开语言识别模型覆盖 176 种语言，训练来源包括 Wikipedia、Tatoeba、SETimes；Dolma 的课程示例是保留 `p(English)>=0.5` 的页面（[11:42](https://www.youtube.com/watch?v=5sxHosTLPF8&t=702s)）。

一个页面可能是：

```text
今天我们讨论 attention. The code is below.
```

这是 code-switching（语码切换）：同一文档混用多种语言。若只取“最高概率语言”，就会抹掉这种文本。还要注意：

- 相近语言、方言可能互相误判；
- URL、代码、姓名会扰动分数；
- 阈值 0.5 只是具体 pipeline 选择，不是“英语真理线”；
- 低资源语言正例少，classifier 的 FN 可能更高。

### 9.2 OpenWebMath：组合规则、KenLM 和 fastText

【课程勘误】课程源码和口述把该项目叫作“OpenMathText”，但其一手论文 arXiv:2310.06786 的正式名称是 **OpenWebMath**。本文采用论文正式名，保留误名只为方便回查课程。OpenWebMath 流程包括：

1. 规则找 LaTeX 命令；
2. 在 ProofPile 上训练 KenLM，保留 perplexity 小于 15,000 的候选；
3. fastText 预测数学写作；含数学标记的阈值 0.17，无数学标记的阈值 0.8；
4. 得到 14.7B token；在论文特定 1.4B 参数训练设置中，比某个使用约 20 倍更多、但未这样过滤的数据基线更好（[13:08](https://www.youtube.com/watch?v=5sxHosTLPF8&t=788s)）。

`B` 在 `14.7B` 中是 billion，即十亿：

```math
14.7\text{B}=14.7\times10^9=14{,}700{,}000{,}000\text{ tokens}.
```

为什么两个 fastText 阈值不同？【补充解释】LaTeX 已提供“像数学”的额外证据，所以 classifier 可以用较低门槛；无 LaTeX 时要更确信。不能由此推出“有 LaTeX 的文档一定是数学”，因为网页模板、价格公式或坏 OCR 也会命中。

### 9.3 GPT-3 与 LLaMA 的 target 选择

【课程】GPT-3 的课程口径：正例来自 Wikipedia、WebText2、Books1、Books2，负例抽自 Common Crawl，再用词特征训练线性分类器（[14:27](https://www.youtube.com/watch?v=5sxHosTLPF8&t=867s)）。LLaMA 的正例不是 Wikipedia 文章本身，而是 **Wikipedia 引用/链接到的网页**（[14:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=894s)）。

target 的定义会决定 classifier 学到什么。若 Wikipedia 引用偏向某语言、地区和文体，模型也会复制这种选择偏差。

### 9.4 phi-1：昂贵 teacher 标小样本，便宜模型扫大池

【课程勘误】源码第 118 行一处把 phi-1 写成 1.5B，但 phi-1 论文和源码第 126–127 行都写 1.3B。本文采用论文的 **1.3B 参数**口径，不把 1.5B 混入计算。

【课程】流程是：

```text
Python subset of The Stack (R)
    │ GPT-4 按“教学价值”提示标 100K
    ▼
目标正例 T
    │ CodeGen embedding + random forest
    ▼
便宜过滤器扫完整 R
```

- **embedding（嵌入）**：把代码压成数字向量，供小分类器读取。
- **random forest（随机森林）**：许多决策树投票的分类器。
- **teacher（教师模型）**：为训练数据生成标签/回答的较强模型；teacher 也会错。

先补四个术语：

- **training step（训练步）**：optimizer 执行一次参数更新；96K steps 是 96,000 次更新，不等于 96,000 条样本。
- **optimizer（优化器）**：把 gradient 转成参数更新的规则；gradient（梯度）用人话说，是参数轻微改变时 loss 往哪个方向、变化多快。
- **learning rate（学习率）**：控制每一步更新幅度的超参数。
- **HumanEval pass@1**：给每道编程题生成一份候选，按 HumanEval 测试协议估计该一份候选通过单元测试的比例；越高越好。它不是“人类评价分”，也不等于所有真实编程任务成功率。

课程给出的论文特定结果：1.3B 模型用原始 Python 子集训练 96K steps 后 HumanEval pass@1 为 12.19%；用过滤子集 36K steps 后为 17.68%（[16:17](https://www.youtube.com/watch?v=5sxHosTLPF8&t=977s)）。绝对差为 $`17.68-12.19=5.49`$ 个百分点，不是“提高 5.49%”。比较成立的口径是同论文 HumanEval 协议；训练数据与 steps 同时不同，所以不能把全部差值只归因于“少训练 60K steps”或单一过滤因素，也不证明所有模型/benchmark 同样改善。

### 9.5 毒性过滤：目标本身包含价值判断

【课程】Dolma 的示例训练标签来自 2018 Jigsaw Toxic Comment 数据：Wikipedia talk page 被标成 toxic、severe toxic、obscene、threat、insult、identity hate 等类别（[16:36](https://www.youtube.com/watch?v=5sxHosTLPF8&t=996s)）。

风险包括：

- 引用有害言论进行批评，也可能被误删；
- 群体身份词可能被 classifier 当成毒性捷径；
- 不同文化/语境对冒犯的判断不同；
- “删掉所有负面内容”会损害安全研究、历史与文学覆盖。

因此需要人工 slice audit、阈值曲线、申诉/删除记录，而不只是报总体 accuracy。

## 10. 为什么没有一个永远最好的过滤阈值

【课程】如果训练 token 少，常希望保留更少、更高分的数据；训练 token 多，有限高分池会被重复，可能需要加入较低分但新鲜的数据（[17:42](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1062s)）。

### 10.1 同一个高分池，两种训练长度

假设：

- 高分池有 100M token；`M=million=10^6`；
- 低分但可用池有 900M token；
- 短训练要 80M token；长训练要 800M token。

短训练只用高分池：

```math
80/100=0.8\text{ epoch}.
```

长训练若仍只用高分池：

```math
800/100=8\text{ epochs}.
```

长训练加入更多低分但不重复的文档，可能比把同一高分池看 8 遍更好。这里“可能”很重要：要实验验证。

### 10.2 怎样读课程折线图

源码本地图 `data-filtering-scale.png` 是 157M 参数模型、100 WARC 小池的 preliminary experiment。横轴是训练 token 的对数刻度，纵轴是 eval/lima loss（同图口径下越低越好），虚线表示各数据版本走完一 epoch 的位置。曲线精确名称包括 `high_quality`、`med_quality`、`low_quality`、`dclm`、`llm_curated`、`llm_curated_dclm_filtered`、`nemotron_full`、`nemotron_qhigh`、`resiliparse`；“强筛选”只是对某些曲线的教学归纳，不能替代图例原名。

【视频补充】精确按图例说：**蓝色 `dclm` 曲线**先下降，但越过其 97.6M-token 一-epoch 虚线、继续大量重复后转而上升；紫色 `resiliparse` 曲线早期 loss 更高，但因数据池大，到更晚 token 区间仍下降（[18:53](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1133s)）。这只支持“最佳过滤依赖训练长度”的例证，不支持“弱过滤最终总会赢”。

### 10.3 单次 run 的不确定性

课堂问答指出每个点是一条训练 run；理想上应重复并给 confidence interval（置信区间），但训练昂贵，论文常缺（[20:34](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1234s)）。置信区间是从重复实验估计不确定范围，不保证真值一定在内。

### 10.4 过滤审计清单

1. target 谁选的？负例从哪里来？
2. 文档长度、语言、域名是否泄漏标签？
3. score 是否校准？各 slice 的 FP/FN 是多少？
4. 阈值改变后，token 数和 epoch 怎样变？
5. 下游消融是否固定模型、optimizer、训练 token 和 seed？
6. 只做一次 run，差异是否大到超过训练噪声？
7. 删除的文档是否留 hash/provenance，便于审计而又避免再暴露原文？

## 11. Deduplication：先把问题拆成三根轴

deduplication（去重）是发现重复项并采取动作。decontamination（去污染）是特别防止训练集含评估集或其改写；算法可能相似，目的不同。

【课程】网页镜像、GitHub fork 会产生 exact duplicates；许可证、页眉页脚、模板替换和标点差异会产生 near duplicates（[23:18](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1398s)）。

### 11.1 三根设计轴

| 轴 | 问题 | 可选例子 |
|---|---|---|
| item | 比较什么？ | 句子、段落、文档、代码文件 |
| match | 怎样算相同？ | exact、共同子串、Jaccard 超阈值 |
| action | 命中后怎么办？ | 全删、留一份、跨 split 优先留 train 外 |

【课程】C4 审计中一段产品描述重复 61,036 次（[25:47](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1547s)）。去重能省训练 FLOPs，也能降低大量复制造成的记忆，但不保证彻底消除记忆或法律/隐私风险。

### 11.2 为什么不能两两比较

`n` 篇文档两两比较的对数是：

```math
\frac{n(n-1)}{2}.
```

若 `n=1,000,000`：

```math
\frac{1{,}000{,}000\times999{,}999}{2}
=499{,}999{,}500{,}000,
```

约五千亿对。需要 hash/索引先找少量 candidate（候选对），再精查，而不是全量平方比较（[27:53](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1673s)）。

## 12. Hash 与 exact dedup：先做最容易的一层

### 12.1 Hash 是短指纹，不是身份证

hash function（哈希函数）把任意长度 item 映射成固定范围的整数/字符串。collision（碰撞）是 `x!=y` 却 `h(x)=h(y)`。

- SHA-256：密码学哈希，重点是难以人为制造碰撞，通常更慢；
- MurmurHash：非密码学哈希，适合快速 hash table；碰撞必须靠原文二次确认。

【课程】本讲用 `mmh3.hash("hello")` 演示 MurmurHash（[29:31](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1771s)）。不能把“hash 相同”直接当“文档相同”。

### 12.2 源码小例逐行

```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
hash_items = itertools.groupby(
    sorted(items, key=mmh3.hash),
    key=mmh3.hash,
)
deduped_items = [next(group) for h, group in hash_items]
```

人话翻译：

1. `sorted(..., key=mmh3.hash)` 按 hash 排序，让相同 hash 相邻；
2. `groupby(..., key=mmh3.hash)` 把相邻相同 hash 分组；
3. `for h, group in hash_items` 逐组拿到 hash 值 `h` 和迭代器 `group`；
4. `next(group)` 只取组内第一个。

结果一定把两个完全相同的 `"hello"` 压成一份；`"Hello!"` 仍不同于 `"hello"`。

**生产边界：** 若只按 hash 分组、不核原字符串，罕见 collision 会误删。还要用稳定分区、记录 canonical item（代表项）与来源。

### 12.3 normalize 会改变“相等”

normalize（规范化）可做小写、Unicode 统一、空白折叠、删标点：

```text
"Hello!" → lower → "hello!" → remove punctuation → "hello"
```

这样 recall 增加，但 FP 也可能增加。例如代码 `A-B` 与 `AB` 删符号后相同，却可能语义不同。规则必须按文档类型审计。

### 12.4 MapReduce 心智模型

MapReduce 是大规模“先并行产生 key-value，再按 key 聚合”的模式：

```text
Map: 文档 → (hash, 文档ID)
Shuffle: 相同hash送到同一组
Reduce: 核原文，选择保留项
```

源码是教学缩影，不包含分布式 shuffle、碰撞核验、失败重试与 provenance。

【课程】C4 以三句 span（连续片段）做 exact match，再删重复片段；这可能从文档中间挖掉三句，破坏 coherence（连贯性）（[30:39](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1839s)）。

## 13. Jaccard 与 MinHash：把近似相同变成可数事件

### 13.1 先把文档变成集合

shingle 是连续 `k` 个 token/字符的小片段。例：字符 2-shingle：

```text
"abcd" → {"ab", "bc", "cd"}
"abce" → {"ab", "bc", "ce"}
```

集合丢掉重复次数；若需要多重计数，要用别的相似度。

### 13.2 Jaccard 手算

定义：

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
```

课程例：

```math
A=\{1,2,3,4\},\qquad B=\{1,2,3,5\}.
```

- 交集 `A∩B={1,2,3}`，大小 3；
- 并集 `A∪B={1,2,3,4,5}`，大小 5；
- `J=3/5=0.6`。

`J=0` 表示没有共同项；`J=1` 表示两个集合相同（[31:44](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1904s)）。

### 13.3 MinHash 性质从五个事件证明

MinHash 做法：对并集元素使用一次随机 permutation（随机排列），每个集合取排列中最先出现的成员。

在课程例中，并集有五个元素，假设每个元素成为第一名的概率都为 `1/5`：

| 并集第一名 | `min(A)` 与 `min(B)` | 原因 |
|---:|---|---|
| 1 | 相同 | 1 在交集 |
| 2 | 相同 | 2 在交集 |
| 3 | 相同 | 3 在交集 |
| 4 | 不同 | 4 只在 A |
| 5 | 不同 | 5 只在 B |

所以：

```math
P[h_{min}(A)=h_{min}(B)]=3/5=J(A,B).
```

这个等式依赖 minwise independence（每个并集元素同样可能排第一）的随机排列/理想 hash 假设。普通任意 hash 不自动满足（[34:43](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2083s)）。

### 13.4 多个 signature 只是估计

若做 8 个独立 MinHash，观察到 6 个相同：

```math
\hat J=6/8=0.75.
```

帽子 `^` 表示估计值。真实 Jaccard 仍要从原集合算；`0.75` 不等于精确 Jaccard。

如果真实 `J=0.6`，单个 match 是 0/1 随机量；用更多独立 hash，平均的波动通常变小。源码用 100 个 seed 并断言误差小于 0.01 只是一条固定样例检查，不是概率保证，也不替代 production 测试（[36:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2190s)）。

## 14. LSH：用 AND-OR 把相似度曲线变陡

LSH（Locality-Sensitive Hashing，局部敏感哈希）让相近项更容易成为候选。这里使用 MinHash signature。

### 14.1 bands 与 rows

共 `n=b*r` 个 MinHash：

- `b`：band 数；
- `r`：每个 band 内的 hash 数；
- 一个 band 的 `r` 个值全部相同才算该 band match（AND）；
- 任意一个 band match 就成为 candidate（OR）。

课程 `n=12,b=3,r=4`：

```text
band1: h1 h2 h3 h4
band2: h5 h6 h7 h8
band3: h9 h10 h11 h12
```

### 14.2 从单个 hash 推到候选概率

设真实 Jaccard `s`，各 MinHash 独立：

1. 一个 hash 相同概率：`s`；
2. 固定 band 的 `r` 个都相同：`s^r`；
3. 固定 band 不匹配：`1-s^r`；
4. `b` 个 band 全不匹配：`(1-s^r)^b`；
5. 至少一个 band 匹配：

```math
P_{candidate}=1-(1-s^r)^b.
```

### 14.3 `s=0.8,b=5,r=10` 完整算

```math
s^r=0.8^{10}=0.1073741824.
```

```math
1-s^r=0.8926258176.
```

```math
(1-s^r)^b=0.8926258176^5\approx0.5666921796.
```

```math
P_{candidate}=1-0.5666921796=0.4333078204\approx43.33\%.
```

视频口头说“约 0.4”（[41:27](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2487s)），精确到四位是 0.4333。

### 14.4 调 `b` 与 `r`

- `r` 增大：每 band 要同时通过更多 hash，更难，曲线右移；
- `b` 增大：尝试次数更多，更容易，曲线左移；
- 两者都增大可使过渡更陡，但 signature、内存和索引成本上升。

例 `s=0.9`：

| `b,r` | `1-(1-s^r)^b` |
|---|---:|
| 10,10 | 0.9863 |
| 10,20 | 0.7264 |
| 20,20 | 0.9252 |

这验证：只增 `r` 更难；再增 `b` 又变容易（[45:07](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2707s)）。

### 14.5 heuristic threshold 不是 50% 点

课程真实设置例：`b=20,r=450,n=9000`。常用启发式阈值：

```math
s_*=(1/b)^{1/r}=(1/20)^{1/450}\approx0.9933649271.
```

因为 `s_*^r=1/b=0.05`，此点候选概率是：

```math
1-(1-1/20)^{20}=1-(19/20)^{20}
\approx0.641514.
```

所以它约是 64.15% 候选概率点，不是精确 50% threshold。课程称 phase transition 大致发生附近，是 heuristic（启发式），不是硬门槛（[46:42](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2802s)）。

### 14.6 Candidate 后仍要精算

LSH 只负责召回少量候选。标准防错流程：

```text
MinHash/LSH 找候选对
   → 取原 shingle 集合
   → 精算 Jaccard
   → 达真实阈值才合并/删除
```

- false positive：不够相似却撞 band；后精算可删掉；
- false negative：很相似却没撞 band；后精算看不到，需调 `b,r`、多 probe 或换候选策略。

## 15. 跨来源、跨 split 去重与 contamination

【课程】只在每个数据集内部去重不够；两个数据集之间也可能重复（[48:55](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2935s)）。

### 15.1 三层检查

1. **within-source**：同一来源内部；
2. **cross-source**：Wikipedia、Common Crawl、代码来源之间；
3. **cross-split**：train、validation、test 之间。

若 benchmark 问题出现在训练集，evaluation 分数会被 contamination（污染）抬高。decontamination 不只查题目逐字相同，还要考虑答案、解释、翻译和模板化改写；越模糊的匹配越可能误删普通知识。

### 15.2 action 需要优先级

候选集 `{train_doc, validation_item, test_item}` 重复时，通常优先保留评估集，删除训练副本；若评估题已公开多年且被广泛引用，完全清除可能不现实，此时应报告检测规则、残留风险与时间边界。

### 15.3 去重不能替代这些事

- hash 去重不能判断版权许可；
- near dedup 不能保证隐私文本不被记忆；
- benchmark 字面去重不能发现概念泄漏；
- 删除重复不能修复错误抽取；
- 去重后 token 数变化，mixture 和 epoch 必须重算。

## 16. Mixing：多个来源到底各抽多少

【课程】语言模型会混合 Wikipedia、网页、代码、书籍、数学等来源。课程的 Marin token viewer 图片只是一份当时计划/数据快照；横轴是各来源 token 数（十亿），颜色是 web、multilingual、math、code 等类别，不是推荐权重（[49:49](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2989s)）。

### 16.1 先分开 token share $`p_i`$ 与 sequence 概率 $`q_i`$

有 $`m`$ 个来源。本文为保证 epoch 公式量纲正确，固定：

- $`p_i`$：来源 $`i`$ 占全部**训练 token 位置**的份额；
- $`q_i`$：按整条 sequence 抽来源时，抽到来源 $`i`$ 的概率。

两组都各自非负且和为 1：

```math
p_i\ge0,\qquad \sum_{i=1}^{m}p_i=1.
```

```math
q_i\ge0,\qquad \sum_{i=1}^{m}q_i=1.
```

token share 例：

```text
Wikipedia 0.3 + Common Crawl 0.5 + GitHub 0.2 = 1.0
```

这表示长期约 30%、50%、20% 的 **token 位置**来自各来源。

为什么 $`p_i`$ 不一定等于 $`q_i`$？设 code sequence 平均 100 token，web sequence 平均 400 token；若 $`q_{\text{code}}=q_{\text{web}}=0.5`$，每抽两条期望得到 100 个 code token、400 个 web token，所以

```math
p_{\text{code}}=\frac{100}{100+400}=0.2,\qquad
p_{\text{web}}=0.8.
```

要实现目标 token share $`p_i`$，可按 token budget 直接调度；若只能按 sequence 抽样，近似需要“与 $`p_i/\bar L_i`$ 成正比”。符号 $`\propto`$ 读作“正比”：它只给各来源的**相对权重**，尚未保证和为1。完整归一化式是

```math
q_i
=\frac{p_i/\bar L_i}
{\sum_j p_j/\bar L_j}.
```

其中 $`\bar L_i`$ 是来源 $`i`$ 的平均 sequence 长度，分母把所有未归一化权重加起来。

完整手算：目标 $`p_{\text{code}}=p_{\text{web}}=0.5`$，平均长度分别100、400 token。

```math
\frac{p_{\text{code}}}{\bar L_{\text{code}}}
=0.5/100=0.005,
\qquad
\frac{p_{\text{web}}}{\bar L_{\text{web}}}
=0.5/400=0.00125.
```

相对比为 $`0.005:0.00125=4:1`$，和为 $`0.00625`$，所以

```math
q_{\text{code}}=0.005/0.00625=0.8,
\qquad
q_{\text{web}}=0.00125/0.00625=0.2.
```

每次 sequence 抽样贡献的期望 token 分量是

```math
0.8\times100=80\ \text{code tokens},
\qquad
0.2\times400=80\ \text{web tokens}.
```

两边都是80，所以 token share 为 $`80/(80+80)=0.5`$。等长 sequence 时 $`\bar L_i`$ 相同，才有 $`q_i=p_i`$。

### 16.2 四种 baseline

假设来源 token 数：A=100，B=300，C=600，总计 1000。

| 方法 | 定义 | A/B/C 权重 |
|---|---|---|
| vibes | 人凭经验填 | 例如 0.2/0.3/0.5 |
| uniform | 每来源相同 | 1/3,1/3,1/3 |
| proportional | 按 token 数 | 0.1/0.3/0.6 |
| power | `p_i ∝ N_i^alpha` | `alpha` 在 0 与 1 间时介于前两者 |

`∝` 读“成比例”。使用 power 时还要 normalize：先算各 `N_i^alpha`，再除以它们的和。

例 `alpha=1/2` 即开平方：

```math
\sqrt{100}=10,\quad\sqrt{300}\approx17.32,\quad\sqrt{600}\approx24.49.
```

总和 `51.81`，所以 A 权重 `10/51.81≈0.193`，B `≈0.334`，C `≈0.473`。

【课程】uniform 与 proportional 都有合理直觉，也都有缺陷：大但低质来源会支配 proportional；uniform 会让很小来源重复很多（[51:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3114s)）。

课程的 The Pile 表还说明“原始大小”和“有效混合大小”不是同一列。Wikipedia 原始大小为 6.38 GiB，权重/重复系数为 3，所以

```math
6.38\ \mathrm{GiB}\times3=19.14\ \mathrm{GiB}.
```

表中全部来源有效大小总计 1254.20 GiB，因此 Wikipedia 的有效混合占比约

```math
19.14/1254.20\approx0.01526=1.526\%.
```

这里 GiB 是二进制字节单位，$`1\ \mathrm{GiB}=2^{30}`$ bytes；它不是 token 数。课程表的“weight/epochs”是该数据集配方口径，不可直接当本文所有 $`p_i`$ 的定义。

### 16.3 positions 与 epochs：最重要的两式

符号卡：

- `T_train`：整次训练要消费的 token 位置数；
- `N_i`：来源 `i` 有多少可用 token；
- `p_i`：来源 `i` 的 token share，不是默认的 sequence 抽样概率；
- `q_i`：若按 sequence 抽来源时的概率，不直接进入下面 epoch 公式；
- `p_i T_train`：该来源被请求的训练位置数；
- `e_i`：该来源平均被重复几轮。

```math
\text{positions}_i=p_iT_{train},
\qquad
e_i=\frac{p_iT_{train}}{N_i}.
```

**单位检查：** 分子和分母都是 token，所以相除后 epoch 没有单位。

### 16.4 课程 10T/10B 例一步不跳

- 低质量大来源：`N_low=10T=10*10^12 token`；
- 高质量小来源：`N_high=10B=10*10^9 token`；
- 总训练：`T_train=1T=10^12 token`；
- 混合：`p_low=p_high=0.5`。

低质量来源：

```math
p_{low}T_{train}=0.5\times10^{12}=5\times10^{11}=500\text{B tokens},
```

```math
e_{low}=\frac{0.5\times10^{12}}{10\times10^{12}}
=\frac{0.5}{10}=0.05\text{ epoch}.
```

高质量来源：

```math
p_{high}T_{train}=500\text{B tokens},
```

```math
e_{high}=\frac{500\text{B}}{10\text{B}}=50\text{ epochs}.
```

【视频补充】讲者明确说，50 epochs 不是“需要 50 遍”，而是盲目设 50/50 mixture 后不知不觉造成的；最好是浪费计算，最坏是过拟合（[56:37](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3397s)）。

### 16.5 mixture 在 batch 中怎样实现

batch（批次）是一次并行训练的一组 sequence（序列）。常见实现是每条 sequence 先按 $`q_i`$ 抽来源，再取一条；不是在一个句子内部逐 token 切换来源。

例：batch size=8、$`q_{\text{code}}=0.25,q_{\text{web}}=0.75`$，期望 2 条 code、6 条 web，但单 batch 可为 1/7 或 3/5。若两来源都填充/打包成相同有效长度，token share 也约为 25%/75%；若 code 平均更短，实际 $`p_{\text{code}}`$ 会低于 25%。可按 token 计数反馈调整 $`q_i`$，或用 token-budget scheduler 直接控制 $`p_i`$（[57:26](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3446s)）。

## 17. UniMax：给重复次数装一个保险丝

【课程】UniMax 面向多语言预训练：尽量均匀覆盖，但给每个来源设置最大 epoch `C`，避免低资源语言被重复过多（[58:33](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3513s)）。

### 17.1 讲义公式的量纲错误

源码写：

```text
p(s) * num_training_tokens <= C
```

左边单位是 token，右边 `C` 若是 epoch 就没有单位，不能直接比较。缺了来源大小 `N_s`。正确 cap 是：

```math
\frac{p_iT_{train}}{N_i}\le C
```

等价于：

```math
p_iT_{train}\le C N_i.
```

左、右两边现在都是 token。

### 17.2 两来源完整分配

设：

- A 有 `N_A=100` token；
- B 有 `N_B=20` token；
- 训练 `T_train=100` token；
- cap `C=2 epochs`。

若 uniform：A 50、B 50。

```math
e_A=50/100=0.5,
```

```math
e_B=50/20=2.5>2,
```

B 超 cap。B 最多能贡献：

```math
C N_B=2\times20=40\text{ token}.
```

把剩余 10 个位置给 A：

```text
A: 60 token → 0.6 epoch
B: 40 token → 2.0 epochs
总计 100 token
```

权重变为 `p_A=0.6,p_B=0.4`。这只是最小教学例；真实 UniMax 有多来源分配程序。

### 17.3 cap 不是“重复绝对有害”

第二次看数据仍可学习；cap 是风险控制，不是自然常数。`C=20` 是课程转述论文设置的例子，不应复制到所有模型（[59:14](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3554s)）。

## 18. RegMix 与 simulated epoching

### 18.1 RegMix 的四步

regression（回归）是从输入数字预测连续值，例如从 mixture 权重预测 validation loss。

【课程】RegMix 类方法：

1. 从 mixture 空间采样许多权重；
2. 每组权重训练一个小 proxy model（代理模型）；
3. 用 `(mixture → loss)` 数据拟合 regression；
4. 在预测曲面找较低点，用于大模型（[60:45](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3645s)）。

课程图 `regmix.png` 的例子有 Hacker News/GitHub/Philpapers 三个权重，目标 loss 越低越好。它展示方法流程，不证明图中 22.8%/67.0%/10.2% 是任何别的项目的最佳混合。

**$`R^2`$（决定系数）**粗略问“回归比只猜平均值解释了多少验证 loss 变化”；接近 1 表示已采点上的拟合较好，不证明外推最低点正确。**bootstrap（自助法）**是对已有实验点有放回重抽、反复重拟合，观察最佳 mixture 漂移多大；它只能反映现有样本中的不稳定性，不能补上没采到的区域。

### 18.2 Dirichlet 只需先懂这个

Dirichlet distribution（狄利克雷分布）是一种“随机生成一组非负且和为 1 的权重”的分布。三来源可抽到：

```text
(0.2,0.3,0.5), (0.8,0.1,0.1), (0.05,0.15,0.8)
```

它不是 optimizer，也不保证覆盖极端角落。要记录采样先验和覆盖范围。

### 18.3 一个 tiny regression 例

只混 A/B，令 `p_B=1-p_A`。三次小 run：

| `p_A` | `p_B` | 验证 loss |
|---:|---:|---:|
| 0.2 | 0.8 | 2.4 |
| 0.5 | 0.5 | 2.0 |
| 0.8 | 0.2 | 2.2 |

模型可拟合一条 U 形曲线并预测 0.5 附近较好。但若只试 `0.2–0.8`，optimizer 却给 `p_A=0.99`，那是 extrapolation（外推）；数据几乎没约束，风险高。

### 18.4 两个 leap of faith

【视频补充】讲者把它称为两个“信念跳跃”（[64:50](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3890s)）：

1. regression 在找到的 minimizer（预测最低点）仍准确；
2. 小模型最佳 mixture 能迁移到大模型。

还要防 evaluation overfit：若目标全是 code benchmark，回归自然把 code 权重推高；这不等于通用写作也更好。

### 18.5 Simulated epoching 手算

目标：让小试验也感受到大训练的“数据稀缺压力”。

- 小 run：`10B token`；
- 大 run：`1T=1000B token`；
- 比例：

```math
\rho=10B/1000B=0.01.
```

把每个来源大小乘缩放比例 $`\rho`$。这里不用 $`q_i`$，避免和 §16 的 sequence 抽样概率混淆：

```math
10T\times0.01=0.1T=100B,
```

```math
10B\times0.01=0.1B=100M.
```

若小 run 仍用 50/50：各要 5B token。

```math
e_{low}=5B/100B=0.05,
```

```math
e_{high}=5B/0.1B=50.
```

正好复现大 run 的 0.05/50 epoch 压力（[68:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4112s)）。

### 18.6 它仍会失真

- 小到某来源只剩不到一条 sequence，会发生 rounding（取整）误差；
- 小模型与大模型能力/optimizer 不同；
- 训练 token 比例相同不代表梯度动态相同；
- 来源质量随时间、抽取器、去重器版本变化；
- proxy 试验太多次盯同一 eval，会把 eval 当训练目标。

【视频补充】课堂问答明确提到，下采样太小会把“训练一次”舍入成零次，需要最低覆盖等修正（[70:52](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4252s)）。

## 19. Post-training 数据从哪里来

【课程】基本 recipe（流程配方）是：

```text
environment（环境）
   + task/prompt（任务/提示）
   + teacher response（教师回答）
   → 验证/过滤 → 后训练样本
```

- **environment**：任务运行所需的世界，例如仓库、依赖、测试、文件系统；
- **prompt**：给模型的输入要求；
- **response/trajectory**：回答，或包含多步工具动作的轨迹；
- **scaffold**：围绕语言模型的工具调用、记忆、循环与错误处理代码；
- **synthetic data（合成数据）**：全部或部分由程序/模型生成的数据。

【视频补充】开源社区的后训练回答大量由强模型生成；human response 更慢、更贵，也不自动无错（[73:48](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4428s)）。

### 19.1 三种 prompt 来源

| 类型 | environment | task/prompt | 例子 | 主要风险 |
|---|---|---|---|---|
| fully synthetic | 合成或无环境 | 合成 | 模型造数学题 | 模板单一、自我复制错误 |
| semi-synthetic | 真实 | 合成 | 真实 repo 上造 bug task | 任务不自然、许可/隐私 |
| real prompt | 真实或线上 | 真实 | 真实 PR/用户请求 | 隐私、同意、可复现性 |

“真实”不等于无污染；“合成”也不等于没有版权/隐私来源，因为 teacher 及环境仍有来源链。

## 20. OpenThoughts3：强模型不一定是更好的老师

【课程快照】这里的 1.2M/每题 16 responses 绑定到课程引用的 **OpenThoughts3 数据快照**，不是所有 OpenThoughts 版本的通用规模：

- 1.2M examples；
- 使用 QwQ-32B 作为 teacher；
- 问题来自 27 个人类与合成来源；
- 每 prompt 采样 16 个 response 有帮助；
- 在论文该设置里 QwQ-32B 比 DeepSeek-R1 更适合作 teacher；
- 基础 answer filtering 没有帮助；较小高质量来源优于把所有来源都纳入（[76:26](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4586s)）。

### 20.1 1.2M 除以 16

若每个问题恰生成 16 个答案：

```math
1{,}200{,}000/16=75{,}000\text{ questions}.
```

课程流程图也显示 science 6K + code 16K + math 53K = 75K；`75K*16=1.2M`。这解释“example 数”和“独特 prompt 数”为什么不同。

### 20.2 为什么更强的答题模型未必更会教

teacher 数据还需要：

- 展示适合学生模仿的步骤；
- 不把猜测写成事实；
- 风格多样；
- 对任务分布匹配；
- 生成成本可接受；
- 通过独立 verifier（验证器）检查。

某 benchmark 更强只说明它在那个协议下得分高，不保证其轨迹更可学。OpenThoughts 结论是该论文设置的 ablation，不能推广为 QwQ 永远优于 DeepSeek-R1。

## 21. SWE-smith、SWE-Zero、SWE-Hero、SWE-rebench

SWE 是 Software Engineering（软件工程）。PR 是 pull request（代码修改请求）。trajectory 是 agent 为完成任务产生的观察、思考、工具调用和修改序列。

### 21.1 SWE-smith：真实仓库 + 合成任务

【课程快照】流程图：真实 repository 与 unit tests → 构建可运行环境 → 程序修改/LM 生成/组合 bug/PR mirror → 新任务实例；128 个 GitHub 仓库得到约 50K tasks（[78:14](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4694s)）。

unit test（单元测试）是检查一小块程序行为的自动化测试。**通过已有测试不等于完全正确**：测试可能漏 edge case（边界情况），也可能奖励投机 patch。

### 21.2 SWE-Zero：没有 repo execution feedback

【课程快照】软件仓库依赖复杂，旧 commit 可能难安装。课程介绍 SWE-Zero：使用真实 GitHub PR，生成约 300K 不要求仓库特定执行的 agent trajectories，涉及约 150K PR，使用 OpenHands scaffold（[79:19](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4759s)）。

课程图比较 execution 与 no-execution 的解题率；其中 SWE-bench **V=Verified（人工核验子集）**，**M=Multilingual（多语言子集）**。百分比是对应 benchmark、agent scaffold 和预算下的解决率。不能推出“执行不重要”，只能说明一些强模型在该设置下即使无执行也能完成不少任务。

【安全边界】论文/课程提到移除未来 commit，防止 agent 直接读取答案式未来修改。这里仅保留防守原则：隔离任务时点后的信息、审计工具访问、记录每次命令；不提供绕过方法。

### 21.3 teacher、过滤与 SWE-Hero

【课程快照】SWE-Zero 轨迹由 Qwen3-Coder-480B teacher 生成并过滤掉违反无执行协议的样本；SWE-Hero 约 13K 轨迹需要 execution feedback（[81:05](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4865s)）。

过滤器要检查：

1. 是否遵守工具规则；
2. patch 是否修改了禁止区域；
3. 测试是否真的运行、版本是否固定；
4. PR 的 license 与作者信息；
5. secrets、email、用户数据等 PII（Personally Identifiable Information，个人可识别信息）；
6. benchmark task 是否在 teacher 训练/检索内容中。

### 21.4 SWE-rebench 与 12M trajectories

【课程快照】SWE-rebench：约 21K 交互式 Python SWE tasks，来自约 3.4K 仓库、450K PR；用 Qwen2.5-72B-Instruct 辅助安装依赖与评估 PR 质量（[81:45](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4905s)）。

课程随后介绍当日发布的数据快照：12M trajectories；基于 SWE-rebench-v2，约 32K executable tasks 和 120K non-executable tasks；课程转述 mini-coder-1.7B 在对应固定模型、prompt、temperature、预算、scaffold 和评测协议下 **pass@100=50.4%**（[82:18](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4938s)）。

pass@100 不是“单次回答正确率 50.4%”，而是考察 $`k=100`$ 份候选中至少一份通过；真实评测可从每题已经生成的 $`n\ge100`$ 份候选估计。

先区分两个公式。

1. **独立同成功率玩具模型**：每次独立、成功率都为 $`p`$，则

```math
P(\text{至少一中})=1-(1-p)^k.
```

例如 $`p=0.2`$，2 次为 $`1-0.8^2=0.36`$，3 次为 $`1-0.8^3=0.488`$。

2. **有限已有样本的常用无放回估计**：某题已经生成 $`n`$ 份，其中 $`c`$ 份通过；从这 $`n`$ 份里不重复地均匀选 $`k`$ 份，要求 $`n\ge k`$：

```math
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom nk}.
```

$`\binom ab`$ 是“从 $`a`$ 个不同对象里不重复选 $`b`$ 个子集”的数量。若 $`n=5,c=2,k=2`$，总子集 $`\binom52=10`$，全失败子集从3个失败候选选2个，$`\binom32=3`$，所以

```math
\widehat{\mathrm{pass@}2}=1-3/10=0.7.
```

若 $`n-c<k`$，失败候选不够组成一个大小为 $`k`$ 的全失败子集，约定 $`\binom{n-c}{k}=0`$，所以估计值为 $`1-0=1`$。benchmark 通常先对**每一道题**算 pass@k，再对题目取平均。

这是对已生成候选做无放回子集估计，不是说语言模型生成时在无放回抽答案。对固定题目、固定生成分布做独立随机采样时，可把候选近似看作条件独立；但 beam search、共享 candidate pool、复用生成状态、自适应 retry 或后一次读取前一次结果时，候选可能相关。不同题目成功率也高度异质，不存在跨题统一的一个 $`p`$。因此汇总 pass@100 不能用单一独立 $`p`$ 反推单次正确率。这些数据是 2026 快照，不是当前 leaderboard。

### 21.5 代码数据需要互补的多轴验证

这些检查不是 strict “弱 < 强”的全序；尤其 static type check 与 dynamic tests 能发现不同错误：

| 防线 | 能发现 | 不能保证 |
|---|---|---|
| 语法解析 | 括号、语法树错误 | import 存在、运行正确 |
| compile/import | 模块找不到、部分类型/链接错误 | 业务语义正确 |
| static type/lint | 声明类型、未用变量、部分危险模式 | 运行时数据和所有动态行为 |
| 原有 dynamic tests | 已覆盖输入上的行为回归 | 未写的 edge case 与任务新要求 |
| 新增任务测试 | 针对本任务的预期行为 | 测试本身完整、无投机 patch |
| 人工 review | 意图、可维护性、明显安全/性能问题 | 审查者不漏错 |
| 真实使用反馈 | 真实分布中的失败 | 低频/未出现问题及隐私安全 |

应组合多轴证据，而非看到一项通过就停止。无法执行的轨迹可扩大数量，但少了动态证据；可执行任务更贵，也会因环境搭建失败产生 selection bias。

## 22. 端到端数据流水线与决策树

### 22.1 一条可审计流水线

```text
1. 定义用途、法务/隐私边界和允许来源
2. 抓取/接收 dump，记录 URL、时间、许可、版本
3. 按类型路由 HTML/PDF/repo
4. 抽取；保存 extractor/version/置信度
5. 规则安全筛查与语言识别
6. 质量/领域/toxicity model 打分
7. 在人工 gold 上选阈值；做语言/群体 slice
8. normalize；先 exact dedup，再 near-dedup candidate+精算
9. 对全来源及 train/eval 做 decontamination
10. 重算每来源 token 数 N_i
11. 设计 mixture，计算 positions 与 epochs，应用 cap
12. 小规模 ablation；保留未调参验证集
13. 冻结 dataset manifest 后训练
14. 监控异常、可删除性、模型记忆与评估污染
```

### 22.2 “这条数据能不能进？”决策树

```text
来源和访问/许可/隐私边界清楚吗？──否→隔离，补证据；不要靠过滤洗白
  │是
能可靠抽取语义吗？────────────否→换抽取器/保留结构/丢弃
  │是
与目标用途相符吗？────────────否→低权重、分域或删除
  │是
是否 exact/near duplicate？────是→按优先级留代表项
  │否
是否撞 benchmark/保留集？─────是→从训练删除并记录
  │否
mixture 会让它超过 epoch cap 吗？─是→降低 p_i 或扩充真实新数据
  │否
写入版本化 manifest，做下游 ablation
```

manifest 是列出数据文件、hash、数量、版本和处理参数的清单；它让同一训练配方可复现。

## 23. Audit tables：真正做项目时填什么

### 23.1 Extraction card

| 字段 | 示例 |
|---|---|
| 来源/时间 | `example.org`, crawl `2026-03` |
| 输入 | HTML / PDF / repo |
| 抽取器 | Resiliparse x.y / OCR model hash |
| 文档数 | 10M |
| 人工 gold | 分语言各 200 页 |
| precision/recall | 92% / 84% |
| 特殊结构 | 表格、双栏、公式、代码 |
| 已知失败 | 阿拉伯语 RTL 顺序错误 |

### 23.2 Filtering card

| 字段 | 必须回答 |
|---|---|
| target positives | 谁选、哪一时期、哪些语言？ |
| negatives | 是否只是“随机网页”？ |
| features | 是否泄漏 URL、长度、来源域？ |
| threshold | 为什么选？P/R 曲线在哪里？ |
| stochastic rule | 公式、seed、期望保留率 |
| slices | 每语言/文体/身份词的 FP/FN |
| scale | 保留 token、预期 epochs |
| ablation | 固定了什么？重复几次？ |

### 23.3 Dedup card

| 字段 | 示例 |
|---|---|
| item | 5-token shingles / document |
| normalization | Unicode NFC、折空白；不删标点 |
| exact hash | SHA-256 + 原文核验 |
| near metric | Jaccard |
| LSH | `b=20,r=450` |
| exact threshold | 0.995 |
| action | 同簇留最早、许可最清楚的一份 |
| scope | 全来源 + 所有 split |

### 23.4 Mixture card

| source | `N_i` token | `p_i` | positions `p_iT` | epochs | cap |
|---|---:|---:|---:|---:|---:|
| books | 20B | .10 | 100B | 5 | 5 |
| code | 100B | .20 | 200B | 2 | 5 |
| web | 2T | .70 | 700B | .35 | 5 |

这里 `T_train=1T`。检查：权重 `.1+.2+.7=1`，没有来源超过 5 epochs。

### 23.5 Synthetic provenance card

```text
environment/repo + commit + license:
prompt/task origin:
teacher model + version + decoding settings:
scaffold/tools + versions:
execution/network policy:
verifier/tests:
filter/drop reasons:
PII/secrets scan:
benchmark overlap scan:
human review sample:
```

## 24. 常见误区：错误 → 原因 → 正确说法

1. **“网页抓下来就是文本。”** HTML 含结构和噪声。→ 先抽取并审计 reading order。
2. **“抽到所有字符就无损。”** 表格关系、图像指代仍会丢。→ 同时评结构语义。
3. **“PDF 天然比 HTML 干净。”** PDF 可能扫描、截断、乱序。→ 质量与格式是两件事。
4. **“更智能的 VLM 抽取必然更好。”** 它更慢且会幻觉。→ 用 gold 与下游 ablation 比。
5. **“score=0.9 就有 90% 真优质。”** 未校准时无此含义。→ 做 reliability/calibration 检查。
6. **“quality 是客观单值。”** 数学、诗歌、代码目标不同。→ 写清 target 与用途。
7. **“阈值越高越好。”** recall 和数据量下降，重复增加。→ 联合训练长度选阈值。
8. **“总体 accuracy 高就公平。”** 少数语言可被大量误删。→ 做 slice FP/FN。
9. **“`score` 就是 GPT-3 保留概率。”** Pareto 规则非线性。→ 保留率为 `(2-score)^-9`。
10. **“随机保留等于无偏。”** classifier 偏差仍在。→ 随机性只软化边界。
11. **“训练更久只用最高质池。”** 有限池会多 epoch。→ 比较新鲜度与质量。
12. **“一次 run 的曲线就是规律。”** seed/设置有限。→ 重复、置信区间、跨规模验证。
13. **“hash 相同就是原文相同。”** 非密码学 hash 有 collision。→ 核原文。
14. **“SHA-256 就完全不会碰撞。”** 概率极低，不是数学不可能。→ 关键删除仍核内容。
15. **“normalize 越多越好。”** 可能把不同代码/语言合并。→ 类型化规范化。
16. **“exact dedup 能处理模板改写。”** 一个标点就不 exact。→ near dedup。
17. **“Jaccard 0.6 表示 60% 字符相同。”** 它是所选集合 item 的交并比。→ 先说明 shingle。
18. **“6/8 MinHash match 就是精确 J=0.75。”** 那是随机估计。→ 候选后精算。
19. **“任何 hash 的 MinHash 都严格满足定理。”** 需要近似随机 minwise 假设。→ 验证 hash family。
20. **“LSH candidate 就是 duplicate。”** 有 FP。→ 再算真实 Jaccard。
21. **“`(1/b)^(1/r)` 是硬阈值/50% 点。”** 它是启发式中心，例中概率约 .6415。→ 报完整曲线。
22. **“增大 `b` 更严格。”** band 越多越容易任一命中。→ `b` 左移，`r` 右移。
23. **“只在每个来源内部去重够了。”** 跨来源会重复。→ 联合索引。
24. **“去重等于 decontamination。”** 后者专注 train/eval 泄漏。→ 目的与优先级分开。
25. **“权重和为 1 就是好 mixture。”** 还要查 `N_i` 与 epochs。→ 算 `p_iT/N_i`。
26. **“50 epochs 表示模型需要 50 遍。”** 是混合导致的暴露量。→ cap 或调权重。
27. **“每一步只抽一个来源才符合 mixture。”** 可在 batch 内混。→ 长期频率才是定义。
28. **“UniMax 源码式量纲正确。”** 少 `N_i`。→ `p_iT<=CN_i`。
29. **“cap=20 是通用常数。”** 是论文/项目选择。→ 按风险与消融定。
30. **“proxy 最佳 mixture 一定迁移。”** 小大规模会变。→ simulated epoching + 大规模验证。
31. **“回归模型 R² 高就能安全优化。”** 最低点可能在样本稀少的边缘。→ 留验证 mixture。
32. **“downsample 只减少算力，不改问题。”** 太小会整条来源消失。→ 最低覆盖与取整规则。
33. **“teacher benchmark 更强就更会教。”** 可学性与正确率不同。→ 教师消融。
34. **“合成数据没有许可和隐私问题。”** 环境、prompt、teacher 都有来源。→ provenance card。
35. **“代码测试通过等于语义完全正确。”** 测试会漏边界。→ 多层验证与人审。
36. **“no-execution 数据没有价值。”** 可扩大真实任务轨迹。→ 但验证信号更弱。
37. **“execution 数据总更真实。”** 能成功安装的仓库形成 selection bias。→ 报安装失败分布。
38. **“课程模型/数据集数字是当前榜单。”** 它们是 2026 课程快照。→ 链接具体版本与日期。
39. **“过滤能修复来源许可问题。”** 质量处理不改变法律权利。→ 许可/ToS/隐私独立审查。
40. **“数据工作只是套算法。”** 真实工作大量依赖看失败案例。→ 版本化审计与人工抽样。

## 25. 公式、符号和单位卡

| 公式 | 含义 | 条件/边界 |
|---|---|---|
| `precision=TP/(TP+FP)` | predicted positive 中 actual positive 的比例 | 先声明 extraction/filtering 的标签契约 |
| `recall=TP/(TP+FN)` | actual positive 中被预测 positive 的比例 | actual positive 必须由明确规则/gold 定义 |
| `P_keep(s)=(2-s)^-9` | 课程 GPT-3 Pareto 代码的保留概率 | NumPy Lomax、`0<=s<=1` |
| `J=|A∩B|/|A∪B|` | 集合相似度 | item/shingle 定义固定 |
| `P(MinHash match)=J` | 单个随机 MinHash 碰撞概率 | minwise/随机排列假设 |
| `P_LSH=1-(1-s^r)^b` | LSH 成为候选概率 | hash 独立近似 |
| `s*=(1/b)^(1/r)` | 常用 LSH 启发式中心 | 非硬阈值、非 50% 点 |
| `sum p_i=1` | token share 权重 | `p_i>=0`；不要默认等于 sequence 概率 |
| `sum q_i=1` | sequence 来源抽样概率 | 等长或 token-budget 校正时才可与 `p_i` 相同 |
| `positions_i=p_iT_train` | 来源训练位置 | 期望/长期口径 |
| `epochs_i=p_iT_train/N_i` | 平均重复轮数 | token 采样口径 |
| `p_iT_train<=CN_i` | epoch cap | `C` 是轮数 |
| `rho=T_small/T_large` | simulated epoching 下采样比 | 所有来源同比例；与 `q_i` 无关 |

单位：

- `1K=10^3=1,000`；
- `1M=10^6=1,000,000`；
- `1B=10^9=1,000,000,000`；
- `1T=10^12=1,000,000,000,000`。

这里是十进制数量，不是 KiB/MiB/GiB 字节单位。

## 26. 自测题（80 题）

> 本节 80 题全部有题型标签。互斥按题号行统计：29 道【手算】+1 道【手算+判断解释】+6 道【推导】+4 道【填表】+11 道【分类】+2 道【综合手算】=53 道明确操作题。其余为判断解释、设计与综合题；每题只归入一个标签。

### A. Transformation 与过滤（1–20）

1. 【判断解释】导航、正文段落、表格、扫描图片，哪些可直接视为线性正文？分别还需什么处理？
2. 【分类】把“字符是否保留”和“结构/关系是否保留”分成两类判断：“抽取器输出包含原 PDF 每个字符，所以无损”对吗？给反例。
3. 【设计】为双栏论文写出至少四项抽取检查。
4. 【判断解释】WARC、HTML、OCR、VLM、provenance 各在流水线中扮演什么角色？
5. 【手算】固定 extraction 标签：actual positive=gold 正文块，predicted positive=系统抽出块。gold 有 20 个正文块；系统抽出 18 块，其中 15 块是 gold 正文。列 TP/FP/FN 并算 precision、recall。
6. 【手算】沿用第5题 extraction 标签。另一抽取器输出 14 块，其中 13 块是 gold 正文。列 TP/FP/FN，算 P/R；两个抽取器谁 precision 高、谁 recall 高？
7. 【分类】把下列信息分成“P/R 能直接量到”和“P/R 不能直接量到”：正文块保留、表格关系、事实错误、OCR 幻觉、下游训练效果。
8. 【判断解释】`T`、`R`、`T'` 在过滤问题中分别是什么？
9. 【设计】目标正例全是长英文论文。列出三个 classifier 可能学到的捷径。
10. 【手算】固定 filtering-eval 标签：actual positive=按人工规则判为好，predicted positive=系统保留。100 篇中 40 篇真好；系统留 60 篇，其中 36 篇真好。算 TP、FP、FN、TN、precision、recall。
11. 【手算】阈值提高后留下 32 篇，其中 28 篇真好。重新算 P/R，并说明 trade-off。
12. 【分类】把 score=0.8 分到“可解释成80%真好”或“不可如此解释”，并写出 calibration 条件。
13. 【判断解释】KenLM 的 `p_T(x)` 与 fastText 的 `p(T|x)` 分别问什么？
14. 【判断解释】含 LaTeX 的 OpenWebMath 候选为何可以采用不同 threshold？这能证明它一定是数学吗？课程为何会出现 OpenMathText 这个名字？
15. 【手算】把 14.7B token 写成完整十进制整数。
16. 【推导】从 `P(X>u)=(1+u)^-a`、`u=1-s`、`a=9` 推出 Pareto 保留概率。
17. 【手算】`s=0` 时保留概率是多少？10,240 篇同分文档期望留多少篇？
18. 【手算】`s=0.5` 时用 `1.5^9≈38.443` 算保留概率；10,000 篇期望留多少？
19. 【手算】`s=0.9` 时用 `1.1^9≈2.35795` 算保留概率。
20. 【分类】区分“分数到保留决定的随机性”与“分数本身的群体偏差”：Pareto 随机保留修复了哪一个、没修复哪一个？

### B. 过滤规模、exact dedup 与 Jaccard（21–35）

21. 【手算】高分池 100M token、训练 800M token，只用该池是多少 epochs？
22. 【设计】列出过滤 ablation 必须固定的四项，以及至少一个应重复估计的不确定性。
23. 【分类】语言识别在 code-switching、方言、纯单语标准文本上分别可能遇到什么难点？
24. 【判断解释】课程 157M/100-WARC 图能否证明低质量数据训练久后总更好？
25. 【填表】对一组 near-duplicate 决策，填 item、match、action 三轴各一个选择。
26. 【手算】`n=10` 篇文档两两比较有多少对？写公式。
27. 【手算】`n=1,000` 时有多少对？为什么不是线性？
28. 【分类】SHA-256 与 MurmurHash 的主要目标差异是什么？
29. 【判断解释】MurmurHash 相同后为何要核原文？
30. 【填表】源码六个字符串中，exact dedup 至少会合并哪两个？为何 `Hello!` 不与 `hello` 合并？
31. 【判断解释】若先 lower-case 再删标点，第 30 题结果怎样变？有什么风险？
32. 【设计】把 exact dedup 写成 Map、Shuffle、Reduce 三步。
33. 【判断解释】从文档中删重复的三句话为何可能破坏 coherence？
34. 【手算】`A={1,2,3,4}`, `B={1,2,3,5}`：列交集、并集并算 Jaccard。
35. 【手算】字符 2-shingle 集合 `A={ab,bc,cd}`, `B={ab,bc,ce}` 的 Jaccard 是多少？

### C. MinHash 与 LSH（36–50）

36. 【推导】用并集五个元素的“第一名”事件证明第 34 题 MinHash match 概率为 3/5。
37. 【判断解释】为什么这个证明需要随机排列/minwise 假设？
38. 【手算】8 个 MinHash 中 6 个 match，估计 Jaccard 是多少？它是 exact 值吗？
39. 【手算】100 个 signature 若 61 个 match，`J_hat` 是多少？与真实 0.6 的绝对误差是多少？
40. 【填表】`n=12,b=3,r=4` 时，把 `h1..h12` 分 band。
41. 【推导】从单 hash match 概率 `s` 推到 `1-(1-s^r)^b`。
42. 【手算】`s=0.8,r=10`，固定 band match 概率是多少？
43. 【手算】承接第 42 题，`b=5` 时 candidate 概率是多少？
44. 【手算】`s=0` 与 `s=1` 时 LSH candidate 概率分别是多少？
45. 【判断解释】只增大 `r`，曲线向哪边移？为什么？
46. 【判断解释】只增大 `b`，曲线向哪边移？为什么？
47. 【手算】`s=0.9,b=10,r=10` 的候选概率约 0.9863。若 `r` 改 20，课程复算值约多少？
48. 【推导】`b=20,r=450` 时为什么启发式点满足 `s*^r=1/20`？
49. 【手算】在第 48 题点，候选概率 `1-(19/20)^20` 约多少？是否 0.5？
50. 【设计】LSH 给出候选后，写出避免 false positive 的下一步；false negative 为什么更难补？

### D. 数据混合（51–65）

51. 【手算】权重 `.3,.5,.2` 的和是多少？各从 1,000 个位置中期望得到多少？
52. 【手算】来源大小 100/300/600，proportional 权重是多少？
53. 【手算】同一来源按 `alpha=0` 的 power mixing 会得到什么？按 `alpha=1` 呢？
54. 【推导】解释为什么 `epochs_i=p_iT_train/N_i` 没有单位。
55. 【手算】完整复算 10T low、10B high、1T 训练、50/50 时两个 epoch 数。
56. 【判断解释】第 55 题的 50 epochs 是否表示模型“需要”50遍？
57. 【手算】第一问：batch size=12、sequence 抽样概率 $`q_{\mathrm{code}}=.25`$，长期每 batch 期望多少 code sequence？第二问：目标 token share 是 code/web 各0.5，平均长度100/400，用 §16.1 归一化式算 $`q_{\mathrm{code}},q_{\mathrm{web}}`$，并验证期望 token 各80。
58. 【推导】从 `epochs_i<=C` 推出 `p_iT_train<=CN_i`。
59. 【判断解释】为什么源码 `p_iT_train<=C` 有量纲问题？
60. 【手算】A=100 token、B=20、训练100、cap2；uniform 时各 epoch？怎样改为 60/40 满足 cap？
61. 【分类】vibes、uniform、proportional、power mixing 各自最明显的风险是什么？
62. 【填表】给 `T_train=1T`、books `N=20B,p=.1`、code `100B,.2`、web `2T,.7`，算 positions 与 epochs。
63. 【手算】小 run=10B、大 run=1T，simulated epoching 比例是多少？10T/10B 两来源缩成多大？
64. 【手算】在第 63 题缩小池上，小 run 50/50 的 epoch 数；验证是否与大 run 相同。
65. 【判断解释】为什么小 proxy regression 在训练样本上预测准，仍不保证找到的最优 mixture 可靠？

### E. 合成数据、端到端审计（66–80）

66. 【分类】环境、task/prompt、teacher response、scaffold 分别是什么？
67. 【分类】完全合成、半合成、真实 prompt 各举一例。
68. 【手算】课程引用的 OpenThoughts3 快照有 1.2M examples、每 prompt 16 responses，对应多少 prompts？
69. 【判断解释】为什么 benchmark 更强的模型未必是更好的 teacher？
70. 【设计】为 teacher 数据写四项独立验证。
71. 【分类】SWE-smith 为什么属于“真实环境 + 合成任务”？
72. 【手算+判断解释】无 execution feedback 的轨迹有什么收益与代价？另有 $`n=5,c=2,k=2`$，用无放回公式算 pass@2，并说明它与独立 $`1-(1-p)^k`$ 不是同一估计口径。
73. 【判断解释】为什么“已有 unit tests 全过”仍不等于 patch 完全正确？
74. 【设计】写出一条不包含攻击步骤的 future-information 防泄漏原则。
75. 【分类】license、PII、benchmark contamination、dependency version 分别属于哪类审计风险？
76. 【设计】若抽取器在阿拉伯语上 recall 35%、总体 90%，你会怎样处理而不是直接上线？
77. 【综合手算】过滤后来源 A=40B、B=10B；计划训练100B，权重 .6/.4。算 epochs。若 cap=2，哪个超限？B 最大权重是多少？余量给 A 后新权重是多少？
78. 【综合手算】两个文档精确 Jaccard `s=.8`；LSH `b=5,r=10`。10,000 对这样的候选对，期望多少进入精算？若精算阈值 .85，它们最终会删吗？
79. 【综合设计】按顺序列出一个网页从 WARC 到训练 batch 的至少八个可审计步骤。
80. 【综合判断】“我们用了高质量 classifier、MinHash LSH、RegMix 和强 teacher，所以数据集客观正确、安全、合法且最优。”逐项指出至少五个不成立的跳跃。

## 27. 自测答案

### A. Transformation 与过滤答案（1–20）

1. 导航是 boilerplate 候选，通常删除但要按用途判断；正文段落可线性化；表格必须保留行列关系；扫描图片需 OCR/VLM，且要核阅读顺序与识别错。

2. 不对。双栏 PDF 即使每个字符都在，若按“左一行→右一行→左二行”交错，句子关系已错；表格字符全在但列归属丢失也属有损。

3. 至少检查：左右栏阅读顺序、标题/正文层级、脚注位置、公式与编号、图注与图片关联、页眉页脚是否误混。答任四项即可。

4. WARC 保存抓取响应；HTML 是网页结构；OCR 把图片变文字；VLM 可帮助理解布局/图文；provenance 记录 URL、时间、工具版本等来源链。

5. 按题干 extraction 标签，actual positive 是 gold 正文、predicted positive 是抽出块。于是 `TP=15`；输出18，所以 `FP=18-15=3`；gold20，所以 `FN=20-15=5`。没有给全部非正文块数，TN 无法从这些数推出，但 P/R 不需要 TN。

   $`P=15/18=0.8333=83.33\%,`$

   $`R=15/20=0.75=75\%.`$

6. 沿用相同 extraction 标签：`TP=13,FP=14-13=1,FN=20-13=7`。

   $`P=13/14\approx92.86\%,`$

   $`R=13/20=65\%.`$

   第二个 precision 更高；第 5 题的 recall 更高。

7. P/R 只检查人工定义的块是否保留，不直接量到表格关系、事实错误、OCR 幻觉或训练效果。要固定其他因素做下游 ablation。

8. `T` 是小而高质量的目标示例；`R` 是巨大原料池；`T'` 是过滤器从 `R` 找出的、预计类似 `T` 的新子集。

9. 可能捷径：文档长度、英文/非英文、引用符号数量、`.edu` 域名、排版、年份。它们能区分收集来源，却不等于内容质量。

10. 按 filtering-eval 标签，actual positive 是人工判好、predicted positive 是保留。`TP=36`；`FP=60-36=24`；`FN=40-36=4`；全部坏文档 60 篇，所以 `TN=60-24=36`。

    $`P=36/(36+24)=36/60=60\%,`$

    $`R=36/(36+4)=36/40=90\%.`$

11. `TP=28,FP=32-28=4,FN=40-28=12`。

    $`P=28/32=87.5\%,`$

    $`R=28/40=70\%.`$

    阈值提高后留下的更纯，但漏掉更多好文档。

12. 不意味着。只有在相似样本上做过 calibration，且“0.8 桶里长期约80%真好”，才可作该概率解释。

13. `p_T(x)` 问“目标语言模型觉得文本 x 多像/多常见”；`p(T|x)` 问“给定 x，分类标签是 target 的概率”。训练目标和归一化方向不同。

14. LaTeX 是额外数学信号，所以可对含 LaTeX 候选放低 classifier threshold；无 LaTeX 时要求更高。它不能证明必是数学，因为模板、价格、坏 OCR 也会含符号。课程源码/口述的 OpenMathText 是误名；论文 arXiv:2310.06786 正式名为 OpenWebMath。

15. `14.7B=14.7*1,000,000,000=14,700,000,000 token`。

16. 令 `u=1-s,a=9`：

    $`P(keep|s)=P(X>1-s)=[1+(1-s)]^{-9}=(2-s)^{-9}.`$

17. `s=0`：

    $`P=2^{-9}=1/512=0.001953125.`$

    $`10{,}240/512=20,`$

    所以期望留 20 篇，不保证每次刚好 20。

18. `P=1/38.443≈0.026012=2.6012%`；

    $`10{,}000*0.026012≈260.12,`$

    期望约 260 篇。

19. `P=1/2.35795≈0.42410=42.410%`。

20. 随机规则只改变分数到保留概率的映射。若 classifier 系统性压低某语言分数，该语言仍会少保留；随机性没有修复标签与特征偏差。

### B. 过滤规模、exact dedup 与 Jaccard 答案（21–35）

21. `800M/100M=8 epochs`。

22. 固定原始抓取、过滤后 token 预算、模型结构、optimizer/learning rate、训练 steps、tokenizer、评估协议和尽量相同 seed；至少重复训练或 bootstrap 评估，估计置信范围。

23. code-switching 需要多标签/片段级判断；方言可能因正例少而误判成别的语言或低质量；标准单语通常最容易，但仍会被姓名、代码等噪声干扰。

24. 不能。它是 157M 模型、100 WARC 小池的 preliminary experiment；只说明该设置里过滤强度与训练长度交互。

25. 例：item=文档的5-token shingles；match=Jaccard>=0.95；action=同簇只留来源许可最清楚的一篇。三项都必须写。

26. `10*9/2=45` 对。

27. `1000*999/2=499,500` 对。输入增约 100 倍（10→1000），比较从45增到约50万，因主项是 `n^2/2`，不是 `n`。

28. SHA-256 重点是难以构造碰撞，适合完整性/安全；MurmurHash 重点是快，适合哈希表和候选索引，不提供密码学碰撞抗性。

29. 不同原文可能碰撞成同 hash。核原文可避免因为碰撞误删。

30. 两个完全相同的 `"hello"` 会合并。`"Hello!"` 大小写和标点不同，所以原始字符串不 exact equal。

31. `"Hello!"` 会规范化成 `"hello"`，因此三个版本可能合并。风险是把标点/大小写有语义的代码、名称或语言错误合并。

32. Map：每篇输出 `(hash,docID)`；Shuffle：相同 hash 聚到同一 reducer；Reduce：核原文并选 canonical 一份，记录其他来源。

33. 前文可能说“下面三点解释原因”，三句被删后直接跳结论；指代和论证链断裂，所以整体文档不连贯。

34. `A∩B={1,2,3}` 大小3；`A∪B={1,2,3,4,5}` 大小5；`J=3/5=0.6`。

35. 交集 `{ab,bc}` 大小2；并集 `{ab,bc,cd,ce}` 大小4；`J=2/4=0.5`。

### C. MinHash 与 LSH 答案（36–50）

36. 并集五个元素各有 `1/5` 概率排第一。1、2、3 属于交集，排第一时两个集合 MinHash 相同；4 或5排第一时不同。因此 match 概率 `3*(1/5)=3/5`。

37. 若 hash 偏爱某些元素，它们不再各有 `1/5` 概率排第一，事件计数就不成立。MinHash 需要随机排列或足够接近 minwise independent 的 hash family。

38. `J_hat=6/8=0.75`。它只是8次 Bernoulli match 的样本平均，不是从原集合算出的 exact Jaccard。

39. `J_hat=61/100=0.61`；绝对误差 `|0.61-0.60|=0.01`。

40. `band1={h1,h2,h3,h4}`；`band2={h5,h6,h7,h8}`；`band3={h9,h10,h11,h12}`。

41. 单 hash match 概率 `s`；固定 band 的 `r` 个独立 hash 全 match 是 `s^r`；该 band 不 match 是 `1-s^r`；`b` 个 band 全不 match 是 `(1-s^r)^b`；取反得到至少一 band match：`1-(1-s^r)^b`。

42. `0.8^10=0.1073741824`，约 10.74%。

43. 固定 band 不 match `=1-0.1073741824=0.8926258176`；五个都不 match `=0.8926258176^5≈0.5666921796`；candidate `=1-0.5666921796≈0.4333078204=43.33%`。

44. `s=0`：`1-(1-0^r)^b=1-1=0`。`s=1`：`1-(1-1^r)^b=1-0=1`。

45. 向右移、更严格。因为 `0<s<1` 时增大指数 `r` 会让 `s^r` 变小，单 band 更难全匹配。

46. 向左移、更宽松。因为 band 数变多，任意一个成功的机会增多，`(1-s^r)^b` 变小，候选概率变大。

47. `r=20,b=10` 时约 `0.7264487568=72.64%`，低于 `r=10` 的 98.63%。

48. 定义 `s*=(1/b)^(1/r)`。两边取 `r` 次幂：

    $`(s_*)^r=[(1/b)^{1/r}]^r=1/b=1/20.`$

49. `1-(19/20)^20≈0.641514=64.1514%`，不是 0.5，因此该启发式点不是 50% 硬阈值。

50. 对候选取原 shingle 集合，精算 Jaccard，只有达到真实阈值才合并/删除。false negative 没进入候选，后处理看不到；要从候选生成阶段调 `b,r`、增 hash 或换索引来补。

### D. 数据混合答案（51–65）

51. `.3+.5+.2=1.0`。1,000 位置中期望 Wikipedia `300`、CC `500`、GitHub `200`。

52. 总大小 `100+300+600=1000`；权重分别 `100/1000=.1`、`300/1000=.3`、`600/1000=.6`。

53. `alpha=0` 时每个正数 `N_i^0=1`，normalize 后 uniform。`alpha=1` 时 `N_i^1=N_i`，normalize 后 proportional。

54. `p_i` 无单位；`T_train` 单位 token，所以分子是 token；`N_i` 也是 token；`token/token=1`，因此 epoch 是无量纲轮数。

55. 低来源 positions：`.5*1T=.5T`；

    $`e_{low}=.5T/10T=.05.`$

    高来源 positions 同为 `.5T=500B`；

    $`e_{high}=500B/10B=50.`$

56. 不是。它只表示 mixture 会平均暴露 50 次；讲者指出最好是浪费，最坏会过拟合。应调权重或用 cap。

57. 第一问：$`12\times0.25=3`$，所以长期每 batch 期望3条 code sequence；单 batch 不必恰好3。

    第二问：未归一化权重是 $`0.5/100=0.005`$ 与 $`0.5/400=0.00125`$，和为0.00625：

    $`q_{\mathrm{code}}=0.005/0.00625=0.8,\qquad q_{\mathrm{web}}=0.00125/0.00625=0.2.`$

    期望 token 分量是 $`0.8\times100=80`$ 与 $`0.2\times400=80`$，所以 token share 正好50%/50%。这也证明 $`p_i`$ 与 $`q_i`$ 在不等长时不能直接相等。

58. 从：

    $`e_i=\frac{p_iT_{train}}{N_i}\le C`$

    两边乘正数 `N_i`：

    $`p_iT_{train}\le CN_i.`$

59. `p_iT_train` 的单位是 token，`C` 的单位是 epoch（无量纲），不能比较。右边要乘 `N_i token`。

60. uniform 分配 50/50：`e_A=50/100=.5`；`e_B=50/20=2.5`，B 超 cap2。B 最多 `2*20=40` token；余下60给A：`e_A=60/100=.6`、`e_B=40/20=2`，权重 `.6/.4`。

61. vibes：主观且难复现；uniform：小来源重复；proportional：大而低质来源支配；power：`alpha` 仍是需验证的超参数，且只看大小不看质量。

62. `T=1T=1000B`：

    - books positions `.1*1000B=100B`；epochs `100/20=5`；
    - code positions `.2*1000B=200B`；epochs `200/100=2`；
    - web positions `.7*1T=.7T`；epochs `.7T/2T=.35`。

63. 缩放比例 $`\rho=10B/1T=10B/1000B=.01`$。这里 $`\rho`$ 不是 §16 的 sequence 概率 $`q_i`$。于是 `10T*.01=.1T=100B`；`10B*.01=.1B=100M`。

64. 小 run 10B 的一半是5B。低来源 `e=5B/100B=.05`；高来源 `e=5B/.1B=50`，与大 run 相同。

65. regression 可能只在已采 mixture 周围准确；optimizer 会走到边缘/外推区。且小模型、少 token 的最优权重可能随规模、epoch、optimizer 改变，所以需留出 mixture 验证和大规模确认。

### E. 合成数据、端到端审计答案（66–80）

66. environment 是任务运行世界；task/prompt 是要求；teacher response 是示范答案/轨迹；scaffold 是把模型接到工具、循环和状态管理的外围程序。

67. 完全合成：模型造数学题并作答；半合成：在真实 repo 上生成 bug task；真实 prompt：使用真实 PR 描述或经同意的用户请求。

68. `1.2M=1,200,000`；

    $`1{,}200{,}000/16=75{,}000\text{ prompts}.`$

69. teacher 数据要求正确、步骤可学、风格匹配、成本可承受；benchmark 分高只量特定协议，不量这些全部属性。

70. 例：独立答案/测试 verifier；随机人工审计；检查遵守工具/执行规则；benchmark overlap；PII/secrets scan；teacher 多样性与错误 slice。答四项即可。

71. repository、依赖和 tests 来自真实项目；任务/bug 由程序或 LM 生成，所以是 semi-synthetic。

72. 收益：无需为每个 repo 安装依赖，便宜、可扩到更多 PR。代价：无法用运行结果验证语义，错误轨迹更难发现，学习信号较弱。

    总2子集 $`\binom52=10`$；3份失败候选的全失败2子集 $`\binom32=3`$：

    $`\widehat{\mathrm{pass@}2}=1-3/10=0.7.`$

    这是从已有5份候选不重复选2份的估计；$`1-(1-p)^2`$ 则假设固定题目下每次按同一分布独立采样、成功率为 $`p`$，契约不同。若用 beam/shared pool 或自适应 retry，候选可能相关；跨题成功率也不同。benchmark 应逐题算 pass@k 再平均。若 $`n-c<k`$，全失败组合数按0，估计为1。

73. tests 只覆盖作者写到的案例；可能漏边界、性能、安全、未测试接口，或测试本身错误。通过是证据，不是完整正确性证明。

74. 例：构建任务快照时隔离任务时间之后的 commits 和网络来源；固定允许工具，记录全部访问并审计。这里只给防泄漏原则。

75. license 是权利/许可风险；PII 是隐私风险；benchmark contamination 是评估有效性风险；dependency version 是可复现与执行环境风险。

76. 暂停统一阈值上线；扩充阿拉伯语 gold 与正例；查编码/RTL/方言 slice；调整专用模型或阈值；比较误删成本；做下游 ablation，并持续报分组指标。

77. A positions `.6*100B=60B`，epochs `60/40=1.5`。B positions `.4*100B=40B`，epochs `40/10=4`，超 cap2。B 最大 positions `2*10B=20B`，最大权重 `20B/100B=.2`。余下80B给A，新权重 `.8/.2`；A epochs `80/40=2`，B epochs `20/10=2`。

78. 候选概率约 `.4333078204`；

    $`10{,}000*.4333078204\approx4{,}333.08,`$

    期望约4,333对进入精算。它们 exact `J=.8<.85`，所以按该 action 不应判重复删除。

79. 一种合格顺序：WARC 取 body → 解码/HTML 抽取 → 保存 provenance → 语言/规则过滤 → model score 与阈值 → normalize → exact dedup → MinHash/LSH 候选 → exact Jaccard → cross-source/test decontam → 重算 `N_i` → mixture/epoch cap → sequence 采样进 batch。超过八步即可。

80. 至少五个跳跃：

    1. “高质量”由 target 定义，不客观；
    2. classifier 有校准、群体 FP/FN 和阈值问题；
    3. MinHash/LSH 只是候选概率，有 FP/FN；
    4. 去重不决定许可、隐私或事实正确；
    5. RegMix 可能 eval overfit，且小→大不保证迁移；
    6. 强 teacher 未必可学、无幻觉或获授权；
    7. unit tests 不完全；
    8. 合法性与访问/过滤算法是独立层；
    9. “最优”依训练规模、模型、任务与时间变化。

<a id="lecture14-video-nav"></a>

## 28. 视频导航（人工字幕真实 cue）

> 所有链接的显示时间都等于 URL 的 `t=` 秒数；全文不重复使用同一秒点。导航从开场覆盖到 84:13 总结。

### 28.1 开场、Transformation、Filtering 框架

| 时间 | 内容 | 对应正文 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=5sxHosTLPF8&t=5s) | 开场 | §0 |
| [00:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=54s) | 四段 pipeline | §2 |
| [01:39](https://www.youtube.com/watch?v=5sxHosTLPF8&t=99s) | GitHub 是目录结构 | §3.4 |
| [02:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=145s) | 导航有时也有学习价值 | §3.1 |
| [03:16](https://www.youtube.com/watch?v=5sxHosTLPF8&t=196s) | 规则抽取器为何快 | §4.3 |
| [04:05](https://www.youtube.com/watch?v=5sxHosTLPF8&t=245s) | Resiliparse/Trafilatura 比较 | §4.3 |
| [04:51](https://www.youtube.com/watch?v=5sxHosTLPF8&t=291s) | Common Crawl 中的 PDF | §3.3 |
| [05:36](https://www.youtube.com/watch?v=5sxHosTLPF8&t=336s) | OCR 与 VLM | §3.3 |
| [06:24](https://www.youtube.com/watch?v=5sxHosTLPF8&t=384s) | HTML tags 与 PDF layout | §4.2 |
| [07:09](https://www.youtube.com/watch?v=5sxHosTLPF8&t=429s) | 小 target、大 raw | §6.1 |
| [07:56](https://www.youtube.com/watch?v=5sxHosTLPF8&t=476s) | quality filtering | §6 |
| [08:44](https://www.youtube.com/watch?v=5sxHosTLPF8&t=524s) | 常只留下很小比例 | §6.2 |
| [09:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=572s) | 过滤器必须便宜 | §7.1 |
| [10:18](https://www.youtube.com/watch?v=5sxHosTLPF8&t=618s) | fastText | §7.2 |
| [11:04](https://www.youtube.com/watch?v=5sxHosTLPF8&t=664s) | compute-rich/poor 语境 | §7.3 |
| [11:50](https://www.youtube.com/watch?v=5sxHosTLPF8&t=710s) | 176 languages | §9.1 |
| [12:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=755s) | 语言阈值是 heuristic | §9.1 |
| [13:20](https://www.youtube.com/watch?v=5sxHosTLPF8&t=800s) | ProofPile KenLM | §9.2 |
| [14:07](https://www.youtube.com/watch?v=5sxHosTLPF8&t=847s) | OpenWebMath 的 20x 对照（课程误名见正文） | §9.2 |
| [14:56](https://www.youtube.com/watch?v=5sxHosTLPF8&t=896s) | LLaMA positives 是被 Wiki 引用页 | §9.3 |
| [15:44](https://www.youtube.com/watch?v=5sxHosTLPF8&t=944s) | GPT-4 标 100K target | §9.4 |
| [16:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=992s) | toxicity 也是同一框架 | §9.5 |
| [17:20](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1040s) | classifier 扫 Common Crawl | §6–§9 |
| [18:07](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1087s) | 短训练偏高质量 | §10 |
| [18:56](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1136s) | 图中 DCLM 曲线 | §10.2 |
| [19:42](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1182s) | 弱过滤后期下降 | §10.2 |
| [20:27](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1227s) | 学生问置信区间 | §10.3 |
| [21:14](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1274s) | 高质量+长训练问答 | §10.1 |
| [21:59](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1319s) | filtering 总结 | §10.4 |
| [22:46](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1366s) | teacher 先标、便宜模型外推 | §9.4 |

### 28.2 Dedup、MinHash、LSH

| 时间 | 内容 | 对应正文 |
|---|---|---|
| [23:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1412s) | mirror 产生 exact duplicate | §11 |
| [24:17](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1457s) | near-duplicate examples | §11 |
| [25:02](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1502s) | LM1B 标点差异 | §11 |
| [25:49](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1549s) | C4 重复产品描述 | §11.1 |
| [26:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1595s) | memorization 风险 | §11 |
| [27:20](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1640s) | item/match/action | §11.1 |
| [28:05](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1685s) | filtering 与 dedup 复杂度差别 | §11.2 |
| [28:52](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1732s) | hash 映射 | §12.1 |
| [29:38](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1778s) | exact dedup 开始 | §12.2 |
| [30:24](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1824s) | MapReduce 风格 | §12.4 |
| [31:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1870s) | 三句 span 删除 | §12.4 |
| [31:57](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1917s) | Jaccard 交集/并集 | §13.2 |
| [32:45](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1965s) | MinHash 开始 | §13.3 |
| [33:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2012s) | 想要“受控碰撞” | §13.3 |
| [34:18](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2058s) | 随机排列心智图 | §13.3 |
| [35:07](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2107s) | 交集元素排第一 | §13.3 |
| [35:55](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2155s) | min 相同事件 | §13.3 |
| [36:41](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2201s) | 100 个 hash 的代码检查 | §13.4 |
| [37:29](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2249s) | 单碰撞还不能判阈值 | §14 |
| [38:17](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2297s) | LSH 目标 | §14.1 |
| [39:05](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2345s) | bands 与 rows | §14.1 |
| [39:55](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2395s) | “某 band 全同” | §14.1 |
| [40:41](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2441s) | AND-OR 结构 | §14.1 |
| [41:34](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2494s) | 固定 band 概率 | §14.2 |
| [42:19](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2539s) | 全 band 不匹配 | §14.2 |
| [43:09](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2589s) | 曲线端点 | §14.4 |
| [43:57](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2637s) | 参数实验 | §14.4 |
| [44:48](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2688s) | false positives | §14.6 |
| [45:36](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2736s) | 增大 r 更难 | §14.4 |
| [46:23](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2783s) | 更陡但更贵 | §14.4 |
| [47:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2830s) | 启发式 threshold | §14.5 |
| [47:55](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2875s) | 中心概率约 0.64 | §14.5 |
| [49:01](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2941s) | 需要跨 dataset dedup | §15 |

### 28.3 Mixing、合成数据与收尾

| 时间 | 内容 | 对应正文 |
|---|---|---|
| [49:53](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2993s) | Marin 多来源 | §16 |
| [50:38](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3038s) | The Pile mixture | §16 |
| [51:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3085s) | vibes | §16.2 |
| [52:11](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3131s) | proportional mixing | §16.2 |
| [52:57](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3177s) | 不可直接比较的来源 | §16.2 |
| [53:44](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3224s) | 10T/10B 例开始 | §16.4 |
| [54:31](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3271s) | low source epoch | §16.4 |
| [55:21](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3321s) | high source 500B positions | §16.4 |
| [56:14](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3374s) | 学生问为什么50 epochs | §16.4 |
| [57:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3420s) | 看清实际 epochs | §16.4 |
| [57:50](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3470s) | sequence 级抽来源 | §16.5 |
| [58:36](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3516s) | UniMax 多语言背景 | §17 |
| [59:21](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3561s) | 20 epoch cap 例 | §17.3 |
| [60:08](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3608s) | 50个来源的难题 | §18 |
| [60:55](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3655s) | proxy 模型 | §18.1 |
| [61:44](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3704s) | 优化预测 loss | §18.1 |
| [62:29](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3749s) | Dirichlet | §18.2 |
| [63:15](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3795s) | eval overfit | §18.4 |
| [64:01](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3841s) | 方法表各列 | §18.1 |
| [64:53](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3893s) | 两个 leap of faith | §18.4 |
| [65:38](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3938s) | optimizer 走向覆盖外 | §18.4 |
| [66:23](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3983s) | scale-dependent 转场 | §18.5 |
| [67:09](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4029s) | 大 run 重复高质量池 | §18.5 |
| [67:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4074s) | 与 muP 的类比 | §18.5 |
| [68:39](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4119s) | 10B/1T 比例 | §18.5 |
| [69:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4165s) | 模拟 data scarcity | §18.5 |
| [70:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4210s) | 小到大要谨慎 | §18.6 |
| [70:57](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4257s) | 下采样太小问答 | §18.6 |
| [71:43](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4303s) | 域内也可混合 | §18 |
| [72:28](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4348s) | topic×quality 网格 | §18 |
| [73:13](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4393s) | post-training 转场 | §19 |
| [74:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4440s) | 定义 tasks/prompts | §19 |
| [74:47](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4487s) | teacher 产生 responses | §19 |
| [75:34](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4534s) | OpenThoughts3 快照来源 | §20 |
| [76:22](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4582s) | code review 来源 | §20 |
| [77:08](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4628s) | answer filtering 观察 | §20.2 |
| [77:53](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4673s) | agentic coding data | §21 |
| [78:39](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4719s) | SWE-smith 验证 tasks | §21.1 |
| [79:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4765s) | repo dependency 困难 | §21.2 |
| [80:14](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4814s) | no-execution 仍有能力 | §21.2 |
| [81:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4870s) | teacher filtering | §21.3 |
| [81:56](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4916s) | SWE-rebench repo 流程 | §21.4 |
| [82:42](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4962s) | executable/non-executable | §21.4 |
| [83:27](https://www.youtube.com/watch?v=5sxHosTLPF8&t=5007s) | 三类 prompt 来源 | §19.1 |
| [84:13](https://www.youtube.com/watch?v=5sxHosTLPF8&t=5053s) | 数据工作很脏、很具体 | §22–§24 |

<a id="lecture14-source-coverage"></a>

## 29. 官方源码 1–464 行覆盖索引

### 29.1 固定版本

本讲使用 `lecture_14.py`：464 个物理行、24,633 bytes、SHA256：

```text
53E2B997DCE030AF6DAA519BE7F297AFD2CCF23E2152CE8DD776D85CD26E36B7
```

### 29.2 连续区间：无 gap、无 overlap

| 源码行 | 内容 | 笔记映射 |
|---:|---|---|
| 1–10 | imports：NumPy、mmh3、edtrace、Lecture13/The Pile references | §0、§12、§31 |
| 11–38 | `main()`：上讲回顾、四段 pipeline、总结 | §1–§2、§22、§32 |
| 39–59 | `transformation()`：HTML/PDF、抽取器、FinePDFs | §3–§5 |
| 60–144 | `filtering()`：T/R/T'、两类模型、语言/数学/GPT-3/LLaMA/phi-1/toxicity/scale | §6–§10 |
| 145–177 | `deduplication()`：exact/near、案例、三轴、子函数调用 | §11、§15 |
| 178–190 | `hash_functions()`：collision、安全与速度 | §12.1 |
| 191–216 | `exact_deduplication()`：sort/groupby、C4 三句 span | §12.2–§12.4 |
| 217–267 | `jaccard_minhash()`：集合、Jaccard、MinHash 与 100 seeds | §13 |
| 268–323 | `locality_sensitive_hashing()`：bands/rows、概率、真实参数 | §14 |
| 324–330 | `billion()`、`trillion()` 数量 helper | §16.4、§25 |
| 331–408 | `data_mixing()`：Marin/Pile、epoch、UniMax、RegMix、simulated epoching | §16–§18 |
| 409–462 | `post_training_data()`：OpenThoughts、SWE-smith/Zero/Hero/rebench/12M | §19–§21 |
| 463–464 | Python main guard | 本节版本说明 |

### 29.3 函数与运行边界

源码 12 个函数全部有正文映射：`main`、`transformation`、`filtering`、`deduplication`、`hash_functions`、`exact_deduplication`、`jaccard_minhash`、`locality_sensitive_hashing`、`billion`、`trillion`、`data_mixing`、`post_training_data`。

`edtrace` 负责把 executable lecture 的 `text/image/link` 渲染出来；`mmh3` 提供 MurmurHash；NumPy 提供 Pareto 随机数。本文独立复算了 Pareto、Jaccard、MinHash 事件、LSH、epoch/cap 等数值，但没有伪称在本机把带课程专用 `edtrace` 的整份讲义作为应用运行。源码的 `assert abs(estimated_jaccard-jaccard)<.01` 是固定 100 seeds 的样例，不是定理测试。

## 30. 图片覆盖：13 张本地 + 5 张远程

### 30.1 核验方法

- 13 张仓库本地图全部用原分辨率打开；
- 5 张远程图先按源码 URL 下载到 `work/lecture14_inspection/remote_images/`，再按原分辨率打开；
- 对表格读列名/数值，对曲线读横纵轴/图例，对流程图读箭头；
- 图片只支持其具体论文/课程设置，不被改写成普遍因果规律。

### 30.2 逐图视觉记录

| 源码行 | 资产 | 分辨率 | 实际看到的语义 | 笔记 |
|---:|---|---:|---|---|
| 49 | `dclm-wet.png` | 486×180 | Resiliparse/Trafilatura/WET 在 CORE/EXTENDED 的 3×2 数表 | §4.3 |
| 52 | FinePDF remote WebP | 1408×768 | 左为 PDF objects/content stream，右为标题、双栏、图像/图注；强调 PDF 缺显式语义 | §3.3–§4.2 |
| 63 | `raw-target-schema.png` | 1651×861 | 大 raw R、小 target T、R 内筛出的 T' | §6.1 |
| 138 | `data-filtering-scale.png` | 1930×1094 | 157M、100 WARC；loss 对 log tokens；各数据的一-epoch 虚线 | §10.2 |
| 152 | formulaic remote PNG | 1188×354 | Wiki-40B/LM1B/C4 近重复文本，只改实体或标点 | §11 |
| 299 | LSH remote 1 | 1280×720 | similarity 横轴、candidate 0/1 点、S形候选概率 | §14.4 |
| 310 | LSH remote 2 | 1280×720 | 不同 b 的多条 S 曲线；b 越大越左 | §14.4 |
| 335 | `marin-token-viewer.png` | 2335×834 | 多个 web/multilingual/math/code 来源的 token(B) 横条图 | §16 |
| 338 | The Pile remote PNG | 1022×775 | component、Raw Size、Weight、Epochs、Effective Size 表；总 825.18GiB/1254.20GiB | §16 |
| 374 | `regmix.png` | 1265×788 | proxy mixtures→loss→regression surface→predicted best→large run | §18.1 |
| 379 | `data-mixing-methods.png` | 1311×738 | 多种 mixing 方法的 proxy size、swarm、regression、repetition constraint 比较 | §18.4 |
| 418 | `openthoughts-sources.png` | 804×722 | 多个 code question sources 及各自数量/说明 | §20 |
| 423 | `openthoughts-pipeline.png` | 1330×672 | source→filter→dedup→sample→75K prompts→×16→1.2M | §20.1 |
| 426 | `swe-smith.png` | 1372×405 | real repo→environment creation→四类 task generation→verified instances | §21.1 |
| 434 | `swezero-noexec.png` | 947×299 | 多模型 execution/no-execution 的 SWE-bench V/M 数表 | §21.2 |
| 439 | `swezero-prompt.png` | 952×823 | 标准 OpenHands 与 SWE-Zero execution-free prompt 对照 | §21.2–§21.3 |
| 442 | `swezero-results.png` | 1303×893 | model size 横轴、verified resolve rate 纵轴；SWE-Zero→Hero 箭头 | §21.3 |
| 448 | `swe-rebench.png` | 870×472 | GitHub+GHArchive→环境安装→LLM labeling→21K+ samples | §21.4 |

## 31. 来源、版本与证据边界

### 31.1 课程主来源

- [固定 commit 的 `lecture_14.py`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_14.py)。
- [Stanford Online 视频](https://www.youtube.com/watch?v=5sxHosTLPF8)，人工英文 `en-US` 字幕 1,448 cues，00:05–84:36。
- 源码直接引用的 18 张图，视觉核验见 §30。

课程中的阈值、规模、模型名、曲线、当天发布数据均按 2026 讲义快照理解；除非一手材料另有支持，不当成当前 leaderboard 或行业定律。

### 31.2 Transformation 与 filtering 一手来源

- [DCLM](https://arxiv.org/abs/2406.11794)：抽取器/数据选择实验的具体设置。
- [FinePDFs 官方 Hugging Face 项目说明](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog)：PDF 来源、重抓与抽取流程。
- [Data selection survey](https://arxiv.org/abs/2402.16827)：课程引用的研究综述，用于术语导航。
- [fastText 官方 language identification](https://fasttext.cc/docs/en/language-identification.html)：176 种语言模型范围。
- [OpenWebMath](https://arxiv.org/abs/2310.06786)（课程误称 OpenMathText）、[GPT-3](https://arxiv.org/abs/2005.14165)、[LLaMA](https://arxiv.org/abs/2302.13971)、[phi-1](https://arxiv.org/abs/2306.11644)、[Dolma](https://arxiv.org/abs/2402.00159)。
- [NumPy `Generator.pareto` 官方文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.pareto.html)：Lomax/Pareto II 密度与 API 口径。

论文支持其作者报告的 pipeline 和实验；不自动证明 classifier 是跨语言“客观质量函数”。Kaggle/Jigsaw 页面是课程案例来源，不等同对全部 toxicity 语境的权威定义。

### 31.3 Dedup 与 mixing 一手来源

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)及[作者公开代码](https://github.com/google-research/deduplicate-text-datasets)。
- [Mining of Massive Datasets：LSH 章节](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf)：MinHash/LSH 算法来源。
- [T5/C4](https://arxiv.org/abs/1910.10683)、[The Pile](https://arxiv.org/abs/2101.00027)。
- [UniMax](https://arxiv.org/abs/2304.09151)：通过限制 repeats 平衡多语言来源。
- [RegMix](https://arxiv.org/abs/2407.01492)、[simulated epoching 论文](https://arxiv.org/abs/2501.11747)、[OlMix 课程引用版本](https://arxiv.org/abs/2602.12237)。

源码 UniMax 行 `p(s)*num_training_tokens<=C` 缺 `N_s`，本文按 epoch 定义纠正为 `p_iT_train<=CN_i`。这是讲义勘误，不是改写 UniMax 论文。

### 31.4 Post-training 与 agentic code 一手来源

- [OpenThoughts3](https://arxiv.org/abs/2506.04178)。
- [SWE-smith](https://arxiv.org/abs/2504.21798)。
- [SWE-Zero](https://arxiv.org/abs/2604.01496)。
- [SWE-rebench](https://arxiv.org/abs/2505.20411)。
- [SWE-ZERO-12M trajectories 数据卡](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories)。

这些材料能支持作者公开的数据构造和实验，不能证明每个上游 PR 的许可、隐私、测试完整性或 teacher 训练来源。课程在 82 分钟后介绍“当日发布”材料，特别要锁定论文/数据卡 revision。

### 31.5 课程、视频与教学补充怎样分界

- 讲义/字幕直接出现的算法、案例、数字标【课程】或【视频补充】；
- precision/recall、Pareto survival、LSH、epoch/cap 的逐行推导是【补充解释】，结果与课程代码独立复算；
- slice audit、provenance card、决策树是工程教学补充，不伪装成某论文原样流程；
- 对“更好”“高质量”“strong teacher”的结论都限制在给定模型、数据、指标和时间。

## 32. 学完后的能力清单

现在你应该能：

- 把 HTML、PDF、repo 到文本的有损步骤画出来，并设计 gold 与下游 ablation；
- 从 confusion matrix 手算 precision/recall，解释 threshold 与群体偏差；
- 从 NumPy Pareto 代码推出 `(2-s)^-9`，而不是把 score 当保留率；
- 先声明 item/match/action，再设计 exact/near dedup；
- 从五个排列事件解释 MinHash，从 AND-OR 推出 `1-(1-s^r)^b`；
- 解释 heuristic LSH threshold 为什么不是硬 50% 点；
- 用 `p_iT_train/N_i` 复算每来源 epochs，发现盲目 50/50 的 50-epoch 陷阱；
- 修正 UniMax 讲义量纲，解释 RegMix 与 simulated epoching 的两个迁移风险；
- 把 synthetic dataset 拆成 environment、prompt、teacher、scaffold、verifier；
- 对 agentic coding data 检查执行规则、测试弱点、许可、PII、污染和版本；
- 用 §22–§23 的清单交付一个可追踪、可审计、可撤回的数据版本。

最后记住讲者在结尾的提醒：数据工作不是漂亮公式自动产出好数据。它需要反复看具体失败样本、记录每个版本，并承认算法没有回答的价值、法律和治理问题。
