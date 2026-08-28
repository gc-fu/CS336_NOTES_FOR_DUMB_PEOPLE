# CS336 Lecture 1：课程全景与 Tokenization

> Stanford CS336: Language Modeling from Scratch, Spring 2026  
> 讲师：Percy Liang  
> 视频：[Lecture 1: Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk)（79:15）  
> 官方讲义：[lecture_01.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)（667 行 executable lecture）

> **资料形式说明：**官方没有为 Lecture 1 发布静态 `lecture_01.pdf`。这一讲采用 executable lecture（可执行讲义）：老师运行 Python 程序，程序一边显示文字、图片和代码，一边展示变量如何变化。因此，本笔记以官方 `lecture_01.py` 作为讲义源，并用官方 YouTube 的人工英文字幕补齐老师口头讲解。

这不是逐字字幕翻译，而是一份可以替代视频学习的重构讲义。它保留课程的教学顺序、原始数字、代码例子和口头补充，同时补齐零基础读者需要的定义、推导和手算。

为了分清来源，正文使用三种标签：

- **【课程】**：来自官方讲义或视频；
- **【视频补充】**：幻灯片上没有、但老师在视频中明确说过；
- **【补充】**：本笔记为了让知识闭环而加入的解释或新例子。

---

## 0. 五分钟复习卡

> **第一次学习请跳到第 1 节。**本节故意把整讲压缩到很短，会提前出现还没解释的词。学完正文以后再回来复习。

### 0.1 一句话主线

语言模型接收的是整数 token，不是人眼看到的文字；BPE tokenizer 通过反复合并训练语料中最常见的相邻片段，在“小词表”和“短序列”之间取得实用折中，而整门 CS336 都围绕同一个问题：**在数据和硬件资源有限时，怎样最高效地构建最好的模型？**

### 0.2 全讲知识链

```text
只调用模型 API 很方便，但抽象层会漏出问题
        ↓
想做基础研究，就必须理解并亲手构建底层组件
        ↓
前沿模型太贵，小规模实验又不总能直接代表大规模
        ↓
重点学习可迁移的 mechanics（机制）和 mindset（思维方式）
        ↓
五个模块：basics → systems → scaling laws → data → alignment
        ↓
所有模块共同追求：给定资源，最大化模型效果
        ↓
第一块积木 tokenization：字符串 ↔ 整数 token
        ↓
字符方案词表大；字节方案序列长；单词方案无法覆盖新词
        ↓
BPE：从字节开始，反复合并最常见的相邻 token 对
```

### 0.3 必须记住的五件事

1. Token 是 tokenizer 定义的一个整数编号；它不一定等于一个字、一个词或一个完整 Unicode 字符。
2. Tokenizer 必须能 round trip（往返还原）：`decode(encode(text)) == text`。
3. 压缩率定义为“UTF-8 字节数 ÷ token 数”；越大表示同样文本使用的 token 越少。
4. BPE 训练时学习词表和 merge 顺序；正式使用模型时只应用已经学好的规则，不再统计新文本。
5. BPE 的核心折中是：常见片段用一个 token，罕见片段退回为多个较小 token，因此不会像纯单词词表那样轻易遇到未知词。

### 0.4 必须能重算的三个公式

**语言模型的链式分解：**

$$
P(t_1,t_2,\ldots,t_n)=\prod_{i=1}^{n}P(t_i\mid t_1,\ldots,t_{i-1})
$$

- $t_i$：第 $i$ 个 token；
- $P(\cdot)$：概率；
- 竖线 $\mid$：在左边已有 token 的条件下；
- $\prod$：把每一步的概率全部相乘。

**Tokenizer 压缩率：**

$$
r=\frac{B}{T}
$$

- $r$：每个 token 平均承载多少 UTF-8 字节；
- $B$：原字符串编码成 UTF-8 后的字节数；
- $T$：token 数。

课程例子有 $B=20,T=8$，所以 $r=20/8=2.5$ bytes/token。

**训练计算量的粗略估算：**

$$
C\approx 6ND
$$

- $C$：训练所需浮点运算次数（FLOPs）；
- $N$：模型参数数量；
- $D$：训练 token 数量；
- 6：常见 dense Transformer 训练的经验系数。本讲只预告，Lecture 2 再推导。

### 0.5 四种 tokenizer 一眼比较

| 方案 | 最初的单位 | 优点 | 致命问题 |
|---|---|---|---|
| Character | Unicode code point | 直观 | 词表大，稀有字符浪费位置 |
| Byte | 0 到 255 的字节 | 只需 256 个基础 token，能覆盖任意文本 | 序列很长 |
| Word | 单词或标点 | token 有人类语义，序列较短 | 词表近乎无界，会遇到新词 |
| Byte-level BPE | 字节，再学习合并 | 能覆盖任意文本，常见片段又较短 | 规则仍有人为边界和语言偏差 |

---

## 1. 这门课到底要解决什么问题？

### 1.1 什么是语言模型？

**【补充】**先不管 Transformer。最朴素地说，语言模型做一件事：看到前面的 token 后，给“下一个 token 是什么”分配概率。

假设 tokenizer 把“我 爱 猫”变成三个 token：

```text
t1 = “我”
t2 = “ 爱”
t3 = “ 猫”
```

模型可能给出：

$$
P(t_1)=0.5
$$

$$
P(t_2\mid t_1)=0.4
$$

$$
P(t_3\mid t_1,t_2)=0.1
$$

那么整段 token 序列的概率是：

$$
P(t_1,t_2,t_3)=0.5\times0.4\times0.1=0.02
$$

`0.02` 也就是 2%。真实模型的词表可能有十万到二十万个 token，每一步都要为整个词表给出一组概率。

这里立刻解释两个容易混淆的词：

- **参数（parameter）**：训练过程中学习到的数字，可以粗略理解为模型内部的“旋钮”；
- **训练（training）**：不断给模型文本，让它预测下一个 token，再根据错误调整参数；
- **推理（inference）**：参数已经学好，实际拿模型预测或生成文本。

### 1.2 为什么要“从头构建”？

**【课程】**过去十年，研究者使用语言模型的抽象层不断升高：

| 大致时期 | 常见做法 | 研究者亲自接触什么 |
|---|---|---|
| 2016 | 自己实现并训练模型 | 几乎整个技术栈 |
| 2018 | 下载 BERT 等预训练模型并微调 | 模型结构和训练的一部分 |
| 今天 | 调用 GPT、Claude、Gemini 等 API | 主要写 prompt |

抽象层升高会提高生产力，但语言模型的抽象是 **leaky abstraction（有泄漏的抽象）**：上层接口隐藏了底层细节，可底层限制仍会影响结果。

**【补充例子】**你让 API 模型处理一本很长的书，却得到“超出上下文长度”的错误。只会改 prompt 无法真正解决它。你需要知道：

- tokenizer 把这本书变成了多少 token；
- 模型支持多长的 context（上下文）；
- attention 为什么会随长度变贵；
- 是否能改变分块、检索、模型结构或推理系统。

因此课程主张 **understanding via building（通过构建获得理解）**。不是因为所有项目都应该重新训练模型，而是因为亲手实现能让你看到每一层的约束。

### 1.3 现实难题：语言模型已经工业化

**【课程】**前沿模型已经昂贵到普通课堂无法复现。讲义引用的例子包括：

- 2023 年 GPT-4 的训练成本据称约 1 亿美元；
- 2025 年 xAI 为 Grok 建设拥有约 23 万张 GPU 的集群；
- GPT-4 技术报告明确没有公开架构、硬件、训练计算量、数据集构建等细节。

**【视频补充】**老师说，如今前沿训练成本“可能在 10 亿美元量级”，但马上说明这是推测，不是公开确认的数据。不要把这句话误记成经审计的事实。

课堂当然可以训练小于 10 亿参数的小模型，但小模型不一定是大模型的缩小复刻：

1. **计算热点会变。**课程展示的旧数据中，模型放大后，MLP 层占总 FLOPs 的比例从约 44% 上升到约 80%。只优化小模型中的 attention，不一定能在大模型上获得同样收益。
2. **行为会变。**某些 few-shot 或 zero-shot 能力只在模型达到一定规模后明显出现。

这里的 **FLOP（floating-point operation）** 是一次浮点数学操作，例如一次浮点加法。FLOPs 在上下文中有时表示“总运算次数”，有时也被口语化地用来谈每秒算力；看到单位时要辨别：FLOP 是次数，FLOP/s 才是每秒速度。

### 1.4 哪些知识能从小规模迁移到大规模？

**【课程】**老师把知识分为三类：

| 类型 | 含义 | 能否较好迁移 |
|---|---|---|
| Mechanics（机制） | Transformer 怎样算、模型并行怎样通信 | 通常可以 |
| Mindset（思维方式） | 做 profiling、benchmarking，认真对待扩展和效率 | 通常可以 |
| Intuitions（经验直觉） | 哪种数据配比或结构改动能提升准确率 | 只能部分迁移 |

为什么最后一种更难？因为它往往来自实验，而不是从定义中推出来。课程举了 SwiGLU 论文的幽默结尾：作者观察到若干结构有效，却没有给出令人满意的原因。

这里先解释两个系统词：

- **benchmark（基准测试）**：实际运行程序，测它用了多少时间或资源；
- **profile（性能剖析）**：进一步查时间花在哪个算子、内存传输或通信步骤。

### 1.5 “苦涩的教训”不等于只堆规模

**【课程】**常见错误理解是：

> 规模最重要，所以算法不重要。

老师给出的正确理解是：

> **真正重要的是能够随资源扩展的算法。**

讲义用一个概念式表达：

$$
\text{accuracy}\approx\text{efficiency}\times\text{resources}
$$

- `accuracy`：最终模型效果；
- `resources`：数据、计算硬件、内存和通信带宽等投入；
- `efficiency`：每单位投入能换来多少有效效果。

这不是单位严格相等的物理定律，而是思考框架。

**【补充例子】**两种训练方法都使用 1 万张 GPU 小时：

- 方法 A 只有 30% 时间在做有效计算；
- 方法 B 通过系统优化达到 60%。

若其他条件相同，B 得到的有效计算约为 A 的两倍。小实验慢两倍也许只是多等一天；上亿美元的训练慢 5%，就是巨额浪费。

**【课程】**老师引用 2020 年的研究：2012 到 2019 年，ImageNet 任务上的算法效率提升约 44 倍。硬件和算法效率相乘，才共同带来巨大进步。

所以整门课的问题可以写成：

> 给定数据预算和计算预算，能够构建的最好模型是什么？

---

## 2. 语言模型从哪里来？

本节是课程提供的历史地图，不要求现在理解每个名字的技术细节。第一次阅读的目标只是知道“今天的系统不是突然出现的”。

### 2.1 神经网络以前

**【课程】**1950 年，Claude Shannon 用语言模型估计英语的熵；之后很长一段时间，N-gram 模型被用于机器翻译和语音识别。

**【补充】**N-gram 的意思是“只看最近 $n-1$ 个单位”。例如 3-gram 估计下一个词时只看前两个词。它简单，但无法自然记住很远的上下文。

### 2.2 2010 年代的关键积木

**【课程】**现代语言模型吸收了许多逐步积累的成果：

- LSTM：用循环结构保存序列状态；
- 2003 年神经语言模型：用神经网络预测词；
- sequence-to-sequence：把输入序列映射成输出序列；
- Adam optimizer：决定怎样根据梯度更新参数；
- attention：让模型选择性地查看输入不同位置；
- Transformer：主要依靠 attention 并行处理序列；
- mixture of experts（MoE）：每个 token 只调用部分“专家”子网络；
- model parallelism：把一个模型拆到多台设备上。

### 2.3 从预训练模型到智能体

**【课程】**使用方式发生了四次明显变化：

```text
2018：BERT —— 下载模型，再为具体任务 fine-tune（微调）
2020：GPT-3 —— 给模型 prompt（提示）
2022：ChatGPT —— 与模型多轮对话
2026：agents —— 模型自主执行包含工具调用的长任务
```

ELMo、BERT 和 T5 推动了预训练；GPT-2、scaling laws、GPT-3、PaLM 与 Chinchilla 推动了规模化。随后 The Pile、GPT-J、OPT、BLOOM，以及 Llama、Mistral、DeepSeek、Qwen、Kimi、GLM 等开放权重模型，使外界能看到更多前沿做法。

### 2.4 Open-weight 不等于 open-source

**【补充】**这两个词常被混用：

- **开放权重（open-weight）**：至少可以下载训练好的参数；
- **完整开源（这里按课程语境）**：还尽量公开论文、训练代码和训练数据。

只有权重仍不足以完整复现训练过程。AI2 的 OLMo、NVIDIA 的 Nemotron、Stanford Marin 等项目尝试公开更多环节。老师强调，正是开放生态提供的论文、代码和数据，使 CS336 这样的课程成为可能。

虽然应用形态变化很大，底层仍主要是 Transformer、attention、梯度优化、GPU kernel。变化最大的是规格：上下文更长、推理更多，因此效率更加重要。

---

## 3. 课程形式与学习方法

### 3.1 什么是 executable lecture？

**【视频补充】**约 19:28，老师解释屏幕上的讲义本身是 Python 程序。程序按函数层级组织内容，还能暂停并显示代码变量。比如 BPE 训练部分会实际展示：

```text
当前 token 序列
相邻 pair 的计数
本轮选择的 pair
新 token 编号
合并后的序列
```

这就是为什么 Lecture 1 没有普通 PDF。阅读 `lecture_01.py` 不只是看源代码，它就是本讲的官方讲义。

### 3.2 课程强调“做”，而不只是“看”

**【课程】**五次作业不提供完整 scaffolding code（脚手架代码），但提供单元测试和接口。典型工作流是：

1. 在自己的电脑上实现；
2. 用单元测试检查正确性；
3. 到集群上训练或 benchmark；
4. 在固定预算下比较准确率或速度。

单元测试是用小输入自动检查程序输出的代码。它把“最后才知道全错了”的稀疏反馈，变成每完成一个部件就能验证。

**【视频补充】**老师还特别区分了 AI 的两种用法：让 coding agent 直接生成整份作业，可能完成任务却学不到东西；让 AI 解释错误、追问概念、充当 tutor，则可以帮助学习。这也正是本系列笔记采用 Beginner Reviewer 循环的原因。

---

## 4. 全课程地图：五块拼成一个语言模型

### 4.1 Basics：先让模型真正训练起来

**【课程】**第一部分包含三块：

1. **Tokenization**：原始文字怎样变成模型能处理的整数；
2. **Model architecture**：这些整数进入怎样的神经网络；
3. **Training**：怎样调整网络参数，使预测越来越准。

现代 Transformer 有大量改进方向：

- activation function（激活函数）：给网络加入非线性，例如 ReLU、SwiGLU；
- positional encoding（位置编码）：告诉模型 token 的先后位置，例如 RoPE；
- normalization（归一化）：控制数值尺度，例如 LayerNorm、RMSNorm；
- attention 变体：减少长序列成本，例如局部 attention、GQA、MLA；
- state-space / linear attention：尝试以更低序列复杂度建模；
- dense MLP 与 mixture of experts；
- shape：层数、隐藏维度、head 数、expert 数等尺寸。

现在不用掌握这些缩写。课程在这里想表达的是：一个“Transformer”并不是唯一固定配方；每一个部件都涉及效果、稳定性和效率的权衡。

训练也不是只按一个按钮。需要选择：

- loss function（损失函数）：用数字衡量预测错得多严重；
- optimizer（优化器）：根据错误调整参数，例如 AdamW、Muon；
- initialization（初始化）：参数从什么数值起步；
- learning-rate schedule：每一步更新幅度如何变化；
- regularization：怎样减少过拟合；
- batch size：一次用多少训练样本估计更新方向。

**【视频补充】**老师说，这些看似只是“超参数”，但在大模型训练中，谨慎设置它们可以决定一次训练是稳定达到好结果，还是数值爆炸后完全报废。

Assignment 1 要实现 BPE、Transformer、cross-entropy、AdamW 和训练循环，在 TinyStories 与 OpenWebText 上训练，并在固定时间预算下尽量降低 perplexity（困惑度，越低通常表示下一个 token 预测得越好）。

这一部分的三个目标是：

| 目标 | 人话解释 |
|---|---|
| Expressivity（表达能力） | 模型能表示数据中的复杂关系 |
| Stability（稳定性） | 参数和梯度既不爆炸也不消失 |
| Efficiency（效率） | 在硬件上训练和推理得快 |

### 4.2 Systems：榨出硬件的有效工作

**【课程】**系统部分包含 resource accounting、kernels、parallelism 和 inference。

- **Resource accounting（资源核算）**：数清模型到底用了多少计算和内存；
- **Kernel（核函数）**：在 GPU 上执行的函数；
- **Parallelism（并行）**：把数据、模型或序列拆给多张 GPU；
- **Inference（推理）**：模型训练完后，怎样高效生成 token。

课程预告公式：

$$
C\approx6ND
$$

**【补充：这个 6 从哪里来？】**这是一条针对常见 **dense Transformer 训练**的粗略核算式。`Dense` 表示每个 token 大致都会经过模型中的全部参数，而不是像 MoE 那样只激活一部分专家。

先只看由参数矩阵完成的乘法。对一块权重矩阵 $W$：

1. 前向传播要把输入乘以 $W$。粗略地说，一个参数参与一次乘法和一次加法，约算 2 FLOPs；
2. 总共有 $N$ 个参数、$D$ 个训练 token，所以前向主项约为：

   $$
   C_{\text{forward}}\approx2ND
   $$

3. 反向传播要计算输入梯度和参数梯度，粗略成本约为前向的 2 倍：

   $$
   C_{\text{backward}}\approx2C_{\text{forward}}\approx4ND
   $$

4. 前向与反向相加：

   $$
   C_{\text{train}}
   \approx C_{\text{forward}}+C_{\text{backward}}
   \approx2ND+4ND
   =6ND
   $$

这里每个符号只表示一件事：

- $C_{\text{train}}$：整次训练的粗略浮点运算总数；
- $N$：每个 token 大致会激活的参数数量；
- $D$：整次训练处理的 token 数；
- 2 FLOPs：一次乘法加一次加法的常用计数约定。

这**不是推理公式**。推理没有训练反向传播，因此不能照搬系数 6。

它还把许多成本忽略或合并进了近似：attention 的 $T^2$ 项、embedding、loss、optimizer update、GPU 间通信、数据移动和硬件空闲时间都没有逐项列出。当参数矩阵乘是主要计算时，这条式子很有用；若短序列中的 embedding/loss、极长序列的 attention、通信或其他算子占主导，它就会失真。

对于 **MoE（Mixture of Experts，混合专家）**，每个 token 通常只经过少数 expert。此时 $N$ 应理解为“每 token 实际激活的参数量”，不能直接用模型包含的全部参数量。

手算一个课程数字：训练 $N=70$ billion（700 亿）参数模型，使用 $D=1$ trillion（1 万亿）token：

$$
N=70\times10^9,\qquad D=10^{12}
$$

$$
C\approx6\times70\times10^9\times10^{12}
$$

先乘普通数字：

$$
6\times70=420
$$

再乘 10 的幂：

$$
10^9\times10^{12}=10^{21}
$$

所以：

$$
C\approx420\times10^{21}=4.2\times10^{23}\text{ FLOPs}
$$

**【课程】**B200 的示例峰值约为 2.25 PFLOP/s（BF16）与 8 TB/s 显存带宽。`P` 是 peta，即 $10^{15}$；`T` 是 tera，即 $10^{12}$。关键矛盾是参数存在 HBM（高带宽显存）里，计算单元在芯片另一处，搬数据经常比做乘加更慢。

**【补充例子】**如果操作 A 和 B 分成两个 kernel：

```text
读 HBM → 算 A → 写 HBM → 再读 HBM → 算 B → 再写 HBM
```

把它们 fusion（融合）后：

```text
读 HBM → 连续算 A 和 B → 写 HBM
```

中间值少往返一次显存，因此可能显著加速。

多 GPU 时，设备之间搬数据更慢，需要 gather、reduce、all-reduce 等 collective operation（集合通信），并按数据、张量、流水线、序列或 expert 切分工作。

推理分成：

- **prefill**：一次处理 prompt 的全部 token，建立中间状态；
- **decode**：一次生成一个新 token。

Decode 难以一次并行许多时间步，而且每一步都要读取大量参数，所以经常 memory-bound（受内存带宽限制）。

### 4.3 Scaling laws：大实验前先学会预测

**【课程】**假设只有一次机会使用 $10^{25}$ FLOPs，不能在目标规模上随便试超参数。解决思路不是只设计“一个模型”，而是设计 **scaling recipe（扩展配方）**：

```text
计算预算 → 模型大小、数据量、学习率、batch size 等超参数
```

流程是：

1. 在多个较小预算上训练；
2. 记录 loss；
3. 拟合 scaling law；
4. 外推到目标预算；
5. 再比较和改进整套 recipe。

**Hyperparameter transfer（超参数迁移）**表示小规模的最佳设置能直接用于大规模，或至少按可预测规则变化。若小模型学习率是 $10^{-5}$、稍大模型突然要 $10^{-4}$，没有规律，就很难安全外推。

**【视频补充】**老师强调：scaling law 不是自动存在的自然定律，必须通过精心设计稳定、可预测的训练 recipe “把它做出来”。因此 predictability（可预测性）至少与 optimality（单点最优）一样重要。

经典问题是：固定计算预算时，应该增加参数 $N$，还是增加训练 token $D$？课程给出 Chinchilla 风格的粗略经验：

$$
D\approx20N
$$

若 $N=70$ billion：

$$
D\approx20\times70\text{ billion}=1400\text{ billion}=1.4\text{ trillion tokens}
$$

这只是粗略规则，会随数据和结构改变，也没有纳入部署推理成本。现实中为了让推理模型更小，常会对小模型训练远多于计算最优点的数据。

### 4.4 Data：模型最终学到什么，由数据决定

**【课程】**数据不是“从天上掉下来的现成文本”。来源可能是网页、图书、论文和代码，而原始形式往往是 HTML、PDF 或整个代码目录。

处理步骤包括：

1. **Transformation**：从 HTML/PDF 等提取正文；
2. **Filtering**：去掉低质量或有害内容；
3. **Deduplication**：删除重复内容，避免浪费计算和过度记忆；
4. **Mixing**：决定网页、书籍、代码等来源各占多少；
5. **Synthetic data**：用模型改写或生成更贴近目标任务的数据。

还必须考虑版权、许可和隐私。例如没有许可证的 GitHub 代码不能自动等同于可任意用于训练。

数据按训练阶段可粗分为：

- **pretraining data**：量大、类型多；
- **mid-training data**：预训练后段加入的高质量或长上下文数据；
- **post-training data**：对话、偏好、工具调用轨迹等。

Evaluation（评测）也分两类：

- 内部评测帮助开发，重视跨规模变化平滑、方案相对排名可靠；
- 外部评测面向真实使用，重视任务是否代表用户真正需求。

### 4.5 Alignment：用“好坏反馈”继续改进

**【课程】**预训练提供完整监督：每个位置都有真实的下一个 token。之后可以用较弱的监督继续改进，因为“判断两个答案哪个更好”常常比“从零写出完美答案”容易。

基本循环是：

```text
模型生成多个回答
        ↓
人类、规则验证器或另一个 LM 给分
        ↓
更新模型，使好回答更可能出现
```

课程会涉及 PPO、DPO、GRPO。第一次阅读只需知道它们是把反馈用于更新模型的不同算法。

**【视频补充】**大规模 reinforcement learning（强化学习）不仅有算法问题，还有系统问题：推理服务生成 rollout（完整尝试轨迹），训练服务更新参数；若生成工作者太慢，数据就可能来自旧模型，出现 off-policy 问题。团队必须在“数据尽可能来自当前模型”和“系统吞吐尽可能高”之间权衡。

### 4.6 五部分其实只有一个共同问题

**【课程】**所有设计都可以用效率解释：

- Systems：少搬数据，提高计算利用率；
- Tokenization：用更少 token 表示同样文本；
- Architecture：减少内存或 FLOPs，同时保住效果；
- Data filtering：不把有限计算浪费在重复或低质量数据上；
- Scaling laws：用小实验选择大实验的设置。

今天通常假设 compute-constrained（计算不够）；未来如果高质量数据更稀缺，决策就可能转向 data-constrained（数据不够），但“固定资源下最大化效果”的框架不变。

---

## 5. Tokenization：模型为什么不直接读文字？

### 5.1 电脑看到的不是“字义”

**【课程】**人写下的是 Unicode string（Unicode 字符串），语言模型内部处理的却是整数序列。Tokenizer 提供两个方向：

```text
encode
字符串 ─────────→ token ID 序列

decode
字符串 ←───────── token ID 序列
```

例如课程使用：

```text
原字符串："Hello, 🌍! 你好!"
token IDs：[13225, 11, 130321, 235, 0, 220, 177519, 0]
```

这里的 `13225` 不是数量，也没有数值大小的语义。它只是词表中的一个标签，类似“储物柜 13225 号”。ID 13225 对应字节片段 `Hello`。

### 5.2 Vocabulary 是一张双向字典

**Vocabulary（词表）**列出每个 token ID 代表的字节片段：

```text
13225  →  b"Hello"
11     →  b","
0      →  b"!"
...
```

Vocabulary size，记作 $V$，是允许的不同 token ID 数量。

模型不能直接把任意整数当输入。它通常先通过 embedding table（嵌入表），把 token ID 查成一排浮点数。若词表有 $V$ 个 token，每个 token 的 embedding 有 $d$ 个数，那么表中有：

$$
V\times d
$$

个参数。

**【补充手算】**若 $V=50{,}000,d=768$：

$$
50{,}000\times768=38{,}400{,}000
$$

即 3840 万个参数。若其他不变，把词表加倍到 10 万：

$$
100{,}000\times768=76{,}800{,}000
$$

词表大可以让序列变短，却会增大 embedding 和输出层，并让许多罕见 token 得不到充分训练。这就是课程所说的 sparsity（稀疏）：词表位置很多，但每个罕见位置只在很少训练样本中出现。

### 5.3 最低正确性要求：round trip

Tokenizer 至少应满足：

$$
\operatorname{decode}(\operatorname{encode}(s))=s
$$

- $s$：任意合法输入字符串；
- `encode`：字符串转 token ID；
- `decode`：token ID 转回字符串；
- 等号：每个字符、空格、标点都应完全相同。

**【课程】**老师明确说：自己实现 tokenizer 时，如果 round trip 失败，就有 bug。

这不意味着“任意 token ID 列表都能单独解成合法字符串”。某个 token 可能只代表一个多字节字符的一部分。正确要求是：由合法字符串编码得到的完整 token 序列，应能拼回原始 UTF-8 字节并解码。

---

## 6. 必需前置知识：bit、byte、Unicode 与 UTF-8

### 6.1 bit 和 byte

**【补充】**bit（比特）是一个只能取 0 或 1 的位置。8 个 bit 组成 1 byte（字节）。8 个二进制位一共有：

$$
2^8=256
$$

种组合，所以一个 byte 可看成 0 到 255 的整数。

例如十进制 97 的二进制是：

```text
01100001
```

在 UTF-8 中，英文小写字母 `a` 正好用这个 byte 表示。

### 6.2 Unicode 先给字符编号

**Unicode** 是跨语言的字符编号标准。每个 code point（码点）是一个整数编号：

```python
ord("a") == 97
ord("🌍") == 127757

chr(97) == "a"
chr(127757) == "🌍"
```

`ord` 把字符映射到码点，`chr` 做反向转换。

**【补充：进阶但重要】**“人眼看到的一个符号”不总等于“一个 Unicode code point”。带音标的字母可能由基本字母和组合符号组成；家庭 emoji 也可能由多个码点组合。Tokenizer 不应依赖“一眼一个字就是一个整数”的直觉。

### 6.3 UTF-8 再把码点存成 bytes

Code point 可能远大于 255，不能都塞进一个 byte。UTF-8 是把 Unicode 字符串编码为 byte 序列的一套规则：

**【补充：从 code point 手算 UTF-8】**UTF-8 先根据 code point 大小选择 1、2、3 或 4-byte 模板。模板中的 `x` 用来放 code point 的二进制有效位；开头固定的 `0`、`110`、`1110`、`11110` 和后续 byte 的 `10` 是格式标记。

| Code point 范围 | byte 数 | 二进制模板 | 能装的 payload bits |
|---|---:|---|---:|
| U+0000 到 U+007F | 1 | `0xxxxxxx` | 7 |
| U+0080 到 U+07FF | 2 | `110xxxxx 10xxxxxx` | $5+6=11$ |
| U+0800 到 U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` | $4+6+6=16$ |
| U+10000 到 U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | $3+6+6+6=21$ |

#### 手算 `🌍`：U+1F30D

1. `U+1F30D` 的十进制是 127757，二进制有效位是：

   ```text
   11111001100001101
   ```

2. 它大于 U+FFFF，所以选择 4-byte 模板。模板有 21 个 `x`，把 payload 左侧补 0 到 21 bits：

   ```text
   000011111001100001101
   ```

3. 按模板的 $3+6+6+6$ 个 payload 位分组：

   ```text
   000 | 011111 | 001100 | 001101
   ```

4. 把四组分别填进 `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`：

   ```text
   11110 000   10 011111   10 001100   10 001101
   = 11110000  10011111    10001100    10001101
   ```

5. 每组 8 bits 转成十进制：

   ```text
   11110000 = 128+64+32+16       = 240
   10011111 = 128+16+8+4+2+1    = 159
   10001100 = 128+8+4            = 140
   10001101 = 128+8+4+1          = 141
   ```

因此：

```text
U+1F30D → 11110000 10011111 10001100 10001101
        → [240, 159, 140, 141]
```

#### 为什么中文 `你` 是 3 bytes？

`你` 是 U+4F60，介于 U+0800 和 U+FFFF，所以选 3-byte 模板。它的 16-bit payload 是：

```text
0100111101100000
```

按 $4+6+6$ 分组并填模板：

```text
0100 | 111101 | 100000

1110 0100   10 111101   10 100000
= 11100100  10111101    10100000
= [228, 189, 160]
```

所以“一个汉字”不等于“一个 byte”；在 UTF-8 中，这个汉字用 3 bytes 保存。

```text
"a"   → [97]                    （1 byte）
"🌍"  → [240, 159, 140, 141]    （4 bytes）
"你"  → [228, 189, 160]         （3 bytes）
"好"  → [229, 165, 189]         （3 bytes）
```

课程字符串 `"Hello, 🌍! 你好!"` 为什么有 20 bytes？逐段数：

| 片段 | UTF-8 bytes |
|---|---:|
| `Hello` | 5 |
| `,` | 1 |
| 空格 | 1 |
| `🌍` | 4 |
| `!` | 1 |
| 空格 | 1 |
| `你` | 3 |
| `好` | 3 |
| `!` | 1 |
| 合计 | $5+1+1+4+1+1+3+3+1=20$ |

这条路径不要混淆：

```text
人眼看到的字符串
    ↓ Unicode：每个码点是什么
码点序列
    ↓ UTF-8：怎样存成 0..255
byte 序列
    ↓ tokenizer：怎样把常见 bytes 分块
token ID 序列
```

---

## 7. 怎样判断 tokenizer 好不好？

### 7.1 两个相互拉扯的量

**【课程】**主要观察：

1. **Vocabulary size $V$**：可以使用多少种 token；
2. **Sequence length $T$**：一段文本会被切成多少 token。

通常：

- $V$ 小，基础单位就小，$T$ 容易变大；
- $V$ 大，常见长片段能变成一个 token，$T$ 容易变小；
- 但 $V$ 太大，embedding/output 参数更多，稀有 token 学不充分。

Tokenizer 设计不是单纯追求词表最大或序列最短，而是在二者之间折中。

### 7.2 压缩率逐步计算

课程定义：

$$
r=\frac{B}{T}
$$

其中：

- $r$：compression ratio，单位 bytes/token；
- $B$：原字符串的 UTF-8 byte 数；
- $T$：token 数。

课程用 `o200k_base` 编码 `"Hello, 🌍! 你好!"`：

```text
B = 20 bytes
T = 8 tokens
```

代入：

$$
r=\frac{20\text{ bytes}}{8\text{ tokens}}=2.5\text{ bytes/token}
$$

这句话只表示：平均每个 token 承载 2.5 个原始 byte。它不是“文件真的被无损压成原来的 40%”，因为还要存 token ID，而且训练目标是计算效率，不是制作 zip 文件。

### 7.3 为什么短序列如此重要？

**【课程】**标准 full attention 对长度 $T$ 的主要位置两两交互量是 $T^2$。这里的 $O(T^2)$ 表示：当长度放大 $k$ 倍时，这部分工作大致放大 $k^2$ 倍。

课程给出直觉：1000 bytes 经过 tokenizer 后可能约为 250 tokens，即长度缩短 4 倍。

若直接对 1000 个 byte 位置两两计算：

$$
1000^2=1{,}000{,}000
$$

若只对 250 个 token 位置计算：

$$
250^2=62{,}500
$$

两者相除：

$$
\frac{1{,}000{,}000}{62{,}500}=16
$$

因此在这个极简比较中，attention 位置对减少 16 倍，而不是 4 倍。真实总耗时还包括 MLP、embedding、内存访问等，不能直接宣称整个模型恰好快 16 倍。

### 7.4 Adaptive computation：复杂处多花 token

**【课程】**Tokenization 还有一个更细的作用：adaptive computation（自适应计算）。

```text
训练语料中非常常见的片段 → 合成较大的 token → 少占位置
训练语料中罕见或新奇的片段 → 拆成较小 token → 多占位置
```

Transformer 对每个 token 位置都运行多层计算。于是罕见片段被拆成更多 token，相当于获得更多表示位置；常见片段则被压缩。

**【补充例子】**若 tokenizer 已把常见英文片段 `ing` 学成一个 token，而罕见字符串 `xqz` 仍是 3 个 token：

```text
"ing" → 1 个位置
"xqz" → 3 个位置
```

后者得到三个位置的计算和表示空间。不过“罕见就一定更有意义”并不总成立；错别字、噪声和低资源语言也可能被切得更碎。这既是自适应计算，也是潜在偏差。

---

## 8. 三个看似自然、实际都不理想的方案

### 8.0 可跳过：读懂本讲所需的最小 Python 语法

> **【补充】不打算看代码可以跳过本框。**下面只解释本讲实际出现的四种写法。

#### `map(function, values)`：对每一项调用函数

```python
map(ord, "ab")
```

意思是依次计算 `ord("a")`、`ord("b")`。`map` 返回一个可迭代对象，外层 `list(...)` 把结果真正收集成列表：

```python
list(map(ord, "ab")) == [97, 98]
```

#### `separator.join(parts)`：把许多片段连接起来

```python
"".join(["你", "好"]) == "你好"
b"".join([b"th", b"e"]) == b"the"
```

引号中是分隔符。这里用空字符串 `""` 或空 bytes `b""`，表示片段之间不插入额外内容。

#### Comprehension：用一行循环生成集合

列表推导式：

```python
[ord(c) for c in "ab"] == [97, 98]
```

人话是：“对 `"ab"` 中每个字符 `c`，计算 `ord(c)`，把结果放进列表。”

字典推导式：

```python
{i: bytes([i]) for i in range(256)}
```

人话是：“让 `i` 从 0 到 255，为每个 `i` 建立 `i → 单个 byte` 的映射。”冒号左边是 key，右边是 value。

#### Regular expression（正则表达式）：按文字模式匹配

```python
regex.findall(r"\w+|.", text)
```

- `findall`：按从左到右顺序返回全部匹配；
- `\w+`：一个或更多字母、数字或下划线；
- `|`：或者；
- `.`：若前一个分支不匹配，就取任意单个非换行字符，例如空格或标点；
- 前面的 `r`：raw string，让 Python 不先吞掉反斜杠。

这是课程为了演示 word tokenizer 使用的简化规则，不是适合所有语言的完美分词器。

### 8.1 Character tokenizer

**【课程】**Character tokenizer 直接把每个 Unicode code point 当一个 token：

```python
def encode(text):
    return list(map(ord, text))

def decode(ids):
    return "".join(map(chr, ids))
```

它能 round trip，但有两个问题：

1. Unicode 有约 15 万个字符，词表很大；
2. 大量字符极少出现，却各占一个完整词表位置。

对英文而言，一个字符通常只承载一个字母，压缩率也不高。它同时承受“大词表”和“长序列”，因此课程称为“两个世界最坏的一面”。

### 8.2 Byte tokenizer

**【课程】**Byte tokenizer 先做 UTF-8 编码，再把每个 byte 直接作为 token：

```python
def encode(text):
    return list(text.encode("utf-8"))

def decode(ids):
    return bytes(ids).decode("utf-8")
```

优点非常明确：词表只需 256 个基础值，并且任意合法 UTF-8 文本都能表示。

缺点同样明确。每个 token 恰好代表 1 byte：

$$
r=\frac{B}{T}=\frac{B}{B}=1\text{ byte/token}
$$

英语单词 `hello` 需要 5 个 token，汉字 `你` 需要 3 个 token，emoji `🌍` 需要 4 个 token。标准 Transformer 的有限 context 很快被用完，attention 也会变贵。

### 8.3 Word tokenizer

**【课程】**经典 NLP 更接近按单词和标点切分。课程例子是：

```text
I'll say supercalifragilisticexpialidocious!
```

一个简化正则表达式把连续字母数字放在一起，并保留其他字符：

```python
regex.findall(r"\w+|.", text)
```

可能得到类似：

```text
["I", "'", "ll", " ", "say", " ",
 "supercalifragilisticexpialidocious", "!"]
```

优点是每个 token 常有清楚的人类语义，压缩率高。问题是：

- 训练数据中的不同单词数量可能极大；
- 新姓名、拼写变化、网址、代码标识符可以不断出现，词表没有自然上限；
- 测试时没见过的词只能映射成 `UNK`（unknown，未知）token；
- 多个不同未知词都变成同一个 `UNK`，原文无法 round trip。

**【补充例子】**训练词表只有 `cat`、`dog`。测试出现 `capybara` 与 `platypus`：

```text
"capybara" → UNK
"platypus" → UNK
```

模型看到完全相同的 ID，无法知道原来是哪一个词。BPE 希望保留 word tokenizer 的部分压缩优势，同时像 byte tokenizer 一样覆盖任意文本。

---

## 9. Byte Pair Encoding：从零训练

### 9.1 核心思想

**【课程】**BPE 最初由 Philip Gage 在 1994 年用于数据压缩；Sennrich 等人在 2016 年将 subword 思路用于神经机器翻译；GPT-2 后来采用 byte-level BPE。

课程版本从所有单字节 token 开始：

```text
基础 vocabulary：0, 1, 2, ..., 255
```

然后反复执行：

1. 统计 token 序列中每种相邻 pair 出现多少次；
2. 找出现次数最多的 pair；
3. 为这个 pair 创建一个新 token ID；
4. 从左到右，把该 pair 的每个不重叠出现合并；
5. 重复指定轮数。

常见 byte 序列逐渐变成一个 token；罕见序列仍能退回基础 byte。因此不存在纯单词 tokenizer 的普通未知词问题。

### 9.2 课程原例：`the cat in the hat`

原字符串共有 18 个 ASCII 字符，每个都占 1 byte：

```text
t h e _ c a t _ i n _ t h e _ h a t
```

这里用 `_` 显示空格，实际 byte ID 是 32。其他关键 byte ID：

```text
t = 116
h = 104
e = 101
```

#### 第 0 步：初始状态

每个 byte 都是一个 token：

```text
[t, h, e, _, c, a, t, _, i, n, _, t, h, e, _, h, a, t]
```

token 数：

$$
T_0=18
$$

#### 第 1 轮：找最常见 pair

**【课程原例复算】**18 个 token 之间恰好有 $18-1=17$ 个相邻位置。这里把它们一个不漏地列出；`_` 仍表示空格：

| 相邻位置 | 左 token | 右 token | pair |
|---:|---|---|---|
| 1-2 | `t` | `h` | `(t,h)` |
| 2-3 | `h` | `e` | `(h,e)` |
| 3-4 | `e` | `_` | `(e,_)` |
| 4-5 | `_` | `c` | `(_,c)` |
| 5-6 | `c` | `a` | `(c,a)` |
| 6-7 | `a` | `t` | `(a,t)` |
| 7-8 | `t` | `_` | `(t,_)` |
| 8-9 | `_` | `i` | `(_,i)` |
| 9-10 | `i` | `n` | `(i,n)` |
| 10-11 | `n` | `_` | `(n,_)` |
| 11-12 | `_` | `t` | `(_,t)` |
| 12-13 | `t` | `h` | `(t,h)` |
| 13-14 | `h` | `e` | `(h,e)` |
| 14-15 | `e` | `_` | `(e,_)` |
| 15-16 | `_` | `h` | `(_,h)` |
| 16-17 | `h` | `a` | `(h,a)` |
| 17-18 | `a` | `t` | `(a,t)` |

把相同 pair 汇总：

| Pair | 次数 | 第一次出现的位置 |
|---|---:|---:|
| `(t,h)` | 2 | 1-2 |
| `(h,e)` | 2 | 2-3 |
| `(e,_)` | 2 | 3-4 |
| `(a,t)` | 2 | 6-7 |
| `(_,c)`、`(c,a)`、`(t,_)`、`(_,i)`、`(i,n)`、`(n,_)`、`(_,t)`、`(_,h)`、`(h,a)` | 各 1 | 其余位置 |

所以并列最高的不是三个，而是四个：

```text
(t,h), (h,e), (e,space), (a,t) 都出现 2 次
```

为什么课程代码最终选 `(t,h)`？`count_adjacent_pairs` 从左到右扫描。Python 字典第一次看到新 pair 时才插入 key；后面再次遇到只把计数加一，不改变 key 的插入位置。因此这四个候选在字典中的先后顺序是：

```text
(t,h) → (h,e) → (e,space) → (a,t)
```

`max(counts, key=counts.get)` 发现四者计数相同，就返回遍历时最先遇到的 `(t,h)`，也就是 byte IDs `(116,104)`。若实现规定了另一种 tie-break，可能选出不同 pair；训练和使用时保持规则一致才是关键。

创建新 token：

```text
256 → bytes "th"
```

替换两个不重叠出现：

```text
[256, e, _, c, a, t, _, i, n, _, 256, e, _, h, a, t]
```

每次合并把两个 token 变成一个。这里合并 2 处，所以：

$$
T_1=18-2=16
$$

#### 第 2 轮：重新统计，不沿用旧计数

现在 `(256,e)` 出现 2 次。创建：

```text
257 → vocab[256] + vocab[e]
    → bytes "th" + bytes "e"
    → bytes "the"
```

替换后：

```text
[257, _, c, a, t, _, i, n, _, 257, _, h, a, t]
```

长度再次减少 2：

$$
T_2=16-2=14
$$

#### 第 3 轮

`(257, _)`，也就是 `("the", 空格)`，出现 2 次。创建：

```text
258 → bytes "the "
```

替换后：

```text
[258, c, a, t, _, i, n, _, 258, h, a, t]
```

所以：

$$
T_3=14-2=12
$$

### 9.3 词表和压缩率怎样变化？

开始有 256 个 byte token，每轮增加 1 个 token。做 3 轮后：

$$
V=256+3=259
$$

暂时不计 special token。原文仍为 18 bytes，现在是 12 tokens：

$$
r=\frac{18}{12}=1.5\text{ bytes/token}
$$

这与视频中显示的 1.5 一致。

注意：原始内容没有丢失。ID 258 内部仍保存 `b"the "`，解码时展开即可。

### 9.4 对应的极简训练代码

**【课程代码】**下面代码来自官方 executable lecture；其后的逐行语法拆解是本笔记补充。

```python
indices = list(text.encode("utf-8"))
vocab = {i: bytes([i]) for i in range(256)}
merges = {}

for i in range(num_merges):
    counts = count_adjacent_pairs(indices)
    pair = max(counts, key=counts.get)
    new_id = 256 + i

    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    indices = merge(indices, pair, new_id)
```

课程中的 pair 计数函数是：

```python
def count_adjacent_pairs(indices):
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts
```

下面不跳步地解释核心语法和数据变化。

#### `indices = list(text.encode("utf-8"))`

`text.encode("utf-8")` 得到 bytes。遍历 bytes 时，每一项是 0 到 255 的整数；`list(...)` 把它们收集成当前 token ID 列表。

#### `vocab = {i: bytes([i]) for i in range(256)}`

这是字典推导式。`range(256)` 依次产生 0 到 255，不包含 256。对每个整数 `i`：

```text
key   = i
value = bytes([i])
```

为什么写 `bytes([i])` 而不是 `bytes(i)`？

- `bytes([97])` 把列表中的数 97 当成一个 byte 值，得到 `b"a"`；
- `bytes(97)` 的意思是创建 97 个值为 0 的 bytes，完全不是这里想要的结果。

所以初始 `vocab` 保存：

```text
0 → 一个值为 0 的 byte
1 → 一个值为 1 的 byte
...
97 → b"a"
...
255 → 一个值为 255 的 byte
```

#### `zip(indices, indices[1:])` 为什么得到相邻 pair？

若：

```text
indices     = [A, B, C, D]
indices[1:] = [B, C, D]
```

`zip` 把同一列位置配起来：

```text
(A,B), (B,C), (C,D)
```

它按从左到右顺序产生 pair。`defaultdict(int)` 使尚不存在的 pair 默认计数为 0，因此：

```python
counts[(index1, index2)] += 1
```

第一次看到 pair 时相当于 $0+1$，以后再看到就在原计数上加 1。

#### `pair = max(counts, key=counts.get)` 到底在比较什么？

直接遍历 Python 字典 `counts` 时，得到的是 **key**，这里也就是 `(左 ID, 右 ID)` 这样的 pair，而不是计数 value。

`key=counts.get` 中的 `key` 是 `max` 函数的命名参数，不是字典 key 的另一个变量。它告诉 `max`：

```text
对每个候选 pair p，使用 counts.get(p) 作为比较分数
```

等价于这段更啰嗦的伪代码：

```python
best_pair = None
best_count = -1
for candidate_pair in counts:       # 遍历的是字典 keys
    candidate_count = counts.get(candidate_pair)
    if candidate_count > best_count:
        best_pair = candidate_pair
        best_count = candidate_count
```

现代 Python 字典保留 key 的首次插入顺序。`count_adjacent_pairs` 又从左到右扫描，所以 key 的遍历顺序就是“每种 pair 第一次在文本中出现”的顺序。`max` 遇到相同最大分数时返回遍历中先到的元素。

这四件事连起来才解释了课程的 tie-break：

```text
从左到右首次插入
    ↓
字典按首次插入顺序遍历 keys
    ↓
counts.get 给出每个 key 的计数
    ↓
max 并列时保留先到者
    ↓
(t,h) 胜出
```

如果 `count_adjacent_pairs` 不是从左到右插入，或者程序先排序了 keys，这个并列结果就可能不同。

#### `new_id`、`pair[0]` 和 `pair[1]`

`pair` 是长度为 2 的 tuple（元组）。例如：

```text
pair = (116, 104)
pair[0] = 116    # 第一个元素，对应 t
pair[1] = 104    # 第二个元素，对应 h
```

Python 下标从 0 开始，所以 `[0]` 是第一项、`[1]` 是第二项。

```python
new_id = 256 + i
```

使第 0 轮的新 ID 为 256，第 1 轮为 257，避开已有的 0 到 255。

```python
vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
```

先用两个旧 ID 分别查出 byte 串，再用 `+` 首尾连接。第一轮是：

```text
vocab[256] = vocab[116] + vocab[104]
           = b"t" + b"h"
           = b"th"
```

这里绝不是整数 $116+104=220$。

#### `merges` 和 `merge(...)`

```python
merges[pair] = new_id
```

保存“旧 pair → 新 ID”以及规则加入的先后顺序。`merge(indices, pair, new_id)` 再从左到右把当前序列中所有不重叠目标 pair 换成 `new_id`。

#### 为什么强调“不重叠”？

**【补充】**序列 `[A,A,A]` 有两个相邻位置对：位置 `(1,2)` 和 `(2,3)`。但中间的 `A` 不能同时参加两次合并。左到右合并后只能得到：

```text
[AA, A]
```

而不是两个 `AA`。统计 pair 可以按相邻位置计数；真正替换时必须避免一个 token 被重复使用。

### 9.5 再来一个额外例子：`haha haha`

**【补充例子】**用符号而不是 byte ID 演示：

```text
h a h a _ h a h a
```

`(h,a)` 出现 4 次，是最高频 pair。令 `X = "ha"`：

```text
X X _ X X
```

序列从 9 个 token 变成 5 个 token。重新统计后，`(X,X)` 出现 2 次。令 `Y = "haha"`：

```text
Y _ Y
```

序列变成 3 个 token。这个例子展示了 BPE 的层层组合：后来的 token 可以由先前新建的 token 继续合并而成。

---

## 10. 训练完以后，怎样编码新文本？

### 10.1 训练与使用是两个阶段

**【课程】**最重要的边界先说清楚：

```text
BPE 训练阶段：
训练语料 → 统计频率 → 得到 vocab + 有顺序的 merge 规则

模型训练 / 推理阶段：
新文本 → 只应用固定规则 → token IDs
```

编码新文本时不能重新选择“新文本里最常见的 pair”。否则同一句话在不同上下文中可能得到不同词表含义，模型也不知道每个 ID 代表什么。

### 10.2 课程原例：`the quick brown fox`

课程在 `the cat in the hat` 上学到三条规则：

```text
规则 1：(t, h)       → 256，即 "th"
规则 2：(256, e)     → 257，即 "the"
规则 3：(257, space) → 258，即 "the "
```

对新文本：

```text
the quick brown fox
```

一开始是 19 个 ASCII byte token：

```text
t h e _ q u i c k _ b r o w n _ f o x
```

按训练顺序应用规则：

```text
规则 1 后：256 e _ q u i c k _ b r o w n _ f o x
规则 2 后：257 _ q u i c k _ b r o w n _ f o x
规则 3 后：258 q u i c k _ b r o w n _ f o x
```

这一次每条规则都只匹配一处，所以每轮恰好少 1 个 token：

```text
初始 19
  ↓ 合并 t+h
规则 1 后 18
  ↓ 合并 th+e
规则 2 后 17
  ↓ 合并 the+space
规则 3 后 16
```

即：

$$
19\to18\to17\to16
$$

最终 16 个 token。前四个 bytes `the ` 被 ID 258 代替，其余没有匹配已学规则的片段仍保持为 byte token。

### 10.3 为什么 merge 顺序不能丢？

**【补充】**假设词表同时有：

```text
(A, B) → X
(B, C) → Y
```

输入是 `[A,B,C]`。如果先合并左边，得到 `[X,C]`；先合并右边，得到 `[A,Y]`。结果不同。

因此 tokenizer 参数不仅要保存“有哪些 token”，还要保存 merge rank（合并优先级）。课程的简化代码利用字典插入顺序，按训练得到的先后依次应用规则。

### 10.4 解码为什么简单？

**【课程】**课程代码对每个 ID 查 `vocab`，连接 bytes，最后统一做 UTF-8 decode：

```python
byte_pieces = [vocab[token_id] for token_id in ids]
all_bytes = b"".join(byte_pieces)
text = all_bytes.decode("utf-8")
```

形式化地写：

$$
s=\operatorname{UTF8Decode}\left(
\operatorname{vocab}[i_1]\Vert
\operatorname{vocab}[i_2]\Vert\cdots\Vert
\operatorname{vocab}[i_T]
\right)
$$

- $s$：恢复出的字符串；
- $i_1,\ldots,i_T$：共 $T$ 个 token ID；
- `vocab[i]`：ID $i$ 对应的 bytes；
- $\Vert$：把 byte 串首尾连接；
- `UTF8Decode`：把完整 bytes 解释成 Unicode 字符串。

合并训练从未删除 byte 内容，只是给常见的 byte 串新增短编号，所以能恢复原文。

### 10.5 真实例子：一个 emoji 可以跨两个 token

**【课程原例复算】**官方代码使用 tiktoken 的 `o200k_base`：

```text
文本："Hello, 🌍! 你好!"
IDs：[13225, 11, 130321, 235, 0, 220, 177519, 0]
词表大小：200019
```

每个 ID 的实际 byte 片段是：

| ID | byte 片段 | 人眼可见含义 |
|---:|---|---|
| 13225 | `b"Hello"` | `Hello` |
| 11 | `b","` | 逗号 |
| 130321 | `b" \xf0\x9f\x8c"` | 空格 + `🌍` 的前 3 bytes |
| 235 | `b"\x8d"` | `🌍` 的最后 1 byte |
| 0 | `b"!"` | 感叹号 |
| 220 | `b" "` | 空格 |
| 177519 | `你好` 的 6 个 UTF-8 bytes | `你好` |
| 0 | `b"!"` | 感叹号 |

把每个 token 真正携带的 byte 数相加：

$$
5+1+4+1+1+1+6+1=20\text{ bytes}
$$

注意第三个 token 的 4 bytes 是“1 个空格 + emoji 的前 3 bytes”；第四个 token 再提供 emoji 的最后 1 byte。8 个 token 的 byte 长度不相等，但拼接后恰好还原课程最初数出的 20 bytes。

ID 130321 或 235 单独都不能解成合法的 `🌍`。把两者 bytes 拼起来才是：

```text
f0 9f 8c 8d → 🌍
```

所以 **token 不保证是完整字符**。许多界面为显示方便会把无法单独解码的片段画成替代符号，这不代表 tokenizer 丢失了数据。

### 10.6 真实 tokenizer 还需要三类工程处理

**【课程】**Assignment 1 会把玩具实现补成更现实的版本。

#### 1. Special tokens

Special token（特殊 token）不是普通文本片段，而是模型协议中的控制标记，例如：

```text
<|endoftext|>
```

它可以表示一篇文档结束。Tokenizer 必须识别并完整保留它，不能先把它拆成 `<`、`|`、`end` 等普通片段。

若基础 byte 数为 256，训练了 $M$ 次 merge，又有 $S$ 个 special token，粗略词表大小为：

$$
V=256+M+S
$$

具体实现可能保留空洞 ID 或额外 token，因此实际 `n_vocab` 不一定仅由这个简单公式决定。

#### 2. Pre-tokenization

现代 BPE 通常不会先把整份文档当一个无限长 byte 串。它会用正则表达式等规则先切成较小 chunk，再在每个 chunk 内应用 BPE。

作用包括：

- 避免在不希望的边界上合并；
- 限制单次处理的长度；
- 可以并行处理多个 chunk；
- 让常见的空格、单词、数字模式更可控。

课程视频中“带前导空格的词经常是一个 token”，就与 pre-tokenization 和训练数据分布有关。

#### 3. 更快的数据结构

若词表有约 20 万项，merge 数大约也是这个量级。玩具 `encode` 对每条 merge 规则都扫描一次序列。

若暂时把序列长度记为 $T$，merge 数记为 $M$，朴素工作量可粗略看成：

$$
O(MT)
$$

例如 $M=200{,}000,T=1{,}000$，最坏的扫描量级是：

$$
200{,}000\times1{,}000=200{,}000{,}000
$$

约 2 亿个位置检查，显然太慢。实际实现会维护当前存在的相邻 pair、优先级和位置索引，只处理可能生效的 merge；训练长语料时还会分块并行统计。精确复杂度取决于数据结构，不能把所有实现简单说成同一个 $O(\cdot)$。

---

## 11. 为什么 tokenizer 看起来经常“不讲道理”？

### 11.1 空格可能属于后一个词

**【课程】**许多 tokenizer 会把 `" world"`（前面带空格）作为一个 token，而不是把 `world` 单独作为 token。

原因不是模型理解“空格属于这个词”，而是训练文本中“空格 + 常见词”这个 byte 片段频繁出现，BPE 就可能把它合并。

因此：

```text
"hello"          开头的 hello
" hello"         中间、前面带空格的 hello
```

可能是完全不同的 token ID。两个 ID 的数字接近与否也不表示语义接近。

### 11.2 数字可能每几位切一次

**【课程】**不同 tokenizer 对数字的切法不同：可能逐位，也可能每几位一块。

```text
"20260827" → ["202", "608", "27"]   （仅为示意）
```

逐位切分更规则，却会增加 token 数；合并多位数字更短，却可能让算术的位结构变得不统一。这是效率和规律性之间的折中。

### 11.3 同样字数不等于同样 token 数

一个 tokenizer 的 merge 由训练数据频率决定。训练语料中丰富的语言通常有更多常见片段被合并；低资源语言、罕见字符或代码可能被拆得更细。

**【补充例子】**模型 context 上限为 1000 tokens：

- 文本 A 平均 4 bytes/token，可容纳约 4000 bytes；
- 文本 B 平均 2 bytes/token，只能容纳约 2000 bytes。

虽然模型配置写着相同的 1000 tokens，两种文本真正能放入的字符量可能差很多。这会影响成本、延迟和可用上下文，是 tokenizer 的公平性问题之一。

### 11.4 不同 tokenizer 的 perplexity 不能直接比

**【补充】**Perplexity 通常根据“每个 token 的平均负对数概率”计算。Tokenizer 不同，token 单位就不同。

极端地说：

- tokenizer A 把整句话当 1 个 token；
- tokenizer B 把同一句话拆成 20 个 token。

两者的“每 token 难度”不是同一测量单位。课程提到 word tokenizer 的 `UNK` 也会扰乱 perplexity：许多罕见词都被压成同一个易预测的 `UNK`，表面数值可能变好，但信息已经丢失。

跨 tokenizer 比较时，更合理的选择包括换算到相同 byte/character 单位，或直接比较下游任务和总计算预算。

---

## 12. BPE 的价值、限制与 tokenizer-free 梦想

### 12.1 BPE 为什么实用？

BPE 不是从语言学理论推出的完美分词法，而是一个 data-driven heuristic（数据驱动启发式方法）：

- data-driven：规则从训练语料频率中学到；
- heuristic：通常有效，但不保证全局最优或符合词义。

它解决了三个现实问题：

1. 以 byte 为后备，能覆盖任意 UTF-8 文本；
2. 常见片段合并后，序列显著短于纯 byte；
3. 通过规定 merge 次数，大致控制词表大小。

### 12.2 BPE 没有解决什么？

**【补充】**它仍然把 tokenization 与模型训练分成两个阶段，并带来：

- 空格和大小写导致看似奇怪的边界；
- 不同语言压缩率不均衡；
- 错别字的一个小变化可能改变后续多个 token；
- token 边界未必对应词素或语义；
- 固定词表一旦训练完，很难自然适应全新分布；
- tokenizer 的漏洞可能影响字符计数、拼写和数字任务。

所以“BPE 有效”不等于“BPE 是语言的真实原子”。

### 12.3 为什么还不直接使用 bytes？

**【课程】**按 byte 建模很优雅，因为不需要外部 tokenizer。但在今天常见的 Transformer 上，序列太长，计算效率不够。ByT5、MEGABYTE、BLT、T-Free、H-Net 等工作尝试让模型直接或更动态地处理 bytes；课程认为它们很有前景，但尚未在公开前沿模型中全面替代 tokenizer。

### 12.4 替代方案必须满足的两个条件

**【课程】**老师在结尾给出比“是否使用 BPE”更深的要求：

1. **模型需要在有意义的 chunk（块、抽象单位）上工作。**原始 byte 信号太细，文本、视频和 DNA 都需要某种层级抽象。
2. **Chunk 应该可变。**不同区域的信息密度不同，计算量不应机械地平均分给每个 byte。

**【补充类比】**读书时，我们不会让大脑对每一滴墨水投入相同注意力。我们先把墨水看成笔画、字、词和句，再在难句上停留更久。未来 tokenizer-free 模型也许把“怎样分块”变成模型内部可学习的一部分，但仍需要抽象和自适应计算。

---

## 13. 把 BPE 算法完整连成一条流水线

### 13.1 训练 tokenizer

```text
原始训练文档
    ↓ 明确文档边界和 special tokens
UTF-8 bytes
    ↓ pre-tokenization
许多较小 chunk
    ↓ 统计当前相邻 token pair
选择最高频 pair
    ↓ 创建新 ID，保存 bytes 和 merge rank
重复 M 次
    ↓
固定 vocab + 固定 merge 规则
```

### 13.2 用 tokenizer 训练语言模型

```text
训练文本
    ↓ 固定 tokenizer.encode
token IDs
    ↓ embedding lookup
向量序列
    ↓ Transformer
下一个 token 的概率
    ↓ loss + optimizer
更新模型参数
```

Tokenizer 的规则通常已经固定；更新的是语言模型参数。

### 13.3 用语言模型生成文本

```text
用户 prompt
    ↓ encode
输入 token IDs
    ↓ 模型反复预测并选择下一个 ID
输入 IDs + 新 IDs
    ↓ decode
显示给用户的字符串
```

模型一次“生成一个 token”，不等于一次生成一个词。一个 token 可能是空格加单词、半个 emoji、几个数字或一个 byte。

---

## 14. 课堂口头补充与容易遗漏的细节

以下内容能从完整视频中听到，但只看静态代码容易略过：

1. **课程不会真的重造全部技术栈。**“From scratch”仍要选择最高学习价值的部分，否则十周不够。
2. **2026 版增加了 MoE、long context 与 agents 背景。**但课程重点仍是底层机制，而不是追逐每个最新应用。
3. **小规模实验的价值有边界。**机制和效率方法较容易迁移，某种结构或数据配方是否提升效果则必须谨慎外推。
4. **Benchmark 和 profile 是课程心态的一部分。**不能只凭代码看起来更聪明就断言更快。
5. **Scaling law 需要被工程出来。**不稳定的训练 recipe 不会自动给出平滑可预测曲线。
6. **推理正在变得更重要。**聊天、强化学习 rollout、test-time compute、合成数据和评测都依赖推理。
7. **Tokenization 本质上也是系统决策。**它决定序列长度，从而影响 attention、显存、训练费用和推理费用。
8. **视频的 BPE 实现能正确工作但非常慢。**它是为了让变量变化可见，不是生产实现。
9. **视频在 19:10、27:04 和 64:46 左右停下来询问问题。**录音中没有出现需要转写的实质性学生问答，因此本讲不存在被省略的课堂答案。
10. **老师最后的判断不是“BPE 永远正确”。**他希望未来不用再教外置 tokenizer，但替代模型仍必须形成可变的高层 chunk。

---

## 15. 常见误区

### 误区 1：一个 token 就是一个单词

错。Token 可以是一个单词、前导空格加单词、词的一部分、几个数字、一个标点、一个 byte，甚至一个 Unicode 字符的部分 bytes。

### 误区 2：Token ID 越接近，意思越接近

错。ID 是词表标签。`100` 和 `101` 的片段可能毫无语义关系。语义接近性要看模型学到的 embedding，不看 ID 差值。

### 误区 3：Unicode code point 和 UTF-8 byte 是同一个东西

错。`🌍` 有一个 code point 127757，但 UTF-8 使用 4 bytes `[240,159,140,141]` 存它。

### 误区 4：Byte tokenizer 不能处理中文或 emoji

错。它能处理任意合法 UTF-8 字符串。问题是这些字符会占多个 token，使序列长，而不是无法表示。

### 误区 5：压缩率 2.5 表示文件大小正好变成 40%

错。这里的指标是 2.5 UTF-8 bytes/token，用来估计模型序列长度。Token ID 自身的存储、词表和模型参数都没有计入。

### 误区 6：词表越大越好

错。更大的词表通常缩短序列，却增加 embedding/output 参数，让罕见 token 更稀疏。需要权衡。

### 误区 7：BPE 每轮都在原始 bytes 上统计

错。第一轮之后，序列中既有 byte token，也有新建 token。每轮都在**当前 token 序列**上重新统计。

### 误区 8：编码新 prompt 时重新训练 BPE

错。模型绑定固定 tokenizer。新文本只能应用训练时保存的 vocab 和 merge rank。

### 误区 9：最高频 pair 并列时随便选，不影响结果

错。不同 tie-break（并列处理规则）可能生成不同词表。训练和测试必须使用同一组确定规则，才能复现。

### 误区 10：有 byte 后备就永远不可能出现任何未知问题

对普通合法 UTF-8 文本，byte-level BPE 能以基础 bytes 表示，不需要普通 `UNK`。但 special token、非法 byte 序列、实现允许的 ID 范围和解码错误仍需单独处理。

### 误区 11：Tokenizer 只影响输入预处理，不影响模型能力和费用

错。它改变 token 数、context 可容纳的文字量、attention 成本、训练 token 预算、不同语言的相对费用，还会影响拼写和数字等任务。

### 误区 12：The Bitter Lesson 表示算法没有价值

错。课程的解释恰好相反：能高效利用更多资源、随规模继续有效的算法最有价值。

---

## 16. 自测题

先只看题目。能够不用正文完成大部分题，再对照答案，才说明真正掌握。

### 16.1 题目

1. 用一句话说明 language model 的基本任务。
2. 为什么只会调用 API 会限制基础研究的设计空间？举一个具体例子。
3. Mechanics、mindset、intuitions 中，哪两类最容易从课堂小模型迁移到前沿规模？
4. 课程怎样纠正“苦涩的教训 = 只要堆规模”的误解？
5. 列出 CS336 的五个模块，并各用一句话说明目标。
6. 什么是 executable lecture？Lecture 1 为什么没有普通 PDF？
7. 写出 tokenizer 的 round-trip 条件，并解释 $s$ 是什么。
8. `🌍` 的 Unicode code point 是 127757，UTF-8 bytes 是 `[240,159,140,141]`。这两个表示为什么不矛盾？
9. 为什么一个 byte 有 256 种可能值？
10. 字符、byte、word 三种 tokenizer 各有什么主要缺点？
11. 字符串有 30 UTF-8 bytes，编码后有 12 tokens。压缩率是多少？单位是什么？
12. 若长度从 800 byte token 压到 200 BPE token，只比较 $T^2$ attention 位置对，减少多少倍？
13. 若 vocabulary size 从 50,000 增至 100,000，embedding dimension 为 1,024，embedding 参数增加多少？
14. 在课程例子 `the cat in the hat` 中，第 1 轮为什么选择 `(t,h)`？新 ID 是多少？
15. 同一例子做 3 轮 merge 后，token 数、vocab size 和压缩率分别是多少？请写计算。
16. 为什么 BPE 必须在每轮 merge 后重新统计 pair？
17. 对 `[A,A,A]` 合并 `(A,A)`，为什么结果是 `[AA,A]` 而不是 `[AA,AA]`？
18. 训练 tokenizer 与使用 tokenizer 的区别是什么？
19. 为什么 merge 规则必须保存顺序或 rank？
20. 为什么 `o200k_base` 中一个 `🌍` 可以被两个 token 分担，却仍能正确 decode？
21. 什么是 special token？为什么不能把它当普通文本随意拆开？
22. 什么是 pre-tokenization？至少说出两个用途。
23. 若 $M=200{,}000$ 条 merge，长度 $T=1{,}000$，朴素地每条规则扫描整段文本，位置检查量级是多少？
24. 为什么不同 tokenizer 的 per-token perplexity 不能直接横向比较？
25. 即使未来取消外置 BPE，课程认为替代方案仍必须具备哪两个性质？

### 16.2 完整答案

#### 1. Language model 做什么？

它根据已经出现的 token，为下一个 token 分配概率；反复执行就能给整段 token 序列赋概率或生成新序列。

#### 2. 为什么 API 会限制研究空间？

API 隐藏 tokenizer、结构、训练和系统实现。比如输入超过 context 上限时，只改 prompt 无法研究更好的 tokenization、attention、分块或推理系统；你只能在接口允许的选项内行动。

#### 3. 哪两类最容易迁移？

Mechanics（机制）和 mindset（思维方式）。具体数据或结构选择的 intuitions 可能随规模改变，只能部分迁移。

#### 4. 怎样理解“苦涩的教训”？

不是“算法不重要”，而是“能持续利用更多计算和数据、随规模有效的算法最重要”。效果来自资源与效率共同作用，在大规模下浪费一点比例都会非常昂贵。

#### 5. 五个模块是什么？

1. Basics：实现 tokenizer、模型结构和训练；
2. Systems：核算资源，优化 kernel、并行和推理；
3. Scaling laws：从小实验预测大规模设置与结果；
4. Data：评测、收集、清洗、去重、混合数据；
5. Alignment：根据人类、验证器或模型给出的好坏反馈继续改进。

#### 6. 什么是 executable lecture？

讲义本身是可以执行的 Python 程序，执行时显示结构、代码与变量变化。官方给 Lecture 1 的权威材料是 `lecture_01.py`，因此没有另外的静态 `lecture_01.pdf`。

#### 7. Round trip 条件是什么？

$$
\operatorname{decode}(\operatorname{encode}(s))=s
$$

$s$ 是任意合法输入字符串；空格、标点和所有字符都必须原样恢复。

#### 8. Code point 和 bytes 为什么不矛盾？

127757 是 Unicode 给字符 `🌍` 的抽象编号；UTF-8 再按自己的可变长度规则把这个编号存成 4 个 8-bit bytes。它们处在表示流水线的不同层。

#### 9. 为什么是 256？

一个 byte 有 8 个 bit，每个 bit 有 0/1 两种选择：

$$
2^8=256
$$

所以数值范围是 0 到 255，共 256 个值。

#### 10. 三种简单 tokenizer 的缺点是什么？

- Character：词表大且大量码点稀有，序列也不够短；
- Byte：词表只有 256，但压缩率恒为 1 byte/token，序列太长；
- Word：词表可能无界，测试时出现新词，只能丢进 `UNK`。

#### 11. 压缩率是多少？

$$
r=\frac{B}{T}=\frac{30}{12}=2.5\text{ bytes/token}
$$

#### 12. Attention 位置对减少多少倍？

原来：

$$
800^2=640{,}000
$$

后来：

$$
200^2=40{,}000
$$

相除：

$$
\frac{640{,}000}{40{,}000}=16
$$

所以该部分减少 16 倍。长度只减少 4 倍，但平方工作减少 16 倍。

#### 13. Embedding 参数增加多少？

词表多了：

$$
100{,}000-50{,}000=50{,}000
$$

每个 token 多 1,024 个参数：

$$
50{,}000\times1{,}024=51{,}200{,}000
$$

增加 5120 万个参数。这里还没有计算输出层可能增加的参数。

#### 14. 第一轮为什么选 `(t,h)`？

`(t,h)`、`(h,e)`、`(e,space)` 和 `(a,t)` 都出现 2 次，并列最高。`count_adjacent_pairs` 从左到右首次插入字典 key，所以它们的遍历顺序也正是以上顺序；`max(counts, key=counts.get)` 在计数并列时返回先遇到的 `(t,h)`。程序于是创建 ID 256，代表 bytes `th`。

#### 15. 三轮后各是多少？

每一轮都合并两处：

$$
18\to16\to14\to12\text{ tokens}
$$

基础词表 256，每轮新增一个 ID：

$$
V=256+3=259
$$

原文 18 bytes：

$$
r=\frac{18}{12}=1.5\text{ bytes/token}
$$

#### 16. 为什么重新统计？

合并后旧 pair 会消失，新 token 又会与左右邻居形成全新的 pair。旧计数已经不描述当前序列，继续使用会选错。

#### 17. 为什么不能得到两个 `AA`？

三个 `A` 只有三个 token。若得到两个 `AA`，中间那个 `A` 就被同时使用两次，相当于凭空复制数据。左到右的不重叠替换只能先用前两个，留下最后一个，得到 `[AA,A]`。

#### 18. 训练和使用有什么区别？

训练 tokenizer 在大语料上统计频率并产生固定 vocab 与 merge rank；使用 tokenizer 只把新文本转成 bytes，并按这些固定规则编码，不能重新发明 ID。

#### 19. 为什么保存 rank？

相互重叠的 pair 会竞争。例如 `[A,B,C]` 可以先合并 `(A,B)`，也可以先合并 `(B,C)`，结果不同。固定 rank 才能保证编码确定且与训练时一致。

#### 20. 两个 token 怎样还原 emoji？

第一个 token 保存空格和 `f0 9f 8c`，第二个保存 `8d`。解码不是分别把每个 token 强行转成字符，而是先连接 bytes：

```text
f0 9f 8c + 8d = f0 9f 8c 8d
```

完整 byte 序列再进行 UTF-8 decode，就得到 `🌍`。

#### 21. 什么是 special token？

它是模型协议里的控制标记，如 `<|endoftext|>`，可表示文档结束而不是普通可见文本。若随意拆开，模型就收不到单一的控制信号，decode 和文档边界也可能出错。

#### 22. Pre-tokenization 有什么用？

它先把长文本切成较小 chunk，再分别做 BPE。用途包括限制不合理的跨边界合并、加速处理、支持并行，以及让空格/单词/数字模式更可控。答出任意两个即可。

#### 23. 朴素检查量级是多少？

$$
MT=200{,}000\times1{,}000=200{,}000{,}000
$$

约 2 亿次位置检查，所以生产实现必须只维护当前可能生效的 pair 和位置。

#### 24. Perplexity 为什么不能直接比？

它通常按 token 平均，而不同 tokenizer 的一个 token 承载的信息量不同。一个把句子切成 5 块，另一个切成 20 块，“每块难度”不是同一单位；还可能被 `UNK` 人为改变。

#### 25. 替代方案仍需什么性质？

模型必须在比原始 byte 更高层、有意义的 chunk 上工作，并且 chunk 应可变，让不同区域获得不同计算量，也就是抽象化与 adaptive computation。

---

## 17. 术语表

| 术语 | 最简解释 |
|---|---|
| Language model | 根据前文预测下一个 token 概率的模型 |
| Parameter | 训练中学到的模型内部数字 |
| Training | 用数据和错误信号调整参数 |
| Inference | 使用训练好的参数做预测或生成 |
| Token | Tokenizer 定义的整数编号对应单位，不保证是词或字符 |
| Token ID / index | Token 在词表中的整数标签 |
| Tokenizer | 在字符串与 token IDs 之间转换的程序 |
| Encode | 字符串转 token IDs |
| Decode | Token IDs 还原字符串 |
| Round trip | 编码再解码后与原输入完全相同 |
| Vocabulary / vocab | Token ID 到 byte 片段或控制意义的映射 |
| Vocabulary size $V$ | 词表中可用 token 的数量 |
| Bit | 取 0 或 1 的最小二进制位置 |
| Byte | 8 bits，共 256 种可能值 |
| Unicode | 为世界文字字符定义码点等信息的标准 |
| Code point | Unicode 中字符的整数编号 |
| UTF-8 | 把 Unicode 字符编码成 1 到 4 个 bytes 的常用格式 |
| Character tokenizer | 每个 Unicode code point 一个 token |
| Byte tokenizer | 每个 UTF-8 byte 一个 token |
| Word tokenizer | 按单词、标点等单位建立 token |
| `UNK` | 未知词占位 token，多个原词会丢失为同一 ID |
| BPE | 反复合并最高频相邻 token pair 的方法 |
| Pair | 当前 token 序列中相邻的两个 token |
| Merge | 把指定相邻 pair 替换为一个新 token |
| Merge rank | 多条 merge 规则的应用优先级 |
| Compression ratio | UTF-8 bytes 数除以 token 数，单位 bytes/token |
| Pre-tokenization | 在 BPE 前先按规则把长文本切成 chunk |
| Special token | 表示文档边界、角色等协议含义的控制 token |
| Embedding | 把离散 token ID 查成一排可学习浮点数 |
| Context length | 模型一次能处理的 token 数上限 |
| Attention | 让 token 位置根据相关性读取其他位置的信息 |
| FLOP | 一次浮点运算；FLOP/s 才是每秒速度 |
| Kernel | 在 GPU 上执行的函数 |
| HBM | GPU 的高带宽显存，容量大但离计算单元更远 |
| Scaling law | 用小规模实验拟合并外推大规模表现的经验关系 |
| Perplexity | 由 token 概率导出的预测难度指标，通常越低越好 |
| Alignment | 用偏好或评分信号让模型行为更符合目标 |

---

## 18. 视频时间导航

时间来自官方 YouTube 的人工英文字幕。不同播放器若插入片头，可能相差几秒。

| 时间 | 内容 |
|---:|---|
| [00:04](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4s) | 课程与教学团队介绍 |
| [02:23](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=143s) | 2026 版变化：MoE、long context、agents |
| [03:23](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=203s) | 为什么开设“from scratch”课程 |
| [04:51](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=291s) | 语言模型工业化与前沿成本 |
| [05:54](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=354s) | 小模型不一定代表大模型：FLOPs 分布 |
| [06:33](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=393s) | 规模扩大后的行为变化 |
| [07:09](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=429s) | Mechanics、mindset、intuitions |
| [08:21](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=501s) | 经验直觉有时只能靠实验 |
| [09:15](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=555s) | The Bitter Lesson 的正确理解 |
| [11:36](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=696s) | 语言模型历史 |
| [14:52](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=892s) | 开放模型生态 |
| [17:32](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1052s) | 从 fine-tune、prompt、chat 到 agents |
| [19:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1168s) | Executable lecture 是什么 |
| [20:03](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1203s) | 课程强度和适合人群 |
| [23:29](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1409s) | 五次作业与 leaderboard |
| [25:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1500s) | 课程 AI 使用政策 |
| [27:17](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1637s) | 五个课程模块总览 |
| [28:06](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1686s) | Tokenization 在课程中的位置 |
| [29:50](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1790s) | Transformer 架构改进地图 |
| [32:17](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1937s) | 训练：loss、optimizer、稳定性 |
| [33:45](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2025s) | Assignment 1 |
| [34:36](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2076s) | 表达能力、稳定性、效率三角 |
| [35:53](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2153s) | Systems 与 resource accounting |
| [38:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2308s) | GPU kernel、fusion 与数据移动 |
| [40:25](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2425s) | 多 GPU 并行与 collective operations |
| [41:39](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2499s) | Inference：prefill 与 decode |
| [45:12](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2712s) | Scaling laws 与 scaling recipe |
| [48:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2880s) | Hyperparameter transfer 与 predictability |
| [49:17](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2957s) | Chinchilla 式 compute-optimal scaling |
| [53:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3208s) | Evaluation 与 data 模块 |
| [56:40](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3400s) | 数据收集、版权与处理 |
| [60:20](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3620s) | Alignment：PPO、DPO、GRPO |
| [62:54](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3774s) | 整门课的效率主线 |
| [65:06](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3906s) | Tokenization 正式开始 |
| [66:03](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3963s) | 空格、词首和数字的奇怪切分 |
| [67:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4020s) | `o200k_base` 编码与 round trip |
| [67:25](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4045s) | 20 bytes / 8 tokens = 2.5 |
| [68:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4108s) | Character tokenizer |
| [69:45](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4185s) | Byte tokenizer 与 UTF-8 |
| [70:45](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4245s) | Word tokenizer 与 `UNK` |
| [71:58](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4318s) | BPE 的历史和直觉 |
| [73:20](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4400s) | 手算 `the cat in the hat` 的三轮 merge |
| [75:21](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4521s) | 用固定 BPE 编码新文本 |
| [76:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4560s) | 生产实现：速度、special token、pre-tokenization |
| [77:29](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4649s) | 总结与 tokenizer-free 的两个要求 |
| [79:01](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4741s) | 下一讲：resource accounting |

---

## 19. 来源与内容边界

### 19.1 本讲的核心课程来源

1. [Stanford CS336 Spring 2026 官方课程页面](https://cs336.stanford.edu/)：确认课程安排、Lecture 1 标题与材料入口。
2. [官方 executable lecture：`lecture_01.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)：正文结构、代码、课程例子与引用的权威讲义源。Lecture 1 没有静态 PDF。
3. [Stanford Online 官方视频：Lecture 1](https://www.youtube.com/watch?v=JuoVZkPBiKk)：79:15 的完整课堂录像。
4. **字幕来源：**上述 YouTube 视频的人工字幕轨 `English (United States)`，共提取 1509 个带时间片段；用于还原口头补充和时间导航。自动生成的 `English` 轨仅用于确认轨道存在，没有作为主要文字源。

### 19.2 Tokenization 的原始或官方资料

1. [Sennrich, Haddow, Birch, 2016：Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)：BPE/subword 用于开放词表神经机器翻译的原始论文。
2. [OpenAI GPT-2 技术报告](https://cdn.openai.com/better-language-models/language-models.pdf)：byte-level BPE 在 GPT-2 中的说明。
3. [OpenAI `tiktoken` 官方仓库](https://github.com/openai/tiktoken)：课程实际调用的 tokenizer 实现；本笔记用 `o200k_base` 复算 20 bytes、8 tokens 与各 ID 的 byte 片段。
4. [The Unicode Standard 17.0, Chapter 1](https://unicode.org/versions/Unicode17.0.0/core-spec/chapter-1/) 与 [Unicode UTF FAQ](https://www.unicode.org/faq/utf_bom.html)：code point、UTF-8 与 byte 表示的定义来源。

### 19.3 课程原内容与本笔记补充的边界

- 课程原有：从头构建哲学、三类可迁移知识、五模块地图、效率主线、四类 tokenizer 比较、`Hello, 🌍! 你好!`、压缩率、`the cat in the hat` 的 BPE 代码和 3 轮 merge、special token/pre-tokenization/性能要求、tokenizer-free 的两个条件。
- 视频口头补充：前沿训练成本的推测性表述、scale 下 FLOPs 比例变化、课程 AI 政策、predictability 的强调、推理和 RL 系统背景、多个段落中的工程判断。
- 本笔记新增：语言模型概率手算、bit/byte 和 UTF-8 位级模板与手算、$6ND$ 的近似来源和失效边界、最小 Python 语法框、embedding 参数量、$T^2$ 的 16 倍手算、BPE 的 17 个相邻位置与 tie-break、`haha haha`、emoji 跨 token 的 byte 级解释、朴素编码复杂度、跨 tokenizer perplexity 警告、常见误区、自测与术语表。

如果补充解释与课程后续讲次的更精确推导冲突，应以后续官方讲义为准；本讲中的系统与 scaling 公式只是课程全景预告。
