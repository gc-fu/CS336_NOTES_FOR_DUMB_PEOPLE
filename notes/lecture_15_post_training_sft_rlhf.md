# Lecture 15：Post-training、SFT、RLHF、PPO 与 DPO

> CS336 Spring 2026 · Post-training
>
> 官方讲义：lecture_15.pdf，65 页；官方视频：Stanford Online，79:45。
>
> 目标：让只会四则运算的第一次学习者，不看视频也能沿着数据、公式和算法走完整讲。

## 0. 第一次怎么读、材料边界与全讲地图

### 0.1 第一次阅读顺序

1. 先读 §2 的最小词典，分清 prompt、response、token、trajectory 和 batch。
2. 读 §3–§11，理解监督微调（SFT）到底在训练什么。
3. 读 §12–§17，理解“模仿答案”和“优化人类偏好”为何不是一回事。
4. 读 §18–§20，慢慢拆 PPO；第一次不要背公式，只跟着数字例子判断更新方向。
5. 读 §21–§24，理解 Best-of-N、DPO 与其他替代路线。
6. 读 §25–§28，再做 §29 自测；最后用 §31 的视频导航查漏。

§1 是五分钟复习卡，第一次阅读可以先跳过，因为它只给结论，不负责第一次教学。

### 0.2 来源标签

- 【课程内容】：PDF 或课堂主线直接教授的内容。
- 【视频补充】：讲者口头解释、课堂问答或 PDF 没展开的提醒。
- 【补充解释】：为零基础读者补出的中间步骤、小数字例子和类比。
- 【补充】：用一手论文核对或补足的知识。
- 【延伸】：不是掌握本讲所必需，跳过不影响主线。

动态图表、模型名称、数据规模和经验结论是 2026 年课程时点快照，不是永久定律。PDF 逐页覆盖见 §32，视频导航见 §31，外部一手来源见 §33。

### 0.3 一句话地图

~~~text
预训练：学“互联网文本通常怎样接下去”
        ↓
SFT：学“遇到指令时，理想助手应怎样回答”
        ↓
偏好数据/奖励模型：学“两个回答中，人更喜欢哪一个”
        ↓
PPO 等：让模型更常生成高奖励回答，同时别偏离参考模型太远
        ↓
DPO 等：直接从成对偏好更新策略
        ↓
评估：检查帮助性、安全性、真实性、能力和副作用
~~~

核心矛盾：**会续写文本，不等于会按人的意图做事；会模仿示范，也不等于能在候选答案中找到人真正更喜欢的那个。**

### 0.4 稳定目录

- [最低词典与层级](#l15-vocabulary)
- [SFT 模板、mask 与交叉熵](#l15-sft-mask)
- [偏好数据与 Bradley–Terry](#l15-preference-bt)
- [RLHF、KL 与 PPO](#l15-rlhf-ppo)
- [Best-of-N 与 DPO](#l15-dpo)
- [失败模式与决策树](#l15-failures)
- [80 道自测](#l15-questions)
- [80 道答案](#l15-answers)
- [视频导航](#l15-video-nav)
- [PDF 覆盖](#l15-pdf-coverage)
- [来源与视觉核验](#l15-sources)

这些显式 HTML id 不依赖标题自动转写规则，后续微调中文标题也不会让目录失效。

## 1. 五分钟复习卡（首次阅读先跳过）

1. **SFT**：Supervised Fine-Tuning，监督微调；提升示范中 assistant token 的概率。详见 §4–§6。
2. **mask**：0/1 开关。prompt token 常设 0，不计损失；assistant token 设 1，计损失。详见 §5–§6。
3. SFT 主要模仿示范分布；未覆盖的情境和需要搜索候选的任务，模仿未必足够。详见 §12。
4. **偏好数据**：同一 prompt 下比较两个 response，记录 chosen 与 rejected。详见 §13。
5. **Bradley–Terry（BT）模型**：奖励差越大，A 胜过 B 的概率越大，即 $`\sigma(r_A-r_B)`$。详见 §14。
6. RLHF/PPO 五角色：policy 更新；old policy 是本批 rollout 快照；reference 长期约束；reward model 打分；value model 预测回报。详见 §16、§18。
7. **KL 惩罚**防止新策略为钻奖励漏洞而离参考策略太远。单 token 的 log-ratio 可负，但完整 KL 期望非负。详见 §17。
8. **advantage** 是实际结果比 value 预期好多少：正值鼓励动作，负值压低动作。详见 §18。
9. **PPO clip** 限制一次更新把 token 概率推得过猛；正、负 advantage 要分别判断。详见 §19。
10. **DPO** 把 KL 正则化最优策略关系代回偏好损失，不另跑 PPO 的 rollout/value 循环。详见 §22–§23。
11. **Best-of-N**：采样 N 个回答，用 verifier/reward 选最好；成本约乘 N，评分器也可能被钻空子。详见 §21。
12. 失败模式：长度偏差、reward hacking、分布偏移、mode collapse、alignment tax。详见 §25。

<a id="l15-vocabulary"></a>

## 2. 最低前置知识与词典：先把对象和层级分开

### 2.1 六个最小对象

【补充解释】把训练助手想成教一个新客服：

- **prompt（提示）**：用户输入，例如“把 3/4 化成小数”。
- **response（回答）**：模型完整回复，例如“$`3\div4=0.75`$”。
- **token（词元）**：tokenizer（分词器）把文字切成的整数单位；一个汉字不保证正好一个 token。
- **logit（未归一化分数）**：模型对下一个 token 的原始分数；softmax 把多个 logits 变成概率。
- **trajectory（轨迹）**：从 prompt 起的一串状态、动作/token、工具结果和奖励；普通回答可看成简化轨迹。
- **batch（批次）**：一次更新同时处理的多条训练样本。

~~~text
1 batch
├─ prompt 1 → response 1 → token 1, token 2, ...
├─ prompt 2 → response 2 → token 1, token 2, ...
└─ ...
~~~

“每 token 平均”“每 response 平均”“每 prompt 平均”“每 batch 平均”可能给出不同权重。公式必须说明分母。

### 2.2 参数、梯度、学习率

- **parameter（参数）**：训练时可改变的数字。
- **loss（损失）**：把“这次答得多坏”压成一个数；越小通常越好。
- **gradient（梯度）**：参数轻微变化时，loss 往哪边、以多快速度变化。
- **learning rate（学习率）**：每次沿反梯度方向走多大一步。
- **optimizer（优化器）**：根据梯度更新参数的规则，例如 AdamW。

```math
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta\nabla_\theta L.
```

$`\theta`$ 是参数，$`L`$ 是 loss，$`\nabla_\theta L`$ 是梯度，$`\eta`$ 是学习率。若参数为 2，梯度为 0.3，学习率为 0.1：

```math
2-0.1\times0.3=1.97.
```

### 2.3 exp、log、sigmoid、softmax

- $`e\approx2.71828`$ 是自然常数，$`e^x`$ 是指数函数。
- $`\ln x`$ 是自然对数，是 $`e^x`$ 的反函数：$`\ln(e^x)=x`$。
- $`\ln(ab)=\ln a+\ln b`$，所以 token 概率连乘可改成 log 概率相加。
- sigmoid：$`\sigma(z)=1/(1+e^{-z})`$，把实数变到 0 和 1 之间。
- softmax：$`p_i=e^{z_i}/\sum_j e^{z_j}`$，把多个 logits 变成和为 1 的概率。

计算器输入 exp(-1.5) 得约 0.2231；再算 1/(1+0.2231) 得 0.8176。§14 会用它。

### 2.4 训练阶段不是一锅粥

【课程内容，PDF p.1–5】

- **pre-training（预训练）**：海量普通文本上的下一个 token 预测。
- **mid-training（中期训练）**：预训练与最终后训练之间的过渡阶段，可引入长上下文、领域或初步指令数据；边界并不统一。
- **post-training（后训练）**：让 base model 更会遵循指令、满足偏好和安全要求的一组步骤。
- **SFT（Supervised Fine-Tuning，监督微调）**：用示范答案做有监督学习。
- **RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）**：从人类比较中学奖励，再优化策略。

post-training 不是一种单独算法，而是数据、SFT、偏好优化、安全和评估的一整套配方。

## 3. 为什么预训练后还要后训练

### 3.1 预训练目标与用户目标不同

【课程内容，PDF p.2–5】预训练只要求：给定前面 token，预测训练语料接下来出现的 token。它会学到语言和知识，却不保证自动理解聊天协议。

prompt 是“法国首都是什么？”时：

- 互联网式续写可能是“——这是地理测验中的常见问题”。
- 用户想要的也许只是“巴黎”。

两者都像自然文本，但只有后者更符合用户意图。

### 3.2 InstructGPT 的经典三步地图

【课程内容，PDF p.5–8】

1. 人类写理想示范，用它做 SFT。
2. 同一 prompt 采样多个回答，人类排序；用排序训练 reward model（奖励模型）。
3. 用 PPO 优化 policy（策略模型），增加高奖励回答概率，并用 KL 约束它别离 reference 太远。

这是重要历史模板，不是所有 2026 系统的唯一配方。后来还有 DPO、在线偏好优化和可验证奖励等路线。

### 3.3 能力、行为与接口

把模型想成一位读过很多书的人：

- 预训练形成知识与文字模式；
- SFT 教它按某种接口回答；
- 偏好学习教它在多个答案中选人更喜欢的；
- 安全训练教它何时帮助、何时拒绝、怎样拒绝。

阶段会互相影响。不要把名称误当成互不相干的抽屉。

## 4. SFT 数据：从人工写到大规模合成

### 4.1 instruction tuning

【课程内容，PDF p.6–15】**instruction tuning（指令微调）** 是在大量“指令 → 理想输出”上训练。

~~~text
instruction: 把下面句子翻译成英文
input: 今天天气很好。
response: The weather is nice today.
~~~

FLAN 等早期路线把许多 NLP（Natural Language Processing，自然语言处理）任务改写成文字指令，使模型学习“读懂任务说明”，而不只记某个标签。

### 4.2 Self-Instruct / Alpaca 的合成链

【课程内容，PDF p.9–12】人工示范昂贵，可以让强模型生成候选，再过滤、去重并训练目标模型。

~~~text
少量种子任务
  ↓ teacher model 生成
候选指令与回答
  ↓ 规则/模型/人工过滤
SFT 数据
  ↓
student model
~~~

**teacher model（教师模型）** 提供示范或评分；**student model（学生模型）** 是被训练对象。teacher 流畅不等于正确；只检查格式会批量复制错误。

### 4.3 开放与社区数据

【课程内容，PDF p.13–15】OpenAssistant、Tulu 等路线让研究者能检查对话和处理过程。课程还展示 agentic（智能体式）数据：模型可能调用工具、读观察、继续行动。

~~~text
用户：查某城市天气并换成华氏度
动作1：调用天气工具
观察：20°C
动作2：计算 20×9/5+32
最终回答：68°F
~~~

这条 trajectory 监督的不只是最终一句话，还可能包括工具选择、参数和观察后的下一步。

### 4.3.1 p.9–15 的样本到底哪里不同

【课程内容，PDF p.9–15】【补充解释】不能只把这几页读成数据集名字列表：

| 图中对象 | 课件样本能直接看出的差异 | 可以推出 | 不可推出 |
|---|---|---|---|
| FLAN | 传统 NLP 任务被写成短指令；样例包含邮件标题、新闻等 | instruction tuning 能统一多任务接口 | 所有 FLAN 样本都短、都高质量 |
| Alpaca | 健康建议、定义、短代码等较通用的单轮问答 | teacher 合成可快速扩大任务覆盖 | teacher 回答自动正确；50k 规模自动胜过人工 |
| ShareGPT/Vicuna | 人类与在线聊天模型的多轮对话，往往更长、更口语 | 真实聊天可带来多轮风格 | 分享用户代表全部用户；日志没有隐私/许可问题 |
| WizardLM | 从较简单指令演化出更复杂指令 | 数据生成可主动改变难度 | “复杂”一定等于更有用或更正确 |
| Nemotron-SFT-OpenCode-v1 | 代码仓库式提示、AGENTS.md、tool call/多步环境 | agentic SFT 不只监督最终一句话 | 一张样例证明整个数据集正确或无污染 |

课程时间线同时列出 Self-Instruct、OpenAssistant、Tulu3 和 tool use。它支持“数据形态从任务改写→聊天→工具轨迹扩展”，不支持“越新的名字必然越好”。历史/配方一手入口：[Self-Instruct](https://arxiv.org/abs/2212.10560)、[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)、[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)、[WizardLM](https://arxiv.org/abs/2304.12244)、[Tulu 3](https://arxiv.org/abs/2411.15124)。

### 4.4 数据量不是唯一尺度

【视频补充】几百条精心安全样本有时能明显改变行为；几十万条重复、单一、错误样本也可能有害。至少记录 prompt 来源、response 作者/验证者、语言领域难度、去重筛选、安全审核、许可隐私和训练采样权重。

<a id="l15-sft-mask"></a>

## 5. chat template、token 与 loss mask

### 5.1 chat template

【补充解释；连接 PDF p.28–31 的因果语言模型训练】**chat template（聊天模板）** 把结构化消息变成 token 序列。课件 p.28–30 讲 fine-tuning/mid-training，并未逐字给出下面模板；这是为理解实现补的桥。

~~~text
<|system|>你是数学助手。<|end|>
<|user|>2+3=?<|end|>
<|assistant|>5<|end|>
~~~

模型看到的是 token ID，不是聊天气泡。训练与推理模板不同会造成 **format mismatch（格式不匹配）**，例如误认角色或不会停止。

### 5.2 因果预测

**causal（因果）** 表示位置 $`t`$ 只能看左侧。完整序列为

```math
z_1,z_2,\ldots,z_T,
```

模型在位置 $`t`$ 给真实 token 的概率为

```math
p_\theta(z_t\mid z_{<t}).
```

$`\theta`$ 是参数，$`T`$ 是序列总 token 数，$`z_{<t}`$ 是位置 $`t`$ 前的 token。

### 5.2.1 teacher forcing 与推理时误差累积

**teacher forcing（教师强制）**：SFT 训练每个位置时，左边喂的是数据里的真实 token，而不是模型刚才自己生成的 token。设标准回答只有两枚 token：`4`、`。`。

| 阶段/位置 | 模型看到的左侧 | 要预测/产生什么 |
|---|---|---|
| 训练，answer token 1 | prompt | 真实 token 1：`4` |
| 训练，answer token 2 | prompt + **真实** `4` | 真实 token 2：`。` |
| 推理，answer token 1 | prompt | 模型采样 token 1；可能是 `5` |
| 推理，answer token 2 | prompt + **模型生成** `5` | 在错误前缀后继续生成 |

若推理第一步错成 `5`，第二步面对的前缀在这条 SFT 样本里没出现过；错误可能累积。这叫 **exposure mismatch（暴露不匹配）**。边界：它不表示 teacher forcing 一定失败，语言模型预训练和多样数据可能已覆盖许多错误前缀；也不表示 RL 自动修好一切。它只说明训练输入分布与自由生成分布不完全相同。

### 5.3 prompt 为什么常不计 loss

定义 mask：

- $`m_t=0`$：该位置不计 loss；
- $`m_t=1`$：该位置计 loss。

| token | user 标记 | 2+3 | assistant 标记 | 5 | end |
|---|---:|---:|---:|---:|---:|
| mask $`m_t`$ | 0 | 0 | 0 | 1 | 1 |

prompt 仍输入模型、作为回答条件，只是不要求模型模仿用户 token。

### 5.4 多轮与 padding

多轮样本可以训练全部 assistant turns，也可只训练最后一轮；必须写清协议。**padding（填充）** 是为把不同长度样本组成矩形 batch 而补的假 token，也必须 mask 掉。

## 6. SFT 交叉熵：从一个 token 算到一个 batch

### 6.1 单 token 负对数

```math
\ell=-\ln p.
```

- $`p=1`$：loss $`=0`$。
- $`p=0.5`$：loss $`=0.6931`$。
- $`p=0.01`$：loss $`=4.6052`$。

概率越小，正确 token 越意外，惩罚越大。

### 6.2 masked cross-entropy

```math
L_{\mathrm{SFT}}
=-\frac{\sum_{t=1}^{T}m_t\ln p_\theta(z_t\mid z_{<t})}
{\sum_{t=1}^{T}m_t}.
```

- $`T`$：序列 token 数；
- $`m_t\in\{0,1\}`$：mask；
- $`p_\theta`$：模型给真实 token 的概率；
- 分子：被训练 token 的负 log 概率之和；
- 分母：被训练 token 数。

结果是“每个被训练 token 的平均 loss”，无字节或秒单位。

### 6.3 两 token 手算

两个 assistant token 的正确概率是 0.5 和 0.25：

1. $`-\ln0.5=0.6931`$；
2. $`-\ln0.25=1.3863`$；
3. 和为 2.0794；
4. 除以 2 得 $`1.0397`$。

若错把 3 个 prompt token 放进分母，会得 $`2.0794/5=0.4159`$。数更小，却只是口径错。

### 6.4 token-average 与 response-average

batch 中 A 长 2 token，B 长 8 token：

- token-average：A 占 $`2/10=20\%`$，B 占 $`8/10=80\%`$；
- response-average：先各自平均，A、B 各占 50%。

所以“平均 loss”必须说明分母。

### 6.5 shape

若 batch size $`B=2`$，填充长度 $`T=5`$，词表 $`V=100`$：

- input IDs：[2,5]；
- logits：[2,5,100]；
- label：[2,5]；
- mask：[2,5]。

每个位置从 100 个 logits 取真实 label 的 log probability，再只对 mask=1 位置平均。

## 7. 风格、长度与能力：表面变好不等于能力变强

### 7.1 风格会被直接模仿

【课程内容，PDF p.16–18】回答风格包括先结论还是先解释、是否分点、长度、语气和引用方式。SFT 会直接模仿这些模式。

同一道题的两个正确答案：

- A：“4。”
- B：“我们逐步计算：2+2=4，因此答案为 4。”

若数据大量偏向 B，模型会变长；这不证明推理能力更强。

### 7.2 capability 与 preference

- **capability（能力）**：能否完成任务，例如算对或写出能运行的程序。
- **preference（偏好）**：多个可行答案中，人更喜欢哪一个，例如更简洁、更礼貌。

一个冗长错误的答案可能风格讨喜；一个简洁正确答案可能被偏长的评分规则压低。二者相关，但不可互换。

### 7.3 长度是混杂变量

**confound（混杂因素）** 是同时影响某个特征和评分，让我们误判因果的变量。若长答案常被偏好，reward model 可能学成“越长越好”。

检查方法：

1. 在正确性相近的答案中比较长度；
2. 报告不同长度区间的胜率；
3. 使用长度控制评估；
4. 人工审查高奖励但异常冗长的回答。

## 8. SFT 能不能注入知识

### 8.1 三个问题不要混

【课程内容，PDF p.19–21】

1. 模型从未见过某事实，SFT 能否把它写进参数？
2. 模型已有知识但不会在正确问题上调用，SFT 能否教会触发？
3. 模型会答但格式不合要求，SFT 能否修接口？

后两种更像 SFT 擅长的事。第一种可能成功，却也可能造成过拟合、遗忘或幻觉；没有一个脱离模型、数据频率和评估条件的统一结论。

### 8.2 新知识与幻觉

只给一次“虚构城市蓝港的市花是银莲花”：

- 记住该句不代表学会蓝港全部事实；
- 问“蓝港人口多少”时，模型可能按回答风格自信编造；
- 应同时测已给事实、未给事实、近邻干扰和“知道自己不知道”的校准。

【补充】Gekhman 等研究为课程讨论提供一手背景，但其具体实验不能外推成“SFT 一定不能教新知识”。

## 9. 安全 SFT：违规和错误拒绝要一起看

### 9.1 四格表

【课程内容，PDF p.22–27】

- **violation（违规）**：本应拒绝的危险请求却提供不当帮助。
- **false refusal（错误拒绝）**：安全、正常请求也被拒绝。

| 真实请求 | 模型帮助 | 模型拒绝 |
|---|---|---|
| 安全 | 正常帮助 | false refusal |
| 不安全 | violation | 合理拒绝 |

只把 violation 降为 0 的笨办法是拒绝一切，但助手也失去价值，因此两轴必须同时报告。

### 9.2 少量高质量样本的边界

【视频补充】课程用 Tulu 3 等案例讨论少量精心安全样本的显著作用。准确说法是“某些行为边界可被少量高质量数据明显改变”，不是“固定 500 条对任何模型、语言和风险都够”。

至少测试：明显危险、含敏感词但安全、多语言改写、部分可帮助的混合请求、分布外新情境。

### 9.3 拒绝也有质量

合理拒绝可以说明边界并提供安全替代。只用一刀切模板，会教会关键词触发，不保证模型理解上下文。

## 10. mid-training 与两阶段配方

### 10.1 为什么需要过渡

【课程内容，PDF p.28–30】从海量预训练文本突然跳到少量聊天数据，分布变化很大。**distribution shift（分布偏移）** 指训练数据与将来输入的统计规律不同。

mid-training 可引入长上下文、领域文本、文档问答、工具格式或初步指令数据。它像预科班，但没有行业统一边界。

### 10.2 小配方

有 1,000,000 条领域文本和 20,000 条指令样本：

1. 先用领域/长上下文数据适应内容；
2. 再用指令数据教聊天接口。

若直接等权混合，普通文本可能淹没聊天格式；若只做小 SFT，领域覆盖又可能不足。应靠 **ablation（消融实验）**——只改一个因素、其余尽量不变——判断。

## 11. SFT 训练循环与实现边界

### 11.1 骨架伪代码

以下是骨架伪代码，不是完整可运行项目；model、optimizer、batch、tokenizer 和分布式逻辑均为占位。

先认一个 Python 名词：`dataloader` 是按批次交出训练样本的迭代器；这里每次交出一个 `batch`，直到遍历完当前轮数据。

~~~python
for batch in dataloader:
    logits = model(batch.input_ids)       # [B,T] -> [B,T,V]
    token_loss = cross_entropy(
        logits[:, :-1, :],                # 位置t预测t+1
        batch.input_ids[:, 1:],
        reduction="none",
    )                                     # [B,T-1]
    mask = batch.assistant_mask[:, 1:]    # prompt/padding为0
    loss = (token_loss * mask).sum() / mask.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

**forward（前向）** 从输入算 logits/loss；**backward（反向）** 从 loss 反推参数梯度；zero_grad 清上一批梯度。

### 11.2 三个检查

1. mask.sum() 不能为 0。
2. logits 与 label 必须右移一位。
3. 多卡有效 token 数不等时，不能把各卡局部均值不加权平均；应按总有效 token 数加权。

**gradient accumulation（梯度累积）** 是做多个小 microbatch 的 forward/backward 后才更新。4 个 microbatch × 8 条样本对应最多 32 条/step，但有效 assistant token 数仍由长度决定。

## 12. 从 imitation 到 optimization

### 12.1 imitation：只学数据里出现的答案

【课程内容，PDF p.32】**imitation（模仿）** 试图让模型分布 $`\hat p(y\mid x)`$ 接近示范分布 $`p^*(y\mid x)`$：

```math
\hat p(y\mid x)\approx p^*(y\mid x).
```

$`x`$ 是 prompt，$`y`$ 是 response。若示范只给一种正确解法，SFT 会提高那种解法概率，却没有直接比较所有其他候选。

### 12.2 optimization：在自己生成的答案中追求高奖励

【课程内容，PDF p.32–34】奖励优化的抽象目标：

```math
\max_{\pi}\ \mathbb E_{y\sim\pi(\cdot\mid x)}[R(x,y)].
```

- $`\pi`$：策略，即模型给 response 的概率分布；
- $`y\sim\pi`$：从该模型采样回答；
- $`R(x,y)`$：回答的奖励；
- $`\mathbb E`$：按生成概率做加权平均。

若模型只会生成 A、B，概率为 0.7、0.3，奖励为 1、3，则期望奖励

```math
0.7\times1+0.3\times3=1.6.
```

若更新后概率变 0.4、0.6，则

```math
0.4\times1+0.6\times3=2.2.
```

### 12.3 generate–verify gap

【视频补充】有些任务“验证答案”比“从零生成答案”容易。例如从 100 个候选程序里跑测试选对的，可能比一次就写对容易。这个差距支持 Best-of-N 和可验证奖励。

但 verifier（验证器）若不完整，模型会学会通过测试而不真正满足意图；能验证不等于验证无漏洞。

<a id="l15-preference-bt"></a>

## 13. 偏好数据：同一 prompt 下比较

### 13.1 一条样本

【课程内容，PDF p.35–38】

~~~text
prompt x: 解释为什么天空看起来是蓝色
response A: 简洁、正确地解释瑞利散射
response B: 很长但含事实错误
label: A preferred over B
~~~

记 $`y_w`$ 为 winner/chosen，$`y_l`$ 为 loser/rejected。偏好样本是 $`(x,y_w,y_l)`$，不是“给每个答案一个绝对真分数”。

### 13.2 从排序到 pair

若标注员把四个回答排成 A>B>C>D，可以拆成 6 对：

```math
\binom{4}{2}=\frac{4\times3}{2}=6.
```

即 A-B、A-C、A-D、B-C、B-D、C-D。但这些 pair 来自同一排序者和同一 prompt，不是 6 个完全独立观察；统计不确定性不能假装它们彼此独立。

### 13.3 标注协议决定“偏好”的含义

应明确要求标注员考虑：正确性、相关性、清晰度、安全性、引用、长度，遇到平局怎么办。若没有统一 rubric（评分准则），不同人可能在回答“我个人喜欢哪个”，而不是“哪个更符合产品目标”。

### 13.4 数据层级与平均

- 一个 prompt 可有多个 response；
- 一个 pair 有 chosen/rejected 两条 response；
- 一个 response 有多个 token；
- 一个 batch 有多个 pair。

BT loss 通常先按 pair 算，再在 batch 的 pair 上平均；不是按两个回答的 token 总数平均。

## 14. Bradley–Terry 奖励模型：从分数差到概率

### 14.1 sigmoid 把差值变成胜率

【课程内容，PDF p.49–52】reward model 给一对 $`(x,y)`$ 一个标量（单个数）$`r_\phi(x,y)`$。$`\phi`$ 是奖励模型参数。

```math
P(y_w\succ y_l\mid x)
=\sigma(r_w-r_l)
=\frac{1}{1+e^{-(r_w-r_l)}}.
```

只有差 $`r_w-r_l`$ 影响概率：两分数都加 100，胜率不变。因此 reward 的零点本身没有绝对意义。

### 14.2 完整数字例

设 $`r_w=2.0`$，$`r_l=0.5`$：

1. 差值 $`d=2.0-0.5=1.5`$；
2. $`e^{-1.5}\approx0.2231`$；
3. $`1+0.2231=1.2231`$；
4. 胜率 $`1/1.2231\approx0.8176`$；
5. 正确标签的负 log loss：$`-\ln0.8176\approx0.2014`$。

若二者相等，差为 0，$`\sigma(0)=1/(1+1)=0.5`$，loss 为 $`-\ln0.5=0.6931`$。

### 14.3 batch loss

```math
L_{\mathrm{RM}}
=-\frac{1}{M}\sum_{i=1}^{M}
\ln\sigma\!\left(r_\phi(x_i,y_{w,i})-r_\phi(x_i,y_{l,i})\right).
```

- $`M`$：pair 数；
- 每项单位是一个 pair；
- 括号里是无单位的 reward 差；
- sigmoid 输出概率；
- loss 对 $`M`$ 个 pair 平均。

如果一个 batch 有两对，loss 分别 0.2、0.8，则 batch loss 为 $`(0.2+0.8)/2=0.5`$。

### 14.4 BT 模型假设和盲点

BT 把偏好压成单轴标量，并假设胜率只由分数差决定。现实偏好可能循环：有人选 A>B、B>C，却也选 C>A；或者安全、正确、风格无法压成一个稳定轴。reward model 是“按给定数据与协议预测偏好”的模型，不是真理机器。

BT 公式本身只写二选一。若允许 tie（平局），数据契约必须另写：丢弃 tie、按半胜计、或使用显式 tie model，会得到不同 loss 和有效样本数。不能偷偷把 tie 记成 chosen。

## 15. 谁提供反馈：人、专家和模型

### 15.1 标注劳动与代表性

【课程内容，PDF p.39–44】课程讨论标注员招募、工资、培训、人口统计和工作内容。偏好不是脱离人群的自然常数。若只调查某一平台、地区、语言或教育背景的数据劳动者，结果可能不代表其他 **RLHF 标注员/数据劳动者**；这张劳动者调查图本身不是用户代表性调查。

应报告：标注员群体、报酬与时间预算、资格测试、分歧、平局处理、质量控制，以及潜在心理负担。高一致率也可能来自过度僵化的规则。

### 15.1.1 p.39–48 图表逐项读，不把图当装饰

| PDF | 图中对象与数字 | 可推出 | 不可推出/来源边界 |
|---:|---|---|---|
| p.39 | Outlier/ScaleAI **单个平台样本**；年龄 $`n=914`$：18–24 为6%、25–34为25%、35–44为34%、45–54为23%、55–64为10%、65–74为2%；教育 $`n=911`$：本科44%、硕士/专业32%等 | 描述该平台受访数据劳动者的年龄/教育构成 | 不能把单平台样本当全球 RLHF 标注员/数据劳动者，更不能把它误当用户调查；图源标 Oxford Economics，课程未给完整抽样权重 |
| p.40 | 课程截图转述一篇 Business Insider 报道：Handshake AI 项目至少 \$50/h、约3000–4000 freelancers；右图不同专家类别中点约 \$50–\$120/h | 专家标注报酬跨度很大 | 这是[原报道链接](https://www.businessinsider.com/ai-data-labeling-annotators-pay-subject-experts-generalists-gig-workers-2025-12)的二手新闻材料，不是审计过的平台工资总表或因果研究 |
| p.43 | 标注者人口统计表与 OpinionQA 群体分数；红框展示不同宗教群体的模型匹配数值不同 | 标注/目标人群选择可能影响行为 | 不能由表证明“人口统计单独导致全部模型差异”；一手论文：[Santurkar et al. 2023](https://proceedings.mlr.press/v202/santurkar23a.html) |
| p.44 | assertiveness/complexity 热图；格内数是 **crowdsourced annotations 与 expert annotations 的错误检出率差**。例如 factuality 在 assertiveness++ 条件为 $`-22.3\%`$，complexity-- 为 $`-19.8\%`$ | 负值表示在该 style 条件下，众包标注比专家**少检出**这种错误；style 会改变众包—专家差距 | `Baseline` 是一种 style 条件行/列，不是所有格都拿来相减的数值基准；负值也不表示回答更正确。一手论文：[Human Feedback Is Not Gold Standard](https://openreview.net/forum?id=7W3GLNImfS) |
| p.45 | AlpacaFarm 系统级 simulated win-rate 对 human win-rate：Spearman $`0.98`$、$`R^2=0.87`$；成本—agreement 图 | 在这组**系统点**上排序相关很高 | 不能推出逐样本 judge 98% 准；Spearman 是排序相关，$`R^2`$ 是该拟合解释的系统间方差比例；一手论文：[AlpacaFarm](https://openreview.net/forum?id=4hturzLcKX) |
| p.46–47 | UltraFeedback、Zephyr、Tulu3 与 Constitutional AI 的生成—批评—修订/偏好链 | 模型反馈可扩反馈规模 | judge 偏差、提示与自我风格仍会复制；不能当成人类真值 |
| p.48 | 同一例子 SFT(before) 59 tokens、RLHF(after) 243 tokens，输出相似但后者更长；另有 length-reward 散点 | RLHF 可能主要改变长度/细节 | 单例不证明所有 RLHF 都放大 4.12 倍；$`243/59\approx4.1186`$ 只属于该样例；一手论文：[Singhal et al.](https://openreview.net/forum?id=G8LaO1P0xv) |

**Spearman correlation（斯皮尔曼等级相关）** 比较两个排序是否一致；1 表示完全同序。**$`R^2`$（决定系数）** 描述给定回归在这组点上解释的变异比例。两者都不是“每个 pair 判对概率”。

### 15.2 专家何时必要

普通表达偏好可由一般标注员完成；医学、法律、高难数学等正确性判断可能需要专家。专家更贵且人数少，所以常见折衷是：普通标注员覆盖广度，专家抽查高风险样本。

### 15.3 AI feedback 与 Constitutional AI

【课程内容，PDF p.45–48】可以让语言模型批评、修订或比较回答。**Constitutional AI（宪法式 AI）** 用一组文字原则指导自我批评和偏好反馈。

收益：扩大规模、统一格式、减少部分有害内容的人类暴露。风险：judge model 的偏差、长度偏好和错误会被复制；模型同意自己风格不等于人类真偏好。

### 15.4 长度 hacking 小例

若真实质量 $`q\in\{0,1\}`$，错误评分器却用

```math
\hat r=q+0.01\times\text{token数},
```

正确 20-token 答案得 $`1+0.2=1.2`$，错误 200-token 答案得 $`0+2=2`$。优化 $`\hat r`$ 会选错误长答案，这就是 reward hacking 的一个玩具反例。

<a id="l15-rlhf-ppo"></a>

## 16. RLHF 流程与五个模型角色

### 16.1 流程

【课程内容，PDF p.49–53】

~~~text
prompt batch
  ↓ policy 采样 responses（rollout）
reward model 给每条完整response打分
  ↓ 加入与reference的KL代价
value model 估计各位置未来回报
  ↓ 算 advantage
PPO 更新 policy 与 value
  ↓
重复生成—评分—更新
~~~

**rollout（采样展开）** 是让当前策略实际生成一批轨迹。因为模型在改变，新 rollout 的分布也在改变，这称为 online/on-policy 味道。

### 16.2 五个角色：old 不是 reference

| 角色 | 人话 | 是否更新 |
|---|---|---|
| policy $`\pi_\theta`$ | 当前正在学的生成模型 | 更新 |
| old policy $`\pi_{\mathrm{old}}`$ | 产生当前 rollout 的 policy 快照；放在 PPO ratio 分母 | 本批 update 中冻结 |
| reference $`\pi_{\mathrm{ref}}`$ | 冻结的起点，常由 SFT 模型复制 | 通常冻结 |
| reward $`r_\phi`$ | 给完整回答或部分轨迹打分 | PPO 阶段通常冻结 |
| value $`V_\psi`$ | 预测从当前位置起会得多少回报 | 更新 |

old policy 与 reference 不能混：

- old policy 每轮 rollout 后可以刷新，用来问“本次更新相对刚采样时改了多少”；
- reference 通常长期冻结，用来问“当前策略相对 SFT 起点跑了多远”。

reward 是评分信号；value 是对未来回报的预测基线，也不能混。

### 16.3 InstructGPT 式总目标

课程 p.51 的核心可简写为

```math
\mathbb E\left[
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right],
```

另可混入预训练/SFT 项以减轻能力退化。

- $`\beta>0`$：KL 约束强度；
- log-ratio 大：新策略比参考更偏爱该回答，代价更大；
- 这是一条 response 级简写；实现常把 log-ratio 分解到 token。

## 17. KL 正则：为什么不让策略跑太远

### 17.1 KL 的完整定义

```math
D_{\mathrm{KL}}(p\|q)
=\sum_i p_i\ln\frac{p_i}{q_i}.
```

$`p`$ 是新策略分布，$`q`$ 是参考分布，$`i`$ 枚举所有可能 token。它无单位，并满足 $`D_{\mathrm{KL}}\ge0`$，等号在两分布相同时成立。

### 17.2 两 token 手算

令 $`p=(0.75,0.25)`$，$`q=(0.5,0.5)`$：

```math
0.75\ln(1.5)+0.25\ln(0.5)
```

```math
=0.75\times0.4055+0.25\times(-0.6931)
=0.3041-0.1733
=0.1308.
```

第二项为负，但加权总和为正。**单个 sampled token 的 log-ratio 可以负；不要把它误叫完整 KL。**

### 17.3 序列 log-ratio

自回归模型中

```math
\log\pi(y\mid x)=\sum_{t=1}^{T_y}\log\pi(y_t\mid x,y_{<t}).
```

若三个生成 token 的 policy/reference log-ratio 分别 0.2、-0.1、0.3，序列 log-ratio 是 $`0.2-0.1+0.3=0.4`$。若 $`\beta=0.05`$，该采样轨迹的 KL 风格代价为 $`0.05\times0.4=0.02`$。

若报告“每 token 平均”，则 $`0.4/3\approx0.1333`$，平均代价是 $`0.05\times0.1333\approx0.00667`$。sum 与 mean 都可能成为实现口径，但不能用一种训练、用另一种解释数值。response 越长，sum 往往累积更多；mean 会先除有效生成 token 数。prompt/padding 是否排除也必须声明。

### 17.4 $`\beta`$ 的权衡

- $`\beta`$ 太小：policy 容易跑远、钻 reward 漏洞；
- $`\beta`$ 太大：几乎被 reference 锁住，奖励难提升；
- 最佳值依模型、reward 尺度、数据和训练阶段而变，不是通用常数。

## 18. PPO 前置：policy gradient、return、value、advantage

### 18.1 action 是 token

在语言模型 RL 中：

- state：prompt 加已经生成的前缀；
- action：下一枚 token；
- policy：下一个 token 的概率；
- episode/trajectory：整条回答；
- terminal reward：回答结束后的 reward-model 分数；
- 也可每 token 加 KL 代价。

### 18.2 policy gradient 的方向

【补充解释】若某 token 导致比预期更好的结果，就增加其 log probability；更差则降低。抽象项：

```math
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
```

$`s_t`$ 是状态，$`a_t`$ 是 token 动作，$`A_t`$ 是 advantage。它表达方向，不是说直接把概率加 $`A_t`$。

原始 policy-gradient estimator 可能**高方差**：同一 prompt 重采几次，刚好抽到的好/坏回答不同，梯度差很大。另一个问题是 **off-policy reuse（离策略复用）**：数据由 old policy 采样，current 已改变，却还想复用这批数据。importance ratio（重要性比率）

```math
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\rm old}(a_t|s_t)}
```

把“old 多常见、current 多常见”的差别纳入权重。例如 old=0.2、current=0.3，ratio $`=1.5`$。它允许有限复用，不保证策略差很远时仍稳定；极端 ratio 会放大噪声。

### 18.3 return 与 value

**return（回报）** 是从当前时刻往后奖励之和。若结尾 reward 为 3，每 token KL 代价依次 0.1、0.2，则从开头的回报可简化为

```math
3-0.1-0.2=2.7.
```

value $`V(s_t)`$ 预测这个回报。若真实 return 为 3.0，value 预测 2.2，最简单 advantage：

```math
A_t=3.0-2.2=0.8.
```

结果比预期好 0.8，应鼓励；若 return 为 1.7，则 $`A_t=1.7-2.2=-0.5`$，应压低。

### 18.4 为什么需要 baseline

若所有回答奖励都很高，仅看 reward=8 不知道这是不是“比该 prompt 的常态更好”。减 value baseline 后，8 与预期 7.5 比得 advantage 0.5；另一个 prompt 的 3 与预期 1 比得 2。后者虽绝对奖励低，惊喜更大。

【课程内容】实际 PPO 常用 GAE（Generalized Advantage Estimation，广义优势估计）在偏差与方差间折衷。本讲只需掌握“回报减基线”的核心，不把 GAE 的超参数细节强塞成主线。

### 18.5 PG → TRPO → PPO 的完整桥

【课程内容，PDF p.53】

1. **Policy gradient**：方向正确但样本方差高，且 old rollout 复用需要 importance ratio。
2. **TRPO（Trust Region Policy Optimization）**：在 old policy 附近把目标作局部/一阶近似，最大化 ratio-weighted advantage，同时约束
   $`\widehat{\mathbb E}_t[ D_{\rm KL}(\pi_{\rm old}(\cdot|s_t)\|\pi_\theta(\cdot|s_t))] \le\delta.`$
   这是 **old/current trust-region KL**，限制单次更新。
3. **PPO**：用 ratio clipping 做更容易实现的近似简化，不再求解同样的显式受约束优化问题。

这里有两个 KL，目的不同：

| KL | 比较对象 | 时间尺度 |
|---|---|---|
| TRPO trust-region KL | old 与 current | 限制这次更新 |
| RLHF reference KL | current 与长期 frozen reference | 防止长期偏离 SFT 起点 |

把二者都叫“KL 约束”不代表可以互换。

式中：

- $`t`$ 枚举当前采样 rollout 中的 state/action 位置；在语言模型里通常是生成 token 位置；
- $`\widehat{\mathbb E}_t`$ 不是“已知真实世界期望”，而是把这批采样位置上的数相加，再除以位置数的 **empirical mean（经验平均）**；
- $`\delta>0`$ 是允许的最大平均 KL budget（预算），无单位。

小例：三个采样位置的 old/current KL 是 0.006、0.010、0.008：

```math
\widehat{\mathbb E}_t[D_{KL}]
=\frac{0.006+0.010+0.008}{3}
=\frac{0.024}{3}=0.008.
```

若 $`\delta=0.01`$，则 $`0.008\le0.01`$，这批经验平均满足预算。它不保证每个位置都小于 0.01，也不保证未采样位置满足约束。

## 19. PPO ratio 与 clipping：四种情况全手算

### 19.1 ratio 比较新旧概率

PPO 在一批 rollout 上保存 old policy 概率，再更新 current policy。对 token：

```math
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
```

- $`r_t=1`$：概率没变；
- $`r_t=1.3`$：新概率是旧概率 1.3 倍；
- $`r_t=0.7`$：降到 0.7 倍。

若旧概率 0.2、新概率 0.26，则 ratio $`=0.26/0.2=1.3`$。

### 19.2 clipped objective

【课程内容，PDF p.53】

```math
L^{\mathrm{clip}}_t
=\min\left(
r_tA_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right).
```

$`\epsilon`$ 是允许变化带宽。若 $`\epsilon=0.2`$，clip 把 ratio 限到 [0.8,1.2] 后再乘 advantage。优化时希望这个 surrogate objective（代理目标）变大。

代码通常最小化 loss，因此：

```math
\text{policy\_loss}=-\mathrm{mean}(L_t^{\rm clip}).
```

若两个有效 token 的 $`L^{clip}`$ 为 2.4 和 $`-1.6`$，mean $`=(2.4-1.6)/2=0.4`$，policy loss $`=-0.4`$。optimizer 让 $`-0.4`$ 更小，等价让 objective 0.4 更大。

### 19.3 四格手算

| $`A_t`$ | $`r_t`$ | 原项 $`rA`$ | clip 后项 | min | 含义 |
|---:|---:|---:|---:|---:|---|
| 2 | 1.3 | 2.6 | $`1.2\times2=2.4`$ | 2.4 | 好动作涨太多，封顶 |
| -2 | 1.3 | -2.6 | $`1.2\times-2=-2.4`$ | -2.6 | 坏动作还涨，不能获益 |
| -2 | 0.7 | -1.4 | $`0.8\times-2=-1.6`$ | -1.6 | 坏动作降太多，封顶 |
| 2 | 0.7 | 1.4 | $`0.8\times2=1.6`$ | 1.4 | 好动作反降，不能获益 |

为什么负数时容易看错？因为 $`-2.6<-2.4`$，min 选更负的 -2.6。clip 不是简单“先把 ratio 截断，再永远用截断值”。

### 19.4 clip 不等于 KL

- clip 约束 sampled action 在这次 update 的概率比；
- KL 比较整个 token 分布或其采样估计；
- 二者可同时用，但解决的角度不同。

## 20. 一次 PPO iteration 的状态变化

### 20.1 从 prompt 到更新

【课程内容，PDF p.51–53】

1. 从 prompt 数据取 batch。
2. policy 生成 responses，并保存每 token old log-probability。
3. reference 计算同 token 的 log-probability。
4. reward model 给完整回答分数。
5. 每 token 加 KL penalty，构造 rewards/returns。
6. value model 预测每位置 value，算 advantage。
7. 在相同 rollout 上做若干小批次 PPO update。
8. 更新 policy 和 value；reference/reward 通常冻结。
9. 丢弃这批 rollout，使用新 policy 再采样。

### 20.2 一个 batch 的分母

设 2 条 response 长度分别 2、4 token，共 6 个有效 token。可能有三种平均：

- policy loss：对 6 个 token 平均；
- reward：先得到 2 个 sequence scalar，再对 2 条平均；
- value loss：对 6 个有效 token 位置平均。

实现也可能先按 response 平均再按 batch 平均。比较论文或代码时，必须查分母。

### 20.3 资源成本

PPO 可能同时持有或运行 policy、reference、reward、value，并需要 rollout。即使部分权重共享，显存、推理和通信仍比普通 SFT 复杂。DPO 受欢迎的一项原因正是工程流程更接近监督学习。

### 20.4 PPO 骨架伪代码：最大化与最小化不要反

以下是结构伪代码；`sample`、`reward_model`、`gae`、mask 和 optimizer 都是占位：

读代码前先认三项：`no_grad()` 表示块内只算数、不记录参数梯度；`masked_mse` 是只在有效 token 上算均方误差；$`c_v\ge0`$ 是 value loss 的非负权重。代码后再逐项展开。

~~~python
responses, old_logp = sample(policy, prompts)       # old rollout
with no_grad():
    ref_logp = reference(prompts, responses)
    score = reward_model(prompts, responses)        # per response
    old_value = value_model(prompts, responses)     # per token

advantages, returns = gae(score, old_value, ref_logp, old_logp)
for _ in range(update_epochs):
    current_logp = policy.logp(prompts, responses)
    ratio = exp(current_logp - old_logp)
    unclipped = ratio * advantages
    clipped = clip(ratio, 1-eps, 1+eps) * advantages
    policy_objective = masked_mean(min(unclipped, clipped))
    policy_loss = -policy_objective                 # maximize -> minimize negative
    value_loss = masked_mse(value_model(...), returns)
    optimizer.minimize(policy_loss + c_v * value_loss)
~~~

reference KL 可放进 token reward 或 loss，具体实现要声明。old/reference/value/reward 五角色不能因伪代码短就合并。

伪代码词典：

- `no_grad()`：在这个块里只做数值计算，不建立供反向传播使用的梯度图；reference/reward 的参数不会被这一步更新。
- `masked_mse(pred,target)`：只在 mask=1 的有效 token 上算 squared error $`(pred-target)^2`$，再除以有效 token 数。
- $`c_v\ge0`$：value loss 的非负权重；$`c_v=0`$ 表示这行总 loss 不训练 value，值越大表示更重视 value 拟合。
- `...`：为缩短骨架而省略的真实参数，不是可直接运行的 Python。

<a id="l15-dpo"></a>

## 21. 不只有 PPO：Best-of-N、拒绝采样、专家迭代与 RLVR 边界

### 21.0 control token：最小但无保证的替代想法

【课程内容，PDF p.54】**control token（控制 token）** 是放在序列前、告诉模型要生成哪类回答的特殊标记。对同一偏好 pair，可构造两条 SFT 序列：

~~~text
prompt + [GOOD] + chosen response
prompt + [BAD]  + rejected response
~~~

`[GOOD]`/`[BAD]` 是 **prompt 之后、回答之前**的回答控制前缀，不是放在整个 prompt 最前面。训练让模型学“在同一个 prompt 后看到 [GOOD] 时续写 chosen，看到 [BAD] 时续写 rejected”。推理时给：

~~~text
prompt + [GOOD]
~~~

希望模型生成好回答。为什么没有保证？

1. 标签只在离线 pairs 上学，未直接优化模型自己新生成的分布；
2. 模型可能忽略 token，或只学表面风格；
3. chosen/rejected 支持范围外没有闭式最优保证；
4. 错标会把坏行为放进 [GOOD] 条件。

它是课程用来说明“人们尝试去掉 PPO”的最小方案，不是说 control token 一定无用。

### 21.1 Best-of-N

【课程内容，PDF p.54–55】对同一 prompt 从 policy 采样 $`N`$ 个回答，再由 reward model/verifier 选最高分。

若每个独立样本答对概率 $`p=0.3`$，至少一个答对的概率为

```math
1-(1-p)^N.
```

当 $`N=4`$：单个答错概率为 0.7；四个全错为 $`0.7^4=0.2401`$；至少一个正确为 $`1-0.2401=0.7599`$。

只有 verifier 能认出正确答案时，75.99% 才可能变成最终成功率。独立假设也可能不成立：同一模型的错误常相关。

### 21.2 rejection sampling

本讲的 **rejection-sampling fine-tuning（拒绝采样微调）** 常指生成多份候选，丢弃低分答案，把高分答案当新 SFT 数据。它与概率论里为精确保持目标分布的经典 rejection sampling 不完全是同一具体算法。

每个 prompt 生成 8 条、保留最高 2 条时，100 个 prompt 共生成 800 条，最多保留 200 条。生成成本看 800，训练集大小看 200。

### 21.3 expert iteration

【课程内容，PDF p.59】**expert iteration（专家迭代）** 反复执行：

~~~text
当前模型生成多个候选
→ verifier/reward 选好答案
→ 用好答案继续训练
→ 新模型再生成
~~~

它把搜索和学习交替起来。verifier 有漏洞时，循环也会放大漏洞；模型从不生成正确候选时，筛选无法凭空创造它。

### 21.4 RLVR 只作下一讲预告

【视频补充】**RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）** 用可程序验证的结果，如数学答案或单元测试，提供奖励。本讲结尾只预告，完整方法属于后续课程；不可假装本讲已教完 GRPO/RLVR。

可验证只表示“按这套检查器能判”，不表示用户意图被完整验证。程序可能通过弱测试但仍错误。

## 22. DPO 从 KL 正则化目标一步步推导

### 22.1 离线数据契约：先写清平均单位

DPO 使用固定离线数据：

```math
\mathcal D=\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^{M}.
```

- $`M`$：完整离线 dataset 的 pair 总数；
- $`x_i`$：prompt；
- $`y_{w,i}`$：chosen/winner；
- $`y_{l,i}`$：rejected/loser；
- 完整 empirical objective 是 $`M`$ 个 **pair losses 的 mean**，不是按两条回答的 token 数平均：

```math
L_{\mathcal D}=\frac1M\sum_{i=1}^{M}\ell_i.
```

训练时不会每步都装下全部 $`M`$ 对。若当前 mini-batch 有 $`m`$ 对，该 step 计算 $`m^{-1}\sum_{j=1}^{m}\ell_j`$；经过 shuffle/多个 steps 才覆盖完整 dataset。不能把完整数据量 $`M`$ 与当前 mini-batch size $`m`$ 写成同一个分母。

数据通常由某个 behavior policy/多模型候选和标注协议产生。若当前 policy 跑到离线数据支持范围外，DPO 没有新在线标签纠正它。**support（支持集）** 是数据分布给非零概率的区域；下面出现的 log-ratio 还要求 chosen/rejected 在 reference 下概率 $`>0`$，否则 $`\log0`$ 不有限。

有限 logits 的数学 softmax 对词表每项给严格正概率；但计算机有限精度中，先算很小的 `exp(logit)` 或把大量 token 概率相乘，可能被舍入成 0。这叫 **underflow（下溢）**：真实正小数小到存储格式表示不了。防守方法是直接用稳定 `log_softmax` 得每 token log-prob，再对 response token 求和，不先把 sequence probability 乘出来再取 log。

### 22.2 从目标开始

【课程内容，PDF p.56–58】对固定 prompt $`x`$，考虑

```math
\max_{\pi}
\sum_y \pi(y\mid x)
\left[
r(x,y)-\beta\ln\frac{\pi(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
\right],
```

并要求 $`\sum_y\pi(y\mid x)=1`$。

- $`\pi`$：待优化 policy；
- $`\pi_{\mathrm{ref}}`$：冻结 reference；
- $`r`$：reward；
- $`\beta>0`$：参考约束强度；
- $`y`$：所有可能 response。

### 22.3 加入概率和为 1 的约束

**Lagrange multiplier（拉格朗日乘子）** 是在目标中加一项来强制约束。令 $`\pi_y=\pi(y\mid x)`$：

```math
\mathcal J
=\sum_y \pi_y
\left[
r_y-\beta\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}}
\right]
+\lambda\left(\sum_y\pi_y-1\right).
```

所需求导规则是

```math
\frac{d}{du}[u\ln u]=\ln u+1.
```

所以

```math
\frac{\partial\mathcal J}{\partial\pi_y}
=r_y-\beta\left(
\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}}+1
\right)+\lambda.
```

内部最优点坡度为 0：

```math
r_y-\beta\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}}-\beta+\lambda=0.
```

移项：

```math
\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}}
=\frac{r_y}{\beta}+\frac{\lambda-\beta}{\beta}.
```

右边第二项对所有 $`y`$ 相同，把它并入归一化常数 $`Z(x)`$：

```math
\pi_r(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
```

```math
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
```

$`Z`$ 让所有 response 概率和为 1。

### 22.4 反解 reward

取 log：

```math
\ln\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\frac{r(x,y)}{\beta}-\ln Z(x).
```

因此

```math
r(x,y)
=\beta\ln\frac{\pi_r(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\ln Z(x).
```

同一 prompt 的 chosen 与 rejected 共享 $`\beta\ln Z(x)`$，相减后消失：

```math
r_w-r_l
=\beta\left[
\ln\frac{\pi(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\ln\frac{\pi(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right].
```

### 22.5 代回 BT

```math
L_{\mathrm{DPO}}
=-\mathbb E
\ln\sigma\left(
\beta
\left[
\ln\frac{\pi_\theta(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\ln\frac{\pi_\theta(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
\right).
```

DPO 鼓励 policy **相对 reference** 更偏爱 chosen。它的推导依赖固定 reference、KL 正则目标和 BT/logistic 偏好模型；不证明真实人类偏好完全服从这些假设。

### 22.6 nonparametric assumption 与有限网络边界

PDF/视频推导先假设 $`\pi`$ 是 **nonparametric（非参数化）** 的：对每个 response 概率都能自由选择，可表达任意合法分布，因此 KL 正则目标能闭式解出
$`\pi_r\propto\pi_{\rm ref}\exp(r/\beta)`$。

真实 Transformer 是有限参数网络：许多 response 概率由共享参数耦合，只能近似这个闭式分布；optimizer 也未必找到全局最优。故“DPO loss 可算”不等于“有限模型严格达到推导中的 $`\pi_r`$”。

## 23. DPO 数字例与长度问题

### 23.1 四个概率逐项算

| | chosen $`y_w`$ | rejected $`y_l`$ |
|---|---:|---:|
| reference 概率 | 0.4 | 0.2 |
| policy 概率 | 0.5 | 0.1 |

chosen log-ratio：

```math
\ln(0.5/0.4)=\ln1.25=0.223143551.
```

rejected log-ratio：

```math
\ln(0.1/0.2)=\ln0.5=-0.693147181.
```

margin：

```math
0.223143551-(-0.693147181)=0.916290732.
```

设 $`\beta=0.5`$，DPO logit 为 $`0.5\times0.916290732=0.458145366`$。于是

```math
\sigma(0.458145366)=0.612574113,\qquad
-\ln0.612574113=0.490085343.
```

若 policy=reference，两个 log-ratio 都为 0，loss 为 $`-\ln0.5=0.6931`$。$`0.490085343<0.6931`$，所以本例 policy 相对 reference 的 chosen/rejected 排序更符合标签。

### 23.2 response 概率是 token 条件概率乘积

chosen 有两个 token，条件概率 0.5、0.4：

```math
\pi(y_w\mid x)=0.5\times0.4=0.2,
```

```math
\ln\pi(y_w\mid x)=\ln0.5+\ln0.4
\approx-0.6931-0.9163=-1.6094.
```

序列越长，log-probability 通常加得越负，所以一些 DPO 变体讨论长度归一化；但归一化会改变目标，不是无代价修复。

### 23.3 p.58 梯度权重：排错 pair 更新更大

令 DPO margin/logit 为 $`z`$，loss $`-\log\sigma(z)`$ 对 margin 的权重大小含 $`\sigma(-z)`$。直觉：

- 已排错：$`z=-2`$，$`\sigma(-z)=\sigma(2)=0.8808`$，更新权重大；
- 已排对且很自信：$`z=+2`$，$`\sigma(-2)=0.1192`$，更新权重小。

两者梯度方向都提高 chosen relative log-prob、降低 rejected relative log-prob；差别是当前排得越错，修正越大。这是 PDF p.58 “higher weight when reward estimate is wrong”的含义，不是每个 pair 等权。

### 23.4 $`\beta`$ 两种看法必须调和

$`\beta`$ 不是 optimizer learning rate。

1. 在 KL-regularized RL 目标
   $`\mathbb E[r]-\beta D_{\rm KL}(\pi\|\pi_{\rm ref})`$
   中，$`\beta`$ 越大，reference 约束越强；闭式最优里的 $`e^{r/\beta}`$ 越平，最优 policy 越靠 reference。
2. 若**固定当前 policy/reference log-ratios 不变**，只观察 DPO logit $`z=\beta\Delta`$，增大 $`\beta`$ 会把这个既定 margin 乘大。
3. 但真实训练比较不同 $`\beta`$ 时，所得最优 policy 也会变化；不能把第2点的“固定 margin 局部观察”冒充第1点的完整优化实验。

learning rate 只控制参数一步走多大，不定义最终 KL 目标。

### 23.5 DPO 骨架伪代码

读代码前先认三个名字：`dataloader` 每次从完整 $`M`$-pair 数据集交出一个 $`m`$-pair mini-batch；`no_grad()` 让冻结的 reference 只算数、不记录梯度；`logsigmoid(u)` 稳定计算 $`\log\sigma(u)`$。

~~~python
for pairs in dataloader:                 # x, chosen, rejected
    pi_w = policy.sequence_logp(x, chosen)
    pi_l = policy.sequence_logp(x, rejected)
    with no_grad():
        ref_w = reference.sequence_logp(x, chosen)
        ref_l = reference.sequence_logp(x, rejected)
    margin = (pi_w - ref_w) - (pi_l - ref_l)
    pair_loss = -logsigmoid(beta * margin)
    loss = pair_loss.mean()              # divide by current mini-batch size m
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

`sequence_logp` 是生成 token 条件 log-prob 的和，并 mask prompt/padding。reference 冻结；chosen/rejected 顺序不能反。

这里：

- `dataloader`：把完整 $`M`$-pair dataset 打乱并逐次交出当前 $`m`$-pair mini-batch 的迭代器；
- `no_grad()`：reference 只算 log-prob，不建立它的梯度图；
- `logsigmoid(u)`：数值稳定地直接算 $`\log\sigma(u)`$，避免先算一个极接近 0 的 $`\sigma(u)`$ 再取 log；
- `.mean()`：本 step 除以当前 $`m`$，不是声称当前内存里有全部 $`M`$。完整 empirical objective 的 $`1/M`$ 由遍历数据得到。

## 24. DPO 变体与经验结论边界

### 24.1 为什么变体很多

【课程内容，PDF p.59–61】变体可能处理 response 长度、偏好噪声、reference、margin、reward 尺度和 offline 分布差异。没有脱离模型、数据和评估协议的“永久最佳方法”。

### 24.2 长度归一化直觉

总 log-probability 都是 -20：

- 长度 10：每 token 平均 $`-20/10=-2`$；
- 长度 20：每 token 平均 $`-20/20=-1`$。

按总和和按平均会给不同排序。长度归一化减少 token 数效应，也可能引入新偏好，必须实测。

### 24.3 如何公平读图

某图中一个方法胜出，只说明在该模型、pair 数据、超参数和 judge 下胜出。至少控制 base/reference、训练 token、学习率搜索、生成长度、评估器和置信区间。

<a id="l15-failures"></a>

## 25. 失败模式、评估与停止条件

### 25.1 reward hacking

**reward hacking（奖励钻空子）**：policy 找到令代理 reward 高、但不满足真实目标的行为。例如评分器偏长，模型堆文字；测试弱，程序针对样例。

防守：人工盲审、独立评估器、分布外测试、长度分层、对抗审计，并限制离 reference 的距离。

### 25.2 overoptimization

【课程内容，详细图见 PDF p.63；p.62 是风险转场】早期 proxy reward 与真实质量一起升；继续优化后 proxy 仍升，真实质量却可能跌。

【补充解释：下表是便于手算的玩具数，不是课件图的原始坐标。】
| 阶段 | reward model | 人工质量 |
|---|---:|---:|
| 起点 | 0.50 | 0.55 |
| 中段 | 0.75 | 0.72 |
| 后段 | 0.92 | 0.60 |

只看 reward 会错选后段。**checkpoint（检查点）** 是某一步保存的模型状态。

### 25.3 mode collapse

【课程内容，PDF p.64】**mode collapse（模式坍缩）** 是生成分布失去多样性，许多 prompt 都得到相似模板。对同一 prompt 多次采样，可检查不同答案数、重复片段、语义聚类和成功率。

### 25.4 calibration

【课程内容，PDF p.64】**calibration（校准）** 指置信度与长期正确率匹配。若模型对 100 道题都说“80% 确信”，理想情况约 80 道正确。更自信的语气不等于更正确。PDF p.63 是 overoptimization 的详细 proxy-vs-gold 图；p.64 才把 calibration 与 mode collapse 放在同一风险页。

### 25.5 alignment tax 与分布偏移

**alignment tax（对齐税）** 指目标行为改善时，某些原能力或效率下降；不是固定税率。要对 base、SFT、偏好模型同时测能力、安全、帮助、长度和延迟。

离线 pair 来自旧 policy，新 policy 可能生成完全不同文本；reward model 会外推失败。需要新采样/标注、保守更新和独立测试。

## 26. 实践 recipe 与数据卡

### 26.1 决策树

~~~text
模型不会按格式回答？
└─ 先做高质量SFT，检查chat template与mask

模型能答，但多个答案中常选错？
├─ 有可靠自动verifier：Best-of-N / rejection / RLVR方向
└─ 只能靠偏好：收pair数据
   ├─ 想要简单离线流程：DPO类
   └─ 需要在线探索且有工程预算：reward model + PPO类

任何路线
└─ 独立评估长度、安全、事实、能力、校准、多样性与成本
~~~

### 26.2 SFT 数据卡

| 字段 | 必答问题 |
|---|---|
| 来源 | prompt、response 从何而来？ |
| 作者 | 人、专家还是模型？谁复核？ |
| 模板 | 角色特殊 token 怎样放？ |
| mask | 哪些 token 计 loss？分母是什么？ |
| 混合 | 各来源权重与总 token？ |
| 风险 | 隐私、许可、安全、群体偏差？ |
| 评估 | 能力、风格、拒绝、长度是否分开？ |

### 26.3 偏好数据卡

| 字段 | 必答问题 |
|---|---|
| 候选生成 | 哪个 policy、temperature、长度上限？ |
| 协议 | 正确、帮助、安全、风格如何权衡？ |
| 标注员 | 人群、专家性、报酬、分歧？ |
| pair | 排序拆 pair 后相关性如何处理？ |
| reference | DPO/RLHF 用哪个冻结模型？ |
| 审计 | 长度、位置、模型自偏好？ |
| holdout | 是否有独立测试？ |

### 26.4 最小实验顺序

1. 小规模检查模板、mask、loss 和生成。
2. 固定评估集，训练 SFT baseline。
3. 做数据源 ablation，不只堆总量。
4. reward model 做 held-out pair accuracy、长度分层与人工审查。
5. 用短训练和多个 $`\beta`$ 做 sweep（超参数扫描）。
6. 保存多个 checkpoint，按真实评估而非 reward 峰值选择。
7. 扩大前复核成本、隐私和安全。

## 27. 常见误区：错误说法 → 原因 → 正确说法

1. **“预训练模型毫无能力。”** 目标与能力不是一回事；它可很强，只是接口未必合用。
2. **“post-training 就是 RLHF。”** 还包括 SFT、安全、DPO、评估等。
3. **“SFT 样本越多越好。”** 重复、错误、单一数据会放大偏差。
4. **“prompt mask=0，所以模型看不到 prompt。”** mask 只控制 loss；prompt 仍是条件。
5. **“mask 掉 prompt 就不用右移 label。”** 因果模型仍用位置 $`t`$ 预测 $`t+1`$。
6. **“平均 loss 可直接比较。”** token/response/batch 分母可能不同。
7. **“长答案表示能力强。”** 可能只是风格或 judge 偏差。
8. **“SFT 绝不能教知识。”** 结论过强，取决于模型和数据。
9. **“降低违规率只要拒绝更多。”** 会提高 false refusal。
10. **“固定 500 条安全样本对任何模型都够。”** 案例有边界。
11. **“偏好标签是客观真理。”** 它依协议、标注员和场景。
12. **“四回答排序拆 6 pair，就是 6 个独立样本。”** 它们共享 prompt 和标注。
13. **“reward 10 是 reward 1 的十倍好。”** BT 用差值，尺度无绝对单位。
14. **“reward model 就是 value model。”** 前者评分，后者预测回报。
15. **“负 log-ratio 说明 KL 为负。”** 单项可负，完整 KL 期望非负。
16. **“KL 越小越好。”** 太小可能无法改善行为。
17. **“advantage=reward。”** advantage 通常是 return 减 value。
18. **“PPO 永远先截 ratio 再乘 A。”** objective 取原项和截断项的 min。
19. **“clip 保证模型绝不大改。”** 它只约束代理目标中的 sampled ratio。
20. **“PPO 只涉及一个模型。”** 还有 reference、reward、value。
21. **“Best-of-N 成功率一定是 $`1-(1-p)^N`$。”** 需独立且 verifier 能选对。
22. **“后训练 rejection sampling 必然精确保持目标分布。”** 它常只是筛高分数据。
23. **“原始 DPO 不需要 reference。”** 原始目标显式用 reference log-ratio。
24. **“DPO 完全等同 PPO。”** 假设、数据和流程不同。
25. **“DPO $`\beta`$ 就是学习率。”** 一个调参考约束，一个调参数步长。
26. **“response 概率是 token 概率平均。”** 它是条件概率乘积，log 后求和。
27. **“长度归一化总能修偏差。”** 它会改变目标。
28. **“reward 一直升就一直变好。”** 可能 overoptimize。
29. **“回答更统一就是更对齐。”** 可能 mode collapse。
30. **“更自信就是更正确。”** 要检查 calibration。
31. **“单一 benchmark 赢就普遍更优。”** 需要控制条件与不确定性。
32. **“可验证奖励就是目标完全可验证。”** verifier 可能不完备。
33. **“安全和帮助性是一条单轴。”** 二者可以同时好或同时差。
34. **“后训练只改变风格。”** 它也可能改变能力、遗忘与分布。
35. **“本讲已完整讲完 RLVR/GRPO。”** 本讲结尾只预告。

## 28. 公式卡

| 名称 | 公式 | 人话 |
|---|---|---|
| SFT | $`L=-\sum_t m_t\ln p_t/\sum_t m_t`$ | 有效 assistant token 的平均负 log 概率 |
| BT | $`P(w\succ l)=\sigma(r_w-r_l)`$ | reward 差变成胜率 |
| RM loss | $`-\ln\sigma(r_w-r_l)`$ | chosen 胜率低则 loss 高 |
| KL | $`\sum_i p_i\ln(p_i/q_i)`$ | 新分布相对参考分布的平均差 |
| 期望奖励 | $`\sum_y\pi_yR_y`$ | 按生成概率加权奖励 |
| advantage | $`A_t=G_t-V(s_t)`$ | 实际回报比预测好多少 |
| PPO ratio | $`r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)`$ | 新旧 token 概率倍数 |
| PPO clip | $`\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)`$ | 限制有利方向大步 |
| KL 最优策略 | $`\pi_r=\pi_{\mathrm{ref}}e^{r/\beta}/Z`$ | reference 被 reward 指数倾斜 |
| DPO margin | $`\beta[\ln(\pi_w/\pi_{\mathrm{ref},w})-\ln(\pi_l/\pi_{\mathrm{ref},l})]`$ | 相对 reference 更偏 chosen 的程度 |
| Best-of-N | $`1-(1-p)^N`$ | 独立候选至少一个成功 |

<a id="l15-questions"></a>

## 29. 自测题（80 题）

> 每题都要求操作、解释或诊断。不要只对照答案背句子；先在纸上写出分母、方向和层级。

### A. 对象、SFT 与实现（1–20）

1. 【分类】把“用户问题、完整助手回答、回答中的一个词元、含工具观察的全过程、一次并行处理的 8 条样本”分别归到 prompt、response、token、trajectory、batch。
2. 【手算】参数 2、gradient 0.3、learning rate 0.1，更新后是多少？
3. 【手算】正确 token 概率为 1、0.5、0.25 时，负 log loss 分别是多少？
4. 【填表】给 token 序列 user标记/问题/assistant标记/答案/end，填一份只训练答案与 end 的 mask。
5. 【判断解释】prompt mask=0 是否表示模型 forward 时看不到 prompt？
6. 【手算】两个 assistant token 概率 0.5、0.25，算 masked token-average SFT loss。
7. 【错误诊断】第6题若有3个 prompt token，有人把 loss 和除以5，得到0.4159。错在哪里？
8. 【手算】两条 response 长2和8 token。token-average 时各占多少权重？response-average 呢？
9. 【shape】$`B=3,T=7,V=50`$，写 input IDs、logits、label、mask shape。
10. 【判断解释】为什么 logits 与 labels 要右移一位？再用两枚回答 token 说明 teacher forcing 在训练第2枚 token 时喂什么、推理第2枚 token 时喂什么，以及第一枚推错后为何可能误差累积。
11. 【错误诊断】一个 batch 的 assistant mask 全为0，直接除 mask.sum() 会发生什么？如何防守？
12. 【手算】两张卡有效 token 分别2和8，local mean loss 分别1和3。正确 global token mean 是多少？简单平均 local mean 又是多少？
13. 【手算】4个 microbatch、每个8条，最多多少条样本后做一次 optimizer step？为何仍不能推出有效 token 数？
14. 【分类】天气工具调用例中，哪些是 prompt、action、observation、final response？
15. 【设计】列出 SFT 数据卡至少五个字段。
16. 【判断解释】训练数据全是长解释，模型变长，能否据此说能力变强？
17. 【错误诊断】评分器偏长时，为什么“reward高”可能只是长度 confound？
18. 【分类】把“SFT修输出格式、教会触发已有知识、写入完全新事实”分成较稳妥与需谨慎验证两类。
19. 【填表】安全请求/危险请求 × 模型帮助/拒绝的四格，填正常帮助、false refusal、violation、合理拒绝。
20. 【判断解释】某模型用500条安全样本改善，能否推出任何模型用500条都够？

### B. 配方、模仿与偏好数据（21–35）

21. 【判断解释】mid-training 为什么像预科班？它和 SFT 的边界是否行业统一？
22. 【设计】写一个只检验“是否需要 mid-training”的两-run ablation。
23. 【手算】策略只生成 A/B，概率0.7/0.3，奖励1/3，期望奖励是多少？概率改0.4/0.6后呢？
24. 【分类】SFT imitation 与 reward optimization 分别直接使用示范 response 还是当前 policy 采样？
25. 【判断解释】为什么“验证候选”可能比“从零生成”容易？验证器有什么边界？
26. 【手算】同一 prompt 的5个回答完全排序，可拆多少 pair？
27. 【判断解释】第26题的10个 pair 是否是10个完全独立观察？
28. 【设计】偏好 rubric 至少写出四个比较维度和一种平局协议。
29. 【分类】pair-average、response-average、token-average 的分母分别是什么？
30. 【错误诊断】把 chosen/rejected 写反，会怎样推动 reward model 与 DPO？
31. 【手算】BT 中 $`r_w=r_l`$，chosen 胜率和 loss 各是多少？
32. 【手算】$`r_w=2,r_l=0.5`$，用 $`e^{-1.5}=0.2231`$ 算胜率与 loss。
33. 【推导】把第32题两 reward 都加100，证明胜率不变。
34. 【手算】两个 pair 的 RM loss 为0.2、0.8，batch pair-average 是多少？
35. 【判断解释】BT reward 10 能否解释成 reward 1 的“十倍质量”？

### C. 反馈、RLHF、KL 与 PPO（36–58）

36. 【判断解释】A>B、B>C、C>A 的循环偏好说明 BT 单轴假设有什么限制？
37. 【手算】评分器 $`\hat r=q+0.01L`$。正确20-token回答 $`q=1`$ 与错误200-token回答 $`q=0`$ 各得多少？
38. 【设计】标注数据应报告哪四类劳动/代表性信息？
39. 【判断解释】AI feedback 能扩规模，为什么仍不能代替独立人类验证？
40. 【分类】policy、old policy、reference、reward、value 五个角色分别做什么？本批 update 谁更新？
41. 【判断解释】old policy 与 reference 为什么不是同一个概念？
42. 【判断解释】reward 与 value 为什么不是同一个模型角色？
43. 【手算】$`p=(0.75,0.25),q=(0.5,0.5)`$，完整计算 $`D_{\mathrm{KL}}(p\|q)`$。
44. 【判断解释】第43题第二项为负，为什么不能说 KL 为负？
45. 【手算】三个 token log-ratio 为0.2、-0.1、0.3，算 sequence sum 和每-token mean。
46. 【手算】第45题 $`\beta=0.05`$，按 sum 和按 mean 的代价各是多少？
47. 【手算】终点 reward=3，两个 token KL 代价0.1、0.2，简化 return 是多少？
48. 【手算】真实 return=3、value=2.2，advantage 是多少？return=1.7时呢？
49. 【判断解释】绝对 reward 8 不一定比 reward 3 的 advantage 大，为什么？
50. 【手算】old probability=0.2，new probability=0.26，PPO ratio 是多少？
51. 【手算】$`\epsilon=0.2,A=2,r=1.3`$，算 PPO 两项和 min。
52. 【手算】$`\epsilon=0.2,A=-2,r=1.3`$，算两项和 min。
53. 【手算】$`\epsilon=0.2,A=-2,r=0.7`$，算两项和 min。
54. 【手算】$`\epsilon=0.2,A=2,r=0.7`$，算两项和 min。
55. 【手算】$`\epsilon=0.1`$ 时 clip 区间是什么？ratio=1.25会被截成多少？
56. 【错误诊断】为什么“永远先 clip ratio，再乘 advantage”不是 PPO 公式？
57. 【分类+手算】TRPO trust-region KL、PPO clip 与 RLHF reference KL 分别比较/约束什么？TRPO 中 $`\widehat{\mathbb E}_t`$、$`t`$、$`\delta`$ 各是什么？KL 为 0.006、0.010、0.008，$`\delta=0.01`$ 时是否满足经验平均预算？
58. 【排序】把 rollout、reward/reference/value 计算、advantage、PPO update、刷新 rollout 按先后排序。

### D. Best-of-N、DPO 与替代路线（59–70）

59. 【手算】独立单样本成功率0.3，Best-of-4 至少一中概率是多少？
60. 【判断解释】为什么第59题公式不保证真实 Best-of-4 成功率？
61. 【手算】100 prompts，每个生成8条、保留2条：共生成多少、最多留下多少？
62. 【判断解释】expert iteration 中模型从不生成正确候选，筛选器能否凭空补出？
63. 【分类】RLVR 在本讲是完整主讲还是下一讲预告？另写 control-token pair 的两条训练序列和推理前缀；为什么都没有完整目标保证？
64. 【推导】从 $`\pi_r=\pi_{\rm ref}e^{r/\beta}/Z`$ 反解 reward，并说明 chosen-rejected 相减时什么消失。
65. 【手算】reference chosen/rejected=0.4/0.2，policy=0.5/0.1，$`\beta=0.5`$，算两个 log-ratio、margin、DPO logit、概率和 loss。
66. 【手算】policy=reference 时，DPO margin、概率、loss 各是多少？
67. 【手算】两-token response 条件概率0.5、0.4，算 sequence probability 与 log-probability。
68. 【判断解释】为什么长 response 的 log-probability 总和通常更负？长度归一化为何不是免费修复？
69. 【分类】DPO $`\beta`$ 与 optimizer learning rate 分别控制什么？为什么“固定 margin 看 logit”与“比较不同 $`\beta`$ 的最优 policy”不可混？
70. 【判断解释】DPO 推导依赖哪些关键假设、support 条件与平均单位？完整 dataset 有 $`M`$ 对、当前 mini-batch 有 $`m`$ 对时，两种 mean 各除什么？为何不能说它与 PPO 在所有设置完全等价？

### E. 失败诊断与综合设计（71–80）

71. 【填表】proxy reward/人工质量从起点0.50/0.55，中段0.75/0.72，后段0.92/0.60。按真实质量选哪个 checkpoint？为什么？
72. 【判断解释】所有回答都用同一长模板，可能是哪种失败？怎样测？
73. 【手算】100道“80%确信”的题只对60道，实际正确率多少？是否校准？
74. 【判断解释】alignment tax 是固定税率或必然发生吗？应怎样测？
75. 【判断解释】offline pair 来自旧 policy，新 policy 分布改变后有什么风险？
76. 【决策】模型不会遵守格式、有可靠 verifier、只有主观偏好三种情形，分别先选什么路线？
77. 【设计】列出偏好数据卡至少五项。
78. 【手算】两条 response 长2和8 token，若各 token loss 都分别为1和3，token-average 与 response-average 各是多少？
79. 【错误诊断】reward model 分数持续上升、人工质量下降，给出失败名与三项防守。
80. 【综合设计】为一个数学助手写一条从 SFT 到偏好优化再到独立评估的最小可审计方案，至少八步，并明确每步的数据层级与平均分母。

<a id="l15-answers"></a>

## 30. 自测答案

### A. 对象、SFT 与实现答案（1–20）

1. 用户问题=prompt；完整助手回答=response；一个词元=token；工具观察全过程=trajectory；一次并行的8条样本=batch。

2. 更新式是 $`2-0.1\times0.3`$。先算 $`0.1\times0.3=0.03`$，再算 $`2-0.03=1.97`$。

3. $`p=1`$：$`-\ln1=0`$。$`p=0.5`$：$`-\ln0.5=0.6931`$。$`p=0.25`$：$`-\ln0.25=1.3863`$。

4. 一份合格 mask 是 $`0,0,0,1,1`$。三个0对应 user标记、问题、assistant标记；两个1对应答案与 end。若项目选择不训练 end，最后也可为0，但协议必须声明。

5. 不是。prompt token 仍进入 forward，影响 assistant 概率；mask=0 只让该位置的预测错误不进入 SFT loss。

6. 两项 loss 为0.6931、1.3863；和为2.0794；有效 token 数2；所以 $`2.0794/2=1.0397`$。

7. 分母应是 $`\sum m_t=2`$，即有效 assistant token 数，不是整条序列5个 token。0.4159 是用错误分母稀释出的数。

8. token-average：A $`2/(2+8)=20\%`$，B $`8/10=80\%`$。response-average：先各自平均，两条各 $`1/2=50\%`$。

9. input IDs [3,7]；logits [3,7,50]；label [3,7]；mask [3,7]。

10. 因果模型在读到前缀位置 $`t`$ 后预测下一枚 $`t+1`$。若不右移，会把当前位置输入与当前位置 label 错位成“看见答案再预测自己”。设 gold 回答是 token1=`4`、token2=`。`：训练预测 token2 时喂的是 prompt+**gold `4`**；推理预测 token2 时喂的是 prompt+**模型自己生成的 token1**。若第一步错成 `5`，第二步就在 prompt+`5` 这个不同前缀上继续，错误可能累积；这就是 teacher forcing 的 exposure mismatch 边界。

11. 分母为0，会得到除零、NaN 或异常。应在数据构造时保证至少一个 assistant token，并在训练中断言 mask.sum()>0；无有效标签的样本应丢弃或单独处理。

12. 总 loss 和是 $`2\times1+8\times3=26`$，总 token 10，所以正确 global mean $`=26/10=2.6`$。简单平均 local means 得 $`(1+3)/2=2`$，错误地让2-token卡与8-token卡等权。

13. $`4\times8=32`$ 条。每条 response 长度和 mask 可不同，所以32条不能确定 assistant token 总数。

14. 用户要求是 prompt；调用天气工具与计算是 actions；20°C 是 observation；68°F 是 final response。

15. 例：来源、作者/teacher、模板、mask与分母、来源混合权重、许可隐私、安全审核、独立评估。任五项。

16. 不能。变长可能只是模仿数据风格。要在长度控制下测正确性、任务成功和未训练任务。

17. 长度同时影响评分与表面形式。模型可能通过堆字提高 reward，而非提高正确性；应做长度分层与人工审查。

18. 修格式、触发已有知识通常较稳妥；写入完全新事实需谨慎测记忆、近邻干扰、未给事实和幻觉。

19. 安全+帮助=正常帮助；安全+拒绝=false refusal；危险+帮助=violation；危险+拒绝=合理拒绝。

20. 不能。效果依 base model、语言、风险覆盖、样本设计和评估；500只是某案例规模，不是定律。

### B. 配方、模仿与偏好答案（21–35）

21. 它先让模型适应长上下文、领域或初步指令分布，再进入最终助手训练，像预科班。mid-training 的名称、数据和边界没有行业统一定义。

22. Run A：base→同一SFT；Run B：base→mid-training→同一SFT。固定模型、SFT、评估和尽量相同总计算，并重复 seed；差值才较能归于 mid-training。

23. 起点：$`0.7\times1+0.3\times3=0.7+0.9=1.6`$。更新后：$`0.4\times1+0.6\times3=0.4+1.8=2.2`$。

24. imitation 直接用数据中的示范 response；reward optimization 评估并推动当前 policy 采样的 response。

25. 候选可用测试或规则逐一筛，往往比一次构造答案容易。但 verifier 可能漏边界或被投机利用，通过检查不等于完整满足意图。

26. $`\binom52=5\times4/2=10`$ 对。

27. 不是。它们共享同一 prompt、候选、排序者，误差相关；不能当10个独立人的判断。

28. 例：正确性、相关性、清晰度、安全性；若无法区分允许 tie，并规定 tie 是保留、半分还是交第三人复核。

29. pair-average 分母是 pair 数；response-average 分母是 response 数；token-average 分母是有效 token 数。

30. 奖励模型会把 rejected 分数推高、chosen 推低；DPO 会让 policy 相对 reference 更偏 rejected。方向完全反了。

31. 差为0，$`\sigma(0)=1/(1+1)=0.5`$；loss $`=-\ln0.5=0.6931`$。

32. 差 $`=2-0.5=1.5`$。胜率 $`=1/(1+0.2231)=1/1.2231\approx0.8176`$。loss $`=-\ln0.8176\approx0.2014`$。

33. 新差 $`(2+100)-(0.5+100)=102-100.5=1.5`$，与原差相同，所以 sigmoid 胜率仍0.8176。

34. $`(0.2+0.8)/2=0.5`$。

35. 不能。reward 无物理单位，BT 只看差；缩放、平移和训练协议都会改变数值意义。

### C. 反馈、RLHF、KL 与 PPO 答案（36–58）

36. 单个标量难以稳定表示循环、多轴和人群依赖的偏好；BT 是有用近似，不是真实偏好定律。

37. 正确短答：$`1+0.01\times20=1.2`$。错误长答：$`0+0.01\times200=2.0`$。错误回答反而赢，展示长度 hacking。

38. 例：标注员群体/语言与地区、报酬和时间预算、培训与资格、分歧/平局/复核、心理风险。任四项。

39. judge model 会复制自己的事实错误、风格与长度偏差；规模扩大不能证明偏好代表人类。仍需独立人审和分组评估。

40. policy：当前生成并更新；old policy：本批 rollout 快照，冻结；reference：长期起点，冻结；reward：评分，通常冻结；value：预测回报，更新。

41. old 用于 PPO ratio，通常每轮刷新；reference 用于 KL，通常长期冻结。两者可能某一时刻参数相同，但角色和生命周期不同。

42. reward 给观察到的轨迹评分；value 在每个状态预测未来 return，作 baseline。一个是信号，一个是预测器。

43. 计算如下：

```math
0.75\ln(0.75/0.5)+0.25\ln(0.25/0.5)
```

```math
=0.75\ln1.5+0.25\ln0.5
=0.75(0.4055)+0.25(-0.6931)
```

```math
=0.3041-0.1733=0.1308.
```

44. 第二个 token 上 $`p<q`$，所以 log-ratio 为负；KL 是所有 token 按 $`p_i`$ 加权后的和，完整和0.1308仍非负。

45. sum $`=0.2-0.1+0.3=0.4`$。mean $`=0.4/3=0.1333`$。

46. sum 口径：$`0.05\times0.4=0.02`$。mean 口径：$`0.05\times0.1333\approx0.00667`$。二者不可混报。

47. $`3-0.1-0.2=2.7`$。

48. 第一种 $`3-2.2=0.8`$；第二种 $`1.7-2.2=-0.5`$。

49. advantage 看“比该状态预期好多少”。8相对7.5只好0.5；3相对1好2，所以绝对 reward 小的反而 advantage 大。

50. $`0.26/0.2=1.3`$。

51. 原项 $`1.3\times2=2.6`$；clip 后 $`1.2\times2=2.4`$；min=2.4。

52. 原项 $`1.3\times(-2)=-2.6`$；clip 后 $`1.2\times(-2)=-2.4`$；min=-2.6。

53. 原项 $`0.7\times(-2)=-1.4`$；clip 后 $`0.8\times(-2)=-1.6`$；min=-1.6。

54. 原项 $`0.7\times2=1.4`$；clip 后 $`0.8\times2=1.6`$；min=1.4。

55. 区间 $`[1-0.1,1+0.1]=[0.9,1.1]`$。1.25 截为1.1；最终仍需与未截项取 min。

56. PPO 是 $`\min(rA,\mathrm{clip}(r)A)`$。负 $`A`$ 会翻转大小关系；若永远只用 clip 项，会错误奖励某些不利大改。

57. TRPO trust-region KL 比较 old/current 的整分布，并把本次更新限制在 old 附近；PPO clip 则截住这批 sampled action 的 current/old probability ratio，是较易实现的局部近似；RLHF reference KL 比较 current 与长期冻结 reference，限制多轮训练后的累计漂移。前两者围绕“本次 update 相对 old”，后者围绕“长期相对 reference”。$`t`$ 是采样 rollout 的位置，$`\widehat{\mathbb E}_t`$ 是这些位置的经验平均，$`\delta>0`$ 是最大平均 KL 预算。手算：$`(0.006+0.010+0.008)/3=0.024/3=0.008\le0.01`$，所以该批经验平均满足预算；不表示每个未采样位置都满足。

58. rollout → reward/reference/value 计算 → advantage → PPO update → 用新 policy 刷新 rollout。

### D. Best-of-N、DPO 与替代路线答案（59–70）

59. 全错概率 $`=(1-0.3)^4=0.7^4=0.2401`$；至少一中 $`=1-0.2401=0.7599`$。

60. 它假设四次独立、同成功率，而且 verifier 能选中正确候选。真实同模型错误相关，评分器也会错。

61. 生成 $`100\times8=800`$ 条；保留最多 $`100\times2=200`$ 条。

62. 不能。筛选只能从已生成候选中选；需改善探索、teacher、prompt 或 verifier。

63. 本讲只预告 RLVR，下一讲才展开；弱测试仍可能漏真实目标。control-token 训练是 `prompt + [GOOD] + chosen response`、`prompt + [BAD] + rejected response`；推理输入 `prompt + [GOOD]` 后再生成。控制 token 位于 prompt 之后、回答之前，不是整个 prompt 前缀。它可能只学风格、忽略 token，且离线支持外无保证。

64. 取 log：

```math
\ln\pi_r=\ln\pi_{\rm ref}+r/\beta-\ln Z.
```

移项：

```math
r=\beta\ln(\pi_r/\pi_{\rm ref})+\beta\ln Z.
```

同一 prompt 下 $`Z(x)`$ 相同，所以 $`r_w-r_l`$ 中两个 $`\beta\ln Z`$ 抵消。

65. chosen：$`\ln(0.5/0.4)=0.223143551`$。rejected：$`\ln(0.1/0.2)=-0.693147181`$。margin $`=0.223143551-(-0.693147181)=0.916290732`$。logit $`=0.5\times0.916290732=0.458145366`$。概率 $`\sigma(0.458145366)=0.612574113`$。loss $`=-\ln0.612574113=0.490085343`$。

66. 两个 log-ratio 都0，margin=0，$`\sigma(0)=0.5`$，loss=0.6931。

67. probability $`=0.5\times0.4=0.2`$。log $`=\ln0.5+\ln0.4=-0.6931-0.9163=-1.6094`$。

68. 每个条件概率不大于1，其 log 通常非正；token 越多，和会继续变负。除以长度改变了目标，可能偏向另一种长度，不能保证解决真实偏好。

69. KL 目标中 $`\beta`$ 越大，最终最优 policy 通常越受 reference 约束；固定当前 margin 时，DPO logit 又是 $`\beta\times margin`$。但改 $`\beta`$ 后训练得到的 margin 也会变，不能冻结它来推断完整最优解。learning rate 只控制 optimizer 单步参数变化。

70. 依赖固定 reference、KL 正则化目标、BT/logistic 偏好模型、nonparametric 闭式解假设和给定 offline pairs；chosen/rejected 在 reference 下须有正、可计算的 log-prob，有限网络只能近似。完整 empirical objective 对全部 $`M`$ 对除以 $`M`$；一个训练 step 的 `.mean()` 只对当前 $`m`$ 对除以 $`m`$。真实偏好、在线采样与 PPO dynamics 不必满足这些假设，因此不能称完全等价。

### E. 失败诊断与综合设计答案（71–80）

71. 选中段。人工质量0.72最高；后段虽 proxy 0.92最高，人工质量跌到0.60，已经过度优化代理。

72. 可能 mode collapse。对同 prompt 多次采样，统计独特回答、重复片段、语义簇和任务成功率；还应跨 prompt 看模板重复。

73. $`60/100=0.60=60\%`$。声明80%但长期只对60%，模型过度自信、未校准。

74. 不是固定税率，也不是必然。对 base、SFT、偏好模型用同协议测能力、安全、帮助、长度、校准和成本，报告差值与不确定性。

75. reward model 在新文本上外推，可能错打高分；offline 数据不覆盖新的 reward-hacking 行为。需新采样/标注、独立审计和保守更新。

76. 不遵守格式：先高质量SFT并查模板/mask；有可靠 verifier：Best-of-N、拒绝采样或 RLVR 方向；只有主观偏好：收 pair 后用 DPO 或 reward+PPO。

77. 例：候选生成 policy/temperature/长度、rubric、标注员群体与报酬、pair 构造相关性、reference 版本、长度/位置偏差审计、独立 holdout。任五项。

78. token-average：

```math
\frac{2\times1+8\times3}{2+8}
=\frac{26}{10}=2.6.
```

response-average：

```math
(1+3)/2=2.
```

79. 观察到的现象名是 **reward overoptimization（奖励过度优化）**：proxy 上升而人工质量下降。reward hacking 只是可能机制；只有发现模型利用评分漏洞的证据时才这样诊断。防守例：独立人工盲审；长度分层/分布外测试；多个 checkpoint 早停；独立评估器；加 KL 约束。任三项。

80. 一份合格方案：

   1. 定义数学任务和安全边界（prompt 层）；
   2. 收人工/验证过的 prompt-response（response 层）；
   3. 固定 chat template；
   4. prompt/padding mask=0、assistant=1，SFT 按有效 token 平均；
   5. 保存 SFT baseline；
   6. 同 prompt 采多个 response；
   7. 用答案 verifier 与人类 rubric 产生 pair，BT/DPO 按 pair 平均；
   8. 若 DPO，锁定 reference 与 $`\beta`$；若 PPO，记录五角色及 token/sequence 分母；
   9. 保存多 checkpoint；
   10. 在未参与训练的题上测正确性、长度、安全、校准、多样性和成本；
   11. 人工审查高 reward 失败；
   12. 版本化记录数据、代码、模型与停止理由。

<a id="l15-video-nav"></a>

## 31. 视频时间导航（人工英文字幕）

> 为避免“时间戳能点开、主题却错位”，以下 171 行都逐条回查该秒前后的多条人工字幕。第二列是根据上下文写成的中文主题，回答“这一段到底解决什么”；第三列保留一小段英文 cue 作为可审计证据。中文主题不是把一个残句逐字翻译成中文。

> 每个链接都命中人工 en-US 字幕 cue；显示时间与 URL 的 `t=秒` 严格一致，且全文不重复使用秒点。英文证据只截取足以辨认主题的短语，完整上下文以本地人工字幕和视频为准。

### 31.1 开场与后训练地图（00:05–06:53）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=5s) | 开场：课程从 pre-training 转入更杂乱的 post-training | `OK.`；随后说 “move away from pre-training” |
| [00:38](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=38s) | 继续堆数据/算力可让 GPT-3 稍强，但 base model 的直接用途仍有限 | `So you can make something a little bit better than GPT3.` |
| [01:04](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=64s) | 早期 GPT-3 多用于文案等低可靠任务，难稳定遵循复杂指令 | `And the only thing you could really do with it` |
| [01:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=95s) | 定义本讲桥梁：从 base model 到 ChatGPT 式助手的过程叫 post-training | `will be a process that usually people call post-training.` |
| [02:08](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=128s) | 早期模型难 steer；GPT-3.5/4 才更能一次执行长而程序化的 prompt | `In practice, your ability to steer these models or limited.` |
| [02:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=160s) | 不能跳过 pre-training 只靠后训练；先获得广泛能力，再抽取目标行为 | `try to train our way to victory, we will get none of the things that we want.` |
| [03:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=183s) | post-training 依赖显式数据收集、steering 和大量现实工程 | `a lot of messy engineering` |
| [03:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=214s) | frontier post-training 公开信息稀少，课程只能大量借助较早资料 | `information about frontier post-training is honestly pretty sparse` |
| [03:57](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=237s) | 旧论文附录曾公开详细 annotation guidelines；竞争加剧后透明度下降 | `go read some of the appendices.` |
| [04:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=269s) | 开源配方常靠蒸馏，与 frontier lab 的人工数据收集并非同一种证据 | `a lot of the open source recipes rely on distillation.` |
| [04:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=298s) | 用 2023 年材料泄露例说明后训练数据被当作商业秘密 | `I found this very fun example back in 2023 about the trade secretness of post-training data` |
| [05:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=327s) | 竞争者尝试反向工程 GPT-4，并要求标注员写得更详细 | `we need to get our annotators to produce responses that are more detailed` |
| [05:51](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=351s) | 后训练的主要杠杆往往是数据；经典 RLHF 可拆 demonstration 与 preference/RL 两段 | `It's going to be the data.` |
| [06:21](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=381s) | RL 阶段塑造更符合人类偏好的行为；接下来先讲 SFT 数据 | `more in line with what humans think are good responses.` |
| [06:53](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=413s) | 前半路线图：看数据样本、历史演化与开放 SFT 数据趋势 | `We will go through the data.` |

### 31.2 SFT 数据演化与 agentic 数据（07:23–18:33）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [07:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=443s) | SFT 数据工程会重新遇到经典监督学习的许多数据问题 | `issues that you see when you start building classic supervised deep learning models` |
| [07:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=478s) | 开源社区不断试验：什么样的 instruction data 真能提升模型 | `what factors of this kind of data are useful to train models` |
| [08:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=507s) | FLAN 的“汇总旧监督数据集”不是最终答案；随后转入 Self-Instruct | `this was not the right thing to do ... Self-instruct` |
| [08:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=536s) | Alpaca/Vicuna 代表从强模型蒸馏聊天输入输出的路线 | `Alpaca is one of them ... Vicuna` |
| [09:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=563s) | WizardLM、Tulu3 用语言模型生成更复杂的合成指令数据 | `increasingly complicated ways of generating instruction following data` |
| [09:54](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=594s) | 从聊天转向 agent/tool-use 数据；闭源实验室还多了未公开的人类采集环节 | `new generation of SFT pipelines ... tool use and agentic data` |
| [10:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=620s) | 从历史地图转到具体数据集样本，并回答 input-output 正确性问题 | `look at some of the details of these data sets` |
| [10:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=640s) | 正确性很重要但不是单一开关：坏答案会教坏行为，强 base model 也能容忍部分噪声 | `how much does the correctness ... matter? That's a nuanced question` |
| [11:12](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=672s) | 预训练带来的泛化能力让 instruction-following 对数据缺陷有一定容忍度 | `pre-training generalization behaviors that let you get away with worse quality data` |
| [11:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=700s) | FLAN 的核心想法：把许多 downstream supervised tasks 合并训练 | `go collect all the downstream tasks and train on it` |
| [12:07](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=727s) | FLAN 样本结构常很不自然，因为它把旧 benchmark 硬改成 instruction 格式 | `FLAN is generated from existing data sets` |
| [12:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=754s) | 旧摘要数据的目标过短且可能含输入未支持的细节，不像真实聊天回答 | `the summaries are often hallucinated` |
| [13:04](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=784s) | 汇总旧数据也会继承旧数据的质量缺陷和不自然结构 | `inherit a whole bunch of deficiencies from those data sets` |
| [13:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=814s) | 早期 FLAN 假设 post-training 也需要海量数据，后来经验修正了这个想法 | `the theory ... was that you needed scale` |
| [14:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=843s) | 强预训练模型常可用少量高质量 SFT 例子激活行为，质量可胜过盲目堆量 | `sufficiently strong ... pre-trained model ... very few high quality examples` |
| [14:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=875s) | Alpaca 从 ChatGPT 轨迹蒸馏出更自然、较长、聊天式的训练样本 | `Alpaca ... distilled ChatGPT traces to get input-output pairs` |
| [15:16](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=916s) | Chat-style 数据配合合适的 LLaMA base model，才开始稳定诱导 ChatGPT 式行为 | `only when we did it to the original Llama models ... these things started to work` |
| [15:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=948s) | ChatGPT/Alpaca 后出现“收集足够好的开放指令数据就能追平”的乐观浪潮 | `enormous optimism ... sufficiently high quality and large instruction-tuning data set` |
| [16:15](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=975s) | OpenAssistant 用众包方式收集困难 prompt 与高质量回答 | `hard and interesting prompts ... good, high quality responses` |
| [16:42](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1002s) | OpenAssistant 样本更像长对话和专家回答，但仍有后续会讲的陷阱 | `very long, detailed, high quality expert responses` |
| [17:06](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1026s) | 产品目标从纯聊天扩大到 tool calls、待办列表和 agent 行为 | `We don't want just textual responses. We want tool calls` |
| [17:31](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1051s) | agentic SFT 会把自然语言回复和并行 tool call 一起显式监督进模型 | `responses ... but also tool calls that can happen in parallel` |
| [18:07](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1087s) | 数据风格从“benchmark 输入→程序式输出”转为更详细、更像人类交流的回答 | `classic NLP data sets ... programmatic output ... shift ... human-like responses` |
| [18:33](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1113s) | 总结三次变化：更高质量标注、更人类化回复、工具/API 结构化接口 | `experts write your responses ... tool use ... right interface and API` |

### 31.3 风格、知识与安全（19:03–35:31）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [19:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1143s) | 从数据集历史转到“若你负责收集 SFT 人工数据，应注意什么” | `you are put in charge of collecting human annotation data for SFT` |
| [19:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1169s) | 标注设计的四个轴：知识、样本量、安全，以及长度/风格 | `How much data points ... safety ... length ... style variation` |
| [20:02](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1202s) | 不同数据集回答长度差异很大，偏好评估会被风格强烈影响 | `wide variation ... length ... stylistic factors matter a ton` |
| [20:30](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1230s) | 人会偏好更长、带列表的答案，这会把聊天机器人的语气推向非自然风格 | `more detailed and list like responses ... induce ... distortions` |
| [21:02](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1262s) | engagement 上升不等于能力提高；风格信号很容易欺骗评估者 | `easy to fool yourself ... models capabilities are not changing` |
| [21:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1294s) | AlpacaEval 等偏好分数可大幅变化，而标准能力 benchmark 未必同步变化 | `doesn't necessarily change standard benchmarking evaluations` |
| [21:59](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1319s) | 把 style control 与 capability control 分开；OpenAssistant 用来展示知识监督陷阱 | `style control separately from capabilities control` |
| [22:28](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1348s) | 带文献引用的 SFT 回答同时教两件事：引用内容和“应该引用”的格式 | `teaching the model two different things at once` |
| [22:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1375s) | 同一条 next-token 样本既注入知识，也诱导模型输出细节/引用的行为 | `teaching ... pieces of knowledge ... also ... emit detailed knowledge` |
| [23:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1400s) | 模型未必知道引用真伪；强迫其引用未知事实容易诱发伪造文献 | `Models don't really know necessarily whether a reference is true or false` |
| [23:53](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1433s) | “未知知识 + 引用格式”会把模板行为错误泛化到模型不知道的内容上 | `forcibly emit unknown knowledge` |
| [24:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1463s) | 课堂提出 RL 的动机：让监督依赖模型自己的输出与所知状态 | `reason why you need reinforcement learning and training` |
| [24:45](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1485s) | 外部专家直接塞入知识，未必能教会模型校准“我知道/不知道” | `calibrated about what it knows and doesn't` |
| [25:14](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1514s) | RL 可用模型生成的结果来奖励诚实引用、惩罚不知道时硬编 | `knowing what you know, RL is a very useful way` |
| [25:38](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1538s) | post-training 很杂乱：知识、格式和最终用户场景必须一起考虑 | `deal with all of these realities of serving a system eventually to a user` |
| [26:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1563s) | “tail knowledge”没有精确边界；Wikipedia 页面长度只能作知名度代理 | `length of a Wikipedia article ... proxy for its well-knownness` |
| [26:31](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1591s) | RL 减少幻觉的 folk story：模型内部可能已有“我知道”的方向 | `why RL might help ... folk story` |
| [26:59](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1619s) | 若引用与“我知道”状态一致得高奖励、与“不知道”状态一致得低奖励，策略可学会校准 | `good rewards ... I know ... bad rewards ... I don't know` |
| [27:26](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1646s) | 学生追问：错误引用在 SFT loss 中到底在哪里受罚 | `if it emitted a bad reference, where do you get penalized` |
| [27:52](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1672s) | SFT 样本本身可以正确，问题在泛化：模型可能学到引用模板而没学到事实边界 | `two disentangled things ... reference template ... actual content` |
| [28:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1700s) | 从知识转到 safety：post-training 必须直接面对现实滥用场景 | `engaging with ... how are people going to use this` |
| [28:46](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1726s) | 安全后训练是部署前最后一道行为控制，通过拒答恶意输入降低滥用 | `last line of defense ... apply safety controls` |
| [29:18](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1758s) | 公开 safety SFT 细节比能力数据更少；LLaMA 2 也未公开完整样本量 | `safety SFT information seems even more sparse` |
| [29:44](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1784s) | 安全调优要平衡 violation rate 与 over-refusal，避免把正常请求误拒 | `balance ... violation rate ... not over refuse` |
| [30:07](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1807s) | 用定制安全数据寻找两类错误的 Pareto 折中；课程举例量级为数千到数万 | `tailored data to navigate this trade off` |
| [30:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1840s) | OLMo 是当时少见的、能看到完整 post-training pipeline 的公开案例 | `post-training pipeline for the OLMo models` |
| [31:14](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1874s) | WildChat 用免费聊天/API 换取真实交互日志，再从中挖掘不安全请求 | `free API or free chat ... collected what they were doing` |
| [31:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1908s) | 对挖出的攻击/恶意请求编写拒绝答案，形成安全 SFT 的“打地鼠”闭环 | `created ... preferred response ... resist the jailbreak ... say no` |
| [32:18](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1938s) | 强 base model 只需少量示例也能被明显 steer，不等于精细安全已解决 | `does not take very many examples to steer these systems` |
| [32:47](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1967s) | 课堂案例：约 500 个拒绝样本就能显著降低多类恶意指令服从率 | `as little as 500 examples ... drops dramatically` |
| [33:21](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2001s) | 少量数据能改变模式，但细粒度安全边界仍会受益于大规模、多样数据 | `doesn't mean that there's no benefit to more examples` |
| [33:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2028s) | SFT 最擅长抽取预训练中已有的行为模式，而不是凭空创造能力 | `extracting pre-training behaviors ... pull out the right modes` |
| [34:14](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2054s) | SFT 小结：高质量常比数量重要，错误注入“新知识”甚至会增幻觉 | `focus on quality instead of quantity` |
| [34:38](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2078s) | 如何判断某行为是否已在预训练中没有可靠直接测试，只能看部分反例与泛化表现 | `we don't really know per se` |
| [35:04](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2104s) | 学生问 SFT 与 RL 会否破坏/增强同一特征；讲者强调二者边界模糊 | `whether SFT is destroying features in RLs like pushing up or down` |
| [35:31](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2131s) | 更关键的差异是反馈：SFT 给 dense token supervision，RL 用自身 policy 输出获稀疏反馈 | `distinction in the kinds of feedback ... SFT ... dense ... RL ... own policy` |

### 31.4 SFT 训练、mid-training 与偏好入口（35:56–49:19）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [35:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2156s) | SFT 算法本身并不神秘：仍是 next-token loss、backward 和 gradient descent | `you just do gradient descent ... loss.backwards` |
| [36:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2183s) | pre-training 与 post-training 曾经分开，如今高质量 instruction data 常混入后段预训练 | `why separate two things when you can mix them together` |
| [36:52](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2212s) | mid-training/第二阶段改变数据混合，让低学习率尾段更强调高质量数据 | `emphasize higher quality data ... second phase pre-training` |
| [37:25](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2245s) | “base model”不再纯粹：现代 base checkpoint 可能已看 UltraChat 等合成聊天数据 | `base models today are pre-trained on ultra chat` |
| [37:57](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2277s) | 两阶段示例把普通互联网占比降下，换入 StackExchange、UltraChat 和 SFT 数据 | `switching to this higher quality, and also very chatty data set` |
| [38:26](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2306s) | mid-training 模糊 pre-training/SFT 边界；纯预训练口径下 prompt token 也参与预测 | `boundaries ... being blurred ... pure pre-training` |
| [38:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2335s) | decay 最接近部署且学习率最低，课堂直觉是放更高质量而非更差的数据 | `decay is the most important part ... lowest learning rate` |
| [39:18](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2358s) | Wiki、StackExchange 等被课程当作高质量来源；学生追问混合比例如何定 | `Wiki and Stack Exchange is usually considered high quality` |
| [39:42](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2382s) | 数据混合虽有论文算法，实际仍大量依赖 trial-and-error 与经验判断 | `data mixtures ... are very trial and error` |
| [40:05](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2405s) | mid-training 比完整预训练短，因此可以较便宜地跑多组 mixture 实验 | `midtraining is much shorter than full pre-training` |
| [40:33](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2433s) | 在 decay 阶段做 domain ablation，估计数据价值，再反馈到主预训练 mixture | `do a lot of data ablations ... reflect that ... back to the first stage` |
| [40:54](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2454s) | 不能让主预训练全用“高质量”数据，因为合格 token 量根本不够 | `you just don't have enough tokens` |
| [41:21](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2481s) | 从小实验推回大 mixture 的模型往往脆弱，实际会按 ablation 排名再人工决策 | `fit models ... often brittle ... much more trial and error` |
| [41:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2508s) | 课程以书籍子集 ablation 文档说明真实团队如何估计各数据源价值 | `doing ablations and trying to estimate how useful are each ... subsets` |
| [42:08](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2528s) | SFT 部分结束，RLHF 改为按人类/奖励模型评分上调或下调模型输出 | `upweight or downweight different model outputs` |
| [42:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2554s) | 先建立概念差异：SFT/预训练是拟合数据分布，RLHF 是优化奖励 | `different way of thinking about ... SFT ... and RLHF` |
| [42:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2578s) | SFT 仍做 generative modeling；RLHF 不再只问“像不像数据” | `SFT ... predicting the next word ... RLHF ... maximize a reward game` |
| [43:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2603s) | RLHF 把模型分布视作 policy，目标是最大化定义好的 downstream reward | `policy ... maximize some downstream reward` |
| [43:47](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2627s) | 只看奖励时，policy 即使对每个 prompt 坍缩到单一答案也可能得高分 | `single answer, not a distribution ... OK, as long as it got a good reward` |
| [44:18](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2658s) | 为什么不只收 SFT：人能判断偏好，却未必能亲手生成自己最喜欢的答案 | `difference in what they say they want and what they generate` |
| [44:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2688s) | 旧实验里部分专业写作者反而偏好模型输出，显示“生成”和“评判”能力有差距 | `preferred Instruct Davinci ... over their own writing` |
| [45:09](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2709s) | 标注员看到候选后能识别更好写法，所以 pairwise rating 有时比 demonstration 更容易 | `when they judge things, it's actually different` |
| [45:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2735s) | 数学证明是 generate–verify gap 的例子：验证通常比从零生成容易 | `verifying a proof is probably much easier than generating the proof` |
| [46:01](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2761s) | 本讲聚焦 RLHF 的数据与算法；可验证奖励 RLVR 留到下一讲 | `today, I'm going to talk about RLHF` |
| [46:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2787s) | RLHF 起点：SFT 模型对同一 prompt 采样多个有差异的候选回答 | `good model after SFT ... generate a couple of different outputs` |
| [46:52](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2812s) | 人给候选排序，训练 reward model，再用 RL 最大化其分数 | `model is going to be trained on those rankings ... maximize the score` |
| [47:17](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2837s) | 偏好采集的基本界面：并排展示两个 AI 回答，让人选更好的一个 | `a couple AI responses ... response one or two is better` |
| [47:50](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2870s) | InstructGPT 标注协议同时要求 helpful、truthful、harmless | `rate these outputs for being helpful, truthful, and harmless` |
| [48:16](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2896s) | harmless 维度会奖励对可疑 prompt 的恰当拒答，但三目标需共同权衡 | `upweight things that are refusing ... questionable prompts` |
| [48:47](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2927s) | 泄露的 Bard 指南同样评 helpfulness/呈现/事实性，只是用 Likert 分数而非成对选择 | `rating in a Likert scale rather than pairwise feedback` |
| [49:19](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2959s) | 转入标注劳动：行业逐渐使用更贵、教育程度更高的专家 | `worker distribution ... shifted upwards ... towards experts` |

### 31.5 标注员、模型反馈与长度偏差（49:48–65:42）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [49:48](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2988s) | 单平台调查快照：样本中多数标注员有本科/硕士学历，但不代表全部 RLHF 数据劳动者 | `not representative ... majority ... bachelor's degrees or master's holders` |
| [50:22](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3022s) | 面向真实白领任务后，实验室开始招医生、律师等领域专家提供数据 | `deploying these systems in actual white collar jobs ... doctors ... lawyers` |
| [50:52](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3052s) | 专家标注可能超过每小时 100 美元，低价海外众包已不是完整行业图景 | `experts ... paid more than $100 per hour` |
| [51:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3089s) | 高质量数据难收集：既要验证专家身份，也要确认其没有偷偷用 AI 作答 | `getting verifiable annotators ... sure that they're not using AI` |
| [51:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3115s) | 时间压力下核验长回答几乎不可完成，标注指南可能现实上执行不了 | `difficult to get truly correct responses when people are under time pressure` |
| [52:22](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3142s) | 专家数据增长与低薪众包并存，行业呈现高低工资分化 | `growing amount of expert data ... also ... low paid annotation groups` |
| [53:01](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3181s) | 标注员 demographics 会在模型发布前最后塑造其行为 | `annotators have a surprising amount of influence over what the model does` |
| [53:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3209s) | 课程研究用民调题比较模型观点与不同人群，但这只是特定测量设计 | `which kinds of ideological opinions language models were more aligned to` |
| [53:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3238s) | 研究观察：post-training 后模型回答与某些宗教/地区群体的距离发生变化 | `if you post-train these models ... becomes more similar` |
| [54:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3269s) | 讲者把变化与 InstructGPT 标注员地域构成联系起来，但只能视作相关性线索 | `annotators ... Southeast Asians ... West Coast of America` |
| [54:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3298s) | 合成数据可传递非常隐蔽的偏好；表面无害文本也可能携带 teacher bias | `subliminal transfer effects ... quite hard to catch` |
| [55:22](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3322s) | 政治意见测量本身脆弱，但标注员构成也会影响更具体的事实/一致性错误 | `annotator distribution ... matters ... material errors` |
| [55:57](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3357s) | 非专家更容易抓格式，专家更关注 factuality 和 inconsistency | `non-expert annotators ... formatting ... factuality or inconsistency ... expert` |
| [56:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3383s) | 标注质量不只由人口统计决定，也取决于专业知识、投入程度与可核验能力 | `how much they care and what their expertise is` |
| [56:53](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3413s) | 没有机械的“好标注员”金标准；可先用详细 guideline 检查可观测错误 | `no gold standard rule ... detailed annotation guideline` |
| [57:18](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3438s) | 质量证据之一是是否遵守半客观规范，另一项是群体间一致性 | `doesn't follow the guideline ... inter-annotator agreement` |
| [57:39](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3459s) | 一致性只反映 variance，不保证没共同偏差；主观喜好本来就可高分歧 | `tells you the variance ... if they're all using ChatGPT, the variance will also be zero` |
| [58:13](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3493s) | 招专家主要因为某些任务只有受训人士能判，例如法律引注，而非学历天然保证一切更好 | `you need a lawyer to check if a Blue Book annotation is correct` |
| [58:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3520s) | 即使一般任务也转向高质量、可验证人工，因为低监督环境易被廉价 LLM 代写 | `if you don't ... supervise ... they will use the cheapest LLM` |
| [59:07](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3547s) | 学生转问：领域模型当标注员的最大问题是什么 | `What's the largest problem using a domain-specific model as an annotator?` |
| [59:30](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3570s) | 模型标注很适合追赶 frontier：强模型常优于随机众包且可规模化 | `Model-based data annotation is quite good ... catch up to the frontier` |
| [59:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3596s) | 领域模型未必强过通用 frontier model，所以“domain-specific”本身没有天然优势 | `can you get domain-specific models ... better ... Sometimes, quite difficult` |
| [60:28](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3628s) | GPT-4 与精心人工标注的系统排序/多数意见曾较一致且便宜很多，但不是逐样本完美 | `compare the rankings of systems ... pretty good ... cost ... less` |
| [60:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3656s) | 若目标只是追上已有 frontier capability，模型蒸馏通常比昂贵人工收集更划算 | `no space for human collected data if all you want ... catch up to the frontier` |
| [61:30](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3690s) | Zephyr 曾刻意避免 model distillation，投入供应商和人工数据做对照 | `wanted ... to not do any model distillation` |
| [61:57](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3717s) | 该项目人工方案昂贵且不更好，最终改用 model-based feedback | `results ... not actually better ... just used model-based feedback` |
| [62:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3749s) | Tulu3 代表开放 pipeline 大规模采用 model-based annotations 的趋势 | `uses model-based annotations for all of its pipeline` |
| [62:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3775s) | 追赶可以蒸馏；要推进知识/能力 frontier，仍常依赖新的人工或环境信号 | `if you want to push the frontier out ... reliant on human-driven data` |
| [63:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3800s) | Zephyr 当时在 7B 尺度有代表性，但该尺度结论不能自动外推到更大 run | `at least, at that scale, they weren't seeing any differences` |
| [63:45](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3825s) | Constitutional AI 是模型自举安全数据的例子，不等同于单向抄强 teacher | `not purely distillation ... prompted a model to generate safety data` |
| [64:13](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3853s) | Self-Instruct 也能自举，但律师/科学等新知识仍不能只靠模型凭空产生 | `bootstrap your own post-training data ... can't get world knowledge without ... people` |
| [64:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3880s) | 模型 judge 也有偏差：把回答越写越长，win rate 可能继续升 | `push length ... continue to get improvements in ... model-judged performance` |
| [65:11](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3911s) | 只优化长度也可能提升多个偏好 benchmark，说明“胜率高”不等于内容更好 | `RLHF on length alone ... do quite well on many ... benchmarks` |
| [65:42](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3942s) | 数据部分结束；剩余时间简讲 PPO，再讲 DPO，完整 RL 算法留下一讲 | `briefly talk about PPO ... more extended treatment next lecture ... DPO` |

### 31.6 PPO、DPO、变体与失败（66:05–79:45）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据 |
|---|---|---|
| [66:05](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3965s) | 单轮 RLHF 接近 contextual bandit，不是复杂多轮环境 RL，所以基本目标较简单 | `baby reinforcement learning ... not true multi-turn ... RL` |
| [66:33](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3993s) | 奖励之外加 policy/reference KL，防止策略远离原模型后退化 | `stay close to my pre-trained model ... don't want to ... become degenerate` |
| [67:00](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4020s) | 用 PPO 优化 RLHF；零基础起点是 policy-gradient identity | `We use an algorithm called PPO ... starting point ... policy gradient identity` |
| [67:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4047s) | policy gradient 把 log-prob 梯度按 rollout reward 加权 | `gradient of my log probabilities and weigh it by my rewards` |
| [67:52](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4072s) | 朴素 on-policy 每步都重新采样很贵，因此希望一批 rollout 复用多次更新 | `sample every time ... sampling is very expensive ... roll out once` |
| [68:17](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4097s) | TRPO 用 importance correction 复用旧样本，同时限制 current policy 不离 old policy 太远 | `off-policy ... not go too far ... TRPO ... importance weighting correction` |
| [68:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4120s) | PPO 用 clipping 近似难处理的 trust-region constraint | `heuristic clipping ... discourage ... too far from the original policy` |
| [69:08](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4148s) | PPO 公式/系统复杂，研究者长期寻找能否用更像 SFT 的替代方法 | `Can we get rid of PPO?` |
| [69:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4175s) | control-token 尝试：prompt 后给 chosen 加 `[GOOD]` 前缀、rejected 加 `[BAD]`，推理时给 `[GOOD]` | `good ... prepend a good token ... bad ... prepend a bad token` |
| [69:59](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4199s) | 只保留 reward model 选中的候选再 SFT（rejection sampling）能工作一些，但通常不如完整 RL | `select only the stuff ... and train on them ... does not work as well` |
| [70:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4227s) | DPO 去掉显式 reward model 和 on-policy rollout，把 RLHF 变成离线 pair loss | `get rid of the reward model ... anything ... on-policy` |
| [70:53](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4253s) | DPO 直觉：提高 chosen log-prob、降低 rejected log-prob，但需正确相对权重 | `steps in the direction ... good stuff ... negative ... bad stuff` |
| [71:21](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4281s) | 推导从 KL-regularized reward objective 开始：既要高奖励，也要贴近 reference | `expected reward ... second term ... KL distance ... close to my reference` |
| [71:46](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4306s) | 关键 nonparametric 假设：policy 可取任意分布，才可写出闭式最优策略 | `set of all possible policies ... nonparametric ... solve ... closed form` |
| [72:17](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4337s) | 闭式解用 `exp(reward/beta)` 对 reference 概率上调或下调 | `reward is really bad ... exponentially downweight ... good ... upweight` |
| [72:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4360s) | 把闭式最优式反解成 implied reward，再代入 pairwise preference model | `solve for the implied reward` |
| [73:09](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4389s) | 得到 DPO：chosen 方向为正、rejected 方向为负，两项共同决定更新 | `positive direction ... negative gradient steps in the bad direction` |
| [73:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4414s) | 每个 pair 都提高 winner、降低 loser；步长还取决于当前 implied reward 排得多错 | `increase the likelihood ... decrease ... scale the step size ... how ... wrong` |
| [74:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4443s) | 当前 margin 已大且方向正确时权重小；打平或排错时权重更大 | `based on the probability differences ... bigger or smaller step sizes` |
| [74:29](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4469s) | DPO 像普通反向传播、实现简单且通常有效；不保证永远胜 PPO | `much simpler than PPO ... does work reasonably well` |
| [74:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4495s) | LLaMA 案例把 SFT→DPO→rejection sampling 置于多轮 outer loop 中 | `SFTd it, they did DPO ... generate candidates ... rejection sampled` |
| [75:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4527s) | SimPO/length-normalized DPO 修改 reference 或长度处理，目标之一是减少 length hacking | `length normalized DPO ... avoid certain length hacking` |
| [75:57](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4557s) | Ai2 报告中 PPO/DPO 胜负可随实现翻转，实验细节比算法标签更重要 | `depending on how you execute this, one can be better than the other` |
| [76:26](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4586s) | 实用小结：chosen 正向、rejected 负向，关键是步长与配方是否设置合理 | `negative gradient step from the bad stuff ... set the step sizes right` |
| [76:58](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4618s) | 仅靠更多 thumbs-up/down 不能无限提升；强推 RLHF 会开始拟合 learned reward 的漏洞 | `start overfitting to your learned reward model` |
| [77:23](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4643s) | reference/KL 等约束对防止 reward overoptimization 很关键；随后转向 mode collapse | `prevent ... overfitting your reward model ... model collapse` |
| [77:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4676s) | 奖励优化可牺牲多样性，让 policy 集中在少数高分回答模式 | `policy that can collapse as long as it gets a good reward` |
| [78:27](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4707s) | RLHF 后概率可能失去 calibration；RLVR 又需要足够 entropy 做探索 | `models are uncalibrated after ... RLHF ... entropy and exploration ... critical` |
| [78:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4736s) | 全讲总结：数据难、PPO 工程复杂，并非换一个 objective 就结束 | `RLHF data collection is also very hard ... algorithms are quite complex` |
| [79:17](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4757s) | 下一讲与作业会用较简化的 GRPO，并继续讨论 reward overoptimization | `simpler variant ... GRPO ... assignments ... over optimization` |
| [79:45](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4785s) | 课程收尾：RLVR 的吸引力来自可验证奖励可能持续改善的希望 | `rewards where ... performance ... keeps monotonically getting better ... RLVR` |

<a id="l15-pdf-coverage"></a>

## 32. PDF p.1–65 逐页覆盖索引

> 每页在下表恰好出现一次。连续页段只用于压缩表长，段内每页均通过渲染图目视检查；不能把这张索引误解成“每页只看标题”。

| PDF 页 | 该页段实际内容 | 正文落点 |
|---:|---|---|
| p.1–5 | 课程标题；pre-training 到 InstructGPT 的差距；指令控制示例；本讲三个问题；现代后训练信息稀缺 | §0、§2–§3、§33 |
| p.6–8 | InstructGPT 三阶段；SFT 的数据与方法；指令数据的两个观察问题 | §3–§4、§11、§16 |
| p.9–15 | FLAN→Self-Instruct/Alpaca→ShareGPT/Vicuna→OpenAssistant→WizardLM→Tulu3/Nemotron/tool-use 演化；具体样本与数据差异 | §4、§7、§26 |
| p.16–18 | 数据/模型 style 差异；偏好评估受 style 影响；benchmark 与风格/能力不等价 | §7、§25、§27 |
| p.19–21 | 引用、复杂知识、事实性；knowledge extraction/alignment 图；知识注入边界 | §8、§25 |
| p.22–27 | safety 定义；野外安全 SFT；详细 pipeline；用户场景；少量安全数据；SFT 数据总表 | §9、§4.4、§26 |
| p.28–30 | fine-tune 训练；instruction tuning 的 pretraining 形式；midtraining/two-phase | §5–§6、§10–§11 |
| p.31–34 | RLHF 第二部分；imitation 与 optimization；generate–verify gap；总览 | §12、§16、§21 |
| p.35–38 | RLHF/偏好数据；标准比较设置；InstructGPT guideline；旧 Bard annotation | §13–§14、§26 |
| p.39–44 | worker 地区、报酬、crowdsourcing、伦理、人口统计、标注员 style | §15、§25、§26 |
| p.45–48 | LM-generated feedback；self-training；length effects | §15.3–§15.4、§21、§25 |
| p.49–53 | RLHF；imitation→optimization；语言模型 PPO 总目标；Stiennon 细节；PG/TRPO/PPO 公式 | §14、§16–§20 |
| p.54–58 | control token/只训 preferred/Best-of-N 等去 PPO 尝试；DPO 总览；非参数闭式最优；DPO loss 与 gradient weight | §21–§23、§28 |
| p.59–61 | LLaMA/open-model expert iteration；偏好目标变体；PPO/DPO 经验比较 | §21.3、§24 |
| p.62–65 | 风险转场；p.63 overoptimization 详细图；p.64 calibration/mode collapse；recap 与 RLVR 预告 | §25、§27、§34 |

### 32.1 公式页视觉核对结果

- PDF p.51：确认 reward 减 $`\beta`$ 乘 policy/reference log-ratio，并另含可选 pretraining 项；没有把 old policy 当 reference。
- PDF p.52：确认 BT loss 使用 $`r(x,y_w)-r(x,y_l)`$，chosen 在前；并区分 token KL 与 value function。
- PDF p.53：确认 PPO surrogate 是原项与 clipped 项取 min；因此 §19 对正、负 advantage 分四格计算。
- PDF p.56：确认最优策略为 $`\pi_{\rm ref}\exp(r/\beta)/Z`$，反解 reward 后有同 prompt 的 $`\beta\log Z(x)`$。
- PDF p.57：确认 DPO 是 chosen relative log-ratio 减 rejected relative log-ratio，方向未反。
- PDF p.58：确认 gradient 权重随当前 preference margin 改变，不能解释成所有 pair 等权。
- PDF p.60：确认展示多种具体目标/长度处理，不是“长度归一化必胜”的定律。

<a id="l15-sources"></a>

## 33. 来源、SHA、字幕与图像视觉核验

### 33.1 课程主来源

- [Stanford CS336 Spring 2026 官方课程页](https://cs336.stanford.edu/)（当前权威课程入口）。
- [官方 Lecture 15 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)。
- [Stanford Online Lecture 15 视频](https://www.youtube.com/watch?v=2oH6PWPrYFo)。

| 本地材料 | 数量/大小 | SHA256 |
|---|---:|---|
| lecture_15.pdf | 65 页；6,395,151 bytes | 42A4BC02408A96E0B414F789F32780AE1D8461000F872D93AA38C94FE369E862 |
| transcript_en_us.txt | 人工 en-US；1,872 cues/lines；95,650 bytes；00:05–79:45 | 3EE06131726998E2088672634CC13311B34B35D84DFB9220A95309EC1D69BA83 |
| lecture_15_extracted.txt | 21,463 bytes；只辅助检索，不代替看图 | DD2E63EA1499BCD863880D9BB96F36BA76E10BAAAD8A1E804CB5154ABCA9D094 |

同一 79:45 有两条结束 cue，§31 只使用该秒一次，保证链接秒点唯一。

### 33.2 PDF 视觉检查方法

1. 用文本提取建立页码/关键词索引，但不相信 p.51–60 的 OCR 公式。
2. 用 pypdfium2 渲染 65/65 页普通 PNG 与 65/65 页高分辨率 PNG。
3. 打开 7 张 contact sheet，覆盖 p.1–10、11–20、21–30、31–40、41–50、51–60、61–65。
4. 对 49 个含公式、表格、流程或密集示例的页面逐张打开原分辨率图：

~~~text
p.2,3,5,6,7,9,10,11,12,13,16,17,18,19,20,22,23,24,25,26,
p.28,30,31,33,36,37,38,39,40,42,43,44,45,46,47,48,
p.51,52,53,55,56,57,58,59,60,61,62,63,64
~~~

其余16页通过 contact sheets 与普通单页图检查。关键公式以原分辨率页为准。

### 33.3 主要图表逐项读法

| PDF 页 | 视觉对象 | 图中能支持什么 | 不能推出什么 |
|---:|---|---|---|
| p.2–3 | GPT-3→InstructGPT 与指令控制示例 | 后训练能显著改变接口行为 | 所有能力都由后训练产生 |
| p.6 | SFT→reward model→PPO 箭头 | 经典 InstructGPT 三阶段关系 | 2026 所有系统同配方 |
| p.9 | SFT 数据演化时间线 | 数据从任务改写走向对话/工具轨迹 | 越新必然越好 |
| p.10–13 | 四类数据样本截图 | prompt/response 风格与跨度不同 | 单截图代表全数据质量 |
| p.16–18 | style、偏好与 benchmark 图 | style 会影响偏好和评估 | 长度/礼貌等于能力 |
| p.19–21 | knowledge 图 | 新事实与已有知识调用需分开测 | SFT 永远能/不能教知识 |
| p.23–26 | safety pipeline 与结果 | violation/false refusal 都要测 | 固定样本数跨模型泛化 |
| p.30 | two-phase 示意 | 分阶段可缓和分布跳变 | mid-training 有唯一标准 |
| p.33 | generate–verify gap | 搜索+验证在某些任务有优势 | verifier 一定完整 |
| p.36–38 | pair UI/guideline/标注截图 | rubric 和界面共同定义标签 | preference 脱离协议而存在 |
| p.39–44 | 地区、报酬、人口统计、style | 标注群体会进入数据分布；p.44 比较众包与专家检错差 | 单平台代表所有 RLHF 标注员/劳动者；把 `Baseline` 当相减基准 |
| p.45–48 | LM feedback/self-training/length | 可扩反馈，长度会成捷径 | AI judge 自动消除偏差 |
| p.51–53 | PPO 目标、组件、clip | 五角色关系与更新限制 | clip 等于完整 KL 保证 |
| p.55–58 | DPO 流程与推导 | 给定假设下得到 pair loss | DPO 在所有设置等同 PPO |
| p.59–61 | expert iteration、变体、比较 | 多路线结果依设置 | 曲线给永久排名 |
| p.62–64 | overoptimization/calibration/mode collapse | proxy 与真实质量可分离 | reward 升必然更好 |

本文没有直接嵌入课件截图；表中把坐标、箭头或比较关系转成中文，并明确推论边界。

### 33.4 外部一手补充

- [InstructGPT / Ouyang et al.](https://arxiv.org/abs/2203.02155)：示范、比较数据、reward model 与 PPO。
- [Stiennon et al.](https://arxiv.org/abs/2009.01325)：偏好摘要、BT reward 与 RLHF。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：clipped surrogate objective。
- [DPO 原论文](https://arxiv.org/abs/2305.18290)：KL 正则最优策略、隐式 reward 与 DPO loss。
- [Constitutional AI](https://arxiv.org/abs/2212.08073)：模型反馈与文字原则。
- [Tulu 3](https://arxiv.org/abs/2411.15124)：开放 post-training 案例。
- [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)：proxy 过度优化。
- [Fine-Tuning on New Knowledge and Hallucinations](https://arxiv.org/abs/2405.05904)：§8 的补充边界。
- [AlpacaFarm](https://openreview.net/forum?id=4hturzLcKX)：p.45 的系统级模拟反馈相关与成本图；不支持逐样本 98% 准确。
- [Whose Opinions Do Language Models Reflect?](https://proceedings.mlr.press/v202/santurkar23a.html)：p.43 的群体/意见分布讨论。
- [Human Feedback Is Not Gold Standard](https://openreview.net/forum?id=7W3GLNImfS)：p.44 的 assertiveness/complexity 混杂。
- [A Long Way to Go](https://openreview.net/forum?id=G8LaO1P0xv)：p.48 的长度效应与 59/243-token 个例。
- [Business Insider 原报道](https://www.businessinsider.com/ai-data-labeling-annotators-pay-subject-experts-generalists-gig-workers-2025-12)：p.40 课程采用的新闻截图；这是二手报道，不与论文证据同级。

论文只支持各自实验和公式，不被用来伪造课程没有公开的现代公司配方。Best-of-N、SFT 分母、KL sum/mean、PPO 四格和 DPO 数字均为【补充解释】，不是声称幻灯片逐字给出的数字。

### 33.5 课程内容与补充知识分栏

| 类别 | 本文内容 |
|---|---|
| 【课程内容】 | SFT 数据演化、style/knowledge/safety、偏好标注、PPO/DPO 主线、风险图 |
| 【视频补充】 | prompt masking 问答、标注劳动、长度效应、RLVR 下一讲边界 |
| 【补充解释】 | 四则手算、shape、平均分母、KL sum/mean、PPO 四格、DPO 数字、决策树 |
| 【补充】 | 一手论文核对公式与历史，不替课程宣称 |
| 【延伸】 | GAE、RLVR 等只给入口，跳过不破坏主线 |

## 34. 一页复习流程与学完能力清单

### 34.1 从问题倒推方法

~~~text
先问：模型不会格式，还是不会选好答案？
├─ 不会格式/协议 → SFT；查chat template、mask、有效token分母
└─ 会生成好候选但不常选中
   ├─ 有可靠自动验证 → Best-of-N / rejection / 后续RLVR
   └─ 只有人类偏好
      ├─ 简单离线配方 → DPO类
      └─ 在线探索、有工程预算 → reward model + PPO类

训练中：
chosen必须在前；reference与old分开；KL sum/mean写清；PPO正负A分开算

选模型：
不只看训练reward；同时看人工质量、能力、安全、错误拒绝、
长度、校准、多样性、分布外表现与推理成本
~~~

### 34.2 你现在应能独立完成

- 给 chat transcript 写 assistant-only mask，并用 $`\sum m_t`$ 作分母；
- 区分 prompt、response、token、trajectory、pair 和 batch；
- 手算 SFT cross-entropy，解释 token-average 与 response-average；
- 用四格表同时诊断 violation 和 false refusal；
- 从 reward 差手算 BT 胜率/loss，保持 chosen-rejected 方向；
- 区分 policy、old、reference、reward、value 五角色；
- 复算 KL 的负单项、非负总期望、sequence sum 和 token mean；
- 对 $`A>0/A<0`$、$`r>1/r<1`$ 四种 PPO clip 逐格计算；
- 从 KL 正则目标推到 $`\pi_r`$，再推到 DPO loss；
- 复算 §23 DPO 数字，并解释 $`\beta`$ 不是 learning rate；
- 写出 Best-of-N 公式的独立性/verifier 条件；
- 发现长度 hacking、overoptimization、mode collapse、失准和分布偏移；
- 用 §26 数据卡与 §34 决策树设计可审计 post-training 实验。
