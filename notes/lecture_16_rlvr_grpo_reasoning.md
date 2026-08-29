# Lecture 16：RLVR、GRPO 与推理模型训练

> CS336 Spring 2026 · Reasoning RL
>
> 官方讲义：`lecture_16.pdf`，61 页；官方视频：Stanford Online，约 75:42。
>
> 目标：让只会四则运算的第一次学习者，不看视频也能分清 PPO、GRPO、Dr.GRPO、DeepSeek-R1、Kimi k1.5 与 Qwen3 的公式、数据和工程边界。

## 0. 第一次怎么读、来源标签与全讲地图

### 0.1 第一次阅读顺序

1. 先读 §2 的最小词典，分清 prompt、response、token、group 和 batch。
2. 读 §3–§6，知道为什么奖励可验证以后仍然需要小心做强化学习。
3. 读 §7–§12，拿纸笔重算完整 GRPO group；这是本讲核心。
4. 读 §13–§20，把算法放回 DeepSeek、Kimi、Qwen3 和训练系统。
5. 读 §21–§23 复习，再做 §24 的题，最后用 §26 视频导航补漏。

§1 是五分钟复习卡。第一次阅读可以先跳过：它只负责压缩，不负责第一次教学。

### 0.2 来源标签

- 【课程内容】：PDF 主线或公式。
- 【视频补充】：讲者口头解释、问答或课件未展开的提醒。
- 【补充解释】：为零基础补出的四则运算、形状和类比。
- 【补充】：一手论文给出的核对、澄清或更新。
- 【延伸】：跳过不影响本讲主线。

模型数据、benchmark（基准测试）数字和工程配方是 **2026 年课程时点快照**，不是永久排行或普遍定律。PDF 覆盖见 §27，来源边界见 §28。

### 0.3 一句话地图

~~~text
普通 RLHF：主观 reward model 难保证继续扩展时仍可靠
        ↓
RLVR：改用答案、代码测试、形式证明等可自动检查的奖励
        ↓
同一 prompt 采样一组 responses，用组内结果估计“谁比平均好”
        ↓
GRPO：不训练 value model；用 clipped ratio + reference KL 更新 policy
        ↓
检查偏差：自身参与均值、随机标准差、每回答长度归一化
        ↓
Dr.GRPO 等变体修正；真实系统还要处理 rollout、长 CoT 和 straggler
~~~

核心句：**奖励容易自动检查，不代表奖励完整；优化代码短，不代表估计量无偏；回答变长，不自动证明模型学会了更深推理。**

### 0.4 稳定目录

- [最低词典与数学桥](#l16-vocabulary)
- [RLVR 与奖励边界](#l16-rlvr)
- [PPO 到 GRPO](#l16-grpo)
- [完整 GRPO 手算](#l16-worked-group)
- [偏差、长度与 Dr.GRPO](#l16-bias)
- [DeepSeek、Kimi、Qwen](#l16-cases)
- [系统与决策树](#l16-systems)
- [80 道自测](#l16-questions)
- [80 道答案](#l16-answers)
- [视频导航](#l16-video-nav)
- [PDF 覆盖](#l16-pdf-coverage)
- [来源与视觉核验](#l16-sources)

## 1. 五分钟复习卡（首次阅读先跳过）

1. **RLVR** 是 Reinforcement Learning from Verifiable Rewards，可验证奖励强化学习；验证器检查的是代理条件，不是宇宙真相。详见 §3。
2. policy gradient 的核心是 `reward × log-prob gradient`；高回报动作概率上升，低回报动作概率下降。详见 §4。
3. baseline 只要在给定 prompt 后不依赖当前采样 response，减掉它不会改变期望梯度，却可减小方差。详见 §5、§11。
4. PPO 用新旧策略概率比 $`\rho_t`$ 和 clip 限制一次更新过猛；正、负 advantage 必须分别判断。详见 §6、§9。
5. GRPO 对同一 prompt 采样 $`G`$ 个 responses，以组内奖励均值和标准差得到 advantage，省去 value model。详见 §7–§8。
6. 组内 advantage 不是 value function：它只比较这组样本，换一组数就可能变。详见 §8。
7. 原始 GRPO 的“response 内 token mean”、组标准差和含自身均值会改变权重或产生偏差；Dr.GRPO 去掉其中一些来源。详见 §11–§12。
8. DeepSeek-R1-Zero 展示纯 RL 起步；生产版 R1 加 cold-start SFT（Supervised Fine-Tuning，监督微调：用标准答案做下一 token 训练）、再 RL、再 SFT/RLHF。详见 §13–§15。
9. “aha” 和 CoT 变长是观察，不足以单独证明 RL 从零创造了推理；base model 已可能含类似模式。详见 §13。
10. Kimi k1.5 公开了数据筛选、另一种正则化 PG 和长度奖励；PDF 的 best-of-8 与报告截图的十次估难是两个口径。详见 §16–§17。
11. Qwen3 用 3995 条 cold-start 样本、thinking/non-thinking 融合、一般 RL 和强到弱蒸馏；这是报告配方，不是所有模型定律。详见 §19。
12. 长 CoT 让最慢 rollout 拖住整组；算法、推理引擎、训练引擎和调度必须一起设计。详见 §18。

<a id="l16-vocabulary"></a>

## 2. 最低前置知识、符号和四层平均

### 2.1 从“出题—作答—判卷”开始

【补充解释】把 RLVR 想成数学竞赛训练：

- **prompt $`x`$**：一道题；
- **response $`o_i`$**：第 $`i`$ 份完整解答；
- **token $`o_{i,t}`$**：解答中的第 $`t`$ 个小文字单位；
- **reward $`r_i`$**：判卷得到的分数；
- **policy $`\pi_\theta`$**：当前会写答案的模型，参数是 $`\theta`$；
- **rollout**：模型从 prompt 开始实际采样出一份 response 的过程和结果。

一个 group 是**同一道题**的多份回答；一个 batch 可含多道题的多个 groups：

~~~text
batch
├─ prompt x1
│  ├─ response o1: token 1, 2, ...
│  ├─ response o2: token 1, 2, ...
│  └─ response o3: token 1, 2, ...
└─ prompt x2
   ├─ response o1: ...
   └─ ...
~~~

因此下面四种分母不是一回事：

1. sequence sum：一条回答所有 token 相加；
2. token mean：除以这条回答长度；
3. group mean：除以同题回答数 $`G`$；
4. batch mean：再对 prompts 平均。

### 2.2 概率、log probability 与梯度

- **probability（概率）** 在 0 到 1 之间。
- **log probability（对数概率）** 是 $`\log p`$，本讲 log 指自然对数 $`\ln`$。因为 $`0<p\le1`$，所以 $`\log p\le0`$。
- $`e\approx2.71828`$，$`\log`$ 与指数互逆：$`\log(e^a)=a`$。
- **loss（损失）** 是“坏程度”，训练通常让它变小；**objective（目标）** 若写成最大化，就让它变大。
- **gradient（梯度）** 是参数轻微变化时，目标变化的局部方向和速度。

若当前 token 概率从 0.20 变到 0.24，概率比为 $`0.24/0.20=1.2`$。用 log 写：

```math
\exp(\log 0.24-\log0.20)=\exp(\log(0.24/0.20))=1.2.
```

### 2.3 均值、方差和两种标准差

给 $`G`$ 个数 $`r_1,\dots,r_G`$，均值是：

```math
\bar r=\frac{1}{G}\sum_{i=1}^{G}r_i.
```

**population variance（总体方差）** 把这组数据当完整总体，分母是 $`G`$：

```math
\sigma_{\mathrm{pop}}^2=\frac1G\sum_i(r_i-\bar r)^2.
```

**sample variance（样本方差）** 把这组数据当作更大总体的样本，常用分母 $`G-1`$：

```math
s^2=\frac1{G-1}\sum_i(r_i-\bar r)^2.
```

标准差是方差开平方。平方根 $`\sqrt a`$ 是“乘自己得到 $`a`$ 的非负数”；$`\sqrt{0.5}\approx0.7071`$，因为 $`0.7071^2\approx0.5`$。课程的简化 NumPy 实现用 `numpy.std()` 默认的总体标准差，即分母 $`G`$（PDF p.20）。读其他实现必须先查 `ddof`。

### 2.4 五个策略角色不要混

- **current policy $`\pi_\theta`$**：正在更新的模型。
- **old policy $`\pi_{\theta_{old}}`$**：采样该批 rollout 时的冻结快照，用作 PPO 比率分母。
- **reference policy $`\pi_{ref}`$**：较长期冻结的参考模型，用 KL 约束 current 不要跑太远。
- **value model $`V_\phi`$**：PPO 中预测未来回报的模型；GRPO 的卖点之一是不用它。
- **reward/verifier**：根据完整输出给分；不一定是神经网络。

old 和 reference 可能在某一时刻数值相同，但职责不同。old 会按训练批次更新；reference 通常长期冻结。

### 2.5 本讲符号卡

| 符号 | 含义 | 层级/单位 |
|---|---|---|
| $`x`$ | prompt | 一道题 |
| $`G`$ | 每个 prompt 的 responses 数 | 条 |
| $`o_i`$ | 第 $`i`$ 条完整 response | token 序列 |
| $`\lvert o_i\rvert`$ | 第 $`i`$ 条 response 长度 | token 数 |
| $`t`$ | token 位置 | 整数下标 |
| $`r_i`$ | verifier 给 response 的 reward | 分数，无统一单位 |
| $`A_i`$ | 第 $`i`$ 条 response 的 advantage | 相对分数 |
| $`\rho_{i,t}`$ | current/old token 概率比 | 无单位 |
| $`\epsilon_{clip}`$ | PPO clip 半宽 | 无单位 |
| $`\beta`$ | KL 惩罚系数 | 使两项量级匹配的权重 |

<a id="l16-rlvr"></a>

## 3. 为什么从 RLHF 走到 RLVR

### 3.1 RLHF 外推为什么难

【课程内容，PDF p.1–3】RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）常让 reward model 模仿人的偏好。问题是：策略不断优化后，会探索 reward model 没见过的奇怪输出。

生活类比：老师用“字多、格式整齐”粗略打作文分。学生若只优化这个规则，可能写 20 页空话。分数升了，作文没有更好。这叫 **reward hacking（奖励钻空子）**：策略找到代理指标漏洞。

【视频补充】讲者把它称为过度优化/外推问题：继续往同一个不完美 reward model 加算力，收益可能停滞或反转。不是“所有 RLHF 一定失败”，而是外推风险增加。

### 3.2 RLVR 到底可验证什么

【课程内容，PDF p.2–4】RLVR 是 **Reinforcement Learning from Verifiable Rewards（可验证奖励强化学习）**。典型 verifier（验证器）有：

- 数学题：最终答案能否与标准答案等价；
- 代码题：程序是否通过测试；
- 形式证明：证明是否被 proof checker 接受；
- 格式：是否把答案放进规定标签。

“verifiable”只表示规则可重复执行，不表示规则完整：

先固定二分类标签契约，否则 FP/FN 会说反：

- **actual positive**：response 在真实任务规格下确实正确；
- **predicted positive**：verifier 判定通过；

| | verifier pass（预测正） | verifier reject（预测负） |
|---|---|---|
| actual correct（真实正） | TP：对且通过 | FN：对却被拒绝 |
| actual wrong（真实负） | FP：错却通过 | TN：错且被拒绝 |

例如：数学答案 `42` 实际正确，但 parser 只接受 `\boxed{42}` 而拒绝裸 `42`，这是 **FN**；实际错误答案借格式漏洞被 parser 当成 `42`，是 **FP**。若换成别的“positive”定义，四格名称也会变；本文这一节始终固定“真实正确”为正类。

| 验证器通过 | 仍可能漏掉 |
|---|---|
| 最终答案正确 | 推理过程可能胡写、抄捷径 |
| 单元测试通过 | 隐藏输入、性能、安全、规范 |
| proof checker 接受 | 规格本身写错或过弱 |
| 格式正确 | 内容可能完全错误 |

### 3.3 小奖励例：代理条件不是最终目标

设 reward 为：答案正确给 1，错误给 0；格式正确再加 0.1。

| response | 正确分 | 格式分 | 总 reward |
|---|---:|---:|---:|
| 正确且格式对 | 1 | 0.1 | 1.1 |
| 正确但格式错 | 1 | 0 | 1.0 |
| 错但格式对 | 0 | 0.1 | 0.1 |

这能训练“正确+格式”，却没有直接验证解释是否诚实、是否引用可靠、是否安全。设计 RLVR 必须同时写 **reward contract（奖励契约）**：验证了什么、没验证什么、错误奖励的代价是什么。

格式映射也属于 verifier：`answer=42`、`The answer is 42.`、`\boxed{42}` 可能语义相同，parser 却可能把其中某种解析失败成 FN。反过来，宽松正则若从错误解释里误抓到字符串 `42`，可能造成 FP。上线前应分别造 TP/FP/FN/TN 测试集。

## 4. Policy gradient：奖励怎样进入梯度

### 4.1 从抽奖券类比到公式

策略从动作集合采样回答，目标是最大化期望奖励：

```math
J(\theta)=\mathbb E_{o\sim\pi_\theta(\cdot|x)}[R(x,o)].
```

$`J`$ 是越大越好的目标；$`\mathbb E`$ 表示按模型概率加权平均；$`R`$ 是完整回答奖励。REINFORCE 的 log-derivative trick 给出：

```math
\nabla_\theta J
=\mathbb E\left[R(x,o)\nabla_\theta\log\pi_\theta(o|x)\right].
```

人话：抽到好回答，就沿“提高这条回答 log 概率”的方向走；抽到坏回答，就少鼓励或反向压低。

### 4.2 为什么乘 log 概率梯度

离散回答时：

```math
J=\sum_o\pi_\theta(o|x)R(o).
```

因为 $`\nabla \pi=\pi\nabla\log\pi`$，所以：

```math
\nabla J
=\sum_oR(o)\nabla\pi(o)
=\sum_o\pi(o)R(o)\nabla\log\pi(o).
```

最后一行正是“按当前策略采样后求平均”的形式。reward 本身不必可微；只要能打分，梯度通过 log probability 走回模型。

### 4.3 两动作手算

模型答 A 的概率 0.25、答 B 的概率 0.75；奖励 $`R(A)=1,R(B)=0`$。期望奖励：

```math
J=0.25\times1+0.75\times0=0.25.
```

若更新让 A 概率升到 0.30，B 自动降到 0.70：

```math
J_{\mathrm{new}}=0.30\times1+0.70\times0=0.30.
```

增加 $`0.30-0.25=0.05`$。但一次采样可能碰巧只看到 B，估计会很吵，这就是 **high variance（高方差）**：重复估计差别很大。

### 4.4 stop-gradient / detach：先画清梯度能走哪条路

**stop-gradient（停止梯度）** 或 **detach（脱离计算图）** 的意思是：本次更新仍使用这个数值，但把它当成常数，它的导数按 0 处理。这里的“常数”只针对**当前这一次 policy update**，不是说它在下一批数据里永远不变。

一次最小 policy-gradient 更新可读成下面的骨架伪代码：

```python
# 骨架伪代码：函数名只表示职责，不是一段可直接运行的完整程序
responses = sample(old_policy, prompts)              # 固定样本，不反传
rewards = verifier(responses)                        # 固定分数，不反传
group_mean, group_std = group_statistics(rewards)    # 固定统计量，不反传
advantages = detach(normalize(rewards, group_mean, group_std))
old_logprob = detach(old_policy.logprob(responses))
ref_logprob = detach(reference_policy.logprob(responses))
current_logprob = current_policy.logprob(responses)  # 唯一连到参数 theta 的路径
loss = policy_objective(current_logprob, old_logprob,
                        ref_logprob, advantages)
loss.backward()                                      # 只更新 current policy
```

因此，本次更新中要视为常数的是：已经采出的 responses、reward、advantage、group mean/std、old log-prob 和 reference log-prob；只有重新计算的 **current log-prob** 对 $`\theta`$ 求导。若忘记 detach，梯度可能错误地穿过 advantage 或旧模型；若把 current log-prob 也 detach，就完全没有 policy 梯度。后文 §11.1 的 baseline 证明和 §17.3 的 Kimi 梯度都沿用这条梯度路径。

## 5. Baseline、TRPO 与 PPO 的直觉

### 5.1 baseline 为什么有用

把 reward 换成 advantage：

```math
A(x,o)=R(x,o)-b(x).
```

**baseline（基线）** 是“这题通常能得多少”。若某题平均 0.8，得 1 只比预期好 0.2；另一难题平均 0.1，得 1 比预期好 0.9。相对比较通常比裸 reward 方差更小。

**state（状态）** 是模型在选择下一个动作时已经掌握的信息；在语言模型里就是 prompt 加上已经生成的前缀。基线可以依赖 prompt/state，却不能依赖当前采样动作。严格证明放在 §11。

### 5.2 TRPO：别一步把策略推翻

【课程内容，PDF p.5–8】TRPO 是 **Trust Region Policy Optimization（信赖域策略优化）**。生活类比：改菜谱时一次只改一点，否则即使昨天的评分方向正确，跨太远也会失效。

TRPO 把更新限制在一个 KL 邻域内，理论清楚但实现较复杂。KL（Kullback–Leibler divergence，KL 散度）是两概率分布差异的一种方向性量；不是普通距离，因为交换两边通常不同。

### 5.3 PPO：把硬约束变成可剪切目标

PPO 是 **Proximal Policy Optimization（近端策略优化）**。定义 token 概率比：

```math
\rho_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}.
```

- 分子：current policy 给已采样 token 的概率；
- 分母：old policy 采样时给它的概率；
- $`\rho=1`$：没变化；
- $`\rho=1.3`$：概率变成 1.3 倍；
- $`\rho=0.7`$：概率变成 0.7 倍。

**surrogate（替代目标）** 是“比原始目标更容易优化、希望近似其更新效果的公式”；它不是环境的真实 reward。PPO 的 clipped surrogate：

```math
L^{clip}_t=\min\left(
\rho_t A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right).
```

`clip` 把数夹在区间内。若 $`\epsilon=0.2`$，区间是 $`[0.8,1.2]`$：1.4 夹成 1.2，0.6 夹成 0.8，1.1 不变。

### 5.4 正、负 advantage 为什么分支不同

设 $`A=+2,\rho=1.4`$：

- 裸项 $`1.4\times2=2.8`$；
- clip 项 $`1.2\times2=2.4`$；
- min 取 2.4，阻止过度提高好动作概率。

设 $`A=-2,\rho=0.6`$：

- 裸项 $`0.6\times(-2)=-1.2`$；
- clip 项 $`0.8\times(-2)=-1.6`$；
- min 取 $`-1.6`$，阻止过度降低坏动作概率。

只背“ratio 超界就裁剪”会做错。min 和 advantage 符号共同决定分支。

## 6. 语言模型 PPO 为什么工程复杂

### 6.1 token 是动作，response 是轨迹

【课程内容，PDF p.6–15】生成第 $`t`$ 个 token 前的 prompt 与已有前缀是 state $`s_t`$，新 token 是 action $`a_t`$。完整回答才拿到终局 reward。PPO 通常需要：

1. policy 生成 rollout；
2. reward/verifier 打分；
3. value model 预测每 token 的回报；
4. reference 给 KL；
5. old policy 给概率比分母；
6. 反向传播更新 policy/value。

这会同时占模型显存、**KV cache（键值缓存）**、**activation（激活）** 和通信资源。KV cache 是为已生成前缀保存 attention 的 key/value 中间量，避免每步全部重算；activation 是前向计算产生、反向传播可能还要使用的中间 tensor。

### 6.2 value、advantage 与 GAE

**value function $`V(s_t)`$** 预测从当前前缀继续时的期望总回报。最简单 advantage 可写“实际回报减预测”。GAE 是 **Generalized Advantage Estimation（广义优势估计）**，把多步时序差分按衰减权重混合，以权衡噪声和偏差。

本讲只需知道：GAE 不是一个神秘奖励；它是 PPO 里构造 token-level advantage 的方法。GRPO 用同题组内相对奖励替代 value model，是减少内存和复杂度的关键。

### 6.3 per-token KL 的口径

对生成 token，常见样本级 log-ratio 是：

```math
d_t=\log\pi_\theta(o_t|x,o_{<t})-
\log\pi_{ref}(o_t|x,o_{<t}).
```

单个 sampled log-ratio 可以正或负；理论 $`D_{KL}(current\|reference)`$ 是对 current policy 的期望，所以非负。课程 p.14 的单边 clamp 与 GRPO 常用的逐样本非负 $`e^d-d-1`$ estimator 不是同一公式，见 §10。必须写明“token 求和”还是“token 平均”；回答长短不同时结果会变。

<a id="l16-grpo"></a>

## 7. GRPO：不用 value model 的 group-relative 更新

### 7.1 Group Relative Policy Optimization

【课程内容，PDF p.16–18】GRPO 是 **Group Relative Policy Optimization（组相对策略优化）**。对每个 prompt $`x`$：

1. old policy 采样 $`G`$ 条 responses $`o_1,\ldots,o_G`$；
2. verifier 给 rewards $`r_1,\ldots,r_G`$；
3. 组内标准化得到每条 response 的 $`A_i`$；
4. 把同一个 $`A_i`$ 赋给该 response 的每个生成 token；
5. 用 PPO clip 与 reference KL 更新 current policy。

它没有训练一个 $`V_\phi`$。因此组内 advantage **不是 value function**，也不预测未来；它只是“这条回答比这次同题同组平均好多少”。

### 7.2 原始组内 advantage

```math
A_i=\frac{r_i-\bar r}{\sigma_r+\varepsilon_{std}},
\qquad
\bar r=\frac1G\sum_{j=1}^{G}r_j.
```

- $`r_i`$：第 $`i`$ 条完整回答 reward；
- $`\bar r`$：同一 prompt 的 group mean；
- $`\sigma_r`$：课程实现中的 group population std；
- $`\varepsilon_{std}`$：防止除以 0 的小数；PDF p.20 代码为 $`10^{-4}`$；
- $`A_i`$：无单位相对分数。

同一 $`A_i`$ 常复制到回答内每个 token。这不表示每个 token 都独立被 verifier 判过。

### 7.3 原始论文总览式与 token 实现式不能混

【课程材料符号边界】PDF p.18 截取的原始 GRPO 总览式直接写 response 概率比

```math
\frac{\pi_\theta(o_i\mid x)}{\pi_{old}(o_i\mid x)},
```

即一条 response 一个 ratio/项。PDF p.23 在讨论真实训练偏差时，才明确展开为每个生成 token 的 ratio、先对 token 求和，并在原 GRPO 中除以本条长度 $`|o_i|`$。下面的逐 token 手算采用 **p.23 的 implementation-shaped（实现形状）公式**，不是偷偷声称 p.18 已经写出内层 token 求和：

```math
J_{\mathrm{GRPO}}=\mathbb E\left[
\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}
\left(
\min(\rho_{i,t}A_i,\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i)
-\beta\widehat D_{\mathrm{KL},i,t}
\right)
\right].
```

从里到外读这份 token 实现式：

1. 每 token 算 clipped PG 项和 KL；
2. 除以该 response 长度 $`|o_i|`$，得到 response 内 token mean；
3. 对 $`G`$ 条 responses 求 group mean；
4. 对 prompts/batches 求期望。

若代码最小化 loss，则写 $`L=-J`$。忘记负号会把更新方向说反。

### 7.4 online 第一小步为何 ratio 可能是 1

rollout 刚由 current policy 采样并复制为 old policy 时，二者相同：

```math
\rho=\pi_\theta/\pi_{\mathrm{old}}=1.
```

**minibatch（小批）** 是把本轮 rollout batch 再切成若干小块逐次更新；**epoch（遍历轮）** 是把同一批数据完整过一遍。第一次更新内 clip 不起作用；做多个 minibatch、多个 epoch 后 current 变化，ratio 才偏离 1。clip 仍有意义，因为它约束同一批数据上的后续复用。

<a id="l16-worked-group"></a>

## 8. 完整 group 手算：均值、两种标准差、零方差

### 8.1 奖励组

同一道题采样 $`G=4`$ 条回答，rewards：

```math
[r_1,r_2,r_3,r_4]=[0,1,1,2].
```

均值：

```math
\bar r=(0+1+1+2)/4=4/4=1.
```

离均差：$`[-1,0,0,1]`$；平方：$`[1,0,0,1]`$；平方和是 2。

### 8.2 population std：课程代码口径

```math
\sigma_{\mathrm{pop}}^2=2/4=0.5,
\qquad
\sigma_{\mathrm{pop}}=\sqrt{0.5}\approx0.7071.
```

先忽略 $`10^{-4}`$ 时：

| response | reward | $`r_i-\bar r`$ | $`A_i=(r_i-\bar r)/0.7071`$ |
|---|---:|---:|---:|
| 1 | 0 | -1 | -1.4142 |
| 2 | 1 | 0 | 0 |
| 3 | 1 | 0 | 0 |
| 4 | 2 | 1 | 1.4142 |

加 $`10^{-4}`$ 后分母 0.7072，数值略变为约 $`\pm1.4140`$。这只是数值稳定项，不是新奖励。

### 8.3 sample std：另一口径，不可混用

若某实现用样本标准差：

```math
s^2=2/(4-1)=2/3\approx0.6667,
\qquad s\approx0.8165.
```

优势变为 $`[-1.2247,0,0,1.2247]`$。两套都可定义，但不能拿一种标准差算正文、另一种标准差验代码。

### 8.4 零方差组

若 rewards 是 $`[1,1,1,1]`$：均值 1，每个分子都为 0，标准差为 0。

- 加 epsilon：$`A_i=0/(0+10^{-4})=0`$；
- 跳过 group：避免做没有相对信号的更新；
- 不能写成 $`0/0`$，那会得到 NaN（Not a Number，非法数值）。

epsilon 只防除零，**不会凭空造 reward advantage**。因此：

- 只看 policy-gradient/reward 项时，所有 $`A_i=0`$，该项更新为 0；
- 若实现直接 skip 整个 group，则这组总更新为 0；
- 若不 skip，且仍保留 $`\beta>0`$ 的 reference KL，current 又不同于 reference，则 KL 正则项仍可能有梯度，把 current 往 reference 拉。不能把“reward 没信号”写成“完整 loss 一定没梯度”。

同理 $`G=1`$ 时组内没有比较信息。实现必须定义行为。

## 9. 完整 token clip 表：正负 advantage 都算

### 9.1 设置

继续用 §8 的四条 responses，长度为 $`[2,1,2,1]`$，共 6 个生成 token。取 $`\epsilon=0.2`$，clip 区间 $`[0.8,1.2]`$。为突出 PPO 项，先令 KL 为 0。

给每 token 的 current/old 概率比：

| response | token ratios |
|---|---|
| 1，$`A=-1.4142`$ | 0.7，1.1 |
| 2，$`A=0`$ | 1.3 |
| 3，$`A=0`$ | 0.9，1.1 |
| 4，$`A=1.4142`$ | 1.3 |

### 9.2 每个 token 逐项算

| resp/token | $`\rho`$ | clip($`\rho`$) | $`\rho A`$ | clip$`\times A`$ | min |
|---|---:|---:|---:|---:|---:|
| 1/1 | 0.7 | 0.8 | -0.9899 | -1.1314 | -1.1314 |
| 1/2 | 1.1 | 1.1 | -1.5556 | -1.5556 | -1.5556 |
| 2/1 | 1.3 | 1.2 | 0 | 0 | 0 |
| 3/1 | 0.9 | 0.9 | 0 | 0 | 0 |
| 3/2 | 1.1 | 1.1 | 0 | 0 | 0 |
| 4/1 | 1.3 | 1.2 | 1.8385 | 1.6970 | 1.6970 |

第一格验证：$`0.7\times(-1.4142)=-0.98994`$；$`0.8\times(-1.4142)=-1.13136`$；min 是更小的 $`-1.13136`$。最后一格：$`1.3\times1.4142=1.83846`$，裁剪后 $`1.2\times1.4142=1.69704`$。

### 9.3 response mean 与 global token mean 不同

原始 GRPO 先做每回答 token mean：

- response 1：$`(-1.1314-1.5556)/2=-1.3435`$；
- response 2：0；
- response 3：0；
- response 4：1.6970。

再除以 $`G=4`$：

```math
J_{\mathrm{PG}}=(-1.3435+0+0+1.6970)/4=0.088375.
```

若改成全局 token mean：

```math
(-1.1314-1.5556+1.6970)/6=-0.1650.
```

一个为正、一个为负。原因不是算错，而是**分母定义改变了每条长短回答的权重**。实现和论文比较必须先对齐聚合口径。

## 10. Reference KL：别把 old 与 reference 混在一起

### 10.1 两个约束各管什么

- old policy 出现在 $`\rho=\pi_\theta/\pi_{\mathrm{old}}`$，约束“对这一批 rollout 别一次改太猛”；
- reference 出现在 KL，约束“长期别离出发模型太远”。

### 10.2 三种很像、其实不同的 KL 量

PDF p.14 与 p.18 展示了不同实践，必须拆开。

先定义 **support（支持集）**：一个概率分布里“概率不为 0 的动作集合”。例如 $`[0.6,0.4,0]`$ 的 support 是前两个动作，不含第三个。下面出现 $`\pi_{ref}(a)/\pi_\theta(a)`$ 时，分母必须大于 0；而要把期望中的求和完整还原成 $`\sum_a\pi_{ref}(a)=1`$，current policy 的 support 还必须覆盖 reference 的全部概率质量。最省心的充分条件是双方在同一组动作上都有正概率；普通有限-logit softmax 通常满足，但截断采样、top-k 或硬 mask 可能破坏它。

**第一种：裸 sampled log-ratio。** 对 current 采到的一个 token $`a`$：

```math
g(a)=\log\pi_\theta(a)-\log\pi_{ref}(a).
```

它逐样本可正可负。例如 current/ref 概率为 $`0.4/0.2`$，$`g=\log2=0.6931`$；若为 $`0.2/0.4`$，$`g=-0.6931`$。只有在 $`a\sim\pi_\theta`$、覆盖支持集且 log-prob 精确时，期望才是：

```math
\mathbb E_{a\sim\pi_\theta}[g(a)]
=D_{KL}(\pi_\theta\|\pi_{ref})\ge0.
```

**第二种：p.14 的单边 reward-shaping heuristic。** **heuristic（启发式）** 是经验上可能有用、但并非由目标严格推出的规则；**reward shaping（奖励塑形）** 是先修改喂给学习器的奖励信号。课程截图代码是：

```python
kl_one_sided = torch.clamp(logprobs - ref_logprobs, min=0.0)
```

即 $`\max(g,0)`$。若 $`g=-0.4`$，clamp 后 0；若 $`g=0.7`$，保留 0.7。它逐样本非负、可防某一方向的分数爆大，但它是**单边启发式 shaping**，不是对称距离，也不是下一个 estimator 的同义写法。

**第三种：p.18/GRPO 常见的 $`k_3`$ estimator。** 令相反方向的样本 log-ratio：

```math
d=\log\pi_{ref}(a)-\log\pi_\theta(a)=-g(a),
```

则：

```math
\widehat D_{k_3}=e^d-d-1.
```

因为对任意实数 $`d`$，$`e^d\ge1+d`$，所以它**每个样本都非负**。若 $`d=\log2\approx0.6931`$：

```math
e^{0.6931}-0.6931-1\approx2-0.6931-1=0.3069.
```

在 $`a\sim\pi_\theta`$、current/reference 概率均精确且支持集条件成立时：

```math
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}[e^d-d-1]
&=\sum_a\pi_\theta(a)\frac{\pi_{ref}(a)}{\pi_\theta(a)}
-\mathbb E[d]-1\\
&=1-\mathbb E[\log\pi_{ref}-\log\pi_\theta]-1\\
&=\mathbb E[\log\pi_\theta-\log\pi_{ref}]\\
&=D_{KL}(\pi_\theta\|\pi_{ref}).
\end{aligned}
```

这里的 **Monte Carlo estimator（蒙特卡洛估计量）** 是“用随机抽到的少量样本平均，近似无法逐项枚举的完整期望”。在上述共同正 support、$`a\sim\pi_\theta`$、精确 log-prob 条件下，它是 forward $`D_{KL}(current\|reference)`$ 的无偏 Monte Carlo estimator；不要只凭 $`d`$ 的书写方向把它叫“reverse KL”。off-policy 样本、近似 log-prob、截断/温度不一致都会破坏这条等式。

两动作反例能看清 support 为什么不是小字备注。令：

```math
\pi_\theta=[1,0],\qquad \pi_{ref}=[0.5,0.5].
```

current 只能采到动作 A；在 A 上 $`\pi_{ref}/\pi_\theta=0.5/1=0.5`$，所以：

```math
\mathbb E_{a\sim\pi_\theta}\left[\frac{\pi_{ref}(a)}{\pi_\theta(a)}\right]
=1\times0.5=0.5\ne1.
```

reference 在动作 B 上还有 0.5 概率，但 current 永远采不到 B；而 B 上又会出现 $`0.5/0`$ 的非法除法。于是上面把第一项写成 1 的桥断了，$`k_3`$ 的无偏等式也不能照搬。

若 $`\beta=0.05`$，数例的 penalty 为 $`0.05\times0.3069=0.015345`$。若 $`d`$ 是整条 response 的 sequence log-ratio，结果是 response-level；若 $`d`$ 是某 token，再聚合就是 token-level。代码必须声明层级。

### 10.3 sum 还是 mean

三 token KL estimates 为 $`[0.1,0.2,0.3]`$：

- sequence sum：$`0.1+0.2+0.3=0.6`$；
- token mean：$`0.6/3=0.2`$。

长度 30 的回答若每 token 同为 0.2，sum 是 6，mean 仍是 0.2。两种正则对长回答施加的总压力不同，不能只写“加 KL”而不写分母。

<a id="l16-bias"></a>

## 11. GRPO baseline：什么无偏，什么不一定

### 11.1 state-dependent baseline 为什么不改期望梯度

本节沿用 §4.4 的 stop-gradient 契约：sample、reward 与 baseline 数值在当前更新中固定，只有 current log-prob 对 $`\theta`$ 求导。

固定 prompt $`x`$，baseline $`b(x)`$ 不依赖采样 response $`o`$：

```math
\begin{aligned}
\mathbb E_{o\sim\pi}[b(x)\nabla\log\pi(o|x)]
&=b(x)\sum_o\pi(o|x)\nabla\log\pi(o|x)\\
&=b(x)\sum_o\nabla\pi(o|x)\\
&=b(x)\nabla\sum_o\pi(o|x)\\
&=b(x)\nabla1=0.
\end{aligned}
```

所以减去它只改估计噪声，不改期望方向。关键条件是：给定 state/prompt 后，它不看当前 sampled action。

### 11.2 group mean 含自身：有固定缩放

先固定成立条件：给定同一 prompt，$`G`$ 条 responses 是 **IID（independent and identically distributed，独立同分布）** 采样。**独立**是某条 rollout 的随机生成不查看其他 rollout 的结果；**同分布**是它们都来自同一个 policy 与同一套采样配置。本段只研究 **减 group mean**，暂时不含随机 std、PPO clip、KL 和每回答长度权重。以这 $`G`$ 个 samples 的均值为 baseline 时，第 $`i`$ 项：

```math
r_i-\bar r
=r_i-\frac{r_i+\sum_{j\ne i}r_j}{G}
=\frac{G-1}{G}r_i-\frac1G\sum_{j\ne i}r_j.
```

第二项相对 $`o_i`$ 独立，可作 baseline；第一项把有效 reward 乘了 $`(G-1)/G`$。例如 $`G=4`$，缩放是 $`3/4=0.75`$。若只做 mean subtraction，可乘 $`G/(G-1)=4/3`$ 校正固定比例。

这不是说“完整 GRPO 只差一个常数”，也不是说“GRPO 完全没梯度”；它只证明 **mean-subtraction 这一项**相对理想 policy gradient 有固定比例。把 std、clip、token 权重加回来后，不能继续沿用这一个比例概括整个算法。

### 11.3 leave-one-out baseline

**leave-one-out（留一法）** 对第 $`i`$ 条回答，只用另外 $`G-1`$ 条的均值：

```math
b_{-i}=\frac1{G-1}\sum_{j\ne i}r_j.
```

在 §8 的 $`[0,1,1,2]`$ 中：

- $`i=1`$：别人均值 $`(1+1+2)/3=4/3`$，advantage $`0-4/3=-4/3`$；
- $`i=2`$：别人均值 $`(0+1+2)/3=1`$，advantage 0；
- $`i=3`$：同样 0；
- $`i=4`$：别人均值 $`(0+1+1)/3=2/3`$，advantage $`2-2/3=4/3`$。

在给定 prompt 后其他 rollouts 与当前 $`o_i`$ 条件独立时，$`b_{-i}`$ 不包含自身，因此满足 action-independent 条件。若采样过程共享自适应状态、后一次会看前一次，条件独立还要重新检查。

### 11.4 除 group std 为什么更麻烦

原 GRPO 还除以由全组 rewards 计算的随机 $`\sigma_r`$。改变当前 $`o_i`$ 会改变 $`r_i`$，进而改变均值和标准差；分母依赖当前 action。它通常不能直接套 §11.1 的零期望 baseline 证明，因此一般不是原始目标的严格无偏估计。

更直白地说：

- rewards $`[0,0,0,1]`$ 的 std 与 $`[0,1,1,1]`$ 相同但方向分布不同；
- 很容易或很难、但仍有非零离差的题，若 $`\sigma`$ 很小，会被 $`1/(\sigma+\epsilon)`$ 放大；完全相等时分子全 0，不会被放大成 reward 信号；
- epsilon 防数值崩溃，不自动修统计偏差。

课程与 Dr.GRPO 论文的批评重点正是：**“减均值可用 baseline 理论解释”不等于“再除一个含当前样本的随机 std 也自动无偏”。**

视频约 22 分钟处口头用了 “descend the reward”。按本文符号 $`J`$ 是越大越好的 reward objective，正确方向是 **ascend reward**；若代码把 $`loss=-J`$ 定义成越小越好的 loss，才说 **descend negative loss**。这只是符号口径纠正，不改变讲者在讨论“更新是否沿原 reward 正确方向”的问题。

## 12. Length bias 与 Dr.GRPO

### 12.1 原始每回答 $`1/|o_i|`$ 做了什么

原始目标先让每条 response 的 token 项求平均。因此一个 response 的总权重不随长度线性增加。

若同一个负 advantage $`A=-1`$：

- 长度 2，每 token 约分到 $`-1/2=-0.5`$；
- 长度 10，每 token 约分到 $`-1/10=-0.1`$。

错误长回答的每 token 惩罚更弱；正确短回答的每 token 奖励更强。训练动态可能鼓励错误时继续写长，或改变长度分布。这是优化权重效应，不是语言上“长一定坏”。

### 12.2 Dr.GRPO 的教学口径

【课程内容，PDF p.21–24】Dr.GRPO 来自 *Understanding R1-Zero-Like Training: A Critical Perspective*。按 p.23 的简化公式（先略 KL）直接写：

```math
J_{\mathrm{Dr.GRPO}}
=\frac1G\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left[
\rho_{i,t}\hat A_i,
\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right],
```

```math
\hat A_i=r_i-\bar r,
\qquad
\bar r=\frac1G\sum_{j=1}^{G}r_j.
```

和原 GRPO 对照，p.23 明确删除两项：

1. 删除每条 response 自己的 $`1/|o_i|`$，保留 token **sum**；
2. 删除 group standard deviation，不再除 $`\sigma_r`$。

但它**仍保留含自身的** $`r_i-\bar r`$，并没有自动变成 leave-one-out。按 §11.2 的 iid、同 prompt、只看 mean-subtraction 条件，其期望 reward-gradient 是理想 REINFORCE/LOO 方向的固定 $`(G-1)/G`$ 缩放。固定正比例不改变 ascent 方向，可吸收到学习率，所以论文/课件称其去掉关键偏差；这不等于逐样本 estimator 与 LOO 完全相同。

例如 $`G=4`$，固定比例为 $`3/4`$；LOO advantages 是 $`[-4/3,0,0,4/3]`$，含自身均值是 $`[-1,0,0,1]`$，恰好前者乘 $`3/4`$。因此课件脚注说 “pretty close to REINFORCE with leave-one-out”，不能改写成“Dr.GRPO 就是 LOO”。

不同代码库的 `Dr.GRPO` 细节可能不同。必须核对公式、mask 和总分母，不能只看算法名字。

### 12.3 common denominator 小例（补充解释）

PDF p.23 的核心是“不各除自己的长度”。为让数值小一些，假设某实现对整个例子统一除以同一个最大生成长度 10；这是教学换算，不是说幻灯片公式额外写了这个 10：

- 短回答每 token 系数 $`-1/10`$，2 token 总 $`-0.2`$；
- 长回答每 token 系数 $`-1/10`$，10 token 总 $`-1.0`$。

长回答实际用了更多动作，因此总影响更大；不再因为“每条都先平均到 1”而抹平 token 数。是否采用最大长度、全局总 token 或固定常数，是具体实现选择。

### 12.4 “长度增长”和“aha”不能直接做因果结论

【视频补充】课上提醒：base model 本来就可能有自我检查、回溯和较长 CoT（Chain of Thought，思维链）模式；RL 可能提高它们出现概率。看到平均长度变长和 benchmark 上升，只能说明相关共变，不能单凭图证明“长度导致能力”或“RL 从零发明 aha”。

<a id="l16-cases"></a>

## 13. DeepSeek-R1-Zero：纯 RL 案例该怎样读

### 13.1 起点与奖励

【课程内容，PDF p.25–30】DeepSeek-R1-Zero 从 DeepSeek-V3-Base 出发，不先做 reasoning SFT，使用 GRPO 和可验证奖励。课程列出两类主要 reward：

- accuracy reward：数学答案或代码测试的结果；
- format reward：要求推理/答案放在指定标签内。

数据细节没有完全公开，所以不能从论文名称反推出精确 prompt 总数或全部过滤规则。

### 13.2 outcome reward 与 process reward

- **ORM（Outcome Reward Model，结果奖励）** 只看最终结果；
- **PRM（Process Reward Model，过程奖励）** 给中间推理步骤打分。

R1-Zero 的展示强调简单结果奖励也能产生强学习信号。这不证明 PRM 永远无用；它只说明在该模型、数据、训练预算和实现下，团队报告的 PRM 路线没有带来预期收益（PDF p.38）。

### 13.3 “aha moment”要降温解读

【课程内容/视频补充，PDF p.29–30】论文展示模型开始回溯、自我验证、写更长 CoT。三层说法要分开：

1. **观察**：样本输出出现“等等，我再检查”等文字；
2. **统计相关**：RL 训练进度、长度和部分准确率一起变化；
3. **因果主张**：“RL 从零创造新推理算法”——前两项不足以证明。

Dr.GRPO 论文后来指出 DeepSeek-V3-Base 已可出现类似行为。更稳妥的解释是：RL 可能重新加权、放大 base model 已有模式，同时也可能产生长度优化偏差。

### 13.4 reward 能验证与漏掉的东西

若最终答案 42 正确，reward=1；但中间写“7×8=42”是错等式。ORM 仍可能给 1。若训练目标只看末答案，它不会直接惩罚错误中间步骤。评估必须另查过程忠实性、鲁棒性和作弊路径。

## 14. DeepSeek-R1：从研究演示到多阶段产品配方

### 14.1 流程

【课程内容，PDF p.31–36】生产版 R1 的简化链：

~~~text
DeepSeek-V3-Base
→ 少量 cold-start reasoning SFT
→ reasoning RL（GRPO）
→ 收集/过滤大量 reasoning + non-reasoning 数据
→ SFT
→ 面向帮助性/安全性等的一般 RLHF
~~~

**cold-start SFT** 是 RL 前用高质量示范把输出格式、可读性和初步策略放到较好起点。它不是“重新预训练”。

### 14.2 课程公开数字与未知项

【课程内容，PDF p.33–35】课件写：

- DeepSeek-R1 cold-start：公开报告只说 **small amount / 少量**长 CoT；没有公开可核验的精确条数，本文不填 1K；
- 后续收集：约 600k reasoning、200k non-reasoning；
- SFT：课件提到两轮 epoch 的口径；
- reasoning 数据由模型生成、过滤，并借助 DeepSeek-V3 judge。

数字是论文/课程快照。公开描述不足以重建所有提示、采样温度、去重、拒绝标准和人工审计。

【课程材料来源拆分】PDF p.33 上“1K Math and science questions + Long CoTs”与右侧 s1 点，来自独立的 [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) 案例：s1-32B 在 s1K（1000 examples）上做 SFT，并结合 budget forcing。它不是 DeepSeek-R1 cold-start 数量证据。课件把这张样本效率图放在 R1 讨论附近，容易误归；本文把它单独记录。

### 14.3 language consistency reward

R1-Zero 出现语言混杂。生产流程加入语言一致性 reward，使目标语言更稳定。课程图提示它可能让某些 reasoning 指标略降：这是多目标权衡，不是“加规则必然全面提升”。

小例：accuracy 1、language consistency 0.8，若权重各 1，总 reward 1.8；若另一个回答 accuracy 1、consistency 1，总 2。模型会偏向后者，即使答案正确性相同。

## 15. Distillation 与“未成功路线”的证据边界

### 15.1 从 R1 轨迹蒸馏到小模型

【课程内容，PDF p.36–37】**distillation（蒸馏）** 让较小 student 模型学习较强 teacher 生成的数据/分布。课件写 R1 生成约 800k traces，用于 Qwen2.5/Llama 系列学生模型。

蒸馏不是把 teacher 权重复制进去。它更像让学生看老师的解题册：

- 优点：无需小模型从头探索所有 RL 轨迹；
- 局限：teacher 错误、风格和 verifier 偏差也会进入数据；
- 结果：能在特定 benchmark 提升，不保证完全继承 teacher 的隐藏能力。

### 15.2 PRM/MCTS “unsuccessful”不等于永远无用

MCTS 是 **Monte Carlo Tree Search（蒙特卡洛树搜索）**，通过分支、评估和回传在解题树中搜索。R1 报告说团队尝试的 PRM/MCTS 路线在其设置下遇到困难。正确结论：

> 在其任务、模型、计算预算、process labels 和搜索实现下未达到预期。

错误结论：

> 所有 PRM/MCTS 在任何未来模型上都无用。

## 16. Kimi k1.5：数据、难度过滤与课程内部口径

### 16.1 数据依然决定 RL 学什么

【课程内容，PDF p.39–41】Kimi k1.5 强调 broad coverage、difficulty filtering、可验证数学/代码 reward 和 curriculum。**curriculum（课程式训练）** 是在训练的不同阶段改变总体难度或长度分布，常见直觉是先建立较容易的基础，再逐渐加入更难、更长的任务。

若训练集只有竞赛代数，RL 不会自动获得网页检索、客服沟通和医学判断能力。可验证性让某种技能容易打分，不等于覆盖所有技能。

### 16.2 best-of-8 难度小例

PDF p.41 的课程 bullet 说：只保留模型在 best-of-8 中失败的问题。若每次成功概率 $`p=0.1`$，并暂时假设 8 次条件独立：

```math
P(\text{8 次全失败})=(1-0.1)^8=0.9^8\approx0.4305.
```

至少一次成功：

```math
1-0.4305=0.5695.
```

所以一个真实成功率 10% 的题，仍有约 43.05% 概率被“8 次全失败”选为难题。筛选是有噪声的。

### 16.3 [课程材料口径差异] 8 次还是 10 次

PDF bullet/课堂口述用 **best-of-8**；同页嵌入的 Kimi 报告文字写，为估计难度让模型以高温回答 **十次**。本文保留两者：前者是讲者的课程概括，后者是报告截图的具体估计口径。它们不能暗自当成同一个数字。

### 16.4 SFT 证据边界

【视频补充】课程指出 Kimi 报告对 SFT 初始化细节没有完整展开。我们只能说其 pipeline 中存在 warmup/起始策略与 RL 数据工程，不能伪造未公开 SFT 样本量。

### 16.5 prioritized sampling、代码与数学 verifier

【课程内容，PDF p.44】这里更准确叫 **prioritized sampling（优先级采样）**：在当前训练阶段内，按模型当前成功率调整某道题被抽到的概率。它和 curriculum 有联系，但不是同义词：

| 机制 | 改变什么 | 最小例子 |
|---|---|---|
| curriculum | 跨训练阶段改变整体题目难度/长度 | 阶段1以短题为主，阶段2加入长难题 |
| prioritized sampling | 当前阶段内改变各题抽样概率 | 同一题池中，成功率低但仍可学的题多抽 |

若题 $`i`$ 当前成功率为 $`s_i`$，课程写相对抽样权重：

```math
w_i\propto1-s_i.
```

$`\propto`$ 表示“只给相对权重”；要变成概率还要归一化：

```math
p_i=\frac{1-s_i}{\sum_j(1-s_j)}.
```

三题成功率 $`[0.2,0.8,0.5]`$，未归一权重 $`[0.8,0.2,0.5]`$，总和 1.5：

```math
p=[0.8/1.5,\ 0.2/1.5,\ 0.5/1.5]
\approx[0.5333,0.1333,0.3333].
```

已掌握的第二题少采；仍可能学会的题多采。若成功率估计噪声大或 verifier 有偏，sampling 也会跟着偏。

代码 reward：从已有 ground-truth solution 的题生成新 test cases，再用测试验证候选程序。新测试可补覆盖，但生成器也可能造错 oracle，仍需去重、执行隔离和抽查。

数学 reward：课件写约 **800k samples** 训练 Chain-of-Thought reward model 做答案等价检查；嵌入报告的 manual spot-check 把 Classic RM 与 CoT RM 的正确率分别报告为约 **84.4%** 与 **98.5%**。这里没有展示抽查样本数、置信区间或完整分布，所以只能当团队报告的 spot-check 快照，不能解释为“所有数学答案 98.5% 可靠”。

### 16.6 RL infrastructure 与对照实验能/不能推出什么

【课程内容，PDF p.45–48】RL infra（infrastructure，训练基础设施）把 rollout workers、trainer workers、reward models、master 和 replay buffer 接起来；权重从 trainer 发到 rollout，轨迹反向送回训练。partial rollout 把超长未完成轨迹存到 buffer，下一轮续写，以减少长 CoT 独占 worker。

p.47 的小模型数学曲线展示训练步、accuracy 与 token length 同时变化；它支持“该 run 中存在共同变化”，不单独证明变长导致变强。p.48 比较：

- **expert iteration / ReST**：从当前模型采样，只保留验证正确的正样本，再做 maximum-likelihood/SFT；没有针对错误 response 的负梯度；
- **RL 方法**：高于 baseline 的回答正向、低于 baseline 的回答负向，显式利用失败样本。

该 Kimi 小模型、多数学任务图中，橙色 RL 曲线总体高于蓝色 ReST。可推出“负梯度在这个设置的 sample efficiency 更好”；不能推出 expert iteration 在任何任务都无用，也不能把多条相关曲线当普遍因果定律。

## 17. Kimi 的正则化 PG、平方 surrogate 与长度奖励

### 17.1 KL 正则化目标

【课程内容，PDF p.42】Kimi 报告写一类目标：

```math
\max_\theta\ 
\mathbb E_{(x,y^*)\sim D,\,(y,z)\sim\pi_\theta}
\left[
r(x,y,y^*)-\tau\,\mathrm{KL}
(\pi_\theta(\cdot|x)\|\pi_{\theta_i}(\cdot|x))
\right].
```

- $`x`$：题目；$`y^*`$：参考答案；
- $`y,z`$：模型输出答案与推理轨迹；
- $`r`$：可验证 reward；
- $`\pi_{\theta_i}`$：本轮开始时冻结的策略快照；在 Kimi 这套特例里，它同时充当“产生本批样本的 behavior/old policy”和“KL 拉回的 reference policy”。一般 PPO 中 old 与长期 reference 不必是同一个模型，见 §10.1；
- $`\tau`$：偏离惩罚强度。

$`\tau=0`$ 时没有这项拉回力；$`\tau`$ 太大时策略可能几乎不学新行为。

### 17.2 从最优分布关系到平方 surrogate

课件给出最优策略关系：

```math
r(x,y,y^*)-\tau\log Z(x)
=\tau\log\frac{\pi^*(y,z|x)}{\pi_{\theta_i}(y,z|x)},
```

这条闭式关系先用了 **nonparametric assumption（非参数化假设）**：对每个 prompt，$`\pi^*`$ 可以在“所有可能的输出概率分布”中自由选择，而不是受某个有限神经网络的共享参数和容量限制。现实中的有限 neural network 未必能精确表达每个 prompt 的闭式 $`\pi^*`$；后面的 surrogate 只是用同一组参数去拟合这份理想关系。

这里先把层级固定：

- $`D`$：完整 prompt/参考答案分布；
- $`x`$：当前同一道 prompt；$`y^*`$：参考答案；
- $`k`$：对这个 $`x`$ 采的 responses 数；
- 第 $`j`$ 条由答案 $`y_j`$ 与 reasoning trace $`z_j`$ 组成；
- $`r_j=r(x,y_j,y^*)`$；
- $`\bar r=k^{-1}\sum_{j=1}^k r_j`$；
- $`\pi_{\theta_i}`$：产生本批 samples 的 behavior/old 策略，同时也是这套 Kimi 特例中的 KL reference；它在本次更新中冻结；
- $`\pi_\theta`$：正在更新的策略。

$`Z(x)`$ 是只依赖 prompt 的归一化常数：

```math
Z(x)=\mathbb E_{(y,z)\sim\pi_{\theta_i}}
\left[e^{r(x,y,y^*)/\tau}\right].
```

若只有本 prompt 的 $`k`$ 个 samples，一个直接 Monte Carlo 近似是 **log-mean-exp**：

```math
\tau\log Z(x)
\approx\tau\log\left[
\frac1k\sum_{j=1}^{k}e^{r_j/\tau}
\right].
```

log-mean-exp 不是普通均值；高 reward 会被指数放大。Kimi 报告说明，当 $`\tau`$ 较大时，这个量趋近 reward mean，因此实践推导用 $`\bar r`$ 近似 $`\tau\log Z`$。有限 $`\tau`$、小 $`k`$、重尾 reward 时两者可明显不同，不能把近似写成恒等式。

训练时未知 $`\pi^*`$。Kimi 来源给出的有限样本平方目标先是：

```math
L_{sq}^{\mathrm{source}}(\theta)
=\frac1k\sum_{j=1}^{k}
\left[
r_j-\tau\log Z(x)
-\tau\log\frac{\pi_\theta(y_j,z_j|x)}
{\pi_{\theta_i}(y_j,z_j|x)}
\right]^2.
```

为让后面的梯度系数更整洁，本文重新标记一个**正数缩放版本**：

```math
\widetilde L_{sq}
=\frac{L_{sq}^{\mathrm{source}}}{2\tau}
=\frac1{2\tau k}\sum_{j=1}^{k}[\cdots]^2.
```

因为 $`\tau>0`$，除以 $`2\tau`$ 不改变极小点；但会改变 loss 的数值和梯度整体尺度，所以不能把 $`\widetilde L_{sq}`$ 冒充来源原式。它们都是要**最小化**的平方 surrogate（surrogate=更容易优化、用来近似原目标的替代目标）。

实践近似把 $`\tau\log Z(x)`$ 换成 $`\bar r`$。注意：平方括号里 reward 差与 log-ratio 正则在**同一项**，外面是对同一 prompt 的 $`k`$ 条回答求和。

### 17.3 对平方 loss 求梯度，得到正则化 policy-gradient

令：

```math
\ell_j(\theta)=\log\frac{\pi_\theta(y_j,z_j|x)}
{\pi_{\theta_i}(y_j,z_j|x)}.
```

因为 $`\pi_{\theta_i}`$ 冻结，$`\nabla_\theta\ell_j=\nabla_\theta\log\pi_\theta(y_j,z_j|x)`$。这里严格沿用 §4.4：sample、$`r_j`$、$`\bar r`$ 和旧策略 log-prob 全部 stop-gradient，只有 current log-prob 对 $`\theta`$ 求导。把 $`\tau\log Z`$ 用 $`\bar r`$ 近似，对本文的 $`\widetilde L_{sq}`$ 做 gradient descent，等价于沿下面方向做 ascent：

```math
\begin{aligned}
-\nabla_\theta \widetilde L_{sq}
&\approx\frac1k\sum_{j=1}^{k}
\left[(r_j-\bar r)-\tau\ell_j\right]
\nabla_\theta\log\pi_\theta(y_j,z_j|x)\\
&=\frac1k\sum_{j=1}^{k}
\left[
(r_j-\bar r)\nabla_\theta\log\pi_\theta(y_j,z_j|x)
-\frac\tau2\nabla_\theta\ell_j^2
\right].
\end{aligned}
```

第一项鼓励高于均值的回答；第二项惩罚 log-ratio 绝对值过大。它不是 DPO 的成对 chosen/rejected loss；“DPO-style”只说明使用了相似的 KL 正则最优关系。

### 17.4 一个 $`k=2`$ 完整数例

同一 prompt 采两条，$`r_1=1,r_2=0,\tau=0.5`$，current/old log-ratios $`\ell_1=0.2,\ell_2=-0.2`$。

先算精确样本 log-mean-exp 近似：

```math
\tau\log Z\approx0.5\log\frac{e^{1/0.5}+e^{0/0.5}}2
=0.5\log\frac{e^2+1}{2}
\approx0.5\log4.1945\approx0.7169.
```

精确桥的两个括号：

```math
a_1=1-0.7169-0.5(0.2)=0.1831,
```

```math
a_2=0-0.7169-0.5(-0.2)=-0.6169.
```

本文缩放后的平方 loss：

```math
\widetilde L_{sq}=\frac{0.1831^2+(-0.6169)^2}{2\times0.5\times2}
\approx\frac{0.0335+0.3806}{2}=0.2071.
```

这里 $`2\tau=2\times0.5=1`$，所以 $`\widetilde L_{sq}=L_{sq}^{\mathrm{source}}`$ **恰好数值相同**；换一个 $`\tau`$ 就不再相同。

实践均值近似是 $`\bar r=(1+0)/2=0.5`$，于是 ascent 系数：

```math
c_1=(1-0.5)-0.5(0.2)=0.4,
\qquad
c_2=(0-0.5)-0.5(-0.2)=-0.4.
```

所以第一条概率被推高，第二条被压低；同时平方 log-ratio 把两条都约束在旧策略附近。均值近似下 $`\widetilde L_{sq}=[0.4^2+(-0.4)^2]/2=0.16`$。它与 0.2071 不同，正好展示 $`\bar r\approx\tau\log Z`$ 不是精确等式。

### 17.5 长度奖励 $`\lambda`$

PDF p.43 的训练配方不是一开始就施加长度压力：团队先做正常训练，等策略经过 warm-up（暖启动，先获得基础解题能力）后，才在后期启用固定 length penalty。否则早期模型还不会解题时就强迫变短，可能压住探索和错误后的恢复。这是经验配方选择，不是下面公式数学上必然要求的时序。

启用后，PDF p.43 写：

```math
\lambda_i=0.5-\frac{\mathrm{len}(i)-\min\_len}
{\max\_len-\min\_len}.
```

若组内长度 $`[100,200,300]`$，最短 100、最长 300：

- 100：$`0.5-(100-100)/200=0.5`$；
- 200：$`0.5-100/200=0`$；
- 300：$`0.5-200/200=-0.5`$。

正确回答得到 $`\lambda_i`$；错误回答得到 $`\min(0,\lambda_i)`$。所以错误且短的 $`\lambda=0.5`$ 会被截成 0，不会拿正长度奖励；错误且长的拿 $`-0.5`$。

若所有回答等长，$`\max len=\min len`$，原分母为 0。**Kimi 报告给出的 fallback 是：所有 length reward 都设为 0。** 其他代码可以选择跳过，但必须另标“其他实现”，不能冒充论文口径。

## 18. On-policy rollout 与系统瓶颈

### 18.1 什么叫 on-policy

**on-policy（同策略）** 表示训练数据由当前/很近的策略生成。采样时 current 复制成 old，初始 ratio 为 1；更新后 current 改变。

若同一 rollout 反复用很多轮，或由老很多版本的模型生成，就更 off-policy。PPO ratio/clip 能缓和一定差异，不是无限期复用许可证。

### 18.2 长 CoT 的 straggler 手算

**straggler（拖尾任务）** 是比同组其他任务慢很多、让大家等待的任务。4 个 rollout 长度：

```math
[100,120,110,1000].
```

若同步批处理都 pad（填充）到 1000 token：

- 实际有用 token：$`100+120+110+1000=1330`$；
- 分配 token slots：$`4\times1000=4000`$；
- 利用率：$`1330/4000=0.3325=33.25\%`$；
- 浪费：$`4000-1330=2670`$ slots。

长度分桶、异步调度、partial rollout 可减少等待，却会改变 batch 组成、on-policy 新鲜度或梯度统计，需要共同审计。

### 18.3 train/inference 切换

【课程内容，PDF p.45–46】训练框架擅长反向传播、参数分片；推理引擎擅长 KV cache、连续批处理。RL 循环要：

~~~text
训练参数 → 推理引擎 → rollout
rollout/logprobs/rewards → 训练引擎 → 更新
~~~

若每轮传 100 GB 权重、链路 20 GB/s，光单向理想传输下界：

```math
100/20=5\text{ 秒}.
```

还未含同步、格式转换、网络竞争。Kimi 的混合 Megatron/vLLM 设计是其工程案例，不是唯一实现。

## 19. Qwen3：少量冷启动、thinking fusion 与 test-time budget

### 19.1 四阶段教学图

【课程内容，PDF p.49–54】Qwen3 的简化流程：

1. long-CoT cold-start SFT；
2. reasoning RL；
3. thinking/non-thinking mode fusion；
4. general RL，再做强到弱蒸馏构建小模型。

PDF p.51 给出 **3995 examples**，不是“正好 4000”。字幕口头说约 4000，是合理近似；本文计算用精确课件数。

若每 prompt 采样 8 responses：

```math
3995\times8=31{,}960\text{ responses}.
```

若平均每条 2048 token：

```math
31{,}960\times2048=65{,}454{,}080\text{ generated tokens}.
```

这是教学预算例，不是报告公开的真实总 rollout token 数。

### 19.2 thinking/non-thinking fusion

通过特殊标签/控制，让一个模型学习两种模式：

- thinking：允许较长内部推理；
- non-thinking：较快、较短回答；
- early termination：达到 budget 或特殊终止条件时结束。

同模型并不表示两模式行为完全相同；prompt template、解码设置和 budget 都是评测协议的一部分。

### 19.3 test-time scaling

PDF p.53 展示若干任务在 1k、2k、4k、8k、16k、32k thinking budget 下的曲线。在图示任务/模型里总体上升，但不能改写成“每个问题 token 越多越准”：

- 简单题可能浪费时延；
- 错误路线可能越写越长；
- verifier/答案提取可能受长度影响；
- 成本随生成 token 增加。

### 19.4 general RL 的 trade-off

PDF p.54 的表显示一般能力/偏好训练可提升部分通用任务，同时部分 math/code 指标下降。这是 **multi-objective trade-off（多目标权衡）**：帮助性、安全、风格、推理、成本不一定同向。不能只看总平均。

## 20. Qwen3-Coder-Next：midtraining、专家蒸馏与 agent RL

### 20.1 课程时点快照

【课程内容，PDF p.55–60】Qwen3-Coder-Next 是 2026 课程的最新案例。官方技术报告是动态材料；本文只按课件和 2026-02 的一手报告表述，不将榜单数字当永久事实。

### 20.2 midtraining 与 600B repository tokens

**midtraining（中期训练）** 位于通用预训练与 post-training 之间，强化长上下文、代码仓库、agent 交互等分布。课件写 600B repository-level tokens；$`B`$ 在这里是 billion（十亿）token，不是 bytes。

仓库级数据保留跨文件依赖、测试和提交结构；它不等于把孤立代码文件拼起来。

### 20.3 expert distillation

课件展示四类专家模型/技能蒸馏到一个基础模型。课堂问答提醒：各专家可由不同团队并行训练，之后把数据或行为汇入统一训练。不能由图推断所有权重直接平均，或每个专家恰好独立承担一个固定路由槽。

### 20.4 自动构造环境与 agent RL

PDF p.58–59：

- 从仓库/任务构造可执行环境；
- 约 800k tasks 是课程/报告快照；
- 模型使用工具编辑、运行测试，reward 来自环境结果。

agent reward 验证的是“在给定容器、测试、工具版本和时间限制下成功”。测试弱时可能把错误补丁判成功。

### 20.5 reward hacking 防守例

PDF p.60 讲到模型利用 Git 历史/remote 找答案。这是 verifier/环境泄漏，不是我们要教的攻击技巧。防守清单：

1. 移除不应可见的答案与历史；
2. 使用独立隐藏测试；
3. 记录完整工具轨迹；
4. 轮换/私有验证任务；
5. 人工审计高 reward 异常轨迹。

Lean 等形式系统也只验证形式化规格；规格或允许的环境若泄漏，仍可能被钻空子。

<a id="l16-systems"></a>

## 21. 一套可执行的 RLVR 决策树

1. **先写目标**：要提升数学结果、代码功能、工具成功率还是风格？不可揉成“智能”一个数。
2. **写 verifier 契约**：accept 什么、reject 什么、漏什么；造至少 20 个反例。
3. **定层级**：reward 是 per-token、per-response 还是 per-trajectory？平均分母是什么？
4. **先做小批手算**：输出 reward、mean/std、advantage、ratio、clip、KL、最终 loss。
5. **查估计偏差**：baseline 是否依赖当前 action？std 是否随机含自身？长度如何加权？
6. **查新鲜度**：rollout 由哪个 checkpoint 生成？训练几轮？ratio/clip fraction 多大？
7. **查系统**：最长/中位长度、padding 利用率、推理—训练切换、最慢 worker。
8. **查结果**：训练 reward、独立 verifier、人工审计、分布外 benchmark、长度和格式分别画图。
9. **做消融**：去掉 length reward、std、KL、format reward，判断增益来源。
10. **小规模失败后再放大**：大算力不会修复错误 reward 或错误分母。

## 22. 常见误区：错误说法 → 为什么错 → 正确说法

1. **“可验证=真实正确。”** 验证器只检查规格；正确说法：列出漏检面。
2. **“RLVR 不会 reward hack。”** 测试/解析器也可有漏洞；正确说法：使用隐藏验证与轨迹审计。
3. **“GRPO advantage 是 value。”** 它只看当前 group；正确说法：它是组相对分数。
4. **“同一 reward 的 advantage 固定。”** 组换了均值/std 就换；正确说法：写出 group。
5. **“std 总是除 $`G-1`$。”** NumPy 默认除 $`G`$；正确说法：查实现。
6. **“epsilon 修复无偏性。”** 它只防除零；正确说法：统计依赖仍在。
7. **“baseline 可以看当前 action。”** 会改变期望；正确说法：prompt-dependent、action-independent。
8. **“含自身均值完全没问题。”** 有 $`(G-1)/G`$ 缩放；正确说法：校正或 leave-one-out。
9. **“除 group std 是合法 baseline。”** 分母也依赖 action；正确说法：另做偏差分析。
10. **“PPO ratio 是 current/reference。”** 分母是 old；reference 用于 KL。
11. **“ratio 超界就一定取 clip。”** advantage 符号影响 min；正确说法：算两项。
12. **“KL 每个 token 都非负。”** log-ratio 样本可负；正确说法：KL 期望非负，特定 estimator 可逐样本非负。
13. **“KL sum=KL mean。”** 长度不同权重不同；正确说法：标分母。
14. **“目标和 loss 符号一样。”** 最大化 $`J`$ 等价最小化 $`-J`$。
15. **“每回答平均很公平。”** 它会改变每 token 权重并引入 length bias。
16. **“Dr.GRPO 是唯一固定代码。”** 实现细节不同；正确说法：核对公式/mask/denominator。
17. **“CoT 变长证明推理变强。”** 相关不等于因果；还可能是长度偏差。
18. **“aha 一定由 RL 首创。”** base model 已可能出现；正确说法：做基线比较。
19. **“R1-Zero 就是生产 R1。”** 生产 R1 是多阶段流程。
20. **“R1 数据全部公开。”** 关键细节不完整；正确说法：标未知。
21. **“R1 的 PRM/MCTS 失败证明方法无用。”** 只支持该实验设置。
22. **“distillation 复制 teacher 权重。”** 它训练 student 模仿数据/分布。
23. **“Kimi best-of-8 与十次估难完全相同。”** 课程 bullet 与报告截图口径不同。
24. **“Kimi 错误短答案拿 +0.5。”** 规则是 $`\min(0,\lambda)`$，所以拿 0。
25. **“所有回答等长仍可直接算 $`\lambda`$。”** 分母为 0，必须定义 fallback。
26. **“3995=4000 精确值。”** 4000 是口头近似。
27. **“thinking budget 越大必然越好。”** 图只支持特定任务/模型区间。
28. **“general RL 所有指标同升。”** 多目标会 trade off。
29. **“on-policy rollout 可无限复用。”** current 漂移后数据变 stale。
30. **“padding 只浪费显存不浪费算力。”** 很多 kernel 仍处理 token slots。
31. **“推理快就代表 RL 训练快。”** 权重同步、反向和 straggler 也可能主导。
32. **“agent 测试通过=补丁完全正确。”** 测试可能不完整或环境泄漏。
33. **“形式验证器不可被钻。”** 它只保证相对形式规格成立。
34. **“benchmark 上升=产品一定更好。”** 还缺真实流量、成本、安全和分布外评估。
35. **“全组 reward 相等时，epsilon 会造出一点学习信号。”** 分子全为 0，reward advantage 仍全为 0；只有未跳组时另加的 reference-KL 项可能更新。
36. **“Dr.GRPO 就是 leave-one-out。”** 它仍减含自身的组均值；在 iid 简化条件下只与 LOO 相差固定 $`(G-1)/G`$，但逐样本公式并不相同。
37. **“clamp 正 log-ratio 与 $`e^d-d-1`$ 是同一个 KL estimator。”** 前者是课程代码的单边 reward-shaping heuristic；后者只有在 current-sampling、精确 log-prob 且 current support 覆盖 reference 概率质量等条件下，才无偏估计 $`D_{KL}(current\|reference)`$。
38. **“Kimi 用 $`\bar r`$ 是精确算出了 $`\tau\log Z`$。”** 它是实践近似；有限 $`k`$ 或较小 $`\tau`$ 时 log-mean-exp 可与普通均值明显不同。

## 23. 公式卡与聚合审计表

| 目的 | 公式 | 最先检查 |
|---|---|---|
| group mean | $`\bar r=G^{-1}\sum_i r_i`$ | 同一 prompt 吗 |
| population std | $`\sqrt{G^{-1}\sum_i(r_i-\bar r)^2}`$ | 分母 $`G`$ |
| sample std | $`\sqrt{(G-1)^{-1}\sum_i(r_i-\bar r)^2}`$ | 与代码一致吗 |
| GRPO advantage | $`(r_i-\bar r)/(\sigma+\epsilon)`$ | 零方差、含自身 |
| PPO ratio | $`\pi_\theta/\pi_{old}`$ | old 版本 |
| PPO token term | $`\min(\rho A,\mathrm{clip}(\rho)A)`$ | $`A`$ 正负 |
| sampled log-ratio | $`g=\log\pi_\theta-\log\pi_{ref}`$ | 单样本可正可负 |
| p.14 one-sided shaping | $`\max(g,0)`$ | heuristic，不是完整KL |
| current\|\|reference KL estimate | $`e^d-d-1`$ | $`d=-g`$，current采样、精确log-prob、support覆盖 |
| leave-one-out | $`b_{-i}=(G-1)^{-1}\sum_{j\ne i}r_j`$ | 不含当前样本 |
| Dr.GRPO | $`G^{-1}\sum_i\sum_t\min(\rho A_i,\mathrm{clip}(\rho)A_i)`$ | 无每条长度除数、无std，仍含自身mean |
| Kimi logZ | $`\tau\log[k^{-1}\sum_j e^{r_j/\tau}]`$ | $`\bar r`$ 只是实践近似 |
| Kimi 来源平方式 | $`L_{sq}^{source}=k^{-1}\sum_j a_j^2`$ | 来源有限样本原式 |
| 本文梯度缩放式 | $`\widetilde L_{sq}=L_{sq}^{source}/(2\tau)`$ | 正数缩放；极小点相同，数值/梯度尺度不同 |
| Kimi length | $`0.5-(l_i-l_{min})/(l_{max}-l_{min})`$ | 等长除零 |

聚合审计顺序：

~~~text
token term
→ token sum or token mean?
→ response mean?
→ group mean?
→ prompt/batch mean?
→ maximize objective or minimize loss?
~~~

<a id="l16-questions"></a>

## 24. 自测题（80 题）

先自己写，再看 §25。标【手算】的必须写中间步骤；【判断解释】必须先写“对/错”再说明条件。

1. 【分类】prompt、response、token、group、batch 各是哪一层？
2. 【手算】一个 batch 有 3 prompts，每题 8 responses，每条平均 100 token；共有多少 responses 和生成 token？
3. 【填表+判断解释】固定 actual positive=真实正确、predicted positive=verifier pass，写 TP/FP/FN/TN 四格；verifier pass 是否等于现实目标完全满足？
4. 【设计】数学答案 verifier 至少列出两个漏检面。
5. 【手算】正确奖 1、格式奖 0.1；正确但格式错、错误但格式对各得多少？
6. 【判断解释】reward 必须可微，policy gradient 才能训练吗？
7. 【手算】两动作概率 $`[0.25,0.75]`$，reward $`[1,0]`$，期望 reward 是多少？
8. 【手算】Q7 中第一动作概率升到 0.30 后，期望 reward 增多少？
9. 【解释】high variance 用人话是什么意思？
10. 【判断解释】baseline 可以依赖 prompt 吗？可以依赖当前 sampled response 吗？
11. 【手算】reward 1、baseline 0.8，advantage 是多少？baseline 0.1 时呢？
12. 【定义】TRPO、PPO 的全称与共同直觉是什么？
13. 【手算】old probability 0.2、current 0.24，ratio 是多少？
14. 【手算】$`\epsilon=0.2`$，把 1.4、0.6、1.1 分别 clip。
15. 【手算】$`A=2,\rho=1.4`$，算 PPO 两项和 min。
16. 【手算】$`A=-2,\rho=0.6`$，算 PPO 两项和 min。
17. 【错误诊断】“ratio 只要超界，一律使用 clipped ratio。”哪里错？
18. 【分类】current、old、reference、value、reward 五角色分别做什么？
19. 【判断解释】old 与 reference 是否可能数值相同？职责是否相同？
20. 【解释】为什么语言模型 PPO 比二动作玩具复杂？
21. 【定义】GAE 是什么？本讲需要掌握到什么边界？
22. 【填表】GRPO 对一个 prompt 的五个步骤是什么？PDF p.18 的 response-level 总览式与 p.23 的 token 实现式有什么不同？
23. 【判断解释】GRPO group advantage 是 value function 吗？
24. 【手算】rewards $`[0,1,1,2]`$ 的均值。
25. 【手算】Q24 的离均差、平方和、population variance。
26. 【手算】Q24 的 population std 与四个 advantages。
27. 【手算】Q24 的 sample variance/std 与首尾 advantages。
28. 【判断解释】Q26 和 Q27 哪套一定正确？
29. 【手算+判断解释】rewards $`[1,1,1,1]`$、epsilon $`10^{-4}`$，advantages 是什么？完整更新是否一定为0？
30. 【错误诊断】零方差时直接算 $`0/0`$ 会怎样？
31. 【手算】长度 $`[2,1,2,1]`$ 共有多少 token？
32. 【手算】$`A=-1.4142,\rho=0.7,\epsilon=0.2`$，算两项和 min。
33. 【手算】$`A=1.4142,\rho=1.3`$，算两项和 min。
34. 【手算】§9 六 token 表的 response 1 token mean。
35. 【手算】§9 四 response mean 再做 group mean。
36. 【手算】§9 若改 global token mean，结果是多少？
37. 【判断解释】Q35 与 Q36 符号不同是否说明有一个算错？
38. 【手算】刚采样完、current=old 时 ratio 是多少？clip 是否起作用？
39. 【解释】为何做多个 minibatch epochs 后 clip 才更常起作用？
40. 【手算】对 $`k_3`$ estimator，$`d=\log2`$，算 $`e^d-d-1`$。
41. 【手算】Q40 乘 $`\beta=0.05`$ 的 KL penalty。
42. 【手算】KL estimates $`[0.1,0.2,0.3]`$ 的 sum 和 mean。
43. 【判断解释】单 token log-ratio 必须非负吗？理论 KL 呢？$`k_3`$ 无偏等式还需要什么 support 条件？
44. 【推导】写出 state-dependent、action-independent baseline 的零期望四行证明。
45. 【手算+判断解释】只研究 IID（independent and identically distributed，独立同分布）rollouts 的 mean-subtraction、忽略 std/clip/KL/长度时，$`G=4`$ 会把 reward-gradient 缩放多少？“独立”和“同分布”各是什么意思？校正因子多少？为什么不能据此说完整 GRPO 只差这个常数？
46. 【推导】把 $`r_i-\bar r`$ 拆成含 $`r_i`$ 与不含 $`r_i`$ 两部分。
47. 【手算】对 $`[0,1,1,2]`$ 算四个 leave-one-out baselines/advantages。
48. 【判断解释】为什么 group std 不能自动套合法 baseline 证明？
49. 【判断解释】epsilon 能修复 group std 的统计偏差吗？
50. 【手算】同 $`A=-1`$，长度 2 与 10，在每回答 token mean 下每 token 系数各多少？
51. 【手算】若统一除 10，Q50 两回答总系数各多少？
52. 【判断解释】response 变长能否单独证明推理变强？
53. 【分类】R1-Zero 的 accuracy reward 与 format reward 分别验证什么？
54. 【错误诊断】最终答案正确为何仍可能有错误推理？
55. 【判断解释】R1-Zero 与生产 R1 是否同一 pipeline？
56. 【填表】写出生产 R1 的完整链，并标出哪一环是“收集/过滤数据”而不是一次 optimizer 训练阶段。
57. 【手算】accuracy 1、consistency 0.8、权重各 1，总 reward；consistency 1 时呢？
58. 【判断解释】R1 中 PRM/MCTS 未成功能否证明它们永远无用？
59. 【解释】distillation 与复制 teacher 权重有什么不同？
60. 【手算】独立成功率 $`p=0.1`$，best-of-8 全失败概率和至少一次成功概率。
61. 【判断解释】Kimi 的 best-of-8 与报告截图十次估难应如何记录？
62. 【手算】Kimi $`k=2`$ 例：$`r=[1,0],\tau=.5,\ell=[.2,-.2]`$。算样本 log-mean-exp 的 $`\tau\log Z`$、两个精确括号与本文缩放式 $`\widetilde L_{sq}`$；为什么此处它与来源 $`L_{sq}^{source}`$ 数值巧合相同？
63. 【手算+解释】Q62 改用 $`\bar r`$ 近似，算两个 ascent 系数与 $`\widetilde L_{sq}`$；为何和Q62不同？
64. 【手算】长度 $`[100,200,300]`$ 的三个 $`\lambda`$。
65. 【填表】Q64 中正确/错误回答各拿怎样的 length reward？
66. 【错误诊断】全组等长时直接使用长度公式有什么问题？Kimi报告的 fallback 是什么？
67. 【判断解释】on-policy 是否意味着同一 rollout 可以无限复用？
68. 【手算】长度 $`[100,120,110,1000]`$ 的有用 token、pad slots、利用率、浪费。
69. 【手算】每轮传 100 GB 权重、链路 20 GB/s，单向理想下界多少秒？
70. 【手算】3995 prompts、每题 8 responses，共多少 responses？
71. 【手算】Q70 每条 2048 token，共多少 generated tokens？
72. 【判断解释】3995 与“约 4000”是否冲突？
73. 【判断解释】thinking budget 从 1k 加到 32k 是否保证每题更准？
74. 【解释】general RL 为什么可能让部分 math/code 指标下降？
75. 【单位】600B repository tokens 中 B 是 bytes 还是 billion？
76. 【设计】agent RL 环境防 reward hacking 至少列四项。
77. 【判断解释】通过单元测试是否证明补丁完全正确？
78. 【设计】上线 RLVR 前的最小消融实验列四个。
79. 【错误诊断】训练 reward 上升、回答变长，所以推理因长度而提升。至少指出两处逻辑问题。
80. 【综合设计】给代码修复 RLVR 写一张六行 reward contract：目标、输入、通过条件、漏检面、聚合层级、独立复核。

<a id="l16-answers"></a>

## 25. 自测答案（1–80）

### 25.1 第 1–20 题

1. prompt 是题；response 是完整答案；token 是答案中的动作单位；group 是同一 prompt 的多回答；batch 是多 prompts/groups 的更新集合。
2. responses $`=3\times8=24`$；tokens $`=24\times100=2400`$。
3. 四格：真实对且通过=TP；真实错却通过=FP；真实对却拒绝=FN；真实错且拒绝=TN。**pass 不等于完整现实目标满足。** 例如 parser 可误抓答案形成 FP；也可能因格式拒绝等价正确答案形成 FN。
4. 例：只比末答案，漏错误中间推理；答案解析器可能接受格式漏洞。还可漏单位、证明完整性。
5. 正确格式错：$`1+0=1`$；错误格式对：$`0+0.1=0.1`$。
6. **不必。** reward 可是离散测试结果；梯度通过 $`\nabla\log\pi_\theta`$ 回到 policy。
7. $`0.25\times1+0.75\times0=0.25`$。
8. 新期望 $`0.30`$；增加 $`0.30-0.25=0.05`$。
9. 同一真实目标重复采样，梯度估计忽高忽低，需要更多样本才能看清方向。
10. 可依赖 prompt/state；不能依赖当前 sampled response/action，否则减项期望一般不为 0。
11. $`1-0.8=0.2`$；$`1-0.1=0.9`$。
12. TRPO：Trust Region Policy Optimization；PPO：Proximal Policy Optimization。共同直觉是一次别把策略推得太远。
13. $`\rho=0.24/0.20=1.2`$。
14. 区间 $`[0.8,1.2]`$：1.4→1.2；0.6→0.8；1.1→1.1。
15. 裸项 $`1.4\times2=2.8`$；clip 项 $`1.2\times2=2.4`$；min=2.4。
16. 裸项 $`0.6\times(-2)=-1.2`$；clip 项 $`0.8\times(-2)=-1.6`$；min=$`-1.6`$。
17. min 比较的是乘 advantage 后的两项；负 advantage 会翻转大小关系。
18. current 更新；old 提供本批 ratio 分母；reference 提供长期 KL 锚；value 预测回报；reward/verifier 打分。
19. 可能在初始时相同；职责不同，old 随批次更新，reference 通常长期冻结。
20. 每个 token 是动作、reward 常到末尾才有；还需 rollout、value/GAE、reference KL、old logprobs、长序列显存与系统同步。

### 25.2 第 21–40 题

21. GAE 是 Generalized Advantage Estimation，把多步时序差分按衰减混合。本讲只需知道它给 PPO 构造 token advantage，GRPO 用组相对分数省去 value。
22. 同 prompt 采 $`G`$ 条→verifier 打分→组内标准化→把 response advantage 给各 token→clip+KL 更新。p.18 总览式直接写整条 response 概率比、每 response 一项；p.23 为分析实现偏差，把它展开成每 token ratio 的内层和，原 GRPO 再除本条 $`|o_i|`$。二者不能无说明混成同一层级公式。
23. **不是。** 它不预测未来，只比较本次同题 group。
24. $`(0+1+1+2)/4=1`$。
25. 离均差 $`[-1,0,0,1]`$；平方 $`[1,0,0,1]`$；和 2；population variance $`2/4=0.5`$。
26. std $`=\sqrt{0.5}=0.7071`$；advantages 约 $`[-1.4142,0,0,1.4142]`$。
27. sample variance $`2/3=0.6667`$；std $`0.8165`$；首尾 $`\mp1/0.8165=\mp1.2247`$。
28. 都可能是定义；课程 NumPy 实现用 population std。必须与代码的分母一致。
29. 均值 1，分子全 0；$`0/(0+10^{-4})=0`$，所以 reward advantages 全 0。若 skip group，总更新0；若不skip且保留 $`\beta>0`$ KL、current又偏离reference，KL项仍可能更新。
30. 得 NaN；后续 loss/gradient 可被污染。
31. $`2+1+2+1=6`$ token。
32. clip ratio=0.8；裸 $`0.7(-1.4142)=-0.9899`$；clip $`0.8(-1.4142)=-1.1314`$；min=-1.1314。
33. clip=1.2；裸 $`1.3(1.4142)=1.8385`$；clip $`1.2(1.4142)=1.6970`$；min=1.6970。
34. $`(-1.1314-1.5556)/2=-2.687/2=-1.3435`$。
35. $`(-1.3435+0+0+1.6970)/4=0.3535/4=0.088375`$。
36. token 和 $`-1.1314-1.5556+1.6970=-0.9900`$；除 6 得 $`-0.1650`$。
37. **否。** 二者优化的加权目标不同；先每 response 平均会让短长回答各占一票。
38. ratio=1；第一次该点不触发 clip。
39. current 更新后与 frozen old 不同，ratio 才离 1；后续 epoch 需要 clip 限幅。
40. $`e^{\log2}-\log2-1=2-0.6931-1=0.3069`$。

### 25.3 第 41–60 题

41. $`0.3069\times0.05=0.015345`$。
42. sum $`=0.1+0.2+0.3=0.6`$；mean $`=0.6/3=0.2`$。
43. 裸样本 $`g=\log\pi_\theta-\log\pi_{ref}`$ 可正可负；其 current-sampling 期望 KL 非负。p.14 $`\max(g,0)`$ 是单边 heuristic；$`e^d-d-1,d=-g`$ 逐样本非负。要使其无偏等于 $`D_{KL}(current\|reference)`$，还需 current 采样、精确 log-prob，并让 current support 覆盖 reference 全部概率质量；最好双方在共同动作集上概率都为正。否则 $`\sum\pi_\theta(\pi_{ref}/\pi_\theta)`$ 未必等于 1。
44. $`\mathbb E[b\nabla\log\pi]=b\sum\pi\nabla\log\pi=b\sum\nabla\pi=b\nabla\sum\pi=b\nabla1=0`$。条件：给定 prompt 后 $`b`$ 不看当前 action。
45. IID 中，独立表示每条 rollout 不查看其他 rollout 的结果；同分布表示都来自同一 policy 和采样配置。缩放 $`(4-1)/4=3/4=0.75`$；校正 $`4/3`$。这只适用于题干固定的 mean-subtraction 分析；随机 std 依赖当前 reward，clip 改变 surrogate，KL 与长度权重也加入其他项，因此完整 GRPO 不能用一个 $`4/3`$ 全部校正。
46. $`r_i-\bar r=\frac{G-1}{G}r_i-\frac1G\sum_{j\ne i}r_j`$。
47. baselines $`[4/3,1,1,2/3]`$；advantages $`[-4/3,0,0,4/3]`$。
48. std 包含 $`r_i`$，改变当前 action 会改分母；它不是 action-independent baseline。
49. **不能。** epsilon 只防除 0，不消除随机分母对 action 的依赖。
50. 长 2：每 token $`-1/2=-0.5`$；长 10：每 token $`-1/10=-0.1`$。
51. 都按每 token $`-1/10`$：短回答总 $`2(-0.1)=-0.2`$；长回答总 $`10(-0.1)=-1`$。
52. **不能。** 可能是优化长度偏差、base 模式被放大或任务需要更多字；须做长度控制与消融。
53. accuracy 检最终正确/测试；format 检标签/格式。都不自动验证全过程真实。
54. ORM 只看末答案；例如错误等式碰巧得到正确数字仍可得 1。
55. **不是。** R1-Zero 是纯 RL 演示；R1 有 cold-start SFT、RL、后续 SFT/RLHF。
56. V3-Base→cold-start SFT→reasoning RL→**收集/过滤 reasoning 与 non-reasoning 数据（数据步骤）** →SFT→一般 RLHF。链有六个节点，但只有带训练更新的节点才叫训练阶段；不能为凑“五阶段”偷偷删掉数据构造。
57. $`1+0.8=1.8`$；一致性 1 时 $`1+1=2`$。
58. **不能。** 只说明其具体模型、数据、预算和实现未达预期。
59. 蒸馏用 teacher 生成的数据/概率训练 student；不是复制权重，也不保证继承全部能力。
60. 全失败 $`0.9^8\approx0.4305`$；至少一次成功 $`1-0.4305=0.5695`$。

### 25.4 第 61–80 题

61. 明写“PDF/口述 best-of-8；同页报告截图十次高温采样估 pass rate”，作为口径差异保留。
62. $`\tau\log Z=.5\log[(e^2+1)/2]\approx.7169`$。括号 $`a_1=1-.7169-.5(.2)=.1831`$，$`a_2=0-.7169-.5(-.2)=-.6169`$。本文 $`\widetilde L_{sq}`$ 的分母 $`2\tau k=2`$，所以 $`\widetilde L_{sq}=(.1831^2+.6169^2)/2\approx(.0335+.3806)/2=.2071`$。来源式是 $`L_{sq}^{source}=k^{-1}\sum a_j^2`$；这里 $`2\tau=1`$，所以除以 $`2\tau`$ 没改数值，这是 $`\tau=.5`$ 的巧合。
63. $`\bar r=.5`$。ascent系数 $`c_1=(1-.5)-.5(.2)=.4`$，$`c_2=(0-.5)-.5(-.2)=-.4`$；$`\widetilde L_{sq}=(.4^2+(-.4)^2)/2=.16`$。不同是因为有限 $`\tau,k`$ 时 $`\bar r`$ 只是 $`\tau\log Z`$ 的实践近似，不是恒等式。
64. 分母 $`300-100=200`$：100→0.5；200→$`0.5-100/200=0`$；300→$`0.5-200/200=-0.5`$。
65. 正确拿 $`[0.5,0,-0.5]`$；错误拿 $`\min(0,\lambda)=[0,0,-0.5]`$。
66. $`\max len-\min len=0`$，直接算会除 0；Kimi 报告的 fallback 是把该组所有 length reward 设为 0。其他实现的 skip 只能另标。
67. **否。** 更新后 current 漂移，rollout 变 stale；要限制复用并监控 ratio。
68. 有用 $`=100+120+110+1000=1330`$；slots $`=4\times1000=4000`$；利用率 $`1330/4000=33.25\%`$；浪费 2670。
69. 理想下界 $`100/20=5`$ 秒；未含同步和竞争。
70. $`3995\times8=31{,}960`$ responses。
71. $`31{,}960\times2048=65{,}454{,}080`$ tokens。
72. 不冲突：3995 是课件精确值；4000 是口头近似。计算要声明所用值。
73. **不保证。** 课件图只支持特定任务/预算；长推理可浪费或走错。
74. 多目标梯度可能冲突；风格/安全/通用偏好优化会改变原 math/code 分布。
75. billion，十亿 token；不是 byte。
76. 例如移除答案历史、隐藏独立测试、记录工具轨迹、轮换私有任务、人工审计异常高分；任选四项。
77. **不能。** 测试覆盖可能弱，未检查性能、安全或隐藏输入。
78. 例：分别去掉 length reward、std normalization、KL、format reward；保持其他条件相同，并看独立评测和长度。
79. 两处问题：相关不等于长度导致能力；reward/verifier 可能偏好长或有漏洞；还需与 base/Dr.GRPO/长度匹配对照。
80. 示例：目标=修复真实 bug；输入=隔离仓库+issue；通过=隐藏功能/回归测试；漏检=安全/性能/未覆盖平台；聚合=每 trajectory 0/1、按任务平均；复核=新隐藏测试+人工 trace audit。六行必须都写。

<a id="l16-video-nav"></a>

## 26. 视频时间导航（人工英文字幕）

> 字幕轨：人工 `en-US`，共 1,687 cues，范围 00:05–75:42。下表 160 个秒点全部唯一；显示时间与 URL 的 `t=` 秒数一致。每行都给“这一段解决什么”的中文主题，并保留该秒或紧邻几秒的英文人工 cue 作为核对证据。英文证据中的省略号、合句与轻微清洗均标作“按字幕整理”，不是连续逐字引文；点击后仍应连听前后句。中文主题是对这个局部语境的概括，不是假装逐字翻译。

### 26.1 为什么要 RLVR；从 policy gradient 复习 PPO（00:05–13:29）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [00:05](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=5s) | 开场：本讲进入“从可验证奖励学习”（RLVR）。 | “This is the second of the post-training lectures … RLVR.” |
| [00:38](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=38s) | 把能解难数学题的 thinking model 与 RLVR 主线连起来。 | “That's exactly this thinking model stuff or RLVR stuff.” |
| [01:17](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=77s) | RLHF 的 reward overoptimization 限制了继续外推。 | “The RLHF wasn't really going to get us where we wanted to go because of … overoptimization.” |
| [01:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=103s) | 再强的正则也不能永久阻止 reward model 上过拟合。 | “No matter how good of a job you do at regularizing, eventually … overfitting.” |
| [02:06](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=126s) | 提问：RLHF 与围棋等更“原生 RL”任务差在哪。 | “What is the difference between what we're doing in RLHF and … RL natural domains?” |
| [02:37](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=157s) | 可验证数学/代码更像有明确终点的搜索问题。 | “In some sense, these are search problems.” |
| [03:01](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=181s) | 算法名字可能相近，但训练结果与行为会明显不同。 | “The algorithms aren't going to be that different fundamentally, but where we will end up … different.” |
| [03:26](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=206s) | 本讲前半路线：先复习 PPO，再理解 GRPO。 | “What is GRPO, how does PPO work.” |
| [03:58](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=238s) | 为什么还要再讲一遍 PPO：它足够容易混淆。 | “PPO is confusing enough that … you will benefit from doing it twice.” |
| [04:26](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=266s) | policy-gradient 中每条 log-prob 梯度会乘正或负权重。 | “The weights might be positive or negative.” |
| [04:47](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=287s) | 新 rollout 很贵，因此希望复用旧 rollout 做多个梯度步。 | “Every time we want to take a gradient step. Can we reuse our rollouts?” |
| [05:13](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=313s) | PPO 是许多强化学习任务中的常用工作马。 | “PPO … is a real workhorse of RL.” |
| [05:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=343s) | 用 OpenAI Five 说明 PPO 能处理高维动作与状态。 | “Their OpenAI bot, which they trained using PPO … high-dimensional action and state spaces.” |
| [06:16](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=376s) | 纸面 PPO 伪代码看起来短而容易实现。 | “If you look at the pseudocode of PPO … this is not that bad.” |
| [06:45](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=405s) | 转折：公式简单不等于工程实现稳健。 | “This is all relatively easy looking in practice … But then you see blog posts …” |
| [07:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=430s) | 不同库与实现细节会产生非常不同的结果。 | “Different libraries and implementations … give totally different numbers.” |
| [07:36](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=456s) | 语言模型 PPO 的生产实现尤其复杂。 | “The implementations of PPO … for language models are not particularly pleasant.” |
| [08:02](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=482s) | PPO 工程包含旧样本 buffer、value model 与 advantage estimation。 | “You've got an experience buffer … training this value model … advantage estimation.” |
| [08:28](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=508s) | PPO 有许多敏感、看似古怪但影响结果的实现细节。 | “A variety of really tricky implementation things …” |
| [08:56](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=536s) | 一个“可靠参考实现”也需要很长时间调通。 | “A nice robust reference implementation of PPO … took a long time to get working.” |
| [09:18](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=558s) | PPO 内层仍是 advantage、ratio 与 clipping 的标准更新。 | “The actual inner compute loss … advantages … clipping ratios.” |
| [09:44](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=584s) | 课程代码的单边 clamp 不是标准 KL 本身。 | “Need KL penalties … but actually this only works if you clip the KL off at 0 …” |
| [10:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=610s) | 高方差和多组件让 RL 实现很敏感。 | “Implementations for RL algorithms can be … sensitive … high variance in the gradient estimates.” |
| [10:35](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=635s) | 传统 PPO 会用 value function 与折扣回报估计每个位置的 advantage。 | “Gamma discounted reward … value function … every token generation step.” |
| [11:06](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=666s) | 训练曲线应同时监看任务 reward 与 KL 正则变化。 | “The rewards go up … and then the negative KL regularizers go down.” |
| [11:39](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=699s) | 大规模 PPO 可行，但从零实现很挑细节。 | “PPO can just be really finicky and complicated.” |
| [12:06](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=726s) | PPO 还会占用本可服务模型或推理的计算资源。 | “Be using for other stuff like models or inference servers.” |
| [12:33](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=753s) | DPO 的成对偏好结构不自然适配单题可验证奖励。 | “My math problems don't come in the form of inherently pairwise comparisons.” |
| [12:55](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=775s) | 讲者认为“DPO 离线、PPO 在线”的界线常被夸大，因为 DPO 也能反复迭代成在线流程。 | “Although I think this distinction is very overstated … it can be made online by just iterating DPO repeatedly.”（按字幕整理） |
| [13:29](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=809s) | 过渡：GRPO 是可验证任务上更简单的 RL 路线。 | “It is the simpler way to do RL for verifiable task.” |

### 26.2 GRPO 公式、代码、baseline、标准差与长度偏差（13:59–25:48）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [13:59](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=839s) | GRPO 保留 PPO 更新，但拿掉最麻烦的 value function。 | “PPO is a good idea, but we want to change … the value function.” |
| [14:21](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=861s) | 没有 value network 后，仍要构造低方差 advantage。 | “We get rid of the value network, but we still need an advantage.” |
| [14:40](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=880s) | 用同题其他 rollout 的表现代替 value prediction 作比较基线。 | “Normally … if you had a value function, you would compare it to your predicted value.” |
| [15:01](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=901s) | 同一 prompt 采样一组回答；高于组均值就是正 advantage。 | “Sample 10 other rollouts … If I'm doing better than my mean, then I have a high advantage.” |
| [15:24](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=924s) | DeepSeek 版目标含 PPO clipped term 与 reference KL。 | “This is the objective … min clipped advantage, plus a KL term … close to the reference.” |
| [15:54](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=954s) | 原始 GRPO 把组 reward 减均值，再除组标准差。 | “Take the reward … subtract the mean and … divide by the standard deviation.” |
| [16:18](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=978s) | 首次更新时 old/current ratio 为 1，clip 暂不起作用。 | “The ratio between pi theta old and pi theta is 1 … clipping … never does anything.” |
| [16:48](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1008s) | 先把 GRPO 数学看懂，再进入实现细节。 | “It's worth understanding in a little bit of detail before … nitty-gritty details.” |
| [17:22](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1042s) | 最小实现：同题 rollout (k) 次，z-score 后做 REINFORCE 梯度。 | “Roll out k times … z-score it and then just take a reinforced gradient.” |
| [17:51](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1071s) | 自动微分实现时必须在 advantage 处 stop-gradient。 | “In order to do this using autodiff, you will have to do a stop grad somewhere.” |
| [18:22](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1102s) | 代码给标准差加很小 epsilon，避免除零。 | “They add a little tiny … to the standard deviation calculation.” |
| [18:47](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1127s) | GRPO 流行的重要原因是短、容易实现与理解。 | “GRPO is … straightforward … easy to implement, easy to understand.” |
| [19:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1159s) | 课程展示 GRPO 相对 rejection fine-tuning 的实验结果。 | “GRPO … does much better than RFT, like rejection fine tuning.” |
| [19:44](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1184s) | 只验最终答案与给中间步骤过程奖励是不同设计。 | “Grading the intermediates give you some gains … one of the big design decisions.” |
| [20:11](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1211s) | 开始审计：GRPO 的更新真的是原目标的 policy gradient 吗。 | “Are we actually taking policy gradients?” |
| [20:36](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1236s) | 什么 baseline 才是不改变期望梯度的好 advantage。 | “Is this a good advantage function? What makes a good advantage?” |
| [21:06](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1266s) | state-dependent baseline 可减 reward 而不改期望梯度方向。 | “I can subtract any … state-dependent baseline.” |
| [21:31](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1291s) | baseline 选法改变方差，却不应改变期望下降方向。 | “Still going to descend in the same direction … lower or higher variance.” |
| [22:01](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1321s) | 追问组统计加入后，更新是否仍沿原 reward 的正确方向；讲者口头说“descend reward”，按本文最大化目标应读成 ascend reward，或等价地 descend negative loss。 | “If you really want … an algorithm that really does what's written on the tin, it actually descends the reward. GRPO does not do that.”（按字幕整理；符号口径见§11.4） |
| [22:29](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1349s) | GRPO 还会按 response 长度做 per-token 平均。 | “GRPO … does this almost per token … divide by the total length.” |
| [22:57](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1377s) | Dr.GRPO 去掉长度除法与标准差归一化。 | “You won't have the standard deviation normalization … don't do these two things …” |
| [23:28](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1408s) | 这些归一化让 GRPO 优化的量与原始 reward 不同。 | “It is actually doing something slightly different … not actually directly descending on the reward objective.” |
| [23:55](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1435s) | 长度平均会让错误回答“拖长再错”受到较小的每 token 惩罚。 | “When you're wrong … encourage the model to generate … long outputs.” |
| [24:21](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1461s) | 模型发现不会做后继续啰嗦，是 length bias 的直观后果。 | “If you divide by the output, you encourage the model to blab on …” |
| [24:51](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1491s) | 错误样本尤其不应因变长而稀释惩罚。 | “Especially on the incorrect cases, you really don't want … longer and longer outputs.” |
| [25:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1519s) | 过易或过难题组的 reward 方差接近零。 | “Problems are too easy or too hard … 0 variation.” |
| [25:48](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1548s) | 课程质疑：标准差归一反而可能忽略可学习难度范围。 | “We want our models to potentially learn on things that are within its solvability range.” |

### 26.3 DeepSeek-R1-Zero、R1 多阶段与蒸馏（26:22–39:52）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [26:22](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1582s) | 算法部分收束，讲者暂停回答 GRPO 问题。 | “If anyone has algorithmic questions, I'll take them before I move on.” |
| [26:44](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1604s) | R1 论文推动了开源 RLVR 模型浪潮。 | “A lovely paper [that] kicked off the wave of open source RLVR models.” |
| [27:17](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1637s) | GRPO 的研究价值之一：普通研究者也能复现与实验。 | “This kind of GRPO thing, which anyone can really play with …” |
| [27:46](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1666s) | R1 建立在 DeepSeekMath 已积累的数学 RL 经验上。 | “They had this DeepSeekMath paper, where they were doing GRPO.” |
| [28:12](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1692s) | outcome supervision 只看最终答案对错。 | “Outcome supervision … reward only for whether the final answer is correct or incorrect.” |
| [28:41](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1721s) | R1-Zero 从 base model 直接做较干净的 RL 实验。 | “R1 zero … they don't really do that much post training, so they have a base model.” |
| [29:11](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1751s) | format reward 强制模型把思维链放进指定标签，便于之后剥离。 | “Format rewards … strip out the chain of thought later … thinking tags.” |
| [29:35](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1775s) | 干净配方得到接近当时 O1 的课程时点结果。 | “Only a little bit worse than OpenAI O1 … really clean.” |
| [30:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1800s) | 论文报告长 CoT 与所谓“aha moment”，但只是观察。 | “Interesting phenomena they claim to have observed … longer and longer …” |
| [30:26](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1826s) | 讲者质疑“aha”是否真是新涌现的因果证据。 | “There's an aha moment … It is unclear … whether … particularly impressive.” |
| [30:58](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1858s) | “aha”语言已在预训练数据中，不能归因于 RL 独自创造。 | “It can't just be a result of the RL algorithm … learned it during pre-training.” |
| [31:32](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1892s) | R1-Zero 是受控算法展示，R1 则把它产品化。 | “R1 zero … clean controlled setting. And then R1 was their attempt to productionize …” |
| [32:02](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1922s) | 产品配方要把 midtraining、reasoning RL、SFT/RLHF 串起来。 | “How do we compose all of the pieces to actually get a system …” |
| [32:27](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1947s) | production R1 加 language-consistency reward。 | “Added a language consistency reward for the chain of thought.” |
| [32:49](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1969s) | 一致语言奖励主要服务可读性，并混入非可验证奖励。 | “A single consistent language … interpretability … non-verifiable rewards.” |
| [33:26](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2006s) | 报告只说 small amount of long-CoT data，来源细节不可凭空补。 | “We construct and collect a small amount of long COT data …” |
| [33:50](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2030s) | long-CoT SFT 可先解锁能力，再作为 RL 起点。 | “SFT on long COT can unlock … capabilities … a great starting point for RL.” |
| [34:21](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2061s) | 好 base model 配少量长 CoT SFT 就可能得到大量能力。 | “With the right kinds of base model … long COT reasoning … just from SFT.” |
| [34:45](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2085s) | RL 的价值之一是自生成长 CoT；生成后也可被蒸馏模仿。 | “RL allows you to self-generate that … also learn from imitation.” |
| [35:14](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2114s) | 最终阶段回到 instruction SFT 与非可验证任务 RLHF。 | “Basic instruction tuning style SFT and then RLHF.” |
| [35:48](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2148s) | 课程时点结果：简单配方复现许多 test-time scaling 行为。 | “Matched a lot of the test time scaling behavior … came from a very simple recipe.” |
| [36:17](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2177s) | 只要有合适长 CoT，蒸馏模型也能接近专门 reasoning 模型。 | “In some cases, matching … specialized thinking models.” |
| [36:50](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2210s) | DeepSeek 报告的价值还在于公开 ablation 与失败路线。 | “They often do both ablations and … tell you failed things that didn't work.” |
| [37:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2239s) | outcome reward 易扩数据；process reward 需要逐步 rubric。 | “Outcome reward models are … good enough … where are you going to get these step-by-step rubrics.” |
| [37:50](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2270s) | R1 报告称其 MCTS 尝试没调好，不代表 MCTS 永远无用。 | “They tried a lot of MCTS … couldn't get it to work very well.” |
| [38:17](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2297s) | 课堂问答转向正负 reward 与长度变化的图。 | “You should explain the positive reward basis …” |
| [38:47](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2327s) | 正 reward 可能鼓励缩短 CoT：省推理成本但可能伤准确率。 | “Encourages you to shorten the COT … save on inference costs, bad if … accuracy.” |
| [39:16](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2356s) | R1 图聚合所有评测回答，不只正样本。 | “In the R1 diagram, this is all aggregated … not just the positive ones.” |
| [39:52](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2392s) | 受控比较阶段收束，转向 Kimi 的替代做法。 | “The story … in these sets of controlled comparisons … clean and clear … alternative method.” |

### 26.4 Kimi k1.5：数据、目标、长度和系统（40:29–55:29）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [40:29](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2429s) | Kimi 与 DeepSeek 路线不同，比较两者能揭示哪些选择不是唯一的。 | “Kimi … do some sets of things quite differently than DeepSeek.” |
| [40:54](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2454s) | Kimi 报告更细讲 RL 数据构造与 curriculum。 | “More detail in talking about the data set construction and curriculum generation.” |
| [41:23](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2483s) | RL 仍受数据难度课程影响，不是换成 GRPO 就自动成功。 | “The data remains very important … additional wrinkle of curriculum.” |
| [41:53](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2513s) | RL 与 SFT 对训练样本质量的失败方式不同。 | “In SFT, there's … subtlety … sometimes you don't want to train on some of this data.” |
| [42:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2539s) | Kimi 先收集广覆盖、需要长思考的问题，并排除多选题。 | “A whole big, broad range … exclude multiple choice … require long, deep thought.” |
| [42:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2563s) | best-of-8 用成功率筛掉过易与过难问题。 | “Filter … based on a best of k filter … if they succeed on best of 8 …” |
| [43:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2599s) | 难度过滤既省 rollout 计算，也把训练集中在可学习区间。 | “This saves compute … skip problems … neither too hard nor too easy.” |
| [43:47](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2627s) | 未公开信息只能标未知；随后进入 DPO-inspired 推导。 | “We can speculate … but we have no concrete information … DPO inspired argument.” |
| [44:15](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2655s) | 起点是“最大化期望 reward，同时用 KL 留在 base/old 附近”。 | “Maximize the expected reward … KL regularizer … close to my base policy.” |
| [44:46](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2686s) | 解正则化最优条件会得到 policy ratio 关系。 | “That's going to give me this ratio of policies.” |
| [45:15](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2715s) | Kimi 把最优关系改写成可最小化的平方 surrogate。 | “Minimize the squared loss?” |
| [45:38](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2738s) | 虽非从 PPO 推出，对平方 loss 求梯度仍得到 GRPO-like 更新。 | “We're not going through the PPO path … take a gradient … looks … like GRPO.” |
| [46:11](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2771s) | 梯度里出现 group mean baseline 与平方 log-ratio 正则。 | “The KL term … in GRPO … reinvented the group mean normalized baseline.” |
| [46:47](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2807s) | Kimi 把无限增长的长 CoT 视为推理成本问题。 | “If we're making long CoTs … that's very wasteful.” |
| [47:16](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2836s) | 目标是难题仍做对，但回答尽可能短。 | “Compress the length … CoTs to be as short as possible … solve hard problems.” |
| [47:45](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2865s) | 先能长时间思考，再做长度压缩，是能力与效率的阶段性权衡。 | “Thinking for five minutes … great place to be … then compress the CoTs.” |
| [48:14](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2894s) | 正确答案应短；错误答案不能短到失去自我纠正机会。 | “Correct answers also should be short … incorrect answers too short … no way to recover.” |
| [48:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2923s) | 因此 length reward 不应一味惩罚所有长错误轨迹。 | “You don't force the incorrect answers to be super short.” |
| [49:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2950s) | 用各题历史 success rate 动态调采样难度。 | “They also look at the success rates of how often they're able to solve particular problems.” |
| [49:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2983s) | 代码靠生成测试用例验证，数学靠 answer-equivalence reward model。 | “For code … generate some new test cases … for math … reward model … answer equivalence.” |
| [50:09](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3009s) | Kimi 的数学 reward model 只是 verifier，准确率仍需 spot-check。 | “A reward model that checks the correctness of math answers.” |
| [50:40](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3040s) | 严格格式检查会把语义正确但格式稍异的答案误拒绝。 | “Sometimes it skips the boxed … ways … fail … by a strict correctness checker.” |
| [51:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3070s) | “verifiable”本身需要大量解析器与判定器工程。 | “It's a real rabbit hole getting the verified part of RLVR.” |
| [51:49](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3109s) | 系统例：一个极长 rollout 会成为整批的 straggler。 | “One really hard math problem … model is really chugging along.” |
| [52:09](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3129s) | 同步 batch 要等最慢样本，长 rollout 拉低利用率。 | “Long rollouts can really hurt you.” |
| [52:37](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3157s) | rollout 与训练可用不同机器，或在同机框架间切换。 | “Some … pure rollout machines … some … training machines, or … switching out frameworks.” |
| [53:07](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3187s) | 想复用 rollout 虽提高利用率，却会偏离严格 on-policy。 | “I could do so much better if only I could reuse my rollouts …” |
| [53:34](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3214s) | 现代技术报告会专门描述 rollout/training RL infrastructure。 | “Technical reports … a section talking about their RL infra.” |
| [54:05](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3245s) | 训练与推理可能共享机器以填补各自空闲期，但调度复杂。 | “Maybe they even share the same machines … complexities.” |
| [54:32](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3272s) | token 增长并非唯一进步来源；还可用筛选后的正样本迭代。 | “It's not that we're just unboundedly increasing tokens … OmniMath …” |
| [55:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3300s) | expert iteration/ReST 只模仿筛出的成功样本，与 RL 负梯度不同。 | “This is called expert iteration … unstable stuff … use expert iteration instead.” |
| [55:29](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3329s) | Kimi 部分结束，转入 Qwen3。 | “Questions about the Kimi stuff … move on … Qwen 3.” |

### 26.5 Qwen3、Qwen3-Coder-Next 与 reward hacking（55:58–69:31）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [55:58](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3358s) | Qwen Coder-Next 报告给出较多 agentic RLVR 细节。 | “Has the most details … agentic RLVR training out of the major tech reports.” |
| [56:35](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3395s) | 大模型完成 RLHF 后，再蒸馏到更小的可服务模型。 | “They do RLHF … then … distillation to get their smaller models.” |
| [57:05](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3425s) | Qwen3 组合 Kimi/DeepSeek 的成熟配方，并做难度过滤。 | “Tried and tested playbook … best parts of Kimi and DeepSeek … filtering for difficulty.” |
| [57:30](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3450s) | 去验证集近邻、人工筛 CoT；RL 数据只有约 4,000 例的课程口径。 | “Remove things … similar to validation data … manual filtering … just 4,000 examples.” |
| [58:08](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3488s) | Qwen3 用标签混合 thinking 与 non-thinking 样本。 | “They mix thinking and non-thinking things with tags.” |
| [58:39](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3519s) | 特殊字符串可 early exit CoT，切换到立即回答。 | “Early exiting thinking … append a special string … immediately stop the COT.” |
| [59:08](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3548s) | early termination 用来研究 thinking budget 与性能的关系。 | “As you vary the thinking budget … early termination trick … performance …” |
| [59:42](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3582s) | 小预算下 thinking 模式在数学/代码上仍优于 instant 模式。 | “With very small-thinking budgets, the thinking mode models are much better …” |
| [60:11](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3611s) | reasoning RL 后再做 general RL，可补回一般任务表现。 | “Reasoning RL and then general RL … improvements … general tasks.” |
| [60:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3643s) | 混合 thinking/non-thinking 是否保留会随模型版本变化。 | “In some … later releases, they've gone back on fusing both … into a single model.” |
| [61:07](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3667s) | 过渡到 Qwen3-Coder-Next，并纠正讲者口头型号顺序。 | “It's Coder-Next not Next-Coder.” |
| [61:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3703s) | agent post-training 没有全新魔法算法，关键仍是数据。 | “It's not like there's a new agent training algorithm … Data is the important thing.” |
| [62:06](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3726s) | agent 能力不能只在末端注入，需要较早的 coding/agent midtraining。 | “You can't just inject them at the end … extensive mid-training phase.” |
| [62:31](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3751s) | 从 PR 和相关代码构造长轨迹与 synthetic context。 | “Take pull requests … construct synthetic context … using RAG.” |
| [62:58](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3778s) | 用 LLM 转 markdown，并加入 coding web 与公开代码合成数据。 | “Use an LLM to transform this to … markdown … generate coding-ish synthetic data.” |
| [63:27](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3807s) | midtraining 也加入 fill-in-the-middle 代码补全任务。 | “Fill in the middle of a span … throw some of that data into mid training.” |
| [63:55](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3835s) | 从同一 midtrained 模型训练多个专门 coding agent experts。 | “Train … different expert models for different kinds of coding adjacent tasks.” |
| [64:27](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3867s) | 某些专家只负责数据格式与处理；型号口述本身不确定。 | “DeepSeek V3.5 or 3.2 … experts whose only job … format and process data.” |
| [64:52](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3892s) | 每个专家分别接受 SFT/RL，再汇合到主模型。 | “Each of these experts … do full on RL or and/or SFT training.” |
| [65:21](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3921s) | 专家覆盖多工具格式、QA 与 software engineering 环境。 | “Trained on many different tool formats … QA … software engineering agent.” |
| [65:59](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3959s) | 在可执行环境做 agent RL，曲线能上升但仍需防漏洞。 | “Do RL on these environments … RL performance … go up.” |
| [66:29](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3989s) | 模型可能偷看 Git future commit，绕过真实修复任务。 | “Look at future commits … just look up what the fix was.” |
| [67:03](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4023s) | 突然性能跃升可能是学会操纵 Git 历史，而非能力涌现。 | “Emergent jump … learned how to manipulate the Git calls to get the history.” |
| [67:32](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4052s) | RLVR 的可靠程度受 verifier 最弱环节限制。 | “RLVR is only as robust as your reward.” |
| [68:03](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4083s) | 即使形式证明系统也可能有不应开放的验证模式。 | “Verify proofs that are not meant to be verified in certain modes.” |
| [68:31](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4111s) | 课程快照展示小 active-parameter 模型在 SWE-bench 的高分。 | “70.6% on SWE-bench … three billion active parameter model.” |
| [68:59](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4139s) | 特定任务高分不保证跨域泛化，比较也须统一协议。 | “Task specific performance doesn't necessarily mean it'll generalize to broader domains.” |
| [69:31](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4171s) | 全讲结论：更难被 hack 的 reward 才能安全扩大 RL 计算。 | “We want more unhackable rewards … put in much more compute.” |

### 26.6 课堂问答：模式、midtraining、SFT 与蒸馏（70:00–75:42）

| 时间 | 中文主题 / 这一段解决什么 | 英文 cue 证据（本列均按字幕整理） |
|---|---|---|
| [70:00](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4200s) | 总结工程难度：RL 仍嘈杂难调，但已比早期复杂环境 PPO 顺滑。 | “RL remains very finicky and noisy … But it's not that hard … a lot smoother.” |
| [70:33](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4233s) | 问答：thinking mode 通常可由同一个模型的控制方式切换。 | “Thinking mode … what's happening on the back … it's actually one model.” |
| [70:53](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4253s) | 控制机制可放在 prompt/tag 中，而不只是外部 API flag。 | “The control mechanism in this case is in the prompt.” |
| [71:16](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4276s) | 问答：midtraining 数据格式与覆盖会怎样影响后续 RLVR。 | “How much … mid training … informs what is able to be learned during the RLVR post training.” |
| [71:43](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4303s) | 没有一刀切答案；pretraining 与 SFT 承担大量能力铺垫。 | “No one size fits all … pre-training and SFT are doing a lot of the heavy lifting.” |
| [72:07](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4327s) | 有代码/GitHub 覆盖很好；SFT 也能把模型带到开始获得 RL reward 的区域。 | “SFT will allow you to get close enough to start getting some rewards for RL.” |
| [72:33](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4353s) | midtraining 有助泛化，但讲者不把它说成所有设置的绝对成败点。 | “Mid-training is very nice to have … important … but not necessarily make or break.” |
| [73:07](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4387s) | 专家蒸馏需要一组 prompt，把专家行为迁到最终模型。 | “Distillation … require … a sequence of prompts on which the experts will be distilled.” |
| [73:27](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4407s) | 若算力与目标齐全，统一大训练目标可省去复杂蒸馏流程。 | “You might as well just throw it into the big training loop … don't have to deal with … distillation.” |
| [73:52](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4432s) | long-CoT SFT 通常属于 post-training，而非传统 midtraining 定义。 | “Long COT SFT … Long COT is not traditionally part of mid-training.” |
| [74:22](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4462s) | 长上下文扩展也用书、代码、合成等长数据，但目标不同。 | “Extend the context … books, code, synthetic data … long enough.” |
| [74:52](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4492s) | 问答：reasoning RL 与一般 RL 分阶段时如何避免遗忘。 | “Do it sequentially … how do you avoid forgetting earlier parts …” |
| [75:20](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4520s) | 课程图示把 reasoning 问题与非 reasoning RLHF 放在不同阶段。 | “Non reasoning problems happen at the final RLHF phase … stage IV.” |
| [75:42](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4542s) | 本讲正式结束，现场掌声收尾。 | “[APPLAUSE]” |

<a id="l16-pdf-coverage"></a>

## 27. PDF p.1–61 逐页覆盖索引

> 下表让每页恰好出现一次。连续页段只是压缩表长；段内每页都经过普通单页图和 contact sheet 目视检查，公式/代码/图表页另看原分辨率图。

| PDF 页 | 该页段内容 | 正文落点 |
|---:|---|---|
| p.1 | 标题：Post-training 2 / RL from verifiable rewards | §0、§3 |
| p.2–4 | 课程进度；RLHF overoptimization；RLVR 目标；算法/现象/案例地图 | §0–§3、§21 |
| p.5–8 | policy gradient→TRPO→PPO；PPO 概念与实现复杂性 | §4–§6 |
| p.9–16 | LM PPO：token action、代码外循环/loss/rollout、KL shaping、GAE、训练曲线 | §5–§6、§10 |
| p.17 | PPO/value 成本与 DPO 离线/pairwise 限制 | §6–§7 |
| p.18–21 | GRPO 总览式、简化实现、`numpy.std()+1e-4`、DeepSeekMath 结果 | §7–§10 |
| p.22–24 | baseline 定理、std 的偏差、Dr.GRPO 与 response-length bias | §11–§12 |
| p.25 | R1/Kimi/Qwen 三个案例地图 | §13–§20 |
| p.26–30 | R1 社会现象；GRPO；R1-Zero reward/data/base；长度与 aha 争议 | §13 |
| p.31–35 | R1 vs R1-Zero；R1只公开small-amount cold-start；p.33 的1K是独立s1案例；language consistency；600k/200k；后续 SFT/RLHF | §14 |
| p.36–38 | R1 结果；800k traces 蒸馏；PRM/MCTS unsuccessful attempts | §15 |
| p.39–44 | Kimi k1.5；难度筛选/SFT 边界；正则化 PG；长度 reward；curriculum/reward | §16–§17 |
| p.45–48 | on-policy系统/partial rollout；scaling图；expert-iteration/ReST与RL负梯度对照 | §16.6、§18、§21 |
| p.49–54 | Qwen3 总流程；3995 examples；thinking fusion；test-time scaling；general RL trade-off | §19 |
| p.55–60 | Qwen3-Coder-Next：agent RL、600B repository tokens、experts、800k environments、reward hacking | §20–§21 |
| p.61 | overoptimization、GRPO 缺陷与三个案例 recap | §1、§21、§29 |

### 27.1 关键公式/图表的原分辨率核对

- p.5：确认三行逻辑是高方差 policy gradient → 当前策略附近线性化的 TRPO → ratio clipping 的 PPO。
- p.14：确认截图代码是 clamp(logprobs-ref_logprobs,min=0) 单边 shaping；它与 p.18 的 $`e^d-d-1`$ estimator 分开讲。
- p.18：确认原始截图片段写的是 **response-level** (pi(o_i|q)/pi_{old}(o_i|q))，所以 §7.3 没把它伪装成 token 内层求和。
- p.20：确认代码是 `rewards.std() + 1e-4`；NumPy 默认 `ddof=0`，对应 population std，而不是 sample std。
- p.23：确认 GRPO 的 $`1/|o_i|`$ 与 std 被红色标出；Dr.GRPO 对照删除二者，保留 token sum、outer $`1/G`$ 和含自身的 reward mean。
- p.24：确认五幅图分别是 reward、总长度、正确长度、错误长度、平均 benchmark；图只支持该实验中的曲线，不证明长度对能力的因果。
- p.28–35：确认 R1-Zero 与 R1 是不同 pipeline；p.33 的 1K 是独立 s1 图，不能归入 R1；p.35 写 600k reasoning/non-verifiable 与 200k non-reasoning 的课程口径。
- p.33：确认 1K、s1/s1-32B 属于独立 s1 样本效率图，不是 DeepSeek-R1 cold-start 数量。
- p.37–38：确认 800k traces 属于蒸馏页；PRM/MCTS 位于“unsuccessful attempts”页，不能升级为永远无用。
- p.41：确认课程 bullet 写 best-of-8，而嵌入报告文字谈十次高温估难；§16.3 保留冲突。
- p.42：确认 Kimi 页同时展示 KL-regularized objective、nonparametric 最优关系、平方 surrogate 和 baselined PG。
- p.43：确认 $`\lambda`$ 从 0.5 到 -0.5，正确用 $`\lambda`$，错误用 $`\min(0,\lambda)`$，等长 fallback 全设0。
- p.44：确认 $`p_i\propto1-success\_rate`$、代码新测试、数学800k RM与84.4/98.5 spot-check。
- p.45–46：确认 on-policy inference、train/rollout framework 切换、长 CoT 不均衡是三项系统障碍。
- p.51：确认数字是 **3995 examples**；p.53 的横轴/标记是不同 thinking budgets；p.54 的表说明 general RL 后并非所有 math/STEM 都上升。
- p.56–60：确认 600 Billion 是 repository-level tokens，800k 是构造的 agent tasks；Git 泄漏页只用作防守性 reward-audit 案例。

<a id="l16-sources"></a>

## 28. 来源、SHA、字幕与视觉核验边界

### 28.1 课程主来源

- [Stanford CS336 Spring 2026 官方课程页](https://cs336.stanford.edu/)。
- [官方 Lecture 16 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)。
- [Stanford Online Lecture 16 视频](https://www.youtube.com/watch?v=dIFAi87Ws4E)。

| 本地材料 | 数量/大小 | SHA256 |
|---|---:|---|
| `lecture_16.pdf` | 61 页；6,755,659 bytes | `1A2BEA825499F226B1C3FE990606253AD1B4FDBDF2AE90B4D12D7D0C07F2C8B3` |
| `transcript_en_us.txt` | 人工 en-US；1,687 cues；86,851 bytes；00:05–75:42 | `BBF3C3F754214978495327B1B563E2F6234BB3AAF5BA5F5F760B2597BB4E8904` |
| `lecture_16_extracted.txt` | 13,139 bytes；只作检索索引 | `AD245E6537C761B3A1145A371A96B04342F5CA43C1629621A51C6EC76CF9F056` |

### 28.2 PDF 视觉核验记录

1. 用 pypdf/pypdfium2 确认 61 页，并生成 61/61 普通页、61/61 原分辨率关键页。
2. 打开 7 张 contact sheets：p.1–10、11–20、21–30、31–40、41–50、51–60、61。
3. 对 49 张公式、代码、表格或案例关键页逐张看原分辨率：

~~~text
p.5,7,9,12,14,15,18–24,26–39,40–61
~~~

其中 p.18、20、23、24、42、43、51 的公式/代码不是从 OCR 猜；其余页面也由 contact sheet 与普通单页图覆盖。本文没有直接嵌入截图，而是把箭头、坐标轴、数字和“不能推出什么”转成中文。

### 28.3 一手补充来源

- [PPO 原论文](https://arxiv.org/abs/1707.06347)：ratio/clipped surrogate 的原始算法边界。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：PPO value/advantage 的背景。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 最初的公开模型/数学任务设置。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：R1-Zero、R1 多阶段、蒸馏和 unsuccessful attempts。
- [Understanding R1-Zero-Like Training / Dr.GRPO](https://arxiv.org/abs/2503.20783)：std、长度归一化与 base behavior 的复核。
- [Kimi k1.5](https://arxiv.org/abs/2501.12599)：难度估计、正则化 PG、长度 reward 与系统配方。
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)：p.33 独立 1K/SFT 案例的来源，不能移作 R1 cold-start 数量。
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)：thinking/non-thinking 融合、thinking budget 与模型系列。
- [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)：midtraining、可验证 coding environments 与 agent RL。

这些论文只支持各自公开实验。课程的图表、口头比较和 2026 配置是**课程时点快照**；本文不把 benchmark 排名写成当前永久排名，也不推断公司未公开的 prompts、采样温度或集群细节。

### 28.4 三处材料口径必须保留

1. **Kimi 难度估计**：PDF/口述概括为 best-of-8；同页报告截图写十次高温回答。两者分别记录。
2. **Qwen3 样本数**：PDF p.51 是 3995；口述“约 4000”是近似。
3. **Coder 名称**：视频 55:58 一度说 `Qwen 3.5-Next-Coder`，随后 61:20 自行纠正为 `Coder-Next`；PDF 和 2026 官方报告名称是 **Qwen3-Coder-Next**。本文采用官方名，并保留导航原 cue 供审计。

### 28.5 本文补充与未验证边界

- 【补充解释】完整 group、PPO 六 token 表、padding、权重传输和 3995×8×2048 都是教学手算，不冒充真实训练日志。
- 【补充】baseline 的零期望证明、含自身均值缩放与 leave-one-out 推导用于解释课程批评；真实优化器、clip、多 epoch 会再改变估计。
- 本地没有真实 RL 集群、Megatron/vLLM 联合环境或 Qwen3-Coder-Next 训练权重，因此只做公式、字幕、PDF 和数值复算，不声称复现实测吞吐或 benchmark。

## 29. 一页复习流程与学完能力清单

### 29.1 遇到一条 GRPO 公式时按这个顺序读

~~~text
先圈层级：prompt → group → response → token
  ↓
reward 是谁判的？验证了什么、漏了什么？
  ↓
mean/std 是同题 group 吗？std 除 G 还是 G-1？std=0 怎么办？
  ↓
advantage 是否含自身？是否再除随机 std？
  ↓
ratio 是 current/old；KL 是 current/reference，不要串角色
  ↓
每 token 先算 raw 与 clipped 两项；A 正负分别取 min
  ↓
token sum/mean → response → group → batch，每个分母写出来
  ↓
最后才看 reward 曲线；同时看长度、独立准确率、人工审计和系统利用率
~~~

### 29.2 学完后你应能独立做到

- 从四个 rewards 算 mean、population/sample std、epsilon 后 advantage；
- 解释 std=0、(G=1)、含自身 mean 和 leave-one-out 的不同；
- 对正负 advantage 逐 token 算 PPO clip，并区分 old 与 reference；
- 复算 response mean、global token mean 与 Dr.GRPO token sum 的权重差；
- 解释为什么组内 advantage 不是 value function，为什么随机 std 不自动无偏；
- 读懂 R1-Zero、生产 R1、Kimi、Qwen3 和 Qwen3-Coder-Next 的公开配方边界；
- 不把 CoT 变长、aha、benchmark 上升写成未经对照的因果结论；
- 为数学、代码或 agent 任务写 reward contract、反例、聚合口径和独立复核；
- 用 rollout 长度、padding 利用率与权重传输账诊断 RLVR 系统瓶颈。

如果这些还不能独立完成，回到 §8 重算 group，再做题 24–51；如果公式会算但不知道何时可信，回到 §3、§11、§21。
