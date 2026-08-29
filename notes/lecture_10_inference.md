# CS336 Lecture 10：Inference（用已训练模型产生输出）——从第一枚 token（模型处理的离散编号）到动态批处理

> **页首词卡（防止目录先冒出陌生词）：**
>
> - Transformer 是用 attention（注意力：按相关性取信息）和 MLP（Multi-Layer Perceptron，多层感知机）反复变换 token 向量的模型。request 是一次服务任务，prompt 是输入 token 序列；tokenizer（分词器）是把文本转换成模型所用 token IDs 的规则/程序；autoregressive 是按已有前缀逐枚生成，prefill 是一次处理 prompt，decode 是逐枚生成，logit 是 softmax 前分数，softmax 把一组分数变成和为1的概率；greedy 是总选最高概率项，sampling 是按概率随机抽取；KV cache 是保存历史 key/value 的缓存。
> - GPU（Graphics Processing Unit，图形处理器）负责并行计算；HBM（High-Bandwidth Memory，高带宽显存）存模型和缓存。FLOP 是一次浮点运算，byte=8 bits，BF16 是常占2 bytes的 bfloat16；arithmetic intensity 是 FLOPs/bytes，bandwidth 是每秒搬运量，compute-bound/memory-bound 表示计算/搬运先卡住，kernel 是 GPU 执行的一段底层计算程序。
> - latency 是等待时间；TTFT（Time To First Token）是首枚时间，ITL（Inter-Token Latency）是相邻输出间隔，throughput 是总完成率，goodput 是满足 SLO（Service-Level Objective，服务目标）的有效完成率。
> - MHA/MQA/GQA（Multi-Head/Multi-Query/Grouped-Query Attention）分别是不共享/全共享/分组共享 KV heads；MLA（Multi-head Latent Attention）缓存低维 latent，CLA（Cross-Layer Attention）跨层共享 KV。
> - quantization 用少 bit 近似数值，scale 是步长，zero point 是整数零点，clamp 是截到边界；PTQ（Post-Training Quantization）和 QAT（Quantization-Aware Training）是训练后/训练中量化，Hessian 是 loss（模型“坏程度”）对参数的二阶变化信息。pruning 是删除结构，distillation 是 student 学 teacher，calibration 是用小样本估计范围/重要性。
> - proposal/target 是提候选/决定最终分布的模型，rejection/residual 是拒绝过量候选/补足概率；ragged batch 是长度不齐的批，iteration scheduling 是每轮重排请求；fragmentation 是碎片，paging/block table 是分页与块映射，copy-on-write 是要写共享块时才复制。

> **目标读者：** 只会加、减、乘、除，第一次系统学习大语言模型推理。  
> **目标：** 不看完整 85 分钟视频，也能解释一次请求怎样运行、重算课程的 FLOPs/显存/延迟模型，并知道每种优化究竟压缩了什么。  
> **官方代码讲义：**[`lecture_10.py`，commit `8b59b507`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py)。  
> **官方视频：**[Stanford Online Lecture 10](https://www.youtube.com/watch?v=EfM546A79aM)。

## 0. 阅读方法、来源边界与整讲地图

### 0.1 第一次阅读怎么走

第一次阅读请跳过 §1 的五分钟复习卡，按下面路线：

1. §2 先弄清“一次请求快不快”究竟有哪几种指标。
2. §3–§4 看 autoregressive 生成、prefill、decode 与每个 tensor 的形状。
3. §5–§9 手算 arithmetic intensity、KV cache、延迟和吞吐。
4. §10–§14 看模型侧捷径：GQA、MLA、CLA、稀疏注意力、量化、剪枝与蒸馏。
5. §15–§17 看系统侧捷径：speculative sampling、continuous batching、PagedAttention。
6. 最后读 §18–§20，再独立完成 §21；§22 有完整答案。

### 0.2 可点击目录

- [§1 五分钟复习卡](#1-五分钟复习卡首次阅读请跳过)
- [§2 使用场景与指标](#2-inference-为什么重要怎样才叫快)
- [§3 Autoregressive、prefill、decode 与 KV cache](#3-autoregressive-生成为什么必须一枚一枚-token-来)
- [§4 Shape 字典](#4-shape-字典与一个最小-transformer-地图)
- [§5–§7 Arithmetic intensity](#5-arithmetic-intensity从一次矩阵乘开始)
- [§8–§9 Llama 2 13B 显存、延迟与吞吐](#8-参数量与-kv-cache一笔完整的-llama-2-13b-账)
- [§10–§12 KV-cache 架构优化](#10-mhamqagqa压缩-head-轴)
- [§13 量化](#13-quantization把每个数存得更小)
- [§14 剪枝与蒸馏](#14-pruning-与-distillation删掉再修复)
- [§15 Speculative sampling](#15-speculative-sampling让小模型先猜大模型一次验一串)
- [§16 Continuous batching](#16-continuous-batching请求随到随进随完随退)
- [§17 PagedAttention](#17-pagedattention把-kv-cache-切成可搬动的固定小块)
- [§18–§20 决策树、误区、术语](#18-性能诊断与优化决策树)
- [§21 自测](#21-自测题80-题) · [§22 答案](#22-自测答案180)
- [§23 视频导航](#23-视频时间导航) · [§24 源码与图片覆盖](#24-源码函数611-行与-22-张图片覆盖)

### 0.3 本讲的主因果链

**【课程内容】【源码 `main`】【视频补充】[00:06](https://www.youtube.com/watch?v=EfM546A79aM&t=6s)**

```text
模型已经训练好 + 用户给 prompt
        ↓
prefill 一次处理整段 prompt，建立 KV cache
        ↓
decode 每步只能知道下一枚 token；生成依赖前一步
        ↓
矩阵变得“又薄又频繁”，权重与 KV cache 搬运常成为瓶颈
        ↓
模型侧：少存 KV / 少存每个数 / 少算参数
系统侧：猜后验证 / 动态组 batch / 分页管理 KV
        ↓
必须同时检查 accuracy、延迟、吞吐、SLO 与显存
```

视频 [00:14](https://www.youtube.com/watch?v=EfM546A79aM&t=14s) 从 scaling laws 转入 inference；[08:21](https://www.youtube.com/watch?v=EfM546A79aM&t=501s) 给出与上面相同的全讲预告。

### 0.4 四类来源标签

- **【课程内容】**：当前 commit 的代码讲义、22 张本地图或官方视频明确出现。
- **【视频补充】**：人工英文字幕中的口头解释、课堂问答或当场纠正。
- **【补充理解/例子】**：为零基础读者增加的中间算术、反例、SLO/goodput 等桥梁。
- **【延伸】**：来自论文、官方硬件或框架文档；不是老师逐字说法。

动态数字均写明型号、精度、日期或材料版本。源码中的 OpenAI “约 8.6T tokens/day”来自第三方新闻估计，不是 OpenAI 官方审计数据；本文只把它记录为课程在 2026 年采用的数量级快照，不用它推导任何结论。

### 0.5 资料核验摘要

- 本地源码为 611 个物理行，SHA-256 为 `35CC8938F4E577A70783616703C8C7C16932DDE78BC11CC081D4F753228529E4`；以任务指定 commit 为版本边界。
- 人工字幕轨为 `English (United States)`，共 1509 段；首段 00:05，末段从 85:21 开始，正文约到 85:24。
- 源码实际引用 22 张互不重复的本地图，全部以原分辨率查看；另有 6 个远程图 URL，不计入“22 张本地图”。
- 本讲没有 PDF，因此没有 PDF 页覆盖表；§24 用无缺口源码行段表与逐图核验表代替。

#### 源码勘误与本文口径（读代码时先看）

当前 commit 的教学代码有几处内部不一致；本文保留课程意图，同时把修正写透明：

| 源码位置 | 原文问题 | 本文怎样处理 |
|---|---|---|
| line 228–230 | 先把 Q/K 写成扁平的 `[B,T,D]`、`[B,S,D]`，下一行又突然出现 `[B,S,T,K,G]`，缺少拆 heads、配组和 transpose 的桥 | §4.3 明写 `Q[B,T,N,H]`、`K/V[B,S_total,K,H]` 与 score `[B,T,N,S_total]`；§7 的扁平公式只是把 `N×H=D` 合并后的资源账 |
| line 267 | 把 `num_params` 说明成“in bytes” | `num_params` 是**参数个数**；line 291 只算个数，line 294 才乘 BF16 的 2 bytes 得参数显存 |
| line 398 左右 | GQA `K=8,B=64` 后写 “Worse latency” | §10.4 依照同一源码公式重算；memory 变小，所以理想 latency 应更低 |
| line 451 | 写 “Less memory means higher latency/throughput” | 按 `latency=memory/bandwidth`，应是 **lower latency、higher throughput** |
| FP8 口径 | 把 H100 E4M3 范围概括为 `[-240,240]` | §13.4 与 NVIDIA 当前官方文档分开列，不把简化范围当所有 E4M3 变体的事实 |

这些是教学源码勘误，不改变本讲“推理常受内存搬运限制”的主线。

### 0.6 最少前置知识

只需要会四则运算，并接受下面五件事；后文会在使用时再解释：

1. 一个 tensor 是规则排成多维表的一堆数字，`shape` 说明每个方向有多少格。
2. 矩阵乘的一个输出格，是一行与一列对应相乘再相加。
3. $`2^3=2\times2\times2=8`$；$`O(T^2)`$ 只说增长量级，不是精确等号。
4. 概率在 0 到 1 之间，一组完整结果的概率和为 1。
5. GPU 同时受“每秒能算多少”和“每秒能搬多少 bytes”限制。

不要求先会 Transformer、CUDA（NVIDIA 的 GPU 编程平台）、概率论或服务系统；首次出现的专门词都会就地定义。

---

## 1. 五分钟复习卡（首次阅读请跳过）

1. **Inference（推理）**：模型训练完成后，给它 prompt，让它生成 response。（§2）
2. **TTFT** 是请求到第一枚输出 token 的时间；**ITL/TPOT** 是后续 token 之间的时间；end-to-end latency 是请求到最后一枚 token 的总时间。三者不能互换。（§2.3）
3. Prefill 能并行处理 prompt；decode 必须逐 token 串行。（§3.2）
4. 不缓存时 attention 累计约 $`\sum t^2=O(T^3)`$；缓存 K/V 后约 $`\sum t=O(T^2)`$。（§3.4）
5. Arithmetic intensity（算术强度）是 FLOPs/bytes。矩阵 $`[B,D]@[D,F]`$ 在 $`B\ll D,F`$ 时约为 $`B`$。（§5）
6. 课程 H100 示例的理论 roof point 是 $`989/3.35\approx295`$ FLOP/byte；这是特定 BF16 dense 峰值模型，不是所有 kernel 的实测阈值。（§5.5）
7. SwiGLU MLP 强度在权重占主导时约为 $`BT`$；attention 用 $`S_{total}=S_{old}+T`$ 后，强度为 $`TS_{total}/(S_{total}+T)`$。（§6–§7）
8. Prefill 令 $`S_{old}=0,T=S_{total}=P`$，强度为 $`P/2`$；decode 令 $`T=1`$，强度为 $`S_{total}/(S_{total}+1)<1`$。（§7.4）
9. KV cache 只长期存历史 K/V，不存历史 Q。BF16 每序列字节数为 $`S_{total}\times K\times H\times L\times2\times2`$。（§8.2）
10. MHA/GQA/MQA 压缩 KV-head 轴；MLA 压缩 head-feature 轴；CLA 压缩 layer 轴；local attention 截断 sequence 轴。（§10–§12）
11. 量化缩小每个数，但只有硬件和 kernel 真支持时才一定可能加速；它也可能因 dequant 开销或精度下降而不划算。（§13）
12. Speculative sampling 用 draft $`p`$ 提议、target $`q`$ 验证；正确接受/残差规则能保持 target 分布完全一致。（§15）
13. Continuous batching 每个 decode iteration 都可加入新请求、移走已完成请求。（§16）
14. PagedAttention 用 block table 把连续逻辑 KV 映射到不连续物理块，并支持 prefix sharing 与 copy-on-write。（§17）
15. 优化顺序：先定义 SLO，再量出 TTFT/ITL/显存/吞吐，最后按瓶颈选择手段并回归 accuracy。（§18）

---

## 2. Inference 为什么重要，怎样才叫“快”

### 2.1 request、prompt、token、response

**Inference（推理）** 是使用已经训练好的模型产生预测或文本。一次 **request（请求）** 是客户端交给服务的一项工作；**prompt（提示）** 是请求中的输入文本；**tokenizer（分词器）** 是把文本按一套固定规则转换成 token IDs 的程序；**token（词元）** 是模型实际处理的离散编号单位；**response（响应）** 是模型输出。同一段可见文字若用了不同 tokenizer，token IDs 可能不同，KV cache 也不能直接共享。

视频 [00:17](https://www.youtube.com/watch?v=EfM546A79aM&t=17s) 用最短定义开场：模型已训练好，收到 prompt，要尽可能准确、快速地给 response。

Inference 不只等于聊天：

- 实际产品：聊天、代码补全、agent、批量数据处理；视频 [00:39](https://www.youtube.com/watch?v=EfM546A79aM&t=39s)。
- Evaluation（评测）：需要模型真正生成答案；视频 [01:12](https://www.youtube.com/watch?v=EfM546A79aM&t=72s)。
- Reinforcement learning（强化学习）：生成 rollouts、打分、再更新；视频 [01:21](https://www.youtube.com/watch?v=EfM546A79aM&t=81s)。

训练通常是一次大成本；推理是每次请求都再次发生的成本。视频 [01:48](https://www.youtube.com/watch?v=EfM546A79aM&t=108s) 强调 one-time 与 repeated cost 的区别。课程 [02:01](https://www.youtube.com/watch?v=EfM546A79aM&t=121s) 引用 8.6T/day 估计只是时点快照；真正不变的结论是：请求量乘每请求成本，可能让长期推理成本超过训练成本。

### 2.2 online 与 offline

- **Online inference（在线推理）**：用户请求随时到达，通常有交互延迟目标。流量会突增、prompt/output 长度未知。
- **Offline inference（离线推理）**：数据集合预先已知，例如给一百万文档打标签；通常更关心总作业时间和每美元 tokens。

视频 [02:40](https://www.youtube.com/watch?v=EfM546A79aM&t=160s) 说明 agent 的内部 token 不一定给人阅读；[03:04](https://www.youtube.com/watch?v=EfM546A79aM&t=184s) 描述 query→工具调用/思考→最终输出的长链。离线任务可以等凑大 batch；在线请求若等太久，TTFT 会先坏掉。

### 2.3 六个性能词，用同一条时间线定义

**【课程内容】【视频补充】[05:02](https://www.youtube.com/watch?v=EfM546A79aM&t=302s)**

- **Latency（延迟）** 泛指等待多久；必须写清是 TTFT、ITL 还是端到端，不能只写“latency”。
- **TTFT（Time To First Token，首 token 时间）**：请求到达至第一枚输出 token 可见。
- **ITL（Inter-Token Latency，token 间延迟）**：相邻输出 token 出现时间之差。文献也常写 **TPOT（Time Per Output Token）**。
- **End-to-end latency（端到端延迟）**：请求到达到最后一枚输出 token 完成。
- **Throughput（吞吐）**：所有请求合计每秒完成多少 tokens 或 requests；必须带单位。
- **SLO（Service-Level Objective，服务级目标）**：服务承诺要达到的目标，例如“至少 95% 请求 TTFT≤1 s 且平均 ITL≤0.2 s”。
- **Goodput（有效吞吐）**：不仅完成，而且满足 SLO 的请求速率。原始 throughput 很高但多数请求超时，goodput 仍很低。[DistServe 原论文](https://arxiv.org/abs/2401.09670)

课程 [05:13](https://www.youtube.com/watch?v=EfM546A79aM&t=313s) 定义 TTFT；[06:00](https://www.youtube.com/watch?v=EfM546A79aM&t=360s) 转到单请求 token 流速；[06:21](https://www.youtube.com/watch?v=EfM546A79aM&t=381s) 定义多请求 throughput。

**【补充例子】一条请求时间线：**

```text
t=0.00  请求到达
t=0.20  排队结束，开始 prefill
t=0.70  输出 token 1
t=0.80  输出 token 2
t=0.95  输出 token 3
t=1.10  输出 token 4，请求结束
```

逐项算：

1. TTFT：$`0.70-0=0.70`$ s。
2. 三个 ITL：$`0.80-0.70=0.10`$、$`0.95-0.80=0.15`$、$`1.10-0.95=0.15`$ s。
3. 平均 ITL：$`(0.10+0.15+0.15)/3=0.133`$ s。
4. 端到端 latency：$`1.10-0=1.10`$ s。
5. 若 2 秒内共完成 10 个同类请求、40 个 token，request throughput 是 $`10/2=5`$ requests/s，token throughput 是 $`40/2=20`$ tokens/s。
6. 若只有 8 个请求满足 SLO，goodput 是 $`8/2=4`$ compliant requests/s。

这说明“latency = seconds/token”太粗：0.70 s 的首等候、0.133 s 的 token 间隔和 1.10 s 的完整等待是三件事。视频 [06:46](https://www.youtube.com/watch?v=EfM546A79aM&t=406s) 说明离线批处理可主要优化 throughput。

### 2.4 为什么训练和推理的形状不同

训练时，目标序列已知，许多 token 位置能一起进入矩阵乘。Inference 的 **autoregressive（自回归）** 意思是下一枚 token 的概率依赖已经出现的所有 token；尚未生成的 token 不存在，不能提前并行算。

视频 [07:47](https://www.youtube.com/watch?v=EfM546A79aM&t=467s) 开始对比；[07:50](https://www.youtube.com/watch?v=EfM546A79aM&t=470s) 明确说生成必须 sequential。这个串行依赖是后面薄矩阵、低 arithmetic intensity 与动态 batching 的根源。

---

## 3. Autoregressive 生成：为什么必须一枚一枚 token 来

### 3.1 从 logits 到下一枚 token

模型读入 token IDs 后，对词表中每个候选输出一个 **logit（未归一化分数）**。若词表大小为 $`V`$，一个位置的 logits shape 是 $`[V]`$；batch 版本是 $`[B,V]`$。Softmax 把 logits 变成总和为 1 的概率。**Greedy decoding（贪心解码）** 每次直接选概率最大的 token；**sampling（随机采样）** 按概率随机抽一枚，所以小概率 token 仍可能被抽中。

例如 logits 对三个词是 $`[2,1,0]`$。减去最大值后为 $`[0,-1,-2]`$，取指数约 $`[1,0.368,0.135]`$，总和 $`1.503`$，概率约 $`[0.665,0.245,0.090]`$。选出的 token 拼到历史末尾，模型才能计算下一步。

### 3.2 prefill 与 decode/generation

**Prefill（预填充）**：一次处理完整 prompt，并为每层、每个 prompt token 建立 K/V；prefill 的最后一个位置已经给出第一枚输出 token 的分布。  
**Decode / generation（解码/生成）**：把上一调用刚采样的 token 作为本次输入，追加它的 K/V，再采样下一枚 token。

视频 [20:30](https://www.youtube.com/watch?v=EfM546A79aM&t=1230s) 开始推理算术；[20:41](https://www.youtube.com/watch?v=EfM546A79aM&t=1241s) 展示最朴素的“整段历史重新过模型”。

**为什么 token 间不能完全并行？** 第 2 枚输出的 probability 要看第 1 枚实际采样结果；第 3 枚又要看前两枚。若第 1 枚不同，后面条件分布也会不同。

### 3.3 KV cache 存什么，为什么不存 Q

Attention 把当前 hidden state 投影成：

- **Q（Query，查询向量）**：当前这个位置“要找什么”。
- **K（Key，键向量）**：历史位置“可用什么索引来匹配”。
- **V（Value，值向量）**：匹配后真正汇总的内容。

**KV cache（键值缓存）** 把历史 token 的 K 和 V 保存在 GPU 的 **HBM（High-Bandwidth Memory，高带宽显存）** 中。历史 token 的 Q 不会被未来位置重复使用；未来位置会产生自己的新 Q，所以不需要长期缓存历史 Q。

视频 [21:28](https://www.youtube.com/watch?v=EfM546A79aM&t=1288s) 指出朴素重复计算很昂贵；[22:09](https://www.youtube.com/watch?v=EfM546A79aM&t=1329s) 解释前缀工作能复用；[22:45](https://www.youtube.com/watch?v=EfM546A79aM&t=1365s) 引入 KV cache。因 causal attention 中旧位置不看未来，追加 token 不会改变旧 K/V。

### 3.4 无 cache 的 $`O(T^3)`$ 与有 cache 的 $`O(T^2)`$

这里临时用 $`T_{gen}`$ 表示“连续生成多少枚 token”，避免与 §4 中一次并行处理的 query 长度 $`T`$ 混淆。

**无 cache：** 第 $`t`$ 步把长度约 $`t`$ 的整段历史重新送入 full attention。一层 attention 的 score 数约 $`t^2`$：

```math
1^2+2^2+\cdots+T_{gen}^2
=\frac{T_{gen}(T_{gen}+1)(2T_{gen}+1)}6.
```

最高次是 $`2T_{gen}^3/6`$，所以记作 $`O(T_{gen}^3)`$。

**有 cache：** 第 $`t`$ 步只用一个新 Q 去看 $`t`$ 个 K，score 数约 $`t`$：

```math
1+2+\cdots+T_{gen}
=\frac{T_{gen}(T_{gen}+1)}2
=O(T_{gen}^2).
```

**纯求和玩具：** 先只把四次 forward 的可见长度编号为 1、2、3、4，用来观察增长率；这张表**不是**下面“prompt=3、生成4枚”的真实调用表。

| 步 $`t`$ | 无 cache attention 量 $`t^2`$ | 有 cache 新 Q 对历史量 $`t`$ |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 4 | 2 |
| 3 | 9 | 3 |
| 4 | 16 | 4 |
| 合计 | $`1+4+9+16=30`$ | $`1+2+3+4=10`$ |

现在把 prompt 与输出放回同一口径。完整 prompt 有 3 枚，prefill 已经采样出 $`y_1`$。要生成 4 枚输出 $`y_1,y_2,y_3,y_4`$，只需 **1 次 prefill + 3 次后续 decode**：

- 无 cache：四次 forward 分别重算长度 $`3,4,5,6`$ 的整段，attention 方格数为

  $`3^2+4^2+5^2+6^2=9+16+25+36=\boxed{86}.`$

- 有 cache、把 prefill 粗略当一个 dense $`3\times3`$ 方格：

  $`3\times3+4+5+6=9+4+5+6=\boxed{24}.`$

- 有 cache、严格数 causal attention 允许的边：prefill 三行分别能看 1、2、3 个位置，随后三次 decode 看 4、5、6 个位置：

  $`1+2+3+4+5+6=\boxed{21}.`$

所以旧式写法 `3+4+5+6=18` 漏掉了 prefill 内前两行的 attention，不是这条请求的完整 cached attention 账。

边界：这是 attention 主项。MLP 无 cache 会反复处理整段，累计约 $`O(T_{gen}^2)`$；有 cache 后只处理新 token，累计约 $`O(T_{gen})`$。Prefill 自身、投影、softmax、采样和通信仍要算。视频 [23:10](https://www.youtube.com/watch?v=EfM546A79aM&t=1390s) 展示 cached 流程，[23:12](https://www.youtube.com/watch?v=EfM546A79aM&t=1392s) 命名 prefill，[24:19](https://www.youtube.com/watch?v=EfM546A79aM&t=1459s) 逐轴描述 cache。

### 3.5 生成四枚 token 的状态表

令 prompt 是 `[我, 喜欢, 学习]`，四枚采样输出依次是 $`y_1=`$`机器`、$`y_2=`$`学习`、$`y_3=`$`。`、$`y_4=`$`<eos>`。一次 **forward call（前向调用）**，就是把本次输入 tensor 送进模型、从输入层一路算到输出 logits 的一次完整模型调用；采样输出发生在这次调用得到 logits 之后：

| forward call | 调用前 cache $`S_{old}`$ | 本次输入 | 本次 $`T`$ | 调用后 cache $`S_{total}`$ | 本次采样输出 |
|---|---:|---|---:|---:|---|
| prefill | 0 | 3 枚完整 prompt | 3 | $`0+3=3`$ | $`y_1=`$`机器` |
| decode 1 | 3 | $`y_1=`$`机器` | 1 | $`3+1=4`$ | $`y_2=`$`学习` |
| decode 2 | 4 | $`y_2=`$`学习` | 1 | $`4+1=5`$ | $`y_3=`$`。` |
| decode 3 | 5 | $`y_3=`$`。` | 1 | $`5+1=6`$ | $`y_4=`$`<eos>` |

关键：$`y_4`$ 是最后一次调用**产出的结果**，尚未作为下一次输入，因此它的 K/V 还没进 cache。只有还要继续生成 $`y_5`$ 时，才会把 $`y_4`$ 输入下一次 decode，把 cache 从 6 加到 7。

`<eos>` 是 end-of-sequence（序列结束）token。视频 [24:38](https://www.youtube.com/watch?v=EfM546A79aM&t=1478s) 再次区分两阶段；[24:55](https://www.youtube.com/watch?v=EfM546A79aM&t=1495s) 强调 generation 仍串行，但无需重算旧 cache。

---

## 4. Shape 字典与一个最小 Transformer 地图

### 4.1 本讲所有大写维度

**【课程内容】【源码 8–11、100–117】【视频补充】[09:23](https://www.youtube.com/watch?v=EfM546A79aM&t=563s)**

| 符号 | 含义 | 例子 |
|---|---|---:|
| $`B`$ | batch 中并发序列数 | 4 requests |
| $`S_{old}`$ | 本次 forward 调用前已经缓存的历史 token 数 | prefill 0；某次 decode 128 |
| $`T`$ | 本次 forward 一起输入、产生 query 的 token 数 | prefill 128；decode 1 |
| $`S_{total}`$ | 本次 append 后 attention 可见的全部 source 数，$`S_{total}=S_{old}+T`$ | $`0+128=128`$ 或 $`128+1=129`$ |
| $`D`$ | model/hidden width | 5120 |
| $`F`$ | MLP 中间宽度 | 13824 |
| $`N`$ | query heads 数 | 40 |
| $`K`$ | KV heads/groups 数 | MHA 40；GQA 8 |
| $`G`$ | 每个 KV head 服务的 query heads 数 | $`G=N/K=5`$ |
| $`H`$ | 每个 head 的向量长度 | 128 |
| $`L`$ | Transformer layers 数 | 40 |
| $`V`$ | vocabulary（词表）大小 | 32000 |

课程约定 $`D=NH`$、$`N=KG`$。字幕中的课堂提问指出图上 group/head 解释可能颠倒；老师在 [20:11](https://www.youtube.com/watch?v=EfM546A79aM&t=1211s) 确认：$`K`$ 是 KV groups，$`G`$ 是每组 query heads。

### 4.2 contraction 与 batch dimension

**Contraction dimension（收缩维）** 在两个输入中都出现、在输出中消失，表示沿该轴乘加求和。**Batch dimension（批维）** 在两个输入和输出里都保留，表示多份相互独立的计算。

矩阵例：

```math
X[B,D]@W[D,F]\to Y[B,F].
```

$`D`$ 被收缩。若 $`B=2,D=3,F=4`$，shape 是 $`[2,3]@[3,4]\to[2,4]`$；输出有 $`2\times4=8`$ 个数，每个数做长度 3 的 dot product。

视频 [10:03](https://www.youtube.com/watch?v=EfM546A79aM&t=603s) 解释红色 contraction；[10:28](https://www.youtube.com/watch?v=EfM546A79aM&t=628s) 解释保留的 batch 维。

### 4.3 Attention 每一步 shape

输入 activation：$`X[B,T,D]`$。本节统一使用：调用前 cache 长度 $`S_{old}`$；本次输入/query 长度 $`T`$；append 后总可见长度 $`S_{total}=S_{old}+T`$。

1. Query projection：$`Q[B,T,N,H]`$，因为 $`D=NH`$。
2. 历史 cache 各为 $`[B,S_{old},K,H]`$；新 K/V 各为 $`[B,T,K,H]`$；append 后 merged K/V 各为 $`[B,S_{total},K,H]`$。
3. 每个 query head 配到一个 KV group，score shape 为 $`[B,T,N,S_{total}]`$。
4. 对最后的 $`S_{total}`$ 轴做 softmax，再加权 V，输出 $`[B,T,N,H]`$。
5. 合并 $`N,H`$，回到 $`[B,T,D]`$。
6. 最终输出 projection 到 logits：$`[B,T,D]@[D,V]\to[B,T,V]`$。

例如一次 decode 的 $`B=2,S_{old}=4,T=1,S_{total}=5,N=4,K=2,G=2,H=3`$：

```text
Q      [2,1,4,3]
old KV [2,4,2,3]
new KV [2,1,2,3]
merged [2,5,2,3]
scores [2,1,4,5]
output [2,1,4,3] -> [2,1,12]
```

这也是对源码 line 228–230 扁平写法的教学性修复：资源公式可把 $`N\times H`$ 合成 $`D`$，但实现 shape 不能从 `[B,T,D]@[B,S,D]` 直接跳到五维，必须先拆 heads、让 K/V 在 head 组上匹配 query heads，再把 key 轴转到点积需要的位置。

视频 [11:33](https://www.youtube.com/watch?v=EfM546A79aM&t=693s) 从 activation 进入 attention；[11:50](https://www.youtube.com/watch?v=EfM546A79aM&t=710s) 逐一指出 Q/K/V；[13:11](https://www.youtube.com/watch?v=EfM546A79aM&t=791s) 给出 $`F\approx4D`$ 的课程简化；[13:28](https://www.youtube.com/watch?v=EfM546A79aM&t=808s) 给 $`D=NH`$；[13:51](https://www.youtube.com/watch?v=EfM546A79aM&t=831s) 区分 $`S,T`$。

---

## 5. Arithmetic intensity：从一次矩阵乘开始

### 5.1 先定义 FLOP、byte、BF16、HBM

- **FLOP（floating-point operation，浮点运算）**：一次浮点加或乘。一次 multiply-add 按 1 乘 + 1 加 = 2 FLOPs。
- **Byte（字节）**：8 bits。本文十进制 $`1\text{ GB}=10^9`$ bytes；二进制 $`1\text{ GiB}=2^{30}=1,073,741,824`$ bytes。
- **BF16（bfloat16）**：16-bit 浮点格式，每元素 2 bytes。
- **HBM**：GPU 上容量较大、带宽很高但比片上 SRAM/register 更远的显存。这里的 bytes 指模型数据与 HBM 之间的教学性读写量。
- **Bandwidth（带宽）**：每秒最多搬多少 bytes，例如 3.35 TB/s。
- **Arithmetic intensity（算术强度）**：

  $`I=\frac{\text{FLOPs}}{\text{bytes moved}},`$

  单位是 FLOP/byte。越高表示每搬一 byte 做越多计算。

视频 [14:23](https://www.youtube.com/watch?v=EfM546A79aM&t=863s) 回顾 arithmetic intensity；[14:56](https://www.youtube.com/watch?v=EfM546A79aM&t=896s) 开始逐项数矩阵乘。

### 5.2 $`[B,D]@[D,F]`$ 的 FLOPs

一个输出元素：

```math
Y_{b,f}=\sum_{d=1}^{D}X_{b,d}W_{d,f}.
```

长度 $`D`$ 的 dot product 约有 $`D`$ 次乘与 $`D`$ 次加，按 $`2D`$ FLOPs。输出有 $`B\times F`$ 个元素，所以：

```math
\boxed{\text{FLOPs}=2BDF}.
```

视频 [15:21](https://www.youtube.com/watch?v=EfM546A79aM&t=921s) 给出同一结果。

### 5.3 BF16 bytes 逐项数

假设 X、W、Y 各从 HBM 读/写一次：

| 动作 | 元素数 | BF16 bytes |
|---|---:|---:|
| 读 X | $`BD`$ | $`2BD`$ |
| 读 W | $`DF`$ | $`2DF`$ |
| 写 Y | $`BF`$ | $`2BF`$ |
| 合计 |  | $`2BD+2DF+2BF`$ |

所以精确教学式：

```math
I=\frac{2BDF}{2BD+2DF+2BF}
=\frac{BDF}{BD+DF+BF}.
```

视频 [15:08](https://www.youtube.com/watch?v=EfM546A79aM&t=908s) 明确 BF16 每元素 2 bytes；[15:59](https://www.youtube.com/watch?v=EfM546A79aM&t=959s) 定义 FLOPs/bytes。

### 5.4 为什么 $`B\ll D,F`$ 时 $`I\approx B`$

不能从上式直接跳答案。分子分母同时除以 $`DF`$：

```math
I
=\frac{B}{B/F+1+B/D}.
```

若 $`B/F`$ 和 $`B/D`$ 都很小，分母接近 $`0+1+0=1`$：

```math
I\approx B.
```

**小数例：**$`B=2,D=4,F=8`$：

```math
\text{FLOPs}=2(2)(4)(8)=128,
```

```math
\text{bytes}=2(2)(4)+2(4)(8)+2(2)(8)=16+64+32=112,
```

```math
I=128/112\approx1.143\text{ FLOP/byte}.
```

这里 $`D,F`$ 不够大，所以还没接近 $`B=2`$。视频 [16:23](https://www.youtube.com/watch?v=EfM546A79aM&t=983s) 用极限说明近似；本文的小例显示了近似条件不满足时的差距。

### 5.5 Roofline：295 从哪里来

**Compute-bound（计算受限）** 表示计算单元峰值先成为上限；**memory-bound（内存带宽受限）** 表示搬数据先成为上限。Roofline 教学上界：

```math
P_{actual}\le\min(P_{peak},\ I\times BW).
```

**Tensor Core（张量核心）** 是 NVIDIA GPU 中专门加速矩阵乘加的硬件单元；它只在受支持的 dtype、shape 和指令路径上达到相应峰值。课程采用 H100 SXM 的 dense BF16 Tensor Core 峰值约 $`989\times10^{12}`$ FLOP/s，以及 HBM 带宽 $`3.35\times10^{12}`$ byte/s：

```math
I_{roof}=\frac{989\times10^{12}}{3.35\times10^{12}}
=\frac{989}{3.35}
\approx295.22\text{ FLOP/byte}.
```

因此在这个特定理论模型中，$`I<295`$ 倾向 memory-bound，$`I>295`$ 才可能 compute-bound。视频 [17:15](https://www.youtube.com/watch?v=EfM546A79aM&t=1035s) 开始硬件 roof 比较；[17:50](https://www.youtube.com/watch?v=EfM546A79aM&t=1070s) 给出 H100 条件。

边界很重要：**structured sparsity（结构化稀疏）** 要求权重按硬件支持的固定模式出现零，例如每小组满足规定数量的非零；硬件才能跳过部分工作，它不同于随便把零散权重置零。[NVIDIA H100 官方规格](https://www.nvidia.com/en-sg/data-center/h100/)表中 BF16 `1,979 TFLOPS` 带 structured sparsity 星号；dense 口径约为一半，即课程使用的 989。实际 kernel 还受 shape、时钟、同步、cache 命中和软件效率影响，因此 295 是理论 roof point，不是实测承诺。

$`B=1`$ 且 $`D=F=4096`$ 时：

```math
I=\frac1{1/4096+1+1/4096}\approx0.9995,
```

远小于 295。视频 [18:03](https://www.youtube.com/watch?v=EfM546A79aM&t=1083s) 把这种 matrix-vector workload 连接到 decode。

---

## 6. MLP inference：为什么强度约是 $`BT`$

### 6.1 SwiGLU 的三个矩阵

**SwiGLU** 是现代 Transformer 常用的 gated MLP。课程为方便把非线性写成 GeLU，但资源主项相同：

```text
X [B,T,D]
 ├─ @ W_up   [D,F] -> U [B,T,F]
 ├─ @ W_gate [D,F] -> G [B,T,F]
 └─ activation(G) * U @ W_down [F,D] -> Y [B,T,D]
```

三个矩阵乘各做 $`2BTDF`$ FLOPs：

```math
2BTDF+2BTDF+2BTDF
=\boxed{6BTDF}.
```

视频 [25:27](https://www.youtube.com/watch?v=EfM546A79aM&t=1527s) 开始 MLP/attention 资源账；[26:07](https://www.youtube.com/watch?v=EfM546A79aM&t=1567s) 进入逐项计算。

### 6.2 HBM 流量必须分三本账

这里每个 BF16 元素取 2 bytes。三种账不能混在一起：

#### 账 A：课程源码的“单边中间量”计数

源码写：

| 数据 | 被源码计入的动作 | bytes |
|---|---|---:|
| X $`[B,T,D]`$ | 读一次 | $`2BTD`$ |
| 三个 weights | 读 $`W_{up},W_{gate},W_{down}`$ | $`3\times2DF=6DF`$ |
| U 与 G | 各写一次 | $`2\times2BTF=4BTF`$ |
| Y $`[B,T,D]`$ | 写一次 | $`2BTD`$ |
| 合计 |  | $`\boxed{4BTD+4BTF+6DF}`$ |

问题是：U/G 写到了 HBM，后面的 activation、乘法和 down projection 又要用它们；源码计了“写”，却漏了“再读”。因此这是**课程给出的单边教学账**，不能直接解释成完整自洽的物理 HBM 流量。

#### 账 B：完全融合，U/G 从不落 HBM

若 kernel 能让 U/G 在片上产生、消费，不写回 HBM，则理想下界为：

```math
\boxed{4BTD+6DF}.
```

这里 $`4BTD`$ 是 X 读 $`2BTD`$ 加 Y 写 $`2BTD`$；$`6DF`$ 是三份权重。U/G 没有 HBM 读写项。

#### 账 C：U/G 真落 HBM，随后还要读回来

U/G 各写一次共 $`4BTF`$ bytes，再各读一次又是 $`4BTF`$ bytes，所以至少：

```math
\boxed{4BTD+8BTF+6DF}.
```

若 up 与 gate 不能共享 X 的那次 HBM 读取，或中间乘积也落 HBM，流量还会更多。**Compiler（编译器）** 把高层代码变成硬件可执行的程序；**layout（数据布局）** 说明 tensor 元素在内存里的排列方式；**fusion（算子融合）** 把原本分开的操作合进更少的 kernel，以减少中间数据落回 HBM。它们连同 cache 决定实测值；不能从 Python 表达式臆测固定 kernel 数或固定 bytes。

视频 [26:53](https://www.youtube.com/watch?v=EfM546A79aM&t=1613s) 总结 FLOPs 依赖 $`B,T,D,F`$；[27:15](https://www.youtube.com/watch?v=EfM546A79aM&t=1635s) 再次用 FLOPs/bytes。

### 6.3 从精确式推到 $`BT`$

```math
I_{MLP,course}
=\frac{6BTDF}{4BTD+4BTF+6DF}.
```

分子分母同除 $`DF`$：

```math
I_{MLP,course}
=\frac{6BT}{4BT/F+4BT/D+6}.
```

若 $`BT\ll D,F`$，前两项接近 0：

```math
I_{MLP,course}\approx\frac{6BT}{6}=\boxed{BT}.
```

另外两种口径也有同一主导极限：完全融合的分母同除 $`DF`$ 后是 $`4BT/F+6`$；真落 HBM 的分母是 $`4BT/F+8BT/D+6`$。当 $`BT\ll D,F`$ 时，二者也都趋近 $`6BT/6=BT`$。所以“约 $`BT`$”主要来自**权重 $`6DF`$ 占主导**，不代表三个精确流量式相同。

视频 [27:24](https://www.youtube.com/watch?v=EfM546A79aM&t=1644s) 明确给出近似条件。

**小例：**$`B=1,T=2,D=2,F=4`$：

```math
\text{FLOPs}=6(1)(2)(2)(4)=96,
```

三本流量账分别是：

1. 课程单边账：$`4BTD+4BTF+6DF=16+32+48=96`$ bytes，$`I=96/96=1`$。
2. 完全融合账：$`4BTD+6DF=16+48=64`$ bytes，$`I=96/64=1.5`$。
3. U/G 真落 HBM 再读：$`4BTD+8BTF+6DF=16+64+48=128`$ bytes，$`I=96/128=0.75`$。

三者都不是近似值 $`BT=2`$；原因正是这个 tiny 例中 $`BT=2`$ 没有远小于 $`D=2`$。大模型 decode 时权重项通常更占主导，三种模型才都接近 $`BT`$。

### 6.4 prefill 与 decode

- Prefill：$`T`$ 是一段 prompt/query token 数，$`BT`$ 容易变大，MLP 较容易 compute-bound。视频 [28:12](https://www.youtube.com/watch?v=EfM546A79aM&t=1692s)。
- Decode：每个 request 每步 $`T=1`$，所以近似强度只剩 $`B`$。视频 [28:20](https://www.youtube.com/watch?v=EfM546A79aM&t=1700s)。
- Online 场景的 $`B`$ 是当时能一起执行的并发请求数，随到达/结束变化；视频 [28:40](https://www.youtube.com/watch?v=EfM546A79aM&t=1720s)。

这解释了 batching 为什么能摊薄 MLP weights：同一份 $`W_{up},W_{gate},W_{down}`$ 被 $`B`$ 个请求复用。

---

## 7. Attention inference：为什么 batching 救不了 KV 搬运

### 7.1 教学模型与 shape

**【课程内容】【源码 218–263】[29:18](https://www.youtube.com/watch?v=EfM546A79aM&t=1758s)**

这里假设使用类似 FlashAttention 的 fused 实现，不把完整 $`[B,T,N,S_{total}]`$ score matrix 写回 HBM。再次固定三个符号：

- $`S_{old}`$：调用前 cache 长度；
- $`T`$：本次输入/query 数；
- $`S_{total}=S_{old}+T`$：append 本次 K/V 后，attention 可见的 source 总数。

只数主矩阵乘：

1. 读 $`Q[B,T,D]`$：$`2BTD`$ bytes。
2. 读 merged $`K[B,S_{total},D]`$ 与 $`V[B,S_{total},D]`$：$`2BS_{total}D+2BS_{total}D=4BS_{total}D`$ bytes。
3. $`QK^T`$：$`2BTS_{total}D`$ FLOPs。
4. probabilities 乘 V：$`2BTS_{total}D`$ FLOPs。
5. 写输出 $`Y[B,T,D]`$：$`2BTD`$ bytes。

因此：

```math
\boxed{\text{FLOPs}=4BTS_{total}D},
```

```math
\boxed{\text{bytes}=4BS_{total}D+4BTD}.
```

视频 [29:25](https://www.youtube.com/watch?v=EfM546A79aM&t=1765s) 定义这里的 $`S,T`$；[29:40](https://www.youtube.com/watch?v=EfM546A79aM&t=1780s) 逐项走 Q/K/V。

### 7.2 强度 $`TS_{total}/(S_{total}+T)`$ 逐步约掉

```math
I_{attn}
=\frac{4BTS_{total}D}{4BS_{total}D+4BTD}.
```

分母提出 $`4BD`$：

```math
4BS_{total}D+4BTD=4BD(S_{total}+T).
```

所以：

```math
I_{attn}
=\frac{4BDTS_{total}}{4BD(S_{total}+T)}
=\boxed{\frac{TS_{total}}{S_{total}+T}}.
```

$`B,D`$ 被完全约掉，不是假装它们不存在，而是 FLOPs 与 KV bytes 都随 $`B,D`$ 同比例增加。视频 [30:21](https://www.youtube.com/watch?v=EfM546A79aM&t=1821s) 给出同一因子。

### 7.3 Tiny 数字例

取一次 decode：$`B=2,S_{old}=3,T=1`$，所以 $`S_{total}=3+1=4`$，$`D=8`$：

```math
\text{FLOPs}=4(2)(1)(4)(8)=256,
```

```math
\text{bytes}=4(2)(4)(8)+4(2)(1)(8)=256+64=320,
```

```math
I=256/320=0.8.
```

公式也给：

```math
\frac{TS_{total}}{S_{total}+T}=\frac{1\times4}{4+1}=0.8.
```

### 7.4 Prefill 与 decode 两个极端

**Prefill 简化：** 设 prompt 长度为 $`P`$。调用前没有 cache，所以 $`S_{old}=0`$；本次输入 $`T=P`$；append 后 $`S_{total}=P`$：

```math
I_{prefill}
=\frac{P\times P}{P+P}
=\frac{P^2}{2P}
=\boxed{P/2}.
```

若 $`P=1024`$，强度约 $`512`$ FLOP/byte，高于课程 H100 roof point 295，理论上可能 compute-bound。视频 [30:33](https://www.youtube.com/watch?v=EfM546A79aM&t=1833s) 推这一项。

**Decode：** 调用前 cache 为 $`S_{old}`$，本次输入一枚，故 $`T=1,S_{total}=S_{old}+1`$：

```math
I_{decode}=\frac{S_{total}}{S_{total}+1}<1.
```

若调用前 $`S_{old}=1023`$，append 后 $`S_{total}=1024`$：

```math
I=1024/1025\approx0.9990.
```

视频 [30:58](https://www.youtube.com/watch?v=EfM546A79aM&t=1858s) 转入 generation；[31:07](https://www.youtube.com/watch?v=EfM546A79aM&t=1867s) 给 $`S/(S+1)`$。这里把视频/源码的 $`S`$ 明确解释为本文的 append 后 $`S_{total}`$，避免与调用前 cache 混淆。

### 7.5 为什么提高 $`B`$ 不提高 attention 强度

MLP 的 weights 对所有序列相同，加载一次可服务更多请求。每个请求的 KV cache 却不同；把 $`B`$ 加倍，attention FLOPs 加倍，同时要读的 KV bytes 也加倍，比例不变。

视频 [31:46](https://www.youtube.com/watch?v=EfM546A79aM&t=1906s) 对比 MLP；[32:09](https://www.youtube.com/watch?v=EfM546A79aM&t=1929s) 说所有序列命中同一 MLP weights；[33:02](https://www.youtube.com/watch?v=EfM546A79aM&t=1982s) 解释 attention 中每序列有自己的 cache。

因此“attention 强度不靠 B 提高”只针对这份 per-sequence KV 教学账。跨请求共享 prefix、cache 命中、压缩 K/V、不同 attention kernel 仍能减少绝对 bytes。

### 7.6 结论不要写得过头

视频 [34:00](https://www.youtube.com/watch?v=EfM546A79aM&t=2040s) 总结 prefill 常偏 compute-bound、generation 常偏 memory-bound；[34:44](https://www.youtube.com/watch?v=EfM546A79aM&t=2084s) 停下来回答问题。

这是 roofline 的一阶分类，不表示：

- 所有 prefill 都 compute-bound；短 prompt、很小 batch 可能不是。
- 所有 decode 时间都只由 HBM 决定；kernel launch、通信、sampling、排队也会出现。
- memory-bound 是“GPU 什么都没做”；它是在等数据，不是零计算。

---

## 8. 参数量与 KV cache：一笔完整的 Llama 2 13B 账

### 8.1 参数公式每一项来自哪里

**【课程内容】【源码 264–331】【视频补充】[35:07](https://www.youtube.com/watch?v=EfM546A79aM&t=2107s)**

源码用：

```math
P
=2VD+3DFL+(2DNH+2DKH)L.
```

逐项解释：

| 项 | 模块 | 参数数 |
|---|---|---:|
| $`VD`$ | input embedding | $`VD`$ |
| $`VD`$ | output/unembedding | $`VD`$ |
| $`3DFL`$ | 每层 SwiGLU 的 up、gate、down 三矩阵 | $`3DFL`$ |
| $`2DNHL`$ | 每层 Q 与 output projection | $`2DNH\times L`$ |
| $`2DKHL`$ | 每层 K 与 V projection | $`2DKH\times L`$ |

假设 input/output embedding 不共享；若 weight tying，共享后少一个 $`VD`$。公式忽略 biases、norm 参数等小项，也不是任意架构的精确参数公式。

### 8.2 KV cache 公式

每个序列：

```math
M_{KV,seq}
=S_{total}\times K\times H\times L
\times2_{K,V}
\times2_{\text{BF16 bytes}}.
```

逐轴说：在一次调用完成后，cache 中每个可见 token（共 $`S_{total}`$ 个）、每个 KV head、每个 head 位置、每层都存一份 K 和一份 V；每个 BF16 数 2 bytes。Q 不存。源码变量名写 `S`，本文在 forward/caching 语境把它翻成无歧义的 $`S_{total}`$。

视频 [36:04](https://www.youtube.com/watch?v=EfM546A79aM&t=2164s) 指定 Llama 2 13B/H100 例；[36:16](https://www.youtube.com/watch?v=EfM546A79aM&t=2176s) 读模型维度；[37:49](https://www.youtube.com/watch?v=EfM546A79aM&t=2269s) 开始 KV cache 账。

### 8.3 Llama 2 13B 配置

课程配置：

```text
源码 S=1024（本文 S_total=1024）, D=5120, F=13824,
N=40, K=40, H=128, L=40, V=32000,
BF16=2 bytes, bandwidth=3.35e12 bytes/s
```

检查 $`D=NH=40\times128=5120`$，这是 MHA，因为 $`K=N=40`$。

### 8.4 参数量逐项手算

Embedding/output：

```math
2VD=2(32000)(5120)=327,680,000.
```

MLP：

```math
3DFL=3(5120)(13824)(40)=8,493,465,600.
```

Q 与 output projections：

```math
2DNHL=2(5120)(40)(128)(40)=2,097,152,000.
```

K/V projections：

```math
2DKHL=2(5120)(40)(128)(40)=2,097,152,000.
```

总和：

```math
P=327,680,000+8,493,465,600+2,097,152,000+2,097,152,000
=\boxed{13,015,449,600}.
```

即约 $`13.015`$B parameters。BF16 参数 bytes：

```math
2P=26,030,899,200\text{ bytes}=\boxed{26.031\text{ GB}}.
```

视频 [36:41](https://www.youtube.com/watch?v=EfM546A79aM&t=2201s) 进入统计函数；[37:16](https://www.youtube.com/watch?v=EfM546A79aM&t=2236s) 从 embeddings/MLP/QKV 数参数；[39:33](https://www.youtube.com/watch?v=EfM546A79aM&t=2373s) 报告约 13B。

### 8.5 每序列 KV cache 手算

```math
M_{KV,seq}
=1024\times40\times128\times40\times2\times2.
```

逐步：

```math
40\times128=5120,
```

```math
1024\times5120=5,242,880,
```

```math
5,242,880\times40=209,715,200,
```

```math
209,715,200\times2\times2
=\boxed{838,860,800\text{ bytes}}.
```

十进制：$`0.8388608`$ GB；二进制：

```math
838,860,800/1,073,741,824=\boxed{0.78125\text{ GiB}}.
```

视频 [39:44](https://www.youtube.com/watch?v=EfM546A79aM&t=2384s) 指出 memory 对 $`B`$ 呈线性。

### 8.6 总 memory、理想 ITL 下界与 throughput

课程的一阶模型：

```math
M(B)=2P+B\,M_{KV,seq},
```

```math
\text{step latency lower bound}=\frac{M(B)}{BW},
```

```math
\text{throughput upper bound}=\frac{B}{\text{step latency}}.
```

它把一次 decode step 近似为读取全部参数和相关全 cache；假设完美 overlap，忽略 kernel/通信/排队/allocator 与输出写入。它是**理想带宽下界/上界模型**，不是 H100 实测。

视频 [38:32](https://www.youtube.com/watch?v=EfM546A79aM&t=2312s) 用 memory/bandwidth 定 latency；[39:06](https://www.youtube.com/watch?v=EfM546A79aM&t=2346s) 用 $`B/latency`$ 定 throughput；[40:20](https://www.youtube.com/watch?v=EfM546A79aM&t=2420s) 解释随 $`B`$ 的形状。

| $`B`$ | 参数+KV memory | 理想 step latency | 理想 throughput |
|---:|---:|---:|---:|
| 1 | $`26.0308992+0.8388608=26.86976`$ GB | $`26.86976/3350=0.0080208`$ s = 8.021 ms | $`1/0.0080208=124.68`$ tok/s |
| 64 | $`26.0308992+64(0.8388608)=79.7179904`$ GB | 23.796 ms | $`64/0.023796=2689.48`$ tok/s |
| 256 | $`26.0308992+256(0.8388608)=240.779264`$ GB | 71.874 ms | $`256/0.071874=3561.77`$ tok/s |

这里的 `tok/s` 更精确地说是 **decode token-steps/s**：一个 decode step 让 $`B`$ 个活跃请求各前进一枚。它不是完整 `requests/s`；要把一条请求算“完成”，还必须知道每条请求要生成多少枚、何时遇到 `<eos>`。

80 GB H100 上，$`B=64`$ 仅按十进制裸数据勉强小于 80 GB，未留 allocator、workspace 与运行时余量；不是部署保证。$`B=256`$ 明确远超 80 GB。

---

## 9. Latency–throughput trade-off、复制与切分

### 9.1 为什么 $`B`$ 增大时二者方向相反

视频 [41:35](https://www.youtube.com/watch?v=EfM546A79aM&t=2495s) 开始代 batch；[41:40](https://www.youtube.com/watch?v=EfM546A79aM&t=2500s) 给 $`B=1`$；[41:55](https://www.youtube.com/watch?v=EfM546A79aM&t=2515s) 报约 124 tok/s。

在课程模型中：

- latency 随 $`B`$ 增大，因为每 step 要处理更多请求的 KV。
- throughput 增大，因为同一份参数读取摊给更多 token。
- throughput 最终趋于：

  $`\lim_{B\to\infty}\frac{B\cdot BW}{2P+BM_{KV}} =\frac{BW}{M_{KV}},`$

  不是无穷大。

视频 [42:02](https://www.youtube.com/watch?v=EfM546A79aM&t=2522s) 展示 latency 上升、throughput 上升；[42:42](https://www.youtube.com/watch?v=EfM546A79aM&t=2562s) 继续增到 256；[43:03](https://www.youtube.com/watch?v=EfM546A79aM&t=2583s) 说明更大显存也只是把上限推后。

### 9.2 排队与动态负载

课程公式没有 queue。在线系统中，总 TTFT 至少包含：

```math
\text{queue wait}+\text{prefill}+\text{first sampling/network}.
```

为了凑 $`B=64`$ 等 50 ms，虽然 kernel throughput 变高，每个早到请求却平白多等 50 ms。流量突发时还会出现 head-of-line blocking（短请求被长请求挡住）。所以应优化 SLO 下的 goodput，而不是只追最大 batch。

视频 [43:30](https://www.youtube.com/watch?v=EfM546A79aM&t=2610s) 用“等公交车”类比单请求等待；[44:17](https://www.youtube.com/watch?v=EfM546A79aM&t=2657s) 总结 trade-off。

### 9.3 Replication 与 sharding

- **Replication（复制）**：在 $`M`$ 张 GPU 各放一份完整模型，分流独立请求。理想情况下单请求 latency 不变，总 throughput 乘 $`M`$；显存也复制 $`M`$ 份。
- **Sharding（切分）**：一份模型跨多 GPU。单卡能放不下的模型可运行，或一条请求并行算；但每层可能通信，同步和网络会增加 latency。

视频 [44:30](https://www.youtube.com/watch?v=EfM546A79aM&t=2670s) 提到模型切分；[44:47](https://www.youtube.com/watch?v=EfM546A79aM&t=2687s) 给 $`M`$ copies 的理想例。

选择例：模型单卡装得下、在线请求很多且每请求追低延迟，优先考虑 replication；模型本身装不下时必须 shard。现实常两者组合。

### 9.4 Prefill 与 decode 不该强迫用同一 batch

Prefill 常 compute-heavy，TTFT 对排队敏感；decode 常 memory-heavy，适合把更多活跃请求拼在一起摊权重。视频 [45:01](https://www.youtube.com/watch?v=EfM546A79aM&t=2701s) 把 TTFT 连接到 prefill；[45:20](https://www.youtube.com/watch?v=EfM546A79aM&t=2720s) 建议 prefill 小 batch、generation 大 batch。

【延伸】生产系统还可能 prefill/decode disaggregation（分离部署），但会多出传 KV 的通信；是否值得取决于 TTFT/ITL SLO 与链路，不是固定答案。

---

## 10. MHA/MQA/GQA：压缩 head 轴

### 10.1 三个缩写和 $`N=KG`$

**【课程内容】【源码 371–407】【视频补充】[45:48](https://www.youtube.com/watch?v=EfM546A79aM&t=2748s)**

- **MHA（Multi-Head Attention，多头注意力）**：$`K=N`$，每个 query head 有自己的 K/V head。
- **MQA（Multi-Query Attention，多查询注意力）**：$`K=1`$，所有 query heads 共用一份 K/V。
- **GQA（Grouped-Query Attention，分组查询注意力）**：$`1<K<N`$；每个 KV head 服务 $`G=N/K`$ 个 query heads。

例：$`N=8,K=2`$，所以 $`G=8/2=4`$：

```text
KV head 0 <- query heads 0,1,2,3
KV head 1 <- query heads 4,5,6,7
```

Query 数仍是 8，head dimension 也没有变成 1。视频 [47:00](https://www.youtube.com/watch?v=EfM546A79aM&t=2820s) 复习 GQA；[47:36](https://www.youtube.com/watch?v=EfM546A79aM&t=2856s) 把 MHA/MQA 放在两端。

### 10.2 cache 缩减倍数

KV 公式中只有 $`K`$ 改变：

```math
M_{KV}\propto K.
```

MHA 的 $`K=N`$，GQA 的 $`K<N`$，所以缩减倍数：

```math
\frac{M_{MHA}}{M_{GQA}}=\frac NK.
```

$`N=40,K=8`$ 时，$`40/8=5`$ 倍。视频 [48:39](https://www.youtube.com/watch?v=EfM546A79aM&t=2919s) 提问原因；[49:00](https://www.youtube.com/watch?v=EfM546A79aM&t=2940s) 回到 Llama 例。

### 10.3 把课程公式重新代一遍

保持 Llama 配置其它量不变，只把 $`K:40\to8`$。

K/V projection parameters：

```math
2DKHL=2(5120)(8)(128)(40)=419,430,400.
```

总参数：

```math
327,680,000+8,493,465,600+2,097,152,000+419,430,400
=\boxed{11,337,728,000}.
```

BF16 参数 memory：$`22.675456`$ GB。

每序列 KV：

```math
1024\times8\times128\times40\times2\times2
=167,772,160\text{ bytes}
=0.16777216\text{ GB}.
```

| 配置 | memory | 理想 latency | 理想 throughput |
|---|---:|---:|---:|
| MHA $`K=40,B=64`$ | 79.7179904 GB | 23.796 ms | 2689.48 tok/s |
| GQA $`K=8,B=64`$ | $`22.675456+64(0.16777216)=33.41287424`$ GB | 9.974 ms | 6416.69 tok/s |
| GQA $`K=8,B=256`$ | 65.62512896 GB | 19.590 ms | 13068.16 tok/s |

视频 [49:19](https://www.youtube.com/watch?v=EfM546A79aM&t=2959s) 固定 $`B=64`$；[49:32](https://www.youtube.com/watch?v=EfM546A79aM&t=2972s) 改为 $`K=8`$；[49:44](https://www.youtube.com/watch?v=EfM546A79aM&t=2984s) 口头明确说 latency 与 throughput 都改善。

### 10.4 源码中的 “Worse latency” 是内部矛盾

源码约第 398 行在 `K=8,B=64` 后写：`Result: Worse latency, but better throughput`。但：

1. 同一函数定义 $`latency=memory/bandwidth`$。
2. $`K=8`$ 同时减少 K/V projection parameters 与 KV cache。
3. memory 从 79.718 GB 降到 33.413 GB。
4. latency 因而从 23.796 ms 降到 9.974 ms，是**更低/更好**。
5. 视频 [49:49](https://www.youtube.com/watch?v=EfM546A79aM&t=2989s) 也说这不是 latency/throughput 永远冲突；减 memory 会同时改善两者。

所以本文把该句标为 **source text inconsistency（源码文字不一致）**，不替它编造解释。后一个 `K=8,B=256` 相比 `K=8,B=64` latency 从 9.974 ms 升到 19.590 ms，才可说因 batch 增大而 worse；视频 [50:02](https://www.youtube.com/watch?v=EfM546A79aM&t=3002s) 开始联合调 batch，[50:15](https://www.youtube.com/watch?v=EfM546A79aM&t=3015s) 说明原先 256 不 fit、现在能 fit。

### 10.5 速度和 accuracy 都不是普遍定律

[GQA 原论文](https://arxiv.org/abs/2305.13245)的本地 `gqa-speed.png` 横轴是 groups、纵轴是 time per sample：K 较小时接近 MQA 的低时间，接近 64 groups 时靠近 MHA。`gqa-accuracy.png` 表中 GQA-8-XXL 的 average 47.1 接近 MHA-XXL 的 47.2，且 inference time 0.28 s 对 1.51 s；这是该论文模型/任务的结果。

课程视频 [50:52](https://www.youtube.com/watch?v=EfM546A79aM&t=3052s) 要求检查 accuracy；[51:14](https://www.youtube.com/watch?v=EfM546A79aM&t=3074s) 随即提醒不同研究可能得到不同 trade-off。不能写成“GQA 永不掉点”或“MQA 没人用”。

---

## 11. MLA 与 CLA：压缩 feature 轴和 layer 轴

### 11.1 MLA 的 latent cache

**MLA（Multi-head Latent Attention，多头潜在注意力）** 不直接缓存完整 K/V，而先把 hidden state 压成小 latent：

```math
c_t=W_{down}^{KV}h_t,
```

需要时再上投影：

```math
k_t=W_{up}^{K}c_t,
\qquad
v_t=W_{up}^{V}c_t.
```

**Latent（潜变量）** 是更短的中间向量。缓存 $`c_t`$ 而不是完整 $`k_t,v_t`$，压缩的是每 token 的 feature dimension。视频 [51:33](https://www.youtube.com/watch?v=EfM546A79aM&t=3093s) 转入 MLA；[51:45](https://www.youtube.com/watch?v=EfM546A79aM&t=3105s) 对比 MHA/GQA；[52:43](https://www.youtube.com/watch?v=EfM546A79aM&t=3163s) 说先投影到 $`C`$。

### 11.2 为什么可以在 query 侧“吸收”上投影

先忽略位置编码：

```math
q_s^Tk_t
=q_s^TW_{up}^{K}c_t
=\left((W_{up}^{K})^Tq_s\right)^Tc_t.
```

括号中的 transformed query 可在当前步算一次，然后直接与 cache 中的 $`c_t`$ 点积；无需为所有历史位置显式 materialize（生成并存下）完整 key。

### 11.3 RoPE 为什么需要解耦分量

**RoPE（Rotary Position Embedding，旋转位置编码）** 按位置给 Q/K 施加不同旋转。带 RoPE 的 score 类似：

```math
(R_s q_s)^T(R_tW_{up}^{K}c_t)
=q_s^TR_s^TR_tW_{up}^{K}c_t.
```

$`R_s^TR_t`$ 随历史位置 $`t`$ 改变，不能用一份与 $`t`$ 无关的固定 transformed query 覆盖所有历史位置。因此 MLA 把小块 rotary key 单独保留，其余 non-RoPE 内容走 latent cache；这叫 **decoupled RoPE（解耦旋转位置编码）**。视频 [53:14](https://www.youtube.com/watch?v=EfM546A79aM&t=3194s) 指出这个 wrinkle；[53:30](https://www.youtube.com/watch?v=EfM546A79aM&t=3210s) 仍强调总体大幅缩小。

### 11.4 Tiny cache 数字例

这是教学简化，不是 DeepSeek 完整实现。设完整每 token/layer 的 K 和 V 各 512 维；latent $`C=64`$，另存 rotary key 16 维；BF16，cache 完成后 $`S_{total}=100,L=2`$。

普通 cache：

```math
100\times2\times(512+512)\times2
=409,600\text{ bytes}.
```

MLA 简化 cache：

```math
100\times2\times(64+16)\times2
=32,000\text{ bytes}.
```

缩减：$`409,600/32,000=12.8`$ 倍。

课程列 DeepSeek-V2 的对比是完整维度 $`N H=16,384`$，压成 latent 512，再加 64 维 RoPE 路径，共 576；这个数字是该模型快照，不是所有 MLA 固定配置。[DeepSeek-V2/MLA 技术报告](https://arxiv.org/abs/2405.04434)

### 11.5 图表怎样读

本地 `mla-accuracy.png` 的 7B dense 表中，MHA 在 BBH/MMLU/C-Eval/CMMLU 都高于该表 GQA/MQA，说明压 KV heads 可能伤能力。`mla-accuracy2.png` 的 MoE 表中，MLA KV elements/token 从 110.6K 降至 15.6K（small）或 860.2K 降至 34.6K（large），多数列得分不降；只是论文设置的实证，不是 MLA 必然胜 MHA。

视频 [53:54](https://www.youtube.com/watch?v=EfM546A79aM&t=3234s) 开始 accuracy 检查；[54:07](https://www.youtube.com/watch?v=EfM546A79aM&t=3247s) 说这与 GQA 论文有张力；[54:23](https://www.youtube.com/watch?v=EfM546A79aM&t=3263s) 解读 MLA 表。

### 11.6 CLA 共享 layer 轴

**CLA（Cross-Layer Attention，跨层注意力）** 让相邻层共享 K/V；每层仍有自己的 Q 与 attention 计算。它压缩的是 layer 轴 $`L`$，不是 KV head 或 feature 轴。[CLA 原论文](https://arxiv.org/abs/2405.12981)

例：4 层、每 2 层共享一次：

```text
layer 1 产生 KV-A；layer 2 复用 KV-A
layer 3 产生 KV-B；layer 4 复用 KV-B
```

原来缓存 4 layer-caches，现在 2 个，理想缩小 2 倍。Layer 2/4 的 query 和 residual state 仍不同，因此不是把整层输出复制一遍。

视频 [55:02](https://www.youtube.com/watch?v=EfM546A79aM&t=3302s) 回答“直接缩 model width 呢”；[55:30](https://www.youtube.com/watch?v=EfM546A79aM&t=3330s) 引入 CLA；[55:43](https://www.youtube.com/watch?v=EfM546A79aM&t=3343s) 解释复用前层 KV；[56:15](https://www.youtube.com/watch?v=EfM546A79aM&t=3375s) 说论文改善 memory/accuracy Pareto frontier。

本地 `cla-results.png` 横轴是 BF16 KV bytes/token（log scale，越左越省），纵轴 validation perplexity（越低越好）。红色 CLA 点在多个相近 cache 预算下低于蓝色传统点，表示该实验中 trade-off 更好；不是所有规模都已证明。

---

## 12. Local/hybrid attention 与 DeepSeek V4：压缩 sequence 轴

### 12.1 Sliding window

**Local/sliding-window attention（局部/滑动窗口注意力）** 让当前 token 只看最近 $`W`$ 个 token。Full attention 每层 cache 随 $`S_{total}`$ 增长；local layer 只需保留最近 $`\min(S_{total},W)`$ 个位置。

本地 `longformer-attention.png` 依次画：full $`n^2`$、连续窗口、dilated window（间隔采样）与 global+window（少数全局 token 加局部带）。视频 [56:53](https://www.youtube.com/watch?v=EfM546A79aM&t=3413s) 开始 whirlwind tour；[57:01](https://www.youtube.com/watch?v=EfM546A79aM&t=3421s) 命名 sliding window。

例：$`S_{total}=1024,L=8`$，其中 2 层 full、6 层 local，$`W=128`$。忽略共同的 $`K,H,4`$ bytes 因子：

```math
\text{full-only layer-token entries}=8(1024)=8192,
```

```math
\text{hybrid entries}=2(1024)+6(128)=2048+768=2816.
```

缩减 $`8192/2816\approx2.91`$ 倍，不是 $`1024/128=8`$ 倍，因为 2 个 full layers 仍保存全历史。

### 12.2 有效感受野不是“无损长上下文”

多层 local attention 能逐层传播信息：粗略感受野可随 $`L\times W`$ 增大。视频 [57:29](https://www.youtube.com/watch?v=EfM546A79aM&t=3449s) 说 cache 与总 sequence length 脱钩；[57:51](https://www.youtube.com/watch?v=EfM546A79aM&t=3471s) 讲跨层传播。

但若要精确找 100 万 token 前的一串密码，本地窗口可能已丢掉它。视频 [58:30](https://www.youtube.com/watch?v=EfM546A79aM&t=3510s) 明确承认 accuracy 代价；[58:43](https://www.youtube.com/watch?v=EfM546A79aM&t=3523s) 给 hybrid：少量 full layers 保长程，其余 local layers 省成本。

视频课堂问答 [59:12](https://www.youtube.com/watch?v=EfM546A79aM&t=3552s) 比较 linear attention 与 window；[59:45](https://www.youtube.com/watch?v=EfM546A79aM&t=3585s) 说 recurrence 把历史压进固定状态；[60:04](https://www.youtube.com/watch?v=EfM546A79aM&t=3604s) 强调两类可混合；[60:54](https://www.youtube.com/watch?v=EfM546A79aM&t=3654s) 用 needle-in-a-haystack 说明固定压缩不可能保留所有细节。

### 12.3 DeepSeek V4 三个缩写的准确关系

以下是 2026-08-28 查询的官方技术报告边界，不凭课件缩写猜：

- **CSA（Compressed Sparse Attention，压缩稀疏注意力）**：先把每 $`m`$ 个 KV entries 压成 1 个，再在压缩 entries 上应用 DSA 式选择，并保留小 local window。
- **DSA（DeepSeek Sparse Attention，DeepSeek 稀疏注意力）**：lightning indexer 给候选打分，让 query 只看 top-$`k`$；它是 CSA 中的稀疏选择部件，不是和 CSA 完全并列的第三套 V4 layer。
- **HCA（Heavily Compressed Attention，重压缩注意力）**：以远大于 $`m`$ 的 $`m'`$ 把更多 token 合成一个 entry，再对这些高度压缩 entries 做 attention。
- V4 把 CSA 与 HCA layers 交错成 hybrid。

视频 [62:18](https://www.youtube.com/watch?v=EfM546A79aM&t=3738s) 开始 V4；[62:38](https://www.youtube.com/watch?v=EfM546A79aM&t=3758s) 列缩写；[63:01](https://www.youtube.com/watch?v=EfM546A79aM&t=3781s) 说每 $`m`$ 个压成一个；[63:06](https://www.youtube.com/watch?v=EfM546A79aM&t=3786s) 转入 top-$`k`$；[63:15](https://www.youtube.com/watch?v=EfM546A79aM&t=3795s) 描述 lightning indexer。

本地 `deepseek-v4-attention.png` 的数据流是：原始 hidden states→token-level compressor→compressed entries；另一路 compressor 产生 indexer keys，当前 query 产生 indexer query，轻量 MQA 得 scores，**top-$`k`$（只保留分数最高的 $`k`$ 项）** 选 compressed entries；最后把 selected compressed entries 与 sliding-window entries 拼给 shared-KV MQA。

[DeepSeek V4 官方模型卡/报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)声称：1M context 下 V4-Pro 的 single-token inference FLOPs 为 V3.2 的 27%，KV cache 为 10%。这是 DeepSeek 在指定模型/上下文下的报告值，不可推广成所有 CSA 模型。官方摘要准确说 hybrid 是 CSA+HCA；课件把 CSA/DSA/HCA 连续列出容易让人误以为三者同层级。

视频 [63:40](https://www.youtube.com/watch?v=EfM546A79aM&t=3820s) 因时间跳过细节；[63:44](https://www.youtube.com/watch?v=EfM546A79aM&t=3824s) 回到“少 KV memory”；[64:02](https://www.youtube.com/watch?v=EfM546A79aM&t=3842s) 总结可跨 head/layer/feature/sequence 压缩；[64:20](https://www.youtube.com/watch?v=EfM546A79aM&t=3860s) 提到非 autoregressive 架构方向。

---

## 13. Quantization：把每个数存得更小

### 13.1 量化公式：round、zero point、clamp

**Quantization（量化）** 把高精度实数映射到较少离散整数/浮点档位。对常见 affine integer quantization：

格式名先认清：**FP32** 是 32-bit floating-point（浮点）格式，通常每元素 4 bytes；**FP8** 是多种 8-bit 浮点格式的家族；**INT8/INT4** 是 8-bit/4-bit integer（整数）格式，分别只给每个数 256/16 个离散编码档位。

```math
q=\mathrm{clamp}\left(\mathrm{round}(x/s)+z, q_{min},q_{max}\right),
```

```math
\hat x=s(q-z).
```

- $`x`$：原实数。
- $`s>0`$：**scale（缩放因子）**，一个整数档代表多少实数。
- $`z`$：**zero point（零点，代码里也常缩写 `zp`）**，让实数 0 对应整数 $`z`$。
- `round`：四舍五入到整数。
- **clamp（截断）**：小于 $`q_{min}`$ 的改成最小值，大于 $`q_{max}`$ 的改成最大值。
- $`\hat x`$：反量化近似值。

视频 [64:29](https://www.youtube.com/watch?v=EfM546A79aM&t=3869s) 转入量化；[64:44](https://www.youtube.com/watch?v=EfM546A79aM&t=3884s) 给核心直觉。

课程代码例 $`x=5.2342,s=0.1,z=4`$：

```math
x/s=5.2342/0.1=52.342,
```

```math
\mathrm{round}(52.342)=52,
```

```math
q=52+4=56,
```

```math
\hat x=(56-4)(0.1)=52(0.1)=5.2.
```

误差 $`\hat x-x=5.2-5.2342=-0.0342`$，绝对误差 0.0342。若 q-range 是 INT4 $`[-8,7]`$，56 会 clamp 成 7，误差巨大；range 不能漏。

### 13.2 symmetric/asymmetric 与 scale 粒度

- **Symmetric（对称）**：常取 $`z=0`$，正负范围近似对称；逻辑简单。
- **Asymmetric（非对称）**：$`z`$ 可非零，适合数据范围明显偏一侧。
- **Per-tensor**：整张 tensor 共用一个 $`s,z`$；metadata 少，但 **outlier（离群值）**——绝对值远大于同组多数元素的少数值——会把共享量化范围拉宽，使普通值可用的刻度变粗。
- **Per-channel**：每个输出/输入 channel 一组 scale；更准，scale 更多。
- **Per-group**：每连续一小组 weights 共用 scale；在精度、metadata、kernel 之间折中。

**Calibration（校准）** 是在代表性样本上观察 weight/activation range，决定 scale、zero point 或 clipping threshold；样本不代表真实流量时，量化误差也会偏。

### 13.3 Weight-only 与 W+A

- **Weight-only quantization**：只压 weights，例如 W4A16 表示 4-bit weights、16-bit activations。减少参数带宽，但 activation/KV 仍大。
- **Weight-and-activation quantization**：weights 与 activations 都压，例如 W8A8。潜在收益更大，也更依赖受支持的 matrix kernel 与数值稳定性。

量化不等于必然加速：若硬件没有该 dtype 的快 kernel、需要频繁 dequant、group 太小导致 metadata 多，或 shape 不匹配，wall-clock 可能不降。视频 [65:05](https://www.youtube.com/watch?v=EfM546A79aM&t=3905s) 展示 BF16 到 INT4 选择；课程口头也同时提醒 accuracy。

### 13.4 FP8 范围的源码边界

源码说“FP32 训练需要、BF16 默认推理、INT8 只用于推理”只是课程的粗略教学对比，不是普遍限制：现代训练常让 BF16/FP8 参与 forward/backward，同时为敏感状态保留更高精度；也存在量化训练研究。究竟能否训练或推理，要看算法、硬件和 kernel，不能只看位数。

源码写 `fp8 e4m3 [-240,240] on H100s`。但 [NVIDIA Transformer Engine 官方文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) 对其 H100 E4M3 明确写最大有限值 $`\pm448`$，E5M2 为 $`\pm57344`$ 且支持 inf。

因此本文不把 `[-240,240]` 当通用 E4M3 范围；FP8 的 E4M3/E5M2 以及 FN/FNUZ 等具体变体有不同特殊值和范围，必须看硬件/框架的确切 dtype。课程数字在这里至少与 NVIDIA 当前官方实现口径不一致。

### 13.5 QAT、PTQ、GPTQ、AWQ

- **QAT（Quantization-Aware Training，量化感知训练）**：训练 forward 中模拟 quantize→dequantize，让 weights 适应误差。Round 不可导，实践常用 straight-through estimator 近似传 gradient。优点是可适应；缺点是要训练。视频 [65:21](https://www.youtube.com/watch?v=EfM546A79aM&t=3921s)。
- **PTQ（Post-Training Quantization，训练后量化）**：模型训练完才校准和量化，便宜。视频 [65:45](https://www.youtube.com/watch?v=EfM546A79aM&t=3945s)；[65:53](https://www.youtube.com/watch?v=EfM546A79aM&t=3953s) 正式命名。
- **Hessian（海森矩阵）**：loss 对 weights 的二阶偏导矩阵；人话是不同方向的曲率，告诉某个量化误差有多敏感。
- **GPTQ**：一种 weight-only PTQ，利用近似 Hessian，在逐列/逐块量化时补偿还未量化 weights，以减小 layer output 重建误差。[GPTQ 原论文](https://arxiv.org/abs/2210.17323)。视频 [66:18](https://www.youtube.com/watch?v=EfM546A79aM&t=3978s)。
- **AWQ（Activation-aware Weight Quantization，激活感知权重量化）**：用 calibration activations 找 salient channels，再通过 per-channel scaling 保护对应 weights。[AWQ 原论文](https://arxiv.org/abs/2306.00978)。视频 [66:40](https://www.youtube.com/watch?v=EfM546A79aM&t=4000s)。

本地 `awq-schema.png` 有三个 panel：朴素 RTN INT3 perplexity 43.2；把约 1% salient weights 留 FP16 可降至 13.0，但图明确标“bad hardware efficiency”；AWQ 最终方案是先按 activation magnitude 缩放 weights，再统一 INT3，仍得 13.0。源码文字“keep 0.1–1% high precision”只描述动机/对照，不是 AWQ 最终硬件友好方案。视频 [66:59](https://www.youtube.com/watch?v=EfM546A79aM&t=4019s) 看图；[67:13](https://www.youtube.com/watch?v=EfM546A79aM&t=4033s) 讲 salient channels。

---

## 14. Pruning 与 distillation：删掉，再修复

### 14.1 三个动作

**Pruning（剪枝）** 删除模型的一部分。典型流程：

1. **Importance estimation（重要性估计）**：用 calibration set 观察删除某 layer/head/channel 后的影响。
2. Remove/trim：真正移除低重要性结构。
3. **Repair（修复）**：继续训练或 distill，恢复损失的能力。

视频 [67:39](https://www.youtube.com/watch?v=EfM546A79aM&t=4059s) 转入 pruning；[67:45](https://www.youtube.com/watch?v=EfM546A79aM&t=4065s) 用“rip out and fix”概括；[68:05](https://www.youtube.com/watch?v=EfM546A79aM&t=4085s) 开始 importance。

### 14.2 Structured 与 unstructured

- **Structured pruning（结构化剪枝）**：删整层、整 head、整 hidden channel，得到真正更小的 dense shapes；普通 GEMM（General Matrix Multiplication，通用稠密矩阵乘）kernel 更容易利用。
- **Unstructured pruning（非结构化剪枝）**：把零散单个 weight 置零。参数数学上稀疏，但若仍用 dense storage/kernel，硬件照读照算，未必更快；需要受支持的稀疏格式/kernel。

因此“参数少”不等于“实际 latency 按比例少”。视频 [68:14](https://www.youtube.com/watch?v=EfM546A79aM&t=4094s) 说可删 hidden units 甚至 layers。

### 14.3 Calibration 的反例

若 calibration 1024 个样本全是英文，某个只在代码中活跃的 head 看起来“不重要”，删后代码能力会崩。Magnitude 大也不自动等于重要：一个 activation 永远是常数 100，可能可折进 bias，也可能下游依赖它；要做 ablation 与验证。

视频课堂问答 [69:54](https://www.youtube.com/watch?v=EfM546A79aM&t=4194s) 问如何判重要；[70:05](https://www.youtube.com/watch?v=EfM546A79aM&t=4205s) 回答用 calibration activations；[70:30](https://www.youtube.com/watch?v=EfM546A79aM&t=4230s) 追问“一直很大就一定有意义吗”；[70:43](https://www.youtube.com/watch?v=EfM546A79aM&t=4243s) 承认这是经验假设；[71:03](https://www.youtube.com/watch?v=EfM546A79aM&t=4263s) 讨论高均值低方差。

### 14.4 Distillation 是什么

**Distillation（知识蒸馏）** 让小 **student（学生模型）** 模仿大 **teacher（教师模型）** 的 output probabilities、logits 或中间 features，而不只学习 **one-hot 标签**。One-hot 是“正确类别位置写1，其余全写0”的向量，例如三分类第2类是 `[0,1,0]`；teacher 的软概率还能告诉 student 其它类别有多相似。

本地 `pruning-kd-loop.png` 画出：trained LLM→估 importance→rank→trim→distillation，并可迭代。本地 `pruning-kd.png` 横轴是训练成本（trillion tokens），纵轴 MMLU；图宣称 Minitron 4B/8B 从 15B pruning start 出发，以远少于从零训练的 token 成本接近相似规模模型。这是 [NVIDIA Minitron 论文](https://arxiv.org/abs/2407.14679) 的实验快照。

视频 [68:29](https://www.youtube.com/watch?v=EfM546A79aM&t=4109s) 说删后继续训练 heal；[68:55](https://www.youtube.com/watch?v=EfM546A79aM&t=4135s) 给 15B→8B 案例；[69:17](https://www.youtube.com/watch?v=EfM546A79aM&t=4157s) 总结目标；[69:31](https://www.youtube.com/watch?v=EfM546A79aM&t=4171s) 对比从零 recipe；[69:39](https://www.youtube.com/watch?v=EfM546A79aM&t=4179s) 说可从原模型初始化；[69:47](https://www.youtube.com/watch?v=EfM546A79aM&t=4187s) 用 distillation 修复。

源码 `model_pruning()` 最后保留一个“待继续实现”的占位注释，说明代码讲义本身没有展开剪枝公式/实现；本文的 structured/unstructured 与 teacher/student 解释属于补充，不冒充源码已有课程内容。

---

## 15. Speculative sampling：让小模型先猜，大模型一次验一串

### 15.1 四个角色先认清

**【课程内容】【源码 507–552】【视频补充】[71:43](https://www.youtube.com/watch?v=EfM546A79aM&t=4303s)**

- **Proposal model（提议模型，也叫 draft model）**：较便宜，先提出候选 token；其概率分布记为 $`p`$。
- **Target model（目标模型）**：真正想从中采样的大模型；分布记为 $`q`$。
- **Rejection sampling（拒绝采样）**：候选太偏向 $`p`$ 时，随机拒绝一部分。
- **Residual distribution（残差分布）**：拒绝以后，从“大模型还欠下的概率质量”里补采样。

这里的“概率质量”只是所有概率加起来的份额。例如 $`q(A)=0.4`$，就表示 A 拿到总份额的 40%。视频 [71:58](https://www.youtube.com/watch?v=EfM546A79aM&t=4318s) 先给出“small model propose, large model verify”的直觉；[72:02](https://www.youtube.com/watch?v=EfM546A79aM&t=4322s) 强调目标不是近似大模型分布，而是在实现正确时**保持同一分布**。

### 15.2 一枚候选怎样验收

提议模型先按 $`p(x)`$ 抽到 token $`x`$。目标模型接受它的概率是：

```math
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right).
```

逐符号解释：

- $`p(x)`$：小模型给 $`x`$ 的概率；
- $`q(x)`$：大模型给 $`x`$ 的概率；
- $`q(x)/p(x)`$：大模型相对小模型有多喜欢 $`x`$；
- $`\min(1,r)`$：在 $`1`$ 和 $`r`$ 中取较小者，保证接受概率不超过 1。

若拒绝，不是直接改为从 $`q`$ 重抽，而是从

```math
r(x)\propto\max(q(x)-p(x),0)
```

抽。符号 $`\propto`$ 读作“成比例”：先算右侧，再除以所有 token 的右侧之和，使结果重新加到 1。视频 [72:16](https://www.youtube.com/watch?v=EfM546A79aM&t=4336s) 展开接受式；[72:30](https://www.youtube.com/watch?v=EfM546A79aM&t=4350s) 解释拒绝后的 corrected distribution。

边界 $`p=q`$ 要单独说：对任何实际能从 proposal 抽到的 token，必有 $`p(x)>0`$，此时接受率都是 $`\min(1,q/p)=1`$，拒绝概率为 0；$`p(x)=q(x)=0`$ 的 token 根本不会被 proposal 抽到。残差向量 $`\max(q-p,0)`$ 全为 0。实现应直接走“全部接受”，**不需要也不应该把全零残差除以 0**。

### 15.3 两词完整事件树：为什么最终恰好等于 $`q`$

**【补充理解/例子】** 设只有 A、B：

```math
p=[0.7,0.3],\qquad q=[0.4,0.6].
```

**路径 1：proposal 抽到 A。** 概率为 $`0.7`$。

接受 A 的概率：

```math
\min(1,0.4/0.7)=4/7.
```

因此最终经“抽 A 且接受”输出 A 的概率：

```math
0.7\times(4/7)=0.4.
```

抽 A 后被拒绝的概率：

```math
0.7\times(1-4/7)=0.7\times3/7=0.3.
```

**路径 2：proposal 抽到 B。** 概率为 $`0.3`$。因为 $`q(B)/p(B)=0.6/0.3=2`$，所以接受率为 $`\min(1,2)=1`$，最终直接输出 B 的概率是 $`0.3`$。

拒绝路径的残差先算：

```math
\max(q-p,0)=[\max(0.4-0.7,0),\max(0.6-0.3,0)]=[0,0.3].
```

归一化后为 $`[0,1]`$，所以被拒绝的那 $`0.3`$ 全部补给 B。最终：

```math
P(A)=0.4,\qquad P(B)=0.3+0.3=0.6=q(B).
```

视频 [72:40](https://www.youtube.com/watch?v=EfM546A79aM&t=4360s) 开始说明 exactness；[73:01](https://www.youtube.com/watch?v=EfM546A79aM&t=4381s) 把接受/拒绝连接到“保留 target distribution”。

### 15.4 三词例：不是二词巧合

设

```math
p=[0.5,0.3,0.2],\qquad q=[0.2,0.5,0.3].
```

每个词经“proposal 抽到且接受”留下的概率就是 $`\min(p_i,q_i)`$：

| token | $`p_i`$ | $`q_i`$ | 被直接接受的总概率 $`\min(p_i,q_i)`$ |
|---|---:|---:|---:|
| A | 0.5 | 0.2 | 0.2 |
| B | 0.3 | 0.5 | 0.3 |
| C | 0.2 | 0.3 | 0.2 |

直接接受共 $`0.2+0.3+0.2=0.7`$，所以拒绝总概率为 $`1-0.7=0.3`$。残差原始质量：

```math
\max(q-p,0)=[0,0.2,0.1],
```

它们也加到 $`0.3`$。除以 $`0.3`$ 后，残差分布为 $`[0,2/3,1/3]`$。拒绝路径贡献：

```math
0.3[0,2/3,1/3]=[0,0.2,0.1].
```

加回直接接受质量：

```math
[0.2,0.3,0.2]+[0,0.2,0.1]=[0.2,0.5,0.3]=q.
```

这就是“一枚候选”的一般证明：直接接受给出 $`\min(p,q)`$，残差正好补上 $`q`$ 比 $`p`$ 少给的部分。

### 15.5 一次猜 $`K`$ 枚，为什么可能更快

**【课程内容】【视频补充】[73:17](https://www.youtube.com/watch?v=EfM546A79aM&t=4397s)**

小模型连续猜 $`K`$ 枚。目标模型可以把这 $`K`$ 个位置作为一次 prefill-like 批量验证，因为候选整串已经可见。遇到第一枚拒绝时，后面的候选不再有效；接受得越长，一次昂贵验证换回的 token 越多。[73:52](https://www.youtube.com/watch?v=EfM546A79aM&t=4432s) 展示源码引用的算法图；[74:04](https://www.youtube.com/watch?v=EfM546A79aM&t=4444s) 说明并行验证利用 GPU 的原因。

若每个位置独立地以概率 $`a`$ 接受，只数被接受的 draft tokens，期望长度为：

```math
E[A]=a+a^2+\cdots+a^K
=\frac{a(1-a^K)}{1-a}\quad(a\ne1).
```

为什么第一项是 $`a`$？至少接受 1 枚的概率是 $`a`$。至少接受 2 枚要前两枚都接受，概率 $`a^2`$。依次相加就是期望接受数。

一次验证通常还会产出拒绝位置的 corrected token，或在全接受时额外产出 target token，所以产出数的简化期望为：

```math
E[Y]=1+a+a^2+\cdots+a^K
=\frac{1-a^{K+1}}{1-a}.
```

这是假设每步接受率相同且彼此独立的教学近似；真实接受率随上下文和位置变化。

### 15.6 Break-even 手算

设：

- 普通 target decode 1 枚的成本为 1 个单位；
- 一次批量验证 $`K=4`$ 枚的 target 成本为 1.4；
- draft 每猜一枚成本 0.08，所以四枚共 $`4\times0.08=0.32`$。

总 speculative 成本：

```math
1.4+0.32=1.72.
```

若 $`a=0.8`$：

```math
E[Y]=1+0.8+0.8^2+0.8^3+0.8^4
=1+0.8+0.64+0.512+0.4096=3.3616.
```

普通方法生成 3.3616 枚平均要 3.3616 个成本单位；现在用 1.72，理想加速比约：

```math
3.3616/1.72\approx1.95.
```

若 $`a=0.3`$：

```math
E[Y]=1+0.3+0.09+0.027+0.0081=1.4251<1.72,
```

反而亏。更一般的 break-even 条件是：

```math
Kc_d+c_q(K)<E[Y]c_q(1),
```

其中 $`c_d`$ 是每枚 draft 成本，$`c_q(K)`$ 是 target 批量验证成本。视频 [74:14](https://www.youtube.com/watch?v=EfM546A79aM&t=4454s) 强调 draft overhead；[74:20](https://www.youtube.com/watch?v=EfM546A79aM&t=4460s) 讨论 acceptance rate；[74:31](https://www.youtube.com/watch?v=EfM546A79aM&t=4471s) 展示实测取决于模型与任务。

### 15.7 Exact 的条件与不 exact 的边界

**Temperature（温度）** 把 logits 除以一个正数再做 softmax：温度低通常更尖锐，温度高通常更平；**top-$`k`$ sampling**只保留概率最高的 $`k`$ 项再归一化；**top-$`p`$ sampling**保留累计概率刚达到阈值 $`p`$ 的最小候选集合再归一化。

**正确实现**需要：proposal/target 针对同一上下文；目标概率与实际 target 采样设置一致；temperature、top-k、top-p 等概率变换在验收式中一致处理；随机数和 residual normalization 没有 bug。若把 $`\min(1,q/p)`$ 省掉、拒绝后直接从错误分布抽、或 target 验证时用了另一套采样设置，结果就不再保证为 $`q`$。

“greedy decode”每步取最大概率 token，不是在随机分布上采样；它可以做 speculative greedy verification，但上面的事件树 exactness 证明针对随机 sampling。视频 [74:36](https://www.youtube.com/watch?v=EfM546A79aM&t=4476s) 区分 lossless/exact 与近似方法。

### 15.8 Medusa 与 EAGLE：只学课程要求的层级

**【课程内容】【源码图 `medusa-eagle.png`】**

- Medusa 在 target 模型上加多个预测 heads，提出多步候选；不一定另放完整小模型。
- EAGLE 在 feature/hidden-state 层面预测候选，再由 target 验证。

共同点是“便宜地造候选，昂贵模型批量验”；具体训练目标、树状候选和实现版本会变，本讲不把它们简化成同一个算法。视频 [74:57](https://www.youtube.com/watch?v=EfM546A79aM&t=4497s) 引出 multi-token prediction；[75:03](https://www.youtube.com/watch?v=EfM546A79aM&t=4503s) 介绍 Medusa；[75:17](https://www.youtube.com/watch?v=EfM546A79aM&t=4517s) 转向 EAGLE；[75:40](https://www.youtube.com/watch?v=EfM546A79aM&t=4540s) 总结 draft quality 与成本的权衡。

---

## 16. Continuous batching：请求随到随进、随完随退

### 16.1 Static batch 为什么浪费

**【课程内容】【源码 553–575】【视频补充】[76:55](https://www.youtube.com/watch?v=EfM546A79aM&t=4615s)**

静态批处理把一批请求一起开始，并等最慢者结束。若 A 只生成 2 枚、B 要 6 枚，A 完成后仍占着一个空槽。**Continuous batching（连续批处理）** 在每个 decode iteration 边界重新排队：完成的请求立即退出，新请求可立即填空。

这里 **iteration-level scheduling（迭代级调度）** 指“每生成一轮 token，就重新决定这一轮哪些请求上 GPU”，不是等整批全部结束才调度。视频 [77:01](https://www.youtube.com/watch?v=EfM546A79aM&t=4621s) 从不同完成长度说明浪费；[77:09](https://www.youtube.com/watch?v=EfM546A79aM&t=4629s) 给出 continuous batching。

### 16.2 到达—完成时间表

假设每轮每个活跃请求生成 1 枚：

| 轮次开始 | 新到达 | 活跃请求 | 本轮后发生什么 |
|---:|---|---|---|
| 1 | A（需2枚）、B（需4枚） | A,B | A剩1，B剩3 |
| 2 | C（需2枚） | A,B,C | A完成；B剩2，C剩1 |
| 3 | D（需1枚） | B,C,D | C、D完成；B剩1 |
| 4 | 无 | B | B完成 |

静态 batch 若必须等 A、B 都结束，C 最早第 5 轮才进；连续批处理让 C 第 2 轮就进。视频 [77:19](https://www.youtube.com/watch?v=EfM546A79aM&t=4639s) 画出请求长度不齐；[77:30](https://www.youtube.com/watch?v=EfM546A79aM&t=4650s) 说明完成请求释放容量。

### 16.3 Ragged selective batching

**Ragged batch（参差批）** 表示不同请求的有效序列长度不同，不能简单堆成一个完整长方体而不产生 **padding（填充）**；padding 是为了凑齐 shape 人为加入的无效占位 token，若 kernel 不跳过它们就会浪费计算。**Selective batching（选择性拼批）** 是：

- attention 需要各自长度、位置与 KV block table，通常按请求边界处理或由 ragged attention kernel 处理；
- LayerNorm、MLP 等非 attention 操作只关心每个 token 的 hidden 向量，可以把所有有效 token 拼在一起。

设三个请求本轮参加计算的有效 token 数分别为 3、9、5，hidden width 为 $`D`$；这里 $`D`$ 是整个模型隐藏向量宽度，不能和 §4 的每个 head 宽度 $`H`$ 混用：

```text
A activation: [3, D]
B activation: [9, D]
C activation: [5, D]

非 attention 拼接：[(3+9+5), D] = [17, D]
attention：仍需 lengths=[3,9,5] 与各自 KV 地址
```

这样大矩阵乘看到 17 行，硬件利用率较好；attention 又不会让 A 错看 B 的 token。视频 [77:39](https://www.youtube.com/watch?v=EfM546A79aM&t=4659s) 提到 selective batching；[77:45](https://www.youtube.com/watch?v=EfM546A79aM&t=4665s) 分开 attention 与 non-attention；[77:53](https://www.youtube.com/watch?v=EfM546A79aM&t=4673s) 解释不同 sequence lengths。

### 16.4 调度不是“只把 batch 塞满”

调度器还要检查：KV cache 是否放得下、某请求是否快超过 SLO、prefill 会不会阻塞 decode、租户优先级、公平性与取消请求。batch 越大吞吐可能越高，但单请求排队时间也可能越长，所以优化目标应是 goodput，不只是 tokens/s。视频 [77:58](https://www.youtube.com/watch?v=EfM546A79aM&t=4678s) 讨论 GPU utilization；[78:08](https://www.youtube.com/watch?v=EfM546A79aM&t=4688s) 连接 latency constraint；[78:13](https://www.youtube.com/watch?v=EfM546A79aM&t=4693s) 强调动态 workload。

**【延伸】** Orca 论文把 iteration-level scheduling 与 selective batching 系统化；具体 serving engine 的策略、抢占和 chunked prefill 属于实现选择，并非源码这几行就保证。视频 [78:22](https://www.youtube.com/watch?v=EfM546A79aM&t=4702s) 开始从调度转向内存管理。

---

## 17. PagedAttention：把 KV cache 切成可搬动的固定小块

### 17.1 两种 fragmentation

**【课程内容】【源码 576–611】【视频补充】[79:34](https://www.youtube.com/watch?v=EfM546A79aM&t=4774s)**

**Fragmentation（碎片）** 指总空闲空间看似够，却因分配粒度或位置而浪费。

1. **Internal fragmentation（内部碎片）**：给一项任务保留的块内部没用满。例：预留 16 个 token 槽，只用 6 个，浪费 $`16-6=10`$ 个，浪费率 $`10/16=62.5\%`$。
2. **External fragmentation（外部碎片）**：空闲块散开。例：两处空洞分别有 3 和 4 个槽，总共 $`3+4=7`$；若旧分配器要求连续 5 槽，任何一个洞都放不下。

源码图 `paged-attention-fragmentation.png` 正在画这两种浪费。视频 [79:38](https://www.youtube.com/watch?v=EfM546A79aM&t=4778s) 定义预分配问题；[79:49](https://www.youtube.com/watch?v=EfM546A79aM&t=4789s) 说明请求长度事前未知；[80:03](https://www.youtube.com/watch?v=EfM546A79aM&t=4803s) 对比 internal 与 external waste。

### 17.2 Paging、logical block、physical block

**Paging（分页）** 把逻辑上连续的 KV cache 切成等长小块；物理上这些块可以散落。**Logical block（逻辑块）** 表示序列眼中的第 0、1、2 块；**physical block（物理块）** 表示 HBM 里的真实槽位。**Block table（块表）** 保存逻辑块到物理块的映射。

块大小为 4 tokens，一条 10-token 序列需要：

```math
\lceil10/4\rceil=3\text{ blocks}.
```

$`\lceil x\rceil`$ 表示“向上取整”：$`10/4=2.5`$，但不能分配半块，所以取 3。前两块各用 4 槽，最后一块用 2 槽、空 2 槽。设块表：

| logical block | 覆盖 token | physical block |
|---:|---|---:|
| 0 | 0–3 | 7 |
| 1 | 4–7 | 1 |
| 2 | 8–9 | 5 |

物理顺序 7、1、5 并不连续，但 attention kernel 查块表后仍按逻辑 0、1、2 读。视频 [80:19](https://www.youtube.com/watch?v=EfM546A79aM&t=4819s) 引出 operating-system paging 类比；[80:26](https://www.youtube.com/watch?v=EfM546A79aM&t=4826s) 展示 blocks；[80:39](https://www.youtube.com/watch?v=EfM546A79aM&t=4839s) 解释 block table。

### 17.3 每步 append 怎样定位

若 block size 为 $`P=4`$，准备写第 $`t`$ 个 token（从 0 开始）：

```math
\text{logical block}=\lfloor t/P\rfloor,
\qquad
\text{offset}=t\bmod P.
```

$`\lfloor x\rfloor`$ 是向下取整；$`a\bmod b`$ 是 $`a`$ 除以 $`b`$ 的余数。写 $`t=9`$：

```math
\lfloor9/4\rfloor=2,\qquad9\bmod4=1.
```

先查逻辑块 2 映到物理块 5，再写该块内 offset 1。视频 [80:41](https://www.youtube.com/watch?v=EfM546A79aM&t=4841s) 展示 logical view；[80:52](https://www.youtube.com/watch?v=EfM546A79aM&t=4852s) 对照 physical blocks；[81:04](https://www.youtube.com/watch?v=EfM546A79aM&t=4864s) 说明按需追加。

### 17.4 Prefix sharing 与 copy-on-write

**Prefix sharing（前缀共享）** 让 tokenized prefix 完全相同、模型/KV 语义也相同的请求指向同一批只读物理块。**Copy-on-write，COW（写时复制）** 表示“只读时继续共享；某请求要改共享块时，才为它复制一份”。

例：块大小 4，两个请求共同拥有 8-token system prompt，因此共享逻辑块 0、1；每个物理块引用计数（reference count）为 2。

```text
请求 A table: [physical 3, physical 8]
请求 B table: [physical 3, physical 8]
reference count: block3=2, block8=2
```

若两者都只在新逻辑块 2 追加，各自分配新块，不必复制 0、1。若某实现需要在共享块 1 内写入 A 的分支数据：

1. 发现 physical 8 的引用计数是 2，不能原地改；
2. 复制 physical 8 到新 physical 11；
3. A 的 table 改为 `[3,11]`，B 仍为 `[3,8]`；
4. 修改 11；8 不受影响。

视频 [81:11](https://www.youtube.com/watch?v=EfM546A79aM&t=4871s) 连接 beam search；[81:19](https://www.youtube.com/watch?v=EfM546A79aM&t=4879s) 展示共享逻辑块；[81:27](https://www.youtube.com/watch?v=EfM546A79aM&t=4887s) 解释 copy-on-write；[81:32](https://www.youtube.com/watch?v=EfM546A79aM&t=4892s) 展示分叉后的映射。

### 17.5 和 OS paging 哪里像，哪里不像

相同：都有固定块、逻辑地址、物理地址、映射表和按需分配。

不同：这里管理的是 GPU HBM 中的张量块；block 大小按 kernel 和 KV 布局选择；通常不是 CPU 操作系统把冷页换到磁盘；attention kernel 必须理解块表。课程图是帮助理解的类比，不表示 vLLM 直接调用普通 OS page fault。视频 [81:52](https://www.youtube.com/watch?v=EfM546A79aM&t=4912s) 解释类比边界；[81:58](https://www.youtube.com/watch?v=EfM546A79aM&t=4918s) 回到 GPU kernel。

### 17.6 它不是免费午餐

- 块太大：最后一块内部碎片多。
- 块太小：block table 与 metadata（元数据，即描述数据放在哪里、属于谁的小记录）变多，查表和调度成本上升。
- kernel 要做间接寻址，必须专门优化。
- prefix sharing 只在前缀 token、模型权重、位置编码和 cache 语义一致时安全。

源码图 `paged-attention-parallel.png` 的实际内容不是“并行读取”：它画两个 sample 先共享 `Four score and seven / years ago our` 的 prefix；末词分成 `fathers` 与 `mothers` 时发生 COW，原共享块 reference count 从 $`2\to1`$，新物理块保存另一分支。这正是 §17.4 的“共享后写时分叉”。

“把 block read 与 attention 融合、使用更合适 kernel”来自源码 line 603–605 的文字列表，不是这张图本身的证据。视频 [82:05](https://www.youtube.com/watch?v=EfM546A79aM&t=4925s) 讨论 block granularity；[82:24](https://www.youtube.com/watch?v=EfM546A79aM&t=4944s) 说明 metadata；[82:29](https://www.youtube.com/watch?v=EfM546A79aM&t=4949s) 提到 specialized kernels；[82:42](https://www.youtube.com/watch?v=EfM546A79aM&t=4962s) 总结内存利用率收益。

**【延伸】** PagedAttention 由 vLLM 论文提出；当前 vLLM 实现细节会变化，本文只把论文/官方文档用于解释机制，不把某一版本性能当保证。视频 [83:02](https://www.youtube.com/watch?v=EfM546A79aM&t=4982s) 给出 serving throughput 结果图；[83:17](https://www.youtube.com/watch?v=EfM546A79aM&t=4997s) 提醒 workload 会改变收益；[83:28](https://www.youtube.com/watch?v=EfM546A79aM&t=5008s) 进入全讲收束。

---

## 18. 性能诊断与优化决策树

先问目标，再动手。把“模型慢”拆成可测问题：

```text
0. 输出是否正确、质量是否达标？
   └─ 否：先修 correctness；更快的错误答案没有价值。

1. 违反哪个 SLO？
   ├─ TTFT 高：查排队、prefill、prompt 长度、prefill batch。
   ├─ ITL 高：查 decode 权重/KV 带宽、batch、通信。
   ├─ E2E 高：同时查排队、TTFT、输出长度、ITL。
   └─ goodput 低：查超时请求比例，而不只看总 tokens/s。

2. 先卡显存吗？
   ├─ 参数太大：量化、剪枝、分片；核对 kernel 是否支持。
   ├─ KV 太大：GQA/MQA、MLA、CLA、local/CSA、PagedAttention。
   └─ 碎片/长度不齐：分页、continuous batching、准入控制。

3. Roofline 证据指向哪里？
   ├─ memory-bound：减少 bytes、提高复用、增大可承受 batch。
   ├─ compute-bound：减少 FLOPs、用合适精度/高效 kernel。
   └─ 都不是：查小 kernel launch、同步、排队、通信、CPU 调度。

4. 请求形态是什么？
   ├─ 长 prompt：优化 prefill、chunking、prompt cache。
   ├─ 长输出：重点优化 decode 和 KV。
   ├─ 离线：可容忍排队，尽量批量提高吞吐。
   └─ 在线：以 SLO 内 goodput 为目标。

5. 可否用候选换并行？
   └─ 测 draft 成本、接受率和 E[Y]；满足 break-even 才上 speculative。

6. 每次改完：同一 workload 复测
   correctness → quality → TTFT/ITL/E2E 分位数 → throughput/goodput → memory。
```

所谓“分位数”，例如 p95 latency，是把 100 个请求从快到慢排序，约第 95 个的时间；它比平均值更能看到慢尾。诊断必须记录模型、精度、prompt/output 长度分布、并发数、硬件和采样设置，否则两次数字不能公平比较。

---

## 19. 常见误区：错误在哪里，正确说法是什么

| # | 错误说法 | 为什么错 | 正确说法/反例 |
|---:|---|---|---|
| 1 | inference 就是“训练少跑一次 backward” | 请求、排队、逐 token 串行和 KV 生命周期都不同 | inference 是用已训练模型产生输出；服务系统也是问题的一半 |
| 2 | latency 就是 seconds/token | latency 可能指 TTFT、ITL 或 E2E | 先写清测量起止点和单位 |
| 3 | throughput 高，用户一定快 | 大 batch 可提高总吞吐却加长排队 | 同时看 TTFT、ITL、E2E 与 goodput |
| 4 | goodput 等于 throughput | 超过 SLO 的完成量不算 goodput | goodput 是满足质量和 SLO 的有效吞吐 |
| 5 | prefill 和 decode 是同一 shape | prefill 的 $`T`$ 大，decode 通常 $`T=1`$ | 分开建模和批处理 |
| 6 | autoregressive 一次能并行知道所有未知答案 | 第 $`t+1`$ 枚依赖第 $`t`$ 枚实际输出 | 单请求的 decode 位置间有数据依赖 |
| 7 | KV cache 也缓存 Q | 历史 query 不被未来 token 再查询 | 通常缓存历史 K/V；当前步 Q 现算 |
| 8 | 有 KV cache 后所有推理都是 $`O(T^2)`$ | 这只说 attention 累计项；MLP、投影等仍在 | 分项计算 FLOPs 和 bytes |
| 9 | FLOP 是时间 | FLOP 是操作数量 | 时间下界还要除以 FLOP/s 或带宽 |
| 10 | BF16 永远正好达到 H100 989 TFLOP/s | 那是特定峰值口径 | 实际受 shape、kernel、时钟和利用率限制 |
| 11 | arithmetic intensity 越高，程序一定越快 | 工作量也可能同时增加 | 它只比较 FLOPs 与数据流量，需结合 Roofline 和实测 |
| 12 | $`B\ll D,F`$ 时强度“严格等于” $`B`$ | 精确分母还有 $`B/F`$ 与 $`B/D`$ | 只能写 $`I\approx B`$ 并检查条件 |
| 13 | attention 强度可靠 batch $`B`$ 无限提高 | $`B`$ 在其 FLOPs/bytes 教学式中约掉 | 约为 $`TS_{total}/(S_{total}+T)`$ |
| 14 | decode attention 的强度随上下文线性增长 | $`T=1`$ 时为 $`S_{total}/(S_{total}+1)<1`$ | 上下文越长，搬 KV 越多，但强度仍接近 1 |
| 15 | 参数 B 和 batch B 是一个量 | 一个是 billion 的单位，一个是符号 | 本文数量单位写 `B params`，批量写数学 $`B`$ 并看上下文 |
| 16 | GB 与 GiB 相同 | $`1\text{ GB}=10^9`$ bytes，$`1\text{ GiB}=2^{30}`$ bytes | 换算前先声明单位 |
| 17 | MQA 是 head dimension 变成 1 | MQA 是 KV-head count $`K=1`$ | 每个 head 的 feature width $`H`$ 仍可为 128 |
| 18 | GQA 只减 KV，不改模型参数 | K/V projection 矩阵也变小 | 参数和 per-sequence KV 都随 $`K`$ 减少 |
| 19 | 源码写 “Worse latency” 就一定对 | 其同一公式给 K=8 更少 memory、更低理想 latency | 这是已复算的 source-text inconsistency |
| 20 | MLA 就是 GQA 的另一个名字 | 两者压缩轴不同 | GQA 减 KV heads；MLA 缓存低维 latent |
| 21 | MLA 可把带 RoPE 的 K 投影无条件吸收到 Q 侧 | 位置旋转依赖历史位置 | 通常分 rotary/non-rotary 部分处理 |
| 22 | CLA 是 sequence-window attention | CLA 压缩 layer 轴 | 多层共享 K/V cache |
| 23 | local attention 永远只看固定数量 token，所以信息永不过远处 | 多层可传播，hybrid 还插 full/global 层 | 直接边数减少不等于完全没有长程通路 |
| 24 | DeepSeek V4 的 CSA、DSA、HCA 是三个随便可互换的缩写 | 它们有组件/层级关系 | 官方材料中 CSA 结合压缩与 DSA top-k/local 路径；HCA 更激进压缩 |
| 25 | 2026 模型表就是永恒事实 | 架构与版本会更新 | 标注官方材料和查询日期 |
| 26 | quantization 只是把小数四舍五入 | 还需要 scale、可能有 zero point 与 clamp | 量化和反量化要成对定义 |
| 27 | $`x/s+z`$ 不需要 round | 整数存储不能保存任意小数 | 必须 round，并超范围时 clamp |
| 28 | symmetric 与 asymmetric 一样 | asymmetric 可用非零 zero point 表示偏移区间 | 代价是额外参数/计算和 kernel 约束 |
| 29 | FP8 只有一个最大值 | E4M3/E5M2 及具体变体范围不同 | 课程 ±240 需标格式边界；NVIDIA TE 文档给 E4M3 ±448 |
| 30 | 量化一定加速 | 若硬件/kernel 不支持，dequant 反而有开销 | 同时测 memory、kernel 和精度 |
| 31 | AWQ 的最终方案是永远保留 1% FP16 权重 | 图中该项是动机性 ablation，硬件效率差 | AWQ 核心是 activation-aware scaling 后量化 |
| 32 | calibration 是继续完整训练 | 它用小样本估计范围/重要性 | 不等于大规模更新全部权重 |
| 33 | 剪去 50% 非结构化权重，GPU 必快 2 倍 | 稀疏索引、零值和 kernel 可能浪费 | 结构化形状或受支持稀疏格式更容易提速 |
| 34 | pruning 后无需 repair | 删除会改变输出 | 常用 fine-tuning 或 distillation 修复 |
| 35 | distillation 就是复制 teacher 参数 | student 通常结构不同 | 它学习 teacher 输出/logits/hidden 信号 |
| 36 | 源码的 pruning 占位注释已给完整算法 | 函数只有讲义边界与图片 | 本文不伪造未写实现 |
| 37 | speculative sampling 用小模型输出替代大模型 | target 仍验证并纠偏 | 正确拒绝采样保持 target 分布 |
| 38 | 接受率高就一定加速 | draft 和批量 target 验证也有成本 | 检查 $`Kc_d+c_q(K)<E[Y]c_q(1)`$ |
| 39 | 被拒绝后直接从 $`q`$ 重抽也保持证明 | 会重复分配已经接受过的概率质量 | 要从 normalized $`\max(q-p,0)`$ 抽 |
| 40 | speculative exact 与任何 top-p/temperature 组合自动兼容 | proposal、target 和验收概率必须使用一致的变换后分布 | 实现设置不一致会失真 |
| 41 | continuous batching 等于把 batch 固定设最大 | 它每个 iteration 重新进出请求 | 还受显存、SLO 与公平性约束 |
| 42 | ragged batch 可直接 padding 到最长，成本完全一样 | padding 会算无效位置 | selective/ragged kernels 只处理有效 token |
| 43 | attention 和 MLP 都能无脑拼成同一 `[sum tokens,H]` | attention 还需要序列边界、位置和 KV | 非 attention 更容易直接拼接 |
| 44 | paging 会把 KV 自动换到磁盘 | OS 只是类比 | PagedAttention 主要是 GPU KV 物理块管理 |
| 45 | total free blocks 足够就一定能分配连续 cache | 旧连续分配器可能受 external fragmentation | block table 允许物理不连续 |
| 46 | block size 越小越好 | metadata、查表和 kernel 开销会上升 | 在内部碎片与管理开销间权衡 |
| 47 | prefix 文本看起来相同就能共享 | tokenizer、token IDs、模型、位置和 cache 语义都需一致 | 共享前先验证 exact cache identity |
| 48 | copy-on-write 是每次都复制 | 只读共享时不复制 | 真要写共享块时才复制 |
| 49 | 平均 latency 足够 | 少数慢请求会被平均值遮住 | 同看 p50/p95/p99 和 SLO violation |
| 50 | 一张 benchmark 图能证明所有 workload | prompt/output 长度、并发和硬件都会改变结果 | 记录条件，并在自己的流量上复测 |

---

## 20. 术语表：看到英文不再卡住

| 术语 | 一句话人话解释 |
|---|---|
| inference | 用训练好的模型接收输入并产生输出 |
| request / prompt / token | 一次服务任务 / 输入文字或 token 序列 / 模型处理的离散编号单位 |
| autoregressive | 下一枚输出依赖已经生成的前缀 |
| prefill / decode | 并行处理 prompt 建 cache / 每步生成下一枚 token |
| logit | softmax 前的原始分数 |
| KV cache | 保存历史 key/value，避免以后反复重算 |
| HBM | GPU 的高带宽主显存 |
| FLOP / FLOP/s | 一次浮点运算 / 每秒可做多少浮点运算 |
| byte / BF16 | 8 bits 的存储单位 / 每元素常占 2 bytes 的 bfloat16 格式 |
| arithmetic intensity | FLOPs 除以搬运 bytes |
| bandwidth | 每秒可搬多少 bytes |
| compute-/memory-bound | 计算峰值 / 数据搬运先成为上限 |
| latency / TTFT / ITL | 请求时间 / 到第一枚时间 / 相邻输出 token 间隔 |
| throughput / goodput / SLO | 总完成率 / 达标完成率 / 服务必须满足的目标 |
| MHA / MQA / GQA | 每 query 独享 KV / 全共享一组 KV / 分组共享 KV |
| MLA / CLA | 缓存低维 latent / 跨层共享 KV |
| quantization | 用更少 bit 的数近似表示原权重/激活 |
| scale / zero point / clamp | 量化步长 / 整数零点 / 把超范围值截到边界 |
| PTQ / QAT | 训练后量化 / 训练时模拟量化误差 |
| Hessian | loss 对参数的二阶变化信息；GPTQ 用近似曲率决定误差修正 |
| pruning / distillation | 删除不重要结构 / 让 student 学 teacher |
| calibration | 用代表性小样本估计范围或重要性 |
| proposal / target | 提候选的便宜模型 / 决定最终分布的模型 |
| rejection / residual | 随机拒绝过量候选 / 拒绝后补足的概率分布 |
| ragged batch | 不同请求有效长度不同的批 |
| iteration-level scheduling | 每一轮 token 后重新安排活跃请求 |
| fragmentation | 空间因块内浪费或洞太散而难利用 |
| paging / block table | 分块管理 / 逻辑块到物理块的映射表 |
| copy-on-write | 先共享，某一方要写时才复制 |

---

## 21. 自测题（80 题）

> 建议：先遮住 §22。标有“手算/填表”的题必须写中间步骤；只看懂答案不算会。

### 21.1 基础概念（1–10）

1. inference 与 training 的目标分别是什么？
2. request、prompt、token 各是什么？
3. online inference 与 offline inference 的主要目标有什么不同？
4. TTFT、ITL、E2E latency、throughput、goodput、SLO 各是什么？
5. prefill 与 decode 的输入形状和任务有什么区别？
6. 写出 $`B,S_{old},T,S_{total},D,F,N,K,G,H,L,V`$ 的含义，并写 $`S_{total}`$ 与前两者的关系。
7. HBM、byte、BF16、FLOP、bandwidth 各是什么？
8. 用一句话分别说 MHA、MQA、GQA、MLA、CLA 压缩什么。
9. quantization、pruning、distillation 的动作分别是什么？
10. continuous batching 与 PagedAttention 分别解决哪类浪费？

### 21.2 手算、shape 与时间线（11–75）

11. **【手算】** 请求在 0 ms 到达；排队 10 ms；prefill 30 ms；四枚输出在 40、55、70、85 ms 可见。求 TTFT、三个 ITL、平均 ITL、E2E latency。
12. **【手算】** 同一请求输出 5 枚，TTFT=120 ms，之后 4 个 ITL 都是 25 ms。最后一枚何时到？E2E 是多少？
13. **【手算】** 10 秒完成 800 枚 token，其中 600 枚满足 SLO。求 throughput 与 goodput。
14. **【填表】** prompt 长 3，生成 $`y_1,y_2,y_3,y_4`$ 四枚。列出 1 次 prefill 和 3 次后续 decode 的“调用前 cache、本次输入、采样输出、调用后 cache”；解释为什么不是 4 次 decode，以及最后 cache 为什么是 6 不是 7。
15. **【手算】** 不用 KV cache，四步 attention 重算量按 $`1^2,2^2,3^2,4^2`$。求总和，并和末长度 $`T=4`$ 的 $`T^3=64`$ 比较常数差。
16. **【手算】** 有 KV cache，四步按 $`1,2,3,4`$。求总和，并写成 $`T(T+1)/2`$ 验证。
17. **【画 shape】**$`B=2,S_{old}=5,T=3,S_{total}=8,N=4,H=8`$。写 Q、merged K/V 和 attention score 的 shape；若 GQA $`K=2`$，K/V shape 怎么变？
18. **【手算】**$`N=16,K=4`$，求 $`G`$；列出 query heads 0–7 分别使用哪一个 KV head，假设连续每 $`G`$ 个一组。
19. **【手算】** 矩阵 $`[B,D]@[D,F]`$，$`B=2,D=4,F=8`$、BF16。求 FLOPs、读 X、读 W、写 Y bytes、总 bytes、强度。
20. **【手算】**$`B=1,D=F=4096`$，由 $`I=B/(B/F+1+B/D)`$ 算近似强度。
21. **【手算】** H100 教学峰值 989 TFLOP/s、带宽 3.35 TB/s。求 ridge point。
22. **【判断+手算】** 若 kernel 强度 100 FLOP/byte，带宽 roof 是多少 TFLOP/s？与 989 比，教学 Roofline 判为哪边受限？
23. **【手算】** SwiGLU MLP 取 $`B=2,T=3,D=4,F=6`$。求 $`6BTDF`$ FLOPs，并分别求：课程单边账 $`4BTD+4BTF+6DF`$、完全融合账 $`4BTD+6DF`$、U/G 落 HBM 再读账 $`4BTD+8BTF+6DF`$。
24. **【手算】** 用第 23 题结果求三种强度；再算近似 $`BT`$。为什么三种精确值不同，却能在什么条件下趋近同一个 $`BT`$？
25. **【手算】** attention 取 $`B=2,S_{old}=2,T=3`$，故 $`S_{total}=5`$，$`D=4`$。求 $`4BTS_{total}D`$ FLOPs、$`4BS_{total}D+4BTD`$ bytes 和强度。
26. **【手算】** prompt 长 $`P=8`$ 时，prefill 的 $`S_{old},T,S_{total}`$ 和强度各是什么？某次 decode 调用前 $`S_{old}=7`$ 时，append 后 $`S_{total}`$ 和强度是什么？
27. **【填表】** 给出课程参数公式 $`2VD+3DFL+2DNHL+2DKHL`$；把四组项分别对应到模型模块。
28. **【手算】** Llama 2 13B 教学配置 $`V=32000,D=5120,F=13824,L=40,N=K=40,H=128`$。分别算 embedding、MLP、Q/O、K/V 参数，再相加。
29. **【手算】** 第 28 题参数用 BF16 存储，求十进制 GB。
30. **【手算】** 调用完成后 $`S_{total}=1024,K=40,H=128,L=40`$、K/V 两份、BF16。求单序列 KV bytes、GB、GiB。
31. **【手算】** MHA 参数 26.0308992 GB，每序列 KV 0.8388608 GB，$`B=64`$，带宽 3.35 TB/s。求总 memory、理想 decode-step latency 下界、decode token-steps/s。为什么最后一个单位不是完整 requests/s？
32. **【手算】** GQA 改 $`K=8`$，参数 22.675456 GB、每序列 KV 0.16777216 GB，$`B=64`$。求同三项，并继续使用 decode token-steps/s。
33. **【判断+复算】** 比较第31、32题。源码称 GQA 的 latency “Worse”是否符合它自己的 `memory/bandwidth` 模型？
34. **【手算】** 把 0.8388608 GB 换成 GiB；写出用到的 byte 定义。
35. **【手算】** 一张卡每秒处理 100 requests，复制到 4 张独立卡且流量充分，理想总吞吐多少？每请求是否必须跨卡通信？
36. **【填表】** 模型按 2 卡 sharding。列出“参数每卡”“单请求计算”“通信”“单请求 latency 风险”相对单卡的变化。
37. **【画时间线】** batch 容量3。A在轮1到达需2枚，B轮1到达需4枚，C轮2到达需2枚，D轮3到达需1枚。按 continuous batching 列四轮活跃集合。
38. **【手算】** MHA $`K=40`$ 改 MQA $`K=1`$。只看 KV cache，缩小多少倍？若原来单序列 0.8388608 GB，新的多少 GB？
39. **【手算】** 简化 MLA：普通每 token 每层 K/V 各 $`H=128`$、40 heads、BF16；latent 维 320，另存 RoPE key 64 维。求一层普通 bytes 与 MLA bytes、压缩倍数。
40. **【手算】** 40 层原本各自存一份每层 1000 bytes 的 KV。CLA 每 4 层共享一份。共有几组？总 bytes 与缩小倍数？
41. **【手算】** 调用后总长度 $`S_{total}=8`$ 的 causal full attention 有多少允许边？用 $`1+2+\cdots+8`$ 算。若每个 query 最多看自己和前2枚，逐行数边并求总边数。
42. **【手算】** 8层 attention 中2层 full、6层每 query 平均看128个 key，$`S_{total}=1024`$。用“每层每 query 边数”的教学近似，求总边数并与8层 full比较倍数。
43. **【手算】**$`x=5.2342,s=0.1,z=4`$，用 $`q=\mathrm{round}(x/s)+z`$ 与 $`\hat x=s(q-z)`$ 量化/反量化，求误差。
44. **【手算】** INT8 范围 $`[-128,127]`$。若未 clamp 的 q=140，存什么？若 q=-150 呢？
45. **【填表】** symmetric 与 asymmetric quantization 的 zero point 通常怎样；各适合什么范围？
46. **【手算】** 两输出通道权重最大绝对值分别 1 和 10。若都映射到 symmetric INT8 的 127，per-channel scales 各是多少？per-tensor scale 由全局最大值决定时是多少？
47. **【判断】** 课件写 FP8 约 ±240，NVIDIA TE 文档写 E4M3 ±448。是否能说“一个必错”？应该怎样记录？
48. **【填步骤】** 给 pruning 流程 importance→remove→repair→verify，每步写输入/动作/输出。
49. **【判断+解释】** 100万个权重里把50%变成零，普通 dense GEMM 仍读算全部位置。能否只因“零很多”宣称2倍加速？
50. **【画数据流】** 写 teacher logits、student logits、ground-truth 三者怎样进入 distillation loss；说明 inference 时留下谁。
51. **【手算】**$`p=[0.7,0.3],q=[0.4,0.6]`$。画 A/B 的 proposal、接受、拒绝残差事件树，证明最终为 q。
52. **【手算】**$`p=[0.5,0.3,0.2],q=[0.2,0.5,0.3]`$。算直接接受质量、拒绝总量、normalized residual 和最终分布。
53. **【手算】**$`K=4,a=0.8`$，求 $`E[A]`$ 与 $`E[Y]`$。
54. **【手算】**$`K=4,a=0.3`$，求 $`E[Y]`$；若 speculative 成本1.72、普通每 token 成本1，是否划算？
55. **【手算】**$`K=4,c_d=0.08,c_q(K)=1.4,a=0.8`$。求 break-even 两边与理想加速比。
56. **【填表】** 列三种让 speculative 不再保证 exact 的实现错误，并写修法。
57. **【填表】** 按第37题的请求，分别写静态 batch（A/B全结束后才收C/D）与 continuous batch 中 C 第一次运行的轮次。
58. **【画 shape】** 三个请求有效 token 为3、9、5，模型 hidden width $`D=512`$（不是 head width $`H`$）。写各 activation shape、拼接后的 non-attention shape，以及 attention 额外需要的元数据。
59. **【判断+解释】** 把 decode batch 从16增到64，总 tokens/s 上升，但 p95 ITL 超过SLO。goodput一定上升吗？下一步应测什么？
60. **【手算】** 预留16槽只用6槽，求内部碎片槽数与比例。
61. **【手算】** 空洞为3槽和4槽，请求要连续5槽。总空闲够不够？传统连续分配能否满足？这是哪类碎片？
62. **【手算】** block size4，序列长10。需几块？最后一块用几槽、浪费几槽？
63. **【手算】** block table `[7,1,5]`，block size4。token index9 落在哪个 logical block、块内 offset、physical block？
64. **【手算】** 两请求共享8-token prefix，block size4，每物理KV块100 bytes。不共享需多少 bytes？共享需多少？省多少？忽略块表。
65. **【填表】** A/B共享physical block8，引用计数2；A要写。列 COW 前后 A table、B table、block8/new block引用计数。
66. **【解释+数字】** block size从16降为4时，一条18-token序列内部浪费从多少槽变多少槽？为什么仍不能断言4一定更好？
67. **【手算】** 生成 $`T=4`$，无cache累计 $`1^2+2^2+3^2+4^2`$，有cache累计 $`1+2+3+4`$。求比值；说明大T的量级分别为何。
68. **【手算】** 公式中 K/V 参数为 $`2LDHK=2LDK H`$。固定 $`L=40,D=5120,H=128`$，K从40降到8，分别求 K/V 参数与减少量。
69. **【填表】** 一个请求 prompt=1024、共生成128枚输出。写 prefill 的 $`S_{old},T,S_{total}`$；解释第一枚已由 prefill 采样；再写为了采样第2枚的首个后续 decode，以及为了采样第128枚的最后一个后续 decode 的 $`S_{old},T,S_{total}`$。各阶段主要并行轴是什么？
70. **【手算】** 1000 requests 中900完成，800满足TTFT SLO，750同时满足TTFT与ITL SLO。以“两个SLO都满足”为达标，completion rate与goodput fraction各多少？
71. **【手算】** 已知 q=56、s=0.1、z=4，反量化；若原值5.24求误差。再说明 z=4 对应哪个实数值。
72. **【手算+判断】** 1B参数由BF16 2GB量化到INT8 1GB，权重bytes减多少倍？若硬件只能先解量化回BF16且kernel更慢，能否推出端到端2倍？
73. **【填流程】** 从 dense checkpoint 到“剪枝+蒸馏”的五步实验流程，至少包含 calibration、quality baseline、hardware benchmark。
74. **【手算】** 二词 speculative 例中，如果错误地“拒绝后从q重抽”，求最终 A 概率，说明为何不等0.4。
75. **【手算】** block size8，block table `[4,9,2]`。token17读哪个logical block、offset和physical block？若physical block起始byte地址10000、每token KV占256 bytes，求地址。

### 21.3 综合判断（76–80）

76. 为什么“量化、GQA、PagedAttention 都省显存”仍不是同一类优化？
77. 在线服务 TTFT 高而 ITL 正常，按 §18 决策树先查哪三类证据？
78. KV cache OOM（Out Of Memory，显存不足而失败），但模型权重能放下。请给至少四个候选动作，并写每个动作的代价。
79. 把 GQA、MLA、CLA、local/CSA、quantization 分别对应到 head、feature、layer、sequence、bit-width 五个轴。
80. 综合题：服务使用13B MHA BF16，$`B=64`$ 时显存和TTFT超标。设计“先测—改造—复测”的最小计划，至少涉及架构、数值格式、调度和正确性四方面。

---

## 22. 自测答案（1–80）

### 22.1 答案 1–10

1. **Inference** 用固定的已训练参数产生输出；**training** 通过 loss、backward 和 optimizer 更新参数。训练追求学到参数，推理追求在质量约束下低成本地回答请求。
2. request 是一次服务任务；prompt 是任务的输入；token 是 tokenizer 把输入切成的离散编号单位。一个 request 可含许多 prompt tokens，并生成许多 output tokens。
3. online 面向正在等答案的用户，重视 TTFT/ITL/E2E 与 SLO；offline 可积累任务后批量跑，通常更重视总吞吐和单位成本。
4. TTFT 是到第一枚 token 的时间；ITL 是相邻输出 token 的时间间隔；E2E 是到最后一枚的总时间；throughput 是单位时间总完成量；goodput 是达标的有效完成量；SLO 是预先约定的服务目标。
5. prefill 一次处理多枚 prompt tokens，建立每层 KV cache；decode 通常每请求每步只输入 1 枚新 token，并读取历史 KV 生成下一枚。
6. $`B`$ batch；$`S_{old}`$ 调用前 cache 长度；$`T`$ 本次输入/query tokens；$`S_{total}=S_{old}+T`$ 是 append 后可见 source 总长；$`D`$ hidden width；$`F`$ MLP width；$`N`$ query heads；$`K`$ KV heads；$`G=N/K`$ 每KV组query数；$`H`$ head width；$`L`$ layers；$`V`$ vocabulary size。
7. HBM 是 GPU 高带宽主显存；byte=8 bits；BF16 通常每元素2 bytes；FLOP 是一次浮点运算；bandwidth 是每秒搬运 bytes。
8. MHA 不共享 KV；MQA 把 KV-head 轴压到1；GQA 把 KV-head 轴压到少数组；MLA 压 feature 到 latent；CLA 跨 layer 共享 KV。
9. quantization 减少每数 bit；pruning 删除权重/通道/层等；distillation 让 student 学 teacher 信号。
10. continuous batching 减少完成时间不同造成的空 batch slots；PagedAttention 减少 KV 连续预留与碎片，并支持共享块。

### 22.2 答案 11–40

11. 到达0，第一枚40，所以 TTFT=$`40-0=40`$ ms。ITL：$`55-40=15`$、$`70-55=15`$、$`85-70=15`$ ms；平均 $`(15+15+15)/3=15`$ ms。最后一枚85到，所以 E2E=85 ms。排队10和prefill30正好组成首枚前40 ms。
12. 第一枚在120 ms。5枚之间有4个间隔，因此最后一枚在 $`120+4\times25=220`$ ms；E2E=220 ms。
13. throughput=$`800/10=80`$ tokens/s；goodput=$`600/10=60`$ SLO-qualified tokens/s。
14. Prefill：调用前0，输入3枚prompt，采样$`y_1`$，调用后cache3；decode1：调用前3，输入$`y_1`$，采样$`y_2`$，调用后4；decode2：调用前4，输入$`y_2`$，采样$`y_3`$，调用后5；decode3：调用前5，输入$`y_3`$，采样$`y_4`$，调用后6。Prefill 已经采样第一枚，所以只需3次后续decode。$`y_4`$ 是最后一次调用的**输出**，尚未作为下一次输入，故它的K/V还没append，最后是6而不是7。
15. $`1^2+2^2+3^2+4^2=1+4+9+16=30`$。$`T^3=64`$；30不是64，因为 $`\sum_{t=1}^Tt^2=T(T+1)(2T+1)/6\approx T^3/3`$，大O忽略常数约1/3。
16. $`1+2+3+4=10`$。公式 $`T(T+1)/2=4\times5/2=10`$。因此累计主 attention 点积量约 $`O(T^2)`$。
17. Q 为 `[B,T,N,H]=[2,3,4,8]`。MHA merged K/V 为 `[B,S_total,N,H]=[2,8,4,8]`；score 用本文轴顺序 `[B,T,N,S_total]=[2,3,4,8]`。GQA $`K=2`$ 时 merged K/V 改为 `[2,8,2,8]`；Q不变，每个KV head供$`G=N/K=2`$个query heads用。
18. $`G=N/K=16/4=4`$。heads 0–3用KV0；4–7用KV1；8–11用KV2；12–15用KV3。题目只要求0–7，所以答案是0,0,0,0,1,1,1,1。
19. FLOPs=$`2BDF=2\times2\times4\times8=128`$。读X=$`2BD=2\times2\times4=16`$ bytes；读W=$`2DF=2\times4\times8=64`$；写Y=$`2BF=2\times2\times8=32`$。总112 bytes。强度$`128/112=1.142857\ldots`$ FLOP/byte。
20. $`I=1/(1/4096+1+1/4096)=1/(1+2/4096)=1/1.00048828125\approx0.999512`$ FLOP/byte。
21. ridge=$`989/3.35=295.2239\ldots`$ FLOP/byte，约295。$`10^{12}`$ 在分子分母抵消。
22. memory roof=$`100\text{ FLOP/byte}\times3.35\text{ TB/s}=335\text{ TFLOP/s}`$。$`335<989`$，教学 Roofline 判为 memory-bound；实际还需实测。
23. FLOPs=$`6(2)(3)(4)(6)=864`$。公共项：$`4BTD=96`$，$`4BTF=144`$，$`8BTF=288`$，$`6DF=144`$。课程单边账$`=96+144+144=384`$ bytes；完全融合账$`=96+144=240`$ bytes；U/G落HBM再读账$`=96+288+144=528`$ bytes。
24. 三种强度：课程$`864/384=2.25`$；完全融合$`864/240=3.6`$；落HBM再读$`864/528=18/11\approx1.636`$ FLOP/byte。近似$`BT=2\times3=6`$。Tiny例的$`BT=6`$没有远小于$`D=4,F=6`$；当$`BT\ll D,F`$且权重项$`6DF`$主导时，三种分母除以$`DF`$后其它项趋小，才都趋近$`BT`$。
25. $`S_{total}=S_{old}+T=2+3=5`$。FLOPs=$`4(2)(3)(5)(4)=480`$。bytes=$`4(2)(5)(4)+4(2)(3)(4)=160+96=256`$。强度$`=480/256=1.875`$，也等于$`TS_{total}/(S_{total}+T)=3\times5/(5+3)=15/8=1.875`$。
26. Prefill：$`S_{old}=0,T=8,S_{total}=8`$，强度$`=8\times8/(8+8)=4`$ FLOP/byte。Decode：$`S_{old}=7,T=1,S_{total}=8`$，强度$`=1\times8/(8+1)=8/9\approx0.889<1`$。
27. $`2VD`$ 是 input embedding 与 output/unembedding 各一份 $`VD`$；$`3DFL`$ 是每层 SwiGLU 的 up、gate、down 三矩阵；$`2DNHL`$ 是每层 Q 与 output projection；$`2DKHL`$ 是每层 K 与 V projection。MHA 配置满足 $`D=NH`$，所以 $`2DNHL=2D^2L`$；GQA 中通常 $`KH<D`$，不能把 K/V 项仍写成 $`2D^2L`$。
28. embedding=$`32000\times5120=163{,}840{,}000`$；若输入embedding与输出头不共享/两份，课程公式取$`2VD=327{,}680{,}000`$。MLP=$`40\times3\times5120\times13824=8{,}493{,}465{,}600`$。Q/O=$`40\times2\times5120^2=2{,}097{,}152{,}000`$。K/V=$`40\times2\times5120\times40\times128=2{,}097{,}152{,}000`$。总$`327{,}680{,}000+8{,}493{,}465{,}600+2{,}097{,}152{,}000+2{,}097{,}152{,}000=13{,}015{,}449{,}600`$。
29. BF16 bytes=$`13{,}015{,}449{,}600\times2=26{,}030{,}899{,}200`$ bytes。除$`10^9`$得26.0308992 GB。
30. 元素数=$`S_{total}\times K\times H\times L\times2=1024\times40\times128\times40\times2=419{,}430{,}400`$。乘BF16 2 bytes=$`838{,}860{,}800`$ bytes。GB=$`0.8388608`$；GiB=$`838{,}860{,}800/1{,}073{,}741{,}824=0.78125`$。
31. KV batch=$`64\times0.8388608=53.6870912`$ GB。总memory=$`26.0308992+53.6870912=79.7179904`$ GB。带宽$`3.35\text{ TB/s}=3350`$ GB/s，所以理想decode-step latency=$`79.7179904/3350=0.0237964`$ s。每step为64个活跃请求各产一枚，故$`64/0.0237964\approx2689.48`$ **decode token-steps/s**。它不是完整requests/s：一个请求通常要许多输出steps才结束，长度也不相同。
32. KV batch=$`64\times0.16777216=10.73741824`$ GB。总=$`22.675456+10.73741824=33.41287424`$ GB。latency=$`33.41287424/3350=0.00997399`$ s。decode token-steps/s=$`64/0.00997399\approx6416.69`$。仍不能把它叫完整requests/s。
33. 不符合。GQA教学下界$`9.974`$ ms，小于MHA的$`23.796`$ ms；throughput也更高。因此源码那句“Worse latency”与其同一公式/数值冲突，视频口头也纠正为更低 latency。
34. $`1\text{ GB}=10^9`$ bytes；$`1\text{ GiB}=2^{30}=1{,}073{,}741{,}824`$ bytes。$`0.8388608`$ GB=$`838{,}860{,}800`$ bytes；除$`2^{30}`$得0.78125 GiB。
35. 理想总吞吐=$`4\times100=400`$ requests/s。完整复制时一个请求可只在一张卡完成，无需为每层跨卡通信；入口调度仍要把请求送到某副本。前提是流量充足且没有共享CPU/网络瓶颈。
36. 2卡sharding：参数每卡约一半；单请求同时用两卡完成各自分片；每层或若干层要collective/p2p通信；单请求latency可能因通信和同步上升。优点是模型或更大KV批可放下，不保证单请求更快。
37. 轮1 `{A,B}`；轮2 C到达，容量3，所以 `{A,B,C}`，该轮后A完成；轮3 D填入A释放的槽，集合 `{B,C,D}`，该轮后C/D完成；轮4 `{B}`，该轮后B完成。每轮都不超过容量3。
38. 缩小$`40/1=40`$倍。新KV=$`0.8388608/40=0.02097152`$ GB。
39. 普通：K/V两份、40 heads、每head128、BF16，所以$`2\times40\times128\times2=20{,}480`$ bytes/token/layer。简化MLA：latent320加RoPE key64，共384元素，BF16为$`384\times2=768`$ bytes。压缩$`20{,}480/768\approx26.67`$倍。真实MLA布局和精度可能不同。
40. 每4层一组，$`40/4=10`$组。总$`10\times1000=10{,}000`$ bytes；原$`40\times1000=40{,}000`$；缩小4倍。

### 22.3 答案 41–80

41. causal full边数=$`1+2+3+4+5+6+7+8=36`$。窗口看自己及前2枚，各行边数为1,2,3,3,3,3,3,3；总$`1+2+6\times3=21`$。
42. full边数教学近似：每层$`1024^2`$，2层为$`2\times1{,}048{,}576=2{,}097{,}152`$。local为$`6\times1024\times128=786{,}432`$。合计$`2{,}883{,}584`$。8层full=$`8{,}388{,}608`$；缩小$`8{,}388{,}608/2{,}883{,}584\approx2.91`$倍。causal边界会改变精确数，不改教学结论。
43. $`x/s=5.2342/0.1=52.342`$；round为52；$`q=52+4=56`$。反量化$`\hat x=0.1(56-4)=5.2`$。有符号误差$`5.2-5.2342=-0.0342`$，绝对误差0.0342。
44. clamp到上界：140存127；clamp到下界：-150存-128。
45. symmetric通常令zero point为0或固定中心，适合以0近似对称的范围；asymmetric允许非零zero point，把偏移区间更充分映到整数范围。后者要存/处理zero point。
46. per-channel：$`s_1=1/127\approx0.007874`$，$`s_2=10/127\approx0.078740`$。per-tensor全局最大10，所以统一$`s=10/127\approx0.078740`$；小通道1只能用约13个整数步，精度较差。
47. 不能脱离格式判谁“必错”。FP8是家族；E4M3/E5M2和是否含Inf/NaN的变体范围不同。应写“课件使用约±240的简化/变体口径；NVIDIA Transformer Engine当前E4M3文档给±448”，并注明来源版本。
48. importance：输入checkpoint+calibration data，输出分数；remove：按分数/结构预算删除，输出瘦模型；repair：fine-tune或distill，输出修复模型；verify：对质量、显存、真实hardware latency/throughput复测。
49. 不能。dense GEMM仍读100万个位置并发出同样乘加；零值未被受支持的稀疏kernel跳过。2倍参数零不等于2倍wall-clock。
50. 数据流：输入同时送teacher和student；teacher logits作为软目标，ground-truth作为硬目标；student logits进入`软目标loss + 硬标签loss`，只对student反传。部署时通常只留下student。
51. proposal A概率0.7，接受$`0.4/0.7=4/7`$，输出A质量$`0.7\times4/7=0.4`$；A拒绝质量0.3。proposal B概率0.3，接受$`\min(1,0.6/0.3)=1`$，先给B 0.3。残差$`\max(q-p,0)=[0,0.3]`$归一化为[0,1]，拒绝的0.3补B。最终[0.4,0.6]。
52. 直接接受$`\min(p,q)=[0.2,0.3,0.2]`$，和0.7；拒绝0.3。原始残差$`[0,0.2,0.1]`$，除0.3得$`[0,2/3,1/3]`$。乘拒绝量0.3得$`[0,0.2,0.1]`$；相加得$`[0.2,0.5,0.3]=q`$。
53. $`E[A]=0.8+0.64+0.512+0.4096=2.3616`$ accepted draft tokens。$`E[Y]=1+E[A]=3.3616`$ emitted tokens under the simplified model。
54. $`E[Y]=1+0.3+0.09+0.027+0.0081=1.4251`$。普通生成这些token成本1.4251，小于speculative 1.72，所以不划算。
55. 左边$`Kc_d+c_q(K)=4\times0.08+1.4=0.32+1.4=1.72`$。右边$`E[Y]c_q(1)=3.3616\times1=3.3616`$，满足左<右。理想加速$`3.3616/1.72\approx1.954`$倍。
56. 例1：target验收用未做temperature的q，而实际采样用了temperature；修：验收与输出使用同一变换后分布。例2：拒绝后直接从q抽；修：用normalized $`\max(q-p,0)`$。例3：候选上下文或tokenizer不一致；修：确保同一token IDs和prefix state。
57. 容量3时，static若固定首批只含A/B并等它们都结束，C最早轮5运行；continuous在C到达的轮2就把它放入第三个空槽。因此答案是static轮5、continuous轮2。
58. 因为 $`D=512`$，shapes 是 `[3,D]=[3,512]`、`[9,D]=[9,512]`、`[5,D]=[5,512]`；non-attention concat为`[17,D]=[17,512]`。attention还需lengths `[3,9,5]`、sequence boundaries、positions和每请求block table/KV pointers。
59. 不一定。超过ITL SLO的token/request不能计入goodput。应测达标请求数/s、p50/p95/p99 ITL、排队、HBM占用、batch分布，并找SLO内最佳batch。
60. 浪费$`16-6=10`$槽；比例$`10/16=0.625=62.5\%`$。
61. 总空闲$`3+4=7\ge5`$，数量够；但没有一段连续5，所以传统连续分配失败。这是external fragmentation。
62. $`\lceil10/4\rceil=3`$块。最后一块装token8、9，共2槽；块容量4，所以浪费2槽。
63. logical=$`\lfloor9/4\rfloor=2`$；offset=$`9\bmod4=1`$；table第2项是physical5。
64. 8 tokens需要$`8/4=2`$块。两请求不共享：$`2\times2\times100=400`$ bytes；共享：2块共200 bytes；省200 bytes，即50%。
65. 前：A、B都指8，block8 ref=2。A写时复制到new11：A指11，B指8；block8 ref从2降1，block11 ref=1。写发生在11，B看到的8不变。
66. block16时需$`\lceil18/16\rceil=2`$块，容量32，浪费14。block4时需$`\lceil18/4\rceil=5`$块，容量20，浪费2。虽少12槽，但块数2变5，metadata、映射读取和kernel间接寻址更多，所以不能只看碎片。
67. 无cache=30，有cache=10，比值3。大T时$`\sum t^2\approx T^3/3`$，量级$`O(T^3)`$；$`\sum t\approx T^2/2`$，量级$`O(T^2)`$。
68. K=40：$`2\times40\times5120\times128\times40=2{,}097{,}152{,}000`$。K=8：$`2\times40\times5120\times128\times8=419{,}430{,}400`$。减少$`1{,}677{,}721{,}600`$参数，正好5倍缩小。
69. Prefill：$`S_{old}=0,T=1024,S_{total}=1024`$，沿prompt token维并行，并采样第1枚输出$`y_1`$。为了采样第2枚的首个后续decode：输入$`y_1`$，$`S_{old}=1024,T=1,S_{total}=1025`$。为了采样第128枚的最后一个后续decode：此前已有prompt与前126枚输出进cache，所以$`S_{old}=1024+126=1150`$；输入$`y_{127}`$，$`T=1,S_{total}=1151`$，采样$`y_{128}`$。Decode单请求时间轴串行，但可跨请求batch并行。
70. completion fraction=$`900/1000=90\%`$。两个SLO都满足的goodput fraction=$`750/1000=75\%`$。只满足TTFT的800不能全算达标。
71. $`\hat x=0.1(56-4)=5.2`$。误差$`5.2-5.24=-0.04`$，绝对0.04。整数$`q=z=4`$时，反量化$`0.1(4-4)=0`$，所以zero point对应实数0。
72. 权重bytes从2GB到1GB，缩小2倍。但不能推出端到端2倍：dequant、unsupported kernel、activation/KV、调度和compute仍在，端到端由最慢部分决定。
73. ①在同一validation/workload测dense质量和hardware baseline；②用代表性calibration data算importance；③按结构预算prune；④用teacher做distillation/fine-tune repair；⑤复测quality、显存、真实latency/throughput/goodput，并与baseline同条件比较。
74. proposal A且接受仍贡献0.4；A被拒绝概率0.3。错误地从q重抽会再给A $`0.3\times0.4=0.12`$。最终A=$`0.4+0.12=0.52\ne0.4`$，所以不能从q直接重抽。
75. logical=$`\lfloor17/8\rfloor=2`$；offset=$`17\bmod8=1`$；table第2项physical2。地址=$`10000+1\times256=10256`$ bytes。题目把10000定义为该physical block起始地址，所以不再乘physical编号。
76. GQA压head数量，MLA压feature维，CLA压layer副本，local/CSA压sequence可见/缓存，quantization压每数bit；PagedAttention主要改分配和映射。它们改变不同轴，也有不同质量/kernel代价。
77. 先看：①排队分位数与准入/调度；②prefill kernel、prompt长度与prefill batch；③是否有长prefill阻塞decode或CPU/tokenization开销。ITL正常说明先别把主要精力放在逐token KV带宽。
78. ①GQA/MQA：少KV heads，可能损质量/需模型支持；②MLA：少feature cache，需相应架构/训练；③CLA：跨层共享，可能损质量；④local/CSA：少sequence历史直接访问，可能丢长程信息；⑤PagedAttention：降碎片但增metadata/kernel复杂度；⑥量化KV：减bytes但有误差和kernel要求；⑦降低batch/context：直接省内存但降吞吐/能力。
79. GQA→head；MLA→feature；CLA→layer；local/CSA→sequence；quantization→bit-width。
80. 先固定数据集、prompt/output分布和SLO，测correctness、质量、TTFT/ITL/E2E、HBM和Roofline证据。架构先试K=8 GQA并复算参数/KV与质量；数值格式试受支持的weight-only或W+A量化并校准；调度用continuous batching和PagedAttention控制KV/碎片，按goodput调batch。每一步只改一个因素，复测相同请求；speculative另测接受率与break-even；任何速度结果都要伴随质量和错误率。

---

## 23. 视频时间导航

下面每个链接使用一条尚未在正文使用的人工字幕 cue；全文 URL 秒点不重复。导航刻意覆盖开头到 85:24，而不是只集中在某一节。

| 时间 | 听什么 | 对应正文 |
|---|---|---|
| [00:21](https://www.youtube.com/watch?v=EfM546A79aM&t=21s) | 已训练模型、prompt、response | §2 |
| [05:05](https://www.youtube.com/watch?v=EfM546A79aM&t=305s) | “快”有多种 metrics | §2.3 |
| [10:06](https://www.youtube.com/watch?v=EfM546A79aM&t=606s) | contraction dimensions | §4.2 |
| [15:04](https://www.youtube.com/watch?v=EfM546A79aM&t=904s) | X 从 HBM 读取 | §5.3 |
| [20:12](https://www.youtube.com/watch?v=EfM546A79aM&t=1212s) | 课堂纠正 K 是 KV groups | §4.1 |
| [25:02](https://www.youtube.com/watch?v=EfM546A79aM&t=1502s) | KV cache 避免重算旧 token | §3.4 |
| [30:07](https://www.youtube.com/watch?v=EfM546A79aM&t=1807s) | matmul FLOPs 比 bytes 高一阶 | §6 |
| [35:13](https://www.youtube.com/watch?v=EfM546A79aM&t=2113s) | 回到 throughput、latency、TTFT | §2、§9 |
| [40:06](https://www.youtube.com/watch?v=EfM546A79aM&t=2406s) | memory/bandwidth latency 模型 | §8.6–§9 |
| [45:04](https://www.youtube.com/watch?v=EfM546A79aM&t=2704s) | prefill 完成后才开始生成 | §3.2、§3.5 |
| [50:06](https://www.youtube.com/watch?v=EfM546A79aM&t=3006s) | 增大 batch 与 OOM | §9.2 |
| [55:06](https://www.youtube.com/watch?v=EfM546A79aM&t=3306s) | MLA ablation 怎样读 | §11.5 |
| [60:14](https://www.youtube.com/watch?v=EfM546A79aM&t=3614s) | local 与 full 混合 | §12.1 |
| [65:19](https://www.youtube.com/watch?v=EfM546A79aM&t=3919s) | quantization-aware training | §13.5 |
| [70:00](https://www.youtube.com/watch?v=EfM546A79aM&t=4200s) | 怎样评估层的重要性与 calibration | §14.1、§14.3 |
| [75:04](https://www.youtube.com/watch?v=EfM546A79aM&t=4504s) | speculative rejection sampling | §15.2–§15.4 |
| [80:04](https://www.youtube.com/watch?v=EfM546A79aM&t=4804s) | fragmentation 类比 | §17.1 |
| [85:06](https://www.youtube.com/watch?v=EfM546A79aM&t=5106s) | 为 inference 重新设计架构 | §26 |

---

## 24. 源码函数、611 行与 22 张图片覆盖

### 24.1 连续源码行段索引

下面区间从 1 到 611 首尾相接，无 gap、无 overlap。“覆盖”表示每段已映射到正文；不表示正文逐字符抄写代码。

| 源码行 | 函数/内容 | 正文位置 |
|---|---|---|
| 1–15 | imports、lecture helper 与图片目录 | §0.4–§0.5、§24.3 |
| 16–62 | `main`：全讲调用顺序、课程景观/开场数字 | §0.3、§2、§25 |
| 63–99 | `landscape`：训练与推理成本、服务公司/市场快照 | §2.1、§25.2 |
| 100–117 | `review_transformer`：Transformer forward 复习 | §4.1–§4.3 |
| 118–161 | `review_of_arithmetic_intensity` | §5 |
| 162–262 | `arithmetic_intensity_of_inference`：shape、MLP、attention | §4、§6、§7 |
| 263–286 | `TransformerPerformanceStats` dataclass | §8.1、§8.6；line 267 勘误见§0.5 |
| 287–316 | `compute_transformer_performance_stats` | §8.2–§8.6 |
| 317–331 | `llama2_13b_config` | §8.3 |
| 332–370 | `throughput_and_latency` | §2.3、§8.6、§9 |
| 371–448 | `reduce_kv_cache_size`：GQA/MLA/CLA/local/DSA | §10–§12 |
| 449–489 | `quantization` | §13 |
| 490–506 | `model_pruning` 与源码未实现占位 | §14 |
| 507–552 | `speculative_sampling` | §15 |
| 553–575 | `continuous_batching` | §16 |
| 576–609 | `paged_attention`：碎片、块表、共享、kernel建议 | §17 |
| 610–611 | Python 入口 guard 与 `main()` | §0.5、本表 |

上述每个任务指定函数都有且只有一个直接正文映射：`main`、`landscape`、`review_transformer`、`review_of_arithmetic_intensity`、`arithmetic_intensity_of_inference`、`compute_transformer_performance_stats`、`throughput_and_latency`、`reduce_kv_cache_size`、`quantization`、`model_pruning`、`speculative_sampling`、`continuous_batching`、`paged_attention`。dataclass 也单独列出。

### 24.2 22 张本地图逐图视觉核验

全部图片都从任务指定 commit 的本地 `images/` 目录按原分辨率查看；表中不是从文件名猜结论。

| # | 图片（像素） | 目视核验到的内容 | 正文 |
|---:|---|---|---|
| 1 | `inference-schema.png` 1498×519 | online/offline 请求与模型服务流程图 | §2.2 |
| 2 | `gqa-speed.png` 789×490 | 横轴 KV groups、纵轴生成时间；向 MHA 端上升 | §10.3 |
| 3 | `gqa-accuracy.png` 1501×318 | 不同 GQA 设置的质量表；不是“永不掉点” | §10.3 |
| 4 | `mla-schema.png` 1853×525 | hidden→低维 latent→up projections 的缓存路径 | §11.1 |
| 5 | `mla-accuracy.png` 1886×673 | MLA ablation 质量表 | §11.5 |
| 6 | `mla-accuracy2.png` 912×298 | 另一组 MLA/rope component 对照 | §11.5 |
| 7 | `cla-diagram.png` 795×820 | 多层共享 K/V 的 cross-layer arrows | §11.6 |
| 8 | `cla-results.png` 1119×838 | 横轴 KV bytes/token（log scale）、纵轴 perplexity，展示 Pareto 权衡 | §11.6 |
| 9 | `longformer-attention.png` 1861×421 | full、window、dilated、global+window 四种邻接图 | §12.1 |
| 10 | `deepseek-v4-attention.png` 1726×871 | token compressor、lightning indexer、MQA scores、top-k 与 sliding window 汇合 | §12.3 |
| 11 | `awq-schema.png` 2632×651 | naïve INT3、保留1% FP16的硬件代价、activation-aware scaling 最终路线 | §13.5 |
| 12 | `pruning-kd-loop.png` 2196×941 | importance→rank→trim→distill 的循环 | §14.4 |
| 13 | `pruning-kd.png` 944×762 | model-size/accuracy 的剪枝蒸馏结果图 | §14.4 |
| 14 | `speculative-sampling-algorithm.png` 1943×1376 | proposal、accept/reject、corrected distribution 算法 | §15.2–§15.5 |
| 15 | `speculative-sampling-results.png` 1962×772 | 不同模型/任务的速度结果表 | §15.6 |
| 16 | `speculative-sampling-stats.png` 1943×820 | $`K`$、接受长度与开销统计 | §15.5–§15.6 |
| 17 | `medusa-eagle.png` 1168×397 | 多heads候选与feature-level draft概览 | §15.8 |
| 18 | `paged-attention-fragmentation.png` 2601×453 | reserved/used/internal/external fragmentation | §17.1 |
| 19 | `paged-attention-blocks.png` 1240×570 | 固定大小 physical KV blocks | §17.2 |
| 20 | `paged-attention-logical.png` 1457×642 | logical blocks 与物理不连续映射 | §17.2–§17.3 |
| 21 | `paged-attention-sharing.png` 1595×637 | shared prefix、reference、copy-on-write | §17.4 |
| 22 | `paged-attention-parallel.png` 1496×688 | 两个 sample 共享 prefix 后在 fathers/mothers 分叉；COW 把引用计数从2降到1并复制物理块 | §17.4、§17.6 |

源码还引用 6 个远程图 URL（Transformer 结构、naive/cached inference、GQA 示意、FP8 图、static-batching 图）；它们不是上述 22 张本地资产，未混入“22 张”计数。动态网页可能变化，因此正文关键公式均由源码、字幕或本地静态图支持。

### 24.3 环境验证边界

本笔记做了纯 CPU 的整数/浮点复算、文本结构检查与图片原尺寸目视检查。当前环境没有用 H100 跑 serving benchmark，也没有声称实测 CUDA kernel、NCCL、vLLM 或某个量化 kernel。所有 `memory/bandwidth` 数字都明确叫“理想带宽下界”。

---

## 25. 来源、课程边界与一手补充

### 25.1 课程来源

- [官方 `lecture_10.py`（固定 commit）](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py)：函数、公式、配置和22张本地图的主来源。
- [Stanford Online 视频](https://www.youtube.com/watch?v=EfM546A79aM)：人工 `English (United States)` 字幕1509段，00:05–85:24；用于口头解释、问答与源码文字纠错。
- 本讲无 PDF。源码中的公司用量、硬件、模型/论文结果是课程在该 commit 时采用的快照，不自动代表 2026-08-28 之后的最新事实。

### 25.2 一手补充来源

- [NVIDIA H100 官方规格](https://www.nvidia.com/en-sg/data-center/h100/)：3.35 TB/s、Tensor Core peak及structured-sparsity脚注。
- [NVIDIA Transformer Engine FP8 官方说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)：E4M3/E5M2 动态范围；用来限定课件±240的格式边界。
- [GQA 原论文](https://arxiv.org/abs/2305.13245)、[DeepSeek-V2/MLA 技术报告](https://arxiv.org/abs/2405.04434)、[CLA 原论文](https://arxiv.org/abs/2405.12981)、[Longformer](https://arxiv.org/abs/2004.05150)：只支持对应机制/实验语境，不证明所有模型都同样掉点或提速。
- [DeepSeek-V4-Pro 官方 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)：查询于2026-08-28；用于 CSA/HCA/1M context 与官方效率摘要。正文没有凭缩写猜 DSA/HCA 的关系。
- [AWQ 原论文](https://arxiv.org/abs/2306.00978)、[NVIDIA Minitron 技术报告](https://arxiv.org/abs/2408.11796)：量化、剪枝和蒸馏补充。
- [Speculative decoding](https://arxiv.org/abs/2211.17192)、[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)：proposal/target、接受率与候选机制。
- [Orca](https://arxiv.org/abs/2206.02658)：iteration-level/selective batching；[PagedAttention/vLLM论文](https://arxiv.org/abs/2309.06180)与[vLLM官方文档](https://docs.vllm.ai/)：分页KV、block table与当前实现边界。

课程没有展开、论文也随版本变化的实现细节，本文标成【延伸】；补充来源用于核边界，不把论文结果冒充老师逐字说法。

---

## 26. 一页复习流程与学完能力清单

### 26.1 一页复习流程

1. 画 request 时间线：排队→prefill→第一枚→逐枚decode→最后一枚。
2. 写清指标：TTFT、ITL、E2E、throughput、goodput、SLO。
3. 写 shape：`Q [B,T,N,H]`、merged `K/V [B,S_total,K,H]`、score `[B,T,N,S_total]`，检查 $`S_{total}=S_{old}+T`$ 与 $`N=KG`$。
4. 算工作与流量：matmul、MLP、attention 的 FLOPs/bytes，再看 Roofline。
5. 算显存：parameter bytes + $`B\times`$per-sequence KV bytes + 运行时余量。
6. 找压缩轴：GQA/head、MLA/feature、CLA/layer、local/CSA/sequence、quant/bit。
7. 找系统浪费：排队/空batch→continuous；碎片→PagedAttention；串行decode→speculative。
8. 最后才下结论：同条件复测质量、TTFT/ITL/E2E分位数、throughput/goodput、HBM。

### 26.2 你现在应该能做到

- 不把 latency、ITL、throughput 和 goodput 混为一谈。
- 从 $`[B,D]@[D,F]`$ 自己推到 FLOPs、bytes 与 intensity。
- 从 $`B,S_{old},T,S_{total},D,F`$ 自己推出 MLP/attention 的课程公式。
- 完整复算 Llama 2 13B 的参数、KV、GB/GiB、理想 latency 与 throughput。
- 解释源码 GQA “Worse latency”为什么是文字矛盾。
- 说清 GQA、MLA、CLA、local/CSA 与 quantization 分别压哪一轴。
- 逐步量化/反量化一个数，并解释为什么更小不保证更快。
- 用二词和三词事件树证明 speculative sampling 的 exactness。
- 画 continuous batching 请求表和 PagedAttention block table/COW。
- 面对新优化先问：省的是 FLOPs、bytes、cache、碎片还是排队？质量和SLO付了什么代价？

> **最后一句：** Transformer 的训练形状适合大矩阵并行，逐 token inference 却常是薄矩阵、频繁搬权重和不断增长的 KV。整讲所有技术，都在重新安排“算多少、搬多少、存多少、等多久”。
