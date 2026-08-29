# CS336 Notes for Dumb People

Stanford CS336（Spring 2026）中文学习笔记，面向第一次系统学习语言模型的读者。

这套笔记以官方课程材料和 Stanford Online 视频为主线，把课堂中默认读者已经掌握的概念拆开解释，并补充必要的推导、手算、代码阅读提示和工程背景。当前覆盖课程的全部 17 讲。

> 本项目是独立整理的非官方学习资料，不代表 Stanford 或课程教师。

## 从这里开始

- **第一次学习 CS336：** 从 [Lecture 1：课程全景与 Tokenization](notes/lecture_01_overview_tokenization.md) 开始，按讲次顺序阅读。
- **只想查某个主题：** 使用下面的课程地图，或浏览[完整讲义目录](notes/README.md)。
- **想对照原课：** 查看[官方讲义、视频与本仓库笔记的对应关系](SOURCES.md)。

## 课程地图

| 学习阶段 | 讲义 | 核心主题 |
|---|---|---|
| 基础与模型 | [Lecture 1](notes/lecture_01_overview_tokenization.md) · [Lecture 2](notes/lecture_02_pytorch_einops_resource_accounting.md) · [Lecture 3](notes/lecture_03_architectures_hyperparameters.md) | Tokenization、PyTorch / einops、资源核算、Transformer 架构与超参数 |
| 高效架构与计算系统 | [Lecture 4](notes/lecture_04_attention_alternatives_moe.md) · [Lecture 5](notes/lecture_05_gpus_tpus.md) · [Lecture 6](notes/lecture_06_kernels_triton.md) · [Lecture 7](notes/lecture_07_parallelism_1.md) · [Lecture 8](notes/lecture_08_parallelism_2.md) | Attention alternatives、MoE、GPU / TPU、Triton 与分布式并行 |
| 规模、推理与评测 | [Lecture 9](notes/lecture_09_scaling_laws_1.md) · [Lecture 10](notes/lecture_10_inference.md) · [Lecture 11](notes/lecture_11_scaling_laws_2_mup.md) · [Lecture 12](notes/lecture_12_evaluation.md) | Scaling laws、推理系统、μP 与模型评测 |
| 训练数据 | [Lecture 13](notes/lecture_13_data_sources_datasets.md) · [Lecture 14](notes/lecture_14_data_filtering_dedup_mixing_synthetic.md) | 数据来源、版权与血缘、过滤、去重、混合和合成数据 |
| 后训练与多模态 | [Lecture 15](notes/lecture_15_post_training_sft_rlhf.md) · [Lecture 16](notes/lecture_16_rlvr_grpo_reasoning.md) · [Lecture 17](notes/lecture_17_multimodal_models.md) | SFT、RLHF、PPO、DPO、RLVR、GRPO、推理模型与多模态模型 |

## 每讲有什么

每篇讲义都尽量做到可以独立阅读，通常包括：

- **五分钟复习卡：** 学完之后快速回忆核心概念；
- **前置知识：** 补齐本讲默认使用的数学、系统或模型背景；
- **逐步讲解：** 拆解公式、张量 shape、算法流程与资源开销；
- **手算与例子：** 用小规模数字验证抽象结论；
- **常见误区：** 区分容易混淆的概念和适用边界；
- **自测题与答案：** 检查是否真正掌握主线；
- **视频导航与来源：** 通过时间戳返回课堂原始上下文。

## 推荐阅读方式

1. 第一次学习时跳过开头的“五分钟复习卡”，直接从前置知识和正文开始。
2. 遇到公式或 shape 变化时，先暂停并自己重算，再继续看答案。
3. 完成自测题后再展开答案；答错时回到对应小节，而不是只记结论。
4. 对存在版本、硬件或实验条件限制的结论，沿时间戳和来源链接查看原始语境。
5. 学完一讲后再用复习卡压缩记忆，并按需要进入下一讲。

## 内容来源与标注

正文优先使用以下官方材料：

- [Stanford CS336 课程主页](https://cs336.stanford.edu/)
- [官方课程材料仓库](https://github.com/stanford-cs336/lectures)
- [Stanford Online 视频播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

笔记中的标注用于区分信息边界：

- **课程内容：** 能在官方讲义、代码、视频或课堂问答中找到；
- **补充解释：** 为降低前置知识要求而加入的拆步推导、类比或小例子；
- **延伸知识：** 不属于课堂主线，但有助于建立完整理解的背景内容。

逐讲的官方材料和视频链接见 [`SOURCES.md`](SOURCES.md)。第三方材料、论文、代码和视频的版权归各自作者所有。

## 参与改进

欢迎通过 Issue 或 Pull Request 修正事实错误、失效链接、推导跳步和 GitHub 渲染问题。修改讲义后可以运行：

```bash
python3 scripts/validate_notes.py
```

这个脚本只检查文档结构和一部分 Markdown 兼容性；检查通过不代表内容事实或教学表达一定正确，仍需要人工阅读确认。
