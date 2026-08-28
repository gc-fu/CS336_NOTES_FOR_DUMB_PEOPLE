# CS336 Notes for Dumb People

面向完全初学者的 Stanford CS336（Spring 2026）中文重构笔记。

**状态：Lecture 1–17 全部完成，共经过 50 轮 Beginner Review，机械校验全部通过。**

完整的官方课程材料与视频对应关系见 [`SOURCES.md`](SOURCES.md)。

这不是视频字幕的简单翻译。每一讲都综合：

- 官方讲义或 executable lecture；
- 官方 YouTube 课程视频及字幕；
- 为补齐前置知识而增加的推导、手算和工程例子；
- 同一位 Beginner Reviewer 的逐轮学习反馈。

目标是让读者在不观看完整视频的情况下，仍能获得完整课程体验，并能够独立完成核心推导与自测。

## 阅读方式

每讲开头的“五分钟复习卡”只适合学完后回忆。第一次学习请直接从正文第 1 节开始。

## 课程目录

完整目录见 [notes/README.md](notes/README.md)。

机器可检查的 Markdown 完整性可运行 `python scripts/validate_notes.py`；机器通过不能代替 Beginner Reviewer 的逐讲学习审核。

## 内容边界

- “课程内容”来自讲义、视频或课堂问答。
- “补充”是为了让零基础读者形成闭环而加入的解释。
- 笔记是独立的学习资料，不是 Stanford 官方发布。
- 第三方讲义、论文、代码与视频版权归原作者所有；仓库只保存原创重构文字与必要的短代码/公式。

## 审核标准

每讲经过以下循环：

1. Primary Writer 结合讲义与完整字幕编写；
2. Beginner Reviewer 以极低前置知识从头学习；
3. 对所有理解障碍补充定义、推导或例子；
4. 重复审核，直到 reviewer 明确确认没有阻碍主线或自测的问题。
