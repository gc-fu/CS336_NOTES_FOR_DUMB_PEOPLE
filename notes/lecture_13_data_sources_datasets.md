# Lecture 13：Data Sources & Datasets——训练数据不是“从互联网自动掉下来”

> CS336 Spring 2026，Lecture 13: Data I。  
> 官方视频：[Stanford Online](https://www.youtube.com/watch?v=-qm0ln33G24)。  
> 官方可执行讲义：[lecture_13.py 固定版本](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_13.py)。  
> 本笔记结合 622 行官方源码、完整人工英文字幕、全部讲义图片与一手论文/官方法律资料，目标是让零基础读者不用看视频也能学懂。

> **法律边界：**本讲涉及美国版权法与正在变化的诉讼，只用于课程学习，**不是法律意见**。版权期限、许可、合同和 fair use（合理使用）会因作品、日期、事实与法域而不同。真实项目必须让具备相关法域资格的律师审查。

> **第一次阅读：**
>
> 1. 跳过 §1 的五分钟复习卡，从 §0、§2 开始。
> 2. 看到 B、GB、TB、token、document 时先看 §23 的单位卡，别把不同单位直接相加。
> 3. 【课程内容】来自讲义；【视频补充】来自课堂口头解释；【补充解释】是本笔记拆小的教学；【补充】来自一手论文或官方资料；【延伸】首次可跳过。
> 4. 课程中的网页规模、模型与案件状态是 **2026 年课程快照**；本文会标出能独立核验与不能独立核验之处。
> 5. §24–§25 有 80 道题和完整答案；建议学完一节就做对应题。

---

<a id="s0"></a>
## 0. 导航、学习目标与最低词汇

### 0.1 可点击目录

- [§1 五分钟复习卡](#s1)
- [§2 数据为什么是模型差异化核心](#s2)
- [§3 从 live service 到训练 dataset](#s3)
- [§4 爬虫与访问限制](#s4)
- [§5 版权、许可、ToS 与 fair use](#s5)
- [§6 课程诉讼快照](#s6)
- [§7 Common Crawl、WARC、WET 与 HTML 提取](#s7)
- [§8 Wikipedia、GitHub、Software Heritage、arXiv](#s8)
- [§9 BERT 到 GPT-3](#s9)
- [§10 The Pile 及其来源](#s10)
- [§11 Gopher、LLaMA、RedPajama、SlimPajama](#s11)
- [§12 RefinedWeb、FineWeb、Dolma](#s12)
- [§13 DCLM 的 classifier filtering](#s13)
- [§14 Nemotron-CC](#s14)
- [§15 The Stack 与 Stack v2](#s15)
- [§16 CommonPile 与许可链](#s16)
- [§17 去重、分类器、MinHash、Bloom filter](#s17)
- [§18 PII、偏差、poisoning 与防守性验证](#s18)
- [§19 数据集血缘时间线](#s19)
- [§20 “能否纳入”决策树](#s20)
- [§21 Dataset provenance card](#s21)
- [§22 常见误区](#s22)
- [§23 公式与单位卡](#s23)
- [§24 自测题](#s24)
- [§25 自测答案](#s25)
- [§26 视频导航](#s26)
- [§27 源码覆盖](#s27)
- [§28 图片覆盖](#s28)
- [§29 来源与证据边界](#s29)
- [§30 学完后的能力](#s30)

### 0.2 本讲的因果链

视频 [00:04](https://www.youtube.com/watch?v=-qm0ln33G24&t=4s) 从 data（数据）开始。全讲不是背数据集名字，而是追一条链：

    活着的网站、仓库、论文库
      -> 获得一份 raw copy
      -> 把 HTML/PDF/repository 变成文本或结构化序列
      -> 过滤、去重、脱敏、选择和混合
      -> 得到有版本的数据集
      -> 用它训练模型
      -> 模型能力、偏差、风险和法律暴露都受上游决定影响

最重要的两句话：

1. **访问得到，不等于允许使用。**
2. **同一个 source（来源）经过不同 processing（处理），会变成很不一样的 dataset（数据集）。**

### 0.3 最低词汇与前置知识

- **Data（数据）**：训练算法实际读取的记录，如文本、代码或对话。
- **Source（来源）**：数据最初从哪里产生，如 Wikipedia、GitHub、arXiv。
- **Dataset（数据集）**：在特定版本、许可与处理规则下整理出的有限记录集合。
- **Document（文档）**：处理流水线的一条基本记录，可能是一页网页、一篇论文或一个代码文件。
- **Token（词元）**：tokenizer 把文本切成的模型输入编号；一个 token 不等于一个字符或一个单词。
- **Raw（原始）**：尽量保留来源形态的副本；raw 仍可能已经被 crawler 选择过，不代表绝对无处理。
- **Processing（处理）**：提取、过滤、去重、脱敏、格式化和混合。
- **Provenance（来源血缘）**：一条记录从哪里来、何时取得、经过什么处理、依据什么许可。
- **Snapshot（快照）**：某个时间点的状态；网站和许可后来可能改变。
- **Open weights（开放权重）**：可以下载模型参数；不自动代表训练代码、数据或许可细节都开放。

---

<a id="s1"></a>
## 1. 五分钟复习卡（首次阅读跳过）

1. 模型权重开放不等于训练数据开放；数据常是竞争优势和法律风险集中处。（§2）
2. Pre-training、mid-training、post-training 通常从大量较杂数据走向较少但更针对的数据，边界并非法律或科学定律。（§2）
3. Internet、public web、技术可访问、合同允许、取得许可是五个不同集合。（§3–5）
4. Crawler 用 seed URL 初始化 frontier，下载页面，再把新链接放回 frontier。（§4）
5. robots.txt 是 crawler 协议线索；ToS 是合同层；license/copyright 是权利层；privacy 又是另一层。（§4–6）
6. 美国当前一般期限不是“一律 75 年”：1978 年后自然人作品通常作者终身加 70 年；职务、匿名、笔名作品常按发表 95 年或创作 120 年取较短者。（§5）
7. Fair use 必须综合四因素，不能用“研究”“transformative”一个词自动得到答案。（§5）
8. Bartz 与 Kadrey 的 2025 summary judgment 只处理特定原告、证据与请求；训练使用与盗版取得/保存必须分开。（§6）
9. WARC 保留 HTTP response；WET 是抽取后的纯文本，体积小但有损。（§7）
10. HTML→text 会决定保留正文还是导航、广告和模板垃圾，因此会影响下游模型。（§7）
11. Wikipedia dump、Git clone、arXiv bulk download 往往比爬网页界面更稳定；但每条作品的许可仍需检查。（§8）
12. Exact duplicate 是字节/规范化后完全相同；near duplicate 是改过少量字仍高度相似。（§17）
13. 在 quality task（正类=有用、预测为正=留下）中，false positive 会把坏数据留下；false negative 会误删好数据，可能尤其伤害低资源语言或方言。（§13、§17）
14. DCLM 的“质量”来自正负样本定义；selection bias 会把“像 OpenHermes/ELI5”偷换成“普遍高质量”。（§13）
15. Collection license 不自动覆盖每个成员作品；synthetic data 也要追模型、提示、来源和输出许可。（§16）
16. Dataset card/provenance card 应记录 source、时间、许可、处理、PII、去重、已知缺口和可撤回机制。（§21）

---

<a id="s2"></a>
## 2. 数据为什么是模型差异化核心

### 2.1 “给定数据怎么训练”之后，才轮到“数据从哪来”

【课程内容，源码 6–47；视频 [00:24](https://www.youtube.com/watch?v=-qm0ln33G24&t=24s)】前 12 讲大量讨论 architecture、optimization、parallelism、inference 和 evaluation。这些都隐含一个前提：已有训练数据。

Lecture 13/14 改问：

- 数据最初由谁产生？
- 怎样拿到一份稳定副本？
- 怎样从网页、PDF、代码仓库变成 token？
- 什么应保留，什么应删除？
- 是否具有访问、复制、再分发和训练所需权利？

### 2.2 Open weights 不等于 open data

【课程内容，源码 48–67；视频 [00:35](https://www.youtube.com/watch?v=-qm0ln33G24&t=35s)】Llama 3 报告会公开架构与训练过程，但对 pre-training data 只给很高层描述。原因至少两类：

1. **Competitive dynamics（竞争因素）**：数据来源和处理 recipe 可以是效果差异。
2. **Copyright liability（版权责任风险）**：公开具体来源可能暴露争议取得或使用。

逻辑不要倒过来：

- “没公开”不能证明违法；
- “开源权重”也不能证明数据可复现；
- 架构相同不代表数据、tokenizer、训练顺序相同。

视频 [01:21](https://www.youtube.com/watch?v=-qm0ln33G24&t=81s) 说数据工作是 long-tail problem（长尾问题）：常见页面容易处理，少数异常格式、语言、许可、重复和隐私情况会不断出现，仍需要大量人力。

### 2.3 Pre、mid、post-training

【课程内容，源码 68–89；视频 [02:18](https://www.youtube.com/watch?v=-qm0ln33G24&t=138s)】

| 阶段 | 人话 | 常见输入 | 目标 |
|---|---|---|---|
| pre-training | 先读大量材料学语言和世界模式 | 网页、书、论文、代码 | 建 base capabilities |
| mid-training | 再集中读高质量或目标能力材料 | 高质量网页、数学、长上下文、合成数据 | 加强特定能力 |
| post-training | 学会按用户意图回答和行动 | 指令、偏好、RL 环境、chat transcript | 形成 instruct/chat behavior |

**Base model（基础模型）**通常指 pre+mid 后、post 前的模型；**instruct/chat model（指令/聊天模型）**通常指 post-training 后的模型。视频 [03:13](https://www.youtube.com/watch?v=-qm0ln33G24&t=193s) 提醒这些边界越来越模糊，部分大模型只发布 instruct checkpoint。

【补充解释】“数据量越来越少、质量越来越高”只是常见趋势：

- pre：3T token；
- mid：100B token；
- post：1M 条对话。

这里 T=trillion=万亿，B=billion=十亿。不能说 1 条对话一定比 1 个网页 token “质量高”；质量必须相对训练目标定义。

### 2.4 OLMo 图教我们读什么

源码三张表分别展示 OLMo 2 pre-training mix、Dolmino mid-training mix 和 Tülu post-training prompt mix：

- pre-training 表同时给 tokens、words、bytes、documents，说明单位不能互换；
- mid-training 把高质量网页、论文、百科与 synthetic math 分开；
- post-training 又区分 SFT（supervised fine-tuning，监督微调）和 DPO（Direct Preference Optimization，直接偏好优化）所用 prompts。

【补充解释】一张数据表至少问：

1. raw size 还是重复采样后的 effective size？
2. tokens 用哪个 tokenizer？
3. document 怎样切分？
4. 数据版本日期是什么？
5. 合成数据由哪个模型生成？

---

<a id="s3"></a>
## 3. 从 live service 到训练 dataset

### 3.1 Internet、web、public web 不是同义词

【课程内容，源码 90–149；视频 [04:42](https://www.youtube.com/watch?v=-qm0ln33G24&t=282s)】

- **Internet（互联网）**：全球互联网络基础设施，还包括邮件、私有 API、点对点协议等。
- **World Wide Web（万维网）**：通过 HTTP/HTTPS 和 URL 访问的资源。
- **Public web（公开网页）**：无需特定账号就能从浏览器看到的部分。
- **Accessible（技术可访问）**：你的程序当前能拿到 response。
- **Permitted（被允许）**：合同、许可、版权、隐私与其他法律层允许特定动作。

集合关系不是“公开=随便用”。一页免费可看文章可以同时是：

- 公开网页；
- 技术可下载；
- 受版权保护；
- ToS 禁止自动抓取；
- 含个人信息；
- 未授权再分发。

### 3.2 Live server 不能直接当静态训练文件

**Live service（在线服务）**是会随请求、账户、时间和交互改变的服务器。视频 [05:16](https://www.youtube.com/watch?v=-qm0ln33G24&t=316s) 解释：pre-training 通常先做 snapshot，训练程序不会在每一步现场访问所有网站。

最小流水线：

1. 请求 URL；
2. 保存 response 与时间、状态码、headers；
3. 解析 HTML；
4. 抽正文；
5. 过滤与去重；
6. 切 document、tokenize；
7. 固定 dataset version。

因此 “trained on website X” 至少还缺 acquisition date、页面集合、抽取器和过滤规则。

### 3.3 Dynamic、auth、paywall

【课程内容；视频 [06:05](https://www.youtube.com/watch?v=-qm0ln33G24&t=365s)】

- **Dynamic content（动态内容）**：URL 不变，但点击、滚动、表单或 JavaScript 状态决定内容。
- **Authentication（身份认证）**：需账号或凭证。
- **Paywall（付费墙）**：需订阅或付款。
- **Deep web（深网）**：普通链接爬取不能直接索引的内容；不等于犯罪网络。

Discord、Facebook、X、LinkedIn、付费新闻站都说明“存在于 web”不等于普通 crawler 可取得。

---

<a id="s4"></a>
## 4. Crawler：seed、frontier、下载与限制

### 4.1 四步最小 crawler

**Crawler（网络爬虫）**是自动发现并下载网页的程序。视频 [05:36](https://www.youtube.com/watch?v=-qm0ln33G24&t=336s) 给出核心：

1. **Seed set（种子集合）**：起始 URL，如 A、B。
2. **Frontier（待抓取队列）**：尚未访问的 URL 集。
3. 下载 frontier 中一个 URL。
4. 解析超链接，把新的 URL 加回 frontier。

小例：

| 步 | 取出 | 新发现 | frontier 结束状态 |
|---:|---|---|---|
| 0 | 无 | seed={A,B} | [A,B] |
| 1 | A | C,D | [B,C,D] |
| 2 | B | D,E | [C,D,E]，D 不重复加 |
| 3 | C | 无 | [D,E] |

若不做 URL 去重，A→B、B→A 会让队列循环。

### 4.2 三类技术限制

【课程内容；视频 [07:43](https://www.youtube.com/watch?v=-qm0ln33G24&t=463s)】

- **robots.txt**：站点发布给 crawler 的路径规则；它本身不是版权许可，也不等于各法域下完整法律结论。
- **Rate limit（速率限制）**：单位时间只允许有限请求，防止服务器过载。
- **CAPTCHA**：让用户证明是人，常用来阻止自动程序。
- **IP/geoblock**：按网络地址或地区阻止访问。

服务器例：站点最多 2 requests/s，你发 100 requests/s。

- 超出倍数：$`100/2=50`$；
- 可能造成额外费用和服务下降；
- “我能绕过”不是“我应绕过”。

### 4.3 Selection、politeness、revisit

【课程内容，Common Crawl 部分；视频 [34:06](https://www.youtube.com/watch?v=-qm0ln33G24&t=2046s)】

- **Selection policy（选择策略）**：抓哪些页面。
- **Politeness policy（礼貌策略）**：请求间隔、并发、robots 处理。
- **Revisit policy（重访策略）**：多久检查更新。

如果新闻页每天变、历史 PDF 十年不变，全部每小时重抓会浪费资源。策略本身也造成 sampling bias：被 seed 或高链接度覆盖的网站更易进入 dataset。

### 4.4 访问规则的四层

| 层 | 问题 | 例子 |
|---|---|---|
| 技术 | 能不能拿到 bytes？ | 登录、CAPTCHA、IP block |
| 网站协议 | 站点怎样请求 crawler 行为？ | robots.txt、crawl-delay |
| 合同 | 使用服务时承诺什么？ | Terms of Service |
| 权利与法规 | 能否复制、训练、再分发、处理个人数据？ | copyright、license、privacy |

同一动作可能同时触及多层。robots 允许不代表获得 copyright license；CC license 也不自动取消网站 ToS。

### 4.5 Decline of consent 图怎么读

源码引用研究观察 C4、RefinedWeb、Dolma URL 的 robots/ToS 限制随时间增加。图的纵轴是 URL 比例，横轴是年份；多条竖线标志 ChatGPT、GPTBot 等时期。

正确结论：在该研究样本与分类规则中，限制比例增长。不能推出：

- 全世界精确有多少网页禁止 AI；
- 每个 robots 指令都等同法律禁令；
- 2019 抓到的副本今天一定可继续使用。

### 4.6 Shadow libraries 只讲风险

**Shadow library（影子图书馆）**聚合大量未经权利人授权的书或论文，并可能绕过付费墙。课程列 LibGen、Z-Library、Anna's Archive、Sci-Hub。

本笔记不提供访问或绕过操作。风险拆开：

1. acquisition：副本怎样取得；
2. storage：是否建立永久库；
3. training：哪些副本实际用于训练；
4. redistribution：是否把副本发给别人。

这四步可能有不同法律分析，不能用“最终只训练模型”抹掉前面的取得行为。

---

<a id="s5"></a>
## 5. 版权、public domain、license、ToS 与 fair use

### 5.1 法律词汇先分开

【课程内容，源码 150–240；视频 [14:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=868s)】

【官方一手资料】美国版权局的 [What Is Copyright?](https://www.copyright.gov/what-is-copyright/) 解释版权保护的基本对象与权利；以下只讨论美国法课程语境。

- **Copyright（版权）**：保护原创表达的法定权利集合，包括复制等专有权。
- **Expression（表达）**：具体文字、代码、图像或结构；与抽象 idea（思想）区分。
- **Fixed（固定）**：作品被记录在可感知或复制的载体中，如文件或纸。
- **Original（原创）**：独立创作并有最低创造性；不等于从未有人想到。
- **Registration（登记）**：向版权局登记。美国当前保护通常从原创作品固定时发生；登记不是保护成立的一般前提，但对美国作品提起侵权诉讼通常需登记或被拒。
- **Public domain（公有领域）**：不受版权专有权限制的材料，如期限届满作品、不可版权的事实/思想、部分美国联邦政府职务作品。
- **License（许可）**：权利人授权特定行为及条件。
- **Terms of Service, ToS（服务条款）**：使用网站服务时的合同规则；它和版权许可不是同一问题。

### 5.2 Expression 不等于 idea

例：quicksort（快速排序）的抽象步骤属于 idea/procedure；某人写下的具体源代码是 expression。

- 重新实现算法：可能不复制原代码表达；
- 整段复制代码：涉及具体表达；
- 改几个变量名：仍可能保留大量表达。

“模型没逐字输出”只能减少某些复制证据，不能自动解决所有版权、衍生表达或市场问题。

### 5.3 “一律 75 年”是课程口语化过度简化

【美国官方核验】[17 U.S.C. §302 官方条文](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A302+edition%3Aprelim%29) 与美国版权局 [Circular 15A: Duration of Copyright](https://www.copyright.gov/circs/circ15a.pdf)：

- 1978-01-01 后自然人作品：通常作者终身加 70 年；
- joint work：最后去世作者终身加 70 年；
- anonymous、pseudonymous、work made for hire：首次发表 95 年或创作 120 年，取先到期者；
- 1978 年前作品适用更复杂的发表、登记和续展规则；
- 其他国家/地区期限可能不同。

数字例：某作者 2000 年创作，2050 年去世。只按一般规则：

```math
2050+70=2120.
```

不是从创作年算 75 年的 2075。若是 2000 年发表的 work made for hire，候选到期年：

```math
2000+95=2095,\qquad 2000+120=2120,
```

取较早的 2095。真实期限仍需查作品类型、首次发表、作者身份和历史规则。

### 5.4 Creative Commons 不是一个单一“随便用”按钮

**Creative Commons, CC（知识共享许可）**是一族标准许可：

- BY：要求署名；
- SA：衍生作品按相同许可；
- NC：限制商业使用；
- ND：限制改编；
- CC0：尽量放弃权利、接近公有领域工具。

必须保存 exact license、version、URL、取得日期。说“这是 CC”仍然不够。

### 5.5 Fair use 四因素

【美国法，[17 U.S.C. §107 官方条文](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A107+edition%3Aprelim%29)；视频 [21:21](https://www.youtube.com/watch?v=-qm0ln33G24&t=1281s)】**Fair use（合理使用）**允许某些未经许可的使用，但要综合具体事实；美国版权局 [Fair Use Index](https://www.copyright.gov/fair-use/) 也强调具体案件事实：

1. use 的 purpose and character，包括商业性、教育性与 transformative 性；
2. 原作品 nature，事实性还是高度创作性；
3. 使用 portion 的 amount and substantiality；
4. 对原作品实际或潜在市场的 effect。

四因素不是打勾计分：

| 场景 | 因素 1 | 因素 2 | 因素 3 | 因素 4 | 能否自动下结论 |
|---|---|---|---|---|---|
| 教师展示短引文评论 | 教育/评论较有利 | 视作品而定 | 少量较有利 | 替代市场弱 | 仍需综合 |
| 商业服务复制整本小说 | 商业但可能主张转化 | 创作性强 | 全部 | 市场证据关键 | 不能只靠“AI” |

**Transformative（转化性）**是用途或意义是否不同，不是“用神经网络处理过”就自动满足。

### 5.6 ToS 是独立层

视频 [27:03](https://www.youtube.com/watch?v=-qm0ln33G24&t=1623s) 用 YouTube 类比：内容可能采用 CC，但网站服务条款仍可能限制自动下载。

决策必须分别问：

1. 我有访问账号/权限吗？
2. ToS 是否允许 bot、批量下载和训练？
3. 每个作品的许可是否覆盖复制/商业使用/再分发？
4. 若无许可，是否有经过律师审查的例外主张？
5. 是否含 PII、保密信息或受其他法规约束数据？

---

<a id="s6"></a>
## 6. Bartz、Kadrey、NYT：只读成特定案件快照

### 6.1 Summary judgment 是什么

**Summary judgment（简易判决）**是法院认为对某请求不存在需要陪审团审理的重大事实争议，依现有记录依法裁判。它不是：

- 最高法院对所有 AI 训练立下全国统一规则；
- 对案件里所有行为都判同一结果；
- 对不在 record（案卷证据）中的未来模型自动适用。

### 6.2 Bartz v. Anthropic

【课程内容，视频 [28:08](https://www.youtube.com/watch?v=-qm0ln33G24&t=1688s)；2025 法院 order 核对】可回查 [Justia Dockets 的 Filing 231 全文页](https://docs.justia.com/cases/federal/district-courts/california/candce/3%3A2024cv05417/434709/231)、[CourtListener RECAP docket](https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/) 与 [Document 231 的 RECAP PDF 镜像](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf)。Justia/RECAP 是法院提交文件的公开镜像，不是法院官网；本笔记不声称保存或核对过一个未列出的固定本地副本。

法院把不同用途拆开：

- 使用相关书籍训练 Claude/前身：该 order 在特定记录上认定为 fair use；
- 合法购买纸书后破坏装订、扫描形成数字副本：order 对特定内部用途给出有利结论；
- 从盗版来源取得副本并建立永久 central library：没有因后续训练而整体获得 fair-use 保护。

所以“training fair use”不能改写成“pirated acquisition 也合法”。视频 [28:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=1709s) 正是在强调这两步分开。

课程口述 Anthropic 以 15 亿美元和解。【课后状态更新，截至 2026-08-28】[N.D. Cal. Document 680 的法院文件镜像](https://law.justia.com/cases/federal/district-courts/california/candce/4%3A2024cv05417/434709/680/) 显示法院于 2026-07-20 最终批准 settlement，并在 order 中写明 15 亿美元 settlement fund。该镜像不是法院官网，但呈现带案号、签署日期和法官签名的法院提交文件；PACER 官方 docket 在当前环境不能匿名完整读取。和解仍只是双方结束特定争议的协议，不等于法院建立“所有训练合法/非法”的普遍判例，也不等于一方承认全部法律主张。

### 6.3 Kadrey v. Meta

【课程内容；2025 法院 order 核对】可回查 [CourtListener RECAP docket](https://www.courtlistener.com/docket/67569326/kadrey-v-meta-platforms-inc/)；[N.D. Cal. Document 598 镜像](https://law.justia.com/cases/federal/district-courts/california/candce/3%3A2023cv03417/415175/598/) 展示案号、签署日期和 order 全文。CourtListener/Justia 都是公开法院文件镜像，不是法院官网；PACER 官方 docket 在当前环境不能匿名完整读取。法院在该 plaintiffs、claims、evidence 的 record 上，对“复制这些原告的书用于 Llama training”的 fair-use defense 给 Meta summary judgment。

该 order 自己使用 “on this record” 的窄措辞。它不能推出：

- 任何作品、任何取得方式、任何模型训练都 fair use；
- torrent distribution 等独立主张自动消失；
- 其他法院必须对不同事实作同样权衡。

### 6.4 The New York Times v. Microsoft/OpenAI

课程把它列为 2023 年提起、涉及训练与输出文章的 allegation（指控）。**Complaint 是原告主张，不是已判事实。**截至课程讲述，不能把 pending allegation 写成法院已认定侵权或已认定 fair use。

### 6.5 课堂结论怎样改写得准确

不准确：

> 美国法院已经证明训练全部合法；盗版书全部非法，所以问题结束。

准确：

> 2025 年两个地区法院 order 在各自案卷上，对特定训练复制给出 fair-use 结论；Bartz 又把训练用途与盗版取得/永久库拆开。fair use 仍然是事实密集型分析，其他作品、取得方式、输出、市场证据、合同与法域可能不同。

美国版权局 [Copyright and Artificial Intelligence, Part 3: Generative AI Training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf) 也把复制发生于数据准备、训练和模型部署等环节分开讨论，并没有给“所有训练统一合法/非法”的机械规则。该链接是 Copyright Office 官方预发布版本；引用时应同时记录发布日期和最终版状态。

---

<a id="s7"></a>
## 7. Common Crawl、WARC、WET 与 HTML 提取

### 7.1 Common Crawl 是“定期抓到的一部分 web”，不是整个 Internet

【课程内容，源码 241–271；视频 [32:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=1951s)】**Common Crawl** 是 2007 年成立的非营利 web archive（网页档案）项目。它定期运行 crawler，并开放抓取结果。课程在 2026 年用“每轮约 30–50 亿页、累计约 3000 亿页”帮助感受规模；这些是**课程时点快照**，不是永久常数。

要把三个数分开：

- URL 数：发现了多少地址；
- page fetch 数：实际下载了多少次页面；
- unique document 数：处理和去重后还剩多少不同文档。

例：frontier 有 100 个 URL，其中 10 个失败、20 个 URL 返回相同模板页。则成功 fetch 是 $`100-10=90`$，但 unique document 最多 $`90-20+1=71`$。为什么加 1？20 个相同副本会留下 1 个代表，不是全删掉。

视频 [33:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=2012s) 提到某次 crawl 为数百 TB。**TB（terabyte）**在十进制口径是 $`10^{12}`$ bytes；若论文写 TiB，则 $`1\text{ TiB}=2^{40}`$ bytes。看到数字先查单位，不能把 TB 与 token 数相加。

### 7.2 WARC、WAT、WET 分别保留什么

【课程内容；Common Crawl 官方格式说明核对】

| 格式 | 人话 | 可能保留 | 主要损失/代价 |
|---|---|---|---|
| WARC | Web ARChive，网页抓取档案 | HTTP response、headers、HTML/原始 payload、URL、时间 | 文件大，需自己解析 |
| WAT | Web Archive Transformation | 从 WARC 计算的 metadata，如 headers、links | 通常不含完整正文 |
| WET | WARC Encapsulated Text | 已抽取的 plaintext 和少量 WARC metadata | HTML 结构、图片、脚本和部分正文线索丢失 |

**HTTP response（HTTP 响应）**是服务器回给 crawler 的状态、headers（头信息）和内容。视频 [35:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=2100s) 开始对比格式，[35:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2129s) 强调 WET 的转换是 lossy（有损）：丢掉的信息以后不能从 WET 唯一还原。

官方说明可回查：[Common Crawl Get Started](https://commoncrawl.org/get-started) 与 [Web Archiving File Formats Explained](https://commoncrawl.org/blog/web-archiving-file-formats-explained)。

### 7.3 HTML→text：一个六行例子

原网页：

```html
<nav>首页 | 登录 | 广告</nav>
<article><h1>猫为什么呼噜？</h1>
<p>呼噜可能与交流和自我安抚有关。</p></article>
<footer>隐私 | 条款 | © 2026</footer>
```

提取器 A 直接取所有可见文字：

```text
首页 登录 广告 猫为什么呼噜？
呼噜可能与交流和自我安抚有关。 隐私 条款 © 2026
```

提取器 B 识别 `<article>`，只留：

```text
猫为什么呼噜？ 呼噜可能与交流和自我安抚有关。
```

若每页模板有 8 个词、正文 12 个词，抓 100 万页：模板词为

```math
8\times1{,}000{,}000=8{,}000{,}000,
```

正文词为

```math
12\times1{,}000{,}000=12{,}000{,}000.
```

不去模板时，$`8/(8+12)=40\%`$ 的训练词都可能是重复导航。视频 [36:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=2161s) 对比 extraction 工具；DCLM 实验也说明 extraction 选择会改变下游结果。这里的因果解释只适用于被比较的 pipeline，不是说某工具永远最好。

### 7.4 Raw 也已经被选择过

WARC 常被叫 raw，但它至少已经经过：seed、URL priority、robots/politeness、时间、网络失败、地区和 fetch-size 上限。它是“相对后续处理更原始”，不是整个 web 的无偏样本。

一个反例：若 crawler 只从英语新闻站 seed 出发，它即使完全不做语言过滤，得到的 raw crawl 也会偏英语新闻。偏差在 extraction 之前就发生了。

---

<a id="s8"></a>
## 8. Wikipedia、GitHub、Software Heritage 与 arXiv

### 8.1 为什么优先 bulk dump，而不是爬网页界面

【课程内容，源码 274–327；视频 [36:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2190s)】**Bulk dump（批量转储）**是来源方专门发布的大批数据文件。相比爬 UI，它通常：

- 版本和时间更清楚；
- 对服务器负担更小；
- 少混入导航、广告和动态界面；
- 更容易校验 checksum（校验和，即文件内容指纹）。

但 dump 只解决“怎样稳定取得”；它不自动解决每条记录的 copyright、privacy、许可和质量。

### 8.2 Wikipedia：有引用要求，不等于没有偏差

【课程内容；视频 [36:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2219s)】Wikipedia 是有 notability（关注度）和可靠来源要求的百科，不是个人观点仓库。它定期提供 dumps，通常不必爬网页。

三个边界：

1. **Contributor skew（贡献者偏斜）**：少量活跃编辑贡献很多内容，编辑人群不等于世界人口。
2. **Language skew（语言偏斜）**：不同语言版规模、主题和引用资源差异很大。
3. **Timing risk（时间风险）**：dump 恰好截到临时错误或恶意编辑。

视频 [37:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2279s) 提到贡献集中；[38:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2309s) 强调 dump 比反复 crawling 更合适。

【补充解释】防守性做法：同时保存 revision id、page id、dump date；抽查近期大改；与前后 dump 比差异；对高风险实体再核来源。不要把“任何人可编辑”直接等同“全都不可信”。

### 8.3 Poisoning 的时间窗

【课程内容；视频 [39:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=2368s)】**Data poisoning（数据投毒）**是攻击者让恶意或误导数据进入训练流程，以改变模型行为。这里只讲 threat model（威胁模型）和防守，不提供如何实施攻击。

最小时间线：

```text
09:00 正常页面
09:55 恶意编辑
10:00 dump 截止
10:10 管理员回退
```

只看 10:10 的 live page 会以为没问题，但 10:00 dump 已包含恶意版本。防守要检查 snapshot 本身，而不是只看事后网页。

### 8.4 GitHub：public 只保证看得见，不等于得到无限再利用许可

【课程内容，视频 [39:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2399s)】一个 repository（代码仓库）包含文件树和 commit history；issue、pull request（PR）、review、comment 属于 metadata/协作记录。训练代码时要决定要哪一层：

| 数据对象 | 取得方式 | 可学到什么 | 风险 |
|---|---|---|---|
| repository files | `git clone` 或 archive | 代码与目录结构 | fork/复制、生成物、密钥、许可混杂 |
| commits/diffs | git history | 修改过程 | 作者信息、误提交秘密 |
| issues/PR/comments | API/archive | 问题→讨论→修复 | PII、机器人垃圾、许可不清 |

视频 [40:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2459s) 区分 repository 与 metadata，[41:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2489s) 提到事件流。

GitHub 官方说明：公开仓库允许平台用户查看和 fork；若没有 license，默认 copyright 仍适用，不能把“public”当“任意复制、分发、制作衍生物”的授权。[GitHub Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。课程源码“任何 public permissive repo 都可训练”的一句话仍需项目律师按具体 license、依赖和用途审查。

### 8.5 Fork 去重小例

原仓库有文件 A、B、C；3 个 fork 都没有改动。按仓库计共有 $`4\times3=12`$ 个文件记录，但 unique content 只有 3 份。

后来 fork 1 修改 C→C1：unique content 变 A、B、C、C1，共 4 份，而不是 12。若不去重，热门仓库和 fork 网络会被过度加权。

### 8.6 Software Heritage：保存软件，不等于替所有文件重新授权

【课程内容，源码 311–315】Software Heritage 从 GitHub、GitLab、包仓库等保存 source history，重点不是 issues/comments。课程源码写“28.8M source files”是 **2026 讲义快照中的口径/疑似数量级笔误**；[Software Heritage 2025 官方活动报告](https://www.softwareheritage.org/2026/01/16/software-heritage-activity-report-2025/)称 archive 已保护超过 27B unique source files、来自 421M projects。动态计数还可通过[官方 statistics API 文档](https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api-stat.html)核验，因此不能把讲义的 28.8M 用于严肃容量规划。

更重要的是：archive 提供保存和检索基础设施，不会让上游每个文件的原 license 消失。使用者仍要保留 origin、commit、path、license detection 与冲突记录。

### 8.7 arXiv：PDF、LaTeX、metadata 是三类不同对象

【课程内容；视频 [42:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=2521s)】arXiv submission 可能含：

- metadata：title、authors、abstract、category；
- rendered PDF；
- LaTeX source 和图表（作者可选择提供）。

视频 [42:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2549s) 提醒 PDF→text 需要解析；[42:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2579s) 提到 bulk route。metadata 的开放口径不能自动传到论文正文；作者选择 all rights reserved 或不同 CC license，需逐条保留。

例：同一公式 PDF 抽出 `n1/4`，LaTeX source 是 `n^{-1/4}`。如果只拿 OCR 文本，数学含义会错。因此 raw source 更结构化，但宏、引用、图片、编译失败也要处理。

---

<a id="s9"></a>
## 9. 从 BERT 到 GPT-3：数据 pipeline 逐步变复杂

### 9.1 BERT 与 BooksCorpus

【课程内容，源码 330–350；视频 [45:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2729s)】BERT 使用 English Wikipedia 与 BooksCorpus。课程强调“sequence 是连续 document 片段，而不是互不相关的句子”，因为上下文结构影响预训练任务。

BooksCorpus 来自 Smashwords 上免费自出版书，论文报告约 7000 本、985M words。**M（million）**是 $`10^6`$，所以 985M words 是 $`985\times10^6=985{,}000{,}000`$ words。免费价格不等于 public domain；后续下架与 ToS 争议说明必须记录取得方式和授权口径。

视频 [46:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=2762s) 的口述容易让人误听成“free 就 legally allowed”。正确拆法是：价格、访问权、copyright、license、ToS 分别判断。

### 9.2 GPT-2 WebText：用 Reddit karma 当质量 proxy

【课程内容，源码 353–361；视频 [47:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2850s)】WebText 取 Reddit 帖子里、karma 至少 3 的 outgoing links，得到约 8M pages、40GB text。**Karma** 是平台用户投票汇总；这里被当作“人觉得值得分享”的 proxy。

因果链：

```text
Reddit 用户群体
  -> 什么链接会被发帖
  -> 什么会得到 >=3 karma
  -> 哪些站能被下载/解析
  -> WebText 的主题、语言和风格
```

因此 WebText 不是“全网质量抽样”，而是 Reddit 人群和投票机制筛出的分布。OpenWebText 是开放复现：从 Reddit submissions 取 URLs、做语言识别和 near dedup；复现 pipeline 不保证得到与原私有快照完全相同的 bytes。

### 9.3 CCNet：去重、语言识别、Wikipedia-like 质量

【课程内容，源码 364–376；视频 [48:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=2912s)】CCNet 主要步骤：

**Classifier（分类器）**是读入一条 document，再输出类别或分数的模型；例如输出“English 0.95”或“像 Wikipedia 0.72”。分数代表它在训练定义下的判断，不自动等于事实、价值或合法性的概率。

1. 轻量规范化后 paragraph dedup；
2. fastText language ID；
3. 用 KenLM 5-gram 模型按“像 Wikipedia 的程度”分质量桶。

**Language identification（语言识别）**是预测文本属于哪种语言。若 English 阈值是 0.9：分数 0.95 留下，0.85 删除。阈值提高通常减少误留非英语，也会增加误删混合语言、方言或短文本。

**KenLM 5-gram** 用前 4 个 token 预测下一个 token。它评分“统计风格像不像参考语料”，不直接判断事实真伪、原创性或社会价值。

视频 [48:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=2938s) 讨论 probability score；CCNet 对低资源语言的目标不能改写成“分类器已经消除语言偏差”。

论文在自己的 BERT 训练/evaluation 协议中发现 filtered Common Crawl 可优于只用 Wikipedia。它支持“过滤后的 web 能补充百科”，不证明每个任务、语言或模型都按同一排序。

### 9.4 C4：一堆小规则叠成一种数据价值观

【课程内容，源码 379–404；视频 [49:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=2998s)】C4 从 2019-04 Common Crawl snapshot 约 1.4T tokens 出发，使用：句末标点、每行至少 5 words、每页至少 3 sentences、语言阈值、脏词表、去代码/模板规则等。课程代码给出约 806GB、156B tokens；[T5 原论文](https://arxiv.org/abs/1910.10683)报告发布/序列化口径约 745GB。两者不是同一 bytes 口径，不能悄悄选一个当唯一真值。

保留比例按 token 粗算：

```math
156\text{B}/1400\text{B}=0.1114\approx11.14\%.
```

也就是约 88.86% token 没进入结果。但这不是“删掉的都坏”：

- `}` 可能表示代码，也可能出现在数学/文本；
- 没有句末标点可能是标题、诗、聊天或某些语言书写习惯；
- bad-word list 会误伤引用、身份词或讨论伤害的文档。

视频 [51:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=3062s) 用 curly brace 说明手工规则的强假设。[51:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=3092s) 转入对 C4 domain composition 的分析：来源域分布是 pipeline 的结果，不是自然界给定。

### 9.5 GPT-3：多来源、质量 classifier 与 mixture weights

【课程内容+论文核对，源码 407–418；视频 [53:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=3210s)】课程把 GPT-3 的多个来源过度压成“570GB、约400B”。按 [GPT-3 原论文 Table 2](https://arxiv.org/abs/2005.14165)，**570GB 与约 410B tokens 只对应 filtered Common Crawl**；其余 pool 还含 WebText2 约19B、Books1约12B、Books2约55B、Wikipedia约3B。合计：

```math
410+19+12+55+3=499\text{B pool tokens}.
```

GPT-3 实际训练消费 300B token positions，并按 mixture weights 重采样，并非把 499B pool 每个 token 恰好看一次。

Common Crawl quality classifier 的 positives 来自 WebText、Wikipedia、books，negatives 来自未过滤 crawl。它学习的是“像这些已选 positives 吗”，不是宇宙中的客观质量函数。

例：raw pool 90% web、10% books；训练 mixture 设 web 60%、books 40%。抽 1000 token 时期望 web 600、books 400。books 虽只占 pool 的 10%，训练权重却放大为 40%。这就是 **data mixture（数据混合）**改变学习分布。

### 9.6 五个数据集不能只按大小排

| 数据集 | source | 处理核心 | 可见偏差 |
|---|---|---|---|
| BooksCorpus | 免费自出版书 | scrape/文档切分 | 类型、作者平台、ToS |
| WebText | Reddit outgoing links | karma threshold | Reddit 人群与链接选择 |
| CCNet | Common Crawl | lang ID、Wikipedia-like LM、dedup | 参考语料价值观 |
| C4 | Common Crawl | 手工规则、语言、去重 | 规则误删与英语中心 |
| GPT-3 mixture | 多来源 | classifier、fuzzy dedup、重采样 | 私有来源、权重不透明 |

结论不是“越新越好”，而是每个 dataset 都是 source、processing、mixture、版本和治理决策的乘积。

---

<a id="s10"></a>
## 10. The Pile：把“一个 web corpus”改成 22 个可见来源

### 10.1 为什么 The Pile 重要

【课程内容，源码 421–465；视频 [54:56](https://www.youtube.com/watch?v=-qm0ln33G24&t=3296s)】The Pile 是对 GPT-3 私有数据描述的一种开放回应：志愿者整理 22 个 domains，报告约 **825 GiB** text、约 275B tokens。GiB 是二进制单位：

```math
825\times2^{30}\text{ bytes}/10^9\approx885.84\text{ GB},
```

即约 886 个十进制 GB，不能写成 825GB。其价值不只是大，而是把 component（组成部分）明确列出，让研究者能审计和重混。

**Domain（领域/来源域）**在这里是具有相似来源或体裁的数据分区，如 arXiv、Stack Exchange、Pile-CC，不等于互联网域名。

### 10.2 Pile-CC、PubMed Central、arXiv、Enron

- Pile-CC 从 WARC 自己做 extraction，而不是直接接受 Common Crawl WET；
- PubMed Central 含可公开获取的生物医学论文，但“可公开读取”的许可仍需逐条看；
- arXiv 用 LaTeX/source 处理学术结构；
- Enron emails 是调查后公开的企业邮件，真实但带人群、年代与隐私历史边界。

一个 mixture 例：四域各有 100GB。若按 bytes 均匀，每域 25%；若 books 平均每 token 4 bytes、code 平均 2 bytes，同样 100GB 对应 token 数并不相同。按 bytes、documents、tokens 或 examples 加权会得到不同训练分布。

### 10.3 Project Gutenberg 与 PG-19

【课程内容；视频 [55:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=3328s)】Project Gutenberg 对作品做 copyright clearance，许多作品在美国 public domain。PG-19 是 2019 年前的书籍子集。

“mostly public domain”不能改写成全球所有法域都 public domain，也不能抹掉 Gutenberg 的使用条款、转录贡献和 metadata 条件。版本卡应记录下载日期与 clearance 标记。

### 10.4 Books3：来源风险会沿着 dataset 血缘传播

【课程内容；视频 [56:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=3361s)】Books3 包含约 196K books，来源被报告为 shadow library。它后来作为 The Pile component 被很多下游 dataset 使用。

血缘链：

```text
上游作品 -> shadow library copy -> Books3
       -> The Pile bundle -> 下游混合数据 -> 模型训练
```

“我只下载 The Pile，不知道 Books3 来源”不会让 provenance 问题自动消失。视频 [56:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3389s) 强调它后来被移除/下架；本笔记不提供任何获取路径。

### 10.5 Stack Exchange：问答结构和 metadata 都有价值

【课程内容；视频 [56:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=3419s)】Stack Exchange dump 不只是文字：question、answers、score、accepted flag、tags、comments 和 anonymized user metadata 可帮助排序和过滤。

例：一个问题有三答，scores 为 10、3、-2。若按 score 排序，序列是 A→B→C；若原网页时间顺序是 C→A→B，模型看到的对话结构不同。处理必须把排序规则写进 provenance card。

【补充解释】Q&A 形式接近 instruction→response，但不等于 post-training 对话：它来自特定社区规范、专家比例、年代和投票机制。

---

<a id="s11"></a>
## 11. Gopher、LLaMA、RedPajama 与 SlimPajama

### 11.1 MassiveText：数据很大，不等于模型全看过

【课程内容，源码 469–486；视频 [58:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3509s)】Gopher 的 MassiveText 报告多个 components：MassiveWeb、C4、books、news、GitHub、Wikipedia。对一些非 web 来源描述很少，这本身就是 reproducibility（可复现性）边界。

MassiveWeb 处理包括 English filtering、dedup、train-test overlap removal、手工质量规则与 SafeSearch toxicity filtering。课程报告 corpus 约 10.5TB，但 Gopher 训练只取 300B tokens，约为语料 token 的 12%。

若 pool 有 2.5T tokens：

```math
300\text{B}/2500\text{B}=0.12=12\%.
```

这说明：**dataset size** 是可选池规模，**training tokens** 是训练真正抽取的 token 次数，两者不是同一个数。可能重采样，也可能只采一部分。

### 11.2 LLaMA：列出 component 仍不等于完全开放

【课程内容+论文核对，源码 489–500；视频 [59:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=3599s)】课程用“约1.2T”概括 LLaMA 1，但 [LLaMA 原论文](https://arxiv.org/abs/2302.13971) 的口径分模型大小：tokenized dataset 约 1.4T；6.7B/13B 模型训练 1.0T token positions；32.5B/65.2B 模型训练 1.4T。不存在所有 LLaMA 1 统一训练 1.2T 的精确口径。来源包括：CCNet-processed Common Crawl、C4、GitHub、Wikipedia、Gutenberg/Books3、arXiv、Stack Exchange。

重要的 processing 细节：

- Common Crawl 按是否被 Wikipedia 引用分类；
- GitHub 做 permissive-license 与手工规则过滤；
- Wikipedia 覆盖 20 languages；
- arXiv 删除 comments、展开 inline macros、处理 bibliography；
- Stack Exchange 选择较大站点，并按 answer score 排序。

视频 [60:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3629s) 用 “good websites” 解释 Wikipedia reference proxy；它会偏向可被 Wikipedia 引用的主题和语言，不能等同事实质量判定。

### 11.3 RedPajama 与 SlimPajama：复现不是复制 bytes

【课程内容；视频 [61:27](https://www.youtube.com/watch?v=-qm0ln33G24&t=3687s)】RedPajama v1 尝试复现 LLaMA 的 component recipe，约 1T tokens。SlimPajama 再对其做清洗与 MinHashLSH near-dedup，得到约 627B tokens。

若从 1T 粗略缩到 627B，保留率：

```math
627/1000=0.627=62.7\%,
```

删除率约 $`37.3\%`$。但不能说 37.3% 全是重复，因为版本、过滤和单位口径也可能影响数字。

**Replication（复现）**是根据公开 recipe 独立构造近似数据；URL 的内容会变、crawl snapshot 不同、私有 books 不可得、工具版本不同，所以它通常不可能与原数据逐 byte 相同。

### 11.4 三张 provenance 卡

| 数据集 | source | processing | 课程规模快照 | 主要边界 |
|---|---|---|---|---|
| MassiveText | web+C4+books+news+code+wiki | manual rules、lang、dedup、toxicity | 10.5TB pool；Gopher 300B train tokens | 部分来源不透明 |
| LLaMA | 7 类公开/半公开组件 | 每源不同规则与采样权重 | dataset约1.4T；6.7/13B训练1.0T，32.5/65.2B训练1.4T | Books3、许可、不可完整复现 |
| SlimPajama | RedPajama v1 | 清洗、MinHashLSH dedup | 627B tokens | 继承上游来源与许可 |

---

<a id="s12"></a>
## 12. RefinedWeb、FineWeb 与 Dolma

### 12.1 RefinedWeb：“web data is all you need”是实验主张，不是自然定律

【课程内容，源码 504–520；视频 [62:03](https://www.youtube.com/watch?v=-qm0ln33G24&t=3723s)】RefinedWeb 从 Common Crawl WARC 开始，用 trafilatura 做 HTML→text、Gopher-style rules、语言/URL 处理和 5-gram MinHash fuzzy dedup。论文处理约 5T tokens，公开约 600B。

公开比例粗算：

```math
600\text{B}/5000\text{B}=12\%.
```

标题“web data is all you need”只能读成：在论文的模型、预算、过滤和 benchmarks 下，高质量 web-only corpus 有竞争力。它不能推出 books/code/science 永远没价值。

视频 [62:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=3750s) 谈规则过滤“像英语”；RefinedWeb 避免 learned-quality classifier 是为了减少某类 reference bias，但手工规则自身也有价值判断。

### 12.2 FineWeb：更多 dumps，也要更多版本元数据

【课程内容；视频 [63:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=3780s)】FineWeb 起初复现 RefinedWeb，扩展到 95 个 Common Crawl dumps，加入 URL filtering、language ID、Gopher/C4/manual rules、MinHash dedup，并匿名化 emails 和 public IPs；课程报告 15T tokens。

95 dumps 不等于 95 倍 unique data。若每个 dump 100 documents、相邻 dump 80% 重叠：两个 dump 的 unique 数不是 200，而是

```math
100+(100-80)=120.
```

跨 snapshot dedup 是否做、按哪一级做，会强烈影响规模与时间分布。

### 12.3 Dolma：多来源开放 dataset 也需要 member-level provenance

【课程内容，源码 523–537；视频 [63:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=3811s)】Dolma components 包括 Reddit/Pushshift、PeS2o academic papers、C4、Gutenberg、Wikipedia/Wikibooks 等。web 处理含 language ID、Gopher/C4 rules、toxicity rules/classifier 与 Bloom-filter dedup；课程报告约 3T tokens。

视频 [64:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=3841s) 提到 PeS2o 论文来源，[64:27](https://www.youtube.com/watch?v=-qm0ln33G24&t=3867s) 提到 toxicity 处理。

一份 dataset 的 collection license 说明“整理后的集合”怎样提供，不自动证明每个 Reddit comment、paper 或 book 的底层权利相同。下载者需要 component manifest，而不是只保存顶层 LICENSE。

### 12.4 三者比较

| 项目 | RefinedWeb | FineWeb | Dolma |
|---|---|---|---|
| 主来源 | Common Crawl | 95 CC dumps | web+Reddit+papers+books+wiki |
| extraction | WARC→trafilatura | 自建 CC pipeline | 多来源；web 自处理 |
| quality | Gopher rules | Gopher+C4+额外规则 | rules，另做 toxicity classifier |
| dedup | MinHash 5-gram | MinHash | Bloom-filter based |
| 课程规模 | 发布 600B | 15T tokens | 3T tokens |
| 最大误读 | web 永远够 | 多就自动多样 | 顶层开放=成员都开放 |

---

<a id="s13"></a>
## 13. DCLM：classifier filtering 会把“正样本是谁”写进数据

### 13.1 DCLM pool 与 baseline

【课程内容，源码 539–556；视频 [65:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3929s)】DataComp-LM（DCLM）建立标准化实验框架，让不同 data processing 在相同训练/evaluation 预算下比较。DCLM-pool 约 240T tokens；DCLM-baseline 用 classifier 筛到约 3.8T。

保留率：

```math
3.8/240=0.015833\approx1.58\%.
```

也就是约 98.42% pool token 没进入 baseline。保留少不证明保留的一定无错，也不证明删除的一定没价值。

### 13.2 Positive 和 negative 怎样定义“质量”

【课程内容；视频 [65:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=3958s)】课程给的训练集：

- positives 约 200K：OpenHermes-2.5 instruction data + ELI5 questions/answers；
- negatives 约 200K：RefinedWeb samples。

总训练例数约

```math
200{,}000+200{,}000=400{,}000.
```

若二分类器输出 score 0.82，含义是“按该模型和训练集，它更像 positives”，不是“82% 事实正确”。视频 [66:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=3988s) 展示 ELI5 风格，[66:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4019s) 提到线性 classifier。

### 13.3 Selection bias：目标标签本身不是宇宙真值

**Selection bias（选择偏差）**是进入训练/评估样本的机制与目标总体不同。这里 positives 偏：

- instruction/answer 风格；
- curiosity questions；
- 英语；
- 特定生成模型与 Reddit 社区。

反例：一份严谨 API reference 很短、重复格式多、不像 ELI5，却对 code model 很有价值。classifier 低分不能证明它“低质量”。

### 13.4 Confusion matrix 小例

先固定 **positive（正类）到底是什么**。在本小节的 quality filtering 任务里：

- 实际 positive = 文档真的有用；实际 negative = 文档无用；
- predicted positive = classifier 决定“留下”；predicted negative = classifier 决定“删除”。

于是四格语义是：

- **TP（true positive）**：真有用，而且被留下；
- **FP（false positive）**：其实无用，却被留下；
- **FN（false negative）**：真有用，却被删除；
- **TN（true negative）**：其实无用，而且被删除。

人工抽查 100 documents：实际 useful 40、not useful 60。classifier 留下 50，其中 35 useful、15 not useful：

| | 实际 useful | 实际 not useful |
|---|---:|---:|
| 留下 | TP=35 | FP=15 |
| 删除 | FN=5 | TN=45 |

```math
\text{precision}=35/(35+15)=70\%,
```

```math
\text{recall}=35/(35+5)=87.5\%.
```

**Precision（精确率）**问“所有留下的文档里，多少真的有用”；**recall（召回率）**问“所有真的有用的文档里，多少被留下”。这里 false positive 是误留坏文档，false negative 是误删好文档。对低资源语言，少量 FN 就可能删掉很大比例的可用语料。

### 13.5 “下游成绩更好”能说明什么

【课程内容】DCLM 在固定实验协议中，classifier filtering 的 downstream evaluation 优于若干替代过滤。这支持“该 recipe 在这些预算与指标下有效”，不能支持：

- classifier 学到普遍、跨语言、跨任务的质量；
- 模型所有风险更低；
- 数据许可更清楚；
- 只要分数高就保留不会损害 diversity。

---

<a id="s14"></a>
## 14. Nemotron-CC：ensemble 与 synthetic rephrasing

### 14.1 为什么说 DCLM/FineWebEdu 过滤太 aggressive

【课程内容，源码 559–575；视频 [67:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=4051s)】Nemotron-CC 的动机是：高阈值 classifier 可能删掉约 90% 数据；大模型训练仍需要足量 tokens。因此尝试在 quality 与 quantity 间回收更多数据。

若 10T token pool 留 10%：

```math
10\text{T}\times0.10=1\text{T tokens}.
```

若目标训练 6T、又不想重复 6 epochs，就需要更多可用 token，或接受重采样/合成的代价。**Epoch（轮）**是“把所定义的训练集平均看一遍”的口径；1T-token 数据抽 6T token positions，粗略就是 6 epochs，但加权 mixture 下各 component 的遍数会不同。

### 14.2 Classifier ensemble

**Ensemble（集成）**是组合多个模型/规则的判断。课程介绍两类信号：

Nemotron-CC 最终 raw web input 是 **99 个 Common Crawl snapshots**，不是把 FineWeb 全部文档直接当最终原料。FineWeb documents 的角色是帮助构造 educational-quality labels/classifier recipe：

1. 用 Nemotron-340B-Instruct 给 FineWeb documents 打 educational-value 分，再 distill（蒸馏）到更快模型；
2. 使用 DCLM classifier 信号。

然后把这些过滤/分类信号应用到 99 个 Common Crawl snapshots，并结合合成改写等步骤形成 Nemotron-CC。要分清“训练 classifier 的参考文档”和“最终被处理的 raw corpus”。

例：教育分 $`e=0.8`$，DCLM 分 $`d=0.6`$。若教学规则取平均：

```math
(e+d)/2=(0.8+0.6)/2=0.7.
```

若阈值 0.65 则留；若取最小值 (min(e,d)=0.6) 则删。**组合规则本身也是 recipe**，不能只写“用了两个 classifier”。

### 14.3 Synthetic rephrasing 与 task generation

【课程内容；视频 [67:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4079s)】对低质量 document，让 LM rephrase；对高质量 document，生成 summary、QA、信息抽取等 tasks。视频 [69:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4140s) 给出类似“总结文档”的任务化方式。

原文：`水在100°C沸腾。` 可能被改成：

```text
问：在标准大气压下，水的沸点是多少？
答：100°C。
```

新增结构更像 instruction data，但会引入：

- teacher model 的错误与风格；
- prompt template 重复；
- 原文 license/provenance 是否仍需追踪；
- synthetic output 的许可与可撤回性；
- 成本和 detector 误判。

### 14.4 规模结果只作课程快照

课程报告 Nemotron-CC 6.3T tokens，其中 HQ subset 1.1T。HQ 占比：

```math
1.1/6.3\approx0.1746=17.46\%.
```

视频 [69:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4170s) 对比 Llama 3 15T 和课程口述 Qwen3 36T；这些只能表示课程当时报告口径，不能当“今天最大模型必须训练多少 token”的规则。

---

<a id="s15"></a>
## 15. The Stack 与 Stack v2：代码不是只有 `.py` 文件

### 15.1 The Stack v1

【课程内容，源码 578–597；视频 [71:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4260s)】The Stack 从 GitHub Archive 取得 repository names，clone 大量 repositories，按 permissive license 过滤、去 binary、做 MinHash/Jaccard near-dedup，报告约 3.1TB code。

课程把数字四舍五入为“137M repositories、51B files、5B unique”；[The Stack 原论文](https://arxiv.org/abs/2211.15533)更精确的口径约为 51.76B file occurrences、5.28B unique files。按论文口径：

```math
5.28/51.76\approx0.1020=10.20\%
```

文件内容 unique；约 89.80% file occurrences 与别处重复。这里的 “unique” 受规范化/hash 定义影响，不是永恒真值。

视频 [71:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4290s) 口头自我修正数字，笔记保留 source 表述并标课程快照，不用口误推新结论。

### 15.2 License detection 不是法律保证

go-license-detector 或 GitHub Licensee 会把 LICENSE 文本匹配已知模板。它可能遇到：

- monorepo 各子目录不同 license；
- vendored dependency 与主仓库不同；
- 无 license；
- 自定义条款；
- 作者无权给第三方复制内容再授权。

所以 `detected=MIT` 是 metadata signal，不是律师审查结论。

### 15.3 Stack v2 增加了哪些结构

【课程内容；视频 [71:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4319s)】Stack v2 加入：

- Software Heritage repositories；
- GitHub issues、comments、PRs；
- PyPI/npm/devdocs documentation；
- binary/malware/bot filtering、PII redaction、dedup；
- source code 与 LLVM intermediate representation（IR）配对；
- code contests、StackOverflow、arXiv、Wikipedia 等既有数据。

**LLVM IR（中间表示）**是很多语言编译时都会变成的低层共同表示。视频 [72:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4379s) 的直觉是：低资源语言可借共同 IR 对齐结构；这不是保证所有语义都保留。

### 15.4 PR linearization 小例

PR 是结构对象：title、description、old file、diff、review comment、state。模型吃的是 token sequence，所以要 linearize（线性化）：

```text
<TITLE> 修复除零
<FILE> calc.py
<CONTEXT> def div(a,b): ...
<DIFF> - return a/b
       + if b==0: raise ValueError
<REVIEW> 请补测试
<STATE> changes_requested
```

视频 [73:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=4412s) 说明 PR 结构，[73:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4439s) 说明 diff 需要 surrounding context，[74:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=4469s) 提到 comment/review state。

顺序错了会把“评论针对哪版代码”打乱；只留 diff 会缺函数上下文；全留又可能超过 context budget 或放大 PII。

---

<a id="s16"></a>
## 16. CommonPile：permissive collection 也有 provenance 难题

### 16.1 问题不是“能否找到一个 LICENSE 文件”

【课程内容，源码 600–618；视频 [74:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4499s)】CommonPile 问：能否只用 permissively licensed/openly licensed sources 训练有竞争力模型？课程报告约 8TB collection，并展示 Comma models 的结果。

视频 [75:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4530s) 更准确的问题是“how far can you get”，不是先保证能追平任何闭源模型。

### 16.2 License laundering

**License laundering（许可漂白）**：上游无权再许可的内容，被中间人放进一个带宽松 license 的仓库或 dataset，造成看似合法的外观。

例：

```text
作者 A 的 all-rights-reserved 文章
 -> B 未经许可复制到“CC-BY dataset”
 -> C 只看顶层 CC-BY 声明
```

B 贴出的标签不一定能创造 B 从未拥有的权利。视频 [76:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=4618s) 说“很难判断是否真实”；正确做法是保留 upstream evidence，不是盲信标签。

### 16.3 Collection license 与 member license

【课程内容；视频 [77:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4650s)】数据库的 selection/arrangement 可能有独立权利；顶层 collection license 许可的是集合层权利，不必然覆盖每个 member work。

一个 3 行 manifest：

| record | collection license | member license | 能否只凭第一列纳入 |
|---|---|---|---|
| A | ODC-By | CC-BY | 仍需遵守署名等条件 |
| B | ODC-By | MIT | 仍需保留 MIT notice |
| C | ODC-By | unknown | 不能仅凭 ODC-By 判断 |

### 16.4 Synthetic provenance 不能从零开始

【课程内容；视频 [78:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4680s)】synthetic text 还需记录：teacher model/version、prompt、输入 documents、sampling 参数、过滤、输出 license/terms 与生成日期。teacher 若在不透明数据上训练，不代表每个 output 自动侵权；也不能反过来说 provenance 风险已消失。这是尚有法律与政策不确定性的层，需专门审查。

### 16.5 能说明与不能说明

课程图显示 permissive-data model 可以取得有意义的 benchmark 结果。视频 [78:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=4738s) 把它与较老模型比较。能说明：这条路线不是“完全训练不了”。不能说明：

- benchmark 差距只由 token 数造成；
- collection 内每条 license 都无误；
- 成本、语言覆盖和现实任务都达到同等水平；
- permissive-only 是所有法域唯一可行方案。

---

<a id="s17"></a>
## 17. Exact dedup、near dedup、Jaccard、MinHash 与 Bloom filter

> 【课程边界】Lecture 13 只建立前置词汇；算法实现、filtering 和 mixture 的更多细节在 Lecture 14。这里用小数例保证以后不会只背名字。

### 17.1 Exact duplicate 与 near duplicate

- **Exact duplicate（完全重复）**：在约定规范化后内容完全相同。
- **Near duplicate（近似重复）**：有少量改写、模板差异或重新排版，但大部分内容相同。

先声明 normalization（规范化）：是否 lowercase、合并空格、去 HTML、去标点。例：

```text
A = "Hello,  cat!"
B = "hello cat"
```

原 bytes 不同；若 lowercase、去标点、合并空格，两者都变成 `hello cat`，才算 exact duplicate。规则越激进，越可能把本来不同的文本误合并。

### 17.2 为什么重复会改变训练权重

数据有 A、B、C 三篇；A 被复制 8 次。共 10 records：A 类占 $`8/10=80\%`$，B、C 各 10%。若 exact dedup 后各留一份，三者各 $`1/3=33.33\%`$。

去重不是只省磁盘；它改变 sampling probability。反过来，同一个定义或版权声明的合理重复也可能被删，需按用途选择粒度。

### 17.3 Jaccard similarity 手算

把 document 切成 shingles（连续 token 片段）。为方便手算，用 token set：

```math
A=\{\text{cat,sits,on,mat}\},
```

```math
B=\{\text{cat,sits,near,mat}\}.
```

交集有 cat、sits、mat，共 3；并集有 cat、sits、on、near、mat，共 5。**Jaccard similarity（杰卡德相似度）**：

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac35=0.6.
```

阈值 0.8 时不算 near duplicate；阈值 0.5 时算。阈值是 precision/recall tradeoff，不是数学给出的唯一答案。

### 17.4 MinHash 的直觉

**Hash（哈希）**是把输入按稳定规则映射到一个数字或桶位置的函数：同一输入、同一算法与 seed 应得到同一输出；不同输入仍可能得到同一输出，这叫 **collision（碰撞）**。它是短指纹/索引，不是“唯一身份证”。

直接比较 10 亿文档两两 Jaccard 需要约 $`10^{18}`$ 对，做不起。**MinHash** 用多个随机 hash 排序，每个排序只保存集合中最小 hash 的元素，形成短 signature（签名）。两个集合的 signature 相等比例估计 Jaccard。

极小例，集合：

```math
A=\{a,b,c\},\qquad B=\{b,c,d\}.
```

两个假想 hash 排序：

| hash | 顺序 | min(A) | min(B) | 相同？ |
|---|---|---|---|---|
| h1 | a,b,c,d | a | b | 否 |
| h2 | c,d,b,a | c | c | 是 |

2 个位置中 1 个相同，估计 Jaccard 为 $`1/2=0.5`$。真实 Jaccard 是交集 $`\{b,c\}`$ 两个、并集四个，也是 $`2/4=0.5`$；这里只是碰巧精确。signature 越短，随机误差越大。

**LSH（Locality-Sensitive Hashing，局部敏感哈希）**把相似 signatures 更可能放进同一候选桶，减少需要精确比较的 pairs。它可能漏掉 near duplicate，也可能产生候选误报。

### 17.5 Bloom filter 的直觉与误报

**Bit（比特）**是只能装 0 或 1 的一个格。**Bloom filter（布隆过滤器）**是省内存的 membership test（成员查询）结构。加入元素时把多个 hash 指向的 bits 置 1；查询时若任一 bit 为 0，肯定没见过；若全为 1，只能说“可能见过”。

8-bit tiny example，2 个 hash：

```text
index: 0 1 2 3 4 5 6 7
初始: 0 0 0 0 0 0 0 0
加入 A，位置1和5置1:
       0 1 0 0 0 1 0 0  = 01000100
加入 B，位置2和5置1:
       0 1 1 0 0 1 0 0  = 01100100
查询 C，hash 到位置1和2：两位都是1 -> 回答“可能见过”
```

C 其实没加入，这是 **false positive（假阳性）**。标准 Bloom filter 没有 false negative 的前提是：位数组没被错误清除、hash 和数据编码一致。用它去重时，假阳性会误删新文档。

### 17.6 去重层级选择

| 层级 | 能发现 | 容易漏/伤 |
|---|---|---|
| URL | 同地址 | 同文不同 URL、动态 URL |
| document hash | 完全相同正文 | 小改写 |
| paragraph | 模板/转载段落 | 合理引用、通用定义 |
| n-gram MinHash | 近似转载 | 短文、跨语言改写 |
| repository/file | fork 和 vendored code | 相同模板的合法项目差异 |

必须记录：单位、normalization、shingle size、threshold、跨 split/跨 dump 范围，以及保留哪个代表。

---

<a id="s18"></a>
## 18. PII、分类器偏差、consent 与 poisoning 防守

### 18.1 PII 从零定义

**PII（Personally Identifiable Information，个人可识别信息）**是能单独或与其他信息组合识别个人的数据，如姓名+地址、email、电话、政府证件号。各法域定义和义务不同。

公开网页出现 email 不等于当事人同意把它永久收入训练集。**Consent（同意）**还要问：谁同意、同意什么用途、能否撤回、告知是否充分。

### 18.2 Redaction 小例及错误

原文：

```text
联系 Alice：alice@example.com，工单 12345。
```

规则可能变成：

```text
联系 Alice：[EMAIL]，工单 12345。
```

它删了 email，却仍留下姓名和可关联的 ticket id。PII redaction 不是一个 regex 就完成；还要考虑上下文组合、图像、代码 secret、metadata。

误差表：

| 模型判断 | 实际 PII | 实际非 PII |
|---|---:|---:|
| 删除 | TP | FP：误删普通文本 |
| 保留 | FN：隐私泄漏 | TN |

这里换了任务，所以 positive 的含义也换了：实际 positive = **含 PII**；predicted positive = **决定删除**。因此 TP 是含 PII 且删除，FP 是不含 PII 却删除，FN 是含 PII 却保留，TN 是不含 PII 且保留。PII recall 问“所有含 PII 的文本里删掉多少”，precision 问“所有被 PII 规则删除的文本里多少真的含 PII”。隐私场景常更怕 FN，但无限提高 recall 会大量误删少数语言、地址格式、代码和专业文本。

### 18.3 Language/quality/toxicity classifier 的群体影响

一个 English classifier 对标准美式英语 recall 98%，对某方言 recall 70%。两组各有 1000 篇好文：

```math
\text{标准英语保留}=1000\times0.98=980,
```

```math
\text{方言保留}=1000\times0.70=700.
```

即使原数据同量，过滤后变成 980:700，方言少 280 篇。分类器不是只提高“质量”，还重写 population distribution。

**Toxicity（毒性）**标签也依上下文：引用辱骂进行批评、少数群体自称、医学/安全讨论可能被 word list 误删。应按语言和群体分层测 precision/recall，并人工审计边界案例。

同一个“无毒的安全文本被删除”的物理错误，落在哪一格取决于任务怎样定义正类：

- **quality task**：positive=有用，predicted positive=留下。该文本实际有用、却被预测为删除，所以是 **FN**；
- **toxicity-removal task**：positive=含毒，predicted positive=删除。该文本实际无毒、却被预测为删除，所以是 **FP**。

因此，不能脱离 label contract（标签口径）裸说“误删就是 FN”或“误删就是 FP”；必须先写清 actual positive 和 predicted positive 各表示什么。

### 18.4 Dataset poisoning 的防守性 checklist

不提供注入步骤，只讲检测和减损：

1. 记录 snapshot/date/origin 和 cryptographic hash；
2. 比较前后版本异常突增；
3. 对新域名、新作者、重复 trigger 做速率/聚类审计；
4. 高风险来源多方交叉验证；
5. 训练前保留 quarantine（隔离区），不要自动直通；
6. 小模型 ablation（消融：去掉某数据再训练）检查可疑行为；
7. 支持撤回、重建和 incident response（事件响应）。

### 18.5 Privacy、copyright、license、ToS 不能合并成一列

| 例子 | copyright/license | ToS/access | privacy/ethics |
|---|---|---|---|
| CC-BY 公开博客含手机号 | 许可或许允许并需署名 | 网页可访问 | 仍有隐私风险 |
| 私有账号里的自创日记 | 作者持版权 | 无访问权限 | 高隐私/consent 风险 |
| public-domain 老书 | 美国版权可能到期 | 网站条款另查 | 个人隐私通常低但非零 |
| MIT repo 含误提交 key | 代码许可明确 | public repo | secret 不能因 MIT 就传播 |

一列绿灯不能覆盖其他红灯。

---

<a id="s19"></a>
## 19. 数据集血缘时间线与统一比较卡

### 19.1 时间线只表达演化，不表达单向进步

| 年代 | 数据集 | 关键变化 | 继承/回应 |
|---:|---|---|---|
| 2015 | BooksCorpus | 自出版长文 | 后进 BERT/GPT mixtures |
| 2018 | BERT data | Wikipedia+BooksCorpus | 少来源、文档上下文 |
| 2019 | WebText/OpenWebText | Reddit-link quality proxy | web selection 可见化 |
| 2019 | CCNet | lang ID+Wikipedia-like LM+dedup | 大规模 multilingual CC |
| 2019 | C4/T5 | 手工规则清洗单个 CC dump | 成为大量后续 component |
| 2020 | GPT-3 mixture | classifier+多来源重采样 | 私有 books/weights |
| 2020 | The Pile | 22 个开放 components | 也携带 Books3 等风险 |
| 2021 | MassiveText | 细致 web rules+多来源 | 说明充分但数据未发布 |
| 2023 | LLaMA/RedPajama | component recipe→开放复现 | SlimPajama 再去重 |
| 2023 | RefinedWeb | web-only+WARC extraction | FineWeb 扩多 dumps |
| 2024 | Dolma/DCLM | 开放混合；标准化 data competition | rule vs model filtering |
| 2024 | Nemotron-CC | ensemble+synthetic rephrasing | quality/quantity 折中 |
| 2022–24 | Stack/Stack v2 | code→repo+PR+IR+docs | 许可、fork、PII 治理 |
| 2025 | CommonPile | permissive-source experiment | license chain 审计 |

视频 [79:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=4769s) 总结“数据不会从天上掉下来”；时间线不是“新数据集全面支配旧数据集”，因为目标、语言、许可和预算不同。

### 19.2 一张统一比较表

| dataset | source | extraction/filter | dedup | reported size | 主要不可见量 |
|---|---|---|---|---|---|
| WebText | Reddit links | karma+HTML text | 未完整公开 | 40GB | URLs/快照/细节 |
| C4 | CC 2019-04 | rules+lang | document | 课程806GB/156B；论文release约745GB | 被删群体、bytes口径 |
| GPT-3 | CC+webtext+books+wiki | classifier+mixture | fuzzy | filtered CC 410B；全pool约499B；train 300B positions | books/weights details |
| Pile | 22 components | per-source | component-specific | 825GiB≈886GB/275B | 成员许可/PII |
| LLaMA | 7 groups | per-source | 多层 | dataset约1.4T；小模型1.0T、大模型1.4T train positions | exact corpus/weights |
| FineWeb | 95 CC dumps | rules+lang+PII | MinHash | 15T tokens | crawl/threshold effects |
| DCLM | CC pool | learned classifier | pipeline-defined | 240T→3.8T | label value system |
| Nemotron-CC | 99 CC snapshots | FineWeb labels帮助构造classifier；ensemble+synthetic | pipeline-defined | 6.3T | teacher/prompt provenance |

同一行同时出现 GB 和 tokens 只是原报告的两个视角，不允许用“806GB 小于 1.4T tokens”这种不同单位比较大小。

### 19.3 血缘图

```text
Common Crawl
├─ CCNet ──> LLaMA web component
├─ C4 ─────> LLaMA/Dolma components
├─ LLaMA component recipe ──> RedPajama v1 ──> SlimPajama
├─ RefinedWeb recipe ──> FineWeb
├─ 99 CC snapshots ──> Nemotron-CC 最终 raw corpus
│                    └─ FineWeb documents 只帮助标 quality/构造 classifier
└─ DCLM-pool ───> DCLM-baseline

GPT-3：filtered Common Crawl + WebText2 + Books1/2 + Wikipedia
（不是 C4 的下游）

GitHub/Software Heritage
└─ The Stack ───> Stack v2 + PR/metadata/IR

BooksCorpus / Books3 / Gutenberg
└─ BERT / Pile / LLaMA mixtures（权利来源不同）
```

箭头表示数据或 recipe 继承，不表示 byte-identical。

---

<a id="s20"></a>
## 20. Decision tree：一条数据“能否纳入”

> 这是工程审计框架，不是自动法律结论。

```text
1. 数据从哪里来？能定位原始 source、owner、时间吗？
   ├─ 不能 -> 隔离；补 provenance，不进主池
   └─ 能
       ↓
2. 我是否有技术和账户访问权限？取得过程是否遵守限制？
   ├─ 否 -> 停止
   └─ 是
       ↓
3. ToS/robots/API contract 是否允许这种自动取得和用途？
   ├─ 不确定 -> 法务/平台审查；不是“抓到了就算”
   └─ 清楚
       ↓
4. 每个 member 的 license/copyright/public-domain 状态是什么？
   ├─ permissive/license -> 履行署名、notice、share-alike等条件
   ├─ public domain -> 核对作品、日期、法域与版本
   └─ 无许可/有版权 -> 只在律师评估例外后决定
       ↓
5. 是否含 PII、secret、confidential、儿童/健康等敏感数据？
   ├─ 是 -> 最小化、同意/合法基础、redact、访问控制
   └─ 否/已治理
       ↓
6. 数据质量和安全怎样？
   -> extraction、lang、dedup、poison、malware、label audit
       ↓
7. 能否复现、撤回和解释？
   -> version、hash、manifest、lineage、deletion list、owner
```

### 20.1 三个 worked decisions

**例 A：MIT repository，但含第三方图片。** 代码文件可能纳入并保留 notice；第三方图片未必受 MIT 覆盖，单独排除/核权。public repository 本身不是总许可。

**例 B：CC-BY 文章，网页 robots 禁 bot。** 内容许可与自动访问规则分层；优先找官方 dump/API，履行 attribution，并审查 ToS。

**例 C：public-domain 1900 年美国书籍的现代译本。** 原作可能 public domain，现代译文可能有新 copyright；不能只看故事年代。

### 20.2 什么时候必须暂停自动 pipeline

- 来源/许可字段缺失率突然升高；
- 新 crawl 的语言/域名分布突变；
- PII 或 secrets detector 报警；
- 大量内容来自单一账号/短时间；
- 版权/撤回投诉无法映射到 records；
- benchmark contamination 命中；
- classifier 对某语言 recall 明显下降。

---

<a id="s21"></a>
## 21. Dataset provenance card 模板

### 21.1 每个 dataset 至少回答 18 问

1. 名称、版本、发布日期、维护者；
2. intended use（预期用途）与禁止/不建议用途；
3. source URLs/archives 与 acquisition dates；
4. crawler/API/dump 版本和 user agent；
5. access、robots、ToS 审查；
6. collection license；
7. member-level license/copyright/public-domain 证据；
8. consent 与 privacy legal basis；
9. raw formats（WARC/PDF/git/XML 等）；
10. extraction 工具和版本；
11. language ID 模型、threshold、分语言误差；
12. quality/toxicity classifier 的 positives/negatives；
13. exact/near dedup normalization、unit、threshold；
14. PII/secrets/malware 处理；
15. mixture weights、sampling、epochs/repetition；
16. counts：documents、bytes、tokens，单位分别写；
17. audits、known gaps、biases、contamination；
18. contact、撤回、修订、tombstone（删除记录）机制。

### 21.2 一个 3-record manifest

```text
record_id: 001
source: https://example.org/a
fetched_at: 2026-05-01T10:00Z
content_hash: sha256:...
member_license: CC-BY-4.0
extractor: trafilatura-x.y
language: en (score=0.97)
dedup_cluster: 42
pii_action: email_redacted
```

Manifest 不是为了“填表好看”。当作者要求撤回 `source=a` 时，可以找出 derived records、重新构建 dataset，并证明哪个版本受影响。

### 21.3 Dataset card 与 model card 区别

- dataset card 讲数据来源、构成、处理、限制；
- model card 讲模型训练、能力、风险、评测和部署边界；
- 两者用版本和 lineage 链接。

只发布 model card 写“trained on public data”不够，因为 public 没说明许可、时间、processing 或 privacy。

---

<a id="s22"></a>
## 22. 常见误区：错误 → 为什么错 → 正确说法

| # | 错误说法 | 为什么错 | 正确说法 |
|---:|---|---|---|
| 1 | 模型训练了整个 Internet | Internet 还含私网/协议；web 也是动态且抓不全 | 写具体 crawl、snapshot 与过滤 |
| 2 | public web=public domain | 可看见不等于版权到期 | 分查 access、license、copyright |
| 3 | 免费电子书可随便训练 | 价格为零不授予复制权 | 查 license、ToS、例外与法域 |
| 4 | open weights 就有 open data | 权重许可与数据披露不同 | 分开记录权重、代码、数据 |
| 5 | raw crawl 是无偏总体 | seed、priority、失败已选择 | raw 只相对后处理而言 |
| 6 | robots.txt 就是版权许可 | 它是 crawler 协议信号 | ToS、copyright、privacy 另查 |
| 7 | 遵守 robots 就一定合法 | 仍可能违反合同/版权/隐私 | 多层审查 |
| 8 | 违反 robots 就自动版权侵权 | 法律问题与协议层不同 | 不混概念，具体审查 |
| 9 | copyright 一律 75 年 | 期限依作者、作品、日期、法域 | 按适用规则核对 |
| 10 | 不登记就没有版权 | 美国版权通常在原创固定时产生 | 登记与权利产生/诉讼条件分开 |
| 11 | idea 和 expression 都受同样保护 | copyright 通常保护表达，不垄断抽象想法 | 具体表达/专利等另论 |
| 12 | collection license 覆盖所有成员 | 集合与成员权利不同 | 保存 member-level evidence |
| 13 | CC 就是无条件使用 | BY/SA/NC/ND 条件不同 | 读具体 license version |
| 14 | fair use 四因素投票 3:1 | 是整体权衡，不是计票 | 结合具体事实和市场证据 |
| 15 | 研究用途自动 fair use | 商业性只是因素之一 | 不用单一标签下结论 |
| 16 | transformative=跑过神经网络 | 法律分析更细 | 记录用途、复制、输出、市场 |
| 17 | Bartz 证明所有训练合法 | 特定 record，且拆 acquisition/library | 只写案件范围 |
| 18 | settlement=普遍判例 | 和解不是法院对所有争点判决 | 区分 order、verdict、settlement |
| 19 | complaint 中写的就是事实 | complaint 是一方指控 | 标 pending/已认定状态 |
| 20 | shadow library 是普通开放库 | 绕付费墙并有重大法律风险 | 不提供获取操作，隔离来源 |
| 21 | WET 可还原原 HTML | extraction 有损 | 需要结构时从 WARC 重做 |
| 22 | HTML extractor 只是格式工具 | 它决定哪些词进入训练 | 版本化并做 downstream audit |
| 23 | Wikipedia 没偏差 | 编辑者、语言、notability 都有选择 | 记录分布和版本 |
| 24 | dump 一定没有投毒 | snapshot 可截到临时恶意编辑 | 比版本、审异常、保留 hash |
| 25 | GitHub public repo 都是开源 | 无 license 时默认版权仍适用 | 查 root/子目录/依赖 license |
| 26 | clone repo 等于抓 GitHub 网页 | repo 与 issues/PR metadata 不同 | 明确所需对象与 API |
| 27 | arXiv PDF、LaTeX、metadata 同许可 | 三者可能不同 | 按对象保留 license |
| 28 | 复现 recipe 就得到相同 bytes | web、工具和快照会变 | 报差异和 hashes |
| 29 | GB 越大 token 越多 | 编码、语言和压缩不同 | 同时报告 bytes/tokens |
| 30 | pool tokens=training tokens | 可抽样、重复、只用一部分 | 分报 pool 和 train |
| 31 | karma 是客观质量 | 是特定社区投票 proxy | 记录人群与 selection bias |
| 32 | Wikipedia-like=事实正确 | classifier 学风格/分布 | 另测事实、许可和多样性 |
| 33 | bad-word filter 只删坏文 | 会误删引用/身份/安全讨论 | 分群体测误差 |
| 34 | classifier 0.9=90% 高质量真值 | score 依模型/训练标签 | 解释 positives/negatives |
| 35 | aggressive filtering 永远好 | 会损失 coverage/diversity/token量 | 画质量—数量曲线 |
| 36 | synthetic data 没有来源问题 | teacher/input/prompt 仍有 lineage | 记录全链 |
| 37 | exact dedup 只有一个定义 | normalization 会改变 equality | 公布规范化规则 |
| 38 | MinHash 给精确 Jaccard | 它是随机估计 | 增 signature、抽样复核 |
| 39 | Bloom “可能见过”就是见过 | 允许 false positive | 评估误删率 |
| 40 | near dedup 阈值越高删得越多 | 阈值越高越难判相似，通常删得更少 | 用小样本验证方向 |
| 41 | 去重只省存储 | 它重写 sampling weights | 比较前后来源分布 |
| 42 | PII regex 能解决隐私 | 组合信息、图像、metadata 会漏 | 多层检测+人工审计 |
| 43 | license 允许就代表有 consent | copyright许可与个人同意不同 | privacy 单独治理 |
| 44 | toxicity filter 对所有语言一样 | classifier error 可按群体变化 | 分语言测 precision/recall |
| 45 | 开放数据天然可复现 | URL 消失、工具/version 未给仍难复现 | 发布 manifest 与 hashes |
| 46 | benchmark 提升证明数据全面更好 | 指标覆盖有限，超参也会混杂 | 固定预算并多维审计 |
| 47 | 新 dataset 必然比旧 dataset 好 | 目标和来源不同 | 按任务、权利、覆盖选择 |
| 48 | permissive-only 已证明追平所有模型 | 课程只是特定快照实验 | 诚实报告 token/预算差距 |
| 49 | license detector 是法律判决 | 模板匹配会漏复杂层级 | 用作 signal，不作最终保证 |
| 50 | 删除记录即可忘掉影响 | 数据可能进派生集和 checkpoint | 维护 lineage、重训/修复策略 |

---

<a id="s23"></a>
## 23. 公式、单位与分类器速查卡

### 23.1 大小单位

- $`1\text{ KB}=1000`$ bytes，$`1\text{ MB}=10^6`$ bytes，$`1\text{ GB}=10^9`$ bytes，$`1\text{ TB}=10^{12}`$ bytes。
- $`1\text{ KiB}=1024`$ bytes，$`1\text{ MiB}=2^{20}`$ bytes，$`1\text{ GiB}=2^{30}`$ bytes，$`1\text{ TiB}=2^{40}`$ bytes。
- B 可能表示 billion（$`10^9`$）或 byte 的符号 B；由上下文判断。`156B tokens` 是 156 billion tokens，`156 GB` 是 bytes。
- document、word、character、token、byte 是五种单位，不能直接加减。

例：3GB 十进制 bytes：

```math
3\times10^9=3{,}000{,}000{,}000\text{ bytes}.
```

若粗假设平均 4 bytes/token，才可估：

```math
3{,}000{,}000{,}000/4=750{,}000{,}000\text{ tokens}.
```

这只是特定编码/文本的估计。

### 23.2 保留率、删除率

```math
\text{keep rate}=\frac{\text{kept}}{\text{input}},\qquad
\text{remove rate}=1-\text{keep rate}.
```

240T→3.8T：keep $`=3.8/240=1.583\%`$，remove $`=98.417\%`$。

### 23.3 Jaccard

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
```

交集 6、并集 10，$`J=6/10=0.6`$。分母不是 $`|A|+|B|`$，因为交集会被重复算。

### 23.4 Classification metrics

```math
\text{precision}=\frac{TP}{TP+FP},\qquad
\text{recall}=\frac{TP}{TP+FN}.
```

若 TP=30、FP=10、FN=20：precision $`=30/40=75\%`$，recall $`=30/50=60\%`$。

### 23.5 Mixture 与 epoch

**Epoch（轮）**是每个被定义为训练集的样本平均看一遍的口径。若 dataset 100B tokens、训练抽 300B token positions，粗略是 3 epochs；但有加权重采样时，某些 component 可能 10 epochs、另一些不到 1 epoch。

component (i) 的期望 token 数：

```math
T_i=w_iT,
```

其中 $`w_i`$ 是 mixture probability，总和为 1；$`T`$ 是总训练 tokens。若 $`T=1\text{T}`$、books 权重 0.2，则期望 books tokens $`=0.2\text{T}=200\text{B}`$。

视频 [79:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=4799s) 回到 live services→raw→processed 链；[80:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=4831s) 提问怎样从巨大 pool 选训练 tokens；[81:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4860s) 强调选择会显著改变结果；[81:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4890s) 预告下一讲会进一步讨论 classifier/rules/filtering。[81:52](https://www.youtube.com/watch?v=-qm0ln33G24&t=4912s) 是人工字幕最后 cue。

---

<a id="s24"></a>
## 24. 自测题（80 题）

> 80 题全部带【手算】【分类】【判断解释】【设计】或【填表】标签；至少先写一句理由或一行算式，再看 §25。

1. 【填表】source、raw copy、processed dataset 各是什么？用 Wikipedia 举例。
2. 【分类】“网页能在浏览器打开”只证明 access、ToS、license、copyright、privacy 中的哪一层？
3. 【手算】100 URLs：10 fetch 失败，成功页中有 20 个完全相同副本。按 §7 口径最多多少 unique documents？
4. 【手算】3.5TB 十进制是多少 bytes？
5. 【填表】WARC、WAT、WET 分别主要存什么？
6. 【手算】每页模板 8 words、正文 12 words，100 万页中模板 words 及其占比是多少？
7. 【判断解释】为什么 WET 不能唯一恢复 HTML？
8. 【判断解释】为什么 bulk dump 通常比爬网页 UI 好？仍没解决什么？
9. 【设计】给 Wikipedia dump 设计三项 poisoning 防守检查。
10. 【手算】原 repo 3 files，3 个未改 fork；总 file occurrences 与 unique content 各多少？
11. 【分类】public GitHub repo 没有 LICENSE：是 permissive、public domain、还是默认 copyright？
12. 【填表】arXiv metadata、PDF、LaTeX source 有何区别？
13. 【判断解释】BooksCorpus 的书价为 0，是否等于可任意训练？
14. 【判断解释】WebText 的 Reddit karma proxy 会带入哪两层 selection bias？
15. 【手算】C4 从 1.4T tokens 变 156B，保留率和删除率各多少？
16. 【分类】在 quality task 中，实际 positive=有用、predicted positive=留下。bad-word rule 删除一篇有用的仇恨言论研究文章，这是 TP/FP/FN/TN 哪格？
17. 【手算】训练 1000 tokens，web/books weights 为 0.6/0.4；期望各多少 tokens？
18. 【判断解释】为什么 classifier 的 0.82 不是“82% 事实正确”？
19. 【手算】四域各 100GB，按 bytes 均匀时各占多少比例？
20. 【分类】Books3→The Pile→下游模型体现 source、processing、lineage 中哪一概念最重要？
21. 【判断解释】Stack Exchange 按 score 排答案会改变什么？
22. 【手算】2.5T token pool 中训练 300B，粗略使用比例多少？
23. 【判断解释】MassiveText 有 10.5TB，为什么不能说 Gopher 看完全部一次？
24. 【填表】LLaMA 的 web、code、paper、Q&A 各列一个 component 和处理。
25. 【手算】RedPajama 1T→SlimPajama 627B，保留率和删除率多少？
26. 【判断解释】复现 LLaMA recipe 为什么不会 byte-identical？
27. 【手算】RefinedWeb 处理 5T、发布 600B，发布比例多少？
28. 【判断解释】“web data is all you need”为什么不能当普遍定律？
29. 【手算】两个 crawl 各 100 docs，第二个与第一个重叠 80 docs；unique 总数多少？
30. 【分类】Dolma 顶层 ODC-By、某 member license unknown；能否只凭顶层纳入？
31. 【手算】DCLM 240T→3.8T，保留率与删除率多少？
32. 【手算】DCLM positives 200K、negatives 200K，总训练 examples 多少？正类比例多少？
33. 【手算】在 quality task 中，positive=有用、predicted positive=留下。TP=35、FP=15、FN=5，precision 和 recall 各多少？各自用人话问什么？
34. 【分类】在同一 quality task 中，有用的 API reference 被 classifier 删除，是 TP/FP/FN/TN 哪格？
35. 【设计】怎样检测 DCLM-like classifier 是否伤害低资源语言？
36. 【手算】10T pool 留 10%，剩多少 tokens？
37. 【手算】ensemble 分数 $`e=0.8,d=0.6`$，简单平均是多少？阈值 0.65 留不留？
38. 【判断解释】synthetic rephrasing 增加了哪四项 provenance？
39. 【手算】Nemotron-CC 6.3T 中 HQ 1.1T，占比多少？
40. 【手算】The Stack 原论文约 51.76B file occurrences、5.28B unique，unique 比例多少？
41. 【分类】root LICENSE=MIT，子目录 vendored code=GPL；license detector 只报 MIT。能否按全仓 MIT？
42. 【判断解释】LLVM IR 为什么可能帮助低资源编程语言？边界是什么？
43. 【设计】把一个 PR linearize，至少写五个字段。
44. 【分类】作者保留权利文章被第三方放入 CC-BY collection，是什么风险？
45. 【判断解释】collection license 与 member license 为什么不能合并？
46. 【分类】teacher model 的输入来源不透明，生成的 QA 应把 provenance 记成“无来源”吗？
47. 【判断解释】exact duplicate 为什么必须先声明 normalization？
48. 【手算】10 records 中 A 有 8 份、B/C 各 1；去重前后 A 的采样占比是多少？
49. 【手算】集合 A={cat,sits,on,mat}，B={cat,sits,near,mat}，Jaccard 是多少？
50. 【手算】MinHash signature 8 个位置中 6 个相同，估计 Jaccard 是多少？
51. 【判断解释】MinHash 估计为什么不是精确 Jaccard？
52. 【分类】Bloom filter 回答“可能见过”，但文档实际没见过，这叫什么？会造成什么去重错误？
53. 【判断解释】near-dedup threshold 从 0.7 提到 0.9，通常判重更多还是更少？
54. 【设计】一份 dedup report 至少记录六项什么？
55. 【分类】regex 删了 email 但留下姓名+ticket id，是否可宣称无 PII？
56. 【手算】在“有用=positive、留下=predicted positive”的语言过滤任务中，两语言各有 1000 篇有用文，recall 98%/70%，各留多少？各产生多少 FN？留下量相差多少？
57. 【判断解释】toxicity word list 为什么会误删安全研究或群体自称文本？同一个无毒安全文本被删除，在两种 label contract 中分别是哪一格：① quality task：positive=有用、predicted positive=留下；② toxicity-removal task：positive=含毒、predicted positive=删除？
58. 【设计】列四项不包含攻击步骤的 poisoning 防守。
59. 【分类】CC-BY 博客含手机号：copyright/license 与 privacy 是否同时绿灯？
60. 【手算】3GB text，假设 4 bytes/token，粗估多少 tokens？为什么只是估算？
61. 【手算】240T→3.8T 的 remove rate 复算到两位小数。
62. 【手算】PII task 把“含PII=positive、删除=predicted positive”。TP=30、FP=10、FN=20，precision/recall 各多少？FN 在这里是什么风险？
63. 【手算】100B-token dataset 训练 300B token positions，粗略几 epochs？
64. 【手算】总训练 1T tokens、books weight 0.2，期望 books tokens 多少？
65. 【分类】浏览器可访问、robots 允许、但 ToS 禁 bulk download；应继续吗？
66. 【分类】美国 1900 年原著已有 public-domain 标记，但 2025 年译本刚出版；译本是否自动 public domain？
67. 【判断解释】summary judgment 为什么不能泛化为所有模型训练？
68. 【分类】Bartz 中训练用途与 pirated acquisition 应合并还是分开判断？
69. 【判断解释】settlement 为什么不是普遍判例？
70. 【设计】给一个 permissive code dataset 写五项 license audit。
71. 【填表】access、ToS、license/copyright、privacy 各写一个红灯例。
72. 【设计】作者撤回一篇文章时，provenance card 怎样帮助删除？
73. 【判断解释】为什么 GB、documents、tokens 不能直接互换？
74. 【分类】pool 有 1T unique tokens，训练跑 3T positions；这是 pool size 3T 还是约 3 epochs？
75. 【设计】比较两个 HTML extractors，怎样避免只看输出 token 数？
76. 【判断解释】benchmark 提升为什么不能证明数据在全部用途上更好？
77. 【分类】CommonPile 成绩不错能否证明其每条 member license 都正确？
78. 【设计】为新 web dataset 写一条 source→processing→audit→release 因果链。
79. 【判断解释】为什么“数据不会从天上掉下来”是技术、法律和组织三重命题？
80. 【综合】用不超过八步写出一条数据从 live webpage 到可审计训练样本的完整流程，每步至少写一个失败点。

---

<a id="s25"></a>
## 25. 自测答案

### 25.1 第 1–20 题

1. source 是 Wikipedia live/dump 的上游来源；raw copy 是某日期下载的 XML/WARC；processed dataset 是抽正文、按语言过滤、去重并版本化后的 records。三者不能同名混用。
2. 只证明技术 access。ToS 是否允许、作品 license/copyright、privacy/consent 都没有由“能打开”推出。
3. 成功页 $`=100-10=90`$。20 个相同副本只留 1 个，少 $`20-1=19`$ 个，所以 $`90-19=71`$ unique documents。
4. $`3.5\times10^{12}=3{,}500{,}000{,}000{,}000`$ bytes。题目写 TB，不用 $`2^{40}`$。
5. WARC→原始 crawl/HTTP response；WAT→从 WARC 算的 metadata/links；WET→抽取 plaintext。WET 最小但最有损。
6. 模板 $`=8\times1{,}000{,}000=8{,}000{,}000`$ words。总 words $`=(8+12)\times1{,}000{,}000=20{,}000{,}000`$。占比 $`8/20=40\%`$。
7. WET 已丢 HTML tags、layout、图片、脚本和部分结构；多个 HTML 页面可抽成同一文字，逆映射不是唯一。
8. Dump 降低服务器负担、版本更清楚、少 UI 噪声、易校验。它没自动解决 member license、copyright、PII、偏差和质量。
9. 任三项：保存 revision/time/hash；与前后 dumps 比异常；抽查临近 cutoff 的大改；高风险事实核来源；隔离异常账号/批量修改。不能写攻击步骤。
10. 原 repo 加 3 forks 共 $`4\times3=12`$ occurrences；因 forks 未改，unique content 仍 3。
11. 默认 copyright。Public 只让人查看/fork 的平台使用口径，不等于一个 permissive open-source license。
12. Metadata 是题名/作者/摘要等；PDF 是渲染版；LaTeX source 是结构化源文件/宏/引用。对象和许可可不同。
13. 不能。价格为 0 只说明不收费，不说明 copyright、license、ToS 或再利用授权。
14. 第一层：哪些 Reddit 用户发哪些 links；第二层：哪些用户给 karma。还叠加网站可下载/可解析偏差。
15. $`1.4\text{T}=1400\text{B}`$。keep $`=156/1400=0.11143=11.14\%`$；remove $`=100-11.14=88.86\%`$。
16. FN：它实际有用（positive），规则却预测删除（predicted negative）。
17. web $`=1000\times0.6=600`$；books $`=1000\times0.4=400`$ tokens。
18. 0.82 是在特定 positives/negatives、features 和校准下的模型 score，通常只表示更像正样本；事实正确未被逐条标成目标。
19. 四域等 bytes：每域 $`100/(4\times100)=25\%`$。若按 tokens/records 加权，比例可能变。
20. Lineage/provenance。要沿 Books3→Pile→下游追上游来源；换一个 bundle 名不会切断风险。

### 25.2 第 21–40 题

21. 它改变模型看到的 answer 顺序、正负例权重和对话因果；高票答案更靠前。必须记录是按时间、score 还是 accepted flag 排。
22. $`300\text{B}/2500\text{B}=0.12=12\%`$。
23. 10.5TB 是 corpus bytes；300B 是训练 token positions。编码单位不同，且训练只抽一部分/可重采样，不能说“完整一遍”。
24. 示例：web→CCNet Common Crawl+reference filtering；code→GitHub+permissive/manual filter；paper→arXiv+去 comment/展开 macro；Q&A→Stack Exchange+按 score 排 answers。
25. $`627/1000=62.7\%`$ keep；remove $`=37.3\%`$。不能把全部删除都归因 near duplicates，除非口径完全一致。
26. URL 内容和 crawl 时间变；私有 component 不可得；工具/阈值版本不同；随机 dedup/过滤和下载失败不同。
27. $`600/5000=0.12=12\%`$。
28. 它是特定论文在特定模型、预算、过滤和 benchmarks 下的经验结果；不覆盖所有任务、语言、法域与 future models。
29. 第一 crawl 100；第二新增 $`100-80=20`$；总 $`100+20=120`$ unique。
30. 不能。ODC-By 可能只覆盖 collection arrangement；member unknown 需单独核权或隔离。
31. keep $`=3.8/240=0.015833=1.58\%`$；remove $`=98.42\%`$。
32. 总 examples $`=200K+200K=400K`$。正类比例 $`200/400=50\%`$。
33. precision $`=35/(35+15)=35/50=70\%`$，问“留下的里面多少真有用”；recall $`=35/(35+5)=35/40=87.5\%`$，问“全部真有用的里面留下多少”。
34. FN：实际 useful（positive），却被分类器判为删除（predicted negative）。
35. 分语言建立人工标注 audit set；分别算 precision/recall；固定阈值比较保留率；检查方言/短文/混合语言；对错误样本人工归因；做阈值和下游 ablation。
36. $`10\text{T}\times0.10=1\text{T}`$ tokens。
37. $`(0.8+0.6)/2=0.7`$。$`0.7\ge0.65`$，按此规则留下。
38. 至少：teacher model/version、input source、prompt/template、sampling 参数；还应记过滤、生成日期、输出 terms/license。
39. $`1.1/6.3=0.174603\approx17.46\%`$。
40. $`5.28/51.76\approx0.1020=10.20\%`$。其余约 $`100\%-10.20\%=89.80\%`$ occurrences 不一定全是同一种重复，但数字显示复制规模大。

### 25.3 第 41–60 题

41. 不能。子目录 GPL 与 root MIT 冲突/分层，vendored code 不由 root maintainer 自动重新许可。需 file-level provenance。
42. 多种语言会编译到共同 IR，可提供结构对齐和迁移信号；边界是编译器、优化会丢高层语义，且并非所有项目可正确编译。
43. 示例：title、description、file path、surrounding context、diff、review comments、state/timestamps。至少五项并保留先后关系。
44. License laundering：第三方可能无权给上游文章重新贴 CC-BY。不能只信 collection label。
45. Collection license 可覆盖 selection/arrangement；member work 的 copyright/license 属于原作者。两层权利主体和条件不同。
46. 不应写“无来源”。应记录 teacher、prompt、input documents 和生成设置；不透明也要作为 known gap 明写。
47. 大小写、空格、标点、HTML 去除会改变 equality。同两 bytes 可不同规范化；不同 bytes 也可被规范化成同文。
48. 去重前 A $`=8/10=80\%`$。去重后 A/B/C 各一份，A $`=1/3=33.33\%`$。
49. 交集 3，包含 cat/sits/mat；并集 5，包含 cat/sits/on/near/mat；$`J=3/5=0.6`$。
50. $`6/8=0.75`$。这是估计，不是精确计算集合交并。
51. 只抽有限随机 hash/signature；不同随机排序会有抽样波动。signature 越长通常方差越小。
52. Bloom false positive。去重系统会把一个真正新文档误判“见过”，从而误删。
53. 通常更少。要求 $`J\ge0.9`$ 比 $`J\ge0.7`$ 更严格，只有更相似的 pair 才判重。
54. 任六项：normalization、unit(document/paragraph)、shingle size、hash/signature length、LSH 参数、similarity threshold、跨 dump/split 范围、代表保留规则、抽样误差审计。
55. 不能。姓名+ticket 可能组合识别个人；还需上下文和其他 metadata 检查。
56. 第一语言留下 $`1000\times0.98=980`$，FN为 $`1000-980=20`$；第二留下 $`1000\times0.70=700`$，FN为 $`1000-700=300`$；留下量相差 $`980-700=280`$ 篇。
57. 词表只匹配表面词形，不理解引用、否定、研究/防御语境和群体自称，因此会把实际无毒且有用的文本删除。先固定 label contract：①在 quality task 中，该文本实际有用（actual positive），却被判为删除（predicted negative），所以是 **FN**；②在 toxicity-removal task 中，该文本实际无毒（actual negative），却被判为删除（predicted positive），所以是 **FP**。物理错误相同，但正类定义不同，四格名称就不同；不能裸写“误删产生 FN”。
58. 任四项：保存 snapshot/hash；前后版本差异；异常来源/速率聚类；高风险多源核对；quarantine；小模型 ablation；撤回与 incident response。
59. 不同时绿。CC-BY 可能解决一部分 copyright permission 并要求署名；手机号仍有 privacy/consent 风险。
60. $`3\times10^9/4=0.75\times10^9=750{,}000{,}000`$ tokens。不同语言、UTF-8、标点和 tokenizer 的 bytes/token 不同，所以只是估算。

### 25.4 第 61–80 题

61. keep $`=3.8/240=1.5833\%`$。remove $`=100-1.5833=98.4167\%\approx98.42\%`$。
62. precision $`=30/(30+10)=75\%`$，问“所有被删文本中多少真含PII”；recall $`=30/(30+20)=60\%`$，问“所有含PII文本中删除多少”。FN 是含 PII 却被保留，构成隐私泄漏风险。
63. $`300\text{B}/100\text{B}=3`$ epochs 的粗口径。weighted sampling 时各 component 的实际遍数不同。
64. $`1\text{T}\times0.2=0.2\text{T}=200\text{B}`$ tokens。
65. 不应继续 bulk download。Robots 只是一个信号；明确 ToS 红灯需停止或取得授权/法律审查，不能用技术可访问覆盖合同层。
66. 不自动。原著与译文是不同表达；现代译者可能对译文有新 copyright。应核具体 edition、法域和许可。
67. 它只在特定 parties、claims、evidence 和 record 上裁判；不同 acquisition、作品、输出和市场证据可能改变结果。
68. 分开。Bartz 明确训练用途的 fair-use 分析不自动治愈盗版 acquisition/永久 library 行为。
69. Settlement 是双方解决争议的协议，通常不要求法院为所有争点建立可普遍适用的法律规则，也不等于承认全部主张。
70. 示例：保存 repo/commit/path；识别 root 与 nested licenses；排 vendored/generated files；核 copyright notices/attribution；记录 detector confidence/人工复核；处理无 license/多 license；维护撤回清单。
71. access→登录墙无账号；ToS→禁 automated bulk download；license/copyright→all-rights-reserved 且无授权/例外结论；privacy→含未经同意的医疗记录。任一红灯都不能被其他绿灯覆盖。
72. 通过 source URL、record id、content hash、dedup cluster 和 derived dataset lineage 找到原记录及副本；写 tombstone；重建受影响版本；通知下游并记录 checkpoint 处置边界。
73. GB 测 bytes，documents 测记录数，tokens 取决于 tokenizer；语言、格式、压缩、平均文档长度都会改变换算。只有声明平均值和误差后才能估。
74. Pool 仍是约 1T unique tokens；训练抽 3T positions，粗略约 3 epochs，不应把重复训练说成 3T unique pool。
75. 固定同一 WARC sample；人工标正文/模板；比较 precision/recall、语言/域名切片、重复率、速度；再用相同模型预算做下游 ablation；保存版本和失败案例。
76. Benchmark 覆盖有限；数据还会改变安全、偏差、事实性、长尾和其他语言；训练超参/随机性也可能混杂。只能在固定协议下解释。
77. 不能。成绩是模型/数据效用证据，不是 member-level license audit。许可正确性需独立 provenance 与法律审查。
78. 例：seed/官方 dump→WARC+timestamp/hash→HTML extraction→lang/quality/PII→exact+near dedup→人工分层审计→member license manifest→版本化 release+撤回通道。
79. 技术上要 crawl/extract/filter/dedup；法律上要分 access/ToS/license/copyright/privacy；组织上要持续人工审计、记录、响应投诉和重建。三者都不会自动完成。
80. 一种合格链：①确定 source/owner（失败：未知来源）；②合法/礼貌取得 snapshot（失败：auth/ToS）；③保存 raw+hash（失败：不可复现）；④ extraction（失败：模板污染）；⑤ lang/quality/PII filtering（失败：群体误差）；⑥ exact/near dedup（失败：误删/权重偏）；⑦ license/privacy/poison audit（失败：laundering/泄漏）；⑧发布 versioned manifest 并维护撤回（失败：下游不可追踪）。

---

<a id="s26"></a>
## 26. 视频时间导航（完整人工字幕）

下表只使用人工 `English (United States)` 字幕中的真实 cue；正文和下表不重复同一秒。时间显示与 `t=` 秒数相同。

| 时间 | 课堂内容 | 笔记 |
|---:|---|---|
| [00:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=29s) | 为什么 data 最重要 | §2.1 |
| [00:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=58s) | 公司保密数据的原因 | §2.1–2.2 |
| [01:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=88s) | 过去依赖人工标注 | §2.3 |
| [01:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=119s) | 数据是 long-tail 人力问题 | §2.1 |
| [02:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=150s) | mid-training | §2.3 |
| [02:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=179s) | 三阶段只是基本模板 | §2.3 |
| [03:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=210s) | 大模型不一定发布 base | §2.3 |
| [04:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=241s) | web、论文、数学等来源 | §2.4、§8 |
| [04:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=270s) | 怎样选择和处理数据 | §3、§20 |
| [05:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=301s) | Internet 上的 live service | §3.1–3.2 |
| [05:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=331s) | 不能直接在 live server 上训练 | §3.2 |
| [05:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=359s) | crawler 也抓不到全部网页 | §3.2、§4 |
| [06:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=392s) | dynamic app 例 | §3.3 |
| [07:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=420s) | 登录与付费墙 | §3.3 |
| [07:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=450s) | 技术访问边界 | §4.2 |
| [08:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=480s) | crawler user agents | §4.1–4.2 |
| [08:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=511s) | robots 预期行为 | §4.4 |
| [09:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=541s) | IP/国家封锁 | §4.2 |
| [09:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=570s) | rate limit | §4.2–4.3 |
| [10:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=600s) | Decline of Consent 研究 | §4.5 |
| [10:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=629s) | 多种限制叠加 | §4.4–4.5 |
| [11:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=660s) | 历史网页限制较少 | §4.5 |
| [11:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=689s) | robots 是 guidelines/protocol | §4.4 |
| [11:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=719s) | 文档站被 crawler 压垮 | §4.3 |
| [12:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=751s) | server load 与成本 | §4.3 |
| [12:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=778s) | shadow library 例 | §4.6 |
| [13:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=810s) | 跨国服务器与执法 | §4.6 |
| [13:57](https://www.youtube.com/watch?v=-qm0ln33G24&t=837s) | web 是巨大而复杂的来源 | §3–4 |
| [14:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=871s) | 进入法律边界 | §5 |
| [15:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=900s) | intellectual property | §5.1 |
| [15:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=929s) | copyright/patent/trademark/trade secret | §5.1 |
| [16:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=961s) | 美国 Copyright Act | §5.1–5.3 |
| [16:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=991s) | collection 的选择/编排 | §5.1、§16.3 |
| [17:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=1021s) | published 到 fixed | §5.2 |
| [17:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=1051s) | copyright 与 patent 登记差异 | §5.1 |
| [17:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=1079s) | copyright registration | §5.1 |
| [18:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=1108s) | 创作者固定作品即可能有权利 | §5.2–5.3 |
| [19:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1140s) | “网上基本都有版权”的直觉 | §5.2 |
| [19:28](https://www.youtube.com/watch?v=-qm0ln33G24&t=1168s) | license 的人话直觉 | §5.1、§5.4 |
| [20:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1200s) | Creative Commons 背景 | §5.4 |
| [20:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=1229s) | CC materials 与 permissive 条件 | §5.4 |
| [21:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=1261s) | 商业数据许可案例 | §5.4 |
| [21:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=1290s) | fair use 四因素开场 | §5.5 |
| [22:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=1321s) | purpose/character | §5.5 |
| [22:25](https://www.youtube.com/watch?v=-qm0ln33G24&t=1345s) | nature of work | §5.5 |
| [23:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1380s) | market effect | §5.5 |
| [23:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=1410s) | fair-use examples | §5.5 |
| [23:57](https://www.youtube.com/watch?v=-qm0ln33G24&t=1437s) | Google Books 类案例 | §5.5 |
| [24:33](https://www.youtube.com/watch?v=-qm0ln33G24&t=1473s) | copyright 不只逐字复制 | §5.2、§5.5 |
| [25:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=1502s) | parody/exception 边界 | §5.5 |
| [25:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=1530s) | 数据准备中的复制 | §5.5 |
| [25:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=1558s) | transformative 只是论点之一 | §5.5 |
| [26:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=1589s) | idea 与 concrete expression | §5.2 |
| [26:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=1618s) | fair use 事实密集 | §5.5 |
| [27:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=1650s) | ToS 是额外一层 | §5.6 |
| [28:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=1682s) | 诉讼部分开场 | §6 |
| [28:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=1738s) | pirated acquisition 独立问题 | §6.2 |
| [29:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=1769s) | Kadrey/Meta | §6.3 |
| [30:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1800s) | 课程法律状态总结 | §6.5 |
| [30:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=1831s) | 来源法律课堂问答 | §5–6 |
| [30:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=1859s) | 课堂问答边界 | §6.5 |
| [31:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=1889s) | 错误再许可问题 | §16.2–16.3 |
| [31:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=1919s) | web 持续产生新内容 | §7.1 |
| [33:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1980s) | Common Crawl 课程规模快照 | §7.1 |
| [33:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2039s) | frontier 加链接 | §4.1、§7.1 |
| [34:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2070s) | revisit policy | §4.3、§7.1 |
| [43:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2609s) | 学生问数据限制 | §8.7、§5 |
| [43:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=2639s) | pirated books 问题 | §4.6、§6 |
| [44:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2670s) | 可训练来源的谨慎边界 | §5、§20 |
| [45:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=2700s) | 极谨慎的数据选择 | §20 |
| [47:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=2822s) | 早期研究已知 Common Crawl | §9.2–9.4 |
| [47:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=2878s) | 数据设计选择值得审计 | §9.2–9.6 |
| [49:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=2969s) | C4/T5 转场 | §9.4 |
| [50:32](https://www.youtube.com/watch?v=-qm0ln33G24&t=3032s) | T5 text-to-text 背景 | §9.4 |
| [51:59](https://www.youtube.com/watch?v=-qm0ln33G24&t=3119s) | WebText-like C4 子集 | §9.4 |
| [52:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=3150s) | 数据处理影响 benchmark | §9.4、§22 |
| [53:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=3180s) | Common Crawl overlap | §9.4–9.5 |
| [54:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=3241s) | GPT-3 quality classifier | §9.5 |
| [54:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3269s) | positives 定义“高质量” | §9.5、§13 |
| [57:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3449s) | Q&A 接近模型使用形式 | §10.5 |
| [58:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=3482s) | Stack Exchange dump | §10.5 |
| [58:57](https://www.youtube.com/watch?v=-qm0ln33G24&t=3537s) | Gopher manual rules | §11.1 |
| [59:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=3569s) | corpus size 与训练 token 不同 | §11.1 |
| [61:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=3660s) | 数据保密动机回看 | §2.2、§11.2 |
| [65:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=3902s) | Dolma 结果与开放处理 | §12.3 |
| [68:31](https://www.youtube.com/watch?v=-qm0ln33G24&t=4111s) | classifier ensemble / synthetic 转场 | §14.2–14.3 |
| [70:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4200s) | Nemotron-CC 结果 | §14.4 |
| [70:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=4229s) | classifier 训练思想回顾 | §13–14 |
| [72:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=4350s) | 低资源编程语言 | §15.3 |
| [76:01](https://www.youtube.com/watch?v=-qm0ln33G24&t=4561s) | permissive sources 构成 | §16.1 |
| [76:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=4589s) | CommonPile 工程难度 | §16.2–16.4 |
| [78:29](https://www.youtube.com/watch?v=-qm0ln33G24&t=4709s) | 诚实报告许可边界 | §16.4–16.5 |

---

<a id="s27"></a>
## 27. 官方源码 1–622 行覆盖

### 27.1 版本

使用 commit `8b59b50730766695c2ffedd1a79c50cd09b9eb91` 的 `lecture_13.py`；SHA256 为 `BF7A17C20528C58394DFEFC3B840076CC10AAAEA6860AFFF341D3DF4196F01D7`。Python `splitlines()` 得 622 个物理行。

“覆盖”表示每个连续 line range 有正文教学映射，不表示把展示代码逐行重抄。

### 27.2 无 gap、无 overlap 的连续索引

| 物理行 | 官方函数/内容 | 正文 |
|---:|---|---|
| 1–5 | imports、课程基础设施 | §27.1、§29 |
| 6–47 | `main`：全讲顺序与 summary | §0–2、§19、§30 |
| 48–89 | `motivation`：data secrecy、训练阶段、OLMo 图 | §2 |
| 90–149 | `raw_sources`：live web、crawler、限制、shadow libraries | §3–4 |
| 150–240 | `copyright`：IP、license、fair use、ToS、lawsuits | §5–6 |
| 241–273 | `common_crawl`：规模、crawler、WARC/WET、extraction | §7 |
| 274–296 | `wikipedia`：scope、contributors、dumps、poisoning | §8.1–8.3、§18 |
| 297–317 | `github`：repo/metadata、duplicates、Software Heritage | §8.4–8.6 |
| 318–329 | `arxiv`：PDF/LaTeX/metadata/license/bulk | §8.7 |
| 330–341 | `bert` | §9.1 |
| 342–352 | `books_corpus` | §9.1 |
| 353–363 | `gpt2_webtext` | §9.2 |
| 364–378 | `ccnet` | §9.3 |
| 379–406 | `t5_c4` | §9.4 |
| 407–420 | `gpt3` | §9.5–9.6 |
| 421–439 | `the_pile` 与 component 引入 | §10.1–10.2 |
| 440–448 | `project_gutenberg` | §10.3 |
| 449–456 | `books3` | §10.4、§6 |
| 457–468 | `stackexchange` | §10.5 |
| 469–488 | `gopher_massivetext` | §11.1 |
| 489–503 | `llama`、RedPajama、SlimPajama | §11.2–11.4 |
| 504–522 | `refinedweb`、FineWeb | §12.1–12.2 |
| 523–538 | `dolma` | §12.3–12.4 |
| 539–558 | `dclm` | §13 |
| 559–577 | `nemotron_cc` | §14 |
| 578–599 | `the_stack`、Stack v2、PR linearization | §15 |
| 600–618 | `common_pile` | §16 |
| 619–622 | main guard | §27.3 |

### 27.3 函数覆盖与运行边界

`main` 调用 motivation/raw_sources/copyright/common_crawl/wikipedia/github/arxiv/bert/books_corpus/gpt2_webtext/ccnet/t5_c4/gpt3/the_pile/project_gutenberg/books3/stackexchange/gopher_massivetext/llama/refinedweb/dolma/dclm/nemotron_cc/the_stack/common_pile。上述函数均映射到正文。

`if __name__ == "__main__": main()` 表示文件直接运行时进入讲义；本笔记没有在浏览器 lecture runtime 中重新执行互动展示，使用固定源码、图片和字幕核对语义。

---

<a id="s28"></a>
## 28. 图片覆盖：18 次调用、18 个唯一资产

### 28.1 核验方法

- 源码引用 14 张本地图、4 张远程图，均无重复。
- 14 张本地图用原始分辨率逐张目视核验，不只读文件名。
- 4 张远程图已下载到 inspection 目录并查看原图；crawler SVG 转 PNG 后透明线条较淡，语义同时与官方源码邻近文字核对。
- 图里的排行榜、模型数值和 web 规模只按课程快照解释。

### 28.2 逐图语义

| 源码行 | 资产 | 原图尺寸/访问 | 目视核验到的语义 | 正文 |
|---:|---|---|---|---|
| 55 | llama3-data.png | 1608×257 | Llama 3 report 对 architecture/recipe 说明多，对 pretraining data 只给高层信息 | §2.2 |
| 81 | olmo2-pretraining.png | 1478×711 | OLMo 2 pretraining 数据 mixture/阶段图 | §2.3–2.4 |
| 83 | olmo2-dolmino.png | 1219×1214 | Dolmino mid-training 数据与曲线/表 | §2.3–2.4 |
| 85 | tulu.png | 1613×1294 | Tülu post-training 数据 mixture 与流程 | §2.3–2.4 |
| 128 | decline-consent.png | 1345×1041 | robots/ToS restrictions 随时间和数据来源上升的多面板图 | §4.5 |
| 131 | anthropic-crawling.png | 760×554 | 网站方描述 bot traffic/server load 的帖文快照 | §4.3–4.5 |
| 254 | WebCrawlerArchitecture.svg.png | 远程，已下载 | seed/frontier/fetch/parser/new URLs 的 crawler 架构箭头 | §4.1、§7.1 |
| 271 | dclm-wet.png | 486×180 | 不同 HTML extraction/WET pipeline 的小型下游比较 | §7.3 |
| 399 | c4-domains.png | 远程，已下载 | C4 来源域分布，说明 web corpus 并非中性均匀 | §9.4 |
| 427 | the-pile.png | 远程，已下载 | The Pile 22 components 的组成图 | §10.1–10.2 |
| 525 | Dolma composition image | 远程，已下载 | Dolma 多来源比例/组成可视化 | §12.3–12.4 |
| 544 | dclm-filter.png | 1949×912 | DCLM-pool→classifier filtering→baseline 的 pipeline | §13.1–13.2 |
| 556 | dclm-quality.png | 1370×526 | 固定协议下不同 filtering recipes 的模型结果比较 | §13.5 |
| 575 | nemotron-results.png | 1902×384 | Nemotron-CC subsets 与基线的课程快照结果 | §14.4 |
| 597 | stackv2-pr1.png | 437×521 | PR 页面/结构对象的第一部分 | §15.4 |
| 597 | stackv2-pr2.png | 691×480 | diff、review/context 线性化示意 | §15.4 |
| 609 | commonpile.png | 1266×451 | permissive-source components/数据量组成 | §16.1–16.4 |
| 617 | comma-results.png | 1831×570 | Comma model 与若干历史模型的结果快照 | §16.5 |

图能证明“讲义当时展示了什么”，不能独立证明 dataset 所有 license 或因果结论。

---

<a id="s29"></a>
## 29. 来源与证据边界

### 29.1 课程来源

- [官方 lecture_13.py 固定 commit](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_13.py)，622 个物理行，SHA256 见 §27。
- [Stanford Online 完整视频](https://www.youtube.com/watch?v=-qm0ln33G24)；人工 `English (United States)` 字幕 1419 cues，首 cue 00:04、末 cue 81:52。
- 讲义的 18 次 image calls 已按 §28 核验。

### 29.2 官方法律资料与法院文件

- [U.S. Copyright Office: What Is Copyright?](https://www.copyright.gov/what-is-copyright/)、[Circular 15A](https://www.copyright.gov/circs/circ15a.pdf)：保护对象和期限。
- [17 U.S.C. §107](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A107+edition%3Aprelim%29)、[§302](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A302+edition%3Aprelim%29)：House 法典官方文本。
- [Copyright Office Fair Use Index](https://www.copyright.gov/fair-use/) 与 [AI Part 3 training report](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)。
- Bartz：[Justia Filing 231 全文页](https://docs.justia.com/cases/federal/district-courts/california/candce/3%3A2024cv05417/434709/231)、[RECAP docket](https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/)、[Document 231 fair-use order PDF mirror](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf)、[Document 680 final settlement approval mirror](https://law.justia.com/cases/federal/district-courts/california/candce/4%3A2024cv05417/434709/680/)。
- Kadrey：[RECAP docket](https://www.courtlistener.com/docket/67569326/kadrey-v-meta-platforms-inc/)、[Document 598 court-file mirror](https://law.justia.com/cases/federal/district-courts/california/candce/3%3A2023cv03417/415175/598/)。

RECAP/Justia 是法院提交文件的公开镜像，不是法院官网；PACER 在当前环境无法匿名完整读取。NYT 案件未取得可稳定匿名访问的官方完整 docket，本笔记只报告课程对 complaint/pending posture 的描述，不把 allegation 写成法院事实。所有法律内容是教学材料，不是法律意见。

### 29.3 官方数据基础设施

- [Common Crawl Get Started](https://commoncrawl.org/get-started)、[formats explanation](https://commoncrawl.org/blog/web-archiving-file-formats-explained)：WARC/WAT/WET 和访问方式。
- [Wikimedia dumps](https://dumps.wikimedia.org/) 与 [Wikimedia licensing](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)：dump 与许可边界。
- [GitHub licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)：public repository 与 license 的区别。
- [Software Heritage archive](https://www.softwareheritage.org/archive/)、[2025 官方活动报告](https://www.softwareheritage.org/2026/01/16/software-heritage-activity-report-2025/)与[官方 statistics API](https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api-stat.html)：软件保存基础设施与动态规模口径。
- [arXiv bulk data access](https://info.arxiv.org/help/bulk_data_s3.html) 与 [license information](https://info.arxiv.org/help/license/index.html)。

### 29.4 主要原始论文/技术报告

- [BERT](https://arxiv.org/abs/1810.04805)、[BooksCorpus](https://arxiv.org/abs/1506.06724)、[GPT-2/WebText](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)、[OpenWebText](https://arxiv.org/abs/1908.07195)。
- [CCNet](https://arxiv.org/abs/1911.00359)、[T5/C4](https://arxiv.org/abs/1910.10683)、[Documenting C4](https://arxiv.org/abs/2104.08758)、[GPT-3](https://arxiv.org/abs/2005.14165)。
- [The Pile](https://arxiv.org/abs/2101.00027)、[Gopher/MassiveText](https://arxiv.org/abs/2112.11446)、[LLaMA](https://arxiv.org/abs/2302.13971)。
- [RefinedWeb](https://arxiv.org/abs/2306.01116)、[Dolma](https://arxiv.org/abs/2402.00159)、[DCLM](https://arxiv.org/abs/2406.11794)、[Nemotron-CC](https://arxiv.org/abs/2412.02595)。
- [The Stack](https://arxiv.org/abs/2211.15533)、[Stack v2](https://arxiv.org/abs/2402.19173)、[Common Pile](https://arxiv.org/abs/2506.05209)。

论文能支持作者报告的 source、processing 和实验，不自动证明所有 upstream rights 或未来模型结果。

### 29.5 证据层级

- 【课程内容】：固定源码、图片、字幕；动态数字标 2026 快照。
- 【补充解释】：本笔记自建小数字、表格和决策树，用来教学，不伪装成论文实验。
- 【补充】：官方文档、法条、法院文件或原始论文。
- 课程引用的新闻、Wikipedia、社交媒体只用于理解讲义当时叙事；法律结论不以这些二手材料为唯一依据。

---

<a id="s30"></a>
## 30. 学完后的能力清单

你现在应当能：

- 画出 live service→crawl/raw→extract→filter→dedup→mix→dataset→model 的完整链；
- 区分 Internet、public web、技术 access、ToS、license/copyright 和 privacy；
- 解释 WARC/WAT/WET，手算 extraction 模板污染比例；
- 读 dataset card 时分清 bytes、documents、tokens、pool size 和 training tokens；
- 用小集合手算 Jaccard，并解释 MinHash/LSH/Bloom filter 的误差；
- 从 confusion matrix 计算 classifier precision/recall，识别低资源语言和方言被误删；
- 给 BERT、WebText、C4、Pile、LLaMA、FineWeb、Dolma、DCLM、Nemotron、Stack、CommonPile 写统一 provenance 卡；
- 解释 public repo、collection license、member license、synthetic provenance 和 license laundering；
- 准确复述 Bartz/Kadrey 只适用于特定 record，不把 complaint、summary judgment、settlement 混为一谈；
- 为新 dataset 设计可撤回、可重建、可审计的 manifest 与 decision tree。

最后只记一句：

> 模型吃到的不是“互联网”，而是人在特定时间、权限、工具、规则和价值判断下做出来的一份版本化数据产品。
