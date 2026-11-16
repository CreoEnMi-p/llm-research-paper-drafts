# ③ Linear Attention Skip Lookups — 立项前给导师的证据简报

**日期：2026-11-15（美西） | 判断：现象值得讨论，方法暂不以 ICML 级新架构立项。**

## 科学问题（而非模型名字）

线性注意力以有限状态写入 key–value 关联。MQAR 中模型生成一个正确答案后，答案 token 也可能被写入状态，使后续相关 key 的读出受到干扰。我们问：**什么时候一个写入真正有害？能否低成本、因果地识别它，而不误杀之后必须加入的新关联？**

## 已复现的最小正结果

1. **冻结后干预，因果性较强：**原始双阶段 MQAR，5 个配对训练种子，直接关闭已有 checkpoint 的答案位置写入：g 平均 answer-loss **0.572→0.287**；delta 在同时关闭 erase 和 write 后 **0.888→0.459**。关闭 query-key 更新几乎无影响。说明这个特定任务/模型下**答案位置的状态更新对后续检索有害**。证据：[run 43800060886](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43800060886)。
2. **自回归仍存在：**新测试集 96 条序列 ×16 个查询、3 个模型种子，g 训练 1500 步后的自回归准确率 **92.73%**，冻结该 checkpoint 在答案位置禁写，变成 **97.94%**。所有之前生成的答案由模型自行预测，不用 teacher forcing；但这是利用已知任务结构的“oracle”，不是可部署门控。证据：[run 43804365839](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804365839)。
3. **一个小幅优化线索：**独立测试集、固定 margin=0.2、3 配对种子，previous-read 门控训练 300 步，真实自回归准确率 **93.36%**（CPU 训练 ~25 秒）；g 基线 1500 步 **92.73%**（~33 秒）。配对差 +1.04、+0.39、+0.46 个百分点；仅有 3 种子，符号检验单侧 p=0.125，**不足以证明稳定 0.63pp 效果或声称优越性**。同等 margin 的 **current-key 读出为 93.58%**，更不能主张 previous-read 独有机制。证据：[PR #4](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/4)。

## 已证伪 / 尚未通过的顶会门槛

- **原 oracle 的 Delta 实验有定义错误：**只关 write 未关 erase；历史某表格数值缺少可追踪对应日志，不能引用作性能上界。见[独立审计 PR #1](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/1)。
- **原 predicted-skip 没有 chunk-independent 语义：**改 chunk size，前向输出会大幅变化；当整个序列为一块时 Gate 直接失效。Gemini [PR #2](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/2) 提供数学自洽的逐 token 对照，但训练比无门控明显更慢，尚不是高效 kernel。
- **无 margin 版在真实自回归条件下不稳：**3 种子只有 1 个好于 baseline，另 2 个更差。fixed margin 才是单独后续候选；不能把它的结果倒灌为原方法的成功。见 [PR #3](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/3)。
- **新写入不受损尚未证明：**`mqar2` 读/写阶段交错实验显示提升，但新写入 key 均与旧 key 不同；没有**旧 key 覆盖、同值不同 key**等反例。该任务 exact gate ~100 秒训练 vs 基线 ~22 秒。见 [Gemini 结果 run 43802940476](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43802940476)。
- **强基线缺席：**当前的代码 `e` 是简化 scalar erase，不是 NVIDIA 2026 **Gated DeltaNet-2** 完整通道 erase/write/decay。PH-Attention 等既有 associability/read-write 方法也没有严格对比。现有“记忆写入干扰”机制已在 **How Linear Attention Remembers**（arXiv:2609.33093）等工作中研究，概念新颖性有限。二维合成任务并不能证明真实语言泛化。

## 下一步只做两个决定性实验（不再随意调整 gate）

**A. 强基线反证（第一优先）：**在相同关联检索数据、训练 token、模型参数规模下复现真正 GDN-2；记录有无干预的答案写入影响、准确率和单位计算预算。**如果成熟门控已经消掉干扰或覆盖我们的 tiny margin gain，停止单独追求 predicted-skip 论文。** 不可把 `e` 简化版冒充 GDN-2。

**B. 必要写入反证（仅在 A 显示真空白时）：**随机查询与写入交错，包含 (i) 新 key 绑定以前出现过的同一 value，(ii) 旧 key 被覆盖为新 value，(iii) 之后查询旧值和最新值的正确性。对**保留旧记忆**和**完成新写入**分别评分。如果仅靠压制答案写入提高分数，却损害新关联学习，则不能称通用 write-skip 机制。

通过 A/B 之后才考虑由逐 token 参考发展高效、chunk-independent kernel，并测试自然序列和大模型；否则以机制审计结果结题、转向其他候选稿件。

**与老师讨论的核心问题：**这是否是既有门控尚未解决的*真实记忆干扰*？有没有合适的小规模预训练模型/资源能直接验证 A？现有证据支持“值得做一次有期限的判定实验”，不支持“已有新方法可投稿”。
