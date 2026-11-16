# Codex 本地执行任务｜03 Linear Attention Skip Lookups — 官方 GDN-2 强基线验收

**性质：执行任务书，不是研究结论。** 任务主体是本地 Codex。日期 2026-11-16。主 Agent 已审计 PR #1–#9。本文档刻意把先验事实、待检验假设、止损条件分开。请按下面的阶段自行执行，遇到可修复错误先排查；不要只写一份计划或向用户复述已有报告。

## 一、你是谁、接手什么

你负责对 fork **`CreoEnMi-p/llm-research-paper-drafts` 的第 03 篇初稿**进行本地高可信度实验，验证“previous-read 预测性跳写”是否在真正的强基线面前仍有研究价值。主 Agent 不允许把失效的 CPU 自实现当作官方 GDN-2，不允许没有数据语义审查就批量跑模型，不允许只汇报好看的平均数。

用户的本地电脑规格未知；**先识别系统、显卡、显存和 CUDA**，不能假设有 NVIDIA GPU。如果没有可用 GPU，记录阻塞证据并停止官方 GDN-2 训练；不要回退到 `--mix Z`，更不要自己租机器。

**工作基线分支：**
```bash
git fetch origin
git switch -c research/03-local-official-gdn2-validation origin/research/03-interleaved-hardness-pilot-20261115
```
如果本地分支已存在，安全切换，不要重建或覆盖。先检查 `git status`，不覆盖用户未提交的改动；有冲突则先保护已有改动。所有实验代码放独立分支，不修改 `main`；无须等用户审批每个可逆研究步骤，但不做付费云计算、破坏性操作或未获授权的资源开销。

**务必先读：**
- `03-linear-attention-skip-lookups/rc.py`，`mix.py`，`gate_exact.py`。
- `03-linear-attention-skip-lookups/audit_stream_task.py`、`audit_stream_eval.py`、`test_stream_task.py`、`test_stream_hardness.py`。
- `RESEARCH_03_GDN2_STRONG_BASELINE_AND_RANDOM_STREAM_VERDICT.md`。
- `RESEARCH_03_MEMORY_UPDATE_COUNTEREXAMPLES_2026-11-15.md` 可从 `research/03-update-compute-matched-20261115` 分支阅读。
- 官方 NVIDIA 代码：https://github.com/NVlabs/GatedDeltaNet-2 ，尤其 `README.md`、`requirements.txt`、`LICENSE`、`lit_gpt/gdn2.py`、`lit_gpt/gdn2_ops/{chunk_gdn2.py,fused_recurrent_gdn2.py}`。外部代码优先独立 clone 并固定 commit SHA，不复制受限许可代码到我们的公开 fork。

**已有实验证据（不要重复跑）：**
- 第一种随机读写任务（4 初始 key + 2 新 key）：普通 Delta Rule `--mix D`，300 步后独立 96 条样本上自回归全部 96.45%、旧 94.27%、覆盖 97.63%、新 97.71%。成功 Actions https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43832170838。
- 更难的随机读写任务（8 初始 + 4 新 key，10 后续写入、30 后续查询）：同一模型，300 步后自回归全部 **95.83%**，旧 92.07%、覆盖 97.24%、新 98.98%。成功 Actions https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43832622685。
- 这两种任务已经被 D 近乎解决，**不能把它们作为大型 Gate 对比的发现平台**。如果 GDN-2 完整能力也已饱和，就把它们作为训练与数据验收的 *smoke test*，不要为了寻找 1–2pp 的漂亮提升再跑多种子。
- `--mix D` = 普通 Delta Rule；`--mix e` = 我们的简化 scalar erase；`--mix Z` = 先前 **未通过 learnability 验收的 CPU 公式级模仿**；三者都 **不是** 官方 Gated DeltaNet-2。尤其不能用 `Z` 的失败分数声称打败 GDN-2。
- Previous-read 原始 chunk-start Gate 在更改 chunk 大小时会改变预测；`--lpsI 1` 是慢速、tokenwise、自洽参考实现，**不是高效模型**。current-key Gate 是重要竞争对照。

## 二、阶段 0：硬件与真实软件依赖（禁止进入训练前绕过）

在用户本地工作区检查：
```bash
git status --short
git branch --show-current
nvidia-smi
python -c "import sys,platform,torch; print(sys.version); print(platform.platform()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'NO_CUDA')"
```
Windows 请考虑 WSL2+Linux 的 CUDA 兼容性；Mac/MPS/AMD 不等于官方 NVIDIA CUDA Triton 的有效环境。失败输出原样保留，不假设能安装成功。

写 `local_experiments/gdn2_environment.md` 记录 OS、显卡、显存、驱动、Python、PyTorch、CUDA、`nvcc`/Triton/FLA、磁盘空间、git SHA、时间和实际可用性。不要打印用户本地路径以外的个人机密或保存 token。若无 CUDA 或官方 kernel 不支持显卡，就**停止 GPU 任务**，返回“blocked: no supported GPU”，不启动 CPU 伪 GDN-2 对比。

若有合适 GPU：新建独立 venv/conda 或容器，优先按官方 `requirements.txt` 记录的 PyTorch/CUDA 基座及其手动安装 FlashAttention / Flash Linear Attention 的说明来配置。官方基础镜像说明为 `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`，但仍应以当前 upstream 的实际 requirements 为准。避免污染用户已有 ML 环境。构建失败要诊断日志，不许静默改成非官方近似实现。固定 upstream commit 与 license；不启动 1.3B/100B-token 预训练，不下载 FineWeb 大数据集。

## 三、阶段 1：只验证官方 GDN-2 kernel 与 layer 是否可用

**必须从上游 import 原版 `lit_gpt/gdn2.py` 的 `GatedDeltaNet2`（及它调用的官方 kernel），不把我们自己的 `Z` 改名。**

先写最小自动化 smoke tests：
1. 选择官方支持的维度，极小 batch、短序列（例如 B=2、T=128/256、隐层 128 或 256；从上游 API 确认真实有效组合），使用模型默认相关 gate、norm、conv 和 beta safeguard。
2. CUDA 前向和 loss.backward() 都完成；关键投影/decay/erase/write 参数有有限且非零梯度；无 NaN/Inf；峰值显存与前后向耗时可报告。检查官方代码对 `forward` 参数、`past_key_values`、dtype 的真实契约，不从仓库外猜 API。
3. 对同一权重/输入，比对官方 `chunk` 前向与官方 `fused_recurrent` 推理输出和连续增量状态。注意 **recurrent 路径 forward-only，不能拿它训练**；长度 >64 的 case 可以避免上游自动切换短序列推理路径。chunk/recurrent 的精度容差按实际 fp32/bf16、相关上游测试约定预先声明，允许合理浮点误差，不能为了“通过”临时放宽几个数量级。
4. 至少跨过 chunk 分界（例如 T=128 或 256），核对末尾输出、state continuity，尽可能验证不同分块预填充与逐 token decode 等价。**如果官方 kernel 自己不一致，要确认复现方式/缓存用法，而不是直接判官方论文错误。**
5. 输出 `local_experiments/gdn2_smoke_report.md`、可直接执行的复现脚本、日志、上游 commit。

任一核心检查失败 → **不进入训练**。先修复运行环境或测试调用；若无法解决，明确阻塞后退出，禁止跑 3 seed。

## 四、阶段 2：学习能力验收，只做一个种子，不调参找赢面

为原有 `mqar_stream` 数据写 **官方 GDN-2 layer 的薄适配器**：保留官方投影、独立逐通道 erase/write/decay、beta guard、output gated norm 和官方 chunk 训练路径。只在我们的外层接 token embedding、块堆叠/FFN、预测 head。**不要更改 NVIDIA 核心算子来适配 toy loss。** 该适配器是“官方 GDN-2 层用于新 toy 任务”，不是论文同规模官方训练复现。

训练输入和数据校验直接复用现有 `audit_stream_task.py`。必须先运行 `test_stream_task.py` 和 `test_stream_hardness.py`；第三方代码可另加 test，禁止禁用严格断言。

只跑一个预注册种子，使用现有两种**已被 D 解决的随机 interleaved** 数据，作为官方层的 **learning smoke test**，不是研究挑战。先短小版（T96，6 active keys）再有需要时较难版（T144，12 active keys），不要一开始就跑很长的训练。为官方新参数采用 upstream 的合理 init/优化 recipe，而非盲目照搬 toy Muon 大学习率；训练前声明学习率、warmup 和 batch，评估只看 train/validation，独立 test split 保持封存。

可采用既有 learnability 判据：`teacher-forced overall >=82%`，`old/overwrite/new 各 >=75%`，每类 >=100 个查询，**连续两次 eval 达标**。如果官方层未达标：
- **不开始 Gate 多种子对比**；
- 核对 loss mask、token 位置、learning-rate scaling、模型维度与官方层初始化；
- 允许**一次有明确技术理由的修复/重新 smoke**，记录原始失败和改动；
- 仍未达标则按“官方 GDN-2 toy adapter 当前未通过学习能力验收”报告，**不归因于官方模型方法失败**。
- 若刚好通过，但 AR 误差过高或无法复现，停止进入研究对比。

仅在通过 validation gate 后，以独立 heldout seed92742 做 answer-only greedy AR（先前真实写入保持正确，前面生成的查询答案进入后续上下文），记录 old/overwrite/new 分组、`stale old value` 比例、每类样本数。输出训练曲线、参量、总 tokens、train wallclock、peak VRAM、hardware version。

## 五、阶段 3：只在有信息价值时，设计下一组反证

**不要在现有 T96/T144 近满分任务上跑 previous-read Gate 与官方 GDN-2 的 3×N 批量矩阵。**

先在代码层分析 Delta/GDN2 对哪些状态变化**可能**失效：重复覆盖、key 方向高度相似导致键空间干扰、大规模同时存储导致固定状态容量瓶颈、旧值复用/跨跨度的读取；随机 event order 本身已不是有效差异。任何候选 challenge 必须符合：
- 用可审计的流式 oracle 保证因果标签正确、必要写入与查询位置都有效；避免角色标签泄漏，也避免删除标记后造成不可解歧义。
- **固定候选条件，最多 2 个仅供单种子筛选**，先运行 D 和官方 GDN-2（或先只 D 作廉价可学性测试），以分组数据判定是否真正存在可学习但未饱和的 gap；别扫参数直到某模型表现差。
- 若 D 与官方 GDN-2 都达到 >95% 的关键自回归子组准确率，该任务信息价值低，STOP。
- 若 D 和官方 GDN-2 都不能学基本关联（初始/未修改旧 key 各 <75%，或明显随机猜测），**视为无效试验设置**，STOP。不能拿别人学不会做新方法的优势。
- 如果基础记忆学会而特定的 *interference* / 必要写入子组持续存在失败（尤其可解释的 stale-value 或错误抑制），先检查是否被更充分训练、角色/当前 key 的强基线消除。只有强基线真的存在新缺口，才值得展开后续研究。

最后才允许做 **小规模配对方法对比**：官方 GDN-2，原生 D，exact previous-read+margin，current-key+margin，学得会的 local-role control（如有）和按目标选取的现代强对照。模型若不能在参数量/隐藏状态规模完全一致，则在结论中明确两种方案的参数/状态/计算差异，必要时补匹配变体。记录按步数/训练 tokens 与实际同硬件 GPU time 的比较，避免拿 CPU serial exact Gate 和 CUDA kernel 的墙钟时间直接对比。PS 现在是慢速参考，不能声称训练/部署更高效。

**若想声称在 GDN-2 上补充创新，优先考虑官方 GDN-2 vs 同架构 GDN-2 + proposed rule 的严格 ablation**；把跨架构小 toy 胜负当作初筛，不当论文证据。绝不改上游模块后继续叫它“原版官方 GDN-2”。

## 六、预算硬限制和过程记录

- 先检查环境、依赖、数据和数值正确性，**这些全部通过才分配训练显存**。
- 第一阶段 tiny kernel smoke 尽可能在几分钟完成；**单次 toy baseline 训练最多约 10 GPU 分钟**（需 GPU sync 测量）；任何一次 smoke 到上限还未通过就停止并报告瓶颈。首次探索的总 GPU 训练预算建议**不超过 30 GPU 分钟**，后续大型/多 seed 另行报告成本和依据，再由用户决定资源投入。若本机更慢，提前降低步数或先停止，不偷偷延长。
- **绝不**预先分发 3 seed × 4 model 矩阵，绝不在失败基线上烧同样配置的更多种子，绝不为追求某个方法赢而无穷调整数据/学习率。
- 每次训练命令、模型参数数、所有 seed（训练/验证/测试）、实际 token 数、checkpoint 选择依据、软件 SHA、GPU 最大分配显存和 elapsed seconds 必须写 `local_experiments/run_manifest.jsonl`；支持断点恢复但不能把未运行记录说成完成。
- 数据训练/调参可看 train+validation；独立 test 冻结。任何设计因 validation 结果变化，注明“探索性改变”，冻结新测试集后才允许声称在该新测试集评估；若多次使用相同 test，不再称其为未接触 test。

## 七、你最终要交付的不是流水账，而是结论

在独立研究分支提交（不合并 main）：
- 环境与官方上游定位、commit/licensing 记录；
- 官方 GDN-2 的 unit/numerical/grad smoke 代码与日志、可复现依赖/入口；
- 若真实可训练，**1 个** GDN-2 baseline 的完整 learning gate + AR 子组结果和显存、时间；
- 只有前面全通过，才提交预注册单点 challenge 的结果；不扩充昂贵矩阵；
- 一页 `local_experiments/RESEARCH_DECISION.md`：表格明确区分「已验证」「未验证」「失败/无效」，回答 **普通 Delta / 官方 GDN-2 是否已解决干扰？previous-read 还有没有独立空间？下一次 GPU 分钟是否值得？**，给出明确 GO / NO-GO。

**执行顺序：真正的官方实现 → 先证明基线能学 → 再先找反例 → 最后才验证新 Gate。用户要求的是科研方向的可靠证据，而不是算力消耗。**
