# Paper 05 — ICML decision gate after independent learnable occurrence-pointer trials
**Research status:** NO-GO for a general attention-head or novel log-depth pointer algorithm claim **as of verified 2026-11-17 experiments**. The mechanism can be a productive research direction, but its current unique contribution is not established. All code and experiments are on `research/05-pointer-doubling-audit` (Draft PR #10), never `main`.

## Executive result
A trainable sparse position-pointer module, 595 learned parameters, solves the **original pinned Sanford `InductionHopsFullSequenceTask`** terminal-only k=7 benchmark. The target is the very last character **only**; no intermediate hop or position supervision. It gets 100% both character and exact occurrence-position accuracy on held-out samples and generalizes to length400 and unseen k15 (length200).

**Critical qualifier:** A parameter-matched **serial one-hop recurrence using exactly the same matcher** also reaches 100% under matched training. In the terminal-only **single-query** regime, it is *faster* than sparse doubling. Hence the new result demonstrates a learnable *occurrence-aware pointer*, **not an indispensable advantage of doubling**.

### A. Original architecture is not rescued by 20k updates
[Complete five-seed 20k run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046440163), [results](sanford_terminal7_20k_fiveseed_verified.json): official terminal k7, char4, no invalid targets, 20k steps × batch48 AdamW LR.001, 4-layer RoPE **32.89±0.70 percentage points SD** and original BLN **33.67±1.96**; four-symbol uniform chance 25%. No reliable model advantage. Direct first-pointer probe: BLN's newest-occurrence Top-1 at initialization 0%; near-tie recency can fix first hop, but the Q stream fails on hop3/hop7 even with oracle gate bits. The original architecture loses occurrence identity.

### B. New trained location state — direct terminal-only k7
New [trainable model](train_sparse_occurrence.py) has tied initial orthogonal token embeddings, Q/K initially identity-aligned, learned Q/K matrices, learned log-distance recency coefficient initialized 0, a learned self-loop threshold, and a Top-K (`K=4`) table of **occurrence indices** and differentiable selected weights. No hard exact token-ID mask; Top-K *index selection* itself is nondifferentiable. k7 composition is `A @ A² @ A⁴` in sparsely pruned path form. The full candidate logits still cost **O(T²d)**, so don't claim globally linear attention.

[Five-seed, 1000-step official run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44127390187), [curves/results](official_sparse_5seed_1000_results.json):
- Doubling K4: 5/5 seeds **100%** final character and true position, validation **k7/L100 n512, k7/L200 n256, k7/L400 n128, k15/L200 n256** per seed.
- Identical recurrent K4: **4/5** seeds succeed; fifth optimization collapses at chance.
- Freeze recency strength at zero: **0/2** succeed.
- K2 2/2 succeed; K8 1/1 succeed; dense1/1 succeeds.
- This experiment **already gives Q/K correct initial matching geometry** (shared identity-initialized Q/K). A high positive result alone cannot justify claim of learning the matching algorithm from scratch.

### C. Critical training initialization and curriculum ablation
[Random QK original, 1200 steps](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44127755953), [raw](qk_inductive_bias_1200_results.json): independent random orthogonal query and key projections (no pre-alignment) fail 3/3 seeds, ~24–28% final char and zero exact occurrence positions. Increasing K to8 alone does not rescue. This **is not an impossibility result**.

[Temperature×curriculum](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44129596452), [raw](low_temperature_qk_curriculum_verified.json):
- Random Q/K, direct terminal k7 training, 1600 steps, `initial_temp={4,8,16}`: all **0/6** fail to solve.
- Random Q/K, initial temp4 or8, **400-step k1 terminal curriculum with dense differentiable attention**, then 1200 steps k7 sparse: **4/4** seeds reach 99.8–100% exact final char and 100% actual position. Temperature16 warmup fails 2/2.
- This demonstrates a *non-prealigned* Q/K matcher can be **learned from final-answer-only k1 and k7 labels** if the optimization schedule is carefully designed. It is **not pure k7 terminal-only training** and uses a known simple k1 scaffold.

**Decisive [five-seed matched recurrence](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44130187189), [raw](random_qk_curriculum_5seed_fair.json):** with identical random Q/K, initial temperature4, 400-step dense k1 terminal curriculum and 1200-step k7 sparse, batch48 AdamW:
- Doubling K4 **5/5** at 100% correct character **and exact occurrence**, all four validation settings including OOD k15 and length400.
- Recurrent one-hop K4 **5/5** at exactly the **same 100%** in those conditions.
No identifiable accuracy/generalization advantage for doubling under this fair condition. This is a **hard negative for the uniqueness thesis**.

### D. Compute — one query vs all-position parallelism
[Same CPU runner, terminal-only](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44128555445), [profile](occurrence_cost_verified.json): batch16, 2 threads, forward+backward median; k7 length100 doubling5.92ms vs recurrence **5.00ms**; k7 length400 doubling68.5ms vs recurrence **52.36ms**; k15 length400 doubling70.24ms vs recurrence **53.31ms**. **Sequential one-hop is cheaper** for a single terminal query. Both share O(T²d) candidate similarity and 595 trainable parameters.

[All-position official task profile](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44129269542), [raw](official_all_position_compute_verified.json), all non-null outputs scored, same initialized strong match+recency, **no trained model checkpoint**: k15 length100 doubling5.20ms vs recurrence7.83ms, length200 12.95ms vs17.39ms, length400 46.48ms vs53.09ms. This is a *legitimate query-parallel amortization regime*, not proof of GPU advantage, equal accuracy or differentiable end-to-end advantage. Fixed initial matcher yields ~85–93% non-null output accuracy, different for the two sparsification strategies.

### E. External real-code and scope benchmark are NOT passed
Previous Python-executed normalized assignment programs: BLN4 handles simple assignment-time snapshot/shadowing but fails unseen 8 writes **even when names are all different** (matched-length control), so duplicate-name ambiguity is not the sole failure. A near-logit recency rank boosts both repeated and unique-key cases, yet direct reading a global name after function-local shadowing remains unreliable; exact-ID recency likewise fails. [Full controls](scope_matched_length_results.json). We **have not trained the new sparse occurrence model on raw Python programs, reassignments, nested scopes or realistic code**. A language-model byte loss over real Python source is not an authentic use-def reasoning benchmark.

### F. Prior art/novelty risk
- [Sanford, Hsu, Telgarsky ICML 2024](https://proceedings.mlr.press/v235/sanford24a.html): log-depth causal transformer construction and learned k-hop induction.
- [Pointer (arXiv:2508.02631)](https://arxiv.org/abs/2508.02631): explicit position-chained pointer models and linear O(NK) long-range reasoning architecture, distinct designs but directly competing.
- [Wu, Geiger, Millière ICML 2025](https://proceedings.mlr.press/v267/wu25j.html): trained transformers can learn symbolic variable binding and chain dereference without handcrafted pointer architecture.
- [NVIDIA RULER](https://github.com/NVIDIA/RULER): existing external variable tracking generator; not equal to actual code assignment semantics.
The present implementation's **O(T²d)** candidate score computation and standard binary lifting are not enough for novelty, and the recurrence baseline already solves the headline task.

## Explicit GO / NO-GO
**NO-GO now:** submit as "general-purpose better Attention", "new log-depth k-hop algorithm", "learned Sanford reasoning requiring pointer doubling", or "real-code tracking". The data does not support any of these claims.

**Potential next GO gate:** A model with learned position+binding+scope state operating on *real program text* (not oracle-resolved bindings), which beats tuned RoPE5/6 and matched one-hop pointer on held-out deep aliases, 8/16 reassignment lengths, lexical shadowing, true runtime execution, and actual non-template source. Report ≥5 seeds, independent example intervals, equal-budget wall/GPU memory and measured FLOPs. Alternatively show substantial **parallel all-position GPU speedup** over a fused recurrent pointer at matched quality, not just 1.1–1.5× untrained CPU.

If neither gate passes, the scientifically responsible direction is to **shelve paper05 as an engineering/teaching demonstration** rather than expand synthetic experiments.

### Pending, not evidence
- [RoPE L5/L6 official terminal 20k](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44127469803) (9 jobs; compare 5-layer theoretical-depth baseline); not finished when this document authored.
- [RoPE L5/L6 k1→k7 matched curriculum](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44130348908) (6 jobs), not finished when authored.
No unfinished job has been counted as a successful result.

**Repository boundary:** branch `research/05-pointer-doubling-audit`, Draft PR #10, no edits to `main`.
