# Independent audit — Pointer Doubling (2026-11-16)

Scope: [original paper](../PAPER.md), [original results](../RESULTS.md), [original code](../pdg.py), and independent [causal-order oracle](causal_oracle.py). Base revision: `4e5f54d93bc27ec2cb1587414fa1ac056b359c6a`.

## Decision (provisional)
**Continue only as a narrow, falsifiable mechanism study, not yet as a generally useful LLM attention improvement.** No exact same combination was found in the papers checked, but log-depth pointer jumping, tied-key associative retrieval and learnable graph search are established prior art. The new claim must be *learnability/robustness/transfer under realistic constraints*, not the logarithmic hop-count bound itself.

### Literature boundaries
- Sanford, Hsu & Telgarsky, ICML 2024, [arXiv:2402.09268](https://arxiv.org/abs/2402.09268): logarithmic-depth k-hop construction is known.
- Merrill & Sabharwal, NeurIPS 2025, [arXiv:2503.03961](https://arxiv.org/abs/2503.03961): logarithmic-depth graph connectivity is known.
- Saparov et al., 2024, [arXiv:2412.04703](https://arxiv.org/abs/2412.04703): trained transformers can expand graph reach exponentially with layers, though scaling learnability is hard.
- Zhang et al., ICLR 2025, [arXiv:2405.06394](https://arxiv.org/abs/2405.06394): associative identity matching is not novel.
- Li, 2025, [arXiv:2508.02631](https://arxiv.org/abs/2508.02631): layerwise pointer chains, distinct objective and complexity, still important comparison.
- Ma & Sojoudi, **November 10, 2026**, [arXiv:2611.13594](https://arxiv.org/abs/2611.13594): training dynamics and exponential-depth graph search; missing from initial related work.

**Novelty at best** is a specific learnable causal head with a doubled pointer stream, doubled no-backtrack and gated second pointer. This should be compared directly to frozen/algorithmically initialized variants and to learnable nearest-neighbor retrieval, not only vanilla/Primer/KV-shift baselines.

## Evidence already present
Original author reports 218 runs (single seed per configuration, mostly 50–100 optimization steps), reverse-order 7-hop/3-layer and 15-hop/4-layer near 0.98 and 31-hop/5-layer near 0.96. Strong performance in the *constructed reverse-order lookup* is plausible from the code architecture. The current two-pointer head does **not** improve the single FineWeb language-model cell; that test is too small to establish real-text performance.

## Independent algorithmic experiment (executed in Python/NumPy)
[causal_oracle.py](causal_oracle.py) abstracts the **same causal P and hard-gated Q update**, with unique tokens, perfect exact-key match, and optimal per-query gate schedule. This removes learning failure and softmax ambiguity. It is a test of this recurrence, **not neural training** and **not an impossibility theorem for arbitrary transformers**.

| Test | Reverse stage order | Forward stage order | Random stage order |
| --- | --- | --- | --- |
| 7 hops / 3 layers | reachable | max 3 hops | **315/5040 = 6.25%** of all block permutations reachable |
| 15 hops / 4 layers | reachable | max 4 hops | **0/3000** sampled permutations reachable; mean maximal reach 7.76 |
| 31 hops / 5 layers | reachable | max 5 hops | **0/3000** sampled permutations reachable; mean maximal reach 12.22 |

Random results for 15/31 hops are samples, not zero-probability claims. Code: `python3 audit/causal_oracle.py`. **Interpretation:** the reverse-order precondition is structural, not merely a poor optimization run. Perfect attention and oracle gates do not rescue general ordering.

## Major design gaps ranked by decision impact
1. **Query-dependent hops**: `pdg.py` uses `S['pg']` of shape `(nl,)` (one global scalar per layer); each run fixes `--hops`. A single model has not learned query-dependent k. To test, encode `k` in each query, mix k within each batch, and evaluate held-out k/lengths. Global fixed gates alone cannot express different k for identical start and graph.
2. **Causality/order**: reverse-stage order is artificial in general code/text. Predecessor doubling rescues *forward* order at half reach, but random/mixed dependency order remains unsolved. First test mixed ordering with a properly matched dual-direction version; if no improvement, restrict claims to structured memory tables.
3. **Appropriate controls**: current comparisons use 50–100 steps, and do not include trained structured pointer-jump/algorithmic retrieval, frozen-random exact-match pointer stream plus trained readout, equal wall-clock/FLOPs, or long-trained standard attention. Re-running 218 nearby cells adds little evidence.
4. **Robustness/semantics**: duplicates drop recall (reported 0.763), 14/4 and 18/5,20/5 fail, synonyms and realistic fact wording are untested, and vocab stages are disjoint. Test same-token reuse, aliasing, masked/conflicting edges, nonuniform chains, and instructions describing the hop count.
5. **Practical value**: one FineWeb cell with 150 steps has no positive language-model result, and the head adds extra full T×T softmax/similarity operations beyond the normal attention; matching parameter count is **not** matching compute. Need actual FLOPs/memory/latency and a text/code task where real multihop utility exceeds that cost.
6. **Reproducibility**: `pdg.py` fixes `torch.manual_seed(0)` and training `np.random.default_rng(0)`; no seed flag. Its default `--dev mps` makes unmodified `reproduce.sh` fail on Linux CPU/CUDA systems. Add seed/device selection and 3+ seeds per decision-critical cell.

## Next 3 decisive tests
**E1 — fixed versus query-conditioned gates.** Mix k∈{3,4,5,6,7} at 3 layers in *one training run*, query includes k. Compare globally learned scalars, learned per-query bits, and explicit bit-selection with equal training budget; report individual k accuracies and held-out k. Failing this falsifies general-purpose binary lifting.

**E2 — causal mixed-order generalization.** Train dual-direction head on reverse+forward+random permutations, evaluate each separately on novel samples and distractors. Use exact oracle above to bound what the existing recurrence can achieve. Failure on random order means position-structured memory only.

**E3 — honest compute and stronger baselines.** At 7/3 and 15/4, 3+ seeds, train to equal FLOPs/wall time (not only equal steps). Compare (a) standard causal attention, (b) tied identity one-hop, (c) frozen-random pointer doubling + trainable readout, (d) current full, (e) an explicit algorithmic key-value pointer-jump implementation. Test 1k+ steps if baseline still learning.

**Go criterion:** E1+E2 pass (or a compelling real-world ordered-data application is demonstrated), E3 shows a meaningful compute-adjusted improvement, and one realistic text/code retrieval cell beats strong baselines. **Otherwise stop** pursuing an ICML claim that this is a general LLM head; retain it as a specialized learned algorithm.


## Additional close prior work and genuine application evidence
- **MemReasoner**, Ko et al. (NeurIPS 2024 workshop) / Das et al. (2025), [arXiv:2503.07903](https://arxiv.org/abs/2503.07903), is a particularly important missing architecture comparator. It end-to-end learns a temporal ordering of facts and **iteratively updates a memory query**; experiments cover variable tracking, distractors and weak supporting-fact supervision. It is **not** the same logarithmic-depth P-table construction, but already addresses the learning-to-hop pain with an explicit memory mechanism. Contrast depth, memory cost, supervision, order/generalization, and query-dependent hops.
- [RULER](https://arxiv.org/abs/2404.06654) (COLM 2024) includes **variable_tracking** with configurable chains and hops, a better next benchmark than adding more disjoint-vocabulary toy cells.
- Assogba & Ren, [*Evaluating Long Range Dependency Handling in Code Generation LLMs*](https://machinelearning.apple.com/research/evaluating-long-range), TMLR 2025, find multi-step key retrieval performance declines on long code and call-graph-informed prompt rearrangement can improve it substantially. **The pain is real, but cheap structural preprocessing is a strong baseline.**
- Reverse-order edges are not inherently unnatural: some *use-def backward dependencies* in code place the referenced definition before its use. Test real code-like alias/dataflow chains with lexical binding, repeated variable names, and distracting assignments rather than asserting all reverse order is artificial.


## Executed neural experiments — original pdg.py, GitHub Actions CPU
All tests use the original model code checked out on GitHub Actions, with narrowly scoped ephemeral patches in the workflow (training seed, frozen-Q rows, or initialized gate bits). Dataset `hopx`, reverse order, width 64, batch 16, eval 256 rows, seed 12345 for evaluation; PyTorch CPU with 2 threads. These tests are **independently executed**, not numbers copied from the initial paper. Full logs provide commands, training curves and JSON.

### Original-code 7/3 reproduction
[Run 43946265567](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43946265567), 50 steps:

| Arm | Seed | Validation accuracy | Steps to 0.9 |
|---|---:|---:|---:|
| Tuned attention | 0 | 0.174316 | Never by step 50 |
| Full BLN | 0 | 0.982910 | 25 |
| Full BLN | 1 | 0.980957 | 25 |

Seed 1 modifies the training torch seed and NumPy training generator; evaluation examples are held fixed. This corroborates the specialized task effect, with **only two BLN seeds**.

### Pointer-Q learning ablation
[Run 43947012092](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43947012092): Freeze the first two heads' Q projection rows during **all training steps** by restoring their initialized values after the Muon update; all other parameters (including embeddings, gates, V/O/MLP) train normally.

| Cell | Accuracy after Q freeze | Steps to 0.9 |
|---|---:|---:|
| 7 hops, 3 layers, 50 steps | 0.979004 | 25 |
| 15 hops, 4 layers, 50 steps | 0.966797 | 25 |

**Interpretation:** trained Q projection changes are *not required* for near-ceiling performance here. The core exact-match retrieval rule is largely injected as a hard architectural prior; training still changes many other weights, so this is **not** evidence that the whole model works untrained. Test a frozen-random full pointer mechanism with only readout trained next.

### Diagnose failing hop counts with target-bit initialization
[Run 43947383139](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43947383139): Initial `S['pg'][layer]` to +8 for set bits and -8 for unset bits of fixed known `--hops` (near-binary, **not frozen during training**). Else original BLN+no-backtrack. This supplies the correct task-wide gate schedule in advance; it is a **mechanism/optimization diagnostic**, not a valid solution to dynamic mixed-k retrieval.

| k/L | Default gate init | Target-bit gate init | Steps |
|---|---:|---:|---:|
| 14/4 | 0.518066 (own reproducible run) | **0.993652** | 100 |
| 18/5, pairs=4 | 0.288 (original author log) | **0.935547** | 50 |
| 20/5, pairs=4 | 0.237 (original author log) | **0.953125** | 50 |

**Interpretation:** all three originally failed counts are representable by the model and learn readily when the gate schedule is initialized correctly. Main bottleneck is **learning/discretizing gate bits from answer loss**, not insufficient nominal binary reach. This is a diagnosis, not a complete causal explanation of optimization dynamics.

## Revised research priority after independent experiments
1. **Highest priority: query-conditioned gate design with mixed k, plus control on explicit bit selection.** Without a per-query k embedding or end-of-chain stop condition, the static gates do not solve genuinely mixed-length reasoning. Compare learned bits, supplied bits and global scalar gates on the same mixed-k dataset.
2. **Second: realistic ordering, repeated identifiers, and aliases.** RULER variable_tracking is directly applicable. Test normal source code with *reverse use-def order*, forward reference, lexical shadowing, and noncontiguous distractors. If restricted to reverse-order associative-memory data, say so.
3. **Third: compute-matched prior architecture baselines**, especially MemReasoner and an explicitly programmed/baked-in pointer-jump oracle, alongside longer-trained standard attention.
4. **Only after the above:** fused-kernel optimization and larger language-model training. Current single tiny FineWeb cell has no gain; spending on 125M+ before query-dependent usefulness would be premature.

**Current go/no-go:** Continue as a **targeted architecture research hypothesis**; reject claims that a general-purpose, trainable multi-hop attention improvement is already established.


## New experiment E1 — mixed-hop query-conditioned control (actual neural training)
Source: [mixed-k reproducible generator](build_mixedk.py) creates a patched copy of the original `pdg.py` with a **single model per arm**, random query-specific hop count k∈[1,7], a count token before each query key, original reverse-ordered 7-stage key/value graph and the same final-answer-only supervision. Comparison arms:
- `global`: unchanged scalar per layer (global `pg[l]`);
- `gate-table`: task-k-conditioned learned `gt[l,k]`;
- `oracle-bits`: explicitly supplied binary gate bits +8/-8 from count token (not trained; upper-bound *controller*, not upper-bound final accuracy);
- `content-gate`: learned gate from the embedding of the count token preceding the query, `sigmoid(e(count)·w[l]+pg[l])` (no explicit k-bits supplied).

All use original BLN pointer mechanism. Same seed 0, same validation rows across arms; single seed, no cost-normalized standard-Attention competitor yet. `acc_by_hop` and query counts are stored in each run's JSON and logs.

### E1a: eight queries in one row
[CPU Actions run 43948872849](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43948872849), 150 steps, batch=16, eight queries per row.

| Arm | Aggregate accuracy |
|---|---:|
| global | 0.521484 |
| gate-table | 0.758789 |
| oracle-bits | 0.756836 |
| content-gate | 0.798828 |

**Interpretation:** per-query gates materially help on this controlled setting. No arm reaches 0.9. `oracle-bits` is not a score ceiling because it only fixes the gate and other readout/attention weights remain trained. Earlier answered queries appear in later queries' causal context, creating potential repeated-token distractors.

### E1b: one query per independent row
[CPU Actions run 43949460087](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43949460087), 250 steps, batch=64, one query per row (previous query answers no longer in context), same randomized k∈[1,7].

| Arm | Aggregate accuracy | Accuracy by k=1,2,3,4,5,6,7 |
|---|---:|---|
| global | 0.546875 | .757, .973, .494, .855, .046, .047, .570 |
| gate-table | 0.843750 | .851, .945, .805, .971, .742, .766, .823 |
| oracle-bits | 0.882812 | .987, 1.000, .874, 1.000, .742, .766, .798 |
| content-gate | **0.888672** | .905, .904, .828, .986, .955, .844, .823 |

Eval N=512 rows, ~64–87 queries per hop; per-k uncertainty is nontrivial. The **learned content-conditioned gate** is +34.2 pp over global in this particular configuration. But neither the learned gate nor the explicitly provided binary-bit controller exceeds 0.9 overall at 250 steps.

**Caveat on causal interpretation:** E1a and E1b use different step counts, batches and evaluated query counts, so this does NOT isolate the causal effect of prior-answer distractors. A fully matched multi-query vs single-query experiment with the same update/eval budget would be required. The residual errors despite oracle bits could stem from learned readout, softmax collisions, gate/auxiliary-state dynamics, or insufficient optimization; none is established as uniquely causal.

**Important scope:** the count is explicit in a *synthetic count token* and sampled from the same 1–7 range during training. This proves neither automatic hop-count inference from natural language nor generalization to unseen hop counts or new domain-specific graphs.

### E1 conclusion and next experiment decision
- **Confirmed:** static global layer gates are inadequate for general mixed-k tasks under these settings; learned query-conditioned gating provides a material improvement in one seed.
- **Not yet confirmed:** near-perfect dynamic mixed-k success or real text/code transfer. The correct binary bits themselves are not sufficient to guarantee model prediction success.
- **Next:** decompose pointer correctness vs final head readout on single-query samples, inspect oracle-gated P/Q trajectories at each layer, then train multiple seeds to >90% with a genuine input-conditioned halting/gating rule and benchmark realistic RULER variable-tracking with repeated symbols.
- The branch's Actions workflows are **manual-only** after the tested runs to prevent unrequested repeated computation.
