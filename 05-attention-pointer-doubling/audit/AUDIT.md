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
