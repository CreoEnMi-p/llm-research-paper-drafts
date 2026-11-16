# Gemini proposed patch: independent research audit

**Branch**: `research/03-gemini-gate-audit-20261115`  
**Scope**: Engineering/experimental audit of advice from a separate Gemini agent. This is **not** a new research-direction proposal. Do not merge into main while PR #1 (oracle audit) is active or until all controls are reviewed.

## Claim-by-claim disposition

| Gemini assertion | Audit status | Reason / needed evidence |
| --- | --- | --- |
| Chunk-start gate is blind to within-chunk writes | **Confirmed** | For `C=L`, initial state is zero, gate factor is exactly 1. Old gate and baseline have identical seed-0 loss 0.650108. |
| The supplied `lpsI` matrix patch is *exact* and nearly free | **Rejected** | Within-chunk read uses pre-gate `beta*v`, not effective gated writes; delta family does not include within-chunk erasures. The proposed read cannot equal its own fully gated state recurrence. An exact serial oracle works but is slower. |
| Stop-gradient is necessary to prevent collapse | **Unproven** | Seed-0 isolated `detach` worsens C64 loss 0.202899 -> 0.261195. Need independent seeds and gradient-mechanism measurements. |
| Cosine margin `0.2` removes random false positives | **Partly motivated, not proved** | For unrelated normalized random vectors in 16 dimensions, margin `0.2` still activates on roughly 22% of pairs. Seed-0 `margin` improved loss 0.202899 -> 0.177473, but needs repeated seeds. |
| Its two-step smoothed cosine implements PH-Attention | **Rejected** | Published PH-Attention (arXiv:2508.08289) uses an explicit feature-indexed Pearce–Hall associability state; the proposed control is not that algorithm. Do not label it a published baseline. |
| Interleaved `mqar2` establishes safe online updates | **Only partially tested** | Existing `mqar2` appends distinct new keys after an earlier query. It never overwrites the same key with a conflicting value, so it cannot test correctness of memory overwrites. Add phase-specific gate diagnostics and a strict current-key comparator. |
| One-seed P4 / C64 / LM success proves top-conference novelty | **Rejected** | Accuracy gains on synthetic teacher-forced MQAR are task-specific; no official strong published baseline, natural text reproducibility, compute normalization, or efficient GPU kernel. |
| Suppression-only ceiling `gf<=1` is a fatal bug | **Rejected as bug** | The operator intentionally only suppresses updates relative to learned base `beta`; a learned base gate may increase from its initial 0.5. It is a modeling constraint needing comparison to alternatives, not evidence of a broken implementation. |
| Margin 0.2 eliminates chance cosine matches | **False as stated** | For two independent uniform unit vectors in dimension 16, `P(cos>0.2)≈0.221` (exact beta-distribution geometry). Margin removes many but not all random activations. |


## Verified published PH-attention comparator (2026 revision)
The current paper at [arXiv:2508.08289v3, Section 5](https://arxiv.org/html/2508.08289v3#S5) gives a specific feature-indexed associability state `a_t` updated with unsigned error weighted by squared key features, and a learned cue-dependent rate `beta_t`. This is **not** a two-timestep average of `cos(S k_t, v_t)` and cannot be implemented by Gemini's `lpsP`. In its own regime-switch/mixed-reliability evaluation (Appendix C, Table 3), PH-attention did **not** beat DeltaNet or the global-surprise ablation; the paper explicitly reports this negative result. Therefore, do not present mere error-based write suppression as a new principle or claim that PH-type gates necessarily improve recall.

## Independent correctness reference

`gate_exact.py` implements, for each position `t`, a gate calculated from **the actual** prior post-update state `S_(t-1)`, then applies the current gated write (optionally gated erase), then computes the output. It is deliberately sequential and is **not a hardware-efficient replacement** for chunkwise kernels.

`test_gate_exact.py` checks:
- Without predicted-skip, `gate_exact` equals the original chunked recurrence for `g,d,D,e` across chunk sizes.
- With `C=1`, original gated chunked matches the exact sequential reference (independent implementation check).
- The sequential reference is invariant to the `lpsC` parameter.
- Backward gradients remain finite; detach changes the backward pathway, not the forward function.
- The old `C=L` gate differs from the fully causal gate.
- The `mqar2` diagnostics distinguish first facts, first query, second facts, second query.

## C64 matched-seed synthetic MQAR benchmark

[First paired-matrix run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43801548839). Same width32, layers2, L64, pairs16, C64, 300 steps, fixed heldout evaluation, CPU, three paired training seeds.

| train seed | no gate loss | exact prior-query gate loss | exact current-key gate loss | wrong-row (content shuffled) loss |
|:--:|--:|--:|--:|--:|
| 0 | 0.650108 | 0.202899 | **0.190913** | 0.394608 |
| 1 | 0.615676 | 0.267844 | **0.230524** | 0.455641 |
| 2 | 0.591561 | **0.221982** | 0.230794 | not run |
| mean (3 seeds) | 0.619115 | 0.230908 | 0.217410 | — |

**Decision**: State-aware gating is promising on the synthetic toy, but **the specific previous-query gate has not outperformed the exact same-cost current-key gate**. Three seeds are insufficient for a confident difference test. Wrong-row controls also improve baseline, so neither generic suppression nor other confounds are excluded.

Exact prior-query C64 CPU runtime was ~30 s vs ~8 s for ungated baseline in the same run (approx. 3.7x slower), reflecting an unoptimized Python loop.

## Single-seed extensions (not replicated yet)

- P4: ungated loss 0.115988 vs prior-query gate 0.048561.
- Interleaved `mqar2`: ungated loss 0.550820 vs prior-query gate 0.143961.
- C64 detach-only: 0.261195; margin-only: 0.177473; detach+margin: 0.181017; no-detach/no-margin: 0.202899.

A parallel [stage-two Actions matrix](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43802940476) is testing `mqar2` with full phase diagnostics and current-key/shuffled-row controls, independent P4 / two-hop seeds, and independent detach/margin ablations.

### Initial margin / detach ablation (later cross-run conflicts; g C64)
The margin suite from the stage-two run has passed; matched base/prev values came from stage one.

| seed | exact prev, no D/M | +M=.2 | +D only | +D +M=.2 |
|:--:|--:|--:|--:|--:|
| 0 | 0.202899 | **0.177473** | 0.261195 | 0.181017 |
| 1 | 0.267844 | **0.184112** | 0.260943 | 0.191183 |
| 2 | 0.221982 | **0.163650** | 0.296092 | 0.212070 |

Within this initial experimental batch, adding detach to margin was worse than margin-only in all three seeds, and detach-only was also weak. However, the no-margin comparator rows came from an earlier Actions job; subsequent same-seed cross-job margin results conflicted substantially. **Retract any cross-job claim that margin robustly beats no-margin.** Use the later one-host paired suite for this inference. Do not set detach as a default or assert a proven gradient-conflict mechanism.

A 1000-step ungated `g` baseline is also being trained against the 300-step, ~3.7x slower Python reference gate to check the equal-CPU-time comparison ([budget audit](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803567236)).

## P4 and two-hop falsification (stage two complete)

[Follow-up Actions run 43802940476](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43802940476), `p4_hop2` job completed.

| P4 seed | base loss | exact previous-query loss | exact current-key loss |
|--:|--:|--:|--:|
| 0 | 0.115988 | 0.048561 | not run |
| 1 | 0.098012 | 0.056408 | **0.033245** |
| 2 | **0.129533** | 0.143398 | **0.023512** |

P4 seed 2 is a **counterexample** to a universal improvement claim for the previous-query read; it is worse than ungated while current-key is much better. This further defeats an evidence-free claim of previous-query superiority.

`hop2`, 450 steps, seeds 0/1: base accuracy 0.0555/0.0145, previous-query 0.0592/0.0583, current-key 0.0508/0.0604. No arm has learned reliable two-hop retrieval, so no claimed two-hop breakthrough.

## CPU wall-time control and reproducibility warning

[Budget run: 1500-step ungated g](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803827494), C64/P16, 3 seeds. Training time 28.6–31.1s vs exact sequential gate 300-step time ~29–30s **in the original matrix**.

| seed | ungated g 300 steps | ungated g 1500 steps | exact prev g 300 steps |
|---:|---:|---:|---:|
| 0 | 0.650108 | 0.233753 | 0.202899 |
| 1 | 0.615676 | **0.210861** | 0.267844 |
| 2 | 0.591561 | 0.272681 | **0.221982** |
| mean | 0.619115 | 0.239098 | 0.230908 |

The huge equal-step advantage **almost disappears with equal CPU wall-time**. This is not a FLOP/GPU comparison and has changing learning-rate schedule; a serial Python gate must not be presented as an efficient hardware-operator innovation.

**Critical reproducibility conflict:** A separate [same-margin current-key comparison](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803459265) reproduced seed-0 prev-margin loss 0.177473 exactly but obtained seed-1 **0.163071**, seed-2 **0.292281**. The prior margin suite found seed-1 **0.184112**, seed-2 **0.163650**, despite identical source SHA for `mix.py`, `gate_exact.py`, and `rc.py`, and apparently identical flags and train seeds. Accordingly, **retract the claim that margin improves in all three seeds across runs** until deterministic repeatability is demonstrated. A [within-run replay with 1/2 threads](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804122341) has been launched to isolate this discrepancy. Any conclusions about training-stability fixes are provisional.

The same-margin current-key control losses were 0.168743, 0.191712, 0.178424 for seeds 0/1/2, respectively; the previous-query margin losses in that same Action were 0.177473, 0.163071, 0.292281. **Previous-query read does not show consistent dominance even with matched margin.**

## High-LR DeltaNet counterexample (Gemini experiment 2, two seeds)

[Stability Actions run 43803782282](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803782282): `--mix d --lr .05`, L64, P16, C16, 300 steps, two paired seeds:

| method | seed0 loss | seed1 loss |
|---|---:|---:|
| no gate | 2.200434 | 0.722578 |
| legacy chunkstart gate | **0.313543** | **0.152013** |
| self-consistent exact gate | 0.364423 | 0.333683 |
| exact + margin .2 | 0.489748 | 0.336466 |
| exact + detach + margin | 1.064501 | 0.352176 |
| exact + margin + scale erase (`lpsE=1`) | 0.382344 | 0.308481 |

**Gemini's proposed D+M high-LR stability fix fails this test.** Legacy approximate gate outperformed the exact gate in both seeds; precision in recurrence is not synonymous with optimization advantage. For `d`, default `lpsE=0` scales writes but not the erasure factor, so no full-update-skip claim is possible without the `lpsE` control. Reproducibility caveat from margin repeats still applies.

## Interleaved MQAR2, two stronger controls, and actual new-fact retention

[Completed follow-up run 43802940476](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43802940476). Interleaved task with first fact keys, query on those, **distinct** new fact keys, then query both (L128, P16, C16, width32, 2 layers, 450 steps).

| seed | no gate | exact previous-query | exact current-key |
|---:|---:|---:|---:|
| 0 | 0.550820 (earlier run) | **0.123220** | 0.196003 |
| 1 | 0.515202 | **0.113971** | 0.118259 |
| 2 | 0.557471 | 0.130330 | **0.089050** |

The content-shuffled seed0 control loss was 0.511068: barely improved over baseline; true state-aware gate significantly better on this setting. However previous-query does not win reliably over current-key (2/3 seeds here) and this task has no overwritten keys.

For exact previous-query gate, layer-0 mean factor across seeds:
- first fact values `factA_v`: 0.835–0.843
- second fact values `factB_v`: 0.821–0.829
- first query answer values `queryA_v`: 0.070–0.073
- second query answer values `queryAB_v`: 0.103–0.107

This establishes a **measured selective gating pattern** on this dataset, *not* a causal demonstration or a generic guarantee that new writes are protected. The old assertion that factB factors return to 1.00 is false: they are ~0.82. Also one same-seed mqar2 replicate varied in loss (prior seed0 exact 0.143961, now 0.123220), underscoring the need to diagnose cross-run reproducibility.

## Deterministic replay check (same code, independent runners)

[Replay run #43804122341](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804122341): same C64 margin configuration repeated twice for seeds 1/2 at 1 and 2 CPU threads, i.e. 8 replays, with *bitwise-identical* final losses, curves and learned lpg values within that job: seed1 **0.163071**, seed2 **0.292281**.

[Cross-run matrix #43804770888](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804770888): three independent ubuntu-26.04 runners, with **pinned torch 2.15.0+cpu, numpy 2.4.6, CPU capability AVX512**, all returned identical seed1 **0.163071** and seed2 **0.292281** with the same curves and learned lpg.

This does *not* explain why prior [margin suite #43802940476](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43802940476) produced **0.184112** and **0.163650** for these seeds despite identical source SHA. There may be hardware kernel/environment differences not captured in the older logs; that explanation remains **unverified**. Importantly, the earlier result that margin improves in all three seeds is **withdrawn**. Do not pool different runner results as if random seed were the only changing factor.

[Single-run fully paired control #43804762931](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804762931) executes no-margin, margin, current-key margin, detach+margin in one job with full effective argument snapshots and code hashes; use that as the next primary fair comparison.

## Final matched within-host ablation (primary result for Gemini knobs)

[Successful single-host CI #43804762931](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804762931): 3 training seeds × 4 arms, CPU one thread, same exact source binaries and fully logged parsed flags; L64/P16/C64, width32, 2 layers, 300 steps. This run must take priority over contradictory unpaired cross-job results.

| seed | prior-query no margin | prior-query margin .2 | current-key margin .2 | prior-query margin+detach |
|---:|---:|---:|---:|---:|
| 0 | 0.202899 | 0.177473 | **0.168743** | 0.181017 |
| 1 | 0.267844 | **0.163071** | 0.191712 | 0.173060 |
| 2 | 0.221982 | 0.292281 | 0.178424 | **0.163311** |
| mean | 0.230908 | 0.210942 | 0.179626 | **0.172463** |

**Verdict:**
- Margin alone helps 2/3 seeds but *hurts seed2* (0.22198→0.29228), so it is **not a universal stability fix**. Earlier first-run 3/3 benefit is not reproducible.
- Current-key gate plus the *same* margin outperforms prior-query plus margin in 2/3 seeds, with lower 3-seed mean. The proposed previous-query novelty is unsupported.
- Detach+margin has the lowest 3-seed mean, but per-seed ranking is inconsistent; this is a *post hoc* metric over many tried configurations, not a validated generalization finding. Detach-only previously hurt all 3 seeds, and DeltaNet experiments rejected the asserted high-LR repair.
- **No performance-based justification for merging any Gemini knobs into the main method**. Keep sequential exact mode as diagnostic reference and do not announce efficient chunked computation.

## Go/no-go requirements before a paper claim

1. **Prior query must beat strong current-key residual/novelty baselines** on a pre-specified falsification test, or abandon this as the main new mechanism.
2. **Actual online-update stress** (same-key corrections, query-after-correction, time-varying distractors) must protect new writes without allowing echo interference. Current `mqar2` is only an interleaved insertion test.
3. **Reproducible independent seeds and fixed checkpoint interventions** to separate causal influence from retrained optimizer effects; PR #1 handles oracle interventions.
4. Compare to *correct implementations* of published baselines; no misleading `--lpsP` shortcut.
5. For scaling claims, first produce a true efficient chunked/fused implementation with consistent gradient semantics and measure GPU throughput, wall-clock, memory as well as model quality. A serial correctness oracle is not evidence of negligible overhead.
6. `lm.py` requires external FineWeb binary slices not present in this repository, so Gemini's `lm128` proposal is not yet reproduced.

**Research recommendation**: If the current-key gate remains tied/better, treat the previous-query variant as an analysis ablation rather than the paper's central method. Keep the PR draft and unmerged; coordinate any rebase conflicts with the separate `--seed` addition in PR #1.
