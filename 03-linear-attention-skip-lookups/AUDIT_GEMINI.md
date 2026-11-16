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

### Repeated margin / detach ablation (complete, g C64)
The margin suite from the stage-two run has passed; matched base/prev values came from stage one.

| seed | exact prev, no D/M | +M=.2 | +D only | +D +M=.2 |
|:--:|--:|--:|--:|--:|
| 0 | 0.202899 | **0.177473** | 0.261195 | 0.181017 |
| 1 | 0.267844 | **0.184112** | 0.260943 | 0.191183 |
| 2 | 0.221982 | **0.163650** | 0.296092 | 0.212070 |

On these three paired seeds: margin helps every time; detach alone loses every time; adding detach to margin loses to margin-only every time. **Do not** set detach as a default, or claim the explanation about early-gradient conflict is established. A margin must also be applied to the `current-key` competitor before it can be credited as a novel mechanism; [same-margin control launched](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803459265).

A 1000-step ungated `g` baseline is also being trained against the 300-step, ~3.7x slower Python reference gate to check the equal-CPU-time comparison ([budget audit](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43803567236)).

## Go/no-go requirements before a paper claim

1. **Prior query must beat strong current-key residual/novelty baselines** on a pre-specified falsification test, or abandon this as the main new mechanism.
2. **Actual online-update stress** (same-key corrections, query-after-correction, time-varying distractors) must protect new writes without allowing echo interference. Current `mqar2` is only an interleaved insertion test.
3. **Reproducible independent seeds and fixed checkpoint interventions** to separate causal influence from retrained optimizer effects; PR #1 handles oracle interventions.
4. Compare to *correct implementations* of published baselines; no misleading `--lpsP` shortcut.
5. For scaling claims, first produce a true efficient chunked/fused implementation with consistent gradient semantics and measure GPU throughput, wall-clock, memory as well as model quality. A serial correctness oracle is not evidence of negligible overhead.
6. `lm.py` requires external FineWeb binary slices not present in this repository, so Gemini's `lm128` proposal is not yet reproduced.

**Research recommendation**: If the current-key gate remains tied/better, treat the previous-query variant as an analysis ablation rather than the paper's central method. Keep the PR draft and unmerged; coordinate any rebase conflicts with the separate `--seed` addition in PR #1.
