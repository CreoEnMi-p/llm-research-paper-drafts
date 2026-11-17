> **⚠ Superseded evaluation interpretation (2026-11-17):** The original eight-question task has a teacher-forced answer-elimination chance curve; its position-averaged reference is H8/8 = 33.973%. This report's 34% RoPE/NoPE results should *not* be interpreted as partial multihop skill. Corrected per-question scores, first-only AdamW 5 seeds, L+2, long chains, Sanford terminal k7 and semantic reassignment are in [REVIEWER_SIX_POINTS_2026-11-17.md](REVIEWER_SIX_POINTS_2026-11-17.md). Retain the old measurements as historical data only.

# Paper 05 — Fairness and external validity re-evaluation
**Date:** 2026-11-17 UTC · **Status:** Independent audit, **not publication-ready**  
**Base:** original paper `PAPER.md`, isolated branch `research/05-pointer-doubling-audit`, Draft PR #10.

## Executive decision
**The central reverse-ordered toy result is genuine, large and robust to longer training, NoPE/RoPE position controls and three seeds. The headline extrapolation to generic multi-hop reasoning / an improved real-language attention head is not supported.** On the *actual* Sanford k-hop induction data generator, regular 4-layer RoPE learns the task extremely well at 2,000 steps, and an 8-iteration parameter-shared NoPE competitor is also very strong. On held-out Python source-code next-byte modeling, the full doubling head does **not** outperform a simpler pair-match identity head. The biggest remaining blocker is a genuine real-world *semantic* variable/dataflow-tracing benchmark with measured benefit over a strong baseline.

We distinguish (a) *same-task learning efficiency*, (b) *external algorithmic benchmark*, and (c) *real-code language modeling*. Their accuracies/losses cannot be pooled into one effect size.

## 1. Long-trained and positional fair controls on author's exact reverse-order task
Protocol: original `pdg.py` patched only for random seeds, NoPE and shared-layer controls, with every-50-step full validation curves. 8 pairs, one fixed held-out eval set of 256 sequences (2,048 related answers), batch 16, 1,200 steps, no early stopping, width64, depth 3 for k7 and depth4 for k15, 4 heads. All arms omit ALiBi and attention sink; full BLN retains its head-specific algorithmic bias; ordinary RoPE/NoPE models use standard causal QK-norm etc. BLN lr=.02, plain lr=.04, both as in authors' grids. Three different training torch seeds and NumPy train streams. The common eval seed is deliberately fixed for paired comparisons.

| Task | Arm | Seed0 | Seed1 | Seed2 | Mean ± sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| k7/L3 | standard RoPE | .340332 | .339355 | .346191 | .3420 ± .0037 |
| k7/L3 | standard NoPE | .338379 | .343750 | .334473 | .3389 ± .0047 |
| k7/L3 | full BLN | 1.000000 | 1.000000 | 1.000000 | 1.0000 ± .0000 |
| k15/L4 | standard RoPE | .340332 | .341797 | .349121 | .3438 ± .0047 |
| k15/L4 | standard NoPE | .328613 | .331543 | .334473 | .3315 ± .0029 |
| k15/L4 | full BLN | 1.000000 | 1.000000 | .999512 | .9998 ± .0003 |

7/3, across 3 seeds: BLN−RoPE mean +.6580; an illustrative paired Student-t95 interval [+.6489,+.6672]. With only three training seeds, this is **descriptive**, not definitive calibrated uncertainty. Correlated within-row answers additionally prevent interpreting 2,048 as independent examples. *Do not use legacy variation across different configuration tables as a seed SD estimate.*

Selected 7/3 curves (mean over 3 seeds; steps 50,100,200,400,600,800,1000,1200):
- RoPE: .1258, .1711, .2892, .3371, .3457, .3320, .3361, .3420
- NoPE: .1222, .1216, .1434, .2980, .3328, .3306, .3328, .3389
- BLN: .9591, .9771, .9901, .9912, .9982, .9989, .9992, 1.0000

The entire learning curves, **not just cherry-picked endpoint values**, are committed in [part 1](fair_1200_part1.json), [part 2](fair_1200_part2.json), [part 3](fair_1200_part3.json). [Full run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44040112264).

### Even longer controls, including roughly runtime-equivalent training
On k15/L4, plain models received 4,000 steps; BLN was already ~100% at 1,200. Final results:
- RoPE s0 **.337402**, s1 **.341797**, maximum along each logged curve .35645/.35938.
- NoPE s0 **.342773**, s1 **.353516**, maximum .35742/actual seed1 logged maximum in [long-run results](longer_controls_part2.json).
All stayed close to ~.34–.36; no delayed breakthrough was found *within 4,000 steps*. That does **not** show an asymptotic barrier for trained vanilla attention. CPU time varied widely due to hosted workers, so runtime equivalence is approximate. Source [4000-step follow-up](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44041283779), archived in [part 1](longer_controls_part1.json) and [part 2](longer_controls_part2.json).

### Looped / Universal-style controls
A one-block **weight-shared** NoPE transformer with **8 unrolls** was built. At 1,200 steps:
- k7: **.3457**, k15: **.3296** (one seed).
- Looped *one-hop pointer* model is highly unstable rather than simply incapable: k7 lr=.02 reached **.9448 at 200 steps**, then **.0859 at step1,200**; lr=.005 peak **.8906** then .1382; lr=.01 peak **.9819** then .1475. Proper checkpoint selection and optimizer tuning are necessary before claiming superiority over a looped pointer. Not equivalent compute or depth to L=3/4 BLN; do not label it an identical-budget experiment.
- This is a **weight-shared looped adaptation** of our own model, not an official reproduced Universal Transformer implementation; the exact architecture comparison remains incomplete.

## 2. CPU cost / approximate compute budget
Same-runner, 2 CPU threads, 15-hop inputs, batch16, 10 forward+backward timed samples after warmup, median:
| Arm | ms/step | Relative | Trainable parameters |
| --- | ---: | ---: | ---: |
| RoPE, 4 layers | 88.86 | 1.00× | 228,032 |
| NoPE, 4 layers | 91.03 | 1.02× | 228,032 |
| BLN, 4 layers | 189.44 | **2.13×** | 228,040 |
| Looped NoPE, 8 unrolls | 160.79 | 1.81× | 81,632 |

`torch.profiler` partial matmul/conv training FLOPs: 5.24G RoPE vs **14.20G** BLN, ratio **2.71×**. This is *not exact total FLOPs* and is not a fused-kernel/GPU throughput evaluation. Thus **equal parameter counts are not equal compute**. [Profiler run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44042328854), [raw summary](cost_profile_verified.json).

## 3. Original Sanford hop_k task: a genuine external benchmark, unlike reverse-block chain
The source is pinned [official author code](https://github.com/chsanford/hop-induction-heads), commit `5735bca9545545774098994e21392b51b285bcc5`, `InductionHopsFullSequenceTask` from `src/tasks.py`. This task asks for k hops after repeated preceding character matches in a random sequence and supervises all positions. It is **not** our disjoint-vocabulary reverse-block lookup. Trained our *same-model code* 4-layer RoPE, NoPE, BLN for 2,000 steps, 2 seeds each, and one 8-unroll weight-shared NoPE arm. Real author published model uses a distinct GPT2 architecture, 100,001-step training and much larger width; therefore this is **an exact data-generator transfer, NOT a paper-author training reproduction**.

Metric is mean over k=0…7 of accuracy on targets that are actual characters (exclude the no-ancestor `~` and special blank `_`); report by k to avoid inflating long-hop performance with nulls.

| Arm | Seed0 | Seed1 |
| --- | ---: | ---: |
| RoPE, 4 layers | **.984** | **.999** |
| NoPE, 4 layers | .841 | .901 |
| BLN, 4 layers | .975 | .993 |
| Looped NoPE, 8 unrolls | **.999** | not run |

Hardest k=7 non-null accuracy: RoPE s0 .8764, s1 .9968; BLN s0 .8661, s1 .9585; looped NoPE s0 .9916. **All prove that ordinary Transformer can learn this multihop benchmark; BLN is not uniquely required**. Looped has greater effective depth, so do not interpret as an iso-compute win. [Run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44040397669), [full per-k metrics/learning curves](sanford_2000_results.json). See [Sanford et al., ICML 2024](https://proceedings.mlr.press/v235/sanford24a.html).

## 4. Real open-source Python code: a narrow external validity test, not semantic tracing
Train byte LM on pinned CPython standard library (`Lib/**/*.py`: 2,006 files, 41,507,346 bytes) and validate held-out `psf/requests` Python files (19 files, 216,142 bytes). No overlap in source projects; source SHAs recorded in the code and logs. Tokenization is **actual UTF-8 bytes, V256**, unlike paper's initial tiny V2000 FineWeb text cell. Same `pdlm.py`, 1,500 steps, batch8 × 256 bytes = 3,072,000 train bytes per arm, context256, 400 val windows. Metric held-out **cross entropy in nats/byte**, lower better. This is tiny pretraining, not a general LM scale-up.

| Arm | Seed0 | Seed1 |
| --- | ---: | ---: |
| RoPE + ALiBi | 1.48063 | 1.47234 |
| Pair-match + ALiBi | **1.26751** | **1.27362** |
| BLN no ALiBi | 1.47035 | 1.45307 |

Additional seed0 position controls (all **no ALiBi**): plain NoPE **1.83899**, plain RoPE **1.50041**, pair-match Identity **1.30771**, BLN + pair-match **1.33077**. BLN has **no win versus the stronger, simpler Identity match**, nor when combined with it. [2-seed run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44040620606), [position control run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44041379971), [raw curves & provenance](real_code_lm_results.json), [control curves](real_code_lm_controls.json).

**Critical limitation:** Predicting the next code byte does **NOT** test following an actual chain of variable assignments. The requested **real semantic multihop benchmark remains unsatisfied**. RULER variable_tracking uses explicit natural-language-ish synthetic assignments and asks for *all variables sharing a value*, so merely calling that dataset “real code” would also be misleading. Need real program variable/use-def provenance or tested program state, then end-to-end answer accuracy.

## 5. Impact on paper statements
1. **Retain a carefully limited result:** an explicitly algorithmic and strongly initialized pointer head learns reverse-ordered chain facts much faster under exactly specified model, compute and data constraints.
2. **Remove or rewrite:** “competitors cannot solve multihop” broadly. Sanford original task shows ordinary RoPE and looped models solve it rapidly. Existing known parallel pointer jumping and exact tied-query retrieval are also important prior art.
3. **Do not claim generality / practical superiority:** current head is order-dependent, adds ~2.13× actual CPU fwd/back step time, has no held-out code LM win against pair-match, and has not passed an actual semantic variable tracking task.
4. **Looped controls remain incomplete:** one learned pointer loop is optimization unstable. Report its peak and final score, and compare checkpointed/long-trained with matched FLOPs; do not dismiss on final-only score.
5. **Confidence:** 3 seeds for two synthetic cells is a meaningful improvement over 1, but for paper-level estimates use ≥5 seeds and report intervals on independent *rows*, not correlated query tokens. Two seeds Sanford and code are preliminary only.
6. **Next decision gate:** run actual semantic variable tracking on real open-source code, with repeated variable names/shadowing/reassignments, prompt order and length shifts; compare L-layer BLN to compute-matched RoPE/NoPE and a tuned Looped architecture. If no meaningful win, **do not pursue “general purpose attention head” ICML claim**; consider a niche structured-memory mechanism paper.

## Reproducibility index
- Generated baseline instrumenter: [build_fair.py](build_fair.py)
- Sanford official source task harness: [sanford_official_task.py](sanford_official_task.py)
- Real corpus preparer: [prepare_code_corpus.py](prepare_code_corpus.py)
- Cost profiler: [measure_cost.py](measure_cost.py)
- GitHub workflows in `.github/workflows/paper05-{fair-baselines,longer-controls,sanford,code-real,code-controls,profile}.yml`
- Full primary evidence and curves: `fair_1200_part*.json`, `longer_controls_part*.json`, `sanford_2000_results.json`, `real_code_lm_results.json`, `real_code_lm_controls.json`, `cost_profile_verified.json`.

**Audit confidence:** specific numbers traceable to completed GitHub Actions runs and committed JSON snapshots. Model selection / hyperparameter tuning remains incomplete; no generality beyond tested distributions is implied.
