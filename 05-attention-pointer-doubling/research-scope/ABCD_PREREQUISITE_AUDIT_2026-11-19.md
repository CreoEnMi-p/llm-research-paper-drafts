# Paper 5 — prerequisite learning, counterfactual supervision and scope-transfer audit
**Independent continuation · 2026-11-19 · branch:** `research/05-ab-cd-prerequisite-20261119`

## Decision
**Continue as an optimization/inductive-bias investigation, not yet an established Binding+Scope model or ICML-ready result.** 
The crucial missing prerequisite was hidden in the original frozen V2 report: both original models almost completely failed **in-distribution A+B counterfactual pairs**. A negative A+C cross-composition result by itself therefore did not identify a special *composition* failure.

A new **5,000-update A+B-only paired-supervision control** now shows the `pointer-depth` model reaches **88.75%** dev pair both-correct over five seeds, whereas the direct 16-class Transformer reaches **22.19%**. The pointer is *capable of learning this simpler prerequisite* with more training. This is **not evidence for lexical-scope generalization, runtime frames, or unseen A+C transfer**. Output-space constraints also differ between these models; see fair-control gate below.

## Ground truth and existing heldout result (do not reuse for selection)
- Grammar mechanisms: `A` assignment snapshot/alias; `B` later reassignment; `C` lexical shadow/separate scopes; `D` closure/runtime invocation environment.
- Training combinations: `AB + CD`; originally frozen `AC` test: **256 strict counterfactual pairs**, four production clusters, five random seeds.
- Frozen V2 SHA256: `e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce`.
- Original `AC` individual accuracy / both-correct: 2-layer Transformer **35.898% / 0.156%**; source-literal `pointer-depth` **37.891% / 0.078%**. GitHub exact-code replay agreed in mechanism-level failure, though numeric values varied across CPU environment/seed.
- All V2 programs contain exactly four distinct candidate literals. Conditional random-source-literal guessing is **25% per program**. Independent random pair success is **6.25%**; if both distinct labels are known and sample without replacement, **8.33%**. These aren't substitutes for a trained structure-aware baseline.
- V2 is consumed. The new scripts **never read or evaluate any frozen A+C sample**. New method claims require a genuinely new preregistered heldout grammar/distribution, not simply reusing V2 outcomes.

## New experiment 1 — counterfactual paired supervision on AB/CD development only
**[Actions run 44510753852](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44510753852)** · 20/20 training jobs and aggregation passed · both models, five seeds, 1,000 steps, batch 32, d64/2 layers, AdamW 0.001.

Same program budget per optimizer step: `single` uses 32 independently generated programs; `paired` uses 16 independently generated source skeletons, each with two distinct final print queries. Same fixed 64 AB and 64 CD dev pairs per seed; **paired both-correct**, not single accuracy.

| Model | AB mixed-single | AB mixed-paired | CD mixed-single | CD mixed-paired |
|---|---:|---:|---:|---:|
| Transformer 16-class | 0.3125% | 0.6250% | 54.6875% | 55.3125% |
| Source-literal pointer + depth | 1.2500% | 4.3750% | 54.6875% | 65.0000% |

**Interpretation:** pairing the two query variants improves the pointer but does not teach reliable A+B within 1k steps. CD improvement is substantially driven by a single random seed. CD 54.6875% being nearly seed-invariant is suspicious and requires per-production analysis: one CD subtype may be easy while the other is unsolved.

## New experiment 2 — remove multi-mechanism interference
**[Actions run 44511446836](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44511446836)** · 20/20 training and aggregation passed · **train AB only** for 1,000 steps, five seeds, same model/optimizer/sample counts, evaluate fixed AB dev pairs.

| Model | AB-only single | AB-only paired |
|---|---:|---:|
| Transformer | 0.3125% | 0.6250% |
| Pointer-depth | 2.1875% | 10.3125% |

**Interpretation:** multitask interference is present, but eliminating CD alone is not enough. This 10.31% average would be grossly inadequate evidence of learning a compositional algorithm.

## New experiment 3 — 5,000 steps of AB-only paired supervision
**[Actions run 44512019617](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44512019617)** · 10/10 training and aggregation passed · same small models, d64/2 layers, batch32, AdamW .001, five seeds, no AC access.

| AB dev pair **both-correct**, 5-seed mean | Step 1000 | Step 2500 | Step 5000 |
|---|---:|---:|---:|
| Direct Transformer classifier | 0.3125% | 1.2500% | 22.1875% |
| Source-literal pointer + depth | 11.5625% | 71.2500% | **88.7500%** |

At step 5000, pointer individual accuracy **94.0625%** and prediction-flip **91.875%**; direct Transformer individual accuracy **58.90625%**, prediction-flip **28.125%**. Pointer 5-seed both-correct outcomes: **[89.0625%, 100%, 100%, 100%, 54.6875%]**. Thus the result is strong on most seeds but still seed-sensitive.

**Interpretation:** the old 1k apparent A+B failure is an optimization/data/training-budget result, not proof that pointer-based models lack the representational capability. More importantly, the pointer's literal-location readout may be providing *candidate-space restriction*, not learning scope identity. All current successful AB samples follow one grammar production with randomized names, literals, noise positions and alias lengths.

## Remaining high-priority fair-control gate
The pointer model is constrained to select one of **four source literals**, while the original Transformer predicts unconstrained among **16 output values**. Moreover only the pointer includes syntax-derived depth embeddings. Their 5k comparison therefore does **not isolate pointer-location mechanism**.

A new control uses the **same 2-layer encoder, syntax-derived depth embedding, and a four-source-literal candidate mask** with a conventional 16-way classifier. It has no gold AST binding links, scope IDs, or execution trace. Its outputs are restricted to exactly the same candidates as the pointer. Additionally, run `pointer-depth` and the matched classifier on **mixed AB+CD 5k paired training**, to test whether both known prerequisites can be learned concurrently.

Status when this report first written: [Actions run 44512873352](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44512873352) submitted; update this section with verified aggregate only.

## Scientific inference and next gates
1. **Separate output-head advantage from binding algorithm.** If source-literal-masked/depth-matched Transformer catches up, the claimed pointer-specific effect weakens. If pointer remains substantially better across seeds at matched training budgets, inspect mechanisms/attention and alias-depth out-of-distribution behavior, not just headline accuracy.
2. **Prove both training ingredients independently and jointly**, using pair-both-correct, prediction flips, per-production accuracy, learning curves, alpha-renaming, and lexically distinct sibling scopes. Pair prediction invariance is a **failure** when correct labels differ.
3. **Only then freeze fresh V3 composition evaluation.** New production families with stronger source-only heuristics and no reused `AC` family as sole fresh OOD basis; challenge scope siblings, `global/nonlocal`, late closure reads and distinct dynamic frames, not just alias snapshots. Predeclare metrics, models, five+ seeds, CPU/version provenance, and test SHA before first new test read.
4. **Novelty boundary:** AB source-position selection is far weaker than learned `(P,B,S,E)` and not novel lexical binding. A restricted symbolic AST interpreter already obtains perfect correctness on V2; it is the scripted/oracle ceiling, not a learned model. Published assignment/selection, symbol-invariance and graph/AST-guided methods must be compared.

## Implementation and audit trace
- Base branch `research/05-position-binding-scope`, base commit `183e7a0e30f0c1aadf7bebdd634de969cb088416`.
- Work branch: `research/05-ab-cd-prerequisite-20261119`; **main unchanged**, no merge.
- New dev-only scripts: `repro/train_dev_abcd_prerequisite.py`, `repro/train_dev_ab_only_probe.py`, `repro/train_dev_ab_only_long.py`, `repro/train_dev_candidate_matched.py`; aggregated metrics and GitHub workflows committed alongside scripts.
- All models use source tokens and permitted syntax depth, **no provided lexical binding IDs, gold def-use graph, dynamic execution trace or test labels at training time**.
- This is synthetic restricted Python, four candidate numerals per program and small model/cluster count, not an assertion about real-code model generalization or a publishable architectural discovery.
