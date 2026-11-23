# Paper 5 — V3 independent audit, decision and research boundary
**Date:** 2026-11-23. **Decision: NO-GO for a top-conference mechanism claim with the present position-only architecture.** All findings below are from verified GitHub Actions and archived raw predictions, not reconstructed estimates.

## 1. Preregistration, frozen test and methodological integrity
- Preregistration was committed before the first V3 neural evaluation: [PROTOCOL_LOCK.json](PROTOCOL_LOCK.json). Four model arms, five training seeds, fixed 5,000 AdamW steps, batch 32, 2 layers/d64, source-only inputs, fixed training/development seeds, final checkpoint evaluation, primary metric and inference count are locked.
- Independent V3 generation: **448 strict counterfactual pairs, 896 programs**, 7 new combination families × 2 variable-name strata (main bound identifier either among the twelve training names or four held-out names). Each program has exactly four distinct numeric source literals; paired programs have identical bodies but different final queries and different CPython outputs.
- Frozen SHA256: **514f0adf76994c2b8d17e8c985847619519467cba63a6226a97b26618a2e4aa2**. See [FROZEN_MANIFEST.json](FROZEN_MANIFEST.json) and [frozen dataset](FROZEN_V3_448pairs.jsonl).
- [Freeze run 45176133603](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45176133603) succeeded **before** [neural run 45177337953](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45177337953). The latter completed **20/20 train jobs and aggregation successfully**. Model/training grammar Git blob identities and data hash are checked before prediction. No V3 labels or examples were read for model selection or training. Pre-frozen V2 is consumed and never used as new confirmatory evidence.
- Dev-only smoke, full independent AST audit, alias OOD dev diagnostics, five-seed neural outputs and exact archive all succeeded: [freeze](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45176133603), [AST/heuristics](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45176589372), [V3 models](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45177337953), [alias dev](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45177413471), [archive](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/45178779855).
- Raw **20 locked V3 model prediction JSON files**, raw alias development records, and recomputed summaries are permanently committed under this folder. V3 is now **consumed** for any future architecture choice.

## 2. The results: five seeds, strict causal-pair metric

| Locked arm | Trainable parameters | Individual accuracy | **Pair both-correct** | Pair prediction flip | Correct label-address mass* |
|---|---:|---:|---:|---:|---:|
| Source-literal pointer + depth | **86,273** | 25.87% | **4.20%** | 18.39% | 26.37% |
| **Pointer with scrambled occurrence/value map** | **86,273** | 24.96% | **0.04%** | 0.45% | 25.00% |
| First-occurrence alpha-canonicalization + pointer | **86,273** | 29.87% | **10.54%** | 27.37% | 30.97% |
| Candidate-mask + depth Transformer classifier | 78,992 | 27.72% | **7.01%** | 17.37% | n/a |
| Non-oracle script heuristic preselected on **AB/CD dev** | — | 32.48% | **14.29%** | — | — |
| Independent **restricted symbolic AST interpreter** | — | 100% | **100%** | 100% | — |

*For the scrambled arm, mass is measured using its deliberately permuted candidate mapping, not the original gold source location. For the other pointer arms the mapping is the true source-literal position. The four candidate source numerals imply single-query uniform 25%, independent-uniform paired 6.25%, or distinct-label-aware uniform ordered paired 8.33%. These are reference rates, **not universal bounds**. A systematic policy that answers both members identically can score approximately 50% single-query and **0% pair-correct**. Requiring the pair to be simultaneously correct is essential here.

Seven-family × five-seed cluster bootstrap percentile intervals for pair both-correct: pointer **[0,17.32]%**, canonical pointer **[0,37.01]%**, scrambled **[0,0.27]%**, candidate classifier **[0,21.96]%**. These are wide: only seven independent family types. The paired pointer-minus-scrambled difference is **+4.15pp**, bootstrap **[0,+17.14]pp**, with lower bound at zero. No robust evidence of transferable binding. See [archived aggregate](V3_FIVE_SEED_AGGREGATE.json).

### What actually fails, by new combination
| Held-out program family | Pointer pair both | Canonical pointer pair both | Candidate-classifier pair both |
|---|---:|---:|---:|
| Same-depth sibling scopes | 1.56% | 0.31% | 0.31% |
| **Global/nonlocal interaction** | **0%** | **0%** | **0%** |
| Late closure read + assignment snapshot | 25.94% | **73.13%** | 48.75% |
| **Different closure frames after rebinding** | **0.31%** | **0%** | **0%** |
| Nonlocal mutation vs outer snapshot | 1.56% | 0% | 0% |
| Nested local shadowing | 0% | 0% | 0% |
| Global write + late closure snapshot | 0% | 0.31% | 0% |

**Crucial benchmark limitation:** The AB/CD-development-selected simple script policy gets **100%** on the late-closure+snapshot family via a literal-position shortcut. It gets zero paired correctness on the other six families. Therefore the canonical pointer's 10.54% aggregate and much of its 73% performance on that family are **not evidence of learning closure semantics**. Even the frozen V3 grammar is finite and contains one easily exploitable family; future publication needs a harder independent V4, not just a new V3 score.

## 3. Mechanism, parameters/compute alignment, and OOD alias depth
The pointer and pointer-scrambled models have **identical 86,273 trainable parameters**, same encoder/head architecture, matrix dimensions, softmax, scatter-add operations, same per-step examples and length schedules, 5,000 updates and batch 32. Scrambled mapping is a deterministic, source-body-dependent permutation of four candidate values. It destroys the consistent location-to-literal correspondence; it does **not** remove parameters or reduce neural FLOPs. Python preprocessing and system runtimes can differ; a separate hardware FLOP trace has **not** been completed. The classifier has fewer parameters and therefore is not the strictly compute-matched control.

Development-only [alias-depth probe](ALIASDEV_5SEED_SUMMARY.json): each cell averages **five seeds × 96 strictly paired programs**, never reads V3.

| Source-only model | Depth 3 / seen names | Depth 3 / unseen main name | Depth 4 / seen | Depth 5 / seen | Depth 6 / seen |
|---|---:|---:|---:|---:|---:|
| Pointer | **90.83%** | 26.46% | 17.71% | 0.42% | **0%** |
| Pointer + alpha canonicalization | **100%** | **100%** | **0%** | 15.21% | **0%** |

The canonicalization control makes identical variable-renaming patterns transfer to unseen *names* while failing even a modest alias-depth extension. That is evidence of a **useful rename-invariant prior**, not a transferable alias algorithm. Address mass on the original V3 pointer is **26.37%**, close to four-way uniform 25%, despite high address mass in AB/CD development. Source-token location retrieval itself is not preserving the right binding under composition.

## 4. External baselines and novelty
[External-baseline boundary report](EXTERNAL_BASELINES_NOVELTY_20261123.md) checks official sources for ICML 2025 variable-binding circuits, ICML 2026 Symbol-Invariant Transformer, ICML 2026 topology-aligned deep dependencies, GraphCodeBERT, GGNN, and Oct 2026 AST/scope/def-use work.

**Not completed:** a faithful trained reproduction of the published symbol-invariant architecture, GraphCodeBERT/GGNN, topology-aligned training, or benchmark-compatible compiler-assisted graph neural baseline. The deterministic alpha-canonicalizer in this experiment is **not** the published invariance model; the independent AST interpreter is a **symbolic rule ceiling**, not a learned graph neural competitor. No top-conference novelty claim can be made while this gate is open.

## 5. Decision and scientific next question
**Top-conference readiness: NO-GO for the present Paper 5 position-pointer/scope-compositionality claim.** All three prerequisites are not met: V3 has an independently verified freeze and correct labels but one shallow-shortcut family; the learned models fail unseen binding/scope/frame composition and alias-depth extrapolation; published neural structure/invariance comparators are not reproduced.

This is a useful falsification, not a reason to spend more compute on the same token-location readout. Next research should ask: **Can a learnable model infer a lexical environment and a distinct dynamic call-frame/write-version state from raw source text, rather than simply choose one of four numeric literals?** The appropriate development mechanism separates P=source/event pointer, B=lexical binding, S=scope ancestry, E=runtime frame and write version. Compiler-provided binding graphs must remain labelled oracle/auxiliary-supervision arms, not hidden test-time inputs.

**Research discipline:** from this point, any method changes suggested by V3 are exploratory; do NOT reuse V3 as a fresh success test. After a new mechanism is developed and audited on disjoint development grammars, preregister independent V4 and compare faithfully implemented external models plus broader code distributions.
