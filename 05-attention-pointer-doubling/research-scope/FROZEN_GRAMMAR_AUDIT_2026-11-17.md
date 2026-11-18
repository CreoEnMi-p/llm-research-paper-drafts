# New grammar composition & corrected evaluation — 2026-11-17

**Status:** independent first-pass experiment, **negative result** for learned P+B+S in current shallow models. Source scripts, frozen V2 raw examples and 5×2 model prediction files are distributed in the accompanying reproducibility ZIP; unlike this report, that ZIP includes the exact experiments. None of the neural-training scripts were committed to this GitHub branch by this report. This is a reproducibility limitation, not a claim of GitHub Actions validation.

## 1. Recompute old chance: 30 categories and 5 hard pairs
The old pilot's universal **6.25%** 16-class uniform baseline was inadequate for a model that can only output a source-code integer literal. Compute for each source: (i) number of distinct numeric literal values U; (ii) number of numeric literal **occurrence** positions O; (iii) `1/U` source-candidate uniform chance and `1/O` occurrence-uniform chance; (iv) first/last literal, last constant assignment, scope-blind most recent assignment to printed name, and source modal literal **without executing the program or using oracle binding links**.

| Old (already repeatedly used) split | 16-way uniform | Source-unique conditional uniform | First literal | Last literal | Modal literal |
|---|---:|---:|---:|---:|---:|
| 12 train grammar templates | 6.25% | **50.00%** | 50.0% | 50.0% | 51.11% |
| 8 old OOD templates | 6.25% | **41.15%** | 37.5% | **62.50%** | 41.99% |
| 10 hard templates | 6.25% | **36.67%** | 30.0% | **50.0%** | 34.14% |

The *single* source-only policy chosen on TRAIN categories was `modal literal` (51.11% train, 41.99% old OOD, 34.14% hard), while inspecting old OOD afterwards shows `last literal` gets 62.5%. This makes the original 56.9% "scope" result uninterpretable as evidence of binding without further controls. Picking the best heuristic per *test* kind is an invalid learned baseline; do not do that.

Original 5-seed macro across old 8 OOD categories and cluster+seed percentile bootstrap (5000 resamples, 8 categories as sampling units):
- Standard Transformer: **22.21%**, interval **[4.14%,46.15%]**.
- Literal pointer + source indentation embedding: **56.91%**, interval **[33.30%,80.29%]**.
The interval is broad due to few template clusters; does not establish generalization beyond template population. Old paired predictions are only available for **seeds1–4**, not seed0 (no invented 5th seed):
- Transformer pair prediction invariant **69.08%**.
- Literal pointer pair invariant **65.49%**.
- Pointer with depth input pair invariant **63.74%**.
The old paired sets are exploratory and frequently modify more than a single code token; compare strict novel pairs separately.

## 2. Mechanism grammar instead of 20 hand-written final-program templates

**A** = snapshot/alias, **B** = later rebinding, **C** = lexical shadow / distinct scope, **D** = live closure and runtime frame. Grammar production randomly selects identifier names, literal values, alias chains (1–3), sibling vs nested vs parameter scopes, conditional expression targets, call targets and noise-statement placement. Python `compile` + restricted `exec` verifies every output. Model sees Python's lexical tokens, INDENT/DEDENT markers, optionally derived indentation depth. It NEVER sees true scope ID, compiler def-use edges, executable state or event trace.

Training uses **A+B** and **C+D** only; AB/CD development sets are seeded independently. Hold out **A+C** with four independently generated production families:
- `alias_crosses_local_shadow`
- `alias_and_parameter_shadow`
- `alias_nested_shadow`
- `sibling_scope_alias`

All test examples contain exactly **four distinct numeric source literals, each occurring once**; this is intentional candidate-count control, not a guarantee against stronger syntax heuristics.

### Frozen V2 test protocol
An earlier V1 test seed was accidentally exposed during a 20-step code smoke test and **retired**. V2 seed `2026111704` was regenerated and frozen, *before final model evaluation*, to **256 pairs / 512 programs**.

- V2 test SHA-256 **`e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce`**.
- Four production clusters with **70,54,57,75** pairs respectively.
- Locked models *before evaluation*: `transformer` and `pointer-depth`, layers2, d64, AdamW lr.001, steps1000, batch32, five seeds0–4, final checkpoint only. No A+C-based architecture/hyperparameter selection.
- Source literal candidate chance for one program: **1/4=25%**. If predictions on paired programs are independent uniform candidate samples, chance of *both* correct is **1/16=6.25%**; if an oracle knows labels are distinct and predicts an ordered distinct pair uniformly, **1/(4×3)=8.33%**. Neither is a universal lower bound for code heuristics.
- The old experimental 20 kind OOD split cannot be reused as a fresh final test.

### Frozen V2 five-seed result (negative)
| Pre-registered small architecture | Individual answer | Both members correct | Prediction flip | **Prediction unchanged** |
|---|---:|---:|---:|---:|
| 2-layer Transformer | **35.90%** | **0.16%** | 14.30% | **85.70%** |
| 2-layer source-literal pointer with depth encoding | **37.89%** | **0.08%** | 15.23% | **84.77%** |

Every causal pair has **different correct answers**. Consequently high prediction invariance is explicitly a **failure** to adapt the answer, not desired robustness. For the pointer, per-seed individual accuracy = **32.81%,40.43%,50.00%,35.74%,30.47%** (SD ~7.7pp); Transformer = **39.06%,34.96%,37.70%,24.80%,42.97%**.

Bootstrap over four **entire production families** and five training seeds (7000 resamples): individual-accuracy 95% percentile range **[26.36%,42.91%]** for Transformer and **[22.93%,49.05%]** for pointer. Few clusters -> extremely weak extrapolative CI. For the pointer pair both-correct, range **[0.00%,0.56%]**. Do not quote as a general real-Python population estimate.

Per-production failures (five-seed means):
| Test-only production | Transformer accuracy / pair invariance | Pointer-depth accuracy / pair invariance |
|---|---:|---:|
| alias_and_parameter_shadow | 43.0% / 99.7% | 46.3% / 98.6% |
| alias_crosses_local_shadow | 40.6% / 99.6% | 48.3% / 98.9% |
| alias_nested_shadow | 32.6% / 99.6% | 46.1% / 99.6% |
| sibling_scope_alias | 28.4% / 52.0% | 16.3% / 50.4% |

**Script heuristic check on the sealed test** (predefined before V2): first source numeric literal **37.30% individual accuracy**, most frequent source numeric literal **26.56%**, last numeric literal **12.70%**. All of these oblivious heuristics are **100% prediction invariant and 0% pair both-correct**. Pointer 37.89% is nearly identical to first-literal heuristic 37.30%, and both are useless on strict pair correctness.

**Decision:** the apparent 2026-11-17 source-literal pointer OOD effect is not evidence of compositional learned lexical scope, and the new mechanism-combination counterfactual split *disconfirms* generalization. Before architecture work, expand the production grammar and real code distributions and compare stronger structure-sensitive baselines. The new test set is **consumed**; do not revise a model using this V2 outcome and then reuse the same V2 as an untouched test. A separate V3 must be preregistered for any further method claims.

## 3. Next hypothesis, not yet validated
`(P=source or event position, B=lexical binding, S=ancestral scopes, E=runtime call frame and write version)` must be learned or inferred from code tokens. The key missing test is *paired mechanism-composition* correctness, not 16-class raw accuracy. A legitimate compiler-provided binding table is an **oracle-assisted** upper bound, never a test-time input for a model claimed to infer binding itself. One-neighbor most-recent matching cannot solve local/global shadow, two independent closure frames or uninitialized lexical-local errors.

## 4. Reproducibility and caution
- The original ZIP retains the original local run. The complete generator, frozen V2 JSONL and manifest, training and metric aggregation scripts are also now committed under [`repro/`](repro), with [Actions full 5-seed replay](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44233344730) **12/12 passed**, and [Actions full AST symbolic ceiling](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44233615260) passed.
- **Numerical drift warning:** CI re-training is scientifically concordant but *not numerically identical* to the previous local results, especially for some pointer-depth seeds. Report original and rerun separately. See [per-seed replay variance archive](repro/frozen_v2_ci_replay_variance_verified.json).
- This grammar still contains a *finite set of syntax production types*; it is more compositional than the 20 fixed original templates, but NOT an open-ended arbitrary Python program generator. Four unseen test productions are too few for a conference paper.

## Independently implemented restricted AST symbolic ceiling (2026-11-18 UTC)
[AST baseline](ast_reference.py) interprets the source AST with lexical binding frames, statically determined locals (including `UnboundLocalError`), runtime closure frame instances, `global/nonlocal`, assignments, function calls and returns. **It does NOT use Python `exec` or `eval` at inference, nor does it read source labels.** It is **symbolic program interpretation, not a weak blind heuristic nor a learned model**.

The same source interpreter has now been [fully independently verified in Actions run 44233615260](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44233615260): fresh AB/CD/AC **600/600 each**, plus **512/512 individual programs** in frozen V2 and **256/256 paired both-correct**. Its input is source code, not the CPython labels, and it only handles the restricted benchmark grammar. Additional [regression CI 44231498354](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44231498354) independently checks 10 CPython lexical/closure/static-local examples. **The complete exact V2 source, 256-pair frozen JSONL, manifest, model code, heuristic code and aggregation code have now been committed under [`repro/`](repro)**, correcting the earlier missing-GitHub-files caveat. Old ZIP artifacts still provide the original locally run measurements.

**Scientific interpretation:** On this deliberately limited grammar, exact rule-driven binding is straightforward and full correctness is attainable. This is the **script upper bound**; the ~38% neural score and <0.2% strict-pair score demonstrate the tiny models did not discover the corresponding algorithm under the present training distribution. The script cannot by itself support an ICML architecture novelty claim.

### Verification log for corrected figures
Locally re-ran `test_grammar_protocol.py` (3×200 single examples, 3×100 paired examples, plus 256 frozen pairs and exact source hash), `eval_frozen_heuristics.py`, and `aggregate_new_grammar.py` (1,000 fresh cluster×seed bootstrap replicates; point means unchanged). The sealed V2 test SHA remained `e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce`. Bootstrap intervals vary with RNG/replications; the pre-existing 7,000-rep interval in this report remains the archived designated estimate.


### Post-hoc matched-sample comparison with the first-literal heuristic

Rather than compare point estimates on different samples, the frozen V2 script and neural predictions were matched by **pair ID** and trained seed. Bootstrap resamples **entire production types and entire model training seeds** (10,000 resamples, four production types). On *the same exact 256 held-out pairs* per seed:

| Neural model minus predeclared first-source-literal script | Mean individual-accuracy difference | 95% production×seed percentile interval |
|---|---:|---:|
| Transformer (five seeds) | **−1.406pp** | **[−15.97pp, +12.06pp]** |
| Pointer + indentation-depth (five seeds) | **+0.586pp** | **[−22.32pp, +17.47pp]** |

Pointer seed-specific differences vs script: [−4.49,+3.13,+12.70,−1.56,−6.84] pp. The 0.59pp point improvement is **not stable**, and uncertainty spans significant losses; four grammar production clusters are insufficient for publication-style generalization claims. This analysis is explicitly **post hoc on consumed V2**, not an architecture-selection criterion. Source and output are included in the updated attached ZIP under `repro/compare_to_script_posthoc.py` and `repro/frozen_v2_posthoc_neural_vs_script.json`.

## 5. Independently completed GitHub neural replay (2026-11-18 UTC)

The [full CI pipeline](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44233344730) completed **12/12 successful jobs**: generator + frozen SHA validation, exact predeclared script-heuristic replay, two architecture × five train-seed original code runs, and five-seed production-cluster aggregate. The same test SHA256 `e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce` and 256-pair count were verified in GitHub Actions. The per-seed differences from local runs are archived in [`repro/frozen_v2_ci_replay_variance_verified.json`](repro/frozen_v2_ci_replay_variance_verified.json).

| Metric | Original local 5-seed run | Fresh GitHub 5-seed replay |
|---|---:|---:|
| Transformer individual accuracy | 35.898% | **35.664%** |
| Transformer both-correct pairs | 0.156% | **0.156%** |
| Transformer prediction invariance | 85.703% | **85.625%** |
| Pointer+depth individual accuracy | 37.891% | **38.164%** |
| Pointer+depth both-correct pairs | 0.078% | **0.078%** |
| Pointer+depth prediction invariance | 84.766% | **82.344%** |

The maximum *individual seed* accuracy change from local to CI is **0.977pp for Transformer** and **11.719pp for pointer+depth**. It is important not to claim bitwise reproducibility when model performance depends strongly on numerical/backend/optimization variability. A separate, same-runner duplicate-seed audit has been initiated to distinguish within-run nondeterminism from environment changes. No model or hyperparameter selection was made using the consumed AC split.

These two independent training runs agree on the primary falsification: **strict A+C paired counterfactual correctness is <0.2% across both model families**, a far stronger negative result than any small shifts in single-example accuracy.
