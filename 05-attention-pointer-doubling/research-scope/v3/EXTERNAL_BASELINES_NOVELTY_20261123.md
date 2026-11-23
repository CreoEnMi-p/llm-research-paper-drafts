# Paper 5 — external baselines and novelty boundary (2026-11-23)
**Status:** literature checked, published neural baseline reproductions not completed; this note was prepared independently of the frozen V3 model results.

## Existing model and novelty boundary
The present two-layer d64 pointer model encodes source tokens, positions, and indentation depth with a standard Transformer. A final learned query/key head scores four **numeric source occurrences** and accumulates their mass by numeric class. It has no explicitly represented compiler-quality scope table, binding ID, dynamic frame identity, or write history. Whether it learns them implicitly is exactly the counterfactual hypothesis under test.

The pointer-scrambled arm keeps the same trainable tensors and neural operations, but randomizes the stable correspondence between source occurrence and output literal. The pointer-canonical arm first alpha-normalizes names by first occurrence; it is **not** the published symbol-invariant architecture.

## Verified published competitors

| Work | Established contribution | Required comparison |
|---|---|---|
| [Wu, Geiger, Millière — How Do Transformers Learn Variable Binding in Symbolic Programs?, ICML 2025](https://proceedings.mlr.press/v267/wu25j.html) | Learned addressable memories and attention routing on assignment chains | Basic source lookup and variable chain learning are known, not novel |
| [Işık & Li — Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning, ICML 2026](https://proceedings.mlr.press/v306/isik26a.html) | Provably rename-invariant transformer; [source code](https://github.com/BU-DEPEND-Lab/Symbol-Invariant-Transformer) | Reproduce the real architecture, not just alpha-canonicalization |
| [Zhao & Du — Syntax vs. Semantics: How Transformers Learn Deep Dependencies, ICML 2026](https://proceedings.mlr.press/v306/zhao26h.html) | Deep dependency learning and topology-aligned objective; [project](https://github.com/jr-zhao/Deep-Dependencies) | Compare matched training, strong optimization and topology-aligned objectives |
| [Guo et al. — GraphCodeBERT, ICLR 2021](https://openreview.net/pdf?id=jLoC4ez43PZ) | Dataflow-guided neural code representation; [repository](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT) | Mark gold dataflow/AST edges as compiler-assisted, not model-inferred |
| [Allamanis et al. — Learning to Represent Programs with Graphs, ICLR 2018](https://arxiv.org/abs/1711.00740) | GGNN structured variable naming/misuse | Evaluate a genuine learned graph comparator |
| [Program Bindings Inside Code Models, Oct 2026](https://arxiv.org/abs/2610.39489) | Static AST/scope/def-use decomposition and causal editing | Inferred source structure is the open problem, not static annotated links |

## Strong baseline evidence status
- Candidate-masked depth Transformer: implemented and preregistered for V3. Same legal source literals and 2-layer/5k training, but not parameter- or FLOP-matched.
- Pointer-scrambled: implemented. Parameter counts and main forward operations matched exactly; tests stable occurrence-to-value map, **not by itself learned lexical scope**.
- Pointer-canonical: implemented deterministic first-appearance name canonicalization, but **not** Işık–Li's published model.
- Restricted symbolic AST interpreter: implemented and verified on all 448/448 V3 pairs. It is a scripted ceiling, not a learned neural baseline.
- Published GraphCodeBERT, GGNN, Işık–Li, and Zhao–Du implementations: **not reproduced on V3**, therefore external-baseline gate remains open.
- Non-oracle script baseline is chosen on AB/CD dev only and applied once to V3. The chosen simple policy scores 14.29% paired both-correct. One of seven families, late_closure_snapshot, is trivially solved by it; do not hide that benchmark weakness.

## Conditions for conference-level claim
1. Compare learned competitors on the same frozen pairs, name split, source literal candidates, data budget and random seeds; report learned parameter count, measured training/inference compute and learning curves.
2. Report strict pair correctness, prediction flips, family and name-stratum metrics, alias-depth extrapolation, attention-address mass, and entire-family/seed uncertainty; do not optimize on sealed test families.
3. Keep compiler-free source-only learned models separate from compiler-assisted graph/dataflow models, whose extra test-time structure changes information access.
4. Novel binding/scope/frame claims require genuine transfer to unseen lexical sibling scopes, global/nonlocal semantics, late-closure state and distinct runtime-frame instances, plus evidence outside these finite synthetic grammar families.
5. If V3 informs a new architecture, all confirmatory claims need a new preregistered V4. A better restricted Python benchmark score alone is not an ICML innovation.

**Research judgment:** a reproducible learned mechanism for binding/scope/runtime composition without compiler test-time edges could be novel. The current source-location readout does not establish that claim.
