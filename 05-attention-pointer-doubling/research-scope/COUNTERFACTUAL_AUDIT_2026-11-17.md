# Counterfactual lexical-binding audit — 2026-11-17

**Research status:** completed *local CPU pilot* on synthetic CPython-executable program templates. **This document records experimental outputs; neural training scripts and raw per-seed JSON are in the downloadable reproduction bundle in the corresponding chat**, not yet committed to this branch. Do not treat this branch alone as reproducing all ML numbers.

## Input/label contract
- 20 randomized, restricted Python program families; **12 train and 8 unseen grammar families**. Every generated sample has a terminating CPython `compile/exec` output, checked against independently specified expected result.
- Model input is lexical Python tokens (including INDENT/DEDENT), **not provided lexical binding IDs, exact scope IDs, compiler dataflow edges or execution traces**.
- Strong shortcut present: source-literal pointer models read a distribution over constant positions and the output value is a numeral printed in the source. This is an architecture-specific prior and not unconstrained code execution.
- Final target one of 16 integers, 20–35 (uniform class chance 6.25%), so trivial numeric heuristics can outperform chance.
- 2-layer width64 Transformer or source-literal position pointer; AdamW, 1000 steps, batch32, 5 training seeds; 128 validation examples per grammar class. Train and OOD grammar types disjoint, both synthetic.

| Model | 8 OOD grammar families, macro mean | Sample SD across 5 training seeds |
|---|---:|---:|
| Transformer classifier | 22.21% | 3.82pp |
| Transformer + syntax indent depth encoding | 23.12% | 4.55pp |
| Source-literal position pointer | 43.79% | 10.82pp |
| Source-literal position pointer + indent depth encoding | **56.91%** | **17.28pp** |
| Pointer + indent encoding and depth-logit bias | 55.47% | 16.24pp |

These are measured on a fixed validation protocol and **not independent sampling evidence for ICML generalization**; performance fluctuates strongly by seed.

## Hard holdout exposes noncompositional behavior
Additional **10** unseen counterexample families (128 programs each) challenge independent sibling lexical scopes at identical depth, repeated function calls and closure instances, global/nonlocal mutation, same source evaluated at different runtime events, and scope shadowing. Five paired interventions (64 **pairs** each) only count if **both** program outputs are correct.

| Seed0 model, train length | Hard 10-way macro | Paired full-correct |
|---|---:|---:|
| Transformer 1000 updates | 39.8% | 3.1% |
| Source-literal pointer 1000 updates | 42.5% | 4.7% |
| Pointer + depth encoding 1000 updates | 48.0% | 7.5% |
| Transformer 3000 updates | 47.5% | 19.1% |
| Pointer + depth encoding 3000 updates | 35.0% | 4.4% |

The hard probe is **one seed**, not a new five-seed estimate. Even the best seed can predict many individually randomized code templates yet fail paired binding changes. The sibling-scope query-flip pair at 1k had **0% both-correct** for Transformer, pointer and pointer+depth (seed0). All three effectively lack robust bound scope identity.

A longer-trained Transformer can memorize in-distribution templates (~99.9% training-type validation) while remaining poor on the original unseen grammar split (~22.4% macro) and hard paired tests (~19.1%). More training alone did not solve these one-seed examples.

## Why P+B+S is semantically incomplete

A static lexical scope ID is not a dynamic closure environment:

```python
def make(x):
    def f(): return x
    return f
a = make(20)
b = make(21)
assert (a(), b()) == (20, 21)
```

Each `f` has the **same** code location, name and lexical scope, but references a **different runtime frame**. The minimally adequate conceptual state must distinguish `(P=occurrence or event, B=lexical binding ID, S=scope ancestry, E=frame and event version)`.

Also, simple `y=x` copies a value at an **execution event**, while function closures capture references to environments, and Python local assignment makes a name lexically local even before the first write (causing `UnboundLocalError`). The [CPython independent oracle tests](test_scope_semantics.py) must pass before neural training modifications.

## Claim decision
- **Supported:** location-constrained output models are effective structural priors on these simple synthetic program families; adding syntax depth to pointer tokens helps OOD macro performance in this local setup.
- **Not supported:** learning of lexical binding identity, closure frame identity, global/nonlocal semantics under counterfactual interventions, real-source-code generalization, an ICML-ready architecture.
- **Novelty requirement:** a **learned** scope-equivalence/ancestor relation and dynamic environment/version retrieval, without feeding compiler gold edges at test time, must improve *paired* correctness and transfer; stronger and longer trained RoPE, compiler-assisted graph models and simple structured baselines must be included.

## Next falsifiable gates
1. Learn distinct **sibling scope identities** instead of a scalar depth embedding. Gold scope links can be used **only** in an explicitly auxiliary-supervised or compiler-oracle arm; primary model must infer from source token structure.
2. Represent **dynamic execution frames**; test two closures produced by the same source function with different values and late binding before/after updates.
3. Train on randomized 1–4 scopes/frames, test 5–8 + 8/16 writes. Report pair accuracy and exact source constant/event provenance, not merely correct final integer.
4. Compare on hand-edited and real open-source Python snippets with reproducible CPython output while avoiding unsupported dynamic imports/external dependencies. Favor a code-pretrained baseline or acknowledge research sandbox constraints.

**Status:** narrow GO for a mechanistic follow-up; **NO-GO** for ICML submission claims today.
