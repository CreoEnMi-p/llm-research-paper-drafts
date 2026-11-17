# Position + Binding + Scope (+ Execution Frame): Independent research assessment
*17 November 2026 · Experimental research branch only · NOT an ICML-ready claim.*

## Research verdict
Pivoting away from doubling is justified as an **experimental question**, not yet as a publication-ready model. The distinctive hypothesis should be:

> A neural model that retrieves **source occurrences conditioned on their lexical binding and runtime environment** can extrapolate across new variable names, lexical scopes, assignment versions, and multiple calls more systematically than ordinary transformers and position-only retrieval—*without receiving an oracle definition-use graph at inference*.

**Important semantic correction:** `State = (Position, Binding, Scope)` alone cannot model Python closures invoked multiple times. For a language with dynamic executions, minimally distinguish `(source position or execution event P, lexical binding B, lexical scope S, frame/time/version E)`.

## Formal problem statement (no compiler-answer leakage)
For a variable use `u` at runtime event `t`:
1. **Lexical resolution:** `b = Resolve(name(u), scope(u), lexical declarations)`, independent of current value. Names can be local, free/cell, global, or nonlocal; *uninitialized local* must be an error, not a fallback global.
2. **Runtime store:** `v = Store[b, frame(t), latest completed write before t]`. Source line number and execution time differ, especially for closures/loops; a function may be instantiated in multiple runtime environments.
3. **Assignment semantics:** `y=x` copies the resolved value at the **assignment event**; `def inner(): return x` captures an environment **by reference**, not the value of x at function definition.
4. **Pointer provenance:** the candidate address is a source-token or execution-event occurrence, not merely token identity. Two occurrences of the same `x` can mean different bindings or different write versions.

Never feed true `binding_id`, `scope_id`, `def_use` edges, or execution trace to a *learned inference model* while claiming these are learned. Compiler-provided gold edges may be used for analysis, an explicitly **separately labelled supervised** variant, or the oracle upper bound.

## Mechanistic counterexamples that name+recency cannot solve
- Shadowing: `x=20; def f(): x=21; return x; f(); print(x)` ⇒ 20; naive newest same-name assignment returns 21.
- Explicit global: adding `global x` inside f changes the final global print to 21 despite same source name.
- Assignment snapshot: `x=20; y=x; x=21; print(y)` ⇒ 20.
- Late closure read: `x=20; def f(): return x; x=21; print(f())` ⇒ 21 (not definition-time x).
- **Two closure instances:** `make(x)` creates distinct `inner` closures with the *same source location and lexical scope*; `make(20)()` and `make(21)()` return distinct values. Runtime frame identity `E` is indispensable.
- Static-local exception: `x=20; def f(): y=x; x=21; return y; f()` raises `UnboundLocalError`, even though the global x was initialized. This demands full-block lexical analysis.

Relevant language authority: [Python execution model](https://docs.python.org/3/reference/executionmodel.html).

## Prior art and novelty boundary (verified Nov 2026)
| Reference | Demonstrated prior work | Missing opening for us |
|---|---|---|
| [Wu, Geiger, Millière, ICML 2025](https://proceedings.mlr.press/v267/wu25j.html) | Transformers learn assignment chains (up to 4 hops), specialized memory-routing heads | Need new OOD scopes, calls, execution frames, and interventions |
| [Işık & Li, ICML 2026](https://proceedings.mlr.press/v306/isik26a.html) | Provable invariance to variable-symbol renaming in a Transformer | Cannot claim renaming invariance alone as new |
| [Zhao & Du, ICML 2026](https://proceedings.mlr.press/v306/zhao26h.html) | Gradient-starvation explanation and topology-aligned training for deep semantic binding | Need show gains beyond structural auxiliary loss/curriculum |
| [Rajput et al., Oct 2026 preprint](https://arxiv.org/abs/2610.39489) | AST scope/definition-use conditioned decomposition and causal edits of frozen Code LMs | Their structural labels are provided by external analyzer; our hypothesis requires learning scope selection from input at inference |
| [GraphCodeBERT, ICLR 2021](https://arxiv.org/abs/2009.08366) | Dataflow structure injected into code representation | Must compare compiler-assisted graph baseline fairly |
| [Allamanis et al., ICLR 2018](https://arxiv.org/abs/1711.00740) | Graph learning from program syntax/semantic edges for variable misuse | Must not rediscover graph-based def-use reasoning |
| [CODE2BENCH, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f8e55d98b0c2569bd0aa25b076e6b3f8-Abstract-Conference.html) | Uses scope-graph dependency analysis in real-code data construction | Existing tools generate scope-aware evidence; use as benchmark methods |

**A compiler/interpreter exactly solves valid Python.** A learned scope mechanism's claim cannot be to outperform a Python interpreter; rather it must deliver an *inductive bias transferable to incomplete/noisy text, undertrained code models, natural language binding, or pretrained LM integration*, with measured accuracy/cost.

## Initial self-contained executable Python benchmark
A fixed synthetic Python **source-code grammar**, *not* human-written production repositories:
- **12 training grammar families**, 8 OOD families: direct copy, alias snapshot, global mutation, function local shadows, parameters, closure reads, nested lexical scopes, nonlocal mutation, separate closure instances, unseen reassignment lengths and deeper nesting.
- Python's `tokenize` lexer produces NAME/NUMBER/OP and INDENT/DEDENT markers. No ground truth binding edge or scope ID is an input. A legitimate syntax **indentation depth** is available only to the depth models; distinguish this from true lexical scope identity.
- Every generated program is **executed by CPython** with restricted print to validate a unique final integer (16-class output, 20–35). One answer per program, not multiple teacher-forced queries. Train samples fresh with randomized identifiers and values. The OOD grammar families are excluded from training. Validation seeds independently fixed. No model pretrained on code.
- First pilot settings: 1000 updates AdamW, batch32, two-layer d64 Transformer encoder or learned numeric-literal source-position pointer. **Five** random initialization seeds.

| Model | OOD grammar macro accuracy mean ± sample SD (5 seeds) | Important qualification |
|---|---:|---|
| Transformer direct classifier | **22.2% ± 3.8pp** | 78.7k parameters |
| Transformer + parsed indent-depth embedding | **23.1% ± 4.6pp** | Depth alone does not help direct classifier |
| Transformer + learned pointer to literal **source occurrences** | **43.8% ± 10.8pp** | 86.0k parameters; no binding/scope learned identity |
| Pointer + syntax-derived depth embedding | **56.9% ± 17.3pp** | 86.3k parameters; **large seed instability** |
| Pointer + depth embedding + extra depth scoring bias | **55.5% ± 16.2pp** | Extra bias did not improve mean |

The best-performing seed does not establish a systematic algorithm; the variance itself is a major negative result. The output pointer selects among numeral literals in code, not assignments or oracle binding IDs. It is **not** a model of all Python.

**Next methodological test:** paired causal programs differing only in query target, `nonlocal/global`, or function choice, scored by whether both outputs are correct. Additional held-out tests with two **sibling functions at identical indentation depths** and multiple closure instances expose depth as an inadequate proxy for lexical scope identity. Those tests are separate from the original eight OOD families.

## Proposed model architecture (research hypothesis, NOT validated)
1. **P — Occurrence pointer:** trainable distribution over source occurrences, with optional execution-event cursor; *must not* collapse to token identity. Use a common learned matcher and position-aware state, as validated in Sanford k7, but do not force doubling.
2. **S — Learned lexical environment:** update a **soft scope stack** from syntax boundaries (`INDENT`, `DEDENT`, `def`, `global`, `nonlocal`). A scope is not equivalent to a scalar indentation depth: two sibling scopes at depth1 are different. Predict scope ancestry/reachability from syntax, not using a compiler-given scope-ID.
3. **B — Binding identity:** factor variable-name equivalence from scope membership. `name=x` plus active lexical environment predicts which binding address should be used; track parameter bindings, shadowing, globals and nonlocals. Desired equivariance under alpha-renaming.
4. **E — Runtime environment/version:** represent dynamic invocation frames and write events. A closure carries an environment *reference*; a simple assignment copies the resolved value. `UnboundLocalError` must be a legitimate output, not global fallback.
5. **Read operation:** candidate scores combine learned name similarity, learned scope reachability, legal runtime event ordering and learned recency. No test-time gold `binding_id` mask. Distinguish full smooth differentiable training from sparse Top-K at inference; random QK and training curriculum sensitivity were decisive on Sanford.

Possible combined score
`s(u,j) = q(u)·k(j)/√d + λs ScopeReach(u,j) + λb SameBinding(u,j) + λr Recency(event(j), event(u))`,
where **all three functions** must be defined with explicit causal inputs, learned vs oracle status, and ablations. If exact scope graph or execution traces are provided, that treatment is **compiler-assisted**, not a learned resolver.

## Mandatory evidence gates before calling it ICML research
1. **Existence of a learnable scope mechanism:** ≥5 seeds, one-to-one causal counterfactual *pair* accuracy; same-depth sibling scopes, `global/nonlocal`, closure frame instances, Python static-local errors, alias versions, and heldout 8/16 writes. Report source-literal or event provenance accuracy, not only final-value accuracy. Distinguish label-flip accuracy from ordinary macro.
2. **Matched strong alternatives:** tuned deeper RoPE encoder, learned literal pointer, recurrence, symbol-invariant baseline, graph/AST-assisted reference (explicitly marked as oracle-assisted), identical training labels/tokens and compute/FLOPs/memory. More training for baseline to rule out slow emergence. Multiple independent program-level bootstrap intervals, not token-level i.i.d. fiction.
3. **Real value beyond toy grammar:** test on real open-source Python selected by AST and execution, with disjoint projects (no lexical-template leakage); or cross-domain transfer to natural language entity-reference/temporal binding. Include execution-trace validation, name-renaming counterfactuals and dependency-graph oracle. If compiler exact solve dominates and no transfer benefit remains, stop.

**Go/no-go today:** narrow GO for a *mechanistic pilot* (positive OOD hints); NO-GO for paper claims. The novelty is **not** `scope embedding + pointer`; it would have to be a trainable, compositional binding-and-event state that a scope-blind pointer and strong transformer cannot replicate.
