# LLM Research · Main-agent research control plan

> Status: Phase A — critical validation. Research branch only. Last planning update: 2026-11-15 (America/Los_Angeles).
> This is a plan, not a claim that the assigned research tasks have been completed.

## Objective

Find **one defensible, non-trivial research question** with evidence robust enough to justify further work toward a top ML conference. Do **not** aim to polish the original six drafts mechanically or maximize a single synthetic benchmark.

Priority hypothesis: **03 — memory interference / unnecessary writes in linear attention**.
High-risk alternative: **05 — causal pointer-doubling for arbitrary edge order and query-conditioned hop count**.
Audit reserve: **01 — dynamic output clustering under Adam reparameterization**.
02/04/06: literature monitoring and reference baselines only unless an unexpectedly strong counterexample changes the decision.

## Evidence already established

- [Original-code CPU reproduction](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43792798882): 03 MQAR baseline loss 0.373 vs predicted-skip 0.173. 05 reverse 7-hop/3-layer 0.972 vs forward 0.081 and random-order 0.056. This establishes **small synthetic reproducibility**, not real-LM effectiveness.
- [Matched four-seed screen](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43793394215): 03 average loss base 0.6147; previous-query gating 0.2593; shuffled-content gating 0.4642; current-key gating 0.2153. 05 reverse ≈0.980; forward ≈0.077; random ≈0.048; distinct predecessor mechanism on forward 8-hop/4-layer ≈0.926.
- Known hazards: 03 compared against an author's 'GDN-2-style' approximation, not verified against the reference implementation; current-key read is a strong competitor. 05 single scalar gate per layer is not query-conditioned. 01 paper's pseudocode resets Adam state after regrouping, but actual grid.py does not. FineWeb data dependencies missing for 01/02/06.

## Assignments for independent subagents

### A — Paper 03: Mechanism and falsification (highest priority)

**Research question:** Are repeated answer-token writes a *distinct, measurable cause* of interference in associative memory, beyond generic reduction of writing and existing delta-residual controls?

1. Reproduce core claims in >=5 matched seeds, log mean/dispersion and per-seed differences, fixed held-out evaluation.
2. Compare previous-query read gate, current-key read gate, shuffled/read-broken gate, *write-rate-matched* static gate, and (where feasible) token-only learned gate, not merely gated vs ungated.
3. Measure causal influence of a write on **other** stored key-value pairs / subsequent reads. Is the extra error due to redundant write, overall memory capacity, or an optimization/training transient?
4. Before large-scale training, reproduce in an official Gated DeltaNet-2 / suitable reference implementation and report architecture/kernel differences. Clearly distinguish reconstructed baselines from official ones.

**Go condition:** content-specific benefit survives matched generic-write controls, has an interpretable interference mechanism, and has at least one credible stronger-baseline or real-text retrieval indication.
**Stop/pivot condition:** strongest existing residual/read gate removes the gap, or benefit only appears in engineered MQAR and cannot be linked to observable memory corruption.

### B — Independent prior-art and novelty red-team

**Research question:** What exact scientific claim, if any, remains unclaimed after close reading of related papers?

- For 03: compare Delta Rule, PH-attention, Gated DeltaNet-2, 'How Linear Attention Remembers', relevant 2026 linear-memory work including Muir; check method equations and actual controls, not abstracts only.
- For 05: compare Memory Mosaics, k-hop induction / log-depth theory, classic pointer jumping, graph/algorithmic generalization, and methods supporting variable-hop queries.
- Provide one table: ours / closest 2-3 methods / identical aspects / substantive difference / critical experiment to discriminate.
- Include real links, version/date, quotations limited to necessary claims. Assign novelty-risk (high/medium/low) and confidence based on evidence, no speculative novelty endorsements.

**Go condition:** precise, important unsolved gap and a discriminating experiment.
**Stop condition:** method is substantially rebranding already published mechanism without a new empirical/theoretical claim.

### C — Paper 05: Generalization and causality stress test

**Research question:** Can one trained model perform variable-hop lookups when graph edges appear in arbitrary causal order?

1. Baseline with forward/reverse/random-order on matched seeds, include predecessor doubling and conventional attention / explicit-hop baseline.
2. One model trained on mixed edge orders *and mixed hop counts per query*. The current gate is global per-layer; do not label a fixed-hop model as variable-hop generalization.
3. Held-out graph size, distractors, repeats/aliasing, unseen lengths; report split by hop count/order, not just aggregate accuracy.
4. Document whether the representation of the graph provides a hand-coded algorithmic advantage unavailable to baselines; compare equal compute and note quadratic attention overhead.

**Go condition:** reliable generalization with arbitrary order + query-dependent hops, and advantage over the closest algorithmic baseline.
**Stop condition:** only curriculum-specific/reverse-order wins or performance follows exact-token shortcuts.

### D — Paper 01: Optimization/reparameterization audit (small scoped reserve)

**Research question:** Is dynamic cluster sharing a real rare-word generalization mechanism or mainly an altered Adam preconditioning path?

- Resolve missing train/validation data provenance, exact tokenization and baselines *before* numerical comparisons.
- Compare regroup with Adam state held vs reset vs optimizer-state-preserving transformation, and non-sharing matched parameterization controls.
- Separate rare-token validation from unknown-token artefacts; test larger effective batch with controlled tokens.
- Do not expand into a broad hyperparameter grid unless a robust mechanistic gap survives.

**Go condition:** gains persist after optimizer-state controls and realistic batch changes, with interpretable rare-word mechanism.
**Stop condition:** claimed effect disappears under consistent optimizer treatment.

## Subagent reporting contract

Every report must have:
- **Research question / hypothesis** (one falsifiable statement).
- **Baseline and closest prior art** (exact code/version/links).
- **Evidence**: code commit/branch, exact command, dataset provenance, seeds, eval protocol, metrics, statistical dispersion.
- **Adverse and null results**, not only improvements.
- **Interpretation** clearly separated from measurement.
- **Decision requested**: continue / revise hypothesis / stop, with next single discriminating experiment.

All agents use independent `research/agent-<name>` branches and upload raw logs/results (including failures). Do not overwrite original `PAPER.md` or push into `main`. Do not merge unless reviewed by main agent. Any changed data, eval metric or code version invalidates a direct numerical comparison unless re-run in a matched setup.

## Decision gates

| Stage | Main-agent question | Output |
|---|---|---|
| A0: Baselines | Can the strongest *published* baseline be reproduced fairly? | Comparison specification and reproduction receipt |
| A1: Causality | Can alternative explanations for the claimed gain be ruled out? | Matched ablation matrix, >=5 seeds for noisy training |
| A2: Generalization | Does the claimed mechanism survive beyond toy assumptions? | Real text/held-out task evaluation |
| A3: Research value | Does the new contribution remain after prior-art check and compute accounting? | Formal go/pivot/stop decision |
| A4: Paper preparation | Only after A0-A3 pass | Scope, experiments, claims, disclosure, reproducibility package |

## Portfolio discipline

Prioritize `03` until a hard counterexample or publication overlap changes its expected value. `05` runs as the high-risk alternative; `01` limited to one debugging/causal audit. Prevent scope creep: no fourth major experiment track and no polishing manuscripts before A2. Avoid using number of runs as a substitute for independent evidence.

## Main agent status ledger

| Track | Current status | Next needed evidence |
|---|---|---|
| 03 | Replicated on toy MQAR; 4-seed intervention evidence | matched write-rate controls, official baseline, non-synthetic retrieval |
| 05 | Reverse-only win replicated; forward/random failures stable; predecessor baseline works | mixed order + per-query hop-count training |
| 01 | Code/paper optimizer mismatch identified | controlled Adam-state study |
| 02 | No independent numeric reproduction (missing data/logs) | provenance if prioritized |
| 04 | Background/control for 05 | no standalone work |
| 06 | No independent numeric reproduction (missing data) | provenance only if prioritized |
| Prior art | Initial scan incomplete for final novelty claim | precise overlap matrix from B |

## How to synchronize

Agents can work independently; when a result or repo commit is ready, send main agent the branch/commit and the report or exact Action run URL. Main agent will review the **claim and evidence**, not recite the agent's conclusions, and update this ledger with accepted, rejected or pending assertions. Progress is reviewed when evidence is brought into the conversation; no background monitoring is implied.
