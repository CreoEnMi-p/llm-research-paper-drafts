# Paper 05 — Reviewer’s six decisive follow-up questions (2026-11-17)

**Branch:** `research/05-pointer-doubling-audit` · **Draft PR:** [#10](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/10) · **Source:** paper 05 `pdg.py` adapted in independent generated audit files; paper and main unchanged.

## Completed follow-up addendum (validated after initial report)
**Five-seed corrected run now fully complete:** [Action 44046493616](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046493616), [archived final metrics](first5_3000_adamw_results.json). BLN3 has **5/5 seeds at 100% first question**; RoPE3 five-seed first mean **12.578125%**, sample SD **1.5963 percentage points** (candidate chance 12.5%). The trained L6 onehop pointer is no straw man: **97.8516% and 99.6094%** on its two seeds after 3k AdamW steps. Time per run is hosted-CPU noisy; separate same-runner profiler needed to claim compute superiority.

**Real assignment aliasing challenge:** [Action 44047579349](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44047579349), [archived results](semantic_adversarial_results.json). With training up to four writes to the same variable and evaluation on eight (96 independent held-out programs), **plain BLN4 gets 0/96 on both training seeds** (RoPE4 likewise 0/96). In contrast, a **strong, fixed near-maximum-logit recency RANK heuristic** boosts BLN4 to **47/96=48.9583% (seed0)** and **43/96=44.7917% (seed1)**. This is strong evidence for a substantive *recency ambiguity and length-shift problem*, plus an incomplete but positive improvement. However **0.45–0.49 is not generalization solved**; the heuristic considers scores within a tolerance of top logits rather than only exact symbol matches, and it adds full T² rank/near-max operations. Do not call this an efficient exact lexical tie-break implementation.

**k63/127 controller diagnostic finished:** [Run 44046954757](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046954757), [metrics](longchain_gate_800_results.json). k63 first-query oracle +8 gate init .3125 vs learned init .3281; k127 .3125 vs .28125 after 800 steps of all-query training, N=4, eval64. Results are close to 0.25 candidate prior with large sampling uncertainty; no demonstration of 63/127 capability. The bottleneck may be attention-match fidelity, long-state propagation, data coverage, or optimization; adding gate bits alone did not suffice within this tiny pilot.

**20k AdamW five-seed Sanford terminal-only comparison:** [workflow 44046440163](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046440163), marked in-progress when compiling this report. It has **not** been validated complete and cannot be used as a publication claim.

---

## Executive decision
**The headline 34% RoPE/NoPE number was contaminated by a combinatorial answer-elimination opportunity, not evidence of partial multihop capability. Correcting to first-question accuracy makes ordinary 3–5-layer baselines roughly chance; Pointer Doubling is reproducibly near 100% under both Muon and AdamW in the deliberately reversed chain task. However 6-layer sequential one-hop pointer reaches ~98% after stronger training, and terminal-only Sanford k7 has not shown BLN superiority. Do not claim generic multihop capability or use 34% as chance baseline in the paper.**

### A. Fix all-query scoring bias — confirmed and re-measured
`hopx` draws N unique final values and asks each of the N corresponding queries once in random order. In teacher-forced causal inputs, previous query answers are visible to later queries. A candidate-set-aware model can select by elimination without tracing earlier chain facts:
`chance_oracle_by_question[j] = 1 / (N-j)` for zero-based j (N=8 → 12.5%,14.2857%,16.67%,20%,25%,33.33%,50%,100%).
Aggregate `H_8/8 = 0.339732142857`. This is **a specific candidate-elimination strategy’s expected success**, not uniform chance over the 256-token vocabulary (0.390625%) or a proof about arbitrary models. Last question can be solved perfectly without looking up any multi-hop chain.

[Metrics adapter](build_reviewer.py) records `by_index`, `first_acc`, `index_curve`, `chance_elimination_by_query`, `first_only`, and supports AdamW. [Raw 800-step per-index results](indexed_800_results.json) and [run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44045092321).

| k/layers, seed0, 800 steps | First question | All 8 questions |
| --- | ---: | ---: |
| k7 RoPE L3 | .128906 | .351563 |
| k7 NoPE L3 | .105469 | .334473 |
| k7 RoPE L5 | .105469 | .321289 |
| k7 NoPE L5 | .121094 | .339355 |
| k7 onehop pointer L6 | **.777344** | .769531 |
| k7 BLN L3 | **1.000000** | .999512 |
| k15 RoPE L4 | .121094 | .340820 |
| k15 RoPE L6 | .152344 | .345703 |
| k15 NoPE L6 | .117188 | .328613 |
| k15 onehop pointer L8 | .121094 | .339355 |
| k15 BLN L4 | **.996094** | .999512 |

The first question has no previous query answers. Training with *first-question loss only* likewise fixes this leakage in the learning objective.

### B. Strong deep baseline; AdamW and 5 independent training seeds
[Run 44046493616](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046493616): reverse-block k7, first-question loss only, AdamW LR .003, batch32, 3,000 steps, 512 fixed evaluation rows, separate training seeds 0–4.
- **RoPE L3 first question:** [.134765625,.130859375,.101562500,.119140625,.142578125] → **mean .12578125**, sample SD .015963; aligns with candidate chance 1/8.
- **BLN L3 first question:** five seeds all **1.000000**.
- **Sequential onehop pointer L6:** seed0 **.978516** after 3,000 steps (seed1 status should be verified in its Actions job before paper inclusion). Seed0 training ~451 sec vs BLN ~404–478 sec across five jobs; hosted-runner walltimes are noisy, this is not a FLOP-exact guarantee.
- L+2 depth plain RoPE/NoPE remains near first-query chance at 800 steps for both k7 and k15. The one-hop pointer is **not** a straw man; after sufficient training it nearly solves k7. The strong claim must be matched-compute learning efficiency / depth compression, not uniqueness of multi-hop skill.

These corrected results **supersede** all earlier interpretations of the 34% all-query accuracy. 3,000 steps × batch32 is still far short of the requested 20,000-step baseline; do not claim absence of late learning transitions.

### C. Sanford author generator: terminal-only k7, valid (nonnull) answer
[Terminal script](sanford_terminal7.py) calls the **exact pinned official author generator** `InductionHopsFullSequenceTask` (source commit `5735bca9545545774098994e21392b51b285bcc5`). Changes experiment protocol *on purpose*: fixed k7; one target at **last** position only; train/eval reject `~` null results; full-sequence per-position supervision **off**; same training/held-out validation across arms. Label alphabet 4; uniform random chance .25. This is NOT the original paper’s GPT2 100,001-step training.

[3k-step CPU AdamW batch32 pilot](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44045307737), [saved metrics](sanford_terminal7_3000_pilot.json):
| Model | Seed0 LR .001 | Seed1 LR .001 |
| --- | ---: | ---: |
| 4-layer RoPE | .335938 | .308594 |
| 4-layer NoPE | .312500 | n/a |
| 4-layer BLN | .316406 | .308594 |
| 8-layer sequential pointer | .339844 | n/a |
| 8-loop tied NoPE | .289063 | n/a |

On this harder terminal-only task, RoPE and BLN both fail to solve at 3k steps. **No evidence for the hypothesized RoPE drop / BLN survives story.** Full-sequence original Sanford transfer numbers from the older audit are *not comparable* due the changed loss.

A high-budget, real 20,000-update AdamW comparison with **batch48 and 5 seeds per RoPE/BLN** was launched in [Action 44046440163](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046440163). **Do not report a 20k result unless those jobs finish successfully; in-progress does not count.** That run directly addresses slow-phase-transition concerns. Five seeds in the 3k reverse-task benchmark are complete (above) but do not replace these long tasks.

### D. k63/6 and k127/7
[First-only 400-step stress run 44045871247](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44045871247), [raw diagnostics](stress_400_results.json): N4, V1024, fixed reverse block order, k63/L6 and k127/L7 BLN, 2×depth onehop, and L+2 RoPE. With only **B2 for k63, B1 for k127**, and eval 32 rows, all arms produce first-question accuracy .09375–.15625 — **no supported positive claim**. Four known candidates gives .25 as elimination-prior reference, but ordinary random over vocabulary is 1/1024.

To isolate controller optimization from pointer-state degradation, [gate warm-start run 44046954757](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046954757) trains with all query labels and compares `--pgi 8` (the correct +1 binary gates for k=63 and127), versus default `--pgi 0`. At 800 steps k63 first-query .3125 (oracle gate init) vs .3281 (learned), no convincing gain. Any later k127 results must be confirmed from job logs. **Failure to converge under these tiny long-sequence CPU training budgets is not a representational impossibility result.**

### E. Executable semantics, lexical shadowing and recency bias
[Executable Python benchmark](semantic_assignments.py) generates **synthetic valid Python programs** of assignments, RHS copies, overwrites, function-local shadowing and returns; runs them with CPython `exec` as reference truth before tokenization. Model input is a **normalized pair-token representation** with function markers; this is not raw production Python, not RULER, and not proof of real-code generalization. Training loss supervises only final query result.

[Simple grammar run 44045753765](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44045753765), [saved results](semantic_1500_results.json): RoPE L4 shadow ~.510, NoPE L4 shadow ~.531, BLN L4 shadow 1.0 for two seeds, sequential pointer L7 shadow 1.0 for one; longer chain7 solved by BLN and onehop. Adding a small near-tie recency bonus initially gives no measurable benefit over BLN, since unmodified head already solves easy repeated-name cases.

To stress the precise proposed new contribution, [adversarial generator](build_semantic_adversarial.py) adds held-out 8 same-name overwrites (training sees 4), deeper function shadow, and optionally stronger **near-max matching score recency ranks** ([rank ablation](build_semantic_rank.py)). Original BLN seed0/1 achieves 100% on 4 overwrites but **0%** on held-out 8 overwrites; RoPE also 0%, deep onehop also 0%. This is a real *distributional failure within synthetic executable programs*. A score-near-tie bonus is **not the same as an exact token-equality-only tie-break**; do not claim the latter is implemented. Verify [adversarial run 44047579349](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44047579349) for recency ablation and full metrics before making a novelty claim.

### F. Computation and uncertainty requirements still outstanding
- Previous same-runner k15 CPU fwd/back BLN ~189ms vs standard RoPE ~89ms (2.13×), with partial sampled matmul FLOPs ~2.71×. This does not automatically transfer to k7 or k127. Proper fair comparative tables should report param count, actual training/validation wallclock, FLOPs and max GPU memory; match optimizer, training examples/tokens and report best checkpoint and final result.
- New 5-seed k7 results strongly establish effect on **first-query synthetic reverse-order**. With only 512 fixed eval rows and correlated runs on same validation data, use bootstrap over independent *rows*, coupled with seed variation; do not treat question slots as independent examples or advertise tiny 0.01 accuracy differences as significant.
- For general-purpose paper acceptability still need: 20k+ terminal Sanford outcomes with chosen tuned hyperparameters; compute-matched 1hop at 15/4 and long chains; k63/k127 data sufficient for learning curves; external realistic code dependency tasks with fresh data/text and non-template semantics; independently implemented nearest-match and dynamic-scope algorithmic baselines.

## Paper editing direction
**KEEP:** 'Architecture-injected causal pointer doubling can learn reverse-ordered associative chains with very few effective layers; first-question precision remains close to perfect across 5 training seeds under AdamW while conventional RoPE3 is at 1/8.' **QUALIFY:** 'A more deeply unrolled one-hop pointer approaches the same ceiling with more effective layers; this is a depth-vs-compute tradeoff.' **DELETE:** 'RoPE and NoPE understand one-third of 7-hop queries', '33% is above chance for ordinary attention', 'standard attention cannot perform multihop in general', 'real-code task improvement', '63/127 demonstrated.' **NEW OPEN HYPOTHESIS:** exact-match-only recency prioritization plus causal assignment-time snapshots could enable true last-write-wins / shadowing, but must survive adversarial 8+ repeated bindings and strong looped baseline before claiming success.

## Reproduction
- Original `pdg.py` main unchanged; new generators in `audit/`.
- All raw successful Actions executions and outputs linked above; one failed adversarial workflow attempt from an artifact path bug was repaired, not counted as experimental evidence.
- Core corrected files: `audit/indexed_800_results.json`, `audit/sanford_terminal7_3000_pilot.json`, `audit/semantic_1500_results.json`, `audit/stress_400_results.json`, `audit/build_reviewer.py`.
- Do not merge Draft PR until 20k experiment, adversarial recency experiment and long-chain controller diagnostic are independently inspected.


## Additional mechanistic checks — occurrence identity vs. recency (2026-11-17)

### A. Exact per-query chance correction is now in the paper branch
The original [PAPER.md](../PAPER.md) has been explicitly annotated and its invalid Figure 3 horizontal `1/N` chance line retired (not redrawn as equivalent). The new [query-index plot](../figs/query_index_corrected.svg) shows the valid analytic `1/(N-j+1)` elimination baseline and exact by-index validated data. Legacy Table 1/2 values are retained **with superseded labels**, not silently converted to unrun first-only measurements.

### B. Duplicate-name failures are confounded with **OOD assignment length**
In [scope control run 44049635221](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44049635221), [machine-readable results](scope_matched_length_results.json), all programs are checked by executing actual Python before labels are used. `latest8_plain` has eight writes to one name; `unique8` has eight writes to eight different names. **Both are eight two-token assignments of matched encoded length**.

| Arm, 1,500 AdamW steps | `latest8_plain` | `unique8` | Global `x` queried directly after local function `x` |
|---|---:|---:|---:|
| BLN no recency seed0 | .0547 | .1094 | .1563 |
| BLN no recency seed1 | .0078 | .0781 | .2188 |
| Near-logit rank recency seed0 | .7109 | .7656 | .5703 |
| Near-logit rank recency seed1 | .8516 | .6797 | .1484 |
| **Hard decoded exact-ID tie recency** seed0 | .2422 | .3438 | .2734 |
| Hard exact-ID tie recency seed1 | .1875 | .2422 | .1719 |
| RoPE seed0 | .0000 | .0000 | **.9922** |

Each cell 128 examples; seeds are **training seeds**, not separate datasets. BLN fails even without name collision, so the prior 0% at eight overwrites **cannot be attributed just to identity ambiguity**. The near-score rank's improvement is partly general positional extrapolation. Hard ID decoding adds a non-differentiable embedding nearest-neighbor oracle and still fails badly; do not call it a successful exact-match-only solution. The rank mechanism is **not scope-aware**. RoPE's advantage on global-after-local lexical scope reveals a different failure than copying snapshots.

Explicit counterexample: `x=20; def f(): x=21; return x; y=f(); z=x` ⇒ Python `z=20`, whereas blindly selecting the newest textual `x` would use 21. A semantics-correct lookup must select the latest **visible in the applicable lexical scope**, and for dynamic execution must respect runtime/call time.

### C. Official Sanford k7 requires ~20–24 repeated-key candidates per hop
Using exactly [pinned author `InductionHopsFullSequenceTask`](https://github.com/chsanford/hop-induction-heads), 4,096 valid k7 terminal queries, [collision profile run 44050026110](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44050026110), [verified JSON](sanford_collision_verified.json): every first hop has >1 previous identical character; first hop has **24.171** candidate keys on average, seventh hop **19.553**. Choosing randomly among first-hop matching candidates hits the **correct predecessor occurrence** with probability **.0421**, not `1/4`. `1/4` refers to guessing an output character from four symbols; incorrect predecessor positions may still produce the correct output char by collision.

### D. Decisive causal head-level measurement (without training)
In [pointer-index run 44051026021](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44051026021), [raw JSON](sanford_pointer_first_verified.json), 256 official k7 terminal queries and 2 initialization seeds:
- BLN head0's attention **top1 picks the correct newest predecessor value position in 0/256 per seed**; correct location has **~4.2% attention mass**, essentially uniform over 24 same-name keys.
- A near-logit rank prior or a discrete-ID-only tied rank prior changes initialized head0 top1 to **256/256 per seed** and ~**95.0% mass** at the correct value position.
- This is an **architectural prior** and is only **first hop**, not multihop success.

### E. Correct first hop + oracle binary gates still does not give k7
In [Q-state trace run 44051590165](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44051590165), [verified JSON](sanford_qstate_7hop_verified.json), 128 terminal author samples, manual gate logits `[+8,+8,+8,-8]`, and two random initialized models per variant. Decode the Q-stream after layers as nearest token embedding and compare to author true hop1 / hop3 / hop7 **characters**.

| Architecture, doubled no-backtrack=3 | after 1 hop | after 3 hops | after 7 hops |
|---|---:|---:|---:|
| BLN no recency | .414–.461 | .297–.344 | .297–.313 |
| BLN + near-logit rank | **1.000** | .406–.469 | .250–.305 |
| BLN + hard exact-ID recency | **1.000** | .438–.492 | .266–.305 |

Even when nearest-match hop1 succeeds **and** gate bits are correct, downstream Q states are at roughly four-class chance by hop7. This is a random-init diagnostic, not an impossibility theorem for learned networks. A serious candidate mechanism is lost **occurrence index**: a token-value embedding `a` alone is not enough to decide which preceding `a` is correct at the *current pointed location*.

### F. Trained terminal Sanford recency ablation confirms lack of easy fix
[Run 44050026110](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44050026110), [results JSON](sanford_recency_3000_verified.json), AdamW 3,000 steps batch32 fixed k7 only final token, 256 validation:
- BLN original seed0 .3203.
- BLN near-logit recency seeds0/1 .3086/.3242.
- BLN hard exact-ID recency seeds0/1 .3320/.3086.
- RoPE control seed0 .3398.
All are close to four-way label chance .25; merely fixing first-hop tie selection fails to solve seven-hop propagation. The 1-hop 1-layer variant at 1,500 steps is .598 (none), .547 (rank), .621 (exact) for seed0; these are **final-char** scores, not exact pointer-index performance.

### Updated decision gates
1. **Before more Sanford training:** test an actual **position/occurrence-carrying Q stream**, where the next attention lookup masks against the *pointed occurrence*, not just original query token position. Consider versioned keys with lexical scope and definition time. Compare to cheap hard binary-lifting/position-aware indexing oracle and trained deeper one-hop baselines; known algorithmic pointer jumping itself is not novelty.
2. **Before claiming semantic generalization:** train across assignment counts, not just templates up to four writes; then test OOD length, direct global-after-function scope, multiple function calls, and actual Python source representation. A recency rank is not enough.
3. **Before finalizing Sanford comparison:** require all 20k-batch48-five-seed jobs of [run 44046440163](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44046440163) successfully completed. No claims from ongoing jobs.

