# Paper 05 — Reviewer’s six decisive follow-up questions (2026-11-17)

**Branch:** `research/05-pointer-doubling-audit` · **Draft PR:** [#10](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/10) · **Source:** paper 05 `pdg.py` adapted in independent generated audit files; paper and main unchanged.

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
