# Research memo 03 (2026-11-15): Read–write interference in linear attention

Status: **research discovery / not a publishable method yet**. Independent audit, not the original author's claims. Research code lives on branch `research/03-oracle-audit-20261115`, draft PR #1. Keep main unchanged until reviewed.

## First-principles question

The current thesis is not "build a bigger gate": does an already-retrieved answer make a subsequent recurrent-state write causally harmful, and can a low-cost causal proxy suppress harmful writes *without losing necessary new associations*?

For a simple state update `S_t = a_t S_(t-1) + beta_t v_t k_t^T`, the immediate effect of write t on the read at an old query q_j is `beta_t v_t(k_t^T q_j)`, irrespective of whether v_t happened to be predicted at q_(t-1). Previous-read/value similarity and interference are related hypotheses, NOT identical quantities.

## Evidence ledger (fact vs interpretation)

**A. Original 03 paper/results (single-seed, selected LR, width32, synthetic teacher-forced MQAR):**
- g loss 0.379 -> 0.148 and simplified e loss 0.427 -> 0.125 in S1 after adding predicted-skip.
- `e` is a scalar learned-erase gate variant, *not* official full GDN-2 (channel-wise erase, write, and decay). There is no official-GDN2 baseline yet.
- For delta family, `--orq 0` originally suppresses the write `beta` but not `er`: an **erase-without-write** intervention, not a skip-update intervention. Available `results.jsonl` logged this delta oracle at loss 4.244, not the Table 1 0.173.

**B. New retrained oracle audit (GitHub Actions run 43799414398; two seeds, same lr 0.04):**

| train arm | seed 0 loss | seed 1 loss |
| --- | ---: | ---: |
| g baseline | 0.562 | 0.501 |
| g + predicted skip | 0.144 | 0.357 |
| D baseline | 0.766 | 0.729 |
| D write-only oracle | 1.941 | 1.369 |
| D whole-update oracle | 0.444 | 0.753 |
| D + predicted skip | 0.732 | 0.188 |

**Interpretation:** retrained write-only oracle is invalid as a test of skip-update; fully skipped oracle is unstable across these 2 seeds. Cannot infer the causal interference mechanism from retraining experiments.

**C. Frozen-checkpoint causal audit (GitHub Actions run 43800060886; five paired seeds, 300 steps):**

| intervention at validation only | g mean answer loss | D mean answer loss |
| --- | ---: | ---: |
| no intervention | 0.57158 | 0.88846 |
| disable only answer write | 0.28693 | 0.50212 |
| disable full answer update | 0.28693 | 0.45902 |
| disable query-key update | 0.57167 | 0.88560 |
| disable entire query phase | ~0.2846 | ~0.4561 |

- Every one of the five trained models per architecture gains when answer updates are disabled, and query-key suppression has nearly no effect.
- Mean paired loss drop for g = 0.28465, for D = 0.42944 (rough paired t 95% intervals under normal-seed assumption 0.247–0.323 and 0.367–0.492).
- For g and D, answer update suppression accounts for ~99% of the total gap between no intervention and suppressing every query-phase update.
- This identifies **answer-position update interference in the specific trained MQAR teacher-forced setup**. It does NOT establish that the previous-read cosine reliably detects harmful writes, nor that the effect transfers to pretrained language models.

**D. Content-specificity falsification:** five-seed fixed-LR study currently in GitHub Actions (script `audit_content.py`), comparing learned previous-read cosine gate, cross-example shuffled match, learned scalar blanket write attenuation, fixed gamma cosine, current-key read. Pre-registered criterion: previous-read content gate beats shuffled AND uniform suppression in at least 4/5 paired seeds. A negative outcome stops the specificity claim, without modifying criteria after observing results.

## Known prior research (must compare)

- Hatamizadeh et al. (2026), Gated DeltaNet-2, arXiv:2605.22791; official code `NVlabs/GatedDeltaNet-2`, independent channel-wise erase/write/decay.
- Qiao (2025/2026), Attention as Conditioning, arXiv:2508.08289; PH-attention and conditioning analogies / feature-dependent associability.
- Lee et al. (2026), How Linear Attention Remembers, arXiv:2609.33093; pretrained GLA/GDN causal interference and overlapping writes.
- Gangavarapu et al. (2026), Muir, arXiv:2611.08376; readout interference and stronger trained retrieval.
- Li et al. (2026), Erase-then-Delta Attention, arXiv:2606.26560.

**Novelty must be a constrained incremental claim**. A fresh statement that "overlapping writes cause interference" would largely repeat the prior literature.

## Next decisions and stop conditions

1. **Before proposing a new gate:** randomized / interleaved key-value writes and queries, counterexample with overwrites and aliases, true autoregressive exposure, and content-shuffled plus token-only learned write controls, on at least 5 paired seeds.
2. **Before any top-conference architecture claim:** compare true official GDN-2 and PH or closest mathematically matched implementations, equal token budget and matched training setup, plus a full softmax attention reference. Log parameter counts, wall-clock, and train/eval setup.
3. **Deployment correctness:** arbitrary compute chunk size must not change recurrent semantics; evaluate frozen checkpoints at C=1/8/16/32 and compare train chunking to incremental decode. If differences matter, fix and reevaluate before claiming free reuse of chunked output.
4. **Go/no-go (preliminary, fix thresholds before seeing future results):** do not pursue predicted-skip as standalone ICML method if strong baseline closes the gain, if previous-read matching fails to beat matched no-content controls reproducibly, or if gains rely on a strictly predictable key-value phase not found in natural workloads.
5. **If method fails but mechanism stands:** reframe as a mechanistic diagnostic of self-generated answer contamination, with explicit novelty comparison against arXiv:2609.33093.
6. **Professor meeting gate:** present verified five-seed causal effect, a falsified control, proper prior-work caveat, and one crisp prospective experimental question. The finding merits *discussion*, not yet full project validation.

Evidence is separated here as: **observations** (numbers and code), **mechanism interpretations** (testable causal hypotheses), and **extrapolation** (unproven until larger models and real tasks).

## Two algebraic counterexamples against equating prediction match with write redundancy

1. **False positive (necessary new key):** let `S = v e1^T`, previous query `q_prev = e1`, next write `k_t = e2` and `v_t = v`. Then `S q_prev = v`, cosine = 1, so predicted-skip attenuates the update; but `e2 -> v` is an entirely new association, required for future query `q=e2`. Same value does not mean redundant key-value association.
2. **False negative (harm to old key):** let `S = v1 e1^T + v2 e2^T`, with orthogonal v1 and v2. Previous query `q_prev = e2` reads `v2`, next write `k_t = e1`, `v_t = -v1`. The similarity is zero despite the new write directly interfering with the valid existing association `e1 -> v1` (if it remains valid).

These establish a **formal insufficiency** of previous-read/value cosine as a general interference indicator; they do not prove how often the failures occur in trained models. Construct alias/overwrite counterexamples before claiming an architecture contribution.

## E. Content-specificity results (CI #43800664340, success; 5 seeds x 6 arms, 30 train runs, lr=.04)

| g variant | mean answer loss | mean accuracy |
| --- | ---: | ---: |
| no extra gate | 0.568326 | 0.863379 |
| predicted-skip `S_c q_(t-1)` | 0.319877 | 0.907690 |
| shuffled value control `--lpsc1` | 0.532858 | 0.868579 |
| learned blanket suppression `--lpsU1` | 0.596687 | 0.856079 |
| fixed cosine gamma=3 | 0.620375 | 0.828003 |
| current-key read `S_c k_t` | 0.267237 | 0.919604 |

- Predicted-skip improves over baseline in 5/5 seeds, over shuffled 5/5, over blanket suppression 4/5. This meets **the preregistered exploratory content-specificity gate** (>=4/5 both).
- Mean paired loss gap (shuffled minus predicted-skip) = 0.21298; rough 95% paired t CI = [0.1185, 0.3074]. Gaps are positive in every seed.
- Crucial counterexample to uniqueness: **current-key read beats previous-read gate in 4/5 seeds** and has lower mean loss, but variance is larger and the 5-seed paired difference is not decisive. Thus there is no evidence previous-read is a uniquely superior *signal*; its potential advantage, if any, may be computational reuse in a specific chunked kernel, which is currently unvalidated.
- **Caveat** shuffled and blanket suppression are not yet matched on total write budget / average gate strength. Matching the strength distribution is required before concluding that per-token content matching is itself the cause rather than a phase-selective suppression distribution.
- Fixed gamma 3 wins over learned previous-read in 2/5 seeds but fails badly in others; do not claim learned always beats fixed for every initialization.
- This validates a narrow synthetic observation, not a conference-level method. The outstanding critical falsifiers are official GDN-2, mixed writes/queries and prefill/decode consistency. Avoid further gate design until those are tested.

References: [run B frozen causal audit](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43800060886), [run E content controls](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43800664340).

## F. Critical architecture falsifier: chunk-size semantics (CI #43801785143, success)

Frozen-checkpoint study, trained at chunk C=16, 3 independent seeds for g and D, each with or without predicted-skip. Evaluation on the same 128 held-out sequences at C∈{1,2,4,8,16,32,64}; logits compared on the same fixed 16 sequences.

| arm | C=1 loss | C=16 loss | C=32 loss | C=64 loss |
| --- | ---: | ---: | ---: | ---: |
| g baseline | 0.5860 | 0.5860 | 0.5860 | 0.5860 |
| g + predicted-skip | 0.2633 | 0.2020 | 0.1701 | 0.7419 |
| D baseline | 0.9793 | 0.9793 | 0.9793 | 0.9793 |
| D + predicted-skip | 0.6202 | 0.4721 | 0.4617 | 1.3415 |

Baselines' max per-logit differences vs C16 at any C were <2.5e-5. Gated models had some logits change by 7–16 absolute units even at C=1/32, far beyond numerical error.

At sequence-length C=64, all within-chunk gates see the zero *chunk start* state, hence gate multiplier=1 identically and previous learned suppression disappears. This is a structural consequence of `r=S_chunkstart q_prev`. The gate remains useful at C=1 (better loss than the baseline), so these data falsify **chunk-independent recurrent semantics / zero-cost arbitrary-kernel reuse**, not the existence of a content-suppression benefit.

**Decision:** DO NOT treat the present predicted-skip formula as a ready-to-ship architecture. A new method must have one well-defined recurrent transition with the same outputs under prefill at any C and tokenwise decoding (subject to FP error), or explicitly define a fixed chunk-dependent architecture and analyze its online serving constraints and wall-time cost. A new Gate is NOT the next step; first implement the exact semantics or accept the stop criterion.

Raw artifact/logs: [chunk audit run #43801785143](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43801785143).

## Research project go/no-go after chunk falsifier

**Proven small insight suitable for a professor discussion:** in synthetic teacher-forced MQAR, frozen-checkpoint removal of answer-position state updates improves answer retrieval consistently in five trained seeds for each of g and D; query-key suppression is nearly inert; content-aware suppression outperforms shuffled/uniform controls in an initial fixed-LR five-seed test.

**Unproven mechanism:** whether a prior read/value match is optimal; the current-key read outperformed previous-read in 4/5 matched seeds, though not decisively overall. Also not established whether equal write-budget matching explains any gains.

**Failed deployment claim:** computational chunk partition is not only a performance knob for this Gate. No justification remains for unqualified cheap reuse of existing chunk kernel or straightforward incremental decoding.

**Novelty warning:** existing pretrained causal interference work arXiv:2609.33093 and official GDN-2 arXiv:2605.22791 sharply limit novelty of a generic "answer writes interfere" conclusion.

**Meeting recommendation:** present the small mechanism result + negative chunk result together, and propose a bounded next stage: (1) true GDN-2 / PH-style strong baseline; (2) randomized, mixed retrieval/update stream including overwrites and key aliasing; (3) clearly specified chunk-independent online update with measured overhead. Stop standalone method development if (1) closes the gap or (2) exposes failure to retain necessary new associations; do not relabel that as successful novelty.
