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
