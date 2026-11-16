# Paper 03: independent necessary-write counterexamples and compute-matched controls (2026-11-15)

**Research status: exploratory synthetic finding; not a method validated for ICML.** Independent from the Gemini proposal patch. All 12 GitHub Actions jobs completed with success, 36 training records (24 primary + 12 stronger controls). Nothing merged to main.

Sources: [3-seed two-task counterexample run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43824638167) and [3-seed two-task compute controls](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43825204609), reviewed as CI job logs, plus [PR #5](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/5) and [PR #6](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/6).

## Protocol (specified before results)
- N=12 distinct initial keys, W=6 facts in second write block, T=96, two query phases Q1=15, Q2=15. Q2 queries are balanced between changed and unchanged categories (7 and 8 per sequence), with randomized order.
- **Rewrite:** existing keys are assigned guaranteed different values. Score retrieval of the *newest* value, and stale old-value predictions.
- **Alias:** newly added keys get values already present under some old key. Equal values are NOT redundant keys.
- Train width32, 2 recurrent mixer layers, 2 heads, B24, 650 steps, LR (Muon .05 / Adam .04), 3 paired train/initialization seeds 0/1/2, heldout 96 fresh rows generated with seed 92741.
- Gate is the **self-consistent sequential exact reference**, margin fixed to 0.2 before these tasks. Comparison: gated-linear (g), g+previous-read gate, g+current-key gate, and ungated Delta Rule (D). The two gated runs differ only by read source.
- At test, greedy **answer-only autoregressive rollout** replaces earlier query answer tokens with predictions. All externally supplied fact values, keys, and phase positions remain correct; no future targets enter any prefix model.
- Compute controls use g trained 2500 steps and D trained 2000 steps, deliberately adding tokens to roughly match the gated 650-step CPU wall time. Actual wall times recorded; matching is approximate across runners.
- Generator semantic/determinism unit tests passed on all jobs. Per-task subgroup supports across all 3 seeds: 96×7=672 changed and 96×8=768 old queries per seed. These are clustered sequences, not 672 independent training seeds.

## Main outcomes: self-generated answer accuracy (3 training seeds)

### Old-key overwrite

| arm | Q1 initial retrieval | Q2 changed/new keys | Q2 untouched/old keys | mean training CPU time | wrong stale old value rate |
|---|---:|---:|---:|---:|---:|
| g_base | 85.19% | 51.79% | 76.61% | 21.6s | 32.64% |
| g_prev_margin | 97.62% | 90.03% | 88.45% | 88.1s | 7.69% |
| g_key_margin | 97.11% | 89.24% | 86.55% | 86.9s | 7.34% |
| D_delta | 89.86% | 89.63% | 76.39% | 24.8s | 3.42% |
| g_base_2500 | 91.99% | 71.53% | 82.77% | 85.0s | 17.16% |
| D_delta_2000 | 92.13% | 89.78% | 79.08% | 76.8s | 4.32% |

Per-seed Q2 changed/new correctness (rows are paired seed 0, 1, 2):

| seed | g (650) | prev read (650) | current key (650) | D (650) | g (2500) | D (2000) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 53.1% | 89.9% | 90.6% | 92.4% | 74.3% | 95.5% |
| 1 | 47.9% | 88.4% | 86.3% | 85.6% | 74.1% | 92.6% |
| 2 | 54.3% | 91.8% | 90.8% | 90.9% | 66.2% | 81.3% |

### Same-value new-key alias

| arm | Q1 initial retrieval | Q2 changed/new keys | Q2 untouched/old keys | mean training CPU time | extra |
|---|---:|---:|---:|---:|---:|
| g_base | 90.76% | 76.64% | 83.38% | 22.2s | n/a |
| g_prev_margin | 96.74% | 92.51% | 90.76% | 90.4s | n/a |
| g_key_margin | 97.85% | 93.45% | 92.62% | 91.4s | n/a |
| D_delta | 91.23% | 85.86% | 71.05% | 26.6s | n/a |
| g_base_2500 | 94.31% | 88.54% | 89.93% | 90.9s | n/a |
| D_delta_2000 | 92.06% | 90.72% | 75.91% | 84.8s | n/a |

Per-seed Q2 changed/new correctness (rows are paired seed 0, 1, 2):

| seed | g (650) | prev read (650) | current key (650) | D (650) | g (2500) | D (2000) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 77.1% | 91.7% | 94.2% | 87.5% | 93.8% | 91.5% |
| 1 | 74.1% | 95.4% | 91.2% | 88.1% | 84.7% | 93.0% |
| 2 | 78.7% | 90.5% | 94.9% | 82.0% | 87.2% | 87.6% |

## Precommitted validity and interpretation

- **Original g 650-step overwrite baseline did NOT learn new values adequately:** AR accuracy on overwritten keys 51.79%, well below 70% in every seed. Therefore its 38pp deficit to gated methods is *not* a valid measure of unique architectural improvement.
- **Stronger g 2500-step overwrite control passes the preregistered minimal learnability bar in two of three seeds** (changed key >=70%, untouched >=70%) but still trails previous-read on changed and old-key subgroups. This is an *exploratory positive signal under approximately comparable CPU time*, not full baseline saturation or novelty proof.
- **Delta Rule matters:** it handles overwrites well without previous-read matching. At 2000 steps, its mean overwritten-key accuracy approximately equals previous-read, but its untouched-old-key accuracy remains lower. There is a potential **write/edit versus retention tradeoff**, not a demonstration that delta-rule models cannot solve it.
- **Same-value alias is NOT a learned-model counterexample in this phase-structured generator:** the exact gate handles necessary new-key writes well at 650 steps. The formal logical counterexample from the prior memo still exists; this task does not force the previous q to be precisely a distinct old key when the new value is written. Never report it as proof that theoretical false positives do not exist.
- **Previous-read signal is not uniquely justified:** current-key matching performs similarly or better in multiple paired seeds. A useful content-suppression phenomenon is separable from the particular proposed gate.
- **Crucial unanswered baseline:** neither D nor scalar learned-erase e is NVIDIA's official Gated DeltaNet-2; that method has *channel-wise independent erase/write and decay*, and official accelerated implementations also contain a safety guard. A fair official GDN-2 comparison is required before any top-conference architecture claim. We did not run it.
- **Other caveats:** fixed global stage ordering (not random interleaving), teacher-forced training with answer-only autoregressive evaluation, synthetic token vocabulary, only 3 paired seeds, frozen margin 0.2 selected from earlier independent tests, and sequential gate reference with much greater per-token CPU compute. No full learned phase-only control, no high-efficiency chunked gate kernel, and no real-world language evaluation.

## Decision gate: what the evidence permits

**Do not discard the science problem**: sequential answer contamination and necessary-write/retention tension are reproducible in these toy models. Stronger compute-matched simple baselines do not erase all differences.

**Do not advance the proposed previous-read Gate as a top-conference method yet.** Read source is non-unique; official GDN2 has not been tested; no parallel kernel or deployment efficiency; the stage structure provides easy task-role shortcuts. The next honest discriminating tests are:
1. Official GDN2 (or explicitly labeled mathematically faithful recurrent reference with verified channel-wise gates, beta safeguard, equal compute/parameters) against *both* prior-read and current-key gates, across rewrite and alias, to see whether the extra mechanism is redundant;
2. learned token-/phase-only content-free suppression, matched update distribution/compute (not an oracle at inference), to test whether phase recognition alone explains this gain;
3. randomized mixed updates/queries and generalization across sequence length and key count, but only after stronger baselines pass.

**Go/no-go:** if official GDN2 or a phase-only learned control matches the gain and efficiency, close the standalone method attempt and retain the causality/diagnostics finding for discussion with the professor. If a stable gap survives, invest in a chunk-independent *efficient* kernel and validate on real workloads before pursuing an ICML submission.
