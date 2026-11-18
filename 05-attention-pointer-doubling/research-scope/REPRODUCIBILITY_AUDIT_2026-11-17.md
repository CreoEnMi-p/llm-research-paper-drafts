# Frozen V2 reproducibility audit — source-identical CI disagreement

**Status:** Verified data and code identities; same-runner deterministic repeatability; hosted-runner across-job discrepancies remain under investigation. Do NOT present a single CI score as a precise universally reproducible experimental fact.

## What was actually replayed
[Complete GitHub Actions run #1](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44233344730) 12/12 success; [full replay run #2](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44234485167) 12/12 success. Each independently generated/validated grammar, checked immutable V2 A+C SHA-256 `e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce`, re-evaluated source-only heuristics against their archived value, trained **5 Transformer + 5 literal-pointer-depth seeds** with locked 1000-step settings, and computed paired metrics and production-family bootstrap.

**Source-code integrity:** Compared GitHub blob SHAs between the two run head commits for `train_composition_protocol.py`, `grammar_compositions.py`, `scope_program_benchmark.py`, `aggregate_new_grammar.py`, `FROZEN_AC_TEST_256pairs.jsonl` and `PROTOCOL_MANIFEST.json`: **all six files bit-for-bit identical**. Config, training seeds and sealed test SHA are the same.

| V2 mean (five training seeds) | Original archived local | CI #1 | CI #2 |
|---|---:|---:|---:|
| Transformer individual accuracy | 35.898% | 35.664% | 36.016% |
| Pointer-depth individual accuracy | 37.891% | 38.164% | 35.703% |
| Transformer pair both correct | 0.156% | 0.156% | 0.156% |
| Pointer-depth pair both correct | 0.078% | 0.078% | 0% |
| Transformer predictions **invariant** within counterfactual pair | 85.703% | 85.625% | 85.703% |
| Pointer-depth predictions invariant within pair | 84.766% | 82.344% | 80.391% |

**Do not average these three runs and claim new independent seeds.** They are **repeats of the same five seeds and fixed test**, not 15 independent experiments. Pointer advantage is unstable, pair success remains effectively nil, and there is no support for compositional scope learning.

## A narrower but important successful control
[Same-runner repeated seeded training](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44234036141), completed 3/3 jobs. Repeated full training sequentially **twice on the same hosted runner** for pointer-depth seeds1,4 and Transformer seed0. Every one of **256 paired test predictions** was exactly equal between those two copies for all three tested combinations. Therefore, local run-to-run noise within an identical process environment was absent for the checked combinations. This does **not** prove bitwise reproducibility across different hardware/torch dispatch paths.

## Confirmed first source of cross-Runner divergence: forward numerical path

**Strict CPU results completed.** [Nine independent-runner controls](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44236159785) pinned `torch==2.15.0+cpu`, `numpy==2.4.6`, single-threaded CPU, `torch.use_deterministic_algorithms(True)` and disabled MKLDNN. The **Transformer seed0** had identical pair-prediction SHA-256 on three independent runners, but **pointer-depth seed1** had three different hashes, with individual accuracy **34.96%, 43.55%, 42.58%**. Pointer-depth seed4 had two matching runs and one different run, with **40.82%, 40.82%, 44.92%**. Strict kernel selection is **not enough** for cross-hardware exact repetition.

**First divergence *localized* to the first forward pass.** A separate [six independent-runner full-step trace](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44236762811) compares CPU model, unmodified *initial parameter bytes*, first training batch/labels, PyTorch RNG state, first forward logits, and weight hashes after optimizer updates. [Complete machine-readable per-step traces](repro/cross_cpu_first_divergence_verified.json).

- For Transformer seed0, all three runners share **the same initial weights SHA `65736736d735cf2ab591`, input SHA `2d2520cafb55cb63fbb4`, and RNG SHA**, yet initial forward logits differ between **AMD EPYC 7763** (`550aa5af4c2aab9d4648`) and two Intel Xeon CPUs (`34974ca3a62382a6676e`). The two different Intel models agree on that hash.
- For pointer-depth seed1, all three share identical initial weights SHA `5d727a28072701230138`, input SHA `5a9dd6c94f1c8ab4ace4`, and RNG SHA. **AMD EPYC 7763** first logits SHA is `0b6cb97724666423176d`, while two **EPYC 9V74** runners agree on `9fe4676c65f9a197391d`.
- This difference exists **BEFORE the first optimizer step**, disproving data-generation RNG, training seed, or optimizer-update ordering as its original cause. It correlates with the hosted CPU architecture / numerical backend. **The exact floating-point instruction or operator has not been isolated**, so the mechanistic statement is *hardware-dependent first-forward numerical output*, not proven "the CPU itself is wrong".
- Small floating differences can plausibly be amplified by optimization, especially in the pointer's numerically sensitive training regime; this amplification is consistent with observed accuracy spread but **we did not isolate the precise amplification path**.

**Reporting rule:** Pin CPU/GPU *model*, runtime wheels, deterministic flags and preprocessing, not just the Python seed, and report **multiple independently trained seeds plus independent compute-environment sensitivity**. Two separate V2 CI replays are not 10 fresh seeds each, nor a legitimate new test. Continue only on a *new* sealed holdout for architecture claims.

## Controls and limitations
The V2 A+C test is **already consumed**. Re-running identical fixed models on the same test validates reproducibility, but it creates no new fresh generalization evidence. Method changes made after examining V2 must be evaluated on a genuinely new preregistered V3, rather than selecting the best of multiple replays. With only four production clusters, large bootstrap intervals remain scientifically appropriate.
