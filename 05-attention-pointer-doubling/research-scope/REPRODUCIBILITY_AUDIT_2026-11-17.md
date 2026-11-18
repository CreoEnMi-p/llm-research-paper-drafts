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

## Open uncertainty and strict CPU experiment
Package installations in replay use unpinned `torch` from CPU wheel index and `numpy` from PyPI; hardware/microkernel dispatch may differ across runner instances. The source has `torch.set_num_threads(2)`, with PyTorch's CPU kernels choosing implementations at runtime. A plausible explanation is small cross-environment floating-point differences amplified by training optimization; **this is not yet proven to be the cause**.

A separate (do not overwrite original preregistered model code) generated strict-CPU variant pins torch2.15.0, numpy2.4.6, single-thread math and disables MKLDNN and nondeterministic algorithms. [Nine independent-runner replication cells](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44236159785) compare exact per-pair SHA-256 across 3 identical replicas for each selected model/seed. Only refer to that run as complete after its aggregate job passes.

## Controls and limitations
The V2 A+C test is **already consumed**. Re-running identical fixed models on the same test validates reproducibility, but it creates no new fresh generalization evidence. Method changes made after examining V2 must be evaluated on a genuinely new preregistered V3, rather than selecting the best of multiple replays. With only four production clusters, large bootstrap intervals remain scientifically appropriate.
