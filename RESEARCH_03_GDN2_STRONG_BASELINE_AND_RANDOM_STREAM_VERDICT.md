# Paper 03 — official GDN-2 baseline feasibility and learnability-gated random streams

Date: 2026-11-15 PST. **Research-lead decision: DO NOT run any gate/seed matrix on this task.** This follows the user's explicit rejection of unlearned baselines. Every numeric claim below comes from the linked successful GitHub Actions job logs, not from a method paper draft.

## 1. Official Gated DeltaNet-2: actual evidence and hard environment requirement

Official NVIDIA upstream: https://github.com/NVlabs/GatedDeltaNet-2 ; paper arXiv:2605.22791; official docs https://research.nvidia.com/index.php/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention .

- The actual official trainable implementation includes **independent per-key-channel erase `b`**, **per-value-channel write `w`**, **per-key-channel decay `g`**, and GPU Triton recurrent/chunked algorithms, plus a **beta contrast safeguard**. This is not the scalar `e` mixer or the earlier incomplete CPU `Z` port in our repo.
- Official repository assumes PyTorch 2.9/CUDA 12.8 with compiled Flash Attention and Flash Linear Attention dependencies (upstream `requirements.txt`). Current chat container has `torch ... +cpu` with `torch.cuda.is_available() == False` and no `nvidia-smi`. GitHub Actions standard `ubuntu-latest` is a CPU runner; GitHub's larger GPU runners require an eligible provisioned organization/enterprise runner. **No authorized, tested CUDA GPU runner is currently accessible in this workflow.**
- Official published GDN-2 1.3B / 100B token comparisons already show a real strong retrieval baseline. Upstream README reports recurrent RULER S-NIAH-3 (2K): Gated DeltaNet **54.2**, GDN-2 **89.8**, and MK-NIAH-1 (4K): GDN **27.8**, GDN-2 **37.8**. These are **official reported metrics on a different benchmark**, not an experiment we ran or a score on our random stream.
- **Outcome:** GDN-2 is now concretely documented as a required external official strong method, but **an official same-data head-to-head was NOT RUN** because its GPU dependency is unmet. Do not spend further CPU cycles on an undertrained pseudo-GDN-2 and do not cite our earlier low-scoring `Z` result as evidence against NVIDIA.

## 2. Proper randomized event ordering and enforced competence gate

A new dataset `mqar_stream` has event records `WRITE key value` and `READ key answer` with disjoint marker/key/value IDs. Following initial facts, each sequence **independently randomly permutes** later read/new-write/overwrite event types. No fixed second query phase, no dedicated new-write phase. Queries target the **latest visible** key value. New keys can deliberately reuse old values. State replay independently verifies every label and mask; the model cannot see future answers, and 96-row fresh validation generates its own previous read answers left-to-right, while external fact writes remain true.

Semantic/determinism unit tests **all passed before either training run**. Training uses fixed seeded randomly generated examples, teacher-forced input, and CE only at query-key positions. The single D baseline has independent erase-on-key via Delta Rule (not official GDN-2). For baseline competence, precommitted success required overall TF accuracy >=82% and every category (untouched/overwritten/new) >=75%, with >=100 examples per subgroup, for **two consecutive fixed-validation checks 100 steps apart**. If it had failed at the capped steps, no gate/multiseed comparisons were permitted.

### A. Initial random stream (4 old keys + 2 new; 6 total late writes / 22 late reads)

[Actual successful Action run 43832170838](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43832170838). Ordinary Delta Rule `D`, width64, 2 layers/4heads, batch24, seed0, max1800 steps; stopped at step300 after consecutive passes at 200/300.

| Checkpoint | TF all | TF old untouched | TF after overwrite | TF new key |
|---|---:|---:|---:|---:|
| step100 | 79.52% | 71.84% | 84.65% | 82.38% |
| step200 | 96.58% | 94.55% | 97.58% | 98.11% |
| step300 | 98.02% | 97.42% | 98.42% | 98.24% |

Fresh **self-generated-answer AR** validation (seed92742; 96 rows, 2112 scored queries): all **96.45%**, untouched **94.27%**, overwritten **97.63%**, new keys **97.71%**, stale previous-value answer rate 0.00%; 105 of 192 newly written facts intentionally aliased an old value.

### B. Larger-memory random stream (8 old keys + 4 new; 10 late writes / 30 late reads)

[Actual successful Action run 43832622685](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43832622685). Ordinary Delta Rule `D`, **same width64/2 layers/4 heads**, batch24, seed0, max1600 steps; 6 deterministic/preflight tests passed. Again stopped step300 after success at 200/300.

| Checkpoint | TF all | TF old untouched | TF after overwrite | TF new key |
|---|---:|---:|---:|---:|
| step100 | 56.77% | 43.47% | 64.11% | 65.52% |
| step200 | 94.01% | 89.07% | 96.22% | 98.10% |
| step300 | 96.19% | 93.56% | 97.18% | 98.65% |

Fresh **self-generated-answer AR** validation (seed92742; 96 rows, 2880 scored queries): all **95.83%**, untouched **92.07%**, overwritten **97.24%**, new keys **98.98%**, stale previous-value answer rate 1.33%; 200 of 384 newly written facts alias an earlier value.

## 3. Interpretation and stop decision

**Observed fact:** A competent classical **Delta Rule** solves these two randomly interleaved, explicitly event-marked memory update tasks at >95% AR accuracy after only 300 train steps, including successful overwrites and same-value new-key aliasing. Earlier phase-structured GLA difficulties DO NOT automatically imply a general unsolved architecture problem.

**Mechanism hypothesis (not proved):** Delta erase/write dynamics protect against the exact answer-contamination interference that hurts plain scalar gated-linear mixing. The read/write event markers additionally reveal which tokens are answers or external facts, making simple grammar-aware suppression possible; this task is *not unstructured natural language*. No official GDN-2 head-to-head has been run here.

**Research/architecture judgment:** Both variants are at or near saturation even with D. Pursuing a previous-read+margin Gate matrix or extra seeds here would have little discriminating power, and would violate the counterexample-first standard. **No such computations were started.**

### Next experiment only if official GPU or a genuinely novel failure case is available

- Use NVIDIA's official GDN-2 implementation on a verified CUDA/Triton GPU, with official chunk vs recurrent correctness checks and a trainable matched official recipe. First establish official GDN-2 learns all test subgroups; **never** compare to a method that cannot learn basic facts.
- Any proposed harder setting must be designed **before** looking at outcomes, explicitly target a novel failure mode *not already handled by GDN-2/Delta*, and pass strong baseline competence. Test read-source current-key and input-role-only controls with matched training budget and actual wall-clock time.
- If no GPU is available, **stop this architecture branch** at the proven frozen-state causal interference result + negative original Gate chunk dependence; offer those bounded results to the professor. No further pseudo-GDN-2 training or synthetic Gate patching by default.

All code is on isolated draft research branches, not `main`.
