# Paper 03: honest GDN-2 equation-level reference control (2026-11-15)

**Status: correctness smoke test passed; this reference is NOT a validated official Gated DeltaNet-2 baseline.** [All six Actions jobs](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43826454050) succeeded and the underlying [implementation is in PR #7](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/7), but mathematical recurrence correctness and optimization adequacy are distinct questions.

## What was actually implemented
- Own CPU PyTorch implementation of the public recurrence `S <- diag(alpha) S; delta_v <- w⊙v - (b⊙k)^T S; S <- S + k delta_v^T` (transposed internally to this toy bed's [value,key] layout). Dynamic per-key-channel decay; separate per-key-channel erase b and per-value-channel write w.
- No GDN-2 upstream Triton fast kernels, conditional beta-contrast guard, full official output gated norm or projection/init recipe, grouped heads, or GPU throughput benchmark. Upstream code: https://github.com/NVlabs/GatedDeltaNet-2 ; paper: https://arxiv.org/abs/2605.22791 . Do not name the result 'official GDN-2'.
- Same toy block width32, two layers/two heads, 650 training steps, batch24, T96,P12, 3 matched seed indices, independent heldout 92741; same overwrite+alias data as PR #5/6. Total trainable parameters **47,596** vs **41,256** for the ordinary gated-linear/delta toy model and **41,260** for the margin Gate model (+15.4%). Matched overall architecture except core recurrence and these control parameters.
- Unit tests check zero-write, distinct erase/write overwrite behavior, and nonzero finite gradients across erase/write/decay/rate projections. These do **not** establish full fidelity to NVIDIA GPU code.

## Autoregressive Q1 / Q2 new / Q2 old results (three seeds)

| task | train seed | Q1 | Q2 updated/new | Q2 untouched/old | CPU train time | loss |
|---|---:|---:|---:|---:|---:|---:|
| mqar_rewrite | 0 | 87.15% | 79.46% | 66.80% | 96.4s | 0.477 |
| mqar_rewrite | 1 | 89.51% | 84.08% | 70.96% | 65.3s | 0.382 |
| mqar_rewrite | 2 | 88.54% | 85.71% | 72.14% | 67.8s | 0.408 |
| **mqar_rewrite average** | 3 seeds | **88.40%** | **83.09%** | **69.97%** | 76.5s | — |
| mqar_alias | 0 | 90.14% | 70.68% | 83.20% | 51.5s | 0.545 |
| mqar_alias | 1 | 67.71% | 45.68% | 53.78% | 64.1s | 1.211 |
| mqar_alias | 2 | 42.43% | 23.96% | 37.63% | 69.0s | 2.131 |
| **mqar_alias average** | 3 seeds | **66.76%** | **46.78%** | **58.20%** | 61.5s | — |

## Validity audit and decision

- **Alias fails the adequacy bar:** the independent GDN-2 equation-level reference does not reliably learn even the initial keys/values: in seeds 1 and 2, Q1 autoregressive accuracy is only ~67.7% and 42.4%. Those outcomes are **undertraining / instability**, NOT evidence that published Gated DeltaNet-2 is worse than a small custom Gate.
- Rewrite is learned to a useful but not dominant degree (averages above); however performance may be highly sensitive to omitted official safeguards and initialization/training recipe. Total parameter count is larger, and CPU wall time is not identically matched to the previous gated comparisons.
- An ill-performing *approximate* implementation cannot be cited as a strong published baseline. **GDN-2 head-to-head is STILL NOT DONE.**
- We will **not chase optimizer/margin settings on this heldout split** to manufacture a win. The scientifically justified next steps are official GDN-2 on GPU with a matched training recipe, or a verified reference reproduction including relevant stabilizing components and an independent validation split.
- Separately, the three-seed positive result for prior-read or current-key margin Gate on phase-structured toy tasks remains valid as an observation; it is neither a proof of general interference prediction nor an ICML-ready method.

**Go/no-go remains unchanged:** preserve the causal interference discovery and combined old/new retrieval tradeoff, but defer standalone architecture development until an actual strong baseline and phase-only control are passed.
