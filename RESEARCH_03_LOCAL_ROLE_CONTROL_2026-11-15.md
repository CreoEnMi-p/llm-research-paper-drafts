# Paper 03: learned local-only write vs known-answer-role upper bound (2026-11-15)

**Independent scientific falsifier, not another gate modification.** [Corrected CI run, all six jobs successful](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43827765266); the earlier CI run failed only because a test fixture omitted default arguments (no experimental runs were produced). Branch and [Draft PR #8](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/8). The synthetic data generator and prefix role masks passed all unit tests.

## Question and setup
Does the previous-read gate's benefit on retaining old and adding new associations genuinely require reading the recurrent state? Or can a cheaper per-token learned gate infer answer-token roles? Keep all seeds, token length and heldout untouched.

- 3 paired seeds × 2 tasks (old key rewrite; same-value new-key alias) × 3 arms = 18 runs, width32, two layers, P12, T96, batch24, heldout 96 fresh rows generated from seed 92741, fixed Muon LR .05 + Adam LR .04.
- `local_input_gate_650`: learned token-hidden-state sigmoid added to base beta, **no explicit recurrent-state lookup** at that layer. This receives 132 additional learned parameters total versus the previous-read gate's 4; no claim about layer-wise h being free of contextual information.
- `local_input_gate_2500`: same control but ~4× steps to approximate **wall-time** of exact previous-read reference (650 steps). Wall-clock varied by CI runner; record actual seconds.
- `phase_answer_oracle_650`: multiplies beta=0 only at **known answer-token positions** in both query phases; never suppresses necessary second-stage factual writes, and uses configured full positions under prefix-only AR calls. **Task-role labels are externally provided — this is NOT a deployable learned model, only a mechanism upper bound.**
- Q1/Q2 teacher-forced training, but self-generated prior query answers at evaluation; externally supplied new facts remain ground truth. Evaluated on exactly the same split as [36-run main memory-update memo](https://github.com/CreoEnMi-p/llm-research-paper-drafts/blob/research/03-update-compute-matched-20261115/RESEARCH_03_MEMORY_UPDATE_COUNTEREXAMPLES_2026-11-15.md).

## Complete three-seed autoregressive results

### Overwrite: required revised association + untouched old memory

| model | mean Q1 accuracy | mean Q2 changed/new | mean Q2 untouched/old | mean train CPU time | trainable params |
|---|---:|---:|---:|---:|---:|
| local_input_gate_650 | 85.19% | 53.52% | 74.09% | 28.0s | 41,388 |
| local_input_gate_2500 | 91.04% | 73.26% | 82.55% | 108.6s | 41,388 |
| phase_answer_oracle_650 | 97.73% | 91.96% | 90.67% | 27.7s | 41,256 |

Per-seed changed/new-key accuracy (seed 0 / 1 / 2):
- local_input_gate_650: 55.1% / 56.0% / 49.6%
- local_input_gate_2500: 76.2% / 74.6% / 69.0%
- phase_answer_oracle_650: 92.3% / 93.2% / 90.5%

### Alias: new key that reuses an existing value

| model | mean Q1 accuracy | mean Q2 changed/new | mean Q2 untouched/old | mean train CPU time | trainable params |
|---|---:|---:|---:|---:|---:|
| local_input_gate_650 | 77.71% | 61.16% | 68.32% | 25.0s | 41,388 |
| local_input_gate_2500 | 91.62% | 82.09% | 84.07% | 94.9s | 41,388 |
| phase_answer_oracle_650 | 97.43% | 94.64% | 90.58% | 24.5s | 41,256 |

Per-seed changed/new-key accuracy (seed 0 / 1 / 2):
- local_input_gate_650: 55.4% / 47.0% / 81.1%
- local_input_gate_2500: 78.3% / 76.3% / 91.7%
- phase_answer_oracle_650: 97.5% / 90.9% / 95.5%

## Independent verdict and hypotheses

1. **The ordinary learned local-only gate did NOT recover the previous-read gate gains**, even with roughly equal training CPU time. The local-gated model often *underperformed the ungated longer-trained baseline* and in some seeds did not robustly learn even Q1. This refutes the narrow claim that a naive extra sigmoid of current hidden state is an adequate replacement *under this recipe*. It does **not** prove the previous-read signal is necessary: the original base beta is already input-conditioned, the added gate may be redundant or harmful, and other local/role-conditioned architectures can exist.
2. **Role-only answer suppression works as an oracle upper bound.** Because fact updates are preserved, the task-aware no-answer-write intervention achieves high accuracy in all phases without comparing previous-state read to new values. This independently supports the *answer contamination / required-write selectivity* mechanism, but role labels are privileged experimental information. It must never be compared as a deployable baseline.
3. **The previous-read signal is STILL not uniquely justified.** Current-key read, tested with identical margin/steps and autoregressive heldout, was equally competitive, especially in new-key alias. No trained content-free control has yet been matched on the distribution of aggregate write suppression.
4. **Paper novelty remains blocked** by official GDN-2 (and other modern memory editing baselines), random mixed streams, efficient recurrent/prefill chunk equivalence, and genuine language workloads. The [equation-only GDN-2 reference](https://github.com/CreoEnMi-p/llm-research-paper-drafts/blob/research/03-gdn2-equation-control-20261115/RESEARCH_03_GDN2_EQUATION_CONTROL_2026-11-15.md) failed to learn alias adequately; it is NOT a fair negative result against official GDN-2.
5. **Three training seeds remain exploratory.** Fixed phase order is a strong synthetic shortcut; independent rows within the same training seed are not independent model repetitions. No causal conclusion about how individual learned gate factors correlate with genuinely harmful/new writes is justified merely by high accuracy.

## Decision

The useful small result is a well-audited **selective write/forget tension** with a topically appropriate negative chunk-semantics finding, not a newly established production architecture. We should take that to the professor and only proceed on the method if a proper official GDN-2 baseline, input-role-matched controls, and realistic mixed streams expose a reproducible *new* gap. Do not continue adding margins and pseudo-PH knobs without satisfying those falsifiers.
