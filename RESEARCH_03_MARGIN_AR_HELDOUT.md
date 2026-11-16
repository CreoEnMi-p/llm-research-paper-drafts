# Research 03: fixed-margin autoregressive confirmation on independent heldout (2026-11-15)

**Status:** Three-seed exploratory *pass* of a **narrow optimization-efficiency gate**, NOT validation of a new top-conference architecture. Parent [AR exposure PR #3](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/3) remains a valid falsifier of **no-margin** exact previous-read gating. Gemini [PR #2](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/2) independently proposed margin .2; do not recast this as a margin discovered from the new heldout.

[CI evidence, three successful seed-specific jobs, run #43804365839](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43804365839) • [stacked review PR #4](https://github.com/CreoEnMi-p/llm-research-paper-drafts/pull/4)

## Precommitted validation, no new tuning

- All arms: width32, 2 layers, 2 heads, synthetic two-phase MQAR L64/P16, B32, init .5, lr .05 / elr .04, seeds0/1/2, CPU, 2 threads.
- Gated-linear baseline trained 300 steps (same tokens) AND 1500 steps (approximately the same or longer CPU wall time as sequential Gate).
- Gate with self-consistent sequential read `S_(t-1) q_(t-1)`; fixed gamma init .25 and margin .2; separate `S_(t-1) k_t` *current-key* signal with **identical** margin. Both trained 300 steps.
- **Completely independent heldout:** 96 sequences generated using RNG seed **91731**, not the seed12345 512-example validation monitored during training, nor prior AR seed12345 data.
- Autoregressively generate all 16 query answers; insert each model's own generated answer before later queries, leaving fact writes and query keys fixed. All 3 jobs passed all evaluation assertions, including first answer parity with teacher forcing and unchanged known tokens.
- **Decision fixed before seeing new split:** previous-read + margin must outperform g1500 in >=2/3 seeds **and have positive average AR improvement**, else stop *that* candidate. Competing current-key success rejects uniqueness of prior-read signal.

## New heldout results (AR answer accuracy)

| arm | seed0 | seed1 | seed2 | mean | CPU train sec (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| g 300 steps | 0.833984 | 0.834635 | 0.850911 | **0.839843** | 6.71 |
| g 1500 steps | 0.920573 | 0.935547 | 0.925781 | **0.927300** | 33.21 |
| exact previous-read + margin .2, 300 | 0.930990 | 0.939453 | 0.930339 | **0.933594** | 24.78 |
| exact current-key + same margin .2, 300 | 0.942057 | 0.927734 | 0.937500 | **0.935764** | 24.88 |
| g1500 frozen answer-position write suppression | 0.969401 | 0.987630 | 0.981120 | **0.979384** | N/A (posthoc) |

Paired previous-read margin vs g1500 difference: seed0 **+1.0417 pp**, seed1 **+0.3906 pp**, seed2 **+0.4558 pp**. Three positive signs and +0.6294 pp mean. **The precommitted exploratory go criterion passes 3/3.** This is a small effect: with three independent seeds even a 3/3 sign test gives only one-sided p=0.125 under an equiprobable null, and there are only 96 heldout sequences per seed with correlated query answers; cannot call statistically conclusive.

Paired same-margin current-key vs previous-read: **+1.1067, −1.1719, +0.7161 pp**; average current-key accuracy is 0.217 pp higher. **No unique prior-query signal** is established.

Model/time caveats: “matched time” is the wall time of one GitHub Actions CPU worker (32–33 s g1500 vs ~24–26 s sequential Gate). Different input token budgets (5× more for baseline) and different implementation efficiency; neither theoretical complexity nor accelerator throughput is established. Exact Gate remains sequential O(Td²) reference, not a production chunk kernel.

## Relation to parallel results and the architecture decision

- Gemini 3-seed teacher-forced `mqar2` mixed fact/query/fact/query saw previous-read exact Gate improve accuracy from ~0.89 to ~0.96, but required ~100 s vs ~22 s baseline **at equal steps**; no compute-matched longer baseline there. mqar2 adds *distinct* keys, not overwrites.
- Gemini equal-margin previous-read vs current-key on original MQAR reported comparable results; our heldout AR result likewise fails to establish superiority of previous-read.
- Original no-margin AR falsifier (PR #3) observed g mean .8550 vs exact previous-read .8561 with paired signs +,-,-. **This remains a valid negative result about that specific no-margin proposal.** Margin is a distinct follow-up hypothesis proposed independently and then validated on a new split; don't retroactively alter the stop criterion.
- Frozen answer-position write suppression remains consistently helpful; *that is a causal observation*, not a learned policy or unique paper-level discovery, and overlaps literature on linear attention memory interference.

**Decision:** Retain **“appropriate content/phase-selective write suppression can improve training/sample efficiency in small synthetic associative retrieval”** as a narrow observation. **Do not** claim “previous-read match is the right signal,” a fast low-overhead kernel, natural language transfer, or method novelty. Before any further architecture development, require (1) official GDN2/PH-like baseline evaluated for ability to suppress answer writes, with equal training tokens and equivalent optimized compute; (2) randomized mixed stream with *necessary new keys sharing the same value and key overwrites*, tracking old/new association correctness separately; (3) an efficient chunk-independent implementation with validated recurrent equivalence. If existing methods already solve interference or aliases/overwrites fail, close Paper 03 as an architecture route and preserve the causal pilot as a mechanistic research note.

## Meeting-ready claim

> At matched wall-time scale in a small synthetic associative retrieval task, an exact recurrent write-suppression gate with a fixed cosine margin achieves 93.36% autoregressive answer accuracy vs 92.73% for a much longer-trained gated-linear baseline, over three paired seeds on fresh heldout data. This 0.63 pp preliminary advantage also occurs with a current-key read (93.58%), and the proposed previous-read Gate has no demonstrated unique mechanism, kernel efficiency, or advantage over contemporary GDN2. Independently, disabling answer writes in a frozen baseline raises autoregressive accuracy from 92.73% to 97.94%, validating a strong task-specific interference phenomenon. Seeking advisor guidance on whether testing natural workloads vs state-of-the-art memory mechanisms is worth further investment.
