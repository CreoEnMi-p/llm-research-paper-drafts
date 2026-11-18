# Paper05 old-line reviewer closure: script ceiling, single query, L+2 and long-chain compute

**17 November 2026 research audit. All source and CI on independent `research/05-pointer-doubling-audit`; main unchanged.**

## Correct metric, scripted reference, exact chance baselines
The original `hopx` generator is deterministic given the context (not intrinsically ambiguous). Scripted dictionary traversal that only reads the context and not the gold answer slots achieves **512/512 first-query correct** and **all queries correct** under k7,N8; k15,N8; k63,N4; k63,N8; k127,N4. It is a *symbolic ceiling*, not a trained neural comparator. [Successfully validated script run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44227905878), [source](oldline_symbolic_ceiling.py), [raw JSON artifact from Actions](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44227905878).

**Three different priors, do not conflate:** vocabulary-uniform `1/V`; candidate-known first-question `1/N`; candidate-elimination teacher-forced query index `1/(N-j)`. For N8 query average `H8/8=.339732`, for N4 average `H4/4=.520833`. Raw random guessing over eligible answers in the same run follows these analytic curves. The last question can reach 100% with no multi-hop reasoning. A corrected first-query-only training loss and per-query-index evaluation exist in `audit/build_reviewer.py`.

## Deep official Sanford k7 (completed)
The pinned author Sanford source generator, terminal valid k7 only, 20k AdamW updates, batch48:
- RoPE L4, 5 seeds **32.89%** final mean;
- RoPE **L5**, 5 seeds, **33.4375%** final mean; individual [.32421875,.343750,.33203125,.3515625,.3203125], mean peaks ~35.31%;
- RoPE **L6**, 2 seeds, final [.3359375,.33203125], mean 33.40%.
None learns reliable k7 terminal under this specific tiny model training configuration. It is **not Sanford's 100k-step published full-sequence GPT2 protocol** and does not refute its theory.
[RoPE L5/6 20k jobs (9/9 completed)](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44127469803).
A separate k1→k7 terminal curriculum across RoPE L5/6 (3 seeds each, 3k total steps) also stays ~32–34% final. [Jobs](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44130348908).

## Matched same-runner 63/6 and 127/7 cost (completed)
[CPU profiler 44228066817](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44228066817), [machine-readable archive](oldline_longchain_compute_verified.json). Batch1, 2 threads, original `pdg.py` generator, two warmup+five timed forward/backward passes, same sequence length per k; partial profiler matmul FLOPs, **not** GPU or full FLOPs.

| Context | Arm | Effective layers | CPU median ms/update | Partial matmul FLOPs (G) |
|---|---|---:|---:|---:|
| 63 hop, N4 | RoPE L+2 | 8 | 62.40 | 1.411 |
| 63 hop, N4 | NoPE L+2 | 8 | 62.55 | 1.409 |
| 63 hop, N4 | Doubling with preset gates | 6 | 114.26 | 4.532 |
| 63 hop, N4 | Doubling default gates | 6 | 119.10 | 4.532 |
| 63 hop, N4 | Sequential 1-hop pointer | 12 | 150.03 | 5.174 |
| 127 hop, N4 | Doubling with preset gates | 7 | 352.54 | 18.190 |
| 127 hop, N4 | Sequential 1-hop pointer | 14 | 422.58 | 18.838 |

**Interpretation:** relative to learned 1-hop L12, 63/6 BLN is ~1.31× lower CPU step cost; relative to RoPE8, BLN is ~1.83× more expensive in walltime and ~3.21× partial matmul FLOPs. Same parameter count or update count is **not compute matching**. Claim only a depth-vs-compute tradeoff; do not state unqualified efficiency advantage.

## 63/6 learned model: all ten first-only training runs completed, negative

[Completed k63 first-only batch8×1200 run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44228198848), [machine-readable results](oldline_63_firstonly_1200_verified.json). Same reverse-block original `hopx` generator, N4, V1024, 128 *shared* fixed validation rows, train only **first query** to prevent later-answer elimination. Known-candidate first-question reference = **25%**.

| Model | n train seeds | First-query accuracies |
|---|---:|---|
| Doubling L6, intended gate initialized +8 | 3 | **25.78%, 28.91%, 29.69%** |
| Doubling L6, gate learned from 0 | 2 | 23.44%, 24.22% |
| Standard RoPE L8 | 2 | 22.66%, 25.00% |
| Standard NoPE L8 | 1 | 25.00% |
| Sequential one-hop identity pointer L12 | 2 | 21.09%, 27.34% |

**No arm reliably learns k63.** Even oracle-style *gate initialization* is not enough within this limited budget, so a claim of 63/6 working would be false. A +4.7pp deviation in a single model run is smaller than the large 128-row sampling uncertainty. Seeds reuse a fixed eval set; do not treat 3×128 as independent heldout examples.

These runs were **update-count matched, NOT compute matched**. Profiling shows BLN6 ~114ms/step and ~4.53G partial matmul FLOPs, versus RoPE8 ~62ms and ~1.41G. A separate [seven-arm approximate compute-matched RoPE8/NoPE8 run](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44232391911) **completed successfully**; see [raw verified rows](oldline_63_compute_matched_verified.json). At 2,200 steps (roughly BLN CPU-time ratio 114/62), RoPE seeds0/1 first-query **24.22% / 22.66%**, NoPE seeds0/1 **21.09% / 23.44%**. At 3,900 steps (roughly BLN profiler matmul-FLOP ratio 4.53/1.41), RoPE seeds0/1 **24.22% / 29.69%**, NoPE seed0 **32.03%**. None reliably solves 63-hop, but each result uses only n=128 fixed validation examples and **one** 3,900-step NoPE seed cannot support "no delayed learning" at still-longer budgets. Across Actions runners the CPU time varies, and profiler FLOPs omit substantial operations. **This is approximate resource matching, not a controlled exact GPU FLOP experiment.** The separate **k127/L7** model run in the earlier workflow is still pending and cannot be counted complete without its own success marker.

## Research decision
The 7/15 reverse-order toy demonstrates a real architecture optimization advantage after corrected first-query 5 seeds, but the difference narrows against trained deeper one-hop pointer. Current Sanford k7 20k model does not show transfer. The separate `research/05-position-binding-scope` branch now tests a much more difficult lexical binding problem under a frozen grammar split; it must be judged on paired counterfactual correctness rather than raw literal prediction.
