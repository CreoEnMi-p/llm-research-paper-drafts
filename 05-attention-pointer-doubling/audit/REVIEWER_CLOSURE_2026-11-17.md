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

## 63/6 learned model status
Earlier 400-step first-query batch2 and 800-step all-query gate-init comparisons did not solve 63/6 and could not prove representational impossibility. A stronger [1,200-step × batch8 first-question-only comparison](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/44228198848) is running **BLN6 (3 preset-gate and 2 learned-gate seeds), RoPE8 (2), NoPE8 (1), 1-hop L12 (2), BLN127/7 (1)**. Do **not** treat running jobs as validated results. The 63 first-query chance among four known candidates is 25%; the 4-query average .520833 is irrelevant to first query. Updated Actions run status and result logs should be read before reporting any 63-hop neural success claim.

Even when this run finishes, it matches update count, **not actual FLOPs**; interpret together with profiler. A genuine fixed-FLOPs study should allocate more training steps to RoPE/NoPE8 relative to BLN6.

## Research decision
The 7/15 reverse-order toy demonstrates a real architecture optimization advantage after corrected first-query 5 seeds, but the difference narrows against trained deeper one-hop pointer. Current Sanford k7 20k model does not show transfer. The separate `research/05-position-binding-scope` branch now tests a much more difficult lexical binding problem under a frozen grammar split; it must be judged on paired counterfactual correctness rather than raw literal prediction.
