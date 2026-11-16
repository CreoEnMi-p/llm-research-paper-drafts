# Independent reproduction and research audit (November 2026)

This branch intentionally does not change original model source files. It adds CPU-only workflows and a seed-intervention runner. Original code base: `4e5f54d93bc27ec2cb1587414fa1ac056b359c6a`.

**CI receipts**

- [Original-code smoke reproduction](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43792798882)
- [Four-seed / negative-control follow-up](https://github.com/CreoEnMi-p/llm-research-paper-drafts/actions/runs/43793394215)

**Paper 03 — linear attention predicted-skip**

Original script: MQAR baseline loss 0.373, accuracy 0.906; predicted-skip loss 0.173, accuracy 0.947.

Four-seed matched comparisons (mean loss, lower is better): base 0.6147; previous-query read gate 0.2593; content-shuffled gate 0.4642; current-key read gate 0.2153. Predicted-skip beats baseline 4/4 seeds. Current-key memory read is similarly effective, narrowing the novelty claim of the previous-query-read mechanism. **Not** evidence of superiority on production GDN-2 or Muir, which were not evaluated here. Only 4 seeds, fixed synthetic validation task.

**Paper 05 — attention pointer doubling**

Four-seed result (mean exact-match accuracy): reversed 7-hop / 3-layer 0.9803; forward 0.0765; random edge order 0.0482. A distinct predecessor-doubling mechanism on forward 8-hop/4-layer gives 0.9258. Thus the *specific doubling orientation* is order-dependent, not that all causal attention is impossible on forward chains. Implementation uses one global per-layer hop gate; mixed query-specific hop lengths have not been tested.

**Reproducibility blockers in remaining drafts**

- Paper 01 reads data from missing sibling path `../llm-per-word-weights/data`. `PAPER.md` shows an Adam-state reset after reclustering, while `grid.py`'s `recluster()` and training loop do not reset the optimizer state. Needs a reset-vs-keep ablation.
- Paper 02 requires missing `data/fineweb_train_000001.slice` and `data/fineweb_val_000000.slice`; its claimed run-level `results.jsonl` is absent from the checked directory.
- Paper 06 requires missing FineWeb `.bin` files and `counts.npy`; cannot reproduce published bucket statistics from this fork as-is.
- Paper 04 is self-contained synthetic code, but its core identity matching is precedent to paper 05.

**Research recommendation:** Focus on paper 03 as an interference-mechanism question, not simply another gating formula. Highest-risk potential upside is paper 05 if it solves arbitrary edge order and query-dependent hop count with one model. These are research hypotheses, not submission-ready claims.

Run the included workflows via the Actions tab; the runs above contain uploaded logs/artifacts. No original paper result has been changed in this branch.
