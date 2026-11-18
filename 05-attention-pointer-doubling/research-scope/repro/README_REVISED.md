# PBS reviewer revision — reproducibility

Requirements: Python 3.11+, numpy, torch CPU; standard library only otherwise.

## Invalidation and freeze
- The first test V1 (seed 2026111703) was inadvertently inspected in a smoke run. Its results are invalid for final testing and not included.
- The FINAL V2 test (seed 2026111704) is frozen at `frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl` with SHA256 `e28b40b6a8b6e1b015bbfd202687ae95ea5b0387c1f626c4d1ffa9a1028227ce`.
- `frozen_protocol_v2/MODEL_CONFIG_LOCK.json` fixes 2-layer d64 Transformer and pointer-depth, 1000 updates batch32 AdamW LR.001 seeds 0–4, before first V2 use.

## Integrity checks (do not select architectures using V2)
```
python test_grammar_protocol.py
python freeze_composition_test.py --outdir frozen_protocol_v2 --test-seed 2026111704
sha256sum frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl
```

## Old chance and source-only baselines
```
python audit_chance_heuristics.py --n 128 --out chance_heuristics_verified.json
python recompute_metrics.py --out old_recomputed_tables.json
python eval_frozen_heuristics.py --out frozen_v2_script_heuristics.json
```

## Fixed one-off V2 evaluation replay
```
for arm in transformer pointer-depth; do
  for seed in 0 1 2 3 4; do
    python train_composition_protocol.py --model "$arm" --seed "$seed" --out "grammar_${arm}_s${seed}.json"
  done
done
python aggregate_new_grammar.py --out grammar_frozen_5seed_summary.json --reps 7000
```

Model selects ONLY final step; no tuning on V2. Note test has been consumed once in the archived experiment. Reusing V2 for model changes is prohibited if treating it as a virgin holdout; create a new prespecified split and protocol.

Potential bias: finite program grammar, four distinct source constants per program, source-literal output architecture; only 4 A+C production clusters. No claim of production-code understanding, compiler parity, or ICML-level novelty.

For original Paper05 63/6 and 127/7 script ceiling, deep/compute comparisons and raw logs, see separate branch `research/05-pointer-doubling-audit` in `CreoEnMi-p/llm-research-paper-drafts`, notably workflow run IDs 44227905878, 44228066817, 44228198848.
