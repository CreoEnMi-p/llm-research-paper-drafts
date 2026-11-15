#!/bin/sh
# Headline of the identity/pointer-heads paper in about 1 minute on a laptop (Apple MPS or CPU).
# Recall: tuned plain attention (ALiBi + sink) vs identity heads at 2 hops, and vs pointer heads + no-backtrack at 4 hops.
# Expected: R-h2-F about 0.33 (guessing), R-h2-ID about 0.99, R-h4L4-F about 0.33, R-h4L4-PTRnb about 0.99 at step 25.
# "sh reproduce.sh lm" also runs the text pair (S5: width 64, 4 layers, L 256, 150 steps; F about 3.800, PM NA about 3.778; about 90 s more).
# Each run appends one line to results.jsonl with a tag starting with "R-".
cd "$(dirname "$0")"
python3 grid.py <<'B'
R-h2-F      | rbg.py | --task hopx --hops 2 --pairs 8 --layers 3 --steps 300 {REC} {F}
R-h2-ID     | rbg.py | --task hopx --hops 2 --pairs 8 --layers 3 --steps 300 {REC} {ID}
R-h4L4-F    | rbg.py | --task hopx --hops 4 --pairs 8 --layers 4 --steps 250 {REC} {F}
R-h4L4-PTRnb | rbg.py | --task hopx --hops 4 --pairs 8 --layers 4 --steps 250 {REC} {PTR} --pnb 3
B
if [ "$1" = "lm" ]; then
python3 grid.py <<'B'
R-S5-LF     | atg.py | {LM} {LF}
R-S5-LPMNA  | atg.py | {LM} {LPMNA}
B
fi
