#!/bin/sh
# Headline of the pointer-doubling paper in under 2 minutes on a laptop (Apple MPS or CPU).
# 7 hops in 3 layers and 15 hops in 4 layers, 8 pairs, chains stored in reverse order (--rev 1).
# Expected (run 2026-11-15, 67 s): RP-h7L3-F 0.17 (chance 0.125), RP-h7L3-BLN 0.97; RP-h15L4-PNB (one-hop pointer) 0.17, RP-h15L4-BLN 0.99, both pointer runs past 0.9 at step 25.
# Each run appends one line to results.jsonl with a tag starting with "RP-".
cd "$(dirname "$0")"
python3 grid.py <<'B'
RP-h7L3-F     | pdg.py | --task hopx --hops 7 --pairs 8 --layers 3 --rev 1 --steps 50 --batch 16 {REC} {F}
RP-h7L3-BLN   | pdg.py | --task hopx --hops 7 --pairs 8 --layers 3 --rev 1 --steps 50 --batch 16 {REC} {BLN}
RP-h15L4-PNB  | pdg.py | --task hopx --hops 15 --pairs 8 --layers 4 --rev 1 --steps 50 --batch 16 {REC} {PNB}
RP-h15L4-BLN  | pdg.py | --task hopx --hops 15 --pairs 8 --layers 4 --rev 1 --steps 50 --batch 16 {REC} {BLN}
B
