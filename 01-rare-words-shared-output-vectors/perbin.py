"""Per-bin gain (nats per target) of arm Y over arm X across cells: python3 perbin.py X Y cell1 cell2 ..."""
import json, os, sys
R = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid.jsonl"))}
ALIAS = {"v6_base": "v6_base_h1", "v6_heps": "v6_heps5_h1", "v6_clus": "v6_clus", "v6_ce": "v6_clus_heps5"}
g = lambda c, a: R.get(f"{c}_{a}") or R[ALIAS[f"{c}_{a}"]]
X, Y = sys.argv[1:3]; BN = ["0", "1-9", "10-31", "32-99", "100-315", "316-999", "1k-3k", "3k+", "unk"]
print("cell".ljust(14), *[b.rjust(13) for b in BN])
for c in sys.argv[3:]:
    x, y = g(c, X), g(c, Y)
    print(c.ljust(14), *[(f"{x['bins'][b][0]-y['bins'][b][0]:+.3f}/{x['bins'][b][1]:.3f}" if b in x["bins"] else "-").rjust(13) for b in BN])
