"""Summarize grid cells: python3 cells.py [prefix ...]. A cell is tags <cell>_base/_heps/_clus/_ce (+ optional others).
Prints B, E, C, CE, gains B-C, E-CE, the non-unknown part of E-CE, and the share of targets on words seen 10-31 times."""
import json, os, sys
R = {}
for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid.jsonl")):
    r = json.loads(l); R[r["tag"]] = r
ALIAS = {"v6_base": "v6_base_h1", "v6_heps": "v6_heps5_h1", "v6_clus": "v6_clus", "v6_ce": "v6_clus_heps5"}
def get(c, arm):
    t = f"{c}_{arm}"; return R.get(t) or R.get(ALIAS.get(t, ""))
def nonunk(a, b):
    return sum(sh * (a["bins"][k][0] - b["bins"][k][0]) for k, (l, sh) in a["bins"].items() if k != "unk" and k in b["bins"])
def band(r, ks): return sum(r["bins"][k][1] for k in ks if k in r["bins"])
cells = sorted({t.rsplit("_", 1)[0] for t in R if t.endswith("_ce")} | {"v6"})
if len(sys.argv) > 1: cells = [c for c in cells if any(c.startswith(p) for p in sys.argv[1:])]
print(f"{'cell':22s} {'B':>7s} {'E':>7s} {'C':>7s} {'CE':>7s} {'B-C':>7s} {'E-CE':>7s} {'nonunk':>7s} {'E-CE/E':>7s} {'s1-31':>6s} {'s10-99':>6s} {'unk':>5s} tpw")
for c in cells:
    b, e, cl, ce = (get(c, x) for x in ("base", "heps", "clus", "ce"))
    if not ce: continue
    v = lambda r: r["scores"]["val"] if r else float("nan")
    nu = nonunk(e, ce) if e else float("nan")
    print(f"{c:22s} {v(b):7.4f} {v(e):7.4f} {v(cl):7.4f} {v(ce):7.4f} {v(b)-v(cl):+7.4f} {v(e)-v(ce):+7.4f} {nu:+7.4f} {'':7s} {band(ce,['1-9','10-31']):6.3f} {band(ce,['10-31','32-99']):6.3f} {ce['bins'].get('unk',[0,0])[1]:5.3f} {ce['tok_per_weight']}")
