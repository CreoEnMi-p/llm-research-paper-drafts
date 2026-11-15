"""Figures for PAPER.html (numbers copied from RESULTS.md round 1 / results.jsonl)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
C = {"BLN": "#2a78d6", "DNB": "#7fb2ee", "PNB": "#eb6834", "F": "#8a8984", "KVS": "#b0afa8", "PRI": "#cfcec8", "NG": "#eda100"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})

# 1. reach: largest k solved (>= 0.9) vs layers, plus the BLN accuracy map at 3 and 4 layers
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axs[0]; L = [1, 2, 3, 4, 5]
ax.plot(L, [1, 3, 7, 15, 31], color=C["BLN"], lw=2.2, marker="o", ms=6, label="pointer doubling + lifting")
ax.plot([2, 3, 4], [3, 6, 9], color=C["PNB"], lw=2.2, marker="s", ms=6, label="one-hop pointer + no-backtrack")
ax.set_xticks(L); ax.set_xlabel("layers L"); ax.set_ylabel("most hops solved"); ax.set_yscale("log", base=2); ax.set_yticks([1, 3, 7, 15, 31], [1, 3, 7, 15, 31])
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_title("(a) reach grows as 2^L - 1", fontsize=10)
ax = axs[1]; ks = list(range(3, 17))
m3 = {3: .989, 4: .999, 5: .986, 6: .978, 7: .979, 8: .131}; m4 = {3: .979, 9: .987, 10: .991, 11: .981, 12: .992, 13: .923, 14: .438, 15: .977, 16: .120}
M = np.full((2, len(ks)), np.nan)
for r, d in enumerate([m3, m4]):
    for k, v in d.items(): M[r, ks.index(k)] = v
ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
for r, d in enumerate([m3, m4]):
    for k, v in d.items(): ax.text(ks.index(k), r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "#222")
for r, lim in enumerate([7, 15]): ax.plot([ks.index(lim) + 0.5] * 2, [r - 0.5, r + 0.5], color="#e34948", lw=1.5, ls="--")
ax.set_xticks(range(len(ks)), ks); ax.set_yticks([0, 1], ["3 layers", "4 layers"]); ax.set_xlabel("hops k (dashed: k = 2^L - 1)")
ax.set_title("(b) accuracy of doubling + lifting, 50 steps (k = 14: 100 steps)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "reach.png"), dpi=160); plt.close(fig)

# 2. recall in four settings, best lr per arm
sets = ["7 hops\n3 layers", "11 hops\n4 layers", "15 hops\n4 layers", "7 hops, 3 layers\n16 pairs (256 tokens)"]
arms = [("Tuned attention", [.269, .293, .295, .125], C["F"]), ("KV shifting", [.269, .289, .273, .072], C["KVS"]), ("Primer", [.263, .275, .311, .063], C["PRI"]),
        ("n-gram heads", [.157, .170, .219, .055], C["NG"]), ("One-hop pointer + nb", [.208, .194, .288, .081], C["PNB"]),
        ("Doubling, no lifting", [.981, .207, .993, None], C["DNB"]), ("Doubling + lifting", [.987, .981, .984, .991], C["BLN"])]
fig, ax = plt.subplots(figsize=(8.4, 3.5)); w = 0.115; x = np.arange(len(sets))
for i, (n, v, c) in enumerate(arms):
    ax.bar(x + (i - 3) * w, [np.nan if a is None else a for a in v], w * 0.9, color=c, label=n)
for xi, g in zip(x, [.125, .125, .125, .0625]): ax.plot([xi - 0.45, xi + 0.45], [g, g], color="#333", lw=0.8, ls="--")
ax.text(-0.45, 0.145, "chance", fontsize=8, color="#555"); ax.set_xticks(x, sets); ax.set_ylabel("final-answer accuracy"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2)); fig.tight_layout(); fig.savefig(os.path.join(OUT, "recall.png"), dpi=160); plt.close(fig)

# 3. init probe: pointer mass on chain stage 2^l at step 0
fig, ax = plt.subplots(figsize=(5.6, 2.9)); lay = np.arange(5); w = 0.27
for i, (n, v, c) in enumerate([("doubling, 15 hops", [.90, .47, .41, .23, None], "#b7d3f6"), ("+ doubled no-backtrack, 15 hops", [.90, .85, .86, .83, None], "#6da7ec"),
                               ("+ doubled no-backtrack, 31 hops", [.90, .84, .86, .84, .82], C["BLN"])]):
    ax.bar(lay + (i - 1) * w, [np.nan if a is None else a for a in v], w * 0.9, color=c, label=n)
ax.set_xticks(lay, [f"layer {l}\n(stage {2 ** l})" for l in range(5)]); ax.set_ylabel("pointer mass on right token"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25)); fig.tight_layout(); fig.savefig(os.path.join(OUT, "probe.png"), dpi=160); plt.close(fig)
print("figs in", OUT)
