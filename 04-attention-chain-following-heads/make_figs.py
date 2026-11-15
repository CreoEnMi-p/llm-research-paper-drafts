"""Figures for PAPER.html (numbers copied from RESULTS.md / results.jsonl, grid tags B1-B10)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
C = {"PM": "#2a78d6", "NG2": "#eb6834", "ID": "#1baf7a", "PTR": "#6250d6", "F": "#8a8984"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})

# 1. recall headline: 4 settings x arms
sets = ["2 hops\n8 pairs", "2 hops\n32 pairs", "3 hops\n4 layers", "6 hops\n6 layers"]
arms = [("Plain (ALiBi+sink)", [0.333, 0.107, 0.328, 0.316], C["F"]), ("KV shifting", [0.344, 0.117, 0.331, 0.332], "#b0afa8"),
        ("Primer", [0.343, 0.126, 0.334, 0.332], "#cfcec8"), ("n-gram heads", [0.172, 0.062, 0.277, 0.336], "#eda100"),
        ("Identity heads", [0.991, 0.075, 0.328, 0.332], C["ID"]), ("Pointer heads (+ no-backtrack at 6 hops)", [0.999, 0.999, 1.000, 0.990], C["PTR"])]
fig, ax = plt.subplots(figsize=(8, 3.4)); w = 0.13; x = np.arange(len(sets))
for i, (n, v, c) in enumerate(arms):
    vv = [np.nan if a is None else a for a in v]; ax.bar(x + (i - 2.5) * w, vv, w * 0.9, color=c, label=n)
ax.set_xticks(x, sets); ax.set_ylabel("final-answer accuracy"); ax.set_ylim(0, 1.05); 
for xi, g in zip(x, [0.34, 0.11, 0.34, 0.34]): ax.plot([xi - 0.45, xi + 0.45], [g, g], color="#333", lw=0.8, ls="--")
ax.text(-0.45, 0.36, "guessing", fontsize=8, color="#555", ha="left"); ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
ax.set_title("Multi-hop recall: only pointer heads keep working as hops and pairs grow", fontsize=10, pad=34); fig.tight_layout(); fig.savefig(os.path.join(OUT, "recall.png"), dpi=160); plt.close(fig)

# 2. hops x layers map for PTR (with and without no-backtrack)
H, Ls = [2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]
plain = {(2, 1): .114, (2, 2): .999, (3, 2): .307, (3, 3): .998, (3, 4): 1.0, (3, 6): .998, (4, 3): .334, (4, 4): .995, (4, 6): .992,
         (5, 4): .340, (5, 5): .971, (5, 6): .929, (6, 6): .325, (8, 8): .339}
nb = {(3, 3): 1.0, (4, 4): 1.0, (4, 5): .993, (5, 5): .990, (6, 5): .328, (6, 6): .990, (7, 6): .354, (7, 7): .433, (8, 8): .835, (3, 2): .159}
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
for ax, d, t in [(axs[0], plain, "pointer heads"), (axs[1], nb, "pointer heads + no-backtrack")]:
    M = np.full((len(H), len(Ls)), np.nan)
    for (h, l), v in d.items(): M[H.index(h), Ls.index(l)] = v
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto"); ax.grid(False)
    for (h, l), v in d.items(): ax.text(Ls.index(l), H.index(h), f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "#222")
    ax.plot([-0.5, 7.5], [-1.5, 6.5], color="#e34948", lw=1, ls="--")
    ax.set_ylim(-0.5, len(H) - 0.5); ax.set_xticks(range(len(Ls)), Ls); ax.set_yticks(range(len(H)), H); ax.set_xlabel("layers"); ax.set_ylabel("hops"); ax.set_title(t, fontsize=10)
fig.suptitle("Accuracy (8-32 pairs, 250 steps). Dashed line: layers = hops; cells right of it can solve", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(OUT, "hops_layers.png"), dpi=160); plt.close(fig)

# 3. LM gain vs F along three knobs
knobs = [("width (150 steps, L 256)", [32, 64, 128], {"NG2": [.008, .017, .016], "ID": [-.003, .013, .021], "PM": [.006, .022, .030]}),
         ("training steps (width 64, L 256)", [150, 300], {"NG2": [.017, .016], "ID": [.013, .024], "PM": [.022, .031]}),
         ("context length (width 64, 150 steps)", [64, 128, 256, 512], {"NG2": [-.001, .010, .017, .024], "ID": [-.001, .008, .013, None], "PM": [.005, .010, .022, .034]}),
         ("layers (width 64, 150 steps, L 256)", [2, 4, 6], {"NG2": [.019, .017, .013], "ID": [None, .013, None], "PM": [.032, .022, .028]})]
names = {"NG2": "n-gram heads, order 2", "ID": "identity heads", "PM": "pair-match identity heads"}
fig, axs = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
for ax, (t, xs, ser) in zip(axs, knobs):
    for k in ["NG2", "ID", "PM"]:
        ys = [np.nan if v is None else v for v in ser[k]]; ax.plot(range(len(xs)), ys, color=C[k], lw=2, marker="o", ms=5, label=names[k])
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_xticks(range(len(xs)), xs); ax.set_title(t, fontsize=9)
axs[0].set_ylabel("loss gain over tuned plain attention"); axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Text (FineWeb, V 2,000): the pair-match gain grows with width, steps and context and holds with depth; width 128 + 300 steps: +0.049", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "lm_scaling.png"), dpi=160); plt.close(fig)

# 4. pointer mass per layer (4 hops, 4 layers)
fig, ax = plt.subplots(figsize=(5.2, 2.8)); lay = np.arange(4); w = 0.27
for i, (n, v, c) in enumerate([("at init", [.97, .46, .17, .07], "#b7d3f6"), ("trained 125 steps", [.93, .56, .52, .82], "#6da7ec"), ("no-backtrack, at init", [.97, .88, .88, .87], C["PTR"])]):
    ax.bar(lay + (i - 1) * w, v, w * 0.9, color=c, label=n)
ax.set_xticks(lay, ["layer 0\n(stage 1)", "layer 1\n(stage 2)", "layer 2\n(stage 3)", "layer 3\n(answer)"]); ax.set_ylabel("pointer mass on right token"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18)); fig.tight_layout(); fig.savefig(os.path.join(OUT, "pointer_mass.png"), dpi=160); plt.close(fig)
print("figs in", OUT)
