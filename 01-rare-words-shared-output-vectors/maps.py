"""Figures for RESULTS.md / PAPER.md from grid.jsonl. python3 maps.py -> fig_*.png"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for l in open(os.path.join(HERE, "grid.jsonl")):
    r = json.loads(l); R[r["tag"]] = r
ALIAS = {"v6_base": "v6_base_h1", "v6_heps": "v6_heps5_h1", "v6_clus": "v6_clus", "v6_ce": "v6_clus_heps5"}
v = lambda t: (R.get(t) or R.get(ALIAS.get(t, ""), {})).get("scores", {}).get("val", np.nan)
S1, S2, S3, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=160); plt.close(fig); print(name)


# 1. gain over the strongest competitor vs vocabulary, three model variants
fig, ax = plt.subplots(figsize=(6, 3.6))
Vs = [2, 4, 6, 10, 16, 24]
ax.plot(Vs, [v(f"v{x}_heps") - v(f"v{x}_ce") for x in Vs], "o-", color=S1, ms=6, label="untied")
tv = [2, 6, 16]
ax.plot(tv, [v(f"v{x}tie_heps") - v(f"v{x}tie_ce") for x in tv], "s-", color=S2, ms=6, label="tied input/output")
ax.plot(tv, [v(f"v{x}si_heps") - v(f"v{x}si_ce") for x in tv], "^-", color=S3, ms=7, label="strong input side")
ax.set_xscale("log"); ax.set_xticks(Vs); ax.set_xticklabels([f"{x}k" for x in Vs]); ax.minorticks_off()
ax.set_xlabel("vocabulary size"); ax.set_ylabel("loss gain over competitor (nats)"); ax.set_ylim(0, None)
ax.set_title("Cluster head minus best head-only epsilon, 400 steps", color=INK, fontsize=10, loc="left")
ax.legend(frameon=False); save(fig, "fig_vocab.png")


# 2. heatmap vocabulary x training steps
def heat(name, rows, cols, tagf, xl, yl, title):
    M = np.array([[v(tagf(r, c)[0]) - v(tagf(r, c)[1]) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(cols), 0.9 + 0.7 * len(rows)))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.04, aspect="auto"); ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color=INK if M[i, j] < 0.025 else "white", fontsize=9)
            else: ax.text(j, i, "not run", ha="center", va="center", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([str(c[0]) for c in cols]); ax.set_yticks(range(len(rows))); ax.set_yticklabels([str(r[0]) for r in rows])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="gain (nats)"); save(fig, name)


VR = [("2k", 2), ("6k", 6), ("16k", 16)]
def st(r, c):
    x, s = r[1], c[1]
    if s == 400: return f"v{x}_heps", f"v{x}_ce"
    return f"v{x}s{s}_heps", f"v{x}s{s}_ce"
heat("fig_heat_steps.png", VR, [(200, 200), (400, 400), (800, 800), (1600, 1600)], st, "training steps (batch 32)", "vocabulary", "Gain over competitor: vocabulary x steps")
def wd(r, c):
    x, w = r[1], c[1]
    if w == 32: return f"v{x}_heps", f"v{x}_ce"
    return f"v{x}w{w}_heps", f"v{x}w{w}_ce"
heat("fig_heat_width.png", VR[:2], [(16, 16), (32, 32), (64, 64)], wd, "width", "vocabulary", "Gain over competitor: vocabulary x width")
def dp(r, c):
    x, L = r[1], c[1]
    if L == 2: return f"v{x}_heps", f"v{x}_ce"
    return f"v{x}L{L}_heps", f"v{x}L{L}_ce"
heat("fig_heat_depth.png", VR[:2], [(1, 1), (2, 2), (4, 4)], dp, "layers", "vocabulary", "Gain over competitor: vocabulary x depth")

# 3. per-word gain by corpus frequency
fig, ax = plt.subplots(figsize=(6, 3.6))
order = ["1e-6-3e-6", "3e-6-1e-5", "1e-5-3e-5", "3e-5-1e-4", "1e-4-3e-4", "3e-4-1e-3", ">1e-3"]
for (e, c, lab, col, mk) in [("v6si_heps", "v6si_ce", "V 6k, 400 steps (strong input)", S1, "o"), ("v16s800_heps", "v16s800_ce", "V 16k, 800 steps", S2, "s"),
                             ("v6b128s400_heps", "v6b128s400_ce", "V 6k, batch 128", S3, "^")]:
    re, rc = R.get(e) or R.get(ALIAS.get(e, "")), R.get(c) or R.get(ALIAS.get(c, ""))
    if not re or not rc: continue
    keys = [k for k in re["fbins"] if k != "unk"]
    xs, ys = [], []
    for k in keys:
        if k in rc["fbins"] and re["fbins"][k][1] > 0.003:
            xs.append(k); ys.append(re["fbins"][k][0] - rc["fbins"][k][0])
    FK = ["3e-6-1e-5", "1-3e-5", "3e-5-1e-4", "1-3e-4", "3e-4-1e-3", ">1e-3"]
    ax.plot([FK.index(k) for k in xs], ys, mk + "-", color=col, ms=6, label=lab)
    ax.set_xticks(range(len(FK))); ax.set_xticklabels(["3e-6..1e-5", "1e-5..3e-5", "3e-5..1e-4", "1e-4..3e-4", "3e-4..1e-3", ">1e-3"], rotation=20, fontsize=8)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("corpus frequency of the target word"); ax.set_ylabel("gain per target (nats)")
ax.set_title("Which words gain: cluster head minus competitor", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False)
save(fig, "fig_freq.png")

# 4. batch at fixed tokens
fig, ax = plt.subplots(figsize=(5, 3.4))
bt = [(16, "v6b16"), (32, "v6"), (64, "v6b64"), (128, "v6b128s100")]
ys = [v(f"{t}_heps") - v(f"{t}_ce") for _, t in bt]
ax.plot([b for b, _ in bt], ys, "o-", color=S1, ms=6, label="clusters over competitor")
ax.plot([b for b, _ in bt], [v(f"{t}_base") - v(f"{t}_heps") for _, t in bt], "s-", color=S2, ms=6, label="competitor over baseline")
ax.set_xscale("log", base=2); ax.set_xticks([b for b, _ in bt]); ax.set_xticklabels([str(b) for b, _ in bt]); ax.minorticks_off()
ax.set_xlabel("batch (sequences), same 819K tokens"); ax.set_ylabel("gain (nats)"); ax.set_ylim(0, None)
ax.set_title("Batch size at V 6k", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False); save(fig, "fig_batch.png")

# 5. probes at V 6k and 16k: what carries the gain
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
for ax, (title, items) in zip(axs, [("V 6k", [("competitor (eps 1e-5)", "v6_heps"), ("+ post-hoc shrink to centroid", None, R["v6_heps_shr"]["scores"]["val_shr3"]),
        ("+ clusters, shared vectors frozen", "v6_ce_frozenC"), ("+ random clusters", "v6_ce_rand"), ("+ frequency clusters", "v6_ce_freq"),
        ("+ k-means, stop re-cluster at 25%", "v6_ce_stop25"), ("+ k-means, stop at 50%", "v6_ce_stop50"), ("+ k-means clusters (full method)", "v6_ce")]),
        ("V 16k", [("competitor (eps 1e-5)", "v16_heps"), ("+ post-hoc shrink to centroid", None, R["v16_heps_shr"]["scores"]["val_shr1"]),
        ("+ random clusters", "v16_rand_ce"), ("+ frequency clusters", "v16_freq_ce"), ("+ k-means clusters (full method)", "v16_ce")])]):
    base = v(items[0][1]); labs = [it[0] for it in items[1:]]
    gains = [base - (it[2] if it[1] is None else v(it[1])) for it in items[1:]]
    cols = [S1 if "full" in l else MUTED for l in labs]
    ax.barh(range(len(labs)), gains, color=cols, height=0.6); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color=MUTED, lw=1); ax.grid(axis="y", visible=False)
    for i, g in enumerate(gains): ax.text(max(g, 0) + 0.0005, i, f"{g:+.4f}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_xlabel("loss gain over competitor (nats)"); ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlim(min(0, min(gains)) - 0.006, max(gains) + 0.009)
save(fig, "fig_probes.png")

# 6. cluster count x vocabulary
fig, ax = plt.subplots(figsize=(5, 3.4))
for x, col, mk in [(2, S1, "o"), (6, S2, "s"), (16, S3, "^")]:
    Ks = [16, 64, 256] + ([1024] if x > 2 else [])
    ys = [v(f"v{x}_heps") - (v(f"v{x}_ce") if k == 64 else v(f"v{x}_ce_k{k}")) for k in Ks]
    ax.plot(Ks, ys, mk + "-", color=col, ms=6, label=f"V {x}k")
ax.set_xscale("log", base=4); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1024"]); ax.minorticks_off()
ax.set_xlabel("number of clusters"); ax.set_ylabel("gain over competitor (nats)"); ax.set_ylim(0, None)
ax.set_title("Cluster count does not need to grow with V", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False); save(fig, "fig_K.png")

# 7. width, each arm at its best head rate / epsilon; cluster shared-vector rate scaled as 1/width
fig, ax = plt.subplots(figsize=(5.4, 3.4))
best = lambda tags: min(v(t) for t in tags)
W = [32, 64, 128]
v16t = [best(["v16_heps", "v16_heps3e5"]) - best(["v16_ce", "v16_ce_eps3e5"]),
        best(["v16w64_heps", "v16w64_heps_e3", "v16w64_heps_h05", "v16w64_heps_h05e3"]) - best(["v16w64_ce", "v16w64_ce_e3", "v16w64_ce_h05", "v16w64_ce_h05c01"]),
        best(["v16w128_heps", "v16w128_heps_e3", "v16w128_heps_h05", "v16w128_heps_h03", "v16w128_heps_h02"]) - best([t for t in R if t.startswith("v16w128_ce")])]
v6t = [best(["v6_heps"]) - best(["v6_ce", "v6_ce_eps3e5"]), v("v6w64_heps") - v("v6w64_ce"),
       best(["v6w128_heps", "v6w128_heps_h05", "v6w128_heps_h03", "v6w128_heps_h02", "v6w128_heps_lr01"]) - best([t for t in R if t.startswith("v6w128_ce")])]
v16d = [v("v16_heps") - v("v16_ce"), v("v16w64_heps") - v("v16w64_ce"), v("v16w128_heps") - v("v16w128_ce")]
ax.plot(W, v16t, "s-", color=S2, ms=6, label="V 16k, rates tuned per arm")
ax.plot(W, v6t, "o-", color=S1, ms=6, label="V 6k, tuned per arm (width 64: default)")
ax.plot(W, v16d, "s--", color=S2, ms=5, alpha=0.6, lw=1.5, label="V 16k, default rates")
ax.set_xscale("log", base=2); ax.set_xticks(W); ax.set_xticklabels([str(w) for w in W]); ax.minorticks_off()
ax.set_xlabel("width"); ax.set_ylabel("gain over competitor (nats)"); ax.set_ylim(0, None)
ax.set_title("Width: per-arm tuning keeps most of the gain", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8)
save(fig, "fig_width.png")
print("width tuned", [round(x, 4) for x in v16t], [round(x, 4) for x in v6t], "default", [round(x, 4) for x in v16d])

# 8. predicted vs measured, all prediction rounds
P = [("V 2k", 0.005, v("v2_heps") - v("v2_ce")), ("V 4k", 0.012, v("v4_heps") - v("v4_ce")), ("V 10k", 0.025, v("v10_heps") - v("v10_ce")),
     ("V 16k", 0.030, v("v16_heps") - v("v16_ce")), ("V 24k", 0.028, v("v24_heps") - v("v24_ce")), ("tail 4", 0.007, v("v6t4_heps") - v("v6t4_ce")),
     ("batch 16", 0.030, v("v6b16_heps") - v("v6b16_ce")), ("tail 2", 0.012, v("v6t2_heps") - v("v6t2_ce")), ("tail 8", 0.007, v("v6t8_heps") - v("v6t8_ce")),
     ("V 16k tail 4", 0.020, v("v16t4_heps") - v("v16t4_ce")), ("V 24k 800 st", 0.035, v("v24s800_heps") - v("v24s800_ce")),
     ("V 10k 800 st", 0.021, v("v10s800_heps") - v("v10s800_ce")), ("V 16k width 64", 0.026, v("v16w64_heps") - v("v16w64_ce")),
     ("V 6k b64 2x tok", 0.011, v("v6b64s400_heps") - v("v6b64s400_ce")), ("V 16k b128", 0.009, v("v16b128s100_heps") - v("v16b128s100_ce"))]
fig, ax = plt.subplots(figsize=(5, 4.4))
pp = np.array([p for _, p, _ in P]); mm = np.array([m for _, _, m in P])
ok = np.abs(pp - mm) <= 0.005
ax.plot([0, 0.045], [0, 0.045], color=MUTED, lw=1); ax.fill_between([0, 0.045], [-0.005, 0.040], [0.005, 0.050], color=GRID, alpha=0.6, lw=0)
ax.scatter(pp[ok], mm[ok], color=S1, s=40, zorder=3, label="within 0.005")
ax.scatter(pp[~ok], mm[~ok], color=S2, marker="D", s=40, zorder=3, label="miss")
for (n, p, m), o in zip(P, ok):
    if not o: ax.annotate(n, (p, m), textcoords="offset points", xytext=(5, -3), fontsize=8, color=INK)
ax.set_xlim(0, 0.045); ax.set_ylim(0, 0.045); ax.set_xlabel("predicted gain (written before the run)"); ax.set_ylabel("measured gain")
ax.set_title(f"Predictions: {ok.sum()} of {len(P)} within 0.005", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig_predictions.png")
for (n, p, m) in P: print(n, p, round(m, 4))
