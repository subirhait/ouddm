"""Figure: half-life (ln2/theta) and volatility estimates by estimator and
trials per session. Fits at the theta bound are counted above each box.
Usage: python step1_figure.py results_dir"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ouddm import OUDDMParams

TRUE = OUDDMParams()
BOUND = np.exp(3.0)
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
df = pd.read_csv(d / "study_reps.csv")
df["halflife"] = np.log(2) / df["theta"]

est_order = ["oracle", "naive", "corrected", "exact"]
labels = {"oracle": "Oracle (true drift)", "naive": "Two-stage EZ, naive",
          "corrected": "Two-stage EZ, constant-error", "exact": "Exact collapsed likelihood"}
colors = {"oracle": "#8c8c8c", "naive": "#c0392b", "corrected": "#d68910", "exact": "#2471a3"}
trials = sorted(df["n_trials"].unique())
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, par, truth, ylab, logy in [
        (axes[0], "halflife", np.log(2) / TRUE.theta, "Drift half-life, ln2/θ (days)", True),
        (axes[1], "sigma", TRUE.sigma, "Volatility σ", False)]:
    for k, est in enumerate(est_order):
        data = [df.loc[(df.n_trials == n) & (df.estimator == est), par].to_numpy() for n in trials]
        pos = np.arange(len(trials)) + (k - 1.5) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        whis=(5, 95), showfliers=False,
                        medianprops={"color": "black", "linewidth": 1})
        for b in bp["boxes"]:
            b.set_facecolor(colors[est]); b.set_alpha(0.8); b.set_edgecolor(colors[est])
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color(colors[est])
        if par == "halflife" and est == "naive":
            for p, n in zip(pos, trials):
                nb = int((df.loc[(df.n_trials == n) & (df.estimator == est), "theta"] >= 0.99 * BOUND).sum())
                if nb:
                    ax.text(p, 0.045, f"{nb}", ha="center", va="bottom", fontsize=7, color=colors[est])
    ax.axhline(truth, color="black", linestyle="--", linewidth=1)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.03, 12)
    ax.set_xticks(np.arange(len(trials)), [str(n) for n in trials])
    ax.set_xlabel("Trials per session")
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].text(0.01, 0.01, "Numbers: naive fits at the θ bound (half-life ≈ 0.03 d)",
             transform=axes[0].transAxes, fontsize=7, color="#555")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[e], alpha=0.8) for e in est_order]
fig.legend(handles, [labels[e] for e in est_order], loc="outside upper center", ncol=4, frameon=False)
fig.text(0.5, -0.02, "Boxes: 25–75%; whiskers: 5–95%. 100 replications, 50 persons × 10 sessions.",
         ha="center", fontsize=8, color="#555")
fig.savefig(d / "study_figure.png", dpi=200, bbox_inches="tight")
print("saved", d / "study_figure.png")
