"""Figures for the manuscript.
Usage: python paper_figures.py results_dir out_dir
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ouddm.wiener import information_ratio_choice

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
HL = np.log(2) / 0.5

# ---------------- Figure 2: recovery ----------------
df = pd.read_csv(res / "main_reps.csv")
summ = pd.read_csv(res / "main_summary.csv")
ests = ["oracle", "ez_naive", "ez_const", "ml2s_prop", "exact"]
lab = {"oracle": "Oracle (true drifts)", "ez_naive": "EZ two-stage, naive",
       "ez_const": "EZ two-stage, constant error", "ml2s_prop": "ML two-stage, propagated",
       "exact": "Exact collapsed likelihood"}
col = {"oracle": "#8c8c8c", "ez_naive": "#c0392b", "ez_const": "#d68910",
       "ml2s_prop": "#7d3c98", "exact": "#2471a3"}
trials = sorted(df.n_trials.unique())
w = 0.16
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.7, 1]},
                         constrained_layout=True)
ax = axes[0]
for k, e in enumerate(ests):
    data = [df.loc[(df.n_trials == n) & (df.estimator == e), "halflife"].to_numpy() for n in trials]
    pos = np.arange(len(trials)) + (k - 2) * w
    bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, whis=(5, 95),
                    showfliers=False, medianprops={"color": "black", "linewidth": 1})
    for b in bp["boxes"]:
        b.set_facecolor(col[e]); b.set_alpha(0.85); b.set_edgecolor(col[e])
    for x in bp["whiskers"] + bp["caps"]:
        x.set_color(col[e])
ax.axhline(HL, color="black", ls="--", lw=1)
ax.set_yscale("log"); ax.set_ylim(0.03, 12)
ax.set_xticks(range(len(trials)), [str(n) for n in trials])
ax.set_xlabel("Trials per session"); ax.set_ylabel("Estimated drift half-life (days, log scale)")
ax.set_title("A. Half-life estimates", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for e, mk in [("oracle", "s"), ("ez_const", "^"), ("ml2s_prop", "D"), ("exact", "o")]:
    s = summ[summ.estimator == e].sort_values("n_trials")
    ax.plot(range(len(trials)), s["coverage95"], marker=mk, color=col[e], lw=1.2, ms=5)
ax.axhline(0.95, color="black", ls="--", lw=1)
ax.set_ylim(0.6, 1.0)
ax.set_xticks(range(len(trials)), [str(n) for n in trials])
ax.set_xlabel("Trials per session"); ax.set_ylabel("Coverage of nominal 95% interval")
ax.set_title("B. Interval coverage", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=col[e], alpha=0.85) for e in ests]
fig.legend(handles, [lab[e] for e in ests], loc="outside lower center", ncol=5, frameon=False, fontsize=8)
fig.savefig(out / "figure2_recovery.png", dpi=300)
plt.close(fig)

# ---------------- Figure 1: information ratio ----------------
a = 1.2
v = np.linspace(0, 4.5, 300)
acc = 1 / (1 + np.exp(-a * v))
fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
ax.plot(acc[1:], information_ratio_choice(v[1:], a), color="#2471a3", lw=1.8)
for vv in (1.5, 2.5, 4.0):
    p = 1 / (1 + np.exp(-a * vv)); r = float(information_ratio_choice(vv, a))
    ax.plot(p, r, "o", color="black", ms=4)
    ax.annotate(f"v = {vv}", (p, r), textcoords="offset points", xytext=(-38, -2), fontsize=8)
ax.set_xlabel("Expected accuracy, 1 / (1 + exp(−av))")
ax.set_ylabel("Share of drift information\nretained by choices alone")
ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(out / "figure1_information.png", dpi=300)
plt.close(fig)

# ---------------- Figure 3: design ----------------
ds = Path(res / "design_summary.csv")
if ds.exists():
    d = pd.read_csv(ds)
    hls = sorted(d.halflife_true.unique())
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, axes = plt.subplots(1, len(hls), figsize=(10, 3.4), sharey=True, constrained_layout=True)
    YMAX = 3.5
    for ax, hl in zip(axes, hls):
        for sp, colr, mk in [("even", "#2471a3", "o"), ("burst", "#d68910", "s")]:
            g = d[(d.halflife_true == hl) & (d.spacing == sp)].sort_values("sessions")
            x = np.arange(len(g))
            yc = np.minimum(g["calc_se"].to_numpy(), YMAX)
            ax.plot(x, yc, color=colr, lw=1.4, label=f"{sp}: calculator")
            ax.plot(x, g["sim_sd"], mk, color=colr, ms=5, mfc="white", label=f"{sp}: simulation")
            for xi, v in zip(x, g["calc_se"]):
                if v > YMAX:
                    ax.annotate(f"{v:.1f}", (xi, YMAX), textcoords="offset points", xytext=(6, -10),
                                fontsize=7, color=colr)
        ax.set_yscale("log")
        ax.set_ylim(0.15, YMAX)
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1, 2, 3]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "2", "3"])
        ax.set_xticks(x, [f"{K}×{400 // K}" for K in g["sessions"]])
        ax.set_xlabel("Sessions × trials per session")
        ax.set_title(f"True half-life {hl:.2g} days", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SE of log half-life")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(out / "figure3_design.png", dpi=300)
    plt.close(fig)
print("figures written to", out)

# ---------------- Figure 4: Color Shapes profile likelihood ----------------
# Each curve is referenced to its own fitted maximum (the fit is one of the profile points).
pfs = [res / f"colorshapes_revision_profile_{k}.csv" for k in "ABC"]
if all(f.exists() for f in pfs):
    p = pd.concat([pd.read_csv(f) for f in pfs])
    summ = pd.concat([pd.read_csv(res / f"colorshapes_revision_{k}.csv") for k in "ABC"]).set_index("analysis")
    spec = [("M2n no practice", "No practice predictor (M2n)", "#d68910", "-"),
            ("M3n main", "Common practice slope (M3n)", "#2471a3", "-"),
            ("S1 practice slopes", "Person-specific practice slopes (S1)", "#1e8449", "-"),
            ("S2 long deadline only", "M3n, 10-s window only (S2)", "#2471a3", ":"),
            ("S3 study time x probe", "M3n + study time x probe (S3)", "#7d3c98", "--")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for key, lab_, col, ls in spec:
        g = p[p.analysis == key].sort_values("halflife_days")
        ref = summ.loc[key, "ref_loglik"]
        dev = 2 * (ref - g.loglik)
        main = ls == "-"
        ax.plot(g.halflife_days, dev, ls=ls, lw=1.5 if main else 1.1, color=col, label=lab_,
                marker="o" if main else None, ms=3)
    ax.axhline(3.84, color="black", ls="--", lw=0.9)
    ax.text(2e-5, 4.6, "approx. 95% cut-off (3.84)", fontsize=7)
    ax.set_xscale("log"); ax.set_ylim(-1, 32); ax.set_xlim(1e-5, 1e5)
    ax.text(2e-5, 28.3, "M2n rises to %.0f at both ends" % summ.loc["M2n no practice", "lr_vs_limit"],
            fontsize=7, color="#d68910")
    for x, t in ((1 / 1440, "1 min"), (1 / 24, "1 h"), (1, "1 day"), (8, "8 days")):
        ax.axvline(x, color="0.85", lw=0.7, zorder=0)
        ax.text(x * 1.15, 31, t, fontsize=6.5, color="0.4")
    ax.set_xlabel("Half-life of the OU component (days, log scale)")
    ax.set_ylabel("Profile deviance from the fitted maximum")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.42))
    fig.savefig(out / "figure4_colorshapes_profile.png", dpi=300)
    plt.close(fig)

# ---------------- Figure 5: Space Observer profile likelihood ----------------
sp = [res / "spaceobs_profile_person.csv", res / "spaceobs_profile_person_sv.csv"]
if all(f.exists() for f in sp):
    iv = pd.read_csv(res / "spaceobs_profile_intervals.csv").set_index("model")
    ivsv = pd.read_csv(res / "spaceobs_profile_interval_sv.csv").iloc[0]
    refs = (iv.loc["person|ou|nosv", "ref_loglik"], ivsv.ref_loglik)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for f, ref, lab_, col in ((sp[0], refs[0], "Without drift variability", "#2471a3"),
                              (sp[1], refs[1], "With across-trial drift variability", "#1e8449")):
        g = pd.read_csv(f).sort_values("halflife_days")
        ax.plot(g.halflife_days, 2 * (ref - g.loglik), "-o", ms=3, lw=1.5, color=col, label=lab_)
    ax.axhline(3.84, color="black", ls="--", lw=0.9)
    ax.text(2e-4, 4.4, "approx. 95% cut-off (3.84)", fontsize=7)
    for x, t in ((1 / 24, "1 h"), (1, "1 day"), (14, "14 days")):
        ax.axvline(x, color="0.85", lw=0.7, zorder=0)
        ax.text(x * 1.15, 15.3, t, fontsize=6.5, color="0.4")
    ax.set_xscale("log"); ax.set_ylim(-0.5, 16.5)
    ax.set_xlabel("Half-life of the OU component (days, log scale)")
    ax.set_ylabel("Profile deviance from the fitted maximum")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(out / "figure5_spaceobs_profile.png", dpi=300)
    plt.close(fig)

# ---------------- Figure 6: session schedules (Simulation study 4) ----------------
sf = res / "schedule" / "schedule_summary.csv"
if sf.exists():
    s = pd.read_csv(sf)
    s = s[s.truth != "none"].copy()
    s["hl"] = s.truth.astype(float)
    order = ["alternate", "daily", "twice", "paired"]
    slab = ["Alternate days\n7 × 60", "Daily\n14 × 30", "Twice daily\n28 × 15", "Paired 4–6 h\n14 × 30"]
    cols = {0.25: "#c0392b", 0.65: "#2471a3", 2.0: "#1e8449"}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), constrained_layout=True)
    for ax, (col_, ttl) in zip(axes, (("detect", "Detection (calibrated 5% test)"),
                                       ("resolved", "Interval closed and within a factor of 10"),
                                       ("rmse_log", "RMSE of log half-life"))):
        for hl, g in s.groupby("hl"):
            g = g.set_index("schedule").loc[order]
            y = 1 - g.too_wide if col_ == "resolved" else g[col_]
            ax.plot(range(4), y, "-o", ms=3.5, lw=1.4, color=cols[hl], label=f"{hl:g} days")
        ax.set_xticks(range(4)); ax.set_xticklabels(slab, fontsize=6.5)
        ax.set_title(ttl, fontsize=8)
        if col_ != "rmse_log":
            ax.set_ylim(-0.03, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="True half-life", frameon=False, fontsize=7, title_fontsize=7)
    fig.savefig(out / "figure6_schedules.png", dpi=300)
    plt.close(fig)
