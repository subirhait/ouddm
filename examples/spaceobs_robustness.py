"""Space Observer: sample-rule robustness and a final-session holdout check.

All models use person-specific practice slopes, independent session deviations
and across-trial drift variability (sv), as in Table 8.

robust A   drop trials without a confidence rating
robust B   rule A, then drop sessions that start within 1 hour of the
           previous retained session of the same person
           (the 20-trial session rule is applied after the trial rule)
holdout    main sample; for each person with at least 2 sessions, the final
           session is held out; models are fit to the other sessions and
           predict held-out choices (RTs not used); paired bootstrap over persons

Usage:
  python -I spaceobs_robustness.py robust A <space-observer-dta.csv> <task_data.csv> <outdir>
  python -I spaceobs_robustness.py robust B <space-observer-dta.csv> <task_data.csv> <outdir>
  python -I spaceobs_robustness.py holdout  <space-observer-dta.csv> <task_data.csv> <outdir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouddm.design_model import DesignData, fit_design, predict_choices  # noqa: E402
from space_observer_application import load  # noqa: E402

BASE = dict(drift=("log_int_c", "game_c"), nugget=True, slope="game_c")


def apply_rule(d, rule):
    d = d.copy()
    if rule in ("A", "B"):
        d = d[d.confidence.notna()]
    n = d.groupby(["person", "occasion"]).trial.transform("size")
    d = d[n >= 20]
    if rule == "B":
        st = d.groupby(["person", "occasion"]).time.first().reset_index().sort_values(["person", "time"])
        keep = []
        for _, g in st.groupby("person"):
            last = -np.inf
            for o, t in zip(g.occasion, g.time):
                if t - last >= 1.0 / 24.0:
                    keep.append((g.person.iloc[0], o)); last = t
        d = d.set_index(["person", "occasion"]).loc[keep].reset_index()
    first = d.groupby("person").time.transform("min")
    d["time"] = d.time - first
    return d


def describe(d):
    return {"persons": d.person.nunique(), "sessions": d.groupby(["person", "occasion"]).ngroups,
            "trials": len(d)}


def fit(d, kern, sv=True):
    t0 = time.time()
    dd = DesignData.from_trials(d, "stim", **BASE, kernel=kern, sv=sv)
    f = fit_design(dd, se=False)
    return dd, f, time.time() - t0


def robust(d, rule, outdir):
    d = apply_rule(d, rule)
    info = describe(d)
    print(rule, info, flush=True)
    rows = []
    for kern in ("none", "ou", "se"):
        _, f, sec = fit(d, kern)
        rows.append({"rule": rule, **info, "kernel": kern, "loglik": f.loglik, "aic": f.aic,
                     "halflife_days": f.halflife if kern == "ou" else np.nan, "ell_days": f.params["ell"],
                     "sv_hat": f.params["sv"], "sd_stat": f.params["sd_stat"], "converged": f.converged,
                     "seconds": sec})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res["lr_vs_none"] = 2 * (res.loglik - res.loc[res.kernel == "none", "loglik"].iloc[0])
    res["daic"] = res.aic - res.aic.min()
    res.to_csv(outdir / f"spaceobs_robust_{rule}.csv", index=False)
    print(res.round(3).to_string(index=False), flush=True)


def holdout(d, outdir, n_boot=2000, seed=2026):
    d = apply_rule(d, "main")
    ns = d.groupby("person").occasion.nunique()
    d = d[d.person.isin(ns[ns >= 2].index)]
    last = d.groupby("person").occasion.transform("max")
    train, test = d[d.occasion < last], d[d.occasion == last]
    print("train", describe(train), "test", describe(test), flush=True)
    models = [("none", False), ("none", True), ("ou", True), ("se", True)]
    rows, loss = [], {}
    for kern, sv in models:
        name = f"{kern}|{'sv' if sv else 'nosv'}"
        dtr, f, sec = fit(train, kern, sv)
        dte = DesignData.from_trials(test, "stim", **BASE, kernel=kern, sv=sv)
        pr = np.clip(predict_choices(f.x, dtr, dte), 1e-12, 1 - 1e-12)
        ll = -(dte.upper * np.log(pr) + (1 - dte.upper) * np.log1p(-pr))
        loss[name] = pd.Series(ll).groupby(dte.person[dte.sess]).agg(["sum", "size"])
        rows.append({"model": name, "train_loglik": f.loglik, "train_aic": f.aic,
                     "halflife_days": f.halflife if kern == "ou" else np.nan, "sv_hat": f.params["sv"],
                     "test_log_loss": ll.mean(), "test_trials": ll.size, "test_persons": len(loss[name]),
                     "seconds": sec, "converged": f.converged})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res["train_daic"] = res.train_aic - res.train_aic.min()
    res.to_csv(outdir / "spaceobs_holdout.csv", index=False)
    # paired bootstrap over persons of the difference in mean held-out log loss
    rng = np.random.default_rng(seed)
    persons = loss["none|sv"].index.to_numpy()
    n_tr = loss["none|sv"].loc[persons, "size"].to_numpy()
    tot = {k: v.loc[persons, "sum"].to_numpy() for k, v in loss.items()}
    idx = rng.integers(0, persons.size, size=(n_boot, persons.size))
    comps = [("ou|sv", "none|sv"), ("se|sv", "none|sv"), ("se|sv", "ou|sv"), ("none|sv", "none|nosv")]
    brows = []
    for a_, b_ in comps:
        diff = (tot[a_] - tot[b_])
        est = diff.sum() / n_tr.sum()
        bs = diff[idx].sum(1) / n_tr[idx].sum(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        brows.append({"model": a_, "reference": b_, "diff_log_loss": est, "lo95": lo, "hi95": hi,
                      "persons": persons.size, "n_boot": n_boot})
    b = pd.DataFrame(brows)
    b.to_csv(outdir / "spaceobs_holdout_bootstrap.csv", index=False)
    print(res.round(5).to_string(index=False)); print(b.to_string(index=False), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "robust":
        rule, tp, sp, outdir = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    else:
        rule, tp, sp, outdir = None, Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    outdir.mkdir(parents=True, exist_ok=True)
    d, _ = load(tp, sp)
    robust(d, rule, outdir) if mode == "robust" else holdout(d, outdir)
