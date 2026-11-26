"""Main simulation study for the paper.

All non-oracle estimators estimate boundary separation a and non-decision
time ter from the data.

  oracle     : OU fit to the true session drifts
  ez_naive   : EZ per session; OU fit treating EZ drifts as true
  ez_const   : EZ per session; OU fit with a constant measurement-error term
  ml2s_naive : shared (a, ter) by fixed-effect ML; OU fit to session drift MLEs
  ml2s_prop  : same, with session variance 1/S_j propagated
  exact      : joint ML from the exact collapsed likelihood

Usage: python main_study.py [reps] [outdir]
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import DesignParams, OUDDMParams, ez_by_occasion, fit_ou, simulate_study
from ouddm.estimate import SessionData, fit_exact, fit_two_stage_ml

TRUE = OUDDMParams()
TRIALS = [20, 40, 80, 160]
HL_TRUE = np.log(2) / TRUE.theta
# upper bound of log theta used by each estimator's optimizer (lower bound is -6 for all)
UPPER = {"oracle": 3.0, "ez_naive": 3.0, "ez_const": 3.0, "ml2s_naive": 3.0, "ml2s_prop": 3.0, "exact": 4.0}


def one_rep(args):
    n_trials, rep = args
    rng = np.random.default_rng([n_trials, rep, 7031])
    sim = simulate_study(TRUE, DesignParams(n_persons=50, n_occasions=10, n_trials=n_trials), rng)
    tr, occ = sim["trials"], sim["occasions"]
    data = SessionData.from_trials(tr)
    est = ez_by_occasion(tr)
    fits = {
        "oracle": fit_ou(occ["person"], occ["time"], occ["v_true"], se=True).as_dict(),
        "ez_naive": fit_ou(est["person"], est["time"], est["v_hat"]).as_dict(),
        "ez_const": fit_ou(est["person"], est["time"], est["v_hat"], measurement_error=True, se=True).as_dict(),
    }
    f, nu = fit_two_stage_ml(tr, propagate=False, data=data)
    fits["ml2s_naive"] = {**f.as_dict(), **nu}
    f, nu = fit_two_stage_ml(tr, propagate=True, data=data, se=True)
    fits["ml2s_prop"] = {**f.as_dict(), **nu}
    fits["exact"] = fit_exact(tr, data=data).as_dict()
    out = []
    for name, d in fits.items():
        d = dict(d)
        d.update(estimator=name, n_trials=n_trials, rep=rep)
        out.append(d)
    return out


def summarize(df):
    rows = []
    for (n, e), g in df.groupby(["n_trials", "estimator"]):
        hl = g["halflife"].to_numpy()
        lhl = np.log(hl)
        r = {"n_trials": n, "estimator": e, "reps": len(g),
             "hl_median": np.median(hl), "hl_q05": np.quantile(hl, .05), "hl_q95": np.quantile(hl, .95),
             "log_hl_bias": np.mean(lhl) - np.log(HL_TRUE),
             "log_hl_rmse": np.sqrt(np.mean((lhl - np.log(HL_TRUE)) ** 2)),
             "sigma_median": g["sigma"].median(),
             "sigma_rmse": np.sqrt(np.mean((g["sigma"] - TRUE.sigma) ** 2)),
             "theta_at_upper": int((np.log(g["theta"]) >= UPPER[e] - 1e-6).sum()),
             "theta_at_lower": int((np.log(g["theta"]) <= -6.0 + 1e-6).sum()),
             "optimizer_converged": int(g["converged"].astype(bool).sum())}
        ok = g["se_log_theta"].notna() & np.isfinite(g["se_log_theta"])
        if ok.any():
            gg = g[ok]
            r["coverage95"] = np.mean((gg["hl_lo"] <= HL_TRUE) & (HL_TRUE <= gg["hl_hi"]))
            r["ci_reps"] = int(ok.sum())   # denominator of coverage95
            r["median_ci_ratio"] = np.median(gg["hl_hi"] / gg["hl_lo"])
        if "a" in g and g["a"].notna().any():
            r["a_median"] = g["a"].median()
            r["ter_median"] = g["ter"].median()
        rows.append(r)
    order = {k: i for i, k in enumerate(["oracle", "ez_naive", "ez_const", "ml2s_naive", "ml2s_prop", "exact"])}
    return pd.DataFrame(rows).sort_values(["n_trials", "estimator"],
                                          key=lambda c: c.map(order) if c.name == "estimator" else c)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, r) for n in TRIALS for r in range(reps)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = [x for chunk in ex.map(one_rep, jobs, chunksize=2) for x in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "main_reps.csv", index=False)
    s = summarize(df)
    s.to_csv(outdir / "main_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; true half-life {HL_TRUE:.3f}")
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
