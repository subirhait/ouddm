"""Simulation study: recovery of OU dynamics (reversion speed theta and
drift half-life ln2/theta) from short repeated DDM sessions.

Estimators (all fit the same hierarchical OU model for the drift):
  oracle    : true session drift rates (benchmark)
  naive     : EZ estimates treated as true drift rates
  corrected : EZ estimates + constant white-noise measurement-error term
  exact     : exact collapsed likelihood of all choices and RTs, latent
              drifts integrated out (boundary a and ter known)

Usage: python step1_bias_study.py [reps] [outdir]
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import DesignParams, OUDDMParams, fit_collapsed, oracle, simulate_study, two_stage

TRUE = OUDDMParams()
TRIALS = [20, 40, 80, 160]
THETA_UPPER = np.exp(3.0)   # optimizer bound on theta


def one_rep(args):
    n_trials, rep = args
    rng = np.random.default_rng([n_trials, rep, 2026])
    design = DesignParams(n_persons=50, n_occasions=10, n_trials=n_trials)
    sim = simulate_study(TRUE, design, rng)
    fits = {"oracle": oracle(sim["occasions"])}
    fits["naive"], est = two_stage(sim["trials"], measurement_error=False)
    fits["corrected"], _ = two_stage(sim["trials"], measurement_error=True, occasion_estimates=est)
    fits["exact"], _ = fit_collapsed(sim["trials"], TRUE.a, TRUE.ter)
    err_sd = float(np.nanstd(est["v_hat"].to_numpy() - sim["occasions"]["v_true"].to_numpy()))
    out = []
    for name, f in fits.items():
        d = f.as_dict()
        d.update(estimator=name, n_trials=n_trials, rep=rep, stage1_error_sd=err_sd,
                 n_timeout=sim["n_timeout"])
        out.append(d)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    hl_true = np.log(2) / TRUE.theta
    rows = []
    for (n, est), g in df.groupby(["n_trials", "estimator"]):
        th = g["theta"].to_numpy()
        hl = np.log(2) / th
        rows.append({
            "n_trials": n, "estimator": est, "reps": len(g),
            "theta_median": np.median(th),
            "theta_q05": np.quantile(th, 0.05), "theta_q95": np.quantile(th, 0.95),
            "theta_rmse": np.sqrt(np.mean((th - TRUE.theta) ** 2)),
            "halflife_median": np.median(hl),
            "halflife_q05": np.quantile(hl, 0.05), "halflife_q95": np.quantile(hl, 0.95),
            "log_halflife_rmse": np.sqrt(np.mean((np.log(hl) - np.log(hl_true)) ** 2)),
            "sigma_median": g["sigma"].median(),
            "sigma_rmse": np.sqrt(np.mean((g["sigma"] - TRUE.sigma) ** 2)),
            "at_theta_bound": int((th >= 0.99 * THETA_UPPER).sum()),
            "reported_converged": int(g["converged"].sum()),
        })
    out = pd.DataFrame(rows)
    order = {"oracle": 0, "naive": 1, "corrected": 2, "exact": 3}
    return out.sort_values(["n_trials", "estimator"],
                           key=lambda c: c.map(order) if c.name == "estimator" else c)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, r) for n in TRIALS for r in range(reps)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = [row for chunk in ex.map(one_rep, jobs, chunksize=4) for row in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "study_reps.csv", index=False)
    summ = summarize(df)
    summ.to_csv(outdir / "study_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; true theta = {TRUE.theta}, "
          f"true half-life = {np.log(2) / TRUE.theta:.3f} days; "
          f"timed-out trials total = {df.drop_duplicates(['n_trials', 'rep'])['n_timeout'].sum()}")
    cols = ["n_trials", "estimator", "theta_median", "theta_q05", "theta_q95", "theta_rmse",
            "halflife_median", "halflife_q05", "halflife_q95", "log_halflife_rmse", "at_theta_bound"]
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(summ[cols].to_string(index=False))


if __name__ == "__main__":
    main()
