"""Simulation study 3: task conditions and practice.

Design modeled on the smartphone Color Shapes study (Kim et al., 2025):
68 persons, 8 days, two task versions per day about one hour apart,
60 trials per version. Trial drift = OU state + x'beta with

  condition : -0.5 / +0.5 for the two task versions (alternating order
              across days), effect beta_cond on drift
  practice  : day number centered at its mean, effect beta_prac per day

Scenarios: condition effect only, practice effect only, both.
For each data set the exact likelihood is fitted without and with the
covariates. True half-life 1.39 days.

Usage: python condition_study.py [reps] [outdir]
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import DesignParams, OUDDMParams, simulate_study
from ouddm.estimate import fit_exact

TRUE = OUDDMParams()
PERSONS, DAYS, TRIALS = 68, 8, 60
GAP = 1.0 / 24.0          # one hour between the two versions
BETA_COND, BETA_PRAC = 0.4, 0.08
SCEN = {"condition": (BETA_COND, 0.0), "practice": (0.0, BETA_PRAC), "both": (BETA_COND, BETA_PRAC)}


def design():
    day = np.repeat(np.arange(DAYS), 2).astype(float)
    times = day + np.tile([0.0, GAP], DAYS)
    order = np.where(np.arange(DAYS) % 2 == 0, 1.0, -1.0)            # alternate which version comes first
    cond = np.column_stack([order, -order]).ravel() * 0.5
    prac = day - day.mean()
    return times, np.column_stack([cond, prac])


def one_rep(args):
    scen, rep = args
    bc, bp = SCEN[scen]
    times, X = design()
    rng = np.random.default_rng([len(scen), rep, 515])
    sim = simulate_study(TRUE, DesignParams(n_persons=PERSONS, n_occasions=2 * DAYS, n_trials=TRIALS),
                         rng, times=times, x_session=X, beta=[bc, bp])
    tr = sim["trials"]
    out = []
    for label, cov in [("ignore covariates", ()), ("model covariates", ("x0", "x1"))]:
        f = fit_exact(tr, covariates=cov)
        d = f.as_dict()
        d.update(scenario=scen, rep=rep, analysis=label)
        out.append(d)
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r) for s in SCEN for r in range(reps)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = [x for chunk in ex.map(one_rep, jobs, chunksize=1) for x in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "condition_reps.csv", index=False)
    hl = np.log(2) / TRUE.theta
    summ = []
    for (s, an), g in df.groupby(["scenario", "analysis"], sort=False):
        ok = g["se_log_theta"].notna()
        r = {"scenario": s, "analysis": an, "reps": len(g),
             "hl_median": g["halflife"].median(), "hl_q05": g["halflife"].quantile(.05),
             "hl_q95": g["halflife"].quantile(.95),
             "log_hl_rmse": np.sqrt(np.mean((np.log(g["halflife"]) - np.log(hl)) ** 2)),
             "coverage95": np.mean((g.loc[ok, "hl_lo"] <= hl) & (hl <= g.loc[ok, "hl_hi"])),
             "ci_reps": int(ok.sum()),
             "sigma_median": g["sigma"].median(),
             "theta_at_upper": int((g["theta_at_bound"] == "upper").sum()),
             "theta_at_lower": int((g["theta_at_bound"] == "lower").sum()),
             "optimizer_converged": int(g["converged"].astype(bool).sum())}
        if "beta_x0" in g and g["beta_x0"].notna().any():
            r.update(beta_cond_median=g["beta_x0"].median(), beta_prac_median=g["beta_x1"].median(),
                     beta_cond_cov=np.mean(np.abs(g["beta_x0"] - SCEN[s][0]) <= 1.96 * g["se_beta_x0"]),
                     beta_prac_cov=np.mean(np.abs(g["beta_x1"] - SCEN[s][1]) <= 1.96 * g["se_beta_x1"]))
        summ.append(r)
    s = pd.DataFrame(summ)
    s.to_csv(outdir / "condition_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; true half-life {hl:.3f}")
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
