"""Simulation study 4: across-trial drift variability (sv).

Data are simulated with sv = 0 or sv = 1 (on the s = 1 scale). Each data set is
fitted with the exact likelihood with and without sv. Outcomes: half-life
recovery (bias and RMSE of log half-life, coverage of the 95% Wald interval)
and recovery of sv.

Usage: python sv_study.py <n_reps> <outdir>
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import ouddm
from ouddm.design_model import DesignData, fit_design

N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
THETA = 0.5
DESIGN = ouddm.DesignParams(n_persons=50, n_occasions=10, n_trials=40)


def one(args):
    sv_true, rep = args
    rng = np.random.default_rng(10_000 * int(10 * sv_true) + rep)
    sim = ouddm.simulate_study(ouddm.OUDDMParams(theta=THETA, sv=sv_true), DESIGN, rng)
    tr = sim["trials"].copy()
    tr["upper"] = tr["correct"].astype(int)
    tr["stim"] = 1.0
    out = []
    for fit_sv in (False, True):
        d = DesignData.from_trials(tr, "stim", sv=fit_sv)
        f = fit_design(d, se=True)
        lo, hi = f.halflife_ci
        out.append({"sv_true": sv_true, "rep": rep, "fit_sv": fit_sv, "halflife": f.halflife,
                    "covered": bool(lo <= np.log(2) / THETA <= hi), "sv_hat": f.params["sv"],
                    "a_hat": float(np.exp(f.params["log_a:intercept"])), "loglik": f.loglik,
                    "converged": f.converged})
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(sv, r) for sv in (0.0, 1.0) for r in range(N_REPS)]
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = [row for res in ex.map(one, jobs) for row in res]
    reps = pd.DataFrame(rows)
    reps.to_csv(OUT / "sv_reps.csv", index=False)
    true_hl = np.log(2) / THETA
    reps["err"] = np.log(reps.halflife) - np.log(true_hl)
    summ = (reps.groupby(["sv_true", "fit_sv"])
            .agg(bias_log_hl=("err", "mean"), rmse_log_hl=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                 coverage=("covered", "mean"), sv_hat_mean=("sv_hat", "mean"), sv_hat_sd=("sv_hat", "std"),
                 a_hat_mean=("a_hat", "mean"), n=("rep", "size")).reset_index())
    lr = reps.pivot_table(index=["sv_true", "rep"], columns="fit_sv", values="loglik")
    lrstat = 2 * (lr[True] - lr[False])
    summ["lr_sv_median"] = summ.sv_true.map(lrstat.groupby(level=0).median())
    summ.to_csv(OUT / "sv_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
