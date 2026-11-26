"""Fixed-budget design study: 400 trials per person over a 14-day window,
split into K sessions, with even or burst (paired) spacing, for three true
half-lives. Compares the design calculator's SE of log half-life with the
simulation spread of the exact joint ML estimate.

Usage: python design_study.py [reps] [outdir]
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import DesignParams, OUDDMParams, simulate_study
from ouddm.design import design_information, schedule
from ouddm.estimate import fit_exact

BUDGET, WINDOW, PERSONS = 400, 14.0, 50
HALFLIVES = [0.5, np.log(2) / 0.5, 4.0]
SESSIONS = [4, 10, 20, 40]
SPACINGS = ["even", "burst"]
STAT_SD = 0.35


def params(hl):
    theta = np.log(2) / hl
    return OUDDMParams(theta=theta, sigma=STAT_SD * np.sqrt(2 * theta))


def one_rep(args):
    hl, K, sp, rep = args
    P = params(hl)
    t = schedule(K, WINDOW, sp)
    rng = np.random.default_rng([int(hl * 1000), K, SPACINGS.index(sp), rep, 99])
    sim = simulate_study(P, DesignParams(n_persons=PERSONS, n_occasions=K, n_trials=BUDGET // K), rng, times=t)
    f = fit_exact(sim["trials"], se=False)
    return {"halflife_true": hl, "sessions": K, "spacing": sp, "rep": rep,
            "theta_hat": f.theta, "a_hat": f.a, "ter_hat": f.ter,
            "converged": f.converged, "at_bound": f.at_bound}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(hl, K, sp, r) for hl in HALFLIVES for K in SESSIONS for sp in SPACINGS for r in range(reps)]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one_rep, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "design_reps.csv", index=False)

    out = []
    for (hl, K, sp), g in df.groupby(["halflife_true", "sessions", "spacing"]):
        P = params(hl)
        calc = design_information(schedule(K, WINDOW, sp), BUDGET // K, P.a, P.mu0, P.tau, P.theta,
                                  P.sigma, PERSONS)
        calc_choice = design_information(schedule(K, WINDOW, sp), BUDGET // K, P.a, P.mu0, P.tau,
                                         P.theta, P.sigma, PERSONS, rt=False)
        err = np.log(g["theta_hat"]) - np.log(P.theta)   # = -(log hl error)
        q25, q75 = np.quantile(err, [0.25, 0.75])
        out.append({"halflife_true": hl, "sessions": K, "trials_per_session": BUDGET // K, "spacing": sp,
                    "calc_se": calc["se_log_halflife"], "calc_se_oracle": calc["se_oracle"],
                    "calc_se_choice_only": calc_choice["se_log_halflife"],
                    "sim_sd": err.std(ddof=1), "sim_robust_sd": (q75 - q25) / 1.349,
                    "sim_rmse": np.sqrt(np.mean(err ** 2)), "sim_bias": err.mean(),
                    "theta_at_upper": int((g["at_bound"] == "upper").sum()),
                    "theta_at_lower": int((g["at_bound"] == "lower").sum()),
                    "optimizer_converged": int(g["converged"].astype(bool).sum()),
                    "reps": len(g)})
    s = pd.DataFrame(out)
    s.to_csv(outdir / "design_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
