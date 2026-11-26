"""Empirical application: Color Shapes smartphone study (Kim et al., 2025).

Data: data_merged.csv from the authors' OSF repository (https://osf.io/ty4uf/,
"Files for June 11 2025 submission/2_processing/"), which already excludes
trials without a response and RTs < 200 ms or > 7000 ms.

Coding
  upper boundary = "different" response (button_pressed = 1); stimulus sign
  s = +1 for "different" trials, -1 for "same" trials (stimulus coding, as
  in the authors' JAGS model). Session = one task version (60 trials); two
  versions per day, done back to back. Session time = day - 1, plus the
  duration of the first version (task_time, minutes) for the second version.

Models (all fitted with the exact likelihood, design_model.fit_design)
  M0  no design: one boundary, one start point, no drift predictors
  M1  nuisance design: boundary by choice urgency, start point by
      probability of change, no drift predictors
  M2  M1 + the four task manipulations as drift predictors
  M3  M2 + practice (day, centered) as a drift predictor
  M2n, M3n  M2 and M3 plus independent session-level drift deviations
            (variance omega^2). Each person completes each of the 16 task
            versions once, so person-specific version effects appear as
            session-specific deviations; the nugget absorbs them.
Also: two-stage EZ-diffusion, naive (accuracy coding), for reference.

Usage: python -I color_shapes_application.py <data_merged.csv> [outdir]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import ez_by_occasion, fit_ou
from ouddm.design_model import DesignData, fit_design


def load(path):
    d = pd.read_csv(path, low_memory=False)
    d = d.rename(columns={"participant_id": "person"})
    d["task2"] = (d["sequential_task_order"] == "task2").astype(int)
    d["occasion"] = d["day"] * 2 + d["task2"]
    first = (d[d.task2 == 0].groupby(["person", "day"])["task_time"].first().rename("t1_dur"))
    d = d.merge(first, on=["person", "day"], how="left")
    d["t1_dur"] = d["t1_dur"].fillna(d["task_time"].median())
    d["time"] = (d["day"] - 1) + d["task2"] * d["t1_dur"] / 1440.0
    d["rt"] = d["response_time"] / 1000.0
    d["upper"] = d["button_pressed"].astype(int)
    d["stim"] = np.where(d["trial_type"] == 1, 1.0, -1.0)
    d["correct"] = d["accuracy"].astype(int)
    for c in ["StudyTimeCoded", "ChoiceUrgencyCoded", "ProbOfChangeCoded", "ProbeTypeCoded"]:
        d[c + "_c"] = d[c] - 0.5
    d["day_c"] = d["day"] - 4.5
    return d


MANIP = ["StudyTimeCoded_c", "ChoiceUrgencyCoded_c", "ProbOfChangeCoded_c", "ProbeTypeCoded_c"]
MODELS = {
    "M0 no design": dict(boundary=(), bias=(), drift=()),
    "M1 nuisance design": dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",), drift=()),
    "M2 + manipulations on drift": dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",), drift=tuple(MANIP)),
    "M3 + practice on drift": dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",), drift=tuple(MANIP) + ("day_c",)),
    "M2n M2 + session deviations": dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",), drift=tuple(MANIP), nugget=True),
    "M3n M3 + session deviations": dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",), drift=tuple(MANIP) + ("day_c",), nugget=True),
}


def main():
    path = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    d = load(path)
    desc = {"persons": d.person.nunique(), "sessions": d.groupby(["person", "occasion"]).ngroups,
            "trials": len(d), "accuracy": d.correct.mean(), "median_rt": d.rt.median()}
    print(desc)
    rows, coefs = [], []
    for name, spec in MODELS.items():
        t0 = time.time()
        data = DesignData.from_trials(d, "stim", **spec)
        f = fit_design(data)
        lo, hi = f.halflife_ci
        rows.append({"model": name, "halflife_days": f.halflife, "hl_lo": lo, "hl_hi": hi,
                     "theta": f.params["theta"], "tau": f.params["tau"], "sigma": f.params["sigma"],
                     "stationary_sd": f.params["sd_stat"], "omega": f.params["omega"],
                     "kappa": f.params["kappa"], "loglik": f.loglik, "n_params": f.n_params, "aic": f.aic,
                     "converged": f.converged, "theta_at_bound": f.theta_at_bound,
                     "seconds": time.time() - t0})
        for k, v in f.params.items():
            coefs.append({"model": name, "param": k, "estimate": v, "se": f.se.get(k, np.nan)})
        print(rows[-1])
    # two-stage EZ, naive (accuracy coding)
    tr = d[["person", "occasion", "time", "rt", "correct"]]
    est = ez_by_occasion(tr)
    ez = fit_ou(est.person, est.time, est.v_hat, se=True)
    lo, hi = ez.halflife_ci()
    rows.append({"model": "EZ two-stage, naive", "halflife_days": ez.halflife, "hl_lo": lo, "hl_hi": hi,
                 "theta": ez.theta, "tau": ez.tau, "sigma": ez.sigma,
                 "stationary_sd": ez.sigma / np.sqrt(2 * ez.theta), "converged": ez.converged,
                 "theta_at_bound": "upper" if np.log(ez.theta) >= 3 - 1e-6 else ""})
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "colorshapes_models.csv", index=False)
    pd.DataFrame(coefs).to_csv(outdir / "colorshapes_coefficients.csv", index=False)
    pd.DataFrame([desc]).to_csv(outdir / "colorshapes_descriptives.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 30):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
