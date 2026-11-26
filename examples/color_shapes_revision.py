"""Color Shapes: refined profile likelihood and sensitivity analyses.

part A: main model M3n
          - fit, dense profile over log theta from theta = 1e-5 to 1e5 per day
            (half-lives from about 1 second to 190 years), with the fitted
            maximum as the reference, and interval crossings by root-finding
          - the limit model shared by both ends (random intercept + session
            deviations, no OU), fitted directly
        S1: M3n + person-specific practice slopes (correlated with the intercept),
            and its no-OU limit model
part B: S2: M3n on long-deadline (10,000 ms) versions only
        S3: M3n + study time x probe type interaction on drift
part C: M2n (no practice predictor), refined interval for Table 5

Usage: python -I color_shapes_revision.py <data_merged.csv> <A|B|C> [outdir]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_shapes_application as app  # noqa: E402
from ouddm.design_model import DesignData, fit_design, profile_interval, profile_theta  # noqa: E402

BASE = dict(boundary=("ChoiceUrgencyCoded",), bias=("ProbOfChangeCoded",),
            drift=tuple(app.MANIP) + ("day_c",), nugget=True)
DENSE = np.log(np.geomspace(1e-5, 1e5, 31))
COARSE = np.log(np.geomspace(1e-4, 1e4, 17))


def analyze(label, data, grid, limit_data=None):
    t0 = time.time()
    fit = fit_design(data)
    prof = profile_theta(data, grid, x_start=fit.x)
    # add the fitted maximum itself to the profile table
    prof = pd.concat([prof, pd.DataFrame([{"log_theta": np.log(fit.params["theta"]), "theta": fit.params["theta"],
                                           "halflife_days": fit.halflife, "loglik": fit.loglik,
                                           "tau": fit.params["tau"], "sd_stat": fit.params["sd_stat"],
                                           "omega": fit.params["omega"], "slope_sd": fit.params["slope_sd"]}])])
    prof = prof.sort_values("log_theta").reset_index(drop=True)
    iv = profile_interval(prof, ref_loglik=fit.loglik, d=data, fit=fit)
    row = {"analysis": label, "loglik": fit.loglik, "aic": fit.aic, "n_params": fit.n_params,
           "halflife_days": fit.halflife, "tau": fit.params["tau"], "sd_stat": fit.params["sd_stat"],
           "omega": fit.params["omega"], "slope_sd": fit.params["slope_sd"], "rho": fit.params["rho"],
           "converged": fit.converged, "theta_at_bound": fit.theta_at_bound, **iv}
    if limit_data is not None:
        lim = fit_design(limit_data, se=False)
        row.update(limit_loglik=lim.loglik, limit_tau=lim.params["tau"], limit_omega=lim.params["omega"],
                   lr_vs_limit=2 * (fit.loglik - lim.loglik))
    row["seconds"] = time.time() - t0
    coefs = pd.DataFrame({"analysis": label, "param": list(fit.se.keys()),
                          "estimate": [fit.params[k] for k in fit.se], "se": list(fit.se.values())})
    prof["analysis"] = label
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row, prof, coefs


def main():
    d = app.load(Path(sys.argv[1]))
    part = sys.argv[2]
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    rows, profs, coefs = [], [], []
    if part == "A":
        data = DesignData.from_trials(d, "stim", **BASE)
        lim = DesignData.from_trials(d, "stim", **{**BASE, "ou": False})
        for r, p, c in [analyze("M3n main", data, DENSE, lim)]:
            rows.append(r); profs.append(p); coefs.append(c)
        sdat = DesignData.from_trials(d, "stim", **BASE, slope="day_c")
        slim = DesignData.from_trials(d, "stim", **{**BASE, "ou": False}, slope="day_c")
        r, p, c = analyze("S1 practice slopes", sdat, COARSE, slim)
        rows.append(r); profs.append(p); coefs.append(c)
    elif part == "C":
        spec = {**BASE, "drift": tuple(app.MANIP)}
        r, p, c = analyze("M2n no practice", DesignData.from_trials(d, "stim", **spec), COARSE,
                          DesignData.from_trials(d, "stim", **{**spec, "ou": False}))
        rows.append(r); profs.append(p); coefs.append(c)
    else:
        long = d[d.ChoiceUrgencyCoded == 0].copy()
        spec = dict(boundary=(), bias=("ProbOfChangeCoded",),
                    drift=("StudyTimeCoded_c", "ProbOfChangeCoded_c", "ProbeTypeCoded_c", "day_c"), nugget=True)
        r, p, c = analyze("S2 long deadline only", DesignData.from_trials(long, "stim", **spec), COARSE,
                          DesignData.from_trials(long, "stim", **{**spec, "ou": False}))
        rows.append(r); profs.append(p); coefs.append(c)
        d["st_x_probe"] = d["StudyTimeCoded_c"] * d["ProbeTypeCoded_c"]
        spec3 = {**BASE, "drift": BASE["drift"] + ("st_x_probe",)}
        r, p, c = analyze("S3 study time x probe", DesignData.from_trials(d, "stim", **spec3), COARSE,
                          DesignData.from_trials(d, "stim", **{**spec3, "ou": False}))
        rows.append(r); profs.append(p); coefs.append(c)
    pd.DataFrame(rows).to_csv(outdir / f"colorshapes_revision_{part}.csv", index=False)
    pd.concat(profs).to_csv(outdir / f"colorshapes_revision_profile_{part}.csv", index=False)
    pd.concat(coefs).to_csv(outdir / f"colorshapes_revision_coefs_{part}.csv", index=False)


if __name__ == "__main__":
    main()
