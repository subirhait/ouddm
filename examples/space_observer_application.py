"""Application 2: Space Observer (Marzuki et al., 2026, Nature Communications).

Trial-level data from https://osf.io/x7h5k/ (files space-observer-dta.csv and
task_data.csv). The data are not redistributed.

Models (all with independent session deviations, drift predictors log stimulus
intensity and session number):
  practice: common slope, or person-specific slopes correlated with the person mean
  dynamics: none, OU, random walk (rw), squared-exponential Gaussian process (se)
  plus the best model with across-trial drift variability (sv)
Profile-likelihood intervals for the OU half-life under both practice models.

Usage: python -I space_observer_application.py <space-observer-dta.csv> <task_data.csv> <outdir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm.design_model import DesignData, _names, fit_design, profile_interval, profile_theta


def load(trials_path, sess_path):
    t = pd.read_csv(trials_path)
    s = pd.read_csv(sess_path, usecols=["dataID", "game", "DateTime"])
    s["DateTime"] = pd.to_datetime(s["DateTime"])
    s = s.drop_duplicates(["dataID", "game"])
    d = t.merge(s, left_on=["subject", "game"], right_on=["dataID", "game"], how="inner")
    n_raw = len(d)
    d["rt"] = d["RT_choice"] / 1000.0
    d = d[(d.rt >= 0.2) & (d.rt <= 5.0)].copy()
    d["person"] = d["subject"]
    d["occasion"] = d["game"]
    first = d.groupby("person")["DateTime"].transform("min")
    d["time"] = (d["DateTime"] - first).dt.total_seconds() / 86400.0
    d["upper"] = d["response"].astype(int)
    d["stim"] = np.where(d["signal"] == 1, 1.0, -1.0)
    d["correct"] = d["accuracy"].astype(int)
    d["log_int_c"] = np.log(d["stimulus_intensity"]) - np.log(d["stimulus_intensity"]).mean()
    d["game_c"] = d["game"] - d["game"].mean()
    n_sess = d.groupby(["person", "occasion"]).size()
    keep = n_sess[n_sess >= 20].index
    d = d.set_index(["person", "occasion"]).loc[keep].reset_index()
    return d, n_raw


if __name__ == "__main__":
    trials_path, sess_path, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    d, n_raw = load(trials_path, sess_path)
    gaps = d.groupby(["person", "occasion"]).time.first().groupby(level=0).diff().dropna()
    desc = {"trials_raw": n_raw, "trials": len(d), "persons": d.person.nunique(),
            "sessions": d.groupby(["person", "occasion"]).ngroups,
            "sessions_per_person_median": float(d.groupby("person").occasion.nunique().median()),
            "trials_per_session_median": float(d.groupby(["person", "occasion"]).size().median()),
            "accuracy": d.correct.mean(), "median_rt": d.rt.median(),
            "gap_median_days": float(gaps.median()), "gap_p10_hours": float(24 * gaps.quantile(0.10)),
            "span_median_days": float(d.groupby("person").time.max().median())}
    print(desc, flush=True)
    pd.DataFrame([desc]).to_csv(outdir / "spaceobs_descriptives.csv", index=False)

    base = dict(drift=("log_int_c", "game_c"), nugget=True)
    specs = {}
    for prac, extra in (("common", {}), ("person", {"slope": "game_c"})):
        for kern in ("none", "ou", "rw", "se"):
            specs[f"{prac}|{kern}|nosv"] = dict(base, kernel=kern, **extra)
    rows, coefs, dobj, fits = [], [], {}, {}

    def run(name, spec):
        t0 = time.time()
        dd = DesignData.from_trials(d, "stim", **spec)
        f = fit_design(dd, se=True)
        dobj[name], fits[name] = dd, f
        prac, kern, sv = name.split("|")
        p = f.params
        rows.append({"model": name, "practice": prac, "kernel": kern, "sv": sv == "sv",
                     "loglik": f.loglik, "aic": f.aic, "n_params": f.n_params,
                     "halflife_days": f.halflife if kern == "ou" else np.nan,
                     "tau": p["tau"], "sd_stat": p["sd_stat"], "ell": p["ell"], "sd_rw": p["sd_rw"],
                     "omega": p["omega"], "slope_sd": p["slope_sd"], "rho": p["rho"], "sv_hat": p["sv"],
                     "converged": f.converged, "seconds": time.time() - t0})
        for k_, v_ in zip(_names(dd), f.x):
            coefs.append({"model": name, "param": k_, "x": v_, "se": f.se.get(k_, np.nan)})
        print({k_: rows[-1][k_] for k_ in ("model", "loglik", "aic", "halflife_days", "seconds")}, flush=True)

    for name, spec in specs.items():
        run(name, spec)
    res = pd.DataFrame(rows)
    best = res.sort_values("aic").iloc[0]
    prac, kern, _ = best.model.split("|")
    sv_spec = dict(base, kernel=kern, sv=True, **({"slope": "game_c"} if prac == "person" else {}))
    run(f"{prac}|{kern}|sv", sv_spec)
    res = pd.DataFrame(rows)
    res["daic"] = res.aic - res.aic.min()
    res.to_csv(outdir / "spaceobs_models.csv", index=False)
    pd.DataFrame(coefs).to_csv(outdir / "spaceobs_coefs.csv", index=False)
    print(res[["model", "loglik", "aic", "daic", "halflife_days", "sv_hat"]].round(3).to_string(index=False), flush=True)

    grid = np.log(np.geomspace(1e-4, 1e4, 17))
    ivs = []
    for name in ("common|ou|nosv", "person|ou|nosv"):
        dd, f = dobj[name], fits[name]
        prof = profile_theta(dd, np.sort(np.concatenate([grid, [np.log(f.params["theta"])]])), x_start=f.x)
        prof["model"] = name
        prof.to_csv(outdir / f"spaceobs_profile_{name.split('|')[0]}.csv", index=False)
        ref = max(f.loglik, prof.loglik.max())
        iv = profile_interval(prof, ref_loglik=ref, d=dd, fit=f)
        none_ll = res.loc[res.model == name.replace("|ou|", "|none|"), "loglik"].iloc[0]
        iv.update(model=name, lr_vs_none=2 * (ref - none_ll), regions=str(iv["regions"]))
        ivs.append(iv)
        print(name, {k_: iv[k_] for k_ in ("halflife_lo", "halflife_hi", "long_end_open", "short_end_open",
                                           "disconnected", "lr_vs_none")}, flush=True)
    pd.DataFrame(ivs).to_csv(outdir / "spaceobs_profile_intervals.csv", index=False)
