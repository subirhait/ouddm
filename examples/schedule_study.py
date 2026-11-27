"""Simulation study 4: session schedules for a fixed trial budget, calibrated
to the Space Observer fit (person-specific practice slopes, session
deviations, across-trial drift variability, OU dynamics).

Every simulated person gets 420 trials over 14 days under four schedules:
  alternate  7 sessions of 60 trials, every other day (like Space Observer)
  daily     14 sessions of 30 trials, once a day
  twice     28 sessions of 15 trials, at about 9:00 and 19:00
  paired    14 sessions of 30 trials, every other day, two sessions 4-6 h apart
Truths: no OU (all session variance independent) and OU half-lives of 0.25,
0.65 and 2 days. The four schedules of one replication share the persons, the
OU path (simulated on the union of their session times) and the seed, so
schedule differences are paired (common random numbers).

Fitting, per data set:
  1. full ML of the model without dynamics (trial-level and session-level parameters);
  2. the OU model with the trial-level parameters (boundary, start point,
     non-decision time, drift variability, drift predictors) held at their
     values from step 1; only the session-level parameters are refit;
  3. under OU truths, a profile likelihood of the half-life on a grid, with the
     same held values.
Decision times are simulated with Euler steps of 2 ms and a Brownian-bridge
crossing correction.
The test statistic is 2 x (loglik step 2 - loglik step 1). Its null
distribution is non-standard (the half-life is not identified when the OU
variance is zero), so the critical value for each schedule is the 95th
percentile of the statistic in the no-OU data sets for that schedule.

Usage:
  python -I schedule_study.py calibrate <space-observer-dta.csv> <task_data.csv> <outdir>
  python -I schedule_study.py run <calibration.csv> <outdir> <truth> <rep_from> <rep_to> [n_persons]
  python -I schedule_study.py check <calibration.csv> <outdir>     # held vs full fit, one data set
  python -I schedule_study.py summary <outdir> [<extra no-OU dir> ...]   # critical values from all no-OU sets
  python -I schedule_study.py nullcheck <outdir> <checkdir>   # split-half check of the critical values
  truth: none, 0.25, 0.65 or 2
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ouddm.design_model import DesignData, _bounds, _cov_profile, _names, _session_kernels, _unpack, design_loglik
from ouddm.design_model import _logistic
from ouddm.simulate import simulate_wiener

SCHEDULES = ("alternate", "daily", "twice", "paired")
SPEC = dict(drift=("log_int", "game_c"), nugget=True, slope="game_c", sv=True)
SESSION_PARAMS = ("log_tau", "log_theta", "log_sd_stat", "log_omega", "log_slope_sd", "atanh_rho")
HL_GRID = np.geomspace(0.02, 50.0, 13)
CUT = 3.841


def schedule_times(name, rng):
    j = lambda n: rng.normal(0.0, 1.0 / 24.0, n)   # SD 1 h around the planned start
    if name == "alternate":
        return np.arange(0, 14, 2) + 10 / 24 + j(7)
    if name == "daily":
        return np.arange(14) + 10 / 24 + j(14)
    if name == "twice":
        t = np.sort(np.r_[np.arange(14) + 9 / 24, np.arange(14) + 19 / 24])
        return t + j(28)
    first = np.arange(0, 14, 2) + 10 / 24 + j(7)
    return np.sort(np.r_[first, first + rng.uniform(4, 6, 7) / 24])


N_TRIALS = {"alternate": 60, "daily": 30, "twice": 15, "paired": 30}


def simulate(cal, hl, n_persons, rng):
    """Returns a dict schedule -> trial DataFrame (same persons and OU paths)."""
    c = cal
    ou = hl is not None
    sd_ou = c["sd_stat"] if ou else 0.0
    omega = c["omega"] if ou else np.sqrt(c["omega"] ** 2 + c["sd_stat"] ** 2)
    theta = np.log(2) / hl if ou else 1.0
    G = np.array([[c["tau"] ** 2, c["rho"] * c["tau"] * c["slope_sd"]],
                  [c["rho"] * c["tau"] * c["slope_sd"], c["slope_sd"] ** 2]])
    out = {s: [] for s in SCHEDULES}
    for i in range(n_persons):
        u0, u1 = rng.multivariate_normal([0.0, 0.0], G)
        times = {s: schedule_times(s, rng) for s in SCHEDULES}
        allt = np.concatenate([times[s] for s in SCHEDULES])
        order = np.argsort(allt)
        path = np.empty(allt.size)
        x, tprev = rng.normal(0, sd_ou), None
        for k, idx in enumerate(order):
            if tprev is not None:
                r = np.exp(-theta * (allt[idx] - tprev))
                x = r * x + sd_ou * np.sqrt(1 - r * r) * rng.standard_normal()
            path[idx] = x; tprev = allt[idx]
        pos = 0
        for s in SCHEDULES:
            t = times[s]; K = t.size; n = N_TRIALS[s]
            state = path[pos:pos + K]; pos += K
            game_c = np.arange(1, K + 1) - (K + 1) / 2
            nug = rng.normal(0, omega, K)
            sess_drift = c["mu0"] + u0 + state + nug + (c["b_game"] + u1) * game_c
            li = rng.normal(0, c["log_int_sd"], K * n)
            stim = rng.choice([-1.0, 1.0], K * n)
            sess = np.repeat(np.arange(K), n)
            v = stim * (sess_drift[sess] + c["b_int"] * li + c["sv"] * rng.standard_normal(K * n))
            dt, up = simulate_wiener(v, c["a"], 1.0, rng, dt=2e-3)
            rt = c["ter"] + dt
            keep = (up >= 0) & (rt >= 0.2) & (rt <= 5.0)
            out[s].append(pd.DataFrame({"person": i, "occasion": sess[keep], "time": t[sess[keep]],
                                        "game_c": game_c[sess[keep]], "rt": rt[keep],
                                        "upper": up[keep].astype(int), "stim": stim[keep], "log_int": li[keep]}))
    return {s: pd.concat(v, ignore_index=True) for s, v in out.items()}


def truth_x(d, cal):
    names = _names(d)
    v = {"log_a:intercept": np.log(cal["a"]), "logit_w:intercept": 0.0, "logit_kappa": 2.0,
         "log_tau": np.log(cal["tau"]), "log_theta": np.log(np.log(2) / 0.65),
         "log_sd_stat": np.log(cal["sd_stat"]), "log_omega": np.log(np.hypot(cal["omega"], cal["sd_stat"])),
         "log_sv": np.log(cal["sv"]), "log_slope_sd": np.log(max(cal["slope_sd"], 0.005)),
         "atanh_rho": 0.5, "beta:log_int": cal["b_int"], "beta:game_c": cal["b_game"]}
    return np.array([v[n] for n in names])


def fit_null(df, cal):
    d = DesignData.from_trials(df, "stim", **SPEC, kernel="none")
    f = lambda x: -design_loglik(x, d)
    r = minimize(f, truth_x(d, cal), method="L-BFGS-B", bounds=_bounds(d))
    return d, r.x, -r.fun


class SessionLik:
    """Session-level log-likelihood of the OU model with trial-level parameters held."""

    def __init__(self, df, x_null, d_null):
        self.d = DesignData.from_trials(df, "stim", **SPEC, kernel="ou")
        self.names = _names(self.d)
        held = dict(zip(_names(d_null), x_null))
        self.x = np.array([held.get(n, 0.0) for n in self.names])
        self.free = [self.names.index(n) for n in SESSION_PARAMS]
        p = _unpack(self.x, self.d)
        a = np.exp(self.d.Z @ p["alpha"]); w = _logistic(self.d.Q @ p["gamma"])
        T = self.d.rt - p["kappa"] * self.d.minrt_trial
        b, S, _ = _session_kernels(p, self.d, a, w, T)
        self.m, self.var = b / S, 1.0 / S
        self.const = design_loglik(self.x, self.d) - _cov_profile(self.m, self.var, self.d, p)[0]
        bnd = _bounds(self.d)
        self.bounds = [bnd[i] for i in self.free]

    def ll(self, z, log_theta=None):
        x = self.x.copy(); x[self.free] = z
        if log_theta is not None:
            x[self.names.index("log_theta")] = log_theta
        return self.const + _cov_profile(self.m, self.var, self.d, _unpack(x, self.d))[0]

    def fit(self, x_null, d_null):
        held = dict(zip(_names(d_null), x_null))
        best = None
        for th, sd in ((np.exp(8.0), 1e-3), (np.log(2) / 2.0, 0.2), (np.log(2) / 0.5, 0.2), (np.log(2) / 0.1, 0.2)):
            z0 = np.array([held["log_tau"], np.log(th), np.log(sd), held["log_omega"],
                           held["log_slope_sd"], held["atanh_rho"]])
            r = minimize(lambda z: -self.ll(z), z0, method="L-BFGS-B", bounds=self.bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best.x, -best.fun

    def profile(self, z_hat):
        i = SESSION_PARAMS.index("log_theta")
        keep = [k for k in range(len(SESSION_PARAMS)) if k != i]
        bnd = [self.bounds[k] for k in keep]
        out = []
        for hl in HL_GRID:
            lt = np.log(np.log(2) / hl)
            best = None
            for start in (z_hat, np.r_[z_hat[:2], np.log(0.2), z_hat[3:]]):
                zz = np.delete(start, i)
                fun = lambda q: -self.ll(np.insert(q, i, lt), log_theta=lt)
                r = minimize(fun, zz, method="L-BFGS-B", bounds=bnd)
                if best is None or r.fun < best.fun:
                    best = r
            out.append(-best.fun)
        return np.array(out)


def interval(hl_grid, prof, ll_max, hl_hat):
    """Approximate 95% profile interval on the grid (log-linear interpolation)."""
    dev = 2 * (ll_max - prof)
    lh = np.log(hl_grid)
    inside = dev <= CUT
    lo_open, hi_open = bool(inside[0]), bool(inside[-1])
    lo = hi = np.nan
    below = np.flatnonzero(lh < np.log(hl_hat))
    above = np.flatnonzero(lh > np.log(hl_hat))
    if not lo_open:
        k = [j for j in below if dev[j] > CUT]
        if k:
            j = max(k); nxt = j + 1
            d1 = dev[nxt] if nxt < len(dev) and lh[nxt] < np.log(hl_hat) else 0.0
            l1 = lh[nxt] if nxt < len(dev) and lh[nxt] < np.log(hl_hat) else np.log(hl_hat)
            lo = np.exp(lh[j] + (dev[j] - CUT) / (dev[j] - d1) * (l1 - lh[j]))
    if not hi_open:
        k = [j for j in above if dev[j] > CUT]
        if k:
            j = min(k); prv = j - 1
            d0 = dev[prv] if prv >= 0 and lh[prv] > np.log(hl_hat) else 0.0
            l0 = lh[prv] if prv >= 0 and lh[prv] > np.log(hl_hat) else np.log(hl_hat)
            hi = np.exp(l0 + (CUT - d0) / (dev[j] - d0) * (lh[j] - l0))
    return lo, hi, lo_open, hi_open


def one_rep(cal, truth, rep, n_persons):
    hl = None if truth == "none" else float(truth)
    seed = [2027, rep, {"none": 0, "0.25": 1, "0.65": 2, "2": 3}[truth]]
    rng = np.random.default_rng(seed)
    data = simulate(cal, hl, n_persons, rng)
    rows = []
    for s in SCHEDULES:
        t0 = time.time()
        d0, x0, ll0 = fit_null(data[s], cal)
        sl = SessionLik(data[s], x0, d0)
        z, ll1 = sl.fit(x0, d0)
        ll1 = max(ll1, ll0)
        hl_hat = np.log(2) / np.exp(z[1])
        if hl is None:   # the profile is needed only for estimation under OU truths
            lo = hi = np.nan; lo_open = hi_open = False
        else:
            prof = sl.profile(z)
            ll_max = max(ll1, prof.max())
            lo, hi, lo_open, hi_open = interval(HL_GRID, prof, ll_max, hl_hat)
        rows.append({"truth": truth, "rep": rep, "schedule": s, "n_persons": n_persons,
                     "trials": len(data[s]), "lr": 2 * (ll1 - ll0), "halflife_hat": hl_hat,
                     "sd_stat_hat": np.exp(z[2]), "hl_lo": lo, "hl_hi": hi, "lo_open": lo_open, "hi_open": hi_open,
                     "sv_hat": np.exp(dict(zip(_names(d0), x0))["log_sv"]), "seconds": time.time() - t0})
    return rows


def summary(outdir, extra=(), wide=10.0):
    r = pd.concat([pd.read_csv(p) for p in sorted(outdir.glob("schedule_reps_*.csv"))]
                  + [pd.read_csv(p) for e in extra for p in sorted(Path(e).glob("schedule_reps_none_*.csv"))],
                  ignore_index=True)
    crit = r[r.truth == "none"].groupby("schedule").lr.quantile(0.95).rename("crit")
    r = r.join(crit, on="schedule")
    r["detect"] = r.lr > r.crit
    r["detect_nominal"] = r.lr > CUT
    r["open"] = r.lo_open | r.hi_open
    r["too_wide"] = r.open | (r.hl_hi / r.hl_lo > wide)
    t = pd.to_numeric(r.truth, errors="coerce")
    r["log_err"] = np.log(r.halflife_hat) - np.log(t)
    r["covered"] = (~r.lo_open & ~r.hi_open & (r.hl_lo <= t) & (t <= r.hl_hi)) | \
                   (r.lo_open & ~r.hi_open & (t <= r.hl_hi)) | (r.hi_open & ~r.lo_open & (r.hl_lo <= t)) | \
                   (r.lo_open & r.hi_open)
    g = r.groupby(["truth", "schedule"])
    s = g.agg(reps=("rep", "size"), crit=("crit", "first"), detect=("detect", "mean"),
              detect_nominal=("detect_nominal", "mean"), median_lr=("lr", "median"),
              median_hl=("halflife_hat", "median"), bias_log=("log_err", "mean"),
              rmse_log=("log_err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
              mad_log=("log_err", lambda e: float(np.median(np.abs(e)))),
              open=("open", "mean"), too_wide=("too_wide", "mean"), coverage=("covered", "mean"),
              seconds=("seconds", "mean")).reset_index()
    s.to_csv(outdir / "schedule_summary.csv", index=False)
    r.to_csv(outdir / "schedule_all_reps.csv", index=False)
    with pd.option_context("display.width", 200):
        print(s.round(3).to_string(index=False))


def calibrate(tp, sp, outdir):
    """Generating values from the Space Observer OU fit with drift variability (Table 8)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from space_observer_application import load
    d, _ = load(tp, sp)
    x = np.array([float(v) for v in pd.read_csv(outdir / "dynamics_sv_so.csv").set_index("kernel").loc["ou", "x"].split(";")])
    dd = DesignData.from_trials(d, "stim", drift=("log_int_c", "game_c"), nugget=True, slope="game_c", kernel="ou", sv=True)
    _, mu0 = design_loglik(x, dd, return_mu0=True)
    p = _unpack(x, dd)
    cal = {"a": float(np.exp(p["alpha"][0])), "mu0": mu0, "tau": p["tau"], "sd_stat": p["sd_stat"],
           "omega": p["omega"], "sv": p["sv"], "slope_sd": p["slope_sd"], "rho": p["rho"],
           "b_int": float(p["beta"][0]), "b_game": float(p["beta"][1]), "theta": p["theta"],
           "ter": float(p["kappa"] * d.groupby("person").rt.min().median()), "log_int_sd": float(d.log_int_c.std())}
    pd.DataFrame([cal]).to_csv(outdir / "schedule_calibration.csv", index=False)
    print(cal)


def check(cal, outdir):
    """Held trial-level parameters versus full joint ML for one data set (twice daily, half-life 0.65)."""
    from ouddm.design_model import fit_design
    data = simulate(cal, 0.65, 160, np.random.default_rng([2027, 0, 2]))
    df = data["twice"]
    d0, x0, ll0 = fit_null(df, cal)
    sl = SessionLik(df, x0, d0)
    z, ll1 = sl.fit(x0, d0)
    f = fit_design(DesignData.from_trials(df, "stim", **SPEC, kernel="ou"), se=False)
    r = pd.DataFrame([{"lr_held": 2 * (ll1 - ll0), "hl_held": np.log(2) / np.exp(z[1]),
                       "lr_full": 2 * (f.loglik - ll0), "hl_full": f.halflife}])
    r.to_csv(outdir / "schedule_check.csv", index=False)
    print(r)


def nullcheck(outdir, checkdir):
    """Split-half check: critical values estimated from one set of no-OU data sets,
    applied to the other set (each way)."""
    first = pd.concat([pd.read_csv(p) for p in sorted(outdir.glob("schedule_reps_none_*.csv"))], ignore_index=True)
    fresh = pd.concat([pd.read_csv(p) for p in sorted(checkdir.glob("schedule_reps_none_*.csv"))], ignore_index=True)
    rows = []
    for sch in SCHEDULES:
        f1, f2_ = first[first.schedule == sch].lr, fresh[fresh.schedule == sch].lr
        c1, c2 = f1.quantile(0.95), f2_.quantile(0.95)
        rows.append({"schedule": sch, "n_first": len(f1), "n_fresh": len(f2_), "crit_first": c1, "crit_fresh": c2,
                     "k_fresh_over_crit_first": int((f2_ > c1).sum()), "k_first_over_crit_fresh": int((f1 > c2).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(checkdir / "schedule_nullcheck_summary.csv", index=False)
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "nullcheck":
        nullcheck(Path(sys.argv[2]), Path(sys.argv[3]))
        sys.exit()
    if sys.argv[1] == "check":
        check(pd.read_csv(sys.argv[2]).iloc[0].to_dict(), Path(sys.argv[3]))
        sys.exit()
    if sys.argv[1] == "calibrate":
        calibrate(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        sys.exit()
    if sys.argv[1] == "summary":
        summary(Path(sys.argv[2]), sys.argv[3:])
        sys.exit()
    cal = pd.read_csv(sys.argv[2]).iloc[0].to_dict()
    outdir = Path(sys.argv[3]); truth = sys.argv[4]
    r0, r1 = int(sys.argv[5]), int(sys.argv[6])
    n_persons = int(sys.argv[7]) if len(sys.argv) > 7 else 160
    rows = []
    for rep in range(r0, r1):
        rows += one_rep(cal, truth, rep, n_persons)
        print(rows[-4:], flush=True)
        pd.DataFrame(rows).to_csv(outdir / f"schedule_reps_{truth}_{r0}_{r1}.csv", index=False)
