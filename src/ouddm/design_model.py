"""Exact likelihood for the OU-drift DDM with a full trial-level design.

This generalizes fit_exact() to the structure of real ambulatory tasks:

  * stimulus coding: the upper boundary is one response option (e.g.
    "different"), and the trial drift is s_r * eta_r, where s_r = +1 or -1
    is the stimulus type (+1 when the upper response is correct);
  * eta_r = u_i(t_ij) + x_r' beta: the OU state plus drift predictors;
  * boundary a_r = exp(alpha' z_r) depends on boundary predictors z_r;
  * start point w_r = logistic(gamma' q_r) depends on bias predictors q_r;
  * person-specific non-decision time ter_i = kappa * min RT_i, 0 < kappa < 1;
  * session-level OU state (optional), independent session deviations
    (optional, variance omega^2), and a person-specific random slope on a
    session-level variable such as practice (optional, correlated with the
    random intercept).

For any a, w, s the trial log density in eta is exactly
    log f(T | 0, a_r, w_r) + c_r eta - T eta^2 / 2,
    c_r = s_r * a_r (1 - w_r)   (upper response)
    c_r = -s_r * a_r w_r        (lower response),
so each session is still a Gaussian kernel in u with curvature S_j = sum T,
and Proposition 2 (all latent states integrate out) applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from .estimate import LOG2PI, _num_hessian
from .wiener import _log_g


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class DesignData:
    rt: np.ndarray            # RT (s) per trial
    upper: np.ndarray         # 1 = upper-boundary response
    stim: np.ndarray          # +1 / -1 stimulus sign
    sess: np.ndarray          # session index per trial (0..J-1)
    minrt_trial: np.ndarray   # min RT of the trial's person
    Z: np.ndarray             # boundary design (n, pa), first column = 1
    Q: np.ndarray             # bias design (n, pw), first column = 1
    X: np.ndarray             # drift predictors (n, k), no intercept
    time: np.ndarray          # session time
    person: np.ndarray        # person id per session
    person_slices: list
    J: int
    names: dict = field(default_factory=dict)
    zcache: dict = field(default_factory=dict)
    nugget: bool = False      # independent session-level drift deviations (variance omega^2)
    ou: bool = True           # OU state across sessions
    slope_z: np.ndarray | None = None   # session-level variable with a person-specific random slope

    @classmethod
    def from_trials(cls, trials: pd.DataFrame, stim_col: str, boundary=(), bias=(), drift=(),
                    nugget=False, ou=True, slope=None):
        need = {"person", "occasion", "time", "rt", "upper", stim_col, *boundary, *bias, *drift}
        if slope:
            need.add(slope)
        miss = need - set(trials.columns)
        if miss:
            raise ValueError(f"missing columns: {sorted(miss)}")
        tr = trials.dropna(subset=list(need)).sort_values(["person", "time", "occasion"]).reset_index(drop=True)
        agg = {"time": ("time", "first")}
        if slope:
            agg["slope_z"] = (slope, "first")
        s = (tr.groupby(["person", "occasion"], sort=False).agg(**agg).reset_index()
             .sort_values(["person", "time"]).reset_index(drop=True))
        key = {(p, o): i for i, (p, o) in enumerate(zip(s.person, s.occasion))}
        sess = np.array([key[(p, o)] for p, o in zip(tr.person, tr.occasion)])
        slices, start = [], 0
        for _, cnt in s.groupby("person", sort=False).size().items():
            slices.append(slice(start, start + cnt)); start += cnt
        minrt = tr.groupby("person")["rt"].transform("min").to_numpy(float)
        n = len(tr)
        Z = np.column_stack([np.ones(n)] + [tr[c].to_numpy(float) for c in boundary])
        Q = np.column_stack([np.ones(n)] + [tr[c].to_numpy(float) for c in bias])
        X = np.column_stack([tr[c].to_numpy(float) for c in drift]) if drift else np.zeros((n, 0))
        return cls(rt=tr.rt.to_numpy(float), upper=tr.upper.to_numpy(int), stim=np.sign(tr[stim_col].to_numpy(float)),
                   sess=sess, minrt_trial=minrt, Z=Z, Q=Q, X=X, time=s.time.to_numpy(float),
                   person=s.person.to_numpy(), person_slices=slices, J=len(s),
                   names={"boundary": ["intercept", *boundary], "bias": ["intercept", *bias], "drift": list(drift),
                          "slope": slope},
                   nugget=nugget, ou=ou, slope_z=(s.slope_z.to_numpy(float) if slope else None))

    @property
    def sizes(self):
        return self.Z.shape[1], self.Q.shape[1], self.X.shape[1]


def _names(d: DesignData):
    return ([f"log_a:{n}" for n in d.names["boundary"]] + [f"logit_w:{n}" for n in d.names["bias"]]
            + ["logit_kappa", "log_tau"]
            + (["log_theta", "log_sd_stat"] if d.ou else [])
            + (["log_omega"] if d.nugget else [])
            + (["log_slope_sd", "atanh_rho"] if d.slope_z is not None else [])
            + [f"beta:{n}" for n in d.names["drift"]])


def _unpack(x, d: DesignData):
    """Parameters by name. The OU is parameterized by its stationary SD,
    sd_stat = sigma / sqrt(2 theta), so both limits theta -> 0 (a person-level
    constant) and theta -> infinity (independent session deviations) are reachable."""
    names = _names(d)
    v = dict(zip(names, x))
    pa, pw, k = d.sizes
    out = {"alpha": np.asarray(x[:pa]), "gamma": np.asarray(x[pa:pa + pw]),
           "kappa": _logistic(v["logit_kappa"]), "tau": float(np.exp(v["log_tau"])),
           "theta": float(np.exp(v["log_theta"])) if d.ou else np.inf,
           "sd_stat": float(np.exp(v["log_sd_stat"])) if d.ou else 0.0,
           "omega": float(np.exp(v["log_omega"])) if d.nugget else 0.0,
           "slope_sd": float(np.exp(v["log_slope_sd"])) if d.slope_z is not None else 0.0,
           "rho": float(np.tanh(v["atanh_rho"])) if d.slope_z is not None else 0.0,
           "beta": np.asarray(x[len(names) - k:]) if k else np.zeros(0)}
    out["sigma"] = out["sd_stat"] * np.sqrt(2 * out["theta"]) if d.ou else 0.0
    return out


def _cov_profile(m, var, d: DesignData, p):
    """Gaussian log-likelihood of the session statistics with mu0 profiled out.
    Person covariance: [1 z] G [1 z]' + K(theta, sd_stat) + diag(var + omega^2)."""
    num = den = 0.0
    facs = []
    tau, sd, th, om = p["tau"], p["sd_stat"], p["theta"], p["omega"]
    s_sd, rho = p["slope_sd"], p["rho"]
    for sl in d.person_slices:
        t = d.time[sl]
        C = np.full((t.size, t.size), tau ** 2)
        if d.ou:
            C += sd ** 2 * np.exp(-th * np.abs(t[:, None] - t[None, :]))
        if d.slope_z is not None:
            z = d.slope_z[sl]
            C += rho * tau * s_sd * (z[:, None] + z[None, :]) + s_sd ** 2 * np.outer(z, z)
        C[np.diag_indices_from(C)] += var[sl] + om ** 2 + 1e-10
        try:
            c = cho_factor(C, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, np.nan
        one = np.ones(t.size)
        Si1 = cho_solve(c, one, check_finite=False)
        num += Si1 @ m[sl]; den += Si1.sum()
        facs.append(c)
    mu0 = num / den
    ll = 0.0
    for sl, c in zip(d.person_slices, facs):
        r = m[sl] - mu0
        ll -= 0.5 * (2 * np.sum(np.log(np.diag(c[0]))) + r @ cho_solve(c, r, check_finite=False) + r.size * LOG2PI)
    return ll, mu0


def design_loglik(x, d: DesignData, return_mu0=False):
    p = _unpack(x, d)
    alpha, gamma, kappa, beta = p["alpha"], p["gamma"], p["kappa"], p["beta"]
    a = np.exp(d.Z @ alpha)
    w = _logistic(d.Q @ gamma)
    T = d.rt - kappa * d.minrt_trial
    key = (tuple(alpha), tuple(gamma), float(kappa))
    if d.zcache.get("key") == key:
        zero = d.zcache["val"]
    else:
        wb = np.where(d.upper == 1, 1.0 - w, w)     # zero-drift density: lower uses w, upper 1 - w
        zero = float(np.sum(-2.0 * np.log(a) + _log_g(T / a ** 2, wb)))
        d.zcache.update(key=key, val=zero)
    c = d.stim * np.where(d.upper == 1, a * (1.0 - w), -a * w)
    S = np.bincount(d.sess, weights=T, minlength=d.J)
    b = np.bincount(d.sess, weights=c, minlength=d.J)
    extra = 0.0
    if beta.size:
        xb = d.X @ beta
        b = b - np.bincount(d.sess, weights=T * xb, minlength=d.J)
        extra = float(np.sum(c * xb) - 0.5 * np.sum(T * xb ** 2))
    ll = zero + extra + np.sum(b ** 2 / (2 * S) + 0.5 * (LOG2PI - np.log(S)))
    cov, mu0 = _cov_profile(b / S, 1.0 / S, d, p)
    return (ll + cov, mu0) if return_mu0 else ll + cov


DESIGN_LOG_THETA = (-6.0, 9.0)   # fitting range; profiles may go beyond


@dataclass
class DesignFit:
    params: dict
    se: dict
    x: np.ndarray
    halflife: float
    halflife_ci: tuple
    loglik: float
    n_params: int
    converged: bool
    theta_at_bound: str

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.n_params


def _bounds(d: DesignData, fix_theta=None):
    b = []
    for n in _names(d):
        if n.startswith("log_a:"):
            b.append((-3, 3))
        elif n.startswith("logit_w:"):
            b.append((-4, 4))
        elif n == "logit_kappa":
            b.append((-8, 8))
        elif n == "log_theta":
            b.append(DESIGN_LOG_THETA if fix_theta is None else (fix_theta, fix_theta))
        elif n == "atanh_rho":
            b.append((-3, 3))
        elif n.startswith("beta:"):
            b.append((-10, 10))
        else:
            b.append((-8, 2))
    return b


def _starts(d: DesignData, thetas=(0.3, 2.0, 50.0, 2000.0), taus=(0.3, 0.03)):
    names = _names(d)
    out = []
    for th0 in (thetas if d.ou else (None,)):
        for tau0 in taus:
            z = np.zeros(len(names))
            z[0] = np.log(1.5)
            z[names.index("logit_kappa")] = 2.0
            z[names.index("log_tau")] = np.log(tau0)
            if d.ou:
                z[names.index("log_theta")] = np.log(th0)
                z[names.index("log_sd_stat")] = np.log(0.3)
            if d.nugget:
                z[names.index("log_omega")] = np.log(0.2)
            if d.slope_z is not None:
                z[names.index("log_slope_sd")] = np.log(0.05)
            out.append(z)
    return out


def fit_design(d: DesignData, se: bool = True, x0: np.ndarray | None = None) -> DesignFit:
    """Joint ML for the design model. The likelihood in theta can have separate
    optima at slow and at very fast reversion, so starts span theta from 0.3
    to 2000 per day and two values of tau."""
    names = _names(d)
    f = lambda x: -design_loglik(x, d)
    bounds = _bounds(d)
    starts = _starts(d)
    if x0 is not None:
        starts.insert(0, np.asarray(x0, float))
    best = None
    for z in starts:
        r = minimize(f, z, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    ll, mu0 = design_loglik(x, d, return_mu0=True)
    p = _unpack(x, d)
    sev = np.full(len(names), np.nan)
    if se:
        try:
            cov = np.linalg.inv(_num_hessian(f, x))
            sev = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            pass
    params = dict(zip(names, x)); params["mu0"] = mu0
    params.update({k: p[k] for k in ("kappa", "tau", "theta", "sigma", "omega", "sd_stat", "slope_sd", "rho")})
    sed = dict(zip(names, sev))
    hl, ci, atb = np.nan, (np.nan, np.nan), ""
    if d.ou:
        ilt = names.index("log_theta")
        hl = np.log(2) / p["theta"]
        ci = (float(np.exp(np.log(hl) - 1.96 * sev[ilt])), float(np.exp(np.log(hl) + 1.96 * sev[ilt])))
        lt = x[ilt]
        atb = ("upper" if lt >= DESIGN_LOG_THETA[1] - 1e-6 else
               ("lower" if lt <= DESIGN_LOG_THETA[0] + 1e-6 else ""))
    return DesignFit(params=params, se=sed, x=x, halflife=float(hl), halflife_ci=ci,
                     loglik=float(ll), n_params=len(names) + 1, converged=bool(best.success), theta_at_bound=atb)


def profile_point(d: DesignData, log_theta: float, warm=()) -> tuple[float, np.ndarray]:
    """Maximized log-likelihood with log theta fixed; starts from the warm
    vectors given and from the default starts at this theta."""
    names = _names(d)
    ilt = names.index("log_theta")
    f = lambda x: -design_loglik(x, d)
    bounds = _bounds(d, fix_theta=log_theta)
    cands = [np.asarray(w, float).copy() for w in warm] + _starts(d, thetas=(np.exp(log_theta),))
    best = None
    for z in cands:
        z = z.copy(); z[ilt] = log_theta
        r = minimize(f, z, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return -float(best.fun), best.x


def profile_theta(d: DesignData, log_thetas, x_start=None) -> pd.DataFrame:
    """Profile log-likelihood over a grid of log theta (warm-started along the grid)."""
    out, prev = [], None
    for lt in log_thetas:
        warm = [w for w in (prev, x_start) if w is not None]
        ll, x = profile_point(d, lt, warm)
        prev = x
        p = _unpack(x, d)
        out.append({"log_theta": lt, "theta": np.exp(lt), "halflife_days": np.log(2) / np.exp(lt), "loglik": ll,
                    "tau": p["tau"], "sd_stat": p["sd_stat"], "omega": p["omega"], "slope_sd": p["slope_sd"]})
    return pd.DataFrame(out)


def supported_regions(dev: np.ndarray, cut: float = 3.841) -> list:
    """Index ranges (start, end), inclusive, of the connected runs of grid
    points whose deviance is at or below the cut-off."""
    inside = np.asarray(dev) <= cut
    runs, start = [], None
    for k, v in enumerate(inside):
        if v and start is None:
            start = k
        if not v and start is not None:
            runs.append((start, k - 1)); start = None
    if start is not None:
        runs.append((start, len(inside) - 1))
    return runs


def profile_interval(prof: pd.DataFrame, ref_loglik: float | None = None, d: DesignData | None = None,
                     fit: DesignFit | None = None, cut: float = 3.841) -> dict:
    """Approximate profile-likelihood interval for the half-life.

    The deviance is measured from ref_loglik (the full maximum; defaults to the
    grid maximum). If d and fit are given, each crossing bracketed by the grid
    is located by root-finding on the profile; otherwise by linear
    interpolation.

    The supported set {deviance <= cut} need not be connected. Each connected
    run of grid points gives one range in ``regions`` (half-lives, shortest
    range first), and ``disconnected`` is True when there is more than one.
    ``halflife_lo`` and ``halflife_hi`` are the outer ends; when the set is
    disconnected they span points outside it, so use ``regions`` instead.
    The open-end flags refer to the outer ends.

    If no grid point lies within the cut-off, the supported set is reported
    as empty (``regions == []``, ``empty`` and ``needs_refinement`` True, the
    ends NaN). This happens when ref_loglik is above every grid point by more
    than cut/2; add the fitted maximum to the profile or refine the grid near
    ``best_grid_halflife``.
    """
    p = prof.sort_values("log_theta").reset_index(drop=True)
    ref = p.loglik.max() if ref_loglik is None else max(ref_loglik, p.loglik.max())
    dev = 2 * (ref - p.loglik.to_numpy())
    lt = p.log_theta.to_numpy()
    n = len(lt)
    runs = supported_regions(dev, cut)
    if not runs:
        # No grid point lies within the cut-off of the reference: the grid
        # misses the supported set (e.g. the fitted maximum was not added and
        # the grid is too coarse). Report an empty set rather than guess.
        imin = int(np.argmin(dev))
        nan = float("nan")
        return {"theta_lo": nan, "theta_hi": nan, "halflife_lo": nan, "halflife_hi": nan,
                "long_end_open": False, "short_end_open": False,
                "disconnected": False, "regions": [], "empty": True,
                "needs_refinement": True, "best_grid_halflife": float(np.log(2) / np.exp(lt[imin])),
                "ref_loglik": float(ref), "max_dev_on_grid": float(dev.max())}

    def cross(a, b, warm):
        if d is None or fit is None:
            return float(np.interp(cut, sorted([dev[a], dev[b]]), [lt[a], lt[b]] if dev[a] < dev[b] else [lt[b], lt[a]]))
        g = lambda x: 2 * (ref - profile_point(d, x, warm)[0]) - cut
        return float(brentq(g, lt[a], lt[b], xtol=0.01))

    warm = [fit.x] if fit is not None else []
    bounds = []  # (lo log-theta, hi log-theta) per run
    for a, b in runs:
        lo_lt = lt[a] if a == 0 else cross(a - 1, a, warm)
        hi_lt = lt[b] if b == n - 1 else cross(b, b + 1, warm)
        bounds.append((lo_lt, hi_lt))
    lo_lt, hi_lt = bounds[0][0], bounds[-1][1]
    regions = [(float(np.log(2) / np.exp(h)), float(np.log(2) / np.exp(l))) for l, h in reversed(bounds)]
    return {"theta_lo": float(np.exp(lo_lt)), "theta_hi": float(np.exp(hi_lt)),
            "halflife_lo": float(np.log(2) / np.exp(hi_lt)), "halflife_hi": float(np.log(2) / np.exp(lo_lt)),
            "long_end_open": bool(runs[0][0] == 0), "short_end_open": bool(runs[-1][1] == n - 1),
            "disconnected": len(runs) > 1, "regions": regions, "empty": False, "needs_refinement": False,
            "ref_loglik": float(ref), "max_dev_on_grid": float(dev.max())}
