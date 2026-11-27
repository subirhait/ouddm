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

Across-trial drift variability (sv): if eta_r = m_r + e_r with e_r ~ N(0, sv^2),
integrating e_r gives exactly
    log f(T | 0) + (c m - T m^2 / 2) / k + c^2 sv^2 / (2k) - log(k) / 2,  k = 1 + T sv^2,
which is again quadratic in the session state, with weights 1 / k.

Session dynamics (kernel):
    "ou"   Ornstein-Uhlenbeck, K = sd^2 exp(-theta |dt|)        (half-life ln2 / theta)
    "rw"   random walk from the first session, K = s_rw^2 min(t - t0, t' - t0)
    "se"   squared-exponential Gaussian process, K = sd^2 exp(-dt^2 / (2 ell^2))
    "none" no dynamic component
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
    ou: bool = True           # OU state across sessions (kept for compatibility; see kernel)
    slope_z: np.ndarray | None = None   # session-level variable with a person-specific random slope
    kernel: str = "ou"        # session dynamics: "ou", "rw", "se", or "none"
    sv: bool = False          # estimate across-trial drift variability
    trial_index: np.ndarray | None = None   # index of each trial in the input DataFrame
    has_outcomes: bool = True               # False: built for prediction only (no rt, no response)

    @classmethod
    def from_trials(cls, trials: pd.DataFrame, stim_col: str, boundary=(), bias=(), drift=(),
                    nugget=False, ou=True, slope=None, kernel=None, sv=False, outcomes=True):
        """kernel: "ou" (default), "rw", "se" or "none"; ou=False is the same as kernel="none".
        sv: also estimate across-trial drift variability.
        outcomes=False builds data for prediction only (future sessions): the
        columns rt and upper are not needed, and the object can be passed to
        predict_choices as d_test but not fitted. Trials keep their DataFrame
        index in trial_index, so predictions can be matched to input rows."""
        kernel = ("ou" if ou else "none") if kernel is None else kernel
        if kernel not in ("ou", "rw", "se", "none"):
            raise ValueError("kernel must be 'ou', 'rw', 'se' or 'none'")
        need = {"person", "occasion", "time", stim_col, *boundary, *bias, *drift}
        if outcomes:
            need |= {"rt", "upper"}
        if slope:
            need.add(slope)
        miss = need - set(trials.columns)
        if miss:
            raise ValueError(f"missing columns: {sorted(miss)}")
        tr = trials.dropna(subset=list(need)).sort_values(["person", "time", "occasion"], kind="mergesort")
        idx = tr.index.to_numpy()
        tr = tr.reset_index(drop=True)
        if not outcomes:
            tr = tr.assign(rt=np.nan, upper=-1)
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
        minrt = (tr.groupby("person")["rt"].transform("min").to_numpy(float) if outcomes
                 else np.full(len(tr), np.nan))
        n = len(tr)
        Z = np.column_stack([np.ones(n)] + [tr[c].to_numpy(float) for c in boundary])
        Q = np.column_stack([np.ones(n)] + [tr[c].to_numpy(float) for c in bias])
        X = np.column_stack([tr[c].to_numpy(float) for c in drift]) if drift else np.zeros((n, 0))
        return cls(rt=tr.rt.to_numpy(float), upper=tr.upper.to_numpy(int), stim=np.sign(tr[stim_col].to_numpy(float)),
                   sess=sess, minrt_trial=minrt, Z=Z, Q=Q, X=X, time=s.time.to_numpy(float),
                   person=s.person.to_numpy(), person_slices=slices, J=len(s),
                   names={"boundary": ["intercept", *boundary], "bias": ["intercept", *bias], "drift": list(drift),
                          "slope": slope},
                   nugget=nugget, ou=(kernel == "ou"), slope_z=(s.slope_z.to_numpy(float) if slope else None),
                   kernel=kernel, sv=sv, trial_index=idx, has_outcomes=outcomes)

    @property
    def sizes(self):
        return self.Z.shape[1], self.Q.shape[1], self.X.shape[1]


def _names(d: DesignData):
    return ([f"log_a:{n}" for n in d.names["boundary"]] + [f"logit_w:{n}" for n in d.names["bias"]]
            + ["logit_kappa", "log_tau"]
            + {"ou": ["log_theta", "log_sd_stat"], "se": ["log_ell", "log_sd_stat"],
               "rw": ["log_sd_rw"], "none": []}[d.kernel]
            + (["log_omega"] if d.nugget else [])
            + (["log_sv"] if d.sv else [])
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
           "theta": float(np.exp(v["log_theta"])) if d.kernel == "ou" else np.inf,
           "sd_stat": float(np.exp(v["log_sd_stat"])) if d.kernel in ("ou", "se") else 0.0,
           "ell": float(np.exp(v["log_ell"])) if d.kernel == "se" else np.nan,
           "sd_rw": float(np.exp(v["log_sd_rw"])) if d.kernel == "rw" else 0.0,
           "sv": float(np.exp(v["log_sv"])) if d.sv else 0.0,
           "omega": float(np.exp(v["log_omega"])) if d.nugget else 0.0,
           "slope_sd": float(np.exp(v["log_slope_sd"])) if d.slope_z is not None else 0.0,
           "rho": float(np.tanh(v["atanh_rho"])) if d.slope_z is not None else 0.0,
           "beta": np.asarray(x[len(names) - k:]) if k else np.zeros(0)}
    out["sigma"] = out["sd_stat"] * np.sqrt(2 * out["theta"]) if d.kernel == "ou" else 0.0
    return out


def _kcross(t1, t2, z1, z2, t_first, d: DesignData, p):
    """Covariance of latent session drifts between two sets of sessions of one
    person (no independent deviations): tau^2 + dynamics kernel + slope terms."""
    tau, sd = p["tau"], p["sd_stat"]
    C = np.full((t1.size, t2.size), tau ** 2)
    dt = t1[:, None] - t2[None, :]
    if d.kernel == "ou":
        C += sd ** 2 * np.exp(-p["theta"] * np.abs(dt))
    elif d.kernel == "se":
        C += sd ** 2 * np.exp(-0.5 * (dt / p["ell"]) ** 2)
    elif d.kernel == "rw":
        C += p["sd_rw"] ** 2 * np.minimum((t1 - t_first)[:, None], (t2 - t_first)[None, :])
    if d.slope_z is not None:
        s_sd, rho = p["slope_sd"], p["rho"]
        C += rho * tau * s_sd * (z1[:, None] + z2[None, :]) + s_sd ** 2 * np.outer(z1, z2)
    return C


def _kbatch(t, z, d: DesignData, p):
    """Batched _kcross for G persons with the same number of sessions: t, z are (G, K)."""
    tau, sd = p["tau"], p["sd_stat"]
    G, K = t.shape
    C = np.full((G, K, K), tau ** 2)
    dt = t[:, :, None] - t[:, None, :]
    if d.kernel == "ou":
        C += sd ** 2 * np.exp(-p["theta"] * np.abs(dt))
    elif d.kernel == "se":
        C += sd ** 2 * np.exp(-0.5 * (dt / p["ell"]) ** 2)
    elif d.kernel == "rw":
        t0 = t - t.min(axis=1, keepdims=True)
        C += p["sd_rw"] ** 2 * np.minimum(t0[:, :, None], t0[:, None, :])
    if d.slope_z is not None:
        s_sd, rho = p["slope_sd"], p["rho"]
        C += rho * tau * s_sd * (z[:, :, None] + z[:, None, :]) + s_sd ** 2 * z[:, :, None] * z[:, None, :]
    return C


def _person_groups(d: DesignData):
    """Session indices of persons grouped by their number of sessions (cached)."""
    g = d.zcache.get("groups")
    if g is None:
        by = {}
        for sl in d.person_slices:
            by.setdefault(sl.stop - sl.start, []).append(np.arange(sl.start, sl.stop))
        g = [np.vstack(v) for v in by.values()]
        d.zcache["groups"] = g
    return g


def _cov_profile(m, var, d: DesignData, p):
    """Gaussian log-likelihood of the session statistics with mu0 profiled out.
    Person covariance: [1 z] G [1 z]' + K(theta, sd_stat) + diag(var + omega^2).
    Persons with the same number of sessions are processed as one batch."""
    om2 = p["omega"] ** 2
    parts = []
    num = den = 0.0
    for idx in _person_groups(d):
        t = d.time[idx]
        z = d.slope_z[idx] if d.slope_z is not None else np.zeros_like(t)
        C = _kbatch(t, z, d, p)
        K = t.shape[1]
        C[:, np.arange(K), np.arange(K)] += var[idx] + om2 + 1e-10
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -np.inf, np.nan
        mm = m[idx]
        sol = np.linalg.solve(C, np.stack([np.ones_like(mm), mm], axis=2))   # (G, K, 2)
        a11 = sol[:, :, 0].sum(1); a1m = sol[:, :, 1].sum(1); amm = np.einsum("gk,gk->g", mm, sol[:, :, 1])
        logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        num += a1m.sum(); den += a11.sum()
        parts.append((a11, a1m, amm, logdet, K))
    mu0 = num / den
    ll = 0.0
    for a11, a1m, amm, logdet, K in parts:
        q = amm - 2 * mu0 * a1m + mu0 ** 2 * a11
        ll -= 0.5 * np.sum(logdet + q + K * LOG2PI)
    return float(ll), mu0


def _session_kernels(p, d: DesignData, a, w, T):
    """Per-session Gaussian kernel in the latent drift (b_j, S_j) and the constant term."""
    c = d.stim * np.where(d.upper == 1, a * (1.0 - w), -a * w)
    xb = d.X @ p["beta"] if p["beta"].size else np.zeros_like(T)
    sv2 = p["sv"] ** 2
    k = 1.0 + T * sv2                       # = 1 without drift variability
    S = np.bincount(d.sess, weights=T / k, minlength=d.J)
    b = np.bincount(d.sess, weights=(c - T * xb) / k, minlength=d.J)
    extra = float(np.sum((c * xb - 0.5 * T * xb ** 2) / k))
    if d.sv:
        extra += float(np.sum(c ** 2 * sv2 / (2 * k) - 0.5 * np.log(k)))
    return b, S, extra


def design_loglik(x, d: DesignData, return_mu0=False):
    if not d.has_outcomes:
        raise ValueError("this DesignData was built with outcomes=False (prediction only)")
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
    b, S, extra = _session_kernels(p, d, a, w, T)
    ll = zero + extra + np.sum(b ** 2 / (2 * S) + 0.5 * (LOG2PI - np.log(S)))
    cov, mu0 = _cov_profile(b / S, 1.0 / S, d, p)
    return (ll + cov, mu0) if return_mu0 else ll + cov


def p_upper(v, a, z):
    """Probability that a Wiener process with unit diffusion, drift v and
    boundary a, started z above the lower boundary, ends at the upper boundary:
    (1 - exp(-2 v z)) / (1 - exp(-2 v a))."""
    v, a, z = np.broadcast_arrays(np.asarray(v, float), np.asarray(a, float), np.asarray(z, float))
    neg = v < 0
    vv = np.abs(v)
    zz = np.where(neg, a - z, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.expm1(-2.0 * vv * zz) / np.expm1(-2.0 * vv * a)
    pr = np.where(vv * a < 1e-10, zz / a, pr)
    return np.where(neg, 1.0 - pr, pr)


def session_drift_predictive(x, d_train: DesignData, d_test: DesignData):
    """Mean and variance of the latent drift of each session in d_test, given a
    fit x to d_train. Test sessions are new sessions; their persons may appear in
    d_train. Each session drift is the Gaussian conditional of the person's
    session process given the training sessions' drift kernels (Proposition 2);
    persons absent from d_train get the population distribution."""
    if (d_test.kernel, d_test.sv, d_test.nugget, d_test.names) != (d_train.kernel, d_train.sv,
                                                                   d_train.nugget, d_train.names):
        raise ValueError("d_test must use the same model specification as d_train")
    p = _unpack(x, d_train)
    a = np.exp(d_train.Z @ p["alpha"]); w = _logistic(d_train.Q @ p["gamma"])
    T = d_train.rt - p["kappa"] * d_train.minrt_trial
    b, S, _ = _session_kernels(p, d_train, a, w, T)
    _, mu0 = design_loglik(x, d_train, return_mu0=True)
    m, var = b / S, 1.0 / S
    om2 = p["omega"] ** 2
    zeros = lambda n: np.zeros(n)
    if not d_train.has_outcomes:
        raise ValueError("d_train must contain outcomes")
    tr_slices = {d_train.person[sl.start]: sl for sl in d_train.person_slices}
    mean = np.empty(d_test.J); v_out = np.empty(d_test.J)
    for sl in d_test.person_slices:
        tt = d_test.time[sl]
        zt = d_test.slope_z[sl] if d_test.slope_z is not None else zeros(tt.size)
        trs = tr_slices.get(d_test.person[sl.start])
        if trs is None:
            K0 = _kcross(tt, tt, zt, zt, tt.min(), d_train, p)
            mean[sl] = mu0; v_out[sl] = np.diag(K0) + om2
            continue
        t = d_train.time[trs]
        z = d_train.slope_z[trs] if d_train.slope_z is not None else zeros(t.size)
        t1 = t.min()
        if d_train.kernel == "rw" and tt.min() < t1:
            raise ValueError(f"person {d_test.person[sl.start]!r}: a test session precedes the first training "
                             "session; the random walk is not defined before it starts")
        C = _kcross(t, t, z, z, t1, d_train, p)
        C[np.diag_indices_from(C)] += var[trs] + om2 + 1e-10
        Kx = _kcross(tt, t, zt, z, t1, d_train, p)
        A = cho_solve(cho_factor(C, lower=True), Kx.T)
        mean[sl] = mu0 + A.T @ (m[trs] - mu0)
        v_out[sl] = np.diag(_kcross(tt, tt, zt, zt, t1, d_train, p)) + om2 - np.sum(Kx * A.T, axis=1)
    scale = np.maximum(1.0, np.abs(v_out).max())
    if np.any(v_out < -1e-8 * scale):
        raise ValueError("negative predictive variance; check the test-session times")
    return mean, np.maximum(v_out, 0.0)


def predict_choices(x, d_train: DesignData, d_test: DesignData, n_nodes: int = 40) -> np.ndarray:
    """Predicted probability of an upper-boundary response for each trial of
    d_test from a fit x to d_train. The trial drift is
    stim * (session drift + X beta) plus across-trial variability; it is
    integrated over its predictive normal distribution by Gauss-Hermite
    quadrature. Test-trial RTs are not used."""
    p = _unpack(x, d_train)
    mean, var = session_drift_predictive(x, d_train, d_test)
    a = np.exp(d_test.Z @ p["alpha"]); w = _logistic(d_test.Q @ p["gamma"])
    xb = d_test.X @ p["beta"] if p["beta"].size else np.zeros(d_test.rt.size)
    mv = d_test.stim * (mean[d_test.sess] + xb)
    sd = np.sqrt(var[d_test.sess] + p["sv"] ** 2)
    nodes, wts = np.polynomial.hermite_e.hermegauss(n_nodes)
    wts = wts / wts.sum()
    v = mv[:, None] + sd[:, None] * nodes[None, :]
    return p_upper(v, a[:, None], (w * a)[:, None]) @ wts


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
        elif n == "log_ell":
            b.append((-7.0, 6.0))
        elif n == "log_sv":
            b.append((-6.0, 1.5))
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
    dyn = {"ou": thetas, "se": (1.0 / 0.3, 1.0 / 2.0, 1.0 / 50.0), "rw": (None,), "none": (None,)}[d.kernel]
    for th0 in dyn:
        for tau0 in taus:
            z = np.zeros(len(names))
            z[0] = np.log(1.5)
            z[names.index("logit_kappa")] = 2.0
            z[names.index("log_tau")] = np.log(tau0)
            if d.kernel == "ou":
                z[names.index("log_theta")] = np.log(th0)
                z[names.index("log_sd_stat")] = np.log(0.3)
            elif d.kernel == "se":
                z[names.index("log_ell")] = np.log(th0)
                z[names.index("log_sd_stat")] = np.log(0.3)
            elif d.kernel == "rw":
                z[names.index("log_sd_rw")] = np.log(0.1)
            if d.sv:
                z[names.index("log_sv")] = np.log(0.3)
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
    params.update({k: p[k] for k in ("kappa", "tau", "theta", "sigma", "omega", "sd_stat", "slope_sd", "rho",
                                     "ell", "sd_rw", "sv")})
    sed = dict(zip(names, sev))
    hl, ci, atb = np.nan, (np.nan, np.nan), ""
    if d.kernel == "ou":
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
