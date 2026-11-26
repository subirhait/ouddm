"""Joint maximum likelihood for the OU-drift DDM using the exact collapsed
likelihood, with boundary separation a and non-decision time ter estimated.

Full log marginal likelihood of all choices and RTs:

    log L(a, ter, mu0, tau, theta, sigma)
      = sum_trials log f(T_r | v = 0, a)                       (zero-drift Wiener terms)
      + sum_j [ b_j^2 / (2 S_j) + 0.5 log(2 pi / S_j) ]
      + sum_i log N( m_i | mu0 1, tau^2 11' + K_i(theta, sigma) + diag(1/S_i) )

with T_r = rt_r - ter, S_j = sum_r T_jr = R_j - n_j ter, b_j = (a/2) d_j,
d_j = (#upper - #lower) in session j, m_j = b_j / S_j.

mu0 is profiled out by GLS. Also provided:

  * fit_two_stage_ml: the comparator that estimates (a, ter) shared across
    sessions with session drifts as free parameters (stage 1), then fits the
    OU model to the session drift MLEs m_j, either ignoring their sampling
    variance ('naive') or propagating it as known variance 1/S_j.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .ou import fit_ou
from .wiener import _log_g

LOG2PI = np.log(2 * np.pi)
THETA_BOUNDS = (-6.0, 4.0)   # log theta


@dataclass
class SessionData:
    rt: np.ndarray          # all trial RTs
    upper: np.ndarray       # 1 upper, 0 lower
    n: np.ndarray           # trials per session
    R: np.ndarray           # sum of RTs per session
    d: np.ndarray           # (#upper - #lower) per session
    time: np.ndarray        # session time
    person: np.ndarray      # person id per session
    person_slices: list = field(default_factory=list)
    min_rt: float = 0.0
    covariates: tuple = ()
    # drift-covariate sums (k = number of covariates); all None when k = 0
    XR: np.ndarray | None = None    # (J, k) sum_r RT_r x_r per session
    X1: np.ndarray | None = None    # (J, k) sum_r x_r per session
    Cs: np.ndarray | None = None    # (k,)   sum over all trials of sign_r x_r
    QR: np.ndarray | None = None    # (k, k) sum RT_r x_r x_r'
    Q1: np.ndarray | None = None    # (k, k) sum x_r x_r'

    @property
    def k(self) -> int:
        return len(self.covariates)

    @classmethod
    def from_trials(cls, trials: pd.DataFrame, covariates=()) -> "SessionData":
        covariates = tuple(covariates)
        req = {"person", "occasion", "time", "rt", "correct", *covariates}
        missing = req - set(trials.columns)
        if missing:
            raise ValueError(f"trials is missing columns: {sorted(missing)}")
        tr = trials.dropna(subset=["rt", "correct", *covariates])
        tr = tr[tr["correct"].isin([0, 1])].sort_values(["person", "time", "occasion"]).reset_index(drop=True)
        sgn = np.where(tr["correct"].to_numpy() == 1, 1.0, -1.0)
        g = tr.assign(sgn=sgn).groupby(["person", "occasion"], sort=False)
        s = g.agg(time=("time", "first"), n=("rt", "size"), R=("rt", "sum"), d=("sgn", "sum")).reset_index()
        s = s.sort_values(["person", "time"]).reset_index(drop=True)
        slices, start = [], 0
        for _, cnt in s.groupby("person", sort=False).size().items():
            slices.append(slice(start, start + cnt))
            start += cnt
        kw = {}
        if covariates:
            key = {(p, o): i for i, (p, o) in enumerate(zip(s["person"], s["occasion"]))}
            sidx = np.array([key[(p, o)] for p, o in zip(tr["person"], tr["occasion"])])
            X = tr[list(covariates)].to_numpy(float)
            rt = tr["rt"].to_numpy(float)
            J, k = len(s), X.shape[1]
            XR = np.zeros((J, k)); X1 = np.zeros((J, k))
            np.add.at(XR, sidx, rt[:, None] * X)
            np.add.at(X1, sidx, X)
            kw = dict(XR=XR, X1=X1, Cs=(sgn[:, None] * X).sum(0),
                      QR=(X * rt[:, None]).T @ X, Q1=X.T @ X)
        return cls(rt=tr["rt"].to_numpy(float), upper=tr["correct"].to_numpy(int),
                   n=s["n"].to_numpy(float), R=s["R"].to_numpy(float), d=s["d"].to_numpy(float),
                   time=s["time"].to_numpy(float), person=s["person"].to_numpy(),
                   person_slices=slices, min_rt=float(tr["rt"].min()), covariates=covariates, **kw)


# ---------- parameter transforms ----------
# x = [log a, eta (ter = min_rt * sigmoid(eta)), log tau, log theta, log sigma, beta_1..beta_k]

def _ter(eta, min_rt):
    return min_rt / (1.0 + np.exp(-eta))


def _eta(ter, min_rt):
    p = np.clip(ter / min_rt, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


_ZD_CACHE: dict = {}


def _zero_drift_term(data: SessionData, a: float, ter: float) -> float:
    """sum_r log f(T_r | v = 0, a, w = 1/2). Depends on (a, ter) only, so the
    last value is cached per data set (finite-difference gradients and
    Hessians change one parameter at a time)."""
    key = (id(data), float(a), float(ter))
    hit = _ZD_CACHE.get(id(data))
    if hit is not None and hit[0] == key:
        return hit[1]
    T = data.rt - ter
    val = float(np.sum(-2.0 * np.log(a) + _log_g(T / a ** 2, 0.5)))
    _ZD_CACHE[id(data)] = (key, val)
    return val


def _ou_profile(m, var, data: SessionData, tau, theta, sigma):
    """Profiled (over mu0) Gaussian log-likelihood of the session statistics."""
    num = den = 0.0
    facs = []
    for sl in data.person_slices:
        t = data.time[sl]
        dd = np.abs(t[:, None] - t[None, :])
        C = tau ** 2 + (sigma ** 2 / (2 * theta)) * np.exp(-theta * dd)
        C[np.diag_indices_from(C)] += var[sl] + 1e-10
        try:
            c = cho_factor(C, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, np.nan
        one = np.ones(t.size)
        Si1 = cho_solve(c, one, check_finite=False)
        num += Si1 @ m[sl]
        den += Si1.sum()
        facs.append(c)
    mu0 = num / den
    ll = 0.0
    for sl, c in zip(data.person_slices, facs):
        r = m[sl] - mu0
        ll -= 0.5 * (2 * np.sum(np.log(np.diag(c[0]))) + r @ cho_solve(c, r, check_finite=False)
                     + r.size * LOG2PI)
    return ll, mu0


def exact_loglik(x, data: SessionData, return_mu0=False):
    """Exact log marginal likelihood. With drift covariates (trial drift
    v = u_i(t) + x'beta), each session keeps the quadratic form in the latent
    state u with b_j(beta) = (a/2) d_j - G_j beta, G_j = sum_r T_r x_r, plus a
    beta-only term (a/2) sum_r sign_r x_r'beta - 0.5 beta' (sum_r T_r x_r x_r') beta."""
    a = np.exp(x[0])
    ter = _ter(x[1], data.min_rt)
    tau, theta, sigma = np.exp(x[2:5])
    S = data.R - data.n * ter
    b = 0.5 * a * data.d
    extra = 0.0
    if data.k:
        beta = np.asarray(x[5:5 + data.k], float)
        G = data.XR - ter * data.X1
        b = b - G @ beta
        extra = 0.5 * a * data.Cs @ beta - 0.5 * beta @ (data.QR - ter * data.Q1) @ beta
    ll = _zero_drift_term(data, a, ter) + np.sum(b ** 2 / (2 * S) + 0.5 * (LOG2PI - np.log(S))) + extra
    ou, mu0 = _ou_profile(b / S, 1.0 / S, data, tau, theta, sigma)
    return (ll + ou, mu0) if return_mu0 else ll + ou


def _num_hessian(f, x, h=1e-3):
    k = x.size
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


@dataclass
class JointFit:
    a: float
    ter: float
    mu0: float
    tau: float
    theta: float
    sigma: float
    loglik: float
    converged: bool
    se_log_theta: float = np.nan
    cov_x: np.ndarray | None = None
    beta: dict = field(default_factory=dict)
    se_beta: dict = field(default_factory=dict)
    at_bound: str = ""

    @property
    def halflife(self) -> float:
        return np.log(2) / self.theta

    def halflife_ci(self, level: float = 0.95) -> tuple[float, float]:
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        lo = np.log(self.halflife) - z * self.se_log_theta
        hi = np.log(self.halflife) + z * self.se_log_theta
        return float(np.exp(lo)), float(np.exp(hi))

    def as_dict(self) -> dict:
        lo, hi = self.halflife_ci()
        out = {"a": self.a, "ter": self.ter, "mu0": self.mu0, "tau": self.tau, "theta": self.theta,
               "sigma": self.sigma, "halflife": self.halflife, "se_log_theta": self.se_log_theta,
               "hl_lo": lo, "hl_hi": hi, "loglik": self.loglik, "converged": self.converged,
               "theta_at_bound": self.at_bound}
        for k, v in self.beta.items():
            out[f"beta_{k}"] = v
            out[f"se_beta_{k}"] = self.se_beta.get(k, np.nan)
        return out


def _stage1_shared(data: SessionData, a0=1.0, ter0=None):
    """Fixed-effect ML of (a, ter) with each session drift free:
    maximize sum log f(T|0,a) + sum b_j^2 / (2 S_j)."""
    ter0 = 0.5 * data.min_rt if ter0 is None else min(ter0, 0.95 * data.min_rt)

    def nll(z):
        a = np.exp(z[0]); ter = _ter(z[1], data.min_rt)
        S = data.R - data.n * ter
        b = 0.5 * a * data.d
        return -(_zero_drift_term(data, a, ter) + np.sum(b ** 2 / (2 * S)))

    res = minimize(nll, np.array([np.log(a0), _eta(ter0, data.min_rt)]), method="L-BFGS-B",
                   bounds=[(-3, 3), (-12, 12)])
    a = float(np.exp(res.x[0])); ter = float(_ter(res.x[1], data.min_rt))
    S = data.R - data.n * ter
    b = 0.5 * a * data.d
    return a, ter, b / S, 1.0 / S


def fit_two_stage_ml(trials: pd.DataFrame, propagate: bool = True, data: SessionData | None = None,
                     se: bool = False):
    """Two-stage comparator with (a, ter) shared across sessions.

    Stage 1: (a, ter) by fixed-effect ML with free session drifts; session
    drift MLEs m_j = b_j / S_j and their inverse observed information 1/S_j.
    Stage 2: hierarchical OU fit to m_j, with var 1/S_j added (propagate=True)
    or ignored (propagate=False).
    """
    data = data or SessionData.from_trials(trials)
    a, ter, m, var = _stage1_shared(data)
    fit = fit_ou(data.person, data.time, m, obs_var=var if propagate else None, se=se)
    return fit, {"a": a, "ter": ter}


def fit_exact(trials: pd.DataFrame, se: bool = True, data: SessionData | None = None,
              covariates=()) -> JointFit:
    """Joint ML of (a, ter, mu0, tau, theta, sigma[, beta]) from the exact collapsed likelihood.

    covariates: names of trial-level columns x with trial drift
    v = u_i(t) + x'beta, where u_i is the OU process. Do not include an
    intercept (mu0 plays that role); centered covariates are recommended.

    Starting values come from the shared-nuisance two-stage fit. Standard
    errors use the numerical Hessian of the (mu0-profiled) log-likelihood.
    """
    data = data or SessionData.from_trials(trials, covariates)
    k = data.k
    a0, ter0, m, var = _stage1_shared(data)
    ou0 = fit_ou(data.person, data.time, m, obs_var=var)
    f = lambda x: -exact_loglik(x, data)
    bounds = [(-3, 3), (-12, 12), (-8, 3), THETA_BOUNDS, (-8, 3)] + [(-10, 10)] * k
    # The likelihood can have two basins: a slow OU process with tau near 0
    # can mimic a person-level random intercept. Starting tau both at the
    # two-stage value and at half the SD of the session statistics covers both.
    tau_big = max(0.5 * float(np.std(m)), 0.05)
    starts = [(ou0.theta, max(ou0.tau, 1e-3)), (ou0.theta, tau_big), (1.0, tau_big)]
    best = None
    for th0, tau0 in starts:
        x0 = np.concatenate([[np.log(a0), _eta(ter0, data.min_rt), np.log(tau0),
                              np.clip(np.log(th0), *THETA_BOUNDS), np.log(max(ou0.sigma, 1e-3))], np.zeros(k)])
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    ll, mu0 = exact_loglik(x, data, return_mu0=True)
    se_lt, cov = np.nan, None
    se_b = np.full(k, np.nan)
    if se:
        try:
            cov = np.linalg.inv(_num_hessian(f, x))
            if cov[3, 3] > 0:
                se_lt = float(np.sqrt(cov[3, 3]))
            dg = np.diag(cov)[5:5 + k]
            se_b = np.where(dg > 0, np.sqrt(np.abs(dg)), np.nan)
        except np.linalg.LinAlgError:
            pass
    tau, theta, sigma = np.exp(x[2:5])
    tol = 1e-6
    at_bound = "upper" if x[3] >= THETA_BOUNDS[1] - tol else ("lower" if x[3] <= THETA_BOUNDS[0] + tol else "")
    return JointFit(a=float(np.exp(x[0])), ter=float(_ter(x[1], data.min_rt)), mu0=float(mu0),
                    tau=float(tau), theta=float(theta), sigma=float(sigma), loglik=float(ll),
                    converged=bool(best.success), se_log_theta=se_lt, cov_x=cov,
                    beta={c: float(b) for c, b in zip(data.covariates, x[5:5 + k])},
                    se_beta={c: float(sb) for c, sb in zip(data.covariates, se_b)},
                    at_bound=at_bound)
