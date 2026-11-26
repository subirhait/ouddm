"""Hierarchical OU model for (possibly noisy) occasion-level drift rates.

For person i with observed series y_i at times t_i:

    y_i ~ N(mu0 * 1, tau^2 11' + K_i + omega^2 I),
    K_i[j, k] = sigma^2 / (2 theta) * exp(-theta |t_ij - t_ik|)

omega = 0 gives the 'naive' model that treats stage-1 estimates as the
true drift rates. omega > 0 adds a white-noise measurement-error term
(a nugget), which absorbs the stage-1 estimation error.

mu0 is profiled out by generalized least squares; the remaining
parameters are fit by maximum likelihood on the log scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


@dataclass
class OUFit:
    mu0: float
    tau: float
    theta: float
    sigma: float
    omega: float
    loglik: float
    converged: bool
    measurement_error: bool
    se_log_theta: float = np.nan

    @property
    def stationary_sd(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.theta)

    @property
    def halflife(self) -> float:
        return np.log(2) / self.theta

    def halflife_ci(self, level: float = 0.95) -> tuple[float, float]:
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        c = np.log(self.halflife)
        return float(np.exp(c - z * self.se_log_theta)), float(np.exp(c + z * self.se_log_theta))

    def as_dict(self) -> dict:
        lo, hi = self.halflife_ci()
        return {"mu0": self.mu0, "tau": self.tau, "theta": self.theta,
                "sigma": self.sigma, "omega": self.omega, "halflife": self.halflife,
                "se_log_theta": self.se_log_theta, "hl_lo": lo, "hl_hi": hi,
                "stationary_sd": self.stationary_sd, "loglik": self.loglik,
                "converged": self.converged}


def _split_series(person, time, y, obs_var=None):
    person = np.asarray(person)
    time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    obs_var = np.zeros_like(y) if obs_var is None else np.asarray(obs_var, dtype=float)
    ok = np.isfinite(y) & np.isfinite(time) & np.isfinite(obs_var)
    series = []
    for p in np.unique(person[ok]):
        m = ok & (person == p)
        order = np.argsort(time[m])
        series.append((time[m][order], y[m][order], obs_var[m][order]))
    return series


def _person_cov(t, tau, theta, sigma, omega, obs_var=None):
    d = np.abs(t[:, None] - t[None, :])
    K = (sigma ** 2 / (2.0 * theta)) * np.exp(-theta * d)
    nug = omega ** 2 + 1e-10 + (0.0 if obs_var is None else obs_var)
    return tau ** 2 + K + np.diag(np.broadcast_to(nug, t.shape))


def _profile_negloglik(logpars, series, measurement_error):
    tau, theta, sigma = np.exp(logpars[:3])
    omega = np.exp(logpars[3]) if measurement_error else 0.0
    # pass 1: GLS for mu0
    num = den = 0.0
    facs = []
    for t, y, ov in series:
        S = _person_cov(t, tau, theta, sigma, omega, ov)
        try:
            c = cho_factor(S, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e12, np.nan
        one = np.ones(t.size)
        Si1 = cho_solve(c, one, check_finite=False)
        num += Si1 @ y
        den += Si1 @ one
        facs.append(c)
    mu0 = num / den
    # pass 2: log-likelihood
    nll = 0.0
    for (t, y, _), c in zip(series, facs):
        r = y - mu0
        logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
        nll += 0.5 * (logdet + r @ cho_solve(c, r, check_finite=False) + t.size * np.log(2 * np.pi))
    return nll, mu0


def fit_ou(person, time, y, measurement_error: bool = False,
           start: dict | None = None, obs_var=None, se: bool = False) -> OUFit:
    """Maximum likelihood fit of the hierarchical OU model.

    obs_var: optional known per-observation measurement variances
    (added to the diagonal). Used by the exact collapsed likelihood.
    """
    series = _split_series(person, time, y, obs_var)
    yall = np.concatenate([s[1] for s in series])
    sd = np.std(yall)
    start = start or {}
    x0 = np.log([start.get("tau", 0.5 * sd), start.get("theta", 0.5),
                 start.get("sigma", sd)])
    if measurement_error:
        x0 = np.append(x0, np.log(start.get("omega", 0.5 * sd)))

    f = lambda lp: _profile_negloglik(lp, series, measurement_error)[0]
    best = None
    # starts on theta guard against the flat ridge between theta and omega;
    # starts on tau cover the basin in which a slow OU process stands in for
    # the person-level random intercept (tau near 0)
    for th0, ltau0 in [(x0[1], x0[0]), (np.log(0.2), x0[0]), (np.log(1.5), x0[0]),
                       (x0[1], np.log(max(0.05 * sd, 1e-3)))]:
        xs = x0.copy()
        xs[1] = th0
        xs[0] = ltau0
        res = minimize(f, xs, method="L-BFGS-B",
                       bounds=[(-8, 3), (-6, 3), (-8, 3)] + ([(-8, 3)] if measurement_error else []))
        if best is None or res.fun < best.fun:
            best = res
    nll, mu0 = _profile_negloglik(best.x, series, measurement_error)
    tau, theta, sigma = np.exp(best.x[:3])
    omega = float(np.exp(best.x[3])) if measurement_error else 0.0
    se_lt = np.nan
    if se:
        from .estimate import _num_hessian
        try:
            cov = np.linalg.inv(_num_hessian(f, best.x))
            if cov[1, 1] > 0:
                se_lt = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            pass
    return OUFit(mu0=float(mu0), tau=float(tau), theta=float(theta), sigma=float(sigma),
                 omega=omega, loglik=-float(nll), converged=bool(best.success),
                 measurement_error=measurement_error, se_log_theta=se_lt)
