"""Exact collapsed (marginal) likelihood for the OU-drift DDM.

With s = 1, a centered start (w = 1/2), and drift v_j constant within
session j, the session log-likelihood is exactly quadratic in v_j:

    l_j(v) - l_j(0) = b_j v - S_j v^2 / 2,
    b_j = (a / 2) (N_j+ - N_j-),   S_j = sum_r T_jr  (decision times).

So, up to a factor free of v, session j contributes a Gaussian
'observation' m_j = b_j / S_j of v_j with known variance 1 / S_j.
Because the OU states and person means are Gaussian, all of them
integrate out exactly:

    log L = sum_j l_j(0) + sum_j [ b_j^2 / (2 S_j) + 0.5 log(2 pi / S_j) ]
            + sum_i log N( m_i | mu0 1, tau^2 11' + K_i(theta, sigma) + diag(1/S_i) ).

The first two sums depend on (a, ter) and the data only. For fixed
(a, ter) the OU parameters are estimated from the last term, which is
the hierarchical OU model with known heteroskedastic noise. The noise
variance 1/S_j is exact and session-specific: it shrinks with more
trials and with longer decision times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .ou import OUFit, fit_ou
from .wiener import wiener_logpdf


def session_summaries(trials: pd.DataFrame, a: float, ter: float) -> pd.DataFrame:
    """Per person-session sufficient statistics for the drift.

    Returns person, occasion, time, n, b, S, m (= b/S), var (= 1/S).
    Trials with rt <= ter are invalid under the model and raise.
    """
    T = trials["rt"].to_numpy() - ter
    if np.any(T <= 0):
        raise ValueError("some rt <= ter; decision times must be positive")
    df = trials.assign(T=T, sgn=np.where(trials["correct"].to_numpy() == 1, 1.0, -1.0))
    g = df.groupby(["person", "occasion"], sort=True)
    out = g.agg(time=("time", "first"), n=("T", "size"), S=("T", "sum"), d=("sgn", "sum")).reset_index()
    out["b"] = 0.5 * a * out["d"]
    out["m"] = out["b"] / out["S"]
    out["var"] = 1.0 / out["S"]
    return out.drop(columns="d")


def collapsed_constant(trials: pd.DataFrame, summ: pd.DataFrame, a: float, ter: float) -> float:
    """sum_j l_j(0) + sum_j [b_j^2/(2 S_j) + 0.5 log(2 pi / S_j)]: the part of the
    log marginal likelihood free of the OU parameters (needed to estimate a, ter)."""
    l0 = wiener_logpdf(trials["rt"].to_numpy() - ter, trials["correct"].to_numpy(), 0.0, a).sum()
    return float(l0 + np.sum(summ["b"] ** 2 / (2 * summ["S"]) + 0.5 * np.log(2 * np.pi / summ["S"])))


def collapsed_loglik(trials: pd.DataFrame, a: float, ter: float, mu0: float, tau: float,
                     theta: float, sigma: float) -> float:
    """Exact log marginal likelihood of all choices and RTs, with every session
    drift and person home base integrated out."""
    summ = session_summaries(trials, a, ter)
    ll = collapsed_constant(trials, summ, a, ter)
    for _, g in summ.groupby("person"):
        t = g["time"].to_numpy()
        d = np.abs(t[:, None] - t[None, :])
        C = tau ** 2 + sigma ** 2 / (2 * theta) * np.exp(-theta * d) + np.diag(g["var"].to_numpy())
        ll += multivariate_normal(mean=np.full(t.size, mu0), cov=C).logpdf(g["m"].to_numpy())
    return float(ll)


def fit_collapsed(trials: pd.DataFrame, a: float, ter: float) -> tuple[OUFit, pd.DataFrame]:
    """ML estimates of (mu0, tau, theta, sigma) from the exact collapsed
    likelihood, for known boundary separation a and non-decision time ter."""
    summ = session_summaries(trials, a, ter)
    fit = fit_ou(summ["person"], summ["time"], summ["m"], obs_var=summ["var"])
    return fit, summ
