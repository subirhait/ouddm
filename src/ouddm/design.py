"""Design calculator: expected precision of the drift half-life for a study
schedule, before data are collected.

Under the exact collapsed likelihood, session j contributes a Gaussian
observation of its drift with variance 1/S_j, S_j = total decision time.
The calculator replaces 1/S_j by its plug-in expectation 1/(n I_trial),
where I_trial is the per-trial Fisher information about drift:

    choice + RT :  I_trial = E[T | v] = (a / (2v)) tanh(a v / 2)
    choice only :  I_trial = a^2 p (1 - p),  p = 1 / (1 + exp(-a v))

evaluated at the population home base v = mu0. The expected Fisher
information of the hierarchical Gaussian OU model then follows in closed
form, giving the standard error of log half-life (= SE of log theta).
The oracle (session drifts observed without error) sets the variance to 0.

This is an approximation (plug-in session variance, a and ter treated as
known). Rankings of designs should be checked by simulation; see
examples/design_study.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trial_information(v: float, a: float, rt: bool = True) -> float:
    if rt:
        return (a / (2.0 * v)) * np.tanh(a * v / 2.0) if v != 0 else a ** 2 / 4.0
    p = 1.0 / (1.0 + np.exp(-a * v))
    return a ** 2 * p * (1.0 - p)


def schedule(n_sessions: int, window: float, spacing: str = "even", burst_gap: float = 0.1) -> np.ndarray:
    """Session times (days) over [0, window].

    even  : equally spaced.
    burst : sessions in pairs, pair starts equally spaced, sessions within a
            pair burst_gap days apart (a mix of short and long gaps).
    """
    if spacing == "even":
        return np.linspace(0.0, window, n_sessions)
    if spacing == "burst":
        if n_sessions % 2:
            raise ValueError("burst spacing needs an even number of sessions")
        starts = np.linspace(0.0, window - burst_gap, n_sessions // 2)
        return np.sort(np.concatenate([starts, starts + burst_gap]))
    raise ValueError("spacing must be 'even' or 'burst'")


def _person_information(times, obs_var, tau, theta, sigma):
    t = np.asarray(times, float)
    D = np.abs(t[:, None] - t[None, :])
    E = np.exp(-theta * D)
    c = sigma ** 2 / (2 * theta)
    S = tau ** 2 + c * E + np.diag(np.broadcast_to(obs_var, t.shape))
    Si = np.linalg.inv(S)
    dS = [2 * tau ** 2 * np.ones_like(S),            # d / d log tau
          -c * E - 0.5 * sigma ** 2 * D * E,         # d / d log theta
          2 * c * E]                                 # d / d log sigma
    I = np.empty((3, 3))
    for k in range(3):
        for l in range(k, 3):
            I[k, l] = I[l, k] = 0.5 * np.trace(Si @ dS[k] @ Si @ dS[l])
    return I


def design_information(times, n_trials: int, a: float, mu0: float, tau: float, theta: float,
                       sigma: float, n_persons: int = 1, rt: bool = True) -> dict:
    """Expected SE of log half-life for a common schedule given to n_persons.

    Returns se_log_halflife, se_oracle, relative_efficiency (oracle variance /
    design variance; 1 = as good as observing the true session drifts), and
    the plug-in session variance.
    """
    v_obs = 1.0 / (n_trials * trial_information(mu0, a, rt))
    I = n_persons * _person_information(times, v_obs, tau, theta, sigma)
    I0 = n_persons * _person_information(times, 0.0, tau, theta, sigma)
    se = float(np.sqrt(np.linalg.inv(I)[1, 1]))
    se0 = float(np.sqrt(np.linalg.inv(I0)[1, 1]))
    return {"se_log_halflife": se, "se_oracle": se0, "relative_efficiency": (se0 / se) ** 2,
            "session_var": v_obs}


def compare_designs(budget: int, sessions: list[int], window: float, halflife: float,
                    a: float = 1.2, mu0: float = 1.5, tau: float = 0.3, stationary_sd: float = 0.35,
                    n_persons: int = 50, spacings=("even", "burst"), rt: bool = True) -> pd.DataFrame:
    """Fixed per-person trial budget split into K sessions of budget/K trials."""
    theta = np.log(2) / halflife
    sigma = stationary_sd * np.sqrt(2 * theta)
    rows = []
    for K in sessions:
        n = budget // K
        for sp in spacings:
            try:
                t = schedule(K, window, sp)
            except ValueError:
                continue
            r = design_information(t, n, a, mu0, tau, theta, sigma, n_persons, rt)
            rows.append({"sessions": K, "trials_per_session": n, "spacing": sp, "halflife": halflife, **r})
    return pd.DataFrame(rows)
