"""Simulation of drift diffusion data whose drift rate follows an
Ornstein-Uhlenbeck (OU) process across measurement occasions.

Model
-----
Person i, occasion j at time t_ij (days, possibly irregular).

Latent drift rate (OU, exact discretization):
    v_i(t_1)        ~ N(mu_i, sigma^2 / (2 theta))
    v_i(t_j) | prev ~ N(mu_i + (v_prev - mu_i) exp(-theta dt),
                        sigma^2 / (2 theta) (1 - exp(-2 theta dt)))
Person home base:
    mu_i ~ N(mu0, tau^2)

Trials within an occasion: Wiener diffusion with drift v_i(t_j),
boundary separation a, unbiased start a/2, diffusion coefficient s,
non-decision time ter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OUDDMParams:
    """Population parameters of the OU-drift DDM."""

    mu0: float = 1.5      # population mean home base of drift rate
    tau: float = 0.3      # between-person SD of home base
    theta: float = 0.5    # OU reversion speed (per day)
    sigma: float = 0.35   # OU diffusion (volatility) of drift rate
    a: float = 1.2        # boundary separation
    ter: float = 0.3      # non-decision time (s)
    s: float = 1.0        # within-trial diffusion coefficient (scaling)
    sv: float = 0.0       # across-trial SD of drift within a session

    @property
    def stationary_sd(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.theta)


@dataclass
class DesignParams:
    """Study design."""

    n_persons: int = 50
    n_occasions: int = 10
    n_trials: int = 40
    p_skip: float = 0.15      # probability an occasion is skipped (gap of 2 days)
    jitter: float = 0.2       # SD of time-of-day jitter (days)
    seed_offset: int = field(default=0, repr=False)


def simulate_times(design: DesignParams, rng: np.random.Generator) -> np.ndarray:
    """Irregular measurement times, shape (n_persons, n_occasions)."""
    gaps = 1.0 + rng.binomial(1, design.p_skip, size=(design.n_persons, design.n_occasions))
    gaps[:, 0] = 0.0
    base = np.cumsum(gaps, axis=1)
    times = base + rng.normal(0.0, design.jitter, size=base.shape)
    # keep strictly increasing
    times = np.maximum.accumulate(times, axis=1)
    times[:, 1:] = np.maximum(times[:, 1:], times[:, :-1] + 1e-3)
    return times


def simulate_ou(times: np.ndarray, mu: np.ndarray, theta: float, sigma: float,
                rng: np.random.Generator) -> np.ndarray:
    """Exact OU paths at the given times. times: (P, J); mu: (P,)."""
    P, J = times.shape
    v = np.empty((P, J))
    sd_stat = sigma / np.sqrt(2.0 * theta)
    v[:, 0] = mu + sd_stat * rng.standard_normal(P)
    for j in range(1, J):
        dt = times[:, j] - times[:, j - 1]
        rho = np.exp(-theta * dt)
        sd = sd_stat * np.sqrt(1.0 - rho ** 2)
        v[:, j] = mu + (v[:, j - 1] - mu) * rho + sd * rng.standard_normal(P)
    return v


def simulate_wiener(v: np.ndarray, a: float, s: float, rng: np.random.Generator,
                    dt: float = 1e-3, max_time: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """First-passage times of Brownian motion with drift between 0 and a,
    started at a/2.

    Uses Euler steps with a Brownian-bridge crossing correction: after a
    step from x0 to x1 that stays inside (0, a), the path is counted as
    having crossed the upper (lower) boundary with probability
    exp(-2 (a - x0)(a - x1) / (s^2 dt))  (exp(-2 x0 x1 / (s^2 dt))).
    This removes the main discretization bias of plain Euler.

    Returns (decision_time, upper) where upper is 1 for the upper
    (correct, when v > 0) boundary, 0 for the lower boundary, and -1 for
    trials with no boundary hit by max_time (decision_time is NaN for these).
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    x = np.full(n, a / 2.0)
    t_out = np.full(n, np.nan)
    up_out = np.full(n, -1, dtype=np.int8)
    active = np.arange(n)
    sd = s * np.sqrt(dt)
    t = 0.0
    var2 = s * s * dt
    while active.size > 0 and t < max_time:
        t += dt
        x0 = x[active]
        x1 = x0 + v[active] * dt + sd * rng.standard_normal(active.size)
        hit_up = x1 >= a
        hit_lo = x1 <= 0.0
        inside = ~(hit_up | hit_lo)
        # bridge correction for paths that end inside
        if inside.any():
            xi0, xi1 = x0[inside], x1[inside]
            p_up = np.exp(-2.0 * (a - xi0) * (a - xi1) / var2)
            p_lo = np.exp(-2.0 * xi0 * xi1 / var2)
            u = rng.random(xi0.size)
            br_up = u < p_up
            br_lo = (~br_up) & (u < p_up + p_lo)
            idx_inside = np.flatnonzero(inside)
            hit_up[idx_inside[br_up]] = True
            hit_lo[idx_inside[br_lo]] = True
        done = hit_up | hit_lo
        if done.any():
            ids = active[done]
            # place exit uniformly within the step (half-step on average)
            t_out[ids] = t - dt * rng.random(ids.size)
            up_out[ids] = hit_up[done].astype(np.int8)
        x[active] = x1
        active = active[~done]
    return t_out, up_out


def simulate_study(pars: OUDDMParams, design: DesignParams,
                   rng: np.random.Generator, times: np.ndarray | None = None,
                   x_session: np.ndarray | None = None, beta=None) -> dict:
    """Simulate a full study.

    times: optional fixed schedule, shape (n_occasions,) shared by all
    persons or (n_persons, n_occasions). If None, irregular times are drawn.
    x_session, beta: optional session-level drift covariates, shape
    (n_occasions, k) shared by persons or (n_persons, n_occasions, k), with
    effects beta (k,). Trial drift = OU state + x'beta. Covariates are
    returned as columns x0, x1, ... of the trial table.

    Returns a dict with:
      'trials'    : DataFrame (person, occasion, time, rt, correct)
      'occasions' : DataFrame (person, occasion, time, v_true)
      'mu'        : array of true person home bases
    """
    P, J, N = design.n_persons, design.n_occasions, design.n_trials
    if times is None:
        times = simulate_times(design, rng)
    else:
        times = np.broadcast_to(np.asarray(times, float), (P, J)).copy()
    mu = pars.mu0 + pars.tau * rng.standard_normal(P)
    v = simulate_ou(times, mu, pars.theta, pars.sigma, rng)

    shift = np.zeros((P, J))
    X = None
    if x_session is not None:
        X = np.asarray(x_session, float)
        if X.ndim == 2:
            X = np.broadcast_to(X, (P, J, X.shape[1]))
        shift = X @ np.asarray(beta, float)
    v_trial = np.repeat((v + shift).ravel(), N)
    if pars.sv > 0:
        v_trial = v_trial + pars.sv * rng.standard_normal(v_trial.size)
    dtime, upper = simulate_wiener(v_trial, pars.a, pars.s, rng)
    person = np.repeat(np.arange(P), J * N)
    occ = np.tile(np.repeat(np.arange(J), N), P)
    trials = pd.DataFrame({
        "person": person,
        "occasion": occ,
        "time": np.repeat(times.ravel(), N),
        "rt": dtime + pars.ter,
        "correct": upper,
    })
    if X is not None:
        for c in range(X.shape[2]):
            trials[f"x{c}"] = np.repeat(X[:, :, c].ravel(), N)
    # timed-out trials (no response by max_time) are dropped, as in a
    # task with a response deadline; their count is kept for reporting
    n_timeout = int((upper < 0).sum())
    trials = trials[trials["correct"] >= 0].reset_index(drop=True)
    occasions = pd.DataFrame({
        "person": np.repeat(np.arange(P), J),
        "occasion": np.tile(np.arange(J), P),
        "time": times.ravel(),
        "v_true": v.ravel(),
    })
    return {"trials": trials, "occasions": occasions, "mu": mu, "times": times, "v": v,
            "n_timeout": n_timeout}
