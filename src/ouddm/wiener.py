"""Wiener first-passage-time density (two boundaries, s = 1).

Uses the Navarro & Fuss (2009) small-time and large-time series on the
normalized time u = t / a^2. Lower-boundary density:

    f_lo(t | v, a, w) = a^{-2} exp(-v a w - v^2 t / 2) g(u | w)

where g is the zero-drift, unit-boundary density. The upper-boundary
density uses v -> -v and w -> 1 - w.

Key property used by ouddm: for a centered start (w = 1/2) the drift
enters only through exp(+/- a v / 2 - v^2 t / 2), so the session
log-likelihood is exactly quadratic in a session-constant drift v.
"""

from __future__ import annotations

import numpy as np


def _signed_lse(logmag, sgn):
    """log(sum_k sgn_k exp(logmag_k)) along axis 1; -inf where the sum is <= 0."""
    m = logmag.max(axis=1, keepdims=True)
    tot = np.sum(sgn * np.exp(logmag - m), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, np.log(np.where(tot > 0, tot, 1.0)) + m[:, 0], -np.inf)


def _log_g(u: np.ndarray, w: float, k_small: int = 5, k_large: int = 12) -> np.ndarray:
    """log of the zero-drift, unit-boundary first-passage density at normalized
    time u (Navarro & Fuss, 2009). Both series are summed on the log scale
    (signed log-sum-exp), so very short times give their true, very negative
    log density instead of underflowing to a numerical floor. With the switch
    at u = 1, the omitted terms are below exp(-30) relative to the sum."""
    u = np.asarray(u, dtype=float)
    if np.ndim(w):
        w = np.broadcast_to(np.asarray(w, dtype=float), u.shape)
    out = np.full(u.shape, -np.inf)
    small = (u > 0) & (u < 1.0)
    if small.any():
        us = u[small][:, None]
        k = np.arange(-k_small, k_small + 1)[None, :]
        wk = (w[small][:, None] if np.ndim(w) else w) + 2.0 * k
        logmag = np.log(np.abs(wk)) - wk ** 2 / (2.0 * us)
        out[small] = _signed_lse(logmag, np.sign(wk)) - 0.5 * np.log(2.0 * np.pi * us[:, 0] ** 3)
    large = u >= 1.0
    if large.any():
        ul = u[large][:, None]
        k = np.arange(1, k_large + 1)[None, :]
        sk = np.sin(k * np.pi * (w[large][:, None] if np.ndim(w) else w))
        logmag = np.log(k) - (k ** 2) * np.pi ** 2 * ul / 2.0 + np.log(np.maximum(np.abs(sk), 1e-300))
        out[large] = np.log(np.pi) + _signed_lse(logmag, np.sign(sk))
    return out


def wiener_logpdf(t, upper, v, a, w: float = 0.5) -> np.ndarray:
    """Log density of decision time t (> 0) and response (upper = 1/0)."""
    t = np.asarray(t, dtype=float)
    upper = np.asarray(upper).astype(bool)
    v = np.broadcast_to(np.asarray(v, dtype=float), t.shape)
    vv = np.where(upper, -v, v)
    ww = np.where(upper, 1.0 - w, w)
    out = np.empty_like(t)
    for wval in np.unique(ww):
        m = ww == wval
        out[m] = (-2.0 * np.log(a) - vv[m] * a * wval - vv[m] ** 2 * t[m] / 2.0
                  + _log_g(t[m] / a ** 2, wval))
    return out


def wiener_cdf(t, upper: int, v: float, a: float, w: float = 0.5, n_grid: int = 4000) -> np.ndarray:
    """Defective CDF P(T <= t, response) by trapezoid integration of the density."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tmax = t.max()
    grid = np.concatenate([[0.0], np.geomspace(1e-5, tmax, n_grid)])
    dens = np.zeros_like(grid)
    dens[1:] = np.exp(wiener_logpdf(grid[1:], np.full(n_grid, upper), v, a, w))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return np.interp(t, grid, cum)


def information_ratio_choice(v, a):
    """Fraction of per-trial Fisher information about drift v retained by
    the choice alone, relative to choice + RT, for the centered DDM with
    s = 1 and known a, ter:  x / sinh(x) with x = a v (1 at x = 0)."""
    x = np.asarray(a * np.asarray(v, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.abs(x) < 1e-8, 1.0, x / np.sinh(x))
    return r


def wald_logpdf(t, v, a):
    """Log density of the first passage of Brownian motion with drift v and unit
    diffusion, started at 0, through a single boundary at a > 0 (shifted Wald
    without the shift)."""
    t = np.asarray(t, float)
    return np.log(a) - 0.5 * np.log(2 * np.pi * t ** 3) - (a - v * t) ** 2 / (2 * t)


def first_passage_kernel(hit, tau):
    """Gaussian kernel of a session's likelihood in a constant drift v
    (Theorem 1). For Brownian motion with unit diffusion stopped at the first
    passage through any boundaries, the log likelihood is
        log L(v) - log L(0) = b v - S v^2 / 2,
    with b = sum of (boundary value hit - start point) and S = sum of decision
    times. hit: boundary value minus start point per trial (e.g. a(1 - w) or
    -a w for the DDM, a for a single-boundary model); tau: decision times.
    Returns (b, S)."""
    return float(np.sum(hit)), float(np.sum(tau))
