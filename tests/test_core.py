import numpy as np
import pandas as pd

from ouddm import (DesignParams, OUDDMParams, ez_diffusion, fit_ou, simulate_ou,
                   simulate_wiener)


def analytic_pc(v, a, s=1.0):
    return 1.0 / (1.0 + np.exp(-a * v / s ** 2))


def analytic_mdt(v, a, s=1.0):
    return (a / (2.0 * v)) * np.tanh(a * v / (2.0 * s ** 2))


def test_wiener_sampler_matches_analytic():
    rng = np.random.default_rng(1)
    for v, a in [(1.5, 1.2), (0.5, 1.0), (2.5, 1.5)]:
        t, up = simulate_wiener(np.full(60000, v), a, 1.0, rng)
        assert np.isfinite(t).all()
        assert abs(up.mean() - analytic_pc(v, a)) < 0.006
        assert abs(t.mean() - analytic_mdt(v, a)) / analytic_mdt(v, a) < 0.015


def test_ez_recovers_parameters_large_n():
    rng = np.random.default_rng(2)
    v, a, ter = 1.5, 1.2, 0.3
    t, up = simulate_wiener(np.full(100000, v), a, 1.0, rng)
    rt = t + ter
    vh, ah, th = ez_diffusion(up.mean(), rt[up == 1].var(ddof=1), rt[up == 1].mean(), up.size)
    assert abs(vh - v) < 0.05
    assert abs(ah - a) < 0.03
    assert abs(th - ter) < 0.02


def test_ou_fit_recovers_parameters_without_noise():
    rng = np.random.default_rng(3)
    P, J = 400, 12
    times = np.tile(np.arange(J, dtype=float), (P, 1)) + rng.normal(0, 0.2, (P, J))
    times = np.sort(times, axis=1)
    mu = 1.5 + 0.3 * rng.standard_normal(P)
    v = simulate_ou(times, mu, theta=0.5, sigma=0.35, rng=rng)
    fit = fit_ou(np.repeat(np.arange(P), J), times.ravel(), v.ravel())
    assert abs(fit.theta - 0.5) < 0.08
    assert abs(fit.sigma - 0.35) < 0.04
    assert abs(fit.tau - 0.3) < 0.04
    assert abs(fit.mu0 - 1.5) < 0.05


def test_ou_fit_with_known_noise_recovers_omega():
    rng = np.random.default_rng(4)
    P, J = 400, 12
    times = np.tile(np.arange(J, dtype=float), (P, 1))
    mu = 1.5 + 0.3 * rng.standard_normal(P)
    v = simulate_ou(times, mu, theta=0.5, sigma=0.35, rng=rng)
    y = v + 0.3 * rng.standard_normal(v.shape)
    naive = fit_ou(np.repeat(np.arange(P), J), times.ravel(), y.ravel())
    corr = fit_ou(np.repeat(np.arange(P), J), times.ravel(), y.ravel(), measurement_error=True)
    assert naive.theta > 0.8          # noise makes dynamics look faster
    assert abs(corr.theta - 0.5) < 0.15
    assert abs(corr.omega - 0.3) < 0.05
