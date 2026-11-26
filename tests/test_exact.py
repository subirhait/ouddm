import numpy as np
import pandas as pd

from ouddm import (DesignParams, OUDDMParams, collapsed_loglik, fit_collapsed,
                   information_ratio_choice, session_summaries, simulate_study,
                   simulate_wiener, wiener_cdf, wiener_logpdf)


def test_density_normalizes_and_matches_choice_probability():
    for v, a in [(1.5, 1.2), (-0.7, 1.0), (2.5, 1.5)]:
        p_up = wiener_cdf([30.0], 1, v, a)[0]
        p_lo = wiener_cdf([30.0], 0, v, a)[0]
        assert abs(p_up + p_lo - 1.0) < 2e-4
        assert abs(p_up - 1 / (1 + np.exp(-a * v))) < 2e-4


def test_loglik_is_exactly_quadratic_in_drift():
    rng = np.random.default_rng(10)
    a = 1.2
    t, up = simulate_wiener(np.full(25, 1.3), a, 1.0, rng)
    d = np.where(up == 1, 1.0, -1.0).sum()
    S = t.sum()
    l0 = wiener_logpdf(t, up, 0.0, a).sum()
    for v in [-2.0, -0.3, 0.8, 1.7, 3.5]:
        lv = wiener_logpdf(t, up, v, a).sum()
        assert abs((lv - l0) - (0.5 * a * d * v - 0.5 * S * v ** 2)) < 1e-9


def test_sampler_rt_quantiles_match_exact_cdf_and_step_size():
    v, a = 1.5, 1.2
    t, up = simulate_wiener(np.full(80000, v), a, 1.0, np.random.default_rng(11))
    t2, up2 = simulate_wiener(np.full(80000, v), a, 1.0, np.random.default_rng(12), dt=2e-4)
    for resp in (1, 0):
        p_resp = wiener_cdf([30.0], resp, v, a)[0]
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            tq = np.quantile(t[up == resp], q)
            # exact conditional CDF at the simulated quantile should equal q
            assert abs(wiener_cdf([tq], resp, v, a)[0] / p_resp - q) < (0.012 if resp == 1 else 0.03)
            tq2 = np.quantile(t2[up2 == resp], q)
            assert abs(tq - tq2) / tq2 < (0.03 if resp == 1 else 0.06)


def test_no_trial_coded_as_hit_without_response():
    t, up = simulate_wiener(np.full(2000, 0.0), 3.0, 1.0, np.random.default_rng(13), max_time=0.5)
    assert set(np.unique(up)) <= {-1, 0, 1}
    assert np.all(np.isnan(t[up == -1]))
    assert np.all(np.isfinite(t[up >= 0]))
    assert (up == -1).sum() > 0


def test_collapsed_loglik_matches_numerical_integration():
    rng = np.random.default_rng(14)
    a, ter = 1.2, 0.3
    times = np.array([0.0, 1.3, 3.1])
    v_true = np.array([1.4, 1.0, 1.8])
    rows = []
    for j, (tj, vj) in enumerate(zip(times, v_true)):
        dt, up = simulate_wiener(np.full(6, vj), a, 1.0, rng)
        for x, u in zip(dt, up):
            rows.append((0, j, tj, x + ter, u))
    trials = pd.DataFrame(rows, columns=["person", "occasion", "time", "rt", "correct"])
    mu0, tau, theta, sigma = 1.5, 0.3, 0.5, 0.35

    exact = collapsed_loglik(trials, a, ter, mu0, tau, theta, sigma)

    # brute force: Gauss-Hermite over the 3 latent drifts, prior N(mu0, tau^2 + K)
    d = np.abs(times[:, None] - times[None, :])
    C = tau ** 2 + sigma ** 2 / (2 * theta) * np.exp(-theta * d)
    L = np.linalg.cholesky(C)
    x, w = np.polynomial.hermite_e.hermegauss(40)  # probabilists' Hermite: weight exp(-x^2/2)
    Z = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(w, w, w, indexing="ij")).reshape(3, -1), axis=0) / (2 * np.pi) ** 1.5
    V = mu0 + L @ Z
    ll = np.zeros(V.shape[1])
    for j in range(3):
        g = trials[trials.occasion == j]
        T = g["rt"].to_numpy() - ter
        U = g["correct"].to_numpy()
        for tr, ur in zip(T, U):
            ll += wiener_logpdf(np.full(V.shape[1], tr), np.full(V.shape[1], ur), V[j], a)
    mx = ll.max()
    brute = mx + np.log(np.sum(W * np.exp(ll - mx)))
    assert abs(exact - brute) < 1e-8


def test_information_ratio():
    a = 1.2
    for v in (0.5, 1.5, 2.5, 4.0):
        p = 1 / (1 + np.exp(-a * v))
        i_choice = a ** 2 * p * (1 - p)
        i_full = (a / (2 * v)) * np.tanh(a * v / 2)   # E[decision time]
        assert abs(information_ratio_choice(v, a) - i_choice / i_full) < 1e-12
    assert information_ratio_choice(0.0, a) == 1.0


def test_fit_collapsed_recovers_dynamics_with_short_sessions():
    pars = OUDDMParams()
    sim = simulate_study(pars, DesignParams(n_persons=300, n_occasions=10, n_trials=20),
                         np.random.default_rng(15))
    fit, summ = fit_collapsed(sim["trials"], pars.a, pars.ter)
    assert abs(fit.theta - pars.theta) < 0.15
    assert abs(fit.sigma - pars.sigma) < 0.06
    assert abs(fit.mu0 - pars.mu0) < 0.05


def test_density_near_zero_decision_time_no_floor():
    # leading small-time term: log w - w^2/(2u) - 0.5 log(2 pi u^3) - 2 log a, u = t / a^2
    a, w = 1.2, 0.5
    for t in (1e-5, 1e-4, 1e-3):
        u = t / a ** 2
        lead = np.log(w) - w ** 2 / (2 * u) - 0.5 * np.log(2 * np.pi * u ** 3) - 2 * np.log(a)
        got = wiener_logpdf(np.array([t]), np.array([1]), 0.0, a)[0]
        assert np.isfinite(got)
        assert abs(got - lead) < 1e-6 * abs(lead) + 1e-9
    # strictly decreasing toward t -> 0 (no plateau from a numerical floor)
    ts = np.geomspace(1e-6, 1e-2, 40)
    lp = wiener_logpdf(ts, np.ones(40, int), 0.0, a)
    assert np.all(np.diff(lp) > 0)


def test_density_series_switch_is_continuous():
    a = 1.2
    for w in (0.3, 0.5, 0.7):
        lo = wiener_logpdf(np.array([a * a * (1 - 1e-9)]), np.array([1]), 0.7, a, w)[0]
        hi = wiener_logpdf(np.array([a * a * (1 + 1e-9)]), np.array([1]), 0.7, a, w)[0]
        assert abs(lo - hi) < 1e-6


def test_session_statistic_bias_identity():
    # E[m | v] = v E[S] E[1/S] for the centered model (sign independent of decision time)
    a, v, n, R = 1.2, 1.5, 20, 40000
    T, up = simulate_wiener(np.full(R * n, v), a, 1.0, np.random.default_rng(31))
    T = T.reshape(R, n); up = up.reshape(R, n)
    S = T.sum(1); m = 0.5 * a * np.where(up == 1, 1.0, -1.0).sum(1) / S
    pred = v * S.mean() * (1 / S).mean()
    assert abs(m.mean() - pred) < 4 * m.std() / np.sqrt(R)
    assert m.mean() > v
