import numpy as np
import pandas as pd
import pytest

from ouddm import (DesignParams, OUDDMParams, collapsed_loglik, design_information, fit_exact,
                   fit_two_stage_ml, schedule, simulate_study, trial_information)
from ouddm.design import _person_information
from ouddm.estimate import SessionData, _eta, exact_loglik


@pytest.fixture(scope="module")
def sim():
    return simulate_study(OUDDMParams(), DesignParams(n_persons=60, n_occasions=10, n_trials=40),
                          np.random.default_rng(21))


def test_exact_loglik_matches_collapsed_loglik(sim):
    P = OUDDMParams()
    data = SessionData.from_trials(sim["trials"])
    x = np.array([np.log(P.a), _eta(P.ter, data.min_rt), np.log(P.tau), np.log(P.theta), np.log(P.sigma)])
    ll, mu0 = exact_loglik(x, data, return_mu0=True)
    ref = collapsed_loglik(sim["trials"], P.a, P.ter, mu0, P.tau, P.theta, P.sigma)
    assert abs(ll - ref) < 1e-6


def test_fit_exact_recovers_nuisance_and_dynamics(sim):
    P = OUDDMParams()
    f = fit_exact(sim["trials"])
    assert abs(f.a - P.a) < 0.03
    assert abs(f.ter - P.ter) < 0.01
    assert abs(np.log(f.theta) - np.log(P.theta)) < 3 * f.se_log_theta
    lo, hi = f.halflife_ci()
    assert lo < f.halflife < hi


def test_two_stage_ml_shares_nuisance(sim):
    f, nu = fit_two_stage_ml(sim["trials"], propagate=True)
    assert abs(nu["a"] - 1.2) < 0.03 and abs(nu["ter"] - 0.3) < 0.01
    f0, _ = fit_two_stage_ml(sim["trials"], propagate=False)
    assert f0.theta > f.theta          # ignoring session noise inflates theta


def test_input_validation():
    with pytest.raises(ValueError):
        SessionData.from_trials(pd.DataFrame({"person": [0], "rt": [0.5]}))


def test_design_information_matches_numerical_derivative():
    # Fisher information equals the variance of the score; check the log-theta
    # entry against a finite-difference second derivative of the expected loglik
    t = schedule(8, 10.0, "even")
    tau, theta, sigma, ov = 0.3, 0.5, 0.35, 0.05
    I = _person_information(t, ov, tau, theta, sigma)

    def cov(lt):
        th = np.exp(lt)
        D = np.abs(t[:, None] - t[None, :])
        return tau ** 2 + sigma ** 2 / (2 * th) * np.exp(-th * D) + ov * np.eye(t.size)

    S0 = cov(np.log(theta))

    def expected_ll(lt):  # E_{y~N(0,S0)} log N(y | 0, S(lt))
        S = cov(lt)
        return -0.5 * (np.linalg.slogdet(S)[1] + np.trace(np.linalg.solve(S, S0)))

    h = 1e-4
    lt = np.log(theta)
    d2 = (expected_ll(lt + h) - 2 * expected_ll(lt) + expected_ll(lt - h)) / h ** 2
    assert abs(-d2 - I[1, 1]) / I[1, 1] < 1e-4


def test_design_calculator_behaviour():
    r = design_information(schedule(10, 14, "even"), 40, 1.2, 1.5, 0.3, 0.5, 0.35, 50)
    assert 0 < r["relative_efficiency"] < 1
    assert r["se_oracle"] < r["se_log_halflife"]
    # RTs help more at high accuracy
    assert trial_information(2.5, 1.2, rt=False) / trial_information(2.5, 1.2) < \
        trial_information(1.0, 1.2, rt=False) / trial_information(1.0, 1.2)


def test_covariate_loglik_matches_numerical_integration():
    from ouddm import simulate_wiener, wiener_logpdf
    rng = np.random.default_rng(41)
    a, ter, beta = 1.2, 0.3, 0.6
    times = np.array([0.0, 1.3, 3.1]); u = np.array([1.4, 1.0, 1.8])
    rows = []
    for j in range(3):
        x = rng.choice([-0.5, 0.5], 6)            # covariate varies within session
        dt, up = simulate_wiener(u[j] + beta * x, a, 1.0, rng)
        rows += [(0, j, times[j], t + ter, c, xx) for t, c, xx in zip(dt, up, x)]
    tr = pd.DataFrame(rows, columns=["person", "occasion", "time", "rt", "correct", "x0"])
    data = SessionData.from_trials(tr, covariates=["x0"])
    tau, theta, sigma = 0.3, 0.5, 0.35
    xv = np.array([np.log(a), _eta(ter, data.min_rt), np.log(tau), np.log(theta), np.log(sigma), beta])
    exact, mu0 = exact_loglik(xv, data, return_mu0=True)
    D = np.abs(times[:, None] - times[None, :])
    C = tau ** 2 + sigma ** 2 / (2 * theta) * np.exp(-theta * D)
    L = np.linalg.cholesky(C)
    z, w = np.polynomial.hermite_e.hermegauss(40)
    Z = np.array(np.meshgrid(z, z, z, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(w, w, w, indexing="ij")).reshape(3, -1), 0) / (2 * np.pi) ** 1.5
    U = mu0 + L @ Z
    ll = np.zeros(U.shape[1])
    for _, r in tr.iterrows():
        ll += wiener_logpdf(np.full(U.shape[1], r.rt - ter), np.full(U.shape[1], int(r.correct)),
                            U[int(r.occasion)] + beta * r.x0, a)
    mx = ll.max()
    brute = mx + np.log(np.sum(W * np.exp(ll - mx)))
    assert abs(exact - brute) < 1e-8


def test_fit_exact_recovers_condition_effect():
    P = OUDDMParams()
    J = 12
    xs = np.tile([[-0.5], [0.5]], (J // 2, 1))
    sim = simulate_study(P, DesignParams(n_persons=60, n_occasions=J, n_trials=40),
                         np.random.default_rng(42), x_session=xs, beta=[0.8])
    f = fit_exact(sim["trials"], covariates=["x0"])
    assert abs(f.beta["x0"] - 0.8) < 4 * f.se_beta["x0"]
    assert f.se_beta["x0"] < 0.1


def test_fit_exact_escapes_low_tau_basin_with_practice_trend():
    # common practice trend: without tau multistarts the fit stuck at tau ~ 0, theta too small
    P = OUDDMParams()
    J = 16
    t = np.repeat(np.arange(8.0), 2) + np.tile([0.0, 1 / 24], 8)
    prac = np.repeat(np.arange(8.0), 2); prac -= prac.mean()
    sim = simulate_study(P, DesignParams(n_persons=68, n_occasions=J, n_trials=60),
                         np.random.default_rng([8, 0, 515]), times=t, x_session=prac[:, None], beta=[0.08])
    f = fit_exact(sim["trials"], covariates=["x0"])
    assert f.tau > 0.15
    assert abs(np.log(f.theta) - np.log(P.theta)) < 3 * f.se_log_theta
