import numpy as np
import pandas as pd

from ouddm import simulate_wiener, wiener_logpdf
from ouddm.design_model import DesignData, design_loglik, fit_design
from ouddm.wiener import _log_g


def test_log_g_accepts_vector_w():
    u = np.array([0.05, 0.5, 2.0])
    w = np.array([0.3, 0.5, 0.7])
    vec = _log_g(u, w)
    sca = np.array([_log_g(np.array([uu]), ww)[0] for uu, ww in zip(u, w)])
    assert np.allclose(vec, sca, atol=1e-12)


def _toy(rng):
    # one person, three sessions; stimulus coding, two boundary groups, two bias groups
    a0, a1, g0, g1, kappa = np.log(1.1), 0.3, 0.2, -0.5, 0.7
    beta = np.array([0.5])
    times = np.array([0.0, 1.2, 2.9]); u = np.array([1.3, 0.9, 1.7])
    rows = []
    for j in range(3):
        zb, qb = j % 2, (j + 1) % 2
        a = np.exp(a0 + a1 * zb); w = 1 / (1 + np.exp(-(g0 + g1 * qb)))
        for r in range(7):
            s = rng.choice([-1.0, 1.0]); x = rng.choice([-0.5, 0.5])
            v = s * (u[j] + beta[0] * x)
            # simulate with start w: decision process from w*a; use symmetric sampler by reflection
            # (sampler is centered); draw via rejection-free approach: use centered sampler only for check data
            dt, up = simulate_wiener(np.array([v]), a, 1.0, rng)
            rows.append((0, j, times[j], dt[0], up[0], s, zb, qb, x))
    tr = pd.DataFrame(rows, columns=["person", "occasion", "time", "dt", "upper", "stim", "zb", "qb", "x"])
    tr["rt"] = tr.dt + 0.4   # placeholder; ter_i = kappa * min rt
    return tr, (a0, a1, g0, g1, kappa, beta, times)


def test_design_loglik_matches_numerical_integration():
    rng = np.random.default_rng(77)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"])
    tau, theta, sigma = 0.3, 0.5, 0.35
    xv = np.array([a0, a1, g0, g1, np.log(kappa / (1 - kappa)), np.log(tau), np.log(theta), np.log(sigma / np.sqrt(2 * theta)), beta[0]])
    exact, mu0 = design_loglik(xv, d, return_mu0=True)
    # brute force over the three latent states
    D = np.abs(times[:, None] - times[None, :])
    C = tau ** 2 + sigma ** 2 / (2 * theta) * np.exp(-theta * D)
    L = np.linalg.cholesky(C)
    z, wq = np.polynomial.hermite_e.hermegauss(40)
    Zg = np.array(np.meshgrid(z, z, z, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(wq, wq, wq, indexing="ij")).reshape(3, -1), 0) / (2 * np.pi) ** 1.5
    U = mu0 + L @ Zg
    minrt = tr.rt.min()
    ll = np.zeros(U.shape[1])
    for _, r in tr.iterrows():
        a = np.exp(a0 + a1 * r.zb); w = 1 / (1 + np.exp(-(g0 + g1 * r.qb)))
        v = r.stim * (U[int(r.occasion)] + beta[0] * r.x)
        ll += wiener_logpdf(np.full(U.shape[1], r.rt - kappa * minrt), np.full(U.shape[1], int(r.upper)), v, a, w)
    m = ll.max()
    brute = m + np.log(np.sum(W * np.exp(ll - m)))
    assert abs(exact - brute) < 1e-8


def test_design_loglik_with_nugget_matches_numerical_integration():
    rng = np.random.default_rng(78)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], nugget=True)
    tau, theta, sd, om = 0.3, 0.5, 0.4, 0.25
    xv = np.array([a0, a1, g0, g1, np.log(kappa / (1 - kappa)), np.log(tau), np.log(theta), np.log(sd), np.log(om), beta[0]])
    exact, mu0 = design_loglik(xv, d, return_mu0=True)
    D = np.abs(times[:, None] - times[None, :])
    C = tau ** 2 + sd ** 2 * np.exp(-theta * D) + om ** 2 * np.eye(3)
    L = np.linalg.cholesky(C)
    z, wq = np.polynomial.hermite_e.hermegauss(40)
    Zg = np.array(np.meshgrid(z, z, z, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(wq, wq, wq, indexing="ij")).reshape(3, -1), 0) / (2 * np.pi) ** 1.5
    U = mu0 + L @ Zg
    minrt = tr.rt.min()
    ll = np.zeros(U.shape[1])
    for _, r in tr.iterrows():
        a = np.exp(a0 + a1 * r.zb); w = 1 / (1 + np.exp(-(g0 + g1 * r.qb)))
        v = r.stim * (U[int(r.occasion)] + beta[0] * r.x)
        ll += wiener_logpdf(np.full(U.shape[1], r.rt - kappa * minrt), np.full(U.shape[1], int(r.upper)), v, a, w)
    m = ll.max()
    brute = m + np.log(np.sum(W * np.exp(ll - m)))
    assert abs(exact - brute) < 1e-8


def test_design_loglik_with_random_slope_matches_numerical_integration():
    rng = np.random.default_rng(79)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    tr["day"] = tr.occasion.astype(float) - 1.0
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], nugget=True, slope="day")
    tau, theta, sd, om, ss, rho = 0.3, 0.5, 0.4, 0.25, 0.2, 0.4
    xv = np.array([a0, a1, g0, g1, np.log(kappa / (1 - kappa)), np.log(tau), np.log(theta), np.log(sd), np.log(om),
                   np.log(ss), np.arctanh(rho), beta[0]])
    exact, mu0 = design_loglik(xv, d, return_mu0=True)
    z = np.array([-1.0, 0.0, 1.0])
    D = np.abs(times[:, None] - times[None, :])
    C = (tau ** 2 + rho * tau * ss * (z[:, None] + z[None, :]) + ss ** 2 * np.outer(z, z)
         + sd ** 2 * np.exp(-theta * D) + om ** 2 * np.eye(3))
    L = np.linalg.cholesky(C)
    zz, wq = np.polynomial.hermite_e.hermegauss(40)
    Zg = np.array(np.meshgrid(zz, zz, zz, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(wq, wq, wq, indexing="ij")).reshape(3, -1), 0) / (2 * np.pi) ** 1.5
    U = mu0 + L @ Zg
    minrt = tr.rt.min()
    ll = np.zeros(U.shape[1])
    for _, r in tr.iterrows():
        a = np.exp(a0 + a1 * r.zb); w = 1 / (1 + np.exp(-(g0 + g1 * r.qb)))
        v = r.stim * (U[int(r.occasion)] + beta[0] * r.x)
        ll += wiener_logpdf(np.full(U.shape[1], r.rt - kappa * minrt), np.full(U.shape[1], int(r.upper)), v, a, w)
    m = ll.max()
    assert abs(exact - (m + np.log(np.sum(W * np.exp(ll - m))))) < 1e-8


def test_ou_limits_equal_random_intercept_plus_nugget():
    # with both tau and omega free, theta -> 0 and theta -> infinity both reduce to the
    # random-intercept-plus-session-deviation model (variance reallocated)
    rng = np.random.default_rng(80)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    d_ou = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], nugget=True)
    d_ri = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], nugget=True, ou=False)
    base = [a0, a1, g0, g1, np.log(kappa / (1 - kappa))]
    tau, sd, om = 0.3, 0.4, 0.25
    ri_slow = design_loglik(np.array(base + [np.log(np.hypot(tau, sd)), np.log(om), beta[0]]), d_ri)
    ri_fast = design_loglik(np.array(base + [np.log(tau), np.log(np.hypot(om, sd)), beta[0]]), d_ri)
    slow = design_loglik(np.array(base + [np.log(tau), np.log(1e-9), np.log(sd), np.log(om), beta[0]]), d_ou)
    fast = design_loglik(np.array(base + [np.log(tau), np.log(1e9), np.log(sd), np.log(om), beta[0]]), d_ou)
    assert abs(slow - ri_slow) < 1e-6 and abs(fast - ri_fast) < 1e-6


def test_profile_interval_flags_disconnected_set():
    import numpy as np
    import pandas as pd
    from ouddm.design_model import profile_interval
    lt = np.linspace(-4, 4, 9)
    dev = np.array([20, 1, 0, 2, 16, 3, 0.5, 2, 20], float)  # two supported runs
    prof = pd.DataFrame({"log_theta": lt, "loglik": -dev / 2})
    r = profile_interval(prof, ref_loglik=0.0)
    assert r["disconnected"] and len(r["regions"]) == 2
    (lo1, hi1), (lo2, hi2) = r["regions"]
    assert lo1 < hi1 < lo2 < hi2
    hl_mid = np.log(2) / np.exp(lt[4])  # deviance 16: must lie between the ranges
    assert hi1 < hl_mid < lo2
    one = profile_interval(prof.iloc[:4], ref_loglik=0.0)
    assert not one["disconnected"] and len(one["regions"]) == 1


def test_profile_interval_reports_empty_set():
    import math
    import numpy as np
    import pandas as pd
    from ouddm.design_model import profile_interval
    lt = np.linspace(-4, 4, 5)
    dev = np.array([30, 12, 9, 15, 40], float)  # nothing within 3.84 of the reference
    r = profile_interval(pd.DataFrame({"log_theta": lt, "loglik": -dev / 2}), ref_loglik=0.0)
    assert r["empty"] and r["needs_refinement"] and r["regions"] == []
    assert math.isnan(r["halflife_lo"]) and math.isnan(r["halflife_hi"])
    assert np.isclose(r["best_grid_halflife"], np.log(2) / np.exp(lt[2]))
