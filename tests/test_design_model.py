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


def _brute(tr, pars, C, mu0, sv=0.0, nq=25, nsv=20):
    """Brute-force log likelihood: Gauss-Hermite over the three session states
    and, if sv > 0, over each trial's drift deviation."""
    a0, a1, g0, g1, kappa, beta = pars
    L = np.linalg.cholesky(C)
    z, wq = np.polynomial.hermite_e.hermegauss(nq)
    Zg = np.array(np.meshgrid(z, z, z, indexing="ij")).reshape(3, -1)
    W = np.prod(np.array(np.meshgrid(wq, wq, wq, indexing="ij")).reshape(3, -1), 0) / (2 * np.pi) ** 1.5
    U = mu0 + L @ Zg
    zs, ws = np.polynomial.hermite_e.hermegauss(nsv) if sv > 0 else (np.zeros(1), np.full(1, np.sqrt(2 * np.pi)))
    ws = ws / np.sqrt(2 * np.pi)
    minrt = tr.rt.min()
    ll = np.zeros(U.shape[1])
    for _, r in tr.iterrows():
        a = np.exp(a0 + a1 * r.zb); w = 1 / (1 + np.exp(-(g0 + g1 * r.qb)))
        eta = U[int(r.occasion)][:, None] + beta[0] * r.x + sv * zs[None, :]
        lf = wiener_logpdf(np.full(eta.shape, r.rt - kappa * minrt), np.full(eta.shape, int(r.upper)), r.stim * eta, a, w)
        mx = lf.max(1, keepdims=True)
        ll += (mx + np.log(np.exp(lf - mx) @ ws[:, None]))[:, 0]
    m = ll.max()
    return m + np.log(np.sum(W * np.exp(ll - m)))


def test_design_loglik_with_drift_variability_matches_numerical_integration():
    rng = np.random.default_rng(81)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], nugget=True, sv=True)
    tau, theta, sd, om, sv = 0.3, 0.5, 0.4, 0.25, 0.7
    xv = np.array([a0, a1, g0, g1, np.log(kappa / (1 - kappa)), np.log(tau), np.log(theta), np.log(sd),
                   np.log(om), np.log(sv), beta[0]])
    exact, mu0 = design_loglik(xv, d, return_mu0=True)
    D = np.abs(times[:, None] - times[None, :])
    C = tau ** 2 + sd ** 2 * np.exp(-theta * D) + om ** 2 * np.eye(3)
    assert abs(exact - _brute(tr, (a0, a1, g0, g1, kappa, beta), C, mu0, sv=sv)) < 1e-7


def test_design_loglik_random_walk_and_se_kernels_match_numerical_integration():
    rng = np.random.default_rng(82)
    tr, (a0, a1, g0, g1, kappa, beta, times) = _toy(rng)
    base = [a0, a1, g0, g1, np.log(kappa / (1 - kappa))]
    tau, sdrw, sd, ell = 0.3, 0.35, 0.4, 1.5
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], kernel="rw")
    exact, mu0 = design_loglik(np.array(base + [np.log(tau), np.log(sdrw), beta[0]]), d, return_mu0=True)
    t0 = times - times.min()
    C = tau ** 2 + sdrw ** 2 * np.minimum(t0[:, None], t0[None, :]) + 1e-10 * np.eye(3)
    assert abs(exact - _brute(tr, (a0, a1, g0, g1, kappa, beta), C, mu0)) < 1e-7
    d = DesignData.from_trials(tr, "stim", boundary=["zb"], bias=["qb"], drift=["x"], kernel="se")
    exact, mu0 = design_loglik(np.array(base + [np.log(tau), np.log(ell), np.log(sd), beta[0]]), d, return_mu0=True)
    D = times[:, None] - times[None, :]
    C = tau ** 2 + sd ** 2 * np.exp(-0.5 * (D / ell) ** 2)
    assert abs(exact - _brute(tr, (a0, a1, g0, g1, kappa, beta), C, mu0)) < 1e-7


def test_first_passage_reduction_for_ddm_and_wald():
    from ouddm import first_passage_kernel, wald_logpdf
    a, w = 1.3, 0.45
    for t, up in [(0.4, 1), (1.3, 0), (2.5, 1)]:
        hit = a * (1 - w) if up else -a * w
        b, S = first_passage_kernel([hit], [t])
        for v in (-1.2, 0.7, 2.0):
            lhs = wiener_logpdf(t, up, v, a, w) - wiener_logpdf(t, up, 0.0, a, w)
            assert abs(lhs - (b * v - S * v ** 2 / 2)) < 1e-9
            lw = wald_logpdf(t, v, a) - wald_logpdf(t, 0.0, a)
            assert abs(lw - (a * v - t * v ** 2 / 2)) < 1e-12


def test_p_upper_matches_integrated_density():
    from ouddm.design_model import p_upper
    tg = np.linspace(1e-4, 30.0, 300001)
    for v, a, w in ((0.0, 1.2, 0.5), (1.3, 1.0, 0.4), (-0.8, 1.5, 0.65), (3.0, 2.0, 0.3)):
        dens = np.exp(wiener_logpdf(tg, np.ones_like(tg, dtype=int), v, a, w))
        num = np.sum((dens[1:] + dens[:-1]) / 2 * np.diff(tg))
        assert abs(p_upper(v, a, w * a) - num) < 2e-4
    assert np.isclose(p_upper(0.7, 1.1, 0.3), 1 - p_upper(-0.7, 1.1, 1.1 - 0.3))


def test_session_drift_predictive_matches_joint_gaussian_conditioning():
    # one person, OU dynamics with a person slope: compare with direct conditioning
    from ouddm.design_model import _kcross, _session_kernels, _unpack, _logistic, session_drift_predictive
    rng = np.random.default_rng(5)
    tr, _ = _toy(rng)
    tr["g"] = tr.occasion - 1.0
    spec = dict(drift=("x",), nugget=True, slope="g", kernel="ou", sv=True)
    train, test = tr[tr.occasion < 2], tr[tr.occasion == 2]
    dtr = DesignData.from_trials(train, "stim", **spec)
    dte = DesignData.from_trials(test, "stim", **spec)
    from ouddm.design_model import _names
    x = np.zeros(len(_names(dtr)))
    nm = _names(dtr)
    for k, val in {"log_a:intercept": np.log(1.1), "logit_kappa": 1.0, "log_tau": np.log(0.4),
                   "log_theta": np.log(0.8), "log_sd_stat": np.log(0.5), "log_omega": np.log(0.2),
                   "log_sv": np.log(0.3), "log_slope_sd": np.log(0.1), "atanh_rho": 0.3, "beta:x": 0.5}.items():
        x[nm.index(k)] = val
    mean, var = session_drift_predictive(x, dtr, dte)
    p = _unpack(x, dtr)
    a = np.exp(dtr.Z @ p["alpha"]); w = _logistic(dtr.Q @ p["gamma"])
    b, S, _ = _session_kernels(p, dtr, a, w, dtr.rt - p["kappa"] * dtr.minrt_trial)
    _, mu0 = design_loglik(x, dtr, return_mu0=True)
    t_all = np.r_[dtr.time, dte.time]; z_all = np.r_[dtr.slope_z, dte.slope_z]
    K = _kcross(t_all, t_all, z_all, z_all, t_all.min(), dtr, p) + np.eye(3) * p["omega"] ** 2
    Koo = K[:2, :2] + np.diag(1 / S)
    gain = np.linalg.solve(Koo, K[:2, 2])
    assert np.isclose(mean[0], mu0 + gain @ (b / S - mu0))
    assert np.isclose(var[0], K[2, 2] - K[2, :2] @ gain)


def test_predict_choices_uses_session_dependence():
    # strong slow OU dependence: held-out sessions are predicted better than by a model without it
    from ouddm import DesignParams, OUDDMParams, simulate_study
    from ouddm.design_model import predict_choices
    rng = np.random.default_rng(11)
    tr = simulate_study(OUDDMParams(mu0=0.8, tau=0.2, theta=0.2, sigma=0.5, a=1.2, ter=0.3),
                        DesignParams(n_persons=40, n_occasions=8, n_trials=30), rng)["trials"]
    tr["stim"] = 1.0; tr["upper"] = tr["correct"]
    last = tr.groupby("person").occasion.transform("max")
    out = {}
    for kern in ("ou", "none"):
        spec = dict(nugget=(kern == "none"), kernel=kern)
        dtr = DesignData.from_trials(tr[tr.occasion < last], "stim", **spec)
        dte = DesignData.from_trials(tr[tr.occasion == last], "stim", **spec)
        f = fit_design(dtr, se=False)
        pr = predict_choices(f.x, dtr, dte)
        assert np.all((pr > 0) & (pr < 1))
        out[kern] = -np.mean(dte.upper * np.log(pr) + (1 - dte.upper) * np.log1p(-pr))
    assert out["ou"] < out["none"]


def _rw_toy():
    rng = np.random.default_rng(3)
    tr, _ = _toy(rng)
    return tr


def test_prediction_only_input_matches_full_input():
    from ouddm.design_model import predict_choices, _names
    tr = _rw_toy()
    train, test = tr[tr.occasion < 2], tr[tr.occasion == 2]
    spec = dict(drift=("x",), nugget=True, kernel="ou", sv=True)
    dtr = DesignData.from_trials(train, "stim", **spec)
    full = DesignData.from_trials(test, "stim", **spec)
    future = DesignData.from_trials(test.drop(columns=["rt", "upper"]), "stim", **spec, outcomes=False)
    x = np.zeros(len(_names(dtr))); x[_names(dtr).index("logit_kappa")] = 1.0
    p1, p2 = predict_choices(x, dtr, full), predict_choices(x, dtr, future)
    assert np.allclose(p1, p2)
    assert np.array_equal(np.sort(future.trial_index), np.sort(test.index.to_numpy()))
    import pytest
    with pytest.raises(ValueError):
        fit_design(future)


def test_random_walk_rejects_test_session_before_training_start():
    import pytest
    from ouddm.design_model import predict_choices, _names
    tr = _rw_toy()
    train, test = tr[tr.occasion > 0], tr[tr.occasion == 0]   # test session is the earliest
    spec = dict(nugget=True, kernel="rw")
    dtr = DesignData.from_trials(train, "stim", **spec)
    dte = DesignData.from_trials(test, "stim", **spec, outcomes=False)
    x = np.zeros(len(_names(dtr))); x[_names(dtr).index("logit_kappa")] = 1.0
    with pytest.raises(ValueError, match="precedes"):
        predict_choices(x, dtr, dte)
