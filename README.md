# ouddm

Exact likelihood and study design for drift diffusion models (DDMs) whose drift
rate follows an Ornstein–Uhlenbeck (OU) process across repeated sessions. It is
built for intensive longitudinal cognitive data such as smartphone-based
ambulatory assessments.

Author: Subir Hait, Michigan State University · License: MIT

## What it does

Person *i* completes short two-choice sessions at times *t_ij* (days, irregular
spacing allowed). Within a session, trials follow a Wiener diffusion with drift
`v_ij`, boundary `a`, start `w·a`, diffusion scale 1, and non-decision time `ter`.
Across sessions, the drift follows an OU process:

    dv = θ (μ_i − v) dt + σ dW,    μ_i ~ N(μ0, τ²),    half-life = ln 2 / θ

**Key result.** A session-constant drift enters the Wiener likelihood only through
a quadratic term, so as a function of the drift each session's likelihood is an
exact Gaussian kernel:

    l_j(v) − l_j(0) = b_j v − S_j v² / 2,
    b_j = a[(1−w) N_jU − w N_jL],   S_j = Σ decision times

The kernel is centered at `m_j = b_j / S_j` with curvature `S_j`. (This is a
statement about the likelihood; `m_j` itself is biased upward in short sessions:
`E[m|v] = v E[S] E[1/S]`.) All latent drifts and person means integrate out in
closed form, giving the exact likelihood of all choices and RTs. No MCMC and no
stage-one approximation are needed. The reduction stays exact when observed trial
or session predictors (task condition, practice) shift the drift.

## Install

```bash
pip install ouddm            # once released on PyPI
pip install -e ".[dev]"      # from a clone, with test dependencies
```

## Quick start

```python
import numpy as np
import ouddm

# data: one row per trial with columns person, occasion, time (days),
#       rt (seconds), correct (1 = upper boundary, 0 = lower boundary)
sim = ouddm.simulate_study(ouddm.OUDDMParams(),
                           ouddm.DesignParams(n_persons=50, n_occasions=10, n_trials=40),
                           np.random.default_rng(1))
trials = sim["trials"]

fit = ouddm.fit_exact(trials)          # joint ML: a, ter, mu0, tau, theta, sigma
print(fit.halflife, fit.halflife_ci())  # half-life (days) with 95% Wald interval
print(fit.a, fit.ter)

# task condition and practice as drift predictors (trial drift = OU state + x'beta)
# trials["cond"] = ...; trials["day_c"] = ...
# fit = ouddm.fit_exact(trials, covariates=["cond", "day_c"]); print(fit.beta, fit.se_beta)

# plan a study: plug-in SE of log half-life for a schedule
sched = ouddm.schedule(n_sessions=20, window=14.0, spacing="even")
print(ouddm.design_information(sched, n_trials=20, a=1.2, mu0=1.5, tau=0.3,
                               theta=0.5, sigma=0.35, n_persons=50))
print(ouddm.compare_designs(budget=400, sessions=[4, 10, 20, 40], window=14.0, halflife=1.4))
```

## Main functions

| Function | Purpose |
|---|---|
| `fit_exact` | Joint ML from the exact collapsed likelihood, with half-life interval and optional drift predictors |
| `fit_two_stage_ml` | Two-stage comparator with shared `a`, `ter`; optional variance propagation |
| `DesignData`, `fit_design` | Exact likelihood for real task designs: stimulus coding, condition-specific boundary and start point, person-specific non-decision time, drift predictors, session deviations |
| `profile_theta`, `profile_interval` | Profile-likelihood interval for the half-life (use when θ is near a bound) |
| `two_stage`, `ez_by_occasion` | EZ-diffusion two-stage estimators (naive, constant error) |
| `design_information`, `compare_designs`, `schedule` | Design calculator for the half-life |
| `information_ratio_choice` | Share of per-trial drift information kept by choices alone, x / sinh(x) |
| `wiener_logpdf`, `wiener_cdf` | Exact Wiener first-passage density and CDF |
| `simulate_study` | Simulator with exact OU transitions and a bridge-corrected Wiener sampler |

## Assumptions

- Diffusion scale fixed at 1. Drift constant within a session apart from observed predictors.
- `a`, `w`, and `ter` are shared across sessions and persons. In the current
  version, `w = 1/2` is fixed during estimation.
- No across-trial variability in non-decision time or starting point, and no
  contaminant responses.

## Reproducing the paper

```bash
pytest -q
python examples/main_study.py 100 results         # Simulation study 1 (~45 min, 2 cores)
python examples/design_study.py 40 results        # Simulation study 2 (~60 min, 2 cores)
python examples/condition_study.py 40 results     # Simulation study 3 (~25 min, 2 cores)
# Application: download data_merged.csv and ColorShapesTrialData_n69.csv from https://osf.io/ty4uf/
python -I examples/color_shapes_application.py <path>/data_merged.csv results            # ~12 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv A results            # M3n + S1, ~56 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv B results            # S2 + S3, ~39 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv C results            # M2n, ~20 min
python -I examples/color_shapes_ou_bound.py <path>/data_merged.csv results              # OU SD bounds under S1
python -I examples/color_shapes_s1_noou.py <path>/data_merged.csv results               # S1 without OU, Table 6
python -I examples/color_shapes_counts.py <path>/data_merged.csv results
python -I examples/color_shapes_descriptives.py <path>/ColorShapesTrialData_n69.csv results
python examples/stage1_error_diagnostic.py 20 results
python examples/paper_figures.py results figures
```

The manuscript build files (`paper/`) are not part of the package. They are in
the replication archive that accompanies the article, with instructions in
`REPLICATION.md`.

## Citation

Hait, S. (2026). *Estimating the half-life of cognitive fluctuations from short
repeated tasks: An exact likelihood for diffusion models with Ornstein–Uhlenbeck
drift.* Manuscript in preparation.
