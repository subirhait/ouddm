# Changelog

## 0.6.1

- `schedule_example.py` now applies the same rule as Table 12 of the
  manuscript: a closed profile interval that spans more than a factor of 10 is
  reported as unresolved.
- `schedule_study.py nullcheck`: split-half check of the simulation-calibrated
  critical values (threshold from one set of no-OU data sets, applied to an
  independent set). `summary` can pool no-OU data sets from several folders.
- No change to the package code.

## 0.6.0

- Prediction for future sessions: `DesignData.from_trials(..., outcomes=False)`
  builds test data without `rt` or `upper` columns; such data can be passed to
  `predict_choices` as `d_test` but cannot be fitted. `DesignData.trial_index`
  keeps each trial's row index in the input DataFrame, so predictions can be
  matched to input rows.
- `predict_choices` / `session_drift_predictive` reject a random-walk
  prediction for a session before the person's first training session (the
  process is not defined there) and raise an error instead of clipping a
  materially negative predictive variance.
- Faster likelihood: persons with the same number of sessions are processed as
  one batch (about three times faster on the Space Observer data; values
  unchanged up to rounding).
- New examples: `schedule_study.py` (Simulation study 4: session schedules for
  a fixed trial budget, calibrated to Space Observer), `schedule_example.py`
  (one simulated study per schedule, with a guide to reading an unresolved
  half-life), and `spaceobs_rolling.py` (rolling-origin holdout by gap).
- `results/schedule_calibration.csv`: generating values for the schedule
  study (fitted parameters only; no data).

## 0.5.1

- `predict_choices(x, d_train, d_test)`: held-out choice probabilities for new
  sessions from a `fit_design` fit. Each new session's drift is the Gaussian
  conditional given the person's training-session kernels
  (`session_drift_predictive`); the trial drift, with across-trial variability,
  is integrated by Gauss-Hermite quadrature. `p_upper` gives the probability of
  an upper-boundary response for a given drift.
- Internal: the covariance of session drifts and the session kernels are now
  shared helpers (`_kcross`, `_session_kernels`); likelihood values are unchanged.
- New tests: `p_upper` against the integrated first-passage density; the
  predictive mean and variance against direct joint Gaussian conditioning; a
  simulation in which OU dependence improves held-out prediction.
- New example: `spaceobs_robustness.py` (sample-rule checks and a final-session
  holdout for the Space Observer data).

## 0.5.0

- Across-trial drift variability: `DesignData.from_trials(..., sv=True)`. The
  trial drift varies around the session value with SD `sv`; it integrates out
  exactly (Proposition 6), so the likelihood stays exact. Simulator: `OUDDMParams(sv=...)`.
- Session dynamics: `kernel="ou"` (default), `"rw"` (random walk from the first
  session), `"se"` (squared-exponential Gaussian process), or `"none"`
  (Corollary 1). `ou=False` still works and means `kernel="none"`.
- `wald_logpdf` and `first_passage_kernel`: the single-boundary density and the
  Gaussian drift kernel of any first-passage model (Theorem 1).
- New tests: drift variability, random-walk and squared-exponential kernels
  against nested Gauss-Hermite quadrature; the first-passage identity for the
  two-boundary and Wald models.
- New examples: `sv_study.py`, `color_shapes_kernels.py`, `dynamics_with_sv.py`,
  `space_observer_application.py`, `spaceobs_profile_sv.py`.
- README: the install line no longer says "once released on PyPI".

## 0.4.4

- No change to the package code (identical to 0.4.2 and 0.4.3).
- The manuscript build files (`paper/`) are no longer shipped in the source
  distribution. They are in the replication archive that accompanies the
  article. A `.gitignore` keeps them out of the public repository.

## 0.4.3

- No change to the package code (identical to 0.4.2).
- Manuscript: abstract shortened to under 250 words for the Journal of
  Mathematical Psychology; the build now checks the abstract length and
  writes a separate highlights file (3–5 bullets, at most 85 characters each).
- Software and data availability section names the PyPI release.

## 0.4.2

- `profile_interval` no longer returns the best grid point as a one-point
  range when no grid point lies within the cut-off. It reports an empty set
  (`regions == []`, `empty=True`, `needs_refinement=True`, NaN ends) and gives
  `best_grid_halflife` as the place to refine. New test.
- Manuscript: Section 9.1 no longer says that joint OU models for drift and
  boundary keep the Gaussian reduction. The drift states integrate out
  exactly given the boundary trajectory; the boundary needs a separate
  integration step.

## 0.4.1

- `profile_interval` now detects a disconnected supported set. It returns
  each connected range in `regions` and sets `disconnected=True`; the outer
  ends `halflife_lo`/`halflife_hi` are kept but should not be read as one
  interval in that case. New helper `supported_regions`; new test.
- New example `color_shapes_s1_noou.py`: the practice-slope model without an
  OU part, reported in Table 6.
- The manuscript build checks that every archived profile has one connected
  supported region.

## 0.4.0

- New: person-specific random slope on a session predictor in `design_model`
  (`DesignData.from_trials(..., slope="day_c")`), correlated with the person
  mean. The likelihood stays exact; checked against quadrature.
- New: `ou=False` fits the no-OU limit model (random person intercept plus
  session deviations). With tau and omega free, both ends of the theta range
  reduce to this model; a test checks this.
- `profile_interval` takes the fitted maximum as the reference and finds the
  interval ends by root-finding; it reports whether either end is open.
- New examples: `color_shapes_revision.py` (refined profile from theta = 1e-5
  to 1e5 per day, practice-slope, long-window, and interaction sensitivity
  analyses), `color_shapes_ou_bound.py`, `color_shapes_counts.py`.
  `color_shapes_profile.py` is removed; its results are superseded.

## 0.3.0

- New: `design_model` for real task designs, with the exact likelihood kept:
  stimulus coding (drift sign by stimulus), start point depending on bias
  predictors, boundary depending on boundary predictors, person-specific
  non-decision time (kappa x fastest RT), drift predictors, and independent
  session-level drift deviations (nugget). `fit_design`, `profile_theta`,
  `profile_interval`.
- The OU part of `design_model` is parameterized by its stationary SD, so the
  white-noise limit (theta -> infinity) is reachable; starts span theta from
  0.3 to 2000 per day, because real data can have separate slow and fast optima.
- `_log_g` accepts a start point per trial.
- New tests: extended likelihood (with and without session deviations) against
  3-D quadrature; vector start points.
- New example: reanalysis of the Color Shapes data (Kim et al., 2025;
  https://osf.io/ty4uf/), with a profile-likelihood script.

## 0.2.0

- Fix: the zero-drift Wiener density is now summed on the log scale (signed
  log-sum-exp). Before, very short decision times hit a numerical floor
  (log density about -678 instead of -1788 at 0.1 ms with a = 1.2), which
  could make a large non-decision time look too plausible.
- Fix: `fit_exact` and `fit_ou` add starting values for the person-level SD
  tau. With a common practice trend, a single start could end in a second
  optimum with tau near 0 and a slow OU process standing in for it.
- New: drift predictors. `fit_exact(trials, covariates=[...])` fits trial
  drift = OU state + x'beta with the exact likelihood (Proposition 4).
- New: `simulate_study(..., x_session=, beta=)` for session-level predictors.
- New: fits report which optimizer bound (if any) theta reached; study
  scripts count both bounds, optimizer status, and interval denominators.
- Faster density evaluation (trimmed series, cached zero-drift term).
- New tests: density near zero decision time, series continuity, the bias
  identity E[m|v] = v E[S] E[1/S], covariate likelihood against quadrature,
  and recovery with a practice trend.

## 0.1.0

- Exact collapsed marginal likelihood for the DDM with OU drift dynamics
  (`fit_exact`): all session drifts and person home bases integrated out in
  closed form; boundary separation and non-decision time estimated jointly;
  Wald intervals for the drift half-life.
- Shared-nuisance two-stage comparator (`fit_two_stage_ml`), with or without
  propagation of session drift variance.
- EZ-diffusion two-stage estimators (naive and constant measurement error).
- Design calculator (`design_information`, `compare_designs`): expected SE
  of the log half-life for a session schedule, with or without RTs.
- Exact Wiener first-passage density and CDF; information ratio x / sinh(x).
- Simulator with exact OU transitions at irregular or fixed times and a
  bridge-corrected Wiener sampler.
