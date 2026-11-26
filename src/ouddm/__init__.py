"""ouddm: drift diffusion models with Ornstein-Uhlenbeck dynamics in drift rate.

Main entry points
-----------------
fit_exact(trials, covariates=())   joint ML from the exact collapsed likelihood,
                             optionally with drift predictors (task condition, practice)
fit_two_stage_ml(trials)     shared-nuisance two-stage comparator
fit_design(DesignData...)    exact likelihood for real task designs: stimulus coding,
                             condition-specific boundary and start point, person-specific
                             non-decision time, drift predictors, session deviations
profile_theta(...)           profile likelihood of the half-life
design_information(...)      plug-in SE of the drift half-life for a schedule
compare_designs(...)         fixed-budget comparison of session schedules
simulate_study(...)          simulate OU-drift DDM data

Input data: one row per trial with columns person, occasion, time (days),
rt (seconds), correct (1 = upper boundary, 0 = lower boundary).
"""

from .collapsed import collapsed_loglik, fit_collapsed, session_summaries
from .design import compare_designs, design_information, schedule, trial_information
from .design_model import DesignData, DesignFit, fit_design, profile_interval, profile_theta, supported_regions
from .estimate import JointFit, SessionData, fit_exact, fit_two_stage_ml
from .ez import ez_by_occasion, ez_diffusion
from .ou import OUFit, fit_ou
from .simulate import DesignParams, OUDDMParams, simulate_ou, simulate_study, simulate_wiener
from .twostage import oracle, two_stage
from .wiener import information_ratio_choice, wiener_cdf, wiener_logpdf

__version__ = "0.4.4"

__all__ = [
    "fit_exact", "fit_two_stage_ml", "JointFit", "SessionData",
    "DesignData", "DesignFit", "fit_design", "profile_theta", "profile_interval", "supported_regions",
    "design_information", "compare_designs", "schedule", "trial_information",
    "collapsed_loglik", "fit_collapsed", "session_summaries",
    "DesignParams", "OUDDMParams", "simulate_ou", "simulate_study", "simulate_wiener",
    "ez_by_occasion", "ez_diffusion", "two_stage", "oracle", "fit_ou", "OUFit",
    "information_ratio_choice", "wiener_cdf", "wiener_logpdf",
]
