"""Two-stage estimators of OU dynamics in DDM drift rate.

Stage 1: EZ-diffusion per person-occasion gives v_hat.
Stage 2: hierarchical OU model on v_hat.
  - 'naive'     : treats v_hat as the true drift rate (omega = 0).
  - 'corrected' : adds a measurement-error nugget omega.
"""

from __future__ import annotations

import pandas as pd

from .ez import ez_by_occasion
from .ou import OUFit, fit_ou


def two_stage(trials: pd.DataFrame, s: float = 1.0, measurement_error: bool = False,
              occasion_estimates: pd.DataFrame | None = None) -> tuple[OUFit, pd.DataFrame]:
    est = occasion_estimates if occasion_estimates is not None else ez_by_occasion(trials, s=s)
    fit = fit_ou(est["person"], est["time"], est["v_hat"], measurement_error=measurement_error)
    return fit, est


def oracle(occasions: pd.DataFrame) -> OUFit:
    """OU fit to the true drift rates (benchmark, not available in practice)."""
    return fit_ou(occasions["person"], occasions["time"], occasions["v_true"],
                  measurement_error=False)
