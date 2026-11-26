"""EZ-diffusion (Wagenmakers, van der Maas, & Grasman, 2007).

Closed-form estimates of drift rate v, boundary a, and non-decision
time ter from proportion correct, the variance of correct RTs, and the
mean of correct RTs. Used here as the stage-1 estimator of a common
two-stage workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ez_diffusion(pc: float, vrt: float, mrt: float, n: int, s: float = 1.0) -> tuple[float, float, float]:
    """EZ estimates (v, a, ter).

    Edge correction: pc of 1 is set to 1 - 1/(2n); pc of exactly 0.5 is
    nudged by 1/(2n) (EZ is undefined at 0.5). pc below 0.5 gives a
    negative drift.
    """
    if pc >= 1.0:
        pc = 1.0 - 1.0 / (2.0 * n)
    elif pc <= 0.0:
        pc = 1.0 / (2.0 * n)
    if pc == 0.5:
        pc = 0.5 + 1.0 / (2.0 * n)
    if not np.isfinite(vrt) or vrt <= 0:
        return np.nan, np.nan, np.nan
    L = np.log(pc / (1.0 - pc))
    x = L * (L * pc ** 2 - L * pc + pc - 0.5) / vrt
    v = np.sign(pc - 0.5) * s * x ** 0.25
    a = s ** 2 * L / v
    y = -v * a / s ** 2
    mdt = (a / (2.0 * v)) * (1.0 - np.exp(y)) / (1.0 + np.exp(y))
    ter = mrt - mdt
    return float(v), float(a), float(ter)


def ez_by_occasion(trials: pd.DataFrame, s: float = 1.0) -> pd.DataFrame:
    """Apply EZ-diffusion to each person-occasion.

    Expects columns person, occasion, time, rt, correct.
    Returns person, occasion, time, n, pc, v_hat, a_hat, ter_hat.
    """
    rows = []
    for (p, o), g in trials.groupby(["person", "occasion"], sort=True):
        n = len(g)
        pc = g["correct"].mean()
        rt_c = g.loc[g["correct"] == 1, "rt"].to_numpy()
        vrt = rt_c.var(ddof=1) if rt_c.size > 1 else np.nan
        mrt = rt_c.mean() if rt_c.size > 0 else np.nan
        v, a, ter = ez_diffusion(pc, vrt, mrt, n, s=s)
        rows.append((p, o, g["time"].iloc[0], n, pc, v, a, ter))
    return pd.DataFrame(rows, columns=["person", "occasion", "time", "n", "pc",
                                       "v_hat", "a_hat", "ter_hat"])
