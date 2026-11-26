"""Stage-1 error by true drift: EZ-diffusion vs the exact session statistic
m = b/S (known a, ter), for short sessions.

Shows whether stage-1 error is constant (the assumption of the 'corrected'
two-stage estimator) or depends on the true drift.
Usage: python stage1_error_diagnostic.py [n_trials] [outdir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ouddm import OUDDMParams, ez_diffusion, simulate_wiener

P = OUDDMParams()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
outdir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(77)
n_sess = 3000
rows = []
for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    t, up = simulate_wiener(np.full(n_sess * n, v), P.a, P.s, rng)
    t = t.reshape(n_sess, n) + P.ter
    up = up.reshape(n_sess, n)
    ez, ceiling = [], 0
    for k in range(n_sess):
        pc = up[k].mean()
        ceiling += pc == 1.0
        rc = t[k][up[k] == 1]
        ez.append(ez_diffusion(pc, rc.var(ddof=1) if rc.size > 1 else np.nan, rc.mean(), n)[0])
    ez = np.array(ez)
    S = (t - P.ter).sum(1)
    m = 0.5 * P.a * np.where(up == 1, 1.0, -1.0).sum(1) / S
    rows.append({"true_v": v, "n_trials": n,
                 "ez_bias": np.nanmean(ez) - v, "ez_sd": np.nanstd(ez), "ez_ceiling_rate": ceiling / n_sess,
                 "exact_bias": m.mean() - v, "exact_sd": m.std(),
                 "exact_sd_theory": 1 / np.sqrt(n * (P.a / (2 * v)) * np.tanh(P.a * v / 2))})
df = pd.DataFrame(rows)
df.to_csv(outdir / f"stage1_error_n{n}.csv", index=False)
with pd.option_context("display.precision", 3, "display.width", 200):
    print(df.to_string(index=False))
