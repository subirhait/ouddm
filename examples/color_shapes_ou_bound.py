"""Upper 95% profile bound on the SD of an OU component with a given half-life,
under the person-specific practice-slope model (S1). The half-life is fixed;
the OU stationary SD is profiled; all other parameters are re-maximized.

Usage: python -I color_shapes_ou_bound.py <data_merged.csv> [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_shapes_application as app  # noqa: E402
from color_shapes_revision import BASE  # noqa: E402
from ouddm.design_model import DesignData, _bounds, _names, _starts, design_loglik, fit_design  # noqa: E402

d = app.load(Path(sys.argv[1]))
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
data = DesignData.from_trials(d, "stim", **BASE, slope="day_c")
names = _names(data)
ilt, isd = names.index("log_theta"), names.index("log_sd_stat")
fit = fit_design(data, se=False)
ref = fit.loglik


def prof(log_theta, log_sd):
    b = _bounds(data, fix_theta=log_theta)
    b[isd] = (log_sd, log_sd)
    best = None
    for z in [fit.x] + _starts(data, thetas=(np.exp(log_theta),)):
        z = z.copy(); z[ilt] = log_theta; z[isd] = log_sd
        r = minimize(lambda x: -design_loglik(x, data), z, method="L-BFGS-B", bounds=b)
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun


rows = []
for hl in (1.0, 2.0):
    lt = np.log(np.log(2) / hl)
    g = lambda lsd: 2 * (ref - prof(lt, lsd)) - 3.841
    lo, hi = np.log(0.01), np.log(0.6)
    sd_up = float(np.exp(brentq(g, lo, hi, xtol=0.005)))
    rows.append({"halflife_days": hl, "ou_sd_upper95": sd_up, "ref_loglik": ref})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(outdir / "colorshapes_ou_sd_bounds.csv", index=False)
