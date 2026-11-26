"""Person-specific practice-slope model without an OU component (the no-OU
limit of S1), reported as its own row in Table 6.

Usage: python -I color_shapes_s1_noou.py <data_merged.csv> [outdir]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_shapes_application as app  # noqa: E402
from color_shapes_revision import BASE  # noqa: E402
from ouddm.design_model import DesignData, fit_design  # noqa: E402

d = app.load(Path(sys.argv[1]))
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
data = DesignData.from_trials(d, "stim", **BASE, slope="day_c", ou=False)
fit = fit_design(data, se=True)
row = {"analysis": "S1 practice slopes, no OU", "loglik": fit.loglik, "aic": fit.aic,
       "n_params": fit.n_params, "converged": fit.converged}
row.update({k: v for k, v in fit.params.items() if k in ("tau", "omega", "slope_sd", "rho")})
pd.DataFrame([row]).to_csv(outdir / "colorshapes_s1_noou.csv", index=False)
print(row)
