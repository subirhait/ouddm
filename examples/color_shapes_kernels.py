"""Color Shapes: comparison of session dynamics (none, OU, random walk,
squared-exponential) under common and person-specific practice slopes, and
the best model with across-trial drift variability.

Usage: python -I color_shapes_kernels.py <data_merged.csv> [outdir]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_shapes_application as app  # noqa: E402
from color_shapes_revision import BASE  # noqa: E402
from ouddm.design_model import DesignData, fit_design  # noqa: E402

d = app.load(Path(sys.argv[1]))
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
rows = []


def run(prac, kern, sv=False):
    t0 = time.time()
    spec = dict(BASE, kernel=kern, sv=sv, **({"slope": "day_c"} if prac == "person" else {}))
    f = fit_design(DesignData.from_trials(d, "stim", **spec), se=False)
    p = f.params
    rows.append({"model": f"{prac}|{kern}|{'sv' if sv else 'nosv'}", "practice": prac, "kernel": kern, "sv": sv,
                 "loglik": f.loglik, "aic": f.aic, "n_params": f.n_params,
                 "halflife_days": f.halflife if kern == "ou" else np.nan,
                 "tau": p["tau"], "sd_stat": p["sd_stat"], "ell": p["ell"], "sd_rw": p["sd_rw"],
                 "omega": p["omega"], "slope_sd": p["slope_sd"], "sv_hat": p["sv"],
                 "converged": f.converged, "seconds": time.time() - t0})
    print({k: rows[-1][k] for k in ("model", "loglik", "aic", "halflife_days", "sv_hat", "seconds")}, flush=True)


for prac in ("common", "person"):
    for kern in ("none", "ou", "rw", "se"):
        run(prac, kern)
res = pd.DataFrame(rows)
best = res.sort_values("aic").iloc[0]
run(best.practice, best.kernel, sv=True)
res = pd.DataFrame(rows)
res["daic"] = res.aic - res.aic.min()
res.to_csv(outdir / "colorshapes_kernels.csv", index=False)
print(res[["model", "loglik", "aic", "daic", "halflife_days", "sv_hat"]].round(3).to_string(index=False))
