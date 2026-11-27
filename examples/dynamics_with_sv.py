"""Session dynamics compared with across-trial drift variability (sv) in the
model, under person-specific practice slopes, for both applications.

Usage:
  python -I dynamics_with_sv.py cs <data_merged.csv> <outdir>
  python -I dynamics_with_sv.py so <space-observer-dta.csv> <task_data.csv> <outdir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouddm.design_model import DesignData, fit_design  # noqa: E402

which = sys.argv[1]
if which == "cs":
    import color_shapes_application as app  # noqa: E402
    from color_shapes_revision import BASE  # noqa: E402
    d = app.load(Path(sys.argv[2])); outdir = Path(sys.argv[3])
    base = dict(BASE, slope="day_c")
else:
    from space_observer_application import load  # noqa: E402
    d, _ = load(Path(sys.argv[2]), Path(sys.argv[3])); outdir = Path(sys.argv[4])
    base = dict(drift=("log_int_c", "game_c"), nugget=True, slope="game_c")

rows = []
for kern in ("none", "ou", "rw", "se"):
    t0 = time.time()
    f = fit_design(DesignData.from_trials(d, "stim", **base, kernel=kern, sv=True), se=False)
    p = f.params
    rows.append({"dataset": which, "kernel": kern, "loglik": f.loglik, "aic": f.aic, "n_params": f.n_params,
                 "halflife_days": f.halflife if kern == "ou" else np.nan, "sv_hat": p["sv"],
                 "tau": p["tau"], "sd_stat": p["sd_stat"], "ell": p["ell"], "sd_rw": p["sd_rw"],
                 "omega": p["omega"], "slope_sd": p["slope_sd"], "x": ";".join(f"{v:.6g}" for v in f.x),
                 "converged": f.converged, "seconds": time.time() - t0})
    print({k: rows[-1][k] for k in ("kernel", "loglik", "aic", "halflife_days", "sv_hat", "omega", "seconds")}, flush=True)
res = pd.DataFrame(rows)
res["lr_vs_none"] = 2 * (res.loglik - res.loc[res.kernel == "none", "loglik"].iloc[0])
res["daic"] = res.aic - res.aic.min()
res.to_csv(outdir / f"dynamics_sv_{which}.csv", index=False)
print(res[["kernel", "loglik", "aic", "daic", "lr_vs_none", "halflife_days", "sv_hat"]].round(3).to_string(index=False))
