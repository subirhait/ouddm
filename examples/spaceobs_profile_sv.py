"""Profile-likelihood interval for the OU half-life in the Space Observer data
under the preferred model: person-specific practice slopes, session deviations,
and across-trial drift variability.

Usage: python -I spaceobs_profile_sv.py <space-observer-dta.csv> <task_data.csv> <outdir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from space_observer_application import load  # noqa: E402
from ouddm.design_model import DesignData, fit_design, profile_interval, profile_theta  # noqa: E402

d, _ = load(Path(sys.argv[1]), Path(sys.argv[2])); outdir = Path(sys.argv[3])
base = dict(drift=("log_int_c", "game_c"), nugget=True, slope="game_c", sv=True)
dd = DesignData.from_trials(d, "stim", **base, kernel="ou")
f = fit_design(dd, se=False)
grid = np.log(np.geomspace(1e-4, 1e4, 17))
prof = profile_theta(dd, np.sort(np.concatenate([grid, [np.log(f.params["theta"])]])), x_start=f.x)
prof.to_csv(outdir / "spaceobs_profile_person_sv.csv", index=False)
ref = max(f.loglik, prof.loglik.max())
iv = profile_interval(prof, ref_loglik=ref, d=dd, fit=f)
none = fit_design(DesignData.from_trials(d, "stim", **base, kernel="none"), se=False)
iv.update(model="person|ou|sv", fit_loglik=f.loglik, fit_halflife=f.halflife,
          lr_vs_none=2 * (ref - none.loglik), regions=str(iv["regions"]))
pd.DataFrame([iv]).to_csv(outdir / "spaceobs_profile_interval_sv.csv", index=False)
print({k: iv[k] for k in ("halflife_lo", "halflife_hi", "long_end_open", "short_end_open",
                          "disconnected", "lr_vs_none", "ref_loglik", "fit_halflife")})
