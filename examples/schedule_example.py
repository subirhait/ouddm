"""Compare session schedules for one simulated study and read the result.

Simulates one data set per schedule (same persons; 420 trials per person over
14 days), using the generating values in results/schedule_calibration.csv
(the Space Observer fit), and reports for each schedule the likelihood-ratio
statistic for OU dynamics, the half-life estimate and its approximate 95%
profile interval. Runs in a few minutes.

Usage: python -I schedule_example.py <schedule_calibration.csv> [half-life in days] [n_persons]

Reading the output
- An interval closed at both ends and narrower than a factor of 10 (upper bound
  / lower bound) gives a usable range for the half-life. A closed interval that
  spans more than a factor of 10 is reported as unresolved, the same rule as in
  Table 12 of the manuscript.
- An interval open at the short end means the data cannot rule out
  dependence that fades faster than the closest pair of sessions; open at the
  long end means they cannot rule out a near-permanent component. In either
  case report the closed bound only, and treat the point estimate as unresolved.
- The statistic has no chi-square reference distribution (the half-life is not
  identified when there is no OU component). Compare it with values simulated
  under no dynamics for the same schedule (schedule_study.py), not with 3.84.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedule_study import HL_GRID, SCHEDULES, SessionLik, fit_null, interval, simulate  # noqa: E402

WIDE = 10.0   # same rule as Table 12: a closed interval wider than this factor counts as unresolved

cal = pd.read_csv(sys.argv[1]).iloc[0].to_dict()
hl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
n_persons = int(sys.argv[3]) if len(sys.argv) > 3 else 160
data = simulate(cal, hl, n_persons, np.random.default_rng(1))
rows = []
for s in SCHEDULES:
    d0, x0, ll0 = fit_null(data[s], cal)
    sl = SessionLik(data[s], x0, d0)
    z, ll1 = sl.fit(x0, d0)
    hl_hat = np.log(2) / np.exp(z[1])
    prof = sl.profile(z)
    lo, hi, lo_open, hi_open = interval(HL_GRID, prof, max(ll1, prof.max()), hl_hat)
    if lo_open and hi_open:
        note = "unresolved: no range of half-lives is excluded"
    elif lo_open:
        note = f"open at the short end: half-life below {hi:.2f} days"
    elif hi_open:
        note = f"open at the long end: half-life above {lo:.2f} days"
    elif hi / lo > WIDE:
        note = f"unresolved: {lo:.2f} to {hi:.2f} days spans more than a factor of {WIDE:g}"
    else:
        note = f"resolved: {lo:.2f} to {hi:.2f} days"
    rows.append({"schedule": s, "LR": round(2 * max(ll1 - ll0, 0), 2), "half_life": round(hl_hat, 3), "interval": note})
print(f"true half-life {hl} days, {n_persons} persons, 420 trials each")
print(pd.DataFrame(rows).to_string(index=False))
