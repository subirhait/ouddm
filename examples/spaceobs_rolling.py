"""Space Observer: rolling-origin holdout, with prediction gains by the gap
from the previous session.

Sessions are ranked within each person by time (rank 0 = first session).
Fold k holds out every session of rank k and fits the models to all sessions
of rank < k; folds start at k = 2, so each person contributes at least two
training sessions. Models: no dynamics and OU, both with across-trial drift
variability, person-specific practice slopes and session deviations. Choices
of held-out sessions are predicted conditional on the presented difficulty;
held-out RTs are used only for the trial screen (0.2 to 5 s).

A second version ("logprac") adds the centered log session number as a
further drift predictor, so practice can level off.

Usage:
  python -I spaceobs_rolling.py fit <space-observer-dta.csv> <task_data.csv> <outdir> <k> [<k> ...]
  python -I spaceobs_rolling.py fit-logprac <space-observer-dta.csv> <task_data.csv> <outdir> <k> [<k> ...]
  python -I spaceobs_rolling.py insample-logprac <space-observer-dta.csv> <task_data.csv> <outdir>
  python -I spaceobs_rolling.py summary <outdir> [logprac]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ouddm.design_model import DesignData, fit_design, predict_choices  # noqa: E402
from space_observer_application import load  # noqa: E402

BASE = dict(drift=("log_int_c", "game_c"), nugget=True, slope="game_c", sv=True)
BINS = [(0.0, 0.25, "under 6 h"), (0.25, 1.0, "6 to 24 h"), (1.0, 2.0, "1 to 2 days"), (2.0, np.inf, "2 days or more")]


def add_logprac(d):
    d = d.copy()
    lg = np.log(d["game"])
    d["log_game_c"] = lg - lg.mean()
    return d


def fold(d, k, base=BASE):
    st = d.groupby(["person", "occasion"]).time.first().reset_index().sort_values(["person", "time"])
    st["rank"] = st.groupby("person").cumcount()
    st["gap_days"] = st.groupby("person").time.diff()
    d = d.merge(st[["person", "occasion", "rank", "gap_days"]], on=["person", "occasion"])
    train, test = d[d["rank"] < k], d[d["rank"] == k]
    rows = []
    preds = {}
    for kern in ("none", "ou"):
        t0 = time.time()
        dtr = DesignData.from_trials(train, "stim", **base, kernel=kern)
        dte = DesignData.from_trials(test, "stim", **base, kernel=kern, outcomes=False)
        f = fit_design(dtr, se=False)
        pr = np.clip(predict_choices(f.x, dtr, dte), 1e-12, 1 - 1e-12)
        preds[kern] = pr
        rows.append({"k": k, "kernel": kern, "train_loglik": f.loglik, "train_aic": f.aic,
                     "halflife_days": f.halflife, "seconds": time.time() - t0})
        print(rows[-1], flush=True)
    # trial order of the test DesignData: sorted by person, time, occasion (same for both kernels)
    tt = test.sort_values(["person", "time", "occasion"]).reset_index(drop=True)
    y = tt.upper.to_numpy()
    out = tt[["person", "occasion", "gap_days"]].copy()
    for kern, pr in preds.items():
        out[f"loss_{kern}"] = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    sess = (out.groupby(["person", "occasion"])
            .agg(gap_days=("gap_days", "first"), n=("loss_ou", "size"),
                 loss_none=("loss_none", "sum"), loss_ou=("loss_ou", "sum")).reset_index())
    sess["k"] = k
    return sess, pd.DataFrame(rows)


def summary(outdir, tag="", n_boot=2000, seed=2026):
    s = pd.concat([pd.read_csv(p) for p in sorted(outdir.glob(f"spaceobs_rolling{tag}_sessions_k*.csv"))])
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("all", s)] + [(lab, s[(s.gap_days >= lo) & (s.gap_days < hi)]) for lo, hi, lab in BINS]
    for lab, g in groups:
        per = g.groupby("person").agg(n=("n", "sum"), dn=("loss_none", "sum"), do=("loss_ou", "sum"))
        diff = (per.do - per.dn).to_numpy(); n = per.n.to_numpy()
        est = diff.sum() / n.sum()
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        bs = diff[idx].sum(1) / n[idx].sum(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        rows.append({"gap": lab, "persons": len(per), "sessions": len(g), "choices": int(g.n.sum()),
                     "log_loss_none": g.loss_none.sum() / g.n.sum(), "log_loss_ou": g.loss_ou.sum() / g.n.sum(),
                     "diff_ou_minus_none": est, "lo95": lo, "hi95": hi})
    res = pd.DataFrame(rows)
    res.to_csv(outdir / f"spaceobs_rolling{tag}_summary.csv", index=False)
    print(res.to_string(index=False))


def insample(d, outdir, base):
    rows = []
    for kern in ("none", "ou"):
        t0 = time.time()
        f = fit_design(DesignData.from_trials(d, "stim", **base, kernel=kern), se=False)
        rows.append({"kernel": kern, "loglik": f.loglik, "aic": f.aic, "halflife_days": f.halflife,
                     "beta_log_game": f.params["beta:log_game_c"], "beta_game": f.params["beta:game_c"],
                     "seconds": time.time() - t0})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r["lr_vs_none"] = 2 * (r.loglik - r.loglik.iloc[0])
    r.to_csv(outdir / "spaceobs_logprac_insample.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(Path(sys.argv[2]), "_" + sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        mode = sys.argv[1]
        tp, sp, outdir = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
        d, _ = load(tp, sp)
        base, tag = BASE, ""
        if mode.endswith("logprac"):
            d = add_logprac(d)
            base, tag = dict(BASE, drift=("log_int_c", "game_c", "log_game_c")), "_logprac"
        if mode == "insample-logprac":
            insample(d, outdir, base)
            sys.exit()
        for k in map(int, sys.argv[5:]):
            sess, fits = fold(d, k, base)
            sess.to_csv(outdir / f"spaceobs_rolling{tag}_sessions_k{k}.csv", index=False)
            fits.to_csv(outdir / f"spaceobs_rolling{tag}_fits_k{k}.csv", index=False)
