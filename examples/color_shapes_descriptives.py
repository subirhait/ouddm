"""Missing-response (timeout) rates in the raw Color Shapes trial file, by choice
urgency. Usage: python -I color_shapes_descriptives.py <ColorShapesTrialData_n69.csv> [outdir]"""
import sys
from pathlib import Path

import pandas as pd

raw = pd.read_csv(sys.argv[1])
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
raw = raw[raw.participant_id != 4039]          # excluded by the original authors
raw["urgent"] = raw.task_version.str.contains("RT-")
rows = []
for u, g in raw.groupby("urgent"):
    rows.append({"window": "3000 ms" if u else "10000 ms", "trials": len(g),
                 "no_response_rate": (g.response_time == -999).mean()})
rows.append({"window": "all", "trials": len(raw), "no_response_rate": (raw.response_time == -999).mean()})
pd.DataFrame(rows).to_csv(outdir / "colorshapes_timeouts.csv", index=False)
print(pd.DataFrame(rows))
