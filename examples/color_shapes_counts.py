"""Trial, session, and person counts for the Color Shapes analysis subsets.

Usage: python -I color_shapes_counts.py <data_merged.csv> [outdir]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_shapes_application as app  # noqa: E402

d = app.load(Path(sys.argv[1]))
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
rows = []
for name, sub in (("all", d), ("long deadline only", d[d.ChoiceUrgencyCoded == 0])):
    rows.append({"subset": name, "trials": len(sub), "persons": sub["person"].nunique(),
                 "sessions": sub.groupby(["person", "time"]).ngroups})
pd.DataFrame(rows).to_csv(outdir / "colorshapes_counts.csv", index=False)
print(pd.DataFrame(rows))
