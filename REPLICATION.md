# Replication archive

Hait, S. *Estimating the Half-Life of Cognitive Fluctuations from Short Repeated
Tasks: An Exact Likelihood for Diffusion Models with Ornstein–Uhlenbeck Drift.*

This archive rebuilds every number, table, and figure in the manuscript. You can
use it in two ways:

- **Rebuild only** (minutes): use the result files in `results/`.
- **Rerun everything** (about two hours on 2 cores): rerun the simulations first.

## Contents

| Path | What it is |
|---|---|
| `ouddm-0.4.4.tar.gz` | Package source with tests and simulation scripts |
| `paper/` | Manuscript build files (template, build script, Word reference style) |
| `results/main_reps.csv`, `results/main_summary.csv` | Simulation study 1 (Table 2, Figure 2) |
| `results/design_reps.csv`, `results/design_summary.csv` | Simulation study 2 (Table 3, Figure 3) |
| `results/condition_reps.csv`, `results/condition_summary.csv` | Simulation study 3 (Table 4) |
| `results/stage1_error_n20.csv` | Stage-one error diagnostic (Table S1) |
| `results/colorshapes_*.csv` | Color Shapes application (Tables 5–6, Figure 4, Section 8) |
| `figures/` | Figures 1–4 (300 dpi) |
| `Hait_OU_DDM_manuscript.docx` | Manuscript built from these files |
| `Hait_OU_DDM_highlights.docx` | Highlights for submission, built by the same script |
| `environment.txt` | Software versions used for the reported runs |

## Software

Python 3.10 or later with numpy, scipy, pandas, and matplotlib. The reported runs
used the versions listed in `environment.txt`. Building the Word file also needs
pandoc (3.1 was used).

## Commands

```bash
tar xzf ouddm-0.4.4.tar.gz
cd ouddm-0.4.4
python -m pip install -e ".[dev]"
python -m pytest -q                      # 30 tests

# --- manuscript build files are not in the package; copy them in ---
cp -r ../paper .

# --- rebuild only: copy the archived results in ---
mkdir -p results figures
cp ../results/*.csv results/

# --- or rerun everything (wall times on 2 cores) ---
python examples/main_study.py 100 results            # Study 1, ~36 min
python examples/condition_study.py 40 results        # Study 3, ~26 min
python examples/design_study.py 40 results           # Study 2, ~36 min
python examples/stage1_error_diagnostic.py 20 results

# --- Color Shapes application (data not redistributed) ---
# Download from https://osf.io/ty4uf/ ("Files for June 11 2025 submission"):
#   2_processing/data_merged.csv and 1_raw_data/ColorShapesTrialData_n69.csv
python -I examples/color_shapes_application.py <path>/data_merged.csv results            # ~12 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv A results            # M3n + S1, ~56 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv B results            # S2 + S3, ~39 min
python -I examples/color_shapes_revision.py <path>/data_merged.csv C results            # M2n, ~20 min
python -I examples/color_shapes_ou_bound.py <path>/data_merged.csv results              # OU SD bounds under S1
python -I examples/color_shapes_s1_noou.py <path>/data_merged.csv results               # S1 without OU, Table 6
python -I examples/color_shapes_counts.py <path>/data_merged.csv results
python -I examples/color_shapes_descriptives.py <path>/ColorShapesTrialData_n69.csv results

# --- figures and manuscript ---
python examples/paper_figures.py results figures      # Figures 1-4
python paper/build_manuscript.py                     # writes paper/Hait_OU_DDM_manuscript.docx
```

`paper/build_manuscript.py` reads every number in the text and tables from
`results/*.csv`. It stops with an error if any placeholder lacks a value.

## Random seeds

Every replication uses a NumPy `default_rng` seeded from the condition and the
replication index; see `one_rep()` in each script. Results do not depend on the
number of worker processes. Floating-point results can differ in the last digits
across platforms and library versions.
