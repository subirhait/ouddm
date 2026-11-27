# Replication archive

Hait, S. *Estimating the Half-Life of Cognitive Fluctuations from Short Repeated
Tasks: An Exact Likelihood for Diffusion Models with Ornstein–Uhlenbeck Drift.*

This archive rebuilds every number, table, and figure in the manuscript. You can
use it in two ways:

- **Quick rebuild** (a few minutes): rebuild the figures and the manuscript
  from the archived result files in `results/`.
- **Full rerun** (several hours on 2 cores; Simulation study 4 alone takes about
  four): rerun every simulation and analysis, then rebuild. The two data sets must
  be downloaded from OSF first.

## Contents

| Path | What it is |
|---|---|
| `ouddm-0.6.1.tar.gz` | Package source with tests and simulation scripts |
| `paper/` | Manuscript build files (template, build script, Word reference style) |
| `results/main_reps.csv`, `results/main_summary.csv` | Simulation study 1 (Table 2, Figure 2) |
| `results/design_reps.csv`, `results/design_summary.csv` | Simulation study 2 (Table 3, Figure 3) |
| `results/condition_reps.csv`, `results/condition_summary.csv` | Simulation study 3 (Table 4) |
| `results/stage1_error_n20.csv` | Stage-one error diagnostic (Table S1) |
| `results/colorshapes_*.csv` | Color Shapes application (Tables 6–8, Figure 4, Section 8) |
| `results/spaceobs_*.csv`, `results/dynamics_sv_*.csv` | Space Observer application (Tables 8–11, Figure 5, Section 9) |
| `results/schedule_calibration.csv`, `results/schedule/`, `results/schedule_nullcheck/` | Simulation study 4 (Table 12, Figure 6, Section 10) |
| `figures/` | Figures 1–6 (300 dpi) |
| `Hait_OU_DDM_manuscript.docx` | Manuscript built from these files |
| `Hait_OU_DDM_highlights.docx` | Highlights for submission, built by the same script |
| `environment.txt` | Software versions used for the reported runs |

## Software

Python 3.10 or later with numpy, scipy, pandas, and matplotlib. The reported runs
used the versions listed in `environment.txt`. Building the Word file also needs
pandoc (3.1 was used).

## Commands

### Setup (both routes)

```bash
tar xzf ouddm-0.6.1.tar.gz
cd ouddm-0.6.1
python -m pip install -e ".[dev]"
python -m pytest -q                      # 38 tests
cp -r ../paper .                         # manuscript build files are not in the package
mkdir -p results figures
```

### Quick rebuild (a few minutes)

```bash
cp -r ../results/. results/              # archived result files
python examples/paper_figures.py results figures      # Figures 1-6
python paper/build_manuscript.py                     # writes paper/Hait_OU_DDM_manuscript.docx
```

### Full rerun (several hours on 2 cores)

Run these in place of the `cp` line above, then run the two rebuild commands.
Wall times are for 2 cores.

```bash
# --- simulation studies 1-3 ---
python examples/main_study.py 100 results            # Study 1, ~36 min
python examples/condition_study.py 40 results        # Study 3, ~26 min
python examples/sv_study.py 50 results               # Study 3, drift variability, ~40 min
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
python -I examples/color_shapes_s1_noou.py <path>/data_merged.csv results               # S1 without OU, Table 7
python -I examples/color_shapes_kernels.py <path>/data_merged.csv results               # Table 8 (Color Shapes columns)
python -I examples/dynamics_with_sv.py cs <path>/data_merged.csv results
python -I examples/color_shapes_counts.py <path>/data_merged.csv results
python -I examples/color_shapes_descriptives.py <path>/ColorShapesTrialData_n69.csv results

# --- Space Observer application (data not redistributed) ---
# Download space-observer-dta.csv and task_data.csv from https://osf.io/x7h5k/
python -I examples/space_observer_application.py <so>/space-observer-dta.csv <so>/task_data.csv results   # ~60 min
python -I examples/dynamics_with_sv.py so <so>/space-observer-dta.csv <so>/task_data.csv results
python -I examples/spaceobs_profile_sv.py <so>/space-observer-dta.csv <so>/task_data.csv results          # Figure 5
python -I examples/spaceobs_robustness.py robust A <so>/space-observer-dta.csv <so>/task_data.csv results   # Table 9, ~8 min
python -I examples/spaceobs_robustness.py robust B <so>/space-observer-dta.csv <so>/task_data.csv results   # Table 9, ~7 min
python -I examples/spaceobs_robustness.py holdout <so>/space-observer-dta.csv <so>/task_data.csv results    # Table 10, ~7 min
python -I examples/spaceobs_rolling.py fit <so>/space-observer-dta.csv <so>/task_data.csv results 2 3 4 5 6 7 8          # Table 11, ~25 min
python -I examples/spaceobs_rolling.py fit-logprac <so>/space-observer-dta.csv <so>/task_data.csv results 2 3 4 5 6 7 8
python -I examples/spaceobs_rolling.py insample-logprac <so>/space-observer-dta.csv <so>/task_data.csv results
python -I examples/spaceobs_rolling.py summary results
python -I examples/spaceobs_rolling.py summary results logprac

# --- Simulation study 4 (Table 12, Figure 6): about 4 h; reps can be split across processes ---
python -I examples/schedule_study.py calibrate <so>/space-observer-dta.csv <so>/task_data.csv results   # writes results/schedule_calibration.csv
python -I examples/schedule_study.py run results/schedule_calibration.csv results/schedule none 0 60
python -I examples/schedule_study.py run results/schedule_calibration.csv results/schedule 0.25 0 30
python -I examples/schedule_study.py run results/schedule_calibration.csv results/schedule 0.65 0 30
python -I examples/schedule_study.py run results/schedule_calibration.csv results/schedule 2 0 30
python -I examples/schedule_study.py check results/schedule_calibration.csv results/schedule
python -I examples/schedule_study.py run results/schedule_calibration.csv results/schedule_nullcheck none 100 160   # fresh no-OU data sets
python -I examples/schedule_study.py nullcheck results/schedule results/schedule_nullcheck   # split-half check
python -I examples/schedule_study.py summary results/schedule results/schedule_nullcheck      # critical values from all 120 no-OU sets
```

`paper/build_manuscript.py` reads every number in the text and tables from
`results/*.csv`. It stops with an error if any placeholder lacks a value.

## Random seeds

Every replication uses a NumPy `default_rng` seeded from the condition and the
replication index; see `one_rep()` in each script. Results do not depend on the
number of worker processes. Floating-point results can differ in the last digits
across platforms and library versions.
