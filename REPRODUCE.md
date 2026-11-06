# Reproduce the results

Run commands from the repository root unless stated otherwise. CPU analysis uses Python 3.12; the tested environment is Linux with Python 3.12.14. Allow roughly 4 GB of disk space for data and archive assembly.

## Data and analysis

Start with the [README quick start](README.md#quick-start) to install dependencies, assemble the data and run the nine statistical checks. Assembly verifies archive and file checksums. The checks take several minutes on CPU and write logs to `results/audit/viability/rerun/` and calculations to `results/audit/viability/reanalysis/`. Run Python without `-O`, because the checks use assertions.

The compact download omits `data_archives/`. Put its three companion files—`Evolving_AI_Study1_Data.zip`, `Evolving_AI_Study2_Data.zip` and `Evolving_AI_Study3_Data.zip`—in one directory, and replace the assembly command with:

```bash
.venv-reanalysis/bin/python scripts/release/assemble_data.py --assets /path/to/data
```

To check tuning selections and saved policy coverage:

```bash
.venv-reanalysis/bin/python scripts/release/check_saved_models.py
```

This checks 48 tuning scores, six selected configurations, 60 final policies and 600 training snapshots. Core tuning policy files were not retained; their scores and trial records are included.

## Implementation tests

Install the broader CPU environment and run the simulator and reporting tests:

```bash
python3 -m venv .venv-validation
.venv-validation/bin/python -m pip install -r requirements-validation.txt
PYTHONPATH=src JAX_PLATFORMS=cpu .venv-validation/bin/python -m pytest tests/test_part3_analysis.py tests/test_part3_addendum.py tests/test_part3_explore_report.py tests/test_part2_sim.py tests/test_rules.py tests/test_report_numbers.py tests/test_report_bib.py tests/test_publication_preparation.py -q
```

Running plain `pytest` also selects learning benchmarks. Package versions are pinned in `requirements-validation.txt`; the installed environment is listed in `environments/validation-freeze.txt`. Previous check results are in [VALIDATION.md](VALIDATION.md).

## Figures and manuscript

After assembling the data, regenerate figure data and PDFs with:

```bash
.venv-validation/bin/python scripts/report/prepare_publication.py
.venv-validation/bin/python scripts/release/sync_manuscript_figures.py
```

The first command generates ten figures, numerical JSON and CSV exports, and `figures/publication/figure_review.pdf`. The second copies the seven manuscript figures and checks their datasets against the manuscript inputs.

To rebuild the tables and paper, install TeX with the packages used in `manuscript/main.tex`, then run:

```bash
cd manuscript
python3 -W error build_tables.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The table generator uses Python's standard library. The compiled paper is `manuscript/main.pdf`. Older report helpers target `paper/v2/`; use `manuscript/` for the current paper.

## Training new runs

The neural experiments used Python 3.13.18 and RTX 4090 hardware, with dependencies in `environments/historical_requirements_part3.txt`. Trait experiment dependencies are in `environments/historical_requirements.txt`.

Training entry points are `scripts/part3/train.py` and `scripts/part2/run_part2.py`; use `--help` for options and `PYTHONPATH=src` for neural scripts. Training requires a Git checkout with a commit, and full trait runs require a clean working tree. Experiment settings are in `configs/`, with protocols in `docs/`.
