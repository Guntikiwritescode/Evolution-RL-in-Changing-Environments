# Reproduce the results

Commands below run from the repository root unless stated otherwise.

## 1. Assemble and verify data

Follow the four commands in the root README. They install NumPy, SciPy and pandas, assemble all three data archives from the checked-in parts under `data_archives/`, and run nine statistical checks. Logs are written to `results/audit/viability/rerun/`; independent calculations go to `results/audit/viability/reanalysis/`. Assembly stops on missing files or checksum mismatches; the check chain stops on its first failure.

If using the separately distributed complete ZIP files, put them in one directory and pass `--assets /path/to/that/directory` to `assemble_data.py`. Both forms use the same archive and payload checksums.

## 2. Check policies and tuning records

```bash
.venv-reanalysis/bin/python scripts/release/check_saved_models.py
```

This checks all 48 tuning scores, the six selected configurations, and the presence of 60 final policies and 600 training snapshots. File integrity is checked during archive assembly.

## 3. Selected implementation and reporting tests

Use the broader pinned CPU environment for the synthetic simulator/rule tests and reporting regressions:

```bash
python3 -m venv .venv-validation
.venv-validation/bin/python -m pip install -r requirements-validation.txt
PYTHONPATH=src JAX_PLATFORMS=cpu .venv-validation/bin/python -m pytest tests/test_part3_analysis.py tests/test_part3_addendum.py tests/test_part3_explore_report.py tests/test_part2_sim.py tests/test_rules.py tests/test_report_numbers.py tests/test_report_bib.py tests/test_publication_preparation.py -q
```

This command selects 86 implementation and reporting tests, including synthetic simulator cases. Running plain `pytest` also selects learning benchmarks.

`requirements-validation.txt` pins the CPU test dependencies; `environments/validation-freeze.txt` records the installed environment. Historical trait and neural dependencies are listed separately in `environments/`.

## 4. Rebuild the manuscript

With Python 3 and a TeX installation containing the packages listed in `manuscript/main.tex`:

```bash
cd manuscript
python3 -W error build_tables.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The table generator uses Python's standard library. The build uses the supplied figures and bibliography and produces `main.pdf`.

## 5. Regenerate figures from retained records

From the repository root, after assembling the data, use the validation environment:

```bash
.venv-validation/bin/python scripts/report/prepare_publication.py
```

This regenerates the ten-figure preparation set, numerical JSON and CSV exports, and `figures/publication/figure_review.pdf`. Seven of these figures are used in the manuscript. To copy those rebuilt PDFs into the manuscript and verify that their numerical datasets match the manuscript's frozen inputs:

```bash
.venv-validation/bin/python scripts/release/sync_manuscript_figures.py
```

Recompile the manuscript afterward. Legacy report helpers use `paper/v2/`; the current manuscript source is in `manuscript/`.

## Training new runs

The commands above analyze existing runs. Historical neural training used Python 3.13.18 and RTX 4090 hardware and has not been repeated for this release. Its dependency list is `environments/historical_requirements_part3.txt`; the trait experiment's list is `environments/historical_requirements.txt`. These are separate from the checked CPU reanalysis environment. Training entry points require a Git checkout with a commit; their command-line options are documented by `--help`.
