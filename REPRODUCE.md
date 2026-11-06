# Reproduce the results

Commands below run from the repository root unless stated otherwise.

## 1. Assemble and verify data

Follow the four commands in the root README. They install NumPy, SciPy and pandas, assemble all three data archives and run nine statistical checks. Logs are written to `results/audit/viability/rerun/`; independent calculations go to `results/audit/viability/reanalysis/`. Assembly stops on missing files or checksum mismatches; the check chain stops on its first failure.

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

This command runs 86 implementation and reporting tests, including small synthetic simulator cases. It does not rerun the research experiments. All 16 test files are included; running plain `pytest` also selects learning benchmarks outside this validation suite.

`requirements-validation.txt` records the directly relevant package versions used for validation. `environments/validation-freeze.txt` records the complete environment when available. These are CPU reanalysis records, not a freshly verified historical GPU-training environment. Do not merge historical trait and neural requirement files.

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

The table generator uses only Python's standard library. These commands use the supplied vector figures and bibliography and do not access the network. They produce `main.pdf`.

## 5. Regenerate figures from retained records

From the repository root, after assembling the data, use the validation environment:

```bash
.venv-validation/bin/python scripts/report/prepare_publication.py
```

This regenerates the ten-figure preparation set, numerical JSON and CSV exports, and `figures/publication/figure_review.pdf`. Seven of these figures are used in the manuscript. To copy those rebuilt PDFs into the manuscript and verify that their numerical datasets match the manuscript's frozen inputs:

```bash
.venv-validation/bin/python scripts/release/sync_manuscript_figures.py
```

Recompile the manuscript afterward. Historical report helpers still write under `paper/v2/`; that directory contains only their legacy numeric/bibliography support files and is not the final manuscript source. Some optional archival scripts require older external reference material and are outside the validated quickstart. Do not run training, evaluation or cloud provisioning commands to check this release.

## Scope

These commands recompute the reported statistics and figures and check selected implementation behavior. A fresh GPU training and policy-evaluation reproduction has not been performed for this release. Historical neural training used Python 3.13.18 and RTX 4090 hardware; the CPU validation environment is recorded separately. Training entry points require a Git checkout with a commit.

The manuscript's limits remain: fixed test environments, separate tuning, a selected empirical normalization reference, and incomplete mechanism controls. The prospective protocols were private; ES-fixed and the longer-training extension have their own later chronology.
