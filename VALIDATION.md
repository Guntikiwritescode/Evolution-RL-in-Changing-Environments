# Release validation

Checked on 6 November 2026 (America/Los_Angeles), using Linux and Python 3.12.14.

| Check | Result |
|---|---|
| Data assembly | All three archive hashes and all 3,081 file hashes verified |
| Data preservation | 2,859 files identical to their sources; 222 JSON files differ only in declared administrative fields or commentary |
| Scientific implementation | All 40 source modules and 16 configurations match the audited working source |
| Statistical checks | All nine checks passed from the assembled release |
| Tuning and saved policies | All 48 trial scores and six selections reproduced; 60 final policies and 600 training snapshots present |
| Implementation and reporting tests | 86 passed; one dependency deprecation warning; 101.04 seconds |
| Figures and tables | Eleven tables rebuilt; ten figures regenerated; all seven manuscript figure datasets match the frozen numerical inputs |
| Manuscript | 24 pages compiled without TeX warnings or unresolved references; all rendered pages visually inspected |

The commands are in `REPRODUCE.md`. The documented eight-file test suite was run in the checked CPU environment. All sixteen test files are distributed; the full suite also contains learning benchmarks. The warning concerns a deprecated JAX interface used by TensorFlow Probability. Installation on other platforms was not tested.

The final review repeated data assembly, all nine statistical scripts, the saved-model checks, the 86-test suite and figure regeneration. It also checked citations, numerical claims, archive contents and manuscript layout. Edits clarified the limits of causal interpretation and removed redundant prose; scientific values, source modules and experimental configurations are unchanged.

Fresh GPU training and policy evaluation were not performed during release verification. Core tuning scores and run records are included, but their policy files were not retained. The paper describes the statistical and experimental limits of the results.
