# Release validation

Checked on 6 November 2026 (America/Los_Angeles), using Linux and Python 3.12.14.

| Check | Result |
|---|---|
| Data assembly | All three archive hashes and all 3,081 file hashes verified |
| Data preservation | 2,859 files identical to their sources; 222 JSON files differ only in declared administrative fields or commentary |
| Scientific implementation | Computational AST of all 40 source modules and values in all 16 configurations match the recorded source |
| Statistical checks | All nine checks passed from the assembled release |
| Tuning and saved policies | All 48 trial scores and six selections reproduced; 60 final policies and 600 training snapshots present |
| Recorded implementation and reporting test run | 86 passed; one dependency deprecation warning; 101.04 seconds |
| Figures and tables | Eleven tables rebuilt; ten figures regenerated; all seven manuscript figure datasets match the frozen numerical inputs |
| Manuscript | 22 pages compiled without TeX warnings or unresolved references; all rendered pages visually inspected |

Commands are in `REPRODUCE.md`. The selected tests run on CPU; the full suite also includes learning benchmarks. The dependency warning concerns TensorFlow Probability's use of a deprecated JAX interface. GPU training and policy evaluation were not rerun.
