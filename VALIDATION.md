# Validation

The following results were recorded for the 6 November 2026 package build on Linux with Python 3.12.14. Commands to repeat them are in [REPRODUCE.md](REPRODUCE.md).

| Check | Recorded result |
|---|---|
| Data assembly | Three archive hashes and 3,081 file hashes verified |
| Statistical analysis | Nine checks passed |
| Tuning and saved policies | 48 tuning scores and six selections reproduced; 60 final policies and 600 snapshots present |
| Implementation and reporting tests | 86 passed in 101.04 seconds, with one dependency deprecation warning |
| Figures and tables | Eleven tables and ten figures generated; seven manuscript figure datasets matched their inputs |

The dependency warning concerns TensorFlow Probability's use of a deprecated JAX interface. These checks ran on CPU and did not include GPU training or policy evaluation.
