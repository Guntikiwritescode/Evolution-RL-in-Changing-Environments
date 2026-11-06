# Evolution and Reinforcement Learning in Changing Environments

Abhinav Guntaka · November 2026

[Read the paper](manuscript/paper.pdf) · [Reproduction guide](REPRODUCE.md) · [Validation](VALIDATION.md)

This study compares ecological neuroevolution, evolution strategies and PPO in a simulated foraging environment. The neural experiment uses ten runs per configuration, with separate tuning for fixed and varied training environments. A supporting trait experiment examines search objectives, mutation and population composition.

Under the tested settings, floor-supported ecological neuroevolution performed below PPO with domain randomization during adaptation. ES-DR scored above ES-fixed, while varied OE scored below OE-fixed. Each comparison uses the selected settings for both configurations.

## Reproduce the results

The checkout includes the three data archives as parts in `data_archives/`. The assembler joins and verifies those parts before extracting the records. Use Python 3.12; the checked interpreter was 3.12.14 on Linux. Allow roughly 4 GB of free disk space for the checkout, extracted records and temporary assembly files.

```bash
python3 -m venv .venv-reanalysis
.venv-reanalysis/bin/python -m pip install -r requirements-reanalysis.txt
.venv-reanalysis/bin/python scripts/release/assemble_data.py
.venv-reanalysis/bin/python scripts/report/run_viability_checks.py
```

The assembler verifies archive and file checksums before extraction. The nine statistical checks take several minutes on a CPU and use the saved experimental records. Run Python without `-O`, since the checks use assertions. [REPRODUCE.md](REPRODUCE.md) also covers the implementation tests, figure generation and manuscript build.

Alternatively, put the three separately distributed companion ZIP files in one directory and replace the assembly command with `.venv-reanalysis/bin/python scripts/release/assemble_data.py --assets /path/to/data`. Both forms contain the same records and use the same archive checksums.

## Contents

| Path | Contents |
|---|---|
| `manuscript/` | PDF, LaTeX source, bibliography, figures and table inputs |
| `src/`, `configs/` | Trait simulator, neural methods and experiment settings |
| `scripts/`, `tests/` | Training, evaluation, analysis and verification code |
| `results/publication/` | Figure data and numerical summaries |
| `docs/` | Scientific protocols, specifications and result reports |
| `environments/` | Historical dependencies and the checked CPU environment |
| `reference/` | Wrapper for the pinned MinuteLabs Rust simulator |
| `provenance/` | File checksums and source revisions |
| `data_archives/` | Archive parts containing raw results, trained policies, snapshots and tuning records |

The data include final policies and snapshots for all six neural configurations, together with tuning scores and trial records. Core tuning policy files were not retained. Source revisions and checksums are in `provenance/`.

## Funding and attribution

The author personally funded the project. Total expenditure was **US$87**. `results/publication/compute_spend.csv` contains partial historical stage estimates, not the project total.

The software uses the existing GNU GPL version 3 license in [LICENSE](LICENSE). The trait simulator derives from MinuteLabs.io, and PPO adapts PureJaxRL. Upstream revisions and license notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The manuscript and research data have no separate reuse license assigned. Cite the manuscript using [CITATION.cff](CITATION.cff).
