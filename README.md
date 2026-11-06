# Evolution and Reinforcement Learning in Changing Environments

Abhinav Guntaka · November 2026

[Read the paper](manuscript/paper.pdf) · [Reproduction guide](REPRODUCE.md) · [Validation](VALIDATION.md)

This study compares ecological neuroevolution, evolution strategies and PPO in a simulated foraging environment. The neural experiment uses ten runs per configuration, with separate tuning for fixed and varied training environments. A supporting trait experiment examines search objectives, mutation and population composition.

Under the tested settings, floor-supported ecological neuroevolution performed below PPO with domain randomization during adaptation. ES-DR scored above ES-fixed, while varied OE scored below OE-fixed. Separate tuning prevents these comparisons from isolating the effect of environmental variation. Objectives and population structure also differ between methods.

## Reproduce the results

Clone this repository or download its source ZIP. The complete data are included in `data_archives/` as 8 MiB archive parts. Use Python 3.12; the checked interpreter was 3.12.14 on Linux. Allow roughly 4 GB of free disk space for the checkout, extracted records and temporary assembly files.

```bash
python3 -m venv .venv-reanalysis
.venv-reanalysis/bin/python -m pip install -r requirements-reanalysis.txt
.venv-reanalysis/bin/python scripts/release/assemble_data.py
.venv-reanalysis/bin/python scripts/report/run_viability_checks.py
```

The assembler verifies archive and file checksums before extraction. The nine statistical checks take several minutes on a CPU and use the saved experimental records. Run Python without `-O`, since the checks use assertions. [REPRODUCE.md](REPRODUCE.md) also covers the implementation tests, figure generation and manuscript build.

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
| `provenance/` | File checksums, source revisions and distribution edits |
| `data_archives/` | Raw results, trained policies, snapshots and tuning records |

The protocols were written privately before the corresponding runs. ES-fixed was specified after the core results and before its own runs; longer OE training was exploratory. The paper and protocol excerpts identify these stages.

Final policies and snapshots are included for all six neural configurations. Core tuning scores and trial records are included; the core tuning policy files were not retained. Checkpoint bytes were verified against the source archive; fresh GPU training and policy evaluation have not been repeated for this release.

The release preserves scientific values and removes machine host identifiers and administrative account details. Protocol excerpts retain analysis rules and dated design decisions. The original experimental source revision is recorded in `provenance/`.

## Funding and attribution

The author personally funded the project. Total expenditure was **US$87**. `results/publication/compute_spend.csv` contains partial historical stage estimates, not the project total.

The software uses the existing GNU GPL version 3 license in [LICENSE](LICENSE). The trait simulator derives from MinuteLabs.io, and PPO adapts PureJaxRL. Upstream revisions and license notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The manuscript and research data have no separate reuse license assigned. Cite the manuscript using [CITATION.cff](CITATION.cff).
