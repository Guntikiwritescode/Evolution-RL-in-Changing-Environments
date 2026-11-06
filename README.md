# Evolution and Reinforcement Learning in Changing Environments

Abhinav Guntaka · November 2026

[Paper](manuscript/paper.pdf) · [Reproduction](REPRODUCE.md) · [Documentation](docs/README.md)

Code and data for a study of ecological neuroevolution, evolution strategies and PPO in a simulated foraging environment. The experiments compare fixed and varied training environments, and examine how search objectives, mutation and population composition affect adaptation.

## Quick start

Analyze the supplied experimental data on CPU with Python 3.12:

```bash
python3 -m venv .venv-reanalysis
.venv-reanalysis/bin/python -m pip install -r requirements-reanalysis.txt
.venv-reanalysis/bin/python scripts/release/assemble_data.py
.venv-reanalysis/bin/python scripts/report/run_viability_checks.py
```

The GitHub checkout includes the data archive parts in [data_archives/](data_archives/). For the compact download, use the companion ZIP instructions in [REPRODUCE.md](REPRODUCE.md). That guide also covers tests, figures, the manuscript build and training new runs.

## Repository structure

| Path | Contents |
|---|---|
| [src/](src/), [configs/](configs/) | Simulators, learning methods and experiment settings |
| [scripts/](scripts/), [tests/](tests/) | Training, evaluation, analysis and tests |
| [results/publication/](results/publication/) | Figure data and numerical summaries |
| [manuscript/](manuscript/) | Paper PDF, LaTeX source, figures and tables |
| [docs/](docs/) | Experiment protocols, specifications and results |
| [environments/](environments/) | Training and CPU analysis dependencies |
| [data_archives/](data_archives/) | Data archive parts in the GitHub checkout |
| [provenance/](provenance/) | Data manifests, checksums and source revisions |
| [reference/](reference/) | Wrapper for the MinuteLabs Rust simulator |

## License and citation

The software is licensed under [GNU GPL version 3](LICENSE). The trait simulator derives from MinuteLabs.io, and PPO adapts PureJaxRL; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for credits and license notices. The manuscript and research data have no separate reuse license assigned.

To cite this work, use [CITATION.cff](CITATION.cff).
