"""Training curves (PREREG Section 8.4, defined 3 Nov 2026 before any final run).

For each finished run folder: every snapshot (one per tenth of training) is scored like a tuning
trial, the mean over the 24 validation environments of (competence - random competence),
zero-shot, 8 episodes, validation episode seeds (scripts/part3/tune.py: score). Writes
curves.json in the run folder. Uses validation environments only; never the test split.

  venv3/bin/python scripts/part3/curves.py --runs results/part3/final/oe_seed5000 ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from evo2.run import load_methods_config
from evo2.runio import enable_compile_cache, load_pickle, require_clean, write_json

_spec = importlib.util.spec_from_file_location("tune", Path(__file__).resolve().parent / "tune.py")
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)


def run_curve(run: Path, n_episodes: int, max_envs: int | None = None) -> dict:
    envs, ep_seeds = tune.validation_envs()
    if max_envs:
        envs = envs[:max_envs]
    points = []
    for k in range(1, 11):
        snap = load_pickle(run / "snapshots" / f"{k}.pkl")
        s = tune.score(snap["policy"], envs, ep_seeds, n_episodes)
        points.append({"tenth": k, "agent_steps": snap["agent_steps"], "score": s["score"], "envs": s["envs"]})
    return {"run": str(run), "n_envs": len(envs), "episodes_per_env": n_episodes, "points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--quick", action="store_true", help="2 environments, 2 episodes (smoke only)")
    a = ap.parse_args()
    enable_compile_cache()   # saves compile time only; see evo2.runio
    g = require_clean("test" if a.quick else "final", a.quick)
    n_ep = 2 if a.quick else int(load_methods_config()["evaluation"]["episodes_per_env"])
    for run in map(Path, a.runs):
        if not (run / "done.json").exists():
            raise SystemExit(f"{run} has not finished")
        out = run_curve(run, n_ep, 2 if a.quick else None)
        out.update({"git": g, "quick": a.quick})
        write_json(run / "curves.json", out)
        print(run.name, [round(p["score"], 2) for p in out["points"]], flush=True)


if __name__ == "__main__":
    main()
