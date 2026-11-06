"""Part 3 pilot stage P0: speed and repeatability on one machine (check-in 8).

For each method: two short training runs from the same pilot seed (repeatability:
are the trained policies bit-identical?) and the speed in living agent-steps per
second, measured after the first chunk so that compile time is left out. For OE
it also runs several OE trainings at once, as separate processes, to see how
speed adds up when one machine runs many seeds. Writes summary.json in --out.

  venv3/bin/python scripts/part3/throughput.py --out results/part3/throughput/<machine>
  venv3/bin/python scripts/part3/throughput.py --quick       # tiny version, any machine

Seeds: pilot range, 3090-3099 (kept apart from the pilot runs, 3000-3089).
Never touches the test split.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from evo2.run import METHODS, load_methods_config
from evo2.runio import ROOT, enable_compile_cache, hardware, load_pickle, require_clean, write_json
from evo2.seeds import check_seed
from evo2.train_loop import train_run

SEED = 3090
CONCURRENT_SEEDS = range(3091, 3099)   # up to 8 OE runs at once


def leaves_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _units(line: dict) -> int:
    for k in ("updates", "generations", "steps"):
        if k in line:
            return int(line[k])
    return 0


def speed(run_dir: Path) -> dict:
    """Living agent-steps per second over full-size chunks only, leaving out the
    first one (it includes compiling). Smaller chunks near the end are left out
    too: in P0 each compiled its own size (since 2 Nov 2026 all sizes share one
    program, but a short chunk still carries more fixed overhead per step)."""
    lines = [json.loads(ln) for ln in (run_dir / "progress.jsonl").read_text().splitlines() if ln.strip()]
    done = json.loads((run_dir / "done.json").read_text())
    out = {"agent_steps": done["agent_steps"], "wall_seconds_total": done["wall_seconds"], "chunks": len(lines)}
    full = max(_units(ln) for ln in lines)
    steps = secs = 0.0
    seen_full = False
    for prev, ln in zip([None] + lines[:-1], lines):
        if _units(ln) != full:
            continue
        if not seen_full:           # the first full chunk compiles
            seen_full = True
            continue
        steps += ln["agent_steps"]
        secs += ln["wall_seconds"] - prev["wall_seconds"]
    out["full_chunks_measured"] = sum(1 for ln in lines if _units(ln) == full) - 1
    out["agent_steps_per_second"] = steps / secs if secs > 0 else None
    return out


def concurrent_oe(out: Path, k: int, budget: int, purpose: str, quick: bool) -> dict:
    """k OE trainings at once, as separate processes; total speed."""
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"        # several processes share one GPU
    if jax.default_backend() == "cpu":
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    seeds = list(CONCURRENT_SEEDS)[:k]
    t0 = time.time()
    procs = []
    for s in seeds:
        cmd = [sys.executable, str(ROOT / "scripts/part3/train.py"), "--method", "oe", "--purpose", purpose,
               "--seed", str(s), "--budget", str(budget), "--out", str(out / f"concurrent{k}")]
        if quick:
            cmd = [sys.executable, str(ROOT / "scripts/part3/train.py"), "--method", "oe", "--quick",
                   "--seed", str(s + 900), "--out", str(out / f"concurrent{k}")]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    errors = [p.communicate()[1][-2000:] for p in procs if p.wait() != 0]
    wall = time.time() - t0
    rows = []
    for s in seeds:
        d = out / f"concurrent{k}" / f"oe_seed{s + 900 if quick else s}"
        if (d / "done.json").exists():
            rows.append(speed(d))
    total = sum(r["agent_steps"] for r in rows)
    return {"processes": k, "finished": len(rows), "errors": errors, "wall_seconds": round(wall, 1),
            "total_agent_steps": total, "total_agent_steps_per_second_incl_compile": total / wall if wall else None,
            "per_run_agent_steps_per_second": [r.get("agent_steps_per_second") for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--budget", type=int, default=2_000_000, help="PPO and ES agent-steps per run")
    ap.add_argument("--oe-budget", type=int, default=500_000)
    ap.add_argument("--concurrent", type=int, nargs="*", default=[1, 4, 8], help="OE processes at once")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    enable_compile_cache()   # saves compile time only; see evo2.runio
    purpose = "test" if a.quick else "pilot"
    seed = 3990 if a.quick else SEED
    check_seed(purpose, seed)
    if a.quick:
        a.budget, a.oe_budget, a.concurrent = 50_000, 50_000, [2]
    g = require_clean(purpose, a.quick)
    out = Path(a.out or ROOT / "results/part3/throughput" / ("quick" if a.quick else os.uname().nodename))
    out.mkdir(parents=True, exist_ok=True)
    mcfg = load_methods_config()
    summary = {"git": g, "hardware": hardware(), "budget": a.budget, "oe_budget": a.oe_budget, "methods": {}}
    for m in a.methods:
        budget = a.oe_budget if m in ("oe", "oe_fixed") else a.budget
        dirs = [out / f"{m}_seed{seed}", out / f"{m}_seed{seed}_repeat"]
        for d in dirs:
            train_run(m, purpose, seed, d, budget, mcfg, quick=a.quick)
        summary["methods"][m] = {
            "runs": [speed(d) for d in dirs],
            "repeat_bit_identical": leaves_equal(load_pickle(dirs[0] / "policy.pkl"), load_pickle(dirs[1] / "policy.pkl")),
        }
        print(m, json.dumps(summary["methods"][m]), flush=True)
        write_json(out / "summary.json", summary)
    summary["concurrent_oe"] = [concurrent_oe(out, k, a.oe_budget, purpose, a.quick) for k in a.concurrent]
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["concurrent_oe"]), flush=True)


if __name__ == "__main__":
    main()
