"""Training one run to a fixed agent-step budget, resumably (SPEC_part3.md, Section 6).

A run folder holds:
  run.json            provenance (commit, clean tree, hardware), settings, budget
  progress.jsonl      one line per chunk: agent-steps so far, wall time, method numbers
  checkpoint.pkl      the full state (to resume after a stop): saved at most once a
                      minute, at every snapshot and at every stop; on resume, progress
                      lines after it are dropped and those chunks run again (same result)
  snapshots/k.pkl     the policy at the first chunk end past each tenth of the
                      budget (k = 1..10), for learning curves and diversity over time
  policy.pkl          the trained policy (written at the end)
  done.json           totals; written last, so its presence means the run finished
  failed.json         written instead if parameters stop being finite (exclusion rule)

Agent-steps are counted the same way for every method: agents alive at the
start of each step. Chunk sizes follow evo2.run.chunk_size, so no method
passes the budget by a full unit, and a resumed run takes the same chunks as
an uninterrupted one. Snapshots are taken at the first chunk end past each
tenth of the budget (their exact agent-steps are saved with them).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from evo2.run import advance, get_trainer, to_numpy
from evo2.runio import (Timer, append_line, hardware, load_pickle, require_clean, save_pickle,
                        write_atomic, write_json)
from evo2.seeds import base_key, stream_key

N_SNAPSHOTS = 10
CHECKPOINT_SECONDS = 60.0   # the full state is saved at most this often, and always at a stop


def all_finite(tree) -> bool:
    return all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree)
               if np.issubdtype(np.asarray(x).dtype, np.floating))


def _trim_progress(path: Path, chunks: int):
    """Keep only progress lines up to the checkpoint's chunk count."""
    if not path.exists():
        return
    keep = [ln for ln in path.read_text().splitlines() if ln.strip() and json.loads(ln)["chunk"] <= chunks]
    write_atomic(path, ("\n".join(keep) + ("\n" if keep else "")).encode())


def train_run(method: str, purpose: str, seed: int, out_dir: Path, budget: int, mcfg: dict,
              quick: bool = False, max_chunks: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, failed = out_dir / "done.json", out_dir / "failed.json"
    g = require_clean(purpose, quick)
    key = stream_key(base_key(purpose, seed), "train_run")
    meta = {"method": method, "purpose": purpose, "seed": seed, "budget_agent_steps": budget,
            "methods_config": mcfg, "git": g, "hardware": hardware(), "quick": quick}
    run_json = out_dir / "run.json"
    if run_json.exists():
        old = json.loads(run_json.read_text())
        same = all(old[k] == meta[k] for k in ("method", "purpose", "seed", "budget_agent_steps",
                                                "methods_config")) and old["git"]["commit"] == g["commit"]
        if not same:
            raise SystemExit("run folder was started with another method, seed, purpose, commit, "
                             "config or budget; refusing to mix")
        for f in (done, failed):
            if f.exists():
                return json.loads(f.read_text())
    else:
        write_json(run_json, meta)

    trainer = get_trainer(method, mcfg)
    ckpt, prog, snaps = out_dir / "checkpoint.pkl", out_dir / "progress.jsonl", out_dir / "snapshots"
    snaps.mkdir(exist_ok=True)
    if ckpt.exists():
        saved = load_pickle(ckpt)
        state = jax.tree_util.tree_map(jax.numpy.asarray, saved["state"])
        steps, chunks, wall = saved["agent_steps"], saved["chunks"], saved["wall_seconds"]
        _trim_progress(prog, chunks)   # chunks after the checkpoint will be run again
    else:
        state, steps, chunks, wall = trainer.init(key), 0, 0, 0.0

    timer = Timer()
    counter = {"chunks": chunks, "bad": False, "saved_at": 0.0}

    def checkpoint(st, total):
        now = wall + timer.elapsed()
        save_pickle(ckpt, {"state": to_numpy(st), "agent_steps": total, "chunks": counter["chunks"],
                           "wall_seconds": now})
        counter["saved_at"] = timer.elapsed()

    def record(st, total, summary):
        counter["chunks"] += 1
        line = {"chunk": counter["chunks"], "agent_steps_total": total,
                "wall_seconds": round(wall + timer.elapsed(), 2), **summary}
        finite = all_finite(trainer.policy(st))
        line["finite"] = finite
        append_line(prog, line)
        counter["bad"] = counter["bad"] or not finite
        saved = False
        for k in range(1, N_SNAPSHOTS + 1):   # a snapshot at the first chunk end past each tenth
            snap = snaps / f"{k}.pkl"
            if total >= budget * k // N_SNAPSHOTS and not snap.exists():
                save_pickle(snap, {"agent_steps": total, "policy": to_numpy(trainer.policy(st))})
                saved = True
        if saved or timer.elapsed() - counter["saved_at"] >= CHECKPOINT_SECONDS:
            checkpoint(st, total)
        return counter["bad"]

    state, steps = advance(trainer, state, steps, budget, on_chunk=record, max_chunks=max_chunks)
    if counter["chunks"] > chunks:
        checkpoint(state, steps)   # always save at a stop

    total = {"method": method, "seed": seed, "agent_steps": steps, "chunks": counter["chunks"],
             "wall_seconds": round(wall + timer.elapsed(), 2), "budget_agent_steps": budget,
             "reached_budget": steps >= budget}
    if counter["bad"]:
        total["reason"] = "parameters not finite"
        write_json(failed, total)
        return total
    if steps >= budget:
        save_pickle(out_dir / "policy.pkl", to_numpy(trainer.policy(state)))
        write_json(done, total)
    return total
