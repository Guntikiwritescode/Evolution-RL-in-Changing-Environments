"""Part 3 pilot: throughput, repeatability, room to move, variance, power.

For each method, on pilot seeds (3000-3099):
  1. train to a short budget (and train the first seed twice: repeatability);
  2. zero-shot competence on validation environments (never the test split);
  3. the adaptation protocol on the pilot stand-in shifted environments
     (configs/part3/pilot_shift.yaml: T2 and T3 ranges, new environments with
     pilot-evaluation seeds), to estimate the run-to-run spread of the primary
     measure for the power analysis.
Writes pilot_summary.json in --out. Nothing here chooses method settings.

  venv3/bin/python scripts/part3/pilot.py --out results/part3/pilot --budget 20000000 --adapt-budget 2000000
  venv3/bin/python scripts/part3/pilot.py --quick     # tiny version on test seeds, any machine

Split across machines: run each with its own --methods and --out, then
  venv3/bin/python scripts/part3/pilot.py --merge DIR1 DIR2 ... --out results/part3/pilot
recomputes the normalized AUC (whose "best observed" spans every method) and
the power estimate over all parts. A part's own summary covers only its methods.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np

from evo2 import metrics
from evo2.env.splits import load_pilot_shift, load_validation
from evo2.eval.protocol import adaptation, zero_shot
from evo2.run import METHODS, load_methods_config
from evo2.runio import enable_compile_cache, hardware, load_pickle, require_clean, write_json
from evo2.seeds import base_key, check_seed, stream_key
from evo2.train_loop import train_run


def leaves_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def runs_needed(sd_x: float, sd_y: float, half_width: float, minimum: int = 10) -> int:
    """Runs per method for an expected 95% CI half-width of a difference of
    per-run means: 1.96 * sqrt((sd_x^2 + sd_y^2) / n) <= half_width. A normal
    approximation; with 3 pilot seeds per method the SDs are rough."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    n = math.ceil((1.96 ** 2) * (sd_x ** 2 + sd_y ** 2) / half_width ** 2)
    return max(n, minimum)


def add_power(summary: dict) -> dict:
    """Normalized adaptation AUC per run and stand-in environment (best = the
    highest competence seen in the pilot, over every method in `summary`), and
    the power estimate (OE vs PPO-DR). Run again on merged summaries when the
    pilot is split across machines (--merge), so "best" covers every method."""
    rows0 = next(iter(summary["methods"].values()))["runs"][0]["shift"]
    n_env = len(rows0)
    best = np.full(n_env, -np.inf)
    for m in summary["methods"].values():
        for r in m["runs"]:
            if len(r["shift"]) != n_env:
                raise ValueError("runs have different numbers of stand-in environments")
            for j, row in enumerate(r["shift"]):
                best[j] = max(best[j], max(p["competence"] for p in row["adaptation"]["curve"]))
    x = np.linspace(0.0, 1.0, 11)
    auc = {}
    for name, m in summary["methods"].items():
        per_run = []
        for r in m["runs"]:
            vals = []
            for j, row in enumerate(r["shift"]):
                rnd = row["zero_shot"]["random_competence"]
                y = [p["competence"] for p in row["adaptation"]["curve"]]
                vals.append(float(metrics.curve_auc(x, metrics.normalize(y, rnd, best[j])))
                            if best[j] > rnd else float("nan"))
            per_run.append(vals)
        auc[name] = per_run
        m["normalized_auc"] = per_run
    summary["normalized_over_methods"] = sorted(summary["methods"])
    summary.pop("power", None)
    if "oe" in auc and "ppo_dr" in auc:
        sd = {k: float(np.nanstd(np.nanmean(np.array(auc[k]), axis=1), ddof=1)) for k in ("oe", "ppo_dr")}
        summary["power"] = {
            "per_run_sd": sd,
            "note": "normal approximation from per-run mean AUC; very rough with few pilot seeds",
            "runs_needed": {str(h): runs_needed(sd["oe"], sd["ppo_dr"], h) for h in (0.025, 0.05)},
        }
    return summary


SAME_IN_EVERY_PART = ("budget", "adapt_budget", "seeds", "evaluation_settings", "n_val_envs", "n_shift_envs")


def merge(parts: list[dict]) -> dict:
    """One pilot summary from parts run on different machines (each with
    different methods). The parts must share settings and code commit."""
    first = parts[0]
    for p in parts[1:]:
        for k in SAME_IN_EVERY_PART:
            if p.get(k) != first.get(k):
                raise ValueError(f"parts differ in {k}")
        if p["git"]["commit"] != first["git"]["commit"]:
            raise ValueError("parts ran different code commits")
    methods = {}
    for p in parts:
        for m, v in p["methods"].items():
            if m in methods:
                raise ValueError(f"method {m} appears in more than one part")
            methods[m] = v
    merged = {k: first[k] for k in SAME_IN_EVERY_PART}
    merged.update({"git": first["git"], "methods": methods,
                   "parts": [{"git": p["git"], "hardware": p["hardware"], "methods": sorted(p["methods"])}
                             for p in parts]})
    return add_power(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part3/pilot")
    ap.add_argument("--budget", type=int, default=20_000_000)
    ap.add_argument("--adapt-budget", type=int, default=2_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--n-val-envs", type=int, default=4)
    ap.add_argument("--n-shift-envs", type=int, default=5, help="per stand-in set (P2, P3)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", nargs="+", metavar="DIR",
                    help="merge the pilot_summary.json of these folders into --out (no runs)")
    a = ap.parse_args()
    if a.merge:
        parts = [json.loads((Path(d) / "pilot_summary.json").read_text()) for d in a.merge]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        write_json(Path(a.out) / "pilot_summary.json", merge(parts))
        return
    enable_compile_cache()   # saves compile time only; see evo2.runio
    purpose = "test" if a.quick else "pilot"
    if a.quick:
        a.budget, a.adapt_budget, a.seeds = 50_000, 140_000, [3990, 3991]
        a.n_val_envs, a.n_shift_envs, a.out = 1, 1, "results/part3/smoke_pilot"
    for s in a.seeds:
        check_seed(purpose, s)
    g = require_clean(purpose, a.quick)
    mcfg = load_methods_config()
    ecfg = dict(mcfg["evaluation"])
    if a.quick:
        ecfg["persistence_steps"] = 200
        ecfg["episodes_per_env"] = 2
    val = load_validation()
    shift = load_pilot_shift()
    shift_envs = [(n, i, p) for n in ("P2", "P3") for i, p in enumerate(shift[n][: a.n_shift_envs])]
    out = Path(a.out)
    summary = {"git": g, "hardware": hardware(), "budget": a.budget, "adapt_budget": a.adapt_budget,
               "seeds": a.seeds, "evaluation_settings": ecfg, "n_val_envs": a.n_val_envs,
               "n_shift_envs": a.n_shift_envs, "methods": {}}
    for m in a.methods:
        rows = []
        for s in a.seeds:
            run_dir = out / f"{m}_seed{s}"
            r = train_run(m, purpose, s, run_dir, a.budget, mcfg, quick=a.quick)
            pol = load_pickle(run_dir / "policy.pkl")
            val_rows = []
            for j, env in enumerate(val["drawn"][: a.n_val_envs]):
                k = stream_key(base_key("validation", val["episode_seeds"][j]), "eval_episodes")
                val_rows.append(zero_shot(k, env, pol, ecfg))
            shift_rows = []
            for j, (name, i, env) in enumerate(shift_envs):
                k = stream_key(base_key("pilot_eval", shift["episode_seeds"][j]), "eval_episodes")
                zs = zero_shot(k, env, pol, ecfg)
                ad = adaptation(jax.random.fold_in(k, 1),
                                jax.random.fold_in(stream_key(base_key(purpose, s), "adapt_run"), j),
                                m, mcfg, ecfg, env, pol, a.adapt_budget, 10, zs["competence"])
                shift_rows.append({"set": name, "index": i, "zero_shot": zs, "adaptation": ad})
            rows.append({"seed": s, "train": r,
                         "agent_steps_per_second": r["agent_steps"] / max(r["wall_seconds"], 1e-9),
                         "validation": val_rows, "shift": shift_rows})
            print(m, s, round(rows[-1]["agent_steps_per_second"]),
                  [round(v["competence"], 2) for v in val_rows], flush=True)
        rep_dir = out / f"{m}_seed{a.seeds[0]}_repeat"
        train_run(m, purpose, a.seeds[0], rep_dir, a.budget, mcfg, quick=a.quick)
        same = leaves_equal(load_pickle(out / f"{m}_seed{a.seeds[0]}" / "policy.pkl"),
                            load_pickle(rep_dir / "policy.pkl"))
        summary["methods"][m] = {"runs": rows, "repeat_bit_identical": same}

    add_power(summary)
    write_json(out / "pilot_summary.json", summary)


if __name__ == "__main__":
    main()
