"""Part 1: run the ET and GS conditions and save raw per-seed logs.

Usage:
  python scripts/part1/run_part1.py --config configs/part1/final.yaml --name final
  python scripts/part1/run_part1.py --quick          # smoke test: 2 seeds, 5+5 gens, 2x2x2 grid

Writes results/part1/<name>/:
  gs_search.json             every design's score and cost, the winner, provenance
  ET_seed<k>.npz, GS_seed<k>.npz   raw per-generation and per-blob logs + provenance
  run_info.json              seeds, timing, compute totals
Metrics and statistics are computed afterwards by scripts/part1/analyze_part1.py.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, gridsearch, runner, world  # noqa: E402

QUICK = {
    "phases": {"train_generations": 5, "transfer_generations": 5},
    "gs": {"grid_points": 2, "eval_generations": 2},
    "seeds": {"final": [9900, 9901]},  # validation-range seeds, never used for results
}


def _save(path, log, prov, extra):
    arrays = log.arrays()
    meta = dict(prov)
    meta.update(extra)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=float)), **arrays)


def _run_one(args):
    cond, seed, cfg, traits, n_start, sd, outdir, prov = args
    t0 = time.time()
    # Each condition gets its own stream from the run seed (spawn key = condition index).
    key = {"ET": 0, "GS": 1}[cond]
    ss = np.random.SeedSequence(seed, spawn_key=(key,))
    log = runner.run_lineage(traits, n_start, sd, cfg, ss)
    dt = time.time() - t0
    _save(os.path.join(outdir, f"{cond}_seed{seed:04d}.npz"), log, prov,
          {"condition": cond, "seed": seed, "start_traits": traits, "mutation_sd": sd,
           "wall_seconds": dt})
    a = log.arrays()
    return cond, seed, dt, int(a["g_creature_steps"].sum()), int(a["g_ticks"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "part1", "base.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--conditions", default="ET,GS")
    args = ap.parse_args()

    cfg = config.load(args.config, QUICK if args.quick else None)
    name = args.name or ("quick" if args.quick else "run")
    outdir = os.path.join(ROOT, "results", "part1", name)
    os.makedirs(outdir, exist_ok=True)
    prov = config.provenance(cfg)
    lo, hi = cfg["seeds"]["final"]
    seeds = list(range(lo, hi + 1))
    conds = args.conditions.split(",")
    t_start = time.time()

    with Pool(args.procs) as pool:
        gs_info = None
        if "GS" in conds:
            if cfg["gs"]["search"] != "once":
                raise NotImplementedError("only gs.search = once is implemented")
            t0 = time.time()
            search = gridsearch.grid_search(cfg, int(cfg["gs"]["search_seed"]), pool)
            gs_info = {"best": search["best"], "best_index": search["best_index"],
                       "n_ties_at_best": search["n_ties_at_best"],
                       "search_creature_steps": search["creature_steps"],
                       "search_seconds": time.time() - t0,
                       "designs": search["designs"], "results": search["results"],
                       "provenance": prov}
            with open(os.path.join(outdir, "gs_search.json"), "w") as f:
                json.dump(gs_info, f, indent=1, default=float)
            print(f"grid search: {len(search['designs'])} designs, best #{search['best_index']} "
                  f"{search['best']}, ties at best: {search['n_ties_at_best']}, "
                  f"{search['creature_steps']:,} creature-steps, {gs_info['search_seconds']:.1f}s")

        jobs = []
        zero_sd = {t: 0.0 for t in world.TRAITS}
        for seed in seeds:
            if "ET" in conds:
                jobs.append(("ET", seed, cfg, runner.et_start_traits(cfg), int(cfg["et"]["n_start"]),
                             cfg["mutation_sd"], outdir, prov))
            if "GS" in conds:
                jobs.append(("GS", seed, cfg, gs_info["best"], int(cfg["gs"]["n_clones"]),
                             zero_sd, outdir, prov))
        done = []
        for r in pool.imap_unordered(_run_one, jobs):
            done.append(r)
            print(f"  {r[0]} seed {r[1]}: {r[2]:.1f}s, {r[3]:,} creature-steps")

    info = {"name": name, "seeds": seeds, "conditions": conds, "wall_seconds": time.time() - t_start,
            "runs": [{"condition": c, "seed": s, "seconds": dt, "creature_steps": cs, "ticks": tk}
                     for c, s, dt, cs, tk in sorted(done)],
            "provenance": prov}
    for c in conds:
        info[f"{c}_creature_steps_total"] = int(sum(r[3] for r in done if r[0] == c))
    if gs_info:
        info["GS_creature_steps_total_incl_search"] = info["GS_creature_steps_total"] + gs_info["search_creature_steps"]
    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    print(f"done in {info['wall_seconds']:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
