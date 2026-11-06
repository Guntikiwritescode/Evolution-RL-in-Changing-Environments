"""Check the Python port against the original Rust simulator (reference/s1ref).

The two use different random number generators, so single runs differ. The check
is statistical: run many seeds in both, then compare the per-generation averages.
If the rules match, differences should look like noise (|z| mostly below 2-3).

Usage: python scripts/part1/compare_reference.py [--seeds 200] [--gens 100] [--quick]
Writes results/part1/reference_check/<case>.json and a summary to stdout.
"""
import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, runner  # noqa: E402

BIN = os.path.join(ROOT, "reference", "s1ref", "target", "release", "s1ref")
STATS = ("population", "food_eaten", "prey_eaten", "survivors", "births", "ticks", "creature_steps",
         "mean_speed", "mean_size", "mean_sense", "mean_reach", "var_speed", "var_size", "var_sense")

# Cases: (name, trait_sd, reach_sd, cannibalism)
CASES = [
    ("s1_default", 0.5, 0.0, True),      # S1 exactly as shipped
    ("part1_base", 0.5, 0.25, False),    # our base config (cannibalism off, reach mutates)
    ("predation_stress", 2.0, 0.0, True),  # big mutations so predation actually happens
]


def rust_run(args):
    seed, gens, trait_sd, reach_sd, cannibalism = args
    cmd = [BIN, "--seed", str(seed), "--gens", str(gens), "--trait-sd", str(trait_sd),
           "--reach-sd", str(reach_sd)]
    if not cannibalism:
        cmd.append("--no-cannibalism")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    table = np.full((gens, len(STATS)), np.nan)
    table[:, :7] = 0.0  # generations after extinction: zero counts, undefined traits
    for line in out.strip().splitlines():
        r = json.loads(line)
        g = r["gen"]
        table[g] = [r["population"], r["food_balls_eaten"], r["prey_eaten"], r["survivors"],
                    r["births"], r["ticks"], r["creature_steps"],
                    r["mean"]["speed"], r["mean"]["size"], r["mean"]["sense"], r["mean"]["reach"],
                    r["var"]["speed"], r["var"]["size"], r["var"]["sense"]]
    return table


def python_run(args):
    seed, gens, trait_sd, reach_sd, cannibalism = args
    cfg = config.load(os.path.join(ROOT, "configs", "part1", "base.yaml"), {
        "world": {"cannibalism": cannibalism, "tick_cap": 0},
        "mutation_sd": {"speed": trait_sd, "size": trait_sd, "sense": trait_sd, "reach": reach_sd},
        "phases": {"train_generations": gens, "transfer_generations": 0}})
    log = runner.run_lineage(runner.et_start_traits(cfg), 50, cfg["mutation_sd"], cfg,
                             np.random.SeedSequence(seed))
    a = log.arrays()
    table = np.full((gens, len(STATS)), np.nan)
    for g in range(gens):
        table[g, 0] = a["g_population"][g]
        table[g, 1] = a["g_food_eaten"][g]
        table[g, 2] = a["g_prey_eaten"][g]
        table[g, 3] = a["g_survivors"][g]
        table[g, 4] = a["g_births"][g]
        table[g, 5] = a["g_ticks"][g]
        table[g, 6] = a["g_creature_steps"][g]
        sel = a["b_gen"] == g
        if sel.any():
            for j, t in enumerate(("speed", "size", "sense", "reach")):
                table[g, 7 + j] = a["b_" + t][sel].mean()
            for j, t in enumerate(("speed", "size", "sense")):
                table[g, 11 + j] = a["b_" + t][sel].var()
    return table


def compare(rust, py):
    """Per generation and statistic: z = (mean_py - mean_rust) / combined standard error."""
    out = {}
    for j, name in enumerate(STATS):
        r = rust[:, :, j]
        p = py[:, :, j]
        mr = np.nanmean(r, axis=0)
        mp = np.nanmean(p, axis=0)
        nr = np.sum(~np.isnan(r), axis=0)
        npn = np.sum(~np.isnan(p), axis=0)
        se = np.sqrt(np.nanvar(r, axis=0, ddof=1) / nr + np.nanvar(p, axis=0, ddof=1) / npn)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(se > 0, (mp - mr) / se, np.where(mp == mr, 0.0, np.inf))
        out[name] = {"rust_mean": mr.tolist(), "py_mean": mp.tolist(), "z": z.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=9000)  # validation range, see configs/part1/base.yaml
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.gens = 10, 10
    if not os.path.exists(BIN):
        sys.exit("build the reference first: bash reference/fetch_and_build.sh")
    outdir = os.path.join(ROOT, "results", "part1", "reference_check")
    os.makedirs(outdir, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    with Pool(args.procs) as pool:
        for name, tsd, rsd, cann in [c for c in CASES if c[0] in args.cases.split(",")]:
            jobs = [(s, args.gens, tsd, rsd, cann) for s in seeds]
            rust = np.stack(pool.map(rust_run, jobs))
            py = np.stack(pool.map(python_run, jobs))
            res = compare(rust, py)
            res["_meta"] = {"case": name, "trait_sd": tsd, "reach_sd": rsd, "cannibalism": cann,
                            "seeds": [seeds.start, seeds.stop - 1], "gens": args.gens,
                            "provenance": config.provenance({"case": name})}
            with open(os.path.join(outdir, f"{name}.json"), "w") as f:
                json.dump(res, f)
            np.savez_compressed(os.path.join(outdir, f"{name}_raw.npz"), rust=rust, python=py,
                                stats=np.array(STATS))
            print(f"\n== {name}: trait_sd={tsd} reach_sd={rsd} cannibalism={cann}, "
                  f"{args.seeds} seeds x {args.gens} gens")
            print(f"{'statistic':16s} {'rust(avg gens)':>14s} {'python':>10s} {'max|z|':>7s} {'|z|>3 cells':>11s}")
            for s in STATS:
                z = np.abs(np.array(res[s]["z"], dtype=float))
                z = z[np.isfinite(z)]
                print(f"{s:16s} {np.nanmean(res[s]['rust_mean']):14.3f} {np.nanmean(res[s]['py_mean']):10.3f} "
                      f"{(z.max() if z.size else 0):7.2f} {int((z > 3).sum()):>5d}/{args.gens}")


if __name__ == "__main__":
    main()
