"""Part 2 runs (docs/PREREG_part2.md). One file set per run seed.

Usage:
  python scripts/part2/run_part2.py --config configs/part2/<cfg>.yaml --name pilot --seeds pilot --all-exploratory
  python scripts/part2/run_part2.py --config configs/part2/<cfg>.yaml --name final --seeds final_100
  python scripts/part2/run_part2.py --quick --name quick --seeds smoke   # tiny settings, for testing

Writes results/part2/<name>/:
  seed<k>.npz   every raw log of that seed (search scores, A and B logs, A-checks, landscape)
  seed<k>.json  meta: winners, ties, U2 values, steps, process times, provenance
  seed<k>.done  written last: {"commit", "config_hash"}; a restart skips a seed only if
                this file exists and matches the current commit and config hash
  run.json      the run's settings and provenance

Rules (Section 18.8): refuses to start with uncommitted changes (except --quick);
writes each file to a temporary name and renames it; stops at the first error.
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, run2  # noqa: E402

QUICK = {
    "phases": {"train_generations": 12, "transfer_generations": 50},
    "part2": {"grid": {"points": 3}, "search": {"stage2_sets": 2, "stage2_top": 3},
              "long_B_generations": 60, "acheck_generations": 5, "suite_sides": [500],
              "suite_food": [15], "mutation_factors": [2.0]},
}


def seed_list(cfg, spec):
    rng = cfg["part2"]["seeds"].get(spec)
    if rng is not None:
        lo, hi = (rng, rng) if isinstance(rng, int) else rng
        return list(range(int(lo), int(hi) + 1))
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _write_atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def _done_ok(outdir, seed, commit, chash):
    p = os.path.join(outdir, f"seed{seed:04d}.done")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        d = json.load(f)
    if d.get("commit") != commit or d.get("config_hash") != chash:
        raise SystemExit(f"seed {seed}: finished file is from commit {d.get('commit')} / config "
                         f"{d.get('config_hash')}, not {commit} / {chash}; refusing to mix")
    # the result files must load
    np.load(os.path.join(outdir, f"seed{seed:04d}.npz"), allow_pickle=False).files
    with open(os.path.join(outdir, f"seed{seed:04d}.json")) as f:
        json.load(f)
    return True


def _one(args):
    seed, cfg, outdir, prov, exploratory, landscape = args
    try:
        meta, arrays = run2.run_seed(seed, cfg, exploratory=exploratory, landscape=landscape)
        meta["provenance"] = {k: prov[k] for k in ("config_hash", "git", "versions", "hardware")}
        base = os.path.join(outdir, f"seed{seed:04d}")
        np.savez_compressed(base + ".tmp.npz", **arrays)
        os.replace(base + ".tmp.npz", base + ".npz")
        _write_atomic_json(base + ".json", meta)
        _write_atomic_json(base + ".done", {"commit": prov["git"]["commit"],
                                            "config_hash": prov["config_hash"]})
        return seed, None, meta["process_seconds_conf"], meta["process_seconds_expl"]
    except Exception:
        return seed, traceback.format_exc(), 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "part2", "base.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", required=True, help="a range name from the config, 'a-b', or 'a,b,c'")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all-exploratory", action="store_true",
                    help="run the exploratory work on every seed (the pilot)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    over = dict(QUICK) if args.quick else None
    cfg = config.load(args.config, over)
    if args.quick and cfg["part2"].get("arm3_side") is None:
        cfg["part2"]["arm3_side"] = 400.0
    if args.quick and cfg["part2"].get("arm4_energy") is None:
        cfg["part2"]["arm4_energy"] = 25.0  # quick runs exercise arm 4 too
    prov = config.provenance(cfg)
    p1_frozen = run2.frozen_hash(config.load(os.path.join(ROOT, "configs", "part1", "final.yaml")))
    if run2.frozen_hash(cfg) != p1_frozen and not args.quick:
        raise SystemExit("the frozen Part 1 sections differ from configs/part1/final.yaml (Section 2)")
    if prov["git"]["dirty"] and not args.quick:
        raise SystemExit("uncommitted changes: commit first (Section 18.8)")
    seeds = seed_list(cfg, args.seeds)
    outdir = os.path.join(ROOT, "results", "part2", args.name)
    os.makedirs(outdir, exist_ok=True)
    p2s = cfg["part2"]["seeds"]
    lands = set(seed_list(cfg, "pilot_landscape"))

    def is_expl(s):
        return True if args.all_exploratory else run2.in_range(s, p2s["exploratory"])

    def is_land(s):
        return (s in lands) if args.all_exploratory else run2.in_range(s, p2s["landscape"])

    commit, chash = prov["git"]["commit"], prov["config_hash"]
    todo = [s for s in seeds if not _done_ok(outdir, s, commit, chash)]
    _write_atomic_json(os.path.join(outdir, "run.json"),
                       {"name": args.name, "config_path": os.path.relpath(args.config, ROOT),
                        "seeds": seeds, "quick": args.quick, "all_exploratory": args.all_exploratory,
                        "procs": args.procs, "provenance": prov,
                        "frozen_hash": run2.frozen_hash(cfg)})
    print(f"{len(seeds)} seeds, {len(seeds) - len(todo)} already done, {len(todo)} to run "
          f"(commit {commit[:7]}, config {chash[:12]})", flush=True)
    t0 = time.time()
    jobs = [(s, cfg, outdir, prov, is_expl(s), is_land(s)) for s in todo]
    if args.procs <= 1:
        results = map(_one, jobs)
        pool = None
    else:
        pool = Pool(args.procs)
        results = pool.imap_unordered(_one, jobs)
    done = 0
    try:
        for seed, err, tc, te in results:
            if err:
                with open(os.path.join(outdir, f"seed{seed:04d}.error"), "w") as f:
                    f.write(err)
                raise SystemExit(f"seed {seed} failed; stopping (traceback in seed{seed:04d}.error)\n{err}")
            done += 1
            print(f"seed {seed} done ({done}/{len(todo)}); process seconds: confirmatory {tc:.0f}, "
                  f"exploratory {te:.0f}; elapsed {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
    print(f"all {len(seeds)} seeds done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
