"""A-checks (PREREG_part2.md Section 4) for run seeds that did not get them in the run.

Section 4 says every run seed gets the A-checks, shown next to rows 5 and 6. The run
code (run2.run_seed at a1e1cdc) ran them only on exploratory seeds (2000-2099). This
script recomputes them for any seed, from scratch, on the same streams and with the same
calls as run_seed: ET's A phase, the clone pick, then the three A-check runs (MISTAKES #37).

  --verify   recompute seeds that already have A-checks and require every array to be
             bit-for-bit equal to the saved run (results/part2/<run>/seed*.npz)
  --compute  compute seeds that lack them; writes results/part2/<run>/achecks/seed<k>.npz
             plus achecks/run.json (provenance)

Usage: python scripts/part2/achecks_all.py --run final --verify --compute
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, sim2  # noqa: E402
from evosim.runner import Log, et_start_traits  # noqa: E402


def achecks_for_seed(seed, cfg):
    """The same steps as run2.run_seed: ET in A, the clone pick, the three A-checks."""
    p2 = cfg["part2"]
    sd = {t: float(v) for t, v in cfg["mutation_sd"].items()}
    nc = int(p2["n_clones"])
    n_a = int(cfg["phases"]["train_generations"])
    n_ac = int(p2["acheck_generations"])
    default = et_start_traits(cfg)
    log_a = Log()
    end_et = sim2.deploy_A([default] * nc, cfg, seed, sd, log_a)
    idx = sim2.pick_clone_index(end_et, seed, 0)
    clone_pop = sim2.clones_of(end_et, idx) if idx >= 0 else sim2.empty_like(end_et)
    arrays = {f"A__ET__{k}": v for k, v in log_a.arrays().items()}
    for name, pop0, sdg in (("mut", end_et, sd), ("frozen", end_et, sim2.ZERO_SD),
                            ("clone", clone_pop, sim2.ZERO_SD)):
        log = Log()
        sim2.run_streamed(copy.deepcopy(pop0), cfg["env_A"], cfg["world"], sdg, n_ac, seed, "acheck", (),
                          log, n_a, "A")
        arrays.update({f"ACHECK__{name}__{k}": v for k, v in log.arrays().items()})
    return arrays, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--compute", action="store_true")
    args = ap.parse_args()
    rundir = os.path.join(ROOT, "results", "part2", args.run)
    with open(os.path.join(rundir, "run.json")) as f:
        run = json.load(f)
    cfg = run["provenance"]["config"]
    prov = config.provenance(cfg)
    if prov["git"]["dirty"]:
        raise SystemExit("uncommitted changes: commit first")
    seeds = sorted(int(os.path.basename(p)[4:8]) for p in glob.glob(os.path.join(rundir, "seed*.done")))
    have, lack = [], []
    for s in seeds:
        with np.load(os.path.join(rundir, f"seed{s:04d}.npz")) as z:
            (have if "ACHECK__mut__g_gen" in z.files else lack).append(s)
    print(f"{len(seeds)} seeds: {len(have)} with A-checks in the run, {len(lack)} without")
    if args.verify:
        bad = []
        for s in have:
            arrays, idx = achecks_for_seed(s, cfg)
            with np.load(os.path.join(rundir, f"seed{s:04d}.npz")) as z:
                with open(os.path.join(rundir, f"seed{s:04d}.json")) as f:
                    meta = json.load(f)
                if idx != meta["clone_index"]:
                    bad.append((s, "clone_index"))
                for k, v in arrays.items():
                    if k not in z.files or z[k].dtype != v.dtype or z[k].shape != v.shape or z[k].tobytes() != v.tobytes():
                        bad.append((s, k))
                        break
        print(f"verify: {len(have)} seeds recomputed; {'all arrays bit-for-bit equal' if not bad else 'MISMATCH ' + str(bad[:5])}")
        if bad:
            raise SystemExit(1)
    if args.compute:
        out = os.path.join(rundir, "achecks")
        os.makedirs(out, exist_ok=True)
        for s in lack:
            arrays, idx = achecks_for_seed(s, cfg)
            arrays = {k: v for k, v in arrays.items() if k.startswith("ACHECK__")}
            path = os.path.join(out, f"seed{s:04d}.npz")
            np.savez_compressed(path + ".tmp.npz", **arrays)
            os.replace(path + ".tmp.npz", path)
        with open(os.path.join(out, "run.json"), "w") as f:
            json.dump({"seeds": lack, "why": "Section 4 A-checks for seeds the run skipped (MISTAKES #37)",
                       "verified_against_run_seeds": have if args.verify else [], "provenance": prov}, f, indent=1)
        print(f"computed A-checks for {len(lack)} seeds into {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
