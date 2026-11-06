"""Arm 3's map size (docs/PREREG_part2.md, Section 9.2), set before the pilot.

For each of seeds 2800-2829 (deployment streams of that seed): ET runs
through A once; from its end-of-A population it runs 50 generations in B
with each candidate side (400, 500, 600, 800, 1000). The chosen side is the
smallest where ET's mean P1 <= 20 and ET is extinct at the end of B in at
most 3 of the 30 runs. If none qualifies: stop and ask AA.

Writes results/part2/maprule/ (maprule.json with every run's outcomes,
raw logs in maprule.npz) and, if a side qualifies, configs/part2/arm3.yaml.

Usage: python scripts/part2/maprule.py [--quick]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, p2metrics, run2, sim2  # noqa: E402
from evosim.runner import Log, et_start_traits  # noqa: E402

BASE = os.path.join(ROOT, "configs", "part2", "base.yaml")


def one_seed(args):
    seed, cfg = args
    sd = {t: float(v) for t, v in cfg["mutation_sd"].items()}
    nc = int(cfg["part2"]["n_clones"])
    la = Log()
    end = sim2.deploy_A([et_start_traits(cfg)] * nc, cfg, seed, sd, la)
    out = {"seed": seed, "A_final_population": int(la.arrays()["g_population"][-1]), "sides": {}}
    arrays = {f"A__{k}": v for k, v in la.arrays().items()}
    for side in cfg["part2"]["arm3_candidates"]:
        env = dict(cfg["env_B"])
        env["size"] = float(side)
        lb = Log()
        sim2.deploy_B(end, env, cfg["world"], sd, int(cfg["phases"]["transfer_generations"]), cfg, seed, lb)
        o = p2metrics.outcomes(p2metrics.gen_arrays({f"B__{k}": v for k, v in lb.arrays().items()}, "B"))
        out["sides"][str(int(side))] = o
        arrays.update({f"B{int(side)}__{k}": v for k, v in lb.arrays().items()})
    return out, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2),
                    help="2 = sides 650/700/750 on seeds 2830-2859 (AA, 1 Nov 2026)")
    args = ap.parse_args()
    cfg = config.load(BASE)
    prov = config.provenance(cfg)
    p1_frozen = run2.frozen_hash(config.load(os.path.join(ROOT, "configs", "part1", "final.yaml")))
    if run2.frozen_hash(cfg) != p1_frozen and not args.quick:
        raise SystemExit("the frozen Part 1 sections differ from configs/part1/final.yaml (Section 2)")
    if prov["git"]["dirty"] and not args.quick:
        raise SystemExit("uncommitted changes: commit first")
    seed_key = "maprule" if args.round == 1 else "maprule_round2"
    cand_key = "arm3_candidates" if args.round == 1 else "arm3_candidates_round2"
    cfg["part2"]["arm3_candidates"] = list(cfg["part2"][cand_key])
    lo, hi = cfg["part2"]["seeds"][seed_key]
    seeds = list(range(int(lo), int(hi) + 1))
    if args.quick:
        seeds = seeds[:3]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(one_seed, [(s, cfg) for s in seeds])
    rule = cfg["part2"]["arm3_rule"]
    summary = {}
    chosen = None
    for side in cfg["part2"]["arm3_candidates"]:
        k = str(int(side))
        p1 = [r[0]["sides"][k]["P1"] for r in res]
        tot = sum(r[0]["sides"][k]["P1_total"] for r in res)
        ext = sum(r[0]["sides"][k]["P2"] for r in res)
        # exact: mean P1 <= 20  <=>  sum of totals <= 20 x 10 x n
        ok = tot <= int(round(float(rule["max_mean_p1"]) * 10)) * len(res) and ext <= int(rule["max_extinct"])
        summary[k] = {"mean_P1": float(np.mean(p1)), "sd_P1": float(np.std(p1, ddof=1)),
                      "extinct": int(ext), "n": len(p1), "qualifies": bool(ok)}
        if ok and chosen is None:
            chosen = float(side)
    outdir = os.path.join(ROOT, "results", "part2", "maprule" + ("" if args.round == 1 else "_round2")
                          + ("_quick" if args.quick else ""))
    os.makedirs(outdir, exist_ok=True)
    arrays = {}
    for r in res:
        arrays.update({f"seed{r[0]['seed']}__{k}": v for k, v in r[1].items()})
    np.savez_compressed(os.path.join(outdir, "maprule.tmp.npz"), **arrays)
    os.replace(os.path.join(outdir, "maprule.tmp.npz"), os.path.join(outdir, "maprule.npz"))
    doc = {"round": args.round, "seeds": seeds, "rule": rule, "summary": summary, "chosen_side": chosen,
           "runs": [r[0] for r in res], "provenance": prov}
    with open(os.path.join(outdir, "maprule.json.tmp"), "w") as f:
        json.dump(doc, f, indent=1, default=float)
    os.replace(os.path.join(outdir, "maprule.json.tmp"), os.path.join(outdir, "maprule.json"))
    for k, v in summary.items():
        print(f"side {k}: mean P1 {v['mean_P1']:.2f} (SD {v['sd_P1']:.2f}), extinct {v['extinct']}/{v['n']}"
              f"{'  <- qualifies' if v['qualifies'] else ''}")
    if chosen is None:
        raise SystemExit("no candidate side qualifies: stop and ask AA (Section 9.2)")
    print(f"chosen side: {chosen:g}")
    if not args.quick:
        path = os.path.join(ROOT, "configs", "part2", "arm3.yaml")
        with open(path, "w") as f:
            f.write(f"# Written by scripts/part2/maprule.py (round {args.round}) from "
                    f"{os.path.relpath(outdir, ROOT)}/maprule.json.\n"
                    "# Do not edit by hand. Arm 3's map side (PREREG_part2.md Section 9.2).\n"
                    f"extends: base.yaml\npart2:\n  arm3_side: {chosen:g}\n"
                    f"  arm3_rule_commit: {prov['git']['commit']}\n")
        print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
