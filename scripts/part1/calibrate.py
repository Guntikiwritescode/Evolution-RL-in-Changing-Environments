"""Part 1 calibration (PROJECT_PLAN.md Part 1 Step 4). Rules: calibration/targets.json.

  python scripts/part1/calibrate.py --stage 1a        # 288-setting grid
  python scripts/part1/calibrate.py --stage 1b        # refinement around the best 1a setting
  python scripts/part1/calibrate.py --stage 2         # metric definitions; writes the frozen config

Every setting run is appended to calibration/log.csv (rows are never deleted) and
its per-target results saved to calibration/runs/<id>__<stage>.json. A setting
already in the log for the same stage is skipped, so a restart does not rerun it.
"""
import argparse
import csv
import datetime
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import calibration as C  # noqa: E402
from evosim import config  # noqa: E402

BASE = os.path.join(ROOT, "configs", "part1", "base.yaml")
CAL = os.path.join(ROOT, "calibration")
LOG = os.path.join(CAL, "log.csv")
LOG_FIELDS = ["time_utc", "stage", "id", "trait_sd", "reach_sd", "tick_cap", "cannibalism", "seeds",
              "total_score", "score_fitness", "score_population", "score_survival", "score_traits",
              "score_diversity", "n_pass", "n_targets", "wall_seconds", "git_commit", "git_dirty", "targets_sha"]


def logged(stage):
    if not os.path.exists(LOG):
        return {}
    with open(LOG) as f:
        return {r["id"]: r for r in csv.DictReader(f) if r["stage"] == stage}


def all_logged():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return list(csv.DictReader(f))


def append_log(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_setting(pool, spec, stage, trait_sd, reach_sd, tick_cap, cann, seeds, targets_sha):
    cid = C.combo_id(trait_sd, reach_sd, tick_cap, cann)
    over = C.combo_overrides(trait_sd, reach_sd, tick_cap, cann)
    t0 = time.time()
    runs = pool.map(C.run_seed, [(BASE, over, s) for s in seeds])
    res = C.score(spec["targets"], runs)
    dt = time.time() - t0
    git = config.git_commit()
    os.makedirs(os.path.join(CAL, "runs"), exist_ok=True)
    with open(os.path.join(CAL, "runs", f"{cid}__{stage}.json"), "w") as f:
        json.dump({"id": cid, "stage": stage, "overrides": over, "seeds": list(seeds), "score": res,
                   "per_seed_scalars": [r["scalars"] for r in runs],
                   "mean_curves": {k: np.nanmean(np.stack([r["series"][k] for r in runs]), axis=0).tolist()
                                   for k in runs[0]["series"]}}, f, default=float)
    append_log({"time_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "stage": stage,
                "id": cid, "trait_sd": trait_sd, "reach_sd": reach_sd, "tick_cap": tick_cap, "cannibalism": cann,
                "seeds": f"{seeds[0]}-{seeds[-1]}", "total_score": res["total"],
                **{f"score_{g}": res["groups"][g] for g in C.GROUPS},
                "n_pass": res["n_pass"], "n_targets": res["n_targets"], "wall_seconds": round(dt, 1),
                "git_commit": git["commit"], "git_dirty": git["dirty"], "targets_sha": targets_sha})
    print(f"[{stage}] {cid}: score {res['total']:.3f} "
          + " ".join(f"{g[:4]}={res['groups'][g]:.2f}" for g in C.GROUPS)
          + f" pass {res['n_pass']}/{res['n_targets']} ({dt:.0f}s)", flush=True)
    return res


def best_row(rows):
    return min(rows, key=lambda r: float(r["total_score"]))  # min() keeps the first on exact ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["1a", "1b", "2"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    with open(os.path.join(CAL, "targets.json"), "rb") as f:
        raw = f.read()
    spec = json.loads(raw)
    import hashlib
    targets_sha = hashlib.sha256(raw).hexdigest()[:12]
    seeds = list(range(spec["seeds"][0], spec["seeds"][1] + 1))
    git = config.git_commit()
    if git["dirty"]:
        sys.exit("commit targets.json and the calibration code before calibrating (working tree is dirty)")

    if args.stage in ("1a", "1b"):
        done = logged(args.stage)
        if args.stage == "1a":
            s = spec["search"]["stage_1a"]
            grid = list(itertools.product(s["trait_sd (speed, size, sense together)"],
                                          [("same" if r == "same as trait_sd" else r) for r in s["reach_sd"]],
                                          s["tick_cap"], s["cannibalism"]))
        else:
            b = best_row([r for r in all_logged() if r["stage"] == "1a"])
            sd0, cap0 = float(b["trait_sd"]), int(b["tick_cap"])
            rsd = b["reach_sd"] if b["reach_sd"] == "same" else float(b["reach_sd"])
            cann = b["cannibalism"] == "True"
            sds = [round(sd0 + d, 4) for d in (-0.125, -0.0625, 0, 0.0625, 0.125) if sd0 + d >= 0.25]
            caps = [cap0] if cap0 == 0 else [cap0 + d for d in (-25, -12, 0, 12, 25) if cap0 + d >= 25]
            grid = [(sd, rsd, cap, cann) for sd in sds for cap in caps]
            print(f"stage 1b around {b['id']} (score {float(b['total_score']):.3f}): {len(grid)} settings")
        with Pool(args.procs) as pool:
            for sd, rsd, cap, cann in grid:
                cid = C.combo_id(sd, rsd, cap, cann)
                if cid in done:
                    continue
                run_setting(pool, spec, args.stage, sd, rsd, cap, cann, seeds, targets_sha)
        return

    # ---- stage 2: metric definitions for the selected setting
    rows = [r for r in all_logged() if r["stage"] in ("1a", "1b")]
    b = best_row(rows)
    with open(os.path.join(CAL, "runs", f"{b['id']}__{b['stage']}.json")) as f:
        best = json.load(f)
    sc = best["per_seed_scalars"]

    def mean(k):
        v = np.array([float(r[k]) for r in sc], dtype=float)
        return float(np.nanmean(v))

    choices = {
        "final_fitness": ([("final_fitness_last1"), ("final_fitness_last10")], 1.03, 0.03),
        "final_median": ([("final_median_last1"), ("final_median_last10")], 1.00, 0.10),
        "survival": ([("survival_all100"), ("survival_last10"), ("survival_last1")], 0.665, 0.0665),
        "cv2": ([f"cv2_{n}_{d}_{w}" for n in (3, 4) for d in ("pop", "samp") for w in ("last1", "trainmean")], 0.41, 0.29),
        "shannon": ([f"H_{t}_{w}" for t in ("exact4", "round1_4", "round0_4", "exact3", "round1_3", "round0_3")
                     for w in ("last1", "trainmean")], 3.77, 0.06),
        "energy_efficiency": (["eff_E1_cost_drop", "eff_E2_cost_per_dist_drop", "eff_E3_cost_per_dist_drop_trainmean",
                               "eff_E4_cv_cost_last1", "eff_E5_cv_cost_per_dist_last1"], 0.20, 0.18),
    }
    out = {"selected_setting": b["id"], "selected_from_stage": b["stage"], "score": float(b["total_score"]),
           "definitions": {}}
    for name, (cands, paper, tol) in choices.items():
        vals = {c: mean(c) for c in cands}
        pick = min(cands, key=lambda c: abs(vals[c] - paper))  # first wins exact ties
        out["definitions"][name] = {"chosen": pick, "paper": paper, "tolerance": tol,
                                    "candidates": {c: {"mean": v, "error_in_tol": (v - paper) / tol}
                                                   for c, v in vals.items()}}
        print(f"{name:18s} chosen {pick:38s} = {vals[pick]:.4f} (paper {paper}, tol {tol})")
    with open(os.path.join(CAL, "selected.json"), "w") as f:
        json.dump(out, f, indent=1)
    over = best["overrides"]
    final = {"extends": "base.yaml", "world": over["world"], "mutation_sd": over["mutation_sd"],
             "calibration": {"selected_setting": b["id"], "score": float(b["total_score"]),
                             "targets_sha": targets_sha, "frozen_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}}
    with open(os.path.join(ROOT, "configs", "part1", "final.yaml"), "w") as f:
        f.write("# Frozen Part 1 configuration, written by scripts/part1/calibrate.py --stage 2.\n"
                "# Do not edit by hand. Calibration details: calibration/selected.json and calibration/log.csv.\n")
        yaml.safe_dump(final, f, sort_keys=False)
    with open(os.path.join(ROOT, "configs", "part1", "metric_defs.yaml"), "w") as f:
        f.write("# Metric definitions chosen in calibration stage 2 (calibration/selected.json).\n")
        defs = {"final_fitness": out["definitions"]["final_fitness"]["chosen"],
                "final_median": out["definitions"]["final_median"]["chosen"],
                "cv2": out["definitions"]["cv2"]["chosen"], "shannon": out["definitions"]["shannon"]["chosen"],
                "energy_efficiency": out["definitions"]["energy_efficiency"]["chosen"],
                "survival": out["definitions"]["survival"]["chosen"]}
        yaml.safe_dump(defs, f, sort_keys=False)
    print("wrote calibration/selected.json, configs/part1/final.yaml, configs/part1/metric_defs.yaml")


if __name__ == "__main__":
    main()
