"""Part 1 analysis: metrics per run and the Figure 11 statistics table, from saved logs.

Usage: python scripts/part1/analyze_part1.py --name quick [--defs configs/part1/metric_defs.yaml]

Every candidate metric definition is computed (see src/evosim/metrics.py). The
table uses the definitions named in the defs file. Until calibration picks them,
the spec's main choices are used (DEFAULT_DEFS below).
Writes results/part1/<name>/per_run_metrics.csv and summary.json.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import metrics, stats  # noqa: E402

# Spec main choices (docs/SPEC_part1.md Section 8). Calibration may replace these.
DEFAULT_DEFS = {
    "final_fitness": "final_fitness_last1",
    "final_median": "final_median_last1",
    "transfer_first10": "transfer_fit_first10",
    "transfer_last10": "transfer_fit_last10",
    "gap": "gap_last1_first10",
    "slope": "transfer_slope",
    "cv2": "cv2_4_samp_last1",
    "shannon": "H_exact4_last1",
    "energy_efficiency": "eff_E2_cost_per_dist_drop",
}
# Figure 11 rows in the paper's order; True = zero by design for GS clones.
FIG11 = [("Final training fitness (mean)", "final_fitness", False),
         ("Final training fitness (median)", "final_median", False),
         ("Transfer fitness (gen 1-10)", "transfer_first10", False),
         ("Transfer fitness (last 10 gen)", "transfer_last10", False),
         ("Generalization gap", "gap", False),
         ("Transfer trend slope", "slope", False),
         ("Trait diversity (CV2)", "cv2", True),
         ("Shannon diversity (H)", "shannon", True),
         ("Energy efficiency", "energy_efficiency", True)]


def load_runs(folder, cond):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, f"{cond}_seed*.npz"))):
        z = np.load(path)
        a = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
        out.append((meta, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--defs", default=None)
    args = ap.parse_args()
    folder = os.path.join(ROOT, "results", "part1", args.name)
    defs = dict(DEFAULT_DEFS)
    if args.defs:
        with open(args.defs) as f:
            defs.update({k: v for k, v in yaml.safe_load(f).items() if k in DEFAULT_DEFS})
        # the gap uses the same final-fitness window as the final-fitness row (M7 = M3 - M5)
        window = "last10" if defs["final_fitness"].endswith("last10") else "last1"
        defs["gap"] = f"gap_{window}_first10"

    per_cond = {}
    rows = []
    for cond in ("ET", "GS"):
        runs = load_runs(folder, cond)
        if not runs:
            continue
        scal = []
        for meta, a in runs:
            cfg = meta["config"]
            s = metrics.series(a)
            sc = metrics.run_scalars(s, int(cfg["phases"]["train_generations"]),
                                     int(cfg["phases"]["transfer_generations"]))
            sc = {"condition": cond, "seed": meta["seed"], **sc}
            scal.append(sc)
            rows.append(sc)
        per_cond[cond] = scal

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("condition", "seed"), k))
    with open(os.path.join(folder, "per_run_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {"definitions": defs, "table": [], "condition_means": {}}
    for cond, scal in per_cond.items():
        means = {}
        for k in scal[0]:
            if k in ("condition", "seed"):
                continue
            v = np.array([float(r[k]) for r in scal], dtype=float)
            means[k] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else None
        summary["condition_means"][cond] = means
    if "ET" in per_cond and "GS" in per_cond:
        print(f"{'metric':34s} {'ET (n)':>21s} {'GS (n)':>21s} {'t':>8s} {'df':>6s} {'p':>8s} {'p_bonf':>8s} {'d':>7s}")
        for label, key, zero_gs in FIG11:
            col = defs[key]
            x = [r.get(col, np.nan) for r in per_cond["ET"]]
            y = [r.get(col, np.nan) for r in per_cond["GS"]]
            res = stats.welch(x, y, zero_by_design=zero_gs)
            res.update(metric=label, definition=col)
            summary["table"].append(res)
            flag = "  (zero by design for GS: not meaningful)" if zero_gs else ""
            print(f"{label:34s} {res['mean_x']:7.3f}±{res['sd_x']:<7.3f}({res['n_x']:2d}) "
                  f"{res['mean_y']:7.3f}±{res['sd_y']:<7.3f}({res['n_y']:2d}) "
                  f"{res['t']:8.2f} {res['df']:6.1f} {res['p']:8.4f} {res['p_bonferroni']:8.4f} {res['d']:7.2f}{flag}")
    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"wrote {folder}/per_run_metrics.csv and summary.json")


if __name__ == "__main__":
    main()
