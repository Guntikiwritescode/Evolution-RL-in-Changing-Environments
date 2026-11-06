"""R2 check for calibration (calibration/PRELAUNCH.md, stage 1b): how much does the
calibration score move with the seed set? Reruns a few top settings on two extra
seed sets. Report only: selection stays on seeds 1000-1029 as written in targets.json."""
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import calibration as C  # noqa: E402

BASE = os.path.join(ROOT, "configs", "part1", "base.yaml")
SETTINGS = [(1.0, 0.25, 200, False), (1.0, 0.25, 0, False), (1.0625, 0.25, 175, False),
            (0.875, 0.25, 225, False), (1.0, 0.25, 175, False), (1.0, 0.25, 212, False)]
SEED_SETS = {"calib_1000": range(1000, 1030), "extra_17000": range(17000, 17030), "extra_17030": range(17030, 17060)}

if __name__ == "__main__":
    spec = json.load(open(os.path.join(ROOT, "calibration", "targets.json")))
    out = []
    with Pool(2) as pool:
        for sd, rsd, cap, cann in SETTINGS:
            row = {"id": C.combo_id(sd, rsd, cap, cann)}
            for name, seeds in SEED_SETS.items():
                runs = pool.map(C.run_seed, [(BASE, C.combo_overrides(sd, rsd, cap, cann), s) for s in seeds])
                res = C.score(spec["targets"], runs)
                row[name] = {"total": res["total"], "n_pass": res["n_pass"], "groups": res["groups"]}
            out.append(row)
            print(row["id"], " ".join(f"{k}: {v['total']:.3f} ({v['n_pass']}/76)" for k, v in row.items() if k != "id"), flush=True)
    with open(os.path.join(ROOT, "calibration", "score_noise_check.json"), "w") as f:
        json.dump(out, f, indent=1)
