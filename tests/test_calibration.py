"""The calibration code runs and scores (uses test seeds, not calibration seeds)."""
import json
import os

import numpy as np

from evosim import calibration as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_score_runs_on_two_test_seeds():
    spec = json.load(open(os.path.join(ROOT, "calibration", "targets.json")))
    over = C.combo_overrides(0.5, 0.25, 50, False)
    runs = [C.run_seed((os.path.join(ROOT, "configs", "part1", "base.yaml"), over, s)) for s in (16100, 16101)]
    res = C.score(spec["targets"], runs)
    assert res["n_targets"] == len(spec["targets"])
    assert set(res["groups"]) == set(C.GROUPS)
    assert np.isfinite(res["total"]) and res["total"] >= 0
    pop = next(r for r in res["targets"] if r["id"] == "population_final")
    assert 0 <= pop["sim"] <= 200
    assert C.combo_id(0.5, "same", 0, True) == "sd0.5_reachsame_cap0_cann1"
