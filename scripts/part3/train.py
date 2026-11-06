"""Train one Part 3 run: one method, one seed, to the agent-step budget.

  venv3/bin/python scripts/part3/train.py --method ppo_dr --purpose pilot --seed 3000 --out results/part3/pilot
  venv3/bin/python scripts/part3/train.py --method oe --quick          # tiny smoke run on a test seed

Resumable: rerun the same command after a stop. The test split is never read here.

Addendum A (PREREG Section 16): an addendum method (ES-fixed) needs its tuned settings,
--tuned-extra FILE (written by tune.py --write-addendum), for any run that is not --quick.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo2.run import ADDENDUM_METHODS, ALL_METHODS, load_methods_config
from evo2.runio import enable_compile_cache
from evo2.seeds import check_seed
from evo2.train_loop import train_run

QUICK_BUDGET = 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--purpose", default="test", choices=["test", "pilot", "tuning", "final"])
    ap.add_argument("--seed", type=int, default=3900)
    ap.add_argument("--out", default="results/part3/smoke")
    ap.add_argument("--budget", type=int, default=None, help="agent-steps; default from methods.yaml")
    ap.add_argument("--quick", action="store_true", help=f"{QUICK_BUDGET} agent-steps on a test seed")
    ap.add_argument("--tuned-extra", default=None, help="the addendum's tuned settings (addendum methods only)")
    a = ap.parse_args()
    enable_compile_cache()   # saves compile time only; see evo2.runio
    addendum = a.method in ADDENDUM_METHODS
    if a.tuned_extra and not addendum:
        raise SystemExit(f"--tuned-extra is for the addendum methods {ADDENDUM_METHODS} only")
    if addendum and not a.tuned_extra and not a.quick:
        raise SystemExit(f"{a.method} runs need its tuned settings: --tuned-extra FILE (PREREG Section 16)")
    mcfg = load_methods_config(extra_tuned=a.tuned_extra) if a.tuned_extra else load_methods_config()
    if a.purpose == "final" and "tuned_from" not in mcfg:
        raise SystemExit("final runs use the tuned settings: configs/part3/tuned.yaml is missing")
    if addendum and a.tuned_extra and "settings" not in mcfg.get(a.method, {}):
        raise SystemExit(f"{a.tuned_extra} has no settings for {a.method}")
    if a.quick:
        if a.purpose != "test":
            raise SystemExit("--quick runs use test seeds only")
        budget = QUICK_BUDGET
    else:
        budget = a.budget if a.budget is not None else int(mcfg["budget"]["train_agent_steps"])
    check_seed(a.purpose, a.seed)
    out = Path(a.out) / f"{a.method}_seed{a.seed}"
    print(json.dumps(train_run(a.method, a.purpose, a.seed, out, budget, mcfg, quick=a.quick), indent=1))


if __name__ == "__main__":
    main()
