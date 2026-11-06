"""Evaluate trained Part 3 runs: zero-shot competence and persistence, then adaptation.

  venv3/bin/python scripts/part3/evaluate.py --runs results/part3/pilot/ppo_dr_seed3000 --split validation
  venv3/bin/python scripts/part3/evaluate.py --runs ... --split test --final      # final evaluation only

The test split is locked: it loads only with --split test AND --final
(evo2.env.splits.load_test_split(final=True)). Validation uses the validation
environments and episode seeds 4930-4999.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from evo2.env.generator import params_to_dict
from evo2.env.splits import load_test_split, load_validation
from evo2.eval.protocol import adaptation, zero_shot
from evo2.run import load_methods_config
from evo2.runio import ROOT, append_line, enable_compile_cache, hardware, load_pickle, require_clean, write_json
from evo2.seeds import base_key, stream_key


def environments(split: str, final: bool):
    if split == "validation":
        v = load_validation()
        envs = [("V", i, p) for i, p in enumerate(v["drawn"])] + [("C", i, p) for i, p in enumerate(v["corners"])]
        return envs, v["episode_seeds"], "validation"
    if split == "test":
        t = load_test_split(final=final)
        envs = [(name, i, p) for name in ("T1", "T2", "T3") for i, p in enumerate(t[name])]
        return envs, t["episode_seeds"], "test_env"
    raise SystemExit(f"unknown split {split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with done.json and policy.pkl")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true", help="required to load the locked test split")
    ap.add_argument("--adapt", action="store_true", help="also run the adaptation protocol")
    ap.add_argument("--adapt-budget", type=int, default=None)
    ap.add_argument("--max-envs", type=int, default=None, help="evaluate only the first N environments (smoke)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    enable_compile_cache()   # saves compile time only; see evo2.runio
    if a.split == "test" and not a.final:
        raise SystemExit("the test split needs --final")
    if a.final and (a.max_envs or a.quick or a.split != "test" or not a.adapt or a.adapt_budget is not None):
        raise SystemExit("--final evaluates the whole test split with the full adaptation (--adapt, the "
                         "configured budget), and nothing else")
    metas = {}
    for run in a.runs:
        run = Path(run)
        if not (run / "done.json").exists():
            raise SystemExit(f"{run} has not finished")
        metas[run] = json.loads((run / "run.json").read_text())
        if a.final and metas[run]["purpose"] != "final":
            raise SystemExit(f"{run} is a {metas[run]['purpose']} run; only final runs meet the test split")
    g = require_clean("final" if a.final else ("test" if a.quick else "pilot"), a.quick and not a.final)
    ecfg = load_methods_config()["evaluation"]   # the evaluation settings are the same for every run
    if a.final:
        # an append-only record of every look at the test split
        append_line(ROOT / "results/part3/test_access_log.jsonl",
                    {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": g["commit"],
                     "runs": [str(r) for r in a.runs], "adapt": a.adapt})
    envs, ep_seeds, purpose = environments(a.split, a.final)
    if a.max_envs:
        envs = envs[: a.max_envs]
    for run, meta in metas.items():
        mcfg = meta["methods_config"]          # the run's own method settings
        budget = a.adapt_budget or int(mcfg["budget"]["adapt_agent_steps"])
        policy = load_pickle(run / "policy.pkl")
        out = {"run": str(run), "split": a.split, "git": g, "hardware": hardware(),
               "evaluation_settings": ecfg, "envs": []}
        run_base = base_key(meta["purpose"], meta["seed"])
        for j, (name, i, env) in enumerate(envs):
            # environment j uses episode seed ep_seeds[j]: the same episodes for every method and run
            k = base_key(purpose, ep_seeds[j])
            zs = zero_shot(stream_key(k, "eval_episodes"), env, policy, ecfg)
            row = {"set": name, "index": i, "env": params_to_dict(env), "zero_shot": zs}
            if a.adapt:
                row["adaptation"] = adaptation(
                    jax.random.fold_in(stream_key(k, "eval_episodes"), 1),
                    jax.random.fold_in(stream_key(run_base, "adapt_run"), j),
                    meta["method"], mcfg, ecfg, env, policy, budget,
                    int(mcfg["budget"]["adapt_checkpoints"]), zs["competence"])
            out["envs"].append(row)
            print(name, i, json.dumps(zs), flush=True)
        write_json(run / f"eval_{a.split}.json", out)


if __name__ == "__main__":
    main()
