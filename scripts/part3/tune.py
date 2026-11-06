"""Part 3 tuning: random search, the same for every method (PREREG Section 5).

For each method: TRIALS trials. Trial t of method m uses tuning seed 4000 + 100 * m + t
(m = 0 OE, 1 OE-fixed, 2 PPO-fixed, 3 PPO-DR, 4 ES-DR; 5 ES-fixed, Addendum A, PREREG Section 16). Its settings are drawn from the
declared ranges with the seed's "tuning_draw" stream; it trains for a quarter of the final
training budget; its score is the mean, over the 24 validation environments, of (competence -
random-policy competence), zero-shot, with the validation episode seeds (the same numbers
evaluate.py --split validation gives). The best score wins (ties: the lower trial number).
The test split is never read.

  venv3/bin/python scripts/part3/tune.py --methods oe --out results/part3/tuning
  venv3/bin/python scripts/part3/tune.py --write-tuned results/part3/tuning     # tuned.yaml from the summaries
  venv3/bin/python scripts/part3/tune.py --quick                                # tiny run on test seeds

Addendum A (PREREG Section 16): ES-fixed is tuned the same way, from the same untuned settings the
core methods started from, after tuned.yaml exists; its choice goes to its own file:
  venv3/bin/python scripts/part3/tune.py --methods es_fixed --out <out>
  venv3/bin/python scripts/part3/tune.py --write-addendum <out> --to <file>

Writes <out>/tuning_<method>.json per method (every trial's settings, training record, per-environment
scores) and the trial run folders <out>/<method>/trial<t>_seed<seed>/.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np
import yaml

from evo2.env.splits import load_validation
from evo2.eval.protocol import episode_keys, get_world, policy_competence
from evo2.eval.evaluate import random_competence_jit
from evo2.run import ADDENDUM_METHODS, ALL_METHODS, METHODS, TUNED_FILE, load_methods_config
from evo2.runio import enable_compile_cache, hardware, load_pickle, require_clean, write_json
from evo2.seeds import base_key, check_seed, stream_key
from evo2.train_loop import train_run

TRIALS = 8                 # Selected on 3 Nov 2026; the draft proposed 16
BUDGET_FRACTION = 0.25     # of the final training budget (PREREG Section 5)
METHOD_INDEX = {"oe": 0, "oe_fixed": 1, "ppo_fixed": 2, "ppo_dr": 3, "es_dr": 4, "es_fixed": 5}
FAMILY = {"oe": "oe", "oe_fixed": "oe", "ppo_fixed": "ppo", "ppo_dr": "ppo", "es_dr": "es", "es_fixed": "es"}
# declared in PREREG Section 5 before any tuning run
SPACES = {
    "ppo": [("LR", "loguniform", 1e-4, 1e-3), ("NUM_STEPS", "choice", [32, 64, 128]),
            ("UPDATE_EPOCHS", "choice", [2, 4, 8]), ("GAMMA", "choice", [0.95, 0.99, 0.995]),
            ("ENT_COEF", "choice", [0.0, 0.001, 0.01]), ("NORMALIZE_REWARD", "choice", [True, False])],
    "es": [("SIGMA", "loguniform", 0.01, 0.1), ("LR", "loguniform", 0.003, 0.03),
           ("POP_SIZE", "choice", [32, 64, 128]), ("EPISODES", "choice", [1, 2, 4])],
    "oe": [("WEIGHT_SD", "loguniform", 0.005, 0.05)],
}
QUICK = {"budget": 50_000, "trials": 2, "n_envs": 2, "episodes": 2}
DEFAULT_OUT = "results/part3/tuning"


def trial_seed(method: str, t: int, quick: bool = False) -> int:
    if quick:
        return 3950 + 8 * METHOD_INDEX[method] + t          # test seeds 3950-3991
    return 4000 + 100 * METHOD_INDEX[method] + t


def draw(method: str, seed: int, purpose: str = "tuning") -> dict:
    """One trial's settings, from the seed's tuning_draw stream (one key per searched setting)."""
    space = SPACES[FAMILY[method]]
    keys = jax.random.split(stream_key(base_key(purpose, seed), "tuning_draw"), len(space))
    out = {}
    for k, (name, kind, *args) in zip(keys, space):
        if kind == "loguniform":
            lo, hi = args
            out[name] = float(np.exp(float(jax.random.uniform(k, minval=np.log(lo), maxval=np.log(hi)))))
        else:
            opts = args[0]
            out[name] = opts[int(jax.random.randint(k, (), 0, len(opts)))]
    return out


def trial_config(mcfg: dict, method: str, settings: dict) -> dict:
    """The methods config for one trial: the drawn settings in the method's own settings block
    (OE and OE-fixed already have one; PPO and ES methods override their shared block)."""
    c = copy.deepcopy(mcfg)
    block = c.setdefault(method, {})
    block["settings"] = {**block.get("settings", {}), **settings}
    return c


def validation_envs():
    v = load_validation()
    envs = [("V", i, p) for i, p in enumerate(v["drawn"])] + [("C", i, p) for i, p in enumerate(v["corners"])]
    return envs, v["episode_seeds"]


def score(policy: dict, envs, ep_seeds, n_episodes: int) -> dict:
    """Mean over environments of (competence - random competence). Environment j uses validation
    episode seed ep_seeds[j] and the same key path as evo2.eval.protocol.zero_shot, so these are
    the numbers evaluate.py --split validation reports."""
    world = get_world("episodic")
    rows = []
    for j, (name, i, env) in enumerate(envs):
        k = stream_key(base_key("validation", ep_seeds[j]), "eval_episodes")
        k_c, _ = jax.random.split(k)                       # as in zero_shot
        c = policy_competence(world, k_c, env, policy, n_episodes)
        ep_keys, _ = episode_keys(k_c, n_episodes)
        r = float(random_competence_jit(world, ep_keys, env))
        rows.append({"set": name, "index": i, "competence": c, "random_competence": r})
    return {"score": float(np.mean([x["competence"] - x["random_competence"] for x in rows])), "envs": rows}


def best_trial(trials: list) -> dict:
    """Highest score; ties go to the lower trial number."""
    return max(trials, key=lambda tr: (tr["score"], -tr["trial"]))


def tune_method(method: str, mcfg: dict, out: Path, quick: bool) -> dict:
    purpose = "test" if quick else "tuning"
    budget = QUICK["budget"] if quick else int(BUDGET_FRACTION * int(mcfg["budget"]["train_agent_steps"]))
    n_trials = QUICK["trials"] if quick else TRIALS
    envs, ep_seeds = validation_envs()
    if quick:
        envs = envs[: QUICK["n_envs"]]
    n_episodes = QUICK["episodes"] if quick else int(mcfg["evaluation"]["episodes_per_env"])
    trials = []
    for t in range(n_trials):
        seed = check_seed(purpose, trial_seed(method, t, quick))
        settings = draw(method, seed, purpose)
        run_dir = out / method / f"trial{t}_seed{seed}"
        rec = train_run(method, purpose, seed, run_dir, budget, trial_config(mcfg, method, settings), quick=quick)
        row = {"trial": t, "seed": seed, "settings": settings, "train": rec}
        if (run_dir / "done.json").exists():
            row.update(score(load_pickle(run_dir / "policy.pkl"), envs, ep_seeds, n_episodes))
        else:                                              # a failed trial (parameters not finite) scores lowest
            row.update({"score": float("-inf"), "envs": [], "failed": True})
        trials.append(row)
        print(method, t, seed, json.dumps(settings), round(row["score"], 3), flush=True)
    chosen = best_trial(trials)
    if chosen.get("failed"):
        raise SystemExit(f"every {method} trial failed; stopping under PREREG Section 9")
    return {"method": method, "budget_agent_steps": budget, "trials": trials,
            "n_envs": len(envs), "episodes_per_env": n_episodes,
            "chosen": {"trial": chosen["trial"], "seed": chosen["seed"], "settings": chosen["settings"],
                       "score": chosen["score"]}}


def write_tuned(folder: Path, path: Path = TUNED_FILE) -> dict:
    """configs/part3/tuned.yaml from the tuning summaries (one per method, all five needed)."""
    settings, source = {}, {}
    for m in METHODS:
        s = json.loads((folder / f"tuning_{m}.json").read_text())
        if s.get("quick"):
            raise SystemExit("quick tuning results cannot set the final settings")
        settings[m] = s["chosen"]["settings"]
        source[m] = {"trial": s["chosen"]["trial"], "seed": s["chosen"]["seed"], "score": s["chosen"]["score"],
                     "commit": s["git"]["commit"]}
    doc = {"source": {"written_by": "scripts/part3/tune.py --write-tuned", "folder": str(folder), "per_method": source},
           "settings": settings}
    path.write_text("# Written by scripts/part3/tune.py --write-tuned. Do not edit by hand.\n"
                    + yaml.safe_dump(doc, sort_keys=True))
    return doc


def write_addendum(folder: Path, path: Path) -> dict:
    """The addendum's tuned settings (ADDENDUM_METHODS only) from FOLDER's summaries, written to PATH
    (outside configs/, so tuned.yaml and the core runs stay exactly as they were)."""
    settings, source = {}, {}
    for m in ADDENDUM_METHODS:
        s = json.loads((folder / f"tuning_{m}.json").read_text())
        if s.get("quick"):
            raise SystemExit("quick tuning results cannot set the final settings")
        if s["method"] != m or len(s["trials"]) != TRIALS:
            raise SystemExit(f"tuning_{m}.json is not a full {TRIALS}-trial tuning of {m}")
        settings[m] = s["chosen"]["settings"]
        source[m] = {"trial": s["chosen"]["trial"], "seed": s["chosen"]["seed"], "score": s["chosen"]["score"],
                     "commit": s["git"]["commit"]}
    doc = {"source": {"written_by": "scripts/part3/tune.py --write-addendum", "folder": str(folder),
                      "per_method": source}, "settings": settings}
    Path(path).write_text("# Written by scripts/part3/tune.py --write-addendum (PREREG Section 16). Do not edit by hand.\n"
                          + yaml.safe_dump(doc, sort_keys=True))
    return doc


def config_for(methods: list) -> dict:
    """The settings tuning starts from. Core methods: only before tuned.yaml exists.
    Addendum methods (alone): the untuned settings, read without tuned.yaml, which are the
    settings every core method started from."""
    if any(m in METHODS for m in methods):
        if any(m in ADDENDUM_METHODS for m in methods):
            raise SystemExit("tune core and addendum methods separately")
        if TUNED_FILE.exists():
            raise SystemExit("configs/part3/tuned.yaml exists: tuning starts from the untuned settings")
        return load_methods_config()
    return load_methods_config(tuned=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=ALL_METHODS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write-tuned", metavar="FOLDER", help="write configs/part3/tuned.yaml from FOLDER's summaries")
    ap.add_argument("--write-addendum", metavar="FOLDER", help="write the addendum's tuned settings (needs --to)")
    ap.add_argument("--to", metavar="FILE", help="where --write-addendum writes")
    a = ap.parse_args()
    if a.write_tuned:
        print(yaml.safe_dump(write_tuned(Path(a.write_tuned))))
        return
    if a.write_addendum:
        if not a.to:
            raise SystemExit("--write-addendum needs --to FILE")
        print(yaml.safe_dump(write_addendum(Path(a.write_addendum), Path(a.to))))
        return
    if not a.quick and any(m in ADDENDUM_METHODS for m in a.methods) and Path(a.out) == Path(DEFAULT_OUT):
        raise SystemExit("addendum tuning needs its own --out (the core tuning folder stays as it was)")
    enable_compile_cache()   # saves compile time only; see evo2.runio
    mcfg = config_for(a.methods)
    if a.quick:
        a.out = "results/part3/smoke_tuning"
    g = require_clean("test" if a.quick else "tuning", a.quick)
    out = Path(a.out)
    for m in a.methods:
        s = tune_method(m, mcfg, out, a.quick)
        s.update({"git": g, "hardware": hardware(), "quick": a.quick})
        write_json(out / f"tuning_{m}.json", s)


if __name__ == "__main__":
    main()
