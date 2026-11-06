"""Tests for the Part 3 run scripts: resuming, budgets and the test-split lock.

Run with: venv3/bin/python -m pytest tests/test_part3_scripts.py
Seeds used here: 3900-3999 (the "test" range).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from evo2.run import load_methods_config
from evo2.runio import load_pickle
from evo2.train_loop import train_run

ROOT = Path(__file__).resolve().parents[1]


def small_cfg():
    m = copy.deepcopy(load_methods_config(tuned=None))   # small shared settings; tuned per-method ones would override them
    m["ppo"]["n_envs"] = 2
    m["ppo"]["settings"].update({"NUM_STEPS": 8, "NUM_MINIBATCHES": 4, "UPDATE_EPOCHS": 1})
    m["ppo"]["chunk_updates"] = 2
    m["es"]["settings"].update({"POP_SIZE": 4, "EPISODES": 1})
    m["es"]["chunk_generations"] = 2
    m["oe"]["chunk_steps"] = 50
    return m


def leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


@pytest.mark.parametrize("method", ["ppo_dr", "es_dr", "oe"])
def test_resume_gives_the_same_result(tmp_path, method):
    m = small_cfg()
    per_unit = {"ppo_dr": 8 * 2 * 16, "es_dr": 4 * 16 * 500, "oe": 128 * 50}[method]
    budget = 6 * per_unit
    a, b = tmp_path / "a", tmp_path / "b"
    first = train_run(method, "test", 3960, a, budget, m, quick=True, max_chunks=1)
    assert not first["reached_budget"] and not (a / "done.json").exists()
    train_run(method, "test", 3960, a, budget, m, quick=True, max_chunks=3)   # stop again, nearer the end
    train_run(method, "test", 3960, a, budget, m, quick=True)
    train_run(method, "test", 3960, b, budget, m, quick=True)
    assert (a / "done.json").exists() and (b / "done.json").exists()
    assert leaves_equal(load_pickle(a / "policy.pkl"), load_pickle(b / "policy.pkl"))
    da, db = json.loads((a / "done.json").read_text()), json.loads((b / "done.json").read_text())
    assert da["agent_steps"] == db["agent_steps"]


class FakeTrainer:
    """Units whose agent-steps follow a rule, to test the budget logic exactly."""

    def __init__(self, unit_max, chunk_units, per_unit):
        self.unit_max_agent_steps, self.chunk_units, self.per_unit = unit_max, chunk_units, per_unit

    def chunk(self, state, n=None):
        n = n or self.chunk_units
        steps = sum(self.per_unit(state + i) for i in range(n))
        return state + n, {"agent_steps": steps, "updates": n}


@pytest.mark.parametrize("rule", ["full", "growing", "shrinking"])
def test_budget_logic_never_passes_target_by_a_unit(rule):
    from evo2.run import advance
    u = 128
    per = {"full": lambda i: u, "growing": lambda i: min(u, 20 + 3 * i),
           "shrinking": lambda i: max(1, u - 2 * i)}[rule]
    for target in (1, u - 1, u, 4 * u + 1, 37 * u + 5, 1000 * u + 3):
        t = FakeTrainer(u, 16, per)
        _, steps = advance(t, 0, 0, target)
        assert target <= steps < target + u, (rule, target, steps)


def test_resumed_budget_path_matches_uninterrupted():
    """Stopping after any number of chunks (including just before the end) and
    resuming gives the same chunks and the same end as an uninterrupted run."""
    from evo2.run import advance
    per = lambda i: min(128, 20 + 3 * i)   # a growing population, like OE
    target = 300 * 128 + 17
    sizes = []
    full_state, full_steps = advance(FakeTrainer(128, 16, per), 0, 0, target,
                                     on_chunk=lambda st, total, summary: sizes.append(summary["updates"]))
    assert len(sizes) > 5 and sizes[-1] < 16      # the end of the run uses smaller chunks
    for stop_after in range(1, len(sizes)):
        st, steps = advance(FakeTrainer(128, 16, per), 0, 0, target, max_chunks=stop_after)
        rest = []
        st, steps = advance(FakeTrainer(128, 16, per), st, steps, target,
                            on_chunk=lambda s_, total, summary: rest.append(summary["updates"]))
        assert (st, steps) == (full_state, full_steps)
        assert sizes[stop_after:] == rest


@pytest.mark.parametrize("method", ["ppo_dr", "es_dr", "oe"])
def test_real_runs_stop_within_one_unit(tmp_path, method):
    from evo2.run import make_trainer
    m = small_cfg()
    unit = make_trainer(method, m).unit_max_agent_steps
    budget = 4 * unit + 1
    out = train_run(method, "test", 3961, tmp_path / method, budget, m, quick=True)
    assert budget <= out["agent_steps"] < budget + unit
    snaps = sorted((tmp_path / method / "snapshots").glob("*.pkl"))
    assert len(snaps) == 10


def test_run_folder_refuses_another_config(tmp_path):
    m = small_cfg()
    unit = 8 * 2 * 16
    train_run("ppo_fixed", "test", 3962, tmp_path / "half", 4 * unit, m, quick=True, max_chunks=1)
    train_run("ppo_fixed", "test", 3962, tmp_path / "done", unit, m, quick=True)
    assert (tmp_path / "done" / "done.json").exists()
    m2 = copy.deepcopy(m)
    m2["ppo"]["settings"]["LR"] = 0.123
    for folder, budget in (("half", 4 * unit), ("done", unit)):   # unfinished and finished folders
        with pytest.raises(SystemExit):
            train_run("ppo_fixed", "test", 3962, tmp_path / folder, budget, m2, quick=True)
    with pytest.raises(SystemExit):
        train_run("ppo_fixed", "test", 3962, tmp_path / "half", 8 * unit, m, quick=True)


def test_non_test_purposes_need_a_clean_tree(monkeypatch, tmp_path):
    import evo2.runio as runio
    monkeypatch.setattr(runio, "git_state", lambda: {"commit": "x", "clean": False, "dirty_files": ["a"]})
    for purpose in ("pilot", "tuning", "final"):
        with pytest.raises(SystemExit):
            runio.require_clean(purpose, quick=False)
        with pytest.raises(SystemExit):
            runio.require_clean(purpose, quick=True)
    runio.require_clean("test", quick=True)


def test_evaluate_refuses_test_split_without_final():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/part3/evaluate.py"), "--runs", "nowhere",
                        "--split", "test"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "needs --final" in (r.stderr + r.stdout)


def test_train_script_has_no_way_to_the_test_split():
    text = (ROOT / "scripts/part3/train.py").read_text()
    assert "--final" not in text and "load_test_split" not in text


def test_evaluate_final_refuses_runs_that_are_not_final(tmp_path):
    log = ROOT / "results/part3/test_access_log.jsonl"
    before = log.read_text() if log.exists() else None
    run = tmp_path / "r"
    run.mkdir()
    (run / "done.json").write_text("{}")
    (run / "run.json").write_text(json.dumps({"purpose": "pilot", "seed": 3000, "method": "oe"}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/part3/evaluate.py"), "--runs", str(run),
                        "--split", "test", "--final", "--adapt"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "only final runs" in (r.stderr + r.stdout)
    # the final evaluation is the whole split with the full adaptation, or nothing
    for extra in (["--adapt", "--max-envs", "2"], [], ["--adapt", "--adapt-budget", "1000"], ["--adapt", "--quick"]):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/part3/evaluate.py"), "--runs", str(run),
                            "--split", "test", "--final", *extra], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "whole test split" in (r.stderr + r.stdout), extra
    assert (log.read_text() if log.exists() else None) == before   # refused looks are not logged


def test_failed_run_is_recorded_when_parameters_stop_being_finite(tmp_path, monkeypatch):
    import evo2.train_loop as tl
    monkeypatch.setattr(tl, "all_finite", lambda tree: False)
    out = train_run("ppo_fixed", "test", 3963, tmp_path, 8 * 2 * 16 * 4, small_cfg(), quick=True)
    assert out["reason"] == "parameters not finite"
    assert (tmp_path / "failed.json").exists() and not (tmp_path / "done.json").exists()


def test_throughput_speed_uses_full_chunks_after_the_first(tmp_path):
    """Speed leaves out the first full chunk (compiling) and the smaller chunks near the end."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("throughput", ROOT / "scripts/part3/throughput.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    lines = [  # chunk, units, agent-steps in the chunk, wall clock at its end
        (1, 10, 1000, 30.0),   # first full chunk: includes compiling, left out
        (2, 10, 1000, 31.0),
        (3, 10, 1000, 32.0),
        (4, 2, 200, 40.0),     # smaller chunk near the end (compiles its own size): left out
    ]
    total = 0
    with open(tmp_path / "progress.jsonl", "w") as fh:
        for c, u, a, t in lines:
            total += a
            fh.write(json.dumps({"chunk": c, "updates": u, "agent_steps": a, "agent_steps_total": total,
                                 "wall_seconds": t}) + "\n")
    (tmp_path / "done.json").write_text(json.dumps({"agent_steps": total, "wall_seconds": 40.0}))
    out = tp.speed(tmp_path)
    assert out["agent_steps_per_second"] == pytest.approx(1000.0) and out["full_chunks_measured"] == 2


def _pilot_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot", ROOT / "scripts/part3/pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pilot_part(method_curves: dict, commit="abc") -> dict:
    """A fake pilot summary: per method, per run, one stand-in environment."""
    common = {"budget": 10, "adapt_budget": 5, "seeds": [3000, 3001], "evaluation_settings": {"e": 1},
              "n_val_envs": 1, "n_shift_envs": 1, "git": {"commit": commit}, "hardware": {}}
    methods = {}
    for m, curves in method_curves.items():
        runs = [{"shift": [{"zero_shot": {"random_competence": 0.0},
                            "adaptation": {"curve": [{"competence": c} for c in curve]}}]} for curve in curves]
        methods[m] = {"runs": runs}
    return {**common, "methods": methods}


def test_pilot_merge_normalizes_over_every_method_and_refuses_mismatches():
    pilot = _pilot_module()
    flat = lambda v: [v] * 11
    a = _pilot_part({"oe": [flat(2.0), flat(4.0)]})
    b = _pilot_part({"ppo_dr": [flat(8.0), flat(6.0)]})
    alone = pilot.add_power(json.loads(json.dumps(a)))
    assert alone["methods"]["oe"]["normalized_auc"] == [[0.5], [1.0]]      # best within the part: 4
    merged = pilot.merge([json.loads(json.dumps(a)), json.loads(json.dumps(b))])
    assert merged["methods"]["oe"]["normalized_auc"] == [[0.25], [0.5]]    # best over both parts: 8
    assert merged["methods"]["ppo_dr"]["normalized_auc"] == [[1.0], [0.75]]
    assert merged["normalized_over_methods"] == ["oe", "ppo_dr"] and "power" in merged
    with pytest.raises(ValueError):
        pilot.merge([a, _pilot_part({"oe": [flat(1.0), flat(1.0)]})])       # a method twice
    with pytest.raises(ValueError):
        pilot.merge([a, _pilot_part({"ppo_dr": [flat(1.0), flat(1.0)]}, commit="def")])
    c = _pilot_part({"ppo_dr": [flat(1.0), flat(1.0)]})
    c["budget"] = 11
    with pytest.raises(ValueError):
        pilot.merge([a, c])


def test_pilot_merge_command_writes_the_merged_summary(tmp_path):
    """The command-line path (pilot.py --merge DIR... --out NEW_FOLDER), not only the function."""
    flat = lambda v: [v] * 11
    for name, part in (("a", _pilot_part({"oe": [flat(2.0), flat(4.0)]})),
                       ("b", _pilot_part({"ppo_dr": [flat(8.0), flat(6.0)]}))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pilot_summary.json").write_text(json.dumps(part))
    out = tmp_path / "new" / "merged"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/part3/pilot.py"), "--merge", str(tmp_path / "a"),
                        str(tmp_path / "b"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    merged = json.loads((out / "pilot_summary.json").read_text())
    assert merged["methods"]["oe"]["normalized_auc"] == [[0.25], [0.5]]


# --- tuning (scripts/part3/tune.py; PREREG Section 5) ---

def _tune_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts/part3/tune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tuning_seeds_draws_and_ranges_follow_the_preregistration():
    from evo2.seeds import SEED_RANGES
    tune = _tune_module()
    assert tune.TRIALS == 8 and tune.BUDGET_FRACTION == 0.25
    lo, hi = SEED_RANGES["tuning"]
    for m, idx in (("oe", 0), ("oe_fixed", 1), ("ppo_fixed", 2), ("ppo_dr", 3), ("es_dr", 4)):
        seeds = [tune.trial_seed(m, t) for t in range(tune.TRIALS)]
        assert seeds == [4000 + 100 * idx + t for t in range(tune.TRIALS)]
        assert all(lo <= x <= hi for x in seeds)
        qlo, qhi = SEED_RANGES["test"]
        assert all(qlo <= tune.trial_seed(m, t, quick=True) <= qhi for t in range(tune.QUICK["trials"]))
        draws = [tune.draw(m, x) for x in seeds]
        assert draws[0] == tune.draw(m, seeds[0])                       # deterministic
        for d in draws:
            for name, kind, *args in tune.SPACES[tune.FAMILY[m]]:
                if kind == "loguniform":
                    assert args[0] <= d[name] <= args[1]
                else:
                    assert d[name] in args[0]
        assert len({json.dumps(d, sort_keys=True) for d in draws}) == len(draws)   # trials differ
    # a log-uniform draw covers both halves of its log range over many seeds
    lrs = np.log10([tune.draw("ppo_dr", 4300 + t)["LR"] for t in range(60)])
    assert lrs.min() < -3.5 < lrs.max()


def test_tuned_settings_reach_the_trainer_for_that_method_only(tmp_path):
    import yaml
    from evo2.run import load_methods_config, make_trainer, method_settings
    tune = _tune_module()
    base = load_methods_config(tuned=None)
    c = tune.trial_config(base, "ppo_dr", {"LR": 0.000123, "NUM_STEPS": 32})
    assert method_settings(c, "ppo_dr", "ppo")["LR"] == 0.000123
    assert method_settings(c, "ppo_fixed", "ppo")["LR"] == base["ppo"]["settings"]["LR"]
    tr = make_trainer("ppo_dr", c)
    assert tr.ppo.cfg["LR"] == 0.000123 and tr.ppo.cfg["NUM_STEPS"] == 32
    t = tmp_path / "tuned.yaml"
    t.write_text(yaml.safe_dump({"source": {"x": 1}, "settings": {"es_dr": {"SIGMA": 0.02}, "oe": {"WEIGHT_SD": 0.01}}}))
    m = load_methods_config(tuned=t)
    assert make_trainer("es_dr", m).es.cfg["SIGMA"] == 0.02
    assert m["oe"]["settings"]["WEIGHT_SD"] == 0.01 and m["oe"]["settings"]["MIN_POP"] == base["oe"]["settings"]["MIN_POP"]
    assert m["tuned_from"] == {"x": 1}


def test_tuning_score_matches_zero_shot_and_best_trial_breaks_ties_low():
    from evo2.agents.brain import init_mlp
    from evo2.eval.protocol import zero_shot
    from evo2.seeds import base_key, stream_key
    tune = _tune_module()
    envs, ep_seeds = tune.validation_envs()
    assert len(envs) == 24
    w_traits = [1.0, 1.0, 10.0]
    pol = {"kind": "shared", "brain": init_mlp(jax.random.PRNGKey(3), 30), "traits": np.array(w_traits, np.float32)}
    s = tune.score(pol, envs[:1], ep_seeds, 2)
    zs = zero_shot(stream_key(base_key("validation", ep_seeds[0]), "eval_episodes"), envs[0][2], pol,
                   {"episodes_per_env": 2, "persistence_agents": 32, "persistence_steps": 10})
    assert s["envs"][0]["competence"] == zs["competence"]
    assert s["envs"][0]["random_competence"] == zs["random_competence"]
    assert s["score"] == zs["competence"] - zs["random_competence"]
    trials = [{"trial": 0, "score": 1.0}, {"trial": 1, "score": 2.0}, {"trial": 2, "score": 2.0}]
    assert tune.best_trial(trials)["trial"] == 1


def _script_module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/part3/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tuning_failed_trials_score_lowest_and_all_failed_stops(tmp_path, monkeypatch):
    """PREREG 5: a trial whose parameters stop being finite scores lowest; if every trial fails, stop."""
    tune = _tune_module()
    failed = {4000, 4002, 4007}                       # trial 7 would otherwise score highest

    def fake_train(method, purpose, seed, run_dir, budget, cfg, quick=False):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ("failed.json" if seed in failed else "done.json")).write_text("{}")
        return {"seed": seed}

    monkeypatch.setattr(tune, "train_run", fake_train)
    monkeypatch.setattr(tune, "load_pickle", lambda p: {"seed": int(p.parent.name.split("seed")[1])})
    monkeypatch.setattr(tune, "score", lambda pol, envs, seeds, n: {"score": float(pol["seed"] - 4000), "envs": []})
    out = tune.tune_method("oe", load_methods_config(tuned=None), tmp_path / "a", quick=False)
    rows = {r["seed"]: r for r in out["trials"]}
    assert len(rows) == 8 and all(rows[s]["score"] == float("-inf") and rows[s]["failed"] for s in failed)
    assert out["chosen"]["seed"] == 4006 and out["budget_agent_steps"] == 50_000_000
    failed.update(range(4000, 4008))
    with pytest.raises(SystemExit, match="every oe trial failed"):
        tune.tune_method("oe", load_methods_config(tuned=None), tmp_path / "b", quick=False)


def test_write_tuned_round_trip_keeps_every_method_apart(tmp_path):
    import yaml
    from evo2.run import make_trainer, method_settings
    tune = _tune_module()
    chosen = {"oe": {"WEIGHT_SD": 0.011}, "oe_fixed": {"WEIGHT_SD": 0.022},
              "ppo_fixed": {"LR": 0.000101, "NUM_STEPS": 128}, "ppo_dr": {"LR": 0.000202, "NUM_STEPS": 32},
              "es_dr": {"SIGMA": 0.055, "POP_SIZE": 32}}
    folder = tmp_path / "tuning"
    folder.mkdir()
    for i, (m, s) in enumerate(chosen.items()):
        (folder / f"tuning_{m}.json").write_text(json.dumps(
            {"quick": False, "git": {"commit": "c0ffee"}, "chosen": {"trial": i, "seed": 4000 + i, "score": 1.0, "settings": s}}))
    path = tmp_path / "tuned.yaml"
    tune.write_tuned(folder, path)
    base, m = load_methods_config(tuned=None), load_methods_config(tuned=path)
    assert yaml.safe_load(path.read_text())["settings"] == chosen
    assert m["tuned_from"]["per_method"]["ppo_dr"]["seed"] == 4003
    for meth in ("oe", "oe_fixed"):                    # OE settings merge into each method's own block
        assert m[meth]["settings"] == {**base[meth]["settings"], **chosen[meth]}
    assert make_trainer("oe_fixed", m).oe.cfg["WEIGHT_SD"] == 0.022
    for meth in ("ppo_fixed", "ppo_dr"):               # the two PPO methods keep their own settings
        assert method_settings(m, meth, "ppo") == {**base["ppo"]["settings"], **chosen[meth]}
        assert make_trainer(meth, m).ppo.cfg["LR"] == chosen[meth]["LR"]
    assert make_trainer("es_dr", m).es.cfg["SIGMA"] == 0.055
    (folder / "tuning_es_dr.json").write_text(json.dumps({"quick": True, "git": {"commit": "x"},
                                                          "chosen": {"trial": 0, "seed": 3950, "score": 1.0, "settings": {}}}))
    with pytest.raises(SystemExit, match="quick"):
        tune.write_tuned(folder, tmp_path / "t2.yaml")
    (folder / "tuning_es_dr.json").unlink()
    with pytest.raises(FileNotFoundError):
        tune.write_tuned(folder, tmp_path / "t3.yaml")


def test_final_training_needs_the_tuned_settings(tmp_path, monkeypatch):
    train = _script_module("train")
    calls = []
    monkeypatch.setattr(train, "train_run", lambda *a, **k: calls.append(a) or {"ok": True})
    monkeypatch.setattr(train, "load_methods_config", lambda: load_methods_config(tuned=None))
    monkeypatch.setattr(sys, "argv", ["train.py", "--method", "oe", "--purpose", "final", "--seed", "5000",
                                      "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="tuned.yaml is missing"):
        train.main()
    assert calls == []
    monkeypatch.setattr(train, "load_methods_config", lambda: {**load_methods_config(tuned=None), "tuned_from": {"x": 1}})
    train.main()
    assert len(calls) == 1 and calls[0][:3] == ("oe", "final", 5000)
