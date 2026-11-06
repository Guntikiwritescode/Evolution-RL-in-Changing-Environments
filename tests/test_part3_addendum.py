"""Addendum A (PREREG_part3.md Section 16): ES-fixed, its tuning, its settings file and its analysis.

Run with: venv3/bin/python -m pytest tests/test_part3_addendum.py
Each test here has a break in scripts/part3/break_tests.py that makes it fail.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from evo2 import analysis as A
from evo2.env.generator import params_to_dict
from evo2.env.splits import load_train
from evo2.run import ADDENDUM_METHODS, METHODS, env_fn_for, load_methods_config, make_trainer, method_settings

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/part3/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b):
    return params_to_dict(a) == params_to_dict(b)


def test_core_method_list_is_unchanged():
    # the pre-registered core analysis reads METHODS; the addendum must not change it
    assert METHODS == ("oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr")
    assert ADDENDUM_METHODS == ("es_fixed",)


def test_es_fixed_trains_in_the_canonical_environment_and_es_dr_does_not():
    _, canon = load_train()
    keys = jax.random.split(jax.random.PRNGKey(3901), 4)
    fixed, dr = env_fn_for("es_fixed"), env_fn_for("es_dr")
    assert all(_same(fixed(k), canon) for k in keys)
    assert not all(_same(dr(k), canon) for k in keys)
    assert env_fn_for("es_fixed", adapt=True) is None and env_fn_for("es_dr", adapt=True) is None


def test_es_fixed_uses_es_with_its_own_tuned_settings(tmp_path):
    extra = tmp_path / "tuned_addendum.yaml"
    extra.write_text(yaml.safe_dump({"source": {"x": 1}, "settings": {"es_fixed": {"SIGMA": 0.077, "POP_SIZE": 32}}}))
    m = load_methods_config(extra_tuned=extra)
    assert m["tuned_from_addendum"] == {"x": 1}
    assert method_settings(m, "es_fixed", "es")["SIGMA"] == 0.077
    assert method_settings(m, "es_dr", "es") == method_settings(load_methods_config(), "es_dr", "es")
    t = make_trainer("es_fixed", m)
    assert t.es.cfg["SIGMA"] == 0.077 and t.es.cfg["POP_SIZE"] == 32
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"source": {}, "settings": {"es_dr": {"SIGMA": 0.5}}}))
    with pytest.raises(ValueError, match="unknown method"):
        load_methods_config(extra_tuned=bad)


def test_es_fixed_tuning_seeds_space_and_starting_settings(monkeypatch):
    tune = _module("tune")
    assert [tune.trial_seed("es_fixed", t) for t in range(8)] == list(range(4500, 4508))
    assert tune.FAMILY["es_fixed"] == "es" and set(tune.draw("es_fixed", 4500)) == {"SIGMA", "LR", "POP_SIZE", "EPISODES"}
    assert tune.config_for(["es_fixed"]) == load_methods_config(tuned=None)
    monkeypatch.setattr(tune, "TUNED_FILE", ROOT / "configs/part3/tuned.yaml")
    with pytest.raises(SystemExit, match="tuned.yaml exists"):
        tune.config_for(["es_dr"])
    with pytest.raises(SystemExit, match="separately"):
        tune.config_for(["es_dr", "es_fixed"])


def _summary(quick=False, trials=8, method="es_fixed"):
    return {"method": method, "quick": quick, "git": {"commit": "c0ffee"},
            "trials": [{"trial": t} for t in range(trials)],
            "chosen": {"trial": 3, "seed": 4503, "score": 2.0, "settings": {"SIGMA": 0.03, "LR": 0.01, "POP_SIZE": 64, "EPISODES": 2}}}


def test_write_addendum_round_trip_and_refusals(tmp_path):
    tune = _module("tune")
    folder = tmp_path / "tuning"
    folder.mkdir()
    (folder / "tuning_es_fixed.json").write_text(json.dumps(_summary()))
    out = tmp_path / "tuned_addendum.yaml"
    tune.write_addendum(folder, out)
    m = load_methods_config(extra_tuned=out)
    assert method_settings(m, "es_fixed", "es") == {**load_methods_config()["es"]["settings"], **_summary()["chosen"]["settings"]}
    assert m["tuned_from_addendum"]["per_method"]["es_fixed"]["seed"] == 4503
    for bad in (_summary(quick=True), _summary(trials=7), _summary(method="es_dr")):
        (folder / "tuning_es_fixed.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            tune.write_addendum(folder, tmp_path / "x.yaml")


def test_train_cli_needs_tuned_settings_for_es_fixed_and_refuses_them_for_core(tmp_path, monkeypatch):
    train = _module("train")
    calls = []
    monkeypatch.setattr(train, "train_run", lambda *a, **k: calls.append((a, k)) or {"ok": True})
    base = ["train.py", "--purpose", "final", "--seed", "5000", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base + ["--method", "es_fixed"])
    with pytest.raises(SystemExit, match="tuned-extra"):
        train.main()
    extra = tmp_path / "t.yaml"
    extra.write_text(yaml.safe_dump({"source": {"y": 2}, "settings": {"es_fixed": {"SIGMA": 0.02}}}))
    monkeypatch.setattr(sys, "argv", base + ["--method", "es_dr", "--tuned-extra", str(extra)])
    with pytest.raises(SystemExit, match="addendum methods"):
        train.main()
    assert calls == []
    monkeypatch.setattr(sys, "argv", base + ["--method", "es_fixed", "--tuned-extra", str(extra)])
    train.main()
    (args, _), = calls
    assert args[0] == "es_fixed" and args[4] == 200_000_000
    assert args[5]["es_fixed"]["settings"]["SIGMA"] == 0.02 and args[5]["tuned_from_addendum"] == {"y": 2}


# ---------- analysis ----------

def _arrays(level, n=10, envs=20, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return level + spread * rng.standard_normal((n, envs))


def test_a1_and_family_read_es_dr_above_es_fixed_as_support():
    prim = {"es_dr": _arrays(0.88, seed=1), "es_fixed": _arrays(0.80, seed=2)}
    r = A.a1(prim, reps=2000)
    assert r["D"] > 0 and r["ci"][0] > 0 and r["verdict"] == "support"
    meas = {"primary": {m: np.concatenate([np.zeros((10, 10)), v], axis=1) for m, v in prim.items()},
            "zero_shot": {"es_dr": np.concatenate([_arrays(0.5, envs=10, seed=3), np.zeros((10, 20))], axis=1),
                          "es_fixed": np.concatenate([_arrays(0.7, envs=10, seed=4), np.zeros((10, 20))], axis=1)}}
    rows = {r["name"]: r for r in A.compare_family(meas, A.ADDENDUM_FAMILY, reps=2000)}
    assert rows["A2"]["result"] == "supported" and rows["A2"]["sets"] == ["T2", "T3"]
    # A3 is the zero-shot measure on T1 only: here ES-fixed is higher, against the prediction
    assert rows["A3"]["measure"] == "zero_shot" and rows["A3"]["sets"] == ["T1"]
    assert rows["A3"]["result"] == "evidence against" and rows["A3"]["iqm_difference"] < 0


def test_a4_interaction_sign_and_reading():
    prim = {"es_dr": _arrays(0.88, seed=1), "es_fixed": _arrays(0.80, seed=2),
            "oe": _arrays(0.29, seed=3), "oe_fixed": _arrays(0.41, seed=4)}
    r = A.a4(prim, reps=2000)
    assert r["variety_es"]["value"] > 0 > r["variety_oe"]["value"]
    assert r["interaction"]["value"] == pytest.approx(r["variety_es"]["value"] - r["variety_oe"]["value"])
    assert r["verdict"] == "variety effect larger for ES than for OE"
    assert A.interaction_verdict(-0.2, -0.1) == "variety effect larger for OE than for ES"
    assert A.interaction_verdict(-0.1, 0.1) == "no difference found"


def test_a1_margin_and_family_size_are_the_preregistered_ones():
    assert A.MARGIN == 0.05 and len(A.ADDENDUM_FAMILY) == 2
    prim = {"es_dr": _arrays(0.83, seed=1, spread=0.002), "es_fixed": _arrays(0.80, seed=2, spread=0.002)}
    r = A.a1(prim, reps=2000)
    assert r["margin"] == 0.05 and r["verdict"] == "support (below the margin)"
    meas = {"primary": {m: np.concatenate([np.zeros((10, 10)), v], axis=1) for m, v in prim.items()},
            "zero_shot": {m: np.concatenate([v[:, :10], np.zeros((10, 20))], axis=1) for m, v in prim.items()}}
    rows = A.compare_family(meas, A.ADDENDUM_FAMILY, reps=200)
    lo = min(r["p"] for r in rows)
    assert min(r["p_holm"] for r in rows) == pytest.approx(min(1.0, 2 * lo))


def test_readings_and_rerun_flag():
    r = A.addendum_readings("support (below the margin)", "supported")
    assert "below the pre-set margin" in r["adaptation"] and "as Milano et al. found" in r["milano_claim"]
    assert "does not by itself support Milano et al.'s direction" in r["adaptation"]
    assert "not for survival-based OE" in r["milano_claim"] and "which part of it is not tested" in r["milano_claim"]
    assert "did not reward variety" in A.addendum_readings("against", "evidence against")["milano_claim"]
    with pytest.raises(ValueError):
        A.addendum_readings("support", "difference (X higher)")
    assert not A.rerun_flag([0.857, 0.885], 0.856, 0.881)
    assert A.rerun_flag([0.892], 0.856, 0.881) and A.rerun_flag([0.84], 0.856, 0.881)


def test_addendum_tuning_refuses_the_core_tuning_folder(monkeypatch):
    tune = _module("tune")

    def would_tune(*a, **k):
        raise RuntimeError("tuning would have started")
    monkeypatch.setattr(tune, "tune_method", would_tune)
    monkeypatch.setattr(tune, "require_clean", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["tune.py", "--methods", "es_fixed"])
    with pytest.raises(SystemExit, match="its own --out"):
        tune.main()
    monkeypatch.setattr(sys, "argv", ["tune.py", "--methods", "es_fixed", "--out", "/tmp/elsewhere"])
    with pytest.raises(RuntimeError, match="would have started"):
        tune.main()


# ---------- the addendum analysis end to end, on fake run folders ----------

EVAL = {"episodes_per_env": 8, "oe_sample": 16, "persistence_agents": 32, "persistence_steps": 5000}
SETTINGS = {"SIGMA": 0.05, "LR": 0.01, "POP_SIZE": 64, "EPISODES": 1}


def _fake_run(root, method, seed, zero, final, commit="c1", random=1.0):
    """A run folder that the core analysis's reader (analyze.py: read_eval) accepts."""
    d = root / f"{method}_seed{seed}"
    d.mkdir(parents=True)
    mc = {"tuned_from": {"x": 1}, "budget": {"adapt_agent_steps": 20_000_000}}
    if method == "es_fixed":
        mc.update({"es_fixed": {"settings": SETTINGS}, "tuned_from_addendum": {"y": 1}})
    git = {"commit": commit, "clean": True}
    (d / "run.json").write_text(json.dumps({"method": method, "seed": seed, "purpose": "final", "quick": False,
                                            "git": git, "hardware": {"gpus": ["G"]}, "methods_config": mc}))
    (d / "done.json").write_text(json.dumps({"reached_budget": True, "budget_agent_steps": 200_000_000,
                                             "agent_steps": 200_000_001, "chunks": 1, "wall_seconds": 30.0}))
    (d / "progress.jsonl").write_text(json.dumps({"agent_steps_total": 200_000_001}) + "\n")
    rng = np.random.default_rng(seed + len(method))
    envs = []
    for k, s in enumerate(("T1", "T2", "T3")):
        for i in range(10):
            j = 0.05 * rng.standard_normal()
            curve = [{"competence": float(zero + j + (final - zero) * min(1.0, p / 3)), "agent_steps": p * 2_000_000}
                     for p in range(11)]
            chunks = [{"births": 0, "floor_births": 0, "steps_all_dead_before_floor": 0}]
            envs.append({"set": s, "index": i,
                         "zero_shot": {"competence": curve[0]["competence"], "random_competence": random,
                                       "persistence": {"extinct": False, "steps_survived": 5000, "births": 0}},
                         "adaptation": {"curve": curve, "wall_seconds": 1.0, "chunks": chunks}})
    (d / "eval_test.json").write_text(json.dumps({"split": "test", "git": git, "hardware": {"gpus": ["G"]},
                                                  "evaluation_settings": EVAL, "envs": envs}))
    (d / "curves.json").write_text(json.dumps({"quick": False, "n_envs": 24, "episodes_per_env": 8, "git": git,
                                               "points": [{"tenth": t, "envs": [{"random_competence": 0.5}]}
                                                          for t in range(1, 11)]}))
    return d


def _fake_study(tmp_path, es_fixed=(3.0, 8.0), es_dr=(5.0, 9.0), skip=(), rerun=True):
    final, add = tmp_path / "final", tmp_path / "add"
    levels = {"es_dr": es_dr, "oe": (3.0, 3.5), "oe_fixed": (4.0, 4.5)}
    for m, (z, f) in levels.items():
        for s in range(5000, 5010):
            _fake_run(final, m, s, z, f)
    (final / "analysis.json").write_text(json.dumps({"best": [11.0] * 30, "random": [1.0] * 30,
                                                     "runs": {m: list(range(5000, 5010)) for m in levels}}))
    for s in range(5000, 5010):
        if s not in skip:
            _fake_run(add, "es_fixed", s, *es_fixed, commit="c2")
    if rerun:
        for s in (5000, 5001):
            _fake_run(add, "es_dr", s, *es_dr, commit="c2")
    (add / "tuning").mkdir(parents=True)
    (add / "tuning/tuning_es_fixed.json").write_text(json.dumps({"method": "es_fixed", "quick": False,
        "trials": [{"trial": t, "seed": 4500 + t, "settings": SETTINGS, "score": float(t)} for t in range(8)],
        "chosen": {"trial": 7, "seed": 4507, "score": 7.0, "settings": SETTINGS}}))
    (add / "tuned_addendum.yaml").write_text(yaml.safe_dump({"source": {}, "settings": {"es_fixed": SETTINGS}}))
    return final, add


def _run(final, add, tmp_path):
    return _module("analyze_addendum").run(final, add, 300, tmp_path / "r.md", tmp_path / "f.png")


def test_addendum_analysis_end_to_end_reads_es_dr_ahead(tmp_path):
    final, add = _fake_study(tmp_path)
    res = _run(final, add, tmp_path)
    fam = {r["name"]: r for r in res["family"]}
    assert res["verdicts_given"] and res["a1"]["D"] > 0.05 and res["a1"]["verdict"] == "support"
    assert fam["A2"]["result"] == "supported" and fam["A3"]["result"] == "supported"
    assert "as Milano et al. found" in res["readings"]["milano_claim"]
    assert res["es_dr_check"]["flag"] is False and res["es_dr_check"]["missing"] == []
    assert "Pre-set readings" in (tmp_path / "r.md").read_text()


def test_addendum_analysis_left_out_seeds(tmp_path):
    final, add = _fake_study(tmp_path / "a", skip=(5003, 5007))
    res = _run(final, add, tmp_path / "a")
    assert res["verdicts_given"] and [x["seed"] for x in res["excluded_es_fixed"]] == [5003, 5007]
    final, add = _fake_study(tmp_path / "b", skip=(5001, 5002, 5009))
    (add / "es_fixed_seed5002").mkdir()                          # started, never finished
    res = _run(final, add, tmp_path / "b")
    assert not res["verdicts_given"] and "a1" not in res and len(res["excluded_es_fixed"]) == 3
    assert "No verdicts" in (tmp_path / "b/r.md").read_text()


def test_addendum_analysis_stops_on_wrong_settings_random_or_commits(tmp_path):
    final, add = _fake_study(tmp_path / "a")
    (add / "tuned_addendum.yaml").write_text(yaml.safe_dump({"source": {}, "settings": {"es_fixed": {**SETTINGS, "LR": 0.02}}}))
    with pytest.raises(SystemExit, match="tuning's choice"):
        _run(final, add, tmp_path / "a")
    final, add = _fake_study(tmp_path / "b", skip=(5004,))
    _fake_run(add, "es_fixed", 5004, 3.0, 8.0, commit="c2", random=1.01)
    with pytest.raises(SystemExit, match="random competence"):
        _run(final, add, tmp_path / "b")
    final, add = _fake_study(tmp_path / "c", skip=(5004,))
    _fake_run(add, "es_fixed", 5004, 3.0, 8.0, commit="c3")
    with pytest.raises(SystemExit, match="one commit"):
        _run(final, add, tmp_path / "c")


def test_addendum_analysis_reruns_are_optional_and_flagged(tmp_path):
    final, add = _fake_study(tmp_path / "a", rerun=False)
    res = _run(final, add, tmp_path / "a")
    assert res["es_dr_check"]["missing"] == [5000, 5001] and res["es_dr_check"]["rows"] == []
    final, add = _fake_study(tmp_path / "b", rerun=False)
    _fake_run(add, "es_dr", 5000, 7.0, 10.5, commit="c2")
    res = _run(final, add, tmp_path / "b")
    assert res["es_dr_check"]["flag"] is True and res["es_dr_check"]["missing"] == [5001]


def test_every_reading_is_quoted_from_the_preregistration():
    """Every reading is made of fragments that appear word for word in PREREG 16.4 (MISTAKES #90)."""
    from pathlib import Path
    pre = (Path(__file__).resolve().parents[1] / "docs/PREREG_part3.md").read_text()
    sec = pre[pre.index("### 16.4"):pre.index("### 16.5")]
    for a1 in ("support", "support (below the margin)", "against", "against (below the margin)",
               "no meaningful difference", "inconclusive"):
        for a3 in ("supported", "evidence against", "not supported"):
            r = A.addendum_readings(a1, a3)
            for frag in r["fragments"]["milano_claim"] + r["fragments"]["adaptation"]:
                assert frag in sec, frag
            assert r["milano_claim"] == "; ".join(r["fragments"]["milano_claim"])
            assert r["adaptation"] == "; ".join(r["fragments"]["adaptation"])
