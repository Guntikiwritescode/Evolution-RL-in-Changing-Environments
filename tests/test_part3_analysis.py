"""The Part 3 final analysis (src/evo2/analysis.py, scripts/part3/analyze.py) against hand-worked
examples (PREREG Sections 7-8). Synthetic data only; no test environment is run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from evo2 import analysis as A

ROOT = Path(__file__).resolve().parents[1]


def flat_curves(levels_by_run, n_env=30):
    """(runs, 30, 11) curves, each flat at the run's level."""
    return np.stack([np.full((n_env, 11), float(v)) for v in levels_by_run])


def test_best_valid_normalized_and_auc_by_hand():
    random = np.full(30, 1.0)
    curves = {"a": flat_curves([3.0, 5.0]), "b": flat_curves([2.0, 2.0])}
    curves["b"][1, 4, 7] = 9.0                         # one later checkpoint sets env 4's best
    random[7] = 6.0                                    # env 7: best (5) <= random (6): left out
    best = A.best_per_env(curves)
    assert best[0] == 5.0 and best[4] == 9.0 and best[7] == 5.0
    valid = A.valid_envs(best, random)
    assert not valid[7] and valid.sum() == 29
    n = A.normalized_curves(curves["a"], random, best, valid)
    assert n[0, 0, 0] == pytest.approx((3 - 1) / (5 - 1)) and n[1, 0, 5] == pytest.approx(1.0)
    assert n[0, 4, 0] == pytest.approx((3 - 1) / (9 - 1)) and np.isnan(n[0, 7]).all()
    auc = A.auc(n)
    assert auc[0, 0] == pytest.approx(0.5) and np.isnan(auc[0, 7])   # a flat curve at h has area h
    ramp = np.zeros((1, 30, 11))
    ramp[0, :, :] = np.linspace(0, 1, 11)
    assert A.auc(A.normalized_curves(ramp * 4 + 1, np.ones(30), np.full(30, 5.0), np.ones(30, bool)))[0, 0] == pytest.approx(0.5)
    sel = A.select(auc, A.env_index("T1"))
    assert sel.shape == (2, 9)                          # env 7 (in T1) dropped


def test_holm_by_hand_and_against_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = [0.01, 0.04, 0.03, 0.2]
    h = A.holm(p)
    # sorted 0.01 x4 = 0.04; 0.03 x3 = 0.09; 0.04 x2 = 0.08 -> 0.09 (monotone); 0.2 x1 -> 0.2
    assert [x["p_holm"] for x in h] == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert [x["reject"] for x in h] == [True, False, False, False]
    rej, adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    assert list(rej) == [x["reject"] for x in h] and np.allclose(adj, [x["p_holm"] for x in h])


def test_mann_whitney_probability_of_improvement_by_hand():
    t = A.mwu(np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0]))
    assert t["U"] == pytest.approx(8.5) and t["p_x_greater"] == pytest.approx(8.5 / 9)


def _levels(by_set, runs=10, noise=0.01, seed=0):
    """(runs, 30): each environment set at its own level, with small run-to-run noise."""
    rng = np.random.default_rng(seed)
    out = np.zeros((runs, 30))
    for s, v in by_set.items():
        out[:, A.env_index(s)] = v + rng.normal(0, noise, (runs, 10))
    return out


def test_h1_verdict_and_probability_on_clear_synthetic_data():
    rng = np.random.default_rng(0)
    level = {"oe": 0.9, "oe_fixed": 0.6, "ppo_dr": 0.5, "ppo_fixed": 0.1, "es_dr": 0.3}
    primary = {m: np.full((10, 30), v) + rng.normal(0, 1e-3, (10, 30)) for m, v in level.items()}
    h = A.h1({m: A.select(v, A.env_index("T2", "T3")) for m, v in primary.items()}, reps=300)
    assert h["verdict"] == "support" and h["D"] == pytest.approx(0.4, abs=0.01) and h["p_oe_greater"] == 1.0


def test_each_secondary_comparison_uses_its_own_methods_measure_and_sets():
    """Every S row is built to give a different result from the one its neighbours would give, so a
    row that read another measure, another set of environments or another method would fail."""
    z = {"T1": 0.0, "T2": 0.0, "T3": 0.0}
    primary = {"oe": _levels({"T1": 0.9, "T2": 0.5, "T3": 0.5}, seed=1),
               "es_dr": _levels({"T1": 0.95, "T2": 0.2, "T3": 0.2}, seed=2),
               "oe_fixed": _levels({"T1": 0.95, "T2": 0.3, "T3": 0.3}, seed=3),
               "ppo_dr": _levels({"T1": 0.6, "T2": 0.7, "T3": 0.7}, seed=4),
               "ppo_fixed": _levels({"T1": 0.9, "T2": 0.4, "T3": 0.4}, seed=5)}
    shared = _levels({"T1": 0.5, "T2": 0.5, "T3": 0.5}, seed=6)
    zero_shot = {m: _levels(z, seed=7) for m in primary}
    zero_shot["oe"], zero_shot["ppo_dr"] = shared.copy(), shared.copy()      # T2, T3: identical
    zero_shot["oe"][:, A.env_index("T1")] += 0.4                            # T1: OE far higher
    final = {m: _levels(z, seed=8) for m in primary}
    final["oe"] = _levels({"T1": 0.1, "T2": 0.8, "T3": 0.8}, seed=9)      # T2, T3: OE higher (the primary
    final["ppo_dr"] = _levels({"T1": 0.9, "T2": 0.2, "T3": 0.2}, seed=10) # metric has PPO-DR higher there)
    extinct = {m: np.zeros((10, 30)) for m in primary}
    extinct["oe"][:, A.env_index("T1")] = 1.0                               # T1: OE always extinct
    extinct["ppo_dr"][:, A.env_index("T2", "T3")] = 1.0                     # T2, T3: PPO-DR always extinct
    extinct["oe"][:3, 10:13] = 1.0                                          # OE: 0.15 of T2, T3 in runs 0-2
    measures = {"primary": primary, "zero_shot": zero_shot, "final": final, "extinct": extinct}
    rows = {r["name"]: r for r in A.secondary(measures, reps=200)}
    assert [r["name"] for r in A.secondary(measures, reps=20)] == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    want = {"S1": "supported", "S2": "supported", "S3": "supported", "S4": "no difference found",
            "S5": "supported", "S6": "difference (X higher)", "S7": "supported"}
    assert {k: r["result"] for k, r in rows.items()} == want
    assert rows["S4"]["p"] == pytest.approx(1.0) and rows["S4"]["p_x_greater"] == pytest.approx(0.5)
    from evo2 import metrics
    oe_rate = A.per_run_mean(extinct["oe"][:, A.env_index("T2", "T3")])
    assert rows["S7"]["iqm_difference"] == pytest.approx(metrics.iqm(oe_rate[:, None]) - 1.0)
    # per-run fractions: 7 runs at 0, 3 at 0.15; IQM (2 cut from each end of 10) = 0.15 / 6 = 0.025.
    # Over the 200 (run, environment) 0/1 values the IQM would be 0, giving -1.0.
    assert rows["S7"]["iqm_difference"] == pytest.approx(0.025 - 1.0)
    assert rows["S1"]["iqm_difference"] == pytest.approx(
        metrics.iqm(primary["oe"][:, 10:]) - metrics.iqm(primary["es_dr"][:, 10:]))


def test_all_tied_comparison_reports_p_1_and_is_not_supported():
    same = {m: np.zeros((10, 30)) for m in ("oe", "oe_fixed", "ppo_dr", "ppo_fixed", "es_dr")}
    t = A.mwu(np.zeros(10), np.zeros(10))
    assert t["p"] == 1.0 and t["all_tied"] and t["p_x_greater"] == 0.5
    rows = A.secondary({"primary": same, "zero_shot": same, "final": same, "extinct": same}, reps=20)
    assert all(r["p"] == 1.0 and r["p_holm"] == 1.0 and not r["reject"] for r in rows)
    assert {r["result"] for r in rows} == {"not supported", "no difference found"}


def test_secondary_not_supported_when_no_difference():
    rng = np.random.default_rng(1)
    same = {m: rng.normal(0.5, 0.1, (10, 30)) for m in ("oe", "oe_fixed", "ppo_dr", "ppo_fixed", "es_dr")}
    measures = {"primary": same, "zero_shot": same, "final": same,
                "extinct": {m: np.zeros((10, 30)) for m in same}}
    rows = A.secondary(measures, reps=100)
    assert not any(r["reject"] for r in rows)
    assert {r["result"] for r in rows} <= {"not supported", "no difference found"}


def test_h2b_point_values_and_whole_run_bootstrap():
    level = {"oe": 0.9, "oe_fixed": 0.6, "ppo_dr": 0.5, "ppo_fixed": 0.1}
    const = {m: np.full((10, 20), v) for m, v in level.items()}
    b = A.h2b(const, reps=50)
    # V = ((0.9-0.6) + (0.5-0.1))/2 = 0.35; M = ((0.9-0.5) + (0.6-0.1))/2 = 0.45; I = 0.3 - 0.4 = -0.1
    assert b["variety"]["value"] == pytest.approx(0.35) and b["method"]["value"] == pytest.approx(0.45)
    assert b["interaction"]["value"] == pytest.approx(-0.1) and b["delta"]["value"] == pytest.approx(-0.1)
    assert b["verdict"] == "method matters more" and b["interaction_excludes_0"]
    # the bootstrap resamples whole runs: spread between runs widens the CI, spread between
    # environments within a run (the same in every run) does not
    rng = np.random.default_rng(3)
    by_run = {m: np.repeat(v + rng.normal(0, 0.2, (10, 1)), 20, axis=1) for m, v in level.items()}
    by_env = {m: np.repeat(v + rng.normal(0, 0.2, (1, 20)), 10, axis=0) for m, v in level.items()}
    w_run = np.diff(A.h2b(by_run, reps=300)["variety"]["ci"])[0]
    w_env = np.diff(A.h2b(by_env, reps=300)["variety"]["ci"])[0]
    assert w_run > 0.1 and w_env == pytest.approx(0.0, abs=1e-12)
    assert A.h2b(by_run, reps=300) == A.h2b(by_run, reps=300)


def test_time_to_90_and_diversity_by_hand():
    curve = np.zeros((1, 30, 11))
    curve[0, 10] = [5, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]   # reaches 9 = 0.9 x 10 at point 2
    curve[0, 11] = 1.0                                  # never
    t = A.time_to_90(curve, np.array([10.0]), [10, 11])
    assert t["pairs"] == 2 and t["reached"] == 1 and t["median_time_if_reached"] == pytest.approx(0.2)
    assert A.mean_pairwise_distance(np.array([[0, 0], [3, 4], [0, 0]])) == pytest.approx((5 + 0 + 5) / 3)
    acts = np.zeros((2, 7, 2))
    acts[1, :, :] = [3, 4]
    assert A.behavioral_distance(acts) == pytest.approx(5.0)


def _fake_final(tmp: Path, seeds=range(5000, 5010)):
    """A final-runs folder in the analyze.py format, from real untrained policies (no training,
    no test environment): curves, persistence and population counts are synthetic."""
    from evo2.run import load_methods_config, make_trainer, to_numpy
    from evo2.runio import save_pickle
    mcfg = {**load_methods_config(tuned=None), "tuned_from": {"x": 1}}
    rng = np.random.default_rng(2)
    random = rng.uniform(0.5, 1.5, 30)
    val_random = [{"random_competence": float(x)} for x in rng.uniform(0.5, 1.5, 24)]
    level = {"oe": 0.3, "oe_fixed": 0.25, "ppo_fixed": 0.8, "ppo_dr": 0.85, "es_dr": 0.9}
    git = {"commit": "abc", "clean": True}
    for m in level:
        tr = make_trainer(m, mcfg)
        pol = to_numpy(tr.policy(tr.init(jax.random.PRNGKey(5))))
        for s in seeds:
            d = tmp / f"{m}_seed{s}"
            (d / "snapshots").mkdir(parents=True)
            save_pickle(d / "policy.pkl", pol)
            for k in range(1, 11):
                save_pickle(d / "snapshots" / f"{k}.pkl", {"agent_steps": k * 10, "policy": pol})
            (d / "run.json").write_text(json.dumps({"purpose": "final", "seed": s, "method": m, "git": git, "quick": False,
                                                    "methods_config": mcfg, "hardware": {"gpus": ["train gpu"]}}))
            (d / "done.json").write_text(json.dumps({"agent_steps": 200000000 + s, "budget_agent_steps": 200000000,
                                                     "reached_budget": True, "wall_seconds": 60.0, "chunks": 4}))
            (d / "curves.json").write_text(json.dumps({"quick": False, "n_envs": 24, "episodes_per_env": 8, "git": git, "points": [
                {"tenth": k, "score": level[m] * k, "envs": val_random} for k in range(1, 11)]}))
            (d / "progress.jsonl").write_text("".join(json.dumps(
                {"steps": 2000, "births": 10 + c, "floor_births": 30, "deaths": 40, "steps_all_dead_before_floor": c % 2,
                 "agent_steps_total": 200000000 + s if c == 3 else c}) + "\n"
                for c in range(4)))
            envs = []
            for j, (name, i) in enumerate([(n, i) for n in ("T1", "T2", "T3") for i in range(10)]):
                c0 = random[j] + 10 * level[m] + rng.normal(0, 0.5)
                curve = [c0 + 0.1 * p for p in range(11)]
                envs.append({"set": name, "index": i, "zero_shot": {
                    "competence": curve[0], "random_competence": float(random[j]),
                    "persistence": {"extinct": bool(m.startswith("oe") and j % 3 == 0), "steps_survived": 4000.0 + j,
                                    "births": 3.0}},
                    "adaptation": {"curve": [{"competence": c, "agent_steps": p * 2_000_000 + 7} for p, c in enumerate(curve)],
                                   "chunks": [{"floor_births": 5, "births": 1, "steps_all_dead_before_floor": 0},
                                              {"floor_births": 2, "births": 3, "steps_all_dead_before_floor": 1}],
                                   "wall_seconds": 30.0}})
            (d / "eval_test.json").write_text(json.dumps({"split": "test", "envs": envs, "git": git,
                                                          "hardware": {"gpus": ["fake"]}, "evaluation_settings": mcfg["evaluation"]}))
    (tmp / "test_access_log.jsonl").write_text(json.dumps({"time": "t", "commit": "abc", "runs": ["a", "b"], "adapt": True}) + "\n")
    (tmp / "spend.json").write_text(json.dumps({"total": 30.5, "machine_hours": 34.0, "price_per_hour": 0.89}))


def _analyze():
    spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts/part3/analyze.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    return an


def _edit(path: Path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


def test_analyze_runs_end_to_end_on_a_fake_final_folder(tmp_path, monkeypatch):
    an = _analyze()
    _fake_final(tmp_path / "final")
    monkeypatch.setattr(an, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["analyze.py", "--final", str(tmp_path / "final"), "--reps", "200"])
    an.main()
    res = json.loads((tmp_path / "final" / "analysis.json").read_text())
    assert res["h1"]["verdict"] == "against" and res["runs"]["oe"] == list(range(5000, 5010))
    assert [r["name"] for r in res["secondary"]] == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    assert set(res["diversity_end"]) == {"oe", "oe_fixed"} and res["diversity_end"]["oe"]["weights"] > 0
    assert len(res["diversity_over_time"]["oe"]["behavior"]) == 10
    assert len(res["figures"]) == 8 and all((tmp_path / f["file"]).exists() for f in res["figures"])
    t = res["oe_population"]["oe"]["training_mean"]
    assert t["births"] == 46 and t["floor_births"] == 120 and t["steps_all_dead_before_floor"] == 2
    assert t["floor_births_per_1000_steps"] == pytest.approx(15.0) and t["floor_share_of_new_agents"] == pytest.approx(120 / 166)
    a = res["oe_population"]["oe_fixed"]["adaptation_mean_per_env_by_set"]
    assert a["floor_births"]["T2"] == 7.0 and a["births"]["T3"] == 4.0 and a["steps_all_dead_before_floor"]["T1"] == 1.0
    assert len(res["adaptation_curves"]["oe"]["T2"]) == 11 and res["compute"]["oe"]["adapt_final_agent_steps_mean"] == 20_000_007
    assert res["descriptive"]["es_dr"]["steps_survived_mean_by_set"]["T3"] == pytest.approx(4024.5)
    assert res["descriptive"]["oe"]["persistence_births_mean_by_set"]["T2"] == 3.0
    assert len(res["descriptive"]["oe"]["zero_shot_competence_mean_per_env"]) == 30
    assert len(res["per_run"]["ppo_dr"]) == 10 and res["per_run"]["ppo_dr"][0]["adapt_wall_hours"] == pytest.approx(900 / 3600)
    assert res["cost"]["total"] == 30.5 and len(res["test_access_log"]) == 1
    assert res["compute"]["oe"]["train_hardware"] == ['["train gpu"]'] and res["random_max_difference"]["test"] == 0.0
    report = (tmp_path / "docs/RESULTS_part3.md").read_text()
    sec = {r["name"]: r["result"] for r in res["secondary"]}
    for hyp, test, s_ in (("H2-OE", "S2", "S2"), ("H2-PPO", "S3", "S3"), ("H3", "S1", "S1"), ("H5", "S6", "S6"), ("H6", "S7", "S7")):
        row = next(x for x in report.splitlines() if x.startswith(f"| {hyp} |"))
        assert row.endswith(f"| {test} | {sec[s_]} |"), row
    assert next(x for x in report.splitlines() if x.startswith("| H1 |")).endswith(f"| {res['h1']['verdict']} |")
    for text in ("Verdict: against", "S7", "OE populations", "steps survived", "Looks at the test split", "$30.50",
                 "Diversity over training", "| OE | 5009 |"):
        assert text in report, text


def test_analyze_refuses_runs_that_are_not_comparable(tmp_path, monkeypatch):
    an = _analyze()
    f = tmp_path / "final"
    _fake_final(f)
    monkeypatch.setattr(an, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["analyze.py", "--final", str(f), "--reps", "50"])
    cases = [  # (file, edit, expected message)
        ("oe_seed5003/eval_test.json", lambda e: e["envs"][5]["zero_shot"].__setitem__("random_competence",
                                                                                     e["envs"][5]["zero_shot"]["random_competence"] + 1e-3), "random policy"),
        ("ppo_dr_seed5001/curves.json", lambda c: c["points"][0]["envs"][3].__setitem__("random_competence", 9.0), "random policy"),
        ("es_dr_seed5004/run.json", lambda r: r["methods_config"]["es"]["settings"].__setitem__("SIGMA", 0.5), "different settings"),
        ("oe_seed5000/run.json", lambda r: r["methods_config"].pop("tuned_from"), "tuned settings"),
        ("oe_seed5002/eval_test.json", lambda e: e["git"].__setitem__("clean", False), "uncommitted"),
        ("ppo_fixed_seed5006/curves.json", lambda c: c.__setitem__("quick", True), "training curve"),
        ("ppo_fixed_seed5006/curves.json", lambda c: c.__setitem__("episodes_per_env", 2), "training curve"),
        ("es_dr_seed5000/eval_test.json", lambda e: e["envs"][29]["adaptation"]["curve"][10].__setitem__("agent_steps", 19_999_999), "budget"),
        ("es_dr_seed5000/eval_test.json", lambda e: e["envs"][0]["adaptation"]["curve"][4].__setitem__("agent_steps", 100), "budget"),
        ("oe_fixed_seed5008/done.json", lambda d: d.__setitem__("reached_budget", False), "training did not reach"),
        ("oe_fixed_seed5008/eval_test.json", lambda e: e["evaluation_settings"].__setitem__("episodes_per_env", 4), "pre-registered ones"),
        ("oe_fixed_seed5008/eval_test.json", lambda e: e["evaluation_settings"].__setitem__("extra", 1), "different evaluation settings"),
        ("oe_seed5001/run.json", lambda r: r.__setitem__("seed", 5002), "folder name"),
        ("oe_seed5001/run.json", lambda r: r.__setitem__("method", "oe_fixed"), "folder name"),
        ("ppo_dr_seed5000/curves.json", lambda c: c["git"].__setitem__("clean", False), "own clean commit"),
        ("ppo_dr_seed5000/curves.json", lambda c: c["git"].__setitem__("commit", "zzz"), "own clean commit"),
        ("oe_seed5004/done.json", lambda d: d.__setitem__("chunks", 5), "progress.jsonl"),
        ("es_dr_seed5009/eval_test.json", lambda e: e["envs"][12]["adaptation"]["curve"][6].__setitem__("competence", float("nan")), "not finite"),
    ]
    for rel, fn, msg in cases:
        p = f / rel
        before = p.read_text()
        _edit(p, fn)
        with pytest.raises(SystemExit, match=msg):
            an.main()
        p.write_text(before)
    # one run trained (and its curve scored) from another commit
    saved = {n: (f / "oe_seed5002" / n).read_text() for n in ("run.json", "curves.json")}
    for n in saved:
        _edit(f / "oe_seed5002" / n, lambda r: r["git"].__setitem__("commit", "def"))
    with pytest.raises(SystemExit, match="more than one commit"):
        an.main()
    for n, t in saved.items():
        (f / "oe_seed5002" / n).write_text(t)
    # a last-digit difference in the random policy's competence is allowed (and reported)
    _edit(f / "oe_seed5003/eval_test.json", lambda e: e["envs"][5]["zero_shot"].__setitem__(
        "random_competence", e["envs"][5]["zero_shot"]["random_competence"] + 1e-9))
    an.main()
    res = json.loads((f / "analysis.json").read_text())
    assert 0 < res["random_max_difference"]["test"] < 1e-8


def test_load_runs_replaces_failed_runs_with_the_next_seed_and_stops_past_20_percent(tmp_path):
    an = _analyze()
    f = tmp_path / "final"
    for m in ("oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr"):
        for s in range(5000, 5013):
            (f / f"{m}_seed{s}").mkdir(parents=True)
            (f / f"{m}_seed{s}" / "done.json").write_text("{}")
    def fail(m, s):
        (f / f"{m}_seed{s}" / "done.json").unlink(missing_ok=True)
        (f / f"{m}_seed{s}" / "failed.json").write_text(json.dumps({"reason": "parameters not finite"}))
    fail("oe", 5003)
    out = an.load_runs(f)
    assert [int(d.name.split("seed")[1]) for d in out["runs"]["oe"]] == [5000, 5001, 5002] + list(range(5004, 5011))
    assert out["excluded"] == [{"method": "oe", "seed": 5003, "reason": "parameters not finite"}]
    assert [int(d.name.split("seed")[1]) for d in out["runs"]["ppo_dr"]] == list(range(5000, 5010))
    (f / "oe_seed5010" / "done.json").unlink()                 # the replacement is missing: no skipping to 5011
    with pytest.raises(SystemExit, match="next unused final seed"):
        an.load_runs(f)
    (f / "oe_seed5010" / "done.json").write_text("{}")
    fail("es_dr", 5001); fail("es_dr", 5002)                   # 2 of 10 failed: still allowed
    assert len(an.load_runs(f)["runs"]["es_dr"]) == 10
    fail("es_dr", 5011)                                         # a third failure: more than 20%
    with pytest.raises(SystemExit, match="more than 20%"):
        an.load_runs(f)
    (f / "ppo_fixed_seed5007" / "done.json").unlink()          # a planned run unfinished
    with pytest.raises(SystemExit, match="missing or unfinished"):
        an.load_runs(f)


def test_analyze_requires_the_preregistered_evaluation_settings(tmp_path, monkeypatch):
    an = _analyze()
    f = tmp_path / "final"
    _fake_final(f)
    for d in f.glob("*_seed*"):                       # every run agrees, but on 4 episodes instead of 8
        _edit(d / "eval_test.json", lambda e: e["evaluation_settings"].__setitem__("episodes_per_env", 4))
    monkeypatch.setattr(an, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["analyze.py", "--final", str(f), "--reps", "50"])
    with pytest.raises(SystemExit, match="pre-registered ones"):
        an.main()
