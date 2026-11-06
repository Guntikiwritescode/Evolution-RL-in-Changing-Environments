"""Tests for the Part 3 world, environment splits and seeds (SPEC_part3.md, Section 10).

Run with the Part 3 environment:  venv3/bin/python -m pytest tests/test_part3_env.py
Seeds used here: 3900-3999 (the "test" range).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evo2.env import splits
from evo2.env.generator import env_from_seed, params_to_dict, ranges_from_dict, sample_env_params
from evo2.env.world import World, make_env_params
from evo2.seeds import SEED_RANGES, STREAMS, base_key, check_seed, stream_key

ROOT = Path(__file__).resolve().parents[1]
EPI = splits.load_world_config("episodic")
ECO = splits.load_world_config("ecological")
RANGES, CANON = splits.load_train()


def key(i=0):
    return base_key("test", 3900 + i)


def default_traits(w):
    return jnp.tile(w.default_traits, (w.cfg.n_slots, 1))


def run_steps(w, s, params, k, n_steps, actions=None):
    """Run n_steps with random actions; return final state and per-step info."""
    infos = []
    for kk in jax.random.split(k, n_steps):
        ka, ks = jax.random.split(kk)
        a = jax.random.normal(ka, (w.cfg.n_slots, 2)) if actions is None else actions
        before = s
        s, info = w.step(ks, s, params, a)
        infos.append((before, s, info))
    return s, infos


# ----- conservation ------------------------------------------------------

@pytest.mark.parametrize("cfg", [EPI, ECO], ids=["episodic", "ecological"])
def test_energy_conservation_every_step(cfg):
    w = World(cfg)
    params = make_env_params(L=30.0, n_food=60, layout="uniform", rho=0.2)
    s = w.reset(key(1), params, default_traits(w), cfg.n_slots // 2)
    # rich agents, so births happen in ecological mode; some nearly starved agents and one
    # at the age limit, so deaths (and their removed energy) happen too
    s = s.replace(energy=jnp.where(s.alive, 230.0, 0.0).at[0:3].set(1.0).at[3].set(150.0),
                  age=s.age.at[3].set(ECO.max_age - 1),
                  pos=s.pos.at[0:3].set(0.0),                       # starving agents in a corner,
                  food_pos=jnp.clip(s.food_pos, 5.0, 30.0))         # out of reach of every item
    _, infos = run_steps(w, s, params, key(2), 60)
    total_births, total_deaths = 0, 0
    for before, after, info in infos:
        e0 = float(jnp.sum(jnp.where(before.alive, before.energy, 0.0)))
        e1 = float(jnp.sum(jnp.where(after.alive, after.energy, 0.0)))
        expected = e0 + float(info["food_gain"]) - float(info["cost"]) \
            - float(info["removed_at_death"]) - float(info["birth_overhead"])
        assert e1 == pytest.approx(expected, abs=1e-2), (e0, e1, expected)
        assert float(info["food_gain"]) == pytest.approx(cfg.food_energy * int(info["n_eat"].sum()))
        total_births += int(info["n_births"])
        total_deaths += int(info["died"].sum())
        assert not bool(jnp.any(after.energy[~after.alive] != 0.0))
    assert total_deaths >= 3
    if cfg.ecological:
        assert total_births > 0
    else:
        assert total_births == 0


def test_food_conservation_without_regrowth():
    w = World(EPI)
    params = make_env_params(L=20.0, n_food=60, layout="uniform", rho=0.0)
    s = w.reset(key(3), params, default_traits(w), 16)
    _, infos = run_steps(w, s, params, key(4), 80)
    eaten_total = 0
    for before, after, info in infos:
        gone = int(before.food_present.sum()) - int(after.food_present.sum())
        assert gone == int(info["n_eat"].sum())
        eaten_total += gone
    assert eaten_total > 0


def test_one_item_eaten_once_and_ties_random():
    w = World(EPI)
    params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
    winners = set()
    for i in range(12):
        s = w.reset(key(5 + i), params, default_traits(w), 2)
        s = s.replace(pos=s.pos.at[0].set(jnp.array([30.0, 30.0])).at[1].set(jnp.array([30.0, 30.5])),
                      food_pos=s.food_pos.at[0].set(jnp.array([30.0, 30.2])))
        stay = jnp.full((16, 2), -50.0)   # speed ~ 0
        s2, info = w.step(key(40 + i), s, params, stay)
        assert int(info["n_eat"].sum()) == 1
        assert not bool(s2.food_present[0])
        winners.add(int(jnp.argmax(info["n_eat"])))
    assert winners == {0, 1}


def test_inactive_food_slots_never_present():
    w = World(EPI)
    params = make_env_params(L=40.0, n_food=10, layout="clusters", k=3, rho=1.0)
    s = w.reset(key(6), params, default_traits(w), 16)
    s, _ = run_steps(w, s, params, key(7), 30)
    assert not bool(jnp.any(s.food_present[10:]))
    assert int(s.food_present[:10].sum()) == 10  # rho = 1: everything regrows


# ----- determinism -------------------------------------------------------

def test_determinism_same_key_same_result():
    w = World(ECO)
    params = sample_env_params(key(8), RANGES)

    def go(k):
        s = w.reset(k, params, default_traits(w), 32)
        s = s.replace(energy=jnp.where(s.alive, 210.0, 0.0))
        s, _ = run_steps(w, s, params, k, 25)
        return s

    a, b, c = go(key(9)), go(key(9)), go(key(10))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert not np.array_equal(np.asarray(a.pos), np.asarray(c.pos))


# ----- bounds ------------------------------------------------------------

def test_action_and_trait_bounds_hold():
    w = World(EPI)
    wild = jnp.array([[1e6, 1e6], [-1e6, -1e6], [jnp.inf, -jnp.inf]] + [[0.0, 0.0]] * 13)
    traits = jnp.tile(jnp.array([[5.0, -1.0, 100.0]]), (16, 1))
    s = w.reset(key(11), CANON, traits, 16)
    assert np.allclose(s.traits, np.array([2.0, 0.5, 20.0]))
    turn, speed, a = w.squash(wild, s.traits)
    assert bool(jnp.all(jnp.abs(turn) <= math.pi / 4 + 1e-6))
    assert bool(jnp.all((speed >= 0.0) & (speed <= s.traits[:, 0] + 1e-6)))
    assert bool(jnp.all(jnp.abs(a) <= 1.0))
    s2, _ = w.step(key(12), s, CANON, wild)
    assert bool(jnp.all((s2.pos >= 0.0) & (s2.pos <= CANON.L)))


def test_energy_cost_formula():
    w = World(EPI)
    traits = jnp.array([[1.0, 1.0, 10.0], [2.0, 2.0, 20.0], [0.25, 0.5, 4.0]])
    speed = jnp.array([1.0, 2.0, 0.0])
    cost = w.step_cost(traits, speed)
    assert np.allclose(cost, [2.0, 0.5 + 8.0 * 4.0 + 1.0, 0.5 + 0.2])


# ----- sensors -----------------------------------------------------------

def test_sensors_hand_computed():
    w = World(EPI)
    params = make_env_params(L=60.0, n_food=2, layout="uniform", rho=0.0)
    s = w.reset(key(13), params, default_traits(w), 2)
    s = s.replace(
        pos=s.pos.at[0].set(jnp.array([5.0, 10.0])).at[1].set(jnp.array([5.0, 7.0])),
        heading=s.heading.at[0].set(0.0),
        food_pos=s.food_pos.at[0].set(jnp.array([10.0, 10.0])).at[1].set(jnp.array([5.0, 14.0])),
        energy=s.energy.at[0].set(150.0),
        prev_action=s.prev_action.at[0].set(jnp.array([0.5, -0.25])),
    )
    o = np.asarray(w.observe(s, params)[0])
    food, agents, walls = o[0:8], o[8:16], o[16:24]
    # food dead ahead at distance 5 -> sector 0 = 0.5; food 4 to the left -> sector 2 = 0.6
    assert food[0] == pytest.approx(0.5, abs=1e-5) and food[2] == pytest.approx(0.6, abs=1e-5)
    assert np.count_nonzero(food) == 2
    # agent 3 to the right (heading 0, so -90 degrees) -> sector 6 = 0.7
    assert agents[6] == pytest.approx(0.7, abs=1e-5) and np.count_nonzero(agents) == 1
    # walls from (5, 10): behind (sector 4) at 5 -> 0.5; sectors 3 and 5 (135, 225 deg) reach x=0
    # after 5*sqrt2 -> 1 - 0.7071; right (sector 6) at exactly 10 = range -> 0; the rest beyond range
    assert walls[4] == pytest.approx(0.5, abs=1e-5)
    assert walls[3] == pytest.approx(1 - 5 * math.sqrt(2) / 10, abs=1e-5)
    assert walls[5] == pytest.approx(1 - 5 * math.sqrt(2) / 10, abs=1e-5)
    assert np.count_nonzero(walls) == 3
    # from (5, 5), sector 5 reaches both walls after 5*sqrt2
    s3 = s.replace(pos=s.pos.at[0].set(jnp.array([5.0, 5.0])))
    walls3 = np.asarray(w.observe(s3, params)[0])[16:24]
    assert walls3[5] == pytest.approx(1 - 5 * math.sqrt(2) / 10, abs=1e-5)
    # energy, previous action, scaled traits
    assert o[24] == pytest.approx(1.5)
    assert np.allclose(o[25:27], [0.5, -0.25])
    assert np.allclose(o[27:30], [(1.0 - 0.25) / 1.75, 0.5 / 1.5, 6.0 / 16.0])
    # dead slots read zeros
    assert np.all(np.asarray(w.observe(s, params)[2:]) == 0.0)
    # sector edges: sector 0 covers -22.5 to 22.5 degrees around the heading, so an item
    # 30 degrees to the left is in sector 1 and one 20 degrees to the right is in sector 0
    def at(deg, d):
        r = math.radians(deg)
        return jnp.array([5.0 + d * math.cos(r), 10.0 + d * math.sin(r)])
    s4 = s.replace(food_pos=s.food_pos.at[0].set(at(30.0, 5.0)).at[1].set(at(-20.0, 2.0)))
    food4 = np.asarray(w.observe(s4, params)[0])[0:8]
    assert food4[1] == pytest.approx(0.5, abs=1e-5) and food4[0] == pytest.approx(0.8, abs=1e-5)
    assert np.count_nonzero(food4) == 2


# ----- ecological mode ---------------------------------------------------

def test_births_rules():
    w = World(ECO)
    params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
    traits = default_traits(w).at[0].set(jnp.array([1.5, 0.7, 12.0]))
    s = w.reset(key(14), params, traits, 3)
    s = s.replace(energy=s.energy.at[0].set(250.0).at[1].set(150.0).at[2].set(199.0),
                  pos=s.pos.at[0].set(jnp.array([30.0, 30.0])),
                  food_pos=s.food_pos.at[0].set(jnp.array([0.0, 0.0])))
    still = jnp.full((128, 2), -50.0)
    s2, info = w.step(key(15), s, params, still)
    assert int(info["n_births"]) == 1
    child = int(jnp.argmax(info["parent_idx"] >= 0))
    assert int(info["parent_idx"][child]) == 0 and child == 3  # first free slot
    cost0 = float(w.step_cost(s.traits[0:1], jnp.zeros(1))[0])
    assert float(s2.energy[0]) == pytest.approx(250.0 - cost0 - 120.0, abs=1e-4)
    assert float(s2.energy[child]) == pytest.approx(100.0)
    assert np.allclose(s2.traits[child], s2.traits[0])
    assert float(jnp.linalg.norm(s2.pos[child] - s2.pos[0])) == pytest.approx(1.0, abs=1e-4)
    assert int(s2.age[child]) == 0 and bool(s2.alive[child])


def test_no_births_when_full_parent_keeps_energy():
    w = World(ECO)
    params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
    s = w.reset(key(16), params, default_traits(w), 128)
    s = s.replace(energy=jnp.full((128,), 300.0), food_pos=s.food_pos.at[0].set(jnp.array([0.0, 0.0])))
    s = s.replace(pos=jnp.full((128, 2), 30.0))
    still = jnp.full((128, 2), -50.0)
    s2, info = w.step(key(17), s, params, still)
    assert int(info["n_births"]) == 0
    cost = w.step_cost(s.traits, jnp.zeros(128))
    assert np.allclose(s2.energy, 300.0 - cost, atol=1e-4)


def test_births_limited_by_free_slots():
    w = World(ECO)
    params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
    s = w.reset(key(18), params, default_traits(w), 125)  # 3 free slots
    s = s.replace(energy=jnp.where(s.alive, 300.0, 0.0), pos=jnp.full((128, 2), 30.0),
                  food_pos=s.food_pos.at[0].set(jnp.array([0.0, 0.0])))
    still = jnp.full((128, 2), -50.0)
    s2, info = w.step(key(19), s, params, still)
    assert int(info["n_births"]) == 3 and int(s2.alive.sum()) == 128
    cost = float(w.step_cost(s.traits[:1], jnp.zeros(1))[0])
    paid = np.isclose(np.asarray(s2.energy[:125]), 300.0 - cost - 120.0, atol=1e-3)
    assert int(paid.sum()) == 3


def test_death_by_age_only_in_ecological_mode():
    for cfg, should_die in [(ECO, True), (EPI, False)]:
        w = World(cfg)
        params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
        s = w.reset(key(20), params, default_traits(w), 1)
        s = s.replace(age=s.age.at[0].set(ECO.max_age - 1))
        s2, info = w.step(key(21), s, params, jnp.full((cfg.n_slots, 2), -50.0))
        assert bool(s2.alive[0]) is (not should_die)


# ----- food layouts and dynamics ------------------------------------------

def test_layouts_place_food_where_expected():
    w = World(EPI)
    for layout in ("uniform", "clusters", "patches"):
        params = make_env_params(L=60.0, n_food=60, layout=layout, k=3, sigma=2.0, patch_r=4.0)
        s = w.reset(key(22), params, default_traits(w), 16)
        fp = np.asarray(s.food_pos[:60])
        assert np.all((fp >= 0) & (fp <= 60.0))
        c = np.asarray(s.centres[:3])
        nearest = np.min(np.linalg.norm(fp[:, None, :] - c[None, :, :], axis=-1), axis=1)
        if layout == "patches":
            assert np.all(nearest <= 4.0 + 1e-4)
        if layout == "clusters":
            assert np.mean(nearest) < 4.0
        if layout == "uniform":
            assert np.mean(nearest) > 8.0
    cent = np.asarray(s.centres)
    assert np.all((cent >= 0.15 * 60 - 1e-4) & (cent <= 0.85 * 60 + 1e-4))


def test_drift_moves_food_and_bounces():
    w = World(EPI)
    params = make_env_params(L=20.0, n_food=64, layout="uniform", rho=0.0, drift=0.3)
    s = w.reset(key(23), params, default_traits(w), 1)
    # the one agent sits in a corner, out of reach of every item
    s = s.replace(pos=jnp.zeros_like(s.pos), food_pos=jnp.clip(s.food_pos, 2.0, 18.0))
    s2, _ = w.step(key(24), s, params, jnp.full((16, 2), -50.0))
    moved = np.linalg.norm(np.asarray(s2.food_pos - s.food_pos), axis=1)
    present = np.asarray(s.food_present & s2.food_present)
    assert np.allclose(moved[present], 0.3, atol=1e-4)
    s3, _ = run_steps(w, s, params, key(25), 200, actions=jnp.full((16, 2), -50.0))
    assert np.all((np.asarray(s3.food_pos) >= 0.0) & (np.asarray(s3.food_pos) <= 20.0))


def test_season_moves_centres_only_on_period():
    w = World(EPI)
    params = make_env_params(L=60.0, n_food=40, layout="clusters", k=3, rho=0.0, season_P=5)
    s = w.reset(key(26), params, default_traits(w), 1)
    changes = []
    for i, kk in enumerate(jax.random.split(key(27), 12)):
        s2, _ = w.step(kk, s, params, jnp.full((16, 2), -50.0))
        changes.append(not np.allclose(s2.centres, s.centres))
        s = s2
    assert changes == [(i + 1) % 5 == 0 for i in range(12)]


def test_regrowth_probability_extremes():
    w = World(EPI)
    for rho, regrows in [(0.0, False), (1.0, True)]:
        params = make_env_params(L=60.0, n_food=10, layout="uniform", rho=rho)
        s = w.reset(key(28), params, default_traits(w), 1)
        s = s.replace(food_present=s.food_present.at[:5].set(False))
        s2, _ = w.step(key(29), s, params, jnp.full((16, 2), -50.0))
        assert bool(jnp.all(s2.food_present[:5])) is regrows


def test_change_env_clips_agents_and_lays_out_food():
    w = World(ECO)
    big = make_env_params(L=80.0, n_food=60, layout="uniform")
    small = make_env_params(L=40.0, n_food=20, layout="patches", k=2, patch_r=5.0)
    s = w.reset(key(30), big, default_traits(w), 32)
    s2 = w.change_env(key(31), s, small)
    assert bool(jnp.all(s2.pos <= 40.0))
    assert int(s2.food_present.sum()) == 20
    assert np.array_equal(np.asarray(s2.energy), np.asarray(s.energy))


# ----- splits and the test lock ------------------------------------------

def test_train_ranges_and_canonical_match_spec():
    d = params_to_dict(CANON)
    assert (d["L"], d["n_food"], d["layout"], d["drift"], d["season_P"]) == (60.0, 40, 0, 0.0, 0)
    assert d["rho"] == pytest.approx(0.03)
    draws = [params_to_dict(sample_env_params(k, RANGES)) for k in jax.random.split(key(32), 300)]
    assert all(40.0 <= x["L"] <= 80.0 and 20 <= x["n_food"] <= 60 and 2 <= x["k"] <= 5 for x in draws)
    assert all(0.01 <= x["rho"] <= 0.05 and x["drift"] == 0.0 and x["season_P"] == 0 for x in draws)
    assert {x["layout"] for x in draws} == {0, 1, 2}
    assert {x["n_food"] for x in draws} >= {20, 60}  # inclusive ends are reachable


def test_validation_set():
    v = splits.load_validation()
    assert len(v["drawn"]) == 20 and len(v["corners"]) == 4
    assert v["episode_seeds"][0] == 4930 and v["episode_seeds"][-1] == 4999
    again = splits.load_validation()
    assert params_to_dict(v["drawn"][7]) == params_to_dict(again["drawn"][7])


def test_test_split_locked(tmp_path):
    (tmp_path / "test_split.yaml").write_text("T1: {}\n")
    with pytest.raises(splits.TestSplitLocked):
        splits.load_test_split(config_dir=tmp_path)
    with pytest.raises(splits.TestSplitLocked):
        splits.load_test_split(final=1, config_dir=tmp_path)
    with pytest.raises(splits.TestSplitLocked):
        splits.load_test_split(final="yes", config_dir=tmp_path)


def test_test_split_file_parses_with_final():
    """Reads the real test file only to check that it parses and matches the spec.
    No environment from it is run here."""
    t = splits.load_test_split(final=True)
    assert [len(t[n]) for n in ("T1", "T2", "T3")] == [10, 10, 10]
    t2 = [params_to_dict(p) for p in t["T2"]]
    assert all(30.0 <= x["L"] <= 40.0 and 10 <= x["n_food"] <= 20 and x["layout"] == 1 for x in t2)
    t3 = [params_to_dict(p) for p in t["T3"]]
    assert all(0.05 <= x["drift"] <= 0.3 and 100 <= x["season_P"] <= 300 for x in t3)


def test_only_evaluate_may_name_the_test_split():
    """No code except splits.py and scripts/part3/evaluate.py names the test file
    or calls load_test_split."""
    # break_tests.py only names them inside the text of its deliberate breaks
    allowed = {ROOT / "src/evo2/env/splits.py", ROOT / "scripts/part3/evaluate.py",
               ROOT / "scripts/part3/break_tests.py"}
    offenders = []
    for folder in (ROOT / "src/evo2", ROOT / "scripts/part3"):
        for f in folder.rglob("*.py"):
            text = f.read_text()
            if re.search(r"test_split|load_test_split", text) and f not in allowed:
                offenders.append(str(f.relative_to(ROOT)))
    assert offenders == []
    for name in ("pilot.py", "tune.py", "train.py"):
        f = ROOT / "scripts/part3" / name
        if f.exists():
            assert "--final" not in f.read_text() and "final=True" not in f.read_text()


# ----- seeds -------------------------------------------------------------

def test_seed_ranges_do_not_overlap_and_match_plan():
    spans = sorted(SEED_RANGES.values())
    for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
        assert a_hi < b_lo
    # plan: pilots 3000-3999, tuning 4000-4999, final 5000-5999
    for name in ("pilot", "pilot_eval", "test"):
        lo, hi = SEED_RANGES[name]
        assert 3000 <= lo <= hi <= 3999
    for name in ("tuning", "stats", "validation"):
        lo, hi = SEED_RANGES[name]
        assert 4000 <= lo <= hi <= 4999
    for name in ("final", "test_env"):
        lo, hi = SEED_RANGES[name]
        assert 5000 <= lo <= hi <= 5999


def test_check_seed_rejects_wrong_purpose():
    assert check_seed("final", 5000) == 5000
    with pytest.raises(ValueError):
        check_seed("final", 4999)
    with pytest.raises(ValueError):
        check_seed("tuning", 4900)  # validation seed
    with pytest.raises(ValueError):
        env_from_seed("validation", 5900, RANGES)


def test_stream_keys_serve_one_purpose_each():
    assert len(set(STREAMS.values())) == len(STREAMS)
    k = key(33)
    keys = {name: tuple(np.asarray(stream_key(k, name)).ravel()) for name in STREAMS}
    assert len(set(keys.values())) == len(STREAMS)
    draws = {name: float(jax.random.uniform(stream_key(k, name))) for name in STREAMS}
    assert len(set(draws.values())) == len(STREAMS)
    with pytest.raises(ValueError):
        stream_key(k, "no_such_stream")


def test_validation_and_test_ranges_parse_like_train():
    r = ranges_from_dict({"L": [1.0, 2.0], "F": [3, 3], "layouts": ["patches"], "k": [2, 2],
                          "sigma": [1.0, 1.0], "patch_r": [1.0, 1.0], "rho": [0.0, 0.0],
                          "drift": [0.0, 0.0], "season_P": [0, 0]})
    p = params_to_dict(sample_env_params(key(34), r))
    assert p["n_food"] == 3 and p["layout"] == 2
    good = {"L": [1.0, 2.0], "F": [3, 3], "layouts": ["patches"], "k": [2, 2], "sigma": [1.0, 1.0],
            "patch_r": [1.0, 1.0], "rho": [0.0, 0.0], "drift": [0.0, 0.0], "season_P": [0, 0]}
    with pytest.raises(ValueError):
        ranges_from_dict({**good, "L": [2.0, 1.0]})       # low above high
    with pytest.raises(ValueError):
        ranges_from_dict({k: v for k, v in good.items() if k != "rho"})   # missing key
    with pytest.raises(ValueError):
        ranges_from_dict({**good, "layouts": ["spiral"]})


def test_part3_libraries_import():
    """Every library pinned for Part 3 that the code uses must import (MISTAKES #41)."""
    import importlib
    for name in ("jax", "jaxlib", "flax", "optax", "distrax", "chex", "evosax", "gymnax",
                 "rliable.library", "rliable.metrics", "arch", "pandas", "scipy", "numpy", "yaml"):
        importlib.import_module(name)


def test_pilot_shift_copies_test_ranges_with_pilot_seeds():
    """The pilot's stand-ins use exactly the T2 and T3 ranges, but new environments
    drawn with pilot-evaluation seeds (none is a test environment)."""
    import yaml
    with open(ROOT / "configs/part3/pilot_shift.yaml") as fh:
        ps = yaml.safe_load(fh)
    t = splits._read("test_split.yaml")
    assert ps["P2"]["ranges"] == t["T2"]["ranges"] and ps["P3"]["ranges"] == t["T3"]["ranges"]
    p = splits.load_pilot_shift()
    assert len(p["P2"]) == 5 and len(p["P3"]) == 5
    for name in ("P2", "P3"):
        for s_ in range(ps[name]["env_seeds"][0], ps[name]["env_seeds"][1] + 1):
            check_seed("pilot_eval", s_)
    for s_ in p["episode_seeds"]:
        check_seed("pilot_eval", s_)
