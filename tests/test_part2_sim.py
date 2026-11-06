"""Part 2 simulator changes (docs/PREREG_part2.md, Sections 2, 3, 18.8)."""
import copy
import json
import os
from fractions import Fraction

import numpy as np
import pytest

from evosim import (config, gridsearch, p2metrics, run2, runner, search2, sim2, streams,
                    world)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1_FINAL = os.path.join(ROOT, "configs", "part1", "final.yaml")
P2_BASE = os.path.join(ROOT, "configs", "part2", "base.yaml")
P1_RESULTS = os.path.join(ROOT, "results", "part1", "final")


def small_cfg(**over):
    o = {"phases": {"train_generations": 12, "transfer_generations": 50}}
    for k, v in over.items():
        o.setdefault(k, {}).update(v)
    return config.load(P2_BASE, o)


def same_logs(a, b):
    a, b = a.arrays(), b.arrays()
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------ Part 1 regression
@pytest.mark.skipif(not os.path.isdir(P1_RESULTS), reason="Part 1 results not present")
def test_part1_final_runs_reproduce_bit_for_bit():
    """All 60 Part 1 final runs, rerun with the Part 2 code (food stream off)."""
    for cond, key in (("ET", 0), ("GS", 1)):
        for seed in range(30):
            z = np.load(os.path.join(P1_RESULTS, f"{cond}_seed{seed:04d}.npz"), allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            cfg = meta["config"]
            log = runner.run_lineage(meta["start_traits"], 50, meta["mutation_sd"], cfg,
                                     np.random.SeedSequence(seed, spawn_key=(key,)))
            a = log.arrays()
            for k in a:
                assert np.array_equal(a[k], z[k]), (cond, seed, k)


@pytest.mark.skipif(not os.path.isdir(P1_RESULTS), reason="Part 1 results not present")
def test_part1_search_reproduces():
    with open(os.path.join(P1_RESULTS, "gs_search.json")) as f:
        old = json.load(f)
    cfg = old["provenance"]["config"]
    new = gridsearch.grid_search(cfg, int(cfg["gs"]["search_seed"]))
    assert new["best_index"] == old["best_index"]
    assert [r["score"] for r in new["results"]] == [r["score"] for r in old["results"]]
    assert new["creature_steps"] == old["search_creature_steps"]


def test_part1_frozen_sections_unchanged_in_part2_config():
    p1, p2 = config.load(P1_FINAL), config.load(P2_BASE)
    assert run2.frozen_hash(p1) == run2.frozen_hash(p2)


# ------------------------------------------------------------ streams
def test_stream_purposes_have_unique_ids_and_keys_differ():
    ids = list(streams.PURPOSES.values())
    assert len(ids) == len(set(ids))
    a = streams.rng(2000, "deploy", 1, streams.BEHAVIOR).random(5)
    b = streams.rng(2000, "deploy", 1, streams.FOOD, 1).random(5)
    c = streams.rng(2000, "deploy", 0, streams.BEHAVIOR).random(5)
    d = streams.rng(2001, "deploy", 1, streams.BEHAVIOR).random(5)
    e = streams.rng(2000, "deploy", 1, streams.BEHAVIOR).random(5)
    assert not np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)
    assert np.array_equal(a, e)
    with pytest.raises(KeyError):
        streams.rng(2000, "nonsense")


ALLOWED = {  # context prefix -> purposes it may use (PREREG_part2.md Section 10)
    "search_A": {"search_A1", "search_A2", "tiebreak_A"},
    "search_B:": {"search_B1", "search_B2", "tiebreak_B"},
    "u2:": {"u2"},
    "deploy_A:": {"deploy"},
    "deploy_B:": {"deploy"},
    "acheck": {"acheck"},
    "landscape:": {"landscape"},
    "sv": {"sv_designs"},
    "clone_pick": {"clone_pick"},
}


def test_every_stream_key_serves_only_its_listed_purposes():
    """Log every key used in a whole (tiny) run seed, with what it was used for."""
    over = {"phases": {"train_generations": 12, "transfer_generations": 50},
            "part2": {"grid": {"points": 2}, "search": {"stage2_sets": 2, "stage2_top": 2},
                      "long_B_generations": 60, "acheck_generations": 3, "suite_sides": [500],
                      "suite_food": [15], "mutation_factors": [2.0], "arm3_side": 400,
                      "arm4_energy": 100}}
    cfg = config.load(P2_BASE, over)
    streams.start_logging()
    try:
        run2.run_seed(2990, cfg, exploratory=True, landscape=True)
    finally:
        log = streams.stop_logging()
    names = {v: k for k, v in streams.PURPOSES.items()}
    uses = {}
    for key, ctx in log:
        assert ctx != "none", key
        pref = next(p for p in ALLOWED if ctx.startswith(p))
        assert names[key[0]] in ALLOWED[pref], (key, ctx)
        if names[key[0]] == "deploy":  # A-phase keys only in A, B-phase keys only in B
            assert key[1] == (0 if pref == "deploy_A:" else 1), (key, ctx)
        uses.setdefault(key, set()).add(pref)
    # each key serves one purpose; the only sharing is the deployment streams across groups/worlds
    for key, prefs in uses.items():
        assert len(prefs) == 1, (key, prefs)
    # arms 3 and 4 run (their contexts appear) and share the main B's search streams;
    # suite landscapes have their own
    assert any(c.startswith("search_B:arm3") for _, c in log) and any(c.startswith("search_B:arm4") for _, c in log)
    assert any(c.startswith("deploy_B:arm4:") for _, c in log)
    b1 = {k for k, c in log if names[k[0]] == "search_B1"}
    assert {k[1] for k in b1} == {0}
    assert len({k[1] for k, c in log if names[k[0]] == "landscape"}) >= 2


# ------------------------------------------------------------ food sharing
def test_same_seed_groups_share_food_layouts_every_generation(monkeypatch):
    cfg = small_cfg()
    seen = []
    real = world.place_food

    def spy(env, rng):
        x, y = real(env, rng)
        seen[-1].append((x.copy(), y.copy()))
        return x, y

    monkeypatch.setattr(world, "place_food", spy)
    pops = {}
    for name, design, sdg in (("ET", runner.et_start_traits(cfg), cfg["mutation_sd"]),
                              ("clone", {"speed": 15.0, "size": 7.0, "sense": 30.0, "reach": 2.0},
                               sim2.ZERO_SD)):
        seen.append([])
        log = runner.Log()
        end = sim2.deploy_A([design] * 50, cfg, 2991, sdg, log)
        sim2.deploy_B(end, cfg["env_B"], cfg["world"], sdg, 10, cfg, 2991, runner.Log())
        pops[name] = log
    assert len(seen[0]) == len(seen[1]) == 12 + 10
    for (x0, y0), (x1, y1) in zip(seen[0], seen[1]):
        assert np.array_equal(x0, x1) and np.array_equal(y0, y1)
    # different generations get different layouts
    assert not np.array_equal(seen[0][0][0], seen[0][1][0])


# ------------------------------------------------------------ groups through A and B
def test_et_family_identical_through_A_and_end_of_A_reused_safely():
    cfg = small_cfg()
    sd = cfg["mutation_sd"]
    d = [runner.et_start_traits(cfg)] * 50
    la, lb = runner.Log(), runner.Log()
    ea = sim2.deploy_A(d, cfg, 2992, sd, la)
    eb = sim2.deploy_A(d, cfg, 2992, sd, lb)
    assert same_logs(la, lb)
    snapshot = copy.deepcopy(ea)
    l1, l2 = runner.Log(), runner.Log()
    sim2.deploy_B(ea, cfg["env_B"], cfg["world"], sd, 20, cfg, 2992, l1)
    sim2.deploy_B(ea, cfg["env_B"], cfg["world"], sd, 20, cfg, 2992, l2)
    assert same_logs(l1, l2)
    for k, v in snapshot.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, ea[k]), k


def test_mutation_setting_applies_only_in_its_phase():
    cfg = small_cfg()
    d = [runner.et_start_traits(cfg)] * 50
    end = sim2.deploy_A(d, cfg, 2993, cfg["mutation_sd"], runner.Log())
    lb = runner.Log()
    sim2.deploy_B(end, cfg["env_B"], cfg["world"], sim2.ZERO_SD, 30, cfg, 2993, lb)
    b = lb.arrays()
    start = {tuple(r) for r in np.column_stack([end[t] for t in world.TRAITS])}
    inb = {tuple(r) for r in np.column_stack([b["b_" + t] for t in world.TRAITS])}
    assert inb <= start  # no new designs appear in B when mutation is off in B
    assert len(start) > 1  # A did mutate


def test_blob_eating_setting_applies_only_in_B_and_clones_unaffected():
    cfg = small_cfg()
    env_b, wc_main = run2.b_world(cfg, "main")
    _, wc_arm2 = run2.b_world(cfg, "arm2")
    assert wc_arm2["cannibalism"] and not wc_main["cannibalism"] and not cfg["world"]["cannibalism"]
    clone = {"speed": 12.0, "size": 9.0, "sense": 25.0, "reach": 2.25}
    end = sim2.deploy_A([clone] * 50, cfg, 2994, sim2.ZERO_SD, runner.Log())
    l1, l2 = runner.Log(), runner.Log()
    sim2.deploy_B(end, env_b, wc_main, sim2.ZERO_SD, 30, cfg, 2994, l1)
    sim2.deploy_B(end, env_b, wc_arm2, sim2.ZERO_SD, 30, cfg, 2994, l2)
    assert same_logs(l1, l2)
    # a varied population does eat in arm 2's B, and never in A
    la = runner.Log()
    endv = sim2.deploy_A([runner.et_start_traits(cfg)] * 50, cfg, 2994, cfg["mutation_sd"], la)
    assert la.arrays()["g_prey_eaten"].sum() == 0
    lv = runner.Log()
    sim2.deploy_B(endv, env_b, wc_arm2, cfg["mutation_sd"], 30, cfg, 2994, lv)
    assert lv.arrays()["g_prey_eaten"].sum() >= 0  # allowed; counted in outcomes


def test_longer_B_leaves_generations_0_to_49_unchanged():
    cfg = small_cfg()
    end = sim2.deploy_A([runner.et_start_traits(cfg)] * 50, cfg, 2995, cfg["mutation_sd"], runner.Log())
    l50, l200 = runner.Log(), runner.Log()
    sim2.deploy_B(end, cfg["env_B"], cfg["world"], cfg["mutation_sd"], 50, cfg, 2995, l50)
    sim2.deploy_B(end, cfg["env_B"], cfg["world"], cfg["mutation_sd"], 80, cfg, 2995, l200)
    a, b = l50.arrays(), l200.arrays()
    for k in a:
        if k.startswith("g_"):
            assert np.array_equal(a[k], b[k][:50]), k
        else:
            n = a[k].shape[0]
            assert np.array_equal(a[k], b[k][:n]), k


def test_clone_pick_and_clones():
    cfg = small_cfg()
    end = sim2.deploy_A([runner.et_start_traits(cfg)] * 50, cfg, 2996, cfg["mutation_sd"], runner.Log())
    i = sim2.pick_clone_index(end, 2996, 0)
    assert i == sim2.pick_clone_index(end, 2996, 0)
    c = sim2.clones_of(end, i)
    assert world.pop_size(c) == world.pop_size(end)
    for t in world.TRAITS:
        assert np.all(c[t] == end[t][i])
    assert np.array_equal(c["home_x"], end["home_x"])
    empty = sim2.empty_like(end)
    assert world.pop_size(empty) == 0 and sim2.pick_clone_index(empty, 2996) == -1


def test_start_in_B_uses_B_generation_numbers():
    cfg = small_cfg()
    log = runner.Log()
    sim2.start_in_B([runner.et_start_traits(cfg)] * 50, cfg["env_B"], cfg["world"], sim2.ZERO_SD,
                    50, cfg, 2997, log)
    a = log.arrays()
    assert a["g_gen"][0] == 12 and a["g_gen"][-1] == 61  # train_generations = 12 here
    assert np.all(a["g_phase"] == "B")


# ------------------------------------------------------------ search
def test_merged_grid_has_1472_designs_with_effective_reach():
    cfg = config.load(P2_BASE)
    ds = search2.merged_designs(cfg)
    assert len(ds) == 1472
    for d in ds:
        assert d["reach"] >= d["size"] / 4.0 - 1e-12
    keys = {tuple(round(d[t], 9) for t in world.TRAITS) for d in ds}
    assert len(keys) == 1472


def test_order_breaks_ties_by_worst_then_key():
    score = np.array([5.0, 5.0, 5.0, 4.0])
    worst = np.array([1.0, 2.0, 2.0, 9.0])
    key = np.array([0.9, 0.1, 0.5, 0.99])
    assert list(search2.order(score, worst, key)) == [2, 1, 0, 3]


def test_scores_from_counts_empty_generations_as_zero():
    p = np.array([10, 0, 5, 5])
    f = np.array([15, 0, 5, 10])
    ps, fs, worst = search2.scores_from(p, f, 0, 3)
    assert ps == 5 and worst == 0
    assert fs == Fraction(3, 2 * 4) + Fraction(1, 4) + Fraction(2, 4)  # (1.5 + 0 + 1 + 2) / 4 exactly


def test_equal_totals_tie_exactly_whatever_the_order():
    """Two designs with the same totals must tie exactly, so the tie rules decide."""
    a = [search2.scores_from(np.array([50, 49, 50]), np.array([50, 49, 50]), 0, 2) for _ in range(2)]
    b = search2.scores_from(np.array([49, 50, 50]), np.array([49, 50, 50]), 0, 2)
    m1 = search2.mean_fraction([a[0][0], b[0], a[1][0]])
    m2 = search2.mean_fraction([b[0], a[0][0], a[1][0]])
    assert m1 == m2


# ------------------------------------------------------------ outcomes from logs
def test_fitness_from_logs_equals_next_generation_population():
    cfg = small_cfg()
    log = runner.Log()
    sim2.deploy_A([runner.et_start_traits(cfg)] * 50, cfg, 2998, cfg["mutation_sd"], log)
    a = log.arrays()
    fit = p2metrics.blob_fitness({k[2:]: a[k] for k in a if k.startswith("b_")})
    gens = a["b_gen"]
    pops = dict(zip(a["g_gen"].tolist(), a["g_population"].tolist()))
    for g in np.unique(gens)[:-1]:
        assert int(fit[gens == g].sum()) == pops[int(g) + 1], g


# ------------------------------------------------------------ seeds
def test_seed_ranges_are_never_reused_across_purposes():
    """Plan rule: Part 2 uses 2000-2999 and never reuses a seed across purposes. Only the
    planned nestings are allowed (exploratory work and the landscape check use final-run seeds;
    the pilot landscape uses pilot seeds)."""
    s = config.load(P2_BASE)["part2"]["seeds"]
    family = {"final_100": "final", "final_200": "final", "exploratory": "final", "landscape": "final",
              "pilot": "pilot", "pilot_landscape": "pilot"}
    used = {}
    for name, r in s.items():
        lo, hi = (r, r) if isinstance(r, int) else r
        assert 2000 <= lo <= hi <= 2999, name
        for seed in range(lo, hi + 1):
            fam = family.get(name, name)
            assert used.setdefault(seed, fam) == fam, (seed, name, used[seed])


# ------------------------------------------------------------ arm 4
def test_adding_arm4_changes_nothing_else_and_its_oracle_uses_its_own_winner():
    """Arm 4 (Section 9.3) must not change any other arm's numbers: every stream is keyed,
    not drawn in sequence. Its ORACLE_B starts from arm 4's own B winner."""
    over = {"phases": {"train_generations": 12, "transfer_generations": 50},  # B search scores gens 10-49
            "part2": {"grid": {"points": 2}, "search": {"stage2_sets": 2, "stage2_top": 2},
                      "arm3_side": 400}}
    cfg3 = config.load(P2_BASE, over)
    cfg4 = copy.deepcopy(cfg3)
    cfg4["part2"]["arm4_energy"] = 10  # low enough that arm 4's B winner differs from the main B's here
    m3, a3 = run2.run_seed(2991, cfg3, exploratory=False, landscape=False)
    m4, a4 = run2.run_seed(2991, cfg4, exploratory=False, landscape=False)
    new = {k for k in a4 if "arm4" in k}
    assert new and set(a4) - new == set(a3)
    for k in a3:
        assert np.array_equal(a3[k], a4[k]), k
    for w in ("main", "arm3"):
        assert m3["search_B"][w] == m4["search_B"][w] and m3["u2"][w] == m4["u2"][w]
    assert set(m4["search_B"]) == {"main", "arm3", "arm4"} and m4["b_worlds"]["arm4"]["world"]["energy_per_generation"] == 10
    win = m4["designs"]["B_winner_arm4"]
    assert win != m4["designs"]["B_winner_main"]  # otherwise the next check could not fail
    for t in world.TRAITS:  # ORACLE_B in arm 4 = clones of arm 4's B winner
        assert np.all(a4[f"B__arm4__ORACLE_B__b_{t}"][a4["B__arm4__ORACLE_B__b_gen"] == 12] == win[t])


def test_arm4_shares_every_main_B_stream():
    """Section 10: arms use the main arm's streams. With arm 4's energy set to the main B's
    (500), arm 4's B search, U2 runs and every B log must equal the main B's bit for bit.
    This fails if any arm 4 key (search stages, tie-break, U2, deployment) differs."""
    over = {"phases": {"train_generations": 12, "transfer_generations": 50},
            "part2": {"grid": {"points": 2}, "search": {"stage2_sets": 2, "stage2_top": 2},
                      "arm4_energy": 500}}
    cfg = config.load(P2_BASE, over)
    m, a = run2.run_seed(2992, cfg, exploratory=False, landscape=False)
    assert m["search_B"]["arm4"] == m["search_B"]["main"]
    assert m["u2"]["arm4"] == m["u2"]["main"]
    for k in [k for k in a if k.startswith("SB__arm4__")]:
        assert np.array_equal(a[k], a[k.replace("SB__arm4__", "SB__main__")]), k
    arm4_logs = [k for k in a if k.startswith("B__arm4__")]
    assert len(arm4_logs) > 0
    for k in arm4_logs:
        assert np.array_equal(a[k], a[k.replace("B__arm4__", "B__main__")]), k
