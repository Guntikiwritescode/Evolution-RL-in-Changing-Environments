"""Whole-run tests and hand-calculated metric checks."""
import os

import numpy as np
import pytest
from scipy import stats as sps

from evosim import config, gridsearch, metrics, runner, stats, world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "configs", "part1", "base.yaml")


def cfg_small(**over):
    o = {"phases": {"train_generations": 15, "transfer_generations": 8}, "world": {"tick_cap": 500}}
    for k, v in over.items():
        o.setdefault(k, {}).update(v)
    return config.load(BASE, o)


def run(cfg, seed, traits=None, sd=None, n=50):
    return runner.run_lineage(traits or runner.et_start_traits(cfg), n,
                              sd if sd is not None else cfg["mutation_sd"], cfg,
                              np.random.SeedSequence(seed)).arrays()


def test_food_eaten_never_exceeds_food_placed():
    a = run(cfg_small(), 9100)
    assert np.all(a["g_food_eaten"] <= a["g_food_placed"])
    assert np.all(a["g_food_placed"][:15] == 50) and np.all(a["g_food_placed"][15:] == 25)


def test_same_seed_same_output():
    c = cfg_small()
    a, b = run(c, 9101), run(c, 9101)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    d = run(c, 9102)
    assert not np.array_equal(a["b_speed"], d["b_speed"])


def test_clones_without_mutation_stay_identical():
    c = cfg_small()
    design = {"speed": 12.0, "size": 8.0, "sense": 25.0, "reach": 1.0}
    a = run(c, 9103, traits=design, sd={t: 0.0 for t in world.TRAITS})
    for t in world.TRAITS:
        assert np.all(a["b_" + t] == design[t])


def test_extinct_population_stays_extinct():
    c = cfg_small()
    doomed = {"speed": 10.0, "size": 40.0, "sense": 20.0, "reach": 1.0}  # dies on its first move
    a = run(c, 9104, traits=doomed)
    pop = a["g_population"]
    assert pop[0] == 50
    first_zero = int(np.argmax(pop == 0))
    assert first_zero > 0 and np.all(pop[first_zero:] == 0)


def test_traits_stay_in_bounds():
    c = cfg_small(mutation_sd={"speed": 4.0, "size": 4.0, "sense": 4.0, "reach": 4.0})
    a = run(c, 9105)
    for t in world.TRAITS:
        v = a["b_" + t]
        assert np.all(np.isfinite(v))
        assert np.all(v >= world.TRAIT_FLOOR[t])
        if t != "sense":
            assert np.all(v > 0)


def test_population_bookkeeping():
    """Next population = survivors + births (R1-R6)."""
    a = run(cfg_small(), 9106)
    pop, surv, births = a["g_population"], a["g_survivors"], a["g_births"]
    for g in range(len(pop) - 1):  # also across the A -> B switch at g = 14
        assert pop[g + 1] == surv[g] + births[g]


# ---------------------------------------------------------------- metrics by hand
def test_shannon_by_hand():
    rows = np.array([[1, 1], [1, 1], [2, 2], [3, 3]], float)
    expected = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25))
    assert metrics.shannon(rows) == pytest.approx(expected)
    assert metrics.shannon(np.ones((5, 4))) == 0.0


def test_series_by_hand():
    """Two generations, three blobs then two, numbers worked out by hand."""
    a = {
        "g_gen": np.array([0, 1]), "g_population": np.array([3, 2]),
        "g_food_eaten": np.array([4, 2]), "g_prey_eaten": np.array([0, 1]),
        "g_survivors": np.array([2, 1]), "g_ticks": np.array([10, 5]),
        "g_creature_steps": np.array([25, 8]),
        "b_gen": np.array([0, 0, 0, 1, 1]),
        "b_speed": np.array([10.0, 10.0, 13.0, 10.0, 10.0]),
        "b_size": np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        "b_sense": np.array([20.0, 20.0, 20.0, 20.0, 20.0]),
        "b_reach": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "b_food_balls": np.array([2, 2, 0, 2, 0]), "b_prey": np.array([0, 0, 0, 0, 1]),
    }
    s = metrics.series(a)
    assert s["fitness"][0] == pytest.approx(4 / 3)
    assert s["fitness"][1] == pytest.approx(3 / 2)
    assert s["fitness_balls_only"][1] == pytest.approx(1.0)
    assert s["median"][0] == 2.0
    assert s["survival"][0] == pytest.approx(2 / 3)
    speeds = np.array([10.0, 10.0, 13.0])
    assert s["cv2_speed"][0] == pytest.approx((speeds.std() / speeds.mean()) ** 2)
    assert s["cv2_4_pop"][0] == pytest.approx((speeds.std() / speeds.mean()) ** 2)  # other traits equal
    assert s["H_exact4"][0] == pytest.approx(-(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)))
    assert s["H_exact4"][1] == 0.0


def test_welch_matches_scipy_and_paper_d_rule():
    rng = np.random.default_rng(1)
    x = rng.normal(1.03, 0.03, 30)
    y = rng.normal(1.30, 0.10, 30)
    r = stats.welch(x, y)
    ref = sps.ttest_ind(x, y, equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic)
    assert r["p"] == pytest.approx(ref.pvalue)
    assert r["d"] == pytest.approx(r["t"] * np.sqrt(2 / 30))  # Equal-sized samples: relation between t and standardized mean difference
    assert r["p_bonferroni"] == pytest.approx(min(1, 10 * r["p"]))
    z = stats.welch(x, np.zeros(30))
    assert z["df"] == pytest.approx(29.0)  # one group with zero spread


# ---------------------------------------------------------------- grid search
def test_grid_values_and_order():
    c = config.load(BASE)
    axes = gridsearch.grid_axes(c)
    assert np.allclose(axes["speed"], 2.5 * 16 ** (np.arange(8) / 7))
    assert np.allclose(axes["sense"], 5.0 * 16 ** (np.arange(8) / 7))
    d = gridsearch.grid_designs(c)
    assert len(d) == 512 and d[0]["reach"] == 1.0
    assert d[0]["speed"] == d[63]["speed"] != d[64]["speed"]


def test_welch_handles_too_few_values():
    r = stats.welch([np.nan, np.nan], [1.0, 2.0, 3.0])
    assert r["n_x"] == 0 and np.isnan(r["t"]) and r["stars"].startswith("n/a")
    r = stats.welch([1.0], [1.0, 2.0])
    assert np.isnan(r["p"])
    # the group that has enough values keeps its own mean and SD (bug found 31 Oct)
    r = stats.welch([1.0, 2.0, 3.0], [np.nan, 5.0])
    assert r["n_x"] == 3 and r["n_y"] == 1
    assert r["mean_x"] == 2.0 and r["sd_x"] == 1.0
    assert r["mean_y"] == 5.0 and np.isnan(r["sd_y"]) and np.isnan(r["t"])
    # normal case unchanged: matches scipy's Welch test
    from scipy import stats as sps
    a, b = [1.0, 2.0, 4.0, 7.0], [2.0, 3.0, 3.5]
    r = stats.welch(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert abs(r["t"] - ref.statistic) < 1e-12 and abs(r["p"] - ref.pvalue) < 1e-12


def test_fixed_clusters_keep_centers_and_counts():
    """Environment B side check W8 (cluster spots fixed for the whole transfer phase)."""
    from evosim import world
    env = {"size": 300.0, "food": 25, "layout": "clusters", "n_clusters": 5, "cluster_sd": 15.0,
           "cluster_margin_sds": 2.0, "redraw_each_generation": False}
    rng = np.random.default_rng(16100)
    cx, cy = world.fixed_cluster_centers(env, rng)
    assert cx.size == 5 and np.all((cx >= 30) & (cx <= 270)) and np.all((cy >= 30) & (cy <= 270))
    pts = [world.place_food_fixed_clusters(env, (cx, cy), rng) for _ in range(200)]
    for x, y in pts:
        assert x.size == 25 and np.all((x >= 0) & (x <= 300) & (y >= 0) & (y <= 300))
    # food item i always belongs to cluster i // 5; averaged over 200 draws it sits on that center
    mx = np.mean([p[0] for p in pts], axis=0)
    my = np.mean([p[1] for p in pts], axis=0)
    assert np.allclose(mx, np.repeat(cx, 5), atol=4.0) and np.allclose(my, np.repeat(cy, 5), atol=4.0)
    # and the spread around the center matches cluster_sd (15) within sampling error
    sx = np.std(np.concatenate([p[0] - np.repeat(cx, 5) for p in pts]))
    assert 13.5 < sx < 16.5
