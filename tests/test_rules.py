"""Hand-checked tests of single rules (docs/SPEC_part1.md Sections 4-6)."""
import math

import numpy as np
import pytest

from evosim import kernel as K


def arr(*v):
    return np.array(v, dtype=float)


def one_gen(traits, homes, food, L=500.0, cap=0, cannibalism=False, seed=0):
    n = len(homes)
    sp = np.array([t[0] for t in traits], float)
    sz = np.array([t[1] for t in traits], float)
    se = np.array([t[2] for t in traits], float)
    re = np.array([t[3] for t in traits], float)
    fl = np.full(n, 1e12)
    hx = np.array([h[0] for h in homes], float)
    hy = np.array([h[1] for h in homes], float)
    fx = np.array([f[0] for f in food], float) if food else np.zeros(0)
    fy = np.array([f[1] for f in food], float) if food else np.zeros(0)
    return K.run_generation(sp, sz, se, re, fl, hx, hy, fx, fy, L, 500.0, cannibalism, 0.8, cap,
                            np.random.default_rng(seed))


# ---- B1, B5: speed and energy cost
def test_energy_cost_default_blob():
    assert K.effective_speed(10.0, 10.0) == pytest.approx(10.0)
    assert K.motion_cost(10.0, 10.0, 20.0) == pytest.approx(1e-4 * (1000 * 100 + 20))


# ---- 5.3: goal urgency and ties
def test_add_objective_priority_and_ties():
    ox, oy, oi = np.zeros(1), np.zeros(1), np.full(1, K.NO_OBJECTIVE, np.int64)
    px, py = arr(0.0), arr(0.0)
    K._add_objective(0, 5.0, 0.0, K.MINOR_CRAVING, ox, oy, oi, px, py)
    assert (ox[0], oi[0]) == (5.0, K.MINOR_CRAVING)
    K._add_objective(0, 9.0, 0.0, K.MINOR_CRAVING, ox, oy, oi, px, py)  # same urgency, farther: keep
    assert ox[0] == 5.0
    K._add_objective(0, 2.0, 0.0, K.MINOR_CRAVING, ox, oy, oi, px, py)  # same urgency, closer: take
    assert ox[0] == 2.0
    K._add_objective(0, 2.0, 0.0, K.MINOR_CRAVING, ox, oy, oi, px, py)  # equal distance: keep old
    K._add_objective(0, 50.0, 0.0, K.MODERATE_CRAVING, ox, oy, oi, px, py)  # more urgent: take
    assert (ox[0], oi[0]) == (50.0, K.MODERATE_CRAVING)
    K._add_objective(0, 1.0, 0.0, K.MINOR_AVERSION, ox, oy, oi, px, py)  # less urgent: keep
    assert ox[0] == 50.0


# ---- B4: reach along the last path, only "beside" it
def test_can_reach_beside_path_but_not_behind():
    px, py = arr(10.0), arr(0.0)        # now at (10, 0)
    lx, ly = arr(0.0), arr(0.0)         # came from (0, 0)
    has = np.array([True])
    r = arr(1.0)
    assert K._can_reach(0, 5.0, 0.9, px, py, lx, ly, has, r)        # beside the path
    assert not K._can_reach(0, 5.0, 1.1, px, py, lx, ly, has, r)    # too far to the side
    assert not K._can_reach(0, -0.5, 0.0, px, py, lx, ly, has, r)   # within 1 of the last spot, but behind it
    assert K._can_reach(0, 10.5, 0.0, px, py, lx, ly, has, r)       # within reach of where it is now
    has[0] = False
    assert not K._can_reach(0, 5.0, 0.9, px, py, lx, ly, has, r)    # no path yet: only "now" counts


def test_can_reach_zero_length_path_is_false():
    px, py, lx, ly = arr(3.0), arr(3.0), arr(3.0), arr(3.0)
    assert not K._can_reach(0, 3.0, 5.0, px, py, lx, ly, np.array([True]), arr(1.0))


# ---- PRE: home = nearest point on nearest edge, ties to the first edge (bottom, right, top, left)
def test_edge_home():
    assert K._edge_home(100.0, 30.0, 500.0) == (100.0, 0.0)
    assert K._edge_home(480.0, 300.0, 500.0) == (500.0, 300.0)
    assert K._edge_home(200.0, 495.0, 500.0) == (200.0, 500.0)
    assert K._edge_home(3.0, 200.0, 500.0) == (0.0, 200.0)
    assert K._edge_home(250.0, 250.0, 500.0) == (250.0, 0.0)  # tie: bottom first


def test_heading_fallbacks():
    px, py, lx, ly = arr(5.0), arr(5.0), arr(5.0), arr(5.0)
    assert K._heading(0, px, py, lx, ly, np.array([False])) == (1.0, 0.0)
    hx, hy = K._heading(0, px, py, lx, ly, np.array([True]))  # zero-length last move
    assert math.isnan(hx) and math.isnan(hy)


def test_clamp_matches_rust_max_min():
    assert K._clamp(-3.0, 10.0) == 0.0
    assert K._clamp(12.0, 10.0) == 10.0
    assert K._clamp(float("nan"), 10.0) == 0.0


# ---- whole-generation rules
def test_eats_nearby_food_and_survives():
    st, fb, prey, hx, hy, mv, cons, eaten, ticks, err = one_gen(
        [(10, 10, 20, 1)], [(0.0, 250.0)], [(6.0, 250.0)])
    assert err == 0
    assert fb[0] == 1 and eaten[0]
    assert st[0] != K.DEAD  # 1 food = survive


def test_zero_food_dies_when_food_runs_out():
    # Blob 0 sits right next to the only food; blob 1 is far away and can't see it.
    st, fb, prey, hx, hy, mv, *_ = one_gen(
        [(10, 10, 20, 1), (10, 10, 20, 1)], [(0.0, 250.0), (500.0, 10.0)], [(2.0, 250.0)])
    assert fb[0] == 1 and fb[1] == 0
    assert st[1] == K.DEAD
    assert mv[1] == 1  # killed in POST of tick 1, right after the last food was eaten
    assert st[0] != K.DEAD


def test_out_of_energy_dies_after_one_move():
    st, fb, prey, hx, hy, mv, cons, eaten, ticks, err = one_gen(
        [(10, 40, 20, 1)], [(0.0, 250.0)], [(400.0, 400.0)])
    assert mv[0] == 1 and st[0] == K.DEAD


def test_two_food_then_home_and_sleep():
    st, fb, *_ = one_gen([(10, 10, 20, 1)], [(0.0, 250.0)], [(6.0, 250.0), (16.0, 250.0)])
    assert fb[0] == 2 and st[0] == K.ASLEEP


def test_never_eats_more_than_two_food_items():
    food = [(3.0 * k, 250.0) for k in range(1, 8)]
    st, fb, *_ = one_gen([(10, 10, 20, 1)], [(0.0, 250.0)], food)
    assert fb[0] == 2


def test_tick_cap():
    *_, ticks, err = one_gen([(2.5, 2.5, 5, 1)], [(0.0, 250.0)], [(400.0, 400.0)], cap=7)
    assert ticks == 7 and err == 0


def test_mild_home_goal_loses_to_wander():
    """Q10: a "mild want" home goal (homesick number 5-10) loses to wander's
    goal 1 unit ahead; a "strong want" home goal (homesick number 0-5) wins."""
    ox, oy, oi = np.zeros(1), np.zeros(1), np.full(1, K.NO_OBJECTIVE, np.int64)
    px, py = arr(100.0), arr(100.0)
    K._add_objective(0, 101.0, 100.0, K.MINOR_CRAVING, ox, oy, oi, px, py)  # wander
    K._add_objective(0, 0.0, 100.0, K.MINOR_CRAVING, ox, oy, oi, px, py)    # home, mild
    assert ox[0] == 101.0
    K._add_objective(0, 0.0, 100.0, K.MAJOR_CRAVING, ox, oy, oi, px, py)    # home, strong
    assert ox[0] == 0.0


def test_predation_only_with_size_ratio():
    # Big blob (size 10) next to a small one (size 8 = 0.8 x 10): it catches it.
    # The prey runs at a random angle, so allow up to 30 ticks.
    traits = [(10, 10, 20, 1), (10, 8, 20, 1)]
    homes = [(0.0, 250.0), (1.0, 250.0)]
    for seed in range(5):
        st, fb, prey, *_ = one_gen(traits, homes, [(450.0, 450.0)], cannibalism=True, cap=30, seed=seed)
        assert prey[0] == 1 and st[1] == K.DEAD
    # Size 8.1 is above 0.8 x 10: no predation.
    traits = [(10, 10, 20, 1), (10, 8.1, 20, 1)]
    st, fb, prey, *_ = one_gen(traits, homes, [(450.0, 450.0)], cannibalism=True, cap=30)
    assert prey.sum() == 0
    # Switched off: no predation even when possible.
    traits = [(10, 10, 20, 1), (10, 8, 20, 1)]
    st, fb, prey, *_ = one_gen(traits, homes, [(450.0, 450.0)], cannibalism=False, cap=30)
    assert prey.sum() == 0
