"""One generation of the blob world: a rule-by-rule port of the MinuteLabs simulator.

Ported from the MinuteLabs.io Evolution Simulator by Jasper Palfree,
https://github.com/minutelabsio/evolution-simulator, commit 55af956, GPL-3.0.
File references below ("S1 creature/mod.rs:171") are relative to that repo's
src/wasm/src/ folder. See docs/SPEC_part1.md for the plain-English rules.

The kernel is compiled with numba. It loops over blobs one at a time, in the
shuffled list order, exactly as S1 does. That keeps order-dependent rules
(who eats a contested food item, predator-prey pairs) faithful while staying fast.

Random numbers do not match S1 bit for bit (S1 uses Rust's SmallRng); only the
rules match.
"""
import math

import numpy as np
from numba import njit

# Creature states, S1 creature/mod.rs:19-24
DEAD = 0
ASLEEP = 1
ACTIVE = 2

# Objective urgency, lowest to highest, S1 creature/mod.rs:27-41
NO_OBJECTIVE = -1
MINOR_CRAVING = 0
MINOR_AVERSION = 1
MODERATE_CRAVING = 2
MODERATE_AVERSION = 3
MAJOR_CRAVING = 4
MAJOR_AVERSION = 5
VITAL_CRAVING = 6
VITAL_AVERSION = 7

ENERGY_COST_SCALE = 1.0 / 10_000.0  # S1 creature/mod.rs:10
QUARTER_PI = math.pi / 4.0  # S1 behaviours/mod.rs:2 (FRAC_PI_4)
# S1 generation.rs:7,85: assert!(steps < 1_000_000) with steps starting at 1,
# so S1 aborts once 999_999 ticks have run.
S1_TICK_LIMIT = 999_999

# Error codes returned by the kernel
OK = 0
ERR_S1_TICK_LIMIT = 1


@njit(cache=True)
def effective_speed(speed, size):
    """S1 creature/mod.rs:171 get_speed = speed * size / 10."""
    return speed * size / 10.0


@njit(cache=True)
def motion_cost(speed, size, sense):
    """S1 creature/mod.rs:214-218 (Primer's formula)."""
    eff = effective_speed(speed, size)
    return ENERGY_COST_SCALE * (size ** 3 * eff ** 2 + sense)


@njit(cache=True)
def _norm(x, y):
    return math.sqrt(x * x + y * y)


@njit(cache=True)
def _clamp(v, hi):
    """S1 stage/mod.rs:65-70: v.max(0.).min(hi). Rust's f64::max returns the
    other operand when one is NaN, so NaN becomes 0."""
    if not (v >= 0.0):
        v = 0.0
    if v > hi:
        v = hi
    return v


@njit(cache=True)
def _add_objective(i, ox, oy, inten, obj_x, obj_y, obj_i, pos_x, pos_y):
    """S1 creature/mod.rs:241-262: keep the more urgent goal; on a tie keep the
    strictly closer one; otherwise keep the old one."""
    cur = obj_i[i]
    if cur == NO_OBJECTIVE or inten > cur:
        obj_x[i] = ox
        obj_y[i] = oy
        obj_i[i] = inten
    elif inten == cur:
        d_old = _norm(pos_x[i] - obj_x[i], pos_y[i] - obj_y[i])
        d_new = _norm(pos_x[i] - ox, pos_y[i] - oy)
        if d_new < d_old:
            obj_x[i] = ox
            obj_y[i] = oy
            obj_i[i] = inten


@njit(cache=True)
def _can_reach(i, px, py, pos_x, pos_y, last_x, last_y, has_last, reach_eff):
    """S1 creature/mod.rs:300-309, 320-322 and math/mod.rs:13-39.

    True if the point is within reach of the current position, or within reach
    of the last movement segment *and* its perpendicular foot falls on that
    segment (S1's project_to_line returns None otherwise)."""
    dx = px - pos_x[i]
    dy = py - pos_y[i]
    if _norm(dx, dy) <= reach_eff[i]:
        return True
    if not has_last[i]:
        return False
    vx = last_x[i] - pos_x[i]
    vy = last_y[i] - pos_y[i]
    ln = _norm(vx, vy)
    if ln == 0.0:
        # S1 normalizes a zero vector to NaN; every later comparison is false.
        return False
    nx = vx / ln
    ny = vy / ln
    pax = pos_x[i] - px
    pay = pos_y[i] - py
    pbx = last_x[i] - px
    pby = last_y[i] - py
    da = pax * nx + pay * ny
    db = pbx * nx + pby * ny
    if da * db > 0.0:
        return False
    qx = -da * nx + pax
    qy = -da * ny + pay
    return _norm(qx, qy) <= reach_eff[i]


@njit(cache=True)
def _heading(i, pos_x, pos_y, last_x, last_y, has_last):
    """S1 creature/mod.rs:231-238: fallback direction = last move, else +x.
    A zero vector normalizes to NaN in S1 (nalgebra Unit::new_normalize)."""
    if has_last[i]:
        dx = pos_x[i] - last_x[i]
        dy = pos_y[i] - last_y[i]
    else:
        dx = 1.0
        dy = 0.0
    ln = _norm(dx, dy)
    if ln == 0.0:
        return np.nan, np.nan
    return dx / ln, dy / ln


@njit(cache=True)
def _edge_home(x, y, L):
    """S1 behaviours/edge_home.rs:11-29: nearest point on the nearest edge.
    Edges in S1 order (stage/mod.rs:28-35): bottom, right, top, left.
    Ties go to the first edge (Iterator::min_by keeps the first minimum)."""
    best = y
    hx = x
    hy = 0.0
    d = L - x
    if d < best:
        best = d
        hx = L
        hy = y
    d = L - y
    if d < best:
        best = d
        hx = x
        hy = L
    d = x
    if d < best:
        best = d
        hx = 0.0
        hy = y
    return hx, hy


@njit(cache=True)
def run_generation(speed, size, sense, reach, flee, home_x, home_y,
                   food_x, food_y, L, energy0, cannibalism, size_ratio,
                   tick_cap, rng):
    """Run one generation to completion.

    Inputs are per-blob trait arrays and home positions, and the food positions
    placed for this generation. tick_cap <= 0 means no limit (S1's rule).

    Returns (state, food_balls, prey, home_x_out, home_y_out, moves, consumed,
    food_eaten, ticks, error_code).
    """
    n = speed.shape[0]
    m = food_x.shape[0]

    state = np.full(n, ACTIVE, np.int64)
    food_balls = np.zeros(n, np.int64)
    prey = np.zeros(n, np.int64)
    food_count = np.zeros(n, np.int64)  # foods_eaten.len(): food balls + prey
    moves = np.zeros(n, np.int64)
    consumed = np.zeros(n)
    pos_x = home_x.copy()
    pos_y = home_y.copy()
    hx = home_x.copy()
    hy = home_y.copy()
    last_x = home_x.copy()
    last_y = home_y.copy()
    has_last = np.zeros(n, np.bool_)
    obj_x = np.zeros(n)
    obj_y = np.zeros(n)
    obj_i = np.full(n, NO_OBJECTIVE, np.int64)
    eaten = np.zeros(m, np.bool_)
    n_avail = m

    eff = np.empty(n)
    cost = np.empty(n)
    reach_eff = np.empty(n)
    for i in range(n):
        eff[i] = effective_speed(speed[i], size[i])
        cost[i] = motion_cost(speed[i], size[i], sense[i])
        # S1 creature/mod.rs:177: reach is at least a quarter of size
        reach_eff[i] = max(reach[i], size[i] / 4.0)

    # The creature list order. S1 shuffles the list in place every tick
    # (generation.rs:79-82, 88), so the order carries over between ticks.
    order = np.arange(n)

    # INIT, S1 behaviours/starve.rs:16-23
    for i in range(n):
        if eff[i] == 0.0:
            state[i] = DEAD

    cx = 0.5 * L
    cy = 0.5 * L
    ticks = 0
    error = OK
    prev_speed = np.empty(n)

    while True:
        any_active = False
        for i in range(n):
            if state[i] == ACTIVE:
                any_active = True
                break
        if not any_active:
            break
        if tick_cap > 0 and ticks >= tick_cap:
            break
        if ticks >= S1_TICK_LIMIT:
            error = ERR_S1_TICK_LIMIT
            break

        # Shuffle (Fisher-Yates), S1 generation.rs:79-82
        for k in range(n - 1, 0, -1):
            j = rng.integers(0, k + 1)
            tmp = order[k]
            order[k] = order[j]
            order[j] = tmp

        # ---- PRE: ResetBehaviour (simulation/mod.rs:24-41) and
        #      EdgeHomeBehaviour (behaviours/edge_home.rs:32-39), living blobs
        for k in range(n):
            i = order[k]
            if state[i] != DEAD:
                obj_i[i] = NO_OBJECTIVE
                hx[i], hy[i] = _edge_home(pos_x[i], pos_y[i], L)

        # ---- ORIENT a: WanderBehaviour (behaviours/wander.rs:16-33)
        for k in range(n):
            i = order[k]
            if state[i] != ACTIVE:
                continue
            ux, uy = _heading(i, pos_x, pos_y, last_x, last_y, has_last)
            ang = rng.uniform(-QUARTER_PI, QUARTER_PI)
            c = math.cos(ang)
            s = math.sin(ang)
            tx = pos_x[i] + c * ux - s * uy
            ty = pos_y[i] + s * ux + c * uy
            # stage/mod.rs:37-42 can_move_to; NaN fails every comparison
            if tx >= 0.0 and ty >= 0.0 and tx <= L and ty <= L:
                _add_objective(i, tx, ty, MINOR_CRAVING, obj_x, obj_y, obj_i, pos_x, pos_y)
            else:
                _add_objective(i, cx, cy, MINOR_CRAVING, obj_x, obj_y, obj_i, pos_x, pos_y)

        # ---- ORIENT b: CannibalismBehaviour (behaviours/cannibalism.rs:35-81)
        if cannibalism:
            for i in range(n):
                prev_speed[i] = np.inf  # empty HashMap of target prey speeds
            for k in range(1, n):
                # S1 checks creature k (not k-1) here: quirk Q1
                if state[order[k]] != ACTIVE:
                    continue
                a = order[k - 1]
                for kk in range(k, n):
                    b = order[kk]
                    if state[b] != ACTIVE:
                        continue
                    if size[a] > size[b]:
                        pr = a
                        py_ = b
                    else:
                        pr = b
                        py_ = a
                    if state[pr] != ACTIVE or state[py_] != ACTIVE:
                        continue
                    if size[pr] * size_ratio < size[py_]:
                        continue
                    dx = pos_x[pr] - pos_x[py_]
                    dy = pos_y[pr] - pos_y[py_]
                    d = _norm(dx, dy)
                    # prey runs away (creature/mod.rs:311-318 within_flee_distance)
                    if d <= sense[py_] and d < flee[py_]:
                        ang = rng.uniform(-QUARTER_PI, QUARTER_PI)
                        c = math.cos(ang)
                        s = math.sin(ang)
                        nx_ = pos_x[py_] + c * dx - s * dy
                        ny_ = pos_y[py_] + s * dx + c * dy
                        _add_objective(py_, nx_, ny_, VITAL_AVERSION, obj_x, obj_y, obj_i, pos_x, pos_y)
                    if d > sense[pr]:
                        continue
                    if prev_speed[pr] != np.inf and prev_speed[pr] <= eff[py_]:
                        continue
                    prev_speed[pr] = eff[py_]
                    fc = food_count[pr]
                    if fc == 0:
                        inten = VITAL_CRAVING
                    elif fc == 1:
                        inten = MODERATE_CRAVING
                    else:
                        inten = MINOR_CRAVING
                    _add_objective(pr, pos_x[py_], pos_y[py_], inten, obj_x, obj_y, obj_i, pos_x, pos_y)

        # ---- ORIENT c: ScavengeBehaviour look_for_food (behaviours/scavenge.rs:11-29, 57-65)
        for k in range(n):
            i = order[k]
            if state[i] != ACTIVE or food_count[i] >= 2:
                continue
            best = -1
            bestd = np.inf
            for j in range(m):
                if eaten[j]:
                    continue
                d = _norm(food_x[j] - pos_x[i], food_y[j] - pos_y[i])
                if d < bestd:  # first minimum wins, like Iterator::min_by
                    bestd = d
                    best = j
            if best >= 0 and bestd <= sense[i]:
                inten = VITAL_CRAVING if food_count[i] == 0 else MODERATE_CRAVING
                _add_objective(i, food_x[best], food_y[best], inten, obj_x, obj_y, obj_i, pos_x, pos_y)

        # ---- ORIENT d: SatisfiedBehaviour (behaviours/satisfied.rs:7-41)
        #      with HomesickBehaviour::how_homesick (behaviours/homesick.rs:7-27)
        for k in range(n):
            i = order[k]
            if state[i] != ACTIVE:
                continue
            fc = food_count[i]
            if fc == 0:
                continue
            if fc > 1:
                inten = MAJOR_CRAVING
            else:
                dist = _norm(hx[i] - pos_x[i], hy[i] - pos_y[i])
                steps_home = dist / eff[i]
                left = max(energy0 - consumed[i], 0.0)
                factor = left / cost[i] - steps_home
                if factor > 10.0:
                    continue
                elif factor > 5.0:
                    inten = MINOR_CRAVING
                elif factor > 0.0:
                    inten = MAJOR_CRAVING
                else:
                    inten = VITAL_CRAVING
            _add_objective(i, hx[i], hy[i], inten, obj_x, obj_y, obj_i, pos_x, pos_y)
            if _can_reach(i, hx[i], hy[i], pos_x, pos_y, last_x, last_y, has_last, reach_eff):
                state[i] = ASLEEP

        # ---- MOVE: BasicMoveBehaviour (behaviours/move.rs:7-24),
        #      Creature::move_to and apply_energy_cost (creature/mod.rs:207-239, 328-334)
        for k in range(n):
            i = order[k]
            if state[i] != ACTIVE:
                continue
            ux = np.nan
            uy = np.nan
            use_fallback = True
            if obj_i[i] != NO_OBJECTIVE:
                dx = obj_x[i] - pos_x[i]
                dy = obj_y[i] - pos_y[i]
                if obj_i[i] % 2 == 1:  # aversions are the odd urgencies: go the other way
                    dx = -dx
                    dy = -dy
                if not (dx == 0.0 and dy == 0.0):
                    ln = _norm(dx, dy)
                    ux = dx / ln
                    uy = dy / ln
                    use_fallback = False
            if use_fallback:
                ux, uy = _heading(i, pos_x, pos_y, last_x, last_y, has_last)
            nxp = _clamp(pos_x[i] + eff[i] * ux, L)
            nyp = _clamp(pos_y[i] + eff[i] * uy, L)
            last_x[i] = pos_x[i]
            last_y[i] = pos_y[i]
            pos_x[i] = nxp
            pos_y[i] = nyp
            has_last[i] = True
            consumed[i] += cost[i]
            moves[i] += 1
            if max(energy0 - consumed[i], 0.0) <= 0.0:
                state[i] = DEAD

        # ---- ACT a: CannibalismBehaviour eat (behaviours/cannibalism.rs:84-93)
        if cannibalism:
            for k in range(1, n):
                if state[order[k]] != ACTIVE:
                    continue
                a = order[k - 1]
                for kk in range(k, n):
                    b = order[kk]
                    if state[b] != ACTIVE:
                        continue
                    if size[a] > size[b]:
                        pr = a
                        py_ = b
                    else:
                        pr = b
                        py_ = a
                    if state[pr] != ACTIVE or state[py_] != ACTIVE:
                        continue
                    if size[pr] * size_ratio < size[py_]:
                        continue
                    if not _can_reach(pr, pos_x[py_], pos_y[py_], pos_x, pos_y,
                                      last_x, last_y, has_last, reach_eff):
                        continue
                    food_count[pr] += 1
                    prey[pr] += 1
                    state[py_] = DEAD

        # ---- ACT b: ScavengeBehaviour eat (behaviours/scavenge.rs:31-39, 67-82)
        for k in range(n):
            i = order[k]
            if state[i] != ACTIVE or food_count[i] >= 2:
                continue
            best = -1
            bestd = np.inf
            for j in range(m):
                if eaten[j]:
                    continue
                d = _norm(food_x[j] - pos_x[i], food_y[j] - pos_y[i])
                if d < bestd:
                    bestd = d
                    best = j
            if best >= 0 and _can_reach(i, food_x[best], food_y[best], pos_x, pos_y,
                                        last_x, last_y, has_last, reach_eff):
                eaten[best] = True
                n_avail -= 1
                food_balls[i] += 1
                food_count[i] += 1

        # ---- POST: StarveBehaviour (behaviours/starve.rs:25-35)
        if n_avail == 0:
            for i in range(n):
                if state[i] == ACTIVE and food_count[i] == 0:
                    state[i] = DEAD

        ticks += 1

    # ---- FINAL: StarveBehaviour (behaviours/starve.rs:37-43)
    for i in range(n):
        if state[i] != DEAD and food_count[i] < 1:
            state[i] = DEAD

    return (state, food_balls, prey, hx, hy, moves, consumed, eaten, ticks, error)
