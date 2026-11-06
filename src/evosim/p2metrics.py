"""Part 2 outcomes from raw logs (docs/PREREG_part2.md, Section 5).

Generation numbers: A is 0-99, B generation g is 100 + g (also for groups
that start directly in B). "Population in a generation" is the number of
blobs alive at its start (g_population).
"""
import math

import numpy as np

N_A = 100


def gen_arrays(arrays, prefix):
    """The g_* arrays of one logged run, by field name."""
    p = prefix + "__g_"
    return {k[len(p):]: arrays[k] for k in arrays if k.startswith(p)}


def blob_arrays(arrays, prefix):
    p = prefix + "__b_"
    return {k[len(p):]: arrays[k] for k in arrays if k.startswith(p)}


def b_slice(g, lo, hi, n_a=N_A):
    """Rows of B generations lo..hi (inclusive)."""
    gen = np.asarray(g["gen"])
    m = (gen >= n_a + lo) & (gen <= n_a + hi)
    return m


def outcomes(g, food_count=25, s1_level=None, late=None, n_a=N_A):
    """Outcomes of one group's B phase.

    P1_total: sum of populations over B generations 40-49 (P1 = P1_total / 10).
    P2: True if survivors + births in B generation 49 are 0 (nobody starts
        generation 50). A run that died out in A has 0 everywhere in B.
    S5: B generation 0 started with blobs. S6: mean population, B gens 1-49.
    S4: mean food per blob over B gens 40-49 among generations with blobs.
    S1: first B generation >= 1 with population >= s1_level (default:
        ceil(0.9 x food count), 23 for 25 food); None if never.
    late: (lo, hi) of a later window (longer B, Section 18.5) for P1/P2 there.
    n_a: number of A generations (100; smaller only in --quick test runs)."""
    pop = np.asarray(g["population"], np.int64)
    gen = np.asarray(g["gen"], np.int64)
    by_gen = dict(zip(gen.tolist(), range(len(gen))))

    def row(bg):
        return by_gen[n_a + bg]

    w = b_slice(g, 40, 49, n_a)
    if int(w.sum()) != 10:
        raise ValueError("B generations 40-49 are not all logged")
    p1_total = int(pop[w].sum())
    r49 = row(49)
    p2 = int(g["survivors"][r49]) + int(g["births"][r49]) == 0
    s5 = int(pop[row(0)]) > 0
    s6 = float(pop[b_slice(g, 1, 49, n_a)].mean())
    food = (np.asarray(g["food_eaten"], float) + np.asarray(g["prey_eaten"], float))[w]
    pw = pop[w]
    has = pw > 0
    s4 = float((food[has] / pw[has]).mean()) if has.any() else float("nan")
    lvl = s1_level if s1_level is not None else math.ceil(0.9 * food_count)
    s1 = None
    for bg in range(1, 50):
        if pop[row(bg)] >= lvl:
            s1 = bg
            break
    out = {"P1_total": p1_total, "P1": p1_total / 10.0, "P2": bool(p2), "S5": bool(s5),
           "S6": s6, "S4": s4, "S4_n": int(has.sum()), "S1": s1,
           "B0_population": int(pop[row(0)]),
           "prey_eaten_B": int(np.asarray(g["prey_eaten"])[b_slice(g, 0, 49, n_a)].sum()),
           "steps_B": int(np.asarray(g["creature_steps"])[b_slice(g, 0, 49, n_a)].sum())}
    if late is not None:
        lo, hi = late
        wl = b_slice(g, lo, hi, n_a)
        if int(wl.sum()) == hi - lo + 1:
            rl = row(hi)
            out["late_P1_total"] = int(pop[wl].sum())
            out["late_P2"] = int(g["survivors"][rl]) + int(g["births"][rl]) == 0
    return out


def a_mean_population(g, lo=10, hi=99):
    """Mean population over A generations lo..hi (U3). hi is capped at the last A generation."""
    gen = np.asarray(g["gen"])
    m = (gen >= lo) & (gen <= hi)
    return float(np.asarray(g["population"])[m].mean())


def blob_fitness(b):
    """Blobs each blob puts into the next generation: alive at the end of the
    generation x (1 + [food items + prey >= 2]) -- exactly world.reproduce."""
    alive = np.asarray(b["alive_end"]).astype(bool)
    two = (np.asarray(b["food_balls"]) + np.asarray(b["prey"])) >= 2
    return alive.astype(np.int64) * (1 + two.astype(np.int64))
