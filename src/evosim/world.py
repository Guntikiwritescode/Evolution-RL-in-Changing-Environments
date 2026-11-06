"""Food placement, starting spots, and reproduction.

Rules follow docs/SPEC_part1.md. S1 = MinuteLabs evolution-simulator,
commit 55af956 (GPL-3.0); file refs are relative to its src/wasm/src/.
"""
import numpy as np

from .kernel import DEAD

MIN_POSITIVE = np.finfo(np.float64).tiny  # Rust f64::MIN_POSITIVE, S1 creature/mutatable.rs:44
TRAITS = ("speed", "size", "sense", "reach")
# Lowest allowed value per trait, S1 creature/mod.rs:57-60 and creature/mutatable.rs
TRAIT_FLOOR = {"speed": MIN_POSITIVE, "size": MIN_POSITIVE, "sense": 0.0, "reach": MIN_POSITIVE}


# ---------------------------------------------------------------- food
def place_food(env, rng):
    """Food for one generation. Returns (x, y) arrays.

    uniform: S1 simulation/mod.rs:133-139 and stage/mod.rs:46-51 (x, y ~ U[0, L)).
    clusters: Environment B, [GUESS] W7-W10 in the spec.
    """
    L = float(env["size"])
    n = int(env["food"])
    layout = env["layout"]
    if layout == "uniform":
        x = rng.uniform(0.0, L, n)
        y = rng.uniform(0.0, L, n)
        return x, y
    if layout == "clusters":
        k = int(env["n_clusters"])
        sd = float(env["cluster_sd"])
        margin = float(env["cluster_margin_sds"]) * sd
        per = np.full(k, n // k)
        per[: n % k] += 1
        cx = rng.uniform(margin, L - margin, k)
        cy = rng.uniform(margin, L - margin, k)
        x = np.repeat(cx, per) + rng.normal(0.0, sd, n)
        y = np.repeat(cy, per) + rng.normal(0.0, sd, n)
        return np.clip(x, 0.0, L), np.clip(y, 0.0, L)
    raise ValueError(f"unknown food layout {layout!r}")


def fixed_cluster_centers(env, rng):
    """Cluster centers drawn once, for the 'redraw_each_generation: false' option."""
    L = float(env["size"])
    k = int(env["n_clusters"])
    margin = float(env["cluster_margin_sds"]) * float(env["cluster_sd"])
    return rng.uniform(margin, L - margin, k), rng.uniform(margin, L - margin, k)


def place_food_fixed_clusters(env, centers, rng):
    n = int(env["food"])
    k = int(env["n_clusters"])
    sd = float(env["cluster_sd"])
    L = float(env["size"])
    per = np.full(k, n // k)
    per[: n % k] += 1
    x = np.repeat(centers[0], per) + rng.normal(0.0, sd, n)
    y = np.repeat(centers[1], per) + rng.normal(0.0, sd, n)
    return np.clip(x, 0.0, L), np.clip(y, 0.0, L)


# ---------------------------------------------------------------- starting spots
def random_edge_points(n, L, rng):
    """S1 api/square_world.rs:31-33 with stage/mod.rs:53-63: a random point,
    moved to the nearest edge (x edge if strictly closer, else y edge)."""
    x = rng.uniform(0.0, L, n)
    y = rng.uniform(0.0, L, n)
    hw = 0.5 * L
    ex = np.where(x > hw, L, 0.0)
    ey = np.where(y > hw, L, 0.0)
    use_x = np.abs(ex - x) < np.abs(ey - y)
    return np.where(use_x, ex, x), np.where(use_x, y, ey)


# ---------------------------------------------------------------- populations
def new_population(n, traits, L, rng, flee=1.0e12):
    """n identical blobs at random edge points (S1 defaults: src/store/simulation.js:18-27)."""
    hx, hy = random_edge_points(n, L, rng)
    pop = {t: np.full(n, float(traits[t])) for t in TRAITS}
    pop["flee"] = np.full(n, float(flee))
    pop["home_x"] = hx
    pop["home_y"] = hy
    pop["age"] = np.zeros(n, np.int64)
    pop["uid"] = np.arange(n, dtype=np.int64)
    pop["parent"] = np.full(n, -1, np.int64)
    pop["next_uid"] = n
    return pop


def pop_size(pop):
    return int(pop["speed"].shape[0])


def replace_on_edges(pop, L, rng):
    """Transfer to a new stage: fresh random edge points ([GUESS] T3)."""
    n = pop_size(pop)
    out = dict(pop)
    out["home_x"], out["home_y"] = random_edge_points(n, L, rng)
    return out


def reproduce(pop, result, mutation_sd, rng):
    """Between generations, S1 behaviours/reproduction.rs:10-34 and creature/mod.rs:125-168.

    Dead blobs are removed. Living blobs survive (age + 1) and start the next
    generation at their home spot. Living blobs with 2+ food (food items plus
    prey) also make exactly one offspring: each trait = parent value + normal
    noise with the trait's standard deviation, clipped to its floor."""
    state, food_balls, prey, hx, hy = result[0], result[1], result[2], result[3], result[4]
    alive = state != DEAD
    parent = alive & ((food_balls + prey) >= 2)
    n_kids = int(parent.sum())

    surv = {t: pop[t][alive] for t in TRAITS}
    surv["flee"] = pop["flee"][alive]
    surv["home_x"] = hx[alive]
    surv["home_y"] = hy[alive]
    surv["age"] = pop["age"][alive] + 1
    surv["uid"] = pop["uid"][alive]
    surv["parent"] = pop["parent"][alive]

    kids = {}
    for t in TRAITS:
        base = pop[t][parent]
        sd = float(mutation_sd[t])
        if sd > 0.0:
            base = base + rng.normal(0.0, sd, n_kids)
        kids[t] = np.maximum(base, TRAIT_FLOOR[t])
    kids["flee"] = pop["flee"][parent]
    kids["home_x"] = hx[parent]
    kids["home_y"] = hy[parent]
    kids["age"] = np.zeros(n_kids, np.int64)
    kids["uid"] = np.arange(pop["next_uid"], pop["next_uid"] + n_kids, dtype=np.int64)
    kids["parent"] = pop["uid"][parent]

    out = {k: np.concatenate([surv[k], kids[k]]) for k in surv}
    out["next_uid"] = pop["next_uid"] + n_kids
    return out
