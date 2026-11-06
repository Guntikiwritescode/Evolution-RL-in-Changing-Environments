"""Grid search (GS condition): score every design as clones in Environment A, pick the best.

Settings are [GUESS] items GS2-GS5 in docs/SPEC_part1.md.
"""
import itertools

import numpy as np

from . import world
from .runner import Log, run_phase


def grid_axes(cfg):
    """8 points per trait, evenly spaced on a log scale between low and high
    multiples of the ET default (GS3)."""
    g = cfg["gs"]
    d = cfg["blob_default"]
    k = int(g["grid_points"])
    lo, hi = float(g["grid_low_factor"]), float(g["grid_high_factor"])
    factors = np.exp(np.linspace(np.log(lo), np.log(hi), k))
    return {t: float(d[t]) * factors for t in g["traits"]}


def grid_designs(cfg):
    """All designs in grid order (first trait slowest). Traits not searched keep the default."""
    axes = grid_axes(cfg)
    names = list(axes)
    out = []
    for combo in itertools.product(*(axes[t] for t in names)):
        design = {t: float(cfg["blob_default"][t]) for t in world.TRAITS}
        design.update(dict(zip(names, combo)))
        out.append(design)
    return out


def score_log(log, n_gens):
    """GS4: mean over the evaluation generations of food per creature (M1, food
    items + prey). A generation with no blobs left scores 0."""
    a = log.arrays()
    pop = a["g_population"].astype(float)
    food = (a["g_food_eaten"] + a["g_prey_eaten"]).astype(float)
    fit = np.where(pop > 0, food / np.maximum(pop, 1), 0.0)
    return float(fit[:n_gens].mean())


def evaluate_design(args):
    """Run one design as clones (no mutation) for n_gens generations in A."""
    design, cfg, seed_entropy, n_gens = args
    rng_init, rng_a = (np.random.default_rng(s) for s in np.random.SeedSequence(seed_entropy).spawn(2))
    env = cfg["env_A"]
    pop = world.new_population(int(cfg["gs"]["n_clones"]), design, float(env["size"]), rng_init,
                               flee=float(cfg["blob_default"]["flee"]))
    zero_sd = {t: 0.0 for t in world.TRAITS}
    log = Log()
    run_phase(pop, env, n_gens, cfg["world"], zero_sd, rng_a, log, 0, "A")
    a = log.arrays()
    return {"score": score_log(log, n_gens),
            "creature_steps": int(a["g_creature_steps"].sum()),
            "ticks": int(a["g_ticks"].sum()),
            "final_population": int(a["g_population"][-1])}


def grid_search(cfg, seed_entropy, pool=None):
    """Score every design with the same seed (common random numbers), then pick
    the highest score; ties go to the first design in grid order."""
    designs = grid_designs(cfg)
    n_gens = int(cfg["gs"]["eval_generations"])
    jobs = [(d, cfg, seed_entropy, n_gens) for d in designs]
    results = list(pool.imap(evaluate_design, jobs, chunksize=4)) if pool else [evaluate_design(j) for j in jobs]
    scores = np.array([r["score"] for r in results])
    best = int(np.argmax(scores))
    return {"designs": designs, "results": results, "best_index": best, "best": designs[best],
            "creature_steps": int(sum(r["creature_steps"] for r in results)),
            "n_ties_at_best": int(np.sum(scores == scores[best]))}
