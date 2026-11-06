"""Part 2 simulation helpers (docs/PREREG_part2.md, Sections 2 and 4).

Same world rules as Part 1 (runner.run_phase, kernel, world). What is new:
- food placement has its own stream per generation (runner.run_phase's
  food_rng_for_gen), and deployment streams are chosen by the run seed only,
  so every group with the same run seed sees the same food layouts;
- settings (world, mutation) can differ by phase;
- groups can start from a list of designs, or directly in B;
- the population at the end of A can be kept and restarted from.
"""
import copy

import numpy as np

from . import streams, world
from .kernel import DEAD
from .runner import GEN_FIELDS, Log, run_phase

ZERO_SD = {t: 0.0 for t in world.TRAITS}


class GenLog(Log):
    """Generation-level records only (used by the searches, which run
    ~150,000 short runs per seed and only need population and food)."""

    def add(self, gen, phase, pop, food_n, result):
        n = world.pop_size(pop)
        if result is None:
            row = dict(gen=gen, phase=phase, population=0, food_placed=food_n, food_eaten=0,
                       prey_eaten=0, survivors=0, births=0, deaths=0, ticks=0,
                       creature_steps=0, error=0)
        else:
            state, food_balls, prey, _, _, moves, consumed, eaten, ticks, err = result
            alive = state != DEAD
            row = dict(gen=gen, phase=phase, population=n, food_placed=food_n,
                       food_eaten=int(eaten.sum()), prey_eaten=int(prey.sum()),
                       survivors=int(alive.sum()),
                       births=int((alive & ((food_balls + prey) >= 2)).sum()),
                       deaths=int((~alive).sum()), ticks=int(ticks),
                       creature_steps=int(moves.sum()), error=int(err))
        for k in GEN_FIELDS:
            self.gen_rows[k].append(row[k])


def population_from_designs(designs, L, spots_rng, flee):
    """One blob per design (a list of dicts with the four traits), at random edge points."""
    n = len(designs)
    pop = world.new_population(n, designs[0] if n else {t: 1.0 for t in world.TRAITS}, L,
                               spots_rng, flee=flee)
    for t in world.TRAITS:
        pop[t] = np.array([float(d[t]) for d in designs], dtype=np.float64)
    return pop


def clones_of(pop, index):
    """Same number of blobs as pop, all with the traits of blob `index`.
    Positions, ages and ids are kept; only traits change."""
    out = copy.deepcopy(pop)
    n = world.pop_size(pop)
    for t in world.TRAITS + ("flee",):
        out[t] = np.full(n, float(pop[t][index]))
    return out


def empty_like(pop):
    out = copy.deepcopy(pop)
    keep = np.zeros(world.pop_size(pop), bool)
    for k, v in pop.items():
        if isinstance(v, np.ndarray):
            out[k] = v[keep]
    return out


def run_streamed(pop, env, world_cfg, mutation_sd, n_gens, run_seed, purpose, key, log,
                 gen_offset, phase):
    """Run one phase with behavior and food drawn from named streams:
    behavior = (purpose, *key, BEHAVIOR); food for generation g = (purpose, *key, FOOD, g + 1)."""
    behav = streams.rng(run_seed, purpose, *key, streams.BEHAVIOR)
    food = streams.food_rng_for_gen(run_seed, purpose, *key)
    return run_phase(pop, env, n_gens, world_cfg, mutation_sd, behav, log, gen_offset, phase,
                     food_rng_for_gen=food)


def spots_rng(run_seed, purpose, *key):
    return streams.rng(run_seed, purpose, *key, streams.SPOTS)


# ---------------------------------------------------------------- deployment
def deploy_A(designs, cfg, run_seed, mutation_sd, log, world_cfg=None):
    """Start from `designs` (one per blob) at random edge points of A and run
    all of A on the deployment streams. Returns the population that starts
    B generation 0 (after A generation 99's reproduction, with A's mutation)."""
    env = cfg["env_A"]
    pa = streams.PHASE_ID["A"]
    pop = population_from_designs(designs, float(env["size"]),
                                  spots_rng(run_seed, "deploy", pa),
                                  float(cfg["blob_default"]["flee"]))
    n = int(cfg["phases"]["train_generations"])
    return run_streamed(pop, env, world_cfg or cfg["world"], mutation_sd, n, run_seed,
                        "deploy", (pa,), log, 0, "A")


def deploy_B(pop_end_A, env_B, world_cfg_B, mutation_sd_B, n_gens, cfg, run_seed, log):
    """Move the end-of-A population to random edge points of B and run B
    on the B-phase deployment streams (shared by every group and B world)."""
    pb = streams.PHASE_ID["B"]
    pop = world.replace_on_edges(pop_end_A, float(env_B["size"]), spots_rng(run_seed, "deploy", pb))
    n_a = int(cfg["phases"]["train_generations"])
    return run_streamed(pop, env_B, world_cfg_B, mutation_sd_B, n_gens, run_seed,
                        "deploy", (pb,), log, n_a, "B")


def start_in_B(designs, env_B, world_cfg_B, mutation_sd_B, n_gens, cfg, run_seed, log):
    """ORACLE_B and ET_scratchB: start directly in B at random edge points,
    on the B-phase deployment streams. Generations are numbered as B
    generations (100 + g) so outcomes line up with the other groups."""
    pb = streams.PHASE_ID["B"]
    pop = population_from_designs(designs, float(env_B["size"]),
                                  spots_rng(run_seed, "deploy", pb),
                                  float(cfg["blob_default"]["flee"]))
    n_a = int(cfg["phases"]["train_generations"])
    return run_streamed(pop, env_B, world_cfg_B, mutation_sd_B, n_gens, run_seed,
                        "deploy", (pb,), log, n_a, "B")


def pick_clone_index(pop, run_seed, setting_id=0):
    """ET_cloneB: one blob of the end-of-A population, each equally likely."""
    n = world.pop_size(pop)
    if n == 0:
        return -1
    return int(streams.rng(run_seed, "clone_pick", setting_id).integers(0, n))


def scaled_sd(mutation_sd, factor):
    return {t: float(v) * float(factor) for t, v in mutation_sd.items()}
