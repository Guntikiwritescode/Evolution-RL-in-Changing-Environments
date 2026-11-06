"""Run phases and conditions (ET, GS deployment, transfer), logging every generation.

Every number saved comes from simulation output. Raw per-blob logs are kept so
any metric definition can be recomputed later without rerunning.
"""
import numpy as np

from . import world
from .kernel import DEAD, run_generation

GEN_FIELDS = ("gen", "phase", "population", "food_placed", "food_eaten", "prey_eaten",
              "survivors", "births", "deaths", "ticks", "creature_steps", "error")
BLOB_FIELDS = ("gen", "uid", "parent", "age", "speed", "size", "sense", "reach",
               "food_balls", "prey", "alive_end", "moves", "consumed")


class Log:
    """Per-generation and per-blob records for one run."""

    def __init__(self):
        self.gen_rows = {k: [] for k in GEN_FIELDS}
        self.blob_parts = {k: [] for k in BLOB_FIELDS}

    def add(self, gen, phase, pop, food_n, result):
        n = world.pop_size(pop)
        if result is None:  # extinct: nothing to simulate
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
            self.blob_parts["gen"].append(np.full(n, gen, np.int64))
            for t in ("uid", "parent", "age", "speed", "size", "sense", "reach"):
                self.blob_parts[t].append(np.asarray(pop[t]))
            self.blob_parts["food_balls"].append(food_balls)
            self.blob_parts["prey"].append(prey)
            self.blob_parts["alive_end"].append(alive)
            self.blob_parts["moves"].append(moves)
            self.blob_parts["consumed"].append(consumed)
        for k in GEN_FIELDS:
            self.gen_rows[k].append(row[k])

    def arrays(self, prefix=""):
        out = {}
        for k in GEN_FIELDS:
            v = self.gen_rows[k]
            out[prefix + "g_" + k] = np.asarray(v) if k != "phase" else np.asarray(v, dtype="U1")
        for k in BLOB_FIELDS:
            parts = self.blob_parts[k]
            out[prefix + "b_" + k] = np.concatenate(parts) if parts else np.zeros(0)
        return out


def run_phase(pop, env, n_gens, world_cfg, mutation_sd, rng, log, gen_offset, phase,
              food_rng_for_gen=None):
    """Run n_gens generations in one environment. Returns the population that
    would start the next generation (after the last reproduction step).

    food_rng_for_gen: None (Part 1 behavior: one stream `rng` drives food,
    behavior and reproduction) or a function g -> numpy Generator that gives
    generation g of this phase its own food stream (Part 2, PREREG_part2.md
    Section 2). g = -1 gives the stream for fixed cluster centers."""
    L = float(env["size"])
    food_rng = (lambda g: rng) if food_rng_for_gen is None else food_rng_for_gen
    fixed_centers = None
    if env["layout"] == "clusters" and not env.get("redraw_each_generation", True):
        fixed_centers = world.fixed_cluster_centers(env, food_rng(-1))
    for g in range(n_gens):
        frng = food_rng(g)
        if fixed_centers is not None:
            fx, fy = world.place_food_fixed_clusters(env, fixed_centers, frng)
        else:
            fx, fy = world.place_food(env, frng)
        if world.pop_size(pop) == 0:
            log.add(gen_offset + g, phase, pop, fx.shape[0], None)
            continue
        result = run_generation(
            pop["speed"], pop["size"], pop["sense"], pop["reach"], pop["flee"],
            pop["home_x"], pop["home_y"], fx, fy, L,
            float(world_cfg["energy_per_generation"]), bool(world_cfg["cannibalism"]),
            float(world_cfg["cannibalism_size_ratio"]), int(world_cfg["tick_cap"]), rng)
        log.add(gen_offset + g, phase, pop, fx.shape[0], result)
        if result[9] != 0:
            raise RuntimeError(f"generation {gen_offset + g}: S1 tick limit reached (error {result[9]})")
        pop = world.reproduce(pop, result, mutation_sd, rng)
    return pop


def run_lineage(start_traits, n_start, mutation_sd, cfg, seed_seq):
    """Train in A, then transfer to B. Used for ET and for GS deployment.

    seed_seq: a numpy SeedSequence for this run. Child streams: 0 = starting
    spots, 1 = Environment A, 2 = transfer placement, 3 = Environment B."""
    s_init, s_a, s_move, s_b = seed_seq.spawn(4)
    wc = cfg["world"]
    log = Log()
    env_a, env_b = cfg["env_A"], cfg["env_B"]
    pop = world.new_population(n_start, start_traits, float(env_a["size"]),
                               np.random.default_rng(s_init), flee=float(cfg["blob_default"]["flee"]))
    n_a = int(cfg["phases"]["train_generations"])
    n_b = int(cfg["phases"]["transfer_generations"])
    pop = run_phase(pop, env_a, n_a, wc, mutation_sd, np.random.default_rng(s_a), log, 0, "A")
    if n_b > 0:
        pop = world.replace_on_edges(pop, float(env_b["size"]), np.random.default_rng(s_move))
        run_phase(pop, env_b, n_b, wc, mutation_sd, np.random.default_rng(s_b), log, n_a, "B")
    return log


def et_start_traits(cfg):
    d = cfg["blob_default"]
    return {t: float(d[t]) for t in world.TRAITS}
