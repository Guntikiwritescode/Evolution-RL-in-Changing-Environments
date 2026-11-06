"""OE: open-ended evolution in an ecological world (SPEC_part3.md, Section 5).

One population lives in one shared world (ecological mode: births when energy
is high enough, death by starvation or old age, no fitness function). Every
child's genome is its parent's, mutated. Every `CHANGE_EVERY` steps the world
moves to a new environment from `env_fn`; agents stay where they are (clipped
into the arena) and food is laid out afresh. OE-fixed is the same with
CHANGE_EVERY = 0 (the environment never changes).

Population floor. Preliminary runs on seeds 3980-3984 (2 Nov 2026, no floor,
training distribution) went extinct every time, after 113, 176, 297, 1,292 and 2,166 steps (0, 0, 3, 29
and 173 births before that). With MIN_POP > 0, whenever fewer than MIN_POP
agents are alive after a step, the empty slots are filled up to MIN_POP with
mutated copies of agents chosen uniformly at random (no fitness: each equally
likely) from those alive and not born this step, placed at random positions
with birth energy. This energy comes from outside the world; every such
"floor birth" is counted and reported. With MIN_POP = 0 a population can go
extinct, and the run then continues with an empty world (counted).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from evo2.agents.brain import apply_per_agent
from evo2.agents.genome import Genome, MutationConfig, apply_births, init_population
from evo2.env.world import EnvParams, World, WorldState

DEFAULTS = {
    "N_INITIAL": 32,
    "CHANGE_EVERY": 2000,   # 0 = never (OE-fixed)
    "WEIGHT_SD": 0.02,
    "TRAIT_FRAC": 0.05,
    "MIN_POP": 16,          # population floor; 0 = none
}


@struct.dataclass
class OEState:
    world: WorldState
    pop: Genome
    env: EnvParams
    key: jax.Array
    t: jax.Array              # steps so far
    since_change: jax.Array   # steps since the last environment change


class OE:
    def __init__(self, cfg: dict, world: World, env_fn: Callable | None, evolve_body: bool,
                 mutate_on: bool = True):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown OE settings {sorted(unknown)}")
        if not world.cfg.ecological:
            raise ValueError("OE runs in ecological mode")
        self.cfg = {**DEFAULTS, **cfg}
        if env_fn is None and self.cfg["CHANGE_EVERY"] > 0:
            raise ValueError("OE with environment changes needs env_fn")
        self.world = world
        self.env_fn = env_fn
        self.mcfg = MutationConfig(weight_sd=self.cfg["WEIGHT_SD"], trait_frac=self.cfg["TRAIT_FRAC"],
                                   evolve_body=evolve_body)
        self.mutate_on = mutate_on

    def init(self, key, pop: Genome | None = None, n_alive: int | None = None,
             env: EnvParams | None = None) -> OEState:
        """Start a run. Without `pop`, N_INITIAL new agents with default bodies.
        Without `env`, the first environment comes from env_fn (same key either way)."""
        k_pop, k_env, k_reset, k_run = jax.random.split(key, 4)
        n = self.world.cfg.n_slots
        if pop is None:
            pop = init_population(k_pop, n, self.world.cfg.obs_dim, self.world.default_traits)
            n_alive = self.cfg["N_INITIAL"]
        if env is None:
            if self.env_fn is None:
                raise ValueError("OE without env_fn needs the environment: init(key, ..., env=...)")
            env = self.env_fn(k_env)
        ws = self.world.reset(k_reset, env, pop.traits, n_alive)
        return OEState(world=ws, pop=pop, env=env, key=k_run, t=jnp.int32(0),
                       since_change=jnp.int32(0))

    def move_to(self, key, s: OEState, env: EnvParams) -> OEState:
        """Put a running population into a given environment (adaptation phase)."""
        ws = self.world.change_env(key, s.world, env)
        return s.replace(world=ws, env=env, since_change=jnp.int32(0))

    def _floor_parents(self, key, ws: WorldState, newborn, alive_start):
        """Parent index per slot for floor births (-1 = none): empty slots are
        filled up to MIN_POP with copies of agents chosen uniformly from those
        alive now that were not born this step (or, if none, those alive at the
        start of the step)."""
        n = self.world.cfg.n_slots
        older = ws.alive & ~newborn
        pool = jnp.where(jnp.any(older), older, alive_start)
        n_pool = jnp.sum(pool)
        need = jnp.where(n_pool > 0, jnp.maximum(self.cfg["MIN_POP"] - jnp.sum(ws.alive), 0), 0)
        free_order = jnp.argsort(jnp.where(ws.alive, n + jnp.arange(n), jnp.arange(n)))
        pool_order = jnp.argsort(jnp.where(pool, jnp.arange(n), n + jnp.arange(n)))
        pick = jax.random.randint(key, (n,), 0, jnp.maximum(n_pool, 1))
        slot = jnp.where(jnp.arange(n) < need, free_order, n)
        parent_idx = jnp.full((n,), -1, jnp.int32).at[slot].set(
            pool_order[pick].astype(jnp.int32), mode="drop")
        return parent_idx, need.astype(jnp.int32)

    def _step(self, s: OEState, _):
        w = self.world
        n = w.cfg.n_slots
        key, k_step, k_mut, k_env, k_chg, k_floor, k_spawn = jax.random.split(s.key, 7)
        obs = w.observe(s.world, s.env)
        action = apply_per_agent(s.pop.brain, obs)
        ws, info = w.step(k_step, s.world, s.env, action)
        newborn = info["parent_idx"] >= 0
        extinct = ~jnp.any(ws.alive)

        floor_idx = jnp.full((n,), -1, jnp.int32)
        floor_births = jnp.int32(0)
        if self.cfg["MIN_POP"] > 0:
            floor_idx, floor_births = self._floor_parents(k_floor, ws, newborn, info["alive_start"])
        # one mutation pass for both kinds of birth (their slots never overlap)
        parent_all = jnp.where(newborn, info["parent_idx"], floor_idx)
        # mutation work is skipped on steps without any birth (most steps)
        pop = jax.lax.cond(
            jnp.any(parent_all >= 0),
            lambda p: apply_births(k_mut, p, parent_all, self.mcfg, w.trait_lo, w.trait_hi,
                                   mutate_on=self.mutate_on),
            lambda p: p, s.pop)
        ws = ws.replace(traits=jnp.where(newborn[:, None], pop.traits, ws.traits))
        if self.cfg["MIN_POP"] > 0:
            ws = w.spawn(k_spawn, ws, s.env, floor_idx >= 0, pop.traits)

        since = s.since_change + 1
        env = s.env
        every = self.cfg["CHANGE_EVERY"]
        changed = jnp.bool_(False)
        if every > 0:
            changed = since >= every

            def change(args):
                ws_, _ = args
                new_env = self.env_fn(k_env)
                return w.change_env(k_chg, ws_, new_env), new_env

            ws, env = jax.lax.cond(changed, change, lambda args: args, (ws, env))
            since = jnp.where(changed, 0, since)

        new = OEState(world=ws, pop=pop, env=env, key=key, t=s.t + 1, since_change=since)
        log = {
            "agent_steps": jnp.sum(info["alive_start"]).astype(jnp.int32),
            "alive": jnp.sum(ws.alive).astype(jnp.int32),
            "births": info["n_births"],
            "floor_births": floor_births,
            "deaths": jnp.sum(info["died"]).astype(jnp.int32),
            "food": jnp.sum(info["n_eat"]).astype(jnp.int32),
            "extinct": extinct,
            "env_changed": changed,
        }
        return new, log

    def run_chunk(self, s: OEState, n_steps: int):
        """n_steps steps in one compiled call. Returns (state, per-step logs)."""
        return jax.lax.scan(self._step, s, None, n_steps)

    @staticmethod
    def chunk_summary(log: dict) -> dict:
        """Plain Python totals for one chunk (agent-steps summed exactly)."""
        g = {k: np.asarray(jax.device_get(v)) for k, v in log.items()}
        return {
            "steps": int(g["alive"].shape[0]),
            "agent_steps": int(g["agent_steps"].astype(np.int64).sum()),
            "mean_alive": float(g["alive"].mean()),
            "min_alive": int(g["alive"].min()),
            "births": int(g["births"].astype(np.int64).sum()),
            "floor_births": int(g["floor_births"].astype(np.int64).sum()),
            "deaths": int(g["deaths"].astype(np.int64).sum()),
            "food": int(g["food"].astype(np.int64).sum()),
            "steps_all_dead_before_floor": int(g["extinct"].sum()),
            "env_changes": int(g["env_changed"].sum()),
        }

    @staticmethod
    def sample_agents(key, s: OEState, n: int):
        """n agents from the living population: without replacement when at
        least n are alive, otherwise every living agent and then repeats.
        Returns (genomes with leading axis n, number alive)."""
        alive = s.world.alive
        n_alive = jnp.sum(alive)
        prio = jnp.where(alive, jax.random.uniform(key, alive.shape), jnp.inf)
        order = jnp.argsort(prio)
        idx = order[jnp.arange(n) % jnp.maximum(n_alive, 1)]
        return jax.tree_util.tree_map(lambda x: x[idx], s.pop), n_alive
