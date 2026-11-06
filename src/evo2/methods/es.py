"""ES-DR: OpenAI-ES over one policy (SPEC_part3.md, Section 5).

Uses evosax 0.3.2 `Open_ES` (Apache-2.0) with antithetic sampling and
centred-rank fitness shaping. evosax minimizes, so the fitness passed to it
is minus the score (higher score = better).

`ES` is the generic loop: every candidate in a generation is scored by
`score_fn(solution, key) -> (score, agent_steps)` with the same key (common
random numbers). A runner started with `ctx` (any arrays; in the world, a
fixed environment) calls `score_fn(solution, key, ctx)` instead, so the
environment is data in the state, not a constant in the compiled program.
Compute is counted as the agent-steps the score function
reports (in the world: agents alive at the start of each step, the count
every Part 3 method uses). The same
loop runs in the Part 3 world (`world_es`) and on the sanity task.

In the world: each generation, every candidate plays the same EPISODES
episodes (same environments, same reset and step keys), each with 16 copies
of the candidate. Score = mean over episodes of mean food per agent. With
evolve_body=True (option (a)) the solution also holds 3 numbers that map to
the body traits through a sigmoid; they start at the default body.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms import Open_ES
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from flax import struct

from evo2.agents.brain import init_mlp
from evo2.env.world import World
from evo2.eval.episodes import run_episode_counted, shared_act_fn

DEFAULTS = {
    "POP_SIZE": 64,     # even (antithetic pairs)
    "SIGMA": 0.05,      # noise sd, constant
    "LR": 0.01,         # Adam step size on the mean
    "EPISODES": 2,      # episodes per candidate per generation (world only)
}


@struct.dataclass
class ESRunner:
    state: object       # evosax state
    key: jax.Array
    generation: jax.Array
    ctx: object = None  # passed to score_fn when not None (a fixed environment)


class ES:
    def __init__(self, cfg: dict, template: dict, score_fn: Callable, max_agent_steps_per_candidate: int):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown ES settings {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        self.score_fn = score_fn
        self.strategy = Open_ES(
            population_size=c["POP_SIZE"], solution=template, use_antithetic_sampling=True,
            optimizer=optax.adam(c["LR"]), std_schedule=optax.constant_schedule(c["SIGMA"]),
            fitness_shaping_fn=centered_rank_fitness_shaping_fn)
        self.es_params = self.strategy.default_params
        self.max_agent_steps_per_generation = c["POP_SIZE"] * max_agent_steps_per_candidate

    def init(self, key, solution: dict, ctx=None) -> ESRunner:
        k_init, k_run = jax.random.split(key)
        state = self.strategy.init(k_init, solution, self.es_params)
        return ESRunner(state=state, key=k_run, generation=jnp.int32(0), ctx=ctx)

    def mean_solution(self, r: ESRunner) -> dict:
        return self.strategy.get_mean(r.state)

    def _generation(self, r: ESRunner, _):
        key, k_ask, k_score, k_tell = jax.random.split(r.key, 4)
        pop, state = self.strategy.ask(k_ask, r.state, self.es_params)
        if r.ctx is None:
            score, steps = jax.vmap(lambda s: self.score_fn(s, k_score))(pop)  # same key: common random numbers
        else:
            score, steps = jax.vmap(lambda s: self.score_fn(s, k_score, r.ctx))(pop)
        state, _ = self.strategy.tell(k_tell, pop, -score, state, self.es_params)
        metrics = {"mean_score": jnp.mean(score), "max_score": jnp.max(score),
                   "min_score": jnp.min(score), "agent_steps": jnp.sum(steps).astype(jnp.int32)}
        return r.replace(state=state, key=key, generation=r.generation + 1), metrics

    def run_chunk(self, r: ESRunner, n_generations: int):
        return jax.lax.scan(self._generation, r, None, n_generations)


class WorldPolicy:
    """How an ES solution becomes a brain and a body in the Part 3 world."""

    def __init__(self, world: World, evolve_body: bool):
        self.world = world
        self.evolve_body = evolve_body
        lo, hi, d = world.trait_lo, world.trait_hi, world.default_traits
        frac = (d - lo) / (hi - lo)
        self._body_offset = jnp.log(frac) - jnp.log1p(-frac)  # logit: z = 0 gives the default body

    def initial_solution(self, key) -> dict:
        sol = {"brain": init_mlp(key, self.world.cfg.obs_dim)}
        if self.evolve_body:
            sol["body_z"] = jnp.zeros((3,), jnp.float32)
        return sol

    def traits_of(self, solution: dict) -> jax.Array:
        if not self.evolve_body:
            return self.world.default_traits
        lo, hi = self.world.trait_lo, self.world.trait_hi
        return lo + (hi - lo) * jax.nn.sigmoid(solution["body_z"] + self._body_offset)


def world_es(cfg: dict, world: World, env_fn: Callable | None, evolve_body: bool) -> tuple[ES, WorldPolicy]:
    """ES in the Part 3 world (episodic mode). With env_fn=None every episode
    is played in the fixed environment the runner was started with
    (es.init(key, solution, ctx=env))."""
    if world.cfg.ecological:
        raise ValueError("ES runs in episodic mode")
    full = {**DEFAULTS, **cfg}
    pol = WorldPolicy(world, evolve_body)
    n, e = world.cfg.n_slots, full["EPISODES"]

    def score_fn(solution, key, env=None):
        if (env_fn is None) != (env is not None):
            raise ValueError("a fixed environment is needed exactly when env_fn is None")
        k_env, k_eps = jax.random.split(key)
        if env_fn is None:
            env_params = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (e,) + jnp.shape(x)), env)
        else:
            env_params = jax.vmap(env_fn)(jax.random.split(k_env, e))
        traits = jnp.broadcast_to(pol.traits_of(solution), (n, 3))
        act = shared_act_fn(solution["brain"])
        food, steps = jax.vmap(lambda k, p: run_episode_counted(world, k, p, act, traits))(
            jax.random.split(k_eps, e), env_params)
        return jnp.mean(food.astype(jnp.float32)), jnp.sum(steps)

    template = pol.initial_solution(jax.random.PRNGKey(0))
    es = ES(cfg, template, score_fn, max_agent_steps_per_candidate=n * world.cfg.episode_len * e)
    return es, pol
