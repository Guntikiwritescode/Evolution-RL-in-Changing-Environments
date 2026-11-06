"""Zero-shot competence and ecological persistence (SPEC_part3.md, Section 7).

competence: mean food per agent in episodic episodes (16 agents, 500 steps),
  averaged over the given episode keys.
persistence: up to 32 agents placed in ecological mode (births without
  mutation, no population floor, no environment change) for a fixed number of
  steps; reports whether the population went extinct.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from evo2.agents.genome import Genome
from evo2.env.world import EnvParams, World
from evo2.eval.episodes import per_agent_act_fn, run_episode, run_random_episode, shared_act_fn
from evo2.methods.adapters import fixed_env_fn
from evo2.methods.oe import OE


def competence(world: World, ep_keys: jax.Array, params: EnvParams, act_fn: Callable,
               traits: jax.Array) -> jax.Array:
    """Mean food per agent over the episodes in ep_keys (one environment)."""
    food = jax.vmap(lambda k: run_episode(world, k, params, act_fn, traits))(ep_keys)
    return jnp.mean(food.astype(jnp.float32))


def random_competence(world: World, ep_keys: jax.Array, params: EnvParams) -> jax.Array:
    food = jax.vmap(lambda k: run_random_episode(world, k, params))(ep_keys)
    return jnp.mean(food.astype(jnp.float32))


def genome_competence(world: World, ep_keys, params: EnvParams, genomes: Genome) -> jax.Array:
    """Competence of a group of individual genomes (leading axis = n_slots)."""
    return competence(world, ep_keys, params, per_agent_act_fn(genomes.brain), genomes.traits)


# Compiled once per world (worlds are passed as static arguments, so callers
# should reuse one World object per mode; see evo2.eval.protocol.get_world).
# Without this every call compiled a new episode loop, and a long evaluation
# ran the CPU compiler out of room (an XLA "failed to materialize symbols" error).

@partial(jax.jit, static_argnums=0)
def competence_shared(world: World, ep_keys, params: EnvParams, brain: dict, traits) -> jax.Array:
    """Competence of 16 copies of one policy (compiled)."""
    return competence(world, ep_keys, params, shared_act_fn(brain), traits)


@partial(jax.jit, static_argnums=0)
def competence_genomes(world: World, ep_keys, params: EnvParams, genomes: Genome) -> jax.Array:
    """Competence of a group of individual genomes (compiled)."""
    return genome_competence(world, ep_keys, params, genomes)


random_competence_jit = jax.jit(random_competence, static_argnums=0)


def broadcast_genome(brain: dict, traits: jax.Array, n: int) -> Genome:
    """n identical copies of one policy (PPO, ES)."""
    b = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), brain)
    return Genome(brain=b, traits=jnp.broadcast_to(traits, (n, 3)))


@partial(jax.jit, static_argnums=0, static_argnames=("n_start", "steps"))
def persistence(world_eco: World, key, params: EnvParams, genomes: Genome, n_start: int = 32,
                steps: int = 5000) -> dict:
    """Ecological persistence. `genomes` has a leading axis of n_start; they
    fill the first n_start of the ecological world's slots."""
    n = world_eco.cfg.n_slots
    pad = jax.tree_util.tree_map(
        lambda x: jnp.concatenate([x, jnp.broadcast_to(x[:1], (n - n_start,) + x.shape[1:])]), genomes)
    oe = OE({"CHANGE_EVERY": 0, "MIN_POP": 0}, world_eco, fixed_env_fn(params),
            evolve_body=False, mutate_on=False)
    s = oe.init(key, pop=pad, n_alive=n_start)
    s, log = oe.run_chunk(s, steps)
    alive = log["alive"]
    dead = alive == 0
    # steps after which at least one agent was still alive
    survived = jnp.where(jnp.any(dead), jnp.argmax(dead), steps)
    return {"extinct": jnp.any(dead), "steps_survived": survived,
            "final_alive": alive[-1], "mean_alive": jnp.mean(alive.astype(jnp.float32)),
            "max_alive": jnp.max(alive), "births": jnp.sum(log["births"])}
