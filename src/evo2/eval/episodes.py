"""Running whole episodes with fixed policies (SPEC_part3.md, Sections 3.3 and 7).

A policy here is an `act_fn(obs) -> raw actions`: obs (n_slots, obs_dim) to
(n_slots, 2). Policies act deterministically (no sampling): the brain's
output is the action. The world squashes it into range.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from evo2.agents.brain import apply_mlp, apply_per_agent
from evo2.env.world import EnvParams, World


def shared_act_fn(brain: dict) -> Callable:
    """Every agent uses the same weights."""
    return lambda obs: apply_mlp(brain, obs)


def per_agent_act_fn(brains: dict) -> Callable:
    """Agent i uses brains[...][i]."""
    return lambda obs: apply_per_agent(brains, obs)


def run_episode_counted(world: World, key, params: EnvParams, act_fn: Callable, traits: jax.Array,
                        n_alive: int | None = None):
    """One episodic-mode episode. Returns (food eaten per slot (n_slots,),
    agent-steps: the number of (agent, step) pairs with the agent alive at
    the start of the step)."""
    cfg = world.cfg
    n_alive = cfg.n_slots if n_alive is None else n_alive
    k_reset, k_steps = jax.random.split(key)
    s = world.reset(k_reset, params, traits, n_alive)

    def body(s, k):
        a = act_fn(world.observe(s, params))
        s, info = world.step(k, s, params, a)
        return s, jnp.sum(info["alive_start"]).astype(jnp.int32)

    s, alive = jax.lax.scan(body, s, jax.random.split(k_steps, cfg.episode_len))
    return s.eaten, jnp.sum(alive)


def run_episode(world: World, key, params: EnvParams, act_fn: Callable, traits: jax.Array,
                n_alive: int | None = None) -> jax.Array:
    """One episodic-mode episode. Returns food eaten per slot, shape (n_slots,)."""
    return run_episode_counted(world, key, params, act_fn, traits, n_alive)[0]


def run_random_episode(world: World, key, params: EnvParams) -> jax.Array:
    """The random policy for normalization: default body for every agent;
    turn and speed drawn uniformly within their bounds each step."""
    cfg = world.cfg
    k_reset, k_steps = jax.random.split(key)
    traits = jnp.tile(world.default_traits, (cfg.n_slots, 1))
    s = world.reset(k_reset, params, traits, cfg.n_slots)
    edge = 1.0 - 1e-6  # so that tanh(arctanh(u)) = u stays inside (-1, 1)

    def body(s, k):
        k_a, k_w = jax.random.split(k)
        u = jax.random.uniform(k_a, (cfg.n_slots, 2), minval=-edge, maxval=edge)
        s, _ = world.step(k_w, s, params, jnp.arctanh(u))
        return s, None

    s, _ = jax.lax.scan(body, s, jax.random.split(k_steps, cfg.episode_len))
    return s.eaten
