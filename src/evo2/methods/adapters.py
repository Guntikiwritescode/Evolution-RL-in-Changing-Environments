"""Batched episodic worlds for the training methods (PPO, ES).

`EpisodicWorlds` runs B worlds in episodic mode (16 agents, up to 500 steps,
no births). An episode ends after 500 steps or as soon as every agent in it
is dead (so no compute goes to empty worlds, and worlds drift out of step).
When an episode ends, that world restarts at once in a new environment from
`env_fn` (a draw from the train distribution for the DR methods, or always
the canonical environment for PPO-fixed training). With env_fn=None (the
adaptation phase) the environment is given to
`reset` and every world restarts in the environment it already has: the
environment is then data in the state, not a constant in the compiled
program, so one compiled program serves every fixed environment.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from evo2.env.world import EnvParams, World


def fixed_env_fn(params: EnvParams) -> Callable:
    """An env_fn that always returns the same environment."""
    return lambda key: params


def tree_where(mask_b, a, b):
    """Per-world choice between two batched pytrees: mask_b has shape (B,)."""
    def pick(x, y):
        m = mask_b.reshape(mask_b.shape + (1,) * (x.ndim - 1))
        return jnp.where(m, x, y)
    return jax.tree_util.tree_map(pick, a, b)


class EpisodicWorlds:
    """PPO adapter: reset/step over B worlds with automatic restarts."""

    def __init__(self, world: World, n_envs: int, env_fn: Callable | None, traits: jax.Array):
        if world.cfg.ecological:
            raise ValueError("EpisodicWorlds needs an episodic world")
        self.world = world
        self.n_envs = n_envs
        self.env_fn = env_fn          # None: fixed environment, given to reset
        self.n_agents = world.cfg.n_slots
        # traits: (3,) one body for every agent, or (n_agents, 3)
        t = jnp.asarray(traits, jnp.float32)
        self.traits = jnp.broadcast_to(t, (self.n_agents, 3)) if t.ndim == 1 else t

    def _fresh(self, key, params=None):
        """New worlds: in new environments from env_fn, or (fixed mode) in the
        per-world environments `params` (the key split is the same either way)."""
        k_env, k_reset = jax.random.split(key)
        if self.env_fn is not None:
            params = jax.vmap(self.env_fn)(jax.random.split(k_env, self.n_envs))
        state = jax.vmap(lambda k, p: self.world.reset(k, p, self.traits, self.n_agents))(
            jax.random.split(k_reset, self.n_envs), params)
        return state, params

    def reset(self, key, env: EnvParams | None = None):
        """New worlds. In fixed mode (env_fn=None) `env` is required: every
        world gets it; otherwise `env` must be None."""
        if (self.env_fn is None) != (env is not None):
            raise ValueError("pass env exactly when the adapter has no env_fn")
        params = None
        if env is not None:
            params = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.n_envs,) + jnp.shape(x)), env)
        state, params = self._fresh(key, params)
        obs = jax.vmap(self.world.observe)(state, params)
        return (state, params), obs

    def step(self, key, env_state, action):
        state, params = env_state
        k_step, k_new = jax.random.split(key)
        new, info = jax.vmap(self.world.step)(jax.random.split(k_step, self.n_envs), state, params, action)
        reward = info["n_eat"].astype(jnp.float32)
        valid = info["alive_start"]
        ep_end = (new.t >= self.world.cfg.episode_len) | ~jnp.any(new.alive, axis=1)
        done = info["died"] | ep_end[:, None]
        ep_score = jnp.mean(new.eaten.astype(jnp.float32), axis=1)

        def restart(_):
            fresh_state, fresh_params = self._fresh(k_new, params)
            return tree_where(ep_end, fresh_state, new), tree_where(ep_end, fresh_params, params)

        # building fresh worlds is skipped on steps where no episode ended
        state2, params2 = jax.lax.cond(jnp.any(ep_end), restart, lambda _: (new, params), None)
        obs = jax.vmap(self.world.observe)(state2, params2)
        return (state2, params2), obs, reward, done, valid, {"ep_end": ep_end, "ep_score": ep_score}
