"""Standard tasks for checking the baselines before use (SPEC_part3.md, Section 5).

The same PPO and ES code that runs in the Part 3 world is run on gymnax
Pendulum-v1 (gymnax 1.0.1, Apache-2.0). The pass targets and their sources
are fixed in docs/PREREG_part3.md.
"""
from __future__ import annotations

import gymnax
import jax
import jax.numpy as jnp


class GymnaxAdapter:
    """PPO adapter for a single-agent gymnax task with continuous actions.
    Actions are clipped to the action space, as PureJaxRL's ClipAction does."""

    def __init__(self, env_name: str, n_envs: int):
        self.env, self.params = gymnax.make(env_name)
        self.n_envs = n_envs
        space = self.env.action_space(self.params)
        self.low, self.high = float(space.low), float(space.high)
        self.obs_dim = int(self.env.observation_space(self.params).shape[0])
        self.act_dim = int(space.shape[0])

    def reset(self, key):
        obs, st = jax.vmap(self.env.reset, in_axes=(0, None))(jax.random.split(key, self.n_envs), self.params)
        ret = jnp.zeros((self.n_envs,))
        return (st, ret), obs[:, None, :]

    def step(self, key, env_state, action):
        st, ret = env_state
        a = jnp.clip(action[:, 0, :], self.low, self.high)
        # gymnax 1.0.1 returns (obs, state, reward, terminated, truncated, info); like
        # PureJaxRL, both count as the end of an episode
        obs, st, reward, term, trunc, _ = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(key, self.n_envs), st, a, self.params)
        done = term | trunc
        ret = ret + reward
        info = {"ep_end": done, "ep_score": ret}
        ret = jnp.where(done, 0.0, ret)
        ones = jnp.ones((self.n_envs, 1), bool)
        return (st, ret), obs[:, None, :], reward[:, None], done[:, None], ones, info


def pendulum_returns(policy_fn, keys) -> jax.Array:
    """Undiscounted return of a deterministic policy on Pendulum-v1, one
    200-step episode per key. Jittable."""
    env, params = gymnax.make("Pendulum-v1")
    space = env.action_space(params)
    low, high = float(space.low), float(space.high)

    def one(k):
        k_r, k_s = jax.random.split(k)
        obs, st = env.reset(k_r, params)

        def body(c, kk):
            obs, st, total = c
            a = jnp.clip(policy_fn(obs), low, high)
            obs, st, r, _, _, _ = env.step(kk, st, a, params)
            return (obs, st, total + r), None

        (_, _, total), _ = jax.lax.scan(body, (obs, st, 0.0), jax.random.split(k_s, params.max_steps_in_episode))
        return total

    return jax.vmap(one)(keys)


def pendulum_return(policy_fn, key, n_episodes: int = 32) -> float:
    """Mean return over n_episodes."""
    return float(jnp.mean(pendulum_returns(policy_fn, jax.random.split(key, n_episodes))))


def pendulum_es(cfg: dict, episodes: int = 4):
    """The Part 3 ES loop on Pendulum-v1, with the Part 3 brain (3 -> 32 -> 32 -> 1)."""
    from evo2.agents.brain import apply_mlp, init_mlp
    from evo2.methods.es import ES

    def score_fn(sol, key):
        ret = jnp.mean(pendulum_returns(lambda o: apply_mlp(sol["brain"], o), jax.random.split(key, episodes)))
        return ret, jnp.int32(200 * episodes)

    template = {"brain": init_mlp(jax.random.PRNGKey(0), 3, 1)}
    return ES(cfg, template, score_fn, max_agent_steps_per_candidate=200 * episodes)
