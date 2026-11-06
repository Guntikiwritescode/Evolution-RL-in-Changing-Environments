"""PPO with parameter sharing (SPEC_part3.md, Section 5: PPO-fixed and PPO-DR).

Adapted from PureJaxRL's ppo_continuous_action.py (Apache-2.0, Chris Lu,
github.com/luchris429/purejaxrl, commit 31756b1). Changes from the original:
  * Many agents per environment share one network. Every agent's experience
    is its own trajectory; all of them pool into one batch (parameter sharing).
  * A `valid` mask: steps where an agent was dead at the start of the step
    are left out of every loss term and of advantage normalization.
  * The network is the shared Part 3 brain (evo2.agents.brain), with separate
    actor and critic MLPs and a state-independent log std, as in PureJaxRL.
  * Training runs in chunks of updates (resumable, with progress records)
    instead of one scan over the whole run.
  * Optional running reward normalization follows PureJaxRL's
    NormalizeVecReward, with its statistics taken over valid (living) agents
    only; observation normalization is not used (the world's inputs are
    already scaled to [0, 2]).
  * Compute is counted as valid agent-steps (agents alive at the start of the
    step), the same count every Part 3 method uses.

Environments are reached through an adapter with
    reset(key) -> (env_state, obs)                       obs: (B, A, obs_dim)
    step(key, env_state, action) -> (env_state, obs, reward, done, valid, info)
with reward, done, valid of shape (B, A) and info holding "ep_end" (B,) and
"ep_score" (B,) for logging.
"""
from __future__ import annotations

from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import struct

from evo2.agents.brain import apply_mlp, init_mlp

DEFAULTS = {  # PureJaxRL's continuous-action defaults, except where noted
    "LR": 3e-4,
    "NUM_STEPS": 64,          # PureJaxRL: 10 (with 2048 envs); set by tuning
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": False,
    "NORMALIZE_REWARD": True,
    "LOG_STD_INIT": 0.0,
}


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array


@struct.dataclass
class RewardNorm:
    mean: jax.Array
    var: jax.Array
    count: jax.Array
    return_val: jax.Array


@struct.dataclass
class Runner:
    params: dict
    opt_state: optax.OptState
    env_state: object
    obs: jax.Array
    rew_norm: RewardNorm
    key: jax.Array
    update: jax.Array          # updates done so far


def init_params(key, obs_dim: int, act_dim: int, log_std_init: float = 0.0) -> dict:
    k_a, k_c = jax.random.split(key)
    return {
        "actor": init_mlp(k_a, obs_dim, act_dim, out_gain=0.01),
        "log_std": jnp.full((act_dim,), log_std_init, jnp.float32),
        "critic": init_mlp(k_c, obs_dim, 1, out_gain=1.0),
    }


def policy_mean(params: dict, obs: jax.Array) -> jax.Array:
    """The deterministic policy used at evaluation: the actor's mean."""
    return apply_mlp(params["actor"], obs)


def _dist(params, obs):
    return distrax.MultivariateNormalDiag(apply_mlp(params["actor"], obs), jnp.exp(params["log_std"]))


def _value(params, obs):
    return apply_mlp(params["critic"], obs)[..., 0]


def masked_mean(x, w):
    return jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0)


def ppo_loss(params, batch: Transition, gae, targets, cfg):
    """PureJaxRL's clipped PPO loss, with every mean taken over valid steps only."""
    w = batch.valid.astype(jnp.float32)
    pi = _dist(params, batch.obs)
    value = _value(params, batch.obs)
    log_prob = pi.log_prob(batch.action)

    value_pred_clipped = batch.value + (value - batch.value).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * masked_mean(jnp.maximum(value_losses, value_losses_clipped), w)

    ratio = jnp.exp(log_prob - batch.log_prob)
    g_mean = masked_mean(gae, w)
    g_std = jnp.sqrt(masked_mean(jnp.square(gae - g_mean), w))
    gae = (gae - g_mean) / (g_std + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * gae
    loss_actor = masked_mean(-jnp.minimum(loss_actor1, loss_actor2), w)
    entropy = masked_mean(pi.entropy(), w)

    total = loss_actor + cfg["VF_COEF"] * value_loss - cfg["ENT_COEF"] * entropy
    return total, (value_loss, loss_actor, entropy)


def compute_gae(traj: Transition, last_val, gamma, lam):
    """PureJaxRL's GAE. A step with done=1 does not look past itself."""
    def body(carry, tr):
        gae, next_value = carry
        delta = tr.reward + gamma * next_value * (1 - tr.done) - tr.value
        gae = delta + gamma * lam * (1 - tr.done) * gae
        return (gae, tr.value), gae

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_val), last_val), traj, reverse=True)
    return adv, adv + traj.value


class PPO:
    def __init__(self, cfg: dict, adapter, obs_dim: int, act_dim: int, n_envs: int, n_agents: int):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown PPO settings {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        self.adapter = adapter
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.n_envs, self.n_agents = n_envs, n_agents
        c = self.cfg
        self.batch_size = c["NUM_STEPS"] * n_envs * n_agents
        if self.batch_size % c["NUM_MINIBATCHES"] != 0:
            raise ValueError("NUM_STEPS x envs x agents must divide into NUM_MINIBATCHES")
        self.max_agent_steps_per_update = self.batch_size   # if every agent were alive throughout
        self.total_updates = None  # set by set_total_updates when ANNEAL_LR

    def set_total_updates(self, n: int):
        self.total_updates = int(n)

    def _tx(self):
        c = self.cfg
        if c["ANNEAL_LR"]:
            if not self.total_updates:
                raise ValueError("ANNEAL_LR needs set_total_updates() first")
            per = c["NUM_MINIBATCHES"] * c["UPDATE_EPOCHS"]
            total = self.total_updates

            def lr(count):
                return c["LR"] * (1.0 - (count // per) / total)
        else:
            lr = c["LR"]
        return optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]), optax.adam(lr, eps=1e-5))

    def init(self, key, params: dict | None = None, rew_norm: dict | None = None, env=None) -> Runner:
        """Fresh runner. Pass params (and the reward-normalization statistics
        mean, var, count) to continue from a trained policy with a fresh
        optimizer (used by the adaptation phase). `env` is the fixed
        environment when the adapter has none of its own (see EpisodicWorlds)."""
        k_p, k_env, k_run = jax.random.split(key, 3)
        if params is None:
            params = init_params(k_p, self.obs_dim, self.act_dim, self.cfg["LOG_STD_INIT"])
        tx = self._tx()
        env_state, obs = self.adapter.reset(k_env) if env is None else self.adapter.reset(k_env, env)
        rs = rew_norm or {"mean": 0.0, "var": 1.0, "count": 1e-4}
        rn = RewardNorm(mean=jnp.float32(rs["mean"]), var=jnp.float32(rs["var"]),
                        count=jnp.float32(rs["count"]),
                        return_val=jnp.zeros((self.n_envs, self.n_agents), jnp.float32))
        return Runner(params=params, opt_state=tx.init(params), env_state=env_state, obs=obs,
                      rew_norm=rn, key=k_run, update=jnp.int32(0))

    def _normalize_reward(self, rn: RewardNorm, reward, done, valid):
        gamma = self.cfg["GAMMA"]
        ret = rn.return_val * gamma * (1 - done) + reward
        w = valid.astype(jnp.float32)
        b_count = jnp.sum(w)
        b_mean = jnp.sum(ret * w) / jnp.maximum(b_count, 1.0)
        b_var = jnp.sum(jnp.square(ret - b_mean) * w) / jnp.maximum(b_count, 1.0)
        delta = b_mean - rn.mean
        tot = rn.count + b_count
        new_mean = rn.mean + delta * b_count / tot
        m2 = rn.var * rn.count + b_var * b_count + jnp.square(delta) * rn.count * b_count / tot
        rn = RewardNorm(mean=new_mean, var=m2 / tot, count=tot, return_val=ret)
        return rn, reward / jnp.sqrt(rn.var + 1e-8)

    def _update(self, r: Runner, _):
        c = self.cfg
        tx = self._tx()

        def env_step(carry, _):
            env_state, obs, rn, key = carry
            key, k_act, k_step = jax.random.split(key, 3)
            pi = _dist(r.params, obs)
            action = pi.sample(seed=k_act)
            log_prob = pi.log_prob(action)
            value = _value(r.params, obs)
            env_state, obs2, reward, done, valid, info = self.adapter.step(k_step, env_state, action)
            done = done.astype(jnp.float32)
            if c["NORMALIZE_REWARD"]:
                rn, reward_used = self._normalize_reward(rn, reward, done, valid)
            else:
                reward_used = reward
            tr = Transition(obs, action, log_prob, value, reward_used, done, valid)
            log = {"raw_reward": jnp.sum(reward * valid), "valid": jnp.sum(valid),
                   "ep_end": jnp.sum(info["ep_end"]),
                   "ep_score_sum": jnp.sum(jnp.where(info["ep_end"], info["ep_score"], 0.0))}
            return (env_state, obs2, rn, key), (tr, log)

        (env_state, obs, rn, key), (traj, logs) = jax.lax.scan(
            env_step, (r.env_state, r.obs, r.rew_norm, r.key), None, c["NUM_STEPS"])
        last_val = _value(r.params, obs)
        adv, targets = compute_gae(traj, last_val, c["GAMMA"], c["GAE_LAMBDA"])

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            flat = jax.tree_util.tree_map(
                lambda x: x.reshape((self.batch_size,) + x.shape[3:]), (traj, adv, targets))
            perm = jax.random.permutation(k_perm, self.batch_size)
            mbs = jax.tree_util.tree_map(
                lambda x: jnp.take(x, perm, axis=0).reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                flat)

            def minibatch(carry, mb):
                params, opt_state = carry
                b, g, t = mb
                (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, b, g, t, c)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), (loss, *aux)

            (params, opt_state), losses = jax.lax.scan(minibatch, (params, opt_state), mbs)
            return (params, opt_state, key), losses

        (params, opt_state, key), losses = jax.lax.scan(
            epoch, (r.params, r.opt_state, key), None, c["UPDATE_EPOCHS"])
        new = Runner(params=params, opt_state=opt_state, env_state=env_state, obs=obs, rew_norm=rn,
                     key=key, update=r.update + 1)
        metrics = {
            "agent_steps": jnp.sum(logs["valid"]).astype(jnp.int32),
            "food_per_valid_step": jnp.sum(logs["raw_reward"]) / jnp.maximum(jnp.sum(logs["valid"]), 1.0),
            "episodes_ended": jnp.sum(logs["ep_end"]),
            # NaN when no episode ended during this update
            "mean_ep_score": jnp.where(jnp.sum(logs["ep_end"]) > 0,
                                       jnp.sum(logs["ep_score_sum"]) / jnp.maximum(jnp.sum(logs["ep_end"]), 1.0),
                                       jnp.nan),
            "loss": jnp.mean(losses[0]), "value_loss": jnp.mean(losses[1]),
            "actor_loss": jnp.mean(losses[2]), "entropy": jnp.mean(losses[3]),
        }
        return new, metrics

    def run_chunk(self, r: Runner, n_updates: int):
        """n_updates updates in one compiled call. Returns (runner, metrics per update)."""
        return jax.lax.scan(self._update, r, None, n_updates)

