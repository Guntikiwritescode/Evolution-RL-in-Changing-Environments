"""The brain every Part 3 method uses (SPEC_part3.md, Section 4.1).

A multilayer perceptron, obs_dim -> 32 (tanh) -> 32 (tanh) -> 2, written as
plain functions over a dict of arrays so that the same weights can be
(a) trained by PPO with gradients, (b) flattened into one vector for ES, and
(c) held per agent with a leading axis for OE.

Initial weights follow PureJaxRL's ActorCritic (Apache-2.0,
github.com/luchris429/purejaxrl, commit 31756b1): orthogonal initialization
with gain sqrt(2) for hidden layers and 0.01 for the action layer (1.0 for a
value head), and zero biases.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

HIDDEN = 32
ACT_DIM = 2
LAYERS = ("1", "2", "3")


def init_mlp(key, obs_dim: int, out_dim: int = ACT_DIM, hidden: int = HIDDEN,
             out_gain: float = 0.01) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    orth = jax.nn.initializers.orthogonal
    return {
        "W1": orth(math.sqrt(2.0))(k1, (obs_dim, hidden), jnp.float32),
        "b1": jnp.zeros((hidden,), jnp.float32),
        "W2": orth(math.sqrt(2.0))(k2, (hidden, hidden), jnp.float32),
        "b2": jnp.zeros((hidden,), jnp.float32),
        "W3": orth(out_gain)(k3, (hidden, out_dim), jnp.float32),
        "b3": jnp.zeros((out_dim,), jnp.float32),
    }


def apply_mlp(params: dict, obs: jax.Array) -> jax.Array:
    """obs (..., obs_dim) with one set of weights -> (..., out_dim)."""
    h = jnp.tanh(obs @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]


def apply_per_agent(params: dict, obs: jax.Array) -> jax.Array:
    """Each agent with its own weights: params leaves have a leading axis N,
    obs is (N, obs_dim). Returns (N, out_dim)."""
    return jax.vmap(apply_mlp)(params, obs)


def n_weights(params: dict) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
