"""Genomes and mutation for the evolving methods (SPEC_part3.md, Section 4.2).

A genome is a brain (dict of weight arrays) plus the 3 body traits. In a
population every leaf has a leading axis for the agent slot.

Mutation at birth: every weight gets Gaussian noise with sd `weight_sd`; every
trait gets Gaussian noise with sd `trait_frac` x (its range), then is clipped
to its bounds. Weights have no bounds. With evolve_body=False the traits are
copied unchanged (option (b) in Section 5).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from evo2.agents.brain import init_mlp


@struct.dataclass
class Genome:
    brain: dict
    traits: jax.Array  # (..., 3)


@struct.dataclass
class MutationConfig:
    weight_sd: float = struct.field(pytree_node=False, default=0.02)
    trait_frac: float = struct.field(pytree_node=False, default=0.05)
    evolve_body: bool = struct.field(pytree_node=False, default=True)


def init_population(key, n: int, obs_dim: int, default_traits: jax.Array) -> Genome:
    """n genomes: brains from the shared initializer, default bodies."""
    brains = jax.vmap(lambda k: init_mlp(k, obs_dim))(jax.random.split(key, n))
    traits = jnp.tile(default_traits[None, :], (n, 1))
    return Genome(brain=brains, traits=traits)


def mutate(key, g: Genome, mcfg: MutationConfig, trait_lo, trait_hi) -> Genome:
    """Mutate one genome (no leading axis)."""
    leaves, treedef = jax.tree_util.tree_flatten(g.brain)
    keys = jax.random.split(key, len(leaves) + 1)
    new_leaves = [x + mcfg.weight_sd * jax.random.normal(k, x.shape, x.dtype)
                  for x, k in zip(leaves, keys[:-1])]
    brain = jax.tree_util.tree_unflatten(treedef, new_leaves)
    if mcfg.evolve_body:
        sd = mcfg.trait_frac * (trait_hi - trait_lo)
        traits = jnp.clip(g.traits + sd * jax.random.normal(keys[-1], g.traits.shape),
                          trait_lo, trait_hi)
    else:
        traits = g.traits
    return Genome(brain=brain, traits=traits)


def apply_births(key, pop: Genome, parent_idx: jax.Array, mcfg: MutationConfig,
                 trait_lo, trait_hi, mutate_on: bool = True) -> Genome:
    """Write children into their slots. parent_idx[slot] >= 0 marks a child of
    that parent; other slots are left alone. With mutate_on=False children are
    exact copies (used by the persistence test)."""
    n = parent_idx.shape[0]
    p = jnp.maximum(parent_idx, 0)
    parents = jax.tree_util.tree_map(lambda x: x[p], pop)
    if mutate_on:
        children = jax.vmap(lambda k, g: mutate(k, g, mcfg, trait_lo, trait_hi))(
            jax.random.split(key, n), parents)
    else:
        children = parents
    is_child = parent_idx >= 0

    def pick(new, old):
        mask = is_child.reshape((n,) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(pick, children, pop)
