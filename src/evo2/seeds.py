"""Seed ranges and named random streams for Part 3.

Every seed belongs to exactly one purpose (SPEC_part3.md, Section 8). Every
random draw comes from a named stream: the key for a stream is the run's base
key folded with that stream's fixed number, so two purposes never share a key.
"""
from __future__ import annotations

import jax

# Section 8 of docs/SPEC_part3.md. Inclusive ranges.
SEED_RANGES: dict[str, tuple[int, int]] = {
    "pilot": (3000, 3099),
    "pilot_eval": (3100, 3199),
    "test": (3900, 3999),
    "tuning": (4000, 4799),
    "stats": (4800, 4899),   # bootstrap (4800) and the diversity probe set (4801)
    "validation": (4900, 4999),
    "final": (5000, 5499),
    "test_env": (5900, 5999),
}

# Fixed stream numbers. Never renumber an existing stream: that would change
# every result drawn from it. Numbers 2-8 and 10 belonged to streams in the
# first draft that no code used; they stay retired.
STREAMS: dict[str, int] = {
    "env_params": 1,      # drawing a fixed environment's parameters from its seed
    "eval_episodes": 9,   # evaluation episodes (the same for every method and run)
    "tuning_draw": 11,    # random-search draws of tuning settings
    "train_run": 12,      # everything inside one training run (each method splits it further)
    "adapt_run": 13,      # everything inside one adaptation run (folded with the environment index)
    "probe": 14,          # the fixed probe observations for behavioral diversity
}


def check_seed(purpose: str, seed: int) -> int:
    """Return seed if it lies in the range for this purpose, else raise."""
    if purpose not in SEED_RANGES:
        raise ValueError(f"unknown seed purpose {purpose!r}")
    lo, hi = SEED_RANGES[purpose]
    if not (lo <= int(seed) <= hi):
        raise ValueError(f"seed {seed} is outside the {purpose} range {lo}-{hi}")
    return int(seed)


def base_key(purpose: str, seed: int) -> jax.Array:
    """The base PRNG key for a seed, after checking its purpose."""
    return jax.random.PRNGKey(check_seed(purpose, seed))


def stream_key(key: jax.Array, stream: str) -> jax.Array:
    """The key for one named stream of a base key."""
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}")
    return jax.random.fold_in(key, STREAMS[stream])
