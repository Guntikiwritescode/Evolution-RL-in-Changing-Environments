"""One compiled loop for any chunk length.

`lax.scan` needs its length when the program is compiled, so every new chunk
length compiled a new program. The budget logic (evo2.run.chunk_size) asks
for the full chunk and then powers of two down to 1 near each target, so one
short OE run compiled its chunk about a dozen times (2 Nov 2026, measured on
CPU: about 6 s each; far slower on the P0 GPUs). `run_dynamic` takes the
length as data instead: one compiled program serves every length up to
`max_n`.

It runs the same step function the same number of times as
`lax.scan(step, state, None, n)` (scan itself compiles to a while loop), and
keeps the per-step logs in a buffer of `max_n` rows; rows from n on are zeros,
so callers read logs[:n].
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def run_dynamic(step: Callable, state, n, max_n: int):
    """Apply step(state, None) -> (state, log) n times (0 <= n <= max_n; n may
    be a traced integer). Returns (state, logs) with logs stacked along a
    leading axis of length max_n; only the first n rows are meaningful."""
    log_shape = jax.eval_shape(lambda s: step(s, None)[1], state)
    buf = jax.tree_util.tree_map(lambda s: jnp.zeros((max_n,) + s.shape, s.dtype), log_shape)

    def body(carry):
        i, s, b = carry
        s, log = step(s, None)
        b = jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), b, log)
        return i + 1, s, b

    n = jnp.minimum(jnp.asarray(n, jnp.int32), max_n)
    _, state, buf = jax.lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(0), state, buf))
    return state, buf
