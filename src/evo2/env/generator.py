"""Drawing environments from a range of parameters (SPEC_part3.md, Section 3.4).

One sampler serves every use: training draws (PPO-DR, ES-DR, OE), and the
fixed validation and test environments, which are drawn from their seeds with
the "env_params" stream. Integer ranges are inclusive at both ends.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from evo2.env.world import LAYOUT_CODES, EnvParams, make_env_params
from evo2.seeds import check_seed, stream_key

_FLOAT_KEYS = ("L", "sigma", "patch_r", "rho", "drift")
_INT_KEYS = ("F", "k", "season_P")
RANGE_KEYS = _FLOAT_KEYS + _INT_KEYS + ("layouts",)


@struct.dataclass
class EnvRanges:
    L: jax.Array          # (2,) float
    F: jax.Array          # (2,) int
    layout_p: jax.Array   # (3,) probability of uniform, clusters, patches
    k: jax.Array          # (2,) int
    sigma: jax.Array
    patch_r: jax.Array
    rho: jax.Array
    drift: jax.Array
    season_P: jax.Array   # (2,) int; [0, 0] means no seasonal moves


def ranges_from_dict(d: dict) -> EnvRanges:
    """Build EnvRanges from a config dict. Every key must be present."""
    missing = set(RANGE_KEYS) - set(d)
    unknown = set(d) - set(RANGE_KEYS)
    if missing or unknown:
        raise ValueError(f"range keys: missing {sorted(missing)}, unknown {sorted(unknown)}")
    for name in _FLOAT_KEYS + _INT_KEYS:
        lo, hi = d[name]
        if lo > hi:
            raise ValueError(f"range {name}: low {lo} > high {hi}")
    layouts = d["layouts"]
    if not layouts or any(x not in LAYOUT_CODES for x in layouts):
        raise ValueError(f"bad layouts {layouts}")
    p = jnp.zeros(3).at[jnp.array([LAYOUT_CODES[x] for x in layouts])].set(1.0)
    f = {k: jnp.asarray(d[k], jnp.float32) for k in _FLOAT_KEYS}
    i = {k: jnp.asarray(d[k], jnp.int32) for k in _INT_KEYS}
    return EnvRanges(layout_p=p / p.sum(), **f, **i)


def sample_env_params(key, r: EnvRanges) -> EnvParams:
    """Draw one environment. Floats uniform; ints uniform over the inclusive range."""
    ks = jax.random.split(key, 9)

    def u(k, lohi):
        return jax.random.uniform(k, (), minval=lohi[0], maxval=lohi[1])

    def ri(k, lohi):
        return jax.random.randint(k, (), lohi[0], lohi[1] + 1).astype(jnp.int32)

    return EnvParams(
        L=u(ks[0], r.L),
        n_food=ri(ks[1], r.F),
        layout=jax.random.choice(ks[2], 3, p=r.layout_p).astype(jnp.int32),
        k=ri(ks[3], r.k),
        sigma=u(ks[4], r.sigma),
        patch_r=u(ks[5], r.patch_r),
        rho=u(ks[6], r.rho),
        drift=u(ks[7], r.drift),
        season_P=ri(ks[8], r.season_P),
    )


def env_from_seed(purpose: str, seed: int, r: EnvRanges) -> EnvParams:
    """A fixed environment for a validation or test seed."""
    key = stream_key(jax.random.PRNGKey(check_seed(purpose, seed)), "env_params")
    return sample_env_params(key, r)


def params_from_dict(d: dict) -> EnvParams:
    """A single fixed environment from a config dict (e.g. the canonical one)."""
    allowed = {"L", "F", "layout", "k", "sigma", "patch_r", "rho", "drift", "season_P"}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"unknown environment keys {sorted(unknown)}")
    return make_env_params(L=d["L"], n_food=d["F"], layout=d["layout"], k=d.get("k", 2),
                           sigma=d.get("sigma", 5.0), patch_r=d.get("patch_r", 5.0),
                           rho=d["rho"], drift=d.get("drift", 0.0),
                           season_P=d.get("season_P", 0))


def stack_params(ps: list[EnvParams]) -> EnvParams:
    """Stack environments along a new first axis (for jax.vmap)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ps)


def params_to_dict(p: EnvParams) -> dict:
    """Plain Python values, for writing to results files."""
    out = {}
    for f in dataclasses.fields(p):
        v = jnp.asarray(getattr(p, f.name))
        out[f.name] = float(v) if jnp.issubdtype(v.dtype, jnp.floating) else int(v)
    return out
