"""One interface over the five core methods, for the Part 3 scripts.

A Trainer has:
  init(key, policy=None, env=None) -> state
                                       start fresh, or continue from a policy; an
                                       adaptation trainer (adapt=True) needs the
                                       test environment as env
  chunk(state, n=None) -> (state, summary)
                                       one chunk of n units (OE steps, PPO updates
                                       or ES generations; 1 <= n <= chunk_units,
                                       default chunk_units); summary has exact
                                       Python counts, including "agent_steps"
                                       (agents alive at the start of each step,
                                       for every method)
  chunk_units, unit_max_agent_steps    the default chunk length, and the most
                                       agent-steps one unit can take
  policy(state) -> dict                what evaluation and adaptation need

Compiling (2 Nov 2026, after P0): every chunk length runs in one compiled
program (evo2.methods.loop.run_dynamic), and an adaptation trainer keeps the
test environment in its state, not in the compiled program, so one trainer
(see get_trainer) serves every run and every test environment.

Addendum A (PREREG_part3.md Section 16, 4 Nov 2026): one more method, ES-fixed, ES-DR's
algorithm trained in the canonical environment only. It is listed in ADDENDUM_METHODS, not in
METHODS, so the pre-registered core analysis (scripts/part3/analyze.py, which reads METHODS) is
unchanged. Its tuned settings come from a separate file (load_methods_config(extra_tuned=...)).

Policies are plain dicts of arrays:
  {"kind": "shared", "brain": ..., "traits": (3,), "train_params": ...}   (PPO, ES)
  {"kind": "population", "genomes": Genome, "alive": (N,)}              (OE)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from evo2.env.generator import sample_env_params
from evo2.env.splits import CONFIG_DIR, load_train, load_world_config
from evo2.env.world import World
from evo2.methods.adapters import EpisodicWorlds, fixed_env_fn
from evo2.methods.es import world_es
from evo2.methods.loop import run_dynamic
from evo2.methods.oe import OE
from evo2.methods.ppo import PPO

METHODS = ("oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr")   # the pre-registered core
ADDENDUM_METHODS = ("es_fixed",)                                # PREREG Section 16 (Addendum A)
ALL_METHODS = METHODS + ADDENDUM_METHODS


TUNED_FILE = CONFIG_DIR / "tuned.yaml"


def _merge_tuned(mcfg: dict, t: dict, allowed: tuple, what: str) -> None:
    for method, settings in t["settings"].items():
        if method not in allowed:
            raise ValueError(f"{what} names an unknown method {method!r}")
        block = mcfg.setdefault(method, {})
        block["settings"] = {**block.get("settings", {}), **settings}


def load_methods_config(path: Path | None = None, tuned: Path | None = TUNED_FILE,
                        extra_tuned: Path | None = None) -> dict:
    """methods.yaml, plus (when it exists) the settings chosen by tuning (configs/part3/tuned.yaml,
    written by scripts/part3/tune.py): each method's tuned settings go under
    mcfg[method]["settings"], which overrides the shared block (see method_settings). Every run
    stores the merged config in its run.json.

    extra_tuned: the addendum's tuned settings (scripts/part3/tune.py --write-addendum), for
    ADDENDUM_METHODS only; recorded as mcfg["tuned_from_addendum"]. It must exist when given."""
    with open(path or CONFIG_DIR / "methods.yaml") as fh:
        mcfg = yaml.safe_load(fh)
    if tuned is not None and Path(tuned).exists():
        with open(tuned) as fh:
            t = yaml.safe_load(fh)
        _merge_tuned(mcfg, t, METHODS, "tuned.yaml")
        mcfg["tuned_from"] = t["source"]
    if extra_tuned is not None:
        with open(extra_tuned) as fh:
            t = yaml.safe_load(fh)
        _merge_tuned(mcfg, t, ADDENDUM_METHODS, "the addendum's tuned settings")
        mcfg["tuned_from_addendum"] = t["source"]
    return mcfg


def method_settings(mcfg: dict, method: str, shared: str) -> dict:
    """A method's settings: the shared block (mcfg["ppo"] or mcfg["es"]) overridden by the
    method's own block, mcfg[method]["settings"], when there is one (tuned separately)."""
    return {**mcfg[shared]["settings"], **mcfg.get(method, {}).get("settings", {})}


def env_fn_for(method: str, adapt: bool = False):
    """Where a method trains: the train distribution or the canonical
    environment. None for adaptation: the test environment is given to init
    and kept in the state."""
    ranges, canon = load_train()
    if adapt:
        return None
    if method in ("oe", "ppo_dr", "es_dr"):
        return lambda k: sample_env_params(k, ranges)
    if method in ("oe_fixed", "ppo_fixed", "es_fixed"):
        return fixed_env_fn(canon)
    raise ValueError(f"unknown method {method!r}")


def _check_env(adapt: bool, env) -> None:
    if adapt and env is None:
        raise ValueError("an adaptation trainer needs the test environment: init(key, policy, env)")
    if not adapt and env is not None:
        raise ValueError("env is only for adaptation trainers (adapt=True)")


def _head(logs: dict, n: int) -> dict:
    """The first n rows of run_dynamic's per-unit logs, as NumPy arrays."""
    return {k: np.asarray(jax.device_get(v))[:n] for k, v in logs.items()}


class _Trainer:
    chunk_units: int

    def _units(self, n: int | None) -> int:
        n = self.chunk_units if n is None else int(n)
        if not 1 <= n <= self.chunk_units:
            raise ValueError(f"a chunk is 1..{self.chunk_units} units, not {n}")
        return n

    def _compile(self, step):
        # n is data (an int32), so every chunk length shares one compiled program
        return jax.jit(lambda s, n: run_dynamic(step, s, n, self.chunk_units))


class OETrainer(_Trainer):
    def __init__(self, method: str, mcfg: dict, adapt: bool = False):
        self.world = World(load_world_config("ecological"))
        body = mcfg["body_option"] == "a"
        settings = dict(mcfg[method]["settings"])
        if adapt:
            settings["CHANGE_EVERY"] = 0   # adaptation: the test environment stays
        self.oe = OE(settings, self.world, env_fn_for(method, adapt), evolve_body=body)
        self.adapt = adapt
        self.n_steps = mcfg[method]["chunk_steps"]
        self.chunk_units = self.n_steps
        self.unit_max_agent_steps = self.world.cfg.n_slots   # one OE step, every slot alive
        self._run = self._compile(self.oe._step)

    def init(self, key, policy=None, env=None):
        _check_env(self.adapt, env)
        if policy is None:
            return self.oe.init(key, env=env)
        k_init, k_move = jax.random.split(key)
        s = self.oe.init(k_init, pop=policy["genomes"], n_alive=0, env=env)
        ws = s.world.replace(alive=jnp.asarray(policy["alive"]),
                             energy=jnp.where(policy["alive"], self.world.cfg.birth_energy, 0.0),
                             traits=policy["genomes"].traits)
        s = s.replace(world=ws)
        return self.oe.move_to(k_move, s, env) if self.adapt else s

    def chunk(self, state, n: int | None = None):
        n = self._units(n)
        state, log = self._run(state, np.int32(n))
        return state, OE.chunk_summary(_head(log, n))

    def policy(self, state):
        return {"kind": "population", "genomes": state.pop, "alive": state.world.alive}


class PPOTrainer(_Trainer):
    def __init__(self, method: str, mcfg: dict, adapt: bool = False):
        self.world = World(load_world_config("episodic"))
        p = mcfg["ppo"]
        body = p["body"] if mcfg["body_option"] == "a" else list(self.world.cfg.default_traits)
        self.traits = jnp.clip(jnp.asarray(body, jnp.float32), self.world.trait_lo, self.world.trait_hi)
        ad = EpisodicWorlds(self.world, p["n_envs"], env_fn_for(method, adapt), self.traits)
        self.ppo = PPO(method_settings(mcfg, method, "ppo"), ad, self.world.cfg.obs_dim, 2, p["n_envs"],
                       self.world.cfg.n_slots)
        self.adapt = adapt
        self.n_updates = p["chunk_updates"]
        self.chunk_units = self.n_updates
        self.unit_max_agent_steps = self.ppo.max_agent_steps_per_update
        self.rew_norm_keys = ("mean", "var", "count")
        self._run = self._compile(self.ppo._update)

    def init(self, key, policy=None, env=None):
        _check_env(self.adapt, env)
        if policy is None:
            return self.ppo.init(key, env=env)
        return self.ppo.init(key, policy["train_params"], policy.get("rew_norm"), env=env)

    def chunk(self, state, n: int | None = None):
        n = self._units(n)
        state, m = self._run(state, np.int32(n))
        g = _head(m, n)
        return state, {
            "agent_steps": int(g["agent_steps"].astype(np.int64).sum()),
            "updates": n,
            "food_per_valid_step": float(g["food_per_valid_step"].mean()),
            "mean_ep_score": float(np.nanmean(g["mean_ep_score"])) if np.isfinite(g["mean_ep_score"]).any() else None,
            "entropy": float(g["entropy"][-1]),
            "loss": float(g["loss"][-1]),
        }

    def policy(self, state):
        rn = state.rew_norm
        return {"kind": "shared", "brain": state.params["actor"], "traits": self.traits,
                "train_params": state.params,
                "rew_norm": {"mean": rn.mean, "var": rn.var, "count": rn.count}}


class ESTrainer(_Trainer):
    def __init__(self, method: str, mcfg: dict, adapt: bool = False):
        self.world = World(load_world_config("episodic"))
        e = mcfg["es"]
        self.es, self.pol = world_es(method_settings(mcfg, method, "es"), self.world, env_fn_for(method, adapt),
                                     evolve_body=mcfg["body_option"] == "a")
        self.adapt = adapt
        self.n_gens = e["chunk_generations"]
        self.chunk_units = self.n_gens
        self.unit_max_agent_steps = self.es.max_agent_steps_per_generation
        self._run = self._compile(self.es._generation)

    def init(self, key, policy=None, env=None):
        _check_env(self.adapt, env)
        k_sol, k_init = jax.random.split(key)
        sol = self.pol.initial_solution(k_sol) if policy is None else policy["train_params"]
        return self.es.init(k_init, sol, ctx=env)

    def chunk(self, state, n: int | None = None):
        n = self._units(n)
        state, m = self._run(state, np.int32(n))
        g = _head(m, n)
        return state, {
            "agent_steps": int(g["agent_steps"].astype(np.int64).sum()),
            "generations": n,
            "mean_score": float(g["mean_score"].mean()),
            "max_score": float(g["max_score"].max()),
        }

    def policy(self, state):
        sol = self.es.mean_solution(state)
        return {"kind": "shared", "brain": sol["brain"], "traits": self.pol.traits_of(sol),
                "train_params": sol}


def make_trainer(method: str, mcfg: dict, adapt: bool = False):
    """A new trainer. adapt=True: the adaptation phase (no environment
    changes; the test environment is given to init)."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}; one of {ALL_METHODS}")
    if mcfg.get("body_option") not in ("a", "b"):
        raise ValueError(f"body_option must be 'a' or 'b', not {mcfg.get('body_option')!r}")
    if method in ("oe", "oe_fixed"):
        return OETrainer(method, mcfg, adapt)
    if method in ("ppo_fixed", "ppo_dr"):
        return PPOTrainer(method, mcfg, adapt)
    if method in ("es_dr", "es_fixed"):
        return ESTrainer(method, mcfg, adapt)
    raise ValueError(f"no trainer for {method!r}")


@lru_cache(maxsize=None)
def _cached_trainer(method: str, mcfg_json: str, adapt: bool):
    return make_trainer(method, json.loads(mcfg_json), adapt)


def get_trainer(method: str, mcfg: dict, adapt: bool = False):
    """The same as make_trainer, but one trainer (and so one compiled chunk
    program) per method, config and phase in this process. Trainers keep no
    run state between calls, so runs can share one."""
    return _cached_trainer(method, json.dumps(mcfg, sort_keys=True), bool(adapt))


def chunk_size(trainer, steps: int, target: int) -> int:
    """How many units (OE steps, PPO updates, ES generations) the next chunk
    runs: the full chunk if even a chunk of fully alive units cannot pass the
    target; otherwise the largest power of two of units that cannot pass it
    (at least 1). It depends only on (steps, target), so a resumed run takes
    the same chunks as an uninterrupted one, and no run passes its target by
    a full unit's worth of agent-steps."""
    room = (target - steps) // trainer.unit_max_agent_steps
    if room >= trainer.chunk_units:
        return trainer.chunk_units
    n = 1
    while n * 2 <= room:
        n *= 2
    return n


def advance(trainer, state, steps: int, target: int, on_chunk=None, max_chunks: int | None = None):
    """Run chunks until `steps` reaches `target` agent-steps (see chunk_size).
    on_chunk(state, steps, summary) is called after every chunk; if it returns
    True the run stops there. max_chunks also stops early (used by tests of
    resuming). Returns (state, steps)."""
    done = 0
    while steps < target and (max_chunks is None or done < max_chunks):
        done += 1
        state, summary = trainer.chunk(state, chunk_size(trainer, steps, target))
        steps += int(summary["agent_steps"])
        if on_chunk is not None and on_chunk(state, steps, summary):
            break
    return state, steps


def to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


__all__ = ["ADDENDUM_METHODS", "ALL_METHODS", "METHODS", "advance", "chunk_size", "get_trainer", "load_methods_config", "make_trainer",
           "method_settings", "to_numpy"]
