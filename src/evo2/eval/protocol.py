"""The evaluation protocols (SPEC_part3.md, Section 7): zero-shot and adaptation.

Every method is measured the same way: competence = mean food per agent in
episodic episodes (16 agents, 500 steps). Shared policies (PPO, ES) field 16
copies; an OE population fields 16 agents sampled from its living members,
with a new sample for every episode.
"""
from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp

from evo2.agents.genome import Genome
from evo2.env.splits import load_world_config
from evo2.env.world import EnvParams, World
from evo2.eval.episodes import shared_act_fn
from evo2.eval.evaluate import (broadcast_genome, competence_shared, genome_competence, persistence,
                                random_competence_jit)
from evo2.run import advance, get_trainer
from evo2.runio import Timer


def _sample_from_population(key, policy: dict, n: int) -> tuple[Genome, jax.Array]:
    """n agents from the living members of a saved OE population."""
    return _sample(key, policy["genomes"], policy["alive"], n)


@lru_cache(maxsize=None)
def get_world(mode: str) -> World:
    """One World object per mode, so compiled evaluation functions are reused."""
    return World(load_world_config(mode))


@partial(jax.jit, static_argnums=(0, 5))
def _population_competence(world: World, ep_keys, sample_keys, env: EnvParams, policy_arrays, n: int):
    genomes, alive = policy_arrays

    def one(k_ep, k_s):
        g, _ = _sample(k_s, genomes, alive, n)
        return genome_competence(world, k_ep[None], env, g)

    return jnp.mean(jax.vmap(one)(ep_keys, sample_keys))


def _sample(key, genomes, alive, n: int):
    alive = jnp.asarray(alive)
    n_alive = jnp.sum(alive)
    prio = jnp.where(alive, jax.random.uniform(key, alive.shape), jnp.inf)
    order = jnp.argsort(prio)
    idx = order[jnp.arange(n) % jnp.maximum(n_alive, 1)]
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[idx], genomes), n_alive


def episode_keys(key, n_episodes: int):
    """(episode keys, sampling key) for one competence measurement."""
    k_eps, k_sample = jax.random.split(key)
    return jax.random.split(k_eps, n_episodes), k_sample


def policy_competence(world: World, key, env: EnvParams, policy: dict, n_episodes: int) -> float:
    ep_keys, k_sample = episode_keys(key, n_episodes)
    n = world.cfg.n_slots
    if policy["kind"] == "shared":
        traits = jnp.broadcast_to(jnp.asarray(policy["traits"]), (n, 3))
        return float(competence_shared(world, ep_keys, env, policy["brain"], traits))
    return float(_population_competence(world, ep_keys, jax.random.split(k_sample, n_episodes), env,
                                        (policy["genomes"], jnp.asarray(policy["alive"])), n))


def policy_persistence(world_eco: World, key, env: EnvParams, policy: dict, n_agents: int, steps: int) -> dict:
    k_s, k_p = jax.random.split(key)
    if policy["kind"] == "shared":
        g = broadcast_genome(policy["brain"], jnp.asarray(policy["traits"]), n_agents)
    else:
        g, _ = _sample_from_population(k_s, policy, n_agents)
    out = persistence(world_eco, k_p, env, g, n_start=n_agents, steps=steps)
    return {k: (bool(v) if v.dtype == bool else float(v)) for k, v in jax.device_get(out).items()}


def zero_shot(key, env: EnvParams, policy: dict, ecfg: dict) -> dict:
    """Competence, random-policy competence on the same episodes, and persistence."""
    world = get_world("episodic")
    world_eco = get_world("ecological")
    k_c, k_p = jax.random.split(key)
    n_ep = ecfg["episodes_per_env"]
    ep_keys, _ = episode_keys(k_c, n_ep)   # the same episodes the policy plays
    return {
        "competence": policy_competence(world, k_c, env, policy, n_ep),
        "random_competence": float(random_competence_jit(world, ep_keys, env)),
        "persistence": policy_persistence(world_eco, k_p, env, policy, ecfg["persistence_agents"],
                                          ecfg["persistence_steps"]),
    }


def adaptation(eval_key, run_key, method: str, mcfg: dict, ecfg: dict, env: EnvParams, policy: dict,
               budget: int, n_checkpoints: int, start_competence: float) -> dict:
    """Continue each method's own improvement in env. Point 0 is the zero-shot
    competence (passed in, so both protocols report the same number); points
    1..n are measured after each 1/n of the budget. eval_key sets the
    evaluation episodes (the same for every method and run); run_key sets the
    adaptation's own randomness (different for every run). mcfg is the run's
    own methods config; ecfg the evaluation settings."""
    world = get_world("episodic")
    trainer = get_trainer(method, mcfg, adapt=True)   # one compiled program for every env and run
    state = trainer.init(run_key, policy, env)
    eval_keys = jax.random.split(eval_key, n_checkpoints + 1)
    curve = [{"agent_steps": 0, "competence": start_competence}]
    chunks = []
    timer = Timer()
    steps = 0
    for c in range(1, n_checkpoints + 1):
        state, steps = advance(trainer, state, steps, budget * c // n_checkpoints,
                               on_chunk=lambda st, total, summary: chunks.append({"agent_steps_total": total, **summary}))
        pol = jax.device_get(trainer.policy(state))
        curve.append({"agent_steps": steps,
                      "competence": policy_competence(world, eval_keys[c], env, pol, ecfg["episodes_per_env"])})
    return {"curve": curve, "chunks": chunks, "wall_seconds": round(timer.elapsed(), 2)}
