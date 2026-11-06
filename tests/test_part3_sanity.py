"""Baseline sanity checks on a standard gymnax task (SPEC_part3.md, Section 5).

The same PPO and ES code used in the Part 3 world is run on Pendulum-v1.
Targets (draft; fixed in docs/PREREG_part3.md):
  * PPO: mean return over 32 deterministic episodes >= -230.42, the mean
    reward listed on the model card of the Stable-Baselines3 RL Zoo PPO agent
    for Pendulum-v1 (huggingface.co/sb3/ppo-Pendulum-v1). Settings follow
    that card: gamma 0.9, learning rate 1e-3, GAE lambda 0.95, clip 0.2,
    10 epochs, no entropy bonus, no normalization. Budget: 1,310,720 steps
    (the card's agent used 100,000 steps; this checks correctness, not speed).
  * ES: the mean policy's return improves by at least 300 over the starting
    policy within 200 generations (my own threshold: the spec asks only that
    ES improves).
Seeds 3955-3959 (the "test" range).
"""
from __future__ import annotations

import jax

from evo2.agents.brain import apply_mlp, init_mlp
from evo2.methods.ppo import PPO, policy_mean
from evo2.methods.sanity import GymnaxAdapter, pendulum_es, pendulum_return
from evo2.seeds import base_key

PPO_TARGET = -230.42
ES_MIN_GAIN = 300.0


def test_ppo_solves_pendulum():
    ad = GymnaxAdapter("Pendulum-v1", 64)
    cfg = {"NUM_STEPS": 128, "NUM_MINIBATCHES": 32, "UPDATE_EPOCHS": 10, "LR": 1e-3, "GAMMA": 0.9,
           "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2, "ENT_COEF": 0.0, "NORMALIZE_REWARD": False}
    ppo = PPO(cfg, ad, ad.obs_dim, ad.act_dim, 64, 1)
    r = ppo.init(base_key("test", 3955))
    r, m = jax.jit(ppo.run_chunk, static_argnums=1)(r, 160)
    assert int(m["agent_steps"].sum()) == 1_310_720
    score = pendulum_return(lambda o: policy_mean(r.params, o), base_key("test", 3956))
    print("PPO Pendulum-v1 return:", score)
    assert score >= PPO_TARGET


def test_es_improves_on_pendulum():
    es = pendulum_es({"POP_SIZE": 64, "SIGMA": 0.1, "LR": 0.03})
    sol0 = {"brain": init_mlp(base_key("test", 3957), 3, 1)}
    k_eval = base_key("test", 3958)
    start = pendulum_return(lambda o: apply_mlp(sol0["brain"], o), k_eval)
    r = es.init(base_key("test", 3959), sol0)
    r, _ = jax.jit(es.run_chunk, static_argnums=1)(r, 200)
    sol = es.mean_solution(r)
    end = pendulum_return(lambda o: apply_mlp(sol["brain"], o), k_eval)
    print("ES Pendulum-v1 return: start", start, "end", end)
    assert end >= start + ES_MIN_GAIN
