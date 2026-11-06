"""Tests for Part 3 brains, genomes, methods, evaluation and metrics.

Run with: venv3/bin/python -m pytest tests/test_part3_methods.py
Seeds used here: 3900-3999 (the "test" range).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evo2 import metrics
from evo2.agents.brain import apply_mlp, apply_per_agent, init_mlp, n_weights
from evo2.agents.genome import Genome, MutationConfig, apply_births, init_population, mutate
from evo2.env import splits
from evo2.env.generator import sample_env_params
from evo2.env.world import World, make_env_params
from evo2.eval.episodes import run_episode, run_episode_counted, run_random_episode, shared_act_fn
from evo2.eval.evaluate import broadcast_genome, competence, persistence, random_competence
from evo2.methods.adapters import EpisodicWorlds, fixed_env_fn
from evo2.methods.es import world_es
from evo2.methods.oe import OE
from evo2.methods.ppo import PPO, Transition, compute_gae, init_params, ppo_loss
from evo2.seeds import base_key

EPI = splits.load_world_config("episodic")
ECO = splits.load_world_config("ecological")
RANGES, CANON = splits.load_train()


def key(i=0):
    return base_key("test", 3950 + i)


# ----- brain and genome ----------------------------------------------------

def test_brain_shape_and_size():
    p = init_mlp(key(0), EPI.obs_dim)
    assert n_weights(p) == 30 * 32 + 32 + 32 * 32 + 32 + 32 * 2 + 2 == 2114
    out = apply_mlp(p, jnp.ones((5, 30)))
    assert out.shape == (5, 2)
    # PureJaxRL's initialization: zero biases, small action layer
    assert float(jnp.abs(p["b1"]).sum()) == 0.0
    assert float(jnp.linalg.norm(p["W3"])) < 0.1


def test_per_agent_brains_differ_and_match_single():
    pops = init_population(key(1), 4, EPI.obs_dim, World(EPI).default_traits)
    obs = jax.random.normal(key(2), (4, EPI.obs_dim))
    out = apply_per_agent(pops.brain, obs)
    for i in range(4):
        one = jax.tree_util.tree_map(lambda x: x[i], pops.brain)
        assert np.allclose(out[i], apply_mlp(one, obs[i]), atol=1e-6)
    assert not np.allclose(out[0], out[1])


@pytest.mark.parametrize("start", ["low", "high", "default"])
def test_mutation_keeps_traits_in_bounds(start):
    w = World(ECO)
    lo, hi = w.trait_lo, w.trait_hi
    t0 = {"low": lo, "high": hi, "default": w.default_traits}[start]
    g = Genome(brain=init_mlp(key(3), EPI.obs_dim), traits=t0)
    mcfg = MutationConfig(trait_frac=0.5)  # large steps, to hit the bounds often
    for k in jax.random.split(key(4), 300):
        g = mutate(k, g, mcfg, lo, hi)
        assert bool(jnp.all((g.traits >= lo) & (g.traits <= hi)))


def test_mutation_sizes_and_body_switch():
    w = World(ECO)
    g = Genome(brain=init_mlp(key(5), EPI.obs_dim), traits=w.default_traits)
    m = mutate(key(6), g, MutationConfig(), w.trait_lo, w.trait_hi)
    diffs = np.concatenate([np.ravel(a - b) for a, b in zip(jax.tree_util.tree_leaves(m.brain),
                                                             jax.tree_util.tree_leaves(g.brain))])
    assert abs(diffs.std() - 0.02) < 0.002 and abs(diffs.mean()) < 0.002
    many = jax.vmap(lambda k: mutate(k, g, MutationConfig(), w.trait_lo, w.trait_hi).traits)(
        jax.random.split(key(7), 4000))
    sd = np.asarray(many).std(axis=0)
    assert np.allclose(sd, 0.05 * np.asarray(w.trait_hi - w.trait_lo), rtol=0.1)
    fixed = mutate(key(8), g, MutationConfig(evolve_body=False), w.trait_lo, w.trait_hi)
    assert np.array_equal(np.asarray(fixed.traits), np.asarray(g.traits))


def test_apply_births_writes_only_children():
    w = World(ECO)
    pop = init_population(key(9), 6, EPI.obs_dim, w.default_traits)
    parent_idx = jnp.array([-1, -1, 0, -1, 1, -1])
    new = apply_births(key(10), pop, parent_idx, MutationConfig(), w.trait_lo, w.trait_hi)
    for i in (0, 1, 3, 5):
        assert np.array_equal(np.asarray(new.brain["W1"][i]), np.asarray(pop.brain["W1"][i]))
    for child, parent in ((2, 0), (4, 1)):
        d = np.asarray(new.brain["W1"][child] - pop.brain["W1"][parent])
        assert 0.01 < d.std() < 0.03
    copies = apply_births(key(10), pop, parent_idx, MutationConfig(), w.trait_lo, w.trait_hi,
                          mutate_on=False)
    assert np.array_equal(np.asarray(copies.brain["W1"][2]), np.asarray(pop.brain["W1"][0]))


# ----- PPO -----------------------------------------------------------------

def test_gae_hand_computed():
    # one agent, 3 steps; done after step 2 (index 1)
    r = jnp.array([1.0, 2.0, 3.0])[:, None]
    v = jnp.array([0.5, 1.0, 2.0])[:, None]
    d = jnp.array([0.0, 1.0, 0.0])[:, None]
    tr = Transition(obs=None, action=None, log_prob=None, value=v, reward=r, done=d, valid=None)
    g, lam = 0.9, 0.5
    adv, targ = compute_gae(tr, jnp.array([4.0]), g, lam)
    a2 = 3.0 + g * 4.0 - 2.0                      # step 3 bootstraps from last value
    a1 = 2.0 - 1.0                                # done: no look-ahead
    a0 = (1.0 + g * 1.0 - 0.5) + g * lam * a1
    assert np.allclose(np.asarray(adv[:, 0]), [a0, a1, a2], atol=1e-6)
    assert np.allclose(np.asarray(targ), np.asarray(adv + v))


def _batch(n_agents=2, t=4, seed=11):
    k = jax.random.split(key(seed), 4)
    obs = jax.random.normal(k[0], (t, 1, n_agents, EPI.obs_dim))
    act = jax.random.normal(k[1], (t, 1, n_agents, 2))
    return obs, act, k


def test_parameter_sharing_one_network_for_all_agents():
    params = init_params(key(12), EPI.obs_dim, 2)
    assert set(params) == {"actor", "log_std", "critic"}
    assert params["actor"]["W1"].shape == (EPI.obs_dim, 32)   # no agent axis: one set of weights
    obs = jax.random.normal(key(13), (3, 16, EPI.obs_dim))
    out = apply_mlp(params["actor"], obs)
    for i in range(16):
        assert np.allclose(out[:, i], apply_mlp(params["actor"], obs[:, i]), atol=1e-6)


def test_gradients_pool_across_agents_and_skip_dead_steps():
    params = init_params(key(14), EPI.obs_dim, 2)
    obs, act, k = _batch()
    shape = obs.shape[:3]
    old_lp = jnp.zeros(shape)
    val = jnp.zeros(shape)
    gae = jax.random.normal(k[2], shape)
    targ = jax.random.normal(k[3], shape)
    cfg = {"CLIP_EPS": 0.2, "VF_COEF": 0.5, "ENT_COEF": 0.0}

    def grads(valid, obs=obs):
        b = Transition(obs, act, old_lp, val, jnp.zeros(shape), jnp.zeros(shape), valid)
        return jax.grad(lambda p: ppo_loss(p, b, gae, targ, cfg)[0])(params)

    only0 = jnp.zeros(shape, bool).at[:, :, 0].set(True)
    only1 = jnp.zeros(shape, bool).at[:, :, 1].set(True)
    both = jnp.ones(shape, bool)
    g0, g1, gb = grads(only0), grads(only1), grads(both)
    w0, w1, wb = g0["actor"]["W1"], g1["actor"]["W1"], gb["actor"]["W1"]
    assert float(jnp.abs(w0).sum()) > 0 and float(jnp.abs(w1).sum()) > 0
    assert not np.allclose(w0, w1)
    assert not np.allclose(wb, w0) and not np.allclose(wb, w1)   # both agents' data reach the one network
    # steps marked invalid (agent dead) do not change the gradient
    junk = obs.at[:, :, 1].set(1e3)
    gj = grads(only0, obs=junk)
    for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(gj)):
        assert np.allclose(a, b, atol=1e-6)


def test_ppo_runs_and_counts_agent_steps():
    w = World(EPI)
    ad = EpisodicWorlds(w, 2, lambda k: sample_env_params(k, RANGES), w.default_traits)
    ppo = PPO({"NUM_STEPS": 8, "NUM_MINIBATCHES": 4, "UPDATE_EPOCHS": 1}, ad, EPI.obs_dim, 2, 2, 16)
    r = ppo.init(key(15))
    r, m = jax.jit(ppo.run_chunk, static_argnums=1)(r, 2)
    assert ppo.max_agent_steps_per_update == 8 * 2 * 16
    assert np.all(np.asarray(m["agent_steps"]) <= 8 * 2 * 16)

    # compute = valid (living) agent-steps: with odd-numbered agents marked dead, PPO
    # must count exactly half the slot-steps
    class HalfDead:
        def __init__(self, inner):
            self.inner = inner

        def reset(self, k):
            return self.inner.reset(k)

        def step(self, k, st, a):
            st, obs, rew, done, valid, info = self.inner.step(k, st, a)
            return st, obs, rew, done, valid & (jnp.arange(16) % 2 == 0)[None, :], info

    ppo2 = PPO({"NUM_STEPS": 8, "NUM_MINIBATCHES": 4, "UPDATE_EPOCHS": 1}, HalfDead(ad), EPI.obs_dim, 2, 2, 16)
    r2, m2 = jax.jit(ppo2.run_chunk, static_argnums=1)(ppo2.init(key(15)), 2)
    # agents still alive at the start of the update are all present for 8 steps (no deaths that early)
    assert list(np.asarray(m2["agent_steps"])) == [8 * 2 * 8, 8 * 2 * 8]
    assert np.all(np.isfinite(np.asarray(m["loss"])))
    with pytest.raises(ValueError):
        PPO({"NO_SUCH": 1}, ad, EPI.obs_dim, 2, 2, 16)


def test_episodic_worlds_restart_when_episode_ends():
    w = World(EPI)
    ad = EpisodicWorlds(w, 2, fixed_env_fn(CANON), w.default_traits)
    (state, params), obs = ad.reset(key(16))
    state = state.replace(t=jnp.array([EPI.episode_len - 1, 3]))
    (s2, p2), obs2, rew, done, valid, info = ad.step(key(17), (state, params), jnp.zeros((2, 16, 2)))
    assert list(np.asarray(info["ep_end"])) == [True, False]
    assert int(s2.t[0]) == 0 and int(s2.t[1]) == 4
    assert bool(jnp.all(done[0])) and bool(s2.alive[0].all())


# ----- ES ------------------------------------------------------------------

def test_es_common_random_numbers_and_counts():
    w = World(EPI)
    es, pol = world_es({"POP_SIZE": 4, "EPISODES": 1}, w, lambda k: sample_env_params(k, RANGES), True)
    sol = pol.initial_solution(key(18))
    assert np.allclose(pol.traits_of(sol), w.default_traits, atol=1e-5)
    # the same solution scored twice with the same key gives the same score
    (a, sa), (b, sb) = es.score_fn(sol, key(19)), es.score_fn(sol, key(19))
    assert float(a) == float(b) and int(sa) == int(sb)
    # agent-steps = living agents per step, counted from the same episode run directly
    env0 = sample_env_params(jax.random.split(jax.random.split(key(19))[0], 1)[0], RANGES)
    k_ep = jax.random.split(jax.random.split(key(19))[1], 1)[0]
    _, direct = run_episode_counted(w, k_ep, env0, shared_act_fn(sol["brain"]),
                                    jnp.broadcast_to(pol.traits_of(sol), (16, 3)))
    assert int(sa) == int(direct) and 0 < int(sa) <= 16 * EPI.episode_len
    r = es.init(key(20), sol)
    r, m = jax.jit(es.run_chunk, static_argnums=1)(r, 1)
    assert 0 < int(m["agent_steps"][0]) <= es.max_agent_steps_per_generation == 4 * 16 * EPI.episode_len
    t = pol.traits_of({"brain": sol["brain"], "body_z": jnp.array([50.0, -50.0, 0.0])})
    assert np.allclose(t, [2.0, 0.5, 10.0], atol=1e-4)
    # common random numbers inside a generation: with no noise every candidate is the
    # same solution, so the scores are equal only if all share one key
    es0, _ = world_es({"POP_SIZE": 4, "EPISODES": 1, "SIGMA": 0.0}, w,
                      lambda k: sample_env_params(k, RANGES), True)
    _, m0 = jax.jit(es0.run_chunk, static_argnums=1)(es0.init(key(31), sol), 1)
    assert float(m0["min_score"][0]) == float(m0["max_score"][0])


# ----- OE ------------------------------------------------------------------

def test_oe_floor_and_counts():
    w = World(ECO)
    scarce = make_env_params(L=60.0, n_food=2, layout="uniform", rho=0.0)
    oe = OE({"MIN_POP": 8, "CHANGE_EVERY": 50}, w, fixed_env_fn(scarce), True)
    s = oe.init(key(21))
    assert int(s.world.alive.sum()) == 32
    s, log = jax.jit(oe.run_chunk, static_argnums=1)(s, 300)
    sm = OE.chunk_summary(log)
    # with almost no food the population falls to the floor, which then holds it exactly at 8
    assert sm["min_alive"] == 8 and sm["floor_births"] > 0 and sm["env_changes"] == 6
    alive_log = np.asarray(log["alive"])
    assert alive_log[-1] == 8
    assert sm["agent_steps"] == int(np.asarray(log["agent_steps"]).sum())
    # living agents' body traits match their genomes
    alive = np.asarray(s.world.alive)
    assert np.allclose(np.asarray(s.world.traits)[alive], np.asarray(s.pop.traits)[alive])


def test_oe_without_floor_can_go_extinct():
    w = World(ECO)
    params = make_env_params(L=60.0, n_food=1, layout="uniform", rho=0.0)
    oe = OE({"MIN_POP": 0, "CHANGE_EVERY": 0}, w, fixed_env_fn(params), True)
    s = oe.init(key(22))
    s, log = jax.jit(oe.run_chunk, static_argnums=1)(s, 200)
    assert int(s.world.alive.sum()) == 0 and OE.chunk_summary(log)["floor_births"] == 0


def test_oe_sample_agents():
    w = World(ECO)
    oe = OE({}, w, fixed_env_fn(CANON), True)
    s = oe.init(key(23))
    g, n_alive = OE.sample_agents(key(24), s, 16)
    assert int(n_alive) == 32 and g.traits.shape == (16, 3)
    s5 = s.replace(world=s.world.replace(alive=jnp.arange(128) < 5))
    g5, n5 = OE.sample_agents(key(25), s5, 16)
    w1 = np.asarray(g5.brain["W1"]).reshape(16, -1)
    assert int(n5) == 5 and len({row.tobytes() for row in w1}) == 5


# ----- evaluation ----------------------------------------------------------

def test_competence_matches_episode_runs_and_random_policy_is_weak():
    w = World(EPI)
    brain = init_mlp(key(26), EPI.obs_dim)
    traits = jnp.tile(w.default_traits, (16, 1))
    ks = jax.random.split(key(27), 3)
    c = float(competence(w, ks, CANON, shared_act_fn(brain), traits))
    manual = np.mean([np.asarray(run_episode(w, k, CANON, shared_act_fn(brain), traits)).mean() for k in ks])
    assert c == pytest.approx(manual, rel=1e-6)
    rc = float(random_competence(w, ks, CANON))
    assert 0.0 <= rc < 5.0
    food = np.asarray(run_random_episode(w, ks[0], CANON))
    assert food.shape == (16,)


def test_persistence_runs_and_can_end_in_extinction():
    w = World(ECO)
    brain = init_mlp(key(28), EPI.obs_dim)
    g = broadcast_genome(brain, w.default_traits, 32)
    out = persistence(w, key(29), CANON, g, steps=300)
    assert bool(out["extinct"]) and 0 < int(out["steps_survived"]) < 300


# ----- metrics -------------------------------------------------------------

def test_metrics_hand_computed():
    assert metrics.normalize(7.0, 2.0, 12.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.normalize(1.0, 3.0, 3.0)
    x = [0.0, 0.5, 1.0]
    assert metrics.curve_auc(x, [0.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert metrics.curve_auc([0, 1, 3], [2.0, 2.0, 2.0]) == pytest.approx(2.0)
    assert metrics.curve_auc([0, 2], [0.0, 1.0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.curve_auc([0, 0, 1], [1, 1, 1])
    assert metrics.time_to_fraction([0, 1, 2, 3], [0.1, 0.5, 0.95, 0.99], 0.9) == 2.0
    assert math.isnan(metrics.time_to_fraction([0, 1], [0.1, 0.2], 0.9))
    # IQM of 1..8: drop the lowest and highest 2, mean of 3, 4, 5, 6
    assert metrics.iqm(np.arange(1, 9).reshape(4, 2)) == pytest.approx(4.5)
    x = np.array([[3.0], [4.0]])
    y = np.array([[1.0], [2.0]])
    assert metrics.prob_improvement(x, y) == pytest.approx(1.0)
    assert metrics.prob_improvement(y, x) == pytest.approx(0.0)
    assert metrics.prob_improvement(np.array([[1.0], [3.0]]), np.array([[2.0], [2.0]])) == pytest.approx(0.5)
    ci = metrics.iqm_with_ci({"a": np.arange(1, 9).reshape(4, 2)}, reps=200, seed=3999)
    assert ci["a"][0] == pytest.approx(4.5) and ci["a"][1] <= 4.5 <= ci["a"][2]
    # performance profile: fraction of scores above each threshold
    assert list(metrics.performance_profile([[0.1, 0.5], [0.9, 0.3]], [0.0, 0.4, 1.0])) == [1.0, 0.5, 0.0]
    # the H1 verdict is a partition, checked in order
    m = 0.05
    assert metrics.h1_verdict(0.06, 0.2, m) == "support"
    assert metrics.h1_verdict(0.01, 0.04, m) == "support (below the margin)"
    assert metrics.h1_verdict(-0.2, -0.06, m) == "against"
    assert metrics.h1_verdict(-0.04, -0.01, m) == "against (below the margin)"
    assert metrics.h1_verdict(-0.04, 0.03, m) == "no meaningful difference"
    assert metrics.h1_verdict(-0.1, 0.03, m) == "inconclusive"


def test_bootstrap_resamples_whole_runs_and_is_reproducible():
    rng = np.random.default_rng(3998)
    # 10 runs x 20 environments: a large run effect, small within-run noise
    run_effect = rng.normal(0, 0.15, size=(10, 1))
    x = 0.5 + run_effect + rng.normal(0, 0.05, size=(10, 20))
    y = 0.5 + rng.normal(0, 0.15, size=(10, 1)) + rng.normal(0, 0.05, size=(10, 20))
    d, lo, hi = metrics.iqm_difference_ci(x, y, reps=2000, seed=4800)
    # resampling whole runs: the CI reflects the run-to-run spread (about 0.15 x sqrt(2/10)),
    # not the within-run noise
    assert (hi - lo) / 2 > 0.08
    np.random.seed(1)
    a = metrics.iqm_difference_ci(x, y, reps=500, seed=4800)
    np.random.seed(2)
    b = metrics.iqm_difference_ci(x, y, reps=500, seed=4800)
    assert a == b                                    # the global NumPy state does not matter
    c = metrics.iqm_difference_ci(x, y, reps=500, seed=4801)
    assert c != a                                    # the seed does


def test_adapter_marks_death_step_valid_and_later_steps_invalid():
    """An agent's death step counts (valid and done); its later steps do not count."""
    w = World(EPI)
    ad = EpisodicWorlds(w, 1, fixed_env_fn(CANON), w.default_traits)
    (state, params), _ = ad.reset(key(32))
    state = state.replace(energy=state.energy.at[0, 0].set(0.1))   # agent 0 starves this step
    still = jnp.full((1, 16, 2), -50.0)
    (s1, p1), _, _, done1, valid1, _ = ad.step(key(33), (state, params), still)
    assert bool(valid1[0, 0]) and bool(done1[0, 0]) and not bool(s1.alive[0, 0])
    (_, _), _, rew2, done2, valid2, _ = ad.step(key(34), (s1, p1), still)
    assert not bool(valid2[0, 0]) and bool(valid2[0, 1]) and float(rew2[0, 0]) == 0.0


def test_episode_ends_when_every_agent_is_dead():
    w = World(EPI)
    ad = EpisodicWorlds(w, 2, fixed_env_fn(CANON), w.default_traits)
    (state, params), _ = ad.reset(key(35))
    # world 0: everyone starves now (no food present, so nobody can eat first)
    state = state.replace(energy=state.energy.at[0].set(0.1),
                          food_present=state.food_present.at[0].set(False))
    (s1, _), _, _, done, _, info = ad.step(key(36), (state, params), jnp.full((2, 16, 2), -50.0))
    assert list(np.asarray(info["ep_end"])) == [True, False]
    assert int(s1.t[0]) == 0 and bool(s1.alive[0].all()) and int(s1.t[1]) == 1


def test_oe_trainer_continues_from_a_saved_population():
    from evo2.run import load_methods_config, make_trainer
    mcfg = load_methods_config()
    tr = make_trainer("oe", mcfg)
    st = tr.init(key(37))
    pol = jax.device_get(tr.policy(st))
    alive = np.zeros(128, bool)
    alive[[0, 5, 9]] = True
    pol = {**pol, "alive": alive}
    t2 = make_trainer("oe", mcfg, adapt=True)
    s2 = t2.init(key(38), pol, CANON)
    assert np.array_equal(np.asarray(s2.world.alive), alive)
    assert np.allclose(np.asarray(s2.world.energy)[alive], ECO.birth_energy)
    assert np.allclose(np.asarray(s2.world.traits), np.asarray(pol["genomes"].traits))
    assert t2.oe.cfg["CHANGE_EVERY"] == 0 and float(s2.env.L) == float(CANON.L)


def test_zero_shot_random_baseline_uses_the_policy_episodes():
    from evo2.eval.protocol import episode_keys, zero_shot
    w = World(EPI)
    brain = init_mlp(key(39), EPI.obs_dim)
    pol = {"kind": "shared", "brain": brain, "traits": w.default_traits}
    ecfg = {"episodes_per_env": 2, "persistence_agents": 32, "persistence_steps": 50}
    k = key(40)
    zs = zero_shot(k, CANON, pol, ecfg)
    ep_keys, _ = episode_keys(jax.random.split(k)[0], 2)
    assert zs["random_competence"] == pytest.approx(float(random_competence(w, ep_keys, CANON)))
    c = competence(w, ep_keys, CANON, shared_act_fn(brain), jnp.tile(w.default_traits, (16, 1)))
    assert zs["competence"] == pytest.approx(float(c))


def test_adaptation_curve_points_and_budget():
    from evo2.eval.protocol import adaptation
    from evo2.run import load_methods_config
    mcfg = load_methods_config(tuned=None)   # small shared settings; tuned per-method settings would override them
    mcfg["ppo"]["n_envs"] = 2
    mcfg["ppo"]["settings"].update({"NUM_STEPS": 8, "NUM_MINIBATCHES": 4, "UPDATE_EPOCHS": 1})
    w = World(EPI)
    pol = {"kind": "shared", "brain": init_mlp(key(41), EPI.obs_dim), "traits": w.default_traits,
           "train_params": init_params(key(42), EPI.obs_dim, 2)}
    ecfg = {"episodes_per_env": 1}
    out = adaptation(key(43), key(44), "ppo_dr", mcfg, ecfg, CANON, pol, 4 * 256, 4, start_competence=3.5)
    curve = out["curve"]
    assert len(curve) == 5 and curve[0] == {"agent_steps": 0, "competence": 3.5}
    marks = [256, 512, 768, 1024]
    for pt, mark in zip(curve[1:], marks):
        assert mark <= pt["agent_steps"] < mark + 256
    assert sum(c["agent_steps"] for c in out["chunks"]) == curve[-1]["agent_steps"]


def test_config_body_option_b_gives_default_bodies_everywhere():
    """AA's check-in 7 decision: every method uses the default body; OE mutates brains only."""
    from evo2.run import load_methods_config, make_trainer
    mcfg = load_methods_config()
    assert mcfg["body_option"] == "b"
    w = World(EPI)
    oe = make_trainer("oe", mcfg)
    assert oe.oe.mcfg.evolve_body is False
    st = oe.init(key(45))
    st, _ = oe.chunk(st, 200)
    assert np.allclose(np.asarray(st.pop.traits), np.asarray(w.default_traits))      # no trait mutation
    assert np.allclose(np.asarray(st.world.traits)[np.asarray(st.world.alive)], np.asarray(w.default_traits))
    ppo = make_trainer("ppo_dr", mcfg)
    assert np.allclose(np.asarray(ppo.traits), np.asarray(w.default_traits))
    es = make_trainer("es_dr", mcfg)
    sol = es.pol.initial_solution(key(46))
    assert "body_z" not in sol and np.allclose(np.asarray(es.pol.traits_of(sol)), np.asarray(w.default_traits))
    with pytest.raises(ValueError):
        make_trainer("oe", {**mcfg, "body_option": "c"})


# --- one compiled program per trainer (2 Nov 2026, after P0) ---

def k2(j):
    """Keys for these tests, derived from test seed 3997 (key(i) itself only reaches i = 49)."""
    return jax.random.fold_in(key(47), j)


def _leaves_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_run_dynamic_matches_scan_for_every_length():
    """run_dynamic must run the step exactly n times and keep exactly n log rows."""
    from evo2.methods.loop import run_dynamic

    def step(s, _):
        k, x = s
        k, k1 = jax.random.split(k)
        x = 0.9 * x + jax.random.normal(k1, x.shape)
        return (k, x), {"sum": jnp.sum(x), "n": jnp.int32(x.size)}

    s0 = (k2(0), jnp.zeros(5))
    dyn = jax.jit(lambda s, n: run_dynamic(step, s, n, 8))
    for n in (1, 3, 8):
        want_s, want_log = jax.lax.scan(step, s0, None, n)
        got_s, got_log = dyn(s0, np.int32(n))
        assert _leaves_equal(got_s, want_s)
        assert _leaves_equal({k: v[:n] for k, v in got_log.items()}, want_log)
        assert not np.asarray(got_log["n"][n:]).any()   # rows past n stay empty
    assert dyn._cache_size() == 1


def test_trainer_chunks_of_any_length_share_one_program_and_match_scan():
    from evo2.run import load_methods_config, make_trainer
    mcfg = load_methods_config()
    tr = make_trainer("oe", mcfg)
    st = tr.init(k2(1))
    seq = [tr.chunk(st, 7)]
    seq.append(tr.chunk(seq[-1][0], 2))
    seq.append(tr.chunk(seq[-1][0], 1))
    assert tr._run._cache_size() == 1
    assert [sm["steps"] for _, sm in seq] == [7, 2, 1]     # summaries read exactly n rows
    want, log = jax.jit(tr.oe.run_chunk, static_argnums=1)(st, 10)
    assert _leaves_equal(seq[-1][0], want)
    assert sum(sm["agent_steps"] for _, sm in seq) == int(np.asarray(log["agent_steps"]).astype(np.int64).sum())
    with pytest.raises(ValueError):
        tr.chunk(st, tr.chunk_units + 1)
    with pytest.raises(ValueError):
        tr.chunk(st, 0)


def test_fixed_env_as_data_plays_the_same_worlds_as_env_fn():
    """PPO and ES adaptation keep the test environment in the state (env_fn=None).
    The worlds they play must be those of the env_fn version (fixed_env_fn).
    Both sides get the environment as data: with the environment built into
    the compiled program instead, the compiler rounds some arithmetic
    differently in the last bit, and the world, being chaotic, then drifts
    apart (2 Nov 2026: first difference 9.5e-7 at step 37 of a trained ES
    policy's episode); that is a compiler effect, not this logic."""
    from evo2.methods.es import world_es
    w = World(EPI)
    env = sample_env_params(k2(2), splits.load_train()[0])
    data = EpisodicWorlds(w, 2, None, w.default_traits)

    def via_env_fn(e):
        return EpisodicWorlds(w, 2, fixed_env_fn(e), w.default_traits)

    sc, oc = jax.jit(lambda k, e: via_env_fn(e).reset(k))(k2(3), env)
    sd, od = jax.jit(data.reset)(k2(3), env)
    assert _leaves_equal(sc, sd) and _leaves_equal(oc, od)
    act = jnp.zeros((2, EPI.n_slots, 2))
    step_c = jax.jit(lambda k, s, a, e: via_env_fn(e).step(k, s, a))
    step_d = jax.jit(data.step)
    ended = 0
    for k in jax.random.split(k2(4), 600):        # past the 500-step episode end: restarts happen
        sc, oc, *_, ic = step_c(k, sc, act, env)
        sd, od, *_, idd = step_d(k, sd, act)
        ended += int(np.asarray(ic["ep_end"]).sum())
    assert ended > 0 and _leaves_equal(sc, sd) and _leaves_equal(oc, od)
    assert all(np.all(np.asarray(p) == np.asarray(e))     # every world still plays env
               for p, e in zip(jax.tree_util.tree_leaves(sd[1]), jax.tree_util.tree_leaves(env)))
    with pytest.raises(ValueError):
        data.reset(k2(3))
    with pytest.raises(ValueError):
        via_env_fn(env).reset(k2(3), env)

    cfg = {"POP_SIZE": 4, "EPISODES": 2}
    es_d, pol = world_es(cfg, w, None, evolve_body=False)
    sol = pol.initial_solution(k2(5))
    a = jax.jit(lambda s, k, e: world_es(cfg, w, fixed_env_fn(e), False)[0].score_fn(s, k))(sol, k2(6), env)
    b = jax.jit(es_d.score_fn)(sol, k2(6), env)
    assert _leaves_equal(a, b)
    with pytest.raises(ValueError):
        es_d.score_fn(sol, k2(6))
    # a whole ES generation: the runner's environment reaches every candidate
    r0 = es_d.init(k2(11), sol)
    rc, mc = jax.jit(lambda r, e: world_es(cfg, w, fixed_env_fn(e), False)[0]._generation(r, None))(r0, env)
    rd, md = jax.jit(es_d._generation)(es_d.init(k2(11), sol, ctx=env), None)
    assert _leaves_equal(mc, md) and _leaves_equal(rc.state, rd.state) and _leaves_equal(rd.ctx, env)


def test_adaptation_trainer_is_shared_and_uses_the_env_it_is_given():
    from evo2.run import get_trainer, load_methods_config
    mcfg = load_methods_config()
    env2 = sample_env_params(k2(7), splits.load_train()[0])
    for m in ("ppo_dr", "es_dr", "oe"):
        t = get_trainer(m, mcfg, adapt=True)
        assert get_trainer(m, mcfg, adapt=True) is t and get_trainer(m, mcfg) is not t
        with pytest.raises(ValueError):
            t.init(k2(8))                    # an adaptation trainer needs the environment
        with pytest.raises(ValueError):
            get_trainer(m, mcfg).init(k2(8), None, CANON)
    t = get_trainer("ppo_dr", mcfg, adapt=True)
    for env in (CANON, env2):
        r = t.init(k2(9), None, env)
        r, _ = t.chunk(r, 1)
        params = r.env_state[1]
        assert all(np.all(np.asarray(p) == np.asarray(e)) for p, e in zip(jax.tree_util.tree_leaves(params),
                                                                         jax.tree_util.tree_leaves(env)))
    assert t._run._cache_size() == 1
    t = get_trainer("es_dr", mcfg, adapt=True)
    r = t.init(k2(10), None, env2)
    assert _leaves_equal(r.ctx, env2)
