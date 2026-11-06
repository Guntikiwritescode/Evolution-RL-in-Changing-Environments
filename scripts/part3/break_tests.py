"""Show that every Part 3 test can fail.

For each test: copy the repo to a temporary folder, break the code in one
named way, run only that test there, and record whether it failed. A test
"counts" only if it fails on its broken copy. Writes
results/part3/break_tests.json and exits non-zero if any test still passed.

  venv3/bin/python scripts/part3/break_tests.py            # all
  venv3/bin/python scripts/part3/break_tests.py --only sensors
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = "src/evo2/env/world.py"

# (name, test node id, [(file, old, new)]) ; new=None with old=None creates a file with content in "new_file"
BREAKS = [
    ("energy", "tests/test_part3_env.py::test_energy_conservation_every_step",
     [(W, "removed_at_death = jnp.sum(jnp.where(died, energy, 0.0))", "removed_at_death = 0.0 * jnp.sum(jnp.where(died, energy, 0.0))")]),
    ("food", "tests/test_part3_env.py::test_food_conservation_without_regrowth",
     [(W, "food_present=state.food_present & ~food_eaten,", "food_present=state.food_present,")]),
    ("ties", "tests/test_part3_env.py::test_one_item_eaten_once_and_ties_random",
     [(W, "winner = jnp.argmin(jnp.where(cand, draw[:, None], jnp.inf), axis=0)", "winner = jnp.argmax(cand, axis=0)")]),
    ("inactive", "tests/test_part3_env.py::test_inactive_food_slots_never_present",
     [(W, "return jnp.arange(self.cfg.max_food) < params.n_food", "return jnp.arange(self.cfg.max_food) <= params.n_food")]),
    ("determinism", "tests/test_part3_env.py::test_determinism_same_key_same_result",
     [(W, "draw = jax.random.uniform(k_eat, (n,))", "draw = jnp.asarray(__import__('numpy').random.rand(n))"),
      (W, "eligible = state.alive & (state.energy >= cfg.birth_threshold)",
       "eligible = state.alive & (state.energy >= cfg.birth_threshold) & jnp.asarray(__import__('numpy').random.rand(n) < 0.5)")]),
    ("bounds", "tests/test_part3_env.py::test_action_and_trait_bounds_hold",
     [(W, "a = jnp.tanh(action)", "a = action")]),
    ("cost", "tests/test_part3_env.py::test_energy_cost_formula",
     [(W, "size ** 3 * speed ** 2", "size ** 2 * speed ** 2")]),
    ("sensors", "tests/test_part3_env.py::test_sensors_hand_computed",
     [(W, "jnp.mod(ang + width / 2.0, 2.0 * jnp.pi)", "jnp.mod(ang, 2.0 * jnp.pi)")]),
    ("births", "tests/test_part3_env.py::test_births_rules",
     [(W, "cfg.parent_birth_cost, mode=\"drop\")", "cfg.birth_energy, mode=\"drop\")")]),
    ("full", "tests/test_part3_env.py::test_no_births_when_full_parent_keeps_energy",
     [(W, "n_births = jnp.minimum(jnp.sum(eligible), jnp.sum(free)).astype(jnp.int32)", "n_births = jnp.sum(eligible).astype(jnp.int32)")]),
    ("free_slots", "tests/test_part3_env.py::test_births_limited_by_free_slots",
     [(W, "n_births = jnp.minimum(jnp.sum(eligible), jnp.sum(free)).astype(jnp.int32)", "n_births = jnp.sum(eligible).astype(jnp.int32)")]),
    ("age", "tests/test_part3_env.py::test_death_by_age_only_in_ecological_mode",
     [(W, "        if cfg.ecological:\n            died = died | (alive0 & (age >= cfg.max_age))",
       "        if True:\n            died = died | (alive0 & (age >= cfg.max_age))")]),
    ("layouts", "tests/test_part3_env.py::test_layouts_place_food_where_expected",
     [(W, "rad = params.patch_r * jnp.sqrt(", "rad = 2.0 * params.patch_r * jnp.sqrt(")]),
    ("drift", "tests/test_part3_env.py::test_drift_moves_food_and_bounces",
     [(W, "moved = food_pos + params.drift * jnp.stack([dx, dy], axis=1)", "moved = food_pos + 0.5 * params.drift * jnp.stack([dx, dy], axis=1)")]),
    ("season", "tests/test_part3_env.py::test_season_moves_centres_only_on_period",
     [(W, "(jnp.mod(state.t + 1, jnp.maximum(params.season_P, 1)) == 0)", "(jnp.mod(state.t, jnp.maximum(params.season_P, 1)) == 0)")]),
    ("regrowth", "tests/test_part3_env.py::test_regrowth_probability_extremes",
     [(W, "(jax.random.uniform(k_u, (m,)) < params.rho)", "(jax.random.uniform(k_u, (m,)) < 0.5)")]),
    ("change_env", "tests/test_part3_env.py::test_change_env_clips_agents_and_lays_out_food",
     [(W, "pos=jnp.clip(state.pos, 0.0, params.L),\n            food_pos=", "pos=state.pos,\n            food_pos=")]),
    ("int_ranges", "tests/test_part3_env.py::test_train_ranges_and_canonical_match_spec",
     [("src/evo2/env/generator.py", "lohi[0], lohi[1] + 1)", "lohi[0], lohi[1])")]),
    ("validation", "tests/test_part3_env.py::test_validation_set",
     [("src/evo2/env/splits.py", "range(int(lo), int(hi) + 1)", "range(int(lo), int(hi))")]),
    ("lock", "tests/test_part3_env.py::test_test_split_locked",
     [("src/evo2/env/splits.py", "if final is not True:", "if not final:")]),
    ("test_file", "tests/test_part3_env.py::test_test_split_file_parses_with_final",
     [("configs/part3/test_split.yaml", "    L: [30.0, 40.0]", "    L: [30.0, 60.0]")]),
    ("naming", "tests/test_part3_env.py::test_only_evaluate_may_name_the_test_split",
     [("scripts/part3/tune.py", None, "from evo2.env.splits import load_test_split\n")]),
    ("seed_ranges", "tests/test_part3_env.py::test_seed_ranges_do_not_overlap_and_match_plan",
     [("src/evo2/seeds.py", '"final": (5000, 5499),', '"final": (4990, 5499),')]),
    ("check_seed", "tests/test_part3_env.py::test_check_seed_rejects_wrong_purpose",
     [("src/evo2/seeds.py", "if not (lo <= int(seed) <= hi):", "if False:")]),
    ("streams", "tests/test_part3_env.py::test_stream_keys_serve_one_purpose_each",
     [("src/evo2/seeds.py", '"probe": 14,', '"probe": 13,')]),
    ("range_checks", "tests/test_part3_env.py::test_validation_and_test_ranges_parse_like_train",
     [("src/evo2/env/generator.py", "        if lo > hi:", "        if False:")]),
    ("imports", "tests/test_part3_env.py::test_part3_libraries_import",
     [("src/rliable/__init__.py", None, "raise ImportError('broken on purpose')\n")]),
    ("brain_size", "tests/test_part3_methods.py::test_brain_shape_and_size",
     [("src/evo2/agents/brain.py", "HIDDEN = 32", "HIDDEN = 33")]),
    ("per_agent", "tests/test_part3_methods.py::test_per_agent_brains_differ_and_match_single",
     [("src/evo2/agents/brain.py", "return jax.vmap(apply_mlp)(params, obs)",
       "return apply_mlp(jax.tree_util.tree_map(lambda x: x[0], params), obs)")]),
    ("mut_bounds", "tests/test_part3_methods.py::test_mutation_keeps_traits_in_bounds",
     [("src/evo2/agents/genome.py", "traits = jnp.clip(g.traits + sd * jax.random.normal(keys[-1], g.traits.shape),\n                          trait_lo, trait_hi)",
       "traits = g.traits + sd * jax.random.normal(keys[-1], g.traits.shape)")]),
    ("mut_sizes", "tests/test_part3_methods.py::test_mutation_sizes_and_body_switch",
     [("src/evo2/agents/genome.py", "new_leaves = [x + mcfg.weight_sd *", "new_leaves = [x + 2 * mcfg.weight_sd *")]),
    ("births_write", "tests/test_part3_methods.py::test_apply_births_writes_only_children",
     [("src/evo2/agents/genome.py", "        return jnp.where(mask, new, old)", "        return new")]),
    ("gae", "tests/test_part3_methods.py::test_gae_hand_computed",
     [("src/evo2/methods/ppo.py", "gae = delta + gamma * lam * (1 - tr.done) * gae", "gae = delta + gamma * lam * gae")]),
    ("sharing", "tests/test_part3_methods.py::test_parameter_sharing_one_network_for_all_agents",
     [("src/evo2/methods/ppo.py", '"actor": init_mlp(k_a, obs_dim, act_dim, out_gain=0.01),',
       '"actor": jax.vmap(lambda k: init_mlp(k, obs_dim, act_dim, out_gain=0.01))(jax.random.split(k_a, 16)),')]),
    ("masking", "tests/test_part3_methods.py::test_gradients_pool_across_agents_and_skip_dead_steps",
     [("src/evo2/methods/ppo.py", "    return jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0)", "    return jnp.mean(x)")]),
    ("ppo_count", "tests/test_part3_methods.py::test_ppo_runs_and_counts_agent_steps",
     [("src/evo2/methods/ppo.py", '"agent_steps": jnp.sum(logs["valid"]).astype(jnp.int32),',
       '"agent_steps": jnp.int32(self.max_agent_steps_per_update),')]),
    ("es_count", "tests/test_part3_methods.py::test_es_common_random_numbers_and_counts",
     [("src/evo2/methods/es.py", "return jnp.mean(food.astype(jnp.float32)), jnp.sum(steps)",
       "return jnp.mean(food.astype(jnp.float32)), jnp.int32(n * world.cfg.episode_len * e)")]),
    ("restart", "tests/test_part3_methods.py::test_episodic_worlds_restart_when_episode_ends",
     [("src/evo2/methods/adapters.py", "ep_end = (new.t >= self.world.cfg.episode_len)", "ep_end = (new.t > self.world.cfg.episode_len)")]),
    ("all_dead_end", "tests/test_part3_methods.py::test_episode_ends_when_every_agent_is_dead",
     [("src/evo2/methods/adapters.py", " | ~jnp.any(new.alive, axis=1)", "")]),
    ("valid_mask", "tests/test_part3_methods.py::test_adapter_marks_death_step_valid_and_later_steps_invalid",
     [("src/evo2/methods/adapters.py", 'valid = info["alive_start"]', "valid = new.alive")]),
    ("oe_continue", "tests/test_part3_methods.py::test_oe_trainer_continues_from_a_saved_population",
     [("src/evo2/run.py", 'ws = s.world.replace(alive=jnp.asarray(policy["alive"]),',
       'ws = s.world.replace(alive=jnp.ones_like(jnp.asarray(policy["alive"])),')]),
    ("same_episodes", "tests/test_part3_methods.py::test_zero_shot_random_baseline_uses_the_policy_episodes",
     [("src/evo2/eval/protocol.py", '"random_competence": float(random_competence_jit(world, ep_keys, env)),',
       '"random_competence": float(random_competence_jit(world, jax.random.split(k_p, n_ep), env)),')]),
    ("adapt_point0", "tests/test_part3_methods.py::test_adaptation_curve_points_and_budget",
     [("src/evo2/eval/protocol.py", 'curve = [{"agent_steps": 0, "competence": start_competence}]',
       'curve = [{"agent_steps": 0, "competence": 0.0}]')]),
    ("run_bootstrap", "tests/test_part3_methods.py::test_bootstrap_resamples_whole_runs_and_is_reproducible",
     [("src/evo2/metrics.py", "    idx = rng.integers(0, scores.shape[0], size=scores.shape[0])\n    return scores[idx]",
       "    idx = rng.integers(0, scores.shape[0], size=scores.shape)\n    return np.take_along_axis(scores, idx, axis=0)")]),
    ("body_b", "tests/test_part3_methods.py::test_config_body_option_b_gives_default_bodies_everywhere",
     [("configs/part3/methods.yaml", "body_option: b ", "body_option: a ")]),
    ("pilot_shift", "tests/test_part3_env.py::test_pilot_shift_copies_test_ranges_with_pilot_seeds",
     [("configs/part3/pilot_shift.yaml", "    L: [30.0, 40.0]", "    L: [30.0, 45.0]")]),
    ("crn", "tests/test_part3_methods.py::test_es_common_random_numbers_and_counts",
     [("src/evo2/methods/es.py", "score, steps = jax.vmap(lambda s: self.score_fn(s, k_score))(pop)",
       "score, steps = jax.vmap(self.score_fn)(pop, jax.random.split(k_score, self.cfg['POP_SIZE']))")]),
    ("floor", "tests/test_part3_methods.py::test_oe_floor_and_counts",
     [("src/evo2/methods/oe.py", 'jnp.maximum(self.cfg["MIN_POP"] - jnp.sum(ws.alive), 0)', 'jnp.maximum(self.cfg["MIN_POP"] // 2 - jnp.sum(ws.alive), 0)')]),
    ("no_floor", "tests/test_part3_methods.py::test_oe_without_floor_can_go_extinct",
     [("src/evo2/methods/oe.py", "self.cfg = {**DEFAULTS, **cfg}", 'self.cfg = {**DEFAULTS, **cfg, "MIN_POP": 16}')]),
    ("sample", "tests/test_part3_methods.py::test_oe_sample_agents",
     [("src/evo2/methods/oe.py", "idx = order[jnp.arange(n) % jnp.maximum(n_alive, 1)]", "idx = order[jnp.arange(n)]")]),
    ("competence", "tests/test_part3_methods.py::test_competence_matches_episode_runs_and_random_policy_is_weak",
     [("src/evo2/eval/evaluate.py", "    return jnp.mean(food.astype(jnp.float32))\n\n\ndef random_competence",
       "    return jnp.max(food.astype(jnp.float32))\n\n\ndef random_competence")]),
    ("persistence", "tests/test_part3_methods.py::test_persistence_runs_and_can_end_in_extinction",
     [("src/evo2/eval/evaluate.py", 'oe = OE({"CHANGE_EVERY": 0, "MIN_POP": 0}', 'oe = OE({"CHANGE_EVERY": 0, "MIN_POP": 16}')]),
    ("auc", "tests/test_part3_methods.py::test_metrics_hand_computed",
     [("src/evo2/metrics.py", "return np.sum(widths * heights, axis=-1) / (x[-1] - x[0])", "return np.sum(widths * heights, axis=-1)")]),
    ("resume", "tests/test_part3_scripts.py::test_resume_gives_the_same_result",
     [("src/evo2/train_loop.py", 'steps, chunks, wall = saved["agent_steps"], saved["chunks"], saved["wall_seconds"]',
       "steps, chunks, wall = 0, 0, 0.0")]),
    ("overshoot", "tests/test_part3_scripts.py::test_budget_logic_never_passes_target_by_a_unit",
     [("src/evo2/run.py", "    if room >= trainer.chunk_units:\n        return trainer.chunk_units",
       "    if True:\n        return trainer.chunk_units")]),
    ("resume_path", "tests/test_part3_scripts.py::test_resumed_budget_path_matches_uninterrupted",
     [("src/evo2/run.py", "state, summary = trainer.chunk(state, chunk_size(trainer, steps, target))",
       "state, summary = trainer.chunk(state, None if done == 1 else chunk_size(trainer, steps, target))")]),
    ("real_overshoot", "tests/test_part3_scripts.py::test_real_runs_stop_within_one_unit",
     [("src/evo2/run.py", "self.unit_max_agent_steps = self.world.cfg.n_slots   # one OE step, every slot alive",
       "self.unit_max_agent_steps = 1")]),
    ("mixing", "tests/test_part3_scripts.py::test_run_folder_refuses_another_config",
     [("src/evo2/train_loop.py", "        if not same:\n", "        if False:\n")]),
    ("final_runs_only", "tests/test_part3_scripts.py::test_evaluate_final_refuses_runs_that_are_not_final",
     [("scripts/part3/evaluate.py", 'if a.final and metas[run]["purpose"] != "final":', "if False:")]),
    ("failed_run", "tests/test_part3_scripts.py::test_failed_run_is_recorded_when_parameters_stop_being_finite",
     [("src/evo2/train_loop.py", 'counter["bad"] = counter["bad"] or not finite', 'counter["bad"] = False')]),
    ("clean", "tests/test_part3_scripts.py::test_non_test_purposes_need_a_clean_tree",
     [("src/evo2/runio.py", 'if not g["clean"] and not (quick and purpose == "test"):', 'if not g["clean"] and not quick:')]),
    ("eval_lock", "tests/test_part3_scripts.py::test_evaluate_refuses_test_split_without_final",
     [("scripts/part3/evaluate.py", 'if a.split == "test" and not a.final:', "if False:")]),
    ("train_final", "tests/test_part3_scripts.py::test_train_script_has_no_way_to_the_test_split",
     [("scripts/part3/train.py", 'ap.add_argument("--quick",', 'ap.add_argument("--final", action="store_true")\n    ap.add_argument("--quick",')]),
    ("speed", "tests/test_part3_scripts.py::test_throughput_speed_uses_full_chunks_after_the_first",
     [("scripts/part3/throughput.py", "        if not seen_full:           # the first full chunk compiles\n            seen_full = True\n            continue\n", "")]),
    # one compiled program per trainer (2 Nov 2026, after P0)
    ("dyn_count", "tests/test_part3_methods.py::test_run_dynamic_matches_scan_for_every_length",
     [("src/evo2/methods/loop.py", "lambda c: c[0] < n, body", "lambda c: c[0] <= n, body")]),
    ("dyn_logs", "tests/test_part3_methods.py::test_run_dynamic_matches_scan_for_every_length",
     [("src/evo2/methods/loop.py", "lambda x, y: x.at[i].set(y), b, log", "lambda x, y: x.at[0].set(y), b, log")]),
    ("dyn_head", "tests/test_part3_methods.py::test_trainer_chunks_of_any_length_share_one_program_and_match_scan",
     [("src/evo2/run.py", "return {k: np.asarray(jax.device_get(v))[:n] for k, v in logs.items()}",
       "return {k: np.asarray(jax.device_get(v))[:n + 1] for k, v in logs.items()}")]),
    ("dyn_static", "tests/test_part3_methods.py::test_trainer_chunks_of_any_length_share_one_program_and_match_scan",
     [("src/evo2/run.py", "return jax.jit(lambda s, n: run_dynamic(step, s, n, self.chunk_units))",
       "return jax.jit(lambda s, n: run_dynamic(step, s, n, self.chunk_units), static_argnums=1)")]),
    ("env_reset", "tests/test_part3_methods.py::test_fixed_env_as_data_plays_the_same_worlds_as_env_fn",
     [("src/evo2/methods/adapters.py", "lambda x: jnp.broadcast_to(x, (self.n_envs,) + jnp.shape(x)), env)",
       "lambda x: jnp.zeros((self.n_envs,) + jnp.shape(x), x.dtype), env)")]),
    ("env_restart", "tests/test_part3_methods.py::test_fixed_env_as_data_plays_the_same_worlds_as_env_fn",
     [("src/evo2/methods/adapters.py", "fresh_state, fresh_params = self._fresh(k_new, params)",
       "fresh_state, fresh_params = self._fresh(k_new, jax.tree_util.tree_map(lambda x: jnp.where(x == x, x, x) if x.dtype != jnp.float32 else x * 1.5, params))")]),
    ("es_ctx_episodes", "tests/test_part3_methods.py::test_fixed_env_as_data_plays_the_same_worlds_as_env_fn",
     [("src/evo2/methods/es.py", "env_params = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (e,) + jnp.shape(x)), env)",
       "env_params = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x * 2, (e,) + jnp.shape(x)), env)")]),
    ("es_ctx_runner", "tests/test_part3_methods.py::test_fixed_env_as_data_plays_the_same_worlds_as_env_fn",
     [("src/evo2/methods/es.py", "score, steps = jax.vmap(lambda s: self.score_fn(s, k_score, r.ctx))(pop)",
       "score, steps = jax.vmap(lambda s: self.score_fn(s, k_score, jax.tree_util.tree_map(lambda x: x * 2, r.ctx)))(pop)")]),
    ("trainer_cache", "tests/test_part3_methods.py::test_adaptation_trainer_is_shared_and_uses_the_env_it_is_given",
     [("src/evo2/run.py", "return _cached_trainer(method, json.dumps(mcfg, sort_keys=True), bool(adapt))",
       "return _cached_trainer(method, json.dumps(mcfg, sort_keys=True), False)")]),
    ("pilot_merge", "tests/test_part3_scripts.py::test_pilot_merge_normalizes_over_every_method_and_refuses_mismatches",
     [("scripts/part3/pilot.py", "    return add_power(merged)", "    return add_power(parts[0])")]),
    ("pilot_merge_commit", "tests/test_part3_scripts.py::test_pilot_merge_normalizes_over_every_method_and_refuses_mismatches",
     [("scripts/part3/pilot.py", 'if p["git"]["commit"] != first["git"]["commit"]:', "if False:")]),
    ("pilot_merge_cli", "tests/test_part3_scripts.py::test_pilot_merge_command_writes_the_merged_summary",
     [("scripts/part3/pilot.py", "        Path(a.out).mkdir(parents=True, exist_ok=True)\n", "")]),
    # tuning (scripts/part3/tune.py)
    ("tune_seeds", "tests/test_part3_scripts.py::test_tuning_seeds_draws_and_ranges_follow_the_preregistration",
     [("scripts/part3/tune.py", "return 4000 + 100 * METHOD_INDEX[method] + t", "return 4000 + 10 * METHOD_INDEX[method] + t")]),
    ("tune_range", "tests/test_part3_scripts.py::test_tuning_seeds_draws_and_ranges_follow_the_preregistration",
     [("scripts/part3/tune.py", "minval=np.log(lo), maxval=np.log(hi)", "minval=np.log(lo), maxval=np.log(hi) + 1.0")]),
    ("tune_override", "tests/test_part3_scripts.py::test_tuned_settings_reach_the_trainer_for_that_method_only",
     [("src/evo2/run.py", 'return {**mcfg[shared]["settings"], **mcfg.get(method, {}).get("settings", {})}',
       'return dict(mcfg[shared]["settings"])')]),
    ("tune_merge", "tests/test_part3_scripts.py::test_tuned_settings_reach_the_trainer_for_that_method_only",
     [("src/evo2/run.py", 'block["settings"] = {**block.get("settings", {}), **settings}\n        mcfg["tuned_from"]',
       'block["settings"] = dict(settings)\n        mcfg["tuned_from"]')]),
    ("tune_score_key", "tests/test_part3_scripts.py::test_tuning_score_matches_zero_shot_and_best_trial_breaks_ties_low",
     [("scripts/part3/tune.py", "k_c, _ = jax.random.split(k)                       # as in zero_shot",
       "k_c = k")]),
    ("tune_ties", "tests/test_part3_scripts.py::test_tuning_score_matches_zero_shot_and_best_trial_breaks_ties_low",
     [("scripts/part3/tune.py", 'key=lambda tr: (tr["score"], -tr["trial"])', 'key=lambda tr: (tr["score"], tr["trial"])')]),
    ("ppo_sanity", "tests/test_part3_sanity.py::test_ppo_solves_pendulum",
     [("src/evo2/methods/ppo.py", "loss_actor = masked_mean(-jnp.minimum(loss_actor1, loss_actor2), w)",
       "loss_actor = masked_mean(jnp.minimum(loss_actor1, loss_actor2), w)")]),
    ("es_sanity", "tests/test_part3_sanity.py::test_es_improves_on_pendulum",
     [("src/evo2/methods/es.py", "state, _ = self.strategy.tell(k_tell, pop, -score, state, self.es_params)",
       "state, _ = self.strategy.tell(k_tell, pop, score, state, self.es_params)")]),
    # the final job's failure paths (scripts/part3/pod/final_job.sh, scripts/part3/evaluate.py)
    ("pod_failed_kept", "tests/test_part3_pod.py::test_failed_and_crashed_runs_are_left_out_and_the_rest_evaluated",
     [("scripts/part3/pod/final_job.sh", '&& [ -f "$d/done.json" ] && [ ! -f "$d/failed.json" ]; then', '; then')]),
    ("pod_crash_stops", "tests/test_part3_pod.py::test_failed_and_crashed_runs_are_left_out_and_the_rest_evaluated",
     [("scripts/part3/pod/final_job.sh", 'log "run_failed $m $s (see job.log)"', 'log "run_failed $m $s (see job.log)"; break')]),
    ("pod_log_truncated", "tests/test_part3_pod.py::test_access_log_is_on_the_volume_and_only_appended",
     [("scripts/part3/pod/final_job.sh", 'touch "$OUT/test_access_log.jsonl"', ': > "$OUT/test_access_log.jsonl"')]),
    ("pod_log_local", "tests/test_part3_pod.py::test_access_log_is_on_the_volume_and_only_appended",
     [("scripts/part3/pod/final_job.sh", '  && ln -sfn "$OUT/test_access_log.jsonl" results/part3/test_access_log.jsonl \\\n', '')]),
    ("pod_empty", "tests/test_part3_pod.py::test_no_trained_run_means_no_evaluation_and_a_failure",
     [("scripts/part3/pod/final_job.sh", '[ ${#dirs[@]} -gt 0 ] || { log "no run trained"; exit 1; }\n', '')]),
    ("eval_final_full", "tests/test_part3_scripts.py::test_evaluate_final_refuses_runs_that_are_not_final",
     [("scripts/part3/evaluate.py", " or not a.adapt or a.adapt_budget is not None):", "):")]),
    ("tune_failed_score", "tests/test_part3_scripts.py::test_tuning_failed_trials_score_lowest_and_all_failed_stops",
     [("scripts/part3/tune.py", 'row.update({"score": float("-inf"), "envs": [], "failed": True})',
       'row.update({"score": 0.0, "envs": [], "failed": True})')]),
    ("tune_all_failed", "tests/test_part3_scripts.py::test_tuning_failed_trials_score_lowest_and_all_failed_stops",
     [("scripts/part3/tune.py", '    if chosen.get("failed"):\n', '    if False:\n')]),
    ("tuned_quick", "tests/test_part3_scripts.py::test_write_tuned_round_trip_keeps_every_method_apart",
     [("scripts/part3/tune.py", '        if s.get("quick"):\n', '        if False:\n')]),
    ("final_untuned", "tests/test_part3_scripts.py::test_final_training_needs_the_tuned_settings",
     [("scripts/part3/train.py", 'if a.purpose == "final" and "tuned_from" not in mcfg:', 'if False:')]),
    # the final analysis (src/evo2/analysis.py, scripts/part3/analyze.py)
    ('an_s1_y', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '("S1", "oe", "es_dr", "primary"', '("S1", "oe", "ppo_dr", "primary"')]),
    ('an_s4_measure', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '("S4", "oe", "ppo_dr", "zero_shot"', '("S4", "oe", "ppo_dr", "primary"')]),
    ('an_s5_measure', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '("S5", "oe", "ppo_dr", "final"', '("S5", "oe", "ppo_dr", "primary"')]),
    ('an_s6_sets', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '("S6", "oe", "ppo_dr", "primary", ("T1",), None)', '("S6", "oe", "ppo_dr", "primary", ("T2", "T3"), None)')]),
    ('an_s7_sets', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '("S7", "oe", "ppo_dr", "extinct", ("T2", "T3"), "<")', '("S7", "oe", "ppo_dr", "extinct", ("T1", "T2", "T3"), "<")')]),
    ('an_s7_iqm', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', '            sx, sy = per_run_mean(sx)[:, None], per_run_mean(sy)[:, None]\n', '')]),
    ('an_side', 'tests/test_part3_analysis.py::test_each_secondary_comparison_uses_its_own_methods_measure_and_sets',
     [('src/evo2/analysis.py', 'on_side = row["p_x_greater"] > 0.5 if row["predicted"] == ">" else row["p_x_greater"] < 0.5', 'on_side = row["p_x_greater"] > 0.5')]),
    ('an_tied', 'tests/test_part3_analysis.py::test_all_tied_comparison_reports_p_1_and_is_not_supported',
     [('src/evo2/analysis.py', '"p": 1.0 if all_tied else p,', '"p": p,')]),
    ('an_h2b_unit', 'tests/test_part3_analysis.py::test_h2b_point_values_and_whole_run_bootstrap',
     [('src/evo2/analysis.py', 'boots[b] = effects({m: metrics.iqm(metrics._resample_runs(arrays[m], rng)) for m in names})', 'boots[b] = effects({m: metrics.iqm(arrays[m][:, rng.integers(0, arrays[m].shape[1], arrays[m].shape[1])]) for m in names})')]),
    ('an_replacement', 'tests/test_part3_analysis.py::test_load_runs_replaces_failed_runs_with_the_next_seed_and_stops_past_20_percent',
     [('scripts/part3/analyze.py', '            else:\n                raise SystemExit(f"{d} is missing or unfinished" + ("" if seed in SEEDS else', '            elif seed > SEEDS[-1]:\n                pass\n            else:\n                raise SystemExit(f"{d} is missing or unfinished" + ("" if seed in SEEDS else')]),
    ('an_twenty', 'tests/test_part3_analysis.py::test_load_runs_replaces_failed_runs_with_the_next_seed_and_stops_past_20_percent',
     [('scripts/part3/analyze.py', 'if fails > 0.2 * len(SEEDS):', 'if fails > 0.3 * len(SEEDS):')]),
    ('an_random_tol', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', 'RANDOM_TOL = 1e-6 ', 'RANDOM_TOL = 1.0 ')]),
    ('an_settings', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '        if any(r["methods_config"] != rs[0]["methods_config"] for r in rs):\n', '        if False:\n')]),
    ('an_commit', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', 'if len(train_commits) != 1 or len(eval_commits) != 1:', 'if len(eval_commits) != 1:')]),
    ('an_curves_quick', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', 'if c.get("quick") or c["n_envs"]', 'if c["n_envs"]')]),
    ('an_adapt_budget', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '(steps < need[None, :]).any()', '(steps[:, -1] < 0).any()')]),
    ('an_tuned', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if "tuned_from" not in meta["methods_config"]:\n', '    if False:\n')]),
    ('an_eval_settings', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if any(x != ev[0] for x in ev):\n', '    if False:\n')]),
    ('an_pop_floor', 'tests/test_part3_analysis.py::test_analyze_runs_end_to_end_on_a_fake_final_folder',
     [('scripts/part3/analyze.py', '"floor_births_per_1000_steps": tot["floor_births"] * per_k', '"floor_births_per_1000_steps": tot["births"] * per_k')]),
    ('an_survived', 'tests/test_part3_analysis.py::test_analyze_runs_end_to_end_on_a_fake_final_folder',
     [('scripts/part3/analyze.py', 'float(np.mean([r["steps_survived"][A.env_index(s)] for r in data[m]]))', 'float(np.mean([r["steps_survived"][A.env_index("T1")] for r in data[m]]))')]),
    ('an_folder_name', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if d.name != f"{meta[\'method\']}_seed{meta[\'seed\']}":\n', '    if False:\n')]),
    ('an_eval_prereg', 'tests/test_part3_analysis.py::test_analyze_requires_the_preregistered_evaluation_settings',
     [('scripts/part3/analyze.py', '    if any(e["evaluation_settings"].get(k) != v for k, v in EVALUATION.items()):\n', '    if False:\n')]),
    ('an_progress', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if len(prog) != done["chunks"] or prog[-1]["agent_steps_total"] != done["agent_steps"]:\n', '    if False:\n')]),
    ('an_curves_git', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if not c["git"].get("clean") or c["git"]["commit"] != meta["git"]["commit"]:\n', '    if False:\n')]),
    ('an_nonfinite', 'tests/test_part3_analysis.py::test_analyze_refuses_runs_that_are_not_comparable',
     [('scripts/part3/analyze.py', '    if not np.all(np.isfinite(curve)):\n', '    if False:\n')]),
    ('an_adapt_pop', 'tests/test_part3_analysis.py::test_analyze_runs_end_to_end_on_a_fake_final_folder',
     [('scripts/part3/analyze.py', 'for k in ("births", "floor_births", "steps_all_dead_before_floor")} if oe else None)', 'for k in ("floor_births", "floor_births", "steps_all_dead_before_floor")} if oe else None)')]),
    ("an_hyp_map", 'tests/test_part3_analysis.py::test_analyze_runs_end_to_end_on_a_fake_final_folder',
     [("scripts/part3/analyze.py", '("H3", "OE > ES-DR", "S1", sec["S1"]),', '("H3", "OE > ES-DR", "S1", sec["S2"]),')]),
    # Addendum A (PREREG_part3.md Section 16): ES-fixed
    ('add_novol_create', 'tests/test_part3_control.py::test_create_without_a_network_volume',
     [('scripts/part3/pod/control.py', '    if plan.get("volume", "network") == "none":\n        body["volumeInGb"] = 0\n    else:\n        body.update(', '    if False:\n        body["volumeInGb"] = 0\n    else:\n        body.update(')]),
    ('add_novol_env', 'tests/test_part3_control.py::test_create_without_a_network_volume',
     [('scripts/part3/pod/control.py', '"VOLUME_KIND": "none" if no_volume else "network"}', '"VOLUME_KIND": "network"}')]),
    ('add_vol_body', 'tests/test_part3_control.py::test_create_with_a_network_volume_is_unchanged',
     [('scripts/part3/pod/control.py', 'body.update({"networkVolumeId": p["volume_id"], "volumeMountPath": "/workspace"})', 'body.update({"networkVolumeId": p["volume_id"]})')]),
    ('add_novol_retry', 'tests/test_part3_control.py::test_finish_without_a_volume_retries_the_last_copy_and_deletes_no_volume',
     [('scripts/part3/pod/control.py', '    tries = 1 if p.get("volume_id") else LAST_COPY_TRIES', '    tries = 1')]),
    ('add_novol_keep', 'tests/test_part3_control.py::test_finish_without_a_volume_retries_the_last_copy_and_deletes_no_volume',
     [('scripts/part3/pod/control.py', '    if not p.get("volume_id"):\n        print(', '    if False:\n        print(')]),
    ('add_margin', 'tests/test_part3_addendum.py::test_a1_margin_and_family_size_are_the_preregistered_ones',
     [('src/evo2/analysis.py', 'MARGIN = 0.05                                  # PREREG 8.1', 'MARGIN = 0.02                                  # PREREG 8.1')]),
    ('add_holm_size', 'tests/test_part3_addendum.py::test_a1_margin_and_family_size_are_the_preregistered_ones',
     [('src/evo2/analysis.py', '    for name, x, y, measure, sets, side in family:', '    for name, x, y, measure, sets, side in family[:1] + family:')]),
    ('add_readings', 'tests/test_part3_addendum.py::test_readings_and_rerun_flag',
     [('src/evo2/analysis.py', '"support (below the margin)": [small, ', '"support (below the margin)": ["varied training made ES adapt better"], "_unused": [small, ')]),
    ('add_readings_quoted', 'tests/test_part3_addendum.py::test_every_reading_is_quoted_from_the_preregistration',
     [('src/evo2/analysis.py', '"either"],', '"either, in this world"],')]),
    ('add_rerun_flag', 'tests/test_part3_addendum.py::test_readings_and_rerun_flag',
     [('src/evo2/analysis.py', 'return any(m < core_min - tol or m > core_max + tol for m in rerun_means)', 'return any(m > core_max + tol for m in rerun_means)')]),
    ('add_tune_out', 'tests/test_part3_addendum.py::test_addendum_tuning_refuses_the_core_tuning_folder',
     [('scripts/part3/tune.py', 'if not a.quick and any(m in ADDENDUM_METHODS for m in a.methods) and Path(a.out) == Path(DEFAULT_OUT):', 'if False:')]),
    ('add_e2e_verdicts', 'tests/test_part3_addendum.py::test_addendum_analysis_end_to_end_reads_es_dr_ahead',
     [('scripts/part3/analyze_addendum.py', '           "verdicts_given": verdicts,', '           "verdicts_given": False,')]),
    ('add_e2e_left_out', 'tests/test_part3_addendum.py::test_addendum_analysis_left_out_seeds',
     [('scripts/part3/analyze_addendum.py', '    verdicts = len(excluded) <= A.ADDENDUM_LEFT_OUT_MAX', '    verdicts = True')]),
    ('add_e2e_tuned', 'tests/test_part3_addendum.py::test_addendum_analysis_stops_on_wrong_settings_random_or_commits',
     [('scripts/part3/analyze_addendum.py', '    if tuned["settings"]["es_fixed"] != summary["chosen"]["settings"]:', '    if False:')]),
    ('add_e2e_random', 'tests/test_part3_addendum.py::test_addendum_analysis_stops_on_wrong_settings_random_or_commits',
     [('scripts/part3/analyze_addendum.py', '    if rand_diff > RANDOM_STOP:', '    if False:')]),
    ('add_e2e_commit', 'tests/test_part3_addendum.py::test_addendum_analysis_stops_on_wrong_settings_random_or_commits',
     [('scripts/part3/analyze_addendum.py', '    if len(one) != 1 or len({c for pair in one for c in pair}) != 1:', '    if False:')]),
    ('add_e2e_reruns', 'tests/test_part3_addendum.py::test_addendum_analysis_reruns_are_optional_and_flagged',
     [('scripts/part3/analyze_addendum.py', 'for s in CHECK_SEEDS if finished(add / f"es_dr_seed{s}")]', 'for s in CHECK_SEEDS]')]),
    ('add_pod_write', 'tests/test_part3_pod.py::test_addendum_stops_before_training_if_the_settings_are_not_written',
     [('scripts/part3/pod/addendum_job.sh', '  && [ -f "$OUT/tuned_addendum.yaml" ] || { log "FAILED: writing the tuned settings"; exit 2; }', '  || true')]),
    ('add_vol_delete', 'tests/test_part3_control.py::test_volume_delete_needs_the_pod_deleted_first',
     [('scripts/part3/pod/control.py', '        if not p.get("pod_deleted_ts"):\n            sys.exit(f"{part}: its pod is not deleted; delete the pod first")', '        if False:\n            pass')]),
    ('add_core_list', 'tests/test_part3_addendum.py::test_core_method_list_is_unchanged',
     [('src/evo2/run.py', 'METHODS = ("oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr")   # the pre-registered core', 'METHODS = ("oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr", "es_fixed")   # the pre-registered core')]),
    ('add_env', 'tests/test_part3_addendum.py::test_es_fixed_trains_in_the_canonical_environment_and_es_dr_does_not',
     [('src/evo2/run.py', 'if method in ("oe", "ppo_dr", "es_dr"):', 'if method in ("oe", "ppo_dr", "es_dr", "es_fixed"):')]),
    ('add_extra_scope', 'tests/test_part3_addendum.py::test_es_fixed_uses_es_with_its_own_tuned_settings',
     [('src/evo2/run.py', '_merge_tuned(mcfg, t, ADDENDUM_METHODS, "the addendum\'s tuned settings")', '_merge_tuned(mcfg, t, ALL_METHODS, "the addendum\'s tuned settings")')]),
    ('add_tune_seed', 'tests/test_part3_addendum.py::test_es_fixed_tuning_seeds_space_and_starting_settings',
     [('scripts/part3/tune.py', '"es_dr": 4, "es_fixed": 5}', '"es_dr": 4, "es_fixed": 4}')]),
    ('add_tune_start', 'tests/test_part3_addendum.py::test_es_fixed_tuning_seeds_space_and_starting_settings',
     [('scripts/part3/tune.py', '    return load_methods_config(tuned=None)\n\n\ndef main():', '    return load_methods_config()\n\n\ndef main():')]),
    ('add_tune_mixed', 'tests/test_part3_addendum.py::test_es_fixed_tuning_seeds_space_and_starting_settings',
     [('scripts/part3/tune.py', '        if any(m in ADDENDUM_METHODS for m in methods):\n            raise SystemExit("tune core and addendum methods separately")\n', '')]),
    ('add_write_count', 'tests/test_part3_addendum.py::test_write_addendum_round_trip_and_refusals',
     [('scripts/part3/tune.py', 'if s["method"] != m or len(s["trials"]) != TRIALS:', 'if False:')]),
    ('add_train_needs', 'tests/test_part3_addendum.py::test_train_cli_needs_tuned_settings_for_es_fixed_and_refuses_them_for_core',
     [('scripts/part3/train.py', 'if addendum and not a.tuned_extra and not a.quick:', 'if False:')]),
    ('add_train_core', 'tests/test_part3_addendum.py::test_train_cli_needs_tuned_settings_for_es_fixed_and_refuses_them_for_core',
     [('scripts/part3/train.py', 'if a.tuned_extra and not addendum:', 'if False:')]),
    ('add_a1_order', 'tests/test_part3_addendum.py::test_a1_and_family_read_es_dr_above_es_fixed_as_support',
     [('src/evo2/analysis.py', 'metrics.iqm_difference_ci(primary["es_dr"], primary["es_fixed"], reps, seed)', 'metrics.iqm_difference_ci(primary["es_fixed"], primary["es_dr"], reps, seed)')]),
    ('add_a3_sets', 'tests/test_part3_addendum.py::test_a1_and_family_read_es_dr_above_es_fixed_as_support',
     [('src/evo2/analysis.py', '("A3", "es_dr", "es_fixed", "zero_shot", ("T1",), ">")', '("A3", "es_dr", "es_fixed", "zero_shot", ("T2", "T3"), ">")')]),
    ('add_a3_measure', 'tests/test_part3_addendum.py::test_a1_and_family_read_es_dr_above_es_fixed_as_support',
     [('src/evo2/analysis.py', '("A3", "es_dr", "es_fixed", "zero_shot", ("T1",), ">")', '("A3", "es_dr", "es_fixed", "primary", ("T1",), ">")')]),
    ('add_a4_sign', 'tests/test_part3_addendum.py::test_a4_interaction_sign_and_reading',
     [('src/evo2/analysis.py', '        return np.array([es, oe, es - oe])', '        return np.array([es, oe, oe - es])')]),
    ('add_final_extra', 'tests/test_part3_pod.py::test_final_job_passes_the_addendum_settings_to_es_fixed_only',
     [('scripts/part3/pod/final_job.sh', 'if [ "$m" = "es_fixed" ]; then', 'if [ "$m" = "es_dr" ]; then')]),
    ('add_final_noextra', 'tests/test_part3_pod.py::test_final_job_leaves_out_es_fixed_without_the_addendum_settings',
     [('scripts/part3/pod/final_job.sh', '[ -n "${TUNED_EXTRA:-}" ] || { log "run_failed $m $s (no TUNED_EXTRA)"; continue; }', 'true')]),
    ('add_job_order', 'tests/test_part3_pod.py::test_addendum_tunes_writes_settings_then_runs_the_final_job',
     [('scripts/part3/pod/addendum_job.sh', 'TUNED_EXTRA="$OUT/tuned_addendum.yaml" \\', 'TUNED_EXTRA="" \\')]),
    ('add_job_stop', 'tests/test_part3_pod.py::test_addendum_stops_before_training_if_tuning_fails',
     [('scripts/part3/pod/addendum_job.sh', '|| { rc=$?; log "tuning failed (exit $rc)"; exit "$rc"; }', '|| true')]),
    ('add_check_part', 'tests/test_part3_control.py::test_check_part_for_an_addendum_needs_evaluation_tuning_and_the_access_log',
     [('scripts/part3/pod/control.py', '    if job == "addendum":\n', '    if False:\n')]),
    ('add_check_eval', 'tests/test_part3_control.py::test_check_part_for_an_addendum_needs_evaluation_tuning_and_the_access_log',
     [('scripts/part3/pod/control.py', '    evaluated = job in ("final", "addendum")', '    evaluated = False')]),
]


def copy_repo(dst: Path):
    for name in ("src", "configs/part3", "tests", "scripts/part3", "pyproject.toml", "docs/PREREG_part3.md"):
        s, d = ROOT / name, dst / name
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy2(s, d)
    # the run code checks for a clean, committed tree, so the copy is its own git repo
    git = ["git", "-c", "user.name=break-check", "-c", "user.email=break-check@localhost"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "copy"]):
        subprocess.run(git + cmd, cwd=dst, check=True, capture_output=True)


def apply(dst: Path, edits):
    for f, old, new in edits:
        p = dst / f
        if old is None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new)
            continue
        text = p.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f"break text not found exactly once in {f}: {old[:60]!r}")
        p.write_text(text.replace(old, new))


def run_test(dst: Path, node: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(dst / "src")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", node],
                          cwd=dst, env=env, capture_output=True, text=True, timeout=900)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "results/part3/break_tests.json"))
    a = ap.parse_args()
    chosen = [b for b in BREAKS if not a.only or b[0] in a.only]
    results = []
    for name, node, edits in chosen:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            copy_repo(dst)
            ok = run_test(dst, node)          # the unbroken copy must pass
            try:
                apply(dst, edits)
            except RuntimeError as e:         # a break that no longer matches the code counts as a failure
                row = {"name": name, "test": node, "passes_unbroken": ok.returncode == 0,
                       "fails_broken": False, "error": str(e)}
                results.append(row)
                print(json.dumps(row), flush=True)
                continue
            subprocess.run(["git", "-c", "user.name=b", "-c", "user.email=b@localhost", "commit", "-q",
                            "-am", "break"], cwd=dst, capture_output=True)
            subprocess.run(["git", "-c", "user.name=b", "-c", "user.email=b@localhost", "add", "-A"],
                           cwd=dst, capture_output=True)
            subprocess.run(["git", "-c", "user.name=b", "-c", "user.email=b@localhost", "commit", "-q",
                            "-m", "break2"], cwd=dst, capture_output=True)
            bad = run_test(dst, node)
        row = {"name": name, "test": node, "passes_unbroken": ok.returncode == 0,
               "fails_broken": bad.returncode != 0}
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n")
    bad_rows = [r for r in results if not (r["passes_unbroken"] and r["fails_broken"])]
    covered = {r["test"] for r in results}
    print(f"{len(results) - len(bad_rows)}/{len(results)} breaks behave as required; "
          f"{len(covered)} tests covered")
    sys.exit(1 if bad_rows else 0)


if __name__ == "__main__":
    main()
