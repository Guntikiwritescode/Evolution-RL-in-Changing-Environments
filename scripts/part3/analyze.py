"""Part 3 final analysis (PREREG_part3.md Sections 7-8): every number and figure from saved results.

Reads the final run folders (<final>/<method>_seed<seed>/ with run.json, done.json, eval_test.json,
curves.json, progress.jsonl, snapshots/, policy.pkl), then writes
  <final>/analysis.json        every number below
  figures/part3/*.png          the figures the plan lists (core only; Baldwin plots belong to Q3)
  docs/RESULTS_part3.md        the generated results report (check-in 10)

  venv3/bin/python scripts/part3/analyze.py --final results/part3/final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from evo2 import analysis as A
from evo2 import metrics
from evo2.agents.brain import apply_mlp
from evo2.env.splits import load_validation, load_world_config
from evo2.env.world import World
from evo2.eval.protocol import _sample
from evo2.run import METHODS
from evo2.runio import ROOT, load_pickle, write_json
from evo2.seeds import base_key, stream_key

NAMES = {"oe": "OE", "oe_fixed": "OE-fixed", "ppo_fixed": "PPO-fixed", "ppo_dr": "PPO-DR", "es_dr": "ES-DR"}
SEEDS = list(range(5000, 5010))          # PREREG Section 6 (check-in 8): 10 runs per method
TRAIN_BUDGET = 200_000_000               # PREREG Section 4 (check-in 8)
ADAPT_BUDGET = 20_000_000
N_POINTS = 11
N_VALIDATION = 24                        # training curves: 24 validation environments, 8 episodes (PREREG 8.4)
EPISODES = 8
EVALUATION = {"episodes_per_env": 8, "oe_sample": 16, "persistence_agents": 32, "persistence_steps": 5000}   # PREREG 7.1, 7.3
RANDOM_TOL = 1e-6                        # largest allowed difference in the random policy's competence between runs
N_DIVERSITY = 16                         # PREREG 7.5
N_PROBES = 256
PROBE_SEED = 4801                        # PREREG 7.5 ("stats" range), "probe" stream


# ---------- loading ----------

def load_runs(final: Path) -> dict:
    """{method: [run folders]} for the planned seeds, applying the exclusion rule (PREREG 9): a run
    that failed technically (failed.json) is reported and replaced by the next unused final seed of
    that method (5010, 5011, ...). A planned or replacement run that is missing or unfinished stops
    the analysis, and so do more than 20% failed runs for a method."""
    runs, excluded = {}, []
    for m in METHODS:
        got, seed, fails = [], SEEDS[0], 0
        while len(got) < len(SEEDS):
            d = final / f"{m}_seed{seed}"
            if (d / "failed.json").exists():
                excluded.append({"method": m, "seed": seed,
                                 "reason": json.loads((d / "failed.json").read_text()).get("reason")})
                fails += 1
                if fails > 0.2 * len(SEEDS):
                    raise SystemExit(f"{m}: {fails} failed runs, more than 20%: stop and ask AA (PREREG 9)")
            elif (d / "done.json").exists():
                got.append(d)
            else:
                raise SystemExit(f"{d} is missing or unfinished" + ("" if seed in SEEDS else
                                 " (it is the next unused final seed, needed to replace a failed run: PREREG 9)"))
            seed += 1
        runs[m] = got
    return {"runs": runs, "excluded": excluded}


def read_eval(d: Path) -> dict:
    """One final run's records, checked to be a full final run and a full final evaluation."""
    e = json.loads((d / "eval_test.json").read_text())
    meta = json.loads((d / "run.json").read_text())
    done = json.loads((d / "done.json").read_text())
    if meta["purpose"] != "final" or meta.get("quick") or e["split"] != "test" or len(e["envs"]) != 30:
        raise SystemExit(f"{d}: not a full final evaluation")
    if not (done.get("reached_budget") and done["budget_agent_steps"] == TRAIN_BUDGET and done["agent_steps"] >= TRAIN_BUDGET):
        raise SystemExit(f"{d}: training did not reach the pre-registered budget")
    if not (meta["git"].get("clean") and e["git"].get("clean")):
        raise SystemExit(f"{d}: trained or evaluated from a tree with uncommitted changes")
    if "tuned_from" not in meta["methods_config"]:
        raise SystemExit(f"{d}: trained without the tuned settings")
    if d.name != f"{meta['method']}_seed{meta['seed']}":
        raise SystemExit(f"{d}: the folder name does not match its run ({meta['method']}, seed {meta['seed']})")
    if any(e["evaluation_settings"].get(k) != v for k, v in EVALUATION.items()):
        raise SystemExit(f"{d}: evaluation settings differ from the pre-registered ones: {e['evaluation_settings']}")
    prog = [json.loads(x) for x in (d / "progress.jsonl").read_text().splitlines() if x.strip()]
    if len(prog) != done["chunks"] or prog[-1]["agent_steps_total"] != done["agent_steps"]:
        raise SystemExit(f"{d}: progress.jsonl is incomplete ({len(prog)} of {done['chunks']} chunks)")
    names = [(r["set"], r["index"]) for r in e["envs"]]
    want = [(s, i) for s in ("T1", "T2", "T3") for i in range(10)]
    if names != want:
        raise SystemExit(f"{d}: environments out of order")
    curve = np.array([[p["competence"] for p in r["adaptation"]["curve"]] for r in e["envs"]])
    steps = np.array([[p["agent_steps"] for p in r["adaptation"]["curve"]] for r in e["envs"]])
    if curve.shape != (30, N_POINTS):
        raise SystemExit(f"{d}: adaptation curves are not 11 points")
    if not np.all(np.isfinite(curve)):
        raise SystemExit(f"{d}: a competence is not finite; stop and ask AA (not covered by PREREG 9)")
    need = np.arange(N_POINTS) * ADAPT_BUDGET // (N_POINTS - 1)
    if int(meta["methods_config"]["budget"]["adapt_agent_steps"]) != ADAPT_BUDGET or (steps < need[None, :]).any():
        raise SystemExit(f"{d}: adaptation did not use the pre-registered budget at every checkpoint")
    c = json.loads((d / "curves.json").read_text())
    if c.get("quick") or c["n_envs"] != N_VALIDATION or c["episodes_per_env"] != EPISODES or len(c["points"]) != 10:
        raise SystemExit(f"{d}: the training curve is not the pre-registered one (PREREG 8.4)")
    if not c["git"].get("clean") or c["git"]["commit"] != meta["git"]["commit"]:
        raise SystemExit(f"{d}: the training curve was not computed from the run's own clean commit")
    oe = meta["method"] in ("oe", "oe_fixed")
    adapt_pop = ({k: np.array([sum(ch[k] for ch in r["adaptation"]["chunks"]) for r in e["envs"]], float)
                  for k in ("births", "floor_births", "steps_all_dead_before_floor")} if oe else None)
    pers = [r["zero_shot"]["persistence"] for r in e["envs"]]
    return {"curve": curve, "adapt_steps": steps, "random": np.array([r["zero_shot"]["random_competence"] for r in e["envs"]]),
            "extinct": np.array([float(p["extinct"]) for p in pers]),
            "steps_survived": np.array([float(p["steps_survived"]) for p in pers]),
            "persistence_births": np.array([float(p["births"]) for p in pers]),
            "zero_shot": np.array([r["zero_shot"]["competence"] for r in e["envs"]]),
            "adapt_population": adapt_pop,
            "adapt_wall_seconds": float(sum(r["adaptation"]["wall_seconds"] for r in e["envs"])),
            "commit": e["git"]["commit"], "train_commit": meta["git"]["commit"],
            "hardware": e["hardware"], "train_hardware": meta["hardware"],
            "methods_config": meta["methods_config"], "evaluation_settings": e["evaluation_settings"],
            "curves_random": [x["random_competence"] for x in c["points"][0]["envs"]],
            "done": done, "seed": meta["seed"], "method": meta["method"]}


def training_population(d: Path) -> dict:
    """OE and OE-fixed during training (PREREG 9, 11): births, floor births and steps on which the
    whole population had died before the floor refilled it, from the chunk summaries."""
    rows = [json.loads(x) for x in (d / "progress.jsonl").read_text().splitlines() if x.strip()]
    tot = {k: int(sum(r[k] for r in rows)) for k in ("steps", "births", "floor_births", "deaths", "steps_all_dead_before_floor")}
    per_k = 1000.0 / tot["steps"]
    return {**tot, "births_per_1000_steps": tot["births"] * per_k, "floor_births_per_1000_steps": tot["floor_births"] * per_k,
            "floor_share_of_new_agents": tot["floor_births"] / max(1, tot["births"] + tot["floor_births"])}


def check_comparable(data: dict) -> dict:
    """Stop if the runs being pooled differ in anything but their seed: one training commit, one
    evaluation commit, the same settings within each method, the same evaluation settings, and the
    same random-policy competence (same episodes) on the test and validation environments."""
    train_commits = {r["train_commit"] for rs in data.values() for r in rs}
    eval_commits = {r["commit"] for rs in data.values() for r in rs}
    if len(train_commits) != 1 or len(eval_commits) != 1:
        raise SystemExit(f"runs come from more than one commit: training {sorted(train_commits)}, evaluation {sorted(eval_commits)}")
    for m, rs in data.items():
        if any(r["methods_config"] != rs[0]["methods_config"] for r in rs):
            raise SystemExit(f"{m}: runs were trained with different settings")
    ev = [r["evaluation_settings"] for rs in data.values() for r in rs]
    if any(x != ev[0] for x in ev):
        raise SystemExit("runs were evaluated with different evaluation settings")
    rnd = np.stack([r["random"] for rs in data.values() for r in rs])
    val = np.array([r["curves_random"] for rs in data.values() for r in rs], float)
    diff_test, diff_val = float(np.max(np.abs(rnd - rnd[0]))), float(np.max(np.abs(val - val[0])))
    if diff_test > RANDOM_TOL or diff_val > RANDOM_TOL:
        raise SystemExit(f"the random policy's competence differs between runs (test {diff_test:.3g}, validation "
                         f"{diff_val:.3g}): the episodes are not the same; stop and ask AA")
    return {"train_commit": train_commits.pop(), "eval_commit": eval_commits.pop(), "random": rnd[0],
            "random_max_difference_test": diff_test, "random_max_difference_validation": diff_val}


# ---------- diversity (OE and OE-fixed only; PPO and ES deploy one policy) ----------

def probe_observations() -> np.ndarray:
    """256 observations drawn once from validation episodes (PREREG 7.5): the random policy plays
    one episode in each of the 24 validation environments (keys from seed 4801's "probe" stream,
    one per environment); observations of living agents are pooled and 256 drawn without replacement."""
    world = World(load_world_config("episodic"))
    v = load_validation()
    envs = v["drawn"] + v["corners"]
    key = stream_key(base_key("stats", PROBE_SEED), "probe")
    edge = 1.0 - 1e-6
    traits = jnp.tile(world.default_traits, (world.cfg.n_slots, 1))

    @jax.jit
    def episode(k, env):                       # one compiled program for every environment
        k_reset, k_steps = jax.random.split(k)
        s = world.reset(k_reset, env, traits, world.cfg.n_slots)

        def body(s, kk):
            obs = world.observe(s, env)
            k_a, k_w = jax.random.split(kk)
            u = jax.random.uniform(k_a, (world.cfg.n_slots, 2), minval=-edge, maxval=edge)
            s2, _ = world.step(k_w, s, env, jnp.arctanh(u))
            return s2, (obs, s.alive)

        _, out = jax.lax.scan(body, s, jax.random.split(k_steps, world.cfg.episode_len))
        return out

    pool = []
    for j, env in enumerate(envs):
        obs, alive = episode(jax.random.fold_in(key, j), env)
        o, a = np.asarray(obs).reshape(-1, obs.shape[-1]), np.asarray(alive).reshape(-1)
        pool.append(o[a])
    pool = np.concatenate(pool)
    idx = np.asarray(jax.random.choice(jax.random.fold_in(key, 999), pool.shape[0], (N_PROBES,), replace=False))
    return pool[idx]


def population_diversity(policy: dict, probes: np.ndarray, key) -> dict:
    """Genotypic (weights, traits) and behavioral diversity of 16 agents sampled from the living
    population (without replacement when at least 16 are alive). Actions are the brain outputs
    squashed by tanh, before scaling by the (default) body."""
    g, n_alive = _sample(key, policy["genomes"], policy["alive"], N_DIVERSITY)
    flat = np.concatenate([np.asarray(x).reshape(N_DIVERSITY, -1) for x in jax.tree_util.tree_leaves(g.brain)], axis=1)
    acts = np.stack([np.tanh(np.asarray(apply_mlp(jax.tree_util.tree_map(lambda x: x[i], g.brain), jnp.asarray(probes))))
                     for i in range(N_DIVERSITY)])
    return {"weights": A.mean_pairwise_distance(flat), "traits": A.mean_pairwise_distance(np.asarray(g.traits)),
            "behavior": A.behavioral_distance(acts), "n_alive": int(n_alive)}


# ---------- figures ----------

def figures(out_dir: Path, res: dict, data: dict, norm: dict, curves_tr: dict, div_time: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name, what):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        made.append({"file": f"figures/part3/{name}", "shows": what})

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in METHODS:
        ys = np.array([[p["score"] for p in c["points"]] for c in curves_tr[m]])
        ax.errorbar(np.arange(1, 11) / 10, ys.mean(0), yerr=ys.std(0, ddof=1), label=NAMES[m], capsize=2)
    ax.set_xlabel("fraction of training budget"); ax.set_ylabel("validation score (competence − random)")
    ax.legend(); ax.set_title("Learning curves (mean ± SD over runs)")
    save(fig, "learning_curves.png", "validation score of each training snapshot, mean and SD over the 10 runs per method")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, s in zip(axes, ("T1", "T2", "T3")):
        envs = A.env_index(s)
        for m in METHODS:
            ax.plot(A.X_POINTS, np.nanmean(norm[m][:, envs, :], axis=(0, 1)), label=NAMES[m])
        ax.set_title(s); ax.set_xlabel("fraction of adaptation budget")
    axes[0].set_ylabel("normalized competence"); axes[0].legend(fontsize=8)
    save(fig, "adaptation_curves.png", "normalized competence during adaptation, mean over runs and environments, per test set")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, m in enumerate(METHODS):
        v, lo, hi = res["iqm_primary"][m]
        ax.errorbar([v], [i], xerr=[[v - lo], [hi - v]], fmt="o", capsize=3)
    ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS]); ax.set_xlabel("IQM of the primary metric (T2, T3)")
    ax.set_title("IQM with 95% whole-run bootstrap CI")
    save(fig, "iqm_primary.png", "IQM of the primary metric per method with its 95% whole-run bootstrap CI")

    taus = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        ax.plot(taus, metrics.performance_profile(A.select(norm["_primary"][m], A.env_index("T2", "T3")), taus), label=NAMES[m])
    ax.set_xlabel("primary metric threshold τ"); ax.set_ylabel("share of (run, environment) scores > τ"); ax.legend()
    save(fig, "performance_profiles.png", "performance profiles of the primary metric on T2 and T3")

    pm = np.array(res["prob_improvement_matrix"]["values"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pm, vmin=0, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(METHODS)), [NAMES[m] for m in METHODS], rotation=30)
    ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            if i != j:
                ax.text(j, i, f"{pm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("P(row > column), per-run primary means")
    save(fig, "prob_improvement.png", "probability that a row method's run beats a column method's run on the primary metric (per-run means)")

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    w = 0.35
    vals = {m: res["iqm_primary"][m][0] for m in ("oe", "oe_fixed", "ppo_dr", "ppo_fixed")}
    ax.bar(x - w / 2, [vals["oe"], vals["ppo_dr"]], w, label="variable training")
    ax.bar(x + w / 2, [vals["oe_fixed"], vals["ppo_fixed"]], w, label="fixed training")
    ax.set_xticks(x, ["evolution", "PPO"]); ax.set_ylabel("IQM primary metric"); ax.legend()
    ax.set_title("Variety vs method (2 × 2)")
    save(fig, "ablation_variety.png", "the 2 × 2 of evolution/PPO × variable/fixed training on the primary metric's IQM")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ("oe", "oe_fixed"):
        for ax, key in zip(axes, ("weights", "behavior")):
            ys = np.array([[d[key] for d in run] for run in div_time[m]])
            ax.errorbar(np.arange(1, 11) / 10, ys.mean(0), yerr=ys.std(0, ddof=1), label=NAMES[m], capsize=2)
            ax.set_title(f"{key} diversity"); ax.set_xlabel("fraction of training budget")
    axes[0].legend()
    save(fig, "diversity_over_time.png", "OE and OE-fixed genotypic (weights) and behavioral diversity of 16 sampled agents at each snapshot")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    hrs = [res["compute"][m]["train_wall_hours_mean"] for m in METHODS]
    gpus = sorted({g for m in METHODS for g in res["compute"][m]["train_hardware"]})
    ax.bar([NAMES[m] for m in METHODS], hrs); ax.set_ylabel("hours per training run")
    ax.set_xlabel("training hardware: " + "; ".join(gpus), fontsize=7)
    ax.set_title("Compute: same agent-steps, different wall time")
    save(fig, "compute.png", "mean wall time per training run (all methods used the same agent-step budget)")
    return made


# ---------- report ----------

def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def write_report(res: dict, path: Path):
    out = []
    w = out.append
    w("# Part 3 main experiment: results (check-in 10)\n")
    w("Generated by `scripts/part3/analyze.py` from the final runs in `results/part3/final/` (numbers in "
      "`results/part3/final/analysis.json`). Pre-registration: `docs/PREREG_part3.md`. Code commits of the runs and "
      f"evaluations: training {res['train_commit'][:7]}, evaluation {res['eval_commit'][:7]} (every run; clean trees). "
      f"Bootstrap: {res['bootstrap_reps']:,} whole-run resamples, seed {A.STATS_SEED}. The random policy's competence was "
      f"the same in every run's evaluation (largest difference: test {res['random_max_difference']['test']:.3g}, "
      f"validation {res['random_max_difference']['validation']:.3g}).\n")
    w("Runs per method (final seeds): " + "; ".join(f"{NAMES[m]} {len(v)} (seeds {', '.join(map(str, v))})" for m, v in res["runs"].items()) + ".")
    w(f"Excluded runs (technical failures, PREREG 9): {res['excluded_runs'] or 'none'}. "
      f"Test environments left out of normalized measures (best ≤ random, PREREG 7.2): {res['environments_left_out'] or 'none'}.\n")
    h = res["h1"]
    sec = {r["name"]: r["result"] for r in res["secondary"]}
    w("## Hypotheses (PREREG Section 1) and their pre-set verdicts\n")
    w("**Historical hypothesis wording:** the original H1 phrase 'adapts faster' is retained below for provenance. "
      "The measured endpoint is performance during adaptation, including starting competence, and does not isolate "
      "adaptation speed. Current interpretations and uncertainty qualifications are in SCIENTIFIC_REVIEW_REPAIRS.md.\n")
    w("| Hypothesis | Prediction | Test | Verdict |")
    w("|---|---|---|---|")
    for hyp, pred, test, verdict in [
            ("H1", "OE adapts faster than PPO-DR (T2, T3)", "8.1", h["verdict"]),
            ("H2-OE", "OE > OE-fixed", "S2", sec["S2"]), ("H2-PPO", "PPO-DR > PPO-fixed", "S3", sec["S3"]),
            ("H2b", "variety vs method (no direction)", "8.3", res["h2b"]["verdict"]),
            ("H3", "OE > ES-DR", "S1", sec["S1"]),
            ("H4", "zero-shot: no direction; final: OE > PPO-DR", "S4, S5", f"S4 {sec['S4']}; S5 {sec['S5']}"),
            ("H5", "T1 (in distribution): no direction", "S6", sec["S6"]),
            ("H6", "OE goes extinct less often than PPO-DR", "S7", sec["S7"])]:
        w(f"| {hyp} | {pred} | {test} | {verdict} |")
    w("")
    w("## Primary result (H1: OE vs PPO-DR, adaptation on T2 and T3)\n")
    w(f"**Verdict: {h['verdict']}.** D = IQM(OE) − IQM(PPO-DR) = {f3(h['D'])}, 95% CI [{f3(h['ci'][0])}, {f3(h['ci'][1])}] "
      f"(margin {h['margin']}). P(OE > PPO-DR) on per-run means: {f3(h['p_oe_greater'])}. With 10 runs per method the "
      "verdict \"no meaningful difference\" was practically unreachable (PREREG 6, check-in 8).\n")
    w("| Method | IQM of the primary metric | 95% CI |")
    w("|---|---|---|")
    for m in METHODS:
        v, lo, hi = res["iqm_primary"][m]
        w(f"| {NAMES[m]} | {f3(v)} | [{f3(lo)}, {f3(hi)}] |")
    w("")
    w("## Secondary comparisons (one Holm family, α = 0.05)\n")
    w("Two-sided Mann-Whitney U on per-run scores; P(X > Y) = U ÷ (n_X n_Y); IQM difference with its whole-run bootstrap CI.\n")
    w("| | X vs Y | Measure | Sets | P(X > Y) | p | p (Holm) | IQM difference [CI] | Predicted | Result |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["secondary"]:
        w(f"| {r['name']} | {NAMES[r['x']]} vs {NAMES[r['y']]} | {r['measure']} | {', '.join(r['sets'])} | "
          f"{f3(r['p_x_greater'])} | {'1 (all tied)' if r.get('all_tied') else format(r['p'], '.4g')} | {r['p_holm']:.4g} | {f3(r['iqm_difference'])} "
          f"[{f3(r['ci'][0])}, {f3(r['ci'][1])}] | {'X > Y' if r['predicted'] == '>' else 'X < Y' if r['predicted'] == '<' else 'none'} | {r['result']} |")
    w("\nS7's IQM difference is taken over the per-run extinction fractions (one value per run). \"All tied\" means "
      "every per-run score was equal, so the test cannot reject (p = 1).\n")
    b = res["h2b"]
    w("## Variety vs method (H2b, 2 × 2 on the primary metric's IQMs)\n")
    for k in ("variety", "method", "interaction", "delta"):
        w(f"- {k}: {f3(b[k]['value'])} [{f3(b[k]['ci'][0])}, {f3(b[k]['ci'][1])}]")
    w(f"\n**Verdict: {b['verdict']}**" + (" (the interaction's CI excludes 0, so this verdict carries that caveat)" if b["interaction_excludes_0"] else "")
      + ". Δ = |variety| − |method|; its CI is unreliable when either effect is near 0, so both are shown.\n")
    w("## Descriptive results (not tested)\n")
    w("| Method | primary T1 | primary T2 | primary T3 | zero-shot norm. (T2, T3) | final norm. (T2, T3) | reached 90% of pre-shift (T2, T3) | median time if reached | extinction T1 / T2 / T3 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        d = res["descriptive"][m]
        t = d["time_to_90_T2_T3"]
        e = d["extinction_rate_by_set"]
        w(f"| {NAMES[m]} | {f3(d['primary_mean_by_set']['T1'])} | {f3(d['primary_mean_by_set']['T2'])} | {f3(d['primary_mean_by_set']['T3'])} | "
          f"{f3(d['zero_shot_normalized_mean_T2_T3'])} | {f3(d['final_normalized_mean_T2_T3'])} | {t['reached']} of {t['pairs']} | "
          f"{f3(t['median_time_if_reached'])} | {e['T1']:.2f} / {e['T2']:.2f} / {e['T3']:.2f} |")
    w("\nTime to 90% compares raw competence with the run's own mean T1 zero-shot competence, so it depends on how "
      "much food each test environment holds. Environments where it was reached (number of runs): " + "; ".join(
        f"{NAMES[m]} " + ", ".join(f"{('T2', 'T3')[(j - 10) // 10]}-{(j - 10) % 10} ({n})" for j, n in
                                   d["time_to_90_T2_T3"]["runs_reaching_by_env"].items())
        for m, d in res["descriptive"].items()) + ".\n")
    w("Means over runs and environments (the tested measures above use IQMs and per-run scores). Time to 90% "
      "is counted on T2 and T3. Normalization uses the random policy's competence on the zero-shot episodes; "
      "adaptation points 1–10 are measured on other episodes of the same environment (the same for every method "
      "and run, PREREG 7.3).\n")
    w("**Persistence, steps survived** (mean over runs and environments, of 5,000): " + "; ".join(
        f"{NAMES[m]} T1 {d['steps_survived_mean_by_set']['T1']:.0f}, T2 {d['steps_survived_mean_by_set']['T2']:.0f}, "
        f"T3 {d['steps_survived_mean_by_set']['T3']:.0f} (births during the test: {d['persistence_births_mean_by_set']['T1']:.0f} / "
        f"{d['persistence_births_mean_by_set']['T2']:.0f} / {d['persistence_births_mean_by_set']['T3']:.0f})"
        for m, d in res["descriptive"].items()) + ". Raw zero-shot competence per environment (mean over runs) and the "
      "random policy's are in `analysis.json`.\n")
    w("**OE populations** (PREREG 9: never excluded for low population; floor births reported). Training, mean "
      "over runs:\n")
    w("| Method | births per 1,000 steps | floor births per 1,000 steps | floor share of new agents | steps with every agent dead before the floor refilled |")
    w("|---|---|---|---|---|")
    for m, v in res["oe_population"].items():
        t = v["training_mean"]
        w(f"| {NAMES[m]} | {t['births_per_1000_steps']:.1f} | {t['floor_births_per_1000_steps']:.1f} | "
          f"{t['floor_share_of_new_agents']:.2f} | {t['steps_all_dead_before_floor']:.0f} |")
    w("\nDuring adaptation (per test environment, mean over runs; T1 / T2 / T3):\n")
    w("| Method | births | floor births | steps with every agent dead before the floor refilled |")
    w("|---|---|---|---|")
    for m, v in res["oe_population"].items():
        a = v["adaptation_mean_per_env_by_set"]
        w(f"| {NAMES[m]} | " + " | ".join(" / ".join(f"{a[k][s]:.0f}" for s in ("T1", "T2", "T3"))
                                          for k in ("births", "floor_births", "steps_all_dead_before_floor")) + " |")
    w("")
    w("**Learning curves** (validation score at each tenth of training, mean over runs):\n")
    w("| Method | " + " | ".join(f"{k}/10" for k in range(1, 11)) + " |")
    w("|---|" + "---|" * 10)
    for m in METHODS:
        w(f"| {NAMES[m]} | " + " | ".join(f"{v:.2f}" for v in res["learning_curves"][m]["mean"]) + " |")
    w("\n**Diversity at the end of training** (16 sampled agents; mean over runs): " + "; ".join(
        f"{NAMES[m]}: weights {f3(v['weights'])}, traits {f3(v['traits'])}, behavior {f3(v['behavior'])}" for m, v in res["diversity_end"].items())
      + f". {res['diversity_note']}. Every agent has the default body (AA's check-in 7 decision: OE mutates brains only), "
      "so trait diversity is 0 by design.\n")
    w("**Diversity over training** (OE snapshots, mean over runs, tenths 1–10):\n")
    w("| Method | measure | " + " | ".join(f"{k}/10" for k in range(1, 11)) + " |")
    w("|---|---|" + "---|" * 10)
    for m, v in res["diversity_over_time"].items():
        for kk in ("weights", "traits", "behavior"):
            w(f"| {NAMES[m]} | {kk} | " + " | ".join(f"{x:.3f}" for x in v[kk]) + " |")
    w("\nProbe observations for behavioral diversity come from the random policy playing one episode in each "
      "validation environment (seed 4801, \"probe\" stream).\n")
    w("**Compute** (per run, mean): " + "; ".join(
        f"{NAMES[m]} {min(v['train_agent_steps']):,}–{max(v['train_agent_steps']):,} training agent-steps, "
        f"training {v['train_wall_hours_mean']:.2f} h, adaptation (30 environments) {v['adapt_wall_hours_mean']:.2f} h"
        for m, v in res["compute"].items()) + ". Training hardware: " + "; ".join(
        sorted({g for v in res["compute"].values() for g in v["train_hardware"]})) + "; evaluation hardware: " + "; ".join(
        sorted({g for v in res["compute"].values() for g in v["eval_hardware"]})) + "; software: " + "; ".join(
        sorted({x for v in res["compute"].values() for x in v["software"]})) + ". Adaptation used on average "
      + ", ".join(f"{NAMES[m]} {v['adapt_final_agent_steps_mean']:,.0f}" for m, v in res["compute"].items())
      + " agent-steps per environment (budget 20,000,000).\n")
    w("**Adaptation curves** (normalized competence, mean over runs and environments, points 0–10):\n")
    w("| Method | set | " + " | ".join(str(k) for k in range(11)) + " |")
    w("|---|---|" + "---|" * 11)
    for m, by in res["adaptation_curves"].items():
        for st, v in by.items():
            w(f"| {NAMES[m]} | {st} | " + " | ".join(f3(x) for x in v) + " |")
    w("")
    c = res["cost"]
    w("**Cost**: " + (f"${c['total']:.2f} for the final stage ({c['machine_hours']:.1f} machine-hours at "
                      f"${c['price_per_hour']:.2f}/hour, plus storage); details in `results/part3/final/spend.json`."
                      if c else "not recorded in `results/part3/final/spend.json`.") + "\n")
    w("**Per run** (training wall time, adaptation wall time over the 30 environments):\n")
    w("| Method | seed | training agent-steps | training h | adaptation h |")
    w("|---|---|---|---|---|")
    for m, rows in res["per_run"].items():
        for r in rows:
            w(f"| {NAMES[m]} | {r['seed']} | {r['train_agent_steps']:,} | {r['train_wall_hours']:.2f} | {r['adapt_wall_hours']:.2f} |")
    log = res["test_access_log"]
    w("\n**Looks at the test split** (`results/part3/final/test_access_log.jsonl`, PREREG 3): " + (
        "; ".join(f"{x['time']} ({len(x['runs'])} runs, commit {x['commit'][:7]}, adaptation {x['adapt']})" for x in log)
        if log else "no log found") + ".\n")
    w("## Figures\n")
    for f in res["figures"]:
        w(f"- `{f['file']}`: {f['shows']}.")
    w("\nBaldwin plots belong to Q3 (OE+L), which is not part of the core runs.\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="results/part3/final")
    ap.add_argument("--reps", type=int, default=A.REPS, help="bootstrap resamples (50,000 as pre-registered)")
    a = ap.parse_args()
    final = Path(a.final)
    loaded = load_runs(final)
    data = {m: [read_eval(d) for d in dirs] for m, dirs in loaded["runs"].items()}
    same = check_comparable(data)
    commits = sorted({same["train_commit"], same["eval_commit"]})
    random = same["random"]
    curves = {m: np.stack([r["curve"] for r in rs]) for m, rs in data.items()}
    for m, rs in data.items():
        if not np.array_equal(curves[m][:, :, 0], np.stack([r["zero_shot"] for r in rs])):
            raise SystemExit(f"{m}: adaptation point 0 is not the zero-shot competence")
    best = A.best_per_env(curves)
    valid = A.valid_envs(best, random)
    norm = {m: A.normalized_curves(c, random, best, valid) for m, c in curves.items()}
    primary = {m: A.auc(n) for m, n in norm.items()}
    measures = {"primary": primary,
                "zero_shot": {m: n[:, :, 0] for m, n in norm.items()},
                "final": {m: n[:, :, -1] for m, n in norm.items()},
                "extinct": {m: np.stack([r["extinct"] for r in data[m]]) for m in METHODS}}
    t23 = A.env_index("T2", "T3")
    prim23 = {m: A.select(primary[m], t23) for m in METHODS}

    res = {"commits": commits, "train_commit": same["train_commit"], "eval_commit": same["eval_commit"],
           "random_max_difference": {"test": same["random_max_difference_test"],
                                     "validation": same["random_max_difference_validation"]},
           "runs": {m: [r["seed"] for r in data[m]] for m in METHODS},
           "excluded_runs": loaded["excluded"], "bootstrap_reps": a.reps,
           "environments_left_out": [int(j) for j in np.where(~valid)[0]],
           "best": best.tolist(), "random": random.tolist()}
    res["h1"] = A.h1(prim23, a.reps)
    res["iqm_primary"] = metrics.iqm_with_ci(prim23, a.reps, A.STATS_SEED)
    res["secondary"] = A.secondary(measures, a.reps)
    res["h2b"] = A.h2b(prim23, a.reps)
    means = {m: A.per_run_mean(prim23[m]) for m in METHODS}
    res["prob_improvement_matrix"] = {"methods": list(METHODS), "values": [
        [float("nan") if x == y else metrics.prob_improvement(means[x][:, None], means[y][:, None]) for y in METHODS]
        for x in METHODS]}
    res["descriptive"] = {}
    for m in METHODS:
        pre = curves[m][:, A.env_index("T1"), 0].mean(axis=1)
        res["descriptive"][m] = {
            "time_to_90_T2_T3": A.time_to_90(curves[m], pre, t23),
            "zero_shot_normalized_mean_T2_T3": float(np.nanmean(measures["zero_shot"][m][:, t23])),
            "final_normalized_mean_T2_T3": float(np.nanmean(measures["final"][m][:, t23])),
            "primary_mean_by_set": {s: float(np.nanmean(primary[m][:, A.env_index(s)])) for s in ("T1", "T2", "T3")},
            "extinction_rate_by_set": {s: float(measures["extinct"][m][:, A.env_index(s)].mean()) for s in ("T1", "T2", "T3")},
            "steps_survived_mean_by_set": {s: float(np.mean([r["steps_survived"][A.env_index(s)] for r in data[m]]))
                                           for s in ("T1", "T2", "T3")},
            "persistence_births_mean_by_set": {s: float(np.mean([r["persistence_births"][A.env_index(s)] for r in data[m]]))
                                               for s in ("T1", "T2", "T3")},
            "zero_shot_competence_mean_per_env": curves[m][:, :, 0].mean(axis=0).tolist(),
        }
    res["per_run"] = {m: [{"seed": r["seed"], "train_agent_steps": r["done"]["agent_steps"],
                           "train_wall_hours": r["done"]["wall_seconds"] / 3600, "train_commit": r["train_commit"],
                           "eval_commit": r["commit"], "adapt_wall_hours": r["adapt_wall_seconds"] / 3600,
                           "train_gpu": r["train_hardware"].get("gpus"), "eval_gpu": r["hardware"].get("gpus")}
                          for r in data[m]] for m in METHODS}
    res["compute"] = {m: {"train_agent_steps": [r["done"]["agent_steps"] for r in data[m]],
                          "train_wall_hours_mean": float(np.mean([r["done"]["wall_seconds"] for r in data[m]]) / 3600),
                          "adapt_wall_hours_mean": float(np.mean([r["adapt_wall_seconds"] for r in data[m]]) / 3600),
                          "train_hardware": sorted({json.dumps(r["train_hardware"].get("gpus")) for r in data[m]}),
                          "software": sorted({json.dumps({k: r[h].get(k) for k in ("jax", "jax_platform_version", "cuda_driver_version", "python")})
                                              for r in data[m] for h in ("train_hardware", "hardware")}),
                          "adapt_final_agent_steps_mean": float(np.mean([r["adapt_steps"][:, -1] for r in data[m]])),
                          "eval_hardware": sorted({json.dumps(r["hardware"].get("gpus")) for r in data[m]})}
                      for m in METHODS}
    spend = final / "spend.json"
    res["cost"] = json.loads(spend.read_text()) if spend.exists() else None
    access = final / "test_access_log.jsonl"
    res["test_access_log"] = [json.loads(x) for x in access.read_text().splitlines() if x.strip()] if access.exists() else None
    res["oe_population"] = {}
    for m in ("oe", "oe_fixed"):
        tr = [training_population(d) for d in loaded["runs"][m]]
        ap = {k: np.stack([r["adapt_population"][k] for r in data[m]]) for k in ("births", "floor_births", "steps_all_dead_before_floor")}
        res["oe_population"][m] = {
            "training_per_run": tr,
            "training_mean": {k: float(np.mean([t[k] for t in tr])) for k in tr[0]},
            "adaptation_mean_per_env_by_set": {k: {s: float(v[:, A.env_index(s)].mean()) for s in ("T1", "T2", "T3")}
                                               for k, v in ap.items()}}

    curves_tr = {m: [json.loads((d / "curves.json").read_text()) for d in loaded["runs"][m]] for m in METHODS}
    res["learning_curves"] = {m: {"mean": np.mean([[p["score"] for p in c["points"]] for c in curves_tr[m]], axis=0).tolist(),
                                  "sd": np.std([[p["score"] for p in c["points"]] for c in curves_tr[m]], axis=0, ddof=1).tolist()}
                              for m in METHODS}

    probes = probe_observations()
    key = stream_key(base_key("stats", PROBE_SEED), "probe")
    div_end, div_time = {}, {}
    for mi, m in enumerate(("oe", "oe_fixed")):
        div_end[m], div_time[m] = [], []
        for ri, d in enumerate(loaded["runs"][m]):
            k = jax.random.fold_in(jax.random.fold_in(key, 1000 + mi), ri)
            div_end[m].append(population_diversity(load_pickle(d / "policy.pkl"), probes, jax.random.fold_in(k, 0)))
            div_time[m].append([population_diversity(load_pickle(d / "snapshots" / f"{t}.pkl")["policy"], probes,
                                                     jax.random.fold_in(k, t)) for t in range(1, 11)])
    res["diversity_end"] = {m: {kk: float(np.mean([x[kk] for x in v])) for kk in ("weights", "traits", "behavior")}
                            for m, v in div_end.items()}
    res["diversity_over_time"] = {m: {kk: np.mean([[x[kk] for x in run] for run in v], axis=0).tolist()
                                      for kk in ("weights", "traits", "behavior")} for m, v in div_time.items()}
    res["diversity_note"] = "PPO-fixed, PPO-DR and ES-DR deploy one policy: diversity 0 by design, not tested"

    res["adaptation_curves"] = {m: {st: np.nanmean(norm[m][:, A.env_index(st), :], axis=(0, 1)).tolist()
                                    for st in ("T1", "T2", "T3")} for m in METHODS}
    norm["_primary"] = primary
    res["figures"] = figures(ROOT / "figures/part3", res, data, norm, curves_tr, div_time)
    write_json(final / "analysis.json", res)
    write_report(res, ROOT / "docs/RESULTS_part3.md")
    print(json.dumps({"h1": res["h1"], "h2b_verdict": res["h2b"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
