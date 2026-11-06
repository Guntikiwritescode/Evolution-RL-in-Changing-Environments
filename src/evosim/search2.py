"""Part 2 grid search (docs/PREREG_part2.md, Section 3).

- 4 traits x 8 log-spaced points; reach replaced by max(reach, size / 4) and
  duplicates merged (1,472 designs, each carrying its effective reach).
- Stage 1: every design, 50 clones, 30 generations, one shared stream set;
  population and food scores over generations 10-29.
- Stage 2: the top 32 of each list re-run on 8 (or 16) fresh shared stream
  sets; population list for 100 generations (scored 10-99), food-only designs
  for 30 generations (scored 10-29).
- Ties: higher score, then higher worst generation, then a random key drawn
  once per design per search.
"""
import itertools
from fractions import Fraction

import numpy as np

from . import sim2, streams, world

STAGE1_GENS = 30
SCORE_FROM = 10  # burn-in: generations 0-9 are not scored


def grid_points(cfg):
    d = cfg["blob_default"]
    g = cfg["part2"]["grid"]
    k = int(g["points"])
    f = np.exp(np.linspace(np.log(float(g["low_factor"])), np.log(float(g["high_factor"])), k))
    return {t: float(d[t]) * f for t in world.TRAITS}


def merged_designs(cfg):
    """Distinct designs in grid order (speed slowest, reach fastest). Each
    point's reach is replaced by its effective value before merging, and the
    first point of each group of duplicates is kept."""
    axes = grid_points(cfg)
    seen = {}
    for combo in itertools.product(*(axes[t] for t in world.TRAITS)):
        d = dict(zip(world.TRAITS, (float(x) for x in combo)))
        d["reach"] = max(d["reach"], d["size"] / 4.0)
        key = tuple(np.round([d[t] for t in world.TRAITS], 12))
        if key not in seen:
            seen[key] = d
    return list(seen.values())


def scores_from(pop_by_gen, food_by_gen, lo, hi):
    """Population and food scores over generations lo..hi (inclusive), plus
    the worst (lowest) population in those generations. Generations with no
    blobs count 0 in both scores. Scores are exact fractions, so two designs
    with equal totals always tie exactly (the tie rules then decide)."""
    p = [int(v) for v in pop_by_gen[lo:hi + 1]]
    f = [int(v) for v in food_by_gen[lo:hi + 1]]
    n = len(p)
    food = sum((Fraction(fi, pi) for pi, fi in zip(p, f) if pi > 0), Fraction(0))
    return Fraction(sum(p), n), food / n, min(p)


def mean_fraction(vals):
    return sum(vals, Fraction(0)) / len(vals)


def run_clones(design, env, world_cfg, n_gens, run_seed, purpose, key, cfg, n_clones):
    """One design as clones (no mutation) on the stream set (purpose, *key)."""
    pop = world.new_population(n_clones, design, float(env["size"]),
                               sim2.spots_rng(run_seed, purpose, *key),
                               flee=float(cfg["blob_default"]["flee"]))
    log = sim2.GenLog()
    sim2.run_streamed(pop, env, world_cfg, sim2.ZERO_SD, n_gens, run_seed, purpose, key, log, 0, "S")
    a = log.arrays()
    return a["g_population"], a["g_food_eaten"] + a["g_prey_eaten"], int(a["g_creature_steps"].sum())


def order(score, worst, tiekey):
    """Indices sorted best first: score, then worst generation, then tie key
    (all higher is better). Scores may be exact fractions."""
    n = len(score)
    return np.array(sorted(range(n), key=lambda i: (-score[i], -worst[i], -float(tiekey[i]))), dtype=np.int64)


def stage1(designs, env, world_cfg, run_seed, purpose, key, cfg):
    """Every design on one shared stream set for 30 generations."""
    n = len(designs)
    nc = int(cfg["part2"]["n_clones"])
    pops = np.zeros((n, STAGE1_GENS), np.int32)
    out = {"pop_exact": [], "food_exact": [], "worst": np.zeros(n), "steps": 0}
    for i, d in enumerate(designs):
        p, f, s = run_clones(d, env, world_cfg, STAGE1_GENS, run_seed, purpose, key, cfg, nc)
        pops[i] = p
        ps, fs, wo = scores_from(p, f, SCORE_FROM, STAGE1_GENS - 1)
        out["pop_exact"].append(ps)
        out["food_exact"].append(fs)
        out["worst"][i] = wo
        out["steps"] += s
    out["pop_score"] = np.array([float(v) for v in out["pop_exact"]])
    out["food_score"] = np.array([float(v) for v in out["food_exact"]])
    out["pops"] = pops
    return out


def a_search(cfg, run_seed, n_sets=8, top=32):
    """The confirmatory A search for one run seed. Returns everything needed
    for the winners, the stage-1-only winners, the checks and the cost."""
    designs = merged_designs(cfg)
    n = len(designs)
    env, wc = cfg["env_A"], cfg["world"]
    nc = int(cfg["part2"]["n_clones"])
    n_train = int(cfg["phases"]["train_generations"])
    tie = streams.rng(run_seed, "tiebreak_A").random(n)

    s1 = stage1(designs, env, wc, run_seed, "search_A1", (), cfg)
    o_pop = order(s1["pop_exact"], s1["worst"], tie)
    o_food = order(s1["food_exact"], s1["worst"], tie)
    top_pop, top_food = [int(i) for i in o_pop[:top]], [int(i) for i in o_food[:top]]
    tied_pop = sum(1 for v in s1["pop_exact"] if v == s1["pop_exact"][o_pop[0]])
    tied_food = sum(1 for v in s1["food_exact"] if v == s1["food_exact"][o_food[0]])

    union = sorted(set(top_pop) | set(top_food))
    long_ids = set(top_pop)
    s2 = {}
    steps2 = 0
    for i in union:
        gens = n_train if i in long_ids else STAGE1_GENS
        rows = []
        for j in range(n_sets):
            p, f, s = run_clones(designs[i], env, wc, gens, run_seed, "search_A2", (j,), cfg, nc)
            rows.append((p, f))
            steps2 += s
        s2[i] = rows

    def s2_scores(i, which):
        rows = s2[i]
        if which == "pop":
            vals = [scores_from(p, f, SCORE_FROM, n_train - 1) for p, f in rows]
            return mean_fraction([v[0] for v in vals]), min(v[2] for v in vals)
        vals = [scores_from(p, f, SCORE_FROM, STAGE1_GENS - 1) for p, f in rows]
        return mean_fraction([v[1] for v in vals]), min(v[2] for v in vals)

    def winner(cands, which):
        sc = [s2_scores(i, which)[0] for i in cands]
        wo = [s2_scores(i, which)[1] for i in cands]
        tk = [tie[i] for i in cands]
        o = order(sc, wo, tk)
        best = cands[int(o[0])]
        return best, sc[int(o[0])], sum(1 for v in sc if v == sc[int(o[0])])

    pop_w, pop_w_score, tied2_pop = winner(top_pop, "pop")
    food_w, food_w_score, tied2_food = winner(top_food, "food")
    return {
        "designs": designs, "tie_keys": tie,
        "stage1": {k: s1[k] for k in ("pop_score", "food_score", "worst", "pops")},
        "top_pop": top_pop, "top_food": top_food,
        "stage2_pop_scores": {i: float(s2_scores(i, "pop")[0]) for i in top_pop},
        "stage2_food_scores": {i: float(s2_scores(i, "food")[0]) for i in top_food},
        "pop_winner": pop_w, "food_winner": food_w,
        "pop_winner_stage2_score": float(pop_w_score), "food_winner_stage2_score": float(food_w_score),
        "pop_winner_stage2_exact": [pop_w_score.numerator, pop_w_score.denominator],
        "stage1_pop_winner": int(o_pop[0]), "stage1_food_winner": int(o_food[0]),
        "tied_stage1_pop": tied_pop, "tied_stage1_food": tied_food,
        "tied_stage2_pop": tied2_pop, "tied_stage2_food": tied2_food,
        "n_sets": n_sets, "top": top,
        "steps_stage1": int(s1["steps"]), "steps_stage2": int(steps2),
    }


def b_search(cfg, run_seed, world_id, env_B, world_cfg_B, n_sets=8, top=32):
    """Search directly in a B world (ORACLE_B, U2): clones start in B;
    stage 1 30 generations (scored 10-29); stage 2 the top 32 by population
    for 50 generations on 8 sets (scored 10-49); population score only.
    world_id selects the stream keys (arm 3 uses the main B's, Section 10)."""
    designs = merged_designs(cfg)
    n = len(designs)
    nc = int(cfg["part2"]["n_clones"])
    n_b = int(cfg["phases"]["transfer_generations"])
    tie = streams.rng(run_seed, "tiebreak_B", world_id).random(n)
    s1 = stage1(designs, env_B, world_cfg_B, run_seed, "search_B1", (world_id,), cfg)
    o = order(s1["pop_exact"], s1["worst"], tie)
    cands = [int(i) for i in o[:top]]
    sc, wo, steps2 = [], [], 0
    for i in cands:
        vals = []
        for j in range(n_sets):
            p, f, s = run_clones(designs[i], env_B, world_cfg_B, n_b, run_seed, "search_B2",
                                 (world_id, j), cfg, nc)
            vals.append(scores_from(p, f, SCORE_FROM, n_b - 1))
            steps2 += s
        sc.append(mean_fraction([v[0] for v in vals]))
        wo.append(min(v[2] for v in vals))
    o2 = order(sc, wo, [tie[i] for i in cands])
    best = cands[int(o2[0])]
    top_exact = s1["pop_exact"][int(o[0])]
    return {"stage1_pop_score": s1["pop_score"], "stage1_worst": s1["worst"],
            "stage1_pops": s1["pops"], "top": cands,
            "stage2_scores": {i: float(v) for i, v in zip(cands, sc)},
            "winner": best, "winner_stage2_score": float(sc[int(o2[0])]),
            "tied_stage1": sum(1 for v in s1["pop_exact"] if v == top_exact),
            "tied_stage2": sum(1 for v in sc if v == sc[int(o2[0])]),
            "steps_stage1": int(s1["steps"]), "steps_stage2": int(steps2)}


def landscape_stage1(cfg, run_seed, world_id, env_B, world_cfg_B):
    """Landscape check (Section 18.4): B stage 1 for one suite variant, on its own streams."""
    designs = merged_designs(cfg)
    s1 = stage1(designs, env_B, world_cfg_B, run_seed, "landscape", (world_id,), cfg)
    return {"pop_score": s1["pop_score"], "steps": int(s1["steps"])}


def u2_runs(cfg, run_seed, world_id, env_B, world_cfg_B, b_winner, a_winner, n_sets=8):
    """U2: the B winner and the A population winner as clones in B on 8 fresh
    sets of streams; scores over generations 10-49. Also returns the exact
    population totals, so U2's 2.5-blob rule can be checked in whole numbers."""
    designs = merged_designs(cfg)
    nc = int(cfg["part2"]["n_clones"])
    n_b = int(cfg["phases"]["transfer_generations"])
    out, steps = {}, 0
    for name, i in (("b_winner", b_winner), ("a_winner", a_winner)):
        total, count = 0, 0
        for j in range(n_sets):
            p, f, s = run_clones(designs[i], env_B, world_cfg_B, n_b, run_seed, "u2",
                                 (world_id, j), cfg, nc)
            seg = [int(v) for v in p[SCORE_FROM:n_b]]
            total += sum(seg)
            count += len(seg)
            steps += s
        out[name] = total / count
        out[name + "_total"] = total
    out["count"] = count
    out["steps"] = steps
    return out
