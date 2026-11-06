"""Part 2 secondary and exploratory analyses (docs/PREREG_part2.md, Sections 5 and 18.4-18.7).

Nothing here decides a confirmatory verdict. Every number is an effect size with
a 95% CI (or a plain summary) and is labeled exploratory in the report.

Usage:
  python scripts/part2/explore_part2.py --run final [--quick]
Writes results/part2/<run>/analysis/exploratory.json and series.npz (curves for figures).
"""
import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "part2"))
from evosim import config, metrics, p2metrics as M, p2stats as S, world  # noqa: E402
import analyze_part2 as A  # noqa: E402

N_A = 100  # A generations; set from the run's config in main() (12 in --quick test runs)
EXPL_SEED_ARM = 50  # stats streams for exploratory work: arm ids 50+ (never used by confirmatory arms 1-4)
# B worlds with their own B search; the exploratory items run in each (Sections 9.2, 9.3, 18).
# W_OFF keeps their stats streams apart.
SEARCHED_WORLDS = ("main", "arm3", "arm4")
W_OFF = {"main": 0, "arm3": 1, "arm4": 2}
GROUPS = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen", "GS_pop_mut",
          "SV_frozen", "ORACLE_B")


def boot_paired(d, rng, n_boot=10000):
    return S.boot_mean_ci(d, rng, 0.95, n_boot)


def diff_ci_95(x1, y1, x2, y2, arm_id, comp_id, n_boot):
    """Unpaired 95% CIs: P1 (studentized bootstrap), P2 (Newcombe)."""
    xs, ys = np.asarray(x1) / 10.0, np.asarray(y1) / 10.0
    ci1 = S.boot_t_ci(xs, ys, S.stats_rng(arm_id, comp_id, 1, S.BOOT), (0.95,), n_boot)[0.95]
    xc, yc = int(np.sum(x2)), int(np.sum(y2))
    ci2 = S.newcombe(xc, len(x2), yc, len(y2), 0.95)
    return {"P1_diff": float(xs.mean() - ys.mean()), "P1_ci95": ci1,
            "P2_diff": xc / len(x2) - yc / len(y2), "P2_ci95": ci2}


def get(outs, s, world, group):
    return outs[s].get(f"B__{world}__{group}")


# ------------------------------------------------------------------ secondary (confirmatory arms)
def secondary(seeds, metas, outs, n_boot):
    res = {}
    for arm, A_ in A.ARMS.items():
        w = A_["world"]
        if get(outs, seeds[0], w, "ET") is None:
            continue
        r = {"paired": {}, "reached_B_ci": {}, "S6": {}, "S4": {}, "S1": {}, "survival_curve": {},
             }  # Mann-Whitney U and Cliff's delta are in confirmatory.json (raw[c]["P1"])
        for c in A_["comps"]:
            g1, g2 = A.COMPARISONS[c]
            x1 = np.array(A.series(seeds, outs, w, g1, "P1_total"))
            y1 = np.array(A.series(seeds, outs, w, g2, "P1_total"))
            x2 = np.array(A.series(seeds, outs, w, g1, "P2"))
            y2 = np.array(A.series(seeds, outs, w, g2, "P2"))
            d = (x1 - y1) / 10.0
            pr = S.stats_rng(EXPL_SEED_ARM + A_["id"], c, 1, S.PAIRED_PERM)
            r["paired"][c] = {"signflip_p": S.paired_signflip(d, pr),
                              "P1_mean_diff": float(d.mean()),
                              "P1_ci95": boot_paired(d, S.stats_rng(EXPL_SEED_ARM + A_["id"], c, 1, S.PAIRED_BOOT), n_boot),
                              "mcnemar_p": S.mcnemar_exact(x2, y2)}
            s5x = np.array(A.series(seeds, outs, w, g1, "S5"))
            s5y = np.array(A.series(seeds, outs, w, g2, "S5"))
            both = s5x & s5y
            if both.sum() >= 2:
                d1 = (x1[both] - y1[both]) / 10.0
                d2 = x2[both].astype(int) - y2[both].astype(int)
                r["reached_B_ci"][c] = {"n_both": int(both.sum()), "P1_diff": float(d1.mean()),
                                        "P1_ci95": boot_paired(d1, S.stats_rng(EXPL_SEED_ARM + A_["id"], c, 2, S.PAIRED_BOOT), n_boot),
                                        "P2_diff": float(d2.mean()),
                                        "P2_ci95": boot_paired(d2, S.stats_rng(EXPL_SEED_ARM + A_["id"], c, 3, S.PAIRED_BOOT), n_boot)}
        for g in GROUPS:
            if get(outs, seeds[0], w, g) is None:
                continue
            r["S6"][g] = float(np.mean(A.series(seeds, outs, w, g, "S6")))
            s4 = [v for v in A.series(seeds, outs, w, g, "S4") if not np.isnan(v)]
            r["S4"][g] = {"mean": float(np.mean(s4)) if s4 else None, "n_runs": len(s4)}
            s1 = A.series(seeds, outs, w, g, "S1")
            r["S1"][g] = {"median": (float(np.median([v for v in s1 if v is not None]))
                                     if any(v is not None for v in s1) else None),
                          "never": int(sum(v is None for v in s1))}
            # S2: share of runs with blobs, by B generation
            curve = []
            for bg in range(50):
                alive = 0
                for s in seeds:
                    a = outs[s]["_arrays"]
                    gen = a[f"B__{w}__{g}__g_gen"]
                    pop = a[f"B__{w}__{g}__g_population"]
                    alive += int(pop[gen == N_A + bg][0] > 0)
                curve.append(alive / len(seeds))
            r["survival_curve"][g] = curve
        res[arm] = r
    # arm 2 descriptive extra: ET's mean P1 above 25?
    if "arm2" in res:
        et = A.series(seeds, outs, "arm2", "ET", "P1_total")
        res["arm2"]["ET_mean_P1"] = sum(et) / (10 * len(et))
        res["arm2"]["ET_mean_P1_above_25"] = Fraction(sum(et), 10 * len(et)) > 25
        res["arm2"]["blobs_eaten_per_group"] = {g: float(np.mean(A.series(seeds, outs, "arm2", g, "prey_eaten_B")))
                                                for g in GROUPS if get(outs, seeds[0], "arm2", g)}
    return res


def compute_cost(seeds, metas, outs):
    """S3: creature-steps per group (guess K1), search cost included."""
    out = {}
    for arm in SEARCHED_WORLDS:
        if get(outs, seeds[0], arm, "ET") is None:
            continue
        rows = {}
        for g in GROUPS:
            costs = []
            for s in seeds:
                m = metas[s]
                a_steps = outs[s].get(f"A__{g}", {}).get("steps_A", 0)
                if g in ("ET_freezeB", "ET_cloneB"):
                    a_steps = outs[s]["A__ET"]["steps_A"]
                b_steps = outs[s][f"B__{arm}__{g}"]["steps_B"]
                search = 0
                if g.startswith("GS_"):
                    search = m["search_A"]["steps_stage1"] + m["search_A"]["steps_stage2"]
                if g == "ORACLE_B":
                    sb = m["search_B"][arm]
                    search = sb["steps_stage1"] + sb["steps_stage2"]
                    a_steps = 0
                costs.append(a_steps + b_steps + search)
            rows[g] = {"mean": float(np.mean(costs)), "sd": float(np.std(costs, ddof=1)) if len(costs) > 1 else None}
        out[arm] = rows
    return out


# ------------------------------------------------------------------ suite (18.4)
def suite(seeds, metas, outs, n_boot):
    expl = [s for s in seeds if metas[s]["exploratory"]]
    if not expl:
        return {}
    variants = sorted({k.split("__")[1] for k in outs[expl[0]] if k.startswith("B__")} -
                      {"main", "arm2", "arm3", "arm4"})
    main_dirs = {}
    for c in A.ARMS["main"]["comps"]:
        g1, g2 = A.COMPARISONS[c]
        x = np.array(A.series(expl, outs, "main", g1, "P1_total")) / 10
        y = np.array(A.series(expl, outs, "main", g2, "P1_total")) / 10
        ci = S.boot_t_ci(x, y, S.stats_rng(EXPL_SEED_ARM + 10, c, 1, S.BOOT), (0.95,), n_boot)[0.95]
        main_dirs[c] = 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)
    res = {"variants": {}, "summary": {}}
    has_arm3 = get(outs, expl[0], "arm3", "ET") is not None
    for v in variants:
        food = int(v[4:]) if v.startswith("food") else 25
        thr = Fraction(food, 10)
        cap = None if v == "cann_arm3" else food
        vr = {"groups": {}, "comparisons": {}, "threshold_P1": float(thr)}
        for g in GROUPS[:-1]:
            src = v
            if v == "cann_arm3" and g in A.CLONE_GROUPS and has_arm3:
                src = "arm3"  # clones are identical with blob-eating on (Section 9.1)
            if get(outs, expl[0], src, g) is None:
                continue
            p1 = np.array(A.series(expl, outs, src, g, "P1"))
            vr["groups"][g] = {"P1": float(p1.mean()), "P2": float(np.mean(A.series(expl, outs, src, g, "P2"))),
                               "S6": float(np.mean(A.series(expl, outs, src, g, "S6"))),
                               "near_cap": (float(np.mean(p1 >= food - float(thr))) if cap else None),
                               "src": src}
        for c in A.ARMS["main"]["comps"]:
            g1, g2 = A.COMPARISONS[c]
            if g1 not in vr["groups"] or g2 not in vr["groups"]:
                continue
            s1, s2 = vr["groups"][g1]["src"], vr["groups"][g2]["src"]
            x1 = np.array(A.series(expl, outs, s1, g1, "P1_total"))
            y1 = np.array(A.series(expl, outs, s2, g2, "P1_total"))
            sx = np.array(A.series(expl, outs, s1, g1, "S5")) & np.array(A.series(expl, outs, s2, g2, "S5"))
            if sx.sum() < 2:
                continue
            d = (x1[sx] - y1[sx]) / 10.0
            ci = boot_paired(d, S.stats_rng(EXPL_SEED_ARM + 20, c, abs(hash_variant(v)), S.PAIRED_BOOT), n_boot)
            x2 = np.array(A.series(expl, outs, s1, g1, "P2"))[sx].astype(int)
            y2 = np.array(A.series(expl, outs, s2, g2, "P2"))[sx].astype(int)
            direction = 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)
            vr["comparisons"][c] = {"n_both_reached_B": int(sx.sum()), "P1_diff": float(d.mean()), "P1_ci95": ci,
                                    "P2_diff": float((x2 - y2).mean()),
                                    "P2_ci95": S.newcombe(int(x2.sum()), int(sx.sum()), int(y2.sum()), int(sx.sum()), 0.95),
                                    "direction": direction,
                                    "reaches_threshold": bool(abs(Fraction(int((x1[sx] - y1[sx]).sum()), 10 * int(sx.sum()))) >= thr)}
        res["variants"][v] = vr
    for c in A.ARMS["main"]["comps"]:
        dirs = [res["variants"][v]["comparisons"][c]["direction"] for v in res["variants"]
                if c in res["variants"][v]["comparisons"]]
        reach = [res["variants"][v]["comparisons"][c]["reaches_threshold"] for v in res["variants"]
                 if c in res["variants"][v]["comparisons"]]
        res["summary"][c] = {"main_direction": main_dirs[c], "n_variants": len(dirs),
                             "same_direction": (sum(1 for d in dirs if d == main_dirs[c] and d != 0)
                                                if main_dirs[c] != 0 else None),
                             "other_direction": (sum(1 for d in dirs if d == -main_dirs[c] and d != 0)
                                                 if main_dirs[c] != 0 else None),
                             "positive": sum(1 for d in dirs if d == 1), "negative": sum(1 for d in dirs if d == -1),
                             "reaches_threshold": sum(reach)}
    return res


def hash_variant(v):
    """Stable small integer per variant name (for stats stream keys)."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(v)) % 100000


def landscape(seeds, metas, outs):
    res = {}
    for s in seeds:
        a = outs[s]["_arrays"]
        if "SA__stage1_pop_score" not in a:
            continue
        sa = a["SA__stage1_pop_score"]
        winner = metas[s]["search_A"]["pop_winner"]
        worlds = {w: f"SB__{w}__stage1_pop_score" for w in SEARCHED_WORLDS}
        worlds.update({k.split("__")[1]: k for k in a if k.startswith("LS__")})
        for w, key in worlds.items():
            if key not in a:
                continue
            sb = a[key]
            tau_all = sps.kendalltau(sa, sb, variant="b").statistic
            alive = sa > 0
            tau_alive = sps.kendalltau(sa[alive], sb[alive], variant="b").statistic if alive.sum() > 2 else None
            ranks = sps.rankdata(sb)  # average ranks for ties; higher score = higher rank
            pct = float((ranks[winner] - 1) / (len(sb) - 1)) if len(sb) > 1 else None
            res.setdefault(w, []).append({"seed": s, "tau_b_all": float(tau_all),
                                          "tau_b_A_alive": None if tau_alive is None else float(tau_alive),
                                          "A_winner_B_rank_percentile": pct})
    return {w: {"n_seeds": len(v), "mean_tau_b_all": float(np.nanmean([r["tau_b_all"] for r in v])),
                "mean_tau_b_A_alive": float(np.nanmean([r["tau_b_A_alive"] for r in v if r["tau_b_A_alive"] is not None]))
                if any(r["tau_b_A_alive"] is not None for r in v) else None,
                "mean_A_winner_rank_pct": float(np.nanmean([r["A_winner_B_rank_percentile"] for r in v])),
                "per_seed": v} for w, v in res.items()}


# ------------------------------------------------------------------ sensitivity (18.5)
def mutation_sensitivity(seeds, metas, outs, n_boot):
    expl = [s for s in seeds if metas[s]["exploratory"]]
    if not expl:
        return {}
    factors = sorted({k.split("__")[2].split("_x")[1] for k in outs[expl[0]]
                      if k.startswith("B__main__ET_x")})
    res = {}
    # "1" is the frozen setting on the same exploratory seeds, so each row compares like for like
    # (stats arm ids 80/180/280; never used elsewhere)
    for f in ["1"] + factors:
        t = "" if f == "1" else f"_x{f}"
        ren = {"ET": "ET" + t, "ET_freezeB": "ET_freezeB" + t, "ET_cloneB": "ET_cloneB" + t, "GS_pop_mut": "GS_pop_mut" + t}
        arm_base = EXPL_SEED_ARM + 30 + (0 if f == "1" else int(float(f) * 10))
        for w in SEARCHED_WORLDS:
            if get(outs, expl[0], w, "ET" + t) is None:
                continue
            rows = {}
            for c, (g1, g2) in A.COMPARISONS.items():
                h1, h2 = ren.get(g1, g1), ren.get(g2, g2)
                x1, y1 = A.series(expl, outs, w, h1, "P1_total"), A.series(expl, outs, w, h2, "P1_total")
                x2, y2 = A.series(expl, outs, w, h1, "P2"), A.series(expl, outs, w, h2, "P2")
                rows[c] = diff_ci_95(x1, y1, x2, y2, arm_base + 100 * W_OFF[w], c, n_boot)
                # runs with any population in B gens 40-49; an interval from a group with almost
                # no such runs means little (the report flags 3 or fewer)
                rows[c]["n_nonzero"] = [int(np.sum(np.asarray(x1) > 0)), int(np.sum(np.asarray(y1) > 0))]
                rows[c]["n"] = len(x1)
            res[f"x{f}_{w}"] = rows
    return res


def threshold_sensitivity(conf_results):
    res = {}
    for arm, r in conf_results.items():
        caps = {c: A.ARMS[arm]["cap"] for c in r["raw"]}
        for tag, thr in (("half", {"P1": Fraction(5, 4), "P2": Fraction(1, 20)}),
                         ("double", {"P1": Fraction(5), "P2": Fraction(1, 5)})):
            d = S.decide_family(r["raw"], A.ARMS[arm]["k"], caps, thr=thr)
            base = r["decided"]
            res[f"{arm}_{tag}"] = {"verdicts": {c: d[c]["overall_label"] for c in d},
                                   "changed": [c for c in d if d[c]["overall_label"] != base[c]["overall_label"]]}
    return res


def longer_b(seeds, metas, outs):
    expl = [s for s in seeds if metas[s]["exploratory"]]
    res = {}
    for w in SEARCHED_WORLDS:
        for g in GROUPS:
            o = [get(outs, s, w, g) for s in expl]
            if not o or o[0] is None or "late_P1_total" not in o[0]:
                continue
            res.setdefault(w, {})[g] = {"late_P1": float(np.mean([x["late_P1_total"] for x in o]) / 10),
                                        "late_P2": float(np.mean([x["late_P2"] for x in o])),
                                        "P1": float(np.mean([x["P1"] for x in o])), "n": len(o)}
    return res


def stage1_winners(seeds, metas, outs):
    expl = [s for s in seeds if metas[s]["exploratory"]]
    res = {}
    for w in SEARCHED_WORLDS:
        rows = {}
        for g in ("GS_pop_s1", "GS_food_s1", "GS_pop_frozen", "GS_food_frozen"):
            o = [get(outs, s, w, g) for s in expl]
            if not o or o[0] is None:
                continue
            rows[g] = {"P1": float(np.mean([x["P1"] for x in o])), "P2": float(np.mean([x["P2"] for x in o])),
                       "died_in_A": int(sum(not x["S5"] for x in o)), "n": len(o)}
        if rows:
            res[w] = rows
    res["same_as_two_stage"] = {"pop": int(sum(metas[s]["search_A"]["stage1_pop_winner"] == metas[s]["search_A"]["pop_winner"] for s in expl)),
                                "food": int(sum(metas[s]["search_A"]["stage1_food_winner"] == metas[s]["search_A"]["food_winner"] for s in expl)),
                                "n": len(expl)}
    return res


def scratch_b(seeds, metas, outs, n_boot):
    expl = [s for s in seeds if metas[s]["exploratory"]]
    res = {}
    for w in SEARCHED_WORLDS:
        if not expl or get(outs, expl[0], w, "ET_scratchB") is None:
            continue
        res[w] = diff_ci_95(A.series(expl, outs, w, "ET_scratchB", "P1_total"), A.series(expl, outs, w, "ET", "P1_total"),
                            A.series(expl, outs, w, "ET_scratchB", "P2"), A.series(expl, outs, w, "ET", "P2"),
                            EXPL_SEED_ARM + 40 + W_OFF[w], 1, n_boot)
    return res


# ------------------------------------------------------------------ mechanisms (18.7)
def gradients(seeds, metas, outs, n_boot):
    expl = [s for s in seeds if metas[s]["exploratory"]] or seeds
    b0, b9, a90, a99 = N_A, N_A + 9, N_A - 10, N_A - 1
    plans = {("ET", "A"): ("A__ET", a90, a99), ("ET", "B"): ("B__main__ET", b0, b9),
             ("GS_pop_mut", "A"): ("A__GS_pop_mut", a90, a99), ("GS_pop_mut", "B"): ("B__main__GS_pop_mut", b0, b9),
             ("ET_freezeB", "B"): ("B__main__ET_freezeB", b0, b9),
             ("SV_frozen", "A"): ("A__SV_frozen", 0, 9), ("SV_frozen", "B"): ("B__main__SV_frozen", b0, b9)}
    names = ("log_speed", "log_size", "log_sense", "log_reach_eff_over_size")
    res = {}
    for (g, ph), (pref, lo, hi) in plans.items():
        betas, n_ok = [], 0
        for s in expl:
            a = outs[s]["_arrays"]
            b = M.blob_arrays(a, pref)
            if "gen" not in b:
                continue
            gen = b["gen"]
            m = (gen >= lo) & (gen <= hi)
            if m.sum() < 30:
                continue
            fit = M.blob_fitness({k: b[k][m] for k in ("alive_end", "food_balls", "prey")}).astype(float)
            sp, sz, se, re = (b[t][m] for t in ("speed", "size", "sense", "reach"))
            X = np.column_stack([np.log(sp), np.log(sz), np.log(np.maximum(se, 1e-12)),
                                 np.log(np.maximum(re, sz / 4.0) / sz)])
            gg = gen[m]
            rows, ys = [], []
            for gv in np.unique(gg):
                k = gg == gv
                wbar = fit[k].mean()
                if wbar == 0:
                    continue
                Xk = X[k]
                sd = Xk.std(axis=0)
                Z = np.where(sd > 0, (Xk - Xk.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)
                rows.append(Z)
                ys.append(fit[k] / wbar)
            if not rows:
                continue
            Z = np.vstack(rows)
            y = np.concatenate(ys)
            keep = Z.std(axis=0) > 0
            if Z.shape[0] < 30 or keep.sum() == 0:
                continue
            Zk = np.column_stack([np.ones(len(y)), Z[:, keep]])
            if np.linalg.cond(Zk) >= 1000:
                continue
            coef = np.linalg.lstsq(Zk, y, rcond=None)[0][1:]
            full = np.full(4, np.nan)
            full[keep] = coef
            betas.append(full)
            n_ok += 1
        if betas:
            B = np.array(betas)
            r = S.stats_rng(EXPL_SEED_ARM + 60, len(res), 1, S.PAIRED_BOOT)
            cis = []
            for j in range(4):
                col = B[:, j][~np.isnan(B[:, j])]
                cis.append(S.boot_mean_ci(col, r, 0.95, n_boot) if len(col) >= 2 else (None, None))
            res[f"{g}_{ph}"] = {"n_runs_with_gradient": n_ok, "n_runs": len(expl),
                                "mean": {nm: float(np.nanmean(B[:, j])) for j, nm in enumerate(names)},
                                "ci95": {nm: cis[j] for j, nm in enumerate(names)}}
        else:
            res[f"{g}_{ph}"] = {"n_runs_with_gradient": 0, "n_runs": len(expl)}
    return res


def first_gen_deaths(seeds, outs, w="main"):
    res = {}
    for g in GROUPS:
        share, surv_tr, dead_tr = [], [], []
        for s in seeds:
            b = M.blob_arrays(outs[s]["_arrays"], f"B__{w}__{g}")
            if "gen" not in b:
                continue
            m = b["gen"] == N_A
            if m.sum() == 0:
                continue
            alive = b["alive_end"][m].astype(bool)
            share.append(1 - alive.mean())
            surv_tr.append([b[t][m][alive].mean() if alive.any() else np.nan for t in world.TRAITS])
            dead_tr.append([b[t][m][~alive].mean() if (~alive).any() else np.nan for t in world.TRAITS])
        if share:
            res[g] = {"share_dying_B0": float(np.mean(share)), "n_runs": len(share),
                      "survivor_traits": dict(zip(world.TRAITS, np.nanmean(surv_tr, axis=0).tolist())),
                      "dead_traits": dict(zip(world.TRAITS, np.nanmean(dead_tr, axis=0).tolist()))}
    return res


def energy_use(seeds, outs, w="main"):
    """Section 18.7, B generations 0-49 only (exploratory seeds also hold the longer B, which
    is left out here), every seed. Ticks are averaged over generations that had blobs."""
    res = {}
    for g in GROUPS:
        cpm, ticks = [], []
        for s in seeds:
            a = outs[s]["_arrays"]
            b = M.blob_arrays(a, f"B__{w}__{g}")
            if "gen" not in b or len(b["moves"]) == 0:
                continue
            inb = (np.asarray(b["gen"]) >= N_A) & (np.asarray(b["gen"]) <= N_A + 49)
            mv = np.asarray(b["moves"])[inb]
            k = mv > 0
            if k.any():
                cpm.append(float((np.asarray(b["consumed"])[inb][k] / mv[k]).mean()))
            gg = M.gen_arrays(a, f"B__{w}__{g}")
            win = (np.asarray(gg["gen"]) >= N_A) & (np.asarray(gg["gen"]) <= N_A + 49) & (np.asarray(gg["population"]) > 0)
            if win.any():
                ticks.append(float(np.mean(np.asarray(gg["ticks"])[win])))
        if cpm:
            res[g] = {"energy_per_move": float(np.mean(cpm)), "ticks_per_generation_with_blobs": float(np.mean(ticks)),
                      "n_runs": len(cpm), "window": "B generations 0-49"}
    return res


def achecks(seeds, metas, outs, rundir=None):
    """Section 4 A-checks for every run seed. Seeds whose run skipped them (2100-2199 in the
    final run) use results/<run>/achecks/seed<k>.npz from achecks_all.py (MISTAKES #37)."""
    arr = {}
    for s in seeds:
        if "ACHECK__mut__g_gen" in outs[s]["_arrays"]:
            arr[s] = outs[s]["_arrays"]
        elif rundir and os.path.exists(os.path.join(rundir, "achecks", f"seed{s:04d}.npz")):
            with np.load(os.path.join(rundir, "achecks", f"seed{s:04d}.npz")) as z:
                arr[s] = {k: z[k] for k in z.files}
    expl = [s for s in seeds if s in arr]
    if not expl:
        return {}
    vals = {}
    for name in ("mut", "frozen", "clone"):
        v = []
        for s in expl:
            g = M.gen_arrays(arr[s], f"ACHECK__{name}")
            pop = np.asarray(g["population"])
            v.append(float(pop[-10:].mean()))
        vals[name] = np.array(v)
    r = S.stats_rng(EXPL_SEED_ARM + 70, 1, 1, S.PAIRED_BOOT)
    return {"n": len(expl), "means": {k: float(v.mean()) for k, v in vals.items()},
            "row5_frozen_minus_mut": {"diff": float((vals["frozen"] - vals["mut"]).mean()),
                                      "ci95": S.boot_mean_ci(vals["frozen"] - vals["mut"], r)},
            "row6_clone_minus_frozen": {"diff": float((vals["clone"] - vals["frozen"]).mean()),
                                        "ci95": S.boot_mean_ci(vals["clone"] - vals["frozen"], r)}}


def a_ties_in_b(seeds, outs):
    """Section 5 search check: the B stage-1 population score of every design tied for best in
    A's stage 1, per searched B world (does being best in A predict doing well in B?)."""
    res = {}
    for w in SEARCHED_WORLDS:
        means, shares, n_tied, best_b = [], [], [], []
        for s in seeds:
            a = outs[s]["_arrays"]
            if f"SB__{w}__stage1_pop_score" not in a:
                continue
            sa, sb = np.asarray(a["SA__stage1_pop_score"]), np.asarray(a[f"SB__{w}__stage1_pop_score"])
            tied = sa == sa.max()
            means.append(float(sb[tied].mean()))
            shares.append(float(np.mean(sb[tied] >= 22.5)))
            n_tied.append(int(tied.sum()))
            best_b.append(float(sb.max()))
        if means:
            res[w] = {"n_seeds": len(means), "mean_n_tied_in_A": float(np.mean(n_tied)),
                      "mean_B_score_of_A_tied": float(np.mean(means)),
                      "mean_share_of_A_tied_with_B_score_ge_22_5": float(np.mean(shares)),
                      "mean_best_B_score": float(np.mean(best_b))}
    return res


def curves(seeds, outs):
    """Mean population and trait means / variety over time per group, in each searched B world, for figures."""
    out = {}
    for w in SEARCHED_WORLDS:
        for g in GROUPS:
            if get(outs, seeds[0], w, g) is None:
                continue
            pops, cv2, shan, traits = [], [], [], {t: [] for t in world.TRAITS}
            for s in seeds:
                a = outs[s]["_arrays"]
                parts = []
                if g not in ("ORACLE_B",):
                    src = "A__ET" if g in ("ET_freezeB", "ET_cloneB") else f"A__{g}"
                    if f"{src}__g_gen" in a:
                        parts.append(src)
                parts.append(f"B__{w}__{g}")
                pop_s, cv_s, h_s, tr_s = [], [], [], {t: [] for t in world.TRAITS}
                for p in parts:
                    sub = {k[len(p) + 2:]: a[k] for k in a if k.startswith(p + "__")}
                    off = int(sub["g_gen"][0])
                    sub = dict(sub)
                    sub["g_gen"] = sub["g_gen"] - off
                    sub["b_gen"] = sub["b_gen"] - off
                    se = metrics.series(sub)
                    pop_s.append(se["population"])
                    cv_s.append(se["cv2_4_samp"])
                    h_s.append(se["H_exact4"])
                    for t in world.TRAITS:
                        tr_s[t].append(se[f"mean_{t}"])
                L = N_A + 50 if g != "ORACLE_B" else 50
                pops.append(np.concatenate(pop_s)[:L])
                cv2.append(np.concatenate(cv_s)[:L])
                shan.append(np.concatenate(h_s)[:L])
                for t in world.TRAITS:
                    traits[t].append(np.concatenate(tr_s[t])[:L])
            out[f"{w}__{g}__population_mean"] = np.nanmean(np.array(pops, float), axis=0)
            out[f"{w}__{g}__cv2_mean"] = np.nanmean(np.array(cv2, float), axis=0)
            out[f"{w}__{g}__shannon_mean"] = np.nanmean(np.array(shan, float), axis=0)
            for t in world.TRAITS:
                out[f"{w}__{g}__{t}_mean"] = np.nanmean(np.array(traits[t], float), axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global N_A
    rundir = os.path.join(ROOT, "results", "part2", args.run)
    with open(os.path.join(rundir, "run.json")) as f:
        N_A = int(json.load(f)["provenance"]["config"]["phases"]["train_generations"])
    seeds, metas, outs = A.load_run(rundir)
    n_boot = 2000 if args.quick else 10000
    n_perm, n_boot_c = (1000, 2000) if args.quick else (10000, 20000)
    conf = {}
    for arm in A.ARMS:
        if get(outs, seeds[0], A.ARMS[arm]["world"], "ET") is not None:
            conf[arm] = A.analyze_arm(arm, seeds, metas, outs, n_perm, n_boot_c)
    res = {"secondary": secondary(seeds, metas, outs, n_boot),
           "compute_cost": compute_cost(seeds, metas, outs),
           "suite": suite(seeds, metas, outs, n_boot),
           "landscape": landscape(seeds, metas, outs),
           "mutation_sensitivity": mutation_sensitivity(seeds, metas, outs, n_boot),
           "threshold_sensitivity": threshold_sensitivity(conf),
           "longer_B": longer_b(seeds, metas, outs),
           "stage1_winners": stage1_winners(seeds, metas, outs),
           "scratch_B": scratch_b(seeds, metas, outs, n_boot),
           "gradients": gradients(seeds, metas, outs, n_boot),
           "first_gen_deaths": first_gen_deaths(seeds, outs),
           "energy_use": energy_use(seeds, outs),
           "achecks": achecks(seeds, metas, outs, rundir),
           "a_ties_in_b": a_ties_in_b(seeds, outs),
           "provenance": {"git": config.git_commit(), "seeds": seeds}}
    outdir = os.path.join(rundir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "exploratory.json"), "w") as f:
        json.dump(res, f, indent=1, default=A.jsonable)
    np.savez_compressed(os.path.join(outdir, "series.npz"), **curves(seeds, outs))
    print("written", os.path.relpath(outdir, ROOT))


if __name__ == "__main__":
    main()
