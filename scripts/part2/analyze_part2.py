"""Part 2 confirmatory analysis and pilot checks (docs/PREREG_part2.md, Sections 5-10).

Usage:
  python scripts/part2/analyze_part2.py --run pilot --pilot     # pilot report (Section 10)
  python scripts/part2/analyze_part2.py --run final              # confirmatory results

Reads results/part2/<run>/seed*.{npz,json} (only seeds with a .done file);
writes results/part2/<run>/analysis/ (outcomes.csv, checks.csv, confirmatory.json
or pilot.json). Every number comes from the saved runs.
"""
import argparse
import csv
import glob
import json
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, p2metrics as M, p2stats as S  # noqa: E402

COMPARISONS = {1: ("GS_food_frozen", "ET"), 2: ("GS_pop_frozen", "ET"), 3: ("GS_pop_mut", "ET"),
               4: ("SV_frozen", "ET"), 5: ("ET_freezeB", "ET"), 6: ("ET_cloneB", "ET_freezeB"),
               7: ("GS_pop_frozen", "GS_food_frozen"), 8: ("GS_pop_mut", "GS_pop_frozen")}
ARMS = {"main": {"world": "main", "cap": 25, "k": 16, "ks": (16,), "comps": list(range(1, 9)), "id": 1},
        "arm2": {"world": "arm2", "cap": None, "k": 14, "ks": (14, 16), "comps": [1, 2, 3, 4, 5, 6, 8], "id": 2},
        "arm3": {"world": "arm3", "cap": 25, "k": 16, "ks": (16,), "comps": list(range(1, 9)), "id": 3},
        "arm4": {"world": "arm4", "cap": 25, "k": 16, "ks": (16,), "comps": list(range(1, 9)), "id": 4}}
# Which B search and U2 runs each arm uses. Arm 2 shares the main B's (clones aren't affected by
# blob-eating); arms 3 and 4 have their own (Sections 9.2, 9.3).
U2_WORLD = {"main": "main", "arm2": "main", "arm3": "arm3", "arm4": "arm4"}


def arms_present(seeds, metas):
    """Arms in this run: main and arm 2 always; arm 3 and arm 4 when every seed has their B search."""
    return ["main", "arm2"] + [a for a in ("arm3", "arm4") if all(a in metas[s]["u2"] for s in seeds)]
CLONE_GROUPS = ("GS_food_frozen", "GS_pop_frozen", "ET_cloneB", "ORACLE_B")
U2_ROWS = (2, 5, 6, 8)

TEXT = {  # Section 7, the guide made exact: (first better, second better, the same)
    1: ("The 2025-style search beats evolution here; the 2025 claim fails",
        "Evolution beats the 2025-style search, as in 2025", "The two do equally well"),
    2: ("A fixed design aimed at the right goal beats evolution", "Evolution beats the best fixed design",
        "Being able to change doesn't help a well-chosen design here"),
    3: ("Starting at the A optimum and evolving beats starting simple",
        "Evolving from a simple start beats starting at the optimum and evolving",
        "Once both can evolve, the starting point doesn't matter"),
    4: ("Variety without change beats evolution", "Evolution beats variety without change",
        "Starting variety alone is enough, and new mutations aren't needed"),
    5: ("Stopping mutation after the move helps", "New mutations after the move help",
        "New mutations after the move don't matter"),
    6: ("Copies of one typical member do better than the varied population",
        "Variety built up in A helps after the move",
        "Copies of one typical member do as well; variety from A doesn't matter"),
    7: ("Aiming at population beats aiming at food per blob: part of the 2025 result came from the wrong goal",
        "Aiming at food per blob does better", "The search's goal doesn't matter"),
    8: ("Being able to change helps a good fixed design", "Being able to change hurts",
        "Being able to change doesn't matter for this design"),
}
ROW4_SV1 = ("The design that selection in A picked from 50 random starts beats evolution",
            "Evolution beats the design that selection in A picked from 50 random starts",
            "The design that selection in A picked from 50 random starts does as well as evolution")


# ------------------------------------------------------------------ loading
def load_run(rundir):
    with open(os.path.join(rundir, "run.json")) as f:
        run_cfg = json.load(f)["provenance"]["config"]
    n_a = int(run_cfg["phases"]["train_generations"])
    seeds, metas, outs = [], {}, {}
    for done in sorted(glob.glob(os.path.join(rundir, "seed*.done"))):
        seed = int(os.path.basename(done)[4:8])
        with open(os.path.join(rundir, f"seed{seed:04d}.json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(rundir, f"seed{seed:04d}.npz"))
        arrays = {k: z[k] for k in z.files}
        o = {}
        for pref in sorted({k.rsplit("__g_", 1)[0] for k in arrays if "__g_" in k}):
            parts = pref.split("__")
            if parts[0] == "B":
                food = 25
                world_name = parts[1]
                if world_name.startswith("food"):
                    food = int(world_name[4:])
                long_b = int(np.sum(arrays[pref + "__g_gen"] >= n_a)) >= 200
                o[pref] = M.outcomes(M.gen_arrays(arrays, pref), food_count=food,
                                     late=(190, 199) if long_b else None, n_a=n_a)
            elif parts[0] == "A":
                g = M.gen_arrays(arrays, pref)
                gp = np.asarray(g["population"], np.int64)
                gg = np.asarray(g["gen"], np.int64)
                win = (gg >= 10) & (gg <= n_a - 1)
                o[pref] = {"A_mean_pop_10_99": M.a_mean_population(g, 10, n_a - 1),
                           "A_mean_pop_10_99_exact": Fraction(int(gp[win].sum()), int(win.sum())),
                           "A_final_pop": int(np.asarray(g["population"])[-1]),
                           "steps_A": int(np.asarray(g["creature_steps"]).sum())}
        seeds.append(seed)
        metas[seed] = meta
        outs[seed] = o
        outs[seed]["_arrays"] = arrays
    return seeds, metas, outs


def series(seeds, outs, world, group, key):
    return [outs[s][f"B__{world}__{group}"][key] for s in seeds]


# ------------------------------------------------------------------ confirmatory
def analyze_arm(arm, seeds, metas, outs, n_perm=10000, n_boot=20000):
    A = ARMS[arm]
    w = A["world"]
    raw, s5_all, b0_mean, aphase, reached = {}, {}, {}, {}, {}
    for c in A["comps"]:
        g1, g2 = COMPARISONS[c]
        x1, y1 = series(seeds, outs, w, g1, "P1_total"), series(seeds, outs, w, g2, "P1_total")
        x2, y2 = series(seeds, outs, w, g1, "P2"), series(seeds, outs, w, g2, "P2")
        raw[c] = S.compare(x1, y1, x2, y2, A["ks"], A["id"], c, n_perm=n_perm, n_boot=n_boot)
        s5x, s5y = series(seeds, outs, w, g1, "S5"), series(seeds, outs, w, g2, "S5")
        s5_all[c] = (all(s5x), all(s5y))
        b0_mean[c] = (float(np.mean(series(seeds, outs, w, g1, "B0_population"))),
                      float(np.mean(series(seeds, outs, w, g2, "B0_population"))))
        aphase[c] = S.a_phase_label(s5x, s5y)
        both = [i for i in range(len(seeds)) if s5x[i] and s5y[i]]
        reached[c] = {"n_both": len(both),
                      "P1": (Fraction(sum(x1[i] - y1[i] for i in both), 10 * len(both)) if both else Fraction(0)),
                      "P2": (Fraction(sum(int(x2[i]) - int(y2[i]) for i in both), len(both)) if both else Fraction(0)),
                      "reached_B": (int(sum(s5x)), int(sum(s5y))),
                      "extinct_in_A": (int(sum(1 for i in range(len(seeds)) if not s5x[i])),
                                       int(sum(1 for i in range(len(seeds)) if not s5y[i])))}
    caps = {c: A["cap"] for c in A["comps"]}
    dec = S.decide_family(raw, A["k"], caps)
    for c in dec:
        dec[c]["a_phase"] = bool(aphase[c])
        dec[c]["arose_in_A"] = (S.arose_in_A(dec[c]["overall"], dec[c]["verdict"], raw[c],
                                             reached[c], reached[c]["n_both"]) if aphase[c] else None)
    control = S.positive_control(dec, s5_all, b0_mean)
    # U2 for this arm's B (arm 2 shares the main B search: clones aren't affected by blob-eating)
    u2_diff = [u2_exact(metas[s]["u2"][U2_WORLD[arm]]) for s in seeds]
    u2 = S.u2_fires(u2_diff)
    sv_single = sum(1 for s in seeds if metas[s]["distinct_designs_B0"]["SV_frozen"] == 1) * 2 > len(seeds)
    rows = guide_rows(dec, raw, control, u2, sv_single)
    combined = combined_rows(dec, control)
    return {"raw": raw, "decided": dec, "positive_control": control, "u2_fires": u2, "u2_diff": u2_diff,
            "s5_all": s5_all, "b0_mean": b0_mean, "reached_B_analysis": reached,
            "sv_single_design_majority": sv_single, "guide": rows, "combined": combined}


def u2_exact(u):
    """B winner minus A winner over the U2 sets, as an exact fraction."""
    return Fraction(int(u["b_winner_total"]) - int(u["a_winner_total"]), int(u["count"]))


def u3_pairs(seeds, metas, outs):
    st = [Fraction(*metas[s]["search_A"]["pop_winner_stage2_exact"]) for s in seeds]
    dep = [outs[s]["A__GS_pop_frozen"]["A_mean_pop_10_99_exact"] for s in seeds]
    return st, dep


def guide_rows(dec, raw, control, u2, sv_single):
    out = {}
    for c, d in dec.items():
        first, second, same = TEXT[c]
        if c == 4 and sv_single:
            first, second, same = ROW4_SV1
        ov, lab = d["overall"], d["p1_label"]
        g1, g2 = COMPARISONS[c]
        if ov in ("mixed", "inconclusive"):
            txt = "No conclusion"
        elif ov == "first group better":
            txt = first
        elif ov == "second group better":
            txt = second
        else:
            # Section 7: floor and ceiling texts are given "instead"; plain "the same" and
            # "capped on one side" need the positive control, and U2 not fired where marked.
            if lab == "at the floor":
                txt = "Both groups did about as badly as possible; no conclusion"
            elif lab == "at the ceiling":
                txt = "Both groups did about as well as this B allows; this B can't show whether one is better"
            elif not control:
                txt = "No conclusion (this B does not separate any groups)"
            elif c in U2_ROWS and u2:
                txt = "No conclusion (B doesn't call for a different design)"
            elif lab == "capped on one side":
                hi_first = raw[c]["P1"]["mean_x"] >= raw[c]["P1"]["mean_y"]
                lower, higher = (g2, g1) if hi_first else (g1, g2)
                txt = f"{lower} is not meaningfully worse than {higher}; this B can't show whether it is better"
            else:
                txt = same
        if d["a_phase"] and ov in ("first group better", "second group better", "mixed") and txt != "No conclusion":
            txt += (", but the difference arose in A, before the move" if d["arose_in_A"]
                    else " (part of this difference happened before the move)")
        out[c] = txt
    return out


def combined_rows(dec, control):
    """Section 7: the plan's combined rows. "SV_frozen ≈ ET" is row 4 giving a
    sameness conclusion: "the same" (or capped on one side) with the positive
    control holding; a ceiling or floor label gives no such conclusion."""
    def ov(c):
        return dec[c]["overall"] if c in dec else None
    same3 = dec.get(3, {}).get("overall") == "the same" and dec[3]["p1_label"] != "at the floor"
    both = (ov(2) == "second group better" and ov(4) == "second group better"
            and ov(8) == "first group better" and same3)
    lab3 = dec.get(3, {}).get("p1_label")
    return {"ET ≈ GS_pop_mut, and both > frozen groups": bool(both),
            "label": lab3 if both else None,
            "SV_frozen ≈ ET": bool(ov(4) == "the same" and control
                                   and dec[4]["p1_label"] not in ("at the floor", "at the ceiling"))}


def holds_changes(res_main, res_arm, arm):
    """Section 9.1: verdict categories at a fixed per-test level (k = 16)."""
    caps_m = {c: 25 for c in res_main["raw"]}
    caps_a = {c: ARMS[arm]["cap"] for c in res_arm["raw"]}
    dm = S.decide_family(res_main["raw"], 16, caps_m, fixed_level=True)
    da = S.decide_family(res_arm["raw"], 16, caps_a, fixed_level=True)
    out = {}
    for c in res_arm["raw"]:
        out[c] = {"main": dm[c]["overall"], arm: da[c]["overall"],
                  "result": "holds" if dm[c]["overall"] == da[c]["overall"] else "changes"}
    if arm == "arm2":
        out[7] = {"result": "identical in both arms by construction (clone groups only)"}
    return out


# ------------------------------------------------------------------ checks
def checks_table(seeds, metas, outs):
    rows = []
    for s in seeds:
        m = metas[s]
        sa = m["search_A"]
        dep = outs[s]["A__GS_pop_frozen"]["A_mean_pop_10_99"]
        r = {"seed": s, "tied_stage1_pop": sa["tied_stage1_pop"], "tied_stage1_food": sa["tied_stage1_food"],
             "tied_stage2_pop": sa["tied_stage2_pop"], "tied_stage2_food": sa["tied_stage2_food"],
             "pop_winner_stage2_score": sa["pop_winner_stage2_score"], "GS_pop_frozen_A_mean_10_99": dep,
             "u3_shortfall": float(Fraction(*sa["pop_winner_stage2_exact"])
                                   - outs[s]["A__GS_pop_frozen"]["A_mean_pop_10_99_exact"]),
             "winners_same": sa["pop_winner"] == sa["food_winner"],
             "steps_search_A": sa["steps_stage1"] + sa["steps_stage2"],
             "process_conf": m["process_seconds_conf"], "process_expl": m["process_seconds_expl"],
             "landscape_seconds": m["process_seconds"].get("expl:landscape", 0.0),
             "sv_distinct_B0": m["distinct_designs_B0"]["SV_frozen"],
             "et_distinct_B0": m["distinct_designs_B0"]["ET"],
             "gs_pop_mut_distinct_B0": m["distinct_designs_B0"]["GS_pop_mut"]}
        for w in m["u2"]:
            r[f"u2_{w}"] = float(u2_exact(m["u2"][w]))
            r[f"steps_search_B_{w}"] = m["search_B"][w]["steps_stage1"] + m["search_B"][w]["steps_stage2"]
        rows.append(r)
    return rows


def clone_check(seeds, outs):
    """Section 9.1: clone groups identical in arm 2 and the main B (first 50 B generations)."""
    bad = []
    for s in seeds:
        a = outs[s]["_arrays"]
        for g in CLONE_GROUPS:
            pm, p2 = f"B__main__{g}__", f"B__arm2__{g}__"
            gm = a[pm + "g_gen"]
            last = int(a[p2 + "g_gen"].max())
            km = gm <= last
            for k in [k for k in a if k.startswith(p2)]:
                field = k[len(p2):]
                vm = a[pm + field]
                if field.startswith("g_"):
                    vm = vm[km]
                else:
                    vm = vm[a[pm + "b_gen"] <= last]
                if not np.array_equal(vm, a[k]):
                    bad.append((s, g, field))
                    break
    return bad


def u1_shares(seeds, outs, world, groups, cap=25):
    out = {}
    for g in groups:
        p1 = np.array(series(seeds, outs, world, g, "P1"))
        out[g] = {"share_P1_ge_cap_minus_2.5": float(np.mean(p1 >= cap - 2.5)),
                  "share_P1_le_2.5": float(np.mean(p1 <= 2.5)), "mean_P1": float(p1.mean()),
                  "sd_P1": float(p1.std(ddof=1)) if len(p1) > 1 else float("nan"),
                  "extinct": int(sum(series(seeds, outs, world, g, "P2"))),
                  "died_in_A": int(sum(not x for x in series(seeds, outs, world, g, "S5"))),
                  "n": len(p1)}
    return out


def pilot_separation(seeds, outs, world, comps):
    """Section 10's pilot version of the positive control (no tests)."""
    hits = []
    for c in comps:
        g1, g2 = COMPARISONS[c]
        s5 = series(seeds, outs, world, g1, "S5") + series(seeds, outs, world, g2, "S5")
        b0 = (np.mean(series(seeds, outs, world, g1, "B0_population")),
              np.mean(series(seeds, outs, world, g2, "B0_population")))
        if not all(s5) or min(b0) < 40:
            continue
        t1 = sum(series(seeds, outs, world, g1, "P1_total"))
        t2 = sum(series(seeds, outs, world, g2, "P1_total"))
        d2 = abs(sum(series(seeds, outs, world, g1, "P2")) - sum(series(seeds, outs, world, g2, "P2")))
        if abs(t1 - t2) >= 25 * len(seeds) or d2 >= 3:  # exact: mean gap >= 2.5 blobs
            hits.append({"comparison": c, "P1_gap": abs(t1 - t2) / (10 * len(seeds)), "extinction_gap": int(d2)})
    return hits


def pilot_report(rundir, seeds, metas, outs):
    ck = checks_table(seeds, metas, outs)
    n_cpu = 2
    conf = np.array([r["process_conf"] for r in ck])
    expl = np.array([r["process_expl"] - r["landscape_seconds"] for r in ck])
    land = [r["landscape_seconds"] for r in ck if r["landscape_seconds"] > 0]
    proj = {f"confirmatory_N{n}_hours": float(conf.mean() * n / n_cpu / 3600) for n in (100, 200)}
    proj["exploratory_100_seeds_hours"] = float(expl.mean() * 100 / n_cpu / 3600)
    proj["landscape_10_seeds_hours"] = float(np.mean(land) * 10 / n_cpu / 3600) if land else None
    n_choice = 200 if proj["confirmatory_N200_hours"] <= 24 else (100 if proj["confirmatory_N100_hours"] <= 24 else None)
    worlds = arms_present(seeds, metas)  # each arm's B world has the arm's name
    u3 = S.u3_fires(*u3_pairs(seeds, metas, outs))
    rep = {"seeds": seeds, "n": len(seeds), "projection": proj, "N_choice_by_rule": n_choice,
           "u3": {"fires": u3, "mean_shortfall": float(np.mean([r["u3_shortfall"] for r in ck]))},
           "u2": {}, "u1_u4": {}, "separation": {}, "clone_check_mismatches": clone_check(seeds, outs),
           "ties": {k: [r[k] for r in ck] for k in ("tied_stage1_pop", "tied_stage1_food",
                                                     "tied_stage2_pop", "tied_stage2_food")},
           "sv_distinct_B0": [r["sv_distinct_B0"] for r in ck], "checks": ck}
    groups = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen", "GS_pop_mut",
              "SV_frozen", "ORACLE_B")
    for w in worlds:
        d = [u2_exact(metas[s]["u2"][U2_WORLD[w]]) for s in seeds]
        rep["u2"][w] = {"fires": S.u2_fires(d), "n_below_2.5": int(sum(x < Fraction(5, 2) for x in d)),
                        "diffs": [float(x) for x in d]}
        rep["u1_u4"][w] = u1_shares(seeds, outs, w, groups, cap=25)
        rep["separation"][w] = pilot_separation(seeds, outs, w, ARMS[w]["comps"])
    u4 = rep["u1_u4"]["main"]["GS_pop_frozen"]["died_in_A"] * 2 > len(seeds)
    triggers = {
        "U2 fires (at least half the pilot runs)": {w: rep["u2"][w]["fires"] for w in rep["u2"] if w != "arm2"},
        "B separates no groups": {w: len(rep["separation"][w]) == 0 for w in worlds},
        "U4 fires for GS_pop_frozen": bool(u4),
        "projected confirmatory time at N=100 over 24 h": proj["confirmatory_N100_hours"] > 24,
        "clone check failed": len(rep["clone_check_mismatches"]) > 0,
    }
    rep["stop_and_ask_triggers"] = triggers
    rep["stop_and_ask"] = any(v if isinstance(v, bool) else any(v.values()) for v in triggers.values())
    return rep


# ------------------------------------------------------------------ output
def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def jsonable(o):
    if isinstance(o, Fraction):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer permutations/resamples (testing only)")
    args = ap.parse_args()
    rundir = os.path.join(ROOT, "results", "part2", args.run)
    seeds, metas, outs = load_run(rundir)
    if not seeds:
        raise SystemExit("no finished seeds")
    outdir = os.path.join(rundir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for s in seeds:
        for pref, o in outs[s].items():
            if pref.startswith("B__"):
                _, w, g = pref.split("__")
                rows.append({"seed": s, "world": w, "group": g, **{k: v for k, v in o.items()}})
    write_csv(os.path.join(outdir, "outcomes.csv"), rows)
    write_csv(os.path.join(outdir, "checks.csv"), checks_table(seeds, metas, outs))
    prov = {"git": config.git_commit(), "seeds": seeds}
    if args.pilot:
        rep = pilot_report(rundir, seeds, metas, outs)
        rep["provenance"] = prov
        with open(os.path.join(outdir, "pilot.json"), "w") as f:
            json.dump(rep, f, indent=1, default=jsonable)
        print(json.dumps({k: rep[k] for k in ("projection", "N_choice_by_rule", "u3", "stop_and_ask_triggers",
                                              "stop_and_ask", "clone_check_mismatches")}, indent=1, default=jsonable))
        for w, d in rep["u1_u4"].items():
            print(w, {g: (round(v["mean_P1"], 2), round(v["sd_P1"], 2), v["extinct"], v["died_in_A"]) for g, v in d.items()})
        print("separation", rep["separation"])
        print("u2", {w: (v["n_below_2.5"], [round(x, 2) for x in v["diffs"]]) for w, v in rep["u2"].items()})
        print("ties", rep["ties"])
        print("SV distinct designs at B0", rep["sv_distinct_B0"])
        return
    n_perm, n_boot = (1000, 2000) if args.quick else (10000, 20000)
    res = {}
    for arm in arms_present(seeds, metas):
        res[arm] = analyze_arm(arm, seeds, metas, outs, n_perm, n_boot)
    res["holds_changes"] = {a: holds_changes(res["main"], res[a], a) for a in res if a in ("arm2", "arm3", "arm4")}
    ck = checks_table(seeds, metas, outs)
    res["U3"] = {"fires": S.u3_fires(*u3_pairs(seeds, metas, outs)),
                 "mean_shortfall": float(np.mean([r["u3_shortfall"] for r in ck]))}
    res["U1_U4"] = {a: u1_shares(seeds, outs, ARMS[a]["world"], [g for c in ARMS[a]["comps"] for g in COMPARISONS[c]] + ["ORACLE_B"])
                    for a in res if a in ARMS}
    res["clone_check_mismatches"] = clone_check(seeds, outs)
    res["provenance"] = prov
    with open(os.path.join(outdir, "confirmatory.json"), "w") as f:
        json.dump(res, f, indent=1, default=jsonable)
    for arm in [a for a in res if a in ARMS]:
        print(f"== {arm}: positive control {res[arm]['positive_control']}, U2 {res[arm]['u2_fires']}")
        for c, d in res[arm]["decided"].items():
            print(f"  {c} {COMPARISONS[c]}: {d['overall_label']} | P1 {d['verdict']['P1']} "
                  f"({float(res[arm]['raw'][c]['P1']['diff']):+.2f}) | P2 {d['verdict']['P2']} "
                  f"({float(res[arm]['raw'][c]['P2']['diff']):+.2f}) -> {res[arm]['guide'][c]}")


if __name__ == "__main__":
    main()
