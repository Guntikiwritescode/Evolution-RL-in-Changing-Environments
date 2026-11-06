"""Section 18.3: check the Part 2 analysis code on made-up data (seed 2950).

Usage:
  python scripts/part2/validate_analysis.py            # all checks (about 30-60 min on 2 CPUs)
  python scripts/part2/validate_analysis.py --quick    # tiny version, for testing the script
  python scripts/part2/validate_analysis.py --only V2  # one check

Writes results/part2/validation/<check>.json (one file per check, so a stopped
session keeps what is done) and prints a pass/fail line per check.
No simulator is used. Pass rules are the ones written in PREREG_part2.md 18.3.
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, p2metrics, p2stats as S, sim2, streams, world  # noqa: E402

SEED = 2950
OUT = os.path.join(ROOT, "results", "part2", "validation")


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=tuple(int(k) for k in key)))


GIT = {}  # provenance taken once, at start: the code under test is what was loaded then


def save(name, obj, quick):
    os.makedirs(OUT, exist_ok=True)
    obj["git"] = GIT
    obj["quick"] = quick
    path = os.path.join(OUT, f"{name}{'_quick' if quick else ''}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, Fraction) else str(o))
    os.replace(tmp, path)
    print(f"{name}: {'PASS' if obj['pass'] else 'FAIL'}  {obj.get('summary', '')}", flush=True)


# ------------------------------------------------------------------ data makers
KIND_ID = {"et": 1, "spread": 2, "crash": 3, "normal04": 4, "normal5": 5, "zeros": 6,
           "crashes": 7, "P2_05_15": 8, "P2_00_10": 9}  # stable ids for stream keys
_CAPPED = []


def capped_mean():
    """E[min(N(24.3, 0.4), 25)], estimated once from 2 million draws."""
    if not _CAPPED:
        _CAPPED.append(float(np.minimum(rng_for(99).normal(24.3, 0.4, 2_000_000), 25.0).mean()))
    return _CAPPED[0]


def p1_totals(kind, n, rng, shift=0.0):
    """Per-run P1 totals (integers = 10 x P1)."""
    if kind == "et":
        v = np.minimum(rng.normal(24.3, 0.4, n), 25.0)
    elif kind == "spread":
        v = np.clip(rng.normal(15.0, 5.0, n), 0.0, 25.0)
    elif kind == "crash":
        v = np.minimum(rng.normal(24.3, 0.4, n), 25.0)
        v[rng.random(n) < 0.05] = 0.0
    elif kind == "normal04":
        v = rng.normal(20.0, 0.4, n)
    elif kind == "normal5":
        v = rng.normal(15.0, 5.0, n)
    else:
        raise KeyError(kind)
    return np.rint((v - shift) * 10).astype(np.int64)


def boundary_pair(kind, n, rng):
    """Two P1 samples whose true mean difference is 2.5 (Section 18.3, V2)."""
    if kind in ("normal04", "normal5"):
        return p1_totals(kind, n, rng), p1_totals(kind, n, rng, shift=2.5)
    x = np.minimum(rng.normal(24.3, 0.4, n), 25.0)
    y = np.minimum(rng.normal(24.3, 0.4, n), 25.0)
    big = capped_mean()
    if kind == "zeros":
        y[rng.random(n) < 2.5 / big] = 0.0
    elif kind == "crashes":
        m = rng.random(n) < 2.5 / (big - 5.0)
        y[m] = rng.uniform(0.0, 10.0, int(m.sum()))
    else:
        raise KeyError(kind)
    return np.rint(x * 10).astype(np.int64), np.rint(y * 10).astype(np.int64)


# ------------------------------------------------------------------ V0
def v0(quick):
    res = []

    def g_rows(pops, surv, births, prey=None, b_from=100):
        n = len(pops)
        return {"gen": np.arange(b_from, b_from + n), "population": np.array(pops),
                "survivors": np.array(surv), "births": np.array(births),
                "food_eaten": np.array(pops), "prey_eaten": np.zeros(n, int) if prey is None else np.array(prey),
                "creature_steps": np.ones(n, int)}

    # died in A: every B generation empty
    o = p2metrics.outcomes(g_rows([0] * 50, [0] * 50, [0] * 50))
    res.append(("died in A", o["P1_total"] == 0 and o["P2"] and not o["S5"]))
    # last generation dies: population 5 at B gen 49 but no survivors or births
    pops = [20] * 49 + [5]
    surv = [20] * 49 + [0]
    o = p2metrics.outcomes(g_rows(pops, surv, [0] * 50))
    res.append(("dies in last generation", o["P2"] and o["P1_total"] == 20 * 9 + 5 and o["S5"]))
    # survives: survivors in gen 49
    o = p2metrics.outcomes(g_rows([20] * 50, [10] * 50, [3] * 50))
    res.append(("survives", (not o["P2"]) and o["P1_total"] == 200))
    # S6 = mean of B gens 1-49
    pops = list(range(50))
    o = p2metrics.outcomes(g_rows(pops, [1] * 50, [0] * 50))
    res.append(("S6", abs(o["S6"] - np.mean(range(1, 50))) < 1e-12 and o["S1"] == 23))
    # clone pick = the documented stream
    pop = {t: np.arange(7.0) + 1 for t in world.TRAITS}
    pop.update(flee=np.ones(7), home_x=np.zeros(7), home_y=np.zeros(7), age=np.zeros(7, int),
               uid=np.arange(7), parent=np.full(7, -1), next_uid=7)
    i = sim2.pick_clone_index(pop, 2001, 0)
    j = int(streams.rng(2001, "clone_pick", 0).integers(0, 7))
    res.append(("clone pick", i == j))
    # selection fitness
    b = {"alive_end": np.array([1, 1, 0, 1, 1]), "food_balls": np.array([2, 1, 2, 0, 1]),
         "prey": np.array([0, 1, 0, 0, 0])}
    res.append(("fitness", list(p2metrics.blob_fitness(b)) == [2, 2, 0, 1, 1]))
    ok = all(r[1] for r in res)
    save("V0", {"pass": ok, "cases": res, "summary": f"{sum(r[1] for r in res)}/{len(res)} cases"}, quick)


# ------------------------------------------------------------------ V1 / V1b
def _family(args):
    scen, n, fam, n_perm, n_effect = args
    r = rng_for(1, KIND_ID[scen], n, fam)
    ps, which = [], []
    for c in range(8):
        eff = c < n_effect
        if scen == "et":
            x, y = p1_totals("et", n, r), p1_totals("et", n, r)
            px, py = r.random(n) < 0.02, r.random(n) < 0.02
        elif scen == "spread":
            x, y = p1_totals("spread", n, r), p1_totals("spread", n, r)
            px, py = r.random(n) < 0.2, r.random(n) < 0.2
        else:
            x, y = p1_totals("crash", n, r), p1_totals("crash", n, r)
            px, py = r.random(n) < 0.02, r.random(n) < 0.02
        if eff:
            y = y - 60  # 6 blobs lower
            py = r.random(n) < 0.5
        p1 = S.perm_test_mean(x / 10, y / 10, rng_for(2, KIND_ID[scen], n, fam, c), n_perm)
        p2 = S.fisher_p(int(px.sum()), n, int(py.sum()), n)
        ps += [p1, p2]
        which += [eff, eff]
    adj = S.holm(ps)
    null_sig = [a < S.ALPHA for a, e in zip(adj, which) if not e]
    return any(null_sig)


def v1(quick, pool):
    n_fam = 40 if quick else 1000
    n_perm = 200 if quick else 2000
    rows = {}
    ok = True
    for scen in ("et", "spread", "crash"):
        for n in (100, 200):
            t0 = time.time()
            hits = pool.map(_family, [(scen, n, f, n_perm, 0) for f in range(n_fam)], chunksize=10)
            rate = float(np.mean(hits))
            rows[f"{scen}_n{n}"] = {"families_with_false_difference": rate, "seconds": time.time() - t0}
            ok &= rate <= 0.064
    save("V1", {"pass": ok, "rows": rows, "n_families": n_fam, "n_perm": n_perm,
                "rule": "any Holm-significant test among 16 null tests in at most 6.4% of families",
                "summary": str({k: round(v["families_with_false_difference"], 3) for k, v in rows.items()})}, quick)


def v1b(quick, pool):
    n_fam = 40 if quick else 1000
    n_perm = 200 if quick else 2000
    rows, ok = {}, True
    for n in (100, 200):
        hits = pool.map(_family, [("et", n, f, n_perm, 2) for f in range(n_fam)], chunksize=10)
        rate = float(np.mean(hits))
        rows[f"n{n}"] = rate
        ok &= rate <= 0.064
    save("V1b", {"pass": ok, "rows": rows, "n_families": n_fam,
                 "rule": "2 comparisons (4 tests) with large true differences; the 12 null tests give a false difference in at most 6.4% of families",
                 "summary": str(rows)}, quick)


# ------------------------------------------------------------------ V2
def _boundary(args):
    kind, n, i, n_boot = args
    r = rng_for(3, KIND_ID[kind], n, i)
    lv = S.eq_level(16)
    if kind.startswith("P2"):
        a, b = (0.05, 0.15) if kind == "P2_05_15" else (0.0, 0.10)
        x, y = int((r.random(n) < a).sum()), int((r.random(n) < b).sum())
        lo, hi = S.newcombe(x, n, y, n, lv)
        return lo >= -0.1 - S.TOL and hi <= 0.1 + S.TOL
    x, y = boundary_pair(kind, n, r)
    ci = S.boot_t_ci(x / 10, y / 10, rng_for(4, KIND_ID[kind], n, i), (lv,), n_boot)[lv]
    return ci[0] >= -2.5 - S.TOL and ci[1] <= 2.5 + S.TOL


def v2(quick, pool):
    n_sets = 30 if quick else 1000
    n_boot = 2000 if quick else 20000
    rows, ok = {}, True
    for kind in ("normal04", "normal5", "zeros", "crashes", "P2_05_15", "P2_00_10"):
        for n in (100, 200):
            t0 = time.time()
            hits = pool.map(_boundary, [(kind, n, i, n_boot) for i in range(n_sets)], chunksize=5)
            rate = float(np.mean(hits))
            rows[f"{kind}_n{n}"] = {"equivalent_rate": rate, "seconds": time.time() - t0}
            ok &= rate <= 0.01
    save("V2", {"pass": ok, "rows": rows, "n_sets": n_sets, "n_boot": n_boot,
                "rule": "true difference exactly on the threshold: 'equivalent' in at most 1%",
                "summary": str({k: v["equivalent_rate"] for k, v in rows.items()})}, quick)


# ------------------------------------------------------------------ V3
def v3(quick):
    n_sets = 300 if quick else 5000
    rows, ok = {}, True
    for (a, b) in ((0.0, 0.15), (0.05, 0.20)):
        for n in (100, 200):
            r = rng_for(5, int(a * 100), int(b * 100), n)
            hits = 0
            for _ in range(n_sets):
                x, y = int((r.random(n) < a).sum()), int((r.random(n) < b).sum())
                p = S.fisher_p(x, n, y, n)
                if p < S.ALPHA / 16 and abs(Fraction(y - x, n)) >= Fraction(1, 10):
                    hits += 1
            rate = hits / n_sets
            exact = S.fisher_power(n, a, b, 16, require_estimate=True)
            se = np.sqrt(exact * (1 - exact) / n_sets)
            good = abs(rate - exact) <= 4 * se + 1e-12
            rows[f"{a}_vs_{b}_n{n}"] = {"simulated": rate, "exact": exact, "se": se, "within_4se": bool(good)}
            ok &= good
    save("V3", {"pass": ok, "rows": rows, "n_sets": n_sets,
                "summary": str({k: (round(v["simulated"], 3), round(v["exact"], 3)) for k, v in rows.items()})}, quick)


# ------------------------------------------------------------------ V4
def _r(diff, thr, ci, mx=None, my=None):
    d = Fraction(diff).limit_denominator(10000)
    return {"diff_exact": d, "diff": float(d), "thr": Fraction(thr), "ci_eq": {16: ci, 14: ci},
            "mean_x": Fraction(mx).limit_denominator(1000) if mx is not None else Fraction(0),
            "mean_y": Fraction(my).limit_denominator(1000) if my is not None else Fraction(0)}


def v4(quick):
    cases = []

    def check(name, got, want):
        cases.append((name, str(got), str(want), got == want))

    T1, T2 = Fraction(5, 2), Fraction(1, 10)
    # outcome verdict paths
    check("meaningful", S.outcome_verdict(_r(3, T1, (2.6, 3.4)), 0.001, 16), S.MEANINGFUL)
    check("equivalent", S.outcome_verdict(_r(0.1, T1, (-1, 1)), 0.4, 16), S.EQUIV)
    check("equivalent small real", S.outcome_verdict(_r(0.4, T1, (0.2, 0.6)), 0.001, 16), S.EQUIV_SMALL)
    check("size unclear", S.outcome_verdict(_r(2, T1, (0.5, 3.5)), 0.01, 16), S.UNCLEAR)
    check("inconclusive", S.outcome_verdict(_r(1, T1, (-3, 5)), 0.3, 16), S.INCONCL)
    check("exactly on threshold reaches", S.outcome_verdict(_r(Fraction(5, 2), T1, (2.4, 2.6)), 0.001, 16), S.MEANINGFUL)
    check("P2 21/200-1/200 reaches 0.10", abs(Fraction(21, 200) - Fraction(1, 200)) >= T2, True)
    check("CI end exactly on threshold counts as inside", S.outcome_verdict(_r(0, T1, (-2.5, 2.5)), 0.9, 16), S.EQUIV)
    # end to end: exact-threshold constant groups
    raw = S.compare(np.full(100, 250), np.full(100, 225), np.zeros(100, bool), np.zeros(100, bool),
                    (16,), 9, 1, n_perm=2000, n_boot=20000, seed=SEED)
    check("constant groups 2.5 apart: P1 meaningful", S.outcome_verdict(raw["P1"], raw["P1"]["p"], 16), S.MEANINGFUL)
    check("constant identical groups: p = 1", raw["P2"]["p"], 1.0)
    raw = S.compare(np.full(100, 240), np.full(100, 240), np.zeros(100, bool), np.zeros(100, bool),
                    (16,), 9, 2, n_perm=2000, n_boot=20000, seed=SEED)
    check("constant identical groups: equivalent", S.decide_family({1: raw}, 16, {1: 25})[1]["overall"], "the same")
    y = np.full(100, 250)
    y[0] = 249
    raw = S.compare(np.full(100, 250), y, np.zeros(100, bool), np.zeros(100, bool), (16,), 9, 3,
                    n_perm=2000, n_boot=20000, seed=SEED)
    check("one constant, one nearly constant: equivalent", S.outcome_verdict(raw["P1"], 0.9, 16), S.EQUIV)
    # labels
    check("ceiling", S.p1_label(_r(0, T1, (-1, 1), 24, 23), 25), "at the ceiling")
    check("capped one side", S.p1_label(_r(1.5, T1, (-1, 2), 23.5, 22), 25), "capped on one side")
    check("floor", S.p1_label(_r(0.2, T1, (-1, 1), 1.0, 0.8), 25), "at the floor")
    check("no ceiling label without a cap (arm 2)", S.p1_label(_r(0, T1, (-1, 1), 24, 23), None), None)
    check("default threshold: ceiling at 22.5", S.p1_label(_r(0, T1, (-1, 1), 23, 22.6), 25), "at the ceiling")
    check("half threshold moves the ceiling to 23.75", S.p1_label(_r(0, T1, (-1, 1), 23, 22.6), 25, Fraction(5, 4)), None)
    check("half threshold, one side", S.p1_label(_r(0, T1, (-1, 1), 24, 22.6), 25, Fraction(5, 4)), "capped on one side")
    # overall verdicts
    M, E, U = S.MEANINGFUL, S.EQUIV, S.UNCLEAR
    res = {"P1": _r(3, T1, (2, 4)), "P2": _r(Fraction(-1, 5), T2, (-0.3, -0.1))}
    check("first better (P1 up, P2 down)", S.overall_verdict(res, {"P1": M, "P2": M}), "first group better")
    res = {"P1": _r(3, T1, (2, 4)), "P2": _r(Fraction(1, 5), T2, (0.1, 0.3))}
    check("mixed (P1 up, P2 extinction up)", S.overall_verdict(res, {"P1": M, "P2": M}), "mixed")
    check("mixed (P2 worse but size unclear)", S.overall_verdict(res, {"P1": M, "P2": U}), "mixed")
    res = {"P1": _r(-3, T1, (-4, -2)), "P2": _r(0, T2, (-0.05, 0.05))}
    check("second better", S.overall_verdict(res, {"P1": M, "P2": E}), "second group better")
    check("the same", S.overall_verdict(res, {"P1": E, "P2": E}), "the same")
    check("inconclusive overall", S.overall_verdict(res, {"P1": S.INCONCL, "P2": E}), "inconclusive")
    # A-phase rules
    check("A-phase label on any difference", S.a_phase_label([1] * 99 + [0], [1] * 100), True)
    check("no A-phase label", S.a_phase_label([1] * 100, [1] * 100), False)
    rawA = {"P1": _r(-5, T1, (-6, -4)), "P2": _r(Fraction(2, 5), T2, (0.3, 0.5))}
    vA = {"P1": M, "P2": M}
    check("arose in A: under 30 seeds", S.arose_in_A("second group better", vA, rawA, {"P1": -5, "P2": Fraction(2, 5)}, 12), True)
    check("arose in A: reached-B estimates too small",
          S.arose_in_A("second group better", vA, rawA, {"P1": -1, "P2": Fraction(1, 50)}, 60), True)
    check("part before the move", S.arose_in_A("second group better", vA, rawA, {"P1": -3, "P2": 0}, 60), False)
    check("rule doesn't apply to 'the same'", S.arose_in_A("the same", vA, rawA, {}, 60), None)
    # positive control, U2, U3
    dec = {1: {"overall": "the same"}, 2: {"overall": "second group better"}}
    check("control holds", S.positive_control(dec, {1: (True, True), 2: (True, True)}, {1: (47, 50), 2: (47, 45)}), True)
    check("control fails: limping group", S.positive_control(dec, {1: (True, True), 2: (True, True)}, {1: (47, 50), 2: (47, 12)}), False)
    check("control fails: deaths in A", S.positive_control(dec, {1: (True, True), 2: (True, False)}, {1: (47, 50), 2: (47, 45)}), False)
    check("U2 fires at half", S.u2_fires([1, 1, 5, 5]), True)
    check("U2 does not fire", S.u2_fires([1, 5, 5, 5]), False)
    check("U3 fires", S.u3_fires([50, 50], [46, 47]), True)
    check("U3 does not fire", S.u3_fires([50, 50], [49, 48]), False)
    # arm 2 family of 14: Holm over 14 and no ceiling labels
    raws = {}
    r0 = rng_for(6, 1)
    for c in range(1, 8):
        x, y = p1_totals("et", 100, r0), p1_totals("et", 100, r0)
        raws[c] = S.compare(x, y, np.zeros(100, bool), np.zeros(100, bool), (14, 16), 2, c,
                            n_perm=1000, n_boot=20000, seed=SEED)
    d14 = S.decide_family(raws, 14, {c: None for c in raws})
    check("arm 2: no ceiling labels", all(d14[c]["p1_label"] in (None, "at the floor") for c in d14), True)
    v4_interpretation(check)
    ok = all(c[3] for c in cases)
    save("V4", {"pass": ok, "cases": cases, "summary": f"{sum(c[3] for c in cases)}/{len(cases)} cases"}, quick)


def _analysis_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze_part2", os.path.join(ROOT, "scripts", "part2", "analyze_part2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_run(n, group_maker, u2_b_minus_a_total=0, sv_distinct=5, world="main", u2_other=None):
    """Built outcome tables for analyze_arm: group_maker(group, i) -> (P1_total, P2, S5, B0).
    u2_other: {world: B-minus-A total} for the U2 runs of other searched B worlds (arms 3, 4)."""
    seeds = list(range(n))
    outs, metas = {}, {}
    groups = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen", "GS_pop_mut",
              "SV_frozen", "ORACLE_B")
    for i in seeds:
        o = {}
        for g in groups:
            t, p2, s5, b0 = group_maker(g, i)
            o[f"B__{world}__{g}"] = {"P1_total": int(t), "P1": t / 10, "P2": bool(p2), "S5": bool(s5),
                                     "B0_population": int(b0)}
        outs[i] = o
        metas[i] = {"u2": {"main": {"b_winner_total": 8000 + u2_b_minus_a_total, "a_winner_total": 8000,
                                    "count": 320}},
                    "distinct_designs_B0": {"SV_frozen": sv_distinct}}
        for w, d in (u2_other or {}).items():
            metas[i]["u2"][w] = {"b_winner_total": 8000 + d, "a_winner_total": 8000, "count": 320}
    return seeds, metas, outs


def v4_interpretation(check):
    """Section 18.3 V4, interpretation code in analyze_part2: reached-B analysis,
    'arose in A', positive control inputs, guide texts, U2, row-4 qualifier, combined rows."""
    A = _analysis_module()
    kw = dict(n_perm=2000, n_boot=20000)

    def r(i, g, base=243):
        return int(base + (rng_for(8, i, len(g)).integers(-4, 5)))

    # Case 1: GS_food_frozen dies in A in 40 of 100 runs, otherwise like ET
    def mk1(g, i):
        if g == "GS_food_frozen" and i < 40:
            return 0, True, False, 0
        return r(i, g), False, True, 47
    seeds, metas, outs = _fake_run(100, mk1)
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    d1 = res["decided"][1]
    check("A-phase: row 1 ET better", d1["overall"], "second group better")
    check("A-phase: label set", d1["a_phase"], True)
    check("A-phase: arose in A (reached-B gap is 0)", d1["arose_in_A"], True)
    check("A-phase: guide text ending", res["guide"][1].endswith("arose in A, before the move"), True)
    check("A-phase: reached-B seeds counted", res["reached_B_analysis"][1]["n_both"], 60)

    # Case 2: everything the same near the cap; control fails; ceiling text still given
    def mk2(g, i):
        return r(i, g), False, True, 47
    seeds, metas, outs = _fake_run(100, mk2)
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    check("ceiling: control fails", res["positive_control"], False)
    check("ceiling: row 5 overall", res["decided"][5]["overall_label"], "the same (at the ceiling)")
    check("ceiling text given even without the control", res["guide"][5].startswith("Both groups did about as well"), True)

    # Case 3: mid-range populations (no ceiling); one clear difference makes the control hold;
    # U2 fires, so rows 2, 5, 6, 8 give no conclusion; rows 3 and 4 give sameness
    def mk3(g, i):
        if g == "ET_cloneB":
            return r(i, g, 100), False, True, 47
        return r(i, g, 150), False, True, 47
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=100)
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    check("control holds (row 6 differs)", res["positive_control"], True)
    check("U2 fires", res["u2_fires"], True)
    check("U2 blocks row 5", res["guide"][5], "No conclusion (B doesn't call for a different design)")
    check("row 3 sameness text", res["guide"][3], A.TEXT[3][2])
    check("row 4 sameness text", res["guide"][4], A.TEXT[4][2])
    check("combined SV ≈ ET", res["combined"]["SV_frozen ≈ ET"], True)
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=800)  # B winner ahead by exactly 2.5
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    check("U2 exactly 2.5 ahead: does not fire", res["u2_fires"], False)
    check("row 5 sameness text when U2 not fired", res["guide"][5], A.TEXT[5][2])

    # Case 4: row-4 qualifier when SV_frozen has one design left in most runs
    seeds, metas, outs = _fake_run(100, mk3, sv_distinct=1)
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    check("row 4 qualifier", res["guide"][4], A.ROW4_SV1[2])

    # Case 5: combined row "ET ≈ GS_pop_mut, both > frozen"
    def mk5(g, i):
        low = g in ("GS_pop_frozen", "SV_frozen")
        return r(i, g, 150 if low else 220), False, True, 47
    seeds, metas, outs = _fake_run(100, mk5)
    res = A.analyze_arm("main", seeds, metas, outs, **kw)
    check("combined row holds", res["combined"]["ET ≈ GS_pop_mut, and both > frozen groups"], True)

    # Case 6 (added 2 Nov 2026 with arm 4): each arm reads the right U2 runs. Arms 3 and 4 use
    # their own; arm 2 uses the main B's. 100/320 blobs ahead fires U2; 1600/320 = 5 doesn't.
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=100, world="arm4", u2_other={"arm4": 1600})
    res = A.analyze_arm("arm4", seeds, metas, outs, **kw)
    check("arm 4 reads its own U2 (main's fires, arm 4's doesn't)", res["u2_fires"], False)
    check("arm 4 row 5 sameness text", res["guide"][5], A.TEXT[5][2])
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=1600, world="arm4", u2_other={"arm4": 100})
    res = A.analyze_arm("arm4", seeds, metas, outs, **kw)
    check("arm 4 U2 fires from its own runs (main's doesn't)", res["u2_fires"], True)
    check("arm 4 U2 blocks row 5", res["guide"][5], "No conclusion (B doesn't call for a different design)")
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=1600, world="arm3", u2_other={"arm3": 100, "arm4": 1600})
    res = A.analyze_arm("arm3", seeds, metas, outs, **kw)
    check("arm 3 reads its own U2", res["u2_fires"], True)
    seeds, metas, outs = _fake_run(100, mk3, u2_b_minus_a_total=100, world="arm2", u2_other={"arm3": 1600, "arm4": 1600})
    res = A.analyze_arm("arm2", seeds, metas, outs, **kw)
    check("arm 2 reads the main B's U2", res["u2_fires"], True)
    check("arms present: all four", A.arms_present(seeds, metas), ["main", "arm2", "arm3", "arm4"])
    del metas[seeds[-1]]["u2"]["arm4"]
    check("arms present: arm 4 left out if any seed lacks it", A.arms_present(seeds, metas), ["main", "arm2", "arm3"])


def v5(quick):
    r = rng_for(7, 1)
    base = {}
    for c in range(1, 9):
        x, y = p1_totals("et", 100, r), p1_totals("et", 100, r)
        base[c] = S.compare(x, y, r.random(100) < 0.02, r.random(100) < 0.02, (14, 16), 1, c,
                            n_perm=1000, n_boot=20000, seed=SEED)
    target = S.compare(p1_totals("et", 100, r), p1_totals("et", 100, r, shift=0.3),
                       np.zeros(100, bool), np.zeros(100, bool), (14, 16), 1, 99, n_perm=1000, n_boot=20000,
                       seed=SEED)
    fam1 = dict(base)
    fam1[99] = target
    fam2 = {c: base[c] for c in (1, 2, 3)}
    for c in (4, 5, 6, 7):  # strong effects elsewhere change Holm's steps
        fam2[c] = S.compare(p1_totals("et", 100, r), p1_totals("et", 100, r, shift=5), np.zeros(100, bool),
                            r.random(100) < 0.6, (14, 16), 1, 50 + c, n_perm=1000, n_boot=20000, seed=SEED)
    fam2[99] = target
    caps = {c: 25 for c in list(fam1) + list(fam2)}
    a = S.decide_family(fam1, 16, caps, fixed_level=True)[99]["overall_label"]
    b = S.decide_family(fam2, 16, caps, fixed_level=True)[99]["overall_label"]
    save("V5", {"pass": a == b, "verdicts": [a, b], "summary": f"{a} / {b}"}, quick)


def v6(quick):
    rows, ok = [], True

    def chk(name, got, want, places):
        good = round(got, places) == want
        rows.append((name, got, want, good))
        return good

    for (p1, p2, want) in ((0, .10, .55), (0, .15, .94), (.05, .20, .55), (.10, .30, .68)):
        ok &= chk(f"power n100 {p1} vs {p2}", S.fisher_power(100, p1, p2), want, 2)
    for (p1, p2, want) in ((0, .10, .534), (0, .15, .985), (.05, .20, .941), (.10, .30, .981)):
        ok &= chk(f"meaningful n200 {p1} vs {p2}", S.fisher_power(200, p1, p2), want, 3)
    ok &= chk("test power n200 0 vs 0.10", S.fisher_power(200, 0, .10, require_estimate=False), .996, 3)
    worst50 = max(S.fisher_power(50, a, b) for a, b in ((0, .10), (0, .15), (.05, .20), (.10, .30)))
    ok &= chk("n50 all at most 0.33", worst50, 0.33, 2) or worst50 <= 0.33
    lv = S.eq_level(16)
    for n, want in ((50, .13), (100, .07), (200, .036)):
        hw = S.newcombe(0, n, 0, n, lv)[1]
        ok &= chk(f"0 vs 0 half-width n{n}", hw, want, 2 if n < 200 else 3)
    eq100 = sorted((x, y) for x in range(0, 15) for y in range(0, 15)
                   if S.newcombe(x, 100, y, 100, lv)[0] >= -0.1 - S.TOL and S.newcombe(x, 100, y, 100, lv)[1] <= 0.1 + S.TOL)
    want100 = sorted({(a, a) for a in range(5)} | {(a, a + 1) for a in range(3)} | {(a + 1, a) for a in range(3)})
    rows.append(("P2 equivalence cases n100", eq100, want100, eq100 == want100))
    ok &= eq100 == want100
    eq200 = max(a for a in range(0, 101) if S.newcombe(a, 200, a, 200, lv)[0] >= -0.1 - S.TOL)
    rows.append(("P2 equivalence up to (equal counts) n200", eq200, 30, eq200 == 30))
    ok &= eq200 == 30
    save("V6", {"pass": bool(ok), "rows": rows, "summary": f"{sum(r[3] for r in rows)}/{len(rows)}"}, quick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    GIT.update(config.git_commit())
    if GIT["dirty"] and not args.quick:
        raise SystemExit("uncommitted changes: commit first, so the checks record the code they test")
    todo = args.only.split(",") if args.only else ["V0", "V4", "V5", "V6", "V3", "V1", "V1b", "V2"]
    with Pool(os.cpu_count()) as pool:
        for name in todo:
            t0 = time.time()
            if name == "V0":
                v0(args.quick)
            elif name == "V1":
                v1(args.quick, pool)
            elif name == "V1b":
                v1b(args.quick, pool)
            elif name == "V2":
                v2(args.quick, pool)
            elif name == "V3":
                v3(args.quick)
            elif name == "V4":
                v4(args.quick)
            elif name == "V5":
                v5(args.quick)
            elif name == "V6":
                v6(args.quick)
            print(f"  ({name} took {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
