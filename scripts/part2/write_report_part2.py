"""Generate docs/RESULTS_part2.md from the saved Part 2 analysis files.

Usage: python scripts/part2/write_report_part2.py --run final [--pilot-run pilot]
Numbers and decision labels are read from
results/part2/<run>/analysis/{confirmatory,exploratory}.json, the pilot report and the
map-size rule files. Fixed explanatory text describes the design and interpretation.
"""
import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPS = {"1": ("GS_food_frozen", "ET"), "2": ("GS_pop_frozen", "ET"), "3": ("GS_pop_mut", "ET"),
         "4": ("SV_frozen", "ET"), "5": ("ET_freezeB", "ET"), "6": ("ET_cloneB", "ET_freezeB"),
         "7": ("GS_pop_frozen", "GS_food_frozen"), "8": ("GS_pop_mut", "GS_pop_frozen")}
GROUPS_ORDER = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen", "GS_pop_mut", "SV_frozen", "ORACLE_B")
ARM_NAME = {"main": "Main arm (world B as in Part 1)", "arm2": "Arm 2 (blob-eating on in B)",
            "arm3": "Arm 3 (harder B: larger map)",
            "arm4": "Arm 4 (B with less energy per generation)"}


def f2(x, d=2):
    return "–" if x is None else f"{x:+.{d}f}"


def ci(c, d=2):
    return "–" if c is None or c[0] is None else f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def load(path):
    with open(path) as f:
        return json.load(f)


def arm_section(arm, r, u1, n, ctx):
    """One arm's confirmatory results. ctx: counts from checks.csv, the A-checks, and the
    arm's secondary results (for runs that reached B and, in arm 2, blobs eaten)."""
    L = [f"### {ARM_NAME[arm]}", ""]
    L.append(f"- Positive control: **{'holds' if r['positive_control'] else 'does not hold'}**.")
    L.append(f"- U2 (B-specific search advantage below 2.5): **{'fired' if r['u2_fires'] else 'did not fire'}** "
             f"(B winner ahead by less than 2.5 blobs in {sum(1 for d in r['u2_diff'] if d < 2.5)} of {len(r['u2_diff'])} runs).")
    cnt = ctx["single_design"]
    L.append(f"- Runs with a single design left at B generation 0: SV_frozen {cnt['sv']} of {n}, "
             f"GS_pop_mut {cnt['gs_pop_mut']} of {n}, ET {cnt['et']} of {n}."
             + (" SV_frozen therefore usually entered B as a single design." if r["sv_single_design_majority"] else ""))
    if arm == "arm2":
        L.append("- Predation requires prey no larger than 80% of the predator. Clone populations cannot use this food "
                 "source; access in varied populations depends on the size distribution. Comparisons 1, 2, 6 and 8 "
                 "therefore combine variation with access to predation.")
        be = ctx["sec"].get("blobs_eaten_per_group", {})
        if be:
            L.append("- Blobs eaten per run in B generations 0–49 (mean over runs; divide by 50 for per generation): "
                     + "; ".join(f"{g} {v:.1f}" for g, v in be.items()) + ".")
            if r["sv_single_design_majority"]:
                L.append(f"  In the {cnt['sv']} runs where SV_frozen is a single design at B generation 0, it is a clone "
                         "group, so row 4 is then also clone vs varied.")
    L.append("")
    L.append("| # | Comparison (first − second) | P1 difference [95% CI] | P2 difference [95% CI] | Holm p (P1, P2) | Verdicts (P1; P2) | Overall | Interpretation |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c, d in r["decided"].items():
        raw = r["raw"][c]
        g1, g2 = COMPS[c]
        lab = d["overall_label"] + (" — includes A-phase losses" if d.get("a_phase") else "")
        L.append(f"| {c} | {g1} − {g2} | {f2(raw['P1']['diff'])} {ci(raw['P1']['ci95'])} | "
                 f"{f2(raw['P2']['diff'])} {ci(raw['P2']['ci95'])} | {d['p_adj']['P1']:.3g}, {d['p_adj']['P2']:.3g} | "
                 f"{d['verdict']['P1']}{' (' + d['p1_label'] + ')' if d['p1_label'] else ''}; {d['verdict']['P2']} | "
                 f"{lab} | {r['guide'][c]} |")
    L.append("")
    # notes next to the table (Sections 4, 6 and 7)
    one = [c for c, d in r["decided"].items() if d["overall"] in ("first group better", "second group better")
           and "inconclusive" in (d["verdict"]["P1"], d["verdict"]["P2"])]
    if one:
        L.append("- **Verdicts that rest on one outcome** (the other is inconclusive): " + "; ".join(
            f"row {c} on {'P2 (dying out)' if r['decided'][c]['verdict']['P1'] == 'inconclusive' else 'P1 (population)'}"
            for c in one) + ".")
    ap = [c for c, d in r["decided"].items() if d.get("a_phase")]
    for c in ap:
        rb = ctx["sec"].get("reached_B_ci", {}).get(c)
        if rb:
            L.append(f"- Row {c} includes A-phase losses. Over the {rb['n_both']} runs where both groups reached B"
                     f": P1 {f2(rb['P1_diff'])} {ci(rb['P1_ci95'])}, "
                     f"P2 {f2(rb['P2_diff'])} {ci(rb['P2_ci95'])}.")
    a = ctx.get("achecks")
    if a and arm == "main":
        L.append(f"- **No-shift controls:** ET's end-of-A populations continued for 50 generations in A "
                 f"({a['n']} runs). Mean population over the last 10 generations: with mutation "
                 f"{a['means']['mut']:.2f}, without mutation {a['means']['frozen']:.2f}, clones {a['means']['clone']:.2f}. "
                 f"Without − with mutation: {f2(a['row5_frozen_minus_mut']['diff'])} "
                 f"{ci(a['row5_frozen_minus_mut']['ci95'])}; clones − without mutation: "
                 f"{f2(a['row6_clone_minus_frozen']['diff'])} {ci(a['row6_clone_minus_frozen']['ci95'])}. "
                 "These controls describe the same interventions without an environmental shift.")
    L.append("")
    cmb = r["combined"]
    L.append(f"- Combined criterion \"ET ≈ GS_pop_mut, and both > frozen groups\": "
             f"**{'met' + (' (' + cmb['label'] + ')' if cmb.get('label') else '') if cmb['ET ≈ GS_pop_mut, and both > frozen groups'] else 'not met'}**.")
    if cmb["SV_frozen ≈ ET"] and r["sv_single_design_majority"]:
        L.append(f"- Criterion \"SV_frozen ≈ ET\": **met by the rule**. SV_frozen entered B as a single "
                 f"design in {cnt['sv']} of {n} runs.")
    else:
        L.append(f"- Criterion \"SV_frozen ≈ ET\": **{'met' if cmb['SV_frozen ≈ ET'] else 'not met'}**.")
    L.append("")
    L.append("Mann-Whitney U and Cliff's delta for P1 (descriptive):")
    L.append("")
    L.append("| # | " + " | ".join(r["raw"]) + " |")
    L.append("|---" * (len(r["raw"]) + 1) + "|")
    L.append("| Mann-Whitney p | " + " | ".join(f"{r['raw'][c]['P1']['mwu_p']:.3g}" for c in r["raw"]) + " |")
    L.append("| Cliff's delta | " + " | ".join(f"{r['raw'][c]['P1']['cliffs_delta']:+.2f}" for c in r["raw"]) + " |")
    L.append("")
    L.append("Group summary (n = " + str(n) + " runs each):")
    L.append("")
    L.append("| Group | Mean P1 | SD | Extinct at end of B | Died out in A | Share with P1 ≥ 22.5 | Share with P1 ≤ 2.5 |")
    L.append("|---|---|---|---|---|---|---|")
    for g, v in u1.items():
        L.append(f"| {g} | {v['mean_P1']:.2f} | {v['sd_P1']:.2f} | {v['extinct']} | {v['died_in_A']} | "
                 f"{v['share_P1_ge_cap_minus_2.5']:.2f} | {v['share_P1_le_2.5']:.2f} |")
    L.append("")
    u4 = [g for g, v in u1.items() if v["died_in_A"] * 2 > v["n"]]
    L.append(f"- U4 (a group dies out in A in more than half its runs): "
             + ("**none**" if not u4 else "**" + ", ".join(u4) + "**") + ".")
    # U5: a test is degenerate when both groups have the same constant value (Section 6 gives it p = 1)
    deg1 = [c for c in r["raw"] if u1.get(COMPS[c][0], {}).get("sd_P1") == 0 and u1.get(COMPS[c][1], {}).get("sd_P1") == 0
            and r["raw"][c]["P1"]["diff"] == 0]
    deg2 = [c for c in r["raw"] if r["raw"][c]["P2"]["count_x"] == r["raw"][c]["P2"]["count_y"]
            and r["raw"][c]["P2"]["count_x"] in (0, r["raw"][c]["P2"]["n_x"])]
    perm1 = [c for c in r["raw"] if r["raw"][c]["P1"]["p"] >= 1]
    d2txt = ", ".join(f"row {c} ({'no run' if r['raw'][c]['P2']['count_x'] == 0 else 'every run'} of either group died out)"
                      for c in deg2) or "none"
    L.append(f"- U5 (degenerate tests: both groups the same constant value, given p = 1): P1 {', '.join('row ' + c for c in deg1) or 'none'}; "
             f"P2 {d2txt}. Permutation tests with p = 1: {', '.join('row ' + c for c in perm1) or 'none'}.")
    L.append("")
    return L


ARM_WORLD = {"main": "main", "arm2": "arm2", "arm3": "arm3", "arm4": "arm4"}


def predictions_section(conf, ex, ck):
    """Sections 12 and 18.9: the predictions written before any run, checked against the results."""
    m = conf["main"]["decided"]
    ov = lambda arm, c: conf[arm]["decided"][c]["overall"]
    lab = lambda arm, c: conf[arm]["decided"][c]["overall_label"]
    u1 = conf["U1_U4"]["main"]
    n = u1["ET"]["n"]
    st2 = sum(int(x["tied_stage2_pop"]) > 1 for x in ck)
    su = ex.get("suite", {}).get("variants", {})
    lead = {int(v[4:]): su[v]["groups"]["ET"]["P1"] - su[v]["groups"]["GS_pop_frozen"]["P1"]
            for v in su if v.startswith("side")}
    sides = sorted(lead)
    gr = ex.get("gradients", {})
    rows = [
        (1, "GS_food_frozen dies out in A in most runs", 0.6, u1["GS_food_frozen"]["died_in_A"] * 2 > n,
         f"{u1['GS_food_frozen']['died_in_A']} of {n}"),
        (2, "GS_pop_frozen reaches B in at least 95% of runs", 0.8, (n - u1["GS_pop_frozen"]["died_in_A"]) >= 0.95 * n,
         f"{n - u1['GS_pop_frozen']['died_in_A']} of {n}"),
        (3, "The A search's population tie set has more than one design in most runs", 0.7, st2 * 2 > len(ck),
         f"read as the tie set after stage 2: more than one design in {st2} of {len(ck)} runs"),
        (4, "U2 fires", 0.4, conf["main"]["u2_fires"], ""),
        (5, "The positive control holds", 0.5, conf["main"]["positive_control"], ""),
        (6, "Row 2 is \"the same (at the ceiling)\"", 0.4, lab("main", "2") == "the same (at the ceiling)", lab("main", "2")),
        (7, "Row 2 is \"second group better\" (ET better)", 0.4, ov("main", "2") == "second group better", lab("main", "2")),
        (8, "Row 5 is \"the same (at the ceiling)\"", 0.7, lab("main", "5") == "the same (at the ceiling)", lab("main", "5")),
        (9, "Row 6 is \"the same (at the ceiling)\"", 0.6, lab("main", "6") == "the same (at the ceiling)", lab("main", "6")),
        (10, "Row 4 is \"second group better\" (ET better)", 0.5, ov("main", "4") == "second group better", lab("main", "4")),
        (11, "Any row is \"the same\" without a label", 0.2, any(d["overall_label"] == "the same" for d in m.values()), ""),
        (12, "In arm 2, ET's mean P1 is above 25", 0.7, bool(ex["secondary"]["arm2"]["ET_mean_P1_above_25"]),
         f"{ex['secondary']['arm2']['ET_mean_P1']:.2f}"),
        (13, "At least one overall verdict changes between the main arm and arm 2", 0.3,
         any(v.get("result") == "changes" for v in conf["holds_changes"]["arm2"].values()), ""),
        (14, "Arm 3's map-size rule finds a qualifying side", 0.7, False,
         "no side qualified in the rule as written (sides 400–1,000); 750 qualified in an amended second round"),
        (15, "In arm 3, row 5 is \"second group better\"", 0.35, ov("arm3", "5") == "second group better", lab("arm3", "5")),
        (16, "In arm 3, at least one of rows 1–4 is \"first group better\"", 0.3,
         any(ov("arm3", c) == "first group better" for c in ("1", "2", "3", "4")),
         ", ".join(f"row {c}: {lab('arm3', c)}" for c in ("1", "2", "3", "4"))),
        (17, "In the difficulty curve, ET's lead over GS_pop_frozen on P1 grows with map size", 0.5,
         (lead[sides[-1]] > lead[sides[0]]) if sides else None,
         "read as: the lead at the largest side is bigger than at the smallest; ET − GS_pop_frozen by side: "
         + ", ".join(f"{k}: {lead[k]:+.2f}" for k in sides)),
        (18, "The A–B landscape rank correlation is lower in arm 3's B than in the main B", 0.6,
         ex["landscape"]["arm3"]["mean_tau_b_all"] < ex["landscape"]["main"]["mean_tau_b_all"],
         f"tau-b {ex['landscape']['arm3']['mean_tau_b_all']:.2f} vs {ex['landscape']['main']['mean_tau_b_all']:.2f}"),
        (19, "ET_scratchB does worse than ET in the main B", 0.7, ex["scratch_B"]["main"]["P1_ci95"][1] < 0,
         f"read as: the 95% CI for P1 is below 0; P1 {ex['scratch_B']['main']['P1_diff']:+.2f} "
         f"{ci(ex['scratch_B']['main']['P1_ci95'])}, far below the 2.5-blob threshold; P2 "
         f"{ex['scratch_B']['main']['P2_diff']:+.2f}"),
        (20, "The stage-1-only population winner dies out in A more often than the two-stage winner", 0.6,
         ex["stage1_winners"]["main"]["GS_pop_s1"]["died_in_A"] > ex["stage1_winners"]["main"]["GS_pop_frozen"]["died_in_A"],
         f"{ex['stage1_winners']['main']['GS_pop_s1']['died_in_A']} vs {ex['stage1_winners']['main']['GS_pop_frozen']['died_in_A']} "
         f"of {ex['stage1_winners']['main']['GS_pop_s1']['n']}"),
        (21, "In B generations 0–9, ET's selection gradient on size has the opposite sign to A generations 90–99", 0.3,
         (gr["ET_A"]["mean"]["log_size"] * gr["ET_B"]["mean"]["log_size"] < 0) if "ET_A" in gr and "ET_B" in gr else None,
         (f"A {gr['ET_A']['mean']['log_size']:+.3f}, B {gr['ET_B']['mean']['log_size']:+.3f}" if "ET_A" in gr and "ET_B" in gr else "")),
    ]
    L = ["## Predictions written before any run (Sections 12 and 18.9; no part in the analysis)", "",
         "Main arm unless stated. Where a prediction's wording needed a reading, the reading is given.", "",
         "| # | Prediction | Probability given | Came true? | Result |", "|---|---|---|---|---|"]
    for num, txt, pr, ok, note in rows:
        L.append(f"| {num} | {txt} | {pr} | {'–' if ok is None else ('yes' if ok else 'no')} | {note} |")
    judged = [(pr, ok) for _, _, pr, ok, _ in rows if ok is not None]
    brier = sum((pr - (1.0 if ok else 0.0)) ** 2 for pr, ok in judged) / len(judged)
    L += ["", f"{sum(ok for _, ok in judged)} of {len(judged)} came true. Brier score {brier:.3f} "
              "(0 is perfect; always saying 0.5 scores 0.25).", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pilot-run", default="pilot")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "RESULTS_part2.md"))
    args = ap.parse_args()
    adir = os.path.join(ROOT, "results", "part2", args.run, "analysis")
    conf = load(os.path.join(adir, "confirmatory.json"))
    ex = load(os.path.join(adir, "exploratory.json"))
    run = load(os.path.join(ROOT, "results", "part2", args.run, "run.json"))
    seeds = conf["provenance"]["seeds"]
    n = len(seeds)
    p2c = run['provenance']['config']['part2']
    L = ["# Trait experiment: results", "",
         f"We compared trait-search and evolving populations across four transfer environments, using {n} runs per "
         f"group and arm. Arm 3 used a map side of {p2c.get('arm3_side')}."]
    if p2c.get("arm4_energy") is not None:
        L.append(f"Arm 4 supplied {p2c['arm4_energy']:g} energy per generation in B, compared with "
                 f"{run['provenance']['config']['world']['energy_per_generation']:g} in A and the main B.")
    L += ["", "Population search scored generations 10–99 in stage two; food search scored generations 10–29. "
          "The comparison therefore includes both objective and scoring horizon.", ""]
    with open(os.path.join(adir, "checks.csv")) as f:
        ck = list(csv.DictReader(f))
    single = {"sv": sum(int(x["sv_distinct_B0"]) == 1 for x in ck),
              "gs_pop_mut": sum(int(x["gs_pop_mut_distinct_B0"]) == 1 for x in ck),
              "et": sum(int(x["et_distinct_B0"]) == 1 for x in ck)}
    L += ["## Confirmatory results", "",
          "P1 is mean population in B generations 40–49; P2 is extinction by the end of B. "
          "The registered label ‘meaningfully different’ requires Holm-adjusted p < 0.05 and a point estimate "
          "at least as large as the practical threshold (2.5 agents for P1; 0.10 for P2). "
          "Displayed intervals are nominal per-comparison 95% intervals. "
          "Analyses restricted to runs reaching B condition on survival and may be affected by selection bias.", ""]
    for arm in ("main", "arm2", "arm3", "arm4"):
        if arm in conf:
            ctx = {"single_design": single, "achecks": ex.get("achecks"), "sec": ex["secondary"].get(arm, {})}
            L += arm_section(arm, conf[arm], conf["U1_U4"][arm], n, ctx)
    L += ["### Comparison across arms", "",
          "Cross-arm comparisons use p < 0.05 ÷ 16 and 99.375% equivalence intervals. "
          "This fixed threshold avoids changes caused by the other tests in a Holm family.", ""]
    for arm, rows in conf.get("holds_changes", {}).items():
        L.append(f"- {ARM_NAME[arm]} vs main: " + "; ".join(
            f"{c}: {v['result']}" + (f" ({v['main']} → {v[arm]})" if v.get("result") == "changes" else "")
            for c, v in rows.items()))
        diff = [f"row {c} in {nm}: {v[a]} here, {conf[a]['decided'][c]['overall']} in the table"
                for c, v in rows.items() for a, nm in (("main", "the main arm"), (arm, ARM_NAME[arm].split(" (")[0]))
                if a in v and c in conf[a]["decided"] and v[a] != conf[a]["decided"][c]["overall"]]
        if diff:
            L.append("  - Fixed-level category differs from the table (Holm): " + "; ".join(sorted(set(diff))) + ".")
    L.append("")
    L.append(f"- U3 (search winner doesn't hold up): **{'fired' if conf['U3']['fires'] else 'did not fire'}** "
             f"(mean shortfall {conf['U3']['mean_shortfall']:.2f} blobs).")
    L.append(f"- Clone check (clone groups identical with blob-eating on and off): "
             f"{'passed' if not conf['clone_check_mismatches'] else 'FAILED: ' + str(conf['clone_check_mismatches'])}.")
    L.append("")
    # per-run S4 generation counts (Section 5 asks for the number of generations behind S4)
    s4n = {}
    with open(os.path.join(adir, "outcomes.csv")) as f:
        for row in csv.DictReader(f):
            if row.get("S4_n") not in (None, ""):
                s4n.setdefault((row["world"], row["group"]), []).append(int(row["S4_n"]))
    L += ["## Secondary results", "",
          "S4 measures food per agent and can rise as a population shrinks; it is reported alongside population counts.", "",
          ""]
    for arm, r in ex["secondary"].items():
        L.append(f"**{ARM_NAME[arm]}.** Paired by run seed (sign-flip test for P1, McNemar for P2):")
        L.append("")
        L.append("| # | P1 paired mean difference [95% CI] | sign-flip p | McNemar p | Runs where both reached B: P1 diff [95% CI], P2 diff [95% CI] |")
        L.append("|---|---|---|---|---|")
        for c, p in r["paired"].items():
            rb = r["reached_B_ci"].get(c)
            rbt = (f"n = {rb['n_both']}: {f2(rb['P1_diff'])} {ci(rb['P1_ci95'])}, {f2(rb['P2_diff'])} {ci(rb['P2_ci95'])}"
                   if rb else "–")
            L.append(f"| {c} | {f2(p['P1_mean_diff'])} {ci(p['P1_ci95'])} | {p['signflip_p']:.3g} | {p['mcnemar_p']:.3g} | {rbt} |")
        L.append("")
        L.append("Secondary outcomes:")
        L.append("")
        L.append("| Group | S6: mean population, B gens 1–49 | S4: food per blob, B gens 40–49 (runs with a value; mean gens with blobs) | S1: first B gen with 23+ blobs, median (runs never) |")
        L.append("|---|---|---|---|")
        for g in GROUPS_ORDER:
            if g not in r["S6"]:
                continue
            s4 = r["S4"][g]
            ns = [x for x in s4n.get((ARM_WORLD.get(arm, arm), g), []) if x > 0]  # runs with a value only
            s1 = r["S1"][g]
            L.append(f"| {g} | {r['S6'][g]:.2f} | "
                     + (f"{s4['mean']:.3f} ({s4['n_runs']} runs; {sum(ns) / len(ns):.1f} gens)" if s4["mean"] is not None and ns else "– (no run)")
                     + f" | {'–' if s1['median'] is None else format(s1['median'], 'g')} ({s1['never']}) |")
        L.append("")
        if arm == "arm2":
            L.append(f"- ET's mean P1 with blob-eating on: {r['ET_mean_P1']:.2f} "
                     f"({'above' if r['ET_mean_P1_above_25'] else 'not above'} 25; the paper reports about 29).")
            L.append("")
    L += ["### Search performance", ""]
    def rng_txt(key):
        v = [int(x[key]) for x in ck]
        return f"mean {sum(v) / len(v):.1f}, range {min(v)}–{max(v)}"
    L.append(f"- Tie-set sizes, A search, population score: stage 1 {rng_txt('tied_stage1_pop')}; stage 2 {rng_txt('tied_stage2_pop')}. "
             f"Food score: stage 1 {rng_txt('tied_stage1_food')}; stage 2 {rng_txt('tied_stage2_food')}.")
    L.append(f"- Search score vs deployed performance: U3's mean shortfall (stage-2 score minus GS_pop_frozen's deployed "
             f"A mean, generations 10–99) is {conf['U3']['mean_shortfall']:.4f} blobs.")
    at = ex.get("a_ties_in_b", {})
    if at:
        L.append("- B stage-1 score of every design tied for best in A's stage 1 (does being best in A predict doing well in B?): "
                 + "; ".join(f"{ARM_NAME.get(w, w).split(' (')[0]}: {v['mean_n_tied_in_A']:.0f} tied designs on average, "
                             f"mean B score {v['mean_B_score_of_A_tied']:.2f} (best B design {v['mean_best_B_score']:.2f}), "
                             f"share with B score ≥ 22.5: {v['mean_share_of_A_tied_with_B_score_ge_22_5']:.2f}"
                             for w, v in at.items()) + f" ({next(iter(at.values()))['n_seeds']} seeds).")
    L.append("")
    L += ["### Compute (creature-steps per run, including search)", ""]
    for arm, rows in ex["compute_cost"].items():
        L.append(f"- {ARM_NAME[arm]}: " + "; ".join(f"{g} {v['mean']:,.0f}" for g, v in rows.items()))
    L.append("")
    L += ["## Exploratory results", "",
          "Exploratory items use the 100 exploratory seeds (2000–2099) unless stated, so their numbers can differ "
          "slightly from the 200-seed tables above.", ""]
    su = ex.get("suite", {})
    if su:
        L.append("**Robustness suite.** Counts of B variants whose 95% interval excludes zero, by direction, and whose "
                 "point estimate reaches the practical threshold. Comparisons use runs where both groups reached B. "
                 "The main-arm direction uses the same 100 seeds; zero denotes an interval containing zero.")
        L.append("")
        L.append("| # | Main-arm direction | Variants | Positive | Negative | Reach threshold |")
        L.append("|---|---|---|---|---|---|")
        for c, v in su["summary"].items():
            L.append(f"| {c} | {v['main_direction']:+d} | {v['n_variants']} | {v['positive']} | {v['negative']} | "
                     f"{v['reaches_threshold'] if v['main_direction'] != 0 else '–'} |")
        L.append("")
        L.append("Per variant, per group: P1 / P2 (share extinct) / S6 / share near the variant's cap:")
        L.append("")
        L.append("| Variant | " + " | ".join(g for g in GROUPS_ORDER if g != "ORACLE_B") + " |")
        L.append("|---" * len(GROUPS_ORDER) + "|")
        for v, vr in su["variants"].items():
            L.append(f"| {v} | " + " | ".join(
                (f"{vr['groups'][g]['P1']:.2f} / {vr['groups'][g]['P2']:.2f} / {vr['groups'][g]['S6']:.2f} / "
                 f"{'–' if vr['groups'][g]['near_cap'] is None else format(vr['groups'][g]['near_cap'], '.2f')}")
                if g in vr["groups"] else "–" for g in GROUPS_ORDER if g != "ORACLE_B") + " |")
        L.append("")
        L.append("Per variant, per comparison (runs where both groups reached B): P1 difference [95% CI]; P2 difference [95% CI]; n:")
        L.append("")
        L.append("| Variant | " + " | ".join(f"#{c}" for c in COMPS) + " |")
        L.append("|---" * (len(COMPS) + 1) + "|")
        for v, vr in su["variants"].items():
            L.append(f"| {v} | " + " | ".join(
                (f"{f2(vr['comparisons'][c]['P1_diff'])} {ci(vr['comparisons'][c]['P1_ci95'])}; "
                 f"{f2(vr['comparisons'][c]['P2_diff'])} {ci(vr['comparisons'][c].get('P2_ci95'))}; "
                 f"{vr['comparisons'][c]['n_both_reached_B']}") if c in vr["comparisons"] else "–" for c in COMPS) + " |")
        L.append("")
        L.append("The [trait-experiment figure](../figures/publication/fig07_trait_experiment.png) summarizes the four primary arms.")
        L.append("")
    if ex.get("landscape"):
        L.append("**Landscape check** (Kendall tau-b between A and B stage-1 population scores; the A winner's rank in B). "
                 "The main B, arm 3 and arm 4 use every seed's own B search; the variants use seeds 2000–2009:")
        L.append("")
        for w, v in sorted(ex["landscape"].items()):
            L.append(f"- {w}: tau-b {v['mean_tau_b_all']:.2f} (designs alive in A: "
                     f"{'–' if v['mean_tau_b_A_alive'] is None else format(v['mean_tau_b_A_alive'], '.2f')}); "
                     f"A winner's B percentile {v['mean_A_winner_rank_pct']:.2f}; n = {v['n_seeds']} seeds")
        L.append("")
    ms = ex.get("mutation_sensitivity", {})
    if ms:
        L += ["**Mutation size × 0.5 and × 2** (P1 and P2 differences, first − second, with 95% CIs, on "
              "the 100 exploratory seeds; the ×1 rows are the frozen setting on the same seeds, for a like-for-like "
              "comparison). \\* = both groups had 3 or fewer runs with any population in B generations 40–49, so the "
              "interval is poorly resolved.", "",
              "| Setting | " + " | ".join(f"#{c}" for c in COMPS) + " |", "|---" * (len(COMPS) + 1) + "|"]
        order = sorted(ms, key=lambda k: (k.split("_", 1)[1], float(k.split("_")[0][1:])))
        for k in order:
            rows = ms[k]
            L.append(f"| {k} | " + " | ".join(
                (f"P1 {f2(rows[c]['P1_diff'])} {ci(rows[c]['P1_ci95'])}; P2 {f2(rows[c]['P2_diff'])} {ci(rows[c]['P2_ci95'])}"
                 + (" \\*" if max(rows[c].get("n_nonzero", [99, 99])) <= 3 else ""))
                if c in rows else "–" for c in COMPS) + " |")
        L.append("")
    sb = ex.get("scratch_B", {})
    if sb:
        L.append("**ET_scratchB minus ET** (50 default blobs started directly in B, vs ET trained in A): "
                 + "; ".join(f"{ARM_NAME.get(w, w)}: P1 {f2(v['P1_diff'])} {ci(v['P1_ci95'])}, P2 {f2(v['P2_diff'])} {ci(v['P2_ci95'])}"
                             for w, v in sb.items()) + ".")
        L.append("")
    lb = ex.get("longer_B", {})
    if lb:
        L += ["**Longer B** (B phase continued to B generation 199; P1 at B generations 40–49 "
              "→ the same measure at 190–199, and the share extinct at the end of B generation 199):", "",
              "| World | " + " | ".join(GROUPS_ORDER) + " |", "|---" * (len(GROUPS_ORDER) + 1) + "|"]
        for w, rows in lb.items():
            L.append(f"| {ARM_NAME.get(w, w)} (n = {next(iter(rows.values()))['n']}) | " + " | ".join(
                (f"{rows[g]['P1']:.2f} → {rows[g]['late_P1']:.2f}; extinct {rows[g]['late_P2']:.2f}" if g in rows else "–")
                for g in GROUPS_ORDER) + " |")
        L.append("")
    s1 = ex.get("stage1_winners", {})
    if s1:
        same = s1.get("same_as_two_stage", {})
        L.append("**Effect of the second search stage** (stage-1-only winners deployed as frozen clones). "
                 f"The stage-1 winner was the same design as the two-stage winner in {same.get('pop')} of {same.get('n')} runs "
                 f"(population score) and {same.get('food')} of {same.get('n')} (food score).")
        L.append("")
        for w, rows in s1.items():
            if w == "same_as_two_stage":
                continue
            L.append(f"- {ARM_NAME.get(w, w)}: " + "; ".join(
                f"{g} P1 {v['P1']:.2f}, extinct {v['P2']:.2f}, died in A {v['died_in_A']} of {v['n']}" for g, v in rows.items()))
        L.append("")
    th = ex.get("threshold_sensitivity", {})
    if th:
        L.append("**Threshold sensitivity** (rows whose verdict changes with half / double thresholds, all "
                 "200 seeds): " + "; ".join(f"{k}: {v['changed'] or 'none'}" for k, v in th.items()) + ".")
        L.append("")
    g = ex.get("gradients", {})
    if g:
        L.append("**Selection gradients** (main B only, B generations 0–9; A generations 90–99 for the A "
                 "rows, 0–9 for SV_frozen in A; mean over runs that give one; 95% CI):")
        L.append("")
        for k, v in g.items():
            if v.get("n_runs_with_gradient"):
                L.append(f"- {k} ({v['n_runs_with_gradient']} of {v['n_runs']} runs): " + "; ".join(
                    f"{t} {v['mean'][t]:+.3f} {ci(v['ci95'][t], 3)}" for t in v["mean"]))
            else:
                L.append(f"- {k}: no run gave a gradient")
        L.append("")
    fd = ex.get("first_gen_deaths", {})
    if fd:
        L.append("**First-generation deaths in B** (main B, all 200 seeds): share of blobs that die in B "
                 "generation 0, and mean traits (speed, size, sense, reach) of survivors vs those that die:")
        L.append("")
        L.append("| Group | Share dying | Survivors | Died |")
        L.append("|---|---|---|---|")
        clones = ("GS_food_frozen", "GS_pop_frozen", "ET_cloneB", "ORACLE_B")
        for gname, v in fd.items():
            tr = lambda d: "–" if not d else ", ".join(f"{d[t]:.1f}" for t in ("speed", "size", "sense", "reach"))
            sv, dv = ("clones: no split", "–") if gname in clones else (tr(v.get("survivor_traits")), tr(v.get("dead_traits")))
            L.append(f"| {gname} | {'–' if v.get('share_dying_B0') is None else format(v['share_dying_B0'], '.2f')} | "
                     f"{sv} | {dv} |")
        L.append("")
        L.append("Clone groups have one design, so survivors and blobs that died have the same traits; their trait averages "
                 "would only differ because they come from different sets of runs, so no split is shown.")
        L.append("")
    eu = ex.get("energy_use", {})
    if eu:
        L.append("**Energy use** (main B, B generations 0–49, all 200 seeds): mean energy cost per move, and "
                 "ticks used per generation, over generations that had blobs: "
                 + "; ".join(f"{gname} {v['energy_per_move']:.2f} per move, {v['ticks_per_generation_with_blobs']:.0f} ticks"
                             for gname, v in eu.items() if v.get("energy_per_move") is not None) + ".")
        L.append("")
    L += ["## Data", "",
          f"Per-run outcomes and search diagnostics are in `results/part2/{args.run}/analysis/`. "
          "Environment selection and decision rules are described in [PREREG_part2.md](PREREG_part2.md).", ""]
    text = "\n".join(L) + "\n"
    # Update interpretive prose only. Preserve saved statistical results, decision
    # labels and the historical protocol/guide in confirmatory.json.
    corrections = {
        "Evolution beats the 2025-style search, as in 2025":
            "ET outperformed frozen food-objective search",
        "Aiming at population beats aiming at food per blob: part of the 2025 result came from the wrong goal":
            "Population-objective search outperformed food-objective search",
        "A fixed design aimed at the right goal beats evolution":
            "Frozen population-objective search outperformed ET",
        "Variety built up in A helps after the move":
            "The frozen population outperformed the cloned population",
        "Starting at the A optimum and evolving beats starting simple":
            "Mutating population-search initialization outperformed ET",
        "Conclusion (Section 7)": "Interpretation",
        "Being able to change hurts":
            "The mutating population-search group underperformed its frozen counterpart",
        "Both groups did about as well as this B allows; this B can't show whether one is better":
            "Both groups were near the population ceiling",
        "The design that selection in A picked from 50 random starts does as well as evolution":
            "SV_frozen and ET met the equivalence criterion",
        "Evolution beats the design that selection in A picked from 50 random starts":
            "ET outperformed SV_frozen",
        "New mutations after the move help":
            "ET outperformed ET_freezeB",
        "No conclusion (B doesn't call for a different design)":
            "The U2 criterion limits interpretation",
        "(part of this difference happened before the move)":
            "(including losses in A)",
    }
    for old, new in corrections.items():
        text = text.replace(old, new)
    with open(args.out, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()

