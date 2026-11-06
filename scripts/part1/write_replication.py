"""Write docs/REPLICATION.md from docs/templates/REPLICATION_part1.md and the comparison file.

Every number in the output comes from results/part1/<name>/comparison.json, the run files,
or docs/paper_numbers_part1.yaml (the paper's printed numbers). The template holds only
words and {{blank}} markers; a marker with no value stops the script with an error.
Sentences whose content depends on the results (which numbers miss, in which direction,
under which setting) are also built here from the data, not written into the template.

Usage: python scripts/part1/write_replication.py --name final
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_paper import ROOT, finite, fmt  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, metrics  # noqa: E402

STATUS = {"target": "calibration target", "defn": "used to pick the definition", "held_out": "held out",
          "setting": "used to set a guess"}
BORDER = (0.8, 1.2)  # |error / tolerance| within 20% of the tolerance edge is reported as borderline


def pm(m, sd, nd=2):
    if not finite(m):
        return "no value"
    return f"{fmt(m, nd)} ± {fmt(sd, nd)}" if finite(sd) else fmt(m, nd)


def mark(v, posthoc=None):
    base = {"replicates": "yes", "misses": "**no**", "no value": "no value"}[v]
    if posthoc and posthoc != v:
        base += f" (after-data reading: {'yes' if posthoc == 'replicates' else 'no'})"
    return base


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def target_label(tid):
    fam, _, g = tid.partition("_")
    fam = {"reach": "reach", "cv2": "CV²", "shannon": "Shannon", "speed": "speed", "size": "size",
           "sense": "sense", "fitness": "fitness", "population": "population"}.get(fam, fam)
    gen = "99" if g == "final" else g.lstrip("g")
    return fam, gen


def describe_misses(rows, which):
    """Group failing targets by trait family and direction, e.g. 'reach at gens 40, 60 (too low)'."""
    groups = {}
    for r in rows:
        if not which(r):
            continue
        fam, gen = target_label(r["id"])
        e = r["fresh_e"] if "fresh" in which.__name__ else r["cal_e"]
        key = (fam, "too low" if e < 0 else "too high")
        groups.setdefault(key, []).append(gen)
    parts = []
    for (fam, d), gens in groups.items():
        gens = sorted(gens, key=lambda x: int(x) if x.isdigit() else 999)
        parts.append(f"{fam} at generation{'s' if len(gens) > 1 else ''} {join_words(gens)} ({d})")
    return join_words(parts) if parts else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    folder = os.path.join(ROOT, "results", "part1", args.name)
    c = json.load(open(os.path.join(folder, "comparison.json")))
    info = json.load(open(os.path.join(folder, "run_info.json")))
    paper = yaml.safe_load(open(os.path.join(ROOT, "docs", "paper_numbers_part1.yaml")))
    held = json.load(open(os.path.join(ROOT, "results", "part1", "paper_heldout_digitized.json")))["figures"]
    cfg = info["provenance"]["config"]
    ex = c["extra"]
    v = {}

    # ---- run facts
    prov = info["provenance"]
    v["run_commit"] = prov["git"]["commit"][:7]
    v["run_dirty"] = "clean" if not prov["git"]["dirty"] else "DIRTY"
    v["config_hash"] = prov["config_hash"][:12]
    v["analysis_commit"] = open(os.path.join(folder, "analysis_commit.txt")).read().strip()
    gb = ex["generated_by"]
    v["gen_commit"] = gb["commit"][:7]
    v["gen_dirty"] = "clean" if not gb["dirty"] else "DIRTY"
    # the comparison file must come from clean code, and the code must not have changed since
    import subprocess
    me = config.git_commit()
    code_changes = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs", "docs/templates",
                                   "docs/paper_numbers_part1.yaml"], cwd=ROOT, capture_output=True, text=True).stdout
    if gb["dirty"] or me["commit"] != gb["commit"] or code_changes.strip():
        raise SystemExit(f"stop: comparison.json made at {gb}, now at {me}, code changes: {code_changes!r}")
    v["n_et"], v["n_gs"] = c["n_runs"]["ET"], c["n_runs"]["GS"]
    v["seeds"] = f"{min(info['seeds'])}–{max(info['seeds'])}"
    v["wall_s"] = fmt(info["wall_seconds"], 1)
    v["n_train"] = cfg["phases"]["train_generations"]
    v["n_transfer"] = cfg["phases"]["transfer_generations"]
    v["n_clones"] = cfg["gs"]["n_clones"]
    v["eval_gens"] = cfg["gs"]["eval_generations"]
    v["search_seed"] = cfg["gs"]["search_seed"]
    v["tick_cap"] = cfg["world"]["tick_cap"]
    v["b_food"] = cfg["env_B"]["food"]
    v["a_food"] = cfg["env_A"]["food"]
    v["mut"] = ", ".join(f"{k} {fmt(x, 2)}" for k, x in cfg["mutation_sd"].items())
    v["cannibalism"] = "on" if cfg["world"]["cannibalism"] else "off"
    v["cal_setting"] = cfg["calibration"]["selected_setting"]
    v["hardware"] = f"{prov['hardware']['cpu']} ({prov['hardware']['cpu_count']} CPUs)"
    v["versions"] = ", ".join(f"{k} {x}" for k, x in prov["versions"].items())

    row = {r["paper"]["key"]: r for r in c["fig11"]}
    tx = {t["id"]: t for t in c["text"]}

    # ---- held-out summary over DISTINCT quantities (each rebuilt quantity counted once)
    def item(label, paper_s, reb, sd, n, verdict, posthoc=None, note=""):
        return {"label": label, "paper": paper_s, "rebuild": f"{pm(reb, sd)} ({n})" if n else "no value (0)",
                "verdict": verdict, "posthoc": posthoc, "note": note}

    def f11(key, grp, label, note=""):
        r = row[key][grp]
        return item(label, pm(*row[key]["paper"][grp]), r["mean"], r["sd"], r["n"], r["verdict"], r["posthoc_verdict"], note)

    def txt(i, label, note="", nd=2):
        t = tx[i]
        return item(label, pm(t["value"], t["sd"], nd), t["rebuilt"], t["rebuilt_sd"], t["n"], t["verdict"],
                    t["posthoc_verdict"], note)

    et_items = [
        txt("et_transfer_gen0", "Fitness, first transfer generation (text)",
            "if B had no food source besides its own items, the paper's value needs more food than B places (see below)"),
        f11("transfer_first10", "et", "Fitness, transfer gens 0–9 (table row \"gen 1-10\")"),
        f11("transfer_last10", "et", "Fitness, transfer gens 40–49 (table row \"last 10 gen\")",
            f"the text's {pm(tx['et_transfer_stable']['value'], tx['et_transfer_stable']['sd'])} is also matched"),
        f11("gap", "et", "Generalization gap (table)",
            "final training fitness matches the paper, so this is the same shortfall as the row above"
            if row["final_fitness"]["et"]["verdict"] == "replicates" and row["transfer_first10"]["et"]["verdict"] == "misses"
            and row["gap"]["et"]["verdict"] == "misses" else ""),
        f11("slope", "et", "Trend slope in transfer (table)",
            f"the text gives {pm(tx['slope_text_et']['value'], tx['slope_text_et']['sd'])} instead; "
            f"that version {'is' if tx['slope_text_et']['verdict'] == 'replicates' else 'is not'} matched"),
        txt("et_transfer_population", "Population, late transfer (Fig. 7 caption)"),
    ]
    gs_items = [f11(k, "gs", row[k]["row"] + " (table)",
                    "zero by design for identical clones" if row[k]["zero_by_design"] else
                    (f"rests on {row[k]['gs']['n']} runs" if 0 < row[k]["gs"]["n"] < 10 else ""))
                for k in ("final_fitness", "final_median", "transfer_first10", "transfer_last10", "gap", "slope",
                          "cv2", "shannon", "energy_efficiency")]
    gs_items += [txt("gs_population_gen100", "Population at generation 100 (text)"),
                 txt("gs_survival", "Survival in training (text)",
                     (f"rests on {tx['gs_survival']['n']} runs, with "
                      + join_words(f"{sum(1 for x in r['populations_gens90_99'] if x > 0)} generations of "
                                   f"{min(x for x in r['populations_gens90_99'] if x > 0)}–"
                                   f"{max(r['populations_gens90_99'])} blobs"
                                   for r in ex["gs_survival_runs"]))
                     if 0 < tx["gs_survival"]["n"] < 10 else "", nd=3),
                 txt("gs_transfer_population_late", "Population, transfer gens 25–30 (text: 2–3)")]

    def counts(items):
        n = len(items)
        rep = sum(i["verdict"] == "replicates" for i in items)
        post = sum((i["posthoc"] or i["verdict"]) == "replicates" for i in items)
        return n, rep, post

    for pre, items in (("et", et_items), ("gs", gs_items)):
        v[f"{pre}_n"], v[f"{pre}_rep"], v[f"{pre}_rep_post"] = counts(items)
        v[f"T_{pre}_items"] = md_table(
            ["Quantity", "Paper", "Rebuild (n)", "Within? (rule as written)", "Note"],
            [[i["label"], i["paper"], i["rebuild"], mark(i["verdict"], i["posthoc"]), i["note"]] for i in items])
    v["et_post_sentence"] = (f" ({v['et_rep_post']} of {v['et_n']} with the after-the-data reading)"
                             if v["et_rep_post"] != v["et_rep"] else "")
    gs_matched = [i for i in gs_items if i["verdict"] == "replicates"]
    def lower_first(t):
        return t if t.startswith("Shannon") else t[0].lower() + t[1:]
    v["gs_matched_list"] = join_words(f"{lower_first(i['label'].split(' (')[0])} ({i['note']})" for i in gs_matched)
    ce = tx["compute_et"]
    v["compute_et_verdict"] = mark(ce["verdict"])

    # ---- training targets on fresh seeds, described from data
    tb = ex["targets_both"]
    tr = c["targets_recheck"]
    v["targets_pass"], v["targets_n"], v["cal_pass"] = tr["n_pass"], tr["n_targets"], tr["calibration_seeds_n_pass"]

    def fresh_fail_both(r):
        return (not r["fresh_pass"]) and (not r["cal_pass"])

    def fresh_fail_only(r):
        return (not r["fresh_pass"]) and r["cal_pass"]

    def cal_fail_only(r):
        return r["fresh_pass"] and (not r["cal_pass"])

    v["miss_both"] = describe_misses(tb, fresh_fail_both)
    v["miss_fresh_only"] = describe_misses(tb, fresh_fail_only)
    v["miss_cal_only"] = describe_misses(tb, cal_fail_only)
    rg = ex["reach_gen99"]
    v["reach_f"] = pm(rg["fresh_mean"], rg["fresh_sd"])
    v["reach_c"] = pm(rg["cal_mean"], rg["cal_sd"])
    v["reach_min"], v["reach_max"] = fmt(rg["fresh_min"], 2), fmt(rg["fresh_max"], 2)
    v["reach_se"], v["reach_tol"] = fmt(rg["fresh_se"], 2), fmt(rg["tolerance"], 2)
    v["reach_p"] = fmt(rg["welch_p_fresh_vs_cal"], 2)
    v["reach_nf"], v["reach_nc"] = rg["n_fresh"], rg["n_cal"]
    ratio = rg["fresh_se"] / rg["tolerance"]
    v["noise_sentence"] = (f"Run-to-run variation is large for some of these targets. For reach at generation 99, the "
                           f"standard error of a 30-run mean ({fmt(rg['fresh_se'], 2)}) is {fmt(ratio, 1)} times the "
                           f"target's tolerance ({fmt(rg['tolerance'], 2)}), and the two seed sets give "
                           f"{fmt(rg['fresh_mean'], 2)} and {fmt(rg['cal_mean'], 2)} (Welch p = "
                           f"{fmt(rg['welch_p_fresh_vs_cal'], 2)}). So {v['targets_pass']} vs {v['cal_pass']} passing "
                           f"is not evidence of a real change.")

    # ---- like-for-like diversity and window notes
    for k in ("shannon", "cv2"):
        lf = ex[f"{k}_like_for_like"]
        v[f"{k}_g99"] = pm(lf["rebuild_gen99_mean"], lf["rebuild_gen99_sd"])
        v[f"{k}_p99"] = fmt(lf["paper_fig_gen99"], 2)
        v[f"{k}_pmean"] = fmt(lf["paper_fig_mean_gens0_99"], 2)
        v[f"{k}_def"] = lf["frozen_definition"]
        v[f"{k}_tm"] = pm(row[k]["et"]["mean"], row[k]["et"]["sd"])
        v[f"{k}_paper"] = pm(*row[k]["paper"]["et"])
    sh_sd = row["shannon"]["paper"]["et"][1]
    sh_e = (ex["shannon_like_for_like"]["rebuild_gen99_mean"] - row["shannon"]["paper"]["et"][0]) / sh_sd
    v["shannon_g99_e"] = fmt(sh_e, 1)
    v["shannon_g99_verdict"] = "within" if abs(sh_e) <= 1 else "outside"
    def within_sd(m, key):
        pv, psd = row[key]["paper"]["et"]
        return "within the paper's SD" if abs(m - pv) <= psd else "outside the paper's SD"
    lf = ex["cv2_like_for_like"]
    v["cv2_within_sentence"] = (f"frozen definition {within_sd(row['cv2']['et']['mean'], 'cv2')}; "
                                f"generation-99 value {within_sd(lf['rebuild_gen99_mean'], 'cv2')}")
    ff = ex["final_fitness_gen99"]
    v["ff_within_sentence"] = (f"last-10 mean {within_sd(row['final_fitness']['et']['mean'], 'final_fitness')}; "
                               f"generation-99 value {within_sd(ff['mean'], 'final_fitness')}")
    v["ff_g99"] = pm(ff["mean"], ff["sd"])
    v["ff_l10"] = pm(row["final_fitness"]["et"]["mean"], row["final_fitness"]["et"]["sd"])
    v["ff_paper"] = pm(*row["final_fitness"]["paper"]["et"])
    a1 = ex["transfer_fit_gens1_10"]
    v["t110"] = pm(a1["mean"], a1["sd"])
    p_t = row["transfer_first10"]["paper"]["et"]
    v["t110_verdict"] = "within" if abs(a1["mean"] - p_t[0]) <= p_t[1] else "outside"
    v["t09"] = pm(row["transfer_first10"]["et"]["mean"], row["transfer_first10"]["et"]["sd"])
    v["t09_paper"] = pm(*p_t)

    # ---- extinction and grid
    e = c["extinction"]["GS"]
    v["gs_extinct"] = e["extinct_runs"]
    v["gs_first_min"], v["gs_first_med"], v["gs_first_max"] = e["first_zero_min"], fmt(e["first_zero_median"], 1), e["first_zero_max"]
    al = c["gs_runs_alive_at_gen"]
    v["gs_alive_20"], v["gs_alive_50"], v["gs_alive_99"] = al["20"], al["50"], al["99"]
    v["et_extinct"] = c["extinction"]["ET"]["extinct_runs"]
    g = c["grid"]
    v["k"], v["n_designs"], v["best_index"] = g["points_per_axis"], g["n_designs"], g["best_index"]
    for t in ("speed", "size", "sense"):
        v[f"best_{t}"] = fmt(g["best"][t], 2)
        v[f"best_pos_{t}"] = g["best_axis_index"][t] + 1
    v["ties"] = g["ties_at_best"]
    v["best_ticks"] = fmt(g["best_energy_ticks"], 1)
    d = cfg["blob_default"]
    dcost = metrics.blob_cost(np.array([d["speed"]]), np.array([d["size"]]), np.array([d["sense"]]))[0]
    v["default_ticks"] = fmt(cfg["world"]["energy_per_generation"] / dcost, 0)
    v["best_search_score"], v["second_score"] = fmt(g["best_search_score"], 3), fmt(g["second_best_score"], 3)
    v["best_search_final_pop"] = g["best_search_final_population"]
    dep = g["deployed_score_first30"]
    v["dep"], v["dep_n"] = f"{fmt(dep['mean'], 3)} ± {fmt(dep['sd'], 3)}", dep["n"]
    v["search_steps"] = fmt(g["search_creature_steps"] / 1e6, 1) + "M"
    v["gs_deaths"], v["gs_deaths_fed"] = f"{g['gs_deaths']:,}", f"{g['gs_deaths_after_eating']:,}"
    v["gs_deaths_fed_pct"] = fmt(100 * g["gs_deaths_after_eating"] / g["gs_deaths"], 0)
    gp = ex["grid_score_vs_pop"]
    v["rho"] = fmt(gp["spearman"], 2)
    v["n_above1"], v["med_pop_above1"] = gp["n_score_above_1"], fmt(gp["median_final_pop_score_above_1"], 0)
    v["top10_min"], v["top10_max"] = gp["top10_final_pop_min"], gp["top10_final_pop_max"]

    # ---- paper numbers used in the prose (from the yaml)
    tp = {t["id"]: t for t in paper["text"]}
    v["p_gs_pop"] = tp["gs_population_gen100"]["value"]
    v["p_et_b_pop"] = tp["et_transfer_population"]["value"]
    v["p_et_b_fit"] = fmt(tp["et_transfer_stable"]["value"], 2)
    v["p_compute_et"] = fmt(tp["compute_et"]["value"] / 1e6, 1) + "M"
    v["p_compute_gs"] = fmt(tp["compute_gs"]["value"] / 1e6, 1) + "M"
    p6 = held["fig6_fitness_transfer"]["curves"]["et"]
    v["p_et_b_peak"] = fmt(max(x for gg, x in p6.items() if int(gg) <= 9), 2)
    pg0 = ex["paper_gen0_food"]
    v["pg0_fit"], v["pg0_pop"], v["pg0_food"] = fmt(pg0["paper_fitness"], 2), fmt(pg0["paper_population_fig7"], 0), fmt(pg0["food_needed"], 1)

    # ---- ET transfer facts
    f = c["et_transfer_facts"]
    v["b_max_eaten"], v["b_mean_eaten"] = f["max_food_eaten_per_generation_B"], fmt(f["mean_food_eaten_per_generation_B"], 2)
    v["b_prey_total"] = f["blobs_eaten_total_all_runs"]
    v["p_food_needed"] = fmt(f["paper_food_needed_late_B"], 1)
    v["p_min_pop_b"], v["r_min_pop_b"] = fmt(f["paper_min_population_B"], 1), fmt(f["min_mean_population_B"], 1)
    v["gen0_all_eaten"] = f["B_gen0_runs_all_food_eaten"]
    v["size_99"], v["size_101"] = fmt(f["size_mean_gen99"], 2), fmt(f["size_mean_gen101"], 2)
    v["r_b_peak"] = fmt(f["rebuild_max_fitness_B_gens0_9"], 2)
    v["p_b_2to9"], v["r_b_2to9"] = fmt(f["paper_mean_fitness_B_gens2_9"], 2), fmt(f["rebuild_mean_fitness_B_gens2_9"], 2)
    v["size_live"], v["n_live"] = fmt(f["B_gen0_size_survivors"], 2), f["B_gen0_n_survivors"]
    v["size_dead"], v["n_dead"] = fmt(f["B_gen0_size_died"], 2), f["B_gen0_n_died"]
    r_late = tx["et_transfer_population"]
    v["r_et_b_pop"] = pm(r_late["rebuilt"], r_late["rebuilt_sd"], 1)
    v["r_et_b_fit"] = pm(tx["et_transfer_stable"]["rebuilt"], tx["et_transfer_stable"]["rebuilt_sd"])
    v["r_et_gap"] = pm(row["gap"]["et"]["mean"], row["gap"]["et"]["sd"])
    v["p_et_gap"] = pm(*row["gap"]["paper"]["et"])
    v["r_gs_fit_n"] = row["final_fitness"]["gs"]["n"]
    v["r_compute_et"] = fmt(ce["rebuilt"] / 1e6, 1) + "M"
    v["compute_et_pct"] = fmt(100 * (ce["rebuilt"] / ce["value"] - 1), 0)
    v["r_compute_gs"] = fmt(tx["compute_gs"]["rebuilt"] / 1e6, 1) + "M"
    sa = c["survival_all_training"]["GS"]
    v["gs_surv_all"], v["gs_surv_all_n"] = pm(sa["mean"], sa["sd"], 3), sa["n"]
    v["gs_surv_n"] = tx["gs_survival"]["n"]
    cb = c["calibration_blob_eating"]
    v["cal_off_score"], v["cal_off_pass"] = fmt(cb["off"]["score"], 3), cb["off"]["n_pass"]
    v["cal_on_score"], v["cal_on_pass"] = fmt(cb["on"]["score"], 3), cb["on"]["n_pass"]
    v["cal_n_targets"] = cb["off"]["n_targets"]
    cur = {x["curve"]: x for x in c["curves"]}
    s3, r3 = cur["Fig. 3 ET size, transfer (gens 100-149)"], cur["Fig. 3 ET reach, transfer (gens 100-149)"]
    v["size_paper_b"], v["size_rebuild_b"] = fmt(s3["paper_mean"], 2), fmt(s3["rebuilt_mean"], 2)
    v["reach_paper_b"], v["reach_rebuild_b"] = fmt(r3["paper_mean"], 2), fmt(r3["rebuilt_mean"], 2)
    for key, label in (("Fig. 6 ET fitness, transfer", "sh_fit"), ("Fig. 7 ET population, transfer", "sh_pop"),
                       ("Fig. 9 ET CV2, transfer", "sh_cv2"), ("Fig. 3 ET speed, transfer (gens 100-149)", "sh_speed"),
                       ("Fig. 3 ET sense, transfer (gens 100-149)", "sh_sense"),
                       ("Fig. 3 ET size, transfer (gens 100-149)", "sh_size"),
                       ("Fig. 3 ET reach, transfer (gens 100-149)", "sh_reach")):
        v[label] = fmt(100 * cur[key]["share_within_10pct"], 0) + "%"

    # ---- added after the second independent check
    gsplit = ex["grid_split"]
    v["n_died_out"], v["max_died_out"] = gsplit["n_died_out"], fmt(gsplit["max_score_died_out"], 3)
    v["n_alive_designs"], v["rho_alive"] = gsplit["n_alive"], fmt(gsplit["rho_alive"], 2)
    v["n_above1_b"], v["rho_above1"] = gsplit["n_above_1"], fmt(gsplit["rho_above_1"], 2)
    v["start_pop"] = gsplit["start_population"]
    lf_sh = ex["shannon_like_for_like"]
    v["shannon_50"] = pm(lf_sh["rebuild_gens50_99_mean"], lf_sh["rebuild_gens50_99_sd"])
    v["shannon_p50"] = fmt(lf_sh["paper_fig_mean_gens50_99"], 2)
    v["r_min_pop_early"] = fmt(ex["et_min_pop_B_gens1_9"], 1)
    ga = ex["gen0_averaging"]
    v["avg_prod"] = fmt(ga["product"], 2)
    v["avg_fit"], v["avg_pop"] = fmt(ga["rebuild_fitness_mean"], 3), fmt(ga["rebuild_population_mean"], 2)
    # verdicts as the rules give them
    v["v_et_train"] = ("meets the plan's rule (every target passes)" if v["targets_pass"] == v["targets_n"] else
                       f"does not meet the plan's rule (every target passes): {v['targets_pass']} of {v['targets_n']} pass")
    v["v_et_transfer"] = (f"{v['et_rep']} of {v['et_n']} held-out numbers within tolerance"
                          + (f" ({v['et_rep_post']} with the after-data reading)" if v["et_rep_post"] != v["et_rep"] else ""))
    zero_rep = [i for i in gs_items if i["verdict"] == "replicates" and i["note"].startswith("zero by design")]
    few_rep = [i for i in gs_items if i["verdict"] == "replicates" and i["note"].startswith("rests on")]
    other_rep = [i for i in gs_items if i["verdict"] == "replicates" and i not in zero_rep and i not in few_rep]
    v["v_gs"] = (f"{v['gs_rep']} of {v['gs_n']} held-out numbers within tolerance, but {len(zero_rep)} of these are "
                 f"zero by design for identical clones and {len(few_rep)} rests on {tx['gs_survival']['n']} runs"
                 + ("; no other GS number is within tolerance" if not other_rep else
                    f"; {len(other_rep)} other(s) within tolerance"))
    # sentences that used to be hand-written
    v["gs_when"] = ("during training" if c["extinction"]["GS"]["extinct_in_training"] == c["extinction"]["GS"]["extinct_runs"]
                    else "during training or transfer")
    v["edge_ties"] = (("It is not on the edge of the grid" if not g["on_grid_edge"] else "It is on the edge of the grid")
                      + (" and has no ties." if g["ties_at_best"] == 1 else f" and ties with {g['ties_at_best'] - 1} other(s)."))
    v["size_after"] = ("It then drifted back down" if f["size_mean_gen101"] > s3["rebuilt_mean"] else "It then stayed up")
    sp = ex["side_provenance"]
    ident = all(x["training_identical_runs"] == x["runs"] for x in sp)
    v["side_identical"] = ("bit-identical to the main run's (checked for every run)" if ident else
                           "NOT identical to the main run's in some runs")

    def prov_name(st):
        return st.split(": ", 1)[1] + (" (main)" if st.startswith("main") else "")
    v["side_prov"] = "; ".join(f"{prov_name(x['setting'])} — runs `{x['run_commit']}` "
                               f"({'clean' if not x['run_dirty'] else 'DIRTY'}), analysis `{x['analysis_commit']}`"
                               for x in sp)
    tg = ex["trait_target_gens"]
    v["trait_targets"] = "; ".join(f"{k}: generations {join_words(str(gg) for gg, _ in vals)}" for k, vals in tg.items())
    v["text_targets"] = join_words(sorted({k for k, vals in tg.items() for gg, src in vals if not src.startswith("Fig")}))
    tbd = ex["targets_borderline"]
    v["targets_border"] = join_words(
        f"`{t['id']}` ({fmt(t['fresh_e'], 2)} on seeds 0–29, {fmt(t['cal_e'], 2)} on seeds 1000–1029)" for t in tbd) or "none"

    # ---- tables
    rows = []
    for r in c["fig11"]:
        p = r["paper"]
        rows.append([r["row"], f"{STATUS[r['et']['status']]} / {STATUS[r['gs']['status']]}",
                     pm(*p["et"]), f"{pm(r['et']['mean'], r['et']['sd'])} ({r['et']['n']})",
                     mark(r["et"]["verdict"], r["et"]["posthoc_verdict"]),
                     pm(*p["gs"]), f"{pm(r['gs']['mean'], r['gs']['sd'])} ({r['gs']['n']})",
                     mark(r["gs"]["verdict"], r["gs"]["posthoc_verdict"]), f"`{r['definition']}`"])
    v["T_fig11"] = md_table(["Row", "Status (ET / GS)", "ET paper", "ET rebuild (n)", "ET within?", "GS paper",
                             "GS rebuild (n)", "GS within?", "Definition used"], rows)
    rows = []
    for r in c["fig11"]:
        p, s = r["paper"], r["stats"]
        rows.append([r["row"], f"{fmt(p['t'])} / {fmt(s['t'])}", f"{fmt(p['df'], 1)} / {fmt(s['df'], 1)}",
                     f"{p['p']} / {fmt(s['p'], 4)}", fmt(s["p_bonferroni"], 4), f"{fmt(p['d'])} / {fmt(s['d'])}",
                     f"{p['sig']} / {s['stars']}"])
    v["T_stats"] = md_table(["Row", "t paper / rebuild", "df paper / rebuild", "p paper (printed) / rebuild (uncorrected)",
                             "p rebuild × 10 (Bonferroni)", "d paper / rebuild", "stars paper / rebuild"], rows)
    rows = []
    for t in c["text"]:
        big = t["id"].startswith("compute")
        nd = 3 if "survival" in t["id"] else 2
        pv = fmt(t["value"] / 1e6, 1) + "M" if big else pm(t["value"], t["sd"], nd)
        rv = (fmt(t["rebuilt"] / 1e6, 2) + "M") if big and finite(t["rebuilt"]) else pm(t["rebuilt"], t["rebuilt_sd"], nd)
        tol = fmt(t["tolerance"] / 1e6, 2) + "M" if big else fmt(t["tolerance"], 3)
        rows.append([t["what"], pv, t["source"], STATUS[t["status"]], f"{rv} ({t['n']})", t["how"],
                     f"{tol} ({t['tolerance_basis']})", mark(t["verdict"], t["posthoc_verdict"])])
    v["T_text"] = md_table(["What", "Paper", "Where", "Status", "Rebuild (n)", "How the rebuild number is computed",
                            "Tolerance", "Within?"], rows)
    rows = []
    for b in c["fig8"]:
        rows.append([b["condition"], f"{fmt(b['paper_bar'], 3)} ± {fmt(b['paper_error_half'], 3)}",
                     "transfer gens 0–4" if b["window"].endswith("first5") else "transfer gens 0–9",
                     f"{pm(b['rebuilt'], b['rebuilt_sd'], 3)} ({b['n']})", mark(b["verdict"])])
    v["T_fig8"] = md_table(["Group", "Paper bar ± error bar (read from Fig. 8)",
                            "Rebuild window (late training = last 10 training gens)", "Rebuild (n)",
                            "Within the error bar?"], rows)
    v["fig8_et_err"] = fmt(held["fig8_gap_bars"]["et"]["error_bar_half_length"], 3)
    v["fig8_gs_err"] = fmt(held["fig8_gap_bars"]["gs"]["error_bar_half_length"], 3)
    v["tab_et_gap_sd"] = fmt(row["gap"]["paper"]["et"][1], 2)
    v["tab_gs_gap_sd"] = fmt(row["gap"]["paper"]["gs"][1], 2)
    rows = []
    for x in c["curves"]:
        src = ("calibration/digitized_paper_figures.json" if x["curve"].startswith("Fig. 3")
               else "paper_heldout_digitized.json")
        if x["generations_compared"] == 0:
            rows.append([x["curve"], f"0 of {x['paper_generations']}", "–", "–", "–", "–",
                         "no rebuilt runs alive in this phase", src])
            continue
        rows.append([x["curve"], f"{x['generations_compared']} of {x['paper_generations']}", fmt(x["paper_mean"], 3),
                     fmt(x["rebuilt_mean"], 3), fmt(x["mean_abs_diff"], 3), f"{fmt(100 * x['share_within_10pct'], 0)}%",
                     f"{x['min_runs_behind_rebuilt']} ({x['rebuilt_how']})", src])
    v["T_curves"] = md_table(["Curve", "Generations compared", "Paper mean", "Rebuild mean", "Mean gap",
                              "Generations within 10%", "Fewest rebuilt runs behind a point", "Paper reading from"], rows)
    rows = [[t["id"], fmt(t["paper"], 3), fmt(t["fresh"], 3), fmt(t["fresh_e"], 2), "pass" if t["cal_pass"] else "fail",
             fmt(t["cal"], 3), fmt(t["cal_e"], 2)] for t in tb if not (t["fresh_pass"] and t["cal_pass"])]
    v["T_targets"] = md_table(["Target", "Paper", "Seeds 0–29", "Error ÷ tolerance", "Seeds 1000–1029",
                               "Value there", "Error ÷ tolerance there"], rows)
    sides = [s for s in c["side_checks"] if not s.get("missing")]
    names = [s["setting"] for s in sides]
    rows = []
    for i, m0 in enumerate(sides[0]["metrics"]):
        cells = [f"{pm(s['metrics'][i]['rebuilt'], s['metrics'][i]['rebuilt_sd'])} "
                 f"{mark(s['metrics'][i]['verdict'], s['metrics'][i]['posthoc_verdict'])}" for s in sides]
        rows.append([m0["metric"], pm(m0["paper"], m0["paper_sd"])] + cells)
    v["T_side"] = md_table(["ET metric (n = 30 per cell)", "Paper"] + names, rows)
    v["side_dirty"] = "all clean" if all(not s["commit"]["dirty"] for s in sides) else "SOME DIRTY"
    def setting_name(st):
        return st.split(": ", 1)[1] if ": " in st else st

    all_miss = []
    for i, m0 in enumerate(sides[0]["metrics"]):
        if all(s["metrics"][i]["verdict"] == "misses" for s in sides):
            ph = {s["metrics"][i]["posthoc_verdict"] for s in sides}
            all_miss.append(f"“{m0['metric']}”" + (" (with the after-data reading it matches)" if ph == {"replicates"} else ""))
    mixed = []
    for i, m0 in enumerate(sides[0]["metrics"]):
        vs = [s["metrics"][i]["verdict"] for s in sides]
        if len(set(vs)) > 1:
            miss_in = [setting_name(s["setting"]) for s in sides if s["metrics"][i]["verdict"] == "misses"]
            mixed.append(f"“{m0['metric']}” misses only under {join_words(miss_in)}")
    v["side_all_miss"] = join_words(all_miss) or "none"
    v["side_mixed"] = (". ".join(mixed) + ".") if mixed else "No verdict changes between settings."
    # borderline verdicts anywhere in the comparison
    border = []
    for r in c["fig11"]:
        for grp in ("et", "gs"):
            ee = r[grp]["e"]
            if ee is not None and BORDER[0] <= abs(ee) <= BORDER[1]:
                border.append(f"{grp.upper()} “{r['row']}” (error = {fmt(ee, 2)} tolerances)")
    for t in c["text"]:
        if t["e"] is not None and BORDER[0] <= abs(t["e"]) <= BORDER[1]:
            border.append(f"“{t['what']}” (error = {fmt(t['e'], 2)} tolerances)")
    for s in sides:
        for m in s["metrics"]:
            if m["e"] is not None and BORDER[0] <= abs(m["e"]) <= BORDER[1]:
                border.append(f"“{m['metric']}” under {setting_name(s['setting'])} (error = {fmt(m['e'], 2)} tolerances)")
    v["borderline"] = join_words(border) if border else "none among the held-out numbers"
    rows = [[t["index"], ", ".join(f"{k} {x}" for k, x in t["design"].items() if k != "reach"), fmt(t["score"], 3),
             t["final_population"]] for t in g["top10"]]
    v["T_top10"] = md_table(["Design #", "Traits", f"Search score (seed {v['search_seed']})",
                             f"Blobs left after {v['eval_gens']} search generations"], rows)
    def cut(t, n=260):
        if not t.endswith("..."):
            return t
        t = t[:-3]
        return t[:t.rfind(" ")] + " … (cut; full text at that spec line)"
    v["L_guess"] = "\n".join(f"- **{x['id']}** (spec line {x['line']}): {cut(x['rule'])}" for x in c["guess_rules"])
    v["n_guess"] = len(c["guess_rules"])
    v["L_incons"] = "\n".join(f"- {x}" for x in c["paper_inconsistencies"])
    fd = c["frozen_definitions"]
    defs_rows = [[k, f"`{x}`"] for k, x in fd.items()]
    defs_rows += [["transfer_first10", "`transfer_fit_first10` (transfer gens 0–9; spec M5)"],
                  ["transfer_last10", "`transfer_fit_last10` (transfer gens 40–49; spec M6)"],
                  ["gap", f"`{row['gap']['definition']}` (same final-fitness window as the final-fitness row; spec M7)"],
                  ["slope", "`transfer_slope` (least squares over transfer gens 0–49; spec M8)"]]
    v["T_defs"] = md_table(["Metric", "Definition used"], defs_rows)
    gsx = c["extinction"]["GS"]
    degenerate = gsx["extinct_runs"] == v["n_gs"]
    b_before10 = sum(1 for z in gsx["first_zero_gen"] if z is not None and z < 110)
    yes = "**yes**"
    r2 = [["The grid search's winner sits on the edge of the grid",
           f"winner at position {v['best_pos_speed']}/{v['k']} (speed), {v['best_pos_size']}/{v['k']} (size), "
           f"{v['best_pos_sense']}/{v['k']} (sense)", yes if g["on_grid_edge"] else "no"],
          ["The winner ties with other designs", f"{v['ties']} design(s) at the best score", "no" if v["ties"] == 1 else yes],
          ["The winner is degenerate, so GS is a different thing from the paper's GS",
           f"its population dies out in {gsx['extinct_runs']} of {v['n_gs']} runs; the paper's GS kept about {v['p_gs_pop']} blobs",
           yes if degenerate else "no"],
          ["All GS runs die out in B before transfer generation 10",
           f"{b_before10} of {v['n_gs']} GS runs had no blobs by transfer generation 10 "
           f"({gsx['extinct_in_training']} of them already in training)",
           (yes + (" (worse: no GS run reached B)" if gsx["extinct_in_training"] == v["n_gs"] else ""))
           if b_before10 == v["n_gs"] else "no"],
          ["ET dies out in some runs", f"{v['et_extinct']} of {v['n_et']} runs", "no" if v["et_extinct"] == 0 else yes]]
    v["n_r2_fired"] = sum(1 for r in r2 if r[2].startswith("**yes"))
    v["T_r2"] = md_table(["Useless if (fixed before the run)", "What happened", "Triggered?"], r2)

    tmpl = open(os.path.join(ROOT, "docs", "templates", "REPLICATION_part1.md")).read()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tmpl)) - set(v))
    if missing:
        raise KeyError(f"template blanks with no value: {missing}")
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: str(v[m.group(1)]), tmpl)
    out = os.path.join(ROOT, "docs", "REPLICATION.md")
    with open(out, "w") as fh:
        fh.write(text)
    unused = sorted(set(v) - set(re.findall(r"\{\{(\w+)\}\}", tmpl)))
    print("wrote", os.path.relpath(out, ROOT), "| unused values:", unused)


if __name__ == "__main__":
    main()
