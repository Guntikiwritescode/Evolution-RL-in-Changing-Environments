"""Compare the reconstructed experiment with the original paper's numerical results.

Reference values come from docs/paper_numbers_part1.yaml, the held-out curves in
results/part1/paper_heldout_digitized.json and the calibration curves in
calibration/digitized_paper_figures.json.

Comparison rule fixed before the run:
a number replicates if the rebuilt mean is within the paper's SD across seeds, or within 10%
of the paper value where no SD is given. A printed SD of 0.00 is taken as 0, so the rule then
asks for an exact match (differences below 1e-12, i.e. floating-point round-off, count as
equal). Exception: calibration-target rows use calibration's own rule (calibration/targets.json,
set before calibration), which uses 10% where the paper prints SD 0.00.
Where the rule as written asks for an exact match, a second reading is also reported: the
paper's printed precision (+/- 0.005). That reading was chosen after seeing the data and is
always labeled as such; it never replaces the verdict.

Usage: python scripts/part1/compare_paper.py --name final
Writes results/part1/<name>/comparison.json.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import calibration, metrics  # noqa: E402

TRAIN, TRANSFER = 100, 50
EPS = 1e-12  # floating-point round-off: smaller differences count as equal
POSTHOC_TOL = 0.005  # paper's printed precision; reading chosen after seeing the data


# ------------------------------------------------------------------ helpers
def finite(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def fmt(x, nd=2):
    if not finite(x):
        return "n/a"
    s = f"{float(x):.{nd}f}"
    return "0." + "0" * nd if s in ("-0." + "0" * nd,) else s


def fmt_big(x):
    return "n/a" if not finite(x) else f"{float(x) / 1e6:.1f}M"


def tolerance(value, sd, status="held_out"):
    if status == "target" and sd == 0 and value != 0:
        return 0.1 * abs(value), "10% (calibration's rule for a printed SD of 0.00, set before calibration)"
    if sd is not None:
        return float(sd), "paper SD" if sd > 0 else "paper SD printed 0.00: exact match required"
    if value != 0:
        return 0.1 * abs(value), "10% (no SD printed)"
    return 0.0, "paper value 0 with no SD: exact match required"


def judge(paper, sd, rebuilt, n, status="held_out"):
    tol, basis = tolerance(paper, sd, status)
    out = {"tolerance": tol, "tolerance_basis": basis, "e": None, "verdict": "no value",
           "posthoc_verdict": None}
    if not finite(rebuilt) or n == 0:
        return out
    diff = abs(float(rebuilt) - paper)
    out["e"] = (float(rebuilt) - paper) / tol if tol > 0 else None
    out["verdict"] = "replicates" if diff <= tol + EPS else "misses"
    if tol == 0:  # the rule as written asks for an exact match; add the labeled after-the-data reading
        out["posthoc_verdict"] = "replicates" if diff <= POSTHOC_TOL + EPS else "misses"
    return out


def mean_sd_n(vals):
    v = np.array([x for x in vals if finite(x)], dtype=float)
    n = int(v.size)
    return (float(v.mean()) if n else None, float(v.std(ddof=1)) if n >= 2 else None, n)


def load(folder, cond):
    out = []
    i = 0
    while True:
        path = os.path.join(folder, f"{cond}_seed{i:04d}.npz")
        if not os.path.exists(path):
            break
        z = np.load(path)
        out.append({k: z[k] for k in z.files if k != "meta"})
        i += 1
    return out


def curve(series_list, key, zero_if_extinct=False):
    st = np.stack([s[key] for s in series_list])
    if zero_if_extinct:
        st = np.nan_to_num(st, nan=0.0)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(st, axis=0)
    n_alive = np.sum(~np.isnan(np.stack([s[key] for s in series_list])), axis=0)
    return m, n_alive


def compare_curve(paper_curve, rebuilt, n_alive, gens, skip=()):
    """Mean |difference| and share of generations within 10% of the paper value."""
    rows = []
    for g in gens:
        if str(g) not in paper_curve or g in skip:
            continue
        if n_alive[g] == 0 or not finite(rebuilt[g]):
            rows.append((g, paper_curve[str(g)], None))
            continue
        rows.append((g, paper_curve[str(g)], float(rebuilt[g])))
    both = [(g, p, r) for g, p, r in rows if r is not None]
    if not both:
        return {"generations_compared": 0, "paper_generations": len(rows), "mean_abs_diff": None,
                "share_within_10pct": None, "min_runs_behind_rebuilt": 0}
    diffs = [abs(r - p) for _, p, r in both]
    within = [abs(r - p) <= 0.1 * abs(p) for _, p, r in both]
    return {"generations_compared": len(both), "paper_generations": len(rows),
            "first_gen": both[0][0], "last_gen": both[-1][0],
            "mean_abs_diff": float(np.mean(diffs)), "share_within_10pct": float(np.mean(within)),
            "paper_mean": float(np.mean([p for _, p, _ in both])), "rebuilt_mean": float(np.mean([r for _, _, r in both])),
            "min_runs_behind_rebuilt": int(min(n_alive[g] for g, _, _ in both))}


def cfg_food_b(info):
    return info["provenance"]["config"]["env_B"]["food"]


def guess_rules(spec_path):
    """Every table row in the spec tagged [GUESS]: (id, item, choice)."""
    out = []
    spec_lines = open(spec_path, encoding="utf-8").readlines()
    for line in spec_lines:
        if not line.startswith("|") or "[GUESS]" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: "):
            continue
        # tables differ in width; keep every cell except the id and the tag cell (last), in order
        text = re.sub(r"\*\*", "", " · ".join(c for c in cells[1:-1] if c))
        out.append({"id": cells[0], "rule": text[:260] + ("..." if len(text) > 260 else ""),
                    "line": spec_lines.index(line) + 1})
    return out


# ------------------------------------------------------------------ main computation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    folder = os.path.join(ROOT, "results", "part1", args.name)
    paper = yaml.safe_load(open(os.path.join(ROOT, "docs", "paper_numbers_part1.yaml")))
    held = json.load(open(os.path.join(ROOT, "results", "part1", "paper_heldout_digitized.json")))["figures"]
    cal_dig = json.load(open(os.path.join(ROOT, "calibration", "digitized_paper_figures.json")))["figures"]
    summary = json.load(open(os.path.join(folder, "summary.json")))
    info = json.load(open(os.path.join(folder, "run_info.json")))
    search = json.load(open(os.path.join(folder, "gs_search.json")))
    with open(os.path.join(folder, "per_run_metrics.csv")) as f:
        per_run = list(csv.DictReader(f))
    defs = summary["definitions"]
    # every frozen definition (calibration stage 2), including ones the Fig. 11 table doesn't use
    frozen = yaml.safe_load(open(os.path.join(ROOT, "configs", "part1", "metric_defs.yaml")))

    def col(cond, key):
        return [float(r[key]) if r[key] not in ("", "nan", "None") else np.nan
                for r in per_run if r["condition"] == cond]

    runs = {c: load(folder, c) for c in ("ET", "GS")}
    ser = {c: [metrics.series(a) for a in runs[c]] for c in runs}
    n_runs = {c: len(runs[c]) for c in runs}
    out = {"name": args.name, "n_runs": n_runs, "provenance": info["provenance"]}

    # ---- extinction and population facts
    ext = {}
    for c in runs:
        first0 = []
        for a in runs[c]:
            z = np.where(a["g_population"] == 0)[0]
            first0.append(int(z[0]) if z.size else None)
        e = [x for x in first0 if x is not None]
        ext[c] = {"extinct_runs": len(e), "extinct_in_training": sum(1 for x in e if x < TRAIN),
                  "extinct_in_transfer": sum(1 for x in e if x >= TRAIN), "first_zero_gen": first0,
                  "first_zero_min": min(e) if e else None, "first_zero_median": float(np.median(e)) if e else None,
                  "first_zero_max": max(e) if e else None}
    out["extinction"] = ext
    gs_alive = {g: int(sum(a["g_population"][g] > 0 for a in runs["GS"])) for g in (0, 10, 20, 30, 50, 80, 99, 100)}
    out["gs_runs_alive_at_gen"] = gs_alive

    # ---- Figure 11 rows
    fig11 = []
    table = {r["metric"]: r for r in summary["table"]}
    for row in paper["fig11"]:
        r = table[row["row"].replace("CV2", "CV2")] if row["row"] in table else None
        if r is None:  # labels match analyze_part1.FIG11
            raise KeyError(row["row"])
        et = judge(row["et"][0], row["et"][1], r["mean_x"], r["n_x"], row["et_status"])
        gs = judge(row["gs"][0], row["gs"][1], r["mean_y"], r["n_y"], row["gs_status"])
        fig11.append({"row": row["row"], "definition": r["definition"], "paper": row,
                      "et": {"mean": r["mean_x"], "sd": r["sd_x"], "n": r["n_x"], **et, "status": row["et_status"]},
                      "gs": {"mean": r["mean_y"], "sd": r["sd_y"], "n": r["n_y"], **gs, "status": row["gs_status"]},
                      "stats": {k: r[k] for k in ("t", "df", "p", "p_bonferroni", "d", "stars")},
                      "zero_by_design": r["zero_by_design"]})
    out["fig11"] = fig11

    # ---- text numbers
    def popat(c, g):
        return mean_sd_n([a["g_population"][g] for a in runs[c]])

    def survival(c):
        return mean_sd_n([np.nanmean(s["survival"][:TRAIN]) if np.any(~np.isnan(s["survival"][:TRAIN])) else np.nan
                          for s in ser[c]])

    def gs_late_transfer():
        vals = [float(np.mean(a["g_population"][TRAIN + 25: TRAIN + 31])) for a in runs["GS"]]
        return mean_sd_n(vals)

    rebuilt_text = {
        "et_population_gen100": popat("ET", TRAIN - 1),
        "gs_population_gen100": popat("GS", TRAIN - 1),
        "et_survival": mean_sd_n(col("ET", frozen["survival"])),
        "gs_survival": mean_sd_n(col("GS", frozen["survival"])),
        "et_transfer_gen0": mean_sd_n(col("ET", "transfer_fit_gen0")),
        "et_transfer_stable": mean_sd_n(col("ET", "transfer_fit_last10")),
        "et_transfer_population": mean_sd_n(col("ET", "transfer_pop_last10")),
        "gs_transfer_population_late": gs_late_transfer(),
        "slope_text_et": mean_sd_n(col("ET", "transfer_slope")),
        "slope_text_gs": mean_sd_n(col("GS", "transfer_slope")),
        "compute_et": (float(info["ET_creature_steps_total"]), None, n_runs["ET"]),
        "compute_gs": (float(info["GS_creature_steps_total_incl_search"]), None, n_runs["GS"]),
    }
    how = {
        "et_population_gen100": "mean population at generation 99 (last training generation), as in calibration",
        "gs_population_gen100": "same, GS; extinct runs count as 0",
        "et_survival": "per run: share of blobs that survive a generation, averaged over the last 10 training generations (frozen definition); then mean over runs",
        "gs_survival": "same, GS (only runs with blobs in those generations)",
        "et_transfer_gen0": "fitness in transfer generation 0",
        "et_transfer_stable": "mean fitness over transfer generations 40-49",
        "et_transfer_population": "mean population over transfer generations 40-49",
        "gs_transfer_population_late": "mean population over transfer generations 25-30 (extinct = 0)",
        "slope_text_et": "least-squares slope of fitness over the 50 transfer generations",
        "slope_text_gs": "same, GS",
        "compute_et": "creature-steps, all 30 ET runs, both phases (spec K1)",
        "compute_gs": "creature-steps, grid search + all 30 GS runs (spec K1)",
    }
    def tp_val(i):
        return next(t["value"] for t in paper["text"] if t["id"] == i)

    text_rows = []
    for t in paper["text"]:
        m, sd, n = rebuilt_text[t["id"]]
        text_rows.append({**t, "rebuilt": m, "rebuilt_sd": sd, "n": n, "how": how[t["id"]],
                          **judge(t["value"], t["sd"], m, n, t["status"])})
    out["text"] = text_rows
    out["frozen_definitions"] = frozen
    # survival over all training generations (calibration's target definition), for the note
    out["survival_all_training"] = {c: dict(zip(("mean", "sd", "n"), survival(c))) for c in ("ET", "GS")}

    # ---- Fig. 8 bar chart (held out, read from the figure, error bars = paper spread)
    gaps = {}
    for c in ("ET", "GS"):
        for w in ("gap_last10_first10", "gap_last10_first5", "gap_last1_first10", "gap_last1_first5"):
            gaps[(c, w)] = mean_sd_n(col(c, w))
    fig8 = []
    for c in ("ET", "GS"):
        b = held["fig8_gap_bars"][c.lower()]
        for w in ("gap_last10_first5", "gap_last10_first10"):
            m, sd, n = gaps[(c, w)]
            fig8.append({"condition": c, "window": w, "paper_bar": b["bar_top"],
                         "paper_error_half": b.get("error_bar_half_length"), "rebuilt": m, "rebuilt_sd": sd, "n": n,
                         **judge(b["bar_top"], b.get("error_bar_half_length"), m, n)})
    out["fig8"] = fig8

    # ---- held-out curves read from figures
    curves = []
    et_fit, et_fit_n = curve(ser["ET"], "fitness")
    et_pop, et_pop_n = curve(ser["ET"], "population", True)
    et_cv2, et_cv2_n = curve(ser["ET"], "cv2_4_samp")
    gs_fit, gs_fit_n = curve(ser["GS"], "fitness")
    gs_pop0, _ = curve(ser["GS"], "population", True)
    gs_pop_alive_n = np.sum(np.stack([s["population"] for s in ser["GS"]]) > 0, axis=0)
    gs_cv2, gs_cv2_n = curve(ser["GS"], "cv2_4_samp")
    full_n = np.full(TRAIN + TRANSFER, n_runs["GS"])
    spec_c = [
        ("Fig. 1 GS fitness, training", held["fig1_gs_fitness_train"]["curves"]["gs"], gs_fit, gs_fit_n, 0,
         "mean over runs with blobs alive"),
        ("Fig. 2 GS population, training", held["fig2_gs_population_train"]["curves"]["gs"], gs_pop0, full_n, 0,
         "mean over all runs, extinct = 0"),
        ("Fig. 6 ET fitness, transfer", held["fig6_fitness_transfer"]["curves"]["et"], et_fit, et_fit_n, TRAIN,
         "mean over runs"),
        ("Fig. 6 GS fitness, transfer", held["fig6_fitness_transfer"]["curves"]["gs"], gs_fit, gs_fit_n, TRAIN,
         "mean over runs with blobs alive"),
        ("Fig. 7 ET population, transfer", held["fig7_population_transfer"]["curves"]["et"], et_pop, et_pop_n, TRAIN,
         "mean over runs"),
        ("Fig. 7 GS population, transfer", held["fig7_population_transfer"]["curves"]["gs"], gs_pop0, gs_pop_alive_n,
         TRAIN, "mean over runs with blobs alive"),
        ("Fig. 9 ET CV2, transfer", held["fig9_cv2_transfer"]["curves"]["et"], et_cv2, et_cv2_n, TRAIN,
         "mean over runs (4 traits, sample SD)"),
        ("Fig. 9 GS CV2, transfer", held["fig9_cv2_transfer"]["curves"]["gs"], gs_cv2, gs_cv2_n, TRAIN,
         "mean over runs with blobs alive"),
    ]
    skip_fig2 = {0}  # Fig. 2 GS gen 0 reads 46.2 although it is 50 by design (steep digitized edge)
    for label, pc, reb, n_al, offset, how_c in spec_c:
        # paper curves for transfer use transfer generation numbers 0-49; rebuilt series use 100-149
        shifted = {str(int(g) + offset): v for g, v in pc.items()}
        gens = range(offset, offset + (TRANSFER if offset else TRAIN))
        res = compare_curve(shifted, reb, n_al, gens, skip=skip_fig2 if label.startswith("Fig. 2") else ())
        curves.append({"curve": label, "rebuilt_how": how_c, "paper_last_gen_drawn": max(int(g) for g in pc), **res})
    # Fig. 3 traits in transfer (generations 100-149; never calibration targets)
    f3 = cal_dig["fig3_et_traits"]
    for tr in ("speed", "size", "sense", "reach"):
        bad = set(f3.get("interpolated_gens", {}).get(tr, [])) | set(f3.get("extrapolated_gens", {}).get(tr, []))
        reb, n_al = curve(ser["ET"], f"mean_{tr}")
        res = compare_curve(f3["curves"][tr], reb, n_al, range(TRAIN, TRAIN + TRANSFER), skip=bad)
        curves.append({"curve": f"Fig. 3 ET {tr}, transfer (gens 100-149)", "rebuilt_how": "mean over runs",
                       "paper_last_gen_drawn": 149, **res})
    out["curves"] = curves

    # ---- calibration targets re-checked on the final seeds (0-29, never used in calibration)
    targets = json.load(open(os.path.join(ROOT, "calibration", "targets.json")))["targets"]
    sc = calibration.score(targets, [{"series": s} for s in ser["ET"]])
    cal_rec = json.load(open(os.path.join(ROOT, "calibration", "runs", "sd1_reach0.25_cap200_cann0__1a.json")))
    out["targets_recheck"] = {"n_pass": sc["n_pass"], "n_targets": sc["n_targets"], "total_score": sc["total"],
                              "groups": sc["groups"], "failing": [t for t in sc["targets"] if not t["pass"]],
                              "calibration_seeds_n_pass": cal_rec["score"]["n_pass"],
                              "calibration_seeds_failing": [t["id"] for t in cal_rec["score"]["targets"] if not t["pass"]]}

    # ---- R2 "useless if" checks on the grid search
    k = len(search["designs"]) ** (1 / 3)
    k = int(round(k))
    bi = search["best_index"]
    idx = {"speed": bi // (k * k), "size": (bi // k) % k, "sense": bi % k}
    scores = np.array([r["score"] for r in search["results"]])
    finals = np.array([r["final_population"] for r in search["results"]])
    order = np.argsort(-scores)

    def deployed_score(a, n_gens=30):
        pop = a["g_population"][:n_gens].astype(float)
        food = (a["g_food_eaten"] + a["g_prey_eaten"])[:n_gens].astype(float)
        return float(np.mean(np.where(pop > 0, food / np.maximum(pop, 1), 0.0)))

    dep = mean_sd_n([deployed_score(a) for a in runs["GS"]])
    out["grid"] = {
        "n_designs": len(search["designs"]), "points_per_axis": k, "best_index": bi, "best": search["best"],
        "best_axis_index": idx, "on_grid_edge": any(v in (0, k - 1) for v in idx.values()),
        "ties_at_best": search["n_ties_at_best"], "best_search_score": float(scores[bi]),
        "best_search_final_population": int(finals[bi]),
        "second_best_score": float(scores[order[1]]),
        "designs_alive_after_search": int((finals > 0).sum()),
        "designs_with_final_pop_ge_40": int((finals >= 40).sum()),
        "top10": [{"index": int(i), "design": {kk: round(v, 2) for kk, v in search["designs"][i].items()},
                   "score": float(scores[i]), "final_population": int(finals[i])} for i in order[:10]],
        "best_alive_design_score_rank": None,
        "deployed_score_first30": {"mean": dep[0], "sd": dep[1], "n": dep[2]},
        "search_creature_steps": search["search_creature_steps"],
    }
    # highest-scoring design that ended the search with >= 40 blobs (descriptive)
    for rank, i in enumerate(order):
        if finals[i] >= 40:
            out["grid"]["best_full_population_design"] = {
                "rank": rank + 1, "index": int(i), "score": float(scores[i]), "final_population": int(finals[i]),
                "design": {kk: round(v, 2) for kk, v in search["designs"][i].items()}}
            break

    # GS deaths: blobs that died after eating >= 1 food ran out of energy before getting home
    # (blob-eating is off, so nothing else kills a fed blob; spec E2)
    died = died_fed = blob_gens = 0
    for a in runs["GS"]:
        fed = (a["b_food_balls"] + a["b_prey"]) >= 1
        alive = a["b_alive_end"].astype(bool)
        blob_gens += alive.size
        died += int((~alive).sum())
        died_fed += int((~alive & fed).sum())
    out["grid"]["gs_blob_generations"] = blob_gens
    out["grid"]["gs_deaths"] = died
    out["grid"]["gs_deaths_after_eating"] = died_fed
    out["grid"]["top10_ending_below_40"] = int(sum(1 for t in out["grid"]["top10"] if t["final_population"] < 40))
    # winner's energy budget (ticks of energy) vs ET's start
    d = search["best"]
    eff = d["speed"] * d["size"] / 10.0
    cost = metrics.blob_cost(np.array([d["speed"]]), np.array([d["size"]]), np.array([d["sense"]]))[0]
    out["grid"]["best_energy_ticks"] = float(500.0 / cost)
    out["grid"]["best_effective_speed"] = float(eff)
    # ---- facts used to explain the ET transfer misses (measured, all 30 ET runs)
    fe = np.stack([a["g_food_eaten"] for a in runs["ET"]])
    fpl = np.stack([a["g_food_placed"] for a in runs["ET"]])
    prey = np.stack([a["g_prey_eaten"] for a in runs["ET"]])
    size_live, size_dead = [], []
    for a in runs["ET"]:
        sel = a["b_gen"] == TRAIN
        size_live += list(a["b_size"][sel & a["b_alive_end"].astype(bool)])
        size_dead += list(a["b_size"][sel & ~a["b_alive_end"].astype(bool)])
    et_pop_b = np.stack([a["g_population"][TRAIN:] for a in runs["ET"]]).mean(axis=0)
    out["et_transfer_facts"] = {
        "food_placed_per_generation_B": sorted(set(int(v) for v in fpl[:, TRAIN:].ravel())),
        "max_food_eaten_per_generation_B": int(fe[:, TRAIN:].max()),
        "mean_food_eaten_per_generation_B": float(fe[:, TRAIN:].mean()),
        "blobs_eaten_total_all_runs": int(prey.sum()),
        "cannibalism_setting": bool(info["provenance"]["config"]["world"]["cannibalism"]),
        "paper_food_needed_late_B": None,  # filled below from paper numbers
        "mean_population_B_gen1_to_5": [float(v) for v in et_pop_b[1:6]],
        "min_mean_population_B": float(et_pop_b[1:].min()),
        "B_gen0_size_survivors": float(np.mean(size_live)), "B_gen0_n_survivors": len(size_live),
        "B_gen0_size_died": float(np.mean(size_dead)), "B_gen0_n_died": len(size_dead),
        "B_gen0_runs_all_food_eaten": int(np.sum(fe[:, TRAIN] == fpl[:, TRAIN])),
    }
    size_curve, _ = curve(ser["ET"], "mean_size")
    out["et_transfer_facts"]["size_mean_gen99"] = float(size_curve[TRAIN - 1])
    out["et_transfer_facts"]["size_mean_gen101"] = float(size_curve[TRAIN + 1])
    out["et_transfer_facts"]["rebuild_max_fitness_B_gens0_9"] = float(np.nanmax(et_fit[TRAIN:TRAIN + 10]))
    p6 = held["fig6_fitness_transfer"]["curves"]["et"]
    g29 = [g for g in range(2, 10) if str(g) in p6]
    out["et_transfer_facts"]["paper_mean_fitness_B_gens2_9"] = float(np.mean([p6[str(g)] for g in g29]))
    out["et_transfer_facts"]["rebuild_mean_fitness_B_gens2_9"] = float(np.mean([et_fit[TRAIN + g] for g in g29]))
    out["et_transfer_facts"]["paper_gens_read_2_9"] = g29
    tp = {t["id"]: t for t in paper["text"]}
    out["et_transfer_facts"]["paper_food_needed_late_B"] = tp["et_transfer_population"]["value"] * tp["et_transfer_stable"]["value"]
    p7 = held["fig7_population_transfer"]["curves"]["et"]
    out["et_transfer_facts"]["paper_min_population_B"] = float(min(v for g, v in p7.items() if int(g) >= 1))
    # ---- pre-registered Environment B side checks (spec W8, W10, 12.2): ET transfer numbers only
    size_paper = next(c for c in curves if c["curve"].startswith("Fig. 3 ET size"))["paper_mean"]
    f11 = {r["key"]: r for r in paper["fig11"]}
    side_specs = [("main: spread 15, spots fresh each generation", args.name),
                  ("side: spread 7.5", f"{args.name}_side_b_sd7.5"),
                  ("side: spread 30", f"{args.name}_side_b_sd30"),
                  ("side: spots fixed for all of transfer", f"{args.name}_side_b_fixed")]
    side_metrics = [("Transfer fitness, gen 0", "transfer_fit_gen0", tp_val("et_transfer_gen0"), None),
                    ("Transfer fitness (gen 1-10)", "transfer_fit_first10", *f11["transfer_first10"]["et"]),
                    ("Transfer fitness (last 10 gen)", "transfer_fit_last10", *f11["transfer_last10"]["et"]),
                    ("Generalization gap", "gap_last10_first10", *f11["gap"]["et"]),
                    ("Transfer trend slope", "transfer_slope", *f11["slope"]["et"]),
                    ("Population, last 10 transfer gens", "transfer_pop_last10", tp_val("et_transfer_population"), None),
                    ("Mean size, transfer gens 100-149", "size_B", size_paper, None)]
    sides = []
    for label, nm in side_specs:
        fo = os.path.join(ROOT, "results", "part1", nm)
        if not os.path.exists(os.path.join(fo, "per_run_metrics.csv")):
            sides.append({"setting": label, "folder": nm, "missing": True})
            continue
        with open(os.path.join(fo, "per_run_metrics.csv")) as f:
            pr = [r for r in csv.DictReader(f) if r["condition"] == "ET"]
        et_runs = load(fo, "ET")
        size_b = [float(np.mean(metrics.series(a)["mean_size"][TRAIN:TRAIN + TRANSFER])) for a in et_runs]
        row = {"setting": label, "folder": nm,
               "commit": json.load(open(os.path.join(fo, "run_info.json")))["provenance"]["git"], "metrics": []}
        for mlabel, key, pv, psd in side_metrics:
            vals = size_b if key == "size_B" else [float(r[key]) for r in pr]
            m, sd, nn = mean_sd_n(vals)
            row["metrics"].append({"metric": mlabel, "paper": pv, "paper_sd": psd, "rebuilt": m, "rebuilt_sd": sd,
                                   "n": nn, **judge(pv, psd, m, nn)})
        sides.append(row)
    out["side_checks"] = sides

    # best calibration result with blob-eating on vs off (calibration/log.csv, seeds 1000-1029)
    with open(os.path.join(ROOT, "calibration", "log.csv")) as f:
        log_rows = list(csv.DictReader(f))
    best_by_cann = {}
    for r in log_rows:
        key = "on" if r["cannibalism"] in ("True", "1") else "off"
        if key not in best_by_cann or float(r["total_score"]) < best_by_cann[key]["score"]:
            best_by_cann[key] = {"id": r["id"], "score": float(r["total_score"]), "n_pass": int(r["n_pass"]),
                                 "n_targets": int(r["n_targets"])}
    out["calibration_blob_eating"] = best_by_cann
    out["guess_rules"] = guess_rules(os.path.join(ROOT, "docs", "SPEC_part1.md"))
    out["paper_inconsistencies"] = paper["inconsistencies"]

    # ---- extra measured facts asked for by the independent check (all from saved files)
    from scipy import stats as sps
    from evosim import config as evo_config
    extra = {}
    # seed noise behind the training-target misses
    cal_rec_seeds = cal_rec["per_seed_scalars"]
    reach_fresh = np.array([s["mean_reach"][TRAIN - 1] for s in ser["ET"]])
    reach_cal = np.array([p["trait_reach_final"] for p in cal_rec_seeds])
    reach_tol = next(t["tolerance"] for t in targets if t["id"] == "reach_final")
    wt = sps.ttest_ind(reach_fresh, reach_cal, equal_var=False)
    extra["reach_gen99"] = {"fresh_mean": float(reach_fresh.mean()), "fresh_sd": float(reach_fresh.std(ddof=1)),
                            "fresh_min": float(reach_fresh.min()), "fresh_max": float(reach_fresh.max()),
                            "fresh_se": float(reach_fresh.std(ddof=1) / np.sqrt(reach_fresh.size)),
                            "cal_mean": float(reach_cal.mean()), "cal_sd": float(reach_cal.std(ddof=1)),
                            "tolerance": float(reach_tol), "welch_p_fresh_vs_cal": float(wt.pvalue),
                            "n_fresh": int(reach_fresh.size), "n_cal": int(reach_cal.size)}
    # every target on both seed sets, with direction, so the report can describe the misses from data
    cal_rows = {t["id"]: t for t in cal_rec["score"]["targets"]}
    extra["targets_both"] = [{"id": t["id"], "group": t["group"], "paper": t["paper"], "fresh": t["sim"],
                              "fresh_e": t["error_in_tol"], "fresh_pass": t["pass"],
                              "cal": cal_rows[t["id"]]["sim"], "cal_e": cal_rows[t["id"]]["error_in_tol"],
                              "cal_pass": cal_rows[t["id"]]["pass"]} for t in sc["targets"]]
    # like-for-like diversity: value at generation 99 (the paper's 3.77 matches its own Fig. 5 at gen 99)
    for key, figname, col_def in (("H_exact4", "fig5_et_shannon", "shannon"), ("cv2_4_samp", "fig4_et_cv2", "cv2")):
        g99 = np.array([s[key][TRAIN - 1] for s in ser["ET"]])
        pc = cal_dig[figname]["curves"]["et"]
        extra[f"{col_def}_like_for_like"] = {
            "rebuild_gen99_mean": float(g99.mean()), "rebuild_gen99_sd": float(g99.std(ddof=1)), "n": int(g99.size),
            "paper_fig_gen99": float(pc[str(TRAIN - 1)]),
            "paper_fig_mean_gens0_99": float(np.mean([pc[str(g)] for g in range(TRAIN)])),
            "frozen_definition": frozen[col_def]}
    # final training fitness in generation 99 only (the spec's original M3) vs the frozen last-10 window
    extra["final_fitness_gen99"] = dict(zip(("mean", "sd", "n"), mean_sd_n(col("ET", "final_fitness_last1"))))
    # transfer fitness over transfer gens 1-10 instead of the pre-registered 0-9
    alt = [float(np.nanmean(s["fitness"][TRAIN + 1:TRAIN + 11])) for s in ser["ET"]]
    extra["transfer_fit_gens1_10"] = dict(zip(("mean", "sd", "n"), mean_sd_n(alt)))
    # grid search: score vs blobs left, over all designs
    rho = sps.spearmanr(scores, finals)
    top10_pop = [t["final_population"] for t in out["grid"]["top10"]]
    extra["grid_score_vs_pop"] = {"spearman": float(rho.statistic), "n_designs": int(scores.size),
                                  "n_score_above_1": int((scores > 1.0).sum()),
                                  "median_final_pop_score_above_1": float(np.median(finals[scores > 1.0])),
                                  "top10_final_pop_min": int(min(top10_pop)), "top10_final_pop_max": int(max(top10_pop))}
    # the paper's own first transfer generation: fitness x population vs food placed
    p7_0 = held["fig7_population_transfer"]["curves"]["et"]["0"]
    extra["paper_gen0_food"] = {"paper_fitness": tp["et_transfer_gen0"]["value"], "paper_population_fig7": float(p7_0),
                                "food_needed": float(tp["et_transfer_gen0"]["value"] * p7_0),
                                "food_placed": int(cfg_food_b(info))}
    # --- added after the second independent check ---
    # grid search: split the score / blobs-left relation by group (designs that died out score low by rule)
    died_out = finals == 0
    hi = scores > 1.0
    extra["grid_split"] = {
        "n_died_out": int(died_out.sum()), "max_score_died_out": float(scores[died_out].max()) if died_out.any() else None,
        "n_alive": int((~died_out).sum()), "rho_alive": float(sps.spearmanr(scores[~died_out], finals[~died_out]).statistic),
        "n_above_1": int(hi.sum()), "rho_above_1": float(sps.spearmanr(scores[hi], finals[hi]).statistic),
        "start_population": int(info["provenance"]["config"]["gs"]["n_clones"])}
    # Shannon averaged over generations 50-99 (a plateau window), rebuild and paper's Fig. 5
    Hs = np.array([np.nanmean(s["H_exact4"][50:TRAIN]) for s in ser["ET"]])
    pc5 = cal_dig["fig5_et_shannon"]["curves"]["et"]
    extra["shannon_like_for_like"]["rebuild_gens50_99_mean"] = float(Hs.mean())
    extra["shannon_like_for_like"]["rebuild_gens50_99_sd"] = float(Hs.std(ddof=1))
    extra["shannon_like_for_like"]["paper_fig_mean_gens50_99"] = float(np.mean([pc5[str(g)] for g in range(50, TRAIN)]))
    # lowest mean population early in transfer (transfer gens 1-9)
    pop_mean_b = np.stack([a["g_population"] for a in runs["ET"]]).mean(axis=0)
    extra["et_min_pop_B_gens1_9"] = float(pop_mean_b[TRAIN + 1:TRAIN + 10].min())
    # the GS runs behind the survival_last10 value: blobs in each of the last 10 training generations
    extra["gs_survival_runs"] = [{"seed": i, "populations_gens90_99": [int(x) for x in a["g_population"][TRAIN - 10:TRAIN]]}
                                 for i, a in enumerate(runs["GS"]) if a["g_population"][TRAIN - 10:TRAIN].max() > 0]
    # averaging effect: mean of per-run fitness x mean population vs food placed, rebuild, transfer gen 0
    extra["gen0_averaging"] = {"rebuild_fitness_mean": float(np.mean([s["fitness"][TRAIN] for s in ser["ET"]])),
                               "rebuild_population_mean": float(pop_mean_b[TRAIN])}
    extra["gen0_averaging"]["product"] = extra["gen0_averaging"]["rebuild_fitness_mean"] * extra["gen0_averaging"]["rebuild_population_mean"]
    # side checks: provenance, and whether each side run's training phase equals the main run's
    keys_g = ("g_population", "g_food_eaten", "g_prey_eaten", "g_survivors", "g_ticks", "g_creature_steps")
    keys_b = ("b_speed", "b_size", "b_sense", "b_reach", "b_food_balls", "b_alive_end")
    side_prov = []
    for st in out["side_checks"]:
        if st.get("missing"):
            continue
        fo = os.path.join(ROOT, "results", "part1", st["folder"])
        same = 0
        total = 0
        for cnd in ("ET", "GS"):
            for i, a in enumerate(runs[cnd]):
                z = np.load(os.path.join(fo, f"{cnd}_seed{i:04d}.npz"))
                ok = all(np.array_equal(a[k][:TRAIN], z[k][:TRAIN]) for k in keys_g)
                ok = ok and all(np.array_equal(a[k][a["b_gen"] < TRAIN], z[k][z["b_gen"] < TRAIN]) for k in keys_b)
                same += int(ok)
                total += 1
        ac = os.path.join(fo, "analysis_commit.txt")
        side_prov.append({"setting": st["setting"], "run_commit": st["commit"]["commit"][:7],
                          "run_dirty": st["commit"]["dirty"],
                          "analysis_commit": open(ac).read().strip() if os.path.exists(ac) else None,
                          "training_identical_runs": same, "runs": total})
    extra["side_provenance"] = side_prov
    # targets near the tolerance edge on either seed set (|error| between 0.8 and 1.2 tolerances)
    extra["targets_borderline"] = [t for t in extra["targets_both"]
                                   if 0.8 <= abs(t["fresh_e"]) <= 1.2 or 0.8 <= abs(t["cal_e"]) <= 1.2]
    # which generations of each Fig. 3 trait were calibration targets (from targets.json)
    tg = {}
    for t in targets:
        if t["group"] == "traits":
            tr_name = t["metric"].split(":")[1]
            tg.setdefault(tr_name, []).append((int(t["gen"]), t["source"]))
    extra["trait_target_gens"] = {k: sorted(set(v)) for k, v in tg.items()}
    extra["generated_by"] = evo_config.git_commit()
    out["extra"] = extra

    with open(os.path.join(folder, "comparison.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", os.path.join(folder, "comparison.json"))
    return out


if __name__ == "__main__":
    main()
