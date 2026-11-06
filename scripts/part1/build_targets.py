"""Build calibration/targets.json: the ET training-phase targets, the score, the
search space and selection rules, recorded before calibration.

Sources: numbers printed in the paper (ER = Experiment Report pages) and values
read off its figures by scripts/part1/digitize_figures.py.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIG = json.load(open(os.path.join(ROOT, "calibration", "digitized_paper_figures.json")))["figures"]


def chart(fig, curve, g):
    return round(DIG[fig]["curves"][curve][str(g)], 4)


def bad(fig, curve):
    d = DIG[fig]
    return set(d.get("interpolated_gens", {}).get(curve, [])) | set(d.get("extrapolated_gens", {}).get(curve, []))


targets = []


def add(tid, group, metric, value, tol, source, tol_basis, gen=None, definition="primary"):
    t = {"id": tid, "group": group, "metric": metric, "paper_value": value, "tolerance": tol,
         "tolerance_basis": tol_basis, "source": source, "definition": definition}
    if gen is not None:
        t["gen"] = gen
    targets.append(t)


TEN = "10% of the paper value (no SD reported for this point)"

# ---- fitness curve (Fig. 1) and final value (Fig. 11 table)
for g in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90]:
    if g not in bad("fig1_et_fitness", "et"):
        v = chart("fig1_et_fitness", "et", g)
        add(f"fitness_g{g}", "fitness", "mean_fitness_at_gen", v, round(0.1 * v, 4), "Fig. 1 (digitized)", TEN, gen=g)
add("fitness_final", "fitness", "final_fitness", 1.03, 0.03, "Fig. 11 table, ER p.4", "paper SD across seeds")
add("fitness_peak", "fitness", "max_mean_fitness_gens_1_10", 1.121, 0.1121, "Fig. 1 (digitized max)", TEN)
add("fitness_peak_gen", "fitness", "argmax_mean_fitness_gens_1_10", 5, 2, "Fig. 1 (digitized; gens 4 and 5 within 0.002)",
    "+/- 2 generations [GUESS]")
add("median_final", "fitness", "final_median", 1.00, 0.10, "Fig. 11 table", "10% (paper SD is 0.00, unusable as a tolerance)")

# ---- population curve (Fig. 2) and final value (ER p.4)
for g in [1, 2, 5, 6, 8, 10, 12, 14, 20, 30, 40, 50, 60, 70, 80, 90]:
    if g not in bad("fig2_et_population", "et"):
        v = chart("fig2_et_population", "et", g)
        add(f"population_g{g}", "population", "mean_population_at_gen", v, round(0.1 * v, 3), "Fig. 2 (digitized)", TEN, gen=g)
add("population_final", "population", "population_at_gen", 48.0, 4.8, "ER p.4 ('roughly 48')", TEN, gen=99)
add("population_min", "population", "min_mean_population_gens_1_8", 37.2, 3.72,
    "Fig. 2 (dip hidden under the GS line at gens 3-4; read by eye and by an independent pixel check: 37.3/37.4)", TEN)
add("population_min_gen", "population", "argmin_mean_population_gens_1_8", 3.5, 2, "Fig. 2", "+/- 2 generations [GUESS]")

# ---- survival (ER p.4)
add("survival", "survival", "survival_rate", 0.665, 0.0665, "ER p.4", TEN)

# ---- traits (ER p.5 printed values at gen 100; Fig. 3 for reach and the trajectory)
add("speed_final", "traits", "mean_trait_at_gen:speed", 26.6, 2.66, "ER p.5", TEN, gen=99)
add("size_final", "traits", "mean_trait_at_gen:size", 6.9, 0.69, "ER p.5", TEN, gen=99)
add("sense_final", "traits", "mean_trait_at_gen:sense", 31.2, 3.12, "ER p.5", TEN, gen=99)
v = chart("fig3_et_traits", "reach", 99)
add("reach_final", "traits", "mean_trait_at_gen:reach", v, round(0.1 * v, 4), "Fig. 3 (digitized)", TEN, gen=99)
for trait in ("speed", "size", "sense", "reach"):
    for g in [10, 20, 40, 60, 80]:
        if g not in bad("fig3_et_traits", trait):
            v = chart("fig3_et_traits", trait, g)
            add(f"{trait}_g{g}", "traits", f"mean_trait_at_gen:{trait}", v, round(0.1 * v, 4), "Fig. 3 (digitized)", TEN, gen=g)

# ---- diversity (Fig. 11 table, Figs. 4-5)
add("shannon_final", "diversity", "shannon_final", 3.77, 0.06, "Fig. 11 table, ER p.6", "paper SD across seeds")
for g in [1, 2, 3, 5, 10]:
    v = chart("fig5_et_shannon", "et", g)
    add(f"shannon_g{g}", "diversity", "mean_shannon_at_gen", v, round(0.1 * v, 4), "Fig. 5 (digitized)", TEN, gen=g)
add("cv2_final", "diversity", "cv2_final", 0.41, 0.29, "Fig. 11 table, ER p.6", "paper SD across seeds")
for g in [10, 20, 40, 60, 80]:
    v = chart("fig4_et_cv2", "et", g)
    add(f"cv2_g{g}", "diversity", "mean_cv2_at_gen", v, round(0.1 * v, 4), "Fig. 4 (digitized)", TEN, gen=g)

spec = {
    "version": 1,
    "written_before_any_calibration_run": True,
    "plan_rule": "Calibrate only settings designated as free parameters, on ET training-phase numbers only, seeds 1000-1029. "
                 "Every GS number and every transfer-phase number is held out.",
    "seeds": [1000, 1029],
    "phase": "ET, Environment A, generations 0-99 (no transfer run during calibration)",
    "primary_definitions": {
        "fitness": "M1: (food items + prey eaten) / blobs that started the generation",
        "final_fitness": "M3 primary: generation 99 only",
        "final_median": "median per-blob food count (items + prey), generation 99",
        "survival_rate": "M4 primary: mean over generations 0-99 of survivors / population",
        "shannon": "M10 primary: exact trait vectors (speed, size, sense, reach), natural log",
        "shannon_final": "M10 primary at generation 99",
        "cv2": "M9 primary: sum over speed, size, sense, reach of (sample SD / mean)^2",
        "cv2_final": "M9 primary at generation 99",
        "curve points": "mean over seeds of the per-run value at that generation; a run with no blobs is left out "
                        "for fitness/traits/diversity and counted as 0 for population",
        "peak/min": "computed on the mean-over-seeds curve",
    },
    "score": {
        "error": "e = (simulated mean over seeds - paper value) / tolerance",
        "group_score": "mean of e^2 over the targets in the group",
        "total_score": "mean of the group scores (groups: fitness, population, survival, traits, diversity; equal weight)",
        "pass": "a target passes if |e| <= 1; the plan's replication rule is that every target passes",
        "implementation": "src/evosim/calibration.py: score()",
    },
    "search": {
        "stage_1a": {
            "trait_sd (speed, size, sense together)": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            "reach_sd": [0.1, 0.25, 0.5, "same as trait_sd"],
            "tick_cap": [0, 50, 75, 100, 150, 200],
            "cannibalism": [False, True],
            "note": "0 = no time limit (S1). 288 combinations x 30 seeds.",
        },
        "stage_1b": "Around the best stage-1a combination, with reach_sd and cannibalism fixed: trait_sd in best "
                    "+ {-0.125, -0.0625, 0, +0.0625, +0.125} x tick_cap in best + {-25, -12, 0, +12, +25} "
                    "(tick_cap only refined if the best is not 0). trait_sd values below 0.25 and tick_cap values below 25 "
                    "are dropped. 25 combinations x 30 seeds.",
        "selection": "lowest total_score over all stage-1a and stage-1b combinations; exact ties go to the first in log order",
        "stage_2_metric_definitions": "With the selected simulation settings frozen, for each metric with candidate "
                                      "definitions (M3 window, M4 window, M9 variant, M10 variant, M11 energy efficiency), "
                                      "pick the candidate whose mean over seeds is closest to the paper value (in tolerance units). "
                                      "All candidates' values are reported. Energy efficiency target: ET 0.20, tolerance 0.18 "
                                      "(paper SD), candidates E1-E5 (docs/SPEC_part1.md M11).",
        "if_no_combination_passes": "record that no combination meets the target; do not change the model within this calibration",
    },
    "log": "calibration/log.csv: one row per combination; rows are never deleted",
    "targets": targets,
}
with open(os.path.join(ROOT, "calibration", "targets.json"), "w") as f:
    json.dump(spec, f, indent=1)
print(f"wrote calibration/targets.json with {len(targets)} targets")
for grp in ("fitness", "population", "survival", "traits", "diversity"):
    print(f"  {grp}: {sum(t['group'] == grp for t in targets)}")
