"""Calibration: run one setting on the calibration seeds, compare with the
paper's ET training numbers, score. Rules are in calibration/targets.json."""
import numpy as np

from . import config, metrics, runner

GROUPS = ("fitness", "population", "survival", "traits", "diversity")


def combo_overrides(trait_sd, reach_sd, tick_cap, cannibalism):
    rsd = trait_sd if reach_sd == "same" else float(reach_sd)
    return {"world": {"tick_cap": int(tick_cap), "cannibalism": bool(cannibalism)},
            "mutation_sd": {"speed": float(trait_sd), "size": float(trait_sd), "sense": float(trait_sd), "reach": rsd},
            "phases": {"train_generations": 100, "transfer_generations": 0}}


def combo_id(trait_sd, reach_sd, tick_cap, cannibalism):
    r = "same" if reach_sd == "same" else f"{float(reach_sd):g}"
    return f"sd{float(trait_sd):g}_reach{r}_cap{int(tick_cap)}_cann{int(bool(cannibalism))}"


def run_seed(args):
    """One ET training run (100 generations in A). Returns the per-generation
    series and the per-run scalars needed for targets and metric definitions."""
    base_path, overrides, seed = args
    cfg = config.load(base_path, overrides)
    ss = np.random.SeedSequence(seed, spawn_key=(0,))  # same stream as ET in run_part1.py
    log = runner.run_lineage(runner.et_start_traits(cfg), int(cfg["et"]["n_start"]), cfg["mutation_sd"], cfg, ss)
    a = log.arrays()
    s = metrics.series(a)
    keep = ["population", "fitness", "median", "survival", "mean_speed", "mean_size", "mean_sense", "mean_reach",
            "H_exact4", "cv2_4_samp", "ticks", "creature_steps"]
    return {"seed": seed, "series": {k: s[k] for k in keep}, "scalars": metrics.run_scalars(s, 100, 0)}


def simulated_value(target, runs):
    """The simulated counterpart of one target: a mean over seeds."""
    m = target["metric"]

    def curve(key, zero_if_extinct=False):
        stack = np.stack([r["series"][key] for r in runs])
        if zero_if_extinct:
            stack = np.nan_to_num(stack, nan=0.0)
        with np.errstate(invalid="ignore"):
            return np.nanmean(stack, axis=0)

    if m == "mean_fitness_at_gen":
        return float(curve("fitness")[target["gen"]])
    if m == "final_fitness":
        return float(curve("fitness")[99])
    if m in ("max_mean_fitness_gens_1_10", "argmax_mean_fitness_gens_1_10"):
        c = curve("fitness")[1:11]
        return float(np.nanmax(c)) if m.startswith("max") else float(1 + int(np.nanargmax(c)))
    if m == "final_median":
        return float(curve("median")[99])
    if m in ("mean_population_at_gen", "population_at_gen"):
        return float(curve("population", True)[target["gen"]])
    if m in ("min_mean_population_gens_1_8", "argmin_mean_population_gens_1_8"):
        c = curve("population", True)[1:9]
        return float(np.min(c)) if m.startswith("min") else float(1 + int(np.argmin(c)))
    if m == "survival_rate":
        return float(np.nanmean([np.nanmean(r["series"]["survival"][:100]) for r in runs]))
    if m.startswith("mean_trait_at_gen:"):
        return float(curve("mean_" + m.split(":")[1])[target["gen"]])
    if m == "shannon_final":
        return float(curve("H_exact4")[99])
    if m == "mean_shannon_at_gen":
        return float(curve("H_exact4")[target["gen"]])
    if m == "cv2_final":
        return float(curve("cv2_4_samp")[99])
    if m == "mean_cv2_at_gen":
        return float(curve("cv2_4_samp")[target["gen"]])
    raise KeyError(m)


def score(targets, runs):
    """Total score = mean over groups of mean over targets of ((sim - paper) / tolerance)^2."""
    rows = []
    for t in targets:
        v = simulated_value(t, runs)
        e = (v - t["paper_value"]) / t["tolerance"] if np.isfinite(v) else np.inf
        rows.append({"id": t["id"], "group": t["group"], "paper": t["paper_value"], "sim": v,
                     "tolerance": t["tolerance"], "error_in_tol": e, "pass": bool(abs(e) <= 1)})
    groups = {g: float(np.mean([r["error_in_tol"] ** 2 for r in rows if r["group"] == g])) for g in GROUPS}
    return {"total": float(np.mean(list(groups.values()))), "groups": groups, "targets": rows,
            "n_pass": int(sum(r["pass"] for r in rows)), "n_targets": len(rows)}
