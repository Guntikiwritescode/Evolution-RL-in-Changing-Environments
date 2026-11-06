"""Metrics computed from raw run logs (docs/SPEC_part1.md Section 8).

Where the paper's definition is unknown, every candidate definition is computed
and named; calibration picks among candidates only on ET training numbers.
Generations 0..99 are training (Environment A); 100..149 are transfer (B).
"""
import numpy as np

from .kernel import ENERGY_COST_SCALE

TRAITS4 = ("speed", "size", "sense", "reach")
TRAITS3 = ("speed", "size", "sense")


# ---------------------------------------------------------------- per-generation series
def blob_cost(speed, size, sense):
    """Energy cost per tick, S1 creature/mod.rs:214-218 (vectorized)."""
    eff = speed * size / 10.0
    return ENERGY_COST_SCALE * (size ** 3 * eff ** 2 + sense)


def _groups(a):
    """Indices of blobs per generation, as a list over all generations."""
    n_gen = len(a["g_gen"])
    order = np.argsort(a["b_gen"], kind="stable")
    bounds = np.searchsorted(a["b_gen"][order], np.arange(n_gen + 1))
    return [order[bounds[g]:bounds[g + 1]] for g in range(n_gen)]


def shannon(rows):
    """Natural-log Shannon index over identical rows (types)."""
    if rows.shape[0] == 0:
        return np.nan
    _, counts = np.unique(rows, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def series(a):
    """Per-generation series for one run. Returns dict of arrays (length = generations)."""
    pop = a["g_population"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fit_all = np.where(pop > 0, (a["g_food_eaten"] + a["g_prey_eaten"]) / pop, np.nan)
        fit_balls = np.where(pop > 0, a["g_food_eaten"] / pop, np.nan)
        surv = np.where(pop > 0, a["g_survivors"] / pop, np.nan)
    groups = _groups(a)
    n = len(groups)
    out = {"population": pop, "fitness": fit_all, "fitness_balls_only": fit_balls,
           "survival": surv, "ticks": a["g_ticks"].astype(float),
           "creature_steps": a["g_creature_steps"].astype(float)}
    keys = ["median", "median_balls_only"]
    keys += [f"mean_{t}" for t in TRAITS4] + [f"sd_{t}" for t in TRAITS4]
    keys += [f"cv2_{t}" for t in TRAITS4]
    keys += ["cv2_3_pop", "cv2_3_samp", "cv2_4_pop", "cv2_4_samp"]
    keys += ["H_exact4", "H_round1_4", "H_round0_4", "H_exact3", "H_round1_3", "H_round0_3"]
    keys += ["mean_cost", "mean_cost_per_dist", "cv_cost", "cv_cost_per_dist"]
    for k in keys:
        out[k] = np.full(n, np.nan)
    food_all = a["b_food_balls"] + a["b_prey"]
    for g, idx in enumerate(groups):
        if idx.size == 0:
            continue
        out["median"][g] = np.median(food_all[idx])
        out["median_balls_only"][g] = np.median(a["b_food_balls"][idx])
        tr = {t: a["b_" + t][idx] for t in TRAITS4}
        for t in TRAITS4:
            m = tr[t].mean()
            out[f"mean_{t}"][g] = m
            out[f"sd_{t}"][g] = tr[t].std()
            out[f"cv2_{t}"][g] = (tr[t].std() / m) ** 2 if m > 0 else np.nan
        for ts, tag in ((TRAITS3, "3"), (TRAITS4, "4")):
            for ddof, dt in ((0, "pop"), (1, "samp")):
                if idx.size > ddof:
                    out[f"cv2_{tag}_{dt}"][g] = sum((tr[t].std(ddof=ddof) / tr[t].mean()) ** 2 for t in ts)
        m4 = np.column_stack([tr[t] for t in TRAITS4])
        m3 = m4[:, :3]
        out["H_exact4"][g] = shannon(m4)
        out["H_round1_4"][g] = shannon(np.round(m4, 1))
        out["H_round0_4"][g] = shannon(np.round(m4, 0))
        out["H_exact3"][g] = shannon(m3)
        out["H_round1_3"][g] = shannon(np.round(m3, 1))
        out["H_round0_3"][g] = shannon(np.round(m3, 0))
        c = blob_cost(tr["speed"], tr["size"], tr["sense"])
        u = c / (tr["speed"] * tr["size"] / 10.0)
        out["mean_cost"][g] = c.mean()
        out["mean_cost_per_dist"][g] = u.mean()
        out["cv_cost"][g] = c.std() / c.mean()
        out["cv_cost_per_dist"][g] = u.std() / u.mean()
    return out


# ---------------------------------------------------------------- per-run scalars
def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan


def _slope(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def run_scalars(s, n_train=100, n_transfer=50):
    """Every candidate scalar metric for one run, from its per-generation series s."""
    T = n_train
    last = T - 1
    w10 = slice(max(0, T - 10), T)
    out = {}
    for k in ("fitness", "fitness_balls_only", "median", "median_balls_only"):
        out[f"final_{k}_last1"] = float(s[k][last])
        out[f"final_{k}_last10"] = _nanmean(s[k][w10])
    out["population_final"] = float(s["population"][last])
    out["survival_all100"] = _nanmean(s["survival"][:T])
    out["survival_last10"] = _nanmean(s["survival"][w10])
    out["survival_last1"] = float(s["survival"][last])
    for t in TRAITS4:
        out[f"trait_{t}_final"] = float(s[f"mean_{t}"][last])
    for k in ("cv2_3_pop", "cv2_3_samp", "cv2_4_pop", "cv2_4_samp",
              "cv2_speed", "cv2_size", "cv2_sense", "cv2_reach"):
        out[f"{k}_last1"] = float(s[k][last])
        out[f"{k}_trainmean"] = _nanmean(s[k][:T])
    for k in ("H_exact4", "H_round1_4", "H_round0_4", "H_exact3", "H_round1_3", "H_round0_3"):
        out[f"{k}_last1"] = float(s[k][last])
        out[f"{k}_trainmean"] = _nanmean(s[k][:T])
        out[f"{k}_gen10"] = float(s[k][10]) if s[k].size > 10 else np.nan
    # Energy efficiency candidates (all exactly 0 for clones that never change)
    c0, cT = s["mean_cost"][0], s["mean_cost"][last]
    u0, uT = s["mean_cost_per_dist"][0], s["mean_cost_per_dist"][last]
    out["eff_E1_cost_drop"] = float((c0 - cT) / c0)
    out["eff_E2_cost_per_dist_drop"] = float((u0 - uT) / u0)
    out["eff_E3_cost_per_dist_drop_trainmean"] = _nanmean((u0 - s["mean_cost_per_dist"][:T]) / u0)
    out["eff_E4_cv_cost_last1"] = float(s["cv_cost"][last])
    out["eff_E5_cv_cost_per_dist_last1"] = float(s["cv_cost_per_dist"][last])
    # Compute
    out["creature_steps_train"] = float(np.nansum(s["creature_steps"][:T]))
    out["creature_steps_all"] = float(np.nansum(s["creature_steps"]))
    out["ticks_train"] = float(np.nansum(s["ticks"][:T]))
    out["ticks_all"] = float(np.nansum(s["ticks"]))
    if s["population"].size > T and n_transfer > 0:
        B = slice(T, T + n_transfer)
        fb = s["fitness"][B]
        out["transfer_fit_first10"] = _nanmean(fb[:10])
        out["transfer_fit_first5"] = _nanmean(fb[:5])
        out["transfer_fit_last10"] = _nanmean(fb[-10:])
        out["transfer_n_last10_with_blobs"] = int(np.sum(~np.isnan(fb[-10:])))
        out["gap_last1_first10"] = out["final_fitness_last1"] - out["transfer_fit_first10"]
        out["gap_last1_first5"] = out["final_fitness_last1"] - out["transfer_fit_first5"]
        out["gap_last10_first10"] = out["final_fitness_last10"] - out["transfer_fit_first10"]
        out["gap_last10_first5"] = out["final_fitness_last10"] - out["transfer_fit_first5"]
        out["transfer_slope"] = _slope(fb)
        pb = s["population"][B]
        out["transfer_pop_last10"] = float(np.mean(pb[-10:]))
        out["transfer_pop_first_drop"] = float(pb[1]) if pb.size > 1 else np.nan
        out["extinct_end"] = bool(pb[-1] == 0)
        out["transfer_fit_gen0"] = float(fb[0])
    return out
