"""The Part 3 final analysis, as plain functions (PREREG_part3.md Sections 7-8).

scripts/part3/analyze.py reads the final runs, calls these, and writes the results, figures and
report. Every function here is checked against hand-worked examples in tests/test_part3_analysis.py.

Shapes: a method's data is a dict of arrays with a leading axis of runs and a second axis of the
30 test environments (T1 0-9, T2 10-19, T3 20-29, the order of evaluate.py), e.g.
  curve (runs, 30, 11)   competence at adaptation points 0..10
  random (30,)           the random policy's competence (the same episodes for every run)
  extinct (runs, 30)     persistence test ended extinct (0/1)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from evo2 import metrics

SETS = {"T1": range(0, 10), "T2": range(10, 20), "T3": range(20, 30)}
X_POINTS = np.linspace(0.0, 1.0, 11)          # fraction of the adaptation budget at points 0..10
MARGIN = 0.05                                  # PREREG 8.1
REPS = 50_000                                  # PREREG 8.1
STATS_SEED = 4800                              # PREREG Section 6
ALPHA = 0.05                                   # PREREG 8.2


def env_index(*sets: str) -> list[int]:
    return [j for s in sets for j in SETS[s]]


def best_per_env(curves: dict) -> np.ndarray:
    """The highest competence in each environment over every core method, final run and
    checkpoint (zero-shot point 0 and adaptation points 1-10) (PREREG 7.2)."""
    return np.max(np.stack([np.max(c, axis=(0, 2)) for c in curves.values()]), axis=0)


def valid_envs(best: np.ndarray, random: np.ndarray) -> np.ndarray:
    """Environments kept in every normalized measure: best > random (PREREG 7.2)."""
    return best > random


def normalized_curves(curve: np.ndarray, random: np.ndarray, best: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(competence - random) / (best - random) per point; NaN in left-out environments."""
    out = np.full(curve.shape, np.nan)
    v = np.where(valid)[0]
    out[:, v, :] = (curve[:, v, :] - random[v][None, :, None]) / (best[v] - random[v])[None, :, None]
    return out


def auc(norm: np.ndarray) -> np.ndarray:
    """Primary metric per (run, environment): trapezoid area over x = 0..1 (PREREG 7.4)."""
    out = np.full(norm.shape[:2], np.nan)
    for r in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if not np.isnan(norm[r, j, 0]):
                out[r, j] = metrics.curve_auc(X_POINTS, norm[r, j])
    return out


def select(scores: np.ndarray, envs: list[int]) -> np.ndarray:
    """(runs, envs) for the listed environments, dropping left-out (NaN) environments."""
    sub = scores[:, envs]
    keep = ~np.all(np.isnan(sub), axis=0)
    return sub[:, keep]


def holm(pvalues: list[float], alpha: float = ALPHA) -> list[dict]:
    """Holm's step-down procedure. Returns, in the input order, the adjusted p-value
    (monotone, capped at 1) and whether the test rejects at alpha."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return [{"p": float(p[i]), "p_holm": float(adj[i]), "reject": bool(adj[i] <= alpha)} for i in range(m)]


def mwu(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U on per-run scores, with P(X > Y) = U_x / (n_x n_y)
    (ties count one half), which always agrees with the test (PREREG 8.2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    p = float(res.pvalue)
    # SciPy may return p=1 (rather than NaN) when every observation is equal.
    # Detect the data condition directly; do not classify arbitrary NaNs as ties.
    all_tied = bool(x.size and y.size and np.all(x == x.flat[0]) and np.all(y == x.flat[0]))
    return {"U": float(res.statistic), "p": 1.0 if all_tied else p, "all_tied": all_tied,
            "p_x_greater": float(res.statistic / (len(x) * len(y))), "n_x": len(x), "n_y": len(y)}


def per_run_mean(scores: np.ndarray) -> np.ndarray:
    return np.nanmean(scores, axis=1)


def h1(primary: dict, reps: int = REPS, seed: int = STATS_SEED) -> dict:
    """D = IQM(OE) - IQM(PPO-DR) with its whole-run bootstrap CI and the pre-set verdict (PREREG 8.1)."""
    d, lo, hi = metrics.iqm_difference_ci(primary["oe"], primary["ppo_dr"], reps, seed)
    pi = metrics.prob_improvement(per_run_mean(primary["oe"])[:, None], per_run_mean(primary["ppo_dr"])[:, None])
    return {"D": d, "ci": [lo, hi], "verdict": metrics.h1_verdict(lo, hi, MARGIN), "margin": MARGIN,
            "p_oe_greater": pi, "reps": reps, "seed": seed}


# (name, X, Y, measure, environment sets, predicted side of P(X > Y): ">" , "<" or None)
SECONDARY = [
    ("S1", "oe", "es_dr", "primary", ("T2", "T3"), ">"),
    ("S2", "oe", "oe_fixed", "primary", ("T2", "T3"), ">"),
    ("S3", "ppo_dr", "ppo_fixed", "primary", ("T2", "T3"), ">"),
    ("S4", "oe", "ppo_dr", "zero_shot", ("T2", "T3"), None),
    ("S5", "oe", "ppo_dr", "final", ("T2", "T3"), ">"),
    ("S6", "oe", "ppo_dr", "primary", ("T1",), None),
    ("S7", "oe", "ppo_dr", "extinct", ("T2", "T3"), "<"),
]


def secondary(measures: dict, reps: int = REPS, seed: int = STATS_SEED) -> list[dict]:
    """The Holm family S1-S7 (PREREG 8.2). measures[name][method] has shape (runs, 30)."""
    rows = []
    for name, x, y, measure, sets, side in SECONDARY:
        envs = env_index(*sets)
        sx, sy = select(measures[measure][x], envs), select(measures[measure][y], envs)
        t = mwu(per_run_mean(sx), per_run_mean(sy))
        if measure == "extinct":
            # S7's measure is the per-run fraction of extinct tests (PREREG 8.2), so its IQM is taken
            # over those fractions (one value per run); an IQM of 0/1 values would not be a rate
            sx, sy = per_run_mean(sx)[:, None], per_run_mean(sy)[:, None]
        d, lo, hi = metrics.iqm_difference_ci(sx, sy, reps, seed)
        rows.append({"name": name, "x": x, "y": y, "measure": measure, "sets": list(sets), "predicted": side,
                     **t, "iqm_difference": d, "ci": [lo, hi]})
    for row, h in zip(rows, holm([r["p"] for r in rows])):
        row.update(p_holm=h["p_holm"], reject=h["reject"])
        if row["predicted"] is None:
            row["result"] = ("difference (X higher)" if row["p_x_greater"] > 0.5 else "difference (Y higher)") \
                if h["reject"] else "no difference found"
        else:
            on_side = row["p_x_greater"] > 0.5 if row["predicted"] == ">" else row["p_x_greater"] < 0.5
            row["result"] = ("supported" if on_side else "evidence against") if h["reject"] else "not supported"
    return rows


def h2b(primary: dict, reps: int = REPS, seed: int = STATS_SEED) -> dict:
    """Variety vs method on the primary metric's IQMs, 2 x 2 (PREREG 8.3), with CIs from one
    whole-run bootstrap over all four methods' runs."""
    names = ("oe", "oe_fixed", "ppo_dr", "ppo_fixed")
    arrays = {m: np.asarray(primary[m], float) for m in names}

    def effects(q):
        v = 0.5 * ((q["oe"] - q["oe_fixed"]) + (q["ppo_dr"] - q["ppo_fixed"]))
        m = 0.5 * ((q["oe"] - q["ppo_dr"]) + (q["oe_fixed"] - q["ppo_fixed"]))
        i = (q["oe"] - q["oe_fixed"]) - (q["ppo_dr"] - q["ppo_fixed"])
        return np.array([v, m, i, abs(v) - abs(m)])

    point = effects({m: metrics.iqm(a) for m, a in arrays.items()})
    rng = np.random.default_rng(seed)
    boots = np.empty((reps, 4))
    for b in range(reps):
        boots[b] = effects({m: metrics.iqm(metrics._resample_runs(arrays[m], rng)) for m in names})
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    out = {k: {"value": float(point[n]), "ci": [float(lo[n]), float(hi[n])]}
           for n, k in enumerate(("variety", "method", "interaction", "delta"))}
    dlo, dhi = out["delta"]["ci"]
    out["verdict"] = "variety matters more" if dlo > 0 else "method matters more" if dhi < 0 else "not decided"
    ilo, ihi = out["interaction"]["ci"]
    out["interaction_excludes_0"] = bool(ilo > 0 or ihi < 0)
    return out


def time_to_90(curve: np.ndarray, pre_shift: np.ndarray, envs: list[int]) -> dict:
    """PREREG 7.5 (descriptive): per (run, environment), the first point (fraction of the
    adaptation budget) with competence >= 0.9 x the run's pre-shift competence (its mean
    zero-shot competence over T1); "not reached" if never."""
    times = []
    for r in range(curve.shape[0]):
        for j in envs:
            times.append(metrics.time_to_fraction(X_POINTS, curve[r, j], 0.9 * pre_shift[r]))
    t = np.asarray(times, float)
    reached = ~np.isnan(t)
    per_env = reached.reshape(curve.shape[0], len(envs)).sum(axis=0)
    return {"pairs": int(t.size), "reached": int(reached.sum()), "share_reached": float(reached.mean()),
            "median_time_if_reached": float(np.median(t[reached])) if reached.any() else None,
            "runs_reaching_by_env": {int(j): int(n) for j, n in zip(envs, per_env) if n}}


def mean_pairwise_distance(vectors: np.ndarray) -> float:
    """Mean Euclidean distance over all pairs of rows."""
    v = np.asarray(vectors, float)
    n = v.shape[0]
    d = [np.linalg.norm(v[i] - v[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d)) if d else 0.0


def behavioral_distance(actions: np.ndarray) -> float:
    """actions (agents, probes, action_dim): mean over agent pairs of the mean over probes of the
    Euclidean distance between their actions (PREREG 7.5)."""
    a = np.asarray(actions, float)
    n = a.shape[0]
    d = [np.mean(np.linalg.norm(a[i] - a[j], axis=-1)) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d)) if d else 0.0


# ---------- Addendum A (PREREG_part3.md Section 16, 4 Nov 2026): ES-fixed ----------
# Decided after the core results were known; ES-fixed's own result was unknown when these rules
# were written. The core functions above are unchanged; nothing here changes a core number.

# (name, X, Y, measure, environment sets, predicted side of P(X > Y)); one Holm family
ADDENDUM_FAMILY = [
    ("A2", "es_dr", "es_fixed", "primary", ("T2", "T3"), ">"),
    ("A3", "es_dr", "es_fixed", "zero_shot", ("T1",), ">"),
]


def compare_family(measures: dict, family: list, reps: int = REPS, seed: int = STATS_SEED) -> list[dict]:
    """A Holm family of pairwise comparisons with the same tests and result labels as the core
    secondary family (secondary above): two-sided Mann-Whitney U on per-run means, P(X > Y), the IQM
    difference with its whole-run bootstrap CI. measures[name][method] has shape (runs, 30)."""
    rows = []
    for name, x, y, measure, sets, side in family:
        envs = env_index(*sets)
        sx, sy = select(measures[measure][x], envs), select(measures[measure][y], envs)
        t = mwu(per_run_mean(sx), per_run_mean(sy))
        d, lo, hi = metrics.iqm_difference_ci(sx, sy, reps, seed)
        rows.append({"name": name, "x": x, "y": y, "measure": measure, "sets": list(sets), "predicted": side,
                     **t, "iqm_difference": d, "ci": [lo, hi]})
    for row, h in zip(rows, holm([r["p"] for r in rows])):
        row.update(p_holm=h["p_holm"], reject=h["reject"])
        if row["predicted"] is None:
            row["result"] = ("difference (X higher)" if row["p_x_greater"] > 0.5 else "difference (Y higher)") \
                if h["reject"] else "no difference found"
        else:
            on_side = row["p_x_greater"] > 0.5 if row["predicted"] == ">" else row["p_x_greater"] < 0.5
            row["result"] = ("supported" if on_side else "evidence against") if h["reject"] else "not supported"
    return rows


def a1(primary: dict, reps: int = REPS, seed: int = STATS_SEED) -> dict:
    """A1: D = IQM(ES-DR) - IQM(ES-fixed) on the primary metric (T2, T3), its whole-run bootstrap
    CI and the H1 verdict rule with the same margin (PREREG 16.4)."""
    d, lo, hi = metrics.iqm_difference_ci(primary["es_dr"], primary["es_fixed"], reps, seed)
    pi = metrics.prob_improvement(per_run_mean(primary["es_dr"])[:, None], per_run_mean(primary["es_fixed"])[:, None])
    return {"D": d, "ci": [lo, hi], "verdict": metrics.h1_verdict(lo, hi, MARGIN), "margin": MARGIN,
            "p_es_dr_greater": pi, "reps": reps, "seed": seed}


def interaction_verdict(low: float, high: float) -> str:
    """A4's pre-set reading (PREREG 16.4)."""
    if low > 0:
        return "variety effect larger for ES than for OE"
    if high < 0:
        return "variety effect larger for OE than for ES"
    return "no difference found"


def a4(primary: dict, reps: int = REPS, seed: int = STATS_SEED) -> dict:
    """A4: I = (IQM(ES-DR) - IQM(ES-fixed)) - (IQM(OE) - IQM(OE-fixed)) on the primary metric
    (T2, T3), with a CI from one whole-run bootstrap over all four methods' runs (as h2b)."""
    names = ("es_dr", "es_fixed", "oe", "oe_fixed")
    arrays = {m: np.asarray(primary[m], float) for m in names}

    def effects(q):
        es, oe = q["es_dr"] - q["es_fixed"], q["oe"] - q["oe_fixed"]
        return np.array([es, oe, es - oe])

    point = effects({m: metrics.iqm(a) for m, a in arrays.items()})
    rng = np.random.default_rng(seed)
    boots = np.empty((reps, 3))
    for b in range(reps):
        boots[b] = effects({m: metrics.iqm(metrics._resample_runs(arrays[m], rng)) for m in names})
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    out = {k: {"value": float(point[n]), "ci": [float(lo[n]), float(hi[n])]}
           for n, k in enumerate(("variety_es", "variety_oe", "interaction"))}
    out["verdict"] = interaction_verdict(*out["interaction"]["ci"])
    return out


ADDENDUM_LEFT_OUT_MAX = 2      # PREREG 16.3: more than 2 of 10 ES-fixed seeds left out, no verdicts
RERUN_TOLERANCE = 0.01         # PREREG 16.3: a check rerun this far outside the core ES-DR range flags A1, A2


def addendum_readings(a1_verdict: str, a3_result: str) -> dict:
    """The pre-set readings of PREREG 16.4: A3 answers Milano et al.'s claim, A1 our adaptation question.
    The returned text uses the interpretation rules recorded in PREREG 16.4."""
    milano = {"supported": ["in our world, score-based ES evolved in varied environments did better on new environments "
                            "from the training ranges, before any adaptation, as Milano et al. found",
                            "set against S2 and the exploratory OE result, their direction holds for score-based ES and "
                            "not for survival-based OE, and the kind of evolution (with the other ES/OE differences in "
                            "16.1) is the likely reason",
                            "which part of it is not tested"],
              "evidence against": ["ES-fixed did better, so our world did not reward variety for score-based evolution "
                                   "either"],
              "not supported": ["no difference found",
                                "the reversal for OE cannot be put down to the kind of evolution"]}
    non_support = "With H2-PPO's null (S3), a non-support result means our world may not reward variety in adaptation for any method"
    small = ("a real but small effect, under 5% of the random-to-best range")
    adapt = {"support": ["varied training made ES adapt better"],
             "support (below the margin)": [small, 'reported as "a small advantage, below the pre-set margin", not as a '
                                            "meaningful difference, and it does not by itself support Milano et al.'s "
                                            "direction"],
             "against": ["ES-fixed adapted better", non_support],
             "against (below the margin)": ["ES-fixed adapted better (the same distinction applies)", small, non_support],
             "no meaningful difference": ['"No meaningful difference" or "inconclusive": as worded', non_support],
             "inconclusive": ['"No meaningful difference" or "inconclusive": as worded', non_support]}
    if a3_result not in milano or a1_verdict not in adapt:
        raise ValueError(f"no pre-set reading for A1 {a1_verdict!r}, A3 {a3_result!r}")
    return {"milano_claim": "; ".join(milano[a3_result]), "adaptation": "; ".join(adapt[a1_verdict]),
            "fragments": {"milano_claim": milano[a3_result], "adaptation": adapt[a1_verdict]}}


def rerun_flag(rerun_means: list, core_min: float, core_max: float, tol: float = RERUN_TOLERANCE) -> bool:
    """True if any ES-DR check rerun lies more than tol outside the core ES-DR runs' range (PREREG 16.3)."""
    return any(m < core_min - tol or m > core_max + tol for m in rerun_means)
