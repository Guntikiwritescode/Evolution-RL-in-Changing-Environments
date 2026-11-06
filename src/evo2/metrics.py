"""Part 3 metrics (SPEC_part3.md, Section 7). Plain NumPy; statistics via rliable 1.2.0.

The exact choices (which points enter the curve, what "best observed" means,
bootstrap settings) are fixed in docs/PREREG_part3.md; these functions only
compute.
"""
from __future__ import annotations

import numpy as np
from rliable import metrics as rlm


def normalize(score, random_score, best_score):
    """(score - random) / (best - random). Raises if best <= random."""
    score, random_score, best_score = (np.asarray(x, float) for x in (score, random_score, best_score))
    span = best_score - random_score
    if np.any(span <= 0):
        raise ValueError("best observed score must be above the random policy's score")
    return (score - random_score) / span


def curve_auc(x, y):
    """Area under a curve by the trapezoid rule, divided by the x-range, so a
    flat curve at height h has area h. x must be strictly increasing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape[0] != y.shape[-1] or x.shape[0] < 2:
        raise ValueError("x and y must match and have at least 2 points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    widths = np.diff(x)
    heights = (y[..., 1:] + y[..., :-1]) / 2.0
    return np.sum(widths * heights, axis=-1) / (x[-1] - x[0])


def time_to_fraction(x, y, target):
    """First x at which y >= target; NaN if never (censored)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    hit = np.nonzero(y >= target)[0]
    return float(x[hit[0]]) if hit.size else float("nan")


def iqm(scores):
    """Interquartile mean over all entries (rliable's aggregate_iqm)."""
    return float(rlm.aggregate_iqm(np.asarray(scores, float)))


def _resample_runs(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Whole runs drawn with replacement; each run keeps all its environments
    (a run is the independent unit: one trained run is scored on every
    environment)."""
    idx = rng.integers(0, scores.shape[0], size=scores.shape[0])
    return scores[idx]


def iqm_with_ci(score_dict: dict, reps: int, seed: int) -> dict:
    """IQM and 95% percentile CI per method, by resampling whole runs.
    score_dict values have shape (runs, environments). Returns
    {method: (iqm, low, high)}. Reproducible: uses its own seeded generator."""
    out = {}
    for name, v in score_dict.items():
        v = np.asarray(v, float)
        rng = np.random.default_rng(seed)
        boots = np.array([rlm.aggregate_iqm(_resample_runs(v, rng)) for _ in range(reps)])
        out[name] = (float(rlm.aggregate_iqm(v)), float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5)))
    return out


def iqm_difference_ci(scores_x, scores_y, reps: int, seed: int):
    """D = IQM(x) - IQM(y) and its 95% percentile CI, resampling whole runs of
    each method independently. Inputs (runs, environments)."""
    x, y = np.asarray(scores_x, float), np.asarray(scores_y, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rlm.aggregate_iqm(_resample_runs(x, rng)) - rlm.aggregate_iqm(_resample_runs(y, rng))
                      for _ in range(reps)])
    d = float(rlm.aggregate_iqm(x) - rlm.aggregate_iqm(y))
    return d, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def performance_profile(scores, taus):
    """Fraction of (run, environment) scores above each threshold tau
    (rliable's run-score distribution, without its bootstrap)."""
    s = np.asarray(scores, float).ravel()
    return np.array([np.mean(s > t) for t in np.asarray(taus, float)])


def h1_verdict(low: float, high: float, margin: float) -> str:
    """The pre-registered H1 verdict (PREREG_part3.md Section 8.1), checked in
    this order: support, against, no meaningful difference, inconclusive."""
    if low > 0:
        return "support (below the margin)" if high < margin else "support"
    if high < 0:
        return "against (below the margin)" if low > -margin else "against"
    if -margin <= low and high <= margin:
        return "no meaningful difference"
    return "inconclusive"


def prob_improvement(scores_x, scores_y):
    """P(X > Y) averaged over tasks (rliable). Inputs (runs, tasks)."""
    return float(rlm.probability_of_improvement(np.asarray(scores_x, float), np.asarray(scores_y, float)))
