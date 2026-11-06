"""Part 2 statistics and verdict rules (docs/PREREG_part2.md, Sections 6-9, 18).

Everything that decides a verdict is here, so the made-up-data checks
(Section 18.3) test exactly the code the real analysis uses.

Conventions
- A comparison is "first group minus second group".
- P1 values are per-run totals over B generations 40-49 (integers); P1 = total / 10.
- P2 values are per-run booleans (extinct at the end of B).
- Threshold, ceiling and floor checks use exact fractions (Section 6, exact arithmetic).
"""
from fractions import Fraction

import numpy as np
from scipy import stats as sps

ALPHA = 0.05
STATS_SEED = 2999
TOL = 1e-9

# stream purposes inside the stats seed
PERM, BOOT, PAIRED_PERM, PAIRED_BOOT = 0, 1, 2, 3


def stats_rng(arm_id, comp_id, outcome_id, purpose, seed=STATS_SEED):
    """Every permutation and bootstrap has its own stream from seed 2999,
    split by arm, comparison and outcome (Section 6). The made-up-data checks
    pass seed 2950 (Section 18.3), so they never touch the real streams."""
    return np.random.default_rng(np.random.SeedSequence(
        int(seed), spawn_key=(int(arm_id), int(comp_id), int(outcome_id), int(purpose))))


# ------------------------------------------------------------------ tests
def perm_test_mean(x, y, rng, n_perm=10000):
    """Two-sided permutation test on the difference in means.
    p = (number with |permuted diff| >= |observed diff| - 1e-12, + 1) / (n_perm + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx = len(x)
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 1.0
    z = np.concatenate([x, y])
    obs = abs(x.mean() - y.mean())
    count = 0
    chunk = 2000
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        P = rng.permuted(np.tile(z, (b, 1)), axis=1)
        d = P[:, :nx].mean(axis=1) - P[:, nx:].mean(axis=1)
        count += int(np.sum(np.abs(d) >= obs - 1e-12))
        done += b
    return (count + 1) / (n_perm + 1)


def fisher_p(x_count, nx, y_count, ny):
    if x_count == y_count and nx == ny and x_count in (0, nx):
        return 1.0
    return float(sps.fisher_exact([[x_count, nx - x_count], [y_count, ny - y_count]])[1])


def holm(pvals):
    """Holm step-down adjusted p-values (same order as given)."""
    p = np.asarray(pvals, float)
    m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(o):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------ intervals
def welch_se(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))


def boot_t_ci(x, y, rng, levels=(0.95,), n_boot=20000):
    """Studentized bootstrap CIs for mean(x) - mean(y), resampling within
    groups, Welch standard error. If both groups are constant, every CI is the
    point estimate. A resample whose standard error is 0 uses the original
    standard error instead (this only happens when both groups are almost
    constant). Returns {level: (lo, hi)}."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = float(x.mean() - y.mean())
    se = welch_se(x, y)
    if se == 0.0:
        return {lv: (d, d) for lv in levels}
    nx, ny = len(x), len(y)
    ts = np.empty(n_boot)
    chunk = 2000
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        bx = x[rng.integers(0, nx, (b, nx))]
        by = y[rng.integers(0, ny, (b, ny))]
        ds = bx.mean(axis=1) - by.mean(axis=1)
        ses = np.sqrt(bx.var(axis=1, ddof=1) / nx + by.var(axis=1, ddof=1) / ny)
        ses = np.where(ses > 0, ses, se)
        ts[done:done + b] = (ds - d) / ses
        done += b
    out = {}
    for lv in levels:
        a = (1.0 - lv) / 2.0
        qlo, qhi = np.quantile(ts, [a, 1.0 - a])
        out[lv] = (d - qhi * se, d - qlo * se)
    return out


def wilson(x, n, z):
    p = x / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(x1, n1, x2, n2, level):
    """Newcombe hybrid score CI for p1 - p2 (method 10, Newcombe 1998)."""
    z = sps.norm.ppf(1.0 - (1.0 - level) / 2.0)
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    d = p1 - p2
    return (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def eq_level(k):
    """Equivalence CI level for a family of k tests: 1 - 2 x 0.05 / k."""
    return 1.0 - 2.0 * ALPHA / k


# ------------------------------------------------------------------ one comparison
def compare(x_p1, y_p1, x_p2, y_p2, ks, arm_id, comp_id, thr_p1=Fraction(5, 2),
            thr_p2=Fraction(1, 10), n_perm=10000, n_boot=20000, seed=STATS_SEED):
    """Raw results of one comparison on both outcomes (before Holm).
    x_p1, y_p1: per-run P1 totals (ints). x_p2, y_p2: per-run extinct flags.
    ks: family sizes whose equivalence CIs are needed (e.g. (14, 16) for arm 2:
    its own family and the fixed-level holds/changes check)."""
    ks = tuple(int(k) for k in (ks if np.iterable(ks) else (ks,)))
    x1, y1 = np.asarray(x_p1, np.int64), np.asarray(y_p1, np.int64)
    nx, ny = len(x1), len(y1)
    out = {}
    # P1 (in P1 units: totals / 10)
    xs, ys = x1 / 10.0, y1 / 10.0
    p = perm_test_mean(xs, ys, stats_rng(arm_id, comp_id, 1, PERM, seed), n_perm)
    levels = (0.95,) + tuple(eq_level(k) for k in ks)
    cis = boot_t_ci(xs, ys, stats_rng(arm_id, comp_id, 1, BOOT, seed), levels, n_boot)
    d_exact = Fraction(int(x1.sum()), 10 * nx) - Fraction(int(y1.sum()), 10 * ny)
    out["P1"] = {"diff": float(d_exact), "diff_exact": d_exact, "p": p,
                 "ci95": cis[0.95], "ci_eq": {k: cis[eq_level(k)] for k in ks}, "thr": thr_p1,
                 "mean_x": Fraction(int(x1.sum()), 10 * nx), "mean_y": Fraction(int(y1.sum()), 10 * ny),
                 "mwu_p": _mwu(xs, ys), "cliffs_delta": cliffs_delta(xs, ys)}
    # P2
    xc, yc = int(np.sum(x_p2)), int(np.sum(y_p2))
    mx, my = len(x_p2), len(y_p2)
    d2 = Fraction(xc, mx) - Fraction(yc, my)
    out["P2"] = {"diff": float(d2), "diff_exact": d2, "p": fisher_p(xc, mx, yc, my),
                 "ci95": newcombe(xc, mx, yc, my, 0.95),
                 "ci_eq": {k: newcombe(xc, mx, yc, my, eq_level(k)) for k in ks},
                 "thr": thr_p2, "count_x": xc, "count_y": yc, "n_x": mx, "n_y": my}
    return out


def _mwu(x, y):
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 1.0
    return float(sps.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


# ------------------------------------------------------------------ verdicts
MEANINGFUL, EQUIV, EQUIV_SMALL, UNCLEAR, INCONCL = (
    "meaningfully different", "equivalent", "equivalent, with a small real difference",
    "different, size unclear", "inconclusive")


def outcome_verdict(r, p_adj, k, thr=None):
    """Section 6, checked in order: meaningfully different; equivalent;
    different, size unclear; inconclusive. k selects the equivalence CI;
    thr overrides the threshold (threshold sensitivity, Section 18.5)."""
    thr_f = r["thr"] if thr is None else Fraction(thr)
    sig = p_adj < ALPHA
    reaches = abs(r["diff_exact"]) >= thr_f
    thr = float(thr_f)
    lo, hi = r["ci_eq"][k]
    inside = (lo >= -thr - TOL) and (hi <= thr + TOL)
    if sig and reaches:
        return MEANINGFUL
    if inside:
        return EQUIV_SMALL if sig else EQUIV
    if sig:
        return UNCLEAR
    return INCONCL


def p1_label(r, cap, thr=None):
    """Label on an 'equivalent' P1 verdict: ceiling, capped on one side, floor.
    cap = None: no hard cap (arm 2, blob-eating variant), so no ceiling labels.
    The cut-offs move with the threshold (Section 18.5)."""
    thr = r["thr"] if thr is None else Fraction(thr)
    mx, my = r["mean_x"], r["mean_y"]
    hi, lo = max(mx, my), min(mx, my)
    if cap is not None:
        c = Fraction(cap) - thr
        if lo >= c:
            return "at the ceiling"
        if hi >= c:
            return "capped on one side"
    if hi <= thr:
        return "at the floor"
    return None


def first_better(outcome, r):
    """+1 if the difference favors the first group, -1 the second, 0 none.
    Better = higher P1 or lower P2."""
    s = np.sign(float(r["diff_exact"]))
    return int(s) if outcome == "P1" else int(-s)


def overall_verdict(res, verdicts):  # noqa: C901
    """Section 6 overall verdict from the two outcome verdicts."""
    def better(o, side):  # meaningfully better for `side` (+1 first, -1 second)
        return verdicts[o] == MEANINGFUL and first_better(o, res[o]) == side

    def sig_worse(o, side):
        return verdicts[o] in (MEANINGFUL, UNCLEAR) and first_better(o, res[o]) == -side

    for side, name in ((1, "first group better"), (-1, "second group better")):
        for o, other in (("P1", "P2"), ("P2", "P1")):
            if better(o, side) and sig_worse(other, side):
                return "mixed"
    for side, name in ((1, "first group better"), (-1, "second group better")):
        if any(better(o, side) for o in ("P1", "P2")):
            others_ok = all(not sig_worse(o, side) for o in ("P1", "P2"))
            if others_ok:
                return name
    if verdicts["P1"] in (EQUIV, EQUIV_SMALL) and verdicts["P2"] in (EQUIV, EQUIV_SMALL):
        return "the same"
    return "inconclusive"


def decide_family(raw, k, caps, fixed_level=False, thr=None):
    """Verdicts for a family of comparisons.
    raw: {comp_id: compare(...) output}; caps: {comp_id: cap or None}.
    k: the family size (16 or 14), which sets Holm and the equivalence CI.
    fixed_level=True: the holds/changes rule (Section 9.1): p < 0.05 / k per
    test instead of Holm (call with k = 16 for both arms).
    thr: optional {"P1": ..., "P2": ...} threshold override (Section 18.5)."""
    keys = [(c, o) for c in sorted(raw) for o in ("P1", "P2")]
    ps = [raw[c][o]["p"] for c, o in keys]
    if fixed_level:
        adj = [min(1.0, p * k) for p in ps]  # p < 0.05/k  <=>  k*p < 0.05
    else:
        adj = holm(ps)
    out = {}
    for (c, o), pa in zip(keys, adj):
        out.setdefault(c, {"p_adj": {}, "verdict": {}})
        out[c]["p_adj"][o] = float(pa)
        out[c]["verdict"][o] = outcome_verdict(raw[c][o], pa, k, None if thr is None else thr[o])
    for c in out:
        v = out[c]["verdict"]
        lab = (p1_label(raw[c]["P1"], caps.get(c), None if thr is None else thr["P1"])
               if v["P1"] in (EQUIV, EQUIV_SMALL) else None)
        out[c]["p1_label"] = lab
        ov = overall_verdict(raw[c], v)
        out[c]["overall"] = ov
        out[c]["overall_label"] = (f"the same ({lab})" if ov == "the same" and lab else ov)
    return out


def a_phase_label(s5_x, s5_y):
    """'includes A-phase losses' if, for any run seed, one group reached B and
    the other did not (seed by seed; groups with the same run seed are paired)."""
    return bool(np.any(np.asarray(s5_x, bool) != np.asarray(s5_y, bool)))


def arose_in_A(overall, verdicts, raw, reached_both_diffs, n_both, min_seeds=30):
    """Section 6: for a labeled verdict of first/second better or mixed, decide
    'arose in A' (True) or 'part of it before the move' (False). Returns None
    when the rule doesn't apply.
    reached_both_diffs: {'P1': Fraction or float, 'P2': ...} over seeds where both reached B."""
    if overall not in ("first group better", "second group better", "mixed"):
        return None
    if n_both < min_seeds:
        return True
    meaningful = [o for o in ("P1", "P2") if verdicts[o] == MEANINGFUL]
    still = []
    for o in meaningful:
        d = reached_both_diffs[o]
        same_dir = np.sign(float(d)) == np.sign(float(raw[o]["diff_exact"]))
        still.append(same_dir and abs(Fraction(d)) >= raw[o]["thr"])
    return not any(still)


def positive_control(decided, s5_all, b0_mean):
    """Section 6: at least one planned comparison where both groups reached B
    in every run, both have mean B-generation-0 population >= 40, and the
    overall verdict is first better, second better or mixed.
    s5_all: {comp_id: (bool, bool)}; b0_mean: {comp_id: (float, float)}."""
    for c, d in decided.items():
        if all(s5_all[c]) and min(b0_mean[c]) >= 40 and d["overall"] in (
                "first group better", "second group better", "mixed"):
            return True
    return False


def u2_fires(b_minus_a):
    """U2: the B winner is ahead by less than 2.5 blobs in at least half the runs.
    b_minus_a: exact differences (Fractions) per run."""
    below = sum(1 for v in b_minus_a if Fraction(v) < Fraction(5, 2))
    return below * 2 >= len(b_minus_a)


def u3_fires(stage2_scores, deployed_means):
    """U3: averaged over runs, the deployed mean population (A gens 10-99) is
    more than 2.5 blobs below the winner's stage-2 score. Exact Fractions."""
    d = [Fraction(a) - Fraction(b) for a, b in zip(stage2_scores, deployed_means)]
    return sum(d, Fraction(0)) / len(d) > Fraction(5, 2)


# ------------------------------------------------------------------ secondary
def paired_signflip(dx, rng, n_perm=10000):
    """Sign-flip permutation test on per-seed differences (two-sided)."""
    d = np.asarray(dx, float)
    if np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return (int(np.sum(np.abs((s * d).mean(axis=1)) >= obs - 1e-12)) + 1) / (n_perm + 1)


def mcnemar_exact(x_flags, y_flags):
    x, y = np.asarray(x_flags, bool), np.asarray(y_flags, bool)
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    if b + c == 0:
        return 1.0
    return float(sps.binomtest(b, b + c, 0.5).pvalue)


def boot_mean_ci(v, rng, level=0.95, n_boot=10000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(m, [a, 1 - a])
    return float(lo), float(hi)


# ------------------------------------------------------------------ power (V6, Section 18.2)
def fisher_power(n, p1, p2, k=16, thr=Fraction(1, 10), require_estimate=True):
    """Exact chance of a significant Fisher test at 0.05/k (and, if asked, an
    estimate that reaches thr), for n runs per group and true rates p1, p2."""
    alpha = ALPHA / k
    A = np.arange(n + 1)
    pa = sps.binom.pmf(A, n, p1)
    pb = sps.binom.pmf(A, n, p2)
    tot = 0.0
    for a in range(n + 1):
        if pa[a] < 1e-15:
            continue
        for b in range(n + 1):
            if pb[b] < 1e-15:
                continue
            if require_estimate and abs(Fraction(b - a, n)) < thr:
                continue
            if fisher_p(a, n, b, n) < alpha:
                tot += pa[a] * pb[b]
    return tot
