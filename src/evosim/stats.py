"""The paper's statistics (docs/SPEC_part1.md Section 9).

Welch's two-tailed t-test, Welch-Satterthwaite df, Cohen's d with the averaged
variance, Bonferroni over 10 comparisons. Both the uncorrected p (what the
paper's p column shows) and the corrected p (what its stars show) are reported.
"""
import math

import numpy as np
from scipy import stats as sps

N_COMPARISONS = 10  # [PAPER] ER p.3


def welch(x, y, n_comparisons=N_COMPARISONS, zero_by_design=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = x.size, y.size
    out = {"n_x": nx, "n_y": ny, "zero_by_design": bool(zero_by_design)}
    # Each group's own mean and SD are reported whenever that group has the values for them,
    # even if the other group does not (e.g. every GS run extinct).
    out.update(mean_x=float(x.mean()) if nx else np.nan, sd_x=float(x.std(ddof=1)) if nx >= 2 else np.nan,
               mean_y=float(y.mean()) if ny else np.nan, sd_y=float(y.std(ddof=1)) if ny >= 2 else np.nan)
    if nx < 2 or ny < 2:  # not enough runs with a value for a test
        out.update(t=np.nan, df=np.nan, p=np.nan, p_bonferroni=np.nan, d=np.nan, stars="n/a (n<2)")
        return out
    mx, my, sx, sy = out["mean_x"], out["mean_y"], out["sd_x"], out["sd_y"]
    vx, vy = sx ** 2 / nx, sy ** 2 / ny
    if vx + vy == 0:
        out.update(t=np.nan, df=np.nan, p=np.nan, p_bonferroni=np.nan, d=np.nan, stars="n/a")
        return out
    t = (mx - my) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / ((vx ** 2 / (nx - 1) if nx > 1 else 0) + (vy ** 2 / (ny - 1) if ny > 1 else 0))
    p = float(2 * sps.t.sf(abs(t), df))
    pb = min(1.0, p * n_comparisons)
    d = (mx - my) / math.sqrt((sx ** 2 + sy ** 2) / 2)
    stars = "***" if pb < 0.001 else "**" if pb < 0.01 else "*" if pb < 0.05 else "ns"
    out.update(t=t, df=df, p=p, p_bonferroni=pb, d=d, stars=stars)
    return out
