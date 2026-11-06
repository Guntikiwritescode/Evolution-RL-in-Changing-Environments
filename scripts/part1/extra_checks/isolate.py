"""Extra reference check: first 15 generations, 600 new seeds, three settings (see docs/SPEC_part1.md 15.1)."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from multiprocessing import Pool
import compare_reference as C
cases = [("off_reach0.25", 0.5, 0.25, False), ("off_reach0", 0.5, 0.0, False), ("on_reach0.25", 0.5, 0.25, True)]
seeds = range(9200, 9800)   # validation seeds, separate from the 200-seed check
with Pool(2) as pool:
    for name, tsd, rsd, cann in cases:
        jobs = [(s, 15, tsd, rsd, cann) for s in seeds]
        R = np.stack(pool.map(C.rust_run, jobs)); P = np.stack(pool.map(C.python_run, jobs))
        for j, st in [(7, "mean_speed"), (8, "mean_size"), (0, "population"), (3, "survivors")]:
            r, p = R[:, :, j], P[:, :, j]
            se = np.sqrt(np.nanvar(r, 0, ddof=1)/len(seeds) + np.nanvar(p, 0, ddof=1)/len(seeds))
            z = (np.nanmean(p, 0) - np.nanmean(r, 0)) / np.where(se > 0, se, np.nan)
            print(f"{name:14s} {st:12s} max|z| {np.nanmax(np.abs(z)):.2f}  z@5,10,14: {np.round(z[[5,10,14]],1).tolist()}  diff@10 {np.nanmean(p[:,10])-np.nanmean(r[:,10]):+.3f}")
