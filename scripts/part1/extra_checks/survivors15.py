"""Extra reference check: generations 0-14 over 2,000 new seeds with blob-eating off
(our base settings), to test the small survivor gap seen around generation 10 in
isolate.py (docs/SPEC_part1.md 15.1)."""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compare_reference as C  # noqa: E402

seeds = range(14000, 16000)  # validation-only seeds
with Pool(2) as pool:
    jobs = [(s, 15, 0.5, 0.25, False) for s in seeds]
    R = np.stack(pool.map(C.rust_run, jobs))
    P = np.stack(pool.map(C.python_run, jobs))
np.savez_compressed("results/part1/reference_check/survivors15_raw.npz", R=R, P=P)
for j in range(len(C.STATS)):
    r, p = R[:, :, j], P[:, :, j]
    se = np.sqrt(np.nanvar(r, 0, ddof=1) / r.shape[0] + np.nanvar(p, 0, ddof=1) / p.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (np.nanmean(p, 0) - np.nanmean(r, 0)) / np.where(se > 0, se, np.nan)
    zz = np.round(z, 1)
    print(f"{C.STATS[j]:15s} max|z| {np.nanmax(np.abs(z)) if np.any(np.isfinite(z)) else 0:5.2f}  z by gen: {zz.tolist()}")
