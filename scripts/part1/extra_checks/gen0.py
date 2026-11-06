"""Extra reference check: generations 0-1 over 4,000 seeds, blob-eating off and on (docs/SPEC_part1.md 15.1)."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from multiprocessing import Pool
import compare_reference as C
seeds = range(10000, 14000)   # validation-only seeds
names = C.STATS
with Pool(2) as pool:
    for cann in (False, True):
        jobs = [(s, 2, 0.5, 0.0, cann) for s in seeds]
        R = np.stack(pool.map(C.rust_run, jobs)); P = np.stack(pool.map(C.python_run, jobs))
        np.savez('results/part1/reference_check/gen0_cannibalism_%s.npz' % cann, R=R, P=P)
        print('cannibalism', cann)
        for g in (0, 1):
            for j in range(7):
                r, p = R[:, g, j], P[:, g, j]
                se = np.sqrt(r.var(ddof=1)/r.size + p.var(ddof=1)/p.size)
                z = (p.mean()-r.mean())/se if se > 0 else 0
                print(f"  gen{g} {names[j]:15s} rust {r.mean():9.3f}  py {p.mean():9.3f}  z {z:+.2f}")
