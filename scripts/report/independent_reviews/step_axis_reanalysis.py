"""Descriptive sensitivity using saved observations only; no new evaluations."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/audit/viability/reanalysis/design'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT/'results/publication/all_evaluation_points.csv').open()))
groups = {}
for r in rows:
    groups.setdefault((r['method'], int(r['seed']), int(r['environment_index'])), []).append(r)
nom, actual, over = {}, {}, {}
for k, g in groups.items():
    g.sort(key=lambda r: int(r['checkpoint']))
    assert [int(r['checkpoint']) for r in g] == list(range(11)), k
    xs = [int(r['agent_steps']) for r in g]
    ys = [float(r['normalized']) for r in g]
    assert xs[0] == 0 and all(b > a for a, b in zip(xs, xs[1:])), k
    assert all(math.isfinite(y) for y in ys), k
    if k[2] < 10:
        continue
    m = k[0]
    nom.setdefault(m, []).append(sum((a+b)/20 for a, b in zip(ys, ys[1:])))
    area, budget = 0., 20_000_000
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if a >= budget:
            break
        end = min(b, budget)
        yend = ya + (yb-ya)*(end-a)/(b-a)
        area += (end-a)*(ya+yend)/2
    actual.setdefault(m, []).append(area/budget)
    over.setdefault(m, []).append(xs[-1]-budget)

def iqm(values):
    values = sorted(values)
    cut = len(values)//4
    middle = values[cut:-cut]
    return sum(middle)/len(middle)

pairs = [('H1','oe','ppo_dr'), ('S2','oe','oe_fixed'), ('S3','ppo_dr','ppo_fixed'), ('A1','es_dr','es_fixed')]
summary = {
    'n_rows': len(rows), 'n_curves': len(groups),
    'nominal_iqm': {m: iqm(v) for m, v in nom.items()},
    'actual_steps_clipped_20m_iqm': {m: iqm(v) for m, v in actual.items()},
    'final_overshoot_min_mean_max': {m: [min(v), sum(v)/len(v), max(v)] for m,v in over.items()},
    'contrasts': {n: {'nominal': iqm(nom[x])-iqm(nom[y]),
                      'actual_steps_clipped_20m': iqm(actual[x])-iqm(actual[y])} for n,x,y in pairs}}
(OUT/'step_axis_reanalysis.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
