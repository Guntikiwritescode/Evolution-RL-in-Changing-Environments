"""Read-only reanalysis of provided outcome records; no simulator evaluations."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/audit/viability/reanalysis/design'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT / 'results/publication/all_evaluation_points.csv').open()))
by_run = {}
for row in rows:
    by_run.setdefault((row['method'], int(row['seed'])), []).append(row)
checked = 0
floor = {}
random_by_env = {}
raw_points = {}
for (method, seed), runrows in by_run.items():
    stage = 'addendum' if method == 'es_fixed' else 'final'
    path = ROOT / f'results/part3/{stage}/{method}_seed{seed}/eval_test.json'
    data = json.loads(path.read_text())
    assert len(data['envs']) == 30, path
    lookup = {(int(r['environment_index']), int(r['checkpoint'])): r for r in runrows}
    assert len(lookup) == 330, (method, seed)
    for j, env in enumerate(data['envs']):
        curve = env['adaptation']['curve']
        chunks = env['adaptation']['chunks']
        assert len(curve) == 11, (path, j)
        assert curve[0]['competence'] == env['zero_shot']['competence'], (path, j)
        assert sum(c['agent_steps'] for c in chunks) == curve[-1]['agent_steps'], (path, j)
        assert chunks[0]['agent_steps_total'] == chunks[0]['agent_steps'], (path, j)
        random = env['zero_shot']['random_competence']
        if j in random_by_env:
            assert random == random_by_env[j], (path, j, 'random baseline changed')
        random_by_env[j] = random
        for c, point in enumerate(curve):
            row = lookup[j, c]
            assert point['competence'] == float(row['competence']), (path, j, c)
            assert point['agent_steps'] == int(row['agent_steps']), (path, j, c)
            raw_points[method, seed, j, c] = point['competence']
            checked += 1
        if method in ('oe', 'oe_fixed'):
            k = (method, env['set'])
            v = floor.setdefault(k, {'natural_births': 0, 'floor_births': 0, 'all_dead_steps': 0, 'environments': 0})
            v['natural_births'] += sum(c['births'] for c in chunks)
            v['floor_births'] += sum(c['floor_births'] for c in chunks)
            v['all_dead_steps'] += sum(c['steps_all_dead_before_floor'] for c in chunks)
            v['environments'] += 1
            assert min(c['min_alive'] for c in chunks) >= 16, (path, j)
            assert sum(c['env_changes'] for c in chunks) == 0, (path, j)
best = {j: max(v for (m, s, jj, c), v in raw_points.items() if jj == j and m != 'es_fixed') for j in range(30)}
max_norm_error = 0.0
for row in rows:
    j = int(row['environment_index'])
    expected = (float(row['competence']) - random_by_env[j]) / (best[j] - random_by_env[j])
    max_norm_error = max(max_norm_error, abs(expected - float(row['normalized'])))
assert max_norm_error < 1e-12, max_norm_error
summary = {'runs': len(by_run), 'raw_points_checked': checked, 'max_normalization_error': max_norm_error,
           'all_raw_curves_and_chunk_totals_match': True,
           'oe_adaptation_floor': {f'{m}/{s}': {**v, 'floor_share': v['floor_births']/(v['floor_births']+v['natural_births'])} for (m,s),v in floor.items()}}
(OUT / 'raw_design_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
