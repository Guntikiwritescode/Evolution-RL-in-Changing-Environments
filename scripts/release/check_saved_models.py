"""Check saved policy coverage and tuning selection without evaluating policies."""
from pathlib import Path
import json, math

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/part3'
METHODS = ['oe', 'oe_fixed', 'ppo_fixed', 'ppo_dr', 'es_dr', 'es_fixed']

def read(path):
    return json.loads(path.read_text())

def main():
    policies = snapshots = trials = 0
    for index, method in enumerate(METHODS):
        stage = 'addendum' if method == 'es_fixed' else 'final'
        tuning = DATA / ('addendum/tuning' if method == 'es_fixed' else 'tuning') / f'tuning_{method}.json'
        record = read(tuning)
        rows = record['trials']
        assert record['method'] == method and len(rows) == 8
        assert record['budget_agent_steps'] == 50_000_000
        assert record['n_envs'] == 24 and record['episodes_per_env'] == 8
        assert {r['trial'] for r in rows} == set(range(8))
        for row in rows:
            assert row['seed'] == 4000 + 100 * index + row['trial']
            assert len(row['envs']) == 24
            score = sum(e['competence'] - e['random_competence'] for e in row['envs']) / 24
            assert math.isclose(score, row['score'], rel_tol=0, abs_tol=1e-12)
            assert row['train']['reached_budget']
            trials += 1
        winner = min(rows, key=lambda r: (-r['score'], r['trial']))
        assert all(record['chosen'][k] == winner[k] for k in ['trial', 'seed', 'settings', 'score'])
        for seed in range(5000, 5010):
            folder = DATA / stage / f'{method}_seed{seed}'
            run = read(folder / 'run.json')
            assert run['method'] == method and run['seed'] == seed
            settings = run['methods_config'][method]['settings']
            assert all(settings[k] == v for k, v in winner['settings'].items())
            assert (folder / 'policy.pkl').stat().st_size > 0
            policies += 1
            for checkpoint in range(1, 11):
                assert (folder / 'snapshots' / f'{checkpoint}.pkl').stat().st_size > 0
                snapshots += 1
    print(f'PASS: {trials} tuning scores, six selections, {policies} final policies and {snapshots} snapshots')

if __name__ == '__main__':
    main()
