"""Post hoc descriptive checks of the registered primary comparison.

Uses all ten runs, fixed T2/T3 environments and the original fixed anchors.
Reports all specified summaries, including baseline-adjusted AUC, without
replacing the registered estimand. Run from any directory.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sources = {}
def read(rel):
    path = ROOT / rel
    sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def iqm(x):
    a = np.sort(np.asarray(x).ravel())
    k = len(a) // 4
    return float(a[k:len(a)-k].mean())

c = read('results/part3/final/analysis.json')
curves = {}
for method in ['oe', 'ppo_dr']:
    curves[method] = np.array([
        [[p['competence'] for p in e['adaptation']['curve']]
         for e in read(f'results/part3/final/{method}_seed{s}/eval_test.json')['envs']]
        for s in range(5000, 5010)
    ])[:, 10:]
    assert curves[method].shape == (10, 20, 11)
random = np.array(c['random'])[10:]
denom = np.array(c['best'])[10:] - random
assert np.all(denom > 0)
raw_auc = {m: ((a[:,:,1:]+a[:,:,:-1])*.05).sum(2) for m,a in curves.items()}
normalized = {m: (a-random[None,:,None])/denom[None,:,None] for m,a in curves.items()}
auc = {m: ((a[:,:,1:]+a[:,:,:-1])*.05).sum(2) for m,a in normalized.items()}
for m in auc:
    assert np.isclose(iqm(auc[m]), c['iqm_primary'][m][0], atol=1e-12, rtol=0)
primary = iqm(auc['oe']) - iqm(auc['ppo_dr'])
deletions = []
for method in auc:
    for i in range(10):
        x = {m: np.delete(a,i,axis=0) if m == method else a for m,a in auc.items()}
        deletions.append({'method':method,'seed_removed':5000+i,
                          'iqm_difference':iqm(x['oe'])-iqm(x['ppo_dr'])})
environment_deletions = [iqm(np.delete(auc['oe'],j,axis=1))-iqm(np.delete(auc['ppo_dr'],j,axis=1)) for j in range(20)]
summary = {}
for m in auc:
    summary[m] = {
        'normalized_auc_iqm':iqm(auc[m]),
        'normalized_auc_mean':float(auc[m].mean()),
        'normalized_auc_median':float(np.median(auc[m])),
        'raw_auc_mean':float(raw_auc[m].mean()),
        'normalized_zero_shot_mean':float(normalized[m][:,:,0].mean()),
        'normalized_final_mean':float(normalized[m][:,:,-1].mean()),
        'normalized_auc_above_initial_mean':float((auc[m]-normalized[m][:,:,0]).mean()),
        'normalized_final_minus_initial_mean':float((normalized[m][:,:,-1]-normalized[m][:,:,0]).mean()),
    }
out = {
    'status':'post_hoc_descriptive_only', 'n_runs_per_method':10, 'fixed_test_environments':20,
    'registered_primary_iqm_difference':primary,
    'summaries':summary, 'single_run_deletions':deletions,
    'single_environment_deletion_differences':environment_deletions,
    'raw_mean_auc_difference_by_environment':(raw_auc['oe'].mean(0)-raw_auc['ppo_dr'].mean(0)).tolist(),
    'limitations':[
        'No new p-values or confirmatory decisions; no independence assumed for the deletion checks.',
        'Raw scores are not commensurate across environments; per-environment signs are descriptive.',
        'Baseline-subtracted summaries change the estimand and do not isolate adaptation ability or mechanism.',
        'Fixed normalization anchors; no claim to uncertainty from anchor estimation or new environments.',
        'Stability to one deletion does not establish adequate power, generality, or a valid causal control.'
    ], 'source_sha256':sources,
}
path = ROOT/'results/audit/reviewer_sensitivity.json'
path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','single_run_deletions','single_environment_deletion_differences','limitations']},indent=2))
print('Single-run deletion range:', min(x['iqm_difference'] for x in deletions),max(x['iqm_difference'] for x in deletions))
print('Single-environment deletion range:',min(environment_deletions),max(environment_deletions))
