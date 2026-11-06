"""Validate exported scientific values against all 60 retained evaluation records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/audit/viability/reanalysis/statistics'
OUT.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(ROOT/'results/publication/all_evaluation_points.csv')
ix=df.set_index(['method','seed','environment_index','checkpoint'])
analysis=json.loads((ROOT/'results/part3/final/analysis.json').read_text())
anchors=np.array(analysis['best']),np.array(analysis['random'])
methods=['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr','es_fixed']
training={}; n=0; maxerr=0.; env_reference=None; records=[]
curves_by_method={m:[] for m in methods}
for method in methods:
    sub='addendum' if method=='es_fixed' else 'final'
    vals=[]; steps=[]; configs=[]
    for seed in range(5000,5010):
        base=ROOT/f'results/part3/{sub}/{method}_seed{seed}'
        raw=json.loads((base/'eval_test.json').read_text())
        run=json.loads((base/'run.json').read_text())
        assert run['method']==method and run['seed']==seed
        assert run['budget_agent_steps']==200_000_000
        configs.append(run['methods_config'])
        assert len(raw['envs'])==30
        envs=[x['env'] for x in raw['envs']]
        if env_reference is None: env_reference=envs
        assert envs==env_reference
        rawcurve=[]
        for env_index,e in enumerate(raw['envs']):
            expected_set=['T1','T2','T3'][env_index//10]
            assert e['set']==expected_set and e['index']==env_index%10
            assert e['zero_shot']['random_competence']==anchors[1][env_index]
            curve=e['adaptation']['curve']; assert len(curve)==11
            assert curve[0]['competence']==e['zero_shot']['competence']
            rawcurve.append([p['competence'] for p in curve])
            for checkpoint,point in enumerate(curve):
                row=ix.loc[(method,seed,env_index,checkpoint)]
                assert row.competence==point['competence'] and row.agent_steps==point['agent_steps']
                assert row.extinct==e['zero_shot']['persistence']['extinct']
                assert row.steps_survived==e['zero_shot']['persistence']['steps_survived']
                normalized=(point['competence']-anchors[1][env_index])/(anchors[0][env_index]-anchors[1][env_index])
                err=abs(row.normalized-normalized); maxerr=max(maxerr,err); assert err<1e-12
                n+=1
        curves_by_method[method].append(rawcurve)
        t=json.loads((base/'curves.json').read_text())
        assert len(t['points'])==10
        for point in t['points']:
            assert len(point['envs'])==24
            recompute=np.mean([e['competence']-e['random_competence'] for e in point['envs']])
            assert abs(recompute-point['score'])<1e-12
        vals.append([p['score'] for p in t['points']])
        steps.append([p['agent_steps'] for p in t['points']])
        records.append(str((base/'eval_test.json').relative_to(ROOT)))
    assert all(c==configs[0] for c in configs)
    v=np.array(vals); s=np.array(steps)
    training[method]={'first_mean':float(v[:,0].mean()),'last_mean':float(v[:,-1].mean()),
       'first_last_per_run_change':(v[:,-1]-v[:,0]).tolist(),'n_first_last_increases':int((v[:,-1]>v[:,0]).sum()),
       'first_step_minmax':[int(s[:,0].min()),int(s[:,0].max())],'last_step_minmax':[int(s[:,-1].min()),int(s[:,-1].max())],
       'initialization_observed':False}
actualbest=np.stack([np.array(curves_by_method[m]).max(axis=(0,2)) for m in methods if m!='es_fixed']).max(axis=0)
assert np.array_equal(actualbest,anchors[0])
report={'evaluation_records':60,'exported_rows_compared':n,'maximum_normalization_error':float(maxerr),
        'all_core_anchors_match_raw_maxima':True,'training':training,'sources':records}
(OUT/'raw_record_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
