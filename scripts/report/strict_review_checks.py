"""Retrospective descriptive stress checks, never replacements for registered inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]; sources={}
def read(p):
 f=R/p;sources[p]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
def iqm(x):
 a=np.sort(np.asarray(x).ravel());k=len(a)//4;return float(a[k:len(a)-k].mean())
c=read('results/part3/final/analysis.json');a=read('results/part3/addendum/analysis.json')
methods=['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr','es_fixed']; raw={};norm={};ext={};windows={}
exports=pd.read_csv(R/'results/publication/all_evaluation_points.csv');checks=0
for m in methods:
 base='addendum' if m=='es_fixed' else 'final'
 ev=[read(f'results/part3/{base}/{m}_seed{s}/eval_test.json') for s in range(5000,5010)]
 raw[m]=np.array([[[p['competence'] for p in e['adaptation']['curve']] for e in x['envs']] for x in ev])
 ext[m]=np.array([[e['zero_shot']['persistence']['extinct'] for e in x['envs']] for x in ev],float)
 assert raw[m].shape==(10,30,11)
 norm[m]=(raw[m]-np.asarray(c['random'])[None,:,None])/(np.asarray(c['best'])-c['random'])[None,:,None]
 v=exports[exports.method==m].sort_values(['seed','environment_index','checkpoint'])
 assert np.allclose(raw[m].ravel(),v.competence,rtol=0,atol=1e-12)
 assert np.allclose(norm[m].ravel(),v.normalized,rtol=0,atol=1e-12)
 checks+=2*raw[m].size
 curves=[read(f'results/part3/{base}/{m}_seed{s}/curves.json')['points'] for s in range(5000,5010)]
 scores=np.array([[p['score'] for p in row] for row in curves]);steps=np.array([[p['agent_steps'] for p in row] for row in curves])
 assert scores.shape==(10,10)
 windows[m]={'n_runs':10,'recorded_points_per_run':10,'first_step_range':[int(steps[:,0].min()),int(steps[:,0].max())],'last_step_range':[int(steps[:,-1].min()),int(steps[:,-1].max())],'initialization_observation_present':bool(np.any(steps==0)),'first_score_mean':float(scores[:,0].mean()),'last_score_mean':float(scores[:,-1].mean()),'mean_change':float((scores[:,-1]-scores[:,0]).mean()),'individual_run_changes':(scores[:,-1]-scores[:,0]).tolist(),'positive_run_changes':int(np.sum(scores[:,-1]>scores[:,0]))}
rows=pd.read_csv(R/'results/publication/registered_comparisons.csv').to_dict('records')
rows.insert(0,{'name':'H1','x':'oe','y':'ppo_dr','measure':'primary','iqm_difference':c['h1']['D']})
# A1 and A2 share an estimate; preserve both labels in output.
rows.append({'name':'A1','x':'es_dr','y':'es_fixed','measure':'primary','iqm_difference':a['a1']['D']})
out=[]
for row in rows:
 name,x,y,metric=[row[k] for k in ['name','x','y','measure']]
 sl=slice(0,10) if name in ['S6','A3'] else slice(10,30)
 def get(method,normalized):
  z=(norm if normalized else raw)[method][:,sl,:]
  if metric=='primary':return np.trapezoid(z,dx=.1,axis=2)
  if metric=='zero_shot':return z[:,:,0]
  if metric=='final':return z[:,:,-1]
  if metric=='extinct':return ext[method][:,sl]
  raise ValueError(metric)
 def effect(z):return iqm(z.mean(1)) if metric=='extinct' else iqm(z)
 u,v=get(x,True),get(y,True);ur,vr=get(x,False),get(y,False)
 delta=effect(u)-effect(v);assert np.isclose(delta,row['iqm_difference'],rtol=0,atol=1e-12),(name,delta,row['iqm_difference'])
 gaps=ur.mean(0)-vr.mean(0)
 deletion=[effect(np.delete(u,j,axis=1))-effect(np.delete(v,j,axis=1)) for j in range(u.shape[1])]
 run_deletions=[effect(np.delete(u,i,axis=0))-effect(v) for i in range(10)]+[effect(u)-effect(np.delete(v,i,axis=0)) for i in range(10)]
 out.append({'id':name,'x':x,'y':y,'measure':metric,'set':'T1' if name in ['S6','A3'] else 'T2/T3','registered_iqm_difference':delta,'mean_difference':float(u.mean()-v.mean()),'median_difference':float(np.median(u.mean(1))-np.median(v.mean(1))) if metric=='extinct' else float(np.median(u)-np.median(v)),'raw_per_environment_mean_differences':gaps.tolist(),'raw_environment_positive':int((gaps>0).sum()),'raw_environment_negative':int((gaps<0).sum()),'raw_environment_ties':int((gaps==0).sum()),'environment_deletion_range':[min(deletion),max(deletion)],'single_run_deletion_range':[min(run_deletions),max(run_deletions)]})
d={'status':'post_hoc_descriptive_only','raw_normalized_export_values_checked':checks,'registered_effects_reproduced':len(out),'contrasts':out,'training_observation_windows':windows,'limitations':['No new p-values, decisions, or generalization claims.','Deletion ranges are dependent perturbations, not confidence intervals.','Raw scores across environments have different scales.','A1/A2 share a contrast and are not independent evidence.','S7 registered IQM uses per-run extinction fractions; its raw per-environment signs are different descriptive summaries.','Flat checkpoint means do not establish absence of learning before the first checkpoint, after the horizon, or in individual runs.'],'source_sha256':sources}
(R/'results/audit/strict_review_checks.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
for v in out:print(v['id'],'effect',round(v['registered_iqm_difference'],5),'env signs + / -',v['raw_environment_positive'],v['raw_environment_negative'],'env deletion range',tuple(round(x,5) for x in v['environment_deletion_range']))
for m,v in windows.items():print(m,'first/last/change',*[round(v[k],4) for k in ['first_score_mean','last_score_mean','mean_change']],'positive runs',v['positive_run_changes'])
print('Verified values:',checks)
