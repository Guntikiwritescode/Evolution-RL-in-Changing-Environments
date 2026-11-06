"""Independent reporting cross-checks and four-method bootstrap from raw evaluations.
No project analysis imports, training or new policy evaluation.
"""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2]
def read(p):return json.loads((R/p).read_text())
c=read('results/part3/final/analysis.json');a=read('results/part3/addendum/analysis.json');audit=read('results/audit/raw_audit.json');checks=[]
def check(label,x,y,tol=1e-10):
 err=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));checks.append(dict(check=label,max_error=err,tolerance=tol,passed=err<=tol))
for method,val in audit['iqm'].items():check(method+' IQM',val,(a if method=='es_fixed' else c)['iqm_primary'][method][0])
records={'H1':c['h1'],'A1':a['a1'],**{x['name']:x for x in c['secondary']+a['family']}}
for name,x in audit['comparisons'].items():
 saved=records[name];check(name+' difference',x['difference'],saved.get('D',saved.get('iqm_difference')));check(name+' interval',x['ci_registered'],saved['ci'])
 if 'p_holm' in saved:
  for k,v in [('mwu_U','U'),('mwu_p','p'),('p_holm','p_holm')]:check(name+' '+k,x[k],saved[v])
arrays={}
for m in audit['iqm']:
 base='addendum' if m=='es_fixed' else 'final'
 raw=np.array([[[q['competence'] for q in e['adaptation']['curve']] for e in read(f'results/part3/{base}/{m}_seed{s}/eval_test.json')['envs']] for s in range(5000,5010)])
 norm=(raw-np.array(c['random'])[None,:,None])/(np.array(c['best'])-c['random'])[None,:,None]
 arrays[m]=np.trapezoid(norm,x=np.linspace(0,1,11),axis=2)[:,10:]
def iqm(x):
 x=np.sort(x,axis=-1);n=x.shape[-1];return x[...,n//4:n-n//4].mean(-1)
for name,names,fields in [('H2b',['oe','oe_fixed','ppo_dr','ppo_fixed'],['variety','method','interaction','delta']),('A4',['es_dr','es_fixed','oe','oe_fixed'],['variety_es','variety_oe','interaction'])]:
 ix=np.random.default_rng(4800).integers(0,10,size=(50000,4,10));q=[]
 for j,m in enumerate(names):
  chunks=[iqm(arrays[m][chunk[:,j]].reshape(len(chunk),-1)) for chunk in np.array_split(ix,50)]
  q.append(np.r_[iqm(arrays[m].ravel()),np.concatenate(chunks)])
 q=np.array(q)
 if name=='H2b':
  v=(q[0]-q[1]+q[2]-q[3])/2;me=(q[0]-q[2]+q[1]-q[3])/2;out=[v,me,q[0]-q[1]-q[2]+q[3],abs(v)-abs(me)];saved=c['h2b']
 else:out=[q[0]-q[1],q[2]-q[3],q[0]-q[1]-q[2]+q[3]];saved=a['a4']
 for f,v in zip(fields,out):
  check(name+' '+f+' estimate',v[0],saved[f]['value']);check(name+' '+f+' interval',np.percentile(v[1:],[2.5,97.5]),saved[f]['ci'])
fig=read('results/publication/fig03_variation.json')
check('Figure 3 A3',fig['A3']['iqm_difference'],a['family'][1]['iqm_difference'])
out={'scope':__doc__,'checks':checks,'all_passed':all(x['passed'] for x in checks),'n_checks':len(checks),'bootstrap':{'reps':50000,'seed':4800,'unit':'whole training run; fixed environments and normalization'}}
(R/'results/audit/prewrite_completion_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'n_checks':len(checks),'all_passed':out['all_passed'],'failures':[x for x in checks if not x['passed']]},indent=2))
assert out['all_passed']
