"""Independent raw evaluation audit. Does not import project analysis code.
No training or evaluation of policies occurs. Registered bootstrap is reproduced;
paired-seed sensitivity is explicitly post hoc, not a replacement preregistration.
"""
import itertools, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'audit'
OUT.mkdir(exist_ok=True)
SEED=4800
REPS=50000

def iqm(a,axis=None):
    a=np.asarray(a)
    if axis is None:a=a.reshape(-1);axis=0
    a=np.sort(a,axis=axis)
    n=a.shape[axis];k=n//4
    return np.take(a,np.arange(k,n-k),axis=axis).mean(axis=axis)

def boot(x,y,paired=False):
    rng=np.random.default_rng(SEED)
    # Interleaved draws match the registered sequential RNG calls exactly.
    idx=rng.integers(0,len(x),size=(REPS,1 if paired else 2,len(x)))
    vals=[]
    for start in range(0,REPS,1000):
        ii=idx[start:start+1000]
        xx=x[ii[:,0]].reshape(len(ii),-1)
        yy=y[ii[:,0 if paired else 1]].reshape(len(ii),-1)
        vals.extend((iqm(xx,1)-iqm(yy,1)).tolist())
    return np.percentile(vals,[2.5,97.5]).tolist()

def holm(p):
    out=np.zeros(len(p));last=0
    for rank,j in enumerate(np.argsort(p)):
        last=max(last,min(1,(len(p)-rank)*p[j]));out[j]=last
    return out.tolist()

def signflip(x,y):
    d=x.mean(axis=1)-y.mean(axis=1)
    signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
    return float(np.mean(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))

core=json.loads((ROOT/'results/part3/final/analysis.json').read_text())
add=json.loads((ROOT/'results/part3/addendum/analysis.json').read_text())
curves={};randoms=[];extinct={};checks={};files={}
for method in ['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr','es_fixed']:
    base=ROOT/'results/part3'/('addendum' if method=='es_fixed' else 'final')
    paths=[base/f'{method}_seed{s}/eval_test.json' for s in range(5000,5010)]
    if not all(p.exists() for p in paths):
        checks[method]={'missing':[str(p.relative_to(ROOT)) for p in paths if not p.exists()]};continue
    rows=[json.loads(p.read_text()) for p in paths]
    c=np.array([[[p['competence'] for p in e['adaptation']['curve']] for e in r['envs']] for r in rows])
    assert c.shape==(10,30,11) and np.isfinite(c).all()
    assert all([(e['set'],e['index']) for e in r['envs']]==[(s,i) for s in ['T1','T2','T3'] for i in range(10)] for r in rows)
    steps=np.array([[[p['agent_steps'] for p in e['adaptation']['curve']] for e in r['envs']] for r in rows])
    assert (steps>=np.arange(11)[None,None,:]*2000000).all()
    zero=np.array([[e['zero_shot']['competence'] for e in r['envs']] for r in rows]);assert np.array_equal(zero,c[:,:,0])
    rnd=np.array([[e['zero_shot']['random_competence'] for e in r['envs']] for r in rows])
    randoms.extend(rnd.tolist())
    curves[method]=c
    extinct[method]=np.array([[e['zero_shot']['persistence']['extinct'] for e in r['envs']] for r in rows],float)
    checks[method]={'runs':10,'environments':30,'points':11,'max_checkpoint_overshoot':int(np.max(steps-np.arange(11)[None,None,:]*2000000)),'eval_commits':sorted(set(r['git']['commit'] for r in rows))}
    for p in paths:
        b=p.read_bytes(); files[str(p.relative_to(ROOT))]={'bytes':len(b),'git_blob':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()}
best=np.array(core['best']);rnd=np.array(core['random']);assert (best>rnd).all()
checks['random_max_difference']=float(np.max(np.abs(np.array(randoms)-rnd)))
checks['normalization']='Saved core best/random until every core raw evaluation is present'
if all(m in curves for m in ['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr']):
    direct=np.max(np.stack([curves[m].max(axis=(0,2)) for m in ['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr']]),axis=0)
    assert np.array_equal(direct,best)
    checks['normalization']='Core best independently recomputed from all 50 raw evaluation files; exact match'
norm={m:(c-rnd[None,:,None])/(best-rnd)[None,:,None] for m,c in curves.items()}
auc={m:((c[:,:,1:]+c[:,:,:-1])/2*.1).sum(axis=2) for m,c in norm.items()}
measures={'primary':auc,'zero_shot':{m:c[:,:,0] for m,c in norm.items()},'final':{m:c[:,:,-1] for m,c in norm.items()},'extinct':extinct}
comparisons=[('H1','oe','ppo_dr','primary',10,30),('S1','oe','es_dr','primary',10,30),('S2','oe','oe_fixed','primary',10,30),('S3','ppo_dr','ppo_fixed','primary',10,30),('S4','oe','ppo_dr','zero_shot',10,30),('S5','oe','ppo_dr','final',10,30),('S6','oe','ppo_dr','primary',0,10),('S7','oe','ppo_dr','extinct',10,30),('A1','es_dr','es_fixed','primary',10,30),('A2','es_dr','es_fixed','primary',10,30),('A3','es_dr','es_fixed','zero_shot',0,10)]
res={}
for name,m,n,metric,lo,hi in comparisons:
    if m not in curves or n not in curves:continue
    x,y=measures[metric][m][:,lo:hi],measures[metric][n][:,lo:hi]
    if metric=='extinct':x=x.mean(axis=1)[:,None];y=y.mean(axis=1)[:,None]
    u=mannwhitneyu(x.mean(axis=1),y.mean(axis=1),alternative='two-sided')
    res[name]={'difference':float(iqm(x)-iqm(y)),'ci_registered':boot(x,y),'ci_paired_sensitivity':boot(x,y,True),'mwu_U':float(u.statistic),'mwu_p':float(u.pvalue),'paired_signflip_p':signflip(x,y),'mean_difference':float(x.mean()-y.mean()),'run_means_x':x.mean(axis=1).tolist(),'run_means_y':y.mean(axis=1).tolist()}
for family in [[f'S{i}' for i in range(1,8)],['A2','A3']]:
    if all(n in res for n in family):
        for k,v,w in zip(family,holm([res[n]['mwu_p'] for n in family]),holm([res[n]['paired_signflip_p'] for n in family])):res[k].update(p_holm=v,paired_signflip_holm=w)
out={'checks':checks,'file_hashes':files,'iqm':{m:float(iqm(a[:,10:])) for m,a in auc.items()},'comparisons':res,'source_commit':'ea7792af7f66f575b4cb294469a681055fdf69d0','bootstrap_reps':REPS,'bootstrap_seed':SEED,'sensitivity_note':'Paired seed bootstrap and sign flips are post hoc. Not new preregistered evidence.'}
(OUT/'raw_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'checks':checks,'iqm':out['iqm'],'comparisons':{k:{a:b for a,b in v.items() if not a.startswith('run_means')} for k,v in res.items()}},indent=2))
