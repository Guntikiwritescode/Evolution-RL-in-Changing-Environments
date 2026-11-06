"""Independent existing-data check. No study code is imported; no simulations run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/audit/viability/reanalysis/statistics'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / 'results/publication/all_evaluation_points.csv')
METHODS = ['oe', 'oe_fixed', 'ppo_fixed', 'ppo_dr', 'es_dr', 'es_fixed']
CORE = METHODS[:-1]
raw, norm, step, extinct = {}, {}, {}, {}
assert len(df) == 6 * 10 * 30 * 11
assert not df.duplicated(['method','seed','environment_index','checkpoint']).any()
for m in METHODS:
    part = df.loc[df.method == m].sort_values(['seed','environment_index','checkpoint'])
    assert len(part) == 3300
    assert sorted(part.seed.unique().tolist()) == list(range(5000,5010))
    raw[m] = part.competence.to_numpy().reshape(10,30,11)
    norm[m] = part.normalized.to_numpy().reshape(10,30,11)
    step[m] = part.agent_steps.to_numpy().reshape(10,30,11)
    extinct[m] = part.extinct.to_numpy().reshape(10,30,11)[:,:,0].astype(float)

best = np.max(np.stack([v.max(axis=(0,2)) for m,v in raw.items() if m in CORE]),axis=0)
# Algebraic reconstruction of saved random anchors, to check all published normalized points.
ref = raw['oe'][0,:,0]
refnorm = norm['oe'][0,:,0]
random = (ref - refnorm*best)/(1-refnorm)
residual = max(np.max(np.abs(norm[m] - (raw[m]-random[None,:,None])/(best-random)[None,:,None])) for m in METHODS)
assert residual < 1e-12

def iqm(a):
    a = np.asarray(a)
    flat = np.sort(a.reshape(*a.shape[:-2],-1), axis=-1) if a.ndim > 2 else np.sort(a.ravel())
    k = flat.shape[-1] // 4
    return flat[...,k:-k].mean(axis=-1)

def boot(x,y=None,paired=False,seed=4800,reps=50000):
    rng=np.random.default_rng(seed)
    out=np.empty(reps)
    # Alternating draws replicate the preregistered call order.
    for start in range(0,reps,1000):
        n=min(1000,reps-start)
        if y is None:
            ix=rng.integers(0,len(x),size=(n,len(x)))
            out[start:start+n]=iqm(x[ix])
        elif paired:
            ix=rng.integers(0,len(x),size=(n,len(x)))
            out[start:start+n]=iqm(x[ix])-iqm(y[ix])
        else:
            draws=rng.integers(0,len(x),size=(n,2,len(x)))
            out[start:start+n]=iqm(x[draws[:,0]])-iqm(y[draws[:,1]])
    point=float(iqm(x) if y is None else iqm(x)-iqm(y))
    return {'estimate':point,'ci':np.quantile(out,[.025,.975]).tolist()}

weights=np.array([.05]+[.1]*9+[.05])
auc={m: v@weights for m,v in norm.items()}
rawauc={m:v@weights for m,v in raw.items()}
actual_auc={}
overshoot={}
for m in METHODS:
    z=np.empty((10,30))
    for r in range(10):
        for e in range(30):
            x=step[m][r,e]/20_000_000
            assert np.all(np.diff(x)>0) and x[0]==0 and x[-1]>=1
            keep=x<1
            xx=np.r_[x[keep],1.0]
            yy=np.r_[norm[m][r,e,keep],np.interp(1.0,x,norm[m][r,e])]
            z[r,e]=np.trapezoid(yy,xx)
    actual_auc[m]=z
    overshoot[m]={'final_mean':float(step[m][:,:,-1].mean()),'final_max':int(step[m][:,:,-1].max()),
                  'first_checkpoint_max_fraction_over_target':float((step[m][:,:,1]/2_000_000-1).max()),
                  'max_checkpoint_excess_as_fraction_total_budget':float(((step[m]-np.arange(11)*2_000_000)/20_000_000).max()),
                  'actual_minus_nominal_pooled_iqm':float(iqm(z[:,10:])-iqm(auc[m][:,10:]))}

comparisons=[
 ('H1','oe','ppo_dr','primary',slice(10,30)),
 ('S1','oe','es_dr','primary',slice(10,30)),
 ('S2','oe','oe_fixed','primary',slice(10,30)),
 ('S3','ppo_dr','ppo_fixed','primary',slice(10,30)),
 ('S4','oe','ppo_dr','zero',slice(10,30)),
 ('S5','oe','ppo_dr','final',slice(10,30)),
 ('S6','oe','ppo_dr','primary',slice(0,10)),
 ('S7','oe','ppo_dr','extinct',slice(10,30)),
 ('A1','es_dr','es_fixed','primary',slice(10,30)),
 ('A3','es_dr','es_fixed','zero',slice(0,10))]
measures={'primary':auc,'zero':{m:v[:,:,0] for m,v in norm.items()},'final':{m:v[:,:,-1] for m,v in norm.items()},'extinct':extinct}
checked={}
for name,a,b,measure,sl in comparisons:
    x,y=measures[measure][a][:,sl],measures[measure][b][:,sl]
    xm,ym=x.mean(axis=1),y.mean(axis=1)
    if measure=='extinct': x,y=xm[:,None],ym[:,None]
    mw=stats.mannwhitneyu(xm,ym,alternative='two-sided')
    exact=stats.mannwhitneyu(xm,ym,alternative='two-sided',method='exact') if len(np.unique(np.r_[xm,ym]))==20 else None
    checked[name]={**boot(x,y),'paired_bootstrap':boot(x,y,paired=True),'U':float(mw.statistic),'p':float(mw.pvalue),
                   'exact_p_if_no_ties':None if exact is None else float(exact.pvalue),'run_mean_difference':float(xm.mean()-ym.mean()),
                   'run_mean_seed_correlation':float(np.corrcoef(xm,ym)[0,1])}
    if measure=='primary':
        checked[name]['actual_step_auc']=boot(actual_auc[a][:,sl],actual_auc[b][:,sl])
        checked[name]['raw_auc_difference']=float(iqm(rawauc[a][:,sl])-iqm(rawauc[b][:,sl]))
        deletions=[]
        for e in range(20):
            if sl.start!=10: break
            deletions.append(float(iqm(np.delete(x,e,axis=1))-iqm(np.delete(y,e,axis=1))))
        if deletions: checked[name]['leave_one_env_out_iqm_difference_range']=[min(deletions),max(deletions)]

published=pd.read_csv(ROOT/'results/publication/registered_comparisons.csv').set_index('name')
for name,row in published.iterrows():
    got=checked['A1' if name=='A2' else name]
    assert abs(got['estimate']-row.iqm_difference)<1e-12
    assert abs(got['p']-row.p)<1e-12
    assert np.max(np.abs(np.array(got['ci'])-[row.ci_low,row.ci_high]))<1e-12
    got['matches_published_effect_p_and_ci']=True

for family in ['core','addendum']:
    sub=published.loc[published.family==family]
    p=sub.p.to_numpy(); order=np.argsort(p); adjusted=np.empty(len(p)); running=0.
    for rank,j in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*p[j])); adjusted[j]=running
    assert np.max(np.abs(adjusted-sub.p_holm.to_numpy()))<1e-12

# Re-estimated-anchor bootstrap is only a post hoc sensitivity calculation;
# the maximum estimator and small n do not justify a general coverage claim.
rng=np.random.default_rng(714397)
reps=50000
boot_adaptive={name:[] for name,*_ in comparisons}
boot_fixed={name:[] for name,*_ in comparisons}
for start in range(0,reps,250):
    n=min(250,reps-start)
    ix={m:rng.integers(0,10,size=(n,10)) for m in METHODS}
    anchor=np.max(np.stack([raw[m].max(axis=2)[ix[m]].max(axis=1) for m in CORE]),axis=0)
    for name,a,b,measure,sl in comparisons:
        if measure=='extinct': continue
        d=[]; f=[]
        for m in [a,b]:
            vals=rawauc[m] if measure=='primary' else raw[m][:,:,0 if measure=='zero' else -1]
            d.append((vals[ix[m]]-random)/(anchor[:,None,:]-random))
            f.append(measures[measure][m][ix[m]])
        boot_adaptive[name].append(iqm(d[0][:,:,sl])-iqm(d[1][:,:,sl]))
        boot_fixed[name].append(iqm(f[0][:,:,sl])-iqm(f[1][:,:,sl]))
for name in boot_adaptive:
    if not boot_adaptive[name]: continue
    checked[name]['reestimated_anchor_bootstrap_sensitivity_ci']=np.quantile(np.concatenate(boot_adaptive[name]),[.025,.975]).tolist()
    checked[name]['same_draws_fixed_anchor_ci']=np.quantile(np.concatenate(boot_fixed[name]),[.025,.975]).tolist()

iqms={m:boot(auc[m][:,10:]) for m in METHODS}
published_iqm=json.loads((ROOT/'results/publication/fig02_primary.json').read_text())
for m,d in iqms.items(): assert np.max(np.abs(np.array([d['estimate'],*d['ci']])-published_iqm[m]))<1e-12
report={'scope':'Independent recomputation from publication CSV; not yet raw-record provenance verification.',
        'normalization_residual':float(residual),'core_best':best.tolist(),'reconstructed_random':random.tolist(),
        'random_to_best_span_minmax':[float((best-random).min()),float((best-random).max())],
        'budget_checks':overshoot,'pooled_iqms':iqms,'comparisons':checked}
(OUT/'independent_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_published_iqm_effects_p_values_and_cis_match':True,'normalization_residual':residual,
 'comparisons':{k:{a:b for a,b in d.items() if a in ['estimate','ci','paired_bootstrap','actual_step_auc','reestimated_anchor_bootstrap_sensitivity_ci','run_mean_difference']} for k,d in checked.items()},
 'budget_checks':overshoot},indent=2))
