"""Independent statistical audit of the saved outcome table (not raw trajectories).
No project analysis modules imported. Registered RNG streams retained for exact checks.
"""
from pathlib import Path
import csv,json
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'results/part2/final/analysis'
a=json.loads((D/'confirmatory.json').read_text())
e=json.loads((D/'exploratory.json').read_text())['secondary']
rows=list(csv.DictReader((D/'outcomes.csv').open()))
pairs=[('GS_food_frozen','ET'),('GS_pop_frozen','ET'),('GS_pop_mut','ET'),('SV_frozen','ET'),('ET_freezeB','ET'),('ET_cloneB','ET_freezeB'),('GS_pop_frozen','GS_food_frozen'),('GS_pop_mut','GS_pop_frozen')]
checks=[]; records={}
def rng(arm,c,out,purpose):return np.random.default_rng(np.random.SeedSequence(2999,spawn_key=(arm,c,out,purpose)))
def same(label,x,y):
 err=float(np.max(np.abs(np.asarray(x,float)-np.asarray(y,float))))
 checks.append({'check':label,'max_error':err,'pass':bool(np.allclose(x,y,rtol=1e-10,atol=1e-12))})
def boot(x,y,arm,c,levels):
 d=x.mean()-y.mean(); se=np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
 if se==0:return [[d,d] for lv in levels]
 r=rng(arm,c,1,1);t=[]
 for _ in range(10):
  xx=x[r.integers(len(x),size=(2000,len(x)))];yy=y[r.integers(len(y),size=(2000,len(y)))];s=np.sqrt(xx.var(1,ddof=1)/len(x)+yy.var(1,ddof=1)/len(y));s=np.where(s>0,s,se);t.extend(((xx.mean(1)-yy.mean(1)-d)/s).tolist())
 return [[d-q[1]*se,d-q[0]*se] for lv in levels for q in [np.quantile(t,[(1-lv)/2,(1+lv)/2])]]
def perm(x,y,arm,c):
 if np.all(x==x[0]) and np.all(y==x[0]):return 1.
 r=rng(arm,c,1,0);z=np.r_[x,y];count=0;obs=abs(x.mean()-y.mean())
 for _ in range(5):
  v=r.permuted(np.tile(z,(2000,1)),axis=1);diff=v[:,:len(x)].mean(1)-v[:,len(x):].mean(1);count+=int((np.abs(diff)>=obs-1e-12).sum())
 return (count+1)/10001

def wilson(k,n,z):
 p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return center-h,center+h

def newcombe(x,y,lv):
 p=x.mean();q=y.mean();z=stats.norm.ppf((1+lv)/2);l,u=wilson(x.sum(),len(x),z);ll,uu=wilson(y.sum(),len(y),z)
 return [p-q-np.hypot(p-l,uu-q),p-q+np.hypot(u-p,q-ll)]
def holm(ps):
 out=np.zeros(len(ps));last=0
 for rank,i in enumerate(np.argsort(ps,kind='stable')):last=max(last,min(1,(len(ps)-rank)*ps[i]));out[i]=last
 return out
for arm_id,arm in enumerate(['main','arm2','arm3','arm4'],1):
 records[arm]={};ps=[];refs=[]
 for c,(gx,gy) in enumerate(pairs,1):
  if arm=='arm2' and c==7:continue
  sets=[sorted([r for r in rows if r['world']==arm and r['group']==g],key=lambda r:int(r['seed'])) for g in [gx,gy]]
  assert all([int(r['seed']) for r in rs]==list(range(2000,2200)) for rs in sets)
  x,y=[np.array([int(r['P1_total']) for r in rs])/10 for rs in sets]
  ex,ey=[np.array([r['P2']=='True' for r in rs]) for rs in sets]
  sx,sy=[np.array([r['S5']=='True' for r in rs]) for rs in sets];both=sx&sy
  raw=a[arm]['raw'][str(c)];ks=[14,16] if arm=='arm2' else [16];levels=[.95]+[1-.1/k for k in ks]
  cis=boot(x,y,arm_id,c,levels);pp=perm(x,y,arm_id,c);pf=stats.fisher_exact([[ex.sum(),len(ex)-ex.sum()],[ey.sum(),len(ey)-ey.sum()]]).pvalue
  same(f'{arm}/{c}/P1-diff',x.mean()-y.mean(),raw['P1']['diff']);same(f'{arm}/{c}/P1-p',pp,raw['P1']['p']);same(f'{arm}/{c}/P1-CI',cis[0],raw['P1']['ci95'])
  same(f'{arm}/{c}/P2-diff',ex.mean()-ey.mean(),raw['P2']['diff']);same(f'{arm}/{c}/P2-p',pf,raw['P2']['p']);same(f'{arm}/{c}/P2-CI',newcombe(ex,ey,.95),raw['P2']['ci95'])
  for i,k in enumerate(ks,1):same(f'{arm}/{c}/P1-eq{k}',cis[i],raw['P1']['ci_eq'][str(k)]);same(f'{arm}/{c}/P2-eq{k}',newcombe(ex,ey,levels[i]),raw['P2']['ci_eq'][str(k)])
  d=x-y;signs=rng(50+arm_id,c,1,2).choice([-1.,1.],size=(10000,len(d)));sp=(1+int((np.abs((signs*d).mean(1))>=abs(d.mean())-1e-12).sum()))/10001
  discord=int((ex!=ey).sum());mc=stats.binomtest(int((ex&~ey).sum()),discord,.5).pvalue if discord else 1.
  paired=e[arm]['paired'][str(c)];same(f'{arm}/{c}/paired-signflip',sp,paired['signflip_p']);same(f'{arm}/{c}/mcnemar',mc,paired['mcnemar_p'])
  ci=np.quantile(d[rng(50+arm_id,c,1,3).integers(len(d),size=(10000,len(d)))].mean(1),[.025,.975]);same(f'{arm}/{c}/paired-CI',ci,paired['P1_ci95'])
  reach=a[arm]['reached_B_analysis'][str(c)];same(f'{arm}/{c}/reached-count',both.sum(),reach['n_both'])
  if both.any():same(f'{arm}/{c}/reached-P1',d[both].mean(),reach['P1']);same(f'{arm}/{c}/reached-P2',(ex[both].astype(int)-ey[both]).mean(),reach['P2'])
  records[arm][c]={'P1_diff':float(x.mean()-y.mean()),'P1_ci95':cis[0],'P1_p':pp,'P2_p':float(pf),'paired_P1_p':sp,'paired_P2_p':float(mc),'n_both':int(both.sum())}
  ps.extend([pp,pf]);refs.extend([(c,'P1'),(c,'P2')])
 # Compare adjusted p-values wherever stored.
 adj=holm(ps)
 for (c,o),p in zip(refs,adj):
  rec=a[arm]['decided'][str(c)]
  if 'p_adj' in rec:same(f'{arm}/{c}/{o}-Holm',p,rec['p_adj'][o])
out={'scope':'Independent recomputation from saved per-run outcomes, not raw NPZ trajectories','checks':checks,'comparisons':records,'passed':sum(c['pass'] for c in checks),'failed':[c for c in checks if not c['pass']]}
p=ROOT/'results/audit/part2_outcomes_audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':out['passed'],'failed':out['failed']},indent=2))
assert not out['failed']
