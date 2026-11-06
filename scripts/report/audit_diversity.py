"""Check all 220 OE policy/snapshot diversity records against saved summaries.
Selection keys and the deterministic probe-observation generator are shared with
the original analysis. Pairwise distances and neural forward passes are independently
computed in NumPy/SciPy. No training or test-environment evaluation is performed.
The original simulator generates validation-only probe observations, not new outcomes.
"""
from pathlib import Path
import sys,json,pickle,hashlib,argparse
import numpy as np
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/part3')]
import jax
from evo2.seeds import base_key,stream_key
OUT=ROOT/'results/audit';probe=OUT/'diversity_probe_observations.npy'
a=argparse.ArgumentParser();a.add_argument('--probes-only',action='store_true');args=a.parse_args()
if not probe.exists():
 from analyze import probe_observations
 np.save(probe,probe_observations())
if args.probes_only:print('Saved 256 validation probe observations');sys.exit()
probes=np.load(probe);key=stream_key(base_key('stats',4801),'probe');original=json.loads((ROOT/'results/part3/final/analysis.json').read_text());results={};records=[]
for mi,m in enumerate(['oe','oe_fixed']):
 allrows=[]
 for ri,s in enumerate(range(5000,5010)):
  k=jax.random.fold_in(jax.random.fold_in(key,1000+mi),ri);rows=[]
  for t in range(11):
   path=ROOT/f'results/part3/final/{m}_seed{s}'/('policy.pkl' if t==0 else f'snapshots/{t}.pkl');b=path.read_bytes();pol=pickle.loads(b);pol=pol if t==0 else pol['policy'];alive=np.asarray(pol['alive'],bool);g=pol['genomes'];n=int(alive.sum());assert n>0
   u=np.asarray(jax.random.uniform(jax.random.fold_in(k,t),alive.shape));idx=np.argsort(np.where(alive,u,np.inf),kind='stable')[np.arange(16)%n]
   weights={kk:np.asarray(v)[idx] for kk,v in g.brain.items()};flat=np.concatenate([weights[kk].reshape(16,-1) for kk in sorted(weights)],axis=1)
   actions=[]
   for i in range(16):
    h=np.tanh(probes@weights['W1'][i]+weights['b1'][i]);h=np.tanh(h@weights['W2'][i]+weights['b2'][i]);actions.append(np.tanh(h@weights['W3'][i]+weights['b3'][i]))
   acts=np.array(actions,dtype=float);behavior=np.mean([np.linalg.norm(acts[i]-acts[j],axis=1).mean() for i in range(16) for j in range(i+1,16)])
   row={'weights':float(pdist(flat.astype(float)).mean()),'traits':float(pdist(np.asarray(g.traits)[idx].astype(float)).mean()),'behavior':float(behavior)};rows.append(row)
   records.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'n_alive':n,'diversity':row})
  allrows.append(rows)
 results[m]={'end':{k:float(np.mean([x[0][k] for x in allrows])) for k in rows[0]},'over_time':{k:np.mean([[x[t][k] for t in range(1,11)] for x in allrows],axis=0).tolist() for k in rows[0]}}
checks=[]
for m,res in results.items():
 for section,field in [('end','diversity_end'),('over_time','diversity_over_time')]:
  for k,v in res[section].items():
   error=float(np.max(np.abs(np.asarray(v)-original[field][m][k])));tol=5e-6 if k=='behavior' else 1e-8;checks.append(dict(method=m,field=section+'.'+k,max_absolute_error=error,tolerance=tol,passed=error<=tol))
output={'scope':__doc__,'records':records,'computed':results,'checks':checks,'all_passed':all(x['passed'] for x in checks),'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'jax':jax.__version__,'backend':jax.default_backend()}
(OUT/'diversity_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'records':len(records),'checks':checks,'all_passed':output['all_passed']},indent=2));assert output['all_passed']
