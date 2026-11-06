"""Offline, deterministic prewriting assets. No training or manuscript prose.
Run from any directory: python scripts/report/prepare_publication.py
Outputs: figures/publication (PDF/SVG/PNG), results/publication (plot data/tables),
         paper/v2/numbers_supplement.tex, docs/FIGURE_GUIDE.md.
Registered estimates are preserved. Newly drawn descriptive bands are explicitly
pointwise whole-run percentile bootstrap intervals (20,000 resamples, seed 20261105).
"""
from pathlib import Path
import csv,hashlib,json,sys,io
from PIL import Image
import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from publication_style import *
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/publication'; FIG=ROOT/'figures/publication'
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
setup(); sources=set(); manifest=[]; REPS=20000; SEED=20261105

def read(path):
 p=ROOT/path;sources.add(path);return json.loads(p.read_text())
def savejson(name,d):
 (OUT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def mean_ci(a):
 a=np.asarray(a);idx=np.random.default_rng(SEED).integers(len(a),size=(REPS,len(a)))
 # batch to keep bootstrap curves within a modest memory footprint
 boots=np.concatenate([a[ix].mean(1) for ix in np.array_split(idx,20)])
 return a.mean(0),*np.percentile(boots,[2.5,97.5],axis=0)
def iqm(a):
 a=np.sort(np.ravel(a));k=len(a)//4;return float(a[k:len(a)-k].mean())
def finish(fig,name,caption,data):
 if name in {'fig01_overview','fig02_primary','fig03_variation','fig04_curves'}:
  caption += (' Normalized adaptation intervals' if name == 'fig04_curves' else ' Neural intervals') + ' use a frozen empirical scoring scale: nominal percentile coverage does not account for selected normalization anchors, fixed environments/episodes or tuning selection; formal conditional 95% coverage is not established.'
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
 fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
 image_bytes=io.BytesIO();fig.savefig(image_bytes,format='png',bbox_inches='tight',dpi=300)
 image_bytes.seek(0);Image.open(image_bytes).load()
 (FIG/(name+'.png')).write_bytes(image_bytes.getvalue())
 contact.savefig(fig,bbox_inches='tight');plt.close(fig)
 savejson(name+'.json',data);manifest.append({'id':name,'caption':caption,'data':'results/publication/'+name+'.json'})
def title(ax,text):ax.set_title(text,loc='left',fontweight='bold',pad=10)
C=read('results/part3/final/analysis.json'); A=read('results/part3/addendum/analysis.json'); P=read('results/part1/final/comparison.json')
I={**C['iqm_primary'],'es_fixed':A['iqm_primary']['es_fixed']}
raw={};learning={};auc={};zero={};ext={};curves={};byset=[]
for m in METHODS:
 base='results/part3/'+('addendum' if m=='es_fixed' else 'final')
 rows=[read(f'{base}/{m}_seed{s}/eval_test.json') for s in range(5000,5010)]
 raw[m]=np.array([[[p['competence'] for p in e['adaptation']['curve']] for e in r['envs']] for r in rows])
 curves[m]=(raw[m]-np.array(C['random'])[None,:,None])/(np.array(C['best'])-C['random'])[None,:,None]
 auc[m]=((curves[m][:,:,1:]+curves[m][:,:,:-1])*.05).sum(2)
 assert np.isclose(iqm(auc[m][:,10:]),I[m][0],rtol=0,atol=1e-12)
 zero[m]=curves[m][:,:,0]
 ext[m]=np.array([[e['zero_shot']['persistence']['extinct'] for e in r['envs']] for r in rows],float)
 learning[m]=np.array([[p['score'] for p in read(f'{base}/{m}_seed{s}/curves.json')['points']] for s in range(5000,5010)])
 for j,st in enumerate(['T1','T2','T3']):
  sl=slice(j*10,(j+1)*10)
  byset.append(dict(method=m,set=st,n_runs=10,n_envs=10,zero_shot_mean=float(zero[m][:,sl].mean()),final_mean=float(curves[m][:,sl,-1].mean()),auc_mean=float(auc[m][:,sl].mean()),extinction_rate=float(ext[m][:,sl].mean())))
pd.DataFrame(byset).to_csv(OUT/'study3_descriptives.csv',index=False)
savejson('study3_primary_per_run.json',{m:{'seeds':list(range(5000,5010)),'auc_T2_T3':auc[m][:,10:].tolist()} for m in METHODS})
contact=PdfPages(FIG/'figure_review.pdf',metadata={'Title':'Publication figure review - preparation only','CreationDate':None,'ModDate':None})
# 1: compact overview, values read from verified sources
fig,axs=plt.subplots(1,3,figsize=(10.4,3.4),layout='constrained')
for ax in axs:ax.set_axis_off()
texts=[('A  Trait experiment','200 seeds per primary group\n4 prespecified arms\n\nSearch-pipeline and mutation\ncomparisons\n\nP1: mean population\nP2: terminal extinction'),('B  Neural experiment',f"10 runs per method\n5 core methods + ES-fixed addendum\n\nOE - PPO-DR adaptation IQM:\n{C['h1']['D']:.3f} [{C['h1']['ci'][0]:.3f}, {C['h1']['ci'][1]:.3f}]\n\nZero-shot and adaptation tests"),('C  Inference scope','Separately tuned pipelines\nOne world and variation schedule\n\nMatched nominal agent steps\nWhole-run uncertainty\n\nHistorical reconstruction:\nseparate supporting evidence')]
for ax,(h,t) in zip(axs,texts):ax.text(0,1,h,va='top',weight='bold',fontsize=11);ax.text(0,.82,t,va='top',linespacing=1.6,fontsize=10)
finish(fig,'fig01_overview','Overview of the new trait and neural experiments and their inference scope. Historical reconstruction is separate supporting material, not the evidence for the neural comparison. Counts refer to seeds or complete training runs as labeled. The displayed Study 3 contrast uses the registered 95% whole-run bootstrap CI (50,000 resamples). ES-fixed was added after core results. These studies do not isolate selection mechanism.',{'panels':texts})
# 2 primary
fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained')
for j,m in enumerate(METHODS):
 v,lo,hi=I[m];ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[m],capsize=4)
 ax.annotate(f'{v:.3f} [{lo:.3f}, {hi:.3f}]',(1.02,j),xycoords=('axes fraction','data'),fontsize=8,va='center')
ax.set_yticks(range(6),[NAMES[m] for m in METHODS]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('IQM of normalized adaptation AUC (T2 and T3)');ax.grid(axis='x');title(ax,'Primary performance during adaptation')
finish(fig,'fig02_primary','Dots: IQM across run-environment adaptation AUCs on T2/T3; bars: registered 95% whole-run bootstrap CIs, 50,000 resamples, seed 4800. Ten independent training runs per method, 20 fixed test environments per run. Normalization uses the shared core random-to-best references; it is not absolute competence. *ES-fixed is the later addendum. OE as implemented here performs substantially worse than PPO-DR.',{m:I[m] for m in METHODS})
# 3 variation without implying paired experimental intervention
fig,axs=plt.subplots(2,3,figsize=(9.5,6.4),sharey='row',layout='constrained');vdata={}
for ax,(var,fix),name in zip(axs[0],[('oe','oe_fixed'),('es_dr','es_fixed'),('ppo_dr','ppo_fixed')],['A  OE','B  ES','C  PPO']):
 for j,m in enumerate([fix,var]):
  vals=auc[m][:,10:].mean(1);ax.scatter(j+np.linspace(-.10,.10,10),vals,color=COLORS[m],s=20,alpha=.65)
  v,lo,hi=I[m];ax.errorbar(j+.23,v,yerr=[[v-lo],[hi-v]],fmt='D',color=COLORS[m],capsize=3)
  vdata[m]={'run_means':vals.tolist(),'pooled_iqm_ci':I[m]}
 ax.set_xticks([0,1],['Fixed*' if fix=='es_fixed' else 'Fixed','Varied']);ax.set_xlim(-.4,1.5);ax.grid(axis='y');title(ax,name)
axs[0,0].set_ylabel('Normalized adaptation AUC (T2 and T3)')
for ax,(var,fix),name in zip(axs[1],[('oe','oe_fixed'),('es_dr','es_fixed'),('ppo_dr','ppo_fixed')],['D  OE: zero-shot','E  ES: zero-shot','F  PPO: zero-shot']):
 for j,m in enumerate([fix,var]):
  vals=zero[m][:,:10].mean(1);v,lo,hi=mean_ci(vals)
  ax.scatter(j+np.linspace(-.10,.10,10),vals,color=COLORS[m],s=20,alpha=.65)
  ax.errorbar(j+.23,v,yerr=[[v-lo],[hi-v]],fmt='s',color=COLORS[m],capsize=3)
  vdata[m]['T1_zero_shot_run_means']=vals.tolist();vdata[m]['T1_zero_shot_mean_ci']=[float(v),float(lo),float(hi)]
 ax.set_xticks([0,1],['Fixed*' if fix=='es_fixed' else 'Fixed','Varied']);ax.set_xlim(-.4,1.5);ax.grid(axis='y');title(ax,name)
axs[1,0].set_ylabel('Normalized zero-shot competence (T1)')
a3=next(x for x in A['family'] if x['name']=='A3');vdata['A3']=a3
fig.supxlabel(f"Registered A3 (ES, pooled IQM difference): {a3['iqm_difference']:+.4f} [{a3['ci'][0]:+.4f}, {a3['ci'][1]:+.4f}]",fontsize=9)

finish(fig,'fig03_variation','Top: adaptation AUC on T2/T3; bottom: zero-shot competence on T1. Small circles: individual run means (10 per method), deterministic horizontal spacing for visibility. Top diamonds/bars: pooled IQMs and registered 95% whole-run bootstrap CIs (50,000 resamples). Bottom squares/bars: means with new descriptive 95% whole-run percentile bootstrap CIs (20,000 resamples, seed 20261105), computed with fixed environments and normalization. The OE/PPO T1 comparisons are descriptive, not new confirmatory tests. The A3 annotation reports the separate registered pooled-IQM contrast, not the difference between the bottom squares. The run mean and pooled IQM are different estimands. *ES-fixed was added after core results. The separately tuned varied ES scores modestly higher and varied OE scores lower than their fixed variants; the PPO contrast is not supported under its registered rule. Separately tuned hyperparameters also differ between fixed and varied variants; variation is not isolated as the sole cause.',vdata)
# 4 curves
fig,axs=plt.subplots(2,3,figsize=(10,6.1),layout='constrained');d={}
for j,st in enumerate(['T1','T2','T3']):
 ax=axs[1,j];title(ax,f'{chr(68+j)}  Adaptation: {st}')
 for m in METHODS:
  run=curves[m][:,j*10:(j+1)*10,:].mean(1);v,l,h=mean_ci(run)
  ax.plot(np.arange(11)*2,v,color=COLORS[m],ls=STYLES[m]);ax.fill_between(np.arange(11)*2,l,h,color=COLORS[m],alpha=.08)
  d[m+'_'+st]={'mean':v.tolist(),'ci_low':l.tolist(),'ci_high':h.tolist()}
 ax.set_xlabel('Adaptation steps (millions)');ax.set_ylim(0,1.05);ax.grid()
for ax,ms,label in zip(axs[0],[METHODS[:2],METHODS[2:4],METHODS[4:]],['A  Validation: OE','B  Validation: PPO','C  Validation: ES']):
 title(ax,label)
 for m in ms:
  v,l,h=mean_ci(learning[m]);ax.plot(np.arange(1,11)*20,v,label=NAMES[m],color=COLORS[m],ls=STYLES[m]);ax.fill_between(np.arange(1,11)*20,l,h,color=COLORS[m],alpha=.13)
  d[m+'_validation']={'mean':v.tolist(),'ci_low':l.tolist(),'ci_high':h.tolist()}
 ax.legend();ax.set_xlabel('Training steps (millions)');ax.set_ylim(0,30);ax.grid()
axs[0,0].set_ylabel('Validation competence minus random');axs[1,0].set_ylabel('Normalized competence')
finish(fig,'fig04_curves','Top: validation score across nominal training checkpoints; bottom: normalized adaptation competence, averaging the ten environments in each test set within each run. Curves are means over 10 runs; bands are new descriptive pointwise 95% whole-run percentile bootstrap CIs (20,000 resamples, seed 20261105), not simultaneous confidence bands. *ES-fixed is a later addendum. Training and adaptation axes are nominal living-agent-steps; actual checkpoint overshoots are recorded separately. Training observations begin at nominal 20 million steps and do not measure gains from initialization. Mean curves can conceal different individual-run changes. Adaptation scores include starting performance.',d)
# 5 profiles and probability
fig,axs=plt.subplots(1,2,figsize=(10.3,4.2),layout='constrained');tau=np.linspace(0,1,101);profiles={};pm=[]
for m in METHODS:
 p=(auc[m][:,10:,None]>tau).mean((0,1));profiles[m]=p.tolist();axs[0].plot(tau,p,label=NAMES[m],color=COLORS[m],ls=STYLES[m])
for m in METHODS:
 x=auc[m][:,10:].mean(1);row=[]
 for n in METHODS:
  y=auc[n][:,10:].mean(1);row.append(float(((x[:,None]>y)+.5*(x[:,None]==y)).mean()))
 pm.append(row)
axs[0].set_xlabel('Adaptation AUC threshold');axs[0].set_ylabel('Fraction of run-environment scores above threshold');axs[0].legend();title(axs[0],'A  Empirical performance profiles')
im=axs[1].imshow(pm,vmin=0,vmax=1,cmap='cividis');axs[1].set_xticks(range(6),[NAMES[m] for m in METHODS],rotation=45,ha='right');axs[1].set_yticks(range(6),[NAMES[m] for m in METHODS]);title(axs[1],'B  Probability of improvement')
for i in range(6):
 for j in range(6):axs[1].text(j,i,'—' if i==j else f'{pm[i][j]:.2f}',ha='center',va='center',color='white' if pm[i][j]<.45 else 'black',fontsize=8)
finish(fig,'fig05_profiles','Descriptive plots. Left: empirical fraction above each threshold among 10 runs x 20 fixed T2/T3 environments, with no confidence bands; environment observations are not treated as independent runs. Right: fraction of all 100 cross-run comparisons in which the row method mean exceeds the column method mean, with ties counting one half; diagonal omitted. This is an effect summary, not a significance test. *ES-fixed is a later addendum. No confirmatory ES-DR versus PPO-DR superiority claim is made.',{'threshold':tau.tolist(),'profiles':profiles,'methods':METHODS,'probability':pm})
# 6 original transfer, direct raw count ratios
held=read('results/part1/paper_heldout_digitized.json')['figures'];pop=[];food=[]
for s in range(30):
 path=f'results/part1/final/ET_seed{s:04}.npz';sources.add(path)
 with np.load(ROOT/path) as z:
  pop.append(z['g_population'][100:]);food.append((z['g_food_eaten']/z['g_population'])[100:])
fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained');d={}
for ax,vals,key,ylab in zip(axs,[food,pop],['fig6_fitness_transfer','fig7_population_transfer'],['Food eaten per blob','Population size']):
 v,l,h=mean_ci(vals);x=np.arange(len(v));ax.plot(x,v,color=COLORS['oe'],label='Rebuilt ET');ax.fill_between(x,l,h,color=COLORS['oe'],alpha=.18)
 for c,col in [('et',COLORS['oe']),('gs','#0072B2')]:
  pts=held[key]['curves'][c];xx=range(min(map(int,pts)),max(map(int,pts))+1);ax.plot(xx,[pts.get(str(i),float('nan')) for i in xx],ls='--' if c=='et' else ':',color=col,label='Original '+c.upper()+' (digitized)')
 ax.set_xlabel('Transfer generation');ax.set_ylabel(ylab);ax.legend(fontsize=7);ax.grid();d[key]={'rebuilt_mean':v.tolist(),'ci_low':l.tolist(),'ci_high':h.tolist(),'original_digitized':held[key]['curves']}
title(axs[0],'A  Transfer food intake');title(axs[1],'B  Transfer population')
finish(fig,'fig06_reconstruction','Earlier calibrated reconstruction under paper-described conditions; the recovered original implementation uses different conditions. Solid lines: means over 30 fresh ET trajectories; bands: descriptive pointwise 95% whole-run percentile bootstrap CIs (20,000 resamples, seed 20261105). Dashed/dotted curves are digitized original-paper curves with approximate values and no reconstructed uncertainty; gaps are preserved. No rebuilt GS transfer curve exists: all 30 GS runs became extinct during training. Original ET transfer population is not reproduced by this reconstruction; recovered original CSV outcomes are shown separately in figS02_original_archive.',d)
# 7 study2 means and binomial intervals
path='results/part2/final/analysis/outcomes.csv';sources.add(path);df=pd.read_csv(ROOT/path)
groups=['ET','ET_freezeB','ET_cloneB','GS_food_frozen','GS_pop_frozen','GS_pop_mut','SV_frozen','ORACLE_B'];labels=['ET','ET freeze','ET clone','GS food','GS pop','GS pop mut','SV','Oracle'];table=[]
fig,axs=plt.subplots(2,4,figsize=(11.5,6.2),sharey='row',layout='constrained')
for j,w in enumerate(['main','arm2','arm3','arm4']):
 for i,g in enumerate(groups):
  a=df[(df.world==w)&(df.group==g)];assert len(a)==200
  v,l,h=mean_ci(a.P1.to_numpy());p=float(a.P2.mean());z=norm.ppf(.975);den=1+z*z/200;center=(p+z*z/400)/den;half=z*np.sqrt(p*(1-p)/200+z*z/(4*200**2))/den
  axs[0,j].errorbar(i,v,yerr=[[v-l],[h-v]],fmt='o',color='#0072B2',capsize=2,ms=4);axs[1,j].errorbar(i,p,yerr=[[max(0,p-(center-half))],[max(0,(center+half)-p)]],fmt='o',color='#0072B2',capsize=2,ms=4)
  table.append(dict(arm=w,group=g,n=200,P1_mean=float(v),P1_sd=float(a.P1.std(ddof=1)),P1_ci_low=float(l),P1_ci_high=float(h),P2_rate=p,P2_ci_low=center-half,P2_ci_high=center+half))
 title(axs[0,j],['A  Main','B  Arm 2','C  Arm 3','D  Arm 4'][j])
 for ax in axs[:,j]:ax.set_xticks(range(8),labels,rotation=65,ha='right');ax.grid(axis='y')
axs[0,0].set_ylabel('P1: mean population (blobs)');axs[1,0].set_ylabel('P2: extinction probability');axs[1,0].set_ylim(-.04,1.04)
finish(fig,'fig07_trait_experiment','All eight primary groups across four prespecified arms; n=200 seeds per group. Top: mean P1 (population at generation start, averaged over B generations 40–49, including zero-population generations) and new descriptive whole-seed percentile bootstrap 95% CIs (20,000 resamples). Bottom: P2 extinction proportion (no agent would start B generation 50) and Wilson 95% CIs. These group intervals do not replace registered contrast tests, Holm corrections or equivalence rules. Shared seeds induce dependence across groups/arms. Ceiling behavior in the main arm limits interpretation. The search-pipeline comparison changes both objective and scoring horizon (population: generations 10-99; food: 10-29). Cloning changes realized traits as well as heterogeneity, and clone deficits also occur without a shift. Consult confirmatory tables for verdicts.',table)
pd.DataFrame(table).to_csv(OUT/'study2_group_summary.csv',index=False)
# 8 exploratory long training
fig,ax=plt.subplots(figsize=(8,4.1),layout='constrained');d={}
for s in range(5000,5010):
 pts=read(f'results/part3/final/oe_seed{s}/curves.json')['points'];ax.plot([p['agent_steps']/1e8 for p in pts],[p['score'] for p in pts],color='.75',lw=.8,label='Core OE (10 runs)' if s==5000 else None)
for s,col in zip(range(5000,5003),['#D55E00','#0072B2','#009E73']):
 pts=read(f'results/part3/explore_long/oe_seed{s}/curves.json')['points'];xx=[p['agent_steps']/1e8 for p in pts];yy=[p['score'] for p in pts];gain=(yy[8]+yy[9]-yy[0]-yy[1])/2
 ax.plot(xx,yy,marker='o',ms=3,color=col,label=f'Seed {s}; gain {gain:+.2f}');d[str(s)]={'steps_1e8':xx,'score':yy,'gain':gain}
ax.set_xlabel('Training living-agent-steps (hundreds of millions)');ax.set_ylabel('Validation competence minus random');ax.legend();ax.grid();title(ax,'Exploratory: five-times-longer OE training')
finish(fig,'fig08_long_training','Exploratory, decided after the core results: three newly trained OE runs at 1 billion steps, alongside ten original 200-million-step runs (gray). Lines connect saved validation checkpoints; no uncertainty interval or significance test is assigned to these three runs. Gain is mean of checkpoints 9-10 minus mean of 1-2. None exceeds the prewritten descriptive 4.3-point rule; its noise estimate was later found unreliable, so it is not a calibrated statistical threshold. Same-seed runs are not continuations and need not match across hardware.',d)
# S1 diagnostics
fig,axs=plt.subplots(1,3,figsize=(10.2,3.5),layout='constrained');diagnostic={}
for m in ['oe','oe_fixed']:
 for j,key in enumerate(['weights','behavior']):axs[j].plot(np.arange(1,11)*20,C['diversity_over_time'][m][key],label=NAMES[m],color=COLORS[m],ls=STYLES[m])
 diagnostic[m]={'diversity':C['diversity_over_time'][m],'population':C['oe_population'][m]['training_mean']}
for j,k in enumerate(['Weights diversity','Behavior diversity']):title(axs[j],k);axs[j].set_xlabel('Training steps (millions)');axs[j].legend();axs[j].grid()
for j,m in enumerate(METHODS):axs[2].plot([0,1,2],[ext[m][:,i*10:(i+1)*10].mean() for i in range(3)],marker='o',ms=3,label=NAMES[m],color=COLORS[m],ls=STYLES[m])
axs[2].set_xticks([0,1,2],['T1','T2','T3']);axs[2].set_ylim(0,1);title(axs[2],'Zero-shot extinction rate');axs[2].legend(fontsize=6)
finish(fig,'figS01_diagnostics','Descriptive diagnostics. Left/middle: saved mean diversity among 16 sampled agents per training snapshot over ten runs, weight diversity is mean pairwise Euclidean parameter distance; behavior diversity is mean pairwise Euclidean action distance averaged over 256 fixed validation probes; no confidence bands. The companion diversity audit records checkpoint-level verification and its numerical tolerance; see results/audit/diversity_audit.json. Right: zero-shot persistence-test extinction fraction over ten runs x ten environments per set, distinct from training with a population floor; no intervals. *ES-fixed is the later addendum. Higher diversity alone does not establish a causal robustness mechanism.',diagnostic)
# S2: recovered original archive; fixed run denominator, terminal extinction carried forward.
fig,axs=plt.subplots(1,3,figsize=(10.4,3.6),layout='constrained');archive_data={}
for group,color in [('et','#0072B2'),('gs','#D55E00')]:
 archive_data[group]={}
 for phase,horizon,j in [('training',100,0),('transfer',50,1)]:
  path=f'results/original_study/{group}_{phase}_metrics.csv';sources.add(path);dd=pd.read_csv(ROOT/path)
  values=np.zeros((30,horizon));alive=np.zeros((30,horizon))
  for i,(_,rr) in enumerate(dd.groupby('run')):
   ix=rr.generation.to_numpy(int);values[i,ix]=rr.population.to_numpy()*rr.survivalRate.to_numpy();alive[i,ix]=(values[i,ix]>0)
   assert len(rr)==horizon or rr.survivalRate.iloc[-1]==0
  v,l,h=mean_ci(values);xx=np.arange(horizon)
  axs[j].plot(xx,v,label=group.upper(),color=color);axs[j].fill_between(xx,l,h,color=color,alpha=.18)
  archive_data[group][phase]={'end_generation_alive_by_run':values.tolist(),'mean':v.tolist(),'ci_low':l.tolist(),'ci_high':h.tolist(),'surviving_run_fraction':alive.mean(0).tolist()}
  if phase=='transfer':axs[2].step(xx,alive.mean(0),where='post',label=group.upper(),color=color)
for j,heading in enumerate(['A  Original training','B  Original transfer','C  Transfer persistence']):
 title(axs[j],heading);axs[j].set_xlabel('Generation (zero-based)');axs[j].legend();axs[j].grid()
axs[0].set_ylabel('Living agents at generation end');axs[1].set_ylabel('Living agents at generation end');axs[2].set_ylabel('Fraction of original runs still alive');axs[2].set_ylim(-.03,1.03)
finish(fig,'figS02_original_archive','Recovered historical CSVs, 30 runs per group (seeds 1000–1029); these are not additional independent experiments. Panels A/B show generation-end living agents, computed as population times survival rate. Terminal extinction is carried forward as zero to the common horizon, retaining all 30 runs. Bands: descriptive pointwise 95% whole-run percentile bootstrap intervals (20,000 resamples, seed 20261105). Panel C shows the empirical surviving-run fraction, without intervals. ET persists through all 50 transfer generations; every GS run is extinct by recorded generation 26. These are archived values: current runtime replay is not numerically exact. The code uses different transfer conditions from the report and disables predation through an inverted guard. This plot does not identify a causal effect of diversity.',archive_data)
contact.close()
# Tables preserving all registered comparisons
rows=[]
for family,rr in [('core',C['secondary']),('addendum',A['family'])]:
 for row in rr:rows.append({'family':family,**{k:v for k,v in row.items() if not isinstance(v,(dict,list))},'ci_low':row['ci'][0],'ci_high':row['ci'][1]})
pd.DataFrame(rows).to_csv(OUT/'registered_comparisons.csv',index=False)
savejson('figures.json',manifest)
(ROOT/'docs/FIGURE_GUIDE.md').write_text('# Publication figure guide\n\nFigure sources and captions. Rebuild: `python scripts/report/prepare_publication.py`. PDFs and SVGs are vector outputs; PNGs are 300 dpi. Shared style: `publication_style.py`. The manuscript uses seven of these ten figures.\n\n'+ '\n\n'.join('## '+x['id']+'\n\n'+x['caption']+'\n\nData: `'+x['data']+'`.' for x in manifest)+'\n')
savejson('source_manifest.json',{p:{'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),'bytes':(ROOT/p).stat().st_size} for p in sorted(sources)})
print(f'Created {len(manifest)} figures, review PDF, data tables and source manifest.')
