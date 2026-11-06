"""Generate manuscript tables from retained analysis records; no experimental runs."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
D=R/'data';T=R/'tables';T.mkdir(exist_ok=True)
def rd(n):return json.loads((D/n).read_text())
def csvread(n):
 with (D/n).open(newline='') as stream:return list(csv.DictReader(stream))
def esc(s):
 return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#').replace('−','-').replace('—','; ')
def f(x,n=4):return f'{x:.{n}f}'
def ci(d,n=4):return f"${d['iqm_difference']:+.{n}f}$ $[{d['ci'][0]:.{n}f},{d['ci'][1]:.{n}f}]$"
def pv(x):
 if x>=.001:return f'{x:.4f}'.rstrip('0').rstrip('.')
 a,b=f'{x:.2e}'.split('e');return f'${a}\\times10^{{{int(b)}}}$'
def table(name,caption,label,cols,heads,rows,long=False):
 cols=cols.replace('X','Y').replace('p{','P{')
 if long:
  out=[r'\Needspace{14\baselineskip}','{\\small',r'\begin{longtable}{'+cols+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',' & '.join(heads)+r'\\\midrule',r'\endfirsthead',r'\toprule',' & '.join(heads)+r'\\\midrule',r'\endhead',r'\bottomrule\endfoot']
 else:out=[r'\begin{table}[tb]',r'\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabularx}{\linewidth}{'+cols+'}',r'\toprule',' & '.join(heads)+r'\\\midrule']
 out+=[' & '.join(str(c) for c in row)+r'\\' for row in rows]
 out+=[r'\end{longtable}','}'] if long else [r'\bottomrule',r'\end{tabularx}',r'\end{table}']
 (T/(name+'.tex')).write_text('\n'.join(out)+'\n')
C=rd('neural_core.json');A=rd('neural_addendum.json');P=rd('trait_confirmatory.json')
N={'oe':'OE','oe_fixed':'OE-fixed','ppo_dr':'PPO-DR','ppo_fixed':'PPO-fixed','es_dr':'ES-DR','es_fixed':'ES-fixed*'}
methods=list(N);desc=csvread('study3_descriptives.csv')
table('environments','Training and fixed test suites. Each test suite contains ten environments. The canonical fixed-training setting is given in the text. Unused layout parameters have no effect in a given environment.','tab:env','lXXX',['Parameter','Varied training / T1','T2','T3'],[
['Arena side $L$','40--80','30--40','40--80'],['Food items $F$','20--60','10--20','20--60'],['Layouts','Uniform, clusters, patches','Clusters','Uniform, clusters, patches'],['Cluster count $k$','2--5','2--3','2--5'],['Cluster scale $\\sigma$','3--8','2--4','3--8'],['Patch radius','4--10','Not used','4--10'],['Regrowth probability','0.01--0.05','0.005--0.01','0.01--0.05'],['Food drift','0','0','0.05--0.3'],['Season period (steps)','None','None','100--300']])
table('procedures','Training and transfer procedures. Each row has fixed and varied variants, separately tuned. Body traits and actor architecture are shared.','tab:procedures','lXXX',['Procedure','Training signal','Competence deployment','State at adaptation start'],[
['OE','Energy-dependent survival/reproduction plus floor replacements','16 sampled population members','Genomes/alive mask retained; world state and energy reset'],['ES','Rank-shaped episodic food score','16 copies of mean actor','Mean solution retained; fresh strategy/optimizer'],['PPO','Food reward and clipped policy objective','16 copies of actor','Actor/critic and reward statistics retained; fresh Adam/rollout state']])
table('persistence','Descriptive extinction frequencies in the separate 5,000-step no-floor assay. Each cell pools ten runs across ten fixed environments; these 100 observations are not independent training runs.','tab:persistence','lXXX',['Procedure','T1','T2','T3'],[[N[m]]+[f"{float(next(r for r in desc if r['method']==m and r['set']==st)['extinction_rate']):.2f}" for st in ['T1','T2','T3']] for m in methods])
F=rd('floor_checks.json')['oe_adaptation_floor']
table('floor','Share of new agents supplied by population-floor replacements during adaptation. Fractions pool event counts over ten runs and ten environments per suite.','tab:floor','lXX',['Suite','OE','OE-fixed'],[[st]+[f"{100*F[m+'/'+st]['floor_share']:.2f}\\%" for m in ['oe','oe_fixed']] for st in ['T1','T2','T3']])
# All neural confirmatory contrasts; the p-values test run summaries, not the IQM.
sets={'S1':'T2/T3 AUC','S2':'T2/T3 AUC','S3':'T2/T3 AUC','S4':'T2/T3 zero-shot','S5':'T2/T3 final','S6':'T1 AUC','S7':'T2/T3 extinction','A2':'T2/T3 AUC','A3':'T1 zero-shot'}
rows=[]
for x in C['secondary']+A['family']:
 rows.append([x['name'],N[x['x']]+' minus '+N[x['y']],sets[x['name']],ci(x),pv(x['p_holm'])])
table('neural_tests','Complete core S1--S7 and later A2--A3 families. Differences are pooled IQMs except S7, which uses IQMs of per-run extinction fractions. Intervals are nominal 95\\% whole-run intervals. Holm correction is separate for the two families. The rank-test $p$ values apply to per-run summaries, not directly to the IQM effects.','tab:tests','lp{2.6cm}p{2.2cm}p{4.0cm}r',['ID','Comparison','Outcome','Difference [interval]','Adjusted $p$'],rows,long=True)
table('primary_rules','Interval-rule comparisons. A1 and A2 use the same ES contrast; they are not independent confirmations. The margin is 0.05 on the empirical normalized scale.','tab:primaryrules','lXXX',['ID','Comparison','Difference [interval]','Decision'],[
['H1','OE minus PPO-DR',ci({'iqm_difference':C['h1']['D'],'ci':C['h1']['ci']}),'Against predicted superiority'],['A1','ES-DR minus ES-fixed',ci({'iqm_difference':A['a1']['D'],'ci':A['a1']['ci']}),'Positive, below margin']])
rows=[]
for lab,d in [('H2b',C['h2b']),('A4',A['a4'])]:
 for key,x in d.items():
  if isinstance(x,dict) and 'value'in x:rows.append([lab,esc(key),f"{x['value']:+.6f}",f"[{x['ci'][0]:.6f}, {x['ci'][1]:.6f}]"])
table('four_method','Prespecified four-method summaries. These are contrasts among selected configurations, not identified causal main effects. A4 is a later descriptive comparison whose OE component was already known.','tab:four','llrX',['Analysis','Quantity','Estimate','Nominal 95\\% interval'],rows)
rows=[]
for r in desc:rows.append([N[r['method']],r['set']]+[f(float(r[k])) for k in ['zero_shot_mean','final_mean','auc_mean']])
table('descriptives','All-method descriptive means. These average environments within runs and then runs; they are distinct from pooled IQMs.','tab:descriptives','llrrr',['Procedure','Suite','Zero-shot','Final','AUC'],rows,long=True)
# Selected hyperparameters copied from frozen tuning outputs.
settings=csvread('tuned_settings.csv')
rows=[]
for m in methods:
 vals={r['parameter']:r['value'] for r in settings if r['method']==m}
 labels={'WEIGHT_SD':'Mutation SD','ENT_COEF':'Entropy coefficient','GAMMA':'Discount','LR':'Learning rate','NORMALIZE_REWARD':'Reward normalization','NUM_STEPS':'Rollout length','UPDATE_EPOCHS':'Update epochs','EPISODES':'Episodes per candidate','POP_SIZE':'Population size','SIGMA':'Perturbation SD'}
 def pretty(v):
  if v in ['True','False']:return 'on' if v=='True' else 'off'
  try:return f'{float(v):.6g}'
  except ValueError:return esc(v)
 rows.append([N[m],'; '.join(labels.get(k,esc(k))+' = '+pretty(v) for k,v in vals.items())])
table('settings','Selected settings from the recorded tuning outputs, rounded to six significant digits. Exact values are retained in the accompanying tuning CSV. Fixed and varied configurations were selected separately. Asterisk: later ES-fixed extension.','tab:settings','lp{12cm}',['Procedure','Settings'],rows,long=True)
# Trait complete family.
G={'1':('GS_food_frozen','ET'),'2':('GS_pop_frozen','ET'),'3':('GS_pop_mut','ET'),'4':('SV_frozen','ET'),'5':('ET_freezeB','ET'),'6':('ET_cloneB','ET_freezeB'),'7':('GS_pop_frozen','GS_food_frozen'),'8':('GS_pop_mut','GS_pop_frozen')}
short={'GS_food_frozen':'F-food','GS_pop_frozen':'F-pop','GS_pop_mut':'M-pop','SV_frozen':'SV','ET':'ET','ET_freezeB':'ET-freeze','ET_cloneB':'ET-clone','ORACLE_B':'Oracle-B'}
rows=[];csvrows=[]
for arm in ['main','arm2','arm3','arm4']:
 for k,raw in P[arm]['raw'].items():
  dec=P[arm]['decided'][k];a,b=G[k]
  codes=[]
  for outcome in ['P1','P2']:
   v=dec['verdict'][outcome];codes.append('MD' if v.startswith('meaningfully') else 'EQ' if v.startswith('equivalent') else 'DU' if v.startswith('different') else 'IN')
  effect=lambda o:f"${raw[o]['diff']:+.2f}$ $[{raw[o]['ci95'][0]:.2f},{raw[o]['ci95'][1]:.2f}]$"
  rows.append([arm.replace('arm','A')+'-'+k,short[a]+' / '+short[b],effect('P1'),effect('P2'),pv(dec['p_adj']['P1'])+' / '+pv(dec['p_adj']['P2']),'/'.join(codes)])
  csvrows.append({'arm':arm,'comparison':k,'x':a,'y':b,**{f'{o}_{f}':raw[o][f] for o in ['P1','P2'] for f in ['diff','p']},'P1_CI':raw['P1']['ci95'],'P2_CI':raw['P2']['ci95'],'P1_adjusted_p':dec['p_adj']['P1'],'P2_adjusted_p':dec['p_adj']['P2'],'P1_verdict':dec['verdict']['P1'],'P2_verdict':dec['verdict']['P2'],'P1_boundary_label':dec['p1_label'],'overall':dec['overall_label'],'includes_A_losses':dec['a_phase']})
assert len(rows)==31
table('trait_contrasts','All 31 trait-study comparison pairs; all differences are X minus Y, with comparisons written X / Y. P1 is mean late-B population and P2 is terminal extinction probability. Entries are differences with 95\\% intervals. MD: significant with point estimate at or beyond the margin; EQ: equivalence under the specified interval rule; DU: detected difference with size unresolved; IN: inconclusive. Boundary qualifications are in Table~\\ref{tab:traitgroups}; full labels are retained in the accompanying CSV.','tab:traitcontrasts','lp{2.7cm}p{3cm}p{2.8cm}p{2.5cm}l',['Arm-row','X / Y','P1 difference [CI]','P2 difference [CI]','Adj. $p_1$ / $p_2$','P1/P2'],rows,long=True)
# Correct caption's subtraction direction unambiguously.
p=T/'trait_contrasts.tex';s=p.read_text().replace('the first group is subtracted from the second only in the sense X minus Y shown by the slash (X / Y)','all differences are X minus Y, with comparisons written X / Y');p.write_text(s)
with (D/'trait_complete_contrasts.csv').open('w') as fobj:
 w=csv.DictWriter(fobj,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
rows=[]
for arm,groups in P['U1_U4'].items():
 for g,v in groups.items():rows.append([arm.replace('arm','A'),short[g],f(v['mean_P1'],2),str(v['extinct'])+'/'+str(v['n']),str(v['died_in_A']),f(v['share_P1_ge_cap_minus_2.5'],2),f(v['share_P1_le_2.5'],2)])
table('trait_groups','Trait-study group outcomes and boundary diagnostics. All groups have 200 seeds. High/low are fractions with P1 at least 22.5 / at most 2.5. In arm 2, 25 is not a hard population ceiling. A-dead identifies losses preceding the environmental shift.','tab:traitgroups','llrrrrr',['Arm','Group','Mean P1','Extinct','A-dead','High','Low'],rows,long=True)
print('Generated',len(list(T.glob('*.tex'))),'tables; all31 trait pairs retained.')
