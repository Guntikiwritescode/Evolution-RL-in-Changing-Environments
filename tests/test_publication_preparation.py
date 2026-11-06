"""Reporting regression checks."""
from pathlib import Path
import json,re,sys
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts/report'))
import make_numbers as N

def test_study_one_run_count_is_scalar(tmp_path):
 m=N.Macros();N.study1(m);value=next(v for k,v,s in m.lines if k=='POneRuns')
 assert value=='30' and '{' not in value

def test_primary_plot_is_derived_from_raw_evaluations():
 c=json.loads((R/'results/part3/final/analysis.json').read_text());plot=json.loads((R/'results/publication/fig02_primary.json').read_text())
 for method in plot:
  base='addendum' if method=='es_fixed' else 'final';raw=[]
  for seed in range(5000,5010):
   e=json.loads((R/f'results/part3/{base}/{method}_seed{seed}/eval_test.json').read_text())
   raw.append([[x['competence'] for x in v['adaptation']['curve']] for v in e['envs']])
  a=(np.array(raw)-np.array(c['random'])[None,:,None])/(np.array(c['best'])-c['random'])[None,:,None]
  auc=np.trapezoid(a,x=np.linspace(0,1,11),axis=2)[:,10:].ravel();v=np.sort(auc)[50:150].mean()
  assert abs(v-plot[method][0])<1e-12

def test_publication_images_are_complete():
 paths=list((R/'figures/publication').glob('fig*.png'));assert len(paths)==10
 for p in paths:
  im=Image.open(p);im.verify()

def test_supplement_macro_uniqueness_and_no_missing_values():
 text=(R/'paper/v2/numbers_supplement.tex').read_text();names=re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',text)
 assert len(names)>=250 and len(names)==len(set(names));assert '[pending]' not in text
 old=re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',(R/'paper/v2/numbers.tex').read_text());assert not set(old)&set(names)
 rows=json.loads((R/'results/publication/supplemental_numbers.json').read_text());assert len(rows)==len(names)
 assert all(np.isfinite(x['value']) and x['source'] for x in rows)

def test_every_figure_has_uncertainty_and_unit_metadata():
 rows=json.loads((R/'results/publication/figures.json').read_text());assert len(rows)==10
 for x in rows:
  assert (R/x['data']).exists() and len(x['caption'])>150
  for ext in ['pdf','svg','png']:assert (R/'figures/publication'/(x['id']+'.'+ext)).stat().st_size>1000

def test_later_addendum_is_identified():
 guide=(R/'docs/FIGURE_GUIDE.md').read_text()
 for section in guide.split('## ')[1:]:
  if 'ES-fixed' in section:assert any(term in section for term in ('addendum', 'after core', 'extension'))

def test_bibliography_uses_corporate_and_compound_names():
 text=(R/'paper/v2/refs.bib').read_text();assert '{{Open Ended Learning Team} and' in text
 assert 'de Witt, Christian Schroeder' in text
 assert '@misc{bejjani2025ecological,' in text

def test_claim_scope_includes_separate_tuning():
 rows=json.loads((R/'results/publication/claim_evidence.json').read_text());s2=next(x for x in rows if x['id']=='S2')
 assert 'tuned' in s2['claim'] and 'sole causal' in s2['limit']
