"""Shared, colorblind-friendly publication style; fixed method colors and line encodings."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS=['oe','oe_fixed','ppo_dr','ppo_fixed','es_dr','es_fixed']
NAMES={'oe':'OE','oe_fixed':'OE-fixed','ppo_dr':'PPO-DR','ppo_fixed':'PPO-fixed','es_dr':'ES-DR','es_fixed':'ES-fixed*'}
COLORS=dict(zip(METHODS,['#D55E00','#E69F00','#0072B2','#56B4E9','#009E73','#666666']))
STYLES={m:('--' if m.endswith('fixed') else '-') for m in METHODS}
def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'evolving-ai-publication-20261105','axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.alpha':.2,'savefig.dpi':300})
