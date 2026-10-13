import numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import theilslopes
S=os.path.dirname(os.path.abspath(__file__))
OUT=S+'/sampling_validation.png'
d=pd.read_csv(S+'/validation_points.csv'); A=np.load(S+'/validation_annual.npz')
pts=pd.read_csv(S+'/points.csv').set_index('pid').loc[A['pid']]
order=['hyper-arid','arid','semi-arid','dry sub-humid','humid']
col=dict(zip(order,['#0d366b','#1c5cab','#2a78d6','#5598e7','#86b6ef']))
INK='#2b2b2b'; MUTED='#6b6b6b'; GRID='#e6e5e1'
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,4,figsize=(13,3.5),gridspec_kw=dict(width_ratios=[1,1,1,1.5]))
titles={'compound':'Compound months','heat':'Heatwave months','drought':'Drought months'}
for ax,(v,lab),L in zip(axs[:3],titles.items(),'abc'):
    x=d[f'{v}_trend_24h']; y=d[f'{v}_trend_6h']
    lo=min(x.min(),y.min())-0.05; hi=max(x.max(),y.max())+0.05
    ax.plot([lo,hi],[lo,hi],color=MUTED,lw=1,ls='--',zorder=1)
    for c in order:
        m=d.cls==c; ax.scatter(x[m],y[m],s=22,color=col[c],edgecolor='white',linewidth=0.8,zorder=2,label=c)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal'); ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
    r=np.corrcoef(x,y)[0,1]
    ax.text(0.04,0.96,f'r = {r:.2f}',transform=ax.transAxes,va='top',color=INK)
    ax.set_title(f'({L}) {lab}',loc='left',color=INK,fontsize=10)
    ax.set_xlabel('Trend, all 24 hours (months decade$^{-1}$)'); 
axs[0].set_ylabel('Trend, 00/06/12/18 UTC only\n(months decade$^{-1}$)')
axs[0].legend(frameon=False,fontsize=7.5,loc='upper left',bbox_to_anchor=(0.0,0.9),handletextpad=0.2,title='Aridity class',title_fontsize=7.5)
ax=axs[3]; yrs=np.arange(1950,2026); w=np.cos(np.deg2rad(pts.lat.values))
for tag,c,lab in [('24h','#2a78d6','All 24 hours'),('6h','#eb6834','00/06/12/18 UTC (this study)')]:
    s=(A[f'compound_{tag}']*w[:,None]).sum(0)/w.sum()
    sl,ic,_,_=theilslopes(s,yrs)
    ax.plot(yrs,s,color=c,lw=2,label=f'{lab}: Theil–Sen {sl*10:+.3f} decade$^{{-1}}$')
    ax.plot(yrs,ic+sl*yrs,color=c,lw=1,ls='--')
ax.set_xlim(1950,2025); ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
ax.set_ylabel('Compound months yr$^{-1}$'); ax.set_title('(d) Mean of the 120 cells',loc='left',color=INK,fontsize=10)
ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(OUT,dpi=250); print(OUT)
