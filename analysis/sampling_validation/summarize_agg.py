import numpy as np, pandas as pd, os, sys
from scipy.stats import theilslopes
import pymannkendall as pmk
S=os.path.dirname(os.path.abspath(__file__))
A=np.load(S+'/validation_annual.npz'); pts=pd.read_csv(S+'/points.csv').set_index('pid').loc[A['pid']]
yrs=np.arange(1950,2026); w=np.cos(np.deg2rad(pts.lat.values))
order=['hyper-arid','arid','semi-arid','dry sub-humid','humid','all']
rows=[]
rng=np.random.default_rng(1)
for c in order:
    m=np.ones(len(pts),bool) if c=='all' else (pts.cls.values==c)
    rec={'cls':c}
    for v in ['compound','heat','drought']:
        for tag in ['6h','24h']:
            s=(A[f'{v}_{tag}'][m]*w[m,None]).sum(0)/w[m].sum()
            rec[f'{v}_{tag}']=theilslopes(s,yrs)[0]*10
            rec[f'{v}79_{tag}']=theilslopes(s[29:],yrs[29:])[0]*10
            rec[f'{v}_hrp_{tag}']=pmk.hamed_rao_modification_test(s).p
        # bootstrap over points of the ratio 6h/24h (full record)
        if c=='all':
            idx=np.flatnonzero(m); R=[]
            for b in range(2000):
                k=rng.choice(idx,len(idx)); s6=(A[f'{v}_6h'][k]*w[k,None]).sum(0)/w[k].sum(); s24=(A[f'{v}_24h'][k]*w[k,None]).sum(0)/w[k].sum()
                R.append(theilslopes(s6,yrs)[0]-theilslopes(s24,yrs)[0])
            rec[f'{v}_diff_ci']=np.percentile(np.array(R)*10,[2.5,97.5]).round(3).tolist()
    rows.append(rec)
t=pd.DataFrame(rows); pd.set_option('display.width',250); print(t.round(3).T.to_string()); t.to_csv(S+'/validation_agg.csv',index=False)
