import sys, numpy as np, pandas as pd, xarray as xr
R=str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0,R+'/src'); sys.path.insert(0,R)
from chd import drylands, viz
ai=drylands.aridity_index()
land=viz.land_mask(ai) & (ai.latitude>-60)
edges=drylands.AI_CLASS_EDGES; labels=drylands.AI_CLASS_LABELS
rng=np.random.default_rng(20261008)
lat2,lon2=xr.broadcast(ai.latitude,ai.longitude)
rows=[]
for i,lab in enumerate(labels):
    m=(land & (ai>=edges[i]) & (ai<edges[i+1])).values
    idx=np.flatnonzero(m)
    w=np.cos(np.deg2rad(lat2.values.ravel()[idx])); w/=w.sum()
    pick=rng.choice(idx,size=24,replace=False,p=w)
    for k in pick:
        rows.append(dict(cls=lab,lat=float(lat2.values.ravel()[k]),lon=float(lon2.values.ravel()[k]),ai=float(ai.values.ravel()[k])))
df=pd.DataFrame(rows); df['pid']=range(len(df))
print(df.groupby('cls').size()); print(df.lon.min(),df.lon.max())
df.to_csv(sys.argv[1],index=False)
