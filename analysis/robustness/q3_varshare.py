"""Q3: how much of the month-to-month variability of D=P-PET is carried by PET vs P, by aridity class.
Per pixel and calendar month: r^2(D, PET) and r^2(D, P) over the 76 years; plus SD ratio sd(PET)/sd(P)."""
import sys, numpy as np, pandas as pd, xarray as xr
R=str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0,R+'/src'); sys.path.insert(0,R)
import config as C
from chd import manuscript_figures as mf, drylands
ctx=mf._Ctx(); land=ctx.land.values
mon=xr.open_zarr(C.MONTHLY_ZARR)
P=mon['P'].values.reshape(76,12,721,1440); PET=mon['pet'].values.reshape(76,12,721,1440)
Pa=P-P.mean(0); Ea=PET-PET.mean(0); del P,PET
Da=Pa-Ea
def r2(a,b):
    num=(a*b).sum(0)**2; den=(a*a).sum(0)*(b*b).sum(0)
    with np.errstate(all='ignore'): return num/den
r2pet=np.nanmean(r2(Da,Ea),axis=0); r2p=np.nanmean(r2(Da,Pa),axis=0)   # mean over calendar months
sdr=np.nanmedian(Ea.std(0)/np.where(Pa.std(0)>0,Pa.std(0),np.nan),axis=0)
w2=np.cos(np.deg2rad(ctx.lat))[:,None]*np.ones((1,1440))
E=drylands.AI_CLASS_EDGES; L=drylands.AI_CLASS_LABELS
rows=[]
for i in range(5):
    m=land&(ctx.ai>=E[i])&(ctx.ai<E[i+1])
    W=w2*m
    def wm(x): ok=np.isfinite(x)&m; return float((x*W)[ok].sum()/W[ok].sum())
    def wmed(x):
        ok=np.isfinite(x)&m; v=x[ok]; ww=W[ok]; o=np.argsort(v); c=np.cumsum(ww[o]); return float(v[o][np.searchsorted(c,c[-1]/2)])
    rows.append(dict(cls=L[i],r2_D_PET=wm(r2pet),r2_D_P=wm(r2p),sdPET_over_sdP_median=wmed(sdr)))
df=pd.DataFrame(rows).round(3); df.to_csv(S+'/q3_varshare.csv' if (S:=sys.argv[1]) else 'x.csv',index=False); print(df.to_string())
