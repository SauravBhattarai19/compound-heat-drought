"""Q5: is the aridity gradient an artifact of AI and SPEI sharing ERA5 P/PET over the same years?
(a) AI from early-only windows (1950-1979, 1961-1990), which precede most of the trend;
(b) Koppen-Geiger (Beck 2018; station/independent climatologies, T & P thresholds, no PET)."""
import sys, numpy as np, pandas as pd, xarray as xr, time
R=str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0,R+'/src'); sys.path.insert(0,R)
import config as C
from chd import manuscript_figures as mf, drylands
t0=time.time()
ctx=mf._Ctx()
lat,lon=ctx.lat,ctx.lon
mon=xr.open_zarr(C.MONTHLY_ZARR)
ai_alt={}
for a,b in [(1950,1979),(1961,1990),(1996,2025)]:
    sub=mon.sel(time=slice(f'{a}',f'{b}'))
    v=(sub['P'].sum('time')/sub['pet'].sum('time')).compute()
    v=v.reindex(latitude=lat,longitude=lon,method='nearest').values
    ai_alt[f'AI_{a}_{b}']=v; print('ai',a,b,time.time()-t0,flush=True)
ai_alt['AI_1950_2025']=ctx.ai
np.savez(sys.argv[1]+'/ai_alt.npz',**ai_alt)
land=ctx.land.values
w2=np.cos(np.deg2rad(lat))[:,None]*np.ones((1,len(lon)))
yrs=np.arange(1950,2026).astype(float)
cnt={v:ctx.annual_counts(v).values for v in ('compound','heat','drought')}
print('counts',time.time()-t0,flush=True)
def series(var,msk):
    W=w2*msk; A=cnt[var]
    return np.nansum(A*W[None],axis=(1,2))/W.sum()
edges=drylands.AI_CLASS_EDGES; labs=drylands.AI_CLASS_LABELS
rows=[]
def add(scheme,name,msk):
    rec=dict(scheme=scheme,cls=name,land_share=float((w2*msk).sum()/(w2*land).sum()))
    for v in cnt:
        s=series(v,msk); rec[f'{v}_trend']=mf._ts(s,yrs); rec[f'{v}_hrp']=mf._hr_p(s)
    rows.append(rec)
for key,ai in ai_alt.items():
    for i,l in enumerate(labs):
        add(key,l,land&(ai>=edges[i])&(ai<edges[i+1]))
    add(key,'drylands (AI<0.65)',land&(ai<0.65))
    # class agreement with the full-record AI
    c_full=np.digitize(ctx.ai,edges[1:-1]); c_alt=np.digitize(ai,edges[1:-1])
    ok=land&np.isfinite(ai)&np.isfinite(ctx.ai)
    print(key,'same class (area %)',100*(w2*(ok&(c_full==c_alt))).sum()/(w2*ok).sum(),flush=True)
desert=ctx.desert.astype(bool); B=ctx.arid_all.astype(bool)
add('Koppen','BW desert',land&desert)
add('Koppen','BS steppe',land&B&~desert)
add('Koppen','B (all dry)',land&B)
add('Koppen','non-B',land&~B)
df=pd.DataFrame(rows).round(4); df.to_csv(sys.argv[1]+'/q5_ai_robustness.csv',index=False)
pd.set_option('display.width',250); print(df.to_string())
