import sys, os, numpy as np, pandas as pd, xarray as xr, warnings
warnings.filterwarnings('ignore')
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,S)
from analyze_point import analyze
R=str(__import__('pathlib').Path(__file__).resolve().parents[2]); sys.path.insert(0,R)
import config as C
pts=pd.read_csv(S+'/points.csv')
pts=pts[[os.path.exists(f'{S}/pts/p{p:03d}.nc') for p in pts.pid]]
# pipeline values at the same cells (consistency check of the 6h reconstruction)
L5=xr.open_zarr(C.compound_zarr(C.HW_BASELINES['1961_1990'],1,C.SPEI_CALIB_PERIODS['1950_2025']))
la=xr.DataArray(pts.lat.values,dims='p'); lo=xr.DataArray(pts.lon.values,dims='p')
l5=L5[['compound','heat','drought']].sel(latitude=la,longitude=lo).load()
L2=xr.open_zarr(C.MONTHLY_ZARR)[['P','pet']].sel(latitude=la,longitude=lo).load()
from collections import defaultdict
ANN=defaultdict(list)
rows=[]
for j,r in enumerate(pts.itertuples()):
    rec,r6=analyze(f'{S}/pts/p{r.pid:03d}.nc')
    rec.update(pid=r.pid,cls=r.cls,lat=r.lat,lon=r.lon,ai=r.ai)
    rec['chk_heat_agree']=float((l5.heat.isel(p=j).values.astype(bool)==r6['heat']).mean())
    rec['chk_drought_agree']=float((l5.drought.isel(p=j).values.astype(bool)==r6['drought']).mean())
    rec['chk_compound_agree']=float((l5.compound.isel(p=j).values.astype(bool)==r6['compound']).mean())
    rec['chk_P_maxrel']=float(np.nanmax(np.abs(L2.P.isel(p=j).values-r6['P']))/max(np.nanmean(r6['P']),1e-6))
    rec['chk_PET_maxrel']=float(np.nanmax(np.abs(L2.pet.isel(p=j).values-r6['PET']))/np.nanmean(r6['PET']))
    a6=rec.pop('_ann6'); a24=rec.pop('_ann24')
    for v in a6: ANN[f'{v}_6h'].append(a6[v]); ANN[f'{v}_24h'].append(a24[v])
    rows.append(rec); print(j,r.pid,flush=True)
pd.DataFrame(rows).to_csv(S+'/validation_points.csv',index=False)
np.savez(S+'/validation_annual.npz',pid=pts.pid.values,**{k:np.array(v) for k,v in ANN.items()})
