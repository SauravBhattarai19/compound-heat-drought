"""Re-run the full per-point pipeline (daily fields -> FAO-56 PET -> SPEI-1 GLO -> T90 heatwaves
-> compound months) from ERA5 hourly data, once with the 4 synoptic hours the paper uses
(00/06/12/18 UTC) and once with all 24 hours, and compare the resulting trends."""
import sys, numpy as np, pandas as pd, xarray as xr
R=str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0,R+'/src'); sys.path.insert(0,R)
import config as C
from chd import pet as petmod, spei as sp, thresholds as th, heatwave as hwm
from scipy.special import ndtri
from scipy.stats import theilslopes
K=273.15

def daily(ds, hours):
    df=ds.to_dataframe()[['t2m','d2m','u10','v10','sp','ssrd','strd','tp']]
    df=df[df.index.hour.isin(hours)]
    g=df.groupby(df.index.normalize())
    ws=np.sqrt(df.u10**2+df.v10**2).groupby(df.index.normalize()).mean()
    td=df.d2m-K; ea=(0.6108*np.exp(17.27*td/(td+237.3))).groupby(df.index.normalize()).mean()
    out=pd.DataFrame({'tasmax':g.t2m.max()-K,'tasmin':g.t2m.min()-K,
        'pr':g.tp.mean()*24*1000,'rs':g.ssrd.mean()*24e-6,'rl':g.strd.mean()*24e-6,
        'u2':ws*C.WIND_10M_TO_2M,'ea':ea,'ps':g.sp.mean()/1000})
    return out

def pipeline(d):
    et0=petmod.et0_fao56(d.tasmax.values,d.tasmin.values,d.rs.values,d.rl.values,d.u2.values,d.ea.values,d.ps.values)
    d=d.assign(pet=et0)
    mo=d[['pr','pet']].resample('MS').sum()
    D=(mo.pr-mo.pet).values; T=len(D); mon=np.arange(T)%12
    spei=np.full(T,np.nan)
    for m in range(12):
        x=D[mon==m]; w0,w1,w2=sp._pwm(np.sort(x)[:,None])
        xi,a,k,l2=sp._glo_params(w0,w1,w2)
        spei[mon==m]=ndtri(np.clip(sp._glo_cdf(x[:,None],xi,a,k),1e-6,1-1e-6))[:,0]
    drought=spei<=C.DROUGHT_THRESHOLD
    times=d.index; doy=times.dayofyear.values.astype(np.int64)
    tx=d.tasmax.values.astype(np.float32)[:,None]
    base=(times.year>=1961)&(times.year<=1990)
    widx,cnt=th._window_indices(doy[base],C.HW_WINDOW_DAYS)
    t90=th._percentile_kernel(np.ascontiguousarray(tx[base]),widx,cnt,float(C.HW_PERCENTILE))
    ym=times.year.values*12+times.month.values-1; uniq=np.unique(ym)
    d2m=np.searchsorted(uniq,ym).astype(np.int64)
    hwd,_=hwm._monthly_hw_kernel(np.ascontiguousarray(tx),np.ascontiguousarray(t90),doy,C.HW_MIN_DURATION,d2m,len(uniq))
    heat=hwd[:,0]>0
    comp=heat&drought
    ny=T//12
    ann={k:v.reshape(ny,12).sum(1).astype(float) for k,v in dict(compound=comp,heat=heat,drought=drought).items()}
    return dict(P=mo.pr.values,PET=mo.pet.values,spei=spei,heat=heat,drought=drought,compound=comp,ann=ann,
                tmax=d.tasmax,t90=t90[:,0])

def ts(y,yrs): return theilslopes(y,yrs)[0]*10

def analyze(path):
    ds=xr.open_dataset(path).load()
    r6=pipeline(daily(ds,[0,6,12,18])); r24=pipeline(daily(ds,list(range(24))))
    yrs=np.arange(1950,2026)
    rec={}
    for v in ('compound','heat','drought'):
        for tag,r in (('6h',r6),('24h',r24)):
            rec[f'{v}_trend_{tag}']=ts(r['ann'][v],yrs)
            rec[f'{v}_trend79_{tag}']=ts(r['ann'][v][29:],yrs[29:])
            rec[f'{v}_mean_{tag}']=r['ann'][v].mean()
        rec[f'{v}_month_agree']=float((r6[v]==r24[v]).mean())
        a,b=r6[v],r24[v]
        rec[f'{v}_jaccard']=float((a&b).sum()/max((a|b).sum(),1))
        rec[f'{v}_ann_r']=float(np.corrcoef(r6['ann'][v],r24['ann'][v])[0,1])
    dt=(r24['tmax']-r6['tmax'])
    ady=dt.groupby(dt.index.year).mean()
    rec['tmax_deficit_mean']=float(dt.mean()); rec['tmax_deficit_trend']=ts(ady.values,ady.index.values)
    a6=r6['tmax'].groupby(r6['tmax'].index.year).mean(); a24=r24['tmax'].groupby(r24['tmax'].index.year).mean()
    rec['tmax_trend_6h']=ts(a6.values,a6.index.values); rec['tmax_trend_24h']=ts(a24.values,a24.index.values)
    Py6=r6['P'].reshape(-1,12).sum(1); Py24=r24['P'].reshape(-1,12).sum(1)
    rec['P_ann_mean_24h']=Py24.mean(); rec['P_ratio_6_24']=Py6.sum()/max(Py24.sum(),1e-9)
    rec['P_mon_r']=float(np.corrcoef(r6['P'],r24['P'])[0,1]) if r24['P'].std()>0 else np.nan
    rec['P_trend_6h']=ts(Py6,yrs); rec['P_trend_24h']=ts(Py24,yrs)
    PETy6=r6['PET'].reshape(-1,12).sum(1); PETy24=r24['PET'].reshape(-1,12).sum(1)
    rec['PET_ratio_6_24']=PETy6.sum()/PETy24.sum(); rec['PET_trend_6h']=ts(PETy6,yrs); rec['PET_trend_24h']=ts(PETy24,yrs)
    rec['spei_mon_r']=float(np.corrcoef(r6['spei'],r24['spei'])[0,1])
    rec['_ann6']=r6['ann']; rec['_ann24']=r24['ann']
    return rec, r6

if __name__=='__main__':
    rec,_=analyze(sys.argv[1]); print(pd.Series(rec).round(4).to_string())
