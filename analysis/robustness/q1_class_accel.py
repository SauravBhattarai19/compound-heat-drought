import sys, numpy as np, pandas as pd
R=str(__import__('pathlib').Path(__file__).resolve().parents[2])
sys.path.insert(0,R+'/src'); sys.path.insert(0,R)
from chd import manuscript_figures as mf, drylands
from statsmodels.regression.linear_model import OLS
ctx=mf._Ctx(); land=ctx.land.values
w2=np.cos(np.deg2rad(ctx.lat))[:,None]*np.ones((1,len(ctx.lon)))
yrs=np.arange(1950,2026).astype(float)
cnt={v:ctx.annual_counts(v).values for v in ('compound','heat','drought')}
E=drylands.AI_CLASS_EDGES; L=drylands.AI_CLASS_LABELS
groups=[(L[i],land&(ctx.ai>=E[i])&(ctx.ai<E[i+1])) for i in range(5)]+[('drylands',land&(ctx.ai<0.65)),('land',land)]
rows=[]; ser={}
for name,m in groups:
    W=w2*m
    for v in cnt:
        s=np.nansum(cnt[v]*W[None],axis=(1,2))/W.sum(); ser[f'{name}|{v}']=s
        k=yrs>=1979; t=yrs[k]-1979; h=np.maximum(yrs[k]-2001.5,0)
        r=OLS(s[k],np.column_stack([np.ones(k.sum()),t,h])).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        rows.append(dict(cls=name,var=v,ts_50_78=mf._ts(s[:29],yrs[:29]),ts_79_01=mf._ts(s[29:52],yrs[29:52]),
            ts_02_25=mf._ts(s[52:],yrs[52:]),ts_79_25=mf._ts(s[29:],yrs[29:]),hinge_extra=r.params[2]*10,hinge_p=r.pvalues[2]))
pd.DataFrame(ser,index=yrs.astype(int)).to_csv(S+'/class_series.csv') if (S:=sys.argv[1]) else None
pd.set_option('display.width',200); print(pd.DataFrame(rows).round(3).to_string())
