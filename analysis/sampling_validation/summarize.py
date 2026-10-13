import pandas as pd, numpy as np, os
from scipy import stats
S=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(S+'/validation_points.csv'); pd.set_option('display.width',250)
print('n',len(d)); print('consistency', d.filter(like='chk').agg(['min','max']).round(6).to_string())
order=['hyper-arid','arid','semi-arid','dry sub-humid','humid']
d['w']=np.cos(np.deg2rad(d.lat))
rows=[]
for c in order+['all']:
    s=d if c=='all' else d[d.cls==c]
    r={'cls':c,'n':len(s)}
    for v in ['compound','heat','drought']:
        for per in ['trend','trend79']:
            r[f'{v}_{per}_6h']=s[f'{v}_{per}_6h'].mean(); r[f'{v}_{per}_24h']=s[f'{v}_{per}_24h'].mean()
        r[f'{v}_agree']=s[f'{v}_month_agree'].mean()
    r['tmax_def']=s.tmax_deficit_mean.mean(); r['tmax_def_trend']=s.tmax_deficit_trend.mean()
    r['P_ratio']=s.P_ratio_6_24.median(); r['PET_ratio']=s.PET_ratio_6_24.median(); r['spei_r']=s.spei_mon_r.mean()
    rows.append(r)
t=pd.DataFrame(rows); print(t.round(3).T.to_string())
t.to_csv(S+'/validation_by_class.csv',index=False)
print()
for v in ['compound','heat','drought']:
    for per in ['trend','trend79']:
        a=d[f'{v}_{per}_6h']; b=d[f'{v}_{per}_24h']; diff=a-b
        w=stats.wilcoxon(a,b)
        print(f"{v:9s} {per:8s} mean6={a.mean():.3f} mean24={b.mean():.3f} ratio={a.mean()/b.mean():.3f} meddiff={diff.median():+.3f} MAD={diff.abs().mean():.3f} r_across_pts={np.corrcoef(a,b)[0,1]:.3f} signagree={(np.sign(a)==np.sign(b)).mean():.2f} wilcoxon p={w.pvalue:.3g}")
print('tmax deficit mean (C)',d.tmax_deficit_mean.describe().round(3).to_dict())
print('tmax deficit trend C/dec',d.tmax_deficit_trend.describe().round(4).to_dict())
print('tmax trend 6h vs 24h',d.tmax_trend_6h.mean().round(3),d.tmax_trend_24h.mean().round(3))
print('P ratio',d.P_ratio_6_24.describe().round(3).to_dict()); print('P mon r',d.P_mon_r.describe().round(3).to_dict())
print('P trend rel diff median',((d.P_trend_6h-d.P_trend_24h)/d.P_ann_mean_24h*100).median())
print('PET ratio',d.PET_ratio_6_24.describe().round(3).to_dict()); print('PET trend 6h/24h',d.PET_trend_6h.mean()/d.PET_trend_24h.mean())
print('spei r',d.spei_mon_r.describe().round(3).to_dict())
# does the aridity gradient survive? hyper-arid / humid ratio of compound trend
for tag in ['6h','24h']:
    h=d[d.cls=='hyper-arid'][f'compound_trend_{tag}'].mean(); u=d[d.cls=='humid'][f'compound_trend_{tag}'].mean()
    hd=d[d.cls=='hyper-arid'][f'drought_trend_{tag}'].mean(); ud=d[d.cls=='humid'][f'drought_trend_{tag}'].mean()
    print(tag,'compound hyper/humid',round(h,3),round(u,3),round(h/u,2),'| drought',round(hd,3),round(ud,3))
