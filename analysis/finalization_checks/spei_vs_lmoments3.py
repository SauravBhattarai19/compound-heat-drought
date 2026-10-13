"""Compare the pipeline's vectorized GLO (L-moment) SPEI-1 fit with the independent lmoments3 library
on 40 random non-Antarctic land cells x 12 calendar months (480 cell-months). Needs `pip install lmoments3`.
Run from the repo root."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import numpy as np, xarray as xr, config as C
from chd import spei
from lmoments3 import distr
from scipy.stats import norm
ds = xr.open_zarr(C.MONTHLY_ZARR)
f = xr.open_dataset('figures/manuscript/nature/cache/fields.nc')
land = np.argwhere(((f.land == 1) & (f.latitude > -60)).values)
pick = land[np.random.default_rng(1).choice(len(land), 40, replace=False)]
diffs = []
for i, j in pick:
    d = ds['D'].isel(latitude=int(i), longitude=int(j)).load().values.reshape(-1, 12)
    for m in range(12):
        w0, w1, w2 = spei._pwm(np.sort(d[:, m])[:, None]); xi, al, k, _ = spei._glo_params(w0, w1, w2)
        ours = norm.ppf(np.clip(spei._glo_cdf(d[:, m][:, None], xi, al, k).ravel(), 1e-6, 1 - 1e-6))
        p = distr.glo.lmom_fit(d[:, m]); ref = norm.ppf(np.clip(distr.glo.cdf(d[:, m], **p), 1e-6, 1 - 1e-6))
        diffs.append(np.nanmax(np.abs(ours - ref)))
print('cell-months', len(diffs), 'max |SPEI diff| %.2e  median %.2e' % (max(diffs), np.median(diffs)))
