"""South Asia (AR6 SAS) population-weighted compound frequency: full-record and 1979-2025 trends.
Reproduces population_regions.csv (SAS: -0.069/dec, HR p=0.029) and adds the satellite-era trend.
Run from the repo root: python analysis/finalization_checks/south_asia_perperson.py"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import config as C, xarray as xr, numpy as np, pymannkendall as mk
ds = xr.open_zarr(C.compound_zarr(C.HW_BASELINES['1961_1990'], 1, C.SPEI_CALIB_PERIOD))
f = xr.open_dataset('figures/manuscript/nature/cache/fields.nc')
sub = dict(latitude=slice(42, 0), longitude=slice(55, 105))
m = ((f.ar6 == 37) & (f.land == 1)).sel(**sub)
pop = f.pop2024.sel(**sub).where(m).fillna(0)
cp = ds['compound'].sel(**sub).astype('float32').groupby('time.year').sum().load()
s = (cp * pop).sum(['latitude', 'longitude']) / pop.sum(); yrs = s.year.values; v = s.values
for a, b in [(1950, 2025), (1979, 2025)]:
    sel = (yrs >= a) & (yrs <= b); r = mk.hamed_rao_modification_test(v[sel])
    print(a, b, 'Theil-Sen/dec %.3f  Hamed-Rao p %.3f' % (r.slope * 10, r.p))
print('means 1950-79 %.2f  1979-95 %.2f  1996-2025 %.2f' % (v[yrs <= 1979].mean(), v[(yrs >= 1979) & (yrs <= 1995)].mean(), v[yrs >= 1996].mean()))
