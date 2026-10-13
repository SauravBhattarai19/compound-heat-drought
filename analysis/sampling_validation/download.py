import cdsapi, zipfile, glob, os, sys, shutil, pandas as pd
from concurrent.futures import ThreadPoolExecutor
S=os.path.dirname(os.path.abspath(__file__)); os.makedirs(S+'/pts',exist_ok=True)
pts=pd.read_csv(S+'/points.csv')
V=["2m_temperature","2m_dewpoint_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
   "surface_pressure","surface_solar_radiation_downwards","surface_thermal_radiation_downwards","total_precipitation"]
def get(r):
    out=f"{S}/pts/p{int(r.pid):03d}.nc"
    if os.path.exists(out): return
    for attempt in range(4):
        try:
            c=cdsapi.Client(quiet=True,progress=False)
            z=f"{S}/pts/p{int(r.pid):03d}.zip"
            c.retrieve("reanalysis-era5-single-levels-timeseries",{"variable":V,
              "location":{"longitude":float(r.lon),"latitude":float(r.lat)},
              "date":["1950-01-01/2025-12-31"],"data_format":"netcdf"}).download(z)
            d=f"{S}/pts/tmp{int(r.pid):03d}"; zipfile.ZipFile(z).extractall(d)
            f=glob.glob(d+'/*.nc'); assert len(f)==1
            shutil.move(f[0],out); shutil.rmtree(d); os.remove(z)
            print('ok',r.pid,flush=True); return
        except Exception as e:
            print('retry',r.pid,attempt,repr(e)[:200],flush=True)
    print('FAILED',r.pid,flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(get,[r for r in pts.itertuples()]))
print('DONE',flush=True)
