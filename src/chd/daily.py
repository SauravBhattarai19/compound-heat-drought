"""
L1 — build period-INDEPENDENT daily fields from 6-hourly ERA5.

Per calendar day, per grid cell (float32):
    tasmax, tasmin   [degC]        daily max/min of t2m
    pr               [mm/day]      precip (1h-accum samples, see deaccum.py)
    rs, rl           [MJ/m^2/day]  down short/long-wave (1h-accum samples)
    u2               [m/s]         2 m wind speed (10 m -> 2 m FAO factor)
    ea               [kPa]         actual vapor pressure from d2m (Tetens)
    ps               [kPa]         surface pressure

One Zarr store per year. Period-independent -> built once, never recomputed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io
from chd.deaccum import radiation_daily_total, precip_daily_total

TIME = C.TIME_DIM
KELVIN = 273.15
NEEDED = ["t2m", "d2m", "u10", "v10", "sp", "tp", "ssrd", "strd"]


def _ea_kpa(d2m_k: xr.DataArray) -> xr.DataArray:
    """Actual vapor pressure (kPa) from dewpoint (K), FAO-56 Tetens."""
    td = d2m_k - KELVIN
    return 0.6108 * np.exp(17.27 * td / (td + 237.3))


def daily_from_month(ds: xr.Dataset) -> xr.Dataset:
    """Reduce one 6-hourly monthly Dataset to daily fields."""
    t2m = ds["t2m"]
    tasmax = t2m.resample({TIME: "1D"}).max() - KELVIN
    tasmin = t2m.resample({TIME: "1D"}).min() - KELVIN

    pr = precip_daily_total(ds["tp"])
    rs = radiation_daily_total(ds["ssrd"])
    rl = radiation_daily_total(ds["strd"])

    wspd10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    u2 = wspd10.resample({TIME: "1D"}).mean() * C.WIND_10M_TO_2M

    ea = _ea_kpa(ds["d2m"]).resample({TIME: "1D"}).mean()
    ps = (ds["sp"].resample({TIME: "1D"}).mean()) / 1000.0  # Pa -> kPa

    out = xr.Dataset(
        {
            "tasmax": tasmax, "tasmin": tasmin, "pr": pr,
            "rs": rs, "rl": rl, "u2": u2, "ea": ea, "ps": ps,
        }
    )
    # rename daily time coord to a clean 'time'
    out = out.rename({TIME: "time"})
    for v in out.data_vars:
        out[v] = out[v].astype("float32")
    return out


def build_month(year: int, month: int, overwrite: bool = False) -> Path:
    """Build the daily Zarr store for ONE month (~5 GB RAM peak -> high parallelism).

    Month-level granularity keeps per-worker memory small so we can run many workers
    and saturate the USB read bandwidth (L1 is I/O-bound).
    """
    out_path = C.daily_month_zarr(year, month)
    if io.already_done(out_path) and not overwrite:
        return out_path
    f = C.era5_file(year, month)
    if not f.exists():
        raise FileNotFoundError(f)
    with xr.open_dataset(f)[NEEDED] as ds:
        mds = daily_from_month(ds.load())
    mds.attrs.update(
        title="ERA5 daily fields (L1)",
        note="Accumulated tp/ssrd/strd treated as 1h-accum samples; see deaccum.py",
        source=str(C.ERA5_DIR),
    )
    mds = mds.chunk({"time": -1, "latitude": C.TILE_LAT, "longitude": C.TILE_LON})
    io.write_zarr(mds, out_path, mode="w")
    return out_path


def build_year(year: int, overwrite: bool = False) -> list[Path]:
    """Build all 12 monthly stores for a year (sequential; use the driver for parallel)."""
    return [build_month(year, m, overwrite) for m in range(1, 13)]


ATTRS = {
    "tasmax": ("degC", "daily maximum 2m temperature"),
    "tasmin": ("degC", "daily minimum 2m temperature"),
    "pr": ("mm/day", "daily total precipitation (4/24h sampled)"),
    "rs": ("MJ m-2 day-1", "downward shortwave radiation"),
    "rl": ("MJ m-2 day-1", "downward longwave radiation"),
    "u2": ("m/s", "2m wind speed"),
    "ea": ("kPa", "actual vapour pressure"),
    "ps": ("kPa", "surface pressure"),
}
