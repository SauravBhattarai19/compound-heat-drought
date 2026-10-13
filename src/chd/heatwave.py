"""
L4b — heatwave detection and monthly metrics (baseline-dependent).

Hot day: tasmax > T90(doy). Heatwave: a run of >= HW_MIN_DURATION consecutive hot
days. We read tasmax sequentially into RAM (HDD-friendly; see thresholds.py) and run a
single numba-parallel pass over all grid cells that does run-detection AND monthly
attribution at once -- so runs crossing month/year boundaries are handled naturally and
no 100+ GB intermediate hot/flag arrays are materialised.

Monthly metrics per grid cell:
    hw_days    number of heatwave days in the month
    hw_events  number of heatwaves that START in the month
    hw_present 1 if any heatwave day occurs in the month, else 0
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io, thresholds


@njit(parallel=True, cache=True)
def _monthly_hw_kernel(tx, t90, doy, min_dur, day_to_mon, n_months):
    """One pass over pixels: tx>(t90 by doy) runs>=min_dur, attributed to months.

    tx:(T,P) f32; t90:(366,P) f32; doy:(T,) int; day_to_mon:(T,) int -> two (n_months,P).
    """
    T, P = tx.shape
    hw_days = np.zeros((n_months, P), np.int16)
    hw_events = np.zeros((n_months, P), np.int16)
    for p in prange(P):
        t = 0
        while t < T:
            if tx[t, p] > t90[doy[t] - 1, p]:
                j = t
                while j < T and tx[j, p] > t90[doy[j] - 1, p]:
                    j += 1
                if (j - t) >= min_dur:
                    for k in range(t, j):
                        hw_days[day_to_mon[k], p] += 1
                    hw_events[day_to_mon[t], p] += 1
                t = j
            else:
                t += 1
    return hw_days, hw_events


def monthly_metrics_for_baseline(baseline: C.Period) -> Path:
    """Compute monthly heatwave metrics for the full record under one baseline."""
    t90 = xr.open_zarr(C.HW_DIR / baseline.slug() / "t90.zarr")["t90"].values  # (366,lat,lon)

    years = list(range(C.YEAR_START, C.YEAR_END + 1))
    tx, doy, times = thresholds.load_tasmax_sequential(years)   # (T, lat, lon), doy, times
    T, nlat, nlon = tx.shape

    # absolute month index per day + month-start coord
    ymi = pd.DatetimeIndex(times)
    ym = ymi.year.values.astype(np.int64) * 12 + (ymi.month.values.astype(np.int64) - 1)
    uniq = np.unique(ym)
    n_months = len(uniq)
    day_to_mon = np.searchsorted(uniq, ym).astype(np.int64)
    time_coord = pd.to_datetime(
        [f"{int(u // 12):04d}-{int(u % 12) + 1:02d}-01" for u in uniq]).values

    txf = np.ascontiguousarray(tx.reshape(T, nlat * nlon))
    t90f = np.ascontiguousarray(t90.reshape(t90.shape[0], nlat * nlon))
    hw_days, hw_events = _monthly_hw_kernel(
        txf, t90f, doy, C.HW_MIN_DURATION, day_to_mon, n_months)
    hw_days = hw_days.reshape(n_months, nlat, nlon)
    hw_events = hw_events.reshape(n_months, nlat, nlon)

    ref = xr.open_zarr(C.daily_month_zarr(years[0], 1))
    coords = {"time": time_coord, "latitude": ref["latitude"], "longitude": ref["longitude"]}
    out = xr.Dataset(
        {
            "hw_days": (("time", "latitude", "longitude"), hw_days),
            "hw_events": (("time", "latitude", "longitude"), hw_events),
            "hw_present": (("time", "latitude", "longitude"), (hw_days > 0).astype(np.int8)),
        },
        coords=coords,
    )
    out.attrs.update(title="Monthly heatwave metrics (L4)", baseline=baseline.slug())
    out_path = C.HW_DIR / baseline.slug() / "hw_monthly.zarr"
    io.write_zarr(out.chunk({"time": -1, "latitude": C.TILE_LAT, "longitude": C.TILE_LON}),
                  out_path, mode="w")
    return out_path
