"""
L2 — monthly water balance from L1 daily fields. Period-INDEPENDENT.

  PET_month = sum of daily FAO-56 ET0        [mm/month]
  P_month   = sum of daily precip            [mm/month]
  D_month   = P - PET                        [mm/month]

Output: single Zarr with dims (time=month-start, latitude, longitude).

HDD-friendly design: measured directly against /proc/diskstats, a single sequential
reader sustains ~120 MB/s on this disk; 6 concurrent readers (even ones that are each
individually sequential within their own year) collapse aggregate throughput to
~38 MB/s from head-seek contention -- no better than the original 173-thread thrash.
So this module reads with ONE process, never many.

We also process strictly month-by-month (compute PET, resample to monthly, keep only
the small monthly result, discard the raw daily arrays) rather than concatenating 12
loaded months into a full year first. Concatenating first was measured at ~49 GB peak
RSS for a single year (12 loaded months + the new concatenated copy, plus ~10 transient
full-size intermediates inside the PM equation) -- pure waste, since PET only ever needs
one month of daily fields in RAM at a time.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io, pet


def open_daily(years=None, chunks=None) -> xr.Dataset:
    """Open the full L1 daily record lazily (concatenated per-year Zarr stores)."""
    if years is None:
        years = range(C.YEAR_START, C.YEAR_END + 1)
    paths = [C.daily_month_zarr(y, m) for y in years for m in range(1, 13)
             if io.already_done(C.daily_month_zarr(y, m))]
    if not paths:
        raise FileNotFoundError("No L1 daily Zarr stores found; run 01_build_daily.py first.")
    # IMPORTANT: do NOT force a single time-chunk across the many per-month stores —
    # that builds an enormous rechunk graph and is ~1000x slower. Use native zarr
    # chunks (one chunk per month); callers that need contiguous time load .values.
    ds = xr.open_mfdataset(
        paths, engine="zarr", combine="nested", concat_dim="time", chunks=chunks or {},
    )
    return ds.sortby("time")


def _month_pet_p(year: int, month: int):
    """One month of L1 daily fields -> (month_start, pet_total, p_total). Frees the
    daily arrays before returning; caller never holds more than one month at a time."""
    p = C.daily_month_zarr(year, month)
    if not io.already_done(p):
        raise FileNotFoundError(p)
    ds = xr.open_zarr(p).load()
    et0 = pet.et0_from_daily(ds)                         # mm/day
    pet_tot = et0.sum(dim="time").values.astype("float32")
    p_tot = ds["pr"].sum(dim="time").values.astype("float32")
    tstart = np.datetime64(f"{year:04d}-{month:02d}-01")
    return tstart, pet_tot, p_tot


def build_monthly(years=None, workers: int = None) -> Path:
    """Compute monthly PET, P, D month-by-month (single sequential reader) and write
    one L2 Zarr. `workers` is accepted for signature compatibility but ignored — see
    module docstring: concurrent readers on this HDD are strictly worse."""
    if years is None:
        years = list(range(C.YEAR_START, C.YEAR_END + 1))
    years = list(years)

    ref = xr.open_zarr(C.daily_month_zarr(years[0], 1))
    nlat, nlon = ref.sizes["latitude"], ref.sizes["longitude"]
    lat, lon = ref["latitude"], ref["longitude"]

    months = [(y, m) for y in years for m in range(1, 13)
              if io.already_done(C.daily_month_zarr(y, m))]
    n = len(months)
    times = np.empty(n, dtype="datetime64[ns]")
    petA = np.empty((n, nlat, nlon), np.float32)
    pA = np.empty((n, nlat, nlon), np.float32)

    t0 = time.time()
    for i, (y, m) in enumerate(months):
        tstart, pet_tot, p_tot = _month_pet_p(y, m)
        times[i] = tstart
        petA[i] = pet_tot
        pA[i] = p_tot
        if (i + 1) % 24 == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            print(f"L2: {i + 1}/{n} months done ({rate:.2f} months/s)", flush=True)

    order = np.argsort(times)
    times, petA, pA = times[order], petA[order], pA[order]
    dA = pA - petA

    coords = {"time": times, "latitude": lat, "longitude": lon}
    out = xr.Dataset(
        {"pet": (("time", "latitude", "longitude"), petA),
         "P": (("time", "latitude", "longitude"), pA),
         "D": (("time", "latitude", "longitude"), dA)},
        coords=coords,
    )
    out["pet"].attrs.update(units="mm/month", long_name="reference ET (FAO-56 PM)")
    out["P"].attrs.update(units="mm/month", long_name="total precipitation")
    out["D"].attrs.update(units="mm/month", long_name="climatic water balance P-PET")
    out.attrs.update(title="Monthly water balance (L2)")

    io.write_zarr(out.chunk({"time": -1, "latitude": C.TILE_LAT, "longitude": C.TILE_LON}),
                  C.MONTHLY_ZARR, mode="w")
    return C.MONTHLY_ZARR
