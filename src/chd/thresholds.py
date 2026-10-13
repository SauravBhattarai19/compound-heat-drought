"""
L4a — daily T90 climatology for heatwave detection (baseline-dependent).

T90[doy, lat, lon] = 90th percentile of daily tasmax over the baseline years, using a
+/- HW_WINDOW_DAYS moving window centred on each day-of-year (~31*n_years samples).

HDD-friendly design: the per-month L1 stores make "full time series for a spatial tile"
a scattered read that thrashes a spinning disk. Instead we read tasmax from each month
store SEQUENTIALLY (localized, fast) into RAM, then compute all 366 day-of-year
percentiles with a numba-parallel kernel across the whole grid at once (all cores,
no disk I/O). One store per baseline.
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
from chd import io

N_DOY = 366


def load_tasmax_sequential(years):
    """Read tasmax (only) from each month-store in order -> (T, lat, lon) float32 + doy.

    Sequential single-reader access keeps the HDD near its ~175 MB/s streaming rate. The
    destination array is preallocated (no 2x concat peak — matters for the 115 GB full
    record).
    """
    stores = [C.daily_month_zarr(y, m) for y in years for m in range(1, 13)]
    for p in stores:
        if not io.already_done(p):
            raise FileNotFoundError(p)
    metas = [xr.open_zarr(p) for p in stores]
    lens = [z.sizes["time"] for z in metas]
    Ttot = int(sum(lens))
    nlat = metas[0].sizes["latitude"]
    nlon = metas[0].sizes["longitude"]
    tx = np.empty((Ttot, nlat, nlon), np.float32)
    times = np.empty(Ttot, dtype="datetime64[ns]")
    off = 0
    for z, n in zip(metas, lens):
        tx[off:off + n] = z["tasmax"].values
        times[off:off + n] = z["time"].values
        off += n
    doy = pd.DatetimeIndex(times).dayofyear.values.astype(np.int64)
    return tx, doy, times


def load_tasmax_sequential_cached(years, cache_dir: Path = None):
    """Like load_tasmax_sequential, but caches the result to local NVMe scratch
    (C.TASMAX_CACHE_DIR) keyed by the requested year range. tasmax does not depend on
    the heatwave baseline, so callers that read it once per baseline (events.py) only
    pay the slow HDD read once total instead of once per baseline."""
    cache_dir = Path(cache_dir) if cache_dir is not None else C.TASMAX_CACHE_DIR
    key = f"{years[0]}_{years[-1]}"
    d = cache_dir / key
    tx_p, doy_p, times_p = d / "tx.npy", d / "doy.npy", d / "times_ns.npy"
    if tx_p.exists() and doy_p.exists() and times_p.exists():
        tx = np.load(tx_p)
        doy = np.load(doy_p)
        times = np.load(times_p).astype("datetime64[ns]")
        return tx, doy, times

    tx, doy, times = load_tasmax_sequential(years)
    d.mkdir(parents=True, exist_ok=True)
    np.save(tx_p, tx)
    np.save(doy_p, doy)
    np.save(times_p, times.astype("int64"))
    return tx, doy, times


@njit(parallel=True, cache=True)
def _percentile_kernel(tx, widx, wcount, pct):
    """tx:(T,P) f32; widx:(NDOY,W) int32 time-indices per doy; wcount:(NDOY,) -> (NDOY,P)."""
    T, P = tx.shape
    ndoy = widx.shape[0]
    out = np.empty((ndoy, P), np.float32)
    q = pct / 100.0
    for d in range(ndoy):
        n = wcount[d]
        rank = q * (n - 1)
        lo = int(rank)
        frac = rank - lo
        for p in prange(P):
            buf = np.empty(n, np.float32)
            for k in range(n):
                buf[k] = tx[widx[d, k], p]
            buf.sort()
            if lo + 1 < n:
                out[d, p] = buf[lo] * (1.0 - frac) + buf[lo + 1] * frac
            else:
                out[d, p] = buf[lo]
    return out


def _window_indices(doy, window):
    """For each day-of-year 1..366 build the array of time-indices within +/- window."""
    by_doy = {d: np.where(doy == d)[0] for d in range(1, N_DOY + 1)}
    counts = np.zeros(N_DOY, np.int64)
    lists = []
    for d in range(1, N_DOY + 1):
        offs = ((np.arange(-window, window + 1) + d - 1) % N_DOY) + 1
        idx = np.concatenate([by_doy[o] for o in offs]) if len(offs) else np.array([], int)
        lists.append(idx)
        counts[d - 1] = idx.size
    W = int(counts.max())
    widx = np.zeros((N_DOY, W), np.int64)
    for i, idx in enumerate(lists):
        widx[i, :idx.size] = idx
    return widx, counts


def t90_for_baseline(baseline: C.Period, percentile=None, window=None) -> Path:
    """Compute and store the T90 day-of-year climatology for one baseline."""
    percentile = percentile or C.HW_PERCENTILE
    window = window or C.HW_WINDOW_DAYS
    years = list(range(baseline.start, baseline.end + 1))

    tx, doy, _times = load_tasmax_sequential(years)     # (T, lat, lon)
    T, nlat, nlon = tx.shape
    widx, counts = _window_indices(doy, window)
    txf = tx.reshape(T, nlat * nlon)
    out = _percentile_kernel(np.ascontiguousarray(txf), widx, counts, float(percentile))
    out = out.reshape(N_DOY, nlat, nlon).astype("float32")

    ref = xr.open_zarr(C.daily_month_zarr(baseline.start, 1))
    da = xr.DataArray(
        out, dims=["dayofyear", "latitude", "longitude"],
        coords={"dayofyear": np.arange(1, N_DOY + 1),
                "latitude": ref["latitude"], "longitude": ref["longitude"]},
        name="t90",
    )
    da.attrs.update(units="degC", long_name=f"T{percentile} tasmax climatology",
                    baseline=baseline.slug(), window_days=window)
    out_path = C.HW_DIR / baseline.slug() / "t90.zarr"
    io.write_zarr(da.to_dataset().chunk({"dayofyear": -1, "latitude": C.TILE_LAT,
                                         "longitude": C.TILE_LON}), out_path, mode="w")
    return out_path
