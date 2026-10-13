"""
L6 — spatiotemporal trends of compound (and marginal) event frequency.

Per grid cell we build an annual (and per-season) time series of event-month counts,
then apply the non-parametric Mann-Kendall test (tie-corrected) with the Theil-Sen
slope estimator. Numba-parallel over pixels. Adapted from
wdxx-precipitation-indices/visualization/trends.py.

Outputs, per (baseline, framing, trend-window):
    slope   Sen's slope in events/decade
    z       Mann-Kendall z statistic
    pvalue  two-sided p-value
    signif  |p| < MK_ALPHA
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io


@njit(cache=True)
def _mk_pixel(x):
    """Mann-Kendall S, tie-corrected variance, and Theil-Sen slope for 1D series x."""
    n = x.shape[0]
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = x[j] - x[i]
            if d > 0:
                S += 1
            elif d < 0:
                S -= 1
    # tie correction
    # count tie groups
    xs = np.sort(x)
    var = n * (n - 1) * (2 * n + 5)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t = j - i
        if t > 1:
            var -= t * (t - 1) * (2 * t + 5)
        i = j
    var = var / 18.0
    if var <= 0:
        z = 0.0
    elif S > 0:
        z = (S - 1) / math.sqrt(var)
    elif S < 0:
        z = (S + 1) / math.sqrt(var)
    else:
        z = 0.0
    # Theil-Sen slope (median of pairwise slopes)
    m = n * (n - 1) // 2
    slopes = np.empty(m, np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes[k] = (x[j] - x[i]) / (j - i)
            k += 1
    slope = np.median(slopes)
    return z, slope


@njit(parallel=True, cache=True)
def _mk_grid(data):
    """data: (n_years, P). Returns z, slope arrays (P,). Skips all-equal/NaN pixels."""
    ny, P = data.shape
    zout = np.full(P, np.nan)
    sout = np.full(P, np.nan)
    for p in prange(P):
        col = data[:, p]
        ok = True
        for t in range(ny):
            if math.isnan(col[t]):
                ok = False
                break
        if ok:
            z, s = _mk_pixel(col)
            zout[p] = z
            sout[p] = s
    return zout, sout


def _p_from_z(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z) / math.sqrt(2))))


def annual_counts(compound: xr.DataArray, window: C.Period, season=None,
                   gs_month_mask: xr.DataArray | None = None) -> xr.DataArray:
    """Per-pixel annual event-month count over `window` (optionally restricted to one
    `season`, or to a per-pixel growing-season month mask `gs_month_mask`), rescaled to
    a full-window-equivalent rate.

    `compound` (and the other SPEI-dependent L5 vars) is float32 with NaN where SPEI
    was undefined and not otherwise decided by heat status (see compound.py). A plain
    `.sum()` would silently treat those NaN months as excluded-from-the-sum-but-still-
    in-the-12/3-month denominator, which under-counts exactly the driest cells worst.
    Instead: sum only the valid months, count how many months were valid, and scale
    the sum up to what a fully-observed window (12 months for "annual", len(season)
    for a season) would show at that same rate. A pixel-year with zero valid months
    (only ~128 pixels globally have literally no valid calendar month at all) is NaN,
    which `_mk_grid` already treats as "exclude this pixel's whole series."

    CAUTION (corrected 2026-09-23): n_valid is NOT in general fixed per pixel. Under the
    Kleene logic in compound.py a month with no heatwave is always defined (0), whereas a
    heatwave month in an SPEI-undefined calendar month is NaN; as heatwaves become more
    frequent, n_valid falls, so 12/n_valid drifts upward over time wherever SPEI has
    undefined calendar months. With the log-logistic LL3 fit this affected ~50% of land
    area. Since the switch to the generalized-logistic parameterization in spei.py, no
    land (or ocean) grid-month has undefined SPEI-1 for the full-record calibration, so
    on land n_valid == 12 in every year and this rescaling is exactly 12/12 = 1. The
    rescaling is kept for robustness (and for other calibrations), not because it is
    currently active.

    `gs_month_mask` (dims `month`=1..12, `latitude`, `longitude`, boolean; from
    `chd.agriculture.growing_season_mask`) generalizes the fixed 4-season `season`
    filter to a *per-pixel* calendar-month mask: off-season months are set NaN (so they
    drop out of both the numerator and the valid-month denominator) and `full_n` becomes
    the per-pixel number of growing months, so each pixel is rescaled to its own
    growing-season length. Non-growing pixels (all-False mask) collapse to NaN and are
    excluded, exactly like a SPEI-void pixel. `gs_month_mask` and `season` are mutually
    exclusive; `gs_month_mask` takes precedence if both are given."""
    da = compound
    if gs_month_mask is not None:
        in_season = gs_month_mask.sel(month=da["time"].dt.month)
        da = da.where(in_season)
        full_n = gs_month_mask.sum("month").astype("float32")
    elif season is not None:
        da = da.sel(time=da["time"].dt.month.isin(C.SEASONS[season]))
        full_n = len(C.SEASONS[season])
    else:
        full_n = 12
    da = da.sel(time=slice(f"{window.start}", f"{window.end}"))
    valid = np.isfinite(da)
    n_valid = valid.groupby("time.year").sum().astype("float32")
    raw_sum = da.groupby("time.year").sum(skipna=True).astype("float32")
    scale = full_n / n_valid.where(n_valid > 0)
    return (raw_sum * scale).astype("float32")


# Backward-compatible alias for any external caller still using the old private name.
_annual_counts = annual_counts


def trends_for(var: xr.DataArray, window: C.Period, season=None,
               gs_month_mask: xr.DataArray | None = None) -> xr.Dataset:
    counts = _annual_counts(var, window, season, gs_month_mask).load()
    years = counts["year"].values
    nlat, nlon = counts.sizes["latitude"], counts.sizes["longitude"]
    arr = counts.values.reshape(len(years), nlat * nlon)
    z, slope = _mk_grid(np.ascontiguousarray(arr.astype(np.float64)))
    z = z.reshape(nlat, nlon)
    slope = slope.reshape(nlat, nlon) * 10.0     # per-year -> per-decade
    p = _p_from_z(z)
    coords = {"latitude": counts["latitude"], "longitude": counts["longitude"]}
    return xr.Dataset({
        "slope": (("latitude", "longitude"), slope.astype("float32")),
        "z": (("latitude", "longitude"), z.astype("float32")),
        "pvalue": (("latitude", "longitude"), p.astype("float32")),
        "signif": (("latitude", "longitude"), (p < C.MK_ALPHA).astype("int8")),
    }, coords=coords)


def build_trends(baseline: C.Period, spei_ts=None, var_name="compound",
                  spei_calib: C.Period = None) -> Path:
    """Compute annual + seasonal trend maps for all configured windows."""
    spei_ts = spei_ts or C.SPEI_COMPOUND_TS
    spei_calib = spei_calib or C.SPEI_CALIB_PERIOD
    comp = xr.open_zarr(C.compound_zarr(baseline, spei_ts, spei_calib))[var_name]
    framings = {"annual": None, **{s: s for s in C.SEASONS}}
    results = {}
    for wname, window in C.TREND_WINDOWS.items():
        for fname, season in framings.items():
            ds = trends_for(comp, window, season)
            results[f"{wname}__{fname}"] = ds
    cases = list(results.keys())
    combined = xr.concat([results[k] for k in cases], dim="case")
    # integer case dim + names in attrs: string coords hit a pandas-3 StringDtype bug in
    # the zarr time-coder. Retrieve names via ds.attrs["case_names"].
    combined = combined.assign_coords(case=np.arange(len(cases), dtype="int32"))
    combined.attrs.update(title="Compound event trends (L6)", variable=var_name,
                          baseline=baseline.slug(), spei_timescale=spei_ts,
                          spei_calibration=spei_calib.slug(),
                          case_names=list(cases))
    out_path = C.trends_zarr(baseline, spei_ts, spei_calib, var_name)
    io.write_zarr(combined, out_path, mode="w")
    return out_path
