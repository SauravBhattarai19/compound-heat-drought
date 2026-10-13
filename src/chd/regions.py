"""
AR6 land-region aggregation helpers, used by the sensitivity figure battery (S8/S9/S11c
in `scripts/09_sensitivity_figures.py`/`10_event_sensitivity_figures.py`) to summarize
per-pixel results at a standard ~46-region climate-region granularity instead of only
eyeballing global maps -- some regions are robust to the heatwave-baseline/SPEI-
calibration choice, some are not, and a per-pixel map alone doesn't make that
comparison legible.

Uses `regionmask.defined_regions.ar6.land` (IPCC AR6 WGI reference regions, land-only
subset) -- an existing `regionmask>=0.13` dependency, no new package required.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import viz

_AR6 = None


def _ar6():
    global _AR6
    if _AR6 is None:
        _AR6 = regionmask.defined_regions.ar6.land
    return _AR6


def ar6_land_regions(like: xr.DataArray) -> xr.DataArray:
    """Integer AR6 land-region index on `like`'s grid (NaN where no AR6 land region
    applies -- ocean or excluded areas). Region index matches `ar6_region_table()`'s
    `region_index` column (regionmask's own 0-based `.numbers`, not a remapped index).
    Mirrors `viz.land_mask()`'s call pattern (`.mask(like["longitude"], like["latitude"])`
    with named coordinate DataArrays, which is what makes regionmask return dims named
    `latitude`/`longitude` instead of its own default `lat`/`lon` -- verified: passing
    plain unnamed arrays here would silently produce a result that doesn't align with
    the rest of the pipeline's `latitude`/`longitude` dim convention)."""
    r = _ar6()
    return r.mask(like["longitude"], like["latitude"])


def ar6_region_table() -> pd.DataFrame:
    """region_index -> (abbrev, name) for all 46 AR6 land regions, for use as row/axis
    labels instead of raw integer indices."""
    r = _ar6()
    return pd.DataFrame({
        "region_index": list(r.numbers),
        "abbrev": list(r.abbrevs),
        "name": list(r.names),
    })


def area_weighted_regional_stat(da: xr.DataArray, region_da: xr.DataArray, region_idx: int,
                                 land_only: bool = True, reducer: str = "mean") -> float:
    """Cos-latitude-weighted reduction of a single 2D (latitude, longitude) field over
    one AR6 region. `da.where(mask)` sets non-region/non-land cells to NaN BEFORE the
    weighted reduction, and xarray's `.weighted(...).mean/sum(skipna=True)` (the
    default) excludes NaN cells from the weight normalization automatically -- so a
    region's surviving cells are never diluted by latitudes outside the region. Mirrors
    `chd.viz.timeseries()`'s own `cos(lat)` weighting convention (a relative-area
    weighting, not L7's absolute km^2 convention -- deliberate, kept consistent with the
    rest of this module's figures, not an oversight)."""
    mask = (region_da == region_idx)
    if land_only:
        mask = mask & viz.land_mask(da)
    sub = da.where(mask)
    w = np.cos(np.deg2rad(da["latitude"]))
    if reducer == "mean":
        return float(sub.weighted(w).mean(("latitude", "longitude")).values)
    elif reducer == "sum":
        return float(sub.weighted(w).sum(("latitude", "longitude")).values)
    raise ValueError(f"unknown reducer {reducer!r}")


def regional_annual_series(da_timeseries: xr.DataArray, region_da: xr.DataArray, region_idx: int,
                            land_only: bool = True) -> xr.DataArray:
    """Region-restricted version of the annual-sum-then-weighted-mean logic inside
    `chd.viz.timeseries()`: group a (time, latitude, longitude) DataArray by year, sum
    within each year, then take the region's cos-lat-weighted mean per year. Used by
    the S6/S9 multi-line time series figures so both the global (`viz.timeseries`) and
    regional reductions share the same weighting convention.

    Masking BEFORE the `.groupby("time.year").sum()` would be wrong: `.where(mask)`
    turns out-of-region cells to NaN for every month, and `.sum(skipna=True)` (the
    default) collapses an all-NaN-across-time slice to a literal 0, not NaN -- so
    every out-of-region pixel would silently re-enter the subsequent weighted mean as
    a real zero-valued data point, diluting the region's true mean by the region's
    area fraction of the masked domain. Sum first (unmasked, so every pixel's own
    annual count is unaffected by any other pixel), mask after."""
    mask = (region_da == region_idx)
    if land_only:
        mask = mask & viz.land_mask(da_timeseries.isel(time=0))
    w = np.cos(np.deg2rad(da_timeseries["latitude"]))
    annual = da_timeseries.groupby("time.year").sum().where(mask)
    return annual.weighted(w).mean(("latitude", "longitude"))
