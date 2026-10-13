"""
L3 — SPEI (Standardized Precipitation Evapotranspiration Index).

Method (Vicente-Serrano et al. 2010):
  1. D = P - PET (monthly, from L2).
  2. For timescale k: rolling k-month sum of D.
  3. Per calendar month, fit a 3-parameter log-logistic distribution by the method of
     L-moments (probability-weighted moments) over the CALIBRATION period, in its
     generalized-logistic (GLO) parameterization (changed 2026-09-23; see _glo_params:
     identical to the classic LL3 formulas for positively skewed samples, and also defined
     for negatively skewed ones, which the LL3 formulas previously masked as NaN on ~10% of
     land calendar-months). The LL3 helpers below are kept for reference/tests only.
  4. Transform every month's value through the fitted CDF, then map to the standard
     normal via the inverse normal CDF -> SPEI.

Fully vectorized over grid cells (no per-pixel Python loop): the leading-NaN pattern
from the rolling sum is identical across cells, so we can fit and transform whole
lat-lon planes at once.

SPEI <= -1.0 => drought month (config DROUGHT_THRESHOLD).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import gammaln, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io

_MIN_YEARS = 20  # minimum calibration samples for a stable fit


def _gammaf(x):
    return np.exp(gammaln(x))


def _pwm(arr: np.ndarray):
    """Unbiased probability-weighted moments w0,w1,w2 along axis 0 (samples).

    arr: (n, ...) sorted ASCENDING along axis 0. Returns w0,w1,w2 each shape (...).
    """
    n = arr.shape[0]
    i = np.arange(1, n + 1).reshape((n,) + (1,) * (arr.ndim - 1))
    w0 = arr.mean(axis=0)
    w1 = (((n - i) / (n - 1)) * arr).sum(axis=0) / n
    w2 = (((n - i) * (n - i - 1) / ((n - 1) * (n - 2))) * arr).sum(axis=0) / n
    return w0, w1, w2


def _loglogistic_params(w0, w1, w2):
    """3-parameter log-logistic params (alpha scale, beta shape, gamma location)."""
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g1 = _gammaf(1 + 1.0 / beta)
    g2 = _gammaf(1 - 1.0 / beta)
    alpha = (w0 - 2 * w1) * beta / (g1 * g2)
    gamma = w0 - alpha * g1 * g2
    return alpha, beta, gamma


def _loglogistic_cdf(x, alpha, beta, gamma):
    """F(x) = [1 + (alpha/(x-gamma))^beta]^-1.

    For x <= gamma (below the distribution support) the ratio -> +inf so F -> 0,
    i.e. an extreme deficit maps to a strongly negative SPEI. alpha,beta assumed > 0
    (invalid fits are masked upstream), so the power base is always positive.
    """
    xg = x - gamma
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.where(xg > 0, alpha / np.where(xg > 0, xg, 1.0), np.inf)
        cdf = 1.0 / (1.0 + ratio ** beta)
    return cdf


def _glo_params(w0, w1, w2):
    """Generalized-logistic (GLO; Hosking & Wallis 1997) parameters from the same PWMs.

    L-moments: l1 = w0, l2 = w0 - 2 w1, l3 = w0 - 6 w1 + 6 w2 (w_s are the (1-F)^s-weighted
    PWMs returned by `_pwm`). kappa = -tau3, alpha = l2 sin(kappa pi)/(kappa pi),
    xi = l1 - alpha (1/kappa - pi/sin(kappa pi)); kappa -> 0 is the logistic limit.

    For tau3 > 0 (kappa < 0) this is *exactly* the 3-parameter log-logistic of
    Vicente-Serrano et al. (2010) re-parameterized (beta = 1/tau3, gamma = xi + alpha/kappa,
    scale = -alpha/kappa; verified to machine precision), so every calendar month the LL3
    fit already handled is unchanged. Unlike LL3, GLO stays defined for tau3 <= 0
    (negatively skewed samples), which the LL3 PWM formulas map to beta <= 0 and which
    were previously masked (10% of land calendar-months, in every aridity class)."""
    l1 = w0
    l2 = w0 - 2 * w1
    l3 = w0 - 6 * w1 + 6 * w2
    with np.errstate(all="ignore"):
        k = -(l3 / l2)
        small = np.abs(k) < 1e-6
        kp = np.where(small, 1.0, k * np.pi)
        ratio = np.where(small, 1.0, np.sin(kp) / kp)
        alpha = l2 * ratio
        xi = l1 - np.where(small, 0.0, alpha * (1.0 / np.where(small, 1.0, k) - np.pi / np.sin(kp)))
    return xi, alpha, k, l2


def _glo_cdf(x, xi, alpha, k):
    """GLO CDF F = 1/(1+exp(-y)), y = -log(1 - k (x-xi)/alpha)/k (y = (x-xi)/alpha for k->0).
    Outside the support (k>0: upper bound; k<0: lower bound) F is 1 or 0 respectively and
    is later clipped, i.e. an extreme deficit maps to a strongly negative SPEI."""
    with np.errstate(all="ignore"):
        z = (x - xi) / alpha
        small = np.abs(k) < 1e-6
        arg = 1.0 - k * z
        y = np.where(small, z,
                     -np.log(np.where(arg > 0, arg, 1.0)) / np.where(small, 1.0, k))
        cdf = 1.0 / (1.0 + np.exp(-y))
        cdf = np.where(~small & ~(arg > 0), np.where(k > 0, 1.0, 0.0), cdf)
    return cdf


def spei_from_D(D: xr.DataArray, timescale: int, calib: C.Period) -> xr.DataArray:
    """Compute SPEI-`timescale` from a monthly D DataArray (time, lat, lon)."""
    Dk = D.rolling(time=timescale, min_periods=timescale).sum()
    years = Dk["time"].dt.year.values
    months = Dk["time"].dt.month.values
    vals = Dk.values  # (T, lat, lon)
    out = np.full_like(vals, np.nan)

    calib_mask_year = (years >= calib.start) & (years <= calib.end)
    for m in range(1, 13):
        month_idx = np.where(months == m)[0]
        finite_rows = month_idx[np.isfinite(vals[month_idx]).all(axis=(1, 2))]
        calib_idx = finite_rows[calib_mask_year[finite_rows]]
        if calib_idx.size < _MIN_YEARS:
            continue
        sample = np.sort(vals[calib_idx], axis=0)          # (n_calib, lat, lon)
        w0, w1, w2 = _pwm(sample)
        # GLO parameterization of the log-logistic (identical to LL3 for tau3>0, but also
        # defined for negatively skewed samples -- see _glo_params). Only genuinely
        # degenerate samples (zero L-scale, |tau3|>=1, non-finite) are masked.
        xi, alpha, kappa, l2 = _glo_params(w0, w1, w2)
        valid = np.isfinite(xi) & np.isfinite(alpha) & np.isfinite(kappa) \
            & (alpha > 0) & (l2 > 0) & (np.abs(kappa) < 1)
        # transform every (finite) month-m value across the whole record
        prob = _glo_cdf(vals[month_idx], xi, alpha, kappa)
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        spei_m = ndtri(prob)
        spei_m[:, ~valid] = np.nan
        out[month_idx] = spei_m

    da = xr.DataArray(out, coords=Dk.coords, dims=Dk.dims, name=f"spei{timescale}")
    da.attrs.update(units="1", long_name=f"SPEI-{timescale}",
                    calibration=f"{calib.start}-{calib.end}")
    return da


def build_spei(calib: C.Period = None) -> Path:
    """Compute all SPEI timescales from L2 and write L3 Zarr."""
    calib = calib or C.SPEI_CALIB_PERIOD
    mon = xr.open_zarr(C.MONTHLY_ZARR)
    D = mon["D"].load()  # (912, 721, 1440) float32 ~ 3.8 GB, fits in RAM
    ds = xr.Dataset(
        {f"spei{k}": spei_from_D(D, k, calib) for k in C.SPEI_TIMESCALES}
    )
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")
    ds.attrs.update(title="SPEI (L3)", calibration=calib.slug())
    out_path = C.spei_zarr(calib)
    io.write_zarr(ds.chunk({"time": -1, "latitude": C.TILE_LAT, "longitude": C.TILE_LON}),
                  out_path, mode="w")
    return out_path
