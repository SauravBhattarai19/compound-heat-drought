"""Validate the SPEI log-logistic/L-moment core on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import spei


def _synthetic_D(n_years=76, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("1950-01-01", periods=n_years * 12, freq="MS")
    # seasonal mean water balance + noise, 2x2 grid
    month = time.month.values
    seasonal = 20 * np.sin(2 * np.pi * (month - 1) / 12)      # mm
    base = seasonal[:, None, None] + rng.normal(0, 30, (len(time), 2, 2))
    return xr.DataArray(base, coords={"time": time,
                                      "latitude": [10, 11], "longitude": [20, 21]},
                        dims=["time", "latitude", "longitude"], name="D")


def test_spei_standardized():
    D = _synthetic_D()
    calib = C.Period(1950, 2025)
    s1 = spei.spei_from_D(D, 1, calib)
    v = s1.values[np.isfinite(s1.values)]
    print("SPEI-1 mean=%.3f std=%.3f" % (v.mean(), v.std()))
    assert abs(v.mean()) < 0.05
    assert abs(v.std() - 1.0) < 0.08
    assert np.nanmin(v) < -1.5 and np.nanmax(v) > 1.5


def test_spei_monotonic_extremes():
    # The naturally driest/wettest January should map to strongly -/+ SPEI,
    # without corrupting the fit (no injected outliers).
    D = _synthetic_D()
    s1 = spei.spei_from_D(D, 1, C.Period(1950, 2025))
    jan = s1.values[::12, 0, 0]
    d_jan = D.values[::12, 0, 0]
    driest = float(jan[np.argmin(d_jan)])
    wettest = float(jan[np.argmax(d_jan)])
    print("driest-Jan SPEI=%.2f  wettest-Jan SPEI=%.2f" % (driest, wettest))
    assert driest < -1.3 and wettest > 1.3


if __name__ == "__main__":
    test_spei_standardized()
    test_spei_monotonic_extremes()
    print("SPEI tests passed.")
