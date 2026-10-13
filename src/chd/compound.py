"""
L5 — monthly compound heatwave-drought intersection (baseline + SPEI dependent).

A grid cell's month is a COMPOUND event if it has >=1 heatwave day AND is in drought:
    compound = hw_present AND (SPEI-`SPEI_COMPOUND_TS` <= DROUGHT_THRESHOLD)

Marginal heatwave-only and drought-only months are retained for attribution.
Output aligned monthly (time, lat, lon), one store per (baseline, spei-timescale).

SPEI is undefined for some (pixel, calendar-month) combinations (degenerate log-
logistic fit or <20 calibration years -- see spei.py). `drought = spei <= threshold`
in plain numpy/xarray silently evaluates NaN <= threshold to False, which would make
compound/drought/drought_only look like clean 0/1 boolean fields while actually
hiding "unknown" as "definitely not drought" -- systematically undercounting drought
(and therefore compound) frequency, worst in the driest cells where SPEI validity is
worst (under the old LL3 SPEI fit: hyper-arid land ~35% of calendar-months undefined,
~50% of land area affected to some degree). Since 2026-09-23 spei.py uses the GLO
parameterization and no grid-month of SPEI-1 (full-record calibration) is undefined, so
this Kleene logic is currently a no-op for the manuscript definition; it is kept so that
any genuinely degenerate fit (other calibrations/timescales) is still handled correctly. Fixed here with explicit Kleene 3-valued logic so "unknown" propagates as
NaN instead of collapsing to False, EXCEPT where the other term alone already
determines the outcome (heat=False forces compound/heat_only=False regardless of
drought validity; heat=True forces drought_only=False regardless of drought validity).
`drought`/`compound`/`heat_only`/`drought_only` are therefore float32 (to hold NaN);
`heat` is unaffected by SPEI and stays int8. Downstream annual aggregation
(trends.py::annual_counts) rescales each pixel-year's valid-month sum to a
full-window-equivalent rate rather than treating missing months as zero.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import io


def build_compound(baseline: C.Period, spei_ts: int = None, spei_calib: C.Period = None) -> Path:
    spei_ts = spei_ts or C.SPEI_COMPOUND_TS
    spei_calib = spei_calib or C.SPEI_CALIB_PERIOD
    hw = xr.open_zarr(C.HW_DIR / baseline.slug() / "hw_monthly.zarr")
    spei = xr.open_zarr(C.spei_zarr(spei_calib))[f"spei{spei_ts}"]

    hw, spei = xr.align(hw, spei, join="inner")
    heat = hw["hw_present"].astype(bool)
    drought = xr.where(np.isfinite(spei), (spei <= C.DROUGHT_THRESHOLD).astype("float32"),
                       np.float32("nan"))

    # Kleene AND: unknown only propagates where the known term can't already decide it.
    compound = xr.where(~heat, np.float32(0.0), drought)
    heat_only = xr.where(~heat, np.float32(0.0),
                         xr.where(np.isfinite(drought), 1.0 - drought, np.float32("nan")))
    drought_only = xr.where(heat, np.float32(0.0), drought)

    out = xr.Dataset({
        "compound": compound.astype("float32"),
        "heat_only": heat_only.astype("float32"),
        "drought_only": drought_only.astype("float32"),
        "drought": drought.astype("float32"),
        "heat": heat.astype("int8"),
    })
    out.attrs.update(title="Monthly compound heatwave-drought (L5)",
                     baseline=baseline.slug(), spei_timescale=spei_ts,
                     spei_calibration=spei_calib.slug(),
                     drought_threshold=C.DROUGHT_THRESHOLD,
                     nan_semantics="compound/heat_only/drought_only/drought are float32 "
                                    "with NaN where SPEI is undefined and not otherwise "
                                    "determined by heat status (Kleene 3-valued AND)")
    out_path = C.compound_zarr(baseline, spei_ts, spei_calib)
    io.write_zarr(out.chunk({"time": -1, "latitude": C.TILE_LAT, "longitude": C.TILE_LON}),
                  out_path, mode="w")
    return out_path
