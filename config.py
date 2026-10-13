"""
Central configuration for the ERA5 compound heatwave-drought analysis.

Single source of truth for paths, time periods, thresholds and chunking. Every
stage reads its parameters from here so that changing a *time period* (heatwave
baseline, SPEI calibration, or trend window) never requires editing pipeline code
and never invalidates the period-INDEPENDENT caches (L1 daily fields, L2 monthly
PET/P/D).

Layered cache (see plan):
  L1 daily fields       period-INDEPENDENT  -> DAILY_ZARR
  L2 monthly PET/P/D    period-INDEPENDENT  -> MONTHLY_ZARR
  L3 SPEI-1/3/6/12      depends: SPEI_CALIB_PERIOD
  L4 T90 + HW metrics   depends: HW_BASELINES (one output per baseline)
  L5 monthly compound   depends: baseline + SPEI
  L6 trends             depends: TREND_WINDOWS + framing
  L7 event tracking     depends: baseline + SPEI + EVENT_MIN_AREA_PIXELS + EVENT_LINK_OVERLAP_FRAC
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Raw ERA5 input (monthly NetCDF, see README) and the root for every derived store.
# Set them with the CHD_ERA5_DIR / CHD_DATA_ROOT environment variables.
ERA5_DIR = Path(os.environ.get("CHD_ERA5_DIR", Path(__file__).parent / "era5"))
ERA5_PATTERN = "era5_daily_{year:04d}_{month:02d}.nc"
DATA_ROOT = Path(os.environ.get("CHD_DATA_ROOT", Path(__file__).parent / "data"))

DAILY_ZARR   = DATA_ROOT / "L1_daily"          # per-year zarr stores
MONTHLY_ZARR = DATA_ROOT / "L2_monthly.zarr"   # monthly PET, P, D
SPEI_ZARR    = DATA_ROOT / "L3_spei.zarr"      # SPEI-1/3/6/12
HW_DIR       = DATA_ROOT / "L4_heatwave"       # one subdir per baseline
COMPOUND_DIR = DATA_ROOT / "L5_compound"       # one subdir per (baseline, spei) combo
TRENDS_DIR   = DATA_ROOT / "L6_trends"
EVENTS_DIR   = DATA_ROOT / "L7_events"         # event_id raster + catalog per (baseline, spei) combo
FIG_DIR      = Path(__file__).parent / "figures"

# Local NVMe scratch cache for the full-record tasmax array (~113 GB in RAM/on disk
# uncompressed). tasmax is baseline-INDEPENDENT but events.py's L7 driver reads it once
# per heatwave baseline; caching the one unavoidable slow read (~12 MB/s off the HDD
# JBOD at DATA_ROOT) to local NVMe means the second baseline's read is fast instead of
# re-reading the source disk. Safe to delete -- it is a derived, ephemeral cache, not
# a canonical pipeline output (unlike everything under DATA_ROOT).
TASMAX_CACHE_DIR = Path(__file__).parent / ".tasmax_cache"

# --------------------------------------------------------------------------- #
# ERA5 variable / coordinate names (verified from files 2026-07-16)
# --------------------------------------------------------------------------- #
TIME_DIM = "valid_time"
LAT_DIM  = "latitude"
LON_DIM  = "longitude"
STEPS_PER_DAY = 4  # 00/06/12/18 UTC

# Accumulated fields are 6-hourly snapshots of ERA5's 1-HOUR accumulation
# (value = amount over the hour ending at valid_time). Verified physically.
ACCUM_VARS = ("tp", "ssrd", "strd")
INSTANT_VARS = ("t2m", "d2m", "u10", "v10", "sp")

# --------------------------------------------------------------------------- #
# Full archive span
# --------------------------------------------------------------------------- #
YEAR_START = 1950
YEAR_END   = 2025

# --------------------------------------------------------------------------- #
# Time-period parameters (the modularity knobs)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Period:
    start: int
    end: int
    def slug(self) -> str:
        return f"{self.start}_{self.end}"
    def __contains__(self, year: int) -> bool:
        return self.start <= year <= self.end

# Heatwave T90 baselines (compute both; L4 writes one store per baseline).
HW_BASELINES = {
    "1961_1990": Period(1961, 1990),
    "1991_2020": Period(1991, 2020),
}

# SPEI log-logistic calibration period (full record for a stable climatology).
SPEI_CALIB_PERIOD = Period(YEAR_START, YEAR_END)

# Additional FIXED-baseline calibrations, mirroring HW_BASELINES, to test how much the
# choice of SPEI "time reference" affects drought/compound detection and trends (the
# full-record calibration above pools pre- and post-warming years into one distribution;
# see manuscript/spei_time_reference_explainer.tex). "1950_2025" is kept identical to
# SPEI_CALIB_PERIOD above (same key as its slug) so spei_zarr() below resolves it to the
# already-computed default store instead of recomputing.
SPEI_CALIB_PERIODS = {
    "1950_2025": SPEI_CALIB_PERIOD,
    "1961_1990": Period(1961, 1990),
    "1991_2020": Period(1991, 2020),
}

# Trend analysis windows (Mann-Kendall span). Cheap to add more later.
TREND_WINDOWS = {
    "1950_2025": Period(1950, 2025),
    "1979_2025": Period(1979, 2025),  # satellite-era sensitivity
}

# --------------------------------------------------------------------------- #
# Method parameters
# --------------------------------------------------------------------------- #
# Heatwave
HW_PERCENTILE   = 90     # T90
HW_WINDOW_DAYS  = 15     # +/- window (31-day) for day-of-year climatology
HW_MIN_DURATION = 3      # consecutive hot days

# Drought / SPEI
SPEI_TIMESCALES   = (1, 3, 6, 12)     # months
SPEI_COMPOUND_TS  = 1                  # timescale used in the compound definition
DROUGHT_THRESHOLD = -1.0               # SPEI <= this  => drought month

# Trends
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MK_ALPHA = 0.05

# Event tracking (L7). These are OUR OWN tunables, not literature-sourced constants --
# the China IAD paper (Liu et al. 2025; see docs/compound_heat_drought.md) states "4 grid
# points" only for their regional China domain and gives no numeral at all for the
# day-to-day overlap-linking threshold. Validate against the 2010/2003 real-data test
# runs before treating these as final.
EVENT_MIN_AREA_PIXELS  = 4       # drop connected components smaller than this (per day)
EVENT_LINK_OVERLAP_FRAC = 0.10   # min overlap fraction (of the smaller region) to link
                                  # two components across consecutive days into one event
EVENT_GAP_BRIDGE_DAYS  = 0       # sensitivity knob for the monthly-SPEI/daily-heat month-
                                  # boundary artifact (an event can spuriously split/extend
                                  # right at a calendar-month boundary). Default OFF (0) so
                                  # the production run matches the literal boolean
                                  # definition; a value > 0 would bridge across up to that
                                  # many drought-criterion-only gap days. Not implemented as
                                  # daily-SPEI interpolation -- out of scope.

# PET / physical constants (FAO-56)
ALBEDO      = 0.23        # reference grass
EMISSIVITY  = 0.98        # surface longwave emissivity for Rnl from strd
STEFAN_BOLTZ = 4.903e-9   # MJ K^-4 m^-2 day^-1 (FAO-56)
WIND_10M_TO_2M = 0.748    # FAO log-profile factor: u2 = u10 * 4.87/ln(67.8*10-5.42)

# --------------------------------------------------------------------------- #
# Compute / chunking
# --------------------------------------------------------------------------- #
# The raw archive and the intermediate disk are BOTH spinning HDDs, so the pipeline is
# I/O-bound: concurrent readers past ~4-6 thrash the disk heads and get SLOWER. These
# counts are tuned to the disk, not the 192 CPU threads (which sit idle by necessity).
L1_WORKERS = 6           # month-level readers off the USB archive (benchmarked optimum)
N_WORKERS = 12           # dask workers for L2 (reads L1 off internal HDD)
THREADS_PER_WORKER = 2
# Spatial tile size for time-reducing ops (percentile, SPEI fit, Mann-Kendall).
TILE_LAT = 180
TILE_LON = 360
ZARR_COMPRESSOR = "zstd"
ZARR_CLEVEL = 5


def era5_file(year: int, month: int) -> Path:
    return ERA5_DIR / ERA5_PATTERN.format(year=year, month=month)


def daily_year_zarr(year: int) -> Path:
    return DAILY_ZARR / f"daily_{year:04d}.zarr"


def daily_month_zarr(year: int, month: int) -> Path:
    return DAILY_ZARR / f"daily_{year:04d}_{month:02d}.zarr"


def spei_zarr(calib: Period) -> Path:
    """L3 output path for a given SPEI calibration period. The default full-record
    calibration keeps its original path (SPEI_ZARR) so the already-computed store is
    reused rather than recomputed; the fixed-baseline sensitivity runs get distinct
    paths keyed by the calibration period's slug."""
    if calib.slug() == SPEI_CALIB_PERIOD.slug():
        return SPEI_ZARR
    return DATA_ROOT / f"L3_spei_calib{calib.slug()}.zarr"


def compound_zarr(baseline: Period, spei_ts: int, spei_calib: Period) -> Path:
    """L5 output path for a (heatwave baseline, SPEI timescale, SPEI calibration)
    combination. Default SPEI calibration keeps the original naming (no _calib suffix)."""
    if spei_calib.slug() == SPEI_CALIB_PERIOD.slug():
        return COMPOUND_DIR / f"{baseline.slug()}_spei{spei_ts}.zarr"
    return COMPOUND_DIR / f"{baseline.slug()}_spei{spei_ts}_calib{spei_calib.slug()}.zarr"


def trends_zarr(baseline: Period, spei_ts: int, spei_calib: Period, var_name: str) -> Path:
    """L6 output path, same default-calibration naming convention as compound_zarr()."""
    if spei_calib.slug() == SPEI_CALIB_PERIOD.slug():
        return TRENDS_DIR / f"{baseline.slug()}_spei{spei_ts}_{var_name}.zarr"
    return TRENDS_DIR / f"{baseline.slug()}_spei{spei_ts}_calib{spei_calib.slug()}_{var_name}.zarr"


def events_zarr(baseline: Period, spei_ts: int, spei_calib: Period) -> Path:
    """L7 event_id raster path, same default-calibration naming convention as compound_zarr()."""
    if spei_calib.slug() == SPEI_CALIB_PERIOD.slug():
        return EVENTS_DIR / f"{baseline.slug()}_spei{spei_ts}.zarr"
    return EVENTS_DIR / f"{baseline.slug()}_spei{spei_ts}_calib{spei_calib.slug()}.zarr"


def events_catalog(baseline: Period, spei_ts: int, spei_calib: Period) -> Path:
    """L7 event catalog (Parquet) path, mirroring events_zarr() naming."""
    if spei_calib.slug() == SPEI_CALIB_PERIOD.slug():
        return EVENTS_DIR / f"{baseline.slug()}_spei{spei_ts}_catalog.parquet"
    return EVENTS_DIR / f"{baseline.slug()}_spei{spei_ts}_calib{spei_calib.slug()}_catalog.parquet"
