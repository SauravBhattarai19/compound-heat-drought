"""
ERA5 accumulation handling.

VERIFIED SEMANTICS (from data 2026-07-16): tp/ssrd/strd are 6-hourly snapshots of
ERA5's *1-hour* accumulation (value = amount over the hour ending at valid_time),
NOT 6-hour cumulative sums. Evidence: ssrd@12Z / 3600 s = 839 W/m^2 (correct midday
flux); mean-of-4-samples x 86400 s = 25 MJ/m^2/day at 48N in August (correct).

Therefore we do NOT difference consecutive steps. Each 6-hourly sample is treated as
a ~instantaneous flux (energy per hour) representative of its 6-hour block:

  radiation daily total (MJ/m^2/day) = mean_of_day(J/m^2 per hour) * 24 / 1e6
  precip     daily total (mm/day)    = sum_of_day(m per hour) * (24/n_samples) * 1000

n_samples is normally 4 (00/06/12/18 UTC). Using mean/sum over whatever samples the
day actually has keeps this robust to short months.

LIMITATION: precip is reconstructed from 4 of 24 hours -> unbiased but noisy for
convective precip. Radiation's smooth diurnal cycle makes its 4-point integral accurate.
"""
from __future__ import annotations
import xarray as xr

SECONDS_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0
J_PER_M2_TO_MJ = 1.0e-6
M_TO_MM = 1000.0

TIME_DIM = "valid_time"


def radiation_daily_total(da_j_m2: xr.DataArray) -> xr.DataArray:
    """Accumulated-radiation samples (J/m^2, 1h-accum) -> daily total MJ/m^2/day.

    Each sample is energy over 1 hour ~= mean flux. Daily mean flux * 24h gives the
    daily energy total; convert J->MJ.
    """
    daily_mean_hourly = da_j_m2.resample({TIME_DIM: "1D"}).mean()
    return daily_mean_hourly * HOURS_PER_DAY * J_PER_M2_TO_MJ


def precip_daily_total(da_m: xr.DataArray) -> xr.DataArray:
    """Accumulated-precip samples (m, 1h-accum) -> daily total mm/day.

    Sum the sampled hourly amounts and scale from n_samples hours to 24 h.
    """
    daily_sum = da_m.resample({TIME_DIM: "1D"}).sum()
    n_samples = da_m[TIME_DIM].resample({TIME_DIM: "1D"}).count()
    # mean per sampled hour * 24h * (m->mm)
    return (daily_sum / n_samples) * HOURS_PER_DAY * M_TO_MM
