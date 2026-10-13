"""
FAO-56 Penman-Monteith reference evapotranspiration (ET0).

Inputs are the L1 daily fields. We use the *actual* ERA5 radiation fluxes (rs down
shortwave, rl down longwave) to form net radiation, which is more accurate than
FAO-56's empirical net-longwave estimate since we have measured downwelling longwave.

  Rns = (1 - albedo) * rs
  Rnl_net = eps*sigma*Tmean_K^4 - rl        (net *outgoing* longwave, MJ/m2/day)
  Rn  = Rns - Rnl_net
  ET0 = [0.408 D (Rn - G) + g (900/(T+273)) u2 (es - ea)] / [D + g (1 + 0.34 u2)]

with G = 0 (daily), D = slope of saturation vapour pressure curve, g = psychrometric
constant, es = mean saturation vapour pressure from Tmax/Tmin, ea from L1.

Reference: Allen et al. (1998), FAO Irrigation & Drainage Paper 56.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

KELVIN = 273.15
CP_RATIO = 0.665e-3   # psychrometric constant coefficient (kPa/degC per kPa)


def svp(t_c):
    """Saturation vapour pressure (kPa) at temperature t (degC), FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def svp_slope(tmean_c):
    """Slope of the SVP curve (kPa/degC), FAO-56 eq. 13."""
    return 4098.0 * svp(tmean_c) / (tmean_c + 237.3) ** 2


def et0_fao56(tasmax, tasmin, rs, rl, u2, ea, ps):
    """FAO-56 Penman-Monteith ET0 (mm/day). Inputs are L1 daily fields (see units)."""
    tmean = (tasmax + tasmin) / 2.0
    delta = svp_slope(tmean)                      # kPa/degC
    gamma = CP_RATIO * ps                          # psychrometric constant, kPa/degC

    es = (svp(tasmax) + svp(tasmin)) / 2.0         # kPa
    vpd = np.maximum(es - ea, 0.0)                 # kPa

    # Net radiation (MJ/m2/day) from actual fluxes
    rns = (1.0 - C.ALBEDO) * rs
    tmax_k4 = (tasmax + KELVIN) ** 4
    tmin_k4 = (tasmin + KELVIN) ** 4
    lw_up = C.EMISSIVITY * C.STEFAN_BOLTZ * (tmax_k4 + tmin_k4) / 2.0
    rnl_net = lw_up - rl                           # net outgoing longwave
    rn = rns - rnl_net
    g = 0.0

    num = 0.408 * delta * (rn - g) + gamma * (900.0 / (tmean + KELVIN)) * u2 * vpd
    den = delta + gamma * (1.0 + 0.34 * u2)
    et0 = num / den
    return np.maximum(et0, 0.0)


def et0_from_daily(ds: xr.Dataset) -> xr.DataArray:
    """Compute ET0 DataArray (mm/day) from an L1 daily Dataset."""
    et0 = et0_fao56(
        ds["tasmax"], ds["tasmin"], ds["rs"], ds["rl"],
        ds["u2"], ds["ea"], ds["ps"],
    )
    et0.name = "pet"
    et0.attrs.update(units="mm/day", long_name="FAO-56 Penman-Monteith reference ET")
    return et0.astype("float32")
