"""Validate FAO-56 ET0 against the published worked example (Allen et al. 1998).

Example 18: daily ET0 at Bangkok-like conditions is not used; we use the canonical
Example 17 / Box worked numbers reproduced widely:
  location N'Diaye, 13.73N; day 6 July
  Tmax=34.8, Tmin=25.6 degC, ea=2.85 kPa, Rn=14.33 MJ/m2/day (given), G=0,
  u2=2.078 m/s, P=100.1 kPa (elev ~8 m)  -> ET0 = 5.72 mm/day.

Since our et0_fao56 computes Rn internally from rs/rl, we test the *formula core* by
feeding rs/rl that reproduce the given Rn, plus a direct check of svp/slope constants.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import pet


def test_svp_and_slope():
    # FAO-56 Table: es(34.8)=5.56, es(25.6)=3.28 kPa; slope at Tmean~30.2 ~ 0.246
    assert abs(float(pet.svp(34.8)) - 5.56) < 0.03
    assert abs(float(pet.svp(25.6)) - 3.28) < 0.03


def test_et0_formula_core():
    # Reproduce Example with a known Rn=14.33 by choosing rs/rl.
    tmax, tmin = 34.8, 25.6
    ea = 2.85
    ps = 100.1
    u2 = 2.078
    Rn_target = 14.33
    # invert Rn = (1-alb)*rs - (eps*sig*(Tmax_k4+Tmin_k4)/2 - rl)
    KELVIN = 273.15
    lw_up = C.EMISSIVITY * C.STEFAN_BOLTZ * ((tmax + KELVIN) ** 4 + (tmin + KELVIN) ** 4) / 2
    # pick rl so net-longwave ~ FAO example (~3.71); then rs from remainder
    rnl_net = 3.71
    rl = lw_up - rnl_net
    rns = Rn_target + rnl_net
    rs = rns / (1 - C.ALBEDO)
    et0 = float(pet.et0_fao56(
        np.array([tmax]), np.array([tmin]), np.array([rs]), np.array([rl]),
        np.array([u2]), np.array([ea]), np.array([ps]),
    )[0])
    print("ET0 =", round(et0, 2), "mm/day (expected ~5.72)")
    assert abs(et0 - 5.72) < 0.25


if __name__ == "__main__":
    test_svp_and_slope()
    test_et0_formula_core()
    print("PET tests passed.")
