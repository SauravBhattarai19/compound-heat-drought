"""L5b -- onset-order classification of compound heatwave-drought months.

Each maximal run of compound months (heat AND drought, both confirmed true) is
an *episode*; the episode is labeled by what preconditioned it in the month immediately
before its onset:

    1  heat-leading    : heat present, drought CONFIRMED absent the month before onset
    2  drought-leading : drought CONFIRMED present, heat absent the month before onset
    3  concurrent      : both confirmed absent the month before -> simultaneous onset
    4  ambiguous       : drought status UNDEFINED (SPEI degenerate) the month before
                          onset, regardless of heat's own state -- if heat was absent
                          that month, we cannot distinguish drought-leading from
                          concurrent; if heat was present, we cannot distinguish
                          heat-leading from concurrent (drought might secretly have
                          been present too). Both cases are genuinely unresolved and
                          are treated symmetrically rather than silently forced into
                          one of the other three categories.
    0  not compound / not an episode onset month

`drought` is float32 with real NaN where SPEI is undefined (chd.compound); casting NaN
straight to an integer/boolean type silently yields 0 ("confirmed not drought") in
numpy, which would bias episode onsets away from drought-leading/ambiguous and toward
heat-leading/concurrent whenever the *preceding* month's SPEI happened to be
undefined -- exactly the same class of NaN-as-False bug fixed in chd.compound, and
particularly consequential here since it would inflate this module's own
heat-leading share. `_leading_kernel` therefore takes `drought` as float32 directly and
uses NaN-safe float comparisons (`== 1.0`, which is correctly False for NaN) rather
than an integer cast, and routes EVERY undefined-drought-at-t-1 case to AMBIGUOUS
regardless of heat's state (an earlier version of this kernel only did this when heat
was also absent, which silently forced the heat-present/drought-unknown case into
HEAT_LEAD and inflated that category -- fixed here to be fully symmetric).

This is the monthly-resolution analogue of the daily hazard-sequence classification of
Kim et al. (2026, Sci. Adv.): it uses the same heat/drought monthly booleans that define
the compound field (chd.compound), so it is internally consistent with the rest of the
pipeline. Monthly resolution is a deliberate limitation vs. Kim's daily MYRIAD-HESA
sequencing and is reported as such.

Pure compute (numba kernel + xarray aggregation); mirrors the trends/compound split.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import numba
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

HEAT_LEAD, DROUGHT_LEAD, CONCURRENT, AMBIGUOUS = 1, 2, 3, 4
TYPE_NAMES = {HEAT_LEAD: "heat-leading", DROUGHT_LEAD: "drought-leading",
              CONCURRENT: "concurrent", AMBIGUOUS: "ambiguous"}


@numba.njit(cache=True, parallel=True)
def _leading_kernel(heat, drought):
    """(T, P) uint8 heat / float32 drought (NaN = SPEI undefined) -> (T, P) int8 labels.
    Compound-active test and the drought==True test are both NaN-safe float
    comparisons (NaN == 1.0 is False), so an undefined month is never silently
    counted as confirmed drought or confirmed not-drought."""
    T, P = heat.shape
    out = np.zeros((T, P), np.int8)
    for p in numba.prange(P):
        t = 0
        while t < T:
            if heat[t, p] and drought[t, p] == 1.0:
                if t == 0:
                    lab = CONCURRENT
                else:
                    hp = heat[t - 1, p] == 1
                    dp_prev = drought[t - 1, p]
                    dp_known = not np.isnan(dp_prev)
                    if not dp_known:
                        # Drought status at t-1 unknown: regardless of heat's own
                        # (always-defined) state, we cannot tell HEAT_LEAD from
                        # CONCURRENT (if hp) or DROUGHT_LEAD from CONCURRENT (if not
                        # hp) -- both are genuinely unresolved, not just the
                        # heat-absent case, so treat symmetrically as AMBIGUOUS.
                        lab = AMBIGUOUS
                    else:
                        dp = dp_prev == 1.0
                        if dp and not hp:
                            lab = DROUGHT_LEAD
                        elif hp and not dp:
                            lab = HEAT_LEAD
                        else:
                            lab = CONCURRENT
                while t < T and heat[t, p] and drought[t, p] == 1.0:
                    out[t, p] = lab
                    t += 1
            else:
                t += 1
    return out


def classify(heat: xr.DataArray, drought: xr.DataArray) -> xr.DataArray:
    """Episode-labeled onset-order category on (time, latitude, longitude).
    `heat` is cast to uint8 (always defined, int8 in the L5 store); `drought` is kept
    float32 (real NaN where SPEI undefined) so the kernel can distinguish "confirmed
    absent" from "unknown" -- do not cast `drought` to an integer/bool type here."""
    heat, drought = xr.align(heat.astype("uint8"), drought.astype("float32"), join="inner")
    T = heat.sizes["time"]
    H = np.ascontiguousarray(heat.values.reshape(T, -1))
    D = np.ascontiguousarray(drought.values.reshape(T, -1))
    cat = _leading_kernel(H, D).reshape(heat.shape)
    return xr.DataArray(cat, coords=heat.coords, dims=heat.dims, name="leading")


def classify_from_l5(baseline: C.Period, spei_ts: int, spei_calib: C.Period) -> xr.DataArray:
    ds = xr.open_zarr(C.compound_zarr(baseline, spei_ts, spei_calib))
    return classify(ds["heat"], ds["drought"])


def annual_type_counts(cat: xr.DataArray) -> xr.Dataset:
    """Per-year count of months of each onset type (year, latitude, longitude)."""
    out = {}
    for code, name in TYPE_NAMES.items():
        out[name.replace("-", "_")] = (cat == code).groupby("time.year").sum()
    return xr.Dataset(out).astype("int16")


def total_type_counts(cat: xr.DataArray) -> xr.Dataset:
    """Full-record count of months of each onset type (latitude, longitude)."""
    out = {}
    for code, name in TYPE_NAMES.items():
        out[name.replace("-", "_")] = (cat == code).sum("time")
    return xr.Dataset(out).astype("int32")


def dominance(cat: xr.DataArray) -> xr.DataArray:
    """(heat_leading - drought_leading) / (all classified compound months, including
    ambiguous-onset ones) per pixel, in [-1, 1]: +1 = purely heat-led, -1 = purely
    drought-led. Ambiguous-onset months count toward the denominator (they are real
    compound months) but not the numerator (their leading hazard is genuinely
    unresolved, not defaulted to either side). NaN where no compound."""
    tc = total_type_counts(cat)
    h = tc["heat_leading"].astype("float32")
    d = tc["drought_leading"].astype("float32")
    tot = (h + d + tc["concurrent"].astype("float32") + tc["ambiguous"].astype("float32"))
    with np.errstate(invalid="ignore", divide="ignore"):
        dom = xr.where(tot > 0, (h - d) / tot, np.nan)
    return dom.rename("leading_dominance")


if __name__ == "__main__":
    # quick sanity check on the default definition
    baseline = C.HW_BASELINES["1961_1990"]
    calib = C.SPEI_CALIB_PERIODS["1950_2025"]
    cat = classify_from_l5(baseline, C.SPEI_COMPOUND_TS, calib)
    vals = cat.values
    n_comp = int((vals > 0).sum())
    for code, name in TYPE_NAMES.items():
        n = int((vals == code).sum())
        print(f"{name:16s}: {n:>12,}  ({100*n/n_comp:5.1f}% of compound months)")
    print(f"total compound months: {n_comp:,}")


# --------------------------------------------------------------------------- #
# Base-rate / persistence null model (added 2026-09-23)
# --------------------------------------------------------------------------- #
def shift_drought_surrogate(drought: np.ndarray, seed: int = 0, min_years: int = 5,
                            max_years: int = 70) -> np.ndarray:
    """(T, P) drought array -> surrogate in which each pixel's drought series is circularly
    shifted by a random whole number of years (min_years..max_years) relative to heat.

    This keeps each marginal exactly as it is (heat untouched; drought keeps its seasonal
    cycle, frequency, month-to-month persistence and NaN pattern) and destroys only the
    physical timing relationship between heat and drought. Classifying onsets on
    (heat, surrogate drought) therefore gives the heat-/drought-leading shares that the
    marginals' own frequencies and persistence produce by themselves: the base-rate null
    for the onset-order statistics. Heat-only months are ~3x as common as drought-only
    months and heat persists more from month to month (P(heat_t|heat_t-1)~0.5 vs
    P(drought_t|drought_t-1)~0.2), so a heat-leading majority is expected under this null."""
    rng = np.random.default_rng(seed)
    T, P = drought.shape
    k = rng.integers(min_years, max_years + 1, size=P) * 12
    idx = (np.arange(T)[:, None] - k[None, :]) % T
    return np.take_along_axis(drought, idx, axis=0)


def onset_summary(heat: np.ndarray, drought: np.ndarray, weights: np.ndarray,
                  years: np.ndarray) -> dict:
    """Area-weighted onset-order summary for (T, P) land-pixel arrays: month- and onset-
    weighted category shares, share of area with positive/negative dominance, and the
    heat-/drought-leading annual extent (early vs recent half, Pettitt year)."""
    from scipy.stats import rankdata
    cat = _leading_kernel(np.ascontiguousarray(heat.astype(np.uint8)),
                          np.ascontiguousarray(drought.astype(np.float32)))
    comp = cat > 0
    onset = comp.copy(); onset[1:] &= ~comp[:-1]
    out = {}
    for c, nm in TYPE_NAMES.items():
        nm = nm.replace("-", "_")
        out[f"share_months_{nm}"] = float((((cat == c) * weights).sum()) / ((comp * weights).sum()))
        out[f"share_onsets_{nm}"] = float((((cat == c) & onset) * weights).sum() / ((onset * weights).sum()))
    nh = (cat == HEAT_LEAD).sum(0); nd = (cat == DROUGHT_LEAD).sum(0); nt = comp.sum(0)
    dom = np.where(nt > 0, (nh - nd) / np.maximum(nt, 1), np.nan)
    ok = np.isfinite(dom)
    out["area_dominance_positive"] = float((weights * (dom > 0) * ok).sum() / (weights * ok).sum())
    out["area_dominance_negative"] = float((weights * (dom < 0) * ok).sum() / (weights * ok).sum())
    hb = heat.astype(bool); db = drought == 1.0
    wsum = weights.sum()
    out["heat_only_months_per_yr"] = float(((hb & ~db) * weights).sum() / wsum / len(years))
    out["drought_only_months_per_yr"] = float(((~hb & db) * weights).sum() / wsum / len(years))
    out["p_heat_given_heat_prev"] = float(((hb[1:] & hb[:-1]) * weights).sum() / (hb[:-1] * weights).sum())
    out["p_heat"] = float((hb * weights).sum() / (hb.shape[0] * wsum))
    out["p_drought_given_drought_prev"] = float(((db[1:] & db[:-1]) * weights).sum() / (db[:-1] * weights).sum())
    out["p_drought"] = float((db * weights).sum() / (db.shape[0] * wsum))
    ny = len(years); half = ny // 2
    for c, nm in [(HEAT_LEAD, "heat_leading"), (DROUGHT_LEAD, "drought_leading")]:
        ext = ((cat == c).reshape(ny, 12, -1).any(1) * weights).sum(1) / weights.sum() * 100
        out[f"extent_{nm}_early"] = float(ext[:half].mean())
        out[f"extent_{nm}_late"] = float(ext[half:].mean())
        n = ny; r = rankdata(ext)
        U = np.array([2 * r[:k].sum() - k * (n + 1) for k in range(1, n + 1)])
        out[f"extent_{nm}_pettitt"] = int(years[int(np.argmax(np.abs(U)))])
    return out
