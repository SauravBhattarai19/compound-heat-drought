"""ENSO phase classification and compositing, for the manuscript's ENSO-modulation
figure only (not part of the core L1-L7 pipeline).

Source: NOAA CPC's Relative Oceanic Nino Index (RONI), the operational successor to
ONI (adopted Feb 2026) -- same Nino 3.4 SST anomaly, additionally adjusted for the
mean 20N-20S tropical SST anomaly so the +-0.5C threshold means the same thing in a
warmed-tropics year as it did in 1950 (see manuscript Methods). Downloaded once and
cached to data_aux/roni.txt, mirroring the drylands.py fetch-and-cache pattern.

Phase classification follows the standard operational rule (the same rule described
for ONI by Trenberth 1997, carried over unchanged to RONI): a month is El Nino/La Nina
only if it falls inside a run of >=5 consecutive overlapping 3-month seasons all
>=+0.5C / <=-0.5C; everything else (including short-lived excursions) is Neutral.

Pure aggregation here (no matplotlib), mirroring the trends.py/viz.py and
sensitivity.py/regions.py compute-vs-plot split -- plotting lives in
manuscript_figures.py.
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

DATA_AUX = Path(__file__).resolve().parents[2] / "data_aux"
RONI_TXT = DATA_AUX / "roni.txt"
_RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"

THRESHOLD = 0.5     # degC, same magnitude as ONI's historical threshold
MIN_RUN = 5          # consecutive overlapping 3-month seasons required for an event

# overlapping-season code -> its center (representative) calendar month
_SEASON_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _ensure_roni_txt() -> None:
    if RONI_TXT.exists():
        return
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    print(f"[enso] downloading RONI index from CPC -> {RONI_TXT} ...", flush=True)
    urllib.request.urlretrieve(_RONI_URL, RONI_TXT)


def roni_monthly(recompute: bool = False) -> pd.Series:
    """RONI anomaly (degC), one value per calendar month, indexed by the first-of-month
    Timestamp of that season's center month (so it aligns directly with L5's monthly
    time axis). Each CPC row's YR already labels the center month's year (e.g. "DJF
    1950" = Dec 1949-Feb 1950, center Jan 1950)."""
    if recompute and RONI_TXT.exists():
        RONI_TXT.unlink()
    _ensure_roni_txt()
    df = pd.read_csv(RONI_TXT, sep=r"\s+")
    df["month"] = df["SEAS"].map(_SEASON_CENTER_MONTH)
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["month"], day=1))
    s = df.set_index("date")["ANOM"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _classify_runs(cond: np.ndarray, min_run: int) -> np.ndarray:
    """Boolean array -> boolean array, True only inside runs of >=min_run consecutive
    True values (no gap allowed, matching the operational ENSO-event rule)."""
    out = np.zeros(cond.shape, dtype=bool)
    n = len(cond)
    i = 0
    while i < n:
        if not cond[i]:
            i += 1
            continue
        j = i
        while j < n and cond[j]:
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out


def monthly_phase(recompute: bool = False) -> pd.Series:
    """Monthly ENSO phase ('El Nino'/'La Nina'/'Neutral') aligned to `roni_monthly()`."""
    s = roni_monthly(recompute=recompute)
    warm = _classify_runs((s >= THRESHOLD).values, MIN_RUN)
    cold = _classify_runs((s <= -THRESHOLD).values, MIN_RUN)
    phase = np.full(len(s), "Neutral", dtype=object)
    phase[warm] = "El Nino"
    phase[cold] = "La Nina"
    return pd.Series(phase, index=s.index, name="phase")


def annual_phase(recompute: bool = False) -> pd.Series:
    """Calendar-year -> ENSO phase, using that year's DJF value (the conventional
    year-label for an ENSO winter, e.g. the 1997-98 event's DJF row is "DJF 1998")."""
    phase = monthly_phase(recompute=recompute)
    djf = phase[phase.index.month == 1]
    return pd.Series(djf.values, index=djf.index.year, name="phase")


# --------------------------------------------------------------------------- #
# Compositing against L5 monthly fields
# --------------------------------------------------------------------------- #
def _l5(baseline_key: str, ts: int, calib_key: str) -> xr.Dataset:
    baseline = C.HW_BASELINES[baseline_key]
    calib = C.SPEI_CALIB_PERIODS[calib_key]
    return xr.open_zarr(C.compound_zarr(baseline, ts, calib))


DETREND_WINDOW_YEARS = 21   # centred running-mean window used to remove the secular trend


def _running_mean_years(a: np.ndarray, window: int = DETREND_WINDOW_YEARS) -> np.ndarray:
    """NaN-aware centred running mean along axis 0 (years), window truncated at the record
    edges. Used to remove the low-frequency (secular + multidecadal) component before
    compositing by ENSO phase."""
    a = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(x, axis=0)])
    cn = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(fin, axis=0)])
    n = a.shape[0]; h = window // 2
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[hi] - cs[lo]) / (cn[hi] - cn[lo])


def phase_composite(var: str, baseline_key: str = "1961_1990", ts: int = 1,
                     calib_key: str = "1950_2025", detrend: bool = True) -> xr.Dataset:
    """Per-pixel mean monthly frequency of `var` (0-1) within each ENSO phase, plus the
    El Nino-minus-Neutral and La Nina-minus-Neutral anomaly (percentage points).

    detrend=True (default since 2026-09-23): each pixel's monthly series is first expressed
    as an anomaly from a centred 21-yr running mean of its annual-mean frequency. ENSO
    phases are not evenly spread in time (1950-2025: La Nina years average 1998, Neutral
    years 1980), so a raw composite aliases the secular compound trend into the phase
    difference -- verified: the raw Horn-of-Africa La Nina anomaly (+9.0 pp) shrinks to
    ~+2.7 pp once the trend is removed, while El Nino composites are almost unchanged."""
    ds = _l5(baseline_key, ts, calib_key)
    da = ds[var]
    phase = monthly_phase().reindex(da["time"].values).values
    vals = da.values.astype(np.float32)
    if detrend:
        ny = vals.shape[0] // 12
        with np.errstate(invalid="ignore"):
            ann = np.nanmean(vals.reshape(ny, 12, *vals.shape[1:]), axis=1)
        base = _running_mean_years(ann).astype(np.float32)
        vals = vals - np.repeat(base, 12, axis=0)
    means = {}
    for lab in ("El Nino", "La Nina", "Neutral"):
        with np.errstate(invalid="ignore"):
            means[lab] = xr.DataArray(np.nanmean(vals[phase == lab], axis=0) * 100.0,
                                      coords={"latitude": da["latitude"], "longitude": da["longitude"]},
                                      dims=("latitude", "longitude"))
    out = xr.Dataset({
        "el_nino": means["El Nino"], "la_nina": means["La Nina"], "neutral": means["Neutral"],
        "el_nino_anom": means["El Nino"] - means["Neutral"],
        "la_nina_anom": means["La Nina"] - means["Neutral"],
    })
    out.attrs.update(var=var, baseline=baseline_key, spei_calib=calib_key,
                      units="% of months" + (" (anomaly from 21-yr running mean)" if detrend else ""),
                      threshold=THRESHOLD, min_run=MIN_RUN, detrended=int(detrend))
    return out


def annual_series_by_phase(annual: pd.Series) -> pd.DataFrame:
    """Join a year-indexed series (e.g. global land-mean annual compound count) with
    the annual ENSO phase label; one row per year."""
    ph = annual_phase().reindex(annual.index)
    return pd.DataFrame({"value": annual, "phase": ph})


def phase_stats(annual: pd.Series) -> dict:
    """Group means/n for `annual` by ENSO phase, a Mann-Whitney U test (El Nino vs La
    Nina), and Pearson correlation between `annual` and the annual-mean RONI value --
    both on the raw series and on the linear-detrended residual (so a shared secular
    trend can't manufacture a spurious ENSO correlation)."""
    df = annual_series_by_phase(annual).dropna()
    grp = df.groupby("phase")["value"].agg(["mean", "std", "count"])

    en = df.loc[df["phase"] == "El Nino", "value"]
    ln = df.loc[df["phase"] == "La Nina", "value"]
    u_stat, u_p = stats.mannwhitneyu(en, ln, alternative="two-sided") if len(en) and len(ln) else (np.nan, np.nan)

    roni_annual = roni_monthly().groupby(roni_monthly().index.year).mean().reindex(annual.index)
    valid = annual.notna() & roni_annual.notna()
    r_raw, p_raw = stats.pearsonr(annual[valid], roni_annual[valid])

    years = annual.index.values.astype(float)
    resid = annual.values - np.polyval(np.polyfit(years, annual.values, 1), years)
    r_detrend, p_detrend = stats.pearsonr(resid[valid.values], roni_annual[valid].values
                                           - np.polyval(np.polyfit(years[valid.values], roni_annual[valid].values, 1), years[valid.values]))

    # trend-free comparison: anomalies from the same centred 21-yr running mean used for
    # the spatial composites (phases are unevenly distributed in time, see phase_composite)
    base = _running_mean_years(annual.values.astype(float))
    anom = pd.Series(annual.values - base, index=annual.index)
    dfa = annual_series_by_phase(anom).dropna()
    grp_dt = dfa.groupby("phase")["value"].agg(["mean", "std", "count"])
    en_d = dfa.loc[dfa["phase"] == "El Nino", "value"]; ln_d = dfa.loc[dfa["phase"] == "La Nina", "value"]
    nu_d = dfa.loc[dfa["phase"] == "Neutral", "value"]; ns_d = dfa.loc[dfa["phase"] != "Neutral", "value"]
    nu = df.loc[df["phase"] == "Neutral", "value"]; ns = df.loc[df["phase"] != "Neutral", "value"]
    djf = roni_monthly(); djf = djf[djf.index.month == 1]
    djf = pd.Series(djf.values, index=djf.index.year).reindex(annual.index)
    okd = np.isfinite(anom.values) & djf.notna().values
    r_djf, p_djf = stats.pearsonr(anom.values[okd], djf.values[okd])
    # does ENSO's (slightly La Nina-ward) drift contribute to the secular trend? Remove each
    # phase's mean running-mean anomaly and re-estimate the Theil-Sen trend.
    import pymannkendall as pmk
    ra = roni_monthly(); ra = ra.groupby(ra.index.year).mean().reindex(annual.index)
    ph_all = annual_phase().reindex(annual.index)
    adj = annual.values.astype(float).copy()
    for lab in ("El Nino", "La Nina", "Neutral"):
        m = (ph_all == lab).values
        adj[m] = adj[m] - np.nanmean((annual.values - base)[m])
    yrs_ = annual.index.values.astype(float)
    return {
        "roni_annual_theilsen_per_decade": pmk.original_test(ra.values).slope * 10,
        "roni_annual_mk_p": pmk.original_test(ra.values).p,
        "trend_per_decade_raw": stats.theilslopes(annual.values, yrs_)[0] * 10,
        "trend_per_decade_enso_phase_adjusted": stats.theilslopes(adj, yrs_)[0] * 10,
        "pearson_r_runmean_anom_vs_djf_roni": r_djf, "pearson_p_runmean_anom_vs_djf_roni": p_djf,
        "group_means_detrended": grp_dt,
        "mannwhitney_p_detrended": stats.mannwhitneyu(en_d, ln_d, alternative="two-sided")[1],
        "mannwhitney_p_neutral_vs_enso_raw": stats.mannwhitneyu(nu, ns, alternative="two-sided")[1],
        "mannwhitney_p_neutral_vs_enso_detrended": stats.mannwhitneyu(nu_d, ns_d, alternative="two-sided")[1],
        "mean_year": df.index.to_series().groupby(df["phase"]).mean(),
        "group_means": grp,
        "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
        "pearson_r_raw": r_raw, "pearson_p_raw": p_raw,
        "pearson_r_detrended": r_detrend, "pearson_p_detrended": p_detrend,
    }
