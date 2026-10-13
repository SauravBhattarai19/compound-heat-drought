"""
L6 figures — land-masked global trend maps (Sen's slope + MK significance stippling)
and regional/global time series, plus (from `small_multiples_trend_map` onward) the
cross-definition SENSITIVITY figure battery comparing results across the 2 (heatwave
baseline) x 3 (SPEI calibration) compound-definition matrix. Pure plotting -- data
prep/aggregation lives in `chd.sensitivity`/`chd.regions` (this module must not import
either, to avoid a circular import: both of them import `viz.land_mask`).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

_LAND = None


def land_mask(like: xr.DataArray) -> xr.DataArray:
    """Boolean land mask on the grid of `like` (True over land)."""
    global _LAND
    m = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = m.mask(like["longitude"], like["latitude"])
    return xr.where(np.isnan(mask), False, True)


# --------------------------------------------------------------------------------- #
# Private helpers shared by every global Robinson-projection map below.
# --------------------------------------------------------------------------------- #

def _robinson_axes(figsize=(12, 5.5)):
    """Build one Robinson-projection axes with coastlines, global extent set."""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE, lw=0.4)
    ax.set_global()
    return fig, ax


def _stipple_significant(ax, signif: xr.DataArray, max_points: int = 6000):
    """Subsampled scatter-stipple of significant cells onto an existing PlateCarree-
    transformed axes (no-op if `signif` has no True cells)."""
    ys, xs = np.where(np.asarray(signif.values))
    if len(xs) == 0:
        return
    step = max(1, len(xs) // max_points)
    ax.scatter(signif["longitude"].values[xs[::step]], signif["latitude"].values[ys[::step]],
               s=0.6, c="k", alpha=0.35, transform=ccrs.PlateCarree(), linewidths=0)


def _draw_multiline_band(ax, series_by_label: dict, band: str = "minmax"):
    """Draw one line per (label -> annual xr.DataArray) plus a shaded spread band
    ("minmax" or "iqr") across all of them, onto a plain (non-map) axes. Shared by
    multiline_timeseries and small_multiples_regional_timeseries."""
    labels = list(series_by_label)
    stacked = np.stack([np.asarray(series_by_label[l].values) for l in labels], axis=0)
    years = series_by_label[labels[0]]["year"].values
    if band == "minmax":
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    elif band == "iqr":
        lo, hi = np.percentile(stacked, 25, axis=0), np.percentile(stacked, 75, axis=0)
    else:
        lo = hi = None
    if lo is not None:
        ax.fill_between(years, lo, hi, color="gray", alpha=0.25, label=f"{band} spread")
    for l in labels:
        ax.plot(years, series_by_label[l].values, lw=1.3, label=l)


def _weighted_land_mean(da: xr.DataArray, land_only: bool = True) -> float:
    """cos(lat)-weighted global mean of a 2D (latitude,longitude) field."""
    if land_only:
        da = da.where(land_mask(da))
    w = np.cos(np.deg2rad(da["latitude"]))
    return float(da.weighted(w).mean(("latitude", "longitude")).values)


# --------------------------------------------------------------------------------- #
# Existing single-case figures (trend_map refactored onto the helpers above; behavior
# unchanged).
# --------------------------------------------------------------------------------- #

def trend_map(ds_case: xr.Dataset, title: str, out_png: Path,
              cmap="RdBu_r", vmax=None, units="events/decade", land_only=True):
    """Plot one trend case: filled Sen's slope + stippled significant cells."""
    slope = ds_case["slope"]
    signif = ds_case["signif"].astype(bool)
    if land_only:
        lm = land_mask(slope)
        slope = slope.where(lm)
        signif = signif & lm
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(slope.values), 98)) or 1.0

    fig, ax = _robinson_axes()
    p = slope.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-vmax, vmax=vmax,
        add_colorbar=False,
    )
    _stipple_significant(ax, signif)
    cb = fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7)
    cb.set_label(f"Sen's slope ({units})")
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def event_footprint_map(events_da: xr.DataArray, event_id: int, out_png: Path,
                         bbox=None, n_panels: int = 6):
    """Multi-panel regional map of one L7-tracked event's footprint across its
    lifetime (events_da = the `event_id` raster from chd.events.build_events).
    bbox = (lon_min, lon_max, lat_min, lat_max) in degrees; inferred from the
    event's own extent (+3 deg buffer) if not given. Uses a regional PlateCarree
    extent rather than the global Robinson projection used elsewhere in this module
    -- a single event's footprint isn't legible on a global map."""
    mask = (events_da == event_id)
    active_days = np.where(mask.any(dim=("latitude", "longitude")).values)[0]
    if len(active_days) == 0:
        raise ValueError(f"event_id {event_id} not present in events_da")

    if bbox is None:
        sub = mask.isel(time=active_days).any(dim="time")
        lat_idx, lon_idx = np.where(sub.values)
        lat_vals = events_da["latitude"].values[lat_idx]
        lon_vals = events_da["longitude"].values[lon_idx]
        buf = 3.0
        bbox = (float(lon_vals.min()) - buf, float(lon_vals.max()) + buf,
                float(lat_vals.min()) - buf, float(lat_vals.max()) + buf)

    panel_idx = np.linspace(0, len(active_days) - 1, min(n_panels, len(active_days))).astype(int)
    panel_days = sorted(set(active_days[panel_idx].tolist()))
    n = len(panel_days)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                              subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    axes = axes.ravel()
    for ax, day_idx in zip(axes, panel_days):
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.add_feature(cfeature.BORDERS, lw=0.3, alpha=0.5)
        day_mask = mask.isel(time=day_idx).astype(float)
        day_mask.where(day_mask > 0).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap="Reds", vmin=0, vmax=1,
            add_colorbar=False)
        ax.set_extent(bbox, crs=ccrs.PlateCarree())
        date_str = np.datetime_as_string(events_da["time"].values[day_idx], unit="D")
        ax.set_title(str(date_str), fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Event {event_id} footprint evolution")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def timeseries(compound: xr.DataArray, out_png: Path, label="compound event-months",
               region=None):
    """Global (land) area-weighted mean annual frequency time series."""
    da = compound
    lm = land_mask(da.isel(time=0))
    w = np.cos(np.deg2rad(da["latitude"]))
    # Mask AFTER summing, not before: `.where(lm)` then `.sum(skipna=True)` would let
    # every all-NaN (ocean) year collapse to a literal 0 rather than staying NaN,
    # silently re-admitting ocean as zero-valued "land" data into the weighted mean
    # below and diluting it by the land area fraction (~29% of the globe).
    annual = da.groupby("time.year").sum().where(lm)
    ts = annual.weighted(w).mean(("latitude", "longitude"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts["year"], ts.values, lw=1.5)
    ax.set_xlabel("year"); ax.set_ylabel(f"land-mean {label} / yr")
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S1 -- small-multiples trend map grid (spatial: "how do all N definitions compare
# at once", shared colorbar so panels are visually comparable).
# --------------------------------------------------------------------------------- #

def small_multiples_trend_map(ds_by_combo: dict, title: str, out_png: Path,
                               nrows: int = 2, ncols: int = 3, cmap="RdBu_r", vmax=None,
                               units="events/decade", land_only=True) -> Path:
    items = list(ds_by_combo.items())
    slopes, signifs = [], []
    for _, ds in items:
        s = ds["slope"]
        sg = ds["signif"].astype(bool)
        if land_only:
            lm = land_mask(s)
            s = s.where(lm)
            sg = sg & lm
        slopes.append(s)
        signifs.append(sg)
    if vmax is None:
        allvals = np.concatenate([np.abs(s.values[np.isfinite(s.values)]) for s in slopes])
        vmax = float(np.nanpercentile(allvals, 98)) if allvals.size else 1.0
        vmax = vmax or 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows),
                              subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)
    axes = axes.ravel()
    p = None
    for ax, (label, _ds), s, sg in zip(axes, items, slopes, signifs):
        ax.add_feature(cfeature.COASTLINE, lw=0.3)
        ax.set_global()
        p = s.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                               vmin=-vmax, vmax=vmax, add_colorbar=False)
        _stipple_significant(ax, sg)
        ax.set_title(" / ".join(label) if isinstance(label, tuple) else str(label), fontsize=9)
    for ax in axes[len(items):]:
        ax.axis("off")
    cb = fig.colorbar(p, ax=axes.tolist(), orientation="horizontal", pad=0.04, shrink=0.6)
    cb.set_label(f"Sen's slope ({units})")
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S2 -- per-pixel agreement/robustness map (IPCC-AR6-style categorical).
# --------------------------------------------------------------------------------- #

def agreement_map(agreement_da: xr.DataArray, title: str, out_png: Path, land_only=True) -> Path:
    codes = agreement_da.attrs.get(
        "codes", {0: "no signal", 1: "robust increase", 2: "robust decrease", 3: "mixed/conflicting"})
    da = agreement_da
    if land_only:
        da = da.where(land_mask(da))
    colors = ["#dddddd", "#b2182b", "#2166ac", "#f4a582"]   # no signal / inc / dec / mixed
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = _robinson_axes()
    p = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    cb = fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7, ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels([codes.get(i, str(i)) for i in range(4)], fontsize=8)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S3 -- pairwise spatial-correlation heatmap, hierarchically clustered.
# --------------------------------------------------------------------------------- #

def correlation_heatmap(corr: np.ndarray, labels: list, out_png: Path, cluster: bool = True) -> Path:
    """NxN correlation heatmap with optional hierarchical-clustering reorder. A row/col
    of NaN is possible and NOT necessarily a bug: a slope field that is a legitimate
    constant (e.g. exactly 0.0 everywhere) has zero variance, so Pearson r against it
    is undefined -- this genuinely happens for some (baseline, SPEI-calib) combos under
    seasonal (e.g. JJA) framings, where the underlying annual count series is sparse/
    tie-heavy enough that the Theil-Sen slope estimator (chd.trends._mk_pixel) can
    round to exactly 0 even at pixels where Mann-Kendall separately reports a
    significant trend -- a known property of median-of-pairwise-slopes estimators on
    zero-inflated discrete data, not something this function should paper over.
    Falls back to the given (unclustered) order and renders NaN cells as blank/"n/a"
    instead of crashing when any correlation is undefined."""
    has_nan = bool(np.isnan(corr).any())
    order = np.arange(len(labels))
    if cluster and len(labels) > 2 and not has_nan:
        dist = 1 - corr
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    elif cluster and has_nan:
        print("[correlation_heatmap] WARNING: NaN in correlation matrix (a constant/"
              "zero-variance slope field) -- skipping hierarchical clustering, using "
              "input order instead.", flush=True)
    corr_o = corr[np.ix_(order, order)]
    labels_o = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 2, 0.7 * len(labels) + 2))
    im = ax.imshow(corr_o, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels_o))); ax.set_xticklabels(labels_o, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels_o))); ax.set_yticklabels(labels_o, fontsize=8)
    for i in range(len(labels_o)):
        for j in range(len(labels_o)):
            v = corr_o[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}"
            color = "black" if np.isnan(v) else ("white" if abs(v) > 0.6 else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="spatial Pearson r")
    title = "Pairwise spatial correlation of Sen's slope fields"
    if has_nan:
        title += " (NaN = a constant slope field; see console warning)"
    ax.set_title(title, fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S4 -- pairwise difference map: left = combo A's slope, right = A minus B. Called
# once per contrast the driver script wants to isolate (e.g. HW-baseline effect,
# SPEI-calibration effect); this function itself does no axis-averaging.
# --------------------------------------------------------------------------------- #

def difference_map_pair(ds_a: xr.Dataset, ds_b: xr.Dataset, label_a: str, label_b: str,
                         title: str, out_png: Path, cmap="RdBu_r", vmax=None, land_only=True) -> Path:
    sa = ds_a["slope"]; sb = ds_b["slope"]
    diff = sa - sb
    if land_only:
        lm = land_mask(sa)
        sa = sa.where(lm); diff = diff.where(lm)
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(sa.values[np.isfinite(sa.values)]), 98)) if np.isfinite(sa.values).any() else 1.0
        vmax = vmax or 1.0
    if np.isfinite(diff.values).any():
        vmax_diff = float(np.nanpercentile(np.abs(diff.values[np.isfinite(diff.values)]), 98)) or 1.0
    else:
        vmax_diff = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), subplot_kw={"projection": ccrs.Robinson()})
    for ax, da, vm, ttl in zip(axes, [sa, diff], [vmax, vmax_diff],
                               [label_a, f"{label_a} minus {label_b}"]):
        ax.add_feature(cfeature.COASTLINE, lw=0.3)
        ax.set_global()
        p = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                                vmin=-vm, vmax=vm, add_colorbar=False)
        fig.colorbar(p, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        ax.set_title(ttl, fontsize=10)
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S5 -- variance decomposition: which axis (HW baseline vs SPEI calibration) drives
# more of the pixel-level spread in trend slope.
# --------------------------------------------------------------------------------- #

def variance_decomposition_map(decomp_ds: xr.Dataset, out_png: Path, land_only=True) -> Path:
    fields = ["frac_baseline", "frac_calib", "frac_interaction"]
    titles = ["HW baseline", "SPEI calibration", "Interaction + residual"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={"projection": ccrs.Robinson()})
    p = None
    for ax, f, ttl in zip(axes, fields, titles):
        da = decomp_ds[f]
        if land_only:
            da = da.where(land_mask(da))
        ax.add_feature(cfeature.COASTLINE, lw=0.3)
        ax.set_global()
        p = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis",
                                vmin=0, vmax=1, add_colorbar=False)
        ax.set_title(ttl, fontsize=10)
    cb = fig.colorbar(p, ax=axes.tolist(), orientation="horizontal", pad=0.06, shrink=0.6)
    cb.set_label("fraction of pixel-level slope variance explained")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def variance_decomposition_bar(decomp_ds: xr.Dataset, out_png: Path, land_only=True) -> Path:
    fields = ["frac_baseline", "frac_calib", "frac_interaction"]
    labels = ["HW baseline", "SPEI calibration", "Interaction+residual"]
    means = [_weighted_land_mean(decomp_ds[f], land_only=land_only) for f in fields]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, means, color=["#1b9e77", "#d95f02", "#7570b3"])
    ax.set_ylabel("global land-mean fraction of slope variance")
    ax.set_ylim(0, 1)
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S6/S9 -- multi-line time series with a spread band across definitions (temporal
# analogue of S1: one graph representing all N definitions together).
# --------------------------------------------------------------------------------- #

def multiline_timeseries(series_by_baseline_and_calib: dict, out_png: Path,
                          label: str = "compound event-months", band: str = "minmax") -> Path:
    baselines = list(series_by_baseline_and_calib)
    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 4), squeeze=False)
    axes = axes[0]
    for ax, hw_name in zip(axes, baselines):
        _draw_multiline_band(ax, series_by_baseline_and_calib[hw_name], band=band)
        ax.set_title(f"HW baseline {hw_name}")
        ax.set_xlabel("year"); ax.set_ylabel(f"land-mean {label} / yr")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def small_multiples_regional_timeseries(series_by_region: dict, out_png: Path,
                                         nrows: int = 3, ncols: int = 4, band: str = "minmax") -> Path:
    regions_list = list(series_by_region)
    n = len(regions_list)
    if nrows * ncols < n:
        nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, region_label in zip(axes, regions_list):
        _draw_multiline_band(ax, series_by_region[region_label], band=band)
        ax.set_title(region_label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S7 -- divergence-over-time: does definitional disagreement grow over the record.
# --------------------------------------------------------------------------------- #

def divergence_plot(div_da: xr.DataArray, label_a: str, label_b: str, out_png: Path) -> Path:
    years = div_da["year"].values
    vals = div_da.values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(years, vals, lw=1.3, color="darkred")
    if len(years) > 2:
        coeffs = np.polyfit(years, vals, 1)
        ax.plot(years, np.polyval(coeffs, years), lw=1, ls="--", color="k",
                label=f"linear fit: {coeffs[0]:.4f}/yr")
        ax.legend(fontsize=8)
    ax.set_xlabel("year"); ax.set_ylabel("|difference| in land-mean annual compound freq")
    ax.set_title(f"Divergence over time: {label_a} vs {label_b}")
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S8 -- regional summary heatmap (AR6 region x definition).
# --------------------------------------------------------------------------------- #

def regional_summary_heatmap(region_by_combo_df: pd.DataFrame, out_png: Path) -> Path:
    vmax = float(np.nanmax(np.abs(region_by_combo_df.values))) or 1.0
    fig, ax = plt.subplots(figsize=(0.9 * region_by_combo_df.shape[1] + 2,
                                     0.25 * region_by_combo_df.shape[0] + 2))
    im = ax.imshow(region_by_combo_df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(region_by_combo_df.shape[1]))
    ax.set_xticklabels(region_by_combo_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(region_by_combo_df.shape[0]))
    ax.set_yticklabels(region_by_combo_df.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="area-weighted mean Sen's slope (events/decade)", shrink=0.7)
    ax.set_title("Regional sensitivity: mean trend slope by AR6 region x definition")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S10 -- SPEI fit-validity diagnostic (month x calibration): a necessary caveat for
# any seasonal-sensitivity claim, not itself a climate-signal figure.
# --------------------------------------------------------------------------------- #

def fit_validity_heatmap(validity_pct: np.ndarray, calib_names: list, out_png: Path) -> Path:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(calib_names) + 2))
    im = ax.imshow(validity_pct, cmap="Reds", vmin=0,
                    vmax=max(1.0, float(np.nanmax(validity_pct))), aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(months)
    ax.set_yticks(range(len(calib_names))); ax.set_yticklabels(calib_names)
    for i in range(validity_pct.shape[0]):
        for j in range(validity_pct.shape[1]):
            ax.text(j, i, f"{validity_pct[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% land cells with invalid/masked SPEI-1 fit", shrink=0.8)
    ax.set_title("SPEI-1 fit validity by calendar month x calibration period")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S11 -- L7 event-level comparison (post Phase-0 matrix extension). Matplotlib-only
# (no seaborn dependency); S11c uses catalog centroids, never the daily event_id
# raster (infeasible at ~115 GB/store -- see docs/sensitivity risk notes).
# --------------------------------------------------------------------------------- #

def event_distribution_panels(catalog_all: pd.DataFrame, out_png: Path,
                               metrics=("duration_days", "cumulative_footprint_area_km2",
                                        "min_spei1", "max_heat_exceedance_c"),
                               land_frac_threshold: float = 0.5) -> Path:
    df = catalog_all[catalog_all["land_fraction"] >= land_frac_threshold].copy()
    df["combo"] = df["baseline"] + " / " + df["spei_calibration"]
    combos = sorted(df["combo"].unique())
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, len(combos)))
    for ax, metric in zip(axes, metrics):
        for c, combo in zip(colors, combos):
            vals = df.loc[df["combo"] == combo, metric].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, histtype="step", density=True, color=c, label=combo, lw=1.3)
        ax.set_title(metric, fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(metrics):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(combos), 3), fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Event characteristic distributions by definition (land_fraction>={land_frac_threshold})")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def event_count_bar(catalog_all: pd.DataFrame, out_png: Path, land_frac_threshold: float = 0.5) -> Path:
    df = catalog_all.copy()
    df["combo"] = df["baseline"] + " / " + df["spei_calibration"]
    combos = sorted(df["combo"].unique())
    total = df.groupby("combo").size().reindex(combos, fill_value=0)
    land = df[df["land_fraction"] >= land_frac_threshold].groupby("combo").size().reindex(combos, fill_value=0)
    x = np.arange(len(combos)); w = 0.35
    fig, ax = plt.subplots(figsize=(1.1 * len(combos) + 2, 4.5))
    ax.bar(x - w / 2, total.values, width=w, label="total events")
    ax.bar(x + w / 2, land.values, width=w, label=f"land-dominated (>={land_frac_threshold})")
    ax.set_xticks(x); ax.set_xticklabels(combos, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("event count")
    ax.legend()
    ax.set_title("L7 tracked event counts by definition")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def event_density_region_map(catalog_all: pd.DataFrame, region_da: xr.DataArray,
                              region_table: pd.DataFrame, out_png: Path,
                              nrows: int = 2, ncols: int = 3,
                              land_frac_threshold: float = 0.5) -> Path:
    """Small-multiples choropleth of L7 event count by AR6 region, one panel per
    (baseline, spei_calib) combo. Uses catalog bounding-box CENTROIDS looked up
    against an already-gridded AR6 region index (region_da, from
    chd.regions.ar6_land_regions()) via nearest-neighbor xarray indexing."""
    df = catalog_all[catalog_all["land_fraction"] >= land_frac_threshold].copy()
    df["combo"] = df["baseline"] + " / " + df["spei_calibration"]
    df["centroid_lat"] = (df["lat_min"] + df["lat_max"]) / 2
    df["centroid_lon"] = (df["lon_min"] + df["lon_max"]) / 2

    lat_idx = xr.DataArray(df["centroid_lat"].values, dims="event")
    lon_idx = xr.DataArray(df["centroid_lon"].values, dims="event")
    df["region_index"] = region_da.sel(latitude=lat_idx, longitude=lon_idx, method="nearest").values

    combos = sorted(df["combo"].unique())
    n = len(combos)
    if nrows * ncols < n:
        nrows = int(np.ceil(n / ncols))

    region_vals = np.asarray(region_da.values)
    valid = np.isfinite(region_vals)
    max_region_idx = int(np.nanmax(region_table["region_index"]))

    counts_by_combo = {}
    for combo in combos:
        counts_by_combo[combo] = df.loc[df["combo"] == combo, "region_index"].dropna().astype(int).value_counts()
    global_vmax = max((c.max() if len(c) else 0) for c in counts_by_combo.values()) or 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows),
                              subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)
    axes = axes.ravel()
    p = None
    for ax, combo in zip(axes, combos):
        lut = np.full(max_region_idx + 1, np.nan)
        for ridx, cnt in counts_by_combo[combo].items():
            lut[ridx] = cnt
        count_arr = np.full(region_vals.shape, np.nan)
        count_arr[valid] = lut[region_vals[valid].astype(int)]
        count_grid = xr.DataArray(count_arr, dims=region_da.dims, coords=region_da.coords)

        ax.add_feature(cfeature.COASTLINE, lw=0.3)
        ax.set_global()
        p = count_grid.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="YlOrRd",
                                        vmin=0, vmax=global_vmax, add_colorbar=False)
        ax.set_title(combo, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    cb = fig.colorbar(p, ax=axes.tolist(), orientation="horizontal", pad=0.06, shrink=0.6)
    cb.set_label(f"tracked event count by AR6 region (land_fraction>={land_frac_threshold})")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------- #
# S12 (optional/lower priority) -- Taylor diagram: correlation + relative std vs a
# reference definition, in one summary plot. No off-the-shelf primitive in this
# repo's dependencies -- custom polar axes.
# --------------------------------------------------------------------------------- #

def taylor_diagram(stats: list, reference_label: str, out_png: Path) -> Path:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(-1)
    max_r = max([s["std_ratio"] for s in stats] + [1.2])
    ax.set_rlim(0, max_r)
    corr_ticks = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    ax.set_thetagrids([np.degrees(np.arccos(c)) for c in corr_ticks],
                       labels=[str(c) for c in corr_ticks])
    ax.set_rlabel_position(135)
    ax.plot(0, 1.0, "k*", markersize=14, label=reference_label)
    for s in stats:
        theta = np.arccos(np.clip(s["corr"], -1, 1))
        ax.plot(theta, s["std_ratio"], "o", markersize=8, label=s["label"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    ax.set_title("Taylor diagram: spatial pattern similarity to the reference definition")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
