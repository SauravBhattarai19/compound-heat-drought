"""Manuscript figures for the SINGLE locked compound definition:
heatwave baseline 1961-1990, SPEI-1, SPEI calibration 1950-2025 (full record).

There is deliberately no cross-baseline / cross-calibration sensitivity here -- that is
the S1-S12 battery (chd.sensitivity/regions + scripts/09,10). This module produces the
paper's primary result figures for the one chosen definition. Dryland context (aridity
index + Köppen deserts) comes from chd.drylands.

Outputs -> figures/manuscript/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, RegularPolygon
import matplotlib.patheffects as pe
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde, linregress, rankdata, theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C
from chd import viz, regions, drylands, leading, enso, agriculture, trends as trendsmod

# past / recent 38-yr periods for the onset-order response figures
LEAD_PAST = (1950, 1987)
LEAD_RECENT = (1988, 2025)
HEAT_COLOR = "#c0392b"       # heat-leading (warm)
DROUGHT_COLOR = "#e0a800"    # drought-leading (amber)

# --------------------------------------------------------------------------- #
# The one locked definition (see memory: manuscript-fixed-definition).
# --------------------------------------------------------------------------- #
HW = "1961_1990"
CALIB = "1950_2025"
TS = C.SPEI_COMPOUND_TS          # 1
CASE = "1950_2025__annual"       # full-record annual trend window
UNITS = "compound event-months decade$^{-1}$"
# early vs late 30-yr windows for the absolute-change maps
EARLY_PERIOD = (1950, 1979)
LATE_PERIOD = (1996, 2025)
FIGDIR = C.FIG_DIR / "manuscript"

CMAP = plt.get_cmap("RdBu_r").copy()
GREY = ListedColormap(["#c9c7c2"])       # SPEI-undefined land
LAND_FC = "#fbfaf7"
DESERT_LINE = "#6b4f1d"
# hyper-arid -> humid, dark brown -> green
AI_COLORS = ["#7a3b10", "#c07a2a", "#d9b25b", "#9fbf6a", "#3f7a3f"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 220,
})


# --------------------------------------------------------------------------- #
# Data access (single combo -> all reads are small/fast)
# --------------------------------------------------------------------------- #
ANTARCTIC_LAT = -60.0   # land south of this is excluded from every aggregate (2026-09-23)


def _analysis_land(like: xr.DataArray) -> xr.DataArray:
    """Land mask used for EVERY statistic/map in the paper: Natural Earth land minus
    Antarctica (lat < -60). The AR6 tables already excluded the Antarctic regions (WAN/EAN)
    because a P-PET 'drought' over an ice sheet is not meaningful; before 2026-09-23 the
    global land mean, aridity classes and maps silently still included it (8.3% of
    cos-weighted land). Greenland/Iceland (GIC) is kept, consistent with the AR6 tables."""
    lm = viz.land_mask(like)
    return lm & (like["latitude"] > ANTARCTIC_LAT)


def _pettitt(vals, years):
    """Pettitt change-point year and its approximate two-sided p-value
    (p ~ 2 exp(-6K^2/(n^3+n^2)), Pettitt 1979; assumes independence, so optimistic under
    autocorrelation)."""
    x = np.asarray(vals, float); m = np.isfinite(x)
    x, yy = x[m], np.asarray(years)[m]
    n = len(x)
    r = rankdata(x)
    U = np.array([2 * np.sum(r[:k]) - k * (n + 1) for k in range(1, n + 1)])
    K = np.abs(U).max()
    return int(yy[int(np.argmax(np.abs(U)))]), float(min(1.0, 2 * np.exp(-6 * K**2 / (n**3 + n**2))))


def _ts(y, yrs):
    """Theil-Sen slope per decade."""
    return float(theilslopes(np.asarray(y, float), np.asarray(yrs, float))[0] * 10)


def _hr_p(y):
    import pymannkendall as pmk
    return float(pmk.hamed_rao_modification_test(np.asarray(y, float)).p)


def _mk_p(y):
    import pymannkendall as pmk
    return float(pmk.original_test(np.asarray(y, float)).p)


def _open_trend(var: str) -> xr.Dataset:
    baseline = C.HW_BASELINES[HW]
    calib = C.SPEI_CALIB_PERIODS[CALIB]
    ds = xr.open_zarr(C.trends_zarr(baseline, TS, calib, var))
    idx = list(ds.attrs["case_names"]).index(CASE)
    return ds.isel(case=idx)[["slope", "signif", "z"]].load()


def _land_mean_annual(var: str = "compound") -> pd.Series:
    baseline = C.HW_BASELINES[HW]
    calib = C.SPEI_CALIB_PERIODS[CALIB]
    da = xr.open_zarr(C.compound_zarr(baseline, TS, calib))[var]
    lm = _analysis_land(da.isel(time=0))
    w = np.cos(np.deg2rad(da["latitude"]))
    annual = trendsmod.annual_counts(da, C.TREND_WINDOWS["1950_2025"]).where(lm)
    ts = annual.weighted(w).mean(("latitude", "longitude")).compute()
    return pd.Series(ts.values, index=ts["year"].values)


class _Ctx:
    """Grids/masks shared by every figure, built once."""
    def __init__(self):
        comp = _open_trend("compound")
        self.lat = comp["latitude"].values
        self.lon = comp["longitude"].values
        self.slope = {"compound": comp["slope"]}
        self.signif = {"compound": comp["signif"].astype(bool)}
        self.land = _analysis_land(comp["slope"])
        self.z = {"compound": comp["z"]}
        km = drylands.koppen_desert_mask(comp["slope"])
        self.desert = km["desert"].values
        self.arid_all = km["arid_all"].values
        ai = drylands.aridity_index().reindex(
            latitude=self.lat, longitude=self.lon, method="nearest")
        self.ai = ai.values
        self.lon2d, self.lat2d = np.meshgrid(self.lon, self.lat)
        self._annual = {}

    def annual_counts(self, var="compound"):
        """Per-year absolute count (event-months in year), full record, from L5.
        Cached per var -- one L5 read serves mean_annual / period means / change."""
        if var not in self._annual:
            baseline = C.HW_BASELINES[HW]
            calib = C.SPEI_CALIB_PERIODS[CALIB]
            da = xr.open_zarr(C.compound_zarr(baseline, TS, calib))[var]
            self._annual[var] = trendsmod.annual_counts(da, C.TREND_WINDOWS["1950_2025"]).compute()
        return self._annual[var]

    def mean_annual(self, var="compound"):
        """Climatological ABSOLUTE mean annual count (event-months yr^-1), full record."""
        return self.annual_counts(var).mean("year")

    def mean_annual_period(self, var, y0, y1):
        return self.annual_counts(var).sel(year=slice(y0, y1)).mean("year")

    def change_field(self, var, early, late):
        """Late-minus-early change in mean annual count (absolute, event-months yr^-1)."""
        return self.mean_annual_period(var, *late) - self.mean_annual_period(var, *early)

    # -- onset-order (heat-leading / drought-leading) ---------------------- #
    def leading_cat(self):
        if getattr(self, "_lead_cat", None) is None:
            self._lead_cat = leading.classify_from_l5(
                C.HW_BASELINES[HW], TS, C.SPEI_CALIB_PERIODS[CALIB])
        return self._lead_cat

    def leading_annual(self):
        if getattr(self, "_lead_ann", None) is None:
            self._lead_ann = leading.annual_type_counts(self.leading_cat())
        return self._lead_ann

    def leading_dominance(self):
        if getattr(self, "_lead_dom", None) is None:
            self._lead_dom = leading.dominance(self.leading_cat())
        return self._lead_dom

    def leading_extent_series(self, typ, region_da=None, region_idx=None):
        """Annual spatial extent (% of land area) affected by onset type `typ`
        (`heat_leading`/`drought_leading`/`concurrent`), globally or within one AR6
        region. Extent = area-weighted fraction of land cells with >=1 such month."""
        ann = self.leading_annual()[typ]
        mask = self.land
        if region_da is not None:
            mask = mask & (region_da == region_idx)
        w = np.cos(np.deg2rad(ann["latitude"]))
        aff = (ann > 0).where(mask)
        ext = aff.weighted(w).mean(("latitude", "longitude")) * 100.0
        return pd.Series(ext.values, index=ext["year"].values)

    def region_annual_mean(self, var, region_da, region_idx):
        """Area-weighted regional mean of the annual count of `var`
        (pd.Series year -> mean count, e.g. compound event-months per year)."""
        ann = self.annual_counts(var)
        mask = self.land & (region_da == region_idx)
        w = np.cos(np.deg2rad(ann["latitude"]))
        s = ann.where(mask).weighted(w).mean(("latitude", "longitude"))
        return pd.Series(s.values, index=s["year"].values)

    def slope_of(self, var):
        if var not in self.slope:
            ds = _open_trend(var)
            self.slope[var] = ds["slope"]
            self.signif[var] = ds["signif"].astype(bool)
        return self.slope[var]

    # -- agricultural context (MIRCA2000 cropland + growing season) --------- #
    def crop(self):
        """MIRCA2000 cropland-exposure layers on the ctx grid (frac_total/rainfed/
        irrigated, rainfed_fraction, cropland_mask)."""
        if getattr(self, "_crop", None) is None:
            self._crop = agriculture.cropland_layers(self.slope_of("compound"))
        return self._crop

    def gsmask(self):
        """MIRCA2000 per-pixel growing-season month mask (gs_any/gs_rainfed)."""
        if getattr(self, "_gsmask", None) is None:
            self._gsmask = agriculture.growing_season_mask(self.slope_of("compound"))
        return self._gsmask

    def gs_compound_trend(self):
        """Per-pixel MK/Theil-Sen trend of the compound count restricted to each pixel's
        growing season (slope per decade + signif), computed on the fly from L5."""
        if getattr(self, "_gstrend", None) is None:
            base = C.HW_BASELINES[HW]
            calib = C.SPEI_CALIB_PERIODS[CALIB]
            comp = xr.open_zarr(C.compound_zarr(base, TS, calib))["compound"]
            self._gstrend = trendsmod.trends_for(
                comp, C.TREND_WINDOWS["1950_2025"], gs_month_mask=self.gsmask()["gs_any"])
        return self._gstrend

    def compound_l5(self):
        if getattr(self, "_compl5", None) is None:
            base = C.HW_BASELINES[HW]
            calib = C.SPEI_CALIB_PERIODS[CALIB]
            self._compl5 = xr.open_zarr(C.compound_zarr(base, TS, calib))["compound"]
        return self._compl5

    def land_slope(self, var):
        return self.slope_of(var).where(self.land)

    def vmax(self, var, pct=98):
        v = np.abs(self.land_slope(var).values)
        v = v[np.isfinite(v)]
        return float(np.nanpercentile(v, pct)) or 1.0


# --------------------------------------------------------------------------- #
# Shared map primitives
# --------------------------------------------------------------------------- #
def _basemap(ax, extent=None):
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_aspect("auto")
    ax.add_feature(cfeature.LAND, facecolor=LAND_FC, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)


def _nan_land(ax, ctx, var):
    nan_land = ctx.land.values & ~np.isfinite(ctx.slope_of(var).values)
    g = np.where(nan_land, 1.0, np.nan)
    ax.pcolormesh(ctx.lon, ctx.lat, g, transform=ccrs.PlateCarree(), cmap=GREY,
                  vmin=0, vmax=1, zorder=1.4, rasterized=True)


def _slope_mesh(ax, ctx, var, vmax):
    da = ctx.land_slope(var)
    m = ax.pcolormesh(ctx.lon, ctx.lat, da.values, transform=ccrs.PlateCarree(),
                      cmap=CMAP, vmin=-vmax, vmax=vmax, zorder=1, rasterized=True)
    _nan_land(ax, ctx, var)
    return m


def _stipple(ax, ctx, var, extent=None, target=4500, size=0.35):
    ctx.slope_of(var)                      # ensures ctx.signif[var] is populated
    ys, xs = np.where(ctx.signif[var].values & ctx.land.values)
    if extent is not None:
        lo0, lo1, la0, la1 = extent
        keep = (ctx.lon[xs] >= lo0) & (ctx.lon[xs] <= lo1) & (ctx.lat[ys] >= la0) & (ctx.lat[ys] <= la1)
        xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return
    step = max(1, len(xs) // target)
    ax.scatter(ctx.lon[xs[::step]], ctx.lat[ys[::step]], s=size, c="k", alpha=0.32,
               transform=ccrs.PlateCarree(), linewidths=0, zorder=2)


def _stipple_arr(ax, ctx, signif, extent=None, target=4500, size=0.35):
    ys, xs = np.where(np.asarray(signif) & ctx.land.values)
    if extent is not None:
        lo0, lo1, la0, la1 = extent
        keep = (ctx.lon[xs] >= lo0) & (ctx.lon[xs] <= lo1) & (ctx.lat[ys] >= la0) & (ctx.lat[ys] <= la1)
        xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return
    step = max(1, len(xs) // target)
    ax.scatter(ctx.lon[xs[::step]], ctx.lat[ys[::step]], s=size, c="k", alpha=0.32,
               transform=ccrs.PlateCarree(), linewidths=0, zorder=2)


def _nan_land_da(ax, ctx, da):
    nan_land = ctx.land.values & ~np.isfinite(np.asarray(da.values))
    g = np.where(nan_land, 1.0, np.nan)
    ax.pcolormesh(ctx.lon, ctx.lat, g, transform=ccrs.PlateCarree(), cmap=GREY,
                  vmin=0, vmax=1, zorder=1.4, rasterized=True)


# =========================================================================== #
# Field registry + generic map/zoom engine (the reusable "zoom-anywhere" plan).
# Any field below can be drawn globally or over any named ZOOM_REGION.
# =========================================================================== #
def resolve_field(ctx: "_Ctx", key: str) -> dict:
    """Return everything needed to draw a field: its 2D DataArray, optional
    significance, colour scaling and labels. `key` is '<var>_trend', '<var>_mean'
    (var in compound/heat/drought/heat_only/drought_only) or 'aridity'."""
    if key == "aridity":
        da = xr.DataArray(ctx.ai, coords={"latitude": ctx.lat, "longitude": ctx.lon},
                          dims=("latitude", "longitude"))
        norm = BoundaryNorm([0] + drylands.AI_CLASS_EDGES[1:-1] + [1.5], len(AI_COLORS))
        return dict(data=da, signif=None, cmap=ListedColormap(AI_COLORS), norm=norm,
                    stipple=False, label="Aridity Index  P/PET",
                    title="Aridity Index (UNEP class)", sequential=True)
    var, _, kind = key.rpartition("_")
    if kind == "trend":
        da = ctx.slope_of(var)
        vmax = ctx.vmax(var)
        unit = UNITS if var == "compound" else "months decade$^{-1}$"
        return dict(data=da, signif=ctx.signif[var].values, cmap=CMAP, norm=None,
                    vmin=-vmax, vmax=vmax, stipple=True,
                    label=f"Sen's slope ({unit})", title=f"{var} trend", sequential=False)
    if kind == "mean":
        da = ctx.mean_annual(var)
        v = np.asarray(da.where(ctx.land).values)
        vmax = float(np.nanpercentile(v[np.isfinite(v)], 98)) or 1.0
        unit = "event-months yr$^{-1}$" if var == "compound" else "months yr$^{-1}$"
        return dict(data=da, signif=None, cmap="YlOrRd", norm=None, vmin=0, vmax=vmax,
                    stipple=False, label=f"mean annual {var} ({unit})",
                    title=f"{var} climatology", sequential=True)
    if kind == "change":
        da = ctx.change_field(var, EARLY_PERIOD, LATE_PERIOD)
        v = np.abs(np.asarray(da.where(ctx.land).values))
        vmax = float(np.nanpercentile(v[np.isfinite(v)], 98)) or 1.0
        unit = "event-months yr$^{-1}$" if var == "compound" else "months yr$^{-1}$"
        lbl = (f"Δ mean annual {var} ({unit}), "
               f"{LATE_PERIOD[0]}–{LATE_PERIOD[1]} − {EARLY_PERIOD[0]}–{EARLY_PERIOD[1]}")
        return dict(data=da, signif=None, cmap=CMAP, norm=None, vmin=-vmax, vmax=vmax,
                    stipple=False, label=lbl, title=f"{var} change", sequential=False)
    raise KeyError(f"unknown field key {key!r}")


def _draw_field(ax, ctx, field, extent=None, desert=True, ai_hatch=False,
                stipple_target=4500, stipple_size=0.35, coast=True):
    _basemap(ax, extent)
    da = field["data"].where(ctx.land)
    kw = dict(transform=ccrs.PlateCarree(), cmap=field["cmap"], zorder=1, rasterized=True)
    if field.get("norm") is not None:
        kw["norm"] = field["norm"]
    else:
        kw["vmin"], kw["vmax"] = field["vmin"], field["vmax"]
    m = ax.pcolormesh(ctx.lon, ctx.lat, da.values, **kw)
    _nan_land_da(ax, ctx, field["data"])
    if ai_hatch:
        hyper = np.where((ctx.ai < 0.05) & ctx.land.values, 1.0, np.nan)
        ax.contourf(ctx.lon2d, ctx.lat2d, np.nan_to_num(hyper), levels=[0.5, 1.5],
                    colors="none", hatches=["xxxx"], transform=ccrs.PlateCarree(), zorder=2)
    if desert:
        lw = 0.6 if extent is not None else 0.35
        ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
                   colors=DESERT_LINE, linewidths=lw, transform=ccrs.PlateCarree(), zorder=3)
    if field.get("stipple") and field.get("signif") is not None:
        _stipple_arr(ax, ctx, field["signif"], extent=extent, target=stipple_target, size=stipple_size)
    if coast:
        ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=5)
        if extent is not None:
            ax.add_feature(cfeature.BORDERS, lw=0.25, alpha=0.5, zorder=5)
    return m


# Per-pixel dark/light contrast text was not enough: the hyper-arid hatch is a solid
# BLACK crosshatch, so black text vanishes into it and white text vanishes into the
# lighter gaps between hatch lines -- no single fill colour wins there. A bold white
# halo around a dark fill is the standard cartographic fix: it silhouettes the glyph
# against hatching, dark maroon trend cells, AND light grey cells all at once, so every
# inline label uses it rather than trying to pick a colour per background.
_LABEL_HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# A few AR6 neighbours are narrow enough (Southern Africa, southern South America)
# that their centroids sit closer together than any reasonable font can avoid
# colliding, and Madagascar/the Patagonian tip are too small/narrow for even a haloed
# label to sit inside cleanly -- push just these out to open water/lighter land with a
# short arrow back to the region instead of shrinking every label to accommodate the
# tightest handful. (dlon, dlat) degree offsets, hand-tuned against the render.
_LABEL_NUDGE = {
    "WSAF": (-4.5, -3.0), "ESAF": (3.0, -6.5),
    "SWS": (-4.5, -3.0), "SES": (5.5, 2.5), "SSA": (-9.0, -3.5),
    "MDG": (8.5, 1.0),
}


def _ar6_boundaries(ax, *, extent=None, edgecolor="0.15", linewidth=0.5,
                    label=True, label_fontsize=12.0, label_color="0.15", zorder=6,
                    field=None, ctx=None, nudge=None, halo=False):
    """Draw AR6 land-region outlines (no fill) + abbreviation labels on top of an
    existing GeoAxes that already has a pcolormesh field drawn (Koppen desert contour
    at zorder=3, coastline at zorder=5 in _draw_field) -- zorder=6/7 here sits above
    all of that. Land-clipped via _ar6_clipped() so boundaries never cross open ocean.
    Pass `halo=True` to outline every label in white (see _LABEL_HALO) for legibility
    over a busy data-coloured background. Pass `nudge=_LABEL_NUDGE` (a dict of abbrev
    -> (dlon, dlat) degree offsets) to displace a handful of unavoidably-crowded/tiny
    labels off their centroid, connected back to it by an arrow."""
    ar6 = _ar6_geoms()
    clip, rep = _ar6_clipped()
    for ab in ar6.abbrevs:
        if ab in _EXCLUDE_REGIONS:
            continue
        geoms = _polygonal(clip[ab])
        if not geoms:
            continue
        ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none",
                          edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
        if label:
            lon, lat = rep[ab]
            if extent is not None:
                lo0, lo1, la0, la1 = extent
                if not (lo0 <= lon <= lo1 and la0 <= lat <= la1):
                    continue
            if nudge is not None and ab in nudge:
                dlon, dlat = nudge[ab]
                tlon, tlat = lon + dlon, lat + dlat
                pt = ccrs.PlateCarree()._as_mpl_transform(ax)
                txt = ax.annotate(ab, xy=(lon, lat), xycoords=pt, xytext=(tlon, tlat),
                           textcoords=pt, fontsize=label_fontsize, fontweight="bold",
                           color=label_color, ha="center", va="center", zorder=zorder + 1,
                           arrowprops=dict(arrowstyle="-|>", color="0.1", lw=0.9,
                                          shrinkA=1, shrinkB=6))
                if halo:
                    txt.set_path_effects(_LABEL_HALO)
            else:
                txt = ax.text(lon, lat, ab, transform=ccrs.PlateCarree(), fontsize=label_fontsize,
                              fontweight="bold", color=label_color, ha="center", va="center",
                              zorder=zorder + 1)
                if halo:
                    txt.set_path_effects(_LABEL_HALO)


# Extensible region registry -- add (lon0, lon1, lat0, lat1) entries freely.
ZOOM_REGIONS = {
    # drylands
    "Sahara / Sahel": (-17, 40, 8, 33),
    "Arabian & Middle East": (33, 62, 12, 37),
    "Central Asia / Gobi": (52, 112, 30, 52),
    "Australian interior": (113, 150, -35, -16),
    "SW North America": (-120, -98, 24, 42),
    "Kalahari / S. Africa": (10, 32, -35, -15),
    # broader hotspots
    "Mediterranean": (-10, 42, 28, 47),
    "Horn of Africa": (32, 55, -5, 20),
    "South Asia": (66, 92, 6, 32),
    "East Asia / N. China": (100, 135, 25, 48),
    "SE South America": (-70, -45, -40, -18),
    "Amazonia": (-80, -45, -18, 8),
    "SE Australia": (135, 154, -40, -25),
    "Western US": (-125, -100, 30, 49),
    "Southern Europe / Balkans": (10, 32, 34, 48),
    "Greater Sahel belt": (-18, 45, 8, 20),
}
REGION_GROUPS = {
    "drylands": ["Sahara / Sahel", "Arabian & Middle East", "Central Asia / Gobi",
                 "Australian interior", "SW North America", "Kalahari / S. Africa"],
    "hotspots": ["Mediterranean", "Horn of Africa", "South Asia", "SE South America",
                 "East Asia / N. China", "SE Australia"],
}


def zoom_figure(ctx: "_Ctx", field_key: str, region_names, out: Path, *, ncols: int = 3,
                title: str = None, subtitle: str = None, desert: bool = True,
                ai_hatch: bool = False, legend: bool = True) -> Path:
    """Draw one field over an arbitrary list of named regions -- the general-purpose
    'zoom into many places to show inner detail' tool. `field_key` -> resolve_field()."""
    field = resolve_field(ctx, field_key)
    names = list(region_names)
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.4 * nrows + 1.1),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    axes = axes.ravel()
    m = None
    for k, (ax, name) in enumerate(zip(axes, names)):
        bbox = ZOOM_REGIONS[name]
        m = _draw_field(ax, ctx, field, extent=bbox, desert=desert, ai_hatch=ai_hatch,
                        stipple_target=2500, stipple_size=1.2)
        gl = ax.gridlines(draw_labels=True, lw=0.2, color="0.7", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        ax.set_title(name, fontsize=11, fontweight="semibold")
        ax.text(0.02, 0.96, "abcdefghijkl"[k], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", path_effects=[pe.withStroke(linewidth=2, foreground="w")])
    for ax in axes[n:]:
        ax.axis("off")

    cb = fig.colorbar(m, ax=axes.tolist(), orientation="horizontal", pad=0.04,
                      shrink=0.5, aspect=40, extend="both" if not field["sequential"] else "max")
    cb.set_label(field["label"], fontsize=9.5)
    if legend:
        el = []
        if desert:
            el.append(Line2D([0], [0], color=DESERT_LINE, lw=1.4, label="Köppen desert (BW)"))
        if ai_hatch:
            el.append(Patch(facecolor="none", edgecolor="k", hatch="xxxx", label="hyper-arid AI<0.05"))
        if field.get("stipple"):
            el.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=5,
                             label="MK-significant (p<0.05)", lw=0))
        el.append(Patch(facecolor=GREY(0.0), label="SPEI-undefined"))
        fig.legend(handles=el, loc="lower center", ncol=len(el), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
    if title:
        fig.suptitle(title + (f"\n{subtitle}" if subtitle else ""), fontsize=13.5, y=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIGURE 1 -- primary compound-trend composite
# --------------------------------------------------------------------------- #
def compound_overview(ctx: _Ctx) -> Path:
    fig = plt.figure(figsize=(14, 13.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 0.92, 0.78],
                  hspace=0.34, wspace=0.16, left=0.065, right=0.95, top=0.94, bottom=0.06)

    # (a) hero compound trend map -----------------------------------------
    vmax = ctx.vmax("compound")
    ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    _basemap(ax)
    m = _slope_mesh(ax, ctx, "compound", vmax)
    ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
               colors=DESERT_LINE, linewidths=0.35, transform=ccrs.PlateCarree(), zorder=3)
    _stipple(ax, ctx, "compound")
    ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
    _letter(ax, "a")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.72, pad=0.015, extend="both")
    cb.set_label(f"Sen's slope ({UNITS})", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    # (b) global land-mean time series ------------------------------------
    ax_ts = fig.add_subplot(gs[1, 0])
    ts = _land_mean_annual("compound")
    yrs = ts.index.values
    ax_ts.plot(yrs, ts.values, lw=1.6, color="#b2182b")
    b1, b0 = np.polyfit(yrs, ts.values, 1)
    ax_ts.plot(yrs, b1 * yrs + b0, ls="--", lw=1.1, color="k",
               label=f"OLS +{b1*10:.4f} decade$^{{-1}}$")
    ax_ts.set_xlim(yrs.min(), yrs.max())
    ax_ts.set_xlabel("year")
    ax_ts.set_ylabel("global land-mean\ncompound event-months yr$^{-1}$")
    ax_ts.grid(alpha=0.25, lw=0.5)
    ax_ts.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax_ts.set_title("Global land-mean compound frequency", fontsize=11)
    _letter(ax_ts, "b", inside=False)

    # (c) ridgeline of pixel slope by aridity class -----------------------
    ax_rg = fig.add_subplot(gs[1, 1])
    _ridge_by_aridity(ax_rg, ctx)
    _letter(ax_rg, "c", inside=False)

    # (d) ranked AR6 regional bar -----------------------------------------
    ax_bar = fig.add_subplot(gs[2, :])
    _ar6_ranked_bar(ax_bar, ctx)
    _letter(ax_bar, "d", inside=False, dx=-0.045)

    fig.suptitle("Trend in annual compound heatwave–drought frequency, 1950–2025\n"
                 "HW 1961–1990 · SPEI-1 · calibration 1950–2025   "
                 "(brown = Köppen desert (BW), grey land = SPEI-undefined)",
                 fontsize=13, y=0.975)
    out = FIGDIR / "compound_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _ridge_by_aridity(ax, ctx):
    q = ctx.vmax("compound", pct=99)
    xs = np.linspace(-q, q, 400)
    rng = np.random.default_rng(0)
    slope = ctx.slope_of("compound").values
    landv = ctx.land.values
    aiv = ctx.ai
    row_gap = 1.0
    n = len(drylands.AI_CLASS_LABELS)
    for i in range(n):            # i=0 hyper-arid (bottom) .. humid (top)
        base = i * row_gap
        lo, hi = drylands.AI_CLASS_EDGES[i], drylands.AI_CLASS_EDGES[i + 1]
        mask = landv & np.isfinite(aiv) & (aiv >= lo) & (aiv < hi)
        v = slope[mask]; v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        if len(v) > 40000:
            v = rng.choice(v, 40000, replace=False)
        kde = gaussian_kde(v)
        d = kde(xs); d = d / d.max() * 0.92
        col = AI_COLORS[i]
        ax.fill_between(xs, base, base + d, color=col, alpha=0.6, lw=0, zorder=i)
        ax.plot(xs, base + d, color=col, lw=1.1, zorder=i)
        ax.plot([np.median(v)], [base], marker="|", ms=10, color="k", zorder=i + 0.5)
        ax.text(-q * 0.95, base + 0.62, drylands.AI_CLASS_LABELS[i], color=col,
                fontsize=8.5, fontweight="semibold", va="center")
        ax.axhline(base, color="0.7", lw=0.4, zorder=0)
    ax.axvline(0, color="0.35", lw=0.7, ls=":")
    ax.set_yticks([])
    ax.set_xlim(-q, q)
    ax.set_ylim(-0.2, (n - 1) * row_gap + 1.05)
    ax.set_xlabel(f"per-pixel Sen's slope ({UNITS})", fontsize=9)
    ax.set_title("Pixel-trend distribution by aridity class\n(AI = P/PET; “|” = median)",
                 fontsize=10.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _ar6_ranked_bar(ax, ctx):
    reg_da = regions.ar6_land_regions(ctx.slope_of("compound"))
    table = regions.ar6_region_table()
    da = ctx.slope_of("compound")
    vals = np.array([regions.area_weighted_regional_stat(da, reg_da, int(r), land_only=True)
                     for r in table["region_index"].values])
    order = np.argsort(vals)
    vals_o = vals[order]
    abbr = table["abbrev"].values[order]
    vmax = float(np.nanpercentile(np.abs(vals_o[np.isfinite(vals_o)]), 100)) or 1.0
    norm = Normalize(-vmax, vmax)
    colors = cm.RdBu_r(norm(vals_o))
    x = np.arange(len(vals_o))
    ax.bar(x, vals_o, color=colors, edgecolor="0.3", lw=0.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(abbr, rotation=90, fontsize=6.4)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylabel(f"area-weighted mean slope\n({UNITS})", fontsize=9)
    ax.set_title("Compound trend by AR6 land region (sorted)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.4)


# --------------------------------------------------------------------------- #
# FIGURE 2 -- dryland zoom insets (a thin call into the generic zoom engine)
# --------------------------------------------------------------------------- #
def desert_zooms(ctx: _Ctx) -> Path:
    return zoom_figure(
        ctx, "compound_trend", REGION_GROUPS["drylands"],
        FIGDIR / "compound_desert_zooms.png", ncols=3, ai_hatch=True,
        title="Compound heatwave–drought trend in the world's drylands "
              "(HW 1961–1990, SPEI-1, calibration 1950–2025)",
        subtitle="Desert cores are largely SPEI-undefined; the signal lives on the arid margins")


# --------------------------------------------------------------------------- #
# FIGURE 1 -- orientation figure: global climatology + trend, with AR6 region
# boundaries AND dryland (Koppen/hyper-arid) context overlaid on the SAME map.
# Consolidates what used to be three separate figures (compound_climatology,
# aridity_koppen_overlay, compound_overview's hero panel) into one, and gives every
# later AR6-region figure a map the reader can decode abbreviations against.
# --------------------------------------------------------------------------- #
def global_overview_ar6(ctx: _Ctx) -> Path:
    """The paper's orientation figure. Two data panels, both at native pixel resolution:
    (a) mean annual compound frequency, (b) Theil-Sen trend (MK-significant cells
    stippled). Both carry the Koppen desert/steppe boundary and hyper-arid cross-
    hatching; the 44 (non-Antarctic) IPCC AR6 land-region boundaries + abbreviation
    labels are carried on (b) only (labelling both panels was redundant -- the
    boundaries are identical -- and (a) is almost uniformly dark, leaving nowhere to
    put readable text). Most labels sit directly on their region -- (b) is mostly
    light/grey (SPEI-undefined) so plain dark text reads fine there. The handful that
    sit on the map's few genuinely dark cells, or that are too small/crowded to fit a
    label at all (checked empirically, see _LABEL_NUDGE), are pushed out to nearby
    open water/lighter land with a short arrow back to the region instead."""
    fig = plt.figure(figsize=(16, 6.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.05, left=0.02, right=0.98,
                 top=0.93, bottom=0.11)
    fmean = resolve_field(ctx, "compound_mean")
    # (b) was the per-pixel Theil-Sen slope until 2026-09-23. On zero-inflated monthly
    # counts that slope is exactly 0 on ~70% of land (median of pairwise slopes), so the
    # panel read as almost blank while ~60% of land is MK-significant. The late-minus-
    # early change is estimator-free and in physical units; stipples mark MK-significant
    # INCREASES (p<0.05, z>0), the per-pixel significance criterion used throughout.
    ftrend = resolve_field(ctx, "compound_change")
    ftrend["signif"] = (ctx.signif["compound"].values & (ctx.z["compound"].values > 0))
    ftrend["stipple"] = True
    ftrend["label"] = ("change in mean annual compound frequency\n"
                       f"({LATE_PERIOD[0]}–{LATE_PERIOD[1]} minus {EARLY_PERIOD[0]}–{EARLY_PERIOD[1]}, "
                       "event-months yr$^{-1}$)")
    for j, (field, ttl) in enumerate([
            (fmean, "(a) Mean annual compound frequency, 1950–2025"),
            (ftrend, "(b) Change in compound frequency (late − early 30 yr)")]):
        ax = fig.add_subplot(gs[0, j], projection=ccrs.Robinson())
        m = _draw_field(ax, ctx, field, desert=True, ai_hatch=True, coast=True,
                        stipple_size=0.9, stipple_target=9000)
        if j == 1:
            _ar6_boundaries(ax, edgecolor="0.25", linewidth=0.45, label_fontsize=8.6,
                            label_color="0.05", nudge=_LABEL_NUDGE, halo=True)
        else:
            _ar6_boundaries(ax, edgecolor="0.2", linewidth=0.45, label=False)
        ax.set_title(ttl, fontsize=11.5)
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                          extend="max" if field["sequential"] else "both")
        cb.set_label(field["label"], fontsize=13)
        cb.ax.tick_params(labelsize=11.5)

    legend_el = [
        Line2D([0], [0], color=DESERT_LINE, lw=1.2, label="Köppen desert (BW)"),
        Patch(facecolor="none", edgecolor="k", hatch="xxxx", label="hyper-arid (AI<0.05)"),
        Line2D([0], [0], color="0.2", lw=0.9, label="IPCC AR6 land region"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=5, lw=0,
               label="MK-significant increase (p<0.05)"),
    ]
    fig.legend(handles=legend_el, loc="lower center", ncol=4, fontsize=13, frameon=False,
              bbox_to_anchor=(0.5, 0.0))
    out = FIGDIR / "compound_global_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIGURE -- ABSOLUTE climatology (mean annual compound frequency) + its zoom
# --------------------------------------------------------------------------- #
def compound_climatology(ctx: _Ctx) -> Path:
    """Absolute state: mean annual compound event-months (global) beside the trend,
    so 'how much' and 'how fast' are read together."""
    fig = plt.figure(figsize=(15.5, 6.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.06, left=0.03, right=0.97, top=0.86, bottom=0.06)
    fmean = resolve_field(ctx, "compound_mean")
    ftrend = resolve_field(ctx, "compound_trend")
    for j, (field, ttl, lett) in enumerate([
            (fmean, "Mean annual compound frequency (1950–2025)", "a"),
            (ftrend, "Trend in compound frequency (Sen's slope)", "b")]):
        ax = fig.add_subplot(gs[0, j], projection=ccrs.Robinson())
        m = _draw_field(ax, ctx, field, desert=True, ai_hatch=False,
                        coast=True, stipple_size=0.3)
        ax.set_title(ttl, fontsize=11)
        ax.text(0.01, 1.0, lett, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.82,
                          extend="max" if field["sequential"] else "both")
        cb.set_label(field["label"], fontsize=9)
    fig.suptitle("Compound heatwave–drought state vs. change "
                 "(HW 1961–1990, SPEI-1, calibration 1950–2025)", fontsize=13.5, y=0.98)
    out = FIGDIR / "compound_climatology.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIGURE -- ABSOLUTE change in physical units (late 30-yr minus early 30-yr)
# --------------------------------------------------------------------------- #
def absolute_change(ctx: _Ctx) -> Path:
    """Early vs late 30-yr mean annual compound frequency and their difference --
    'how many more event-months per year now than then', in physical units."""
    e0, e1 = EARLY_PERIOD
    l0, l1 = LATE_PERIOD
    fmean = resolve_field(ctx, "compound_mean")
    seq_vmax = fmean["vmax"]
    early = ctx.mean_annual_period("compound", e0, e1)
    late = ctx.mean_annual_period("compound", l0, l1)
    fchange = resolve_field(ctx, "compound_change")

    panels = [
        (early, "YlOrRd", None, 0, seq_vmax, f"Early mean {e0}–{e1}", "max",
         f"mean annual compound (event-months yr$^{{-1}}$)"),
        (late, "YlOrRd", None, 0, seq_vmax, f"Late mean {l0}–{l1}", "max",
         f"mean annual compound (event-months yr$^{{-1}}$)"),
        (fchange["data"], CMAP, None, fchange["vmin"], fchange["vmax"],
         f"Change (late − early)", "both", fchange["label"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.0),
                             subplot_kw={"projection": ccrs.Robinson()})
    for j, (ax, (da, cmap, norm, vmin, vmax, ttl, ext, clab)) in enumerate(zip(axes, panels)):
        _basemap(ax)
        d = da.where(ctx.land)
        m = ax.pcolormesh(ctx.lon, ctx.lat, d.values, transform=ccrs.PlateCarree(),
                          cmap=cmap, vmin=vmin, vmax=vmax, zorder=1, rasterized=True)
        _nan_land_da(ax, ctx, da)
        ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
                   colors=DESERT_LINE, linewidths=0.35, transform=ccrs.PlateCarree(), zorder=3)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
        ax.set_title(ttl, fontsize=11)
        ax.text(0.01, 1.0, "abc"[j], transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85, extend=ext)
        cb.set_label(clab, fontsize=8.5)
    fig.suptitle("Absolute change in compound heatwave–drought frequency "
                 "(HW 1961–1990, SPEI-1, calibration 1950–2025)", fontsize=13.5, y=1.0)
    out = FIGDIR / "compound_absolute_change.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIGURE -- Aridity Index + Köppen deserts overlaid (dryland context)
# --------------------------------------------------------------------------- #
def aridity_koppen_overlay(ctx: _Ctx) -> Path:
    """AI (UNEP classes, filled) with Köppen desert (BW, brown) and steppe (BS, olive)
    boundaries overlaid, so the two dryland definitions are visually reconciled."""
    fig = plt.figure(figsize=(14, 7.2))
    ax = plt.axes(projection=ccrs.Robinson())
    _basemap(ax)
    field = resolve_field(ctx, "aridity")
    da = field["data"].where(ctx.land)
    m = ax.pcolormesh(ctx.lon, ctx.lat, da.values, transform=ccrs.PlateCarree(),
                      cmap=field["cmap"], norm=field["norm"], zorder=1, rasterized=True)
    ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
               colors="#3a2a0c", linewidths=0.6, transform=ccrs.PlateCarree(), zorder=3)
    steppe = ctx.arid_all & ~ctx.desert
    ax.contour(ctx.lon2d, ctx.lat2d, steppe.astype(float), levels=[0.5],
               colors="#556b2f", linewidths=0.5, linestyles="--",
               transform=ccrs.PlateCarree(), zorder=3)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
    cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                      ticks=[0.025, 0.125, 0.35, 0.575, 1.08])
    cb.ax.set_yticklabels(drylands.AI_CLASS_LABELS, fontsize=9)
    cb.set_label("Aridity Index class (AI = P/PET)", fontsize=10)
    ax.legend(handles=[
        Line2D([0], [0], color="#3a2a0c", lw=1.4, label="Köppen desert (BW) boundary"),
        Line2D([0], [0], color="#556b2f", lw=1.2, ls="--", label="Köppen steppe (BS) boundary"),
    ], loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)
    ax.set_title("Global drylands: UNEP Aridity Index (fill) with Köppen dry-climate "
                 "boundaries (lines)", fontsize=12.5)
    out = FIGDIR / "aridity_koppen_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# FIGURE 3 -- heat vs drought driver attribution (marginal trend maps)
# --------------------------------------------------------------------------- #
def driver_attribution(ctx: _Ctx) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8),
                             subplot_kw={"projection": ccrs.Robinson()})
    for ax, var, ttl in zip(axes, ["heat", "drought"],
                            ["Heatwave-month trend", "Drought-month trend"]):
        vmax = ctx.vmax(var)
        _basemap(ax)
        m = _slope_mesh(ax, ctx, var, vmax)
        _stipple(ax, ctx, var)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
        ax.set_title(ttl, fontsize=11)
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8, extend="both")
        cb.set_label(f"Sen's slope (months decade$^{{-1}}$)", fontsize=9)
    axes[0].text(0.01, 1.0, "a", transform=axes[0].transAxes, fontsize=13, fontweight="bold", va="bottom")
    axes[1].text(0.01, 1.0, "b", transform=axes[1].transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.suptitle("Marginal driver trends behind the compound signal "
                 "(HW 1961–1990, SPEI-1, calibration 1950–2025)", fontsize=12.5, y=1.02)
    out = FIGDIR / "driver_attribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =========================================================================== #
# ONSET ORDER: heat-leading vs drought-leading compound events
# =========================================================================== #
_BIG = {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12}


def _period_fit(series, period):
    yrs = np.array([y for y in series.index if period[0] <= y <= period[1]])
    vals = series.loc[yrs].values
    lr = linregress(yrs, vals)
    return yrs, lr.slope, lr.stderr, lr.intercept


def leading_overview(ctx: _Ctx) -> Path:
    """Hero dominance map + stacked temporal trend (past/recent slopes) + increase-%
    bars -- the onset-order analogue of a 'spatial pattern & temporal trend' figure."""
    with plt.rc_context(_BIG):
        fig = plt.figure(figsize=(16.5, 12.6))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 0.9],
                      hspace=0.24, wspace=0.2, left=0.04, right=0.965, top=0.93, bottom=0.07)

        # (a) hero dominance map -------------------------------------------- #
        ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
        _basemap(ax)
        dom = ctx.leading_dominance().where(ctx.land & (ctx.leading_dominance()["latitude"] > -60))
        m = ax.pcolormesh(ctx.lon, ctx.lat, dom.values, transform=ccrs.PlateCarree(),
                          cmap="RdBu_r", vmin=-1, vmax=1, zorder=1, rasterized=True)
        ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
                   colors=DESERT_LINE, linewidths=0.35, transform=ccrs.PlateCarree(), zorder=3)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
        ax.set_title("Which hazard leads the compound event?", fontsize=17, pad=8)
        ax.text(0.008, 0.98, "a", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
        cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.74, pad=0.012,
                          ticks=[-1, -0.5, 0, 0.5, 1])
        cb.ax.set_yticklabels(["drought-\nleading", "", "balanced", "", "heat-\nleading"])
        cb.set_label("onset-order dominance", fontsize=12)

        # (b) stacked temporal trend ---------------------------------------- #
        ax_ts = fig.add_subplot(gs[1, 0])
        eh = ctx.leading_extent_series("heat_leading")
        ed = ctx.leading_extent_series("drought_leading")
        yrs = eh.index.values
        ax_ts.bar(yrs, ed.values, color=DROUGHT_COLOR, label="drought-leading", width=0.9)
        ax_ts.bar(yrs, eh.values, bottom=ed.values, color=HEAT_COLOR,
                  label="heat-leading", width=0.9)
        # Trend lines are fit to the heat-leading series specifically (not the
        # heat+drought-leading stacked total) -- this is the series the "flat, then
        # sharply accelerating" claim in the text is actually about.
        for period, col, lbl in [(LEAD_PAST, "#2c6fbb", "past"), (LEAD_RECENT, "#111111", "recent")]:
            py, sl, se, ic = _period_fit(eh, period)
            ax_ts.plot(py, sl * py + ic, ls="--", lw=2.2, color=col,
                       label=f"heat-leading {lbl} {period[0]}–{period[1]}: {sl*10:+.2f}$\\pm${se*10:.2f}/dec")
        # Genuine (data-fitted, not fixed-midpoint) Pettitt change point of the
        # heat-leading extent series -- the actual independent comparison against the
        # heatwave-month change point (Sec. results-changepoint), rather than relying
        # on the fixed LEAD_PAST/LEAD_RECENT split coinciding with it by construction.
        cp_year = _pettitt_changepoint(eh.values, yrs)
        ax_ts.axvline(cp_year, color="0.25", lw=1.4, ls=":", zorder=0)
        ax_ts.text(cp_year + 0.8, 2, f"Pettitt CP: {cp_year}",
                   fontsize=9.5, color="0.25", va="bottom", rotation=90)
        ax_ts.set_xlim(yrs.min() - 1, yrs.max() + 1)
        ax_ts.set_xlabel("year"); ax_ts.set_ylabel("land area affected (%)")
        ax_ts.set_title("Temporal trend by onset order", fontsize=15)
        ax_ts.legend(fontsize=10.5, loc="upper left", frameon=False, ncol=1)
        ax_ts.text(-0.02, 1.06, "b", transform=ax_ts.transAxes, fontsize=18, fontweight="bold", va="top")

        # (c) increase-% bars ----------------------------------------------- #
        ax_b = fig.add_subplot(gs[1, 1])
        incs, labs, cols = [], [], []
        for typ, col, lab in [("heat_leading", HEAT_COLOR, "heat-leading"),
                              ("drought_leading", DROUGHT_COLOR, "drought-leading")]:
            s = ctx.leading_extent_series(typ)
            past = s.loc[[y for y in s.index if LEAD_PAST[0] <= y <= LEAD_PAST[1]]].mean()
            rec = s.loc[[y for y in s.index if LEAD_RECENT[0] <= y <= LEAD_RECENT[1]]].mean()
            incs.append(100 * (rec - past) / past); labs.append(lab); cols.append(col)
        bars = ax_b.bar(labs, incs, color=cols, edgecolor="0.3")
        for bar, v in zip(bars, incs):
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:+.0f}%",
                      ha="center", fontsize=14, fontweight="bold")
        ax_b.set_ylabel("increase, recent vs past (%)")
        ax_b.set_title("Relative increase", fontsize=15)
        ax_b.axhline(0, color="k", lw=0.7)
        ax_b.text(-0.04, 1.06, "c", transform=ax_b.transAxes, fontsize=18, fontweight="bold", va="top")

        fig.suptitle("Onset order of compound heatwave–drought events "
                     "(HW 1961–1990, SPEI-1, 1950–2025)", fontsize=17, y=0.975)
        out = FIGDIR / "leading_overview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def leading_response(ctx: _Ctx) -> Path:
    """Joint distribution: each year is one point (drought-leading extent x,
    heat-leading extent y), coloured by year, with period-split marginal densities --
    shows both types rising while heat-leading rises faster (points climb above 1:1)."""
    x = ctx.leading_extent_series("drought_leading")
    y = ctx.leading_extent_series("heat_leading")
    yrs = x.index.values
    with plt.rc_context(_BIG):
        fig = plt.figure(figsize=(10.5, 10.5))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[1, 4],
                      wspace=0.04, hspace=0.04, left=0.1, right=0.96, top=0.96, bottom=0.14)
        ax = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=ax)
        axr = fig.add_subplot(gs[1, 1], sharey=ax)

        sc = ax.scatter(x.values, y.values, c=yrs, cmap="turbo", s=95,
                        edgecolor="k", linewidths=0.4, zorder=3)
        hi = max(x.max(), y.max()) * 1.08
        ax.plot([0, hi], [0, hi], "k--", lw=1.3, alpha=0.6, zorder=1)
        ax.text(hi * 0.62, hi * 0.7, "1:1", rotation=45, fontsize=12, alpha=0.6)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Drought-leading extent (% land area)")
        ax.set_ylabel("Heat-leading extent (% land area)")
        ax.grid(alpha=0.2)

        grid_x = np.linspace(0, hi, 200)
        for period, col, lab in [(LEAD_PAST, "#2c6fbb", f"{LEAD_PAST[0]}–{LEAD_PAST[1]}"),
                                 (LEAD_RECENT, "#c0392b", f"{LEAD_RECENT[0]}–{LEAD_RECENT[1]}")]:
            sel = (yrs >= period[0]) & (yrs <= period[1])
            kx = gaussian_kde(x.values[sel]); ky = gaussian_kde(y.values[sel])
            axt.fill_between(grid_x, kx(grid_x), color=col, alpha=0.4, lw=1.4, ec=col, label=lab)
            axr.fill_betweenx(grid_x, ky(grid_x), color=col, alpha=0.4, lw=1.4, ec=col)
        axt.legend(fontsize=11, frameon=False, loc="upper right", title="period")
        for a in (axt, axr):
            a.axis("off")
        cax = fig.add_axes([0.1, 0.06, 0.6, 0.02])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("year", fontsize=12)
        fig.suptitle("Heat-leading rises faster than drought-leading\n(each point = one year, 1950–2025)",
                     fontsize=16, y=1.0)
        out = FIGDIR / "leading_response.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


_AR6G = None
_LAND_UNION = None
_CLIP = None
_EXCLUDE_REGIONS = {"WAN", "EAN"}   # Antarctica: SPEI/compound not meaningful over ice
_PANEL_FONT = {"font.family": "sans-serif",
               "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
               "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 11,
               "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}


def _nice_max(v):
    v = float(v)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    exp = int(np.floor(np.log10(v)))
    base = 10.0 ** exp
    for m in (1, 1.5, 2, 3, 4, 5, 6, 8, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def _fmt(x):
    return f"{x:g}"


def _ar6_geoms():
    global _AR6G
    if _AR6G is None:
        import regionmask
        _AR6G = regionmask.defined_regions.ar6.land
    return _AR6G


def _land_union():
    """Cached shapely union of Natural-Earth land (50m; 110m fallback) for clipping."""
    global _LAND_UNION
    if _LAND_UNION is None:
        from shapely.ops import unary_union
        try:
            geoms = list(cfeature.NaturalEarthFeature("physical", "land", "50m").geometries())
        except Exception:
            geoms = list(cfeature.NaturalEarthFeature("physical", "land", "110m").geometries())
        _LAND_UNION = unary_union(geoms).buffer(0)
    return _LAND_UNION


def _polygonal(geom):
    if geom.is_empty:
        return []
    t = geom.geom_type
    if t == "Polygon":
        return [geom]
    if t == "MultiPolygon":
        return list(geom.geoms)
    if t == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out += _polygonal(g)
        return out
    return []


def _ar6_clipped():
    """abbrev -> (land-clipped geometry, (label_lon, label_lat))."""
    global _CLIP
    if _CLIP is None:
        ar6 = _ar6_geoms(); land = _land_union()
        cents = np.asarray(ar6.centroids)
        clip, rep = {}, {}
        for i, ab in enumerate(ar6.abbrevs):
            c = ar6.polygons[i].buffer(0).intersection(land)
            clip[ab] = c
            try:
                p = c.representative_point(); rep[ab] = (p.x, p.y)
            except Exception:
                rep[ab] = (float(cents[i, 0]), float(cents[i, 1]))
        _CLIP = (clip, rep)
    return _CLIP


def _region_means(field, reg_da):
    ar6 = _ar6_geoms()
    out = {}
    for ab, num in zip(ar6.abbrevs, ar6.numbers):
        out[ab] = (np.nan if ab in _EXCLUDE_REGIONS
                   else regions.area_weighted_regional_stat(field, reg_da, int(num)))
    return out


# Region carrying the shared example axis labels ("year" / value) — a panel with room
# below/left of it. Was "SAM" (in-map slot); now the bottom-left-most panel.
_AXIS_LABEL_REGION = "WAF"

# Every ice-free AR6 land region must get exactly one panel (guard against typos below).
_ALL_PANEL_REGIONS = frozenset({
    "GIC", "NWN", "NEN", "WNA", "CNA", "ENA", "NCA", "SCA", "CAR", "NWS", "NSA", "NES",
    "SAM", "SWS", "SES", "SSA", "NEU", "WCE", "EEU", "MED", "SAH", "WAF", "CAF", "NEAF",
    "SEAF", "WSAF", "ESAF", "MDG", "RAR", "WSB", "ESB", "RFE", "WCA", "ECA", "TIB",
    "EAS", "ARP", "SAS", "SEA", "EAU", "CAU", "NAU", "SAU", "NZ"})


def _panel_slots():
    """Figure-fraction slots for all 44 ice-free AR6 regions, laid out geographically
    around the map so no connector arrows are needed. Border = top 2x8, bottom 1x8,
    left 2x5, right 2x5. Assignment is continental: top = northern extratropics,
    left = the Americas, right = tropical Asia + Australasia, bottom = Africa. Ordering:
    top/bottom W->E, sides N->S; side outer column = the more extreme-longitude set."""
    # Wider panels and clear gutters accommodate 12–15 pt labels.
    w, h = 0.092, 0.072
    xs8 = [round(0.035 + i * (0.875 - 0.035) / 7, 4) for i in range(8)]   # top/bottom cols
    ys5 = [0.615, 0.495, 0.375, 0.255, 0.135]                               # N->S, 5 side rows
    lx_out, lx_in = xs8[0], xs8[0] + 0.120
    rx_out, rx_in = xs8[-1], xs8[-1] - 0.120

    top1 = ["GIC", "NWN", "NEN", "NEU", "RAR", "WSB", "ESB", "RFE"]   # far north
    top2 = ["WNA", "CNA", "ENA", "WCE", "EEU", "MED", "ECA", "EAS"]   # mid north
    left_out = ["NCA", "SCA", "NWS", "SWS", "SSA"]   # W Americas, N->S
    left_in = ["CAR", "NSA", "NES", "SAM", "SES"]    # E Americas, N->S
    right_in = ["WCA", "TIB", "ARP", "SAS", "SEA"]   # tropical Asia, N->S (near map)
    right_out = ["NAU", "CAU", "EAU", "SAU", "NZ"]   # Australasia, N->S
    bot = ["WAF", "SAH", "WSAF", "CAF", "ESAF", "SEAF", "NEAF", "MDG"]   # Africa, W->E

    slots = {}
    for ab, x in zip(top1, xs8):
        slots[ab] = [x, 0.845, w, h]
    for ab, x in zip(top2, xs8):
        slots[ab] = [x, 0.735, w, h]
    for ab, x in zip(bot, xs8):
        slots[ab] = [x, 0.025, w, h]
    for ab, y in zip(left_out, ys5):
        slots[ab] = [lx_out, y, w, h]
    for ab, y in zip(left_in, ys5):
        slots[ab] = [lx_in, y, w, h]
    for ab, y in zip(right_out, ys5):
        slots[ab] = [rx_out, y, w, h]
    for ab, y in zip(right_in, ys5):
        slots[ab] = [rx_in, y, w, h]

    assert len(slots) == 44 and set(slots) == _ALL_PANEL_REGIONS, \
        f"panel slots must cover exactly the 44 ice-free AR6 regions; got {sorted(slots)}"
    return slots


SLOTS = _panel_slots()


def _draw_ring_map(fig, region_value, *, cmap, norm, cbar_label, cbar_ticks,
                   cbar_ticklabels=None, sm_extend="neither", map_legend=None):
    """Draw the central coastline-clipped AR6 choropleth + ID labels + colorbar; the
    shared core of every ringed-panel figure. Returns the map GeoAxes."""
    ar6 = _ar6_geoms()
    clip, rep = _ar6_clipped()
    abbrevs = list(ar6.abbrevs)
    ax = fig.add_axes([0.265, 0.19, 0.47, 0.50], projection=ccrs.Robinson())
    ax.set_global()
    ax.patch.set_facecolor("white")
    ax.spines["geo"].set_linewidth(0.6)
    ax.add_feature(cfeature.LAND, facecolor="#eeece7", zorder=0)
    for ab in abbrevs:
        if ab in _EXCLUDE_REGIONS:
            continue
        val = region_value.get(ab, np.nan)
        fc = cmap(norm(val)) if np.isfinite(val) else "#e6e6e6"
        geoms = _polygonal(clip[ab])
        if geoms:
            ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor=fc,
                              edgecolor="0.25", linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.COASTLINE, lw=0.35, zorder=3, edgecolor="0.4")
    for ab in abbrevs:
        if ab in _EXCLUDE_REGIONS:
            continue
        lon, lat = rep[ab]
        # Short leader lines keep narrow, adjacent regions legible at this map size.
        offset = {"WSAF": (-25, -8), "ESAF": (22, -12), "MDG": (18, 6),
                  "SWS": (-18, 0), "SES": (10, 4),
                  "CNA": (0, 18), "ENA": (12, -4)}.get(ab, (0, 0))
        ax.annotate(ab, xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=offset, textcoords="offset points", fontsize=14,
                    fontweight="bold", ha="center", va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="0.25", lw=0.7)
                    if offset != (0, 0) else None)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.365, 0.20, 0.27, 0.014])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=cbar_ticks, extend=sm_extend)
    if cbar_ticklabels is not None:
        cb.ax.set_xticklabels(cbar_ticklabels, fontsize=14)
    else:
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))
    cb.ax.tick_params(labelsize=16)
    cb.set_label(cbar_label, fontsize=18, labelpad=3)
    if map_legend is not None:
        ax.legend(handles=map_legend, loc="center left", bbox_to_anchor=(0.17, 0.66),
                  fontsize=10, frameon=True, framealpha=0.9, borderpad=0.5, handlelength=1.4)
    return ax


def regional_panel_figure(region_value, ring_series, *, cmap, norm, cbar_label,
                          cbar_ticks, suptitle, out, cbar_ticklabels=None,
                          sm_extend="neither", example_ylabel="value",
                          map_legend=None) -> Path:
    """Kim-Fig.4A-style figure: AR6 regions clipped to the coastline, ringed by per-region
    time-series panels (`ring_series`: abbrev -> list of (years, values, color, _))."""
    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(20, 15))
        _draw_ring_map(fig, region_value, cmap=cmap, norm=norm, cbar_label=cbar_label,
                       cbar_ticks=cbar_ticks, cbar_ticklabels=cbar_ticklabels,
                       sm_extend=sm_extend, map_legend=map_legend)
        ymax = 5.0  # Requested shared display limit for regional panels.
        yticks = [0, ymax / 2, ymax]
        xticks = [1950, 2025]
        for ab, slot in SLOTS.items():
            if ab not in ring_series:
                continue
            axi = fig.add_axes(slot)
            series = ring_series[ab]
            for yy, vv, color, _ist in series:
                axi.plot(yy, vv, "o", ms=2.7, color=color, mec="none", alpha=0.5)
                b1, b0 = theilslopes(vv, yy)[:2]
                axi.plot(yy, b1 * yy + b0, "-", color=color, lw=2.2)
            axi.set_xlim(1950, 2025); axi.set_ylim(0, ymax)
            axi.set_xticks(xticks); axi.set_yticks(yticks)
            axi.set_xticklabels([str(t) for t in xticks])
            axi.set_yticklabels([_fmt(t) for t in yticks])
            for lab, ha in zip(axi.get_xticklabels(), ("left", "right")):
                lab.set_ha(ha)
            axi.tick_params(labelsize=15, length=3, pad=1.5)
            for sp in ("top", "right"):
                axi.spines[sp].set_visible(False)
            yy0, vv0, col0, _ = series[0]
            slope = _ts(vv0, yy0)
            axi.text(0.0, 1.08, ab, transform=axi.transAxes, fontsize=18,
                     fontweight="bold", va="bottom", ha="left")
            axi.text(1.0, 1.08, f"{slope:+.2f}", transform=axi.transAxes, fontsize=16,
                     color=col0, va="bottom", ha="right", fontweight="bold")
            if ab == _AXIS_LABEL_REGION:
                axi.set_xlabel("year", fontsize=17, labelpad=4)
                axi.set_ylabel(example_ylabel, fontsize=17, labelpad=4)
        fig.suptitle(suptitle, fontsize=23, y=0.995, va="top", fontweight="bold")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def regional_box_figure(region_value, box_data, *, cmap, norm, cbar_label, cbar_ticks,
                        suptitle, out, period_labels, box_colors=("#4a7fb5", "#c0392b"),
                        cbar_ticklabels=None, sm_extend="both", example_ylabel="value",
                        panel_notes=None) -> Path:
    """Ringed figure whose per-region panels are two boxplots (early vs late period),
    `box_data`: abbrev -> (early_values, late_values). Map shaded by `region_value`.
    `panel_notes` (abbrev->str, e.g. 'cp 1997') is shown per panel instead of the median
    change when given (used by the change-point-split version)."""
    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(20, 15))
        _draw_ring_map(fig, region_value, cmap=cmap, norm=norm, cbar_label=cbar_label,
                       cbar_ticks=cbar_ticks, cbar_ticklabels=cbar_ticklabels,
                       sm_extend=sm_extend)
        ymax = 5.0  # Requested shared display limit for regional panels.
        yticks = [0, ymax / 2, ymax]
        for ab, slot in SLOTS.items():
            if ab not in box_data:
                continue
            axi = fig.add_axes(slot)
            early, late = box_data[ab]
            bp = axi.boxplot([early, late], positions=[0, 1], widths=0.62, patch_artist=True,
                             showfliers=False, medianprops=dict(color="black", lw=1.6),
                             whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
            for patch, c in zip(bp["boxes"], box_colors):
                patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor("0.3")
            axi.set_xlim(-0.6, 1.6); axi.set_ylim(0, ymax)
            axi.set_xticks([0, 1]); axi.set_xticklabels(list(period_labels))
            axi.set_yticks(yticks); axi.set_yticklabels([_fmt(t) for t in yticks])
            axi.tick_params(labelsize=15, length=3, pad=1.5)
            for sp in ("top", "right"):
                axi.spines[sp].set_visible(False)
            axi.text(0.0, 1.08, ab, transform=axi.transAxes, fontsize=18,
                     fontweight="bold", va="bottom", ha="left")
            if panel_notes is not None:
                axi.text(1.0, 1.08, panel_notes.get(ab, ""), transform=axi.transAxes,
                         fontsize=14, color="#2c6fbb", va="bottom", ha="right", fontweight="bold")
            else:
                dm = np.nanmedian(late) - np.nanmedian(early)
                axi.text(1.0, 1.08, f"{dm:+.1f}", transform=axi.transAxes, fontsize=16,
                         color=box_colors[1], va="bottom", ha="right", fontweight="bold")
            if ab == _AXIS_LABEL_REGION:
                axi.set_ylabel(example_ylabel, fontsize=17, labelpad=4)
        fig.suptitle(suptitle, fontsize=23, y=0.995, va="top", fontweight="bold")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def _marginal_regional(ctx: _Ctx, var, *, title, out_name, ylabel, cbar_label) -> Path:
    """AR6 land regions (coastline-clipped) shaded by MEAN annual `var`-month count,
    ID-labelled, surrounded by per-region annual count-vs-year panels."""
    field = ctx.mean_annual(var)
    reg_da = regions.ar6_land_regions(field)
    region_value = _region_means(field, reg_da)
    finite = np.array([v for v in region_value.values() if np.isfinite(v)])
    vmax = float(np.nanpercentile(finite, 98)) or 1.0
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("coolwarm")
    ab_to_num = dict(zip(_ar6_geoms().abbrevs, _ar6_geoms().numbers))
    yrs = ctx.annual_counts(var)["year"].values
    ring_series = {ab: [(yrs, ctx.region_annual_mean(var, reg_da, int(ab_to_num[ab])).values,
                         "#b2182b", True)] for ab in SLOTS}
    return regional_panel_figure(
        region_value, ring_series, cmap=cmap, norm=norm, sm_extend="max",
        cbar_label=cbar_label, cbar_ticks=[0, vmax / 2, vmax], example_ylabel=ylabel,
        suptitle=title, out=FIGDIR / out_name)


_REG_SUB = "map: 1950–2025 mean   ·   panels: annual count, number = trend/decade"


def compound_regional(ctx: _Ctx) -> Path:
    return _marginal_regional(
        ctx, "compound", out_name="compound_regional.png",
        title="Compound heatwave–drought months per year, by AR6 region\n" + _REG_SUB,
        ylabel="compound (mo yr$^{-1}$)", cbar_label="mean compound months (yr$^{-1}$)")


def heat_regional(ctx: _Ctx) -> Path:
    return _marginal_regional(
        ctx, "heat", out_name="heat_regional.png",
        title="Heatwave months per year, by AR6 region\n" + _REG_SUB,
        ylabel="heatwave (mo yr$^{-1}$)", cbar_label="mean heatwave months (yr$^{-1}$)")


def drought_regional(ctx: _Ctx) -> Path:
    return _marginal_regional(
        ctx, "drought", out_name="drought_regional.png",
        title="Drought months per year, by AR6 region\n" + _REG_SUB,
        ylabel="drought (mo yr$^{-1}$)", cbar_label="mean drought months (yr$^{-1}$)")


def compound_regional_boxplot(ctx: _Ctx, var="compound", split="fixed") -> Path:
    """Ringed AR6 figure: map shaded by the (right − left) change; each panel two boxplots.
    split="fixed"  -> left = 1950-1979, right = 1996-2025 (shared windows for all regions).
    split="cp"     -> left/right = pre/post each region's Pettitt change point (annotated)."""
    reg_da = regions.ar6_land_regions(ctx.mean_annual(var))
    ar6 = _ar6_geoms()
    region_value, box_data, notes = {}, {}, {}
    for ab, num in zip(ar6.abbrevs, ar6.numbers):
        if ab in _EXCLUDE_REGIONS:
            continue
        s = ctx.region_annual_mean(var, reg_da, int(num))
        yy, v = s.index.values, s.values
        if split == "cp":
            cp, cpp = _pettitt(v, yy)
            left, right = v[yy <= cp], v[yy > cp]
        else:
            left = v[(yy >= EARLY_PERIOD[0]) & (yy <= EARLY_PERIOD[1])]
            right = v[(yy >= LATE_PERIOD[0]) & (yy <= LATE_PERIOD[1])]
        region_value[ab] = float(np.mean(right)) - float(np.mean(left))
        if ab in SLOTS:
            box_data[ab] = (left, right)
            if split == "cp":
                notes[ab] = f"cp {cp}" + ("" if cpp < 0.05 else " n.s.")
    finite = np.array([v for v in region_value.values() if np.isfinite(v)])
    vmax = float(np.nanpercentile(np.abs(finite), 98)) or 1.0
    norm = Normalize(-vmax, vmax)
    cmap = plt.get_cmap("coolwarm")
    if split == "cp":
        return regional_box_figure(
            region_value, box_data, cmap=cmap, norm=norm, sm_extend="both",
            cbar_label=f"change in mean {var} months (post − pre change point, yr$^{{-1}}$)",
            cbar_ticks=[-vmax, 0, vmax], example_ylabel=f"{var} (mo yr$^{{-1}}$)",
            period_labels=("pre", "post"), panel_notes=notes,
            suptitle=f"{var.capitalize()} months per year by AR6 region: before vs after the change point\n"
                     "map: post − pre change · panels: distribution pre (blue) vs post (red) each region's change point",
            out=FIGDIR / f"{var}_regional_boxplot_cp.png")
    pe0, pe1 = EARLY_PERIOD
    pl0, pl1 = LATE_PERIOD
    return regional_box_figure(
        region_value, box_data, cmap=cmap, norm=norm, sm_extend="both",
        cbar_label=f"change in mean {var} months (late − early, yr$^{{-1}}$)",
        cbar_ticks=[-vmax, 0, vmax], example_ylabel=f"{var} (mo yr$^{{-1}}$)",
        period_labels=(f"{pe0}–{str(pe1)[2:]}", f"{pl0}–{str(pl1)[2:]}"),
        suptitle=f"{var.capitalize()} months per year by AR6 region: early vs late\n"
                 f"map: change (late − early) · panels: annual-count distribution, "
                 f"{pe0}–{pe1} (blue) vs {pl0}–{pl1} (red)",
        out=FIGDIR / f"{var}_regional_boxplot.png")


def leading_regional(ctx: _Ctx) -> Path:
    """AR6 land regions (coastline-clipped) shaded by onset-order dominance, ID-labelled,
    surrounded by per-region heat- vs drought-leading extent panels."""
    dom = ctx.leading_dominance()
    reg_da = regions.ar6_land_regions(dom)
    region_value = _region_means(dom, reg_da)
    finite = np.array([v for v in region_value.values() if np.isfinite(v)])
    vmax = float(np.nanmax(np.abs(finite))) or 1.0
    norm = Normalize(-vmax, vmax)
    cmap = plt.get_cmap("coolwarm")
    ab_to_num = dict(zip(_ar6_geoms().abbrevs, _ar6_geoms().numbers))
    yrs = ctx.leading_annual()["year"].values
    ring_series = {}
    for ab in SLOTS:
        num = int(ab_to_num[ab])
        eh = ctx.leading_extent_series("heat_leading", reg_da, num)
        ed = ctx.leading_extent_series("drought_leading", reg_da, num)
        ring_series[ab] = [(yrs, eh.values, HEAT_COLOR, False),
                           (yrs, ed.values, DROUGHT_COLOR, False)]
    legend = [Line2D([0], [0], color=HEAT_COLOR, lw=2.6, label="heat-leading"),
              Line2D([0], [0], color=DROUGHT_COLOR, lw=2.6, label="drought-leading")]
    return regional_panel_figure(
        region_value, ring_series, cmap=cmap, norm=norm, sm_extend="both",
        cbar_label="regional onset-order dominance",
        cbar_ticks=[-vmax, 0, vmax],
        cbar_ticklabels=["drought-leading", "balanced", "heat-leading"],
        example_ylabel="extent (%)", map_legend=legend,
        suptitle="Onset order (heat- vs drought-leading) by AR6 region\n"
                 "map: dominance   ·   panels: annual extent, number = heat-leading trend/decade",
        out=FIGDIR / "leading_regional.png")


def aridity_class_timeseries(ctx: _Ctx, var="compound") -> Path:
    """Categorical UNEP aridity-class map (a-e) + one time-series panel with a line per
    class -- the 'do drylands intensify faster than humid land?' figure. Aridity classes
    are global (non-contiguous), so this deliberately uses a shared multi-line panel
    rather than the placed-around-a-map layout used for AR6 regions."""
    aiclass = np.where(np.isfinite(ctx.ai),
                       np.digitize(ctx.ai, drylands.AI_CLASS_EDGES[1:-1]), -1)
    landv = ctx.land.values
    ann = ctx.annual_counts(var)
    yrs = ann["year"].values
    w = np.cos(np.deg2rad(ann["latitude"]))
    labels = drylands.AI_CLASS_LABELS
    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(17, 7.2))
        # (a) categorical aridity map
        axm = fig.add_axes([0.015, 0.05, 0.55, 0.88], projection=ccrs.Robinson())
        axm.set_global(); axm.patch.set_facecolor("white")
        cmap = ListedColormap(AI_COLORS)
        norm = BoundaryNorm(drylands.AI_CLASS_EDGES, len(AI_COLORS))
        da = xr.DataArray(np.where(landv, ctx.ai, np.nan),
                          coords={"latitude": ctx.lat, "longitude": ctx.lon},
                          dims=("latitude", "longitude"))
        axm.pcolormesh(ctx.lon, ctx.lat, da.values, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, zorder=1, rasterized=True)
        axm.add_feature(cfeature.COASTLINE, lw=0.35, zorder=3, edgecolor="0.35")
        axm.set_title("UNEP aridity class  (AI = P/PET)", fontsize=17)
        axm.legend(handles=[Patch(facecolor=AI_COLORS[i], edgecolor="k",
                                  label=f"{chr(97+i)}   {labels[i]}") for i in range(5)],
                   loc="lower left", fontsize=13, frameon=True, framealpha=0.9,
                   title="class", title_fontsize=13)
        axm.text(0.005, 1.005, "a", transform=axm.transAxes, fontsize=20, fontweight="bold", va="bottom")

        # (b) one line per aridity class
        axt = fig.add_axes([0.635, 0.13, 0.345, 0.74])
        for i in range(5):
            mask = xr.DataArray(landv & (aiclass == i),
                                coords={"latitude": ctx.lat, "longitude": ctx.lon},
                                dims=("latitude", "longitude"))
            if int(mask.values.sum()) == 0:
                continue
            s = ann.where(mask).weighted(w).mean(("latitude", "longitude")).values
            b1 = _ts(s, yrs)
            axt.plot(yrs, s, color=AI_COLORS[i], lw=2.8,
                     label=f"{chr(97+i)}  {labels[i]}  (Theil–Sen {b1:+.2f}/dec)")
        axt.set_xlim(yrs.min(), yrs.max())
        axt.set_xlabel("year", fontsize=15)
        axt.set_ylabel(f"land-mean {var} months yr$^{{-1}}$", fontsize=15)
        axt.tick_params(labelsize=13)
        axt.grid(alpha=0.25)
        axt.legend(fontsize=12.5, frameon=False, loc="upper left")
        axt.set_title(f"{var.capitalize()} frequency by aridity class", fontsize=16)
        axt.text(-0.02, 1.04, "b", transform=axt.transAxes, fontsize=20, fontweight="bold", va="top")

        fig.suptitle("Do drylands intensify faster?  Compound events per year by aridity class"
                     if var == "compound" else
                     f"{var.capitalize()} months per year by aridity class",
                     fontsize=19, y=0.99, fontweight="bold")
        out = FIGDIR / f"aridity_class_{var}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def enso_modulation(ctx: _Ctx, var: str = "compound") -> Path:
    """Composite anomaly of `var` frequency by ENSO phase (El Nino/La Nina minus
    Neutral months, RONI-based -- see chd.enso). Answers a different WHY question than
    the aridity-gradient figure: does ENSO redistribute WHERE compound risk
    concentrates, even in years/regions where it doesn't move the global land mean?"""
    comp = enso.phase_composite(var, baseline_key=HW, ts=TS, calib_key=CALIB)
    ann = _land_mean_annual(var)
    phase = enso.annual_phase().reindex(ann.index)
    st = enso.phase_stats(ann)

    landv = ctx.land.values
    vmax = float(np.nanpercentile(
        np.abs(np.concatenate([comp["el_nino_anom"].values[landv],
                                comp["la_nina_anom"].values[landv]])), 98))

    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(17, 11.5))
        axa = fig.add_axes([0.015, 0.40, 0.485, 0.56], projection=ccrs.Robinson())
        axb = fig.add_axes([0.5, 0.40, 0.485, 0.56], projection=ccrs.Robinson())
        m = None
        for ax, key, ttl, letter in [
            (axa, "el_nino_anom", "El Niño − Neutral", "a"),
            (axb, "la_nina_anom", "La Niña − Neutral", "b"),
        ]:
            _basemap(ax)
            da = comp[key].where(ctx.land)
            m = ax.pcolormesh(ctx.lon, ctx.lat, da.values, transform=ccrs.PlateCarree(),
                              cmap=CMAP, vmin=-vmax, vmax=vmax, zorder=1, rasterized=True)
            ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=3)
            ax.set_title(ttl, fontsize=17)
            ax.text(0.01, 1.0, letter, transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom")
        cax = fig.add_axes([0.30, 0.385, 0.40, 0.018])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(f"{var} frequency anomaly vs. Neutral months, trend removed (percentage points)", fontsize=13)
        cb.ax.tick_params(labelsize=11)

        axt = fig.add_axes([0.08, 0.055, 0.88, 0.27])
        yrs = ann.index.values
        for y, p in zip(yrs, phase.values):
            if p == "El Nino":
                axt.axvspan(y - 0.5, y + 0.5, color="#c0392b", alpha=0.18, lw=0)
            elif p == "La Nina":
                axt.axvspan(y - 0.5, y + 0.5, color="#2166ac", alpha=0.18, lw=0)
        axt.plot(yrs, ann.values, color="0.15", lw=1.8)
        axt.set_xlim(yrs.min(), yrs.max())
        axt.set_ylabel(f"land-mean {var}\nmonths yr$^{{-1}}$", fontsize=14)
        axt.set_xlabel("year", fontsize=14)
        axt.tick_params(labelsize=12)
        axt.grid(alpha=0.25)
        axt.legend(handles=[
            Patch(facecolor="#c0392b", alpha=0.35, label="El Niño year (DJF)"),
            Patch(facecolor="#2166ac", alpha=0.35, label="La Niña year (DJF)"),
        ], loc="upper left", fontsize=11, frameon=False)
        gd = st["group_means_detrended"]
        axt.set_title(
            f"Global land mean, anomaly from 21-yr running mean: El Niño {gd.loc['El Nino','mean']:+.2f}, "
            f"La Niña {gd.loc['La Nina','mean']:+.2f}, Neutral {gd.loc['Neutral','mean']:+.2f} "
            f"event-months yr$^{{-1}}$ (Mann–Whitney El Niño vs. La Niña p={st['mannwhitney_p_detrended']:.2f})",
            fontsize=12.5)
        axt.text(-0.05, 1.08, "c", transform=axt.transAxes, fontsize=20, fontweight="bold", va="top")

        fig.suptitle("ENSO composites of compound frequency, secular trend removed "
                     "(RONI phase, HW 1961–1990, SPEI-1, calib. 1950–2025)",
                     fontsize=17.5, y=0.99, fontweight="bold")
        out = FIGDIR / f"{var}_enso_modulation.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def export_enso_table(ctx: _Ctx, var: str = "compound") -> Path:
    """Write the numbers behind enso_modulation(): per-AR6-region El Nino/La Nina
    composite anomaly (percentage points vs. Neutral months) and the global land-mean
    group stats (means, Mann-Whitney, Pearson vs. annual RONI raw/detrended)."""
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    comp = enso.phase_composite(var, baseline_key=HW, ts=TS, calib_key=CALIB, detrend=True)
    raw = enso.phase_composite(var, baseline_key=HW, ts=TS, calib_key=CALIB, detrend=False)
    tbl = regions.ar6_region_table()
    reg_da = regions.ar6_land_regions(comp["el_nino_anom"])
    rows = []
    for _, r in tbl.iterrows():
        if r["abbrev"] in _EXCLUDE_REGIONS:
            continue
        idx = int(r["region_index"])
        rows.append({
            "abbrev": r["abbrev"], "name": r["name"],
            "el_nino_anom_pp": round(regions.area_weighted_regional_stat(comp["el_nino_anom"], reg_da, idx), 4),
            "la_nina_anom_pp": round(regions.area_weighted_regional_stat(comp["la_nina_anom"], reg_da, idx), 4),
            "el_nino_anom_pp_raw": round(regions.area_weighted_regional_stat(raw["el_nino_anom"], reg_da, idx), 4),
            "la_nina_anom_pp_raw": round(regions.area_weighted_regional_stat(raw["la_nina_anom"], reg_da, idx), 4),
        })
    pd.DataFrame(rows).sort_values("el_nino_anom_pp", ascending=False).to_csv(
        outdir / "enso_regional_anomaly.csv", index=False)

    ann = _land_mean_annual(var)
    st = enso.phase_stats(ann)
    gm = st["group_means"].reset_index().rename(columns={"index": "phase"})
    gm["mannwhitney_p_vs_other_phase"] = st["mannwhitney_p"]
    gm["pearson_r_vs_roni_raw"] = st["pearson_r_raw"]
    gm["pearson_p_vs_roni_raw"] = st["pearson_p_raw"]
    gm["pearson_r_vs_roni_detrended"] = st["pearson_r_detrended"]
    gm["pearson_p_vs_roni_detrended"] = st["pearson_p_detrended"]
    gd = st["group_means_detrended"]
    gm["mean_detrended_anomaly"] = gm["phase"].map(gd["mean"])
    gm["mannwhitney_p_EN_vs_LN_detrended"] = st["mannwhitney_p_detrended"]
    gm["mannwhitney_p_neutral_vs_enso_raw"] = st["mannwhitney_p_neutral_vs_enso_raw"]
    gm["mannwhitney_p_neutral_vs_enso_detrended"] = st["mannwhitney_p_neutral_vs_enso_detrended"]
    gm["mean_year"] = gm["phase"].map(st["mean_year"])
    gm["pearson_r_runmean_anom_vs_djf_roni"] = st["pearson_r_runmean_anom_vs_djf_roni"]
    gm["pearson_p_runmean_anom_vs_djf_roni"] = st["pearson_p_runmean_anom_vs_djf_roni"]
    for k in ("roni_annual_theilsen_per_decade", "roni_annual_mk_p", "trend_per_decade_raw",
              "trend_per_decade_enso_phase_adjusted"):
        gm[k] = st[k]
    gm.round(4).to_csv(outdir / "enso_global_stats.csv", index=False)
    print("wrote ENSO tables to", outdir, flush=True)
    return outdir


# Real-world Old-World B-climate is one unbroken belt from the Atlantic Sahara coast
# to the Gobi -- a naive connectivity-only ndimage.label() on the Köppen-arid mask
# fuses Sahara + Arabia + Iran + Central Asia + Gobi into ONE giant connected
# component, which the old bbox-vote heuristic then mislabelled just "Sahara" (43,967
# px spanning -17.5..118.5 lon / -8.2..54.2 lat -- verified empirically). There's no
# connectivity threshold that fixes this: the belt is genuinely contiguous, so you need
# an external reference for where one *named* desert ends and the next begins. Natural
# Earth's 10m "geography_regions_polys" physical layer (public domain; already this
# module's basemap/feature provider via cfeature.LAND/COASTLINE elsewhere) supplies
# that -- each entry below groups its NE polygon(s) (matched by an uppercase substring,
# since NE's own NAME casing is inconsistent) under one display name.
_NE_DESERT_GROUPS = {
    "Sahara": ["SAHARA", "NUBIAN DESERT", "LIBYAN DESERT", "WESTERN DESERT"],
    "Arabian": ["AL KHALI", "NAFUD", "SYRIAN DESERT"],
    "Kalahari–Namib": ["KALAHARI DESERT", "NAMIB DESERT"],
    "Iranian": ["LUT DESERT", "KAVIR DESERT"],
    "Karakum–Kyzylkum": ["GARAGUM DESERT", "QIZILQUM DESERT"],
    "Taklamakan": ["TAKLIMAKAN DESERT"],
    "Gobi": ["GOBI DESERT"],
    "Thar": ["THAR DESERT"],
    "Sonoran": ["SONORAN DESERT"],
    "Chihuahuan": ["CHIHUAHUAN DESERT"],
    "Great Basin": ["GREAT BASIN", "GREAT SALT LAKE DESERT"],
    "Atacama": ["ATACAMA"],
    "Patagonian": ["PATAGONIA"],
    "Caatinga": ["CAATINGAS"],
    "Australian": ["GREAT VICTORIA DESERT", "GREAT SANDY DESERT", "GIBSON DESERT",
                   "SIMPSON DESERT", "TANAMI DESERT", "STRZELECKI DESERT"],
}
_NE_DESERTS = None


def _ne_desert_regions():
    """display name -> unioned/buffered shapely geometry, built from Natural Earth's
    10m 'geography_regions_polys' (physical). Buffered 1.25 deg so the (cartographically
    generalised) named outline still catches Köppen-arid margin pixels just outside it."""
    global _NE_DESERTS
    if _NE_DESERTS is None:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        path = shpreader.natural_earth(resolution="10m", category="physical",
                                       name="geography_regions_polys")
        records = list(shpreader.Reader(path).records())
        out = {}
        for group, keywords in _NE_DESERT_GROUPS.items():
            geoms = [r.geometry for r in records
                    if any(kw in r.attributes["NAME"].upper() for kw in keywords)]
            if geoms:
                out[group] = unary_union(geoms).buffer(1.25)
        _NE_DESERTS = out
    return _NE_DESERTS


def _named_desert_index(lon, lat):
    """(2D array of group index, 0..K-1, NaN outside every named desert; group name
    list) on the given 1D lon/lat grid, via a single vectorized regionmask pass.
    The 1.25-deg buffer means neighbouring deserts' regions can overlap slightly at
    the seam -- mask_3D + first-match-wins resolves that (regionmask's 2D .mask()
    refuses to combine overlapping regions outright)."""
    import regionmask
    groups = _ne_desert_regions()
    names = list(groups.keys())
    regions = regionmask.Regions([groups[n] for n in names], names=names,
                                 abbrevs=names, name="deserts")
    m3 = regions.mask_3D(lon, lat, drop=False).values   # (region, lat, lon) bool
    region_idx = np.full(m3.shape[1:], np.nan)
    for gi in range(m3.shape[0]):
        unset = np.isnan(region_idx)
        region_idx[unset & m3[gi]] = gi
    return region_idx, names


def _patch_from_mask(pm, name, lat, lon, w2d, annv):
    # annv (compound annual counts) can legitimately be NaN for the handful of pixels
    # with zero SPEI-valid calendar months all year (see trends.annual_counts) -- a
    # plain (unweighted-by-validity) sum/wsum would let one NaN pixel poison every
    # year's mean for the whole patch. np.nansum + a per-year validity-adjusted
    # denominator keeps this correct without materially changing the weighting.
    rows, cols = np.where(pm)
    clat = float(np.average(lat[rows])); clon = float(np.average(lon[cols]))
    wmask = (w2d * pm)[None]
    valid = np.isfinite(annv)
    num = np.nansum(annv * wmask, axis=(1, 2))
    den = np.sum(wmask * valid, axis=(1, 2))
    series = num / den
    return dict(mask=pm, clat=clat, clon=clon, mean=float(series.mean()),
                series=series, name=name, size=int(pm.sum()))


def _build_dryland_patches(ctx: _Ctx, var="compound"):
    """One patch per named desert (see _NE_DESERT_GROUPS): its largest contiguous
    chunk within the Köppen B-group (desert+steppe) mask that clears a 300-pixel
    minimum, lettered a,b,c... by longitude, with a Pettitt change point attached.
    Shared by dryland_patch_regional() (the figure) and export_dryland_patch_table()
    (the numbers cited in the results text) so the two can never drift apart.
    Every _NE_DESERT_GROUPS entry has real overlap with the Köppen mask, so unlike
    the old bbox-vote heuristic there's no need for a generic "leftover dryland"
    fallback bucket: it would just be large *unnamed* corridors (e.g. the Kazakh
    steppe between Karakum and Gobi, 13k+ px) crowding out small-but-real named
    deserts like Atacama/Caatinga in a fixed top-N cut. This is specifically about
    the world's named deserts, so only named patches appear."""
    from scipy import ndimage
    km = drylands.koppen_desert_mask(ctx.slope_of("compound"))
    mask = km["arid_all"].values & ctx.land.values
    lat, lon = ctx.lat, ctx.lon
    w2d = np.cos(np.deg2rad(lat))[:, None]
    annv = ctx.annual_counts(var).values.astype(float)
    yrs = ctx.annual_counts(var)["year"].values

    region_idx, names = _named_desert_index(lon, lat)
    patches = []
    for gi, name in enumerate(names):
        pm_group = mask & (region_idx == gi)
        if not pm_group.any():
            continue
        lab, n = ndimage.label(pm_group, structure=np.ones((3, 3)))
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        best = int(np.argmax(sizes))
        if sizes[best] < 300:
            continue
        patches.append(_patch_from_mask(lab == (best + 1), name, lat, lon, w2d, annv))

    patches.sort(key=lambda p: p["clon"])
    for i, p in enumerate(patches):
        p["letter"] = chr(97 + i)
        p["cp"] = _pettitt_changepoint(p["series"], yrs)
    return patches, yrs


def dryland_patch_regional(ctx: _Ctx, var="compound", split="cp") -> Path:
    """Köppen dryland (B-group) patches, split by named-desert identity (see
    _NE_DESERT_GROUPS) and lettered a,b,c...; centre map framed on all four sides by
    per-patch panels: distribution (boxplot) panels on the TOP and LEFT edges, annual
    trend panels on the BOTTOM and RIGHT edges, aligned pairwise (same column index
    top/bottom, same row index left/right). split="cp" splits the boxplots at each
    patch's Pettitt change point; split="fixed" uses 1950-79 vs 1996-2025."""
    patches, yrs = _build_dryland_patches(ctx, var)
    lat, lon = ctx.lat, ctx.lon
    km = drylands.koppen_desert_mask(ctx.slope_of("compound"))
    mask = km["arid_all"].values & ctx.land.values

    means = np.array([p["mean"] for p in patches])
    vmax = _nice_max(np.nanpercentile(means, 98))
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("coolwarm")

    # value raster for the map fill
    vr = np.full(mask.shape, np.nan)
    for p in patches:
        vr[p["mask"]] = p["mean"]

    shared_max = _nice_max(np.nanmax([p["series"].max() for p in patches]))
    yticks = [0, shared_max / 2, shared_max]
    xticks = [1950, 1988, 2025]
    if split == "fixed":
        box_labels = ["early", "late"]
        tier = ("top + left edges: distribution early (1950–79) vs late (1996–2025)  ·  "
                "centre: 1950–2025 mean  ·  bottom + right edges: annual count + trend "
                "(dashed = change point)")
        out = FIGDIR / f"{var}_dryland_patches_fixed.png"
    else:
        box_labels = ["pre", "post"]
        tier = ("top + left edges: distribution pre vs post change point (cp)  ·  "
                "centre: 1950–2025 mean  ·  bottom + right edges: annual count + trend "
                "(dashed = change point)")
        out = FIGDIR / f"{var}_dryland_patches.png"

    # With 15 named deserts (vs. the original 8) putting every patch in one wide top
    # row + one wide bottom row made the figure absurdly elongated. Instead, frame the
    # map like a picture frame: box (distribution) panels always sit on the TOP and
    # LEFT edges, trend (annual-count) panels always sit on the BOTTOM and RIGHT edges
    # -- position along an edge is paired (same column index top/bottom, same row
    # index left/right), so e.g. the 3rd box panel down the left edge is the same
    # patch as the 3rd trend panel down the right edge. Patches split at the single
    # largest longitude gap in the (clon-sorted) list, which cleanly separates the
    # narrow, N-S-oriented Americas cluster (-> left/right edges, ordered by latitude)
    # from the wide, E-W-oriented Old World/Australia cluster (-> top/bottom edges,
    # ordered by longitude, same as before). Every panel in the figure -- top, bottom,
    # left, right -- shares the same physical width:height aspect ratio (`aspect_in`),
    # just uniformly scaled to fit its slot.
    clons = [p["clon"] for p in patches]
    gaps = [clons[i + 1] - clons[i] for i in range(len(clons) - 1)]
    split_i = (int(np.argmax(gaps)) + 1) if gaps else 0
    side_patches, col_patches = patches[:split_i], patches[split_i:]
    side_patches = sorted(side_patches, key=lambda p: -p["clat"])   # N -> S, top to bottom

    n_col = max(len(col_patches), 1)
    n_rows = len(side_patches)
    margin, gap = 0.015, 0.010
    target_col_in = 1.6
    fig_h = 13.5
    FONT_BOOST = 1.15             # +15% on every panel/cbar/suptitle text size
    tsc = (target_col_in / 1.957) * FONT_BOOST   # mild constant font scale vs. n=8 design
    panel_h = 0.15                # SAME height fraction for every box AND trend panel
    top_row_y = 0.775             # top row's box-panel bottom edge (title sits above it)

    map_y0, map_y1 = 0.30, 0.74
    edge_y_lo, edge_y_hi = 0.05, top_row_y + panel_h
    edge_row_gap = 0.016
    edge_pad = 0.02                # gap left between the edge panels and the map

    # left_w/right_w (map's side margins, sized to hug the left/right edge panels)
    # and cw (edge panel width) are mutually coupled -- cw depends on fig_w, which
    # depends on pw, which depends on left_w/right_w. Iterate a few times to a
    # fixed point instead of leaving a fixed guessed margin (which left a large,
    # user-visible dead gap between the edge panels and the map when it overshot).
    left_w = right_w = 0.135
    for _ in range(6):
        avail_w = (1 - left_w - right_w) - 2 * margin
        pw = (avail_w - (n_col - 1) * gap) / n_col
        fig_w = max(16.5, target_col_in / pw)
        aspect_in = (pw * fig_w) / (panel_h * fig_h)   # every panel's physical width:height
        rh = ((edge_y_hi - edge_y_lo) - max(n_rows - 1, 0) * edge_row_gap) / n_rows if n_rows else 0
        cw = (aspect_in * rh * fig_h) / fig_w
        left_w = right_w = cw + edge_pad if n_rows else 0.03

    def _edge_geometry(n_rows, y_lo, y_hi):
        """Row height + box/trend panel width (matching `aspect_in`) for a left/right
        edge stack of `n_rows` panels between figure-fraction y_lo..y_hi."""
        rh = ((y_hi - y_lo) - max(n_rows - 1, 0) * edge_row_gap) / n_rows if n_rows else 0
        cw = (aspect_in * rh * fig_h) / fig_w
        return rh, edge_row_gap, cw

    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # --- centre map: mean compound months; cropped to the dryland band, framed
        #     by the corner/column panels so it no longer needs the full figure width
        ax = fig.add_axes([left_w, map_y0, 1 - left_w - right_w, map_y1 - map_y0],
                          projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -56, 60], crs=ccrs.PlateCarree())
        ax.set_aspect("auto")
        ax.patch.set_facecolor("white")
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        ax.add_feature(cfeature.LAND, facecolor="#eeece7", zorder=0)
        ax.pcolormesh(lon, lat, vr, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                      zorder=1, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, lw=0.35, zorder=3, edgecolor="0.4")
        for p in patches:
            ax.text(p["clon"], p["clat"], p["letter"], transform=ccrs.PlateCarree(),
                    fontsize=20 * max(tsc, 0.75), fontweight="bold", ha="center", va="center",
                    zorder=4, color="black",
                    path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        map_cx = (left_w + (1 - right_w)) / 2
        cax = fig.add_axes([map_cx - 0.08, map_y0 - 0.05, 0.16, 0.012])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, vmax / 2, vmax], extend="max")
        cb.ax.tick_params(labelsize=13 * FONT_BOOST)
        cb.set_label(f"patch-mean {var} months (yr$^{{-1}}$)", fontsize=14 * FONT_BOOST, labelpad=2)

        def _box_panel(axb, p, fs, title_inside=False):
            if split == "fixed":
                left = p["series"][(yrs >= EARLY_PERIOD[0]) & (yrs <= EARLY_PERIOD[1])]
                right = p["series"][(yrs >= LATE_PERIOD[0]) & (yrs <= LATE_PERIOD[1])]
            else:
                left = p["series"][yrs <= p["cp"]]; right = p["series"][yrs > p["cp"]]
            bp = axb.boxplot([left, right], positions=[0, 1], widths=0.62, patch_artist=True,
                             showfliers=False, medianprops=dict(color="black", lw=1.6),
                             whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
            for patch_, c in zip(bp["boxes"], ("#4a7fb5", "#c0392b")):
                patch_.set_facecolor(c); patch_.set_alpha(0.8); patch_.set_edgecolor("0.3")
            axb.set_xlim(-0.6, 1.6); axb.set_ylim(0, shared_max)
            axb.set_xticks([0, 1]); axb.set_xticklabels(box_labels, fontsize=14 * fs)
            axb.tick_params(labelsize=14 * fs, length=3, pad=1.5)
            for sp in ("top", "right"):
                axb.spines[sp].set_visible(False)
            title = f"{p['letter']}  {p['name']}"
            if title_inside:
                # side panels are stacked vertically with little room between them --
                # an outside set_title() (drawn above the axes box) collides with the
                # tick labels of whichever panel is stacked above it, so keep the
                # title/cp-note INSIDE the box instead (same trick the trend panels
                # already use for their letter/slope corner labels).
                axb.text(0.03, 0.985, title, transform=axb.transAxes, fontsize=16 * fs,
                         fontweight="bold", va="top", ha="left")
                if split == "cp":
                    # own line below the name (not the same row) -- the longer desert
                    # names (e.g. "Chihuahuan") otherwise run into a same-row cp note
                    # in these narrower corner panels.
                    axb.text(0.03, 0.85, f"cp {p['cp']}", transform=axb.transAxes, ha="left",
                             va="top", fontsize=12 * fs, fontweight="bold", color="#2c6fbb")
            else:
                axb.set_title(title, fontsize=16 * fs, fontweight="bold", pad=3)
                if split == "cp":
                    axb.text(0.5, 0.98, f"cp {p['cp']}", transform=axb.transAxes, ha="center",
                             va="top", fontsize=13.5 * fs, fontweight="bold", color="#2c6fbb")

        def _trend_panel(axi, p, fs):
            axi.plot(yrs, p["series"], "o", ms=1.9, color="#b2182b", mec="none", alpha=0.5)
            b1, b0 = theilslopes(p["series"], yrs)[:2]
            axi.plot(yrs, b1 * yrs + b0, "-", color="#b2182b", lw=2.3)
            axi.axvline(p["cp"], color="#2c6fbb", lw=1.5, ls="--")
            axi.set_xlim(1950, 2025); axi.set_ylim(0, shared_max)
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(t) for t in xticks])
            for lab_, ha in zip(axi.get_xticklabels(), ("left", "center", "right")):
                lab_.set_ha(ha)
            axi.tick_params(labelsize=14 * fs, length=3, pad=1.5)
            for sp in ("top", "right"):
                axi.spines[sp].set_visible(False)
            axi.text(0.05, 0.96, p["letter"], transform=axi.transAxes, fontsize=16 * fs,
                     fontweight="bold", va="top")
            axi.text(0.95, 0.92, f"{b1*10:+.2f}", transform=axi.transAxes, fontsize=15 * fs,
                     color="#b2182b", va="top", ha="right", fontweight="bold")

        # --- TOP/BOTTOM columns (col_patches): box above, trend below the map ---- #
        for i, p in enumerate(col_patches):
            x = left_w + margin + i * (pw + gap)
            axb = fig.add_axes([x, top_row_y, pw, panel_h])
            _box_panel(axb, p, tsc)
            axb.set_yticks(yticks); axb.set_yticklabels([_fmt(t) for t in yticks] if i == 0 else [])
            axi = fig.add_axes([x, 0.05, pw, panel_h])
            _trend_panel(axi, p, tsc)
            axi.set_yticks(yticks); axi.set_yticklabels([_fmt(t) for t in yticks] if i == 0 else [])

        # --- LEFT edge (box) / RIGHT edge (trend): side_patches (Americas), stacked
        #     top-to-bottom by latitude, row i on the left is the SAME patch as row i
        #     on the right -- mirrors the top/bottom column pairing but rotated 90°. -#
        n_rows = len(side_patches)
        if n_rows:
            y_lo, y_hi = 0.05, top_row_y + panel_h
            rh, row_gap, cw = _edge_geometry(n_rows, y_lo, y_hi)
            esc = min(tsc, (rh / panel_h) * tsc)   # font scale follows the panel's own height
            for i, p in enumerate(side_patches):
                y = y_hi - rh - i * (rh + row_gap)
                axb = fig.add_axes([0.006, y, cw, rh])
                _box_panel(axb, p, esc, title_inside=True)
                axb.set_yticks(yticks); axb.set_yticklabels([_fmt(t) for t in yticks])
                axi = fig.add_axes([1 - 0.006 - cw, y, cw, rh])
                _trend_panel(axi, p, esc)
                axi.yaxis.tick_right()
                axi.set_yticks(yticks); axi.set_yticklabels([_fmt(t) for t in yticks])

        fig.suptitle("Compound events over the world's Köppen drylands (B-group patches)\n" + tier,
                     fontsize=15 * FONT_BOOST, y=0.995, fontweight="bold")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def dryland_patch_summary(ctx: _Ctx, var="compound") -> Path:
    """Main-text dryland-patch figure (added 2026-09-23; the 30-panel border montage
    `dryland_patch_regional` moved to the supplement). (a) the 15 named-desert patches
    coloured by their Theil-Sen compound trend; (b) the same trends ranked, with 95%
    residual block-bootstrap CIs, filled markers = Hamed-Rao p<0.05, hollow = not
    significant, and the host AR6 region's trend (grey tick) for comparison."""
    patches, yrs = _build_dryland_patches(ctx, var)
    yrsf = yrs.astype(float)
    reg_da = regions.ar6_land_regions(ctx.mean_annual(var)).values
    ab_by_num = dict(zip((int(n) for n in _ar6_geoms().numbers), _ar6_geoms().abbrevs))
    rows = []
    for pt in patches:
        sub = reg_da[pt["mask"]]; sub = sub[np.isfinite(sub)]
        host = int(pd.Series(sub).mode().iloc[0])
        hs = ctx.region_annual_mean(var, xr.DataArray(reg_da, coords=ctx.slope_of("compound").coords), host)
        lo, hi = _block_boot_ci(pt["series"], yrsf)
        rows.append(dict(p=pt, ts=_ts(pt["series"], yrsf), lo=lo, hi=hi, hr=_hr_p(pt["series"]),
                         host=ab_by_num.get(host, ""), host_ts=_ts(hs.values, hs.index.values)))
    rows.sort(key=lambda r: r["ts"])
    vmax = _nice_max(max(abs(r["ts"]) for r in rows))
    norm = Normalize(-vmax, vmax)
    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(17, 7.0))
        ax = fig.add_axes([0.01, 0.12, 0.60, 0.80], projection=ccrs.PlateCarree())
        ax.set_extent([-130, 150, -56, 55], crs=ccrs.PlateCarree()); ax.set_aspect("auto")
        ax.add_feature(cfeature.LAND, facecolor="#eeece7", zorder=0)
        vr = np.full(ctx.land.shape, np.nan)
        for r in rows:
            vr[r["p"]["mask"]] = r["ts"]
        m = ax.pcolormesh(ctx.lon, ctx.lat, vr, transform=ccrs.PlateCarree(), cmap=CMAP,
                          norm=norm, zorder=1, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, lw=0.35, zorder=3, edgecolor="0.4")
        for r in rows:
            pt = r["p"]
            ax.text(pt["clon"], pt["clat"], pt["letter"], transform=ccrs.PlateCarree(),
                    fontsize=15, fontweight="bold", ha="center", va="center", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", shrink=0.55, pad=0.04, extend="both")
        cb.set_label(f"patch Theil–Sen trend ({UNITS})", fontsize=13)
        ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontsize=20, fontweight="bold")
        axd = fig.add_axes([0.72, 0.12, 0.26, 0.80])
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            col = CMAP(norm(r["ts"]))
            axd.plot([r["lo"], r["hi"]], [i, i], color="0.35", lw=1.6, zorder=1)
            axd.plot(r["ts"], i, "o", ms=9, mec="k", mew=0.8, zorder=3,
                     mfc=col if r["hr"] < 0.05 else "white")
            axd.plot(r["host_ts"], i, "|", ms=13, mew=2.2, color="0.45", zorder=2)
        axd.axvline(0, color="k", lw=0.7)
        axd.set_yticks(y)
        axd.set_yticklabels([f"{r['p']['letter']}  {r['p']['name']} ({r['host']})" for r in rows], fontsize=12)
        axd.set_xlabel(f"Theil–Sen trend ({UNITS})", fontsize=12.5)
        axd.grid(axis="x", alpha=0.3)
        axd.legend(handles=[
            Line2D([0], [0], marker="o", color="w", mfc="0.5", mec="k", ms=9, label="patch, Hamed–Rao p<0.05"),
            Line2D([0], [0], marker="o", color="w", mfc="white", mec="k", ms=9, label="patch, not significant"),
            Line2D([0], [0], color="0.35", lw=1.6, label="95% block-bootstrap CI"),
            Line2D([0], [0], marker="|", color="0.45", lw=0, ms=13, mew=2.2, label="host AR6 region trend")],
            loc="upper center", bbox_to_anchor=(0.35, -0.10), ncol=2, fontsize=10.5, frameon=False)
        axd.text(-0.02, 1.02, "b", transform=axd.transAxes, fontsize=20, fontweight="bold", ha="right")
        out = FIGDIR / f"{var}_dryland_patch_summary.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def _pettitt_changepoint(vals, years):
    """Pettitt-test change-point year of a series (single most-probable breakpoint)."""
    x = np.asarray(vals, float)
    m = np.isfinite(x)
    x, yy = x[m], np.asarray(years)[m]
    n = len(x)
    if n < 12:
        return int(np.median(yy)) if n else np.nan
    r = rankdata(x)
    U = np.array([2 * np.sum(r[:k]) - k * (n + 1) for k in range(1, n + 1)])
    return int(yy[int(np.argmax(np.abs(U)))])


def _auto_hex_layout(items, binlon=13.0, binlat=11.0):
    """Snap region centroids (list of (abbrev, lon, lat)) to a pointy-top offset hex
    grid with greedy nearest-free-cell collision resolution -> abbrev -> (x, y)."""
    occupied, placed = {}, {}
    for ab, lon, lat in sorted(items, key=lambda t: (-t[2], t[1])):
        col0 = int(round((lon + 180) / binlon))
        row0 = int(round((90 - lat) / binlat))
        cell = (col0, row0)
        if cell in occupied:
            cell = None
            for rad in range(1, 9):
                cands = [(col0 + dc, row0 + dr) for dc in range(-rad, rad + 1)
                         for dr in range(-rad, rad + 1)
                         if max(abs(dc), abs(dr)) == rad and (col0 + dc, row0 + dr) not in occupied]
                if cands:
                    cell = min(cands, key=lambda c: (c[0] - col0) ** 2 + (c[1] - row0) ** 2)
                    break
            if cell is None:
                cell = (col0, row0)
        occupied[cell] = ab
        placed[ab] = cell
    r = 1.0
    dx, dy = np.sqrt(3) * r, 1.5 * r
    return {ab: (col * dx + (row % 2) * (dx / 2), -row * dy) for ab, (col, row) in placed.items()}


def changepoint_hex(ctx: _Ctx, var="compound") -> Path:
    """Kim-Fig.4B-style hex-grid cartogram: one hexagon per AR6 region (auto-placed
    geographically), labelled with abbrev + the Pettitt change-point year of its annual
    `var` count, shaded by that year."""
    field = ctx.mean_annual(var)
    reg_da = regions.ar6_land_regions(field)
    ar6 = _ar6_geoms()
    cents = np.asarray(ar6.centroids)
    ab_to_num = dict(zip(ar6.abbrevs, ar6.numbers))
    items, cpyear = [], {}
    for i, ab in enumerate(ar6.abbrevs):
        if ab in _EXCLUDE_REGIONS:
            continue
        s = ctx.region_annual_mean(var, reg_da, int(ab_to_num[ab]))
        cp = _pettitt_changepoint(s.values, s.index.values)
        if not np.isfinite(cp):
            continue
        cpyear[ab] = cp
        items.append((ab, float(cents[i, 0]), float(cents[i, 1])))
    xy = _auto_hex_layout(items)
    yrs = np.array(list(cpyear.values()), float)
    vmin, vmax = float(np.nanpercentile(yrs, 5)), float(np.nanpercentile(yrs, 95))
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap("RdBu_r")

    with plt.rc_context(_PANEL_FONT):
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.set_aspect("equal"); ax.axis("off")
        for ab, (x, y) in xy.items():
            fc = cmap(norm(cpyear[ab]))
            ax.add_patch(RegularPolygon((x, y), numVertices=6, radius=0.97, orientation=0,
                                        facecolor=fc, edgecolor="0.3", lw=0.7))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            tc = "white" if lum < 0.5 else "black"
            ax.text(x, y + 0.22, ab, ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
            ax.text(x, y - 0.28, str(int(cpyear[ab])), ha="center", va="center", fontsize=11, color=tc)
        ax.autoscale_view()
        ax.margins(0.02)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig.add_axes([0.30, 0.06, 0.42, 0.022])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("change-point year of the acceleration", fontsize=15)
        cb.ax.tick_params(labelsize=13)
        fig.suptitle(f"When did {var} frequency start accelerating?  Pettitt change-point year by AR6 region",
                     fontsize=19, y=0.965, fontweight="bold")
        out = FIGDIR / f"{var}_changepoint_hex.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def changepoint_map(ctx: _Ctx, var="compound") -> Path:
    """Geographic (clean AR6-polygon) version of the change-point figure: each region is
    shaded by its Pettitt change-point year and labelled with abbrev + year on the map."""
    field = ctx.mean_annual(var)
    reg_da = regions.ar6_land_regions(field)
    ar6 = _ar6_geoms()
    clip, rep = _ar6_clipped()
    ab_to_num = dict(zip(ar6.abbrevs, ar6.numbers))
    cpyear, cpsig = {}, {}
    for ab in ar6.abbrevs:
        if ab in _EXCLUDE_REGIONS:
            continue
        s = ctx.region_annual_mean(var, reg_da, int(ab_to_num[ab]))
        cp, cpp = _pettitt(s.values, s.index.values)
        cpyear[ab] = cp
        cpsig[ab] = cpp < 0.05
    yrs = np.array(list(cpyear.values()), float)
    vmin, vmax = float(np.nanpercentile(yrs, 5)), float(np.nanpercentile(yrs, 95))
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap("RdBu_r")

    with plt.rc_context(_PANEL_FONT):
        fig = plt.figure(figsize=(19, 10.5))
        ax = fig.add_axes([0.015, 0.12, 0.97, 0.82], projection=ccrs.Robinson())
        ax.set_global(); ax.patch.set_facecolor("white"); ax.spines["geo"].set_linewidth(0.6)
        ax.add_feature(cfeature.LAND, facecolor="#eeece7", zorder=0)
        for ab in ar6.abbrevs:
            if ab in _EXCLUDE_REGIONS:
                continue
            val = cpyear.get(ab, np.nan)
            fc = cmap(norm(val)) if np.isfinite(val) else "#e6e6e6"
            geoms = _polygonal(clip[ab])
            if geoms:
                ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor=fc,
                                  edgecolor="0.25", linewidth=0.4, zorder=1,
                                  hatch=None if cpsig.get(ab, True) else "///")
        ax.add_feature(cfeature.COASTLINE, lw=0.35, zorder=3, edgecolor="0.4")
        for ab in ar6.abbrevs:
            if ab not in cpyear:
                continue
            lon, lat = rep[ab]
            fc = cmap(norm(cpyear[ab]))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            tc = "white" if lum < 0.5 else "black"
            ax.text(lon, lat, ab, transform=ccrs.PlateCarree(), fontsize=13,
                    fontweight="bold", ha="center", va="bottom", color=tc, zorder=4)
            ax.text(lon, lat, str(int(cpyear[ab])) + ("" if cpsig[ab] else " n.s."), transform=ccrs.PlateCarree(),
                    fontsize=12, fontweight="bold", ha="center", va="top", color=tc, zorder=4)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig.add_axes([0.32, 0.085, 0.36, 0.022])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))
        cb.set_label("Pettitt change-point year (hatched / n.s.: p ≥ 0.05)", fontsize=19)
        cb.ax.tick_params(labelsize=16)
        fig.suptitle(f"When did {var} frequency shift upward?  "
                     "Pettitt change-point year by AR6 region", fontsize=23, y=0.98, fontweight="bold")
        out = FIGDIR / f"{var}_changepoint_map.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def export_result_tables(ctx: _Ctx) -> Path:
    """Write the aggregated numbers behind the figures to CSVs (figures/manuscript/tables/)
    for citing in the results section: per-AR6-region stats, per-aridity-class stats, and
    the global land-mean annual series.

    Trends are Theil-Sen (per decade) on the area-weighted mean annual series, i.e. the
    estimator the manuscript states (before 2026-09-23 these columns were OLS np.polyfit
    slopes mislabelled as Theil-Sen; the OLS slope is kept as `*_ols_per_decade` for
    reference). Significance: naive Mann-Kendall p, Hamed-Rao (lag-autocorrelation-
    corrected) p, and the Pettitt change-point p-value."""
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    ar6 = _ar6_geoms()
    tbl = regions.ar6_region_table()
    name_by_num = dict(zip(tbl["region_index"], tbl["name"]))
    reg_da = regions.ar6_land_regions(ctx.mean_annual("compound"))

    rows = []
    for ab, num in zip(ar6.abbrevs, ar6.numbers):
        if ab in _EXCLUDE_REGIONS:
            continue
        num = int(num)
        rec = {"abbrev": ab, "name": name_by_num.get(num, "")}
        for v in ("compound", "heat", "drought"):
            s_ = ctx.region_annual_mean(v, reg_da, num)
            rec[f"{v}_mean"] = round(float(s_.mean()), 4)
            rec[f"{v}_trend_per_decade"] = round(_ts(s_.values, s_.index.values), 4)
            rec[f"{v}_hamed_rao_p"] = round(_hr_p(s_.values), 4)
        sc = ctx.region_annual_mean("compound", reg_da, num)
        yy = sc.index.values
        em = float(sc.loc[[y for y in yy if EARLY_PERIOD[0] <= y <= EARLY_PERIOD[1]]].mean())
        lm = float(sc.loc[[y for y in yy if LATE_PERIOD[0] <= y <= LATE_PERIOD[1]]].mean())
        cp, cpp = _pettitt(sc.values, yy)
        post = yy >= 1979
        wm = (ctx.land & (reg_da == num)).values * np.cos(np.deg2rad(ctx.lat))[:, None]
        rec.update(compound_ols_per_decade=round(float(np.polyfit(yy, sc.values, 1)[0] * 10), 4),
                   compound_mk_p=round(_mk_p(sc.values), 4),
                   compound_trend_1979_2025=round(_ts(sc.values[post], yy[post]), 4),
                   compound_early_mean=round(em, 4), compound_late_mean=round(lm, 4),
                   compound_change=round(lm - em, 4),
                   changepoint_year=cp, changepoint_p=round(cpp, 4),
                   pre_cp_mean=round(float(sc.values[yy <= cp].mean()), 4),
                   post_cp_mean=round(float(sc.values[yy > cp].mean()), 4),
                   area_weighted_log_mean_ai=round(float(np.exp(np.nansum(np.log(np.clip(ctx.ai, 1e-4, None)) * wm) / wm.sum())), 4),
                   dryland_area_fraction=round(float((wm * (ctx.ai < 0.65)).sum() / wm.sum()), 4))
        rows.append(rec)
    rdf = pd.DataFrame(rows).sort_values("compound_trend_per_decade", ascending=False)
    rdf.to_csv(outdir / "regional_stats.csv", index=False)

    aiclass = np.where(np.isfinite(ctx.ai), np.digitize(ctx.ai, drylands.AI_CLASS_EDGES[1:-1]), -1)
    landv = ctx.land.values
    w = np.cos(np.deg2rad(ctx.lat))[:, None]
    arows = []
    for i, label in enumerate(drylands.AI_CLASS_LABELS):
        rec = {"class_letter": chr(97 + i), "class": label,
               "ai_range": f"{drylands.AI_CLASS_EDGES[i]:.2f}-{drylands.AI_CLASS_EDGES[i+1]:.2f}"}
        m = landv & (aiclass == i)
        rec["land_area_share"] = round(float((w * m).sum() / (w * landv).sum()), 4)
        wmask = (w * m)[None]
        for v in ("compound", "heat", "drought"):
            annv = ctx.annual_counts(v).values.astype(float)
            yy = ctx.annual_counts(v)["year"].values
            # nansum + per-year validity-adjusted denominator (robust to any NaN pixel-year)
            valid = np.isfinite(annv)
            num = np.nansum(annv * wmask, axis=(1, 2))
            den = np.sum(wmask * valid, axis=(1, 2))
            s_ = num / den
            rec[f"{v}_mean"] = round(float(s_.mean()), 4)
            rec[f"{v}_trend_per_decade"] = round(_ts(s_, yy), 4)
            rec[f"{v}_hamed_rao_p"] = round(_hr_p(s_), 4)
            if v == "compound":
                rec["compound_trend_1979_2025"] = round(_ts(s_[yy >= 1979], yy[yy >= 1979]), 4)
                sig = (ctx.signif["compound"].values & (ctx.z["compound"].values > 0))
                rec["compound_pixel_share_MKsig_increase"] = round(float((w * m * sig).sum() / (w * m).sum()), 4)
        arows.append(rec)
    pd.DataFrame(arows).to_csv(outdir / "aridity_class_stats.csv", index=False)

    gdf = pd.DataFrame({v: _land_mean_annual(v) for v in ("compound", "heat", "drought")})
    gdf.index.name = "year"
    gdf.round(4).to_csv(outdir / "global_landmean_annual.csv")

    # -- regional attribution / convergence statistics cited in Sec. 3.3 / 3.6 ---------- #
    from scipy.stats import pearsonr, norm as _norm
    c_, h_, d_ = (rdf[f"{v}_trend_per_decade"].values for v in ("compound", "heat", "drought"))
    r_cd, r_ch, r_hd = pearsonr(c_, d_)[0], pearsonr(c_, h_)[0], pearsonr(h_, d_)[0]
    n = len(c_)
    # Steiger (1980) Z for two dependent correlations sharing one variable (Meng-Rosenthal-
    # Rubin 1992 computational form)
    rm2 = (r_cd**2 + r_ch**2) / 2
    f = min((1 - r_hd) / (2 * (1 - rm2)), 1.0)
    hh = (1 - f * rm2) / (1 - rm2)
    Zs = (np.arctanh(r_cd) - np.arctanh(r_ch)) * np.sqrt((n - 3) / (2 * (1 - r_hd) * hh))
    X = np.column_stack([np.ones(n), h_, d_])
    beta = np.linalg.lstsq(X, c_, rcond=None)[0]
    em_, lm_ = rdf["compound_early_mean"].values, rdf["compound_late_mean"].values
    lr = linregress(em_, lm_)
    att = dict(
        n_regions=n, r_compound_drought=r_cd, r_compound_heat=r_ch, r_heat_drought=r_hd,
        steiger_z=Zs, steiger_p=2 * (1 - _norm.cdf(abs(Zs))),
        std_beta_heat=beta[1] * h_.std() / c_.std(), std_beta_drought=beta[2] * d_.std() / c_.std(),
        r_trend_vs_log_ai=pearsonr(c_, np.log(rdf["area_weighted_log_mean_ai"].values))[0],
        p_trend_vs_log_ai=pearsonr(c_, np.log(rdf["area_weighted_log_mean_ai"].values))[1],
        r_trend_vs_dryland_frac=pearsonr(c_, rdf["dryland_area_fraction"].values)[0],
        p_trend_vs_dryland_frac=pearsonr(c_, rdf["dryland_area_fraction"].values)[1],
        r_early_late=pearsonr(em_, lm_)[0], p_early_late=pearsonr(em_, lm_)[1], slope_late_on_early=lr.slope,
        cv_early=em_.std(ddof=0) / em_.mean(), cv_late=lm_.std(ddof=0) / lm_.mean(),
        n_late_gt_early=int((lm_ > em_).sum()),
        n_trend_negative=int((c_ < 0).sum()),
        n_hamed_rao_sig=int((rdf["compound_hamed_rao_p"] < 0.05).sum()),
        n_pettitt_sig=int((rdf["changepoint_p"] < 0.05).sum()),
        n_heat_trend_positive=int((h_ > 0).sum()), n_drought_trend_negative=int((d_ < 0).sum()),
        n_post_cp_gt_pre=int((rdf["post_cp_mean"] > rdf["pre_cp_mean"]).sum()),
    )
    pd.DataFrame([att]).round(4).to_csv(outdir / "regional_attribution_stats.csv", index=False)
    print("wrote tables to", outdir, flush=True)
    return outdir


def export_global_stats(ctx: _Ctx) -> Path:
    """Every global-series number cited in the text, in one traceable CSV
    (figures/manuscript/tables/global_stats.csv): OLS & Theil-Sen trends, block-bootstrap
    CI, naive MK and Hamed-Rao tests, lag-1 residual autocorrelation, early/late means,
    sub-period (1950-1978, 1979-2025) Theil-Sen trends, Pettitt years with p-values and a
    joint residual block-bootstrap interval for the break years, and the per-pixel
    significant-area shares. Also the SPEI-1 tail frequency on land."""
    import pymannkendall as pmk
    from statsmodels.nonparametric.smoothers_lowess import lowess
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    ser = {v: _land_mean_annual(v) for v in ("compound", "heat", "drought")}
    yrs = ser["compound"].index.values.astype(float)
    rows = []
    for v, sr in ser.items():
        y = sr.values
        ols = linregress(yrs, y)
        mk = pmk.original_test(y); hr = pmk.hamed_rao_modification_test(y)
        res = y - np.polyval(np.polyfit(yrs, y, 1), yrs)
        lo, hi = _block_boot_ci(y, yrs)
        cp, cpp = _pettitt(y, yrs)
        e = y[(yrs >= EARLY_PERIOD[0]) & (yrs <= EARLY_PERIOD[1])].mean()
        l = y[(yrs >= LATE_PERIOD[0]) & (yrs <= LATE_PERIOD[1])].mean()
        pre, post = yrs <= 1978, yrs >= 1979
        rows.append(dict(
            series=v, ols_per_decade=ols.slope * 10, ols_r=ols.rvalue, ols_p=ols.pvalue,
            theilsen_per_decade=_ts(y, yrs), ci_lo=lo, ci_hi=hi,
            mk_z=mk.z, mk_p=mk.p, hamed_rao_z=hr.z, hamed_rao_p=hr.p,
            rho1_detrended=np.corrcoef(res[:-1], res[1:])[0, 1],
            early_mean=e, late_mean=l, pct_change=100 * (l - e) / e,
            theilsen_1950_1978=_ts(y[pre], yrs[pre]), mk_p_1950_1978=_mk_p(y[pre]),
            theilsen_1979_2025=_ts(y[post], yrs[post]), hamed_rao_p_1979_2025=_hr_p(y[post]),
            pettitt_year=cp, pettitt_p=cpp))
    df = pd.DataFrame(rows)
    # joint residual block bootstrap around LOWESS fits for break-year uncertainty
    fits = {v: lowess(ser[v].values, yrs, frac=0.3, return_sorted=False) for v in ser}
    resid = {v: ser[v].values - fits[v] for v in ser}
    rng = np.random.default_rng(0); B, L = 2000, 5; n = len(yrs); nb = int(np.ceil(n / L))
    bt = {v: np.empty(B, int) for v in ser}
    for i in range(B):
        st = rng.integers(0, n - L + 1, nb)
        for v in ser:
            rs = np.concatenate([resid[v][s0:s0 + L] for s0 in st])[:n]
            bt[v][i] = _pettitt(fits[v] + rs, yrs)[0]
    df["pettitt_boot_p05"] = [np.percentile(bt[v], 5) for v in df.series]
    df["pettitt_boot_p95"] = [np.percentile(bt[v], 95) for v in df.series]
    df["p_heat_break_before_compound"] = np.mean(bt["heat"] < bt["compound"])
    df["p_compound_break_before_drought"] = np.mean(bt["compound"] < bt["drought"])
    df["p_heat_break_before_drought"] = np.mean(bt["heat"] < bt["drought"])
    # per-pixel significant-area shares (area-weighted, analysis land)
    w = np.cos(np.deg2rad(ctx.lat))[:, None] * ctx.land.values
    shares = {}
    for v in ("compound", "heat", "drought"):
        ctx.slope_of(v)
        z = _open_trend(v)["z"].values; sl = ctx.slope_of(v).values
        ok = np.isfinite(z) & (w > 0)
        W = w * ok
        shares[f"{v}_share_MKsig_increase"] = float((W * ((z > 1.959964))).sum() / W.sum())
        shares[f"{v}_share_MKsig_decrease"] = float((W * ((z < -1.959964))).sum() / W.sum())
        shares[f"{v}_share_theilsen_zero"] = float((W * (sl == 0)).sum() / W.sum())
        shares[f"{v}_share_theilsen_positive"] = float((W * (sl > 0)).sum() / W.sum())
    for k, v in shares.items():
        df[k] = v
    # SPEI-1 tail frequency over analysis land (defined grid-months)
    sp = xr.open_zarr(C.spei_zarr(C.SPEI_CALIB_PERIODS[CALIB]))["spei1"]
    fin = np.isfinite(sp.values)
    dr = (sp.values <= C.DROUGHT_THRESHOLD) & fin
    df["spei1_frac_le_minus1_land"] = float((dr * w[None]).sum() / (fin * w[None]).sum())
    df["spei1_frac_undefined_land"] = float(((~fin) * w[None]).sum() / (w[None].sum() * fin.shape[0]))
    comp = ctx.compound_l5()
    df["compound_frac_undefined_land"] = float((np.isnan(comp.values) * w[None]).sum() / (w[None].sum() * comp.shape[0]))
    df.round(5).to_csv(outdir / "global_stats.csv", index=False)
    print("wrote", outdir / "global_stats.csv", flush=True)
    return outdir / "global_stats.csv"


def export_onset_null_table(ctx: _Ctx, n_surrogates: int = 3) -> Path:
    """Onset-order statistics on analysis land (Antarctica excluded; earlier drafts quoted
    all-grid shares that included ocean pixels) for the observed fields and for a
    base-rate/persistence null (leading.shift_drought_surrogate) -> onset_order_null.csv.
    The paper reports the onset-order analysis only against this null."""
    base = C.HW_BASELINES[HW]; calib = C.SPEI_CALIB_PERIODS[CALIB]
    ds = xr.open_zarr(C.compound_zarr(base, TS, calib))
    idx = np.where(ctx.land.values.ravel())[0]
    H = ds["heat"].values.reshape(ds.sizes["time"], -1)[:, idx]
    D = ds["drought"].values.reshape(ds.sizes["time"], -1)[:, idx]
    w = np.repeat(np.cos(np.deg2rad(ctx.lat))[:, None], len(ctx.lon), 1).ravel()[idx]
    years = np.unique(ds["time"].dt.year.values)
    rows = [dict(case="observed", **leading.onset_summary(H, D, w, years))]
    for k in range(n_surrogates):
        rows.append(dict(case=f"null_shift_{k}",
                         **leading.onset_summary(H, leading.shift_drought_surrogate(D, seed=k), w, years)))
    df = pd.DataFrame(rows)
    nulls = df[df.case.str.startswith("null")].drop(columns="case").mean()
    df = pd.concat([df, pd.DataFrame([dict(case="null_mean", **nulls.to_dict())])], ignore_index=True)
    out = FIGDIR / "tables" / "onset_order_null.csv"
    df.round(4).to_csv(out, index=False)
    print("wrote", out, flush=True)
    return out


def export_aridity_driver_decomposition(ctx: _Ctx, B: int = 5000) -> Path:
    """What drives the aridity gradient? (added 2026-09-23 final check)

    (1) Precipitation vs evaporative-demand contributions. For each calendar month and pixel
    the SAME GLO distribution fitted to the real D = P - PET (as in spei.py) is used to
    transform two counterfactual water balances in which only one component varies:
        D_P   = P - PET_clim      (precipitation varies, PET at its calendar-month mean)
        D_PET = P_clim - PET      (PET varies, precipitation at its calendar-month mean)
    giving drought months (SPEI-1 <= -1) and compound months (heat AND drought) whose class-
    mean Theil-Sen trends show which component carries the trend. Not additive (the
    transform is nonlinear), so read as attribution-by-exclusion, not a budget.
    (2) Heat-drought dependence per class: P(drought | heatwave month), P(drought | no HW).
    (3) Significance of each class's compound trend relative to humid land: Theil-Sen trend
    of the paired difference series with a 95% residual block-bootstrap CI and Hamed-Rao p.
    -> aridity_driver_decomposition.csv, aridity_class_differences.csv"""
    import pymannkendall as pmk
    from scipy.special import ndtri
    from chd import spei as spei_mod
    outdir = FIGDIR / "tables"
    mon = xr.open_zarr(C.MONTHLY_ZARR)
    P = mon["P"].values; PET = mon["pet"].values; D = mon["D"].values
    T = D.shape[0]; ny = T // 12
    m = np.arange(T) % 12
    drt = {k: np.zeros(D.shape, bool) for k in ("actual", "P_only", "PET_only")}
    for mm in range(12):
        idx = m == mm
        xi, a, k, l2 = spei_mod._glo_params(*spei_mod._pwm(np.sort(D[idx], axis=0)))
        Pc = P[idx].mean(0); PETc = PET[idx].mean(0)
        for key, Dx in (("actual", D[idx]), ("P_only", P[idx] - PETc), ("PET_only", Pc - PET[idx])):
            z = ndtri(np.clip(spei_mod._glo_cdf(Dx, xi, a, k), 1e-6, 1 - 1e-6))
            drt[key][idx] = z <= C.DROUGHT_THRESHOLD
    del P, PET, D
    base = C.HW_BASELINES[HW]; calib = C.SPEI_CALIB_PERIODS[CALIB]
    heat = xr.open_zarr(C.compound_zarr(base, TS, calib))["heat"].values.astype(bool)
    yrs = np.arange(C.YEAR_START, C.YEAR_START + ny).astype(float)
    w2 = np.cos(np.deg2rad(ctx.lat))[:, None] * np.ones((1, len(ctx.lon)))
    landv = ctx.land.values
    edges = drylands.AI_CLASS_EDGES
    groups = [(drylands.AI_CLASS_LABELS[i], landv & (ctx.ai >= edges[i]) & (ctx.ai < edges[i + 1]))
              for i in range(5)] + [("drylands (AI<0.65)", landv & (ctx.ai < 0.65))]
    rows, comp_series = [], {}
    for name, msk in groups:
        W = w2 * msk
        rec = {"class": name}
        for key in drt:
            dc = (drt[key].reshape(ny, 12, *msk.shape).sum(1) * W[None]).sum((1, 2)) / W.sum()
            cc = ((drt[key] & heat).reshape(ny, 12, *msk.shape).sum(1) * W[None]).sum((1, 2)) / W.sum()
            rec[f"drought_trend_{key}"] = _ts(dc, yrs)
            rec[f"compound_trend_{key}"] = _ts(cc, yrs)
            if key == "actual":
                comp_series[name] = cc
        hk = heat[:, msk]; dk = drt["actual"][:, msk]; ww = W[msk]
        rec["p_drought_given_heat"] = float(((hk & dk) * ww).sum() / (hk * ww).sum())
        rec["p_drought_given_noheat"] = float(((~hk & dk) * ww).sum() / ((~hk) * ww).sum())
        rows.append(rec)
    pd.DataFrame(rows).round(4).to_csv(outdir / "aridity_driver_decomposition.csv", index=False)
    hum = comp_series["humid"]; drows = []
    for name, y in comp_series.items():
        if name == "humid":
            continue
        d = y - hum
        lo, hi = _block_boot_ci(d, yrs, B=B)
        drows.append({"class_minus_humid": name, "diff_trend_per_decade": round(_ts(d, yrs), 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      "hamed_rao_p": round(float(pmk.hamed_rao_modification_test(d).p), 4)})
    pd.DataFrame(drows).to_csv(outdir / "aridity_class_differences.csv", index=False)
    print("wrote aridity driver decomposition tables", flush=True)
    return outdir / "aridity_driver_decomposition.csv"


def export_dryland_patch_table(ctx: _Ctx, var="compound") -> Path:
    """Write the per-named-desert-patch numbers behind Fig. dryland_patches (same
    `_build_dryland_patches()` used by the figure itself, so these can be cited
    directly rather than read off the plot) to
    figures/manuscript/tables/dryland_patch_stats.csv, including each patch's
    dominant (by pixel count) AR6 land region for cross-referencing against
    regional_stats.csv."""
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    patches, yrs = _build_dryland_patches(ctx, var)
    ar6 = _ar6_geoms()
    tbl = regions.ar6_region_table()
    name_by_num = dict(zip(tbl["region_index"], tbl["name"]))
    ab_by_num = dict(zip((int(n) for n in ar6.numbers), ar6.abbrevs))
    reg_da = regions.ar6_land_regions(ctx.mean_annual(var))
    reg_vals = reg_da.values

    rows = []
    for p in patches:
        s = p["series"]
        em = float(np.mean(s[(yrs >= EARLY_PERIOD[0]) & (yrs <= EARLY_PERIOD[1])]))
        lm = float(np.mean(s[(yrs >= LATE_PERIOD[0]) & (yrs <= LATE_PERIOD[1])]))
        trend = _ts(s, yrs)
        lo, hi = _block_boot_ci(s, yrs.astype(float))
        cp, cpp = _pettitt(s, yrs)
        # pre/post-change-point split, identical to what the boxplot panels draw
        pre, post = s[yrs <= p["cp"]], s[yrs > p["cp"]]
        sub = reg_vals[p["mask"]]
        sub = sub[np.isfinite(sub)]
        dom_num = int(pd.Series(sub).mode().iloc[0]) if sub.size else -1
        rows.append(dict(
            letter=p["letter"], name=p["name"], size_pixels=p["size"],
            centroid_lon=round(p["clon"], 2), centroid_lat=round(p["clat"], 2),
            mean=round(p["mean"], 4), trend_per_decade=round(trend, 4),
            trend_ci_lo=round(lo, 4), trend_ci_hi=round(hi, 4),
            ols_per_decade=round(float(np.polyfit(yrs, s, 1)[0] * 10), 4),
            mk_p=round(_mk_p(s), 4), hamed_rao_p=round(_hr_p(s), 4),
            trend_1979_2025=round(_ts(s[yrs >= 1979], yrs[yrs >= 1979]), 4),
            changepoint_p=round(cpp, 4),
            early_mean=round(em, 4), late_mean=round(lm, 4), change=round(lm - em, 4),
            pct_change=round(100 * (lm - em) / em, 1) if em > 0 else np.nan,
            changepoint_year=int(p["cp"]),
            pre_cp_median=round(float(np.median(pre)), 4),
            post_cp_median=round(float(np.median(post)), 4),
            pre_cp_mean=round(float(np.mean(pre)), 4),
            post_cp_mean=round(float(np.mean(post)), 4),
            ar6_abbrev=ab_by_num.get(dom_num, ""), ar6_name=name_by_num.get(dom_num, ""),
        ))
    df = pd.DataFrame(rows).sort_values("trend_per_decade", ascending=False)
    df.to_csv(outdir / "dryland_patch_stats.csv", index=False)
    print("wrote", outdir / "dryland_patch_stats.csv", flush=True)
    return outdir / "dryland_patch_stats.csv"


# =========================================================================== #
# Agricultural exposure & growing-season timing (MIRCA2000)
# =========================================================================== #
# exposure classes -> (weight-var in cropland_layers, colour, label)
_AG_CLASSES = [
    ("__land__", "#7f8c8d", "Global land"),
    ("frac_total", "#8a6d3b", "All cropland"),
    ("frac_rainfed", "#2e7d32", "Rainfed cropland"),
    ("frac_irrigated", "#1f6f9c", "Irrigated cropland"),
]


def _ag_weight(ctx, key):
    """cos-lat x exposure weight (DataArray on the ctx grid) for an `_AG_CLASSES` key."""
    coslat = np.cos(np.deg2rad(ctx.slope_of("compound")["latitude"]))
    if key == "__land__":
        return coslat * ctx.land
    return coslat * ctx.crop()[key]


def _ag_series(counts, w):
    """Exposure-weighted global annual series (year -> mean count). xarray's weighted
    mean (skipna) drops NaN-count pixels from both numerator and weight sum, so the
    handful of all-SPEI-undefined pixels never poison the average."""
    s = counts.weighted(w).mean(("latitude", "longitude"))
    return pd.Series(s.values, index=s["year"].values)


def _block_boot_ci(y, yrs, B=5000, L=5, seed=0):
    """95% CI on the Theil-Sen slope (per decade) via a trend-preserving 5-year block
    bootstrap of the residuals, matching the main-text global-series convention
    (Sec. 2.7)."""
    rng = np.random.default_rng(seed)
    b, a = theilslopes(y, yrs)[:2]
    fit = a + b * yrs
    r = y - fit
    n = len(y)
    nb = int(np.ceil(n / L))
    out = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - L + 1, nb)
        rs = np.concatenate([r[s:s + L] for s in st])[:n]
        out[i] = theilslopes(fit + rs, yrs)[0]
    return np.percentile(out, [2.5, 97.5]) * 10


def _compound_z() -> xr.DataArray:
    """Per-pixel Mann-Kendall z of the full-record annual compound trend (the L6 `z`
    field). Preferred over the Theil-Sen slope for the cropland map: the slope is exactly
    0 on ~71% of cropland (median-of-pairwise-slopes on zero-inflated counts), so a slope
    map reads mostly blank, whereas z shows the actual significance structure."""
    baseline = C.HW_BASELINES[HW]
    calib = C.SPEI_CALIB_PERIODS[CALIB]
    ds = xr.open_zarr(C.trends_zarr(baseline, TS, calib, "compound"))
    idx = list(ds.attrs["case_names"]).index(CASE)
    return ds.isel(case=idx)["z"].load()


def growing_season_impact(ctx: _Ctx) -> Path:
    """Food-security figure (exposure/vulnerability framing): (a) Mann-Kendall z of the
    compound trend over cropland only; (b) compound trend by exposure class with 95%
    block-bootstrap CIs (rainfed and irrigated overlap); (c) rainfed vs irrigated
    cropland-weighted annual compound series, whose trends are indistinguishable since
    ~1970. No causal "irrigation buffers" claim and no growing-season "timing" claim are
    made here (neither is supported by this overlay; see manuscript Sec. 3.9)."""
    cnt_all = ctx.annual_counts("compound")
    yrs = cnt_all["year"].values.astype(float)
    cl = ctx.crop()
    crop_land = cl["cropland_mask"].values & ctx.land.values

    fig = plt.figure(figsize=(14, 10.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 0.9], hspace=0.28, wspace=0.2,
                  left=0.06, right=0.95, top=0.92, bottom=0.08)

    # (a) MK z of compound trend over cropland only ------------------------ #
    crop_da = xr.DataArray(crop_land, coords=ctx.slope_of("compound").coords)
    z = _compound_z().where(crop_da)
    zv = np.abs(z.values); zv = zv[np.isfinite(zv)]
    vmax = float(np.nanpercentile(zv, 98)) if zv.size else 4.0
    ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    _basemap(ax)
    m = ax.pcolormesh(ctx.lon, ctx.lat, z.values, transform=ccrs.PlateCarree(),
                      cmap=CMAP, vmin=-vmax, vmax=vmax, zorder=1, rasterized=True)
    ax.contour(ctx.lon2d, ctx.lat2d, ctx.desert.astype(float), levels=[0.5],
               colors=DESERT_LINE, linewidths=0.35, transform=ccrs.PlateCarree(), zorder=3)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
    _letter(ax, "a")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.7, pad=0.015, extend="both")
    cb.set_label("Mann-Kendall $z$ of compound trend", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_title("Compound trend over cropland only "
                 "(MIRCA2000; $|z|>1.96$ = significant at $p<0.05$)", fontsize=11)

    # (b) trend by exposure class with 95% bootstrap CI -------------------- #
    axb = fig.add_subplot(gs[1, 0])
    labels, trend, err_lo, err_hi = [], [], [], []
    for key, col, lab in _AG_CLASSES:
        s = _ag_series(cnt_all, _ag_weight(ctx, key)).values
        b = theilslopes(s, yrs)[0] * 10
        lo, hi = _block_boot_ci(s, yrs)
        labels.append(lab); trend.append(b)
        err_lo.append(b - lo); err_hi.append(hi - b)
    xpos = np.arange(len(labels))
    cols = [c for _k, c, _l in _AG_CLASSES]
    axb.bar(xpos, trend, width=0.6, color=cols, alpha=0.85)
    axb.errorbar(xpos, trend, yerr=[err_lo, err_hi], fmt="none", ecolor="0.2",
                 capsize=4, lw=1.1)
    for xi, b in enumerate(trend):
        axb.text(xi, b + err_hi[xi] + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(xpos)
    axb.set_xticklabels([l.replace(" cropland", "").replace(" ", "\n") for l in labels], fontsize=9)
    axb.set_ylabel(f"compound trend ({UNITS})", fontsize=9.5)
    axb.axhline(0, color="0.6", lw=0.6)
    axb.grid(axis="y", alpha=0.25, lw=0.5)
    axb.set_title("Compound trend by exposure class (95% CI)", fontsize=11)
    _letter(axb, "b", inside=False)

    # (c) rainfed vs irrigated cropland-weighted annual series ------------- #
    axc = fig.add_subplot(gs[1, 1])
    for key, col, lab in [("frac_rainfed", "#2e7d32", "rainfed cropland"),
                          ("frac_irrigated", "#1f6f9c", "irrigated cropland")]:
        s = _ag_series(cnt_all, _ag_weight(ctx, key))
        b1, a1 = theilslopes(s.values, yrs)[:2]
        axc.plot(yrs, s.values, lw=1.5, color=col, label=f"{lab} ({b1*10:+.3f}/dec)")
        axc.plot(yrs, a1 + b1 * yrs, ls="--", lw=0.9, color=col, alpha=0.75)
    axc.axvline(1970, color="0.6", lw=0.8, ls=":")
    axc.set_xlim(1950, 2025)
    axc.set_xlabel("year", fontsize=9.5)
    axc.set_ylabel("cropland-weighted\ncompound event-months yr$^{-1}$", fontsize=9.5)
    axc.grid(alpha=0.25, lw=0.5)
    axc.legend(fontsize=8.5, frameon=False, loc="upper left")
    axc.set_title("Rainfed vs irrigated (indistinguishable since ~1970)", fontsize=10.5)
    _letter(axc, "c", inside=False)

    fig.suptitle("Compound heatwave–drought intensification on the world's cropland\n"
                 "HW 1961–1990 · SPEI-1 · calibration 1950–2025 · cropland from MIRCA2000",
                 fontsize=13, y=0.985)
    out = FIGDIR / "agriculture_impact.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_agriculture_tables(ctx: _Ctx) -> Path:
    """Write the numbers behind the agriculture figure to
    figures/manuscript/tables/agriculture_exposure.csv (per exposure class: Theil-Sen
    trend + 95% block-bootstrap CI + Hamed-Rao p, absolute early/late 30-yr means and
    change) and agriculture_summary.csv (cropland MK-significant share via z>0, the
    post-1970 rainfed-minus-irrigated difference trend, the hyper-arid-excluded land
    trend, and the growing-season proportionality check)."""
    import pymannkendall as pmk
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    comp = ctx.compound_l5()
    win = C.TREND_WINDOWS["1950_2025"]
    cnt_all = ctx.annual_counts("compound")
    cnt_gs = trendsmod.annual_counts(comp, win, gs_month_mask=ctx.gsmask()["gs_any"])
    yrs = cnt_all["year"].values.astype(float)

    def _period(s, y0, y1):
        return float(s.loc[[y for y in s.index if y0 <= y <= y1]].mean())

    series = {}
    rows = []
    for key, _c, lab in _AG_CLASSES:
        sa = _ag_series(cnt_all, _ag_weight(ctx, key))
        series[lab] = sa.values
        em, lm = _period(sa, *EARLY_PERIOD), _period(sa, *LATE_PERIOD)
        lo, hi = _block_boot_ci(sa.values, yrs)
        rows.append(dict(
            exposure_class=lab,
            trend_per_decade=round(theilslopes(sa.values, yrs)[0] * 10, 4),
            ci_lo=round(lo, 3), ci_hi=round(hi, 3),
            hamed_rao_p=round(float(pmk.hamed_rao_modification_test(sa.values).p), 4),
            trend_growingseason_per_decade=round(
                theilslopes(_ag_series(cnt_gs, _ag_weight(ctx, key)).values, yrs)[0] * 10, 4),
            early_mean=round(em, 4), late_mean=round(lm, 4), change=round(lm - em, 4),
            pct_change=round(100 * (lm - em) / em, 1) if em > 0 else np.nan,
        ))
    pd.DataFrame(rows).to_csv(outdir / "agriculture_exposure.csv", index=False)

    # -- rainfed-minus-irrigated difference: whole record vs post-1979 ----- #
    # The raw-record gap is a pre-1970 feature; from the satellite era (>=1979, the
    # paper's own secondary window) rainfed and irrigated trends are indistinguishable.
    diff = series["Rainfed cropland"] - series["Irrigated cropland"]
    post = yrs >= 1979
    diff_all_p = float(pmk.hamed_rao_modification_test(diff).p)
    diff_post_slope = float(theilslopes(diff[post], yrs[post])[0] * 10)
    diff_post_p = float(pmk.hamed_rao_modification_test(diff[post]).p)
    rf_post = float(theilslopes(series["Rainfed cropland"][post], yrs[post])[0] * 10)
    ir_post = float(theilslopes(series["Irrigated cropland"][post], yrs[post])[0] * 10)
    diff_early_mean = float(diff[yrs <= 1967].mean())
    diff_late_mean = float(diff[yrs >= 1968].mean())

    # -- cropland MK-significant share via z>0 (not Theil-Sen slope>0) ----- #
    cl = ctx.crop()
    coslat = np.cos(np.deg2rad(ctx.slope_of("compound")["latitude"]))
    w_crop = coslat * cl["frac_total"]
    crop_land = cl["cropland_mask"] & ctx.land
    z = _compound_z()
    sig_up = (ctx.signif["compound"] & (z > 0)) & crop_land
    pct_sig_up = float((sig_up * w_crop).sum() / (crop_land * w_crop).sum() * 100)
    okz = crop_land & np.isfinite(z)
    pct_sig_down = float(((ctx.signif["compound"] & (z < 0) & okz) * w_crop).sum() / (okz * w_crop).sum() * 100)
    pct_ts_zero = float(((ctx.slope_of("compound") == 0) & okz).__mul__(w_crop).sum() / (okz * w_crop).sum() * 100)

    # -- cropland-trend-below-land is only ~half hyper-arid ---------------- #
    def _land_trend(mask2d):
        s = _ag_series(cnt_all, coslat * xr.DataArray(mask2d, coords=z.coords))
        return theilslopes(s.values, yrs)[0] * 10
    land_all = _land_trend(ctx.land.values)
    land_no_hyper = _land_trend(ctx.land.values & ~(ctx.ai < 0.05))

    # -- growing-season proportionality (no timing enrichment) ------------- #
    gs_len = float(ctx.gsmask()["gs_any"].sum("month").where(crop_land).mean())
    insea = ctx.gsmask()["gs_any"].sel(month=comp["time"].dt.month)
    frac_pixel = (comp.where(insea).sum("time", skipna=True) /
                  comp.sum("time", skipna=True)).where(comp.sum("time", skipna=True) > 0)
    insea_share = float((frac_pixel * w_crop).sum() / w_crop.where(np.isfinite(frac_pixel)).sum())
    gs_trend = theilslopes(_ag_series(cnt_gs, w_crop).values, yrs)[0] * 10
    all_trend = theilslopes(_ag_series(cnt_all, w_crop).values, yrs)[0] * 10

    # -- confounding of the whole-record rainfed>irrigated gap ------------- #
    reg_da = regions.ar6_land_regions(ctx.slope_of("compound"))
    rtab = regions.ar6_region_table()
    inv = {ab: int(n) for ab, n in zip(rtab["abbrev"], rtab["region_index"])}
    wi = (coslat * cl["frac_irrigated"]).where(ctx.land)
    sas_eas = reg_da.isin([inv["SAS"], inv["EAS"]])
    irr_share_sas_eas = float(wi.where(sas_eas).sum() / wi.sum())
    keep = ~reg_da.isin([inv["SAS"]])
    rf_exsas = theilslopes(_ag_series(cnt_all, (coslat * cl["frac_rainfed"]).where(keep, 0.0)).values, yrs)[0] * 10
    ir_exsas = theilslopes(_ag_series(cnt_all, (coslat * cl["frac_irrigated"]).where(keep, 0.0)).values, yrs)[0] * 10
    # pixel-level slope ~ rainfed_fraction, pooled and with AR6-region fixed effects
    # per-pixel OLS slope of the annual count (per decade). The per-pixel Theil-Sen slope
    # is exactly 0 on ~70% of cropland (zero-inflated counts), which would make it a poor
    # dependent variable for this regression (changed 2026-09-23).
    A_ = cnt_all.values.astype(float)
    yc = yrs - yrs.mean()
    slope = (np.nansum(A_ * yc[:, None, None], axis=0) / np.sum(yc**2)) * 10
    slope = np.where(np.isfinite(A_).all(0), slope, np.nan)
    rfrac = cl["rainfed_fraction"].values
    regv = reg_da.values
    okp = (crop_land.values & np.isfinite(slope) & np.isfinite(rfrac) & np.isfinite(regv))
    wv = (coslat * cl["frac_total"]).values[okp]
    sv, rv, gv = slope[okp], rfrac[okp], regv[okp]
    def _wdemean(x, groups, w):
        out = x.astype(float).copy()
        for r in np.unique(groups):
            mm = groups == r
            out[mm] = x[mm] - np.average(x[mm], weights=w[mm])
        return out
    coef_pool = float(np.sum(wv * _wdemean(rv, np.zeros_like(gv), wv) * _wdemean(sv, np.zeros_like(gv), wv))
                      / np.sum(wv * _wdemean(rv, np.zeros_like(gv), wv) ** 2))
    xfe, yfe = _wdemean(rv, gv, wv), _wdemean(sv, gv, wv)
    coef_fe = float(np.sum(wv * xfe * yfe) / np.sum(wv * xfe * xfe))
    pd.DataFrame([dict(
        irrig_weight_share_SAS_EAS=round(irr_share_sas_eas, 3),
        rainfed_trend_excl_SAS=round(rf_exsas, 4), irrig_trend_excl_SAS=round(ir_exsas, 4),
        slope_on_rainfedfrac_pooled=round(coef_pool, 4),
        slope_on_rainfedfrac_regionFE=round(coef_fe, 4),
    )]).to_csv(outdir / "agriculture_confounding.csv", index=False)

    summ = pd.DataFrame([dict(
        cropland_pct_MKsig_up_z=round(pct_sig_up, 1),
        cropland_pct_MKsig_down_z=round(pct_sig_down, 1),
        cropland_pct_theilsen_zero=round(pct_ts_zero, 1),
        rainfed_minus_irrig_p_allrecord=round(diff_all_p, 4),
        diff_mean_pre1968=round(diff_early_mean, 3),
        diff_mean_post1968=round(diff_late_mean, 3),
        rainfed_trend_post1979=round(rf_post, 4),
        irrig_trend_post1979=round(ir_post, 4),
        rainfed_minus_irrig_trend_post1979=round(diff_post_slope, 4),
        rainfed_minus_irrig_p_post1979=round(diff_post_p, 3),
        land_trend=round(land_all, 4),
        land_trend_excl_hyperarid=round(land_no_hyper, 4),
        mean_growing_season_months=round(gs_len, 2),
        in_season_compound_share_pct=round(insea_share * 100, 1),
        gs_trend_per_season_month=round(gs_trend / gs_len, 4),
        all_trend_per_month=round(all_trend / 12, 4),
    )])
    summ.to_csv(outdir / "agriculture_summary.csv", index=False)
    print("wrote agriculture tables to", outdir, flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(summ.to_string(index=False), flush=True)
    return outdir


# --------------------------------------------------------------------------- #
# Population exposure (LandScan Global 2024; 2000 for the decomposition)
# --------------------------------------------------------------------------- #
POP_YEAR = 2024
POP_YEAR_EARLY = 2000          # LandScan Mosaic Time Series layer for the decomposition
# 15-yr hazard windows paired with the two LandScan snapshots for the exposure
# decomposition: centred on 2000, and the most recent 15 years (nearest to 2024).
POP_HAZ_EARLY = (1993, 2007)
POP_HAZ_LATE = (2011, 2025)
_DENSITY_CLASSES = [("pop_urban", "#5b3a8c", "urban (≥1,500 km$^{-2}$)"),
                    ("pop_intermediate", "#9b7fc4", "intermediate (300–1,500)"),
                    ("pop_rural", "#cdbfe3", "rural (<300 km$^{-2}$)")]


def _pop(ctx: _Ctx, year: int = POP_YEAR) -> xr.Dataset:
    """LandScan population on the ctx grid, restricted to analysis land (ex-Antarctica).
    Cells whose centre is ocean under the land mask are dropped (~3.7% of 2024 people,
    mostly coastal); the retained share is reported in population_summary.csv."""
    from chd import population as popmod
    cache = getattr(ctx, "_pop", {})
    if year not in cache:
        ds = popmod.landscan(ctx.slope_of("compound"), year)
        cache[year] = ds.where(ctx.land, 0.0)
        ctx._pop = cache
    return cache[year]


def _pop_mosaic(ctx: _Ctx, year: int) -> xr.DataArray:
    """LandScan Mosaic Time Series population (temporally consistent) on the ctx grid,
    restricted to analysis land. Used only for the 2000 -> 2024 exposure decomposition."""
    from chd import population as popmod
    cache = getattr(ctx, "_popm", {})
    if year not in cache:
        cache[year] = popmod.landscan_mosaic_ts(ctx.slope_of("compound"), year).where(ctx.land, 0.0)
        ctx._popm = cache
    return cache[year]


def _pop_series(counts: xr.DataArray, w: xr.DataArray) -> pd.Series:
    """Population-weighted mean annual count (event-months yr^-1 per person). No cos-lat
    factor: LandScan counts are already per-cell totals, so the cell area is built in."""
    return _ag_series(counts, w)


def _pop_classes(ctx: _Ctx) -> list[tuple[str, str, str, xr.DataArray]]:
    """(group, label, colour, population-weight DataArray) for the aridity and density
    classes, in plotting order."""
    p = _pop(ctx)
    ai = xr.DataArray(ctx.ai, coords=p["pop"].coords)
    out = []
    edges, labs = drylands.AI_CLASS_EDGES, drylands.AI_CLASS_LABELS
    for i, lab in enumerate(labs):
        m = (ai >= edges[i]) & (ai < edges[i + 1])
        out.append(("aridity", lab, AI_COLORS[i], p["pop"].where(m, 0.0)))
    out.append(("aridity", "drylands (AI<0.65)", "#a0522d", p["pop"].where(ai < 0.65, 0.0)))
    for key, col, lab in _DENSITY_CLASSES:
        out.append(("density", lab, col, p[key]))
    return out


def export_population_tables(ctx: _Ctx) -> Path:
    """Numbers behind the population paragraph, all to figures/manuscript/tables/:
    population_summary.csv (global per-person vs per-area trends for compound/heat/
    drought, person-exposure early/late, share of people under MK-significant increase,
    2000->2024 exposure decomposition), population_classes.csv (per aridity and 1-km
    density class: population share, per-person trend + CI + Hamed-Rao p, early/late),
    population_regions.csv (per AR6 region: population, per-person trend, exposure
    change)."""
    outdir = FIGDIR / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    p = _pop(ctx)["pop"]
    P_tot = float(p.sum())
    from chd import population as popmod
    P_world = float(popmod.landscan(ctx.slope_of("compound"), POP_YEAR).attrs["global_total"])
    coslat = np.cos(np.deg2rad(ctx.slope_of("compound")["latitude"]))
    w_area = coslat * ctx.land
    yrs = ctx.annual_counts("compound")["year"].values.astype(float)
    post = yrs >= 1979

    def _period(s, a, b):
        return float(s[(s.index >= a) & (s.index <= b)].mean())

    summ = dict(pop_year=POP_YEAR, pop_world_bn=P_world / 1e9, pop_analysis_land_bn=P_tot / 1e9,
                pop_share_on_analysis_land=P_tot / P_world)
    for v in ("compound", "heat", "drought"):
        cnt = ctx.annual_counts(v)
        sp = _pop_series(cnt, p)
        sa = _ag_series(cnt, w_area)
        lo, hi = _block_boot_ci(sp.values, yrs)
        summ.update({
            f"{v}_perperson_trend": _ts(sp.values, yrs), f"{v}_perperson_ci_lo": lo,
            f"{v}_perperson_ci_hi": hi, f"{v}_perperson_hr_p": _hr_p(sp.values),
            f"{v}_perperson_trend_1979_2025": _ts(sp.values[post], yrs[post]),
            f"{v}_perperson_early": _period(sp, *EARLY_PERIOD),
            f"{v}_perperson_late": _period(sp, *LATE_PERIOD),
            f"{v}_perarea_trend": _ts(sa.values, yrs),
            f"{v}_perarea_early": _period(sa, *EARLY_PERIOD),
            f"{v}_perarea_late": _period(sa, *LATE_PERIOD),
        })
        if v == "compound":
            sp_c = sp
    # total exposure (billion person-months yr^-1; 1 person x 1 compound month) at fixed 2024 population
    summ["exposure_early_bn"] = summ["compound_perperson_early"] * P_tot / 1e9
    summ["exposure_late_bn"] = summ["compound_perperson_late"] * P_tot / 1e9
    # share of people living where the compound trend is MK-significant
    z = _compound_z()
    for tag, m in (("up", z > 1.959964), ("down", z < -1.959964)):
        summ[f"pop_share_MKsig_{tag}"] = float(p.where(m, 0.0).sum() / P_tot)
    # people with >= 1 more compound month per 5 years (late-early change >= 0.2)
    chg = ctx.change_field("compound", EARLY_PERIOD, LATE_PERIOD)
    summ["pop_share_change_ge_0p2"] = float(p.where(chg >= 0.2, 0.0).sum() / P_tot)
    summ["pop_share_change_ge_1"] = float(p.where(chg >= 1.0, 0.0).sum() / P_tot)
    # 2000 -> 2024 exposure decomposition, E = sum_i P_i H_i (Jones et al. 2015 style).
    # Both population surfaces come from the LandScan Mosaic Time Series, which is built
    # for change analysis; the standard LandScan Global releases are not (methods and
    # inputs change between releases), so they are NOT differenced here.
    p0 = _pop_mosaic(ctx, POP_YEAR_EARLY)
    p1 = _pop_mosaic(ctx, POP_YEAR)
    h0 = ctx.mean_annual_period("compound", *POP_HAZ_EARLY).fillna(0.0)
    h1 = ctx.mean_annual_period("compound", *POP_HAZ_LATE).fillna(0.0)
    E0, E1 = float((p0 * h0).sum()), float((p1 * h1).sum())
    clim = float((p0 * (h1 - h0)).sum()); popc = float(((p1 - p0) * h0).sum())
    inter = float(((p1 - p0) * (h1 - h0)).sum())
    # robustness: per-person trend with the Mosaic 2024 surface as the weight
    summ["compound_perperson_trend_mosaic2024"] = _ts(_pop_series(ctx.annual_counts("compound"), p1).values, yrs)
    summ.update(pop_early_snapshot=f"mosaic_ts_{POP_YEAR_EARLY}",
                pop_early_analysis_land_bn=float(p0.sum()) / 1e9,
                pop_late_mosaic_analysis_land_bn=float(p1.sum()) / 1e9,
                decomp_haz_early=f"{POP_HAZ_EARLY[0]}-{POP_HAZ_EARLY[1]}",
                decomp_haz_late=f"{POP_HAZ_LATE[0]}-{POP_HAZ_LATE[1]}",
                decomp_E0_bn=E0 / 1e9, decomp_E1_bn=E1 / 1e9, decomp_dE_bn=(E1 - E0) / 1e9,
                decomp_climate_share=clim / (E1 - E0), decomp_population_share=popc / (E1 - E0),
                decomp_interaction_share=inter / (E1 - E0),
                decomp_perperson_early=E0 / float(p0.sum()), decomp_perperson_late=E1 / float(p1.sum()))
    # South Asia (declining per-person trend) dilutes the global per-person trend and
    # holds a large share of dryland people; report the ex-SAS numbers alongside
    reg_da = regions.ar6_land_regions(ctx.slope_of("compound"))
    rtab = regions.ar6_region_table()
    inv = {ab: int(n) for ab, n in zip(rtab["abbrev"], rtab["region_index"])}
    ai = xr.DataArray(ctx.ai, coords=p.coords)
    sas = reg_da == inv["SAS"]
    cnt_c = ctx.annual_counts("compound")
    summ["pop_share_SAS"] = float(p.where(sas, 0.0).sum() / P_tot)
    summ["dryland_pop_share_in_SAS"] = float(p.where(sas & (ai < 0.65), 0.0).sum()
                                             / p.where(ai < 0.65, 0.0).sum())
    for tag, m in (("all", ~sas), ("drylands", ~sas & (ai < 0.65)), ("humid", ~sas & (ai >= 0.65))):
        summ[f"compound_perperson_trend_exSAS_{tag}"] = _ts(_pop_series(cnt_c, p.where(m, 0.0)).values, yrs)
    pd.DataFrame([summ]).round(5).to_csv(outdir / "population_summary.csv", index=False)

    # -- per aridity / density class ---------------------------------------- #
    cnt = ctx.annual_counts("compound")
    rows = []
    for grp, lab, _c, w in _pop_classes(ctx):
        s = _pop_series(cnt, w)
        lo, hi = _block_boot_ci(s.values, yrs)
        rows.append(dict(group=grp, cls=lab, pop_bn=float(w.sum()) / 1e9,
                         pop_share=float(w.sum()) / P_tot,
                         perperson_trend=_ts(s.values, yrs), ci_lo=lo, ci_hi=hi,
                         hamed_rao_p=_hr_p(s.values),
                         trend_1979_2025=_ts(s.values[post], yrs[post]),
                         early=_period(s, *EARLY_PERIOD), late=_period(s, *LATE_PERIOD),
                         pop_share_MKsig_up=float(w.where(z > 1.959964, 0.0).sum() / w.sum())))
    # difference series vs the all-population mean (is the class trend distinguishable?)
    for r, (grp, lab, _c, w) in zip(rows, _pop_classes(ctx)):
        d = _pop_series(cnt, w).values - sp_c.values
        r["diff_vs_all_trend"] = _ts(d, yrs); r["diff_vs_all_hr_p"] = _hr_p(d)
    pd.DataFrame(rows).round(5).to_csv(outdir / "population_classes.csv", index=False)

    # -- per AR6 region ------------------------------------------------------ #
    rrows = []
    for idx, ab, nm in zip(rtab["region_index"], rtab["abbrev"], rtab["name"]):
        w = p.where(reg_da == idx, 0.0)
        P_r = float(w.sum())
        if P_r < 1e6:
            continue
        s = _pop_series(cnt, w)
        e, l = _period(s, *EARLY_PERIOD), _period(s, *LATE_PERIOD)
        rrows.append(dict(abbrev=ab, name=nm, pop_mn=P_r / 1e6, pop_share=P_r / P_tot,
                          perperson_trend=_ts(s.values, yrs), hamed_rao_p=_hr_p(s.values),
                          perperson_early=e, perperson_late=l,
                          exposure_change_mn=(l - e) * P_r / 1e6))
    rdf = pd.DataFrame(rrows).sort_values("exposure_change_mn", ascending=False)
    rdf["share_of_global_exposure_change"] = rdf["exposure_change_mn"] / rdf["exposure_change_mn"].sum()
    afr = rdf["abbrev"].isin(["WAF", "CAF", "NEAF", "SEAF", "WSAF", "ESAF", "MDG", "SAH"])
    print(f"sub-Saharan+Sahel AR6 regions: pop share {rdf.loc[afr, 'pop_share'].sum():.3f}, "
          f"exposure-change share {rdf.loc[afr, 'share_of_global_exposure_change'].sum():.3f}", flush=True)
    rdf.round(5).to_csv(outdir / "population_regions.csv", index=False)
    print(pd.DataFrame([summ]).T.to_string(), flush=True)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    print(rdf.head(12).round(3).to_string(index=False), flush=True)
    return outdir


def population_exposure(ctx: _Ctx) -> Path:
    """SI figure: (a) change in compound person-exposure (late-minus-early 30-yr mean
    compound frequency x LandScan 2024 population), 1-deg aggregated; (b) per-person
    (population-weighted) vs per-area (land-area-weighted) global compound series;
    (c) per-person compound trend by aridity class and by 1-km density class with 95%
    block-bootstrap CIs; (d) AR6 regions ranked by exposure change."""
    from matplotlib.colors import SymLogNorm
    p = _pop(ctx)["pop"]
    cnt = ctx.annual_counts("compound")
    yrs = cnt["year"].values.astype(float)
    coslat = np.cos(np.deg2rad(ctx.slope_of("compound")["latitude"]))

    fig = plt.figure(figsize=(14, 9.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1.0], hspace=0.22, wspace=0.34,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a) exposure change map, 1-deg blocks -------------------------------- #
    chg = ctx.change_field("compound", EARLY_PERIOD, LATE_PERIOD).fillna(0.0)
    dE = (chg * p)
    dE1 = dE.isel(latitude=slice(0, 720)).coarsen(latitude=4, longitude=4).sum()
    land1 = ctx.land.isel(latitude=slice(0, 720)).coarsen(latitude=4, longitude=4).max()
    v = dE1.where(land1).values / 1e3            # thousand person-months yr^-1
    ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    _basemap(ax)
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    vm = float(np.nanpercentile(np.abs(v[np.isfinite(v) & (v != 0)]), 99.5))
    norm = SymLogNorm(linthresh=1.0, vmin=-vm, vmax=vm, base=10)
    m = ax.pcolormesh(dE1["longitude"].values, dE1["latitude"].values, np.where(v == 0, np.nan, v),
                      transform=ccrs.PlateCarree(), cmap=CMAP, norm=norm, zorder=1, rasterized=True)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=4)
    cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.72, pad=0.015, extend="both")
    cb.set_label("Δ total exposure (thousand person-months yr$^{-1}$ per 1° cell)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_title(f"Change in total exposure (person-months), {LATE_PERIOD[0]}–{LATE_PERIOD[1]} minus "
                 f"{EARLY_PERIOD[0]}–{EARLY_PERIOD[1]}, at fixed LandScan {POP_YEAR} population",
                 fontsize=11)
    _letter(ax, "a")

    # (b) per-person vs per-area series ------------------------------------ #
    axb = fig.add_subplot(gs[1, 0])
    for s, col, lab in [(_pop_series(cnt, p), "#5b3a8c", "per person (LandScan 2024)"),
                        (_ag_series(cnt, coslat * ctx.land), "0.45", "per unit land area")]:
        b1, a1 = theilslopes(s.values, yrs)[:2]
        axb.plot(yrs, s.values, lw=1.5, color=col, label=f"{lab}  {b1*10:+.2f}/dec")
        axb.plot(yrs, a1 + b1 * yrs, ls="--", lw=0.9, color=col, alpha=0.8)
    axb.set_xlim(1950, 2025)
    axb.set_xlabel("year", fontsize=9.5)
    axb.set_ylabel("compound event-months yr$^{-1}$", fontsize=9.5)
    axb.grid(alpha=0.25, lw=0.5)
    axb.legend(fontsize=8.3, frameon=False, loc="upper left")
    axb.set_title("Global land mean: per person vs per area", fontsize=10.5)
    _letter(axb, "b", inside=False)

    # (c) per-person trend by class ---------------------------------------- #
    axc = fig.add_subplot(gs[1, 1])
    cls = _pop_classes(ctx)
    P_tot = float(p.sum())
    ypos, labs = [], []
    y = 0.0
    prev = None
    for grp, lab, col, w in cls:
        if prev is not None and grp != prev:
            y += 0.7
        s = _pop_series(cnt, w).values
        b = _ts(s, yrs); lo, hi = _block_boot_ci(s, yrs)
        axc.barh(y, b, height=0.7, color=col, alpha=0.9)
        axc.errorbar(b, y, xerr=[[b - lo], [hi - b]], fmt="none", ecolor="0.2", capsize=3, lw=1)
        axc.text(hi + 0.01, y, f"{100*float(w.sum())/P_tot:.0f}% of people", va="center",
                 fontsize=7.8, color="0.25")
        ypos.append(y); labs.append(lab); y += 1.0; prev = grp
    axc.set_yticks(ypos); axc.set_yticklabels(labs, fontsize=8.5)
    axc.invert_yaxis()
    axc.axvline(_ts(_pop_series(cnt, p).values, yrs), color="#5b3a8c", lw=0.8, ls=":")
    axc.axvline(0, color="0.6", lw=0.6)
    axc.set_xlabel(f"per-person compound trend ({UNITS})", fontsize=9)
    axc.grid(axis="x", alpha=0.25, lw=0.5)
    axc.set_xlim(right=axc.get_xlim()[1] + 0.12)
    axc.set_title("By aridity class and 1-km density (95% CI)", fontsize=10.5)
    _letter(axc, "c", inside=False)

    # (d) AR6 regions ranked by exposure change ---------------------------- #
    axd = fig.add_subplot(gs[1, 2])
    rdf = pd.read_csv(FIGDIR / "tables" / "population_regions.csv")
    srt = rdf.sort_values("exposure_change_mn", ascending=False)
    top = pd.concat([srt.head(11), srt.tail(1)])      # 11 largest increases + largest decrease
    cols = [("#b2182b" if pv < 0.05 else "#f4a582") if ch >= 0 else ("#2166ac" if pv < 0.05 else "#92c5de")
            for pv, ch in zip(top["hamed_rao_p"], top["exposure_change_mn"])]
    axd.barh(np.arange(len(top)), top["exposure_change_mn"], color=cols, height=0.7)
    for i, (ch, sh) in enumerate(zip(top["exposure_change_mn"], top["share_of_global_exposure_change"])):
        axd.text(ch + 15 if ch >= 0 else 15, i, f"{100*sh:+.0f}%", va="center", fontsize=7.8, color="0.25")
    axd.set_yticks(np.arange(len(top))); axd.set_yticklabels(top["abbrev"], fontsize=8.5)
    axd.invert_yaxis()
    axd.set_xlabel("Δ total exposure (million person-months yr$^{-1}$)", fontsize=9)
    axd.grid(axis="x", alpha=0.25, lw=0.5)
    axd.set_xlim(right=top["exposure_change_mn"].max() * 1.18)
    from matplotlib.patches import Patch
    axd.legend(handles=[Patch(color="#b2182b", label="per-person trend $p<0.05$"),
                        Patch(color="#f4a582", label="$p\\geq0.05$ (Hamed–Rao)")],
               fontsize=7.8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.07))
    axd.axvline(0, color="0.6", lw=0.6)
    axd.set_title("  AR6 regions: Δ total exposure (share of global)", fontsize=10.5)
    _letter(axd, "d", inside=False, dx=-0.1)

    fig.suptitle("Population exposure to compound heatwave–drought months\n"
                 "HW 1961–1990 · SPEI-1 · calibration 1950–2025 · population from LandScan Global",
                 fontsize=13, y=0.995)
    out = FIGDIR / "population_exposure.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _letter(ax, s, inside=True, dx=0.0):
    if inside:
        ax.text(0.012 + dx, 0.98, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
                va="top", ha="left", zorder=5)
    else:
        ax.text(0.0 + dx, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def build_paper() -> list[Path]:
    """Curated figure/table set for manuscript_heat_spei1/ (revised 2026-09-23 after the
    pre-submission audit; see manuscript_heat_spei1/CHANGES.md).
    Main: overview (Fig 1), aridity class (Fig 2), change points (Fig 3), dryland-patch
    summary (Fig 4), agriculture (Fig 5). SI: regional compound/heat/drought ring figures,
    regional pre/post-change-point boxplots, dryland-patch montage, detrended ENSO
    composites. Onset order is reported only as a table against its base-rate null."""
    ctx = _Ctx()
    return [
        global_overview_ar6(ctx),                        # Fig 1: WHERE (mean + change)
        aridity_class_timeseries(ctx),                     # Fig 2: aridity gradient
        changepoint_map(ctx),                             # Fig 3: WHEN
        dryland_patch_summary(ctx),                        # Fig 4: named deserts
        growing_season_impact(ctx),                        # Fig 5: cropland exposure
        compound_regional(ctx),                            # SI
        heat_regional(ctx),                                # SI
        drought_regional(ctx),                             # SI
        compound_regional_boxplot(ctx, split="cp"),        # SI
        dryland_patch_regional(ctx, split="cp"),           # SI
        enso_modulation(ctx),                              # SI (trend-removed composites)
        export_result_tables(ctx),
        export_global_stats(ctx),
        export_dryland_patch_table(ctx),
        export_enso_table(ctx),
        export_agriculture_tables(ctx),
        export_onset_null_table(ctx),
        export_aridity_driver_decomposition(ctx),
        export_population_tables(ctx),
        population_exposure(ctx),                          # SI (needs population_regions.csv)
    ]


def build_all() -> list[Path]:
    """Full exploratory battery (superset of build_paper()) -- kept for reference /
    sensitivity exploration, not the curated paper figure set. See build_paper()."""
    ctx = _Ctx()
    out = [
        compound_overview(ctx),
        compound_climatology(ctx),            # absolute mean beside the trend
        absolute_change(ctx),                 # early/late/difference in physical units
        aridity_koppen_overlay(ctx),          # AI + Köppen overlaid
        desert_zooms(ctx),                    # dryland zoom group (generic engine)
        driver_attribution(ctx),
        # example zooms over the broader hotspot group -- trend and absolute mean
        zoom_figure(ctx, "compound_trend", REGION_GROUPS["hotspots"],
                    FIGDIR / "zoom_compound_trend_hotspots.png", ncols=3, ai_hatch=True,
                    title="Compound-trend detail over selected hotspots "
                          "(HW 1961–1990, SPEI-1, calibration 1950–2025)"),
        zoom_figure(ctx, "compound_mean", REGION_GROUPS["hotspots"],
                    FIGDIR / "zoom_compound_mean_hotspots.png", ncols=3,
                    title="Absolute mean annual compound frequency over selected hotspots "
                          "(1950–2025)"),
        zoom_figure(ctx, "compound_change", REGION_GROUPS["hotspots"],
                    FIGDIR / "zoom_compound_change_hotspots.png", ncols=3,
                    title="Absolute change (late − early 30-yr) in compound frequency "
                          "over selected hotspots"),
        compound_regional(ctx),               # clean-polygon regional mean + count trends
        heat_regional(ctx),
        drought_regional(ctx),
        compound_regional_boxplot(ctx, split="fixed"),  # early-vs-late boxplots per region
        compound_regional_boxplot(ctx, split="cp"),      # pre/post-change-point boxplots
        changepoint_map(ctx),                 # change-point year on a geographic AR6 map
        dryland_patch_regional(ctx, split="cp"),         # dryland patches, cp-split boxplots
        dryland_patch_regional(ctx, split="fixed"),      # dryland patches, fixed-period boxplots
        aridity_class_timeseries(ctx),        # 5 UNEP aridity classes, global lines
        export_result_tables(ctx),            # CSV tables of the numbers behind the figures
        # onset order: heat-leading vs drought-leading
        leading_overview(ctx),
        leading_response(ctx),
        leading_regional(ctx),
    ]
    return out
