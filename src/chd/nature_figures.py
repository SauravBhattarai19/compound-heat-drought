"""Nature-style figure suite for manuscript_heat_spei1 (single locked definition:
HW baseline 1961-1990, SPEI-1, SPEI calibration 1950-2025, non-Antarctic land).

Two stages, so styling can be iterated without touching the slow data disk:

  build_cache()  -- reads L5/L6/aux once through chd.manuscript_figures (the same data
                    access and estimators that produced every number in the paper) and
                    writes small fields/series to figures/manuscript/nature/cache/.
  fig_*()        -- read only the cache + figures/manuscript/tables/*.csv and draw.

Conventions (Nature): 183 mm double-column / 89 mm single-column width, sans-serif
5-7 pt text, 8 pt bold lowercase panel letters, no in-figure titles (captions carry
them), vector PDF with rasterized map meshes plus a 600-dpi PNG. Colour follows the
dataviz method: one fixed identity colour per quantity across every figure (heat,
drought, compound, P, PET, ...), blue-red diverging maps with a neutral grey midpoint,
one-hue sequential ramps for magnitude, an ordered dry-to-humid ramp for aridity.

Driver: scripts/12_nature_figures.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C  # noqa: E402

NATDIR = C.FIG_DIR / "manuscript" / "nature"
CACHE = NATDIR / "cache"
TABLES = C.FIG_DIR / "manuscript" / "tables"
PAPER_FIGDIR = C.FIG_DIR / "paper"
VARS = ("compound", "heat", "drought")
YEARS = np.arange(1950, 2026)
EARLY = (1950, 1979)
LATE = (1996, 2025)

# Worked example for the framework figure: the 2010 western-Russia heatwave and drought
# (Barriopedro et al. 2011), grid cell near Voronezh. L5 flags June-August 2010 compound.
EXAMPLE = dict(name="Voronezh, Russia", lat=51.75, lon=39.25, year=2010)


# =========================================================================== #
# Stage 1: cache
# =========================================================================== #
def _region_series(ann: np.ndarray, reg: np.ndarray, w2d: np.ndarray, n_reg: int) -> np.ndarray:
    """Area-weighted (cos-lat) regional mean of a (year, lat, lon) array for every region
    id 0..n_reg-1 at once. Equivalent to xarray `.where(mask).weighted(w).mean()` (skipna:
    NaN cells leave numerator and denominator), i.e. mf._Ctx.region_annual_mean()."""
    ok_reg = np.isfinite(reg)
    rid = np.where(ok_reg, reg, 0).astype(np.int64).ravel()
    out = np.full((ann.shape[0], n_reg), np.nan)
    wflat = w2d.ravel()
    for t in range(ann.shape[0]):
        x = ann[t].ravel()
        ok = ok_reg.ravel() & np.isfinite(x)
        num = np.bincount(rid[ok], weights=(x * wflat)[ok], minlength=n_reg)
        den = np.bincount(rid[ok], weights=wflat[ok], minlength=n_reg)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = num / den
    return out


def _example_pixel() -> dict:
    """Daily T_max vs T90(doy), daily heatwave flags, monthly SPEI-1 and L5 heat/drought/
    compound flags for the EXAMPLE cell and year, plus that cell's annual compound counts."""
    lat, lon, yr = EXAMPLE["lat"], EXAMPLE["lon"], EXAMPLE["year"]
    base = C.HW_BASELINES["1961_1990"]
    calib = C.SPEI_CALIB_PERIODS["1950_2025"]
    # daily tasmax for the example year (local NVMe cache written by L7, else L1 stores)
    cache_tx = C.TASMAX_CACHE_DIR / f"{yr}_{yr}" / "tx.npy"
    lats = np.linspace(90, -90, 721); lons = np.linspace(-180, 179.75, 1440)
    iy = int(np.argmin(np.abs(lats - lat))); ix = int(np.argmin(np.abs(lons - lon)))
    if cache_tx.exists():
        tx = np.load(cache_tx, mmap_mode="r")[:, iy, ix].astype(float)
        doy = np.load(C.TASMAX_CACHE_DIR / f"{yr}_{yr}" / "doy.npy")
    else:
        tx = np.concatenate([
            xr.open_zarr(C.daily_month_zarr(yr, m))["tasmax"].isel(latitude=iy, longitude=ix).values
            for m in range(1, 13)]).astype(float)
        doy = np.arange(1, len(tx) + 1)
    t90 = xr.open_zarr(C.HW_DIR / base.slug() / "t90.zarr")["t90"].isel(
        latitude=iy, longitude=ix).values.astype(float)
    t90d = t90[np.asarray(doy) - 1]
    hot = tx > t90d
    # >=3-day runs (within the year; the L5 monthly flag also sees runs crossing the
    # year boundary, checked against below)
    hw = np.zeros_like(hot)
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if j - i >= C.HW_MIN_DURATION:
                hw[i:j] = True
            i = j
        else:
            i += 1
    spei = xr.open_zarr(C.spei_zarr(calib))["spei1"].isel(latitude=iy, longitude=ix)
    spei = spei.sel(time=slice(f"{yr}-01", f"{yr}-12")).values.astype(float)
    l5 = xr.open_zarr(C.compound_zarr(base, C.SPEI_COMPOUND_TS, calib)).isel(latitude=iy, longitude=ix)
    l5y = l5.sel(time=slice(f"{yr}-01", f"{yr}-12")).load()
    comp_all = l5["compound"].values.astype(float)
    annual = np.nansum(comp_all.reshape(-1, 12), axis=1)
    dates = pd.date_range(f"{yr}-01-01", periods=len(tx), freq="D")
    hw_month = pd.Series(hw, index=dates).groupby(dates.month).any().reindex(range(1, 13), fill_value=False)
    return dict(
        name=EXAMPLE["name"], lat=float(lats[iy]), lon=float(lons[ix]), year=yr,
        tx=(tx - 273.15 if np.nanmean(tx) > 200 else tx).tolist(),
        t90=(t90d - 273.15 if np.nanmean(t90d) > 200 else t90d).tolist(),
        hw_day=hw.astype(int).tolist(), spei1=spei.tolist(),
        heat=l5y["heat"].values.astype(int).tolist(),
        drought=l5y["drought"].values.astype(float).tolist(),
        compound=l5y["compound"].values.astype(float).tolist(),
        hw_month_from_daily=hw_month.values.astype(int).tolist(),
        annual_compound=annual.tolist(),
    )


def build_cache(force: bool = False) -> Path:
    """Read the pipeline stores once and write every field/series the Nature figures need.
    All estimators are the pipeline's own (chd.manuscript_figures), so cached series
    reproduce the CSV tables cited in the text."""
    from scipy.stats import theilslopes  # noqa: F401
    from chd import manuscript_figures as mf, regions, enso, drylands

    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "fields.nc").exists() and not force:
        print("cache exists; pass force=True to rebuild", flush=True)
        return CACHE

    ctx = mf._Ctx()
    lat, lon = ctx.lat, ctx.lon
    land = ctx.land.values
    coords = {"latitude": lat, "longitude": lon}
    dims = ("latitude", "longitude")
    w2d = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))).astype(np.float64)
    F = {"land": land.astype("i1"), "ai": ctx.ai.astype("f4"),
         "desert": ctx.desert.astype("i1"), "arid_all": ctx.arid_all.astype("i1")}

    reg = regions.ar6_land_regions(ctx.slope_of("compound")).values
    reg = np.where(land, reg, np.nan)
    F["ar6"] = reg.astype("f4")
    ar6 = mf._ar6_geoms()
    n_reg = int(max(ar6.numbers)) + 1

    series = {}
    reg_series = {}
    aic = np.where(np.isfinite(ctx.ai), np.digitize(ctx.ai, drylands.AI_CLASS_EDGES[1:-1]), -1)
    groups = [(lab, land & (aic == i)) for i, lab in enumerate(drylands.AI_CLASS_LABELS)]
    groups += [("drylands", land & (ctx.ai < 0.65)), ("land", land)]
    for v in VARS:
        print(f"  [{v}] annual counts", flush=True)
        ann = ctx.annual_counts(v)
        a = ann.values.astype(np.float64)
        F[f"{v}_mean"] = np.nanmean(a, axis=0).astype("f4")
        e = np.nanmean(a[(YEARS >= EARLY[0]) & (YEARS <= EARLY[1])], axis=0)
        l = np.nanmean(a[(YEARS >= LATE[0]) & (YEARS <= LATE[1])], axis=0)
        F[f"{v}_early"] = e.astype("f4"); F[f"{v}_late"] = l.astype("f4")
        F[f"{v}_change"] = (l - e).astype("f4")
        tr = mf._open_trend(v)
        sig = tr["signif"].values.astype(bool)
        z = tr["z"].values
        F[f"{v}_z"] = z.astype("f4")
        F[f"{v}_sigup"] = (sig & (z > 0)).astype("i1")
        F[f"{v}_sigdn"] = (sig & (z < 0)).astype("i1")
        # area-weighted class / land series (xarray weighted mean, as in the pipeline)
        for name, m in groups:
            md = xr.DataArray(m, coords=coords, dims=dims)
            s = ann.where(md).weighted(xr.DataArray(w2d, coords=coords, dims=dims)).mean(dims)
            series[f"{name}|{v}"] = s.values
        rs = _region_series(a, reg, w2d, n_reg)
        for ab, num in zip(ar6.abbrevs, ar6.numbers):
            if ab in mf._EXCLUDE_REGIONS:
                continue
            reg_series[f"{ab}|{v}"] = rs[:, int(num)]
        if v == "compound":
            # cropland- and population-weighted series (pipeline helpers)
            crop = ctx.crop()
            for key in ("frac_total", "frac_rainfed", "frac_irrigated"):
                series[f"crop_{key}|compound"] = mf._ag_series(ann, mf._ag_weight(ctx, key)).values
                F[key] = crop[key].values.astype("f4")
            F["cropland_mask"] = crop["cropland_mask"].values.astype("i1")
            pop = mf._pop(ctx)["pop"]
            F["pop2024"] = pop.values.astype("f4")
            series["perperson|compound"] = mf._pop_series(ann, pop).values
        del a

    pd.DataFrame(series, index=YEARS).to_csv(CACHE / "series.csv")
    pd.DataFrame(reg_series, index=YEARS).to_csv(CACHE / "regional_series.csv")

    # named-desert patches (the pipeline's own builder, so letters/masks match Table S3)
    print("  dryland patches", flush=True)
    patches, _ = mf._build_dryland_patches(ctx, "compound")
    lab = np.zeros(land.shape, "i2")
    pmeta = []
    for k, p in enumerate(patches, start=1):
        lab[p["mask"]] = k
        pmeta.append(dict(id=k, letter=p["letter"], name=p["name"], clon=p["clon"],
                          clat=p["clat"], size=p["size"], cp=int(p["cp"])))
    F["patch"] = lab
    pd.DataFrame({p["letter"]: p["series"] for p in patches}, index=YEARS).to_csv(CACHE / "patch_series.csv")
    (CACHE / "patches.json").write_text(json.dumps(pmeta, indent=1))

    # ENSO composites (detrended, the published version)
    print("  ENSO composites", flush=True)
    comp = enso.phase_composite("compound", baseline_key=mf.HW, ts=mf.TS, calib_key=mf.CALIB)
    F["enso_el_nino"] = comp["el_nino_anom"].values.astype("f4")
    F["enso_la_nina"] = comp["la_nina_anom"].values.astype("f4")
    ph = enso.annual_phase().reindex(YEARS)
    land_c = pd.Series(series["land|compound"], index=YEARS)
    base = enso._running_mean_years(land_c.values.astype(float))
    pd.DataFrame({"phase": ph.values, "land": land_c.values, "runmean": base,
                  "anom": land_c.values - base}, index=YEARS).to_csv(CACHE / "enso_series.csv")

    ds = xr.Dataset({k: (dims, v) for k, v in F.items()}, coords=coords)
    enc = {k: {"zlib": True, "complevel": 4} for k in F}
    ds.to_netcdf(CACHE / "fields.nc", encoding=enc)

    print("  example pixel", flush=True)
    (CACHE / "example_pixel.json").write_text(json.dumps(_example_pixel()))
    print(f"cache written to {CACHE}", flush=True)
    return CACHE


# =========================================================================== #
# Stage 2: style, palette, primitives
# =========================================================================== #
import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import (ListedColormap, BoundaryNorm, Normalize,  # noqa: E402
                               LinearSegmentedColormap)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle  # noqa: E402
from scipy.stats import theilslopes, linregress  # noqa: E402

MM = 1 / 25.4
W2 = 183.0            # Nature double-column width, mm
W1 = 89.0             # single column

# ink & chrome (dataviz reference instance, light mode)
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#3d3d3a"
LANDBG = "#e9e8e3"     # land with no value (outside the layer being shown)
NEUTRAL = "#f2f1ee"

# one identity colour per quantity, validated with dataviz/scripts/validate_palette.js
# (--pairs all): compound/heat/drought worst CVD dE 19.8, normal-vision 24.1. Drought
# amber is < 3:1 on white, so every drought mark is direct-labelled or panel-separated.
COMPOUND = "#4a3aa7"
HEAT = "#d03b3b"
DROUGHT = "#eda100"
VAR_COL = {"compound": COMPOUND, "heat": HEAT, "drought": DROUGHT}
VAR_NAME = {"compound": "Compound", "heat": "Heatwave", "drought": "Drought"}
P_COL = "#2a78d6"       # precipitation pathway
PET_COL = "#eb6834"     # evaporative-demand pathway
RAINFED = "#1baf7a"
IRRIG = "#2a78d6"
PERSON = "#4a3aa7"
AREA = "#898781"
ELNINO = "#e34948"
LANINA = "#2a78d6"
# ordered dry -> humid ramp (OKLCH L 0.36 -> 0.74, monotone; semantic multi-hue exception)
AI_COLS = ["#5f2d0c", "#884800", "#a17003", "#979c46", "#82bc83"]
AI_LABELS = ["Hyper-arid", "Arid", "Semi-arid", "Dry sub-humid", "Humid"]
AI_EDGES = [0.0, 0.05, 0.2, 0.5, 0.65, np.inf]

# ColorBrewer RdBu (blue = decrease, red = increase), neutral grey midpoint
_RDBU = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", NEUTRAL,
         "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
DIV = LinearSegmentedColormap.from_list("rdbu_n", _RDBU, N=256)
_REDS = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
SEQ_RED = LinearSegmentedColormap.from_list("reds", _REDS, N=256)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0, "axes.titlepad": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.25, "legend.columnspacing": 0.9,
    "hatch.linewidth": 0.35, "hatch.color": INK2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "figure.dpi": 150, "mathtext.default": "regular",
    "axes.unicode_minus": True,
}
UNIT_C = "event-months yr$^{-1}$"
UNIT_CT = "event-months decade$^{-1}$"
UNIT_M = "months yr$^{-1}$"
UNIT_MT = "months decade$^{-1}$"
MAP_EXTENT = [-180, 180, -57, 84]


def _fig(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.patch.set_facecolor("white")
    return fig


def _ax(fig, x, y, w, h, **kw):
    """Axes from a mm rectangle measured from the figure's bottom-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def plabel(ax, s, dx=-7.0, dy=1.5, **kw):
    """Nature panel letter: 8 pt bold lowercase, offset in points from the axes' top-left."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=8, fontweight="bold",
                ha="left", va="bottom", color=INK, **kw)


def flabel(fig, s, x_mm, y_mm):
    W, H = fig.get_size_inches() / MM
    fig.text(x_mm / W, y_mm / H, s, fontsize=8, fontweight="bold", ha="left", va="top", color=INK)


def _ts(y, x=None):
    """Theil-Sen slope per decade and the fitted line."""
    y = np.asarray(y, float)
    x = YEARS[: len(y)].astype(float) if x is None else np.asarray(x, float)
    b, a = theilslopes(y, x)[:2]
    return b * 10, a + b * x


def _hr_p(y):
    import pymannkendall as pmk
    return float(pmk.hamed_rao_modification_test(np.asarray(y, float)).p)


def _boot_ci(y, x=None, B=5000, L=5, seed=0):
    """Pipeline's trend-preserving 5-yr residual block bootstrap (mf._block_boot_ci)."""
    y = np.asarray(y, float)
    x = YEARS[: len(y)].astype(float) if x is None else np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b, a = theilslopes(y, x)[:2]
    fit = a + b * x
    r = y - fit
    n = len(y); nb = int(np.ceil(n / L))
    out = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - L + 1, nb)
        rs = np.concatenate([r[s:s + L] for s in st])[:n]
        out[i] = theilslopes(fit + rs, x)[0]
    return tuple(np.percentile(out, [2.5, 97.5]) * 10)


def _runmean(y, w=11):
    """Centred running mean, window truncated at the record ends."""
    y = np.asarray(y, float); n = len(y); h = w // 2
    return np.array([np.nanmean(y[max(0, i - h): min(n, i + h + 1)]) for i in range(n)])


def _num(t, _=None):
    """Tick formatter: shortest repr with a true minus sign."""
    return f"{t:g}".replace("-", "\u2212")


def _style_xy(ax, grid_y=True, grid_x=False):
    ax.tick_params(which="both", direction="out")
    if grid_y:
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    if grid_x:
        ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


# ---- cache access ---------------------------------------------------------- #
_F = None


def fields() -> xr.Dataset:
    global _F
    if _F is None:
        _F = xr.open_dataset(CACHE / "fields.nc").load()
    return _F


def series() -> pd.DataFrame:
    return pd.read_csv(CACHE / "series.csv", index_col=0)


def reg_series() -> pd.DataFrame:
    return pd.read_csv(CACHE / "regional_series.csv", index_col=0)


def table(name) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


# ---- maps ----------------------------------------------------------------- #
def _cartopy():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    return ccrs, cfeature


def map_ax(fig, x, y, w, h=None, extent=MAP_EXTENT, proj="robinson"):
    ccrs, cfeature = _cartopy()
    p = ccrs.Robinson() if proj == "robinson" else ccrs.PlateCarree()
    if h is None:
        h = w * map_aspect(extent, proj)
    ax = _ax(fig, x, y, w, h, projection=p)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    ax.patch.set_visible(False)
    return ax


def map_aspect(extent=MAP_EXTENT, proj="robinson"):
    ccrs, _ = _cartopy()
    if proj != "robinson":
        return (extent[3] - extent[2]) / (extent[1] - extent[0])
    p = ccrs.Robinson()
    xs, ys = [], []
    for lo in np.linspace(extent[0], extent[1], 73):
        for la in (extent[2], extent[3], 0.0):
            X, Y = p.transform_point(lo, la, ccrs.PlateCarree())
            xs.append(X); ys.append(Y)
    return (max(ys) - min(ys)) / (max(xs) - min(xs))


def coast(ax, lw=0.25, color="#6b6a66", z=4):
    _, cfeature = _cartopy()
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=lw, edgecolor=color, zorder=z)


def land_bg(ax, color=LANDBG):
    _, cfeature = _cartopy()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=color, edgecolor="none", zorder=0)


def mesh(ax, da, cmap, norm, mask=None):
    ccrs, _ = _cartopy()
    F = fields()
    v = np.asarray(da, float)
    m = F["land"].values.astype(bool) if mask is None else mask
    v = np.where(m, v, np.nan)
    return ax.pcolormesh(F["longitude"].values, F["latitude"].values, v, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), rasterized=True, zorder=1, shading="nearest")


def stipple(ax, flag2d, step_deg=2.0, size=0.18, color=INK, alpha=0.75, min_frac=0.5, mask=None):
    """Regular-grid stippling: a dot at the centre of each step_deg block where at least
    `min_frac` of its land cells are flagged (regular dots read as texture, not data)."""
    ccrs, _ = _cartopy()
    F = fields()
    land = F["land"].values.astype(bool) if mask is None else mask
    f = np.asarray(flag2d).astype(bool) & land
    n = int(round(step_deg / 0.25))
    ny, nx = (f.shape[0] - 1) // n, f.shape[1] // n
    fl = f[: ny * n, : nx * n].reshape(ny, n, nx, n)
    ld = land[: ny * n, : nx * n].reshape(ny, n, nx, n)
    nland = ld.sum(axis=(1, 3)); nflag = fl.sum(axis=(1, 3))
    ok = (nland >= 0.5 * n * n) & (nflag >= min_frac * np.maximum(nland, 1))
    lat = F["latitude"].values[: ny * n].reshape(ny, n).mean(1)
    lon = F["longitude"].values[: nx * n].reshape(nx, n).mean(1)
    LO, LA = np.meshgrid(lon, lat)
    ax.scatter(LO[ok], LA[ok], s=size, c=color, alpha=alpha, lw=0, marker="o",
               transform=ccrs.PlateCarree(), zorder=3, rasterized=True)


def contour_mask(ax, m2d, color, lw=0.35, z=3.5):
    ccrs, _ = _cartopy()
    F = fields()
    ax.contour(F["longitude"].values, F["latitude"].values, np.asarray(m2d, float), levels=[0.5],
               colors=[color], linewidths=lw, transform=ccrs.PlateCarree(), zorder=z)


def ar6_outlines(ax, lw=0.25, color="#4a4a48", labels=False, fs=4.5, skip=()):
    from chd import manuscript_figures as mf
    ccrs, _ = _cartopy()
    clip, rep = mf._ar6_clipped()
    for ab, g in clip.items():
        if ab in mf._EXCLUDE_REGIONS:
            continue
        geoms = mf._polygonal(g)
        if geoms:
            ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none", edgecolor=color,
                              linewidth=lw, zorder=5)
        if labels and ab not in skip:
            lo, la = rep[ab]
            ax.text(lo, la, ab, transform=ccrs.PlateCarree(), fontsize=fs, ha="center", va="center",
                    color=INK, zorder=6, path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])


def cbar(fig, mappable, x, y, w, h, label, ticks=None, orient="horizontal", extend=None,
         ticklabels=None, labelpad=1.5):
    cax = _ax(fig, x, y, w, h)
    cb = fig.colorbar(mappable, cax=cax, orientation=orient, ticks=ticks,
                      extend=extend or "neither", extendfrac=0.06, drawedges=False,
                      spacing="uniform")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5, pad=1.0)
    if ticklabels is not None:
        (cb.ax.set_xticklabels if orient == "horizontal" else cb.ax.set_yticklabels)(ticklabels)
    cb.set_label(label, fontsize=6, labelpad=labelpad)
    return cb


def binned(levels, cmap=DIV, center=None, vmax=None, extend="both"):
    """Discrete colormap whose bin colours are sampled from `cmap` at the bin centres on a
    symmetric scale (equal |value| -> equal colour intensity on either arm)."""
    lv = np.asarray(levels, float)
    vmax = vmax or np.max(np.abs(lv))
    centers = 0.5 * (lv[1:] + lv[:-1])
    norm01 = (lambda v: 0.5 + 0.5 * v / vmax) if center == 0 else (lambda v: (v - lv[0]) / (lv[-1] - lv[0]))
    cols = [cmap(float(np.clip(norm01(c), 0, 1))) for c in centers]
    under = cmap(float(np.clip(norm01(lv[0] - (lv[1] - lv[0]) / 2), 0, 1)))
    over = cmap(float(np.clip(norm01(lv[-1] + (lv[-1] - lv[-2]) / 2), 0, 1)))
    cm_ = ListedColormap(cols)
    if extend in ("both", "min"):
        cm_.set_under(under)
    if extend in ("both", "max"):
        cm_.set_over(over)
    return cm_, BoundaryNorm(lv, cm_.N, extend="neither")


def save(fig, name, paper_name=None):
    NATDIR.mkdir(parents=True, exist_ok=True)
    pdf = NATDIR / f"{name}.pdf"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(NATDIR / f"{name}.png", dpi=450, facecolor="white")
    if paper_name is not None:
        PAPER_FIGDIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIGDIR / f"{paper_name}.pdf", facecolor="white")
    plt.close(fig)
    return pdf


# =========================================================================== #
# Fig. 3  The global land record: shape, rate and timing
# =========================================================================== #
SEGMENTS = [(1950, 1978), (1979, 2001), (2002, 2025)]


def _hinge_p(y):
    """p-value for the change of slope at 2002 in a continuous piecewise-linear OLS fit
    to 1979-2025 with Newey-West (5-lag) errors, as in Methods."""
    from statsmodels.regression.linear_model import OLS
    y = np.asarray(y, float)
    k = YEARS >= 1979
    t = YEARS[k] - 1979.0
    h = np.maximum(YEARS[k] - 2001.5, 0)
    r = OLS(y[k], np.column_stack([np.ones(k.sum()), t, h])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(r.pvalues[2])


def fig_global_record():
    s = series()
    gs = table("global_stats").set_index("series")
    with plt.rc_context(RC):
        fig = _fig(W2, 98)
        x0, wL = 15.5, 101.0
        hP, gap, y0 = 25.0, 3.2, 9.0
        rows = [("compound", 2), ("heat", 1), ("drought", 0)]
        axes = {}
        for v, i in rows:
            ax = _ax(fig, x0, y0 + i * (hP + gap), wL, hP)
            axes[v] = ax
            y = s[f"land|{v}"].values
            col = VAR_COL[v]
            st = gs.loc[v]
            # change point: bootstrap 5-95% range + Pettitt year
            ax.axvspan(st["pettitt_boot_p05"] - 0.5, st["pettitt_boot_p95"] + 0.5,
                       color="#d9d8d2", alpha=0.55, lw=0, zorder=0)
            ax.axvline(st["pettitt_year"], color=INK2, lw=0.5, zorder=1)
            # annual series
            ax.plot(YEARS, y, color=col, lw=0.8, zorder=3)
            ax.plot(YEARS, y, "o", ms=1.7, mfc=col, mec="white", mew=0.25, zorder=4)
            # 30-yr window means
            for (a, b) in (EARLY, LATE):
                k = (YEARS >= a) & (YEARS <= b)
                m = y[k].mean()
                ax.plot([a, b], [m, m], color=MUTED, lw=1.1, zorder=2, solid_capstyle="butt")
            # Theil-Sen by period
            for (a, b) in SEGMENTS:
                k = (YEARS >= a) & (YEARS <= b)
                _, fit = _ts(y[k], YEARS[k])
                ax.plot(YEARS[k], fit, color=INK, lw=1.2, zorder=5)
            ax.set_xlim(1949, 2026)
            ax.set_xticks(np.arange(1950, 2030, 10))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            if v != "drought":
                ax.tick_params(labelbottom=False)
            _style_xy(ax)
            lo, hi = np.nanmin(y), np.nanmax(y)
            pad = 0.10 * (hi - lo)
            ax.set_ylim(lo - pad, hi + 2.2 * pad)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.set_ylabel(f"{VAR_NAME[v]} months\n({UNIT_M})" if v != "compound"
                          else f"Compound frequency\n({UNIT_C})", fontsize=6.2)
            # Pettitt label at top of the band
            ax.text(st["pettitt_year"], 0.97, f" {int(st['pettitt_year'])}", transform=ax.get_xaxis_transform(),
                    fontsize=5.5, color=INK2, ha="left", va="top")
            # early/late mean labels (above each window's mean line)
            e, l = st["early_mean"], st["late_mean"]
            halo = [pe.withStroke(linewidth=1.6, foreground="white")]
            ax.text(1964.5, e + 0.02 * (hi - lo), f"{e:.2f}", fontsize=5.5, color=INK2, ha="center",
                    va="bottom", path_effects=halo, zorder=6)
            ax.text(2010.5, l + 0.02 * (hi - lo), f"{l:.2f}  ({(l / e - 1) * 100:+.0f}%)", fontsize=5.5,
                    color=INK2, ha="center", va="bottom", path_effects=halo, zorder=6)
            ax.text(0.006, 0.97, VAR_NAME[v], transform=ax.transAxes, fontsize=6.5, fontweight="bold",
                    color=INK, ha="left", va="top")
        axes["drought"].set_xlabel("Year")
        for v, letter in zip(("compound", "heat", "drought"), "abc"):
            plabel(axes[v], letter, dx=-30)
        axes["compound"].legend(
            handles=[Line2D([], [], color=COMPOUND, lw=0.8, marker="o", ms=1.7, mec="white", mew=0.25,
                            label="Annual land mean"),
                     Line2D([], [], color=INK, lw=1.2, label="Theil–Sen fit by period"),
                     Line2D([], [], color=MUTED, lw=1.1, label="1950–1979 / 1996–2025 mean"),
                     Patch(facecolor="#d9d8d2", alpha=0.55, label="Pettitt break, 5–95% range")],
            loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=5.4, ncol=1)

        # (d) break timing ------------------------------------------------- #
        xr0 = 131.0; wR = 47.0
        axd = _ax(fig, xr0, 66.0, wR, 26.0)
        order = ["heat", "compound", "drought"]
        for i, v in enumerate(order):
            st = gs.loc[v]
            yy = 2 - i
            axd.plot([st["pettitt_boot_p05"], st["pettitt_boot_p95"]], [yy, yy], color=VAR_COL[v],
                     lw=3.2, alpha=0.35, solid_capstyle="round", zorder=2)
            axd.plot(st["pettitt_year"], yy, "o", ms=4.2, mfc=VAR_COL[v], mec="white", mew=0.6, zorder=3)
            axd.text(st["pettitt_boot_p95"] + 0.9, yy, f"{int(st['pettitt_year'])}", va="center",
                     fontsize=5.8, color=INK)
        axd.set_yticks([2, 1, 0]); axd.set_yticklabels([VAR_NAME[v] for v in order])
        axd.set_ylim(-0.5, 2.5); axd.set_xlim(1983, 2009.6)
        axd.set_xticks([1985, 1990, 1995, 2000, 2005])
        axd.tick_params(axis="y", length=0)
        axd.spines["left"].set_visible(False)
        _style_xy(axd, grid_y=False, grid_x=True)
        axd.set_xlabel("Pettitt change-point year (5–95% bootstrap range)", fontsize=5.8)
        pr = gs.loc["compound"]
        for (ya, yb, key) in ((2, 1, "p_heat_break_before_compound"), (1, 0, "p_compound_break_before_drought")):
            xb = 2006.4
            axd.plot([xb, xb + 0.35, xb + 0.35, xb], [ya - 0.08, ya - 0.08, yb + 0.08, yb + 0.08],
                     color=INK2, lw=0.5)
            axd.text(xb + 0.6, (ya + yb) / 2, f"{pr[key] * 100:.0f}%", fontsize=5.4, color=INK2,
                     va="center", ha="left")
        plabel(axd, "d", dx=-30, dy=2)

        # (e) rate by period ----------------------------------------------- #
        axe = _ax(fig, xr0, 9.0, wR, 37.0)
        mk = [("o", "white"), ("o", "half"), ("o", "full")]
        order_e = ["compound", "heat", "drought"]
        for i, v in enumerate(order_e):
            y = s[f"land|{v}"].values
            yy = 2 - i
            rates = []
            for (a, b) in SEGMENTS:
                k = (YEARS >= a) & (YEARS <= b)
                rates.append(_ts(y[k], YEARS[k])[0])
            axe.plot(rates, [yy] * 3, color=VAR_COL[v], lw=0.8, zorder=2)
            for j, r in enumerate(rates):
                fc = "white" if j == 0 else (VAR_COL[v] if j == 2 else mpl.colors.to_hex(
                    np.array(mpl.colors.to_rgb(VAR_COL[v])) * 0.45 + 0.55))
                axe.plot(r, yy, "o", ms=4.0, mfc=fc, mec=VAR_COL[v], mew=0.8, zorder=3)
            p = _hinge_p(y)
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
            axe.text(1.13, yy + 0.27, f"{rates[1]:+.2f} → {rates[2]:+.2f}\nslope change {ptxt}",
                     fontsize=5.0, color=INK2, ha="right", va="bottom", linespacing=1.1,
                     transform=axe.get_yaxis_transform() if False else axe.transData)
        axe.axvline(0, color=AXIS, lw=0.5, zorder=1)
        axe.set_yticks([2, 1, 0]); axe.set_yticklabels([VAR_NAME[v] for v in order_e])
        axe.set_ylim(-0.5, 2.9); axe.set_xlim(-0.35, 1.15)
        axe.tick_params(axis="y", length=0)
        axe.spines["left"].set_visible(False)
        _style_xy(axe, grid_y=False, grid_x=True)
        axe.set_xlabel(f"Theil–Sen trend ({UNIT_MT})", fontsize=5.8)
        axe.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, mfc=f, mec=INK2, mew=0.8, label=l)
                            for f, l in (("white", "1950–1978"), ("#b9b8b2", "1979–2001"), (INK2, "2002–2025"))],
                   loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=5.3, handletextpad=0.15,
                   columnspacing=0.7)
        plabel(axe, "e", dx=-30, dy=8)
        return save(fig, "fig3_global_record", "fig3_global_record")


# =========================================================================== #
# Fig. 2  Where: change in compound, heatwave and drought months
# =========================================================================== #
CHANGE_LEVELS = {
    "compound": [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5],
    "heat": [-1.0, -0.5, 0.5, 1, 2, 3, 4, 5, 6],
    "drought": [-2.0, -1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5],
}
CHANGE_VMAX = {"compound": 2.75, "heat": 6.5, "drought": 2.75}


def _area_share(flag):
    F = fields()
    land = F["land"].values.astype(bool)
    w = np.cos(np.deg2rad(F["latitude"].values))[:, None] * np.ones((1, F.sizes["longitude"]))
    f = np.asarray(flag).astype(bool) & land
    return 100 * (w * f).sum() / (w * land).sum()


def _change_map(fig, v, x, y, w, stip_step, cb_w, cb_x=None, outlines=True, label_fs=6.3):
    F = fields()
    ax = map_ax(fig, x, y, w)
    h = w * map_aspect()
    land_bg(ax)
    cmap, norm = binned(CHANGE_LEVELS[v], DIV, center=0, vmax=CHANGE_VMAX[v])
    m = mesh(ax, F[f"{v}_change"].values, cmap, norm)
    ns = ~(F[f"{v}_sigup"].values.astype(bool) | F[f"{v}_sigdn"].values.astype(bool))
    stipple(ax, ns, step_deg=stip_step, size=0.16 if stip_step < 2.5 else 0.12)
    contour_mask(ax, F["desert"].values, "#5f2d0c", lw=0.35)
    if outlines:
        ar6_outlines(ax, lw=0.18, color="#7a7975")
    coast(ax)
    up = _area_share(F[f"{v}_sigup"].values); dn = _area_share(F[f"{v}_sigdn"].values)
    ax.text(0.0, 0.17, VAR_NAME[v] + (" months" if v != "compound" else ""), transform=ax.transAxes,
            fontsize=label_fs, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.0, 0.16, f"Significant increase:\n{up:.0f}% of land\nSignificant decrease:\n{dn:.1f}% of land",
            transform=ax.transAxes, fontsize=5.3, color=INK2, ha="left", va="top", linespacing=1.12)
    cb_x = x + (w - cb_w) / 2 if cb_x is None else cb_x
    unit = UNIT_C if v == "compound" else UNIT_M
    cb = cbar(fig, m, cb_x, y - 4.2, cb_w, 1.7,
              f"Change, 1996–2025 minus 1950–1979 ({unit})", ticks=CHANGE_LEVELS[v], extend="both")
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
    return ax, h


def fig_change_maps():
    F = fields()
    with plt.rc_context(RC):
        H = 128.0
        fig = _fig(W2, H)
        wa = 141.0
        ya = H - 4.0 - wa * map_aspect()
        axa, ha = _change_map(fig, "compound", 0.5, ya, wa, 2.0, 80.0, label_fs=6.8)
        plabel(axa, "a", dx=2, dy=-4)
        wb = 90.0
        yb = 12.0
        axc, _ = _change_map(fig, "heat", 0.5, yb, wb, 2.5, 72.0)
        axd, _ = _change_map(fig, "drought", 92.5, yb, wb, 2.5, 76.0)
        plabel(axc, "c", dx=2, dy=-4)
        plabel(axd, "d", dx=2, dy=-4)

        # (b) zonal-mean change over land
        axz = _ax(fig, 151.0, ya + 23.0, 29.0, ha - 25.0)
        lat = F["latitude"].values
        land = F["land"].values.astype(bool)
        for v in ("heat", "drought", "compound"):
            c = np.where(land, F[f"{v}_change"].values, np.nan)
            n = np.isfinite(c).sum(1)
            with np.errstate(invalid="ignore"):
                z = np.nanmean(c, axis=1)
            z[n < 20] = np.nan
            zs = pd.Series(z).rolling(9, center=True, min_periods=5).mean().values
            axz.plot(zs, lat, color=VAR_COL[v], lw=1.0)
        axz.axvline(0, color=AXIS, lw=0.5)
        axz.set_ylim(-56, 84)
        axz.set_yticks([-40, 0, 40, 80])
        axz.set_yticklabels(["40° S", "0°", "40° N", "80° N"])
        axz.set_xlim(-0.6, 5.4)
        axz.set_xticks([0, 2, 4])
        _style_xy(axz, grid_y=True, grid_x=False)
        axz.set_xlabel(f"Zonal-mean change\nover land ({UNIT_M})", fontsize=5.8)
        plabel(axz, "b", dx=-22, dy=0)
        # shared legend for overlays
        fig.legend(handles=[Line2D([], [], color=VAR_COL[v], lw=1.0, label=VAR_NAME[v])
                            for v in ("compound", "heat", "drought")] + [
            Line2D([], [], ls="", marker="o", ms=1.4, mfc=INK, mec="none", label="Not significant (p ≥ 0.05)"),
            Line2D([], [], color="#5f2d0c", lw=0.7, label="Köppen desert (BW)"),
            Line2D([], [], color="#7a7975", lw=0.5, label="IPCC AR6 land region")],
            loc="upper left", bbox_to_anchor=(145.0 / W2, (ya + 14.0) / H), fontsize=5.3, ncol=1,
            handlelength=1.2)
        return save(fig, "fig2_change_maps", "fig2_change_maps")


# =========================================================================== #
# Fig. 4 (and SI)  AR6 regional ring: 44 per-region panels around a choropleth
# =========================================================================== #
RING_W, RING_H = W2, W2 * 0.75
RING_YMAX = {"compound": 6.0, "heat": 11.0, "drought": 8.0}
RING_YTICKS = {"compound": [0, 3, 6], "heat": [0, 5, 10], "drought": [0, 4, 8]}
_RING_OFFSETS = {"WSAF": (-16, -5), "ESAF": (13, -8), "MDG": (10, 3), "SWS": (-11, 0),
                 "SES": (6, 2), "CNA": (0, 9), "ENA": (7, -2), "NEN": (0, 3), "NZ": (5, 0)}


def _pettitt(y):
    from scipy.stats import rankdata
    x = np.asarray(y, float)
    n = len(x)
    r = rankdata(x)
    U = np.array([2 * np.sum(r[:k]) - k * (n + 1) for k in range(1, n + 1)])
    K = np.abs(U).max()
    return int(YEARS[int(np.argmax(np.abs(U)))]), float(min(1.0, 2 * np.exp(-6 * K**2 / (n**3 + n**2))))


def _ring_map(fig, region_value, cmap, norm, hatch=None, label_fs=4.8):
    """Central AR6 choropleth (coastline-clipped polygons) for the ring figures."""
    from chd import manuscript_figures as mf
    ccrs, cfeature = _cartopy()
    clip, rep = mf._ar6_clipped()
    wmap = 92.0
    x0 = (RING_W - wmap) / 2
    ax = map_ax(fig, x0, 58.0, wmap)
    land_bg(ax, "#f4f3f0")
    for ab, g in clip.items():
        if ab in mf._EXCLUDE_REGIONS:
            continue
        val = region_value.get(ab, np.nan)
        fc = cmap(norm(val)) if np.isfinite(val) else LANDBG
        geoms = mf._polygonal(g)
        if not geoms:
            continue
        ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor=fc, edgecolor="white",
                          linewidth=0.35, zorder=1)
        if hatch is not None and hatch.get(ab, False):
            ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none", edgecolor=INK2,
                              linewidth=0, hatch="//////", zorder=1.5)
    coast(ax, lw=0.25, color="#8a8984", z=2)
    tr = ccrs.PlateCarree()._as_mpl_transform(ax)
    for ab, (lo, la) in rep.items():
        if ab in mf._EXCLUDE_REGIONS:
            continue
        off = _RING_OFFSETS.get(ab, (0, 0))
        ax.annotate(ab, xy=(lo, la), xycoords=tr, xytext=off, textcoords="offset points",
                    fontsize=label_fs, fontweight="bold", ha="center", va="center", color=INK, zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.3, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.35, shrinkA=0, shrinkB=0)
                    if off != (0, 0) else None)
    return ax, x0, wmap


def _ring_panel_axes(fig):
    from chd import manuscript_figures as mf
    out = {}
    for ab, (fx, fy, fw, fh) in mf.SLOTS.items():
        out[ab] = fig.add_axes([fx, fy + 0.006, fw, fh * 0.92])
    return out


def _ring_ts_panel(ax, y, var, ymax, yticks, cp=None, trend=None, sig=True, title=None,
                   show_x=False, show_y=True, big=False):
    col = VAR_COL[var]
    if cp is not None:
        ax.axvspan(cp + 0.5, 2025.5, color="#e7e6e1", lw=0, zorder=0)
    ax.plot(YEARS, np.clip(y, 0, ymax), "o", ms=1.9 if big else 1.15, mfc=col, mec="none",
            alpha=0.75, zorder=2, clip_on=False)
    _, fit = _ts(y)
    ax.plot(YEARS, fit, color=INK, lw=1.1 if big else 0.8, zorder=3)
    ax.set_xlim(1948, 2027); ax.set_ylim(0, ymax)
    ax.set_xticks([1950, 2025]); ax.set_yticks(yticks)
    ax.tick_params(length=1.4, width=0.4, pad=0.8, labelsize=5.6 if big else 4.6)
    ax.set_yticklabels([str(t) if i in (0, len(yticks) - 1) else "" for i, t in enumerate(yticks)])
    if not show_x:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(["1950", "2025"])
        for lab, ha in zip(ax.get_xticklabels(), ("left", "right")):
            lab.set_ha(ha)
    if not show_y:
        ax.set_yticklabels([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.4)
    if title is not None:
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=6.2 if big else 5.4,
                fontweight="bold", ha="left", va="bottom", color=INK)
    if trend is not None:
        ax.text(1.0, 1.02, f"{trend:+.2f}".replace("-", "−"), transform=ax.transAxes,
                fontsize=6.0 if big else 5.0, fontweight="bold" if sig else "normal",
                color=INK if sig else MUTED, ha="right", va="bottom")


def fig_ring(var="compound", name=None, paper_name=None):
    rs = table("regional_stats").set_index("abbrev")
    R = reg_series()
    from chd import manuscript_figures as mf
    ymax, yt = RING_YMAX[var], RING_YTICKS[var]
    trend = {ab: _ts(R[f"{ab}|{var}"].values)[0] for ab in mf.SLOTS}
    hr = {ab: rs.loc[ab, f"{var}_hamed_rao_p"] for ab in mf.SLOTS}
    cps = {ab: _pettitt(R[f"{ab}|{var}"].values) for ab in mf.SLOTS}
    with plt.rc_context(RC):
        fig = _fig(RING_W, RING_H)
        if var == "drought":
            lv = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
            cmap, norm = binned(lv, DIV, center=0, vmax=0.55, extend="neither")
        elif var == "heat":
            lv = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
            cmap, norm = binned(lv, SEQ_RED, extend="neither")
        else:
            lv = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            cmap, norm = binned(lv, SEQ_RED, extend="neither")
        hatch = {ab: hr[ab] >= 0.05 for ab in mf.SLOTS}
        axm, x0, wmap = _ring_map(fig, trend, cmap, norm, hatch=hatch)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        unit = UNIT_CT if var == "compound" else UNIT_MT
        cb = cbar(fig, sm, x0 + 18, 52.5, wmap - 36, 1.6,
                  f"{VAR_NAME[var]} trend, Theil–Sen ({unit})", ticks=lv)
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        # key panel: global land mean, annotated
        S = series()
        g = S[f"land|{var}"].values
        gcp, gcpp = _pettitt(g)
        kax = _ax(fig, x0 + 11, 19.0, 32.0, 19.0)
        gsig = _hr_p(g) < 0.05
        _ring_ts_panel(kax, g, var, ymax, yt, cp=gcp if gcpp < 0.05 else None,
                       trend=_ts(g)[0], sig=gsig, title="Global land (key)", show_x=True, big=True)
        kax.set_ylabel(f"{VAR_NAME[var]}\n({UNIT_C if var == 'compound' else UNIT_M})", fontsize=5.4)
        kx = x0 + 48
        items = [("o", VAR_COL[var], "Annual mean (area-weighted)"),
                 ("-", INK, "Theil–Sen fit, 1950–2025"),
                 ("s", "#e7e6e1", "After Pettitt change point (p < 0.05)"),
                 ("t", INK, "Trend per decade (bold: Hamed–Rao p < 0.05)"),
                 ("h", INK2, "Map hatching: trend not significant")]
        for i, (m, c, t) in enumerate(items):
            yy = 36.0 - i * 3.7
            kk = _ax(fig, kx, yy - 1.0, 3.2, 2.0)
            kk.set_axis_off(); kk.set_xlim(0, 1); kk.set_ylim(0, 1)
            if m == "o":
                kk.plot([0.25, 0.5, 0.75], [0.4, 0.65, 0.5], "o", ms=1.9, mfc=c, mec="none", alpha=0.75)
            elif m == "-":
                kk.plot([0, 1], [0.3, 0.7], color=c, lw=1.1)
            elif m == "s":
                kk.add_patch(Rectangle((0, 0.05), 1, 0.9, color=c, lw=0))
            elif m == "t":
                kk.text(0.5, 0.5, f"{_ts(g)[0]:+.2f}", fontsize=5.4, fontweight="bold" if gsig else "normal",
                        color=INK if gsig else MUTED, ha="center", va="center")
            elif m == "h":
                kk.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=cmap(0.5), edgecolor=INK2,
                                       hatch="//////", lw=0))
            fig.text((kx + 4.6) / RING_W, yy / RING_H, t, fontsize=5.4, va="center", ha="left", color=INK)
        # 44 panels
        axes = _ring_panel_axes(fig)
        bottom_row = {"WAF", "SAH", "WSAF", "CAF", "ESAF", "SEAF", "NEAF", "MDG"}
        for ab, ax in axes.items():
            y = R[f"{ab}|{var}"].values
            cp, cpp = cps[ab]
            _ring_ts_panel(ax, y, var, ymax, yt, cp=cp if cpp < 0.05 else None, trend=trend[ab],
                           sig=hr[ab] < 0.05, title=ab, show_x=ab in bottom_row)
        name = name or f"fig4_ring_{var}"
        return save(fig, name, paper_name or name)


# =========================================================================== #
# Fig. 5  Why drylands: aridity gradient, marginals and the P/PET counterfactual
# =========================================================================== #
CLASSES = ["hyper-arid", "arid", "semi-arid", "dry sub-humid", "humid"]
CLASS_TICK = ["Hyper-\narid", "Arid", "Semi-\narid", "Dry sub-\nhumid", "Humid"]


def class_trend_table() -> pd.DataFrame:
    """Theil-Sen trend + 95% block-bootstrap CI (pipeline convention) for every aridity
    class and variable; memoised in the cache."""
    path = CACHE / "class_trend_ci.csv"
    if path.exists():
        return pd.read_csv(path)
    S = series()
    rows = []
    for c in CLASSES + ["drylands", "land"]:
        for v in VARS:
            y = S[f"{c}|{v}"].values
            lo, hi = _boot_ci(y)
            rows.append(dict(cls=c, var=v, trend=_ts(y)[0], lo=lo, hi=hi, hr_p=_hr_p(y)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _ai_class_of(ai):
    return int(np.digitize(ai, AI_EDGES[1:-1]))


def fig_aridity():
    F = fields()
    S = series()
    ct = class_trend_table().set_index(["cls", "var"])
    dec = table("aridity_driver_decomposition").set_index("class")
    acs = table("aridity_class_stats").set_index("class")
    rs = table("regional_stats").set_index("abbrev")
    att = table("regional_attribution_stats").iloc[0]
    with plt.rc_context(RC):
        H = 158.0
        fig = _fig(W2, H)
        # (a) aridity class map ------------------------------------------- #
        wa = 96.0
        axa = map_ax(fig, 0.0, H - 4.5 - wa * map_aspect(), wa)
        cmap = ListedColormap(AI_COLS)
        norm = BoundaryNorm(AI_EDGES[:-1] + [10.0], cmap.N)
        mesh(axa, np.where(np.isfinite(F["ai"].values), np.minimum(F["ai"].values, 9.0), np.nan), cmap, norm)
        contour_mask(axa, F["desert"].values, INK, lw=0.3)
        coast(axa, lw=0.2)
        axa.legend(handles=[Patch(facecolor=AI_COLS[i], edgecolor="none",
                                  label=f"{AI_LABELS[i]}  {acs.loc[CLASSES[i], 'land_area_share'] * 100:.0f}%")
                            for i in range(5)] + [Line2D([], [], color=INK, lw=0.6, label="Köppen desert (BW)")],
                   loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.3, title="UNEP class, share of land",
                   title_fontsize=5.3, handlelength=1.1, handleheight=0.8, alignment="left")
        plabel(axa, "a", dx=2, dy=-4)

        # (b) class-mean series (11-yr running mean) ----------------------- #
        axb = _ax(fig, 113.0, H - 46.0, 67.0, 39.0)
        for i, c in enumerate(CLASSES):
            y = S[f"{c}|compound"].values
            axb.plot(YEARS, _runmean(y, 11), color=AI_COLS[i], lw=1.3 if i in (0, 4) else 0.9,
                     label=f"{AI_LABELS[i]}  {ct.loc[(c, 'compound'), 'trend']:+.2f}", zorder=3 if i in (0, 4) else 2)
        axb.set_xlim(1950, 2025); axb.set_xticks([1950, 1970, 1990, 2010])
        axb.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        axb.set_ylabel(f"Compound frequency\n({UNIT_C}, 11-yr mean)")
        axb.set_xlabel("Year")
        _style_xy(axb)
        axb.legend(loc="upper left", fontsize=5.3, title=f"Trend ({UNIT_CT})", title_fontsize=5.3,
                   alignment="left", handlelength=1.2)
        plabel(axb, "b", dx=-25)

        # (c-e) trend by class: compound, heatwave, drought ---------------- #
        yrow = H - 98.0
        hp = 36.0
        xs = [14.0, 76.0, 138.0]
        ylims = {"compound": (0, 0.52), "heat": (0, 1.12), "drought": (-0.06, 0.62)}
        for k, (v, letter) in enumerate(zip(("compound", "heat", "drought"), "cde")):
            ax = _ax(fig, xs[k], yrow, 43.0, hp)
            for i, c in enumerate(CLASSES):
                r = ct.loc[(c, v)]
                sig = r["hr_p"] < 0.05
                ax.plot([i, i], [r["lo"], r["hi"]], color=AI_COLS[i], lw=0.9, zorder=2)
                ax.plot(i, r["trend"], "o", ms=4.2, mfc=AI_COLS[i] if sig else "white", mec=AI_COLS[i],
                        mew=0.9, zorder=3)
                if v != "heat":
                    d = dec.loc[c]
                    ax.plot(i + 0.28, d[f"{v}_trend_PET_only"], marker="^", ms=3.3, mfc=PET_COL, mec="white",
                            mew=0.3, ls="", zorder=4)
                    ax.plot(i + 0.28, d[f"{v}_trend_P_only"], marker="v", ms=3.3, mfc=P_COL, mec="white",
                            mew=0.3, ls="", zorder=4)
            ratio = ct.loc[("hyper-arid", v), "trend"] / ct.loc[("humid", v), "trend"]
            ax.text(0.98, 0.97, f"hyper-arid / humid\n= {ratio:.1f}×", transform=ax.transAxes, fontsize=5.6,
                    ha="right", va="top", color=INK, linespacing=1.1)
            ax.axhline(0, color=AXIS, lw=0.5, zorder=1)
            ax.set_xlim(-0.5, 4.7); ax.set_ylim(*ylims[v])
            ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_TICK, fontsize=5.2)
            ax.tick_params(axis="x", length=0)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
            _style_xy(ax)
            ax.set_ylabel(f"{VAR_NAME[v]} trend\n({UNIT_CT if v == 'compound' else UNIT_MT})")
            ax.text(0.02, 0.97, VAR_NAME[v] + (" months" if v != "compound" else ""), transform=ax.transAxes,
                    fontsize=6.3, fontweight="bold", ha="left", va="top")
            plabel(ax, letter, dx=-27)
        fig.legend(handles=[
            Line2D([], [], ls="-", marker="o", ms=4, color=INK2, mfc=INK2, mec=INK2, lw=0.8,
                   label="Actual trend, 95% CI (filled: Hamed–Rao p < 0.05)"),
            Line2D([], [], ls="", marker="^", ms=3.5, mfc=PET_COL, mec="white", label="Counterfactual: only PET varies"),
            Line2D([], [], ls="", marker="v", ms=3.5, mfc=P_COL, mec="white", label="Counterfactual: only P varies")],
            loc="lower center", bbox_to_anchor=(0.5, (yrow + hp + 4.2) / H), ncol=3, fontsize=5.4)

        # (f, g) regional attribution scatter ------------------------------ #
        y3 = 12.0; h3 = 37.0
        reg = rs.copy()
        reg["cls"] = [_ai_class_of(a) for a in reg["area_weighted_log_mean_ai"]]
        offs = {"drought": {"SAH": (3, -1), "ARP": (-3, 3), "SAS": (3, -1), "NES": (3, -1), "SEAF": (3, 1)},
                "heat": {"SAH": (3, -1), "ARP": (3, -3), "SAS": (3, -1), "NES": (3, -1), "SEAF": (-3, 2.5)}}
        for k, (v, letter, x) in enumerate((("drought", "f", 14.0), ("heat", "g", 69.0))):
            ax = _ax(fig, x, y3, 43.0, h3)
            xv = reg[f"{v}_trend_per_decade"].values; yv = reg["compound_trend_per_decade"].values
            ax.scatter(xv, yv, s=9, c=[AI_COLS[c] for c in reg["cls"]], edgecolors="white", linewidths=0.35, zorder=3)
            lr = linregress(xv, yv)
            xx = np.linspace(xv.min(), xv.max(), 10)
            ax.plot(xx, lr.intercept + lr.slope * xx, color=VAR_COL[v], lw=0.9, zorder=2)
            rr = att["r_compound_drought"] if v == "drought" else att["r_compound_heat"]
            ax.text(0.03, 0.97, f"r = {rr:.2f}", transform=ax.transAxes, fontsize=6, ha="left", va="top")
            for ab, o in offs[v].items():
                ax.annotate(ab, (reg.loc[ab, f"{v}_trend_per_decade"], reg.loc[ab, "compound_trend_per_decade"]),
                            xytext=o, textcoords="offset points", fontsize=4.8, color=INK2,
                            ha="left" if o[0] > 0 else "right", va="center")
            ax.set_xlabel(f"{VAR_NAME[v]}-month trend ({UNIT_MT})")
            if k == 0:
                ax.set_ylabel(f"Regional compound trend\n({UNIT_CT})")
                ax.axvline(0, color=AXIS, lw=0.5, zorder=1)
            else:
                ax.set_yticklabels([])
                ax.text(0.97, 0.03, f"Steiger Z = {att['steiger_z']:.2f}, p < 0.001\n(r$_{{drought}}$ vs r$_{{heat}}$)",
                        transform=ax.transAxes, fontsize=5.2, ha="right", va="bottom", color=INK2, linespacing=1.15)
            ax.set_ylim(-0.02, 0.42)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
            _style_xy(ax, grid_y=True, grid_x=True)
            plabel(ax, letter, dx=-27 if k == 0 else -8)

        # (h) heat-drought coupling by class -------------------------------- #
        ax = _ax(fig, 138.0, y3, 43.0, h3)
        for i, c in enumerate(CLASSES):
            d = dec.loc[c]
            yy = 4 - i
            a, b = d["p_drought_given_noheat"], d["p_drought_given_heat"]
            ax.plot([a, b], [yy, yy], color=AI_COLS[i], lw=1.0, zorder=2)
            ax.plot(a, yy, "o", ms=3.8, mfc="white", mec=AI_COLS[i], mew=0.9, zorder=3)
            ax.plot(b, yy, "o", ms=3.8, mfc=AI_COLS[i], mec=AI_COLS[i], mew=0.9, zorder=3)
            ax.text(0.375, yy, f"{b / a:.1f}×", fontsize=5.6, va="center", ha="right", color=INK)
        ax.set_yticks(range(5)); ax.set_yticklabels(list(reversed([t.replace("\n", "") for t in CLASS_TICK])), fontsize=5.4)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0.0, 0.38); ax.set_ylim(-0.6, 4.9)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        _style_xy(ax, grid_y=False, grid_x=True)
        ax.set_xlabel("Probability that a month is a drought month")
        ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.8, mfc="white", mec=INK2, label="No heatwave"),
                           Line2D([], [], ls="", marker="o", ms=3.8, mfc=INK2, mec=INK2, label="Heatwave month")],
                  loc="lower center", bbox_to_anchor=(0.45, 0.98), ncol=2, fontsize=5.3)
        plabel(ax, "h", dx=-27)
        return save(fig, "fig5_aridity_mechanism", "fig5_aridity_mechanism")


# =========================================================================== #
# Fig. 6  The world's named deserts
# =========================================================================== #
def fig_deserts():
    ccrs, cfeature = _cartopy()
    F = fields()
    ps = table("dryland_patch_stats").set_index("letter")
    rs = table("regional_stats").set_index("abbrev")
    meta = {m["letter"]: m for m in json.loads((CACHE / "patches.json").read_text())}
    order = ps.sort_values("trend_per_decade").index.tolist()
    lv = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    cmap, norm = binned(lv, DIV, center=0, vmax=0.55, extend="neither")
    with plt.rc_context(RC):
        H = 80.0
        fig = _fig(W2, H)
        ext = [-125, 150, -56, 55]
        wa = 110.0
        ha = wa * map_aspect(ext, "plate")
        axa = map_ax(fig, 0.0, H - 3.0 - ha, wa, extent=ext, proj="plate")
        axa.set_aspect("auto")
        land_bg(axa, "#ecebe6")
        lab = F["patch"].values
        val = np.full(lab.shape, np.nan)
        for k, m in enumerate(meta.values(), start=1):
            val[lab == m["id"]] = ps.loc[m["letter"], "trend_per_decade"]
        axa.pcolormesh(F["longitude"].values, F["latitude"].values, val, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), rasterized=True, zorder=1, shading="nearest")
        coast(axa, lw=0.25, color="#8a8984")
        for L, m in meta.items():
            axa.text(m["clon"], m["clat"], L, transform=ccrs.PlateCarree(), fontsize=5.6,
                     ha="center", va="center", color=INK, zorder=5,
                     bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=INK2, lw=0.4))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = cbar(fig, sm, 25.0, H - 3.0 - ha - 6.0, 60.0, 1.6,
                  f"Patch compound trend, Theil–Sen ({UNIT_CT})", ticks=lv)
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        plabel(axa, "a", dx=1, dy=-1)

        # (b) ranked trends with CI and host region -------------------------- #
        yb, hb = 10.0, H - 14.0
        axb = _ax(fig, 138.0, yb, 22.0, hb)
        axc = _ax(fig, 163.0, yb, 18.0, hb)
        for i, L in enumerate(order):
            r = ps.loc[L]
            sig = r["hamed_rao_p"] < 0.05
            col = cmap(norm(r["trend_per_decade"]))
            axb.plot([r["trend_ci_lo"], r["trend_ci_hi"]], [i, i], color=INK2, lw=0.8, zorder=2)
            axb.plot(rs.loc[r["ar6_abbrev"], "compound_trend_per_decade"], i, marker="|", ms=5.0, mew=1.0,
                     color=MUTED, zorder=2)
            axb.plot(r["trend_per_decade"], i, "o", ms=4.0, mfc=col if sig else "white", mec=INK, mew=0.5, zorder=3)
            # (c) early -> late 30-yr means
            axc.plot([r["early_mean"], r["late_mean"]], [i, i], color=MUTED, lw=0.9, zorder=2)
            axc.plot(r["early_mean"], i, "o", ms=3.4, mfc="white", mec=INK2, mew=0.7, zorder=3)
            axc.plot(r["late_mean"], i, "o", ms=3.4, mfc=INK2, mec=INK2, mew=0.7, zorder=3)
        axb.set_yticks(range(len(order)))
        axb.set_yticklabels([f"{L}  {ps.loc[L, 'name']} ({ps.loc[L, 'ar6_abbrev']})" for L in order], fontsize=5.6)
        axb.axvline(0, color=AXIS, lw=0.5, zorder=1)
        axb.set_xlim(-0.32, 0.56); axb.set_xticks([-0.2, 0, 0.2, 0.4])
        axb.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        axb.set_ylim(-0.7, len(order) - 0.3)
        axb.tick_params(axis="y", length=0)
        _style_xy(axb, grid_y=False, grid_x=True)
        axb.set_xlabel(f"Trend\n({UNIT_CT})", fontsize=5.8)
        axc.set_yticks(range(len(order))); axc.set_yticklabels([])
        axc.tick_params(axis="y", length=0)
        axc.set_ylim(-0.7, len(order) - 0.3); axc.set_xlim(0, 2.8); axc.set_xticks([0, 1, 2])
        _style_xy(axc, grid_y=False, grid_x=True)
        axc.set_xlabel(f"30-yr mean\n({UNIT_C})", fontsize=5.8)
        plabel(axb, "b", dx=-60, dy=1)
        plabel(axc, "c", dx=-4, dy=1)
        fig.legend(handles=[
            Line2D([], [], ls="", marker="o", ms=4, mfc=cmap(norm(0.35)), mec=INK, mew=0.5, label="Patch trend, Hamed–Rao p < 0.05"),
            Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK, mew=0.5, label="Patch trend, not significant"),
            Line2D([], [], color=INK2, lw=0.8, label="95% block-bootstrap CI"),
            Line2D([], [], ls="", marker="|", ms=5, mew=1.0, color=MUTED, label="Host AR6 region trend"),
            Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=INK2, label="1950–1979 mean (c)"),
            Line2D([], [], ls="", marker="o", ms=3.4, mfc=INK2, mec=INK2, label="1996–2025 mean (c)")],
            loc="lower left", bbox_to_anchor=(2.0 / W2, 0.5 / H), ncol=3, fontsize=5.4)
        return save(fig, "fig6_deserts", "fig6_deserts")


# =========================================================================== #
# Fig. 7  Exposure: cropland and people
# =========================================================================== #
def _symlog_binned():
    lv = [-1e4, -1e3, -1e2, -10, -1, 1, 10, 1e2, 1e3, 1e4]
    pos = [0.08, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.92]       # bin colours, symmetric arms
    cm_ = ListedColormap([DIV(p) for p in pos])
    cm_.set_under(DIV(0.0)); cm_.set_over(DIV(1.0))
    return lv, cm_, BoundaryNorm(lv, cm_.N)


def fig_exposure():
    F = fields()
    S = series()
    ag = table("agriculture_exposure").set_index("exposure_class")
    ags = table("agriculture_summary").iloc[0]
    pc = table("population_classes").set_index("cls")
    psum = table("population_summary").iloc[0]
    preg = table("population_regions")
    gs = table("global_stats").set_index("series")
    land = F["land"].values.astype(bool)
    with plt.rc_context(RC):
        H = 152.0
        fig = _fig(W2, H)
        # (a) compound change over cropland ------------------------------- #
        wm = 90.0
        hm = wm * map_aspect()
        ym = H - 3.0 - hm
        crop = F["cropland_mask"].values.astype(bool) & land
        axa = map_ax(fig, 0.0, ym, wm)
        land_bg(axa, "#ecebe6")
        cmap, norm = binned(CHANGE_LEVELS["compound"], DIV, center=0, vmax=CHANGE_VMAX["compound"])
        m = mesh(axa, F["compound_change"].values, cmap, norm, mask=crop)
        ns = ~(F["compound_sigup"].values.astype(bool) | F["compound_sigdn"].values.astype(bool))
        stipple(axa, ns, step_deg=2.5, size=0.12, mask=crop)
        coast(axa, lw=0.2)
        cb = cbar(fig, m, 14.0, ym - 4.0, 62.0, 1.6, f"Compound change over cropland ({UNIT_C})",
                  ticks=CHANGE_LEVELS["compound"], extend="both")
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        axa.text(0.0, 0.03, f"Cropland (MIRCA2000)\nsignificant increase on\n"
                            f"{ags['cropland_pct_MKsig_up_z']:.0f}% of cropland area",
                 transform=axa.transAxes, fontsize=5.3, color=INK2, ha="left", va="bottom", linespacing=1.12)
        plabel(axa, "a", dx=2, dy=-4)
        # (b) change in population exposure ------------------------------ #
        axb = map_ax(fig, 93.0, ym, wm)
        land_bg(axb, "#ecebe6")
        chg = np.nan_to_num(F["compound_change"].values)
        pop = np.nan_to_num(F["pop2024"].values)
        dE = (chg * pop)[:720].reshape(180, 4, 360, 4).sum((1, 3)) / 1e3
        l1 = land[:720].reshape(180, 4, 360, 4).max((1, 3))
        dE = np.where(l1 & (dE != 0), dE, np.nan)
        lat1 = F["latitude"].values[:720].reshape(180, 4).mean(1)
        lon1 = F["longitude"].values.reshape(360, 4).mean(1)
        lv, cm2, nm2 = _symlog_binned()
        mb = axb.pcolormesh(lon1, lat1, dE, cmap=cm2, norm=nm2, transform=_cartopy()[0].PlateCarree(),
                            rasterized=True, zorder=1, shading="nearest")
        coast(axb, lw=0.2)
        lab = {1e4: "10$^4$", 1e3: "10$^3$", 1e2: "10$^2$", 10: "10", 1: "1"}
        cb2 = cbar(fig, mb, 107.0, ym - 4.0, 62.0, 1.6,
                   "Change in exposure (10$^3$ person-months yr$^{-1}$ per 1° cell)", ticks=lv, extend="both",
                   ticklabels=[("−" + lab[-t]) if t < 0 else lab[t] for t in lv])
        axb.text(0.0, 0.03, f"1996–2025 minus 1950–1979,\nLandScan 2024 population",
                 transform=axb.transAxes, fontsize=5.3, color=INK2, ha="left", va="bottom", linespacing=1.12)
        plabel(axb, "b", dx=2, dy=-4)

        # (c) trend by exposure weighting ---------------------------------- #
        rows = [
            ("Per area: all land", gs.loc["compound", "theilsen_per_decade"], gs.loc["compound", "ci_lo"],
             gs.loc["compound", "ci_hi"], gs.loc["compound", "hamed_rao_p"], gs.loc["compound", "theilsen_1979_2025"], AREA),
            ("Cropland: all", *ag.loc["All cropland", ["trend_per_decade", "ci_lo", "ci_hi", "hamed_rao_p"]],
             _ts(S["crop_frac_total|compound"].values[29:], YEARS[29:])[0], "#8a6d3b"),
            ("Cropland: rainfed", *ag.loc["Rainfed cropland", ["trend_per_decade", "ci_lo", "ci_hi", "hamed_rao_p"]],
             ags["rainfed_trend_post1979"], RAINFED),
            ("Cropland: irrigated", *ag.loc["Irrigated cropland", ["trend_per_decade", "ci_lo", "ci_hi", "hamed_rao_p"]],
             ags["irrig_trend_post1979"], IRRIG),
            ("Per person: everyone", psum["compound_perperson_trend"], psum["compound_perperson_ci_lo"],
             psum["compound_perperson_ci_hi"], psum["compound_perperson_hr_p"],
             psum["compound_perperson_trend_1979_2025"], PERSON),
        ]
        for key, lab_ in (("drylands (AI<0.65)", "Per person: drylands"), ("humid", "Per person: humid land"),
                          ("urban (≥1,500 km$^{-2}$)", "Per person: urban"),
                          ("rural (<300 km$^{-2}$)", "Per person: rural")):
            r = pc.loc[key]
            rows.append((lab_, r["perperson_trend"], r["ci_lo"], r["ci_hi"], r["hamed_rao_p"],
                         r["trend_1979_2025"], PERSON))
        y2, h2 = 50.0, 44.0
        axc = _ax(fig, 31.0, y2, 40.0, h2)
        n = len(rows)
        for i, (lab_, t, lo, hi, p, t79, col) in enumerate(rows):
            yy = n - 1 - i
            axc.plot([lo, hi], [yy, yy], color=col, lw=0.9, zorder=2)
            axc.plot(t, yy, "o", ms=3.8, mfc=col if p < 0.05 else "white", mec=col, mew=0.8, zorder=3)
            axc.plot(t79, yy, "D", ms=2.8, mfc="white", mec=INK2, mew=0.6, zorder=3)
        axc.set_yticks(range(n)); axc.set_yticklabels([r[0] for r in reversed(rows)], fontsize=5.5)
        axc.tick_params(axis="y", length=0)
        for yy in (n - 1.5, n - 4.5):
            axc.axhline(yy, color=GRID, lw=0.5)
        axc.set_xlim(0, 0.42); axc.set_ylim(-0.6, n - 0.4)
        axc.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        _style_xy(axc, grid_y=False, grid_x=True)
        axc.set_xlabel(f"Compound trend ({UNIT_CT})")
        axc.legend(handles=[Line2D([], [], ls="-", marker="o", ms=3.6, color=INK2, mfc=INK2, lw=0.8,
                                   label="1950–2025, 95% CI"),
                            Line2D([], [], ls="", marker="D", ms=2.8, mfc="white", mec=INK2, label="1979–2025")],
                   loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=5.3)
        plabel(axc, "c", dx=-85, dy=1)

        # (d) exposure decomposition ~2000 -> ~2024 (waterfall) ---------------- #
        axd = _ax(fig, 86.0, y2, 36.0, h2)
        E0, E1 = psum["decomp_E0_bn"], psum["decomp_E1_bn"]
        dEt = psum["decomp_dE_bn"]
        parts = [("Hazard", psum["decomp_climate_share"] * dEt, "#b2182b"),
                 ("Population", psum["decomp_population_share"] * dEt, "#8a8984"),
                 ("Both", psum["decomp_interaction_share"] * dEt, "#d6604d")]
        xs_ = np.arange(5)
        axd.bar(0, E0, width=0.62, color="#5b5a56", zorder=2)
        base = E0
        for j, (lab_, v, col) in enumerate(parts, start=1):
            axd.bar(j, v, bottom=base, width=0.62, color=col, zorder=2)
            axd.text(j, base + v + 0.25, f"+{v:.1f}\n({v / dEt * 100:.0f}%)", ha="center", va="bottom",
                     fontsize=5.0, color=INK, linespacing=1.05)
            base += v
        axd.bar(4, E1, width=0.62, color="#5b5a56", zorder=2)
        axd.text(0, E0 + 0.25, f"{E0:.1f}", ha="center", va="bottom", fontsize=5.2)
        axd.text(4, E1 + 0.25, f"{E1:.1f}", ha="center", va="bottom", fontsize=5.2)
        axd.set_xticks(xs_); axd.set_xticklabels(["~2000", "Hazard", "Popu-\nlation", "Both", "~2024"], fontsize=5.3)
        axd.tick_params(axis="x", length=0)
        axd.set_ylim(0, 16.5); axd.set_xlim(-0.6, 4.6)
        _style_xy(axd)
        axd.set_ylabel("Total exposure\n(10$^9$ person-months yr$^{-1}$)")
        plabel(axd, "d", dx=-26, dy=1)

        # (e) regional contributions ------------------------------------------ #
        axe = _ax(fig, 145.0, y2, 36.0, h2)
        pr = preg.sort_values("exposure_change_mn", ascending=False)
        top = pr.head(9)
        sel = pd.concat([top, pr[pr["abbrev"] == "SAS"]])
        for i, (_, r) in enumerate(sel.iterrows()):
            yy = len(sel) - 1 - i
            v = r["exposure_change_mn"]
            sig = r["hamed_rao_p"] < 0.05
            col = ("#b2182b" if sig else "#f4a582") if v > 0 else ("#2166ac" if sig else "#92c5de")
            axe.barh(yy, v, height=0.62, color=col, zorder=2)
            share = f"{r['share_of_global_exposure_change'] * 100:+.0f}%".replace("-", "\u2212")
            if v > 0:
                axe.text(v + 25, yy, share, va="center", ha="left", fontsize=5.0, color=INK2)
            else:
                axe.text(v / 2, yy, share, va="center", ha="center", fontsize=5.0, color="white")
        axe.set_yticks(range(len(sel))); axe.set_yticklabels(list(reversed(sel["abbrev"].tolist())), fontsize=5.5)
        axe.tick_params(axis="y", length=0)
        axe.axvline(0, color=AXIS, lw=0.5)
        axe.set_xlim(-1000, 1750)
        axe.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        _style_xy(axe, grid_y=False, grid_x=True)
        axe.set_xlabel("Change in exposure, 1996–2025\nminus 1950–1979 (10$^6$ person-months yr$^{-1}$)", fontsize=5.6)
        axe.legend(handles=[Patch(color="#b2182b", label="Per-person trend p < 0.05"),
                            Patch(color="#f4a582", label="p ≥ 0.05")],
                   loc="lower right", bbox_to_anchor=(1.03, 0.12), fontsize=5.0)
        plabel(axe, "e", dx=-20, dy=1)

        # (f) rainfed vs irrigated; (g) per person vs per area ----------------- #
        y3, h3 = 10.0, 26.0
        axf = _ax(fig, 14.0, y3, 74.0, h3)
        for key, col, lab_ in (("crop_frac_rainfed", RAINFED, "Rainfed cropland"),
                               ("crop_frac_irrigated", IRRIG, "Irrigated cropland")):
            y = S[f"{key}|compound"].values
            k = YEARS >= 1979
            t79, fit = _ts(y[k], YEARS[k])
            axf.plot(YEARS, y, color=col, lw=0.8, label=f"{lab_} ({t79:+.2f})")
            axf.plot(YEARS[k], fit, color=col, lw=1.4, alpha=0.9)
        axf.set_xlim(1950, 2025); axf.set_xticks([1950, 1970, 1990, 2010])
        axf.set_ylabel(f"Compound\n({UNIT_C})")
        axf.set_xlabel("Year")
        _style_xy(axf)
        axf.legend(loc="upper left", fontsize=5.3, ncol=1, title=f"Theil–Sen 1979–2025 ({UNIT_CT})",
                   title_fontsize=5.2, alignment="left")
        plabel(axf, "f", dx=-27, dy=1)
        axg = _ax(fig, 106.0, y3, 74.0, h3)
        for key, col, lab_ in (("land|compound", AREA, "Per unit land area"),
                               ("perperson|compound", PERSON, "Per person")):
            y = S[key].values
            t, fit = _ts(y)
            axg.plot(YEARS, y, color=col, lw=0.8, label=f"{lab_} ({t:+.2f})")
            axg.plot(YEARS, fit, color=col, lw=1.4, alpha=0.9)
        axg.set_xlim(1950, 2025); axg.set_xticks([1950, 1970, 1990, 2010])
        axg.set_xlabel("Year")
        _style_xy(axg)
        axg.legend(loc="upper left", fontsize=5.3, ncol=1, title=f"Theil–Sen 1950–2025 ({UNIT_CT})",
                   title_fontsize=5.2, alignment="left")
        plabel(axg, "g", dx=-12, dy=1)
        return save(fig, "fig7_exposure", "fig7_exposure")


# =========================================================================== #
# Fig. 1  Framework: what was computed, with a worked example (Voronezh 2010)
# =========================================================================== #
def _box(cv, x, y, w, h, fc="#f6f5f2", ec="#b9b8b2", lw=0.5, r=1.2, z=1):
    cv.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _arrow(cv, x0, y0, x1, y1, color=INK2, lw=0.6, z=2):
    cv.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=z)


def _tint(c, a):
    rgb = np.array(mpl.colors.to_rgb(c))
    return mpl.colors.to_hex(rgb * a + (1 - a))


def fig_framework_detailed():
    ex = json.loads((CACHE / "example_pixel.json").read_text())
    with plt.rc_context(RC):
        H = 126.0
        fig = _fig(W2, H)
        cv = fig.add_axes([0, 0, 1, 1])
        cv.set_xlim(0, W2); cv.set_ylim(0, H); cv.set_axis_off()

        # ---- inputs -------------------------------------------------------- #
        _box(cv, 2, 111, 117, 13, fc="#ffffff", ec=INK2)
        cv.text(4, 121.6, "ERA5 reanalysis", fontsize=6.6, fontweight="bold", va="top")
        cv.text(4, 117.8, "1950–2025 · 0.25° global grid · 00, 06, 12, 18 UTC (validated against all 24 h)\n"
                          "2-m temperature and dewpoint · 10-m wind · surface pressure · precipitation ·\n"
                          "downward shortwave and longwave radiation", fontsize=5.3, va="top", color=INK2,
                linespacing=1.2)
        _box(cv, 122, 111, 57.5, 13, fc="#ffffff", ec="#b9b8b2")
        cv.text(124, 121.6, "Ancillary layers", fontsize=6.3, fontweight="bold", va="top")
        cv.text(124, 117.8, "Köppen–Geiger · Natural Earth deserts\nIPCC AR6 regions · MIRCA2000 cropland\n"
                            "LandScan population · NOAA RONI", fontsize=5.3, va="top", color=INK2, linespacing=1.2)

        # ---- lanes ---------------------------------------------------------- #
        lanes = [
            ("Heat", HEAT, 2.0, [
                "Daily T$_{max}$\n(largest of the\nfour samples)",
                "T$_{90}$(day of year):\n90th percentile,\n±15 d, 1961–1990",
                "Heatwave: ≥ 3\nconsecutive days\nwith T$_{max}$ > T$_{90}$",
                "Heatwave month:\n≥ 1 heatwave day\nin the month"]),
            ("Drought", DROUGHT, 93.0, [
                "P and FAO-56\nPenman–Monteith\nET$_0$ → PET",
                "Water balance\nD = P − PET\n(monthly)",
                "Log-logistic (GLO)\nfit per calendar\nmonth, 1950–2025",
                "Drought month:\nSPEI-1 ≤ −1"]),
        ]
        lw_, cw, gap = 86.5, 19.0, 2.5
        for name, col, x0, chips in lanes:
            _box(cv, x0, 57, lw_, 50, fc=_tint(col, 0.07), ec=_tint(col, 0.55), lw=0.6)
            cv.text(x0 + 2, 105.3, f"{name} definition", fontsize=6.6, fontweight="bold", va="top", color=INK)
            ax_ = 45.0 if name == "Heat" else 108.0
            _arrow(cv, ax_, 111, ax_, 107.2)
            for i, t in enumerate(chips):
                cx = x0 + 2 + i * (cw + gap)
                _box(cv, cx, 87.5, cw, 13.0, fc="#ffffff", ec=_tint(col, 0.75), lw=0.6, r=0.8, z=2)
                cv.text(cx + cw / 2, 94.0, t, fontsize=5.0, ha="center", va="center", linespacing=1.12, zorder=3)
                if i < 3:
                    _arrow(cv, cx + cw + 0.2, 94.0, cx + cw + gap - 0.2, 94.0, z=3)

        # (a) heat example: daily Tmax vs T90, Apr-Sep
        tx = np.array(ex["tx"]); t90 = np.array(ex["t90"]); hw = np.array(ex["hw_day"]).astype(bool)
        doy = np.arange(1, len(tx) + 1)
        sl = (doy >= 91) & (doy <= 273)
        axa = _ax(fig, 12.0, 61.5, 74.0, 22.5)
        for d in doy[sl & hw]:
            axa.axvspan(d - 0.5, d + 0.5, color=_tint(HEAT, 0.35), lw=0, zorder=0)
        axa.plot(doy[sl], t90[sl], color=HEAT, lw=1.1, zorder=2, label="T$_{90}$ (1961–1990)")
        axa.plot(doy[sl], tx[sl], color=INK, lw=0.6, zorder=3, label="Daily T$_{max}$, 2010")
        axa.set_xlim(91, 273)
        mstart = [91, 121, 152, 182, 213, 244]
        axa.set_xticks([m + 14 for m in mstart]); axa.set_xticklabels(["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
        axa.tick_params(axis="x", length=0)
        for m in mstart[1:]:
            axa.axvline(m - 0.5, color=GRID, lw=0.4, zorder=0)
        axa.set_ylabel("Temperature (°C)")
        axa.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        axa.legend(handles=[Line2D([], [], color=INK, lw=0.6, label="Daily T$_{max}$, 2010"),
                            Line2D([], [], color=HEAT, lw=1.1, label="T$_{90}$, 1961–1990"),
                            Patch(color=_tint(HEAT, 0.35), label="Heatwave day")],
                   loc="upper left", ncol=1, fontsize=5.0, handlelength=1.1, frameon=True, framealpha=0.92,
                   edgecolor="none", facecolor="white", borderpad=0.3)
        axa.set_ylim(np.nanmin(tx[sl]) - 2, np.nanmax(tx[sl]) + 11)
        plabel(axa, "a", dx=-28, dy=1)
        cv.text(12.0, 85.4, f"Worked example: one 0.25° cell near {ex['name']} "
                            f"({ex['lat']:.2f}° N, {ex['lon']:.2f}° E)", fontsize=5.3, color=INK2, va="center")

        # (b) drought example: monthly SPEI-1, 2010
        sp = np.array(ex["spei1"])
        axb = _ax(fig, 104.0, 61.5, 74.0, 22.5)
        cols = [DROUGHT if v <= -1 else "#d6d5cf" for v in sp]
        axb.bar(np.arange(12), sp, width=0.7, color=cols, zorder=2)
        axb.axhline(-1, color=INK, lw=0.6, zorder=3)
        axb.axhline(0, color=AXIS, lw=0.4, zorder=1)
        axb.text(11.45, -1.07, "−1", fontsize=5.2, ha="right", va="top", color=INK)
        axb.set_xticks(range(12)); axb.set_xticklabels(list("JFMAMJJASOND"))
        axb.tick_params(axis="x", length=0)
        axb.set_ylabel("SPEI-1, 2010")
        axb.set_ylim(-2.6, 2.0); axb.set_xlim(-0.6, 11.6)
        axb.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        _style_xy(axb)
        plabel(axb, "b", dx=-24, dy=1)

        # ---- combination ---------------------------------------------------- #
        _arrow(cv, 45.25, 57, 72, 52.4, color=_tint(HEAT, 0.9))
        _arrow(cv, 136.25, 57, 111, 52.4, color=_tint(DROUGHT, 1.0))
        _box(cv, 56, 44.0, 71, 8.2, fc=_tint(COMPOUND, 0.10), ec=_tint(COMPOUND, 0.6), lw=0.6, r=2.5)
        cv.text(91.5, 49.6, "Compound month = heatwave month AND drought month", fontsize=6.0,
                fontweight="bold", ha="center", va="center", color=INK)
        cv.text(91.5, 46.2, "evaluated for every 0.25° land cell and every month, 1950–2025", fontsize=5.2,
                ha="center", va="center", color=INK2)

        # (c) monthly flags for 2010
        axc = _ax(fig, 22.0, 28.6, 62.0, 11.0)
        rows_ = [("Heatwave", ex["heat"], HEAT), ("Drought", ex["drought"], DROUGHT), ("Compound", ex["compound"], COMPOUND)]
        for r, (lab_, flags, col) in enumerate(rows_):
            yy = 2 - r
            for m, f in enumerate(flags):
                on = bool(f) and np.isfinite(f)
                axc.add_patch(Rectangle((m - 0.42, yy - 0.38), 0.84, 0.76, fc=col if on else "white",
                                        ec=col if on else "#cfcec8", lw=0.5))
        axc.set_xlim(-0.6, 11.6); axc.set_ylim(-0.6, 2.6)
        axc.set_yticks([2, 1, 0]); axc.set_yticklabels([r[0] for r in rows_])
        axc.set_xticks(range(12)); axc.set_xticklabels(list("JFMAMJJASOND"))
        axc.tick_params(length=0)
        for sp_ in axc.spines.values():
            sp_.set_visible(False)
        axc.text(-0.6, 2.75, "Monthly flags, 2010 (3 compound months)", fontsize=5.2, ha="left", va="bottom", color=INK2)
        plabel(axc, "c", dx=-38, dy=1)

        # (d) annual count, 1950-2025, at this cell
        ann = np.array(ex["annual_compound"])
        axd = _ax(fig, 104.0, 28.6, 74.0, 11.0)
        axd.bar(YEARS, ann, width=0.8, color=[COMPOUND if y == ex["year"] else _tint(COMPOUND, 0.45) for y in YEARS],
                zorder=2)
        em = ann[(YEARS >= EARLY[0]) & (YEARS <= EARLY[1])].mean()
        lm = ann[(YEARS >= LATE[0]) & (YEARS <= LATE[1])].mean()
        for (a_, b_), m_ in ((EARLY, em), (LATE, lm)):
            axd.plot([a_ - 0.4, b_ + 0.4], [m_, m_], color=INK, lw=0.9, zorder=3, solid_capstyle="butt")
        axd.set_xlim(1949, 2026); axd.set_xticks([1950, 1970, 1990, 2010])
        axd.set_ylim(0, max(4, ann.max() + 0.5)); axd.set_yticks([0, 2, 4])
        axd.set_ylabel("Months yr$^{-1}$")
        _style_xy(axd)
        axd.text(1950, axd.get_ylim()[1] * 1.02, f"Compound months per year at this cell; 30-yr means "
                 f"{em:.2f} (1950–1979) → {lm:.2f} (1996–2025)", fontsize=5.2, color=INK2, va="bottom", ha="left")
        plabel(axd, "d", dx=-24, dy=1)


        # ---- analyses ----------------------------------------------------------- #
        _arrow(cv, 91.5, 26.6, 91.5, 23.0)
        boxes = [
            ("Trend and timing", "Figs 3, 4",
             "Theil–Sen slope, Mann–Kendall test\n(Hamed–Rao for aggregate series)\n95% block-bootstrap intervals\n"
             "Pettitt change points; slope change\n1979–2001 vs 2002–2025"),
            ("Where", "Figs 2, 4–6",
             "Every 0.25° land cell and the land\nmean (Antarctica excluded)\n44 IPCC AR6 land regions\n"
             "5 UNEP aridity classes (AI = P/PET)\n15 named deserts (Köppen B mask)"),
            ("Why", "Fig. 5",
             "Heatwave and drought marginals\nCounterfactual water balances\n"
             "D$^{P}$ = P − $\\overline{PET}$,  D$^{PET}$ = $\\overline{P}$ − PET\npassed through the same SPEI fit"),
            ("Who and what is exposed", "Fig. 7",
             "Cropland: rainfed and irrigated\nPopulation: per person and total\n"
             "Exposure change: hazard vs\npopulation; ENSO phase (SI)"),
        ]
        bw, bg = 43.5, 2.0
        for i, (ttl, figs, body) in enumerate(boxes):
            bx = 2 + i * (bw + bg)
            _box(cv, bx, 2.2, bw, 20.6, fc="#f6f5f2", ec="#b9b8b2")
            cv.text(bx + 1.8, 21.0, ttl, fontsize=6.3, fontweight="bold", va="top")
            cv.text(bx + bw - 1.8, 21.0, figs, fontsize=5.4, va="top", ha="right", color=INK2)
            cv.text(bx + 1.8, 17.3, body, fontsize=5.1, va="top", color=INK, linespacing=1.22)
        # ancillary layers feed the aggregation / exposure analyses (bottom row)
        cv.plot([179.5, 181.6, 181.6], [117.5, 117.5, 24.6], color="#b9b8b2", lw=0.6, zorder=1,
                solid_capstyle="butt")
        _arrow(cv, 181.6, 24.6, 181.6, 22.9, color="#b9b8b2")
        return save(fig, "fig1_framework_detailed")


# =========================================================================== #
# Supplementary figures
# =========================================================================== #
def figS_mean_map():
    """Mean annual compound, heatwave and drought frequency, 1950-2025."""
    F = fields()
    with plt.rc_context(RC):
        H = 128.0
        fig = _fig(W2, H)
        specs = [("compound", 0.5, H - 4 - 141 * map_aspect(), 141.0,
                  [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], "a", 2.0),
                 ("heat", 0.5, 12.0, 90.0, [3.0, 3.5, 4.0, 4.5, 5.0, 5.5], "c", 2.5),
                 ("drought", 92.5, 12.0, 90.0, [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3], "d", 2.5)]
        seqs = {"compound": ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
                "heat": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"],
                "drought": ["#fffbe6", "#fff0b3", "#fde08a", "#f9c74f", "#eda100", "#c27f00", "#8f5c00", "#5f3d00"]}
        ya = specs[0][2]
        for v, x, y, w, lv, letter, step in specs:
            ax = map_ax(fig, x, y, w)
            land_bg(ax)
            cm_ = LinearSegmentedColormap.from_list(v, seqs[v], N=256)
            cmap, norm = binned(lv, cm_, extend="both")
            m = mesh(ax, F[f"{v}_mean"].values, cmap, norm)
            contour_mask(ax, F["desert"].values, "#5f2d0c", lw=0.35)
            coast(ax)
            unit = UNIT_C if v == "compound" else UNIT_M
            cw = 70.0 if v == "compound" else 60.0
            cb = cbar(fig, m, x + (w - cw) / 2, y - 4.2, cw, 1.7, f"{VAR_NAME[v]} mean frequency, 1950–2025 ({unit})",
                      ticks=lv, extend="both")
            cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
            ax.text(0.0, 0.10, VAR_NAME[v] + (" months" if v != "compound" else ""), transform=ax.transAxes,
                    fontsize=6.5, fontweight="bold", ha="left", va="bottom")
            plabel(ax, letter, dx=2, dy=-4)
        h_a = 141 * map_aspect()
        axz = _ax(fig, 151.0, ya + 23.0, 29.0, h_a - 25.0)
        lat = F["latitude"].values
        land = F["land"].values.astype(bool)
        for v in ("heat", "drought", "compound"):
            c = np.where(land, F[f"{v}_mean"].values, np.nan)
            n = np.isfinite(c).sum(1)
            with np.errstate(invalid="ignore"):
                z = np.nanmean(c, axis=1)
            z[n < 20] = np.nan
            axz.plot(pd.Series(z).rolling(9, center=True, min_periods=5).mean().values, lat, color=VAR_COL[v], lw=1.0,
                     label=VAR_NAME[v])
        axz.set_ylim(-56, 84); axz.set_yticks([-40, 0, 40, 80])
        axz.set_yticklabels(["40° S", "0°", "40° N", "80° N"])
        axz.set_xlim(0, 6.2); axz.set_xticks([0, 2, 4, 6])
        _style_xy(axz)
        axz.set_xlabel(f"Zonal-mean frequency\nover land ({UNIT_M})", fontsize=5.8)
        axz.legend(loc="upper left", bbox_to_anchor=(-0.15, -0.32), fontsize=5.3)
        plabel(axz, "b", dx=-22, dy=0)
        return save(fig, "figS_mean_maps", "figS_mean_maps")


def figS_changepoint_map():
    from chd import manuscript_figures as mf
    ccrs, _ = _cartopy()
    rs = table("regional_stats").set_index("abbrev")
    clip, rep = mf._ar6_clipped()
    bins = [1960, 1980, 1985, 1990, 1995, 2000, 2005]
    cols = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#fdd0a2", "#f16913", "#8c2d04"]
    cols = ["#253494", "#2c7fb8", "#41b6c4", "#a1dab4", "#fed976", "#fd8d3c", "#bd0026"][:len(bins) - 1]
    cmap = ListedColormap(cols); norm = BoundaryNorm(bins, cmap.N)
    with plt.rc_context(RC):
        H = 92.0
        fig = _fig(W2, H)
        w = 166.0
        ax = map_ax(fig, 8.5, H - 6 - w * map_aspect(), w)
        land_bg(ax, "#f4f3f0")
        for ab, g in clip.items():
            if ab in mf._EXCLUDE_REGIONS or ab not in rs.index:
                continue
            cp, p = rs.loc[ab, "changepoint_year"], rs.loc[ab, "changepoint_p"]
            geoms = mf._polygonal(g)
            ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor=cmap(norm(cp)), edgecolor="white", lw=0.4, zorder=1)
            if p >= 0.05:
                ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none", edgecolor=INK2, lw=0,
                                  hatch="//////", zorder=1.5)
        coast(ax, lw=0.25, color="#8a8984", z=2)
        tr = ccrs.PlateCarree()._as_mpl_transform(ax)
        for ab, (lo, la) in rep.items():
            if ab in mf._EXCLUDE_REGIONS or ab not in rs.index:
                continue
            off = {k: (v[0] * 1.6, v[1] * 1.6) for k, v in _RING_OFFSETS.items()}.get(ab, (0, 0))
            cp = int(rs.loc[ab, "changepoint_year"])
            ax.annotate(f"{ab}\n{cp}", xy=(lo, la), xycoords=tr, xytext=off, textcoords="offset points",
                        fontsize=5.0, ha="center", va="center", color=INK, zorder=4, linespacing=1.0,
                        path_effects=[pe.withStroke(linewidth=1.3, foreground="white")],
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.35) if off != (0, 0) else None)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = cbar(fig, sm, 50, 8.0, 83, 1.8, "Pettitt change-point year of the regional annual compound series "
                                             "(hatched: p ≥ 0.05)", ticks=[0.5 * (a_ + b_) for a_, b_ in zip(bins[:-1], bins[1:])])
        cb.ax.set_xticklabels(["before 1980", "1980–84", "1985–89", "1990–94", "1995–99", "2000–04"])
        cb.ax.tick_params(length=0)
        return save(fig, "figS_changepoint_map", "figS_changepoint_map")


def figS_ring_box():
    """Pre/post change-point boxplots around the AR6 map (restyled SI version)."""
    from chd import manuscript_figures as mf
    rs = table("regional_stats").set_index("abbrev")
    R = reg_series()
    with plt.rc_context(RC):
        fig = _fig(RING_W, RING_H)
        diff = {ab: rs.loc[ab, "post_cp_mean"] - rs.loc[ab, "pre_cp_mean"] for ab in mf.SLOTS}
        lv = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
        cmap, norm = binned(lv, DIV, center=0, vmax=2.2, extend="neither")
        hatch = {ab: rs.loc[ab, "changepoint_p"] >= 0.05 for ab in mf.SLOTS}
        axm, x0, wmap = _ring_map(fig, diff, cmap, norm, hatch=hatch)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = cbar(fig, sm, x0 + 18, 52.5, wmap - 36, 1.6,
                  f"Post- minus pre-change-point mean ({UNIT_C})", ticks=lv)
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        fig.text(0.5, 40.0 / RING_H, "Panels: annual compound counts before (open) and after (filled) each region's\n"
                                     "Pettitt change point (year at top right; n.s.: p ≥ 0.05). Map hatching: change point n.s.",
                 fontsize=5.6, ha="center", va="center", color=INK2, linespacing=1.2)
        axes = _ring_panel_axes(fig)
        for ab, ax in axes.items():
            y = R[f"{ab}|compound"].values
            cp, p = int(rs.loc[ab, "changepoint_year"]), rs.loc[ab, "changepoint_p"]
            pre, post = y[YEARS <= cp], y[YEARS > cp]
            bp = ax.boxplot([pre, post], positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, lw=0.8), whiskerprops=dict(color=INK2, lw=0.5),
                            capprops=dict(color=INK2, lw=0.5), boxprops=dict(lw=0.5))
            bp["boxes"][0].set(facecolor="white", edgecolor=COMPOUND)
            bp["boxes"][1].set(facecolor=_tint(COMPOUND, 0.55), edgecolor=COMPOUND)
            ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 6); ax.set_yticks([0, 3, 6])
            ax.set_yticklabels(["0", "", "6"]); ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
            ax.tick_params(length=1.4, width=0.4, pad=0.8, labelsize=4.6)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(0.4)
            ax.text(0.0, 1.02, ab, transform=ax.transAxes, fontsize=5.4, fontweight="bold", ha="left", va="bottom")
            ax.text(1.0, 1.02, f"{cp}" + ("" if p < 0.05 else " n.s."), transform=ax.transAxes, fontsize=5.0,
                    color=INK if p < 0.05 else MUTED, ha="right", va="bottom")
        return save(fig, "figS_ring_cp_boxplot", "figS_ring_cp_boxplot")


def figS_convergence():
    rs = table("regional_stats").set_index("abbrev")
    att = table("regional_attribution_stats").iloc[0]
    with plt.rc_context(RC):
        fig = _fig(W1, 82)
        ax = _ax(fig, 12.0, 10.0, 70.0, 66.0)
        e, l = rs["compound_early_mean"], rs["compound_late_mean"]
        dtr = rs["drought_trend_per_decade"]
        lv = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        cmap, norm = binned(lv, DIV, center=0, vmax=0.55, extend="neither")
        ax.plot([0, 2.8], [0, 2.8], color=MUTED, lw=0.6, zorder=1)
        ax.text(2.55, 2.42, "1:1", fontsize=5.3, color=MUTED, rotation=45, ha="center")
        sc = ax.scatter(e, l, s=12, c=dtr, cmap=cmap, norm=norm, edgecolors=INK2, linewidths=0.3, zorder=3)
        offs = {"NEAF": (3, 3), "SAH": (3, -2)}
        for ab in ("SAH", "NEAF", "SEAF", "ARP", "MDG", "NES", "SAS", "CNA", "SAU"):
            ax.annotate(ab, (e[ab], l[ab]), xytext=offs.get(ab, (3, 0)), textcoords="offset points", fontsize=4.8,
                        color=INK2, va="center")
        ax.set_xlim(0, 2.8); ax.set_ylim(0, 2.8); ax.set_aspect("equal")
        ax.set_xlabel(f"1950–1979 mean ({UNIT_C})"); ax.set_ylabel(f"1996–2025 mean ({UNIT_C})")
        _style_xy(ax, grid_y=True, grid_x=True)
        ax.text(0.03, 0.97, f"r = {att['r_early_late']:.2f}\nCV {att['cv_early']:.2f} → {att['cv_late']:.2f}",
                transform=ax.transAxes, fontsize=5.6, va="top", linespacing=1.2)
        cb = cbar(fig, sc, 60.0, 16.0, 1.6, 26.0, "Drought-month\ntrend", ticks=lv, orient="vertical")
        cb.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        return save(fig, "figS_convergence", "figS_convergence")


def figS_patch_multiples():
    ps = table("dryland_patch_stats").set_index("letter")
    P = pd.read_csv(CACHE / "patch_series.csv", index_col=0)
    order = ps.sort_values("trend_per_decade", ascending=False).index.tolist()
    with plt.rc_context(RC):
        H = 100.0
        fig = _fig(W2, H)
        ncol, nrow = 5, 3
        pw, ph, gx, gy = 29.5, 22.0, 5.2, 9.0
        x0, y0 = 11.0, 8.0
        for k, L in enumerate(order):
            r, c = divmod(k, ncol)
            ax = _ax(fig, x0 + c * (pw + gx), y0 + (nrow - 1 - r) * (ph + gy), pw, ph)
            y = P[L].values
            cp, cpp = int(ps.loc[L, "changepoint_year"]), ps.loc[L, "changepoint_p"]
            sig = ps.loc[L, "hamed_rao_p"] < 0.05
            _ring_ts_panel(ax, y, "compound", 7.0, [0, 3.5, 7], cp=cp if cpp < 0.05 else None,
                           trend=ps.loc[L, "trend_per_decade"], sig=sig, title=f"{L}  {ps.loc[L, 'name']}",
                           show_x=(r == nrow - 1), big=True)
            for (a, b), mcol in ((EARLY, MUTED), (LATE, MUTED)):
                kk = (YEARS >= a) & (YEARS <= b)
                ax.plot([a, b], [y[kk].mean()] * 2, color=mcol, lw=0.9, zorder=2.5, solid_capstyle="butt")
            if c == 0:
                ax.set_ylabel(f"Compound\n({UNIT_C})", fontsize=5.6)
        return save(fig, "figS_desert_series", "figS_desert_series")


def figS_enso():
    F = fields()
    E = pd.read_csv(CACHE / "enso_series.csv", index_col=0)
    st = table("enso_global_stats").set_index("phase")
    with plt.rc_context(RC):
        H = 98.0
        fig = _fig(W2, H)
        lv = [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8]
        cmap, norm = binned(lv, DIV, center=0, vmax=9, extend="both")
        for k, (key, ttl, letter) in enumerate((("enso_el_nino", "El Niño minus Neutral", "a"),
                                                 ("enso_la_nina", "La Niña minus Neutral", "b"))):
            ax = map_ax(fig, 0.5 + k * 92.0, H - 3 - 90 * map_aspect(), 90.0)
            land_bg(ax)
            m = mesh(ax, F[key].values, cmap, norm)
            coast(ax)
            ax.text(0.0, 0.08, ttl, transform=ax.transAxes, fontsize=6.5, fontweight="bold", ha="left")
            plabel(ax, letter, dx=2, dy=-4)
        cb = cbar(fig, m, 56, H - 3 - 90 * map_aspect() - 4.2, 70, 1.7,
                  "Compound-month frequency anomaly, trend removed (percentage points)", ticks=lv, extend="both")
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        ax = _ax(fig, 14.0, 9.0, 120.0, 30.0)
        colmap = {"El Nino": ELNINO, "La Nina": LANINA, "Neutral": "#c9c8c2"}
        ax.bar(E.index, E["anom"], width=0.75, color=[colmap.get(p, "#c9c8c2") for p in E["phase"]], zorder=2)
        ax.axhline(0, color=AXIS, lw=0.5)
        ax.set_xlim(1949, 2026); ax.set_xticks([1950, 1970, 1990, 2010])
        ax.set_ylabel(f"Land-mean anomaly\n({UNIT_C})")
        ax.set_xlabel("Year (phase of the December–February season)")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        _style_xy(ax)
        ax.legend(handles=[Patch(color=ELNINO, label="El Niño"), Patch(color=LANINA, label="La Niña"),
                           Patch(color="#c9c8c2", label="Neutral")], loc="upper left", ncol=3, fontsize=5.4)
        plabel(ax, "c", dx=-27)
        ax2 = _ax(fig, 148.0, 9.0, 32.0, 30.0)
        g = E.groupby("phase")["anom"]
        for i, ph in enumerate(["El Nino", "Neutral", "La Nina"]):
            vals = g.get_group(ph).values
            ax2.scatter(np.full(len(vals), i) + np.linspace(-0.18, 0.18, len(vals)), vals, s=5, color=colmap[ph],
                        edgecolors="none", alpha=0.8, zorder=2)
            ax2.plot([i - 0.3, i + 0.3], [vals.mean()] * 2, color=INK, lw=1.1, zorder=3)
        ax2.axhline(0, color=AXIS, lw=0.5)
        ax2.set_xticks(range(3)); ax2.set_xticklabels(["El Niño", "Neutral", "La Niña"])
        ax2.tick_params(axis="x", length=0)
        ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
        _style_xy(ax2)
        p = st.loc["El Nino", "mannwhitney_p_EN_vs_LN_detrended"]
        ax2.text(0.5, 1.02, f"El Niño vs La Niña p = {p:.3f}", transform=ax2.transAxes, fontsize=5.3,
                 ha="center", va="bottom", color=INK2)
        plabel(ax2, "d", dx=-20, dy=6)
        return save(fig, "figS_enso", "figS_enso")


def figS_sampling():
    d0 = Path(__file__).resolve().parents[2] / "analysis" / "sampling_validation"
    d = pd.read_csv(d0 / "validation_points.csv")
    A = np.load(d0 / "validation_annual.npz")
    pts = pd.read_csv(d0 / "points.csv").set_index("pid").loc[A["pid"]]
    with plt.rc_context(RC):
        H = 46.0
        fig = _fig(W2, H)
        for k, v in enumerate(("compound", "heat", "drought")):
            ax = _ax(fig, 12.0 + k * 38.0, 8.0, 30.0, 30.0)
            x, y = d[f"{v}_trend_24h"], d[f"{v}_trend_6h"]
            lo = min(x.min(), y.min()) - 0.05; hi = max(x.max(), y.max()) + 0.05
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, zorder=1)
            for i, c in enumerate(CLASSES):
                mm = d.cls == c
                ax.scatter(x[mm], y[mm], s=6, color=AI_COLS[i], edgecolors="white", linewidths=0.3, zorder=2,
                           label=AI_LABELS[i])
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_num))
            _style_xy(ax, grid_y=True, grid_x=True)
            ax.text(0.04, 0.96, f"{VAR_NAME[v]}\nr = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes,
                    va="top", fontsize=5.6, linespacing=1.15)
            ax.set_xlabel(f"Trend, all 24 h ({UNIT_MT})", fontsize=5.6)
            if k == 0:
                ax.set_ylabel(f"Trend, 4 synoptic hours\n({UNIT_MT})", fontsize=5.6)
                ax.legend(loc="lower right", fontsize=4.8, handletextpad=0.1, borderaxespad=0.0)
            plabel(ax, "abc"[k], dx=-24 if k == 0 else -12, dy=2)
        ax = _ax(fig, 132.0, 8.0, 48.0, 30.0)
        w = np.cos(np.deg2rad(pts["lat"].values))
        for tag, col, lab in (("24h", P_COL, "All 24 hours"), ("6h", PET_COL, "00/06/12/18 UTC (this study)")):
            s = (A[f"compound_{tag}"] * w[:, None]).sum(0) / w.sum()
            t, fit = _ts(s)
            ax.plot(YEARS, s, color=col, lw=0.8, label=f"{lab} ({t:+.3f})")
            ax.plot(YEARS, fit, color=col, lw=1.2, alpha=0.8)
        ax.set_xlim(1950, 2025); ax.set_xticks([1950, 1970, 1990, 2010])
        ax.set_ylabel(f"Compound ({UNIT_C})", fontsize=5.6)
        _style_xy(ax)
        ax.legend(loc="upper left", fontsize=5.0, title=f"Mean of 120 cells; Theil–Sen ({UNIT_MT})",
                  title_fontsize=5.0, alignment="left")
        plabel(ax, "d", dx=-22, dy=2)
        return save(fig, "figS_sampling", "figS_sampling")


def build_figures(names=None):
    reg = {
        "fig1": fig_framework, "fig2": fig_change_maps, "fig3": fig_global_record,
        "fig4": lambda: fig_ring("compound", "fig4_ring_compound", "fig4_ring_compound"),
        "fig5": fig_aridity, "fig6": fig_deserts, "fig7": fig_exposure,
        "S_mean": figS_mean_map, "S_cp": figS_changepoint_map,
        "S_heat": lambda: fig_ring("heat", "figS_ring_heat", "figS_ring_heat"),
        "S_drought": lambda: fig_ring("drought", "figS_ring_drought", "figS_ring_drought"),
        "S_box": figS_ring_box, "S_conv": figS_convergence, "S_desert": figS_patch_multiples,
        "S_enso": figS_enso, "S_sampling": figS_sampling,
    }
    out = []
    for k, f in reg.items():
        if names is None or k in names:
            out.append(f())
    return out


# =========================================================================== #
# Fig. 1 (representational): the study as a picture, built from data glyphs
# =========================================================================== #
def _rgba_field(values, cmap, norm, land, step=4, lat_rng=(-57, 84)):
    """Downsample a lat x lon field to ~1 deg RGBA (ocean transparent) for glyph layers."""
    F = fields()
    lat = F["latitude"].values
    keep = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    v = np.where(land, values, np.nan)[keep][::step, ::step]
    lm = land[keep][::step, ::step]
    rgba = cmap(norm(v)) if norm is not None else cmap(v)
    rgba[..., 3] = np.where(lm & np.isfinite(v), 1.0, 0.0)
    return rgba


def _layer(cv, rgba, x0, y0, w, h, dx, z, fc="#ffffff", ec="#9a9994", lw=0.45):
    """One map 'sheet' in oblique projection: unit square -> parallelogram (x0,y0,w,h,dx)."""
    from matplotlib.transforms import Affine2D
    from matplotlib.patches import Polygon
    pts = [(x0, y0), (x0 + w, y0), (x0 + w + dx, y0 + h), (x0 + dx, y0 + h)]
    cv.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", zorder=z))
    if rgba is not None:
        M = Affine2D(np.array([[w, dx, x0], [0, h, y0], [0, 0, 1.0]]))
        cv.imshow(rgba, extent=(0, 1, 0, 1), origin="upper", transform=M + cv.transData,
                  zorder=z + 0.05, interpolation="nearest", aspect="auto")
    cv.add_patch(Polygon(pts, closed=True, fc="none", ec=ec, lw=lw, zorder=z + 0.1))


def _stack(cv, top_rgba, x0, y0, w, h, dx, n, gap, z0=2, under="#f3f2ee"):
    for k in range(n - 1, -1, -1):
        _layer(cv, top_rgba if k == 0 else None, x0, y0 - k * gap, w, h, dx, z0 + (n - k),
               fc="#ffffff" if k == 0 else under)


def _glyph_ax(fig, x, y, w, h):
    ax = _ax(fig, x, y, w, h)
    ax.set_axis_off()
    return ax


def fig_framework():
    from shapely.geometry import Point
    from matplotlib.patches import Polygon as MPoly, Circle
    F = fields()
    S = series()
    ex = json.loads((CACHE / "example_pixel.json").read_text())
    land = F["land"].values.astype(bool)
    dec = table("aridity_driver_decomposition").set_index("class")
    gs = table("global_stats").set_index("series")
    with plt.rc_context(RC):
        H = 104.0
        fig = _fig(W2, H)
        cv = fig.add_axes([0, 0, 1, 1])
        cv.set_xlim(0, W2); cv.set_ylim(0, H); cv.set_axis_off()

        def title(x, y, s, letter=None, sub=None, ha="left"):
            if letter:
                cv.text(x, y, letter, fontsize=8, fontweight="bold", va="baseline", ha="left")
                x = x + 4.0
            cv.text(x, y, s, fontsize=6.5, fontweight="bold", va="baseline", ha=ha)
            if sub:
                cv.text(x, y - 3.2, sub, fontsize=5.6, color=INK2, va="baseline", ha=ha)

        arrow = dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color="#7d7c77", lw=0.7,
                     shrinkA=0, shrinkB=0)

        # (a) ERA5 input cube ------------------------------------------------ #
        landmap = _rgba_field(np.where(land, 1.0, np.nan), ListedColormap(["#c9c7c0"]), None, land)
        _stack(cv, landmap, 3.0, 66.0, 25.0, 12.5, 7.0, 5, 3.0)
        title(2.0, 98.0, "ERA5", "a", "1950–2025, 0.25°")
        cv.annotate("", xy=(3.0 + 25 + 7 + 1.0, 78.5 + 1.8), xytext=(3.0 + 25 + 7 + 1.0, 66.0 - 12.0),
                    arrowprops=dict(arrowstyle="<->,head_length=0.25,head_width=0.12", color=MUTED, lw=0.5))
        cv.text(3.0 + 25 + 7 + 2.2, 61.0, "time", rotation=90, fontsize=5.4, color=MUTED, va="center")

        # fork arrows to the two definitions
        cv.annotate("", xy=(47.0, 86.0), xytext=(38.5, 76.0), arrowprops=arrow)
        cv.annotate("", xy=(47.0, 62.0), xytext=(38.5, 70.0), arrowprops=arrow)

        # (b) heatwave glyph: real daily Tmax vs T90, Voronezh 2010 ---------- #
        tx = np.array(ex["tx"]); t90 = np.array(ex["t90"]); hw = np.array(ex["hw_day"]).astype(bool)
        doy = np.arange(1, len(tx) + 1)
        sl = (doy >= 120) & (doy <= 260)
        axb = _glyph_ax(fig, 49.0, 76.0, 46.0, 17.0)
        for d in doy[sl & hw]:
            axb.axvspan(d - 0.5, d + 0.5, color=_tint(HEAT, 0.30), lw=0, zorder=0)
        axb.plot(doy[sl], t90[sl], color=HEAT, lw=1.3, zorder=2)
        axb.plot(doy[sl], tx[sl], color=INK, lw=0.5, zorder=3)
        axb.set_xlim(120, 260)
        axb.text(261, t90[259], " T$_{90}$", color=INK, fontsize=5.6, va="center")
        axb.text(119, tx[119], "T$_{max}$ ", color=INK, fontsize=5.6, va="center", ha="right")
        axb.plot([120, 260], [np.nanmin(tx[sl]) - 1.0] * 2, color="#c9c8c2", lw=0.5)
        title(49.0, 98.0, "Heatwave month", "b", "≥ 3 consecutive days above T$_{90}$")

        # (c) drought glyph: real monthly SPEI-1, 2010 ----------------------- #
        sp = np.array(ex["spei1"])
        axc = _glyph_ax(fig, 49.0, 50.0, 46.0, 15.0)
        axc.bar(np.arange(12), sp, width=0.72, color=[DROUGHT if v <= -1 else "#d6d5cf" for v in sp], zorder=2)
        axc.axhline(-1, color=INK, lw=0.6, zorder=3)
        axc.axhline(0, color="#c9c8c2", lw=0.5, zorder=1)
        axc.text(12.0, -1, " −1", fontsize=5.6, va="center", color=INK)
        axc.set_xlim(-0.6, 11.6)
        title(49.0, 71.0, "Drought month", "c", "SPEI-1 of P − PET ≤ −1")

        # converge to the Venn
        cv.annotate("", xy=(107.5, 84.5), xytext=(98.0, 86.0), arrowprops=arrow)
        cv.annotate("", xy=(117.0, 70.5), xytext=(98.0, 60.0), arrowprops=arrow)

        # (d) compound = overlap, and the 2010 months ----------------------- #
        cx, cy, r, sep = 126.0, 82.0, 10.5, 7.0
        cl, cr = Point(cx - sep, cy).buffer(r, 128), Point(cx + sep, cy).buffer(r, 128)
        cv.add_patch(MPoly(np.array(cl.exterior.coords), closed=True, fc=_tint(HEAT, 0.28), ec=HEAT, lw=0.8, zorder=2))
        cv.add_patch(MPoly(np.array(cr.exterior.coords), closed=True, fc=_tint(DROUGHT, 0.30), ec=DROUGHT, lw=0.8, zorder=2))
        lens = cl.intersection(cr)
        cv.add_patch(MPoly(np.array(lens.exterior.coords), closed=True, fc=COMPOUND, ec=COMPOUND, lw=0.8, zorder=3))
        cv.text(cx - sep - 4.6, cy, "Heat-\nwave", fontsize=6.0, ha="center", va="center", color=INK, linespacing=1.05)
        cv.text(cx + sep + 4.6, cy, "Drought", fontsize=6.0, ha="center", va="center", color=INK)
        cv.text(cx, cy, "CHD", fontsize=6.2, fontweight="bold", ha="center", va="center", color="white", zorder=4)
        title(108.0, 98.0, "Compound month", "d")
        # month strip for the example cell
        hh = np.array(ex["heat"]).astype(bool); dd = np.array(ex["drought"]) > 0.5
        sx, sy, sw = 108.5, 57.5, 2.55
        for m in range(12):
            if hh[m] and dd[m]:
                fc, ec = COMPOUND, COMPOUND
            elif hh[m]:
                fc, ec = _tint(HEAT, 0.28), HEAT
            elif dd[m]:
                fc, ec = _tint(DROUGHT, 0.30), DROUGHT
            else:
                fc, ec = "white", "#c9c8c2"
            cv.add_patch(Rectangle((sx + m * (sw + 0.45), sy), sw, sw, fc=fc, ec=ec, lw=0.5, zorder=3))
            cv.text(sx + m * (sw + 0.45) + sw / 2, sy - 0.9, "JFMAMJJASOND"[m], fontsize=4.8, ha="center",
                    va="top", color=INK2)
        cv.text(sx, sy + sw + 1.4, "One cell near Voronezh, 2010", fontsize=5.5, color=INK2, va="bottom")

        # (e) record: annual counts for every cell and year ---------------- #
        cmap, norm = binned(CHANGE_LEVELS["compound"], DIV, center=0, vmax=CHANGE_VMAX["compound"])
        chg = _rgba_field(F["compound_change"].values, cmap, norm, land)
        cv.annotate("", xy=(153.0, 78.0), xytext=(145.0, 78.0), arrowprops=arrow)
        _stack(cv, chg, 154.0, 70.0, 21.0, 11.0, 6.0, 5, 2.6)
        title(153.0, 98.0, "Every cell, every year", "e", "76 annual counts per cell")
        cv.text(154.0 + 13.5, 56.5, "Change in compound months,\n1996–2025 minus 1950–1979", fontsize=5.3,
                color=INK2, ha="center", va="top", linespacing=1.1)

        # (f) analyses ------------------------------------------------------- #
        _box(cv, 1.5, 2.0, W2 - 3.0, 43.0, fc="#f6f5f2", ec="none", r=1.5)
        cv.annotate("", xy=(167.0, 45.6), xytext=(167.0, 51.0), arrowprops=arrow)
        cv.text(3.0, 40.0, "f", fontsize=8, fontweight="bold", va="baseline")
        xs = [10.0, 54.0, 98.0, 142.0]
        tw = 37.0
        heads = [("When", "Fig. 3, 4"), ("Where", "Figs 2, 4–6"), ("Why", "Fig. 5"), ("Who is exposed", "Fig. 7")]
        for x, (h1, h2) in zip(xs, heads):
            cv.text(x, 40.0, h1, fontsize=6.5, fontweight="bold", va="baseline")
            cv.text(x + tw, 40.0, h2, fontsize=5.6, color=INK2, va="baseline", ha="right")
        # f1: land-mean record with period fits and the change point
        a1 = _ax(fig, xs[0], 9.0, tw, 26.0)
        y = S["land|compound"].values
        st = gs.loc["compound"]
        a1.axvspan(st["pettitt_boot_p05"] - 0.5, st["pettitt_boot_p95"] + 0.5, color="#dddcd6", lw=0)
        a1.plot(YEARS, y, color=COMPOUND, lw=0.8)
        for (a, b) in SEGMENTS:
            k = (YEARS >= a) & (YEARS <= b)
            a1.plot(YEARS[k], _ts(y[k], YEARS[k])[1], color=INK, lw=1.1)
        a1.set_xlim(1950, 2025); a1.set_xticks([1950, 2025]); a1.set_yticks([])
        a1.spines["left"].set_visible(False); a1.patch.set_visible(False)
        a1.tick_params(labelsize=5.2, length=1.5)
        a1.set_title("Trends, acceleration, change points", fontsize=5.6, color=INK2, pad=2, loc="left")
        # f2: aridity classes (units of aggregation)
        a2 = _glyph_ax(fig, xs[1], 9.0, tw, 26.0)
        rg = _rgba_field(np.minimum(F["ai"].values, 9.0), ListedColormap(AI_COLS),
                         BoundaryNorm(AI_EDGES[:-1] + [10.0], 5), land, step=3)
        a2.imshow(rg, extent=(-180, 180, -57, 84), aspect="auto", interpolation="nearest")
        a2.set_xlim(-180, 180); a2.set_ylim(-57, 84)
        a2.set_title("AR6 regions, aridity classes, deserts", fontsize=5.6, color=INK2, pad=2, loc="left")
        # f3: P vs PET counterfactual, drought-month trend
        a3 = _ax(fig, xs[2] + 2.0, 9.0, tw - 2.0, 26.0)
        groups = [("hyper-arid", "Hyper-arid"), ("humid", "Humid")]
        for gi, (c, lab_) in enumerate(groups):
            d = dec.loc[c]
            vals = [d["drought_trend_actual"], d["drought_trend_PET_only"], d["drought_trend_P_only"]]
            for j, (vv, col) in enumerate(zip(vals, (INK2, PET_COL, P_COL))):
                a3.bar(gi * 4 + j, vv, width=0.8, color=col)
            a3.text(gi * 4 + 1, -0.06, lab_, fontsize=5.3, ha="center", va="top")
        a3.axhline(0, color=AXIS, lw=0.5)
        a3.set_ylim(-0.12, 0.45); a3.set_xlim(-0.8, 6.8)
        a3.set_axis_off()
        for j, (lab_, col) in enumerate((("actual", INK2), ("PET only", PET_COL), ("P only", P_COL))):
            a3.text(6.9, 0.42 - j * 0.065, lab_, fontsize=5.2, color=INK, ha="right", va="center")
            a3.add_patch(Rectangle((6.9 - len(lab_) * 0.2 - 0.95, 0.42 - j * 0.065 - 0.02), 0.6, 0.04,
                                   color=col, clip_on=False))
        a3.set_title("Precipitation vs evaporative demand", fontsize=5.6, color=INK2, pad=2, loc="left")
        # f4: population exposure change map
        a4 = _glyph_ax(fig, xs[3], 9.0, tw, 26.0)
        chg0 = np.nan_to_num(F["compound_change"].values)
        pop = np.nan_to_num(F["pop2024"].values)
        dE = (chg0 * pop)[:720].reshape(180, 4, 360, 4).sum((1, 3)) / 1e3
        l1 = land[:720].reshape(180, 4, 360, 4).max((1, 3))
        lat1 = F["latitude"].values[:720].reshape(180, 4).mean(1)
        keep = (lat1 >= -57) & (lat1 <= 84)
        lv, cm2, nm2 = _symlog_binned()
        img = cm2(nm2(np.where(l1 & (dE != 0), dE, np.nan)[keep]))
        img[..., 3] = np.where(l1[keep], 1.0, 0.0)
        img[l1[keep] & ~np.isfinite(np.where(dE != 0, dE, np.nan)[keep])] = mpl.colors.to_rgba("#e3e2dd")
        a4.imshow(img, extent=(-180, 180, -57, 84), aspect="auto", interpolation="nearest")
        a4.set_xlim(-180, 180); a4.set_ylim(-57, 84)
        a4.set_title("Cropland and population", fontsize=5.6, color=INK2, pad=2, loc="left")
        return save(fig, "fig1_framework", "fig1_framework")
