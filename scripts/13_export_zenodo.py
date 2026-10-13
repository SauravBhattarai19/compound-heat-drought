#!/usr/bin/env python3
"""Export the derived data behind the manuscript to a Zenodo-ready folder.

Only the paper's single locked definition is exported: HW baseline 1961-1990,
SPEI-1, SPEI calibration 1950-2025 (full record).

    python scripts/13_export_zenodo.py OUTDIR

Writes, under OUTDIR:
  monthly_flags_1950_2025.nc        L5 monthly heat / drought / compound flags (int8)
  trends_{compound,heat,drought}.nc L6 per-pixel Mann-Kendall + Theil-Sen, all 10 cases
  masks_era5_grid.nc                analysis land mask, aridity index + class, Koppen
                                    B/BW masks, dryland-patch ids (one grid, one file)
  ancillary/                        cropland, growing-season and LandScan layers as cached
  tables/                           CSVs behind every number in the text
  figure_source_data/               the small cache every paper figure is drawn from
  robustness/                       sampling (6-h vs 24-h) validation and AI-robustness
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from chd import drylands, manuscript_figures as mf  # noqa: E402

BASE = C.HW_BASELINES["1961_1990"]
CALIB = C.SPEI_CALIB_PERIODS["1950_2025"]
TS = C.SPEI_COMPOUND_TS

GLOBAL_ATTRS = dict(
    definition=("heatwave month: >=1 day inside a run of >=3 consecutive days with daily "
                "Tmax > day-of-year T90 (1961-1990 baseline, +/-15-day window); drought "
                "month: SPEI-1 <= -1.0 (log-logistic, calibrated 1950-2025); compound "
                "month: both in the same calendar month"),
    hw_baseline="1961_1990", spei_timescale=TS, spei_calibration="1950_2025",
    source="Derived from ERA5 (Hersbach et al. 2020), Copernicus Climate Change Service",
    grid="ERA5 0.25 deg regular lat-lon, latitude 90 to -90, longitude -180 to 179.75",
    note="Ocean cells are computed but not meaningful; use analysis_land from masks_era5_grid.nc",
)

FLAG_META = {
    "heat": "month contains >=1 heatwave day",
    "drought": "SPEI-1 <= -1.0",
    "compound": "heat AND drought",
    "heat_only": "heat AND NOT drought",
    "drought_only": "drought AND NOT heat",
}

TREND_META = {
    "slope": ("Theil-Sen slope of annual event-month count", "event-months yr-1 per year"),
    "z": ("Mann-Kendall z statistic (tie-corrected)", "1"),
    "pvalue": ("Mann-Kendall two-sided p-value", "1"),
    "signif": ("1 where p < 0.05", "1"),
}

ZLIB = dict(zlib=True, complevel=4)


def _clean(ds):
    return ds.drop_vars([v for v in ("number",) if v in ds.coords])


def monthly_flags(out: Path) -> Path:
    path = out / "monthly_flags_1950_2025.nc"
    src = _clean(xr.open_zarr(C.compound_zarr(BASE, TS, CALIB)))
    ds = xr.Dataset(coords=src.coords)
    enc = {}
    for v, desc in FLAG_META.items():
        print(f"  L5 {v}", flush=True)
        da = src[v].load()
        da.attrs = dict(long_name=desc, flag_values="0 1", flag_meanings="no yes", units="1",
                        missing="-1 on disk (NaN when decoded): SPEI-1 undefined and the "
                                "flag is not settled by heat status")
        ds[v] = da
        enc[v] = dict(dtype="int8", _FillValue=-1, chunksizes=(120, 91, 180), **ZLIB)
    ds.attrs = dict(title="Monthly heatwave, drought and compound flags, 1950-2025", **GLOBAL_ATTRS)
    ds.to_netcdf(path, encoding=enc)
    return path


def trends(out: Path) -> list[Path]:
    paths = []
    for var in ("compound", "heat", "drought"):
        print(f"  L6 {var}", flush=True)
        src = _clean(xr.open_zarr(C.trends_zarr(BASE, TS, CALIB, var))).load()
        names = list(src.attrs["case_names"])
        ds = src[["slope", "z", "pvalue", "signif"]]
        # integer case index + byte-string names: netCDF4 rejects xarray's object/str dtype
        ds = ds.assign_coords(case_name=("case", np.array(names, dtype="S")))
        ds["case_name"].attrs = dict(long_name="<trend window>__<annual|season>",
                                     note="window 1950_2025 or 1979_2025; season DJF/MAM/JJA/SON "
                                          "counts only that season's months; decode with .astype(str)")
        for v, (ln, u) in TREND_META.items():
            ds[v].attrs = dict(long_name=ln, units=u)
        ds.attrs = dict(title=f"Per-pixel trends in annual {var} months", variable=var,
                        case_names=", ".join(f"{i}={n}" for i, n in enumerate(names)),
                        **GLOBAL_ATTRS)
        enc = {v: dict(**ZLIB) for v in ("slope", "z", "pvalue")}
        enc["signif"] = dict(dtype="int8", **ZLIB)
        p = out / f"trends_{var}.nc"
        ds.to_netcdf(p, encoding=enc)
        paths.append(p)
    return paths


def masks(out: Path) -> Path:
    print("  masks", flush=True)
    ctx = mf._Ctx()
    lat, lon = ctx.lat, ctx.lon
    dims = ("latitude", "longitude")
    ai = ctx.ai
    ai_cls = np.where(np.isfinite(ai), np.digitize(ai, drylands.AI_CLASS_EDGES[1:-1]), -1)
    patches, _ = mf._build_dryland_patches(ctx)
    pid = np.full(ctx.land.shape, -1, dtype=np.int8)
    for i, p in enumerate(patches):
        pid[p["mask"]] = i
    ds = xr.Dataset(
        {
            "analysis_land": (dims, ctx.land.values.astype("int8"), dict(
                long_name="land used for every statistic in the paper (Natural Earth land, "
                          "excluding Antarctica south of 60S)")),
            "aridity_index": (dims, ai.astype("float32"), dict(
                long_name="aridity index P/PET, ERA5 1950-2025 climatological means", units="1")),
            "aridity_class": (dims, ai_cls.astype("int8"), dict(
                long_name="aridity class", flag_values="0 1 2 3 4",
                flag_meanings=" ".join(s.replace(" ", "_") for s in drylands.AI_CLASS_LABELS),
                class_edges=str(drylands.AI_CLASS_EDGES), missing="-1")),
            "koppen_BW": (dims, ctx.desert.astype("int8"), dict(
                long_name="Koppen-Geiger BW (desert), Beck et al. 2018, regridded")),
            "koppen_B": (dims, ctx.arid_all.astype("int8"), dict(
                long_name="Koppen-Geiger B group (desert + steppe), Beck et al. 2018, regridded")),
            "dryland_patch": (dims, pid, dict(
                long_name="dryland patch id (-1 = none)",
                patch_letters=" ".join(p["letter"] for p in patches),
                patch_names="; ".join(p["name"] for p in patches))),
        },
        coords=dict(latitude=lat, longitude=lon),
    )
    ds.attrs = dict(title="Masks and classes on the ERA5 0.25 deg grid", **GLOBAL_ATTRS)
    path = out / "masks_era5_grid.nc"
    ds.to_netcdf(path, encoding={v: dict(**ZLIB) for v in ds.data_vars})
    return path


def copies(out: Path) -> None:
    aux = drylands.DATA_AUX
    (out / "ancillary").mkdir(exist_ok=True)
    for name in ("cropland.nc", "growing_season_mask.nc",
                 "population_landscan_2000.nc", "population_landscan_2024.nc",
                 "population_landscan_mosaic_ts_2000.nc", "population_landscan_mosaic_ts_2024.nc"):
        shutil.copy2(aux / name, out / "ancillary" / name)
    shutil.copytree(C.FIG_DIR / "manuscript" / "tables", out / "tables", dirs_exist_ok=True)
    shutil.copytree(C.FIG_DIR / "manuscript" / "nature" / "cache", out / "figure_source_data",
                    dirs_exist_ok=True)
    chk = ROOT / "analysis" / "robustness"
    rob = out / "robustness"
    rob.mkdir(exist_ok=True)
    for f in ("class_series.csv", "q3_varshare.csv", "q5_ai_robustness.csv"):
        shutil.copy2(chk / f, rob / f)
    for f in ("points.csv", "validation_points.csv", "validation_by_class.csv",
              "validation_agg.csv", "validation_annual.npz"):
        shutil.copy2(ROOT / "analysis" / "sampling_validation" / f, rob / f"sampling_{f}")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[2:] or ["copies", "masks", "trends", "flags"]
    if "copies" in steps:
        copies(out)
    if "masks" in steps:
        masks(out)
    if "trends" in steps:
        trends(out)
    if "flags" in steps:
        monthly_flags(out)
    print("done", flush=True)
