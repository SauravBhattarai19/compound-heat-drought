"""Population exposure layers (LandScan Global), cached under `<repo>/data_aux/`.

Source: **LandScan Global** (Oak Ridge National Laboratory), 30 arc-second (~1 km)
*ambient* (24-hour average) population counts, CC BY 4.0. Two releases are used:

* LandScan Global 2024 (Lebakula et al. 2025, doi:10.48690/1532445) -- the present-day
  exposure weight and the 1-km density classes;
* LandScan Mosaic Time Series v1.0, 2000 and 2024 layers (Zimmer et al. 2026,
  doi:10.48690/lsm/zew2-1n91) -- the population-change term of the exposure
  decomposition (see `landscan_mosaic_ts`). The standard annual Global releases are not
  differenced, because ORNL does not design them for change analysis.

The GeoTIFFs are fetched through the LandScan portal's download form (it requires an
email; https://landscan.ornl.gov/) and dropped into `data_aux/population/` as
`landscan-global-<year>.tif`. No auto-download here -- the portal issues short-lived
signed URLs per request.

Regridding is exact and conservative. A 0.25deg ERA5 cell is exactly 30 x 30 LandScan
pixels; ERA5 cells are centred on multiples of 0.25deg (edges at +-0.125deg), i.e. offset
by 15 LandScan pixels from the LandScan tiling, so the raster is padded/rolled by 15
pixels and then block-summed. Global totals are preserved to integer precision.

Per ERA5 cell we also split the population by the *1-km* density of the pixel each person
lives in (before aggregation, so the split is not smeared by the 28-km cell):
`pop_urban` (>= 1500 persons km^-2), `pop_intermediate` (300-1500) and `pop_rural`
(< 300). These are the density thresholds of the Degree of Urbanisation (Dijkstra et al.
2021) applied pixel-wise, without its contiguity/cluster-size rules -- a density-class
proxy for urban/rural, not the full DEGURBA classification.

Kept separate from `agriculture.py` (MIRCA2000) and out of `viz.py`, mirroring the
compute-vs-plot split used across the pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_AUX = Path(__file__).resolve().parents[2] / "data_aux"
POP_DIR = DATA_AUX / "population"

_RES = 1.0 / 120.0            # LandScan pixel size (deg)
_BLOCK = 30                   # LandScan pixels per 0.25deg ERA5 cell
_SHIFT = 15                   # half-block offset between ERA5 cell edges and LandScan tiling
URBAN_DENS = 1500.0           # persons km^-2 (Degree of Urbanisation thresholds)
RURAL_DENS = 300.0
_EARTH_RADIUS_KM = 6371.0


def _tif(year: int) -> Path:
    return POP_DIR / f"landscan-global-{year}.tif"


def _cache(year: int) -> Path:
    return DATA_AUX / f"population_landscan_{year}.nc"


def _read_full_globe(year: int) -> tuple[np.ndarray, np.ndarray]:
    """LandScan raster as an int64 array on the full 90N..90S x 180W..180E 30'' grid
    (21600 x 43200, rows N->S), nodata -> 0, plus the pixel-centre latitudes."""
    import rioxarray
    src = _tif(year)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download LandScan Global {year} from https://landscan.ornl.gov/ "
            f"and place the GeoTIFF in {POP_DIR}.")
    da = rioxarray.open_rasterio(src).isel(band=0)
    left, bottom, right, top = da.rio.bounds()
    assert abs(left + 180) < 1e-6 and abs(da.rio.resolution()[0] - _RES) < 1e-9, "unexpected LandScan grid"
    v = da.values.astype(np.int64)
    v[v < 0] = 0
    full = np.zeros((180 * 120, 360 * 120), dtype=np.int64)
    r0 = int(round((90.0 - top) / _RES))              # 2000 release stops at 86N/84S etc.
    full[r0:r0 + v.shape[0], :] = v
    lat_c = 90.0 - (np.arange(full.shape[0]) + 0.5) * _RES
    return full, lat_c


def _block_sum(a: np.ndarray, lon0: float) -> np.ndarray:
    """Sum 30'' pixels into ERA5 0.25deg cells (721 lat, 90..-90; 1440 lon starting at
    cell centre `lon0`, e.g. -180 or 0)."""
    pad = np.zeros((a.shape[0] + 2 * _SHIFT, a.shape[1]), dtype=a.dtype)
    pad[_SHIFT:_SHIFT + a.shape[0]] = a                # rows now start at 90.125N
    # first column must start at the western edge of the first ERA5 cell, lon0 - 0.125deg;
    # the raster's column 0 starts at -180deg
    first = int(round((lon0 - 0.125 + 180.0) / _RES)) % a.shape[1]
    pad = np.roll(pad, -first, axis=1)
    nlat, nlon = pad.shape[0] // _BLOCK, pad.shape[1] // _BLOCK
    return pad.reshape(nlat, _BLOCK, nlon, _BLOCK).sum(axis=(1, 3))


def landscan(like: xr.DataArray, year: int = 2024, recompute: bool = False) -> xr.Dataset:
    """LandScan population on `like`'s ERA5 0.25deg grid (721 x 1440, lat descending,
    lon ascending from -180 or 0). Vars: `pop` (persons per cell), `pop_urban` / `pop_intermediate` /
    `pop_rural` (persons living in 1-km pixels of density >=1500 / 300-1500 / <300
    km^-2), and `pop_density` (persons km^-2 of the whole ERA5 cell). Cached per year."""
    out_nc = _cache(year)
    if out_nc.exists() and not recompute:
        ds = xr.open_dataset(out_nc)
        if (ds.sizes.get("latitude") == like.sizes["latitude"]
                and ds.sizes.get("longitude") == like.sizes["longitude"]):
            return ds
    full, lat_c = _read_full_globe(year)
    # 1-km pixel area (km^2) by row, for the density classes
    dphi = np.deg2rad(_RES)
    pix_km2 = (_EARTH_RADIUS_KM ** 2 * dphi *
               np.abs(np.sin(np.deg2rad(lat_c) + dphi / 2) - np.sin(np.deg2rad(lat_c) - dphi / 2)))
    dens = full / pix_km2[:, None]
    urban = np.where(dens >= URBAN_DENS, full, 0)
    rural = np.where(dens < RURAL_DENS, full, 0)
    del dens
    lon0 = float(like["longitude"].values[0])
    tot = _block_sum(full, lon0)
    urb = _block_sum(urban, lon0); del urban
    rur = _block_sum(rural, lon0); del rural
    assert tot.sum() == full.sum(), "block sum did not preserve the global total"
    lat = 90.0 - 0.25 * np.arange(tot.shape[0])
    lon = lon0 + 0.25 * np.arange(tot.shape[1])
    assert np.allclose(lat, like["latitude"].values) and np.allclose(lon, like["longitude"].values), \
        "ERA5 grid mismatch"
    dphi_c = np.deg2rad(0.25)
    lat_edges_hi = np.clip(np.deg2rad(lat + 0.125), -np.pi / 2, np.pi / 2)
    lat_edges_lo = np.clip(np.deg2rad(lat - 0.125), -np.pi / 2, np.pi / 2)
    cell_km2 = _EARTH_RADIUS_KM ** 2 * dphi_c * (np.sin(lat_edges_hi) - np.sin(lat_edges_lo))
    coords = {"latitude": lat, "longitude": lon}
    dims = ("latitude", "longitude")
    ds = xr.Dataset({
        "pop": (dims, tot.astype("float64")),
        "pop_urban": (dims, urb.astype("float64")),
        "pop_intermediate": (dims, (tot - urb - rur).astype("float64")),
        "pop_rural": (dims, rur.astype("float64")),
        "pop_density": (dims, (tot / cell_km2[:, None]).astype("float32")),
    }, coords=coords)
    ds.attrs.update(source=f"LandScan Global {year} (ORNL), 30 arc-second ambient population",
                    regrid="exact conservative 30x30 block sum onto ERA5 0.25deg cells",
                    density_classes=f"urban >= {URBAN_DENS:g}, rural < {RURAL_DENS:g} persons km^-2 "
                                    "at the 1-km pixel (Degree of Urbanisation thresholds, no contiguity rule)",
                    global_total=float(tot.sum()))
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    return ds


# --------------------------------------------------------------------------- #
# LandScan Mosaic Time Series (temporally consistent; used for 2000 -> 2024 change)
# --------------------------------------------------------------------------- #
# The standard annual LandScan Global releases are not designed for change detection
# (their methods and inputs change between releases). The LandScan Mosaic Time Series
# v1.0 (Zimmer et al. 2026, doi:10.48690/lsm/zew2-1n91) is: every year 1975-2023 is
# backcast from the 2024 benchmark surface with consistent national control totals.
# 3 arc-second (~90 m) float32 COGs, ~290 GB uncompressed per year -> streamed in
# ERA5-row strips. 3'' = 1/1200 deg, so ERA5 cell edges (odd multiples of 0.125 deg)
# fall on pixel boundaries (150 px) and the aggregation is exact.
MOSAIC_DIR = POP_DIR / "mosaic_ts"
_M = 1200                      # 3'' pixels per degree
_MB = 300                      # 3'' pixels per 0.25 deg


def _mosaic_tif(year: int) -> Path:
    return MOSAIC_DIR / f"landscan_mosaic_timeseries_{year}.tif"


def _mosaic_strip(args):
    """Block-sum the pixel rows that fall in ERA5 latitude row `i` into 1440 cells."""
    import rasterio
    from rasterio.windows import Window
    path, i, r0, r1, col_off = args
    with rasterio.open(path) as src:
        a = src.read(1, window=Window(0, r0, src.width, r1 - r0)).astype(np.float64)
    a[~np.isfinite(a) | (a < 0)] = 0.0
    s = a.sum(axis=0)
    buf = np.zeros(1441 * _MB)
    buf[col_off:col_off + s.size] = s
    cells = buf.reshape(1441, _MB).sum(axis=1)
    cells[0] += cells[1440]           # lon >= 179.875 wraps into the -180 cell
    return i, cells[:1440]


def landscan_mosaic_ts(like: xr.DataArray, year: int, recompute: bool = False,
                       workers: int = 16) -> xr.DataArray:
    """LandScan Mosaic Time Series population (persons per ERA5 cell) for `year`,
    aggregated exactly onto `like`'s grid (lat 90..-90, lon from -180). Cached."""
    out_nc = DATA_AUX / f"population_landscan_mosaic_ts_{year}.nc"
    if out_nc.exists() and not recompute:
        return xr.open_dataarray(out_nc)
    import rasterio
    from concurrent.futures import ProcessPoolExecutor
    lon0 = float(like["longitude"].values[0])
    assert abs(lon0 + 180.0) < 1e-9, "mosaic aggregation assumes a -180-start grid"
    path = _mosaic_tif(year)
    with rasterio.open(path) as src:
        H, W = src.shape
        top, left = src.bounds.top, src.bounds.left
        assert abs(src.res[0] * _M - 1) < 1e-9
    # pixel-row offset of the raster top from the northern edge of ERA5 row 0 (90.125N)
    u0 = int(round((90.125 - top) * _M))
    col_off = int(round((left - (lon0 - 0.125)) * _M))   # >= 0 by construction
    assert col_off >= 0 and col_off + W <= 1441 * _MB
    tasks = []
    for i in range(like.sizes["latitude"]):
        r0 = max(i * _MB - u0, 0)
        r1 = min((i + 1) * _MB - u0, H)
        if r1 > r0:
            tasks.append((str(path), i, r0, r1, col_off))
    grid = np.zeros((like.sizes["latitude"], 1440))
    with ProcessPoolExecutor(workers) as ex:
        for i, cells in ex.map(_mosaic_strip, tasks, chunksize=1):
            grid[i] = cells
    da = xr.DataArray(grid, dims=("latitude", "longitude"),
                      coords={"latitude": like["latitude"].values,
                              "longitude": like["longitude"].values}, name="pop")
    da.attrs.update(source=f"LandScan Mosaic Time Series v1.0, {year} layer (ORNL; doi:10.48690/lsm/zew2-1n91)",
                    regrid="exact block sum of 3'' pixels into ERA5 0.25deg cells",
                    global_total=float(grid.sum()))
    da.to_netcdf(out_nc)
    return da
