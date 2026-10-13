"""Agricultural context layers for the food-security analysis, cached under
`<repo>/data_aux/`.

Source: **MIRCA2000** v1.1 (Portmann, Siebert & Doll 2010, doi:10.1029/2008GB003435),
the global data set of monthly irrigated and rainfed crop areas around the year 2000.
We use three small, robust products (fetched once from the Zenodo mirror,
doi:10.5281/zenodo.7422506):

1. **Condensed cropping calendars** (`condensed_cropping_calendars.zip`) -- per calendar
   unit (402 spatial "entities"), per crop class (1-26) and up to 5 sub-crops, the
   growing area (ha) and the start/end month of each cropping period, separately for
   irrigated (CCC-I) and rainfed (CCC-R) crops. Cropping periods may wrap the year
   boundary (e.g. winter wheat Nov->May).
2. **Unit-code grid** (`unit_code_grid.zip`) -- the 5 arc-minute ESRI-ascii grid mapping
   each cell to its calendar unit, so the per-unit calendars can be rasterized.
3. **Maximum cropped-area grids** (`maximum_cropped_area_grid.zip`) -- cell-resolved 5
   arc-minute maximum *monthly* cropped area (ha) for irrigated (IRC), rainfed (RFC) and
   total crops (the peak-month growing area, MIRCA's "MMCA"), i.e. the cropland-extent /
   exposure weight.

Two derived, ERA5-gridded products are cached:

* `growing_season_mask.nc` -- per-pixel, per-calendar-month boolean growing-season mask
  (`gs_any`, all crops; `gs_rainfed`, rainfed crops only), built from products 1+2. A
  month is "growing season" at a pixel if that pixel's calendar unit has crop area
  active in that month of at least `GS_FRAC` of the unit's peak monthly crop area (the
  main growing season, robust to negligible shoulder-season slivers). This is the mask
  fed to `trends.annual_counts(..., gs_month_mask=...)` to restrict the compound trend
  to months when crops are actually in the ground.
* `cropland.nc` -- per-pixel cropland-area fraction (`frac_total`/`frac_irrigated`/
  `frac_rainfed`, ha cropped per ha of ERA5 cell), the `rainfed_fraction` of cropland,
  and a boolean `cropland_mask`, built from product 3. Used as the exposure weight and
  the map mask.

Kept out of `viz.py` (which must not grow non-plotting deps); mirrors the compute-vs-
plot split (`drylands.py`, `enso.py`) used across the pipeline. Regridding from the 5
arc-minute native grid to the ERA5 0.25deg grid is nearest-neighbour for the categorical
growing-season mask, and fraction-based (convert extensive ha to an intensive per-cell
fraction, then nearest-lookup) for the cropland weights. The 5' and 0.25deg grids are
offset by half a 5' cell, so the nearest-lookup is approximate, but it agrees with an exact
1-D-overlap conservative regridding to <0.3% on global cropland totals (verified).
"""
from __future__ import annotations
import gzip
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

DATA_AUX = Path(__file__).resolve().parents[2] / "data_aux"
MIRCA_DIR = DATA_AUX / "mirca"
GS_MASK_NC = DATA_AUX / "growing_season_mask.nc"
CROPLAND_NC = DATA_AUX / "cropland.nc"

# Zenodo mirror of MIRCA2000 v1.1 (doi:10.5281/zenodo.7422506).
_ZENODO = "https://zenodo.org/api/records/7422506/files/{name}/content"
_MIRCA_ZIPS = {
    "condensed_cropping_calendars.zip": MIRCA_DIR / "condensed_cropping_calendars",
    "unit_code_grid.zip": MIRCA_DIR / "unit_code_grid",
    "maximum_cropped_area_grid.zip": MIRCA_DIR / "maximum_cropped_area_grid",
}

_CCC_IRR = MIRCA_DIR / "condensed_cropping_calendars" / "cropping_calendar_irrigated.txt.gz"
_CCC_RF = MIRCA_DIR / "condensed_cropping_calendars" / "cropping_calendar_rainfed.txt.gz"
_UNIT_ASC = MIRCA_DIR / "unit_code_grid" / "unit_code.asc.gz"
_AREA_TOT = MIRCA_DIR / "maximum_cropped_area_grid" / "MAX_CROPPED_AREA_TOTAL_ANNUAL_HA.ASC.gz"
_AREA_IRC = MIRCA_DIR / "maximum_cropped_area_grid" / "MAX_CROPPED_AREA_IRC_HA.ASC.gz"
_AREA_RFC = MIRCA_DIR / "maximum_cropped_area_grid" / "MAX_CROPPED_AREA_RFC_HA.ASC.gz"

# A month counts as "growing season" if its crop area is at least this fraction of the
# unit's peak monthly crop area (the main growing season). Documented sensitivity knob.
GS_FRAC = 0.5
# A pixel counts as cropland if at least this fraction of the ERA5 cell is cropped.
CROPLAND_FRAC_MIN = 0.01

_EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Fetch-once
# --------------------------------------------------------------------------- #
def _ensure_mirca() -> None:
    """Download + unzip the three MIRCA2000 products used here, once, mirroring
    `drylands._ensure_koppen_tif` / `enso._ensure_roni_txt`."""
    MIRCA_DIR.mkdir(parents=True, exist_ok=True)
    for name, outdir in _MIRCA_ZIPS.items():
        if outdir.exists() and any(outdir.iterdir()):
            continue
        zip_path = MIRCA_DIR / name
        if not zip_path.exists():
            print(f"[agriculture] downloading MIRCA2000 {name} from Zenodo ...", flush=True)
            urllib.request.urlretrieve(_ZENODO.format(name=name), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MIRCA_DIR)
    print("[agriculture] MIRCA2000 products ready.", flush=True)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def _months_active(start: int, end: int) -> list[int]:
    """1-based month list for a cropping period [start..end], wrapping the year
    boundary when start > end (e.g. 11->5 gives Nov,Dec,Jan..May)."""
    if start < 1 or end < 1:
        return []
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 13)) + list(range(1, end + 1))


def _parse_calendar(path: Path) -> dict[int, np.ndarray]:
    """Parse a MIRCA condensed cropping-calendar file into {unit_code: monthly_area[12]},
    the total crop growing area active in each calendar month summed over all crop
    classes and sub-crops (area-weighted, so a crop's months carry its own area). Rows:
    `unit crop n_SC [area start end] x n_SC`; sub-crops with zero area are ignored."""
    monthly: dict[int, np.ndarray] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 3 or not tok[0].lstrip("-").isdigit():
                continue          # skip the header lines (non-numeric first token)
            unit = int(tok[0])
            n_sc = int(tok[2])
            arr = monthly.setdefault(unit, np.zeros(12, dtype="float64"))
            for k in range(n_sc):
                base = 3 + 3 * k
                if base + 2 >= len(tok):
                    break
                area = float(tok[base])
                if area <= 0.0:
                    continue
                start, end = int(tok[base + 1]), int(tok[base + 2])
                for m in _months_active(start, end):
                    arr[m - 1] += area
    return monthly


def _read_esri_asc(path: Path, nodata_to: float = 0.0) -> xr.DataArray:
    """Read a gzip'd ESRI-ascii grid (6-line header, rows North->South, cols West->East)
    into a DataArray with `latitude`/`longitude` cell-centre coords."""
    with gzip.open(path, "rt") as fh:
        hdr = {}
        for _ in range(6):
            key, val = fh.readline().split()
            hdr[key.lower()] = float(val)
        ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
        cs = hdr["cellsize"]
        xll, yll = hdr["xllcorner"], hdr["yllcorner"]
        nodata = hdr.get("nodata_value", np.nan)
        data = pd.read_csv(fh, sep=r"\s+", header=None, dtype="float64").values
    assert data.shape == (nrows, ncols), f"{path.name}: {data.shape} != {(nrows, ncols)}"
    data[data == nodata] = nodata_to
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows) + 0.5) * cs      # ascending (South->North)
    lat = lat[::-1]                                # rows are North->South
    return xr.DataArray(data, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})


def _cell_area_ha(lat: np.ndarray, cs_deg: float) -> np.ndarray:
    """Area (ha) of a `cs_deg` x `cs_deg` cell at each latitude, as a column vector for
    broadcasting over longitude."""
    dphi = np.deg2rad(cs_deg)
    lat_r = np.deg2rad(lat)
    # area = R^2 * dlon * (sin(lat+dlat/2) - sin(lat-dlat/2))
    band = _EARTH_RADIUS_M ** 2 * dphi * (np.sin(lat_r + dphi / 2) - np.sin(lat_r - dphi / 2))
    return (np.abs(band) / 1.0e4)[:, None]         # m^2 -> ha


# --------------------------------------------------------------------------- #
# Derived ERA5-gridded products
# --------------------------------------------------------------------------- #
def growing_season_mask(like: xr.DataArray, gs_frac: float = GS_FRAC,
                        recompute: bool = False) -> xr.Dataset:
    """Per-pixel, per-calendar-month growing-season mask on `like`'s (ERA5) grid.

    Returns a Dataset with boolean vars `gs_any` (all crops) and `gs_rainfed` (rainfed
    crops only), each dims `(month=1..12, latitude, longitude)`. A month is True at a
    pixel when that pixel's MIRCA calendar unit has crop area active in that month of at
    least `gs_frac` x the unit's peak monthly crop area. Ocean / no-crop pixels are all
    False. Cached to data_aux keyed to the ERA5 grid."""
    if GS_MASK_NC.exists() and not recompute:
        ds = xr.open_dataset(GS_MASK_NC)
        if (ds.sizes.get("latitude") == like.sizes["latitude"]
                and ds.sizes.get("longitude") == like.sizes["longitude"]):
            return ds
    _ensure_mirca()
    monthly_any = _parse_calendar(_CCC_IRR)
    monthly_rf = _parse_calendar(_CCC_RF)
    # merge irrigated into the "any" profile
    for unit, arr in monthly_rf.items():
        monthly_any.setdefault(unit, np.zeros(12, "float64"))
        monthly_any[unit] = monthly_any[unit] + arr

    unit_grid = _read_esri_asc(_UNIT_ASC, nodata_to=-9999).astype("int32")

    def _season_table(monthly: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        out = {}
        for unit, arr in monthly.items():
            peak = arr.max()
            out[unit] = (arr >= gs_frac * peak) if peak > 0 else np.zeros(12, bool)
        return out

    tab_any = _season_table(monthly_any)
    tab_rf = _season_table(monthly_rf)

    ug = unit_grid.values
    uniq = np.unique(ug)
    # month-major lookup: for each month, map unit-code -> bool via a dict-backed vector
    out_any, out_rf = [], []
    for m in range(12):
        look_any = np.zeros(int(uniq.max()) + 1, bool)
        look_rf = np.zeros(int(uniq.max()) + 1, bool)
        for u in uniq:
            if u < 0:
                continue
            if u in tab_any:
                look_any[u] = tab_any[u][m]
            if u in tab_rf:
                look_rf[u] = tab_rf[u][m]
        safe = np.where(ug < 0, 0, ug)
        sea_any = xr.DataArray(look_any[safe], dims=unit_grid.dims, coords=unit_grid.coords)
        sea_rf = xr.DataArray(look_rf[safe], dims=unit_grid.dims, coords=unit_grid.coords)
        out_any.append(sea_any.reindex(latitude=like["latitude"], longitude=like["longitude"],
                                       method="nearest"))
        out_rf.append(sea_rf.reindex(latitude=like["latitude"], longitude=like["longitude"],
                                     method="nearest"))
    gs_any = xr.concat(out_any, dim="month").assign_coords(month=np.arange(1, 13))
    gs_rf = xr.concat(out_rf, dim="month").assign_coords(month=np.arange(1, 13))
    out = xr.Dataset({"gs_any": gs_any.astype("bool"), "gs_rainfed": gs_rf.astype("bool")})
    out.attrs.update(source="MIRCA2000 v1.1 condensed cropping calendars (Portmann et al. 2010)",
                     gs_frac=gs_frac,
                     definition="month growing if unit crop area >= gs_frac * peak monthly area")
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(GS_MASK_NC)
    return out


def cropland_layers(like: xr.DataArray, recompute: bool = False) -> xr.Dataset:
    """Per-pixel cropland exposure layers on `like`'s (ERA5) grid, from the MIRCA2000
    maximum cropped-area grids. Vars: `frac_total`/`frac_irrigated`/`frac_rainfed`
    (ha cropped per ha of ERA5 cell), `rainfed_fraction` (rainfed / (rainfed+irrigated)
    cropland, NaN where no cropland), and boolean `cropland_mask` (`frac_total` >=
    `CROPLAND_FRAC_MIN`). Cached to data_aux keyed to the ERA5 grid."""
    if CROPLAND_NC.exists() and not recompute:
        ds = xr.open_dataset(CROPLAND_NC)
        if (ds.sizes.get("latitude") == like.sizes["latitude"]
                and ds.sizes.get("longitude") == like.sizes["longitude"]):
            return ds
    _ensure_mirca()
    cs = 0.0833333333333333
    tot = _read_esri_asc(_AREA_TOT, nodata_to=0.0)
    irc = _read_esri_asc(_AREA_IRC, nodata_to=0.0)
    rfc = _read_esri_asc(_AREA_RFC, nodata_to=0.0)
    cell_ha = _cell_area_ha(tot["latitude"].values, cs)          # (nlat, 1)
    # extensive ha -> intensive fraction of cell, then nearest-lookup to ERA5
    frac_tot = (tot / cell_ha).reindex(latitude=like["latitude"], longitude=like["longitude"],
                                       method="nearest")
    frac_irc = (irc / cell_ha).reindex(latitude=like["latitude"], longitude=like["longitude"],
                                       method="nearest")
    frac_rfc = (rfc / cell_ha).reindex(latitude=like["latitude"], longitude=like["longitude"],
                                       method="nearest")
    denom = frac_irc + frac_rfc
    rainfed_fraction = (frac_rfc / denom).where(denom > 0)
    out = xr.Dataset({
        "frac_total": frac_tot.astype("float32"),
        "frac_irrigated": frac_irc.astype("float32"),
        "frac_rainfed": frac_rfc.astype("float32"),
        "rainfed_fraction": rainfed_fraction.astype("float32"),
        "cropland_mask": (frac_tot >= CROPLAND_FRAC_MIN),
    })
    out.attrs.update(source="MIRCA2000 v1.1 maximum monthly cropped area grids (Portmann et al. 2010)",
                     cropland_frac_min=CROPLAND_FRAC_MIN,
                     units="frac_* = peak-month ha cropped per ha of ERA5 cell (0..1)")
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(CROPLAND_NC)
    return out

