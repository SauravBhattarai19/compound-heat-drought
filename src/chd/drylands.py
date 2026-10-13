"""Dryland / aridity helpers for the manuscript figures.

Two independent desert delineations, both cached under `<repo>/data_aux/`:

1. **Aridity Index** AI = P/PET (UNEP), derived from our own L2 monthly P & PET so the
   desert mask is internally consistent with what SPEI actually saw (SPEI goes
   degenerate in hyper-arid cells -- see chd.spei). Classes: hyper-arid <0.05,
   arid <0.2, semi-arid <0.5, dry sub-humid <0.65, humid otherwise.
2. **Köppen-Geiger** present-day map (Beck et al. 2018, 0.083deg), regridded to the
   ERA5 grid, giving recognizable named-desert (BW) / steppe (BS) boundaries. The
   source GeoTIFF is fetched once from figshare if absent.

Kept out of `viz.py` (which must not grow non-plotting deps); mirrors the
compute-vs-plot split used across the pipeline.
"""
from __future__ import annotations
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

DATA_AUX = Path(__file__).resolve().parents[2] / "data_aux"
ARIDITY_NC = DATA_AUX / "aridity_index.nc"
KOPPEN_DIR = DATA_AUX / "koppen"
KOPPEN_TIF = KOPPEN_DIR / "Beck_KG_V1_present_0p083.tif"
KOPPEN_MASK_NC = DATA_AUX / "koppen_desert_mask.nc"
_KOPPEN_ZIP_URL = "https://ndownloader.figshare.com/files/12407516"   # Beck_KG_V1.zip

# UNEP aridity-index class edges and labels (driest -> wettest).
AI_CLASS_EDGES = [0.0, 0.05, 0.2, 0.5, 0.65, np.inf]
AI_CLASS_LABELS = ["hyper-arid", "arid", "semi-arid", "dry sub-humid", "humid"]

# Beck Köppen-Geiger V1 integer codes for the B (dry) group.
_KG_DESERT = (4, 5)          # BWh, BWk
_KG_STEPPE = (6, 7)          # BSh, BSk


def aridity_index(recompute: bool = False) -> xr.DataArray:
    """Long-term Aridity Index AI = sum(P)/sum(PET) over the full L2 record (mm/month
    cancels, so this is the ratio of climatological means). Cached to data_aux."""
    if ARIDITY_NC.exists() and not recompute:
        return xr.open_dataarray(ARIDITY_NC)
    ds = xr.open_zarr(C.MONTHLY_ZARR)
    with np.errstate(divide="ignore", invalid="ignore"):
        ai = (ds["P"].sum("time") / ds["pet"].sum("time")).astype("float32")
    ai.name = "aridity_index"
    ai.attrs.update(long_name="Aridity Index P/PET (UNEP)",
                    classes="hyper-arid<0.05 arid<0.2 semi-arid<0.5 dry-subhumid<0.65 humid")
    ai = ai.compute()
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    ai.to_netcdf(ARIDITY_NC)
    return ai


def aridity_class(like: xr.DataArray | None = None, recompute: bool = False) -> xr.DataArray:
    """Integer aridity class 0..4 (hyper-arid..humid) on the AI grid (or `like`'s grid
    if given, via nearest lookup). NaN where AI undefined."""
    ai = aridity_index(recompute=recompute)
    if like is not None:
        ai = ai.reindex(latitude=like["latitude"], longitude=like["longitude"], method="nearest")
    cls = xr.apply_ufunc(np.digitize, ai, kwargs={"bins": AI_CLASS_EDGES[1:-1]})
    return cls.where(np.isfinite(ai))


def _ensure_koppen_tif() -> None:
    if KOPPEN_TIF.exists():
        return
    KOPPEN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = KOPPEN_DIR / "Beck_KG_V1.zip"
    print(f"[drylands] downloading Köppen-Geiger from figshare -> {zip_path} ...", flush=True)
    urllib.request.urlretrieve(_KOPPEN_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(KOPPEN_TIF.name, KOPPEN_DIR)
    zip_path.unlink()
    print("[drylands] Köppen tif ready.", flush=True)


def koppen_desert_mask(like: xr.DataArray, recompute: bool = False) -> xr.Dataset:
    """Boolean desert / steppe / B-group masks on `like`'s (ERA5) grid, nearest-
    regridded from the 0.083deg Köppen map. Cached to data_aux keyed to the ERA5 grid."""
    if KOPPEN_MASK_NC.exists() and not recompute:
        ds = xr.open_dataset(KOPPEN_MASK_NC)
        if (ds.sizes.get("latitude") == like.sizes["latitude"]
                and ds.sizes.get("longitude") == like.sizes["longitude"]):
            return ds
    import rioxarray  # local import: only needed when (re)building the mask
    _ensure_koppen_tif()
    kg = rioxarray.open_rasterio(KOPPEN_TIF).isel(band=0)
    kg = kg.rename({"x": "longitude", "y": "latitude"}).sortby("longitude").sortby("latitude")
    kg_t = kg.reindex(latitude=like["latitude"], longitude=like["longitude"],
                      method="nearest").astype("int16")
    desert = kg_t.isin(_KG_DESERT)
    steppe = kg_t.isin(_KG_STEPPE)
    out = xr.Dataset({
        "kg_code": kg_t,
        "desert": desert,
        "steppe": steppe,
        "arid_all": desert | steppe,
    })
    out.attrs.update(source="Beck et al. 2018 Köppen-Geiger V1 present 0.083deg",
                     desert_codes="BWh=4,BWk=5", steppe_codes="BSh=6,BSk=7")
    DATA_AUX.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(KOPPEN_MASK_NC)
    return out
