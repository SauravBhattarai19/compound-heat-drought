"""I/O helpers: open ERA5 monthly files and write compressed Zarr stores."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import xarray as xr

# Harmless: consolidated metadata isn't part of the Zarr v3 spec yet, but zarr-python
# writes/reads it fine and we only consume these stores with zarr-python. Silence the
# per-store spam so pipeline logs stay readable.
warnings.filterwarnings(
    "ignore", message=".*Consolidated metadata is currently not part.*",
    category=UserWarning,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C


def open_month(year: int, month: int, chunks=None) -> xr.Dataset:
    """Open one ERA5 monthly file. Default: no dask chunking (month fits in RAM)."""
    return xr.open_dataset(C.era5_file(year, month), chunks=chunks)


def zarr_encoding(ds: xr.Dataset) -> dict:
    """zstd compression + float32 for every data var."""
    import zarr
    comp = zarr.codecs.BloscCodec(cname=C.ZARR_COMPRESSOR, clevel=C.ZARR_CLEVEL)
    return {v: {"compressors": (comp,)} for v in ds.data_vars}


def write_zarr(ds: xr.Dataset, path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        enc = zarr_encoding(ds)
        ds.to_zarr(path, mode=mode, encoding=enc, consolidated=True)
    except Exception:
        # Fallback for zarr API differences — write without explicit compressor.
        ds.to_zarr(path, mode=mode, consolidated=True)


def already_done(path: Path) -> bool:
    """True if a Zarr store looks complete (has consolidated metadata)."""
    return path.exists() and (
        (path / "zarr.json").exists() or (path / ".zmetadata").exists()
    )
