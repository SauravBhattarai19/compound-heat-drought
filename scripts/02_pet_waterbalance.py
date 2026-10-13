#!/usr/bin/env python3
"""L2 driver: monthly PET, P, D from L1 daily fields (threaded dask scheduler)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dask
import config as C
from chd import water_balance

if __name__ == "__main__":
    # Threaded scheduler streams chunk-by-chunk; cap threads to stay HDD-friendly.
    dask.config.set(scheduler="threads", num_workers=C.N_WORKERS)
    t0 = time.time()
    p = water_balance.build_monthly()
    print(f"L2 written -> {p} in {time.time()-t0:.0f}s", flush=True)
