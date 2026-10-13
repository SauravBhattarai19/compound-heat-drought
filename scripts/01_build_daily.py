#!/usr/bin/env python3
"""
L1 driver: build daily fields as one Zarr store PER MONTH, parallel across months.

Month-level tasks keep per-worker memory ~5 GB so we can run many workers and saturate
the USB read bandwidth (L1 is I/O-bound: ~12 s read vs ~5 s compute per month).

Usage:
    python scripts/01_build_daily.py                     # all months 1950-2025
    python scripts/01_build_daily.py 2003                # one year
    python scripts/01_build_daily.py 2000 2005           # inclusive year range
    python scripts/01_build_daily.py --workers 48 --overwrite
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import daily, io


def _one(ym, overwrite):
    year, month = ym
    t0 = time.time()
    if io.already_done(C.daily_month_zarr(year, month)) and not overwrite:
        return ym, "skip", 0.0
    daily.build_month(year, month, overwrite=overwrite)
    return ym, "done", time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if len(a.years) == 0:
        years = list(range(C.YEAR_START, C.YEAR_END + 1))
    elif len(a.years) == 1:
        years = a.years
    else:
        years = list(range(a.years[0], a.years[1] + 1))
    tasks = [(y, m) for y in years for m in range(1, 13)]

    print(f"L1 build: {len(tasks)} months ({len(years)} yr) -> {C.DAILY_ZARR}  "
          f"({a.workers} workers)", flush=True)
    t0 = time.time()
    done = skipped = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, ym, a.overwrite): ym for ym in tasks}
        for fut in as_completed(futs):
            ym, status, dt = fut.result()
            if status == "done":
                done += 1
            else:
                skipped += 1
            n = done + skipped
            if n % 24 == 0 or n == len(tasks):
                rate = done / (time.time() - t0 + 1e-9)
                eta = (len(tasks) - n) / (rate + 1e-9)
                print(f"  {n}/{len(tasks)} months  ({done} built, {skipped} skip)  "
                      f"{rate*60:.0f} mo/min  ETA {eta/60:.0f} min", flush=True)
    print(f"L1 complete: {done} built, {skipped} skipped in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
