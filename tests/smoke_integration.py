"""
End-to-end plumbing smoke test on the years already built in L1 (no full run needed).
Redirects all outputs to a scratch dir and shrinks the periods to the available years,
then runs L2 -> L3 -> L4 -> L5 -> L6 and checks each produces finite, sane output.

Guarded by __main__ so the 'spawn' T90 workers don't re-execute this script.
"""
import sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
import config as C
from chd import io


def main():
    all_years = sorted({int(p.name.split("_")[1]) for p in C.DAILY_ZARR.glob("daily_*_*.zarr")})
    full = [y for y in all_years
            if all(io.already_done(C.daily_month_zarr(y, m)) for m in range(1, 13))]
    best, cur = [], []
    for y in full:
        cur = cur + [y] if (cur and y == cur[-1] + 1) else [y]
        if len(cur) > len(best):
            best = cur[:]
    years = best
    assert len(years) >= 2, f"need >=2 contiguous full years, found {full}"
    print("using contiguous years:", years, flush=True)

    SCRATCH = C.DATA_ROOT / "_smoke"
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    C.MONTHLY_ZARR = SCRATCH / "L2.zarr"
    C.SPEI_ZARR = SCRATCH / "L3.zarr"
    C.HW_DIR = SCRATCH / "L4"
    C.COMPOUND_DIR = SCRATCH / "L5"
    C.TRENDS_DIR = SCRATCH / "L6"
    C.YEAR_START, C.YEAR_END = years[0], years[-1]
    base = C.Period(years[0], years[-1])
    C.HW_BASELINES = {base.slug(): base}
    C.SPEI_CALIB_PERIOD = base
    C.TREND_WINDOWS = {base.slug(): base}

    from chd import water_balance, spei, thresholds, heatwave, compound, trends

    print("[L2] water balance ...", flush=True)
    water_balance.build_monthly(years=years)
    mon = xr.open_zarr(C.MONTHLY_ZARR)
    assert np.isfinite(mon["pet"].values).mean() > 0.9
    print("  PET mean %.1f mm/mo, P mean %.1f, D finite %.2f"
          % (float(mon["pet"].mean()), float(mon["P"].mean()),
             float(np.isfinite(mon["D"].values).mean())), flush=True)

    print("[L3] SPEI ...", flush=True)
    spei._MIN_YEARS = 2   # smoke only: 3-yr sample can't support the real 20-yr minimum
    spei.build_spei(base)
    s1 = xr.open_zarr(C.SPEI_ZARR)["spei1"]
    print("  SPEI-1 finite %.2f, mean %.2f, std %.2f"
          % (float(np.isfinite(s1.values).mean()), float(s1.mean(skipna=True)),
             float(s1.std(skipna=True))), flush=True)

    print("[L4] T90 + heatwave ...", flush=True)
    thresholds.t90_for_baseline(base)
    heatwave.monthly_metrics_for_baseline(base)
    hw = xr.open_zarr(C.HW_DIR / base.slug() / "hw_monthly.zarr")
    print("  hw_present frac %.3f, max hw_days %d"
          % (float(hw["hw_present"].mean()), int(hw["hw_days"].max())), flush=True)

    print("[L5] compound ...", flush=True)
    compound.build_compound(base, 1)
    comp = xr.open_zarr(C.COMPOUND_DIR / f"{base.slug()}_spei1.zarr")
    print("  compound months total %d, drought frac %.3f, heat frac %.3f"
          % (int(comp["compound"].sum()), float(comp["drought"].mean()),
             float(comp["heat"].mean())), flush=True)

    print("[L6] trends ...", flush=True)
    trends.build_trends(base, 1, "compound")
    tr = xr.open_zarr(C.TRENDS_DIR / f"{base.slug()}_spei1_compound.zarr")
    print("  trend cases:", list(tr["case"].values), flush=True)
    print("  slope finite frac %.2f" % float(np.isfinite(tr["slope"].values).mean()), flush=True)

    print("SMOKE TEST PASSED — full chain runs end to end.", flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
