#!/usr/bin/env python3
"""L6 driver: annual + seasonal trends for compound/heat/drought, every
(heatwave baseline, SPEI calibration, window) combination."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import trends

if __name__ == "__main__":
    for hw_name, baseline in C.HW_BASELINES.items():
        for spei_name, spei_calib in C.SPEI_CALIB_PERIODS.items():
            for var in ("compound", "heat", "drought"):
                t0 = time.time()
                p = trends.build_trends(baseline, C.SPEI_COMPOUND_TS, var, spei_calib)
                print(f"[hw={hw_name}/spei_calib={spei_name}/{var}] trends -> {p} "
                      f"in {time.time()-t0:.0f}s", flush=True)
