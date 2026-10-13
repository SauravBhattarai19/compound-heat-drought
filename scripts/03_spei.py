#!/usr/bin/env python3
"""L3 driver: SPEI-1/3/6/12 from L2 water balance.

Runs once for the primary full-record calibration and again for each fixed-baseline
sensitivity calibration (mirroring the heatwave baselines) -- see
manuscript/spei_time_reference_explainer.tex for why the calibration-period choice
matters."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import spei

if __name__ == "__main__":
    for name, calib in C.SPEI_CALIB_PERIODS.items():
        t0 = time.time()
        p = spei.build_spei(calib)
        print(f"[calib={name}] L3 written -> {p} in {time.time()-t0:.0f}s", flush=True)
