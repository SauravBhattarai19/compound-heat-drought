#!/usr/bin/env python3
"""L5 driver: monthly compound events for every (heatwave baseline, SPEI calibration)
combination (SPEI-1). The SPEI-calibration axis is the fixed-baseline sensitivity run
mirroring HW_BASELINES -- see manuscript/spei_time_reference_explainer.tex."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import compound

if __name__ == "__main__":
    for hw_name, baseline in C.HW_BASELINES.items():
        for spei_name, spei_calib in C.SPEI_CALIB_PERIODS.items():
            t0 = time.time()
            p = compound.build_compound(baseline, C.SPEI_COMPOUND_TS, spei_calib)
            print(f"[hw={hw_name}/spei_calib={spei_name}] compound -> {p} "
                  f"in {time.time()-t0:.0f}s", flush=True)
