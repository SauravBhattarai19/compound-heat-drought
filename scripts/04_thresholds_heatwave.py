#!/usr/bin/env python3
"""L4 driver: T90 thresholds + monthly heatwave metrics for every baseline."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import thresholds, heatwave

if __name__ == "__main__":
    for name, baseline in C.HW_BASELINES.items():
        t0 = time.time()
        print(f"[{name}] T90 ...", flush=True)
        thresholds.t90_for_baseline(baseline)
        print(f"[{name}] heatwave metrics ...", flush=True)
        heatwave.monthly_metrics_for_baseline(baseline)
        print(f"[{name}] done in {time.time()-t0:.0f}s", flush=True)
