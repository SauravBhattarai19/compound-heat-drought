#!/usr/bin/env python3
"""Manuscript figure driver -- the SINGLE locked compound definition
(HW baseline 1961-1990, SPEI-1, SPEI calibration 1950-2025).

Produces the paper figure suite under figures/manuscript/.
Use --regional to rebuild only the heat, drought, compound annual panels and
compound change-point boxplots; --all builds the extended figure suite.

Dryland context (aridity index from L2 P/PET, Köppen-Geiger desert mask) is cached under
data_aux/ on first run; the Köppen source GeoTIFF is fetched from figshare if absent.
No sensitivity matrix here -- that is scripts/09,10 (S1-S12).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chd import manuscript_figures as mf

if __name__ == "__main__":
    t0 = time.time()
    full = "--all" in sys.argv
    if "--regional" in sys.argv:
        ctx = mf._Ctx()
        paths = [mf.heat_regional(ctx), mf.drought_regional(ctx),
                 mf.compound_regional(ctx), mf.compound_regional_boxplot(ctx, split="cp")]
    else:
        paths = mf.build_all() if full else mf.build_paper()
    for p in paths:
        print(f"  wrote {p}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
