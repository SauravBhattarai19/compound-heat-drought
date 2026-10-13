#!/usr/bin/env python3
"""Nature-style figure suite for manuscript_heat_spei1 (see src/chd/nature_figures.py).

    python scripts/12_nature_figures.py --cache          # (re)build the data cache (slow disk read)
    python scripts/12_nature_figures.py                  # draw every figure from the cache
    python scripts/12_nature_figures.py fig2 fig5        # draw selected figures only
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chd import nature_figures as nf

if __name__ == "__main__":
    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--cache" in sys.argv:
        nf.build_cache(force="--force" in sys.argv)
    if not ("--cache" in sys.argv and not args):
        for p in nf.build_figures(args or None):
            print(f"  wrote {p}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
