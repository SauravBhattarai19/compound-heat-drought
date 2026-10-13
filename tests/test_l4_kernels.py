"""Validate the new numba L4 kernels against reference numpy implementations."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from chd import thresholds as th
from chd import heatwave as hw


def test_percentile_kernel_matches_numpy():
    rng = np.random.default_rng(0)
    T, P = 3 * 366, 50            # 3 "years", 50 pixels
    doy = np.tile(np.arange(1, 367), 3)[:T].astype(np.int64)
    tx = rng.normal(20, 6, (T, P)).astype(np.float32)
    window = 5
    widx, counts = th._window_indices(doy, window)
    out = th._percentile_kernel(np.ascontiguousarray(tx), widx, counts, 90.0)
    # reference for a few doys/pixels
    for d in [1, 100, 200, 366]:
        offs = ((np.arange(-window, window + 1) + d - 1) % 366) + 1
        mask = np.isin(doy, offs)
        ref = np.percentile(tx[mask], 90, axis=0)
        assert np.allclose(out[d - 1], ref, atol=1e-3), f"doy {d} mismatch"
    print("percentile kernel matches np.percentile")


def test_hw_kernel_counts_runs():
    # 2 pixels, hand-built hot pattern via tx and a flat threshold
    T = 20
    tx = np.zeros((T, 2), np.float32)
    # pixel 0: hot at idx 5-8 (run 4) and 12-13 (run 2, too short)
    tx[[5, 6, 7, 8], 0] = 10
    tx[[12, 13], 0] = 10
    # pixel 1: hot 0-2 (run 3) and 17-19 (run 3)
    tx[[0, 1, 2], 1] = 10
    tx[[17, 18, 19], 1] = 10
    t90 = np.zeros((366, 2), np.float32) + 5.0   # threshold 5 -> tx=10 is hot
    doy = (np.arange(T) % 366 + 1).astype(np.int64)
    # months: first 10 days month0, next 10 month1
    day_to_mon = np.where(np.arange(T) < 10, 0, 1).astype(np.int64)
    hwd, hwe = hw._monthly_hw_kernel(tx, t90, doy, 3, day_to_mon, 2)
    # pixel0: run 5-8 (len4) -> 4 hw_days, all in month0(<10)? idx5-8 <10 -> month0
    assert hwd[0, 0] == 4 and hwd[1, 0] == 0
    assert hwe[0, 0] == 1 and hwe[1, 0] == 0        # 12-13 too short
    # pixel1: run 0-2 in month0, run 17-19 in month1
    assert hwd[0, 1] == 3 and hwd[1, 1] == 3
    assert hwe[0, 1] == 1 and hwe[1, 1] == 1
    print("heatwave kernel counts runs & months correctly")


if __name__ == "__main__":
    test_percentile_kernel_matches_numpy()
    test_hw_kernel_counts_runs()
    print("L4 kernel tests passed.")
