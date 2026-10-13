"""Validate chd.leading's onset-order classification (_leading_kernel + classify() +
the annual/total/dominance aggregations), mirroring test_events.py's style: plain
assert + print, no pytest, hand-built synthetic (T,P) arrays with known expected
labels.

This module classifies each compound episode as heat-leading / drought-leading /
concurrent / ambiguous by looking at the month immediately before onset. It is the
direct analogue of the hazard-sequencing classification in Kim et al. (2026, Sci.
Adv.), a paper now under an Editorial Expression of Concern after the authors
self-reported a coding error that silently mishandled missing data during this exact
kind of processing step and skewed their drought-leading trend. This module's own
docstring documents that an earlier version of THIS kernel had the same class of bug
(undefined/NaN drought status silently falling through to "confirmed not drought" or
being routed to AMBIGUOUS only when heat was also absent, inflating HEAT_LEAD). The
most important test below (test_ambiguous_heat_present_is_not_heat_lead) is a direct
regression test for that exact fixed bug: it is the one case where the old, buggy
asymmetric version of the kernel would have produced a different (wrong) answer than
the current symmetric version.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chd import leading as ld

NAN = np.nan


def _run(heat_row, drought_row):
    """Run the kernel on a single pixel (1-D heat/drought lists) and return the 1-D
    label array. Keeps every individual test terse and focused on one timeline."""
    heat = np.asarray(heat_row, dtype=np.uint8).reshape(-1, 1)
    drought = np.asarray(drought_row, dtype=np.float32).reshape(-1, 1)
    return ld._leading_kernel(heat, drought)[:, 0]


def test_heat_lead_basic():
    # t1: heat-only (precedes onset). t2-3: compound, onset looks back at t1
    # (heat present, drought confirmed absent) -> HEAT_LEAD, and the label persists
    # across the whole 2-month run.
    out = _run(heat_row=[0, 1, 1, 1], drought_row=[0, 0, 1, 1])
    expect = [0, 0, ld.HEAT_LEAD, ld.HEAT_LEAD]
    assert list(out) == expect, f"heat-lead basic: got {list(out)}, expected {expect}"
    print("heat-lead basic case correct")


def test_drought_lead_basic():
    out = _run(heat_row=[0, 0, 1, 1], drought_row=[0, 1, 1, 1])
    expect = [0, 0, ld.DROUGHT_LEAD, ld.DROUGHT_LEAD]
    assert list(out) == expect, f"drought-lead basic: got {list(out)}, expected {expect}"
    print("drought-lead basic case correct")


def test_concurrent_both_confirmed_absent():
    out = _run(heat_row=[0, 0, 1], drought_row=[0, 0, 1])
    expect = [0, 0, ld.CONCURRENT]
    assert list(out) == expect, f"concurrent: got {list(out)}, expected {expect}"
    print("concurrent (both confirmed absent at t-1) case correct")


def test_episode_onset_at_t_zero_is_concurrent():
    # No t-1 exists at all -> defined as CONCURRENT by convention, not an error.
    out = _run(heat_row=[1, 1], drought_row=[1, 1])
    expect = [ld.CONCURRENT, ld.CONCURRENT]
    assert list(out) == expect, f"t=0 onset: got {list(out)}, expected {expect}"
    print("episode starting at t=0 (no t-1) correctly labeled concurrent")


def test_ambiguous_heat_absent_at_t_minus_1():
    # heat absent + drought unknown at t-1 -> AMBIGUOUS. This case alone would NOT
    # have caught the historical bug (the old buggy code got this branch right; it
    # only mishandled the heat-PRESENT+drought-unknown case below).
    out = _run(heat_row=[0, 1], drought_row=[NAN, 1])
    expect = [0, ld.AMBIGUOUS]
    assert list(out) == expect, f"ambiguous (heat absent): got {list(out)}, expected {expect}"
    print("ambiguous case with heat absent at t-1 correct")


def test_ambiguous_heat_present_is_not_heat_lead():
    """THE regression test. t0: heat present, drought status UNDEFINED (SPEI
    degenerate fit) -> NOT compound at t0 (drought[0]==1.0 is False for NaN), so t0
    itself is correctly unlabeled. t1: compound onset; looking back at t0, heat was
    present but drought is unknown, so the true leading hazard is genuinely
    undetermined -- this MUST be AMBIGUOUS.

    The bug this guards against: an earlier version of _leading_kernel only routed
    the undefined-drought-at-t-1 case to AMBIGUOUS when heat was ALSO absent at t-1;
    when heat was present and drought was merely unknown, it fell through and was
    mislabeled HEAT_LEAD -- silently inflating the heat-leading share, exactly the
    kind of missing-data mishandling that produced Kim et al. (2026)'s error (their
    self-reported bug distorted the drought-leading trend specifically). If this
    kernel ever regresses to that behavior, this assertion fails."""
    out = _run(heat_row=[1, 1], drought_row=[NAN, 1])
    expect = [0, ld.AMBIGUOUS]
    assert list(out) == expect, (
        f"ambiguous (heat PRESENT, drought unknown) at t-1: got {list(out)}, "
        f"expected {expect} -- got HEAT_LEAD={ld.HEAT_LEAD} instead of AMBIGUOUS="
        f"{ld.AMBIGUOUS} would mean the old NaN-as-false-drought bug has returned")
    print("REGRESSION GUARD PASSED: heat-present + drought-unknown at t-1 correctly "
          "stays ambiguous, not silently misclassified as heat-leading")


def test_noncompound_months_confirmed_not_drought_stay_zero():
    # heat present throughout but drought is confirmed (not NaN) absent throughout --
    # must never be compound, must never be misread as "unknown" and turned ambiguous.
    out = _run(heat_row=[1, 1, 1, 1], drought_row=[0, 0, 0, 0])
    expect = [0, 0, 0, 0]
    assert list(out) == expect, f"never-compound pixel: got {list(out)}, expected {expect}"
    print("confirmed-absent (non-NaN) drought never spuriously classified")


def test_episode_persists_uniform_label_across_run():
    out = _run(heat_row=[0, 1, 1, 1, 1, 0], drought_row=[0, 0, 1, 1, 1, 1])
    expect = [0, 0, ld.HEAT_LEAD, ld.HEAT_LEAD, ld.HEAT_LEAD, 0]
    assert list(out) == expect, f"episode persistence: got {list(out)}, expected {expect}"
    print("multi-month episode carries one uniform onset label throughout its run")


def test_two_episodes_separated_by_gap_classified_independently():
    # Episode 1 (t1-2): drought-leading. A 2-month gap (heat-only, not compound).
    # Episode 2 (t5): heat-leading. Episode 2 must NOT inherit episode 1's label.
    out = _run(
        heat_row=[0, 1, 1, 1, 1, 1],
        drought_row=[1, 1, 1, 0, 0, 1],
    )
    expect = [0, ld.DROUGHT_LEAD, ld.DROUGHT_LEAD, 0, 0, ld.HEAT_LEAD]
    assert list(out) == expect, f"two independent episodes: got {list(out)}, expected {expect}"
    print("two episodes separated by a gap are classified independently (no label leakage)")


def test_multi_pixel_prange_independence():
    """Run several of the scenarios above as columns of ONE (T,P) array through a
    single kernel call, to exercise numba's parallel (prange) pixel loop and confirm
    there is no cross-pixel contamination -- a different risk surface than any
    single-pixel test above."""
    T = 6
    P = 4
    heat = np.zeros((T, P), dtype=np.uint8)
    drought = np.zeros((T, P), dtype=np.float32)

    # col 0: heat-lead (pad tail with non-compound zeros)
    heat[:, 0] = [0, 1, 1, 1, 0, 0]
    drought[:, 0] = [0, 0, 1, 1, 0, 0]
    # col 1: drought-lead
    heat[:, 1] = [0, 0, 1, 1, 0, 0]
    drought[:, 1] = [0, 1, 1, 1, 0, 0]
    # col 2: the regression case (ambiguous, heat present at t-1)
    heat[:, 2] = [1, 1, 0, 0, 0, 0]
    drought[:, 2] = [NAN, 1, 0, 0, 0, 0]
    # col 3: two independent episodes separated by a gap
    heat[:, 3] = [0, 1, 1, 1, 1, 1]
    drought[:, 3] = [1, 1, 1, 0, 0, 1]

    out = ld._leading_kernel(heat, drought)

    expect = np.zeros((T, P), dtype=np.int8)
    expect[:, 0] = [0, 0, ld.HEAT_LEAD, ld.HEAT_LEAD, 0, 0]
    expect[:, 1] = [0, 0, ld.DROUGHT_LEAD, ld.DROUGHT_LEAD, 0, 0]
    expect[:, 2] = [0, ld.AMBIGUOUS, 0, 0, 0, 0]
    expect[:, 3] = [0, ld.DROUGHT_LEAD, ld.DROUGHT_LEAD, 0, 0, ld.HEAT_LEAD]

    assert np.array_equal(out, expect), (
        f"multi-pixel independence failed:\ngot=\n{out}\nexpected=\n{expect}")
    print("multi-pixel (prange) classification matches per-pixel results exactly, "
          "no cross-pixel contamination")


def test_classify_wrapper_preserves_coords_and_matches_kernel():
    """Integration check on the classify() xarray wrapper: dtype casting, xr.align,
    and the reshape((T,-1)) <-> reshape(heat.shape) round trip must reproduce exactly
    the same labels as calling the kernel directly, and must preserve the input
    coordinates/dims (a wiring bug here -- e.g. a dim-order mismatch between heat and
    drought -- would silently scramble which label lands on which pixel without
    necessarily crashing)."""
    times = pd.date_range("2000-11-01", periods=6, freq="MS")
    lat = np.array([10.0])
    lon = np.array([0.0, 10.0, 20.0])

    heat_vals = np.zeros((6, 1, 3), dtype=np.uint8)
    drought_vals = np.zeros((6, 1, 3), dtype=np.float32)
    # pixel (0,0): heat-lead. pixel (0,1): drought-lead.
    heat_vals[:, 0, 0] = [0, 1, 1, 1, 0, 0]
    drought_vals[:, 0, 0] = [0, 0, 1, 1, 0, 0]
    heat_vals[:, 0, 1] = [0, 0, 1, 1, 0, 0]
    drought_vals[:, 0, 1] = [0, 1, 1, 1, 0, 0]
    # pixel (0,2): heat-lead episode at t2, gap, ambiguous episode at t4-5 (index
    # 2001-01=t2, 2001-03=t4) -- also exercises annual_type_counts' year-boundary
    # grouping later since the episode(s) straddle the 2000/2001 boundary.
    heat_vals[:, 0, 2] = [0, 1, 1, 0, 1, 1]
    drought_vals[:, 0, 2] = [0, 0, 1, NAN, 1, 1]

    heat = xr.DataArray(heat_vals, dims=("time", "latitude", "longitude"),
                         coords={"time": times, "latitude": lat, "longitude": lon})
    drought = xr.DataArray(drought_vals, dims=("time", "latitude", "longitude"),
                            coords={"time": times, "latitude": lat, "longitude": lon})

    cat = ld.classify(heat, drought)

    assert cat.dims == heat.dims, f"dims changed: {cat.dims} vs {heat.dims}"
    assert cat.shape == heat.shape, f"shape changed: {cat.shape} vs {heat.shape}"
    np.testing.assert_array_equal(cat["latitude"].values, lat)
    np.testing.assert_array_equal(cat["longitude"].values, lon)
    np.testing.assert_array_equal(cat["time"].values, times.values)

    expect0 = [0, 0, ld.HEAT_LEAD, ld.HEAT_LEAD, 0, 0]
    expect1 = [0, 0, ld.DROUGHT_LEAD, ld.DROUGHT_LEAD, 0, 0]
    expect2 = [0, 0, ld.HEAT_LEAD, 0, ld.AMBIGUOUS, ld.AMBIGUOUS]
    assert list(cat.values[:, 0, 0]) == expect0, f"pixel0: got {list(cat.values[:,0,0])}"
    assert list(cat.values[:, 0, 1]) == expect1, f"pixel1: got {list(cat.values[:,0,1])}"
    assert list(cat.values[:, 0, 2]) == expect2, f"pixel2: got {list(cat.values[:,0,2])}"
    print("classify() wrapper preserves coords/dims and matches direct-kernel labels")
    return cat


def test_annual_total_and_dominance_aggregations():
    """Uses the same 3-pixel DataArray as the classify() test above to check
    annual_type_counts (year-boundary correctness), total_type_counts, and the
    dominance formula -- specifically that ambiguous months count toward the
    denominator but NOT the numerator, exactly as documented in dominance()'s
    docstring and the manuscript's Sec. 2.9 (Eq. for onset-order dominance index)."""
    cat = test_classify_wrapper_preserves_coords_and_matches_kernel()

    annual = ld.annual_type_counts(cat)
    # pixel2's HEAT_LEAD month is at t=2 -> 2001-01 (year 2001); its two AMBIGUOUS
    # months are at t=4,5 -> 2001-03, 2001-04 (also year 2001). Year 2000 must show
    # zero heat_leading for pixel2 (nothing before the Nov/Dec 2000 gap months).
    assert int(annual["heat_leading"].sel(year=2001, latitude=10.0, longitude=20.0)) == 1
    assert int(annual["heat_leading"].sel(year=2000, latitude=10.0, longitude=20.0)) == 0
    assert int(annual["ambiguous"].sel(year=2001, latitude=10.0, longitude=20.0)) == 2
    print("annual_type_counts groups by calendar year correctly, including a "
          "Nov/Dec-into-January boundary-crossing pixel")

    tot = ld.total_type_counts(cat)
    assert int(tot["heat_leading"].isel(latitude=0, longitude=0)) == 2
    assert int(tot["drought_leading"].isel(latitude=0, longitude=1)) == 2
    assert int(tot["heat_leading"].isel(latitude=0, longitude=2)) == 1
    assert int(tot["ambiguous"].isel(latitude=0, longitude=2)) == 2
    print("total_type_counts matches expected per-pixel full-record counts")

    dom = ld.dominance(cat)
    # pixel0: all heat-leading (h=2,d=0,tot=2) -> +1.0 (purely heat-led)
    assert np.isclose(float(dom.isel(latitude=0, longitude=0)), 1.0)
    # pixel1: all drought-leading (h=0,d=2,tot=2) -> -1.0 (purely drought-led)
    assert np.isclose(float(dom.isel(latitude=0, longitude=1)), -1.0)
    # pixel2: h=1, d=0, ambiguous=2 -> tot = 1+0+2 = 3 -> (1-0)/3, NOT (1-0)/1=1.0.
    # If ambiguous months were wrongly excluded from the denominator (or wrongly
    # folded into the numerator), this would come out as 1.0 instead of 1/3.
    assert np.isclose(float(dom.isel(latitude=0, longitude=2)), 1.0 / 3.0), (
        "ambiguous months must dilute the dominance denominator without entering "
        "the numerator -- got a value consistent with ambiguous being dropped or "
        "double-counted instead")
    print("dominance() correctly folds ambiguous months into the denominator only "
          "(pixel2 = 1/3, not 1.0)")


if __name__ == "__main__":
    test_heat_lead_basic()
    test_drought_lead_basic()
    test_concurrent_both_confirmed_absent()
    test_episode_onset_at_t_zero_is_concurrent()
    test_ambiguous_heat_absent_at_t_minus_1()
    test_ambiguous_heat_present_is_not_heat_lead()
    test_noncompound_months_confirmed_not_drought_stay_zero()
    test_episode_persists_uniform_label_across_run()
    test_two_episodes_separated_by_gap_classified_independently()
    test_multi_pixel_prange_independence()
    test_classify_wrapper_preserves_coords_and_matches_kernel()
    test_annual_total_and_dominance_aggregations()
    print("All chd.leading onset-order classification tests passed.")
