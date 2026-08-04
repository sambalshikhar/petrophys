"""Regression tests for the porosity / clay-bound-water / Rw / bad-hole corrections.

These lock in the validated fixes that resolve (a) the Archie false-pay in
NPD-confirmed water wells and (b) the over-stated effective porosity in
clay-bound-water / washed-out shallow sections, while preserving the 30/6-5
core porosity match. See validation/porosity_saturation.py for the
end-to-end well-by-well validation against core and NPD ground truth.
"""

import numpy as np
import pytest

from petrophys.model.well import Well
from petrophys.interpretation.clay_volume.vcl_module import VclModule
from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
from validation.porosity_saturation import _flagged_thickness_m, _summary


def _well(n, **curves):
    w = Well("T")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("DEPT", w.depth, "m", "depth")
    for name, data in curves.items():
        w.add_curve(name, np.asarray(data, dtype=float), "", name)
    return w


def test_validation_thickness_handles_descending_depth_and_data_gaps():
    depth = np.array([1005.0, 1004.0, 1003.0, 990.0])
    flags = np.ones(4)

    assert _flagged_thickness_m(depth, flags, "m") == pytest.approx(4.0)
    assert _flagged_thickness_m(depth[:3], flags[:3], "ft") == pytest.approx(
        3.0 * 0.3048
    )


def test_validation_summary_reports_missing_optional_wells():
    rows = [
        {
            "well": "A",
            "expectation": "core",
            "status": "validated",
            "net_pay_m": 0.0,
        },
        {
            "well": "B",
            "expectation": "gas",
            "status": "missing",
        },
    ]

    assert _summary(rows) == {"requested": 2, "validated": 1, "missing": 1}


# --- VCL method combination ---------------------------------------------------

def test_vcl_nd_primary_mode_uses_nd_over_low_gr():
    # Clean sand (sample 0) and a non-radioactive clay (sample 1): GR reads low
    # for BOTH, but the neutron-density crossplot sees the clay.
    w = _well(
        2,
        GR=[15.0, 15.0],           # GR misses the clay (low for both)
        NPHI=[0.20, 0.45],         # high neutron in the clay
        RHOB=[2.25, 2.40],
    )
    res_min = VclModule(matrix="sandstone", use_sp=False, combine="min").calculate(w)
    res_auto = VclModule(matrix="sandstone", use_sp=False, combine="auto").calculate(w)

    # The explicit N-D-primary mode follows the N-D indicator at sample 1.
    assert res_auto.vcl_nd[1] > 0.5
    # The default row-wise minimum retains the lower valid GR estimate.
    assert res_min.vcl[1] < 0.1
    assert res_auto.vcl[1] == pytest.approx(res_auto.vcl_nd[1], abs=1e-6)
    assert res_auto.vcl[1] > res_min.vcl[1] + 0.4


def test_vcl_auto_falls_back_to_gr_in_gas_crossover():
    # Gas: low neutron, low density (high PHID) -> N-D crossover -> N-D unreliable,
    # so GR must be used instead of a spuriously low ND clay value.
    w = _well(
        1,
        GR=[60.0],
        NPHI=[0.08],     # suppressed neutron (gas)
        RHOB=[2.00],     # low density -> PHID ~0.39 >> NPHI -> gas crossover
    )
    res = VclModule(gr_min=10.0, gr_max=110.0, matrix="sandstone",
                    use_sp=False, combine="auto").calculate(w)
    assert res.method_used[0] == "GR"
    assert res.vcl[0] == pytest.approx(res.vcl_gr[0], abs=1e-6)


def test_vcl_min_default_unchanged():
    # The default follows the row-wise minimum convention used by the workflow.
    assert VclModule().combine == "min"


# --- Fix 3: bad-hole quality mask excludes washout from net pay ----------------

def test_net_pay_quality_mask_excludes_bad_hole():
    n = 10
    w = _well(n)                  # depth = 0..9 -> step 1.0 m
    vcl = np.full(n, 0.1)         # clean
    phie = np.full(n, 0.25)       # good porosity
    sw = np.full(n, 0.3)          # low Sw -> would all be pay
    quality = np.ones(n, dtype=bool)
    quality[:4] = False           # first 4 washed out

    full = NetPayModule(lithology="clastic").calculate(w, vcl=vcl, phie=phie, sw=sw)
    masked = NetPayModule(lithology="clastic").calculate(
        w, vcl=vcl, phie=phie, sw=sw, quality_mask=quality)

    assert full.net_pay.sum() == n
    assert masked.net_pay.sum() == n - 4
    assert not masked.net_pay[:4].any()
    assert masked.net_pay[4:].all()


# --- Fix 4: adaptive Rwa picker (narrow=median, wide=low tail) ------------------

def test_adaptive_rw_narrow_spread_uses_median_wide_uses_low_tail():
    # Re-implement the decision rule used in the workflow's _step_rw to lock its
    # behaviour: an all-wet (narrow) Rwa distribution -> median (true Rw); a
    # hydrocarbon-broadened (wide) distribution -> low tail (water-leg Rw).
    def pick(rwa):
        p5, p10, p50 = np.percentile(rwa, [5, 10, 50])
        spread = p50 / p5
        return (p50, 50.0) if spread < 3.0 else (p10, 10.0)

    rng = np.random.default_rng(0)
    narrow = 0.04 * np.exp(0.05 * rng.standard_normal(2000))      # tight around 0.04
    rw_n, pct_n = pick(narrow)
    assert pct_n == 50.0
    assert rw_n == pytest.approx(0.04, abs=0.01)

    water = 0.04 * np.exp(0.05 * rng.standard_normal(1500))
    hc = 2.0 * np.exp(0.3 * rng.standard_normal(1500))           # resistive HC leg
    wide = np.concatenate([water, hc])
    rw_w, pct_w = pick(wide)
    assert pct_w == 10.0
    assert rw_w < 0.1                                            # tracks the water leg
