"""Known-value tests for the clay-volume formula family.

These were previously untested at the formula level (audit gap): a transposed
constant in any GR transform would have propagated silently into porosity, Sw
routing, and net pay. Values here are computed by hand from the textbook forms,
not by calling the implementation, so they are non-circular.
"""

import numpy as np
import pytest

from petrophys.interpretation.clay_volume.gr_method import (
    vcl_linear,
    vcl_larionov_tertiary,
    vcl_larionov_older,
    vcl_steiber,
    vcl_clavier,
    vcl_from_gr,
)
from petrophys.interpretation.clay_volume.sp_method import vcl_from_sp
from petrophys.interpretation.clay_volume.nd_method import vcl_from_nd, nd_clean_line
from petrophys.interpretation.clay_volume import calculate_rcl


# IGR = 0.5 at GR=60 for the [20, 100] baseline
GR = 60.0
GR_MIN, GR_MAX = 20.0, 100.0
IGR = 0.5


def test_linear_igr():
    assert vcl_linear(GR, GR_MIN, GR_MAX) == pytest.approx(0.5)


def test_larionov_tertiary_known_value():
    # 0.083 * (2**(3.7*IGR) - 1) at IGR=0.5
    expected = 0.083 * (2 ** (3.7 * IGR) - 1)
    assert vcl_larionov_tertiary(IGR) == pytest.approx(expected, rel=1e-9)
    # sanity: tertiary curve sits well below the linear index
    assert vcl_larionov_tertiary(IGR) < IGR


def test_larionov_older_known_value():
    # 0.33 * (2**(2*IGR) - 1) -> 0.33 at IGR=0.5
    assert vcl_larionov_older(IGR) == pytest.approx(0.33, rel=1e-9)


def test_steiber_known_value():
    # IGR / (3 - 2*IGR) -> 0.25 at IGR=0.5
    assert vcl_steiber(IGR) == pytest.approx(0.25, rel=1e-9)


def test_clavier_known_value():
    # 1.7 - sqrt(3.38 - (IGR + 0.7)**2)
    expected = 1.7 - np.sqrt(3.38 - (IGR + 0.7) ** 2)
    assert vcl_clavier(IGR) == pytest.approx(expected, rel=1e-9)


def test_all_methods_bounded_0_1():
    igr = np.linspace(0, 1, 11)
    for fn in (vcl_larionov_tertiary, vcl_larionov_older, vcl_steiber, vcl_clavier):
        out = np.asarray(fn(igr))
        assert np.all(out >= -1e-9) and np.all(out <= 1 + 1e-9)


def test_endpoints_clean_and_shale():
    # IGR=0 -> 0 for every transform; IGR=1 -> 1 for steiber/clavier, but the
    # Larionov forms saturate just below 1 by construction (older -> 0.33*3 = 0.99,
    # tertiary -> 0.083*(2**3.7 - 1) ~= 0.996).
    for fn in (vcl_larionov_tertiary, vcl_larionov_older, vcl_steiber, vcl_clavier):
        assert fn(0.0) == pytest.approx(0.0, abs=1e-6)
        assert fn(1.0) >= 0.98
    assert vcl_steiber(1.0) == pytest.approx(1.0, abs=1e-9)
    assert vcl_clavier(1.0) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "gr_min,gr_max,expected",
    [
        (40, 80, [0.25, 0.5, 0.75]),
        (10, 80, [1 / 3, 5 / 9, 7 / 9]),
        (40, 120, [0.2, 0.4, 0.6]),
        (10, 120, [3 / 11, 5 / 11, 7 / 11]),
        (20, 100, [0.25, 0.5, 0.75]),
    ],
)
def test_vcl_from_gr_clamps_baselines_to_min_window(gr_min, gr_max, expected):
    # Each bound acts independently; wider caller windows must not become 20/100.
    out = vcl_from_gr(np.array([40.0, 60.0, 80.0]), gr_min, gr_max, method="linear")
    np.testing.assert_allclose(out, expected)


def test_vcl_from_sp_known_value():
    # (sp_max - sp) / (sp_max - sp_min); sp_max = clean (most negative)
    out = vcl_from_sp(-40.0, sp_max=-80.0, sp_min=0.0)
    assert float(out) == pytest.approx(0.5, rel=1e-9)


def test_calculate_rcl_median_over_pure_shale():
    rdep = np.array([2.0, 3.0, 100.0, 4.0])
    vcl = np.array([0.95, 0.92, 0.10, 0.99])  # samples 0,1,3 are pure shale
    rcl = calculate_rcl(rdep, vcl, vcl_threshold=0.9)
    # median of [2, 3, 4] = 3.0
    assert rcl == pytest.approx(3.0, rel=1e-6)


def test_calculate_rcl_falls_back_to_default():
    rdep = np.array([10.0, 12.0, 11.0])
    vcl = np.array([0.1, 0.2, 0.15])  # no pure-shale samples
    rcl = calculate_rcl(rdep, vcl, vcl_threshold=0.9, default_rcl=5.0)
    assert rcl == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Matrix-aware N-D clean line
# ---------------------------------------------------------------------------
def test_nd_clean_line_limestone_is_historical_default():
    # Carbonate back-compat: must equal the original hard-coded limestone line.
    assert nd_clean_line("limestone") == (2.71, 0.00, 2.03, 0.40)
    assert nd_clean_line(None) == (2.71, 0.00, 2.03, 0.40)


def test_nd_clean_line_lithology_aliases():
    assert nd_clean_line("clastic") == nd_clean_line("sandstone")
    assert nd_clean_line("carbonate") == nd_clean_line("limestone")
    assert nd_clean_line("quartz") == nd_clean_line("sandstone")
    assert nd_clean_line("chalk") == nd_clean_line("limestone")


def _vcl_nd_point(nphi, rhob, matrix):
    d1, n1, d2, n2 = nd_clean_line(matrix)
    out = vcl_from_nd(np.array([nphi]), np.array([rhob]),
                      den_cl1=d1, neu_cl1=n1, den_cl2=d2, neu_cl2=n2)
    return float(np.asarray(out)[0])


def test_clean_matrix_point_reads_zero_vcl_under_matching_line():
    # A clean quartz point under the sandstone line, and a clean calcite point
    # under the limestone line, should both read ~0 clay.
    assert _vcl_nd_point(-0.02, 2.65, "sandstone") == pytest.approx(0.0, abs=1e-6)
    assert _vcl_nd_point(0.00, 2.71, "limestone") == pytest.approx(0.0, abs=1e-6)
