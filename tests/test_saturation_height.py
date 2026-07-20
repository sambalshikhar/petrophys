"""Tests for saturation-height modeling (capillary, J-function, Brooks-Corey)."""

import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.saturation.height import (
    pc_from_height,
    height_from_pc,
    convert_pc_lab_to_reservoir,
    leverett_j,
    pc_from_j,
    fit_j_function,
    fit_brooks_corey,
    saturation_height_from_logs,
)


def test_pc_height_round_trip():
    pc = pc_from_height(120.0, sg_water=1.05, sg_hc=0.7)
    assert pc == pytest.approx(0.433 * (1.05 - 0.7) * 120.0)
    assert height_from_pc(pc, 1.05, 0.7) == pytest.approx(120.0)


def test_height_below_fwl_is_zero_pc():
    assert pc_from_height(-50.0) == 0.0  # below FWL clipped


def test_leverett_j_inverse_round_trip():
    j = leverett_j(20.0, perm_md=50.0, phi=0.18, ift=30, theta=30)
    pc_back = pc_from_j(j, 50.0, 0.18, ift=30, theta=30)
    assert pc_back == pytest.approx(20.0, rel=1e-6)


def test_lab_to_reservoir_is_positive_magnitude():
    # air-mercury (theta=140) -> oil-brine: result must stay positive
    pc_res = convert_pc_lab_to_reservoir(100.0, ift_lab=480, theta_lab=140, ift_res=30, theta_res=30)
    assert pc_res > 0
    assert pc_res == pytest.approx(100.0 * abs(30 * np.cos(np.radians(30))) / abs(480 * np.cos(np.radians(140))), rel=1e-6)


def test_fit_j_function_recovers_power_law():
    rng = np.random.default_rng(3)
    j = rng.uniform(0.1, 20.0, 150)
    sw = np.clip(0.30 * j ** -0.45 * np.exp(rng.normal(0, 0.04, j.size)), 0.05, 1.0)
    m = fit_j_function(j, sw)
    assert m.a == pytest.approx(0.30, rel=0.1)
    assert m.b == pytest.approx(-0.45, rel=0.1)
    assert m.r2 > 0.95


def test_sw_decreases_with_height():
    m = fit_j_function(np.array([0.5, 1, 2, 5, 10.0]), np.array([0.5, 0.4, 0.32, 0.25, 0.2]))
    h = np.array([0.0, 10, 50, 100, 200.0])
    sw = m.sw_from_height(h, perm_md=np.full(5, 50.0), phi=np.full(5, 0.18))
    finite = sw[np.isfinite(sw)]
    assert np.all(np.diff(finite) <= 1e-9)  # monotonically non-increasing upward


def test_fit_brooks_corey():
    pc = np.array([2, 3, 5, 8, 12, 20, 40, 80, 150.0])
    sw = np.array([0.95, 0.80, 0.62, 0.50, 0.42, 0.36, 0.31, 0.28, 0.26])
    bc = fit_brooks_corey(pc, sw)
    assert 0.0 <= bc.swir <= 0.30
    assert bc.lam > 0 and bc.pd > 0
    assert bc.r2 > 0.95
    # high Pc -> approaches Swir
    assert bc.sw_from_pc(1e4) == pytest.approx(bc.swir, abs=0.02)


def test_saturation_height_from_logs():
    depth = np.arange(2000.0, 2100.0, 1.0)
    well = Well("SH")
    well.depth = depth
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("PERM", np.full(depth.size, 50.0), "mD", "k")
    well.add_curve("PHIE", np.full(depth.size, 0.18), "v/v", "phi")
    m = fit_j_function(np.array([0.5, 1, 2, 5, 10.0]), np.array([0.5, 0.4, 0.32, 0.25, 0.2]),
                       swir=0.15, form="power_swir")
    fwl = 2090.0
    sw = saturation_height_from_logs(well, m, fwl_depth=fwl)
    assert sw.shape == depth.shape
    # below FWL -> Sw == 1; well above -> below 1; never below Swir
    assert np.all(sw[depth >= fwl] == pytest.approx(1.0))
    assert np.nanmin(sw) >= m.swir - 1e-9
    assert np.nanmin(sw) < 1.0


def test_plot_renders():
    pytest.importorskip("matplotlib")
    from petrophys.visualization import create_saturation_height_plot
    j = np.geomspace(0.2, 20, 40)
    sw = np.clip(0.3 * j ** -0.4, 0.05, 1.0)
    m = fit_j_function(j, sw)
    fig = create_saturation_height_plot(m, j=j, sw=sw)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
