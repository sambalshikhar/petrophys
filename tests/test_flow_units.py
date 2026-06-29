"""Tests for hydraulic flow units (Amaefule FZI method)."""

import numpy as np
import pytest

from petrophys import Well
from petrophys.reservoir import (
    calculate_flow_units,
    assign_flow_units,
    predict_perm_from_fzi,
    FlowUnitModule,
)

_C = 0.0314


def _synthetic(fzi_levels=(0.5, 2.0, 6.0), per=200, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.concatenate([rng.uniform(0.10, 0.30, per) for _ in fzi_levels])
    fzi = np.concatenate([np.full(per, f) for f in fzi_levels])
    phi_z = phi / (1 - phi)
    k = phi * ((fzi * phi_z) / _C) ** 2 * np.exp(rng.normal(0, 0.12, phi.size))
    return phi, k


def test_recovers_known_units():
    phi, k = _synthetic()
    res = calculate_flow_units(k, phi)  # auto n_units
    assert res.n_units == 3
    assert np.allclose(np.sort(res.fzi_unit_mean), [0.5, 2.0, 6.0], rtol=0.12)
    assert res.counts.sum() == phi.size


def test_perm_prediction_round_trips():
    # k computed from FZI must come back through predict_perm_from_fzi
    phi = np.array([0.10, 0.20, 0.25])
    fzi = np.array([1.0, 3.0, 0.5])
    phi_z = phi / (1 - phi)
    k = phi * ((fzi * phi_z) / _C) ** 2
    k_pred = predict_perm_from_fzi(fzi, phi)
    assert np.allclose(k_pred, k, rtol=1e-6)


def test_prediction_correlates_with_truth():
    phi, k = _synthetic()
    res = calculate_flow_units(k, phi)
    ok = np.isfinite(res.perm_predicted) & (k > 0)
    r = np.corrcoef(np.log10(res.perm_predicted[ok]), np.log10(k[ok]))[0, 1]
    assert r > 0.95


def test_explicit_cutoffs():
    phi, k = _synthetic()
    res = calculate_flow_units(k, phi, method="cutoffs", fzi_cutoffs=[1.0, 4.0])
    assert res.n_units == 3
    # all three units populated
    assert np.all(res.counts > 0)


def test_assign_handles_nan_and_zero():
    fzi = np.array([1.0, np.nan, 2.0, 0.0, 5.0])
    labels, means, bounds = assign_flow_units(fzi, n_units=2)
    assert labels[1] == 0 and labels[3] == 0      # nan / non-positive -> no unit
    assert np.all(labels[[0, 2, 4]] >= 1)


def test_module_adds_curves():
    phi, k = _synthetic(per=80)
    depth = np.arange(1000.0, 1000.0 + phi.size * 0.5, 0.5)[: phi.size]
    well = Well("FU")
    well.depth = depth
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("PHIE", phi, "v/v", "phi")
    well.add_curve("PERM", k, "mD", "k")
    FlowUnitModule().add_to_well(well)
    for name in ("RQI", "PHI_Z", "FZI", "HFU", "PERM_HFU"):
        assert well.get(name) is not None


def test_plot_renders():
    pytest.importorskip("matplotlib")
    from petrophys.visualization import create_flow_unit_plot
    phi, k = _synthetic(per=60)
    fig = create_flow_unit_plot(phi_e=phi, perm=k)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
