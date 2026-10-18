"""Regression tests for the NMR, geochemistry and rock-physics review fixes."""

import numpy as np
import pandas as pd
import pytest

from petrophys._internal.benchmark import historical_benchmark
from petrophys.interpretation.geochemistry import interpret_spectral_gamma
from petrophys.interpretation.geochemistry.elemental_spectroscopy import oxide_closure
from petrophys.interpretation.geochemistry.xrf import XRFData
from petrophys.interpretation.nmr import (
    DEFAULT_T2_CENTERS,
    NMRData,
    bvi_from_bins,
    ffi_from_bins,
)
from petrophys.interpretation.rock_physics import calculate_elastic_properties
from petrophys.model.well import Well


def _nmr_with_cutoff_gap():
    b = np.array([0, 1, 2, 3, 4, 3, 2, 1.0])
    return NMRData(
        depth=np.arange(4.0),
        mphi=np.full(4, 20.0),
        bins=np.vstack([b * 20 / b.sum()] * 4),
        t2_cutoff_curve=np.array([33.0, np.nan, 33.0, 33.0]),
    )


@pytest.mark.parametrize("partial_bins", [True, False])
def test_missing_per_depth_cutoff_gives_nan_bvi_ffi(partial_bins):
    nmr = _nmr_with_cutoff_gap()
    bvi = nmr.bvi_from_bins(partial_bins=partial_bins)
    ffi = nmr.ffi_from_bins(partial_bins=partial_bins)
    assert np.isnan(bvi[1]) and np.isnan(ffi[1])
    assert np.all(np.isfinite(bvi[[0, 2, 3]]))
    np.testing.assert_allclose(bvi[[0, 2, 3]] + ffi[[0, 2, 3]], 20.0)
    swirr = nmr.swirr_nmr()
    assert np.isnan(swirr[1]) and np.isfinite(swirr[0])


def test_standalone_bins_nan_cutoff_and_descending_centres():
    bins = np.ones((1, 8))
    assert np.isnan(bvi_from_bins(bins, DEFAULT_T2_CENTERS, cutoff_ms=np.nan)).all()
    assert np.isnan(ffi_from_bins(bins, DEFAULT_T2_CENTERS, cutoff_ms=np.nan)).all()
    with pytest.raises(ValueError, match="increasing"):
        bvi_from_bins(bins, DEFAULT_T2_CENTERS[::-1])
    np.testing.assert_allclose(bvi_from_bins(bins, DEFAULT_T2_CENTERS), [3.544394], rtol=1e-6)


def _nmr_with_stored_perm():
    return NMRData(
        depth=np.arange(3.0),
        mphi=np.full(3, 20.0),
        bins=np.full((3, 8), 2.5),
        stored_permeability={"KSDR": np.ones(3), "KTIM": np.full(3, 2.0)},
    )


def test_explicit_permeability_parameters_are_not_ignored():
    nmr = _nmr_with_stored_perm()
    # No arguments: the stored vendor curve, as documented.
    np.testing.assert_allclose(nmr.permeability_sdr(), 1.0)
    np.testing.assert_allclose(nmr.permeability_coates(), 2.0)
    np.testing.assert_allclose(
        nmr.permeability_sdr(c=10.0), nmr.permeability_sdr(c=10.0, prefer_stored=False)
    )
    assert not np.allclose(nmr.permeability_sdr(c=10.0), 1.0)
    expected = nmr.permeability_coates(c=5.0, a=3.0, prefer_stored=False)
    np.testing.assert_allclose(nmr.permeability_coates(c=5.0, a=3.0), expected)
    assert not np.allclose(expected, 2.0)
    np.testing.assert_allclose(
        nmr.permeability_coates(prefer_stored=False),
        nmr.permeability_coates(c=10.0, a=4.0, b=2.0),
    )


def test_pore_size_rejects_unknown_geometry():
    nmr = NMRData(depth=np.arange(2.0), mphi=np.full(2, 20.0), bins=np.ones((2, 8)))
    with pytest.raises(ValueError, match="geometry"):
        nmr.pore_size(geometry="slab")
    np.testing.assert_allclose(
        nmr.pore_size(geometry="sphere"), 3.0 * nmr.pore_size(geometry="slit")
    )


def test_bin_cumulative_nan_for_missing_distribution():
    nmr = NMRData(
        depth=np.arange(2.0),
        mphi=np.array([20.0, np.nan]),
        bins=np.vstack([np.ones(8), np.full(8, np.nan)]),
    )
    cum = nmr.bin_cumulative()
    assert np.isnan(cum[1]).all()
    assert cum[0, -1] == pytest.approx(1.0)


def _spectral_well(unit):
    well = Well("T")
    well.depth = np.arange(4.0) + 1000
    well.add_curve("POTA", np.array([0.02, 0.03, 0.01, 0.025]), unit=unit, description="Potassium")
    well.add_curve("THOR", np.array([4.0, 12, 10, 8]), unit="ppm", description="Thorium")
    return well


@pytest.mark.parametrize("unit", ["v/v", "frac", "dec"])
def test_spectral_gamma_fraction_units_converted(unit):
    result = interpret_spectral_gamma(_spectral_well(unit), add_curves=False)
    np.testing.assert_allclose(result.measurements["POTASSIUM"], [2.0, 3.0, 1.0, 2.5])
    assert result.conversions["POTASSIUM"] == "fraction_to_percent"


def test_spectral_gamma_unrecognised_unit_warns():
    result = interpret_spectral_gamma(_spectral_well("CPS"), add_curves=False)
    assert result.conversions["POTASSIUM"] == "assumed_percent"
    assert any("potassium unit 'CPS'" in w for w in result.warnings)


def test_spectral_gamma_benchmark_keeps_old_behaviour():
    with historical_benchmark():
        result = interpret_spectral_gamma(_spectral_well("v/v"), add_curves=False)
    np.testing.assert_allclose(result.measurements["POTASSIUM"], [0.02, 0.03, 0.01, 0.025])
    assert result.warnings == []


def test_oxide_closure_missing_yield_is_nan():
    result = oxide_closure(
        {"SI": np.array([0.10, np.nan]), "CA": np.array([0.40, 0.40]), "FE": np.array([0.02, 0.02])},
        {"SI": 1.0, "CA": 0.8, "FE": 2.0},
        {"SI": 2.139, "CA": 2.497, "FE": 1.358},
    )
    assert np.isnan(result.element_fractions["CA"][1])
    assert np.isnan(result.closure[1])
    assert result.closure[0] == pytest.approx(1.0)


def test_elastic_broadcasts_array_dts_against_scalars():
    props = calculate_elastic_properties(2.3, 100.0, np.array([150.0, 160.0]))
    assert props.vs_m_s.shape == (2,)
    ref = calculate_elastic_properties(np.array([2.3, 2.3]), np.array([100.0, 100.0]), np.array([150.0, 160.0]))
    np.testing.assert_allclose(props.young_mod_gpa, ref.young_mod_gpa)


def test_xrf_from_dataframe_null_sentinel():
    xrf = XRFData.from_dataframe(
        pd.DataFrame({"DEPTH": [1.0, 2, 3], "Si": [30.0, 20, 25], "Al": [5.0, -999.25, 10]})
    )
    np.testing.assert_allclose(xrf.mineral_proxies()["clay_index"], [0.0, np.nan, 1.0])
