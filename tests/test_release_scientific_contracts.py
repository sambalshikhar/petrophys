"""Independent limiting cases discovered during release verification."""
import numpy as np
import pytest

from petrophys.interpretation.clay_volume.gr_method import vcl_linear
from petrophys.interpretation.permeability import coates_denoo, timur, wyllie_rose
from petrophys.interpretation.porosity import density_porosity, wyllie_time_average
from petrophys.interpretation.saturation import archie


@pytest.mark.parametrize("function,expected", [
    (timur, 0.136 * 0.05 ** 4.4 / 0.5 ** 2),
    (coates_denoo, 10000 * .0005 ** 4 * ((1 - .005) / .005) ** 2),
    (wyllie_rose, 62500 * .0005 ** 6 / .005 ** 2),
])
def test_small_positive_permeability_inputs_obey_same_equation_for_all_shapes(function, expected):
    assert function(.0005, .005) == pytest.approx(expected, rel=1e-12, abs=0)
    np.testing.assert_allclose(function(np.array([.0005]), np.array([.005])), [expected], rtol=1e-12, atol=0)
    np.testing.assert_allclose(function([.0005], [.005]), [expected], rtol=1e-12, atol=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_unknown_measurements_remain_unknown_in_porosity_and_saturation(value):
    assert np.isnan(density_porosity(value, 1., 1.))
    assert np.isnan(wyllie_time_average(value, 55.5, 55.5))
    assert np.isnan(archie(value, .05, .2))


def test_degenerate_porosity_endpoints_do_not_invent_zero_porosity():
    np.testing.assert_allclose(density_porosity([2.32, 2.32], [2.65, 1.], 1.), [.2, np.nan], equal_nan=True)
    np.testing.assert_allclose(wyllie_time_average([82.2, 82.2], 55.5, [189., 55.5]), [.2, np.nan], equal_nan=True)


def test_time_mixing_remains_valid_when_matrix_is_slower_than_fluid():
    # Independent volume/time balance: 0.8*230 + 0.2*189 = 221.8 us/ft.
    assert wyllie_time_average(221.8, dtc_matrix=230., dtc_fluid=189.) == pytest.approx(.2)
    np.testing.assert_allclose(wyllie_time_average([221.8], 230., 189.), [.2])


def test_mass_balance_remains_valid_when_fluid_is_denser_than_matrix():
    # Independent mass balance: 0.8*1.5 + 0.2*3 = 1.8 g/cc.
    assert density_porosity(1.8, rhob_matrix=1.5, rhob_fluid=3.) == pytest.approx(.2)
    np.testing.assert_allclose(density_porosity([1.8], 1.5, 3.), [.2])


def test_unresolved_gr_contrast_never_becomes_clean_rock():
    assert np.isnan(vcl_linear(50., 20., 20.))
    assert np.isnan(vcl_linear(np.array([np.nan, 50.]), 20., 20.)).all()


def test_scalar_results_keep_the_scalar_api_contract():
    assert isinstance(archie(10., .05, .2), float)
    assert isinstance(archie(np.inf, .05, .2), float)
    assert isinstance(vcl_linear(50., 20., 20.), float)
