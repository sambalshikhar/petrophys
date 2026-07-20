import numpy as np

from petrophys.interpretation.porosity.effective_porosity import effective_porosity
from petrophys.interpretation.porosity.total_porosity import total_porosity


def test_total_porosity_adaptive_falls_back_per_sample_when_neutron_missing():
    phin = np.array([0.10, np.nan, 0.20, np.nan])
    phid = np.array([0.12, 0.25, np.nan, np.nan])

    phit = total_porosity(phin, phid)

    expected = np.array([0.6 * 0.12 + 0.4 * 0.10, 0.25, 0.20, np.nan])
    np.testing.assert_allclose(phit, expected, equal_nan=True)


def test_total_porosity_adaptive_scalar_fallbacks():
    assert total_porosity(np.nan, 0.25) == 0.25
    assert total_porosity(0.18, np.nan) == 0.18
    assert np.isnan(total_porosity(np.nan, np.nan))


def test_effective_porosity_respects_non_clay_volume_closure():
    phit = np.array([0.30, 0.30, 0.20])
    vcl = np.array([0.20, 0.85, 1.00])

    phie = effective_porosity(phit, vcl, phi_clay=0.10)

    expected = np.array([0.28, 0.15, 0.0])
    np.testing.assert_allclose(phie, expected)
    assert np.all(phie + np.clip(vcl, 0.0, 1.0) <= 1.0 + 1e-12)
