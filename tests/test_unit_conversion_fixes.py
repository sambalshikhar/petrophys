"""Fraction-unit labels for NMR porosity and percent core porosity are converted."""
import warnings

import numpy as np
import pytest

from petrophys.interpretation.core_analysis import core_data_from_well
from petrophys.interpretation.nmr.nmr_data import _porosity_to_pu
from petrophys.model.well import Well


@pytest.mark.parametrize("unit", ["m3/m3", "ft3/ft3", "dec", "CFCF", "V/V", "frac"])
def test_nmr_fraction_units_become_porosity_units(unit):
    assert np.allclose(_porosity_to_pu(np.array([0.25, 0.1]), unit, "MPHI"), [25.0, 10.0])


def test_nmr_pu_stays_pu():
    assert np.allclose(_porosity_to_pu(np.array([25.0]), "PU", "MPHI"), [25.0])


def test_percent_core_porosity_is_converted():
    well = Well(name="core")
    well.depth = np.arange(1000.0, 1005.0)
    well.add_curve("PHIC", np.array([19.0, 21.0, 18.5, 22.0, 20.0]), "%", "Core porosity")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        core = core_data_from_well(well)
    assert np.allclose(core.porosity, [0.19, 0.21, 0.185, 0.22, 0.20])
    assert core.metadata["core_porosity_normalization"]["factor"] == 0.01


def test_simandoux_non_porous_shale_tends_to_the_clay_term_limit():
    from petrophys.interpretation.saturation.simandoux import simandoux
    sw = simandoux(rt=5.0, phi=0.0, rw=0.05, vcl=0.5, rcl=2.0)
    assert sw == pytest.approx(2.0 / (0.5 * 5.0))       # Rcl / (Vcl * Rt) = 0.8
    a = np.array([0.2, 0.1]); rt = np.array([20.0, 8.0]); vcl = np.array([0.1, 0.3])
    A, B, C = a ** 2 / 0.05, vcl / 2.0, 1 / rt
    assert np.allclose(simandoux(rt=rt, phi=a, rw=0.05, vcl=vcl, rcl=2.0), (-B + np.sqrt(B * B + 4 * A * C)) / (2 * A))


def test_sp_rw_at_the_clip_floor_is_rejected():
    import petrophys as pp
    well = pp.read(pp.sample("30_6-5"))
    for key in [k for k in list(well._parameters) if k.upper().startswith(("RW", "SAL"))]:
        well._parameters.pop(key)
    result = pp.interpret(well)
    rw = [s for s in result.steps_executed if s.step == "rw"][0]
    assert rw.method != "sp_clean_water"
    assert any("range of the SP method" in w for w in result.warnings)
