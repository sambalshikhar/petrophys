import numpy as np

from petrophys.base.well import Well
from petrophys.petrophysics.permeability import lith_vcl_permeability
import petrophys.petrophysics.permeability.perm_module as perm_module


def _perm_well():
    well = Well("PERM-LITH-VCL")
    well.depth = np.arange(6.0)
    well.add_curve("PHIT", np.array([0.24, 0.20, 0.16, 0.12, 0.18, 0.22]), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.array([0.22, 0.18, 0.14, 0.10, 0.16, 0.20]), "v/v", "Effective Porosity")
    well.add_curve("VCL", np.array([0.05, 0.10, 0.20, 0.30, 0.15, 0.00]), "v/v", "Clay Volume")
    well.add_curve("LITHOLOGY_GEOLINK", np.array([11, 17, 24, 20, 16, 1]), "", "Lithology Code")
    well.add_curve("SW", np.full(6, 0.35), "v/v", "Water Saturation")
    well.add_curve("RDEP", np.full(6, 20.0), "ohm.m", "Deep Resistivity")
    return well


def test_lith_vcl_permeability_matches_lithology_formula():
    well = _perm_well()
    module = perm_module.PermeabilityModule(method="lith_vcl")

    result = module.calculate(well)

    expected = lith_vcl_permeability(
        well.get("PHIT").data,
        well.get("VCL").data,
        well.get("LITHOLOGY_GEOLINK").data,
    )
    np.testing.assert_allclose(result.perm, expected, equal_nan=True)
    assert result.method == "lith_vcl"
    assert result.swi_source == "not_required"
    assert result.metadata["method_metadata"]["phit_curve"] == "PHIT"
    assert result.metadata["method_metadata"]["vcl_curve"] == "VCL"
    assert result.metadata["method_metadata"]["lithology_curve"] == "LITHOLOGY_GEOLINK"


def test_auto_empirical_falls_back_to_lith_vcl_when_timur_fails(monkeypatch):
    well = _perm_well()

    def fail_timur(phi, swi):
        raise ValueError("timur unavailable")

    monkeypatch.setattr(perm_module, "timur", fail_timur)
    module = perm_module.PermeabilityModule(method="auto", fallback_method="timur")

    result = module.calculate(well)

    assert result.metadata["empirical_method"] == "lith_vcl"
    assert result.metadata["empirical_attempted_methods"] == ["timur", "lith_vcl"]
    assert "timur unavailable" in result.metadata["empirical_error"]
    assert result.swi_source == "not_required"
