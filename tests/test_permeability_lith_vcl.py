import numpy as np

from petrophys.model.well import Well
from petrophys.interpretation.permeability import lith_vcl_permeability, timur
import petrophys.interpretation.permeability.perm_module as perm_module


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
    expected = np.minimum(expected, perm_module.PERMEABILITY_MAX_MD)
    np.testing.assert_allclose(result.perm, expected, equal_nan=True)
    assert result.method == "lith_vcl"
    assert result.swi_source == "not_required"
    assert result.metadata["method_metadata"]["phit_curve"] == "PHIT"
    assert result.metadata["method_metadata"]["vcl_curve"] == "VCL"
    assert result.metadata["method_metadata"]["lithology_curve"] == "LITHOLOGY_GEOLINK"
    assert result.metadata["perm_max_clip_md"] == 10000.0


def test_permeability_module_caps_outputs_at_10000_md():
    well = _perm_well()
    well.add_curve("PHIT", np.full(well.n_samples, 0.45), "v/v", "High Total Porosity")
    well.add_curve("VCL", np.full(well.n_samples, 0.01), "v/v", "Clay Volume")
    module = perm_module.PermeabilityModule(method="lith_vcl")

    result = module.calculate(well)

    assert np.nanmax(result.perm) == perm_module.PERMEABILITY_MAX_MD
    assert result.metadata["perm_max_clip_md"] == perm_module.PERMEABILITY_MAX_MD


def test_auto_empirical_prefers_coates_before_timur():
    well = _perm_well()
    module = perm_module.PermeabilityModule(method="auto")

    result = module.calculate(well)

    assert result.metadata["empirical_method"] == "coates"
    assert result.metadata["empirical_attempted_methods"] == ["coates"]
    assert result.metadata["method_metadata"]["equation"] == "10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2"


def test_auto_empirical_falls_back_to_lith_vcl_when_coates_and_timur_fail(monkeypatch):
    well = _perm_well()

    def fail_coates(phi, swi, **kwargs):
        raise ValueError("coates unavailable")

    def fail_timur(phi, swi):
        raise ValueError("timur unavailable")

    monkeypatch.setattr(perm_module, "coates_denoo", fail_coates)
    monkeypatch.setattr(perm_module, "timur", fail_timur)
    module = perm_module.PermeabilityModule(method="auto", fallback_method="timur")

    result = module.calculate(well)

    assert result.metadata["empirical_method"] == "lith_vcl"
    assert result.metadata["empirical_attempted_methods"] == ["coates", "timur", "lith_vcl"]
    assert "coates unavailable" in result.metadata["empirical_error"]
    assert "timur unavailable" in result.metadata["empirical_error"]
    assert result.swi_source == "not_required"


def test_direct_coates_uses_fixed_formula_without_lithology_specific_values():
    well = _perm_well()
    module = perm_module.PermeabilityModule(method="coates")

    result = module.calculate(well, swi=0.25)

    phi = well.get("PHIE").data
    expected = np.minimum(
        10000.0 * phi**4.0 * ((1.0 - 0.25) / 0.25) ** 2,
        perm_module.PERMEABILITY_MAX_MD,
    )
    np.testing.assert_allclose(result.perm, expected)


def test_coates_ignores_lithology_inputs():
    well = _perm_well()
    well.remove_curve("LITHOLOGY_GEOLINK")
    module = perm_module.PermeabilityModule(method="coates")

    result = module.calculate(well, swi=0.25, lithology="carbonate")

    phi = well.get("PHIE").data
    expected = np.minimum(
        10000.0 * phi**4.0 * ((1.0 - 0.25) / 0.25) ** 2,
        perm_module.PERMEABILITY_MAX_MD,
    )
    np.testing.assert_allclose(result.perm, expected)
    assert "lithology_curve" not in result.metadata["method_metadata"]


def test_timur_uses_conventional_0136_coefficient_with_percent_inputs():
    expected = 0.136 * (20.0**4.4) / (20.0**2.0)
    assert timur(0.20, 0.20) == expected


def test_estimate_swirr_app_builds_constant_curve_from_hc_mask_p5():
    sw = np.array([0.60, 0.42, 0.34, 0.28, 0.25, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16])
    rdep = np.full(sw.shape, 12.0)
    resflag = np.ones(sw.shape, dtype=bool)

    swirr_app, swirr, n_samples, source = perm_module.estimate_swirr_app(sw, rdep, resflag)

    assert source == "hc_mask_p5"
    assert n_samples == sw.size
    assert swirr == np.percentile(sw, 5)
    np.testing.assert_allclose(swirr_app, swirr)
