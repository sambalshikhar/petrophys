import numpy as np
import pytest

from petrophys import Well
from petrophys.petrophysics.saturation.sw_module import SaturationModule


def _saturation_well(vcl=0.6):
    well = Well("SATURATION-LOGIC")
    well.depth = np.arange(4.0)
    well.add_curve("RDEP", np.full(4, 20.0), "ohm.m", "Deep Resistivity")
    well.add_curve("PHIE", np.full(4, 0.20), "v/v", "Effective Porosity")
    well.add_curve("PHIT", np.full(4, 0.24), "v/v", "Total Porosity")
    well.add_curve("VCL", np.full(4, vcl), "v/v", "Clay Volume")
    return well


@pytest.mark.parametrize(
    "case_id,vcl,lithology,salinity,cec,b_coeff,expected_model",
    [
        ("clean_clastic_uses_archie", 0.10, "clastic", None, None, None, "archie"),
        ("shaly_clastic_high_salinity_uses_simandoux", 0.60, "clastic", 35000.0, None, None, "simandoux"),
        ("shaly_clastic_low_salinity_uses_indonesia", 0.60, "clastic", 5000.0, None, None, "indonesia"),
        ("shaly_carbonate_uses_dual_water", 0.60, "carbonate", None, None, None, "dual_water"),
        ("shaly_clastic_cec_and_b_coeff_uses_waxman", 0.60, "clastic", None, 0.08, 4.0, "waxman_smits"),
        ("shaly_clastic_cec_without_b_coeff_uses_salinity_route", 0.60, "clastic", 35000.0, 0.08, None, "simandoux"),
    ],
)
def test_saturation_auto_model_selection_matrix(case_id, vcl, lithology, salinity, cec, b_coeff, expected_model):
    well = _saturation_well(vcl=vcl)

    result = SaturationModule(model="auto").calculate(
        well,
        rw=0.05,
        rcl=5.0,
        salinity=salinity,
        lithology=lithology,
        cec=cec,
        b_coeff=b_coeff,
    )

    assert result.model == expected_model
    assert result.model_counts == {expected_model: 4}
    assert np.isfinite(result.sw).all()


@pytest.mark.parametrize(
    "forced_model,kwargs",
    [
        ("archie", {}),
        ("north_sea", {"rtsh": 5.0}),
        ("nigeria_schlumberger", {"rtsh": 5.0}),
        ("simandoux", {}),
        ("indonesia", {}),
        ("dual_water", {"rwb": 0.02}),
        ("waxman_smits", {"cec": 0.08, "b_coeff": 4.0}),
    ],
)
def test_forced_saturation_models_produce_one_curve(forced_model, kwargs):
    well = _saturation_well(vcl=0.6)

    result = SaturationModule(model=forced_model).calculate(
        well,
        rw=0.05,
        rcl=5.0,
        salinity=35000.0,
        lithology="clastic",
        **kwargs,
    )

    assert result.model == forced_model
    assert result.model_counts == {forced_model: 4}
    assert result.sw.shape == (4,)
    assert np.isfinite(result.sw).all()


def test_saturation_defaults_missing_vcl_to_clean_archie_route():
    well = _saturation_well(vcl=0.6)
    well.remove_curve("VCL")

    result = SaturationModule(model="auto").calculate(well, rw=0.05, rcl=5.0)

    assert result.model == "archie"
    assert result.model_counts == {"archie": 4}


def test_saturation_raises_when_required_curves_are_missing():
    well = _saturation_well()
    well.remove_curve("PHIE")

    with pytest.raises(ValueError, match="PHIE"):
        SaturationModule(model="auto").calculate(well, rw=0.05)


def test_saturation_prefers_core_corrected_phie_when_available():
    well = _saturation_well(vcl=0.1)
    well.add_curve("PHIE_CC", np.full(4, 0.30), "v/v", "Core-corrected effective porosity")

    result = SaturationModule(model="archie").calculate(well, rw=0.05)

    expected = np.sqrt(0.05 / (np.power(0.30, 2.0) * 20.0))
    np.testing.assert_allclose(result.sw, np.full(4, expected))


def test_sxo_is_never_below_sw_when_invaded_zone_archie_is_used():
    well = _saturation_well(vcl=0.1)
    well.add_curve("RSHA", np.full(4, 2000.0), "ohm.m", "Shallow resistivity")

    result = SaturationModule(model="archie").calculate(
        well,
        rw=0.05,
        rmf=0.02,
    )

    assert result.sxo_method == "archie_rxo_rmf"
    assert np.all(result.sxo >= result.sw)
    assert np.nanmax(result.sxo) <= 1.0


def test_sxo_falls_back_when_shallow_resistivity_curve_has_no_finite_samples():
    well = _saturation_well(vcl=0.6)
    well.add_curve("RSHA", np.full(4, np.nan), "ohm.m", "Shallow resistivity")

    result = SaturationModule(model="auto").calculate(
        well,
        rw=0.05,
        rcl=5.0,
        rmf=0.08,
        lithology="clastic",
    )

    assert result.sxo_method == "fifth_root_quicklook"
    assert result.sxo_reason == "RSHA/RXO/RESS present but had no finite usable samples"
    np.testing.assert_allclose(result.sxo, np.power(np.clip(result.sw, 0.0, 1.0), 0.2))


def test_sxo_uses_finite_secondary_shallow_alias_when_primary_is_nonfinite():
    well = _saturation_well(vcl=0.1)
    well.add_curve("RSHA", np.full(4, np.nan), "ohm.m", "Empty shallow resistivity")
    well.add_curve("RXO", np.full(4, 8.0), "ohm.m", "Finite flushed resistivity")

    result = SaturationModule(model="archie").calculate(
        well,
        rw=0.05,
        rmf=0.08,
    )

    assert result.sxo_method == "archie_rxo_rmf"
    assert result.sxo_reason == "RSHA/RXO/RESS and RMF available"
    assert np.isfinite(result.sxo).all()


def test_sxo_mixes_invaded_archie_and_fifth_root_depthwise():
    well = _saturation_well(vcl=0.1)
    well.add_curve("RSHA", np.array([8.0, np.nan, np.nan, np.nan]), "ohm.m", "Shallow resistivity")
    well.add_curve("RXO", np.array([np.nan, 8.0, np.nan, np.nan]), "ohm.m", "Flushed resistivity")

    result = SaturationModule(model="archie").calculate(
        well,
        rw=0.05,
        rmf=0.08,
    )

    assert result.sxo_method == "archie_rxo_rmf+fifth_root_quicklook"
    assert result.sxo_method_counts == {"archie_rxo_rmf": 2, "fifth_root_quicklook": 2}
    assert result.sxo_input_sources["rxo"] == {"RSHA": 1, "RXO": 1}
    np.testing.assert_allclose(
        result.sxo[2:],
        np.power(np.clip(result.sw[2:], 0.0, 1.0), 0.2),
    )
    assert np.isfinite(result.sxo).all()
