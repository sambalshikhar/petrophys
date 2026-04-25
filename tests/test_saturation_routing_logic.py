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
