import numpy as np
import pytest

from petrophys import RQIClassification
from petrophys.evaluation.net_pay import (
    add_volumetric_pay_flag,
    generate_volumetric_pay_flag,
)
from petrophys.interpretation.core_analysis import (
    calibrate_saturation,
    core_data_from_well,
)
from petrophys.model import Well
from petrophys.model.core import CoreData


def test_core_data_from_well_recognizes_ambient_core_aliases():
    well = Well("AMBIENT-CORE")
    well.depth = np.array([1000.0, 1001.0, 1002.0])
    well.add_curve("CPOR-AMB", [0.12, 0.15, 0.18], "v/v", "Core porosity")
    well.add_curve("CPERM-AMB-KL", [1.0, 10.0, 100.0], "mD", "Klinkenberg perm")
    well.add_curve("CSW-AMB", [0.70, 0.55, 0.40], "v/v", "Core water saturation")

    core = core_data_from_well(well)

    np.testing.assert_allclose(core.porosity, [0.12, 0.15, 0.18])
    np.testing.assert_allclose(core.permeability, [1.0, 10.0, 100.0])
    np.testing.assert_allclose(core.sw, [0.70, 0.55, 0.40])
    assert core.metadata["core_porosity_curve"] == "CPOR-AMB"
    assert core.metadata["core_permeability_curve"] == "CPERM-AMB-KL"
    assert core.metadata["core_sw_curve"] == "CSW-AMB"


def test_calibrate_saturation_normalizes_percent_core_without_changing_log():
    well = Well("SW-CALIBRATION")
    well.depth = np.array([1000.0, 1001.0, 1002.0])
    original_sw = np.array([0.20, 0.40, 0.60])
    well.add_curve("SW", original_sw, "v/v", "Log water saturation")
    core = CoreData(
        depth=well.depth.copy(),
        sw=np.array([20.0, 40.0, 60.0]),
    )

    result = calibrate_saturation(well, core, core_unit="%")

    assert result.property_name == "Water saturation (SW)"
    assert result.n_points == 3
    assert result.r_squared == pytest.approx(1.0)
    assert result.bias == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    np.testing.assert_allclose(result.core_values, original_sw)
    np.testing.assert_allclose(result.log_values, original_sw)
    np.testing.assert_allclose(well.get("SW").data, original_sw)


def test_calibrate_saturation_excludes_out_of_range_and_missing_core_pairs():
    well = Well("SW-CALIBRATION-QC")
    well.depth = np.arange(1000.0, 1004.0)
    well.add_curve("SW", [0.2, 0.4, np.nan, 1.2], "v/v", "Log water saturation")
    core = CoreData(depth=well.depth.copy(), sw=np.array([0.25, -0.1, np.nan, 0.8]))

    result = calibrate_saturation(well, core)

    assert result.n_points == 1
    np.testing.assert_allclose(result.core_values, [0.25])
    np.testing.assert_allclose(result.log_values, [0.20])


def test_rqi_classification_handles_boundaries_arrays_and_missing_values():
    values = np.array([0.49, 0.50, 1.00, 1.01, np.nan])

    result = RQIClassification.classify(values)

    assert result.tolist() == [
        RQIClassification.LOW_RQI,
        RQIClassification.MODERATE_RQI,
        RQIClassification.MODERATE_RQI,
        RQIClassification.HIGH_RQI,
        None,
    ]
    assert RQIClassification.classify(np.nan) is None


def test_volumetric_pay_flag_is_conservative_and_reservoir_constrained():
    fluids = np.array(["oil", "brine", "gas-condensate", "unknown", None])
    reservoir = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

    result = generate_volumetric_pay_flag(fluids, reservoir_flag=reservoir)

    np.testing.assert_allclose(result[:3], [1.0, 0.0, 0.0])
    assert np.isnan(result[3])
    assert result[4] == 0.0
    assert np.isnan(generate_volumetric_pay_flag("unknown"))


def test_volumetric_pay_flag_supports_operator_labels_and_well_output():
    well = Well("VOLUMETRIC-PAY")
    well.depth = np.arange(1000.0, 1003.0)

    flag = add_volumetric_pay_flag(
        well,
        ["mobile hc", "formation water", "mobile hc"],
        reservoir_flag=[1.0, 1.0, 0.0],
        hydrocarbon_fluids={"mobile hc"},
        non_pay_fluids={"formation water"},
    )

    np.testing.assert_allclose(flag, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(well.get("PAY_FLAG_VOLUMETRIC").data, flag)


def test_volumetric_pay_flag_rejects_ambiguous_label_configuration():
    with pytest.raises(ValueError, match="both pay and non-pay"):
        generate_volumetric_pay_flag(
            "oil",
            hydrocarbon_fluids={"oil"},
            non_pay_fluids={"OIL"},
        )
