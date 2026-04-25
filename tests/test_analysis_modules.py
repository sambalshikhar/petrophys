from types import SimpleNamespace

import numpy as np

from petrophys import Well
from petrophys.analysis import HydrocarbonAnalysis
from petrophys.io.example_format_export import (
    _build_reference_derived_log_status,
    _missing_curve_reason,
)
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow import run_comprehensive_analysis


def _build_analysis_well() -> Well:
    depth = np.arange(1000.0, 1005.0, 0.5)
    well = Well("ANALYSIS-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.array([45, 48, 52, 55, 60, 85, 95, 110, 120, 130]), "API", "Gamma Ray")
    well.add_curve("RHOB", np.array([2.18, 2.20, 2.22, 2.25, 2.30, 2.45, 2.50, 2.58, 2.62, 2.66]), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.22, 0.24, 0.28, 0.30, 0.32]), "v/v", "Neutron Porosity")
    well.add_curve("PHIN", np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.22, 0.24, 0.28, 0.30, 0.32]), "v/v", "Corrected Neutron Porosity")
    well.add_curve("PHID", np.array([0.25, 0.24, 0.23, 0.22, 0.20, 0.12, 0.10, 0.07, 0.05, 0.03]), "v/v", "Density Porosity")
    well.add_curve("PHIS", np.array([0.14, 0.14, 0.14, 0.14, 0.13, 0.10, 0.09, 0.07, 0.06, 0.05]), "v/v", "Sonic Porosity")
    well.add_curve("PHIT", np.array([0.22, 0.21, 0.20, 0.19, 0.18, 0.12, 0.10, 0.08, 0.06, 0.04]), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.array([0.20, 0.19, 0.18, 0.17, 0.16, 0.10, 0.09, 0.07, 0.05, 0.03]), "v/v", "Effective Porosity")
    well.add_curve("VCL", np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.35, 0.40, 0.55, 0.65, 0.75]), "v/v", "Clay Volume")
    well.add_curve("RDEP", np.array([120.0, 110.0, 100.0, 80.0, 50.0, 20.0, 12.0, 6.0, 3.0, 1.5]), "ohm.m", "Deep Resistivity")
    well.add_curve("RXO", np.array([60.0, 55.0, 52.0, 48.0, 35.0, 18.0, 10.0, 5.5, 3.2, 1.4]), "ohm.m", "Flushed Resistivity")
    well.add_curve("SW", np.array([0.12, 0.15, 0.18, 0.22, 0.30, 0.45, 0.55, 0.70, 0.82, 0.92]), "v/v", "Water Saturation")
    well.add_curve("SXO", np.array([0.55, 0.58, 0.60, 0.65, 0.75, 0.80, 0.88, 0.95, 0.98, 1.00]), "v/v", "Flushed Water Saturation")
    well.add_curve("NET_PAY", np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), "", "Net Pay Flag")
    well.add_curve("PERM", np.array([200.0, 150.0, 80.0, 30.0, 8.0, 2.0, 0.8, 0.2, 0.05, 0.01]), "mD", "Permeability")
    well.add_curve("DTC", np.array([80.0, 82.0, 84.0, 86.0, 88.0, 95.0, 100.0, 110.0, 125.0, 135.0]), "us/ft", "Compressional Sonic")
    well.add_curve("DTS", np.array([150.0, 152.0, 134.0, 136.0, 140.0, 160.0, 170.0, 185.0, 210.0, 225.0]), "us/ft", "Shear Sonic")
    well.add_curve("PEF", np.array([2.2, 2.3, 2.4, 2.6, 2.8, 4.6, 4.8, 5.0, 3.0, 2.0]), "", "PEF")
    well.add_curve("DRHO", np.array([0.02, 0.03, 0.02, 0.01, 0.00, -0.02, -0.04, -0.06, 0.12, 0.16]), "g/cc", "Density Correction")
    well.add_curve("CALI", np.array([8.5, 8.6, 8.4, 9.4, 7.9, 8.5, 8.4, 9.2, 8.0, 8.6]), "in", "Caliper")
    well.add_curve("BS", np.array([8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5]), "in", "Bit Size")
    well.add_curve("LITHOLOGY_GEOLINK", np.array([1, 1, 1, 1, 10, 9, 9, 15, 7, 21]), "", "Lithology Code")
    return well


def _build_dynamic_workflow_well() -> Well:
    depth = np.arange(9000.0, 9005.0, 0.5)
    well = Well("WORKFLOW-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.array([35, 38, 42, 45, 50, 60, 75, 85, 95, 110]), "API", "Gamma Ray")
    well.add_curve("RHOB", np.array([2.20, 2.22, 2.24, 2.26, 2.30, 2.36, 2.42, 2.48, 2.55, 2.60]), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.array([0.09, 0.10, 0.11, 0.12, 0.14, 0.18, 0.22, 0.26, 0.28, 0.30]), "v/v", "Neutron Porosity")
    well.add_curve("DTC", np.array([78, 80, 82, 84, 86, 92, 98, 105, 115, 125]), "us/ft", "Compressional Sonic")
    well.add_curve("RDEP", np.array([90, 85, 78, 60, 40, 22, 12, 7, 4, 2]), "ohm.m", "Deep Resistivity")
    well.add_curve("RXO", np.array([45, 42, 40, 34, 28, 18, 9, 5, 3, 1.8]), "ohm.m", "Flushed Resistivity")
    well.add_curve("CALI", np.array([8.5, 8.5, 8.6, 8.5, 8.4, 8.5, 8.6, 8.7, 8.5, 8.5]), "in", "Caliper")
    well.add_curve("BS", np.array([8.5] * 10), "in", "Bit Size")
    well.add_curve("DRHO", np.array([0.01] * 10), "g/cc", "Density Correction")
    return well


def test_hydrocarbon_analysis_adds_requested_curves_and_targets():
    well = _build_analysis_well()

    result = HydrocarbonAnalysis().analyze(well, add_curves=True)
    payload = result.to_dict()["HYDROCARBON_ANALYSIS"]

    assert well.get("CROSSOVER_FLAG") is not None
    assert well.get("VP_VS_RATIO") is not None
    assert well.get("PRIME_GAS_TARGET") is not None
    assert well.get("PRIME_OIL_TARGET") is not None
    assert well.get("NET_PAY") is not None
    np.testing.assert_array_equal(
        well["CROSSOVER_FLAG"].data,
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float),
    )
    assert np.nanmax(well["PRIME_GAS_TARGET"].data) == 1.0
    assert np.nanmax(well["PRIME_OIL_TARGET"].data) == 1.0
    assert "PRIME_GAS_TARGET" in payload["integrated_hydrocarbon_potential"]


def test_input_lithology_is_not_reported_as_missing_litho_fallback():
    well = _build_analysis_well()
    original_curves = set(well.curves)
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[]),
        zone_results=[],
    )

    statuses = _build_reference_derived_log_status(well, workflow_result, original_curves)

    assert "LITHO" not in {item["mnemonic"] for item in statuses}


def test_rmf_missing_reason_does_not_assume_non_obm_mud():
    well = _build_analysis_well()
    reason = _missing_curve_reason(
        "RMF",
        well,
        SimpleNamespace(),
        step_results={},
        skipped_steps=set(),
        computed_curves=set(),
        original_curves=set(well.curves),
    )

    assert "mud type was not OBM" not in reason
    assert "mud type was not provided or was unknown" in reason


def test_hydrocarbon_analysis_rejects_implausible_shear_slowness_for_vpvs():
    well = _build_analysis_well()
    well.add_curve("DTS", np.full(well.n_samples, 4.0e7), "us/ft", "Bad Shear Sonic")

    HydrocarbonAnalysis().analyze(well, add_curves=True)

    assert np.all(np.isnan(well["VP_VS_RATIO"].data))


def test_rock_quality_leaves_invalid_phi_and_perm_as_nan():
    from petrophys.reservoir.rock_quality import calculate_rock_quality

    result = calculate_rock_quality(
        perm_k=np.array([100.0, 0.0, 10.0, np.nan]),
        phi_e=np.array([0.20, 0.20, 0.0, 0.15]),
    )

    assert np.isfinite(result["FZI"][0])
    assert np.isnan(result["RQI"][1])
    assert np.isnan(result["phi_z"][2])
    assert np.isnan(result["FZI"][2])
    assert np.isnan(result["FZI"][3])


def test_invasion_ratio_rejects_tiny_rt_spikes():
    from petrophys.resistivity.invasion import calculate_invasion_ratio

    ratio = calculate_invasion_ratio(
        rxo=np.array([1.0, 50.0, 1.0, 500.0]),
        rt=np.array([1.0, 0.1, 0.0, 1.0]),
    )

    assert ratio[0] == 1.0
    assert np.isnan(ratio[1])
    assert np.isnan(ratio[2])
    assert np.isnan(ratio[3])


def test_multi_mineral_fraction_constraint_enforces_available_volume():
    from petrophys.lithology.multi_mineral_solver import _constrain_mineral_fractions

    constrained = _constrain_mineral_fractions(np.array([0.5, 0.4, 0.2]), 0.65)

    assert np.all(constrained >= 0.0)
    assert np.sum(constrained) <= 0.65 + 1e-12


def test_run_comprehensive_analysis_returns_all_categories_and_curves():
    well = _build_analysis_well()

    result = run_comprehensive_analysis(well, add_curves=True, store_results=True)
    payload = result.to_dict()

    expected_keys = {
        "HYDROCARBON_ANALYSIS",
        "COMPREHENSIVE_LITHOLOGY_ANALYSIS",
        "POROSITY_AND_PORE_SYSTEMS",
        "PERMEABILITY_AND_FLOW_UNITS",
        "SATURATION_AND_FLUID_DISTRIBUTION",
        "NUMERICAL_AND_STATISTICAL_ANALYSIS",
        "WELLBORE_QUALITY_CONTROL",
    }
    assert expected_keys.issubset(payload.keys())
    assert well.get("PHIT_GAS_CORR") is not None
    assert well.get("BADHOLEFLAG") is not None
    assert well.get("ACOUSTIC_IMPEDANCE") is not None
    assert well.get_result("HYDROCARBON_ANALYSIS") is not None


def test_dynamic_workflow_computes_bvw_mhi_and_net_pay_from_canonical_saturation_curves():
    well = _build_dynamic_workflow_well()
    config = DynamicWorkflowConfig(
        bht_f=220.0,
        td_ft=9005.0,
        rw=0.05,
        rmf_at_meas_temp=0.2,
        matrix_type="sandstone",
        lithology="clastic",
        sw_model="archie",
        perm_method="timur",
    )

    result = DynamicWorkflow(config).run(well)
    executed = {step.step for step in result.steps_executed if getattr(step, "success", False)}

    assert {"bvw", "mhi", "net_pay"}.issubset(executed)
    assert well.get("SW") is not None
    assert well.get("SXO") is not None
    assert well.get("SWA") is None
    assert well.get("SWXOA") is None
    assert well.get("BVW") is not None
    assert well.get("MHI") is not None
    assert well.get("NET_PAY") is not None


def test_saturation_auto_routes_by_vcl_into_one_mixed_sw_curve():
    from petrophys.petrophysics.saturation.archie import archie
    from petrophys.petrophysics.saturation.simandoux import simandoux
    from petrophys.petrophysics.saturation.sw_module import SaturationModule

    depth = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    rt = np.array([80.0, 60.0, 12.0, 8.0])
    phie = np.array([0.22, 0.20, 0.18, 0.16])
    vcl = np.array([0.05, 0.15, 0.45, 0.65])

    well = Well("SAT-MIXED-1")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("RDEP", rt, "ohm.m", "Deep Resistivity")
    well.add_curve("PHIE", phie, "v/v", "Effective Porosity")
    well.add_curve("VCL", vcl, "v/v", "Clay Volume")

    result = SaturationModule(a=1.0, m=2.0, n=2.0, model="auto", vcl_cutoff=0.3).calculate(
        well,
        rw=0.05,
        rcl=5.0,
        salinity=35000.0,
        lithology="clastic",
    )

    expected = np.empty_like(rt)
    expected[:2] = archie(rt[:2], 0.05, phie[:2], 1.0, 2.0, 2.0)
    expected[2:] = simandoux(rt[2:], 0.05, phie[2:], vcl[2:], 5.0, 1.0, 2.0, 2.0)

    assert result.sw.shape == depth.shape
    assert result.model == "archie+simandoux"
    assert result.model_counts == {"archie": 2, "simandoux": 2}
    assert np.allclose(result.sw, expected)


def test_temperature_profile_uses_bht_or_fallback_gradient():
    from petrophys.environmental.temperature import calculate_temperature_profile

    well = Well("TEMP-1")
    depth = np.array([0.0, 500.0, 1000.0])
    well.depth = depth
    well.depth_unit = "m"
    well.set_parameter("FMT", 80.0)
    well.set_parameter("FMT__UNIT", "C")
    well.set_parameter("FMTD", 1000.0)
    well.set_parameter("FMTD__UNIT", "m")
    well.add_curve("DEPT", depth, "m", "Depth")

    fallback = calculate_temperature_profile(
        well,
        surface_temp=20.0,
        surface_temp_unit="C",
    )
    assert fallback.source == "fallback_geothermal_gradient"
    assert np.allclose(fallback.temperature, np.array([68.0, 95.0, 122.0]))

    bht_c_well = Well("TEMP-BHT-C")
    bht_c_well.depth = depth
    bht_c_well.depth_unit = "ft"
    bht_c_well.header.bht = 100.0
    bht_c_well.header.td = 1000.0
    bht_c_well.set_parameter("BHT__UNIT", "C")
    bht_c_well.add_curve("DEPT", depth, "ft", "Depth")

    bht_c = calculate_temperature_profile(
        bht_c_well,
        surface_temp=68.0,
        surface_temp_unit="F",
    )
    assert bht_c.source == "linear_gradient_from_BHT"
    assert np.allclose(bht_c.temperature, np.array([68.0, 140.0, 212.0]))


def test_dynamic_workflow_fallback_temperature_and_corrected_rm_curve():
    well = _build_dynamic_workflow_well()
    well.set_parameter("FMT", 80.0)
    well.set_parameter("FMT__UNIT", "C")
    well.set_parameter("FMTD", float(well.depth[-1]))
    well.set_parameter("FMTD__UNIT", "ft")
    config = DynamicWorkflowConfig(
        rm=0.6,
        rm_meas_temp_f=75.0,
        rw=0.05,
        matrix_type="sandstone",
        lithology="clastic",
    )

    result = DynamicWorkflow(config).run(well)
    temp_step = next(step for step in result.steps_executed if step.step == "temperature")

    assert temp_step.method == "fallback_geothermal_gradient"
    assert well.get("TEMP") is not None
    assert well.get("RM") is not None
    assert well.get("RMF") is not None
    assert np.nanmax(well["RM"].data) < 0.6
    assert np.allclose(well["RMF"].data, well["RM"].data * 0.75)


def test_dynamic_workflow_converts_celsius_bht_metadata_for_temperature_profile():
    well = _build_dynamic_workflow_well()
    well.header.bht = 100.0
    well.header.td = float(well.depth[-1])
    well.set_parameter("BHT__UNIT", "C")

    config = DynamicWorkflowConfig(
        rm=0.6,
        rw=0.05,
        matrix_type="sandstone",
        lithology="clastic",
    )

    result = DynamicWorkflow(config).run(well)
    temp_step = next(step for step in result.steps_executed if step.step == "temperature")
    rmf_step = next(step for step in result.steps_executed if step.step == "rmf")

    assert temp_step.method == "linear_gradient_from_BHT"
    assert np.isclose(temp_step.parameters["reference_temperature_f"], 212.0)
    assert np.isclose(well["TEMP"].data[-1], 212.0)
    assert np.isclose(rmf_step.parameters["formation_temp_max_f"], 212.0)


def test_archie_coefficients_cover_lithology_nomenclature():
    from petrophys.analysis.lithology_analysis import GROUND_TRUTH_GROUPS, GROUND_TRUTH_NAMES
    from petrophys.base.constants import ArchieParameterDefaults, DefaultParameters
    from petrophys.petrophysics.cementation import default_archie_params

    labels = set(GROUND_TRUTH_GROUPS.keys()) | set(GROUND_TRUTH_NAMES.values())
    labels.update({
        "SAND",
        "SHALY_SAND",
        "CALCAREOUS_SANDSTONE",
        "SANDY_LIMESTONE",
        "Cross Bedded Sst",
        "Dolomite2",
        "Kaīnite",
        "Aeolian Sandstone",
        "Calcareous Debris Flow",
        "Plutonic Rock",
        "Volcanic Tuff",
        "Porous Limestone",
    })

    for label in labels:
        coeffs = ArchieParameterDefaults.get(label)
        assert coeffs.a > 0
        assert coeffs.m > 0
        assert coeffs.n > 0

    sand = ArchieParameterDefaults.get("SAND")
    assert sand.lithology == "sandstone"
    assert (sand.a, sand.m, sand.n) == (0.81, 2.0, 2.0)
    assert ArchieParameterDefaults.get("Kaīnite").lithology == "evaporite"
    assert ArchieParameterDefaults.get("Dolomite2").lithology == "dolomite"
    assert DefaultParameters.ARCHIE_BY_LITHOLOGY["limestone"] == {"a": 1.0, "m": 2.0, "n": 2.0}

    legacy = default_archie_params("unconsolidated")
    assert (legacy.a, legacy.m, legacy.n) == (0.62, 2.15, 2.0)
