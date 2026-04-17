import numpy as np

from petrophys import Well
from petrophys.analysis import HydrocarbonAnalysis
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
    assert well.get("PAY_FLAG_PETROPHYSICAL") is not None
    assert well.get("PRIME_GAS_TARGET") is not None
    assert well.get("PRIME_OIL_TARGET") is not None
    assert np.nanmax(well["PRIME_GAS_TARGET"].data) == 1.0
    assert np.nanmax(well["PRIME_OIL_TARGET"].data) == 1.0
    assert "PRIME_GAS_TARGET" in payload["integrated_hydrocarbon_potential"]


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


def test_dynamic_workflow_computes_bvw_mhi_and_net_pay_from_sw_aliases():
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
    assert well.get("SWA") is not None
    assert well.get("SWXOA") is not None
    assert well.get("BVW") is not None
    assert well.get("MHI") is not None
    assert well.get("NET_PAY") is not None
