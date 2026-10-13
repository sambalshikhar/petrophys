"""Matrix endpoints, evaporites and effective porosity.

The "current" calculations take density-porosity endpoints from apparent log
densities, regroup GeoLink codes 10 and 26, put the clay porosity on the
endpoints of the PHIT it corrects, read a LAS header shale point (PNSH with
RBSH) as such, and use the root-mean-square in gas crossover when the
hydrocarbon is gas. The "geolink" profile keeps the eval package's
calculations.
"""
import numpy as np
import pytest

from petrophys import Well
from petrophys.model.constants import DefaultParameters, FluidProperties, MatrixProperties
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _endpoints(result):
    return {
        item["lithology_group"]: item
        for item in _step(result, "porosity").parameters["matrix_endpoints"]
    }


def _run(well, profile, **settings):
    config = DynamicWorkflowConfig(
        rw=0.05,
        skip_steps=["temperature", "rmf", "vclay", "lithology"],
        calculation_profile=profile,
        **settings,
    )
    result = DynamicWorkflow(config).run(well)
    assert result.success, result.errors
    return result


def _geolink_well(codes, rhob, nphi, vcl, name="ENDPOINTS"):
    n = len(codes)
    well = Well(name)
    well.depth = np.arange(1000.0, 1000.0 + 0.5 * n, 0.5)
    well.depth_unit = "m"
    well.add_curve("LITHOLOGY_GEOLINK", np.asarray(codes, dtype=float), "", "Lithology")
    well.add_curve("RHOB", np.asarray(rhob, dtype=float), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.asarray(nphi, dtype=float), "v/v", "Neutron")
    well.add_curve("VCL", np.asarray(vcl, dtype=float), "v/v", "Clay volume")
    well.add_curve("GR", 20.0 + 100.0 * np.asarray(vcl, dtype=float), "API", "Gamma Ray")
    well.add_curve("RDEP", np.full(n, 20.0), "ohm.m", "Deep Resistivity")
    return well


# -- Mineral table -------------------------------------------------------------

_ATOMS = {  # atomic number, atomic mass
    "H": (1, 1.008), "O": (8, 15.999), "Na": (11, 22.990), "Mg": (12, 24.305),
    "S": (16, 32.06), "Cl": (17, 35.45), "K": (19, 39.098), "Ca": (20, 40.078),
}
_FORMULAS = {
    "halite": {"Na": 1, "Cl": 1},
    "sylvite": {"K": 1, "Cl": 1},
    "carnallite": {"K": 1, "Mg": 1, "Cl": 3, "H": 12, "O": 6},
    "anhydrite": {"Ca": 1, "S": 1, "O": 4},
    "gypsum": {"Ca": 1, "S": 1, "O": 6, "H": 4},
    "polyhalite": {"K": 2, "Mg": 1, "Ca": 2, "S": 4, "O": 18, "H": 4},
    "kainite": {"Mg": 1, "S": 1, "O": 7, "K": 1, "Cl": 1, "H": 6},
}


@pytest.mark.parametrize("mineral", sorted(_FORMULAS))
def test_evaporite_log_density_follows_the_density_tool_response(mineral):
    """rho_log = 1.0704 * rho_b * (2Z/A) - 0.1883 (Schlumberger, 1989)."""
    formula = _FORMULAS[mineral]
    z = sum(_ATOMS[element][0] * count for element, count in formula.items())
    a = sum(_ATOMS[element][1] * count for element, count in formula.items())
    props = MatrixProperties.get(mineral)

    rho_log = 1.0704 * props.density * 2.0 * z / a - 0.1883

    assert props.log_density == pytest.approx(rho_log, abs=0.015)
    assert MatrixProperties.get_log_density(mineral) == props.log_density


def test_log_density_is_the_density_porosity_endpoint():
    # Chartbook apparent log densities; the grain densities stay for solvers.
    assert MatrixProperties.get_log_density("halite") == pytest.approx(2.032)
    assert MatrixProperties.get_log_density("salt") == pytest.approx(2.032)
    assert MatrixProperties.get_log_density("sylvite") == pytest.approx(1.863)
    assert MatrixProperties.get_log_density("carnallite") == pytest.approx(1.570)
    assert MatrixProperties.get_log_density("gypsum") == pytest.approx(2.351)
    assert MatrixProperties.get_log_density("anhydrite") == pytest.approx(2.977)
    assert MatrixProperties.get_density("halite") == pytest.approx(2.17)
    # Minerals without a tabulated log density use their grain density.
    assert MatrixProperties.get_log_density("sandstone") == pytest.approx(2.65)
    assert MatrixProperties.get_log_density("dolomite") == pytest.approx(2.87)


def test_every_matrix_is_faster_than_the_pore_fluid():
    """Wyllie needs dt_ma < dt_fl; coal reads 110-140 us/ft (bituminous)."""
    water = FluidProperties.get("water").slowness
    for name in MatrixProperties.list_minerals():
        slowness = MatrixProperties.get(name).slowness
        if np.isfinite(slowness):  # kainite has no published slowness
            assert slowness < water, name
    assert 110.0 <= MatrixProperties.get_slowness("coal") <= 140.0


def test_clay_u_is_pef_times_density():
    from petrophys.interpretation.lithology.matrix_properties import MINERAL_PROPERTIES

    clay = MINERAL_PROPERTIES["clay"]
    assert clay["u"] == pytest.approx(clay["pef"] * clay["density"], abs=0.01)


def test_oil_hydrogen_index_matches_its_density():
    """HI = 9 n rho / (12 + n) for a CH_n liquid (Schlumberger, 1989)."""
    oil = FluidProperties.get("oil")
    assert oil.hydrogen_index == pytest.approx(9 * 2 * oil.density / 14.0, abs=0.05)
    assert DefaultParameters.OIL_HYDROGEN_INDEX == pytest.approx(oil.hydrogen_index)


def test_oil_leaves_the_neutron_unchanged_in_the_hc_correction():
    from petrophys.interpretation.porosity.hydrocarbon_correction.neutron_correction import (
        correct_neutron_hc,
    )

    assert correct_neutron_hc(0.25, 0.25, 0.7, "oil", vcl=0.1) == pytest.approx(0.25)
    assert correct_neutron_hc(0.25, 0.25, 0.7, "gas", vcl=0.1) > 0.25


# -- Porosity helpers ------------------------------------------------------------

def test_gas_crossover_takes_the_root_mean_square():
    """A 20 p.u. gas sand (Sxo 0.7, rho_g 0.2, HI 0.4): PHID 0.229, PHIN 0.164."""
    from petrophys.interpretation.porosity.total_porosity import total_porosity

    phid, phin = 0.229, 0.164
    rms = np.sqrt((phid**2 + phin**2) / 2)  # Gaymard and Poupon (1968)
    assert rms == pytest.approx(0.1992, abs=1e-4)
    assert total_porosity(phin, phid) == pytest.approx(rms)
    assert total_porosity(np.array([phin]), np.array([phid]))[0] == pytest.approx(rms)
    assert total_porosity(phin, phid, gas_method="density") == pytest.approx(phid)
    # Outside crossover the density-weighted average applies either way.
    assert total_porosity(0.20, 0.18) == pytest.approx(0.6 * 0.18 + 0.4 * 0.20)
    with pytest.raises(ValueError, match="gas_method"):
        total_porosity(phin, phid, gas_method="neutron")


def test_sonic_compaction_factor_is_applied_before_the_limits():
    from petrophys.interpretation.porosity.sonic_porosity import sonic_porosity

    # (140 - 55.5) / (189 - 55.5) / 1.3 = 0.4869, not min(0.633, 0.5) / 1.3.
    assert sonic_porosity(140.0, "sandstone", compaction_factor=1.3) == pytest.approx(0.4869, abs=1e-4)
    assert sonic_porosity(200.0, "sandstone", compaction_factor=1.3) == pytest.approx(0.5)
    assert sonic_porosity(100.0, "sandstone") == pytest.approx((100.0 - 55.5) / (189.0 - 55.5))


def test_elastic_properties_accept_scalars():
    from petrophys.interpretation.rock_physics.elastic import calculate_elastic_properties

    # Vp 3000 m/s, Vs 1500 m/s, rho 2500 kg/m3.
    props = calculate_elastic_properties(2.5, 101.6, 203.2)
    assert isinstance(props.poisson_ratio, float)
    assert props.vp_m_s == pytest.approx(3000.0)
    assert props.poisson_ratio == pytest.approx(1.0 / 3.0)
    assert props.shear_mod_gpa == pytest.approx(5.625)
    assert props.bulk_mod_gpa == pytest.approx(15.0)
    assert props.young_mod_gpa == pytest.approx(15.0)
    assert props.acoustic_impedance == pytest.approx(7.5e6)
    arrays = calculate_elastic_properties(np.array([2.5]), np.array([101.6]), np.array([203.2]))
    np.testing.assert_allclose(arrays.young_mod_gpa, [15.0])


# -- Workflow --------------------------------------------------------------------

def test_evaporite_endpoints_use_log_densities_under_current():
    # Halite (code 21) at 2.04 g/cc, carnallite (33) at 1.58 g/cc, then sand.
    codes = [21] * 10 + [33] * 10 + [1] * 10
    rhob = [2.04] * 10 + [1.58] * 10 + [2.30] * 10
    nphi = [-0.02] * 10 + [0.62] * 10 + [0.22] * 10
    vcl = [0.0] * 20 + [0.05] * 10

    current = _run(_geolink_well(codes, rhob, nphi, vcl), "current")
    benchmark = _run(_geolink_well(codes, rhob, nphi, vcl), "geolink")

    salt = slice(0, 10)
    carnallite = slice(10, 20)
    # (2.032 - 2.04) / 1.032 < 0: no density porosity in salt.
    assert np.all(current.well.get("PHID").data[salt] == 0.0)
    assert np.all(current.well.get("PHIE").data[carnallite] < 0.02)
    assert _endpoints(current)["HALITE"]["rho_ma"] == pytest.approx(2.032)
    assert _endpoints(current)["CARNALLITE"]["rho_ma"] == pytest.approx(1.570)
    # The "geolink" profile keeps halite's grain density for every salt code.
    np.testing.assert_allclose(benchmark.well.get("PHID").data[salt], (2.17 - 2.04) / 1.17)
    assert _endpoints(benchmark)["HALITE"]["rho_ma"] == pytest.approx(2.17)
    assert "CARNALLITE" not in _endpoints(benchmark)


def test_geolink_codes_10_and_26_and_coal_endpoints_by_profile():
    # Porous Limestone (10), Gap (26), Coal (18) and sandstone (1).
    codes = [10] * 6 + [26] * 6 + [18] * 6 + [1] * 6
    n = len(codes)
    well_c = _geolink_well(codes, [2.45] * n, [0.18] * n, [0.1] * n)
    well_b = _geolink_well(codes, [2.45] * n, [0.18] * n, [0.1] * n)
    well_c.add_curve("DTC", np.full(n, 90.0), "us/ft", "Sonic")
    well_b.add_curve("DTC", np.full(n, 90.0), "us/ft", "Sonic")

    current = _endpoints(_run(well_c, "current"))
    benchmark = _endpoints(_run(well_b, "geolink"))

    assert current["LIMESTONE"]["rho_ma"] == pytest.approx(2.71)
    assert "EVAPORITE" not in current  # Gap samples keep the configured matrix
    assert current["COAL"]["dt_ma"] == pytest.approx(120.0)
    assert set(benchmark) == {"SANDSTONE", "EVAPORITE", "COAL"}
    assert benchmark["EVAPORITE"]["matrix_key"] == "gypsum"
    assert benchmark["EVAPORITE"]["dt_ma"] == pytest.approx(83.0)
    assert benchmark["COAL"]["dt_ma"] == pytest.approx(250.0)
    # The run records its grouping on the lithology curve for the reports.
    assert well_c.get("LITHOLOGY_GEOLINK").metadata.custom["category_groups"][10] == "limestone"
    assert "category_groups" not in well_b.get("LITHOLOGY_GEOLINK").metadata.custom


def test_current_code_groups_agree_with_the_lithology_code_classes():
    from petrophys.interpretation.lithology.matrix_inference import LITHOLOGY_CODE_CLASS
    from petrophys.workflow.dynamic_workflow import (
        _CURRENT_GROUND_TRUTH_CODE_TO_GROUP as groups,
        _CURRENT_LITHOLOGY_GROUP_TO_MATRIX as matrices,
    )

    for code, family in LITHOLOGY_CODE_CLASS.items():
        if family == "evaporite":
            # Every evaporite code reads against its own apparent log density.
            assert MatrixProperties.get(matrices[groups[code]]).log_density is not None, code
    assert LITHOLOGY_CODE_CLASS[10] == "carbonate" and groups[10] == "limestone"
    assert groups[26] is None


def test_lithology_diagnostics_list_the_codes_of_each_group():
    from petrophys.evaluation.diagnostics.lithology_analysis import (
        CURRENT_GROUND_TRUTH_GROUPS,
        GROUND_TRUTH_GROUPS,
        _ground_truth_groups_text,
    )

    # The eval package's text.
    assert _ground_truth_groups_text(GROUND_TRUTH_GROUPS) == (
        "GT_{GROUP} for mapped codes: limestone 9/11/12/13/16, dolomite 15, "
        "anhydrite 22, coal 18, calcareous shale 14, halite 21/27/32/33/34, "
        "sandstone 1/2/3/4/10/30/35, shale 7/8/25, silt 5/6, coarse clastic 17/31/36, "
        "evaporite 26, igneous 19/20/23/24/28, chert 29; otherwise NO_LITHOLOGY_DATA."
    )
    current = _ground_truth_groups_text(CURRENT_GROUND_TRUTH_GROUPS)
    assert "limestone 9/10/11/12/13/16" in current
    assert "sandstone 1/2/3/4/30/35" in current
    assert "evaporite" not in current


@pytest.mark.parametrize("workbook", [None, {
    1: {"attribute": "Sandstone", "lithology_attribute": "Sand"},
    10: {"attribute": "Porous Limestone", "lithology_attribute": "Calcareous Sandstone"},
    26: {"attribute": "Gap", "lithology_attribute": "Gypsum"},
}])
def test_context_export_names_codes_10_and_26_as_the_profile_treats_them(workbook):
    from petrophys.reporting import reference_export as reference
    from petrophys.workflow.dynamic_workflow import _rerun_calculations

    codes = [10] * 6 + [26] * 6 + [1] * 6
    n = len(codes)
    exported = {}
    for profile in ("current", "geolink"):
        # The context export runs the "geolink" profile with the re-run's calculations.
        with _rerun_calculations():
            result = _run(_geolink_well(codes, [2.45] * n, [0.18] * n, [0.1] * n), profile)
        mapping = reference._run_lithology_mapping(workbook, result)
        segments = reference._build_lithology_group_segments(
            result.well, minimum_group_thickness=0.0, lithology_mapping=mapping
        )
        info = reference._build_reference_lithology_info(result.well, lithology_mapping=mapping)
        lithology = reference._category_analysis_result("lithology", result.well, result)
        exported[profile] = (
            [segment["lithology"] for segment in segments],
            (info["valid_lithology_count"], info["nan_count"]),
            list(dict.fromkeys(lithology.sections["lithology_distribution_classification"].raw_labels)),
        )

    # "current": Porous Limestone, a carbonate, and Gap, no lithology, as the
    # calculations treat them.
    assert exported["current"] == (
        ["POROUS LIMESTONE", "NO_LITHOLOGY_DATA", "SAND"],
        (12, 6),
        ["GT_LIMESTONE", "NO_LITHOLOGY_DATA", "GT_SANDSTONE"],
    )
    # The "geolink" profile keeps the eval package's names and groups.
    assert exported["geolink"] == (
        ["CALCAREOUS SANDSTONE", "GYPSUM", "SAND"],
        (18, 0),
        ["GT_SANDSTONE", "GT_EVAPORITE"],
    )


def _shaly_sand_well(name="SHALY-SAND"):
    """Sands of 20 p.u. effective porosity mixed linearly with a shale.

    rho_b = (1 - phi - V) 2.65 + V 2.45 + phi 1.0 and
    NPHI = (1 - phi - V)(-0.02) + V 0.35 + phi, with V the shale volume.
    """
    phi = 0.20
    v = np.r_[np.zeros(4), np.full(4, 0.1), np.full(4, 0.2), np.full(4, 0.3), np.ones(8)]
    porosity = np.where(v < 1.0, phi, 0.0)
    rhob = (1 - porosity - v) * 2.65 + v * 2.45 + porosity * 1.0
    nphi = (1 - porosity - v) * -0.02 + v * 0.35 + porosity
    codes = np.where(v < 1.0, 1.0, 8.0)
    return _geolink_well(codes, rhob, nphi, v, name=name), v


def test_clay_porosity_is_the_shale_porosity_on_the_reservoir_endpoints():
    well, v = _shaly_sand_well()
    result = _run(well, "current")
    porosity = _step(result, "porosity").parameters

    # Shale on sandstone endpoints: 0.6 * 0.20/1.65 + 0.4 * (0.35 + 0.02).
    phi_cl = 0.6 * (2.65 - 2.45) / 1.65 + 0.4 * 0.37
    assert porosity["phi_cl"] == pytest.approx(phi_cl)
    assert porosity["phi_cl_source"] == "lithology_shale_average_phit_nd_on_reservoir_matrix"
    assert porosity["phi_cl_matrix"] == "sandstone"
    # Linear mixing: every shaly sand keeps the clean sand's porosity.
    phie = result.well.get("PHIE").data
    sands = v < 1.0
    np.testing.assert_allclose(phie[sands], phie[0], atol=1e-9)
    # PHIN = NPHI - nphi_ma scales phi by 1.02, so PHIT(clean) = 1.008 phi.
    assert phie[0] == pytest.approx(1.008 * 0.20)
    assert _step(result, "saturation").parameters["phi_cl"] == pytest.approx(phi_cl)


def test_the_rerun_calculations_keep_the_wet_shale_clay_porosity():
    # 34/10-23 is one of the eval package's re-run wells: the "geolink" profile
    # gives it the re-run's calculations.
    well, v = _shaly_sand_well(name="34/10-23")
    result = _run(well, "geolink")
    porosity = _step(result, "porosity").parameters

    # The shale samples' PHIT on the wet-shale point (2.58 g/cc, NPHI 0.30).
    phi_cl = 0.6 * (2.58 - 2.45) / 1.58 + 0.4 * (0.35 - 0.30)
    assert porosity["phi_cl"] == pytest.approx(phi_cl)
    assert porosity["phi_cl_source"] == "lithology_shale_average_phit_nd"
    assert "phi_cl_matrix" not in porosity
    phie = result.well.get("PHIE").data
    shale_phit_on_sand = 0.6 * (2.65 - 2.45) / 1.65 + 0.4 * 0.37
    expected = 1.008 * 0.20 + 0.3 * (shale_phit_on_sand - phi_cl)
    assert phie[12] == pytest.approx(expected)  # V = 0.3: 0.045 too high


# -- LAS header shale point: PNSH ("Neutron Shale") and RBSH ("Shale Density") ----------


def _header_well(**parameters):
    well, v = _shaly_sand_well()
    for name, value in parameters.items():
        well.set_parameter(name, value)
    return well, v


@pytest.mark.parametrize("zone_suffix, label", [("", "LAS"), ("_Z1", "zoned LAS")])
def test_current_reads_pnsh_with_rbsh_as_a_neutron_density_shale_point(zone_suffix, label):
    # As in 15_9-15.las: "PNSH .frac 0.31 : Neutron Shale", "RBSH .g/cc 2.46 : Shale Density".
    well, v = _header_well(**{"PNSH" + zone_suffix: 0.31, "RBSH" + zone_suffix: 2.46})
    result = _run(well, "current")
    porosity = _step(result, "porosity").parameters

    # On the sandstone endpoints: PHID = (2.65 - 2.46) / 1.65 = 0.1152 and
    # PHIN = 0.31 + 0.02 = 0.33, so PHIT = 0.6 * 0.1152 + 0.4 * 0.33 = 0.2011.
    phi_cl = 0.6 * (2.65 - 2.46) / 1.65 + 0.4 * (0.31 + 0.02)
    assert phi_cl == pytest.approx(0.2011, abs=1e-4)
    assert porosity["phi_cl"] == pytest.approx(phi_cl)
    assert porosity["phi_cl_source"] == f"{label} PNSH and RBSH shale point"
    assert porosity["phi_cl_shale_point"]["PHIN"] == pytest.approx(0.33)
    phit = result.well.get("PHIT").data
    np.testing.assert_allclose(result.well.get("PHIE").data[:16], phit[:16] - v[:16] * phi_cl)


def test_current_sets_pnsh_without_rbsh_aside():
    # As in 15_9-17.las, whose only shale parameter is "PNSH 0.4 : Neutron Shale".
    well, v = _header_well(PNSH=0.4)
    porosity = _step(_run(well, "current"), "porosity")

    # The shale samples on sandstone endpoints, as without the header.
    assert porosity.parameters["phi_cl"] == pytest.approx(0.6 * (2.65 - 2.45) / 1.65 + 0.4 * 0.37)
    assert porosity.parameters["phi_cl_source"] == "lithology_shale_average_phit_nd_on_reservoir_matrix"
    assert porosity.parameters["phi_cl_header_note"].startswith("LAS PNSH = 0.4 set aside")
    assert porosity.parameters["phi_cl_header_note"] in porosity.decisions


def test_a_configured_clay_porosity_wins_over_the_header_shale_point():
    well, _ = _header_well(PNSH=0.31, RBSH=2.46)
    porosity = _step(_run(well, "current", phi_cl=0.25), "porosity").parameters
    assert porosity["phi_cl"] == pytest.approx(0.25)
    assert porosity["phi_cl_source"] == "config"


# The eval package's main run and its re-run (its re-run wells and context).
@pytest.mark.parametrize("calculations", ["main_run", "rerun"])
def test_benchmark_takes_pnsh_as_the_clay_porosity(calculations):
    from petrophys.workflow.dynamic_workflow import _rerun_calculations

    well, _ = _header_well(PNSH=0.31, RBSH=2.46)
    with _rerun_calculations(calculations == "rerun"):
        porosity = _step(_run(well, "geolink", phi_cl=0.25), "porosity").parameters
    assert porosity["phi_cl"] == pytest.approx(0.31)
    assert porosity["phi_cl_source"] == "LAS PNSH parameter"
    assert "phi_cl_shale_point" not in porosity and "phi_cl_header_note" not in porosity


def test_header_pnsh_is_a_phi_cl_setting_only_under_the_benchmark_profile():
    from petrophys.workflow.zoned_batch import _extract_global_parameter_overrides

    well, _ = _header_well(PNSH=0.31, RBSH=2.46)
    current, unmapped = _extract_global_parameter_overrides(well, calculation_profile="current")
    assert "phi_cl" not in current and "PNSH" not in unmapped
    assert current["rho_cl"] == pytest.approx(2.46)
    overrides, _ = _extract_global_parameter_overrides(well, calculation_profile="geolink")
    assert overrides["phi_cl"] == pytest.approx(0.31)


# -- External potash intervals -----------------------------------------------------------


@pytest.mark.parametrize("profile", ["current", "geolink"])
def test_external_sylvite_and_potash_labels(profile):
    import pandas as pd

    n = 12
    well = Well("POTASH")
    well.depth = np.arange(1000.0, 1000.0 + 0.5 * n, 0.5)
    well.depth_unit = "m"
    # Sylvite reads 1.863 g/cm3 on the density log (Schlumberger, 1972).
    well.add_curve("RHOB", np.r_[np.full(8, 1.863), np.full(4, 2.30)], "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.r_[np.full(8, -0.02), np.full(4, 0.20)], "v/v", "Neutron")
    well.add_curve("VCL", np.zeros(n), "v/v", "Clay volume")
    well.add_curve("RDEP", np.full(n, 200.0), "ohm.m", "Deep Resistivity")
    intervals = pd.DataFrame({
        "top": [1000.0, 1002.0, 1004.0],
        "bottom": [1002.0, 1004.0, 1006.0],
        "lithology": ["Sylvite", "Potash", "Sandstone"],
    })
    result = _run(well, profile, lithology_intervals=intervals)
    report = _step(result, "input_context").parameters
    phid = result.well.get("PHID").data

    if profile == "current":
        assert report["unmapped_labels"] == []
        assert _endpoints(result)["SYLVINITE"]["rho_ma"] == pytest.approx(1.863)
        np.testing.assert_allclose(phid[:8], 0.0, atol=1e-9)
        assert not np.any(result.well.get("NET_PAY").data[:8] > 0.5)
    else:
        # The benchmark groups have no potash: the labels stay unmapped.
        assert report["unmapped_labels"] == ["Potash", "Sylvite"]
        np.testing.assert_allclose(phid[:8], (2.65 - 1.863) / 1.65)


def test_gas_crossover_counts_by_profile_and_hydrocarbon():
    # Crossover sand: PHID 0.229 and PHIN 0.164 on sandstone endpoints.
    n = 8
    rhob = np.full(n, 2.65 - 0.229 * 1.65)
    nphi = np.full(n, 0.164 - 0.02)

    def run(profile, **settings):
        return _run(_geolink_well([1] * n, rhob, nphi, [0.0] * n), profile, **settings)

    gas = run("current", hc_type="gas")
    counts = _step(gas, "porosity").parameters["phit_source_counts"]
    assert counts["rms_from_gas_crossover"] == n
    # Gaymard and Poupon (1968): sqrt((0.229^2 + 0.164^2) / 2) = 0.1992.
    np.testing.assert_allclose(gas.well.get("PHIT").data, np.sqrt((0.229**2 + 0.164**2) / 2))
    # Oil (the default) leaves the neutron log almost untouched: the crossover is
    # not gas, and PHIT is density porosity, as under the "geolink" profile.
    for result in (run("current"), run("geolink", hc_type="gas"), run("geolink")):
        counts = _step(result, "porosity").parameters["phit_source_counts"]
        assert counts["density_from_gas_crossover"] == n
        assert "rms_from_gas_crossover" not in counts
        np.testing.assert_allclose(result.well.get("PHIT").data, 0.229)


def test_steps_porosity_takes_the_crossover_rule_of_its_hydrocarbon_type():
    from petrophys.workflow import Workflow, steps

    n = 8
    rhob = np.full(n, 2.65 - 0.229 * 1.65)
    nphi = np.full(n, 0.164 - 0.02)

    def phit(**options):
        well = _geolink_well([1] * n, rhob, nphi, [0.0] * n)
        run = Workflow("x", [steps.porosity(vcl_curve=None, **options)]).run(well)
        return run.well.get("PHIT").data

    np.testing.assert_allclose(phit(), 0.229)
    np.testing.assert_allclose(phit(hc_type="gas"), np.sqrt((0.229**2 + 0.164**2) / 2))


def test_rms_porosity_counts_a_negative_neutron_porosity_as_zero():
    from petrophys.interpretation.porosity.total_porosity import total_porosity

    # sqrt((0.30^2 + 0^2) / 2) = 0.2121, not sqrt((0.30^2 + 0.05^2) / 2) = 0.2151.
    assert total_porosity(-0.05, 0.30) == pytest.approx(0.30 / np.sqrt(2))
    assert total_porosity(0.05, 0.30) == pytest.approx(np.sqrt((0.30**2 + 0.05**2) / 2))
    np.testing.assert_allclose(
        total_porosity(np.array([-0.05, 0.05]), np.array([0.30, 0.30])),
        [0.30 / np.sqrt(2), np.sqrt((0.30**2 + 0.05**2) / 2)],
    )
    assert total_porosity(-0.05, 0.30, method="rms") == pytest.approx(0.30 / np.sqrt(2))
    assert total_porosity(-0.05, 0.30, gas_method="density") == pytest.approx(0.30)


def test_context_rebuilds_the_groups_the_workflow_used():
    from petrophys.reporting.reference_export import (
        _compact_curve_note,
        _porosity_curve_route_reason,
        _segment_lithology_groups_for_saturation,
    )
    from petrophys.workflow.dynamic_workflow import StepResult

    def groups(profile):
        well = _geolink_well([10, 26, 1, 1], [2.45] * 4, [0.18] * 4, [0.1] * 4)
        _run(well, profile)
        return list(_segment_lithology_groups_for_saturation(well, "clastic"))

    assert groups("current") == ["limestone", "clastic", "sandstone", "sandstone"]
    assert groups("geolink") == ["sandstone", "evaporite", "sandstone", "sandstone"]

    def porosity_step(gas_key):
        return StepResult(step="porosity", success=True, parameters={
            "phit_method": "adaptive_nd", "phit_source_counts": {gas_key: 3, "weighted_nd": 5},
        })

    rms, density = porosity_step("rms_from_gas_crossover"), porosity_step("density_from_gas_crossover")
    assert "sqrt((PHID^2 + PHIN^2) / 2) where PHID > PHIN + 0.05" in _porosity_curve_route_reason("PHIT", rms)
    assert "PHID where PHIN is missing or PHID > PHIN + 0.05" in _porosity_curve_route_reason("PHIT", density)
    assert "sqrt((PHID^2+PHIN^2)/2) where PHID>PHIN+0.05" in _compact_curve_note("PHIT", "computed", "", rms, None)
    assert _compact_curve_note("PHIT", "computed", "", density, None).startswith(
        "PHIT: PHID where PHIN is missing or PHID>PHIN+0.05; PHIN where PHID is missing;"
    )
