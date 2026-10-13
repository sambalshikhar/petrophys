"""The ``benchmark`` calculation profile: the GeoLink eval package's calculations.

The package was made in a main run, which interpreted 30 of its 32 wells, and a
re-run, which interpreted 15/9-17 and 34/10-23 and wrote the context of every
well. The "geolink" profile gives those two wells and the context export the
re-run's calculations and every other well the main run's. The full
reproduction needs the GeoLink LAS files and is verified outside the test suite
with ``scripts/generate_benchmark_context.py --from-raw``. These tests pin the
individual behaviours the profile switches, which well gets which calculations,
and that the default profile is unaffected.
"""
from pathlib import Path
import runpy

import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.electrical.invasion.invasion_module import InvasionModule
from petrophys.interpretation.saturation.sw_module import SaturationModule
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.schema import describe_setting


def _well(n_samples: int = 60, name: str = "PROFILE-TEST") -> Well:
    well = Well(name)
    well.depth = np.arange(2000.0, 2000.0 + 0.5 * n_samples, 0.5)
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(20.0, 130.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.30, 0.05, n_samples), "v/v", "Neutron")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n_samples), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(40.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    well.add_curve("CALI", np.full(n_samples, 8.7), "in", "Caliper")
    return well


def _run(profile: str, well: Well = None, **settings):
    config = DynamicWorkflowConfig(
        rw=0.05,
        skip_steps=["temperature", "rmf"],
        calculation_profile=profile,
        **settings,
    )
    result = DynamicWorkflow(config).run(well if well is not None else _well())
    assert result.success, result.errors
    return result


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def test_default_profile_and_validation():
    assert DynamicWorkflowConfig().calculation_profile == "current"
    assert DynamicWorkflowConfig(calculation_profile="GEOLINK").calculation_profile == "geolink"
    with pytest.raises(ValueError, match="calculation_profile"):
        DynamicWorkflowConfig(calculation_profile="may2026")
    assert describe_setting("calculation_profile")["choices"] == ["current", "geolink"]


def test_benchmark_rerun_is_not_a_profile():
    """The re-run's calculations come with the wells and the context export, not a profile."""
    from petrophys.mcp.session import default_calculation_profile

    with pytest.raises(ValueError, match="'benchmark_rerun' is not valid; use one of: current, geolink$"):
        DynamicWorkflowConfig(calculation_profile="benchmark_rerun")
    with pytest.raises(ValueError, match="not a calculation profile; use one of: current, geolink$"):
        default_calculation_profile("benchmark_rerun")


def test_the_former_benchmark_profile_points_to_geolink():
    from petrophys.evaluation.diagnostics import PorosityAnalysis
    from petrophys.mcp.session import default_calculation_profile

    renamed = 'calculation_profile "benchmark" is now "geolink"'
    with pytest.raises(ValueError, match=renamed):
        DynamicWorkflowConfig(calculation_profile="benchmark")
    with pytest.raises(ValueError, match=renamed):
        default_calculation_profile("Benchmark")
    with pytest.raises(ValueError, match='class_tables "benchmark" is now "geolink"'):
        PorosityAnalysis(class_tables="benchmark")


def test_benchmark_duplicates_the_bad_hole_flag():
    current = _run("current")
    benchmark = _run("geolink")

    assert current.well.get("BADHOLEFLAG") is None
    flag = benchmark.well.get("BADHOLEFLAG")
    assert flag is not None
    np.testing.assert_array_equal(flag.data, benchmark.well.get("BAD_HOLE").data)


def test_benchmark_takes_clay_porosity_from_every_shale_sample():
    def phi_cl(profile):
        well = _well()
        well.add_curve(
            "LITHOLOGY_GEOLINK",
            np.where(np.arange(well.n_samples) < 40, 1.0, 7.0),
            "",
            "Lithology",
        )
        return _step(_run(profile, well), "porosity").parameters

    current = phi_cl("current")
    benchmark = phi_cl("geolink")

    assert benchmark["phi_cl_source"] == "all_shale_phit"
    assert current["phi_cl_source"] != "all_shale_phit"


def test_benchmark_ignores_core_corrected_curves_downstream():
    def sources(profile):
        well = _well()
        well.add_curve("PHIC", np.full(well.n_samples, 0.18), "v/v", "Core porosity")
        well.add_curve("KH_CORE", np.full(well.n_samples, 50.0), "mD", "Core permeability")
        result = _run(profile, well)
        assert result.well.get("PHIE_CC") is not None, "core correction should still run"
        return _step(result, "rock_quality").decisions[0]

    assert "PHIE_CC" in sources("current")
    assert "PHIE_CC" not in sources("geolink")


def test_benchmark_uses_one_archie_coefficient_set():
    def archie(profile):
        well = _well()
        # Mixed lithology: sandstone (a=0.81) over limestone (a=1.0).
        well.add_curve(
            "LITHOLOGY_GEOLINK",
            np.where(np.arange(well.n_samples) < 30, 1.0, 11.0),
            "",
            "Lithology",
        )
        return _step(_run(profile, well), "saturation").parameters["a"]

    assert archie("geolink") == 0.81
    assert archie("current") == "lithology_dependent"


def test_invasion_medium_proxy_and_curve_merge_are_optional():
    well = Well("INVASION")
    well.depth = np.arange(1000.0, 1004.0)
    well.depth_unit = "m"
    well.add_curve("RSHA", np.array([2.0, np.nan, 2.0, 2.0]), "ohm.m", "Shallow")
    well.add_curve("RMED", np.array([3.0, 3.0, 3.0, 3.0]), "ohm.m", "Medium")
    well.add_curve("RDEP", np.array([4.0, 4.0, 4.0, 4.0]), "ohm.m", "Deep")

    filled = InvasionModule().analyze(well).ratio
    kept = InvasionModule(medium_deep_proxy=False, depthwise_curve_merge=False).analyze(well).ratio

    assert np.isfinite(filled[1])
    assert not np.isfinite(kept[1])


def test_sxo_quicklook_fallback_modes():
    well = Well("SXO")
    well.depth = np.arange(1000.0, 1004.0)
    well.depth_unit = "m"
    well.add_curve("RDEP", np.full(4, 10.0), "ohm.m", "Deep")
    well.add_curve("RSHA", np.array([5.0, np.nan, 5.0, 5.0]), "ohm.m", "Shallow")
    well.add_curve("PHIE", np.full(4, 0.2), "v/v", "Effective porosity")
    well.add_curve("VCL", np.full(4, 0.1), "v/v", "Clay volume")

    per_sample = SaturationModule().calculate(well, rw=0.05, rmf=0.1)
    whole_curve = SaturationModule(sxo_quicklook_fallback="whole_curve").calculate(well, rw=0.05, rmf=0.1)

    assert np.isfinite(per_sample.sxo[1])
    assert not np.isfinite(whole_curve.sxo[1])
    # With no invaded-zone inputs at all, both still fall back for the curve.
    bare = Well("BARE")
    bare.depth = well.depth
    bare.depth_unit = "m"
    bare.add_curve("RDEP", np.full(4, 10.0), "ohm.m", "Deep")
    bare.add_curve("PHIE", np.full(4, 0.2), "v/v", "Effective porosity")
    bare.add_curve("VCL", np.full(4, 0.1), "v/v", "Clay volume")
    assert np.all(np.isfinite(SaturationModule(sxo_quicklook_fallback="whole_curve").calculate(bare, rw=0.05).sxo))


def test_benchmark_keeps_mhi_without_the_invaded_zone_gate():
    def mhi(profile):
        well = _well()
        well.add_curve("RSHA", np.linspace(20.0, 2.0, well.n_samples), "ohm.m", "Shallow")
        well.add_curve("RMF", np.full(well.n_samples, 0.1), "ohm.m", "Rmf")
        result = _run(profile, well)
        curve = result.well.get("MHI")
        return None if curve is None else int(np.count_nonzero(np.isfinite(curve.data)))

    benchmark, current = mhi("geolink"), mhi("current")
    assert benchmark is not None and current is not None
    assert benchmark >= current


def _core_well() -> Well:
    well = _well()
    well.add_curve("PHIC", np.full(well.n_samples, 0.18), "v/v", "Core porosity")
    well.add_curve("KH_CORE", np.full(well.n_samples, 50.0), "mD", "Core permeability")
    return well


@pytest.mark.parametrize(
    "profile, saturation, rock_quality, bvw",
    [
        ("current", "PHIE_CC", ["PERM_CC", "PHIE_CC"], "PHIE_CC"),
        ("geolink", "PHIE", ["PERM", "PHIE"], "PHIE"),
    ],
)
def test_which_porosity_each_step_reads(profile, saturation, rock_quality, bvw):
    """Pin the porosity and permeability source of every downstream step.

    The default must keep reading the core-corrected curves where it always
    has, and permeability must keep reading PHIE rather than PHIE_CC.
    """
    result = _run(profile, _core_well())
    assert result.well.get("PHIE_CC") is not None, "core correction should have run"
    steps = {step.step: step for step in result.steps_executed}

    assert saturation in steps["saturation"].input_curves
    assert steps["rock_quality"].input_curves == rock_quality
    assert bvw in steps["bvw"].input_curves
    # Permeability reads PHIE under both profiles; it never used PHIE_CC.
    assert "PHIE" in steps["permeability"].input_curves
    assert "PHIE_CC" not in steps["permeability"].input_curves


def _flagged_well() -> Well:
    """A well carrying a reservoir flag from somewhere else, as an interpreted LAS does."""
    well = _well()
    well.add_curve(
        "RESFLAG",
        np.r_[np.ones(10), np.zeros(well.n_samples - 10)],
        "",
        "Reservoir Flag",
    )
    return well


def test_input_reservoir_flag_is_recomputed_by_default():
    result = _run("current", _flagged_well())
    perm = _step(result, "permeability")

    assert any("created before permeability" in note for note in perm.decisions)
    assert perm.parameters["swirr_metadata"]["hc_mask_samples"] > 10
    assert any(
        "recomputed from this run's cutoffs" in warning for warning in result.warnings
    ), result.warnings


def test_input_reservoir_flag_can_be_kept_on_request():
    result = _run("current", _flagged_well(), reuse_input_resflag=True)
    perm = _step(result, "permeability")

    assert any("already present before permeability" in note for note in perm.decisions)
    assert perm.parameters["swirr_metadata"]["hc_mask_samples"] == 10
    assert not any("recomputed from this run's cutoffs" in w for w in result.warnings)


def test_reuse_input_resflag_is_validated():
    assert DynamicWorkflowConfig().reuse_input_resflag is False
    with pytest.raises(ValueError, match="reuse_input_resflag"):
        DynamicWorkflowConfig(reuse_input_resflag="yes")


def test_benchmark_export_drops_porosity_curves_that_carry_nothing():
    """A sonic-only pass leaves PHIT == PHIS and PHI_SEC == 0; the eval
    package's interpreted LAS carried neither, so the benchmark export omits
    them. Both are kept when they hold information."""
    from petrophys.workflow.zoned_batch import _finalize_export_curves

    def well(sonic_only: bool) -> Well:
        w = Well("REDUNDANT")
        n = 6
        w.depth = np.arange(1000.0, 1000.0 + n, 1.0)
        w.depth_unit = "m"
        phis = np.linspace(0.10, 0.20, n)
        w.add_curve("PHIS", phis, "v/v", "Sonic porosity")
        w.add_curve("PHIT", phis if sonic_only else phis + 0.02, "v/v", "Total porosity")
        w.add_curve("PHI_SEC", np.zeros(n) if sonic_only else np.full(n, 0.02), "v/v", "Secondary porosity")
        return w

    redundant = well(sonic_only=True)
    _finalize_export_curves(redundant, {"PHIS"}, drop_redundant_porosity=True)
    assert redundant.get("PHIT") is None and redundant.get("PHI_SEC") is None
    assert redundant.get("PHIS") is not None

    informative = well(sonic_only=False)
    _finalize_export_curves(informative, {"PHIS"}, drop_redundant_porosity=True)
    assert informative.get("PHIT") is not None and informative.get("PHI_SEC") is not None

    default = well(sonic_only=True)
    _finalize_export_curves(default, {"PHIS"})
    assert default.get("PHIT") is not None and default.get("PHI_SEC") is not None


def test_configured_clay_porosity_is_not_replaced_by_a_derived_one():
    """An explicit phi_cl must reach PHIE, as an explicit setting beats a LAS
    header value. Left at its default it is derived from the shale samples."""
    import petrophys as pp

    def run(uwi=None, **settings):
        well = pp.read(pp.sample("30_6-5"))
        if uwi is not None:
            well.header.uwi = uwi
        result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, **settings)
        porosity = next(s for s in result.steps_executed if s.step == "porosity")
        phie = np.asarray(result.well.get("PHIE").data, dtype=float)
        return porosity.parameters, float(np.nanmean(phie))

    derived, derived_phie = run()
    assert derived["phi_cl_source"] == "lithology_shale_average_phit_nd_on_reservoir_matrix"
    # 30/6-5 gets the eval package's main-run calculations; under the UWI of
    # 15/9-17, a re-run well, the same logs get the re-run's.
    main_run, _ = run(calculation_profile="geolink")
    assert main_run["phi_cl_source"] == "all_shale_phit"
    rerun, _ = run(uwi="15/9-17", calculation_profile="geolink")
    assert rerun["phi_cl_source"] == "lithology_shale_average_phit_nd"

    configured, configured_phie = run(phi_cl=0.25)
    assert configured["phi_cl"] == 0.25
    assert configured["phi_cl_source"] == "config"
    # A larger clay porosity removes more pore volume from PHIE.
    assert configured_phie < derived_phie


# -- Which calculations a benchmark run gets ------------------------------------


@pytest.mark.parametrize(
    "uwi, name, calculations",
    [
        ("15/9-17", "PROFILE-TEST", "rerun"),  # the UWI names a re-run well
        ("", "34/10-23", "rerun"),  # no UWI: the WELL header does
        ("34/10-23", "34/10-23 Gullfaks Sor", "rerun"),
        (" 15_9-17 ", "", "rerun"),  # "_" for "/", surrounding spaces ignored
        ("15/9-17 A", "15/9-17 A", "main_run"),  # a sidetrack is another well
        ("15/9-171", "34/10-2", "main_run"),
        ("", "PROFILE-TEST", "main_run"),
        ("", "", "main_run"),  # no UWI or WELL header
    ],
)
def test_benchmark_gives_the_rerun_wells_the_rerun_calculations(uwi, name, calculations):
    well = _well(name=name)
    well.header.uwi = uwi
    result = _run("geolink", well)

    settings = result.to_trace_dict()["config"]
    assert settings["benchmark_calculations"] == calculations
    names = list(settings)
    assert names[names.index("calculation_profile") + 1] == "benchmark_calculations"
    # The main run carried BAD_HOLE twice; the re-run did not.
    assert (result.well.get("BADHOLEFLAG") is not None) == (calculations == "main_run")


def test_well_names_compare_without_case_and_with_slash_as_underscore():
    from petrophys.workflow.dynamic_workflow import _is_rerun_well, _normalized_well_name

    assert _normalized_well_name("  16/2-11 A ") == "16_2-11 a"
    assert _normalized_well_name("NO 15/9-17") == "no 15_9-17"
    header_dict = Well()
    header_dict.header = {"WELL": "15/9-17"}
    assert _is_rerun_well(header_dict)
    assert _is_rerun_well(Well("TEST"), file_stem="34_10-23")
    assert not _is_rerun_well(Well("TEST"), file_stem="34_10-23 A")


def test_current_traces_carry_no_benchmark_calculations():
    well = _well(name="15/9-17")
    settings = _run("current", well).to_trace_dict()["config"]

    assert settings["calculation_profile"] == "current"
    assert "benchmark_calculations" not in settings


def test_batch_runner_recognizes_a_rerun_well_by_uwi_well_or_file_name(tmp_path):
    from petrophys.io import export_las
    from petrophys.workflow.zoned_batch import process_las_directory_with_workflow

    source = tmp_path / "raw"
    source.mkdir()
    files = {
        # file name: (UWI, WELL header)
        "15_9-17": ("", "STEM-TEST"),  # only the file name names the well
        "UWI-TEST": ("34/10-23", "UWI-TEST"),
        "WELL-TEST": ("", "15/9-17"),
        "15_9-17 A": ("15/9-17 A", "15/9-17 A"),  # a near miss
        "34_10-23": ("", ""),  # no UWI or WELL header: the file name decides
        "OTHER": ("", "OTHER"),
    }
    for stem, (uwi, name) in files.items():
        well = _well(name=name)
        well.header.uwi = uwi
        export_las(well, source / f"{stem}.las")

    config = DynamicWorkflowConfig(rw=0.05, calculation_profile="geolink")
    results = process_las_directory_with_workflow(source, tmp_path / "processed", config=config)

    assert all(item.success for item in results), [item.errors for item in results]
    calculations = {
        Path(item.source_file).stem: item.result.base_result.to_trace_dict()["config"]["benchmark_calculations"]
        for item in results
    }
    assert calculations == {
        "15_9-17": "rerun", "UWI-TEST": "rerun", "WELL-TEST": "rerun", "34_10-23": "rerun",
        "15_9-17 A": "main_run", "OTHER": "main_run",
    }
    for item in results:
        main_run = calculations[Path(item.source_file).stem] == "main_run"
        assert (item.result.well.get("BADHOLEFLAG") is not None) == main_run


@pytest.mark.parametrize("layout", ["formation_lithology", "category_formation_lithology"])
def test_context_export_gives_every_well_the_rerun_calculations(tmp_path, monkeypatch, layout):
    import petrophys as pp
    from petrophys.io import export_las
    from petrophys.workflow import zoned_batch

    source = tmp_path / "input"
    source.mkdir()
    well = _well(name="CONTEXT-TEST")
    well.add_curve("LITHOLOGY_GEOLINK", np.ones(well.n_samples), "", "Lithology")
    export_las(well, source / "CONTEXT-TEST.las")

    runs = []
    run_with_zones = zoned_batch.run_dynamic_workflow_with_zones

    def recorded(well, config=None):
        runs.append(run_with_zones(well, config))
        return runs[-1]

    monkeypatch.setattr(zoned_batch, "run_dynamic_workflow_with_zones", recorded)
    config = DynamicWorkflowConfig(rw=0.05, calculation_profile="geolink")
    results = pp.export_context(source, tmp_path / "package", layout=layout, workflow_config=config)

    assert results and all(item.success for item in results)
    assert runs
    for run in runs:
        for result in (run.base_result, *run.zone_results):
            assert result.to_trace_dict()["config"]["benchmark_calculations"] == "rerun"
    # Outside the export the same well gets the main run's calculations.
    alone = pp.interpret(pp.read(source / "CONTEXT-TEST.las"), config=config)
    assert alone.to_trace_dict()["config"]["benchmark_calculations"] == "main_run"


def test_generate_benchmark_context_takes_one_calculation_profile():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_benchmark_context.py"
    namespace = runpy.run_path(str(script))
    parser = namespace["build_parser"]()
    required = ["--input-dir", "eval", "--output-dir", "out"]

    assert parser.parse_args([*required, "--from-raw"]).calculation_profile == "geolink"
    assert parser.parse_args([*required, "--calculation-profile", "current"]).calculation_profile == "current"
    assert parser.parse_args(required).dataset == "auto"
    assert parser.parse_args([*required, "--dataset", "costa"]).dataset == "costa"
    for removed in (
        ["--calculation-profile", "benchmark_rerun"],
        ["--las-profile", "geolink"],
        ["--context-profile", "geolink"],
        ["--well-profile", "15_9-17", "geolink"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, *removed])
    command = (
        "python scripts/generate_benchmark_context.py --from-raw "
        "--input-dir PATH/eval --output-dir NEW_DIR"
    )
    assert command in parser.format_help()
    assert command in namespace["__doc__"]


def test_one_context_script_auto_detects_a_costa_well(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_benchmark_context.py"
    namespace = runpy.run_path(str(script))
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "petrophys"
        / "datasets"
        / "costa"
        / "HW-31.las"
    )
    output = tmp_path / "context"

    assert namespace["main"](
        ["--input-dir", str(source), "--output-dir", str(output)]
    ) == 0
    context = [path for path in output.rglob("*.json") if "processed_las" not in path.parts]
    assert len(context) == 25  # 24 category files and the curve-name mapping
    assert not any("HW-32" in path.parts for path in context)
    assert (output / "processed_las" / "HW-31.las").exists()
