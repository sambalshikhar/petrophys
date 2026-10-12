"""One lithology code curve per run, under every name it may carry.

``matrix_inference.lithology_code_curve`` finds the curve: the one
``DynamicWorkflowConfig.lithology_curve`` names, else ``LITHOLOGY_GEOLINK`` as
it stands, else (under ``"current"``) the first lithology-code mnemonic whose
curve holds lithology codes. The vendor lithology, the matrix resolved from it,
the per-depth lithology groups and the readers downstream (net pay,
permeability, the context exporter, the diagnostics, the plots) all read that
one curve. The benchmark profile keeps the eval package's reading: the
name-pattern detector for the vendor lithology and its matrix, per-depth groups
from ``LITHOLOGY_GEOLINK`` alone.
"""
import copy
import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.interpretation.lithology.matrix_inference import (
    find_lithology_code_curve,
    interpreted_lithology_code_curve,
    lithology_code_curve,
    resolve_from_lithology_log,
)
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig

N = 60
# GeoLink codes: 1 sandstone over 11 limestone.
SAND_OVER_LIMESTONE = np.where(np.arange(N) < N // 2, 1.0, 11.0)


def _codes_well(code_curve="LITHOLOGY_GEOLINK", codes=SAND_OVER_LIMESTONE, name="CODES"):
    n = len(codes)
    well = Well(name)
    well.depth = np.arange(2000.0, 2000.0 + 0.5 * n, 0.5)
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(20.0, 130.0, n), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.30, 0.05, n), "v/v", "Neutron")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(40.0, 2.0, n), "ohm.m", "Deep Resistivity")
    well.add_curve("CALI", np.full(n, 8.7), "in", "Caliper")
    if code_curve is not None:
        well.add_curve(code_curve, np.asarray(codes, dtype=float), "", "Lithology code")
    return well


def _config(profile="current", **settings):
    return DynamicWorkflowConfig(
        rw=0.05, skip_steps=["temperature", "rmf"], calculation_profile=profile, **settings
    )


def _run(well, profile="current", **settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DynamicWorkflow(_config(profile, **settings)).run(well)
    assert result.success, result.errors
    return result


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def _decoded_warning(result):
    return [text for text in result.warnings if text.startswith("Lithology codes are read from")]


# -- The resolver --------------------------------------------------------------------


def test_resolver_reads_the_named_curve_then_geolink_then_a_code_mnemonic():
    well = _codes_well()
    well.add_curve("FACIES", np.full(N, 8.0), "", "Facies")
    well.add_curve("ROCKS", np.full(N, np.nan), "", "Empty")

    assert lithology_code_curve(well)[0] == "LITHOLOGY_GEOLINK"
    name, curve = lithology_code_curve(well, explicit=" facies ")
    assert name == "FACIES" and curve is well.get("FACIES")
    # A named curve without a finite sample is passed over.
    assert lithology_code_curve(well, explicit="ROCKS")[0] == "LITHOLOGY_GEOLINK"
    assert lithology_code_curve(well, explicit="NOT_THERE")[0] == "LITHOLOGY_GEOLINK"

    well.remove_curve("LITHOLOGY_GEOLINK")
    assert lithology_code_curve(well)[0] == "FACIES"
    assert lithology_code_curve(well, allow_aliases=False) is None
    assert lithology_code_curve(well, explicit="FACIES", allow_aliases=False)[0] == "FACIES"


def test_geolink_is_read_as_it_stands():
    # Three finite samples, one of them no GeoLink code: far too few for the
    # code-log checks, read all the same.
    codes = np.r_[1.0, 7.0, 99.0, np.full(N - 3, np.nan)]
    assert lithology_code_curve(_codes_well(codes=codes))[0] == "LITHOLOGY_GEOLINK"
    assert lithology_code_curve(_codes_well(codes=np.full(N, np.nan))) is None


@pytest.mark.parametrize(
    "codes, accepted",
    [
        (np.r_[np.full(20, 1.0), np.full(N - 20, np.nan)], True),   # 20 finite codes
        (np.r_[np.full(19, 1.0), np.full(N - 19, np.nan)], False),  # 19 finite codes
        (np.linspace(1.0, 12.0, N), False),                         # not whole numbers
        (np.full(N, 150.0), False),                                 # above 99
        (np.r_[np.full(30, 1.0), np.full(30, 60.0)], True),         # half GeoLink codes
        (np.r_[np.full(29, 1.0), np.full(31, 60.0)], False),        # under half
    ],
)
def test_a_code_mnemonic_is_read_only_when_it_holds_lithology_codes(codes, accepted):
    found = lithology_code_curve(_codes_well("LITHOLOGY", codes=codes))
    assert (found is not None) == accepted
    if accepted:
        assert found[0] == "LITHOLOGY"


@pytest.mark.parametrize(
    "name", ["LITHO_PATT", "GR_LITH", "LITHO", "LITHO_CLASS", "MMQC", "GT_LITHOLOGY_GROUP_CODE"]
)
def test_only_the_lithology_code_mnemonics_are_tried(name):
    # Names that only look lithology-like, the workflow's computed curves and
    # the derived group curves are never a lithology code curve.
    assert lithology_code_curve(_codes_well(name)) is None


def test_resolution_from_a_code_log_follows_the_curve_the_run_recorded():
    # Limestone codes under a display-pattern name.
    well = _codes_well("LITHO_PATT", codes=np.full(N, 11.0))
    # Without a record: the name-pattern detector.
    assert resolve_from_lithology_log(well, warn=False).matrix == "limestone"
    assert resolve_from_lithology_log(well, warn=False, code_curve="LITHO_PATT").matrix == "limestone"
    # A run that found no code curve: no vendor lithology.
    well.set_parameter("_geology_context", {"lithology_code_curve": None})
    assert resolve_from_lithology_log(well, warn=False) is None
    well.set_parameter("_geology_context", {"lithology_code_curve": "LITHO_PATT"})
    assert resolve_from_lithology_log(well, warn=False).matrix == "limestone"


def test_record_helpers():
    well = _codes_well("FACIES")
    assert interpreted_lithology_code_curve(well) == "LITHOLOGY_GEOLINK"
    assert find_lithology_code_curve(well) == "FACIES"

    well.set_parameter("_geology_context", {"lithology_code_curve": None})
    assert interpreted_lithology_code_curve(well) == "LITHOLOGY_GEOLINK"
    assert find_lithology_code_curve(well) is None

    well.set_parameter("_geology_context", {"lithology_code_curve": "FACIES"})
    assert interpreted_lithology_code_curve(well) == "FACIES"
    assert interpreted_lithology_code_curve(well.depth_slice(2000.0, 2004.0)) == "FACIES"


# -- The workflow ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["LITHOLOGY", "FACIES"])
def test_current_reads_a_code_mnemonic_as_it_reads_geolink(name):
    geolink = _run(_codes_well())
    renamed = _run(_codes_well(name))

    assert _step(renamed, "input_context").parameters == {
        "requested_source": "auto", "selected_source": "vendor",
        "external_available": False, "vendor_available": True, "vendor_curve": name,
    }
    porosity = _step(renamed, "porosity").parameters
    assert porosity["matrix_mode"] == "lithology_dependent"
    assert porosity["matrix_source"] == f"vendor_lithology_log[{name}]"
    assert porosity["matrix_groups_used"] == ["SANDSTONE", "LIMESTONE"]
    assert _step(renamed, "saturation").parameters["archie_lithology_source"] == f"vendor_lithology_log[{name}]"
    families = _step(renamed, "net_pay").parameters["cutoffs_by_family"]
    assert (families["clastic"]["samples"], families["carbonate"]["samples"]) == (30, 30)
    lithology = _step(renamed, "lithology").parameters
    assert lithology["lithology_source_priority"] == "input_lithology"
    assert renamed.well.get("LITHO") is None
    assert renamed.well.get_parameter("_geology_context")["lithology_code_curve"] == name
    assert _decoded_warning(renamed) == [
        f"Lithology codes are read from {name} and decoded with the GeoLink/Kingdom lithology "
        "code table, as LITHOLOGY_GEOLINK codes are. If the curve holds other codes, pass "
        "lithology_source='computed' or lithology intervals."
    ]
    assert _decoded_warning(geolink) == []
    assert _step(geolink, "porosity").parameters["matrix_source"] == "ground_truth_lithology_geolink"
    for curve in ("VCL", "PHIT", "PHIE", "SW", "PERM", "NET_RES", "NET_PAY"):
        np.testing.assert_array_equal(renamed.well.get(curve).data, geolink.well.get(curve).data)


def test_benchmark_groups_by_geolink_alone():
    renamed = _run(_codes_well("FACIES"), "benchmark")

    # The eval package's detector still finds the vendor lithology ...
    assert _step(renamed, "input_context").parameters == {
        "requested_source": "auto", "selected_source": "vendor",
        "external_available": False, "vendor_available": True,
    }
    # ... while its per-depth groups come from LITHOLOGY_GEOLINK only.
    porosity = _step(renamed, "porosity").parameters
    assert porosity["matrix_mode"] == "constant"
    assert porosity["matrix_source"] == "config_or_default[sandstone]"
    assert renamed.well.get("LITHO") is not None
    assert "lithology_code_curve" not in renamed.well.get_parameter("_geology_context")
    assert _decoded_warning(renamed) == []

    geolink = _run(_codes_well(), "benchmark")
    assert "vendor_curve" not in _step(geolink, "input_context").parameters
    assert _step(geolink, "porosity").parameters["matrix_source"] == "ground_truth_lithology_geolink"
    assert "lithology_code_curve" not in geolink.well.get_parameter("_geology_context")


def test_a_lithology_like_name_is_a_vendor_lithology_only_under_benchmark():
    current = _run(_codes_well("LITHO_PATT"))
    assert _step(current, "input_context").parameters["selected_source"] == "computed"
    assert _step(current, "input_context").parameters["vendor_curve"] is None

    benchmark = _run(_codes_well("LITHO_PATT"), "benchmark")
    assert _step(benchmark, "input_context").parameters["selected_source"] == "vendor"

    with pytest.raises(ValueError, match="or the curve named by lithology_curve"):
        DynamicWorkflow(_config(lithology_source="vendor", raise_on_error=True)).run(
            _codes_well("LITHO_PATT")
        )


def test_lithology_curve_names_the_code_curve():
    # Ten codes under an unrecognized name: too few for the code-log checks,
    # read all the same because the setting names the curve.
    codes = np.r_[np.full(5, 1.0), np.full(5, 11.0), np.full(N - 10, np.nan)]
    for profile in ("current", "benchmark"):
        result = _run(_codes_well("LITHO_PATT", codes=codes), profile, lithology_curve="litho_patt")
        assert result.config.lithology_curve == "LITHO_PATT"
        assert _step(result, "input_context").parameters["vendor_curve"] == "LITHO_PATT"
        porosity = _step(result, "porosity").parameters
        assert porosity["matrix_source"] == "vendor_lithology_log[LITHO_PATT]"
        assert porosity["matrix_groups_used"] == ["SANDSTONE", "LIMESTONE"]
        assert result.well.get_parameter("_geology_context")["lithology_code_curve"] == "LITHO_PATT"
        assert _decoded_warning(result) == [
            "Lithology codes are read from LITHO_PATT and decoded with the GeoLink/Kingdom "
            "lithology code table, as LITHOLOGY_GEOLINK codes are."
        ]


def test_a_named_curve_the_well_lacks_is_reported():
    result = _run(_codes_well(), lithology_curve="ROCKS")
    assert (
        "lithology_curve='ROCKS': the well has no such curve with finite samples; "
        "LITHOLOGY_GEOLINK is read instead."
    ) in result.warnings
    assert _step(result, "porosity").parameters["matrix_source"] == "ground_truth_lithology_geolink"

    result = _run(_codes_well(None), lithology_curve="ROCKS")
    assert (
        "lithology_curve='ROCKS': the well has no such curve with finite samples; "
        "no lithology code curve is read."
    ) in result.warnings


@pytest.mark.parametrize("value", ["LITHO", "litho_class", "MMQC", "GT_LITHOLOGY_GROUP_CODE", "", "  ", 7])
def test_lithology_curve_must_name_an_input_curve(value):
    with pytest.raises(ValueError, match="lithology_curve"):
        DynamicWorkflowConfig(lithology_curve=value)


def test_computed_lithology_sets_a_code_mnemonic_aside():
    result = _run(_codes_well("FACIES"), lithology_source="computed")
    assert _step(result, "input_context").parameters["vendor_curve"] is None
    assert result.well.get_parameter("_geology_context")["lithology_code_curve"] is None
    assert result.well.get("LITHO") is not None
    assert _decoded_warning(result) == []


def test_the_vendor_matrix_comes_from_the_curve_of_the_groups():
    # LITHOLOGY_GEOLINK holds five sandstone codes; FACIES sixty limestone codes.
    well = _codes_well("FACIES", codes=np.full(N, 11.0))
    well.add_curve("LITHOLOGY_GEOLINK", np.r_[np.full(5, 1.0), np.full(N - 5, np.nan)], "", "")

    def vendor_resolution(profile):
        workflow = DynamicWorkflow(_config(profile, matrix_type="auto", lithology_domain="auto"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.run(copy.deepcopy(well))
        return workflow._lithology_resolution

    # "current": the groups read LITHOLOGY_GEOLINK, so the matrix does too, and
    # five codes are too few to resolve one.
    assert vendor_resolution("current") is None
    # The benchmark detector passes over the five codes and reads FACIES.
    benchmark = vendor_resolution("benchmark")
    assert (benchmark.method, benchmark.matrix) == ("vendor_log", "limestone")


def test_a_rerun_records_its_own_code_curve():
    well = _codes_well("FACIES")
    workflow = DynamicWorkflow(_config())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.run(well)
        assert well.get_parameter("_geology_context")["lithology_code_curve"] == "FACIES"
        # A benchmark run of the same well object reads it without the record.
        DynamicWorkflow(_config("benchmark")).run(well)
    assert "lithology_code_curve" not in well.get_parameter("_geology_context")
    assert well.get("LITHO") is not None


def test_zones_read_the_whole_well_code_curve():
    from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

    def zone_context(profile):
        well = _codes_well("FACIES")
        # A ten-sample zone: too few samples to pass the code-log checks alone.
        well.set_parameter("TOP_Z1", 2010.0)
        well.set_parameter("BOT_Z1", 2014.5)
        well.set_parameter("RW_Z1", 0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_dynamic_workflow_with_zones(well, _config(profile))
        (zone,) = result.zone_results
        assert zone.well.n_samples == 10
        return zone, _step(zone, "input_context").parameters

    zone, context = zone_context("current")
    assert zone.config.lithology_curve == "FACIES"
    assert context["vendor_curve"] == "FACIES" and context["selected_source"] == "vendor"
    assert _step(zone, "porosity").parameters["matrix_source"] == "vendor_lithology_log[FACIES]"

    zone, context = zone_context("benchmark")
    assert zone.config.lithology_curve is None
    assert "vendor_curve" not in context


# -- Readers downstream --------------------------------------------------------------------


def test_context_reads_the_code_curve_the_interpretation_read():
    from petrophys.reporting import reference_export as ref

    current = _run(_codes_well("FACIES")).well
    assert ref._context_lithology_curve(current).mnemonic == "FACIES"
    segments = ref._build_lithology_group_segments(current, minimum_group_thickness=2.0)
    assert [int(segment["mask"].sum()) for segment in segments] == [30, 30]
    groups = ref._segment_lithology_groups_for_saturation(current)
    assert list(groups) == ["sandstone"] * 30 + ["limestone"] * 30
    assert ref._reference_statistics_curve_key("FACIES", current) is None
    assert ref._reference_statistics_curve_key("LITHOLOGY_GEOLINK", current) is None

    # The benchmark profile reads LITHOLOGY_GEOLINK alone, as the eval package did.
    benchmark = _run(_codes_well("FACIES"), "benchmark").well
    assert ref._context_lithology_curve(benchmark) is None
    assert ref._reference_statistics_curve_key("FACIES", benchmark) == "FACIES"
    # Saturation routing groups by the computed LITHO instead.
    from petrophys.workflow.dynamic_workflow import COMPUTED_LITHO_CODE_TO_GROUP

    expected = [
        COMPUTED_LITHO_CODE_TO_GROUP.get(int(round(code)), "clastic") if np.isfinite(code) else "clastic"
        for code in benchmark.get("LITHO").data
    ]
    assert list(ref._segment_lithology_groups_for_saturation(benchmark)) == expected


def test_diagnostics_read_the_code_curve_the_interpretation_read():
    from petrophys.evaluation.diagnostics import LithologyAnalysis, NumericalStatisticalAnalysis

    def group_codes(well):
        result = LithologyAnalysis().analyze(well, add_curves=False)
        return result.derived_curves["GT_LITHOLOGY_GROUP_CODE"].data.tolist()

    current = _run(_codes_well("FACIES")).well
    # GROUP_CODES: SANDSTONE 7, LIMESTONE 1, NO_LITHOLOGY_DATA 0.
    assert group_codes(current) == [7.0] * 30 + [1.0] * 30
    assert "FACIES" not in NumericalStatisticalAnalysis._numeric_curve_names(current)

    benchmark = _run(_codes_well("FACIES"), "benchmark").well
    assert group_codes(benchmark) == [0.0] * N
    assert "FACIES" in NumericalStatisticalAnalysis._numeric_curve_names(benchmark)


def test_net_pay_step_reads_a_code_mnemonic():
    from petrophys.workflow import Workflow, steps

    def net_pay(code_curve):
        well = _codes_well(code_curve)
        well.add_curve("VCL", np.full(N, 0.20), "v/v", "Clay volume")
        well.add_curve("PHIE", np.full(N, 0.10), "v/v", "Effective porosity")
        well.add_curve("SW", np.full(N, 0.65), "v/v", "Water saturation")
        run = Workflow("pay", [steps.net_pay(exclude_non_reservoir=False)]).run(well)
        return run.well.get("NET_PAY").data.tolist(), run.warnings

    # SW 0.65 fails the clastic cutoff (0.60) and passes the carbonate one (0.70).
    pay, notes = net_pay("FACIES")
    assert pay == [0.0] * 30 + [1.0] * 30
    assert notes == [
        "net_pay: Lithology families from FACIES, decoded with the GeoLink/Kingdom "
        "lithology code table, as LITHOLOGY_GEOLINK codes are."
    ]
    assert net_pay("LITHOLOGY_GEOLINK") == (pay, [])
    assert net_pay(None)[0] == [0.0] * N


def _lith_vcl_well(code_curve):
    codes = np.tile([11.0, 17.0, 24.0, 20.0, 16.0, 1.0], 4)
    well = Well("PERM-LITH-VCL")
    well.depth = np.arange(float(codes.size))
    well.add_curve("PHIT", np.linspace(0.24, 0.12, codes.size), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.linspace(0.22, 0.10, codes.size), "v/v", "Effective Porosity")
    well.add_curve("VCL", np.linspace(0.00, 0.30, codes.size), "v/v", "Clay Volume")
    well.add_curve(code_curve, codes, "", "Lithology Code")
    return well


def test_lith_vcl_permeability_reads_the_code_curve():
    from petrophys.interpretation.permeability import PermeabilityModule, lith_vcl_permeability

    well = _lith_vcl_well("FACIES")
    result = PermeabilityModule(method="lith_vcl").calculate(well)
    expected = lith_vcl_permeability(well.get("PHIT").data, well.get("VCL").data, well.get("FACIES").data)
    np.testing.assert_allclose(result.perm, np.minimum(expected, 10000.0), equal_nan=True)
    assert result.metadata["method_metadata"]["lithology_curve"] == "FACIES"

    # Named, the module reads that curve alone; this well has none of it.
    named = PermeabilityModule(method="lith_vcl", lithology_curve="LITHOLOGY_GEOLINK").calculate(well)
    assert named.metadata["method_metadata"]["lithology_curve"] == "default_sandstone"
    assert named.metadata["method_metadata"]["lithology_parameter_groups"] == ["sandstone"]


def test_standard_workflow_permeability_reads_a_code_mnemonic():
    from petrophys.workflow.standard_workflow import StandardWorkflow, WorkflowConfig

    with pytest.warns(DeprecationWarning):
        workflow = StandardWorkflow(WorkflowConfig(perm_method="lith_vcl"))
    trace = workflow._calculate_permeability(_lith_vcl_well("FACIES"))
    assert trace.success, trace.message
    assert "FACIES" in trace.input_curves


def test_core_data_reads_the_code_curve():
    from petrophys.interpretation.core_analysis import core_data_from_well

    well = _lith_vcl_well("FACIES")
    porosity = np.full(well.n_samples, np.nan)
    porosity[[2, 9]] = [0.21, 0.18]
    well.add_curve("PHIC", porosity, "v/v", "Core porosity")

    core = core_data_from_well(well)
    np.testing.assert_array_equal(core.depth, [2.0, 9.0])
    np.testing.assert_array_equal(core.lithology, [24.0, 20.0])
    assert core_data_from_well(well, lithology_curves=("LITHOLOGY_GEOLINK",)).lithology is None


def test_interval_analyzer_reports_the_dominant_lithology_group():
    from petrophys.evaluation.intervals.analyzer import _dominant_lithology

    # Code 7 (silty shale) is the most common code, sandstone (1 and 2) the
    # most common group.
    codes = np.r_[np.full(8, 7.0), np.full(6, 1.0), np.full(6, 2.0)]
    well = _codes_well(codes=codes)
    everywhere = np.ones(codes.size, dtype=bool)
    assert _dominant_lithology(well, everywhere) == "sandstone"
    assert _dominant_lithology(well, np.arange(codes.size) < 8) == "shale"

    # Lithology intervals the interpretation selected give their labels.
    well.add_curve("LITHO_CLASS", np.r_[np.full(5, 1.0), np.full(15, 2.0)], "", "Intervals")
    well.get("LITHO_CLASS").metadata.custom["category_labels"] = {1: "Sand", 2: "Marl"}
    assert _dominant_lithology(well, everywhere) == "Marl"


def test_plots_label_a_code_mnemonic_with_the_geolink_codes():
    from petrophys.visualization.log_viewer import LogViewer, create_adaptive_log_viewer_template

    def plot_well(code_curve):
        well = _codes_well(code_curve, codes=np.where(np.arange(N) < 30, 1.0, 7.0))
        for name, volume in (("VQUAR", 0.6), ("VCALC", 0.2), ("VDOLO", 0.1)):
            well.add_curve(name, np.full(N, volume), "v/v", name)
        return well

    well = plot_well("FACIES")
    template = create_adaptive_log_viewer_template(well)
    tracks = {track.display_name: track for track in template.tracks}
    assert tracks["LITHOLOGY"].curves[0].curve_name == "FACIES"
    assert "MINERAL VOLUMES" not in tracks

    viewer = LogViewer(well, template_defaults="auto")
    # GeoLink 7 is silty shale; the computed LITHO code 7 is coal.
    assert viewer._lithology_label(7.0, "FACIES") == "Silty Shale"
    assert viewer._lithology_label(7.0, "LITHO") == "Coal"

    # A lithology track that lists no curve draws the code curve.
    import matplotlib.pyplot as plt

    track = next(track for track in viewer.template.tracks if track.track_type == "lithology")
    track.curves = []
    figure, axis = plt.subplots()
    try:
        depth = np.asarray(viewer.well.depth, dtype=float)
        viewer._plot_lithology_track(axis, track, depth, depth[0], depth[-1])
    finally:
        plt.close(figure)
    assert viewer.curve_lookup["FACIES"] == "FACIES"

    plain = create_adaptive_log_viewer_template(plot_well("LITHO_PATT"))
    assert "MINERAL VOLUMES" in [track.display_name for track in plain.tracks]


# -- A bundled GeoLink well ----------------------------------------------------------------


def test_bundled_30_6_5_reads_a_renamed_code_curve_as_it_reads_geolink():
    from petrophys.reporting.reference_export import _build_lithology_group_segments

    def interpret(name, **settings):
        well = pp.read(pp.sample("30_6-5"))
        if name != "LITHOLOGY_GEOLINK":
            assert well.rename_curve("LITHOLOGY_GEOLINK", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pp.interpret(well, rw=0.07, rw_meas_temp_f=230, **settings)

    geolink = interpret("LITHOLOGY_GEOLINK")
    for name in ("LITHOLOGY", "FACIES"):
        renamed = interpret(name)
        assert _step(renamed, "porosity").parameters["matrix_mode"] == "lithology_dependent"
        assert (
            _step(renamed, "net_pay").parameters["cutoffs_by_family"]
            == _step(geolink, "net_pay").parameters["cutoffs_by_family"]
        )
        for curve in ("PHIE", "SW", "PERM", "NET_RES", "NET_PAY"):
            np.testing.assert_array_equal(renamed.well.get(curve).data, geolink.well.get(curve).data)
        assert len(_build_lithology_group_segments(renamed.well)) == len(
            _build_lithology_group_segments(geolink.well)
        )

    # The benchmark profile keeps the eval-era reading: no per-depth groups.
    benchmark = interpret("FACIES", calculation_profile="benchmark")
    assert _step(benchmark, "porosity").parameters["matrix_mode"] == "constant"
    assert benchmark.well.get("LITHO") is not None
