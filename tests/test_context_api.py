"""Public contracts for computed, bounded context generation."""

import json
from pathlib import Path

import numpy as np
import pytest

import petrophys as pp
from petrophys.reporting import reference_export as reference
from petrophys.reporting.context_cli import build_package_parser, main


@pytest.fixture
def context_well():
    well = pp.Well("CONTEXT-1")
    well.depth = np.arange(1000.0, 1040.0, 0.5)
    well.depth_unit = "m"
    curves = (
        ("GR", 30, 100, "API"),
        ("RHOB", 2.3, 2.6, "g/cc"),
        ("NPHI", 0.1, 0.3, "v/v"),
        ("RDEP", 10, 40, "ohm.m"),
        ("PHIE", 0.12, 0.25, "v/v"),
        ("SW", 0.4, 0.8, "v/v"),
    )
    for name, start, stop, unit in curves:
        well.add_curve(name, np.linspace(start, stop, well.n_samples), unit, name)
    return well


def test_context_is_json_safe_deterministic_and_nonmutating(context_well):
    context_well.set_parameter("RW", np.nan)
    before = context_well.to_dataframe().copy()
    context = pp.build_context(context_well)
    assert context["schema_version"] == "1.0"
    assert context["interpretation"] is None
    assert context["include_raw_traces_in_context"] is False
    assert context["well"]["header_parameters"]["RW"] is None
    assert context == pp.build_context(context_well)
    json.dumps(context, allow_nan=False)
    assert context_well.to_dataframe().equals(before)


def test_context_accepts_one_canonical_interpretation_result(context_well):
    result = pp.interpret(context_well, rw=0.07).raise_for_errors()
    context = pp.build_context(result)
    assert context["interpretation"] == result.summary()
    assert context["interpretation"]["lithology_domain"] == result.lithology_domain
    assert "interpretation_profile" not in context["interpretation"]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_limits_rejected(context_well, limit):
    with pytest.raises(ValueError, match="positive integer"):
        pp.build_context(context_well, max_bytes=limit)


def test_size_limit_counts_utf8_and_does_not_touch_existing_output(context_well, tmp_path):
    context_well.name = "\u00e9" * 30
    payload = pp.build_context(context_well)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    size = len(text.encode("utf-8"))
    output = tmp_path / "context.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(pp.ContextSizeError):
        pp.export_context(context_well, output, max_bytes=size - 1, overwrite=True)
    assert output.read_text(encoding="utf-8") == "keep"
    pp.export_context(context_well, output, max_bytes=size, overwrite=True)
    assert output.stat().st_size == size
    with pytest.raises(FileExistsError):
        pp.export_context(context_well, output)


def test_removed_context_profiles_are_not_accepted(context_well):
    with pytest.raises(TypeError, match="schema_version"):
        pp.build_context(context_well, schema_version="legacy")
    with pytest.raises(ValueError, match="Unknown category"):
        pp.build_context(context_well, category="invalid")


def test_cli_always_interprets_single_las(context_well, tmp_path, monkeypatch):
    from petrophys.io import export_las

    for name in (
        "PETROPHYS_DATA_DIR",
        "PETROPHYS_FORMATIONS_FILE",
        "PETROPHYS_LITHOLOGY_FILE",
        "PETROPHYS_LITHOLOGY_CODES_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    for layout in ("formation_lithology", "category_formation_lithology"):
        parser = build_package_parser(layout)
        with pytest.raises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--input-dir", str(tmp_path)])
        assert args.formations is None

    source = tmp_path / "CONTEXT-1.las"
    export_las(context_well, source)
    output = tmp_path / "context.json"
    assert main([str(source), str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["interpretation"] is not None


def test_failed_or_oversized_package_is_not_published(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text(json.dumps({"text": "x" * 1000}), encoding="utf-8")
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]

    monkeypatch.setattr(
        reference,
        "export_reference_json_work_package_by_formation_lithology",
        fake_exporter,
    )
    output = tmp_path / "package"
    with pytest.raises(pp.ContextSizeError):
        pp.export_context(
            tmp_path,
            output,
            layout="formation_lithology",
            max_bytes=50,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".petrophys-context-*"))


@pytest.mark.parametrize(
    "layout",
    ["formation_lithology", "category_formation_lithology"],
)
def test_package_context_bytes_match_direct_computed_export(context_well, tmp_path, layout):
    from petrophys.io import export_las

    source = tmp_path / "input"
    source.mkdir()
    context_well.add_curve(
        "LITHOLOGY_GEOLINK",
        np.ones(context_well.n_samples),
        "",
        "Lithology",
    )
    export_las(context_well, source / "CONTEXT-1.las")
    tops = tmp_path / "tops.csv"
    tops.write_text(
        "well,formation,md_top\nCONTEXT-1,TEST FM. TOP,1000\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    actual = tmp_path / "actual"
    exporter = (
        reference.export_reference_json_work_package_by_formation_lithology
        if layout == "formation_lithology"
        else reference.export_reference_json_work_package_by_category_formation_lithology
    )
    direct = exporter(source, baseline, tops)
    assert direct and all(item.success for item in direct), [item.errors for item in direct]
    exported = pp.export_context(
        source,
        actual,
        layout=layout,
        tops_file=tops,
        max_bytes=5_000_000,
    )
    assert len(exported) == len(direct)
    for item in direct:
        path = Path(item.json_file)
        relative = path.relative_to(baseline)
        assert path.read_bytes() == (actual / relative).read_bytes(), relative


def test_package_can_run_without_formations(context_well, tmp_path):
    from petrophys.io import export_las

    source = tmp_path / "input"
    source.mkdir()
    context_well.add_curve(
        "LITHOLOGY_GEOLINK",
        np.ones(context_well.n_samples),
        "",
        "Lithology",
    )
    export_las(context_well, source / "CONTEXT-1.las")
    output = tmp_path / "package"
    results = pp.export_context(source, output, layout="formation_lithology")
    assert results and all(item.success for item in results)
    assert any("WHOLE_WELL" in str(path) for path in output.rglob("*.json"))


@pytest.mark.parametrize("layout", ["formation_lithology", "category_formation_lithology"])
def test_package_groups_by_selected_external_lithology(context_well, tmp_path, layout):
    from petrophys.io import export_las

    source = tmp_path / "input"
    source.mkdir()
    context_well.add_curve("LITHOLOGY_GEOLINK", np.ones(context_well.n_samples), "", "Sand")
    export_las(context_well, source / "CONTEXT-1.las")
    intervals = tmp_path / "lithology.csv"
    intervals.write_text(
        "well,top,bottom,lithology,depth_unit\n"
        "CONTEXT-1,1000,1020,Chalk,m\n"
        "CONTEXT-1,1025,1040,Marl,m\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    results = pp.export_context(
        source, output, layout=layout, lithology_file=intervals,
    )
    assert results and all(r.success for r in results), [r.errors for r in results]
    contexts = [json.loads(Path(r.json_file).read_text(encoding="utf-8")) for r in results]
    named = {d["analysis_results"]["lithology"] for d in contexts}
    assert named == {"CHALK", "MARL", "No lithology information available"}
    for payload in contexts:
        analysis = payload["analysis_results"]
        label = analysis["lithology"]
        if label in {"CHALK", "MARL"}:
            assert set(analysis["lithology_distribution"]) == {label}
            assert analysis["lithology_info"]["unique_lithologies"] == [label]
        else:
            assert analysis["lithology_distribution"] == {}
            assert not analysis["ground_truth_lithology_present"]


def test_cli_domain_option_keeps_the_las_gr_equation(context_well, tmp_path):
    from petrophys.io import export_las
    from petrophys.reporting.context_cli import _package_options

    context_well.set_parameter("GR_METHOD", "steiber")
    context_well.set_parameter("RHO_FL", 1.1)
    source = tmp_path / "well.las"
    export_las(context_well, source)
    args = build_package_parser("category_formation_lithology").parse_args(
        ["--input-dir", str(tmp_path), "--lithology-domain", "carbonate"]
    )
    options = _package_options(args)
    well, config = reference._read_context_input_well(
        source, lithology_domain=options["lithology_domain"],
    )
    assert config.lithology_domain == "carbonate"
    # LAS header parameters are applied by the zoned workflow runner.
    from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

    steps = {step.step: step for step in run_dynamic_workflow_with_zones(well, config).base_result.steps_executed}
    assert steps["vclay"].parameters["gr_method"] == "steiber"
    assert steps["porosity"].parameters["rho_fl"] == 1.1

    output = tmp_path / "context.json"
    assert main([str(source), str(output), "--lithology-domain", "clastic"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["interpretation"] is not None


def test_saturation_notes_follow_selected_external_lithology(context_well):
    import pandas as pd

    context_well.add_curve("LITHOLOGY_GEOLINK", np.ones(context_well.n_samples), "", "Sand")
    intervals = pd.DataFrame({
        "top": [1000.0, 1025.0],
        "bottom": [1020.0, 1040.0],
        "lithology": ["Chalk", "Marl"],
        "depth_unit": ["m", "m"],
    })
    result = pp.interpret(context_well, lithology_intervals=intervals)
    well = result.well
    groups = reference._segment_lithology_groups_for_saturation(well)
    assert set(groups[:40]) == {"limestone"}
    assert set(groups[40:50]) == {"clastic"}
    assert set(groups[50:]) == {"limestone"}

    well.add_curve("VCL", np.full(well.n_samples, 0.5), "v/v")
    well.add_curve("SW", np.full(well.n_samples, 0.5), "v/v")
    counts = reference._segment_saturation_model_counts(well, {"sw_model_requested": "auto"})
    assert counts == {"dual_water": 70, "indonesia": 10}


def test_trace_paths_survive_package_staging_without_payload_wrapper(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text("{}", encoding="utf-8")
        (output_dir / "sample.trace.json").write_text(
            json.dumps(
                {
                    "schema": "petrophys_lithology_group_trace",
                    "context_json_file": str(output),
                }
            ),
            encoding="utf-8",
        )
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]

    monkeypatch.setattr(
        reference,
        "export_reference_json_work_package_by_formation_lithology",
        fake_exporter,
    )
    output = tmp_path / "package"
    pp.export_context(tmp_path, output, layout="formation_lithology")
    payload = json.loads((output / "sample.trace.json").read_text(encoding="utf-8"))
    assert Path(payload["context_json_file"]) == output / "sample.json"
    assert Path(payload["context_json_file"]).is_file()


def test_total_package_budget_is_enforced(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text("{}", encoding="utf-8")
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]

    monkeypatch.setattr(
        reference,
        "export_reference_json_work_package_by_formation_lithology",
        fake_exporter,
    )
    output = tmp_path / "package"
    with pytest.raises(pp.ContextSizeError, match="max_total_bytes"):
        pp.export_context(
            tmp_path,
            output,
            layout="formation_lithology",
            max_total_bytes=10,
        )
    assert not output.exists()
