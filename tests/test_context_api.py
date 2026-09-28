"""Public context, bounded serialization, and legacy compatibility contracts."""

from copy import deepcopy
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
    for name, start, stop, unit in (("GR", 30, 100, "API"), ("RHOB", 2.3, 2.6, "g/cc"),
                                    ("NPHI", .1, .3, "v/v"), ("RDEP", 10, 40, "ohm.m"),
                                    ("PHIE", .12, .25, "v/v"), ("SW", .4, .8, "v/v")):
        well.add_curve(name, np.linspace(start, stop, well.n_samples), unit, name)
    return well


def test_versioned_context_is_json_safe_deterministic_and_nonmutating(context_well):
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


def test_context_accepts_interpretation_result(context_well):
    result = pp.interpret(context_well, rw=.07).raise_for_errors()
    context = pp.build_context(result)
    assert context["interpretation"] == result.summary()
    assert context["interpretation"]["interpretation_profile"] == "generic_inference"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_limits_rejected(context_well, limit):
    with pytest.raises(ValueError, match="positive integer"):
        pp.build_context(context_well, max_bytes=limit)


def test_size_limit_counts_utf8_and_does_not_touch_existing_output(context_well, tmp_path):
    context_well.name = "\u00e9" * 30
    payload = pp.build_context(context_well)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    size = len(text.encode("utf-8"))
    assert size > len(text)
    output = tmp_path / "context.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(pp.ContextSizeError):
        pp.export_context(context_well, output, max_bytes=size-1, overwrite=True)
    assert output.read_text(encoding="utf-8") == "keep"
    pp.export_context(context_well, output, max_bytes=size, overwrite=True)
    assert output.stat().st_size == size
    with pytest.raises(FileExistsError):
        pp.export_context(context_well, output)


def test_legacy_single_context_bytes_match_existing_writer(context_well, tmp_path):
    from petrophys.workflow.zoned_batch import ZonedWorkflowRunResult
    original = set(context_well.curves)
    raw_result = ZonedWorkflowRunResult(context_well, pp.DynamicWorkflowResult(context_well, pp.DynamicWorkflowConfig()))
    expected = reference._write_reference_analysis_json(
        deepcopy(context_well), deepcopy(raw_result), Path("CONTEXT-1.las"), tmp_path / "expected.las", original)
    actual = pp.export_context(context_well, tmp_path / "actual.json", schema_version="legacy",
                               source_file="CONTEXT-1.las", original_curves=original)
    assert actual.read_bytes() == expected.read_bytes()
    assert "schema_version" not in json.loads(actual.read_text(encoding="utf-8"))


def test_legacy_result_requires_honest_raw_curve_provenance(context_well):
    result = pp.interpret(context_well, rw=.07).raise_for_errors()
    with pytest.raises(ValueError, match="original_curves"):
        pp.build_context(result, schema_version="legacy")


def test_context_validates_schema_and_category(context_well):
    with pytest.raises(ValueError, match="schema_version"):
        pp.build_context(context_well, schema_version="future")
    with pytest.raises(ValueError, match="Unknown category"):
        pp.build_context(context_well, category="invalid")


def test_cli_and_script_defaults_do_not_depend_on_missing_workbooks(context_well, tmp_path, monkeypatch):
    from petrophys.io import export_las
    for name in ("PETROPHYS_DATA_DIR", "PETROPHYS_TOPS_FILE", "PETROPHYS_LITHO_FILE"):
        monkeypatch.delenv(name, raising=False)
    for layout in ("formation_lithology", "category_formation_lithology"):
        parser = build_package_parser(layout)
        with pytest.raises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(["--input-dir", str(tmp_path), "--tops-file", "tops.csv"])
        assert args.lithology_mapping_file is None
    source = tmp_path / "well.las"
    export_las(context_well, source)
    output = tmp_path / "context.json"
    assert main([str(source), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_failed_or_oversized_package_is_not_published(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text(json.dumps({"text": "x" * 1000}), encoding="utf-8")
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]
    monkeypatch.setattr(reference, "export_reference_json_work_package_by_formation_lithology", fake_exporter)
    output = tmp_path / "package"
    with pytest.raises(pp.ContextSizeError):
        pp.export_context(tmp_path, output, layout="formation_lithology", tops_file="tops.csv", max_bytes=50)
    assert not output.exists()
    assert not list(tmp_path.glob(".petrophys-context-*"))


@pytest.mark.parametrize("layout", ["formation_lithology", "category_formation_lithology"])
def test_legacy_package_context_bytes_match_direct_export(context_well, tmp_path, layout):
    from petrophys.io import export_las
    source = tmp_path / "input"
    source.mkdir()
    context_well.add_curve("LITHOLOGY_GEOLINK", np.ones(context_well.n_samples), "", "Lithology")
    export_las(context_well, source / "CONTEXT-1.las")
    tops = tmp_path / "tops.csv"
    tops.write_text("well,formation,md_top\nCONTEXT-1,TEST FM. TOP,1000\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    actual = tmp_path / "actual"
    exporter = (reference.export_reference_json_work_package_by_formation_lithology if layout == "formation_lithology"
                else reference.export_reference_json_work_package_by_category_formation_lithology)
    direct = exporter(source, baseline, tops)
    assert direct and all(item.success for item in direct), [item.errors for item in direct]
    exported = pp.export_context(source, actual, layout=layout, tops_file=tops, schema_version="legacy", max_bytes=5_000_000)
    assert len(exported) == len(direct)
    for item in direct:
        path = Path(item.json_file)
        relative = path.relative_to(baseline)
        assert path.read_bytes() == (actual / relative).read_bytes(), relative


def test_trace_paths_survive_package_staging(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text("{}", encoding="utf-8")
        (output_dir / "sample.trace.json").write_text(json.dumps({
            "schema": "petrophys_lithology_group_trace",
            "context_json_file": str(output),
        }), encoding="utf-8")
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]
    monkeypatch.setattr(reference, "export_reference_json_work_package_by_formation_lithology", fake_exporter)
    output = tmp_path / "package"
    pp.export_context(tmp_path, output, layout="formation_lithology", tops_file="tops.csv")
    payload = json.loads((output / "sample.trace.json").read_text(encoding="utf-8"))["payload"]
    assert Path(payload["context_json_file"]) == output / "sample.json"
    assert Path(payload["context_json_file"]).is_file()


def test_total_package_budget_is_enforced(tmp_path, monkeypatch):
    def fake_exporter(input_dir, output_dir, **options):
        output_dir.mkdir()
        output = output_dir / "sample.json"
        output.write_text("{}", encoding="utf-8")
        return [reference.ExampleFormatExportResult("input.las", str(output), True, str(output))]
    monkeypatch.setattr(reference, "export_reference_json_work_package_by_formation_lithology", fake_exporter)
    output = tmp_path / "package"
    with pytest.raises(pp.ContextSizeError, match="max_total_bytes"):
        pp.export_context(tmp_path, output, layout="formation_lithology", tops_file="tops.csv", max_total_bytes=10)
    assert not output.exists()
