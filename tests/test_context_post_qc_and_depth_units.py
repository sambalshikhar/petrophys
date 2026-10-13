"""Context packages: dependent curves after the value-range QC, and depth units."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import petrophys as pp
from petrophys.io import export_las
from petrophys.reporting import reference_export as reference

_INTERVAL = re.compile(r"^-?\d+\.\d--?\d+\.\d( \S+)?$")


def _context_logs(folder, depth_unit):
    """One LAS over 1000-1060 with a washout at 1005-1006 and three formations."""
    well = pp.Well("CONTEXT-1")
    well.depth = np.arange(1000.0, 1060.0, 0.5)
    well.depth_unit = depth_unit
    for name, start, stop, unit in (
        ("GR", 30, 100, "API"),
        ("RHOB", 2.3, 2.6, "g/cc"),
        ("NPHI", 0.1, 0.3, "v/v"),
        ("RDEP", 10, 40, "ohm.m"),
        ("PHIE", 0.12, 0.25, "v/v"),
        ("SW", 0.4, 0.8, "v/v"),
    ):
        well.add_curve(name, np.linspace(start, stop, well.n_samples), unit, name)
    caliper = np.where((well.depth >= 1005.0) & (well.depth <= 1006.0), 12.0, 8.5)
    well.add_curve("CALI", caliper, "in", "Caliper")
    well.add_curve("BS", np.full(well.n_samples, 8.5), "in", "Bit Size")
    well.add_curve("LITHOLOGY_GEOLINK", np.ones(well.n_samples), "", "Lithology")
    folder.mkdir()
    export_las(well, folder / "CONTEXT-1.las")
    tops = folder.parent / "tops.csv"
    tops.write_text(
        "well,formation,md_top\n"
        "CONTEXT-1,UPPER FM. TOP,1000\n"
        "CONTEXT-1,MIDDLE FM. TOP,1020\n"
        "CONTEXT-1,LOWER FM. TOP,1040\n",
        encoding="utf-8",
    )
    return folder, tops


def _interpret_with_rejected_inputs(monkeypatch):
    """Interpret, then give MIDDLE porosity and LOWER permeability that the QC rejects.

    MIDDLE gets porosity 0.50 (above the 0.45 limit); LOWER gets porosity
    0.20 and 0.50 alternately and permeability 1e-5 mD (below 1e-4 mD).
    PHI_Z, RQI, FZI and BVW are recomputed from them, as the workflow writes
    its dependents before the exporter's QC.
    """
    interpret = reference._run_context_workflow

    def run(well, config):
        result = interpret(well, config)
        out = result.well
        depth = np.asarray(out.depth, dtype=float)
        middle = (depth >= 1020.0) & (depth < 1040.0)
        lower = depth >= 1040.0
        phi = np.array(out.get("PHIE").data, dtype=float)
        phi[middle] = 0.50
        phi[lower] = np.resize([0.20, 0.50], int(lower.sum()))
        perm = np.array(out.get("PERM").data, dtype=float)
        perm[lower] = 1e-5
        sw = np.asarray(out.get("SW").data, dtype=float)
        phi_z = phi / (1.0 - phi)
        rqi = 0.0314 * np.sqrt(perm / phi)
        rewritten = {
            "PHIE": phi, "PHIT": phi, "PERM": perm,
            "PHI_Z": phi_z, "RQI": rqi, "FZI": rqi / phi_z, "BVW": phi * sw,
        }
        for name, data in rewritten.items():
            curve = out.get(name)
            values = np.array(curve.data, dtype=float)
            values[middle | lower] = data[middle | lower]
            curve._data = values
        assert out.get("PHIE_CC") is None and out.get("PERM_CC") is None
        return result

    monkeypatch.setattr(reference, "_run_context_workflow", run)


def _export(tmp_path, monkeypatch, profile, depth_unit="m"):
    logs, tops = _context_logs(tmp_path / "logs", depth_unit)
    _interpret_with_rejected_inputs(monkeypatch)
    results = pp.export_context(
        logs,
        tmp_path / "context",
        layout="category_formation_lithology",
        tops_file=tops,
        workflow_config=pp.DynamicWorkflowConfig(calculation_profile=profile),
    )
    assert results and all(item.success for item in results), [item.errors for item in results]
    contexts = {}
    for item in results:
        path = Path(item.json_file)
        category, formation = path.parts[-4], path.parent.name
        contexts[category, formation] = json.loads(path.read_text(encoding="utf-8"))
    return contexts


def _interval_strings(value):
    if isinstance(value, dict):
        return [text for item in value.values() for text in _interval_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _interval_strings(item)]
    return [value] if isinstance(value, str) and _INTERVAL.match(value) else []


def test_current_context_drops_values_computed_from_rejected_inputs(tmp_path, monkeypatch):
    contexts = _export(tmp_path, monkeypatch, "current")

    middle = contexts["porosity", "MIDDLE FM. TOP"]["analysis_results"]
    assert "PERM_overall" in middle["statistics"]
    for curve in ("PHIE", "PHIT", "PHI_NORM", "RQI", "FZI", "BVW"):
        assert f"{curve}_overall" not in middle["statistics"], curve
        assert curve not in middle["derived_logs_computed"], curve

    lower = contexts["porosity", "LOWER FM. TOP"]["analysis_results"]
    # PHI_NORM = 0.20 / (1 - 0.20) where the porosity is kept; RQI and FZI
    # need the rejected PERM.
    statistics = lower["statistics"]
    phi_norm = statistics["PHI_NORM_overall"]
    assert phi_norm["minimum"] == phi_norm["maximum"] == 0.25
    assert not {"PERM_overall", "RQI_overall", "FZI_overall"} & set(statistics)
    assert not {"RQI", "FZI"} & set(lower["derived_logs_computed"])


def test_benchmark_context_keeps_the_eval_package_values(tmp_path, monkeypatch):
    contexts = _export(tmp_path, monkeypatch, "geolink")

    # PHI_NORM from the workflow's PHI_Z of 0.5 / (1 - 0.5), as in the eval package.
    middle = contexts["porosity", "MIDDLE FM. TOP"]["analysis_results"]["statistics"]
    phi_norm = middle["PHI_NORM_overall"]
    assert phi_norm["minimum"] == phi_norm["maximum"] == 1.0
    assert {"RQI_overall", "FZI_overall", "BVW_overall"} <= set(middle)
    lower = contexts["porosity", "LOWER FM. TOP"]["analysis_results"]["statistics"]
    assert lower["PHI_NORM_overall"]["maximum"] == 1.0
    assert {"RQI_overall", "FZI_overall"} <= set(lower)

    upper = contexts["permeability", "UPPER FM. TOP"]
    analysis = upper["analysis_results"]
    assert "depth_unit" not in upper["formation_info"]
    assert "depth_unit" not in analysis["lithology_group_info"]
    assert analysis["BADHOLE_QC_CHECK"]["bad_hole_sequences"] == ["1005.0-1006.0"]
    assert "KH: PERM*depth_step." in analysis["data_completeness"]["derived_log_computation_notes"]


@pytest.mark.parametrize("depth_unit", ["m", "ft"])
def test_current_context_gives_every_depth_the_well_depth_unit(tmp_path, monkeypatch, depth_unit):
    contexts = _export(tmp_path, monkeypatch, "current", depth_unit)

    for document in contexts.values():
        assert document["formation_info"]["depth_unit"] == depth_unit
        assert document["analysis_results"]["lithology_group_info"]["depth_unit"] == depth_unit
        intervals = _interval_strings(document)
        assert intervals and all(text.endswith(f" {depth_unit}") for text in intervals)
    analysis = contexts["permeability", "UPPER FM. TOP"]["analysis_results"]
    # CALI - BS = 12.0 - 8.5 = 3.5 in > 1.5 in at 1005.0-1006.0: a washout.
    assert analysis["BADHOLE_QC_CHECK"]["bad_hole_sequences"] == [f"1005.0-1006.0 {depth_unit}"]
    notes = analysis["data_completeness"]["derived_log_computation_notes"]
    assert f"KH: PERM*depth_step, in mD*{depth_unit}." in notes


def test_formation_lithology_layout_gets_the_same_current_context(tmp_path, monkeypatch):
    logs, tops = _context_logs(tmp_path / "logs", "ft")
    _interpret_with_rejected_inputs(monkeypatch)
    results = pp.export_context(
        logs, tmp_path / "context", layout="formation_lithology", tops_file=tops
    )
    assert results and all(item.success for item in results), [item.errors for item in results]
    contexts = {
        Path(item.json_file).parent.name: json.loads(Path(item.json_file).read_text(encoding="utf-8"))
        for item in results
    }

    middle = contexts["MIDDLE FM. TOP"]
    assert middle["formation_info"]["depth_unit"] == "ft"
    statistics = middle["analysis_results"]["statistics"]
    assert "PERM_overall" in statistics
    assert not {"PHI_NORM_overall", "RQI_overall", "FZI_overall", "BVW_overall"} & set(statistics)
    upper = contexts["UPPER FM. TOP"]["analysis_results"]
    assert upper["lithology_group_info"]["depth_unit"] == "ft"
    assert upper["BADHOLE_QC_CHECK"]["bad_hole_sequences"] == ["1005.0-1006.0 ft"]
