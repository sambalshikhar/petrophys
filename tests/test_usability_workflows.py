"""Own-data journeys shared by plotting and interpretation."""
import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys.evaluation.summaries.formation_analysis import analyze_formations
from petrophys.io import read_csv, read_las
from petrophys.model.formation_tops import FormationTops
from petrophys.model.intervals import Formation
from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig


def _well():
    well = pp.Well("A-1")
    well.depth = np.arange(1000., 1021.)
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(20., 100., 21), "API")
    well.add_curve("RHOB", np.linspace(2.3, 2.6, 21), "g/cc")
    well.add_curve("NPHI", np.linspace(.15, .3, 21), "v/v")
    well.add_curve("RT", np.linspace(10., 30., 21), "ohm.m")
    return well


def _lithology():
    return pd.DataFrame({
        "top": [1000., 1008., 1016.], "bottom": [1005., 1016., 1021.],
        "lithology": ["Sandstone", "Marl", "Unmapped rock"],
    })


def test_external_lithology_runs_without_writing_strings_to_curves():
    raw = _well()
    result = pp.interpret(raw, lithology_intervals=_lithology(), rw=.08)
    result.raise_for_errors()
    assert result.well is not raw
    assert "LITHO_CLASS" not in raw.curves
    curve = result.well["LITHO_CLASS"]
    assert curve.data.dtype.kind == "f"
    assert curve.metadata.custom["category_labels"] == {
        1: "Sandstone", 2: "Marl", 3: "Unmapped rock",
    }
    assert np.isnan(curve.data[5:8]).all()
    report = result.well.get_parameter("_geology_context")["lithology_selection"]
    assert report["unmapped_labels"] == ["Unmapped rock"]
    assert report["uncovered_samples"] == 3
    assert result.warnings


@pytest.mark.parametrize("policy,selected", [
    ("auto", "external"), ("external", "external"),
    ("vendor", "vendor"), ("computed", "computed"), ("config", "config"),
])
def test_lithology_policy_agrees_between_workflow_and_viewer(policy, selected):
    raw = _well()
    raw.add_curve("LITHOLOGY_GEOLINK", np.ones(21))
    options = {"lithology_source": policy, "lithology_intervals": _lithology(), "rw": .08}
    if policy == "config":
        options.update(matrix_type="limestone", lithology="carbonate")
    result = pp.interpret(raw, **options).raise_for_errors()
    assert result.well.get_parameter("_geology_context")["selected_lithology_source"] == selected
    np.testing.assert_array_equal(result.well["LITHOLOGY_GEOLINK"].data, raw["LITHOLOGY_GEOLINK"].data)
    viewer = result.plot()
    try:
        assert viewer.diagnostics["lithology_source"] == selected
        assert (viewer.lithology_intervals is not None) == (selected == "external")
    finally:
        viewer.close()


def test_las_only_entry_points_and_copy_contract():
    raw = _well()
    before = list(raw.curves)
    viewer = pp.plot(raw)
    viewer.close()
    result = pp.interpret(raw, rw=.08).raise_for_errors()
    assert list(raw.curves) == before
    assert result.well is not raw
    assert "LITHO" in result.well.curves
    assert pp.interpret(raw, inplace=True, rw=.08).well is raw


def test_workflow_errors_are_actionable_and_options_are_not_ignored():
    result = pp.interpret(_well(), lithology_source="external")
    assert not result.success
    with pytest.raises(RuntimeError, match="requires lithology"):
        result.raise_for_errors()
    with pytest.raises(ValueError, match="requires lithology"):
        pp.interpret(_well(), lithology_source="external", raise_on_error=True)
    with pytest.raises(TypeError):
        pp.interpret(_well(), imaginary_setting=True)
    config = DynamicWorkflowConfig()
    with pytest.raises(TypeError):
        pp.interpret(_well(), config, imaginary_setting=True)
    assert config.matrix_type == "sandstone"


@pytest.mark.parametrize("headers", [
    {"formation_name": "formation", "top_depth": "top"},
    {"formation_name": "formation", "top_depth": "md_top"},
    {"formation_name": "Surface", "top_depth": "MD"},
])
def test_formation_formats_have_identical_plot_and_analysis_inputs(headers):
    table = pd.DataFrame({"formation_name": ["Alpha", "Beta"], "top_depth": [1000, 1010]}).rename(columns=headers)
    canonical = pp.read_formations(table, well=_well())
    with pytest.raises(ValueError, match="terminal_depth"):
        FormationTops.from_dataframe(canonical)
    model = FormationTops.from_dataframe(canonical, terminal_depth=1020)
    viewer = pp.plot(_well(), formation_tops=model)
    try:
        assert [item.name for item in viewer.formation_tops] == ["Alpha", "Beta"]
    finally:
        viewer.close()
    result = analyze_formations(_well(), canonical, terminal_depth=1020)
    assert len(result.formations) == 2


def test_multiwell_selection_keeps_identifiers_as_text(tmp_path):
    path = tmp_path / "tops.csv"
    path.write_text("well,formation,md_top\n001,Alpha,1000\n002,Beta,1010\n")
    with pytest.raises(ValueError, match="Multiple wells"):
        pp.read_formations(path)
    selected = pp.read_formations(path, well_name="001")
    assert selected.well.tolist() == ["001"]
    assert selected.attrs["import_report"]["filtered_rows"] == 1
    with pytest.raises(ValueError, match="No matching"):
        pp.read_formations(path, well=_well())


def test_depth_units_convert_once_and_reference_changes_require_a_survey():
    well = _well()
    well.set_parameter("depth_reference", "MD")
    well.set_parameter("datum", "RKB")
    table = pd.DataFrame({
        "formation": ["Alpha"], "top": [3300.], "bottom": [3310.],
        "depth_unit": ["ft"], "depth_reference": ["MD"], "datum": ["RKB"],
    })
    first = pp.read_formations(table, well=well)
    second = pp.read_formations(first, well=well)
    assert first.top_depth.iloc[0] == pytest.approx(1005.84)
    assert second.top_depth.iloc[0] == first.top_depth.iloc[0]
    assert second.depth_unit.iloc[0] == "m"
    with pytest.raises(ValueError, match="transformation"):
        pp.read_formations(table.assign(depth_reference="TVDSS"), well=well)
    with pytest.raises(ValueError, match="set well.set_parameter"):
        pp.read_formations(table, well=_well())


def test_custom_mapping_delimiter_and_sheet_selection(tmp_path):
    table = pd.DataFrame({"Start": [1000], "Stop": [1020], "Rock": ["Marl"], "Vendor notes": ["original"]})
    path = tmp_path / "rocks.txt"
    table.to_csv(path, sep=";", index=False)
    options = {"columns": {"top": "Start", "bottom": "Stop", "lithology": "Rock"}}
    actual = pp.read_lithology_intervals(path, sep=";", **options)
    assert actual.lithology.tolist() == ["Marl"]
    assert actual.attrs["import_report"]["ignored_columns"] == ["Vendor notes"]
    path = tmp_path / "rocks.xlsx"
    with pd.ExcelWriter(path) as writer:
        table.to_excel(writer, sheet_name="A", index=False)
        table.to_excel(writer, sheet_name="B", index=False)
    with pytest.raises(ValueError, match="sheet_name"):
        pp.read_lithology_intervals(path, **options)
    assert len(pp.read_lithology_intervals(path, sheet_name="B", **options)) == 1


def test_formation_totals_use_union_and_adjacent_samples_are_not_duplicated():
    well = _well()
    well.add_curve("NET_PAY", np.ones(21))
    parent = Formation("Group", 1000, 1020)
    child = Formation("Member", 1005, 1015)
    group_only = analyze_formations(well, [parent])
    nested = analyze_formations(well, [parent, child])
    assert nested.total_gross == group_only.total_gross
    assert nested.total_net_pay == group_only.total_net_pay
    adjacent = analyze_formations(well, [
        Formation("A", 1000, 1010), Formation("B", 1010, 1020),
    ])
    assert sum(row.zone_summary.gross_thickness for row in adjacent.summaries) == adjacent.total_gross
    viewer = pp.plot(well, formation_tops={"Group": 1000, "Formation": 1000})
    try:
        assert len(viewer.formation_tops) == 2
        assert any("Group\nFormation" in text.get_text() for ax in viewer.axes for text in ax.texts)
    finally:
        viewer.close()


def test_invalid_imports_fail_instead_of_silently_dropping_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.plot(_well(), formation_tops=tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="row 2"):
        pp.read_formations(pd.DataFrame({"formation": ["A"], "top": ["bad"]}))
    path = tmp_path / "logs.csv"
    pd.DataFrame({"MD": [1., 2.], "GR": [30., 40.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="depth_column"):
        read_csv(path)
    with pytest.raises(ValueError, match="depth_unit"):
        pp.read(path, depth_column="MD")
    assert pp.read(path, depth_column="MD", depth_unit="m").depth_unit == "m"
    with pytest.raises(TypeError, match="Unsupported"):
        read_las(pp.sample("30_6-5"), invented_argument=True)


def test_real_las_with_attached_optional_context_uses_only_public_api():
    well = pp.read(
        pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"),
        lithology=pp.sample("lithology_30_6-5"),
    )
    viewer = pp.plot(well, depth_range=(2800, 3000))
    try:
        assert len(viewer.formation_tops) == 20
        assert viewer.lithology_source == "external"
        assert viewer.lithology_intervals is not None
        assert viewer.diagnostics["formations"]["accepted_rows"] == 20
    finally:
        viewer.close()


def test_rank_selection_and_parent_containment_are_explicit():
    table = pd.DataFrame({
        "formation_name": ["Group", "Alpha", "Beta"],
        "top_depth": [1000, 1000, 1010],
        "bottom_depth": [1020, 1010, 1020],
        "rank": ["group", "formation", "formation"],
        "parent": ["", "Group", "Group"],
    })
    model = FormationTops.from_dataframe(table)
    assert model.validate() == []
    peers = table.copy()
    peers.loc[1, "bottom_depth"] = 1012
    assert FormationTops.from_dataframe(peers).has_overlaps()
    picks = table.drop(columns="bottom_depth")
    with pytest.raises(ValueError, match="single rank"):
        FormationTops.from_dataframe(picks, terminal_depth=1020)
    formations = FormationTops.from_dataframe(picks, rank="formation", terminal_depth=1020)
    assert len(formations) == 2
    assert formations.intervals[-1].bottom_depth == 1020
    source = Formation("Age", 1000, 1020, age="Jurassic")
    assert pp.read_formations([source]).age.tolist() == ["Jurassic"]


def test_track_selection_has_one_viewer_contract_and_rejects_unknown_names():
    viewer = pp.plot(_well())
    name = next(track.display_name for track in viewer.template.tracks if not track.depth_track)
    viewer.close()
    selected = pp.plot(_well(), tracks=[name])
    try:
        assert len(selected.axes) == 1
        assert [track.display_name for track in selected.template.tracks] == [name]
    finally:
        selected.close()
    with pytest.raises(ValueError, match="Select unique track names"):
        pp.plot(_well(), tracks=["invented track"])


def test_public_api_keeps_depth_metadata_after_attaching_context(tmp_path):
    path = tmp_path / "logs.csv"
    pd.DataFrame({"DEPTH": [1000, 1005, 1010], "GR": [30, 40, 50]}).to_csv(path, index=False)
    geology = pd.DataFrame({"top": [3280.84], "bottom": [3313.65], "lithology": ["Marl"]})
    well = pp.read(
        path, depth_unit="m", depth_reference="MD", datum="RKB",
        lithology=geology,
        lithology_options={"depth_unit": "ft", "depth_reference": "MD", "datum": "RKB"},
    )
    viewer = pp.plot(well)
    try:
        assert viewer.lithology_intervals.top.iloc[0] == pytest.approx(1000, abs=.001)
        assert viewer.lithology_intervals.attrs["datum"] == "RKB"
    finally:
        viewer.close()


def test_interpret_uses_attached_single_level_formations():
    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"), formations={"UPPER": 2000.0, "BRENT": 2841.0})
    result = pp.interpret(well, rw=0.07)

    step = next(s for s in result.steps_executed if s.step == "input_context")
    assert step.parameters["formations"] == {
        "source": "attached", "used": True, "intervals": ["UPPER", "BRENT"],
    }
    assert [i.name for i in result.config.formation_tops.intervals] == ["UPPER", "BRENT"]


def test_interpret_explains_why_hierarchical_formations_are_not_used():
    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
    result = pp.interpret(well, rw=0.07)

    step = next(s for s in result.steps_executed if s.step == "input_context")
    assert step.parameters["formations"]["used"] is False
    assert "overlapping" in step.parameters["formations"]["reason"]
    assert any("formation_tops=" in decision for decision in step.decisions)
    assert result.config.formation_tops is None
