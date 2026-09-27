"""End-to-end formation analysis entry point."""

from types import SimpleNamespace

import pytest

from petrophys import sample
from petrophys.evaluation.summaries.formation_analysis import run_formation_analysis


def test_run_formation_analysis_keeps_the_workflow_result():
    result = run_formation_analysis(
        sample("30_6-5"), sample("formations_30_6-5"), rw=0.07
    )

    assert result.summaries
    assert result.workflow_result is not None
    assert result.workflow_result.status == "interpreted"


def test_run_formation_analysis_refuses_failed_interpretations(monkeypatch):
    def failing_workflow(*args, **kwargs):
        def raise_for_errors():
            raise RuntimeError("Workflow failed: porosity: boom")

        return SimpleNamespace(raise_for_errors=raise_for_errors)

    monkeypatch.setattr("petrophys.workflow.run_dynamic_workflow", failing_workflow)

    with pytest.raises(RuntimeError, match="Workflow failed"):
        run_formation_analysis(sample("30_6-5"), sample("formations_30_6-5"), rw=0.07)


def _npd_style_tops(tmp_path):
    path = tmp_path / "npd_tops.csv"
    path.write_text(
        "uwi,form,top\n"
        "30/6-5,VIKING GP. Top,2452\n"
        "30/6-5,Draupne Fm. Top,2452\n"
        "30/6-5,Heather Fm. Top,2593\n"
        "30/6-5,BRENT GP. Top,2841\n"
        "30/6-5,Ness Fm. Top,2841\n"
        "30/6-5,Etive Fm. Top,2865\n"
        "30/6-5,DUNLIN GP. Top,2958\n"
        "30/6-5,Drake Fm. Top,2958\n"
    )
    return path


def test_npd_style_names_carry_their_stratigraphic_rank(tmp_path):
    import petrophys as pp

    table = pp.read_formations(_npd_style_tops(tmp_path), well_name="30/6-5")
    assert table.set_index("formation_name")["rank"].to_dict()["BRENT GP."] == "group"
    assert table.set_index("formation_name")["rank"].to_dict()["Etive Fm."] == "formation"
    assert any("rank inferred" in note for note in table.attrs["import_report"]["diagnostics"])


def test_formation_analysis_selects_one_level_of_mixed_tops(tmp_path):
    tops = _npd_style_tops(tmp_path)
    with pytest.raises(ValueError, match="rank=.*'group', 'formation'"):
        run_formation_analysis(sample("30_6-5"), tops, well_name="30/6-5", rw=0.07)

    groups = run_formation_analysis(sample("30_6-5"), tops, well_name="30/6-5", rank="group", rw=0.07)
    formations = run_formation_analysis(sample("30_6-5"), tops, well_name="30/6-5", rank="formation", rw=0.07)
    assert [f.name for f in groups.formations] == ["VIKING GP.", "BRENT GP.", "DUNLIN GP."]
    assert "Etive Fm." in [f.name for f in formations.formations]
