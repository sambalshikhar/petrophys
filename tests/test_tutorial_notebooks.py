"""Lightweight tutorial contracts; fresh-kernel execution runs separately."""
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = runpy.run_path(str(ROOT / "scripts" / "execute_tutorials.py"))["USER_JOURNEY_NOTEBOOKS"]


@pytest.mark.parametrize("name", MANIFEST)
def test_user_journey_notebooks_have_unique_cells_and_no_saved_errors(name):
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    assert all(ids)
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], (str, list))
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error"


def test_quickstart_plots_before_interpreting_and_teaches_optional_geology():
    notebook = json.loads((ROOT / "examples" / "quickstart.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    prose = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert code.index("pp.plot(") < code.index("pp.interpret(")
    assert "optional" in prose.lower()
    assert "raise_for_errors" in code
    assert "result.well is not well" in code


def test_own_data_tutorial_exercises_import_and_diagnostic_contracts():
    notebook = json.loads((ROOT / "examples" / "own_data.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for marker in ("read_formations", "read_lithology_intervals", "sheet_name",
                   "depth_reference", "terminal_depth", "core_saturation_normalization",
                   "show_expected_error", "specific_steps_skipped", "to_trace_dict"):
        assert marker in code
