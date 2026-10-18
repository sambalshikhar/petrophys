"""Regression tests for review fixes in helpers, inputs, aliases and the MCP CLI."""

import copy
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import petrophys as pp
from petrophys._internal.analysis_helpers import safe_divide
from petrophys._internal.benchmark import historical_benchmark
from petrophys._internal.statistics import geometric_mean, harmonic_mean
from petrophys.interpretation import calculate_porosity, calculate_vcl
from petrophys.preprocessing.standardization import add_standard_curve_aliases

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def well_30_6_5():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pp.read(pp.sample("30_6-5"))


def _ait_well(well):
    """30/6-5 with its deep/medium/shallow curves under AIT mnemonics."""
    well = copy.deepcopy(well)
    deep, medium, shallow = (well.get(n).data.copy() for n in ("RDEP", "RMED", "RSHA"))
    for name in ("RDEP", "RMED", "RSHA"):
        well._curves.pop(name)
    well.add_curve("AT60", medium, "ohm.m")
    well.add_curve("AT90", deep, "ohm.m")
    well.add_curve("AT10", shallow, "ohm.m")
    return well


def test_safe_divide_broadcasts_scalars_and_arrays():
    np.testing.assert_allclose(safe_divide(np.array([1.0, 2.0]), 2.0), [0.5, 1.0])
    np.testing.assert_allclose(safe_divide(2.0, np.array([1.0, 4.0])), [2.0, 0.5])
    out = safe_divide(np.array([[1.0], [2.0]]), np.array([1.0, 0.0]))
    assert out.shape == (2, 2)
    assert np.isnan(out[:, 1]).all()


def test_means_propagate_nan_when_not_ignored():
    assert np.isnan(harmonic_mean([10.0, np.nan, 100.0], ignore_nan=False))
    assert np.isnan(geometric_mean([10.0, np.nan, 100.0], ignore_nan=False))
    assert harmonic_mean([10.0, np.nan, 100.0]) == pytest.approx(2 / (0.1 + 0.01))


def test_calculate_vcl_rejects_inverted_gr_bounds(well_30_6_5):
    with pytest.raises(ValueError, match="gr_min"):
        calculate_vcl(well_30_6_5, gr_min=120, gr_max=30)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"matrix": "sandston"}, ValueError),
        ({"matrix": "xyz"}, ValueError),
        ({"rhob_fluid": 1000.0}, ValueError),
        ({"phi_clay": 15.0}, ValueError),
        ({"rhobfluid": 1.1}, TypeError),
    ],
)
def test_calculate_porosity_rejects_bad_inputs(well_30_6_5, kwargs, error):
    with pytest.raises(error):
        calculate_porosity(well_30_6_5, **kwargs)


def test_calculate_porosity_accepts_aliases(well_30_6_5):
    result = calculate_porosity(well_30_6_5, matrix="ss", rhob_fluid=1.1)
    assert np.isfinite(result.phid).any()


def test_rdep_alias_prefers_at90_over_at60(well_30_6_5):
    well = _ait_well(well_30_6_5)
    aliases = add_standard_curve_aliases(copy.deepcopy(well))
    assert aliases.get("AT90") == "RDEP" and "AT60" not in aliases
    # The benchmark packages keep the earlier choice.
    with historical_benchmark():
        assert add_standard_curve_aliases(copy.deepcopy(well)).get("AT60") == "RDEP"
    # Equal sample counts: the alias ranking (AT90 before AT60) decides.
    assert add_standard_curve_aliases(
        copy.deepcopy(well), prefer_deepest_array=False
    ).get("AT90") == "RDEP"


def _vclay_decisions(result):
    step = next(s for s in result.steps_executed if getattr(s, "step", None) == "vclay")
    return [d for d in step.decisions if d.startswith("GR_m")]


def test_user_gr_bounds_not_labelled_as_percentiles(well_30_6_5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pp.interpret(copy.deepcopy(well_30_6_5), rw=0.07, rw_meas_temp_f=230, gr_min=30)
    decisions = _vclay_decisions(result)
    assert "GR_min = 30.0 (user)" in decisions
    assert any(d.startswith("GR_max") and "(auto, P95 of GR)" in d for d in decisions)


def test_mcp_serve_reports_missing_extra(monkeypatch, capsys):
    from petrophys.mcp import cli

    monkeypatch.setattr(cli, "_missing_server_modules", lambda transport: ["pydantic"])
    assert cli.main(["serve", "--http"]) == 2
    err = capsys.readouterr().err
    assert "petrophys[mcp]" in err and "Python 3.10" in err
    assert "server (" not in err  # no banner before the failure


def _script(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / args[0]), *args[1:]],
        capture_output=True, text=True, timeout=300,
    )


def test_trace_script_rejects_missing_input_dir(tmp_path):
    proc = _script("generate_workflow_json_traces.py", "--input-dir", str(tmp_path / "nope"))
    assert proc.returncode == 2 and "input directory not found" in proc.stderr


def test_pattern_key_script_help_writes_nothing():
    proc = _script("generate_lithology_pattern_key.py", "--help")
    assert proc.returncode == 0 and "usage" in proc.stdout and "wrote" not in proc.stdout


def test_check_notebook_outputs_missing_file(tmp_path):
    proc = _script("check_notebook_outputs.py", str(tmp_path / "nope.ipynb"))
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
