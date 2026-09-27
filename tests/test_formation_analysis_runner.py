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
