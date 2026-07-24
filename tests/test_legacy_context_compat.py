"""Context-export determinism and the archived-benchmark compatibility toggle."""

from __future__ import annotations

import pytest

from petrophys._internal.legacy_context import (
    legacy_electrofacies_v1_entry,
    legacy_benchmark_context,
    set_legacy_benchmark_context,
)
from petrophys.reporting.reference_export import (
    CATEGORY_CONTEXT_SPECS,
    _apply_legacy_electrofacies_v1_statistics,
    _input_log_display_list,
)


@pytest.fixture
def legacy_off():
    """Leave the process-wide toggle off regardless of how the test exits."""
    set_legacy_benchmark_context(False)
    yield
    set_legacy_benchmark_context(False)


def test_category_targets_are_ordered_not_sets():
    """Target order reaches derived_log_computation_notes verbatim.

    Set literals made that order depend on PYTHONHASHSEED, so two runs of the
    same well produced different note blocks in every exported file.
    """
    for category, spec in CATEGORY_CONTEXT_SPECS.items():
        targets = spec["targets"]
        assert isinstance(targets, tuple), (
            f"{category} targets must be an ordered tuple, got {type(targets).__name__}"
        )
        assert len(targets) == len(set(targets)), f"{category} targets contain duplicates"


def test_toggle_defaults_off_and_round_trips(legacy_off):
    assert legacy_benchmark_context() is False
    set_legacy_benchmark_context(True)
    assert legacy_benchmark_context() is True
    set_legacy_benchmark_context(False)
    assert legacy_benchmark_context() is False


def test_bs_reported_as_input_log_by_default(legacy_off):
    assert "BS" in _input_log_display_list({"GR", "CALI", "BS", "RHOB"})


def test_legacy_mode_omits_bs_from_input_logs(legacy_off):
    set_legacy_benchmark_context(True)
    displayed = _input_log_display_list({"GR", "CALI", "BS", "RHOB"})
    assert "BS" not in displayed
    assert {"GR", "CALI", "RHOB"}.issubset(set(displayed))


def test_legacy_electrofacies_manifest_contains_archived_statistics():
    key = (
        "15_9-12/AASGARD FM. TOP/"
        "AASGARD FM. TOP_CHALK_GROUP_1_numerical.json"
    )

    known, statistics = legacy_electrofacies_v1_entry(key)

    assert known is True
    assert statistics["arithmetic_mean"] == 2.0763
    assert statistics["mode"] == 1


def test_legacy_context_replays_archived_electrofacies_statistics(
    legacy_off,
    tmp_path,
):
    set_legacy_benchmark_context(True)
    payload = {
        "analysis_results": {
            "statistics": {
                "ELECTROFACIES_overall": {"arithmetic_mean": 999.0}
            }
        }
    }
    json_path = (
        tmp_path
        / "numerical"
        / "15_9-12"
        / "AASGARD FM. TOP"
        / "AASGARD FM. TOP_CHALK_GROUP_1_numerical.json"
    )

    _apply_legacy_electrofacies_v1_statistics(
        payload,
        json_path=json_path,
        source_file=tmp_path / "15_9-12.las",
        formation_name="AASGARD FM. TOP",
    )

    statistics = payload["analysis_results"]["statistics"][
        "ELECTROFACIES_overall"
    ]
    assert statistics["arithmetic_mean"] == 2.0763
    assert statistics["mode"] == 1
