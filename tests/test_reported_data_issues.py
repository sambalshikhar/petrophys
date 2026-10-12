"""Regression checks for the reported real-data and import ambiguities."""

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys.evaluation.net_pay import NetPayModule
from petrophys.io import read_lithology_intervals
from petrophys.interpretation.lithology.lithology_log import load_lithology_csv
from petrophys.model.formation_tops import FormationTops
from petrophys.reporting.reference_export import _formation_segments_for_well, _load_excel_formation_segments
from petrophys.visualization.log_viewer import _lithology_mapping_from_records


def test_codebook_prefers_actual_attribute_over_group_and_allows_explicit_choice():
    records = [{"Abbreviation": 0, "Attribute": "Gap", "Lithology Attribute": "Gypsum"},
               {"Abbreviation": 36, "Attribute": "Arkose", "Lithology Attribute": "Gravel"}]
    with pytest.warns(pp.DataQualityWarning, match="different Attribute"):
        assert pp.read_lithology_codes(records) == {0: "Gap", 36: "Arkose"}
    with pytest.warns(pp.DataQualityWarning):
        assert _lithology_mapping_from_records(records)[36] == "Arkose"
    assert pp.read_lithology_codes(records, label_column="Lithology Attribute") == {0: "Gypsum", 36: "Gravel"}
    from petrophys.visualization.log_viewer import LogViewer
    with pytest.warns(pp.DataQualityWarning, match="different Attribute"):
        mapping = LogViewer._coerce_lithology_mapping(None, None, pd.DataFrame(records))
    assert mapping[36] == "Arkose"


@pytest.mark.parametrize("rows", [
    [{"code": 1, "lithology": "Sand"}, {"code": "1.0", "lithology": "Shale"}],
    [{"code": 1, "lithology": ""}], [{"code": 1.5, "lithology": "Sand"}],
])
def test_codebook_rejects_ambiguous_rows(rows):
    with pytest.raises(ValueError):
        pp.read_lithology_codes(rows)


def test_context_tops_accept_npd_headers_units_and_rank(tmp_path):
    table = pd.DataFrame({"Well identifier": ["30/6-5"] * 3,
                          "Surface": ["BRENT GP. TOP", "NESS FM. TOP", "ETIVE FM. TOP"],
                          "MD": [100, 100, 120], "depth_unit": ["ft"] * 3})
    path = tmp_path / "npd.xlsx"
    table.to_excel(path, index=False)
    segments = _load_excel_formation_segments(path, rank="formation")
    well = pp.Well("30/6-5")
    well.depth = np.array([30.0, 40.0])
    well.depth_unit = "m"
    rows = _formation_segments_for_well(tmp_path / "30_6-5.las", well, segments)
    assert len(rows) == 2
    assert rows[0]["top_depth"] == pytest.approx(30.48)
    assert rows[0]["bottom_depth"] == pytest.approx(36.576)
    assert rows[-1]["bottom_depth"] is None


def test_bundled_formations_declare_units_and_intentional_hierarchy():
    table = pp.read_formations(pp.sample("formations_30_6-5"))
    assert table.attrs["depth_unit"] == "m"
    assert set(table["rank"]) == {"group", "formation"}
    tops = FormationTops.from_csv(pp.sample("formations_30_6-5"))
    assert tops.validate() == []
    assert tops["Heather Fm."].metadata["parent"] == "VIKING GP."


def test_only_the_carbonate_label_selects_carbonate_cutoffs(tmp_path):
    well = pp.Well("CUTOFFS")
    well.depth = np.arange(5.0)
    labels = np.array(["limestone", "dolomite", "marl", "carbonate", "clastic"])
    result = NetPayModule(lithology=labels).calculate(well, vcl=np.full(5, .1), phie=np.full(5, .06), sw=np.full(5, .65))
    np.testing.assert_array_equal(result.net_pay, [False, False, False, True, False])
    path = tmp_path / "marl.csv"
    path.write_text("top,bottom,lithology\n0,5,Marl\n", encoding="utf-8")
    assert load_lithology_csv(path)[0].lithology == "carbonate"


def test_quality_mask_must_match_and_nan_is_not_usable():
    well = pp.Well("QUALITY")
    well.depth = np.arange(3.0)
    arrays = dict(vcl=np.full(3,.1), phie=np.full(3,.2), sw=np.full(3,.4))
    with pytest.raises(ValueError, match="quality_mask"):
        NetPayModule().calculate(well, **arrays, quality_mask=[True])
    result = NetPayModule().calculate(well, **arrays, quality_mask=[1, np.nan, 0])
    np.testing.assert_array_equal(result.net_pay, [True, False, False])


def test_default_uses_the_single_canonical_route_for_embedded_lithology():
    from tests.test_canonical_workflow import _workflow_well

    result = pp.interpret(_workflow_well(True), rw=.07).raise_for_errors()
    assert "interpretation_profile" not in result.to_trace_dict()
    step = next(item for item in result.steps_executed if item.step == "net_pay")
    # Sandstone and shale: the clastic standards; the benchmark profile keeps 0.3 / 0.7.
    assert (step.parameters["vcl_cutoff"], step.parameters["sw_cutoff"]) == (0.4, 0.6)
    benchmark = pp.interpret(_workflow_well(True), rw=.07, calculation_profile="benchmark")
    step = next(item for item in benchmark.raise_for_errors().steps_executed if item.step == "net_pay")
    assert (step.parameters["vcl_cutoff"], step.parameters["sw_cutoff"]) == (0.3, 0.7)


def test_scalar_cmr_does_not_invent_t2lm():
    from petrophys.interpretation.nmr import NMRData
    nmr = NMRData.from_las(pp.sample("daydream1_cmr"))
    assert not nmr.has_t2lm
    with pytest.raises(ValueError, match="T2"):
        nmr.t2_lm()
    assert np.isfinite(nmr.permeability_coates()).any()


def test_codebook_errors_support_noninteger_dataframe_index():
    table = pd.DataFrame({"code": [1], "lithology": [None]}, index=["record-a"])
    with pytest.raises(ValueError, match="row 2"):
        pp.read_lithology_codes(table)


def test_canonical_net_pay_records_clean_rock_assumption_when_vcl_is_missing():
    well = pp.Well("MISSING-CLAY")
    well.depth = np.arange(10.0)
    well.add_curve("PHIE", np.full(10, .2), "v/v", "Porosity")
    well.add_curve("SW", np.full(10, .3), "v/v", "Water saturation")
    result = pp.interpret(well)
    assert "net_pay" not in result.steps_skipped
    np.testing.assert_array_equal(result.well.get("NET_PAY").data, np.ones(10))


def test_attached_shale_name_does_not_replace_calculated_pay_flags():
    from petrophys.workflow import DynamicWorkflow, WorkflowStep
    well = pp.Well("SHALE-CONFLICT")
    well.depth = np.arange(10.0)
    well.depth_unit = "m"
    for name, value in (("VCL", .1), ("PHIE", .2), ("SW", .3)):
        well.add_curve(name, np.full(10, value), "v/v", name)
    well.set_parameter("_geology_context", {"formations": [
        {"formation_name": "Example", "top_depth": 2, "bottom_depth": 5, "lithology": "Shale"},
    ]})
    result = DynamicWorkflow().run_selected_steps(well, [WorkflowStep.NET_PAY]).raise_for_errors()
    assert well.get("NET_PAY_REVIEW") is None
    assert np.all(well.get("NET_PAY").data == 1)
