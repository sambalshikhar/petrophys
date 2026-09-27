import numpy as np
import pytest

from petrophys.model import Well
from petrophys.model.core import CoreData
from petrophys.io import read_ascii, read_csv, read_las
from petrophys.interpretation.lithology import build_lithology_log, load_lithology_csv
from petrophys.interpretation.nmr.permeability import t2lm_from_bins
from petrophys.interpretation.saturation.archie import archie
from petrophys.evaluation.summaries.formation_analysis import load_formations_csv
from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well
from petrophys._internal.interpolation import fill_gaps


def test_read_csv_preserves_negative_log_values_and_custom_nulls(tmp_path):
    path = tmp_path / "messy_logs.csv"
    path.write_text(
        "md,sp,gr\n"
        "1000.0,-42.5,80\n"
        "1000.5,MISSING,-999.25\n"
        "1001.0,-10.0,95\n"
    )

    well = read_csv(path, depth_column="md", depth_unit="m", null_values=["MISSING"])

    assert well.name == "messy_logs"
    assert well.depth_unit == "m"
    assert well.n_samples == 3
    np.testing.assert_allclose(well["SP"].data[[0, 2]], [-42.5, -10.0])
    assert np.isnan(well["SP"].data[1])
    assert np.isnan(well["GR"].data[1])


def test_read_csv_skips_non_numeric_metadata_columns(tmp_path):
    path = tmp_path / "curated_logs.csv"
    path.write_text(
        "DEPT,WELLNAME,FORMATION,GR,DENS\n"
        "1000.0,Kapuni-15,Sandstone,85.0,2.45\n"
        "1000.5,Kapuni-15,Shale,95.0,2.50\n"
    )

    well = read_csv(path, depth_column="DEPT", depth_unit="m")

    assert well.n_samples == 2
    assert "GR" in well
    assert "DENS" in well
    assert "WELLNAME" not in well
    assert "FORMATION" not in well
    assert well.get_parameter("SKIPPED_NON_NUMERIC_COLUMNS") == ["FORMATION", "WELLNAME"]
    assert well.get_parameter("SKIPPED_NON_NUMERIC_VALUES")["WELLNAME"] == ["Kapuni-15"]


def test_read_las_normalizes_f_depth_unit_to_ft(tmp_path):
    path = tmp_path / "feet_unit_f.las"
    path.write_text(
        "\n".join(
            [
                "~VERSION INFORMATION",
                "VERS. 2.0 : CWLS Log ASCII Standard",
                "WRAP. NO",
                "~WELL INFORMATION",
                "STRT.F 1000.0000 : START DEPTH",
                "STOP.F 1001.0000 : STOP DEPTH",
                "STEP.F 0.5000 : STEP",
                "NULL. -999.25 : NULL VALUE",
                "WELL. F-DEPTH : WELL",
                "~CURVE INFORMATION",
                "DEPT.F : DEPTH",
                "GR.API : Gamma Ray",
                "~A",
                "1000.0000 80.0",
                "1000.5000 82.0",
                "1001.0000 85.0",
            ]
        ),
        encoding="utf-8",
    )

    well = read_las(path)

    assert well.depth_unit == "ft"
    np.testing.assert_allclose(well.depth, [1000.0, 1000.5, 1001.0])


def test_read_ascii_skips_units_row_below_header_and_assigns_units(tmp_path):
    path = tmp_path / "space_ascii.txt"
    path.write_text(
        "DEPTH GR SP\n"
        "m API mV\n"
        "1000.0 45.0 -30.0\n"
        "1000.5 -999.25 MISSING\n"
    )

    well = read_ascii(path, units_line=1, null_values=["MISSING"])

    assert well.depth_unit == "m"
    assert well.n_samples == 2
    assert well["GR"].unit == "API"
    assert well["SP"].unit == "mV"
    assert well["SP"].data[0] == -30.0
    assert np.isnan(well["GR"].data[1])
    assert np.isnan(well["SP"].data[1])


def test_core_csv_preserves_extra_metadata_and_filters_it_by_depth(tmp_path):
    path = tmp_path / "core.csv"
    path.write_text(
        "DEPTH,CPOR,CPERM,PLUG_ID\n"
        "1000.0,0.18,12.0,A\n"
        "1001.0,0.21,18.0,B\n"
        "1002.0,0.16,8.0,C\n"
    )

    core = CoreData.from_csv(path)
    filtered = core.filter_by_depth(1000.5, 1002.0)

    np.testing.assert_allclose(filtered.depth, [1001.0, 1002.0])
    np.testing.assert_allclose(filtered.porosity, [0.21, 0.16])
    assert filtered.metadata["PLUG_ID"].tolist() == ["B", "C"]


def test_lithology_csv_synonyms_work_with_descending_depth_log(tmp_path):
    path = tmp_path / "lithology.csv"
    path.write_text(
        "from,to,rock_type\n"
        "100,110,Dolomite\n"
        "110,120,Siltstone\n"
        "not-a-depth,130,Limestone\n"
        "120,130,Unknown\n"
    )

    intervals = load_lithology_csv(path)
    lithology = build_lithology_log(np.array([125.0, 115.0, 105.0]), intervals)

    assert [interval.lithology for interval in intervals] == [
        "carbonate",
        "clastic",
        "clastic",
    ]
    assert lithology.tolist() == ["clastic", "clastic", "carbonate"]


def test_formation_csv_auto_detects_tops_table_filters_well_and_builds_intervals(tmp_path):
    path = tmp_path / "tops.csv"
    path.write_text(
        "Well identifier,Surface,MD,Geological age\n"
        "A-1,Alpha Top,1000,Jurassic\n"
        "B-1,Other Top,1005,Cretaceous\n"
        "A-1,Beta Top,1050,Triassic\n"
    )

    with pytest.raises(ValueError, match="terminal_depth"):
        load_formations_csv(path, well_name="A-1")
    formations = load_formations_csv(path, well_name="A-1", terminal_depth=1150)

    assert [formation.name for formation in formations] == ["Alpha", "Beta"]
    assert formations[0].top_depth == 1000.0
    assert formations[0].bottom_depth == 1050.0
    assert formations[1].bottom_depth == 1150.0


def test_fill_gaps_respects_max_gap_while_preserving_negative_values():
    filled = fill_gaps(
        np.array([-10.0, np.nan, -6.0, np.nan, np.nan, np.nan, 2.0]),
        max_gap=1,
    )

    assert filled[0] == -10.0
    assert filled[1] == -8.0
    assert filled[2] == -6.0
    assert np.isnan(filled[3:6]).all()
    assert filled[6] == 2.0


def test_archie_clips_nonphysical_negative_inputs_to_finite_saturation():
    sw = archie(
        rt=np.array([-5.0, 10.0]),
        rw=0.05,
        phi=np.array([0.20, -0.10]),
    )

    assert np.isfinite(sw).all()
    np.testing.assert_allclose(sw, [1.0, 1.0])


def test_t2lm_from_bins_returns_nan_for_zero_distribution_and_keeps_rows():
    t2lm = t2lm_from_bins(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.array([1.0, 10.0, 100.0]),
    )

    assert t2lm.shape == (2,)
    assert np.isnan(t2lm[0])
    assert t2lm[1] == pytest.approx(10.0)


def test_hcpv_from_well_returns_zero_volumes_when_net_pay_is_zero():
    well = Well("NO-PAY")
    well.depth = np.array([1000.0, 1001.0, 1002.0])
    well.add_curve("NET_PAY", np.array([0.0, 0.0, 0.0]))
    well.add_curve("PHIE", np.array([0.20, 0.21, 0.19]))
    well.add_curve("SW", np.array([0.40, 0.42, 0.38]))

    result = calculate_hcpv_from_well(well, area_acres=40.0)

    assert result.net_pay_ft == 0.0
    assert result.hcpv_acft == 0.0
    assert result.hcpv_bbl == 0.0
    assert result.ooip_stb == 0.0
    assert result.ogip_bcf == 0.0
