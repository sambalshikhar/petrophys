"""Regression tests for reader/exporter defects fixed before release."""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys._internal.benchmark import historical_benchmark
from petrophys.io import export_excel, export_las, read_ascii, read_csv, read_csv_wells, read_dlis, read_las
from petrophys.exceptions import DataQualityWarning
from petrophys.io import read_formations
from petrophys.io.ascii_reader import detect_file_format
from petrophys.model.well import Well


def _sample_lines():
    return pp.sample("30_6-5").read_text(encoding="utf-8").splitlines()


def _write_variant(tmp_path, name, edit):
    lines = _sample_lines()
    index = next(i for i, line in enumerate(lines) if line.lstrip().upper().startswith("DEPT"))
    lines[index] = edit(lines[index])
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_blank_las_index_unit_taken_from_strt_stop_step(tmp_path):
    path = _write_variant(tmp_path, "blank.las", lambda line: re.sub(r"\.m", ".  ", line, count=1))
    well = read_las(path)
    assert well.depth_unit == "m"
    tops = pp.read(path, formations=pp.sample("formations_30_6-5"))
    reference = pp.read(pp.sample("30_6-5"), formations=pp.sample("formations_30_6-5"))
    first = tops.get_parameter("_geology_context")["formations"][0]["top_depth"]
    assert first == pytest.approx(reference.get_parameter("_geology_context")["formations"][0]["top_depth"])
    with historical_benchmark():
        assert read_las(path).depth_unit == "ft"


def test_las_index_recognised_as_first_curve(tmp_path):
    path = _write_variant(tmp_path, "tdep.las", lambda line: line.replace("DEPT", "TDEP", 1))
    well = read_las(path)
    reference = read_las(pp.sample("30_6-5"))
    assert well.depth is not None and "TDEP" not in well.curves
    assert well.depth_unit == "m"
    assert np.allclose(well.depth, reference.depth)


_ZONED_BHT_LAS = """~Version
VERS. 2.0 : CWLS LAS
WRAP. NO : one line
~Well
STRT.m 1000.0 : start
STOP.m 1001.0 : stop
STEP.m 0.5 : step
NULL. -999.25 : null
WELL. T : well
~Params
BHT_Z1.{unit} {value} : zoned BHT
~Curve
DEPT.m : depth
GR.gAPI : gamma
~A
1000.0 50
1000.5 60
1001.0 70
"""


def test_zoned_bht_in_kelvin_is_converted(tmp_path):
    path = tmp_path / "bhtk.las"
    path.write_text(_ZONED_BHT_LAS.format(unit="K", value="373.15"))
    well = read_las(path)
    assert well.header.bht == pytest.approx(212.0)
    assert well.header.bht_unit == "F"


def test_zoned_bht_unknown_unit_not_promoted(tmp_path):
    path = tmp_path / "bhtx.las"
    path.write_text(_ZONED_BHT_LAS.format(unit="R", value="671.67"))
    with pytest.warns(UserWarning, match="not recognised"):
        well = read_las(path)
    assert well.header.bht is None


def test_read_dlis_rejects_unknown_options(tmp_path):
    with pytest.raises(TypeError, match="bogus_option"):
        read_dlis(tmp_path / "missing.dlis", bogus_option=1)


def test_read_csv_wells_numeric_ids_across_chunks(tmp_path):
    path = tmp_path / "wells.csv"
    path.write_text("DEPTH,GR,UWI\n1,10,1001\n2,20,1001\n3,30,\n4,40,1001\n5,50,1002\n6,60,0012\n")
    with pytest.warns(UserWarning, match="1 row"):
        wells = read_csv_wells(path, depth_unit="m", chunksize=2)
    assert {name: list(well.depth) for name, well in wells.items()} == {
        "1001": [1.0, 2.0, 4.0], "1002": [5.0], "0012": [6.0],
    }


def test_read_csv_warns_on_coerced_values(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text('DEPTH,GR\n1,"1,234"\n2,56\n3,7*\n')
    with pytest.warns(UserWarning, match="read as missing"):
        well = read_csv(path, depth_unit="m")
    assert np.isnan(well["GR"].data[[0, 2]]).all()


def test_read_ascii_warns_on_conflicting_depth_unit(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("DEPTH GR\nM GAPI\n1000 50\n1000.5 60\n")
    with pytest.warns(UserWarning, match="differs from depth_unit"):
        well = read_ascii(path, units_line=1, depth_unit="ft")
    assert well.depth_unit == "m"


def test_detect_file_format_comments_and_header_only(tmp_path):
    commented = tmp_path / "c.txt"
    commented.write_text("# a\n# b\nDEPTH,GR\nFT,API\n1,5\n2,6\n")
    info = detect_file_format(commented)
    assert (info["skiprows"], info["header_line"], info["delimiter"], info["has_units"]) == (2, 0, ",", True)
    header_only = tmp_path / "h.txt"
    header_only.write_text("DEPTH,GR\n")
    info = detect_file_format(header_only)
    assert (info["skiprows"], info["delimiter"]) == (0, ",")


def _irregular_well():
    well = Well.from_dataframe(
        pd.DataFrame({"DEPTH": [1.0, 2.0, np.nan, 4.0], "GR": [1.0, 2.0, 3.0, 4.0]}),
        depth_unit="ft",
    )
    well.header.county = "Rawlins"
    well.set_parameter("dlis_merge", {"frames": ["A"]})
    return well


def test_export_las_irregular_step_null_depth_and_header(tmp_path):
    out = tmp_path / "irr.las"
    with pytest.warns(UserWarning, match="without a finite depth"):
        export_las(_irregular_well(), out)
    text = out.read_text()
    assert re.search(r"^STEP\.ft\s+0\s", text, re.M)
    assert "dlis_merge" not in text.lower()
    back = read_las(out)
    assert list(back.depth) == [1.0, 2.0, 4.0]
    assert list(back["GR"].data) == [1.0, 2.0, 4.0]
    assert back.header.county == "Rawlins"


def test_export_las_benchmark_output_unchanged(tmp_path):
    out = tmp_path / "irr_hist.las"
    with historical_benchmark():
        export_las(_irregular_well(), out)
    text = out.read_text()
    assert re.search(r"^STEP\.ft\s+1\.00000", text, re.M)
    assert "Rawlins" not in text and "dlis_merge" in text.lower()


def test_export_excel_nulls_infinite_values(tmp_path):
    pytest.importorskip("openpyxl")
    well = Well.from_dataframe(
        pd.DataFrame({"DEPTH": [1.0, 2.0, 3.0], "GR": [1.0, np.inf, -np.inf]}), depth_unit="m"
    )
    export_excel(well, tmp_path / "out.xls")
    path = tmp_path / "out.xlsx"
    assert path.exists() and not (tmp_path / "out.xls").exists()
    data = pd.read_excel(path, sheet_name="Log Data")
    assert data["GR"].isna().tolist() == [False, True, True]
    info = pd.read_excel(path, sheet_name="Curve Info").iloc[0]
    assert (info["Min"], info["Max"], info["Valid Count"]) == (1, 1, 1)
    header = pd.read_excel(path, sheet_name="Header")
    assert "m" in header.loc[header["Parameter"] == "depth_unit", "Value"].tolist()


def test_geology_without_unit_warns_when_other_unit_fits(tmp_path):
    well = read_las(pp.sample("30_6-5"))  # logged 176.6-3556.1 m
    tops = tmp_path / "tops.csv"
    # Tops written in feet: in metres two of three lie below the log.
    tops.write_text("formation,top,base\nA,200,300\nB,5000,6000\nC,8000,9000\n")
    with pytest.warns(DataQualityWarning, match="no geology depth_unit"):
        read_formations(tops, well=well)
