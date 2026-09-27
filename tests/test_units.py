"""Shared unit parsing used by readers, the Well model and the workflow."""

import numpy as np
import pandas as pd
import pytest

from petrophys import Well
from petrophys.io import read_formations
from petrophys.io.las_reader import _normalize_depth_unit
from petrophys.model.units import normalize_depth_unit, temperature_unit


@pytest.mark.parametrize(
    ("label", "expected"),
    [("m", "m"), ("M", "m"), ("metres", "m"), ("Meter", "m"), (" m ", "m"),
     ("ft", "ft"), ("F", "ft"), ("FT", "ft"), ("ft.", "ft"), ("Feet", "ft"), ("foot", "ft")],
)
def test_depth_unit_aliases_normalize(label, expected):
    assert normalize_depth_unit(label) == expected


def test_unknown_depth_unit_is_rejected_with_guidance():
    with pytest.raises(ValueError, match="use 'm' \\(metres\\) or 'ft' \\(feet\\)"):
        normalize_depth_unit("furlong")


def test_lenient_depth_unit_keeps_unknown_labels_and_applies_default():
    assert normalize_depth_unit("0.1 in", strict=False) == "0.1 in"
    assert normalize_depth_unit("", strict=False, default="ft") == "ft"


def test_well_depth_unit_is_validated_and_normalized():
    well = Well("UNITS")
    well.depth_unit = "Metres"
    assert well.depth_unit == "m"
    with pytest.raises(ValueError, match="Unsupported depth unit"):
        well.depth_unit = "furlong"


def test_from_dataframe_validates_depth_unit():
    frame = pd.DataFrame({"DEPTH": [1.0, 2.0], "GR": [50.0, 60.0]})
    assert Well.from_dataframe(frame, depth_unit="F").depth_unit == "ft"
    with pytest.raises(ValueError, match="Unsupported depth unit"):
        Well.from_dataframe(frame, depth_unit="furlong")


def test_geology_reader_accepts_the_same_aliases_as_the_well():
    tops = pd.DataFrame({"formation": ["Brent", "Dunlin"], "top": [100.0, 200.0], "bottom": [200.0, 300.0]})
    table = read_formations(tops, depth_unit="F")
    assert table.attrs["depth_unit"] == "ft"


def test_las_reader_warns_about_unrecognized_depth_units():
    with pytest.warns(UserWarning, match="Unrecognized LAS depth unit"):
        assert _normalize_depth_unit("S") == "S"
    assert _normalize_depth_unit("") == "ft"


@pytest.mark.parametrize(
    ("label", "expected"),
    [("deg C (from log)", "C"), ("DEGF", "F"), ("°F", "F"), ("kelvin", "K"), ("", None), ("C/F", None)],
)
def test_temperature_unit_classification(label, expected):
    assert temperature_unit(label) == expected


def test_depth_to_meters_accepts_aliases():
    from petrophys._internal.analysis_helpers import depth_to_meters

    np.testing.assert_allclose(depth_to_meters(np.array([10.0]), "Feet"), [3.048])
    np.testing.assert_allclose(depth_to_meters(np.array([10.0]), "M"), [10.0])


def test_tabular_readers_require_a_depth_unit(tmp_path):
    from petrophys.io import read_csv
    from petrophys.io.ascii_reader import read_ascii

    path = tmp_path / "logs.csv"
    path.write_text("DEPTH,GR\n1000,50\n1001,55\n")
    with pytest.raises(ValueError, match="does not record its depth unit"):
        read_csv(path)
    assert read_csv(path, depth_unit="M").depth_unit == "m"

    frame = pd.DataFrame({"DEPTH": [1.0, 2.0], "GR": [50.0, 60.0]})
    with pytest.raises(ValueError, match="needs depth_unit"):
        Well.from_dataframe(frame)

    ascii_path = tmp_path / "logs.txt"
    ascii_path.write_text("DEPTH GR\nft API\n1000 50\n1001 55\n")
    assert read_ascii(ascii_path, units_line=1).depth_unit == "ft"


def test_las_reader_prefers_utf8_and_reports_unreadable_files(tmp_path):
    import petrophys as pp

    assert pp.read(pp.sample("30_6-5")).header.field == "OSEBERG ØST"
    empty = tmp_path / "empty.las"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty.las is not a readable LAS file"):
        pp.read(empty)
