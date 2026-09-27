import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

from petrophys import Well, sample
from petrophys.io import read_lithology_intervals
from petrophys.model.formation_tops import FormationTops
from petrophys.visualization import create_log_viewer


def _well():
    well = Well("TEMPLATE")
    well.depth = np.array([1000., 1050., 1100., 1150., 1200.])
    well.depth_unit = "m"
    well.add_curve("GR", [60, 30, 20, 65, 70], "API")
    return well


def test_packaged_templates_load_in_the_documented_apis():
    intervals = FormationTops.from_csv(sample("formation_intervals_template"))
    assert len(intervals) == 3
    lithology = read_lithology_intervals(sample("lithology_intervals_template"))
    assert lithology.lithology.tolist() == ["Shale", "Sandstone", "Limestone", "Shale"]
    for name in ("formation_tops_template", "formation_intervals_template"):
        viewer = create_log_viewer(
            _well(), formation_tops=sample(name), lithology_intervals=lithology,
        )
        try:
            assert len(viewer.formation_tops) == 3
            assert [top.name for top in viewer.formation_tops] == ["Formation_A", "Formation_B", "Formation_C"]
        finally:
            viewer.close()


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_lithology_reader_roundtrips_csv_excel_and_preserves_labels(tmp_path, suffix):
    table = pd.DataFrame({"top": [1100, 1000], "bottom": [1200, 1050], "lithology": ["Marl", "Custom rock"]})
    path = tmp_path / ("lithology" + suffix)
    if suffix == ".csv":
        table.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        table.to_excel(path, index=False)
    actual = read_lithology_intervals(path)
    assert actual.lithology.tolist() == ["Custom rock", "Marl"]
    assert actual.top.tolist() == [1000, 1100]
    pd.testing.assert_frame_equal(actual, read_lithology_intervals(table))
    assert table.top.tolist() == [1100, 1000]


@pytest.mark.parametrize("top,bottom,label,error", [
    (np.nan, 1050, "Shale", "finite"),
    (1000, np.inf, "Shale", "finite"),
    (1000, 1000, "Shale", "less than"),
    (1050, 1000, "Shale", "less than"),
    (1000, 1050, " ", "nonempty"),
    (1000, 1050, None, "nonempty"),
    (1000, 1050, 1, "nonempty"),
])
def test_invalid_lithology_rows_fail_with_row_number(top, bottom, label, error):
    table = pd.DataFrame({"top": [top], "bottom": [bottom], "lithology": [label]})
    with pytest.raises(ValueError, match=f"row 2:.*{error}"):
        read_lithology_intervals(table)


def test_wrong_schema_empty_and_overlapping_intervals_are_rejected():
    aliases = read_lithology_intervals(pd.DataFrame({"FROM": [1000], "TO": [1100], "LITH": ["Shale"]}))
    assert aliases.lithology.tolist() == ["Shale"]
    with pytest.raises(ValueError, match="template"):
        read_lithology_intervals(pd.DataFrame({"UNKNOWN": [1000], "TO": [1100], "LITH": ["Shale"]}))
    with pytest.raises(ValueError, match="no data"):
        read_lithology_intervals(pd.DataFrame(columns=["top", "bottom", "lithology"]))
    with pytest.raises(ValueError, match="overlapping"):
        read_lithology_intervals(pd.DataFrame({"top": [1000, 1025], "bottom": [1050, 1100], "lithology": ["Shale", "Sandstone"]}))
    with pytest.raises(ValueError, match="Unsupported lithology columns"):
        read_lithology_intervals(pd.DataFrame({"top": [1000], "bottom": [1050], "lithology": ["Shale"], "unmapped": ["A"]}))


def test_external_lithology_overrides_embedded_without_mutation_and_preserves_gaps():
    well = _well()
    well.add_curve("LITHOLOGY_GEOLINK", np.ones(5), "")
    curves_before = list(well.curves)
    table = pd.DataFrame({"top": [1007, 1100], "bottom": [1023, 1150], "lithology": ["Custom rock", "Marl"]})
    viewer = create_log_viewer(well, lithology_intervals=table)
    try:
        lithology_ax = viewer.axes[0]
        labels = [text.get_text() for text in lithology_ax.get_legend().get_texts()]
        assert "Custom rock" in labels
        assert "Sandstone" not in labels
        bounds = sorted(
            (np.min(path.vertices[:, 1]), np.max(path.vertices[:, 1]))
            for collection in lithology_ax.collections for path in collection.get_paths()
            if isinstance(collection, PolyCollection)
        )
        assert bounds == [(1007, 1023), (1100, 1150)]
        assert list(well.curves) == curves_before
        np.testing.assert_allclose(well.get("LITHOLOGY_GEOLINK").data, np.ones(5))
    finally:
        viewer.close()


def test_lithology_codebook_template_works_for_embedded_codes():
    well = _well()
    well.add_curve("LITHOLOGY_GEOLINK", [1, 2, 3, 2, 1], "")
    viewer = create_log_viewer(well, lithology_mapping_df=pd.read_csv(sample("lithology_codes_template")))
    try:
        assert viewer.lithology_mapping[3] == "Limestone"
    finally:
        viewer.close()
