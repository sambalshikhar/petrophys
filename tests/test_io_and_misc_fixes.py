"""Regression guards for IO, workflow, and base-container fixes.

Each test pins the corrected behavior so the bug cannot silently return.
"""

import inspect
import typing
import warnings as warnings_module

import numpy as np
import pytest

lasio = pytest.importorskip("lasio")


def _make_simple_well(name="TEST-WELL", n=11):
    from petrophys.model.well import Well

    well = Well(name=name)
    well.depth = np.linspace(1000.0, 1010.0, n)
    well.add_curve("GR", np.linspace(20.0, 120.0, n), "API", "Gamma Ray")
    return well


# ---------------------------------------------------------------------------
# Fix 1: export_las must forward **kwargs (e.g. fmt) to lasio's write
# ---------------------------------------------------------------------------
def test_export_las_forwards_fmt_kwarg(tmp_path):
    from petrophys.io.export import export_las

    well = _make_simple_well()
    well["GR"].data[0] = 45.123456789

    out = tmp_path / "precision.las"
    export_las(well, out, fmt="%.8f")

    text = out.read_text()
    assert "45.12345679" in text  # eight decimals survive the write


# ---------------------------------------------------------------------------
# Fix 2: a text/string curve in a LAS must be skipped with a warning, not
# crash the whole read
# ---------------------------------------------------------------------------
def test_read_las_skips_text_curve_with_warning(tmp_path):
    from petrophys.io.las_reader import read_las

    las_text = (
        "~Version\n"
        "VERS. 2.0 : \n"
        "WRAP. NO : \n"
        "~Well\n"
        "STRT.M 1000.0 : \n"
        "STOP.M 1002.0 : \n"
        "STEP.M 1.0 : \n"
        "NULL. -999.25 : \n"
        "WELL. TEXTCURVE-1 : \n"
        "~Curve\n"
        "DEPT.M : Depth\n"
        "GR.API : Gamma Ray\n"
        "LITH. : Lithology label\n"
        "~ASCII\n"
        "1000.0 45.0 SAND\n"
        "1001.0 50.0 SHALE\n"
        "1002.0 55.0 SAND\n"
    )
    path = tmp_path / "text_curve.las"
    path.write_text(las_text)

    with pytest.warns(UserWarning, match="LITH"):
        well = read_las(path)

    assert "GR" in well.curves
    assert "LITH" not in well.curves
    np.testing.assert_allclose(well["GR"].data, [45.0, 50.0, 55.0])


# ---------------------------------------------------------------------------
# Fix 3: read_ascii with column_names (headerless file) must not drop the
# first data row
# ---------------------------------------------------------------------------
def test_read_ascii_headerless_keeps_all_rows(tmp_path):
    from petrophys.io.ascii_reader import read_ascii

    path = tmp_path / "headerless.txt"
    path.write_text(
        "1000.0,45.0\n"
        "1001.0,50.0\n"
        "1002.0,55.0\n"
    )

    well = read_ascii(path, column_names=["DEPTH", "GR"])

    assert well.n_samples == 3
    np.testing.assert_allclose(well.depth, [1000.0, 1001.0, 1002.0])
    np.testing.assert_allclose(well["GR"].data, [45.0, 50.0, 55.0])


# ---------------------------------------------------------------------------
# Fix 4: to_merged_well must not label unconverted depth with the requested
# unit. _convert_depth reports the *actual* unit for unknown source units, and
# to_merged_well now warns and skips frames whose unit could not be converted.
# ---------------------------------------------------------------------------
def test_convert_depth_reports_actual_unit_for_unknown_source():
    from petrophys.io.dlis_reader import _convert_depth

    depth = np.array([100.0, 200.0, 300.0])
    converted, unit = _convert_depth(depth, "furlong", "m")

    # Unknown unit: values pass through unchanged and the source unit is
    # reported back, NOT the requested target unit.
    np.testing.assert_allclose(converted, depth)
    assert unit == "furlong"
    assert unit != "m"


def test_to_merged_well_skips_frames_with_unconvertible_depth_unit():
    from petrophys.io import dlis_reader

    source = inspect.getsource(dlis_reader.DLISReader.to_merged_well)
    # The returned unit must be captured (not discarded) and mismatching
    # frames skipped with a warning.
    assert "depth, converted_unit = _convert_depth(" in source
    assert "if converted_unit != depth_unit:" in source
    assert "warnings.warn" in source


# ---------------------------------------------------------------------------
# Fix 5: BatchResult.well_name must use the actual well name
# ---------------------------------------------------------------------------
def test_batch_processor_uses_well_name():
    from petrophys.workflow.batch_processing import BatchProcessor

    processor = BatchProcessor(workflow_func=lambda well: {"ok": True})
    well = _make_simple_well(name="NAMED-1")

    results = processor.process_wells([well])

    assert len(results) == 1
    assert results[0].well_name == "NAMED-1"

    # Fallback still applies when the well has no name.
    unnamed = _make_simple_well(name="")
    results = processor.process_wells([unnamed])
    assert results[0].well_name == "well_0"


# ---------------------------------------------------------------------------
# Fix 6: QuickLookResult.well_name must use the actual well name
# ---------------------------------------------------------------------------
def test_quick_look_uses_well_name():
    from petrophys.workflow.quick_look import QuickLook

    well = _make_simple_well(name="NAMED-2")
    result = QuickLook().analyze(well)
    assert result.well_name == "NAMED-2"

    unnamed = _make_simple_well(name="")
    result = QuickLook().analyze(unnamed)
    assert result.well_name == "Unknown"


# ---------------------------------------------------------------------------
# Fix 7: describe_model_pack's Mapping annotation must be importable/resolvable
# ---------------------------------------------------------------------------
def test_describe_model_pack_mapping_annotation_resolves():
    from petrophys.preprocessing.imputation.training import describe_model_pack

    # Would raise NameError("Mapping") before the typing import was added.
    hints = typing.get_type_hints(describe_model_pack)
    assert "pack" in hints

    df = describe_model_pack({"metadata": {"targets": {}}})
    assert df.empty


# ---------------------------------------------------------------------------
# Fix 8a: pyrite S/Fe weight ratio in the docstring must be ~1.15, not 0.57
# ---------------------------------------------------------------------------
def test_xrf_pyrite_ratio_docstring():
    from petrophys.interpretation.geochemistry.xrf import XRFData

    doc = XRFData.elemental_ratios.__doc__
    assert "1.15" in doc
    assert "0.57" not in doc


# ---------------------------------------------------------------------------
# Fix 8b: XRFData.get must be case-insensitive so chemical-symbol keys
# ("Si") are found by upper-cased queries ("SI") and vice versa
# ---------------------------------------------------------------------------
def test_xrf_get_is_case_insensitive():
    from petrophys.interpretation.geochemistry.xrf import XRFData

    depth = np.array([1.0, 2.0, 3.0])
    si = np.array([30.0, 31.0, 32.0])
    al = np.array([5.0, 5.5, 6.0])

    # Chemical-symbol keys queried with upper-case mnemonics
    xrf = XRFData(depth=depth, elements={"Si": si, "Al": al})
    np.testing.assert_allclose(xrf.get("SI"), si)
    np.testing.assert_allclose(xrf.get("AL"), al)
    np.testing.assert_allclose(xrf.get("si"), si)

    # Upper-case LAS mnemonic keys queried with chemical symbols
    xrf_upper = XRFData(depth=depth, elements={"SI": si, "AL": al})
    np.testing.assert_allclose(xrf_upper.get("Si"), si)
    assert xrf_upper.get("Fe") is None


# ---------------------------------------------------------------------------
# Fix 9: batch export must propagate LogViewer plot warnings into
# BatchVisualizationResult.warnings
# ---------------------------------------------------------------------------
def test_batch_export_propagates_plot_warnings(tmp_path, monkeypatch):
    from petrophys.visualization import batch_export

    class DummyPlot:
        fig = None
        warnings = ["missing NPHI curve"]

        def save(self, path, dpi=150):
            pass

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "dummy.las").write_text("placeholder")

    monkeypatch.setattr(batch_export, "read_las", lambda path: object())
    monkeypatch.setattr(
        batch_export, "standardize_curves", lambda well, **kw: well
    )
    monkeypatch.setattr(
        batch_export, "create_log_viewer", lambda well, **kw: DummyPlot()
    )

    results = batch_export.export_interpretation_plots(input_dir, output_dir)

    assert len(results) == 1
    assert results[0].success
    assert results[0].warnings == ["missing NPHI curve"]


# ---------------------------------------------------------------------------
# Fix 10: Curve.copy() must deep-copy metadata, not share it by reference
# ---------------------------------------------------------------------------
def test_curve_copy_deep_copies_metadata():
    well = _make_simple_well()
    original = well["GR"]
    duplicate = original.copy()

    assert duplicate.metadata is not original.metadata
    duplicate.metadata.custom["flag"] = "edited"
    assert "flag" not in original.metadata.custom


# ---------------------------------------------------------------------------
# Fix 11: Well.__setitem__ must reject a Curve of the wrong length
# ---------------------------------------------------------------------------
def test_well_setitem_rejects_wrong_length_curve():
    from petrophys.model.curve import Curve

    well = _make_simple_well(n=11)

    bad = Curve("PHIT", np.zeros(5))
    with pytest.raises(ValueError, match="5 samples.*11"):
        well["PHIT"] = bad
    assert "PHIT" not in well.curves

    good = Curve("PHIT", np.zeros(11))
    well["PHIT"] = good
    assert "PHIT" in well.curves


# ---------------------------------------------------------------------------
# Fix 12: DepthIndex.resample must not overshoot the bottom depth
# ---------------------------------------------------------------------------
def test_depth_index_resample_does_not_overshoot_bottom():
    from petrophys.model.log_data import DepthIndex

    index = DepthIndex(np.array([0.0, 5.0, 10.0]), unit="m")

    resampled = index.resample(3.0)
    np.testing.assert_allclose(resampled.depth, [0.0, 3.0, 6.0, 9.0])
    assert resampled.depth.max() <= 10.0

    # Exact-fit step keeps the bottom sample.
    exact = index.resample(5.0)
    np.testing.assert_allclose(exact.depth, [0.0, 5.0, 10.0])
