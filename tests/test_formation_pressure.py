"""Field reproduction and invalid-input checks for pressure-derived FWL."""
import numpy as np
import pandas as pd
import pytest
import lasio

from petrophys import sample
from petrophys.interpretation.saturation.height import fit_pressure_fwl, read_formation_pressure


def field_points():
    data = read_formation_pressure(sample("35_11_13_mdt"))
    return data.points.loc[data.points.use_fit]


def test_report_reproduction_from_las():
    las = lasio.read(sample("35_11_13_mdt"))
    assert len(las.index) == 29
    assert las.well.STEP.value == 0
    assert np.isnan(las["PRES"][-1])
    points = field_points()
    fit = fit_pressure_fwl(points.depth, points.pressure, points.fluid)
    assert (fit.hydrocarbon.n, fit.water.n) == (10, 9)
    # Tolerance reflects published interpretation vs unweighted OLS, not
    # a target imposed on the fit or an uncertainty bound.
    assert abs(fit.fwl_depth - 3112.6) < 2.0
    assert fit.hydrocarbon.gradient == pytest.approx(0.0642, abs=0.002)
    assert fit.water.gradient == pytest.approx(0.0951, abs=0.002)
    assert fit.water.pressure_at(fit.fwl_depth) == pytest.approx(
        fit.hydrocarbon.pressure_at(fit.fwl_depth))


def test_units_and_datum_shift_preserve_intersection():
    d = field_points()
    baseline = fit_pressure_fwl(d.depth, d.pressure, d.fluid)
    converted = fit_pressure_fwl(d.depth / 0.3048, d.pressure / 0.06894757293168,
                                 d.fluid, depth_unit="ft", pressure_unit="psi")
    assert converted.fwl_depth * 0.3048 == pytest.approx(baseline.fwl_depth)
    assert converted.fwl_pressure * 0.06894757293168 == pytest.approx(baseline.fwl_pressure)
    shifted = fit_pressure_fwl(d.depth + 25, d.pressure + 1, d.fluid,
                               depth_reference="TVD_RKB")
    assert shifted.fwl_depth == pytest.approx(baseline.fwl_depth + 25)
    assert shifted.fwl_pressure == pytest.approx(baseline.fwl_pressure + 1)


@pytest.mark.parametrize("pressure", [
    [10, 11, 12, 13],  # parallel
    [10, 12, 12, 13],  # hydrocarbon steeper
    [11, 10, 12, 14],  # negative hydrocarbon gradient
    [np.nan, 11, 12, 14],
])
def test_invalid_gradients_or_missing_pressure_rejected(pressure):
    with pytest.raises(ValueError):
        fit_pressure_fwl([100, 110, 120, 130], pressure, ['oil','oil','water','water'])


@pytest.mark.parametrize("kwargs", [{"depth_reference":"MD"}, {"depth_unit":"cm"},
                                    {"pressure_unit":"unknown"}])
def test_invalid_units_or_md_rejected(kwargs):
    d = field_points()
    with pytest.raises(ValueError):
        fit_pressure_fwl(d.depth, d.pressure, d.fluid, **kwargs)


def test_unknown_fluid_and_insufficient_depth_span_rejected():
    with pytest.raises(ValueError):
        fit_pressure_fwl([1,2,3,4], [10,11,12,14], ['gas','oil','water','water'])
    with pytest.raises(ValueError):
        fit_pressure_fwl([1,1,3,4], [10,11,12,14], ['oil','oil','water','water'])


@pytest.mark.parametrize("stem,expected,counts", [
    ("35_11_13_mdt", 3113.7262275822072, (10, 9)),
    ("35_11_10_fensfjord_mdt", 2287.3352250077814, (4, 6)),
])
def test_file_workflow_two_wells(stem, expected, counts):
    las = read_formation_pressure(sample(stem))
    fit = las.fit_fwl()
    assert fit.fwl_depth == pytest.approx(expected)
    assert (fit.hydrocarbon.n, fit.water.n) == counts
    assert fit.water.pressure_at(fit.fwl_depth) == pytest.approx(fit.fwl_pressure)
    assert fit.hydrocarbon.pressure_at(fit.fwl_depth) == pytest.approx(fit.fwl_pressure)


def test_custom_table_mapping_qc_and_delimited_csv(tmp_path):
    source = read_formation_pressure(sample("35_11_13_mdt")).points
    original = pd.DataFrame({
        "TVDSS_M": source.depth,
        "PRES_BAR": source.pressure,
        "FLUID": source.fluid,
        "USE_FIT": source.use_fit,
    })
    frame = original.rename(columns={"TVDSS_M": "Z", "PRES_BAR": "P", "FLUID": "CODE",
                                     "USE_FIT": "KEEP"})
    frame.CODE = frame.CODE.map({"oil": 8, "water": 9, "gas": 7})
    frame.KEEP = frame.KEEP.map({True: "true", False: "false"})
    kwargs = dict(depth_column="Z", pressure_column="P", fluid_column="CODE", use_column="KEEP",
                  depth_unit="m", pressure_unit="bar", depth_reference="TVDSS",
                  fluid_map={7: "gas", 8: "oil", 9: "water"})
    path = tmp_path / "stations.csv"
    frame.to_csv(path, sep=";", index=False)
    for source in [frame, path]:
        data = read_formation_pressure(source, csv_kwargs={"sep": ";"}, **kwargs)
        assert data.fit_mask().sum() == 19
        assert data.fit_fwl().fwl_depth == pytest.approx(3113.7262275822072)
    assert "depth" not in frame  # reader owns its copy
    frame.loc[0, "KEEP"] = "maybe"
    with pytest.raises(ValueError, match="USE_FIT"):
        read_formation_pressure(frame, **kwargs)


def test_las_requires_datum_and_explicit_fluid_code_mapping():
    las = lasio.read(sample("35_11_13_mdt"))
    del las.params["DREF"]
    with pytest.raises(ValueError, match="depth_reference"):
        read_formation_pressure(las)
    del las.params["FLDMAP"]
    with pytest.raises(ValueError, match="fluid_map"):
        read_formation_pressure(las, depth_reference="TVDSS")
    result = read_formation_pressure(las, depth_reference="TVDSS",
                                    fluid_map={1: "gas", 2: "oil", 3: "water"}).fit_fwl()
    assert result.fwl_depth == pytest.approx(3113.7262275822072)


def test_reader_rejects_ambiguous_columns_md_and_mixed_wells():
    source = read_formation_pressure(sample("35_11_13_mdt")).points
    frame = pd.DataFrame({"TVDSS_M": source.depth, "PRES_BAR": source.pressure})
    with pytest.raises(ValueError, match="MD"):
        read_formation_pressure(frame, depth_column="MD_RKB_M", depth_unit="m",
                                depth_reference="TVD_RKB")
    frame["DEPT"] = frame.TVDSS_M
    with pytest.raises(ValueError, match="depth_column"):
        read_formation_pressure(frame)
    frame["WELL"] = "35/11-13"
    frame.loc[0, "WELL"] = "another well"
    with pytest.raises(ValueError, match="one well"):
        read_formation_pressure(frame, depth_column="TVDSS_M")


def test_gas_branch_and_library_plot():
    from petrophys.visualization import create_formation_pressure_plot
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({"TVDSS_M": [100, 200, 300, 400, 500],
                          "PRES_BAR": [25, 30, 40, 50, np.nan],
                          "FLUID": ["GAS", "gas", "water", "water", "water"]})
    data = read_formation_pressure(frame)
    result = data.fit_fwl("gas")
    assert result.fwl_depth == pytest.approx(200)
    assert result.hydrocarbon_fluid == "gas"
    fig = create_formation_pressure_plot(data, hydrocarbon="gas")
    assert fig.axes[0].yaxis_inverted()
    assert "gas: selected" in fig.axes[0].get_legend_handles_labels()[1]
    plt.close(fig)
