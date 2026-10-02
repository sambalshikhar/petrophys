"""Native MDT traces, independent report checks and failure-mode regressions."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import pytest

from petrophys import sample
from petrophys.interpretation.saturation.height import (
    MDTPickingConfig, formation_pressures_from_mdt, pick_mdt_pressure,
    read_mdt_pretest,
)


def signal(level=300., repeat=None):
    """Controlled drawdown/recovery; used only for algorithm edge cases."""
    t = np.arange(501, dtype=float)
    p = np.full(t.shape, level+50.)
    intervals = [(100, 425, level)] if repeat is None else [(100, 250, level), (250, 425, repeat)]
    for start, stop, target in intervals:
        mask = (t >= start) & (t < stop)
        p[mask] = target-30*np.exp(-(t[mask]-start)/10.)
    return t, p


def test_all_native_traces_against_independent_report():
    folder = Path(sample("c0020a_mdt_077")).parent
    manifest = json.loads((folder / "c0020a_mdt_manifest.json").read_text())
    refs = pd.read_csv(folder / manifest["reference_file"]).set_index("FILE_ID")
    assert hashlib.sha256((folder / manifest["reference_file"]).read_bytes()).hexdigest() == manifest["reference_sha256"]
    assert hashlib.sha256((folder / manifest["report_file"]).read_bytes()).hexdigest() == manifest["report_sha256"]
    accepted, failed, valid_review = [], [], []
    assert len(manifest["files"]) == 35
    for item in manifest["files"]:
        path = folder / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        trace = read_mdt_pretest(path)
        # Only raw curves and metadata go to the picker, never report picks/QC.
        pick = trace.pick_pressure()
        ref = refs.loc[item["file_id"]]
        assert trace.pressure_unit == "psi" and trace.depth_reference == "MD"
        if pick.status == "accepted":
            assert ref.REPORT_STATUS == "Valid Test"
            assert abs(pick.pressure-ref.REPORT_PRESSURE_PSI) < 0.2
            assert abs(pick.station_depth-ref.PROBE_MD_M) < .1
            assert pick.pressure < min(pick.mud_before, pick.mud_after)-100
            accepted.append(item["file_id"])
        else:
            assert pick.pressure is None and pick.flags
            (valid_review if ref.REPORT_STATUS == "Valid Test" else failed).append(item["file_id"])
    assert len(accepted) == 22
    assert failed == [86, 105, 112, 118]
    assert valid_review == [70, 85, 87, 95, 96, 97, 113, 114, 115]


def test_retraction_does_not_select_mud_and_repeat_agreement():
    t, p = signal(repeat=300.)
    pick = pick_mdt_pressure(t, p, depth=np.full_like(t, 2000.))
    assert pick.status == "accepted" and len(pick.buildups) == 2
    assert pick.pressure == pytest.approx(300., abs=.01)
    assert pick.pressure != p[-1] == 350.
    assert pick.station_depth == 2000.
    t, p = signal(repeat=301.)
    pick = pick_mdt_pressure(t, p)
    assert pick.pressure is None and "repeat_pretests_disagree" in pick.flags


@pytest.mark.parametrize("case", ["drift", "no_response", "flat_drawdown", "movement", "missing", "gap", "truncated", "final_short", "multiple"])
def test_unsafe_traces_never_emit_pressure(case):
    t, p = signal()
    z = np.full_like(t, 2000.)
    if case == "drift":
        p[(t >= 100) & (t < 425)] = 270.+.06*(t[(t >= 100) & (t < 425)]-100)
    elif case == "no_response":
        p[:] = 350.
    elif case == "flat_drawdown":
        p[(t >= 100) & (t < 425)] = 300.
    elif case == "movement":
        z[250:400] += np.linspace(0, 2, 150)
    elif case == "missing":
        p[250] = np.nan
    elif case == "gap":
        keep = (t < 250) | (t > 275)
        t, p, z = t[keep], p[keep], z[keep]
    elif case == "truncated":
        t, p, z = t[:400], p[:400], z[:400]
    elif case == "final_short":
        mask = (t >= 395) & (t < 425)
        p[mask] = 300-30*np.exp(-(t[mask]-395)/3)
    elif case == "multiple":
        p[240:280] = 350.
        p[280:425] = 300-30*np.exp(-(t[280:425]-280)/10)
    result = pick_mdt_pressure(t, p, depth=z)
    assert result.status == "review" and result.pressure is None and result.flags


@pytest.mark.parametrize("unit,factor", [("bar", 1.), ("psi", .06894757293168), ("kPa", .01), ("MPa", 10.)])
def test_pressure_time_depth_units_and_time_origin(unit, factor):
    t, p = signal()
    result = pick_mdt_pressure((t+12345)*1000, p/factor, depth=np.full_like(t, 2000/.3048),
                               time_unit="ms", pressure_unit=unit, depth_unit="ft")
    assert result.pressure*factor == pytest.approx(300, abs=.01)
    assert result.station_depth*.3048 == pytest.approx(2000)


def test_reader_las_csv_and_custom_columns(tmp_path):
    path = sample("c0020a_mdt_077")
    native = read_mdt_pretest(path)
    las = lasio.read(path)
    frame = pd.DataFrame({"seconds": native.time, "quartz": native.pressure, "probe": native.depth})
    csv = tmp_path / "pretest.csv"
    frame.to_csv(csv, sep=";", index=False)
    custom = read_mdt_pretest(csv, time_column="seconds", pressure_column="quartz",
                              depth_column="probe", time_unit="s", pressure_unit="psi", depth_unit="m",
                              csv_kwargs={"sep": ";"})
    assert custom.pick_pressure().pressure == native.pick_pressure().pressure
    assert read_mdt_pretest(las).pick_pressure().pressure == native.pick_pressure().pressure
    frame = frame.rename(columns={"seconds": "TIME_S", "quartz": "PRES_PSI", "probe": "DEPTH_M"})
    assert read_mdt_pretest(frame).pick_pressure().pressure == native.pick_pressure().pressure
    frame["PRES_BAR"] = frame.PRES_PSI*.06894757293168
    with pytest.raises(ValueError, match="pressure_column"):
        read_mdt_pretest(frame)
    with pytest.raises(ValueError, match="Specify unit"):
        read_mdt_pretest(pd.DataFrame({"TIME": native.time, "PRES": native.pressure}))


def test_invalid_configuration_time_and_array_metadata():
    with pytest.raises(ValueError):
        MDTPickingConfig(tail_seconds=0)
    t, p = signal()
    with pytest.raises(ValueError, match="increase strictly"):
        pick_mdt_pressure(t[::-1], p[::-1])
    with pytest.raises(ValueError, match="matching"):
        pick_mdt_pressure(t[:-1], p)
    with pytest.raises(ValueError, match="must match"):
        pick_mdt_pressure(t, p, depth=[1, 2])
    with pytest.raises(ValueError, match="Use time"):
        pick_mdt_pressure(t, p, pressure_unit="unknown")


def test_station_bridge_excludes_review_and_requires_real_tvd():
    accepted = read_mdt_pretest(sample("c0020a_mdt_077")).pick_pressure()
    review = read_mdt_pretest(sample("c0020a_mdt_086")).pick_pressure()
    with pytest.raises(ValueError, match="Provide TVD"):
        formation_pressures_from_mdt([accepted, review])
    data = formation_pressures_from_mdt([accepted, review], tvd=[2600., 3000.],
                                        depth_unit="m", depth_reference="TVDSS")
    assert data.points.use_fit.tolist() == [True, False]
    assert data.points.fluid.tolist() == ["unknown", "unknown"]
    assert data.points.pressure.iloc[0] == pytest.approx(accepted.pressure*.06894757293168)
    assert np.isnan(data.points.pressure.iloc[1])
    assert data.points.QC_FLAGS.iloc[1] and data.depth_reference == "TVDSS"
    tvd_pick = replace(accepted, depth_reference="TVDSS")
    assert formation_pressures_from_mdt([tvd_pick]).points.depth.iloc[0] == accepted.station_depth
    with pytest.raises(ValueError, match="Changing a depth"):
        formation_pressures_from_mdt([tvd_pick], depth_reference="TVD_RKB")


def test_raw_to_unlabeled_contacts_pipeline():
    # Controlled reservoir for API integration, not evidence of field accuracy.
    depths = np.r_[np.linspace(1300, 1475, 8), np.linspace(1525, 1700, 8)]
    pressures = 225+np.where(depths < 1500, .065, .11)*(depths-1500)
    picks = [pick_mdt_pressure(*signal(p)) for p in pressures]
    data = formation_pressures_from_mdt(picks, tvd=depths, depth_unit="m", depth_reference="TVDSS")
    result = data.infer_contacts()
    assert result.status == "interpreted"
    assert result.fwl_depth == pytest.approx(1500, abs=.1)


def test_library_qc_plot():
    import matplotlib.pyplot as plt
    from petrophys.visualization import create_mdt_pretest_plot
    for number in (77, 86):
        trace = read_mdt_pretest(sample(f"c0020a_mdt_{number:03}"))
        figure = create_mdt_pretest_plot(trace)
        assert len(figure.axes) == 1
        assert len(figure.axes[0].patches) >= 1
        plt.close(figure)
