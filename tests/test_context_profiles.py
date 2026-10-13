"""Computed reproductions of the published COSTA and Danish context packages."""

from __future__ import annotations

import json
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys.io import export_las, read_las
from petrophys.model.well import Well
from petrophys.reporting import context_profiles
from petrophys.reporting._danish_context import danish_group_layout, prepare_danish_las, read_danish_tops
from petrophys.reporting.context_cli import build_package_parser
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow
from petrophys.workflow.dynamic_workflow import _dataset_calculations

COSTA_DIR = Path(pp.__file__).resolve().parent / "datasets" / "costa"


def test_context_profiles_are_named_with_aliases():
    assert pp.list_context_profiles() == ("costa_v1", "danish_v1")
    assert pp.get_context_profile("costa") is pp.get_context_profile("costa-v1")
    assert pp.get_context_profile("danish").name == "danish_v1"
    costa = pp.describe_context_profile("costa")
    assert costa["well_count"] == 17 and costa["required_inputs"] == []
    danish = pp.describe_context_profile("danish")
    assert danish["required_inputs"] == ["tops_file", "lithology_mapping_file"]
    with pytest.raises(ValueError, match="Unknown context profile"):
        pp.get_context_profile("geolink")


def test_identify_context_profile_by_well_names(tmp_path):
    assert pp.identify_context_profile(COSTA_DIR / "HW-31.las") is pp.get_context_profile("costa")
    assert pp.identify_context_profile(COSTA_DIR) is pp.get_context_profile("costa")
    assert pp.identify_context_profile(tmp_path, well_names=["elin-1", "IDA-1"]).name == "danish_v1"
    assert pp.identify_context_profile(tmp_path, well_names=["30_6-5"]) is None
    assert pp.identify_context_profile(tmp_path, well_names=["elin-1", "HW-3"]) is None


def test_export_context_profile_checks_wells_and_danish_inputs(tmp_path):
    with pytest.raises(ValueError, match="Wells not in context profile costa_v1"):
        context_profiles.export_context_profile(
            None, tmp_path / "out", context_profile="costa", work_dir=tmp_path / "work", well_names=["elin-1"]
        )
    with pytest.raises(ValueError, match="requires tops_file, lithology_mapping_file"):
        context_profiles.export_context_profile(
            tmp_path, tmp_path / "out", context_profile="danish", work_dir=tmp_path / "work"
        )


def test_profile_is_only_available_for_category_packages(tmp_path):
    with pytest.raises(ValueError, match="requires layout='category_formation_lithology'"):
        pp.export_context(tmp_path, tmp_path / "out", layout="formation_lithology", context_profile="costa")


def test_context_cli_exposes_profiles_only_on_category_layout():
    category = build_package_parser("category_formation_lithology")
    assert category.parse_args(["--input-dir", "las", "--profile", "danish"]).context_profile == "danish"
    assert "--profile" not in build_package_parser("formation_lithology").format_help()


# -- COSTA -----------------------------------------------------------------------

#: Values of the published COSTA package, saturation context of HW-3's first
#: limestone group: arithmetic mean, minimum, maximum.
_HW3_PACKAGE_SATURATION = {
    "SW_overall": (0.2033, 0.0453, 1),
    "SO_overall": (0.7967, 0, 0.9547),
    "RW_overall": (0.0206, 0.0205, 0.0207),
    "VCL_overall": (0.0068, 0, 0.0915),
    "PHIE_CC_overall": (0.2035, 0.01, 0.31),
    "BVW_overall": (0.0194, 0.01, 0.0354),
}


def test_costa_profile_computes_the_published_hw3_context(tmp_path):
    output = tmp_path / "context"
    results = context_profiles.export_context_profile(
        None, output, context_profile="costa", work_dir=tmp_path / "work", well_names=["HW-3"]
    )
    assert len(results) == 18 and all(item.success for item in results)

    path = output / "saturation" / "HW-3" / "KHARAIB-2 FM. TOP" / "KHARAIB-2 FM. TOP_LIMESTONE_GROUP_1_saturation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis = payload["analysis_results"]
    for key, (mean, minimum, maximum) in _HW3_PACKAGE_SATURATION.items():
        stats = analysis["statistics"][key]
        assert (stats["arithmetic_mean"], stats["minimum"], stats["maximum"]) == (mean, minimum, maximum), key
    assert analysis["lithology_group_info"] == {
        "start_depth": 7805.0, "end_depth": 7867.5, "thickness": 62.5, "data_points": 126, "median_dz": 0.5,
    }
    assert payload["formation_info"] == {
        "name": "KHARAIB-2 FM. TOP", "min_depth": 7805.0, "max_depth": 8020.0, "thickness": 215.0,
    }
    notes = analysis["data_completeness"]["derived_log_computation_notes"]
    assert "VCL: linear GR; (VCL=\\mathrm{clip}((GR-20)/80,0,1))." in notes
    assert (
        "RW: petrophys RW step applies Arps correction from RW=0.0682 ohm.m at 68 F (from costa_params.xlsx); "
        "RW(T)=RW68*(68+6.77)/(TEMP+6.77). RW(T)=0.0205 to 0.0207 ohm.m." in notes
    )
    assert "SXO_overall" not in analysis["statistics"]

    las = lasio.read(tmp_path / "work" / "processed_las" / "HW-3.las")
    source = lasio.read(COSTA_DIR / "HW-3.las")
    sw, so = las["SW"], las["SO"]
    given = np.isfinite(source["SW_INPUT"]) & (source["SW_INPUT"] >= 0) & (source["SW_INPUT"] <= 1)
    np.testing.assert_allclose(sw[given], source["SW_INPUT"][given], atol=5e-6)
    np.testing.assert_allclose(so, 1.0 - sw, atol=1e-5)
    assert set(np.unique(las["LITHOLOGY_GEOLINK"][np.isfinite(las["LITHOLOGY_GEOLINK"])])) <= {11.0, 15.0}
    assert "SXO" not in las.curves.keys() and "MHI" not in las.curves.keys()
    trace = json.loads((tmp_path / "work" / "processed_las" / "workflow_json" / "HW-3.json").read_text(encoding="utf-8"))
    assert [step["step"] for step in trace["costa_precomputed_steps"]][:2] == ["costa_core_curve_aliases", "costa_vcl"]


def test_costa_parameter_tables_are_bundled():
    from petrophys.reporting._costa_context import load_costa_parameters, load_costa_reference_volumes

    parameters = load_costa_parameters()
    assert len(parameters) == 17
    assert parameters["HW-3"].archie == (1.0, 2.03, 2.0)
    assert parameters["HW-7"].rw_68f is None and parameters["HW-7"].rw_68f_used == 0.1
    volumes = load_costa_reference_volumes()
    assert set(volumes) == set(parameters) - {"HW-8", "HW-9"}


# -- Danish ----------------------------------------------------------------------


def _write_raw_danish_las(path: Path, depth_unit: str = "M") -> None:
    las = lasio.LASFile()
    depth = np.round(np.arange(1000.0, 1030.0, 0.15), 2)
    n = depth.size
    rng = np.random.default_rng(3)
    las.append_curve("DEPTH", depth, unit=depth_unit, descr="Measured Depth")
    las.append_curve("GR", 40 + 20 * rng.random(n), unit="GAPI", descr="Gamma Ray")
    las.append_curve("RHOB", 2.3 + 0.2 * rng.random(n), unit="G/C3", descr="v1")
    las.append_curve("NPHI", 20 + 10 * rng.random(n), unit="V/V", descr="v1")
    las.append_curve("RSHAL", 1 + rng.random(n), unit="OHMM", descr="Shallow Resistivity")
    las.append_curve("RDEP", 2 + rng.random(n), unit="OHMM", descr="Deep Resistivity")
    las.append_curve("DTC", 90 + 10 * rng.random(n), unit="US/F", descr="Compressional Slowness")
    las.params.append(lasio.HeaderItem("RW", unit="OHMM", value=0.065, descr="Formation water resistivity"))
    las.params.append(lasio.HeaderItem("BHT", unit="DEGF", value=180, descr="Formation temperature"))
    las.well.WELL.value = path.stem.upper()
    las.write(str(path), version=2.0)


def _write_tops(path: Path, well: str) -> None:
    pd.DataFrame(
        [
            {"Well": well.upper(), "Formation": "EKOFISK", "MD_Top": 3285.0, "MD_Bottom": 3330.0,
             "DepthUnit": "feet", "MajorityLithology": "Chalk"},
            {"Well": well.upper(), "Formation": "TOR", "MD_Top": 3330.0, "MD_Bottom": 3370.0,
             "DepthUnit": "feet", "MajorityLithology": "Marl"},
        ]
    ).to_excel(path, index=False)


def test_danish_preparation_writes_the_prepared_las(tmp_path):
    raw = tmp_path / "test-1.las"
    _write_raw_danish_las(raw)
    _write_tops(tmp_path / "tops.xlsx", "test-1")
    tops = read_danish_tops(tmp_path / "tops.xlsx")["test-1"]
    prepared = prepare_danish_las(raw, tops, tmp_path / "prepared" / "test-1.las")

    las = lasio.read(prepared)
    assert [curve.mnemonic for curve in las.curves] == [
        "DEPT", "GR", "RHOB", "NPHI", "RDEP", "DTC", "LITHOLOGY_GEOLINK", "RSHA",
    ]
    units = {curve.mnemonic: curve.unit for curve in las.curves}
    assert units == {
        "DEPT": "FT", "GR": "API", "RHOB": "g/cm3", "NPHI": "PU", "RDEP": "ohm.m", "DTC": "us/ft",
        "LITHOLOGY_GEOLINK": "", "RSHA": "ohm.m",
    }
    np.testing.assert_allclose(las.index[0], round(1000.0 / 0.3048, 5))
    codes = las["LITHOLOGY_GEOLINK"]
    depth = las.index
    assert np.all(codes[(depth >= 3285.0) & (depth < 3330.0)] == 9)
    assert np.all(codes[(depth >= 3330.0) & (depth < 3370.0)] == 13)
    assert np.all(np.isnan(codes[depth < 3285.0]))


def test_danish_group_layout_follows_the_package_rules(tmp_path):
    raw = tmp_path / "test-1.las"
    _write_raw_danish_las(raw)
    _write_tops(tmp_path / "tops.xlsx", "test-1")
    layout = danish_group_layout(raw, read_danish_tops(tmp_path / "tops.xlsx")["test-1"])
    depth_ft = np.round(np.arange(1000.0, 1030.0, 0.15), 2) * 3.28084
    first = depth_ft[(depth_ft >= 3285.0) & (depth_ft <= 3330.0)]
    assert layout[0]["formation_name"] == "EKOFISK"
    assert layout[0]["group_name"] == "CHALK_GROUP_1" and layout[0]["lithology"] == "CHALK"
    assert layout[0]["formation_depth_info"] == {
        "min_depth": float(first[0]), "max_depth": float(first[-1]), "thickness": round(float(first[-1] - first[0]), 2),
    }
    assert (layout[0]["start_depth"], layout[0]["end_depth"]) == (round(float(first[0]), 2), round(float(first[-1]), 2))
    assert layout[1]["group_name"] == "MARL_GROUP_1"


def test_danish_profile_computes_a_context_from_raw_las(tmp_path):
    raw_dir = tmp_path / "las"
    raw_dir.mkdir()
    _write_raw_danish_las(raw_dir / "elin-1.las")
    _write_tops(tmp_path / "tops.xlsx", "elin-1")
    pd.DataFrame(
        [{"Attribute": "Chalk", "Lithology Attribute": "Chalk", "Abbreviation": 9},
         {"Attribute": "Marl", "Lithology Attribute": "Marlstone", "Abbreviation": 13}]
    ).to_excel(tmp_path / "codes.xlsx", index=False)

    output = tmp_path / "context"
    results = context_profiles.export_context_profile(
        raw_dir, output, context_profile="danish", work_dir=tmp_path / "work",
        tops_file=tmp_path / "tops.xlsx", lithology_mapping_file=tmp_path / "codes.xlsx",
    )
    assert results and all(item.success for item in results)
    stems = sorted({path.name.rsplit("_", 1)[0] for path in (output / "numerical").rglob("*.json")})
    assert stems == ["EKOFISK_CHALK_GROUP_1", "TOR_MARL_GROUP_1"]
    payload = json.loads(
        (output / "numerical" / "elin-1" / "EKOFISK" / "EKOFISK_CHALK_GROUP_1_numerical.json").read_text(encoding="utf-8")
    )
    assert payload["analysis_results"]["lithology"] == "CHALK"
    intervals = json.dumps(payload["analysis_results"]["analysis_summary"])
    assert " FT\"" in intervals and " ft\"" not in intervals
    assert (tmp_path / "work" / "processed_las" / "elin-1.las").exists()


def test_danish_calculations_take_an_rw_without_temperature_at_75f():
    well = Well("RW-TEST")
    depth = np.arange(2000.0, 2010.0, 0.5)
    well.depth = depth
    well.depth_unit = "ft"
    rng = np.random.default_rng(1)
    for name, values, unit in (
        ("GR", 30 + 10 * rng.random(depth.size), "API"),
        ("RHOB", 2.3 + 0.1 * rng.random(depth.size), "g/cm3"),
        ("NPHI", 0.2 + 0.05 * rng.random(depth.size), "v/v"),
        ("RDEP", 5 + rng.random(depth.size), "ohm.m"),
    ):
        well.add_curve(name, values, unit, name)
    config = DynamicWorkflowConfig(calculation_profile="benchmark", rw=0.065, bht_f=200.0)
    with _dataset_calculations("danish"):
        result = run_dynamic_workflow(well, config=config)
    step = next(step for step in result.steps_executed if step.step == "rw")
    assert step.parameters["reference_temp_f"] == 75.0
    assert "directly corrected with Arps" in step.decisions[0]
    temp = np.asarray(result.well.get("TEMP").data, dtype=float)
    np.testing.assert_allclose(result.well.get("RW").data, 0.065 * (75.0 + 6.77) / (temp + 6.77))


# -- LAS writing -------------------------------------------------------------------


def test_export_las_writes_a_source_depth_label_and_pads_by_bytes(tmp_path):
    well = Well("W")
    well.depth = np.array([100.0, 100.5])
    well.depth_unit = "ft"
    well.add_curve("GR", np.array([50.0, 60.0]), "API", "Gamma Ray")
    well.header.field = "SLEIPNER \u00d8ST"
    export_las(well, tmp_path / "a.las", depth_unit_label="FT")
    export_las(well, tmp_path / "b.las", depth_unit_label="FT", pad_header_by_bytes=True)
    a = (tmp_path / "a.las").read_text(encoding="utf-8")
    b = (tmp_path / "b.las").read_text(encoding="utf-8")
    strt_a = next(line for line in a.splitlines() if line.startswith("STRT"))
    strt_b = next(line for line in b.splitlines() if line.startswith("STRT"))
    assert strt_a.startswith("STRT.FT") and strt_b.startswith("STRT.FT")
    assert next(line for line in a.splitlines() if line.startswith("DEPT")).split(":")[0].split(".")[1].strip() == "FT"
    assert len(strt_b) == len(strt_a) + 1
    assert "SLEIPNER \u00d8ST" in b
    assert read_las(tmp_path / "b.las").header.field == "SLEIPNER \u00d8ST"
    with pytest.raises(ValueError, match="does not name the well's depth unit"):
        export_las(well, tmp_path / "c.las", depth_unit_label="M")
