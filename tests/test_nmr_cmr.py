from textwrap import dedent

import numpy as np

from petrophys.io import read_las
from petrophys.nmr import NMRData
from petrophys.qaqc.bad_hole import detect_washout
from petrophys.qaqc.standardization.alias_map import (
    find_curve_type,
    get_standard_mnemonic,
)
from petrophys.qaqc.value_range_qc import clean_curve_values


def _write_cmr_las(tmp_path):
    path = tmp_path / "processed_cmr.las"
    path.write_text(
        dedent("""
            ~VERSION INFORMATION
            VERS. 2.0 : CWLS Log ASCII Standard
            WRAP. NO
            ~WELL INFORMATION
            STRT.M 100.0000 : START DEPTH
            STOP.M 99.5000 : STOP DEPTH
            STEP.M -0.2500 : STEP
            NULL. -999.25 : NULL VALUE
            WELL. CMR-1 : WELL
            ~CURVE INFORMATION
            DEPT.M : DEPTH
            BS.IN : Bit Size
            CMFF.V/V : CMR Free Fluid
            CMR_TEMP.DEGC : CMR Temperature
            CMRP_3MS.V/V : CMR 3ms Porosity
            HD1_PPC1.IN : Pad Caliper 1
            HD2_PPC1.IN : Pad Caliper 2
            KSDR.MD : Permeability from CMR - SDR Model
            KTIM.MD : Permeability from CMR - Timur Model
            T2CUTOFF.MS : T2 Cutoff
            TCMR.V/V : Total CMR Porosity
            ~A
            100.0000 8.5000 0.0200 90.0 0.0100 8.4000 8.6000 0.1000 0.2000 33.0 0.0800
             99.7500 8.5000 0.0300 91.0 0.0150 9.4000 8.7000 0.2000 0.3000 33.0 0.1000
             99.5000 8.5000 0.0400 92.0 0.0200 8.6000 8.7000 0.3000 0.4000 33.0 0.1200
            """).strip(),
        encoding="utf-8",
    )
    return path


def test_processed_cmr_las_auto_maps_units_and_vendor_outputs(tmp_path):
    path = _write_cmr_las(tmp_path)

    nmr = NMRData.from_las(path)

    assert nmr.metadata["profile"] == "schlumberger_cmr"
    assert nmr.metadata["source_mnemonics"]["mphi"] == "TCMR"
    assert nmr.metadata["mbvi_derived"] is True
    np.testing.assert_allclose(nmr.mphi, [8.0, 10.0, 12.0])
    np.testing.assert_allclose(nmr.mffi, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(nmr.mbvi, [6.0, 7.0, 8.0])
    np.testing.assert_allclose(nmr.swirr_nmr(), [0.75, 0.70, 2.0 / 3.0])
    np.testing.assert_allclose(nmr.permeability_sdr(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(nmr.permeability_coates(), [0.2, 0.3, 0.4])
    np.testing.assert_allclose(nmr.extra_curves["CMRP_3MS"], [1.0, 1.5, 2.0])
    assert nmr.t2_cutoff == 33.0

    df = nmr.to_dataframe()
    assert {
        "MPHI",
        "MFFI",
        "MBVI",
        "T2CUTOFF",
        "K_SDR",
        "K_COATES",
        "SWIRR_NMR",
        "SWIRR",
    } <= set(df.columns)
    np.testing.assert_allclose(df["KSDR_LAS"], [0.1, 0.2, 0.3])


def test_explicit_cmr_loader_and_descending_depth_filter(tmp_path):
    nmr = NMRData.from_cmr_las(_write_cmr_las(tmp_path))

    filtered = nmr.filter_by_depth(100.0, 99.75)

    np.testing.assert_allclose(filtered.depth, [100.0, 99.75])
    np.testing.assert_allclose(filtered.mphi, [8.0, 10.0])
    np.testing.assert_allclose(filtered.stored_permeability["KSDR"], [0.1, 0.2])


def test_cmr_pad_calipers_drive_bad_hole_and_descending_thickness_is_positive(tmp_path):
    well = read_las(_write_cmr_las(tmp_path))

    result = detect_washout(well, cali_threshold=0.5)

    np.testing.assert_array_equal(result.washout_flag, [False, True, False])
    assert result.statistics["washout_count"] == 1
    assert result.statistics["washout_thickness"] == 0.25


def test_cmr_mnemonics_are_standardized_and_range_qced():
    assert get_standard_mnemonic("TCMR") == "NMR_PHIT"
    assert get_standard_mnemonic("CMFF") == "NMR_FFI"
    assert get_standard_mnemonic("SWIRR_NMR") == "NMR_SWIRR"
    assert get_standard_mnemonic("KSDR") == "NMR_KSDR"
    assert find_curve_type("KTIM") == "NMR Timur/Coates Permeability"

    cleaned, report = clean_curve_values("KSDR", np.array([0.0, 10.0, 30000.0]))

    np.testing.assert_allclose(cleaned[:2], [0.0, 10.0])
    assert np.isnan(cleaned[2])
    assert report["canonical_curve"] == "NMR_KSDR"
    assert report["changed"] == 1
