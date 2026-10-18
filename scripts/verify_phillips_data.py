"""Verify copied Phillips data and the existing COSTA derivatives offline.

Run from a source checkout. --costa-source-dir optionally points at Craig's
original COSTA checkout for a fresh source-to-derivative numerical comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lasio
import numpy as np
import pandas as pd

from petrophys.datasets import DATA_DIR, list_samples, sample
from petrophys.interpretation.nmr import NMRData


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_blob_hash(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def numeric_digest(values):
    values = np.array(values, dtype="<f8", copy=True, order="C")
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def verify(costa_source_dir=None):
    manifest = json.loads(
        (DATA_DIR / "phillips/source_manifest.json").read_text(encoding="utf-8")
    )
    copied = []
    for item in manifest["files"]:
        raw = (DATA_DIR / item["path"]).read_bytes()
        require(len(raw) == item["bytes"], f"Size changed: {item['path']}")
        require(
            hashlib.sha256(raw).hexdigest() == item["sha256"],
            f"SHA-256 mismatch: {item['path']}",
        )
        require(
            git_blob_hash(raw) == item["upstream_git_blob_sha1"],
            f"Upstream Git blob mismatch: {item['path']}",
        )
        copied.append(item["path"])

    info = manifest["nmr_bin_metadata"]
    nmr = NMRData.from_csv(
        sample(info["sample"]),
        t2_centers=np.asarray(info["t2_centers_ms"], dtype=float),
        t2_cutoff=info["split_ms_matching_supplied_summaries"],
    )
    require(nmr.bins.shape == (51, 8), "Unexpected NMR-bin dimensions")
    require(np.all(np.diff(nmr.depth) > 0), "NMR depths must increase")
    require(np.all(np.isfinite(nmr.bins) & (nmr.bins >= 0)), "Invalid T2 bins")
    closure_errors = {}
    for name, computed, supplied in (
        ("MPHI", nmr.bins.sum(axis=1), nmr.mphi),
        # The supplied partition uses a sharp split below / at-or-above
        # the 32 ms bin centre, rather than the reader's partial-bin default.
        ("MBVI", nmr.bvi_from_bins(partial_bins=False), nmr.mbvi),
        ("MFFI", nmr.ffi_from_bins(partial_bins=False), nmr.mffi),
    ):
        error = float(np.max(np.abs(computed - supplied)))
        require(error <= info["rounding_tolerance_pu"], f"{name} closure: {error}")
        closure_errors[name] = error
    expected_lm = np.exp(
        (nmr.bins * np.log(nmr.t2_centers)).sum(axis=1) / nmr.bins.sum(axis=1)
    )
    np.testing.assert_allclose(nmr.t2_lm(), expected_lm, rtol=1e-12)
    for values in (
        nmr.permeability_sdr(prefer_stored=False),
        nmr.permeability_coates(prefer_stored=False),
    ):
        require(np.all(np.isfinite(values) & (values >= 0)), "Invalid NMR result")

    log = pd.read_csv(sample("phillips_cmr_log")).replace(-999.25, np.nan)
    core = pd.read_csv(sample("phillips_cmr_core")).replace(-999.25, np.nan)
    require(len(log) == 573 and len(core) == 56, "Unexpected CMR/core sizes")
    require(np.all(np.diff(log["DEPTH"]) > 0), "CMR depths must increase")
    require(
        core["DEPTH"].between(log["DEPTH"].min(), log["DEPTH"].max()).all(),
        "Core samples outside paired CMR depth range",
    )
    for table, porosity_column in ((log, "CMRP_3MS"), (core, "CMRP_3ms")):
        np.testing.assert_allclose(
            table[porosity_column], table["CMFF"] + table["BVI"], atol=2e-5, rtol=0
        )
        require(table[porosity_column].between(0, 1).all(), "Invalid CMR fractions")
    require(np.all(np.isfinite(core["Kair"]) & (core["Kair"] > 0)), "Invalid Kair")
    # The CSV porosity coordinate is CMRP_3ms, not total NMR porosity. This
    # exercises calibration on that supplied reference coordinate only.
    calibration_nmr = NMRData(
        depth=core["DEPTH"].to_numpy(),
        mphi=core["CMRP_3ms"].to_numpy() * 100,
        mffi=core["CMFF"].to_numpy() * 100,
        mbvi=core["BVI"].to_numpy() * 100,
        metadata={"porosity_basis": "CMRP_3ms", "cutoff": "not supplied"},
    )
    calibration = calibration_nmr.calibrate_coates(
        core["DEPTH"].to_numpy(), core["Kair"].to_numpy()
    )
    require(calibration.n_points == len(core), "Core calibration dropped samples")
    require(
        np.isfinite(calibration.r2_log) and np.isfinite(calibration.rmse_log),
        "Core calibration returned non-finite statistics",
    )

    gulf = pd.read_excel(sample("phillips_gulfcoast_nmr"), sheet_name="Sheet1")
    gulf = gulf.replace(-999.25, np.nan)
    require(gulf.shape == (2001, 15), "Unexpected Gulf Coast table shape")
    gulf_pu = gulf.copy()
    gulf_pu[["MPHI", "MBVI"]] *= 100
    gulf_nmr = NMRData.from_dataframe(
        gulf_pu, depth_col="DEPTH", mphi_col="MPHI", mbvi_col="MBVI",
        mffi_col=None, bin_cols=[],
    )
    require(not gulf_nmr.has_bins, "Gulf Coast workbook is scalar NMR")

    pc = pd.read_excel(sample("costa_core_pc"), sheet_name="Sheet1")
    require(len(pc) == 11055 and pc["Sample"].nunique() == 110, "COSTA Pc coverage")
    require(
        np.isfinite(pc[["Sample", "Porosity", "Perm", "Sw", "PcHg"]]).all().all(),
        "Non-finite COSTA Pc input",
    )
    require(pc["Sw"].between(0, 1).all(), "Sw must be fractional")
    require(pc["Porosity"].between(0, 100, inclusive="neither").all(), "Porosity %")
    require((pc["Perm"] > 0).all() and (pc["PcHg"] > 0).all(), "Nonpositive Pc/k")
    for sample_id, rows in pc.groupby("Sample"):
        require(np.all(np.diff(rows["PcHg"]) >= 0), f"Pc order: sample {sample_id}")
        require(np.all(np.diff(rows["Sw"]) <= 1e-8), f"Drainage order: {sample_id}")
        # Excel stores two representations of sample 89's porosity, differing
        # by ~4e-16 percentage points. Allow floating-point rounding only.
        for column in ("Porosity", "Perm"):
            np.testing.assert_allclose(
                rows[column], rows[column].iloc[0], rtol=1e-12, atol=0,
                err_msg=f"{column} changes within sample {sample_id}",
            )
    fits = pd.read_excel(sample("costa_thomeer_fits"))
    missing = sorted(set(pc["Sample"]) - set(fits["Sample"]))
    require(len(fits) == 109 and missing == [104], "Thomeer sample correspondence")

    wells = []
    for item in manifest["existing_costa_derivatives"]:
        well = lasio.read(DATA_DIR / item["path"])
        values = np.column_stack([well[name] for name in item["columns"]])
        require(len(values) == item["rows"], f"Row count: {item['path']}")
        require(
            numeric_digest(values) == item["numeric_sha256"],
            f"Retained COSTA values or null masks changed: {item['path']}",
        )
        if costa_source_dir is not None:
            source_path = costa_source_dir / item["upstream_path"]
            require(
                git_blob_hash(source_path.read_bytes()) == item["upstream_git_blob_sha1"],
                f"Original COSTA source differs from pinned upstream: {source_path}",
            )
            original = lasio.read(source_path)
            original_values = np.column_stack(
                [original[name] for name in item["source_columns"]]
            )
            np.testing.assert_allclose(values, original_values, atol=0, rtol=0)
        wells.append(well.data.shape[0])

    return {
        "status": "passed",
        "copied_files_hash_verified": copied,
        "discoverable_samples": list_samples("phillips"),
        "nmr_bins": {
            "shape": list(nmr.bins.shape),
            "t2_centers_ms": nmr.t2_centers.tolist(),
            "summary_max_absolute_error_pu": closure_errors,
            "t2lm_ms_range": [float(expected_lm.min()), float(expected_lm.max())],
            "field_acquisition_provenance_verified": False,
        },
        "paired_cmr_core": {
            "log_rows": len(log), "core_rows": len(core),
            "coates_calibration": {
                "parameters": {k: float(v) for k, v in calibration.params.items()},
                "r2_log": float(calibration.r2_log),
                "rmse_log": float(calibration.rmse_log),
                "n_points": int(calibration.n_points),
            },
            "porosity_basis": "supplied CMRP_3ms", "permeability_basis": "Kair",
            "same_data_fit_is_independent_validation": False,
        },
        "gulfcoast": {
            "rows": len(gulf), "valid_mphi": int(gulf["MPHI"].notna().sum()),
            "has_t2_bins": gulf_nmr.has_bins,
        },
        "costa_pc": {
            "rows": len(pc), "samples": int(pc["Sample"].nunique()),
            "monotonic_drainage_samples": 110,
            "thomeer_fit_rows": len(fits), "missing_fit_sample_ids": missing,
            "pressure_units_and_reservoir_fluid_conversion_verified": False,
            "plug_to_well_depth_mapping_available": False,
        },
        "existing_costa": {
            "wells": len(wells), "rows": sum(wells), "curves_per_well": 9,
            "source_checkout_compared": costa_source_dir is not None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costa-source-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.costa_source_dir)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
