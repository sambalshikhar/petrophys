"""Verify bundled Epslog/RealPore inputs without modifying the source data."""

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
from petrophys.interpretation.core_analysis import CoreMechanical, GrainSizeData
from petrophys.interpretation.geochemistry import XRFData


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(source_dir=None):
    manifest = json.loads(
        (DATA_DIR / "epslog/source_manifest.json").read_text(encoding="utf8")
    )
    copied = []
    for item in manifest["files"]:
        path = DATA_DIR / "epslog" / item["path"]
        raw = path.read_bytes()
        require(len(raw) == item["bytes"], f"Size changed: {path.name}")
        require(hashlib.sha256(raw).hexdigest() == item["sha256"],
                f"Hash changed: {path.name}")
        require(sample(path.stem) == path, f"Sample lookup failed: {path.stem}")
        if source_dir is not None and "source_relative_path" in item:
            original = Path(source_dir) / item["source_relative_path"]
            require(raw == original.read_bytes(), f"Source mismatch: {path.name}")
        copied.append(path.name)
    require(set(list_samples("epslog")) == {Path(p).stem for p in copied},
            "Epslog sample inventory changed")

    core = pd.read_csv(sample("epslog_15_12_20_s_core"))
    require(len(core) == 356 and core.depth.nunique() == 354,
            "Routine-core record counts changed")
    require(set(core.well) == {"15_12-20 S"}, "Wrong well in core extract")
    require(core.excel_row.is_unique, "Duplicate source Excel rows")
    for name, count in [("phi_reported", 345), ("kh_klinkenberg_reported", 243),
                        ("kv_klinkenberg_reported", 66), ("grain_density_gcc", 306)]:
        require(int(core[name].notna().sum()) == count, f"Changed count: {name}")

    core_results = {}
    complete_matches = 0
    for core_id, label, expected in [
        ("c01", "CORE 1", (582, 1527, 1673, 1532, 277, 274)),
        ("c02", "CORE 2", (1117, 2836, 3174, 2845, 200, 192)),
    ]:
        prefix = f"epslog_15_12_20_s_{core_id}"
        xrf = XRFData.from_las(sample(prefix + "_xrf"), error_suffix="_ER")
        require(xrf.n_samples == expected[0] and len(xrf.errors) == 40,
                f"XRF depth/error mapping failed: {core_id}")
        require(not any(k.endswith("_ER") for k in xrf.elements),
                "Error channels incorrectly classified as concentrations")
        ratios = xrf.elemental_ratios()
        np.testing.assert_allclose(ratios["Si_Al"], xrf.get("Si") / xrf.get("Al"))
        proxies = xrf.mineral_proxies()
        require(all(np.isfinite(v).all() for v in proxies.values()),
                f"Nonfinite XRF proxy: {core_id}")

        grain_path = sample(prefix + "_grain_segmentation")
        grain = GrainSizeData.from_las(
            grain_path, d50_mnemonic="P50GSDDML",
            d13_mnemonic=None, d87_mnemonic=None,
        )
        delivered = lasio.read(grain_path)
        finite = np.isfinite(grain.d50)
        require(int(finite.sum()) == expected[1], f"P50 count changed: {core_id}")
        require(np.all(delivered["P20GSDDML"][finite] <= grain.d50[finite]) and
                np.all(grain.d50[finite] <= delivered["P80GSDDML"][finite]),
                f"Invalid grain percentile ordering: {core_id}")
        require(int(np.sum(grain.wentworth() != "Unknown")) == expected[1],
                "Grain classification failed")
        require(np.isnan(grain.sorting_coefficient()).all(),
                "Sorting must remain unavailable without P13/P87")
        laser = GrainSizeData.from_las(
            sample(prefix + "_grain_laser"), d50_mnemonic="E_2DLAS_P50MSTD",
            d13_mnemonic=None, d87_mnemonic=None,
        )
        require(int(np.isfinite(laser.d50).sum()) == expected[2],
                f"Laser estimate count changed: {core_id}")

        mech = CoreMechanical.from_las(
            sample(prefix + "_strength_sonic"), ucs_mnemonic="E_STR001",
            vp_mnemonic="E_UVELP005", vs_mnemonic="E_UVELS005",
            poisson_mnemonic="E_PR005",
        )
        require(int(np.isfinite(mech.ucs).sum()) == expected[3],
                f"Scanner strength count changed: {core_id}")
        paired = np.isfinite(mech.vp) & np.isfinite(mech.vs)
        supplied = paired & np.isfinite(mech.poisson)
        require(int(paired.sum()) == expected[4] and int(supplied.sum()) == expected[5],
                f"Sonic/Poisson counts changed: {core_id}")
        calculated = mech.computed_poisson()
        np.testing.assert_allclose(calculated[supplied], mech.poisson[supplied],
                                   atol=3e-6, rtol=0)

        grains = pd.DataFrame({"grain_depth": grain.depth, "d50_um": grain.d50})
        joined = pd.merge_asof(
            core.loc[core.core == label].sort_values("depth"),
            grains.dropna().sort_values("grain_depth"), left_on="depth",
            right_on="grain_depth", direction="nearest", tolerance=0.05,
        )
        usable = joined[["phi_reported", "kh_klinkenberg_reported", "d50_um"]].notna().all(axis=1)
        usable &= (joined.phi_reported > 0) & (joined.kh_klinkenberg_reported > 0)
        matched = joined.loc[usable]
        matched_grain = GrainSizeData(depth=matched.depth.to_numpy(),
                                     d50=matched.d50_um.to_numpy())
        prediction = matched_grain.kozeny_carman_perm(matched.phi_reported.to_numpy() / 100)
        require(np.all(np.isfinite(prediction) & (prediction > 0)),
                "Kozeny-Carman calculation failed")
        complete_matches += len(matched)
        core_results[core_id] = {
            "xrf_samples": expected[0], "xrf_error_channels": len(xrf.errors),
            "image_p50_valid": expected[1], "laser_proxy_valid": expected[2],
            "scanner_strength_valid": expected[3], "vp_vs_pairs": expected[4],
            "supplied_dynamic_poisson_checked": expected[5],
            "max_poisson_rounding_difference": float(np.max(np.abs(calculated[supplied] - mech.poisson[supplied]))),
            "nominal_grain_phi_kh_matches": len(matched),
        }
    require(complete_matches == 214, "Nominal grain/core match count changed")
    return {
        "status": "passed", "data_files_hash_verified": copied,
        "original_las_compared": 8 if source_dir is not None else 0,
        "cores": core_results, "routine_core_records": len(core),
        "nominal_grain_phi_kh_matches": complete_matches,
        "limitations": [
            "Nominal core depths only; no calibrated depth tie.",
            "Porosity interpreted as percent for the calculation; source permeability units still need confirmation.",
            "Calculated permeability is an API smoke check, not validated prediction accuracy.",
            "No static moduli, independent plug UCS validation, bulk density or same-well wireline logs supplied.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.source_dir)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf8")
    print(encoded)
