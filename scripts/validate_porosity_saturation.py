"""Empirical validation of the porosity / clay-bound-water / bad-hole corrections.

Ground truth:
  * 30/6-5 -- CORE porosity (PHIC curve, cored 2800-2948 m). The corrected PHIE
    must still match core there (the corrections must not touch clean, in-gauge
    reservoir).
  * NPD-confirmed DRY wells -- net pay should collapse toward zero.
  * Gas/HC wells -- net pay should survive.

Run:  python scripts/validate_porosity_saturation.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from petrophys.io import read_las
from petrophys.workflow.dynamic_workflow import run_dynamic_workflow

warnings.filterwarnings("ignore")

# Full multi-well validation runs against a folder of GeoLink LAS files: set
# PETROPHYS_GEOLINK_DIR to point at it. Without that, only the wells bundled in
# the package (petrophys/data/geolink) are validated and the rest are skipped.
BUNDLED = Path(__file__).resolve().parents[1] / "petrophys" / "data" / "geolink"
_env_dir = os.environ.get("PETROPHYS_GEOLINK_DIR")
GEOLINK = Path(_env_dir) if _env_dir else None

# well -> expected outcome (from NPD wellbore reports)
WELLS = {
    "30_6-5":  "core",   # cored; PHIE must match PHIC
    "15_9-23": "dry",
    "16_7-6":  "dry",
    "25_10-9": "dry",
    "16_10-3": "dry",
    "16_10-5": "dry",
    "25_6-3":  "dry",
    "34_11-1": "gas",
    "34_12-1": "gas",
    "35_11-10": "gas",
}


def _find(name: str) -> Path | None:
    for base in (BUNDLED, GEOLINK):
        if base is None:
            continue
        p = base / f"{name}.las"
        if p.exists():
            return p
    return None


def _g(w, *names):
    for n in names:
        c = w.get(n)
        if c is not None:
            return np.asarray(c.data, dtype=float)
    return None


def run_one(name: str, expect: str) -> dict:
    path = _find(name)
    if path is None:
        return {"well": name, "expect": expect, "error": "LAS not found"}
    w = read_las(path)
    # auto-derive Rw (do not override); clastic matrix
    run_dynamic_workflow(w, lithology="clastic")
    dep = np.asarray(w.depth, dtype=float)
    step = w.step or 0.15
    netpay = _g(w, "NET_PAY")
    netres = _g(w, "NET_RES")
    phie = _g(w, "PHIE_CC", "PHIE")
    out = {
        "well": name, "expect": expect,
        "net_pay_m": float(np.nansum(netpay > 0.5) * step) if netpay is not None else None,
        "net_res_m": float(np.nansum(netres > 0.5) * step) if netres is not None else None,
        "gross_m": float((np.nanmax(dep) - np.nanmin(dep))),
        "phie_p50": float(np.nanmedian(phie)) if phie is not None else None,
    }
    # core match for 30/6-5: PHIC is TOTAL (helium) porosity, so compare to PHIT.
    # PHIE (effective) is expected to be lower by the clay-bound water.
    phic = _g(w, "PHIC")
    phit = _g(w, "PHIT")
    if phic is not None and phit is not None:
        m = np.isfinite(phic) & np.isfinite(phit)
        if m.any():
            out["core_phit_p50"] = float(np.nanmedian(phit[m]))
            out["core_phic_p50"] = float(np.nanmedian(phic[m]))
            out["core_bias"] = float(np.nanmean(phit[m] - phic[m]))
            out["core_mae"] = float(np.nanmean(np.abs(phit[m] - phic[m])))
            if phie is not None:
                out["core_phie_p50"] = float(np.nanmedian(phie[m]))
    return out


def main():
    rows = [run_one(n, e) for n, e in WELLS.items()]
    print(f"\n{'well':9s} {'exp':5s} {'net_pay_m':>9s} {'net_res_m':>9s} {'gross_m':>8s} {'PHIE50':>7s}  notes")
    print("-" * 78)
    for r in rows:
        if "error" in r:
            print(f"{r['well']:9s} {r['expect']:5s}  {r['error']}")
            continue
        note = ""
        if r["expect"] == "core" and "core_bias" in r:
            note = (f"CORE PHIT={r['core_phit_p50']:.3f} vs PHIC={r['core_phic_p50']:.3f} "
                    f"(bias {r['core_bias']:+.3f}, MAE {r['core_mae']:.3f}); PHIE={r.get('core_phie_p50', float('nan')):.3f} (<core: clay-bound water)")
        pay = r["net_pay_m"]
        print(f"{r['well']:9s} {r['expect']:5s} {pay:9.1f} {r['net_res_m']:9.1f} {r['gross_m']:8.0f} {r['phie_p50']:7.3f}  {note}")
    # summary verdict
    dry = [r for r in rows if r.get("expect") == "dry" and "net_pay_m" in r]
    gas = [r for r in rows if r.get("expect") == "gas" and "net_pay_m" in r]
    if dry:
        print(f"\nDRY wells   net-pay: median {np.median([r['net_pay_m'] for r in dry]):.1f} m, "
              f"max {max(r['net_pay_m'] for r in dry):.1f} m  (target ~0)")
    if gas:
        print(f"GAS wells   net-pay: median {np.median([r['net_pay_m'] for r in gas]):.1f} m, "
              f"min {min(r['net_pay_m'] for r in gas):.1f} m  (target >0)")


if __name__ == "__main__":
    main()
