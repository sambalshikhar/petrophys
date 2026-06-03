"""
Central value-range quality control for exported petrophysical curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

VALUE_RANGE_QC_PARAMETER = "VALUE_RANGE_QC_REPORT"


@dataclass(frozen=True)
class ValueRangeRule:
    """QC rule for one canonical curve family."""

    low: float
    high: float
    action: str = "nan"
    aliases: tuple[str, ...] = ()


VALUE_RANGE_RULES: dict[str, ValueRangeRule] = {
    "RMF": ValueRangeRule(0.001, 10.0, "nan", ("RMF",)),
    "TEMP": ValueRangeRule(-5.0, 400.0, "nan", ("TEMP", "TMPC", "CMR_TEMP")),
    "RXO": ValueRangeRule(0.01, 2000.0, "nan", ("RXO", "RSHA", "RESS", "MSFL", "SFL")),
    "RDEP": ValueRangeRule(0.01, 2000.0, "nan", ("RDEP", "RT", "RMED", "RESM", "ILD")),
    "GR": ValueRangeRule(0.0, 400.0, "nan", ("GR", "GRC")),
    "DTC": ValueRangeRule(20.0, 300.0, "nan", ("DTC", "DT")),
    "DTS": ValueRangeRule(20.0, 600.0, "nan", ("DTS", "DTE")),
    "RHOB": ValueRangeRule(1.0, 3.5, "nan", ("RHOB",)),
    "NPHI": ValueRangeRule(-0.15, 0.60, "nan", ("NPHI",)),
    "PEF": ValueRangeRule(0.0, 15.0, "nan", ("PEF",)),
    "CALI": ValueRangeRule(
        4.0, 40.0, "nan", ("CALI", "CAL", "DCAL", "HD1_PPC1", "HD2_PPC1")
    ),
    "BS": ValueRangeRule(4.0, 40.0, "nan", ("BS",)),
    "PHIS": ValueRangeRule(-0.05, 0.60, "nan", ("PHIS",)),
    "PERM": ValueRangeRule(1e-4, 2e4, "nan", ("PERM",)),
    "PERM_CC": ValueRangeRule(1e-4, 2e4, "nan", ("PERM_CC",)),
    "PHID": ValueRangeRule(-0.05, 0.60, "nan", ("PHID",)),
    "PHIN": ValueRangeRule(-0.15, 0.60, "nan", ("PHIN",)),
    "PHIND": ValueRangeRule(-0.05, 0.60, "nan", ("PHIND",)),
    "PHIRH": ValueRangeRule(-0.05, 0.60, "nan", ("PHIRH",)),
    "PHIT": ValueRangeRule(0.0, 0.45, "nan", ("PHIT",)),
    "PHIE": ValueRangeRule(0.0, 0.45, "nan", ("PHIE",)),
    "PHIE_CC": ValueRangeRule(0.0, 0.45, "nan", ("PHIE_CC",)),
    "NMR_PHIT": ValueRangeRule(0.0, 100.0, "nan", ("NMR_PHIT", "MPHI", "TCMR")),
    "NMR_FFI": ValueRangeRule(0.0, 100.0, "nan", ("NMR_FFI", "MFFI", "CMFF")),
    "NMR_BVI": ValueRangeRule(0.0, 100.0, "nan", ("NMR_BVI", "MBVI", "CBVI")),
    "NMR_T2LM": ValueRangeRule(0.0, 10000.0, "nan", ("NMR_T2LM", "T2LM", "T2ML")),
    "NMR_T2CUTOFF": ValueRangeRule(0.0, 10000.0, "nan", ("NMR_T2CUTOFF", "T2CUTOFF")),
    "NMR_KSDR": ValueRangeRule(0.0, 2e4, "nan", ("NMR_KSDR", "KSDR")),
    "NMR_KTIM": ValueRangeRule(0.0, 2e4, "nan", ("NMR_KTIM", "KTIM")),
    "NMR_3MS": ValueRangeRule(0.0, 100.0, "nan", ("NMR_3MS", "CMRP_3MS")),
    "PHI_SEC": ValueRangeRule(-0.05, 0.60, "nan", ("PHI_SEC",)),
    "RESFLAG": ValueRangeRule(
        0.0,
        1.0,
        "nan",
        ("RESFLAG", "NET_ROCK", "NET_RES", "NET_PAY", "SEALFLAG", "CROSSOVER_FLAG"),
    ),
    "RW": ValueRangeRule(0.005, 20.0, "nan", ("RW",)),
    "RWA": ValueRangeRule(0.005, 20.0, "nan", ("RWA",)),
    "SW": ValueRangeRule(0.0, 1.0, "clip", ("SW", "SWA", "SWT-LOG")),
    "SO": ValueRangeRule(0.0, 1.0, "clip", ("SO",)),
    "SXO": ValueRangeRule(0.0, 1.0, "clip", ("SXO", "SWXO", "SWXOA", "SXO-LOG")),
    "SWIRR_APP": ValueRangeRule(0.0, 1.0, "nan", ("SWIRR_APP", "SWIRR", "SWI", "SWIR")),
    "BVW": ValueRangeRule(0.0, 1.0, "nan", ("BVW",)),
    "MHI": ValueRangeRule(0.0, 1.0, "nan", ("MHI",)),
    "VDOL": ValueRangeRule(0.0, 1.0, "nan", ("VDOL", "VDOLO")),
    "VLIM": ValueRangeRule(0.0, 1.0, "nan", ("VLIM", "VCALC")),
    "VSAND": ValueRangeRule(0.0, 1.0, "nan", ("VSAND", "VQUAR")),
    "VANHY": ValueRangeRule(0.0, 1.0, "nan", ("VANHY",)),
    "VSH": ValueRangeRule(
        0.0,
        1.0,
        "nan",
        ("VSH", "VCL", "VCLAY", "VSHGR", "VSHND", "VCL_GR", "VCL_SP", "VCL_ND"),
    ),
    "VP_VS_RATIO": ValueRangeRule(0.5, 5.0, "nan", ("VP_VS_RATIO",)),
    "POISSON_RATIO": ValueRangeRule(0.0, 0.49, "nan", ("POISSON_RATIO",)),
    "BRITTLENESS_INDEX": ValueRangeRule(0.0, 1.0, "nan", ("BRITTLENESS_INDEX",)),
    "ACOUSTIC_IMPEDANCE": ValueRangeRule(1.0e6, 3.0e7, "nan", ("ACOUSTIC_IMPEDANCE",)),
    "YOUNG_MOD": ValueRangeRule(0.0, 150.0, "nan", ("YOUNG_MOD",)),
    "S_V": ValueRangeRule(0.0, 250.0, "nan", ("S_V",)),
    "PP_HYDROSTATIC": ValueRangeRule(0.0, 150.0, "nan", ("PP_HYDROSTATIC",)),
    "SH_MIN": ValueRangeRule(0.0, 250.0, "nan", ("SH_MIN",)),
    "DATA_COMPLETENESS": ValueRangeRule(0.0, 1.0, "nan", ("DATA_COMPLETENESS",)),
    "CONSISTENCY_RATIO": ValueRangeRule(0.0, 1.0, "nan", ("CONSISTENCY_RATIO",)),
}


def _rule_lookup() -> dict[str, tuple[str, ValueRangeRule]]:
    lookup: dict[str, tuple[str, ValueRangeRule]] = {}
    for canonical, rule in VALUE_RANGE_RULES.items():
        lookup[canonical.upper()] = (canonical, rule)
        for alias in rule.aliases:
            lookup[alias.upper()] = (canonical, rule)
    return lookup


_RULE_LOOKUP = _rule_lookup()


def rule_for_curve(mnemonic: str) -> Optional[tuple[str, ValueRangeRule]]:
    """Return the canonical range rule for a curve mnemonic."""
    upper = str(mnemonic or "").upper()
    if upper.startswith("SXO-"):
        return "SXO", VALUE_RANGE_RULES["SXO"]
    return _RULE_LOOKUP.get(upper)


def _qc_curve_data(
    data: np.ndarray, rule: ValueRangeRule
) -> tuple[np.ndarray, dict[str, int]]:
    arr = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(arr)
    below = finite & (arr < rule.low)
    above = finite & (arr > rule.high)
    changed = below | above
    if rule.action == "clip":
        arr[finite] = np.clip(arr[finite], rule.low, rule.high)
    else:
        arr[changed] = np.nan
    return arr, {
        "changed": int(np.count_nonzero(changed)),
        "below": int(np.count_nonzero(below)),
        "above": int(np.count_nonzero(above)),
    }


def clean_curve_values(
    mnemonic: str, data: Any
) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
    """Clean one curve array with its configured value-range rule."""
    match = rule_for_curve(mnemonic)
    arr = np.asarray(data, dtype=float)
    if match is None:
        return arr.copy(), None
    canonical, rule = match
    cleaned, stats = _qc_curve_data(arr, rule)
    return cleaned, {
        "curve": str(mnemonic).upper(),
        "canonical_curve": canonical,
        "low": float(rule.low),
        "high": float(rule.high),
        "action": rule.action,
        "changed": stats["changed"],
        "below": stats["below"],
        "above": stats["above"],
    }


def apply_value_range_qc(
    well: "Well",
    curve_names: Optional[Iterable[str]] = None,
    store_report: bool = True,
) -> list[dict[str, Any]]:
    """
    Apply configured value-range QC to curves in a well.

    Values outside ``nan`` rules are set to NaN. Values outside ``clip`` rules
    are clipped to the configured low/high bounds.
    """
    names = list(curve_names or getattr(well, "curves", []))
    report: list[dict[str, Any]] = []

    for name in names:
        curve = well.get(name) if hasattr(well, "get") else None
        if curve is None:
            continue
        try:
            cleaned, item = clean_curve_values(name, curve.data)
        except (TypeError, ValueError):
            continue
        if item is None:
            continue
        if item["changed"] > 0:
            curve._data = cleaned
        report.append(item)

    if store_report and hasattr(well, "set_parameter"):
        well.set_parameter(VALUE_RANGE_QC_PARAMETER, report)
    return report


def value_range_qc_note(
    report: Iterable[Mapping[str, Any]], max_items: int = 12
) -> str:
    """Build a compact human-readable value QC note."""
    items = list(report or [])
    changed_items = [item for item in items if int(item.get("changed") or 0) > 0]
    base = (
        "VALUE RANGE QC: finite values outside configured ranges were set to NaN; "
        "SW/SWA and SXO/SWXO/SWXOA were clipped to 0-1."
    )
    if not changed_items:
        return base + " No values changed in this segment."

    parts = []
    for item in changed_items[:max_items]:
        action = "clipped" if item.get("action") == "clip" else "set NaN"
        parts.append(
            f"{item.get('curve')} {action}={item.get('changed')} "
            f"(<{item.get('low')}:{item.get('below')}, >{item.get('high')}:{item.get('above')})"
        )
    if len(changed_items) > max_items:
        parts.append(f"+{len(changed_items) - max_items} more curves")
    return base + " Changed: " + "; ".join(parts) + "."
