"""Core analysis tables (CSV, Excel, DataFrame) into :class:`~petrophys.model.core.CoreData`.

Laboratories and databases name core columns many ways: ``Depth (m)``,
``Por (%)``, ``Helium porosity``, ``Kh Klinkenberg (mD)``, ``KAIR``, ``Grain
density (g/cc)``. :func:`read_core` finds the columns by name, reads units
in brackets, converts percent porosity and saturations to fractions (also
when a lab labels percentages as fractions) and permeability in darcy to mD,
and converts depths to the well's unit. What it decided is in
``core.metadata["import_report"]``.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from petrophys.exceptions import DataQualityWarning

from petrophys.io.survey_reader import _depth_unit_from, _name_and_unit

_ROLES = {
    "depth": ("depth", "md", "coredepth", "sampledepth", "plugdepth", "depthm", "depthft", "dept",
              "depthmd", "mddriller", "loggerdepth", "depthlogger", "shifteddepth"),
    "porosity": ("porosity", "por", "phi", "cpor", "phic", "heliumporosity", "heporosity", "phihe",
                 "porhe", "corepor", "coreporosity", "phireported", "porosityhelium", "ambientporosity",
                 "cporamb", "phieff"),
    "permeability": ("permeability", "perm", "kh", "kair", "kl", "kklinkenberg", "khklinkenberg",
                     "khorizontal", "khor", "cperm", "kcore", "permcore", "khair", "permh", "permeabilityh",
                     "khklinkenbergreported", "kh90", "kmax", "permeabilityhorizontal", "kliquid"),
    "kv": ("kv", "kvert", "kvertical", "permv", "kvklinkenberg", "kvair", "kvklinkenbergreported",
           "permeabilityvertical"),
    "grain_density": ("graindensity", "gd", "cgd", "rhog", "rhograin", "gden", "graindens", "rhoma",
                      "graindensitygcc", "matrixdensity"),
    "sw": ("sw", "swc", "watersaturation", "csw", "swcore", "sw_dean_stark", "swdeanstark"),
    "so": ("so", "soc", "oilsaturation", "cso", "socore"),
    "lithology": ("lithology", "lith", "description", "rocktype", "facies"),
}


#: Prefixes that mark a role when no exact name matches ("Kh Kl.", "Por He", "Kv90").
_PREFIXES = (
    ("kv", "kv"), ("permv", "kv"), ("kh", "permeability"), ("kair", "permeability"), ("kkl", "permeability"),
    ("perm", "permeability"), ("klink", "permeability"), ("porosity", "porosity"), ("por", "porosity"),
    ("phi", "porosity"), ("graind", "grain_density"), ("rhog", "grain_density"), ("watersat", "sw"),
    ("oilsat", "so"),
)


def _role_of(key: str) -> Optional[str]:
    for role, names in _ROLES.items():
        if key in names:
            return role
    if key == "k":
        return "permeability"
    for prefix, role in _PREFIXES:
        if key.startswith(prefix):
            return role
    return None


def read_core(
    source: Any,
    *,
    columns: Optional[Dict[str, str]] = None,
    depth_unit: Optional[str] = None,
    well: Any = None,
    sheet_name: Any = None,
    porosity_unit: Optional[str] = None,
    permeability_unit: Optional[str] = None,
):
    """Read a core analysis table into ``CoreData``.

    Parameters
    ----------
    source : path or DataFrame
        CSV/TSV/Excel file or DataFrame with one row per plug.
    columns : dict, optional
        Explicit headers for any role: ``depth``, ``porosity``,
        ``permeability`` (horizontal), ``kv``, ``grain_density``, ``sw``,
        ``so``, ``lithology``.
    depth_unit : str, optional
        ``"m"`` or ``"ft"`` of the core depths; read from a header such as
        ``Depth (ft)`` otherwise, else taken as the well's unit.
    well : Well, optional
        Converts core depths to the well's depth unit.
    sheet_name : optional
        Excel sheet (required when a workbook has several).
    porosity_unit, permeability_unit : str, optional
        Override the units of those columns (``"%"``, ``"v/v"``, ``"mD"``,
        ``"D"``) when the headers do not state them.

    Returns
    -------
    CoreData
        Depth, porosity and saturations (v/v), horizontal permeability (mD),
        grain density; vertical permeability and other columns in
        ``metadata``; decisions in ``metadata["import_report"]``.
    """
    from petrophys.interpretation.core_analysis.calibration import _core_permeability_to_md
    from petrophys.io.geology_reader import _read_table
    from petrophys.model.core import CoreData
    from petrophys.model.units import core_saturation_to_fraction

    table = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source, sheet_name=sheet_name)
    found: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    for column in table.columns:
        key, unit = _name_and_unit(column)
        role = _role_of(key)
        if role is None:
            # "Porosity %" or "Kh_mD": a unit glued to the name.
            match = re.match(r"^(.*?)(pct|percent|md|mdarcy|frac|fraction|gcc|gcm3|m|ft)$", key)
            if match and _role_of(match.group(1)):
                role, unit = _role_of(match.group(1)), {"pct": "%", "percent": "%", "mdarcy": "md",
                                                        "gcm3": "g/cc", "gcc": "g/cc"}.get(match.group(2), match.group(2))
        if role and role not in found:
            found[role] = column
            units[role] = unit or ("%" if "%" in str(column) else "")
    for role, header in (columns or {}).items():
        if role not in _ROLES:
            raise ValueError(f"Unknown core column role {role!r}; use {sorted(_ROLES)}")
        if header not in table.columns:
            raise ValueError(f"Core column {header!r} not found; columns are {list(table.columns)}")
        found[role] = header
        units[role] = _name_and_unit(header)[1] or ("%" if "%" in str(header) else "")
    if "depth" not in found:
        raise ValueError(
            f"Core table needs a depth column; none of {list(table.columns)} is recognised. "
            "Name it with columns={'depth': ...}."
        )
    if not any(role in found for role in ("porosity", "permeability", "grain_density", "sw", "so")):
        raise ValueError(
            f"No core property column recognised among {list(table.columns)}; "
            "name them with columns={'porosity': ..., 'permeability': ...}."
        )
    report: Dict[str, Any] = {"columns": {role: str(found[role]) for role in found}, "warnings": []}
    unused_numeric = [
        str(c) for c in table.columns
        if c not in found.values() and pd.to_numeric(table[c], errors="coerce").notna().mean() > 0.5
    ]
    if unused_numeric:
        report["unused_columns"] = unused_numeric
        report["warnings"].append(
            f"Numeric columns not used: {unused_numeric}; map any core property with columns={{role: header}}"
        )

    def numbers(role: str) -> Optional[np.ndarray]:
        if role not in found:
            return None
        column = found[role]
        raw = table[column]
        converted = pd.to_numeric(raw, errors="coerce")
        text = raw.astype(str).str.strip().str.lower()
        present = raw.notna() & ~text.isin(("", "nan", "na", "n/a", "null", "none"))
        invalid = present & converted.isna()
        if invalid.any():
            positions = np.flatnonzero(invalid.to_numpy()).tolist()
            report.setdefault("invalid_numeric_cells", {})[str(column)] = {
                "count": len(positions), "row_positions": positions,
            }
            report["warnings"].append(
                f"Column {column!r}: {len(positions)} nonnumeric value(s) replaced by NaN "
                f"at zero-based data row positions {positions[:5]}"
            )
        values = converted.to_numpy(dtype=float, copy=True)
        values[values <= -999.0] = np.nan
        return values

    depth = numbers("depth")
    unit = _depth_unit_from(depth_unit) if depth_unit else _depth_unit_from(units.get("depth", ""))
    if well is not None:
        target = str(well.depth_unit)
        if unit and unit != target:
            depth = depth / 0.3048 if (unit, target) == ("m", "ft") else depth * 0.3048
            report["depth_conversion"] = f"{unit} to {target}"
        elif unit is None:
            report["warnings"].append(f"Core depth unit not stated; taken as the well's ({target})")
        unit = target
    report["depth_unit"] = unit or ""

    converted: Dict[str, Optional[np.ndarray]] = {}
    for role in ("porosity", "sw", "so"):
        values = numbers(role)
        if values is None:
            converted[role] = None
            continue
        label = porosity_unit if (role == "porosity" and porosity_unit) else units.get(role, "")
        fractions, record = core_saturation_to_fraction(
            values, label, mnemonic=str(found[role]),
            quantity="porosity" if role == "porosity" else "saturation",
        )
        converted[role] = fractions
        report[f"{role}_normalization"] = record
        report["warnings"].extend(record.get("warnings", []))
    perm = numbers("permeability")
    if perm is not None:
        perm = _core_permeability_to_md(perm, permeability_unit or units.get("permeability", ""), str(found["permeability"]))
    kv = numbers("kv")
    if kv is not None:
        kv = _core_permeability_to_md(kv, permeability_unit or units.get("kv", ""), str(found["kv"]))
    grain = numbers("grain_density")
    if grain is not None and np.nanmedian(grain) > 100:  # kg/m3
        grain = grain / 1000.0
        report["warnings"].append("Grain density read as kg/m3 and converted to g/cc")
    keep = np.isfinite(depth)
    order = np.argsort(depth[keep], kind="stable")

    def pick(values):
        return None if values is None else values[keep][order]

    used = set(found.values())
    metadata = {str(c): table[c].to_numpy()[keep][order] for c in table.columns if c not in used}
    if kv is not None:
        metadata["kv"] = pick(kv)
    metadata["import_report"] = report
    for message in report["warnings"]:
        warnings.warn(f"Core table: {message}", DataQualityWarning, stacklevel=2)
    lithology = table[found["lithology"]].to_numpy()[keep][order] if "lithology" in found else None
    return CoreData(
        depth=depth[keep][order], porosity=pick(converted["porosity"]), permeability=pick(perm),
        grain_density=pick(grain), sw=pick(converted["sw"]), so=pick(converted["so"]),
        lithology=lithology, metadata=metadata,
    )


__all__ = ["read_core"]
