"""Validated tabular inputs for deterministic context generation."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional

import numpy as np
import pandas as pd

from petrophys.io.geology_reader import _key, _read_table, _well_key
from petrophys.model.units import normalize_depth_unit, temperature_to_fahrenheit


_PARAMETER_ALIASES = {
    "well": ("well", "wellname", "wellidentifier", "wellid", "uwi", "api"),
    "lithology_domain": ("lithologydomain", "interpretationdomain", "domain"),
    "matrix_type": ("matrixtype", "matrix"),
    "rw": ("rw", "formationwaterresistivity"),
    "rw_meas_temp_f": ("rwreferencetemperature", "rwmeasurementtemperature", "rwtemp"),
    "rw_temperature_unit": ("rwtemperatureunit", "rwtempunit"),
    "rw_unit": ("rwunit",),
    "bht_f": ("bht", "bottomholetemperature"),
    "bht_unit": ("bhtunit", "bottomholetemperatureunit"),
    "surface_temp_f": ("surfacetemperature", "surfacetemp"),
    "surface_temp_unit": ("surfacetemperatureunit", "surfacetempunit"),
    "td_ft": ("td", "totaldepth"),
    "depth_unit": ("depthunit", "tdunit"),
    "gr_min": ("grclean", "grmin"),
    "gr_max": ("grshale", "grmax"),
}

_BOOLEAN_FIELDS = {
    "standardize_input", "rename_input_curves", "allow_default_rw",
    "mud_has_clay_or_kcl", "raise_on_error", "apply_hc_correction", "use_peer_core",
}

_LIST_FIELDS = {"skip_steps"}


def _column_mapping(columns) -> dict[str, str]:
    normalized = {_key(column): str(column) for column in columns}
    if len(normalized) != len(columns):
        raise ValueError("Duplicate input column names after normalizing whitespace/case.")
    mapping: dict[str, str] = {}
    for canonical, aliases in _PARAMETER_ALIASES.items():
        matches = [normalized[name] for name in aliases if name in normalized]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous columns for {canonical}: {matches}")
        if matches:
            mapping[matches[0]] = canonical
    return mapping


def _select_well_row(table: pd.DataFrame, well_name: Optional[str]) -> pd.Series:
    if "well" not in table:
        if len(table) != 1:
            raise ValueError("A multirow well-parameters table requires a well column.")
        return table.iloc[0]
    identifiers = table["well"].astype(str).str.strip()
    if identifiers.eq("").any():
        raise ValueError("Every well-parameters row requires a well identifier.")
    if well_name is None:
        if len(table) != 1:
            raise ValueError("Select well_name when a well-parameters table contains multiple rows.")
        return table.iloc[0]
    selected = table.loc[identifiers.map(_well_key) == _well_key(well_name)]
    if selected.empty:
        raise ValueError(
            f"No well-parameters row matches {well_name!r}; available: {identifiers.tolist()}"
        )
    if len(selected) != 1:
        raise ValueError(f"Well-parameters table contains duplicate rows for {well_name!r}.")
    return selected.iloc[0]


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Well parameter {field} must be numeric; got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError(f"Well parameter {field} must be finite; got {value!r}.")
    return result


def _boolean(value: Any, field: str) -> bool:
    key = str(value).strip().lower()
    if key in {"true", "yes", "y", "1"}:
        return True
    if key in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Well parameter {field} must be true or false; got {value!r}.")


def read_well_parameters(source: Any, *, well_name: Optional[str] = None) -> dict[str, Any]:
    """Read one well's workflow settings from CSV, TSV, Excel or a DataFrame.

    Columns use :class:`petrophys.DynamicWorkflowConfig` names. Friendly aliases
    from the packaged template are accepted, including ``lithology_domain``,
    ``gr_clean``, ``gr_shale``, ``rw_reference_temperature`` and ``bht``.
    Blank cells mean "use the workflow default or LAS metadata". Temperatures
    are converted to degrees Fahrenheit and total depth to feet before the
    configuration is constructed.
    """
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig

    table = _read_table(source)
    if table.empty:
        raise ValueError("Well-parameters input contains no data rows.")
    table = table.rename(columns=_column_mapping(table.columns))

    config_fields = {item.name: item for item in fields(DynamicWorkflowConfig)}
    accepted = set(config_fields) | {
        "well", "rw_temperature_unit", "rw_unit", "bht_unit",
        "surface_temp_unit", "depth_unit",
    }
    unknown = [str(column) for column in table.columns if column not in accepted]
    if unknown:
        raise ValueError(
            f"Unsupported well-parameters columns {unknown}; use the packaged template."
        )

    row = _select_well_row(table, well_name)
    raw = {
        str(name): value for name, value in row.items()
        if name != "well" and not pd.isna(value) and str(value).strip() != ""
    }

    rw_unit = str(raw.pop("rw_unit", "ohm.m")).strip().lower().replace(" ", "")
    if rw_unit not in {"ohm.m", "ohm-m", "ohmm", "ohm*m"}:
        raise ValueError("rw_unit must be ohm.m.")

    rw_temp_unit = raw.pop("rw_temperature_unit", "F")
    bht_unit = raw.pop("bht_unit", "F")
    surface_temp_unit = raw.pop("surface_temp_unit", "F")
    depth_unit = raw.pop("depth_unit", "ft")

    numeric_fields = {
        name for name in config_fields
        if name not in _BOOLEAN_FIELDS | _LIST_FIELDS
        and name not in {
            "lithology_domain", "matrix_type", "lithology", "lithology_csv",
            "lithology_intervals", "lithology_source", "mud_type", "mud_water",
            "hc_type", "gr_method", "perm_method", "perm_fallback_method",
            "formation_tops", "peer_wells", "sw_model",
        }
    }
    options: dict[str, Any] = {}
    for name, value in raw.items():
        if name in _BOOLEAN_FIELDS:
            options[name] = _boolean(value, name)
        elif name in _LIST_FIELDS:
            options[name] = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
        elif name in numeric_fields:
            options[name] = _number(value, name)
        else:
            options[name] = str(value).strip()

    if "rw_meas_temp_f" in options:
        options["rw_meas_temp_f"] = temperature_to_fahrenheit(
            options["rw_meas_temp_f"], rw_temp_unit
        )
    if "bht_f" in options:
        options["bht_f"] = temperature_to_fahrenheit(options["bht_f"], bht_unit)
    if "surface_temp_f" in options:
        options["surface_temp_f"] = temperature_to_fahrenheit(
            options["surface_temp_f"], surface_temp_unit
        )
    if "td_ft" in options:
        unit = normalize_depth_unit(depth_unit)
        if unit == "m":
            options["td_ft"] /= 0.3048

    validated = DynamicWorkflowConfig(**options)
    return {name: getattr(validated, name) for name in options}


_CORE_ALIASES = {
    "well": ("well", "wellname", "wellidentifier", "wellid", "uwi", "api"),
    "depth": ("depth", "md", "measureddepth", "coredepth"),
    "depth_unit": ("depthunit", "unit"),
    "core_to_log_shift": ("coretologshift", "depthshift", "coreshift"),
    "core_porosity": ("coreporosity", "phic", "cpor", "porosity"),
    "porosity_unit": ("porosityunit", "phicunit"),
    "core_permeability": ("corepermeability", "khcore", "cperm", "permeability"),
    "permeability_unit": ("permeabilityunit", "permunit"),
    "core_water_saturation": ("corewatersaturation", "swc", "coresw", "csw"),
    "core_oil_saturation": ("coreoilsaturation", "soc", "coreso", "cso"),
    "saturation_unit": ("saturationunit", "satunit"),
    "grain_density": ("graindensity", "gden", "cgd"),
}


def _normalize_fraction(
    values: pd.Series,
    unit: str,
    name: str,
    *,
    infer_percent: bool = False,
) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if infer_percent:
        from petrophys.model.curve import Curve
        from petrophys.preprocessing.standardization import standardize_core_saturation

        return standardize_core_saturation(
            Curve(name, result, unit),
        ).data
    key = str(unit).strip().lower().replace(" ", "")
    if key in {"%", "percent", "pct", "percentage"}:
        result /= 100.0
    elif key not in {"", "frac", "fraction", "v/v", "v/vfrac", "decimal"}:
        raise ValueError(f"{name} unit must be fraction/v/v or percent; got {unit!r}.")
    invalid = np.isfinite(result) & ((result < 0.0) | (result > 1.0))
    if np.any(invalid):
        raise ValueError(f"{name} contains values outside 0-1 after unit conversion.")
    return result


def attach_core_measurements(well: Any, source: Any, *, well_name: Optional[str] = None) -> dict[str, Any]:
    """Validate sparse core measurements and attach them to nearest log rows."""
    table = _read_table(source)
    if table.empty:
        raise ValueError("Core-measurements input contains no data rows.")
    normalized = {_key(column): str(column) for column in table.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in _CORE_ALIASES.items():
        matches = [normalized[name] for name in aliases if name in normalized]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous core columns for {canonical}: {matches}")
        if matches:
            mapping[matches[0]] = canonical
    table = table.rename(columns=mapping)
    unknown = [str(column) for column in table.columns if column not in _CORE_ALIASES]
    if unknown:
        raise ValueError(f"Unsupported core-measurements columns {unknown}; use the packaged template.")
    if "depth" not in table:
        raise ValueError("Core measurements require a depth column.")
    if "well" in table:
        target = well_name or well.name
        table = table.loc[table["well"].astype(str).map(_well_key) == _well_key(target)]
        if table.empty:
            raise ValueError(f"No core-measurements rows match {target!r}.")

    value_names = (
        "core_porosity", "core_permeability", "core_water_saturation",
        "core_oil_saturation", "grain_density",
    )
    if not any(name in table for name in value_names):
        raise ValueError("Core measurements require at least one measurement column.")

    def one_value(name: str, default: str) -> str:
        if name not in table:
            return default
        values = [
            str(value).strip()
            for value in table[name]
            if not pd.isna(value) and str(value).strip()
        ]
        if len(set(values)) > 1:
            raise ValueError(f"Core table contains conflicting {name} values: {sorted(set(values))}")
        return values[0] if values else default

    source_depth_unit = normalize_depth_unit(one_value("depth_unit", well.depth_unit))
    target_depth_unit = normalize_depth_unit(well.depth_unit)
    depths = pd.to_numeric(table["depth"], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if not np.all(np.isfinite(depths)):
        raise ValueError("Every core depth must be finite.")
    shifts = (
        pd.to_numeric(table["core_to_log_shift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "core_to_log_shift" in table else np.zeros(len(table), dtype=float)
    )
    depths += shifts
    if source_depth_unit != target_depth_unit:
        depths *= 0.3048 if source_depth_unit == "ft" else 1.0 / 0.3048

    log_depth = np.asarray(well.depth, dtype=float)
    finite_log = log_depth[np.isfinite(log_depth)]
    if finite_log.size < 2:
        raise ValueError("The well needs at least two finite depth samples for core alignment.")
    tolerance = 0.51 * float(np.nanmedian(np.abs(np.diff(np.sort(finite_log)))))
    nearest = np.asarray([int(np.nanargmin(np.abs(log_depth - depth))) for depth in depths])
    distance = np.abs(log_depth[nearest] - depths)
    if np.any(distance > tolerance):
        first = int(np.flatnonzero(distance > tolerance)[0])
        raise ValueError(
            f"Core depth {depths[first]:g} {target_depth_unit} does not align with the LAS "
            f"within {tolerance:g} {target_depth_unit}; verify units and core_to_log_shift."
        )

    arrays: dict[str, np.ndarray] = {}
    if "core_porosity" in table:
        arrays["PHIC"] = _normalize_fraction(
            table["core_porosity"], one_value("porosity_unit", "fraction"), "core_porosity"
        )
    if "core_permeability" in table:
        perm = pd.to_numeric(
            table["core_permeability"],
            errors="coerce",
        ).to_numpy(dtype=float, copy=True)
        unit = one_value("permeability_unit", "mD").strip().lower()
        if unit in {"d", "darcy", "darcies"}:
            perm *= 1000.0
        elif unit not in {"md", "millidarcy", "millidarcies"}:
            raise ValueError(f"permeability_unit must be mD or D; got {unit!r}.")
        if np.any(np.isfinite(perm) & (perm < 0.0)):
            raise ValueError("core_permeability cannot be negative.")
        arrays["KH_CORE"] = perm
    saturation_unit = one_value("saturation_unit", "fraction")
    if "core_water_saturation" in table:
        arrays["SWC"] = _normalize_fraction(
            table["core_water_saturation"],
            saturation_unit,
            "core_water_saturation",
            infer_percent=True,
        )
    if "core_oil_saturation" in table:
        arrays["SOC"] = _normalize_fraction(
            table["core_oil_saturation"],
            saturation_unit,
            "core_oil_saturation",
            infer_percent=True,
        )
    if "grain_density" in table:
        grain_density = pd.to_numeric(table["grain_density"], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(grain_density) & (grain_density <= 0.0)):
            raise ValueError("grain_density must be positive.")
        arrays["GDEN"] = grain_density

    descriptions = {
        "PHIC": ("v/v", "Core porosity"),
        "KH_CORE": ("mD", "Core permeability"),
        "SWC": ("v/v", "Core water saturation"),
        "SOC": ("v/v", "Core oil saturation"),
        "GDEN": ("g/cm3", "Core grain density"),
    }
    for mnemonic, source_values in arrays.items():
        aligned = np.full(well.n_samples, np.nan, dtype=float)
        for index in np.unique(nearest):
            selected = source_values[nearest == index]
            finite = selected[np.isfinite(selected)]
            if finite.size:
                aligned[index] = float(np.mean(finite))
        unit, description = descriptions[mnemonic]
        well.add_curve(mnemonic, aligned, unit, description)

    report = {
        "well": well_name or well.name,
        "input_rows": int(len(table)),
        "depth_unit": target_depth_unit,
        "alignment_tolerance": tolerance,
        "curves": sorted(arrays),
        "duplicate_log_rows_averaged": int(len(nearest) - len(np.unique(nearest))),
    }
    context = dict(well.get_parameter("_geology_context") or {})
    context["core_import_report"] = report
    well.set_parameter("_geology_context", context)
    return report
