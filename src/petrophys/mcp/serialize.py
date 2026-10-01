"""Compact JSON responses of the MCP server, built on the library's own summaries.

Responses are read by language models, so they are compact: floats are
rounded to four significant digits (at least two decimals), tables are rows
with a column legend, and every response that contains depths says which
depth unit they are in.
"""

from __future__ import annotations

import difflib
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from petrophys._internal.serialization import (
    curve_statistics,
    json_object,
    json_safe,
)
from petrophys._internal.serialization import depth_mask as _depth_mask

if TYPE_CHECKING:
    from petrophys.model.well import Well

__all__ = [
    "compact",
    "compact_object",
    "curve_statistics",
    "decimated_values",
    "depth_mask",
    "header_notes",
    "header_parameters",
    "interpretation_summary",
    "json_object",
    "json_safe",
    "step_detail",
    "well_summary",
    "workflow_summary",
]

#: Columns of the curve rows in well summaries.
CURVE_COLUMNS = ["mnemonic", "unit", "valid_%", "valid_top", "valid_bottom", "description"]

#: Columns of the formation rows in well summaries.
FORMATION_COLUMNS = ["name", "top", "bottom", "rank"]

#: Most items of a list kept in an explain_step record.
_MAX_LIST_ITEMS = 12


# -- numbers ---------------------------------------------------------------------

def round_float(value: float) -> float:
    """Round to four significant digits, keeping at least two decimals."""
    if not math.isfinite(value) or value == 0:
        return value
    magnitude = int(math.floor(math.log10(abs(value))))
    return round(value, max(2, 3 - magnitude))


def compact(value: Any) -> Any:
    """JSON-safe copy of ``value`` with rounded floats."""
    return _round_all(json_safe(value))


def compact_object(value: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`compact` for a mapping, typed as a mapping."""
    result: Dict[str, Any] = compact(value)
    return result


def _round_all(value: Any) -> Any:
    if isinstance(value, float):
        return round_float(value)
    if isinstance(value, dict):
        return {key: _round_all(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_all(item) for item in value]
    return value


def depth_mask(well: "Well", top: Optional[float], bottom: Optional[float]) -> np.ndarray:
    """Boolean mask selecting samples between ``top`` and ``bottom`` (inclusive)."""
    return _depth_mask(well.depth, top, bottom)


def compact_statistics(values: Any) -> Dict[str, Any]:
    """:func:`curve_statistics` without the sample count (``valid`` stays)."""
    stats = curve_statistics(values)
    stats.pop("samples", None)
    return stats


# -- wells --------------------------------------------------------------------------

def _describe_unit(well: "Well", name: str) -> str:
    unit = well.get_parameter(f"{name}__UNIT")
    return str(unit).strip() if unit else ""


def _format_value(value: Any, unit: str) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = round_float(float(value))
        if float(number).is_integer():
            number = int(number)
        return f"{number} {unit}".strip() if unit else number
    return f"{value} {unit}".strip() if unit else value


def _temperature_f(value: Any, unit: str) -> Optional[float]:
    from petrophys.model.units import temperature_to_fahrenheit

    try:
        return float(temperature_to_fahrenheit(float(value), unit or None))
    except (TypeError, ValueError):
        return None


def header_parameters(well: "Well", limit: int = 30) -> Dict[str, Any]:
    """Header parameters as ``"value unit"`` strings; temperatures also in degF."""
    from petrophys.model.units import temperature_unit

    parameters: Dict[str, Any] = {}
    for name, value in well._parameters.items():
        if name.startswith("_") or "__" in name or value is None or value == "":
            continue
        unit = _describe_unit(well, name)
        text = _format_value(value, unit)
        if temperature_unit(unit) in {"C", "K"}:
            fahrenheit = _temperature_f(value, unit)
            if fahrenheit is not None:
                text = f"{text} (= {round_float(fahrenheit):g} degF)"
        parameters[name] = text
        if len(parameters) >= limit:
            break
    return parameters


def _number(well: "Well", name: str) -> Optional[float]:
    value = well.get_parameter(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def header_notes(well: "Well") -> List[str]:
    """Warnings about header values an interpretation would use."""
    from petrophys.model.units import temperature_unit

    notes: List[str] = []
    rw = _number(well, "RW")
    rwt = _number(well, "RWT")
    rmf = _number(well, "RMF")
    bht = _number(well, "BHT")
    if rw is not None and rwt is None:
        notes.append(
            f"Header RW = {rw:g} {_describe_unit(well, 'RW') or 'ohm.m'} has NO measurement "
            "temperature (no RWT). Without an rw setting interpret_well uses it at an ASSUMED "
            "75 degF. Ask the user for Rw and its temperature; if you use the header value, "
            "report it as unverified."
        )
    elif rw is not None:
        rwt_f = _temperature_f(rwt, _describe_unit(well, "RWT"))
        notes.append(
            f"Header RW = {rw:g} ohm.m at RWT {rwt_f:g} degF; interpret_well uses it when no "
            f"rw setting is given (explicitly: rw={rw:g}, rw_meas_temp_f={rwt_f:g})."
        )
    if rw is not None and rmf is not None and rw > rmf:
        notes.append(
            f"Header RW ({rw:g}) is higher than RMF ({rmf:g}). Formation water is normally "
            "saltier (lower resistivity) than the mud filtrate: check RW for a decimal or unit "
            "error before using it."
        )
    if bht is None:
        notes.append(
            "No BHT in the header: formation temperature comes from a default gradient "
            "unless you pass bht_f (degF)."
        )
    else:
        unit = _describe_unit(well, "BHT")
        if temperature_unit(unit) in {"C", "K"}:
            notes.append(
                f"BHT is {bht:g} {unit} = {_temperature_f(bht, unit):g} degF. Settings such as "
                "bht_f and rw_meas_temp_f are in degF."
            )
    archie = {name: _number(well, name) for name in ("A", "M", "N")}
    if any(value is not None for value in archie.values()):
        given = ", ".join(f"{name.lower()}={value:g}" for name, value in archie.items()
                          if value is not None)
        notes.append(
            f"Header gives Archie parameters ({given}); interpret_well does not read them. "
            "Pass them as settings to use them."
        )
    return notes


def _curve_row(well: "Well", mnemonic: str, curve: Any, depth: np.ndarray) -> List[Any]:
    data = np.asarray(curve.data)
    try:
        finite = np.isfinite(data.astype(float))
    except (TypeError, ValueError):
        finite = np.array([value not in (None, "") for value in data])
    size = int(data.size)
    valid = int(np.count_nonzero(finite))
    top = bottom = None
    if valid and depth.size == size:
        valid_depth = depth[finite]
        top, bottom = float(np.nanmin(valid_depth)), float(np.nanmax(valid_depth))
    description = str(curve.metadata.description or "").strip()
    if description.replace("_", " ").upper() in {mnemonic.replace("_", " ").upper(), ""}:
        description = ""
    percent = round(100.0 * valid / size, 1) if size else 0.0
    return [mnemonic, curve.unit or "", percent, top, bottom, description]


def well_summary(well: "Well", well_id: str, source: str) -> Dict[str, Any]:
    """Compact summary of a loaded well for a model.

    Built on :meth:`Well.describe` plus header parameters with units,
    header notes, curve rows with their valid depth range, and the attached
    formations with depths.
    """
    described = well.describe()
    depth = np.asarray(well.depth, dtype=float) if well.depth is not None else np.array([])
    geology = well.get_parameter("_geology_context") or {}
    formations = [
        [row.get("formation_name"), row.get("top_depth"), row.get("bottom_depth"),
         row.get("rank") or ""]
        for row in geology.get("formations", [])
    ]
    summary: Dict[str, Any] = {
        "well_id": well_id,
        "source": source,
        "name": described["name"],
        "header": described["header"],
        "depth_unit": described["depth_unit"],
        "top": described["top"],
        "bottom": described["bottom"],
        "step": described["step"],
        "n_samples": described["n_samples"],
        "header_parameters": header_parameters(well),
        "header_notes": header_notes(well),
        "curve_columns": CURVE_COLUMNS,
        "curves": [_curve_row(well, name, curve, depth) for name, curve in well._curves.items()],
        "formation_columns": FORMATION_COLUMNS,
        "formations": formations,
        "lithology_intervals": len(geology.get("lithology", [])),
    }
    return compact_object(summary)


# -- interpretations ------------------------------------------------------------------

def _step(result: Any, name: str) -> Any:
    for record in getattr(result, "steps_executed", None) or getattr(result, "steps", []):
        if record.step == name and record.success:
            return record
    return None


_RW_SOURCES = {
    "config.rw": "given in settings",
    "las.RW": "LAS header RW",
    "salinity_ppm": "from the salinity setting",
    "rmf -> SP": "ESTIMATED from the SP log (no Rw given)",
    "rmf -> rxo_rt": "ESTIMATED from Rxo/Rt (no Rw given)",
    "Rwa clean-wet fallback": "ESTIMATED from the Rwa minimum (no Rw given)",
    "default": "DEFAULT 0.05 ohm.m (allow_default_rw)",
}


def _rw_assumption(result: Any) -> tuple[str, Optional[str]]:
    """(assumption text, review note or None) for the Rw a run used."""
    record = _step(result, "rw")
    if record is None:
        reason = getattr(result, "specific_steps_skipped", {}).get("rw", "")
        return (
            "Rw: none found" + (f" ({reason})" if reason else "")
            + "; water saturation and net pay were not computed",
            "No Rw was available, so SW and net pay are missing: ask the user for Rw "
            "(ohm.m) and its temperature (degF), then pass rw and rw_meas_temp_f.",
        )
    parameters = record.parameters or {}
    value = parameters.get("rw_at_measured_temp", parameters.get(
        "rw", parameters.get("rw_at_reference_temp")))
    temperature = parameters.get("measured_temp_f", parameters.get(
        "reference_temp_f", parameters.get("rw_reference_temp_f")))
    source = parameters.get("source_priority", "")
    label = _RW_SOURCES.get(source, source or record.method)
    assumed = "assumed" in (record.method or "")
    text = "Rw"
    if isinstance(value, (int, float)):
        text += f" {round_float(float(value)):g} ohm.m"
    if isinstance(temperature, (int, float)):
        text += f" at {round_float(float(temperature)):g} degF"
        if assumed:
            text += " (temperature ASSUMED, not given)"
    text += f", {label}"
    low, high = parameters.get("rw_min"), parameters.get("rw_max")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        text += (f"; {round_float(float(low)):g}-{round_float(float(high)):g} ohm.m at "
                 "formation temperature (Arps)")
    note = None
    if assumed:
        assumed_text = (f"{round_float(float(temperature)):g}"
                        if isinstance(temperature, (int, float)) else "75")
        note = (
            f"Rw ({label}) was used at an ASSUMED {assumed_text} degF "
            "because no temperature was given. SW and net pay depend strongly on it: confirm "
            "Rw and its temperature with the user and re-run with rw and rw_meas_temp_f."
        )
    elif "ESTIMATED" in label or "DEFAULT" in label:
        note = (f"Rw was {label.split(' (')[0].lower()}; confirm it with a water analysis or "
                "a regional value before relying on SW and net pay.")
    elif source == "las.RW":
        note = ("Rw came from the LAS header, not from the user; say so in the answer and "
                "confirm it if the result matters.")
    return text, note


def _assumptions(result: Any) -> tuple[List[str], List[str]]:
    """Plain-language assumptions of a prebuilt interpretation, and review notes."""
    items: List[str] = []
    notes: List[str] = []
    rw_text, rw_note = _rw_assumption(result)
    items.append(rw_text)
    if rw_note:
        notes.append(rw_note)

    rmf = _step(result, "rmf")
    if rmf is not None and "assumed" in (rmf.method or ""):
        parameters = rmf.parameters or {}
        value = parameters.get("rmf_at_reference_temp", parameters.get("rmf"))
        text = "Rmf"
        if isinstance(value, (int, float)):
            text += f" {round_float(float(value)):g} ohm.m"
        items.append(text + " from the LAS header at an ASSUMED 75 degF (affects SXO and "
                     "invasion only)")

    temperature = _step(result, "temperature")
    if temperature is not None:
        parameters = temperature.parameters or {}
        bht = parameters.get("bht_f")
        td = parameters.get("td_ft")
        unit = parameters.get("td_unit", "ft")
        text = f"Formation temperature: {temperature.method}"
        if isinstance(bht, (int, float)):
            text += f", BHT {round_float(float(bht)):g} degF"
            if isinstance(td, (int, float)):
                text += f" at {round_float(float(td)):g} {unit}"
        surface = parameters.get("surface_temp_f")
        if isinstance(surface, (int, float)):
            text += f", surface {surface:g} degF"
        items.append(text)

    config = getattr(result, "config", None)
    porosity = _step(result, "porosity")
    matrix = getattr(config, "matrix_type", None)
    if matrix is not None:
        text = f"Matrix: {matrix}"
        if porosity is not None:
            parameters = porosity.parameters or {}
            mode = parameters.get("matrix_mode")
            source = str(parameters.get("matrix_source") or "")
            if mode == "lithology_dependent":
                text += f" (endpoints vary with the lithology log: {source})"
            elif source.startswith("config_or_default"):
                density = parameters.get("rho_ma")
                text += " endpoints"
                if isinstance(density, (int, float)):
                    text += f" (rho_ma {density:g} g/cm3)"
                text += ", from matrix_type (given, or resolved from the logs when 'auto')"
            elif source:
                text += f" ({source})"
        items.append(text)
    domain = getattr(result, "lithology_domain", None)
    if domain:
        items.append(f"Lithology domain: {domain}")

    vclay = _step(result, "vclay")
    if vclay is not None:
        parameters = vclay.parameters or {}
        text = f"Clay volume: {vclay.method}"
        if parameters.get("gr_min") is not None:
            text += (f", GR clean {parameters.get('gr_min'):g} / shale "
                     f"{parameters.get('gr_max'):g} API")
        items.append(text)

    saturation = _step(result, "saturation")
    if saturation is not None:
        parameters = saturation.parameters or {}
        text = f"Saturation model: {saturation.method}"
        requested = parameters.get("sw_model_requested")
        if requested and requested != saturation.method:
            text += f" (requested {requested})"
        archie = [f"{key}={parameters[key]:g}" for key in ("a", "m", "n")
                  if isinstance(parameters.get(key), (int, float))]
        lithology_dependent = [key for key in ("a", "m", "n")
                               if isinstance(parameters.get(key), str)]
        if archie:
            text += ", " + ", ".join(archie)
        if lithology_dependent:
            text += ", " + "/".join(lithology_dependent) + " by lithology"
        items.append(text)

    net_pay = _step(result, "net_pay")
    if net_pay is not None:
        parameters = net_pay.parameters or {}
        cutoffs = [
            f"VCL < {parameters.get('vcl_cutoff')}",
            f"PHIE > {parameters.get('phie_cutoff')}",
            f"SW < {parameters.get('sw_cutoff')}",
        ]
        cutoff_source = parameters.get("cutoff_source")
        text = "Net-pay cutoffs: " + ", ".join(cutoffs)
        text += " (defaults)" if cutoff_source == "canonical" else " (from settings)"
        if parameters.get("lithology") == "per-depth":
            text += "; PHIE cutoff 0.045 where samples are carbonate"
        items.append(text)
    return items, notes


def _window_notes(well: "Well", top: Optional[float], bottom: Optional[float],
                  mask: np.ndarray) -> List[str]:
    """Warn when a summary window holds no samples, or spans most of a multi-zone well."""
    notes: List[str] = []
    unit = well.depth_unit
    if not np.any(mask):
        notes.append(
            f"The summary window {top}-{bottom} {unit} holds no samples; the well covers "
            f"{well.top}-{well.bottom} {unit}."
        )
        return notes
    depth = np.asarray(well.depth, dtype=float)
    window = depth[mask]
    span = float(np.nanmax(depth) - np.nanmin(depth)) or 1.0
    covered = float(window.max() - window.min()) / span
    if covered < 0.8:
        return notes
    zones = [
        (row.get("formation_name"), row.get("top_depth"), row.get("bottom_depth"))
        for row in (well.get_parameter("_geology_context") or {}).get("formations") or []
    ]
    if len(zones) == 1 and zones[0][1] is not None and zones[0][2] is not None:
        thickness = float(zones[0][2]) - float(zones[0][1])
        if thickness >= 0.8 * float(window.max() - window.min()):
            return notes  # the file covers one zone, e.g. a reservoir-only file
    text = (
        f"Results cover {'the whole' if covered > 0.99 else 'most of the'} logged interval "
        f"({round_float(float(window.min())):g}-{round_float(float(window.max())):g} {unit}), "
        "which can include non-reservoir rock such as overburden or shale sections; these "
        "totals are not a reservoir figure. Use summary_top/summary_bottom for the reservoir"
    )
    text += (", or summarize_zones(interpretation_id, rank='formation') for per-formation "
             "results." if zones else ".")
    notes.append(text)
    return notes


def pay_averages(well: "Well", top: Optional[float], bottom: Optional[float]) -> Dict[str, Any]:
    """Gross thickness and PHIE, SW, VCL and PERM averaged over net-pay samples."""
    from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

    if well.get("NET_PAY") is None or well.depth is None:
        return {}
    depth = np.asarray(well.depth, dtype=float)
    window_top = float(np.nanmin(depth)) if top is None else float(top)
    window_bottom = float(np.nanmax(depth)) if bottom is None else float(bottom)
    zone = calculate_zone_summary(well, window_top, window_bottom)
    return {
        "gross": zone.gross_thickness,
        "pay_samples": zone.n_pay_samples,
        "PHIE": zone.avg_phie,
        "SW": zone.avg_sw,
        "VCL": zone.avg_vcl,
        "PERM_geomean_mD": zone.avg_perm,
    }


def _thickness(net: Dict[str, Any], gross: Optional[float]) -> Dict[str, Any]:
    """Net thickness with unabbreviated keys, gross first."""
    shown: Dict[str, Any] = {}
    if gross is not None:
        shown["gross"] = gross
    names = {"net_rock": "net_rock", "net_res": "net_reservoir", "net_pay": "net_pay"}
    for key, label in names.items():
        if key in net:
            shown[label] = net[key]
    shown["depth_unit"] = net.get("depth_unit")
    return shown


def _settings_used(settings: Dict[str, Any]) -> Dict[str, Any]:
    shown = {}
    for key, value in settings.items():
        if key in {"formation_tops", "peer_wells", "lithology_intervals"} and value is not None:
            shown[key] = "attached" if key == "formation_tops" else "given"
        else:
            shown[key] = value
    return shown


def interpretation_summary(
    result: Any,
    interpretation_id: str,
    well_id: str,
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    captured_warnings: Sequence[str] = (),
    new_curves: Sequence[str] = (),
) -> Dict[str, Any]:
    """Compact summary of a prebuilt interpretation.

    :meth:`DynamicWorkflowResult.summary` plus ids, plain-language
    ``assumptions`` (Rw source and temperature, temperature profile, matrix,
    domain, models and cutoffs), ``pay_averages`` and ``review_notes``.
    """
    summary = result.summary(top=top, bottom=bottom)
    warnings = list(summary["warnings"])
    warnings.extend(message for message in captured_warnings if message not in warnings)
    well = result.well
    mask = depth_mask(well, top, bottom)
    assumptions, notes = _assumptions(result)
    notes = _window_notes(well, top, bottom, mask) + notes
    failed = [step["name"] for step in summary["executed_steps"] if not step["success"]]
    if failed:
        notes.append(f"Failed steps: {', '.join(failed)}; see explain_step for the error.")
    key_results = {}
    for name, stats in summary["key_results"].items():
        stats = dict(stats)
        stats.pop("samples", None)
        key_results[name] = stats
    averages = pay_averages(well, top, bottom)
    data = {
        "interpretation_id": interpretation_id,
        "well_id": well_id,
        "status": summary["status"],
        "summary_interval": summary["summary_interval"],
        "assumptions": assumptions,
        "review_notes": notes,
        "net_thickness": _thickness(summary["net_thickness"], averages.get("gross")),
        "pay_averages": {key: value for key, value in averages.items() if key != "gross"},
        "key_results": key_results,
        "key_results_basis": "statistics over every sample in the summary interval",
        "new_curves": list(new_curves),
        "resolved": summary["resolved"],
        "settings_used": _settings_used(summary["settings_used"]),
        "executed_steps": {step["name"]: step["method"] + ("" if step["success"] else " (FAILED)")
                           for step in summary["executed_steps"]},
        "skipped_steps": _skipped(summary["skipped_steps"]),
        "warnings": warnings,
        "errors": summary["errors"],
    }
    if data["pay_averages"]:
        data["pay_averages"]["basis"] = "averages over net-pay samples in the summary interval"
    return compact_object(data)


#: Steps pp.interpret never runs; their skip reasons are replaced by short pointers.
_NEVER_RUN: Dict[str, Optional[str]] = {
    "spectral_gamma": "not part of interpret_well; use the spectral_gamma tool",
    "elemental_spectroscopy": "not part of interpret_well",
    "porosity_refinement": None,
}


def _skipped(skipped: Dict[str, str]) -> Dict[str, str]:
    shown = {}
    for name, reason in skipped.items():
        if name in _NEVER_RUN and str(reason).startswith("Not run by the interpretation"):
            pointer = _NEVER_RUN[name]
            if pointer:
                shown[name] = pointer
            continue
        shown[name] = reason
    return shown


def workflow_summary(
    run: Any,
    interpretation_id: str,
    well_id: str,
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    captured_warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    """Compact summary of a custom workflow run (:class:`WorkflowRunResult`)."""
    summary = run.summary(top=top, bottom=bottom)
    warnings = list(summary["warnings"])
    warnings.extend(message for message in captured_warnings if message not in warnings)
    well = run.well
    mask = depth_mask(well, top, bottom)
    notes = _window_notes(well, top, bottom, mask)
    steps = []
    executed = {record.step: record for record in run.steps}
    for step in run.workflow.steps:
        record = executed.get(step.name)
        entry: Dict[str, Any] = {"name": step.name, "kind": step.kind, "params": step.params}
        if record is None:
            entry["status"] = "skipped: " + run.steps_skipped.get(step.name, "")
        else:
            entry["status"] = "ok" if record.success else f"FAILED: {record.message}"
            entry["method"] = record.method
            if step.kind in {"formation_water", "mud_filtrate", "temperature", "net_pay"}:
                entry["decisions"] = list(record.decisions)
        steps.append(entry)
    key_results = {}
    for name, stats in summary["key_results"].items():
        stats = dict(stats)
        stats.pop("samples", None)
        key_results[name] = stats
    averages = pay_averages(well, top, bottom)
    data: Dict[str, Any] = {
        "interpretation_id": interpretation_id,
        "well_id": well_id,
        "workflow": summary["workflow"],
        "status": summary["status"],
        "summary_interval": summary["summary_interval"],
        "steps": steps,
        "review_notes": notes,
        "net_thickness": _thickness(summary["net_thickness"], averages.get("gross")),
        "pay_averages": {key: value for key, value in averages.items() if key != "gross"},
        "key_results": key_results,
        "key_results_basis": "statistics over every sample in the summary interval",
        "output_curves": run.output_curves,
        "warnings": warnings,
        "errors": summary["errors"],
    }
    if data["pay_averages"]:
        data["pay_averages"]["basis"] = "averages over net-pay samples in the summary interval"
    return compact_object(data)


def _trim(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _trim(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [_trim(item) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"... {len(value) - _MAX_LIST_ITEMS} more")
        return items
    return value


#: Common names a model may use for a step of the prebuilt workflow.
STEP_ALIASES = {
    "vcl": "vclay", "clay_volume": "vclay", "vsh": "vclay", "shale_volume": "vclay",
    "sw": "saturation", "water_saturation": "saturation",
    "phi": "porosity", "phie": "porosity",
    "perm": "permeability", "netpay": "net_pay", "pay": "net_pay", "cutoffs": "net_pay",
    "temp": "temperature", "formation_water": "rw", "rmf_step": "rmf",
    "minerals": "lithology", "mineral_volumes": "lithology",
}


def step_detail(result: Any, step: str) -> Dict[str, Any]:
    """Full trace record of one executed or skipped step (compact, long lists trimmed)."""
    records = list(getattr(result, "steps_executed", None) or getattr(result, "steps", []))
    skipped = dict(getattr(result, "specific_steps_skipped", None)
                   or getattr(result, "steps_skipped", {}) or {})
    names = [record.step for record in records] + [name for name in skipped
                                                   if name not in {r.step for r in records}]
    wanted = step.strip()
    if wanted not in names:
        alias = STEP_ALIASES.get(wanted.lower().replace(" ", "_").replace("-", "_"))
        if alias is not None and alias in names:
            wanted = alias
    well = result.well
    for record in records:
        if record.step == wanted:
            data = {"status": "executed" if record.success else "failed", **record.to_dict(),
                    "depth_unit": well.depth_unit}
            if not data.get("data"):
                data.pop("data", None)
            if wanted == "net_pay" and well.depth_unit != "m" and any(
                " m)" in text for text in record.decisions
            ):
                data["note"] = (
                    f"Thicknesses in 'decisions' are labelled m but are in the well's depth "
                    f"unit ({well.depth_unit})."
                )
            return compact_object(_trim(data))
    if wanted in skipped:
        return {"status": "skipped", "step": wanted, "reason": skipped[wanted]}
    close = difflib.get_close_matches(wanted, names, n=2, cutoff=0.5)
    hint = f" Did you mean {', '.join(repr(name) for name in close)}?" if close else ""
    raise KeyError(
        f"Step {step!r} was not part of this run.{hint} Steps: {', '.join(names)}."
    )


# -- curve values ------------------------------------------------------------------

def decimated_values(
    well: "Well",
    curves: Sequence[str],
    mask: np.ndarray,
    max_points: int,
    max_values: Optional[int] = None,
) -> Dict[str, Any]:
    """Curve values at no more than ``max_points`` evenly spaced depths.

    With ``max_values``, the number of depths is reduced further so that
    depths times curves stays within it.
    """
    found = []
    missing = []
    for name in curves:
        curve = well.get(name)
        if curve is None:
            missing.append(name)
        else:
            found.append(curve)
    indices = np.flatnonzero(mask)
    available = int(indices.size)
    limit = int(max_points)
    note = ""
    if max_values is not None and found:
        budget = max(10, int(max_values) // (len(found) + 1))
        if budget < limit:
            note = (f"Capped at {budget} depths so the reply stays under {max_values} values "
                    f"({len(found)} curves + depth).")
            limit = budget
    if indices.size > limit:
        indices = indices[np.linspace(0, indices.size - 1, limit).round().astype(int)]
    values: Dict[str, List[Any]] = {}
    for curve in found:
        values[curve.mnemonic] = json_safe(np.asarray(curve.data, dtype=float)[indices])
    depth = np.asarray(well.depth, dtype=float)[indices]
    spacing = float(np.median(np.diff(depth))) if depth.size > 1 else None
    decimated = available > indices.size
    result: Dict[str, Any] = {
        "depth_unit": well.depth_unit,
        "depth": json_safe(depth),
        "curves": values,
        "missing_curves": missing,
        "points": int(indices.size),
        "samples_in_window": available,
        "decimated": decimated,
        "spacing": spacing,
    }
    if decimated:
        result["note"] = (
            (note + " " if note else "")
            + f"Every {available / max(indices.size, 1):.0f}th sample of {available}: thin beds "
            "can be missed. Narrow top/bottom for detail, or use curve_statistics."
        )
    if missing:
        result["available_curves"] = list(well.curves)
    return compact_object(result)
