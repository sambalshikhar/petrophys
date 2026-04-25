"""
Dynamic petrophysics workflow.

This module provides a DynamicWorkflow class that automatically detects
available input curves and runs the appropriate modules in the correct
order. It supports both granular (single-module) and full pipeline
execution modes.

Pipeline execution order (from design doc):
    Module 1: QC & Pre-Processing
    Module 2: Volume of Clay (VCLAY) & Clay Resistivity
    Module 3: Lithology Determination
    Module 4: Fluid Resistivity (Rmf, Rw)
    Module 5: HC Correction
    Module 6: Porosity
    Module 7: Water Saturation
    Module 8: Permeability
    Module 9: Rock Quality (RQI, FZI)
    Module 10: Net Pay
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import re
from petrophys.base.well import Well
import pandas as pd
from petrophys.io import ascii_reader, las_reader

class WorkflowStep(Enum):
    """Individual workflow steps."""
    NORMALIZATION = "normalization"
    BAD_HOLE = "bad_hole"
    TEMPERATURE = "temperature"
    VCLAY = "vclay"
    RCL = "rcl"
    LITHOLOGY = "lithology"
    RMF = "rmf"
    RW = "rw"
    INVASION = "invasion"
    HC_CORRECTION = "hc_correction"
    POROSITY = "porosity"
    SATURATION = "saturation"
    PERMEABILITY = "permeability"
    ROCK_QUALITY = "rock_quality"
    BVW = "bvw"
    MHI = "mhi"
    NET_PAY = "net_pay"


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric well parameter if present."""
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_text(well: "Well", name: str) -> str:
    """Fetch a text well parameter if present."""
    value = well.get_parameter(name)
    return str(value or "").strip()


def _temperature_to_f(value: float, unit: str) -> float:
    """Convert a temperature value to Fahrenheit when units indicate Celsius."""
    text = (unit or "").strip().lower()
    if "c" in text and "f" not in text:
        return (float(value) * 9.0 / 5.0) + 32.0
    return float(value)


def _constant_zoned_parameter(well: "Well", prefix: str) -> Optional[float]:
    """
    Return a unique zoned LAS parameter value, e.g. RW_Z1/RW_Z2 -> RW.

    If multiple distinct zoned values exist, return None so callers do not
    silently choose the wrong interval constant.
    """
    values = set()
    pattern = re.compile(rf"^{re.escape(prefix.upper())}_Z\d+$")

    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        if not pattern.match(key.upper()):
            continue
        try:
            values.add(float(value))
        except (TypeError, ValueError):
            continue

    if len(values) == 1:
        return next(iter(values))
    return None


def _parameter_temperature_f(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric temperature parameter and normalize it to Fahrenheit."""
    value = _parameter_float(well, name)
    unit = _parameter_text(well, f"{name}__UNIT")
    if value is not None:
        return _temperature_to_f(value, unit)

    value = _constant_zoned_parameter(well, name)
    unit = _parameter_text(well, f"{name}_Z1__UNIT")
    if value is not None:
        return _temperature_to_f(value, unit)

    return None


def _metadata_bht_f(well: "Well") -> Optional[float]:
    """Return BHT metadata in Fahrenheit."""
    if well.header.bht is not None:
        unit = _parameter_text(well, "BHT__UNIT") or getattr(well.header, "bht_unit", "F")
        return _temperature_to_f(float(well.header.bht), unit)

    value = _parameter_float(well, "BHT")
    unit = _parameter_text(well, "BHT__UNIT").lower()
    if value is not None:
        return _temperature_to_f(value, unit)

    value = _constant_zoned_parameter(well, "BHT")
    unit = _parameter_text(well, "BHT_Z1__UNIT").lower()
    if value is not None:
        return _temperature_to_f(value, unit)

    return None


GROUND_TRUTH_CODE_TO_GROUP = {
    1: "sandstone",
    2: "sandstone",
    3: "sandstone",
    4: "sandstone",
    5: "silt",
    6: "silt",
    7: "shale",
    8: "shale",
    9: "limestone",
    10: "sandstone",
    11: "limestone",
    12: "limestone",
    13: "limestone",
    14: "calcareous_shale",
    15: "dolomite",
    16: "limestone",
    17: "clastic_coarse",
    18: "coal",
    19: "igneous",
    20: "igneous",
    21: "halite",
    22: "anhydrite",
    23: "igneous",
    24: "igneous",
    25: "shale",
    26: "evaporite",
    27: "halite",
    28: "igneous",
    29: "chert",
    30: "sandstone",
    31: "clastic_coarse",
    32: "halite",
    33: "halite",
    34: "halite",
    35: "sandstone",
    36: "clastic_coarse",
}

COMPUTED_LITHO_CODE_TO_GROUP = {
    1: "sandstone",
    2: "limestone",
    3: "dolomite",
    4: "shale",
    5: "anhydrite",
    6: "halite",
    7: "coal",
}

LITHOLOGY_GROUP_TO_MATRIX = {
    "sandstone": "sandstone",
    "silt": "sandstone",
    "clastic_coarse": "sandstone",
    "igneous": "quartz",
    "chert": "quartz",
    "limestone": "limestone",
    "dolomite": "dolomite",
    "shale": "shale",
    "calcareous_shale": "shale",
    "anhydrite": "anhydrite",
    "evaporite": "gypsum",
    "halite": "halite",
    "coal": "coal",
}


def _representative_finite_value(values: Any) -> Optional[float]:
    """Return the modal finite value, falling back to median."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size:
        return float(unique[np.argmax(counts)])
    return float(np.nanmedian(finite))


def _coerce_lithology_group(value: Any) -> Optional[str]:
    """Normalize lithology strings to workflow group names."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "-999.25"}:
        return None

    aliases = {
        "sand": "sandstone",
        "ss": "sandstone",
        "lime": "limestone",
        "ls": "limestone",
        "dolo": "dolomite",
        "salt": "halite",
    }
    return aliases.get(text, text)


def _group_array_from_numeric_codes(data: np.ndarray, mapping: Dict[int, str]) -> np.ndarray:
    groups = np.full(data.shape, None, dtype=object)
    finite = np.isfinite(data)
    if not np.any(finite):
        return groups

    rounded = np.rint(data[finite]).astype(int)
    grouped = [mapping.get(code) for code in rounded]
    groups[finite] = grouped
    return groups


def _is_shale_group(name: Any) -> bool:
    text = str(name or "").strip().lower()
    return "shale" in text


def _format_group_list(groups: List[str]) -> str:
    if not groups:
        return ""
    return ", ".join(sorted({str(group).upper() for group in groups if group}))


def _ordered_unique_values(values: List[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _trace_array_summary(value: np.ndarray) -> Dict[str, Any]:
    """Summarize array values without writing full curve payloads to logs."""
    data = np.asarray(value)
    payload: Dict[str, Any] = {
        "type": "ndarray",
        "shape": list(data.shape),
        "size": int(data.size),
    }
    if data.ndim == 0:
        payload["value"] = _json_safe_trace_value(data.item())
        return payload

    sample_count = min(5, int(data.size))
    payload["sample"] = [
        _json_safe_trace_value(item) for item in list(data.flat[:sample_count])
    ]

    if np.issubdtype(data.dtype, np.number):
        numeric = data.astype(float, copy=False)
        finite = numeric[np.isfinite(numeric)]
        payload.update({
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(numeric).sum()),
            "inf_count": int(np.isinf(numeric).sum()),
        })
        if finite.size:
            payload.update({
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "p05": float(np.nanpercentile(finite, 5)),
                "p50": float(np.nanpercentile(finite, 50)),
                "p95": float(np.nanpercentile(finite, 95)),
            })

    return payload


def _json_safe_trace_value(value: Any) -> Any:
    """Convert trace metadata to JSON-safe, compact values."""
    if isinstance(value, np.ndarray):
        return _trace_array_summary(value)
    if isinstance(value, np.generic):
        return _json_safe_trace_value(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_trace_value(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _curve_summaries_for_trace(
    well: "Well",
    curve_names: Optional[List[str]],
) -> Dict[str, Any]:
    """Build compact per-curve summaries for workflow trace exports."""
    if not curve_names:
        return {}

    summaries: Dict[str, Any] = {}
    for name in curve_names:
        curve = well.get(name) if hasattr(well, "get") else None
        if curve is None:
            continue
        summaries[str(name).upper()] = {
            "unit": getattr(curve, "unit", ""),
            "description": getattr(getattr(curve, "metadata", None), "description", ""),
            "data": _trace_array_summary(np.asarray(curve.data)),
        }
    return summaries


@dataclass
class DynamicWorkflowConfig:
    """
    Configuration for the dynamic workflow.

    All parameters are optional. The workflow auto-detects what can
    be computed based on available curves and provided parameters.
    """
    # Temperature
    surface_temp_f: float = 68.0
    bht_f: Optional[float] = None
    td_ft: Optional[float] = None

    # Mud properties
    mud_type: str = "WBM"  # WBM or OBM
    mud_has_clay_or_kcl: bool = False
    rmf_at_meas_temp: Optional[float] = None
    rmf_meas_temp_f: float = 75.0
    rmf_salinity_ppm: Optional[float] = None
    rm: Optional[float] = None
    rm_meas_temp_f: float = 75.0
    mud_water: str = "fresh"  # fresh or salt

    # Formation water
    rw: Optional[float] = None
    rw_meas_temp_f: Optional[float] = None
    salinity_ppm: Optional[float] = None

    # Matrix properties
    matrix_type: str = "sandstone"
    rho_ma: Optional[float] = None
    nphi_ma: Optional[float] = None
    rho_fl: float = 1.0
    rho_cl: float = 2.45
    dt_ma: Optional[float] = None
    dt_fl: float = 189.0

    # Lithology
    lithology: str = "clastic"  # clastic or carbonate
    lithology_csv: Optional[str] = None  # Path to lithology CSV (top, bottom, lithology)

    # Archie parameters
    a: Optional[float] = None
    m: Optional[float] = None
    n: Optional[float] = None

    # Waxman-Smits parameters
    cec: Optional[float] = None
    b_coeff: Optional[float] = None
    m_star: float = 2.0

    # Dual Water parameters
    rwb: Optional[float] = None
    phi_cl: float = 0.15

    # HC correction
    apply_hc_correction: bool = False
    hc_type: str = "oil"  # oil or gas

    # Cutoffs
    vcl_cutoff: float = 0.3
    sw_cutoff: float = 0.7
    phie_cutoff: Optional[float] = None

    # GR parameters (auto-detected if None)
    gr_min: Optional[float] = None
    gr_max: Optional[float] = None
    gr_method: str = "linear"

    # SP parameters (auto-detected if None)
    sp_min: Optional[float] = None
    sp_max: Optional[float] = None

    # Permeability
    perm_method: str = "auto"
    perm_fallback_method: str = "coates"
    swirr: Optional[float] = None
    peer_wells: Optional[List[Any]] = None
    formation_tops: Optional[Any] = None

    # Saturation model (auto = use routing logic)
    sw_model: str = "auto"
    rtsh: Optional[float] = None
    vsh_exp: float = 1.6
    shale_exp: float = 1.6

    # Control which steps to skip
    skip_steps: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """
    Result of a single workflow step with full trace metadata.

    Attributes
    ----------
    step : str
        Step name (e.g., "vclay", "porosity").
    success : bool
        Whether the step completed without error.
    message : str
        Error message if failed, or summary if successful.
    method : str
        Algorithm/equation used (e.g., "larionov_tertiary", "simandoux").
    input_curves : list
        Log curves consumed by this step.
    output_curves : list
        Curves added/modified on the well.
    parameters : dict
        All numeric parameters, thresholds, and constants used.
    decisions : list
        Human-readable trace of decisions/logic branches taken.
    """
    step: str
    success: bool
    message: str = ""
    method: str = ""
    input_curves: List[str] = field(default_factory=list)
    output_curves: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation of this workflow step trace."""
        return {
            "step": self.step,
            "success": bool(self.success),
            "method": self.method,
            "input_curves": list(self.input_curves),
            "output_curves": list(self.output_curves),
            "parameters": _json_safe_trace_value(self.parameters),
            "decisions": list(self.decisions),
            "message": self.message,
            "data": _json_safe_trace_value(self.data or {}),
        }


@dataclass
class DynamicWorkflowResult:
    """Results from dynamic workflow execution."""
    well: "Well"
    config: DynamicWorkflowConfig
    steps_executed: List[StepResult] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    specific_steps_skipped: dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Computed values
    rw_computed: Optional[Union[float, np.ndarray]] = None
    rmf_computed: Optional[Union[float, np.ndarray]] = None
    rcl_computed: Optional[float] = None
    temperature_gradient: Optional[float] = None

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def print_trace(self) -> None:
        """Print a comprehensive human-readable trace of all steps."""
        print(f"\n{'=' * 90}")
        print(f"WORKFLOW TRACE — {self.well.name or 'Unknown Well'}")
        print(f"{'=' * 90}")
        print(f"Overall: {'SUCCESS' if self.success else 'FAILED'} | "
              f"Executed: {len(self.steps_executed)} | Skipped: {len(self.steps_skipped)}")
        if self.rw_computed is not None:
            def _summary(value):
                arr = np.asarray(value, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return "NaN"
                if finite.size == 1:
                    return f"{finite[0]:.4f}"
                return f"{np.nanmin(finite):.4f}-{np.nanmax(finite):.4f}"
            print(f"Rw = {_summary(self.rw_computed)} ohm.m | Rmf = {_summary(self.rmf_computed) if self.rmf_computed is not None else 'None'} | "
                  f"Rcl = {self.rcl_computed}")
        print()

        for sr in self.steps_executed:
            status = "OK" if sr.success else "FAIL"
            print(f"[{status:4s}] {sr.step}")
            if sr.method:
                print(f"       Method: {sr.method}")
            if sr.input_curves:
                print(f"       Inputs: {', '.join(sr.input_curves)}")
            if sr.output_curves:
                print(f"       Outputs: {', '.join(sr.output_curves)}")
            if sr.parameters:
                for k, v in sr.parameters.items():
                    if isinstance(v, float):
                        print(f"       {k}: {v:.6g}")
                    else:
                        print(f"       {k}: {v}")
            for d in sr.decisions:
                print(f"       -> {d}")
            if sr.message and not sr.success:
                print(f"       ERROR: {sr.message}")
            print()

        if self.steps_skipped:
            print(f"Skipped: {', '.join(self.steps_skipped)}")
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"{'=' * 90}")

    def to_dataframe(self):
        """Export step trace to a pandas DataFrame."""
        import pandas as pd
        rows = []
        for sr in self.steps_executed:
            rows.append({
                "Step": sr.step,
                "Status": "OK" if sr.success else "FAIL",
                "Method": sr.method,
                "Input Curves": ", ".join(sr.input_curves),
                "Output Curves": ", ".join(sr.output_curves),
                "Parameters": str(sr.parameters) if sr.parameters else "",
                "Decisions": " | ".join(sr.decisions),
                "Message": sr.message,
            })
        return pd.DataFrame(rows)

    def to_trace_dict(
        self,
        include_curves: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export the full workflow route as a JSON-safe dictionary.

        The trace is recorded at the workflow module boundary, so each step
        shows why that module ran, which method was selected, which curves were
        consumed, which curves were written, and the parameters used.
        """
        try:
            well_name = self.well.name
        except Exception:
            header = getattr(self.well, "header", None)
            well_name = getattr(header, "well_name", "") if header is not None else ""
            if not well_name and isinstance(header, dict):
                well_name = header.get("WELL") or header.get("well_name") or ""

        well_summary = {
            "name": well_name or "Unknown Well",
            "depth_unit": getattr(self.well, "depth_unit", ""),
            "top": _json_safe_trace_value(getattr(self.well, "top", None)),
            "bottom": _json_safe_trace_value(getattr(self.well, "bottom", None)),
            "step": _json_safe_trace_value(getattr(self.well, "step", None)),
            "n_samples": int(getattr(self.well, "n_samples", 0) or 0),
            "curve_count": int(getattr(self.well, "n_curves", 0) or 0),
            "available_curves": sorted(str(curve) for curve in getattr(self.well, "curves", [])),
        }

        return {
            "schema": "dynamic_workflow_trace",
            "schema_version": 1,
            "well": well_summary,
            "success": bool(self.success),
            "executed_steps": [step.step for step in self.steps_executed],
            "skipped_steps": list(self.steps_skipped),
            "specific_steps_skipped": _json_safe_trace_value(self.specific_steps_skipped),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "computed": {
                "rw": _json_safe_trace_value(self.rw_computed),
                "rmf": _json_safe_trace_value(self.rmf_computed),
                "rcl": _json_safe_trace_value(self.rcl_computed),
                "temperature_gradient": _json_safe_trace_value(self.temperature_gradient),
            },
            "config": _json_safe_trace_value(vars(self.config)),
            "steps": [step.to_dict() for step in self.steps_executed],
            "curve_summaries": _curve_summaries_for_trace(self.well, include_curves),
        }

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
        indent: int = 2,
    ) -> Union[str, Path]:
        """
        Export the workflow route as JSON.

        If ``path`` is provided, the JSON file is written and the output path is
        returned. Otherwise the JSON string is returned.
        """
        text = json.dumps(
            self.to_trace_dict(include_curves=include_curves),
            indent=indent,
            sort_keys=False,
            allow_nan=False,
        )
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        return output_path

    def to_markdown(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
    ) -> Union[str, Path]:
        """
        Export the workflow route as a human-readable Markdown report.

        This mirrors ``to_json`` but is easier to scan when checking per-module
        routing decisions for a single well.
        """
        trace = self.to_trace_dict(include_curves=include_curves)
        well = trace["well"]
        lines = [
            "# Dynamic Workflow Trace",
            "",
            f"- Well: `{well['name']}`",
            f"- Depth: `{well['top']} - {well['bottom']} {well['depth_unit']}`",
            f"- Samples: `{well['n_samples']}`",
            f"- Overall success: `{trace['success']}`",
            f"- Executed steps: `{', '.join(trace['executed_steps']) or 'none'}`",
            f"- Skipped steps: `{', '.join(trace['skipped_steps']) or 'none'}`",
        ]

        if trace["warnings"]:
            lines.append(f"- Warnings: `{' ; '.join(trace['warnings'])}`")
        if trace["errors"]:
            lines.append(f"- Errors: `{' ; '.join(trace['errors'])}`")

        computed = trace["computed"]
        lines.extend([
            "",
            "## Computed Values",
            f"- Rw: `{json.dumps(computed['rw'])}`",
            f"- Rmf: `{json.dumps(computed['rmf'])}`",
            f"- Rcl: `{json.dumps(computed['rcl'])}`",
            f"- Temperature gradient: `{json.dumps(computed['temperature_gradient'])}`",
        ])

        for step in trace["steps"]:
            lines.extend([
                "",
                f"## {step['step']}",
                f"- Success: `{step['success']}`",
                f"- Method: `{step['method'] or 'none'}`",
                f"- Inputs: `{', '.join(step['input_curves']) or 'none'}`",
                f"- Outputs: `{', '.join(step['output_curves']) or 'none'}`",
                f"- Parameters: `{json.dumps(step['parameters'], sort_keys=True)}`",
            ])
            if step["decisions"]:
                lines.append("- Decisions:")
                lines.extend(f"  - {decision}" for decision in step["decisions"])
            if step["message"]:
                lines.append(f"- Message: `{step['message']}`")
            if step["data"]:
                lines.append(f"- Data: `{json.dumps(step['data'], sort_keys=True)}`")

        if trace["curve_summaries"]:
            lines.extend(["", "## Curve Summaries"])
            for name, summary in trace["curve_summaries"].items():
                lines.append(f"- {name}: `{json.dumps(summary, sort_keys=True)}`")

        text = "\n".join(lines) + "\n"
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path


class DynamicWorkflow:
    """
    Dynamic petrophysics workflow that auto-detects available inputs.

    This workflow inspects the well's available curves and the provided
    configuration to determine which processing steps can be executed.
    It runs modules in the correct pipeline order and passes results
    between dependent steps.

    Examples
    --------
    >>> config = DynamicWorkflowConfig(
    ...     bht_f=220, td_ft=8500, rw=0.05,
    ...     lithology='clastic', salinity_ppm=25000
    ... )
    >>> workflow = DynamicWorkflow(config)
    >>> result = workflow.run(well)
    >>> print(result.steps_executed)
    """

    def __init__(self, config: Optional[DynamicWorkflowConfig] = None):
        self.config = config or DynamicWorkflowConfig()
        self._result: Optional[DynamicWorkflowResult] = None
        self._lithology_log: Optional[np.ndarray] = None
        self._lithology_source: str = "config"

    def get_pipeline(self):
        return [
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.SATURATION, self._step_saturation, self._can_saturation),
            (WorkflowStep.PERMEABILITY, self._step_permeability, self._can_permeability),
            (WorkflowStep.ROCK_QUALITY, self._step_rock_quality, self._can_rock_quality),
            (WorkflowStep.BVW, self._step_bvw, self._can_bvw),
            (WorkflowStep.MHI, self._step_mhi, self._can_mhi),
            (WorkflowStep.NET_PAY, self._step_net_pay, self._can_net_pay),
        ]

    def run_selected_steps(self, well: "Well", steps: list[WorkflowStep]) -> DynamicWorkflowResult:
        """
        Execute only the specified workflow steps, in pipeline order.
        """
        self._result = DynamicWorkflowResult(well=well, config=self.config)

        allowed = set(steps)

        for step_enum, step_fn, _ in self.get_pipeline():
            print(f"step_enum: {step_enum}")
            if step_enum not in allowed:
                continue

            try:
                step_fn(well)
                self._result.steps_executed.append(step_enum.name)
            except Exception as e:
                self._result.specific_steps_skipped[step_enum.name] = str(e)


        return self._result

    def run(self, well: "Well") -> DynamicWorkflowResult:
        """
        Run the dynamic workflow on a well.

        Automatically detects available curves and executes
        applicable processing steps in pipeline order.

        Parameters
        ----------
        well : Well
            Well object with raw log data

        Returns
        -------
        DynamicWorkflowResult
        """
        self._result = DynamicWorkflowResult(well=well, config=self.config)

        # Detect available curves
        available = self._detect_available_curves(well)

        # Execute pipeline in order
        pipeline = [
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.SATURATION, self._step_saturation, self._can_saturation),
            (WorkflowStep.PERMEABILITY, self._step_permeability, self._can_permeability),
            (WorkflowStep.ROCK_QUALITY, self._step_rock_quality, self._can_rock_quality),
            (WorkflowStep.BVW, self._step_bvw, self._can_bvw),
            (WorkflowStep.MHI, self._step_mhi, self._can_mhi),
            (WorkflowStep.NET_PAY, self._step_net_pay, self._can_net_pay),
        ]

        for step_enum, step_fn, can_fn in pipeline:
            step_name = step_enum.value

            if step_name in self.config.skip_steps:
                self._result.steps_skipped.append(step_name)
                continue

            if can_fn(well):
                try:
                    step_trace = step_fn(well)
                    if step_trace is None:
                        step_trace = StepResult(step=step_name, success=True)
                    step_trace.step = step_name
                    step_trace.success = True
                    self._result.steps_executed.append(step_trace)
                except Exception as e:
                    msg = f"{step_name}: {str(e)}"
                    self._result.warnings.append(msg)
                    self._result.steps_executed.append(
                        StepResult(step=step_name, success=False, message=str(e))
                    )
            else:
                self._result.steps_skipped.append(step_name)

        return self._result

    def _detect_available_curves(self, well: "Well") -> set:
        """Detect which curves are available in the well."""
        available = set()
        for name in ["GR", "SP", "NPHI", "RHOB", "DTC", "DTS", "RDEP", "RT",
                      "RESS", "RESM", "RSHA", "RXO", "PEF", "CALI", "CAL", "BS", "DRHO",
                      "VCLAY", "PHIE", "PHIT", "SWT-LOG", "SXO-LOG", "PERM", "TEMP"]:
            if well.get(name) is not None:
                available.add(name)
        return available

    def _has_curve(self, well: "Well", *names: str) -> bool:
        """Check if any of the named curves exist."""
        for n in names:
            if hasattr(well, "get") and well.get(n) is not None:
                return True
            if hasattr(well, "__getitem__"):
                try:
                    if well[n] is not None:
                        return True
                except (KeyError, IndexError, TypeError):
                    pass
            if hasattr(well, "has_curve") and well.has_curve(n):
                return True
            if hasattr(well, "curves") and n in well.curves:
                return True
        return False

    # ---- CAN-RUN checks ----

    def _can_temperature(self, well: "Well") -> bool:
        return (
            well.depth is not None
            or self._has_curve(well, "DEPT", "DEPTH")
        )

    def _can_bad_hole(self, well: "Well") -> bool:
        return self._has_curve(well, "CALI", "CAL") or self._has_curve(well, "DRHO")

    def _can_vclay(self, well: "Well") -> bool:
        return self._has_curve(well, "GR")

    def _can_rcl(self, well: "Well") -> bool:
        return self._has_curve(well, "RDEP", "RT") and self._has_curve(well, "VCL", "VCLAY")

    def _can_lithology(self, well: "Well") -> bool:
        return (self._has_curve(well, "RHOB") and
                self._has_curve(well, "NPHI") and
                self._has_curve(well, "PEF", "DTC") and
                self._has_curve(well, "VCL"))

    def _can_rmf(self, well: "Well") -> bool:
        return (
            str(self.config.mud_type).upper() == "OBM"
            or self.config.rmf_at_meas_temp is not None
            or self.config.rmf_salinity_ppm is not None
            or self.config.rm is not None
            or well.header.rmf is not None
            or well.header.rm is not None
            or _parameter_float(well, "RMF") is not None
            or _constant_zoned_parameter(well, "RMF") is not None
            or _parameter_float(well, "RM") is not None
            or _constant_zoned_parameter(well, "RM") is not None
        )

    def _can_rw(self, well: "Well") -> bool:
        # Always run the Rw step so the workflow can fall back through
        # metadata, salinity, log-based heuristics, and finally the built-in
        # default when no explicit source is available.
        return True

    def _can_invasion(self, well: "Well") -> bool:
        return (self._has_curve(well, "RSHA", "RESS", "RXO") and
                self._has_curve(well, "RDEP", "RT"))

    def _can_hc_correction(self, well: "Well") -> bool:
        return (
            self.config.apply_hc_correction
            and self._has_curve(well, "NPHI")
            and self._has_curve(well, "RHOB")
        )

    def _can_porosity(self, well: "Well") -> bool:
        return self._has_curve(well, "RHOB", "NPHI", "DTC", "PHIN", "PHIT", "PHID", "PHIS")

    def _can_saturation(self, well: "Well") -> bool:
        return (self._has_curve(well, "RDEP", "RT") and
                self._has_curve(well, "PHIE") and
                self._result.rw_computed is not None)

    def _can_permeability(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") or self._has_curve(well, "PHIT")

    def _can_rock_quality(self, well: "Well") -> bool:
        return self._has_curve(well, "PERM") and self._has_curve(well, "PHIE")

    def _can_bvw(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    def _can_mhi(self, well: "Well") -> bool:
        return self._has_curve(well, "SW", "SWA", "SWT-LOG") and self._has_sxo_curve(well)

    def _can_net_pay(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> StepResult:
        from petrophys.environmental.temperature import (
            calculate_temperature_profile,
            gradient_to_celsius_per_km,
        )

        trace = StepResult(step="temperature", success=True, method="temperature_profile")
        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f
        td_ft = self.config.td_ft
        td_unit = "ft" if td_ft is not None else None

        if bht_f is None:
            bht_f = _metadata_bht_f(well)

        if td_ft is None and well.header.td is not None:
            td_ft = float(well.header.td)
            td_unit = well.depth_unit
        if td_ft is None and well.bottom is not None:
            td_ft = float(well.bottom)
            td_unit = well.depth_unit

        trace.parameters = {
            "surface_temp_f": surface_temp_f,
            "bht_f": bht_f,
            "td_ft": td_ft,
            "td_unit": td_unit,
        }

        try:
            result = calculate_temperature_profile(
                well,
                bht=bht_f,
                surface_temp=surface_temp_f,
                bht_unit="F",
                surface_temp_unit="F",
                td=td_ft,
                td_unit=td_unit,
                depth_unit=well.depth_unit,
            )
        except Exception as exc:
            trace.success = False
            trace.message = str(exc)
            self._result.warnings.append(f"Temperature profile failed: {exc}")
            return trace

        well.add_curve("TEMP", result.temperature, "degF", "Formation Temperature")
        well.set_parameter("temperature_gradient", result.gradient)
        well.set_parameter("surface_temperature", result.surface_temp)
        well.set_parameter("temperature_profile_source", result.source)
        if result.reference_temperature is not None:
            well.set_parameter("temperature_reference_temperature", result.reference_temperature)
        if result.reference_depth is not None:
            well.set_parameter("temperature_reference_depth", result.reference_depth)

        self._result.temperature_gradient = result.gradient
        trace.method = result.source
        trace.output_curves = ["TEMP"]
        trace.parameters["gradient_f_per_100ft"] = result.gradient
        trace.parameters["gradient_c_per_km"] = gradient_to_celsius_per_km(result.gradient)
        if result.reference_temperature is not None:
            trace.parameters["reference_temperature_f"] = result.reference_temperature
        if result.reference_depth is not None:
            trace.parameters["reference_depth_ft"] = result.reference_depth

        if result.source == "fallback_geothermal_gradient":
            trace.decisions.append("No BHT temperature anchor found -> using fallback 30 C/km geothermal gradient")
        else:
            trace.decisions.append("Temperature profile anchored to BHT metadata/input")
        trace.decisions.append(
            f"Gradient = {result.gradient:.4f} F/100ft ({trace.parameters['gradient_c_per_km']:.2f} C/km)"
        )
        trace.decisions.append(
            f"Temp range: {np.nanmin(result.temperature):.1f} - {np.nanmax(result.temperature):.1f} F"
        )
        return trace

    def _step_bad_hole(self, well: "Well") -> StepResult:
        from petrophys.qaqc.bad_hole.flag_generator import generate_bad_hole_flags

        trace = StepResult(step="bad_hole", success=True, method="caliper_drho_flags")
        inputs = []

        cali = well.get("CALI") or well.get("CAL")
        bs_curve = well.get("BS")
        drho = well.get("DRHO")
        rhob = well.get("RHOB")
        pef = well.get("PEF")

        if bs_curve is not None:
            bs = np.asarray(bs_curve.data, dtype=float)
            bit_size = bs
            representative_bit_size = _representative_finite_value(bs)
            bit_size_source = "BS_curve_depth_by_depth"
            finite_bs = bs[np.isfinite(bs)]
            bit_size_unique_values = sorted({float(value) for value in np.round(finite_bs, 4)}) if finite_bs.size else []
        else:
            header_bit_size = getattr(well.header, "bit_size", None)
            representative_bit_size = float(header_bit_size) if header_bit_size is not None else 8.5
            bit_size = representative_bit_size
            bit_size_source = "well_header" if header_bit_size is not None else "fallback_8.5"
            bit_size_unique_values = [float(representative_bit_size)]
        if cali is not None: inputs.append(cali.mnemonic)
        if bs_curve is not None: inputs.append("BS")
        if drho is not None: inputs.append("DRHO")
        if rhob is not None: inputs.append("RHOB")
        if pef is not None: inputs.append("PEF")
        trace.input_curves = inputs

        flags = generate_bad_hole_flags(
            well,
            bit_size=bit_size
        )
        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")
        well.add_curve("BADHOLEFLAG", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

        n_bad = int(np.sum(flags["bad_hole"]))
        trace.output_curves = ["BAD_HOLE", "BADHOLEFLAG"]
        trace.parameters["bit_size_used"] = representative_bit_size
        trace.parameters["bit_size_source"] = bit_size_source
        trace.parameters["bit_size_unique_values"] = bit_size_unique_values
        trace.parameters["bit_size_depth_dependent"] = bs_curve is not None and len(bit_size_unique_values) > 1
        trace.decisions.append(f"{n_bad} bad-hole samples flagged out of {len(flags['bad_hole'])}")
        return trace

    def _get_temperature_profile(self, well: "Well") -> np.ndarray:
        """Return a formation-temperature array in Fahrenheit."""
        temp_curve = well.get("TEMP")
        if temp_curve is not None:
            temp = np.asarray(temp_curve.data, dtype=float)
            if temp.size == well.n_samples:
                return temp

        if well.depth is not None:
            from petrophys.environmental.temperature import calculate_temperature_profile

            try:
                surface_temp_f = (
                    self.config.surface_temp_f
                    if self.config.surface_temp_f is not None
                    else 68.0
                )
                result = calculate_temperature_profile(
                    well,
                    bht=self.config.bht_f or _metadata_bht_f(well),
                    surface_temp=surface_temp_f,
                    bht_unit="F",
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return np.asarray(result.temperature, dtype=float)
            except Exception:
                pass

        avg_temp = self._get_avg_formation_temp(well)
        return np.full(well.n_samples, avg_temp, dtype=float)

    def _reference_temperature(self, temp_profile: np.ndarray) -> float:
        """Choose a reference temperature for scalar resistivity inputs."""
        finite = temp_profile[np.isfinite(temp_profile)]
        if finite.size == 0:
            return 150.0
        return float(np.nanmedian(finite))

    def _step_vclay(self, well: "Well") -> StepResult:
        from petrophys.clay_volume.vcl_module import VclModule

        trace = StepResult(step="vclay", success=True)
        decisions = []

        gr_min = self.config.gr_min
        gr_max = self.config.gr_max
        gr_min_source = "user" if gr_min is not None else "auto"
        gr_max_source = "user" if gr_max is not None else "auto"

        if gr_min is None or gr_max is None:
            gr_curve = well.get("GR")
            if gr_curve is not None:
                valid_gr = gr_curve.data[~np.isnan(gr_curve.data)]
                if len(valid_gr) > 0:
                    if gr_min is None:
                        gr_min = float(np.percentile(valid_gr, 5))
                    if gr_max is None:
                        gr_max = float(np.percentile(valid_gr, 95))

        gr_min = gr_min or 20.0
        gr_max = gr_max or 100.0

        use_sp = (self._has_curve(well, "SP") and
                  not self.config.mud_has_clay_or_kcl)
        use_nd = self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")
        sp_min = self.config.sp_min
        sp_max = self.config.sp_max
        sp_min_source = "user" if sp_min is not None else "auto"
        sp_max_source = "user" if sp_max is not None else "auto"

        if use_sp:
            sp_curve = well.get("SP")
            if sp_curve is not None and (sp_min is None or sp_max is None):
                valid_sp = sp_curve.data[~np.isnan(sp_curve.data)]
                if len(valid_sp) > 0:
                    if sp_max is None:
                        sp_max = float(np.percentile(valid_sp, 5))
                    if sp_min is None:
                        sp_min = float(np.percentile(valid_sp, 95))

        decisions.append(f"GR_min = {gr_min:.1f} ({gr_min_source}, P5 of GR)")
        decisions.append(f"GR_max = {gr_max:.1f} ({gr_max_source}, P95 of GR)")
        decisions.append(f"GR method: {self.config.gr_method}")
        decisions.append(f"SP method: {'enabled' if use_sp else 'disabled'}"
                        + (f" (mud has clay/KCl)" if self.config.mud_has_clay_or_kcl else ""))
        decisions.append(f"N-D method: {'enabled' if use_nd else 'disabled (missing NPHI or RHOB)'}")
        if use_sp and sp_max is not None and sp_min is not None:
            decisions.append(f"SP_clean = {sp_max:.1f} ({sp_max_source}, P5 of SP)")
            decisions.append(f"SP_shale = {sp_min:.1f} ({sp_min_source}, P95 of SP)")

        vcl_module = VclModule(
            gr_min=gr_min, gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp, use_nd=use_nd,
            sp_min=sp_min, sp_max=sp_max
        )
        result = vcl_module.calculate(well)
        vcl_module.add_to_well(well, result)

        inputs = ["GR"]
        outputs = ["VCL", "VCLAY"]
        if use_sp: inputs.append("SP")
        if use_nd: inputs.extend(["NPHI", "RHOB"])
        if result.vcl_gr is not None: outputs.extend(["VCL_GR", "VCLGR"])
        if result.vcl_sp is not None: outputs.extend(["VCL_SP", "VCLSP"])
        if result.vcl_nd is not None: outputs.extend(["VCL_ND", "VCLND"])

        decisions.append(f"Final VCL = min(available methods), range: "
                        f"{np.nanmin(result.vcl):.3f} - {np.nanmax(result.vcl):.3f}")

        trace.method = self.config.gr_method
        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "gr_min": gr_min, "gr_max": gr_max,
            "gr_method": self.config.gr_method,
            "use_sp": use_sp, "use_nd": use_nd,
            "sp_min": sp_min, "sp_max": sp_max,
            "sp_min_source": sp_min_source if use_sp else None,
            "sp_max_source": sp_max_source if use_sp else None,
            "nd_clean1_density": 2.71,
            "nd_clean1_neutron": 0.00,
            "nd_clean2_density": 2.03,
            "nd_clean2_neutron": 0.40,
            "nd_clay_density": 2.62,
            "nd_clay_neutron": 0.38,
        }
        trace.decisions = decisions
        return trace

    def _step_rcl(self, well: "Well") -> StepResult:
        from petrophys.clay_volume.rcl import RclModule

        trace = StepResult(step="rcl", success=True, method="median_Rt_in_shale_zones")
        rdep_curve = well.get("RDEP") or well.get("RT")
        vcl_curve = well.get("VCL") or well.get("VCLAY")
        trace.input_curves = [
            curve.mnemonic for curve in (rdep_curve, vcl_curve) if curve is not None
        ]
        trace.parameters = {"vcl_threshold": 0.9, "default_rcl": 5.0, "statistic": "median"}

        lithology = self._build_lithology_log(well)
        shale_mask = None
        shale_groups: List[str] = []
        if isinstance(lithology, np.ndarray):
            shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            shale_groups = _ordered_unique_values(
                str(item).upper() for item in lithology.tolist() if _is_shale_group(item)
            )

        rcl_module = RclModule(statistic="median")
        self._result.rcl_computed = rcl_module.calculate(
            well,
            vcl=vcl_curve.data if vcl_curve is not None else None,
            shale_mask=shale_mask,
        )
        shale_sample_count = 0
        source = "default"
        if rdep_curve is not None:
            rdep = np.asarray(rdep_curve.data, dtype=float)
            if shale_mask is not None:
                valid = shale_mask & np.isfinite(rdep)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "lithology_shale_median_rt"
            if source == "default" and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                valid = np.isfinite(rdep) & np.isfinite(vcl) & (vcl > 0.9)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "vcl_gt_0.9_median_rt"

        trace.method = source
        trace.parameters["rcl_computed"] = self._result.rcl_computed
        trace.parameters["source"] = source
        trace.parameters["shale_sample_count"] = shale_sample_count
        trace.parameters["shale_lithology_groups"] = shale_groups
        if source == "lithology_shale_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT in shale lithology samples, n={shale_sample_count})"
            )
        elif source == "vcl_gt_0.9_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT where VCL > 0.9, n={shale_sample_count})"
            )
        else:
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (default because no shale lithology or VCL > 0.9 samples were available)"
            )
        return trace

    def _step_lithology(self, well: "Well") -> StepResult:
        from petrophys.lithology.lithology_module import LithologyModule

        trace = StepResult(step="lithology", success=True, method="multi_mineral_solver")
        trace.input_curves = ["RHOB", "NPHI", "VCL"]
        if well.get("PHIE") is not None:
            trace.input_curves.append("PHIE")
        if well.get("PEF") is not None:
            trace.input_curves.append("PEF")
        if well.get("DTC") is not None:
            trace.input_curves.append("DTC")

        litho_module = LithologyModule()
        result = litho_module.calculate(well)
        input_lithology_available = (
            self._has_curve(well, "LITHOLOGY_GEOLINK")
            or self.config.lithology_csv is not None
        )
        write_lithology_code = not input_lithology_available
        litho_module.add_to_well(well, result, include_lithology_code=write_lithology_code)

        trace.output_curves = ["VQUAR", "VCALC", "VDOLO", "VANHY"]
        if write_lithology_code:
            trace.output_curves.append("LITHO")
        trace.parameters["lithology_curve_written"] = write_lithology_code
        trace.parameters["lithology_source_priority"] = (
            "multi_mineral_fallback" if write_lithology_code else "input_lithology"
        )
        if write_lithology_code:
            trace.decisions.append(
                "No input lithology curve was available -> inferred LITHO from mineral volumes"
            )
        else:
            trace.decisions.append(
                "Input lithology was available -> mineral volumes computed, LITHO not inferred or overwritten"
            )
        trace.decisions.append("Raw-log multi-mineral solve with VCL/PHIE clay-porosity terms")
        trace.decisions.append("PEF screened for plausibility; DTC used when PEF is absent or invalid")
        return trace

    def _step_rmf(self, well: "Well") -> StepResult:
        from petrophys.resistivity.rmf.rmf_module import RmfModule
        from petrophys.resistivity.rw.salinity_rw import rw_from_salinity
        from petrophys.environmental.temperature import correct_resistivity_temperature

        trace = StepResult(step="rmf", success=True)
        decisions = []

        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)

        def _finalize_rmf(
            method: str,
            values: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            output_curves: Optional[List[str]] = None,
        ) -> StepResult:
            rmf_curve = np.asarray(values, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            self._result.rmf_computed = rmf_curve
            well.add_curve("RMF", rmf_curve, "ohm.m", "Mud Filtrate Resistivity")
            trace.method = method
            trace.output_curves = output_curves or ["RMF"]
            trace.parameters = {
                **parameters,
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
                "rmf_min": float(np.nanmin(rmf_curve)),
                "rmf_max": float(np.nanmax(rmf_curve)),
            }
            trace.decisions = decisions
            return trace

        if self.config.mud_type == "OBM":
            decisions.append("OBM detected -> Rmf set to non-conductive default and written as a constant curve")
            return _finalize_rmf(
                "OBM_default",
                np.full(well.n_samples, 200.0, dtype=float),
                {"rmf": 200.0, "mud_type": "OBM"},
            )

        rmf_meas_temp_f = (
            self.config.rmf_meas_temp_f
            if self.config.rmf_meas_temp_f is not None
            else 75.0
        )
        if rmf_meas_temp_f == 75.0 and well.header.rm_temp is not None:
            rmf_meas_temp_f = float(well.header.rm_temp)
        metadata_rmf_temp_f = _parameter_temperature_f(well, "RMFT")
        if metadata_rmf_temp_f is not None and rmf_meas_temp_f == 75.0:
            rmf_meas_temp_f = metadata_rmf_temp_f

        las_rmf = well.header.rmf
        if las_rmf is None:
            las_rmf = _parameter_float(well, "RMF")
        if las_rmf is None:
            las_rmf = _constant_zoned_parameter(well, "RMF")

        rm_meas_temp_f = (
            self.config.rm_meas_temp_f
            if self.config.rm_meas_temp_f is not None
            else 75.0
        )
        if rm_meas_temp_f == 75.0 and well.header.rm_temp is not None:
            rm_meas_temp_f = float(well.header.rm_temp)
        metadata_rm_temp_f = _parameter_temperature_f(well, "RMT")
        if metadata_rm_temp_f is not None and rm_meas_temp_f == 75.0:
            rm_meas_temp_f = metadata_rm_temp_f

        las_rm = well.header.rm
        if las_rm is None:
            las_rm = _parameter_float(well, "RM")
        if las_rm is None:
            las_rm = _constant_zoned_parameter(well, "RM")

        if self.config.rmf_at_meas_temp is not None:
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile
            )
            decisions.append(f"Rmf corrected from {self.config.rmf_at_meas_temp:.3f} ohm.m "
                           f"@ {rmf_meas_temp_f:.0f}F across the temperature profile")
            return _finalize_rmf(
                "arps_temperature_profile_correction",
                result.rmf,
                {
                    "rmf_at_surface": self.config.rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )

        if las_rmf is not None:
            rmf_module = RmfModule()
            if metadata_rmf_temp_f is not None or well.header.rm_temp is not None:
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=rmf_meas_temp_f,
                    temp_formation=temp_profile,
                )
                decisions.append(
                    f"Rmf corrected from LAS metadata: {float(las_rmf):.4f} ohm.m "
                    f"@ {rmf_meas_temp_f:.0f}F across the temperature profile"
                )
                return _finalize_rmf(
                    "las_metadata_profile_temperature_correction",
                    result.rmf,
                    {
                        "rmf_at_measured_temp": float(las_rmf),
                        "measured_temp_f": rmf_meas_temp_f,
                    },
                )
            else:
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=reference_temp,
                    temp_formation=temp_profile,
                )
                decisions.append(
                    f"Rmf from LAS metadata used assumed reference temperature {reference_temp:.1f}F and was corrected along the temperature profile"
                )
                return _finalize_rmf(
                    "las_metadata_assumed_reference_temperature",
                    result.rmf,
                    {
                        "rmf_at_reference_temp": float(las_rmf),
                        "reference_temp_f": reference_temp,
                    },
                )

        if self.config.rmf_salinity_ppm is not None:
            rmf_at_meas_temp = float(rw_from_salinity(self.config.rmf_salinity_ppm, rmf_meas_temp_f))
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile,
            )
            decisions.append(
                f"Rmf from CLFL salinity: {self.config.rmf_salinity_ppm:.0f} ppm -> "
                f"{rmf_at_meas_temp:.4f} ohm.m @ {rmf_meas_temp_f:.0f}F and then along the temperature profile"
            )
            return _finalize_rmf(
                "mud_filtrate_salinity_conversion",
                result.rmf,
                {
                    "mud_filtrate_salinity_ppm": self.config.rmf_salinity_ppm,
                    "rmf_at_measured_temp": rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )
        rm_input = self.config.rm if self.config.rm is not None else las_rm
        if rm_input is not None:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            rm_curve = correct_resistivity_temperature(
                float(rm_input),
                rm_meas_temp_f,
                temp_profile,
                "F",
            )
            rmf_curve = np.asarray(rm_curve, dtype=float) * k
            well.add_curve("RM", rm_curve, "ohm.m", "Mud Resistivity")
            rm_source = "config.rm" if self.config.rm is not None else "LAS metadata"
            decisions.append(
                f"Rm from {rm_source} corrected from {float(rm_input):.4f} ohm.m "
                f"@ {rm_meas_temp_f:.0f}F across the temperature profile; Rmf = Rm(T) * {k}"
            )
            return _finalize_rmf(
                f"rm_to_rmf_conversion_factor_{k}",
                rmf_curve,
                {
                    "rm_at_measured_temp": float(rm_input),
                    "rm_measured_temp_f": rm_meas_temp_f,
                    "rm_min": float(np.nanmin(rm_curve)),
                    "rm_max": float(np.nanmax(rm_curve)),
                    "conversion_factor": k,
                    "mud_water": self.config.mud_water,
                    "source_priority": rm_source,
                },
                ["RM", "RMF"],
            )

        trace.decisions = decisions
        return trace

    def _step_rw(self, well: "Well") -> StepResult:
        from petrophys.resistivity.rw.rw_module import RwModule
        from petrophys.resistivity.rw.salinity_rw import rw_from_salinity, salinity_from_rw
        from petrophys.resistivity.rw.sp_rw import estimate_sp_lines, rw_from_sp, rw_from_rxo_rt

        trace = StepResult(step="rw", success=True)
        decisions = []
        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)
        las_rw_temp_f = _parameter_temperature_f(well, "RWT")

        def _finalize_rw(
            method: str,
            value: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            inputs: Optional[List[str]] = None,
        ) -> StepResult:
            rw_curve = np.asarray(value, dtype=float)
            if rw_curve.ndim == 0:
                rw_curve = np.full(well.n_samples, float(rw_curve), dtype=float)
            self._result.rw_computed = rw_curve
            well.add_curve("RW", rw_curve, "ohm.m", "Formation Water Resistivity")
            trace.method = method
            trace.parameters = {
                **parameters,
                "rw_min": float(np.nanmin(rw_curve)),
                "rw_max": float(np.nanmax(rw_curve)),
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
            }
            trace.input_curves = inputs or []
            trace.output_curves = ["RW"]
            trace.decisions = decisions
            return trace

        if self.config.rw is not None:
            if self.config.rw_meas_temp_f is not None:
                salinity_est = float(salinity_from_rw(self.config.rw, self.config.rw_meas_temp_f))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw corrected from {self.config.rw:.4f} ohm.m @ {self.config.rw_meas_temp_f:.0f}F "
                    f"along the temperature profile"
                )
                return _finalize_rw(
                    "user_provided_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": self.config.rw,
                        "measured_temp_f": self.config.rw_meas_temp_f,
                        "salinity_ppm_estimated": salinity_est,
                        "source_priority": "config.rw",
                    },
                )
            salinity_est = float(salinity_from_rw(self.config.rw, reference_temp))
            rw_curve = rw_from_salinity(salinity_est, temp_profile)
            decisions.append(
                f"Rw input {self.config.rw:.4f} ohm.m used assumed reference temperature {reference_temp:.1f}F and was corrected along the temperature profile"
            )
            return _finalize_rw(
                "user_provided_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(self.config.rw),
                    "reference_temp_f": reference_temp,
                    "salinity_ppm_estimated": salinity_est,
                    "source_priority": "config.rw",
                },
            )

        las_rw = _parameter_float(well, "RW")
        if las_rw is None:
            las_rw = _constant_zoned_parameter(well, "RW")
        if las_rw is not None:
            if las_rw_temp_f is not None:
                salinity_est = float(salinity_from_rw(float(las_rw), las_rw_temp_f))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw corrected from LAS metadata: {float(las_rw):.4f} ohm.m @ {las_rw_temp_f:.0f}F "
                    f"along the temperature profile"
                )
                return _finalize_rw(
                    "las_metadata_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": float(las_rw),
                        "measured_temp_f": las_rw_temp_f,
                        "salinity_ppm_estimated": salinity_est,
                        "source_priority": "las.RW",
                    },
                )
            salinity_est = float(salinity_from_rw(float(las_rw), reference_temp))
            rw_curve = rw_from_salinity(salinity_est, temp_profile)
            decisions.append(
                f"Rw from LAS metadata used assumed reference temperature {reference_temp:.1f}F and was corrected along the temperature profile"
            )
            return _finalize_rw(
                "las_metadata_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(las_rw),
                    "reference_temp_f": reference_temp,
                    "salinity_ppm_estimated": salinity_est,
                    "source_priority": "las.RW",
                },
            )

        if self.config.salinity_ppm is not None:
            rw_curve = rw_from_salinity(self.config.salinity_ppm, temp_profile)
            decisions.append(
                f"Rw from salinity {self.config.salinity_ppm:.0f} ppm across the temperature profile"
            )
            return _finalize_rw(
                "salinity_conversion",
                rw_curve,
                {
                    "salinity_ppm": self.config.salinity_ppm,
                    "source_priority": "salinity_ppm",
                },
            )

        if self._result.rmf_computed is not None:
            rmf_curve = np.asarray(self._result.rmf_computed, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            decisions.append(
                "Rw direct source unavailable -> using available RMF curve for heuristic estimation"
            )

            if self._has_curve(well, "SP"):
                if self.config.mud_has_clay_or_kcl:
                    decisions.append("SP method skipped because mud_has_clay_or_kcl=True")
                else:
                    try:
                        sp_module = RwModule(method="sp")
                        sp_curve = well.get("SP")
                        sp_data = np.asarray(sp_curve.data, dtype=float)
                        clean_mask = sp_module._build_clean_water_mask(well)
                        shale_mask = sp_module._build_shale_mask(well)
                        clean_sp = sp_data[clean_mask] if clean_mask is not None else sp_data
                        clean_sp = clean_sp[np.isfinite(clean_sp)]
                        if clean_sp.size > 0:
                            sp_clean, _ = estimate_sp_lines(clean_sp)
                        else:
                            sp_clean, _ = estimate_sp_lines(sp_data)

                        if shale_mask is not None:
                            shale_sp = sp_data[shale_mask]
                            shale_sp = shale_sp[np.isfinite(shale_sp)]
                            if shale_sp.size > 0:
                                sp_shale = float(np.nanmedian(shale_sp))
                            else:
                                _, sp_shale = estimate_sp_lines(sp_data)
                        else:
                            _, sp_shale = estimate_sp_lines(sp_data)

                        rw_curve = rw_from_sp(sp_clean, sp_shale, rmf_curve, temp_profile)
                        decisions.append(
                            f"Rw via SP clean-water method with SPclean={sp_clean:.2f} mV and SPshale={sp_shale:.2f} mV across the temperature profile"
                        )
                        return _finalize_rw(
                            "sp_clean_water",
                            rw_curve,
                            {
                                "rmf_min": float(np.nanmin(rmf_curve)),
                                "rmf_max": float(np.nanmax(rmf_curve)),
                                "sp_clean_mv": float(sp_clean),
                                "sp_shale_mv": float(sp_shale),
                                "ssp_mv": float(sp_clean - sp_shale),
                                "source_priority": "rmf -> SP",
                            },
                            ["SP"],
                        )
                    except Exception as exc:
                        message = f"Rw SP heuristic failed: {exc}"
                        self._result.warnings.append(message)
                        decisions.append(message)
            else:
                decisions.append("SP method skipped because SP curve is unavailable")

            try:
                from petrophys.resistivity.rw.sp_rw import rw_from_rxo_rt
                rxo = well.get("RSHA") or well.get("RXO")
                rt = well.get("RDEP") or well.get("RT")
                if rxo is None or rt is None:
                    raise ValueError("Rxo/Rt curves unavailable")
                rxo_data = np.asarray(rxo.data, dtype=float)
                rt_data = np.asarray(rt.data, dtype=float)
                clean_mask = RwModule()._build_clean_water_mask(well)
                if clean_mask is not None:
                    valid = clean_mask & np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                else:
                    valid = np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                if not np.any(valid):
                    raise ValueError("No valid clean-water Rxo/Rt samples")
                rw_clean = rw_from_rxo_rt(rxo_data[valid], rt_data[valid], rmf_curve[valid])
                salinity_est = float(np.nanmedian(salinity_from_rw(rw_clean, temp_profile[valid])))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw via Rxo/Rt clean-water method using {int(np.count_nonzero(valid))} valid samples and depth-wise temperature correction"
                )
                return _finalize_rw(
                    "rxo_rt_clean_water",
                    rw_curve,
                    {
                        "rmf_min": float(np.nanmin(rmf_curve)),
                        "rmf_max": float(np.nanmax(rmf_curve)),
                        "salinity_ppm_estimated": salinity_est,
                        "n_points": int(np.count_nonzero(valid)),
                        "source_priority": "rmf -> rxo_rt",
                    },
                    ["RSHA", "RDEP"],
                )
            except Exception as exc:
                message = f"Rw log heuristic failed: {exc}"
                self._result.warnings.append(message)
                decisions.append(message)
        else:
            decisions.append("Rw heuristic methods skipped because Rmf is unavailable")

        rt_curve = well.get("RDEP") or well.get("RT")
        phie_curve = well.get("PHIE")
        vcl_curve = well.get("VCL") or well.get("VCLAY")
        if rt_curve is not None and phie_curve is not None and vcl_curve is not None:
            from petrophys.base.constants import ArchieParameterDefaults

            rdep = np.asarray(rt_curve.data, dtype=float)
            phie = np.asarray(phie_curve.data, dtype=float)
            vcl = np.asarray(vcl_curve.data, dtype=float)
            archie_defaults = ArchieParameterDefaults.get(self.config.lithology)
            archie_a = float(self.config.a) if self.config.a is not None else float(archie_defaults.a)
            archie_m = float(self.config.m) if self.config.m is not None else float(archie_defaults.m)
            clean_wet_mask = (
                np.isfinite(rdep)
                & np.isfinite(phie)
                & np.isfinite(vcl)
                & (vcl < 0.3)
                & (phie > 0.05)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                rwa = rdep * np.power(phie, archie_m) / archie_a
            valid = clean_wet_mask & np.isfinite(rwa) & (rwa > 0)
            if np.any(valid):
                rw_value = float(np.percentile(rwa[valid], 5))
                rwa_reference_temp = float(np.nanmedian(temp_profile[valid]))
                salinity_est = float(salinity_from_rw(rw_value, rwa_reference_temp))
                rw_curve = rw_from_salinity(salinity_est, temp_profile)
                decisions.append(
                    f"Rw from Rwa clean-wet fallback: P5 of {rt_curve.mnemonic} * PHIE^{archie_m:g} / a "
                    f"where VCL < 0.3 and PHIE > 0.05 (n={int(np.count_nonzero(valid))}), "
                    "then depth-wise temperature correction"
                )
                return _finalize_rw(
                    "rwa_clean_wet_p5",
                    rw_curve,
                    {
                        "rwa_percentile": 5.0,
                        "rwa_clean_wet_samples": int(np.count_nonzero(valid)),
                        "rwa_min": float(np.nanmin(rwa[valid])),
                        "rwa_max": float(np.nanmax(rwa[valid])),
                        "rw_from_rwa": rw_value,
                        "archie_a": archie_a,
                        "archie_m": archie_m,
                        "archie_default_lithology": archie_defaults.lithology,
                        "rwa_reference_temp_f": rwa_reference_temp,
                        "salinity_ppm_estimated": salinity_est,
                        "vcl_cutoff": 0.3,
                        "phie_cutoff": 0.05,
                        "source_priority": "Rwa clean-wet fallback",
                    },
                    [rt_curve.mnemonic, phie_curve.mnemonic, vcl_curve.mnemonic],
                )
            decisions.append("Rwa clean-wet fallback skipped because no valid VCL < 0.3 and PHIE > 0.05 samples were available")
        else:
            decisions.append("Rwa clean-wet fallback skipped because RDEP/RT, PHIE, or VCL is unavailable")

        self._result.warnings.append("Rw could not be determined. Using default 0.05 ohm.m")
        decisions.append(
            "All Rw priority sources failed (config/LAS RW, salinity, SP, Rxo/Rt, Rwa clean-wet fallback) -> default 0.05"
        )
        rw_curve = rw_from_salinity(float(salinity_from_rw(0.05, reference_temp)), temp_profile)
        return _finalize_rw(
            "default_fallback_profile",
            rw_curve,
            {
                "rw_default": 0.05,
                "reference_temp_f": reference_temp,
                "source_priority": "default",
            },
        )

    def _step_invasion(self, well: "Well") -> StepResult:
        from petrophys.resistivity.invasion.invasion_module import InvasionModule

        trace = StepResult(step="invasion", success=True, method="rxo_rt_ratio")
        invasion_module = InvasionModule()
        result = invasion_module.analyze(well)
        well.add_curve("INVASION_RATIO", result.ratio, "", "Invasion Ratio Rxo/Rt")
        rxo_curve = well.get("RSHA") or well.get("RXO") or well.get("RESS")
        rt_curve = well.get("RDEP") or well.get("RT")
        trace.input_curves = [
            curve.mnemonic for curve in (rt_curve, rxo_curve) if curve is not None
        ]
        trace.output_curves = ["INVASION_RATIO"]
        trace.parameters = dict(result.statistics)
        rejected = int(result.statistics.get("nan_samples") or 0)
        total = int(result.statistics.get("total_samples") or len(result.ratio))
        if rejected:
            trace.decisions.append(
                "INVASION_RATIO samples written as NaN where shallow/deep resistivity inputs were invalid, "
                f"RT/RDEP was below {result.statistics.get('rt_min_used', 0.2)}, "
                f"or RXO/RT exceeded {result.statistics.get('max_ratio_used', 100.0)} "
                f"({rejected} of {total} samples rejected)"
            )
        else:
            trace.decisions.append(f"INVASION_RATIO valid for all {total} samples")
        return trace

    def _step_hc_correction(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.hc_correction.hc_correction_module import HcCorrectionModule
        from petrophys.base.constants import MatrixProperties

        trace = StepResult(step="hc_correction", success=True, method=f"{self.config.hc_type}_correction")
        decisions = []

        rho_ma = self.config.rho_ma
        rho_ma_source = "user"
        if rho_ma is None:
            try:
                rho_ma = MatrixProperties.get(self.config.matrix_type).density
                rho_ma_source = f"MatrixProperties[{self.config.matrix_type}]"
            except KeyError:
                rho_ma = 2.65
                rho_ma_source = "default"
        decisions.append(f"rho_ma = {rho_ma} g/cm3 (source: {rho_ma_source})")

        hc_module = HcCorrectionModule(
            hc_type=self.config.hc_type, rho_ma=rho_ma,
            rho_fl=self.config.rho_fl, phi_cl=self.config.phi_cl,
            rho_cl=self.config.rho_cl
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

        trace.input_curves = ["NPHI", "RHOB"]
        trace.output_curves = ["NPHI_HC", "RHOB_HC"]
        trace.parameters = {"hc_type": self.config.hc_type, "rho_ma": rho_ma,
                           "rho_fl": self.config.rho_fl, "phi_cl": self.config.phi_cl,
                           "rho_cl": self.config.rho_cl}
        trace.decisions = decisions
        return trace

    def _resolve_matrix_arrays(self, well: "Well") -> Dict[str, Any]:
        from petrophys.base.constants import MatrixProperties

        try:
            fallback_props = MatrixProperties.get(self.config.matrix_type)
        except KeyError:
            fallback_props = MatrixProperties.get("sandstone")

        fallback_rho_ma = float(self.config.rho_ma if self.config.rho_ma is not None else fallback_props.density)
        fallback_dt_ma = float(self.config.dt_ma if self.config.dt_ma is not None else fallback_props.slowness)
        fallback_nphi_ma = float(self.config.nphi_ma if self.config.nphi_ma is not None else fallback_props.nphi)

        rho_ma = np.full(well.n_samples, fallback_rho_ma, dtype=float)
        dt_ma = np.full(well.n_samples, fallback_dt_ma, dtype=float)
        nphi_ma = np.full(well.n_samples, fallback_nphi_ma, dtype=float)

        matrix_endpoints: List[Dict[str, Any]] = []
        matrix_groups_used: List[str] = []
        matrix_mode = "constant"
        matrix_source = f"config_or_default[{self.config.matrix_type}]"

        lithology = self._build_lithology_log(well)
        if isinstance(lithology, np.ndarray):
            matrix_mode = "lithology_dependent"
            matrix_source = self._lithology_source
            valid_groups = _ordered_unique_values(
                group for group in lithology.tolist()
                if _coerce_lithology_group(group) is not None
            )
            for raw_group in valid_groups:
                group = _coerce_lithology_group(raw_group)
                matrix_key = LITHOLOGY_GROUP_TO_MATRIX.get(group or "", self.config.matrix_type)
                try:
                    props = MatrixProperties.get(matrix_key)
                except KeyError:
                    props = fallback_props
                    matrix_key = self.config.matrix_type

                mask = np.asarray([item == raw_group for item in lithology], dtype=bool)
                if not np.any(mask):
                    continue

                rho_ma[mask] = float(props.density)
                dt_ma[mask] = float(props.slowness)
                nphi_ma[mask] = float(props.nphi)
                matrix_groups_used.append(str(raw_group).upper())
                matrix_endpoints.append(
                    {
                        "lithology_group": str(raw_group).upper(),
                        "matrix_key": matrix_key,
                        "rho_ma": float(props.density),
                        "dt_ma": float(props.slowness),
                        "nphi_ma": float(props.nphi),
                    }
                )

        return {
            "rho_ma": rho_ma,
            "dt_ma": dt_ma,
            "nphi_ma": nphi_ma,
            "matrix_mode": matrix_mode,
            "matrix_source": matrix_source,
            "matrix_groups_used": matrix_groups_used,
            "matrix_endpoints": matrix_endpoints,
            "fallback_rho_ma": fallback_rho_ma,
            "fallback_dt_ma": fallback_dt_ma,
            "fallback_nphi_ma": fallback_nphi_ma,
        }

    def _resolve_phi_clay(self, well: "Well", vcl_curve) -> Dict[str, Any]:
        explicit_phi_cl = _parameter_float(well, "PNSH")
        explicit_source = "LAS PNSH parameter"
        if explicit_phi_cl is None:
            explicit_phi_cl = _constant_zoned_parameter(well, "PNSH")
            explicit_source = "zoned LAS PNSH fallback"

        if explicit_phi_cl is not None:
            return {
                "phi_clay": float(explicit_phi_cl),
                "phi_clay_source": explicit_source,
                "phi_clay_basis_count": None,
                "phi_clay_basis": "explicit_parameter",
            }

        nphi_curve = well.get("NPHI")
        if nphi_curve is not None:
            nphi = np.asarray(nphi_curve.data, dtype=float)
            shale_mask = np.zeros(well.n_samples, dtype=bool)
            lithology = self._build_lithology_log(well)
            if isinstance(lithology, np.ndarray):
                shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            basis = "lithology_shale_average_nphi"

            valid_mask = shale_mask & np.isfinite(nphi)
            if not np.any(valid_mask) and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                shale_mask = np.isfinite(vcl) & (vcl > 0.9)
                valid_mask = shale_mask & np.isfinite(nphi)
                basis = "vcl_gt_0.9_average_nphi"

            if np.any(valid_mask):
                phi_clay = float(np.clip(np.nanmean(nphi[valid_mask]), 0.0, 0.6))
                return {
                    "phi_clay": phi_clay,
                    "phi_clay_source": basis,
                    "phi_clay_basis_count": int(np.count_nonzero(valid_mask)),
                    "phi_clay_basis": basis,
                }

        return {
            "phi_clay": float(self.config.phi_cl),
            "phi_clay_source": "config_or_default",
            "phi_clay_basis_count": None,
            "phi_clay_basis": "config_or_default",
        }

    def _step_porosity(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.porosity.density_porosity import density_porosity
        from petrophys.petrophysics.porosity.effective_porosity import effective_porosity
        from petrophys.petrophysics.porosity.neutron_porosity import neutron_porosity
        from petrophys.petrophysics.porosity.sonic_porosity import secondary_porosity, wyllie_time_average
        from petrophys.petrophysics.porosity.total_porosity import total_porosity

        trace = StepResult(step="porosity", success=True, method="component_based")
        decisions = []
        inputs = []
        outputs = []

        nphi_curve = well.get("NPHI")
        rhob_curve = well.get("RHOB")
        dtc_curve = well.get("DTC")
        existing_phin_curve = well.get("PHIN")
        existing_phid_curve = well.get("PHID")
        existing_phis_curve = well.get("PHIS")
        existing_phit_curve = well.get("PHIT")
        vcl_curve = well.get("VCL") or well.get("VCLAY")

        matrix_info = self._resolve_matrix_arrays(well)
        rho_ma = matrix_info["rho_ma"]
        dt_ma = matrix_info["dt_ma"]
        nphi_ma = matrix_info["nphi_ma"]
        phi_clay_info = self._resolve_phi_clay(well, vcl_curve)
        phi_clay = float(phi_clay_info["phi_clay"])

        phin = None
        phid = None
        phis = None
        phit = None
        phie_basis = None
        phie_basis_name = None
        phit_method = None
        phie = None

        if nphi_curve is not None:
            inputs.append("NPHI")
            phin = neutron_porosity(nphi_curve.data, nphi_matrix=nphi_ma)
            well.add_curve("PHIN", phin, "v/v", "Neutron Porosity")
            outputs.append("PHIN")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append(
                    "PHIN = clip(NPHI - nphi_ma(depth), 0, 0.6) using lithology-specific matrix neutron endpoints"
                )
            else:
                decisions.append(
                    f"PHIN = clip(NPHI - nphi_ma, 0, 0.6) using nphi_ma={matrix_info['fallback_nphi_ma']:.3f}"
                )
        elif existing_phin_curve is not None:
            phin = existing_phin_curve.data
            inputs.append("PHIN")
            decisions.append("PHIN reused from existing well curve")

        if rhob_curve is not None:
            inputs.append("RHOB")
            phid = density_porosity(rhob_curve.data, rho_ma, self.config.rho_fl)
            well.add_curve("PHID", phid, "v/v", "Density Porosity")
            outputs.append("PHID")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHID = (rho_ma(depth) - RHOB) / (rho_ma(depth) - rho_fl)")
            else:
                decisions.append("PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)")
        elif existing_phid_curve is not None:
            phid = existing_phid_curve.data
            inputs.append("PHID")
            decisions.append("PHID reused from existing well curve")

        if dtc_curve is not None:
            inputs.append("DTC")
            phis = wyllie_time_average(dtc_curve.data, dt_ma, self.config.dt_fl)
            well.add_curve("PHIS", phis, "v/v", "Sonic Porosity")
            outputs.append("PHIS")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHIS from Wyllie time-average equation using dt_ma(depth)")
            else:
                decisions.append("PHIS from Wyllie time-average equation")
        elif existing_phis_curve is not None:
            phis = existing_phis_curve.data
            inputs.append("PHIS")
            decisions.append("PHIS reused from existing well curve")

        if phid is not None and phin is not None:
            phit = total_porosity(phin, phid)
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "adaptive_nd"
            decisions.append("PHIT = adaptive(PHIN, PHID)")
        elif phid is not None:
            phie_basis = phid
            phie_basis_name = "PHID"
            phit_method = "density_only"
            decisions.append("Density-only porosity: PHID computed, PHIT not created")
        elif phin is not None:
            phie_basis = phin
            phie_basis_name = "PHIN"
            phit_method = "neutron_only"
            decisions.append("Neutron-only porosity: PHIN computed, PHIT not created")
        elif phis is not None:
            phie_basis = phis
            phie_basis_name = "PHIS"
            phit_method = "sonic_only"
            decisions.append("Sonic-only porosity: PHIS computed, PHIT not created")
        else:
            if existing_phit_curve is not None:
                phit = existing_phit_curve.data
                phie_basis = phit
                phie_basis_name = "PHIT"
                phit_method = "existing_phit"
                inputs.append("PHIT")
                decisions.append("PHIT reused from existing well curve")

        if phit is not None:
            well.add_curve("PHIT", phit, "v/v", "Total Porosity")
            if "PHIT" not in outputs:
                outputs.append("PHIT")

        if phie_basis is not None:
            if vcl_curve is not None:
                inputs.append(vcl_curve.mnemonic)
                vcl = vcl_curve.data
            else:
                vcl = np.zeros(well.n_samples, dtype=float)

            phie = effective_porosity(phie_basis, vcl, phi_clay)
            well.add_curve("PHIE", phie, "v/v", "Effective Porosity")
            outputs.append("PHIE")
            if vcl_curve is not None:
                decisions.append(
                    f"PHIE = {phie_basis_name} - {vcl_curve.mnemonic} * phi_clay ({phi_clay:.3f})"
                )
            else:
                decisions.append(f"PHIE = {phie_basis_name} (no clay curve available)")

        if phis is not None and phit is not None:
            phi_sec = secondary_porosity(phis, phit)
            well.add_curve("PHI_SEC", phi_sec, "v/v", "Secondary Porosity Index")
            outputs.append("PHI_SEC")
            decisions.append("PHI_SEC = PHIT - PHIS")

        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "matrix_mode": matrix_info["matrix_mode"],
            "matrix_source": matrix_info["matrix_source"],
            "matrix_groups_used": matrix_info["matrix_groups_used"],
            "matrix_endpoints": matrix_info["matrix_endpoints"],
            "nphi_ma": matrix_info["fallback_nphi_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "nphi_ma_min": float(np.nanmin(nphi_ma)),
            "nphi_ma_max": float(np.nanmax(nphi_ma)),
            "rho_ma": matrix_info["fallback_rho_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "rho_ma_min": float(np.nanmin(rho_ma)),
            "rho_ma_max": float(np.nanmax(rho_ma)),
            "rho_fl": self.config.rho_fl,
            "dt_ma": matrix_info["fallback_dt_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "dt_ma_min": float(np.nanmin(dt_ma)),
            "dt_ma_max": float(np.nanmax(dt_ma)),
            "dt_fl": self.config.dt_fl,
            "phi_clay": phi_clay,
            "phi_clay_source": phi_clay_info["phi_clay_source"],
            "phi_clay_basis": phi_clay_info["phi_clay_basis"],
            "phi_clay_basis_count": phi_clay_info["phi_clay_basis_count"],
            "phie_basis": phie_basis_name,
            "phit_method": phit_method,
        }
        trace.decisions = decisions
        return trace

    def _step_saturation(self, well: "Well") -> StepResult:
        from petrophys.base.constants import ArchieParameterDefaults
        from petrophys.petrophysics.saturation.sw_module import SaturationModule

        trace = StepResult(step="saturation", success=True)
        lithology = self._build_lithology_log(well)
        default_coeffs = ArchieParameterDefaults.get(self.config.lithology)
        a_value = self.config.a if self.config.a is not None else default_coeffs.a
        m_value = self.config.m if self.config.m is not None else default_coeffs.m
        n_value = self.config.n if self.config.n is not None else default_coeffs.n

        sw_module = SaturationModule(
            a=a_value, m=m_value, n=n_value,
            model=self.config.sw_model, vcl_cutoff=self.config.vcl_cutoff
        )

        result = sw_module.calculate(
            well, rw=self._result.rw_computed, rcl=self._result.rcl_computed,
            salinity=self.config.salinity_ppm, rmf=self._result.rmf_computed,
            lithology=lithology, cec=self.config.cec, b_coeff=self.config.b_coeff,
            rho_ma=self.config.rho_ma or 2.65, m_star=self.config.m_star,
            rwb=self.config.rwb, phi_cl=self.config.phi_cl,
            rtsh=self.config.rtsh, vsh_exp=self.config.vsh_exp,
            shale_exp=self.config.shale_exp,
        )

        well.add_curve("SW", result.sw, "v/v", f"Water Saturation ({result.model})")
        outputs = ["SW"]
        if result.sxo is not None:
            well.add_curve("SXO", result.sxo, "v/v", "Flushed Zone Water Saturation")
            outputs.append("SXO")

        trace.method = result.model
        rt_curve = well.get("RDEP") or well.get("RT")
        phie_curve = well.get("PHIE")
        vcl_curve = well.get("VCL") or well.get("VCLAY")
        trace.input_curves = [
            curve.mnemonic for curve in (rt_curve, phie_curve, vcl_curve) if curve is not None
        ]
        trace.output_curves = outputs
        trace.parameters = {
            "sw_model": result.model,
            "sw_model_requested": self.config.sw_model,
            "sw_model_sample_counts": result.model_counts or {},
            "a": a_value, "m": m_value, "n": n_value,
            "archie_default_lithology": default_coeffs.lithology,
            "archie_default_note": default_coeffs.note,
            "archie_a_source": "config_or_las" if self.config.a is not None else "ArchieParameterDefaults",
            "archie_m_source": "config_or_las" if self.config.m is not None else "ArchieParameterDefaults",
            "archie_n_source": "config_or_las" if self.config.n is not None else "ArchieParameterDefaults",
            "rw_min": float(np.nanmin(np.asarray(self._result.rw_computed, dtype=float))),
            "rw_max": float(np.nanmax(np.asarray(self._result.rw_computed, dtype=float))),
            "rcl": self._result.rcl_computed,
            "salinity_ppm": self.config.salinity_ppm,
            "cec": self.config.cec,
            "cec_present": self.config.cec is not None,
            "b_coeff": self.config.b_coeff,
            "b_coeff_present": self.config.b_coeff is not None,
            "m_star": self.config.m_star,
            "rwb": self.config.rwb,
            "phi_cl": self.config.phi_cl,
            "rtsh": self.config.rtsh if self.config.rtsh is not None else self._result.rcl_computed,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
            "vcl_cutoff_for_routing": self.config.vcl_cutoff,
            "lithology_type": self.config.lithology if isinstance(lithology, str) else "per-depth array",
            "sxo_method": result.sxo_method,
            "sxo_reason": result.sxo_reason,
        }
        if self._result.rmf_computed is not None:
            trace.parameters["rmf_min"] = float(np.nanmin(np.asarray(self._result.rmf_computed, dtype=float)))
            trace.parameters["rmf_max"] = float(np.nanmax(np.asarray(self._result.rmf_computed, dtype=float)))
        trace.decisions.append(f"Model selected: {result.model}")
        if self.config.sw_model == "auto":
            trace.decisions.append("Auto-routing: depth-by-depth model selection based on VCLAY and lithology")
            if result.model_counts:
                trace.decisions.append(
                    "Depth routing counts: "
                    + ", ".join(f"{model}={count}" for model, count in result.model_counts.items())
                )
        trace.decisions.append(
            f"Archie params: a={a_value}, m={m_value}, n={n_value} "
            f"(defaults from {default_coeffs.lithology} where not provided by config/LAS)"
        )
        trace.decisions.append(
            f"Rw range = {trace.parameters['rw_min']:.4f} to {trace.parameters['rw_max']:.4f}, Rcl = {self._result.rcl_computed}"
        )
        if result.sxo_method == "archie_rxo_rmf":
            trace.decisions.append("SXO from Archie invaded-zone equation using RSHA/RXO and RMF")
        elif result.sxo_method == "fifth_root_quicklook":
            trace.decisions.append(
                f"SXO fallback: SXO = SW^(1/5) because {result.sxo_reason}"
            )
        if np.any(np.isfinite(result.sw)):
            trace.decisions.append(f"Sw range: {np.nanmin(result.sw):.3f} - {np.nanmax(result.sw):.3f}")
        else:
            trace.decisions.append("Sw range: all values are NaN")
        return trace

    def _step_permeability(self, well: "Well") -> StepResult:
        from petrophys.petrophysics.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="permeability", success=True, method=self.config.perm_method)
        decisions = []
        resflag_info = None
        perm_method_norm = str(self.config.perm_method or "").strip().lower().replace("-", "_")
        perm_method_uses_swirr = perm_method_norm not in {"lith_vcl", "lithology_vcl", "dunite", "dunite_basic"}
        if perm_method_uses_swirr and self.config.swirr is None and not self._has_curve(well, "SWIRR", "SWI", "SWIR"):
            resflag_info = self._ensure_resflag_for_swirr(well)
        outputs = []

        if resflag_info is not None:
            if resflag_info.get("created"):
                outputs.append("RESFLAG")
                if resflag_info.get("source_curve") == "NET_RES":
                    decisions.append(
                        "RESFLAG copied from existing NET_RES before permeability "
                        f"({resflag_info['reservoir_samples']} reservoir samples)"
                    )
                else:
                    decisions.append(
                        "RESFLAG created before permeability from NetPayModule net_reservoir "
                        f"using {resflag_info['vcl_curve']}, {resflag_info['phie_curve']}, "
                        f"and {resflag_info['sw_curve']} ({resflag_info['reservoir_samples']} samples)"
                    )
            else:
                decisions.append(
                    f"{resflag_info['source_curve']} already present before permeability "
                    "and available for Swirr routing"
                )

        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
        )
        lithology = None
        if not self._has_curve(well, "LITHOLOGY_GEOLINK", "LITHO_CLASS", "LITHO"):
            lithology = self._build_lithology_log(well)

        result = perm_module.calculate(
            well,
            swi=self.config.swirr,
            lithology=lithology,
            peer_wells=self.config.peer_wells,
            formation_tops=self.config.formation_tops,
        )
        empirical_perm = result.metadata.get("empirical_perm")
        empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
        perm_output = empirical_perm if empirical_perm is not None else result.perm
        perm_desc = empirical_method if empirical_perm is not None else result.method
        well.add_curve("PERM", perm_output, "mD", f"Permeability ({perm_desc})")

        outputs.append("PERM")
        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CC",
                phi_corecal,
                "v/v",
                "Permeability input porosity from core calibration",
            )
            outputs.append("PHIE_CC")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
        swirr_metadata = dict(result.swi_metadata)
        swirr_app = swirr_metadata.pop("swirr_app", None)
        if swirr_app is not None and empirical_method != "lith_vcl":
            well.add_curve(
                "SWIRR_APP",
                swirr_app,
                "v/v",
                "Apparent irreducible water saturation from hydrocarbon-mask P5 method",
            )
            outputs.append("SWIRR_APP")
        method_metadata = result.metadata.get("method_metadata") or {}
        method_input_curves = result.metadata.get("input_curves") or []
        input_curves = list(method_input_curves) if empirical_method == "lith_vcl" else [result.metadata.get("porosity_curve", "PHIE")]
        if empirical_method != "lith_vcl" and result.swi_source in {"hc_mask_p5", "default_0.20"}:
            input_curves.append("SW")
            rt_curve = swirr_metadata.get("rt_curve")
            if rt_curve:
                input_curves.append(rt_curve)
            if swirr_metadata.get("resflag_source"):
                input_curves.append(swirr_metadata["resflag_source"])
        trace.input_curves = list(dict.fromkeys(input_curves))
        trace.output_curves = outputs
        trace.parameters = {
            "method": self.config.perm_method,
            "fallback_method": self.config.perm_fallback_method,
            "swirr_source": result.swi_source,
            "swirr_app_method": swirr_metadata.get("swirr_app_method"),
            "bvw_irr": result.bvw_irr,
            "swirr_metadata": swirr_metadata,
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "empirical_method": empirical_method,
            "empirical_error": result.metadata.get("empirical_error"),
            "empirical_attempted_methods": result.metadata.get("empirical_attempted_methods"),
            "empirical_method_metadata": method_metadata,
            "perm_max_clip_md": result.metadata.get("perm_max_clip_md"),
        }
        if resflag_info is not None:
            trace.parameters["pre_permeability_resflag"] = resflag_info
        if result.metadata.get("core_sections_count"):
            decisions.append(
                f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
            )
        for model in result.metadata.get("core_models", []):
            source = model.get("source_well") or "unknown"
            formation = model.get("source_formation") or "all_depths"
            source_type = model.get("source_type") or "core"
            decisions.append(
                f"{source_type}: {source} / {formation} (n={model.get('n_core_points')}, "
                f"perm R2={model.get('perm_r2'):.3f})"
            )
        if np.isscalar(result.swi_used):
            trace.parameters["swirr"] = float(result.swi_used)
            if result.swi_source == "hc_mask_p5":
                n_samples = swirr_metadata.get("hc_mask_samples")
                rt_curve = swirr_metadata.get("rt_curve", "RDEP")
                decisions.append(
                    f"SWIRR_APP = P5(SW where resflag and {rt_curve} > 5) = {float(result.swi_used):.3f} (n={n_samples})"
                )
            elif result.swi_source == "default_0.20":
                n_samples = swirr_metadata.get("hc_mask_samples")
                reason = swirr_metadata.get("reason")
                if n_samples is not None:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because hydrocarbon mask had {n_samples} samples")
                elif reason:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because {reason}")
                else:
                    decisions.append("SWIRR_APP = 0.200 fallback")
            else:
                decisions.append(f"Swirr = {float(result.swi_used):.3f} ({result.swi_source})")
        elif isinstance(result.swi_used, np.ndarray):
            decisions.append(f"Swirr resolved as curve ({result.swi_source})")
        elif result.swi_source == "not_required" and empirical_method == "lith_vcl":
            decisions.append(
                "Swirr not used because lith_vcl permeability uses PHIT, VCL, and lithology"
            )
        else:
            decisions.append("Swirr not used because permeability came from core calibration")
        if empirical_method == "lith_vcl":
            decisions.append(
                "lith_vcl empirical permeability: "
                "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)"
            )
            groups = method_metadata.get("lithology_parameter_groups") or []
            if groups:
                decisions.append("lith_vcl parameter groups used: " + ", ".join(groups))
        elif empirical_method == "coates":
            decisions.append(
                "Coates-Denoo empirical permeability: "
                "k = 10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2"
            )
        elif empirical_method == "timur":
            decisions.append(
                "Timur empirical permeability: "
                "k = 0.136 * (100 * PHIE)^4.4 / (100 * SWIRR)^2"
            )
        if result.metadata.get("empirical_error") and empirical_method == "lith_vcl":
            decisions.append(f"Earlier empirical method failed before lith_vcl: {result.metadata['empirical_error']}")
        if result.metadata.get("perm_max_clip_md") is not None:
            decisions.append(f"Permeability clipped to <= {result.metadata['perm_max_clip_md']:.0f} mD")
        decisions.append(f"Empirical permeability curve stored as PERM ({perm_desc})")
        if perm_core is not None:
            decisions.append("Core-corrected permeability stored as PERM_CC")
        decisions.append(f"Permeability strategy: {result.method}")
        decisions.append(f"Perm range: {np.nanmin(perm_output):.3f} - {np.nanmax(perm_output):.1f} mD")
        trace.decisions = decisions
        return trace

    def _ensure_resflag_for_swirr(self, well: "Well") -> Optional[Dict[str, Any]]:
        """Create RESFLAG before permeability so Swirr uses workflow net-reservoir logic."""
        resflag_curve = well.get("RESFLAG")
        if resflag_curve is not None:
            data = np.asarray(resflag_curve.data, dtype=float)
            return {
                "source_curve": "RESFLAG",
                "created": False,
                "reservoir_samples": int(np.nansum(data > 0.5)),
            }

        net_res_curve = well.get("NET_RES")
        if net_res_curve is not None:
            data = np.asarray(net_res_curve.data, dtype=float)
            well.add_curve("RESFLAG", data, "", "Reservoir Flag")
            return {
                "source_curve": "NET_RES",
                "created": True,
                "vcl_curve": "existing_NET_RES",
                "phie_curve": "existing_NET_RES",
                "sw_curve": "existing_NET_RES",
                "reservoir_samples": int(np.nansum(data > 0.5)),
            }

        if not self._has_curve(well, "PHIE") or not self._has_curve(well, "SW", "SWA", "SWT-LOG"):
            return None

        from petrophys.reservoir.net_pay.net_pay_module import NetPayModule

        phie_curve = self._first_curve(well, "PHIE")
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        vcl_curve = well.get("VCL") or well.get("VCLAY")
        vcl = np.asarray(vcl_curve.data, dtype=float) if vcl_curve is not None else None

        lithology = self._build_lithology_log(well)
        np_module = NetPayModule(
            vcl_cutoff=self.config.vcl_cutoff,
            phie_cutoff=self.config.phie_cutoff,
            sw_cutoff=self.config.sw_cutoff,
            lithology=lithology,
        )
        result = np_module.calculate(
            well,
            vcl=vcl,
            phie=np.asarray(phie_curve.data, dtype=float),
            sw=np.asarray(sw_curve.data, dtype=float),
        )
        resflag = result.net_reservoir.astype(float)
        well.add_curve("RESFLAG", resflag, "", "Reservoir Flag")

        return {
            "source_curve": "NetPayModule.net_reservoir",
            "created": True,
            "vcl_curve": vcl_curve.mnemonic if vcl_curve is not None else "implicit_0",
            "phie_curve": phie_curve.mnemonic,
            "sw_curve": sw_curve.mnemonic,
            "vcl_cutoff": self.config.vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
            "reservoir_samples": int(np.nansum(resflag > 0.5)),
        }

    def _step_rock_quality(self, well: "Well") -> StepResult:
        from petrophys.reservoir.rock_quality.rock_quality_module import RockQualityModule

        trace = StepResult(step="rock_quality", success=True, method="RQI_FZI_phi_z")
        rq_module = RockQualityModule()
        result = rq_module.calculate(well)
        rq_module.add_to_well(well, result)
        trace.input_curves = ["PERM", "PHIE"]
        trace.output_curves = ["RQI", "FZI", "PHI_Z"]
        trace.decisions.append("RQI = sqrt(K/PHIE), PHI_Z = PHIE/(1-PHIE), FZI = RQI/PHI_Z")
        return trace

    def _step_bvw(self, well: "Well") -> StepResult:
        from petrophys.reservoir.bvw.bvw_module import calculate_bvw

        phie = well.get("PHIE").data
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sw = sw_curve.data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

        trace = StepResult(step="bvw", success=True, method="PHIE_x_SW")
        trace.input_curves = ["PHIE", sw_curve.mnemonic]
        trace.output_curves = ["BVW"]
        trace.decisions.append(f"BVW = PHIE * {sw_curve.mnemonic}")
        return trace

    def _step_mhi(self, well: "Well") -> StepResult:
        from petrophys.reservoir.mhi.mhi_module import MhiModule

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sxo_curve = self._first_sxo_curve(well)
        mhi_module = MhiModule()
        mhi = mhi_module.calculate(well, sxo=sxo_curve.data, sw=sw_curve.data)
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")

        trace = StepResult(step="mhi", success=True, method="moveable_HC_index")
        trace.input_curves = [sw_curve.mnemonic, sxo_curve.mnemonic]
        trace.output_curves = ["MHI"]
        trace.decisions.append(f"MHI = {sw_curve.mnemonic} / {sxo_curve.mnemonic}")
        return trace

    def _step_net_pay(self, well: "Well") -> StepResult:
        from petrophys.analysis import HydrocarbonAnalysis
        from petrophys.reservoir.net_pay.net_pay_module import NetPayModule

        trace = StepResult(step="net_pay", success=True, method="cutoff_flags")
        lithology = self._build_lithology_log(well)

        np_module = NetPayModule(
            vcl_cutoff=self.config.vcl_cutoff,
            phie_cutoff=self.config.phie_cutoff,
            sw_cutoff=self.config.sw_cutoff,
            lithology=lithology
        )
        result = np_module.calculate(well)
        np_module.add_to_well(well, result)

        well.add_curve("RESFLAG", result.net_reservoir.astype(float), "", "Reservoir Flag")

        vcl_curve = well.get("VCL") or well.get("VCLAY")
        if vcl_curve is not None:
            vcl_data = np.asarray(vcl_curve.data, dtype=float)
        else:
            vcl_data = np.zeros(well.n_samples, dtype=float)
        seal_flag = np.full(vcl_data.shape, np.nan, dtype=float)
        valid_vcl = np.isfinite(vcl_data)
        seal_flag[valid_vcl] = (vcl_data[valid_vcl] >= 0.8).astype(float)
        well.add_curve("SEALFLAG", seal_flag, "", "Seal Flag")

        hc_result = HydrocarbonAnalysis().analyze(well, add_curves=True)

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        trace.input_curves = ["VCL", "PHIE", sw_curve.mnemonic]
        trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY", "RESFLAG", "SEALFLAG"]
        if well.get("CROSSOVER_FLAG") is not None:
            trace.output_curves.append("CROSSOVER_FLAG")
        trace.parameters = {
            "vcl_cutoff": self.config.vcl_cutoff,
            "phie_cutoff": self.config.phie_cutoff,
            "sw_cutoff": self.config.sw_cutoff,
            "lithology": self.config.lithology if isinstance(lithology, str) else "per-depth",
            "seal_vcl_cutoff": 0.8,
        }
        n_rock = int(np.nansum(well.get("NET_ROCK").data > 0.5)) if well.get("NET_ROCK") else 0
        n_res = int(np.nansum(well.get("NET_RES").data > 0.5)) if well.get("NET_RES") else 0
        n_pay = int(np.nansum(well.get("NET_PAY").data > 0.5)) if well.get("NET_PAY") else 0
        n_seal = int(np.nansum(seal_flag > 0.5))
        step_m = well.step or 0.15
        trace.decisions.append(f"NET_ROCK: VCL < {self.config.vcl_cutoff} -> {n_rock} samples ({n_rock*step_m:.1f} m)")
        trace.decisions.append(f"NET_RES:  + PHIE > {self.config.phie_cutoff} -> {n_res} samples ({n_res*step_m:.1f} m)")
        trace.decisions.append(f"NET_PAY:  + {sw_curve.mnemonic} < {self.config.sw_cutoff} -> {n_pay} samples ({n_pay*step_m:.1f} m)")
        trace.decisions.append(f"SEALFLAG: VCL >= 0.8 -> {n_seal} samples ({n_seal*step_m:.1f} m)")
        if well.get("CROSSOVER_FLAG") is not None:
            n_crossover = int(np.nansum(np.asarray(well.get("CROSSOVER_FLAG").data, dtype=float) > 0.5))
            trace.decisions.append(
                f"CROSSOVER_FLAG from hydrocarbon analysis -> {n_crossover} samples ({n_crossover*step_m:.1f} m)"
            )
        trace.decisions.extend(
            f"Hydrocarbon analysis: {section_name}"
            for section_name in hc_result.sections.keys()
        )
        return trace

    # ---- Helpers ----

    def _first_curve(self, well: "Well", *names: str):
        for name in names:
            curve = well.get(name)
            if curve is not None:
                return curve
        raise ValueError(f"None of the requested curves are available: {', '.join(names)}")

    def _has_sxo_curve(self, well: "Well") -> bool:
        if self._has_curve(well, "SXO", "SWXOA", "SXO-LOG"):
            return True
        return any(name.upper().startswith("SXO-") for name in well.curves)

    def _first_sxo_curve(self, well: "Well"):
        for name in ("SXO", "SWXOA", "SXO-LOG"):
            curve = well.get(name)
            if curve is not None:
                return curve
        for name in well.curves:
            if name.upper().startswith("SXO-"):
                curve = well.get(name)
                if curve is not None:
                    return curve
        raise ValueError("No SXO curve is available")

    def _build_lithology_log(self, well: "Well") -> Union[str, np.ndarray]:
        """Build per-depth lithology from well curves, CSV, or global config."""
        if self._lithology_log is not None:
            return self._lithology_log

        gt_curve = well.get("LITHOLOGY_GEOLINK")
        if gt_curve is not None:
            try:
                gt_data = np.asarray(gt_curve.data, dtype=float)
                self._lithology_log = _group_array_from_numeric_codes(gt_data, GROUND_TRUTH_CODE_TO_GROUP)
                self._lithology_source = "ground_truth_lithology_geolink"
                return self._lithology_log
            except (TypeError, ValueError):
                pass

        if self.config.lithology_csv is not None:
            from petrophys.lithology.lithology_log import (
                load_lithology_csv, build_lithology_log
            )
            intervals = load_lithology_csv(self.config.lithology_csv)
            depth = well.depth
            self._lithology_log = build_lithology_log(
                depth, intervals, default_lithology=self.config.lithology
            )
            # Add LITHO_CLASS curve to well
            well.add_curve(
                "LITHO_CLASS", self._lithology_log, "", "Lithology Classification"
            )
            self._lithology_source = "lithology_csv"
            return self._lithology_log

        computed_curve = well.get("LITHO")
        if computed_curve is not None:
            try:
                computed_data = np.asarray(computed_curve.data, dtype=float)
                self._lithology_log = _group_array_from_numeric_codes(computed_data, COMPUTED_LITHO_CODE_TO_GROUP)
                self._lithology_source = "computed_lithology_curve"
                return self._lithology_log
            except (TypeError, ValueError):
                pass

        self._lithology_source = "config"
        return self.config.lithology

    def _get_avg_formation_temp(self, well: "Well") -> float:
        """Get average formation temperature."""
        temp = well.get("TEMP")
        if temp is not None:
            return float(np.nanmean(temp.data))

        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f if self.config.bht_f is not None else _metadata_bht_f(well)
        if bht_f is not None:
            return (surface_temp_f + float(bht_f)) / 2.0

        if well.depth is not None:
            from petrophys.environmental.temperature import calculate_temperature_profile

            try:
                result = calculate_temperature_profile(
                    well,
                    surface_temp=surface_temp_f,
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return float(np.nanmean(result.temperature))
            except Exception:
                pass

        return 150.0  # Last-resort estimate when depth is unavailable


def run_dynamic_workflow(
    well: "Well",
    config: Optional[DynamicWorkflowConfig] = None,
    **kwargs
) -> DynamicWorkflowResult:
    """
    Run the dynamic workflow on a well.

    Parameters
    ----------
    well : Well
        Well object with raw log data
    config : DynamicWorkflowConfig, optional
        Workflow configuration
    **kwargs
        Override config parameters

    Returns
    -------
    DynamicWorkflowResult

    Examples
    --------
    >>> result = run_dynamic_workflow(well, rw=0.05, lithology='clastic')
    >>> print(f"Steps executed: {[s.step for s in result.steps_executed]}")
    >>> print(f"Steps skipped: {result.steps_skipped}")
    """
    if config is None:
        config = DynamicWorkflowConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    workflow = DynamicWorkflow(config)
    return workflow.run(well)

def _coerce_to_list(value, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a string or list of strings")

def build_workflow_config(cfg: Dict[str, Any], td_ft: float) -> DynamicWorkflowConfig:
    return DynamicWorkflowConfig(
        bht_f=cfg.get("bht_f"),
        td_ft=td_ft,

        surface_temp_f=cfg.get("surface_temp_f"),
        mud_type=cfg.get("mud_type"),
        rmf_at_meas_temp=cfg.get("rmf_at_meas_temp"),
        rmf_meas_temp_f=cfg.get("rmf_meas_temp_f"),
        rmf_salinity_ppm=cfg.get("rmf_salinity_ppm"),
        rm=cfg.get("rm"),
        rm_meas_temp_f=cfg.get("rm_meas_temp_f"),

        rw=cfg["rw"],
        rw_meas_temp_f=cfg.get("rw_meas_temp_f"),
        matrix_type=cfg["matrix_type"],
        lithology=cfg["lithology"],

        a=cfg.get("a"),
        m=cfg.get("m"),
        n=cfg.get("n"),

        vcl_cutoff=cfg.get("vcl_cutoff"),
        phie_cutoff=cfg.get("phie_cutoff"),
        sw_cutoff=cfg.get("sw_cutoff"),

        gr_method=cfg.get("gr_method"),
        perm_method=cfg.get("perm_method"),
    )

def run_dynamic_workflow_by_level(
    well_names: Union[str, List[str]],
    formation_names: Optional[Union[str, List[str]]] = None,
    formation_configs: Optional[List[Any]] = None,  # Adjust type to your FormationConfig class
) -> Dict[str, Any]:
    try:
        all_well_results = []

        for well in well_names:
            # Load parent well
            parent_well = (
                ascii_reader.read_csv(f, well_name=well, formation_name=formation_names)
                if f.suffix.lower() == ".csv"
                else las_reader.read_las(f, well_name=well)
            )

            df_well = parent_well.to_dataframe()
            if not df_well.empty and "WELLNAME" in df_well.columns:
                df = df_well[df_well["WELLNAME"].astype(str).str.upper() == str(well).upper()].copy()
            else:
                df = df_well.copy()

            if formation_names and "FORMATION" in df.columns:
                df = df[df["FORMATION"].astype(str).str.upper().isin([str(x).upper() for x in formation_names])]

            if df.empty:
                all_well_results.append({"ok": False, "well_name": well, "error": "No data after filtering"})
                continue

            results = []
            has_formation_col = "FORMATION" in df.columns
            formations = sorted(df["FORMATION"].dropna().astype(str).unique().tolist()) if has_formation_col else []

            processed_slices = []  # Collect workflow-modified slices here

            for fm in formations:
                if fm not in formation_configs_by_name:
                    continue

                df_fm = df[df["FORMATION"].astype(str) == fm].copy()
                if df_fm.empty:
                    continue

                fm_top_depth = float(df_fm["DEPTH"].min())
                fm_bottom_depth = float(df_fm["DEPTH"].max())

                interval_well = parent_well.depth_slice(top=fm_top_depth, bottom=fm_bottom_depth)
                cfg = build_workflow_config(formation_configs_by_name[fm], td_ft=interval_well.bottom)
                
                # Run workflow (modifies interval_well in-place with computed curves)
                result = DynamicWorkflow(cfg).run(interval_well)
                processed_slices.append(interval_well)  #  Save modified slice

                print(
                    f"\n=== TRACE | well={well} | level=formation | "
                    f"formation={fm} | depth=[{fm_top_depth}, {fm_bottom_depth}] ==="
                )

                results.append({
                    "run_level": "formation",
                    "formation_name": fm,
                    "depth_range": [fm_top_depth, fm_bottom_depth],
                    "result": result,
                })

            # 🔁 Reconstruct single cohesive well object from all processed intervals
            reconstructed_well = parent_well._reconstruct_well_from_slices_well_obj(processed_slices)

            all_well_results.append({
                "ok": True,
                "well_name": well,
                "n_intervals": len(results),
                "results": results,
                "well_obj": reconstructed_well,  # Now contains all workflow-computed curves
            })

        return {
            "ok": True,
            "n_wells": len(all_well_results),
            "wells": all_well_results,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
