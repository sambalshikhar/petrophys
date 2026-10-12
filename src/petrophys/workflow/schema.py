"""Machine-readable descriptions of the interpretation workflow.

These helpers describe the workflow's settings and steps as plain,
JSON-serializable data, for user interfaces, documentation generators and
LLM agents (see :mod:`petrophys.mcp`).
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from typing import Any, Dict, List, Optional, Sequence

from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, WorkflowStep

__all__ = [
    "WORKFLOW_STEP_DESCRIPTIONS",
    "config_schema",
    "describe_setting",
    "workflow_steps",
]

#: One-line description of every workflow step, keyed by step name.
WORKFLOW_STEP_DESCRIPTIONS: Dict[str, str] = {
    "normalization": "Canonical curve aliases and units; implausible-value screening",
    "temperature": "Formation temperature profile from BHT or a gradient",
    "bad_hole": "Washout and density-correction quality flags",
    "spectral_gamma": (
        "Not run by pp.interpret; call "
        "petrophys.interpretation.geochemistry.interpret_spectral_gamma directly"
    ),
    "elemental_spectroscopy": (
        "Not run by pp.interpret; call "
        "petrophys.interpretation.geochemistry.interpret_elemental_spectroscopy directly"
    ),
    "vclay": "Clay volume (VCL) from gamma ray, SP and neutron-density",
    "rcl": "Clay resistivity from shale intervals",
    "rmf": "Mud-filtrate resistivity at formation temperature",
    "hc_correction": "Neutron-density hydrocarbon correction (opt-in)",
    "porosity": "Density, neutron, sonic, total (PHIT) and effective (PHIE) porosity",
    "lithology": "Mineral volumes: clastic multi-mineral or carbonate solve",
    "porosity_refinement": "Not run by pp.interpret; porosity is computed once, in the porosity step",
    "core_correction": "Porosity and permeability calibrated to core data",
    "rw": "Formation water resistivity at formation temperature",
    "invasion": "Invasion profile from shallow and deep resistivity",
    "saturation": "Water (SW) and flushed-zone (SXO) saturation",
    "permeability": "Core-calibrated or empirical permeability (PERM)",
    "rock_quality": "Reservoir quality index (RQI), flow zone indicator (FZI)",
    "bvw": "Bulk volume water",
    "mhi": "Moveable hydrocarbon index",
    "net_pay": "Net rock, net reservoir and net pay flags",
}


def workflow_steps() -> List[Dict[str, str]]:
    """Return every workflow step name, in pipeline order, with a description.

    These are the names accepted by ``skip_steps`` and used in
    ``result.steps_skipped``. ``spectral_gamma``, ``elemental_spectroscopy``
    and ``porosity_refinement`` are listed but never run by
    :func:`petrophys.interpret`; their descriptions say so. A run's trace
    (``result.steps_executed``) also contains ``input_context``.

    Returns
    -------
    list of dict
        ``{"name": ..., "description": ...}`` for every step, in pipeline
        order.

    Examples
    --------
    >>> from petrophys.workflow.schema import workflow_steps
    >>> workflow_steps()[0]["name"]
    'normalization'
    """
    return [
        {
            "name": step.value,
            "description": WORKFLOW_STEP_DESCRIPTIONS.get(step.value, ""),
        }
        for step in _pipeline_order()
    ]


def config_schema() -> List[Dict[str, Any]]:
    """Describe every :class:`~petrophys.workflow.DynamicWorkflowConfig` setting.

    Returns
    -------
    list of dict
        One entry per setting, in declaration order, with ``name``,
        ``group`` (the section it belongs to, such as ``"Formation water"``),
        ``type`` (as written in the source), ``default`` (JSON-serializable),
        ``description`` and, for settings with a fixed set of values,
        ``choices``. Settings whose default (None) depends on
        ``calculation_profile`` also list ``profile_defaults``, the value
        each profile uses; None there means the description's rule applies.

    Examples
    --------
    >>> from petrophys.workflow.schema import config_schema
    >>> rw = next(item for item in config_schema() if item["name"] == "rw")
    >>> rw["description"].startswith("Formation water resistivity")
    True
    """
    from petrophys.workflow import dynamic_workflow as workflow

    descriptions = _field_comments(DynamicWorkflowConfig)
    groups = _field_groups(DynamicWorkflowConfig)
    choices = _field_choices()
    schema = []
    for item in dataclasses.fields(DynamicWorkflowConfig):
        default: Any
        if item.default is not dataclasses.MISSING:
            default = item.default
        elif item.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = item.default_factory()  # type: ignore[misc]
        else:
            default = None
        entry: Dict[str, Any] = {
            "name": item.name,
            "group": groups.get(item.name, ""),
            "type": _type_name(item.type),
            "default": default,
            "description": descriptions.get(item.name, ""),
        }
        if item.name in choices:
            entry["choices"] = list(choices[item.name])
        if item.name in workflow._PROFILE_DEFAULTS:
            current, benchmark = workflow._PROFILE_DEFAULTS[item.name]
            entry["profile_defaults"] = {
                profile: benchmark if profile in workflow._BENCHMARK_PROFILES else current
                for profile in workflow._CALCULATION_PROFILES
            }
        schema.append(entry)
    return schema


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _pipeline_order() -> Sequence[WorkflowStep]:
    from petrophys.workflow.dynamic_workflow import DynamicWorkflow

    return [step for step, _, _ in DynamicWorkflow().get_pipeline()]


def _field_comments(cls: type) -> Dict[str, str]:
    """Collect ``#:`` comments written directly above dataclass fields."""
    comments: Dict[str, str] = {}
    pending: List[str] = []
    for line in inspect.getsource(cls).splitlines():
        stripped = line.strip()
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        match = re.match(r"^([A-Za-z_]\w*)\s*:", stripped)
        if match and pending:
            text = " ".join(pending)
            comments[match.group(1)] = re.sub(
                r"``([^`]+)``", r"\1", re.sub(r":\w+:`~?([^`]+)`", r"\1", text)
            )
        if stripped and not stripped.startswith("#"):
            pending = []
    return comments


def _field_groups(cls: type) -> Dict[str, str]:
    """Map each dataclass field to the ``# -- Section --`` comment above it."""
    groups: Dict[str, str] = {}
    current = ""
    for line in inspect.getsource(cls).splitlines():
        stripped = line.strip()
        section = re.match(r"^# -- (.+?) -+$", stripped)
        if section:
            current = section.group(1).strip()
            continue
        match = re.match(r"^([A-Za-z_]\w*)\s*:", stripped)
        if match and current:
            groups.setdefault(match.group(1), current)
    return groups


def _field_choices() -> Dict[str, Sequence[str]]:
    from petrophys.interpretation.clay_volume.gr_method import GR_VCL_METHODS
    from petrophys.interpretation.permeability.perm_module import PERMEABILITY_METHODS
    from petrophys.interpretation.saturation.sw_module import SwModel
    from petrophys.model.constants import MatrixProperties
    from petrophys.workflow import dynamic_workflow as workflow

    return {
        "mud_type": workflow._MUD_TYPES,
        "mud_water": workflow._MUD_WATERS,
        "matrix_type": ("auto", *MatrixProperties.names()),
        "lithology_domain": workflow._LITHOLOGY_FAMILIES,
        "lithology_source": workflow._LITHOLOGY_SOURCES,
        "hc_type": workflow._HC_TYPES,
        "swirr_flag_phi_cl": workflow._SWIRR_FLAG_PHI_CL,
        "calculation_profile": workflow._CALCULATION_PROFILES,
        "mineral_solver_version": workflow._MINERAL_SOLVER_VERSIONS,
        "carbonate_calibration": workflow._CARBONATE_CALIBRATIONS,
        "gr_method": GR_VCL_METHODS,
        "perm_method": PERMEABILITY_METHODS,
        "perm_fallback_method": tuple(m for m in PERMEABILITY_METHODS if m != "auto"),
        "sw_model": ("auto", *(model.value for model in SwModel)),
        "skip_steps": tuple(step.value for step in WorkflowStep),
    }


def describe_setting(name: str) -> Optional[Dict[str, Any]]:
    """Return the :func:`config_schema` entry for one setting, or None."""
    return next((item for item in config_schema() if item["name"] == name), None)
