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
    "spectral_gamma": "Clay indicators from potassium, thorium and uranium",
    "elemental_spectroscopy": "Mineral indicators from elemental concentrations or yields",
    "vclay": "Clay volume (VCL) from gamma ray, SP and neutron-density",
    "rcl": "Clay resistivity from shale intervals",
    "rmf": "Mud-filtrate resistivity at formation temperature",
    "hc_correction": "Neutron-density hydrocarbon correction (opt-in)",
    "porosity": "Density, neutron, sonic, total (PHIT) and effective (PHIE) porosity",
    "lithology": "Mineral volumes from an adaptive multi-mineral solve",
    "porosity_refinement": "Porosity recomputed with per-depth lithology endpoints",
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
    """Return the workflow steps in execution order with a description each.

    Returns
    -------
    list of dict
        ``{"name": ..., "description": ...}`` for every step, in the order
        :func:`petrophys.interpret` runs them.

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
        One entry per setting, in declaration order, with ``name``, ``type``
        (as written in the source), ``default`` (JSON-serializable),
        ``description`` and, for settings with a fixed set of values,
        ``choices``.

    Examples
    --------
    >>> from petrophys.workflow.schema import config_schema
    >>> rw = next(item for item in config_schema() if item["name"] == "rw")
    >>> rw["description"].startswith("Formation water resistivity")
    True
    """
    descriptions = _field_comments(DynamicWorkflowConfig)
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
            "type": _type_name(item.type),
            "default": default,
            "description": descriptions.get(item.name, ""),
        }
        if item.name in choices:
            entry["choices"] = list(choices[item.name])
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
        "gr_method": GR_VCL_METHODS,
        "perm_method": PERMEABILITY_METHODS,
        "perm_fallback_method": tuple(m for m in PERMEABILITY_METHODS if m != "auto"),
        "sw_model": ("auto", *(model.value for model in SwModel)),
        "skip_steps": tuple(step.value for step in WorkflowStep),
    }


def describe_setting(name: str) -> Optional[Dict[str, Any]]:
    """Return the :func:`config_schema` entry for one setting, or None."""
    return next((item for item in config_schema() if item["name"] == name), None)
