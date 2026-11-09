"""
Petrophys - A comprehensive petrophysics library for well log analysis.

This library provides tools for:
- Curve standardization and quality control
- Clay volume (GR, SP, N-D methods) and clay resistivity
- Lithology determination (multi-mineral solver, crossplots, CSV intervals)
- Porosity calculation (density, neutron, sonic)
- Water saturation modeling (Archie, Simandoux, Indonesia, etc.)
- Permeability estimation
- Net pay determination
- Visualization and reporting
"""

from petrophys.version import __version__
from petrophys.api import read, plot, interpret
from petrophys.io import read_formations
from petrophys.reporting.context import export_context
from petrophys.model.well import Well
from petrophys.datasets import sample, list_samples
from petrophys.workflow import DynamicWorkflowConfig
from petrophys.preprocessing.imputation import impute

__all__ = [
    "read",
    "interpret",
    "plot",
    "impute",
    "export_context",
    "read_formations",
    "sample",
    "list_samples",
    "Well",
    "DynamicWorkflowConfig",
    "__version__",
]
