"""
Log Normalization Module.

This module provides tools for normalizing well logs to ensure
consistency across multi-well datasets.
"""

from petrophys.preprocessing.normalization.normalization_module import (
    NormalizationModule,
    normalize_log,
    linear_normalization,
)
from petrophys.preprocessing.normalization.histogram_matching import (
    histogram_matching,
    histogram_equalization,
)

__all__ = [
    "NormalizationModule",
    "normalize_log",
    "linear_normalization",
    "histogram_matching",
    "histogram_equalization",
]
