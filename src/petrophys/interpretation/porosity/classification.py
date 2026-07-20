"""
Porosity classification.

This module provides classification of porosity values.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import PorosityClassification


def classify_porosity(
    porosity: Union[float, np.ndarray]
) -> Union[PorosityClassification, np.ndarray]:
    """
    Classify porosity values.

    Parameters
    ----------
    porosity : float or np.ndarray
        Porosity (fraction or percentage)

    Returns
    -------
    PorosityClassification or np.ndarray
        Classification result

    Classification thresholds:
    - EXCELLENT: > 20%
    - GOOD: 12-20%
    - FAIR: 6-12%
    - POOR: <= 6%
    """
    return PorosityClassification.classify(porosity)
