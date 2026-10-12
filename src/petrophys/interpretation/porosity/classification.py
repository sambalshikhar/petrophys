"""
Porosity classification.

This module provides classification of porosity values.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import PorosityClassification


def classify_porosity(
    porosity: Union[float, np.ndarray],
    percent: bool = False,
) -> Union[PorosityClassification, np.ndarray]:
    """
    Classify porosity values.

    Parameters
    ----------
    porosity : float or np.ndarray
        Porosity, a fraction, or percent with ``percent=True``
    percent : bool, default False
        The values are percent. Fractions above 1 raise a
        :class:`~petrophys.exceptions.DataQualityWarning`.

    Returns
    -------
    PorosityClassification or np.ndarray
        Classification result; ``None`` where the value is not finite

    Classification thresholds:
    - EXCELLENT: > 20%
    - GOOD: 12-20%
    - FAIR: 6-12%
    - POOR: <= 6%
    """
    return PorosityClassification.classify(porosity, percent=percent)
