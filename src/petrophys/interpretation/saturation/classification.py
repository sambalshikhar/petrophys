"""
Water saturation classification.

This module provides classification of water saturation values.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import HcSaturationClassification


def classify_sw(
    sw: Union[float, np.ndarray]
) -> Union[HcSaturationClassification, np.ndarray]:
    """
    Classify water saturation.

    Parameters
    ----------
    sw : float or np.ndarray
        Water saturation

    Returns
    -------
    HcSaturationClassification or np.ndarray
        Classification result
    """
    return HcSaturationClassification.classify(sw)
