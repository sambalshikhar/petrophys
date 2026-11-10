"""
Water saturation classification.

This module provides classification of water saturation values.
"""

from typing import Union
import numpy as np

from petrophys.model.classifications import HcSaturationClassification


def classify_sw(
    sw: Union[float, np.ndarray],
    percent: bool = False,
) -> Union[HcSaturationClassification, np.ndarray]:
    """
    Classify water saturation.

    Parameters
    ----------
    sw : float or np.ndarray
        Water saturation, a fraction, or percent with ``percent=True``
    percent : bool, default False
        The values are percent. Without it, values above 2 are still read
        as fractions and raise a
        :class:`~petrophys.exceptions.DataQualityWarning`.

    Returns
    -------
    HcSaturationClassification or np.ndarray
        Classification result; ``None`` where the value is not finite.
        Thresholds: EXCELLENT_HC < 0.20 <= GOOD_HC < 0.40 <= FAIR_HC < 0.60
        <= WATER_BEARING.
    """
    return HcSaturationClassification.classify(sw, percent=percent)
