"""
Invasion profile classification.

This module provides classification of invasion profiles
based on Rxo/Rt ratio.
"""

from typing import Union
from enum import Enum
import numpy as np

from petrophys.core.classifications import InvasionClassification


class InvasionType(Enum):
    """Invasion type enumeration."""
    CONDUCTIVE = "conductive"
    MINIMAL = "minimal"
    RESISTIVE = "resistive"


def classify_invasion(
    ratio: Union[float, np.ndarray]
) -> Union[InvasionClassification, np.ndarray]:
    """
    Classify invasion profile from Rxo/Rt ratio.

    Parameters
    ----------
    ratio : float or np.ndarray
        Rxo/Rt ratio

    Returns
    -------
    InvasionClassification or np.ndarray
        Classification result

    Classification thresholds:
    - CONDUCTIVE_INVASION: Ratio < 0.8
    - MINIMAL_INVASION: 0.8 ≤ Ratio ≤ 1.2
    - RESISTIVE_INVASION: Ratio > 1.2
    """
    return InvasionClassification.classify(ratio)


def get_invasion_type(ratio: float) -> InvasionType:
    """
    Get invasion type from ratio.

    Parameters
    ----------
    ratio : float
        Rxo/Rt ratio

    Returns
    -------
    InvasionType
        Invasion type
    """
    if ratio < 0.8:
        return InvasionType.CONDUCTIVE
    elif ratio > 1.2:
        return InvasionType.RESISTIVE
    else:
        return InvasionType.MINIMAL


def interpret_invasion(ratio: float) -> str:
    """
    Interpret invasion profile.

    Parameters
    ----------
    ratio : float
        Rxo/Rt ratio

    Returns
    -------
    str
        Interpretation text
    """
    if ratio < 0.8:
        return (
            "Conductive invasion: Mud filtrate is more conductive than "
            "formation water. May indicate oil/gas in uninvaded zone."
        )
    elif ratio > 1.2:
        return (
            "Resistive invasion: Mud filtrate is more resistive than "
            "formation water. May indicate high salinity formation water."
        )
    else:
        return (
            "Minimal invasion: Little contrast between flushed and "
            "uninvaded zones. Rmf ≈ Rw."
        )
