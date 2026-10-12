"""
Invasion profile classification.

This module provides classification of invasion profiles
based on Rxo/Rt ratio.
"""

from typing import Optional, Union
from enum import Enum
import numpy as np

from petrophys.model.classifications import InvasionClassification


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

    Classification thresholds (``CLASS_TABLES["RXO_RT"]``; ``None`` where
    the ratio is not finite):
    - CONDUCTIVE_INVASION: Ratio < 0.8
    - MINIMAL_INVASION: 0.8 ≤ Ratio ≤ 1.2
    - RESISTIVE_INVASION: Ratio > 1.2
    """
    return InvasionClassification.classify(ratio)


_INVASION_TYPES = {
    InvasionClassification.CONDUCTIVE_INVASION: InvasionType.CONDUCTIVE,
    InvasionClassification.MINIMAL_INVASION: InvasionType.MINIMAL,
    InvasionClassification.RESISTIVE_INVASION: InvasionType.RESISTIVE,
}


def get_invasion_type(ratio: float) -> Optional[InvasionType]:
    """
    Get invasion type from ratio.

    Parameters
    ----------
    ratio : float
        Rxo/Rt ratio

    Returns
    -------
    InvasionType or None
        Invasion type, from the classes of :func:`classify_invasion`; None
        when the ratio is not finite
    """
    return _INVASION_TYPES.get(classify_invasion(ratio))


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
    invasion_type = get_invasion_type(ratio)
    if invasion_type is None:
        return "No valid Rxo/Rt ratio: the invasion profile cannot be interpreted."
    if invasion_type is InvasionType.CONDUCTIVE:
        return (
            "Conductive invasion: Mud filtrate is more conductive than "
            "formation water. May indicate oil/gas in uninvaded zone."
        )
    elif invasion_type is InvasionType.RESISTIVE:
        return (
            "Resistive invasion: Mud filtrate is more resistive than "
            "formation water. May indicate high salinity formation water."
        )
    else:
        return (
            "Minimal invasion: Little contrast between flushed and "
            "uninvaded zones. Rmf ≈ Rw."
        )
