"""
Clay volume classification.

This module provides classification of clay volume values
into formation quality categories.
"""

from typing import Union
import numpy as np

from petrophys.core.classifications import VclClassification


def classify_vcl(vcl: Union[float, np.ndarray]) -> Union[VclClassification, np.ndarray]:
    """
    Classify clay volume values.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume (0-1)

    Returns
    -------
    VclClassification or np.ndarray
        Classification result(s)

    Classification thresholds:
    - CLEAN_FORMATION: Vcl < 0.25
    - MODERATE_CLAY: 0.25 ≤ Vcl ≤ 0.50
    - SHALY_FORMATION: Vcl > 0.50
    """
    return VclClassification.classify(vcl)


def is_clean_sand(vcl: Union[float, np.ndarray], threshold: float = 0.25) -> Union[bool, np.ndarray]:
    """
    Check if formation is clean sand.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume
    threshold : float, optional
        Clean sand threshold (default: 0.25)

    Returns
    -------
    bool or np.ndarray
        True if clean sand
    """
    return vcl < threshold


def is_shale(vcl: Union[float, np.ndarray], threshold: float = 0.50) -> Union[bool, np.ndarray]:
    """
    Check if formation is shale.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume
    threshold : float, optional
        Shale threshold (default: 0.50)

    Returns
    -------
    bool or np.ndarray
        True if shale
    """
    return vcl > threshold


def get_vcl_quality_code(vcl: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Get numeric quality code for VCL.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume

    Returns
    -------
    int or np.ndarray
        Quality code:
        - 1: Clean (< 0.25)
        - 2: Moderate (0.25-0.50)
        - 3: Shaly (> 0.50)
    """
    if isinstance(vcl, np.ndarray):
        code = np.ones_like(vcl, dtype=int)
        code[(vcl >= 0.25) & (vcl <= 0.50)] = 2
        code[vcl > 0.50] = 3
        return code
    else:
        if vcl < 0.25:
            return 1
        elif vcl <= 0.50:
            return 2
        else:
            return 3
