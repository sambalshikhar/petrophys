"""
Clay volume classification.

This module provides classification of clay volume values
into formation quality categories, with the canonical VCL table
(``CLASS_TABLES["VCL"]`` in :mod:`petrophys.model.class_tables`).
"""

from typing import Optional, Union
import numpy as np

from petrophys.model.classifications import VclClassification

#: Quality code of each clay-volume class; a missing value has code 0.
_QUALITY_CODES = {
    VclClassification.CLEAN_FORMATION: 1,
    VclClassification.MODERATE_CLAY: 2,
    VclClassification.SHALY_FORMATION: 3,
}


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
        Classification result(s); ``None`` where the value is not finite

    Classification thresholds:
    - CLEAN_FORMATION: Vcl < 0.25
    - MODERATE_CLAY: 0.25 ≤ Vcl ≤ 0.50
    - SHALY_FORMATION: Vcl > 0.50
    """
    return VclClassification.classify(vcl)


def _is_class(vcl, member: VclClassification) -> Union[bool, np.ndarray]:
    classes = VclClassification.classify(vcl)
    if isinstance(classes, np.ndarray):
        return np.array([item is member for item in classes.ravel()], dtype=bool).reshape(classes.shape)
    return classes is member


def is_clean_sand(vcl: Union[float, np.ndarray], threshold: Optional[float] = None) -> Union[bool, np.ndarray]:
    """
    Check if formation is clean.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume
    threshold : float, optional
        Clean rock threshold: clean below it. Default: the CLEAN_FORMATION
        class of the canonical VCL table (Vcl < 0.25).

    Returns
    -------
    bool or np.ndarray
        True if clean rock (False where the value is not finite)
    """
    if threshold is None:
        return _is_class(vcl, VclClassification.CLEAN_FORMATION)
    return vcl < threshold


def is_shale(vcl: Union[float, np.ndarray], threshold: Optional[float] = None) -> Union[bool, np.ndarray]:
    """
    Check if formation is rich in clay.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume
    threshold : float, optional
        Argillaceous threshold: clay-rich above it. Default: the
        SHALY_FORMATION class of the canonical VCL table (Vcl > 0.50).

    Returns
    -------
    bool or np.ndarray
        True if clay-rich (False where the value is not finite)
    """
    if threshold is None:
        return _is_class(vcl, VclClassification.SHALY_FORMATION)
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
        - 0: No value (NaN)
        - 1: Clean (< 0.25)
        - 2: Moderate (0.25-0.50)
        - 3: Clay-rich (> 0.50)
    """
    classes = VclClassification.classify(vcl)
    if isinstance(classes, np.ndarray):
        codes = [_QUALITY_CODES.get(item, 0) for item in classes.ravel()]
        return np.array(codes, dtype=int).reshape(classes.shape)
    return _QUALITY_CODES.get(classes, 0)
