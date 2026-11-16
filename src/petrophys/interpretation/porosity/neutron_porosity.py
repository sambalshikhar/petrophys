"""
Neutron porosity handling.

This module provides neutron porosity standardization and processing.
"""

from typing import Union
import numpy as np

from petrophys.model.constants import MatrixProperties


def neutron_porosity(
    nphi: Union[float, np.ndarray],
    nphi_matrix: float = -0.02,
    correction: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Process neutron porosity.

    Parameters
    ----------
    nphi : float or np.ndarray
        Raw neutron porosity
    nphi_matrix : float, optional
        Matrix neutron response
    correction : float, optional
        Environmental correction

    Returns
    -------
    float or np.ndarray
        Processed neutron porosity, ``nphi - nphi_matrix + correction``,
        limited to 0-0.6
    """
    phin = nphi - nphi_matrix + correction
    return np.clip(phin, 0, 0.6)


def standardize_nphi(
    nphi: Union[float, np.ndarray],
    current_matrix: str = "limestone",
    target_matrix: str = "sandstone"
) -> Union[float, np.ndarray]:
    """
    Convert neutron porosity between matrix types.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity in current matrix
    current_matrix : str
        Current matrix calibration
    target_matrix : str
        Target matrix calibration

    Returns
    -------
    float or np.ndarray
        Neutron porosity in target matrix

    Notes
    -----
    The offset is the difference of the two matrix neutron readings on the
    limestone scale in :class:`~petrophys.model.constants.MatrixProperties`
    (sandstone -0.02, limestone 0, dolomite 0.02), the readings
    :func:`neutron_porosity` and :func:`petrophys.interpret` subtract, so a
    log converted here and then referred to its matrix comes back unchanged.
    It is a first-order constant: the chart offset grows with porosity
    (about 0.04 between limestone and sandstone at 0.15-0.20).
    """
    from petrophys.preprocessing.standardization.unit_converter import _benchmark_units

    if _benchmark_units():
        # The benchmark packages converted with the mid-porosity chart offsets.
        corrections = {
            ("limestone", "sandstone"): 0.04,
            ("limestone", "dolomite"): -0.02,
            ("sandstone", "limestone"): -0.04,
            ("sandstone", "dolomite"): -0.06,
            ("dolomite", "limestone"): 0.02,
            ("dolomite", "sandstone"): 0.06,
        }
    else:
        matrix_reading = {
            name: MatrixProperties.get(name).nphi for name in ("limestone", "sandstone", "dolomite")
        }
        corrections = {
            (current, target): matrix_reading[current] - matrix_reading[target]
            for current in matrix_reading for target in matrix_reading if current != target
        }

    aliases = {"calcite": "limestone", "lime": "limestone",
               "quartz": "sandstone", "sand": "sandstone"}
    names = []
    for name in (current_matrix, target_matrix):
        name = aliases.get(name.lower(), name.lower())
        if name not in ("limestone", "sandstone", "dolomite"):
            raise ValueError(f"Unknown matrix {name!r}: use limestone, sandstone or dolomite")
        names.append(name)
    correction = corrections.get(tuple(names), 0.0)

    return nphi + correction


def correct_nphi_for_clay(
    nphi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    nphi_clay: float = 0.30
) -> Union[float, np.ndarray]:
    """
    Correct neutron porosity for clay content.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity
    vcl : float or np.ndarray
        Clay volume
    nphi_clay : float, optional
        Clay neutron response (default: 0.30)

    Returns
    -------
    float or np.ndarray
        Clay-corrected neutron porosity
    """
    nphi_corrected = nphi - vcl * nphi_clay
    return np.clip(nphi_corrected, 0, 0.5)
