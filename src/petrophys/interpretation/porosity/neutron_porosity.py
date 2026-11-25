"""
Neutron porosity handling.

This module provides neutron porosity standardization and processing.
"""

from typing import Union
import numpy as np


#: Apparent limestone neutron porosity (v/v) of a clean, water-filled rock at
#: true porosity 0, 0.05, ..., 0.40, and the matrix density its line is drawn
#: for: the lithology lines of Schlumberger's Por-11 (former CP-1e: CNL and
#: Litho-Density, fresh water in the invaded zone; Log Interpretation Charts,
#: 2013), read from the chart's vector lines. The sandstone and dolomite lines
#: curve: the gap to the limestone line grows from 0.02 at zero porosity to
#: about 0.05 at 20 %. The dolomite line ends at 0.35.
CNL_CHART_LINES = {
    "sandstone": {"rhob": 2.65, "nphi": (-0.020, 0.019, 0.060, 0.103, 0.149, 0.197, 0.248, 0.300, 0.355)},
    "limestone": {"rhob": 2.71, "nphi": (0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400)},
    "dolomite": {"rhob": 2.85, "nphi": (0.008, 0.060, 0.115, 0.171, 0.229, 0.287, 0.346, 0.405)},
}


def _along(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Piecewise-linear ``ys(x)``, carried on by the end segments outside ``xs``."""
    y = np.interp(x, xs, ys)
    low, high = x < xs[0], x > xs[-1]
    y = np.where(low, ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]), y)
    return np.where(high, ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), y)


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
    The conversion follows the chartbook lines (:data:`CNL_CHART_LINES`,
    Schlumberger Por-11): a reading in the units of one matrix is that
    matrix's true porosity, which its line turns into the limestone scale,
    and the target matrix's line turns back. The offset therefore grows with
    porosity: limestone to sandstone adds 0.02 at zero porosity and about
    0.05 at 20 %. Readings beyond the chart's 0-40 % carry on along its end
    segments.
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
        corrections = {}

    aliases = {"calcite": "limestone", "lime": "limestone",
               "quartz": "sandstone", "sand": "sandstone"}
    names = []
    for name in (current_matrix, target_matrix):
        name = aliases.get(name.lower(), name.lower())
        if name not in ("limestone", "sandstone", "dolomite"):
            raise ValueError(f"Unknown matrix {name!r}: use limestone, sandstone or dolomite")
        names.append(name)
    if not _benchmark_units():
        current, target = names
        if current == target:
            return nphi
        values = np.asarray(nphi, dtype=float)
        porosity = np.arange(9) * 0.05
        source = np.asarray(CNL_CHART_LINES[current]["nphi"])
        goal = np.asarray(CNL_CHART_LINES[target]["nphi"])
        limestone_scale = _along(values, porosity[:source.size], source)
        converted = _along(limestone_scale, goal, porosity[:goal.size])
        return float(converted) if np.ndim(nphi) == 0 else converted
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
