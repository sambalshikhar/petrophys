"""
Salinity-based Rw calculation.

This module provides formation water resistivity calculation
from salinity measurements.
"""

from typing import Union
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.units import UnitConverter, temperature_unit

#: Mass of NaCl per unit mass of chloride (molar masses 58.44 and 35.45 g/mol):
#: a chloride concentration times this factor is its NaCl equivalent.
NACL_PER_CHLORIDE = 58.44 / 35.45


def brine_density(salinity_ppm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Density of an NaCl brine at standard conditions (60 degF, 14.7 psia).

    McCain (1991): rho = 62.368 + 0.438603 S + 1.60074e-3 S^2 lbm/ft3, with S
    the dissolved solids in weight percent.

    Parameters
    ----------
    salinity_ppm : float or np.ndarray
        Salinity, ppm by weight (mg/kg)

    Returns
    -------
    float or np.ndarray
        Brine density, g/cm3
    """
    s = np.asarray(salinity_ppm, dtype=float) / 1.0e4
    density = (62.368 + 0.438603 * s + 1.60074e-3 * s ** 2) / 62.428
    return float(density) if density.ndim == 0 else density


def salinity_mg_per_l_to_ppm(salinity_mg_l: float) -> float:
    """
    Convert a salinity in mg/l to ppm by weight.

    ppm = (mg/l) / rho, with rho in g/cm3 from :func:`brine_density`. The
    density depends on the ppm value itself, so the division is iterated.

    Parameters
    ----------
    salinity_mg_l : float
        Dissolved salt, mg per litre of solution

    Returns
    -------
    float
        Salinity, ppm by weight
    """
    concentration = float(salinity_mg_l)
    ppm = concentration
    for _ in range(50):
        updated = concentration / brine_density(ppm)
        if abs(updated - ppm) <= 1e-9 * max(abs(updated), 1.0):
            return updated
        ppm = updated
    return ppm


def chlorides_to_nacl_ppm(chlorides_mg_l: float) -> float:
    """
    NaCl-equivalent salinity of a chloride concentration.

    Mud reports give filtrate chlorides (Cl-) in mg/l, titrated per API RP
    13B-1; :func:`rw_from_salinity` and the resistivity charts take NaCl ppm
    by weight. The chloride is counted as NaCl (x :data:`NACL_PER_CHLORIDE`,
    about 1.65) and converted to ppm with :func:`salinity_mg_per_l_to_ppm`.

    Parameters
    ----------
    chlorides_mg_l : float
        Chloride ion concentration, mg/l

    Returns
    -------
    float
        Salinity, ppm NaCl equivalent

    Examples
    --------
    >>> round(chlorides_to_nacl_ppm(30000.0))
    47864
    """
    return salinity_mg_per_l_to_ppm(NACL_PER_CHLORIDE * float(chlorides_mg_l))


def rw_from_salinity(
    salinity_ppm: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from salinity using empirical relationship.

    Uses the Arps formula variation for NaCl solutions.

    Parameters
    ----------
    salinity_ppm : float or np.ndarray
        Salinity in ppm NaCl equivalent
    temperature : float or np.ndarray
        Temperature
    temp_unit : str, optional
        Temperature unit: "F", "C", "K" or a label such as "degC"; unknown
        labels raise ``ValueError``

    Returns
    -------
    float or np.ndarray
        Formation water resistivity (ohm.m). Zero or negative salinity gives
        NaN.
    """
    # Convert to Fahrenheit
    if temperature_unit(temp_unit, strict=True) != "F":
        temperature = UnitConverter.temperature(temperature, temp_unit, "F")

    # Zero or negative salinity has no Rw. The benchmark context profiles keep
    # the packages' original calculation.
    if not reproducing_benchmark():
        salinity_ppm = np.asarray(salinity_ppm, dtype=float)
        salinity_ppm = np.where(salinity_ppm > 0, salinity_ppm, np.nan)
        if salinity_ppm.ndim == 0:
            salinity_ppm = float(salinity_ppm)
    elif isinstance(salinity_ppm, np.ndarray):
        salinity_ppm = np.where(salinity_ppm <= 0, 1, salinity_ppm)
    else:
        salinity_ppm = max(salinity_ppm, 1)

    # Empirical formula (based on Schlumberger Gen-9)
    # Rw at 75°F: Rw75 = 0.0123 + 3647.5 / salinity^0.955

    rw_75 = 0.0123 + 3647.5 / np.power(salinity_ppm, 0.955)

    # Correct to formation temperature using Arps
    rw = rw_75 * (75 + 6.77) / (temperature + 6.77)

    return rw


def salinity_from_rw(
    rw: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate salinity from Rw.

    Inverse of rw_from_salinity.

    Parameters
    ----------
    rw : float or np.ndarray
        Formation water resistivity (ohm.m)
    temperature : float or np.ndarray
        Temperature
    temp_unit : str, optional
        Temperature unit: "F", "C", "K" or a label such as "degC"; unknown
        labels raise ``ValueError``

    Returns
    -------
    float or np.ndarray
        Salinity in ppm NaCl. Missing Rw, and Rw at or below the 0.0123
        ohm.m (at 75 degF) asymptote of the fit, give NaN.
    """
    # Convert to Fahrenheit
    if temperature_unit(temp_unit, strict=True) != "F":
        temperature = UnitConverter.temperature(temperature, temp_unit, "F")

    # Correct Rw to 75°F
    rw_75 = rw * (temperature + 6.77) / (75 + 6.77)

    # Solve for salinity
    # Rw75 = 0.0123 + 3647.5 / salinity^0.955
    # salinity^0.955 = 3647.5 / (Rw75 - 0.0123)
    # salinity = (3647.5 / (Rw75 - 0.0123))^(1/0.955)

    if not reproducing_benchmark():
        # Rw at or below 0.0123 ohm.m (75 F) has no salinity. The benchmark
        # context profiles keep the packages' original calculation.
        rw_75 = np.asarray(rw_75, dtype=float)
        with np.errstate(invalid="ignore"):
            denominator = np.where(rw_75 > 0.0123, rw_75 - 0.0123, np.nan)
        if denominator.ndim == 0:
            denominator = float(denominator)
    elif isinstance(rw_75, np.ndarray):
        denominator = np.where(rw_75 > 0.0123, rw_75 - 0.0123, 1e-10)
    else:
        denominator = max(rw_75 - 0.0123, 1e-10)

    salinity = np.power(3647.5 / denominator, 1 / 0.955)

    return salinity


def classify_water_salinity(salinity_ppm: Union[float, np.ndarray]) -> str:
    """
    Classify water based on salinity.

    Parameters
    ----------
    salinity_ppm : float or np.ndarray
        Salinity in ppm

    Returns
    -------
    str
        Classification ("fresh", "brackish", "saline", "brine"), or
        "unknown" for a missing salinity
    """
    if isinstance(salinity_ppm, np.ndarray):
        finite = salinity_ppm[np.isfinite(salinity_ppm)]
        salinity_ppm = float(np.mean(finite)) if finite.size else np.nan

    if not np.isfinite(salinity_ppm):
        return "unknown"
    if salinity_ppm < 1000:
        return "fresh"
    elif salinity_ppm < 10000:
        return "brackish"
    elif salinity_ppm < 100000:
        return "saline"
    else:
        return "brine"
