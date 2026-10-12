"""
Classification enumerations for petrophysical properties.

This module defines all classification systems used for categorizing
petrophysical results including clay volume, porosity, saturation,
permeability, invasion, and moveable hydrocarbon index.

Each enumeration names the classes of one canonical table of
:data:`~petrophys.model.class_tables.CLASS_TABLES` and classifies through it,
so it agrees with the diagnostics analyzers. Scalars and arrays take the same
code path: a scalar gives a member, an array (or list) an object array of
members, and a value that is not finite gives ``None``. Values are fractions;
porosity and water saturation take ``percent=True`` for percent values.
"""

import warnings
from enum import Enum
from typing import Dict, Union

import numpy as np

from petrophys.exceptions import DataQualityWarning
from petrophys.model.class_tables import CLASS_TABLES


def _classify(table: str, values, members: Dict[str, Enum]):
    """The member of every value, from the canonical table ``table``.

    ``members`` maps each label of the table to its member; a value without a
    valid label gets ``None``.
    """
    x = np.asarray(values, dtype=float)
    labels = CLASS_TABLES[table].bands.classify(np.atleast_1d(x))
    result = np.full(labels.shape, None, dtype=object)
    for label, member in members.items():
        result[labels == label] = member
    if x.ndim == 0:
        return result[0]
    return result.reshape(x.shape)


def _as_fraction(values, percent: bool, quantity: str, fraction_limit: float):
    """Values as fractions: divided by 100 when ``percent``.

    Warns when fractions exceed ``fraction_limit``, which usually means the
    values are percent.
    """
    x = np.asarray(values, dtype=float)
    if percent:
        return x / 100.0
    finite = x[np.isfinite(x)]
    if finite.size and float(np.max(finite)) > fraction_limit:
        warnings.warn(
            f"{quantity} values above {fraction_limit:g} were classified as fractions; "
            "pass percent=True if they are percent",
            DataQualityWarning,
            stacklevel=3,
        )
    return x


class VclClassification(Enum):
    """
    Clay volume classification.

    Attributes
    ----------
    CLEAN_FORMATION : str
        Vcl < 0.25 - Clean sand/carbonate
    MODERATE_CLAY : str
        0.25 <= Vcl <= 0.50 - Moderate clay content
    SHALY_FORMATION : str
        Vcl > 0.50 - Shaly formation
    """

    CLEAN_FORMATION = "clean"
    MODERATE_CLAY = "moderate"
    SHALY_FORMATION = "shaly"

    @classmethod
    def classify(cls, vcl: Union[float, np.ndarray]) -> Union["VclClassification", np.ndarray, None]:
        """
        Classify clay volume value(s) with ``CLASS_TABLES["VCL"]``.

        Parameters
        ----------
        vcl : float or array-like
            Clay volume fraction (0-1)

        Returns
        -------
        VclClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        return _classify("VCL", vcl, _VCL_MEMBERS)


class InvasionClassification(Enum):
    """
    Invasion profile classification based on Rxo/Rt ratio.

    Attributes
    ----------
    CONDUCTIVE_INVASION : str
        Ratio < 0.8 - Filtrate more conductive than formation water
    MINIMAL_INVASION : str
        0.8 <= Ratio <= 1.2 - Minimal invasion effect
    RESISTIVE_INVASION : str
        Ratio > 1.2 - Filtrate more resistive than formation water
    """

    CONDUCTIVE_INVASION = "conductive"
    MINIMAL_INVASION = "minimal"
    RESISTIVE_INVASION = "resistive"

    @classmethod
    def classify(cls, ratio: Union[float, np.ndarray]) -> Union["InvasionClassification", np.ndarray, None]:
        """
        Classify invasion ratio value(s) with ``CLASS_TABLES["RXO_RT"]``.

        Parameters
        ----------
        ratio : float or array-like
            Rxo/Rt ratio

        Returns
        -------
        InvasionClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        return _classify("RXO_RT", ratio, _INVASION_MEMBERS)


class PorosityClassification(Enum):
    """
    Porosity classification.

    Attributes
    ----------
    EXCELLENT : str
        Porosity > 20%
    GOOD : str
        12% < Porosity <= 20%
    FAIR : str
        6% < Porosity <= 12%
    POOR : str
        Porosity <= 6%
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    @classmethod
    def classify(
        cls,
        porosity: Union[float, np.ndarray],
        percent: bool = False,
    ) -> Union["PorosityClassification", np.ndarray, None]:
        """
        Classify porosity value(s) with ``CLASS_TABLES["PHI"]``.

        Parameters
        ----------
        porosity : float or array-like
            Porosity, a fraction (0-1), or percent (0-100) with ``percent=True``
        percent : bool, default False
            The values are percent. Fractions above 1 raise a
            :class:`~petrophys.exceptions.DataQualityWarning`, since they
            usually are percent.

        Returns
        -------
        PorosityClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        phi = _as_fraction(porosity, percent, "Porosity", 1.0)
        return _classify("PHI", phi, _POROSITY_MEMBERS)


class HcSaturationClassification(Enum):
    """
    Hydrocarbon saturation classification based on water saturation.

    Attributes
    ----------
    EXCELLENT_HC : str
        Sw < 20% - Excellent hydrocarbon saturation
    GOOD_HC : str
        20% <= Sw < 40% - Good hydrocarbon saturation
    FAIR_HC : str
        40% <= Sw < 60% - Fair hydrocarbon saturation
    WATER_BEARING : str
        Sw >= 60% - Water bearing
    """

    EXCELLENT_HC = "excellent"
    GOOD_HC = "good"
    FAIR_HC = "fair"
    WATER_BEARING = "water"

    @classmethod
    def classify(
        cls,
        sw: Union[float, np.ndarray],
        percent: bool = False,
    ) -> Union["HcSaturationClassification", np.ndarray, None]:
        """
        Classify water saturation value(s) with ``CLASS_TABLES["SW"]``.

        A fraction slightly above 1 (e.g. 1.02 from an unclipped Sw) is
        water-bearing.

        Parameters
        ----------
        sw : float or array-like
            Water saturation, a fraction (0-1), or percent (0-100) with
            ``percent=True``
        percent : bool, default False
            The values are percent. Fractions above 2 raise a
            :class:`~petrophys.exceptions.DataQualityWarning`, since they
            usually are percent.

        Returns
        -------
        HcSaturationClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        sat = _as_fraction(sw, percent, "Water saturation", 2.0)
        return _classify("SW", sat, _SATURATION_MEMBERS)


class PermeabilityClassification(Enum):
    """
    Permeability classification.

    Attributes
    ----------
    EXCELLENT : str
        Permeability > 1000 mD
    VERY_GOOD : str
        100 mD < Permeability <= 1000 mD
    GOOD : str
        10 mD < Permeability <= 100 mD
    FAIR : str
        1 mD < Permeability <= 10 mD
    POOR : str
        0.1 mD < Permeability <= 1 mD
    VERY_POOR : str
        Permeability <= 0.1 mD
    """

    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

    @classmethod
    def classify(cls, perm: Union[float, np.ndarray]) -> Union["PermeabilityClassification", np.ndarray, None]:
        """
        Classify permeability value(s) in mD with ``CLASS_TABLES["PERM"]``.

        Parameters
        ----------
        perm : float or array-like
            Permeability in millidarcies (mD)

        Returns
        -------
        PermeabilityClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        return _classify("PERM", perm, _PERMEABILITY_MEMBERS)


class RQIClassification(Enum):
    """Reservoir Quality Index classification for RQI values in microns."""

    HIGH_RQI = "high"
    MODERATE_RQI = "moderate"
    LOW_RQI = "low"

    @classmethod
    def classify(
        cls,
        rqi: Union[float, np.ndarray],
    ) -> Union["RQIClassification", np.ndarray, None]:
        """
        Classify Reservoir Quality Index value(s) with ``CLASS_TABLES["RQI"]``.

        ``HIGH_RQI`` is greater than 1.0 micron, ``MODERATE_RQI`` is
        0.5-1.0 microns inclusive, and ``LOW_RQI`` is below 0.5 micron.
        Missing values remain unclassified.
        """
        return _classify("RQI", rqi, _RQI_MEMBERS)


class MhiClassification(Enum):
    """
    Moveable Hydrocarbon Index (MHI) classification.

    MHI = Sw / Sxo.

    Attributes
    ----------
    HIGH_MOVEABLE_HC : str
        MHI < 0.5 - High moveable hydrocarbon
    MODERATE_MOVEABLE_HC : str
        0.5 <= MHI < 0.7 - Moderate moveable hydrocarbon
    LOW_MOVEABLE_HC : str
        0.7 <= MHI < 0.9 - Low moveable hydrocarbon
    MINIMAL_MOVEABLE_HC : str
        MHI >= 0.9 - Minimal moveable hydrocarbon
    """

    HIGH_MOVEABLE_HC = "high"
    MODERATE_MOVEABLE_HC = "moderate"
    LOW_MOVEABLE_HC = "low"
    MINIMAL_MOVEABLE_HC = "minimal"

    @classmethod
    def classify(cls, mhi: Union[float, np.ndarray]) -> Union["MhiClassification", np.ndarray, None]:
        """
        Classify MHI value(s) with ``CLASS_TABLES["MHI"]``.

        Parameters
        ----------
        mhi : float or array-like
            Moveable hydrocarbon index (0-1)

        Returns
        -------
        MhiClassification or np.ndarray
            Classification result(s); ``None`` where the value is not finite
        """
        return _classify("MHI", mhi, _MHI_MEMBERS)


# The member of each label of the canonical tables.
_VCL_MEMBERS = {
    "CLEAN_FORMATION": VclClassification.CLEAN_FORMATION,
    "MODERATE_CLAY": VclClassification.MODERATE_CLAY,
    "SHALY_FORMATION": VclClassification.SHALY_FORMATION,
}
_INVASION_MEMBERS = {
    "CONDUCTIVE_INVASION": InvasionClassification.CONDUCTIVE_INVASION,
    "MINIMAL_INVASION": InvasionClassification.MINIMAL_INVASION,
    "RESISTIVE_INVASION": InvasionClassification.RESISTIVE_INVASION,
}
_POROSITY_MEMBERS = {
    "POOR_POROSITY": PorosityClassification.POOR,
    "FAIR_POROSITY": PorosityClassification.FAIR,
    "GOOD_POROSITY": PorosityClassification.GOOD,
    "EXCELLENT_POROSITY": PorosityClassification.EXCELLENT,
}
_SATURATION_MEMBERS = {
    "EXCELLENT_HC_SATURATION": HcSaturationClassification.EXCELLENT_HC,
    "GOOD_HC_SATURATION": HcSaturationClassification.GOOD_HC,
    "FAIR_HC_SATURATION": HcSaturationClassification.FAIR_HC,
    "WATER_BEARING": HcSaturationClassification.WATER_BEARING,
}
_PERMEABILITY_MEMBERS = {
    "VERY_POOR_PERMEABILITY": PermeabilityClassification.VERY_POOR,
    "POOR_PERMEABILITY": PermeabilityClassification.POOR,
    "FAIR_PERMEABILITY": PermeabilityClassification.FAIR,
    "GOOD_PERMEABILITY": PermeabilityClassification.GOOD,
    "VERY_GOOD_PERMEABILITY": PermeabilityClassification.VERY_GOOD,
    "EXCELLENT_PERMEABILITY": PermeabilityClassification.EXCELLENT,
}
_RQI_MEMBERS = {
    "LOW_RQI": RQIClassification.LOW_RQI,
    "MODERATE_RQI": RQIClassification.MODERATE_RQI,
    "HIGH_RQI": RQIClassification.HIGH_RQI,
}
_MHI_MEMBERS = {
    "HIGH_MOVEABLE_HC": MhiClassification.HIGH_MOVEABLE_HC,
    "MODERATE_MOVEABLE_HC": MhiClassification.MODERATE_MOVEABLE_HC,
    "LOW_MOVEABLE_HC": MhiClassification.LOW_MOVEABLE_HC,
    "MINIMAL_MOVEABLE_HC": MhiClassification.MINIMAL_MOVEABLE_HC,
}
