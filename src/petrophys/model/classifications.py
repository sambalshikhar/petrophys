"""
Classification enumerations for petrophysical properties.

This module defines all classification systems used for categorizing
petrophysical results including clay volume, porosity, saturation,
permeability, invasion, and moveable hydrocarbon index.
"""

from enum import Enum
from typing import Union
import numpy as np


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
    def classify(cls, vcl: Union[float, np.ndarray]) -> Union["VclClassification", np.ndarray]:
        """
        Classify clay volume value(s).

        Parameters
        ----------
        vcl : float or np.ndarray
            Clay volume fraction (0-1)

        Returns
        -------
        VclClassification or np.ndarray
            Classification result(s)
        """
        if isinstance(vcl, np.ndarray):
            result = np.empty(vcl.shape, dtype=object)
            result[vcl < 0.25] = cls.CLEAN_FORMATION
            result[(vcl >= 0.25) & (vcl <= 0.50)] = cls.MODERATE_CLAY
            result[vcl > 0.50] = cls.SHALY_FORMATION
            return result

        if np.isnan(vcl):
            return None  # match the array path, where NaN yields None
        if vcl < 0.25:
            return cls.CLEAN_FORMATION
        elif vcl <= 0.50:
            return cls.MODERATE_CLAY
        else:
            return cls.SHALY_FORMATION


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
    def classify(cls, ratio: Union[float, np.ndarray]) -> Union["InvasionClassification", np.ndarray]:
        """
        Classify invasion ratio value(s).

        Parameters
        ----------
        ratio : float or np.ndarray
            Rxo/Rt ratio

        Returns
        -------
        InvasionClassification or np.ndarray
            Classification result(s)
        """
        if isinstance(ratio, np.ndarray):
            result = np.empty(ratio.shape, dtype=object)
            result[ratio < 0.8] = cls.CONDUCTIVE_INVASION
            result[(ratio >= 0.8) & (ratio <= 1.2)] = cls.MINIMAL_INVASION
            result[ratio > 1.2] = cls.RESISTIVE_INVASION
            return result

        if np.isnan(ratio):
            return None  # match the array path, where NaN yields None
        if ratio < 0.8:
            return cls.CONDUCTIVE_INVASION
        elif ratio <= 1.2:
            return cls.MINIMAL_INVASION
        else:
            return cls.RESISTIVE_INVASION


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
    def classify(cls, porosity: Union[float, np.ndarray]) -> Union["PorosityClassification", np.ndarray]:
        """
        Classify porosity value(s).

        Parameters
        ----------
        porosity : float or np.ndarray
            Porosity fraction (0-1) or percentage (0-100)

        Returns
        -------
        PorosityClassification or np.ndarray
            Classification result(s)
        """
        # Convert to fraction if percentage
        if isinstance(porosity, np.ndarray):
            phi = porosity.copy()
            phi = np.where(phi > 1, phi / 100, phi)
            result = np.empty(phi.shape, dtype=object)
            result[phi > 0.20] = cls.EXCELLENT
            result[(phi > 0.12) & (phi <= 0.20)] = cls.GOOD
            result[(phi > 0.06) & (phi <= 0.12)] = cls.FAIR
            result[phi <= 0.06] = cls.POOR
            return result

        if np.isnan(porosity):
            return None  # match the array path, where NaN yields None
        phi = porosity / 100 if porosity > 1 else porosity
        if phi > 0.20:
            return cls.EXCELLENT
        elif phi > 0.12:
            return cls.GOOD
        elif phi > 0.06:
            return cls.FAIR
        else:
            return cls.POOR


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
    def classify(cls, sw: Union[float, np.ndarray]) -> Union["HcSaturationClassification", np.ndarray]:
        """
        Classify water saturation value(s).

        Parameters
        ----------
        sw : float or np.ndarray
            Water saturation fraction (0-1) or percentage (0-100)

        Returns
        -------
        HcSaturationClassification or np.ndarray
            Classification result(s)
        """
        if isinstance(sw, np.ndarray):
            sat = sw.copy()
            sat = np.where(sat > 1, sat / 100, sat)
            result = np.empty(sat.shape, dtype=object)
            result[sat < 0.20] = cls.EXCELLENT_HC
            result[(sat >= 0.20) & (sat < 0.40)] = cls.GOOD_HC
            result[(sat >= 0.40) & (sat < 0.60)] = cls.FAIR_HC
            result[sat >= 0.60] = cls.WATER_BEARING
            return result

        if np.isnan(sw):
            return None  # match the array path, where NaN yields None
        sat = sw / 100 if sw > 1 else sw
        if sat < 0.20:
            return cls.EXCELLENT_HC
        elif sat < 0.40:
            return cls.GOOD_HC
        elif sat < 0.60:
            return cls.FAIR_HC
        else:
            return cls.WATER_BEARING


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
    def classify(cls, perm: Union[float, np.ndarray]) -> Union["PermeabilityClassification", np.ndarray]:
        """
        Classify permeability value(s) in mD.

        Parameters
        ----------
        perm : float or np.ndarray
            Permeability in millidarcies (mD)

        Returns
        -------
        PermeabilityClassification or np.ndarray
            Classification result(s)
        """
        if isinstance(perm, np.ndarray):
            result = np.empty(perm.shape, dtype=object)
            result[perm > 1000] = cls.EXCELLENT
            result[(perm > 100) & (perm <= 1000)] = cls.VERY_GOOD
            result[(perm > 10) & (perm <= 100)] = cls.GOOD
            result[(perm > 1) & (perm <= 10)] = cls.FAIR
            result[(perm > 0.1) & (perm <= 1)] = cls.POOR
            result[perm <= 0.1] = cls.VERY_POOR
            return result

        if np.isnan(perm):
            return None  # match the array path, where NaN yields None
        if perm > 1000:
            return cls.EXCELLENT
        elif perm > 100:
            return cls.VERY_GOOD
        elif perm > 10:
            return cls.GOOD
        elif perm > 1:
            return cls.FAIR
        elif perm > 0.1:
            return cls.POOR
        else:
            return cls.VERY_POOR


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
    def classify(cls, mhi: Union[float, np.ndarray]) -> Union["MhiClassification", np.ndarray]:
        """
        Classify MHI value(s).

        Parameters
        ----------
        mhi : float or np.ndarray
            Moveable hydrocarbon index (0-1)

        Returns
        -------
        MhiClassification or np.ndarray
            Classification result(s)
        """
        if isinstance(mhi, np.ndarray):
            result = np.empty(mhi.shape, dtype=object)
            result[mhi < 0.5] = cls.HIGH_MOVEABLE_HC
            result[(mhi >= 0.5) & (mhi < 0.7)] = cls.MODERATE_MOVEABLE_HC
            result[(mhi >= 0.7) & (mhi < 0.9)] = cls.LOW_MOVEABLE_HC
            result[mhi >= 0.9] = cls.MINIMAL_MOVEABLE_HC
            return result

        if np.isnan(mhi):
            return None  # match the array path, where NaN yields None
        if mhi < 0.5:
            return cls.HIGH_MOVEABLE_HC
        elif mhi < 0.7:
            return cls.MODERATE_MOVEABLE_HC
        elif mhi < 0.9:
            return cls.LOW_MOVEABLE_HC
        else:
            return cls.MINIMAL_MOVEABLE_HC
