"""
Main washout detection module.

This module provides the WashoutDetection class for comprehensive
bad hole analysis and quality control.
"""

from typing import TYPE_CHECKING, Optional, Dict, List, Any
from dataclasses import dataclass
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys.preprocessing.quality.bad_hole.flag_generator import bad_hole_flags

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class WashoutResult:
    """
    Result container for washout detection.

    Attributes
    ----------
    washout_flag : np.ndarray
        Boolean array indicating washout zones
    large_hole_flag : np.ndarray
        Boolean array indicating large hole zones
    low_density_flag : np.ndarray
        Boolean array indicating low density zones
    pef_exclusion_flag : np.ndarray
        Boolean array indicating PEF exclusion zones
    bad_hole_flag : np.ndarray
        Combined bad hole flag
    washout_intervals : list
        List of (top, bottom) washout intervals
    statistics : dict
        Summary statistics
    """

    washout_flag: np.ndarray
    large_hole_flag: np.ndarray
    low_density_flag: np.ndarray
    pef_exclusion_flag: np.ndarray
    bad_hole_flag: np.ndarray
    washout_intervals: List[tuple]
    statistics: Dict[str, Any]


class WashoutDetection:
    """
    Comprehensive washout and bad hole detection.

    This class analyzes well log data to identify zones with
    poor borehole conditions that may affect log quality.

    Parameters
    ----------
    cali_threshold : float, optional
        CALI - BS threshold for washout (default: 1.5 inches)
    drho_threshold : float, optional
        abs(DRHO) threshold for washout (default: 0.15 g/cm3)
    rhob_min : float, optional
        Minimum valid RHOB (default: 2.1 g/cm³)
    pef_max : float, optional
        Maximum valid PEF (default: 7)
    pef_min : float, optional
        Minimum valid PEF (default: 0.5)
    large_hole_threshold : float, optional
        Bit size above which the hole is flagged as large (default: 17.5
        inches)

    Examples
    --------
    >>> detector = WashoutDetection()
    >>> result = detector.detect(well)
    >>> print(f"Washout: {result.statistics['washout_percent']:.1f}%")
    """

    def __init__(
        self,
        cali_threshold: float = DefaultParameters.CALI_BS_THRESHOLD,
        drho_threshold: float = DefaultParameters.DRHO_THRESHOLD,
        rhob_min: float = DefaultParameters.RHOB_MIN,
        pef_max: float = DefaultParameters.PEF_MAX,
        pef_min: float = DefaultParameters.PEF_MIN,
        large_hole_threshold: float = DefaultParameters.LARGE_HOLE_BS,
    ):
        self.cali_threshold = cali_threshold
        self.drho_threshold = drho_threshold
        self.rhob_min = rhob_min
        self.pef_max = pef_max
        self.pef_min = pef_min
        self.large_hole_threshold = large_hole_threshold

    def detect(self, well: "Well", bit_size: Optional[float] = None) -> WashoutResult:
        """
        Detect washout and bad hole conditions.

        Parameters
        ----------
        well : Well
            Well object to analyze
        bit_size : float, optional
            Bit size override

        Returns
        -------
        WashoutResult
            Detection results
        """
        # Generate flags
        flags = bad_hole_flags(
            well,
            bit_size=bit_size,
            cali_threshold=self.cali_threshold,
            drho_threshold=self.drho_threshold,
            rhob_min=self.rhob_min,
            pef_max=self.pef_max,
            pef_min=self.pef_min,
            large_hole_threshold=self.large_hole_threshold,
        )

        # Find washout intervals
        intervals = self._find_intervals(
            flags["washout"],
            well.depth if well.depth is not None else np.arange(well.n_samples),
        )

        # Calculate statistics
        statistics = self._calculate_statistics(flags, well)

        return WashoutResult(
            washout_flag=flags["washout"],
            large_hole_flag=flags["large_hole"],
            low_density_flag=flags["low_density"],
            pef_exclusion_flag=flags["pef_exclusion"],
            bad_hole_flag=flags["bad_hole"],
            washout_intervals=intervals,
            statistics=statistics,
        )

    def _find_intervals(self, flag: np.ndarray, depth: np.ndarray) -> List[tuple]:
        """Find continuous intervals where flag is True."""
        intervals = []

        if not np.any(flag):
            return intervals

        # Find transitions
        padded = np.concatenate([[False], flag, [False]])
        transitions = np.diff(padded.astype(int))

        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        for start, end in zip(starts, ends):
            top = depth[start] if start < len(depth) else depth[-1]
            bottom = depth[min(end - 1, len(depth) - 1)]
            intervals.append((float(top), float(bottom)))

        return intervals

    def _calculate_statistics(
        self, flags: Dict[str, np.ndarray], well: "Well"
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        n_samples = well.n_samples
        depth_step = abs(well.step or 1.0)
        # Each sample stands for half the interval to each neighbour, so the
        # thickness is right for irregular or merged sampling as well.
        sample_thickness = np.full(n_samples, depth_step, dtype=float)
        if well.depth is not None and n_samples > 1:
            depth = np.asarray(well.depth, dtype=float)
            if np.all(np.isfinite(depth)):
                sample_thickness = np.abs(np.gradient(depth))

        stats = {}

        for name, flag in flags.items():
            count = np.sum(flag)
            stats[f"{name}_count"] = int(count)
            stats[f"{name}_percent"] = float(count / n_samples * 100) if n_samples else 0.0
            stats[f"{name}_thickness"] = float(np.sum(sample_thickness[np.asarray(flag, dtype=bool)]))

        return stats

    def add_flags_to_well(
        self,
        well: "Well",
        result: Optional[WashoutResult] = None,
        bit_size: Optional[float] = None,
    ) -> "Well":
        """
        Add bad hole flags as curves to well.

        Parameters
        ----------
        well : Well
            Well object
        result : WashoutResult, optional
            Pre-computed result (will compute if not provided)
        bit_size : float, optional
            Bit size override

        Returns
        -------
        Well
            Well with flag curves added
        """
        if result is None:
            result = self.detect(well, bit_size)

        # Add flag curves
        well.add_curve(
            "WASHOUT_FLAG",
            result.washout_flag.astype(float),
            unit="",
            description="Washout Flag (1=bad)",
        )
        well.add_curve(
            "BAD_HOLE_FLAG",
            result.bad_hole_flag.astype(float),
            unit="",
            description="Bad Hole Flag (1=bad)",
        )
        well.add_curve(
            "PEF_EXCL_FLAG",
            result.pef_exclusion_flag.astype(float),
            unit="",
            description="PEF Exclusion Flag (1=exclude)",
        )

        # Store statistics
        well.set_result("washout_detection", result.statistics)

        return well


def detect_washout(
    well: "Well", bit_size: Optional[float] = None, **kwargs
) -> WashoutResult:
    """
    Convenience function for washout detection.

    Parameters
    ----------
    well : Well
        Well object to analyze
    bit_size : float, optional
        Bit size override
    **kwargs
        Additional parameters for WashoutDetection

    Returns
    -------
    WashoutResult
        Detection results
    """
    detector = WashoutDetection(**kwargs)
    return detector.detect(well, bit_size)
