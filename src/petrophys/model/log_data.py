"""
Depth intervals, zone sets and depth-index helpers for log data.

This module provides supporting data structures for handling
well log data including interval management and zone definitions.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from petrophys._internal.statistics import finite_min_max


@dataclass
class DepthInterval:
    """
    Represents a depth interval.

    Attributes
    ----------
    top : float
        Top depth of interval
    bottom : float
        Bottom depth of interval
    name : str
        Name or identifier for the interval
    """

    top: float
    bottom: float
    name: str = ""

    def __post_init__(self):
        if self.top > self.bottom:
            raise ValueError(
                f"Top depth ({self.top}) must be <= bottom depth ({self.bottom})"
            )

    @property
    def thickness(self) -> float:
        """Return interval thickness."""
        return self.bottom - self.top

    def contains(self, depth: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if depth(s) are within this interval."""
        return (depth >= self.top) & (depth <= self.bottom)

    def overlaps(self, other: "DepthInterval") -> bool:
        """Check if this interval overlaps with another."""
        return not (self.bottom < other.top or self.top > other.bottom)

    def intersection(self, other: "DepthInterval") -> Optional["DepthInterval"]:
        """Return intersection with another interval."""
        if not self.overlaps(other):
            return None
        return DepthInterval(
            top=max(self.top, other.top),
            bottom=min(self.bottom, other.bottom),
            name=f"{self.name}_{other.name}"
        )

    def __repr__(self) -> str:
        return f"DepthInterval({self.top}-{self.bottom}, name={self.name})"


@dataclass
class Zone:
    """
    Represents a geological or petrophysical zone.

    Attributes
    ----------
    name : str
        Zone name
    top : float
        Top depth
    bottom : float
        Bottom depth
    formation : str
        Formation name
    lithology : str
        Primary lithology
    fluid : str
        Expected fluid content
    properties : dict
        Additional zone properties
    """

    name: str
    top: float
    bottom: float
    formation: str = ""
    lithology: str = ""
    fluid: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def thickness(self) -> float:
        """Return zone thickness."""
        return self.bottom - self.top

    @property
    def interval(self) -> DepthInterval:
        """Return as DepthInterval."""
        return DepthInterval(self.top, self.bottom, self.name)

    def contains(self, depth: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if depth(s) are within this zone."""
        return (depth >= self.top) & (depth <= self.bottom)


class ZoneSet:
    """
    Collection of zones for a well.

    Manages a set of depth-based zones with utilities for
    querying and manipulating zones.
    """

    def __init__(self):
        self._zones: List[Zone] = []

    def add(self, zone: Zone):
        """Add a zone to the set."""
        self._zones.append(zone)
        self._zones.sort(key=lambda z: z.top)

    def add_zone(
        self,
        name: str,
        top: float,
        bottom: float,
        **kwargs
    ) -> Zone:
        """Create and add a new zone."""
        zone = Zone(name=name, top=top, bottom=bottom, **kwargs)
        self.add(zone)
        return zone

    def get(self, name: str) -> Optional[Zone]:
        """Get zone by name."""
        for zone in self._zones:
            if zone.name == name:
                return zone
        return None

    def get_at_depth(self, depth: float) -> Optional[Zone]:
        """Get zone at a specific depth."""
        for zone in self._zones:
            if zone.contains(depth):
                return zone
        return None

    def get_zones_in_range(self, top: float, bottom: float) -> List[Zone]:
        """Get all zones within a depth range."""
        result = []
        for zone in self._zones:
            if zone.bottom >= top and zone.top <= bottom:
                result.append(zone)
        return result

    @property
    def zones(self) -> List[Zone]:
        """Return list of zones."""
        return self._zones.copy()

    @property
    def names(self) -> List[str]:
        """Return list of zone names."""
        return [z.name for z in self._zones]

    def __len__(self) -> int:
        return len(self._zones)

    def __iter__(self):
        return iter(self._zones)

    def __getitem__(self, key: Union[int, str]) -> Zone:
        if isinstance(key, int):
            return self._zones[key]
        zone = self.get(key)
        if zone is None:
            raise KeyError(f"Zone not found: {key}")
        return zone

    def to_dataframe(self) -> pd.DataFrame:
        """Convert zones to DataFrame."""
        data = []
        for zone in self._zones:
            row = {
                "name": zone.name,
                "top": zone.top,
                "bottom": zone.bottom,
                "thickness": zone.thickness,
                "formation": zone.formation,
                "lithology": zone.lithology,
                "fluid": zone.fluid,
            }
            row.update(zone.properties)
            data.append(row)
        return pd.DataFrame(data)


@dataclass
class LogDataPoint:
    """
    Single data point with depth and value.

    Useful for representing specific measurements or picks.
    """

    depth: float
    value: float
    mnemonic: str = ""
    unit: str = ""
    quality: str = "good"
    comment: str = ""


class DepthIndex:
    """
    Utilities for working with depth-indexed data.
    """

    def __init__(
        self,
        depth: np.ndarray,
        unit: str = "ft"
    ):
        self._depth = np.asarray(depth, dtype=np.float64)
        self.unit = unit

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    @property
    def top(self) -> Optional[float]:
        return finite_min_max(self._depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        return finite_min_max(self._depth)[1]

    @property
    def step(self) -> float:
        """Return median step size."""
        finite = self._depth[np.isfinite(self._depth)]
        if len(finite) < 2:
            return 0.0
        steps = np.abs(np.diff(finite))
        steps = steps[np.isfinite(steps) & (steps > 0)]
        return float(np.median(steps)) if steps.size else 0.0

    @property
    def is_uniform(self) -> bool:
        """Check if depth sampling is uniform."""
        if len(self._depth) < 2:
            return True
        steps = np.diff(self._depth)
        return np.allclose(steps, steps[0], rtol=0.01)

    def get_mask(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> np.ndarray:
        """Get boolean mask for depth range."""
        mask = np.ones(len(self._depth), dtype=bool)
        if top is not None:
            mask &= self._depth >= top
        if bottom is not None:
            mask &= self._depth <= bottom
        return mask

    def get_indices(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> np.ndarray:
        """Get indices for depth range."""
        mask = self.get_mask(top, bottom)
        return np.where(mask)[0]

    def find_nearest(self, target_depth: float) -> Tuple[int, float]:
        """Find index of nearest depth and the actual depth value."""
        finite = np.isfinite(self._depth)
        if not np.any(finite):
            raise ValueError("Cannot find nearest depth in an empty/non-finite DepthIndex")
        candidates = np.where(finite)[0]
        local_idx = np.argmin(np.abs(self._depth[finite] - target_depth))
        idx = candidates[local_idx]
        return int(idx), float(self._depth[idx])

    def resample(
        self,
        new_step: float,
        method: str = "linear"
    ) -> "DepthIndex":
        """Resample to new step size."""
        if new_step <= 0:
            raise ValueError("new_step must be positive")
        top = self.top
        bottom = self.bottom
        if top is None or bottom is None:
            raise ValueError("Cannot resample an empty/non-finite DepthIndex")
        new_depth = np.arange(top, bottom + new_step / 2, new_step)
        new_depth = new_depth[new_depth <= bottom + 1e-9]
        return DepthIndex(new_depth, self.unit)

    def __len__(self) -> int:
        return len(self._depth)

    def __getitem__(self, index) -> float:
        return self._depth[index]


def create_depth_array(
    top: float,
    bottom: float,
    step: float
) -> np.ndarray:
    """
    Create uniform depth array.

    Parameters
    ----------
    top : float
        Starting depth
    bottom : float
        Ending depth
    step : float
        Depth step

    Returns
    -------
    np.ndarray
        Uniform depth array
    """
    return np.arange(top, bottom + step / 2, step)


def align_to_depth(
    data: np.ndarray,
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Resample data to align with target depth.

    Parameters
    ----------
    data : np.ndarray
        Data values to resample
    source_depth : np.ndarray
        Original depth values
    target_depth : np.ndarray
        Target depth values
    method : str
        Interpolation method ("linear", "nearest")

    Returns
    -------
    np.ndarray
        Resampled data aligned to target depth
    """
    from scipy import interpolate

    # Handle NaN values
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full(len(target_depth), np.nan)

    if method == "nearest":
        interp = interpolate.interp1d(
            source_depth[valid_mask],
            data[valid_mask],
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan
        )
    else:  # linear
        interp = interpolate.interp1d(
            source_depth[valid_mask],
            data[valid_mask],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )

    return interp(target_depth)
