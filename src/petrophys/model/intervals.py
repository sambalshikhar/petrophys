"""
Interval type hierarchy for formation-level analysis.

Provides a common base class and four concrete interval types:
Formation, Zone, SubZone, and ManualInterval.
"""

from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class IntervalKind(Enum):
    """Type of geological/petrophysical interval."""
    FORMATION = "formation"
    ZONE = "zone"
    SUB_ZONE = "sub_zone"
    MANUAL = "manual"


@dataclass
class Interval:
    """
    Base class for all depth intervals.

    Attributes
    ----------
    name : str
        Interval name/identifier
    top_depth : float
        Top depth of interval
    bottom_depth : float
        Bottom depth of interval
    kind : IntervalKind
        Type of interval
    lithology : str
        Primary lithology (e.g. "clastic", "carbonate")
    metadata : dict
        Additional user-defined properties
    """

    name: str
    top_depth: float
    bottom_depth: float
    kind: IntervalKind = IntervalKind.MANUAL
    lithology: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.top_depth > self.bottom_depth:
            raise ValueError(
                f"Top depth ({self.top_depth}) must be <= "
                f"bottom depth ({self.bottom_depth})"
            )

    @property
    def thickness(self) -> float:
        """Return interval thickness."""
        return self.bottom_depth - self.top_depth

    def contains(self, depth: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if depth(s) fall within this interval."""
        return (depth >= self.top_depth) & (depth <= self.bottom_depth)

    def overlaps(self, other: "Interval") -> bool:
        """Check if this interval overlaps with another."""
        return not (self.bottom_depth <= other.top_depth
                    or self.top_depth >= other.bottom_depth)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.name!r}, {self.top_depth}-{self.bottom_depth})")


@dataclass
class Formation(Interval):
    """
    A geological formation interval.

    Attributes
    ----------
    age : str
        Geological age of the formation
    """
    age: str = ""

    def __post_init__(self):
        self.kind = IntervalKind.FORMATION
        super().__post_init__()


@dataclass
class Zone(Interval):
    """
    A petrophysical zone within a formation.

    Attributes
    ----------
    parent_formation : str
        Name of the parent formation
    """
    parent_formation: str = ""

    def __post_init__(self):
        self.kind = IntervalKind.ZONE
        super().__post_init__()


@dataclass
class SubZone(Interval):
    """
    A sub-zone within a zone.

    Attributes
    ----------
    parent_zone : str
        Name of the parent zone
    """
    parent_zone: str = ""

    def __post_init__(self):
        self.kind = IntervalKind.SUB_ZONE
        super().__post_init__()


@dataclass
class ManualInterval(Interval):
    """A user-defined arbitrary depth interval."""

    def __post_init__(self):
        self.kind = IntervalKind.MANUAL
        super().__post_init__()
