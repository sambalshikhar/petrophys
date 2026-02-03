"""
Curve data structure for well log data.

This module provides the Curve class which represents a single
well log curve with associated metadata, units, and data quality flags.
"""

from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from petrophys.core.data_types import DataType


@dataclass
class CurveMetadata:
    """
    Metadata for a well log curve.

    Attributes
    ----------
    description : str
        Full description of the curve
    source : str
        Origin of the data (e.g., "LAS", "computed")
    service_company : str
        Company that acquired the log
    tool : str
        Logging tool used
    run_number : int
        Run number for the log
    data_type : DataType
        Classification of data type
    """

    description: str = ""
    source: str = ""
    service_company: str = ""
    tool: str = ""
    run_number: int = 1
    data_type: DataType = DataType.CURVE
    custom: Dict[str, Any] = field(default_factory=dict)


class Curve:
    """
    Well log curve with data, units, and metadata.

    A Curve object represents a single well log measurement with
    depth-indexed values, unit information, and associated metadata.

    Parameters
    ----------
    mnemonic : str
        Curve mnemonic (e.g., "GR", "RHOB", "NPHI")
    data : np.ndarray
        Curve data values
    unit : str
        Unit of measurement
    depth : np.ndarray, optional
        Depth values for the curve
    depth_unit : str, optional
        Unit for depth values (default: "ft")
    metadata : CurveMetadata, optional
        Additional metadata

    Examples
    --------
    >>> import numpy as np
    >>> depth = np.arange(1000, 2000, 0.5)
    >>> gr = np.random.uniform(20, 100, len(depth))
    >>> curve = Curve("GR", gr, "API", depth=depth)
    >>> print(curve)
    Curve(GR, unit=API, samples=2000)
    """

    def __init__(
        self,
        mnemonic: str,
        data: Union[np.ndarray, List[float], pd.Series],
        unit: str = "",
        depth: Optional[Union[np.ndarray, List[float]]] = None,
        depth_unit: str = "ft",
        metadata: Optional[CurveMetadata] = None
    ):
        self.mnemonic = mnemonic.upper()
        self._data = np.asarray(data, dtype=np.float64)
        self.unit = unit
        self._depth = np.asarray(depth, dtype=np.float64) if depth is not None else None
        self.depth_unit = depth_unit
        self.metadata = metadata or CurveMetadata()

        # Validate depth alignment
        if self._depth is not None and len(self._depth) != len(self._data):
            raise ValueError(
                f"Depth length ({len(self._depth)}) must match data length ({len(self._data)})"
            )

    @property
    def data(self) -> np.ndarray:
        """Return curve data as numpy array."""
        return self._data

    @data.setter
    def data(self, value: Union[np.ndarray, List[float]]):
        """Set curve data."""
        self._data = np.asarray(value, dtype=np.float64)

    @property
    def depth(self) -> Optional[np.ndarray]:
        """Return depth values."""
        return self._depth

    @depth.setter
    def depth(self, value: Optional[Union[np.ndarray, List[float]]]):
        """Set depth values."""
        if value is not None:
            self._depth = np.asarray(value, dtype=np.float64)
        else:
            self._depth = None

    @property
    def values(self) -> np.ndarray:
        """Alias for data property."""
        return self._data

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._data)

    def __repr__(self) -> str:
        return f"Curve({self.mnemonic}, unit={self.unit}, samples={len(self)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, index) -> Union[float, "Curve"]:
        """Index into curve data."""
        if isinstance(index, (int, np.integer)):
            return self._data[index]

        # Return new Curve for slices
        new_data = self._data[index]
        new_depth = self._depth[index] if self._depth is not None else None
        return Curve(
            mnemonic=self.mnemonic,
            data=new_data,
            unit=self.unit,
            depth=new_depth,
            depth_unit=self.depth_unit,
            metadata=self.metadata
        )

    def __array__(self) -> np.ndarray:
        """Return data for numpy operations."""
        return self._data

    # Arithmetic operations
    def __add__(self, other: Union["Curve", float, np.ndarray]) -> "Curve":
        if isinstance(other, Curve):
            other = other.data
        return self._create_derived(self._data + other)

    def __radd__(self, other: Union[float, np.ndarray]) -> "Curve":
        return self._create_derived(other + self._data)

    def __sub__(self, other: Union["Curve", float, np.ndarray]) -> "Curve":
        if isinstance(other, Curve):
            other = other.data
        return self._create_derived(self._data - other)

    def __rsub__(self, other: Union[float, np.ndarray]) -> "Curve":
        return self._create_derived(other - self._data)

    def __mul__(self, other: Union["Curve", float, np.ndarray]) -> "Curve":
        if isinstance(other, Curve):
            other = other.data
        return self._create_derived(self._data * other)

    def __rmul__(self, other: Union[float, np.ndarray]) -> "Curve":
        return self._create_derived(other * self._data)

    def __truediv__(self, other: Union["Curve", float, np.ndarray]) -> "Curve":
        if isinstance(other, Curve):
            other = other.data
        return self._create_derived(self._data / other)

    def __rtruediv__(self, other: Union[float, np.ndarray]) -> "Curve":
        return self._create_derived(other / self._data)

    def __pow__(self, power: float) -> "Curve":
        return self._create_derived(self._data ** power)

    def _create_derived(self, data: np.ndarray) -> "Curve":
        """Create a derived curve with same depth and metadata."""
        meta = CurveMetadata(
            description=f"Derived from {self.mnemonic}",
            source="computed",
            data_type=DataType.COMPUTED
        )
        return Curve(
            mnemonic=f"{self.mnemonic}_derived",
            data=data,
            unit=self.unit,
            depth=self._depth,
            depth_unit=self.depth_unit,
            metadata=meta
        )

    def copy(self) -> "Curve":
        """Create a deep copy of the curve."""
        return Curve(
            mnemonic=self.mnemonic,
            data=self._data.copy(),
            unit=self.unit,
            depth=self._depth.copy() if self._depth is not None else None,
            depth_unit=self.depth_unit,
            metadata=self.metadata
        )

    def rename(self, new_mnemonic: str) -> "Curve":
        """Return a copy with a new mnemonic."""
        curve = self.copy()
        curve.mnemonic = new_mnemonic.upper()
        return curve

    def to_series(self) -> pd.Series:
        """Convert to pandas Series with depth as index."""
        if self._depth is not None:
            return pd.Series(self._data, index=self._depth, name=self.mnemonic)
        return pd.Series(self._data, name=self.mnemonic)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with depth column."""
        if self._depth is not None:
            return pd.DataFrame({
                "DEPTH": self._depth,
                self.mnemonic: self._data
            })
        return pd.DataFrame({self.mnemonic: self._data})

    @classmethod
    def from_series(cls, series: pd.Series, unit: str = "") -> "Curve":
        """Create Curve from pandas Series."""
        depth = series.index.values if not series.index.equals(
            pd.RangeIndex(len(series))
        ) else None
        return cls(
            mnemonic=series.name or "UNKNOWN",
            data=series.values,
            unit=unit,
            depth=depth
        )

    # Statistical methods
    def min(self) -> float:
        """Return minimum value (excluding NaN)."""
        return float(np.nanmin(self._data))

    def max(self) -> float:
        """Return maximum value (excluding NaN)."""
        return float(np.nanmax(self._data))

    def mean(self) -> float:
        """Return mean value (excluding NaN)."""
        return float(np.nanmean(self._data))

    def std(self) -> float:
        """Return standard deviation (excluding NaN)."""
        return float(np.nanstd(self._data))

    def median(self) -> float:
        """Return median value (excluding NaN)."""
        return float(np.nanmedian(self._data))

    def percentile(self, q: float) -> float:
        """Return percentile value (excluding NaN)."""
        return float(np.nanpercentile(self._data, q))

    def valid_count(self) -> int:
        """Return count of non-NaN values."""
        return int(np.sum(~np.isnan(self._data)))

    def null_count(self) -> int:
        """Return count of NaN values."""
        return int(np.sum(np.isnan(self._data)))

    def describe(self) -> Dict[str, float]:
        """Return summary statistics."""
        return {
            "count": self.valid_count(),
            "null_count": self.null_count(),
            "min": self.min(),
            "max": self.max(),
            "mean": self.mean(),
            "std": self.std(),
            "median": self.median(),
            "p25": self.percentile(25),
            "p75": self.percentile(75),
        }

    def clip(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> "Curve":
        """Clip values to specified range."""
        clipped = np.clip(self._data, lower, upper)
        curve = self.copy()
        curve._data = clipped
        return curve

    def mask(self, condition: np.ndarray) -> "Curve":
        """Set values to NaN where condition is True."""
        curve = self.copy()
        curve._data = np.where(condition, np.nan, curve._data)
        return curve

    def fillna(self, value: float = 0.0) -> "Curve":
        """Fill NaN values with specified value."""
        curve = self.copy()
        curve._data = np.where(np.isnan(curve._data), value, curve._data)
        return curve

    def interpolate(self, method: str = "linear") -> "Curve":
        """Interpolate NaN values."""
        curve = self.copy()
        series = pd.Series(curve._data)
        curve._data = series.interpolate(method=method).values
        return curve

    def depth_slice(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> "Curve":
        """Extract curve data within depth range."""
        if self._depth is None:
            raise ValueError("Curve has no depth data")

        mask = np.ones(len(self._depth), dtype=bool)
        if top is not None:
            mask &= self._depth >= top
        if bottom is not None:
            mask &= self._depth <= bottom

        return self[mask]
