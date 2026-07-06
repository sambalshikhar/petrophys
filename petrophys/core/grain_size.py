"""
Grain size analysis from core laser scanning or image analysis.

Loads grain size distribution data (P13, P50, P87 percentiles) measured
on continuous core and provides Wentworth classification, sorting, and
permeability/pore-throat estimates.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from petrophys.utils.statistics import finite_min_max

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


# Wentworth (1922) grain size scale, boundaries in µm, coarse to fine.
# Silt spans 3.9-62.5 µm in four subclasses (4-8 phi); everything below
# 3.9 µm (1/256 mm) is clay; boulder starts at 256 mm.
_WENTWORTH = [
    (256_000,   "Boulder"),
    (64_000,    "Cobble"),
    (4_000,     "Pebble"),
    (2_000,     "Granule"),
    (1_000,     "Very Coarse Sand"),
    (500,       "Coarse Sand"),
    (250,       "Medium Sand"),
    (125,       "Fine Sand"),
    (62.5,      "Very Fine Sand"),
    (31.25,     "Coarse Silt"),
    (15.6,      "Medium Silt"),
    (7.8,       "Fine Silt"),
    (3.9,       "Very Fine Silt"),
    (0.0,       "Clay"),
]


def wentworth_class(d_um: float) -> str:
    """Return the Wentworth class name for a grain diameter in µm."""
    for boundary, name in _WENTWORTH:
        if d_um >= boundary:
            return name
    return "Clay"


def um_to_phi(d_um: float) -> float:
    """Convert grain diameter in µm to phi units (φ = -log₂(d_mm))."""
    d_mm = d_um / 1000.0
    if d_mm <= 0:
        return np.nan
    return -np.log2(d_mm)


@dataclass
class GrainSizeData:
    """
    Grain size distribution measurements along a core depth.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth (m or ft).
    d50 : np.ndarray
        Median grain diameter P50 (µm).
    d13 : np.ndarray, optional
        13th percentile grain diameter (µm).
    d87 : np.ndarray, optional
        87th percentile grain diameter (µm).
    metadata : dict
        Additional header information.

    Examples
    --------
    >>> gs = GrainSizeData.from_las("grain_size.las", d50_mnemonic="GS_P50")
    >>> cls = gs.wentworth()           # per-sample class names
    >>> k = gs.kozeny_carman_perm(porosity=0.25)  # mD estimate
    """

    depth: np.ndarray
    d50: np.ndarray
    d13: Optional[np.ndarray] = None
    d87: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Properties ----

    @property
    def n_samples(self) -> int:
        return len(self.depth)

    @property
    def top(self) -> Optional[float]:
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        return finite_min_max(self.depth)[1]

    # ---- Loaders ----

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        d50_mnemonic: str = "GS_P50",
        d13_mnemonic: Optional[str] = "GS_P13",
        d87_mnemonic: Optional[str] = "GS_P87",
        null_value: float = -999.25,
    ) -> "GrainSizeData":
        """
        Load grain size data from a LAS file.

        Depth is read via ``lasio``'s ``las.index`` property (always the
        first column, regardless of mnemonic name).

        Parameters
        ----------
        filepath : str or Path
        d50_mnemonic : str
            Mnemonic for the P50 (median) grain size curve.
        d13_mnemonic : str, optional
            Mnemonic for the P13 grain size curve.
        d87_mnemonic : str, optional
            Mnemonic for the P87 grain size curve.
        null_value : float
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))

        def _get(mn):
            if mn is None or mn not in las.keys():
                return None
            arr = np.array(las[mn], dtype=float)
            arr[arr == null_value] = np.nan
            return arr

        depth = np.array(las.index, dtype=float)
        d50 = _get(d50_mnemonic)
        if d50 is None:
            raise ValueError(f"D50 mnemonic '{d50_mnemonic}' not found in {filepath}")

        meta = {
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
        }

        return cls(
            depth=depth,
            d50=d50,
            d13=_get(d13_mnemonic),
            d87=_get(d87_mnemonic),
            metadata=meta,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "DEPTH",
        d50_col: str = "D50",
        d13_col: Optional[str] = None,
        d87_col: Optional[str] = None,
    ) -> "GrainSizeData":
        """Create GrainSizeData from a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        def _col(name):
            if name and name in df.columns:
                return df[name].values.astype(float)
            return None

        depth = df[depth_col].values.astype(float)
        d50 = _col(d50_col)
        if d50 is None:
            raise ValueError(f"D50 column '{d50_col}' not found")

        return cls(depth=depth, d50=d50, d13=_col(d13_col), d87=_col(d87_col))

    # ---- Classification ----

    def wentworth(self) -> np.ndarray:
        """
        Return the Wentworth class name for each depth sample.

        Returns
        -------
        np.ndarray of str
        """
        return np.array([
            wentworth_class(d) if not np.isnan(d) else "Unknown"
            for d in self.d50
        ])

    def phi_scale(self) -> np.ndarray:
        """Convert d50 to phi units (φ = -log₂(d_mm))."""
        return np.array([um_to_phi(d) for d in self.d50])

    # ---- Distribution statistics ----

    def sorting_coefficient(self) -> np.ndarray:
        """
        Folk & Ward (1957) inclusive graphic standard deviation (sorting).

        Requires d13 and d87. Returns NaN where either is missing.

        Returns
        -------
        np.ndarray
            Sorting in phi units:
            < 0.35  well sorted
            0.35–0.5  moderately well sorted
            0.5–0.71  moderately sorted
            > 0.71  poorly sorted
        """
        if self.d13 is None or self.d87 is None:
            return np.full(self.n_samples, np.nan)
        phi13 = np.array([um_to_phi(d) for d in self.d13])
        phi87 = np.array([um_to_phi(d) for d in self.d87])
        # σ = (φ13 - φ87) / 2  (simplified graphic std dev). d13 < d87 in
        # diameter, so φ13 > φ87 and the spread is positive.
        return (phi13 - phi87) / 2.0

    def skewness(self) -> np.ndarray:
        """
        Grain size skewness from percentiles.

        Requires d13, d50, and d87.
        Positive = fine-skewed (more fines), negative = coarse-skewed.
        """
        if self.d13 is None or self.d87 is None:
            return np.full(self.n_samples, np.nan)
        phi13 = np.array([um_to_phi(d) for d in self.d13])
        phi50 = self.phi_scale()
        phi87 = np.array([um_to_phi(d) for d in self.d87])
        # d13 < d87 in diameter, so the positive phi spread is φ13 - φ87.
        # (φ87 + φ13 - 2φ50) > 0 when the fine tail (high phi) dominates.
        spread = phi13 - phi87
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                spread > 0,
                (phi87 + phi13 - 2 * phi50) / spread,
                np.nan,
            )

    # ---- Permeability & pore throat estimates ----

    def kozeny_carman_perm(
        self,
        porosity: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Estimate permeability using the Kozeny-Carman equation.

        .. math::
            k = \\frac{d_{50}^2 \\cdot \\phi^3}{180 \\cdot (1 - \\phi)^2}

        where k is in mD and d₅₀ is in µm.

        Parameters
        ----------
        porosity : float or np.ndarray
            Porosity as a fraction (0–1). Can be a scalar or a depth array
            matching ``self.depth``.

        Returns
        -------
        np.ndarray
            Estimated permeability (mD). Theoretical upper bound —
            real permeability is lower due to cementation and packing.

        Notes
        -----
        Conversion factor: 1 µm² ≈ 1013 mD, yielding the coefficient 5.63.
        """
        phi = np.asarray(porosity, dtype=float)
        d = self.d50  # µm

        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(
                (phi > 0) & (phi < 1) & (d > 0),
                5.63 * d ** 2 * phi ** 3 / (1 - phi) ** 2,
                np.nan,
            )
        return k

    def pore_throat_radius(self) -> np.ndarray:
        """
        Estimate modal pore throat radius from median grain size.

        Uses the empirical approximation r_pt ≈ d₅₀ / 10, which holds
        for well-sorted, uncemented sands (Beard & Weyl 1973).
        For cemented or poorly sorted material the actual r_pt will be
        significantly smaller.

        Returns
        -------
        np.ndarray
            Estimated pore throat radius (µm).
        """
        return self.d50 / 10.0

    # ---- Utilities ----

    def filter_by_depth(self, top: float, bottom: float) -> "GrainSizeData":
        """Return a new GrainSizeData filtered to [top, bottom]."""
        mask = (self.depth >= top) & (self.depth <= bottom)
        return GrainSizeData(
            depth=self.depth[mask],
            d50=self.d50[mask],
            d13=self.d13[mask] if self.d13 is not None else None,
            d87=self.d87[mask] if self.d87 is not None else None,
            metadata=dict(self.metadata),
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Export to a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        data: Dict[str, np.ndarray] = {"DEPTH": self.depth, "D50_UM": self.d50}
        if self.d13 is not None:
            data["D13_UM"] = self.d13
        if self.d87 is not None:
            data["D87_UM"] = self.d87
        data["PHI_SCALE"] = self.phi_scale()
        data["WENTWORTH"] = self.wentworth()
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        has = ["d50"]
        if self.d13 is not None:
            has.append("d13")
        if self.d87 is not None:
            has.append("d87")
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        d50_min, d50_max = finite_min_max(self.d50)
        d50_text = (
            f"{d50_min:.1f}-{d50_max:.1f}"
            if d50_min is not None and d50_max is not None
            else "None-None"
        )
        return (f"GrainSizeData(samples={self.n_samples}, "
                f"depth={depth_text}, "
                f"d50_range={d50_text} µm, "
                f"data=[{', '.join(has)}])")
