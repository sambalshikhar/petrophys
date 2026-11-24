"""
Core sample data container and loader.

Loads core laboratory measurements (porosity, permeability, grain density,
saturation, etc.) from CSV/Excel/DataFrame and provides depth-aligned access
for log-core calibration workflows.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.advice import Advice
from petrophys._internal.statistics import finite_min_max

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False



def _slice_per_plug(value: Any, mask: np.ndarray) -> Any:
    """Slice a per-plug metadata sequence; keep strings, dicts and scalars whole."""
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__len__"):
        return value
    if len(value) != len(mask):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(item for item, keep in zip(value, mask) if keep)
    try:
        return value[mask]
    except (TypeError, IndexError, KeyError):
        return value

@dataclass
class CoreData:
    """
    Container for core laboratory measurements.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth of each core plug (same unit as well logs).
    porosity : np.ndarray, optional
        Helium porosity (fraction, 0-1).
    permeability : np.ndarray, optional
        Air/Klinkenberg permeability (mD).
    grain_density : np.ndarray, optional
        Grain density from pycnometer (g/cm3).
    sw : np.ndarray, optional
        Water saturation from Dean-Stark (fraction, 0-1).
    so : np.ndarray, optional
        Oil saturation from Dean-Stark (fraction, 0-1).
    lithology : np.ndarray, optional
        Lithology description per plug (string array).
    metadata : dict
        Additional columns or information.

    Examples
    --------
    >>> core = CoreData.from_csv("core_data.csv", depth_col="DEPTH",
    ...     porosity_col="CPOR", perm_col="CPERM", gd_col="CGD")
    >>> print(f"{core.n_plugs} plugs, depth {core.top:.1f}-{core.bottom:.1f}")
    """

    depth: np.ndarray
    porosity: Optional[np.ndarray] = None
    permeability: Optional[np.ndarray] = None
    grain_density: Optional[np.ndarray] = None
    sw: Optional[np.ndarray] = None
    so: Optional[np.ndarray] = None
    lithology: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Properties ----

    @property
    def n_plugs(self) -> int:
        """Number of core plugs."""
        return len(self.depth)

    @property
    def top(self) -> Optional[float]:
        """Shallowest core depth."""
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        """Deepest core depth."""
        return finite_min_max(self.depth)[1]

    # ---- Loaders ----

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "DEPTH",
        porosity_col: Optional[str] = "CPOR",
        perm_col: Optional[str] = "CPERM",
        gd_col: Optional[str] = "CGD",
        sw_col: Optional[str] = None,
        so_col: Optional[str] = None,
        lithology_col: Optional[str] = None,
        porosity_unit: Optional[str] = None,
        saturation_unit: Optional[str] = None,
    ) -> "CoreData":
        """
        Create CoreData from a pandas DataFrame.

        Porosity and saturations are stored as fractions (v/v). Without a
        stated unit, a column whose values are mostly above 1 with a median of
        at least 2 is read as percent and divided by 100, with a warning;
        values outside 0-1 after that are masked with a warning
        (:func:`~petrophys.model.units.core_saturation_to_fraction`).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with core measurements.
        depth_col : str
            Column name for depth.
        porosity_col : str, optional
            Column name for porosity (fraction).
        perm_col : str, optional
            Column name for permeability (mD).
        gd_col : str, optional
            Column name for grain density (g/cm3).
        sw_col : str, optional
            Column name for water saturation.
        so_col : str, optional
            Column name for oil saturation.
        lithology_col : str, optional
            Column name for lithology.
        porosity_unit, saturation_unit : str, optional
            ``"%"`` or ``"v/v"`` (also ``"pu"``, ``"frac"`` ...) to state the
            scale of the porosity or saturation columns instead of inferring it.

        Returns
        -------
        CoreData
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required to load core data")

        depth = df[depth_col].values.astype(np.float64)

        def _get_col(col_name):
            if col_name and col_name in df.columns:
                return df[col_name].values
            return None

        from petrophys.model.units import core_saturation_to_fraction

        messages = []

        def _fraction(values, col_name, unit, quantity):
            if values is None:
                return None
            fractions, record = core_saturation_to_fraction(
                values.astype(np.float64), mnemonic=str(col_name), source_unit=unit, quantity=quantity,
            )
            messages.extend(record["warnings"])
            return fractions

        porosity = _fraction(_get_col(porosity_col), porosity_col, porosity_unit, "porosity")

        perm = _get_col(perm_col)
        if perm is not None:
            perm = perm.astype(np.float64)

        gd = _get_col(gd_col)
        if gd is not None:
            gd = gd.astype(np.float64)

        sw = _fraction(_get_col(sw_col), sw_col, saturation_unit, "saturation")
        so = _fraction(_get_col(so_col), so_col, saturation_unit, "saturation")
        if messages:
            import warnings

            for message in messages:
                # No unit label is involved here; drop the label clause of the shared message.
                message = message.replace(" despite unit ''", "")
                warnings.warn(Advice(f"CoreData: {message} (porosity_unit=/saturation_unit= state it)",
                                     ("porosity_unit", "saturation_unit")), UserWarning, stacklevel=2)

        lithology = _get_col(lithology_col)
        if porosity is None and perm is None and gd is None and sw is None and so is None:
            import warnings

            warnings.warn(
                f"CoreData: none of the property columns {[c for c in (porosity_col, perm_col, gd_col, sw_col, so_col) if c]} "
                f"is in the table ({list(df.columns)}); name them, or use petrophys.io.read_core, "
                "which finds core columns by name and converts units",
                UserWarning, stacklevel=3,
            )

        # Store remaining columns as metadata
        known_cols = {depth_col, porosity_col, perm_col, gd_col,
                      sw_col, so_col, lithology_col}
        extra = {c: df[c].values for c in df.columns if c not in known_cols and c is not None}

        return cls(
            depth=depth,
            porosity=porosity,
            permeability=perm,
            grain_density=gd,
            sw=sw,
            so=so,
            lithology=lithology,
            metadata=extra,
        )

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        **kwargs,
    ) -> "CoreData":
        """
        Load CoreData from a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to CSV file.
        **kwargs
            Column name mappings passed to ``from_dataframe``.

        Returns
        -------
        CoreData
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required to load core data")
        df = pd.read_csv(filepath)
        return cls.from_dataframe(df, **kwargs)

    @classmethod
    def from_excel(
        cls,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        **kwargs,
    ) -> "CoreData":
        """
        Load CoreData from an Excel file.

        Parameters
        ----------
        filepath : str or Path
            Path to Excel file.
        sheet_name : str or int
            Sheet name or index.
        **kwargs
            Column name mappings passed to ``from_dataframe``.

        Returns
        -------
        CoreData
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required to load core data")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        return cls.from_dataframe(df, **kwargs)

    def to_dataframe(self) -> "pd.DataFrame":
        """Export core data to a pandas DataFrame.

        Per-plug ``metadata`` columns (arrays with one value per plug, such as
        the extra columns kept by :meth:`from_dataframe`) are exported after
        the named fields; other metadata is not.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        data = {"DEPTH": self.depth}
        if self.porosity is not None:
            data["CPOR"] = self.porosity
        if self.permeability is not None:
            data["CPERM"] = self.permeability
        if self.grain_density is not None:
            data["CGD"] = self.grain_density
        if self.sw is not None:
            data["CSW"] = self.sw
        if self.so is not None:
            data["CSO"] = self.so
        if self.lithology is not None:
            data["CLITH"] = self.lithology
        for key, value in (self.metadata or {}).items():
            if (
                key not in data
                and isinstance(value, (np.ndarray, list, tuple, pd.Series))
                and np.ndim(value) == 1
                and len(value) == len(self.depth)
            ):
                data[key] = value
        return pd.DataFrame(data)

    def filter_by_depth(
        self, top: float, bottom: float
    ) -> "CoreData":
        """Return a new CoreData filtered to a depth range."""
        mask = (self.depth >= top) & (self.depth <= bottom)
        return CoreData(
            depth=self.depth[mask],
            porosity=self.porosity[mask] if self.porosity is not None else None,
            permeability=self.permeability[mask] if self.permeability is not None else None,
            grain_density=self.grain_density[mask] if self.grain_density is not None else None,
            sw=self.sw[mask] if self.sw is not None else None,
            so=self.so[mask] if self.so is not None else None,
            lithology=self.lithology[mask] if self.lithology is not None else None,
            metadata={k: _slice_per_plug(v, mask) for k, v in self.metadata.items()},
        )

    def __repr__(self) -> str:
        props = []
        if self.porosity is not None:
            props.append("porosity")
        if self.permeability is not None:
            props.append("perm")
        if self.grain_density is not None:
            props.append("GD")
        if self.sw is not None:
            props.append("Sw")
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        return (f"CoreData(plugs={self.n_plugs}, "
                f"depth={depth_text}, "
                f"props=[{', '.join(props)}])")
